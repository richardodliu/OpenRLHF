"""Queue a paired 12-update timing run after the fixed study releases its lock.

Only enforce_eager differs between the two runs. A private source copy stops
after update 12 and skips final model export; the 100-update schedule is intact.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-paused-study', action='store_true')
    args = parser.parse_args()
    study, root = args.study.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = root / 'source'
    if not source.exists():
        shutil.copytree(study / 'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        p = source / 'openrlhf/trainer/ppo_trainer.py'
        text = p.read_text()
        anchor = '                pbar.update(prompts_consumed)'
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + '\n                if global_step >= 12:\n                    break')
        p.write_text(text)
        p = source / 'openrlhf/cli/train_ppo_ray.py'
        text = p.read_text()
        anchor = '    # save model\n'
        assert text.count(anchor) == 1
        p.write_text(text.replace(anchor, '    return  # Timing-only run: no model export.\n\n' + anchor))
    base = json.loads((study / 'baseline/train_command.json').read_text())['argv']
    commands = {}
    for mode in ('eager', 'cuda_graph'):
        folder = root / mode
        folder.mkdir(exist_ok=True)
        command = list(base)
        command[0] = sys.executable
        for flag, value in {'--save_path': folder / 'unused_model', '--ckpt_path': folder / 'unused_ckpt',
                            '--use_tensorboard': folder / 'tensorboard'}.items():
            command[command.index(flag) + 1] = str(value)
        if mode == 'cuda_graph':
            command.remove('--enforce_eager')
        commands[mode] = command
        write(folder / 'command.json', command)
    write(root / 'protocol.json', {'updates': 12, 'warmup_updates': 2,
          'timed_intervals': 'completion of update 2 to completion of update 12',
          'only_execution_difference': '--enforce_eager',
          'unchanged': '3200 prompts, 100-update LR schedule, seed, model, reward, batch sizes, lengths, loss',
          'study': str(study), 'commands': commands})
    write(root / 'source_sha256.json', {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
    if args.prepare_only:
        print('Prepared:', root)
        return
    def state(phase, **kw):
        write(root / 'status.json', {'phase': phase, 'unix_time': time.time(), **kw})
    state('waiting_for_study')
    study_lock = (study / 'study.lock').open('a')
    while True:
        status = json.loads((study / 'status.json').read_text())
        if status['phase'] == 'failed':
            state('blocked', reason='Main study failed; GPU test will not start automatically.')
            return
        if status['phase'] == 'complete' or (args.allow_paused_study and status['phase'] == 'paused'):
            try:
                fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
        time.sleep(30)
    results = {}
    try:
        for mode, command in commands.items():
            state('waiting_for_free_gpus', mode=mode)
            for attempt in range(60):
                mem = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                               '--format=csv,noheader,nounits'], text=True).splitlines()
                if len(mem) == 8 and max(map(int, mem)) < 3000:
                    break
                time.sleep(10)
            else:
                raise RuntimeError('GPUs occupied; no process was stopped.')
            folder = root / mode
            if (folder / 'exit_code.txt').exists():
                raise RuntimeError('Existing timing run; use a fresh output directory.')
            env = os.environ.copy()
            env.update(PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                       WANDB_MODE='disabled', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                       PROMAX_STUDY_METRICS=str(folder / 'metrics'))
            env['PATH'] = str(Path(sys.executable).parent) + ':' + env.get('PATH', '')
            for name in ('WANDB_API_KEY', 'RAY_ADDRESS'):
                env.pop(name, None)
            state('running', mode=mode)
            start = time.time()
            with (folder / 'train.log').open('w') as log, (folder / 'gpu.jsonl').open('w') as gpu:
                proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                (folder / 'pid').write_text(str(proc.pid))
                while proc.poll() is None:
                    sample = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                      '--format=csv,noheader,nounits'], text=True)
                    gpu.write(json.dumps({'time': time.time(), 'gpus': sample}) + '\n')
                    gpu.flush()
                    time.sleep(5)
            (folder / 'exit_code.txt').write_text(str(proc.returncode))
            if proc.returncode:
                raise RuntimeError(f'{mode} failed with exit code {proc.returncode}')
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events = {}
            for path in folder.rglob('*tfevents*'):
                acc = EventAccumulator(str(path), size_guidance={'scalars': 0})
                acc.Reload()
                for event in acc.Scalars('train/reward'):
                    events[event.step] = {'time': event.wall_time, 'reward': event.value}
            assert sorted(events) == list(range(1, 13)), sorted(events)
            text = (folder / 'train.log').read_text(errors='replace')
            evidence = [line for line in text.splitlines() if any(term in line.lower()
                        for term in ('graph captur', 'capturing cuda', 'cudagraph_mode', 'enforce_eager'))]
            results[mode] = {'wall_seconds': time.time() - start,
                             'first_update_seconds': events[1]['time'] - start,
                             'steady_seconds_per_update': (events[12]['time'] - events[2]['time']) / 10,
                             'events': events, 'graph_log_evidence': evidence}
            write(root / 'results.json', results)
        results['speedup'] = results['eager']['steady_seconds_per_update'] / results['cuda_graph']['steady_seconds_per_update']
        results['interpretation'] = 'Single paired timing trial; verify graph capture evidence before attributing speedup.'
        write(root / 'results.json', results)
        state('complete')
    except Exception as exc:
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
