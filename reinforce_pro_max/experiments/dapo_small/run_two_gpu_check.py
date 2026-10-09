"""Run a prepared, isolated two-GPU 12-update stability check."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()


def state(phase, **kwargs):
    tmp = root / 'status.tmp'
    tmp.write_text(json.dumps(dict(phase=phase, unix_time=time.time(), **kwargs), indent=2) + '\n')
    tmp.replace(root / 'status.json')


try:
    used = subprocess.check_output(['nvidia-smi', '-i', '0,1', '--query-gpu=memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    assert max(map(int, used.splitlines())) < 3000, 'Selected GPUs are occupied.'
    assert not (root / 'train.log').exists(), 'Use a fresh output directory.'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1', PYTHONPATH=str(root / 'source'),
               WANDB_MODE='disabled', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               PROMAX_STUDY_METRICS=str(root / 'metrics'))
    env['PATH'] = str(Path(sys.executable).parent) + ':' + env.get('PATH', '')
    for key in ('WANDB_API_KEY', 'RAY_ADDRESS'):
        env.pop(key, None)
    start = time.time()
    state('running', gpus=[0, 1], planned_updates=12)
    peaks = {}
    with (root / 'train.log').open('w') as log, (root / 'gpu.jsonl').open('w') as gpu:
        proc = subprocess.Popen(json.loads((root / 'command.json').read_text()), cwd=root / 'source',
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        (root / 'train.pid').write_text(str(proc.pid))
        while proc.poll() is None:
            sample = subprocess.check_output(['nvidia-smi', '-i', '0,1',
                '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            for line in sample.splitlines():
                index, memory, _ = map(int, line.split(','))
                peaks[index] = max(peaks.get(index, 0), memory)
            gpu.write(json.dumps({'time': time.time(), 'gpus': sample}) + '\n')
            gpu.flush()
            time.sleep(2)
    (root / 'exit_code.txt').write_text(str(proc.returncode))
    if proc.returncode:
        raise RuntimeError(f'Training exited {proc.returncode}; inspect train.log')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import math
    records = {}
    for path in root.rglob('*tfevents*'):
        acc = EventAccumulator(str(path), size_guidance={'scalars': 0})
        acc.Reload()
        for tag in acc.Tags()['scalars']:
            events = acc.Scalars(tag)
            assert all(math.isfinite(e.value) for e in events), tag
            records[tag] = {e.step: {'value': e.value, 'time': e.wall_time} for e in events}
    events = records['train/reward']
    assert sorted(events) == list(range(1, 13)), sorted(events)
    result = {'updates': 12, 'exit_code': proc.returncode, 'sampled_peak_memory_mib': peaks,
              'wall_seconds': time.time() - start,
              'steady_seconds_per_update': (events[12]['time'] - events[2]['time']) / 10,
              'first_update_seconds': events[1]['time'] - start, 'scalars': records,
              'scope': 'One baseline 12-update trial, not four concurrent jobs or long-run stability.'}
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    state('complete')
except Exception as exc:
    state('failed', error=repr(exc))
    raise
