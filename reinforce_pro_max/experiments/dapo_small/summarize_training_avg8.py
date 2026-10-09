"""Export online training avg@8; no generation or model loading."""
import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def handoff_at_completed_training(root):
    """Reload an ordered queue only at the old supervisor's safe summary boundary."""
    import fcntl
    import os
    import signal
    import subprocess
    import sys
    import time

    request = root / 'queue-reload.json'
    if not request.exists():
        return
    data = json.loads(request.read_text())
    parent = os.getppid()
    if data.get('status') != 'pending' or parent != data['runner_pid']:
        return
    completed = root / data['after_arm'] / 'train_exit_code.txt'
    if not completed.exists() or completed.read_text().strip() != '0':
        return
    command = Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
    assert str(root / 'run_small.py').encode() in command
    data['status'] = 'handoff_started'
    request.write_text(json.dumps(data, indent=2) + '\n')
    os.kill(parent, signal.SIGTERM)  # Training has exited; terminate only its supervisor.
    with (root / 'study.lock').open('w') as lock:
        for _ in range(100):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        else:
            raise RuntimeError('Queue handoff could not acquire supervisor lock')
        fcntl.flock(lock, fcntl.LOCK_UN)
    with (root / 'runner.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-u', str(root / 'run_small.py')],
                                 cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=log, start_new_session=True)
    data.update(status='launched', new_runner_pid=child.pid)
    request.write_text(json.dumps(data, indent=2) + '\n')


def summarize(root):
    plan = json.loads((root / 'plan.json').read_text())
    args = plan['original_argv']
    repeats = int(args[args.index('--n_samples_per_prompt') + 1])
    prompts = int(args[args.index('--rollout_batch_size') + 1])
    assert repeats == 8
    output = {'metric': 'online_training_avg_at_8', 'samples_per_prompt': repeats,
              'scope': 'Training rollout correctness under the policy at each step; not final-model dataset accuracy or held-out evaluation.',
              'arms': {}}
    for arm in plan['arms']:
        folder = root / arm
        log = folder / 'train.log'
        if not log.exists():
            continue
        # The saved command is authoritative: completed arms may use a different
        # microbatch size from subsequently restarted arms.
        command = json.loads((folder / 'train_command.json').read_text())['argv']
        micro = int(command[command.index('--micro_train_batch_size') + 1])
        world = (int(command[command.index('--actor_num_nodes') + 1]) *
                 int(command[command.index('--actor_num_gpus_per_node') + 1]))
        batch = int(command[command.index('--train_batch_size') + 1])
        assert batch == prompts * repeats and batch % (micro * world) == 0
        assert int(command[command.index('--max_epochs') + 1]) == 1
        accumulation = batch // (micro * world)
        records = defaultdict(list)
        for path in (folder / 'metrics').glob('policy.*.jsonl'):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[(row['loss_call'] - 1) // accumulation + 1].append(row)
        curve = []
        for match in re.finditer(r'Global step (\d+): (\{[^\n]+\})', log.read_text(errors='replace')):
            step = int(match[1])
            status = ast.literal_eval(match[2])
            rs = records[step]
            # Equal-size microbatches make the logged mean response-weighted.
            # Count all rank records across this optimizer update's accumulation.
            if len(rs) != world * accumulation or {r['responses'] for r in rs} != {micro}:
                raise ValueError(f'{arm} step {step}: response-mean reduction requires review')
            curve.append({'step': step, 'training_avg_at_8': status['score'],
                          'training_reward': status['reward'], 'prompts': prompts,
                          'responses': prompts * repeats})
        if not curve:
            continue
        assert [r['step'] for r in curve] == list(range(1, len(curve) + 1))
        target = folder / 'training_avg8.csv'
        with target.with_suffix('.csv.tmp').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        target.with_suffix('.csv.tmp').replace(target)
        avg = lambda rows: sum(r['training_avg_at_8'] * r['responses'] for r in rows) / sum(r['responses'] for r in rows)
        output['arms'][arm] = {'recorded_steps': len(curve), 'last_step_avg_at_8': curve[-1]['training_avg_at_8'],
                               'first_50_steps_avg_at_8': avg(curve[:50]),
                               'last_50_steps_avg_at_8': avg(curve[-50:]),
                               'all_recorded_steps_avg_at_8': avg(curve),
                               'curve': str(target.relative_to(root))}
    tmp = root / 'training_avg8_summary.json.tmp'
    tmp.write_text(json.dumps(output, indent=2) + '\n')
    tmp.replace(root / 'training_avg8_summary.json')
    print(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    root = parser.parse_args().run_dir.resolve()
    summarize(root)
    handoff_at_completed_training(root)
