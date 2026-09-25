"""Append Pro-only after an existing three-arm study, without interrupting training.

Usage: python run_pro_only_extension.py --run-dir /path/to/frozen/study
The study supplies run_small.py, plan.json, frozen-sha256.json and training source.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def train_arguments(study):
    cfg = study.P['arms']['pro_only']
    assert cfg == {'estimator': 'rloo', 'gate': 'reinforce_pro', 'global_normalization': True}
    assert study.P['arms']['baseline'] == dict(cfg, gate='icepop')
    run = study.R / 'pro_only'
    args = list(study.P['original_argv'])
    args[0] = sys.executable
    changes = {
        '--advantage_estimator': cfg['estimator'],
        '--vllm_is_correction_type': cfg['gate'],
        '--prompt_data': str(study.R / 'train.jsonl'),
        '--max_samples': '3200',
        '--save_path': str(run / 'final_model'),
        '--ckpt_path': str(run / 'checkpoints'),
        '--save_steps': '-1', '--max_ckpt_num': '1',
        '--use_tensorboard': str(run / 'tensorboard'),
    }
    for flag, value in changes.items():
        args[args.index(flag) + 1] = value
    args.append('--study_global_rloo_norm')
    return run, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    spec = importlib.util.spec_from_file_location('study_runner', root / 'run_small.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    run, command = train_arguments(study)
    if args.validate_only:
        print(json.dumps({'status': 'validated', 'argv': command}, indent=2))
        return
    with (root / 'pro_only_extension.lock').open('a') as extension_lock:
        fcntl.flock(extension_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / 'pro_only_extension.pid').write_text(str(os.getpid()) + '\n')
        print('Pro-only queued; waiting for the existing study lock.', flush=True)
        with (root / 'study.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if (root / 'STOP').exists():
                    raise RuntimeError('Study has a STOP marker')
                for name in ['baseline', 'max_only', 'pro_max', 'initial']:
                    assert (root / name / 'evaluation' / '_SUCCESS').exists(), name
                for relative, expected in json.loads((root / 'frozen-sha256.json').read_text()).items():
                    assert study.digest(root / relative) == expected, relative
                run.mkdir(exist_ok=True)
                env = study.environment()
                env['PROMAX_STUDY_METRICS'] = str(run / 'metrics')
                study.state('training', run='pro_only', planned_updates=100)
                study.execute(command, root / 'source', env, run, 'train')
                study.evaluate('pro_only', run / 'final_model')
                study.summarize()
                subprocess.run([sys.executable, str(root / 'analyze.py')], check=True)
                study.state('complete', summary=str(root / 'analysis.json'), completed_arms=4)
            except Exception as exc:
                study.state('failed', run='pro_only', error=repr(exc))
                raise


if __name__ == '__main__':
    main()
