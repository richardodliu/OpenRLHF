"""Run only the fixed four-arm short study with an uncorrected baseline; preserve old cancellation."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,datetime,fcntl,time
R=Path(sys.argv[sys.argv.index('--run-dir')+1]).resolve() if '--run-dir' in sys.argv else Path(__file__).resolve().parent;P=json.loads((R/'plan.json').read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(phase,**kw):
 obj=dict(phase=phase,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);tmp=R/'status.json.tmp';tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(R/'status.json');print(json.dumps(obj),flush=True)
def free_gpus():
 for _ in range(30):
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
  if len(lines)==8 and max(map(int,lines))<3000:return
  time.sleep(10)
 raise RuntimeError('GPUs remain occupied; no existing process is stopped.')
def execute(args,cwd,env,folder,kind):
 folder.mkdir(exist_ok=True,parents=True)
 success=folder/f'{kind}_exit_code.txt'
 if success.exists():
  if success.read_text().strip()=='0':return
  raise RuntimeError(f'Prior failure requires inspection: {success}')
 if (R/'STOP').exists():raise RuntimeError('Study stopped by STOP file')
 free_gpus();(folder/f'{kind}_command.json').write_text(json.dumps({'argv':args,'cwd':str(cwd)},indent=2)+'\n')
 with (folder/f'{kind}.log').open('ab') as log:
  child=subprocess.Popen(args,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT);(folder/f'{kind}.pid').write_text(str(child.pid)+'\n');rc=child.wait()
 success.write_text(str(rc)+'\n')
 if rc:raise RuntimeError(f'{kind} failed ({rc}); inspect {folder}')
 time.sleep(5)
def environment():
 e=os.environ.copy();e.update(PYTHONPATH=str(R/'source'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',WANDB_MODE='disabled',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',VLLM_WORKER_MULTIPROC_METHOD='spawn');e['PATH']=str(Path(sys.executable).parent)+':'+e.get('PATH','');e.pop('WANDB_API_KEY',None);e.pop('RAY_ADDRESS',None);return e
def evaluate(name,model):
 folder=R/name/'evaluation';env=environment();env['CUDA_VISIBLE_DEVICES']='0,1,2,3';state('evaluating',run=name)
 execute([sys.executable,'-u',str(R/'evaluate_aime25.py'),'--model',str(model),'--output',str(folder),'--suite','math'],R,env,folder,'eval')
 if not (folder/'_SUCCESS').exists():raise RuntimeError('Missing completed evaluation marker')
def summarize():
 result={}
 for name in ['initial',*P['arms']]:
  f=R/name/'evaluation/results.json'
  if f.exists():result[name]=json.loads(f.read_text())['benchmarks']['aime25']
 (R/'summary.json').write_text(json.dumps({'completed_evaluations':result,'plan_sha256':digest(R/'plan.json'),'scope':P['scope']},indent=2)+'\n')
def main():
 assert json.loads((R/'preflight.json').read_text())['status']=='passed'
 assert json.loads((R/'reward-validation.json').read_text())['status']=='passed'
 for relative,expected in json.loads((R/'frozen-sha256.json').read_text()).items():assert digest(R/relative)==expected,relative
 for arm,cfg in P['arms'].items():
  run=R/arm;run.mkdir(exist_ok=True);args=list(P['original_argv']);args[0]=sys.executable
  changes={'--advantage_estimator':cfg['estimator'],'--prompt_data':str(R/'train.jsonl'),'--max_samples':'3200','--save_path':str(run/'final_model'),'--ckpt_path':str(run/'checkpoints'),'--save_steps':'-1','--max_ckpt_num':'1','--use_tensorboard':str(run/'tensorboard')}
  for flag,value in changes.items():args[args.index(flag)+1]=value
  if cfg['gate']=='none':
   args.remove('--enable_vllm_is_correction')
   i=args.index('--vllm_is_correction_type');del args[i:i+2]
  else:args[args.index('--vllm_is_correction_type')+1]=cfg['gate']
  if cfg['global_normalization']:args.append('--study_global_rloo_norm')
  env=environment();env['PROMAX_STUDY_METRICS']=str(run/'metrics');state('training',run=arm,planned_updates=100)
  execute(args,R/'source',env,run,'train');evaluate(arm,run/'final_model');summarize()
 evaluate('initial',P['model']);summarize();subprocess.run([sys.executable,str(R/'analyze.py')],check=True);state('complete',summary=str(R/'analysis.json'))
if __name__=='__main__':
 with (R/'study.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(R/'runner.pid').write_text(str(os.getpid())+'\n')
  try:
   state('queued',waiting_for=P['supersedes'],next_run='baseline')
   with (Path(P['supersedes'])/'study.lock').open('a') as previous_lock:
    fcntl.flock(previous_lock,fcntl.LOCK_EX)
   main()
  except Exception as exc:state('failed',error=repr(exc));raise
