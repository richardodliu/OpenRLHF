set -x

source /volume/pt-train/users/rbliu/miniconda3/bin/activate
conda activate openrlhf

export WANDB_API_KEY=local-675f9d5bdf4fa6ab2b8f55ff8561852edea35b63
export WANDB_BASE_URL="http://wandb-app.t-skyinfer-phwu.svc:8080"
export WANDB_MODE=offline

MODEL_SIZE=7B
MODEL_PATH=/volume/pt-train/users/rbliu/model/Qwen2.5-Math-${MODEL_SIZE}
KL_TYPE=k2
KL_COEF=0.000

cd /volume/pt-train/users/rbliu/github/OpenRLHF

timestamp=$(date +"%Y%m%d-%H%M")
logfile="/volume/pt-train/users/rbliu/github/OpenRLHF/logs/reinforce_pro_max.log"

python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.6 \
    --init_kl_coef ${KL_COEF} \
    --entropy_loss_coef 0.0 \
    --gamma 1.0 \
    --use_kl_loss \
    --kl_estimator ${KL_TYPE} \
    --advantage_estimator reinforce_pro_max \
    --pretrain ${MODEL_PATH} \
    --remote_rm_url /volume/pt-train/users/rbliu/github/OpenRLHF/examples/python/mathverify.py \
    --save_path /volume/pt-train/users/rbliu/ckpt/rlhf/reinforce_pro_max/${KL_COEF}-${MODEL_SIZE}-${KL_TYPE}-${timestamp} \
    --ckpt_path /volume/pt-train/users/rbliu/ckpt/rlhf/reinforce_pro_max/${KL_COEF}-${MODEL_SIZE}-${KL_TYPE}-${timestamp} \
    --save_steps 9999 \
    --save_hf_ckpt \
    --micro_train_batch_size 32 \
    --train_batch_size 256 \
    --micro_rollout_batch_size 32 \
    --rollout_batch_size 32 \
    --n_samples_per_prompt 8 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --max_samples 999999999 \
    --generate_max_len 3072 \
    --zero_stage 3 \
    --actor_learning_rate 3e-6 \
    --prompt_data /volume/pt-train/users/rbliu/rbliu-data/dataset/Openr1-Math-7k-4096.jsonl \
    --input_key question \
    --label_key label \
    --apply_chat_template \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --use_wandb local-675f9d5bdf4fa6ab2b8f55ff8561852edea35b63 \
    2>&1 | tee "$logfile" 2>&1