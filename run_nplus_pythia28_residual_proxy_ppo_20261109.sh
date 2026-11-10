#!/usr/bin/env bash
#SBATCH --job-name=nplus-pythia28-residual-proxy-ppo-20261109
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:4
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs models

export TMPDIR=${TMPDIR:-/tmp/${USER}/nplus-${SLURM_JOB_ID:-manual}}
export HF_HOME=${HF_HOME:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/transformers}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH=/scratch/work/kompatj1/summarize_from_feedback_details:${PYTHONPATH:-}
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/hub/models--vwxyzjn--EleutherAI_pythia-2.8b-deduped__sft__tldr/snapshots/9fad61fe3556e5365e485ec84cc790811ef9704c}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-models/EleutherAI/pythia-2.8b-deduped/reward_model_residual_proxy_s1_44413}"
OUTPUT_DIR="${OUTPUT_DIR:-models/EleutherAI/pythia-2.8b-deduped/policy_model_residual_proxy_s1_44413}"
PROXY_SCALE="${PROXY_SCALE:-1.0}"

if [[ ! -f "$REWARD_MODEL_PATH/config.json" ]]; then
  echo "Residual reward model is missing at $REWARD_MODEL_PATH" >&2
  exit 2
fi

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/accelerate launch --config_file deepspeed_4gpu.yaml \
  summarize_from_feedback_details/ppo.py \
  --base_model EleutherAI/pythia-2.8b-deduped \
  --sft_model_path "$SFT_MODEL_PATH" \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed \
  --run_eval \
  --seed 44413 \
  --reward_mode learned_plus_proxy \
  --proxy_reward_scale "$PROXY_SCALE" \
  --local_rollout_forward_batch_size 16 \
  --gradient_accumulation_steps 16 \
  --local_micro_batch_size 2 \
  --local_eval_batch_size 1
