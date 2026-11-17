#!/usr/bin/env bash
#SBATCH --job-name=nplus-smoke-rm-pythia28-4gpu-debug-20261113
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs/smoke models

module load triton/2024.1-gcc gcc/11.4.0

export TMPDIR=${TMPDIR:-/tmp/${USER}/nplus-smoke-${SLURM_JOB_ID:-manual}}
export HF_HOME=${HF_HOME:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/transformers}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PATH=/scratch/work/kompatj1/reward_composition_LLM/.venv/bin:${PATH}
export CC=gcc
export CXX=g++
export PYTHONPATH=/scratch/work/kompatj1/summarize_from_feedback_details:${PYTHONPATH:-}
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/hub/models--vwxyzjn--EleutherAI_pythia-2.8b-deduped__sft__tldr/snapshots/9fad61fe3556e5365e485ec84cc790811ef9704c}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smoke/residual_rm_pythia28_4gpu_debug}"
PROXY_SCALE="${PROXY_SCALE:-1.0}"

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/accelerate launch --config_file deepspeed_4gpu.yaml \
  summarize_from_feedback_details/reward.py \
  --base_model EleutherAI/pythia-2.8b-deduped \
  --sft_model_path "$SFT_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed \
  --seed 44413 \
  --residual_learning \
  --proxy_reward_scale "$PROXY_SCALE" \
  --total_episodes 16 \
  --max_norm_samples 8 \
  --gradient_accumulation_steps 2 \
  --local_micro_batch_size 1 \
  --local_eval_batch_size 1
