#!/usr/bin/env bash
#SBATCH --job-name=nplus-pythia28-residual-rm-hero-norm-h200ellis1n-20261118
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs models

module load triton/2024.1-gcc gcc/11.4.0

export TMPDIR=${TMPDIR:-/tmp/${USER}/nplus-${SLURM_JOB_ID:-manual}}
export HF_HOME=${HF_HOME:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/transformers}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PATH=/scratch/work/kompatj1/reward_composition_LLM/.venv/bin:${PATH}
export CC=gcc
export CXX=g++
export PYTHONPATH=/scratch/work/kompatj1/summarize_from_feedback_details:${PYTHONPATH:-}
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

OUTPUT_DIR="${OUTPUT_DIR:-models/EleutherAI/pythia-2.8b-deduped/reward_model_residual_proxy_s1_44413_hero_norm_h200ellis1n}"

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/accelerate launch --config_file deepspeed_4gpu.yaml \
  summarize_from_feedback_details/reward.py \
  --base_model EleutherAI/pythia-2.8b-deduped \
  --sft_model_path /scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/hub/models--vwxyzjn--EleutherAI_pythia-2.8b-deduped__sft__tldr/snapshots/9fad61fe3556e5365e485ec84cc790811ef9704c \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed \
  --run_eval \
  --seed 44413 \
  --residual_learning \
  --proxy_reward_scale 1.0 \
  --proxy_reward_normalize \
  --proxy_reward_mean 1.3401572846510823 \
  --proxy_reward_std 0.21096500187943543 \
  --proxy_reward_clip 2.0 \
  --gradient_accumulation_steps 8 \
  --local_micro_batch_size 1 \
  --local_eval_batch_size 1
