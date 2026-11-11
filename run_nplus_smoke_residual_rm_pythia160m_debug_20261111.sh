#!/usr/bin/env bash
#SBATCH --job-name=nplus-smoke-rm-pythia160m-debug-20261111
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs/smoke models

export TMPDIR=${TMPDIR:-/tmp/${USER}/nplus-smoke-${SLURM_JOB_ID:-manual}}
export HF_HOME=${HF_HOME:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/transformers}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH=/scratch/work/kompatj1/summarize_from_feedback_details:${PYTHONPATH:-}
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/smoke/residual_rm_pythia160m_debug}"
PROXY_SCALE="${PROXY_SCALE:-1.0}"

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/accelerate launch --config_file deepspeed_1gpu.yaml \
  summarize_from_feedback_details/reward.py \
  --base_model EleutherAI/pythia-160m \
  --sft_model_path EleutherAI/pythia-160m \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed \
  --seed 44413 \
  --residual_learning \
  --proxy_reward_scale "$PROXY_SCALE" \
  --total_episodes 4 \
  --gradient_accumulation_steps 2 \
  --local_micro_batch_size 1 \
  --local_eval_batch_size 1
