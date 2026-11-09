#!/usr/bin/env bash
#SBATCH --job-name=nplus-pythia28-residual-rm-20261109
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs models .tmp

export HF_HOME=${HF_HOME:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_datasets_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/transformers}
export TMPDIR=${TMPDIR:-/scratch/work/kompatj1/summarize_from_feedback_details/.tmp}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

SFT_MODEL_PATH="${SFT_MODEL_PATH:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/hub/models--vwxyzjn--EleutherAI_pythia-2.8b-deduped__sft__tldr/snapshots/9fad61fe3556e5365e485ec84cc790811ef9704c}"
OUTPUT_DIR="${OUTPUT_DIR:-models/EleutherAI/pythia-2.8b-deduped/reward_model_residual_proxy_s1_44413}"
PROXY_SCALE="${PROXY_SCALE:-1.0}"

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/accelerate launch --config_file deepspeed_4gpu.yaml \
  summarize_from_feedback_details/reward.py \
  --base_model EleutherAI/pythia-2.8b-deduped \
  --sft_model_path "$SFT_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed \
  --run_eval \
  --seed 44413 \
  --residual_learning \
  --proxy_reward_scale "$PROXY_SCALE" \
  --local_eval_batch_size 1
