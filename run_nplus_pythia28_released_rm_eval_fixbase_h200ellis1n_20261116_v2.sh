#!/usr/bin/env bash
#SBATCH --job-name=nplus-pythia28-released-rm-eval-fixbase-v2-h200ellis1n-20261116
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs/eval models

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

SFT_MODEL_PATH="${SFT_MODEL_PATH:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/hub/models--vwxyzjn--EleutherAI_pythia-2.8b-deduped__sft__tldr/snapshots/9fad61fe3556e5365e485ec84cc790811ef9704c}"
RELEASED_REWARD_MODEL_PATH="${RELEASED_REWARD_MODEL_PATH:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/hub/models--vwxyzjn--EleutherAI_pythia-2.8b-deduped__reward__tldr/snapshots/1dffce01e3c1227664b807857578add1d33aac58}"
REWARD_MODEL_PATH="${TMPDIR}/released_reward_model_fixed_base"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/released_reward_model_diagnostics_fixbase}"

rm -rf "$REWARD_MODEL_PATH"
cp -aL "$RELEASED_REWARD_MODEL_PATH" "$REWARD_MODEL_PATH"
python - "$REWARD_MODEL_PATH/config.json" "$SFT_MODEL_PATH" <<'PY'
import json
import sys

config_path, sft_model_path = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    config = json.load(f)
config["base_model"] = sft_model_path
config.setdefault("base_config", {})["_name_or_path"] = sft_model_path
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/accelerate launch --config_file deepspeed_4gpu.yaml \
  summarize_from_feedback_details/reward.py \
  --base_model EleutherAI/pythia-2.8b-deduped \
  --sft_model_path "$SFT_MODEL_PATH" \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed \
  --run_eval \
  --seed 44413 \
  --num_train_epochs 0 \
  --total_episodes 32 \
  --max_norm_samples 1024 \
  --local_eval_batch_size 1
