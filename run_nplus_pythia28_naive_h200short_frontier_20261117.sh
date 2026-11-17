#!/usr/bin/env bash
#SBATCH --job-name=nplus-pythia28-naive-h200short-frontier-20261117
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs

module load triton/2024.1-gcc gcc/11.4.0

export HF_HOME=${HF_HOME:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_datasets_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/transformers}
export TMPDIR=${TMPDIR:-/tmp/${USER}/nplus-${SLURM_JOB_ID:-manual}}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PATH=/scratch/work/kompatj1/reward_composition_LLM/.venv/bin:${PATH}
export PYTHONPATH=/scratch/work/kompatj1/summarize_from_feedback_details:${PYTHONPATH:-}
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/python scripts/evaluate_nplus_sft_vanilla_naive_frontier.py \
  --naive-path models/EleutherAI/pythia-2.8b-deduped/policy_model_naive_proxy_s1_44413_h200short \
  --num-prompts 200 \
  --batch-size 2 \
  --max-new-tokens 53 \
  --temperature 0.01 \
  --output-dir outputs/nplus_pythia28_naive_h200short_frontier \
  --run-frontier
