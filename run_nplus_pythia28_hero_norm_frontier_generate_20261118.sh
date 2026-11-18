#!/usr/bin/env bash
#SBATCH --job-name=nplus-pythia28-hero-norm-frontier-generate-20261118
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
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

NAIVE_POLICY=models/EleutherAI/pythia-2.8b-deduped/policy_model_naive_proxy_s1_44413_hero_norm_h200ellis1n
RESIDUAL_POLICY=models/EleutherAI/pythia-2.8b-deduped/policy_model_residual_proxy_s1_44413_hero_norm_reg_h200ellis1n

if [[ ! -f "$NAIVE_POLICY/config.json" ]]; then
  echo "Naive HERO-normalized policy missing at $NAIVE_POLICY" >&2
  exit 2
fi
if [[ ! -f "$RESIDUAL_POLICY/config.json" ]]; then
  echo "Residual HERO-normalized policy missing at $RESIDUAL_POLICY" >&2
  exit 2
fi

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/python scripts/evaluate_nplus_two_policies_frontier.py \
  --base-policy-scores outputs/nplus_pythia28_residual_h200ellis1n_frontier/policy_scores.jsonl \
  --base-policies sft vanilla \
  --policy naive "$NAIVE_POLICY" \
  --policy residual "$RESIDUAL_POLICY" \
  --num-prompts 200 \
  --batch-size 2 \
  --max-new-tokens 53 \
  --temperature 0.01 \
  --output-dir outputs/nplus_pythia28_hero_norm_frontier
