#!/usr/bin/env bash
#SBATCH --job-name=nplus-pythia410m-frontier-20261111
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs

export HF_HOME=${HF_HOME:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_datasets_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/work/kompatj1/summarize_from_feedback_details/.hf_home/transformers}
export TMPDIR=${TMPDIR:-/scratch/work/kompatj1/summarize_from_feedback_details/.tmp}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/python scripts/evaluate_nplus_sft_ppo_frontier.py \
  --sft-repo vwxyzjn/EleutherAI_pythia-410m-deduped__sft__tldr \
  --sft-revision seed444_1704124946 \
  --ppo-repo vwxyzjn/EleutherAI_pythia-410m-deduped__ppo__tldr \
  --ppo-revision seed444_1704212847 \
  --base-tokenizer EleutherAI/pythia-410m-deduped \
  --num-prompts 200 \
  --batch-size 2 \
  --max-new-tokens 53 \
  --temperature 0.01 \
  --output-dir outputs/nplus_pythia410m_seed444_frontier \
  --run-frontier
