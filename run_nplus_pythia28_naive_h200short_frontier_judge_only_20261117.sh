#!/usr/bin/env bash
#SBATCH --job-name=nplus-pythia28-naive-h200short-frontier-judge-20261117
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /scratch/work/kompatj1/summarize_from_feedback_details
mkdir -p logs outputs/nplus_pythia28_naive_h200short_frontier

if [[ -z "${AALTO_OPENAI_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set AALTO_OPENAI_API_KEY or OPENAI_API_KEY in the shell that submits this job." >&2
  exit 2
fi

cat > outputs/nplus_pythia28_naive_h200short_frontier/frontier_config.yaml <<'EOF'
seed: 42
pairwise_judge:
  model: gpt-5.6-sol-2026-07-09
  output_dir: outputs/nplus_pythia28_naive_h200short_frontier/frontier_judge
  num_prompts: 200
  seed: 123
  max_retries: 4
  request_timeout: 300
EOF

/scratch/work/kompatj1/reward_composition_LLM/.venv/bin/python \
  /scratch/work/kompatj1/reward_composition_LLM/scripts/evaluate_frontier_pairwise_judge.py \
  --config outputs/nplus_pythia28_naive_h200short_frontier/frontier_config.yaml \
  --policy-scores outputs/nplus_pythia28_naive_h200short_frontier/policy_scores.jsonl \
  --output-dir outputs/nplus_pythia28_naive_h200short_frontier/frontier_judge \
  --pairs naive:sft naive:vanilla \
  --num-prompts 200 \
  --max-retries 4 \
  --request-timeout 300
