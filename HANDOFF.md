# N+ TL;DR RLHF Reproduction Handoff

Date written: 2026-11-09 UTC
Remote path: `/scratch/work/kompatj1/summarize_from_feedback_details`
Repo: `https://github.com/vwxyzjn/summarize_from_feedback_details`
Clone commit: `ee77a9abc31f5338365ee087945c5bcafac7f1ca`

## Current Status

The N+ repository has been cloned onto Triton. The first released-checkpoint
evaluation passed Stage 0 under the frontier judge. PPO clearly beat SFT.

Downloaded released checkpoints for the first Stage 0 attempt:

```text
SFT:
repo     vwxyzjn/EleutherAI_pythia-2.8b-deduped__sft__tldr
revision sft__44413__1708611267
cache    .hf_home/hub/models--vwxyzjn--EleutherAI_pythia-2.8b-deduped__sft__tldr/snapshots/9fad61fe3556e5365e485ec84cc790811ef9704c

PPO:
repo     vwxyzjn/EleutherAI_pythia-2.8b-deduped__ppo_left_padding_new_nowhiten_reward__tldr
revision ppo_left_padding_new_nowhiten_reward__44413__1710356835
cache    .hf_home/hub/models--vwxyzjn--EleutherAI_pythia-2.8b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/snapshots/3b12839d6b3c0cb0471b7aa5570e5ae34057bfac
```

Prepared scripts:

```text
scripts/evaluate_nplus_sft_ppo_frontier.py
run_nplus_pythia28_frontier_eval_20261109.sh
```

Completed Stage 0 jobs:

```text
21481358 nplus-pythia28-frontier-20261109 COMPLETED
21482935 nplus-pythia28-frontier-judge-20261109 COMPLETED
```

Stage 0 frontier result:

```text
ppo_vs_sft:
  n=200
  ppo_wins=168
  sft_wins=29
  ties=3
  ppo_win_rate_all=0.84
  ppo_win_rate_ignoring_ties=0.8527918781725888
```

Output:

```text
outputs/nplus_pythia28_seed44413_frontier/frontier_judge/frontier_metrics.json
```

Stage 1/2 integration has now been added:

```text
summarize_from_feedback_details/reward_proxy.py
summarize_from_feedback_details/reward.py
summarize_from_feedback_details/ppo.py
deepspeed_4gpu.yaml
run_nplus_pythia28_residual_rm_20261109.sh
run_nplus_pythia28_naive_proxy_ppo_20261109.sh
run_nplus_pythia28_residual_proxy_ppo_20261109.sh
```

Implementation summary:

- `reward.py` has optional `--residual_learning`.
- Residual RM training optimizes the Bradley-Terry margin
  `learned_chosen - learned_rejected + proxy_chosen - proxy_rejected`.
- `ppo.py` has `--reward_mode learned|learned_plus_proxy|proxy_only`.
- Naive summation PPO uses the released N+ RM with `--reward_mode learned_plus_proxy`.
- Residual PPO uses the residual RM with `--reward_mode learned_plus_proxy`.
- Proxy reward implementation is copied into the N+ repo from the previous project
  as `reward_proxy.py`.

Submitted Stage 1/2 jobs:

```text
21483923 nplus-pythia28-residual-rm-20261109      PENDING
21483924 nplus-pythia28-naive-proxy-ppo-20261109  PENDING
21483925 nplus-pythia28-residual-proxy-ppo-20261109 PENDING dependency afterok:21483923
```

The motivation is to stop debugging residual reward composition on a summarization
pipeline where vanilla RLHF itself does not clearly improve over SFT. Instead, use
a summarization RLHF setup that has already been shown to work, then layer the
project's proxy/residual experiments on top of that stable substrate.

## Experimental Ladder

Stage 0: Reproduce or validate a known-working vanilla RLHF summarization result.

- Use the N+ TL;DR setup and released artifacts if possible.
- First evaluate released SFT vs PPO checkpoints under our frontier judge.
- The required gate is clear frontier-judge preference for PPO over SFT.
- If released PPO does not beat released SFT under our arbiter, do not continue
  to residual-vs-naive experiments on this substrate without a new reason.

Stage 1: Add the proxy reward with naive summation.

- Starting from the same working SFT/RM/PPO substrate, train/evaluate a PPO policy
  with reward equal to learned preference reward plus the project's proxy reward.
- The required gate is that naive proxy summation beats vanilla RLHF, not merely
  that it increases the proxy score.

Stage 2: Test residual reward learning against naive summation.

- Train a residual reward model that learns the part of the preference signal not
  already explained by the proxy.
- Compare residual-plus-proxy PPO against naive learned-plus-proxy PPO.
- Main arbiter should be frontier pairwise judge on held-out TL;DR prompts, with
  local reward-model/proxy diagnostics as supporting evidence only.

## Why This Pivot

The current GPT-Neo-1.3B InstructGPT-style TL;DR run did not pass Stage 0:

- Frontier judge on PPO vs SFT was essentially tied:
  - PPO wins: 46
  - SFT wins: 47
  - ties: 107
  - PPO win rate ignoring ties: about 0.495
- The local independent judge also did not prefer PPO over SFT.
- Preference-RM/frontier agreement on the exact PPO-vs-SFT comparisons was weak:
  - preference RM preferred PPO 43 times, SFT 60 times, tie 97 times
  - frontier preferred PPO 46 times, SFT 47 times, tie 107 times
  - agreement on non-tie/non-tie cases: 48/86 = 0.558
- PPO reward logs were mostly flat rather than showing a strong useful climb.

Conclusion: this is not yet evidence against residual reward learning. It is
evidence that the current TL;DR PPO substrate is not a reliable stage-0 benchmark.

## Preferred Next Steps

1. Watch jobs `21483923`, `21483924`, and `21483925`.
2. If `21483923` fails, `21483925` will remain dependency-blocked; inspect the
   residual RM logs first.
3. If PPO jobs OOM on 4xA100, reduce `local_rollout_forward_batch_size` and/or
   `local_micro_batch_size` in the launcher scripts.
4. After each PPO arm completes, evaluate generated summaries with the same
   `policy_scores.jsonl` / frontier-pairwise-judge path used for Stage 0.

Check with:

```bash
cd /scratch/work/kompatj1/summarize_from_feedback_details
squeue -j 21483923,21483924,21483925 -o "%.18i %.50j %.10T %.10M %.80R %.24P"
sacct -j 21483923,21483924,21483925 --format=JobID,JobName%50,State,ExitCode,Start,End,Elapsed -P
tail -120 logs/nplus-pythia28-residual-rm-20261109-21483923.out
tail -120 logs/nplus-pythia28-residual-rm-20261109-21483923.err
tail -120 logs/nplus-pythia28-naive-proxy-ppo-20261109-21483924.out
tail -120 logs/nplus-pythia28-naive-proxy-ppo-20261109-21483924.err
```

## Compute Expectations

Start with released checkpoints. This should require only evaluation/generation
compute plus frontier judge API calls, not full SFT/RM/PPO training.

Full reproduction of the N+ training stack is much more expensive and should be
avoided until released-checkpoint evaluation confirms that this substrate passes
Stage 0 under our frontier judge.

Rough tiers:

- Released-checkpoint evaluation: one GPU, likely hours or less plus API calls.
- One model/one seed training reproduction: likely A100/H100/H200 scale, potentially
  several GPU-hours to day-scale depending model size and script path.
- Full N+ scaling reproduction: unnecessary for the residual reward question.

## Useful Related Files In The Existing Project

Existing project path:
`/scratch/work/kompatj1/reward_composition_LLM`

Useful outputs from the failed GPT-Neo stage-0 attempt:

- `outputs/instructgpt_neo13b_20261021/frontier_judge/frontier_metrics.json`
- `outputs/instructgpt_neo13b_20261021/diagnostics/ppo_reward_log_summary.json`
- `outputs/instructgpt_neo13b_20261021/diagnostics/preference_rm_frontier_agreement_metrics.json`

Current rerank diagnostic replacement job, if still relevant:

- Slurm job: `21478166`
- Output dir:
  `outputs/instructgpt_neo13b_20261021/rerank_pref_rm_sft8_20261109`

Check with:

```bash
squeue -j 21478166 -o "%.18i %.45j %.10T %.10M %.80R %.24P"
sacct -j 21478166 --format=JobID,JobName%45,State,ExitCode,Start,End,Elapsed -P
```

## Notes For Future Work

Keep the stages separate. Do not interpret a proxy/residual result unless the
underlying vanilla RLHF setup first beats SFT under the intended arbiter.

The goal is not to prove that a stronger model is necessary. The goal is to choose
a summarization benchmark where vanilla RLHF is already known to work, so the
residual-vs-naive experiment tests reward composition rather than re-debugging
RLHF from scratch.
