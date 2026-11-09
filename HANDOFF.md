# N+ TL;DR RLHF Reproduction Handoff

Date written: 2026-11-09 UTC
Remote path: `/scratch/work/kompatj1/summarize_from_feedback_details`
Repo: `https://github.com/vwxyzjn/summarize_from_feedback_details`
Clone commit: `ee77a9abc31f5338365ee087945c5bcafac7f1ca`

## Current Status

The N+ repository has been cloned onto Triton. The first released-checkpoint
evaluation has been prepared and submitted, but has not started running yet as of
the latest check.

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

Submitted Slurm job:

```text
21481358 nplus-pythia28-frontier-20261109 PENDING on gpu-a100-80g, reason Priority
```

The job generates SFT/PPO summaries for 200 validation prompts, writes
`outputs/nplus_pythia28_seed44413_frontier/policy_scores.jsonl`, and then runs
the frontier pairwise judge only if `AALTO_OPENAI_API_KEY` or `OPENAI_API_KEY` is
present in the batch environment. If no key is present, it should skip only the
frontier stage after summary generation.

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

1. Watch job `21481358`.
2. If it completes summary generation but skips frontier judging due to missing
   API key, resubmit the same launcher from a shell that exports
   `AALTO_OPENAI_API_KEY`, or run only the frontier command against the generated
   `policy_scores.jsonl`.
3. If the job fails before summary generation completes, inspect the Slurm logs
   and fix the evaluator.
4. If Stage 0 passes, plan minimal training modifications for Stage 1 and Stage 2.

Check with:

```bash
cd /scratch/work/kompatj1/summarize_from_feedback_details
squeue -j 21481358 -o "%.18i %.45j %.10T %.10M %.80R %.24P"
sacct -j 21481358 --format=JobID,JobName%45,State,ExitCode,Start,End,Elapsed -P
tail -120 logs/nplus-pythia28-frontier-20261109-21481358.out
tail -120 logs/nplus-pythia28-frontier-20261109-21481358.err
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
