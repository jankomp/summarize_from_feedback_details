from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def decode_response(tokenizer, response_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> str:
    ids = response_ids.detach().cpu().tolist()
    if eos_token_id in ids:
        ids = ids[: ids.index(eos_token_id)]
    ids = [item for item in ids if item != pad_token_id]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_summaries(
    model_name_or_path: str,
    revision: str | None,
    tokenizer,
    rows,
    output_path: Path,
    policy_name: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
    print(f"Loading {policy_name}: {model_name_or_path} revision={revision}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature + 1e-7,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    results = []
    with output_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            queries = torch.tensor([row["query_token"] for row in batch_rows], dtype=torch.long)
            if torch.cuda.is_available():
                queries = queries.to("cuda")
            context_length = queries.shape[1]
            attention_mask = queries != tokenizer.pad_token_id
            input_ids = torch.masked_fill(queries, ~attention_mask, 0)
            torch.manual_seed(12345 + start)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345 + start)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
            responses = outputs.sequences[:, context_length:]
            for offset, (row, response) in enumerate(zip(batch_rows, responses)):
                prompt_index = start + offset
                summary = decode_response(tokenizer, response, tokenizer.eos_token_id, tokenizer.pad_token_id)
                out = {
                    "policy": policy_name,
                    "prompt_index": prompt_index,
                    "post": row["query"],
                    "reference": row["reference_response"].replace(tokenizer.eos_token or "", "").strip(),
                    "summary": summary,
                    "judge_reward": None,
                    "proxy_reward": None,
                }
                handle.write(json.dumps(out) + "\n")
                handle.flush()
                results.append(out)
            print(f"{policy_name}: generated {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-repo", default="vwxyzjn/EleutherAI_pythia-2.8b-deduped__sft__tldr")
    parser.add_argument("--sft-revision", default="sft__44413__1708611267")
    parser.add_argument("--ppo-repo", default="vwxyzjn/EleutherAI_pythia-2.8b-deduped__ppo_left_padding_new_nowhiten_reward__tldr")
    parser.add_argument("--ppo-revision", default="ppo_left_padding_new_nowhiten_reward__44413__1710356835")
    parser.add_argument("--base-tokenizer", default="EleutherAI/pythia-2.8b-deduped")
    parser.add_argument("--dataset", default="vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=53)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nplus_pythia28_seed44413_frontier"))
    parser.add_argument("--frontier-config", type=Path, default=None)
    parser.add_argument("--run-frontier", action="store_true")
    parser.add_argument("--frontier-script", type=Path, default=Path("/scratch/work/kompatj1/reward_composition_LLM/scripts/evaluate_frontier_pairwise_judge.py"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_scores = args.output_dir / "policy_scores.jsonl"
    if policy_scores.exists():
        policy_scores.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dataset = load_dataset(args.dataset)[args.split]
    rows = [dataset[i] for i in range(min(args.num_prompts, len(dataset)))]

    metadata = {
        "sft_repo": args.sft_repo,
        "sft_revision": args.sft_revision,
        "ppo_repo": args.ppo_repo,
        "ppo_revision": args.ppo_revision,
        "dataset": args.dataset,
        "split": args.split,
        "num_prompts": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    generate_summaries(
        args.sft_repo,
        args.sft_revision,
        tokenizer,
        rows,
        policy_scores,
        "sft",
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
    )
    generate_summaries(
        args.ppo_repo,
        args.ppo_revision,
        tokenizer,
        rows,
        policy_scores,
        "ppo",
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
    )

    if args.run_frontier:
        if not (os.environ.get("AALTO_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            print("Skipping frontier judge: AALTO_OPENAI_API_KEY/OPENAI_API_KEY is not set.", flush=True)
            return
        frontier_config = args.frontier_config or args.output_dir / "frontier_config.yaml"
        if not frontier_config.exists():
            frontier_config.write_text(
                "\n".join(
                    [
                        "seed: 42",
                        "pairwise_judge:",
                        "  model: gpt-5.6-sol-2026-07-09",
                        f"  output_dir: {args.output_dir / 'frontier_judge'}",
                        f"  num_prompts: {len(rows)}",
                        "  seed: 123",
                        "  max_retries: 4",
                        "  request_timeout: 300",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        cmd = [
            sys.executable,
            str(args.frontier_script),
            "--config",
            str(frontier_config),
            "--policy-scores",
            str(policy_scores),
            "--output-dir",
            str(args.output_dir / "frontier_judge"),
            "--pairs",
            "ppo:sft",
            "--num-prompts",
            str(len(rows)),
            "--max-retries",
            "4",
            "--request-timeout",
            "300",
        ]
        print("Running frontier judge:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
