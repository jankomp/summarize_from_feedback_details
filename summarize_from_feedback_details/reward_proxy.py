from __future__ import annotations

import math
import re
from collections import Counter
from typing import Literal

import torch


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def coverage_reward(source: str, summary: str) -> float:
    source_tokens = set(tokenize(source))
    summary_tokens = tokenize(summary)
    if not summary_tokens:
        return 0.0
    return sum(token in source_tokens for token in summary_tokens) / len(summary_tokens)


def length_reward(summary: str, target_length: int = 48, sigma: float = 20.0) -> float:
    n = len(tokenize(summary))
    return math.exp(-0.5 * ((n - target_length) / sigma) ** 2)


def redundancy_reward(summary: str) -> float:
    tokens = tokenize(summary)
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def proxy_reward(
    source: str,
    summary: str,
    lambda_coverage: float = 1.0,
    lambda_length: float = 1.0,
    lambda_redundancy: float = 1.0,
) -> float:
    return (
        lambda_coverage * coverage_reward(source, summary)
        + lambda_length * length_reward(summary)
        - lambda_redundancy * redundancy_reward(summary)
    )


def normalize_value(
    value: float,
    mean: float,
    std: float,
    clip: float | None,
    normalization: Literal["zscore", "hero_minmax"] = "zscore",
    min_value: float = 0.0,
    max_value: float = 1.0,
    target_min: float = -1.0,
    target_max: float = 1.0,
) -> float:
    if normalization == "hero_minmax":
        if max_value <= min_value:
            raise ValueError("proxy_reward_max must be greater than proxy_reward_min")
        clipped = max(min_value, min(max_value, value))
        scaled = (clipped - min_value) / (max_value - min_value)
        return target_min + scaled * (target_max - target_min)
    if normalization != "zscore":
        raise ValueError(f"Unsupported proxy reward normalization: {normalization}")
    if std <= 0:
        raise ValueError("proxy_reward_std must be positive")
    normalized = (value - mean) / std
    if clip is not None:
        normalized = max(-clip, min(clip, normalized))
    return normalized


def batch_proxy_rewards(
    tokenizer,
    query_tokens: torch.Tensor,
    response_tokens: torch.Tensor,
    scale: float,
    normalize: bool,
    mean: float,
    std: float,
    clip: float | None,
    normalization: Literal["zscore", "hero_minmax"] = "zscore",
    min_value: float = 0.0,
    max_value: float = 1.0,
    target_min: float = -1.0,
    target_max: float = 1.0,
    lambda_coverage: float = 1.0,
    lambda_length: float = 1.0,
    lambda_redundancy: float = 1.0,
    skip_special_tokens: bool = True,
) -> torch.Tensor:
    values = []
    queries = query_tokens.detach().cpu()
    responses = response_tokens.detach().cpu()
    for query, response in zip(queries, responses):
        source = tokenizer.decode(query, skip_special_tokens=skip_special_tokens)
        summary = tokenizer.decode(response, skip_special_tokens=skip_special_tokens).strip()
        value = scale * proxy_reward(
            source,
            summary,
            lambda_coverage=lambda_coverage,
            lambda_length=lambda_length,
            lambda_redundancy=lambda_redundancy,
        )
        if normalize:
            value = normalize_value(
                value,
                mean=mean,
                std=std,
                clip=clip,
                normalization=normalization,
                min_value=min_value,
                max_value=max_value,
                target_min=target_min,
                target_max=target_max,
            )
        values.append(value)
    return torch.tensor(values, dtype=torch.float32, device=query_tokens.device)
