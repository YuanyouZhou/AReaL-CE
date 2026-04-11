from __future__ import annotations

from typing import Any

import torch

from areal.utils import stats_tracker


def infer_token_denominator(
    input_data: dict[str, Any],
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Infer the full token mask for stats logging.

    Context parallelism may slice intermediate tensors such as ``loss_mask`` or
    model outputs, while the original micro-batch metadata still describes the
    full logical sequence. Prefer that metadata for ``n_tokens`` so statistics
    stay consistent with and without context parallelism.
    """
    common_kwargs = {"dtype": torch.bool, "device": fallback.device}

    attention_mask = input_data.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return torch.ones_like(attention_mask, **common_kwargs)

    cu_seqlens = input_data.get("cu_seqlens")
    if isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.numel() > 0:
        return torch.ones(int(cu_seqlens[-1].item()), **common_kwargs)

    input_ids = input_data.get("input_ids")
    # Tree-packed batches keep input_ids padded to tree size while token-level
    # stats stay at packed-token length. Only reuse input_ids when it already
    # matches the stat tensor shape.
    if isinstance(input_ids, torch.Tensor) and input_ids.shape == fallback.shape:
        return torch.ones_like(input_ids, **common_kwargs)

    return torch.ones_like(fallback, **common_kwargs)


def log_conditional_entropy_stats(
    trajectory_groups: list[dict[str, Any]],
    tracker: stats_tracker.DistributedStatsTracker | None = None,
) -> None:
    """Log per-prompt conditional entropy estimates from grouped rollouts.

    Each item in ``trajectory_groups`` should correspond to one prompt, with the
    batch dimension containing repeated samples for that prompt.
    """
    if tracker is None:
        tracker = stats_tracker.DEFAULT_TRACKER

    true_ce_values = []
    false_ce_values = []
    success_rates = []
    true_valid = []
    false_valid = []
    devices = []

    for data in trajectory_groups:
        rewards = data.get("rewards")
        logprobs = data.get("logprobs")
        loss_mask = data.get("loss_mask")
        if not (
            isinstance(rewards, torch.Tensor)
            and isinstance(logprobs, torch.Tensor)
            and isinstance(loss_mask, torch.Tensor)
        ):
            continue
        if rewards.numel() == 0 or logprobs.ndim < 2 or loss_mask.ndim < 2:
            continue

        rewards = rewards.flatten().float()
        n_samples = rewards.numel()
        if logprobs.shape[0] != n_samples or loss_mask.shape[0] != n_samples:
            continue

        devices.append(logprobs.device)
        sample_mask = loss_mask.bool()
        surprisal = -(logprobs.float() * sample_mask.float()).sum(dim=-1)

        true_mask = rewards > 0
        false_mask = ~true_mask
        true_count = true_mask.float().sum()
        false_count = false_mask.float().sum()
        n_samples_tensor = true_count.new_tensor(float(n_samples))

        true_mean_surprisal = torch.where(
            true_mask, surprisal, 0.0
        ).sum() / true_count.clamp_min(1.0)
        false_mean_surprisal = torch.where(
            false_mask, surprisal, 0.0
        ).sum() / false_count.clamp_min(1.0)

        true_ce = torch.where(
            true_count > 0,
            torch.log(true_count / n_samples_tensor) + true_mean_surprisal,
            torch.zeros_like(true_count),
        )
        false_ce = torch.where(
            false_count > 0,
            torch.log(false_count / n_samples_tensor) + false_mean_surprisal,
            torch.zeros_like(false_count),
        )

        true_ce_values.append(true_ce)
        false_ce_values.append(false_ce)
        success_rates.append(true_count / n_samples_tensor)
        true_valid.append(true_count > 0)
        false_valid.append(false_count > 0)

    if not true_ce_values:
        return

    device = devices[0]
    true_ce_tensor = torch.stack([x.to(device) for x in true_ce_values]).float()
    false_ce_tensor = torch.stack([x.to(device) for x in false_ce_values]).float()
    success_rate_tensor = torch.stack([x.to(device) for x in success_rates]).float()
    prompt_denominator = torch.ones_like(true_ce_tensor, dtype=torch.bool)

    tracker.denominator(
        conditional_entropy_n_prompts=prompt_denominator,
        true_conditional_entropy_valid_prompts=torch.stack(
            [x.to(device) for x in true_valid]
        ).bool(),
        false_conditional_entropy_valid_prompts=torch.stack(
            [x.to(device) for x in false_valid]
        ).bool(),
    )
    tracker.stat(
        true_conditional_entropy=true_ce_tensor,
        false_conditional_entropy=false_ce_tensor,
        conditional_entropy_success_rate=success_rate_tensor,
        denominator="conditional_entropy_n_prompts",
    )
