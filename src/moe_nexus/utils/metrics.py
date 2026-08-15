from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ExpertUtilizationReport:
    expert_counts: torch.Tensor
    total_tokens: int
    sparsity: float
    entropy: float
    max_utilization: float
    min_utilization: float


def compute_expert_sparsity(
    topk_indices: torch.Tensor,
    num_experts: int,
) -> float:
    unique_experts = torch.unique(topk_indices)
    active_experts = unique_experts.numel()
    return 1.0 - (active_experts / num_experts)


def compute_routing_entropy(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> float:
    B, T, K = topk_scores.shape
    flat_scores = topk_scores.reshape(-1, K)
    flat_indices = topk_indices.reshape(-1, K)

    probs = torch.zeros(flat_scores.shape[0], num_experts, device=flat_scores.device)
    for k in range(K):
        probs.scatter_add_(
            1,
            flat_indices[:, k].unsqueeze(1),
            flat_scores[:, k].unsqueeze(1),
        )

    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
    return entropy.item()
