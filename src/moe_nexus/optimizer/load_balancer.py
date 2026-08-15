from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class ExpertStats:
    expert_id: int
    total_tokens: int = 0
    total_weight: float = 0.0
    capacity_violations: int = 0

    @property
    def avg_weight(self) -> float:
        return self.total_weight / self.total_tokens if self.total_tokens > 0 else 0.0


@dataclass
class BalanceResult:
    imbalance: float
    coefficient_of_variation: float
    max_utilization: float
    min_utilization: float
    suggestions: List[str]


class LoadBalancer:
    def __init__(self, num_experts: int, capacity_factor: float = 1.25) -> None:
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self._stats: Dict[int, ExpertStats] = {
            i: ExpertStats(expert_id=i) for i in range(num_experts)
        }

    def record_routing(
        self,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
    ) -> None:
        flat_indices = topk_indices.reshape(-1).tolist()
        flat_scores = topk_scores.reshape(-1).tolist()

        for idx, score in zip(flat_indices, flat_scores):
            if idx >= 0 and idx < self.num_experts:
                stat = self._stats[idx]
                stat.total_tokens += 1
                stat.total_weight += score

    def analyze(self) -> BalanceResult:
        utilizations = []
        for stat in self._stats.values():
            util = stat.total_tokens
            utilizations.append(util)

        if not utilizations:
            return BalanceResult(0.0, 0.0, 0.0, 0.0, [])

        util_tensor = torch.tensor(utilizations, dtype=torch.float32)
        mean_util = util_tensor.mean().item()
        std_util = util_tensor.std().item()
        cv = std_util / mean_util if mean_util > 0 else 0.0
        max_util = util_tensor.max().item()
        min_util = util_tensor.min().item()
        imbalance = max_util - min_util

        suggestions = []
        if cv > 0.5:
            suggestions.append("High expert imbalance detected. Consider increasing noise during training.")
        if max_util > mean_util * self.capacity_factor:
            suggestions.append("Capacity violations detected. Review expert capacity limits.")
        if min_util < mean_util * 0.1:
            suggestions.append("Some experts are underutilized. Consider expert pruning or merging.")

        return BalanceResult(
            imbalance=imbalance,
            coefficient_of_variation=cv,
            max_utilization=max_util,
            min_utilization=min_util,
            suggestions=suggestions,
        )

    def reset(self) -> None:
        for stat in self._stats.values():
            stat.total_tokens = 0
            stat.total_weight = 0.0
            stat.capacity_violations = 0
