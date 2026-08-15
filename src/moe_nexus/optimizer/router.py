from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterConfig:
    num_experts: int
    top_k: int = 2
    noise_std: float = 0.1
    use_aux_loss: bool = True
    aux_loss_weight: float = 0.01
    hidden_dim: Optional[int] = None


class BaseRouter(nn.Module):
    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


class TopKRouter(BaseRouter):
    def __init__(self, config: RouterConfig) -> None:
        super().__init__(config)
        if config.hidden_dim is None:
            raise ValueError(
                "TopKRouter wymaga podania `hidden_dim` w RouterConfig. "
                "Podaj szerokość wejścia ukrytego stanu, np. hidden_dim=32."
            )
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        self.noise_std = config.noise_std

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        logits = self.gate(hidden_states)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        if self.training and self.config.use_aux_loss:
            aux_loss = _router_z_loss(logits) * self.config.aux_loss_weight
        else:
            aux_loss = None

        return topk_scores, topk_indices, aux_loss


class ExpertChoice(BaseRouter):
    def __init__(self, config: RouterConfig) -> None:
        super().__init__(config)
        if config.hidden_dim is None:
            raise ValueError(
                "ExpertChoice wymaga podania `hidden_dim` w RouterConfig. "
                "Podaj szerokość wejścia ukrytego stanu, np. hidden_dim=16."
            )
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = hidden_states.shape
        logits = self.gate(hidden_states)
        scores = F.softmax(logits, dim=-1)

        flat_scores = scores.view(B * T, self.num_experts)
        topk_scores, topk_indices = torch.topk(flat_scores, self.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        topk_scores = topk_scores.view(B, T, self.top_k)
        topk_indices = topk_indices.view(B, T, self.top_k)

        return topk_scores, topk_indices, None


Router = TopKRouter


def _router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.reshape(-1, logits.shape[-1])
    logits_max, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - logits_max
    logits = logits / logits.shape[-1] ** 0.5
    return torch.sum(logits ** 2) / logits.numel()
