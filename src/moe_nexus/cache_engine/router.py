from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CacheMoERouter(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        self.char_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        nn.init.normal_(self.char_embedding.weight, std=0.02)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.char_embedding(token_ids)
        logits = self.gate(hidden)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        aux_loss = None
        if self.training:
            aux_loss = _router_z_loss(logits) * 0.01

        return topk_scores, topk_indices, aux_loss


def _router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.reshape(-1, logits.shape[-1])
    logits_max, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - logits_max
    logits = logits / logits.shape[-1] ** 0.5
    return torch.sum(logits ** 2) / logits.numel()
