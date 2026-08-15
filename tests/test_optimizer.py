from __future__ import annotations

import pytest
import torch

from moe_nexus.optimizer import (
    BalanceResult,
    ExpertChoice,
    ExpertStats,
    LoadBalancer,
    RouterConfig,
    TopKRouter,
)
from moe_nexus.utils import compute_expert_sparsity, compute_routing_entropy


class TestTopKRouter:
    @pytest.fixture
    def router(self) -> TopKRouter:
        config = RouterConfig(num_experts=8, top_k=2, hidden_dim=32)
        return TopKRouter(config)

    def test_forward_shape(self, router: TopKRouter) -> None:
        B, T, D = 2, 16, 32
        hidden = torch.randn(B, T, D)
        scores, indices, aux = router(hidden)
        assert scores.shape == (B, T, 2)
        assert indices.shape == (B, T, 2)
        assert aux is None or isinstance(aux, torch.Tensor)

    def test_top_k_selection(self, router: TopKRouter) -> None:
        B, T, D = 1, 4, 32
        hidden = torch.randn(B, T, D)
        scores, indices, _ = router(hidden)
        assert torch.allclose(scores.sum(dim=-1), torch.ones(B, T), atol=1e-5)
        assert indices.shape == (B, T, router.top_k)


class TestExpertChoice:
    def test_forward_shape(self) -> None:
        config = RouterConfig(num_experts=4, top_k=2, hidden_dim=16)
        router = ExpertChoice(config)
        hidden = torch.randn(2, 8, 16)
        scores, indices, aux = router(hidden)
        assert scores.shape == (2, 8, 2)
        assert aux is None


class TestLoadBalancer:
    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        return LoadBalancer(num_experts=4)

    def test_record_and_analyze(self, balancer: LoadBalancer) -> None:
        indices = torch.tensor([[[0, 1], [2, 3]], [[1, 1], [0, 2]]])
        scores = torch.ones_like(indices, dtype=torch.float32) / 2
        balancer.record_routing(indices, scores)
        result = balancer.analyze()
        assert isinstance(result, BalanceResult)
        assert result.max_utilization >= result.min_utilization

    def test_reset(self, balancer: LoadBalancer) -> None:
        indices = torch.tensor([[[0, 1]]])
        scores = torch.ones_like(indices, dtype=torch.float32) / 2
        balancer.record_routing(indices, scores)
        balancer.reset()
        result = balancer.analyze()
        assert result.max_utilization == 0.0


class TestMetrics:
    def test_expert_sparsity(self) -> None:
        indices = torch.tensor([[[0, 1], [2, 3]]])
        sparsity = compute_expert_sparsity(indices, num_experts=8)
        assert 0.0 <= sparsity <= 1.0

    def test_routing_entropy(self) -> None:
        scores = torch.ones(2, 4, 2) / 2
        indices = torch.tensor([[[0, 1]] * 4] * 2)
        entropy = compute_routing_entropy(scores, indices, num_experts=4)
        assert entropy > 0.0
