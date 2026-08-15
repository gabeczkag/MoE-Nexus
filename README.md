# MoE-Nexus

<p align="center">
  <img src="docs/assets/architecture.svg" alt="MoE-Nexus Architecture" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/gabeczkag/MoE-Nexus/actions"><img src="https://img.shields.io/github/actions/workflow/status/gabeczkag/MoE-Nexus/ci.yml?branch=master" alt="CI"></a>
  <a href="https://pypi.org/project/moe-nexus/"><img src="https://img.shields.io/pypi/pyversions/moe-nexus.svg" alt="Python"></a>
  <a href="https://pypi.org/project/moe-nexus/"><img src="https://img.shields.io/pypi/v/moe-nexus.svg" alt="PyPI"></a>
  <a href="https://github.com/gabeczkag/MoE-Nexus/blob/master/LICENSE"><img src="https://img.shields.io/github/license/gabeczkag/MoE-Nexus.svg" alt="License: GPL-3.0"></a>
  <img src="https://img.shields.io/badge/status-alpha-yellow.svg" alt="Status: Alpha">
</p>

Framework do **optymalizacji i serwowania architektury Mixture of Experts (MoE)** w produkcji. Zawiera moduły do routingu ekspertów (Top-K, ExpertChoice), load balancing, monitorowania wykorzystania ekspertów oraz warstwę inference z benchmarkiem.

Wykorzystuje **GNU GPL v3**.

## Czym jest MoE?

Mixture of Experts to architektura neuronowa, w której zamiast jednej monolitycznej sieci używamy wielu specjalizowanych podsieci (ekspertów). **Router (brama)** dynamicznie wybiera tylko najbardziej odpowiednich ekspertów dla danego tokenu, co pozwala na skalowanie modeli do bilionów parametrów przy zachowaniu wydajności inference.

## Architektura

```
Input Tokens [B, T, D]
      │
      ▼
  Router (Gate) ──► Top-K / ExpertChoice routing
      │
      ▼
  Expert Pool (N ekspertów FFN)
  E₀  E₁  E₂  E₃  E₄  ...
      │
      ▼
  Weighted Aggregation: Σ scoreᵢ · expertᵢ(x)
      │
      ▼
  Output Hidden States [B, T, D]
      │
      ▼
  Load Balancer + Metrics
```

## Funkcjonalności

### 🧠 Routery
- **TopKRouter** — standardowe top-k routing z szumem treningowym
- **ExpertChoice** — token-level expert selection dla lepszego load balancing
- **Auxiliary Z-loss** — regularyzacja zapobiegająca kolapsowi ekspertów

### ⚖️ Load Balancing
- `LoadBalancer` — śledzi statystyki wykorzystania ekspertów
- `BalanceResult` — raport z analizą nierównowagi i sugestiami
- Wykrywa: capacity violations, underutilization, wysokie CV

### 📊 Monitoring
- `compute_expert_sparsity` — mierzy stosunek nieużywanych ekspertów
- `compute_routing_entropy` — entropia rozkładu routingu
- `ExpertUtilizationReport` — pełny raport eksploatacyjny

### 🚀 Serving
- `InferenceEngine` — warstwa generacji z cache KV, temperature, top-p
- `benchmark()` — pomiar TPS i latencji

## Instalacja

```bash
git clone https://github.com/gabeczkag/MoE-Nexus.git
cd MoE-Nexus
pip install -e ".[dev]"
```

Wymagania:
- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24

## Szybki start

```python
import torch
from moe_nexus.optimizer import TopKRouter, RouterConfig, LoadBalancer
from moe_nexus.serving import InferenceEngine, GenerationConfig
from moe_nexus.utils import compute_expert_sparsity

# Konfiguracja routera
config = RouterConfig(
    num_experts=8,
    top_k=2,
    hidden_dim=512,
    noise_std=0.1,
    use_aux_loss=True,
)

# Inicjalizacja
router = TopKRouter(config)
balancer = LoadBalancer(num_experts=8)

# Przykładowe dane: [batch, seq_len, hidden_dim]
hidden_states = torch.randn(4, 64, 512)

# Forward pass
scores, indices, aux_loss = router(hidden_states)

# Load balancing
balancer.record_routing(indices, scores)
report = balancer.analyze()
print(f"Imbalance: {report.imbalance:.2f}")
print(f"CV: {report.coefficient_of_variation:.3f}")
for suggestion in report.suggestions:
    print(f"  • {suggestion}")

# Metryki
sparsity = compute_expert_sparsity(indices, num_experts=8)
print(f"Expert sparsity: {sparsity:.1%}")
```

## API Reference

### Routery

```python
from moe_nexus.optimizer import RouterConfig, TopKRouter, ExpertChoice

config = RouterConfig(
    num_experts=8,
    top_k=2,
    hidden_dim=512,
    noise_std=0.1,
    use_aux_loss=True,
    aux_loss_weight=0.01,
)

# Top-K routing
router = TopKRouter(config)
scores, indices, aux_loss = router(hidden_states)

# ExpertChoice routing
router_ec = ExpertChoice(config)
scores, indices, _ = router_ec(hidden_states)
```

### Load Balancer

```python
from moe_nexus.optimizer import LoadBalancer

balancer = LoadBalancer(num_experts=8, capacity_factor=1.25)

# Po każdej iteracji forward
balancer.record_routing(indices, scores)
result = balancer.analyze()
print(result.coefficient_of_variation)
print(result.suggestions)

# Reset na początku nowej epoki
balancer.reset()
```

### Inference Engine

```python
from moe_nexus.serving import InferenceEngine, GenerationConfig

engine = InferenceEngine(
    model=my_model,
    router=router,
    load_balancer=balancer,
    device="cuda",
)

config = GenerationConfig(
    max_new_tokens=256,
    temperature=1.0,
    top_p=0.9,
    repetition_penalty=1.1,
)

output = engine.generate(input_ids, config)

# Benchmark
stats = engine.benchmark(input_ids, config, warmup_steps=3, measure_steps=10)
print(f"TPS: {stats['tokens_per_second']:.1f}")
print(f"Latency/token: {stats['latency_per_token_ms']:.2f} ms")
```

### Metryki

```python
from moe_nexus.utils import compute_expert_sparsity, compute_routing_entropy

sparsity = compute_expert_sparsity(indices, num_experts=8)
entropy = compute_routing_entropy(scores, num_experts=8)
```

## Benchmarki

| Model | Eksperci | Top-K | Params (total) | Params (active) | TPS (A100) |
|-------|----------|-------|----------------|-----------------|------------|
| MoE-8E | 8 | 2 | 1.2B | 300M | ~12k |
| MoE-16E | 16 | 2 | 2.4B | 600M | ~11k |
| MoE-32E | 32 | 2 | 4.8B | 1.2B | ~10k |

*Wyniki orientacyjne, zależą od implementacji ekspertów i sprzętu.*

## Roadmap

- [x] Core router API (Top-K, ExpertChoice)
- [x] Load balancing + metrics
- [x] Inference engine z KV cache
- [ ] Expert pruning (magnitude, activation-based)
- [ ] MoE quantization (GPTQ, AWQ dla ekspertów)
- [ ] vLLM / TensorRT-LLM integration
- [ ] Distributed serving (expert parallelism)
- [ ] Dynamic expert capacity
- [ ] Mixture-of-Attention (MoA)

## Struktura projektu

```
moe-gabeczkag-edition/
├── src/moe_nexus/
│   ├── __init__.py
│   ├── optimizer/
│   │   ├── router.py          # TopKRouter, ExpertChoice
│   │   └── load_balancer.py   # LoadBalancer, BalanceResult
│   ├── serving/
│   │   └── engine.py          # InferenceEngine, GenerationConfig
│   └── utils/
│       └── metrics.py         # sparsity, entropy, reports
├── tests/
│   ├── test_optimizer.py
│   └── test_serving.py
├── examples/
├── scripts/
├── docs/
│   └── assets/
│       └── architecture.svg
├── pyproject.toml
├── LICENSE                    # GNU GPL v3
└── README.md
```

## Testy

```bash
pytest tests/ -v
```

## Contributing

1. Fork repo
2. Utwórz branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'feat: add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Otwórz Pull Request

## License

Distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for more information.
