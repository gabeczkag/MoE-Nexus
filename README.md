# MoE-Nexus

<p align="center">
  <img src="docs/assets/architecture.svg" alt="MoE-Nexus Architecture" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/gabeczkag/MoE-Nexus/actions"><img src="https://img.shields.io/github/actions/workflow/status/gabeczkag/MoE-Nexus/ci.yml?branch=main" alt="CI"></a>
  <a href="https://github.com/gabeczkag/MoE-Nexus/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gabeczkag/MoE-Nexus.svg" alt="License: GPL-3.0"></a>
  <img src="https://img.shields.io/badge/status-frozen-success.svg" alt="Status: Frozen">
</p>

**MoE-Nexus** to silnik inference dla architektury **Mixture of Experts (MoE)**, w którym
tokenizacja i dekodowanie odbywa się w dziedzinie liczb (znak → `token_id` → znak),
a routing i eksperci są zaimplementowani w PyTorchu.

Główne cechy:
- **NumberTokenizer** — wektoryzowane kodowanie znaków (`np.frombuffer` + tablica lookup).
- **CPUCacheDecoder** — dekodowanie `token_id → znak` przez `np.take` na prebuildowanej
  lookup table (jeden `tobytes().decode()` zamiast pętli Pythona).
- **Zoptymalizowany model MoE** — wektoryzowany combine ekspertów (`torch.gather` + ważona
  suma zamiast podwójnej pętli po ekspertach) oraz **inkrementalne dekodowanie** dla modeli
  bezstanowych (każdy krok generuje tylko ostatni token, bez przeliczania całej sekwencji).
- **Brak kwantyzacji** — optymalizacja dotyczy kodera, dekodera i samego modelu, nie redukcji precyzji.

Licencja: **GNU GPL v3**.

> Projekt MoE uznany za zakończony. Następny krok to wariant **Dense** (osobny projekt).

## Architektura

```
Text Input
    │
    ▼
NumberTokenizer  ──► encode: text -> List[int] / torch.Tensor   (np.frombuffer + LUT)
    │
    ▼
MoE Model (Top-K router + expert FFNs, wektoryzowany combine)
    │
    ▼
Token IDs output
    │
    ▼
CPUCacheDecoder  ──► decode: token_ids -> text  (np.take + LUT, jeden decode())
    │
    ▼
Text Output
```

### Wersje implementacji
- **Python** — `src/moe_nexus/` (PyTorch, trening + inference)
- **C++** — `cpp/` (wydajna warstwa inference, pybind11 bindings)

## Funkcjonalności

### 🧠 Routery (`src/moe_nexus/optimizer/`)
- `TopKRouter` — top-k routing z opcjonalnym szumem treningowym
- `ExpertChoice` — token-level expert selection
- Auxiliary Z-loss — regularyzacja przeciw kolapsowi ekspertów

### 🔢 CPU Cache Tokenization (`src/moe_nexus/cache_engine/`)
- `NumberTokenizer` — znak → liczba (0..vocab), wektoryzowane `encode`
- `CPUCacheDecoder` — `token_id → znak` przez `np.take` + lookup table
- `CacheMoERouter` — router z embeddingiem znaków

### ⚖️ Load Balancing
- `LoadBalancer` — statystyki wykorzystania ekspertów
- `compute_expert_sparsity`, `compute_routing_entropy` — metryki routingu

### 🚀 Serving (`src/moe_nexus/serving/`)
- `InferenceEngine` — generacja z inkrementalnym dekodowaniem (modele bezstanowe),
  temperature, top-p, repetition penalty
- `benchmark()` — pomiar TPS i latencji

### ⚡ C++ Core (`cpp/`)
- Tokenizer, router, model i engine w C++17
- `pybind11` bindings (`moe_nexus_core`)
- Samodzielny benchmark i testy

## Instalacja

### Python
```bash
git clone https://github.com/gabeczkag/MoE-Nexus.git
cd MoE-Nexus
pip install -e ".[dev]"
```

Wymagania: Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24.

### C++ (opcjonalnie)
```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Wymagania: CMake ≥ 3.20, kompilator C++17, opcjonalnie pybind11.

## Szybki start

```python
import torch
from moe_nexus.cache_engine import CPUCacheDecoder, NumberTokenizer
from moe_nexus.optimizer import LoadBalancer, RouterConfig, TopKRouter
from moe_nexus.serving import GenerationConfig, InferenceEngine

tokenizer = NumberTokenizer()
decoder = CPUCacheDecoder(tokenizer)

config = RouterConfig(num_experts=8, top_k=2, hidden_dim=64, noise_std=0.1, use_aux_loss=False)
router = TopKRouter(config)
balancer = LoadBalancer(num_experts=8)

vocab_size = tokenizer.get_vocab_size()
model = torch.nn.Sequential(
    torch.nn.Embedding(vocab_size, 64),
    torch.nn.Linear(64, vocab_size),
)

engine = InferenceEngine(model=model, decoder=decoder, load_balancer=balancer, device="cpu")

input_ids = tokenizer.encode_tensor("hello", add_bos=True).unsqueeze(0)
gen_config = GenerationConfig(max_new_tokens=32, temperature=1.0, top_p=0.9, do_sample=True,
                              eos_token_id=tokenizer.eos_token_id)
print(engine.generate_text(input_ids, config=gen_config))
```

## Benchmark — Normal MoE vs MoE-Nexus

Uruchomienie: `python examples/benchmark.py` (4 prompty, 64 tokenów każdy, batch=4,
CPU, model 64-dim, 8 ekspertów, top_k=2). Wyniki uśrednione z 10 przebiegów.

### Wykres przepustowości (tok/s)

```
Model                 tok/s      ┃
─────────────────────────────────┻──────────────────────────────────────
Dense (referencja)   25304 tok/s ┃ ████████████████████████████████████████
Normal MoE (Vanilla)  6388 tok/s ┃ ████████
MoE-Nexus             5558 tok/s ┃ ███████
```

Skala: `█` ≈ 800 tok/s.

### Wniosek (uczciwie)

- **Normal MoE i MoE-Nexus są w granicach szumu równo szybkie** (~6k vs ~5.5k tok/s).
  MoE-Nexus różni się od Normal MoE jedynie dekoderem (`CPUCacheDecoder` zamiast
  `StandardDecoder`). Dekoder uruchamia się **raz**, po wygenerowaniu całej sekwencji,
  więc nie wpływa na przepustowość generacji — stąd brak przyspieszenia "Magia CPU cache".
- **Dense jest ~4× szybszy od MoE**, bo ma najmniej rachunków na token (brak routera
  i pętli ekspertów). W tym zabawkowym setupie (8 ekspertów, 64-dim) MoE to czysty
  overhead względem Dense — MoE opłaca się przy dziesiątkach/setkach ekspertów i dużych
  wymiarach.
- **Optymalizacja modelu przyspieszyła MoE ~3.5–4×** (wektoryzacja combine ekspertów
  + inkrementalne dekodowanie bezstanowe). To zysk wspólny dla obu wariantów MoE.

### Zysk z optymalizacji (przed → po)

```
Wariant MoE            przed     po       przyspieszenie
────────────────────────────────────────────────────────
Normal MoE (Vanilla)  ~1530     ~6388     ~4.2x
MoE-Nexus             ~1791     ~5558     ~3.1x
```

Liczby bezwzględne (tok/s) są szumiące przy tak małym modelu (czas generacji 256 tokenów
to ~10–45 ms, narzut Pythona/torcha na krok dominuje). Stabilna jest **relacja** i
**współczynnik przyspieszenia** — te są miarodajne.

## Struktura projektu

```
MoE-Nexus/
├── src/moe_nexus/
│   ├── cache_engine/      # NumberTokenizer, CPUCacheDecoder, CacheMoERouter
│   ├── optimizer/         # TopKRouter, ExpertChoice, LoadBalancer
│   ├── serving/           # InferenceEngine, GenerationConfig
│   └── utils/             # sparsity, entropy, reports
├── cpp/                   # C++17 core + pybind11 bindings
├── tests/                 # testy jednostkowe
├── examples/
│   ├── benchmark.py       # Dense vs Vanilla MoE vs MoE-Nexus
│   ├── cache_moe_demo.py
│   ├── train/             # train_all.py (Dense/Vanilla/MoE-Nexus)
│   └── dataset/sample.txt
├── docs/assets/architecture.svg
├── .github/workflows/ci.yml
├── pyproject.toml
├── LICENSE                # GNU GPL v3
└── README.md
```

## Przykłady

```bash
# Trening (Dense, Vanilla MoE, MoE-Nexus) na lokalnym samplu
python examples/train/train_all.py

# Demo CPU-cache MoE
python examples/cache_moe_demo.py

# Benchmark
python examples/benchmark.py
```

Modele (`.pt`) lądują w `examples/model/` (gitignorowane). Dataset treningowy
(`smollm_corpus.txt`, ~860 MB) jest gitignorowany — pobierz samodzielnie.

## Testy

```bash
pytest tests/ -v
```

## Roadmap (status MoE)

- [x] Core router API (Top-K, ExpertChoice)
- [x] Load balancing + metryki
- [x] NumberTokenizer + CPUCacheDecoder (wektoryzowane)
- [x] Inference engine z inkrementalnym dekodowaniem
- [x] Wektoryzacja combine ekspertów (~3.5×) + inkrementalne dekodowanie (~1.8×)
- [ ] **Następny projekt: wariant Dense** (osobne repo)

## License

Distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
