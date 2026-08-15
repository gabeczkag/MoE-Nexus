"""Quick benchmark: Dense vs Vanilla MoE vs MoE-Nexus (CPU-cache MoE) on 8k tokens."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from moe_nexus.cache_engine import CPUCacheDecoder, NumberTokenizer
from moe_nexus.optimizer import LoadBalancer, RouterConfig, TopKRouter
from moe_nexus.serving import GenerationConfig, InferenceEngine
from examples.train.train_all import DenseModel, VanillaMoEModel, MoENexusModel
from examples.model.config import ModelConfig

# Try to import C++ backend
CPP_AVAILABLE = False
try:
    import moe_nexus_core as cpp_core
    CPP_AVAILABLE = True
except ImportError:
    pass


class ModelAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.stateless = getattr(model, "stateless", False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ) -> "ModelOutput":
        output = self.model(input_ids)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        return ModelOutput(logits=logits, past_key_values=None)


class ModelOutput:
    def __init__(self, logits: torch.Tensor, past_key_values) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class StandardDecoder:
    def __init__(self, tokenizer: NumberTokenizer) -> None:
        self.tokenizer = tokenizer

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.tolist())


def format_tokens_per_second(tokens: int, seconds: float) -> str:
    if seconds <= 0:
        return "inf"
    tps = tokens / seconds
    if tps >= 1_000_000:
        return f"{tps/1_000_000:.2f}M tok/s"
    if tps >= 1000:
        return f"{tps/1000:.2f}k tok/s"
    return f"{tps:.1f} tok/s"


def timed_generate(engine: InferenceEngine, input_ids: torch.Tensor, gen_config: GenerationConfig, reps: int = 10):
    engine.generate(input_ids, gen_config)  # rozgrzewka (cold-start poza pomiarem)
    times = []
    last = None
    for _ in range(reps):
        t = time.perf_counter()
        out = engine.generate(input_ids, gen_config)
        times.append(time.perf_counter() - t)
        last = out
    return sum(times) / len(times), last


def main() -> None:
    tokenizer = NumberTokenizer()
    vocab_size = tokenizer.get_vocab_size()

    hidden_dim = 64
    num_experts = 8
    top_k = 2
    max_new_tokens = 64
    max_steps = 1000

    prompts = ["what is hello", "mixture", "cpu", "token"]
    input_ids = tokenizer.batch_encode(prompts, max_length=16, pad=True)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.9,
        do_sample=False,
        eos_token_id=None,  # Zablokowanie wczesnego wychodzenia
        max_steps=max_steps,
    )

    models_dir = ROOT / "examples/model"
    dense_path = models_dir / "dense.pt"
    vanilla_path = models_dir / "vanilla_moe.pt"
    nexus_path = models_dir / "moe_nexus.pt"

    # Inicjalizacja modeli (hidden_dim musi zgadzać się z wytrenowanymi checkpointami .pt)
    dense_model = DenseModel(vocab_size, hidden_dim)
    if dense_path.exists():
        try:
            ckpt = torch.load(dense_path, map_location="cpu", weights_only=False)
            dense_model.load_state_dict(ckpt["model_state_dict"])
        except Exception:
            print("[INFO] Brak spójnego punktu kontrolnego dla Dense, używam zainicjalizowanych wag.")
    dense_model.eval()

    vanilla_model = VanillaMoEModel(vocab_size, hidden_dim, num_experts, top_k)
    if vanilla_path.exists():
        try:
            ckpt = torch.load(vanilla_path, map_location="cpu", weights_only=False)
            vanilla_model.load_state_dict(ckpt["model_state_dict"])
        except Exception:
            print("[INFO] Brak spójnego punktu kontrolnego dla Vanilla MoE, używam zainicjalizowanych wag.")
    vanilla_model.eval()

    nexus_model = MoENexusModel(vocab_size, hidden_dim, num_experts, top_k)
    if nexus_path.exists():
        try:
            ckpt = torch.load(nexus_path, map_location="cpu", weights_only=False)
            nexus_model.load_state_dict(ckpt["model_state_dict"])
        except Exception:
            print("[INFO] Brak spójnego punktu kontrolnego dla MoE-Nexus, używam zainicjalizowanych wag.")
    nexus_model.eval()

    std_decoder = StandardDecoder(tokenizer)
    cache_decoder = CPUCacheDecoder(tokenizer)

    results = []
    total_tokens = len(prompts) * max_new_tokens

    if CPP_AVAILABLE:
        print("[INFO] C++ backend available, running C++ benchmarks...")

        cpp_tokenizer = cpp_core.NumberTokenizer()
        cpp_model_config = cpp_core.ModelConfig()
        cpp_model_config.vocab_size = vocab_size
        cpp_model_config.hidden_dim = hidden_dim
        cpp_model_config.num_experts = num_experts
        cpp_model_config.top_k = top_k

        cpp_model = cpp_core.MoEModel(cpp_model_config)
        cpp_engine = cpp_core.InferenceEngine(
            cpp_model,
            cpp_tokenizer,
            cpp_core.LoadBalancer(num_experts)
        )

        cpp_gen_config = cpp_core.GenerationConfig()
        cpp_gen_config.max_new_tokens = max_new_tokens
        cpp_gen_config.eos_token_id = -1  # Zablokowanie EOS w C++

        cpp_texts = []
        start = time.perf_counter()
        for i in range(len(prompts)):
            input_list = input_ids[i].tolist()
            cpp_output = cpp_engine.generate(input_list, cpp_gen_config)
            cpp_texts.append(tokenizer.decode(cpp_output))
        cpp_time = time.perf_counter() - start

        results.append(("MoE-Nexus C++", cpp_time, cpp_texts, total_tokens))
        print(f"[C++] MoE-Nexus C++: {cpp_time:.3f}s  ({format_tokens_per_second(total_tokens, cpp_time)})")

    # Python benchmarks (uwredniowane po N przebiegach, by zniwelować szum)
    reps = 10
    # 1. Dense + StandardDecoder
    engine = InferenceEngine(model=ModelAdapter(dense_model), decoder=std_decoder, device="cpu")
    dense_time, dense_ids = timed_generate(engine, input_ids, gen_config, reps=reps)
    dense_texts = [std_decoder.decode(dense_ids[i]) for i in range(len(prompts))]
    dense_tokens = (dense_ids.shape[1] - input_ids.shape[1]) * len(prompts)
    results.append(("Dense + StandardDecoder", dense_time, dense_texts, dense_tokens))

    # 2. Vanilla MoE + StandardDecoder
    engine = InferenceEngine(model=ModelAdapter(vanilla_model), decoder=std_decoder, device="cpu")
    vanilla_time, vanilla_ids = timed_generate(engine, input_ids, gen_config, reps=reps)
    vanilla_texts = [std_decoder.decode(vanilla_ids[i]) for i in range(len(prompts))]
    vanilla_tokens = (vanilla_ids.shape[1] - input_ids.shape[1]) * len(prompts)
    results.append(("Vanilla MoE + StandardDecoder", vanilla_time, vanilla_texts, vanilla_tokens))

    # 3. MoE-Nexus + CPUCacheDecoder
    engine = InferenceEngine(model=ModelAdapter(nexus_model), decoder=cache_decoder, device="cpu")
    nexus_time, nexus_ids = timed_generate(engine, input_ids, gen_config, reps=reps)
    nexus_texts = [cache_decoder.decode(nexus_ids[i]) for i in range(len(prompts))]
    nexus_tokens = (nexus_ids.shape[1] - input_ids.shape[1]) * len(prompts)
    results.append(("MoE-Nexus + CPUCacheDecoder", nexus_time, nexus_texts, nexus_tokens))

    # Uczciwy pomiar kodera i dekodera (koszty poboczne vs forward pass)
    t0 = time.perf_counter()
    for _ in range(200):
        _ = tokenizer.batch_encode(prompts, max_length=16, pad=True)
    encode_per_call = (time.perf_counter() - t0) / 200

    t0 = time.perf_counter()
    for _ in range(200):
        _ = cache_decoder.decode_batch(nexus_ids)
    decode_per_call = (time.perf_counter() - t0) / 200

    print("=" * 70)
    print("CHAT BENCHMARK - MoE-Nexus (Real Evaluation)")
    print("=" * 70)
    print(f"Prompts ({len(prompts)}): {prompts}")
    print(f"Max new tokens per prompt: {max_new_tokens}")
    print(f"Total generated tokens: {total_tokens}")
    print("-" * 70)
    for name, elapsed, texts, tokens in results:
        print(f"{name:35s}: {elapsed:.3f}s  ({format_tokens_per_second(tokens, elapsed)})")
    print("-" * 70)
    print("Sample outputs:")
    for i, prompt in enumerate(prompts):
        print(f"  '{prompt}' -> '{results[-1][2][i][:50]}'")
    print("-" * 70)
    print(f"Koder  (batch_encode x1): {encode_per_call*1000:.3f} ms")
    print(f"Dekoder (decode_batch x1): {decode_per_call*1000:.3f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
