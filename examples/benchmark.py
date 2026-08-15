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


def main() -> None:
    tokenizer = NumberTokenizer()
    vocab_size = tokenizer.get_vocab_size()
    hidden_dim = 64
    num_experts = 8
    top_k = 2
    max_new_tokens = 64
    max_steps = 1000

    prompts = ["hello", "mixture", "cpu", "token"]
    input_ids = tokenizer.batch_encode(prompts, max_length=16, pad=True)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.9,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        max_steps=max_steps,
    )

    # Load trained models
    model_cfg = ModelConfig()
    models_dir = ROOT / "examples/model"

    dense_ckpt = torch.load(models_dir / "dense.pt", map_location="cpu", weights_only=False)
    vanilla_ckpt = torch.load(models_dir / "vanilla_moe.pt", map_location="cpu", weights_only=False)
    nexus_ckpt = torch.load(models_dir / "moe_nexus.pt", map_location="cpu", weights_only=False)

    dense_model = DenseModel(vocab_size, hidden_dim)
    dense_model.load_state_dict(dense_ckpt["model_state_dict"])
    dense_model.eval()

    vanilla_model = VanillaMoEModel(vocab_size, hidden_dim, num_experts, top_k)
    vanilla_model.load_state_dict(vanilla_ckpt["model_state_dict"])
    vanilla_model.eval()

    nexus_model = MoENexusModel(vocab_size, hidden_dim, num_experts, top_k)
    nexus_model.load_state_dict(nexus_ckpt["model_state_dict"])
    nexus_model.eval()

    std_decoder = StandardDecoder(tokenizer)
    cache_decoder = CPUCacheDecoder(tokenizer)

    results = []

    if CPP_AVAILABLE:
        print("[INFO] C++ backend available, running C++ benchmarks...")
        
        # Create C++ models
        cpp_tokenizer = cpp_core.NumberTokenizer()
        cpp_model_config = cpp_core.ModelConfig()
        cpp_model_config.vocab_size = vocab_size
        cpp_model_config.hidden_dim = hidden_dim
        cpp_model_config.num_experts = num_experts
        cpp_model_config.top_k = top_k
        
        # C++ MoE-Nexus
        cpp_model = cpp_core.MoEModel(cpp_model_config)
        cpp_engine = cpp_core.InferenceEngine(
            cpp_model,
            cpp_tokenizer,
            cpp_core.LoadBalancer(num_experts)
        )
        
        cpp_gen_config = cpp_core.GenerationConfig()
        cpp_gen_config.max_new_tokens = max_new_tokens
        cpp_gen_config.eos_token_id = tokenizer.eos_token_id
        
        input_list = input_ids[0].tolist()
        start = time.perf_counter()
        cpp_output = cpp_engine.generate(input_list, cpp_gen_config)
        cpp_time = time.perf_counter() - start
        cpp_text = tokenizer.decode(cpp_output)
        results.append(("MoE-Nexus C++", cpp_time, [cpp_text] * len(prompts)))
        
        print(f"[C++] MoE-Nexus C++: {cpp_time:.3f}s  ({format_tokens_per_second(max_new_tokens, cpp_time)})")
    
    # Python benchmarks
    # 1. Dense + StandardDecoder
    engine = InferenceEngine(model=ModelAdapter(dense_model), decoder=std_decoder, device="cpu")
    start = time.perf_counter()
    dense_ids = engine.generate(input_ids, gen_config)
    dense_time = time.perf_counter() - start
    dense_texts = [std_decoder.decode(dense_ids[i]) for i in range(len(prompts))]
    results.append(("Dense + StandardDecoder", dense_time, dense_texts))

    # 2. Vanilla MoE + StandardDecoder
    engine = InferenceEngine(model=ModelAdapter(vanilla_model), decoder=std_decoder, device="cpu")
    start = time.perf_counter()
    vanilla_ids = engine.generate(input_ids, gen_config)
    vanilla_time = time.perf_counter() - start
    vanilla_texts = [std_decoder.decode(vanilla_ids[i]) for i in range(len(prompts))]
    results.append(("Vanilla MoE + StandardDecoder", vanilla_time, vanilla_texts))

    # 3. MoE-Nexus + CPUCacheDecoder
    engine = InferenceEngine(model=ModelAdapter(nexus_model), decoder=cache_decoder, device="cpu")
    start = time.perf_counter()
    nexus_ids = engine.generate(input_ids, gen_config)
    nexus_time = time.perf_counter() - start
    nexus_texts = [cache_decoder.decode(nexus_ids[i]) for i in range(len(prompts))]
    results.append(("MoE-Nexus + CPUCacheDecoder", nexus_time, nexus_texts))

    total_tokens = len(prompts) * max_new_tokens

    print("=" * 70)
    print("CHAT BENCHMARK - MoE-Nexus (8k tokens)")
    print("=" * 70)
    print(f"Prompts ({len(prompts)}): {prompts}")
    print(f"Max new tokens per prompt: {max_new_tokens}")
    print(f"Total generated tokens: {total_tokens}")
    print("-" * 70)
    for name, elapsed, texts in results:
        print(f"{name:35s}: {elapsed:.3f}s  ({format_tokens_per_second(total_tokens, elapsed)})")
    print("-" * 70)
    print("Sample outputs:")
    for i, prompt in enumerate(prompts):
        print(f"  '{prompt}' -> '{results[0][2][i][:50]}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
