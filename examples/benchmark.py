"""Quick benchmark: standard decode vs CPU-cache decode, and Dense vs MoE speed."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_nexus.cache_engine import CPUCacheDecoder, NumberTokenizer
from moe_nexus.optimizer import LoadBalancer, RouterConfig, TopKRouter
from moe_nexus.serving import GenerationConfig, InferenceEngine
from examples.train.train_moe import MoELanguageModel
from examples.model.config import ModelConfig


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

    prompts = ["hello", "mixture", "cpu", "token"]
    input_ids = tokenizer.batch_encode(prompts, max_length=16, pad=True)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.9,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Load trained MoE model if available
    checkpoint_path = ROOT / "examples/model/moe_checkpoint.pt"
    use_trained = checkpoint_path.exists()

    if use_trained:
        model_config = ModelConfig()
        model = MoELanguageModel(model_config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"[INFO] Loaded trained model from {checkpoint_path}")
    else:
        model = MoELanguageModel(ModelConfig())
        model.eval()
        print("[WARN] No trained model found, using random weights")

    # Benchmark 1: MoE + StandardDecoder
    std_decoder = StandardDecoder(tokenizer)
    moe_std_engine = InferenceEngine(
        model=ModelAdapter(model),
        decoder=std_decoder,
        device="cpu",
    )

    start = time.perf_counter()
    moe_std_ids = moe_std_engine.generate(input_ids, gen_config)
    moe_std_time = time.perf_counter() - start
    moe_std_texts = [std_decoder.decode(moe_std_ids[i]) for i in range(len(prompts))]

    # Benchmark 2: MoE + CPUCacheDecoder
    cache_decoder = CPUCacheDecoder(tokenizer)
    moe_cache_engine = InferenceEngine(
        model=ModelAdapter(model),
        decoder=cache_decoder,
        device="cpu",
    )

    start = time.perf_counter()
    moe_cache_ids = moe_cache_engine.generate(input_ids, gen_config)
    moe_cache_time = time.perf_counter() - start
    moe_cache_texts = [cache_decoder.decode(moe_cache_ids[i]) for i in range(len(prompts))]

    total_tokens = len(prompts) * max_new_tokens

    print("=" * 70)
    print("CHAT BENCHMARK - MoE-Nexus")
    print("=" * 70)
    print(f"Prompts ({len(prompts)}): {prompts}")
    print(f"Max new tokens per prompt: {max_new_tokens}")
    print(f"Total generated tokens: {total_tokens}")
    print("-" * 70)
    print(f"MoE + StandardDecoder:  {moe_std_time:.3f}s  ({format_tokens_per_second(total_tokens, moe_std_time)})")
    print(f"MoE + CPUCacheDecoder:  {moe_cache_time:.3f}s  ({format_tokens_per_second(total_tokens, moe_cache_time)})")
    if moe_cache_time > 0:
        print(f"Speedup:                {moe_std_time / moe_cache_time:.2f}x")
    print("-" * 70)
    print("Output comparison:")
    for i, prompt in enumerate(prompts):
        match = "OK" if moe_std_texts[i] == moe_cache_texts[i] else "DIFF"
        print(f"  [{match}] '{prompt}' -> '{moe_std_texts[i][:40]}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
