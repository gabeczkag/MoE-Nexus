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
        logits = self.model(input_ids)
        return ModelOutput(logits=logits, past_key_values=None)


class ModelOutput:
    def __init__(self, logits: torch.Tensor, past_key_values) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class DenseModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        return self.head(hidden)


class SimpleMoEModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.router = TopKRouter(
            RouterConfig(num_experts=num_experts, top_k=top_k, hidden_dim=hidden_dim)
        )
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)])
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        scores, indices, _ = self.router(hidden)
        expert_out = torch.zeros_like(hidden)
        for k in range(self.router.top_k):
            for e in range(self.router.num_experts):
                mask = (indices[:, :, k] == e)
                if mask.any():
                    expert_out[mask] += scores[mask, k].unsqueeze(-1) * self.experts[e](hidden[mask])
        return self.head(expert_out)


class StandardDecoder:
    def __init__(self, tokenizer: NumberTokenizer) -> None:
        self.tokenizer = tokenizer

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.tolist())


def main() -> None:
    tokenizer = NumberTokenizer()
    vocab_size = tokenizer.get_vocab_size()
    hidden_dim = 64
    num_experts = 8
    top_k = 2

    text = "hello"
    input_ids = tokenizer.encode_tensor(text, add_bos=True).unsqueeze(0)

    gen_config = GenerationConfig(
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.9,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Dense + standard decode
    dense_model = ModelAdapter(DenseModel(vocab_size, hidden_dim))
    dense_decoder = StandardDecoder(tokenizer)
    dense_engine = InferenceEngine(model=dense_model, decoder=dense_decoder, device="cpu")

    start = time.perf_counter()
    dense_ids = dense_engine.generate(input_ids, gen_config)
    dense_time = time.perf_counter() - start
    dense_text = dense_decoder.decode(dense_ids[0])

    # MoE + standard decode
    moe_model = ModelAdapter(SimpleMoEModel(vocab_size, hidden_dim, num_experts, top_k))
    moe_std_decoder = StandardDecoder(tokenizer)
    moe_std_engine = InferenceEngine(model=moe_model, decoder=moe_std_decoder, device="cpu")

    start = time.perf_counter()
    moe_std_ids = moe_std_engine.generate(input_ids, gen_config)
    moe_std_time = time.perf_counter() - start
    moe_std_text = moe_std_decoder.decode(moe_std_ids[0])

    # MoE + CPUCacheDecoder
    cache_decoder = CPUCacheDecoder(tokenizer)
    moe_cache_engine = InferenceEngine(model=moe_model, decoder=cache_decoder, device="cpu")

    start = time.perf_counter()
    moe_cache_ids = moe_cache_engine.generate(input_ids, gen_config)
    moe_cache_time = time.perf_counter() - start
    moe_cache_text = cache_decoder.decode(moe_cache_ids[0])

    # Decoder-only microbenchmark
    N = 5000
    ids = moe_cache_ids[0]

    start = time.perf_counter()
    for _ in range(N):
        _ = tokenizer.decode(ids.tolist())
    std_decode_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(N):
        _ = cache_decoder.decode(ids)
    cache_decode_time = time.perf_counter() - start

    print("=" * 60)
    print(f"Input: {text}")
    print(f"Dense + standard decode:      {dense_time:.4f}s -> {dense_text}")
    print(f"MoE + standard decode:        {moe_std_time:.4f}s -> {moe_std_text}")
    print(f"MoE + CPUCacheDecoder:        {moe_cache_time:.4f}s -> {moe_cache_text}")
    print("-" * 60)
    print(f"Decoder benchmark ({N} iters):")
    print(f"  Standard decode:            {std_decode_time:.4f}s")
    print(f"  CPUCacheDecoder:            {cache_decode_time:.4f}s")
    if cache_decode_time > 0:
        print(f"  Speedup:                    {std_decode_time / cache_decode_time:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
