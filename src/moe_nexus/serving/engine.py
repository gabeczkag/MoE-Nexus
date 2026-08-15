from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

from moe_nexus.cache_engine import CPUCacheDecoder, NumberTokenizer
from moe_nexus.optimizer import LoadBalancer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        decoder: CPUCacheDecoder,
        load_balancer: Optional[LoadBalancer] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.decoder = decoder
        self.load_balancer = load_balancer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        if self.load_balancer is not None:
            self.load_balancer.reset()

        generated = token_ids.clone()
        past_key_values = None

        for _ in range(config.max_new_tokens):
            model_outputs = self.model(
                input_ids=generated[:, -1:] if past_key_values is not None else generated,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = model_outputs.logits[:, -1, :]
            past_key_values = model_outputs.past_key_values

            if config.temperature != 1.0:
                logits = logits / config.temperature

            if config.repetition_penalty != 1.0:
                logits = _apply_repetition_penalty(
                    logits, generated, config.repetition_penalty
                )

            if config.do_sample:
                logits = _top_k_top_p_filtering(
                    logits, top_k=config.top_k, top_p=config.top_p, min_tokens_to_keep=1
                )
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if config.eos_token_id is not None and (next_token == config.eos_token_id).any():
                break

        return generated

    def generate_text(self, token_ids: torch.Tensor, config: GenerationConfig) -> str:
        token_ids = self.generate(token_ids, config)
        return self.decoder.decode(token_ids[0])

    def benchmark(
        self,
        token_ids: torch.Tensor,
        config: GenerationConfig,
        warmup_steps: int = 3,
        measure_steps: int = 10,
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = self.generate(token_ids, config)

            torch.cuda.synchronize() if self.device.type == "cuda" else None
            start = time.perf_counter()

            for _ in range(measure_steps):
                token_output = self.generate(token_ids, config)
                _ = self.decoder.decode_batch(token_output)

            torch.cuda.synchronize() if self.device.type == "cuda" else None
            elapsed = time.perf_counter() - start

        total_tokens = measure_steps * config.max_new_tokens
        return {
            "total_time_s": elapsed,
            "tokens_per_second": total_tokens / elapsed,
            "latency_per_token_ms": (elapsed / total_tokens) * 1000,
        }


def _apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    unique_ids = torch.unique(generated)
    logits[:, unique_ids] = logits[:, unique_ids] / penalty
    return logits


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits
