"""
Example: CPU Cache-based MoE inference with number tokenization.

Demonstrates the core idea:
1. Tokenize text into integer token IDs
2. Run MoE router on token embeddings
3. Decode output token IDs back to text using CPU cache-friendly lookup
"""

import torch

from moe_nexus.cache_engine import CPUCacheDecoder, NumberTokenizer
from moe_nexus.optimizer import LoadBalancer, RouterConfig, TopKRouter
from moe_nexus.serving import GenerationConfig, InferenceEngine


def main() -> None:
    tokenizer = NumberTokenizer()
    decoder = CPUCacheDecoder(tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    config = RouterConfig(
        num_experts=8,
        top_k=2,
        hidden_dim=64,
        noise_std=0.1,
        use_aux_loss=False,
    )
    router = TopKRouter(config)
    balancer = LoadBalancer(num_experts=8)

    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, 64),
        torch.nn.Linear(64, vocab_size),
    )

    engine = InferenceEngine(
        model=model,
        decoder=decoder,
        load_balancer=balancer,
        device="cpu",
    )

    text = "hello world"
    input_ids = tokenizer.encode_tensor(text, add_bos=True).unsqueeze(0)

    gen_config = GenerationConfig(
        max_new_tokens=32,
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = engine.generate(input_ids, config=gen_config)
    output_text = decoder.decode(output_ids[0])

    print(f"Input : {text}")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
