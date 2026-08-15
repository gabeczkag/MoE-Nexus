"""Example model config: CPU-cache MoE model hyperparameters."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 260
    hidden_dim: int = 64
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 2
    noise_std: float = 0.1
    max_seq_len: int = 64
