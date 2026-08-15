"""Train a tiny CPU-cache MoE language model on the sample dataset."""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe_nexus.cache_engine import NumberTokenizer
from examples.model.config import ModelConfig


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 2
    learning_rate: float = 1e-3
    log_interval: int = 10
    checkpoint_path: str = "examples/model/moe_checkpoint.pt"
    log_path: str = "examples/logs/train.log"


class ExpertFFN(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.experts = nn.ModuleList([ExpertFFN(config.hidden_dim) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embedding(token_ids)
        logits = self.gate(hidden)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.config.top_k, dim=-1)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        expert_outputs = torch.zeros_like(hidden)
        for k in range(self.config.top_k):
            expert_idx = topk_indices[:, :, k]
            for e in range(self.config.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    selected = hidden[mask]
                    out = self.experts[e](selected)
                    expert_outputs[mask] += topk_scores[mask, k].unsqueeze(-1) * out

        logits_out = self.output_head(expert_outputs)
        return logits_out, topk_scores


def setup_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def load_dataset(path: str, tokenizer: NumberTokenizer, max_seq_len: int) -> torch.Tensor:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    sequences: list[list[int]] = []
    for line in lines:
        ids = tokenizer.encode(line, add_bos=True, add_eos=True)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        sequences.append(ids)
    max_len = max(len(s) for s in sequences)
    padded = []
    for s in sequences:
        s = s + [tokenizer.pad_token_id] * (max_len - len(s))
        padded.append(s)
    return torch.tensor(padded, dtype=torch.long)


def train(model_config: ModelConfig, train_config: TrainConfig, dataset_path: str) -> None:
    setup_logging(train_config.log_path)
    logger = logging.getLogger("train")

    tokenizer = NumberTokenizer()
    dataset = load_dataset(dataset_path, tokenizer, model_config.max_seq_len)
    logger.info("Loaded dataset with %d sequences, vocab_size=%d", len(dataset), tokenizer.get_vocab_size())

    model = MoELanguageModel(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    for epoch in range(1, train_config.epochs + 1):
        total_loss = 0.0
        for i in range(0, len(dataset) - train_config.batch_size, train_config.batch_size):
            batch = dataset[i : i + train_config.batch_size]
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits, _ = model(inp)
            loss = criterion(logits.reshape(-1, model_config.vocab_size), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i // train_config.batch_size) % train_config.log_interval == 0:
                logger.info(
                    "Epoch %d, batch %d, loss=%.4f", epoch, i // train_config.batch_size, loss.item()
                )

        avg_loss = total_loss / max(1, len(dataset) // train_config.batch_size)
        logger.info("Epoch %d complete, avg_loss=%.4f", epoch, avg_loss)

    os.makedirs(os.path.dirname(train_config.checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.__dict__,
        },
        train_config.checkpoint_path,
    )
    logger.info("Checkpoint saved to %s", train_config.checkpoint_path)


if __name__ == "__main__":
    train(ModelConfig(), TrainConfig(), "examples/dataset/sample.txt")
