"""Train Dense, Vanilla MoE and MoE-Nexus models on the same dataset."""

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
from moe_nexus.optimizer import RouterConfig, TopKRouter
from examples.model.config import ModelConfig


def _moe_combine(
    hidden: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    experts: "torch.nn.ModuleList",
) -> torch.Tensor:
    """Wektoryzowane połączenie wyjść ekspertów (gather + ważona suma).

    Zamiast podwójnej pętli Pythona z maskowaniem boolowskim i fancy indexing,
    liczymy wyjścia wszystkich ekspertów naraz i wybieramy je przez torch.gather.
    Wynik jest numerycznie identyczny z wersją scalarną.
    """
    b, t, d = hidden.shape
    bt = b * t
    h = hidden.reshape(bt, d)
    expert_outs = torch.stack([expert(h) for expert in experts], dim=0)  # [E, BT, D]
    idx = indices.reshape(bt, -1)  # [BT, K]
    sc = scores.reshape(bt, -1)  # [BT, K]
    rows = torch.arange(bt, device=h.device)
    out = torch.zeros_like(h)
    for k in range(idx.shape[1]):
        out = out + sc[:, k : k + 1] * expert_outs[idx[:, k], rows]
    return out.reshape(b, t, d)


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-3
    log_interval: int = 10
    checkpoint_dir: str = "examples/model"
    log_dir: str = "examples/logs"


class DenseModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.stateless = True

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        return self.head(hidden)


class VanillaMoEModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.router = TopKRouter(RouterConfig(num_experts=num_experts, top_k=top_k, hidden_dim=hidden_dim))
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)])
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.stateless = True

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        scores, indices, _ = self.router(hidden)
        expert_out = _moe_combine(hidden, scores, indices, self.experts)
        return self.head(expert_out)


class MoENexusModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.router = TopKRouter(RouterConfig(num_experts=num_experts, top_k=top_k, hidden_dim=hidden_dim))
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)])
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.stateless = True

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embedding(token_ids)
        scores, indices, aux_loss = self.router(hidden)
        expert_out = _moe_combine(hidden, scores, indices, self.experts)
        logits_out = self.head(expert_out)
        return logits_out, aux_loss


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


def train_one(model: nn.Module, name: str, dataset: torch.Tensor, tokenizer: NumberTokenizer, train_cfg: TrainConfig, model_cfg: ModelConfig) -> None:
    log_path = os.path.join(train_cfg.log_dir, f"{name}.log")
    setup_logging(log_path)
    logger = logging.getLogger(name)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    for epoch in range(1, train_cfg.epochs + 1):
        total_loss = 0.0
        batches = 0
        for i in range(0, len(dataset) - train_cfg.batch_size, train_cfg.batch_size):
            batch = dataset[i : i + train_cfg.batch_size]
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            out = model(inp)
            if isinstance(out, tuple):
                logits, aux_loss = out
            else:
                logits = out
                aux_loss = None
            loss = criterion(logits.reshape(-1, model_cfg.vocab_size), tgt.reshape(-1))
            if aux_loss is not None:
                loss = loss + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

            if (i // train_cfg.batch_size) % train_cfg.log_interval == 0:
                logger.info("Epoch %d, batch %d, loss=%.4f", epoch, i // train_cfg.batch_size, loss.item())

        avg_loss = total_loss / max(1, batches)
        logger.info("Epoch %d complete, avg_loss=%.4f", epoch, avg_loss)

    ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"{name}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg.__dict__,
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved to %s", ckpt_path)


def main() -> None:
    tokenizer = NumberTokenizer()
    vocab_size = tokenizer.get_vocab_size()
    dataset_path = os.path.join(ROOT, "examples/dataset/sample.txt")
    dataset = load_dataset(dataset_path, tokenizer, max_seq_len=64)

    model_cfg = ModelConfig()
    train_cfg = TrainConfig(epochs=20, batch_size=2, learning_rate=1e-3)

    models = {
        "dense": DenseModel(vocab_size, model_cfg.hidden_dim),
        "vanilla_moe": VanillaMoEModel(vocab_size, model_cfg.hidden_dim, model_cfg.num_experts, model_cfg.top_k),
        "moe_nexus": MoENexusModel(vocab_size, model_cfg.hidden_dim, model_cfg.num_experts, model_cfg.top_k),
    }

    for name, model in models.items():
        print(f"[TRAIN] {name} ...")
        train_one(model, name, dataset, tokenizer, train_cfg, model_cfg)
        print(f"[DONE]  {name}")


if __name__ == "__main__":
    main()
