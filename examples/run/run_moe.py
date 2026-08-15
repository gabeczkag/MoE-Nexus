"""Run inference with the trained CPU-cache MoE model."""

import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe_nexus.cache_engine import CPUCacheDecoder, NumberTokenizer
from examples.model.config import ModelConfig
from examples.train.train_moe import MoELanguageModel


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


def run(model_config: ModelConfig, checkpoint_path: str, prompt: str, max_new_tokens: int = 32) -> None:
    log_path = "examples/logs/run.log"
    setup_logging(log_path)
    logger = logging.getLogger("run")

    tokenizer = NumberTokenizer()
    decoder = CPUCacheDecoder(tokenizer)
    input_ids = tokenizer.encode_tensor(prompt, add_bos=True).unsqueeze(0)
    logger.info("Prompt: %s", prompt)
    logger.info("Input token ids: %s", input_ids[0].tolist())

    model = MoELanguageModel(model_config)
    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint not found: %s. Run train_moe.py first.", checkpoint_path)
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint_path)

    generated = input_ids.clone()
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits, topk_scores = model(generated[:, -1:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            logger.info("Step %d: token=%d, topk_scores=%s", step + 1, next_token.item(), topk_scores[0, 0].tolist())

    output_text = decoder.decode(generated[0])
    logger.info("Generated text: %s", output_text)
    print(f"\nPrompt : {prompt}")
    print(f"Output : {output_text}")


if __name__ == "__main__":
    run(ModelConfig(), "examples/model/moe_checkpoint.pt", "hello")
