from __future__ import annotations

from typing import List, Union

import numpy as np
import torch


class CPUCacheDecoder:
    def __init__(self, tokenizer: "NumberTokenizer", cache_line_size: int = 64) -> None:
        self.tokenizer = tokenizer
        self.cache_line_size = cache_line_size
        self._lookup = tokenizer.get_lookup_array()
        self._vocab_size = len(self._lookup)

    def decode(self, tokens: Union[List[int], torch.Tensor, np.ndarray]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        elif isinstance(tokens, list):
            tokens = np.asarray(tokens, dtype=np.int32)

        tokens = np.clip(tokens, 0, self._vocab_size - 1)
        codepoints = np.take(self._lookup, tokens)
        mask = codepoints != 0
        valid = codepoints[mask]
        if valid.size == 0:
            return ""
        if valid.max() <= 255:
            return valid.astype(np.uint8).tobytes().decode("utf-8", errors="ignore")
        return "".join(chr(int(cp)) for cp in valid)

    def decode_batch(self, token_batch: Union[torch.Tensor, np.ndarray]) -> List[str]:
        if isinstance(token_batch, torch.Tensor):
            token_batch = token_batch.cpu().numpy()

        codepoints = np.take(self._lookup, token_batch)
        results = []
        for row in codepoints:
            mask = row != 0
            valid = row[mask]
            if valid.size == 0:
                results.append("")
            elif valid.max() <= 255:
                results.append(valid.astype(np.uint8).tobytes().decode("utf-8", errors="ignore"))
            else:
                results.append("".join(chr(int(cp)) for cp in valid))
        return results

    def decode_stream(self, token_stream: List[int]) -> str:
        buffer = np.zeros(len(token_stream), dtype=np.int32)
        buffer[:] = token_stream
        return self.decode(buffer)
