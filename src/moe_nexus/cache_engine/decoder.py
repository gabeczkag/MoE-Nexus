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
        arr = self._to_int_array(tokens)
        codepoints = np.take(self._lookup, arr)
        valid = codepoints[codepoints != 0]
        if valid.size == 0:
            return ""
        return valid.astype(np.uint8, copy=False).tobytes().decode("utf-8", errors="ignore")

    def decode_batch(self, token_batch: Union[torch.Tensor, np.ndarray]) -> List[str]:
        if isinstance(token_batch, torch.Tensor):
            token_batch = token_batch.cpu().numpy()
        codepoints = np.take(self._lookup, np.asarray(token_batch, dtype=np.int32))
        rows = codepoints[codepoints != 0]
        # Rozdziel wyniki per wiersz bez pętli Pythona po znakach:
        # 'rows' to spłaszczony ciąg ważnych kodepunktów całego batcha,
        # dzielimy go na oryginalne wiersze za pomocą znaczników długości.
        lengths = (codepoints != 0).sum(axis=1)
        out: List[str] = []
        start = 0
        for length in lengths:
            seg = rows[start : start + int(length)]
            start += int(length)
            out.append(seg.astype(np.uint8, copy=False).tobytes().decode("utf-8", errors="ignore") if seg.size else "")
        return out

    def decode_stream(self, token_stream: List[int]) -> str:
        return self.decode(token_stream)

    @staticmethod
    def _to_int_array(tokens) -> np.ndarray:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        return np.asarray(tokens, dtype=np.int32)
