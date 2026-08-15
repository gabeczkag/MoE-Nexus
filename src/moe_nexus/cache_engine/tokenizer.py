from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class NumberTokenizer:
    vocab_size: int = 256
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    def __post_init__(self) -> None:
        self._char_to_int: Dict[str, int] = {}
        self._int_to_char: Dict[int, str] = {}
        self._setup_default_vocab()

    def _setup_default_vocab(self) -> None:
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self._char_to_int[token] = i
            self._int_to_char[i] = token
        for i in range(256):
            char = chr(i)
            if char not in self._char_to_int:
                idx = len(self._char_to_int)
                self._char_to_int[char] = idx
                self._int_to_char[idx] = char

        self.vocab_size = len(self._char_to_int)
        self.pad_token_id = self._char_to_int[self.pad_token]
        self.unk_token_id = self._char_to_int[self.unk_token]
        self.bos_token_id = self._char_to_int[self.bos_token]
        self.eos_token_id = self._char_to_int[self.eos_token]

        self._lookup_array = np.zeros(max(self._int_to_char.keys()) + 1, dtype=np.uint32)
        for idx, char in self._int_to_char.items():
            if len(char) == 1:
                self._lookup_array[idx] = ord(char)
            else:
                self._lookup_array[idx] = 0

        # Wektoryzowana tablica kodowania: bajt -> token id.
        self._encode_lut = np.array(
            [self._char_to_int.get(chr(b), self.unk_token_id) for b in range(256)],
            dtype=np.int32,
        )

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)
        if text.isascii():
            arr = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
            tokens.extend(self._encode_lut[arr].tolist())
        else:
            for char in text:
                tokens.append(self._char_to_int.get(char, self.unk_token_id))
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def encode_tensor(self, text: str, add_bos: bool = False, add_eos: bool = False) -> torch.Tensor:
        tokens = self.encode(text, add_bos=add_bos, add_eos=add_eos)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: Union[List[int], torch.Tensor, np.ndarray]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().tolist()
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        chars = []
        for token in tokens:
            if token in self._int_to_char:
                char = self._int_to_char[token]
                if char in (self.pad_token, self.unk_token, self.bos_token, self.eos_token):
                    continue
                chars.append(char)
        return "".join(chars)

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None, pad: bool = True) -> torch.Tensor:
        encoded = [self.encode_tensor(text, add_bos=False, add_eos=False) for text in texts]
        if pad and max_length is None:
            max_length = max(t.size(0) for t in encoded)
        elif not pad:
            return torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True)
        padded = []
        for t in encoded:
            if t.size(0) < max_length:
                pad_size = max_length - t.size(0)
                t = torch.cat([t, torch.full((pad_size,), self.pad_token_id, dtype=torch.long)])
            padded.append(t[:max_length])
        return torch.stack(padded)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_lookup_array(self) -> np.ndarray:
        return self._lookup_array
