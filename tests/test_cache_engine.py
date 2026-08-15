from __future__ import annotations

import pytest
import torch

from moe_nexus.cache_engine import CPUCacheDecoder, NumberTokenizer


class TestNumberTokenizer:
    @pytest.fixture
    def tokenizer(self) -> NumberTokenizer:
        return NumberTokenizer()

    def test_encode_decode_roundtrip(self, tokenizer: NumberTokenizer) -> None:
        text = "hello"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_encode_tensor_shape(self, tokenizer: NumberTokenizer) -> None:
        text = "abc"
        tensor = tokenizer.encode_tensor(text)
        assert tensor.shape == (3,)
        assert tensor.dtype == torch.long

    def test_batch_encode(self, tokenizer: NumberTokenizer) -> None:
        texts = ["ab", "cde"]
        batch = tokenizer.batch_encode(texts, max_length=4)
        assert batch.shape == (2, 4)

    def test_special_tokens(self, tokenizer: NumberTokenizer) -> None:
        ids = tokenizer.encode("hi", add_bos=True, add_eos=True)
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id


class TestCPUCacheDecoder:
    @pytest.fixture
    def decoder(self) -> CPUCacheDecoder:
        tokenizer = NumberTokenizer()
        return CPUCacheDecoder(tokenizer)

    def test_decode_from_tensor(self, decoder: CPUCacheDecoder) -> None:
        tokenizer = NumberTokenizer()
        ids = tokenizer.encode("ab")
        tensor = torch.tensor(ids)
        text = decoder.decode(tensor)
        assert text == "ab"

    def test_decode_ignores_pad_unk(self, decoder: CPUCacheDecoder) -> None:
        tokenizer = NumberTokenizer()
        ids = tokenizer.encode("a")
        ids.append(tokenizer.unk_token_id)
        ids.append(tokenizer.pad_token_id)
        text = decoder.decode(ids)
        assert text == "a"

    def test_decode_batch(self, decoder: CPUCacheDecoder) -> None:
        tokenizer = NumberTokenizer()
        ids = [tokenizer.encode("ab"), tokenizer.encode("c")]
        batch = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in ids], batch_first=True
        )
        texts = decoder.decode_batch(batch)
        assert texts[0] == "ab"
        assert texts[1] == "c"
