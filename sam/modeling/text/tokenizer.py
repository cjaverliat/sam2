# SPDX-License-Identifier: LicenseRef-SAM
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Vendored from sam3/model/tokenizer_ve.py @ commit 5dd401d (Phase 1, Task 3).
# Extraneous deps stripped: iopath (replaced by stdlib gzip.open).
# ftfy and regex are required (available in the pixi env as transitive deps).
#
# BPE vocab / merges are loaded from the bundled asset
# ``sam/modeling/text/assets/bpe_simple_vocab_16e6.txt.gz``.  That file
# originates from OpenAI CLIP and ships unchanged with SAM 3; it is NOT model
# weights — it is vocabulary data and is safe to commit alongside the code.
"""SAM 3 BPE text tokenizer (CLIP-style)."""

import gzip
import html
import os
import string
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import ftfy
import regex as re
import torch

# Suppress HuggingFace parallel tokenizers warning (irrelevant here).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Default context length for SAM 3 (upstream: context_length=32 in VETextEncoder).
DEFAULT_CONTEXT_LENGTH: int = 32

# Bundled BPE asset (shipped alongside this module; committed as vocabulary data).
_ASSETS_DIR = Path(__file__).parent / "assets"
_DEFAULT_BPE_PATH = _ASSETS_DIR / "bpe_simple_vocab_16e6.txt.gz"


@lru_cache()
def bytes_to_unicode():
    """Return a mapping of utf-8 byte values → unique unicode chars.

    The reversible BPE codes work on unicode strings.  This lookup table avoids
    mapping to whitespace / control characters the BPE code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word (tuple of variable-length strings)."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_text(text: str, *, keep_punctuation_exact_string=None) -> str:
    """Lowercase and remove punctuation (from google-research/big_vision)."""
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_canonicalize(x: str) -> str:
    return canonicalize_text(basic_clean(x))


def _clean_lower(x: str) -> str:
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x: str) -> str:
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str):
    if type == "canonicalize":
        return _clean_canonicalize
    elif type == "lower":
        return _clean_lower
    elif type == "whitespace":
        return _clean_whitespace
    else:
        raise ValueError(f"Invalid clean function ({type}).")


class Sam3Tokenizer:
    """CLIP-style BPE tokenizer for SAM 3.

    Vendored from ``sam3/model/tokenizer_ve.py::SimpleTokenizer`` (upstream commit
    5dd401d).  The BPE vocab / merges are loaded from the bundled asset
    ``assets/bpe_simple_vocab_16e6.txt.gz`` by default (originating from OpenAI CLIP).

    Tokenization is byte-pair-encoding over byte-level unicode, with lowercasing
    (``clean="lower"``) by default.  The context length defaults to 32 (the SAM 3
    text-encoder sequence length), padded / truncated as needed.
    """

    def __init__(
        self,
        bpe_path: Optional[Union[str, Path]] = None,
        additional_special_tokens: Optional[List[str]] = None,
        context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
        clean: str = "lower",
    ):
        if bpe_path is None:
            bpe_path = _DEFAULT_BPE_PATH
        bpe_path = Path(bpe_path)

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with gzip.open(bpe_path, "rt", encoding="utf-8") as fh:
            merges = fh.read().split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}

        special = "|".join(special_tokens)
        self.pat = re.compile(
            special
            + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[int] = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def __call__(
        self,
        texts: Union[str, List[str]],
        context_length: Optional[int] = None,
    ) -> torch.LongTensor:
        """Tokenise ``texts`` and return a (N, context_length) LongTensor.

        Pads with 0 and truncates to ``context_length`` (default: 32 for SAM 3).
        SOT and EOT special tokens are prepended / appended automatically.
        """
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length
        assert context_length, "Please set a valid context length"
        all_tokens = [
            [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            for text in texts
        ]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result
