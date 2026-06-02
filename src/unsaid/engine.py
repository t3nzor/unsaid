"""Next-token completion engine.

Provides a backend-agnostic ``CompletionEngine`` interface and a Hugging Face
``transformers`` implementation.

Two completion modes
---------------------
* **Raw next token** (``topk``): the exact distribution over the model's next
  token given the tokenized text. Faithful, but mid-word it exposes a BPE
  artifact: typing ``Hel`` is the single token ``Hel``, and the model almost
  never emits ``lo`` after it (it learned the single token ``Hello``), so
  "Hello"/"Help" don't appear.
* **Token healing** (``complete``, the default): when the cursor sits mid-word,
  condition on the text *before* the partial word and rank the vocabulary
  tokens whose text continues what's been typed. This surfaces the words you'd
  expect (``Hel`` -> ``Hello``, ``Help``, ...). Each candidate's ``text`` is the
  *remainder* still to be typed, so the UI/accept logic (which prepends the
  already-typed prefix) works unchanged.

This project runs CPU-only on purpose (see AGENTS.md): the available GPU is an
RTX 5070 (sm_120) which the installed torch build cannot drive. We never touch
CUDA here.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

_TRAILING_WORD = re.compile(r"\S+$")


def current_word_prefix(text: str) -> str:
    """Return the partial word at the end of ``text``.

    This is the trailing run of non-whitespace characters; empty if ``text`` is
    empty or ends in whitespace (i.e. the next token would start a new word).
    """
    match = _TRAILING_WORD.search(text)
    return match.group(0) if match else ""


@dataclass(frozen=True)
class Candidate:
    """A single completion candidate.

    Attributes:
        token_id: The vocabulary id of the token.
        text: The text this candidate contributes *beyond what is already
            typed*. In raw mode this is the whole next token; in healed mode it
            is the remainder that completes the current word. May be a subword
            fragment and may include a leading space.
        prob: Probability assigned to this candidate in ``[0, 1]``. In healed
            mode it is renormalized over the matching completions.
    """

    token_id: int
    text: str
    prob: float


class CompletionEngine(ABC):
    """Backend-agnostic source of next-token / completion distributions."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` into model token ids."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back into text."""

    @abstractmethod
    def topk(self, token_ids: list[int], k: int) -> list[Candidate]:
        """Return the ``k`` most likely raw next tokens given ``token_ids``.

        Implementations must tolerate an empty ``token_ids`` list (predicting
        the very first token of a sequence).
        """

    def complete(self, text: str, k: int) -> list[Candidate]:
        """Return the ``k`` best completions for ``text``.

        Default implementation is the raw next-token distribution. Backends
        that support token healing override this.
        """
        return self.topk(self.encode(text), k)


class HFEngine(CompletionEngine):
    """A Hugging Face causal-LM backed engine, pinned to CPU."""

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        temperature: float = 1.0,
        num_threads: int | None = None,
        heal: bool = True,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        # Import torch/transformers lazily so that --help and unit tests for
        # pure modules don't pay the (multi-second) import cost.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        self._torch = torch
        self.model_name = model_name
        self.temperature = temperature
        self.heal = heal

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to("cpu")
        self.model.eval()

        # Lazily built map of token id -> decoded text, used for healing.
        self._vocab_strings: list[str] | None = None

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _bos_ids(self) -> list[int]:
        bos = self.tokenizer.bos_token_id
        if bos is None:
            bos = self.tokenizer.eos_token_id
        return [bos] if bos is not None else []

    def _next_token_probs(self, token_ids: list[int]):
        """Softmax over the vocabulary for the position after ``token_ids``."""
        torch = self._torch
        ids = token_ids or self._bos_ids()
        if not ids:
            return None
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(input_ids).logits[0, -1]
        return torch.softmax(logits / self.temperature, dim=-1)

    def topk(self, token_ids: list[int], k: int) -> list[Candidate]:
        if k <= 0:
            raise ValueError("k must be > 0")
        probs = self._next_token_probs(token_ids)
        if probs is None:
            return []
        k = min(k, probs.shape[-1])
        top_probs, top_ids = self._torch.topk(probs, k)
        out: list[Candidate] = []
        for prob, tid in zip(top_probs.tolist(), top_ids.tolist(), strict=True):
            out.append(
                Candidate(
                    token_id=int(tid),
                    text=self.tokenizer.decode([int(tid)]),
                    prob=float(prob),
                )
            )
        return out

    def _vocab(self) -> list[str]:
        if self._vocab_strings is None:
            n = len(self.tokenizer)
            self._vocab_strings = self.tokenizer.batch_decode([[i] for i in range(n)])
        return self._vocab_strings

    def complete(self, text: str, k: int) -> list[Candidate]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if not self.heal:
            return self.topk(self.encode(text), k)

        partial = current_word_prefix(text)
        before = text[: len(text) - len(partial)]
        stripped = before.rstrip(" ")
        had_space = len(stripped) < len(before)
        # GPT-2 attaches a leading space to the *following* word token, so a
        # trailing space tokenizes as a dangling space token and the model
        # predicts garbage after it. Heal it: condition on the text before the
        # space and look for tokens that begin with the space (i.e. new words).
        # The same machinery handles a partial word (`Hel`) and a word boundary
        # (`Hello `) — in both cases we re-predict the trailing fragment.
        search = (" " if had_space else "") + partial
        if not search:
            # Nothing to heal (empty buffer, or trailing newline/tab).
            return self.topk(self.encode(text), k)

        probs = self._next_token_probs(self.encode(stripped))
        if probs is None:
            return self.topk(self.encode(text), k)

        prob_list = probs.tolist()
        matched = [
            (tid, p)
            for tid, (s, p) in enumerate(zip(self._vocab(), prob_list, strict=True))
            if s.startswith(search)
        ]
        if not matched:
            # The boundary has no clean single-token completion; fall back so
            # the panel is never empty.
            return self.topk(self.encode(text), k)

        total = sum(p for _, p in matched) or 1.0
        matched.sort(key=lambda x: x[1], reverse=True)
        cut = len(search)
        out: list[Candidate] = []
        for tid, p in matched:
            remainder = self._vocab()[tid][cut:]
            # Skip candidates that would render as nothing (e.g. a bare extra
            # space token at a word boundary, where there is no typed prefix).
            if not partial and not remainder:
                continue
            out.append(Candidate(token_id=tid, text=remainder, prob=p / total))
            if len(out) >= k:
                break
        return out or self.topk(self.encode(text), k)

