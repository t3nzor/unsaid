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
* **Character completions** (``complete``): rank the next character to type,
  aggregating the probabilities of all tokens that begin with that character.
  When token healing is enabled, this aggregation happens over vocabulary tokens
  that continue the current partial word (``Hel`` -> ``l`` via ``Hello``, ``p``
  via ``Help``, ...).

The Hugging Face backend can run on CPU or CUDA. ``device="auto"`` picks the
CUDA device with the most VRAM when CUDA is available, otherwise CPU.
``dtype="auto"`` uses BF16/FP16 on CUDA and FP32 on CPU. 4-bit loading uses
BitsAndBytes NF4 quantization.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.util import find_spec

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
            typed*. For ``complete`` results this is one character; for ``topk``
            results this is the raw next token. May include whitespace.
        prob: Probability assigned to this candidate in ``[0, 1]``. In healed
            mode it is renormalized over the matching completions.
        continuation: For character candidates, the most likely token remainder
            that starts with ``text``. Rendering uses it as display context while
            accepting the candidate still appends only ``text``.
    """

    token_id: int
    text: str
    prob: float
    continuation: str | None = None


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
        """Return the ``k`` best next-character completions for ``text``.

        Default implementation returns raw next-token candidates. Backends that
        support character aggregation override this.
        """
        return self.topk(self.encode(text), k)

    def surprisal(self, text: str) -> float:
        """Total surprisal of ``text`` in bits (``-sum log2 P(token|context)``).

        Default is ``0.0``; model backends override it.
        """
        return 0.0


class HFEngine(CompletionEngine):
    """A Hugging Face causal-LM backed engine."""

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        temperature: float = 1.0,
        num_threads: int | None = None,
        heal: bool = True,
        device: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        hf_token: str | None = None,
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
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.load_in_4bit = load_in_4bit

        hub_kwargs = {"token": hf_token} if hf_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hub_kwargs)
        model_kwargs = self._model_kwargs()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **hub_kwargs, **model_kwargs
        )
        if not self.load_in_4bit:
            self.model.to(self.device)
        self.model.eval()

        # Lazily built map of token id -> decoded text, used for healing.
        self._vocab_strings: list[str] | None = None

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _resolve_device(self, device: str):
        torch = self._torch
        requested = device.strip().lower()
        if not requested:
            raise ValueError("device must not be empty")

        if requested == "auto":
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                best = max(
                    range(torch.cuda.device_count()),
                    key=lambda i: torch.cuda.get_device_properties(i).total_memory,
                )
                return torch.device(f"cuda:{best}")
            return torch.device("cpu")

        try:
            resolved = torch.device(requested)
        except RuntimeError as exc:
            raise ValueError(f"invalid torch device: {device!r}") from exc

        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA device requested, but CUDA is not available")
            count = torch.cuda.device_count()
            index = 0 if resolved.index is None else resolved.index
            if index < 0 or index >= count:
                raise ValueError(
                    f"CUDA device index {index} out of range; {count} device(s) available"
                )
            if resolved.index is None:
                return torch.device("cuda:0")
        return resolved

    def _resolve_dtype(self, dtype: str):
        torch = self._torch
        requested = dtype.strip().lower()
        if not requested:
            raise ValueError("dtype must not be empty")

        if requested == "auto":
            if self.device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        aliases = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        try:
            return aliases[requested]
        except KeyError as exc:
            choices = "auto, float32, float16, bfloat16"
            raise ValueError(
                f"invalid dtype: {dtype!r}; expected one of: {choices}"
            ) from exc

    def _model_kwargs(self) -> dict[str, object]:
        if self.load_in_4bit:
            if self.device.type != "cuda":
                raise ValueError("4-bit loading requires a CUDA device")
            missing = [
                pkg for pkg in ("accelerate", "bitsandbytes") if find_spec(pkg) is None
            ]
            if missing:
                names = ", ".join(missing)
                raise ValueError(
                    f"4-bit loading requires {names}; install with the quant extra"
                )

            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ValueError(
                    "4-bit loading requires a transformers build with BitsAndBytesConfig"
                ) from exc

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
            return {
                "device_map": {"": str(self.device)},
                "quantization_config": quantization_config,
            }

        return {"dtype": self.dtype}

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
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
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
        for prob, tid in zip(
            top_probs.cpu().tolist(), top_ids.cpu().tolist(), strict=True
        ):
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

    def _letter_candidates(
        self, continuations: list[tuple[int, str, float]], k: int
    ) -> list[Candidate]:
        grouped: dict[str, tuple[float, float, int, str]] = {}
        for tid, continuation, prob in continuations:
            if not continuation:
                continue
            letter = continuation[0]
            group = grouped.get(letter)
            if group is None:
                grouped[letter] = (prob, prob, tid, continuation)
                continue

            total_prob, best_prob, best_token_id, best_continuation = group
            if prob > best_prob:
                grouped[letter] = (total_prob + prob, prob, tid, continuation)
            else:
                grouped[letter] = (
                    total_prob + prob,
                    best_prob,
                    best_token_id,
                    best_continuation,
                )

        total = sum(total_prob for total_prob, _, _, _ in grouped.values()) or 1.0
        out = [
            Candidate(
                token_id=token_id,
                text=letter,
                prob=prob / total,
                continuation=continuation,
            )
            for letter, (prob, _, token_id, continuation) in grouped.items()
        ]
        out.sort(key=lambda cand: cand.prob, reverse=True)
        return out[:k]

    def _raw_letter_candidates(self, token_ids: list[int], k: int) -> list[Candidate]:
        probs = self._next_token_probs(token_ids)
        if probs is None:
            return []

        vocab = self._vocab()
        prob_list = probs.cpu().tolist()
        # Some models pad their LM head vocabulary beyond tokenizer length
        # (e.g. Qwen2.5: 152064 logits vs. 151665 tokenizer entries). Those
        # extra ids are not decodable tokens, so character aggregation ignores
        # them.
        continuations = [
            (tid, continuation, p)
            for tid, (continuation, p) in enumerate(zip(vocab, prob_list, strict=False))
        ]
        return self._letter_candidates(continuations, k)

    def complete(self, text: str, k: int) -> list[Candidate]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if not self.heal:
            return self._raw_letter_candidates(self.encode(text), k)

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
            return self._raw_letter_candidates(self.encode(text), k)

        probs = self._next_token_probs(self.encode(stripped))
        if probs is None:
            return self._raw_letter_candidates(self.encode(text), k)

        vocab = self._vocab()
        prob_list = probs.cpu().tolist()
        # Some models pad their LM head vocabulary beyond tokenizer length
        # (e.g. Qwen2.5: 152064 logits vs. 151665 tokenizer entries). Those
        # extra ids are not decodable tokens, so token healing ignores them.
        cut = len(search)
        matched = [
            (tid, s[cut:], p)
            for tid, (s, p) in enumerate(zip(vocab, prob_list, strict=False))
            if s.startswith(search) and s[cut:]
        ]
        if not matched:
            # The boundary has no clean single-token completion; fall back so
            # the panel is never empty.
            return self._raw_letter_candidates(self.encode(text), k)

        return self._letter_candidates(matched, k)

    def surprisal(self, text: str) -> float:
        """Total surprisal of ``text`` in bits under the model's *true*
        distribution (temperature is not applied here).

        Each token is scored given its predecessors; the first token is scored
        against BOS when the tokenizer provides one.  Returns ``0.0`` for text
        with no scoreable tokens.
        """
        torch = self._torch
        ids = self.encode(text)
        if not ids:
            return 0.0

        bos = self._bos_ids()
        seq = bos + ids
        first = len(bos)
        if first == 0:
            first = 1  # no context to score the first token against
        if first >= len(seq):
            return 0.0

        input_ids = torch.tensor([seq], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids).logits[0]  # [len(seq), vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

        targets = torch.tensor(seq[first:], dtype=torch.long, device=self.device)
        # Token seq[j] is predicted by the logits at position j-1.
        predictors = log_probs[first - 1 : len(seq) - 1]
        token_logp = predictors.gather(1, targets.unsqueeze(1)).squeeze(1)
        nats = -float(token_logp.sum())
        return nats / math.log(2)
