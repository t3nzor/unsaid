"""Next-token completion engine.

Provides a backend-agnostic ``CompletionEngine`` interface and a Hugging Face
``transformers`` implementation that returns the exact top-k next-token
distribution from a causal language model.

This project runs CPU-only on purpose (see AGENTS.md): the available GPU is an
RTX 5070 (sm_120) which the installed torch build cannot drive. We never touch
CUDA here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A single next-token candidate.

    Attributes:
        token_id: The vocabulary id of the token.
        text: The decoded text of *this token alone* (may be a subword
            fragment and may include a leading space).
        prob: Probability mass assigned to this token after temperature
            scaling, in ``[0, 1]``.
    """

    token_id: int
    text: str
    prob: float


class CompletionEngine(ABC):
    """Backend-agnostic source of next-token distributions."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` into model token ids."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back into text."""

    @abstractmethod
    def topk(self, token_ids: list[int], k: int) -> list[Candidate]:
        """Return the ``k`` most likely next tokens given ``token_ids``.

        Implementations must tolerate an empty ``token_ids`` list (predicting
        the very first token of a sequence).
        """


class HFEngine(CompletionEngine):
    """A Hugging Face causal-LM backed engine, pinned to CPU."""

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        temperature: float = 1.0,
        num_threads: int | None = None,
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to("cpu")
        self.model.eval()

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def topk(self, token_ids: list[int], k: int) -> list[Candidate]:
        if k <= 0:
            raise ValueError("k must be > 0")
        torch = self._torch

        if not token_ids:
            # No context: prime with the model's BOS/EOS so we still get a
            # well-defined first-token distribution where one exists.
            bos = self.tokenizer.bos_token_id
            if bos is None:
                bos = self.tokenizer.eos_token_id
            ids = [bos] if bos is not None else []
            if not ids:
                return []
            input_ids = torch.tensor([ids], dtype=torch.long)
        else:
            input_ids = torch.tensor([token_ids], dtype=torch.long)

        with torch.inference_mode():
            logits = self.model(input_ids).logits[0, -1]

        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        k = min(k, probs.shape[-1])
        top_probs, top_ids = torch.topk(probs, k)

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
