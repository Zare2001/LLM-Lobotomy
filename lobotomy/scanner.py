"""Core Lobotomy scanner — layer duplication engine.

Provides the LobotomyScanner class which loads a model, swaps its
layer execution path for a given (i, j) configuration, and runs
inference through the modified architecture.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def build_layer_path(n_layers: int, i: int, j: int) -> list[int]:
    """Build the execution path for configuration (i, j).

    Layers 0..j-1 execute first, then i..j-1 are repeated (the Lobotomy
    duplication), then j..n-1 continue normally.

    (0, 0) returns the unmodified baseline path.
    """
    if i == 0 and j == 0:
        return list(range(n_layers))
    if i < 0 or j < 0 or i >= j or j > n_layers:
        raise ValueError(
            f"Invalid config ({i}, {j}) for model with {n_layers} layers. "
            f"Need 0 <= i < j <= {n_layers}."
        )
    first_pass = list(range(j))
    repeated = list(range(i, j))
    remainder = list(range(j, n_layers))
    return first_pass + repeated + remainder


def build_looped_layer_path(
    n_layers: int, i: int, j: int, n_repeats: int = 2
) -> list[int]:
    """Build a path where the block i..j-1 is repeated n_repeats times.

    The standard Lobotomy is n_repeats=2 (original + one duplication).
    n_repeats=3 means the block executes three times total ("triple pass").

    Suggested by HN discussion: rapatel0, phire proposed looping the
    "reasoning circuit" N times as test-time compute scaling.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if n_repeats == 1:
        return list(range(n_layers))
    if i < 0 or j < 0 or i >= j or j > n_layers:
        raise ValueError(f"Invalid config ({i}, {j}) for {n_layers} layers.")

    first_pass = list(range(j))
    repeated = list(range(i, j)) * (n_repeats - 1)
    remainder = list(range(j, n_layers))
    return first_pass + repeated + remainder


def build_multi_circuit_path(
    n_layers: int, circuits: list[tuple[int, int]]
) -> list[int]:
    """Build a path with multiple independent circuits duplicated.

    Each circuit is an (i, j) pair. Circuits must be non-overlapping and
    sorted by start position. Each block is duplicated once (double pass).

    The original author mentioned experimenting with "multiple layer
    duplications in different regions" and training a meta-model (XGBoost)
    to predict optimal multi-circuit combinations.
    """
    for ci, (i, j) in enumerate(circuits):
        if i < 0 or j < 0 or i >= j or j > n_layers:
            raise ValueError(f"Invalid circuit ({i}, {j})")
        if ci > 0 and i < circuits[ci - 1][1]:
            raise ValueError(
                f"Circuits must be non-overlapping and sorted: "
                f"{circuits[ci-1]} overlaps {(i, j)}"
            )

    path: list[int] = []
    pos = 0
    for i, j in circuits:
        path.extend(range(pos, j))
        path.extend(range(i, j))
        pos = j
    path.extend(range(pos, n_layers))
    return path


def iter_configs(
    n_layers: int,
    i_range: tuple[int, int] | None = None,
    j_range: tuple[int, int] | None = None,
) -> Generator[tuple[int, int], None, None]:
    """Yield all valid (i, j) configurations for the sweep.

    Optionally restrict to a sub-region of the configuration space.
    Always yields (0, 0) first (the baseline).
    """
    yield (0, 0)
    i_lo, i_hi = i_range if i_range else (0, n_layers)
    j_lo, j_hi = j_range if j_range else (1, n_layers + 1)
    for i in range(i_lo, i_hi):
        for j in range(max(i + 1, j_lo), j_hi):
            if j > n_layers:
                continue
            yield (i, j)


class LobotomyScanner:
    """Loads a model and provides layer-path manipulation for scanning."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self._dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        logger.info("Loading tokenizer from %s", model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model from %s", model_name_or_path)
        load_kwargs: dict = dict(
            device_map=device_map,
            dtype=self._dtype,
            trust_remote_code=True,
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._dtype,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        self.model.eval()

        self._original_layers: nn.ModuleList = self.model.model.layers
        self._original_n_layers: int = self.model.config.num_hidden_layers
        self.n_layers: int = self._original_n_layers

        self.params_per_layer: int = sum(
            self._param_numel(p) for p in self._original_layers[0].parameters()
        )
        self.total_params: int = self._count_params()
        self.non_layer_params: int = self.total_params - (self.n_layers * self.params_per_layer)
        logger.info(
            "Model loaded: %d layers, %s parameters (%s per layer)",
            self.n_layers,
            f"{self.total_params:,}",
            f"{self.params_per_layer:,}",
        )

    def effective_params(self, path_length: int) -> int:
        """Effective parameter count for a given execution path length.

        Duplicated layers share weights but contribute to the effective
        computation budget, so they are counted again.
        """
        return self.non_layer_params + path_length * self.params_per_layer

    @staticmethod
    def _param_numel(p: torch.Tensor) -> int:
        """Return the logical number of elements, even for quantized params."""
        if hasattr(p, "quant_state") and p.quant_state is not None:
            import math as _math
            return _math.prod(p.quant_state.shape)
        return p.numel()

    def _count_params(self) -> int:
        seen: set[int] = set()
        total = 0
        for p in self.model.parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            total += self._param_numel(p)
        return total

    # ------------------------------------------------------------------
    # Layer path manipulation
    # ------------------------------------------------------------------

    @contextmanager
    def config(self, i: int, j: int):
        """Context manager that temporarily applies a Lobotomy configuration.

        Replaces the model's layer list with the custom execution path and
        restores it on exit.
        """
        if i == 0 and j == 0:
            yield
            return

        path = build_layer_path(self.n_layers, i, j)
        with self.custom_path(path):
            yield

    @contextmanager
    def custom_path(self, path: list[int]):
        """Context manager for an arbitrary layer execution path.

        Accepts any list of layer indices (built by any of the path builder
        functions). Restores the original model on exit.
        """
        new_layers = nn.ModuleList([self._original_layers[idx] for idx in path])
        self.model.model.layers = new_layers
        self.model.config.num_hidden_layers = len(new_layers)
        try:
            yield
        finally:
            self.model.model.layers = self._original_layers
            self.model.config.num_hidden_layers = self._original_n_layers

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize_chat(self, messages: list[dict[str, str]]) -> torch.Tensor:
        """Tokenize chat messages into input_ids tensor.

        Handles both raw tensor and BatchEncoding returns from
        apply_chat_template (behaviour varies across transformers versions).
        """
        try:
            result = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
        except Exception:
            user_msgs = [m for m in messages if m["role"] == "user"]
            system_msgs = [m for m in messages if m["role"] == "system"]
            prompt_parts = []
            if system_msgs:
                prompt_parts.append(system_msgs[0]["content"])
            if user_msgs:
                prompt_parts.append(user_msgs[0]["content"])
            prompt = "\n".join(prompt_parts)
            return self.tokenizer(prompt, return_tensors="pt").input_ids

        if hasattr(result, "input_ids"):
            return result.input_ids
        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 64,
    ) -> str:
        """Generate a response using the current (possibly lobotomized) model.

        Args:
            messages: Chat-format messages, e.g.
                [{"role": "user", "content": "What is 2+2?"}]
            max_new_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """
        input_ids = self._tokenize_chat(messages)
        input_ids = input_ids.to(self.model.device)

        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response_ids = output[0][input_ids.shape[1] :]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    @torch.no_grad()
    def forward_logits(
        self,
        messages: list[dict[str, str]],
    ) -> torch.Tensor:
        """Run a forward pass and return the logits at the last position.

        Useful for logit-based scoring (EQ probe improvement).
        """
        input_ids = self._tokenize_chat(messages)
        input_ids = input_ids.to(self.model.device)

        outputs = self.model(input_ids, use_cache=False)
        return outputs.logits[0, -1, :]

    def get_digit_token_ids(self) -> list[int]:
        """Return token IDs for the single-character digits '0' through '9'."""
        ids = []
        for digit in "0123456789":
            token_ids = self.tokenizer.encode(digit, add_special_tokens=False)
            ids.append(token_ids[-1])
        return ids
