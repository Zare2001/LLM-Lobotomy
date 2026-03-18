"""Lobotomy surgeon — apply the optimal configuration and save the model.

Two modes:
  - Virtual duplication:  pointer-based, no extra VRAM for weights
  - Physical duplication: deep-copied layers, produces a standalone model
                          that works with any inference framework
"""

from __future__ import annotations

import argparse
import copy
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .scanner import build_layer_path

logger = logging.getLogger(__name__)


def apply_lobotomy(
    model: AutoModelForCausalLM,
    i: int,
    j: int,
    *,
    physical: bool = False,
) -> AutoModelForCausalLM:
    """Apply a Lobotomy configuration to a model in-place.

    Args:
        model:    HuggingFace causal LM (Llama-style with model.model.layers).
        i:        Start of the duplicated block.
        j:        End of the duplicated block (exclusive).
        physical: If True, deep-copy the duplicated layers so they have
                  independent weights (required for saving).  If False,
                  use pointer sharing (no extra VRAM).

    Returns:
        The modified model (same object, mutated in-place).
    """
    original_layers = model.model.layers
    n_layers = len(original_layers)
    path = build_layer_path(n_layers, i, j)

    if physical:
        first_pass_end = j
        dup_start = j
        dup_end = j + (j - i)

        new_layers: list[nn.Module] = []
        for pos, layer_idx in enumerate(path):
            if dup_start <= pos < dup_end:
                cloned = copy.deepcopy(original_layers[layer_idx])
                if hasattr(cloned, "self_attn") and hasattr(cloned.self_attn, "layer_idx"):
                    cloned.self_attn.layer_idx = pos
                new_layers.append(cloned)
            else:
                new_layers.append(original_layers[layer_idx])
        model.model.layers = nn.ModuleList(new_layers)
    else:
        model.model.layers = nn.ModuleList(
            [original_layers[idx] for idx in path]
        )

    model.config.num_hidden_layers = len(path)
    logger.info(
        "Lobotomy applied: (%d, %d) — %d → %d layers (physical=%s)",
        i, j, n_layers, len(path), physical,
    )
    return model


def save_lobotomized_model(
    model_name_or_path: str,
    i: int,
    j: int,
    output_dir: str,
    *,
    load_in_4bit: bool = False,
    torch_dtype: str = "bfloat16",
) -> None:
    """Load a model, apply a physical Lobotomy, and save the result.

    The saved model is fully standalone — no special inference code needed.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    logger.info("Loading model: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    load_kwargs: dict = dict(
        device_map="cpu",
        dtype=dtype,
        trust_remote_code=True,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, **load_kwargs
    )

    apply_lobotomy(model, i, j, physical=True)

    logger.info("Saving lobotomized model to: %s", output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done. Model saved with %d layers.", model.config.num_hidden_layers)


def main():
    parser = argparse.ArgumentParser(description="Apply Lobotomy and save model")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--config", required=True,
        help="Optimal (i,j) configuration, e.g. '25,32'",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--4bit", dest="load_4bit", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    i_str, j_str = args.config.split(",")
    save_lobotomized_model(
        args.model,
        int(i_str.strip()),
        int(j_str.strip()),
        args.output,
        load_in_4bit=args.load_4bit,
        torch_dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
