# LLM Lobotomy: Layer Duplication for EuroLLM-22B

## Overview

**Lobotomy** is a technique for improving LLM reasoning by duplicating contiguous blocks of
internal transformer layers — without modifying any weights. This is an implementation of and
improvement upon the [RYS (Repeat Your Self)](https://dnhkng.github.io/posts/rys/) method
by David Noel Ng, applied to
[EuroLLM-22B-Instruct-2512](https://huggingface.co/utter-project/EuroLLM-22B-Instruct-2512).

### Core Idea

Transformer models develop a functional anatomy during training:

- **Early layers** — encode input tokens into an abstract internal representation
- **Middle layers** — perform reasoning in that abstract space via multi-layer "circuits"
- **Late layers** — decode the abstract representation back into output tokens

By identifying the reasoning circuits in the middle layers and duplicating them (running them
twice during inference), we give the model a "second pass" through its own reasoning pipeline.
No weights are changed. The model simply traverses some of its own layers twice.

### Why "Lobotomy"?

In the original research, duplicating the *wrong* layers produces effects resembling genuine
brain damage — degenerate loops, personality shifts, incoherent outputs. The process of
surgically selecting *which* layers to duplicate (and which to leave alone) is analogous to
neural surgery. We call the full process — scanning, selecting, and duplicating — a **Lobotomy**.

---

## Target Model: EuroLLM-22B-Instruct-2512

| Property | Value |
|---|---|
| Architecture | LlamaForCausalLM |
| Parameters | ~22B |
| Hidden layers | **54** |
| Hidden size | 6144 |
| Attention heads | 48 (8 KV heads, GQA) |
| Head dimension | 128 |
| Intermediate size | 16384 |
| Vocab size | 128,000 |
| Context length | 32,768 |
| Precision | bfloat16 |
| Languages | 35 |

With 54 layers, the total Lobotomy configuration space is:

$$\text{Configurations} = \frac{54 \times 55}{2} + 1 = 1486$$

Each configuration `(i, j)` means: run layers `0..j-1` normally, then repeat layers `i..j-1`,
then continue with layers `j..53`. The pair `(0, 0)` represents the unmodified baseline.

---

## Implementation Plan

### Phase 0: Environment Setup

1. **Hardware requirements**
   - Minimum: 1× GPU with ≥48 GB VRAM (A6000, L40S, A100, H100), or
   - 2× GPUs with ≥24 GB VRAM each (RTX 3090/4090) with model parallelism
   - EuroLLM-22B at FP16 requires ~45 GB; quantized (GPTQ/AWQ 4-bit) requires ~12 GB
   - Recommended: Use quantized model for the sweep, full precision for final validation

2. **Software dependencies**
   ```
   torch>=2.1
   transformers>=4.51
   accelerate
   safetensors
   matplotlib
   seaborn
   numpy
   tqdm
   scikit-optimize    # for Bayesian optimization
   ```

3. **Download the model**
   ```bash
   huggingface-cli download utter-project/EuroLLM-22B-Instruct-2512
   ```

4. **Project structure**
   ```
   LLM Lobotomy/
   ├── LOBOTOMY.md              # This document
   ├── lobotomy/
   │   ├── __init__.py
   │   ├── scanner.py           # Core scanner (layer duplication + advanced path builders)
   │   ├── probes.py            # Evaluation probes (math, EQ, multilingual)
   │   ├── scoring.py           # Scoring functions (partial credit, logit-based)
   │   ├── heatmap.py           # Visualization of scan results
   │   ├── surgeon.py           # Apply optimal config and save model
   │   └── optimize.py          # Bayesian optimization (improvement over exhaustive sweep)
   ├── data/
   │   ├── math_probes.json     # Hard math questions
   │   ├── eq_probes.json       # Emotional quotient scenarios
   │   └── multilingual_probes.json  # Multilingual probes (improvement)
   ├── results/
   │   ├── heatmaps/            # Generated heatmap images
   │   └── scores/              # Raw scoring data per (i,j) config
   ├── requirements.txt
   └── run_lobotomy.py          # Main entry point
   ```

---

### Phase 1: The Lobotomy Scanner (`scanner.py`)

The scanner is the core engine. It takes a model, a configuration `(i, j)`, and modifies the
forward pass to duplicate layers `i` through `j-1`.

**Implementation approach — Hook-based layer duplication:**

Rather than physically copying layer weights (which would double VRAM usage), we modify the
model's `forward()` method to re-traverse the specified layers. This is done by:

1. Loading the model normally
2. Intercepting the layer execution loop
3. For a given `(i, j)`, building the execution path:
   `[0, 1, ..., j-1, i, i+1, ..., j-1, j, j+1, ..., 53]`
4. Running inference with this modified path

```python
def build_layer_path(n_layers: int, i: int, j: int) -> list[int]:
    """Build the Lobotomy execution path for configuration (i, j).
    
    Layers 0..j-1 run first, then i..j-1 are repeated, then j..n-1 continue.
    """
    if i == 0 and j == 0:
        return list(range(n_layers))  # baseline
    
    first_pass = list(range(j))          # 0 to j-1
    repeated = list(range(i, j))         # i to j-1 (duplicated)
    remainder = list(range(j, n_layers)) # j to N-1
    
    return first_pass + repeated + remainder
```

**Modifying the Llama forward pass:**

For HuggingFace `LlamaForCausalLM`, the layers are stored in `model.model.layers` (an
`nn.ModuleList`). We override the model's forward to iterate through our custom path instead
of sequentially through the list:

```python
def lobotomized_forward(model, input_ids, layer_path, **kwargs):
    """Run a single forward pass through the model using a custom layer path."""
    hidden_states = model.model.embed_tokens(input_ids)
    
    position_ids = kwargs.get("position_ids")
    attention_mask = kwargs.get("attention_mask")
    
    for layer_idx in layer_path:
        layer = model.model.layers[layer_idx]
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]
    
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits
```

**Critical detail — KV cache handling:**

When duplicating layers, the KV cache must be handled carefully. During the duplicated
layers' second pass, they receive different hidden states than their first pass, so their KV
cache entries from the first pass are stale. Options:

- **Simplest**: Disable KV cache during scanning (slower but correct)
- **Better**: Clear KV cache entries for layers `i..j-1` before the second pass
- **Best**: Use separate KV cache slots for first-pass and second-pass instances

For the scanning phase, disabling KV cache is acceptable since we only need a few tokens of
output per probe.

---

### Phase 2: Evaluation Probes (`probes.py`)

Two primary probes (from the original research) plus one improvement probe:

#### Probe 1: Hard Math Guessing

Questions where the model must output a numerical answer *without* chain-of-thought reasoning.
The model must guess intuitively — this tests raw numerical intuition.

```json
[
  {"question": "What is 78313086360375 multiplied by 88537453126609?", "answer": 6933469759697285938234375},
  {"question": "What is the cube root of 18228885506341?", "answer": 263147},
  {"question": "What is the cube root of 844178022493, multiplied by 43?", "answer": 40557},
  {"question": "What is 9437 squared?", "answer": 89056969},
  {"question": "What is 2^47?", "answer": 140737488355328}
]
```

**Prompt format** (instruct-tuned model):
```
Answer with ONLY the number, no explanation.
What is the cube root of 74,088,893,247?
```

**Scoring**: Use partial-credit scoring (see `scoring.py` below) — not binary right/wrong.

#### Probe 2: Emotional Quotient (EQ-Bench style)

Social scenarios where the model predicts emotional intensity on a 0-100 scale. Tests theory
of mind, empathy, and social reasoning — maximally orthogonal to math.

```json
{
  "scenario": "Alex discovers that their closest friend has been secretly applying for the same dream job Alex has been working towards for years. The friend never mentioned it.",
  "emotions": ["betrayal", "anger", "sadness", "confusion"],
  "expected": [78, 65, 72, 60]
}
```

**Scoring**: Mean absolute error between predicted and expected emotion intensities.

#### Probe 3 (Improvement): Multilingual Reasoning

Since EuroLLM covers 35 languages, we add a multilingual probe — the same hard math questions
translated into 5-10 of EuroLLM's supported languages (e.g., French, German, Spanish,
Portuguese, Dutch). This ensures the optimal Lobotomy configuration doesn't degrade
multilingual capability.

**Why this is an improvement**: The original RYS research only used English probes. A
multilingual model might have language-specific circuits that could be damaged by layer
duplication optimized only for English.

---

### Phase 3: Scoring Functions (`scoring.py`)

#### Partial Credit Math Scoring

```python
def calculate_score(actual: int, estimate: int) -> float:
    """Partial-credit scoring for numerical answers.
    
    Handles LLM arithmetic quirks: dropped digits, transpositions,
    truncated outputs. Pads shorter answers and penalizes proportionally.
    """
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(estimate))
    except (ValueError, OverflowError):
        return 0.0

    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")

    padding_size = max_length - min(len(actual_str), len(estimate_str))
    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)

    if max(actual_int, estimate_int) == 0:
        return 0.0

    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor
    return max(0.0, min(score, 1.0))
```

#### Logit-Based EQ Scoring (Improvement)

Instead of sampling a single number, capture the model's logit distribution over digit tokens
and compute the expected value:

$$\hat{s} = \sum_{k=0}^{9} k \cdot p(k), \quad p(k) = \frac{\exp(z_k)}{\sum_{m=0}^{9} \exp(z_m)}$$

This produces a smooth score (e.g., 5.4) rather than a noisy sampled integer.

---

### Phase 4: Running the Lobotomy Sweep

#### Option A: Exhaustive Sweep (Original Method)

Iterate over all 1,486 `(i, j)` configurations. For each:

1. Build the layer path
2. Run all math probes → compute average math score
3. Run all EQ probes → compute average EQ score
4. Store `(i, j, math_delta, eq_delta, combined_delta)`

**Estimated time**: With quantized model, ~2 min per config × 1,486 = ~50 hours on a single
GPU. Parallelizable across GPUs.

```bash
python run_lobotomy.py \
  --model utter-project/EuroLLM-22B-Instruct-2512 \
  --mode sweep \
  --output results/scores/eurollm_sweep.csv
```

#### Option B: Bayesian Optimization (Improvement)

Instead of exhaustive sweep, use Bayesian optimization (Gaussian Process with Expected
Improvement acquisition) to find the optimal `(i, j)` in far fewer evaluations.

**Why this is an improvement**: The original method required testing all configurations
because heatmap visualization was a goal. But if you only need the *optimal* configuration,
Bayesian optimization can find it in ~100-200 evaluations instead of 1,486 — a 7-10× speedup.

```python
from skopt import gp_minimize
from skopt.space import Integer

space = [Integer(0, 53, name='i'), Integer(1, 54, name='j')]

def objective(params):
    i, j = params
    if i >= j:
        return 0.0  # invalid config
    math_score = run_math_probes(model, i, j)
    eq_score = run_eq_probes(model, i, j)
    return -(math_score + eq_score)  # minimize negative = maximize

result = gp_minimize(objective, space, n_calls=200, random_state=42)
best_i, best_j = result.x
```

**Recommendation**: Run Option B first to find the approximate optimal region, then run a
focused exhaustive sweep in that neighborhood for the heatmap.

---

### Phase 5: Visualization — The Lobotomy Scan (`heatmap.py`)

Generate heatmaps analogous to functional MRIs of the transformer:

- **X-axis**: End layer `j` (0 to 54)
- **Y-axis**: Start layer `i` (0 to 54)
- **Color**: Score delta vs baseline (red = improvement, blue = degradation)

Generate three heatmaps:
1. **Math delta** — shows where numerical reasoning circuits live
2. **EQ delta** — shows where social/emotional reasoning circuits live
3. **Combined delta** — the sum, used to find the optimal Lobotomy configuration

Also generate **skyline plots** (row/column averages) to visualize:
- Which start positions matter most
- Which end positions matter most
- The boundaries of functional circuits

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_lobotomy_heatmap(scores, title, output_path):
    n = 54
    heatmap = np.full((n + 1, n + 1), np.nan)
    for i, j, delta in scores:
        heatmap[i, j] = delta

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        heatmap, cmap="RdBu_r", center=0,
        xticklabels=5, yticklabels=5,
        ax=ax, cbar_kws={"label": "Score Δ vs baseline"}
    )
    ax.set_xlabel("End layer (j)")
    ax.set_ylabel("Start layer (i)")
    ax.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
```

---

### Phase 6: Apply the Lobotomy (`surgeon.py`)

Once the optimal `(i*, j*)` is found, create the lobotomized model:

#### Method A: Virtual Duplication (No Extra VRAM)

Use pointer-based layer sharing. The duplicated layers reference the same weight tensors
as the originals. This uses no additional GPU memory — only extra compute and KV cache.

```python
def apply_lobotomy(model, i, j):
    """Create a lobotomized model by duplicating layers i..j-1.
    
    Uses pointer-based sharing — no extra VRAM for weights.
    """
    import copy
    original_layers = model.model.layers
    
    new_layers = torch.nn.ModuleList()
    # First pass: layers 0 to j-1
    for idx in range(j):
        new_layers.append(original_layers[idx])
    # Duplicated: layers i to j-1 (same weight objects, different position)
    for idx in range(i, j):
        new_layers.append(original_layers[idx])
    # Remainder: layers j to N-1
    for idx in range(j, len(original_layers)):
        new_layers.append(original_layers[idx])
    
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model
```

#### Method B: Physical Duplication (For Saving/Uploading)

Create actual copies of the duplicated layers and save as a new model with updated config.
This increases model size but produces a standalone model that works with standard inference
frameworks.

```python
def save_lobotomized_model(model, tokenizer, i, j, output_dir):
    """Save a fully materialized lobotomized model."""
    import copy
    
    original_layers = list(model.model.layers)
    new_layers = (
        original_layers[:j]
        + [copy.deepcopy(original_layers[idx]) for idx in range(i, j)]
        + original_layers[j:]
    )
    model.model.layers = torch.nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

---

### Phase 7: Advanced Layer Manipulation (from HN discussion)

Several techniques beyond simple (i,j) block duplication were suggested in the
[Hacker News discussion](https://news.ycombinator.com/item?id=47322887) and confirmed
by the original author. Both are implemented in `scanner.py`.

#### Triple+ Pass (Looped Duplication)

Instead of running the block twice, loop it N times. The author confirmed he experimented
with this. HN users rapatel0 and phire proposed it as a form of test-time compute scaling.

```python
from lobotomy.scanner import build_looped_layer_path

# Triple pass: block executes 3 times total
path = build_looped_layer_path(n_layers=54, i=25, j=32, n_repeats=3)
# path length = 54 + 2*(32-25) = 54 + 14 = 68

with scanner.custom_path(path):
    result = math_probe.run(scanner)
```

HN user hashmap tested this on Qwen2-72B and reported significant improvements — narrowing
the block to L48-53 and repeating it 2× gave +23.5% overall improvement vs baseline.

| Config | Layers | Overall | Δ |
|---|---|---|---|
| baseline | 80 | 0.5391 | — |
| RYS (45,52) | 87 | 0.5452 | +0.0061 |
| focused sub-block ×2 | 92 | 0.7741 | +0.2350 |

#### Multiple Disjoint Circuits

The original author mentioned experimenting with "multiple layer duplications in different
regions" and training a meta-model (XGBoost) to predict optimal combinations. This
duplicates two or more non-overlapping blocks in a single forward pass.

```python
from lobotomy.scanner import build_multi_circuit_path

# Duplicate two independent circuits simultaneously
path = build_multi_circuit_path(54, circuits=[(15, 22), (35, 42)])
# Both blocks get doubled in a single forward pass

with scanner.custom_path(path):
    result = math_probe.run(scanner)
```

#### What Does NOT Work

The original research and HN replication efforts confirmed these approaches fail:

- **Single-layer duplication** — repeating one layer N times "almost always did worse"
  (from the blog post). The middle layers work as multi-step circuits, not interchangeable
  units. Duplicating one step of a recipe doesn't help.
- **Inner-block sub-duplication** — repeating individual layers *within* a duplicated block
  causes interference (HN user hashmap: "if I doubled the layers and doubled that block
  there was interference"). Block duplication or individual layer duplication work
  independently, but not combined.

---

### Phase 8: Validation

Run the final lobotomized model against standard benchmarks to confirm generalization:

1. **Open LLM Leaderboard v2 tasks**: IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-PRO
2. **Multilingual benchmarks**: Since EuroLLM is a multilingual model, validate across
   multiple languages to ensure the Lobotomy didn't damage language-specific circuits
3. **Perplexity**: Measure on held-out text in multiple languages — a degradation here would
   signal damage to the encoding/decoding layers

Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for
standardized benchmarking:

```bash
lm_eval --model hf \
  --model_args pretrained=./lobotomized_eurollm \
  --tasks ifeval,bbh,math_hard,gpqa,musr,mmlu_pro \
  --batch_size auto
```

---

## Summary of Improvements Over Original RYS

| Aspect | Original RYS | Lobotomy (Improved) |
|---|---|---|
| Sweep strategy | Exhaustive (all configs) | Bayesian optimization → focused sweep |
| Probes | English math + EQ only | + Multilingual probes for 35-lang model |
| Scoring | Custom partial-credit | + Logit-distribution EQ scoring |
| Multi-circuit | Single block only | Multiple disjoint circuits in one forward pass |
| Passes | Double pass only | Configurable N-pass looping (triple, quadruple, ...) |

### Sources for Improvements

| Technique | Source |
|---|---|
| N-pass looping | [rapatel0](https://news.ycombinator.com/item?id=47327766), [phire](https://news.ycombinator.com/item?id=47332411) on HN |
| Multiple disjoint circuits | [efromvt](https://news.ycombinator.com/item?id=47327803) on HN; author confirmed in [reply](https://news.ycombinator.com/item?id=47325289) |
| Layer order invariance | [hashmap](https://news.ycombinator.com/item?id=47360629) on HN (empirical replication) |
| Residual connections as enabler | [evangambit](https://news.ycombinator.com/item?id=47344921), [WithinReason](https://news.ycombinator.com/item?id=47324864) on HN |

### Related Academic Work (cited in HN thread)

- [SOLAR/DUS (Kim et al., 2023)](https://arxiv.org/abs/2312.15166) — duplicated transformer
  layers to build a 10.7B model outperforming 30B baselines
- [The Curse of Depth (2025)](https://arxiv.org/abs/2502.05795) — explains *why* duplication
  works: Pre-LN causes deep layers to converge toward identity functions
- [Scaling Test-Time Compute with Latent Reasoning (Geiping et al., 2025)](https://arxiv.org/abs/2502.05171)
  — model trained with a single recurrent block repeated at inference time
- [LoopLM](https://arxiv.org/abs/2510.25741) — explicit loop-based transformer architecture
- [Ouro-LLM](https://ouro-llm.github.io/) — looped layer execution for reasoning scaling

---

## Expected Outcomes

For a 54-layer model like EuroLLM-22B, based on patterns observed in other architectures:

- The **encoding layers** are likely in the range 0–8
- The **decoding layers** are likely in the range 44–53
- The **reasoning circuits** are likely in the range 9–43
- The optimal Lobotomy block will likely be 5–10 contiguous layers somewhere in the 15–40
  range
- The resulting model will have ~54 + N duplicated layers (e.g., 61 layers if 7 are
  duplicated), increasing parameter count by ~10-15% without requiring any new training

**The key prediction**: A lobotomized EuroLLM-22B should show measurable improvements on
reasoning benchmarks (BBH, MATH, MuSR, GPQA) with minimal degradation on instruction
following (IFEval) and knowledge (MMLU-PRO), while preserving multilingual capability.

---

## Quick Start

```bash
# 1. Clone and setup
cd "LLM Lobotomy"
pip install -r requirements.txt

# 2. Run the Lobotomy scan (Bayesian optimization, fast)
python run_lobotomy.py --model utter-project/EuroLLM-22B-Instruct-2512 \
  --mode bayesian --n-calls 200 --output results/

# 3. Run focused exhaustive sweep around the optimum
python run_lobotomy.py --model utter-project/EuroLLM-22B-Instruct-2512 \
  --mode sweep --i-range 15,40 --j-range 25,50 --output results/

# 4. Generate heatmaps
python -m lobotomy.heatmap --input results/scores/ --output results/heatmaps/

# 5. Apply the optimal Lobotomy and save
python -m lobotomy.surgeon --model utter-project/EuroLLM-22B-Instruct-2512 \
  --config "i,j" --output ./EuroLLM-22B-Lobotomy

# 6. Validate
lm_eval --model hf --model_args pretrained=./EuroLLM-22B-Lobotomy \
  --tasks ifeval,bbh,math_hard --batch_size auto
```

---

## Interpreting the Results

### Baseline Evaluation

Before any layer duplication, the unmodified model is evaluated first and printed as a reference banner:

```
============================================================
BASELINE RESULTS (unmodified model)
  Layers:       54 (path length: 54)
  Parameters:   22,637,328,384
  Math:         0.8087
  EQ:           0.8012
  Multilingual: 0.7307
  Combined:     2.3406
  Elapsed:      58.9s
============================================================
```

Every score delta shown in the heatmaps and log output is relative to this baseline. A Lobotomy only makes sense if it can beat this number.

---

### Per-Configuration Log Lines

During the sweep, each completed config prints one line:

```
[45/1486] Config (8, 14) — 6 duplicated layers
  → math=0.8234, eq=0.7991, ml=0.7102, combined=2.3327 | layers=60, params=24,512,049,152 (57.3s)
```

| Field | Meaning |
|---|---|
| `[45/1486]` | Progress counter — config number / total configs |
| `Config (8, 14)` | The `(i, j)` pair being tested |
| `6 duplicated layers` | Block size: `j - i` layers are repeated |
| `math=0.8234` | Math probe score, 0.0–1.0, higher is better |
| `eq=0.7991` | EQ probe score, 0.0–1.0, higher is better |
| `ml=0.7102` | Multilingual probe score, 0.0–1.0, higher is better |
| `combined=2.3327` | Sum of all three scores (max possible: 3.0) |
| `layers=60` | Total execution path: `54 original + 6 duplicated` |
| `params=24.5B` | Effective parameter count — original params + (6 × params per layer) |
| `57.3s` | Wall-clock time for this config |

Note: `params` counts duplicated layers again even though their weights are shared with the originals. This reflects the actual compute budget used, not the stored weight count.

---

### Best Configuration Summary

At the end of the sweep (or after a `Ctrl+C`), the overall best result is printed:

```
============================================================
BEST LOBOTOMY CONFIGURATION
  Config:     (0, 3)
  Dup layers: 3
  Combined:   2.3711 (baseline: 2.3406, Δ: +0.0305)
  Math:       0.767000
  EQ:         0.855000
  Multilingual: 0.749100
============================================================
To apply this configuration:
  python run_lobotomy.py apply --model MODEL --config 0,3 --output OUTPUT_DIR
```

A **positive Δ** means the Lobotomy improved on the unmodified model. A negative Δ is the worst-case result when no configuration beats the baseline — which typically means the sweep is not yet complete.

---

### The `sweep.csv` File

All results are written incrementally (one row per config, flushed to disk immediately) to `results/sweep.csv`. If the job crashes, all completed rows are preserved and the sweep resumes from where it left off on the next run.

| Column | Type | Description |
|---|---|---|
| `i` | int | Start layer of the duplicated block |
| `j` | int | End layer of the duplicated block (exclusive). `(0,0)` = baseline |
| `math_score` | float | Mean math probe score across all questions (0.0–1.0) |
| `eq_score` | float | Mean EQ probe score across all scenarios (0.0–1.0) |
| `multilingual_score` | float | Mean multilingual probe score (0.0–1.0) |
| `combined_score` | float | `math + eq + multilingual` (0.0–3.0) |
| `n_dup_layers` | int | Number of duplicated layers (`j - i`), 0 for baseline |
| `path_length` | int | Total execution path length (`n_layers + n_dup_layers`) |
| `total_params` | int | Effective parameter count including duplicated-layer contribution |
| `elapsed_sec` | float | Wall-clock seconds for this config's full evaluation |

---

### Probe Scores Explained

#### Math Score (0.0–1.0)

Tests numerical intuition by asking the model to answer hard arithmetic questions **without** chain-of-thought. The model must produce an answer in one shot:

```
System: You are a calculator. Answer with ONLY the number.
User:   What is the cube root of 74,088,893,247?
Model:  42049   (expected: 42049 → score ≈ 1.0)
Model:  44000   (wrong but close → partial credit ~0.7)
Model:  hello   (non-numeric → score 0.0)
```

Scored with **partial credit** based on digit-level closeness. An answer of `44000` vs `42049` is not binary-wrong — the model clearly "knows" the order of magnitude, so it receives partial credit proportional to how close the digits are. A score of **0.80** means the model's answers are typically within ~20% of the correct value.

#### EQ Score (0.0–1.0)

Tests emotional and social reasoning by asking the model to rate emotional intensities (0–100) for human scenarios:

```
Scenario: Alex discovers their closest friend secretly applied for the same dream job...
Rate these emotions (0-100): betrayal, anger, sadness, confusion

Expected: [78, 65, 72, 60]
Model:    [71, 58, 68, 55]   → MAE = 6.0 → score = 1 - 6/100 = 0.94
```

Scored as `1 − mean_absolute_error / 100`. A score of **0.80** means the model's emotion ratings are on average within 20 points of the expected values across all emotions and scenarios.

#### Multilingual Score (0.0–1.0)

The same math questions asked in French, German, Spanish, Portuguese, and Dutch — same scoring. This is a safety guard for EuroLLM's 35-language coverage: the optimal Lobotomy configuration must not damage multilingual circuits while improving reasoning.

#### Combined Score (0.0–3.0)

The sum `math + eq + multilingual`. The theoretical maximum is 3.0. For EuroLLM-22B the baseline is approximately **2.34**. The sweep optimises for this combined score. Improvements are typically small in absolute terms (+0.02 to +0.15) but consistent across multiple probe types.

---

### Reading the Heatmaps

Each heatmap visualises the delta vs baseline for every `(i, j)` configuration tested so far.

```
         End layer j →
         0   5   10  15  20  25  30  35  40  45  50  54
       ┌─────────────────────────────────────────────────
    0  │ ·   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■
    5  │     ·   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■
   10  │         ·   ■   ■   ■ 🔴 ■   ■   ■   ■   ■
   15  │             ·   ■   ■   ■   ■   ■   ■   ■
    ↑  │                 ·   ■   ■   ■   ■   ■
Start  │                     ·   ■   ■   ■
layer  │                         ·   ■
    i  │                             ·
```

**Cell colour:**

| Colour | Meaning |
|---|---|
| **Red** | This `(i, j)` improved the model above baseline (positive Δ) |
| **White** | No change from baseline (Δ ≈ 0) |
| **Blue** | This `(i, j)` degraded the model (negative Δ) |
| **Empty / grey** | Not yet evaluated (NaN) |

The colour scale is always **symmetric around zero** — white is locked to Δ=0 regardless of the data range, so even small positive values appear visibly red.

The **lime circle** marks the best-performing configuration found so far.

**Axes:**
- **X-axis** — End layer `j`: where the duplicated block ends
- **Y-axis** — Start layer `i`: where the duplicated block starts

The valid region is the lower-right triangle (`i < j`). Cells on and above the diagonal are always empty. **Cells near the diagonal** represent short blocks (few duplicated layers); **cells far from the diagonal** represent large blocks spanning many layers.

**What to look for:** When the full sweep is complete you typically see:
- A **red region** somewhere in the model's middle layers — these are the reasoning circuits
- **Blue regions** near the bottom (`i ≈ 0`, large `j`) — duplicating the encoding layers is harmful
- **Near-zero values** toward the top-right — duplicating late/decoding layers has little effect
- The optimal block is usually **5–12 layers wide** within the reasoning region

---

### Reading the Skyline Plots

The skyline shows **marginal averages** of the delta matrix — collapsing one axis to reveal which individual layer boundaries matter most.

**Left panel — By start layer (i):**

For each `i`, the average Δ across all valid `j` values. A tall **red bar at i=22** means: *on average, Lobotomies starting at layer 22 improve the model, regardless of where they end.* This identifies which layer is a good **entry point** into the reasoning circuit.

**Right panel — By end layer (j):**

For each `j`, the average Δ across all valid `i` values. A tall **red bar at j=35** means: *on average, Lobotomies ending at layer 35 improve the model.* This identifies which layer is a good **exit point** from the reasoning circuit.

**Bar colours match the heatmap convention:** red = net improvement, blue = net degradation.

Together, the peaks of the two panels bracket the **functional boundaries** of the reasoning circuit. The optimal `(i, j)` block typically falls between the peak of the left panel and the peak of the right panel.

> **Note during an ongoing sweep:** The skylines are computed only from configs evaluated so far. Early on (only a few rows of the heatmap complete) the left panel will have data only for small `i` values, and bars may appear as a single wide column. The pattern becomes interpretable once a substantial fraction of the sweep is done.

---

## References

- Ng, D. N. (2026). *LLM Neuroanatomy: How I Topped the LLM Leaderboard Without Changing a
  Single Weight*. [Blog post](https://dnhkng.github.io/posts/rys/)
- EuroLLM-22B-Instruct-2512.
  [HuggingFace](https://huggingface.co/utter-project/EuroLLM-22B-Instruct-2512)
- Papadimitriou, I., et al. (2025). *EuroLLM: Multilingual LLMs for the European Union*.
  [arXiv:2602.05879](https://arxiv.org/abs/2602.05879)
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian Optimization
  of Machine Learning Hyperparameters*. NeurIPS 2012.
  [arXiv:1206.2944](https://arxiv.org/abs/1206.2944) — theoretical basis for the
  Bayesian optimization sweep (Phase 4, Option B)
- Kim, D., et al. (2023). *SOLAR 10.7B: Scaling Large Language Models with Simple yet
  Effective Depth Up-Scaling*. [arXiv:2312.15166](https://arxiv.org/abs/2312.15166)
- He, B., et al. (2025). *The Curse of Depth in Large Language Models*.
  [arXiv:2502.05795](https://arxiv.org/abs/2502.05795)
- Geiping, J., et al. (2025). *Scaling up Test-Time Compute with Latent Reasoning:
  A Recurrent Depth Approach*. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
- HN discussion. [Show HN: How I topped the HuggingFace open LLM leaderboard on two
  gaming GPUs](https://news.ycombinator.com/item?id=47322887)

---

## Citation

```bibtex
@misc{lobotomy2026,
  title   = {LLM Lobotomy: Layer Duplication for EuroLLM-22B},
  year    = {2026},
  month   = {March},
  note    = {Based on RYS method by David Noel Ng},
}
```
