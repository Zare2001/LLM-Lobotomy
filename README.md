# LLM Lobotomy — Snellius Runbook

Complete step-by-step guide to running the Lobotomy pipeline on the
[Snellius supercomputer](https://www.surf.nl/en/services/snellius-the-national-supercomputer)
(SURF, Netherlands).

---

## Step 0: Prerequisites

**What you need before anything:**

- Snellius account with GPU budget hours (apply via [SURF](https://www.surf.nl/en/research-it/apply-for-computing-time))
- ~48 GPU-hours for the full sweep (same total regardless of parallelization)
- ~5 GPU-hours for phases 2–5
- `uv` package manager (already installed on Snellius)
- This project repo on your local machine

**Estimated budget: ~55 GPU-hours total on H100**

---

## Step 1: Upload code to Snellius

From your local machine:

```bash
scp -r "LLM Lobotomy" user@snellius.surf.nl:~/lobotomy
```

---

## Step 2: Setup environment on Snellius

SSH in and set up the venv with `uv`:

```bash
ssh snellius

cd ~/lobotomy

# Create venv (skip if .venv already exists)
uv venv --python=3.12

# Load modules
module load 2025
module load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

# Activate and install deps
source .venv/bin/activate
uv pip install -r requirements.txt

# Download the model (~45 GB)
uv pip install huggingface-hub
mkdir -p models/EuroLLM-22B-Instruct-2512
huggingface-cli download utter-project/EuroLLM-22B-Instruct-2512 \
    --local-dir models/EuroLLM-22B-Instruct-2512 \
    --local-dir-use-symlinks False

# Create output directories
mkdir -p results/scores results/heatmaps slurm/logs
```

Or run the setup script which does all of the above:

```bash
bash slurm/setup_snellius.sh
```

Edit your budget account in all `.sbatch` files:

```bash
# Uncomment and set in each .sbatch file:
#SBATCH --account=your_project_account
```

**Expected output:** Model downloaded to `~/lobotomy/models/EuroLLM-22B-Instruct-2512`

---

## Step 3: Run the full heatmap sweep

This is the expensive part. 1486 configurations, embarrassingly parallel.

```bash
cd ~/lobotomy

# Submit 32 parallel GPU jobs
sbatch slurm/sweep_array.sbatch
```

**Monitor:**

```bash
squeue -u $USER                                          # See running jobs
sacct -j JOBID --format=JobID,State,Elapsed              # Check status
tail -f slurm/logs/sweep_*_0.out                         # Watch first task
```

**Wall time:** ~1.5 hours with 32 H100s

**Output:** 32 CSV files in `results/scores/sweep_task_*.csv`

---

## Step 4: Merge results and generate heatmaps

After **all** array tasks show `COMPLETED` in `sacct`:

```bash
python slurm/merge_results.py \
    --input-dir results/scores \
    --output results \
    --n-layers 54
```

**Output:**

| File | Content |
|---|---|
| `results/sweep_merged.csv` | All 1486 configs in one file |
| `results/heatmaps/heatmap_math_score.png` | Math probe heatmap |
| `results/heatmaps/heatmap_eq_score.png` | EQ probe heatmap |
| `results/heatmaps/heatmap_combined_score.png` | Combined heatmap (the main one) |
| `results/heatmaps/skyline_*.png` | Row/column marginal averages |

The script also prints the best `(i*, j*)` configuration to the terminal.

**Download the heatmaps to look at them locally:**

```bash
scp -r user@snellius.surf.nl:~/lobotomy/results/heatmaps ./
```

---

## Step 5: Decision point — read the heatmaps

This is the human step. Look at the heatmaps and identify:

1. **The optimal block** `(i*, j*)` — the green dot on the combined heatmap
2. **Circuit boundaries** — sharp edges in the skyline plots where improvement drops off
3. **Are there multiple promising regions?** — if yes, multi-circuit is worth testing
   (Step 7)
4. **How wide is the optimal region?** — narrow = one tight circuit, wide = multiple
   overlapping circuits

Example outcomes (hypothetical for EuroLLM-22B):

- "The best config is (25, 32), 7 duplicated layers, +3.5% combined"
- "There's a second promising region around (38, 44)"
- "The skyline shows the decoding boundary at layer ~46"

---

## Step 6: Test N-pass looping on the optimal block

Test whether repeating the winning block 3, 4, or 5 times is better than 2. This is a
tiny experiment — just a few configs.

```bash
# Interactive GPU session for quick testing
salloc -p gpu_h100 --gpus-per-node=1 --time=02:00:00

# Inside the interactive session:
module load 2025 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0
source ~/lobotomy/.venv/bin/activate
cd ~/lobotomy

python3 -c "
from lobotomy.scanner import LobotomyScanner, build_looped_layer_path
from lobotomy.probes import MathProbe, EQProbe

scanner = LobotomyScanner(
    'models/EuroLLM-22B-Instruct-2512',
    load_in_4bit=True,
)
math_probe = MathProbe('data/math_probes.json')
eq_probe = EQProbe('data/eq_probes.json')

# *** Replace 25, 32 with YOUR optimal (i*, j*) from Step 5 ***
I_STAR, J_STAR = 25, 32

baseline_m = math_probe.run(scanner).mean_score
baseline_e = eq_probe.run(scanner).mean_score
print(f'Baseline:     math={baseline_m:.4f}  eq={baseline_e:.4f}  combined={baseline_m+baseline_e:.4f}')

for n in [2, 3, 4, 5]:
    path = build_looped_layer_path(scanner.n_layers, I_STAR, J_STAR, n_repeats=n)
    with scanner.custom_path(path):
        m = math_probe.run(scanner).mean_score
        e = eq_probe.run(scanner).mean_score
    print(f'n_repeats={n}:  math={m:.4f}  eq={e:.4f}  combined={m+e:.4f}  layers={len(path)}')
"
```

**Expected output:** A table showing which repeat count is best. Typically `n=2` or `n=3`
wins; `n=5` usually degrades.

**Decision:** Pick the best `n_repeats`.

---

## Step 7: Test multi-circuit (if heatmap shows two+ promising regions)

**Skip this if** the heatmap only shows one good region.

If you saw a second promising block (e.g., `(38, 44)` in addition to `(25, 32)`):

```python
# Still in the interactive session from Step 6
from lobotomy.scanner import build_multi_circuit_path

for label, circuits in [
    ('Circuit A only', [(25, 32)]),
    ('Circuit B only', [(38, 44)]),
    ('Both circuits',  [(25, 32), (38, 44)]),
]:
    path = build_multi_circuit_path(scanner.n_layers, circuits)
    with scanner.custom_path(path):
        m = math_probe.run(scanner).mean_score
        e = eq_probe.run(scanner).mean_score
    print(f'{label:20s}  math={m:.4f}  eq={e:.4f}  combined={m+e:.4f}  layers={len(path)}')
```

**Decision:** Does combining circuits help, hurt, or make no difference?

---

## Step 8: Apply the winning configuration and save the model

Now you know the best config. Apply it and save a standalone model.

```bash
python run_lobotomy.py apply \
    --model models/EuroLLM-22B-Instruct-2512 \
    --config "25,32" \
    --output ~/lobotomy/EuroLLM-22B-Lobotomy
```

**Output:** A full HuggingFace model directory at `EuroLLM-22B-Lobotomy/` with physically
duplicated layers. This model works with any standard inference framework — no special code
needed.

---

## Step 9: Validate on benchmarks

Run the saved model against standard benchmarks to confirm the improvement generalises
beyond the probes.

```bash
uv pip install lm-eval

sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=lobotomy-eval
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/eval_%j.out

module purge && module load 2025 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0
source ~/lobotomy/.venv/bin/activate
cd ~/lobotomy

lm_eval --model hf \
    --model_args pretrained=./EuroLLM-22B-Lobotomy,dtype=bfloat16 \
    --tasks ifeval,bbh,gpqa,musr,mmlu_pro \
    --batch_size auto \
    --output_path results/benchmarks/
EOF
```

**Compare against baseline** (run the same benchmarks on the original model):

```bash
lm_eval --model hf \
    --model_args pretrained=./models/EuroLLM-22B-Instruct-2512,dtype=bfloat16 \
    --tasks ifeval,bbh,gpqa,musr,mmlu_pro \
    --batch_size auto \
    --output_path results/benchmarks_baseline/
```

---

## Step 10: Upload to HuggingFace (optional)

If the benchmarks confirm improvement:

```bash
huggingface-cli login
huggingface-cli upload your-username/EuroLLM-22B-Lobotomy ./EuroLLM-22B-Lobotomy .
```

---

## Summary Timeline

| Step | What | Where | Time | GPU-hours |
|---|---|---|---|---|
| 0 | Get Snellius access | SURF portal | Days/weeks | 0 |
| 1 | Upload code | Local → Snellius | 5 min | 0 |
| 2 | Setup environment + download model | Login node | ~30 min | 0 |
| 3 | **Full heatmap sweep** | 32× H100 array | **~1.5h wall** | **~48** |
| 4 | Merge + generate heatmaps | Login node | 2 min | 0 |
| 5 | Read heatmaps, pick (i*, j*) | Your eyes | 15 min | 0 |
| 6 | Test n_repeats 2–5 | 1× H100 interactive | ~30 min | 0.5 |
| 7 | Test multi-circuit (maybe) | 1× H100 interactive | ~15 min | 0.25 |
| 8 | Save final model | CPU or GPU | ~20 min | 0.3 |
| 9 | Benchmark validation | 1× H100 | ~6h | 6 |
| 10 | Upload to HuggingFace | Login node | 10 min | 0 |
| | **Total** | | **~2 days including queue** | **~55** |

Steps 3–4 produce the heatmap. Steps 5–7 are the science — interpreting results and testing
advanced methods. Steps 8–10 are packaging. The whole thing is doable in a single day of
wall time if the queue is fast.
