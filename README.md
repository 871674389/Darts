# DARTs

**A dual-path forecasting graph network for anomaly detection in high-dimensional,
strongly-coupled multivariate time series**, whose detection head adapts to the
nature of the data and is selectable **without any test-set leakage**.

> DARTs is designed for **high-dimensional, strongly-coupled** industrial systems.
> It reaches state-of-the-art threshold-free AUROC on the high-dimensional control
> benchmarks (SWaT, WADI, MSL) while keeping a parameter count that is **invariant
> to the number of sensors**. On the low-dimensional telemetry sets (SMD, PSM) it is
> deliberately *not* the target regime — this boundary is reported honestly.

> **Code ↔ paper naming.** In the source tree the model class is still named
> `DualGraphSTNet` (the method's original working name); **`DualGraphSTNet` is DARTs**.

---

## Key ideas

| | Contribution | One line |
|---|---|---|
| **C1** | **Dual-path forecasting graph framework** with a *fixed backbone + a detection head that adapts to the data*. We observe a trend that the best detection paradigm is largely determined by the normal-state nature of the data (periodic control → reconstruction/VAE head; telemetry → forecasting head). | one fixed backbone, a swappable head, and the swap follows a pattern |
| **C2** | **Leakage-free head selection.** A synthetic-anomaly validation set is built by injecting anomalies into the *training* data only; the correct head is chosen fully unsupervised, with zero test-set leakage. | pick the right head without ever touching test |
| **C3** | **Broad generalization + honest evaluation.** Leading threshold-free AUROC across high-dimensional benchmarks; we also show that point-adjusted F1 is *gameable* and therefore report threshold-free metrics as the primary protocol. | SOTA on high-dim, with honest metrics |
| **C4** | **Dimension-invariant lightweight design.** The parameter count (~0.46 M) does not grow with the number of sensors; Pareto-optimal accuracy–efficiency. | high-dim without parameter blow-up |

### Architecture at a glance
Two paths feed a shared forecasting objective over the next window:
- **Window path** — a multi-head sparse graph is learned over sensors (Gumbel-Softmax
  edge sampling with a categorical head prior) and diffused with a DCGRU.
- **History path** — a dot-product attention graph with temporal decay is diffused
  with a 2-hop message-passing network.

The two path outputs are fused and the **detection head** turns them into an
anomaly signal. The head is selectable:
- **forecasting head** (default) — Gaussian prediction error of the next window;
- **VAE / reconstruction head** (`DUALV4_VAE=1`).

---

## Repository layout

```
darts-release/
├── train_dualpath.py          # train DARTs (env-var driven; see below)
├── test_dualpath.py           # evaluate a trained model (auto-locates weights/scores)
├── modules.py  profiling.py   # shared building blocks + memory/throughput profiling
├── model/                     # DualGraphSTNet (= DARTs) + ablation variants
│   ├── DualGraphSTNet_dualshortpath.py   # ← the main dual-path model used for training/eval
│   ├── DualGraphSTNet.py                  # base variant
│   └── DualGraphSTNet_{single,singleshortpath,only2,only12,graphlearner}.py  # ablations
├── lib/                       # utils (windowing, Gumbel, KL), evaluate, SPOT/POT, prediction
├── config_files/              # one config per dataset + env-var driven config_general
├── scripts/
│   ├── data_prep/             # dataset conversion/cleaning helpers
│   ├── synth_val/             # leakage-free synthetic-validation construction (C2)
│   ├── validation/            # head/epoch selection on synthetic validation
│   ├── analysis/              # AUROC/AP, param counts, tables, multi-seed std
│   └── cluster/               # example multi-GPU job scripts (environment-specific)
├── docs/                      # supplementary notes (efficiency, generalization, PA-F1 gameability)
├── data/                      # datasets are NOT shipped — see data/README.md
├── requirements.txt  LICENSE  CITATION.cff
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```
Tested with Python 3.9 and PyTorch ≥ 1.12 (CUDA recommended).

## Data

Datasets are public but not included. Download them and lay them out as described
in [`data/README.md`](data/README.md).

---

## Quick start

All runtime options are driven by environment variables, so the same scripts
cover every dataset and ablation. **Run from the repository root.**

### Train
```bash
# DARTs forecasting head on WADI (default dataset), GPU 0
DUALV4_DS=WADI DUALV4_DEVICE=cuda:0 python train_dualpath.py

# VAE / reconstruction head on SWaT
DUALV4_DS=SWAT DUALV4_VAE=1 DUALV4_NOKL=1 python train_dualpath.py

# one SMD machine
DUALV4_DS=SMD DUALV4_SMD_MACHINE=machine-1-1 python train_dualpath.py
```
Checkpoints go to `experiments/<ds>/` and per-epoch anomaly scores to `output/<ds>/`
(use `DUALV4_TAG=<name>` to isolate runs in sub-folders).

### Evaluate
```bash
# auto-locates the newest checkpoint + train-score file for the chosen dataset
DUALV4_DS=WADI python test_dualpath.py
# or pin them explicitly:
DUALV4_DS=WADI python test_dualpath.py --params experiments/wadi/epoch_XX.params \
                                       --scores output/wadi/epoch_XX/anomaly_scores.pkl
```

### Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `DUALV4_DS` | dataset: `WADI` / `SWAT` / `MSL` / `SMAP` / `SMD` / `PSM` / `ASD` | `WADI` |
| `DUALV4_DEVICE` | `cuda:0`, `cuda:1`, `cpu` | from config |
| `DUALV4_SEED` | random seed | `42` |
| `DUALV4_EPOCHS` | number of epochs | from config |
| `DUALV4_TAG` | sub-folder tag to isolate experiment artefacts | — |
| `DUALV4_VAE` | `1` → VAE/reconstruction head (instead of forecasting) | off |
| `DUALV4_NOKL` | `1` → drop the KL graph-prior term (ablation) | off |
| `DUALV4_JOINT` | `1` → joint forecast+reconstruction head | off |
| `DUALV4_WINDOW` / `DUALV4_PREVLEN` / `DUALV4_DIFFSTEP` / `DUALV4_BATCH` | override window size / history length / diffusion step / batch size | from config |
| `DUALV4_SMD_MACHINE` | which SMD machine sub-folder to use | `machine-1-1` |

---

## Evaluation protocols

DARTs reports **two** protocols; the paper's headline numbers use the first.

1. **Threshold-free (primary).** AUROC / AUPR on the continuous anomaly score — no
   threshold, no label-dependent tuning. This is the metric we use for the main
   comparison.
2. **Point-adjusted best-F1 (secondary, for comparability).** `test_dualpath.py`
   performs a 2-D threshold grid search and reports the best point-adjusted F1.
   This protocol *uses the test labels to pick the threshold* and is known to be
   gameable; we include it only because prior work does, and we discuss its
   pitfalls in [`docs/pa_f1_gameability.md`](docs/pa_f1_gameability.md). The
   script labels its own output `"protocol": "best-F1 grid + point-adjust"`.

Supporting analyses (threshold-free AUROC/AP, parameter counts, multi-seed std,
result tables) live in `scripts/analysis/`.

---

## Results (from the paper, threshold-free AUROC)

| Dataset | Dim | Nature | Best head | DARTs AUROC | In-target? |
|---|---|---|---|---|---|
| SWaT | 51  | periodic control | reconstruction | **0.840 ± 0.00** | ✅ SOTA |
| WADI | 118 | high-dim control  | reconstruction | **0.733 ± 0.02** | ✅ SOTA |
| MSL  | 55  | telemetry         | forecasting    | **0.647 ± 0.01** | ✅ SOTA |
| SMD  | 38  | telemetry (28 mc) | forecasting    | 0.726 ± 0.11 | out-of-target |
| PSM  | 25  | telemetry         | reconstruction | 0.596 | out-of-target |

High-dimensional control sets (≥ 51 sensors) are the target regime and DARTs
leads there; the low-dimensional telemetry sets are reported as an honest
boundary. Parameter count is **~0.46 M and invariant to dimensionality**.
(Numbers are the authors' reported values; see the paper for full tables and baselines.)

---

## Reproducing the paper

- **Head-swap / leakage-free selection (C1, C2):** build the synthetic validation
  set with `scripts/synth_val/`, then select the head/epoch with
  `scripts/validation/`.
- **Main comparison (C3):** train per dataset, evaluate threshold-free with
  `scripts/analysis/compute_auroc.py`; aggregate seeds with
  `scripts/analysis/seed_std.py`.
- **Efficiency (C4):** `scripts/analysis/count_params.py` and
  `docs/efficiency.md`.
- **Example cluster jobs:** `scripts/cluster/` (adapt paths/conda env to your machine).

---

## Citation

```bibtex
@article{liu2026darts,
  title   = {DARTs: A Dual-Path Forecasting Graph Network for Anomaly Detection
             in High-Dimensional, Strongly-Coupled Multivariate Time Series},
  author  = {Liu, Sherry},
  year    = {2026}
}
```

## Acknowledgements

`lib/spot.py` is an independent, MIT-licensed re-implementation of the SPOT
algorithm (Siffer et al., KDD 2017) built on SciPy's Generalized Pareto fit;
it follows only the published method and contains no third-party code. Please
still cite the original SPOT paper if you use this thresholding. Benchmarks
(SWaT, WADI, SMAP, MSL, SMD, PSM) belong to their respective providers; see
`data/README.md`.

## License

MIT — see [LICENSE](LICENSE).
