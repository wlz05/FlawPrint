# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

**FlawPrint** detects multimodal fake news through the dependence between observable image and narrative traces. Domain changes can alter the magnitude of individual traces while preserving relations within a shared acquisition and transmission process. FlawPrint estimates a domain reference, removes marginal scale through rank normalization, and measures departures from its joint structure. A frozen semantic detector supplies complementary evidence about news meaning.

---

## Architecture

```text
News image and text                         Independent domain reference pool
        |                                                   |
Deterministic trace extraction                 The same extraction and schema
        |                                                   |
20 image traces + 6 narrative traces              Rank marginals and copula
        |                                                   |
        +-------------- Frozen rank transformation ---------+
                                |
                   Copula residual and local sign evidence
                                |
Frozen semantic news score ---> Evidence routing and calibration
                                |
                    Fake-news risk and prediction set
```

### Key Contributions

- **A relation-centered detection criterion.** FlawPrint evaluates the joint organization of traces against a domain reference, addressing changes that individual trace magnitudes cannot characterize.
- **A domain-specific copula residual.** Frozen empirical marginals and a regularized dependence matrix define the physical score; local partial-correlation signs expose which trace relations disagree with the reference.
- **Inference and calibration.** Physical and semantic scores are combined according to the available evidence, with probabilities and prediction sets calibrated on separate source partitions.

## Project Structure

```text
FlawPrint/
|-- main.py                  # Preparation, profiling, inference, and evaluation.
|-- convert.py               # Dataset conversion.
|-- doctor.py                # Dependency and environment inspection.
|-- encode.py                # Optional local CLIP alignment control.
|-- study.py                 # Threshold, propagation, cold-start, and cost studies.
|-- research.py              # Paired, contamination, and channel comparisons.
|-- download.py              # Download reference papers from recorded sources.
|-- test.py                  # Regression test runner.
|-- engine/                  # Trace, copula, calibration, and evaluation modules.
|-- tests/                   # Data, mathematical, leakage, and workflow checks.
|-- configs/                 # Model settings and dataset schemas.
|-- requirements.txt         # Core dependencies.
|-- requirements-optional.txt
|-- tables.txt               # Optional spreadsheet and Parquet readers.
|-- README.md
```

## Requirements

| Component | Version / setting |
|---|---|
| Python | 3.10 or newer |
| NumPy | >=1.26, <3 |
| SciPy | >=1.11, <2 |
| Pillow | >=10, <13 |
| scikit-learn | >=1.4, <2 |
| Requests | >=2.31, <3 |
| pypdf | >=4, <7 |
| Optional semantic inference | PyTorch >=2.2; Transformers >=4.40, <5 |
| GPU | Single NVIDIA GeForce RTX 4090 |
| Experiment seeds | **42, 52, 3047, 3407** |

Trace extraction and copula profiling run on CPU. Optional semantic inference supports CUDA.

### Installation

```bash
git clone https://github.com/wlz05/FlawPrint.git
cd FlawPrint
python -m venv .venv
```

Activate the environment on your platform:

```powershell
.venv/Scripts/Activate.ps1
```

```bash
source .venv/bin/activate
```

Install and inspect the core environment:

```bash
python -m pip install -r ./requirements.txt
python -B doctor.py --output ./outputs/environment.json --freeze ./outputs/versions.txt
```

For Excel, Parquet, and legacy spreadsheet input:

```bash
python -m pip install -r ./tables.txt
python -B doctor.py --tables
```

For optional local CLIP inference, install the PyTorch build appropriate for your CUDA environment and then the optional requirements:

```bash
python -m pip install -r ./requirements-optional.txt
python -B doctor.py --semantic
```

## Quick Start

### 1. Verify the installation

```bash
python -B main.py --help
python -B test.py
```

### 2. Run a synthetic example

```bash
python -B main.py fixture --seed 42 --per-split 24 --output ./data/fixture.npz
python -B main.py benchmark ./data/fixture.npz --config ./configs/quick.json --bootstrap 25 --ablations --output ./outputs/smoke
```

### 3. Prepare data

Place your local dataset tables and images under `./data`. For canonical input:

```bash
python -B main.py prepare ./data/raw.jsonl --labels fake-one --output ./data/manifest.jsonl
python -B main.py split ./data/manifest.jsonl --target-domain politics --seed 42 --output ./data/split.jsonl
python -B main.py validate ./data/split.jsonl --benchmark --check-images --output ./outputs/preflight.json
python -B main.py extract ./data/split.jsonl --workers 1 --output ./data/features.npz
```

Replace `politics` with the held-out domain. For datasets with predefined splits, skip `split` and use the original partitions. Each evaluated domain needs an independent reference pool.

For dataset-specific columns and label conversion:

```bash
python -B convert.py ./data/raw.csv --mapping ./configs/conversion.json --output ./data/converted.jsonl --report ./outputs/conversion.json
python -B convert.py ./data/finefake.json --dataset finefake --image-root ./data/images --output ./data/finefake.jsonl
```

Set the column names in `configs/conversion.json` to match your input table. The FineFake preset converts its original labels to 0 = real and 1 = fake; `--label-map` supports other class mappings. Supported formats include JSONL, NDJSON, JSON, CSV, TSV, Excel, and Parquet. Excel and Parquet require the optional table dependencies. Use `--trusted-pickle` only with trusted local pickle files.

### 4. Supply frozen semantic predictions

Provide one finite logit per sample, with higher values indicating fake news:

```json
{"id":"news001","logit":1.25}
```

Join scores by exact sample identity before extracting the feature bundle:

```bash
python -B main.py join ./data/split.jsonl ./data/semantic.jsonl --score-key logit --output ./data/semantic_manifest.jsonl
python -B main.py extract ./data/semantic_manifest.jsonl --output ./data/semantic_features.npz
```

Use `--real-positive` for scores that increase toward real news. External models are trained on the source partitions and frozen before target inference.

### 5. Run the four-seed benchmark

```bash
python -B main.py benchmark ./data/semantic_features.npz --config ./configs/seed42.json --ablations --output ./outputs/seed42
python -B main.py benchmark ./data/semantic_features.npz --config ./configs/seed52.json --ablations --output ./outputs/seed52
python -B main.py benchmark ./data/semantic_features.npz --config ./configs/seed3047.json --ablations --output ./outputs/seed3047
python -B main.py benchmark ./data/semantic_features.npz --config ./configs/seed3407.json --ablations --output ./outputs/seed3407
```

The four configurations differ only in `seed` and use the same data partitions. When retraining the semantic backbone across seeds, join and extract each run's predictions separately, then pass the matching feature bundle to the benchmark.

Each output directory contains:

| File / directory | Content |
|---|---|
| `report.json` | Metrics, configuration, profile quality, and skipped methods |
| `summary.csv` | AUC, macro F1, accuracy, calibration, domain metrics, and timing |
| `predictions/` | Per-sample predictions for each method |
| `profiles/` | Frozen domain profiles |
| `calibration.json` | Probability and prediction-set calibration states |
| `provenance.json` | Source, input, environment, schema, and configuration fingerprints |

Report mean and sample standard deviation across the four seeds. Use the full seed configurations for experiments; `quick.json` uses fewer bootstrap repetitions and a smaller reference pool.

## Dataset Format & Split Provenance

The canonical label direction is **0 = real, 1 = fake**. Paths are relative to the repository working directory, including image paths stored inside a manifest. Run all commands from the repository root.

```json
{"id":"news001","image":"./data/images/news001.jpg","text":"Example news text.","domain":"politics","platform":"weibo","group":"event001","split":"test","label":1}
```

| Field | Description |
|---|---|
| `id` | Unique, stable string; shared by tables and external scores |
| `image` | Relative local image path, or an empty string for text-only input |
| `text` | News text |
| `domain` | Dataset-defined domain; consistent across partitions |
| `platform` | Acquisition or platform identity when available |
| `group` | Event or duplicate group used for disjointness checks |
| `split` | `train`, `validation`, `calibration`, `profile`, or `test` |
| `label` | Binary news-veracity label; absent from target profile rows |
| `semantic_logit` | Optional finite score increasing toward fake news |

See [configs/schema.json](configs/schema.json) for the row schema and [configs/conversion.json](configs/conversion.json) for column mappings. Adapt the example in [configs/labels.json](configs/labels.json) to your dataset's class definitions.

The source training set fits the comparison heads; validation and calibration sets fit probabilities and prediction sets, respectively. The unlabeled target pool estimates the domain reference. Test labels are used for evaluation. Keep events and duplicate images disjoint across these partitions.

Image manipulation and news veracity are separate attributes. A genuine image paired with false text can require semantic evidence even when the physical reference is satisfied. Missing manipulation attributes remain unknown in stratified analysis.

## Baselines

The built-in benchmark includes the following runnable controls under the same supplied splits:

| Identifier | Comparison role |
|---|---|
| `trace_logistic` | Supervised linear classifier on trace magnitudes |
| `marginal_nb` | Independent univariate trace likelihoods |
| `single_trace` | One trace selected using source validation |
| `text_tfidf` | TF-IDF text classification |
| `gaussian_novelty` | Full Gaussian novelty in trace space |
| `semantic_external` | The frozen external semantic detector |
| `trace_rbf` | Nonlinear RBF control on all traces |
| `image_rbf` | Nonlinear RBF control on image traces |
| `trace_rbf_fusion` | All-trace RBF evidence with the same semantic scores |
| `image_rbf_fusion` | Image RBF evidence with the same semantic scores |

FlawPrint variants include `copula_only`, `sign_only`, `physical_full`, and `flawprint_fusion`. The fused variant runs when complete external semantic scores are supplied. `--ablations` additionally evaluates trace families, sign priors, sign weights, and source-profile transfer. Methods with unavailable required inputs are recorded in `skipped`.

Use the authors' implementations for DAPT, DAMMFND, and other published baselines. `research.py channels` compares semantic, physical, and image-forensics scores, including externally supplied ReAlign outputs. Run `python -B research.py channels --help` for the input format and model metadata.

## Implementation Details

| Setting | Default |
|---|---|
| Resized image longest side | 768 pixels |
| Minimum image side | 32 pixels |
| Local image grid | 4 x 4 |
| Global trace schema | 20 image + 6 narrative traces |
| Global / local shrinkage | 0.1 / 0.35 |
| Reference bootstrap repetitions | 100 |
| Minimum reference samples | 20 |
| Minimum observed feature fraction | 0.8 |
| Sign tolerance / weight | 0.03 / 1.0 |
| Profile separation diagnostic `kappa` | 2.0 |
| Optional selector fake-probability threshold | 0.05 |
| Probability / prediction-set calibration | Separate validation / calibration partitions |
| Benchmark bootstrap repetitions | 200 |
| Benchmark prediction-set `alpha` | 0.1 |
| Default random seed | 42 |
| Full experiment seeds | 42, 52, 3047, 3407 |
| GPU | Single RTX 4090 |

Original image dimensions, EXIF, and JPEG quantization metadata are read before resizing. Missing and constant columns are handled explicitly, and the retained feature order is stored in each profile. The domain reference stores empirical marginals, a regularized copula correlation matrix, local sign weights, fitting identities, and diagnostics.

Domain profiles are fitted without target labels or gradient updates. An optional source-trained selector filters the reference pool using frozen confidence scores. `research.py filtering` measures its behavior as pool contamination changes.

## Inference & Calibration

Fit an independent target profile and inspect its quality:

```bash
python -B main.py profile ./data/semantic_features.npz --domain politics --split profile --config ./configs/seed42.json --output ./models/politics.json
python -B main.py predict ./data/semantic_features.npz --profile ./models/politics.json --domain politics --split test --fusion --explain --output ./outputs/predictions.jsonl
python -B main.py evaluate ./outputs/predictions.jsonl --seed 42 --output ./outputs/evaluation.json
```

`predict` exports raw scores; `--calibration` applies a saved calibration state. `copula_score` and `physical_evidence` increase toward the reference; `physical_risk`, `semantic_logit`, and `fusion_risk` increase toward fake news. Integration uses the larger available risk after the configured semantic threshold shift.

For deployment routing:

```bash
python -B main.py deploy ./data/semantic_features.npz --profile ./models/politics.json --domain politics --split test --explain --output ./outputs/deployment/predictions.jsonl --report ./outputs/deployment/report.json
```

`deploy` selects physical, semantic, fused, or abstaining output and reports coverage over all input samples. For route-specific calibration, run `deploy` on the source validation and calibration sets with their corresponding profiles, collect the predictions into separate files, and fit:

```bash
python -B main.py calibrate --validation ./outputs/source_validation.jsonl --calibration ./outputs/source_calibration.jsonl --routed --output ./models/routes.json
python -B main.py deploy ./data/semantic_features.npz --profile ./models/politics.json --domain politics --calibration ./models/routes.json --output ./outputs/calibrated/predictions.jsonl --report ./outputs/calibrated/report.json
```

## Arguments

| Entry point | Main arguments |
|---|---|
| `main.py extract` | `--max-side`, `--tile-grid`, `--min-side`, `--workers` |
| `main.py profile` | `--domain`, `--split`, `--config`, `--genuine-validation` |
| `main.py profile` selection | `--selector-predictions`, `--selector-metadata`, `--selector-threshold` |
| `main.py benchmark` | `--config`, `--alpha`, `--bootstrap`, `--ablations`, `--output` |
| `main.py predict` | `--profile`, `--domain`, `--split`, `--fusion`, `--calibration` |
| `main.py deploy` | `--min-observed`, `--min-image-features`, `--gate-alpha`, `--calibration` |
| `main.py evaluate` | `--score-key`, `--group-key`, `--bootstrap`, `--seed` |
| `convert.py` | `--dataset`, `--mapping`, `--label-map`, `--split-map`, `--images` |
| `research.py` | `compare`, `strata`, `mixture`, `mechanism`, `orientation`, `quotient`, `filtering`, `channels` |
| `study.py` | `cold-start`, `thresholds`, `propagation`, `latency` |

Each entry point and subcommand provides `--help` with the complete argument list.

## Optional Models & Analysis

`encode.py` loads a local CLIP directory without downloading model files during inference:

```bash
python -B encode.py ./data/split.jsonl --model ./models/clip --device cuda:0 --batch-size 8 --output ./data/clip_predictions.jsonl
```

CLIP outputs negative text-image similarity. The semantic news-veracity channel takes predictions from a frozen news detector.

Additional analyses:

```bash
python -B main.py theory --seed 42 --output ./outputs/theory.json
python -B main.py invariance --seed 42 --output ./outputs/invariance.json
python -B research.py compare ./outputs/seed42/predictions/semantic_external.jsonl ./outputs/seed42/predictions/flawprint_fusion.jsonl --scope real --dataset yourdataset --seed 42 --output ./outputs/comparison.json
python -B download.py --sources ./configs/sources.json --output ./references
```

`download.py` retrieves the reference papers listed in `configs/sources.json`.
