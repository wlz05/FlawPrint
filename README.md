# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

**FlawPrint** detects multimodal fake news through the dependence between observable image and narrative traces. Domain changes can alter the magnitude of individual traces while preserving relations within a shared acquisition and transmission process. FlawPrint estimates a domain reference, removes marginal scale through rank normalization, and measures departures from its joint structure. A frozen semantic detector supplies complementary evidence about news meaning.

The repository provides deterministic trace extraction, copula profiling, local sign evidence, semantic score integration, deployment routing, ten runnable comparison controls, and controlled evaluation tools.

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
- **A complete evidence workflow.** Independent profiling, source calibration, semantic integration, and coverage-aware routing connect the score to reproducible news detection and controlled analyses.

## Project Structure

```text
FlawPrint/
|-- main.py                  # Preparation, profiling, inference, and evaluation.
|-- convert.py               # Audited conversion of dataset tables.
|-- doctor.py                # Dependency and environment inspection.
|-- encode.py                # Optional local CLIP alignment control.
|-- study.py                 # Threshold, propagation, cold-start, and cost studies.
|-- research.py              # Paired, contamination, and channel comparisons.
|-- download.py              # Download reference papers from recorded sources.
|-- test.py                  # Regression test runner.
|-- engine/                  # Trace, copula, calibration, and evaluation modules.
|-- tests/                   # Data, mathematical, leakage, and workflow checks.
|-- configs/                 # Method defaults, seed profiles, and data contracts.
|-- requirements.txt         # Core CPU workflow.
|-- requirements-optional.txt
|-- tables.txt               # Optional spreadsheet and Parquet readers.
|-- README.md
```

Data, checkpoints, references, and generated results are created locally. All Python filenames are single lowercase English words. The modules use Python namespace packages and require no `__init__.py` files.

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
| Author's experiment GPU | Single NVIDIA GeForce RTX 4090 |
| Experiment seeds | **42, 52, 3047, 3407** |

The physical workflow runs on CPU. The single GPU is used by the optional semantic model or external baseline implementation. Record the actual software versions, model artifacts, and device settings with each run.

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
python -B main.py audit --output ./outputs/audit.json
```

The public audit reads this README when the local manuscript is absent. `--manuscript` accepts an explicit relative document. `--accept` writes a snapshot only after the code audit passes.

### 2. Run the synthetic smoke workflow

```bash
python -B main.py fixture --seed 42 --per-split 24 --output ./data/fixture.npz
python -B main.py benchmark ./data/fixture.npz --config ./configs/quick.json --bootstrap 25 --ablations --output ./outputs/smoke
```

This checks the full profiling, calibration, control, and reporting path using artificial features. Synthetic outputs describe implementation behavior; real benchmark measurements require the corresponding datasets and frozen semantic predictions.

### 3. Prepare a real dataset

Place your local dataset tables and images under `./data`. For canonical input:

```bash
python -B main.py prepare ./data/raw.jsonl --labels fake-one --output ./data/manifest.jsonl
python -B main.py split ./data/manifest.jsonl --target-domain politics --seed 42 --output ./data/split.jsonl
python -B main.py validate ./data/split.jsonl --benchmark --check-images --output ./outputs/preflight.json
python -B main.py extract ./data/split.jsonl --workers 1 --output ./data/features.npz
```

Replace `politics` with the actual held-out domain. If an official experiment already supplies split identities, preserve those identities and skip `split`. The source train, validation, and calibration partitions must contain the data needed by the selected controls, while every scored domain needs an independent profile source.

For dataset-specific columns and label conversion:

```bash
python -B convert.py ./data/raw.csv --mapping ./configs/conversion.json --output ./data/converted.jsonl --report ./outputs/conversion.json
python -B convert.py ./data/finefake.json --dataset finefake --image-root ./data/images --output ./data/finefake.jsonl
```

Edit `configs/conversion.json` to match the actual input columns before using it. The FineFake preset reverses its original numeric label convention into the canonical direction. Use `--label-map` for a dataset-specific class mapping, and inspect the conversion report before extraction. JSONL, NDJSON, JSON, CSV, and TSV are supported directly; optional readers handle Excel and Parquet. Pickle input requires `--trusted-pickle` and a trusted local file.

### 4. Supply frozen semantic predictions

Provide one finite, fake-positive logit for every required sample:

```json
{"id":"news001","logit":1.25}
```

Join scores by exact sample identity before extracting the feature bundle:

```bash
python -B main.py join ./data/split.jsonl ./data/semantic.jsonl --score-key logit --output ./data/semantic_manifest.jsonl
python -B main.py extract ./data/semantic_manifest.jsonl --output ./data/semantic_features.npz
```

Use `--real-positive` when the supplied score increases toward real news. Train and freeze external models using the source protocol; store their implementation revision, checkpoint hash, fitting identities, and score direction. Scores from the target test set are inference outputs only.

### 5. Run the four-seed benchmark

```bash
python -B main.py benchmark ./data/semantic_features.npz --config ./configs/seed42.json --ablations --output ./outputs/seed42
python -B main.py benchmark ./data/semantic_features.npz --config ./configs/seed52.json --ablations --output ./outputs/seed52
python -B main.py benchmark ./data/semantic_features.npz --config ./configs/seed3047.json --ablations --output ./outputs/seed3047
python -B main.py benchmark ./data/semantic_features.npz --config ./configs/seed3407.json --ablations --output ./outputs/seed3407
```

The four configurations differ only in `seed`. These commands keep the same extracted inputs and split identities while varying the stochastic profiling and comparison components. If the semantic backbone is retrained for each seed, join and extract its corresponding seed-specific predictions and pass that bundle to the matching run. Explicitly record any protocol that also resamples splits.

Each output directory contains:

| File / directory | Content |
|---|---|
| `report.json` | Metrics, configuration, profile quality, and skipped methods |
| `summary.csv` | AUC, macro F1, accuracy, calibration, domain metrics, and timing |
| `predictions/` | Per-sample predictions for each method |
| `profiles/` | Frozen domain profiles |
| `calibration.json` | Probability and prediction-set calibration states |
| `provenance.json` | Source, input, environment, schema, and configuration fingerprints |

Report all four runs and their mean and sample standard deviation. Retain individual predictions for paired comparisons. The quick configuration reduces bootstrap repetitions and reference size for smoke tests; use the full seed configurations for the experiment protocol.

## Dataset Format & Split Provenance

The canonical label direction is **0 = real, 1 = fake**. Paths are relative to the repository working directory, including image paths stored inside a manifest. Run all commands from the repository root.

```json
{"id":"news001","image":"./data/images/news001.jpg","text":"Example news text.","domain":"politics","platform":"weibo","group":"event001","split":"test","label":1}
```

| Field | Contract |
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

Machine-readable contracts and conversion examples are provided in [configs/schema.json](configs/schema.json), [configs/conversion.json](configs/conversion.json), and [configs/labels.json](configs/labels.json). The label-map file is an example mapping and must match the classes of the dataset being converted.

Source supervision fits the comparison heads, validation fits probability parameters, calibration fits prediction sets, and the unlabeled target profile estimates the reference. Test labels enter evaluation only. Keep sample IDs, events, duplicate images, and recorded content identities disjoint across fitting, calibration, profiling, and evaluation roles. Unspecified partitions must be resolved before benchmarking.

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

DAPT, DAMMFND, and other published news detectors should be run with their official implementations and the declared split protocol. ReAlign supplies an external deep image-forensics channel. These implementations and checkpoints must be obtained from their authors; the local controls retain the identifiers above. `research.py channels` compares frozen semantic, physical, and deep channels using source-fitted comparison heads and independent evaluation identities. Run `python -B research.py channels --help` for its required inputs and provenance contract.

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
| Author's GPU | Single RTX 4090 |

Original image dimensions, EXIF, and JPEG quantization metadata are read before resizing. Missing and constant columns are handled explicitly, and the retained feature order is stored in each profile. The domain reference stores empirical marginals, a regularized copula correlation matrix, local sign weights, fitting identities, and diagnostics.

The physical score contains no target-domain gradient update. Unlabeled reference composition can change the score orientation; maintain an independent pool and evaluate contamination sensitivity. Optional source-trained confidence selection requires frozen predictions and metadata, and it remains subject to minimum sample and dependence diagnostics. `research.py filtering` evaluates that selection under controlled pool compositions.

The source uses the DDIN-style UTF-8 declaration followed by the full paper-title comment and one blank line. All paths in commands and code are relative. The exact installed dependency versions can be exported with `doctor.py --freeze`.

## Inference & Calibration

Fit an independent target profile and inspect its quality:

```bash
python -B main.py profile ./data/semantic_features.npz --domain politics --split profile --config ./configs/seed42.json --output ./models/politics.json
python -B main.py predict ./data/semantic_features.npz --profile ./models/politics.json --domain politics --split test --fusion --explain --output ./outputs/predictions.jsonl
python -B main.py evaluate ./outputs/predictions.jsonl --seed 42 --output ./outputs/evaluation.json
```

`predict` exports raw research scores unless `--calibration` supplies an independently fitted state. `copula_score` and `physical_evidence` increase toward the reference; `physical_risk`, `semantic_logit`, and `fusion_risk` increase toward fake news. Integration uses the larger available risk after the configured semantic threshold shift.

For deployment routing:

```bash
python -B main.py deploy ./data/semantic_features.npz --profile ./models/politics.json --domain politics --split test --explain --output ./outputs/deployment/predictions.jsonl --report ./outputs/deployment/report.json
```

Routing selects physical, semantic, fused, or abstaining output according to observable evidence and profile diagnostics. The report retains the full input count and route coverage. To obtain route-specific calibrated probabilities and prediction sets, first export `deploy` predictions on independent source validation and calibration partitions with their corresponding source profiles, combine the rows by role, then fit:

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

Its output is a negative-alignment control score. Use the chosen news detector's actual frozen predictions for the semantic news-veracity channel.

Additional controlled checks:

```bash
python -B main.py theory --seed 42 --output ./outputs/theory.json
python -B main.py invariance --seed 42 --output ./outputs/invariance.json
python -B research.py compare ./outputs/seed42/predictions/semantic_external.jsonl ./outputs/seed42/predictions/flawprint_fusion.jsonl --scope real --dataset yourdataset --seed 42 --output ./outputs/comparison.json
python -B download.py --sources ./configs/sources.json --output ./references
```

The algebraic and synthetic checks have explicit scopes. Comparison commands operate on the supplied predictions and preserve provenance. Reference downloads retrieve papers listed in `configs/sources.json`; dataset and model acquisition follow their respective official distributions.
