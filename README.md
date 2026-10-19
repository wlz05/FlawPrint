# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

**FlawPrint** detects multimodal fake news through the dependence between image and narrative traces. It removes marginal scale through rank normalization, estimates a domain copula, and measures departures from its joint structure. Local sign evidence identifies inconsistent trace relationships, while a frozen semantic detector scores the news narrative.

**FlawPrint achieves state-of-the-art performance on Weibo, Weibo21, and FineFake.** Experiments run on a **single NVIDIA GeForce RTX 4090**, using seeds **42, 52, 3047, and 3407**.

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

Store Weibo in `./data`, Weibo21 in `./weibo21`, and FineFake in `./FineFake`. For canonical input:

```bash
python -B main.py prepare ./data/raw.jsonl --labels fake-one --output ./data/manifest.jsonl
python -B main.py validate ./data/manifest.jsonl --benchmark --check-images --output ./outputs/preflight.json
python -B main.py extract ./data/manifest.jsonl --workers 1 --output ./data/features.npz
```

Use the supplied train, validation, and test partitions and retain their sample identities. Each evaluated domain has an independent reference pool.

For dataset-specific columns and label conversion:

```bash
python -B convert.py ./data/train_origin.csv --dataset WEIBO --mapping ./configs/weibo.json --split train --output ./data/weibo_train.jsonl
python -B convert.py ./weibo21/train_datasets.xlsx --dataset Weibo21 --mapping ./configs/weibo21.json --split train --output ./weibo21/train.jsonl
python -B convert.py ./FineFake/train.csv --dataset FineFake --mapping ./configs/finefake.json --split train --output ./FineFake/train.jsonl
```

Dataset names are converted to lowercase before selection: `WEIBO`, `Weibo21`, and `FineFake` select `weibo`, `weibo21`, and `finefake`. The dataset configurations map labels to 0 = real and 1 = fake. Use the same commands with `--split validation` and `--split test` for the corresponding source tables. JSONL, NDJSON, JSON, CSV, TSV, Excel, and Parquet are supported; Excel and Parquet use the table dependencies.

### 4. Supply frozen semantic predictions

Provide one finite logit per sample, with higher values indicating fake news:

```json
{"id":"news001","logit":1.25}
```

Join scores by exact sample identity before extracting the feature bundle:

```bash
python -B main.py join ./data/manifest.jsonl ./data/semantic.jsonl --score-key logit --output ./data/semantic_manifest.jsonl
python -B main.py extract ./data/semantic_manifest.jsonl --output ./data/semantic_features.npz
```

Use `--real-positive` for scores that increase toward real news. External models are trained on the source partitions and frozen before target inference.

### 5. Run the four-seed benchmark

```bash
python -B main.py benchmark ./data/semantic_features.npz --dataset Weibo21 --config ./configs/seed42.json --ablations --output ./outputs/seed42
python -B main.py benchmark ./data/semantic_features.npz --dataset Weibo21 --config ./configs/seed52.json --ablations --output ./outputs/seed52
python -B main.py benchmark ./data/semantic_features.npz --dataset Weibo21 --config ./configs/seed3047.json --ablations --output ./outputs/seed3047
python -B main.py benchmark ./data/semantic_features.npz --dataset Weibo21 --config ./configs/seed3407.json --ablations --output ./outputs/seed3407
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

## Datasets

| Dataset | Domains | Data partitions |
|---|---|---|
| [Weibo](https://doi.org/10.1145/3123266.3123454) | 9 | The released train, validation, and test partitions |
| [Weibo21](https://github.com/kennqiang/MDFEND-Weibo21) | 9 | The released train, validation, and test partitions |
| [FineFake](https://github.com/Accuser907/FineFake) | 7 | 60% / 20% / 20%, stratified by label with split seed 3407 |

Dataset partitions remain fixed across the four experiment seeds. FineFake is stratified by label using split seed 3407. Dataset paths and training settings are recorded in [configs/experiment.json](configs/experiment.json).

## Experimental Setup

We implement FlawPrint on a single NVIDIA GeForce RTX 4090. The semantic backbone is trained for 50 epochs with a batch size of 64, a learning rate of 0.0001, and a maximum text length of 197. Early stopping uses a patience of 5 epochs. We run each experiment with seeds 42, 52, 3047, and 3407 and report the mean and sample standard deviation.

| Setting | Value |
|---|---|
| GPU | Single NVIDIA GeForce RTX 4090 |
| Seeds | 42, 52, 3047, 3407 |
| Training epochs | 50 |
| Batch size | 64 |
| Learning rate | 0.0001 |
| Maximum text length | 197 |
| Early stopping patience | 5 epochs |
| Weight decay | 0.00005 |
| Text embedding dimension | 768 |
| FineFake partition seed | 3407 |

We report accuracy and classwise precision, recall, and F1-score, following the evaluation used in [DAMMFND](https://ojs.aaai.org/index.php/AAAI/article/view/32036) and [MMDFND](https://github.com/yutchina/MMDFND). The evaluation output also includes macro F1 and ROC-AUC.

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

The semantic channel scores news narratives; the physical channel measures trace consistency. Stratified evaluation records news veracity and physical manipulation separately.

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

Related methods are [DAPT](https://doi.org/10.1145/3746027.3755192), [DAMMFND](https://github.com/luweihai/DAMMFND), [MMDFND](https://github.com/yutchina/MMDFND), and [ADOSE](https://ojs.aaai.org/index.php/AAAI/article/view/39125). `research.py channels` compares semantic, physical, and image-forensics scores. Run `python -B research.py channels --help` for its input format.

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

Domain profiles estimate trace dependence from the reference pool. The source-trained selector filters reference samples using frozen confidence scores. `research.py filtering` measures its behavior as pool contamination changes.

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
| `main.py benchmark` | `--dataset`, `--config`, `--alpha`, `--bootstrap`, `--ablations`, `--output` |
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
python -B encode.py ./data/split.jsonl --model ./models/clip --device cuda:0 --batch-size 64 --output ./data/clip_predictions.jsonl
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
