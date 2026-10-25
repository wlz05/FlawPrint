# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

**FlawPrint-CRA** learns how domain-referenced trace relationships condition news interactions. An independent domain pool defines pairwise copula residuals. Each content query selects trace coordinates, and the residual matrix is lifted into content-dependent attention through `B = U Gbar U^T`.

The deterministic FlawPrint baseline achieves **state-of-the-art performance on Weibo, Weibo21, and FineFake**. The content-conditioned CRA results are recorded separately. Experiment settings use a **single NVIDIA GeForce RTX 4090** and seeds **42, 52, 3047, and 3407**.

## Method Chain

```text
Original text ---> BERT and CLIP text vectors ----+
Original image -> MAE and CLIP image vectors ----+--> Content queries
        |                                                |
        +--> Pixel and container traces                  |
Text ------> Narrative coordinates                        |
                       |                                 |
Independent pool --> Frozen ranks and domain copula      |
                       |                                 |
                  Pairwise residual G                    |
                       |                                 |
                       +------ Query-selected coordinates U
                                       |
                                B = U Gbar U^T
                                       |
                           Reference-conditioned attention
                                       |
                             News risk and calibration
```

1. **Encode content.** Retain the BERT text, MAE image, CLIP text, and CLIP image vectors as four global content nodes.
2. **Measure trace coordinates.** Read noise, frequency, gradient, JPEG, resampling, and container proxies from the original image. Six narrative statistics complete the default 26-coordinate catalogue.
3. **Define a domain reference.** Frozen empirical marginals establish rank coordinates. The copula precision matrix defines pairwise support relative to independent pairing.
4. **Read relations through content.** Each attention head reuses its query/key scores to assign a distribution over available trace anchors. The lifted matrix `U Gbar U^T` supplies distinct content-to-content, content-to-trace, and trace-to-trace interaction conditions.
5. **Learn the news decision.** Residual attention updates the nodes and the news head reads the class node. Source news labels train projections and interactions; separately observed manipulation labels supervise an optional image head.

The core operation is the reference-conditioned relation readout in step 4. The encoder family supplies content representations. The reference defines measured support, and the current query determines which relationships enter each interaction. The mapping adds no separate gating-network parameters.

## Project Structure

```text
FlawPrint/
|-- deep.py                  # Embedding extraction, neural training, and prediction.
|-- main.py                  # Trace extraction and deterministic controls.
|-- convert.py               # Dataset conversion.
|-- doctor.py                # Environment inspection.
|-- encode.py                # CLIP similarity control.
|-- research.py              # Paired, composition, and channel analyses.
|-- study.py                 # Propagation, reference, and cost studies.
|-- download.py              # Reference paper downloads.
|-- test.py                  # Regression runner.
|-- engine/embeddings.py     # Frozen local encoder features and cache alignment.
|-- engine/neural.py         # Residual attention and comparison modes.
|-- engine/learning.py       # Group cross-fitting, training, and checkpoint loading.
|-- configs/                 # Training, profiling, conversion, and dataset settings.
|-- tests/                   # Data, mathematics, leakage, and workflow checks.
|-- requirements.txt
|-- requirements-optional.txt
|-- tables.txt
|-- README.md
```

## Requirements

| Component | Setting |
|---|---|
| Python | >=3.10 |
| NumPy / SciPy | >=1.26, <3 / >=1.11, <2 |
| Pillow / scikit-learn | >=10, <13 / >=1.4, <2 |
| Deep learning | PyTorch >=2.2; Transformers >=4.40, <6 |
| GPU | Single NVIDIA GeForce RTX 4090 |
| Seeds | 42, 52, 3047, 3407 |

Install the PyTorch build for your CUDA environment, then install the dependencies:

```bash
git clone https://github.com/wlz05/FlawPrint.git
cd FlawPrint
python -m venv .venv
```

```powershell
.venv/Scripts/Activate.ps1
```

```bash
source .venv/bin/activate
python -m pip install -r ./requirements-optional.txt
python -B doctor.py --semantic --tables --output ./outputs/environment.json --freeze ./outputs/versions.txt
python -B test.py
```

Use the activation command for your shell. The deterministic pipeline requires only `requirements.txt`; `tables.txt` adds Excel and Parquet readers. Trace profiling runs on CPU; deep extraction and training use `cuda:0` by default.

## Datasets

| Dataset | Domains | Partitions |
|---|---|---|
| [Weibo](https://doi.org/10.1145/3123266.3123454) | 9 | Released train, validation, and test identities |
| [Weibo21](https://github.com/kennqiang/MDFEND-Weibo21) | 9 | Released train, validation, and test identities |
| [FineFake](https://github.com/Accuser907/FineFake) | 7 | Fixed 60% / 20% / 20%, split seed 3407 |

Dataset partitions remain fixed across experiment seeds. Source validation and calibration identities are separate. Domain-reference pools are independent of evaluation samples. Keep events, original images, and duplicate content disjoint; validate the actual release before running. Settings and paths are recorded in [configs/experiment.json](configs/experiment.json).

## Prepare Data

Run commands from the repository root. All image, model, and output paths are relative to that directory. Canonical labels are **0 = real, 1 = fake**.

```json
{"id":"news001","image":"./data/images/news001.jpg","text":"Example news text.","domain":"politics","group":"event001","split":"test","label":1}
```

| Field | Meaning |
|---|---|
| `id` | Unique stable string shared by all feature caches |
| `image` / `text` | Relative image path or empty string; original news text |
| `domain` / `group` | Domain and event or duplicate-group identity |
| `split` | `train`, `validation`, `calibration`, `profile`, or `test` |
| `label` | News-veracity label; absent from target profile rows |
| `physical_manipulated` | Optional independently observed Boolean image attribute |
| `content_sha256` | Image-byte fingerprint populated by extraction |

Do not infer `physical_manipulated` from news labels. Missing attributes remain unknown. See [configs/schema.json](configs/schema.json), [configs/conversion.json](configs/conversion.json), and [configs/labels.json](configs/labels.json).

```bash
python -B convert.py ./data/train_origin.csv --dataset WEIBO --mapping ./configs/weibo.json --split train --output ./data/weibo_train.jsonl
python -B convert.py ./weibo21/train_datasets.xlsx --dataset Weibo21 --mapping ./configs/weibo21.json --split train --output ./weibo21/train.jsonl
python -B convert.py ./FineFake/train.csv --dataset FineFake --mapping ./configs/finefake.json --split train --output ./FineFake/train.jsonl
```

Repeat conversion for the other partitions with their matching `--split`. Dataset selection normalizes names to lowercase. JSONL, NDJSON, JSON, CSV, TSV, Excel, and Parquet are supported. Merge the converted rows into one manifest while preserving IDs and split roles, then validate and extract:

```bash
python -B main.py prepare ./data/raw.jsonl --labels fake-one --output ./data/manifest.jsonl
python -B main.py validate ./data/manifest.jsonl --benchmark --check-images --output ./outputs/preflight.json
python -B main.py extract ./data/manifest.jsonl --workers 1 --output ./data/features.npz
```

For source domains, training groups supply the physical reference through cross-fitting. Held-out target domains use independent unlabeled `profile` rows. A domain cannot supply both source training rows and target-profile rows in the same neural run.

## Pretrained Encoders

Place complete Hugging Face checkpoints, tokenizer assets, and processor files in `./models/bert`, `./models/mae`, and `./models/clip`. Extraction reads local files only. For Chinese input, compatible checkpoint families include [BERT-base-Chinese](https://huggingface.co/google-bert/bert-base-chinese), [ViT-MAE-base](https://huggingface.co/facebook/vit-mae-base), and [Chinese-CLIP](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16). Use language-matched BERT and CLIP checkpoints for other languages, and keep their exact files fixed within a comparison.

`deep.py` retains the BERT and MAE class-token vectors and both projected CLIP vectors. MAE uses all patches in fixed order. CLIP text is truncated to its checkpoint's supported context. The archive records encoder hashes, availability masks, identities, image checksums, and preprocessing provenance.

```bash
python -B deep.py extract ./data/manifest.jsonl --bert ./models/bert --mae ./models/mae --clip ./models/clip --batch-size 64 --max-length 197 --device cuda:0 --output ./data/deep.npz
```

Add `--forensic ./models/forensic` for a separately trained, Hugging Face-compatible forensic CLIP image encoder. This interface reads pooled image features; raw external checkpoints require conversion to the supported model format. Every compared mode must use the same encoder set. After changing image bytes through editing, JPEG compression, or resizing, regenerate both physical and deep features.

## Train and Predict

The pretrained encoders are frozen. Trainable projections, residual attention, and the news head use source news labels. Source event/image groups define cross-fitting folds; source training alone fits feature standardizers. Validation selects the checkpoint and fits probability calibration; a separate calibration partition fits prediction sets. Target labels do not enter fitting.

```bash
python -B deep.py train ./data/features.npz ./data/deep.npz --dataset Weibo21 --config ./configs/neural.json --profile-config ./configs/default.json --seed 42 --mode coupled --device cuda:0 --output ./models/neural42
python -B deep.py predict ./data/features.npz ./data/deep.npz --model ./models/neural42 --split test --device cuda:0 --output ./outputs/neural42/predictions.jsonl
```

Run the other experiment seeds with separate model directories:

```bash
python -B deep.py train ./data/features.npz ./data/deep.npz --dataset Weibo21 --seed 52 --mode coupled --output ./models/neural52
python -B deep.py train ./data/features.npz ./data/deep.npz --dataset Weibo21 --seed 3047 --mode coupled --output ./models/neural3047
python -B deep.py train ./data/features.npz ./data/deep.npz --dataset Weibo21 --seed 3407 --mode coupled --output ./models/neural3407
```

Predict each checkpoint and report mean and sample standard deviation across the four seeds. The model directory stores `parameters.pt`, `model.json`, `profiles/`, and CLI `provenance.json`. Predictions include calibrated news output, method mode, and reference availability. Labeled predictions also produce a `.report.json` metric file. Loading checks encoder/schema/weight/reference hashes and rejects identities or known lineage used for training, validation, or calibration.

## Baselines and Ablations

| Neural mode | Trace input | Readout |
|---|---|---|
| `deep` | None | Deep multimodal attention |
| `independent` | Rank coordinates, zero relation matrix | Ordinary attention |
| `unbiased` | Rank and residual contributions | Ordinary attention |
| `coupled` | Rank and residual contributions | Content-conditioned reference lifting |
| `concat` | The same rank and residual tokens | Flattened projections and MLP |
| `broadcast` | The same tokens | Shared row-summary bias; zero content-pair block |
| `diagonal` | The same tokens | Lift only the residual diagonal |
| `shuffled` | The same tokens | Lift a spectrum-preserving mismatch of relation identities |

Use `--mode` with a fresh model directory to run each control. The `coupled` versus `unbiased` comparison isolates the attention potential. The `coupled`, `broadcast`, `diagonal`, and `shuffled` modes have identical trainable parameter counts and node inputs. They isolate content conditioning, off-diagonal relationships, and trace-relation identity. All controls retain the same feature caches, source supervision, split identities, and training budget; report their actual parameter counts. If image attributes exist, keep the auxiliary-loss setting identical across controls.

Related complete news models include [DAPT](https://doi.org/10.1145/3746027.3755192), [DAMMFND](https://github.com/luweihai/DAMMFND), [MMDFND](https://github.com/yutchina/MMDFND), and [TERN](https://www.mdpi.com/2076-3417/16/13/6759). [ReAlign](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_ReAlign_Generalizable_Image_Forgery_Detection_via_Reasoning-Aligned_Representation_CVPR_2026_paper.pdf) provides a learned image-forensics comparison. [Falsdo](https://www.ijcai.org/proceedings/2026/275) motivates artifact-matched Real/Control/Fake evaluation.

### Fixed-Checkpoint Intervention

```bash
python -B deep.py predict ./data/features.npz ./data/deep.npz --model ./models/neural42 --split test --device cuda:0 --explain --output ./outputs/neural42/intervention.jsonl
```

`--explain` holds the checkpoint and relation tokens fixed, disables the attention potential, and exports `potential_disabled_risk` and `potential_risk_delta`. It adds one forward pass and records a software intervention on raw news risk. Probabilities and prediction sets still use the original calibrated risk. These fields measure model response; they do not identify a physical cause or establish factual truth.

New checkpoints use `flawprint-neural-v2` and record the relation operator. Existing `v1` coupled checkpoints retain their original broadcast operation when loaded. Prediction output includes the effective mode and checkpoint version.

## Implementation Details

| Setting | Value |
|---|---|
| GPU / seeds | Single RTX 4090 / 42, 52, 3047, 3407 |
| Maximum epochs / batch size | 50 / 64 |
| Optimizer / learning rate / weight decay | AdamW / 0.0001 / 0.00005 |
| Source selection / patience | Validation macro F1 / 5 epochs |
| Maximum text length | 197, limited by each encoder's context |
| Projected width / heads / layers | 128 / 4 / 2 |
| Dropout / gradient norm limit | 0.1 / 1.0 |
| Residual-potential bound | 2.0 |
| Source-reference group folds | 2 |
| Optional image-loss weight | 0.1 |
| Trace image longest / minimum side | 768 / 32 pixels |
| Trace schema / local grid | 20 image + 6 narrative / 4 x 4 |
| Global / local shrinkage | 0.1 / 0.35 |
| Minimum reference samples | 20 |
| Deterministic sign bootstrap | 100 repetitions |

The inverse correlation is estimated only in retained trace coordinates. The neural path uses the global pairwise matrix; local sign penalties belong to the deterministic control. Missing streams are masked. Missing or insufficient references produce zero relation potential and preserve deep inference.

Report Accuracy and classwise Precision, Recall, and F1 alongside macro F1 and ROC-AUC, as used in [DAMMFND](https://ojs.aaai.org/index.php/AAAI/article/view/32036) and [MMDFND](https://github.com/yutchina/MMDFND). Report source-calibrated probability metrics and prediction-set coverage separately. Image-authenticity auxiliary metrics use independent image labels.

## Deterministic Controls

The original trace/copula scoring and external semantic fusion remain available. Verify the core pipeline with a synthetic fixture:

```bash
python -B main.py fixture --seed 42 --per-split 24 --output ./data/fixture.npz
python -B main.py benchmark ./data/fixture.npz --config ./configs/quick.json --bootstrap 25 --ablations --output ./outputs/smoke
```

Provide frozen news logits increasing toward fake news, join by exact identity, and benchmark:

```json
{"id":"news001","logit":1.25}
```

```bash
python -B main.py join ./data/manifest.jsonl ./data/semantic.jsonl --score-key logit --output ./data/semantic_manifest.jsonl
python -B main.py extract ./data/semantic_manifest.jsonl --output ./data/semantic_features.npz
python -B main.py benchmark ./data/semantic_features.npz --dataset Weibo21 --config ./configs/seed42.json --ablations --output ./outputs/seed42
```

Repeat with `seed52.json`, `seed3047.json`, and `seed3407.json` and their matching outputs. Built-in controls include TraceLR, MarginalNB, SingleTrace, CharTFIDF, GaussianNovelty, FrozenSemantic, TraceRBF, ImageRBF, and both RBF fusion variants. Deterministic FlawPrint variants are `copula_only`, `sign_only`, `physical_full`, and `flawprint_fusion`.

```bash
python -B main.py profile ./data/semantic_features.npz --domain politics --split profile --config ./configs/seed42.json --output ./models/politics.json
python -B main.py predict ./data/semantic_features.npz --profile ./models/politics.json --domain politics --split test --fusion --explain --output ./outputs/deterministic.jsonl
python -B main.py evaluate ./outputs/deterministic.jsonl --seed 42 --output ./outputs/evaluation.json
```

`copula_score` and `physical_evidence` increase toward reference support. `physical_risk`, `semantic_logit`, and `fusion_risk` increase toward fake news. Deterministic fusion takes the larger available risk; CRA uses its trained joint head. `main.py deploy` and `main.py calibrate` provide deterministic routing and separate source calibration.

## Additional Analysis

`research.py channels` compares frozen semantic, physical, and forensic risks. `encode.py` supplies a local CLIP negative-similarity control. Neither command trains CRA.

```bash
python -B deep.py --help
python -B research.py channels --help
python -B research.py filtering --help
python -B study.py propagation --help
python -B main.py theory --seed 42 --output ./outputs/theory.json
python -B main.py invariance --seed 42 --output ./outputs/invariance.json
```

The regression suite covers local transformer extraction, checkpoint reload, missing inputs, reference overlap, target-label isolation, equal-summary relation controls, bounded query-conditioned lifting, legacy reload, gradient flow, and all eight neural modes. Synthetic fixtures and randomly initialized small local encoders verify implementation behavior; real-data benchmark results are recorded separately.
