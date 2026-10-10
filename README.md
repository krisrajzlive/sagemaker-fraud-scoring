# SageMaker fraud scoring

Fraud-risk scoring for card transactions, built around the Amazon SageMaker workflow: PyTorch
script-mode training, evaluation against baselines, and (next) real-time inference, all running
in the **official AWS Deep Learning Containers** through the SageMaker Python SDK v3.

The project is honest about where it runs. The AWS account used to develop it is a restricted
lab sandbox that denies `CreateTrainingJob`, `CreateEndpoint` and has no SageMaker execution
role, so nothing here has run as a *managed* SageMaker job. Training runs in **SageMaker local
mode**: the SDK starts the same AWS container, with the same `/opt/ml` layout, `SM_*` environment
and `model.tar.gz` output, on the local Docker daemon. See [Status](#status).

## Data

[ULB credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807
European card transactions over two days, 492 frauds. It is downloaded on demand (151 MB, SHA-256
pinned) and never committed. Released under the Database Contents License (DbCL) v1.0; Dal Pozzolo
et al., *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015.

- 1,081 exact duplicate rows are dropped, leaving 283,726 transactions and 473 frauds (0.17%).
- The split is **chronological** (70/15/15), not random, so the test set is the latest traffic
  the model has never seen. Rows sharing a timestamp are never separated.
- `V1..V28` are PCA components of undisclosed original features.

| split | rows | frauds |
|---|---:|---:|
| train | 198,609 | 366 |
| validation | 42,560 | 55 |
| test | 42,557 | 52 |

## Results

Measured on the held-out test split with the model trained by `fraud-scoring train` (default
hyperparameters, seed 42, one run). The decision threshold was chosen on the validation split.

| scorer | test PR-AUC | 95% bootstrap interval |
|---|---:|---|
| random scores (prevalence) | 0.001 | |
| logistic regression, class-balanced | 0.727 | 0.605 - 0.842 |
| PyTorch MLP (31 -> 64 -> 32 -> 1) | 0.768 | 0.658 - 0.871 |

At the validation-chosen threshold the MLP flags 38 transactions: 35 of the 52 frauds caught
(recall 0.67) with 3 false alarms (precision 0.92).

**Read this carefully:** with 52 positives the intervals overlap almost completely, so this run
gives no evidence that the MLP beats logistic regression. The model is a vehicle for the
SageMaker workflow, not a claim about state-of-the-art fraud detection. Numbers are from a single
seed, and the validation PR-AUC (about 0.87) is noticeably higher than the test PR-AUC, which is
consistent with choosing an epoch and a threshold on 55 validation frauds and then testing on a
later time window.

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker. Training needs no AWS
credentials: the container images are public and are pulled anonymously on first use (about 2 GB).

```bash
uv sync
uv run fraud-scoring data download      # 151 MB, checksum verified
uv run fraud-scoring data prepare       # clean + chronological split -> data/processed/
uv run fraud-scoring train              # SageMaker local mode, official PyTorch training container
uv run fraud-scoring evaluate           # held-out test split, bootstrap intervals, baselines
```

Hyperparameters are command-line options (`fraud-scoring train --help`); they reach the container
exactly as a SageMaker job would pass them (`--hidden_dims 64,32 --epochs 40 ...`).

## How it maps to SageMaker

| This repo | SageMaker concept |
|---|---|
| `fraud_scoring/entrypoints/train.py` | script-mode entry script; reads `SM_CHANNEL_*`, writes `SM_MODEL_DIR` and `SM_OUTPUT_DATA_DIR` |
| `launcher.py` (`ModelTrainer`, `Mode.LOCAL_CONTAINER`) | the SDK v3 training job API, pointed at local Docker |
| `data/processed/{train,validation}` | input channels |
| `artifacts/jobs/<job>/compressed_artifacts/model.tar.gz` | model artifact an endpoint would load |
| `sagemaker_config.METRIC_DEFINITIONS` | `metric_definitions` regexes that scrape training logs |

The model artifact is a `state_dict` loaded with `weights_only=True`, plus JSON for the
architecture, feature scaling and threshold. It contains no pickled code and does not depend on
the NumPy version that wrote it (the training and inference containers ship different NumPy
releases).

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                 # 64 tests, no Docker or dataset needed
uv run pytest -m docker -q       # trains for real in the AWS container
```

Tests never read your `.env`. CI runs lint, the unit tests and a gitleaks scan of the full
history. The Docker tests are not in CI yet.

## Status

| Capability | Status |
|---|---|
| PyTorch script-mode training in the official AWS container | **Ran** (SageMaker local mode, Docker) and covered by a Docker test |
| Chronological split, leakage-safe preprocessing, early stopping | **Ran** |
| Held-out evaluation with bootstrap intervals and baselines | **Ran** |
| Real-time inference endpoint | Not built yet |
| Hyperparameter search, model registry, JumpStart catalog | Not built yet |
| SageMaker Pipeline definition, CloudFormation, AWS deployment | Not built yet; will be validated offline only |
| Managed training job / endpoint on AWS | **Never run** (needs an AWS account with a SageMaker execution role) |
