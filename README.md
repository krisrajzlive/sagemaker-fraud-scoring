# SageMaker fraud scoring

Fraud-risk scoring for card transactions, built around the Amazon SageMaker workflow: PyTorch
script-mode training, evaluation against baselines, and real-time inference, all running
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
uv run fraud-scoring predict            # real-time inference: score real test transactions
```

Hyperparameters are command-line options (`fraud-scoring train --help`); they reach the container
exactly as a SageMaker job would pass them (`--hidden_dims 64,32 --epochs 40 ...`).

## Real-time inference

`predict` and `serve` run the trained model in the **official AWS PyTorch inference container**
(TorchServe plus the SageMaker inference toolkit) with SageMaker local mode, so the endpoint speaks
the same `/ping` and `/invocations` contract as a hosted one. The container's very verbose log goes
to `artifacts/serving/endpoint-<port>.log`, not to your terminal.

```bash
uv run fraud-scoring predict                      # start the container, score 6 real transactions, stop
uv run fraud-scoring serve                        # keep an endpoint on http://127.0.0.1:8080 until Ctrl+C
uv run fraud-scoring predict --url http://127.0.0.1:8080 --sample 10 --frauds 3
uv run fraud-scoring predict --file my_requests.json
```

A transaction carries the dataset's fields (`Time`, `V1`..`V28`, `Amount`):

```bash
curl -X POST http://127.0.0.1:8080/invocations -H "Content-Type: application/json"      -d '{"instances": [{"Time": 406.0, "V1": -2.31, ..., "V28": -0.02, "Amount": 149.62}]}'
# {"predictions": [{"fraud_probability": 0.0031, "flagged": false}], "threshold": 0.9376}
```

`serve` writes a ready-to-use request with three real test transactions to
`artifacts/serving/example_request.json`. Example `predict --sample 8 --frauds 3 --seed 21`:

```
 # | Amount | actual | fraud probability | decision
 1 |  51.79 | legit  |            0.0064 | ok
 2 |  35.00 | FRAUD  |            0.9828 | FLAGGED
 3 |   3.90 | FRAUD  |            0.9963 | FLAGGED
 4 |   0.38 | FRAUD  |            0.9953 | FLAGGED
 5 |  84.00 | legit  |            0.0003 | ok
 ...
3 of 8 flagged; 8 of 8 decisions match the true label
```

Small samples flatter the model: on the whole test split it catches 35 of 52 frauds, and one of the
three transactions in `example_request.json` scores 0.934, just under the 0.9376 threshold, so it
is *not* flagged.

**Request validation.** Bad input gets a 4xx status and a JSON `{"error": ...}` line, never a 500:
invalid JSON, a missing field (the message names the instance), a non-numeric or negative value
(400), more than 1,000 instances (413), a content type other than `application/json` (415) and
`Accept: text/html` (406). The AWS toolkit appends its own traceback after that first line in error
bodies; the `predict` command shows only the message.

**What was checked live** (this laptop, Docker Desktop, one run):

| Check | Result |
|---|---|
| Container (Python 3.11, NumPy 2.2) vs host (Python 3.12) scores, 200 transactions | identical (max difference 0.0) |
| Latency, batch of 1, 60 requests | p50 27 ms, p95 48 ms |
| Latency, batch of 100, 60 requests | p50 30 ms, p95 58 ms |
| Parent process killed without warning | the endpoint removed its own container and network within about 30 s |

The latency figures describe this machine only (loopback to a container in the Docker Desktop VM);
they say nothing about any AWS instance type.

## How it maps to SageMaker

| This repo | SageMaker concept |
|---|---|
| `fraud_scoring/entrypoints/train.py` | script-mode entry script; reads `SM_CHANNEL_*`, writes `SM_MODEL_DIR` and `SM_OUTPUT_DATA_DIR` |
| `launcher.py` (`ModelTrainer`, `Mode.LOCAL_CONTAINER`) | the SDK v3 training job API, pointed at local Docker |
| `data/processed/{train,validation}` | input channels |
| `artifacts/jobs/<job>/compressed_artifacts/model.tar.gz` | model artifact an endpoint would load |
| `sagemaker_config.METRIC_DEFINITIONS` | `metric_definitions` regexes that scrape training logs |
| `entrypoints/inference.py` | the PyTorch container's `model_fn` / `input_fn` / `predict_fn` / `output_fn` contract |
| `serving.py` (`package_model`) | the `model.tar.gz` layout: artifact files at the top, `code/inference.py` beside the package |
| `endpoint_worker.py` (`LocalSession`) | `CreateModel` / `CreateEndpointConfig` / `CreateEndpoint`, pointed at local Docker |

The model artifact is a `state_dict` loaded with `weights_only=True`, plus JSON for the
architecture, feature scaling and threshold. It contains no pickled code and does not depend on
the NumPy version that wrote it (the training and inference containers ship different NumPy
releases).

### SageMaker SDK v3.24 notes

Found while building this; each has a workaround in the code.

- `ModelBuilder(mode=LOCAL_CONTAINER)` validates an IAM role even locally and rejects a placeholder
  ARN, so serving goes through the SDK's `LocalSession` client instead. Training with
  `ModelTrainer` needs no role.
- Importing `sagemaker.serve` creates a boto3 client, which fails without a default AWS region.
  The endpoint process sets `AWS_DEFAULT_REGION` (if unset) before importing; no credentials needed.
- `sagemaker.core.local.image` uses `sagemaker.serve.model_builder` without importing it.
- Stopping a local endpoint on Windows terminates `docker compose` but leaves the container
  running, so `endpoint_worker.py` removes the containers and network it created.
- Importing the SDK for the first time prints `SyntaxWarning: invalid escape sequence` from its own
  docstrings. They are harmless and disappear once Python has cached the compiled files.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                 # 106 tests, no Docker or dataset needed
uv run pytest -m docker -q       # 4 tests: trains and serves for real in the AWS containers
```

Tests never read your `.env`. CI runs lint, the unit tests and a gitleaks scan of the full
history. The Docker tests are not in CI yet.

## Status

| Capability | Status |
|---|---|
| PyTorch script-mode training in the official AWS container | **Ran** (SageMaker local mode, Docker) and covered by a Docker test |
| Chronological split, leakage-safe preprocessing, early stopping | **Ran** |
| Held-out evaluation with bootstrap intervals and baselines | **Ran** |
| Real-time inference in the official AWS inference container (`predict`, `serve`) | **Ran** (SageMaker local mode, Docker) and covered by Docker tests |
| Hyperparameter search, model registry, JumpStart catalog | Not built yet |
| SageMaker Pipeline definition, CloudFormation, AWS deployment | Not built yet; will be validated offline only |
| Managed training job / endpoint on AWS | **Never run** (needs an AWS account with a SageMaker execution role) |
