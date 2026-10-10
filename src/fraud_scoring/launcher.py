"""Run the training script in the official AWS PyTorch container with SageMaker local mode.

Local mode uses the SageMaker SDK's own job machinery (input channels, ``SM_*`` environment,
``/opt/ml`` layout, ``model.tar.gz`` output) but starts the container on this machine with
Docker. It needs no AWS credentials and no IAM role. The container image and the script are
the ones a real ``CreateTrainingJob`` would run.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import fraud_scoring
from fraud_scoring.model import CONFIG_FILE, METADATA_FILE, MODEL_FILE
from fraud_scoring.training import Hyperparameters

ENTRY_SCRIPT = "train.py"
ENTRY_SHIM = '''"""SageMaker entry script: delegates to the packaged training entrypoint."""

from fraud_scoring.entrypoints.train import main

if __name__ == "__main__":
    main()
'''


class LocalModeError(RuntimeError):
    """Docker or the container is not usable for a local-mode run."""


def ensure_docker() -> None:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=60)
    except FileNotFoundError as exc:
        raise LocalModeError("docker was not found on PATH; install Docker Desktop") from exc
    except subprocess.CalledProcessError as exc:
        raise LocalModeError(
            "the Docker daemon is not reachable; start Docker Desktop and retry"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise LocalModeError("`docker info` timed out; is Docker Desktop still starting?") from exc


def ensure_image(image: str) -> None:
    """Pull the container image if it is not already present."""
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
        return
    try:
        subprocess.run(["docker", "pull", image], check=True)
    except subprocess.CalledProcessError as exc:
        raise LocalModeError(f"could not pull {image}") from exc


def stage_source_dir(destination: Path) -> Path:
    """Copy the package and an entry script into a directory SageMaker can upload or mount."""
    package_dir = Path(fraud_scoring.__file__).parent
    shutil.copytree(
        package_dir,
        destination / "fraud_scoring",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (destination / ENTRY_SCRIPT).write_text(ENTRY_SHIM)
    return destination


def remove_job_containers(root: Path) -> None:
    """Remove the exited training container and network that local mode leaves behind."""
    compose_file = root / "docker-compose.yaml"
    if compose_file.exists():
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "down", "--remove-orphans"],
            capture_output=True,
        )


def to_sagemaker_hyperparameters(hp: Hyperparameters) -> dict[str, str]:
    """SageMaker hyperparameters are strings; lists are comma-joined."""
    values = hp.as_dict()
    values["hidden_dims"] = ",".join(str(width) for width in hp.hidden_dims)
    return {name: str(value) for name, value in values.items()}


@dataclass(frozen=True)
class TrainingRun:
    job_name: str
    root: Path
    model_dir: Path
    model_tar: Path


def launch_local_training(
    *,
    data_dir: Path,
    jobs_dir: Path,
    hyperparameters: Hyperparameters,
    image: str,
    job_name: str | None = None,
) -> TrainingRun:
    """Train in the AWS container; ``data_dir`` must hold ``train/`` and ``validation/``."""
    # Imported here: the SageMaker SDK takes seconds to import and most commands do not need it.
    from sagemaker.core.training.configs import Compute, InputData, SourceCode
    from sagemaker.train.model_trainer import Mode, ModelTrainer

    for channel in ("train", "validation"):
        if not (data_dir / channel).is_dir():
            raise LocalModeError(
                f"{data_dir / channel} not found; run `fraud-scoring data prepare`"
            )
    ensure_docker()
    ensure_image(image)

    job = job_name or f"fraud-scoring-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    root = (jobs_dir / job).resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="fraud-scoring-src-") as source_dir:
            stage_source_dir(Path(source_dir))
            trainer = ModelTrainer(
                training_image=image,
                training_mode=Mode.LOCAL_CONTAINER,
                source_code=SourceCode(source_dir=source_dir, entry_script=ENTRY_SCRIPT),
                compute=Compute(instance_type="local_cpu", instance_count=1),
                hyperparameters=to_sagemaker_hyperparameters(hyperparameters),
                base_job_name=job,
                local_container_root=str(root),
            )
            trainer.train(
                input_data_config=[
                    InputData(channel_name=name, data_source=str((data_dir / name).resolve()))
                    for name in ("train", "validation")
                ]
            )
    finally:
        remove_job_containers(root)

    model_dir = root / "model"
    model_tar = root / "compressed_artifacts" / "model.tar.gz"
    expected = [model_dir / name for name in (MODEL_FILE, CONFIG_FILE, METADATA_FILE)]
    if not model_tar.exists() or not all(path.exists() for path in expected):
        raise LocalModeError(
            f"training finished without a complete model artifact under {root}; "
            "check the container log above"
        )
    return TrainingRun(job_name=job, root=root, model_dir=model_dir, model_tar=model_tar)


def latest_model_dir(jobs_dir: Path) -> Path:
    """The model directory of the most recently trained job.

    Ordered by when the artifact was written, not by job name: custom ``--job-name`` values
    would otherwise sort ahead of or behind the timestamped defaults.
    """
    candidates = [p for p in jobs_dir.glob("*/model") if (p / METADATA_FILE).exists()]
    if not candidates:
        raise LocalModeError(f"no trained model under {jobs_dir}; run `fraud-scoring train`")
    return max(candidates, key=lambda p: (p / METADATA_FILE).stat().st_mtime_ns)
