"""Runtime settings, read from FRAUD_SCORING_* environment variables or a local .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# AWS Deep Learning Containers for SageMaker. They are published on the ECR Public Gallery and
# pull anonymously, which is what local mode uses here (no AWS credentials involved).
_REGISTRY = "public.ecr.aws/deep-learning-containers"
_TAG = "2.5.1-cpu-py311-ubuntu22.04-sagemaker"
DEFAULT_TRAINING_IMAGE = f"{_REGISTRY}/pytorch-training:{_TAG}"
DEFAULT_INFERENCE_IMAGE = f"{_REGISTRY}/pytorch-inference:{_TAG}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_SCORING_", env_file=".env", extra="ignore")

    aws_profile: str | None = None
    aws_region: str = "us-east-1"
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    training_image: str = DEFAULT_TRAINING_IMAGE
    inference_image: str = DEFAULT_INFERENCE_IMAGE
    s3_bucket: str = ""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def jobs_dir(self) -> Path:
        return self.artifact_dir / "jobs"


def get_settings() -> Settings:
    return Settings()
