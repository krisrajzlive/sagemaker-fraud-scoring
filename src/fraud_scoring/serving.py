"""Serve a trained model as a real-time endpoint in the official AWS PyTorch inference container.

SageMaker local mode runs the same container image, ``/opt/ml/model`` layout and
``/ping`` + ``/invocations`` contract as a hosted endpoint, on the local Docker daemon. No AWS
credentials or execution role are involved.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import TracebackType

from fraud_scoring import client
from fraud_scoring.docker_utils import container_ids, port_in_use, remove_serving_containers
from fraud_scoring.launcher import LocalModeError, copy_package, ensure_docker, ensure_image
from fraud_scoring.model import CONFIG_FILE, METADATA_FILE, MODEL_FILE, PREPROCESSING_FILE

ARTIFACT_FILES = (MODEL_FILE, CONFIG_FILE, PREPROCESSING_FILE, METADATA_FILE)
INFERENCE_SHIM = '''"""SageMaker inference entry script: delegates to the packaged handlers."""

from fraud_scoring.entrypoints.inference import input_fn, model_fn, output_fn, predict_fn

__all__ = ["input_fn", "model_fn", "output_fn", "predict_fn"]
'''


def package_model(model_dir: Path, destination: Path) -> Path:
    """Lay out ``destination`` the way SageMaker unpacks ``model.tar.gz`` for this container.

    The artifact files sit at the top level and the inference code under ``code/``, which the
    PyTorch container puts on ``sys.path`` and scans for ``inference.py``.
    """
    missing = [name for name in ARTIFACT_FILES if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{model_dir} is not a model artifact; missing {', '.join(missing)}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        shutil.copy2(model_dir / name, destination / name)
    code_dir = destination / "code"
    code_dir.mkdir(exist_ok=True)
    copy_package(code_dir)
    (code_dir / "inference.py").write_text(INFERENCE_SHIM)
    return destination


class LocalEndpoint:
    """A model served in the AWS inference container, started and stopped as a context manager.

    ``start`` blocks until the container answers ``/ping``. The container's own log goes to
    ``log_path``.
    """

    def __init__(
        self,
        model_dir: Path,
        image: str,
        *,
        port: int = 8080,
        workers: int = 2,
        region: str = "us-east-1",
        log_path: Path | None = None,
        startup_timeout: float = 240.0,
    ) -> None:
        self.model_dir = model_dir
        self.image = image
        self.port = port
        self.workers = workers
        self.region = region
        self.startup_timeout = startup_timeout
        self.log_path = (
            log_path or Path(tempfile.gettempdir()) / f"fraud-scoring-endpoint-{port}.log"
        )
        self._process: subprocess.Popen[str] | None = None
        self._staging: tempfile.TemporaryDirectory[str] | None = None
        self._before: set[str] = set()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> LocalEndpoint:
        ensure_docker()
        if port_in_use(self.port):
            raise LocalModeError(
                f"port {self.port} is already in use; stop what is listening there or use --port"
            )
        ensure_image(self.image)
        self._before = container_ids(self.image)
        self._staging = tempfile.TemporaryDirectory(prefix="fraud-scoring-model-")
        staged = package_model(self.model_dir, Path(self._staging.name))

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        isolation = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # no Ctrl+C for the child
            if os.name == "nt"
            else {"start_new_session": True}
        )
        with self.log_path.open("w", encoding="utf-8") as log:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "fraud_scoring.endpoint_worker",
                    "--model-dir",
                    str(staged),
                    "--image",
                    self.image,
                    "--port",
                    str(self.port),
                    "--name",
                    f"fraud-scoring-{uuid.uuid4().hex[:8]}",
                    "--workers",
                    str(self.workers),
                    "--region",
                    self.region,
                ],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                **isolation,
            )
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                raise LocalModeError(
                    f"the endpoint process exited during startup; see {self.log_path}\n"
                    + self._log_tail()
                )
            if client.ping(self.url, timeout=2):
                return
            time.sleep(1.0)
        raise LocalModeError(f"the endpoint did not become ready in {self.startup_timeout:.0f}s")

    def _log_tail(self, lines: int = 15) -> str:
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.strip().splitlines()[-lines:])

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.stdin:
                process.stdin.close()  # the worker's signal to clean up and exit
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
        # Belt and braces: also clean up if the worker died before it could.
        try:
            remove_serving_containers(self.image, self._before)
        except (subprocess.SubprocessError, OSError):
            pass
        if self._staging is not None:
            self._staging.cleanup()
            self._staging = None

    def __enter__(self) -> LocalEndpoint:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
