from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fraud_scoring import docker_utils
from fraud_scoring.client import EndpointError, invoke, ping
from fraud_scoring.model import METADATA_FILE
from fraud_scoring.samples import sample_transactions
from fraud_scoring.schema import INPUT_FIELDS, TARGET
from fraud_scoring.serving import ARTIFACT_FILES, package_model

# ---------------------------------------------------------------------------- packaging


def test_package_model_uses_the_layout_sagemaker_unpacks(trained_model, tmp_path):
    staged = package_model(trained_model, tmp_path / "staged")
    for name in ARTIFACT_FILES:
        assert (staged / name).exists()
    assert (staged / "code" / "inference.py").exists()
    assert (staged / "code" / "fraud_scoring" / "entrypoints" / "inference.py").exists()
    assert not list(staged.rglob("__pycache__"))


def test_staged_code_dir_is_self_contained(trained_model, tmp_path):
    """The container imports code/inference.py with only code/ on the path."""
    staged = package_model(trained_model, tmp_path / "staged")
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import inference, fraud_scoring;"
        "print(fraud_scoring.__file__);"
        "print(all(callable(getattr(inference, n)) for n in "
        "('model_fn', 'input_fn', 'predict_fn', 'output_fn')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, str(staged / "code")],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    package_path, handlers_ok = result.stdout.split("\n")[:2]
    assert str(staged / "code") in package_path, "must import the staged copy, not the repo's"
    assert handlers_ok == "True"


def test_package_model_rejects_a_directory_that_is_not_an_artifact(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        package_model(tmp_path / "empty", tmp_path / "out")


def test_package_model_does_not_modify_the_artifact(trained_model, tmp_path):
    before = sorted(p.name for p in trained_model.iterdir())
    package_model(trained_model, tmp_path / "staged")
    assert sorted(p.name for p in trained_model.iterdir()) == before
    assert (trained_model / METADATA_FILE).exists()


# ---------------------------------------------------------------------------- client


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence test output
        pass

    def do_GET(self):  # noqa: N802
        self.send_response(200 if self.path == "/ping" else 404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        count = len(payload["instances"])
        if count == 3:  # emulate the toolkit: JSON message line, then a traceback
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "bad thing"}\nTraceback (most recent call last):\n  ...')
        elif count == 4:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"plain text failure\nTraceback ...")
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"predictions": [], "got": count}).encode())


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def test_invoke_returns_the_parsed_response(server):
    assert invoke(server, [{}, {}])["got"] == 2


def test_invoke_reports_the_message_line_not_the_traceback(server):
    with pytest.raises(EndpointError) as caught:
        invoke(server, [{}, {}, {}])
    assert caught.value.status == 400 and caught.value.detail == "bad thing"

    with pytest.raises(EndpointError) as caught:
        invoke(server, [{}] * 4)
    assert caught.value.status == 500 and caught.value.detail == "plain text failure"


def test_invoke_reports_an_unreachable_endpoint():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    with pytest.raises(EndpointError) as caught:
        invoke(f"http://127.0.0.1:{free_port}", [{}], timeout=2)
    assert caught.value.status is None and "cannot reach" in caught.value.detail


def test_ping(server):
    assert ping(server) is True
    assert ping("http://127.0.0.1:9", timeout=1) is False


# ---------------------------------------------------------------------------- samples


def test_samples_have_the_requested_mix_and_never_include_the_label(processed_dir):
    instances, labels = sample_transactions(processed_dir / "test", count=8, frauds=3, seed=1)
    assert len(instances) == len(labels) == 8 and sum(labels) == 3
    assert all(set(row) == set(INPUT_FIELDS) and TARGET not in row for row in instances)


def test_samples_are_reproducible_and_seed_dependent(processed_dir):
    first = sample_transactions(processed_dir / "test", count=5, frauds=1, seed=3)
    assert first == sample_transactions(processed_dir / "test", count=5, frauds=1, seed=3)
    assert first != sample_transactions(processed_dir / "test", count=5, frauds=1, seed=4)


def test_samples_cap_frauds_at_what_exists_and_validate_count(processed_dir):
    _, labels = sample_transactions(processed_dir / "test", count=4, frauds=10_000)
    assert sum(labels) == 4
    with pytest.raises(ValueError):
        sample_transactions(processed_dir / "test", count=0, frauds=0)


# ---------------------------------------------------------------------------- docker helpers


def test_port_in_use():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert docker_utils.port_in_use(port) is True
    assert docker_utils.port_in_use(port) is False


def test_cleanup_removes_only_new_serving_containers(monkeypatch):
    calls: list[list[str]] = []
    inspect_output = (
        'serve-new\t["serve"]\tproj-a\ntrain-new\t["train"]\tproj-b\nserve-old\t["serve"]\tproj-c\n'
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        stdout = ""
        if command[:3] == ["docker", "ps", "-a"]:
            stdout = "serve-new\ntrain-new\nserve-old\n"
        elif command[:2] == ["docker", "inspect"]:
            wanted = set(command[4:])  # only the ids the helper asks about
            stdout = "".join(
                line + "\n" for line in inspect_output.splitlines() if line.split("\t")[0] in wanted
            )
        elif command[:3] == ["docker", "network", "ls"]:
            stdout = "net-1\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(docker_utils.subprocess, "run", fake_run)
    removed = docker_utils.remove_serving_containers("image:tag", before={"serve-old"})

    assert removed == 1
    removals = [c for c in calls if c[:3] == ["docker", "rm", "-f"]]
    assert removals == [["docker", "rm", "-f", "serve-new"]]
    assert ["docker", "network", "rm", "net-1"] in calls
    assert not any("proj-b" in " ".join(c) for c in calls), "training project must be untouched"
