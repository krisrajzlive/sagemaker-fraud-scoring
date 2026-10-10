"""Docker helpers for local-mode serving.

The SageMaker SDK stops a local endpoint by terminating its ``docker compose up`` process. On
Windows that leaves the serving container running (and its network behind), so the containers
a run created are tracked and removed explicitly.
"""

from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Iterable

COMPOSE_PROJECT_LABEL = "com.docker.compose.project"


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def container_ids(image: str) -> set[str]:
    """IDs of every container (running or not) created from ``image``."""
    result = subprocess.run(
        ["docker", "ps", "-a", "-q", "--no-trunc", "--filter", f"ancestor={image}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


def serving_containers(ids: Iterable[str]) -> dict[str, str]:
    """Of ``ids``, those started with the ``serve`` command, mapped to their compose project."""
    ids = list(ids)
    if not ids:
        return {}
    template = (
        '{{.Id}}\t{{json .Config.Cmd}}\t{{index .Config.Labels "' + COMPOSE_PROJECT_LABEL + '"}}'
    )
    result = subprocess.run(
        ["docker", "inspect", "--format", template, *ids],
        capture_output=True,
        text=True,
    )
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        container_id, _, rest = line.partition("\t")
        command_json, _, project = rest.partition("\t")
        try:
            command = json.loads(command_json) or []
        except ValueError:
            continue
        if "serve" in command:
            found[container_id] = project
    return found


def remove_serving_containers(image: str, before: set[str]) -> int:
    """Remove serving containers of ``image`` created since the ``before`` snapshot, and their
    compose networks. Training containers and anything that already existed are left alone."""
    created = serving_containers(container_ids(image) - before)
    if not created:
        return 0
    subprocess.run(["docker", "rm", "-f", *created], capture_output=True)
    for project in {p for p in created.values() if p}:
        networks = subprocess.run(
            [
                "docker",
                "network",
                "ls",
                "-q",
                "--filter",
                f"label={COMPOSE_PROJECT_LABEL}={project}",
            ],
            capture_output=True,
            text=True,
        ).stdout.split()
        if networks:
            subprocess.run(["docker", "network", "rm", *networks], capture_output=True)
    return len(created)
