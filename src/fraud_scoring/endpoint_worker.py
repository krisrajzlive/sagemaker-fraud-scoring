"""Child process owning one local SageMaker endpoint (``-m fraud_scoring.endpoint_worker``).

Running the endpoint here keeps the SDK's very chatty container log out of the caller's terminal
(the parent points this process's stdout at a log file) and ties cleanup to this process's
lifetime: it serves until its stdin is closed, which the parent does to stop it and which the
operating system does if the parent dies. On the way out it deletes the endpoint and removes the
serving container and network the SDK leaves behind on Windows.
"""

from __future__ import annotations

import argparse
import os
import sys

from fraud_scoring.docker_utils import container_ids, remove_serving_containers

# The SDK ignores the role in local mode, but its API requires a well-formed ARN.
PLACEHOLDER_ROLE = "arn:aws:iam::000000000000:role/local-mode-not-used"
READY_MARKER = "ENDPOINT-READY"


def serve(model_dir: str, image: str, port: int, name: str, workers: int, region: str) -> None:
    # SDK 3.24 creates a boto3 client while importing sagemaker.serve (ai_registry/air_hub.py),
    # which fails without a default region, although local mode never calls AWS.
    os.environ.setdefault("AWS_DEFAULT_REGION", region)

    import boto3

    # sagemaker.core.local.image uses sagemaker.serve.model_builder without importing it.
    import sagemaker.serve.model_builder  # noqa: F401
    from sagemaker.core.local.local_session import LocalSession

    before = container_ids(image)
    # LocalSession refuses to start without a region, although local mode never calls AWS.
    session = LocalSession(boto_session=boto3.Session(region_name=region))
    session.config = {"local": {"serving_port": port}}
    client = session.sagemaker_client
    try:
        client.create_model(
            ModelName=name,
            PrimaryContainer={
                "Image": image,
                "ModelDataUrl": f"file://{model_dir}",
                # TorchServe starts one worker per CPU by default; each one loads PyTorch.
                "Environment": {"SAGEMAKER_MODEL_SERVER_WORKERS": str(workers)},
            },
            ExecutionRoleArn=PLACEHOLDER_ROLE,
        )
        client.create_endpoint_config(
            EndpointConfigName=name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": name,
                    "InitialInstanceCount": 1,
                    "InstanceType": "local",
                }
            ],
        )
        client.create_endpoint(EndpointName=name, EndpointConfigName=name)
        print(READY_MARKER, flush=True)
        sys.stdin.read()  # blocks until the parent closes the pipe (or exits)
    finally:
        try:
            client.delete_endpoint(EndpointName=name)
        finally:
            remove_serving_containers(image, before)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--name", default="fraud-scoring")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args(argv)
    serve(args.model_dir, args.image, args.port, args.name, args.workers, args.region)


if __name__ == "__main__":
    main()
