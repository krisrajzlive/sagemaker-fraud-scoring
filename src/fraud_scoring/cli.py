"""Command-line interface: ``fraud-scoring --help``."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fraud_scoring.config import get_settings

app = typer.Typer(
    help="Transaction fraud-risk scoring on Amazon SageMaker.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Dataset download and preparation.", no_args_is_help=True)
app.add_typer(data_app, name="data")

console = Console()


@data_app.command("download")
def data_download(
    force: Annotated[
        bool, typer.Option(help="Download again even if a verified copy exists.")
    ] = False,
) -> None:
    """Download the public credit-card fraud dataset and verify its checksum."""
    from fraud_scoring.data import DATASET_URL, download_dataset

    settings = get_settings()
    console.print(f"source: {DATASET_URL}")
    with console.status("downloading (151 MB)..."):
        path = download_dataset(settings.raw_dir, force=force)
    console.print(f"[green]verified[/green] {path}")


@data_app.command("prepare")
def data_prepare() -> None:
    """Clean the dataset and write time-ordered train/validation/test splits as parquet."""
    from fraud_scoring.data import (
        DATASET_SHA256,
        RAW_FILENAME,
        DataIntegrityError,
        clean,
        load_raw,
        split_by_time,
        verify_dataset,
        write_splits,
    )

    settings = get_settings()
    raw_path = settings.raw_dir / RAW_FILENAME
    if not raw_path.exists():
        console.print(f"[red]{raw_path} not found; run `fraud-scoring data download`[/red]")
        raise typer.Exit(1)
    try:
        verify_dataset(raw_path)
        raw = load_raw(raw_path)
        frame = clean(raw)
    except DataIntegrityError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"rows: {len(raw):,} raw -> {len(frame):,} after dropping exact duplicates")
    manifest = write_splits(split_by_time(frame), settings.processed_dir, DATASET_SHA256)

    table = Table(title=f"splits written to {settings.processed_dir}")
    for column in ("split", "rows", "frauds", "fraud rate", "time range (s)"):
        table.add_column(column, justify="right" if column != "split" else "left")
    for name, info in manifest["splits"].items():
        table.add_row(
            name,
            f"{info['rows']:,}",
            str(info["frauds"]),
            f"{info['fraud_rate']:.3%}",
            f"{info['time_start']:.0f} - {info['time_end']:.0f}",
        )
    console.print(table)


@app.command()
def train(
    epochs: int = 40,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dims: Annotated[
        str, typer.Option(help="Comma-separated hidden layer widths.")
    ] = "64,32",
    dropout: float = 0.1,
    pos_weight_power: float = 0.5,
    patience: int = 6,
    seed: int = 42,
    job_name: Annotated[str | None, typer.Option(help="Defaults to a timestamped name.")] = None,
) -> None:
    """Train in the official AWS PyTorch container using SageMaker local mode (needs Docker)."""
    from fraud_scoring.launcher import LocalModeError, launch_local_training
    from fraud_scoring.training import Hyperparameters

    settings = get_settings()
    hp = Hyperparameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_dims=tuple(int(width) for width in hidden_dims.split(",") if width),
        dropout=dropout,
        pos_weight_power=pos_weight_power,
        patience=patience,
        seed=seed,
    )
    try:
        run = launch_local_training(
            data_dir=settings.processed_dir,
            jobs_dir=settings.jobs_dir,
            hyperparameters=hp,
            image=settings.training_image,
            job_name=job_name,
        )
    except LocalModeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]trained[/green] {run.job_name}")
    console.print(f"  model artifact: {run.model_tar}")
    console.print(f"  unpacked:       {run.model_dir}")


@app.command()
def evaluate(
    model_dir: Annotated[
        Path | None, typer.Option(help="Defaults to the latest local training job.")
    ] = None,
    bootstrap: Annotated[int, typer.Option(help="Bootstrap resamples for the intervals.")] = 1000,
) -> None:
    """Score the held-out test split and compare against baselines."""
    from fraud_scoring.evaluation import evaluate_artifact
    from fraud_scoring.launcher import LocalModeError, latest_model_dir

    settings = get_settings()
    try:
        model_dir = model_dir or latest_model_dir(settings.jobs_dir)
    except LocalModeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    report = evaluate_artifact(
        model_dir,
        settings.processed_dir / "test",
        train_dir=settings.processed_dir / "train",
        n_bootstrap=bootstrap,
    )
    (model_dir.parent / "evaluation.json").write_text(json.dumps(report, indent=2))

    def interval(info: dict) -> str:
        low, high = info["pr_auc_ci95"]
        return f"{info['pr_auc']:.3f}  [{low:.3f}, {high:.3f}]"

    table = Table(
        title=f"test split: {report['test_rows']:,} rows, {report['test_frauds']} frauds "
        "(PR-AUC with 95% bootstrap interval)"
    )
    table.add_column("scorer")
    table.add_column("PR-AUC", justify="right")
    table.add_row(
        "random scores (prevalence)", f"{report['baselines']['prevalence']['pr_auc']:.3f}"
    )
    if "logistic_regression" in report["baselines"]:
        table.add_row("logistic regression", interval(report["baselines"]["logistic_regression"]))
    table.add_row("PyTorch MLP (this model)", interval(report["model"]))
    console.print(table)

    at = report["model"]["at_threshold"]
    console.print(
        f"at threshold {at['threshold']:.4f} (chosen on validation): "
        f"precision {at['precision']:.3f}, recall {at['recall']:.3f}, F1 {at['f1']:.3f} "
        f"(TP {at['tp']}, FP {at['fp']}, FN {at['fn']})"
    )
    console.print(f"saved {model_dir.parent / 'evaluation.json'}")


def _resolve_model_dir(model_dir: Path | None) -> Path:
    from fraud_scoring.launcher import LocalModeError, latest_model_dir

    try:
        return model_dir or latest_model_dir(get_settings().jobs_dir)
    except LocalModeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _format_probability(value: float) -> str:
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.1e}"


@contextmanager
def _serving(url: str | None, model_dir: Path | None, port: int) -> Iterator[str]:
    """Yield the base URL of a running endpoint: the one given, or a temporary local one."""
    from fraud_scoring.serving import LocalEndpoint

    if url:
        yield url
        return
    settings = get_settings()
    endpoint = LocalEndpoint(
        _resolve_model_dir(model_dir),
        settings.inference_image,
        port=port,
        region=settings.aws_region,
        log_path=settings.artifact_dir / "serving" / f"endpoint-{port}.log",
    )
    with console.status("starting the SageMaker inference container (about 30 s)..."):
        endpoint.start()
    try:
        yield endpoint.url
    finally:
        with console.status("stopping the endpoint..."):
            endpoint.stop()


@app.command()
def predict(
    file: Annotated[
        Path | None,
        typer.Option(help='JSON file: a list of transactions or {"instances": [...]}.'),
    ] = None,
    sample: Annotated[
        int, typer.Option(help="Real test-split transactions to score when --file is not given.")
    ] = 6,
    frauds: Annotated[int, typer.Option(help="How many of the sampled ones are known frauds.")] = 2,
    seed: Annotated[int, typer.Option(help="Seed for choosing the sample.")] = 7,
    url: Annotated[
        str | None,
        typer.Option(help="Use a running endpoint (see `serve`) instead of starting a new one."),
    ] = None,
    model_dir: Annotated[
        Path | None, typer.Option(help="Defaults to the latest local training job.")
    ] = None,
    port: Annotated[int, typer.Option(help="Port for the temporary endpoint.")] = 8080,
) -> None:
    """Send transactions to a real-time endpoint and show the fraud scores.

    Without --url this starts the model in the AWS inference container (about 30 seconds),
    scores, and shuts it down again.
    """
    from fraud_scoring.client import EndpointError, invoke
    from fraud_scoring.launcher import LocalModeError
    from fraud_scoring.samples import sample_transactions

    settings = get_settings()
    labels: list[int] | None = None
    if file is not None:
        payload = json.loads(file.read_text())
        instances = payload["instances"] if isinstance(payload, dict) else payload
    else:
        test_dir = settings.processed_dir / "test"
        if not test_dir.is_dir():
            console.print(f"[red]{test_dir} not found; run `fraud-scoring data prepare`[/red]")
            raise typer.Exit(1)
        instances, labels = sample_transactions(test_dir, count=sample, frauds=frauds, seed=seed)

    try:
        with _serving(url, model_dir, port) as base_url:
            result = invoke(base_url, instances)
    except (EndpointError, LocalModeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    predictions = result["predictions"]
    table = Table(title=f"fraud scores (flag when probability >= {result['threshold']:.4f})")
    columns = ["#", "Amount", "fraud probability", "decision"]
    if labels is not None:
        columns.insert(2, "actual")
    for column in columns:
        table.add_column(column, justify="left" if column in ("decision", "actual") else "right")
    correct = 0
    for index, (instance, prediction) in enumerate(zip(instances, predictions, strict=True)):
        flagged = prediction["flagged"]
        row = [str(index + 1), f"{instance['Amount']:,.2f}"]
        if labels is not None:
            row.append("FRAUD" if labels[index] else "legit")
            correct += int(bool(labels[index]) == flagged)
        row += [
            _format_probability(prediction["fraud_probability"]),
            "[red]FLAGGED[/red]" if flagged else "ok",
        ]
        table.add_row(*row)
    console.print(table)
    flagged_count = sum(p["flagged"] for p in predictions)
    summary = f"{flagged_count} of {len(predictions)} flagged"
    if labels is not None:
        summary += f"; {correct} of {len(labels)} decisions match the true label"
    console.print(summary)


@app.command()
def serve(
    model_dir: Annotated[
        Path | None, typer.Option(help="Defaults to the latest local training job.")
    ] = None,
    port: Annotated[int, typer.Option(help="Host port for the endpoint.")] = 8080,
    workers: Annotated[int, typer.Option(help="TorchServe workers (each loads PyTorch).")] = 2,
) -> None:
    """Run the model as a local real-time endpoint until you press Ctrl+C."""
    import os
    import time

    from fraud_scoring.launcher import LocalModeError
    from fraud_scoring.samples import sample_transactions
    from fraud_scoring.serving import LocalEndpoint

    settings = get_settings()
    serving_dir = settings.artifact_dir / "serving"
    endpoint = LocalEndpoint(
        _resolve_model_dir(model_dir),
        settings.inference_image,
        port=port,
        workers=workers,
        region=settings.aws_region,
        log_path=serving_dir / f"endpoint-{port}.log",
    )
    try:
        with console.status("starting the SageMaker inference container (about 30 s)..."):
            endpoint.start()
    except LocalModeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        console.print(
            f"[green]endpoint ready[/green] at {endpoint.url}  (POST /invocations, GET /ping)"
        )
        console.print(f"container log: {endpoint.log_path}")
        console.print("\nIn another terminal:")
        console.print(f"  fraud-scoring predict --url {endpoint.url}")
        console.print(f"  fraud-scoring try --url {endpoint.url}    (type your own values)")
        test_dir = settings.processed_dir / "test"
        if test_dir.is_dir():
            instances, _ = sample_transactions(test_dir, count=3, frauds=1)
            example = serving_dir / "example_request.json"
            example.write_text(json.dumps({"instances": instances}, indent=2))
            curl = "curl.exe" if os.name == "nt" else "curl"
            console.print(
                f"  {curl} -s -X POST {endpoint.url}/invocations "
                f'-H "Content-Type: application/json" -d @{example}'
            )
        console.print("\nPress Ctrl+C to stop.")
        while endpoint.alive():
            time.sleep(1)
        console.print("[red]the endpoint process exited unexpectedly[/red]")
    except KeyboardInterrupt:
        pass
    finally:
        with console.status("stopping the endpoint and removing its container..."):
            endpoint.stop()
    console.print("stopped")


@app.command("try")
def try_model(
    url: Annotated[
        str | None,
        typer.Option(help="Use a running endpoint (see `serve`) instead of starting a new one."),
    ] = None,
    model_dir: Annotated[
        Path | None, typer.Option(help="Defaults to the latest local training job.")
    ] = None,
    port: Annotated[int, typer.Option(help="Port for the temporary endpoint.")] = 8080,
) -> None:
    """Type field values and watch the fraud score change (an interactive what-if).

    The model scores numbers, not text: V1..V28 (anonymised), Amount and Time. Start from a
    typical transaction or a real one from the test split, then change any field.
    """
    import secrets

    from rich.markup import escape

    from fraud_scoring.client import EndpointError, invoke
    from fraud_scoring.data import read_split
    from fraud_scoring.explore import (
        FIELD_BY_LOWER_NAME,
        HELP,
        CommandError,
        Session,
        format_score,
        parse_assignments,
        parse_json_fields,
        typical_transaction,
    )
    from fraud_scoring.launcher import LocalModeError
    from fraud_scoring.samples import sample_transactions

    processed = get_settings().processed_dir
    if not (processed / "train").is_dir() or not (processed / "test").is_dir():
        console.print(f"[red]{processed} not found; run `fraud-scoring data prepare`[/red]")
        raise typer.Exit(1)
    session = Session(
        typical_transaction(read_split(processed / "train")),
        origin="a typical legitimate transaction (the median of each field)",
    )

    def score(base_url: str) -> None:
        try:
            result = invoke(base_url, [session.current])
        except EndpointError as exc:
            console.print(
                f"[red]the endpoint rejected this transaction:[/red] {escape(exc.detail)}"
            )
            return
        prediction, threshold = result["predictions"][0], result["threshold"]
        probability = prediction["fraud_probability"]
        verdict = "[red]FLAGGED as fraud[/red]" if prediction["flagged"] else "[green]ok[/green]"
        console.print(
            f"fraud probability [bold]{_format_probability(probability)}[/bold]  "
            f"{escape(format_score(probability, threshold))}  {verdict}"
        )

    def show(everything: bool) -> None:
        changes = session.changed()
        if everything:
            for name, value in session.current.items():
                console.print(f"  {name:6s} {value:>14,.4f}", highlight=False)
        elif changes:
            for name, (before, now) in changes.items():
                console.print(f"  {name:6s} {before:>12,.4f} -> {now:,.4f}", highlight=False)
        else:
            console.print(f"  no changes yet; starting from {session.origin}")

    def repl(base_url: str) -> None:
        console.print(f"Starting from {session.origin}.")
        console.print(HELP, markup=False, highlight=False)
        console.print()
        score(base_url)
        while True:
            try:
                line = input("try> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return
            if not line:
                continue
            head, _, rest = line.partition(" ")
            word = head.lower()
            try:
                if word in ("quit", "exit", "q"):
                    return
                if word == "help":
                    console.print(HELP, markup=False, highlight=False)
                elif word == "show":
                    show(rest.strip().lower() == "all")
                elif word == "reset":
                    session.reset()
                    score(base_url)
                elif word == "random":
                    kind = rest.strip().lower()
                    if kind not in ("fraud", "legit"):
                        raise CommandError("use 'random fraud' or 'random legit'")
                    instances, labels = sample_transactions(
                        processed / "test",
                        count=1,
                        frauds=1 if kind == "fraud" else 0,
                        seed=secrets.randbelow(2**31),
                    )
                    truth = "FRAUD" if labels[0] else "legitimate"
                    session.load(instances[0], f"a real test-split transaction (actually {truth})")
                    console.print(f"loaded {session.origin}")
                    score(base_url)
                elif line.startswith("{"):
                    session.apply(parse_json_fields(line))
                    score(base_url)
                elif word == "set":
                    session.apply(parse_assignments(rest))
                    score(base_url)
                elif "=" in line:
                    session.apply(parse_assignments(line))
                    score(base_url)
                elif word == "hour" or word in FIELD_BY_LOWER_NAME:
                    name = "hour" if word == "hour" else FIELD_BY_LOWER_NAME[word]
                    raise CommandError(f"to change {name}, type {name}=VALUE, e.g. {name}=5")
                else:
                    raise CommandError(f"unknown command {head!r}; type help")
            except CommandError as exc:
                console.print(f"[red]{escape(str(exc))}[/red]")

    try:
        with _serving(url, model_dir, port) as base_url:
            repl(base_url)
    except (EndpointError, LocalModeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
