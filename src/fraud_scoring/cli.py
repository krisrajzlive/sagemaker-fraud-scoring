"""Command-line interface: ``fraud-scoring --help``."""

from __future__ import annotations

import json
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
