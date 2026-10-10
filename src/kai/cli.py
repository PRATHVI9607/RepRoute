"""`kai` command line: train folds, run experiments, build bundles, run the edge and the cloud."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
Overrides = Annotated[list[str] | None, typer.Argument(help="Hydra overrides, e.g. scenario=mini")]


@app.command()
def folds(
    repeats: int = 1,
    folds: int = 5,
    level: str = "L1",
    seed: int = 0,
    epochs_raw: int = 30,
    epochs_latent: int = 15,
    f_raw: str = "spec",
    head_trials: int = 10,
    device: str = "auto",
    force: bool = False,
    overrides: Overrides = None,
) -> None:
    """Train models and build the Disagreement Table for leave-bearings-out folds."""
    from kai.bench.config import load_config, work
    from kai.contracts.windows import WindowSet
    from kai.data.splits import load_manifest
    from kai.tasks.pipeline import FoldConfig, run_fold

    cfg = load_config(overrides)
    ws = WindowSet.load(work() / cfg.dataset.windows)
    (manifest_path,) = sorted(
        (work() / cfg.dataset.splits_dir).glob(f"{ws.dataset_id}_{level}_*.json")
    )
    manifest = load_manifest(manifest_path)
    fold_cfg = FoldConfig(
        f_raw, epochs_raw, epochs_latent, head_trials=head_trials, seed=seed, device=device
    )
    for fold in manifest.folds:
        if fold.repeat >= repeats or fold.fold >= folds:
            continue
        out = (
            work()
            / cfg.dataset.folds_dir
            / manifest.split_id
            / f"s{seed}_r{fold.repeat}f{fold.fold}"
        )
        if (out / "metrics.json").exists() and not force:
            typer.echo(f"skip {out.name} (done)")
            continue
        t0 = time.perf_counter()
        _, metrics = run_fold(ws, fold, fold_cfg, out, work() / "data/interim/cache")
        f1 = {k.split("/")[1]: round(v, 3) for k, v in metrics.items() if k.startswith("macro_f1/")}
        dis = {
            k.split("/")[2]: round(v, 3)
            for k, v in metrics.items()
            if k.startswith("disagreement/test/")
        }
        typer.echo(
            f"{out.name} {time.perf_counter() - t0:.0f}s\n  macro_f1 {json.dumps(f1)}"
            f"\n  disagree {json.dumps(dis)}"
        )


@app.command()
def synth(seed: int = 0, bearings_per_class: int = 10, windows_per_condition: int = 8) -> None:
    """Write the synthetic fixture dataset and its split manifests (smoke tests, CI mini-paper)."""
    import numpy as np

    from kai.bench.config import work
    from kai.data.splits import make_l1_splits, make_l2_splits, save_manifest
    from kai.data.synthetic import CONDITIONS, make_synthetic

    rng = np.random.default_rng(seed)
    ws = make_synthetic(rng, bearings_per_class, windows_per_condition)
    ws.save(work() / "data/processed/synthetic.npz")
    l1 = make_l1_splits(ws.bearing_labels(), ws.dataset_id, seed, n_folds=3, n_repeats=2)
    for manifest in (l1, make_l2_splits(l1, list(CONDITIONS))):
        typer.echo(save_manifest(manifest, work() / "data/splits"))


@app.command()
def exp(name: str, overrides: Overrides = None) -> None:
    """Run one experiment (e1 … e8) and print its summary."""
    from kai.bench import experiments
    from kai.bench.config import load_config

    table = {fn.__name__.split("_")[0]: fn for fn in experiments.EXPERIMENTS}
    if name not in table:
        raise typer.BadParameter(f"unknown experiment {name!r}; choose from {sorted(table)}")
    summary = table[name](load_config(overrides))
    typer.echo(json.dumps(summary, indent=1, default=float))


@app.command()
def bundle(fold: Path, overrides: Overrides = None) -> None:
    """Freeze one trained fold into an immutable calibration bundle for the chosen price preset."""
    from kai.bench.bundle import BundleMeta, build_bundle
    from kai.bench.config import energy_of, load_config, prices_of, work
    from kai.bench.experiments import load_profiles
    from kai.contracts.windows import WindowSet

    cfg = load_config(overrides)
    ws = WindowSet.load(work() / cfg.dataset.windows)
    meta = BundleMeta.from_fold(fold, ws.dataset_id, ws.sample_rate_hz)
    profiles, overheads = load_profiles()
    out = work() / "artifacts" / "bundles"
    path = build_bundle(
        fold, out, profiles, overheads, meta, prices_of(cfg), energy_of(cfg), cfg.profile.mission
    )
    typer.echo(path)


@app.command()
def replay(fold: Path, out: Path, n: int = 2000, overrides: Overrides = None) -> None:
    """Export the test windows of a fold as a replay file for `kai edge` (Docker, Raspberry Pi)."""
    import numpy as np

    from kai.bench.config import load_config, work
    from kai.contracts.windows import WindowSet
    from kai.tasks.pipeline import Table

    ws = WindowSet.load(work() / load_config(overrides).dataset.windows)
    test = np.flatnonzero(Table.load(fold / "table.npz").of("test"))[:n]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=ws.x[test], window_id=ws.window_id[test])
    typer.echo(f"{len(test)} windows → {out}")


@app.command()
def cloud(bundle_dir: Path, host: str = "0.0.0.0", port: int = 7400) -> None:
    """Run the cloud server for one bundle."""
    from kai.cloud.server import serve

    asyncio.run(serve(bundle_dir, host, port))


@app.command()
def edge(
    bundle_dir: Path,
    windows: Path,
    host: str = "127.0.0.1",
    port: int = 7400,
    n: int = 500,
    alpha: float = 0.05,
    deadline_ms: float = 250.0,
    selector: str = "s1_greedy_safe",
    p_audit: float = 0.0,
    shape_bps: float | None = None,
    telemetry: Path = Path("experiments/edge_telemetry.msgpack"),
) -> None:
    """Replay windows through the edge runtime against a running cloud.

    Needs only the edge install (numpy + onnxruntime): prices, energy model and mission come from
    the bundle, so there is no torch and no config system on the device.
    """
    import numpy as np

    from kai.edge.bundle import load_bundle
    from kai.edge.runtime import EdgeConfig, run_edge

    with np.load(windows) as z:
        x, ids = z["x"][:n], z["window_id"][:n]
    b = load_bundle(bundle_dir)
    period = 1000.0 * b.manifest.window_n / b.manifest.sample_rate_hz
    edge_cfg = EdgeConfig(
        alpha=alpha, selector=selector, mission=b.mission, deadline_ms=deadline_ms,
        period_ms=period, p_audit=p_audit, shape_bps=shape_bps,
    )  # fmt: skip
    records = asyncio.run(run_edge(b, edge_cfg, b.prices, b.energy, x, ids, host, port, telemetry))
    done = [r for r in records if "latency_ms" in r]
    typer.echo(f"{len(records)} windows, {len(done)} delivered, telemetry → {telemetry}")


if __name__ == "__main__":
    app()
