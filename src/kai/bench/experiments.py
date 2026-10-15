"""Experiments E1–E8 and E10 (PRD §9.4). Each writes `experiments/<id>/` and a run-registry row."""

from __future__ import annotations

import json
import multiprocessing as mp
import platform
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from kai.analysis import stats
from kai.bench import evaluate as ev
from kai.bench.calibrate import calibrate, threshold
from kai.bench.config import as_dict, energy_of, prices_of, work
from kai.bench.registry import record
from kai.certify import crc
from kai.certify.runtime import Certifier
from kai.cloud.heads import HeadZoo
from kai.contracts.schemas import RepProfile
from kai.contracts.windows import WindowSet
from kai.data.splits import load_manifest
from kai.edge.bundle import Bundle
from kai.link.des import Result
from kai.profile.profiler import co_tenants, measure_overheads, profile_menu, profile_pipeline
from kai.select.base import Window
from kai.select.cost import CostModel, Overheads, Profiles, State
from kai.select.policies import GreedySafe
from kai.tasks.pipeline import Table


def profile_path() -> Path:
    return work() / "artifacts" / "profiles" / "local.json"


# --- shared plumbing -----------------------------------------------------------------------------


def split_id(cfg: DictConfig, level: str = "L1") -> str:
    (path,) = sorted((work() / cfg.dataset.splits_dir).glob(f"{cfg.dataset.id}_{level}_*.json"))
    return load_manifest(path).split_id


def fold_dirs(cfg: DictConfig, level: str = "L1") -> list[Path]:
    """Finished folds of this dataset, split level and training seed, in a stable order.

    Folds of other seeds share test bearings with these and are not further replication units.
    """
    base = work() / cfg.dataset.folds_dir / split_id(cfg, level)
    return sorted(p.parent for p in base.glob(f"s{int(cfg.seed)}_*/metrics.json"))


def load_profiles(
    path: Path | None = None, edge_slowdown: float = 1.0, n_sensors: int = 1
) -> tuple[list[RepProfile], Overheads]:
    """Measured profiles. `edge_slowdown` scales edge CPU times to model a slower edge device.

    `n_sensors` picks the timings measured at that node's duty cycle (one window every
    period / n_sensors): the profile of the largest measured sensor count not above it.
    """
    data = json.loads((path or profile_path()).read_text())
    measured = [int(k) for k in data.get("busy", {}) if int(k) <= n_sensors]
    if measured:
        data = data["busy"][str(max(measured))]
    profiles = [RepProfile(**p) for p in data["profiles"]]
    ov = Overheads(**data["overheads"])
    if edge_slowdown != 1.0:
        profiles = [
            p.model_copy(
                update={
                    k: getattr(p, k) * edge_slowdown
                    for k in ("t_enc_ms", "t_enc_mad_ms", "t_enc_q95_ms")
                }
            )
            for p in profiles
        ]
        ov = replace(
            ov,
            t_probe_ms=ov.t_probe_ms * edge_slowdown,
            t_pred_ms=ov.t_pred_ms * edge_slowdown,
            t_sel_ms=ov.t_sel_ms * edge_slowdown,
            t_io_ms=ov.t_io_ms * edge_slowdown,
        )
    return profiles, ov


def window_period_ms(cfg: DictConfig) -> float:
    ws = WindowSet.load(work() / cfg.dataset.windows)
    return 1000.0 * ws.window_n / ws.sample_rate_hz


def out_dir(name: str) -> Path:
    path = work() / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def finish(
    name: str, cfg: DictConfig, summary: dict[str, Any], frame: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Write the summary (and the per-run table), and register the run."""
    out = out_dir(name)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if frame is not None:
        frame.to_parquet(out / "results.parquet", index=False)
    flat = {k: float(v) for k, v in summary.items() if isinstance(v, int | float | np.floating)}
    record(name, as_dict(cfg), int(cfg.seed), flat)
    return summary


def _governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return path.read_text().strip() if path.exists() else "unknown"


# --- E1: representation profiling ----------------------------------------------------------------


def e1_profile(cfg: DictConfig, repeats: int = 30) -> dict:
    """Size, encode / decode / head time, time-vs-load curve, fidelity and agreement per rep."""
    folds = fold_dirs(cfg)
    ws = WindowSet.load(work() / cfg.dataset.windows)
    table = Table.load(folds[0] / "table.npz")
    zoo = HeadZoo(folds[0])
    reps = [zoo.reps[r] for r in table.rep_ids]
    sample = ws.x[
        np.flatnonzero(table.of("test"))[:: max(1, int(table.of("test").sum()) // 64)][:64]
    ]
    # one call per window period: the duty cycle the edge really runs at (see `time_ms`)
    period = 1000.0 * ws.window_n / ws.sample_rate_hz
    profiles, curves = profile_menu(reps, zoo, sample, repeats, period_ms=period)
    scored = ev.scored_fold(folds[0], seed=int(cfg.seed))
    plan = CostModel(Profiles.from_list(profiles), Overheads(), energy_of(cfg), prices_of(cfg))
    cert = Certifier(
        threshold(calibrate(*scored.part("calib"), table.rep_ids), "v0", 0.05, table.rep_ids)
    )
    greedy, state = GreedySafe(plan), State(1e6, 20.0, 0.0, 0.0, 250.0)
    phi0 = table.phi[0].astype(np.float64)

    def select(ghat: np.ndarray) -> object:
        return greedy.select(Window(cert.safe_set(ghat), ghat, phi0), state)

    overheads = measure_overheads(sample, scored.predictor, select, repeats, period_ms=period)
    # A node with n sensors handles a window every period / n and is never idle between its
    # stages: time the stages inside the whole per-window pipeline at that period (ADR 0008).
    # Decoder and head run in the cloud and keep their timings; the load exponents are those
    # fitted at one sensor.
    busy: dict[str, dict[str, Any]] = {}
    for n in sorted({int(k) for k in cfg.scenario.sensors} - {1}):
        with co_tenants(0):
            encoders, ov_n = profile_pipeline(
                reps, sample, scored.predictor, select, period / n, repeats
            )
        busy[str(n)] = {
            "period_ms": period / n,
            "profiles": [
                base.model_copy(
                    update={"t_enc_ms": med, "t_enc_mad_ms": mad, "t_enc_q95_ms": q95}
                ).model_dump()
                for base, (med, mad, q95) in zip(profiles, encoders, strict=True)
            ],
            "overheads": asdict(replace(ov_n, load_exponent=overheads.load_exponent)),
        }
    document: dict[str, Any] = {
        "machine": {
            "node": platform.node(),
            "cpu": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "energy_source": "modelled",
        "timing": {"duty_cycle_period_ms": period, "governor": _governor()},
        "profiles": [p.model_dump() for p in profiles],
        "overheads": asdict(overheads),
        "busy": busy,
        "load_curves": {r: {str(k): v for k, v in c.items()} for r, c in curves.items()},
    }
    profile_path().parent.mkdir(parents=True, exist_ok=True)
    profile_path().write_text(json.dumps(document, indent=1))
    # the runtime's per-frame overhead is calibrated against the profile just written
    t_io_ms, io_exponent, io_excess = _calibrate_io(cfg)
    overheads = replace(overheads, t_io_ms=t_io_ms, io_load_exponent=io_exponent)
    for entry in (document, *busy.values()):
        entry["overheads"] |= {"t_io_ms": t_io_ms, "io_load_exponent": io_exponent}
    profile_path().write_text(json.dumps(document, indent=1))
    per_fold = [json.loads((f / "metrics.json").read_text()) for f in folds]
    rows = []
    for p in profiles:
        f1 = [m[f"macro_f1/{p.rep_id}"] for m in per_fold]
        dis = [m[f"disagreement/test/{p.rep_id}"] for m in per_fold]
        rows.append(
            {
                **p.model_dump(),
                "macro_f1_mean": np.mean(f1), "macro_f1_sd": np.std(f1),
                "disagreement_mean": np.mean(dis), "disagreement_sd": np.std(dis),
                "t_enc_ms_load0.5": curves[p.rep_id].get(0.5, {}).get("mean_ms", np.nan),
                "t_enc_ms_load0.8": curves[p.rep_id].get(0.8, {}).get("mean_ms", np.nan),
            }
        )  # fmt: skip
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir("e1") / "profile_table.csv", index=False)
    majority = float(np.mean([m["majority_baseline_macro_f1"] for m in per_fold]))
    summary: dict[str, Any] = {
        "n_folds": len(folds),
        "n_reps_profiled": len(profiles),
        "f_raw_macro_f1": float(frame.loc[0, "macro_f1_mean"]),
        "majority_macro_f1": majority,
        "t_probe_ms": overheads.t_probe_ms,
        "t_pred_ms": overheads.t_pred_ms,
        "t_sel_ms": overheads.t_sel_ms,
        "mean_load_exponent": float(frame["load_exponent"].mean()),
        "t_io_ms": t_io_ms,
        "io_load_exponent": io_exponent,
        "io_excess_ms/load0": io_excess[0.0],
        "io_excess_ms/load0.8": io_excess[0.8],
        "f_raw": str(cfg.dataset.f_raw),
        "bottleneck": str(cfg.dataset.bottleneck),
    }
    r6 = table.rep_ids.index("r6_bottleneck") if "r6_bottleneck" in table.rep_ids else 1
    base = {"profiles": [p.model_dump() for p in profiles], "overheads": asdict(overheads)}
    for sensors, entry in {"1": base, **busy}.items():
        summary[f"overhead_ms/sensors{sensors}"] = Overheads(**entry["overheads"]).total_ms
        summary[f"t_enc_ms/{table.rep_ids[r6]}/sensors{sensors}"] = entry["profiles"][r6][
            "t_enc_ms"
        ]
    return finish("e1", cfg, summary, frame)


def _calibrate_io(
    cfg: DictConfig, n_windows: int = 100, bps: float = 20e6
) -> tuple[float, float, dict[float, float]]:
    """Per-frame overhead of the real runtime that the micro-benchmarks do not see (ADR 0008).

    Two short closed-loop runs over loopback with the smallest representation on a fast link, at
    edge load 0 and 0.8. The median latency beyond what the model predicts is `t_io(ρ)`; the two
    points give `t_io(ρ) = t_io / (1 - ρ)**γ`. E4 and E7 then test the model at other rates,
    representations and loads.
    """
    from kai.bench.realrun import cloud_process, real_run
    from kai.edge.runtime import EdgeConfig
    from kai.profile.profiler import LOAD_TENANTS
    from kai.select.policies import Static

    bundle, path, ws, _, test = _bundle_for(cfg)
    prices, energy = prices_of(cfg), energy_of(cfg)
    model = CostModel(Profiles.from_list(bundle.profiles), bundle.overheads, energy, prices)
    rep = int(np.argmin(model.profiles.size_bytes))
    pick = test[:: max(1, len(test) // n_windows)][:n_windows]
    extra: dict[float, float] = {}
    with cloud_process(path) as port:
        for rho in (0.0, 0.8):
            edge_cfg = EdgeConfig(
                deadline_ms=1e9, period_ms=window_period_ms(cfg), load=rho, shape_bps=bps,
                initial_bps=bps,
            )  # fmt: skip
            recs = real_run(
                bundle, port, edge_cfg, prices, energy, ws.x[pick], ws.window_id[pick],
                Static(rep), LOAD_TENANTS[rho],
            )  # fmt: skip
            latencies = [x["latency_ms"] for x in recs if "latency_ms" in x]
            if not latencies:
                raise RuntimeError(f"runtime calibration at load {rho}: no window was delivered")
            measured = np.median(latencies)
            predicted = model.latency_ms(State(bps, 0.0, 0.0, rho, np.inf))[rep]
            extra[rho] = float(measured - predicted)
    shutil.rmtree(path)  # this bundle carries t_io = 0: never leave it for a later run to pick up
    t_io = max(extra[0.0], 0.05)  # a floor keeps the exponent defined when the model is spot on
    exponent = float(np.clip(np.log(max(extra[0.8], t_io) / t_io) / np.log(5.0), 0.0, 3.0))
    return t_io, exponent, extra


# --- E2: certificate validity --------------------------------------------------------------------


def e2_certificate(cfg: DictConfig) -> dict:  # noqa: PLR0915 - one recipe, read top to bottom
    """Held-out risk of V0 / V1 / V2 across α and folds (H2a, H2b)."""
    profiles, overheads = load_profiles()
    period = window_period_ms(cfg)
    rng = np.random.default_rng([int(cfg.seed), 2])
    alphas = tuple(cfg.alphas)
    rows: list[dict[str, Any]] = []
    pooled: dict[tuple[float, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for fold in fold_dirs(cfg):
        ctx = ev.load_fold(
            fold,
            profiles,
            overheads,
            prices_of(cfg),
            energy_of(cfg),
            alphas,
            cfg.delta,
            period,
            seed=int(cfg.seed),
        )
        t = ctx.scored.table
        test = t.of("test")
        ghat, d, blocks = ctx.scored.ghat[test], t.d[test], t.bearing_id[test]
        nbytes = t.payload_bytes[test]
        nominal = ev.Cell("static", 1e6, 250.0, 0.0, 1, 0.05)
        cost = np.where(
            ctx.model.feasible(ev.nominal_state(nominal)),
            ctx.model.cost(ev.nominal_state(nominal)),
            np.inf,
        )
        # Exchangeability control: calibrate on a random half of the calibration windows and test
        # on the other half. A random halving of a fixed set is exchangeable by construction, so
        # the premise of the theorem holds and V0 must stay within α on average; the gap to the
        # test-bearing risk is the effect of bearing shift. A diagnostic of the method, never a
        # generalisation result.
        control_rng = np.random.default_rng([int(cfg.seed), 22])
        control_runs: dict[float, list[float]] = {a: [] for a in alphas}
        for _ in range(20):  # one random halving is as noisy as the effect it should rule out
            cal = control_rng.permutation(np.flatnonzero(t.of("calib")))
            half_a, half_b = cal[: len(cal) // 2], cal[len(cal) // 2 :]
            m_a = crc.first_loss_threshold(ctx.scored.ghat[half_a], t.d[half_a])
            for a in alphas:
                lam_a = crc.crc_threshold(m_a, a)
                loss = crc.safe_set_loss(ctx.scored.ghat[half_b].copy(), t.d[half_b], lam_a)
                control_runs[a].append(float(loss.mean()))
        control = {a: float(np.mean(v)) for a, v in control_runs.items()}
        for alpha in alphas:
            lams = {
                v: threshold(ctx.lambdas[alpha], v, alpha, t.rep_ids)
                for v in ("v0", "v1", "v1_per_rep")
            }
            lams["v2"] = ev.v2_threshold(ctx, replace(nominal, alpha=alpha))
            # constant scores: the safe set is the same for every window (per-rep thresholds ±1)
            lams["v0_static"] = np.where(ctx.static_safe[alpha], crc.ALL, crc.RAW_ONLY)
            prior, lams["v0_cascade"] = ctx.cascade[alpha]
            for variant, lam in lams.items():
                scores = np.minimum(ghat, prior) if variant == "v0_cascade" else ghat
                safe = crc.safe_set(scores, lam)
                worst = (safe & d).any(
                    axis=1
                )  # safe-set loss: the risk of the worst admissible policy
                choice = np.where(safe, cost, np.inf).argmin(axis=1)
                greedy = d[np.arange(len(d)), choice]
                loss = greedy if variant == "v2" else worst  # V2 only promises the greedy policy
                best_bytes = np.where(safe, nbytes, np.inf).min(axis=1)
                rows.append(
                    {
                        "fold": fold.name, "alpha": alpha, "variant": variant,
                        "risk": float(loss.mean()), "risk_worst_policy": float(worst.mean()),
                        "risk_greedy_policy": float(greedy.mean()),
                        "safe_set_size": float(safe.sum(axis=1).mean()),
                        "non_raw_available": float(safe[:, 1:].any(axis=1).mean()),
                        "bytes_vs_raw": float(best_bytes.mean() / nbytes[:, 0].mean()),
                        # the halving control is computed for the plain V0 scores only
                        "risk_exchangeable_control": control[alpha] if variant == "v0" else np.nan,
                        "n_calib": int(t.of("calib").sum()), "n_test": int(test.sum()),
                    }
                )  # fmt: skip
                key = (alpha, variant)
                pooled.setdefault(key, ([], []))
                pooled[key][0].append(loss)
                pooled[key][1].append(np.char.add(fold.name + "/", blocks))
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {"n_folds": int(frame["fold"].nunique())}
    for (alpha, variant), (losses, blocks_) in pooled.items():
        lo, hi = stats.block_bootstrap_ci(np.concatenate(losses), np.concatenate(blocks_), rng)
        sub = frame[(frame.alpha == alpha) & (frame.variant == variant)]
        tag = f"{variant}/a{alpha:g}"
        summary[f"{tag}/risk_mean"] = float(sub.risk.mean())
        summary[f"{tag}/risk_ci_hi"] = hi
        summary[f"{tag}/risk_ci_lo"] = lo
        summary[f"{tag}/folds_within_alpha"] = float((sub.risk <= alpha).mean())
        summary[f"{tag}/bytes_vs_raw"] = float(sub.bytes_vs_raw.mean())
        summary[f"{tag}/non_raw_available"] = float(sub.non_raw_available.mean())
        if variant == "v0":
            summary[f"control/a{alpha:g}/risk_same_bearings"] = float(
                sub.risk_exchangeable_control.mean()
            )
    summary["H2a_pass"] = bool(
        all(
            summary[f"v0/a{a:g}/risk_mean"] <= a and summary[f"v0/a{a:g}/risk_ci_hi"] <= a + 0.01
            for a in alphas
        )
    )
    summary["H2b_pass"] = bool(all(summary[f"v1/a{a:g}/folds_within_alpha"] >= 0.9 for a in alphas))
    return finish("e2", cfg, summary, frame)


# --- E3 / E5: scenario-grid comparisons ----------------------------------------------------------

_CTX: dict[str, ev.FoldCtx] = {}


def _cell_job(args: tuple) -> list[dict]:
    fold, cell, cfg_dict, which, mission, edge_slowdown, only = args
    key = f"{fold}|{edge_slowdown}|{cfg_dict['profile']['prices']['name']}|{cell.n_sensors}"
    if key not in _CTX:  # one context per worker process, fold and duty cycle
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(cfg_dict)
        profiles, overheads = load_profiles(edge_slowdown=edge_slowdown, n_sensors=cell.n_sensors)
        _CTX[key] = ev.load_fold(
            Path(fold), profiles, overheads, prices_of(cfg), energy_of(cfg), tuple(cfg.alphas),
            cfg.delta, cfg_dict["_period_ms"], seed=int(cfg_dict["seed"]),
        )  # fmt: skip
    rows = ev.run_cell(
        _CTX[key], cell, mission, which, cfg_dict["scenario"]["n_windows"], cfg_dict["seed"],
        only, late_cap_factor=float(cfg_dict["late_cap_factor"]),
    )  # fmt: skip
    return [{"fold": Path(fold).name, **asdict(cell), **r} for r in rows]


def grid_cells(cfg: DictConfig, alphas: tuple[float, ...] | None = None) -> list[ev.Cell]:
    sc = cfg.scenario
    links = [("static", b) for b in sc.bandwidths_bps] + [(t, 1e6) for t in sc.traces]
    return [
        ev.Cell(link, float(bps), float(d), float(rho), int(s), float(a), rtt_ms=float(sc.rtt_ms))
        for (link, bps), d, rho, s, a in product(
            links, sc.deadlines_ms, sc.loads, sc.sensors, alphas or cfg.alphas
        )
    ]


def run_grid(
    cfg: DictConfig,
    cells: list[ev.Cell],
    which: str,
    edge_slowdown: float = 1.0,
    max_folds: int | None = None,
    only: tuple[str, ...] = (),
) -> pd.DataFrame:
    cfg_dict = {**as_dict(cfg), "_period_ms": window_period_ms(cfg)}
    folds = fold_dirs(cfg)[:max_folds]
    jobs = [
        (str(f), c, cfg_dict, which, cfg.profile.mission, edge_slowdown, only)
        for f in folds
        for c in cells
    ]
    with ProcessPoolExecutor(int(cfg.workers), mp_context=mp.get_context("spawn")) as pool:
        rows = [r for part in pool.map(_cell_job, jobs, chunksize=4) for r in part]
    return pd.DataFrame(rows)


def stored_or_run(
    cfg: DictConfig, name: str, run: Callable[[tuple[str, ...]], pd.DataFrame]
) -> pd.DataFrame:
    """With `reuse=true`, re-analyse the stored per-cell results instead of simulating again;
    `add_policies=[…]` then simulates only those policies and replaces their stored rows."""
    stored = out_dir(name) / "results.parquet"
    if not (cfg.get("reuse") and stored.exists()):
        return run(())
    frame = pd.read_parquet(stored)
    add = tuple(cfg.get("add_policies") or ())
    if add:
        frame = pd.concat([frame[~frame.policy.isin(add)], run(add)], ignore_index=True)
    return frame


CELL_KEYS = ["link", "bandwidth_bps", "deadline_ms", "load", "n_sensors", "alpha"]


def totals(frame: pd.DataFrame, metric: str = "cost") -> pd.DataFrame:
    """fold × policy table of the metric summed over the grid (the pre-registered 'total cost')."""
    return frame.pivot_table(index="fold", columns="policy", values=metric, aggfunc="sum")


def reduction(
    frame: pd.DataFrame, ours: str, theirs: str, rng: np.random.Generator, metric: str = "cost"
) -> dict:
    """Relative reduction 1 - ours/theirs of the grid total, per fold, with a BCa CI over folds."""
    t = totals(frame, metric)
    t = t[t[theirs] > 0]  # a fold where the reference has nothing to reduce says nothing
    per_fold = 1.0 - t[ours] / t[theirs]
    lo, hi = stats.bca_ci(per_fold.to_numpy(), rng)
    cells = frame.pivot_table(index=["fold", *CELL_KEYS], columns="policy", values=metric)
    ratio = (cells[ours] / cells[theirs]).replace([np.inf, -np.inf], np.nan).dropna()
    geo = float(np.exp(np.log(ratio[ratio > 0]).mean())) if (ratio > 0).any() else float("nan")
    return {
        "mean": float(per_fold.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "geo_ratio": geo,
        "per_fold": per_fold.tolist(),
    }


def dominant_rep(frame: pd.DataFrame, policy: str) -> pd.Series:
    """The most-used representation of a policy in every (fold, cell)."""
    share_cols = [c for c in frame.columns if c.startswith("share/")]
    sub = frame[frame.policy == policy].set_index(["fold", *CELL_KEYS])[share_cols]
    out = sub.idxmax(axis=1).str.removeprefix("share/")
    out[sub.sum(axis=1) == 0] = "none"  # it sent nothing: not a vote for raw (column 0)
    return out


def e3_main(cfg: DictConfig) -> dict:
    """Kai versus baselines B1–B13 and the oracle across the scenario grid (H1, H3).

    Under another price preset (`profile=battery_node`, …) the run is written to
    `experiments/e3_<preset>/`; `max_folds=5` limits it to the first folds.
    """
    rng = np.random.default_rng([int(cfg.seed), 3])
    preset = str(cfg.profile.prices.name)
    name = "e3" if preset == "metered_cellular" else f"e3_{preset}"
    frame = stored_or_run(
        cfg,
        name,
        lambda only: run_grid(
            cfg, grid_cells(cfg), "main", max_folds=cfg.get("max_folds"), only=only
        ),
    )
    # share of all cells; cells where the oracle sent nothing count, but for no representation
    dom = dominant_rep(frame, "oracle").value_counts(normalize=True).drop("none", errors="ignore")
    h1_cost = reduction(frame, "oracle", "b4_best_single", rng)
    h3_cost = reduction(frame, "kai_s1_v0", "b4_best_single", rng)
    bursty = frame[frame.link != "static"]
    h3_slo = reduction(bursty, "kai_s1_v0", "b5_context_blind", rng, "slo_violation")
    summary: dict[str, Any] = {
        "n_folds": int(frame.fold.nunique()),
        "n_cells": len(frame.drop_duplicates(CELL_KEYS)),
        "H1/reps_dominant_in_10pct_of_cells": int((dom >= 0.10).sum()),
        "H1/b4_extra_cost_vs_oracle": float(1.0 / (1.0 - h1_cost["mean"]) - 1.0),
        "H3/cost_reduction_vs_b4": h3_cost["mean"],
        "H3/cost_reduction_vs_b4_ci_lo": h3_cost["ci_lo"],
        "H3/cost_reduction_vs_b4_ci_hi": h3_cost["ci_hi"],
        "H3/cost_geo_ratio_vs_b4": h3_cost["geo_ratio"],
        "H3/slo_reduction_vs_b5_bursty": h3_slo["mean"],
        "H3/slo_reduction_vs_b5_ci_lo": h3_slo["ci_lo"],
        "H3/slo_reduction_vs_b5_ci_hi": h3_slo["ci_hi"],
        "oracle_dominant_share": dom.round(4).to_dict(),
    }
    summary["H1_pass"] = bool(
        summary["H1/reps_dominant_in_10pct_of_cells"] >= 3
        and summary["H1/b4_extra_cost_vs_oracle"] >= 0.25
    )
    summary["H3_pass"] = bool(h3_cost["mean"] >= 0.20 and h3_slo["mean"] >= 0.30)
    by_policy = frame.groupby("policy")[
        [
            "cost",
            "bytes",
            "slo_violation",
            "risk_certified_joint",
            "risk_certified",
            "risk_sent",
            "rate_certified",
            "rate_deferred",
            "rate_uncertified",
            "rate_dropped",
            "latency_p50_ms",
            "latency_p95_ms",
            "energy_j",
        ]
    ].mean()
    by_policy["cost_total_per_fold"] = totals(frame).mean()
    by_policy.round(5).to_csv(out_dir(name) / "by_policy.csv")
    summary["per_fold"] = {
        "H3_cost": h3_cost["per_fold"],
        "H3_slo": h3_slo["per_fold"],
        "H1_cost": h1_cost["per_fold"],
    }
    summary["price_preset"] = preset
    # exploratory variants (ADR 0008, 0011): the same comparison, not part of H3
    for policy in ("kai_s1_v0_static", "kai_s1_cascade"):
        if policy in set(frame.policy):
            summary[f"x/{policy}/cost_reduction_vs_b4"] = reduction(
                frame, policy, "b4_best_single", rng
            )["mean"]
            summary[f"x/{policy}/probe_rate"] = (
                float(frame[frame.policy == policy].probe_rate.mean())
                if "probe_rate" in frame
                else 1.0
            )
    return finish(name, cfg, summary, frame)


def e5_ablation(cfg: DictConfig, edge_slowdown: float = 1.0, name: str = "e5") -> dict:
    """Mechanism ablations and the within-family comparison (H4, H5)."""
    rng = np.random.default_rng([int(cfg.seed), 5])
    cells = grid_cells(cfg, alphas=(0.05,))
    frame = stored_or_run(
        cfg, name, lambda only: run_grid(cfg, cells, "ablation", edge_slowdown, only=only)
    )
    best_family = np.minimum(totals(frame)["compression_only"], totals(frame)["learned_only"])
    h5 = 1.0 - totals(frame)["kai_s1_v0"] / best_family
    high = frame[frame.load >= 0.8]
    changed = (dominant_rep(high, "kai_s1_v0") != dominant_rep(high, "no_encode_cost")).mean()
    slo = high.groupby("policy").slo_violation.mean()
    summary: dict[str, Any] = {
        "edge_slowdown": edge_slowdown,
        "n_folds": int(frame.fold.nunique()),
        "H4/choice_changed_in_high_load_cells": float(changed),
        "H4/slo_with_encode_cost": float(slo["kai_s1_v0"]),
        "H4/slo_ignoring_encode_cost": float(slo["no_encode_cost"]),
        "H5/cost_reduction_vs_best_family": float(h5.mean()),
    }
    summary["H5/ci_lo"], summary["H5/ci_hi"] = stats.bca_ci(h5.to_numpy(), rng)
    summary["H4_pass"] = bool(changed >= 0.15 and slo["no_encode_cost"] > slo["kai_s1_v0"])
    summary["H5_pass"] = bool(h5.mean() >= 0.10)
    for policy in sorted(frame.policy.unique()):
        r = reduction(frame, "kai_s1_v0", policy, rng) if policy != "kai_s1_v0" else None
        summary[f"cost_total/{policy}"] = float(totals(frame)[policy].mean())
        summary[f"risk_certified/{policy}"] = float(
            frame[frame.policy == policy].risk_certified.mean()
        )
        summary[f"slo/{policy}"] = float(frame[frame.policy == policy].slo_violation.mean())
        if r is not None:
            summary[f"kai_cost_reduction_vs/{policy}"] = r["mean"]
    summary["per_fold"] = {"H5": h5.tolist()}
    return finish(name, cfg, summary, frame)


# --- real-socket experiments: E4 phase diagram, E7 sim-to-real, E8 overhead ----------------------


def _bundle_for(cfg: DictConfig) -> tuple[Bundle, Path, WindowSet, Table, np.ndarray]:
    """Build (or reuse) the bundle of the first fold and load it with the local profiles."""
    from kai.bench.bundle import BundleMeta, build_bundle
    from kai.edge.bundle import load_bundle

    fold = fold_dirs(cfg)[0]
    ws = WindowSet.load(work() / cfg.dataset.windows)
    meta = BundleMeta.from_fold(fold, ws.dataset_id, ws.sample_rate_hz)
    profiles, overheads = load_profiles()
    path = build_bundle(
        fold, work() / "artifacts" / "bundles", profiles, overheads, meta,
        prices_of(cfg), energy_of(cfg), cfg.profile.mission,
    )  # fmt: skip
    table = Table.load(fold / "table.npz")
    test = np.flatnonzero(table.of("test"))
    return load_bundle(path), path, ws, table, test


WAVEFORM_FAMILY = ("r0_raw", "r1_lossless", "r2a_wavelet", "r2b_wavelet")


def _measure_latencies(
    cfg: DictConfig,
    n_windows: int,
    max_latency_ms: float,
    grid: np.ndarray,
    loads: tuple[float, ...],
) -> pd.DataFrame:
    """Median end-to-end latency of every representation over real sockets, per rate and load."""
    from kai.bench.realrun import cloud_process, real_run
    from kai.edge.runtime import EdgeConfig
    from kai.profile.profiler import LOAD_TENANTS
    from kai.select.policies import Static

    bundle, path, ws, _, test = _bundle_for(cfg)
    prices, energy = prices_of(cfg), energy_of(cfg)
    model = CostModel(Profiles.from_list(bundle.profiles), bundle.overheads, energy, prices)
    pick = test[:: max(1, len(test) // n_windows)][:n_windows]
    rows: list[dict[str, Any]] = []
    with cloud_process(path) as port:
        for rho, bps in product(loads, grid):
            predicted = model.latency_ms(State(float(bps), 0.0, 0.0, rho, np.inf))
            for r, rep in enumerate(bundle.reps):
                if predicted[r] > max_latency_ms:
                    continue  # hopelessly slow here; it cannot be the fastest representation
                edge_cfg = EdgeConfig(
                    deadline_ms=1e9, period_ms=max(20.0, 1.5 * predicted[r]), load=rho,
                    shape_bps=float(bps), initial_bps=float(bps),
                )  # fmt: skip
                recs = real_run(
                    bundle, port, edge_cfg, prices, energy, ws.x[pick], ws.window_id[pick],
                    Static(r), LOAD_TENANTS[rho],
                )  # fmt: skip
                lat = np.array([x["latency_ms"] for x in recs if "latency_ms" in x])
                rows.append(
                    {
                        "load": rho,
                        "bandwidth_bps": float(bps),
                        "rep_id": rep.rep_id,
                        "latency_ms_measured": float(np.median(lat)),
                        "latency_ms_model": float(predicted[r]),
                    }
                )
    return pd.DataFrame(rows)


def _boundaries(
    frame: pd.DataFrame, model: CostModel, members: tuple[str, ...] | None
) -> tuple[list[dict[str, Any]], int]:
    """Measured argmin changes along bandwidth, matched with the closed-form envelope.

    `members` restricts both to a sub-menu. Returns the boundaries and the number predicted.
    """
    from kai.analysis import phase

    ids = model.profiles.rep_ids
    allowed = np.array([members is None or r in members for r in ids])
    out: list[dict[str, Any]] = []
    n_theory = 0
    for rho, sub in frame.groupby("load"):
        table = sub.pivot(index="bandwidth_bps", columns="rep_id", values="latency_ms_measured")
        measured = table.reindex(columns=list(ids)).to_numpy(copy=True)
        measured[:, ~allowed] = np.nan
        envelope = phase.latency_envelope(model, float(rho), allowed)
        theory = {(b.low, b.high): b.bps for b in envelope}
        n_theory += len(envelope)
        for b in phase.crossings(table.index.to_numpy(), measured):
            predicted_bps = theory.get((b.low, b.high))
            out.append(
                {
                    "load": float(rho),
                    "low": ids[b.low],
                    "high": ids[b.high],
                    "bps_measured": b.bps,
                    "bps_theory": predicted_bps,
                    "rel_error": abs(b.bps - predicted_bps) / predicted_bps
                    if predicted_bps
                    else float("nan"),
                }
            )
    return out, n_theory


def e4_phase(
    cfg: DictConfig,
    n_windows: int = 25,
    max_latency_ms: float = 600.0,
    grid_points: int = 13,
    loads: tuple[float, ...] = (0.0, 0.8),
    reuse: bool = False,
) -> dict[str, Any]:
    """Phase Diagram: closed-form break-evens versus boundaries measured over real sockets (H6).

    `reuse=True` re-analyses the stored measurements instead of measuring again.
    """
    stored = out_dir("e4") / "results.parquet"
    if reuse and stored.exists():
        frame = pd.read_parquet(stored)
    else:
        grid = np.logspace(4.5, 7.5, grid_points)
        frame = _measure_latencies(cfg, n_windows, max_latency_ms, grid, loads)
    profiles, overheads = load_profiles()
    model = CostModel(Profiles.from_list(profiles), overheads, energy_of(cfg), prices_of(cfg))
    boundaries, n_theory = _boundaries(frame, model, None)
    family, n_family = _boundaries(frame, model, WAVEFORM_FAMILY)
    matched = [b for b in boundaries if b["bps_theory"]]
    family_matched = [b for b in family if b["bps_theory"]]
    rel = (frame.latency_ms_measured - frame.latency_ms_model) / frame.latency_ms_model
    summary: dict[str, Any] = {
        "n_boundaries_measured": len(boundaries),
        "n_boundaries_matched": len(matched),
        "n_boundaries_theory": n_theory,
        "max_rel_error": float(max((b["rel_error"] for b in matched), default=float("nan"))),
        "median_latency_model_error": float(rel.abs().median()),
        "boundaries": boundaries,
        # exploratory, not part of H6: the same test inside the waveform-preserving family, where
        # the representations differ enough for a break-even to exist
        "family/n_theory": n_family,
        "family/n_measured": len(family),
        "family/n_matched": len(family_matched),
        "family/max_rel_error": float(
            max((b["rel_error"] for b in family_matched), default=float("nan"))
        ),
        "family_boundaries": family,
    }
    for rho, sub in frame.assign(rel=rel).groupby("load"):
        summary[f"latency_model_bias/load{rho:g}"] = float(sub.rel.mean())
    summary["H6_pass"] = bool(matched and summary["max_rel_error"] <= 0.20)
    return finish("e4", cfg, summary, frame)


def _sim_vs_real(recs: list[dict[str, Any]], sim: Result, bps: float) -> dict[str, float]:
    """One row of the sim-to-real comparison: the same windows, real runtime versus DES."""
    real_rep = np.array([r["rep"] for r in recs])
    real_lat = np.array([r.get("latency_ms", np.nan) for r in recs])
    both = ~np.isnan(real_lat) & ~np.isnan(sim.latency_ms)
    p50_real, p50_sim = float(np.median(real_lat[both])), float(np.median(sim.latency_ms[both]))
    return {
        "bandwidth_bps": bps,
        "p50_real_ms": p50_real,
        "p50_sim_ms": p50_sim,
        "p50_rel_gap": abs(p50_sim - p50_real) / p50_real,
        "p95_real_ms": float(np.percentile(real_lat[both], 95)),
        "p95_sim_ms": float(np.percentile(sim.latency_ms[both], 95)),
        "selection_agreement": float(np.mean(real_rep == sim.rep)),
        "delivered_real": float(both.mean()),
        "bandwidth_estimate_real_bps": float(np.median([r["bandwidth_bps"] for r in recs])),
    }


def _finish_h7(
    name: str, cfg: DictConfig, backend: str, rows: list[dict[str, float]]
) -> dict[str, Any]:
    """H7 verdict from the comparison rows: p50 within 10 %, same choice for ≥ 85 % of windows."""
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "backend": backend,
        "max_p50_rel_gap": float(frame.p50_rel_gap.max()),
        "min_selection_agreement": float(frame.selection_agreement.min()),
        "rows": rows,
    }
    summary["H7_pass"] = bool(
        summary["max_p50_rel_gap"] <= 0.10 and summary["min_selection_agreement"] >= 0.85
    )
    return finish(name, cfg, summary, frame)


def e7_sim2real(cfg: DictConfig, n_windows: int = 300) -> dict:
    """The same bundle, workload and scenario in the DES and over real sockets (H7)."""
    from kai.bench.realrun import cloud_process, real_run
    from kai.edge.runtime import EdgeConfig, build_certifier, build_selector
    from kai.link import traces as tr
    from kai.link.des import Scenario, Workload, simulate

    bundle, path, ws, table, test = _bundle_for(cfg)
    prices, energy = prices_of(cfg), energy_of(cfg)
    model = CostModel(Profiles.from_list(bundle.profiles), bundle.overheads, energy, prices)
    pick = ev.spread(test, n_windows)
    phi = table.phi[pick].astype(np.float64)
    work = Workload(bundle.predictor.predict(phi), phi, table.d[pick], table.payload_bytes[pick])
    period = window_period_ms(cfg)
    rows = []
    with cloud_process(path) as port:
        for bps in (256e3, 1e6, 5e6):
            edge_cfg = EdgeConfig(
                alpha=0.05,
                deadline_ms=250.0,
                period_ms=period,
                load=0.0,
                shape_bps=bps,
                initial_bps=bps,
                mission=cfg.profile.mission,
            )
            recs = real_run(bundle, port, edge_cfg, prices, energy, ws.x[pick], ws.window_id[pick])
            scenario = Scenario(tr.static(bps), 250.0, 0.0, 1, period, rtt_ms=0.0)
            sim = simulate(
                scenario,
                work,
                build_selector(bundle, edge_cfg, prices, energy),
                build_certifier(bundle, edge_cfg),
                model,
            )
            rows.append(_sim_vs_real(recs, sim, bps))
    backend = "loopback sockets, token-bucket pacing, same laptop as the DES (no Pi)"
    return _finish_h7("e7", cfg, backend, rows)


_RATE_UNITS = {"kbit": 1e3, "mbit": 1e6}


def e7d_docker(cfg: DictConfig, delay_ms: float = 10.0, alpha: float = 0.05) -> dict[str, Any]:
    """Simulator versus the Docker + `tc netem` closed loop (PRD §15.3 fallback for H7).

    Reads `experiments/docker/telemetry_<rate>.msgpack` written by `make docker-loop`, which must
    have used the bundle of the first fold. netem delays the edge's egress only, so the round trip
    the edge measures is one `delay_ms`.
    """
    import re

    import msgpack

    from kai.edge.runtime import EdgeConfig, build_certifier, build_selector
    from kai.link import traces as tr
    from kai.link.des import Scenario, Workload, simulate

    bundle, _, _, table, _ = _bundle_for(cfg)
    prices, energy = prices_of(cfg), energy_of(cfg)
    model = CostModel(Profiles.from_list(bundle.profiles), bundle.overheads, energy, prices)
    period = window_period_ms(cfg)
    files = sorted((work() / "experiments" / "docker").glob("telemetry_*.msgpack"))
    if not files:
        raise FileNotFoundError("no Docker telemetry: run `make docker-loop BUNDLE=… FOLD=…` first")
    rows = []
    for path in files:
        value, unit = re.fullmatch(r"telemetry_(\d+)(kbit|mbit)", path.stem).groups()  # type: ignore[union-attr]
        bps = float(value) * _RATE_UNITS[unit]
        recs = msgpack.unpackb(path.read_bytes())
        ids = np.array([r["window_id"] for r in recs])
        phi = table.phi[ids].astype(np.float64)  # window_id is the row index of the table
        simulated = Workload(
            bundle.predictor.predict(phi), phi, table.d[ids], table.payload_bytes[ids]
        )
        edge_cfg = EdgeConfig(
            alpha=alpha, deadline_ms=250.0, period_ms=period, load=0.0, mission=bundle.mission
        )
        sim = simulate(
            Scenario(tr.static(bps), 250.0, 0.0, 1, period, rtt_ms=delay_ms),
            simulated,
            build_selector(bundle, edge_cfg, prices, energy),
            build_certifier(bundle, edge_cfg),
            model,
        )
        rows.append(_sim_vs_real(recs, sim, bps))
    backend = "Docker containers, tc netem rate + delay on the edge egress, same laptop"
    return _finish_h7("e7d", cfg, backend, rows)


def e8_overhead(cfg: DictConfig, n: int = 3000) -> dict:
    """Selector overhead and scalability on this machine: decisions per second (single core)."""
    import time as _time

    from kai.select.policies import MpcSafe

    bundle, _, _, table, test = _bundle_for(cfg)
    model = CostModel(
        Profiles.from_list(bundle.profiles), bundle.overheads, energy_of(cfg), prices_of(cfg)
    )
    phi = table.phi[np.resize(test, n)].astype(np.float64)
    ghat = bundle.predictor.predict(phi)
    lam = threshold(bundle.lambdas, "v0", 0.10, tuple(bundle.manifest.rep_ids))
    cert = Certifier(lam)
    state = State(1e6, 20.0, 2000.0, 0.5, 250.0)
    selectors = {"s1_greedy_safe": GreedySafe(model)} | {
        f"s2_mpc_h{h}": MpcSafe(model, window_period_ms(cfg), h) for h in (3, 4, 5)
    }
    summary: dict[str, Any] = {}
    for name, sel in selectors.items():
        for i in range(200):  # warm-up
            sel.select(Window(cert.safe_set(ghat[i]), ghat[i], phi[i]), state)
        t0 = _time.perf_counter_ns()
        for i in range(n):
            sel.select(Window(cert.safe_set(ghat[i]), ghat[i], phi[i]), state)
        us = (_time.perf_counter_ns() - t0) / n / 1e3
        summary[f"{name}/us_per_decision"] = us
        summary[f"{name}/decisions_per_s"] = 1e6 / us
    ov = bundle.overheads
    cheapest_enc = min(p.t_enc_ms for p in bundle.profiles[1:])
    summary["overhead_ms_probe_pred_select"] = ov.total_ms
    summary["overhead_vs_cheapest_encoder"] = ov.total_ms / cheapest_enc
    summary["overhead_share_of_64ms_window"] = ov.total_ms / window_period_ms(cfg)
    return finish("e8", cfg, summary)


def e6_drift(cfg: DictConfig) -> dict:
    """Operating-condition shift (L2 folds): static λ̂ versus audit-in-the-loop ACI."""
    from kai.certify.drift import DriftMonitor

    profiles, overheads = load_profiles()
    period = window_period_ms(cfg)
    alpha = 0.05
    rows = []
    for level in ("L1", "L2"):
        for fold in fold_dirs(cfg, level)[:5]:
            ctx = ev.load_fold(
                fold,
                profiles,
                overheads,
                prices_of(cfg),
                energy_of(cfg),
                (alpha,),
                cfg.delta,
                period,
                seed=int(cfg.seed),
            )
            t = ctx.scored.table
            test = t.of("test")
            lam = threshold(ctx.lambdas[alpha], "v0", alpha, t.rep_ids)
            worst = (crc.safe_set(ctx.scored.ghat[test], lam) & t.d[test]).any(axis=1)
            monitor = DriftMonitor(t.phi[t.of("calib")].astype(np.float64))
            fired = [monitor.update(p) for p in t.phi[test].astype(np.float64)]
            cell = ev.Cell("static", 1e6, 250.0, 0.0, 1, alpha, p_audit=float(cfg.p_audit))
            runs = ev.run_cell(
                ctx,
                cell,
                cfg.profile.mission,
                "ablation",
                int(test.sum()),
                int(cfg.seed),
                ("kai_s1_v0", "audit_aci"),
                late_cap_factor=float(cfg.late_cap_factor),
            )
            static, aci = (
                next(r for r in runs if r["policy"] == p) for p in ("kai_s1_v0", "audit_aci")
            )
            rows.append(
                {
                    "level": level, "fold": fold.name,
                    "safe_set_risk_static": float(worst.mean()),
                    "policy_risk_static": static["risk_certified"],
                    "policy_risk_aci": aci["risk_certified"],
                    "bytes_static": static["bytes"], "bytes_aci": aci["bytes"],
                    "audit_bytes": aci["audit_bytes"], "drift_fired": float(np.mean(fired)),
                    "risk_audit_estimate": aci["risk_audit_estimate"],
                }
            )  # fmt: skip
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {"alpha": alpha, "p_audit": float(cfg.p_audit)}
    for level, sub in frame.groupby("level") if len(frame) else ():
        summary[f"{level}/n_folds"] = len(sub)
        for col in (
            "safe_set_risk_static",
            "policy_risk_static",
            "policy_risk_aci",
            "bytes_static",
            "bytes_aci",
            "audit_bytes",
            "risk_audit_estimate",
            "drift_fired",
        ):
            summary[f"{level}/{col}"] = float(sub[col].mean())
    return finish("e6", cfg, summary, frame)


def e5pi_ablation_slow_edge(cfg: DictConfig) -> dict[str, Any]:
    """E5 again with edge CPU times × 8: a *modelled* Pi-class edge, a labelled sensitivity run."""
    return e5_ablation(cfg, edge_slowdown=8.0, name="e5_pi_class")


def e10_privacy(cfg: DictConfig) -> dict[str, Any]:
    """Attribute-inference probe (PRD E10): what a payload reveals besides the diagnosis.

    An observer of the uplink fits a logistic-regression attacker on what the cloud decodes from
    each feature representation. Two attacks per fold:

    - *operating condition of unseen machines*: fitted on the `ptrain` bearings, scored on `test`;
    - *re-identification of a machine seen before*: fitted on the first half of every test
      bearing's windows in each condition, scored on the second half. Fit and score share bearings
      on purpose — that is the attack, not a diagnosis result.

    The raw waveform is represented by its band-energy shape, a lower bound on what raw reveals.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from kai.reps.latent import OnnxLatent, OnnxVQ
    from kai.reps.registry import load_menu
    from kai.tasks import train as tr
    from kai.tasks.models import SpecFront
    from kai.tasks.pipeline import classical_outputs, latent_features

    ws = WindowSet.load(work() / cfg.dataset.windows)
    cache = classical_outputs(ws, work() / "data/interim/cache")
    conditions = np.unique(ws.condition, return_inverse=True)[1]
    rows = []
    for fold_dir in fold_dirs(cfg):
        table = Table.load(fold_dir / "table.npz")
        ptrain, test = np.flatnonzero(table.of("ptrain")), np.flatnonzero(table.of("test"))
        bearing = np.unique(ws.bearing_id[test], return_inverse=True)[1]
        # first half of each (bearing, condition) run: windows are stored in recording order
        group = bearing * 16 + conditions[test]
        rank = np.zeros(len(test))
        for g in np.unique(group):
            rank[group == g] = np.linspace(0, 1, int((group == g).sum()), endpoint=False)
        early, late = rank < 0.5, rank >= 0.5

        def attack(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray) -> float:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
            return float((model.fit(xa, ya).predict(xb) == yb).mean())

        for rep in load_menu(fold_dir / "menu"):
            rid = rep.rep_id
            if rid == "r0_raw":
                front = SpecFront(ws.window_n // 4)
                f_ptrain, f_test = (
                    tr.predict(front, ws.x[i].astype(np.float32) / 32768.0) for i in (ptrain, test)
                )
            elif isinstance(rep, OnnxLatent | OnnxVQ):
                f_ptrain, f_test = (latent_features(rep, ws.x[i]) for i in (ptrain, test))
            elif rep.head_input == "features":
                out = np.asarray(cache[f"{rid}.out"])
                f_ptrain, f_test = out[ptrain], out[test]
            else:
                continue  # lossless and wavelets decode to a waveform: covered by the raw row
            rows.append(
                {
                    "fold": fold_dir.name,
                    "rep": rid,
                    "payload_bytes": float(
                        table.payload_bytes[test, table.rep_ids.index(rid)].mean()
                    ),
                    "condition_accuracy": attack(
                        f_ptrain, conditions[ptrain], f_test, conditions[test]
                    ),
                    "condition_chance": 1.0 / len(np.unique(conditions)),
                    "reid_accuracy": attack(
                        f_test[early], bearing[early], f_test[late], bearing[late]
                    ),
                    "reid_chance": 1.0 / len(np.unique(bearing)),
                }
            )
    frame = pd.DataFrame(rows)
    by_rep = frame.groupby("rep", sort=False).mean(numeric_only=True)
    summary: dict[str, Any] = {"n_folds": int(frame.fold.nunique())}
    for rid, row in by_rep.iterrows():
        for col in by_rep.columns:
            summary[f"{rid}/{col}"] = float(row[col])
    return finish("e10", cfg, summary, frame)


EXPERIMENTS = (
    e1_profile,
    e2_certificate,
    e3_main,
    e4_phase,
    e5_ablation,
    e5pi_ablation_slow_edge,
    e6_drift,
    e7_sim2real,
    e7d_docker,
    e8_overhead,
    e10_privacy,
)
