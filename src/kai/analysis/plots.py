"""Paper figures, regenerated from `experiments/*/` (PRD §9.6): `python -m kai.analysis.plots`.

Static figures for print. Colour does one job per figure (identity of a policy group, or of a
representation) from a fixed, validated categorical order; everything else is labelled directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kai.bench.config import work
from kai.contracts.schemas import EnergyModel, Prices, RepProfile
from kai.select.cost import CostModel, Overheads, Profiles, State

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8983"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
            "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
            "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False, "figure.dpi": 150, "lines.linewidth": 2.0,
        }
    )  # fmt: skip


def _exp(name: str) -> Path:
    return work() / "experiments" / name


def _summary(name: str) -> dict:
    return json.loads((_exp(name) / "summary.json").read_text())


def _save(fig: plt.Figure, name: str) -> Path:
    out = work() / "experiments" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pareto() -> Path:
    """F2: what each representation costs (bytes, encode time) and what it loses (agreement)."""
    t = pd.read_parquet(_exp("e1") / "results.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, x, label in (
        (axes[0], t.size_mean_bytes + 28, "bytes on the wire per window (log)"),
        (axes[1], np.maximum(t.t_enc_ms, 1e-3), "edge encode time, ms (log)"),
    ):
        ax.errorbar(
            x,
            t.disagreement_mean,
            yerr=t.disagreement_sd,
            fmt="o",
            ms=6,
            color=SERIES[0],
            ecolor=NEUTRAL,
            elinewidth=1,
        )
        for n, (xi, yi, rid) in enumerate(zip(x, t.disagreement_mean, t.rep_id, strict=True)):
            offset = (
                (5, 6) if n % 2 == 0 else (5, -13)
            )  # alternate sides: neighbours do not collide
            name = rid.split("_", 1)[0]
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=offset, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel(label)
    axes[0].set_ylabel("disagreement with raw decision (test bearings)")
    axes[0].set_title("Representation menu: cost versus decision agreement")
    return _save(fig, "f2_pareto")


def _local_model() -> CostModel:
    data = json.loads((work() / "artifacts" / "profiles" / "local.json").read_text())
    profiles = Profiles.from_list([RepProfile(**p) for p in data["profiles"]])
    prices = Prices(name="latency", story="", p_byte=0, p_j=0, p_cpu_ms=0, p_late_ms=1)
    return CostModel(
        profiles, Overheads(**data["overheads"]), EnergyModel(p_cpu_w=1.0, e_bit_j=0.0), prices
    )


def _phase_panel(ax: plt.Axes, model: CostModel, allowed: np.ndarray, tie: float = 0.05) -> None:
    """Fastest representation over bandwidth × load. Representations within `tie` of the best
    latency are merged into one region: a boundary between near-equal options would be noise."""
    ids = model.profiles.rep_ids
    bw, loads = np.logspace(4, 8, 200), np.linspace(0.0, 0.9, 46)
    region = np.empty((len(loads), len(bw)), dtype=object)
    for a, rho in enumerate(loads):
        for b, bps in enumerate(bw):
            state = State(float(bps), 0.0, 0.0, float(rho), np.inf)
            lat = np.where(allowed, model.latency_ms(state), np.inf)
            near = np.flatnonzero(lat <= lat.min() * (1 + tie))
            region[a, b] = " = ".join(ids[k].split("_", 1)[0] for k in near)
    names = sorted(set(region.ravel()), key=lambda n: float(np.median(np.nonzero(region == n)[1])))
    for i, name in enumerate(names):
        mask = (region == name).astype(float)
        colour = SERIES[i % len(SERIES)]
        ax.contourf(bw, loads, mask, levels=[0.5, 1.5], colors=[colour], alpha=0.85)
        rows, cols = np.nonzero(region == name)
        if len(rows) < 0.04 * region.size:
            continue  # a sliver: its label would sit on top of its neighbours'
        x, y = bw[int(np.median(cols))], loads[int(np.median(rows))]
        ax.text(x, y, name, color=INK, ha="center", va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("uplink bandwidth, bit/s (log)")
    ax.grid(False)


def fig_phase() -> Path:
    """F3: which representation is fastest over bandwidth × edge load, theory versus measurement."""
    model = _local_model()
    ids = model.profiles.rep_ids
    family = np.array([r in ("r0_raw", "r1_lossless", "r2a_wavelet", "r2b_wavelet") for r in ids])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    _phase_panel(axes[0], model, np.ones(len(ids), bool))
    _phase_panel(axes[1], model, family)
    marker = {"marker": "D", "ms": 8, "mfc": SURFACE, "mec": INK, "mew": 1.5, "ls": "none"}
    measured = _summary("e4").get("family_boundaries", [])
    for b in measured:
        axes[1].plot(b["bps_measured"], b["load"], clip_on=False, **marker)
    if measured:
        handle = axes[1].plot([], [], label="boundary measured over real sockets", **marker)
        fig.legend(handles=handle, loc="lower right", bbox_to_anchor=(0.99, -0.06))
    axes[0].set_ylabel("edge CPU load ρ")
    axes[0].set_title("Full menu", fontweight="normal")
    axes[1].set_title("Waveform-preserving family only", fontweight="normal")
    fig.suptitle(
        "Fastest representation, closed form (options within 5 % of the best are merged)",
        x=0.01, ha="left", fontweight="bold",
    )  # fmt: skip
    fig.tight_layout()
    return _save(fig, "f3_phase_diagram")


def fig_certificate() -> Path:
    """F4: held-out risk per fold for every calibration variant, against the target α."""
    t = pd.read_parquet(_exp("e2") / "results.parquet")
    alphas = sorted(t.alpha.unique())
    variants = ["v0", "v1", "v1_per_rep", "v2"]
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.2 * len(alphas), 3.4), squeeze=False)
    for ax, alpha in zip(axes[0], alphas, strict=True):
        for i, v in enumerate(variants):
            risks = t[(t.alpha == alpha) & (t.variant == v)].risk.to_numpy()
            ax.plot(np.full(len(risks), i), risks, "o", ms=5, color=SERIES[0], alpha=0.7)
            ax.plot([i - 0.25, i + 0.25], [risks.mean()] * 2, color=INK, lw=2)
        ax.axhline(alpha, color=SERIES[7], lw=1.2, ls="--")
        ax.set_title(f"target α = {alpha:g} (dashed)", fontweight="normal", fontsize=9)
        ax.set_xticks(range(len(variants)), ["V0\nCRC", "V1\nLTT", "V1\nper-rep", "V2\npolicy"])
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel("risk on test bearings (dot = fold, bar = mean)")
    fig.suptitle("Certificate validity on held-out bearings", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    return _save(fig, "f4_certificate")


def _group(policy: str) -> int:
    return 0 if policy.startswith("kai") else (2 if policy == "oracle" else 1)


def fig_costs() -> Path:
    """F6: total cost over the scenario grid, relative to the oracle."""
    t = pd.read_csv(_exp("e3") / "by_policy.csv", index_col=0)
    rel = (t.cost_total_per_fold / t.cost_total_per_fold["oracle"]).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(rel) + 1.2))
    names = ["Kai", "baseline", "oracle"]
    palette = [SERIES[0], SERIES[1], NEUTRAL]
    ax.barh(rel.index, rel.to_numpy(), height=0.62, color=[palette[_group(p)] for p in rel.index])
    for y, v in enumerate(rel.to_numpy()):
        ax.text(v, y, f" {v:.2f}×", va="center", color=INK)
    for g in range(3):
        ax.barh([], [], color=palette[g], label=names[g])
    ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("total cost over the grid, relative to the oracle (log)")
    ax.set_title("Total cost over the grid (risk not matched: baselines carry no certificate)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "f6_costs")


def fig_ablations() -> Path:
    """F7: cost of each ablation relative to full Kai."""
    s = _summary("e5")
    costs = {k.split("/", 1)[1]: v for k, v in s.items() if k.startswith("cost_total/")}
    rel = pd.Series(costs).div(costs["kai_s1_v0"]).drop("kai_s1_v0").sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(rel) + 1.2))
    ax.barh(rel.index, rel.to_numpy(), height=0.62, color=SERIES[0])
    ax.axvline(1.0, color=INK, lw=1)
    for y, v in enumerate(rel.to_numpy()):
        ax.text(v, y, f" {v:.2f}×", va="center", color=INK)
    ax.set_xlabel("total cost relative to full Kai (1.0 = no change)")
    ax.set_title("Ablations")
    ax.grid(axis="y", visible=False)
    return _save(fig, "f7_ablations")


def fig_sim2real() -> Path:
    """F8: median latency in the simulator against the real-socket run."""
    rows = pd.DataFrame(_summary("e7")["rows"])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    lim = [
        0.8 * rows[["p50_real_ms", "p50_sim_ms"]].min().min(),
        1.25 * rows[["p50_real_ms", "p50_sim_ms"]].max().max(),
    ]
    ax.plot(lim, lim, color=NEUTRAL, lw=1, ls="--")
    ax.plot(rows.p50_real_ms, rows.p50_sim_ms, "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2)
    for r in rows.itertuples():
        ax.annotate(
            f"{r.bandwidth_bps / 1e6:g} Mbit/s\n{r.selection_agreement:.0%} same choice",
            (r.p50_real_ms, r.p50_sim_ms),
            textcoords="offset points",
            xytext=(8, -4),
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p50 latency, real sockets (ms)")
    ax.set_ylabel("p50 latency, simulator (ms)")
    ax.set_title("Sim-to-real (same laptop)")
    return _save(fig, "f8_sim2real")


def fig_overhead() -> Path:
    """F9: selector cost per decision."""
    s = _summary("e8")
    us = pd.Series(
        {k.split("/")[0]: v for k, v in s.items() if k.endswith("/us_per_decision")}
    ).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(us) + 1.2))
    ax.barh(us.index, us.to_numpy(), height=0.6, color=SERIES[0])
    for y, v in enumerate(us.to_numpy()):
        ax.text(v, y, f" {v:.0f} µs  ({1e6 / v:,.0f}/s)", va="center", color=INK)
    ax.set_xlabel("time per decision, µs (single core)")
    ax.set_title("Selector overhead")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=us.max() * 1.6)
    return _save(fig, "f9_overhead")


FIGURES = (
    fig_pareto,
    fig_phase,
    fig_certificate,
    fig_costs,
    fig_ablations,
    fig_sim2real,
    fig_overhead,
)


def make_all() -> list[Path]:
    """Draw every figure whose experiment has been run; skip the rest."""
    style()
    done = []
    for fig in FIGURES:
        try:
            done.append(fig())
        except FileNotFoundError:
            continue  # that experiment has not been run in this workspace
    return done


if __name__ == "__main__":
    for path in make_all():
        print(path)
