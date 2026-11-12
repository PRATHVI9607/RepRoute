"""Collect `experiments/*/summary.json` into `docs/research/results.md` (no hand-copied numbers).

A transfer run reports into its own file:

    KAI_WORK=transfer/hust python scripts/report.py --out docs/research/results-hust.md --dataset HUST
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from kai.analysis import stats
from kai.bench.config import ROOT, load_config, work

OUT = ROOT / "docs" / "research" / "results.md"


def load(name: str) -> dict | None:
    path = work() / "experiments" / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def stale_summaries() -> list[str]:
    """Experiments whose summary predates E1's. E1 is rerun after every fold rebuild, so an older
    summary was computed on folds (and menus) that no longer exist. If E1 itself predates the
    newest fold, every summary is stale."""
    root = work() / "experiments"
    e1 = root / "e1" / "summary.json"
    if not e1.exists():
        return []
    t = e1.stat().st_mtime
    folds = [p.stat().st_mtime for p in (work() / "artifacts" / "folds").glob("*/*/metrics.json")]
    if folds and max(folds) > t:  # E1 itself predates the folds: everything is stale
        return sorted(p.parent.name for p in root.glob("*/summary.json"))
    return sorted(p.parent.name for p in root.glob("*/summary.json") if p.stat().st_mtime < t)


def budget_sweep(s: dict, by: pd.DataFrame) -> list[str]:
    """Edge-memory budget sweep from `X26/required_bytes/*` (ADR 0026): at each distinct required
    size, the deployable policies that fit and the cheapest of them by grid-total cost."""
    need = {
        k.rsplit("/", 1)[1]: float(v) for k, v in s.items() if k.startswith("X26/required_bytes/")
    }
    cost = by.set_index("policy").cost_total_per_fold
    need = {p: b for p, b in need.items() if p != "oracle" and p in cost.index}
    rss = {  # measured peak RSS replaces the file bytes of these two policies when present
        "b14_edge_inference": s.get("rss/f_raw"),
        "kai_s1_v0_classical": s.get("rss/kai_classical"),
    }
    used_rss = [p for p, v in rss.items() if v is not None and p in need]
    need |= {p: float(rss[p]) for p in used_rss}
    if not need:
        return []
    rows, had_b14 = [], False
    for budget in sorted(set(need.values())):
        fit = [p for p, b in need.items() if b <= budget]
        best = min(fit, key=lambda p: cost[p])
        committee = [p for p in fit if p in X26_POLICIES]
        others = [p for p in fit if p not in X26_POLICIES]
        notes = []
        if "b14_edge_inference" in fit and not had_b14:
            notes.append("b14 becomes available")
        had_b14 |= "b14_edge_inference" in fit
        if committee and (
            not others or min(cost[p] for p in committee) < min(cost[p] for p in others)
        ):
            notes.append("best committee policy cheaper than everything else that fits")
        rows.append((int(budget), len(fit), best, float(cost[best]), "; ".join(notes)))
    sizes = [
        (k.removeprefix("X26/size/"), int(v)) for k, v in s.items() if k.startswith("X26/size/")
    ]
    return [
        "",
        "Edge memory budget sweep (exploratory, ADR 0026): at each required model size, the policies "
        "that fit (oracle excluded) and the cheapest of them by grid-total cost. Model size is the file "
        "bytes of the stored model file(s), one convention for every model (f_raw, encoders, heads, "
        "predictors); codecs without a stored file count 0. Required size is the largest over the first "
        "fold's cells."
        + (
            f" For {', '.join(used_rss)} the size is the peak RSS above a numpy-only Python "
            f"(baseline {s['rss/baseline']:,.0f} bytes), not file bytes."
            if used_rss
            else ""
        ),
        "",
        table(
            pd.DataFrame(rows, columns=["budget_bytes", "n_fit", "cheapest", "cost_total", "note"])
        ),
        *(
            [
                "",
                "Model sizes, first fold (bytes):",
                "",
                table(pd.DataFrame(sizes, columns=["model", "bytes"])),
            ]
            if sizes
            else []
        ),
    ]


def h6_of_record(reg: pd.DataFrame | None, s: dict) -> bool:
    """H6's verdict of record is the FIRST E4 run in the registry (hypotheses.md), not the latest;
    the summary's own verdict is used only when the registry has no E4 row (ADR 0028 fix 11)."""
    if reg is not None:
        for rec in reg[reg.experiment == "e4"].itertuples():
            m = json.loads(rec.metrics)
            if "H6_pass" in m:
                return bool(m["H6_pass"])
            break
    return bool(s["H6_pass"])


H6_KEYS = ("n_boundaries_theory", "n_boundaries_measured", "n_boundaries_matched", "max_rel_error")


def h6_numbers(reg: pd.DataFrame | None, s: dict) -> tuple[dict, dict | None]:
    """(numbers of the run H6's verdict comes from, numbers of the latest run if they differ).
    The first E4 run in the registry is the run of record; `s` is the latest run (ADR 0028 fix 11)."""
    latest = {k: s[k] for k in H6_KEYS}
    if reg is not None:
        for rec in reg[reg.experiment == "e4"].itertuples():
            m = json.loads(rec.metrics)
            if "H6_pass" in m and all(k in m for k in H6_KEYS):
                first = {k: m[k] for k in H6_KEYS}
                return first, (latest if latest != first else None)
            break
    return latest, None


C_SHORT = {"b14": "b14 (edge inference)", "b4": "b4 (best single)", "b1": "b1 (raw)"}
C_OURS = ("kai_s1_v0", "kai_s1_v0_classical")
C_POLICIES = (
    "kai_s1_v0",
    "kai_s1_v0_classical",
    "b14_edge_inference",
    "b4_best_single",
    "b1_raw",
    "oracle",
)
COST_TERMS = ("cost", "cost_bytes", "cost_energy_cpu", "cost_energy_link", "cost_cpu", "cost_late")


def _fmt_k(value: float | None, label: str = "") -> str:
    if value is None:
        return label or "n/a"
    return f"{value:.3g}" if label in ("", "crossing") else f"{label}"


def _comparison_tables(s: dict, title: str) -> list[str]:
    """Feasible-cell tables of one e3_k<k> / e3_slow400 summary; every number from `s`."""
    rows, shares, terms, sens = [], [], [], []
    for ours in C_OURS:
        for short, label in C_SHORT.items():
            tag = f"{ours}/vs_{short}"
            if f"cmp/{tag}/mean" not in s:
                continue
            rows.append(
                {
                    "Kai variant": ours,
                    "versus": label,
                    "cost reduction (mean over folds)": s[f"cmp/{tag}/mean"],
                    "CI lo": s[f"cmp/{tag}/ci_lo"],
                    "CI hi": s[f"cmp/{tag}/ci_hi"],
                    "structural": "yes" if s[f"cmp/{tag}/structural"] else "no",
                    "feasible cells": s[f"cmp/{tag}/n_cells"],
                    "per fold": ", ".join(f"{v:+.3f}" for v in s[f"cmp/{tag}/per_fold"]),
                    "folds left out": ", ".join(s.get(f"cmp/{tag}/missing_folds", [])) or "-",
                }
            )
            sens.append(
                {"Kai variant": ours, "versus": label}
                | {f"cap {c}": s[f"cap{c}/{tag}/mean"] for c in (5, 20)}
                | {f"energy x{e}": s[f"energy_e{e}/{tag}/mean"] for e in (0.3, 1, 3)}
            )
    for policy in C_POLICIES:
        if f"infeasible_share/{policy}" in s:
            shares.append({"policy": policy, "infeasible cells": s[f"infeasible_share/{policy}"]})
            terms.append(
                {"policy": policy}
                | {
                    ("total" if t == "cost" else t.removeprefix("cost_")): s[
                        f"cost_terms/{policy}/{t}"
                    ]
                    for t in COST_TERMS
                }
            )
    cap = f"{s.get('late_cap_factor', 10):g}"
    out = [f"#### {title}", ""]
    if rows:
        out += [
            "Cost reduction 1 − C_Kai / C_ref per fold over the cells whose edge utilisation is "
            f"below 1 for both policies (lateness cap {cap}·D). `structural`: the CI is narrower "
            "than 1e-3.",
            "",
            table(pd.DataFrame(rows), "{:+.3f}"),
            "",
        ]
    if shares:
        out += [
            "Share of cells infeasible per policy (edge utilisation ≥ 1; no cost):",
            "",
            table(pd.DataFrame(shares)),
            "",
            "Mean cost terms over each policy's feasible cells (total, bytes, CPU energy, link "
            "energy, CPU time, lateness):",
            "",
            table(pd.DataFrame(terms)),
            "",
        ]
    if sens:
        out += [
            f"Sensitivity of the same reductions to the lateness cap (configured {cap}·D; columns "
            "cap 5 and cap 20) and to the CPU energy price e:",
            "",
            table(pd.DataFrame(sens), "{:+.3f}"),
            "",
        ]
    return out


def slow_edge_section() -> list[str]:
    """Exploratory (ADR 0028 §2, §3 C, D): crossover k* and the measured slow edge. Empty unless
    the summaries exist. Descriptive; none of it enters a verdict."""
    ks = sorted(
        (
            (float(s["edge_slowdown"]), s)
            for path in sorted((work() / "experiments").glob("e3_k*/summary.json"))
            if path.parent.name != "e3_kstar"
            for s in [json.loads(path.read_text())]
        ),
        key=lambda t: t[0],
    )
    star, measured = load("e3_kstar"), load("e3_slow400")
    if not (ks or star or measured):
        return []
    out = [
        "## Slow-edge crossover and measured slow edge (exploratory)",
        "",
        "Decided after the ADR 0025–0027 results were seen (ADR 0028 §2–3): descriptive, no test, "
        "never part of H1–H7. First 5 folds; policies are compared only over cells feasible for both.",
        "",
    ]
    if star:
        rows = []
        for ours in C_OURS:
            if ours not in star:
                continue
            k = star[ours]
            lo, hi = k["k_star_ci_lo"], k["k_star_ci_hi"]
            rng_ = k["k_star_fold_range"]
            rows.append(
                {
                    "Kai variant (vs b14)": ours,
                    "k* (mean curve)": _fmt_k(k["k_star_mean_curve"], k["k_star_mean_curve_label"]),
                    "folds used (Δ defined at every k)": (
                        f"{k['n_folds_used']}/{len(k['folds'])}" if "n_folds_used" in k else "n/a"
                    ),
                    "folds excluded": ", ".join(k.get("folds_excluded", [])) or "none",
                    "folds defined per grid k": ", ".join(
                        map(str, k.get("n_folds_defined_per_k", []))
                    ),
                    "fold-bootstrap CI": "n/a" if lo is None else f"[{lo:.3g}, {hi:.3g}]",
                    "bootstrap draws beyond grid": k["k_star_boot_share_beyond_grid"],
                    "k* per fold": ", ".join(
                        _fmt_k(v, lab)
                        for v, lab in zip(
                            k["k_star_per_fold"], k["k_star_per_fold_label"], strict=True
                        )
                    ),
                    "per-fold range": "n/a" if rng_ is None else f"{rng_[0]:.3g} – {rng_[1]:.3g}",
                    "sign changes (mean curve)": ", ".join(
                        f"{v:.3g}" for v in k["sign_changes_mean_curve"]
                    )
                    or "none",
                }
            )
        grid = star["k_grid"]
        out += [
            "### Crossover k*",
            "",
            "Δ_f(k) = 1 − C_Kai,f / C_b14,f over cells feasible for both. k* is the smallest grid "
            "k with Δ > 0 for every larger grid k, interpolated linearly in log k; `> k` if none, "
            "`<= k` if Δ is positive from the start. Grid points where no cell is feasible for "
            "both are skipped. The mean curve (and its bootstrap) uses only the folds whose Δ is defined "
            "at every grid k; the others are listed as excluded.",
            "",
            table(pd.DataFrame(rows)),
            "",
            *(
                [
                    "Peak RSS above a numpy-only Python (ADR 0028 §3 A), first fold, fresh "
                    "subprocess each:",
                    "",
                    table(
                        pd.DataFrame(
                            [
                                {"what": w, "bytes": float(star[f"rss/{key}"])}
                                for w, key in (
                                    ("f_raw (onnxruntime)", "f_raw"),
                                    ("kai classical (probe, predictor, r1–r4)", "kai_classical"),
                                    ("numpy-only baseline (subtracted)", "baseline"),
                                )
                            ]
                        ),
                        "{:,.0f}",
                    ),
                    "",
                ]
                if "rss/f_raw" in star
                else []
            ),
        ]
        for ours in C_OURS:
            if ours not in star:
                continue
            k = star[ours]
            curve = pd.DataFrame(
                {"k": grid, "Δ mean": k["delta_mean"]}
                | {f: [row[i] for row in k["delta_per_fold"]] for i, f in enumerate(k["folds"])}
            )
            out += [f"Δ(k) of `{ours}` versus b14, every fold:", "", table(curve, "{:+.3f}"), ""]
    for k, s in ks:
        out += _comparison_tables(s, f"k = {k:g} (CPU power ÷ {k:g})")
    if measured:
        out += [
            "### Measured slow edge (`slow400` profile)",
            "",
            f"Edge times measured at 400 MHz on one pinned core, cloud times from `local.json`. "
            f"Measured slowdown s = median over operations of slow / local time: **{measured['slowdown_s']:.3g}** "
            f"(min / q25 / median / q75 / max = "
            + " / ".join(f"{v:.3g}" for v in measured["slowdown_quantiles_min_q25_q50_q75_max"])
            + f" over {len(measured['slowdown_per_op'])} operations); CPU power = 1 W ÷ s = "
            f"{measured['p_cpu_w']:.3g} W.",
            "",
            *_comparison_tables(measured, "Measured edge profile"),
        ]
    return out


# S-H3′ slow edge as in the HMC draft (docs/research/hmc-prereg-draft.md): stored run `e3_k10`, cells
# feasible for each pair. shortcut: SHHS pre-registered S-H3′ over the FULL grid of `e3_slow10`
# instead; add that source before an SHHS run.
CONF_K = 10.0
CONFIRMATORY = ("hmc", "shhs")  # dataset ids with a confirmatory pre-registration
CONF_FOLDS = 5  # every fold enters every confirmatory test
CONF_ALPHAS = (0.02, 0.05, 0.10)


def _ni_p(v0: pd.DataFrame) -> float:
    """ADR 0022 non-inferiority: max over α of the one-sided p of mean V0 risk < α + NI_MARGIN."""
    return max(
        stats.one_sided_p(_folds(sub.risk.to_numpy()), a + NI_MARGIN, greater=False)
        for a, sub in v0.groupby("alpha")
    )


def confirmatory_section(dataset: str, ratio: float = float("nan")) -> list[str]:
    """Confirmatory S-H3′ (k = CONF_K, feasible cells per pair), S-H2a and S-H5 with their Holm
    family, as defined in the HMC draft (not the SHHS S-H3′, see CONF_K). Empty unless a source summary exists; a hypothesis
    with a missing summary or fold is "incomplete" and has no verdict. N–B is a sensitivity only."""
    kname = f"e3_k{CONF_K:g}"
    e3k, e2, e5 = load(kname), load("e2"), load("e5")
    e2_path = work() / "experiments" / "e2" / "results.parquet"
    if not (e3k or e2 or e5):
        return []

    def test(parts: list[tuple[np.ndarray, float, bool]]) -> tuple[bool, float, float]:
        """(threshold met by the fold mean, max p, max N–B p) over the components."""
        met = all(v.mean() >= t if g else v.mean() <= t for v, t, g in parts)
        p = max(stats.one_sided_p(v, t, greater=g) for v, t, g in parts)
        nbp = max(stats.nb_corrected_one_sided_p(v, t, ratio, greater=g) for v, t, g in parts)
        return met, p, nbp

    res: dict[str, tuple[bool, float, float] | None] = dict.fromkeys(("S-H3′", "S-H2a", "S-H5"))
    s3 = e3k or {}
    # B4 and b14 come from c_summary's own feasible-cell comparisons
    refs = (("b4", 0.20), ("b14", 0.10))
    vals = [s3.get(f"cmp/kai_s1_v0/vs_{r}/per_fold") for r, _ in refs]
    if all(v is not None and len(v) == CONF_FOLDS for v in vals) and not any(
        s3.get(f"cmp/kai_s1_v0/vs_{r}/missing_folds") for r, _ in refs
    ):
        res["S-H3′"] = test(
            [(np.asarray(v, float), thr, True) for v, (_, thr) in zip(vals, refs, strict=True)]
        )
    if e2 is not None and e2_path.exists():
        v0 = pd.read_parquet(e2_path)
        v0 = v0[v0.variant == "v0"]
        groups = {float(a): g.risk.to_numpy() for a, g in v0.groupby("alpha")}
        if set(groups) == set(CONF_ALPHAS) and all(len(v) == CONF_FOLDS for v in groups.values()):
            met, _, nbp = test([(v, a + NI_MARGIN, False) for a, v in groups.items()])
            res["S-H2a"] = (met, _ni_p(v0), nbp)  # same p as the ADR 0022 line
    h5 = ((e5 or {}).get("per_fold") or {}).get("H5")
    if h5 is not None and len(h5) == CONF_FOLDS:
        res["S-H5"] = test([(np.asarray(h5, float), 0.10, True)])
    done = {h: r[1] for h, r in res.items() if r is not None}
    adj = stats.holm(done) if len(done) == len(res) else {}
    rows = []
    for h, r in res.items():
        if r is None or not adj:
            rows.append(f"| {h} | incomplete | | | | no verdict |")
            continue
        met, p, nbp = r
        ok = met and adj[h][0] < 0.05
        rows.append(
            f"| {h} | {'met' if met else 'not met'} | {p:.3g} | {adj[h][0]:.3g} | {nbp:.3g} | "
            f"{'**supported**' if ok else 'not supported'} |"
        )
    return [
        f"## Confirmatory family — {dataset}",
        "",
        f"S-H3′ as in the HMC pre-registration (k = {CONF_K:g}, source `{kname}`, `kai_s1_v0` vs B4 ≥ 0.20 and vs "
        "b14 ≥ 0.10, each on the cells feasible for that pair), S-H2a (V0 safe-set risk < α + "
        f"{NI_MARGIN:g}, α ∈ {{0.02, 0.05, 0.10}}) and S-H5 (E5, ≥ 0.10); one-sided t over "
        f"{CONF_FOLDS} folds, p = max over components, Holm at 0.05 within this dataset. "
        "Supported = threshold met and Holm-adjusted p < 0.05. The N–B p is a sensitivity column, "
        "not used for the verdict. A missing fold or summary gives no verdict for the family.",
        "",
        "| hypothesis | threshold | p | Holm-adjusted p | N–B p (sensitivity) | verdict |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def table(frame: pd.DataFrame, fmt: str = "{:.3g}") -> str:
    def cell(v: object) -> str:
        return fmt.format(v) if isinstance(v, float | np.floating) else str(v)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


NI_MARGIN = 0.01  # ADR 0022: H2a non-inferiority margin, chosen before computing
X25_POLICIES = (
    "kai_s1_v0_late",
    "kai_s1_v0_lldefer",
    "kai_s1_v0_latent",
    "kai_s1_v0_abc",
    "kai_s1_v0_zsel",
    "kai_s1_v0_ab_zsel",
)  # ADR 0025
X26_POLICIES = (
    "kai_s1_v0_committee",
    "kai_s1_v0_late_committee",
    "kai_s1_v0_late_committee_dec",
)  # ADR 0026
FAMILY = ("H1", "H2a", "H2b", "H3", "H4", "H5", "H6", "H7")  # the Holm family (hypotheses.md)
_tested: dict[
    str, bool | None
] = {}  # hypotheses with a per-fold test: verdict filled in after Holm


def verdict(passed: bool | None, tested: str | None = None) -> str:
    """Verdict text. A hypothesis with a per-fold test gets a placeholder resolved after Holm."""
    if tested is not None:
        _tested[tested] = passed
        return f"@@{tested}@@"
    if passed is None:
        return "not run"
    return "threshold met (descriptive, no test over folds)" if passed else "**not supported**"


def resolve_verdicts(text: str, adj: dict[str, tuple[float, bool]]) -> str:
    for key, passed in _tested.items():
        if passed is None:
            word = "not run"
        elif not passed:
            word = "**not supported**"
        elif adj[key][1]:
            word = "**supported**"
        else:
            word = f"threshold met, **not significant** (Holm-adjusted p = {adj[key][0]:.2f})"
        text = text.replace(f"@@{key}@@", word)
    return text


def seed_variance() -> pd.DataFrame | None:
    """Disagreement per representation across training seeds of repeat 0, if several were trained."""
    rows = []
    for path in (work() / "artifacts" / "folds").glob("*/s*_r0f*/metrics.json"):
        m = json.loads(path.read_text())
        seed = int(path.parent.name.split("_")[0][1:])
        rows += [
            {"split": path.parent.parent.name, "seed": seed, "rep": k.rsplit("/", 1)[1], "d": v}
            for k, v in m.items()
            if k.startswith("disagreement/test/")
        ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    l1 = {p.stem.rsplit("_", 1)[1] for p in (work() / "data" / "splits").glob("*_L1_*.json")}
    frame = frame[frame.split.isin(l1)]
    if frame.seed.nunique() < 2:
        return None
    per_seed = frame.groupby(["rep", "seed"], sort=False).d.mean().unstack()
    sd_folds = frame[frame.seed == 0].groupby("rep", sort=False).d.std()
    return pd.DataFrame(
        {
            "representation": per_seed.index,
            "seeds": per_seed.shape[1],
            "mean disagreement": per_seed.mean(axis=1),
            "sd over seeds": per_seed.std(axis=1),
            "sd over folds (seed 0)": sd_folds.reindex(per_seed.index),
        }
    ).reset_index(drop=True)  # aligned on the representation index, not by position


def _folds(values: list[float] | np.ndarray, _ids: object = None) -> np.ndarray:
    """Per-fold values: the pre-registered test units."""
    return np.asarray(values, dtype=np.float64)


def window_period_ms() -> float:
    """True window period: E8's `window_period_ms`, else 1000·window_n/sample_rate_hz of the windows."""
    e8 = load("e8")
    if e8 and "window_period_ms" in e8:
        return float(e8["window_period_ms"])
    for path in sorted((work() / "data" / "processed").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z, z.zip.open("x.npy") as f:
            version = np.lib.format.read_magic(f)
            read = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            shape = read(f)[0]  # (n, window_n)
            rate = float(z["sample_rate_hz"])
        return 1000.0 * shape[1] / rate  # shortcut: first processed set, one dataset per KAI_WORK
    return float("nan")


def work_dataset_id() -> str:
    """The dataset of KAI_WORK: the one id among its L1 split manifests, else the configured one."""
    ids = {p.name.split("_L1_")[0] for p in (work() / "data" / "splits").glob("*_L1_*.json")}
    return ids.pop() if len(ids) == 1 else str(load_config().dataset.id)


def nb_ratio(dataset_id: str) -> float:
    """Test units / non-test units of fold 0 of this dataset's L1 split manifest (ADR 0024)."""
    splits = work() / "data" / "splits"
    dataset_id = dataset_id or work_dataset_id()
    for path in sorted(splits.glob(f"{dataset_id}_L1_*.json")):
        fold = json.loads(path.read_text())["folds"][0]
        non_test = sum(len(fold[k]) for k in ("train", "ptrain", "calib"))
        return len(fold["test"]) / non_test
    return float("nan")


def main(out: Path = OUT, dataset: str = "Paderborn", dataset_id: str = "") -> None:
    """`dataset_id` (e.g. har-v1) picks the L1 split for N-B; default: the one in KAI_WORK."""
    e = {
        n: load(n)
        for n in ("e1", "e2", "e3", "e4", "e5", "e5_pi_class", "e6", "e7", "e7d", "e8", "e10")
    }
    title = "# Results" if dataset == "Paderborn" else f"# Results — {dataset} (transfer run)"
    lines = [title, ""]
    registry = work() / "experiments" / "registry.parquet"
    reg: pd.DataFrame | None = None
    n_classes = (e["e1"] or {}).get("n_classes")
    task = f"{dataset} {n_classes}-class task" if n_classes else f"{dataset} task"
    if registry.exists():
        reg = pd.read_parquet(registry)
        last = reg.iloc[-1]
        lines += [
            f"Generated by `scripts/report.py` from the run registry ({len(reg)} runs). "
            f"Last run: git `{last.git_sha[:10]}`{' (uncommitted tree)' if last.git_dirty else ''}, "
            f"lockfile `{last.lock_hash[:10]}`, {last.created_utc}.",
            "",
            f"Everything below is the **{task} with leave-bearings-out (L1) folds**, the "
            "`metered_cellular` preset unless stated, timings from the development laptop, energy "
            "modelled. Read `docs/research/hypotheses.md` for the pre-registered definitions and "
            "`docs/theory/limitations.md` before quoting any number.",
            "",
        ]
    if old := stale_summaries():
        lines += [
            "> **Stale:** "
            + ", ".join(f"`{n}`" for n in old)
            + " predate E1, so they ran on earlier folds. Rerun them before quoting their numbers.",
            "",
        ]
    pvalues: dict[str, float] = {}
    nb: dict[str, float] = {}  # ADR 0024, exploratory N–B p-values (H1, H2a, H3, H5)
    ratio = nb_ratio(dataset_id)

    if e["e1"]:
        s = e["e1"]
        prof = pd.read_csv(work() / "experiments" / "e1" / "profile_table.csv")
        gate = s["f_raw_macro_f1"] >= 0.40
        lines += [
            "## Gate G1 and the representation menu (E1)",
            "",
            f"- Folds: **{s['n_folds']}**. Representations profiled: **{s['n_reps_profiled']}**.",
            f"- Reference model `f_raw`, bearing-wise test macro-F1: **{s['f_raw_macro_f1']:.3f}** "
            f"(majority baseline {s['majority_macro_f1']:.3f}; threshold fixed in ADR 0004: 0.40; "
            f"`f_raw = {s.get('f_raw', '?')}`, R6 encoder `{s.get('bottleneck', '?')}`) → "
            f"G1 label-fidelity criterion {'**met**' if gate else '**not met**'}.",
            f"- Per-window overhead: probe {s['t_probe_ms']:.3f} ms + predictor {s['t_pred_ms']:.3f} ms + "
            f"selector {s['t_sel_ms']:.3f} ms.",
            f"- Mean fitted load exponent γ: {s['mean_load_exponent']:.2f} (1.0 = processor sharing).",
            *(
                [
                    "- The same work at the duty cycle of a busier node (one window every period / sensors; "
                    "ADR 0008) — probe + predictor + selector, and the R6 encoder: "
                    + "; ".join(
                        f"{k.rsplit('sensors', 1)[1]} sensors: {v:.2f} ms and "
                        f"{next(v2 for k2, v2 in s.items() if k2.startswith('t_enc_ms/') and k2.endswith('/' + k.rsplit('/', 1)[1])):.2f} ms"
                        for k, v in s.items()
                        if k.startswith("overhead_ms/sensors")
                    )
                    + ".",
                    "- Runtime overhead per sent frame, calibrated on the real runtime: measured excess over the "
                    f"model {s.get('io_excess_ms/load0', float('nan')):+.2f} ms at load 0 and "
                    f"{s.get('io_excess_ms/load0.8', float('nan')):+.2f} ms at load 0.8, used as "
                    f"{s['t_io_ms']:.2f} ms × (1 − ρ)^−{s['io_load_exponent']:.2f}"
                    + (" (the load-0 value is the 0.05 ms floor)" if s["t_io_ms"] <= 0.05 else "")
                    + ".",
                ]
                if "t_io_ms" in s
                else []
            ),
            f"- Edge inference for comparison: running the whole reference model takes {prof.t_head_ms[0]:.2f} ms "
            "(the cloud-side head time of raw, measured on this machine) and would send 1 byte; the R6 encoder "
            f"takes {prof.set_index('rep_id').t_enc_ms.get('r6_bottleneck', float('nan')):.2f} ms and sends "
            f"{prof.set_index('rep_id').size_mean_bytes.get('r6_bottleneck', float('nan')):.0f} bytes.",
            *(
                [
                    "- The log-mel grids (`r10a_logmel`, `r10b_logmel`) were chosen on the training speakers "
                    "of fold r0f0 (ADR 0018): an exploratory choice, and a possible selection leak into "
                    "the other folds' results for these two representations."
                ]
                if "r10a_logmel" in set(prof.rep_id)
                else []
            ),
            "",
            table(
                prof[
                    ["rep_id", "size_mean_bytes", "size_q95_bytes", "t_enc_ms", "t_dec_ms", "t_head_ms",
                     "load_exponent", "macro_f1_mean", "disagreement_mean", "disagreement_sd"]
                ].rename(columns={"macro_f1_mean": "label macro-F1", "disagreement_mean": "disagreement with raw", "disagreement_sd": "sd over folds"})
            ),
            "",
        ]  # fmt: skip

    seed_rows = seed_variance()
    if seed_rows is not None:
        lines += [
            "### Training-seed variance (repeat 0)",
            "",
            "The same five folds trained with different seeds. The reference model is a deterministic fit, so "
            "its column does not move; the seed changes the learned encoders and the head search. Mean over "
            "folds per seed, then mean and standard deviation over seeds, of the disagreement with the raw "
            "decision on test bearings:",
            "",
            table(seed_rows),
            "",
        ]

    if e["e2"]:
        s = e["e2"]
        res = pd.read_parquet(work() / "experiments" / "e2" / "results.parquet")
        rows = []
        for (alpha, variant), sub in res.groupby(["alpha", "variant"]):
            tag = f"{variant}/a{alpha:g}"
            rows.append(
                {
                    "α": alpha, "variant": variant, "risk (mean over folds)": s[f"{tag}/risk_mean"],
                    "risk (mean over test units)": s.get(f"{tag}/risk_unit_mean", float("nan")),
                    "95 % block-bootstrap CI": f"[{s[f'{tag}/risk_ci_lo']:.3f}, {s[f'{tag}/risk_ci_hi']:.3f}]",
                    "folds within α (descriptive)": f"{round(s[f'{tag}/folds_within_alpha'] * len(sub))}/{len(sub)}",
                    "windows with a non-raw option": s[f"{tag}/non_raw_available"],
                    "cheapest safe bytes / raw": s[f"{tag}/bytes_vs_raw"],
                }
            )  # fmt: skip
        v0 = res[res.variant == "v0"]
        pvalues["H2a"] = max(
            stats.one_sided_p(_folds(sub.risk.to_numpy(), sub.fold.tolist()), a, greater=False)
            for a, sub in v0.groupby("alpha")
        )
        nb["H2a"] = max(
            stats.nb_corrected_one_sided_p(sub.risk.to_numpy(), a, ratio, greater=False)
            for a, sub in v0.groupby("alpha")
        )
        # ADR 0022, exploratory and outside the Holm family: H0 mean risk ≥ α + margin
        ni_p = _ni_p(v0)
        ni_means = ", ".join(f"α = {a:g}: {sub.risk.mean():.4f}" for a, sub in v0.groupby("alpha"))
        lines += [
            "## Certificate validity (E2)",
            "",
            f"- **H2a** (V0, expected risk ≤ α on held-out bearings): {verdict(s['H2a_pass'], 'H2a')}.",
            f"- **H2b** (V1, ≥ 90 % of folds within α): {verdict(s['H2b_pass'])}.",
            "- Risk is the safe-set loss (worst admissible policy) for V0 / V1, and the greedy policy's "
            "loss for V2, which only promises that policy. *Mean over folds* pools each fold's test windows; "
            "*mean over test units* first averages each bearing's windows, so every unit weighs the same. "
            "The CI resamples whole bearings (a bearing tested in several folds is one block).",
            "- The H2a test (one-sided, mean < α) cannot reject by design for a method whose expected risk "
            "lands at α, as CRC's does, so H2a cannot be *supported* under the pre-registered rule (threshold met "
            "**and** Holm rejection) even when the bound holds.",
            f"- **H2a non-inferiority** (exploratory, ADR 0022; margin {NI_MARGIN:g} fixed before it was "
            f"computed; not in the Holm family): one-sided t-test over folds of mean V0 risk < α + {NI_MARGIN:g}, "
            f"maximum p over α = **{ni_p:.3g}** ({'rejects' if ni_p < 0.05 else 'does not reject'} at 0.05). "
            f"Mean risk {ni_means}."
            + (
                " Fold p-values are optimistic: folds of different repeats share test units."
                if v0.fold.str.extract(r"_r(\d+)f")[0].nunique() > 1
                else ""
            ),
            "- *Folds within α* is descriptive: CRC bounds the expected risk, so even under exact "
            "exchangeability about half of the folds (≈ 47 %) can exceed α.",
            *(
                [
                    "- Exploratory (ADR 0025 C), descriptive: mean held-out AUC of ĝ_r against d_r on calib over "
                    f"non-raw representations, φ only **{s['X25/auc/phi/mean']:.3f}**, φ and the r6 latent "
                    f"**{s['X25/auc/phi_z/mean']:.3f}**"
                    + (
                        f", chosen per representation (C′) **{s['X25/auc/phi_sel/mean']:.3f}**"
                        if "X25/auc/phi_sel/mean" in s
                        else ""
                    )
                    + "."
                ]
                if "X25/auc/phi/mean" in s and "X25/auc/phi_z/mean" in s
                else []
            ),
            *(
                [
                    "- Exploratory (ADR 0025 C′): folds that chose ĝ_z per representation: "
                    + ", ".join(
                        f"{k.rsplit('/', 1)[1]} {round(v * s['n_folds'])}/{s['n_folds']}"
                        for k, v in s.items()
                        if k.startswith("X25/zsel/use_z/")
                    )
                    + "."
                ]
                if any(k.startswith("X25/zsel/use_z/") for k in s)
                else []
            ),
            *(
                [
                    "- Exploratory (ADR 0026 D), descriptive: mean held-out AUC of the head-committee score ĝ_D "
                    f"against d_r on calib over non-raw representations **{s['X26/auc/committee/mean']:.3f}**"
                    + (f" (ĝ: {s['X25/auc/phi/mean']:.3f})" if "X25/auc/phi/mean" in s else "")
                    + "; V0 with ĝ_D (`v0_committee`) is in the per-variant tables above when present."
                ]
                if "X26/auc/committee/mean" in s
                else []
            ),
            "- Exchangeability control (V0 calibrated on a random half of the calibration windows and tested on the "
            "other half, mean of 20 halvings — exchangeable by construction, so the theorem's premise holds; "
            "a diagnostic, not a generalisation result): "
            + ", ".join(
                f"α = {a:g} → {s[f'control/a{a:g}/risk_same_bearings']:.3f}"
                for a in sorted(res.alpha.unique())
            )
            + ".",
            "",
            table(pd.DataFrame(rows)),
            "",
        ]

    if e["e3"]:
        s = e["e3"]
        by = pd.read_csv(work() / "experiments" / "e3" / "by_policy.csv").sort_values(
            "cost_total_per_fold"
        )
        pf = s["per_fold"]
        # H1 has two parts; the count of dominant representations has no test, so it gates the p
        # (b) is the pre-registered total_B4 / total_oracle − 1 per fold (H1_cost is 1 − o/b)
        ids = s.get("per_fold_ids")
        pvalues["H1"] = (
            stats.one_sided_p(_folds(1 / (1 - np.array(pf["H1_cost"])) - 1, ids), 0.25)
            if s["H1/reps_dominant_in_10pct_of_cells"] >= 3
            else 1.0
        )
        pvalues["H3"] = max(
            stats.one_sided_p(_folds(pf["H3_cost"], ids), 0.20),
            stats.one_sided_p(_folds(pf["H3_slo"], ids), 0.30),
        )
        h1 = _folds(1 / (1 - np.array(pf["H1_cost"])) - 1, ids)
        nb["H1"] = (
            stats.nb_corrected_one_sided_p(h1, 0.25, ratio)
            if s["H1/reps_dominant_in_10pct_of_cells"] >= 3
            else 1.0
        )
        nb["H3"] = max(
            stats.nb_corrected_one_sided_p(_folds(pf["H3_cost"], ids), 0.20, ratio),
            stats.nb_corrected_one_sided_p(_folds(pf["H3_slo"], ids), 0.30, ratio),
        )
        lines += [
            "## Main comparison over the scenario grid (E3)",
            "",
            f"{s['n_folds']} folds × {s['n_cells']} cells × {len(by)} policies.",
            "",
            f"- **H1** diversity: {verdict(s['H1_pass'], 'H1')}. Representations that are the oracle's main choice "
            f"in ≥ 10 % of cells: **{s['H1/reps_dominant_in_10pct_of_cells']}** (need ≥ 3); best-single costs "
            f"**{s['H1/b4_extra_cost_vs_oracle']:+.1%}** more than the oracle (need ≥ +25 %).",
            f"  Oracle's main choice by share of cells: {s['oracle_dominant_share']}."
            + (
                " A negative figure means best-single is the cheaper one: the oracle never sends a "
                "representation that disagrees, so it pays for raw or wavelet frames on the windows where the "
                "cheapest representation is wrong, and best-single does not."
                if s["H1/b4_extra_cost_vs_oracle"] < 0
                else ""
            ),
            f"- **H3** adaptation value: {verdict(s['H3_pass'], 'H3')}. Kai (S1, V0) total cost versus best-single: "
            f"**{s['H3/cost_reduction_vs_b4']:+.1%}** reduction, 95 % CI [{s['H3/cost_reduction_vs_b4_ci_lo']:+.1%}, "
            f"{s['H3/cost_reduction_vs_b4_ci_hi']:+.1%}] (need ≥ +20 %); SLO violations versus context-blind on "
            f"bursty traces: **{s['H3/slo_reduction_vs_b5_bursty']:+.1%}** reduction, CI "
            f"[{s['H3/slo_reduction_vs_b5_ci_lo']:+.1%}, {s['H3/slo_reduction_vs_b5_ci_hi']:+.1%}] (need ≥ +30 %).",
            f"  Geometric-mean per-cell cost ratio Kai / best-single: {s['H3/cost_geo_ratio_vs_b4']:.3f}.",
            *(
                [
                    "- Exploratory (ADR 0022), not part of H3: `kai_s1_auto` uses the static set in the "
                    f"{s['x/kai_s1_auto/n_static_folds']} of {s['n_folds']} folds where the predictor's unit-grouped "
                    "CV AUC on ptrain is below 0.55, and V0 elsewhere. Cost versus best-single: "
                    f"**{s['x/kai_s1_auto/cost_reduction_vs_b4']:+.1%}** reduction. Gate AUC per fold: "
                    + ", ".join(f"{a:.2f}" for a in s["x/kai_s1_auto/gate_auc"].values())
                    + "."
                ]
                if "x/kai_s1_auto/gate_auc" in s
                else []
            ),
            *(
                ["- Exploratory (ADR 0025), not part of H3 (cost versus best-single, 95 % CI; versus `kai_s1_v0`):"]
                + [
                    f"  - `{p}`: **{s[f'X25/{p}/cost_reduction_vs_b4']:+.1%}** "
                    f"[{s[f'X25/{p}/cost_reduction_vs_b4_ci_lo']:+.1%}, {s[f'X25/{p}/cost_reduction_vs_b4_ci_hi']:+.1%}]; "
                    f"{s[f'X25/{p}/cost_reduction_vs_kai']:+.1%} versus Kai."
                    for p in X25_POLICIES
                    if f"X25/{p}/cost_reduction_vs_b4" in s
                ]
                if any(f"X25/{p}/cost_reduction_vs_b4" in s for p in X25_POLICIES)
                else []
            ),
            *(
                [
                    "- Exploratory (ADR 0026), not part of H3 (cost versus best-single, 95 % CI; versus `kai_s1_v0`). "
                    "The `_dec` variant sends the committee's decisions: a certified edge-or-cloud cascade."
                ]
                + [
                    f"  - `{p}`: **{s[f'X26/{p}/cost_reduction_vs_b4']:+.1%}** "
                    f"[{s[f'X26/{p}/cost_reduction_vs_b4_ci_lo']:+.1%}, {s[f'X26/{p}/cost_reduction_vs_b4_ci_hi']:+.1%}]; "
                    f"{s[f'X26/{p}/cost_reduction_vs_kai']:+.1%} versus Kai."
                    for p in X26_POLICIES
                    if f"X26/{p}/cost_reduction_vs_b4" in s
                ]
                + (
                    [
                        f"  - `b14_edge_inference`: **{s['X26/b14_edge_inference/cost_reduction_vs_b4']:+.1%}** "
                        "versus best-single"
                        + (
                            f"; total cost {s['X26/b14_edge_inference/cost_vs_best_committee']:.2f}× that of the "
                            "best committee variant"
                            if "X26/b14_edge_inference/cost_vs_best_committee" in s
                            else ""
                        )
                        + "."
                    ]
                    if "X26/b14_edge_inference/cost_reduction_vs_b4" in s
                    else []
                )
                if any(k.startswith("X26/") for k in s)
                else []
            ),
            *budget_sweep(s, by),
            "",
            "Means over all cells, folds and the three α levels (cost in bytes-equivalent). "
            "`risk_certified_joint` = disagreeing *and* certified, per offered window: the quantity Proposition 1 "
            "bounds, for Kai with V0 / V1 only. `risk_certified` = disagreement among certified decisions, which "
            "divides by the certified rate and is not bounded. `risk_sent` = disagreement of whatever was "
            "transmitted. Baselines carry no certificate. B4 (best-single) and B8 screen representations with a marginal CRC-style check on calib, but that check bounds only a representation named before calibration; applied to every representation, keeping the cheapest that passes, it bounds nothing (multiple testing). The certified fixed-choice comparator is `kai_s1_v0_static`; `kai_s1_v2` is calibrated for the nominal state only "
            "and is exploratory here (docs/theory/proposition1.md). `kai_s1_cascade` is V0 on the score min(ĝ, prior) "
            "with a value-of-information gate on the probe (ADR 0011, exploratory). `kai_s1_v0_static` is V0 with constant "
            "scores — one safe set for every window, no probe or predictor on the edge — added after the "
            "first results and exploratory (ADR 0008). `kai_s1_v0_group` is V0 calibrated with one sample per "
            "calibration unit (ADR 0020, exploratory; raw only when there are fewer than ⌈1/α⌉ − 1 units). The pooled risks are comparable to no single "
            "α: the table after this one splits them.",
            "",
            table(
                by[["policy", "cost_total_per_fold", "bytes", "slo_violation", "risk_certified_joint",
                    "risk_certified", "risk_sent", "rate_certified", "rate_deferred", "rate_uncertified",
                    "latency_p50_ms",
                    *(["rate_certified_on_time"] if "rate_certified_on_time" in by else [])]]
            ),
            "",
        ]  # fmt: skip

        grid = pd.read_parquet(work() / "experiments" / "e3" / "results.parquet")
        kai = [
            "kai_s1_v0",
            "kai_s1_cascade",
            "kai_s1_v0_static",
            "kai_s1_v0_group",
            "kai_s1_v1",
            "kai_s1_v1_per_rep",
            "kai_s2_v0",
            "kai_s1_v2",
            *X25_POLICIES,
            *X26_POLICIES,
        ]
        kai = [p for p in kai if p in set(grid.policy)]
        by_alpha = grid[grid.policy.isin(kai)].pivot_table(
            index="alpha", columns="policy", values="risk_certified_joint"
        )
        lines += [
            "Certified-and-disagreeing rate per offered window, by α (mean over cells and folds):",
            "",
            table(by_alpha[kai].reset_index()),
            "",
        ]
        focus = [
            "oracle",
            "kai_s1_v0",
            "kai_s1_cascade",
            "kai_s1_v0_static",
            "kai_s1_v2",
            "kai_s1_v0_live",
            "b4_best_single",
        ]
        focus = [p for p in focus if p in set(grid.policy)]
        by_n = grid[grid.policy.isin(focus)].pivot_table(
            index="n_sensors", columns="policy", values="cost"
        )
        by_n = by_n[focus].assign(kai_vs_best_single=lambda t: t.kai_s1_v0 / t.b4_best_single - 1)
        lines += [
            "Exploratory breakdown, not pre-registered — mean cost per window by sensors per edge node "
            "(timings at each node's duty cycle, ADR 0008; with 32 sensors a window arrives every "
            f"{window_period_ms() / 32:.1f} ms):",
            "",
            table(by_n.reset_index()),
            "",
        ]
        small = grid[(grid.n_sensors < 32) & (grid.link == "static")]
        by_bw = small[small.policy.isin(["kai_s1_v0", "b4_best_single"])].pivot_table(
            index="bandwidth_bps", columns="policy", values=["cost", "rate_deferred"]
        )
        by_bw.columns = [f"{metric} {policy}" for metric, policy in by_bw.columns]
        lines += [
            "Exploratory — where Kai (V0) loses to best-single below 32 sensors: static links, mean over α, "
            "deadlines, loads and folds; `rate_deferred` is the share of windows Kai held back because "
            "nothing in their safe set fitted the link:",
            "",
            table(by_bw.reset_index()),
            "",
        ]

    # e3_slow<k> / e3_k<k> are slow-edge sweeps, not price presets (they have no H1 keys)
    presets = sorted(
        p
        for p in (work() / "experiments").glob("e3_*/summary.json")
        if not p.parent.name.startswith(("e3_slow", "e3_k"))
    )
    if presets and e["e3"]:
        rows = []
        for path in [work() / "experiments" / "e3" / "summary.json", *presets]:
            s = json.loads(path.read_text())
            rows.append(
                {
                    "price preset": s.get("price_preset", "metered_cellular"),
                    "folds": s["n_folds"],
                    "oracle's dominant representations (≥ 10 % of cells)": s[
                        "H1/reps_dominant_in_10pct_of_cells"
                    ],
                    "oracle's main choices": ", ".join(
                        f"{k} {v:.0%}" for k, v in s["oracle_dominant_share"].items()
                    ),
                    "best-single cost over oracle": s["H1/b4_extra_cost_vs_oracle"],
                    "Kai (V0) cost reduction vs best-single": s["H3/cost_reduction_vs_b4"],
                    "cascade vs best-single": s.get(
                        "x/kai_s1_cascade/cost_reduction_vs_b4", float("nan")
                    ),
                    "cascade probe rate": s.get("x/kai_s1_cascade/probe_rate", float("nan")),
                }
            )
        lines += [
            "### The same comparison under the other price presets (exploratory)",
            "",
            "H1 and H3 are registered on the `metered_cellular` preset. The other presets price energy, "
            "edge CPU time or lateness instead of bytes; they are reported to show whether the answer "
            "depends on what is priced.",
            "",
            table(pd.DataFrame(rows)),
            "",
        ]

    for name, title in (
        ("e5", "measured laptop timings"),
        ("e5_pi_class", "edge CPU times × 8 (modelled slower edge)"),
    ):
        if not e[name]:
            continue
        s = e[name]
        costs = {k.split("/", 1)[1]: v for k, v in s.items() if k.startswith("cost_total/")}
        rel = pd.DataFrame(
            {
                "policy": list(costs),
                "total cost / Kai": [v / costs["kai_s1_v0"] for v in costs.values()],
                "risk_certified (among certified; not α-bounded)": [
                    s[f"risk_certified/{k}"] for k in costs
                ],
                "SLO violation": [s[f"slo/{k}"] for k in costs],
            }
        ).sort_values("total cost / Kai")
        if name == "e5":
            pvalues["H5"] = stats.one_sided_p(
                _folds(s["per_fold"]["H5"], s.get("per_fold_ids")), 0.10
            )
            nb["H5"] = stats.nb_corrected_one_sided_p(
                _folds(s["per_fold"]["H5"], s.get("per_fold_ids")), 0.10, ratio
            )
        lines += [
            f"## Ablations (E5) — {title}",
            "",
            f"- **H4** encode cost matters: {verdict(s['H4_pass'])}. Main choice changed in "
            f"**{s['H4/choice_changed_in_high_load_cells']:.1%}** of ρ = 0.8 cells (need ≥ 15 %); SLO violation "
            f"{s['H4/slo_with_encode_cost']:.3f} with encode cost vs {s['H4/slo_ignoring_encode_cost']:.3f} ignoring it.",
            f"- **H5** heterogeneity pays: {verdict(s['H5_pass'], 'H5' if name == 'e5' else None)}. Full menu versus the best single family: "
            f"**{s['H5/cost_reduction_vs_best_family']:+.1%}** cost reduction, CI [{s['H5/ci_lo']:+.1%}, {s['H5/ci_hi']:+.1%}] "
            "(need ≥ +10 %).",
            "",
            table(rel),
            "",
        ]

    lines += slow_edge_section()
    if (dataset_id or work_dataset_id()).split("-")[0] in CONFIRMATORY:
        lines += confirmatory_section(dataset, ratio)

    if e["e4"]:
        s = e["e4"]

        def pct(value: float) -> str:
            return "n/a" if np.isnan(value) else f"{value:.1%}"  # NaN: nothing was matched

        h6_first, h6_latest = h6_numbers(reg, s)
        bias = ", ".join(
            f"{v:+.1%} at ρ = {k.removeprefix('latency_model_bias/load')}"
            for k, v in s.items()
            if k.startswith("latency_model_bias/")
        )
        lines += [
            "## Phase diagram: theory versus measurement (E4)",
            "",
            f"- **H6**: {verdict(h6_of_record(reg, s))}. Over the full menu the closed form predicts {h6_first['n_boundaries_theory']:g} boundaries; {h6_first['n_boundaries_measured']:g} argmin changes were measured and {h6_first['n_boundaries_matched']:g} match a predicted one (worst relative error {pct(h6_first['max_rel_error'])}; need ≥ 1 match within 20 %)"
            + (
                f". Re-measurement (latest run, no verdict): {h6_latest['n_boundaries_theory']:g} predicted, {h6_latest['n_boundaries_measured']:g} measured, {h6_latest['n_boundaries_matched']:g} matched, worst relative error {pct(h6_latest['max_rel_error'])}."
                if h6_latest
                else "."
            ),
            *(
                []
                if h6_of_record(reg, s)
                else [
                    "- Why it fails here: the compact representations are both the smallest and among the fastest end to end, and their latencies differ by less than the measurement noise, so the measured argmin flips among them and few break-even bandwidths exist to test."
                ]
            ),
            f"- Exploratory, not part of H6 — the same test inside the waveform-preserving family (raw, lossless, wavelets), where a break-even exists: {s['family/n_theory']} predicted, {s['family/n_measured']} measured, {s['family/n_matched']} matched, worst relative error {pct(s['family/max_rel_error'])}.",
            *(
                [
                    "- If E4 was run more than once (run history), the H6 verdict of record is the first run; a "
                    "re-measurement replaces no verdict, and a verdict that changes between measurements of the "
                    "same thing is not counted as met.",
                    "- Exploratory (ADR 0022), not part of H6 — a measured argmin change counts only if the latency "
                    "gap exceeds 2·√(MAD_lo² + MAD_hi²) at both grid points: "
                    f"{s['H6_noise_aware/n_measured']} changes remain, {s['H6_noise_aware/n_matched']} match a "
                    f"predicted boundary, worst relative error {pct(s['H6_noise_aware/max_rel_error'])} → "
                    + ("threshold met." if s["H6_noise_aware/pass"] else "threshold not met."),
                ]
                if "H6_noise_aware/n_measured" in s
                else []
            ),
            *(
                [
                    "- Exploratory (ADR 0024), not part of H6 — a measured argmin change counts only if the latency "
                    "gap exceeds 2·√(SE_lo² + SE_hi²), SE = 1.2533·1.4826·MAD/√n (standard error of a median), at "
                    f"both grid points: {s['H6x/median_se/changes']} changes remain, "
                    f"{s['H6x/median_se/matched']} match a predicted boundary, worst relative error "
                    f"{pct(s['H6x/median_se/worst_rel_error'])} → "
                    + (
                        "threshold met (exploratory, not a verdict)."
                        if s["H6x/median_se/threshold_met"]
                        else "threshold not met."
                    ),
                ]
                if "H6x/median_se/changes" in s
                else []
            ),
            f"- Latency model versus measured latency: median absolute error {s['median_latency_model_error']:.1%}; mean signed error {bias} (positive = reality slower than the model).",
            f"- Cells measured with no window delivered (no latency, cannot place a crossing): {s.get('n_cells_unmeasured', 'not recorded')}.",
            "- Measured on real sockets with token-bucket pacing, on the same laptop as the profile.",
            "",
        ]
        if s["family_boundaries"]:
            lines += [
                "Waveform-family boundaries:",
                "",
                table(pd.DataFrame(s["family_boundaries"])),
                "",
            ]

    for key, heading in (
        ("e7", "Sim-to-real over loopback sockets (E7)"),
        ("e7d", "Sim-to-real in Docker with netem (E7d)"),
    ):
        if not e[key]:
            continue
        s = e[key]
        lines += [
            f"## {heading}",
            "",
            f"- **H7**: {verdict(s['H7_pass'])}. Worst p50 latency gap **{s['max_p50_rel_gap']:.1%}** (need ≤ 10 %); "
            f"lowest agreement of the chosen representation **{s['min_selection_agreement']:.1%}** (need ≥ 85 %)"
            + (
                f"; **{s['n_rows_unmeasured']}** rate(s) delivered nothing, which fails H7."
                if s.get("n_rows_unmeasured")
                else "."
            ),
            f"- Backend: {s['backend']}.",
            "",
            table(pd.DataFrame(s["rows"])),
            "",
        ]

    if e["e6"]:
        s = e["e6"]
        levels = [lv for lv in ("L1", "L2") if f"{lv}/n_folds" in s]
        rows = [
            {"split": lv, **{k.split("/", 1)[1]: v for k, v in s.items() if k.startswith(f"{lv}/")}}
            for lv in levels
        ]
        lines += [
            "## Drift and operating-condition shift (E6)",
            "",
            f"α = {s['alpha']}, audit probability {s['p_audit']}, 1 Mbit/s, one sensor. L1 holds bearings out; L2 additionally holds one operating condition out.",
            "",
        ]
        for lv in levels:
            lines.append(
                f"- **{lv}** ({int(s[f'{lv}/n_folds'])} folds): safe-set risk (worst admissible policy) {s[f'{lv}/safe_set_risk_static']:.3f} with the static threshold; "
                f"realised risk of the greedy policy {s[f'{lv}/policy_risk_static']:.3f} with the static threshold, "
                f"{s[f'{lv}/policy_risk_aci']:.3f} with audit-in-the-loop ACI, at {s[f'{lv}/bytes_aci'] / s[f'{lv}/bytes_static'] - 1:+.0%} bytes "
                f"(audit copies: {s[f'{lv}/audit_bytes']:.0f} B per window). The cloud's own estimate from audited decisions, per offered window: {s[f'{lv}/risk_audit_estimate']:.3f}. "
                f"The drift monitor is raised on {s[f'{lv}/drift_fired']:.0%} of test windows."
            )
        lines += [
            *(
                [
                    "- The probe-feature drift monitor is up on most test windows even without a condition shift: it "
                    "detects that the test bearings differ from the calibration bearings. As a safe-mode trigger it "
                    "would switch the system to raw-only on every new machine (ADR 0008 §5)."
                ]
                if s.get("L1/drift_fired", 0.0) > 0.5
                else []
            ),
            "",
            table(pd.DataFrame(rows)),
            "",
        ]

    if e["e8"]:
        s = e["e8"]
        rows = [
            {
                "selector": k.split("/")[0],
                "µs per decision": v,
                "decisions per second": s[k.replace("us_per_decision", "decisions_per_s")],
            }
            for k, v in s.items()
            if k.endswith("/us_per_decision")
        ]
        lines += [
            "## Selector overhead (E8, laptop, single core)",
            "",
            table(pd.DataFrame(rows)),
            "",
            f"Probe + predictor + selector take {s['overhead_ms_probe_pred_select']:.3f} ms per window: "
            f"{s['overhead_vs_cheapest_encoder']:.2f}× the cheapest encoder and "
            f"{s.get('overhead_share_of_window_period', s.get('overhead_share_of_64ms_window', float('nan'))):.2%} "
            f"of a window period ({s.get('window_period_ms', float('nan')):.1f} ms).",
            "",
        ]

    if e["e10"]:
        s = e["e10"]
        reps = sorted(
            {k.split("/")[0] for k in s if "/" in k}, key=lambda r: -s[f"{r}/payload_bytes"]
        )
        rows = [
            {
                "what the observer sees": "band-energy shape of the raw window"
                if r == "r0_raw"
                else r,
                "bytes": s[f"{r}/payload_bytes"],
                "operating condition, unseen machines": s[f"{r}/condition_accuracy"],
                "re-identification of a seen machine": s[f"{r}/reid_accuracy"],
            }
            for r in reps
        ]
        lines += [
            "## Privacy probe (E10, exploratory)",
            "",
            f"Accuracy of a logistic-regression attacker on what the cloud decodes, mean of {s['n_folds']} folds. "
            f"Chance is {s['r0_raw/condition_chance']:.2f} for the operating condition "
            f"({round(1 / s['r0_raw/condition_chance'])} conditions) and "
            f"{s['r0_raw/reid_chance']:.2f} on average for re-identification among a fold's test bearings. "
            "Re-identification is fitted and scored on windows of the same bearings by construction: it measures "
            "the attack, it is not a diagnosis result. No hypothesis was registered for this probe.",
            "",
            table(pd.DataFrame(rows)),
            "",
            "A compact payload is not a private one: sending fewer bytes hides the waveform, not the machine's "
            "identity or its operating point, unless the numbers above are at chance.",
            "",
        ]

    # hypotheses without a replication unit enter the family with p = 1: they cannot be rejected
    family = {h: pvalues.get(h, 1.0) for h in FAMILY}
    adj = stats.holm(family)
    adj_nb = stats.holm({h: nb.get(h, 1.0) for h in FAMILY})
    if pvalues:
        lines += [
            "## Multiple testing",
            "",
            "One-sided t-tests of each pre-registered threshold over "
            + "folds, as pre-registered (folds of different repeats share test bearings, so these "
            "p-values are optimistic), " + "Holm–Bonferroni at 0.05 across "
            "all eight hypotheses. H2b, H4, H6 and H7 have no replication unit in this pass: they are "
            "descriptive and enter the family with p = 1. A hypothesis is called *supported* only if its "
            "threshold is met **and** Holm rejects the null. The column *p (N–B, Holm)* is "
            f"**exploratory** (ADR 0024, test/train ratio {ratio:.2f}): a Nadeau–Bengio corrected "
            "t-test for H1, H2a, H3 and H5 that allows for overlapping training sets; no verdict "
            "uses it.",
            "",
            table(
                pd.DataFrame(
                    [
                        {
                            "hypothesis": h,
                            "tested over folds": h in pvalues,
                            "p (raw)": family[h],
                            "p (Holm)": adj[h][0],
                            "p (N–B, Holm)": adj_nb[h][0],
                            "rejects the null": adj[h][1],
                        }
                        for h in FAMILY
                    ]
                )
            ),
            "",
        ]
    if registry.exists():
        keys = {
            "e2": ["v0/a0.05/risk_mean", "v0/a0.05/bytes_vs_raw"],
            "e3": [
                "H3/cost_reduction_vs_b4",
                "H3/slo_reduction_vs_b5_bursty",
                "H1/b4_extra_cost_vs_oracle",
            ],
            "e5": ["H5/cost_reduction_vs_best_family", "H4/choice_changed_in_high_load_cells"],
            "e4": ["H6_pass", "n_boundaries_matched"],
            "e7": ["max_p50_rel_gap", "min_selection_agreement"],
        }
        rows = []
        for rec in reg.itertuples():
            if rec.experiment in keys:
                m = json.loads(rec.metrics)
                rows.append(
                    {"run": rec.created_utc, "experiment": rec.experiment}
                    | {k: m.get(k, float("nan")) for k in keys[rec.experiment]}
                )
        hist = pd.DataFrame(rows)
        lines += [
            "## Run history",
            "",
            "Every run is kept in the registry; nothing is discarded. Experiments that were run more than "
            "once changed for a stated reason. On 2026-10-10 (`docs/adr/0004-reference-model-and-heads.md`): the "
            "first E2 run predates the exchangeability control; E2 / E3 / E5 / E7 were re-run after the timing "
            "profile was re-measured at the real duty cycle; and E3 / E5 were re-run once more after the "
            "simulator stopped charging Kai's probe and predictor to policies that do not run them. "
            "On 2026-10-14 local time (rows from 2026-10-13 22:40 UTC on) everything was re-run with the new "
            "reference model and R6 "
            "(`docs/adr/0006-envelope-reference-model.md`) and the audit-loop fix "
            "(`docs/adr/0007-certificate-review.md`); the rows before that date belong to the SpecNet folds. "
            "The sections above show the latest run.",
            "",
        ]
        for _, sub in hist.groupby("experiment"):
            lines += [table(sub.dropna(axis=1, how="all")), ""]
    out.write_text(resolve_verdicts("\n".join(lines), adj))
    print(out)


if __name__ == "__main__":
    typer.run(main)
