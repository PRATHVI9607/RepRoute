"""Quick screening: will the representation menu have real choices on this dataset?

From the team note "RepRoute: New Datasets and Screening Script" (2026-10-28), adapted to the repo:
reads a repo WindowSet (.npz with x / label / bearing_id) or the note's format (X / y / unit),
passes a Generator instead of a module-level RNG, counts raw as int16 (2 bytes per sample, as
stored), and writes the verdict to experiments/screening/<name>.json. Cheap stand-ins for R2, R3,
R4, R5 and R7; R6 is left out on purpose. A yes/no before the full pipeline, not a result.

    uv run --no-sync python scripts/screen_dataset.py transfer/ptbxl/data/processed/ptbxl.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pywt
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / "experiments" / "screening"


def r0_raw(x):
    # full-resolution log spectrum: what the cloud model sees
    return np.log1p(np.abs(np.fft.rfft(x, axis=1))), x.shape[1] * 2


def r2_wavelet(x, keep=0.10):
    c = np.array([np.concatenate(pywt.wavedec(w, "db4", level=4)) for w in x])
    k = max(1, int(c.shape[1] * keep))
    thresh = -np.sort(-np.abs(c), axis=1)[:, k - 1 : k]
    return np.where(np.abs(c) >= thresh, c, 0.0), k * 6  # value + index per kept coefficient


def r3_spectral(x, bands=32):
    p = np.abs(np.fft.rfft(x, axis=1)) ** 2
    return np.log1p(np.stack([b.mean(1) for b in np.array_split(p, bands, axis=1)], 1)), bands * 4


def r4_summary(x):
    d = np.diff(x, axis=1)
    kurt = ((x - x.mean(1, keepdims=True)) ** 4).mean(1) / (x.var(1) ** 2 + 1e-9)
    zcr = ((x[:, :-1] * x[:, 1:]) < 0).mean(1)
    feats = [x.mean(1), x.std(1), x.min(1), x.max(1), np.percentile(x, 25, 1),
             np.percentile(x, 75, 1), zcr, np.abs(d).mean(1), d.std(1), kurt]  # fmt: skip
    return np.stack(feats, axis=1), 10 * 4


def r5_latent(x, pca):
    return pca.transform(x), pca.n_components_ * 4


def r7_quantised(x, pca, bits=4):
    z = pca.transform(x)
    lo, hi = np.percentile(z, 1, 0), np.percentile(z, 99, 0)
    levels = 2**bits - 1
    q = np.round((np.clip(z, lo, hi) - lo) / (hi - lo + 1e-9) * levels)
    return q / levels * (hi - lo) + lo, int(np.ceil(pca.n_components_ * bits / 8))


def head():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def unit_split(units, test_size, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(np.zeros(len(units)), groups=units))


def build_all(x, pca):
    return {
        "R2_wavelet": r2_wavelet(x),
        "R3_spectral": r3_spectral(x),
        "R4_summary": r4_summary(x),
        "R5_latent": r5_latent(x, pca),
        "R7_quantised": r7_quantised(x, pca),
    }


def load(path: Path):
    d = np.load(path, allow_pickle=False)
    if "x" in d:  # a repo WindowSet
        return d["x"], d["label"], d["bearing_id"]
    return d["X"], d["y"], d["unit"]


def screen(path: Path, max_windows: int, rng: np.random.Generator) -> dict:  # noqa: PLR0915 - the note's five steps, in order
    x, y, unit = load(path)
    if len(x) > max_windows:
        idx = np.sort(rng.choice(len(x), max_windows, replace=False))
        x, y, unit = x[idx], y[idx], unit[idx]
    x = x.astype(np.float32)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    res: dict = {"dataset": path.stem, "windows": len(x), "units": len(np.unique(unit)),
                 "classes": len(np.unique(y)), "window_len": x.shape[1]}  # fmt: skip
    print(res)

    tr, te = unit_split(unit, test_size=0.3)
    xtr, ytr, utr, xte, yte = x[tr], y[tr], unit[tr], x[te], y[te]
    # 1. cloud reference model
    f0_tr, raw_bytes = r0_raw(xtr)
    f0_te, _ = r0_raw(xte)
    f_raw = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(f0_tr, ytr)
    ref, ref_tr = f_raw.predict(f0_te), f_raw.predict(f0_tr)
    res["f_raw_macro_f1"] = float(f1_score(yte, ref, average="macro"))
    res["chance"] = 1 / res["classes"]
    print(
        f"\nf_raw macro-F1 on unseen units = {res['f_raw_macro_f1']:.3f}  "
        f"(chance ~ {res['chance']:.3f})"
    )

    # 2. disagreement per representation
    pca = PCA(n_components=16, random_state=0).fit(xtr)
    reps_tr, reps_te = build_all(xtr, pca), build_all(xte, pca)
    agree, nbytes = {}, {"R0_raw": raw_bytes}
    print("\nrepresentation   bytes   disagreement")
    for name in reps_tr:
        h = head().fit(reps_tr[name][0], ref_tr)  # learn to copy f_raw
        agree[name] = h.predict(reps_te[name][0]) == ref
        nbytes[name] = reps_te[name][1]
        print(f"{name:15s} {nbytes[name]:7d}   {1 - agree[name].mean():.1%}")
    agree["R0_raw"] = np.ones(len(ref), bool)
    res["disagreement"] = {n: float(1 - a.mean()) for n, a in agree.items()}
    res["bytes"] = nbytes
    medium = [n for n in agree if n != "R0_raw" and 0.02 <= 1 - agree[n].mean() <= 0.25]

    # 3. pretend picker at three speeds (bytes allowed per window within the deadline); an oracle
    # on agreement, so an upper bound on how much of the menu a certified picker could use
    order = sorted(nbytes, key=nbytes.get)
    usage = dict.fromkeys(order, 0)
    for budget in (64, 1024, raw_bytes):
        for i in range(len(ref)):
            ok = [n for n in order if nbytes[n] <= budget and agree[n][i]]
            usage[(ok or [n for n in order if nbytes[n] <= budget])[0]] += 1
    total = sum(usage.values())
    res["picker_usage"] = {n: c / total for n, c in usage.items() if c}
    print("\npretend picker usage:", {n: f"{v:.0%}" for n, v in res["picker_usage"].items()})
    used = [n for n, v in res["picker_usage"].items() if v >= 0.10]

    # 4. can cheap features tell hard windows apart?
    hard = ~agree["R4_summary"]
    res["hardness_auc"] = None
    if 0 < hard.mean() < 1:
        a, b = unit_split(unit[te], test_size=0.5, seed=1)
        if len(np.unique(hard[a])) == 2 and len(np.unique(hard[b])) == 2:
            clf = head().fit(reps_te["R4_summary"][0][a], hard[a])
            p = clf.predict_proba(reps_te["R4_summary"][0][b])[:, 1]
            res["hardness_auc"] = float(roc_auc_score(hard[b], p))
            print(f"hardness predictable from cheap features: AUC = {res['hardness_auc']:.2f}")

    # 5. more-units curve
    print("\nmore-units curve (median disagreement of cheap reps):")
    all_units, curve = np.unique(utr), {}
    for n_units in (10, 50, 200, len(all_units)):
        if n_units > len(all_units):
            continue
        keep = np.isin(utr, rng.choice(all_units, n_units, replace=False))
        target = ref_tr[keep]
        if len(np.unique(target)) < 2:
            continue
        dis = [
            1 - (head().fit(reps_tr[n][0][keep], target).predict(reps_te[n][0]) == ref).mean()
            for n in reps_tr
        ]
        curve[int(n_units)] = float(np.median(dis))
        print(f"  {n_units:5d} units -> {curve[n_units]:.1%}")
    res["units_curve"] = curve

    m, u = len(medium), len(used)
    res["medium_reps"], res["used_reps"] = medium, used
    res["verdict"] = "PASS" if m >= 3 and u >= 3 else "MAYBE" if m >= 2 and u >= 2 else "FAIL"
    print(f"\nmedium-good reps: {medium}\nreps used >=10%:  {used}\nVERDICT: {res['verdict']}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--max-windows", type=int, default=20000)
    ap.add_argument("--name", default=None, help="output name (default: the file's stem)")
    args = ap.parse_args()
    res = screen(args.npz, args.max_windows, np.random.default_rng(0))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name or args.npz.stem}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
