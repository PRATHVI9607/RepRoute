"""Choose a dataset's reference model by grouped inner CV on the TRAIN partition of fold r0f0.

Generic over datasets (the Paderborn search is `model_search.py`). Every candidate is recorded in
`experiments/model_search/<dataset>.jsonl`, losers included. ptrain / calib / test are never read.

    KAI_WORK=transfer/ptbxl uv run --no-sync python scripts/select_reference.py dataset=ptbxl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from kai.bench.config import load_config, work
from kai.contracts.windows import WindowSet
from kai.data.splits import load_manifest, partition_mask
from kai.tasks import train as tr

OUT = Path(__file__).resolve().parents[1] / "experiments" / "model_search"


def main(overrides: list[str]) -> None:
    cfg = load_config(overrides)
    ws = WindowSet.load(work() / cfg.dataset.windows)
    (path,) = (work() / cfg.dataset.splits_dir).glob(f"{ws.dataset_id}_L1_*.json")
    fold = load_manifest(path).folds[0]
    m = partition_mask(ws, fold, "train")
    x, y, g = ws.x[m], ws.label[m], ws.bearing_id[m]
    device = tr.resolve_device("auto")
    candidates = {
        "spec (SpecNet, ADR 0004)": lambda xa, ya, ga: tr.train_specnet(xa, ya, ga, 0),
        "env (EnvNet, ADR 0006)": lambda xa, ya, ga: tr.train_envnet(xa, ya, ga, 0),
        "wave (RawNet CNN, 30 epochs)": lambda xa, ya, ga: tr.train_rawnet(xa, ya, 30, 0, device),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fit in candidates.items():
        t0, scores = time.perf_counter(), []
        for tr_idx, va_idx in GroupKFold(3).split(x, y, g):
            model = fit(x[tr_idx], y[tr_idx], g[tr_idx])
            logits = tr.predict(model, x[va_idx].astype(np.float32) / 32768.0)
            scores.append(float(f1_score(y[va_idx], logits.argmax(1), average="macro")))
        row = {
            "dataset": ws.dataset_id, "candidate": name, "inner_folds": 3,
            "train_groups": len(np.unique(g)), "inner_f1_mean": float(np.mean(scores)),
            "inner_f1_per_fold": scores, "seconds": round(time.perf_counter() - t0, 1),
        }  # fmt: skip
        with (OUT / f"{ws.dataset_id}.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{name:32s} inner-CV macro-F1 {row['inner_f1_mean']:.3f}  {np.round(scores, 3)}")


if __name__ == "__main__":
    main(sys.argv[1:])
