"""PTB-XL 12-lead ECG (transfer dataset, other sensor domain): WFDB records → int16 WindowSet.

Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset", Sci Data 7, 154
(2020); PhysioNet v1.0.3, doi:10.13026/kfzx-aw45, CC BY 4.0. 21,799 10-s records from 18,869
patients, 500 Hz, WFDB format 16 (little-endian int16, signals interleaved by sample).
Here the "bearing" is the patient: one record per patient, so splits leave patients out.
Labels: diagnostic superclasses from SCP codes with likelihood 100; a record is kept only if its
superclass set is exactly {NORM}, {MI} or {STTC}. The signal is lead II, raw ADC counts, first
`window_n` samples. Headers and metadata are untrusted data: parsed, never executed.
"""

from __future__ import annotations

import ast
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kai.contracts.windows import WindowSet

SAMPLE_RATE_HZ = 500.0
DATASET_ID = "ptbxl-v1"
LEAD = "II"
LABELS = {"NORM": 0, "MI": 1, "STTC": 2}
LIKELIHOOD = 100.0  # 100 leaves 10,266 patients (≥1,283 per class), so the > 0 fallback is unused
CONDITION = "rest"
DATABASE = "ptbxl_database.csv"
STATEMENTS = "scp_statements.csv"
_RECORD = re.compile(r"records500/\d{5}/\d{5}_hr")  # filename_hr must look like this to be opened
_FORMAT = re.compile(r"16(?:\+(\d+))?")  # format 16, optional byte offset; skew / xsamp rejected


def catalogue(raw_dir: Path) -> tuple[dict[str, tuple[str, int]], dict[str, int]]:
    """record path (filename_hr) → (bearing_id, label); and the dropped-record counts by reason."""
    with (raw_dir / STATEMENTS).open(newline="") as f:
        superclass = {
            r[""]: r["diagnostic_class"] for r in csv.DictReader(f) if float(r["diagnostic"] or 0)
        }
    with (raw_dir / DATABASE).open(newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["ecg_id"]))
    records: dict[str, tuple[str, int]] = {}
    patients: set[str] = set()
    dropped: Counter[str] = Counter()
    for r in rows:
        codes = ast.literal_eval(r["scp_codes"])  # a dict literal; literal_eval never executes
        found = {superclass[c] for c, lik in codes.items() if c in superclass and lik == LIKELIHOOD}
        patient = f"P{int(float(r['patient_id']))}"
        if not found:
            dropped["no_superclass_at_likelihood_100"] += 1
        elif len(found) > 1:
            dropped["several_superclasses"] += 1
        elif (name := found.pop()) not in LABELS:
            dropped[f"superclass_{name}"] += 1
        elif patient in patients:  # the earliest kept record (by ecg_id) represents the patient
            dropped["later_record_of_patient"] += 1
        elif not _RECORD.fullmatch(r["filename_hr"]):
            dropped["bad_filename"] += 1
        else:
            patients.add(patient)
            records[r["filename_hr"]] = (patient, LABELS[name])
    return records, dict(dropped)


def read_lead(record: Path, lead: str = LEAD) -> npt.NDArray[np.int16]:
    """One lead of a WFDB format-16 record (`record` without suffix), checked against its header
    checksum. Raises KeyError if the lead is absent and ValueError on anything unexpected."""
    lines = [s for s in record.with_suffix(".hea").read_text().splitlines() if s.strip()]
    lines = [s for s in lines if not s.startswith("#")]
    head = lines[0].split()
    n_sig, n_samples = int(head[1]), int(head[3])
    signals = [s.split(maxsplit=8) for s in lines[1 : 1 + n_sig]]
    names = [s[8] if len(s) > 8 else "" for s in signals]
    i = names.index(lead) if lead in names else -1
    if i < 0:
        raise KeyError(f"{record.name}: no lead {lead} in {names}")
    files, fmts = {s[0] for s in signals}, {s[1] for s in signals}
    fmt = _FORMAT.fullmatch(fmts.pop()) if len(fmts) == 1 else None
    if files != {f"{record.name}.dat"} or fmt is None:  # one interleaved .dat beside the header
        raise ValueError(f"{record.name}: unsupported layout {files} {signals[i][1]}")
    data = np.fromfile(record.with_suffix(".dat"), dtype="<i2", offset=int(fmt.group(1) or 0))
    if len(data) != n_sig * n_samples:
        raise ValueError(f"{record.name}: {len(data)} values, header says {n_sig}×{n_samples}")
    x = data.reshape(n_samples, n_sig)[:, i].astype(np.int16)
    if int(x.astype(np.int64).sum()) % 65536 != int(signals[i][6]) % 65536:
        raise ValueError(f"{record.name}: lead {lead} fails the header checksum")
    return x


def build_windowset(raw_dir: Path, window_n: int = 4096) -> tuple[WindowSet, dict[str, object]]:
    """Lead II, first `window_n` samples of each catalogued record found under `raw_dir`."""
    records, dropped = catalogue(raw_dir)
    xs, bids, labels = [], [], []
    skipped: dict[str, list[str]] = {
        k: [] for k in ("missing", "lead_missing", "unreadable", "too_short")
    }
    for name, (patient, label) in records.items():
        if not (raw_dir / f"{name}.hea").exists() or not (raw_dir / f"{name}.dat").exists():
            skipped["missing"].append(name)
            continue
        try:
            x = read_lead(raw_dir / name)
        except KeyError:
            skipped["lead_missing"].append(name)
            continue
        except (ValueError, IndexError):  # malformed header or data: counted, never hidden
            skipped["unreadable"].append(name)
            continue
        if len(x) < window_n:
            skipped["too_short"].append(name)
            continue
        xs.append(x[:window_n])
        bids.append(patient)
        labels.append(label)
    x = np.stack(xs)
    y = np.asarray(labels, dtype=np.int64)
    ws = WindowSet(
        x=x,
        window_id=np.arange(len(x), dtype=np.int64),
        bearing_id=np.asarray(bids),
        condition=np.full(len(x), CONDITION),
        label=y,
        sample_rate_hz=SAMPLE_RATE_HZ,
        dataset_id=DATASET_ID,
    )
    rails = (x == np.iinfo(np.int16).min) | (x == np.iinfo(np.int16).max)
    report: dict[str, object] = {
        "n_windows": len(x),
        "n_patients": len(set(bids)),
        "windows_per_class": {c: int((y == k).sum()) for c, k in LABELS.items()},
        "n_catalogued": len(records),
        "dropped_records": dropped,
        "skipped_records": skipped,
        "clipped_fraction": float(rails.mean()),
        "windows_with_clipping": int(rails.any(axis=1).sum()),
        "min": int(x.min()),
        "max": int(x.max()),
        "peak_abs": int(np.abs(x.astype(np.int32)).max()),
        "window_n": window_n,
        "lead": LEAD,
        "likelihood": LIKELIHOOD,
    }
    return ws, report
