"""PTB-XL loader on tiny fake WFDB records and metadata in the published layout (no network)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from kai.data import ptbxl

LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _write_record(raw, name: str, x: np.ndarray, leads=LEADS) -> None:
    """x is [n_samples, n_sig] int16; written as format 16 with the PTB-XL header layout."""
    rec = raw / name
    rec.parent.mkdir(parents=True, exist_ok=True)
    stem = rec.name
    lines = [f"{stem} {len(leads)} 500 {len(x)}"]
    for j, lead in enumerate(leads):
        col = x[:, j].astype(np.int64)
        lines.append(f"{stem}.dat 16 1000.0(0)/mV 16 0 {col[0]} {col.sum() % 65536} 0 {lead}")
    rec.with_suffix(".hea").write_text("\n".join(lines) + "\n")
    x.astype("<i2").tofile(rec.with_suffix(".dat"))


def _write_metadata(raw, rows: list[tuple[int, float, str, str]]) -> None:
    with (raw / ptbxl.STATEMENTS).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "description", "diagnostic", "diagnostic_class"])
        for code, cls in [("NORM", "NORM"), ("IMI", "MI"), ("NDT", "STTC"), ("LAFB", "CD")]:
            w.writerow([code, "x", "1.0", cls])
        w.writerow(["SR", "sinus rhythm", "", ""])  # a rhythm statement: no superclass
    with (raw / ptbxl.DATABASE).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ecg_id", "patient_id", "scp_codes", "strat_fold", "filename_hr"])
        for ecg_id, patient, codes, filename in rows:
            w.writerow([ecg_id, patient, codes, 1, filename])


def _rec(i: int) -> str:
    return f"records500/00000/{i:05d}_hr"


def test_read_lead_picks_lead_ii_raw_int16(tmp_path, rng: np.random.Generator) -> None:
    x = rng.integers(-32768, 32768, size=(5000, 12)).astype(np.int16)
    _write_record(tmp_path, _rec(1), x)
    lead = ptbxl.read_lead(tmp_path / _rec(1))
    assert lead.dtype == np.int16
    assert np.array_equal(lead, x[:, 1])  # II is the second signal: interleaving is respected
    assert np.array_equal(ptbxl.read_lead(tmp_path / _rec(1), "V6"), x[:, 11])

    _write_record(tmp_path, _rec(2), x[:, :1], leads=("I",))
    with pytest.raises(KeyError):
        ptbxl.read_lead(tmp_path / _rec(2))
    hea = (tmp_path / _rec(1)).with_suffix(".hea")
    hea.write_text(hea.read_text().replace(" 16 1000", " 212 1000"))
    with pytest.raises(ValueError, match="unsupported"):  # only format 16 is parsed
        ptbxl.read_lead(tmp_path / _rec(1))


def test_windowset_from_fake_records(tmp_path, rng: np.random.Generator) -> None:
    rows = [
        (1, 101.0, "{'NORM': 100.0, 'SR': 0.0}", _rec(1)),  # P101 → NORM
        (2, 101.0, "{'IMI': 100.0}", _rec(2)),  # later record of P101: dropped
        (3, 102.0, "{'IMI': 100.0, 'NORM': 50.0}", _rec(3)),  # likelihood 50 is ignored → MI
        (4, 103.0, "{'NDT': 100.0, 'IMI': 100.0}", _rec(4)),  # MI + STTC: dropped
        (5, 103.0, "{'NDT': 100.0}", _rec(5)),  # earliest kept record of P103 → STTC
        (6, 104.0, "{'LAFB': 100.0}", _rec(6)),  # CD only: dropped
        (7, 105.0, "{'NORM': 80.0}", _rec(7)),  # nothing at likelihood 100: dropped
        (8, 106.0, "{'NORM': 100.0}", "../../etc/passwd"),  # never becomes a path
        (9, 107.0, "{'NORM': 100.0}", _rec(9)),  # catalogued but not downloaded
        (10, 108.0, "{'NORM': 100.0}", _rec(10)),  # shorter than a window
    ]
    _write_metadata(tmp_path, rows)
    signals = {i: rng.integers(-3000, 3000, size=(5000, 12)).astype(np.int16) for i in range(1, 7)}
    for i, x in signals.items():
        _write_record(tmp_path, _rec(i), x)
    _write_record(tmp_path, _rec(10), signals[1][:4000])

    ws, report = ptbxl.build_windowset(tmp_path)

    assert (len(ws), ws.window_n) == (3, 4096)
    assert (ws.sample_rate_hz, ws.dataset_id) == (500.0, "ptbxl-v1")
    assert ws.bearing_labels() == {"P101": 0, "P102": 1, "P103": 2}
    assert set(ws.condition.tolist()) == {"rest"}
    for i, k in ((1, 0), (3, 1), (5, 2)):
        assert np.array_equal(ws.x[k], signals[i][:4096, 1])  # first 4096 samples of lead II
    assert report["windows_per_class"] == {"NORM": 1, "MI": 1, "STTC": 1}
    assert report["dropped_records"] == {
        "later_record_of_patient": 1,
        "several_superclasses": 1,
        "superclass_CD": 1,
        "no_superclass_at_likelihood_100": 1,
        "bad_filename": 1,
    }
    assert report["skipped_records"] == {
        "missing": [_rec(9)],
        "lead_missing": [],
        "unreadable": [],
        "too_short": [_rec(10)],
    }
    assert (report["n_patients"], report["n_catalogued"]) == (3, 5)
    kept = np.stack([signals[i][:4096, 1] for i in (1, 3, 5)])
    assert (report["min"], report["max"]) == (kept.min(), kept.max())
    assert report["clipped_fraction"] == 0.0
