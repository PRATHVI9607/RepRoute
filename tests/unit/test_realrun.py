"""Unit tests for subprocess telemetry, JSONL parsing, and telemetry correlation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kai.bench.realrun import correlate_telemetry, parse_telemetry_jsonl


def test_parse_telemetry_jsonl_valid(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    lines = [
        {
            "window_id": 1,
            "rep_id": "r0_raw",
            "decision": 0,
            "measured_cloud_dec_ms": 0.5,
            "measured_cloud_head_ms": 1.2,
            "cloud_ms": 1.7,
        },
        {
            "window_id": 2,
            "rep_id": "r4_summary",
            "decision": 1,
            "measured_cloud_dec_ms": 0.2,
            "measured_cloud_head_ms": 0.8,
            "cloud_ms": 1.0,
        },
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    events = parse_telemetry_jsonl(path)
    assert len(events) == 2
    assert (1, "r0_raw") in events
    assert (2, "r4_summary") in events

    e1 = events[(1, "r0_raw")]
    assert e1["decision"] == 0
    assert e1["measured_cloud_dec_ms"] == 0.5
    assert e1["measured_cloud_head_ms"] == 1.2
    assert e1["cloud_ms"] == 1.7


def test_parse_telemetry_jsonl_empty_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n   \n", encoding="utf-8")
    events = parse_telemetry_jsonl(path)
    assert events == {}

    missing_path = tmp_path / "nonexistent.jsonl"
    assert parse_telemetry_jsonl(missing_path) == {}


def test_parse_telemetry_jsonl_malformed_syntax_raises(tmp_path: Path) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text(
        '{"window_id": 1, "rep_id": "r0_raw", "decision": 0}\n{bad json}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed JSONL line 2"):
        parse_telemetry_jsonl(path)


def test_parse_telemetry_jsonl_missing_fields_raises(tmp_path: Path) -> None:
    path = tmp_path / "missing_fields.jsonl"
    # Missing 'decision'
    path.write_text('{"window_id": 1, "rep_id": "r0_raw"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing field 'decision'"):
        parse_telemetry_jsonl(path)


def test_parse_telemetry_jsonl_duplicate_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    lines = [
        {"window_id": 10, "rep_id": "r0_raw", "decision": 0},
        {"window_id": 10, "rep_id": "r0_raw", "decision": 1},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate telemetry event"):
        parse_telemetry_jsonl(path)


def test_correlate_telemetry_success() -> None:
    code_to_rep_id = {0: "r0_raw", 5: "r4_summary"}
    records = [
        {"window_id": 1, "rep_code": 0, "decision": 0, "cloud_ms": 1.5},
        {"window_id": 2, "rep_code": 5, "decision": 1, "cloud_ms": 0.8},
        # telemetry missing for window 3:
        {"window_id": 3, "rep_code": 0, "decision": 0, "cloud_ms": 1.2},
    ]
    events = {
        (1, "r0_raw"): {"measured_cloud_dec_ms": 0.4, "measured_cloud_head_ms": 1.1},
        (2, "r4_summary"): {"measured_cloud_dec_ms": 0.1, "measured_cloud_head_ms": 0.7},
    }

    correlate_telemetry(records, events, code_to_rep_id)

    # Window 1 correlated
    assert records[0]["measured_cloud_dec_ms"] == 0.4
    assert records[0]["measured_cloud_head_ms"] == 1.1
    assert records[0]["cloud_ms"] == 1.5

    # Window 2 correlated
    assert records[1]["measured_cloud_dec_ms"] == 0.1
    assert records[1]["measured_cloud_head_ms"] == 0.7
    assert records[1]["cloud_ms"] == 0.8

    # Window 3 missing telemetry: fields must remain ABSENT (never replaced with 0.0)
    assert "measured_cloud_dec_ms" not in records[2]
    assert "measured_cloud_head_ms" not in records[2]
    assert records[2]["cloud_ms"] == 1.2


def test_correlate_telemetry_skips_unmatched_or_unknown_codes() -> None:
    code_to_rep_id = {0: "r0_raw"}
    records = [
        {"window_id": 1, "rep_code": 99},  # unknown rep_code
        {"window_id": 2},  # missing rep_code
        {"rep_code": 0},  # missing window_id
    ]
    events = {(1, "r0_raw"): {"measured_cloud_dec_ms": 0.5}}

    correlate_telemetry(records, events, code_to_rep_id)
    assert "measured_cloud_dec_ms" not in records[0]
    assert "measured_cloud_dec_ms" not in records[1]
    assert "measured_cloud_dec_ms" not in records[2]


@pytest.mark.xdist_group("fold")
def test_cloud_process_and_real_run_telemetry(bundle_dir, fold_dir, ws, tmp_path: Path) -> None:
    from kai.bench.realrun import cloud_process, real_run
    from kai.contracts.schemas import EnergyModel, Prices
    from kai.edge.bundle import load_bundle
    from kai.edge.runtime import EdgeConfig
    from kai.select.policies import Static

    bundle = load_bundle(bundle_dir)
    prices = Prices(name="t", story="t", p_byte=1.0, p_j=0.0, p_cpu_ms=0.0, p_late_ms=100.0)
    energy = EnergyModel(p_cpu_w=1.0, e_bit_j=1e-7)
    cfg = EdgeConfig(alpha=0.10, deadline_ms=500, period_ms=8, shape_bps=2e7, load=0.0)

    test_ids = [0, 1, 2]
    telemetry_file = tmp_path / "cloud_telemetry.jsonl"

    with cloud_process(bundle_dir, telemetry_out=telemetry_file) as port:
        recs = real_run(
            bundle,
            port,
            cfg,
            prices,
            energy,
            ws.x[test_ids],
            ws.window_id[test_ids],
            selector=Static(0),
            cloud_telemetry=telemetry_file,
        )

    assert len(recs) == 3
    assert telemetry_file.exists()
    for r in recs:
        assert "measured_cloud_dec_ms" in r
        assert "measured_cloud_head_ms" in r
        assert r["measured_cloud_dec_ms"] >= 0.0
        assert r["measured_cloud_head_ms"] >= 0.0
        assert r["cloud_ms"] >= 0.0
