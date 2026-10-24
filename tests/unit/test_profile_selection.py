import json
from pathlib import Path

import pytest

from kai.bench.experiments import load_profiles


def _profile(t_enc: float) -> dict[str, float | str]:
    return {
        "rep_id": "r6_bottleneck",
        "size_mean_bytes": 16.0,
        "size_q95_bytes": 16.0,
        "t_enc_ms": t_enc,
        "t_enc_mad_ms": 0.0,
        "t_enc_q95_ms": t_enc,
        "t_dec_ms": 0.1,
        "t_head_ms": 0.2,
        "load_exponent": 0.0,
    }


def _overheads(t_probe: float) -> dict[str, float]:
    return {
        "t_probe_ms": t_probe,
        "t_pred_ms": 0.0,
        "t_sel_ms": 0.0,
        "load_exponent": 0.0,
        "t_io_ms": 0.05,
        "io_load_exponent": 0.0,
    }


def test_load_profiles_selects_the_duty_cycle_at_or_below_the_node_size(tmp_path: Path) -> None:
    doc = {
        "profiles": [_profile(3.0)],
        "overheads": _overheads(0.3),
        "busy": {
            "8": {"profiles": [_profile(2.0)], "overheads": _overheads(0.2)},
            "32": {"profiles": [_profile(1.0)], "overheads": _overheads(0.1)},
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(doc))

    for n_sensors, enc, probe in (
        (1, 3.0, 0.3),
        (4, 3.0, 0.3),
        (8, 2.0, 0.2),
        (16, 2.0, 0.2),
        (32, 1.0, 0.1),
        (64, 1.0, 0.1),
    ):
        profiles, overheads = load_profiles(path, n_sensors=n_sensors)
        assert profiles[0].t_enc_ms == pytest.approx(enc)
        assert overheads.t_probe_ms == pytest.approx(probe)
