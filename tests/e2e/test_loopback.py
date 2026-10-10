"""Closed loop over real sockets (PRD G2): edge → TCP → cloud, bundle integrity, negotiation."""

from __future__ import annotations

import asyncio
import shutil

import numpy as np
import pytest

from kai.cloud.heads import HeadZoo
from kai.cloud.server import CloudServer
from kai.contracts.schemas import EnergyModel, Prices
from kai.contracts.windows import WindowSet
from kai.edge.bundle import BundleError, load_bundle
from kai.edge.runtime import EdgeConfig, NeedsBundleError, run_edge
from kai.reps.registry import ContractMismatchError
from kai.select.policies import Static
from kai.tasks.pipeline import Table

pytestmark = pytest.mark.xdist_group("fold")
PRICES = Prices(name="t", story="t", p_byte=1.0, p_j=0.0, p_cpu_ms=0.0, p_late_ms=100.0)
ENERGY = EnergyModel(p_cpu_w=1.0, e_bit_j=1e-7)


def loop(
    bundle_dir, ws: WindowSet, table: Table, cfg: EdgeConfig, n: int = 60, selector=None, zoo=None
):
    """Run cloud and edge in one event loop on an ephemeral port; return the edge records."""
    test_ids = np.flatnonzero(table.of("test"))[:n]

    async def main():
        server = CloudServer(zoo or HeadZoo(bundle_dir))
        srv = await asyncio.start_server(server.serve_client, "127.0.0.1", 0)
        port = srv.sockets[0].getsockname()[1]
        try:
            return await run_edge(
                load_bundle(bundle_dir), cfg, PRICES, ENERGY, ws.x[test_ids],
                ws.window_id[test_ids], "127.0.0.1", port, selector=selector,
            )  # fmt: skip
        finally:
            srv.close()
            await srv.wait_closed()

    return asyncio.run(main())


def test_closed_loop_decisions_match_the_disagreement_table(
    bundle_dir, fold_dir, ws: WindowSet
) -> None:
    table = Table.load(fold_dir / "table.npz")
    # every non-raw decision is audited, so the link carries a raw copy too: give it 20 Mbps
    cfg = EdgeConfig(alpha=0.10, deadline_ms=500, period_ms=8, shape_bps=2e7, load=0.0, p_audit=1.0)
    records = loop(bundle_dir, ws, table, cfg)
    assert len(records) == 60
    used = set()
    for rec in records:
        assert rec["flag"] == "certified"
        assert rec["safe"][rec["rep"]]
        # the cloud's decision for (window, representation) is exactly the lab table's entry
        assert rec["decision"] == table.yhat[rec["window_id"], rec["rep"]]
        assert 0 < rec["latency_ms"] < 500
        assert rec["bytes"] > 28
        if rec["rep"] != 0:  # audited: the cloud compared against the raw copy
            assert rec["audited"]
            assert rec["disagreed"] == int(table.d[rec["window_id"], rec["rep"]])
        used.add(rec["rep"])
    assert len(used) >= 2  # the selector really does choose per window


def test_every_representation_round_trips_through_the_wire(
    bundle_dir, fold_dir, ws: WindowSet
) -> None:
    table = Table.load(fold_dir / "table.npz")
    cfg = EdgeConfig(deadline_ms=500, period_ms=2, load=0.0)
    for k in range(len(table.rep_ids)):
        records = loop(bundle_dir, ws, table, cfg, n=12, selector=Static(k))
        assert [r["rep"] for r in records] == [k] * 12
        assert all(r["decision"] == table.yhat[r["window_id"], k] for r in records), table.rep_ids[
            k
        ]


def test_cloud_without_the_matching_head_answers_needs_bundle(
    bundle_dir, fold_dir, ws: WindowSet
) -> None:
    table = Table.load(fold_dir / "table.npz")
    zoo = HeadZoo(bundle_dir)
    wavelet = zoo.reps["r2a_wavelet"]
    del zoo.by_hash[
        bytes.fromhex(wavelet.contract_hash())
    ]  # cloud was built for another codec version
    k = table.rep_ids.index("r2a_wavelet")
    with pytest.raises(NeedsBundleError):
        loop(
            bundle_dir,
            ws,
            table,
            EdgeConfig(period_ms=2, load=0.0),
            n=5,
            selector=Static(k),
            zoo=zoo,
        )


def test_edge_refuses_a_tampered_bundle(bundle_dir, tmp_path) -> None:
    bundle = load_bundle(bundle_dir)
    assert bundle.manifest.bundle_id == bundle_dir.name
    assert set(bundle.manifest.contract_hashes) == set(bundle.manifest.rep_ids)

    bad = tmp_path / "flipped"
    shutil.copytree(bundle_dir, bad)
    onnx = bad / "menu" / "r5_ae.onnx"
    data = bytearray(onnx.read_bytes())
    data[len(data) // 2] ^= 1
    onnx.write_bytes(bytes(data))
    with pytest.raises(BundleError, match="hash mismatch"):
        load_bundle(bad)

    missing = tmp_path / "missing"
    shutil.copytree(bundle_dir, missing)
    (missing / "predictor.npz").unlink()
    with pytest.raises(BundleError, match="missing"):
        load_bundle(missing)

    lock = tmp_path / "uv.lock"
    lock.write_text("not the lockfile this bundle was built with")
    with pytest.raises(BundleError, match="lockfile"):
        load_bundle(bundle_dir, lock_file=lock)


def test_menu_loader_detects_a_codec_parameter_change(fold_dir, tmp_path) -> None:
    import json

    menu = tmp_path / "menu"
    shutil.copytree(fold_dir / "menu", menu)
    index = json.loads((menu / "reps.json").read_text())
    next(e for e in index if e["rep_id"] == "r2a_wavelet")["eps"] = 0.5  # silently re-tuned codec
    (menu / "reps.json").write_text(json.dumps(index))
    from kai.reps.registry import load_menu

    with pytest.raises(ContractMismatchError, match="r2a_wavelet"):
        load_menu(menu)
