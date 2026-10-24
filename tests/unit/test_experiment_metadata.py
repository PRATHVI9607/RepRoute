from types import SimpleNamespace

from kai.bench import experiments
from kai.bench.experiments import _bundle_provenance


def test_bundle_provenance_records_bundle_and_profile_hashes() -> None:
    bundle = SimpleNamespace(
        manifest=SimpleNamespace(
            bundle_id="bb809ab9f5c8",
            files={"profiles.json": "a" * 64, "manifest.json": "b" * 64},
            git_sha="deadbeef",
        )
    )

    assert _bundle_provenance(bundle) == {  # type: ignore[arg-type]
        "bundle_id": "bb809ab9f5c8",
        "bundle_git_sha": "deadbeef",
        "profiles_sha256": "a" * 64,
    }


def test_h7_summary_records_the_selector() -> None:
    rows = [{"p50_rel_gap": 0.02, "selection_agreement": 1.0}]

    summary = experiments._h7_summary("loopback", rows, selector="s1_cascade")

    assert summary["selector"] == "s1_cascade"
    assert summary["H7_pass"] is True
