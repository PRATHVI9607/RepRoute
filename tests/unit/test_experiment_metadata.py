from types import SimpleNamespace

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
