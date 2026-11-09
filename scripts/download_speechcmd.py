"""Download Speech Commands v0.02 into transfer/speechcmd/data/raw and extract it. Resumable.

Source: https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz (Warden, "Speech
Commands: A Dataset for Limited-Vocabulary Speech Recognition", arXiv:1804.03209, 2018).
Licence: Creative Commons Attribution 4.0 International; cite the dataset, never commit the data.
That host is a CNAME to the GCS bucket "download.tensorflow.org" and presents a certificate for
*.storage.googleapis.com only, so the same object is fetched by its path-style GCS URL.
No official checksum is published: the archive is checked against the MD5 GCS stores for the
object (x-goog-hash, an integrity check, not a signature) and its SHA-256 is printed for the record.
The archive is untrusted: tarfile's "data" filter refuses "..", devices and links leaving the
target; `_filter` also refuses every link and absolute path, so only plain files and dirs land.
"""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from pathlib import Path

import typer

URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
MD5 = "6b74f3901214cb2c2934e98196829835"  # GCS x-goog-hash md5=a3TzkBIUyywpNOmBloKYNQ==
RAW = Path(__file__).resolve().parents[1] / "transfer" / "speechcmd" / "data" / "raw"
CURL = ["curl", "-sSL", "--proto", "=https", "--retry", "8", "--retry-delay", "5"]


def _digests(path: Path) -> tuple[str, str]:
    """(md5, sha256) hex digests in one pass."""
    md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def _filter(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
    if member.issym() or member.islnk() or member.name.startswith("/"):
        raise tarfile.FilterError(f"link or absolute path in archive: {member.name}")
    return tarfile.data_filter(member, dest)


def main() -> None:
    """Fetch the archive (resuming a partial one), check its MD5, print its SHA-256, extract."""
    RAW.mkdir(parents=True, exist_ok=True)
    tgz = RAW / "speech_commands_v0.02.tar.gz"
    if not tgz.exists():
        part = tgz.with_suffix(".part")
        subprocess.run([*CURL, "-C", "-", "-o", str(part), URL], check=True)
        md5, _ = _digests(part)
        if md5 != MD5:  # a corrupt partial cannot be fixed by resuming: start over
            part.unlink()
            raise typer.BadParameter(f"md5 {md5} != {MD5}, partial deleted: re-run")
        part.rename(tgz)
    md5, sha = _digests(tgz)
    typer.echo(f"{tgz.name}: {tgz.stat().st_size} bytes, md5 {md5}, sha256 {sha}")
    done = RAW / ".extracted"
    if not done.exists():
        with tarfile.open(tgz) as tar:
            tar.extractall(RAW, filter=_filter)
        done.touch()
    typer.echo(f"extracted into {RAW}")


if __name__ == "__main__":
    typer.run(main)
