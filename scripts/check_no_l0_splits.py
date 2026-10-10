"""Ban random-window (L0) splits (PRD §8.3): fail on any splitter that ignores bearings."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# splitters that shuffle windows without regard to the physical bearing they came from
BANNED = re.compile(
    r"\b(train_test_split|ShuffleSplit|StratifiedShuffleSplit|KFold|StratifiedKFold|RepeatedKFold"
    r"|random_split)\s*\("
)
ALLOWED = re.compile(r"\b(GroupKFold|StratifiedGroupKFold|GroupShuffleSplit|LeaveOneGroupOut)\s*\(")


def main() -> int:
    bad = []
    for path in [
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("notebooks/**/*.py"),
    ]:
        if path.name == Path(__file__).name:
            continue
        for no, line in enumerate(path.read_text().splitlines(), 1):
            if BANNED.search(line) and not ALLOWED.search(line):
                bad.append(f"{path.relative_to(ROOT)}:{no}: {line.strip()}")
    if bad:
        print("random-window (L0) split utilities are forbidden; split by bearing instead:")
        print("\n".join(bad))
        return 1
    print("no L0 splits found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
