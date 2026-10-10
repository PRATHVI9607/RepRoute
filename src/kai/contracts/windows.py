"""WindowSet: the dataset interface every role consumes (PRD §14.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

CLASS_NAMES = ("healthy", "inner", "outer")


@dataclass(frozen=True)
class WindowSet:
    """Non-overlapping int16 windows with the metadata needed for leakage-safe splits."""

    x: npt.NDArray[np.int16]  # [n, N]
    window_id: npt.NDArray[np.int64]  # [n], unique
    bearing_id: npt.NDArray[np.str_]  # [n], physical bearing
    condition: npt.NDArray[np.str_]  # [n], operating condition code
    label: npt.NDArray[np.int64]  # [n], index into CLASS_NAMES
    sample_rate_hz: float
    dataset_id: str

    def __post_init__(self) -> None:
        n = len(self.x)
        for name in ("window_id", "bearing_id", "condition", "label"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, x has {n}")
        if self.x.dtype != np.int16 or self.x.ndim != 2:
            raise ValueError(f"x must be int16 [n, N], got {self.x.dtype} {self.x.shape}")
        if len(np.unique(self.window_id)) != n:
            raise ValueError("window_id must be unique")

    def __len__(self) -> int:
        return len(self.x)

    @property
    def window_n(self) -> int:
        return int(self.x.shape[1])

    def take(self, mask: npt.NDArray[np.bool_] | npt.NDArray[np.intp]) -> WindowSet:
        return WindowSet(
            self.x[mask],
            self.window_id[mask],
            self.bearing_id[mask],
            self.condition[mask],
            self.label[mask],
            self.sample_rate_hz,
            self.dataset_id,
        )

    def bearing_labels(self) -> dict[str, int]:
        """bearing_id → class index. Raises if a bearing carries two labels."""
        out: dict[str, int] = {}
        for b, y in zip(self.bearing_id.tolist(), self.label.tolist(), strict=True):
            if out.setdefault(b, y) != y:
                raise ValueError(f"bearing {b} has more than one label")
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            x=self.x,
            window_id=self.window_id,
            bearing_id=self.bearing_id,
            condition=self.condition,
            label=self.label,
            sample_rate_hz=self.sample_rate_hz,
            dataset_id=self.dataset_id,
        )

    @staticmethod
    def load(path: Path) -> WindowSet:
        with np.load(path, allow_pickle=False) as z:
            return WindowSet(
                z["x"],
                z["window_id"],
                z["bearing_id"],
                z["condition"],
                z["label"],
                float(z["sample_rate_hz"]),
                str(z["dataset_id"]),
            )
