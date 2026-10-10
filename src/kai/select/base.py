"""Selector interface and the degradation ladder (PRD §5.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from kai.contracts.schemas import Flag
from kai.select.cost import F64, Bool, OverheadKind, State

Mission = Literal["fidelity_first", "liveness_first"]
NO_REP = -1


@dataclass(frozen=True)
class Window:
    """What the edge knows about the current window before encoding anything."""

    safe: Bool  # [K] safe set S_λ̂(x); element 0 (raw) is always True
    ghat: F64  # [K] sufficiency-predictor scores
    phi: F64  # [P] probe features


@dataclass(frozen=True)
class Truth:
    """Information no deployable policy has. Only the oracle may read it."""

    d: Bool  # [K] realised disagreement of every representation
    state: State  # the true link state rather than the estimate


@dataclass(frozen=True)
class Decision:
    rep: int  # index into the menu, or NO_REP
    flag: Flag


class Selector(Protocol):
    name: str

    @property
    def overhead(self) -> OverheadKind: ...  # which per-window work the policy needs

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision: ...


def ladder(allowed: Bool, feasible: Bool, cost: F64, mission: Mission) -> Decision:
    """First rule that yields an action wins.

    1. a representation that is allowed (in the safe set) and feasible → cheapest, `certified`
    2. none, fidelity_first → cache raw and retry later (`deferred`); the executor turns this into
       `dropped` when its store-and-forward buffer is full
    3. none, liveness_first → cheapest deadline-feasible representation outside the safe set
       (`uncertified`), or `dropped` when nothing can meet the deadline at all
    """
    candidates = allowed & feasible
    if candidates.any():
        return Decision(int(np.where(candidates, cost, np.inf).argmin()), "certified")
    if mission == "fidelity_first":
        return Decision(NO_REP, "deferred")
    if feasible.any():
        return Decision(int(np.where(feasible, cost, np.inf).argmin()), "uncertified")
    return Decision(NO_REP, "dropped")


def send_anyway(rep: int, allowed: Bool) -> Decision:
    """For policies with no ladder (static, context-blind): always transmit.

    The flag only says whether the certificate covers the choice; lateness is an SLO event that
    the link simulator measures separately (PRD §6.4, item 3).
    """
    return Decision(rep, "certified" if allowed[rep] else "uncertified")
