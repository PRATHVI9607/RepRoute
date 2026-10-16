# ADR 0012 — More machines: UORED-VAFCLS (vibration) and PTB-XL (ECG)

Date: 2026-10-16 · Status: accepted · Touches: `src/kai/data/uored.py`, `src/kai/data/ptbxl.py`,
`scripts/download_*.py`, `scripts/prepare_*.py`, `scripts/select_reference.py`,
`configs/dataset/{uored,ptbxl}.yaml`, `bench/experiments.py` (E1 duty cycle, E10)

## Why

The certificate is exchangeable at the level of windows, but the units that matter are machines.
Paderborn gives about 5 calibration bearings per fold and HUST one per class; a machine-level
guarantee at α = 0.05 needs about 19 calibration units (CRC's `1/(n + 1) ≤ α`). A literature
search (subagent, 2026-10-15) found **no public vibration or acoustic dataset with 15 or more
physical machines per class**. Two additions follow from that.

## Decision

| Dataset | Domain | Units | Role |
|---|---|---|---|
| **UORED-VAFCLS** (Sehri, Dumond, Bouchard, Data in Brief 2023, CC BY 4.0) | bearing vibration, 42 kHz | 20 bearings, each recorded healthy and faulty; 5 inner-race, 5 outer-race faults used | the vibration dataset with the most healthy machines (20, against 6 in Paderborn) |
| **PTB-XL** (Wagner et al., Scientific Data 2020, PhysioNet, CC BY 4.0) | 12-lead ECG, lead II, 500 Hz | 10 266 patients, one record each (NORM / MI / STTC) | many units: about 2 050 calibration patients per fold, so the certificate is a **per-patient** guarantee |

Mappings were fixed before any model was trained on them and are documented in the loaders.

- **UORED:** a bearing's healthy and faulty recordings share one `bearing_id` and always fall in
  the same partition; splits are stratified by fault type (4 strata of 5). One operating
  condition, so no condition-shift (L2) experiment.
- **PTB-XL:** the patient is the unit (`bearing_id = P<patient_id>`); diagnostic superclasses at
  likelihood 100; one record per patient; PTB-XL's own `strat_fold` is not used.
- **Reference models** are chosen by grouped inner CV on the training units of fold r0f0 only
  (`scripts/select_reference.py`, recorded in `experiments/model_search/`). UORED keeps EnvNet. For
  PTB-XL the waveform CNN wins clearly (macro-F1 0.668 against 0.549 for SpecNet and 0.470 for
  EnvNet): with thousands of units a flexible model finally generalises. Its R6 encoder is the
  generic CNN bottleneck, since the CNN has no fixed front-end to reuse.
- **NASA C-MAPSS is not added.** It has no waveforms (one value per sensor per engine cycle) and a
  remaining-life task; most of the menu does not apply. Recorded in ADR 0009's list.

## Consequences and caveats

- **UORED confound:** every inner-race bearing is one manufacturer's model (NSK 6203ZZ) and every
  outer-race bearing another's (FAFNIR 203KD). A model can separate those two classes by bearing
  model rather than by fault; healthy versus faulty is not affected. Two healthy recordings are
  unusually loud, and the ball-fault runs were unloaded (not used).
- **PTB-XL** is health, not industry. It tests whether the method generalises beyond vibration
  and is the only dataset where a unit-level certificate is meaningful.
- **Profiling duty cycle.** E1 times each call at the window period, capped at 64 ms: PTB-XL's
  8.2 s windows would make profiling take hours, and the CPU is already cold at 64 ms
  (ADR 0004 §4). Rerun on the Pi.
- **E10 (privacy)** skips an attack a dataset cannot support: the operating-condition attack with
  one condition (UORED, PTB-XL), and re-identification with one window per unit (PTB-XL).

Results: [results-uored.md](../research/results-uored.md), [results-ptbxl.md](../research/results-ptbxl.md).
