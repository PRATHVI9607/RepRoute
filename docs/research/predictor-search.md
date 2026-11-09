# Sufficiency-predictor search — every candidate tried

Generated from `experiments/model_search/predictor-search.jsonl` (script: `scripts/predictor_search.py`, 2026-11-01).
Selection data: the `ptrain` partition of fold r0f0 only, 5-fold CV grouped by unit; `calib` and `test` were not read.
Cells: ROC-AUC of ĝ_r against d_r (mean over non-raw representations) / mean log-loss / AUC of the hardness target any_r d_r.
d_r is disagreement with the raw model, not label error. Adoption rule: a clear win on ≥ 3 of 4 datasets (fixed before the run), with "clear" taken as ≥ +0.02 AUC and no worse log-loss.

| Candidate | HAR (6 units) | Paderborn (5 units) | Speech Commands (498 units) | Sleep-EDF (15 units) |
|---|---|---|---|---|
| `mlp32_current` | 0.744 / 0.435 / 0.881 | 0.450 / 0.666 / 0.291 | 0.643 / 0.487 / 0.786 | 0.731 / 0.421 / 0.893 |
| `logistic_current` | 0.743 / 0.444 / 0.785 | 0.435 / 0.606 / 0.662 | 0.651 / 0.464 / 0.776 | 0.672 / 0.457 / 0.793 |
| `mlp64` | 0.750 / 0.443 / 0.861 | 0.449 / 0.663 / 0.364 | 0.636 / 0.505 / 0.771 | 0.731 / 0.424 / 0.899 |
| `mlp64x32` | 0.729 / 0.637 / 0.867 | 0.450 / 1.276 / 0.338 | 0.605 / 0.769 / 0.708 | 0.719 / 0.455 / 0.912 |
| `mlp32_a1e-1` | 0.744 / 0.431 / 0.877 | 0.444 / 0.645 / 0.319 | 0.651 / 0.473 / 0.795 | 0.728 / 0.416 / 0.885 |
| `mlp32_a1` | 0.743 / 0.429 / 0.855 | 0.427 / 0.556 / 0.566 | 0.663 / 0.459 / 0.793 | 0.702 / 0.428 / 0.856 |
| `mlp32_log` | 0.752 / 0.427 / 0.882 | 0.439 / 0.686 / 0.361 | 0.638 / 0.483 / 0.792 | 0.728 / 0.419 / 0.900 |
| `mlp32_rankgauss` | 0.756 / 0.470 / 0.879 | 0.476 / 0.725 / 0.542 | 0.646 / 0.484 / 0.787 | 0.723 / 0.430 / 0.894 |
| `hgb` | 0.791 / 0.411 / 0.899 | 0.434 / 0.558 / 0.509 | 0.645 / 0.468 / 0.798 | 0.711 / 0.430 / 0.865 |
| `shared64` | 0.745 / 0.437 / 0.864 | 0.435 / 0.703 / 0.352 | 0.650 / 0.479 / 0.783 | 0.734 / 0.415 / 0.900 |
| `shared128x64` | 0.753 / 0.537 / 0.861 | 0.454 / 1.233 / 0.275 | 0.616 / 0.703 / 0.730 | 0.739 / 0.435 / 0.902 |
| `mlp32_extra` | 0.707 / 0.682 / 0.810 | 0.452 / 0.759 / 0.471 | 0.649 / 0.530 / 0.803 | 0.748 / 0.403 / 0.911 |
| `logistic_extra` | 0.728 / 0.464 / 0.798 | 0.439 / 0.684 / 0.609 | 0.664 / 0.458 / 0.802 | 0.697 / 0.437 / 0.848 |
| `hgb_extra` | 0.787 / 0.413 / 0.896 | 0.433 / 0.546 / 0.571 | 0.669 / 0.453 / 0.842 | 0.732 / 0.407 / 0.895 |
| `shared64_extra` | 0.696 / 0.630 / 0.795 | 0.423 / 0.882 / 0.430 | 0.661 / 0.489 / 0.807 | 0.757 / 0.394 / 0.916 |
| `mlp32_cum_cheaper` | 0.640 / 1.420 / 0.881 | 0.422 / 1.840 / 0.291 | 0.595 / 1.795 / 0.786 | 0.711 / 1.240 / 0.893 |

**Verdict: keep the current predictor (`mlp32_current`).** The best candidate meets the rule on 2 of 4 datasets (`hgb_extra`). On Paderborn every candidate ranks below chance on the held-out bearing (AUC < 0.5): what the probe features learn on four bearings is inverted on the fifth, which a different model cannot fix. One seed, no intervals; HAR and Paderborn have 6 and 5 units and are noisy.
