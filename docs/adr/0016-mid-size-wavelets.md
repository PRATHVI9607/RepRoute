# ADR 0016 — Opt-in mid-size wavelets (exploratory)

Date: 2026-10-30 · Status: accepted (exploratory) · Touches: `contracts/representation.py`
(`REP_CODES`), `reps/registry.py`, `tasks/pipeline.py` (`FoldConfig.extra_wavelets`), `cli.py`
(`kai folds --extra-wavelets`), `bench/experiments.py` (E10 cache)

## Why

On Speech Commands (5 folds, `results-speechcmd.md`) the certified policies fail H3: when the
certificate rejects the 16-byte R6 for a window, the next certified option is near-lossless
(r2a/r2b, several KB; raw is 32 KB at 16 384 samples). Those frames are late, so Kai pays for
lateness and costs 31 % more than the best single representation. The menu has nothing between
"tiny and often wrong" and "large and right". H1 and H5 fail for the same reason.

## Decision

1. Two more wavelet codecs, `r2c_wavelet` (ε = 0.15) and `r2d_wavelet` (ε = 0.30), the same db4
   codec as r2a/r2b at a larger relative L2 error. Wire codes 11 and 12 are **appended** to
   `REP_CODES`; existing codes are never renumbered, so every existing bundle and frame keeps its
   meaning.
2. They are **opt-in** (`FoldConfig.extra_wavelets`, default `False`). The pre-registered menu
   (PRD §7) is unchanged, and `FoldConfig.recipe()` leaves the key out when it is off, so folds
   trained before this ADR still match their stored recipe.
3. They belong to the waveform-preserving family (`WAVEFORM_REPS`) for H5 and the
   compression-only ablation.
4. Any result using them is **exploratory** and reported apart from the pre-registered results
   (`results-speechcmd-x.md`, separate work directory `transfer/speechcmd_x`). The menu was
   changed after seeing the pre-registered result; that is stated wherever the numbers appear.

## Check

`tests/unit/test_extra_wavelets.py`: absent by default; sizes r2d < r2c < r2b; the menu saves and
reloads with matching contract hashes; old recipes unchanged.
