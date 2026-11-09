# ADR 0018 — Opt-in coarse log-mel representation with a distilled CNN head (exploratory)

Date: 2026-11-01 · Status: accepted (exploratory) · Touches: `contracts/representation.py`
(`REP_CODES` 13, 14), `reps/logmel.py`, `reps/registry.py` (`LOGMEL_GRID`, `classical_menu(logmel_hz=)`),
`tasks/models.py` (`LogMelNet`), `tasks/train.py` (`train_logmel_head`), `cloud/heads.py`
(`OnnxHead`), `tasks/pipeline.py` (`FoldConfig.logmel`), `cli.py` (`kai folds --logmel`),
`bench/experiments.py` (E10 cache)

## Context

On Speech Commands, Kai costs about 25 % more than the best single representation (H3 fails,
`updates.md` §4–5). The reason is structural: every payload small enough for a slow link
disagrees with f_raw on 24–65 % of windows (r3–r7), so the certificate refuses it and the window
waits. The mid-size wavelets of ADR 0016 did not help (r2c: 914 B, 17 % disagreement). A word is
mostly its time-frequency envelope, which a coarse log-mel grid keeps in a few hundred bytes.

## Decision

- R10 encoder (edge, numpy only): `n_frames` Hann frames with 50 % overlap, power spectrum,
  `n_mels` triangular mel bands from 20 Hz to Nyquist, natural-log energy. Sent as the window's
  peak (float16) plus uint8 levels over the 80 dB below it: `n_mels × n_frames + 2` bytes, fixed.
  The mel pooling uses `einsum`, not a BLAS matmul (ADR 0017).
- R10 head (cloud): `LogMelNet`, three 3×3 conv blocks + global mean + the peak, trained on the
  fold's `train` windows to reproduce f_raw's decision (distillation, like every other head), fixed
  recipe (AdamW, one-cycle, 30 epochs, time-shift augmentation). Exported to ONNX and served by
  `OnnxHead` (ONNX Runtime; no torch in the cloud).
- Two grids, chosen by 3-fold speaker-grouped inner CV on fold r0f0's `train` speakers only
  (`experiments/model_search/speechcmd-logmel.jsonl`; `test`, `ptrain` and `calib` untouched):

  | grid (mels × frames) | bytes | inner-CV disagreement |
  |---|---|---|
  | 32 × 32 | 1 026 | 7.4 % |
  | 24 × 16 | 386 | 8.2 % |
  | **16 × 16 → r10a_logmel** | **258** | **8.1 %** |
  | **12 × 12 → r10b_logmel** | **146** | **10.1 %** |

  r10a is the smallest grid within 1 point of the best; r10b the smallest tested (an 8 × 8 run was
  interrupted and not repeated).
- Opt-in (`kai folds --logmel`): the pre-registered menu and every fold trained before this ADR are
  unchanged (`FoldConfig.recipe()` drops the key when off). Rep codes are appended (13, 14), never
  renumbered.
- Family: R10 has a fixed encoder and a fitted head, like R3 and R4, so it belongs to neither the
  waveform family (`WAVEFORM_REPS`) nor the learned-encoder family (`LEARNED`) of H5's ablations.

## Consequences

- Results with R10 are exploratory: the hypotheses were registered for the R0–R7 menu. They are
  reported next to the pre-registered numbers, never in place of them.
- The certificate needs nothing new: R10 is one more column of the disagreement table.
- Test: `tests/unit/test_logmel.py` (opt-in, deterministic, fixed size, contract hash, menu
  reload, CNN head learns and round-trips through ONNX).
