# Novelty log

Format (PRD §19): date · query · new hits · overlap with C1–C6 (none / partial / high) · action.

| Date | Query | New hits | Overlap | Action |
|---|---|---|---|---|
| 2026-10-10 | (carried over from PRD §3.3; no new sweep was run in the implementation pass) | Kalør & Ohtsuki 2025 (arXiv 2506.11391); Huang et al. 2025 (arXiv 2510.17543) | partial (C1) | Cited; B11 adapts the first as a baseline |
| 2026-10-14 | The seven PRD §19 strings, 31 follow-up queries, 5 arXiv API listings, cited-by of arXiv 2506.11391 and 2510.17543 (full list in the 2026-10-14 section below) | 24 works added to `lit-matrix.csv` (ids 13–36); closest is AdaEdge (ICDE 2024, doi 10.1109/ICDE60146.2024.00124) | no `high` (nothing found meets §3.4 a + b + c together); partial on C1 (18 works), C2 (2), C3 (3), C5 (1), C6 (4); none on C4 | Cite AdaEdge and add it as an uncertified-selector baseline; correct the §3.3 item 11 / matrix row 11 wording (that paper does choose an encoder/decoder pair); reword the §3.3 "defensible gap" sentence; no pivot |

The first sweep was run on 2026-10-14 (detail below). The next ones are due weekly; hard
checkpoints are W12, T−14 days and T−2 days before submission.

`lit-matrix.csv` is transcribed from PRD §3.3. Entries 11–12 were checked against their arXiv
abstracts by the PRD authors on 2026-10-10; nothing in it was re-verified in the implementation pass.

## 2026-10-14 sweep (detail for the table row above)

**Verdict.** Nothing found meets all three §3.4 kill criteria together, so there is no `high` hit and
no pivot is triggered. Two findings still require re-wording of claims. (1) AdaEdge (ICDE 2024) is
missing from PRD §3.3 and already does per-segment selection over a heterogeneous menu of 6 lossless
and 5 lossy time-series codecs on an edge device under bandwidth, storage and throughput limits, and
it scores a choice by agreement between the model's predictions on decompressed and on raw data. It
meets criterion (b) and most of (c); it does not meet (a): the paper calls itself best-effort, picks
by per-arm bandit averages, and offers no per-window certificate, safe set or risk bound. So neither
"selection over a heterogeneous representation menu" nor "agreement with the raw-data model as the
target" can be presented as new on their own; what remains for C1 is the per-window sufficiency
certificate and the any-selector statement of Proposition 1. (2) PRD §3.3 item 11 and matrix row 11
say Kalør & Ohtsuki select "models not payloads". Their system model has a set of encoder/decoder
pairs, and the experiments use WebP at four quality settings; the pair is chosen a priori from
channel statistics (not per sample), and the CRC loss is a missed true label in a prediction set
(not agreement with a raw model). The differentiation holds but must be re-stated on those grounds.
Beyond that: conformal guarantees on a transmitted payload exist for a single family (rows 20–23),
and the "bound one quantity, optimise the other" pattern exists for model cascades (rows 25–28), so
C1 is a combination of known parts applied to a payload menu. C3 has partial neighbours (ZipMate's
throughput model, Ng et al.'s closed-form queueing cross-overs); C6 has single-family neighbours
(ProgressFTX, Huang & Zhou, ARTOVeQ, Zhao et al.); C5 overlaps partly with AdaEdge's study of lossy
codecs against model agreement on UCR/UCI; nothing was found for C4 or for the Paderborn dataset in
an offloading setting, but those two were searched with one or two queries each.

**Screened.** 38 web queries (about 350 result links, with duplicates) + 5 arXiv API listings (94 of
99 entries read; one listing was truncated at 41 of 46) + Semantic Scholar and OpenAlex cited-by
lookups for both tracked papers. 37 papers opened; 2 were already in the matrix, 24 added, 11 rated
none.

**Queries.** PRD §19: "representation selection" edge offloading · "task-oriented" compression
conformal · "conformal" edge inference payload · "heterogeneous representation" semantic offloading ·
"sufficiency" certificate sensor data · "adaptive bitrate" machine perception · Paderborn bearing
edge cloud offloading adaptive compression representation. Follow-ups: conformal risk control ×
{semantic communication, task-oriented edge inference, split computing, early exit, split /
collaborative inference, remote inference, IoT selective transmission, video analytics
configuration, feature / image compression / token pruning / quantization} · adaptive compression
level selection per-sample with "risk control" OR "conformal" · codec selection / representation
switching per-sample with accuracy guarantee · "what to send" raw vs features vs latent · lossy
compression "downstream task" guarantee · goal-oriented communication "statistical guarantee" rate
selection · semantic "mode selection" bit/semantic switching · "Learn then Test" OR "conformal risk
control" routing / cascade · early exit "risk control" · progressive transmission
uncertainty-triggered refinement · break-even bandwidth "compress or not" closed-form · "phase
diagram" OR "regime map" OR "break-even" offloading · edge-cloud time-series compression benchmark
"trace-driven" · AdaEdge, VergeDB and follow-ups · time-series compressor selection under a
downstream ML accuracy constraint · bearing vibration edge-cloud adaptive compression (three
phrasings, two naming Paderborn / CWRU) · "conformal" bearing fault diagnosis / predictive
maintenance · title searches for both tracked papers. arXiv API: abs:conformal AND (offloading OR
"split computing" OR "edge inference" OR "semantic communication") · (conformal OR "risk control" OR
"learn then test") AND (compression OR codec OR bitrate OR "rate-distortion") AND (edge OR cloud OR
task OR downstream) · (bearing OR vibration OR "predictive maintenance" OR "condition monitoring")
AND (offloading OR "edge-cloud") AND (compression OR representation OR semantic OR bandwidth) ·
all:conformal AND ("edge AI" OR "edge-cloud" OR "remote inference" OR "task-oriented" OR
"goal-oriented" OR "split inference") · ("representation selection" OR "codec selection" OR
"adaptive compression" OR "task-aware compression") AND (edge OR IoT OR offloading).

**Cited-by.** arXiv 2506.11391: Semantic Scholar 0 entries, OpenAlex count 0. arXiv 2510.17543:
Semantic Scholar 1 entry (arXiv 2604.14251), OpenAlex count 0; web search also surfaced arXiv
2512.20012 and a monograph chapter (arXiv 2605.10351) from the same group.

**Works with partial overlap** (none rated `high`; evidence is in `lit-matrix.csv`):

| Id | Work | URL | Touches |
|---|---|---|---|
| 13 | AdaEdge, ICDE 2024 | https://par.nsf.gov/biblio/10613188 | C1, C2, C5 |
| 14 | VergeDB, CIDR 2021 | https://vldb.org/cidrdb/2021/vergedb-a-database-for-iot-analytics-on-edge-devices.html | C1 |
| 15 | ZipMate, HotEdge 2020 | https://www.usenix.org/system/files/hotedge20_paper_lu.pdf | C2, C3 |
| 16 | Huang & Zhou 2020 | https://arxiv.org/abs/2005.12235 | C1, C6 |
| 17 | Dynamic-Deep 2021 | https://arxiv.org/abs/2106.00606 | C1 |
| 18 | CD-CRC 2024 | https://arxiv.org/abs/2409.07902 | C1 |
| 19 | Conformal Lyapunov Optimization 2025 | https://arxiv.org/abs/2503.00486 | C1 |
| 20 | Conformal sparsification for edge-cloud speculative decoding 2025 | https://arxiv.org/abs/2510.09942 | C1 |
| 21 | Online conformal compression 2025 | https://arxiv.org/abs/2503.08340 | C1 |
| 22 | Prediction-powered communication 2025 | https://arxiv.org/abs/2509.24373 | C1 |
| 23 | Weighted conformal semantic communication, ICML 2026 workshop | https://icml.cc/virtual/2026/78447 | C1 |
| 24 | OCP-PLS 2025 | https://arxiv.org/abs/2503.14453 | C1 (audit / ACI loop) |
| 25 | Edge-cloud-expert cascades 2025 | https://arxiv.org/abs/2512.20012 | C1 |
| 26 | Calibrate-Then-Delegate 2026 | https://arxiv.org/abs/2604.14251 | C1 |
| 27 | Fast yet Safe, NeurIPS 2024 | https://arxiv.org/abs/2405.20915 | C1 |
| 28 | Conformal Arbitrage 2025 | https://arxiv.org/abs/2506.00911 | C1 |
| 29 | IB-MHT 2024 | https://arxiv.org/abs/2409.07325 | C1 |
| 30 | COOPERTRIM, ICLR 2026 | https://arxiv.org/abs/2602.13287 | C1 |
| 31 | MuSeCo 2024 | https://arxiv.org/abs/2405.08949 | C1 |
| 32 | ProgressFTX 2021 | https://arxiv.org/abs/2112.07244 | C6 |
| 33 | Zhao et al. 2026 | https://arxiv.org/abs/2601.14942 | C6 |
| 34 | ARTOVeQ 2025 | https://arxiv.org/abs/2501.02521 | C6 |
| 35 | Ng et al. 2025 | https://arxiv.org/abs/2504.15162 | C3 |
| 36 | Sagduyu & Erpek 2026 | https://arxiv.org/abs/2605.05514 | C3 |

**Opened and rated none** (not added to the matrix): arXiv 2507.06040 (EdgeCodec), 2604.05045
(PCA-Triage), 2503.13246 (SHRINK analytics), 2512.09319, 2604.23577 (RouteNLP), 2609.26625,
2609.01082, 2608.14991, 2606.22496, 2505.17604, 2601.18765.

**Not checked / limits.** Abstracts were read through a fetch tool that returns a close paraphrase
with short quotes, not the verbatim text; only AdaEdge (14 pages) and ZipMate (5 pages) were read as
PDFs, and Kalør & Ohtsuki was queried on its HTML full text (first 100k of 130k characters). All
other ratings are abstract-level. Google Scholar, IEEE Xplore and ACM DL were not searched directly;
Semantic Scholar keyword search returned HTTP 429. The cited-by indexes used may lag. IEEE Xplore
document 11431889 ("Reliable Real-Time Edge AI via Conformal Model Selection") surfaced as a possible
published version of arXiv 2506.11391 but could not be opened. Cardiff ORCA 157633 (QoS-aware choice
among autoencoders for compressive offloading) returned HTTP 403 and is not in the matrix. Cited-by
of Qu, Yang and Girgis (also named in §19) was not run. Whether any cascade paper (rows 25–28)
states an any-selector property like Proposition 1 was not checked. Matrix rows 1–10 were not
re-verified.
