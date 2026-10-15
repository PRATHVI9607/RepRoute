# Network traces

| Name | Kind | Source | Licence |
|---|---|---|---|
| `static` | constant rate | generated | n/a |
| `markov_bursty` | two-state Markov-modulated (Gilbert–Elliott), good rate vs 1/20 of it | `kai.link.traces.markov_bursty` | n/a |
| `3g_synth`, `4g_synth`, `5g_synth` | **synthetic** log-normal AR(1) with rare near-outages | `kai.link.traces.cellular` | n/a |
| `ucc4g_static`, `ucc4g_pedestrian`, `ucc4g_car` (also `_bus`, `_train`) | **measured** 4G throughput, one sample per second, 131 logs of 6–50 minutes | Raca, Quinlan, Zahran, Sreenan, *Beyond Throughput: a 4G LTE Dataset with Channel and Context Metrics*, ACM MMSys 2018 — <https://zenodo.org/records/1219679> | CC BY 4.0 |

Every trace is rescaled so that its median equals the cell's nominal bandwidth (the "sensor-data
regime", PRD §5.5): the traces contribute their **shape over time**, not their absolute rate.

## What the measured traces are, and are not

- They are downloaded by `scripts/download_traces.py` (2.3 MB, checksum-verified) into
  `data/raw/traces/ucc4g/`, which is not committed. The licence allows redistribution with
  attribution; they are kept out of the repository only to keep data and code apart.
- The logs were recorded while a phone downloaded continuously. The column used is `DL_bitrate`:
  the throughput the **downlink** delivered. The same logs carry `UL_bitrate`, but it is about 2 %
  of the downlink figure — acknowledgements, not uplink capacity.
- Kai's link is an uplink. No measured uplink capacity is claimed: the simulator uses the measured
  variability of a real cellular link over time and scales it to the scenario's rate. Label any
  figure that uses them "measured 4G downlink throughput shape, rescaled".
- The seed picks one log of the requested mobility class per scenario cell.

The synthetic cellular traces remain in the grid for continuity with the first pass. Their medians,
spread and correlation are order-of-magnitude choices, not fits to a dataset, and they must be
labelled synthetic in every figure.

## Adding another measured trace

1. Write a one-column CSV (bits per second per sample).
2. Record its source URL, licence and sampling period in the table above.
3. Load it with `kai.link.traces.from_csv(path, step_s)`.
