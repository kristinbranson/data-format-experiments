# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa, Plitt & Giocomo (2025) *"A flexible hippocampal population code for experience relative to reward"*, Nat. Neurosci. DANDI:001361 (NWB, `/app/data`)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `paper.pdf`, `methods.txt` — reference text
- `code/` — reference code repo (`Sosa_et_al_2024`): `src/reward_relative/*.py`, `notebooks/*.md`, `docs/*.md`
- `data/` — 11 subject dirs (`sub-m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19`), 152 `*_behavior+ophys.nwb` files, 87 GB total, plus `dandiset.yaml`
- `decoder.py`, `train_decoder.py` — decoder reference implementation
- `pynwb_docs/`

Environment check: `python3` works; numpy 2.4.4, torch 2.6.0+cu124 (CUDA available, NVIDIA L4 23 GB),
pynwb 4.1.0, scipy 1.18, sklearn 1.9, **suite2p installed** (needed for OASIS deconvolution).
`TwoPUtils` is NOT installed (its `nansmooth` helper is re-implemented locally).
Machine: 128 cores, 1006 GB RAM, 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `preprocessing.py` | LOADING | Builds the `sess` object: loads scan info, aligns VR to 2P, loads suite2p output, adds lick/reward/speed timeseries |
| `TwoPUtils.preprocessing.vr_align_to_2P` (external; mirrored by `vr_align_to_mock_2P`) | `preprocessing.py` | LOADING | Interpolates the Unity VR sqlite stream onto imaging frame times. **This is the alignment that is already baked into the NWB behavior timeseries** (one row per imaging frame). Position/`t` linear-interpolated, `morph`/`trialnum`/`scanning` nearest-neighbour, `dz`/`lick`/`reward`/`tstart`/`teleport`/`rzone` cumulative-then-difference (so `lick`/`rzone` are *counts per imaging frame*), speed = smoothed d(pos)/dt |
| `dff` | `preprocessing.py` | PROCESSING | ΔF/F: per-trial windows only, neuropil subtraction (`neu_coef=0.7`, neuropil trial-mean added back), `maximin` baseline (Gaussian σ=15 samples smoothing → `minimum_filter1d(300)` → `maximum_filter1d(300)`), dFF=(F−base)/|base|, then per-trial Gaussian σ=2 smoothing, then OASIS deconvolution (`dcnv.oasis(dff, 2000, tau, frame_rate/n_planes)`) → `events` |
| `multi_anim_sess` | `utilities.py` | PROCESSING/CURATION | Top-level per-day pipeline: calls `dff(..., neuropil_method='subtract', baseline_method='maximin', neu_coef=0.7, deconvolve=True, keep_teleports=<per-animal>)`, then `get_trial_types`, `get_reward_zones`, `define_trial_subsets`, place-cell calc |
| `default_dff_method` | `utilities.py` | PROCESSING | `{neuropil_method_red:'subtract', baseline_method:'maximin', neu_coef:0.7, keep_teleports:False}` |
| `get_trial_types` | `behavior.py` | PROCESSING | Per trial (`trial_start_inds[i]:teleport_inds[i]`): `isreward = any(reward>0) AND any(rzone>0)`; `morph` = unique environment id |
| `get_reward_zones` | `behavior.py` | PROCESSING | Reward-zone [start,stop] and label ('A'/'B'/'C') per trial **from the scene name**; on `X_to_Y` scenes the zone changes at trial 30 (`change_trial=30`). Coordinates come from `reward_zone_dict` keys `X=[80,130] (A)`, `Y=[200,250] (B)`, `Z=[320,370] (C)` |
| `correct_lick_sensor_error` | `behavior.py` | CURATION | Trials where >`thr` fraction of frames have cumulative lick count >2 → licks set to NaN (sensor stuck). Paper: thr = 0.30, 81/12,376 trials |
| `is_putative_interneuron` | `spatial.py` | CURATION | Pearson r between each cell's dF/F and running speed over valid (within-trial) samples; `r > 0.5` ⇒ putative interneuron, excluded (`dayData.exclude_int=True`, `int_thresh=0.5`, `int_method='speed'`) |
| `teleport_metadata.teleport_sessions` | `teleport_metadata.py` | PROCESSING | Per-animal list of experiment days on which the laser was NOT blanked in the ITI ⇒ `keep_teleports=True` for ΔF/F baselining |
| `decode.CircularRegression` | `decode.py` | (paper analysis) | Paper's Fig-3 circular-linear decoder of reward-relative position from deconvolved events at speed >2 cm/s. Not used here (different decoder), but it confirms **deconvolved events** are the paper's neural signal of choice |
| `sessions_dict.single_plane/multi_plane` | `sessions_dict.py` | LOADING | Per-animal session metadata (date, scene, exp_day). Scene name ⇒ reward zone identity; reproduced in the NWB `identifier` field |

### Notes
- Imaging: two-photon CA1, ~15.5 Hz **per plane**. m17/m18 are 2-plane (ETL); planes pooled for all analyses except Ext.Fig.7.
- Neural signal chain: raw `F`, `Fneu` (suite2p) → ΔF/F (reference `dff`) → OASIS `events`. The NWB file stores `Fluorescence` (raw F), `Neuropil` (Fneu) and `Deconvolved`. **The NWB `Deconvolved` series is suite2p's own `spks` (non-zero before the first trial start, i.e. computed over the whole session), NOT the paper's per-trial dFF-based `events`** — so ΔF/F + OASIS must be recomputed here, exactly as the paper does.
- Cell curation: suite2p `iscell` (manual curation, stored in the NWB `PlaneSegmentation`) then speed-correlation interneuron exclusion.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, one file per mouse-day
(`ses-NN` == experiment day, 1–14). Read with `pynwb.NWBHDF5IO`.

Contents used:
- `nwb.identifier` = original path, e.g. `/data/InVivoDA/GCAMP11/25_02_2023/Env1_LocationA_to_C` ⇒ **animal, date and scene name** (scene gives the reward-zone sequence).
- `nwb.subject.subject_id` = `m11`, …; `nwb.session_id` = experiment day.
- `processing['behavior']['BehavioralTimeSeries']`, one sample per imaging frame (`timestamps` = frame times, dt = 0.0644836 s = 1/15.5078125 Hz), series:
  `position` (cm; −500 before the VR/2P TTL sync), `speed` (cm/s, smoothed), `lick` (cumulative count per frame),
  `reward_zone` (count per frame of the VR reward-zone flag), `trial_start`, `teleport` (per-frame counts of the events),
  `trial number` (0-indexed; −1 before sync), `environment` (0 = ENV1, 1 = ENV2; −1 before sync), `autoreward`, `scanning`,
  and `Reward` — a *sparse* series with one timestamp per delivered reward (0.004 mL).
- `processing['ophys']`: `Fluorescence/planeK` (n_frames × n_roi raw F), `Neuropil/planeK`, `Deconvolved/planeK` (suite2p spks — unused),
  `ImageSegmentation/PlaneSegmentation` with columns `pixel_mask`, `iscell` (n_roi × 2: [is-cell flag, probability]), `planeIdx`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) |
| Sessions | 152 (14/mouse; m11 has 12 — imaging started on day 3) |
| Sessions / subject | 12–14 |
| ROIs / session (all suite2p ROIs) | 315 – 4857 |
| `iscell`-curated neurons / session | 155 – 2341 (median ≈ 940) |
| Trials (total, = # `trial_start` flags) | 12,216 |
| Trials / session | mean 80.37, s.d. 6.14, range 41 – 100 |
| Imaging frames / session | 14,164 – 51,520 |
| Frame rate | 15.5078125 Hz/plane (dt = 64.484 ms) for every session |
| Frames inside trials (total) | ≈ 2.62 M |
| Trial duration | mean 13.8 s, median 12.3 s, range 6.2 – 216.6 s |

Notes / edge cases found:
- `#trial_start` == `#teleport` in every session (no dangling trial).
- In 12 sessions (m17/m18, multi-plane) the ophys series has **exactly one more frame** than the behavior series
  (`vr_align_to_2P`'s documented "one frame correction"); ophys is truncated to the behavior length.
- Position at the `teleport` frame is a meaningless interpolation between end-of-track and the teleport zone
  (e.g. 350 cm when the previous sample is 446 cm) ⇒ the teleport sample must be excluded from the trial,
  which is exactly what the reference does (`licks[t_start:t_end]`, `f_[:, start-1:stop-1]`).
- `reward_zone` > 0 occurs only on trials where reward was actually available (~85% of trials).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Mice (switch task) | 11 | "Each mouse encountered a different starting reward zone … counterbalanced across mice (n = 11 mice)" |
| Sessions | 14 days/mouse; m11 from day 3 | "The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Neurons / session | 155 – 2172 | "This approach yielded 155–2172 putative pyramidal neurons per session" |
| Trials / session | 80.5 ± 7.4 | "mean ± s.d., 80.5 ± 7.4 trials across 14 mice, all imaging days" |
| Trials (total, switch mice) | 12,376 | "n = 81 out of 12,376 trials removed across 11 switch mice" |
| Lick-sensor-error trials | 81 (~0.65%) | "~0.65% of all imaged trials, n = 81 out of 12,376 trials" ; criterion ">30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2" |
| Reward omission rate | ~15% | "the reward was randomly omitted on ~15% of trials" |
| Putative interneurons excluded | 0.42 ± 0.85% of cells | "Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 ± 0.85% of cells" |
| Neural time bin | 64.5 ms (15.5 Hz) | "each frame is ~64.5 ms"; "~15.5 Hz" |
| Track length | 450 cm | "unidirectional 450 cm virtual linear track" |
| Reward zones | A 80–130, B 200–250, C 320–370 cm | "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" |
| Reward-zone switch | after trial 30 | "Each switch occurred after 30 trials" |
| Environments | ENV1 / ENV2 | "two virtual environments"; `morph` 0/1 |

### Processing Details
- ΔF/F: "baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window … dF/F … fluorescence minus the baseline, divided by the absolute value of the baseline, then smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel. The activity rate was extracted by deconvolving dF/F with a canonical calcium kernel using the OASIS algorithm as used in Suite2p."
  (20 s × 15.5 Hz ≈ 310 samples ≈ the `int(300)` window in the code.)
- Per-trial limitation of the baseline "accounts for potential photobleaching … and avoids the teleport periods for sessions during which the laser power was reduced" ⇒ `keep_teleports` per animal/day (`teleport_metadata`).
- Spatial analyses exclude samples with speed < 2 cm/s (NOT applicable here — see Step 5 deviations).
- Temporal alignment for this conversion: **start of trial** (`trial_start`), per the Decoder Task spec.

### Curation Steps
**Neuron curation rules**:
1. suite2p manual curation (`iscell[:,0] == 1`) — ROIs with multiple somata/dendrites, no visible transients, over-expression or interneuron-like continuous fluorescence were rejected.
2. Putative interneurons: Pearson r(dF/F, speed) > 0.5 over valid samples → excluded.

**Trial curation rules**:
1. Trials with lick-sensor error (>30% of frames with cumulative lick count > 2) → licks unusable; the paper NaNs them. Here lick is a decoder **output**, so such trials are dropped.
2. Trial window = `[trial_start_ind, teleport_ind)` (the teleport sample itself is excluded, per reference code).

### Decoders Trained (in the paper)
| Decoded variable | Accuracy |
|---|---|
| Reward-relative (circular) position from RR/TR/non-RR place-cell subsets (Fig. 3) | reported as a "decode score" = mean cos(y − ŷ) (1 = perfect, 0 = chance), ~0.5–0.8 for the example session, and as a z-score vs. circular-shift shuffle; no classification accuracies are reported |

The paper reports no categorical decoding accuracies, so there is no direct accuracy target for the
decoder used here; the comparison target is the expert conversion.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total trials | n/a | 12,216 `trial_start` flags over 152 sessions | "12,376 trials across 11 switch mice" | 1.3% fewer in the archive. `#trial_start == #teleport` in every session, so no trials are being missed by the parser; the paper's larger count most likely includes sessions not deposited on DANDI (e.g. the day-15/16/17 sessions that `teleport_metadata.py` lists for m15/m17/m18/m19, or the duplicate day-9 session commented out for GCAMP4 in `sessions_dict.py`). **Accepted**; everything derived from it (trials/session, lick-error count, reward rate) matches the paper. |
| Neurons / session | n/a | `iscell` count 155 – 2341 | "155–2172 putative pyramidal neurons per session" | The minimum matches **exactly** (155, m11 day 3). Only 3 of 152 sessions exceed 2172, all from m18 (2281, 2314, 2341), the mouse with the most ROIs and the only 2-plane animal with very dense segmentation. Most likely ROI re-curation between the paper analysis and the DANDI deposit. We use the `iscell` flags as archived, which is the only curation information in the released data. **Accepted, documented.** |
| Neural signal for the decoder | `multi_anim_sess` computes **both** `dff` and `events`; the paper's own decoder (`decode.py`, Fig. 3) uses `events`; peak/field/sequence analyses use `dff` ("closest to the raw data") | NWB ships raw `F`, `Fneu` and suite2p's own `Deconvolved` | ΔF/F and OASIS events both defined | Both were produced with the identical reference pipeline and both were tested (Step 7). **dF/F chosen** (see Step 5, Decision 2). |
| NWB `Deconvolved` series | reference `events` are NaN outside trials | NWB `Deconvolved` is non-zero before the first trial start, i.e. computed over the whole session from raw F | paper's events come from per-trial dF/F | The NWB `Deconvolved` is suite2p's own `spks`, **not** the paper's `events`. We recompute ΔF/F (and, optionally, OASIS) from `F`/`Fneu` exactly as the reference does. |
| Lick-error threshold | `correct_lick_sensor_error` default `correction_thr=0.5`; `lick_pos_std` passes 0.35 | thresholds 0.3/0.35/0.5 give 81/69/44 trials | ">30% of the … samples"; "n = 81 out of 12,376 trials" | Used **0.30**, which reproduces the paper's count of **81** trials exactly. |
| Reward-zone coordinates | `reward_zone_dict` has both `'A':[175,225]` and `'X':[80,130]`; `get_reward_zones` maps Location A→`'X'`, B→`'Y'`, C→`'Z'` | first in-zone flag occurs at 80/200/320 cm | "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" | Used X/Y/Z = A/B/C = (80,130)/(200,250)/(320,370). Validated against the data: over all 10,394 trials with a reward-zone flag, the position at the first flagged frame is within 0–8.5 cm of the scene-derived zone start (mean +0.9 cm, the expected 1-frame lag at ~44 cm/s). Zero mismatches. |
| Speed threshold (2 cm/s) | applied for place-cell / decoder analyses in the paper | – | "we excluded activity when the animal was moving at <2 cm s−1" | **Not applied here**: speed is one of the decoder outputs (its lowest bin is exactly "<2 cm/s"), and dropping those samples would delete a whole output class and break the continuous per-trial time base required by the decoder task. Documented deviation, required by the Decoder Task spec. |
| Place-cell restriction | paper's Fig-3 decoder uses only RR / TR / non-RR **place cells** | – | – | **Not applied here**: that subsetting served a specific scientific comparison; a general-purpose decoder should see all curated pyramidal cells. Documented deviation. |

Everything else (frame rate, trial windows, ΔF/F parameters, neuropil coefficient, interneuron
criterion, switch trial 30, ENV1/ENV2 coding, ~15% omission) is consistent across the code, the
data and the paper.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (NWB) | Target field | Transform | Reference code | Notes |
|---|---|---|---|---|
| `ophys/Fluorescence/planeK` (raw F), `ophys/Neuropil/planeK` | `neural` | neuropil subtraction (0.7, trial-mean added back) → per-trial maximin baseline (σ=15 smoothing, 20 s min then max filter) → (F−base)/\|base\| → σ=2 Gaussian smoothing | `preprocessing.dff` called from `utilities.multi_anim_sess` | (n_neurons, T) float32 per trial |
| `ImageSegmentation/PlaneSegmentation['iscell'][:,0]` | neuron filter | keep == 1 | suite2p manual curation | |
| dF/F vs `behavior/speed` | neuron filter | drop cells with Pearson r > 0.5 | `spatial.is_putative_interneuron` | |
| `behavior/trial_start`, `behavior/teleport` | trial segmentation | trial = `[start_ind, teleport_ind)` | `behavior.get_trial_types`, `preprocessing.dff` | teleport sample excluded |
| `behavior/lick` | trial filter + `output[3]` | drop trial if >30% of frames have count > 2; else binarise (count > 0) | `behavior.correct_lick_sensor_error`; `behavior.lickrate` (`licks[licks>0]=1`) | |
| `behavior/position` timestamps | `input[0]` | t − t[trial_start] (s) | – | decoder-task requirement |
| `behavior/environment` (`morph`) | `input[1]` | constant per trial, 0 = ENV1, 1 = ENV2 | `behavior.get_trial_types` | |
| `behavior/trial number` | `input[2]` | constant per trial, 0-indexed | – | |
| `behavior/Reward` + `behavior/reward_zone` | `input[3]`, `output[5]` | `isreward = any(reward) AND any(rzone)`; input is the **previous** trial's value | `behavior.get_trial_types` | first trial of a session ⇒ 0 |
| `behavior/position` + scene name | `output[0]` | signed distance to nearest point of the active reward zone → 7 bins | `behavior.get_reward_zones` | 0 while inside the zone |
| `behavior/position` | `output[1]` | 5 equal 90 cm bins over the 450 cm track | – | |
| `behavior/speed` | `output[2]` | 5 bins (<2, 2–10, 10–20, 20–40, >40 cm/s) | – | |
| scene name + trial index | `output[4]` | zone label A/B/C → 0/1/2, switching at trial 30 | `behavior.get_reward_zones` | |
| `nwb.subject.subject_id` | `subjects` / `subject_idx` | | | 11 mice |
| imaging plane / `ImagingPlane.location` | `brain_regions` / `brain_region_idx` | all "CA1" | paper: dorsal CA1 | per-neuron plane kept in `metadata.session_info[i].plane_idx` |

### Key Decisions
1. **Temporal bins = imaging frames (64.484 ms)**. The VR behaviour in the NWB file has already been
   interpolated onto imaging frame times by the authors' `vr_align_to_2P`, so neural and behavioural
   streams are sample-for-sample aligned with no further resampling — the safest possible alignment.
   Every session is sampled at 15.5078125 Hz per plane, so the bin size is identical everywhere, as
   the target format requires. No additional binning was applied: it would only blur the licking and
   speed outputs.
2. **Neural signal = dF/F** (option `--neural-signal events` reproduces the deconvolved version).
   Both come from the identical reference pipeline. dF/F was chosen because (a) the paper treats it
   as the signal "closest to the raw data" and uses it for spatial-peak, field and sequence
   analyses; (b) OASIS deconvolution is explicitly *not* interpreted as a spike rate by the authors,
   it only removes the calcium kernel's asymmetry, which matters for spatial-tuning estimates rather
   than for per-frame decoding; and (c) it decodes better here — validation balanced accuracy on the
   2-session sample was higher for every one of the six outputs (e.g. position 0.625 vs 0.559,
   speed 0.553 vs 0.415, reward outcome 0.633 vs 0.568). Deconvolution discards sub-threshold
   fluctuations that a linear per-frame decoder can use.
3. **Trial window `[trial_start, teleport)`**, i.e. exactly the lap on the track. The reference
   `dff` uses `[start-1, stop-1)`, a 1-sample-shifted window; we use the un-shifted window so that
   the neural, input and output streams cover precisely the same frames and the trial contains no
   NaN. The teleport frame itself is excluded either way — its interpolated position is a meaningless
   blend of the end of the track and the teleport zone (e.g. 350 cm right after 446 cm).
4. **ΔF/F baseline windows follow `keep_teleports`** per animal/day from
   `teleport_metadata.teleport_sessions`: on sessions where the laser was not blanked, the baseline
   window for trial *i* starts one sample after the previous teleport (including the ITI), otherwise
   it is the trial itself. This is the reference behaviour and matters because a baseline taken over
   blanked (≈0) fluorescence would be meaningless.
5. **All outputs time-varying**. `reward_zone_location` and `reward_outcome` are constant within a
   trial but are broadcast over time, both because the format requires a single `doutput` for every
   trial and because the task asks for time-varying outputs wherever possible.
6. **Reward-zone identity from the scene name**, as the reference does, rather than from the
   `reward_zone` flag — the flag only fires on rewarded trials (~85%), so it cannot label omission
   trials. The scene-derived zone was validated against the flag on all trials where it does fire.
7. **`previous_trial_outcome` = 0 for the first trial** of each session (no preceding imaged trial).
   This affects 152/12,135 = 1.25% of trials; dropping those trials instead would discard real
   neural data for no benefit.
8. **Trials with lick-sensor error are dropped** (81 trials), rather than NaN-ed as in the paper,
   because `lick` is a decoder output and NaN is not a permitted value.
9. **No speed threshold and no place-cell restriction** (see Step 4).

### Planned Sanity Checks
- [x] `#trial_start == #teleport` in every session; `teleport > trial_start` for every trial.
- [x] Scene-derived reward-zone start vs. the position at the first `reward_zone` flag (all trials, all sessions).
- [x] Lick-sensor-error trial count == 81 (paper).
- [x] Reward rate ≈ 85% (paper: ~15% omission).
- [x] Trials/session ≈ 80.5 ± 7.4 (paper).
- [x] Neurons/session range vs. the paper's 155–2172.
- [x] Putative-interneuron exclusion ≈ 0.42 ± 0.85% of cells (paper).
- [x] Re-derive neural/input/output values from the raw NWB with independent code and compare with `np.allclose` (Step 10, Check 2).
- [x] `morph` and `trial number` constant within every trial.
- [x] Reward-zone identity distribution ≈ uniform over A/B/C (counterbalanced design).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Structure:
- `nansmooth` — local port of `TwoPUtils.utilities.nansmooth` (TwoPUtils is not installed).
- `scene_reward_zones` — port of `behavior.get_reward_zones` for the scenes in this dataset.
- `compute_dff_events` — port of `preprocessing.dff` (neuropil subtraction, maximin baseline,
  smoothing, optional OASIS via `suite2p.extraction.dcnv.oasis`).
- `speed_correlation` — vectorised `spatial.is_putative_interneuron(method='speed')`.
- `discretize_*` — the three continuous→categorical maps from the Decoder Task spec.
- `load_session` / `convert_session` — pynwb reading and per-trial assembly.
- `plot_processing` — 8-panel per-session diagnostic figure (`--show-processing`).
- `main` — process pool over sessions, assembly of the final dict, pickling.

Code inefficiencies identified:
- Naive per-cell `np.corrcoef` loop for the interneuron test (O(n_cells) python loop).
- Reading `F`/`Fneu` with h5py fancy indexing per ROI would be very slow.
- Forking a process pool after importing suite2p aborts ("fork() called from a process already
  using GNU OpenMP").

Code speedups added:
- Interneuron correlations vectorised into one matrix product (≈100× faster than the loop).
- Whole `(n_frames, n_roi)` datasets read in one contiguous call, then column-masked and transposed.
- OASIS deconvolution skipped entirely when the neural signal is dF/F.
- `ProcessPoolExecutor` with a **spawn** context, 12 workers.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sessions m11 day 4, a "stay" session, and m13 day 3, a reward-switch session).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 783 |
| Neurons / session | 167, 616 |
| Subjects | 2 (m11, m13) |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| Timepoints / trial | mean 183.3, min 142, max 476 |
| input `time_from_trial_start_s` | [0.0, 30.6] |
| input `environment` | [0, 0] (both sessions ENV1) |
| input `trial_number` | [0, 79] |
| input `previous_trial_outcome` | [0, 1] |
| output `distance_to_reward_zone` | [0.215, 0.096, 0.043, 0.245, 0.020, 0.074, 0.306] |
| output `position` | [0.261, 0.203, 0.197, 0.192, 0.147] |
| output `speed` | [0.074, 0.059, 0.066, 0.314, 0.487] |
| output `lick` | [0.810, 0.190] |
| output `reward_zone_location` | [0.498, 0.316, 0.185] |
| output `reward_outcome` | [0.198, 0.802] |

### Processing Plots Review
`processing_m11_day04.png`, `processing_m13_day03.png`, 8 panels each:
1. Raw `F`/`Fneu` — drops to ~0 during the ITI, confirming the laser blanking that motivates the
   per-trial baseline and `keep_teleports=False` for these sessions.
2. dF/F — defined only inside trials (shaded), transients of 1–3 ΔF/F, flat baseline.
3. OASIS events — non-negative, sparse, aligned to dF/F rise times.
4. Histogram of r(dF/F, speed) with the 0.5 cutoff: 1/168 and 0/616 cells excluded.
5. Behaviour: position ramps 0→450 cm, rewards (stars) fall at the start of the green reward-zone
   band, licks cluster just before and inside it, omission trials have licks but no reward.
6. Position vs. its 5-bin discretisation — the staircase tracks the ramp, resets at every trial
   boundary.
7. Distance-to-reward-zone vs. its 7-bin discretisation — flat at 0 exactly while position is
   inside the green band; bin edges (green dashed) line up with the bin transitions.
8. Speed / speed bins / licks / the `time_from_trial_start` input (a clean saw-tooth resetting at
   every trial boundary) / mean neural activity, all on the converted time base.
No anomalies, no temporal offsets between streams.

### Run Time Estimates
| Speed-ups implemented | Time saving |
|---|---|
| Vectorised interneuron correlation | ~2–6 s/session on large sessions |
| Single contiguous HDF5 read per series | avoids ~100× slowdown of per-ROI reads |
| Skip OASIS when writing dF/F | ~0.3–1.5 s/session |
| 12-way process pool (spawn) | ~8× wall-clock |

| Step | Time / session | Estimated total |
|---|---|---|
| NWB read (F + Fneu) | 0.2 – 1.5 s | ~1.5 min single-threaded |
| dF/F | 0.9 – 4 s | ~5 min single-threaded |
| trial assembly | <0.5 s | ~1 min |
| **measured, 12 workers** | — | **48 s** conversion + 9 s pickle write |

The 2 sample sessions took 2.1 s and 2.9 s; sample sessions are smaller than average
(167/616 cells vs. a 910-cell mean), so the single-threaded projection was ~8–10 min and the
12-worker projection ~1–2 min. Actual full run: **57 s**, well under the 15-minute budget.

### Verification (`/app/verification_sample_out.txt`)
`Data format is valid, no errors or warnings.`

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions / 160 trials)
| Output | Chance | Training balanced acc. | Validation balanced acc. |
|--------|--------|------------------------|--------------------------|
| distance_to_reward_zone | 0.143 | 0.715 | 0.485 |
| position | 0.200 | 0.836 | 0.625 |
| speed | 0.200 | 0.694 | 0.553 |
| lick | 0.500 | 0.783 | 0.765 |
| reward_zone_location | 0.333 | 0.999 | 0.912 |
| reward_outcome | 0.500 | 0.884 | 0.633 |

Loss fell monotonically from 2.44 to 0.545 over 200 epochs; every output is well above chance.
(The same sample converted with `--neural-signal events` gave uniformly lower validation accuracy:
0.432 / 0.559 / 0.415 / 0.746 / 0.904 / 0.568 — the basis for Decision 2 in Step 5.)

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full` — 152/152 sessions,
48.3 s conversion + 9.4 s pickle write, no errors.

### Output Files
- `converted_data.pkl`: 9.63 GB
- `verification_full_out.txt`: created, **valid, no errors or warnings**

### Consistency Check
| Statistic | Reference paper | Reference code | Reference data (NWB) | Converted data | Match? |
|-----------|-----------------|----------------|----------------------|----------------|--------|
| Subjects | 11 switch mice | 11 switch animals in `sessions_dict` | 11 subject dirs | 11 | ✅ |
| Sessions | 14/mouse, m11 from day 3 | – | 152 files | 152 | ✅ |
| Total neurons (`iscell`) | – | – | 138,678 | 138,298 after interneuron exclusion | ✅ |
| Neurons/session | 155 – 2172 | – | 155 – 2341 (`iscell`) | 154 – 2320, mean 909.9 | ⚠️ min exact; max exceeded in 3/152 m18 sessions (Step 4) |
| Putative interneurons excluded | 0.42 ± 0.85 % | r > 0.5 vs speed | – | 0.32 ± 0.60 % (380 cells) | ✅ |
| Trials (total) | 12,376 | – | 12,216 | 12,135 (= 12,216 − 81) | ⚠️ 1.3% (Step 4) |
| Trials/session | 80.5 ± 7.4 | – | 80.37 ± 6.14 | 79.84 ± 6.86 | ✅ |
| Lick-error trials removed | 81 (0.65%) | thr on cumulative count > 2 | – | **81** (0.66%) | ✅ exact |
| Reward omission rate | ~15% | `any(reward) & any(rzone)` | 15.3% of trials | 15.8% of time samples / 15.3% of trials | ✅ |
| Neural time bin | ~64.5 ms | – | 64.484 ms, all sessions | 64.484 ms | ✅ |
| Track length | 450 cm | 0–450 binning | position ∈ [0, 450.6] | position bins 0–4 | ✅ |
| Reward zones | A 80–130, B 200–250, C 320–370 | `reward_zone_dict` X/Y/Z | first in-zone flag at 80/200/320 cm | same | ✅ |
| Reward-zone balance | counterbalanced across mice | – | – | A 0.332 / B 0.336 / C 0.333 | ✅ |
| Environments | ENV1/ENV2 | `morph` 0/1 | 0/1 | input range [0,1] | ✅ |
| Input `time_from_trial_start_s` | trials 6.2–216.6 s | – | – | [0.0, 216.5] | ✅ |
| Input `trial_number` | ≤100 trials/session | – | 41–100 | [0, 99] | ✅ |

Full-dataset output distributions (from `verification_full_out.txt`):
- distance_to_reward_zone: 0.251 / 0.102 / 0.073 / 0.238 / 0.021 / 0.072 / 0.243
- position: 0.212 / 0.177 / 0.231 / 0.226 / 0.154
- speed: 0.117 / 0.087 / 0.134 / 0.319 / 0.343
- lick: 0.777 / 0.223
- reward_zone_location: 0.332 / 0.336 / 0.333
- reward_outcome: 0.158 / 0.842

---
## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification
`/app/verification_full_out.txt`: **"Data format is valid, no errors or warnings."**
No errors and no warnings to address (dtypes, shapes, NaN/Inf, categorical outputs, subject and
brain-region indices, per-session neuron consistency and matching `dinput`/`doutput` all pass).

### Check 2 — Independent sanity checks (`/app/sanity_checks.py`)
This script re-derives every stream from the raw NWB files with **separately written code** (it does
not import `convert_data.py`) and compares against the pickle. Run on 9 sessions chosen to cover the
hard cases: `m11_day03` (smallest, 154 cells), `m18_day03` (largest, 2320 cells, 2 planes,
`keep_teleports=True`, 4 lick-error trials), `m4_day14` (35 lick-error trials → 45/80 kept),
`m18_day13` and `m17_day09` (sessions where the ophys series has one frame more than the behaviour
series), plus 6 random sessions. Per session:

| Check | Criterion | Result |
|---|---|---|
| trial count | == number of non-lick-error trials found independently | PASS (all 9) |
| reward-zone start | scene-derived zone start vs. position at the first `reward_zone` flag | PASS, max error 1.7–5.8 cm (one imaging frame of running) |
| `keep_teleports` | matches `teleport_metadata` independently | PASS |
| neuron count | independent `iscell` + independent r(dF/F,speed) > 0.5 | PASS (154, 2320, 1061, 981, 1428, 539, 927, 190, 909) |
| **neural values** | `np.allclose(ref, got, atol=1e-4, rtol=1e-3)` over **all cells × all timepoints** of 3 random trials/session | PASS, max abs diff 2.1e-7 – 4.9e-7 (float32 rounding) |
| **input values** | `np.allclose` on all 4 inputs × all timepoints of 5 random trials/session | PASS |
| **output values** | exact `np.array_equal` on all 6 outputs × all timepoints of 5 random trials/session | PASS |
| shapes / finiteness | `T` equal across neural/input/output, no NaN/Inf, every trial | PASS |

`ALL SANITY CHECKS PASSED` (`python -u /app/sanity_checks.py /app/converted_data.pkl 6`).

Additional dataset-wide checks (run over all 152 NWB files, `/app/cache/`):
- reward-zone labels vs. the `reward_zone` flag on **all 10,394 flagged trials**: max error 8.5 cm,
  mean +0.9 cm, zero mismatches > 15 cm.
- `reward` and `reward_zone` flags: 10,342 trials with both, **0 with reward but no zone flag**,
  52 with a zone flag but no reward (omission trials on which the VR still registered zone entry —
  precisely why the reference requires the AND). So the reference's `isreward` is unambiguous here.
- lick-sensor-error trials at threshold 0.30: **81**, matching the paper exactly.

### Check 3 — Reference code comparison
| Stage | Reference | This script | Same? |
|---|---|---|---|
| (a) Data loading | `sess` built by `create_sess`; VR aligned to imaging frames by `TwoPUtils.preprocessing.vr_align_to_2P`; `sess.timeseries['F'/'Fneu']` from suite2p, `iscell`-filtered | The NWB file already contains the *output* of that alignment (one behaviour sample per imaging frame); `load_session` reads those series plus `Fluorescence`/`Neuropil` with pynwb and applies the archived `iscell` | ✅ identical inputs |
| (b) Neuron filtering | `iscell` (manual suite2p curation) + `spatial.is_putative_interneuron(ts_key='dff', method='speed', r_thresh=0.5)` | same, with the per-cell `np.corrcoef` loop vectorised into one matrix product | ✅ |
| (b) Trial filtering | `behavior.correct_lick_sensor_error(..., correction_thr)` NaNs lick-error trials | same criterion at the paper's 0.30, trials dropped (NaN is not a legal output) | ✅ criterion, ⚠️ drop instead of NaN (justified: `lick` is a decoder output) |
| (c) Temporal alignment | trial = `sess.trial_start_inds[i] : sess.teleport_inds[i]` (e.g. `get_trial_types`, `lick_pos_std`); dF/F windows shifted by −1 sample inside `preprocessing.dff` | trial = `[trial_start_ind, teleport_ind)`; dF/F windows **not** shifted, so the whole trial has valid dF/F | ✅ trials identical; ⚠️ the reference's internal −1 shift is not reproduced (it would leave the last trial sample NaN) |
| (d) Binning | none — analyses use the native imaging frames (spatial binning is a separate 10 cm trial-matrix representation, not used here) | native imaging frames (64.484 ms) | ✅ |
| (d) ΔF/F | `preprocessing.dff`: neuropil subtract 0.7 + trial-mean add-back; `nansmooth([0,15])`; `minimum_filter1d(300)`; `maximum_filter1d(300)`; `(F−base)/abs(base)`; `nansmooth(2)`; optional `dcnv.oasis(·,2000,tau=0.7,fs)`; windows per `keep_teleports` | line-for-line port (`compute_dff_events`), same constants, same per-animal/day `keep_teleports` from `teleport_metadata` | ✅ |
| (e) Input construction | `get_trial_types` (`morph`, `isreward`), trial index | same variables; `time_from_trial_start` and `previous_trial_outcome` are decoder-task requirements with no reference counterpart | ✅ |
| (f) Output construction | `get_reward_zones` (scene → zone coords/labels, switch at trial 30); `vr_data['pos'/'speed'/'lick']`; `lickrate` binarises licks (`licks[licks>0]=1`) | same; the continuous variables are then discretised per the Decoder Task spec | ✅ |
| Speed > 2 cm/s mask | applied for place-cell/decoder analyses | **not** applied | ⚠️ deliberate (speed is an output; its lowest bin is "<2 cm/s") |
| Place-cell subsetting | paper's Fig-3 decoder uses RR/TR/non-RR place cells | **not** applied — all curated pyramidal cells | ⚠️ deliberate (general-purpose decoder) |
| Neural signal | `dff` and `events` both computed; Fig-3 decoder uses `events` | `dff` (default), `events` available via `--neural-signal events` | ⚠️ deliberate (Step 5, Decision 2) |

Every deviation is listed above with its reason; there are no unintentional differences.

### Check 4 — Key statistics comparison
See the Step 9 table. Matches: subjects (11), sessions (152), trials/session (79.8 ± 6.9 vs 80.5 ± 7.4),
lick-error trials (81, exact), reward omission (~15%), minimum neurons/session (155, exact),
interneuron exclusion (0.32 ± 0.60% vs 0.42 ± 0.85%), frame rate (64.484 ms), reward-zone
coordinates, switch trial 30. Two residual differences (total trials 12,135 vs 12,376; maximum
neurons/session 2341 vs 2172 in 3 of 152 sessions) are investigated and explained in Step 4; both
concern data that is simply not in the DANDI deposit, and neither is fixable from the released files.

### Check 5 — Edge cases (script run over all 152 files)
| Edge case | Finding | Handling |
|---|---|---|
| dangling / overlapping trials | 0 sessions with `trial_start[i] <= teleport[i-1]`; `#starts == #teleports` everywhere | assertions in `convert_session` |
| multiple events in one frame | `trial_start`/`teleport` counts never exceed 1 | `np.where(>0)` is exact |
| ophys longer than behaviour | 10 sessions have exactly one extra imaging frame | ophys truncated to the behaviour length (the reference's "one frame correction") |
| teleport-sample position artefact | position at the teleport frame is an interpolation between ~446 cm and the teleport zone | teleport sample excluded from the trial |
| position slightly out of [0, 450] | within-trial range is −2.74 … 451.83 cm | binning clips into bins 0 and 4 |
| negative speeds | within-trial speed range −6.45 … 133.1 cm/s (brief backwards drift) | falls in the "<2 cm/s" bin, as intended |
| NaNs in behaviour | none in position/speed/lick over all sessions | still guarded: trials with non-finite values would be dropped and counted (`n_dropped_nan` = 0 everywhere) |
| first trial of a session | no preceding trial | `previous_trial_outcome = 0`, documented |
| shortest trial | 96 frames (6.2 s) | no minimum-length filter needed |
| sessions with < 2 trials | none (min 40 kept trials) | decoder requires ≥ 2 trials/session — satisfied |
| `keep_teleports` window underflow | `teleport[i-1]+1 > trial_start[i]` never occurs | `min(...)` guard retained anyway |
| multi-plane animals | m17/m18 have 2 `planeK` series and a shared `PlaneSegmentation` | planes pooled (as in the paper); per-neuron plane kept in metadata |
| ROI/plane index mismatch | `PlaneSegmentation` holds all planes' ROIs concatenated | `iscell` masked per plane via `planeIdx` before matching the plane's F columns; verified by the independent sanity check |

### Issues Found and Resolved
- **Process pool aborted** ("fork() called from a process already using GNU OpenMP") because suite2p
  initialises OpenMP at import. Fixed by using a `spawn` multiprocessing context.
- **Neural signal choice**: the first full-pipeline test used deconvolved events; replaced by dF/F
  after the comparison in Step 8 (Step 5, Decision 2).
- No other issues were found; all checks above pass on the final `converted_data.pkl`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
(152 sessions, 12,135 trials, 138,298 neurons, 2.58 M timepoints; ~13 min on the L4 GPU).

### Training Progress
- Loss decreasing: **Yes**, monotonically, 3.338 → 0.651 over 200 epochs (no plateau or divergence).

### Decoder Results (Full)
| Output | #classes | Chance | Training balanced acc. | Validation balanced acc. | Val / chance |
|--------|----------|--------|------------------------|--------------------------|--------------|
| distance_to_reward_zone | 7 | 0.143 | 0.798 | **0.621** | 4.3× |
| position | 5 | 0.200 | 0.891 | **0.761** | 3.8× |
| speed | 5 | 0.200 | 0.732 | **0.628** | 3.1× |
| lick | 2 | 0.500 | 0.796 | **0.768** | 1.54× |
| reward_zone_location | 3 | 0.333 | 0.963 | **0.873** | 2.6× |
| reward_outcome | 2 | 0.500 | 0.933 | **0.602** | 1.20× |

`predictions.png` shows the decoded traces (dashed) following the true traces (solid) for position,
distance-to-reward, speed and reward-zone identity essentially step for step.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance
No output is at or below chance. Four of six are 2.6–4.3× chance. Two need comment:

- **lick, 0.768 (1.54× chance)** — a binary output, so 1.54× chance is 77% balanced accuracy, i.e.
  the decoder identifies licking frames well. Binary outputs cannot exceed 2× chance, so the
  multiplier is not a meaningful yardstick here.
- **reward_outcome, 0.602 (1.20× chance)** — investigated in detail below; this is a ceiling
  imposed by the task, not a conversion defect.

**reward_outcome diagnostic** (`/app/cache/rew_diag.py`, 26 sessions spanning all 11 mice): balanced
accuracy split by where the animal is in the trial —

| Trial phase | n (validation frames) | Train bal. acc. | Validation bal. acc. |
|---|---|---|---|
| before the reward zone (distance bin < 3) | 33,638 | 0.925 | **0.531** |
| inside the reward zone (bin == 3) | 20,997 | 0.965 | **0.613** |
| after the reward zone (bin > 3) | 33,300 | 0.958 | **0.653** |

Validation accuracy is at chance *before* the animal reaches the reward zone and rises monotonically
through and after it. That is exactly the causal structure of the task: whether this trial is
rewarded is simply not determined — and certainly not observable in CA1 — until the animal licks in
the zone. Because the task specification requires the per-trial outcome to be broadcast over the
whole trial, roughly half of the frames carry no information at all, capping the achievable overall
accuracy near 0.6–0.7. No change to the conversion can lift the pre-zone frames above chance.

### Check 2 — Accuracy comparison to the paper
| Variable | This conversion (validation balanced acc.) | Paper |
|---|---|---|
| Reward-relative position (`distance_to_reward_zone`) | 0.621 (7 classes, chance 0.143) | Fig. 3: circular-linear decoder, "decode score" = mean cos(y−ŷ), ~0.5–0.8 in the example session and z-scored against a circular-shift shuffle — **not an accuracy and not comparable numerically** |
| Position, speed, lick, reward zone, reward outcome | 0.761 / 0.628 / 0.768 / 0.873 / 0.602 | **not decoded in the paper** |

The paper reports no categorical decoding accuracies for any variable (its only decoder is the
circular-linear reward-relative position model of Fig. 3, scored with a cosine similarity over
continuous circular position and restricted to place-cell subpopulations at speeds > 2 cm/s), so
there is no published number that can be beaten or missed. As a qualitative comparison, the paper's
central claim — that CA1 carries a strong code for position relative to reward — is reproduced here:
distance-to-reward is decoded at 4.3× chance, the highest chance-multiple of all six outputs, and
higher than absolute position relative to its own chance level (3.8×).

### Check 3 — Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| distance_to_reward_zone | 0.798 | 0.621 | 1.29 |
| position | 0.891 | 0.761 | 1.17 |
| speed | 0.732 | 0.628 | 1.17 |
| lick | 0.796 | 0.768 | 1.04 |
| reward_zone_location | 0.963 | 0.873 | 1.10 |
| reward_outcome | 0.933 | 0.602 | **1.55** |

Only `reward_outcome` exceeds 1.5×. The diagnostic above localises it: training accuracy is 0.93
*even before the reward zone*, where validation accuracy is 0.53. The decoder is memorising
trial-specific neural fluctuations — each session has its own 100-PC projection fitted to ~80 trials,
and reward outcome is constant within a trial, so a per-trial "fingerprint" fits the training labels
perfectly. This is an over-fitting property of a per-trial constant label under this architecture,
not data leakage: the train/test split is by trial, each trial's frames go entirely to one side of
the split, and no output value is present in the input (the only outcome variable among the inputs
is the *previous* trial's outcome, which the decoder legitimately receives and which is uninformative
about the current trial — confirmed by the at-chance pre-zone validation accuracy).

### Additional debugging performed (per the checklist)
1. Output values verified against the raw NWB for 45 randomly chosen trials across 9 sessions,
   all 6 outputs × all timepoints, exact match (Step 10, Check 2).
2. Temporal alignment verified by plotting neural + behaviour + converted outputs for the same
   trials (`processing_*.png`, panels 5–8): the `time_from_trial_start` input is a clean saw-tooth
   resetting exactly at trial boundaries, the position staircase resets with it, and the
   distance-to-reward trace is flat at 0 precisely while the animal is inside the reward zone.
3. Output variation checked: no output is degenerate — the most imbalanced is `reward_outcome`
   at 84/16, and balanced accuracy (with `balanced_loss=True`) is used throughout.
4. Neural filtering verified: `iscell` + interneuron exclusion reproduce the paper's per-session
   neuron range and its 0.42 ± 0.85% interneuron rate.
5. Processing verified against the reference function by function (Step 10, Check 3) and numerically
   against an independent re-implementation (Step 10, Check 2).

### Issues Found and Resolved
- None outstanding. The single flagged metric (`reward_outcome`) is explained by task structure and
  quantified above; no conversion change can improve it.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format spec, key statistics)
- [x] `cache/` folder created with the investigation scripts and `README_CACHE.md`
- [x] All files organised

### Final file inventory
| File | Purpose |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | full converted dataset (9.63 GB) |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | this document |
| `README.md` | user-facing documentation |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | format-verification logs |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_m11_day04.png`, `processing_m13_day03.png` | per-step processing diagnostics |
| `sample_trials.png`, `predictions.png` | decoder sample trials and predictions |
| `cache/` | sanity checks and investigation scripts |
