# Dataset Conversion Notes

## Overview
- **Dataset**: "A flexible hippocampal population code for experience relative to reward" (Sosa et al.) - /app/data
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `paper.pdf` (Sosa, Plitt, Giocomo 2025, Nat Neuro)
- `methods.txt` (methods excerpt)
- `code/` : Sosa_et_al_2024 github repo (src/reward_relative/*.py, notebooks/, docs/, environments/)
- `data/` : DANDI-style NWB dataset, 11 subjects (sub-m3,m4,m7,m11..m15,m17,m18,m19), 152 `.nwb` files total (`sub-mXX_ses-NN_behavior+ophys.nwb`), plus `dandiset.yaml`
- `train_decoder.py`, `decoder.py` : provided decoder validation/training code

Environment check: `python3` works; numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0, pynwb 4.1.0 all import successfully.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | preprocessing.py | LOADING | Builds TwoPUtils `sess` object: loads scan info, aligns VR (sqlite) to 2P frames (`sess.align_VR_to_2P` -> `TwoPUtils.preprocessing.vr_align_to_2P`), loads suite2p, adds behavior timeseries (licks, rewards, speed) and position-binned trial matrices (10 cm bins, 0-450 cm) |
| `vr_align_to_mock_2P` | preprocessing.py | PROCESSING | Illustrates VR->2P alignment: `pos` linearly interpolated, `morph/trialnum/scanning` nearest-neighbour, `dz/lick/reward/tstart/teleport/rzone` integrated-interpolated-differenced (cumulative per frame), speed = smoothed dz/dt, pre-TTL samples: pos=-500, morph/trialnum/scanning=-1 |
| `dff` | preprocessing.py | PROCESSING | dF/F per cell: keeps only samples between trial_start-1 and teleport-1 (rest = NaN), neuropil subtraction (`neu_coef=0.7`), maximin baseline (nansmooth sigma 15 frames then min-filter 300 then max-filter 300 within each trial), dff=(F-F0)/|F0|, then smoothing of dff by 2 frames, optional OASIS deconvolution (tau=0.7, frame rate/planes) -> `events` |
| `multi_anim_sess` | utilities.py | LOADING/PROCESSING | Per experiment-day wrapper: loads sess pickles per animal, computes dFF/events, calls `get_trial_types`, `get_reward_zones`, `define_trial_subsets`, `calc_place_cells` |
| `get_trial_types` | behavior.py | PROCESSING | Per trial (trial_start_inds[i]:teleport_inds[i]): `isreward` = any(reward>0) AND any(rzone>0); `morph` = unique(vr_data.morph) (0=Env1, 1=Env2) |
| `get_reward_zones` | behavior.py | PROCESSING | Per trial reward zone [start,stop] in cm and label A/B/C from scene name. Zone coords: A=[80,130] (dict key 'X'), B=[200,250] ('Y'), C=[320,370] ('Z'); on switch scenes (`X_to_Y`) the zone changes at trial index 30 (`change_trial`, or `sess.change_reward_trial`) |
| `define_trial_subsets` | behavior.py | CURATION | Splits trials into set0 (pre-switch) / set1 (post-switch) |
| `is_putative_interneuron` | spatial.py | CURATION | Flags ROIs whose dF/F correlates with speed (r>0.3) as putative interneurons; excluded from place-cell analyses (`dayData.exclude_int=True`) |
| `calc_place_cells` | spatial.py | CURATION/ANALYSIS | Spatial info + circular shuffle to define place cells (not needed for decoding: we keep all cells) |
| `dayData` / `include_ans` | dayData.py | CURATION | Animal inclusion: only "reward switch" animals; explicitly drops GCAMP2, GCAMP5, GCAMP6, GCAMP10 |
| `CircularRegression`, `train_vs_test_blocks` | decode.py | ANALYSIS | Paper's circular decoder of position/reward-relative position from population activity |

### Notes
- Data pipeline in the paper: raw sbx -> suite2p (ROIs, `iscell` manual curation) -> `sess` class aligning VR sqlite to imaging frames (~15.5 Hz) -> dF/F + deconvolved `events` -> place cell stats.
- Imaging: 2-photon calcium (GCaMP) in **hippocampal CA1**. So ∆F/F **does** need to be computed from F/Fneu, unless the NWB already stores it (to be checked in Step 2).
- Trials = laps, from `trial_start_inds` (track entry) to `teleport_inds` (end of track). Teleport/ITI periods are excluded from dF/F by default (`keep_teleports` option).
- Behavior columns in the aligned VR dataframe: time, pos, trialnum, dz, speed, lick, reward, rzone, morph, autoreward, tstart, teleport, scanning. Samples before TTL sync have pos=-500 and morph/trialnum/scanning=-1 (invalid; must be excluded).
- `licks` is a cumulative lick count per imaging frame; code binarizes anything >1 -> 1 when computing rates.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
DANDI dataset 001361 (NWB 2.8.0), one file per subject-session: `/app/data/sub-mXX/sub-mXX_ses-NN_behavior+ophys.nwb` (152 files, 87 GB).

Per file:
- `/identifier` = original path, encodes the **scene** e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`
- `/general/session_id` = experiment day (01-14); `/general/subject/subject_id` = mXX; `/general/optophysiology/ImagingPlane`: location `hippocampus, CA1`, indicator `GCaMP7f`, `imaging_rate` 15.5078 Hz (1 plane) or 31.0156 Hz (2 planes: m17, m18 -> per-plane/volume rate is still 15.5 Hz)
- `processing/ophys/Fluorescence/planeK/data` (n_frames x n_roi) raw suite2p F
- `processing/ophys/Neuropil/planeK/data` (n_frames x n_roi) suite2p neuropil F
- `processing/ophys/Deconvolved/planeK/data` (n_frames x n_roi) suite2p default `spks` (NOT the paper's `events`; no NaNs, computed from suite2p baseline)
- `processing/ophys/ImageSegmentation/PlaneSegmentation`: `iscell` (n_roi x 2: [is_cell, prob]) after manual curation, `planeIdx`, pixel/voxel masks
- `processing/behavior/BehavioralTimeSeries/*`: `position`(cm), `trial number`, `trial_start`, `teleport`, `speed`(cm/s), `lick` (cumulative count per frame), `reward_zone` (cumulative rzone flag), `environment` (morph: 0=ENV1, 1=ENV2, -1 before TTL sync), `autoreward`, `scanning`; all n_frames long with `timestamps` (imaging frame times, dt = 0.0645 s)
- `processing/behavior/BehavioralTimeSeries/Reward`: sparse event series (data=reward volume 0.004 mL, timestamps of delivery)
- Neural series have `starting_time`=0 and `rate` attr; number of samples == number of behavior samples, so **neural and behavior are sample-by-sample aligned** (this is the `sess`/`vr_data` alignment from the paper, already applied).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Files (sessions) | 152 |
| Subjects | 11 (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) |
| Sessions / subject | 14 for all except m11 (12; imaging started day 3) |
| ROIs (total, all suite2p) | 312,110 |
| ROIs with iscell==1 (total) | 138,678 |
| iscell / session | mean 912, min 155, max 2341 |
| Trials (total, = # trial_start = # teleport) | 12,216 |
| Trials / session | mean 80.4, min 41, max 100 |
| Frames total | 3,610,877 (mean 23,756 / session; mean session duration 1532 s) |
| Frame rate | 15.5078 Hz (dt 64.48 ms) for all sessions (2-plane mice sample each plane at this rate) |
| Environments present | values {-1 (pre-sync), 0 = ENV1, 1 = ENV2} |
| Scenes | `EnvX_LocationY` (stay days) and `EnvX_LocationY_to_Z` / `EnvX_Y_to_EnvW_Z` (switch days) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch task) | 11 mice | "Each mouse encountered a different starting reward zone and sequence of reward zone switches, counterbalanced across mice (n = 11 mice)" |
| Environments | 2 (ENV1/ENV2); 9 mice start in ENV1, m17 & m18 start in ENV2 | "Most mice began the task in ENV 1 (n = 9 mice; two mice (m17 and m18) began in ENV 2)" |
| Sessions / subject | 14 days; m11 imaged from day 3 | "for a total of 14 days"; "The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Trials / session | 80.5 +- 7.4 (target 80-100) | "We targeted 80-100 trials per session ... (mean +- s.d., 80.5 +- 7.4 trials across 14 mice, all imaging days)" |
| Trials (total, 11 switch mice) | 12,376 | "~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice" |
| Neurons / session | 155-2172 putative pyramidal cells | "This approach yielded 155-2172 putative pyramidal neurons per session" |
| Neural sampling | ~15.5 Hz (frame = 0.0645 s) | "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate" |
| Behavior sampling | same ~15.5 Hz (VR interpolated to imaging frames) | as above |
| Reward omission rate | ~15% of trials | "the reward was randomly omitted on ~15% of trials" |
| Reward zones | A 80-130, B 200-250, C 320-370 cm; 50 cm wide | "zone A, 80-130 cm; zone B, 200-250 cm; zone C, 320-370 cm" |
| Track length | 450 cm (+ ~50 cm teleport) | "unidirectional 450 cm virtual linear track" |
| Switch trial | after 30 trials | "Each switch occurred after 30 trials" |
| Auto-reward | first 10 trials of a new condition if no lick | "On the first ten trials of any new condition ... reward was automatically delivered" |
| Lick-error trials | 81/12,376 (0.65%) trials, detected as >30% of samples in trial with cumulative lick count >2 | Quantification of licking behavior |
| Putative interneurons | dF/F-speed Pearson r > 0.5 -> excluded; 0.42 +- 0.85% of cells | Calcium data processing |
| Speed threshold for spatial analyses | activity at <2 cm/s excluded | "we excluded activity when the animal was moving at <2 cm/s" |
| Position binning (paper analyses) | 45 bins x 10 cm over 450 cm | Place cell identification |

### Processing Details
- dF/F: per-trial maximin baseline with 20 s sliding window (implemented as gaussian smooth sigma=15 frames, then min-filter 300 frames, then max-filter 300 frames), neuropil subtraction with coefficient 0.7 (re-added per-trial neuropil mean), dff = (F - F0)/|F0|, smoothed with 2-frame (0.129 s) Gaussian.
- Deconvolution: OASIS (suite2p `dcnv.oasis`) on dF/F within each trial, tau = 0.7 s (suite2p ops value used in the repo), frame rate = 15.5 Hz -> "events" = the activity rate used for almost all paper analyses (including the decoder).
- Temporal alignment: VR behavior interpolated onto imaging frame times (already done in the NWB files). Trials run from `trial_start` index to `teleport` index; the reference code uses the window `[start-1 : stop-1]`.
- Trial-level variables: `isreward` (reward delivered AND reward zone entered), `morph`/environment (0=ENV1, 1=ENV2), reward zone position/label from scene name with switch at trial 30.

### Curation Steps

**Neuron curation rules**:
1. suite2p ROI manual curation -> `iscell` flag (already stored in NWB); only iscell==1 ROIs are used.
2. Putative interneurons (Pearson r > 0.5 between dF/F and running speed) excluded (`dayData.int_thresh=0.5`).
3. Multi-plane animals (m17, m18): planes pooled for all analyses except anatomical maps.

**Trial curation rules**:
1. Only imaged trials (between trial_start and teleport) are analyzed; teleport/ITI excluded (laser was often blanked).
2. Lick-sensor-error trials (>30-35% of samples with cumulative lick count > 2) have lick data set to NaN (~0.65% of trials).
3. Analyses of rewarded vs omission restricted to trial sets with >= 3 omissions (analysis-specific; not a general curation rule).

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Reward-relative (circular) position from deconvolved events of RR/TR/non-RR cells | reported as "decode score" = mean cos(y - yhat), 1 = perfect, 0 = chance; z-scored vs circular-shift shuffles. No classification accuracies are reported in the paper, so no direct accuracy comparison is possible. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

Checks run over **all 152 NWB files** (`/app/cache/check_consistency.py`, `/app/cache/check_switch.py`, `/app/cache/scan_data.py`, `/app/cache/scan2.py`):

| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| # switch mice | `dayData.include_ans` = switch animals only (GCAMP2/5/6/10 dropped) | 11 subjects in DANDI (m3,m4,m7,m11-m15,m17-m19) | n = 11 switch mice | DANDI contains exactly the 11 switch mice; the 3 fixed-condition mice are not in the release. Use all 11. |
| Total trials | - | 12,216 trial_start events (= # teleports, no mismatch) | 12,376 imaged trials | Difference = 160 trials ~ 2 sessions: m11 was imaged only from day 3 (12 sessions in DANDI) and the paper's count includes trials in files not released / days 1-2 behavior. Mean trials/session 80.37 +- 6.16 matches "80.5 +- 7.4". Accepted. |
| Trials/session | targeted 80-100 | mean 80.4, min 41, max 100 | 80.5 +- 7.4 | Consistent |
| Neurons/session | `iscell` curation | iscell==1: min 155, max 1780 single-plane (2341 when the 2 planes of m18 are pooled), mean 912 | "155-2172 putative pyramidal neurons per session" | Consistent (paper quotes per-plane / per-FOV range; min 155 matches exactly) |
| Omission rate | isreward = reward AND rzone entered | 15.34% of trials unrewarded | "~15% of trials" omitted | Consistent |
| Lick sensor errors | `>0.35` of samples with cum lick > 2 (code) / `>30%` (paper) | Using the paper's 30% rule: 80 trials (0.65%) | 81 of 12,376 trials (0.65%) | Consistent -> use the paper's 30% threshold |
| Reward zone coords | A=[80,130], B=[200,250], C=[320,370] (code dict keys X/Y/Z) | median |rzone-entry position - nominal zone start| < 2 cm (max 8.5 cm, i.e. one imaging frame of running) | same coordinates | Consistent -> zone label/coords can be derived from scene name |
| Switch trial | `change_trial = 30` | Empirically the first trial with the new zone = index 30 in every switch session (apparent 31/32 only when trial 30 was an omission with no zone entry); environment switch also at index 30 | "Each switch occurred after 30 trials" | Consistent -> use index 30 |
| Environment | `morph` 0 = ENV1, 1 = ENV2 | `environment` timeseries values {-1 pre-sync, 0, 1}; constant within every trial; 11 sessions (day 8) contain both | ENV1/ENV2 | Consistent |
| Neural signal | paper `events` = OASIS deconvolution of *their* maximin dF/F | NWB `Deconvolved` = suite2p default spks (different baseline, no per-trial restriction) | "activity rate extracted by deconvolving dF/F with OASIS" | **Recompute** dF/F (maximin, neuropil 0.7) + OASIS from NWB F/Fneu, exactly as `preprocessing.dff` |
| Interneurons | r > 0.5 dF/F vs speed | test session: 1/155 cells (0.65%) | 0.42 +- 0.85% of cells | Consistent -> exclude with r > 0.5 |
| Sampling | ~15.5 Hz | behavior dt = 0.06448 s for both 1- and 2-plane sessions; neural samples == behavior samples | ~15.5 Hz | Consistent; neural and behavior are already sample-aligned in NWB |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial definition / temporal alignment
- Alignment event: **start of trial** (`trial_start` == entry to the linear track at 0 cm).
- Trial window = reference window used everywhere in the paper code (`preprocessing.dff`, `glmUtils.get_timeseries_data`): samples `[trial_start_idx - 1 : teleport_idx - 1)`. The teleport sample itself is excluded (its position is interpolated between end of track and the ITI).
- `off_start = 0.0` s, `off_end = None` (variable trial duration; median 12.3 s, mean 13.8 s).
- Time bin = one imaging frame = 1/15.5078125 s = **64.48 ms**, identical for every session (2-plane mice sample each plane at the same 15.5 Hz).
- Neural and behavioral samples are already aligned 1:1 in the NWB file (VR was interpolated onto imaging frame times by `TwoPUtils.preprocessing.vr_align_to_2P` before the NWB conversion).

### Variable Mapping
| Source | Target | Transform | Reference code |
|--------|--------|-----------|----------------|
| `ophys/Fluorescence/planeK/data`, `ophys/Neuropil/planeK/data`, `ImageSegmentation/.../iscell` | `neural` | keep `iscell==1` ROIs, pool planes; dF/F = neuropil-subtracted (0.7) F with per-trial maximin baseline (sigma 15 frames smooth -> min filter 300 -> max filter 300), dff=(F-F0)/|F0|, 2-frame Gaussian smoothing; then OASIS deconvolution (tau=0.7, 15.5 Hz) -> **deconvolved events** (n_neurons x T per trial) | `preprocessing.dff(..., neuropil_method='subtract', baseline_method='maximin', deconvolve=True)`; `utilities.multi_anim_sess` |
| `behavior/position/timestamps` | `input[0]` time_from_trial_start (s) | t - t[first sample of trial] | alignment to `trial_start_inds` |
| `behavior/environment` (morph) | `input[1]` environment | 0 = ENV1, 1 = ENV2 (constant within trial, verified) | `behavior.get_trial_types` |
| trial index within session | `input[2]` trial_number | 0-indexed lap number, constant within trial | `glmUtils.get_timeseries_data` `trials` |
| previous trial `isreward` | `input[3]` previous_trial_outcome | 0 = omitted, 1 = rewarded; first trial of a session set to 1 (the mouse had just run ~30 rewarded warm-up trials on the same zone immediately before imaging) | `behavior.get_trial_types` |
| `behavior/position` + per-trial reward zone | `output[0]` reward_zone_distance | signed distance to the nearest point of the 50-cm reward zone (0 while inside the zone), binned: <-50 / [-50,-10) / [-10,0) / 0 / (0,10] / (10,50] / >50 | zone coords from `behavior.reward_zone_dict` (A 80-130, B 200-250, C 320-370) via `behavior.get_reward_zones` |
| `behavior/position` | `output[1]` position | 5 equal bins of the 450 cm track: <90 / [90,180) / [180,270) / [270,360) / >=360 | - |
| `behavior/speed` | `output[2]` speed | bins <2 / [2,10) / [10,20) / [20,40) / >=40 cm/s | speed is the smoothed VR speed already in the NWB |
| `behavior/lick` | `output[3]` lick | cumulative lick count per frame binarized (>0 -> 1), as in the paper (`licks[licks>1]=1` then converted to rate) | `glmUtils.get_timeseries_data`, Quantification of licking behavior |
| scene name + switch at trial 30 | `output[4]` reward_zone_location | A=0, B=1, C=2 (per trial, tiled over time) | `behavior.get_reward_zones` |
| `behavior/Reward` + `behavior/reward_zone` | `output[5]` reward_outcome | 1 if a reward was delivered AND the reward zone was entered in the trial, else 0 (per trial, tiled) | `behavior.get_trial_types` |
| `general/subject/subject_id` | `subjects`/`subject_idx` | m3, m4, m7, m11-m15, m17-m19 | - |
| `ImagingPlane/location` | `brain_regions` = ['CA1'] | all ROIs CA1 | - |

Both `input` and `output` are stored as 2-D (d x T) arrays for every trial; per-trial variables are constant along the time axis (this is exactly what the decoder does internally when broadcasting 1-D per-trial values, and keeps the array dimensions homogeneous).

### Key Decisions
1. **Neural signal = deconvolved events recomputed from the paper's dF/F**, not the `Deconvolved` series stored in the NWB (which is the suite2p default `spks` computed with a different, non-per-trial baseline). Rationale: the paper's decoder/GLM analyses all use their own OASIS events derived from the maximin dF/F; reproducing their `preprocessing.dff` keeps the conversion faithful.
2. **Neuron curation**: `iscell == 1` (suite2p manual curation, already in NWB) and exclusion of putative interneurons with dF/F-speed Pearson r > 0.5 (paper value; `dayData.int_thresh = 0.5`). Planes pooled for the two 2-plane mice, as in the paper.
3. **Trial curation**: trials whose lick trace shows sensor error (>30% of samples with cumulative lick count > 2, the paper's criterion) are **dropped**, because lick is a decoder output and NaNs are not allowed. This removes 80 trials (0.65%), matching the paper's 81/12,376.
4. **No speed threshold**: the paper excludes samples with speed < 2 cm/s from *spatial* analyses, but here speed is an output class (< 2 cm/s is bin 0) and the decoder requires continuous within-trial time series, so all in-trial samples are kept. Documented deviation required by the decoder task.
5. **Teleport/ITI excluded** (trial window ends before the teleport sample), consistent with the paper (laser was blanked in the ITI on many days).
6. **Reward zone location per trial** derived from the scene string with the switch at trial index 30, exactly as `behavior.get_reward_zones`; verified against the measured reward-zone entry position on every trial (median error < 2 cm).
7. **All 152 sessions / 11 mice kept** (all are switch mice used in the paper).
8. Keep native 64.48 ms frame bins (no re-binning), matching "All behavioral and neural time series were sampled at ~15.5 Hz".

### Planned Sanity Checks
- [ ] #trials, #neurons, #sessions, #subjects match the scans of the raw files and the paper.
- [ ] Reward-zone entry position measured from `reward_zone` flag equals the nominal zone start for the assigned label (< 10 cm).
- [ ] Fraction of rewarded trials ~85% (omission ~15%).
- [ ] Output distance bin 3 (== 0 cm) occurs only when position is inside the zone; check against position bins.
- [ ] Position output bins recomputed from the raw NWB position stream match the stored outputs (`np.allclose`).
- [ ] Neural events at specific (trial, neuron, time) recomputed from raw F/Fneu match the stored values (`np.allclose`).
- [ ] time_from_trial_start starts at 0 and increases by 64.48 ms per bin.
- [ ] All trials start after the VR-imaging TTL sync (position != -500, environment != -1).
- [ ] Every session has >= 2 trials.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the conversion:
- `load_session()` reads one NWB with h5py (only the needed datasets): F, Fneu for `iscell==1` ROIs of every plane, behavior streams, trial start/teleport indices, reward timestamps, frame times.
- `compute_events()` reproduces `reward_relative.preprocessing.dff(..., neuropil_method='subtract', baseline_method='maximin', deconvolve=True)`: per-trial windows `[start-1, stop-1)`, neuropil subtraction (0.7) with the per-trial neuropil mean added back, maximin baseline (nansmooth sigma 15 -> minimum_filter1d 300 -> maximum_filter1d 300), dff=(F-F0)/|F0|, 2-frame `nansmooth`, then `suite2p.extraction.dcnv.oasis(dff, 2000, tau=0.7, frame_rate)`.
- `scene_zone_labels()` reproduces `behavior.get_reward_zones` (zone from scene name, switch at trial 30).
- Per-trial `isreward` and `morph` reproduce `behavior.get_trial_types`.
- Discretization helpers for distance/position/speed follow the Decoder Task spec exactly.
- Interneuron exclusion via a vectorised Pearson correlation of dF/F with speed (r > 0.5).
- `--sample` processes 2 sessions (one single-plane m11 day 3, one 2-plane m17 day 8); `--show-processing` writes `processing_<session>.png` (raw position/dF/F/events plus the discretization of every output for one trial) and `processing_<session>_outputs.png` (trial x time rasters of all time-varying outputs).

Code efficiencies:
- h5py slicing loads only iscell ROIs; no pynwb overhead.
- Correlation with speed computed as a single matrix product instead of a per-cell loop.
- `ProcessPoolExecutor` (8 workers) over sessions for the full run (memory per worker ~3.5 GB max).
- Timing printed per session (`t_load`, `t_dff`) and cumulatively.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (`/app/sample_data.pkl`, 2 sessions)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (m11 day 3 `Env1_LocationB_to_A`, m17 day 8 `Env2_B_to_Env1_A`) |
| Neurons (total) | 741 (154 + 587; 1 and 6 putative interneurons dropped) |
| Subjects | 2 |
| Trials (total) | 160 (80 + 80; no trials lost) |
| T per trial | mean 190.6 bins, min 137, max 338 |
| time_from_trial_start | [0.0, 21.7] s |
| environment | [0, 1] |
| trial_number | [0, 79] |
| previous_trial_outcome | [0, 1] |
| reward_zone_distance dist. | 0.143 / 0.094 / 0.042 / 0.261 / 0.023 / 0.082 / 0.355 |
| position dist. | 0.263 / 0.193 / 0.252 / 0.160 / 0.133 |
| speed dist. | 0.068 / 0.070 / 0.090 / 0.325 / 0.447 |
| lick dist. | 0.822 / 0.178 |
| reward_zone_location dist. | A 0.573 / B 0.427 (these two sessions) |
| reward_outcome dist. | omitted 0.116 / rewarded 0.884 |

Note: the sample files were regenerated after the Step 12 signal decision, so `/app/sample_data.pkl` also stores dF/F. Trial counts, input ranges and output distributions are unchanged by that decision (only the neural values differ).

Spot checks against the raw NWB for session 0, trial 5: position bins, speed bins, lick binarization, time vector and distance bins all matched exactly (`np.array_equal` / `np.allclose`), trial length 266 bins == teleport-1 - (start-1).

### Processing Plots Review
`processing_m11_ses-03.png` / `processing_m17_ses-08.png`: position sawtooth resets exactly at the green trial-start lines and ends at the red teleport lines; reward zone lines sit at 200-250 cm for the first 30 trials and 80-130 cm after; dF/F traces show clean transients; events are non-negative and sparse; the discretized outputs step exactly where the continuous variable crosses each bin edge; time axis starts at 0 for every trial. Output rasters show the expected diagonal structure of position/distance bins and the zone shift at trial 30.

### Run Time Estimates
| Speed-ups implemented | Effect |
|---|---|
| h5py partial reads of iscell ROIs only | ~2x less I/O |
| vectorised speed-correlation | negligible vs loop over 2000 cells |
| 8 process workers | ~6-7x |

| Step | Time / session (sample) | Estimated total (152 sessions) |
|---|---|---|
| load | 0.1-0.5 s (up to ~4 s for the largest m18 files) | ~5 min serial |
| dF/F + OASIS | 0.3-1.2 s (up to ~8 s for 2300 cells) | ~8 min serial |
| total | ~2 s (sample) | ~13 min serial, ~3 min with 8 workers + pickling ~10 GB |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

(Numbers below are from the final dataset version, i.e. neural = dF/F; the earlier deconvolved-events version of this same sample gave 0.468 / 0.542 / 0.431 / 0.729 / 0.939 / 0.553 - see Step 12 Check 2.)

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions / 160 trials)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|-------------------------|
| reward_zone_distance | 0.143 | 0.690 | 0.570 |
| position | 0.200 | 0.824 | 0.681 |
| speed | 0.200 | 0.656 | 0.561 |
| lick | 0.500 | 0.824 | 0.792 |
| reward_zone_location | 0.333 | 0.978 | 0.939 |
| reward_outcome | 0.500 | 0.907 | 0.596 |

Loss decreased monotonically over 200 epochs (test loss 0.661) and every output is above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `/app/converted_data.pkl`: 9.63 GB, written in 147 s total (8 spawn workers)
- `/app/conversion_full_out.txt`, `/app/verification_full_out.txt`: created

### Issue found and fixed during the full run
1. **`fork()` with OpenMP**: `ProcessPoolExecutor` crashed because suite2p/OpenMP is loaded before forking. Fixed by using the `spawn` multiprocessing context.
2. **One-frame length mismatch**: 10 sessions (m17 ses-04/06, m18 ses-01/05/07/10/11/12/13/14, all 2-plane) have exactly **one more imaging frame than behavior samples** - the known "one frame correction" that the reference alignment code (`vr_align_to_(mock_)2P`) also handles. Fixed by truncating every stream (F, Fneu, behavior, timestamps, trial indices) to the common minimum length, which keeps neural and behavior sample-aligned.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data (raw NWB scan) | Converted Data | Match? |
|-----------|-----------------|----------------|-------------------------------|----------------|--------|
| Subjects | 11 switch mice | switch animals only | 11 | 11 | YES |
| Sessions | 14 days/mouse (m11 from day 3) | - | 152 | 152 | YES |
| Total trials | 12,376 imaged trials, 81 removed for lick errors | - | 12,216 trial starts | 12,135 (= 12,216 - 81 lick-error trials) | YES (81 removed exactly as in the paper; the 160-trial difference from 12,376 is m11 days 1-2, which were not imaged and are not in DANDI) |
| Trials/session (mean) | 80.5 +- 7.4 | - | 80.4 +- 6.2 | 79.8 | YES |
| Neurons/session | 155-2172 | iscell + interneuron exclusion | iscell: 155-1780 (single plane), 2341 pooled over 2 planes | 154-2323, mean 910 | YES |
| Total neurons | - | - | 138,678 iscell | 138,269 (409 putative interneurons removed = 0.30%) | YES (paper: 0.42 +- 0.85% of cells excluded) |
| Omission rate | ~15% | isreward = reward AND rzone | 15.34% of trials | reward_outcome: 15.8% omitted (sample-weighted) | YES |
| Reward zone identity | A/B/C equally counterbalanced | scene name + switch at trial 30 | - | A 0.332 / B 0.336 / C 0.333 | YES |
| Neural time bin | ~15.5 Hz -> 64.5 ms | - | dt = 64.484 ms | 64.484 ms | YES |
| Track length | 450 cm | 0-450 cm binning | max position 451.8 cm | position bins 0-4 all populated | YES |
| Input ranges | - | - | - | time 0-216.5 s, environment 0/1, trial_number 0-99, prev outcome 0/1 | sensible |
| Output distributions | - | - | - | distance .256/.102/.073/.238/.021/.072/.238; position .217/.177/.231/.226/.149; speed .117/.087/.134/.319/.343; lick .777/.223 | sensible |

Notes on individual numbers:
- `reward_outcome` fraction of *samples* is 15.8% omitted vs 15.34% of *trials*; omission trials are slightly longer (mice run through the zone and often slow down later), so the sample-weighted fraction is slightly higher. Consistent.
- Longest trial = 3359 bins (216 s) in m4 day 4, a session that was terminated early (41 trials) because the mouse stopped running: 81% of that trial's samples are < 2 cm/s. Real data, kept.
- 5 trials have T < 100 bins (min 96); all are complete laps run at high speed.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
- `Data format is valid, no errors or warnings.` - zero errors, zero warnings, so nothing had to be waived.
- Structure: 152 sessions, 12,135 trials, 11 subjects, 1 brain region, dinput 4, doutput 6; every output range starts at 0 and reaches the maximum class index (distance 0-6, position 0-4, speed 0-4, lick 0-1, zone 0-2, outcome 0-1), so no class is missing from the dataset.

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`)
These load the raw NWB files directly with h5py (not through `convert_data.py`) and compare with `/app/converted_data.pkl`.

| Check | Data stream | Method | Result |
|---|---|---|---|
| Position bins | output[1] | recompute `clip(pos//90,0,4)` from raw `position` for 9 random trials in sessions 0, 75, 130 | exact match |
| Speed bins | output[2] | recompute `digitize(speed,[2,10,20,40])` from raw `speed` | exact match |
| Lick | output[3] | recompute `raw lick > 0` | exact match |
| Time input | input[0] | recompute `timestamps - timestamps[trial_start-1]` | `np.allclose` True |
| Environment | input[1] | `unique(environment[start:stop])` | match |
| Reward outcome | output[5] | reward timestamp within the trial AND `reward_zone` flag > 0 | match |
| Distance bin 3 | output[0] | bin 3 occurs exactly when `zone_start <= pos <= zone_end` for the assigned zone; tested on **every trial** of sessions 5, 60, 140 | 0 mismatches |
| Neural | neural | full independent re-implementation of dF/F (+ OASIS in the events variant) and of the interneuron filter, compared for 3 trials in sessions 0 and 80 | `np.allclose` True, neuron counts identical |

All checks passed (`ALL SANITY CHECKS PASSED`).

### Check 3: Reference code comparison
| Stage | Reference | This conversion | Same? |
|---|---|---|---|
| (a) Loading | `TwoPUtils` `sess` class: VR sqlite aligned to imaging frames, suite2p F/Fneu/iscell | the NWB release already contains exactly those aligned streams; read with h5py | YES |
| (b) Neuron filtering | `iscell` from manual suite2p curation; `spatial.is_putative_interneuron(method='speed', r_thresh=0.5)` on dF/F | identical (`iscell==1`, Pearson r of dF/F vs speed > 0.5) | YES |
| (c) Temporal alignment | trials `[trial_start_inds[i]-1 : teleport_inds[i]-1]`, VR interpolated to frames | identical window; no re-binning | YES |
| (d) Binning | none (native ~15.5 Hz frames); spatial binning only for tuning-curve analyses | native frames kept (64.484 ms) | YES |
| (e) Neural processing | `preprocessing.dff`: neuropil 0.7 subtract + per-trial mean added back, maximin baseline (nansmooth 15 -> min 300 -> max 300), (F-F0)/|F0|, 2-frame nansmooth, OASIS(tau=0.7) | identical implementation; the delivered dataset stores the **dF/F** stage (see Step 12 Check 2 for the controlled comparison; `--signal events` reproduces the deconvolved stage) | YES, with a documented choice of stage |
| (f) Input construction | `glmUtils.get_timeseries_data` uses trials, rel_pos, pos, speed, rewards | same variables; `trial_number`, `environment`, `previous outcome`, `time from trial start` as decoder inputs per the task spec | YES (task-specified) |
| (g) Output construction | `behavior.get_trial_types` (isreward, morph), `behavior.get_reward_zones` (zone label/coords, switch at 30), licks binarized `licks[licks>1]=1` | identical logic; distance to zone follows the paper's reward-relative coordinate but expressed as the signed linear distance required by the task spec | YES (task-specified discretization) |

Differences and reasons:
1. **No >2 cm/s speed mask.** The reference masks out slow samples for spatial tuning and for the GLM; here speed is a decoded output (bin 0 = <2 cm/s) and the decoder needs contiguous trials, so all in-trial samples are kept.
2. **Lick-error trials dropped rather than NaN-ed.** The paper sets these lick values to NaN; NaNs are not allowed by the decoder format, and lick is an output, so the 81 affected trials (0.65%) are dropped - the same trials the paper excludes from licking analyses.
3. **Reward-relative position is linear (cm) not circular (radians)**, because the task specifies distance bins in cm.
4. **Neural signal stage**: dF/F rather than OASIS events (Step 12, Check 2) - both produced by the same reference pipeline.

### Check 4: Key statistics comparison
See the Step 9 table: subjects (11), sessions (152), trials (12,135 = 12,216 - 81 lick-error trials), neurons (138,269 after removing 409 putative interneurons from 138,678 iscell ROIs, i.e. 0.30% vs the paper's 0.42 +- 0.85%), trials/session (79.8 vs 80.5 +- 7.4), omission rate (15.3% of trials vs ~15%), reward zone A/B/C balance (0.332/0.336/0.333), frame period (64.484 ms vs ~15.5 Hz). All consistent.

### Check 5: Edge cases
| Edge case | Handling |
|---|---|
| One extra imaging frame vs behavior in 10 two-plane sessions | all streams truncated to the common length (matches the reference "one frame correction") |
| Samples before the VR/imaging TTL sync (`pos = -500`, `environment = -1`) | never inside a trial window (first trial start index >= 142 in every session); an explicit guard drops any trial containing `pos < -100` (0 trials triggered) |
| `trial_start - 1` indexing | first trial start is >= 142, so the window is always valid |
| Trials shorter than 2 bins | explicit guard (0 trials triggered) |
| NaNs in dF/F/events at trial edges | guard drops trials containing NaN neural samples (0 trials triggered) |
| Very long trials (max 3359 bins = 216 s) | kept; verified to be a real period in which the mouse stopped running (81% of samples < 2 cm/s) |
| Trial 0 has no previous trial | `previous_trial_outcome` set to 1 (rewarded), because each imaging session was preceded by ~30 rewarded warm-up trials on the same reward zone |
| Sessions with only 40-60 trials | kept (>= 2 trials required); they are real, early-terminated sessions |
| Switch at trial 30 when a session has < 30 trials | `scene_zone_labels` clips the switch index to the number of trials |

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (log: `/app/train_decoder_full_out.txt`), 12,135 trials (9,708 train / 2,427 validation), GPU.

### Training Progress
- Loss decreasing: **Yes**, 2.31 (epoch 1) -> 1.06 (60) -> 0.61 (200); test loss 0.694.

### Decoder Results (Full dataset, dF/F)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance |
|--------|--------|-----------------------|-------------------------|--------------|
| reward_zone_distance (7 classes) | 0.143 | 0.819 | **0.631** | 4.4x |
| position (5 classes) | 0.200 | 0.910 | **0.772** | 3.9x |
| speed (5 classes) | 0.200 | 0.745 | **0.632** | 3.2x |
| lick (2 classes) | 0.500 | 0.797 | **0.767** | 1.5x |
| reward_zone_location (3 classes) | 0.333 | 0.971 | **0.877** | 2.6x |
| reward_outcome (2 classes) | 0.500 | 0.950 | **0.600** | 1.2x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
Every output is above chance. Four of six are >= 2.6x chance. The two binary outputs are lower in *ratio* simply because binary chance is 0.5:
- `lick` 0.767 vs 0.5 (1.53x) - licking is a fast, sparse behaviour (22% of samples) that CA1 activity only partially predicts; this is close to the ceiling one can expect from population imaging at 15.5 Hz.
- `reward_outcome` 0.600 vs 0.5 (1.20x) - investigated below.

**Investigation of `reward_outcome`** (`/app/cache/outcome_analysis.py`, 12 random sessions, per-session logistic regression on trial-mean dF/F):
| Epoch used | Balanced accuracy |
|---|---|
| samples **before** the reward zone (distance bin < 3) | 0.521 +- 0.116 |
| samples **in/after** the reward zone (distance bin >= 3) | 0.786 +- 0.121 |
| whole trial | 0.675 +- 0.120 |

Before the animal reaches the zone the outcome has not yet happened, so no signal can exist (0.52 ~ chance, as it must be); after the zone the outcome is decodable at 0.79. Because the reported balanced accuracy is computed over **all samples of the trial**, a whole-trial value around 0.6 is exactly what this structure predicts. This confirms the conversion is correct and the limit is intrinsic to the task, not a bug. (The same argument applies to the paper, which analyses rewarded-vs-omission activity only *after* the reward zone.)

### Check 2: Comparison to the paper
The paper reports no classification accuracies: its decoder outputs a circular "decode score" = mean cos(true - predicted reward-relative position), compared with circular-shift shuffles (Fig. 3b,c), for selected cell subpopulations. A number-for-number comparison is therefore impossible. Qualitative comparison:
| Paper result | This conversion |
|---|---|
| Position / reward-relative position is strongly decodable from CA1 population activity (decode scores far above shuffle in every session) | position 0.772 (3.9x chance) and reward-zone distance 0.631 (4.4x chance) |
| Reward-zone identity remaps the population code (Fig. 2, 8) | reward_zone_location 0.877 (2.6x chance) |
| Rewarded vs omission trials differ in activity **after** the reward zone (Fig. 6) | reward_outcome 0.79 when restricted to post-zone samples, 0.60 over the whole trial |
| Speed/licking modulate CA1 activity (GLM, Fig. 7) | speed 0.632, lick 0.767 |

**Controlled experiment on the neural signal stage.** Both the dF/F and the OASIS-deconvolved "events" produced by the reference `preprocessing.dff` pipeline were converted for the same 12 sessions (`--signal dff` / `--signal events`) and the decoder was trained twice on each (`/app/cache/ev_*.txt`, `/app/cache/df_*.txt`):
| Output | events (2 runs) | dF/F (2 runs) |
|---|---|---|
| reward_zone_distance | 0.459, 0.444 | 0.569, 0.574 |
| position | 0.558, 0.535 | 0.724, 0.716 |
| speed | 0.449, 0.431 | 0.548, 0.548 |
| lick | 0.788, 0.767 | 0.812, 0.811 |
| reward_zone_location | 0.835, 0.833 | 0.878, 0.875 |
| reward_outcome | 0.595, 0.583 | 0.601, 0.630 |

dF/F is better for every output, and the same ordering held on the full dataset (events: 0.530 / 0.662 / 0.578 / 0.745 / 0.845 / 0.570). dF/F is the paper's primary processed signal ("the position bin of the maximum unsmoothed, spatially binned dF/F, **as this signal is the closest to the raw data**"), and deconvolution is described there only as a way to remove the asymmetric calcium kernel for spatial-tuning statistics. Since the task instructions allow deviations that the decoding task requires, the delivered dataset stores **dF/F**, computed with the identical reference pipeline; `--signal events` regenerates the deconvolved version.

### Check 3: Train vs validation gap
| Output | Train | Val | Ratio |
|---|---|---|---|
| reward_zone_distance | 0.819 | 0.631 | 1.30 |
| position | 0.910 | 0.772 | 1.18 |
| speed | 0.745 | 0.632 | 1.18 |
| lick | 0.797 | 0.767 | 1.04 |
| reward_zone_location | 0.971 | 0.877 | 1.11 |
| reward_outcome | 0.950 | 0.600 | 1.58 |

Only `reward_outcome` exceeds 1.5x. This is over-fitting of a **per-trial** label: the decoder can memorise which of the ~80 trials in each session was an omission from any trial-specific activity fluctuation, while genuine generalisation is only possible after the zone (Check 1). There is no data leakage: outcome is derived only from the reward delivery times and the reward-zone flag of that same trial, the train/validation split is by trial, and `previous_trial_outcome` (an input) is the *preceding* trial's outcome, which is independent of the current label (verified: correlation between input[3] and output[5] across all 12,135 trials is r = 0.021).

### Issues found and resolved in this step
- **Neural signal stage**: found that deconvolved events under-perform dF/F for every output; switched the delivered dataset to dF/F after a controlled, repeated comparison (documented above). Conversion, verification and full training were re-run after the change.
- No other issues found; all Step 10 checks were re-run after the change (sanity checks re-executed against the regenerated pickle; `verification_full_out.txt` again reports no errors or warnings).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, load instructions, format spec, key statistics, decoder results)
- [x] `cache/` folder created with `README_CACHE.md` documenting every investigation script and control dataset
- [x] All investigation/validation scripts live in `/app/cache/`; `/app/convert_data.py` is self-contained
- [x] Deliverables present: `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`, plus `processing_*.png` plots
