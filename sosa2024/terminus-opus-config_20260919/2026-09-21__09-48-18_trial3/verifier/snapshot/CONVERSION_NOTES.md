# Dataset Conversion Notes

## Overview
- **Dataset**: "A flexible hippocampal population code for experience relative to reward" (Sosa, Plitt, Giocomo) - /app/data
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `CONVERSION_NOTES.md` (this file)
- `code/` : reference code repo (Sosa_et_al_2024)
- `data/` : DANDI dataset 001361, 11 subject dirs (sub-m3, m4, m7, m11-m15, m17-m19), 152 NWB files, 87 GB
- `decoder.py`, `train_decoder.py` : decoder + format verification code
- `methods.txt`, `paper.pdf` : reference text

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124 (CUDA available), pynwb 4.1.0, h5py 3.16.0, suite2p (suite2p.extraction.dcnv OASIS available).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Repo: `Sosa_et_al_2024` (code for Sosa, Plitt & Giocomo 2025, Nat Neurosci). Pipeline docs: `code/docs/preprocessing_guide.md`, `code/docs/multi_anim_sess_README.md`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `TwoPUtils.preprocessing.vr_align_to_2P` (external repo) | - | LOADING | Interpolates/aligns Unity VR sqlite data to the 2P imaging frame times -> `sess.vr_data` (one row per imaging frame, ~15.5 Hz). In the NWB files this alignment is already done (behavior timeseries share imaging timestamps). |
| `preprocessing.create_sess` / `append_session_data` | src/reward_relative/preprocessing.py | LOADING | Builds the `sess` object: vr_data, timeseries (F, Fneu, licks, rewards, speed), trial_start_inds, teleport_inds, iscell |
| `preprocessing.dff` | src/reward_relative/preprocessing.py | PROCESSING | Computes dF/F from F and Fneu: neuropil subtract (neu_coef=0.7), restricts to within-trial samples `[start-1:stop-1]` (NaN elsewhere), per-trial `maximin` baseline (nansmooth sigma=15 frames, then min filter 300 frames = ~20 s, then max filter 300), dFF=(F-base)/|base|, Gaussian smooth sigma=2 frames, then OASIS deconvolution (`suite2p.extraction.dcnv.oasis`, tau from s2p ops, fs=frame_rate/n_planes) -> `events` |
| `utilities.multi_anim_sess` + `default_dff_method` | src/reward_relative/utilities.py | PROCESSING/CURATION | Runs dff with neuropil_method='subtract', neu_coef=0.7, baseline_method='maximin', keep_teleports=False; computes place cells; stores isreward, morph, rzone, rz label, trial dict |
| `behavior.get_trial_types` | src/reward_relative/behavior.py | PROCESSING | Per trial: `isreward` = any(reward>0) AND any(rzone>0) within [trial_start, teleport); `morph` = unique environment value in trial (0=ENV1, 1=ENV2) |
| `behavior.get_reward_zones` | src/reward_relative/behavior.py | PROCESSING | Per-trial reward zone [start, stop] (cm) and label A/B/C from the scene name; zones: A=[80,130], B=[200,250], C=[320,370]; on switch sessions the zone changes at trial index 30 (`change_trial=30`, 0-indexed) |
| `behavior.correct_lick_sensor_error` | src/reward_relative/behavior.py | CURATION | Trials where >correction_thr (0.35-0.5) of samples have cumulative lick count >2 are set to NaN (lick sensor stuck) |
| `rewardAnalysis.get_omission_trials/get_omission_inds` | src/reward_relative/rewardAnalysis.py | PROCESSING | Omission trials = unrewarded trials where reward zone flag never active |
| `glmUtils.get_timeseries_data` | src/reward_relative/glmUtils.py | PROCESSING | Builds continuous (per-frame) behavior dataframe + deconvolved activity matrix restricted to within-trial samples `[start-1:stop-1]`: rel_pos (circular or linear distance to reward zone start), pos, trials, speed, accel, licks (binary, smoothed), rewarded, rewards; lick sensor error correction at 0.35; speed threshold (2 cm/s) sets samples to NaN and they are dropped |
| `decode.CircularRegression`, `decode.train_vs_test_blocks` | src/reward_relative/decode.py | ANALYSIS | The paper's circular-linear decoder of reward-relative position from deconvolved events (Fig. 3) |
| `sessions_dict.single_plane/multi_plane` | src/reward_relative/sessions_dict.py | LOADING | Per-animal session metadata: date, scene (e.g. `Env1_LocationA_to_C`), session, scan, exp_day (1-14) |

### Notes
- Neural signal used for decoding in the paper = **deconvolved calcium events derived from dF/F** (not raw suite2p spks). The NWB `processing/ophys/Deconvolved` is suite2p's own deconvolution of raw F; the reference pipeline recomputes dF/F (maximin, per trial) and re-deconvolves with OASIS. I will follow the reference: compute dF/F from NWB `Fluorescence` + `Neuropil` and deconvolve with OASIS (suite2p dcnv), matching `preprocessing.dff`.
- Cell curation: suite2p manual curation is stored in NWB as `PlaneSegmentation/iscell[:,0]`; the paper additionally excludes putative interneurons with Pearson r>0.5 between dF/F and running speed (0.42 +/- 0.85% of cells).
- Trials are defined by `trial_start_inds` (entry to track) and `teleport_inds` (end of track). dff/`get_timeseries_data` use the window `[start-1, stop-1)`.
- Behavior and neural streams are already on the same ~15.5 Hz frame clock (64.48 ms/frame).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` is DANDI dandiset 001361 (Sosa, Plitt & Giocomo 2025), 152 NWB files (87 GB), one file per
(subject, session): `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`. `ses-NN` == experiment day (1-14).
11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19 (the 11 switch-task mice of the paper).
m11 has only days 3-14 (12 sessions; no imaging on days 1-2, as stated in Methods); all others have 14.

Contents of each NWB file (verified with h5py):
- `identifier` = original path, e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`, which gives the
  **scene name** needed for reward-zone lookup, exactly as reference `behavior.get_reward_zones` uses `sess.scene`.
- `general/session_id` = experiment day; `general/subject/subject_id` = mouse; `general/optophysiology/ImagingPlane`:
  location `hippocampus, CA1`, indicator GCaMP7f, imaging_rate 15.5078125 Hz (31.015625 for 2-plane mice m17, m18).
- `processing/behavior/BehavioralTimeSeries/` (all 1-D, one sample per imaging frame, shared timestamps, dt=0.064484 s):
  `position` (cm; -500 before VR sync, -50 during teleport/ITI), `speed` (cm/s, smoothed),
  `lick` (cumulative lick count per frame, 0-7), `reward_zone` (cumulative reward-zone-entry flag per frame),
  `environment` (0 = ENV1, 1 = ENV2, -1 before sync), `trial number` (0-indexed lap, -1 before sync),
  `trial_start` (binary), `teleport` (binary), `autoreward`, `scanning` (1 = scanning, -1 before sync).
  `Reward` is a sparse TimeSeries: one sample per delivered reward (data = 0.004 mL) with timestamps that
  exactly equal frame timestamps (verified on all 152 sessions).
- `processing/ophys/`: `Fluorescence/plane<i>/data` (frames x ROIs, raw F), `Neuropil/plane<i>/data` (Fneu),
  `Deconvolved/plane<i>/data` (suite2p deconvolution of raw F, NOT the paper events),
  `ImageSegmentation/PlaneSegmentation` with `iscell` (n_roi x 2: manual-curation flag + probability) and `planeIdx`.
  2-plane mice (m17, m18) have plane0 and plane1 groups; the ROI table is concatenated over planes (planeIdx 0/1).
  **No dF/F is stored**, so it must be computed as in the reference `preprocessing.dff`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, iscell==1, all sessions) | 138,678 |
| Neurons / session | mean 912.4, min 155, max 2341 |
| Subjects | 11 |
| Sessions / subject | 14 for all except m11 (12); 152 total |
| Trials (total, trial_start events) | 12,216 |
| Trials / session | mean 80.37, s.d. 6.16, min 41, max 100 |
| Within-trial frames (total) | 2,620,514 (= 9.7 GB of float32 cell x frame data) |
| Trial duration | median 12.25 s, mean 13.8 s, min 6.2 s, max 216.6 s |
| Frame period | 0.0644836 s (15.5078 Hz) |
| Rewarded trials | 10,342 (84.7%) |
| Omission trials | 1,874 (15.3%) |

### Edge-case checks run on all 152 sessions (0 problems)
- Number of trial_start == number of teleport in every session; starts always precede their teleport; no overlap.
- No session has a trial_start at frame 0, so the reference window `[start-1, stop-1)` is always valid.
- No trial window contains pre-sync samples (`trial number` < 0), non-scanning samples (`scanning` != 1),
  mixed/invalid `environment`, or teleport-period positions (< -20 cm).
- Reward event timestamps exactly match frame timestamps.
- Reward-zone onset position (first frame with `reward_zone` > 0) matches the scene-derived zone from the
  reference dictionary: e.g. `Env1_LocationB_to_A`: trials 0-29 onset ~200 cm (zone B = [200,250]),
  trials 30+ onset ~80 cm (zone A = [80,130]); `Env1_C_to_Env2_B`: 320 cm then 200 cm, environment flips 0 to 1 at trial 30.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch task) | 11 mice | counterbalanced across mice (n = 11 mice); n = 81 out of 12,376 trials removed across 11 switch mice |
| Mice beginning in ENV1 / ENV2 | 9 / 2 (m17, m18 began in ENV 2) | Most mice began the task in ENV 1 (n = 9 mice; two mice (m17 and m18) began in ENV 2) |
| Sessions / subject | 14 days; m11 imaged from day 3 | The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3 |
| Trials / session | 80-100 targeted; mean +/- s.d. 80.5 +/- 7.4 (across 14 mice) | We targeted 80-100 trials per session ... (mean +/- s.d., 80.5 +/- 7.4 trials across 14 mice, all imaging days) |
| Trials (total, imaged, 11 switch mice) | 12,376 | ~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice |
| Neurons / session | 155-2172 putative pyramidal neurons | This approach yielded 155-2172 putative pyramidal neurons per session |
| Neural data time bin | 0.0645 s (~15.5 Hz imaging frame) | All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate |
| Behavior data time bin | same 0.0645 s (VR aligned to imaging frames) | as above |
| Track length | 450 cm | linear track position (from 0 to 450 cm) |
| Reward zone width / locations | 50 cm; A=[80,130], B=[200,250], C=[320,370] | reward_zone_dict in behavior.py (X/Y/Z map to A/B/C) |
| Switch trial | after 30 trials | Each switch occurred after 30 trials |
| Autoreward | first 10 trials of a new condition | On the first ten trials of any new condition ... the reward was automatically delivered at the end of the zone if the mouse had not yet licked |
| Lick-error trials removed | 81 of 12,376 (~0.65%) | detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2 |
| Putative interneurons excluded | 0.42 +/- 0.85% of cells (dF/F vs speed Pearson r > 0.5) | Additional putative interneurons were detected for exclusion ... by a Pearson correlation of >0.5 between their dF/F timeseries and the running speed |
| Speed threshold in paper analyses | 2 cm/s | deconvolved calcium event timeseries (at speeds of >2 cm s-1) |
| Reward omission rate | ~15% (measured in data: 15.3%) | omission trials analysed throughout Fig. 6 |

My measured values (Step 2) agree: 152 sessions, 11 mice, 12,216 trial_start events, 80.37 +/- 6.16 trials/session,
155-2341 cells/session, and **exactly 81 trials** flagged by the paper lick-error rule (>30% of frames with
cumulative lick count > 2), a strong sanity check that loading and trial segmentation match the reference.
The paper 12,376 trials is 160 more than the 12,216 trial_start events in the NWB files (1.3% difference); the
NWB conversion appears to drop a few incomplete laps (e.g. a final lap without a teleport). Every NWB session
has exactly matching numbers of trial_start and teleport events, so all 12,216 complete laps are used.
The max of 2341 cells/session (m18 day 3, 2-plane) slightly exceeds the paper range 155-2172; this is the pooled
two-plane count and the paper range may have been computed per plane or on a subset of days.

### Processing Details
- **Temporal alignment**: VR behaviour is already interpolated onto imaging frame times (vr_align_to_2P in the
  authors TwoPUtils repo); in the NWB files every behavioural series shares the imaging-frame timestamps.
- **Trial window**: reference code (preprocessing.dff, glmUtils.get_timeseries_data) uses
  [trial_start-1, teleport-1), i.e. from one frame before the trial-start flag up to one frame before teleport.
- **dF/F**: per-trial maximin baseline: neuropil subtraction (0.7 x Fneu, per-trial neuropil mean added back),
  Gaussian smooth (sigma = 15 frames), minimum filter (300 frames ~ 20 s), maximum filter (300 frames),
  dF/F = (F - baseline)/|baseline|, then Gaussian smoothing with sigma = 2 frames (~0.129 s).
- **Events**: OASIS deconvolution of smoothed dF/F per trial (suite2p.extraction.dcnv.oasis, batch 2000,
  tau = 0.7 s from the suite2p ops in the repo notebook, fs = frame_rate/n_planes = 15.5078 Hz).
- **Reward-relative position**: distance from the start of the active reward zone (rzone[i,0]) per trial.
- **GLM/decoder variables**: position (0-450 cm), RR position, rewarded binary (0 until reward delivery then 1
  until end of trial), speed, acceleration, smoothed binary licks.

### Curation Steps
**Neuron curation rules**:
1. suite2p manual curation: keep ROIs with iscell[:,0] == 1 (the manual curation of the Methods).
2. Exclude putative interneurons: Pearson r > 0.5 between the cell dF/F and running speed (within-trial samples).
3. Planes are pooled for multi-plane mice (m17, m18).

**Trial curation rules**:
1. Only complete laps, defined by paired trial_start/teleport flags; samples restricted to [start-1, stop-1).
2. Lick-sensor error trials (>30% of frames with cumulative lick count > 2, n = 81) have lick values set to NaN in
   the reference; because lick is a required decoder output here, these trials cannot provide a valid lick label,
   so they are dropped entirely (see Step 5).

### Decoders Trained (paper)
| Decoded variable | Accuracy |
| Reward-relative (circular) position from deconvolved events of RR / track / non-RR remapping cells (Fig. 3) | reported as a mean cosine decode score (1 = perfect, 0 = chance) and as a z-score versus a 100-shuffle null, not as % classification accuracy. Fig. 3a shows a single fold of an example session; group means are well above shuffle. No classification accuracies are reported in the paper, so no direct numeric comparison with this task accuracy is possible. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

I cross-checked the reference code, the NWB data, and the paper text. All questions raised were resolved:

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal for decoding | `glmUtils.get_timeseries_data` uses `sess.timeseries[events]`, produced by `preprocessing.dff(..., deconvolve=True)` from raw F/Fneu | NWB stores raw `Fluorescence`, `Neuropil`, and a `Deconvolved` series whose values are large (0-13,770) and have no NaNs, i.e. suite2p deconvolution of **raw F**, not of the paper dF/F | dF/F computed with per-trial maximin baseline, smoothed, then OASIS-deconvolved | Recompute dF/F + OASIS events from NWB F and Fneu exactly as `preprocessing.dff` does. The stored `Deconvolved` series is not used. |
| Trial window | `[trial_start-1, teleport-1)` | trial_start flag frame has pos ~0-4 cm and the frame before is ~-3 to -7 cm (i.e. the last teleport-tunnel sample); teleport flag frame is the first ITI sample | not specified | Use the reference window `[start-1, stop-1)`; verified that this window never includes teleport-period positions (min pos in window > -20 cm over all sessions). |
| Reward zone per trial | derived from scene name with `change_trial=30` | `reward_zone` flag onset position matches the scene-derived zone for trials 0-29 and 30+; flag is absent on omission trials | zone switched after 30 trials | Use scene name + change at trial 30 (reference logic), and verify against the data-derived onset position per session (sanity check). |
| Rewarded vs omission | `behavior.get_trial_types`: reward>0 AND rzone>0 in trial | 84.7% of trials rewarded | omissions on a subset of trials | Use the reference rule with the sparse `Reward` timeseries mapped to frames. |
| Number of trials | - | 12,216 complete laps | 12,376 imaged trials | 1.3% difference; NWB conversion evidently omits a few incomplete laps. All 12,216 complete laps are used. |
| Cells per session | `iscell` used for curation | 155-2341 (pooled planes) | 155-2172 | Lower bound matches exactly; upper bound differs only for the two 2-plane mice, where the paper may quote per-plane counts. Keep pooled planes (paper pools planes for all analyses except Ext. Data Fig. 7). |
| Lick sensor errors | threshold 0.35 or 0.5 in code | 81 trials at >0.30, 69 at >0.35, 43 at >0.5 | >30% threshold, 81 trials | Use the paper threshold of 0.30, which reproduces 81 trials exactly. |
| Speed threshold | 2 cm/s (samples dropped) in GLM/decoder analyses | speed can be slightly negative (min -2 cm/s) | at speeds of >2 cm s-1 | Cannot be applied here: speed (including a <2 cm/s class) is a required decoder output and the decoder needs contiguous per-trial time series. All within-trial samples are kept (documented deviation). |
| Interneuron exclusion | `spatial.is_putative_interneuron` (dF/F vs speed r > 0.5) | 0.6-3.6% per session in spot checks | 0.42 +/- 0.85% of cells | Apply the r > 0.5 rule using within-trial dF/F samples, and report the overall fraction excluded. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable (NWB) | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Fluorescence/plane*/data` (F), `ophys/Neuropil/plane*/data` (Fneu) | neural | per-trial dF/F (neuropil subtract 0.7, maximin baseline, sigma=2 smoothing) then OASIS deconvolution to events; restricted to `[start-1, stop-1)` | `preprocessing.dff` (deconvolve=True) called as in `utilities.multi_anim_sess`; `glmUtils.get_timeseries_data` for the trial window | float32, shape (n_cells, T_trial) |
| `ophys/ImageSegmentation/PlaneSegmentation/iscell[:,0]`, `planeIdx` | neuron curation | keep iscell==1; pool planes | suite2p manual curation; paper Methods | |
| dF/F vs `speed` Pearson r | neuron curation | exclude cells with r > 0.5 | `spatial.is_putative_interneuron` | paper: 0.42 +/- 0.85% of cells |
| frame timestamps | input[0] `time_from_trial_start` | t - t[trial_start frame], seconds | - | first sample of window = -0.0645 s |
| `environment` | input[1] `environment` | 0 = ENV1, 1 = ENV2 (constant per trial) | `behavior.get_trial_types` (morph) | |
| `trial number` / lap index | input[2] `trial_number` | 0-indexed lap within session (constant per trial) | `glmUtils.get_timeseries_data` (trials) | preserves the original lap index even when trials are dropped, so the switch at lap 30 stays interpretable |
| previous trial `isreward` | input[3] `previous_trial_outcome` | 0 = omitted, 1 = rewarded (constant per trial) | `behavior.get_trial_types` | first lap of a session has no previous lap and is dropped |
| `position` + scene-derived reward zone | output[0] `reward_zone_distance` | signed distance to the nearest point of the reward zone (0 inside the zone), 7 bins: <-50, [-50,-10), [-10,0), 0, (0,10], (10,50], >50 | `glmUtils.get_timeseries_data` rel_pos (linear variant, relative to `rzone[i,0]`), `behavior.get_reward_zones` | task spec asks for distance to **any** location in the reward zone, hence 0 throughout the 50-cm zone |
| `position` | output[1] `position` | 5 equal bins of the 450-cm track: <90, [90,180), [180,270), [270,360), >=360 | `glmUtils.get_timeseries_data` pos | |
| `speed` | output[2] `speed` | 5 bins: <2, [2,10), [10,20), [20,40), >=40 cm/s | `glmUtils.get_timeseries_data` speed | negative speeds fall into the <2 cm/s bin |
| `lick` | output[3] `lick` | binary: cumulative count >= 1 -> 1 | reference sets `licks[licks>1]=1` (any lick in the frame) | trials flagged by the lick-sensor error rule are dropped |
| scene-derived reward zone label | output[4] `reward_zone_location` | A=0, B=1, C=2 (constant per trial) | `behavior.get_reward_zones` (rz_labels) | |
| `Reward` timeseries + `reward_zone` | output[5] `reward_outcome` | 1 if a reward was delivered inside the trial and the reward-zone flag was active, else 0 (constant per trial) | `behavior.get_trial_types` (isreward) | |
| `general/subject/subject_id` | subjects / subject_idx | m3..m19 | - | 11 subjects |
| `general/optophysiology/ImagingPlane/location` | brain_regions | all cells CA1 | - | single region |

### Key Decisions
1. **Neural signal = deconvolved events recomputed from F/Fneu** (not the stored `Deconvolved` series): the paper and
   all reference analyses (GLM, Fig. 3 decoder) use OASIS events derived from the per-trial maximin dF/F. The stored
   NWB `Deconvolved` series is suite2p deconvolution of raw F (no neuropil correction, no per-trial baseline), so it
   would not match the reference processing.
2. **Time bin = one imaging frame (64.4836 ms)**, no re-binning: the reference pipeline processes all neural and
   behavioural data at the imaging frame rate (~15.5 Hz) and the NWB behavioural series are already interpolated
   onto those frame times, so this preserves exact temporal alignment with no resampling.
3. **Trial window `[trial_start-1, teleport-1)`** exactly as in `preprocessing.dff` and
   `glmUtils.get_timeseries_data`; alignment event = trial start (entry to the linear track), off_start = -0.0645 s
   (the window begins one frame before the trial-start flag), off_end = None (variable trial length).
4. **Cell curation**: iscell==1 (suite2p manual curation) and exclusion of putative interneurons by dF/F-speed
   Pearson r > 0.5, both as in the reference. Planes pooled for m17/m18 (paper pools planes).
5. **No speed threshold**: the reference drops samples with speed < 2 cm/s for its GLM/decoder, but speed is a
   required decoder output here (with an explicit <2 cm/s class) and the decoder needs contiguous trial time series,
   so all within-trial samples are kept. This is a required deviation, documented here.
6. **Dropped trials**: (a) the 81 lick-sensor-error trials (>30% of frames with cumulative lick count > 2), because
   lick is a required output and the reference treats these lick values as invalid (NaN); (b) the first lap of each
   session (152 trials), because previous-trial outcome is a required input and is undefined for it (the preceding
   warm-up laps are not in the imaging record). Remaining trials: ~11,980 of 12,216 (98.1%).
7. **Per-trial variables are stored as constant time series** (shape (d, T)) so that every trial has a uniform
   (d_input, T) / (d_output, T) layout, and the instruction to make outputs time-varying where possible is honoured.
8. **Reward zone per trial from the scene name with the switch at lap 30** (reference `behavior.get_reward_zones`),
   verified against the data-derived reward-zone onset position in every session.

### Planned Sanity Checks
- [x] Trial counts: 12,216 complete laps, 80.37 +/- 6.16 per session (paper: 80.5 +/- 7.4), 11 mice, 152 sessions.
- [x] Lick-error rule reproduces the paper 81 trials exactly.
- [x] Cells/session 155-2341 (paper 155-2172) and 138,678 total before interneuron exclusion.
- [ ] Scene-derived reward zone matches the data-derived reward-zone onset position in every session (per-session assert).
- [ ] Fraction of cells excluded as putative interneurons ~0.4-1% (paper 0.42 +/- 0.85%).
- [ ] Reward outcome fraction ~84.7% rewarded / 15.3% omission.
- [ ] Environment: 9 mice start in ENV1, 2 (m17, m18) in ENV2; environment flips at lap 30 only on cross-env days.
- [ ] Output distributions: position bins roughly uniform (~20% each); distance-to-zone bin 3 (inside zone) ~11%
  (50 cm of the 450-cm track); speed mostly in 10-40 cm/s bins; licks on a small fraction of frames.
- [ ] Spot-check raw-data sanity checks with np.allclose on neural, input and output values (Step 10).
- [ ] dF/F values in a plausible range (about -1 to 10) and events non-negative.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the conversion. Structure:
- `load_session(fn)`: reads one NWB file with h5py: behavioural series, frame timestamps, sparse Reward
  timestamps, trial_start/teleport frame indices, scene name from `identifier`, and the iscell-filtered
  F / Fneu matrices of every imaging plane (concatenated over planes).
- `compute_dff_and_events(...)`: port of `reward_relative.preprocessing.dff` with the reference settings
  (neuropil subtract 0.7 with per-trial neuropil mean added back, per-trial maximin baseline:
  gaussian sigma=15 frames, minimum then maximum filter of 300 frames (~20 s), dF/F = (F-base)/|base|,
  gaussian sigma=2 frames), then per-trial OASIS deconvolution (`suite2p.extraction.dcnv.oasis`,
  batch 2000, tau 0.7, fs = imaging_rate/n_planes).
- `nansmooth`: port of `reward_relative.utilities.nansmooth`.
- interneuron exclusion: vectorised Pearson correlation between each cell dF/F and speed over the
  within-trial samples, threshold 0.5 (port of `spatial.is_putative_interneuron`, method=speed).
- `scene_reward_zones(scene, n_trials)`: port of `behavior.get_reward_zones` (dictionary A/B/C zones,
  switch after 30 laps, handles both `Env1_LocationA_to_C` and `Env1_C_to_Env2_B` scene styles).
- trial-type computation mirrors `behavior.get_trial_types` (rewarded = reward delivered AND reward-zone
  flag active) and the Methods lick-sensor-error rule (>30% of frames with cumulative lick count > 2).
- `reward_zone_distance` / `digitize_rz_distance` implement the required output discretisation.
- `--show-processing` writes three figures per session: (1) raw F, neuropil F, dF/F, events, position with
  the reward zone shaded, speed and licks/rewards over four consecutive laps with trial start/teleport
  lines, so temporal alignment of every stream is visible; (2) per-trial comparisons of each continuous
  behavioural variable with its discretised output plus session-wide value-vs-bin scatter plots (proving
  the discretisation) and the converted neural raster next to position; (3) per-lap reward zone, reward
  outcome, previous outcome and environment.
- Sessions are processed in parallel (`multiprocessing`, default 8 workers, `maxtasksperchild=1` to keep
  memory bounded); each worker returns only the converted per-trial arrays.

Code inefficiencies identified: naive per-cell correlation loop for interneuron detection; float64
intermediates for F/Fneu.
Code speedups added: vectorised correlation (matrix-vector product), OASIS on float32 arrays, reading only
iscell columns from HDF5, parallel sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sample = the two smallest sessions from two different mice: m11 day 4, m19 day 2).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 572 (575 iscell, 3 = 0.52% excluded as putative interneurons) |
| Neurons / session | 167, 405 |
| Subjects | 2 (m11, m19) |
| Sessions / subject | 1 |
| Trials (total) | 158 of 160 laps (2 first laps dropped, 0 lick-error laps) |
| Trials / session | 79, 79 |
| Timepoints / trial | mean 206, min 147, max 476 |
| time_from_trial_start range | [-0.064, 30.6] s |
| environment range | [0, 0] (both sessions ENV1) |
| trial_number range | [1, 79] |
| previous_trial_outcome range | [0, 1] |
| reward_zone_distance distribution | [0.259, 0.106, 0.088, 0.253, 0.020, 0.070, 0.205] |
| position distribution | [0.232, 0.174, 0.151, 0.301, 0.142] |
| speed distribution | [0.110, 0.076, 0.160, 0.222, 0.432] |
| lick distribution | [0.816, 0.184] |
| reward_zone_location distribution | [0.440, 0.000, 0.560] (zone A in m11 day 4, zone C in m19 day 2) |
| reward_outcome distribution | [0.212, 0.788] |
| dF/F range | [-0.21, 4.97] |
| median reward-zone onset error | 0.63 cm (data-derived zone entry vs scene-derived zone start) |

### Processing Plots Review
`processing_m11_ses-04_*.png` and `processing_m19_ses-02_*.png`:
- raw F and neuropil traces are continuous; dF/F is flat-baselined within each lap and returns to ~0 between
  transients; events are non-negative and sparse and occur at the rising phase of dF/F transients.
- position sawtooths from 0 to ~450 cm within every marked trial-start/teleport pair; the shaded reward zone
  is at 80-130 cm (m11 day 4, zone A) and 320-370 cm (m19 day 2, zone C), consistent with the scene names.
- rewards (red stars) occur inside the shaded zone on rewarded laps, and licks cluster just before/inside it.
- the value-vs-bin scatter plots are exact step functions at the specified edges (90/180/270/360 cm;
  2/10/20/40 cm/s; -50/-10/0/+10/+50 cm), with a flat bin-3 plateau across the whole 50 cm reward zone.
- per-lap plots show previous_trial_outcome equal to the reward outcome of the preceding lap.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| vectorised speed-correlation, float32 OASIS, reading only iscell ROI columns, 8-way session parallelism | ~8x wall-clock from parallelism; the per-session cost is dominated by HDF5 reading |

| Step | Time / Session | Estimated Total Time |
| load NWB (small session, 168 cells) | 0.2 s | - |
| dF/F + OASIS (small session) | 0.5 s | - |
| whole small session | ~5 s (incl. process start-up) | - |
| large session (1052 cells, 34k frames) prototype | load 1.7 s, dF/F 3.0 s, OASIS 1.3 s, ~10 s total | - |
| full dataset, 152 sessions, 8 workers | mean ~15-20 s/session serial | ~7-10 min wall clock |

`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` -> `Data format is valid, no errors or
warnings.` All input/output ranges and distributions as listed above.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

Training loss decreased monotonically: 2.45 (epoch 8) -> 0.897 (epoch 200); test loss 0.832.

### Decoder Results (Sample, 2 sessions, 158 trials; log `/app/train_decoder_sample_out.txt`)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| reward_zone_distance | 0.565 | 0.598 | 0.143 |
| position | 0.578 | 0.597 | 0.200 |
| speed | 0.521 | 0.501 | 0.200 |
| lick | 0.792 | 0.817 | 0.500 |
| reward_zone_location | 0.998 | 0.998 | 0.333 |
| reward_outcome | 0.616 | 0.558 | 0.500 |

Training loss decreased monotonically (2.45 -> 0.90 over 200 epochs); test loss 0.821.
All outputs are above chance. `reward_outcome` is the weakest (0.558): it is a single per-trial label, so
158 trials from 2 sessions give the decoder very little to learn from, and 79% of laps are rewarded; it is
re-examined on the full dataset in Steps 11-12. `reward_zone_location` is near-perfect because each of these
two sessions has a single constant zone, so it is identifiable from the session projection.

--------|-------------|--------|--------|
| reward_zone_distance | 0.532 | 0.553 | 0.143 |
| position | 0.578 | 0.599 | 0.200 |
| speed | 0.490 | 0.475 | 0.200 |
| lick | 0.803 | 0.829 | 0.500 |
| reward_zone_location | 0.998 | 0.998 | 0.333 |
| reward_outcome | 0.606 | 0.517 | 0.500 |

All outputs are above chance. `reward_outcome` is the weakest (0.517): it is a single per-trial label, so the
sample of 158 trials (2 sessions) gives the decoder very little to learn from, and 79% of laps are rewarded.
To be re-examined on the full dataset (Step 11/12). `reward_zone_location` is near-perfect because each of
these two sessions has a single constant zone, so it is identifiable from the session projection.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Commands:
`python -u /app/convert_data.py /app/converted_data.pkl --full --nproc 12` (199 s wall clock, 12 workers)
`python -u /app/train_decoder.py /app/converted_data.pkl --verify-only`

### Output Files
- `converted_data.pkl`: 9.47 GB
- `conversion_full_out.txt`, `verification_full_out.txt`, `converted_data_session_stats.csv` (per-session stats)

### Issue found and fixed during the first full run
The first full run crashed on 10 sessions of the two 2-plane mice (m17 x2, m18 x8): in those files the ophys
arrays have exactly one frame more than the behavioural series (e.g. 19,821 vs 19,820). Both streams start at
t = 0, so the extra imaging frame is at the end; `load_session` now truncates all streams to the common length
(and prints a NOTE). A second issue was found in the same run: in 8 sessions (again only m17/m18) dF/F reached
|dF/F| ~ 3000 for a handful of cells. Diagnosis: for those cells the neuropil trace is much larger than the ROI
trace (e.g. Fneu 4960 vs F 1487), so after the reference neuropil subtraction the corrected trace crosses zero
and the per-trial maximin baseline collapses to ~0; dF/F = (F - base)/|base| then explodes. Because the decoder
PCA would be dominated by such numerically degenerate cells, cells whose minimum within-trial baseline is below
5% of their median raw fluorescence are now excluded (25 cells = 0.018% of all curated cells). After the fix the
dF/F range over kept cells is [-16.0, 37.2] and the maximum deconvolved event is 7.1.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data (NWB) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 11 switch mice | anim lists include mice not in DANDI | 11 | 11 | yes |
| Sessions | 14 days/mouse, m11 from day 3 | sessions_dict: 14 days/mouse | 152 | 152 | yes |
| Sessions per subject | 14 (m11: 12) | 14 | 14 (m11: 12) | 14 (m11: 12) | yes |
| Total neurons (curated) | 155-2172 per session | iscell | 138,678 iscell | 138,244 after interneuron (409) and unstable-baseline (25) exclusion | yes |
| Mean neurons/session | - | - | 912.4 | 909.5 (min 154, max 2319) | yes |
| Trials (total) | 12,376 imaged | - | 12,216 complete laps | 11,983 (12,216 - 81 lick-error - 152 first laps) | expected |
| Trials/session (mean) | 80.5 +/- 7.4 | - | 80.37 +/- 6.16 | 78.8 (min 39, max 99) | yes |
| Lick-error trials | 81 (0.65%) | rule with thr 0.35/0.5 | 81 at the paper thr 0.30 | 81 dropped | exact |
| Rewarded trials | omissions ~15% | isreward rule | 84.7% | 84.3% of frames rewarded; 84.7% of raw laps | yes |
| Putative interneurons | 0.42 +/- 0.85% | r > 0.5 | - | 409/138,678 = 0.29% | yes |
| Frame period / time bin | 0.0645 s | frame rate 15.5 Hz | 0.0644836 s | 64.4836 ms | yes |
| Track length | 450 cm | 0-450 cm | max position ~451 cm | position bins span 0-450+ cm | yes |
| Reward zones | A/B/C, 50 cm | A=[80,130], B=[200,250], C=[320,370] | zone entry at 80/200/320 cm | median onset-minus-zone-start error 0.78 cm | yes |
| Reward zone balance | counterbalanced across mice | - | - | A 33.1%, B 33.6%, C 33.2% of frames | yes |
| Environment | 9 mice start ENV1, 2 ENV2 | morph 0/1 | env in {0,1} | input range [0,1] | yes |
| time_from_trial_start | - | - | - | [-0.064, 216.5] s (longest lap 216 s) | yes |
| trial_number | 80-100 laps | - | 0-99 | [1, 99] | yes |
| Output distributions | - | - | - | rz-distance [0.255, 0.102, 0.074, 0.239, 0.021, 0.072, 0.238]; position [0.216, 0.176, 0.232, 0.227, 0.149]; speed [0.117, 0.087, 0.133, 0.316, 0.347]; lick [0.778, 0.222]; zone [0.331, 0.336, 0.332]; outcome [0.157, 0.843] | plausible |

Verification output: `Data format is valid, no errors or warnings.`

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` begins with `Data format is valid, no errors or warnings.` There are no errors
and no warnings to address. The only messages in `/app/conversion_full_out.txt` are the informational NOTE
lines for the 10 m17/m18 sessions where the ophys stream is one frame longer than the behavioural stream
(handled by truncation, see Step 9).

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`)
These checks load the raw NWB files directly and re-derive quantities with code written independently of
`convert_data.py`, then compare with `np.allclose`. Three sessions were chosen to cover the different session
types: `m11 ses-03` (1 plane, within-environment reward-zone switch B->A), `m3 ses-08` (cross-environment
switch `Env1_C_to_Env2_B`) and `m18 ses-13` (2-plane mouse, one of the sessions with the frame-count offset).
All 39 checks passed (`0 checks failed`):
- **neural**: an independent re-implementation of the reference dF/F (neuropil subtraction, per-trial maximin
  baseline, sigma=2 smoothing) + OASIS deconvolution, plus the two curation rules, reproduces the number of
  kept cells exactly (154/154, 778/778, 1427/1427) and the stored event matrices for the first, middle and last
  trial of each session (`np.allclose`, atol 1e-5). A single-value spot check
  (`neural[trial 5, neuron 3, timepoint 10]`) also matches, e.g. 0.022363 for m11 ses-03.
- **input**: `time_from_trial_start` equals `timestamps[start-1:stop-1] - timestamps[start]` for the first,
  middle and last trial; `environment` equals the NWB environment value of the lap; `trial_number` equals the
  NWB lap index; `previous_trial_outcome` equals the independently recomputed `isreward` of the preceding lap
  for every trial of the session.
- **output**: every one of the six outputs was recomputed for every kept trial from raw NWB position/speed/lick
  and from the scene-derived reward zone, and matched exactly (reward-zone distance bins, position bins, speed
  bins, lick binary, reward-zone location, reward outcome).
- **structure**: `subject_idx` matches the per-session subject, `brain_region_idx` lengths match the neuron
  counts, and sampled neural arrays contain no NaN/Inf.

### Check 3: Reference code comparison
| Step | Reference | This conversion | Same? |
|------|-----------|-----------------|-------|
| (a) data loading | `TwoPUtils.sess` + `vr_align_to_2P` build `sess.vr_data` (per imaging frame) and `sess.timeseries` from suite2p F/Fneu | NWB behavioural series are exactly that aligned product; F/Fneu read from `processing/ophys` | yes (the NWB files are the published form of `sess`) |
| (b) neuron filtering | `iscell` from suite2p manual curation; `spatial.is_putative_interneuron(ts_key='dff', method='speed', r_thresh=0.5)` | identical rules (iscell + dF/F-speed r > 0.5); plus an extra numerical-stability rule (25 cells, 0.018%) for cells whose maximin baseline collapses to ~0 | yes + documented addition |
| (c) temporal alignment | trial window `[trial_start-1, teleport-1)` in `preprocessing.dff` and `glmUtils.get_timeseries_data`; all streams share the imaging frame clock | identical window, identical clock, and inputs/outputs are taken from exactly the same frames as the neural data | yes |
| (d) binning | no re-binning; everything at the ~15.5 Hz frame rate (paper: all time series sampled at ~15.5 Hz) | one frame per time bin, 64.4836 ms | yes |
| (e) input construction | `glmUtils.get_timeseries_data` builds trials/pos/speed/licks/rewarded per frame | same source variables; decoder inputs are the four required by the task (time from trial start, environment (= `morph`), lap index (= `trials`), previous-lap outcome (= `isreward` shifted by one)) | yes, restricted to the task spec |
| (f) output construction | `rel_pos` = distance from `rzone[i,0]` (`behavior.get_reward_zones`); `pos`; `speed`; binary `licks` (`licks[licks>1]=1`); `isreward` (`behavior.get_trial_types`); `rz label` | same variables, with the discretisation required by the task; distance is to the nearest point of the 50 cm zone (task asks for distance to *any* location in the zone) rather than to the zone start | yes, with the task-mandated discretisation |

Differences and their reasons:
1. **No 2 cm/s speed threshold.** The reference drops sub-threshold samples for place-cell, GLM and decoder
   analyses. Here speed is an output with an explicit `< 2 cm/s` class and the decoder needs contiguous trial
   time series, so all within-trial samples are kept.
2. **Distance measured to the nearest point of the reward zone** (0 throughout the zone) instead of the
   reference RR position relative to the zone start, because the task specifies bins around 0 for the zone.
3. **Linear (not circular) distance**: the paper decoder used circular RR coordinates; the task specifies
   linear signed distance bins in cm.
4. **Extra exclusions**: 81 lick-error laps (the reference NaNs their lick values, which is not possible for a
   required output) and the first lap of each session (no previous-lap outcome).
5. **Extra neuron rule** for numerically unstable dF/F baselines (0.018% of cells), needed because the decoder
   applies PCA to raw event amplitudes.

### Check 4: Key statistics comparison
See the table in Step 9. Highlights: 11 subjects, 152 sessions (14/mouse, m11 12), 12,216 complete laps in the
data vs 12,376 imaged trials quoted by the paper (the NWB release contains 160 fewer laps), 80.37 +/- 6.16 laps
per session vs 80.5 +/- 7.4 in the paper, 155-2341 curated cells per session vs 155-2172 quoted, exactly 81
lick-sensor-error laps as in the paper, 84.7% rewarded laps (paper: omissions on a ~15% subset), interneuron
exclusion 0.29% (paper 0.42 +/- 0.85%), frame period 64.4836 ms (paper 0.0645 s), reward zones at 80/200/320 cm
verified against the measured zone-entry position (median error 0.78 cm).

### Check 5: Edge cases
- Frame-count mismatch between ophys and behaviour in 10 sessions: detected and handled by truncation; the
  affected sessions now convert and pass the independent sanity checks (m18 ses-13 above).
- `trial_start` at frame 0 would make the `[start-1, ...)` window invalid: verified never to occur.
- Trials never contain pre-synchronisation samples (`trial number` = -1, position = -500), non-scanning samples,
  or teleport-period samples (position = -50); asserted on all 152 sessions.
- Sessions with fewer than 2 usable trials would break decoder validation: the conversion drops such sessions
  (none occurred; the smallest session keeps 39 laps).
- Non-finite event values would fail format verification: each trial is checked with `np.all(np.isfinite(...))`
  before being stored (none were dropped).
- Laps beyond lap 30 in switch sessions get the post-switch zone; sessions with fewer than 30 laps (m4 ses-04,
  41 laps) are handled because the zone list is built with `min(CHANGE_TRIAL, n_trials)`.
- The last lap of a session is complete (paired teleport) in every session, so no partial lap is included.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
(ran to completion on the GPU (NVIDIA L4); log in `/app/train_decoder_full_out.txt`, figures
`sample_trials.png` and `predictions.png`).

### Training Progress
- Loss decreasing: Yes. 2.377 (epoch 10) -> 1.429 (50) -> 1.185 (100) -> 1.043 (150) -> 0.935 (200);
  held-out test loss 0.799 (below the final training loss, so no overfitting).

### Decoder Results (Full: 152 sessions, 11,983 trials, 138,244 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Validation / chance | Notes |
|--------|--------|-------------|--------|-------|-------|
| reward_zone_distance (7 classes) | 0.143 | 0.611 | 0.534 | 3.7x | strongest relative decoding, as expected from the paper reward-relative code |
| position (5 classes) | 0.200 | 0.698 | 0.645 | 3.2x | classic CA1 place coding |
| speed (5 classes) | 0.200 | 0.630 | 0.583 | 2.9x | |
| lick (2 classes) | 0.500 | 0.780 | 0.758 | 1.5x | |
| reward_zone_location (3 classes) | 0.333 | 0.891 | 0.854 | 2.6x | per-trial label; identifiable from the session/zone-specific code |
| reward_outcome (2 classes) | 0.500 | 0.798 | 0.590 | 1.18x | weakest; see Step 12 Check 1 for the analysis |

Comparison with the paper: the paper only reports a circular decode score (cosine similarity, 1 = perfect,
0 = chance) for reward-relative position, not classification accuracies, so no direct numeric comparison is
possible. Qualitatively the ordering matches the paper findings: reward-relative distance and track position
are robustly decodable from CA1 deconvolved activity, well above chance.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Chance | Validation acc | Ratio | Assessment |
|---|---|---|---|---|
| reward_zone_distance | 0.143 | 0.534 | 3.74x | good |
| position | 0.200 | 0.645 | 3.23x | good |
| speed | 0.200 | 0.583 | 2.92x | good |
| lick | 0.500 | 0.758 | 1.52x | good (binary variable, ceiling is lower) |
| reward_zone_location | 0.333 | 0.854 | 2.56x | good |
| reward_outcome | 0.500 | 0.590 | 1.18x | below 1.5x -> investigated below |

No output is below chance.

**Investigation of `reward_outcome` (`/app/cache/diag_outcome.py`, `/app/cache/diag_outcome_decode.py`).**
1. Definition check on the raw data: across all 12,216 laps there is **no** lap with a delivered reward but no
   reward-zone flag, and 52 laps with a zone flag but no reward (the reference `get_omission_trials` calls these
   lapsed trials, distinct from true omissions). So my `isreward` equals the reference
   `behavior.get_trial_types` rule and equals simply whether a reward was delivered.
2. `autoreward` is identically zero in every NWB file, so no extra information is available there.
3. Causal structure: on rewarded laps the reward is delivered on average 40.7% of the way through the lap
   (median 40.4%). Before that moment the animal has not yet experienced the outcome, so a frame-wise decoder
   cannot know it; the label is nevertheless defined for the whole lap because it is a per-trial variable.
4. Direct test: a 19-session decoder run (`diag_outcome_decode.py`) split validation frames by reward-zone
   distance bin. Balanced accuracy for `reward_outcome` was **0.531 before the reward zone** (distance bins 0-2)
   and **0.637 in/after the reward zone** (bins 3-6), versus 0.598 overall. The information is therefore present
   in CA1 activity only once the animal reaches the zone, exactly as expected for the physiology (the paper
   Fig. 6 contrasts rewarded vs omission trials *after* reward-zone entry). The modest overall number is a
   property of the task and of a frame-wise per-trial label, not a conversion error.
5. Class balance is 84.3% rewarded / 15.7% omitted, which is the true behavioural rate (paper: omissions on a
   ~15% subset of laps), so this is not a degenerate label either.

### Check 2: Accuracy comparison to paper
Every decoding result in the paper (Fig. 3, Extended Data Fig. 3) is expressed as a circular-linear decode
score (mean cosine of the angular error) or as a z-score relative to a 100-shuffle null, for the decoding of
reward-relative position from selected cell classes (RR, track, non-RR remapping). The paper reports no
classification accuracy for any variable, and it never decodes position, speed, licking, reward-zone identity
or reward outcome. A numeric table comparing accuracies is therefore not possible. The qualitative expectation
from the paper - that reward-relative position is decodable well above chance from deconvolved CA1 events -
is reproduced (3.7x chance, the best of all outputs in relative terms).

### Check 3: Train vs validation gap
| Output | Train | Validation | Train/Val |
|---|---|---|---|
| reward_zone_distance | 0.611 | 0.534 | 1.14 |
| position | 0.698 | 0.645 | 1.08 |
| speed | 0.630 | 0.583 | 1.08 |
| lick | 0.780 | 0.758 | 1.03 |
| reward_zone_location | 0.891 | 0.854 | 1.04 |
| reward_outcome | 0.798 | 0.590 | 1.35 |
All ratios are below 1.5, so there is no evidence of severe overfitting or data leakage; the largest gap
(reward_outcome, 1.35) is expected for a per-trial label with only ~79 labels per session (the network can
partly memorise training laps, but cannot do so for held-out laps).

### Additional debugging checks performed
- Output values verified against the raw NWB data for three whole sessions, trial by trial (Step 10, Check 2).
- Temporal alignment verified visually (`processing_*_timeseries.png`, `processing_*_outputs.png`): the neural
  raster, position, speed and licks of a single trial are plotted on the same time axis and the trial
  boundaries coincide; also verified numerically because inputs, outputs and neural data are extracted from the
  identical frame index range.
- Output variability: no output is degenerate (the most imbalanced is reward_outcome at 84/16).
- Neural filtering follows the reference (iscell + speed-correlated interneurons) and was verified by exact
  recomputation of the kept-cell counts.
- Processing matches the reference dF/F + OASIS pipeline (verified by an independent re-implementation).

### Issues Found and Resolved (all iterations)
1. **Ophys/behaviour frame-count mismatch** (10 m17/m18 sessions) crashed the first full run -> streams are now
   truncated to the common length; those sessions convert and pass sanity checks.
2. **Numerically exploding dF/F** for 25 cells whose maximin baseline collapses to ~0 -> those cells are
   excluded; dF/F range fell from [-931, 3013] to [-16.0, 37.2].
3. **Sample decoder log** initially only contained the tail of the run -> re-run and saved in full.
After each fix, the conversion, verification and decoder-training steps were re-run and all checks of Step 10
were repeated (39/39 independent sanity checks pass; verification reports no errors or warnings).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format specification, key statistics,
      processing summary and decoder performance).
- [x] `cache/` folder created with all exploration/validation scripts and `cache/README_CACHE.md` describing
      each file.
- [x] All required deliverables present in `/app`:
  `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl` (9.47 GB), `sample_data.pkl`, `README.md`,
  `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
  `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`.
- [x] Extra artefacts: `converted_data_session_stats.csv` and `sample_data_session_stats.csv` (per-session
      conversion statistics), `processing_m11_ses-04_*.png` and `processing_m19_ses-02_*.png`
      (`--show-processing` figures), `sample_trials.png` and `predictions.png` (decoder figures).
- [x] `__pycache__` removed.

### Key decisions recap
1. Neural signal = OASIS-deconvolved events recomputed from NWB F/Fneu with the reference per-trial maximin
   dF/F pipeline (the stored `Deconvolved` series is suite2p deconvolution of raw F and does not match the
   reference processing).
2. No re-binning: one imaging frame (64.4836 ms) per time bin, matching the reference and preserving exact
   alignment between neural and behavioural streams.
3. Trials = complete laps with the reference window `[trial_start-1, teleport-1)`, aligned to trial start.
4. Curation = suite2p `iscell` + interneuron exclusion (dF/F-speed r > 0.5) + 25 numerically unstable cells;
   81 lick-sensor-error laps and the 152 first laps (undefined previous outcome) dropped.
5. All per-trial variables stored as constant time series so every trial is (4, T) input / (6, T) output.
6. Deliberate deviation: no 2 cm/s speed threshold (speed is a decoder output and trials must be contiguous).
