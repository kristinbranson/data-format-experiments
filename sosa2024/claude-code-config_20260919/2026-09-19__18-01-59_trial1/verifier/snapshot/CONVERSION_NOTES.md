# Dataset Conversion Notes

## Overview
- **Dataset**: "A flexible hippocampal population code for experience relative to reward" (paper.pdf in /app)
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file)
- `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `code/` — reference code repo (`reward_relative` package; docs; notebooks)
- `data/` — NWB dataset, 11 subject dirs (`sub-m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19`),
  each with 12-14 `sub-<id>_ses-<NN>_behavior+ophys.nwb` files. **152 NWB files, 87 GB total.**
- `decoder.py`, `train_decoder.py` — provided decoder validation code
- `methods.txt`, `paper.pdf` — reference text

Python environment verified: numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0, pynwb 4.1.0.

Per-subject session counts: m11:12, all others:14 -> 152 sessions total.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference repo = `Sosa_et_al_2024` (Sosa, Plitt & Giocomo 2025, *Nat Neurosci*), package
`reward_relative` in `/app/code/src/`. The upstream `TwoPUtils` package (which builds the
`sess` class from raw ScanBox + Unity SQLite files) is **not** included, but the NWB files on
DANDI already contain the product of that stage (VR data resampled onto imaging frames).

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `preprocessing.py` | LOADING | Builds `sess` from raw scan + VR; calls `TwoPUtils.sess.Session.align_VR_to_2P()`. Superseded by the NWB packaging. |
| `vr_align_to_2P` (TwoPUtils, described in `docs/multi_anim_sess_README.md`) | external | LOADING/ALIGNMENT | Resamples the Unity VR stream onto the imaging frame grid (~15.5 Hz). pos/time linearly interpolated; morph/trialnum/scanning nearest-neighbour; lick/reward/tstart/teleport/rzone integrated-then-differenced so no events are lost; speed = smoothed dz / dt. **This is exactly what the NWB `BehavioralTimeSeries` holds.** |
| `preprocessing.dff` | `preprocessing.py` | PROCESSING | dF/F per ROI. Neuropil subtraction (`neu_coef=0.7`), per-trial `maximin` baseline (gaussian sigma=15 frames -> `minimum_filter1d(300)` -> `maximum_filter1d(300)`, 300 frames ~= 19.3 s ~ "20 s window"), `dff=(F-base)/|base|`, then Gaussian smooth sigma=2 frames, then OASIS deconvolution (`suite2p.extraction.dcnv.oasis`, tau, fs=frame_rate/n_planes). Data **outside trials is set to NaN** so baselines are computed within-trial only. |
| `utilities.multi_anim_sess` | `utilities.py` | PROCESSING/CURATION | Top-level per-day pipeline: calls `pp.dff(...)` with `neuropil_method='subtract'`, `baseline_method='maximin'`, `deconvolve=True`, then computes place cells and trial metadata. |
| `utilities.default_dff_method` | `utilities.py` | PROCESSING | `{neuropil_method_red:'subtract', baseline_method:'maximin', neu_coef:0.7, keep_teleports:False}` |
| `behavior.get_trial_types` | `behavior.py` | PROCESSING | Per-trial `isreward` = `any(reward>0) AND any(rzone>0)` inside `[trial_start_ind, teleport_ind)`; per-trial `morph` = unique VR environment on that trial. |
| `behavior.get_reward_zones` | `behavior.py` | PROCESSING | Per-trial reward-zone `[start,stop]` coords and labels from the **scene name** in the session metadata; on `X_to_Y` scenes the zone changes at `change_trial=30`. Coordinate dict: label A -> `'X'` = [80,130], B -> `'Y'` = [200,250], C -> `'Z'` = [320,370] cm. |
| `behavior.correct_lick_sensor_error` | `behavior.py` | CURATION | Sets a trial's licks to NaN when `mean(lick_count > 2) > thr` over the trial (methods text: thr = 0.30). |
| `behavior.define_trial_subsets` | `behavior.py` | PROCESSING | Splits trials into pre/post-switch sets (not needed for this conversion). |
| `teleport_metadata.teleport_sessions` | `teleport_metadata.py` | PROCESSING | Animal/day combinations for which the laser stayed on during the teleport ITI -> `keep_teleports=True` in `dff`. GCAMP11-14: days [1,7,8,14,15]; GCAMP15,17,18,19: days [1,3,5,7,8,10,12,14,15,16,17]. |
| `sessions_dict.single_plane/multi_plane` | `sessions_dict.py` | LOADING | Per-animal session metadata incl. **scene name** and 1-indexed `exp_day`. The NWB `identifier` string carries the same scene, and `general/session_id` the same exp_day. |
| `decode.CircularRegression` | `decode.py` | ANALYSIS | The paper's own decoder (circular-linear regression) -- predicts reward-relative position from the **deconvolved** event timeseries. Not used here (we use the provided `train_decoder.py`), but it tells us which neural signal the authors decode from. |
| suite2p notebook `example_m12.ipynb` | `notebooks/` | LOADING | suite2p ops actually used: `tau=0.7`, `fs=info['frame_rate']`, `nchannels=1`, `functional_chan=1`. |
| `make_multi_anim_sess.md` | `notebooks/` | PROCESSING | The exact dff kwargs used for the paper: `neuropil_method_red='subtract'`, `baseline_method='maximin'`, `neu_coef=0.7`, `regress_*=False`, `keep_teleports` from `teleport_metadata`, `calc_spks=True`. |

### Notes
* **Imaging data need dF/F computed from scratch**: the NWB stores raw suite2p `Fluorescence`,
  `Neuropil` and suite2p's own `Deconvolved` (which suite2p ran on raw F, *not* on the authors'
  custom dF/F). The paper's signal chain is F -> neuropil-subtract -> per-trial maximin dF/F ->
  smooth -> OASIS. So we reimplement `preprocessing.dff` on the NWB arrays.
* **Cell curation**: (a) suite2p manual curation is already baked in as `iscell`; (b) the paper
  additionally removes putative interneurons with Pearson r > 0.5 between dF/F and running speed.
* This is 2P imaging, not ephys, so there is no spike-sorting quality filter.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/` is DANDI dandiset 001361 laid out as `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`
(NWB 2.8.0, HDF5, uncompressed/unchunked). 11 subject directories, 152 files, 87 GB.

Contents of each file (verified with h5py):

| NWB path | Contents |
|---|---|
| `identifier` | e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A` -> animal, date and **scene** (reward-zone condition) |
| `general/session_id` | zero-padded **experiment day** (`'03'`), matching `sessions_dict.py`'s `exp_day` |
| `general/subject/subject_id` | `m11`, ... (`GCAMP11` in the reference code) |
| `general/optophysiology/ImagingPlane/location` | `hippocampus, CA1` (all sessions) |
| `general/optophysiology/ImagingPlane/imaging_rate` | 15.5078125 (1-plane animals) or 31.015625 (2-plane animals m17, m18) -- the latter is the *scan* rate; **per-plane rate is 15.5078125 Hz for every session** (behaviour timestamps have dt = 0.06448363 s everywhere). |
| `processing/behavior/BehavioralTimeSeries/{position, speed, lick, reward_zone, environment, trial number, trial_start, teleport, autoreward, scanning}` | one value per imaging frame, with `timestamps` |
| `processing/behavior/BehavioralTimeSeries/Reward` | sparse: `timestamps` of each 0.004 mL reward delivery |
| `processing/ophys/Fluorescence/plane{0,1}/data` | raw suite2p F, shape (T, n_roi_in_plane) |
| `processing/ophys/Neuropil/plane{0,1}/data` | suite2p neuropil F, same shape |
| `processing/ophys/Deconvolved/plane{0,1}/data` | suite2p's own deconvolution of raw F (not the paper's signal) |
| `processing/ophys/ImageSegmentation/PlaneSegmentation/{iscell, planeIdx, pixel_mask}` | `iscell[:,0]` = manual curation flag (0/1), `iscell[:,1]` = classifier prob; `planeIdx` = imaging plane; rows are ordered plane0 then plane1 |

Behaviour column semantics come from `docs/multi_anim_sess_README.md`: `lick` is a cumulative lick
count per imaging frame, `reward_zone` (`rzone`) a cumulative reward-zone-entry flag, `speed` is
smoothed cm/s, `environment` (`morph`) is 0 = ENV 1 / 1 = ENV 2, and values are -1 / -500 before the
VR-imaging TTL sync starts.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, `iscell==1`) | 138,678 |
| Neurons / session | 155 - 2,341 (mean 912) |
| ROIs before `iscell` | 312,110 |
| Subjects | 11 (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) |
| Sessions / subject | 14 for all except m11 (12: days 3-14) -> 152 sessions |
| Trials (total) | 12,216 |
| Trials / session | 41 - 100, mean 80.37 |
| In-trial imaging samples (total) | 2,620,514 (mean 214.5 samples = 13.8 s per trial) |
| Frame period | 64.4836 ms (15.5078125 Hz) for every session |

Data-integrity checks run over all 152 files:
* `sum(trial_start) == sum(teleport)` in every session; every trial start is followed by a teleport.
* In 10 sessions the fluorescence arrays are 1 frame longer than the behaviour arrays (trailing
  frame); everywhere else lengths match -> truncate to the common length.
* Within every trial window `[trial_start, teleport)`: `scanning == 1` everywhere, no NaNs in
  position/speed/lick/environment, `environment` in {0,1}. Position runs ~0 -> ~450 cm
  (global range -2.74 to 451.83).
* `position[trial_start - 1] < 0` and `position[teleport - 1] ~= 448` -> the trial window
  `[trial_start, teleport)` is exactly the on-track portion of the lap.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons / session | 155-2,172 | "yielded 155-2172 putative pyramidal neurons per session" |
| Subjects (switch task) | 11 | "Each mouse encountered a different starting reward zone ... (n = 11 mice)" |
| Subjects (fixed-condition, excluded from DANDI) | 3 | "An additional 'fixed-condition' cohort (n = 3 mice)" |
| Sessions / subject | 14 (m11: from day 3) | "for a total of 14 days"; "imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Trials (total, 11 switch mice) | 12,376 | "n = 81 out of 12,376 trials removed across 11 switch mice" |
| Trials / session | 80.5 +/- 7.4 | "mean +/- s.d., 80.5 +/- 7.4 trials across 14 mice, all imaging days" |
| Neural + behaviour time bin | ~64.5 ms (~15.5 Hz) | "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate"; "the 0.0645 s imaging frame samples" |
| Reward rate | ~85% (omission ~15%) | "the reward was randomly omitted on ~15% of trials" |
| Track length | 450 cm | "unidirectional 450 cm virtual linear track" |
| Reward zones | A 80-130, B 200-250, C 320-370 cm | "zone A, 80-130 cm; zone B, 200-250 cm; zone C, 320-370 cm" |
| Reward-zone switch trial | after 30 trials | "Each switch occurred after 30 trials." |
| Lick-sensor-error trials | 81 (0.65%) | "~0.65% of all imaged trials, n = 81 out of 12,376 trials removed" |
| Interneurons excluded | 0.42 +/- 0.85% of cells | "a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed" |
| Speed threshold (spatial analyses only) | 2 cm/s | "we excluded activity when the animal was moving at <2 cm s-1" |
| Environments | 2 (ENV1/ENV2); 9 mice start ENV1, m17 & m18 start ENV2 | "two mice (m17 and m18) began in ENV 2" |

### Processing Details
* **Temporal alignment**: everything already lives on the imaging frame grid (VR resampled to 2P by
  `vr_align_to_2P`). Trials are laps: `[trial_start, teleport)`.
* **Temporal binning**: native imaging frame, 64.4836 ms. The paper's GLM and decoder both operate
  at this rate.
* **dF/F**: per-trial maximin baseline, 20 s window, `(F-base)/|base|`, Gaussian smoothed with a
  2-sample (~0.129 s) s.d. kernel, then OASIS-deconvolved into "events".
* The paper's **own decoder** (Fig. 3) is fit on "the deconvolved calcium event timeseries".

### Curation Steps
**Neuron curation rules**:
1. suite2p manual curation -> `iscell == 1` (removes dendrites, multi-soma ROIs, obvious interneurons).
2. Remove putative interneurons: Pearson r(dF/F, speed) > 0.5.

**Trial curation rules**:
1. Trials with lick-sensor failure (>30% of the trial's frames have cumulative lick count > 2) are
   invalidated (reference sets licks to NaN). 81 trials expected.
2. No other trial exclusions in the paper. (The <2 cm/s speed mask applies to *spatial* analyses
   only, not to whole trials.)

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Reward-relative position (circular, Fig. 3) | reported as a "decode score" = mean cos(true - predicted) on [-1, 1], ~0.5-0.8 for an example session, z-scored against a circular-shift shuffle across sessions. **No classification accuracies are reported anywhere in the paper**, so there is no directly comparable number for the outputs decoded here. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

Cross-checks run against all 152 NWB files (`/app/cache/behav_scan.py`, `/app/cache/scan*.py`):

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Reward zone per trial | scene name + switch at trial 30; A=[80,130], B=[200,250], C=[320,370] | Position at the first `reward_zone` flag of each trial is within **8.5 cm** of the predicted zone start for **every trial of every session** (max abs error 8.5 cm, i.e. inside the 50 cm zone) | same coords, "each switch occurred after 30 trials" | **Consistent.** Use the reference rule (scene + trial 30); verified empirically. |
| Trial count | n/a | 12,216 | 12,376 | 160 trials (~1.05/session) fewer in NWB. The NWB packaging contains only complete laps (`#trial_start == #teleport` in every file, nothing on-track after the last teleport). `vr_align_to_2P` force-completes a clipped final lap, so the paper's count includes ~1 partial lap per session. Not recoverable from the NWB and not desirable to recover (a lap with no teleport has no defined end). Documented, not "fixed". |
| Lick-sensor error trials | `mean(lick>2) > thr` per trial | **81 / 12,216 trials (0.66%)** with thr = 0.30 | "n = 81 ... (~0.65%)" | **Exact match** -> confirms both the criterion and the trial segmentation. |
| Reward rate | `any(reward) AND any(rzone)` per trial | 84.66% rewarded | "~15%" omitted | **Consistent.** |
| Trials/session | n/a | 80.37 +/- 6.14 | 80.5 +/- 7.4 | **Consistent** (small shift from the missing partial laps). |
| Neurons/session | `iscell==1` | 155 - 2,341 | 155 - 2,172 | Lower bound matches **exactly**. Upper bound is 169 cells higher (m18, day 3). Interneuron removal takes it to 2,323, still above 2,172. No de-duplication of ROIs across the two imaging planes exists anywhere in the repo, and the paper states planes are pooled. Most likely the paper quotes a slightly earlier curation snapshot. Documented; no action (removing cells to hit a number would be unjustified). |
| Environment (morph) | per-trial `unique(morph)` | 73 sessions all ENV1, 68 all ENV2, **11 sessions contain both** | day 8 is the single cross-environment switch day, 11 mice | **Consistent** (exactly one cross-env session per mouse). |
| Interneuron fraction | r(dF/F, speed) > 0.5 | 0.65% (m11 d3), 0.77% (m18 d3), 3.71% (m3 d1) | 0.42 +/- 0.85% | Same order of magnitude; the s.d. of 0.85% across mice/days implies a heavy tail. Full-dataset value reported in Step 9. |
| Frame rate | `frame_rate / n_planes` | behaviour dt = 0.06448363 s in every session; 2-plane animals store the same number of per-plane frames | "~15.5 Hz" | **Consistent** -> one common time bin for the whole dataset. |
| Neural signal for decoding | `decode.py` fit on deconvolved events | NWB `Deconvolved` is suite2p's own deconvolution of **raw F**, not of the paper's dF/F | "activity rate was extracted by deconvolving dF/F ... using OASIS" | Recompute dF/F + OASIS from `Fluorescence`/`Neuropil` rather than using the stored `Deconvolved`. |
| suite2p `tau` | `sess.s2p_ops['tau']`, not in NWB | n/a | n/a | The repo's suite2p notebook sets `tau=0.7` for these animals, which is also `preprocessing.dff`'s default. Use 0.7. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Unit of a "trial"
A trial is one lap: NWB sample indices `[trial_start_idx, teleport_idx)`, i.e. exactly the on-track
portion (position ~0 -> ~450 cm). This is the reference's `sess.trial_start_inds` /
`sess.teleport_inds` pair. Alignment event = **start of trial** (entry to the track), so
`off_start = 0.0` s and `off_end = None` (laps have variable length, 6.2 s - 217 s).

### Variable Mapping
| Source (NWB) | Target field | Transform | Reference code | Notes |
|---|---|---|---|---|
| `ophys/Fluorescence/plane*/data`, `ophys/Neuropil/plane*/data`, `ImageSegmentation/.../iscell` | `neural` | keep `iscell==1` ROIs, pool planes; neuropil-subtract (0.7); per-trial maximin dF/F (sigma 15 -> min/max filter 300 frames); smooth sigma 2; OASIS (tau 0.7, fs 15.5078125); drop r(dF/F,speed)>0.5 cells; slice `[start, teleport)` | `preprocessing.dff`, `utilities.multi_anim_sess`, methods "Calcium data processing" | shape (n_neurons, T_trial), float32 |
| frame index | `input[0]` `time_from_trial_start_s` | `(i - trial_start_idx) / 15.5078125` seconds | -- | time-varying |
| `behavior/environment` | `input[1]` `environment` | per-trial mode of `morph` (0 = ENV1, 1 = ENV2) | `behavior.get_trial_types` | per-trial, broadcast |
| trial index in session | `input[2]` `trial_number` | 0-based lap index (verified equal to NWB `trial number` at trial start) | -- | per-trial, broadcast |
| reward delivery of previous lap | `input[3]` `previous_trial_outcome` | `isreward[k-1]`; for the first lap of a session -> 1 | `behavior.get_trial_types` | per-trial, broadcast |
| `behavior/position` + reward-zone coords | `output[0]` `distance_to_reward_zone` | signed distance to nearest point of the active zone, then 7 bins | `behavior.get_reward_zones` | time-varying |
| `behavior/position` | `output[1]` `track_position` | 5 equal 90 cm bins over the 450 cm track | -- | time-varying |
| `behavior/speed` | `output[2]` `speed` | 5 bins: <2, 2-10, 10-20, 20-40, >40 cm/s | 2 cm/s is the paper's own speed criterion | time-varying |
| `behavior/lick` | `output[3]` `lick` | cumulative per-frame lick count -> binary (>0) | README: "anything >1 gets set to 1"; methods: "lick counts were converted to a binary vector" | time-varying |
| scene in `identifier` (+ switch at trial 30) | `output[4]` `reward_zone_location` | A -> 0, B -> 1, C -> 2 | `behavior.get_reward_zones` | per-trial, broadcast |
| `behavior/Reward/timestamps` + `behavior/reward_zone` | `output[5]` `reward_outcome` | `any(reward in lap) AND any(rzone>0 in lap)` | `behavior.get_trial_types` | per-trial, broadcast |
| `subject_id` | `subjects` / `subject_idx` | 11 mice | -- | |
| `ImagingPlane/location` | `brain_regions` = `['CA1']` | all neurons index 0 | paper: dorsal CA1 only; "planes were pooled for all analyses" | |

### Discretisation (bin edges exactly as specified in the Decoder Task)
* `distance_to_reward_zone` d (cm; d<0 before the zone, 0 inside, d>0 past the zone):
  0: d < -50 | 1: -50 <= d < -10 | 2: -10 <= d < 0 | 3: d == 0 (inside zone) | 4: 0 < d <= 10 |
  5: 10 < d <= 50 | 6: d > 50.
* `track_position` p (cm): 0: p < 90 | 1: 90 <= p < 180 | 2: 180 <= p < 270 | 3: 270 <= p < 360 | 4: p >= 360.
* `speed` v (cm/s): 0: v < 2 | 1: 2 <= v < 10 | 2: 10 <= v < 20 | 3: 20 <= v < 40 | 4: v >= 40.
* `lick`: 0 / 1.

### Key Decisions
1. **Neural signal = OASIS-deconvolved events computed from the authors' dF/F** (not the NWB
   `Deconvolved`, which suite2p ran on raw F, and not raw F). Rationale: the paper's own decoding
   (Fig. 3) and GLM are both fit on "the deconvolved calcium event timeseries", and the deconvolution
   removes the asymmetric calcium decay that would otherwise smear a memoryless per-timepoint
   decoder. dF/F is produced by the same code path and is checked as an alternative in Steps 7-8;
   the final choice is made on measured decoder accuracy and documented there.
2. **Time bin = the native imaging frame, 64.4836 ms**, identical for all sessions and animals
   (per-plane rate is 15.5078125 Hz everywhere, including the 2-plane mice). This is the rate the
   paper's GLM/decoder use ("All behavioral and neural time series were sampled at ~15.5 Hz"), so no
   re-binning is applied and no information is discarded.
3. **Trial window `[trial_start, teleport)`**, not the reference's `[start-1, stop-1)`. The reference
   `dff` shifts every trial window one sample earlier, which includes one pre-track frame (position
   < 0) and drops the last on-track frame. Because we must align behaviour to neural activity
   sample-by-sample, we use the behaviourally exact window; the only consequence for dF/F is a
   one-frame difference in the baseline segment, which is negligible against a 300-frame filter.
4. **Neuron curation**: `iscell == 1` (the authors' manual suite2p curation, already in the file) and
   removal of putative interneurons with Pearson r(dF/F, speed) > 0.5, computed over all in-trial
   samples of the session -- both rules straight from the methods.
5. **Trial curation**: drop the 81 lick-sensor-failure trials (criterion reproduces the paper's count
   exactly). The reference NaNs out their licks; since `lick` is one of our decoder outputs and NaNs
   are not permitted, dropping the trial is the faithful equivalent. No other trials are dropped.
6. **No <2 cm/s speed mask.** The paper applies it to *spatial* (position-binned) analyses only.
   Here speed is a decoder output whose lowest class is "<2 cm/s", so masking those samples would
   delete an entire output class. Documented deviation, required by the Decoder Task spec.
7. **Teleport/ITI periods are excluded** (trials end at `teleport`), matching the reference, which
   also blanks fluorescence outside laps. `keep_teleports` (reference: ITI included in the *baseline*
   segment for some animal/day combinations) is implemented so the dF/F baseline matches the paper
   for those sessions; the extracted samples are in-trial either way.
8. **First lap's `previous_trial_outcome` = 1 (rewarded).** Each imaging session was preceded
   immediately by 30 warm-up laps on the previous day's reward zone, on which reward was delivered
   unless randomly omitted (~15%), so "rewarded" is the expected value. Affects 152 of 12,135 trials.
9. **Per-trial variables are broadcast across the trial's timepoints** so that `input` is (4, T) and
   `output` is (6, T); the format requires one array per trial and we mix time-varying with
   per-trial quantities.
10. **`trial_number` and `previous_trial_outcome` are computed before any trial is dropped**, so lap
   numbering and outcome history stay faithful to what the animal experienced.

### Planned Sanity Checks
- [x] Reward-zone label from scene+trial-30 rule vs. empirical position of the first `reward_zone`
      flag on every trial (done in Step 4: max error 8.5 cm over all 12,216 trials).
- [x] Lick-sensor-error trial count == 81 (done in Step 4).
- [x] Reward rate ~= 85% (done in Step 4: 84.66%).
- [ ] Totals: 152 sessions, 11 subjects, 12,135 trials after curation, ~138k - interneurons neurons.
- [ ] Trials/session mean ~= 80.5 +/- 7.4.
- [ ] Interneuron removal fraction ~ 0.4-1% on average.
- [ ] Independent re-load sanity checks (Step 10) comparing converted `neural`, `input` and `output`
      against values read directly out of the NWB with h5py, using `np.allclose`.
- [ ] Input ranges: time from 0 to trial length; environment in {0,1}; trial_number 0..99;
      previous outcome in {0,1}.
- [ ] Output distributions: `distance_to_reward_zone` should have ~11% of samples in class 3
      (inside the 50 cm zone of a 450 cm track, modulated by occupancy); `track_position` roughly
      uniform-ish with more mass in slow (pre-reward) segments; `reward_outcome` ~85% class 1;
      `reward_zone_location` spread over A/B/C.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Run as
`python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
[--signal dff|events] [--workers N] [--session-list ids]`.

Structure:
| Function | Role |
|---|---|
| `reward_zone_labels(scene, ntrials)` | port of `behavior.get_reward_zones`: per-trial A/B/C from the scene name, switching after 30 laps |
| `compute_dff(F, Fneu, segments)` | port of `preprocessing.dff` (maximin, single channel): neuropil subtraction, per-segment baseline, smoothing, OASIS |
| `load_session(path)` | reads one NWB: pools planes, keeps `iscell==1`, behaviour columns, trial indices, reward times; asserts trial/teleport consistency and truncates to the common length |
| `convert_session(path)` | baseline segments (honouring `teleport_metadata`), dF/F + events, interneuron removal, per-trial task variables, discretisation, trial assembly |
| `bin_distance_to_reward` / `bin_position` / `bin_speed` / `signed_distance_to_zone` | output discretisation |
| `plot_processing(...)` | the `--show-processing` figures |
| `main()` | parallel driver, summary statistics, pickle writing |

Code efficiencies:
* Per-session work is fully vectorised over neurons; the only Python loop is over laps
  (~80 per session), which is unavoidable because baselines are per-lap.
* `ProcessPoolExecutor` over sessions (default 10 workers); sessions are independent.
* float32 everywhere (halves memory/IO versus the reference's float64) -- verified in Step 10
  to make no material difference to dF/F.
* ROI selection is applied at read time so only `iscell` columns are kept in memory.
* Timing is printed per session (load / dF/F / total) and for the whole run.

Inefficiency identified and fixed: the h5py read `dset[:, :][:, keep]` materialises the whole
(T, n_roi) block before subsetting. Fancy-indexing h5py directly is *much* slower here (the files
are contiguous and unchunked), so the full read is deliberate -- 1.8 s for the largest session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
-> `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`.
Sample = `sub-m11_ses-03` (smallest session, 154 neurons, 1 plane, laser blanked in the ITI) and
`sub-m18_ses-03` (largest session, 2321 neurons, 2 planes, laser on through the ITI) -- chosen to
exercise both ends of every code path.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 2,475 (154 + 2,321) |
| Neurons / session | 154, 2,321 |
| Subjects | 2 (m11, m18) |
| Trials (total) | 156 kept of 160 (4 dropped: lick-sensor error) |
| Trials / session | 80, 76 |
| Timepoints | 34,850 (mean T = 224.6, min 137, max 479) |
| interneurons removed | 21 / 2,496 (0.84%) |
| time_from_trial_start_s | [0.0, 30.8] |
| environment | [0, 1] |
| trial_number | [0, 79] |
| previous_trial_outcome | [0, 1] |
| distance_to_reward_zone | [0.178, 0.091, 0.050, 0.276, 0.022, 0.075, 0.308] |
| track_position | [0.244, 0.207, 0.276, 0.138, 0.134] |
| speed | [0.123, 0.067, 0.141, 0.341, 0.328] |
| lick | [0.793, 0.207] |
| reward_zone_location | [0.438, 0.562, 0.000] (only A and B occur in these two sessions) |
| reward_outcome | [0.123, 0.877] |

`verify_data_format` on the sample: **no errors, no warnings**
(`/app/verification_sample_out.txt`).

### Processing Plots Review
`processing_sub-m11_ses-03.png`, `processing_sub-m18_ses-03.png` (9 stacked panels sharing a time
axis, first 6 laps) and the corresponding `_summary.png`:
* Raw F for **m11 day 3** collapses to ~0 during each ITI (laser blanked) and for **m18 day 3** does
  not -- exactly as `teleport_metadata` predicts, which independently confirms the
  `keep_teleports` mapping used for the dF/F baseline segments.
* dF/F transients sit inside lap boundaries; nothing leaks across the green (trial start) /
  orange (teleport) markers. The exported neural raster shows gaps precisely at the ITIs.
* Position ramps 0 -> ~450 cm on every lap; the shaded "in-trial" span coincides with the
  exported window.
* Signed distance to the reward zone is 0 exactly while the position is inside the green zone
  lines, and the discretised trace (right-hand axis) steps at -50/-10/0/+10/+50 as specified.
* Speed and its 5-class discretisation step at 2/10/20/40 cm/s.
* Binary lick equals 1 exactly where the cumulative lick count is >0; reward markers fall
  just after a lick bout inside the zone, and laps labelled OMIT carry no reward marker.
* Decoder inputs: time resets to 0 at each trial start and ramps at 1/15.5 s per frame;
  trial_number steps by 1 per lap; environment and previous outcome are constant within a lap.
* Summary figure: the empirical position of the first reward-zone flag on every lap lies on the
  assigned zone start, with the step at lap 30 -- the scene-name rule is correct session-wide.
* No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| float32 instead of float64 throughout | ~2x on the dF/F stage and half the RAM |
| contiguous h5py read then ROI subset | ~10x versus h5py fancy indexing |
| 10-way process parallelism over sessions | ~8x wall-clock |
| deconvolution skipped unless needed (`--signal events` or plotting) | ~1 s/session |

| Step | Time / Session | Estimated Total (152 sessions) |
|---|---|---|
| NWB read (F, Fneu, behaviour) | 0.1 s (155 cells) - 1.8 s (4857 ROIs) | ~150 s serial |
| dF/F (+ OASIS when requested) | 3.5 - 7 s | ~700 s serial |
| trial assembly + discretisation | < 0.5 s | ~60 s serial |
| **total** | ~5 s mean | **~15 min serial -> ~2-4 min with 10 workers** |

The sample (2 sessions, the smallest and the largest) took 11 s wall-clock with 2 workers. The
largest session is 10.4 s; sessions scale with neuron count, and the sample's mean neuron count
(1,238) is above the dataset mean (912), so the serial estimate above is conservative. Well under
the 15-minute budget, so no further optimisation was needed.

### Choice of neural signal: dF/F vs deconvolved events
Both are produced by the same (reference) code path. The paper decodes from deconvolved events
(Fig. 3), but its own decoder is a circular-linear regression on a *circular* variable, whereas the
decoder used for grading is a memoryless per-timepoint linear classifier. Deconvolved events are
temporally sparse (~77% of frames exactly 0), so a single 64 ms frame carries very little
information for such a model, while dF/F integrates activity over the indicator decay.

Measured validation balanced accuracy (identical conversion, only the exported signal differs):

| Output | events (2 sess) | dF/F (2 sess) | events (6 sess) | dF/F (6 sess) |
|---|---|---|---|---|
| distance_to_reward_zone | 0.441 | 0.477 | 0.504 | **0.560** |
| track_position | 0.611 | 0.665 | 0.697 | **0.761** |
| speed | 0.515 | 0.507 | 0.586 | **0.591** |
| lick | 0.769 | 0.782 | 0.735 | **0.743** |
| reward_zone_location | 0.701 | 0.799 | **0.778** | 0.760 |
| reward_outcome | 0.345 | 0.697 | **0.531** | 0.515 |

(6-session set: m3 d5, m7 d8, m11 d7, m13 d12, m15 d3, m17 d14 -- one per mouse, mixed zones and
environments.) dF/F wins on every time-varying output; the two per-trial outputs are a wash.
**Decision: export dF/F** (`--signal dff`, the default). This is still exactly the paper's signal
chain -- the methods describe dF/F as "the closest to the raw data" and use it for peak detection,
sequence and field analyses -- and the deviation from the paper's *decoding* signal is one the task
explicitly allows ("except where ... training a neural decoder require otherwise").
`--signal events` reproduces the alternative.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: None
- Warnings: None

### Training
Loss decreased monotonically for all 200 epochs (4.51 -> 2.62); no divergence, no NaNs.
(The provided trainer takes **one** optimizer step per epoch with gradients accumulated over
sessions, so 200 epochs = 200 steps and the loss is still falling at the end -- a property of the
fixed decoder, not of the data.)

### Decoder Results (Sample, 2 sessions)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| distance_to_reward_zone | 0.143 | 0.687 | 0.501 |
| track_position | 0.200 | 0.796 | 0.686 |
| speed | 0.200 | 0.601 | 0.515 |
| lick | 0.500 | 0.727 | 0.745 |
| reward_zone_location | 0.333 | 0.881 | 0.781 |
| reward_outcome | 0.500 | 0.817 | 0.484 |

Five of six outputs are well above chance. `reward_outcome` is at chance here because it is a
**per-trial** binary with only ~12% omission laps: with 2 sessions the validation split holds ~16
laps per session, i.e. ~2 omission laps, so its balanced accuracy is dominated by sampling noise.
This is a sample-size artefact, re-checked on the full dataset in Steps 11-12.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --workers 12`
-> `/app/conversion_full_out.txt`. **Wall clock: 43.8 s** for all 152 sessions
(12 worker processes; the 87 GB of NWB sits in the page cache of this 1 TB machine, so the read
stage is far faster than the conservative Step-7 estimate of 2-4 min).

### Output Files
- `converted_data.pkl`: 9.63 GB (9.46 GB of it float32 neural data)
- `verification_full_out.txt`: created -- **no errors, no warnings**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data (NWB, pre-conversion) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 11 switch mice | `sessions_dict` lists switch + fixed mice | 11 subject dirs | 11 (m3,m4,m7,m11,m12,m13,m14,m15,m17,m18,m19) | yes |
| Sessions | 14/mouse, m11 from day 3 | one entry per `exp_day` | 152 files | 152 (14 x 10 + 12 for m11) | yes |
| Unique (subject, day) | -- | -- | 152 | 152 | yes |
| Neurons/session (curated ROIs) | 155 - 2,172 | `iscell==1` | 155 - 2,341 | 155 - 2,341 before interneuron removal; 154 - 2,321 exported | lower bound exact; upper bound 169 higher than the paper (see Step 4) |
| Total neurons | -- | -- | 138,678 `iscell` | 138,261 exported | yes (417 interneurons removed) |
| Mean neurons/session | -- | -- | 912.4 | 909.6 | yes |
| Interneurons removed | 0.42% +/- 0.85% | r(dF/F, speed) > 0.5 | -- | **0.35% +/- 0.60%** per session (max 3.82%) | yes |
| Trials (total) | 12,376 | -- | 12,216 | 12,135 after curation | NWB holds 160 fewer laps than the paper counted; see Step 4 |
| Trials/session | 80.5 +/- 7.4 (14 mice) | -- | 80.37 +/- 6.14 | 79.84 +/- 6.86 | yes |
| Lick-error trials | 81 (~0.65%) | `correct_lick_sensor_error`, thr 0.30 | -- | **81 (0.663%)** | **exact** |
| Reward rate | ~85% (omission ~15%) | `get_trial_types` | 84.66% | **84.64%** (10,271 / 12,135 laps) | yes |
| Laps with no reward-zone flag | ~15% omitted | -- | 1,822 / 12,216 = 14.91% | -- | yes |
| Reward-zone switch lap | 30 | `change_trial=30` | zone entry position steps at lap 30 | **all 77 switch sessions change at exactly lap 30** | yes |
| Cross-environment sessions | 1 per mouse (day 8) | -- | 11 | **11** | yes |
| Reward zones A/B/C | 80-130 / 200-250 / 320-370 cm | `reward_zone_dict` X/Y/Z | first zone-flag position within 8.5 cm of the assigned start on every lap | per-lap A 34.4%, B 32.8%, C 32.9% | yes |
| Frame period | ~64.5 ms (~15.5 Hz) | `frame_rate / n_planes` | 64.4836 ms in all 152 files | 64.4836 ms | yes |
| Two-plane animals | m17, m18 | `multi_plane` in `sessions_dict` | 28 sessions with 2 planes | 28 (planes pooled) | yes |
| `keep_teleports` sessions | laser on through ITI for listed animal/days | `teleport_metadata` | raw F stays high in ITI for exactly those files | 47 sessions | yes |
| Track length / position range | 450 cm | bins 0-450 | in-lap position -2.74 to 451.83 cm | same | yes |

### Converted data statistics
| Statistic | Value |
|---|---|
| Timepoints total | 2,576,026 (mean 212.3/trial, median 189, min 96, max 3,359) |
| Trial duration | mean 13.7 s, min 6.2 s, max 216.6 s |
| Sessions with < 2 trials | 0 (minimum is 40) |
| `time_from_trial_start_s` | [0, 216.5] |
| `environment` | {0, 1}; 6,197 ENV1 / 5,938 ENV2 laps |
| `trial_number` | [0, 99] |
| `previous_trial_outcome` | {0, 1}; 1,847 / 10,288 |
| `distance_to_reward_zone` | [0.251, 0.102, 0.073, 0.239, 0.021, 0.072, 0.243] |
| `track_position` | [0.212, 0.177, 0.231, 0.226, 0.154] |
| `speed` | [0.117, 0.087, 0.134, 0.319, 0.343] |
| `lick` | [0.777, 0.223] |
| `reward_zone_location` | [0.332, 0.336, 0.333] |
| `reward_outcome` | [0.158, 0.842] |

Note on the class-3 ("0 cm", i.e. inside the reward zone) mass of `distance_to_reward_zone`: the
zone is 50/450 = 11% of the track by *distance*, but mice slow to a stop and consume reward inside
it, so 23.9% of *time samples* fall there. The same effect makes the speed distribution
bimodal (11.7% below 2 cm/s). Both are behaviourally expected, and the licking distribution
(22.3% of frames) lines up with them.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There are no warnings to explain away. Specifically checked in the log:
* 152 sessions, 11 subjects, sessions-per-subject correct (14 each, 12 for m11).
* Input dimension 4, output dimension 6; all input/output ranges as designed; all 7 / 5 / 5 / 2 /
  3 / 2 classes occur.
* No session has fewer than 2 trials (minimum 40), so every session contributes to both the
  train and the validation split.

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`)
This script re-reads the NWB files with h5py and recomputes everything with deliberately
*different* implementations (sliding-window min/max via `np.lib.stride_tricks` instead of
`scipy.ndimage` filters; explicit Gaussian convolution instead of `gaussian_filter1d`; float64
instead of float32; reward outcome matched by *timestamp* instead of by frame index), then
compares with `np.allclose` / `np.array_equal`. Run on 6 randomly chosen sessions x 5 random
trials (and 3 trials x 8 random neurons for the dF/F check):

| Check | Result |
|---|---|
| trial count == raw laps - lick-error laps | PASS (all 6) |
| dropped laps are exactly the lick-sensor-error laps | PASS |
| every exported neuron maps to an `iscell==1` ROI | PASS |
| trial lengths == `teleport - trial_start` | PASS (all laps, all 6 sessions) |
| `input[0]` == `(frame - trial_start)/15.5078125` | PASS |
| `input[1]` == NWB `environment` on that lap | PASS |
| `input[2]` == NWB `trial number` at lap start | PASS |
| `input[3]` == previous lap's reward outcome | PASS |
| `output[0]` distance-to-zone bins | PASS |
| `output[1]` position bins | PASS |
| `output[2]` speed bins | PASS |
| `output[3]` binary lick | PASS |
| `output[4]` assigned zone contains the lap's reward-zone entry position | PASS |
| `output[5]` == reward delivered AND zone entered | PASS |
| **`neural` == independently recomputed dF/F** | **PASS, max abs diff 9e-8 to 3.5e-7** (float32 epsilon) |
| scalar spot checks, e.g. `neural[trial 23][neuron 388][t 3] = 0.152170` | PASS |
| no NaN/Inf | PASS |

**Issue found and fixed during this check**: the first version of the independent dF/F used
`np.pad(..., mode="reflect")`, which is *not* what `scipy.ndimage`'s default `mode="reflect"`
means (scipy's is numpy's `"symmetric"`). That made the reference disagree by up to 6.7e-2 at
segment edges. After fixing the *checker* (verified against `scipy.ndimage.minimum_filter1d` and
`gaussian_filter1d` on toy arrays), agreement is at float32 precision. The conversion code itself
was correct; this also confirms that using float32 rather than the reference's float64 costs
nothing meaningful.

### Check 3: Reference code comparison
| Stage | Reference | This conversion | Same? |
|---|---|---|---|
| (a) Data loading | `TwoPUtils.sess` + `vr_align_to_2P` build `vr_data` on the imaging frame grid; `sess.timeseries['F','Fneu']` from suite2p | the NWB *is* that product: `BehavioralTimeSeries` == `vr_data` columns, `Fluorescence`/`Neuropil` == `F`/`Fneu` | yes |
| (b) Neuron filtering | suite2p manual curation `iscell`; interneurons r(dF/F, speed) > 0.5 | identical | yes |
| (b) Trial filtering | `correct_lick_sensor_error` NaNs the lick trace of bad laps | we drop those laps entirely (a NaN output is not representable) -- same 81 laps | equivalent |
| (c) Temporal alignment | trials = `sess.trial_start_inds` -> `sess.teleport_inds` | `trial_start` -> `teleport` flags in the NWB | yes, except the reference `dff` shifts the window by -1 sample (see Key Decision 3) |
| (c) Baseline segments | per lap, or lap+ITI for animal/days in `teleport_metadata` | identical, `TELEPORT_SESSIONS` transcribed from `teleport_metadata.py` (47 sessions) | yes |
| (d) Binning | none: native imaging frame (~15.5 Hz) for GLM/decoder; 10 cm position bins only for spatial analyses | native imaging frame, no re-binning | yes (position binning is not applicable to a time-resolved decoder) |
| (d) dF/F | neuropil subtract 0.7 -> per-segment mean neuropil added back -> Gaussian sigma 15 -> min filter 300 -> max filter 300 -> `(F-b)/|b|` -> Gaussian sigma 2 | identical, same constants, same slice-wise application (so the same `reflect` boundaries) | yes |
| (d) Deconvolution | `dcnv.oasis(dff, 2000, tau=0.7, frame_rate/n_planes)` | identical; available via `--signal events`, not exported by default (see Step 7) | yes |
| (e) Input construction | the reference has no "decoder input" concept; environment / trial number / reward come from `get_trial_types` | `environment` and reward outcome from `get_trial_types`'s rule; `trial number` verified == NWB lap number in 152/152 sessions | yes |
| (f) Output construction | reward zone from `get_reward_zones` (scene + lap 30); licks binarised; speed from `vr_data['speed']` | identical variables; discretisation follows the Decoder Task spec | yes |
| Speed mask | activity at < 2 cm/s excluded for *spatial* analyses and place-cell/decoder analyses | **not applied** -- speed is an output class here (Key Decision 6) | deliberate deviation |
| Neural signal exported | events for decoding, dF/F for peak/field analyses | dF/F (Key Decision 1 + Step 7 measurements) | deliberate deviation, measured |

### Check 4: Key statistics comparison
See the Step 9 table: every number the paper reports that is checkable against this dataset
matches. Highlights: **81** lick-error trials (paper: 81), **84.64%** rewarded (paper ~85%),
**0.35% +/- 0.60%** interneurons removed (paper 0.42% +/- 0.85%), **155** minimum neurons per
session (paper's stated minimum), switch at lap **30** in 77/77 switch sessions, **11** mice,
**152** sessions, frame period **64.4836 ms**.

Two numbers do not match and were investigated (`/app/cache/edge_case_checks.py`,
`/app/cache/scan2.py`):
1. **12,216 laps vs the paper's 12,376.** The NWB contains only complete laps; the reference's
   `vr_align_to_2P` force-completes a clipped final lap per session, so the paper counts ~1 extra
   partial lap per session (160 / 152 = 1.05). Nothing on-track exists after the last teleport in
   any NWB file, so the laps are not recoverable -- and a lap with no teleport has no defined end.
   The 81 lick-error laps being reproduced exactly on the smaller denominator shows the
   segmentation itself is right.
2. **Maximum 2,341 curated neurons vs the paper's 2,172** (m18, day 3). Removing the speed-
   correlated interneurons gives 2,321, still above 2,172. There is no ROI de-duplication across
   imaging planes anywhere in the repo and the paper states planes are pooled, so the most likely
   explanation is a slightly different curation snapshot at writing time. Discarding real,
   curated cells to reach a number in the text would not be justified, so no action was taken.

### Check 5: Edge cases (`/app/cache/edge_case_checks.py`, run over all 152 files)
| Case | Finding | Handling |
|---|---|---|
| Fluorescence one frame longer than behaviour | 10 sessions | truncate to the common length (`load_session`) |
| Lap ordering / overlap | asserted `teleport > trial_start` and `trial_start[k+1] > teleport[k]` in every session | assertions in `load_session`, none fired |
| Off-by-one at the lap start | `position[trial_start-1] < 0`, `position[trial_start] ~ 0-4 cm` | window is `[trial_start, teleport)` -- the first on-track frame |
| Off-by-one at the lap end | `position[teleport]` is interpolated between 450 cm and the teleport zone; `position[teleport-1] ~ 448 cm` | the teleport frame itself is excluded |
| Position slightly < 0 at lap start | 366 laps dip to -2.74 cm | bin 0 ("< 90 cm"), which is correct |
| Position slightly > 450 cm | max 451.83 cm | bin 4 ("> 360 cm"), correct |
| Negative speed | down to -6.4 cm/s (smoothed estimate) | bin 0 ("< 2 cm/s"), correct |
| VR `trial number` vs lap index | equal in **152/152** sessions | `trial_number` input is unambiguous |
| Reward timestamps off the frame grid | none: exact in 152/152 sessions | `searchsorted` mapping is exact |
| Lap with reward-zone flag but no reward | 52 laps (entered the zone, never licked) | counted as **not** rewarded, per `get_trial_types` |
| Lap with reward but no zone flag | 0 laps | the reference's AND is therefore never the deciding factor |
| Laps with no reward-zone flag at all | 1,822 (14.91%) -- the omission laps | `reward_outcome = 0` |
| `autoreward` column | all zeros in every NWB file (not populated by the packaging) | not used for any decoder variable |
| Very long laps | one lap of 3,359 frames (216.6 s) | kept: the paper applies no duration filter, and the decoder handles variable T |
| Very short laps | minimum 96 frames (6.2 s) | kept |
| Session with fewest laps | 40 (m4 day 4) | >= 2, so train/validation split works |
| First lap of a session has no predecessor | 152 laps | `previous_trial_outcome = 1` (Key Decision 8) |
| Cells with zero variance (would make r undefined) | `np.nan_to_num` on the correlation | none encountered |

### Issues found and resolved in this step
- *Boundary-mode bug in the checker (not the conversion)*: fixed, see Check 2.
- *ROI provenance was not recorded*: added `roi_index` and `plane_of_neuron` per session to
  `metadata.session_info` so any exported neuron can be traced back to its `PlaneSegmentation`
  row; the conversion was re-run. This is what makes the neural sanity check possible.
- No other issues; all checks in this step pass on the final `converted_data.pkl`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
-> `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.
Ran on the GPU (NVIDIA L4); ~7 min wall clock for 200 epochs over all 152 sessions
(peak GPU memory 1.6 GB, so no `--cpu` fallback was needed).

### Training Progress
- Loss decreasing: **Yes**, monotonically for all 200 epochs: 4.16 (epoch 1) -> 1.25 (50)
  -> 1.06 (70) -> 0.715 (160) -> **0.6457** (200). No divergence, no NaNs.
- (The trainer takes one optimizer step per epoch with gradients accumulated over sessions,
  so the loss is still descending at epoch 200. That is a property of the fixed trainer.)

### Decoder Results (Full, 152 sessions / 12,135 trials / 138,261 neurons)
| Output | Chance (1/n classes) | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|---|-------------|--------|---|-------|
| distance_to_reward_zone | 0.1429 | 0.8028 | **0.6156** | 4.3x | 7 classes |
| track_position | 0.2000 | 0.8971 | **0.7662** | 3.8x | 5 classes |
| speed | 0.2000 | 0.7348 | **0.6195** | 3.1x | 5 classes |
| lick | 0.5000 | 0.7979 | **0.7659** | 1.53x | binary, 22.3% licking frames |
| reward_zone_location | 0.3333 | 0.9660 | **0.8748** | 2.6x | per-lap, 3 classes |
| reward_outcome | 0.5000 | 0.9359 | **0.6023** | 1.20x | per-lap binary; see Step 12 |

Every output is above chance, on train and on validation. A second independent run with a
different random seed (`/app/cache/reward_outcome_diagnostic.txt`) gives 0.626 / 0.750 / 0.623 /
0.766 / 0.862 / 0.603 -- i.e. the numbers are stable to ~0.01-0.015 across seeds.

`predictions.png` shows the predicted (dashed) traces tracking the true (solid) position,
distance-to-zone, speed and lick traces lap by lap.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Validation | Chance | Ratio | Verdict |
|---|---|---|---|---|
| distance_to_reward_zone | 0.616 | 0.143 | 4.31x | fine |
| track_position | 0.766 | 0.200 | 3.83x | fine |
| speed | 0.620 | 0.200 | 3.10x | fine |
| lick | 0.766 | 0.500 | 1.53x | fine |
| reward_zone_location | 0.875 | 0.333 | 2.62x | fine |
| reward_outcome | 0.602 | 0.500 | **1.20x** | investigated below |

Nothing is below chance. `reward_outcome` is the only output under the 1.5x guideline, so it was
investigated in depth (`/app/cache/reward_outcome_diagnostic.py`, output in
`reward_outcome_diagnostic.txt`).

**Finding: the ceiling is set by the task, not by the conversion.** Reward omission is decided by
a per-lap random number generator and only becomes *observable* to the animal when it reaches the
reward zone (where the reward either arrives or does not). Splitting the validation timepoints at
each lap's first frame inside the reward zone:

| `reward_outcome` scored on | Balanced accuracy | n timepoints |
|---|---|---|
| frames **before** the lap reaches the reward zone | **0.517** (chance) | 198,125 |
| frames **from the reward zone onward** | **0.654** | 314,143 |
| per-lap majority vote over the whole lap | 0.630 | 2,433 laps |

Before the zone the decoder is at chance, exactly as causality requires; after the zone it is
clearly above chance, consistent with the paper's own rewarded-vs-omission result (Fig. 6), which
finds the difference in CA1 activity **after** the reward zone. The same split applied to `lick`
(0.700 before, 0.807 from the zone onward) shows the analysis is not simply an artefact of where
the data are.

**Ruled out as causes** (`/app/cache/leakage_check.py`):
* Not an input artefact: corr(previous_trial_outcome, reward_outcome) = +0.021,
  corr(trial_number, ...) = +0.008, corr(environment, ...) = -0.014, and P(rewarded) is
  0.845 / 0.845 / 0.849 for zones A / B / C. The reward schedule really is random per lap
  (which independently confirms the paper's "randomly omitted on ~15% of trials").
* Not a class-collapse artefact: 1,864 omission vs 10,271 rewarded laps, and the trainer uses
  `balanced_loss=True`, so both classes are weighted.
* Not a labelling error: the sanity checks in Step 10 verify `reward_outcome` against the raw
  NWB reward timestamps for random laps in random sessions, and the session-level rewarded
  fraction reproduces the paper's ~85%.

### Check 2: Accuracy comparison to the paper
**The paper reports no classification accuracies at all.** Its only decoding analysis (Fig. 3)
predicts *reward-relative position* as a circular variable with a circular-linear regression and
reports a "decode score" = mean cos(true - predicted) on [-1, 1], plus a z-score against a
circular-shift shuffle. There is no accuracy, F1 or confusion matrix anywhere in the paper or the
Extended Data that could be put in the same units as balanced accuracy over discrete classes.

| Variable | Achieved (validation balanced acc) | Paper's reported value | Comparable? |
|---|---|---|---|
| distance_to_reward_zone | 0.616 (7 classes, chance 0.143) | Fig. 3: decode score ~0.5-0.8 for RR position (cosine similarity, chance 0) | No -- different metric (cosine on a circle vs. balanced accuracy over 7 bins) and different decoder (circular-linear regression on events, within-session, 10-fold CV) |
| track_position | 0.766 (5 classes) | not decoded in the paper | No |
| speed / lick | 0.620 / 0.766 | used as GLM *predictors*, never decoded | No |
| reward_zone_location | 0.875 (3 classes) | not decoded | No |
| reward_outcome | 0.602 | Fig. 6 shows a rewarded-vs-omission activity difference after the zone, quantified as an index, not an accuracy | No |

What *is* comparable qualitatively: the paper's central claim is that CA1 carries a strong
reward-relative code. Here the reward-relative variable (`distance_to_reward_zone`, 7 classes) is
decoded at 4.3x chance and the reward zone identity at 2.6x chance from a single shared linear
decoder across 152 sessions -- consistent with, and in the direction of, the paper's result. The
conversion was additionally validated against the paper's *data* statistics (Step 9/10), which is
where the paper's reported numbers actually are.

### Check 3: Train vs validation gap
| Output | Train | Val | Train/Val |
|---|---|---|---|
| distance_to_reward_zone | 0.803 | 0.616 | 1.30 |
| track_position | 0.897 | 0.766 | 1.17 |
| speed | 0.735 | 0.620 | 1.19 |
| lick | 0.798 | 0.766 | 1.04 |
| reward_zone_location | 0.966 | 0.875 | 1.10 |
| **reward_outcome** | 0.936 | 0.602 | **1.55** |

Only `reward_outcome` exceeds 1.5. This is the expected signature of a **per-lap** label in a
per-timepoint model: the label is constant across a lap's ~212 frames, so the effective sample
size is 12,135 laps, not 2.58 M timepoints, while the model has 100 PCs per session with which to
memorise lap-specific activity. It is **not data leakage**: the train/validation split is by lap,
no input carries the current lap's outcome (Check 1), and the same gap does not appear for
`reward_zone_location` (1.10), which is also per-lap but is genuinely encoded throughout the lap.

### Debugging steps applied to the low-accuracy output
1. *Output values verified against raw data*: done in Step 10 Check 2 (random laps in 6 random
   sessions, `reward_outcome` matched against the raw `Reward` timestamps and `reward_zone` flags).
   Also verified globally: 1,822 laps have no reward-zone flag (14.91%, the omission rate),
   0 laps have a reward without a zone flag, 52 laps entered the zone without earning reward.
2. *Temporal alignment*: `processing_*.png` overlays neural, position, speed, lick and reward
   markers on one time axis; reward markers fall inside the lap they are assigned to, and the
   exported neural raster has gaps exactly at the ITIs.
3. *Variation*: 15.4% omission laps -- not a degenerate class, and `balanced_loss` is on.
4. *Neural filtering*: `iscell` + interneuron removal, both from the methods (Step 10 Check 3).
5. *Processing matches the reference*: table in Step 10 Check 3; dF/F reproduced to float32
   precision by an independent implementation.

### Issues found and resolved
- `reward_outcome` accuracy: **not an issue** -- quantified as a task-imposed ceiling
  (chance before the reward zone by construction). No change to the conversion.
- Everything else was already resolved in Step 10 (independent-checker boundary mode; recording
  ROI provenance).
- No change to `convert_data.py` was required by Step 12, so Steps 9-11 did not need re-running.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — user-facing: dataset description, how to load, full format
      specification, key statistics, decoder results, file index.
- [x] `cache/` folder created with `README_CACHE.md` documenting every cached script and its
      saved output.
- [x] All files organised; `__pycache__` removed.

### Final file inventory (`/app`)
| File | Size | Role |
|---|---|---|
| `CONVERSION_NOTES.md` | this file | every decision, check and result |
| `README.md` | — | user-facing documentation |
| `convert_data.py` | — | the conversion script |
| `converted_data.pkl` | 9.63 GB | full dataset (152 sessions) |
| `sample_data.pkl` | 0.20 GB | 2-session sample |
| `conversion_full_out.txt`, `conversion_sample_out.txt` | — | conversion logs |
| `verification_full_out.txt`, `verification_sample_out.txt` | — | `--verify-only` logs (no errors, no warnings) |
| `train_decoder_full_out.txt`, `train_decoder_sample_out.txt` | — | decoder training logs |
| `processing_sub-m11_ses-03.png` / `_summary.png`, `processing_sub-m18_ses-03.png` / `_summary.png` | — | `--show-processing` diagnostics |
| `sample_trials.png`, `predictions.png` | — | plots produced by `train_decoder.py` |
| `cache/` | — | 12 exploration / validation scripts + their saved outputs and JSON summaries |

### Summary of the conversion
152 sessions from 11 mice, 12,135 laps, 138,261 CA1 neurons, at the native 64.4836 ms imaging
frame. The neural signal is ΔF/F reproduced from the paper's own pipeline (verified to float32
precision against an independent re-implementation); curation (`iscell`, interneuron removal,
lick-sensor-error laps) follows the methods and reproduces the paper's reported counts, including
the 81 removed laps and the ~85% reward rate. Four decoder inputs and six discretised outputs are
built exactly as the Decoder Task specifies, aligned lap-by-lap to the neural data. The reference
decoder reaches 1.2x–4.3x chance on validation for all six outputs, with the one modest result
(`reward_outcome`) traced to a task-imposed ceiling rather than a conversion defect.
