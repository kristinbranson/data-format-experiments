# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa, Plitt & Giocomo (2025) *A flexible hippocampal population code for experience
  relative to reward*, Nature Neuroscience. DANDI:001361 (2P CA1 imaging + VR behavior), NWB files in `/app/data`.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents (`/app`):
- `CONVERSION_NOTES.md` (this file), `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `paper.pdf` (43 pages), `methods.txt` (copied methods sections)
- `code/` — reference repo `Sosa_et_al_2024` (src/reward_relative modules, notebooks as .md, docs)
- `data/` — 152 NWB files in 11 subject folders (`sub-m3 … sub-m19`), 92 GB total, plus `dandiset.yaml`
- `decoder.py`, `train_decoder.py` — the provided decoder/validation code
- `cache/` — my scratch analysis scripts

Environment verified: `python3` 3.13, numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0, pynwb 4.1.0,
suite2p (with `suite2p.extraction.dcnv.oasis`), numba 0.67. Hardware: 128 CPUs, 1 TB RAM,
NVIDIA L4 (23 GB), 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified

| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `src/reward_relative/preprocessing.py` | LOADING | Builds `sess` object: suite2p F/Fneu + VR sqlite aligned to 2P frames |
| `TwoPUtils.preprocessing.vr_align_to_2P` (mirrored by `vr_align_to_mock_2P`) | `preprocessing.py` | LOADING | Interpolates VR variables onto imaging-frame grid. `pos`,`t` linear-interp; `morph`,`trialnum`,`scanning` nearest; `lick`,`reward`,`tstart`,`teleport`,`rzone`,`dz` cumsum-interp-diff (counts per frame). Pre-TTL frames get `pos=-500`, nearest-cols `=-1`, count-cols `=0`. **This is exactly what is stored in the NWB `BehavioralTimeSeries`.** |
| `pp.dff` | `preprocessing.py` | PROCESSING | ΔF/F per ROI. Keeps only within-trial (or within-trial+ITI) samples, neuropil subtract (`neu_coef=0.7`, neuropil mean added back per trial), **maximin** baseline (Gaussian σ=15 frames smooth → `minimum_filter1d(300)` → `maximum_filter1d(300)` ≈ 20 s), `dff=(F-base)/|base|`, then Gaussian σ=2 frames smoothing, then OASIS deconvolution (`suite2p.extraction.dcnv.oasis`, `tau` from s2p ops, frame rate per plane) → `events`. |
| `ut.multi_anim_sess` | `utilities.py` | PROCESSING/CURATION | Orchestrates dff+events+place cells per animal/day. Single-channel branch used here: `neuropil_method='subtract'`, `baseline_method='maximin'`, `subtract_baseline=True`, `deconvolve=True`. |
| `behav.get_trial_types` | `behavior.py` | PROCESSING | `isreward[i] = any(reward>0) and any(rzone>0)` within trial; `morph[i]` = unique env value in trial. |
| `behav.get_reward_zones` | `behavior.py` | PROCESSING | Reward-zone [start,stop] per trial from the **scene name**; `X=[80,130]` for "LocationA", `Y=[200,250]` for B, `Z=[320,370]` for C; on `*_to_*` scenes the first `change_trial=30` trials use the first zone, the rest the second. |
| `ra.get_omission_trials` / `get_omission_inds` | `rewardAnalysis.py` | PROCESSING | True omission = trial with **no** `rzone>0` samples (zone inactive). |
| `behav.correct_lick_sensor_error` | `behavior.py` | CURATION | Trials where the fraction of samples with cumulative lick count >2 exceeds a threshold → licks set to NaN (default 0.5 in code, 0.35 in `glmUtils`, **0.30 in the paper Methods**). |
| `glmUtils.get_timeseries_data` | `glmUtils.py` | PROCESSING | The reference "continuous decoder/GLM data" builder used for Fig. 3 and Fig. 7: per-trial slicing of `events`, `pos`, reward-relative position (circular, reward-zone start at 0), trial ids, speed, licks (error corrected, binarised, σ=2 smoothed), `rewarded` step function. Masks out samples with `speed < 2 cm/s` and NaN licks. |
| `spatial.is_putative_interneuron` | `spatial.py` | CURATION | Pearson r between each cell's **dF/F** and running speed over valid samples; `r > r_thresh` (0.5 in `dayData`, and in Methods) → putative interneuron, excluded. |
| `decode.CircularRegression` / `train_vs_test_blocks` | `decode.py` | ANALYSIS | The paper's circular–linear decoder of RR position (Fig. 3). Not used here — we use the provided `train_decoder.py`. |
| `teleport_metadata.teleport_sessions` | `teleport_metadata.py` | PROCESSING | Animal × exp-day list for which the laser was *not* blanked during the teleport ITI → `keep_teleports=True` in `pp.dff`. |
| `dd.define_anim_list` | `dayData.py` | CURATION | Animal cohorts. Switch cohort = GCAMP3,4,7,11,12,13,14,15,17,18,19 (= NWB `m3…m19`); fixed cohort GCAMP2,6,10 is **not** in the DANDI set. |

### Notes
- Parameters actually used in `notebooks/make_multi_anim_sess.md`:
  `neuropil_method_red='subtract'`, `baseline_method='maximin'`, `neu_coef=0.7`, `deconvolve=True`,
  `speed_thr=2`, `bin_size=10`, `min_pos=0`, `max_pos=450`, `ts_key='events'`, `keep_teleports` per animal/day.
- The reference slices trials as `[start-1:stop-1]` inside `pp.dff` / `get_timeseries_data` but as
  `[start:stop]` in `behav.get_trial_types`. See Step 4 for how this is resolved.
- `TwoPUtils` is not installed here; `ut.nansmooth` (same implementation) is used for the Gaussian
  smoothing steps.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, one file per mouse per
experiment day. `ses-NN` == experiment day (1-indexed), verified against
`sessions_dict.py` (e.g. `sub-m11_ses-03` has `identifier`
`/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`, which is exactly GCAMP11 exp_day 3).

Contents of each NWB file (scanned with `cache/scan_data.py`):

- `identifier` → `/data/InVivoDA/GCAMP<n>/<date>/<scene>` — **scene name** (needed for reward zone).
- `general/session_id` → experiment day; `general/subject/subject_id` → `m<n>`;
  `general/optophysiology/ImagingPlane/{location='hippocampus, CA1', indicator='GCaMP7f', imaging_rate}`.
- `processing/behavior/BehavioralTimeSeries/` — all sampled on the imaging-frame grid
  (`timestamps`, ~15.51 Hz, Δt ≈ 0.0645 s), length = n frames:
  - `position` (cm; −500 before imaging TTLs start, −50…0 in the teleport tunnel, 0–450 on track)
  - `trial_start`, `teleport` (binary event per frame), `trial number` (0-indexed, −1 pre-TTL)
  - `environment` (morph: 0 = ENV 1, 1 = ENV 2; −1 pre-TTL)
  - `speed` (cm/s, can be slightly negative), `lick` (cumulative licks per frame, 0–8)
  - `reward_zone` (counts per frame; >0 while the reward zone is *active*, i.e. from zone entry
    until reward delivery — identically 0 on omission trials)
  - `autoreward` (**all zeros in every one of the 152 files** — information lost in the NWB export)
  - `scanning` (±1)
  - `Reward` — a *sparse* TimeSeries: one 0.004 (mL) sample per reward with its own timestamps.
- `processing/ophys/Fluorescence/plane0[,plane1]/data` — raw suite2p **F**, `(n_frames, n_roi_plane)`
- `processing/ophys/Neuropil/plane*/data` — **Fneu**, same shape
- `processing/ophys/Deconvolved/plane*/data` — suite2p `spks` computed from **raw F** (no NaNs,
  raw-F scale up to ~1.1e4) → *not* the paper's `events`, which are deconvolved from the
  custom ΔF/F. We therefore recompute ΔF/F + OASIS ourselves.
- `processing/ophys/ImageSegmentation/PlaneSegmentation/` — `iscell` (n_roi, 2) manual-curation
  flag + probability, `planeIdx`, `pixel_mask`.

### Dataset Size (from data files, before any curation)
| Statistic | Value |
|-----------|-------|
| NWB files / sessions | 152 |
| Subjects | 11 (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) |
| Sessions / subject | 14 each, except m11 = 12 (imaging started on day 3) |
| Neurons (`iscell==1`) total | 138,678 |
| Neurons / session | mean 912, **min 155**, max 2341 |
| Trials (total) | 12,216 |
| Trials / session | mean 80.4 ± 6.2 s.d., min 41, max 100 |
| Imaging frame rate | 15.5078125 Hz per plane (2-plane mice m17, m18 store 31.0156 Hz volume rate) |
| Trial length | median 189 frames (~12.2 s), min 96, max 3359 frames |
| Within-trial frames (total) | ≈ 2.47 M |
| Rewarded trial fraction | 0.8466 (omission 0.1534) |
| Planes | 1 for all mice except m17, m18 (2 planes, pooled) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Mice, switch task | 11 | "Mice were randomly selected to experience the switch task (n = 11 mice) versus the 'fixed-condition' task … (n = 3)" |
| Imaging days | 14 (m11 starts day 3) | "The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Neurons / session | 155–2172 | "This approach yielded 155–2172 putative pyramidal neurons per session" |
| Trials / session | 80.5 ± 7.4 (mean ± s.d.) | "mean ± s.d., 80.5 ± 7.4 trials across 14 mice, all imaging days" |
| Total imaged trials (11 switch mice) | 12,376 | "n = 81 out of 12,376 trials removed across 11 switch mice" |
| Lick-error trials removed | 81 (~0.65%) | same quote; "detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2" |
| Reward omission rate | ~15% | "the reward was randomly omitted on ~15% of trials" |
| Track length | 450 cm | "450 cm virtual linear track" |
| Reward zones | A 80–130, B 200–250, C 320–370 cm | "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" |
| Reward switch trial | after 30 trials | "Each switch occurred after 30 trials." |
| Neural/behaviour sampling | ~15.5 Hz (0.0645 s bins) | "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate." |
| ΔF/F baseline | maximin, 20 s window, within trial | "baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window" |
| ΔF/F smoothing | Gaussian σ = 2 samples (~0.129 s) | "then smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel" |
| Activity rate | OASIS deconvolution of ΔF/F | "The activity rate was extracted by deconvolving dF/F with a canonical calcium kernel using the OASIS algorithm as used in Suite2p." |
| Interneuron exclusion | Pearson r(dF/F, speed) > 0.5, removes 0.42 ± 0.85% of cells | "Additional putative interneurons were detected for exclusion … by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 ± 0.85% of cells" |
| Speed threshold | <2 cm/s excluded for *spatial* analyses | "For all neural spatial activity analyses, we excluded activity when the animal was moving at <2 cm s−1" |
| Spatial bin | 10 cm (45 bins) | "we binned the 450 cm linear track into 45 bins of 10 cm each" |
| ENV | 2 environments, ENV 1 / ENV 2 | "Both environments consisted of a 450 cm linear track…" |

### Processing Details
- Temporal alignment: all behavioural streams are interpolated onto the imaging frame grid
  (already done in the NWB). Trials run from `trial_start` to `teleport`.
- Temporal binning: native imaging frame (~64.5 ms). The paper explicitly uses the frame rate for
  all decoder/GLM timeseries.
- Curation: `iscell` manual curation (suite2p GUI) → putative pyramidal neurons;
  plus speed-correlation interneuron exclusion; plus the lick-sensor-error trial removal.

### Curation Steps
**Neuron curation rules**
1. suite2p manual curation, stored as `iscell[:,0] == 1` in the NWB → keep.
2. Putative interneurons: Pearson r(ΔF/F, speed) > 0.5 over valid samples → drop.

**Trial curation rules**
1. Trials are `trial_start … teleport` (the on-track lap). The frame at the `teleport` index has a
   corrupted interpolated position and is excluded (see Step 4).
2. Lick-sensor-error trials (>30% of samples with cumulative lick count > 2) → the reference NaNs
   the licks, which removes those samples entirely from the GLM/decoder dataset. We drop the trial.

### Decoders Trained (in the paper)
| Decoded variable | Metric | Value |
|---|---|---|
| Reward-relative position (circular), from RR / TR / non-RR place-cell subpopulations | mean "decode score" = mean cos(y−ŷ) | Fig. 3b; effect sizes vs shuffle: test-before RR 2.6, TR 3.3, non-RR 3.2; test-after RR 2.7, TR −0.5, non-RR −0.2 |

The paper's decoder is a **circular–linear regression** scored by `cos(y − ŷ)`, not a categorical
classifier, so there is **no published classification accuracy** directly comparable to the
`train_decoder.py` balanced accuracies. The usable expectation from the paper is qualitative:
position (absolute and reward-relative) is decodable from CA1 population activity far above chance
across essentially the whole track.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Trial slicing | `pp.dff`/`glmUtils` use `[start-1 : stop-1]`; `behav.get_trial_types` uses `[start:stop]` | At `teleport` index the interpolated `position` is a nonsense value between 450 and −50 (e.g. 200.8 between 448.9 and −50). `trial_start` index is the first on-track frame (pos ≈ +0.6, previous frame ≈ −1.1) | "trial = lap of the track" | Use **`[trial_start : teleport)`**. This excludes the corrupted teleport frame (as the reference's `-1` offset also does) and, unlike the reference's legacy offset, does not pull in a teleport-tunnel frame at the start nor drop the last on-track frame. Verified independently: with this definition the lick-error rule reproduces the paper's count exactly (see below). |
| Lick error threshold | 0.5 (`behavior.py` default), 0.35 (`glmUtils`) | with `[start:teleport)` trials: 0.30 → **81** trials, 0.35 → 69, 0.50 → 44 | ">30% … n = 81 out of 12,376 trials" | Use **0.30**, which reproduces the paper's 81 trials exactly. This simultaneously validates the trial-boundary definition and the lick stream. |
| Total trials | – | 12,216 over 152 sessions (days 1–14) | 12,376 | The DANDI set contains only experiment days 1–14; `teleport_metadata.py` shows some mice also had days 15–17. The 160-trial difference ≈ 2 extra sessions. Mean trials/session 80.4 ± 6.2 vs paper 80.5 ± 7.4 → consistent. |
| Neurons/session max | – | 2341 (m18) | "155–2172" | Min matches **exactly** (155 = m11 day 3), so `iscell[:,0]` is the right curation flag. The paper's upper bound presumably refers to the subset of sessions/days entering its analyses (it quantifies switch days) — or is after the interneuron exclusion. Documented, not "fixed". |
| Deconvolved stream | paper's `events` = OASIS on custom ΔF/F | NWB `Deconvolved` is suite2p `spks` on **raw F** (raw-F scale, no NaNs) | "deconvolving dF/F …" | Recompute ΔF/F (`pp.dff` logic) and OASIS ourselves; do not use the NWB `Deconvolved`. |
| `autoreward` | used only for plotting/metadata | all zeros in all 152 files | first 10 trials of a new condition were auto-rewarded | Not needed for any decoder input/output. Noted as unavailable. |
| Reward zone per trial | from scene name + `change_trial=30` | reward-zone-active onset positions cluster at 80 / 200 / 320 cm | zones A/B/C at 80/200/320 | **Cross-validated**: for all 12,216 trials with an active zone, the empirical first in-zone position is within [start−1, start+50) of the scene-derived zone start for every trial (5 trials land 1–4 cm early, i.e. interpolation jitter). The scene+30-trial rule is correct. |
| Speed < 2 cm/s masking | reference masks those samples out of the decoder/GLM data | – | "excluded activity when the animal was moving at <2 cm s−1" | **Deliberate deviation**: the decoder task requires a speed output whose class 0 is "<2 cm/s", and requires contiguous time series per trial. Masking would delete that class and fragment trials. We keep all within-trial samples. Documented in Step 5. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping

| Source (NWB) | Target field | Transform | Reference code | Notes |
|---|---|---|---|---|
| `Fluorescence/plane*/data`, `Neuropil/plane*/data`, `iscell` | `neural` | planes concatenated → keep `iscell[:,0]==1` → `pp.dff` (neuropil subtract 0.7, maximin baseline 20 s, `(F−b)/|b|`, σ=2 smooth) → OASIS (`dcnv.oasis`, tau 0.7, fs = 15.5078125) → slice `[trial_start:teleport)` | `ut.multi_anim_sess`, `pp.dff` | float32, shape (n_neurons, T) per trial |
| ΔF/F vs `speed` | neuron curation | drop cells with Pearson r > 0.5 | `spatial.is_putative_interneuron` | ~0.4% expected |
| `position/timestamps` | `input[0]` time from trial start | `t − t[trial_start]` (s) | – | time-varying |
| `environment` | `input[1]` environment | unique value in trial (0 = ENV1, 1 = ENV2) | `behav.get_trial_types` (`morph`) | per-trial, broadcast over T |
| `trial number` | `input[2]` trial number | 0-indexed trial index in session | – | per-trial, broadcast |
| `Reward` + `reward_zone` | `input[3]` previous trial outcome | `isreward[i-1]`; first trial of a session → 1 | `behav.get_trial_types` | per-trial, broadcast |
| `position` + scene-derived reward zone | `output[0]` distance to reward zone | signed distance to nearest point of [zstart, zstart+50], discretised into 7 bins | `behav.get_reward_zones`, `glmUtils` rel_pos (linear variant) | time-varying |
| `position` | `output[1]` absolute position | 5 equal 90 cm bins over 0–450 | – | time-varying |
| `speed` | `output[2]` speed | 5 bins: <2, 2–10, 10–20, 20–40, >40 | – | time-varying |
| `lick` | `output[3]` lick | `lick > 0` → 1 | `glmUtils` (`licks[licks>1]=1`) | time-varying |
| scene + trial index | `output[4]` reward zone location | A→0, B→1, C→2 | `behav.get_reward_zones` | per-trial, broadcast |
| `Reward` + `reward_zone` | `output[5]` reward outcome | `any(reward in trial) and any(rzone>0)` | `behav.get_trial_types` | per-trial, broadcast |
| `subject_id` | `subjects` / `subject_idx` | 11 mice | – | |
| `ImagingPlane/location` | `brain_regions` | `['CA1']` for all neurons | paper pools planes | |

Discretisation edges (left-closed unless noted):

- **distance to reward zone** `d` (0 inside the zone, negative before, positive after):
  `d < −50` → 0; `−50 ≤ d < −10` → 1; `−10 ≤ d < 0` → 2; `d == 0` → 3;
  `0 < d ≤ 10` → 4; `10 < d ≤ 50` → 5; `d > 50` → 6.
- **absolute position**: `<90`→0, `[90,180)`→1, `[180,270)`→2, `[270,360)`→3, `≥360`→4.
- **speed**: `<2`→0, `[2,10)`→1, `[10,20)`→2, `[20,40)`→3, `≥40`→4.

### Key Decisions
1. **Neural signal = deconvolved events from the paper's own ΔF/F pipeline**, not the NWB
   `Deconvolved` array. Rationale: the paper's decoder/GLM (Fig. 3, Fig. 7) operate on
   "the deconvolved calcium event timeseries"; those events come from `pp.dff(..., deconvolve=True)`
   which applies neuropil subtraction and a per-trial maximin ΔF/F baseline before OASIS. The NWB
   `Deconvolved` is suite2p's spks on raw, neuropil-uncorrected F.
2. **`keep_teleports` per animal/day** exactly as `teleport_metadata.teleport_sessions`
   (m11–m14: days 1,7,8,14; m15–m19: days 1,3,5,7,8,10,12,14; m3,m4,m7: never). This only changes
   which samples enter the ΔF/F baseline windows; the exported trials are always
   `[trial_start:teleport)`.
3. **Trial window = `[trial_start, teleport)`**; alignment event = trial start, `off_start = 0`,
   `off_end = None` (variable-length laps).
4. **No speed threshold** (deviation from the reference, required by the decoder spec — see Step 4).
5. **Drop lick-sensor-error trials** (>30% of frames with cumulative lick > 2): 81 trials.
   Matches the paper's removal exactly. These trials' lick output would be pure artefact.
6. **Neuron curation**: `iscell==1` then speed-correlation interneuron exclusion (r > 0.5).
7. **Cells with all-NaN ΔF/F** (e.g. a cell whose baseline is degenerate) are dropped; events are
   NaN outside trials by construction, and within-trial NaNs (there should be none) are zero-filled.
8. **First trial of a session**: `previous trial outcome` is undefined. Set to 1 (rewarded) — the
   modal outcome (84.7%), and mice ran ~30 warm-up trials on the same reward zone immediately
   before each imaging session. It is an *input*, so a single imputed value per session cannot
   leak or distort the decoded targets.
9. **Brain region** = single region `CA1` for all neurons (the paper pools imaging planes for all
   analyses except Extended Data Fig. 7).
10. **Sessions kept**: all 152. All have ≥ 41 trials, far more than the required 2.

### Planned Sanity Checks
- [x] 152 files, 11 subjects, 14 sessions/subject (12 for m11)
- [x] mean trials/session 80.4 ± 6.2 vs paper 80.5 ± 7.4
- [x] omission fraction 0.153 vs paper ~15%
- [x] lick-error rule at 30% flags exactly 81 trials (paper: 81)
- [x] min neurons/session = 155 (paper: 155–2172)
- [x] scene-derived reward zone matches the empirically observed in-zone positions on every trial
- [ ] interneuron exclusion removes ≈ 0.42 ± 0.85% of cells
- [ ] ΔF/F / events spot-checks recomputed from raw NWB arrays (`np.allclose`)
- [ ] input/output value spot-checks against raw NWB arrays (`np.allclose`)
- [ ] distance-to-reward output == 3 exactly while `position` ∈ [zone start, zone start+50]
- [ ] decoder balanced accuracy above chance for every output

---

## Step 6: Script Development
**Status**: NOT STARTED

---

## Step 7: Sample Conversion and Validation
**Status**: NOT STARTED

---

## Step 8: Sample Decoder Training
**Status**: NOT STARTED

---

## Step 9: Full Conversion and Validation
**Status**: NOT STARTED

---

## Step 10: Critical Review 1
**Status**: NOT STARTED

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED
