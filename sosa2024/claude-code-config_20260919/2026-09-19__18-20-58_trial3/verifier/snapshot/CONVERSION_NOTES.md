# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa, Plitt & Giocomo (2025) *"A flexible hippocampal population code for experience
  relative to reward"*, Nature Neuroscience. DANDI:001361 (NWB), code = `GiocomoLab/Sosa_et_al_2024`.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `paper.pdf` (34 MB), `methods.txt` (49 KB)
- `code/` — reference analysis repo (`src/reward_relative`, `notebooks/`, `docs/`, `environments/`)
- `data/` — 87 GB, 11 subject folders (`sub-m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19`),
  152 `*_behavior+ophys.nwb` files + `dandiset.yaml`
- `decoder.py`, `train_decoder.py` — provided decoder/validation code

Environment verified: `python3` with numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0, pynwb 4.1.0,
scipy, sklearn. Hardware: 1 TB RAM, 128 CPUs, 1× NVIDIA L4 (23 GB), 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `preprocessing.py` | LOADING | Builds the `sess` class (TwoPUtils) that aligns VR/behaviour to the 2P frame grid (`vr_align_to_2P`). The NWB files are an export of exactly this object. |
| `utilities.multi_anim_sess` | `utilities.py` | LOADING/PROCESSING | Top-level pipeline: loads `sess`, computes dF/F, place cells, trial types, reward zones, trial subsets. Shows exactly which arguments are used for dF/F. |
| `preprocessing.dff` | `preprocessing.py` | PROCESSING | **The dF/F computation.** neuropil subtraction (`neu_coef=0.7`) → per-trial neuropil-mean add-back → per-trial `maximin` baseline (Gaussian σ=15 frames, then `minimum_filter1d(300)` then `maximum_filter1d(300)`) → `(F−F0)/|F0|` → per-trial Gaussian σ=2 smoothing. Optional OASIS deconvolution. |
| `utilities.nansmooth` / `TwoPUtils.utilities.nansmooth` | `utilities.py` | PROCESSING | NaN-tolerant Gaussian smoothing (identical to `gaussian_filter1d` when the slice contains no NaNs). |
| `behavior.get_trial_types` | `behavior.py` | PROCESSING | `isreward` = `any(reward>0)` **and** `any(rzone>0)` within `[trial_start_inds[t], teleport_inds[t])`; `morph` = unique environment id per trial. |
| `behavior.get_reward_zones` | `behavior.py` | PROCESSING | Reward-zone `[start,stop]` coords + label ('A'/'B'/'C') per trial from the **scene name**; on switch scenes the zone changes at `change_trial = 30`. Coord dict: A=`[80,130]`, B=`[200,250]`, C=`[320,370]` (dict keys `X`,`Y`,`Z`). |
| `behavior.correct_lick_sensor_error` | `behavior.py` | CURATION | Sets licks to NaN on trials where >`correction_thr` of samples have cumulative lick count >2 (stuck capacitive sensor). Paper text → `correction_thr = 0.3`. |
| `behavior.lickrate` / `lickrate_PETH` | `behavior.py` | PROCESSING | Licks binarised with `licks[licks>0] = 1`; trials sliced as `licks[trial_start_inds[t]:teleport_inds[t]]`. |
| `spatial.is_putative_interneuron` | `spatial.py` | CURATION | Pearson `corr(dff[c, valid], speed[valid]) > r_thresh` ⇒ putative interneuron. `dayData.int_thresh = 0.5`, `int_method='speed'`, `ts_key='dff'`. `valid` = samples where dF/F is not NaN (i.e. within-trial samples). |
| `spatial.calc_place_cells` | `spatial.py` | CURATION | Place-cell identification by shuffled spatial information. **Not used here** (stochastic, and the decoder should see all pyramidal cells). |
| `decode.CircularRegression` | `decode.py` | ANALYSIS | The paper's own decoder (circular–linear regression of reward-relative position from deconvolved events). Reports a cosine "decode score", not classification accuracy. |

### Notes
- dF/F is **not** stored in the NWB files; it has to be recomputed with `preprocessing.dff`
  (the NWB has raw `Fluorescence`, `Neuropil` and suite2p's `Deconvolved` = spks computed from
  **raw F**, which is *not* the signal the paper uses).
- `multi_anim_sess` calls `pp.dff` with:
  `neuropil_method="subtract"`, `baseline_method="maximin"`, `subtract_baseline=True`,
  `regress_ts=None`, `neu_coef=0.7`, `keep_teleports=False` (default), single-channel branch.
- Trial window conventions in the reference code:
  - behaviour code (`get_trial_types`, `lickrate_PETH`, `correct_lick_sensor_error`)
    uses `[trial_start_inds[t], teleport_inds[t])`;
  - `pp.dff` uses `[start-1, stop-1)` (a one-sample shift).
  We use the **behaviour convention** `[trial_start, teleport)` for everything and pass
  `trial_starts+1 / teleports+1` into the dF/F routine so that its internal `start-1:stop-1`
  slices exactly the same samples (see Step 5).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, NWB 2.8.0 (HDF5).
`ses-NN` is the 1-indexed **experiment day** (`general/session_id`). `identifier` holds the original
path, whose last element is the **scene name** (e.g. `Env1_LocationB_to_A`) — this is what
`behavior.get_reward_zones` keys off.

Contents used:
- `processing/behavior/BehavioralTimeSeries/` — one sample per imaging frame (19 k–52 k frames):
  `position` (cm), `speed` (cm/s, smoothed), `lick` (cumulative count/frame), `reward_zone`
  (cumulative rzone-entry flag), `trial_start`, `teleport` (binary flags), `trial number`,
  `environment` (0 = ENV1, 1 = ENV2), `scanning`, `autoreward` (all zeros in the export),
  plus `Reward` (an event series with 1 timestamp per delivered reward).
  Every series carries `timestamps` (imaging frame times, Δ = 0.0644836 s).
- `processing/ophys/Fluorescence/planeK/data` `(nframes, nrois)` float32 — raw F
- `processing/ophys/Neuropil/planeK/data` — Fneu
- `processing/ophys/Deconvolved/planeK/data` — suite2p `spks` from raw F (**not** the paper's events)
- `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell` `(nrois, 2)` — suite2p+manual
  curation flag (col 0) and classifier probability (col 1); `planeIdx` gives the plane of each ROI.
- `general/optophysiology/ImagingPlane/`: `location = "hippocampus, CA1"`, `imaging_rate = 15.5078125`,
  `indicator = GCaMP7f`.
- Mice `m17` and `m18` are **2-plane**; all other mice are single-plane. ROIs of both planes are
  pooled (as in the paper, except Extended Data Fig. 7).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) |
| Sessions | 152 (14/mouse except m11 which has days 3–14 = 12) |
| Sessions / subject | 12 (m11) or 14 (all others) |
| ROIs total (all, uncurated) | 253,886 |
| Curated cells (`iscell[:,0]==1`) total | 138,678 |
| Curated cells / session | mean 912.4, **min 155, max 2341** |
| Trials (total, trial_start/teleport pairs) | **12,216** |
| Trials / session | mean 80.37 ± 6.14 (80 in 140/152 sessions; min 41, max 100) |
| Frames within trials (total) | 2,620,514 |
| Trial duration | 96–3359 frames (6.2 s – 216 s) |
| Neural data size (curated cells × in-trial frames) | 2.42e9 values = 9.7 GB float32 |
| Rewarded trials | 10,342 / 12,216 = **84.66%** |
| Trials with lick-sensor error (thr 0.3) | **81** |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch task) | 11 mice | "Mice were randomly selected to experience the switch task (n = 11 mice) versus the 'fixed-condition' task" |
| Sessions | 14 days/mouse; m11 from day 3 | "The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Switch-day sessions | 77 (= 11 mice × 7 switch days) | "n = 77 sessions, 11 mice, seven switch days" |
| Trials / session | 80.5 ± 7.4 | "mean ± s.d., 80.5 ± 7.4 trials across 14 mice, all imaging days" |
| Trials (total, 11 switch mice) | 12,376 | "~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice" |
| Lick-sensor-error trials | 81 (0.65%) | same quote; "detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2" |
| Neurons / session | 155–2172 | "yielded 155–2172 putative pyramidal neurons per session" |
| Cells imaged on switch days | 954 ± 453 | "459 ± 263 place cells out of 954 ± 453 cells imaged" (mean ± s.d. across 11 mice, 7 switch days) |
| Neural data time bin | ~64.5 ms (15.5 Hz) | "unidirectional scanning at ~15.5 Hz"; "the 0.0645 s imaging frame samples" |
| Behaviour data time bin | same (VR resampled onto the imaging frame grid) | `vr_align_to_2P` |
| Reward rate | ~85% | "the reward was randomly omitted on ~15% of trials" |
| Track length | 450 cm | "traversed a unidirectional 450 cm virtual linear track" |
| Reward zones | A 80–130, B 200–250, C 320–370 cm | "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" |
| Switch trial | after 30 trials | "Each switch occurred after 30 trials." |
| Interneurons excluded | 0.42 ± 0.85% of cells | "Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 ± 0.85% of cells" |
| dF/F baseline | maximin, 20 s window, per trial | "baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window" |
| dF/F smoothing | 2-sample (~0.129 s) Gaussian | "then smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel" |

### Processing Details
- VR behaviour is already interpolated onto the imaging frame grid (one sample per 2P frame).
- Trials = one lap; `trial_start` = entry to the track at 0 cm, `teleport` = end of the track /
  entry to the inter-trial "teleport" zone (gray tunnel of ~50 cm + 1–5 s jitter, 5–10 s after
  omissions). Teleport periods are **excluded** from dF/F (`keep_teleports=False`).
- Place-cell/spatial analyses exclude samples with speed <2 cm/s. **We do not apply this filter**,
  because speed (including the "<2 cm/s" class) is one of the decoder outputs and removing samples
  would break the continuous time base required by the decoder task.

### Curation Steps
**Neuron curation rules**
1. `iscell[:,0] == 1` — suite2p + manual curation (removes ROIs with multiple somata/dendrites,
   no transients, over-expression, putative interneurons).
2. Exclude putative interneurons: Pearson `r(dF/F, speed) > 0.5` over all in-trial samples.

**Trial curation rules**
1. Trials are `[trial_start_inds[t], teleport_inds[t])`.
2. Trials with lick-sensor error (>30% of samples with cumulative lick count >2) are removed
   (the reference sets their licks to NaN; because lick is a decoder *output* and NaNs are
   forbidden, we drop those trials entirely — 81 trials, 0.66%).

### Decoders Trained (in the paper)
| Decoded variable | Accuracy |
|---|---|
| Reward-relative (circular) position from RR/TR/non-RR cells | reported as a **cosine "decode score"** (1 = perfect, 0 = chance), ≈0.5–0.8 for the example session (Fig. 3a,b), and as a z-score vs. circular-shift shuffles. No classification accuracies are reported, so there is no directly comparable number for the categorical decoder used here. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total trials | `len(sess.trial_start_inds)` | 12,216 over 152 NWB sessions | 12,376 over "11 switch mice" | m11 was not *imaged* on days 1–2, so DANDI has 152 sessions; 11×14 = 154 sessions of behaviour exist. 12,376 − 12,216 = 160 = exactly 2 sessions × 80 trials ⇒ the paper's denominator includes m11's two unimaged behaviour days. Our per-trial lick-error detector reproduces the paper's **81** trials exactly, which confirms the trial definition and threshold. |
| Lick-error threshold | function default `correction_thr=0.5`; callers use 0.35/0.3/0.5 | thr 0.3 → 81 trials; 0.35 → 69; 0.5 → 44 | ">30% of the … samples" and n = 81 | Use **0.3** — it is the value in the Methods and it reproduces n = 81 exactly. |
| Cells per session | `iscell` filter | 155 – 2341 | "155–2172" | Min matches exactly. The max comes from m18 (2-plane, planes pooled); 3 sessions exceed 2172 (m18 days 1, 3, 8). Most likely the paper's upper bound was quoted after the additional interneuron/quality exclusions or from a slightly earlier curation. Mean cells on switch days = **969.5 ± 465.8** vs. the paper's **954 ± 453** (1.6% high, and interneuron removal accounts for part of it) ⇒ the `iscell` filter is correct. |
| Reward-zone identity | from scene name, `change_trial=30` | measured reward-zone-entry positions | zones A/B/C at 80–130/200–250/320–370, switch after 30 trials | **0 / 12,216 trials mismatch** between the scene-derived zone and the position of the animal's first reward-zone entry ⇒ scene parsing + `change_trial=30` is exactly right. |
| Reward rate | `isreward = any(reward) & any(rzone)` | 10,342/12,216 = 84.7% | "~15% omitted" | Consistent. |
| Deconvolved data in NWB | paper deconvolves *their* dF/F with OASIS | NWB `Deconvolved` has F-scale amplitudes (max 10976) and no NaNs ⇒ suite2p `spks` from **raw F** | — | Do not use the NWB `Deconvolved`; recompute dF/F with `preprocessing.dff` and use dF/F as the neural signal (see Step 5 key decisions). |
| `autoreward` | used to mark auto-rewarded trials | all zeros in every NWB file | first 10 trials of a new condition were auto-rewarded | Not recoverable from the NWB export; not needed for any decoder input/output. |
| Environment (morph) | `morph` 0 = ENV1, 1 = ENV2 | `environment` ∈ {0,1} (−1 before VR sync); 0 trials have mixed values | 2 environments | Use the unique per-trial value. 11 sessions contain both environments (the day-8 `EnvX_?_to_EnvY_?` sessions), all switching exactly at a trial boundary. |
| `scanning` flag | 1 while imaging | −1 only for the first ~130 frames (before VR/2P TTL sync); **0 trials** overlap those frames | — | No trials need to be dropped for this. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Fluorescence/plane*/data`, `ophys/Neuropil/plane*/data` | `neural` | `iscell` filter → `pp.dff` (neuropil subtract 0.7, per-trial maximin baseline, σ=2 smoothing) → interneuron filter → slice `[tstart, teleport)` | `preprocessing.dff`, `utilities.multi_anim_sess`, `spatial.is_putative_interneuron` | float32, (n_neurons, T) per trial |
| `ImageSegmentation/PlaneSegmentation/iscell[:,0]` | neuron curation | keep == 1 | suite2p curation | |
| `BehavioralTimeSeries/position/timestamps` | `input[0]` `time_from_trial_start` | `t − t[tstart]` (s) | — | time-varying |
| `BehavioralTimeSeries/environment` | `input[1]` `environment` | unique value in trial (0=ENV1, 1=ENV2) | `behavior.get_trial_types` (`morph`) | constant within trial |
| trial index | `input[2]` `trial_number` | 0-based index within session (original indexing, kept through trial exclusion) | — | constant within trial |
| `Reward` + `reward_zone` | `input[3]` `prev_trial_outcome` | `isreward[t−1]`; first trial of a session → 0 | `behavior.get_trial_types` | constant within trial |
| `position` + scene-derived reward zone | `output[0]` `reward_zone_distance` | signed distance to nearest point of the active zone, then 7 bins | `behavior.get_reward_zones` | time-varying |
| `position` | `output[1]` `position` | 5 bins of 90 cm | — | time-varying |
| `speed` | `output[2]` `speed` | 5 bins (<2, 2–10, 10–20, 20–40, >40 cm/s) | — | time-varying |
| `lick` | `output[3]` `lick` | `lick>0 → 1` | `behavior.lickrate`, `lickrate_PETH` | time-varying |
| scene name (+ `change_trial=30`) | `output[4]` `reward_zone_location` | A→0, B→1, C→2 | `behavior.get_reward_zones` | per-trial, broadcast over time |
| `Reward` + `reward_zone` | `output[5]` `reward_outcome` | `any(reward) & any(rzone)` | `behavior.get_trial_types` | per-trial, broadcast over time |
| `general/subject/subject_id` | `subjects` / `subject_idx` | `m3 … m19` | — | |
| `ImagingPlane/location` | `brain_regions` | `['CA1']` | — | all neurons index 0 |

### Key Decisions
1. **Neural signal = dF/F computed with the reference `preprocessing.dff`.**
   The paper's core processed signal is dF/F (maximin baseline within each trial, neuropil
   subtraction with coefficient 0.7, 2-sample Gaussian smoothing). It is *not* in the NWB files, so
   it is recomputed exactly as in `utilities.multi_anim_sess` → `preprocessing.dff`.
   The NWB's `Deconvolved` array is suite2p's `spks` computed from **raw F** (no neuropil
   correction, no per-trial maximin baseline), i.e. *not* the "events" the paper uses, so it is
   not used. We also do **not** re-deconvolve the dF/F, because that step needs the per-session
   suite2p `ops['tau']`, which is not present in the NWB export; introducing a guessed kernel would
   be a worse match to the reference than using the dF/F the paper itself defines.
2. **Trial window = `[trial_start_idx, teleport_idx)`** — exactly the on-track samples, matching
   every behavioural function in the reference (`get_trial_types`, `lickrate_PETH`,
   `correct_lick_sensor_error`). `pp.dff` internally slices `start-1:stop-1`, so we pass
   `trial_starts+1`/`teleports+1` to it, giving the identical sample window for neural and
   behavioural streams (no temporal misalignment).
3. **Temporal alignment at trial start**, `off_start = 0.0`, `off_end = None` (trials have
   different lengths; a trial runs until the teleport, which is when track position ends).
   No resampling: time bin = the native imaging frame, 1000/15.5078125 = **64.4839 ms**.
4. **Neuron curation**: `iscell==1`, then exclude putative interneurons (`r(dF/F, speed) > 0.5`),
   both directly from the reference pipeline.
5. **Trial curation**: drop the 81 lick-sensor-error trials (the reference NaNs their licks; NaN is
   not representable in a categorical output). No other trial exclusions — every trial has valid
   VR data (`scanning == 1`), a single environment value, and ≥96 frames.
6. **No speed threshold.** The paper excludes <2 cm/s samples for spatial analyses; the decoder task
   explicitly requires a "<2 cm/s" speed class, so all samples are kept. Documented deviation.
7. **Reward zone from the scene name** (`behavior.get_reward_zones` logic) rather than from the
   observed rzone-entry position, because the zone is defined even on trials where the animal never
   enters it. Validated against the data (0/12,216 mismatches).
8. **`prev_trial_outcome` for the first trial of a session = 0.** There is no previous trial inside
   the recording (the 30 pre-session warm-up trials are not in the data), so "no known preceding
   reward" is encoded as 0. This affects 152/12,135 = 1.25% of trials.
9. **Trial number keeps its original within-session index** even when an earlier trial has been
   removed by lick-sensor curation, so that the input is the true ordinal position in the session
   and `prev_trial_outcome` refers to the true preceding lap.
10. **Discretisation** exactly as specified by the decoder task (bin edges in Step 6).

### Planned Sanity Checks
- [x] Total trials = 12,216; lick-error trials at thr 0.3 = 81 (paper: 81)
- [x] Reward rate 84.7% (paper: ~85%)
- [x] Cells/session min = 155 (paper: 155–2172); switch-day mean 969.5 ± 465.8 (paper: 954 ± 453)
- [x] Scene-derived reward zone vs. measured rzone-entry position: 0/12,216 mismatches
- [ ] Interneuron exclusion rate ≈ 0.42 ± 0.85% of cells (paper)
- [ ] dF/F in a plausible range, no NaN/Inf
- [ ] Position within trials spans ~0–450 cm; distance-to-reward bin 3 (=0 cm) occupies ≈50/450 of
      the track samples
- [ ] Spot-check raw NWB values against the converted pickle (`np.allclose`) for neural, input and
      output streams
- [ ] Trial/session/subject counts in the converted file match the scan

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Run as
`python -u /app/convert_data.py <outpickle> [--full|--sample] [--show-processing] [--workers N]`.

Structure:
- `get_reward_zone_labels(scene, ntrials)` — re-implementation of `behavior.get_reward_zones`
  (only the label is needed; coordinates come from `REWARD_ZONE_COORDS`).
- `compute_dff_trials(F, Fneu, starts, stops)` — re-implementation of `preprocessing.dff` for the
  single-channel, `neuropil_method='subtract'`, `baseline_method='maximin'`, `keep_teleports=False`
  branch that `utilities.multi_anim_sess` uses. Runs per trial (mathematically identical to the
  reference, which builds a full NaN-padded array and then loops over the same trial windows).
- `find_putative_interneurons` — `spatial.is_putative_interneuron(method='speed', r_thresh=0.5)`,
  computed from streaming per-trial accumulators rather than materialising the concatenation.
- `bin_reward_distance` / `bin_position` / `bin_speed` / `reward_zone_distance` — the decoder-task
  discretisation.
- `read_behavior`, `read_fluorescence` — NWB loading (multi-plane pooling, `iscell` curation,
  reconstruction of the per-frame reward binary from the `Reward` event timestamps).
- `convert_session` — one session; `plot_processing` — the `--show-processing` figure.
- `main` — multiprocessing over sessions, assembly, summary statistics, pickling.

Code inefficiencies identified:
- Naively slicing `F[s:e, keep].T` per trial does a fancy-index + transpose copy on every trial.
- Building the full NaN-padded `(ncells, nframes)` dF/F array (as the reference does) wastes the
  ~15% of frames in teleport periods and doubles peak memory.
- Concatenating all trials' dF/F to correlate with speed for the interneuron test doubles memory.

Code speedups added:
- ROIs are `iscell`-filtered **before** the transpose, once per session
  (`np.ascontiguousarray(F[:, keep].T)`), so per-trial slices are cheap views.
- dF/F is computed directly into per-trial arrays (no NaN padding).
- `find_putative_interneurons` streams sufficient statistics (Σx, Σx², Σxy, …) per trial.
- 16 worker processes over sessions (`multiprocessing.Pool.imap`).
- float64 only inside the per-trial dF/F arithmetic (matching the reference, which uses float64),
  float32 for everything stored.
Result: **full conversion of 152 sessions in 39 s** (of which 11 s is writing the 9.5 GB pickle).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sessions: m11 ses-03, a small single-plane session; m17 ses-05, a larger 2-plane session).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 992 |
| Neurons / session | 154, 838 (mean 496) |
| Subjects | 2 (m11, m17) |
| Sessions / subject | 1 |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| time_from_trial_start range | [0.0, 28.6] s |
| environment range | [0, 1] |
| trial_number range | [0, 79] |
| prev_trial_outcome range | [0, 1] |
| reward_zone_distance distribution | [0.188, 0.082, 0.035, 0.219, 0.024, 0.084, 0.368] |
| position distribution | [0.213, 0.202, 0.213, 0.190, 0.182] |
| speed distribution | [0.053, 0.062, 0.135, 0.505, 0.245] |
| lick distribution | [0.799, 0.201] |
| reward_zone_location distribution | [0.600, 0.168, 0.233] |
| reward_outcome distribution | [0.123, 0.877] |
| dF/F range | [-2.28, 4.43] |

### Processing Plots Review
`processing_m11_ses-03.png`, `processing_m17_ses-05.png` (7 panels per session):
1. **Trial windows** — the shaded `[trial_start, teleport)` windows cover exactly the 0→450 cm
   ramps; the ITI (position pinned at −50 cm) and the pre-sync period (position −500) are excluded.
2. **dF/F pipeline** — raw F, neuropil-corrected F and the maximin baseline for an example cell;
   the baseline hugs the bottom of the trace within each trial and resets at trial boundaries.
3. **dF/F trace** — transients are positive-going, baseline ≈ 0.
4. **Neural matrix** — sparse transients, no blank rows/columns, trial boundaries visible.
5. **Inputs** — time ramps from 0 each trial; environment, trial number and previous outcome are
   constant within a trial and step at trial boundaries.
6. **Position + zone + outputs** — the `reward_zone_distance` bin walks 0→1→2→3→4→5→6 as the animal
   crosses the zone, and equals 3 exactly while position is inside the shaded zone; the position
   bin steps at 90/180/270/360 cm. No temporal offset between position and the derived bins.
7. **Speed / lick / per-trial outputs** — speed bins follow the speed trace, licks cluster around
   and after the zone, per-trial outputs are flat. On m11 trial 1 (a reward omission) the reward
   outcome output correctly drops to 0.
No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| `iscell` filter before transpose; per-trial dF/F without NaN padding | ~1.5× |
| streaming interneuron correlation | avoids a second full copy; ~0.05 s/session |
| 16 processes over sessions | ~10× wall clock |

| Step | Time / Session (sample) | Estimated Total Time (152 sessions) |
|---|---|---|
| read behaviour + fluorescence | 0.1–0.6 s | ~80 s serial |
| dF/F | 0.1–1.0 s | ~230 s serial |
| interneuron test | <0.1 s | ~10 s serial |
| **total (16 workers)** | — | **~40 s** (measured: 39 s) |

Well under the 15 min budget, so no further optimisation was needed.

`/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."** All input
ranges and output distributions are as expected (position roughly uniform over the 5 track bins,
reward rate 0.877 on these two sessions, ~15% omissions overall).

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions / 160 trials)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| reward_zone_distance | 0.7219 | 0.5321 | 0.1429 |
| position | 0.8434 | 0.7131 | 0.2000 |
| speed | 0.7370 | 0.5776 | 0.2000 |
| lick | 0.7546 | 0.7275 | 0.5000 |
| reward_zone_location | 0.9711 | 0.9211 | 0.3333 |
| reward_outcome | 0.9201 | 0.5190 | 0.5000 |

Loss decreased monotonically (2.35 → 0.54 over 200 epochs). Every output is above chance; the only
marginal one is `reward_outcome` (see Step 12, where this is shown to be a property of the task,
not of the conversion).

**Signal choice experiment (done at this stage).** To check decision 1 empirically, the same sample
was rebuilt with the neural signal replaced by OASIS-deconvolved events (`dcnv.oasis(dff, 2000,
tau=0.7, 15.5 Hz)`, i.e. the `deconvolve=True` branch of `preprocessing.dff`) and re-decoded:

| Output | Validation BA, dF/F | Validation BA, events |
|---|---|---|
| reward_zone_distance | **0.553** | 0.395 |
| position | **0.713** | 0.547 |
| speed | **0.582** | 0.561 |
| lick | **0.735** | 0.686 |
| reward_zone_location | **0.927** | 0.934 |
| reward_outcome | **0.516** | 0.377 |

dF/F wins on 5 of 6 outputs, which together with the missing per-session `tau` confirms dF/F as the
neural signal. (Script: `cache/compare_signal.py`.)

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --workers 16` — 38.6 s wall clock
(the per-session timing printed in `conversion_full_out.txt` tracked the Step-7 estimate; no
re-optimisation was needed).

### Output Files
- `converted_data.pkl`: 9.52 GB (152 sessions, 12,135 trials, 138,276 neurons, 2,576,026 timepoints)
- `verification_full_out.txt`: created — **"Data format is valid, no errors or warnings."**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data (NWB) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 11 switch mice | `sessions_dict` lists 11 switch + 3 fixed | 11 subject folders | 11 | ✅ |
| Sessions | 14/mouse, m11 from day 3 | — | 152 files | 152 | ✅ |
| Sessions per mouse | — | — | 12 (m11), 14 (rest) | 12 / 14 | ✅ |
| Total neurons (iscell) | — | `iscell` | 138,678 | 138,678 before, **138,276** after interneuron removal | ✅ |
| Neurons / session | 155–2172 | — | 155–2341 | 154–2323 | ⚠︎ min matches exactly; max 2323 vs 2172 (m18, 2-plane) — see Step 4 |
| Cells/session on switch days | 954 ± 453 (11 mice) | — | 969.5 ± 465.8 | **967.5 ± 464.6** | ✅ (1.4% high) |
| Interneurons excluded | 0.42 ± 0.85% | r > 0.5 vs speed | — | **0.35 ± 0.61%** per session | ✅ |
| Trials (total) | 12,376 (incl. m11's 2 unimaged days) | `len(trial_start_inds)` | 12,216 | **12,135** (after removing 81 lick-error trials) | ✅ see Step 4 |
| Trials/session (mean ± sd) | 80.5 ± 7.4 | — | 80.37 ± 6.14 | **79.84 ± 6.86** | ✅ |
| Lick-sensor-error trials | 81 (0.65%) | thr from Methods = 0.3 | 81 at thr 0.3 | 81 removed (0.66%) | ✅ exact |
| Reward rate | ~85% (≈15% omitted) | `any(reward) & any(rzone)` | 84.66% | **84.64%** per trial (0.842 of timepoints) | ✅ |
| Reward zones | A 80–130, B 200–250, C 320–370 | `reward_zone_dict` X/Y/Z | first rzone entry at 79.7–252 cm | 0/12,216 label mismatches | ✅ |
| Switch trial | after 30 trials | `change_trial=30` | zone changes at trial 30 in every switch session | 30 | ✅ |
| Zone A/B/C per trial | counterbalanced | — | — | 0.344 / 0.327 / 0.329 | ✅ balanced |
| Time bin | ~64.5 ms (15.5 Hz) | — | Δt = 64.4836 ms, identical in all 152 files | **64.4836 ms** | ✅ |
| Track length | 450 cm | — | in-trial position −2.7 … 451.8 cm | position bins 0–4 all populated | ✅ |
| time_from_trial_start | — | — | — | [0, 216.5] s (one 3359-frame stopped trial) | ✅ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output-log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
No errors and no warnings to address. (Earlier iterations produced no warnings either; the int8
output dtype and float32 input/neural dtypes are the types the verifier expects.)

### Check 2 — independent sanity checks (`cache/sanity_checks.py`)
Written without importing `convert_data.py`: every quantity is re-derived from the raw NWB with
separately written code and compared with `np.allclose` / `np.array_equal`. Six sessions were drawn
at random (m7 ses-13, m14 ses-13, m13 ses-09, m11 ses-07, m3 ses-07, m18 ses-02 — covering
non-switch and switch scenes, both environments, single- and 2-plane mice); within each, three
random trials were re-checked for the neural stream and three for every output.
**164 checks, 0 failures** (full log: `cache/sanity_checks_out.txt`). An earlier run of the same
script on a different random draw of six sessions (m14 ses-01, m18 ses-09, m19 ses-07, m14 ses-07,
m11 ses-09, m4 ses-02) also passed with 0 failures.

Structural checks (all sessions): `nsessions == len(session_info)`, `subject_idx`/
`brain_region_idx` lengths, subject names, neuron counts vs `brain_region_idx`, equal trial counts
across neural/input/output, equal time dimensions per trial.

| Check | Result |
|---|---|
| **Neural** — dF/F recomputed from raw `Fluorescence`/`Neuropil` for 18 (session, trial) pairs | PASS (`np.allclose`, atol 1e-5) |
| **Neural** — no NaN/Inf anywhere | PASS |
| **Input 0** — time from trial start vs `position/timestamps − timestamps[trial_start]` | PASS |
| **Input 1** — environment vs `environment[trial_start]` | PASS |
| **Input 2** — trial number vs the original within-session index of the kept trials | PASS |
| **Input 3** — previous outcome vs `any(reward) & any(rzone)` of the preceding lap | PASS |
| **Output 0–3** — distance/position/speed/lick bins recomputed from raw `position`, `speed`, `lick` | PASS (exact) |
| **Output 4–5** — zone label from an independently written scene parser; reward outcome | PASS |
| **Trial counts** — kept trials vs independently recomputed lick-error mask | PASS |
| **FAILURES** | **none** |

dF/F distribution audit (also in `cache/sanity_checks.py`): per-session max |dF/F| has median 4.88
and p90 6.81; 8/152 sessions contain a larger value and only 7.1e-6 of all 2.36e9 values exceed 10.
These come from a handful of cells whose maximin baseline lands near zero on one trial, which is
the behaviour of the reference `preprocessing.dff` itself (it divides by `|F0|` with no guard), so
they are kept rather than clipped; at 7e-6 of the data they cannot materially affect the decoder.

### Check 3 — reference code comparison
| Stage | Reference | This script | Same? |
|---|---|---|---|
| (a) loading | `TwoPUtils` `sess` (F, Fneu, VR resampled onto imaging frames); `multi_anim_sess` | NWB `Fluorescence`/`Neuropil`/`BehavioralTimeSeries`, which are the export of that same `sess` | ✅ identical source arrays |
| (b) neuron filtering | suite2p `iscell` (loaded by `sess.load_suite2p_data`) + `spatial.is_putative_interneuron(ts_key='dff', method='speed', r_thresh=0.5)` | `iscell[:,0]==1`; Pearson r(dF/F, speed) over all in-trial samples > 0.5 | ✅ |
| (c) temporal alignment | behaviour functions slice `[trial_start_inds[t], teleport_inds[t])`; `pp.dff` slices `[start-1, stop-1)` | one window, `[trial_start, teleport)`, used for **both** streams (the dff routine is fed `starts+1/stops+1` so its internal `start-1:stop-1` lands on the same samples) | ⚠︎ deliberate: removes the reference's internal 1-sample inconsistency between its neural and behavioural windows. Verified in Step 4 by reproducing the paper's 81 lick-error trials exactly with the behaviour window. |
| (d) binning | native imaging frame (15.5 Hz); spatial binning into 10 cm bins is used for the paper's *spatial* analyses | native imaging frame, no re-binning | ✅ (spatial binning is not applicable to a time-resolved decoder) |
| (e) input construction | n/a (the paper's decoder takes only neural activity) | time from trial start, environment (`morph`), trial index, previous `isreward` | n/a — specified by the decoder task |
| (f) output construction | `get_trial_types` (isreward, morph), `get_reward_zones` (zone coords/labels), `licks[licks>0]=1`, `vr_data['pos']`, `vr_data['speed']` | identical definitions, then the decoder task's discretisation | ✅ |
| dF/F | `pp.dff(..., neuropil_method='subtract', baseline_method='maximin', subtract_baseline=True, neu_coef=0.7, keep_teleports=False)` | same arithmetic, same constants (σ=15 smooth, 300-sample min then max filter, `(F−F0)/|F0|`, σ=2 smooth) | ✅ |

Documented differences and why:
1. **Trial window** — see above (consistency between streams; verified against the paper's n=81).
2. **No <2 cm/s speed mask** — the paper applies it for place-cell/spatial analyses; the decoder
   task requires a "<2 cm/s" speed class and a continuous time base, so all samples are kept.
3. **No place-cell restriction** — the paper's decoder used RR/TR/non-RR *place-cell*
   subpopulations. Place-cell identification is shuffle-based and stochastic; for a general neural
   decoder all curated pyramidal cells are the right input, and restricting to place cells would
   discard most of the recorded population.
4. **dF/F rather than deconvolved events** — the events the paper deconvolves are not in the NWB
   and need a per-session suite2p `tau` that is also not in the NWB; dF/F is fully specified by the
   reference code and empirically decodes better (Step 8 table).
5. **Lick-error trials dropped rather than NaN-ed** — NaN is not a legal categorical output value.

### Check 4 — key statistics comparison
See the Step 9 table. Every statistic available in the paper is reproduced: 11 mice, 152 imaged
sessions, 80.5→79.8 trials/session, 84.6% reward rate, 81 lick-error trials (exact), min 155
cells/session (exact), 954±453 → 967.5±464.6 cells on switch days, 0.42±0.85% → 0.35±0.61%
interneurons, reward zones/switch trial confirmed against the animals' measured zone entries
(0/12,216 mismatches), 64.5 ms frame time. The two residual differences (paper's 12,376 trials;
paper's 2172-cell maximum) are explained in Step 4 and are not conversion errors.

### Check 5 — edge cases
| Edge case | Handling / verification |
|---|---|
| Frames before VR/imaging TTL sync (`position = −500`, `trial number = −1`, `scanning = −1`) | Never inside a trial window: 0/12,216 trials overlap `scanning != 1` (verified across all files). |
| First trial of a session has no previous lap | `prev_trial_outcome = 0`; documented (152 of 12,135 trials). |
| A trial dropped for lick-sensor error in the middle of a session | `trial_number` keeps the original within-session index and `prev_trial_outcome` refers to the true preceding lap (both checked in `sanity_checks.py`). |
| Sessions with unusually few trials (40, 45, 50, 60, 71, 73, …) | All ≥ 40, so every session has ≥ 2 trials. |
| Switch sessions shorter than the 30-trial switch point | None exist (minimum switch-session length is 41 trials); `get_reward_zone_labels` still clamps with `min(30, ntrials)`. |
| Two-plane mice (m17, m18) | Planes pooled in `rois` order; `iscell` indexed through each plane's `rois`; `ImagingPlane/imaging_rate` is the 31 Hz *scan* rate, so the effective 15.5078125 Hz frame rate is taken from the timestamps instead (asserted equal across all 152 sessions). |
| Position slightly outside [0, 450] at the trial edges (−2.7 … 451.8 cm) | `bin_position` clips to [0, 4]; `reward_zone_distance` is defined for any position. |
| `autoreward` all zeros in the export | Not needed by any input/output; documented. |
| Rewards delivered outside any trial window (3 of 10,345) | Ignored, exactly as `get_trial_types` does. |
| Baseline ≈ 0 → very large dF/F | 7e-6 of values; reference behaviour retained (see Check 2). |
| Trials with no reward-zone entry (omission trials) | `reward_zone_distance` is still defined from the scene-derived zone, which is why the zone is taken from the scene and not from the observed entry. |

### Issues Found and Resolved
- **`ImagingPlane/imaging_rate` is 31 Hz for the 2-plane mice** — the first full run asserted on
  "multiple frame rates". Fixed by deriving the frame rate from the frame timestamps
  (15.5078125 Hz in all 152 sessions).
- **Session ordering** — sessions were initially sorted by the subject *string* (`m11 < m3`).
  Changed to numeric mouse/day order for readability (`subject_idx` was correct either way).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, ~9 min).

### Training Progress
- Loss decreasing: **Yes**, monotonically — 2.35 (epoch 10) → 1.22 (50) → 0.75 (140) → 0.640 (200);
  test loss 0.741.

### Decoder Results (Full: 152 sessions, 12,135 trials, 138,276 neurons)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Val / chance | Notes |
|--------|-------------|--------|--------|--------|-------|
| reward_zone_distance | 0.8084 | **0.6189** | 0.1429 | 4.33× | 7 classes |
| position | 0.8963 | **0.7624** | 0.2000 | 3.81× | 5 classes |
| speed | 0.7339 | **0.6219** | 0.2000 | 3.11× | 5 classes |
| lick | 0.7951 | **0.7660** | 0.5000 | 1.53× | binary |
| reward_zone_location | 0.9676 | **0.8729** | 0.3333 | 2.62× | A/B/C |
| reward_outcome | 0.9519 | **0.6090** | 0.5000 | 1.22× | binary; see Step 12 |

`predictions.png` shows predicted vs. true traces for four held-out trials: the position and
reward-zone-distance predictions track the true step functions with no visible temporal lag, which
is an independent confirmation that the neural and behavioural streams are aligned.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — accuracy vs chance
Every output is above chance. Five of six exceed 1.5× chance; `reward_outcome` (1.22×) was
investigated in depth (below) and the low value is a property of the experiment, not a bug.

### Check 2 — comparison to the paper
The paper reports **no classification accuracies**. Its only decoding analysis (Fig. 3) predicts a
*circular reward-relative position* from the deconvolved activity of RR/TR/non-RR **place-cell
subpopulations**, and reports a cosine "decode score" (1 = perfect, 0 = chance) plus a z-score
against circular-shift shuffles.

| Variable | Achieved (this conversion) | Paper |
|---|---|---|
| Reward-relative position | `reward_zone_distance` (7 bins): 0.619 balanced acc (chance 0.143) | mean decode score ≈0.5–0.8 for RR cells before the switch, significantly above shuffle (P = 7.5×10⁻¹⁴, n = 77 sessions); significant over −104.5 to +152.7 cm around the zone |
| Absolute track position | 0.762 (chance 0.200) | not reported as accuracy |
| Speed / lick / zone identity / outcome | 0.622 / 0.766 / 0.873 / 0.609 | not reported |

The two numbers are not on the same scale (a cosine similarity on a circular variable vs. a
7-way balanced accuracy), so no numeric equality is expected. Qualitatively they agree: position
relative to reward is strongly decodable from CA1 across the whole track. Note that the paper's
0.5–0.8 decode scores come from a decoder trained **per session** on a hand-picked place-cell
subpopulation, whereas this decoder shares one linear read-out across all 152 sessions and uses
all curated cells, so it is if anything a harder setting.

### Check 3 — train vs validation gap
| Output | Train | Val | Train/Val |
|---|---|---|---|
| reward_zone_distance | 0.808 | 0.619 | 1.31 |
| position | 0.896 | 0.762 | 1.18 |
| speed | 0.734 | 0.622 | 1.18 |
| lick | 0.795 | 0.766 | 1.04 |
| reward_zone_location | 0.968 | 0.873 | 1.11 |
| **reward_outcome** | **0.952** | **0.609** | **1.56** |

Only `reward_outcome` exceeds the 1.5× flag. Investigation below. There is no data leakage: the
split is over trials within each session, and every per-trial output value (zone, outcome) is
derived only from that trial's own behaviour.

### Investigation of `reward_outcome`
Reward omission is decided by "a random number generator for each trial" (Methods), and the animal
can only discover the outcome once it licks inside the reward zone. Therefore **no signal about the
outcome can exist in CA1 before the reward zone**, and a whole-trial balanced accuracy is
necessarily diluted by the pre-zone half of each trial.

Test (`cache/outcome_timing_test.py`): the provided decoder was retrained on a random 20-session
subset three times — using all timepoints, only timepoints before the reward zone
(`reward_zone_distance` bin < 3) and only timepoints after it (bin > 3):

| Timepoints used | reward_outcome val. balanced acc |
|---|---|
| all (333,165) | 0.598 |
| **before** the zone (149,592) | **0.558** |
| **after** the zone (106,653) | **0.722** |

This is exactly the predicted pattern: near-chance before the zone, clearly decodable after it
(which is also the paper's own finding that rewarded and omission trials diverge *after* the
zone — Fig. 6). The whole-trial 0.609 is the weighted average of the two regimes, so the value is
a property of the task and of the required trial-start alignment, not of the conversion.
The other outputs in that control run reproduce the full-dataset values (position 0.770, lick
0.761, zone 0.882), confirming the subset is representative.

### Debugging steps applied to the lower-accuracy outputs
1. **Output values verified against raw data** — 18 trials × 6 outputs re-derived from the NWB and
   compared exactly (Step 10, Check 2). PASS.
2. **Temporal alignment verified** — `processing_*.png` panels 5–6 overlay the raw position/speed/
   lick traces with the discretised outputs at the same sample index; `predictions.png` shows
   predicted and true traces in phase. No offset.
3. **Output variation** — no output is degenerate: the rarest class is
   `reward_zone_distance` bin 4 (>0 to +10 cm) at 2.1% of timepoints, which is expected because a
   10 cm window immediately after the zone is traversed quickly; the balanced loss/metric handles
   it. `reward_outcome` is 84/16 and `lick` 78/22.
4. **Neural filtering re-checked** — `iscell` + interneuron removal, reproducing the paper's
   per-session cell counts (Step 9 table).
5. **Processing re-checked against the reference** — Step 10, Check 3.

### Issues Found and Resolved
- No further issues. `speed` (0.622, 3.1× chance) is the lowest of the time-varying outputs, which
  is expected: CA1 dF/F at 15.5 Hz carries speed information only indirectly, and the paper itself
  removes cells whose dF/F correlates strongly with speed (the putative interneurons) — i.e. the
  most speed-informative cells are deliberately excluded by the reference curation.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created (`cache/README_CACHE.md` documents every cached script/artefact)
- [x] All files organized

Deliverables in `/app`:
`CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`, `README.md`,
`conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
`conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`,
`processing_m11_ses-03.png`, `processing_m17_ses-05.png`, `sample_trials.png`, `predictions.png`.
