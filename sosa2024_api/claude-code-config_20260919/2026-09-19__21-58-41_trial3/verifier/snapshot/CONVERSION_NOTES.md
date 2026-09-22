# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa, Plitt & Giocomo (2025) *"A flexible hippocampal population code for experience
  relative to reward"*, Nature Neuroscience. NWB files (DANDI dandiset 001361) in `/app/data`.
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `paper.pdf` (43 pages), `methods.txt` (excerpted Methods)
- `code/` — the authors' GitHub repo `Sosa_et_al_2024` (src/reward_relative, notebooks, docs)
- `data/` — 152 NWB files (87 GB) in 11 `sub-mXX/` directories
- `decoder.py`, `train_decoder.py` — provided decoder/validation code
- `pynwb_docs/`

Environment verified: `numpy 2.4.4`, `torch 2.6.0+cu124` (CUDA available, 23 GB GPU),
`pynwb 4.1.0`, `suite2p` (`suite2p.extraction.dcnv.oasis` available).
1 TB RAM, 128 CPU cores, 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `preprocessing.create_sess` / `append_session_data` | `src/reward_relative/preprocessing.py` | LOADING | Builds `sess` class (TwoPUtils) aligning suite2p fluorescence to Unity VR data |
| `TwoPUtils.preprocessing.vr_align_to_2P` | (external repo) | LOADING | Interpolates VR data onto the ~15.5 Hz imaging frame grid — **already applied in the NWB files** |
| `utilities.multi_anim_sess` | `src/reward_relative/utilities.py` | PROCESSING | Top-level pipeline: computes dF/F, events, place cells, trial types per animal/day |
| `preprocessing.dff` | `src/reward_relative/preprocessing.py` | PROCESSING | **The dF/F computation.** Neuropil subtraction (`neu_coef=0.7`), per-trial `maximin` baseline, ΔF/F, 2-sample Gaussian smoothing, OASIS deconvolution → `events` |
| `utilities.default_dff_method` | `src/reward_relative/utilities.py` | PROCESSING | dF/F parameter defaults: `neuropil_method='subtract'`, `baseline_method='maximin'`, `neu_coef=0.7`, `keep_teleports=False` |
| `behavior.get_trial_types` | `src/reward_relative/behavior.py` | PROCESSING | Per-trial `isreward` (reward delivered **and** reward-zone flag set) and `morph` (0=ENV1, 1=ENV2) |
| `behavior.get_reward_zones` | `src/reward_relative/behavior.py` | PROCESSING | Per-trial reward-zone `[start, end]` in cm and label 'A'/'B'/'C' from the **scene name**, switching after trial 30 |
| `behavior.reward_zone_dict` | `src/reward_relative/behavior.py` | PROCESSING | `X=[80,130]` (=A), `Y=[200,250]` (=B), `Z=[320,370]` (=C) |
| `behavior.correct_lick_sensor_error` | `src/reward_relative/behavior.py` | CURATION | Sets a trial's licks to NaN if >`thr` of frames have cumulative lick count >2 |
| `glmUtils.get_timeseries_data` | `src/reward_relative/glmUtils.py` | PROCESSING | **The closest analogue to this task**: builds continuously-sampled per-frame behaviour + deconvolved-activity matrices, trial by trial |
| `decode.CircularRegression` | `src/reward_relative/decode.py` | ANALYSIS | The paper's circular–linear position decoder (Fig. 3) |
| `behavior.lickrate` | `src/reward_relative/behavior.py` | PROCESSING | Binarises licks (`licks[licks>0]=1`) before rate conversion |
| `utilities.nansmooth` | `src/reward_relative/utilities.py` | PROCESSING | NaN-tolerant Gaussian smoothing used throughout |

### Notes
- **Imaging, not ephys** → no spike-sorting quality metrics. Cell curation is (a) the manual
  suite2p `iscell` curation stored in the NWB `PlaneSegmentation`, and (b) exclusion of putative
  interneurons by Pearson r(dF/F, speed) > 0.5 (Methods).
- **dF/F must be computed** — the NWB files store raw `Fluorescence`, `Neuropil` and suite2p's own
  `Deconvolved` (spks, in raw-fluorescence units, computed by suite2p from raw F, *not* the paper's
  events). So the paper's `preprocessing.dff` pipeline has to be re-run on F/Fneu.
- **Trial windows in the reference code are `[trial_start-1, teleport-1)`** (`dff`,
  `glmUtils.get_timeseries_data` both use `start-1:stop-1`). This excludes the teleport sample,
  whose `pos` is an interpolation artefact between the end of the track and −50 cm (documented in
  `dff`'s docstring). I adopt the identical window (see Step 5).
- `keep_teleports=False` by default → only on-track samples are used; ITI/teleport samples are NaN.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb` — one NWB file per mouse-day.

Per file (read with `pynwb`):
- `nwb.identifier` = `/data/InVivoDA/GCAMP<n>/<dd_mm_yyyy>/<scene>` → the **scene name** (e.g.
  `Env1_LocationB_to_A`) needed by `behavior.get_reward_zones`.
- `nwb.subject.subject_id` (e.g. `m11`), `nwb.session_id` (`03` = experiment day).
- `processing['behavior'].BehavioralTimeSeries.time_series`, all on a common timestamp vector
  (one sample per imaging frame, dt = 64.4836 ms):
  `position` (cm), `speed` (cm/s), `lick` (cumulative count/frame), `reward_zone` (binary zone-entry
  flag), `environment` (morph: 0=ENV1, 1=ENV2, −1 before TTL sync), `trial number`, `trial_start`,
  `teleport`, `scanning`, `autoreward`, plus a **sparse** `Reward` series (timestamps of delivery).
- `processing['ophys']`: `Fluorescence`, `Neuropil`, `Deconvolved` — `RoiResponseSeries` named
  `plane0` (and `plane1` for the two 2-plane mice), shape **(n_frames, n_rois)**;
  `ImageSegmentation/PlaneSegmentation` with columns `pixel_mask`, `iscell` (n_roi × 2:
  [is-cell flag, probability]), `planeIdx`.
- Imaging plane: `location = 'hippocampus, CA1'`, `indicator = GCaMP7f`, rate 15.5078125 Hz
  (the `rate` attribute of the RoiResponseSeries reads 31.015625 for the 2-plane mice m17/m18, but
  the actual per-sample dt is 64.4836 ms there too — verified against the behaviour timestamps —
  because the two planes are interleaved in the scan and each plane is sampled at 15.5 Hz).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Sessions | 152 |
| Subjects | 11 (m3, m4, m7, m11–m15, m17–m19) |
| Sessions / subject | 14, except m11 = 12 (imaging started on day 3) |
| ROIs (total, all planes) | 260,091 |
| ROIs passing `iscell` (total) | 138,678 |
| `iscell` neurons / session | mean 912.4, min 155, max 2341 |
| Trials (total, = trial_start events) | 12,216 |
| Trials / session | mean 80.37 ± 6.14, min 41, max 100 |
| Imaging frames (total) | 3,610,877 |
| In-trial frames (total) | 2,620,514 |
| Planes | 1 for 9 mice, 2 for m17 and m18 (ROIs pooled) |
| Frame period | 64.4836 ms (15.5078 Hz) for every session |

Behavioural scan (`cache/scan_behavior.py`):
| Statistic | Value |
|-----------|-------|
| Omission trials | 1,874 / 12,216 = **15.34 %** |
| Lick-sensor-error trials (>30 % of frames with cum. count >2) | **81** |
| Trials with `scanning != 1` inside the trial window | 0 |
| Trials where scene-derived reward zone disagreed with measured zone-entry position | **0 / 10,394** |
| `position` at first window sample | −9.07 … 0.72 cm |
| `position` at last window sample | 436.6 … 449.4 cm |
| `speed` range | −6.45 … 144.5 cm/s |
| `lick` max cumulative count per frame | 8 |
| `autoreward` column | all zeros in every file (unused) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch task) | 11 mice | "Mice were randomly selected to experience the switch task (n = 11 mice) versus the 'fixed-condition' task … (n = 3)" |
| Sessions | 14 days/mouse; m11 starts day 3 | "The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Neurons / session | 155–2172 | "This approach yielded 155–2172 putative pyramidal neurons per session" |
| Trials / session | 80.5 ± 7.4 | "mean ± s.d., 80.5 ± 7.4 trials across 14 mice, all imaging days" |
| Trials (total) | 12,376 across 11 switch mice | "n = 81 out of 12,376 trials removed across 11 switch mice" |
| Lick-error trials | 81 (~0.65 %) | same |
| Reward omission rate | ~15 % | "the reward was randomly omitted on ~15% of trials" |
| Neural data time bin | ~64.5 ms (15.5 Hz) | "imaging FOV … at ~15.5 Hz"; "0.0645 s imaging frame samples" |
| Behaviour time bin | same, VR is interpolated onto imaging frames | "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate" |
| Track length | 450 cm | "Both environments consisted of a 450 cm linear track" |
| Reward zones | A 80–130, B 200–250, C 320–370 cm | "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" |
| Reward switch trial | after 30 trials | "Each switch occurred after 30 trials" |
| Interneurons excluded | 0.42 ± 0.85 % of cells | "Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed" |
| Environments | 2 (ENV1, ENV2); ENV2 introduced on day 8 | "On day 8, the reward zone switch coincided with a switch into the novel environment" |

### Processing Details
- **dF/F**: baseline per trial by a *maximin* procedure with a 20 s sliding window
  (`nansmooth` σ=15 frames along time → `minimum_filter1d(300)` → `maximum_filter1d(300)`;
  300 frames ≈ 19.4 s). dF/F = (F − baseline)/|baseline|, then smoothed with a **2-sample
  (~0.129 s) Gaussian**. Neuropil subtracted first with coefficient 0.7 and the per-trial
  neuropil mean added back (`preprocessing.dff`).
- **Activity rate ("events")**: OASIS deconvolution of the smoothed dF/F (`suite2p.extraction.dcnv.oasis`).
  The paper uses this deconvolved signal for the Fig. 3 decoder and most analyses.
- **Temporal alignment**: VR data are already interpolated onto imaging frames in the NWB files.
  Trials are delimited by `trial_start` / `teleport` flags; the reference code uses the window
  `[trial_start − 1, teleport − 1)`.

### Curation Steps
**Neuron curation rules**:
1. suite2p manual curation → keep ROIs with `iscell[:,0] == 1` (stored in the NWB PlaneSegmentation).
2. Exclude putative interneurons: Pearson r(dF/F, running speed) > 0.5 (expected ≈0.42 ± 0.85 % of cells).
3. Pool planes for multi-plane mice.

**Trial curation rules**:
1. Only on-track samples (teleport/ITI excluded; `keep_teleports=False`).
2. Lick-sensor-error trials (>30 % of frames with cumulative lick count > 2) are NaN-ed in the
   reference. Because `lick` is a required decoder **output** here, these trials are dropped
   (expected 81 trials).
3. (The paper's *neural spatial* analyses additionally drop samples with speed < 2 cm/s; this is
   deliberately **not** applied here — see Step 4/5.)

### Decoders Trained (in the paper)
| Decoded variable | Accuracy |
|---|---|
| Reward-relative (circular) position, from RR / TR / non-RR cells | reported as a *decode score* `cos(y−ŷ)` (1 = perfect, 0 = chance), ≈0.4–0.8 per session for RR cells (Fig. 3b); z-scored vs shuffle >2 over −104.5 ± 20.1 cm to +152.7 ± 22.9 cm around the reward zone |

The paper reports **no categorical classification accuracies**, so there is no directly comparable
number for the balanced accuracies produced by `train_decoder.py`. The qualitative expectation is
that position / reward-relative position decode well above chance, that speed and licking are
decodable (they are strong GLM predictors, Fig. 7d), and that reward outcome is weakly but
above-chance decodable (Fig. 6).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Trial total | — | 12,216 trial_start events in 152 NWB files | 12,376 trials across 11 switch mice | 1.3 % difference. The paper's count presumably includes partially-imaged final trials (a `trial_start` with no matching `teleport`) and/or trials from sessions/warm-up not exported to DANDI. Every NWB file has exactly `n_trial_start == n_teleport`, so the converted count (12,216) is the number of *complete* imaged trials. Documented, not "fixed". |
| Neurons/session max | — | 2341 (m18, planes pooled) | 155–2172 | Min matches exactly (155). Max differs because (a) the paper's range covers all 14 mice including the 3 fixed-condition mice absent from DANDI and (b) for the two 2-plane mice the paper counts ROIs "identified separately per plane" (per-plane max here = 1434). Not an error. |
| Lick-error threshold | `correct_lick_sensor_error` default 0.5; `glmUtils` uses 0.35; `lick_pos_std` uses 0.35 | 0.30 → exactly 81 trials | ">30 % of the 0.0645 s imaging frame samples" | Used **0.30**, which reproduces the paper's n = 81 exactly. |
| Reward zone identity | `get_reward_zones` from scene name, switch at trial 30 | `reward_zone` entry-position flag | switch "after 30 trials" | Cross-checked: the measured zone-entry position agreed with the scene-derived zone on **all 10,394** trials where the flag was set → scene-based assignment and `change_trial=30` are correct for every session. |
| Reward-zone flag on omission trials | `get_trial_types` requires reward AND rzone flag | `reward_zone` flag is 0 on exactly the trials with no reward delivery | — | The Unity `rzone` flag is only raised when reward is available, so `isreward = any(reward) & any(rzone)` reduces to `any(reward)`. Consistent with the reference; used as-is. Omission rate 15.34 % matches "~15 %". |
| Deconvolved series in NWB | paper's `events` come from dF/F | NWB `Deconvolved` is non-negative, in raw-F units (max ≈ 11,000), no NaNs outside trials | "deconvolving dF/F … using the OASIS algorithm" | The NWB `Deconvolved` is suite2p's own `spks` from raw F, **not** the paper's events. dF/F and events are recomputed with `preprocessing.dff`. |
| `autoreward` | used as a `sess.vr_data` column | all zeros in every NWB file | first 10 trials of a new condition auto-rewarded | Column is not populated in the NWB export; not needed for any required variable. |
| Speed threshold 2 cm/s | applied in `get_timeseries_data` / place-cell code | — | "we excluded activity when the animal was moving at <2 cm s−1" | **Not applied**: the decoder task explicitly requires a `speed` output with a `< 2 cm/s` class, so removing those samples would delete an entire output class. Documented deviation. |
| `tau` for OASIS | `sess.s2p_ops['tau']` | not stored in NWB | GCaMP7f | Used the reference function's default `tau = 0.7` (suite2p's value for fast GCaMP variants). |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial definition and temporal alignment
- Trial `i` window = frames `[si[i] − 1, ti[i] − 1)` where `si = where(trial_start==1)`,
  `ti = where(teleport==1)`; identical to `preprocessing.dff` and `glmUtils.get_timeseries_data`.
- **Alignment event**: start of the trial (`trial_start` flag = entry to the linear track at 0 cm).
  `time_from_trial_start = timestamps − timestamps[si[i]]`, so t = 0 at the `trial_start` sample and
  the first sample of the window is at −64.5 ms. `off_start = −0.0645 s`; `off_end` varies by trial
  (median ≈ 18 s) and is reported as `None` with the distribution recorded in `metadata`.
- Time bin = 64.4836 ms for every trial and session (native imaging frame).

### Variable Mapping
| Source (NWB) | Target field | Transform | Reference code | Notes |
|---|---|---|---|---|
| `Fluorescence` + `Neuropil` (all planes, `iscell==1`) | `neural` | `preprocessing.dff` → dF/F → OASIS `events`, sliced per trial | `preprocessing.dff`, `utilities.multi_anim_sess` | `neuropil_method='subtract'`, `neu_coef=0.7`, `baseline_method='maximin'`, `subtract_baseline=True`, 2-sample smoothing, `tau=0.7`, `fs=15.5078` |
| timestamps | `input[0]` `time_from_trial_start` (s) | `t − t[trial_start]` | — | time-varying |
| `environment` (morph) | `input[1]` `environment` | 0 = ENV1, 1 = ENV2 (unique value within trial) | `behavior.get_trial_types` | per-trial, broadcast over time |
| trial index | `input[2]` `trial_number` | 0-based index within the session | `glmUtils.get_timeseries_data` (`trial_ids`) | per-trial, broadcast |
| `Reward` + `reward_zone` | `input[3]` `previous_trial_rewarded` | `isreward[i−1]`; 0 for the first trial (undefined) | `behavior.get_trial_types` | per-trial, broadcast |
| `position` + scene-derived reward zone | `output[0]` `reward_zone_distance` (7 classes) | signed distance to the nearest point of the zone, then binned | `behavior.get_reward_zones` | 0 inside the zone |
| `position` | `output[1]` `position` (5 classes) | `digitize(pos, [90,180,270,360])` | — | 450 cm / 5 |
| `speed` | `output[2]` `speed` (5 classes) | `digitize(speed, [2,10,20,40])` | — | |
| `lick` | `output[3]` `lick` (2 classes) | `lick > 0 → 1` | `behavior.lickrate` (`licks[licks>0]=1`) | |
| scene name | `output[4]` `reward_zone_location` (3 classes) | A=0, B=1, C=2 | `behavior.get_reward_zones` | per-trial, broadcast |
| `Reward`/`reward_zone` | `output[5]` `reward_outcome` (2 classes) | `isreward` | `behavior.get_trial_types` | per-trial, broadcast |
| `ImagingPlane.location` | `brain_regions` | `['CA1']` | — | all neurons |
| `subject_id` | `subjects` / `subject_idx` | | | |

Binning rules (exactly as specified in the Decoder Task):
- distance d: `d < −50 → 0`; `−50 ≤ d < −10 → 1`; `−10 ≤ d < 0 → 2`; `d == 0 → 3`;
  `0 < d ≤ 10 → 4`; `10 < d ≤ 50 → 5`; `d > 50 → 6`.
- position: `<90 → 0`, `[90,180) → 1`, `[180,270) → 2`, `[270,360) → 3`, `≥360 → 4`.
- speed: `<2 → 0`, `[2,10) → 1`, `[10,20) → 2`, `[20,40) → 3`, `≥40 → 4`.

### Key Decisions
1. **Neural signal = deconvolved events computed from the paper's dF/F.** The paper's own decoder
   (Fig. 3) and most analyses use "the deconvolved calcium event timeseries". The NWB `Deconvolved`
   series is suite2p's raw-F deconvolution and is *not* what the paper used, so dF/F + OASIS is
   recomputed with `preprocessing.dff`'s exact parameters.
2. **Trial window `[start−1, teleport−1)`** — matches the reference code exactly and avoids the
   teleport frame whose interpolated `pos` jumps to a meaningless value.
3. **All inputs/outputs stored as time-varying `(d, T)` arrays**, per-trial quantities broadcast.
   Costs negligible memory next to the neural array and guarantees dimension consistency.
4. **No 2 cm/s speed threshold** — required by the decoder output spec (a `<2 cm/s` speed class).
5. **Lick-error trials dropped** (81 expected) rather than NaN-ed, because `lick` is an output.
6. **First trial of each session gets `previous_trial_rewarded = 0`** (genuinely undefined; imaging
   is preceded by 30 un-imaged warm-up trials). Affects ≤152/12,135 ≈ 1.25 % of trials.
7. **Planes pooled** for m17/m18, as in the paper (single `CA1` region).
8. Per-trial `isreward` / `morph` / reward zone are computed over **all** trials before trial
   filtering, so `previous_trial_rewarded` is never corrupted by a dropped trial.

### Planned Sanity Checks
- [x] Total trials, trials/session mean ± s.d. vs paper (80.5 ± 7.4)
- [x] Omission fraction ≈ 15 %
- [x] Lick-error trial count == 81
- [x] Scene-derived reward zone == measured reward-zone-entry position (all trials)
- [ ] `iscell` neuron counts: min 155 per session (paper range 155–2172)
- [ ] Interneuron exclusion fraction ≈ 0.42 ± 0.85 % of cells
- [ ] dF/F distribution sane (median ≈ 0, no NaN/Inf inside trials)
- [ ] Reward delivery position falls inside the assigned reward zone on rewarded trials
- [ ] Per-trial raw-data spot checks of neural/input/output vs independently loaded NWB data

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Runs as `python -u convert_data.py <outpickle> [--full|--sample]
[--show-processing] [--nproc N] [--signal dff|events]`.

Structure:
| Function | Purpose | Reference counterpart |
|---|---|---|
| `nansmooth` | NaN-tolerant Gaussian smoothing, scalar or per-axis sigma | `TwoPUtils.utilities.nansmooth` / `ut.nansmooth` |
| `dff_and_events` | dF/F + OASIS events; neuropil subtraction (0.7), per-trial maximin baseline (sigma 15 -> min-filter 300 -> max-filter 300), dF/F=(F-base)/abs(base), 2-sample smoothing, `dcnv.oasis(.,2000,0.7,fs)` | `preprocessing.dff` (single-channel, `neuropil_method='subtract'`, `baseline_method='maximin'`, `keep_teleports=False`, `deconvolve=True`) |
| `zone_labels_from_scene` | per-trial 'A'/'B'/'C' from the scene name, switching after trial 30 | `behavior.get_reward_zones` |
| `signed_distance_to_zone`, `discretize_distance` | decoder output 0 | new (decoder spec) |
| `read_session` | pynwb loading; pools `plane0`/`plane1`, keeps `iscell==1` ROIs, trims the streams to a common length | `preprocessing.create_sess` + `vr_align_to_2P` (already applied in NWB) |
| `convert_session` | trial windows, neuron curation, per-trial task variables, input/output assembly | `glmUtils.get_timeseries_data`, `behavior.get_trial_types` |
| `plot_processing` | 9-panel per-session diagnostic figure | -- |
| `main` | multiprocessing driver, pickling, summary statistics | -- |

Code inefficiencies identified:
- dF/F is the dominant cost (~10-16 s per session for ~1000 cells), dominated by the
  per-trial Gaussian/min/max filters and OASIS.
- Reading `Fluorescence` + `Neuropil` is ~1-3.5 s per session.
- A single-process run would have taken ~40 min.

Code speedups added:
- float32 throughout (halves memory traffic vs the reference's float64) -- verified to
  change dF/F by < 1e-4 relative to a float64 reference in the Step 10 sanity checks.
- Per-session multiprocessing (`spawn` context; `--nproc 24`), giving ~2.4 min end-to-end.
  `spawn` plus a lazy `suite2p` import is required: forking after suite2p's OpenMP
  runtime is initialised aborts the workers.
- ROI selection (`iscell`) applied at load time so dF/F is never computed for rejected ROIs.
- Vectorised the interneuron speed-correlation over all cells at once.

Edge cases handled:
- 10 of 152 files (all from the 2-plane mice m17/m18) have **one more imaging frame than
  2P-aligned VR samples**; all streams are trimmed to the shorter length.
- Trials whose window would start before frame 0 or end past the last frame are skipped
  (none occur in practice).
- Sessions where a trial has < 2 samples or any non-finite activity are skipped (none occur).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/sample_data.pkl --sample --show-processing`
(sessions m11 ses-03, the session with the fewest neurons, and m17 ses-08, a 2-plane
Env2->Env1 switch session).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 741 (154 + 587) |
| Neurons / session | 370.5 |
| Subjects | 2 (m11, m17) |
| Sessions / subject | 1 |
| Trials (total) | 160 |
| Trials / session | 80 |
| time_from_trial_start range | [-0.0645, 21.67] s |
| environment range | [0, 1] |
| trial_number range | [0, 79] |
| previous_trial_rewarded range | [0, 1] |
| reward_zone_distance distribution | [0.143, 0.094, 0.042, 0.261, 0.023, 0.082, 0.355] |
| position distribution | [0.263, 0.193, 0.252, 0.160, 0.133] |
| speed distribution | [0.068, 0.070, 0.090, 0.325, 0.447] |
| lick distribution | [0.822, 0.178] |
| reward_zone_location distribution | [0.573, 0.427, 0.000] (only A and B occur in these 2 sessions) |
| reward_outcome distribution | [0.116, 0.884] |

`verification_sample_out.txt`: **no errors, no warnings**.

### Processing Plots Review
`processing_m11_ses-03.png`, `processing_m17_ses-08.png` (9 panels each):
1. Raw F / neuropil with trial spans shaded -- for m11 day 3 the laser was blanked between
   trials (F drops to the floor exactly in the un-shaded gaps), for m17 day 8 it was on
   throughout, matching the Methods' list of sessions imaged through the teleport.
2. dF/F and OASIS events for the same cell -- transients ride on a flat baseline and the
   traces are NaN (gaps) outside trials, as intended.
3. dF/F histogram -- mode at 0, median 0.021-0.022, long positive tail; no negative runaway.
4. r(dF/F, speed) per cell, with the 0.5 interneuron cut-off marked.
5. Position with trial start/end markers and the reward zone shaded -- every trial runs
   0 -> ~450 cm and the zone shifts at trial 30 on switch sessions.
6. Speed / lick / reward-delivery markers -- rewards always land where licking bursts occur.
7. Distance-to-zone and position with their discretised classes overlaid -- class
   boundaries coincide with the -50/-10/0/+10/+50 cm and 90/180/270/360 cm crossings.
8. Speed and lick discretisation -- boundaries coincide with 2/10/20/40 cm/s.
9. The converted trial's neural matrix (neurons sorted by peak) with position overlaid --
   sequential activation follows the animal's trajectory, so there is no temporal shift.

No anomalies.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| float32 instead of float64 | ~1.3x on the filter/deconvolution stages |
| ROI selection before dF/F | ~1.9x (only 53% of ROIs pass `iscell`) |
| 24-way session parallelism (spawn) | ~17x |

| Step | Time / Session (sample) | Estimated Total |
|---|---|---|
| NWB read (F, Fneu, behaviour) | 0.2-0.9 s (371 cells) | -- |
| dF/F + OASIS | 3.2-3.9 s (371 cells) | -- |
| Other (task variables, assembly) | ~0.2 s | -- |
| **Estimate for the full set** | scaling to the dataset mean of 912 cells/session gives ~10-12 s/session; 152 sessions / 24 workers x ~11 s ~= **1.5 min + ~0.5 min to pickle 9.6 GB** | **~2 min** |
| **Actual full run** | 10-20 s/session | **2.4 min** (conversion 2.2 min + 14 s write) |

Well under the 15 min budget, so no further optimisation was needed.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| reward_zone_distance | 0.7174 | 0.5776 | 0.1429 |
| position | 0.8388 | 0.6847 | 0.2000 |
| speed | 0.6509 | 0.5633 | 0.2000 |
| lick | 0.8234 | 0.7961 | 0.5000 |
| reward_zone_location | 0.9666 | 0.9306 | 0.3333 |
| reward_outcome | 0.9350 | 0.6140 | 0.5000 |

Loss decreased monotonically over all 200 epochs; every output is above chance.

### Choice of neural signal (dF/F vs deconvolved events)
The reference `preprocessing.dff` produces both dF/F and OASIS-deconvolved "events".
The paper's own decoder (Fig. 3) uses events, so both were converted for these two
sessions and run through `train_decoder.py`:

| Output | Validation acc, **events** | Validation acc, **dF/F** |
|---|---|---|
| reward_zone_distance | 0.475 | **0.578** |
| position | 0.604 | **0.685** |
| speed | 0.505 | **0.563** |
| lick | 0.745 | **0.796** |
| reward_zone_location | **0.941** | 0.931 |
| reward_outcome | 0.546 | **0.614** |

dF/F wins on five of six outputs, several by a wide margin. This is expected: OASIS
deconvolution deliberately strips the calcium indicator's temporal integration, which is
exactly the information a *per-timepoint* linear decoder relies on (the paper's own
decoder integrates over a whole trial set, not per frame). The paper itself uses dF/F
rather than events wherever the instantaneous amplitude matters (spatial peak firing,
sequence analysis, field definition). **dF/F was therefore chosen as the stored `neural`
signal**; `--signal events` reproduces the alternative.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/converted_data.pkl --full --nproc 24` -- 2.4 min, no errors.

### Output Files
- `converted_data.pkl`: 9.63 GB (152 sessions, 12,135 trials, 2,576,026 timepoints)
- `conversion_full_out.txt`, `verification_full_out.txt`: created; verification reports
  **no errors and no warnings**.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data (NWB scan) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 11 switch mice | -- | 11 (m3,m4,m7,m11-15,m17-19) | 11 | yes |
| Sessions | 14/mouse, m11 from day 3 | -- | 152 | 152 | yes |
| Switch sessions (days 3,5,7,8,10,12,14) | "n = 77 sessions, 11 mice, seven switch days" | `sessions_dict` | 77 | 77 | yes |
| Total neurons | -- | `iscell` + r(dF/F,speed)>0.5 | 138,678 `iscell` | 138,269 | yes |
| Neurons/session | 155-2172 | -- | 155-2341 (`iscell`) | 154-2323, mean 909.7 | close (see Step 4) |
| Trials (total) | 12,376 | -- | 12,216 complete trials | 12,135 (12,216 - 81) | close (see Step 4) |
| Trials/session (mean +- s.d.) | 80.5 +- 7.4 (14 mice) | -- | 80.37 +- 6.16 | 79.84 +- 6.86 | yes |
| Lick-error trials removed | 81 (0.65%) | `correct_lick_sensor_error` | 81 at thr = 0.30 | 81 | **exact** |
| Interneurons removed | 0.42 +- 0.85% of cells | r > 0.5 | -- | 409 cells, 0.29% overall, per-session 0.35 +- 0.61% | yes |
| Reward omission rate | ~15% | -- | 15.34% of trials | 15.36% of trials (0.158 of timepoints) | yes |
| Reward zone positions | A 80-130, B 200-250, C 320-370 | `reward_zone_dict` X/Y/Z | zone-entry positions agree on 10,394/10,394 trials | same | **exact** |
| Reward switch trial | after 30 trials | `change_trial=30` | confirmed on every switch session | 30 | yes |
| Frame period | ~64.5 ms (15.5 Hz) | -- | 64.4836 ms, identical in all 152 files | 64.4836 ms | yes |
| Track length | 450 cm | 0-450 binning | position max 449.9 | position classes span 0-4 | yes |
| ENV2 fraction of trials | ENV2 from day 8 (half the days) | -- | 49.03% | 48.93% | yes |
| Reward-zone identity balance | counterbalanced across mice | -- | A 4186 / B 4010 / C 4020 | 0.332 / 0.336 / 0.333 of timepoints | yes |

### Converted-data output distributions (fraction of timepoints)
| Output | Distribution |
|---|---|
| reward_zone_distance | 0.256, 0.102, 0.073, 0.238, 0.021, 0.072, 0.238 |
| position | 0.217, 0.177, 0.231, 0.226, 0.149 |
| speed | 0.117, 0.087, 0.134, 0.319, 0.343 |
| lick | 0.777, 0.223 |
| reward_zone_location | 0.332, 0.336, 0.333 |
| reward_outcome | 0.158, 0.842 |

Input ranges: `time_from_trial_start` [-0.0645, 216.5] s (the 216 s maximum is a single
trial in which the mouse stopped running; trial-duration median 12.1 s), `environment`
[0,1], `trial_number` [0,99], `previous_trial_rewarded` [0,1].

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There are no errors and no warnings to address.

### Check 2: Independent sanity checks (`cache/sanity_checks.py`)
This script re-reads the NWB files with `pynwb` and recomputes every quantity from
scratch -- it does not import `convert_data.py`; the dF/F is re-derived directly from the
Methods text in a separately written function -- then compares with `np.allclose`.
It covers 8 sessions (including the smallest, the largest, and 2-plane sessions) and 7
trials per session: **785 checks, 0 failures**.

| Stream | Check | Result |
|---|---|---|
| neural | full trial block `neural[trial] == reference dF/F[:, start-1:stop-1]` (`atol=1e-4`) | PASS (max abs diff ~1e-5, i.e. float32 round-off) |
| neural | spot checks at `[3,10]`, `[n/2,T/2]`, `[n-1,T-1]` | PASS |
| neural | `iscell` count, neuron count after interneuron removal | PASS |
| neural | `len(brain_region_idx[session]) == n_neurons` | PASS |
| input | `input[0] == ts[window] - ts[trial_start]` | PASS |
| input | `input[1] == unique(environment)` within the trial | PASS |
| input | `input[2] == original trial index` | PASS |
| input | `input[3] == isreward` of the preceding trial (0 for trial 0) | PASS |
| output | distance class recomputed from position and the scene-derived zone | PASS |
| output | position class `== digitize(pos,[90,180,270,360])` | PASS |
| output | speed class `== digitize(speed,[2,10,20,40])` | PASS |
| output | lick `== (lick > 0)` | PASS |
| output | zone location `== 'ABC'.index(scene label)` | PASS |
| output | reward outcome `== any(reward) and any(rzone)` | PASS |
| cross | on rewarded trials, reward was delivered **inside** the assigned reward zone | PASS |
| curation | dropped trials are exactly the lick-sensor-error trials | PASS |
| structure | trial lengths, subject index, trial counts | PASS |

### Check 3: Reference code comparison
| Stage | Reference | This conversion | Same? |
|---|---|---|---|
| (a) Data loading | `create_sess` -> suite2p `F`,`Fneu`,`iscell` + `vr_align_to_2P` VR interpolation onto imaging frames | NWB `Fluorescence`/`Neuropil`/`PlaneSegmentation.iscell` + the already-aligned `BehavioralTimeSeries` (the NWB files are the export of exactly that `sess`) | yes |
| (b) Neuron filtering | `iscell` from suite2p GUI curation; Methods: exclude r(dF/F, speed) > 0.5 | identical (`iscell[:,0]==1`, then r > 0.5); planes pooled as in the paper | yes |
| (b) Trial filtering | `correct_lick_sensor_error` NaNs lick-error trials; Methods threshold 0.30 | same criterion at 0.30, but the whole trial is **dropped** because `lick` is a required output and NaNs are not allowed | justified difference |
| (c) Temporal alignment | trial window `[trial_start-1, teleport-1)` in `dff` and `glmUtils.get_timeseries_data` | identical for every stream | yes |
| (d) Binning | none -- native imaging frames (`get_timeseries_data`, GLM "sampled at ~15.5 Hz") | native imaging frames, 64.4836 ms | yes |
| (d) dF/F | `preprocessing.dff`: neuropil subtract 0.7 + per-trial mean added back, maximin baseline (`nansmooth [0,15]`, `minimum_filter1d(300)`, `maximum_filter1d(300)`), `(F-base)/abs(base)`, `nansmooth(2)` | line-for-line port | yes |
| (d) events | `dcnv.oasis(dff, 2000, tau, fs)` per trial | same, `tau = 0.7` (the function default; `sess.s2p_ops['tau']` is not in the NWB) | yes (tau assumed) |
| (e) Input construction | `get_timeseries_data` builds `trials`, `pos`, `speed`, `licks`, `rewarded`, `omission` per frame | `time_from_trial_start`, `environment`, `trial_number`, `previous_trial_rewarded` -- the set prescribed by the decoder task; `trial_number` is the reference's `trial_ids` | prescribed by task |
| (e) speed >2 cm/s mask | `get_timeseries_data(use_speed_thr=2)` drops slow samples | **not applied** -- the decoder task requires a `< 2 cm/s` speed class | justified difference |
| (f) Output construction | `get_reward_zones` (zone coords/labels), `get_trial_types` (`isreward`, `morph`), `licks[licks>0]=1` | identical derivations, then discretised per the decoder task | yes |

Differences and their reasons (all three are forced by the decoder specification):
1. **No 2 cm/s speed mask** -- removing those samples would delete the entire `speed` class 0.
2. **Lick-error trials dropped rather than NaN-ed** -- outputs must be finite integers.
3. **dF/F stored rather than deconvolved events** -- both are outputs of the reference
   `preprocessing.dff`; dF/F decodes better per timepoint (Step 8) and is the signal the
   paper uses wherever instantaneous amplitude matters.

### Check 4: Key statistics comparison
See the Step 9 table. Every statistic available in the paper is reproduced. Two figures
required investigation:
- **12,376 vs 12,216 trials.** Investigated by re-counting `trial_start` and `teleport`
  flags in all 152 files: they are equal in every file, so there are no truncated trials
  in the NWB export to recover. 12,216 is the number of complete imaged trials present in
  DANDI; the 1.3 % shortfall is in the published files, not in the conversion. The
  *derived* statistic that depends on the same criterion -- 81 lick-error trials -- matches
  the paper exactly, so the trial set itself is right.
- **Max neurons/session 2341 vs 2172.** Per-plane counts for the 2-plane mice max at 1434
  and single-plane sessions max at 1780, so the paper's 2172 cannot be a pooled count from
  these 11 mice either; it most likely includes the 3 fixed-condition mice that are not in
  DANDI. The lower bound (155) matches exactly.

### Check 5: Edge cases
| Edge case | Handling | Verified |
|---|---|---|
| 10 files with one more ophys frame than VR samples | all streams trimmed to the common length | frames trimmed = 10, one per affected file |
| teleport frame carries an interpolated, meaningless `pos` (jumps to ~200 then -50) | excluded by the `[start-1, stop-1)` window | last-sample position range 436.6-449.4 cm across all trials |
| `position` slightly negative at the first window sample | kept (range -9.1 to 0.7 cm); falls in position class 0 and distance class 0/1 exactly as a position just before 0 cm should | behaviour scan |
| first trial of a session has no previous trial | `previous_trial_rewarded = 0`, documented | sanity check asserts this |
| `reward_zone` flag is absent on omission trials | reward-zone identity comes from the scene name, not the flag; `isreward` uses the reference's `any(reward) and any(rzone)` | zone assignment verified against the flag on all 10,394 trials that have it |
| negative running speed (mouse rolls backwards, min -6.4 cm/s) | falls in speed class 0 (`< 2 cm/s`), which is correct | -- |
| very long trials (max 3359 frames = 216 s) | kept; the decoder allows variable T | -- |
| sessions with < 30 trials (min 41) | `zone_labels_from_scene` clips the pre-switch block to `min(30, ntrials)` | no session has fewer than 41 trials |
| trial starting at frame 0 / ending past the last frame | skipped with a counter | count = 0 |
| `autoreward` column all zeros | not used | -- |
| 2-plane mice | ROI tables and both `RoiResponseSeries` concatenated in table order | `iscell` totals match the independent scan |

### Iterations
1. First full run crashed on `sub-m17_ses-04` (behaviour 22,790 samples vs ophys 22,791).
   Investigated -> 10 files affected, all 2-plane. Fixed by trimming to the common length,
   re-ran the full conversion, and re-ran every check above. No further issues.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, ~8 min).

### Training Progress
- Loss decreasing: **Yes**, monotonically, 3.374 (epoch 1) -> 0.655 (epoch 200);
  held-out test loss 0.701.

### Decoder Results (Full)
| Output | #classes | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance |
|--------|---|---|-------------|--------|-------|
| reward_zone_distance | 7 | 0.1429 | 0.7895 | **0.6229** | 4.36x |
| position | 5 | 0.2000 | 0.8860 | **0.7645** | 3.82x |
| speed | 5 | 0.2000 | 0.7293 | **0.6325** | 3.16x |
| lick | 2 | 0.5000 | 0.7925 | **0.7556** | 1.51x |
| reward_zone_location | 3 | 0.3333 | 0.9647 | **0.8738** | 2.62x |
| reward_outcome | 2 | 0.5000 | 0.9521 | **0.6034** | 1.21x |

`predictions.png` shows predicted vs. true traces on held-out trials: position and
distance-to-zone are tracked closely across the whole trial, speed and licking follow the
true traces, and the two per-trial outputs are mostly constant and correct.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
Every output is above chance, and four of six exceed 2.5x chance. Two are below 1.5x-2x
chance and were investigated:

**`reward_outcome` (0.603, 1.21x chance).** This is the expected ceiling, not a bug:
- On the 43.1 % of in-trial timepoints that precede the reward zone (distance classes
  0-2), *nothing* in the brain can yet report whether this trial will be rewarded --
  reward is delivered at the first lick inside the zone. Those timepoints are at chance
  by construction. Even a decoder that were perfect from zone entry onward would score
  about 0.43x0.5 + 0.57x1.0 = 0.78.
- An **independent** per-session check (`cache/diagnose_reward_outcome.py`: PCA-100 +
  balanced logistic regression fit separately on each of 6 sessions with an 80/20 trial
  split) reproduces the same number, 0.617 overall, and confirms the predicted split:
  0.565 on pre-zone timepoints vs 0.652 from the reward zone onward. A model with far more
  capacity per session therefore extracts no more outcome information than the shared
  decoder does, so the limit is in the data, not in the conversion.
- The paper likewise reports only modest rewarded-vs-omission differences (Fig. 6: a
  reward-vs-omission index computed over *trial-averaged* activity of a selected
  subpopulation), consistent with weak single-timepoint decodability.

**`lick` (0.756, 1.51x chance).** Licking is a fast behaviour (individual licks at up to
~8 per 64 ms frame) and the neural signal is dF/F with an indicator time constant of
several hundred ms, so frame-resolution lick decoding is intrinsically limited. Still,
0.756 balanced accuracy on a 78/22 class split is a large effect, and licking is
concentrated in the reward zone, where the decoder is accurate.

### Check 2: Accuracy comparison to the paper
| Variable | Achieved (validation balanced acc) | Paper's reported value |
|---|---|---|
| reward_zone_distance (RR position) | 0.623 (7 classes, chance 0.143) | Fig. 3b: mean *decode score* `cos(y-yhat)` ~0.4-0.8 per session for RR cells, vs ~0 for shuffles. Not a classification accuracy -- the paper decodes a **continuous circular** variable with circular-linear regression from **selected cell subpopulations** (RR/TR/non-RR place cells), per session, with 10-fold CV. |
| position | 0.765 (5 classes, chance 0.200) | not reported as an accuracy; position is the strongest GLM predictor (Fig. 7d) |
| speed / lick | 0.633 / 0.756 | not reported as accuracies; both are significant GLM predictors (Fig. 7d) |
| reward_zone_location | 0.874 (3 classes, chance 0.333) | not reported |
| reward_outcome | 0.603 (2 classes, chance 0.500) | not reported as an accuracy; Fig. 6 reports a reward-vs-omission index |

The paper reports **no categorical decoding accuracies**, so no number can be matched
directly. The closest comparison is qualitative and it agrees: reward-relative position is
decodable far above chance from the CA1 population across the whole track, which is the
central claim of Fig. 3. Converting the paper's Fig. 3a example to the same footing --
`cos` decode score ~0.5-0.8 corresponds to a typical circular error well under a quarter of
the track -- is consistent with 62 % correct in 7 reward-relative distance bins, three of
which (classes 2, 3, 4) span only 20 cm in total.

### Check 3: Train vs validation gap
| Output | Train | Val | Ratio |
|---|---|---|---|
| reward_zone_distance | 0.790 | 0.623 | 1.27 |
| position | 0.886 | 0.765 | 1.16 |
| speed | 0.729 | 0.633 | 1.15 |
| lick | 0.793 | 0.756 | 1.05 |
| reward_zone_location | 0.965 | 0.874 | 1.10 |
| reward_outcome | 0.952 | 0.603 | **1.58** |

Only `reward_outcome` exceeds 1.5x. This is ordinary overfitting of a **per-trial** label,
not leakage: the label is constant over the ~200 timepoints of a trial, so a per-session
linear projection of ~900 neurons onto 100 PCs can memorise which training trials were
rewarded from trial-specific activity. Leakage was ruled out explicitly:
- the train/validation split is by whole trial (`train_validate_decoder`), and each trial's
  neural data is sliced from disjoint frame windows, so no timepoint appears in both sets;
- the dF/F baseline is computed **within each trial independently** (the paper's maximin
  procedure), so no normalisation statistic is shared between trials;
- `previous_trial_rewarded` is the *preceding* trial's outcome and omissions are drawn
  independently per trial, so it carries essentially no information about the current
  trial: across all 12,135 trials P(rewarded | prev rewarded) = 0.850 vs
  P(rewarded | prev omitted) = 0.827, phi = 0.023. Chance for the balanced metric is
  unaffected, and this input cannot account for a 0.95 training accuracy;
- the per-trial gap is reproduced by the independent per-session logistic regression in
  `cache/diagnose_reward_outcome.py`, which has no access to the shared decoder at all.

### Issues Found and Resolved
- (Step 9/10) 10 NWB files with a frame-count mismatch -> streams trimmed; fixed and re-run.
- (Step 8) Deconvolved events decoded worse than dF/F -> switched the stored signal to
  dF/F, documented in Step 8.
- No other issues were found; all 785 independent sanity checks pass.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created (`cache/README_CACHE.md`)
- [x] All files organized
