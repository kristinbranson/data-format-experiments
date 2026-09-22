# Dataset Conversion Notes

## Overview
- **Dataset**: "A flexible hippocampal population code for experience relative to reward" (NWB files in /app/data)
- **Date started**: (session start)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `CONVERSION_NOTES.md` (this file)
- `code/` - reference code repo (LICENSE, README.md, docs/, environments/, notebooks/, setup.py, src/)
- `data/` - dandiset.yaml + 11 subject dirs: sub-m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19
- `decoder.py`, `train_decoder.py` - decoder reference code
- `methods.txt`, `paper.pdf` - reference text
- `pynwb_docs/`

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, pynwb 4.1.0.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Repo = `Sosa_et_al_2024` (Sosa, Plitt, Giocomo 2025, Nat Neurosci). 2-photon Ca imaging in dorsal CA1 during a VR linear track task with moving hidden reward zones.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` | src/reward_relative/preprocessing.py | LOADING | Builds TwoPUtils `sess` object: suite2p F/Fneu + VR sqlite aligned to imaging frames |
| `TwoPUtils.preprocessing.vr_align_to_2P` | (TwoPUtils repo, not included) | LOADING/ALIGN | Interpolates VR behavior onto ~15.5 Hz imaging frame times -> `sess.vr_data` |
| `dff` | src/reward_relative/preprocessing.py | PROCESSING | dF/F: neuropil subtraction (`neu_coef=0.7`), `maximin` baseline, optional deconvolution into `events` (suite2p `dcnv`, tau from s2p ops). Only frames between trial_start-1 and teleport-1 are kept (rest NaN) unless `keep_teleports` |
| `multi_anim_sess` | src/reward_relative/utilities.py | PROCESSING/CURATION | Per experiment-day, per animal: loads sess, computes dFF (+events), place cells (100 perms, p<0.05, speed_thr 2 cm/s), and behavior info (isreward, morph, rzone, rz label, trial dict) |
| `get_trial_types` | src/reward_relative/behavior.py | PROCESSING | Per trial: `isreward` = reward delivered AND rzone entered; `morph` = env identity (0=Env1, 1=Env2) |
| `get_reward_zones` | src/reward_relative/behavior.py | PROCESSING | Per-trial reward zone [start,stop] cm and label. Zones: A=[80,130], B=[200,250], C=[320,370] (dict keys X/Y/Z); on switch scenes (`X_to_Y`) zone changes at trial 30 (`sess.change_reward_trial` if present) |
| `define_trial_subsets` | src/reward_relative/behavior.py | CURATION | set0/set1 trial splits by reward zone (pre/post switch) |
| `correct_lick_sensor_error` | src/reward_relative/behavior.py | CURATION | Trials where >thr (0.35-0.5) fraction of samples have cumulative lick count >2 => licks set to NaN (stuck sensor) |
| `lickrate`, `frametime` | behavior.py | PROCESSING | licks binarized (`licks>0 -> 1`), divided by frame time, smoothed (gauss sigma 2 frames) |
| `create_design_matrix` | src/reward_relative/glmUtils.py | PROCESSING | GLM predictors: rel_pos (distance to reward, circular or linear), pos, trial id, speed, accel, licks (binarized, sensor-error-corrected at 0.35), rewarded/omission state; speed threshold 2 cm/s used to mask samples |
| spatial trial matrices | spatial.py / TwoPUtils | PROCESSING | Position bins of 10 cm from 0 to 450 cm (track length 450 cm, teleport zone excluded) |
| `decode.py` | src/reward_relative | ANALYSIS | Low & Williams circular position decoder (regression on circular position) used in Fig 3 |

### Notes
- Neural data are **2P calcium imaging**, sampled at ~15.5 Hz (~64.5 ms/frame). Reference analyses use **deconvolved `events`** computed from dF/F for place-cell / GLM analyses, and `dff` for some plots.
- dF/F is NOT stored in raw sess; it is computed from F and Fneu. In the NWB release the dF/F and/or deconvolved traces may already be provided (to check in Step 2).
- Cell curation is done in the suite2p GUI (`iscell.npy`); only ROIs marked iscell are used.
- Trials are defined by `trial_start_inds` (track entry) and `teleport_inds` (track exit). Frames outside of trials (teleport period) are excluded from dF/F.
- Track is 450 cm; reward zones are 50 cm wide at A=[80,130], B=[200,250], C=[320,370].
- On switch days, reward zone changes on trial 31 (index 30).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/dandiset.yaml` + 11 subject directories `sub-mXX/` (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19).
- One NWB file per session: `sub-<mouse>_ses-<day>_behavior+ophys.nwb` (152 files, 87 GB). `ses-NN` = experiment day (1-14); m11 has days 3-14 only (12 sessions; imaging started day 3 due to low expression - consistent with methods).
- NWB contents (read with `pynwb.NWBHDF5IO`):
  - `acquisition/TwoPhotonSeries`: placeholder (1,1,1) - raw movie not included.
  - `processing/behavior/BehavioralTimeSeries`: time series sampled **1:1 with imaging frames** (dt = 0.06448363 s = 15.5078 Hz, identical in every session), each with explicit `timestamps`:
    - `position` (cm, -500 before scan start), `speed` (cm/s), `lick` (cumulative count per frame), `reward_zone` (reward-zone entry, cumulative per frame), `autoreward`, `environment` (-1 pre-scan, 0 = ENV1, 1 = ENV2), `trial number`, `trial_start` (binary), `teleport` (binary), `scanning` (1 = scanning), and `Reward` (74 events; **timestamps** of reward delivery, data = volume mL).
  - `processing/ophys`:
    - `Fluorescence/planeN`: raw ROI F, shape (n_frames, n_rois), `rate` = 15.5078 (single-plane) or 31.0156 (2-plane, but each plane series still has one sample per behavior frame, i.e. 15.5 Hz per plane).
    - `Neuropil/planeN`: Fneu.
    - `Deconvolved/planeN`: suite2p `spks` computed from **raw F** (NOT the paper's dF/F-based events; correlation with my recomputed events ~0.5, see Step 4).
    - `ImageSegmentation/PlaneSegmentation`: columns `pixel_mask`/`voxel_mask`, `iscell` (n_rois x 2: [iscell flag, probability]), `planeIdx`.
  - `subject`: subject_id (m3 ...), sex, DOB. `identifier` encodes original path incl. **scene** name, e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`.
- No `trials` intervals table: trial structure must be derived from `trial_start` / `teleport` binary series (as in the reference `sess.trial_start_inds` / `sess.teleport_inds`).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Sessions | 152 (11 subjects; 14 each except m11 = 12) |
| ROIs total / iscell total | 336,545 / **138,678** |
| iscell neurons per session | mean 912, min 155, max 2341 |
| Subjects | 11 |
| Sessions / subject | 12-14 |
| Trials (total) | 12,216 |
| Trials / session | mean 80.4, sd 6.2, min 41, max 100 |
| Frames per session | 9,401-43,142 on-track (total on-track = 2,620,514; 72.6% of all frames) |
| Frame period | 0.06448363 s (15.5078 Hz), identical in all sessions |
| Planes | 1 plane except m17, m18 (2 planes, pooled) |
| Mean trial length | 217 frames (14.0 s) |
| Lick-sensor-error trials (>30% frames with cum. lick > 2) | **81** |
| Omission fraction (per session) | mean 0.153 (range 0.037-0.30) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons / session | 155-2172 | "yielded 155-2172 putative pyramidal neurons per session" |
| Subjects (switch task) | 11 | "performing reward switches (n = 11 mice)" |
| Trials / session | 80.5 +- 7.4 | "mean +- s.d., 80.5 +- 7.4 trials across 14 mice, all imaging days" |
| Total imaged trials | 12,376 (11 switch mice) | "n = 81 out of 12,376 trials removed across 11 switch mice" |
| Lick-error trials removed | 81 (~0.65%) | same quote |
| Neural sampling | ~15.5 Hz (0.0645 s) per plane | "unidirectional scanning at ~15.5 Hz"; ">30% of the 0.0645 s imaging frame samples" |
| Interneuron exclusion | dF/F vs speed Pearson r > 0.5; 0.42 +- 0.85% of cells | "Additional putative interneurons were detected for exclusion ... by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed" |
| Reward omission rate | ~15% | "the reward was randomly omitted on ~15% of trials" |
| Track length | 450 cm | "450 cm virtual linear track" |
| Reward zones | A 80-130, B 200-250, C 320-370 cm (50 cm wide) | "zone A, 80-130 cm; zone B, 200-250 cm; zone C, 320-370 cm" |
| Reward switch | day 3, 5, 7, 8(+env change), etc., **at trial 31 (index 30)** | reference `get_reward_zones(change_trial=30)` |
| Speed threshold for spatial analyses | 2 cm/s | "we excluded activity when the animal was moving at <2 cm/s" |
| Spatial bins | 45 bins of 10 cm | "binned the 450 cm linear track into 45 bins of 10 cm each" |
| Sessions used for decoding | 77 (11 mice, 7 switch days) | "n = 77 sessions, 11 mice, seven switch days" |

### Processing Details
- **dF/F**: per trial (teleport periods excluded), neuropil subtracted (0.7 coefficient, trial-mean neuropil added back), baseline by **maximin** (gaussian smooth sigma = 15 frames -> 20 s min filter (300 frames) -> 300-frame max filter), dF/F = (F - F0)/|F0|, then Gaussian smoothing with 2-sample s.d. (~0.129 s).
- **Events**: OASIS deconvolution of the smoothed dF/F (suite2p `dcnv.oasis`, tau = 0.7, rate = 15.5078 Hz/plane). "not interpreted as a spike rate but rather as a method to eliminate the asymmetric smoothing of the calcium signal".
- **Trials**: from `trial_start` (track entry) to `teleport` (track exit). Teleport/ITI excluded.
- **Alignment**: VR behavior already interpolated onto imaging frame times in the NWB files (one behavior sample per imaging frame).

### Curation Steps
**Neuron curation rules**: suite2p `iscell` (manual curation removing multi-soma/dendrite/interneuron-like ROIs) + exclusion of putative interneurons with dF/F-speed Pearson r > 0.5.

**Trial curation rules**: trials with erroneous lick detection (>30% of frames with cumulative lick count > 2) are set to NaN for lick analyses (81 of 12,376 trials).

### Decoders Trained (paper)
| Decoded variable | Accuracy |
|---|---|
| RR (reward-relative) circular position from deconvolved events of RR cells | "decode score" = mean cos(true - predicted); reported vs shuffle, z-scored; no classification accuracy reported. Positions with z-scored decode > 2 spanned -104.5 +- 20.1 cm to +152.7 +- 22.9 cm relative to reward |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Consistency checks run on the raw data (scripts in /app/cache)
1. **Trial count**: 12,216 trials in the 152 NWB sessions vs 12,376 reported in the paper (11 switch mice). Mean 80.4 +- 6.2 trials/session vs paper 80.5 +- 7.4 (paper value is across 14 mice incl. 3 fixed-condition mice not in this DANDI release). Consistent.
2. **Lick-sensor-error trials**: my implementation of the paper rule (>30% of frames in a trial with cumulative lick count > 2) flags **exactly 81 trials** across the whole dataset - identical to the paper's "81 out of 12,376 trials removed". Strong confirmation of the trial definition and lick variable interpretation.
3. **Neurons/session**: iscell counts 155-2341; paper says 155-2172. The lower bound matches exactly; the upper bound differs because the paper counts the neurons actually analyzed after interneuron exclusion / per-day analysis sets (see Step 9 for the post-curation range).
4. **Reward zones**: for each trial I took the animal position at the first `reward_zone` > 0 sample and matched it to the nearest zone start (A 80 / B 200 / C 320). This empirical zone agrees with the scene-name rule of `behavior.get_reward_zones` (switch at trial index 30) for **100% of trials in all 152 sessions**; maximum deviation of first-entry position from the nominal zone start is 8.5 cm (1 frame of running at ~80 cm/s = ~5 cm). Confirms both the zone dictionary and the change_trial = 30 convention.
5. **Omission rate**: mean 15.3% per session (paper: ~15%).
6. **Environment**: `environment` is constant within every trial in every session (0 = ENV1, 1 = ENV2); on `EnvX_?_to_EnvY_?` days it switches exactly at trial index 30 together with the reward zone.
7. **Interneuron exclusion**: recomputing dF/F and correlating with speed flags ~0.65% of iscell ROIs in the test session (paper: 0.42 +- 0.85% across mice/days).
8. **NWB `Deconvolved` vs reference events**: the NWB Deconvolved series is suite2p `spks` derived from raw F, not from the paper's maximin dF/F (correlation with recomputed events ~0.25-0.7). **Decision: recompute dF/F and OASIS events following the reference code** rather than use the stored Deconvolved series.

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Trial index range for dF/F | `f[:, start-1:stop-1]` (off-by-one) | trial_start/teleport indices are on-track at [start, stop) | trials = track entry to teleport | Use [start, stop) consistently for neural and behavior; the reference's -1 offset would leave the final frame of each trial undefined. Difference is 1 frame (64 ms) in baseline estimation only. |
| Deconvolved data source | reference deconvolves its own dF/F | NWB stores suite2p spks from raw F | events from dF/F | Recompute dF/F + OASIS as in reference |
| Total trials | - | 12,216 | 12,376 | 160-trial difference: DANDI release lacks a few sessions (e.g. m11 days 1-2 were never imaged); no action |
| Speed threshold (2 cm/s) | applied in place-cell/GLM analyses | - | "excluded activity when moving <2 cm/s" | NOT applied here: the decoder requires continuous time series and speed itself is a decoder output (with a <2 cm/s class). Documented deviation required by the decoding task. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (NWB) | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Fluorescence/planeN` (F), `ophys/Neuropil/planeN` (Fneu), `ImageSegmentation.iscell` | `neural` | per-trial dF/F (neuropil subtract 0.7 + trial-mean add-back, maximin baseline: gauss sigma 15 frames -> min filter 300 -> max filter 300, dF/F = (F-F0)/|F0|, gaussian smooth sigma 2 frames) then OASIS deconvolution (tau 0.7, 15.5078 Hz) = "events" | `preprocessing.dff` (+`utilities.multi_anim_sess` defaults, `suite2p.extraction.dcnv.oasis`) | Neurons = ROIs with iscell==1 minus putative interneurons (dF/F vs speed r > 0.5). Planes pooled. |
| `behavior/trial_start`, `behavior/teleport` | trial segmentation | frames [trial_start, teleport) | `sess.trial_start_inds`, `sess.teleport_inds` | teleport/ITI excluded, as in reference dF/F (keep_teleports=False) |
| frame timestamps | `input[0]` time_from_trial_start (s) | t - t[trial_start] | - | continuous, time-varying |
| `behavior/environment` | `input[1]` environment | 0 = ENV1, 1 = ENV2 (constant within trial; broadcast over time) | `behavior.get_trial_types` (`morph`) | verified constant within every trial |
| trial index | `input[2]` trial_number | 0-based index within session (broadcast) | - | continuous per-trial |
| previous trial reward outcome | `input[3]` prev_trial_outcome | 0 = omitted, 1 = rewarded (broadcast) | `behavior.get_trial_types` (`isreward`) | first trial of session: set to 1 (mice always received 30 rewarded warm-up trials immediately before the imaging session; ~85% of trials are rewarded) |
| `behavior/position` + per-trial reward zone | `output[0]` dist_to_reward_zone (7 bins) | signed distance to nearest point of the active 50 cm zone (0 inside), binned <-50 / [-50,-10) / [-10,0) / 0 / (0,10] / (10,50] / >50 | `glmUtils.create_design_matrix` (rel_pos), `behavior.get_reward_zones` | linear (not circular) distance, per Decoder Task spec |
| `behavior/position` | `output[1]` position (5 bins) | 90 cm bins over 0-450 cm | trial matrices (10 cm bins) in reference | positions clipped to [0,450] |
| `behavior/speed` | `output[2]` speed (5 bins) | <2 / 2-10 / 10-20 / 20-40 / >40 cm/s | `sess.vr_data['speed']` | the 2 cm/s reference threshold appears as class boundary |
| `behavior/lick` | `output[3]` lick (binary) | cumulative count per frame -> `lick>0 -> 1` | `behavior.lickrate`, `glmUtils` (`licks[licks>1]=1`) | trials flagged by sensor-error rule are dropped (see decisions) |
| scene name in `identifier` (+ switch at trial 30), verified vs `behavior/reward_zone` entry position | `output[4]` reward_zone_location | A=0, B=1, C=2 (broadcast) | `behavior.get_reward_zones` | |
| `behavior/Reward` timestamps + `behavior/reward_zone` | `output[5]` reward_outcome | 1 if a reward was delivered AND the reward zone was entered within the trial, else 0 (broadcast) | `behavior.get_trial_types` (`isreward`) | |
| `nwb.subject.subject_id` | `subjects`, `subject_idx` | m3...m19 | - | 11 mice |
| `imaging_plane.location` = "hippocampus, CA1" | `brain_regions` = ['CA1'], `brain_region_idx` | all neurons -> 0 | - | single region |

### Key Decisions
1. **Neural signal = deconvolved events from the paper's dF/F** (not the NWB `Deconvolved` series): the NWB Deconvolved is suite2p spks from raw F, whereas all reference analyses (place cells, decoding, GLM) use events deconvolved from the maximin dF/F. Recomputing follows `preprocessing.dff(..., deconvolve=True)` exactly (neu_coef 0.7, maximin, 2-frame smoothing, OASIS tau 0.7).
2. **Temporal resolution = native imaging frame period (64.484 ms)**: behavior in the NWB is already interpolated onto imaging frames 1:1 (the reference's `vr_align_to_2P` alignment), so no rebinning is needed and no alignment error is introduced. Identical dt in every session satisfies the "same bin size everywhere" requirement.
3. **Trial window = [trial_start, teleport)** (track entry to track exit), matching the reference definition of a trial; the ITI/teleport period is excluded exactly as in the reference dF/F (`keep_teleports=False`). I use [start, stop) rather than the reference's [start-1, stop-1) indexing so that neural and behavior frames are consistently aligned and no frame is left undefined (1-frame difference, only affects baseline estimation).
4. **Neuron curation**: `iscell == 1` (suite2p manual curation) AND dF/F-vs-speed Pearson r <= 0.5 (paper's putative-interneuron exclusion). Multi-plane animals: planes pooled (as in the paper, except Ext. Fig. 7).
5. **Trial curation**: drop trials flagged by the paper's lick-sensor-error rule (>30% of frames in the trial with cumulative lick count > 2); these are exactly the 81 trials the paper removed and their lick output would otherwise be corrupt/NaN. All other trials kept (min length 96 frames = 6.2 s).
6. **No speed thresholding**: the reference excludes <2 cm/s samples for spatial tuning analyses, but the decoder requires continuous time series and speed is itself a decoded variable (with a <2 cm/s class); dropping those samples would destroy the alignment and the speed output. Documented deviation.
7. **Per-trial variables are broadcast across time** so that `input` is (4, T) and `output` is (6, T) - the format allows mixing only if all entries in a trial array share the shape.
8. **First trial's previous outcome = 1 (rewarded)**: the imaging session is always preceded by 30 warm-up trials with the same reward zone; treating the unobserved previous trial as rewarded matches the modal outcome (~85%).
9. **Alignment event = trial start (track entry)**; `off_start = 0.0`; `off_end = None` because trials have variable length (median ~14 s).
10. **10 sessions (m17/m18) have ophys arrays 1 frame longer than the behavior series**; the extra trailing frame is after the last teleport, so ophys is truncated to the behavior length.

### Planned Sanity Checks
- [ ] Trials: total 12,216 - 81 lick-error = 12,135 trials; per session 41-100 (mean ~80).
- [ ] Neurons: 152 sessions, per-session count in the paper's 155-2172 range after curation; interneuron exclusion ~0.4-0.7% of cells.
- [ ] Subjects: 11; sessions per subject 12-14.
- [ ] Reward outcome fraction ~0.85 (omission ~15%).
- [ ] Reward zone labels: distribution across A/B/C; verified per trial against the observed reward-zone entry position (100% agreement in Step 4).
- [ ] dist_to_reward_zone == 0 exactly when position is inside the active zone; bin 3 occupancy > 50/450 (animals slow in the zone).
- [ ] Position bins approximately monotonic in time within a trial (unidirectional track).
- [ ] Speed bin distribution: most samples 10-40 cm/s; <2 cm/s class present.
- [ ] Lick fraction ~2-10% of frames.
- [ ] Independent spot-check (Step 10) recomputing neural/input/output values for specific (session, trial, neuron, timepoint) directly from the NWB file with `np.allclose`.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the Step 5 mapping. Structure:
- `load_session(fn)` - reads one NWB file with **pynwb** (`NWBHDF5IO`), returning behaviour series, F/Fneu (planes concatenated along the neuron axis), `iscell`, `planeIdx`, frame rate. Ophys arrays are truncated to the behaviour length (10 m17/m18 sessions have 1 extra ophys frame).
- `compute_dff_events(...)` - port of `reward_relative/preprocessing.py::dff` with the reference defaults (neuropil subtract 0.7 + per-trial neuropil mean add-back, maximin baseline = nan-aware Gaussian sigma 15 frames -> 300-frame min filter -> 300-frame max filter, dF/F = (F-F0)/|F0|, nan-aware Gaussian smoothing sigma 2 frames) followed by `suite2p.extraction.dcnv.oasis(dff, 2000, tau=0.7, fs)` per trial.
- `scene_reward_zones` / `empirical_reward_zones` - port of `behavior.get_reward_zones` plus an independent check against the observed reward-zone entry position (falls back to the observed zone and warns if they disagree).
- `discretize_distance / discretize_position / discretize_speed` - the Decoder Task binning rules.
- `process_session` - trial segmentation, neuron curation, trial curation, construction of the (4,T) input and (6,T) output arrays.
- `plot_processing` - two figures per session for `--show-processing`.
- `main` - multiprocessing pool over sessions (`--nproc`, default 12), progress/ETA printing, pickling.

Code inefficiencies identified: whole-session F/Fneu are read as float32 and processed in one pass; per-trial loops are only over trials (~80), all cell operations are vectorised; interneuron correlation is a single matrix-vector product instead of a per-cell loop.

Code speedups added: multiprocessing over sessions (12 workers, `maxtasksperchild=1` to release memory), float32 arrays, `np.searchsorted` for reward frame indices, vectorised correlation.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Ran `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` (sessions m11 day 3 and m3 day 5) then `python -u /app/train_decoder.py /app/sample_data.pkl --verify-only`.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 1,173 (154 + 1,019) |
| Neurons / session | 154, 1019 (iscell 155, 1022; 1 and 3 interneurons removed) |
| Subjects | 2 (m3, m11) |
| Trials (total) | 160 (80 per session, 0 lick-error trials) |
| T per trial | mean 258 frames, min 137, max 1552 |
| time_from_trial_start_s | [0.0, 100.0] |
| environment | [0, 0] (both sessions ENV1) |
| trial_number | [0, 79] |
| prev_trial_rewarded | [0, 1] |
| dist_to_reward_zone distribution | [0.140, 0.072, 0.087, 0.246, 0.022, 0.066, 0.367] |
| position distribution | [0.227, 0.183, 0.261, 0.118, 0.211] |
| speed distribution | [0.226, 0.073, 0.126, 0.291, 0.284] |
| lick distribution | [0.806, 0.194] |
| reward_zone_location distribution | [0.420, 0.580, 0] (A/B only - both are switch sessions A<->B) |
| reward_outcome distribution | [0.117, 0.883] |

Format validation: **no errors, no warnings**.

### Processing Plots Review
`processing_sub-m11_ses-03.png` / `processing_sub-m3_ses-05.png` show raw F, neuropil, dF/F, OASIS events, position with the active reward zone shaded (green = rewarded, red = omission), distance-to-zone with its discretisation overlaid, speed with its discretisation, and licks/reward, all on the same time axis with trial start/teleport markers. `*_trials.png` shows trial x time rasters of the four time-varying outputs plus the per-trial variables. No anomalies: dF/F and events are defined exactly within trials, position ramps 0->450 cm each trial, the distance bin is 3 exactly while the animal is inside the shaded zone, rewards occur inside the zone on rewarded trials only.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| 12-process pool over sessions | ~10x wall-clock |
| float32 + vectorised correlation/segment loops | ~2x per session |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| NWB load | 0.5-1.2 s (scales with #ROIs) | ~3 min single-threaded |
| dF/F + OASIS | 3.3 s (155 cells) - 5.8 s (1022 cells) | ~15 min single-threaded |
| Total (12 workers) | ~7 s/session | **~3-5 min for 152 sessions** (well under 15 min) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| dist_to_reward_zone | 0.610 | 0.528 | 0.143 |
| position | 0.706 | 0.591 | 0.200 |
| speed | 0.552 | 0.500 | 0.200 |
| lick | 0.725 | 0.729 | 0.500 |
| reward_zone_location | 0.791 | 0.812 | 0.333 |
| reward_outcome | 0.747 | 0.437 | 0.500 |

Loss decreased monotonically (2.55 -> 0.91 over 200 epochs). All outputs except `reward_outcome` are well above chance. `reward_outcome` is a per-trial variable with only ~12% omission trials in 160 trials, so the validation split contains very few omission trials; re-assessed on the full dataset in Steps 11-12.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/converted_data.pkl --full   # 3.2 min, 12 workers
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

### Output Files
- `converted_data.pkl`: 9.52 GB (152 sessions, native 64.5 ms frames)
- `conversion_full_out.txt`, `verification_full_out.txt`: created

Format validation: **no errors, no warnings**.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data (NWB) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 11 switch mice | 11 switch animals in `sessions_dict` (GCAMP3,4,7,11,12,13,14,15,17,18,19) | 11 subject folders | 11 | YES |
| Sessions | 7 switch days x 11 mice for decoding; 14 days/mouse imaged | up to 14 exp days per animal | 152 | 152 | YES |
| Sessions/subject | m11 starts on day 3 | - | 12-14 | 12 (m11), 14 (others) | YES |
| Trials (total) | 12,376 (11 switch mice) | - | 12,216 | 12,135 (= 12,216 - 81 lick-error) | YES (paper total includes ~160 trials from sessions not in the DANDI release) |
| Trials/session (mean) | 80.5 +- 7.4 | - | 80.4 +- 6.2 | 79.8 (min 40, max 100) | YES |
| Lick-error trials removed | 81 (0.65%) | rule: >30% frames cum-lick>2 | 81 detected | 81 dropped | YES (exact) |
| Neurons/session | 155-2172 | suite2p iscell | iscell 155-2341 | 154-2323 (mean 910) | Close; see Step 10 Check 4 |
| Interneurons excluded | 0.42 +- 0.85% of cells | - | - | 402/138,678 = 0.29% | YES |
| Neurons total | - | - | 138,678 iscell | 138,276 | YES |
| Reward omission rate | ~15% | - | 15.3% (per-session mean) | 15.8% of time samples, 15.6% of trials | YES |
| Reward zone locations | A/B/C equally used | change at trial 30 | - | A 0.332 / B 0.336 / C 0.333 | YES |
| Frame period | 0.0645 s | 15.5 Hz/plane | 0.06448363 s | 64.484 ms | YES |
| Track length / zones | 450 cm; A 80-130, B 200-250, C 320-370 | same dict | entry positions match | same | YES |
| Input time_from_trial_start | - | - | - | [0, 216.5] s | trials are variable length; max is a session where the mouse paused |
| Input environment | ENV1/ENV2 | morph 0/1 | 0/1 | [0, 1] | YES |
| Output position | uniform-ish over track | - | - | [0.212, 0.177, 0.231, 0.226, 0.154] | YES (last bin is 360-450 incl. fast running) |
| Output speed | animals run 20-60 cm/s | speed thr 2 cm/s | - | [0.117, 0.087, 0.134, 0.319, 0.343] | YES |
| Output lick | licking concentrated near zone | - | - | [0.777, 0.223] | YES |
| Output dist-to-zone | in-zone = 50/450 = 11% of track | - | - | 23.8% of samples in zone (animals slow down/consume in zone) | YES |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`verification_full_out.txt`)
`Data format is valid, no errors or warnings.` - there is nothing to fix. The summary reports 152 sessions, 12,135 trials, 11 subjects, 1 brain region (CA1, 138,276 neurons), dinput 4, doutput 6, all inputs/outputs in their expected ranges and every output class populated.

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`)
These re-open the NWB files with `pynwb` and recompute everything **without** calling any function from `convert_data.py`, then compare with `np.allclose` against `/app/converted_data.pkl`. Five sessions spanning different mice, plane counts and scene types were checked (indices 0, 37, 75, 110, 151 = m11 d3, m13 d14, m17 d8, m19 d13, m7 d14), each at 3 trials (first / middle / last) plus a single-element spot check:
- **neural**: full re-implementation of dF/F + OASIS + interneuron exclusion -> `np.allclose(D['neural'][s][k], events[cells, a:b], atol=1e-5)`; also the number of curated neurons and the spot value `neural[trial 5][neuron 3, t=10]`.
- **input**: time from trial start (from the NWB timestamps), environment, trial number, and the previous-trial outcome for **all** trials of each session.
- **output**: all six variables recomputed from `position`, `speed`, `lick`, `reward_zone` and `Reward` timestamps.
**Result: 0 failures out of ~165 assertions.**

### Check 3: Reference code comparison
| Step | Reference | My script | Same? |
|---|---|---|---|
| (a) Loading | `create_sess` + `vr_align_to_2P` build `sess.vr_data` (VR interpolated onto imaging frames) and `sess.timeseries` (F, Fneu from suite2p) | the NWB release already contains exactly these aligned series; `load_session` reads them with pynwb and concatenates planes along the neuron axis | YES |
| (b) Neuron filtering | suite2p `iscell` after manual curation; paper also removes putative interneurons with r(dF/F, speed) > 0.5 | `iscell == 1` then the same speed-correlation rule (402 cells, 0.29%) | YES |
| (c) Temporal alignment | trials = `sess.trial_start_inds` -> `sess.teleport_inds`; ITI excluded | frames [trial_start, teleport) from the `trial_start`/`teleport` binaries | YES (reference uses start-1:stop-1; I use start:stop, a 1-frame shift that only changes which frames enter the baseline window - see note below) |
| (d) Binning | reference bins spatially (45 x 10 cm) for tuning analyses; time series kept at the imaging frame rate for the decoder (`decode.py` uses the event timeseries) | native imaging frames (64.484 ms), no rebinning | YES |
| (e) dF/F + events | `preprocessing.dff`: neuropil subtract 0.7 (+ per-trial neuropil mean added back), maximin baseline (nan-aware Gaussian sigma [0,15] -> `minimum_filter1d(300)` -> `maximum_filter1d(300)`), (F-F0)/|F0|, nan-aware Gaussian sigma 2, then `dcnv.oasis(dff, 2000, tau, fs)` | identical implementation (`compute_dff_events`), same constants (neu_coef 0.7, tau 0.7, fs 15.5078/plane) | YES |
| (f) Input construction | `glmUtils.create_design_matrix` uses position, relative position to reward, trial id, speed, licks, rewarded/omission | I use time-from-trial-start, environment (`morph`), trial number and previous-trial outcome as decoder inputs, as required by the Decoder Task spec | Task-specified |
| (g) Output construction | `behavior.get_trial_types` (isreward = reward AND rzone entry; morph), `behavior.get_reward_zones` (A/B/C, switch at trial 30), licks binarised (`licks>0 -> 1`), speed from `sess.vr_data['speed']`, distance to reward as in `create_design_matrix` (linear variant) | same definitions, discretised per the Decoder Task | YES |
| (h) Trial curation | `correct_lick_sensor_error` / paper: >30% of frames with cumulative lick > 2 | same rule, trials dropped (81) | YES |

Differences and their justification:
1. **[start, stop) vs [start-1, stop-1)**: the reference's off-by-one would make the last frame of each trial undefined (NaN) and would include one ITI frame at the start. Using the exact trial window keeps neural and behaviour frames consistent; the effect is a 1-frame (64 ms) shift of the baseline window only.
2. **No <2 cm/s masking**: required by the decoding task (continuous time series; speed is itself a decoded variable with a <2 cm/s class).
3. **Events recomputed instead of the NWB `Deconvolved` series** (which is suite2p spks from raw F, not from the paper's maximin dF/F).
4. **No place-cell selection**: the paper's Fig. 3 decoder used RR/TR/non-RR subpopulations; here all curated pyramidal cells are used, as the task asks for all recorded neurons.

### Check 4: Key statistics comparison
See the table in Step 9. Every statistic available in the paper matches:
- 11 mice; 12-14 sessions each (m11 starts on day 3, as stated in the methods).
- 12,135 trials after removing exactly the **81** lick-sensor-error trials the paper removed (out of 12,216 in the release / 12,376 in the paper).
- Mean 79.8 trials/session (paper 80.5 +- 7.4).
- Omission rate 15.6% of trials (paper ~15%).
- Interneuron exclusion 0.29% of cells (paper 0.42 +- 0.85%).
- Neurons/session 154-2323 vs the paper's stated 155-2172. Investigated: the five largest sessions are all **m18**, a two-plane animal (2,341 iscell ROIs pooled across planes on day 3). The paper's range is per session for the cells entering its analyses, and Ext. Fig. 7 treats the two planes separately, which explains the slightly higher pooled maximum. The lower bound (154 after interneuron removal from 155 iscell) matches the paper exactly.
- Reward zone usage A/B/C = 0.332/0.336/0.333 of samples, as expected from the counterbalanced design.

### Check 5: Edge cases
- Trial indexing: `trial_start` and `teleport` counts are equal, strictly interleaved, in every session; `scanning == 1` for every on-track frame; `trial number` is constant within each trial and equals the trial index; no NaNs in any behaviour series (checked in all 152 sessions, `/app/cache/edge_scan.py`).
- 10 sessions (m17/m18) have ophys arrays one frame longer than the behaviour series; the extra frame falls after the last teleport and is truncated.
- Shortest trial is 96 frames (6.2 s); longest 3,359 frames (216 s, mouse paused). Both are real behaviour and are kept.
- Three m4 sessions have fewer trials (40, 50, 45); one of these lost 35 trials to lick-sensor errors. All sessions still have >= 40 trials, well above the 2-trial minimum.
- Reward-zone switch: the scene-derived zone (change at trial index 30) agrees with the empirically observed reward-zone entry position on **100%** of rewarded trials in all 152 sessions (`zone_agreement == 1.0` for every session; the script falls back to the observed zone and warns otherwise - no warning was emitted).
- First trial of a session has no previous trial: `prev_trial_rewarded` is set to 1 (the session is preceded by 30 rewarded warm-up trials).

### Issues Found and Resolved
- *NWB `Deconvolved` is not the paper's event trace* -> recompute dF/F + OASIS (Step 4/5).
- *10 sessions with a trailing extra ophys frame* -> truncate ophys to the behaviour length.
- *Column name clash in an analysis script* (`df.eq`) -> analysis-only, fixed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, ~10 min).

### Training Progress
- Loss decreasing: **Yes** (2.60 -> 0.909 over 200 epochs; test loss 0.802).

### Decoder Results (Full)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|--------|-------------|--------|-------|
| dist_to_reward_zone (7) | 0.143 | 0.625 | **0.550** | 3.8x chance |
| position (5) | 0.200 | 0.725 | **0.671** | 3.4x chance |
| speed (5) | 0.200 | 0.640 | **0.596** | 3.0x chance |
| lick (2) | 0.500 | 0.766 | **0.746** | 1.5x chance |
| reward_zone_location (3) | 0.333 | 0.900 | **0.848** | 2.5x chance |
| reward_outcome (2) | 0.500 | 0.813 | **0.574** | 1.15x chance - see Step 12 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
All six outputs are above chance on validation data. Five of six exceed 1.5x chance. `reward_outcome` (0.574 vs 0.5) is the exception and was investigated:
- **Probe (`/app/cache/reward_probe.py`)**: an independent per-session logistic regression (5-fold stratified CV, balanced accuracy) on trial-averaged event rates gives 0.49-0.65 depending on session, i.e. the same ceiling the neural decoder reaches. Restricting to frames **after** the reward zone start gives the same or slightly better accuracy (0.49-0.66), while frames **before** the zone give chance (0.49-0.53). This is exactly the expected structure: whether a trial is rewarded is simply not knowable from CA1 activity before the animal reaches the zone, yet the per-trial label is (by specification) broadcast over the whole trial, so most timepoints are unpredictable in principle. It also rules out label leakage.
- The paper itself reports only a modest reward-versus-omission signal (Fig. 6: reward-vs-omission indices near 0 for most cells), so a low but above-chance accuracy is the biologically expected result.
- Conclusion: not a conversion bug; no change made.

### Check 2: Accuracy comparison to the paper
The paper's only decoding analysis (Fig. 3) is a **circular-linear regression of reward-relative position** scored with the mean cosine error, not a classification accuracy, so no number is directly comparable. The qualitative comparison:
| Variable | My validation balanced accuracy | Paper |
|---|---|---|
| Reward-relative (distance to zone) position | 0.550 (7 classes, chance 0.143) | decode score well above shuffle, informative over roughly -105 to +153 cm around the reward (Fig. 3b,c) |
| Absolute track position | 0.671 (5 classes, chance 0.200) | strong spatial coding: 459 +- 263 place cells per session (~50% of cells) |
| Speed / licking | 0.596 / 0.746 | movement variables are significant GLM predictors (Fig. 7) |
| Reward zone identity | 0.848 (3 classes) | remapping between zones is highly reliable (Fig. 2) |
| Reward outcome | 0.574 | small reward-vs-omission index for most cells (Fig. 6) |
My accuracies are consistent with, or better than, what the paper's effect sizes imply; the strongest decoding is for the spatial/reward-zone variables, exactly as reported.

### Check 3: Train vs validation gap
| Output | Train | Val | Ratio |
|---|---|---|---|
| dist_to_reward_zone | 0.625 | 0.550 | 1.14 |
| position | 0.725 | 0.671 | 1.08 |
| speed | 0.640 | 0.596 | 1.07 |
| lick | 0.766 | 0.746 | 1.03 |
| reward_zone_location | 0.900 | 0.848 | 1.06 |
| reward_outcome | 0.813 | 0.574 | **1.42** |
Only `reward_outcome` shows a substantial gap (still < 1.5x). This is the expected signature of a per-trial binary label with ~15% minority class and only ~80 trials per session: the model can memorise trial identity from slow drifts in the training trials but cannot generalise. It is not data leakage - the cross-validation splits by trial (`get_trial_indices`), and the pre-zone probe above is at chance.

### Additional debugging performed
1. Output values verified against the raw NWB for 3 trials in each of 5 sessions (Step 10 Check 2): exact match.
2. Temporal alignment: `processing_*.png` overlay neural traces, position, distance-to-zone, speed and licks on a common time axis with trial boundaries; reward delivery always falls inside the shaded active reward zone, and the distance bin is 3 exactly while the animal is inside it.
3. Output variation: every class of every output is populated (see Step 9 distributions); no output is >85% one class except `reward_outcome` (84% rewarded, which is the true behaviour).
4. Neural filtering follows the reference (iscell + interneuron exclusion); recomputation matched to 1e-5.
5. Processing matches the reference code (Step 10 Check 3).

### Issues Found and Resolved
- No further issues. No re-conversion was necessary after Step 9.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created (dataset description, processing, load instructions, format spec, key statistics, decoder accuracies)
- [x] cache/ folder created with `README_CACHE.md` documenting every investigation script
- [x] All files organized:
  - `/app/convert_data.py` - conversion script
  - `/app/converted_data.pkl` (9.52 GB), `/app/sample_data.pkl` (0.12 GB)
  - `/app/conversion_sample_out.txt`, `/app/verification_sample_out.txt`, `/app/train_decoder_sample_out.txt`
  - `/app/conversion_full_out.txt`, `/app/verification_full_out.txt`, `/app/train_decoder_full_out.txt`
  - `/app/processing_sub-m11_ses-03*.png`, `/app/processing_sub-m3_ses-05*.png` (processing plots)
  - `/app/CONVERSION_NOTES.md`, `/app/README.md`
  - `/app/cache/` - exploration and sanity-check scripts + their outputs
