# Dataset Conversion Notes

## Overview
- **Dataset**: Data for "A flexible hippocampal population code for experience relative to reward" from `/app/data`
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `multi_anim_sess` | `src/reward_relative/utilities.py` | LOADING / PROCESSING | Loads one `sess` pickle per animal for a given experiment day, computes dF/F and deconvolved `events`, derives reward/trial metadata, and packages per-animal dictionaries used throughout the paper. |
| `dff` | `src/reward_relative/preprocessing.py` | PROCESSING | Computes single-channel deltaF/F from `F`/`Fneu`, optionally deconvolves to `events`, and masks data outside valid trial periods (with configurable teleport handling). |
| `get_trial_types` | `src/reward_relative/behavior.py` | PROCESSING | Produces per-trial reward outcome (`isreward`) and environment identity (`morph`) from framewise synchronized VR data between `trial_start_inds` and `teleport_inds`. |
| `get_reward_zones` | `src/reward_relative/behavior.py` | PROCESSING | Maps each session scene to reward-zone coordinates and A/B/C labels, including switch sessions with a `change_reward_trial`. |
| `define_trial_subsets` | `src/reward_relative/behavior.py` | CURATION | Splits trials into pre/post-switch sets or first/second half sets for non-switch sessions. |
| `get_timeseries_data` | `src/reward_relative/glmUtils.py` | LOADING / PROCESSING | Extracts continuous framewise neural and behavioral streams aligned to trial starts, builds reward-relative position, trial IDs, reward/omission state, speed, and licks, and masks invalid samples. |
| `get_omission_inds` / `get_omission_trials` | `src/reward_relative/rewardAnalysis.py` | PROCESSING | Defines omission trials and identifies reward-zone entry indices on omitted-reward trials. |
| `load_multi_anim_sess` | `src/reward_relative/dayData.py` | LOADING | Loads the cleaned shared pickles used for most manuscript analyses (`toShare/cleaned_w_F/...`). |
| `define_anim_list` | `src/reward_relative/dayData.py` | CURATION | Defines which animals are included on each experiment day, including the combined 2023+2024 cohort used in the manuscript. |
| `dayData` | `src/reward_relative/dayData.py` | PROCESSING | Higher-level per-day analysis container used for figure analyses and decoder notebooks; stores trial subsets, reward zones, activity matrices, cell classes, and behavioral summaries. |

### Notes
- The repo is for two-photon calcium imaging, not electrophysiology. Neural activity begins as raw fluorescence in `sess.timeseries['F']` and neuropil `Fneu`; deltaF/F is **not** precomputed in raw `sess` objects and is added later by `utilities.multi_anim_sess` via `preprocessing.dff`.
- The manuscript pipeline works with three levels of saved data: `sess` (VR-aligned imaging session), `multi_anim_sess` (per experiment day across animals, with dF/F, events, place-cell masks, reward metadata), and `dayData` (downstream analyses).
- The reference decoder notebook (`notebooks/Fig3_Decoder.md`) uses **continuous framewise deconvolved `events`**, not spatial trial matrices, and calls `glmUtils.get_timeseries_data(..., data_types=['trials','rel_pos','pos','speed','rewards'])`.
- `glmUtils.get_timeseries_data` constructs continuous variables by iterating over trial start/end indices and filling arrays only within valid trial epochs (`start-1:stop-1`), leaving out-of-trial periods as `NaN`. This is important for temporal alignment.
- Framewise behavior is already synchronized to imaging in `sess.vr_data`; `docs/multi_anim_sess_README.md` states one sample is one imaging frame at about 15.5 Hz (~64.5 ms/frame).
- `sess.timeseries['licks']` is cumulative per frame and later binarized (`licks[licks > 1] = 1`) for continuous analyses. The code also masks trials with suspected lick sensor failure if a large fraction of frames exceed a cumulative lick count threshold.
- Reward outcome in the reference code is determined trialwise from `vr_data['reward']`, usually requiring reward-zone occupancy as well (`get_trial_types`).
- Environment identity is encoded as a binary `morph` variable in the synchronized VR data (`Env1 -> 0`, `Env2 -> 1` in `behavior.env_morph_dict`; `multi_anim_sess_README.md` also documents trialwise `morph` as 0/1).
- Reward-zone labels A/B/C are not read directly from framewise variables; they are inferred from the `scene` name using `behavior.get_reward_zones`. For switch sessions, the code assumes the switch occurs at `change_reward_trial` when present, otherwise default `change_trial=30`.
- The paper’s continuous decoder notebook downsampled observations to balance occupancy of reward-relative position bins before comparing pre/post-switch maps, but that is specific to the figure analysis rather than the requested export format.
- Place-cell analyses often apply a speed threshold of 2 cm/s, but the stored `sess` trial matrices include all speeds by default. For the requested neural decoder export, the continuous reference path is more relevant than place-cell-specific trial matrices.
- Cell curation in the original preprocessing occurs upstream in suite2p (`iscell.npy` curated manually in GUI). The code here assumes saved `sess` objects already contain curated ROIs; there is no additional electrophysiology-style unit quality filtering stage in this repo.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains a DANDI-style NWB release rather than the paper repo’s original pickle hierarchy.
- Top level contents:
  - `dandiset.yaml`
  - one directory per subject: `sub-m3`, `sub-m4`, `sub-m7`, `sub-m11`, `sub-m12`, `sub-m13`, `sub-m14`, `sub-m15`, `sub-m17`, `sub-m18`, `sub-m19`
  - each subject directory contains one NWB file per session: `sub-<mouse>_ses-XX_behavior+ophys.nwb`
- There are no dataset-local README/MD/TXT files under `/app/data`; metadata are in `dandiset.yaml` and in the NWB file contents.
- Each NWB file contains:
  - `processing/ophys/Deconvolved/plane0/data`: deconvolved neural activity, shape `(n_frames, n_rois)`
  - `processing/ophys/Fluorescence/plane0/data`: raw fluorescence, shape `(n_frames, n_rois)`
  - `processing/ophys/Neuropil/plane0/data`: neuropil fluorescence, shape `(n_frames, n_rois)`
  - `processing/ophys/ImageSegmentation/PlaneSegmentation`: ROI table with columns `pixel_mask`, `iscell`, `planeIdx`
  - `processing/behavior/BehavioralTimeSeries`: synchronized framewise streams `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, `trial_start`, `autoreward`, plus sparse `Reward` timestamps
- Behavioral time series use explicit timestamps at imaging-frame resolution; deconvolved fluorescence uses a uniform sampling rate (`rate` in NWB) and has the same number of frames as the framewise behavior arrays.
- Native behavior conventions observed directly in the data match the repo documentation:
  - `environment` values are `-1` before synchronization/scanning, then `0` or `1`
  - `position` is `-500` before synchronization and otherwise spans the virtual track up to about `450 cm`
  - `trial number` is `-1` before synchronization and otherwise nonnegative integers
  - `trial_start` and `teleport` are binary framewise event indicators
  - `lick` and `reward_zone` can exceed 1 within a frame, consistent with cumulative-per-frame counts
- Imaging rates are mixed across subjects:
  - 124 sessions at `15.5078125 Hz`
  - 28 sessions at `31.015625 Hz`
  - the 31 Hz sessions are exactly subjects `m17` and `m18`
- `planeIdx` values are `0` or `1`, indicating mixed single-plane and two-plane recordings.
- `iscell` is a two-column suite2p-style array; column 0 is a binary cell flag and column 1 is a score/probability. Example rows from `sub-m3_ses-03`: `[0, 0.498]`, `[1, 0.678]`, `[1, 0.886]`, etc.
- One structural anomaly was found during exploration: `sub-m11_ses-03_behavior+ophys.nwb` has 81 unique nonnegative `trial number` values but only 80 `trial_start` and 80 `teleport` pulses during scanning. This will need explicit handling/checking in later steps.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 `iscell`-positive ROIs across all sessions (312,110 total segmented ROIs) |
| Neurons / session | mean 912.36 `iscell`-positive ROIs/session (mean 2053.36 total ROIs/session; range 155-2341 kept) |
| Subjects | 11 |
| Sessions / subject | 14 for 10 subjects; 12 for `m11` (152 sessions total) |
| Trials (total) | 12,216 from `trial_start` pulses during scanning |
| Trials / session | mean 80.37 (range 41-100) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not stated directly in paper for switch cohort total | N/A |
| Neurons / session | 155–2172 putative pyramidal neurons/session | “This approach yielded 155–2172 putative pyramidal neurons per session” |
| Subjects | 11 switch-task mice in manuscript dataset | “Each mouse encountered a different... sequence... (n = 11 mice)” |
| Sessions / subject | 14 planned days; m11 imaging starts on day 3 | “The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3” |
| Trials (total) | 12,376 imaged trials across 11 switch mice for licking QC reference | “81 out of 12,376 trials removed across 11 switch mice” |
| Trials / session | Target 80–100; mean 80.5 ± 7.4 | “We targeted 80–100 trials per session... (mean ± s.d., 80.5 ± 7.4 trials...)” |
| Neural data time bin | ~0.0645 s/frame (~15.5 Hz) | “0.0645 s imaging frame samples”; “sampled at ~15.5 Hz” |
| Behavior data time bin | VR sampled at 50–75 Hz, synchronized to imaging frames; downstream aligned at imaging rate | “Behavioral data were sampled at approximately 50–75 Hz... synchronized with the ~15.5 Hz sampled imaging data” |
| Reward rate | ~85% rewarded, ~15% omission | “Reward was randomly omitted on approximately 15% of trials” |
| Switch point | 30 trials into each switch session | “On day 3... the zone was moved after 30 trials... Each switch occurred after 30 trials.” |
| Reward zones | A: 80–130 cm; B: 200–250 cm; C: 320–370 cm | “zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm” |
| Imaging rates | ~15.5 Hz normally; m17/m18 scanned at ~31 Hz interleaved for ~15.5 Hz/plane | “~15.5 Hz...”; “m17 and m18... ~31 Hz interleaved... ~15.5 Hz per plane” |
| Lick QC removal | ~0.65% of trials removed | “~0.65% of all imaged trials, n = 81 out of 12,376 trials removed” |
| Additional interneuron exclusion | 0.42 ± 0.85% of cells | “excluding 0.42 ± 0.85% of cells (mean ± s.d. across mice and days)” |


### Processing Details
- The experiment is CA1 two-photon calcium imaging during virtual-reality navigation on a 450 cm linear track with hidden reward zones that move across days and within switch sessions.
- Reference task structure:
  - reward zones are 50 cm wide
  - switch days are 3, 5, 7, 8, 10, 12, 14
  - on switch days the reward zone changes after 30 trials
  - first 10 trials of each new condition can be auto-rewarded if the mouse does not lick in the new zone
  - teleport/jitter follows each lap, with longer jitter after omission or no-lick trials
- Alignment:
  - behavioral data were collected at VR frame rate and synchronized to imaging with Unity-generated TTL pulses
  - manuscript analyses then operate on imaging-frame-aligned streams (~15.5 Hz)
- Imaging preprocessing:
  - Suite2p motion correction and ROI extraction
  - manual ROI curation before downstream analysis
  - dF/F baseline computed within each trial independently with a 20 s sliding-window maximin baseline
  - dF/F smoothed with a 2-sample Gaussian kernel (~0.129 s s.d.)
  - deconvolution performed with OASIS to obtain event-like activity
- Spatial analyses:
  - track binned into 45 bins of 10 cm
  - activity below 2 cm/s excluded for spatial place-cell analyses
  - place-cell SI computed from deconvolved activity
- Licking preprocessing:
  - bad lick-sensor trials removed when >30% of imaging frames in a trial have cumulative lick count >2
  - remaining lick counts converted to binary before spatial binning
- Important decoder-specific reference from paper:
  - manuscript RR-position decoder uses deconvolved calcium event time series
  - it excludes samples with running speed <2 cm/s
  - it predicts circular reward-relative position, not absolute track position or the multi-output task requested here

### Curation Steps

**Neuron curation rules**:
- Manual ROI curation removed ROIs with multiple somata/dendrites, lacking obvious transients, suspected overexpression, or continuous high fluorescence suggestive of interneurons.
- Additional putative interneurons were excluded if Pearson correlation between dF/F and running speed exceeded 0.5.
- Multi-plane animals pooled planes for most analyses.

**Trial curation rules**:
- Reward-switch analyses use pre-switch versus post-switch trial sets split at trial 30.
- On non-switch days, analyses using two trial sets use first versus second half of trials.
- Lick analyses set a very small number of lick-fault trials to `NaN`.
- Omission analyses require at least three omission trials in a trial set.

### Decoders Trained
| Decoded variable | Accuracy |
| Reward-relative position from RR/TR/non-RR subpopulations | Before-switch decoding above shuffle for RR, TR, and non-RR; after-switch generalization above shuffle only for RR population |
| Reward-relative position from RR cells across track | Significant above-shuffle decoding across most of environment: z-score > 2 from about -104.5 ± 20.1 cm to +152.7 ± 22.9 cm relative to reward start |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Container format | Repo analyses expect `sess` / `multi_anim_sess` / `dayData` pickle structures | Shared dataset is NWB (`behavior+ophys.nwb`) per subject-session | Data are from same paper but format not specified in Methods | Treat NWB as the authoritative serialized form of the same aligned content. Map NWB `processing/behavior/BehavioralTimeSeries` to `sess.vr_data`-like variables and NWB `processing/ophys/...` to `sess.timeseries`. |
| Subject list | `dayData.define_anim_list(..., year='combined')` includes additional animals such as GCAMP2, GCAMP6, GCAMP10 and fixed-control cohort in some analyses | NWB release has 11 mice: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19 | Main switch-task cohort is `n = 11 mice`; fixed cohort is separate | Use the 11 switch-task mice present in NWB. The shared NWB release appears to exclude the fixed-condition cohort and non-switch animals used elsewhere in the codebase. |
| Session count | One session/day; m11 begins on day 3 | 152 NWB files: 10 mice × 14 sessions + m11 × 12 sessions | “all mice except m11... day 3” | Fully consistent. Use all 152 NWB files as candidate sessions. |
| Neural signal representation | Code computes `dff` from fluorescence, then deconvolves to `events`; many analyses use `events` | NWB already stores `Fluorescence`, `Neuropil`, and `Deconvolved` | Methods specify trialwise maximin dF/F and OASIS deconvolution | Prefer NWB `Deconvolved` as the neural signal for export, because it matches the paper’s event-like activity and avoids reimplementing dF/F from scratch unless needed for a validation check. |
| Behavioral alignment | Code operates on imaging-frame-aligned `vr_data` with trial epochs defined by `trial_start_inds` and `teleport_inds` | NWB behavior streams have the same frame count as neural traces and share imaging-frame timestamps | Methods describe VR synchronized to imaging by TTL | Treat one imaging frame as the canonical temporal grid for all streams. Reconstruct trials from `trial_start` and `teleport` pulses. |
| Imaging rate | Most analyses assume ~15.5 Hz; two-plane mice interleaved | NWB rates are 15.5078125 Hz in 124 sessions and 31.015625 Hz in 28 sessions (all m17/m18) | m17/m18 were scanned at ~31 Hz interleaved for ~15.5 Hz/plane | For a unified export, resample all sessions to a common 15.5078125 Hz bin size. For m17/m18, combine both planes and downsample time by 2 to match the per-plane effective rate described in the paper. |
| ROI/cell curation | Manual ROI curation plus optional speed-correlation interneuron exclusion in `dayData` | NWB provides `iscell` and `planeIdx`; `iscell[:,0]` count ranges 155–2341 | Manual curation yielded 155–2172 pyramidal neurons/session; additional speed-corr exclusion removed 0.42 ± 0.85% of cells | Use `iscell[:,0] == 1` as the base ROI inclusion, then apply an additional speed-correlation interneuron exclusion step if needed to better match manuscript counts and logic. The sessions above 2172 are concentrated in multi-plane mouse m18 and likely shrink after this secondary exclusion. |
| Trial count denominator | Paper cites 12,376 total imaged trials in licking-QC context | Complete trials reconstructed from `trial_start`/`teleport` pulses total 12,216; one file has a trailing partial trial number | Paper: 12,376 trials across 11 switch mice | Use complete trial epochs defined by paired `trial_start`/`teleport` events as authoritative for conversion. Note the paper’s 12,376 denominator likely reflects a slightly different accounting basis; re-check later against derived trial counts and session-level summaries. |
| Partial/incomplete trials | Code implicitly uses explicit start/end indices | One NWB file (`sub-m11_ses-03`) contains `trial number == 80` for only 3 trailing frames without `trial_start` or `teleport` | Paper does not discuss this edge case | Exclude trailing partial trials lacking both explicit start and end events. |
| Reward-zone semantics | Code infers A/B/C labels from scene names and hard-coded coordinates (X/Y/Z = 80–130/200–250/320–370 cm) | NWB has `environment` plus framewise `reward_zone`; session identifier embeds original scene name | Methods specify A/B/C as 80–130 / 200–250 / 320–370 cm | Recover reward-zone label/location per session/day using subject-day metadata logic from the reference code, cross-checked against NWB behavior streams. |
| Lick handling | Code binarizes licks and detects faulty lick-sensor trials if >30–35% of samples exceed a cumulative count threshold | NWB `lick` values are cumulative-per-frame and can exceed 1 | Methods: >30% of 0.0645 s frames with cumulative lick count >2 indicates sensor fault | Match manuscript logic by binarizing licks after detecting and masking faulty trials. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane0/data` | `neural` | Filter ROIs with `iscell[:,0] == 1`, optionally exclude putative interneurons, segment frames from `trial_start` to frame before `teleport`, transpose to `(n_neurons, T)` | `preprocessing.dff`, `utilities.multi_anim_sess`, `glmUtils.get_timeseries_data` | Use deconvolved activity because the paper’s decoders and spatial analyses operate on deconvolved event-like activity. |
| `processing/behavior/BehavioralTimeSeries/position/timestamps` + `trial_start` | `input[0]` = time from start of trial | For each trial, subtract trial-start timestamp from per-frame timestamps | `glmUtils.get_timeseries_data` | Imaging-frame-aligned continuous seconds, shape `(1, T)` row within full input matrix. |
| `processing/behavior/BehavioralTimeSeries/environment/data` | `input[1]` = environment type | Take constant trial value from frames within trial (mode/unique nonnegative value), repeat across `T` | `behavior.get_trial_types` | Encode ENV1=0, ENV2=1. |
| `processing/behavior/BehavioralTimeSeries/trial number/data` | `input[2]` = trial number | Take session-local complete-trial index / trial number, repeat across `T` | `glmUtils.get_timeseries_data` | Use integer-like float values, starting at 0 within session. |
| Reward timestamps / previous trial reward outcome | `input[3]` = previous trial outcome | Compute reward outcome per complete trial, shift by one trial within session, fill first trial with 0, repeat across `T` | `behavior.get_trial_types`, `rewardAnalysis.get_reward_inds` | Binary: omitted/no reward = 0, rewarded = 1. |
| Trial position + reward-zone interval | `output[0]` = distance to reward zone | Signed distance to nearest point in active reward-zone interval: `<start -> pos-start`, `inside -> 0`, `>end -> pos-end`; discretize to 7 bins | `behavior.get_reward_zones` | Time-varying categorical output. |
| Trial position | `output[1]` = absolute corridor position | Discretize cm position into 5 bins spanning 0–450 cm | `glmUtils.get_timeseries_data` | Time-varying categorical output. |
| `processing/behavior/BehavioralTimeSeries/speed/data` | `output[2]` = speed | Use framewise speed in cm/s, discretize into 5 bins | `glmUtils.get_timeseries_data` | Do **not** apply the paper’s >2 cm/s place-cell filter here because speed itself is a target to decode. |
| `processing/behavior/BehavioralTimeSeries/lick/data` | `output[3]` = lick | Detect faulty lick trials, then binarize framewise licks (`>0 -> 1`) | `behavior.correct_lick_sensor_error`, manuscript lick QC | Time-varying binary output. |
| Session/day metadata + reward switch rule | `output[4]` = reward-zone location | Use reference scene/day logic to assign A/B/C per complete trial; repeat across `T` | `sessions_dict.py`, `behavior.get_reward_zones` | Per-trial categorical output encoded as A=0, B=1, C=2. |
| Reward timestamps within trial | `output[5]` = reward outcome | If a reward event timestamp falls between trial start and teleport, output 1 else 0; repeat across `T` | `behavior.get_trial_types`, `rewardAnalysis.get_reward_inds` | Per-trial categorical output. |
| `subject.subject_id` | `subjects`, `subject_idx` | Use NWB subject IDs (`m3`, `m4`, ...) and map sessions in file order to subject indices | N/A | Session order will follow sorted file paths for reproducibility. |
| CA1 recording location | `brain_regions`, `brain_region_idx` | Single brain region `CA1`; assign all kept neurons to index 0 | Paper Methods, NWB imaging plane location | The paper pooled deep/superficial planes for most analyses, so region remains CA1. |

### Key Decisions
1. **Trial boundaries are `trial_start` through the frame immediately before `teleport`**: this matches the reference code’s use of explicit start/end indices and excludes negative-position tunnel frames plus the corrupted teleport frame.
2. **Use imaging-frame-aligned behavioral timestamps as the master clock**: all sessions share the same effective frame interval (~64.4836 ms) in behavior timestamps, including m17/m18, whereas the 31.015625 Hz NWB rate metadata for two-plane files is inconsistent with the timestamped behavioral grid.
3. **Use deconvolved activity rather than recomputing dF/F for the main export**: this most directly matches the manuscript’s RR decoder and place-cell significance computations.
4. **Keep all complete imaging sessions/days in the export**: although several manuscript figures focus on switch days, the user requested the full dataset and the decoder task is well-defined on all sessions.
5. **Filter to `iscell[:,0] == 1` ROIs and consider an additional interneuron exclusion step**: this follows the reference pipeline as closely as possible; whether the extra speed-correlation exclusion is needed will be checked against paper statistics after implementation.
6. **Represent all decoder inputs and outputs as time-varying matrices `(d, T)`**: per-trial variables will be repeated across trial timepoints so the exported structure is uniform and directly usable by `train_decoder.py`.
7. **Compute reward-zone identity from reference session metadata rather than from framewise `reward_zone` counts alone**: the paper/code define zone A/B/C from scene identity and switch timing; the framewise `reward_zone` signal indicates occupancy, not location label.
8. **Use nearest-distance-to-interval for reward-zone distance**: this satisfies the user’s “distance to any location in the reward zone” requirement while remaining faithful to track geometry.
9. **Exclude incomplete trailing trials**: sessions with dangling `trial number` labels but no paired `trial_start`/`teleport` events should not contribute malformed trials.
10. **Encode subjects using NWB IDs (`m3`, `m4`, ...)**: these are the canonical identifiers present in the shared data files and are unambiguous.

### Planned Sanity Checks
- [ ] Neural sanity check: for selected sessions/trials/cells, verify converted `neural[session][trial][cell, :]` matches the raw NWB deconvolved trace segment with `np.allclose()`.
- [ ] Input sanity check: verify converted trial time vector equals raw behavior timestamps minus the raw trial-start timestamp with `np.allclose()`.
- [ ] Output sanity check: verify converted lick binary, speed bins, and position bins on selected trials agree with raw NWB behavior samples and hand-computed discretization.
- [ ] Reward-zone sanity check: verify per-trial reward-zone labels/intervals from session metadata match the expected A/B/C schedule from the reference `sessions_dict.py` entry for at least three subjects, including a switch session and a non-switch session.
- [ ] Trial-boundary sanity check: confirm first and last converted samples per trial correspond to `trial_start` and the frame before `teleport`, not the negative-position tunnel frames.
- [ ] Dataset-statistics sanity check: compare sessions/subject, trials/session, omission rate, and neuron/session ranges against the manuscript values after conversion.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Wrote `/app/convert_data.py` with the required CLI:
  - `python -u /app/convert_data.py <outpicklefile>`
  - `--full`
  - `--sample`
  - `--show-processing`
- Implementation loads NWB directly with `h5py` and imports `sessions_dict.py` from the reference repo to recover experiment-day scene metadata.
- Trialization follows the reference definition: pair `trial_start` and `teleport` pulses, align each trial to `trial_start`, and exclude the teleport frame itself.
- Reward-zone A/B/C identity is reconstructed from the reference scene logic, while trialwise ENV1/ENV2 identity comes from the framewise NWB `environment` stream.
- Neural signal currently uses NWB deconvolved activity directly, filtered to `iscell[:,0] == 1`.
- To keep the full exported pickle tractable, neural trial matrices are stored as `float16`; inputs are `float32` and outputs are integer categorical codes.
- Trials with lick-sensor faults are removed entirely rather than storing invalid lick targets with NaNs.
- `--sample` intentionally picks two representative sessions, including a two-plane session when available, to test the mixed-rate metadata case.

Code inefficiencies identified:
- The full dataset is large: rough estimate for complete-trial deconvolved activity at `iscell[:,0] == 1` is ~2.42 billion values (~9.67 GB float32, ~4.84 GB float16).
- Pickle serialization of many per-trial arrays may be slow on the full dataset.

Code speedups added:
- Process sessions one at a time rather than loading multiple NWB files concurrently.
- Read a session’s deconvolved matrix once, slice trials from memory, then release it before moving to the next file.
- Store neural data as `float16` to reduce disk and memory pressure.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 746 |
| Neurons / session | 155, 591 |
| Subjects | 2 (`m11`, `m17`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 159 |
| Trials / session | 80, 79 |
| `time_from_trial_start_s` range | [0.0, 30.3] |
| `environment_type` range | [0.0, 1.0] |
| `trial_number` range | [0.0, 79.0] |
| `previous_trial_outcome` range | [0.0, 1.0] |
| `distance_to_reward_zone` distribution | [0.217, 0.096, 0.040, 0.225, 0.021, 0.087, 0.315] |
| `absolute_position` distribution | [0.213, 0.196, 0.268, 0.176, 0.149] |
| `speed` distribution | [0.057, 0.060, 0.087, 0.375, 0.421] |
| `lick` distribution | [0.812, 0.188] |
| `reward_zone_location` distribution | [0.272, 0.728, 0.000] |
| `reward_outcome` distribution | [0.130, 0.870] |

### Processing Plots Review
- `processing_m11_ses-03.png` shows trial alignment starting near 0 cm and ending before the teleport frame; no bad lick trials in this sample session.
- `processing_m17_ses-01.png` shows the same boundary handling for a two-plane session and documents one removed bad-lick trial.
- Visual inspection confirms that negative-position tunnel samples are excluded from trial-aligned matrices and that reward-distance bins transition as expected from pre-zone to in-zone to post-zone.
- No temporal mismatch was apparent between neural heatmaps and behavioral trajectories in the representative trials.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Session-by-session loading + `float16` neural storage | Keeps memory bounded and roughly halves neural pickle size versus `float32` |
| Direct NWB slicing into per-trial arrays | Avoids redundant re-reading / intermediate serialization |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion (2 sessions) | ~0.9 s/session on representative sample | Full conversion core processing ~4–7 min by scaling sample workload; allowing for 5 GB pickle serialization, estimate ~6–10 min total |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `distance_to_reward_zone` | 0.5161 | 0.3678 |
| `absolute_position` | 0.5769 | 0.4907 |
| `speed` | 0.5256 | 0.4307 |
| `lick` | 0.6979 | 0.6547 |
| `reward_zone_location` | 0.9732 | 0.9963 |
| `reward_outcome` | 0.6810 | 0.5844 |

Additional notes:
- Training loss decreased from `44.91` at epoch 1 to `0.81` at epoch 200.
- All validation balanced accuracies exceeded uniform chance:
  - 7-class reward-distance chance = 0.1429
  - 5-class absolute-position and speed chance = 0.2000
  - binary lick and reward-outcome chance = 0.5000
  - 3-class reward-zone-location chance = 0.3333

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 4,819,654,451 bytes (~4.82 GB)
- `verification_full_out.txt`: created
- `conversion_full_out.txt`: created

Additional notes:
- Full conversion completed in `106.78 s` across all 152 sessions.
- `train_decoder.py --verify-only` reported: `Data format is valid, no errors or warnings.`
- Lick-QC removal removed exactly `81` trials, leaving `12,135` converted trials from `12,216` complete `trial_start`/`teleport` trial epochs in the NWB release.
- The resulting lick-QC removal fraction is `81 / 12,216 = 0.663%`, closely matching the manuscript’s reported `~0.65%`.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not stated directly | Base inclusion is curated ROIs, pooled across planes | 138,678 `iscell` ROIs | 138,678 | Yes vs release |
| Mean neurons/session | Range 155–2172 putative pyramidal neurons/session | Curated ROIs, optional additional interneuron exclusion in downstream `dayData` | Mean 912.36; range 155–2341 | Mean 912.36; range 155–2341 | Mostly yes; upper tail in NWB exceeds paper text |
| Subjects | 11 | 11 switch-task mice in combined cohort logic | 11 | 11 | Yes |
| Sessions | 152 implied by 10 mice × 14 sessions plus m11 from day 3 | One session/day | 152 NWB files | 152 | Yes |
| Trials (total) | 12,376 imaged trials in lick-QC discussion | Trialwise analyses built from explicit start/end indices | 12,216 complete paired trial epochs; 12,217 unique trial numbers because `sub-m11_ses-03` has one dangling partial trial label | 12,135 kept after removing 81 bad-lick trials | Partial; release denominator is lower, but QC count matches |
| Trials/session (mean) | 80.5 ± 7.4 targeted/imaged | Sessionwise trial sets throughout | 80.37 complete paired trials/session | 79.84 kept/session | Yes |
| Bad lick trials removed | ~81 / 12,376 = 0.65% | `correct_lick_sensor_error` flags trials with high cumulative lick occupancy | 81 / 12,216 = 0.663% using the same threshold logic | 81 removed | Yes |
| `time_from_trial_start_s` range | Trial-aligned at imaging-frame resolution; no explicit max stated | Framewise aligned continuous time within trial | [0.0, 216.54] s | [0.0, 216.54] s | Yes |
| `environment_type` range | Binary Env1 / Env2 | Binary `morph` | [0, 1] | [0, 1] | Yes |
| `trial_number` range | 80–100 trial sessions, switch after trial 30 | Trial IDs used continuously | [0, 99] | [0, 99] | Yes |
| `previous_trial_outcome` range | Binary rewarded / omission history | Reward / omission trial labels | [0, 1] | [0, 1] | Yes |
| `reward_zone_location` distribution | A/B/C zones used across the experiment | Session metadata determine A/B/C | Balanced by design across sessions | [0.332, 0.336, 0.333] | Yes |
| `reward_outcome` distribution | ~15% omission / ~85% reward | Trial reward labels from aligned reward stream | [0.158, 0.842] from raw trialwise reward events | [0.158, 0.842] | Yes |
| `lick` distribution | Lick is sparse, binary after QC | Licks binarized after sensor-error handling | [0.777 no, 0.223 yes] | [0.777 no, 0.223 yes] | Yes |

Spot checks performed:
- Verified the full verifier summary for the exported pickle reports the expected 152-session / 11-subject structure with no shape or datatype warnings.
- Recomputed aggregate trial counts directly from raw NWB `trial_start`/`teleport` pulses; this matched the pre-QC denominator used by the converter (`12,216`).
- Recomputed the bad-lick removal count from `conversion_full_out.txt`; this matched the paper’s reported `81` removed trials exactly.
- Inspected representative sessions including a single-plane session (`m11_ses-03`), a two-plane session (`m17_ses-01`), and a high-removal session (`m4_ses-14`) to confirm the exported trial counts are explained by explicit trial pairing plus lick-QC filtering.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Re-ran `/app/train_decoder.py /app/converted_data.pkl --verify-only` after the metadata fix. Result: `Data format is valid, no errors or warnings.`
2. **Raw-data sanity checks with `np.allclose()`**:
   - Neural, single-plane session: `m11_ses-03`, kept trial 5. Compared the full converted trial matrix `(155, 266)` against the raw NWB deconvolved slice from the paired `trial_start` / `teleport` window. `np.allclose(...) == True` after casting the raw trace to the same `float16` precision used in the export.
   - Neural, two-plane session: `m17_ses-01`, kept trial 10. Independently reconstructed the pooled neural matrix from NWB `plane0` and `plane1` using ROI `planeIdx`, then compared against the converted matrix `(591, 195)`. `np.allclose(...) == True`.
   - Inputs: for the same `m11_ses-03` and `m17_ses-01` trials, independently rebuilt `time_from_trial_start_s`, `environment_type`, `trial_number`, and `previous_trial_outcome` from the raw behavior streams. `np.allclose(...) == True`.
   - Outputs: for the same trials, independently rebuilt reward-distance bins, absolute-position bins, speed bins, lick binary, reward-zone label, and reward outcome from the raw NWB behavior streams plus session scene metadata. `np.allclose(...) == True`.
   - Lick-QC count: `m4_ses-14` contained 80 complete paired trials and 35 trials exceeding the `>30% of frames with cumulative lick count >2` threshold, leaving 45 kept trials. This exactly matched the converted session summary.
3. **Reference code comparison**:
   - **(a) Data loading**: `convert_data.py:295-343` loads synchronized behavior streams and deconvolved activity directly from NWB, analogous to the `sess.vr_data` / `sess.timeseries['events']` content that `utilities.multi_anim_sess` and `glmUtils.get_timeseries_data` expect downstream.
   - **(b) Neuron / trial filtering**: `convert_data.py:314-317` applies the manual ROI curation flag `iscell[:,0] > 0.5`, matching the curated-ROI assumption in the paper code. `convert_data.py:349-350` and `366-370` remove bad lick-sensor trials using the manuscript threshold (>30% of imaging-frame samples with cumulative lick count >2), which recovered the paper’s reported `81` removed trials. This differs from the helper default `correction_thr=0.5` in `behavior.correct_lick_sensor_error`, but it matches the methods text and the continuous-timeseries decoder path more closely.
   - **(c) Temporal alignment**: `convert_data.py:96-111` pairs explicit `trial_start` and `teleport` events, and `convert_data.py:372-378` slices `start:end`, excluding the teleport frame. This mirrors the reference logic in `glmUtils.get_timeseries_data`, which fills arrays only within `trial_start_inds` to `teleport_inds` windows (`start-1:stop-1` in the original indexing convention).
   - **(d) Binning**: `convert_data.py:125-160` implements the user-requested categorical bins for reward distance, absolute position, and speed while preserving the paper’s 50-cm reward-zone geometry from `behavior.get_reward_zones`.
   - **(e) Input construction**: `convert_data.py:397-407` builds framewise time, trialwise environment identity, trial index, and previous trial reward outcome. These correspond to the reference `morph`, `trial_ids`, and reward/omission labels, but are repeated across time to fit the decoder export format.
   - **(f) Output construction**: `convert_data.py:408-419` builds the six categorical targets required by the task. Lick and reward outcome semantics match the reference code; reward-zone location comes from the same scene-based session metadata logic used by the paper code.
4. **Key statistics comparison**:
   - Subjects and sessions match exactly: 11 mice, 152 sessions.
   - Trial structure matches the release and the paper’s targeting: 12,216 complete raw paired trials, 12,135 kept after 81 lick-QC removals, mean raw trials/session `80.37`.
   - Reward rate matches the paper expectation: converted reward outcome `[0.158 no, 0.842 yes]`, consistent with “approximately 15%” omission.
   - Reward-zone labels are balanced across the exported dataset `[0.332, 0.336, 0.333]`, as expected from the experimental design.
   - The paper text reports `155–2172` putative pyramidal neurons/session, whereas the NWB release contains three pooled multi-plane `m18` sessions above this upper bound (`2281`, `2314`, `2341`). I checked the worst case (`m18_ses-03`) with a direct deconvolved-trace versus speed correlation probe and found `0 / 2341` cells with `r > 0.5`, so the small post-hoc interneuron exclusion described in the paper cannot plausibly explain the NWB upper tail. I therefore retained all manually curated `iscell` ROIs and treat this as a release-versus-manuscript reporting discrepancy rather than a conversion bug.
5. **Edge-case checks**:
   - `sub-m11_ses-03` contains one dangling final `trial number` label without a paired `trial_start` / `teleport`; it remains excluded.
   - Trial-boundary check on the first `m11_ses-03` trial: raw position immediately before `trial_start` is `-1.12 cm`, the first included converted sample is `+0.61 cm`, the last included sample before teleport is `448.85 cm`, and the excluded teleport frame is `200.79 cm`. This confirms the expected off-by-one handling at both trial ends.
   - Two-plane sessions with one-frame stream mismatches are safely truncated to the minimum common frame count before trialization; this had already been fixed during script development and remained stable in the Step 10 rerun.

### Issues Found and Resolved
- **Incorrect metadata description of the neural source path**: the exported metadata still said `processing/ophys/Deconvolved/plane0`, which was inaccurate for pooled `m17` / `m18` sessions. Fixed `convert_data.py` to describe pooling across `plane*` datasets by ROI `planeIdx`, then re-ran the full conversion and full verification.
- **Initial sanity-check false alarm on two-plane neural traces**: the first independent reconstruction used plane-local indices counted only within `iscell` ROIs, which did not match the NWB `plane*` column order. After reconstructing plane-local indices across *all* ROIs exactly as stored in NWB, the neural sanity checks passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes
- Device used: CUDA
- Training loss trajectory: `183.7591` (epoch 1) -> `108.9180` (epoch 10) -> `22.5639` (epoch 60) -> `3.7103` (epoch 110) -> `1.2645` (epoch 200)
- Final test loss: `1.156982`
- Plot outputs created by the decoder script: `sample_trials.png`, `predictions.png`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `distance_to_reward_zone` | 0.3836 | 0.3540 | 2.48x uniform chance |
| `absolute_position` | 0.4864 | 0.4660 | 2.33x uniform chance |
| `speed` | 0.3950 | 0.3812 | 1.91x uniform chance |
| `lick` | 0.6510 | 0.6414 | Stable train/validation performance |
| `reward_zone_location` | 0.8438 | 0.8112 | Strongest decoded variable |
| `reward_outcome` | 0.5790 | 0.5237 | Slightly above chance; omissions are randomized |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `distance_to_reward_zone` | 0.3540 balanced accuracy | Paper’s RR-position decoder was above shuffle pre-switch for RR/TR/non-RR and generalized above shuffle post-switch only for RR cells; not directly numerically comparable, but the clear above-chance spatial decoding is qualitatively consistent. |
| `absolute_position` | 0.4660 balanced accuracy | No direct absolute-position decoder metric reported, but the paper shows robust spatial coding across the track; accuracy is consistent with that. |
| `speed` | 0.3812 balanced accuracy | No direct speed-decoder accuracy reported. Above-chance decoding is expected because running speed covaries with trial phase and reward approach. |
| `lick` | 0.6414 balanced accuracy | No direct lick-decoder accuracy reported. Above-chance decoding is plausible and stable across sample/full runs. |
| `reward_zone_location` | 0.8112 balanced accuracy | No direct manuscript metric, but strong reward-location coding is a central finding; high accuracy is consistent with that. |
| `reward_outcome` | 0.5237 balanced accuracy | No direct manuscript metric. Because reward omission was randomized (~15% omission), near-chance current-trial reward prediction is expected for a trial-start-aligned decoder. |

[Analysis of any low accuracies]
- **Accuracy vs chance**: every output exceeded uniform chance. Multiples of chance were:
  - `distance_to_reward_zone`: `2.48x`
  - `absolute_position`: `2.33x`
  - `speed`: `1.91x`
  - `lick`: `1.28x`
  - `reward_zone_location`: `2.43x`
  - `reward_outcome`: `1.05x`
- **Binary-output interpretation**: the user-suggested `1.5x chance` heuristic is informative for the 5-class and 7-class outputs, but is too strict for randomized binary outcomes. In particular, current-trial reward outcome is designed to be difficult because omission is stochastic and the label is repeated across all timepoints, including pre-reward periods.
- **Reward-outcome debugging checks**:
  - Re-checked raw reward labels against the converted output on three specific kept trials (`m11_ses-03` trials 0 and 5; `m17_ses-01` trial 10). All three matched exactly (`np.allclose == True`) and included both rewarded and omission examples.
  - Verified class balance remains sensible (`15.8%` omission, `84.2%` reward), so the low accuracy is not due to a degenerate constant target.
  - Train/validation gap is small (`0.5790 / 0.5237 = 1.106x`), arguing against leakage or severe overfitting.
- **Train vs validation gap across all outputs**: all train/validation balanced-accuracy ratios were small (`1.015x` to `1.106x`), so there is no sign of overfitting-driven pathology.
- **Paper comparison limitations**: the manuscript’s decoder predicts reward-relative position and evaluates performance primarily against shuffled controls and z-scored decode extent across the environment, whereas this task requires multi-output categorical decoding aligned to trial start. The exported data therefore cannot be expected to numerically match the paper’s decoder metric output-for-output; the appropriate comparison is qualitative consistency plus above-chance spatial decoding.

### Issues Found and Resolved
- **No new conversion bugs were found during the accuracy review**: the lower `reward_outcome` accuracy is best explained by the randomized omission schedule and the stricter trial-start-aligned prediction task, not by label corruption or temporal misalignment.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Additional notes:
- Added `/app/README.md` with dataset summary, format description, loading example, and reproduction commands.
- Added `/app/cache/check_sanity_against_raw.py` to reproduce the key raw-vs-converted sanity checks from Step 10.
- Added `/app/cache/summarize_converted_data.py` to print a compact statistical summary of a converted export.
- Added `/app/cache/README_CACHE.md` documenting the cached helper scripts.
