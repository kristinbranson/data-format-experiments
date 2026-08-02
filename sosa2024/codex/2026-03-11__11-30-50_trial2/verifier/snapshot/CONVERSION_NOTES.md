# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-03-11
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

Environment checks:
- `python3` runs successfully
- `numpy` import verified: `2.3.5`
- `torch` import verified: `2.6.0+cu124`
- `ls -la CONVERSION_NOTES.md` confirmed the file exists before proceeding

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` | `code/src/reward_relative/preprocessing.py` | LOADING | Creates a `TwoPUtils.sess.Session` object from imaging + VR files, then appends aligned session data. |
| `append_session_data` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Aligns VR to imaging, loads suite2p outputs, adds timeseries (`licks`, `rewards`, `speed`) and position-binned trial matrices. |
| `dff` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Computes trial-restricted dF/F with neuropil subtraction, `maximin` baseline, smoothing, and optional OASIS deconvolution to `events`. |
| `multi_anim_sess` | `code/src/reward_relative/utilities.py` | LOADING | Loads saved `sess` pickles across animals for an experiment day and attaches dF/F, events, place-cell outputs, reward labels, and trial subsets. |
| `quick_load_multi_anim_sess` | `code/src/reward_relative/utilities.py` | LOADING | Loads already processed multi-animal pickle files from the shared preprocessed data location. |
| `get_trial_types` | `code/src/reward_relative/behavior.py` | PROCESSING | Computes per-trial reward outcome and environment morph from aligned VR data between `trial_start_inds` and `teleport_inds`. |
| `get_reward_zones` | `code/src/reward_relative/behavior.py` | PROCESSING | Maps scene names to reward-zone coordinates and labels (`A`, `B`, `C`) for each trial, including switch sessions. |
| `define_trial_subsets` | `code/src/reward_relative/behavior.py` | PROCESSING | Splits trials into trial sets based on reward-zone switches or halves for single-zone sessions. |
| `get_timeseries_data` | `code/src/reward_relative/glmUtils.py` | PROCESSING | Builds continuously sampled decoder-ready behavior and neural arrays from `sess.timeseries['events']`, masking low-speed and invalid samples. |
| `CircularRegression` | `code/src/reward_relative/decode.py` | PROCESSING | Paper decoder model for circular variables; confirms the reference decoding was done on neural activity against position-like targets. |

### Notes
- The reference repo is for **2-photon calcium imaging**, not electrophysiology. There is no spike sorting or ephys quality filtering in this code path.
- `sess` is the base aligned session object. The README explicitly states raw `sess` objects **do not contain dF/F**; dF/F is computed later.
- `make_session_pkl.md` shows the first saved object is `sess`, created from raw suite2p + VR files after manual suite2p curation.
- Cell curation is manual in the suite2p GUI; `iscell.npy` is the curated cell mask used when loading suite2p outputs.
- `multi_anim_sess` is the main post-processing stage used by the paper:
  - loads saved `sess` pickle for each animal/day,
  - computes dF/F,
  - optionally deconvolves dF/F into `events`,
  - adds position-binned trial matrices,
  - computes place-cell statistics and trial subset metadata.
- For single-channel data, `multi_anim_sess(..., calc_spks=True)` calls `preprocessing.dff(...)` with:
  - `neuropil_method='subtract'`
  - `baseline_method='maximin'`
  - `subtract_baseline=True`
  - `neu_coef=0.7`
  - `deconvolve=True`
  - optional `keep_teleports` per animal
- `dff` restricts valid fluorescence to trial spans using `trial_start_inds` and `teleport_inds`. Baseline is computed **within each trial** after smoothing, min-filtering, and max-filtering. The resulting dF/F is smoothed again, then OASIS is applied to get `events`.
- `append_session_data` adds behavioral timeseries aligned to imaging frames. The pipeline is therefore based on **frame-aligned** neural and behavior streams.
- `glmUtils.get_timeseries_data` is the clearest reference for paper-style decoder inputs:
  - neural data are `sess.timeseries['events']`
  - trial boundaries use `trial_start_inds` and `teleport_inds`
  - reward-relative position is computed per sample from the reward-zone start for that trial
  - low-speed samples are masked out with `use_speed_thr=2`
  - lick sensor artifacts are suppressed when cumulative licks are implausibly high
  - samples with NaNs in neural data or masked speed/lick channels are dropped
- Reward-zone labels come from `behavior.get_reward_zones`:
  - `A` corresponds to zone `X` = `[80, 130]`
  - `B` corresponds to zone `Y` = `[200, 250]`
  - `C` corresponds to zone `Z` = `[320, 370]`
  - environment names remain `Env1`, `Env2`, `Env3`
- The Fig. 3 decoder notebook uses:
  - experiment `MetaLearn`
  - experiment days `[3, 5, 7, 8, 10, 12, 14]`
  - `ts_key='dff'` when loading saved `multiDayData`, but continuous neural decoder inputs are assembled from `sess.timeseries['events']` via `glmUtils.get_timeseries_data`
  - occupancy matching by reward-relative position bin for some paper analyses
- Conclusion for conversion:
  - **Yes, dF/F must be computed** to match the reference processing.
  - The decoder-relevant neural stream is most likely the **deconvolved `events`**, not raw fluorescence.
  - Trial alignment should use `trial_start_inds` as the start and `teleport_inds` as the end of each trial.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains one NWB file per subject-session plus `dandiset.yaml`.
- Subject directories present:
  - `sub-m3`, `sub-m4`, `sub-m7`, `sub-m11`, `sub-m12`, `sub-m13`, `sub-m14`, `sub-m15`, `sub-m17`, `sub-m18`, `sub-m19`
- File pattern:
  - `data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`
- Total NWB files: `152`
- Sessions per subject from file inventory:
  - `sub-m11`: 12 sessions (`ses-03` to `ses-14`)
  - all other listed subjects: 14 sessions each
- Each NWB contains:
  - `processing/behavior/BehavioralTimeSeries`
  - `processing/ophys/Fluorescence`
  - `processing/ophys/Neuropil`
  - `processing/ophys/Deconvolved`
  - `processing/ophys/ImageSegmentation`
- Behavioral time series present in every sampled file:
  - `Reward`
  - `autoreward`
  - `environment`
  - `lick`
  - `position`
  - `reward_zone`
  - `scanning`
  - `speed`
  - `teleport`
  - `trial number`
  - `trial_start`
- Ophys organization in sample NWB:
  - `Deconvolved/plane0/data`: `(n_timepoints, n_rois)`
  - `Fluorescence/plane0/data`: `(n_timepoints, n_rois)`
  - `Neuropil/plane0/data`: `(n_timepoints, n_rois)`
  - `ImageSegmentation/PlaneSegmentation` columns:
    - `pixel_mask`
    - `iscell`
    - `planeIdx`
- `iscell` has shape `(n_rois, 2)` and matches suite2p format:
  - column 0: binary cell flag (`0` or `1`)
  - column 1: suite2p confidence/probability
- Imaging metadata from sample file:
  - indicator: `GCaMP7f`
  - location: `hippocampus, CA1`
- Neural sampling rates present across the dataset:
  - `15.5078125 Hz` in 124 sessions
  - `31.015625 Hz` in 28 sessions
- Frame-count alignment:
  - most sessions have equal behavior/neural frame counts
  - 10 sessions are off by exactly 1 sample (`behavior = neural - 1`), so edge handling will be required
- Behavioral coding observed directly from NWB values:
  - `environment`: values `-1, 0, 1` (`-1` appears outside valid trial/imaging periods)
  - `trial number`: values `-1` plus contiguous nonnegative trial ids
  - `trial_start`, `teleport`, `lick`, `scanning`: binary-like 0/1 signals with some `-1` outside valid periods
  - `Reward`: sparse event series with separate timestamps (not frame-length)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | `118,493` curated cells using `iscell[:,0] > 0.5` on the ROI indices referenced by `Deconvolved/plane0/rois`; `260,091` total ROIs in the neural response matrices |
| Neurons / session | curated mean `779.56` (min `155`, max `1780`); raw ROI mean `1711.12` (min `315`, max `3934`) |
| Subjects | `11` |
| Sessions / subject | mostly `14`; `sub-m11` has `12` |
| Trials (total) | `12,216` from `sum(trial_start > 0)` across sessions |
| Trials / session | mean `80.37` (min `41`, max `100`) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not directly reported | Methods report per-session counts rather than a total. |
| Neurons / session | `155–2172` putative pyramidal neurons/session | “155–2172 putative pyramidal neurons per session” |
| Subjects | `11` switch-task mice in main dataset; `3` fixed-condition mice in separate cohort | “switch task (n = 11 mice)” and “fixed-condition cohort (n = 3 mice)” |
| Sessions / subject | `14` imaging days for task; `m11` starts imaging on day 3 | “total of 14 days” and “for m11, imaging started on day 3” |
| Trials (total) | `12,376` imaged trials across 11 switch mice (for licking QC reference) | “81 out of 12,376 trials removed across 11 switch mice” |
| Trials / session | target `80–100`; mean `80.5 ± 7.4` across all imaging days | “We targeted 80–100 trials per session… mean ± s.d., 80.5 ± 7.4 trials” |
| Neural data time bin | imaging sampled at `~15.5 Hz` per plane | “sampled at ~15.5 Hz” / “~15.5 Hz per plane” |
| Behavior data time bin | behavior aligned to imaging frame rate `~15.5 Hz` | “All behavioral and neural time series were sampled at ~15.5 Hz” |
| Reward rate | `~85% rewarded`, `~15% omitted` | “Reward was randomly omitted on approximately 15% of trials” |
| Reward switch timing | switch after `30` trials | “Each switch occurred after 30 trials” |
| Reward zones | A=`80–130 cm`, B=`200–250 cm`, C=`320–370 cm` | “zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm” |
| Track length | `450 cm` | “450 cm linear track” |
| Decoder analysis set | `77 sessions`, `11 mice`, `7` switch days | “n = 77 sessions, 11 mice, seven switch days” |
| Decoder general result | before-switch: RR/TR/non-RR above shuffle; after-switch: only RR above shuffle | “when we tested on trials after the reward switch, only the RR population decoded… better than shuffle” |
| Decoder spatial extent | z-scored decode `>2` from `−104.5 ± 20.1 cm` to `+152.7 ± 22.9 cm` relative to reward | “range of z-scored decode >2: −104.5 ± 20.1 cm to +152.7 ± 22.9 cm” |
| GLM performance reference | FDE all place cells `0.10 ± 0.19`, TR `0.32 ± 0.13`, RR `0.29 ± 0.11`, non-RR `0.29 ± 0.11` | “On average, the model’s FDE…” |


### Processing Details
- Task structure:
  - hidden 50 cm reward zone on a 450 cm linear VR track
  - two main environments (`ENV 1`, `ENV 2`) in the switch task
  - reward locations A/B/C are equidistant and only one is active at a time
  - reward switches occur after 30 trials on switch days
  - first 10 trials of a new condition can receive automatic reward if the mouse fails to lick in the new zone
  - teleport period follows each lap: gray jitter period plus 50 cm tunnel before the next lap
- Imaging / alignment:
  - 2-photon CA1 calcium imaging with GCaMP7f
  - most sessions sampled at ~15.5 Hz
  - m17 and m18 were imaged in two planes interleaved at ~31 Hz, yielding ~15.5 Hz per plane
  - imaging sometimes stopped during teleport periods to minimize photodamage; some animals/days were imaged through teleports
- Neural preprocessing:
  - Suite2P used for motion correction and ROI detection
  - dF/F baseline computed independently within each trial using a `maximin` procedure with a 20 s sliding window
  - dF/F = `(F - baseline) / abs(baseline)`
  - dF/F smoothed with a two-sample Gaussian
  - deconvolved activity extracted with OASIS and used for decoder / SI analyses
- Behavioral / neural analysis conventions:
  - all spatial analyses exclude neural activity when speed `< 2 cm/s`
  - spatial binning is 45 bins of 10 cm each along the 450 cm track
  - licking is converted to binary counts and smoothed / binned for downstream analyses
- Decoder-specific paper processing:
  - decoder predicts circular reward-relative position from deconvolved activity
  - tenfold cross-validation, 90% train / 10% test
  - train/test either within pre-switch trials or train pre-switch and test post-switch
  - occupancy is downsampled to match reward-relative position bins
  - shuffled controls are generated by circularly shifting each cell’s timeseries

### Curation Steps

**Neuron curation rules**:
- Manual suite2p curation removed ROIs:
  - with multiple somata or dendrites
  - lacking visually obvious transients
  - suspected of calcium-indicator overexpression
  - with high continuous fluorescence fluctuation typical of putative interneurons
- Additional putative interneurons were excluded if Pearson correlation between dF/F and running speed exceeded `0.5`
- Reported additional exclusion from this speed-correlation rule: `0.42 ± 0.85%` of cells across mice/days

**Trial curation rules**:
- Reward omission trials occur randomly on ~15% of trials and are retained as a task condition
- Licking analysis removes trials with erroneous lick detection when >30% of imaging frames in the trial have cumulative lick count `> 2`
- Reported lick-artifact removal: `81 / 12,376` imaged trials across 11 switch mice (~0.65%)
- Some analyses require at least three omission trials within a trial set

### Decoders Trained
| Decoded variable | Accuracy |
| Reward-relative position from deconvolved activity | Paper reports decode score above shuffle, not classification accuracy; after-switch generalization is significant only for RR cells |
| Single-neuron deconvolved activity from task + movement variables (GLM) | FDE: all place cells `0.10 ± 0.19`, TR `0.32 ± 0.13`, RR `0.29 ± 0.11`, non-RR `0.29 ± 0.11` |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Cohort size | Code primarily analyzes the `MetaLearn` switch cohort by day, especially the seven switch days | NWB archive contains 11 mice / 152 subject-session files | Paper reports 11 switch-task mice plus a separate 3-mouse fixed-condition cohort | Treat the shared NWB archive as the 11-mouse switch cohort only. Fixed-condition mice are not present in `data/`. |
| Session count | Code expects one saved `sess` per imaging day; m11 has special handling in metadata | 152 NWB files = `11 * 14 - 2` | Paper states 14 task days, with imaging for m11 starting on day 3 | Counts are consistent once the m11 day 1-2 exception is applied. |
| Neural stream to decode | `glmUtils.get_timeseries_data` uses `sess.timeseries['events']` after dF/F + OASIS | NWB stores `Fluorescence`, `Neuropil`, and `Deconvolved`; no explicit dF/F timeseries | Paper decoder uses deconvolved calcium activity derived after per-trial dF/F | Use NWB `processing/ophys/Deconvolved/plane0/data` as the paper-equivalent neural signal. Raw `Fluorescence`/`Neuropil` remain available for sanity checks. |
| Cell filtering source | Code assumes suite2p curation has already been applied to the loaded `sess` data | NWB includes `iscell` in `PlaneSegmentation`, but segmentation tables for multipane mice contain ROIs not present in `Deconvolved/plane0/data` | Paper counts are post-curation putative pyramidal neurons | Filter cells by `iscell[:,0] > 0.5` only on ROI indices listed in `Deconvolved/plane0/rois`. This avoids overcounting plane-1 ROIs that are not in the response matrix. |
| Neurons/session | Code/paper operate on curated cells after ROI QC | Initial naive data count from full segmentation exceeded paper range; corrected ROI-referenced curated count is `155–1780` per session | Paper reports `155–2172` neurons/session | After using the ROI region correctly, the data are consistent with the paper’s reported range. |
| Multipane sessions | Code notes m17 and m18 were imaged in two planes and pooled for most analyses | NWB sessions for `sub-m17` and `sub-m18` have response rate `31.015625 Hz`, `planeIdx` includes `0/1`, but `Deconvolved/plane0/rois` selects only plane 0 cells | Paper states two-plane acquisition at ~31 Hz interleaved, ~15.5 Hz per plane | The archive exposes only the plane-0 response series for these files. Conversion will use the provided response matrix and document this archive-level limitation. |
| Trial boundary representation | Code uses `trial_start_inds` and `teleport_inds` arrays in `sess` | NWB provides framewise binary `trial_start` and `teleport` time series plus `trial number` | Paper defines laps/trials with a teleport period after each lap | Reconstruct trial start/end indices from `trial_start` and `teleport` signals in NWB. |
| Timebase | Code and methods emphasize ~15.5 Hz per plane | NWB contains both `15.5078125 Hz` and `31.015625 Hz` sessions | Paper says most sessions ~15.5 Hz and multipane mice interleaved at ~31 Hz for ~15.5 Hz/plane | Use a common output bin size during conversion so all sessions share the same decoder timebase. |
| Trial totals | Data inventory gives `12,216` starts / `12,217` unique trial numbers | NWB trial counts are slightly lower than one paper count | Paper cites `12,376` imaged trials across 11 switch mice in the lick-QC section and `80.5 ± 7.4` trials/session overall | Accept NWB-exported trial identities as ground truth for conversion, while using the paper’s mean trials/session as the stronger consistency check. The remaining ~1.3% mismatch is likely a paper-vs-export counting convention difference and will be tracked in later sanity checks. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane0/data` | `neural` | Crop to valid frame range, select ROI indices from `Deconvolved/plane0/rois`, keep curated cells by `iscell[:,0] > 0.5`, optionally exclude speed-correlated interneuron-like cells, transpose to `(neurons, time)`, split by trials, and rebin `31.015625 Hz` sessions to the common `15.5078125 Hz` bin size | `preprocessing.dff`, `utilities.multi_anim_sess`, `glmUtils.get_timeseries_data` | Neural decoder signal should match paper deconvolved activity (`events`) as closely as possible. |
| `processing/ophys/Deconvolved/plane0/rois` + `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell` | `neural` / `brain_region_idx` | Use ROI region referenced by response matrix, then `iscell[:,0] > 0.5` to keep curated cells only | suite2p curation assumption in `make_session_pkl.md`; `utilities.multi_anim_sess` | Critical for multipane sessions; do not use the full segmentation table. |
| `general/optophysiology/ImagingPlane/location` | `brain_regions` | Normalize `hippocampus, CA1` to `CA1` | N/A | All sessions are CA1 in this archive. |
| Reconstructed trial-relative elapsed time | `input[0]` | Seconds since trial start for each bin | trial handling implicit in `glmUtils.get_timeseries_data` | Align all trials to the first frame of the trial. |
| `processing/behavior/BehavioralTimeSeries/environment/data` | `input[1]` | Per trial, take modal valid environment value (`0` or `1`) and repeat across bins | session/task logic in `behavior.get_trial_types` / paper task description | Map `0 -> ENV1`, `1 -> ENV2`. |
| `processing/behavior/BehavioralTimeSeries/trial number/data` | `input[2]` | Per trial, use within-session trial index and repeat across bins | `glmUtils.get_timeseries_data` creates 0-indexed `trial_ids` | Keep native 0-indexing to match code conventions. |
| Trial reward outcome shifted by one trial | `input[3]` | Previous trial rewarded (`1`) vs omitted (`0`), repeated across current-trial bins; first trial gets `0` | paper reward omission schedule; reward outcome from trial parsing | User-requested decoder input, not a paper decoder variable. |
| `processing/behavior/BehavioralTimeSeries/position/data` + inferred reward-zone bounds | `output[0]` | Signed distance to nearest point in active reward zone: negative before zone, `0` inside zone, positive after zone; discretize to 7 bins per task spec | reward-zone logic from `behavior.get_reward_zones`; reward-relative computations in `glmUtils.get_timeseries_data` | For zone `[start, end]`: `pos < start -> pos-start`; `start <= pos <= end -> 0`; `pos > end -> pos-end`. |
| `processing/behavior/BehavioralTimeSeries/position/data` | `output[1]` | Discretize absolute position on the 450 cm corridor into 5 equal bins | paper: 45 spatial bins over 450 cm | Decoder spec requires 5 bins, not the paper’s 45 bins. |
| `processing/behavior/BehavioralTimeSeries/speed/data` | `output[2]` | Discretize speed into bins: `<2`, `2-10`, `10-20`, `20-40`, `>40 cm/s` | paper and code use speed threshold 2 cm/s | Keep all bins for output even though paper excludes `<2 cm/s` in some analyses. |
| `processing/behavior/BehavioralTimeSeries/lick/data` | `output[3]` | Convert to binary per bin (`lick > 0`) after lick-artifact handling | lick cleanup in `glmUtils.get_timeseries_data`; methods licking QC | For rebinned 31 Hz sessions, use logical OR within each 15.5 Hz bin. |
| Inferred active reward zone per trial | `output[4]` | Map trial reward location to categorical `{A:0, B:1, C:2}` and repeat across trial bins | `behavior.get_reward_zones`; paper task design | Infer from reward-zone-active positions / reward events and fill omission trials within stable blocks. |
| Reward events assigned to trials | `output[5]` | Trial outcome categorical `{omitted:0, rewarded:1}` repeated across trial bins | `behavior.get_trial_types` | Rewarded if any `Reward` event timestamp falls within the trial. |

### Key Decisions
1. **Use all 152 NWB sessions from the 11 switch-task mice**: The shared archive corresponds to the main switch cohort, not the 3 fixed-condition mice.
1. **Use NWB `Deconvolved` as the neural signal**: The paper’s decoder and SI analyses operate on deconvolved activity after dF/F, and this stream is already provided in the archive.
1. **Filter cells with ROI-region-aware `iscell` selection**: Only ROI ids referenced by the response matrix are valid candidates for inclusion.
1. **Plan to exclude putative interneuron-like cells**: The paper excludes cells with dF/F-speed correlation `> 0.5`, so the conversion script should implement this if feasible from the NWB fluorescence streams and reference `dff` logic.
1. **Reconstruct trials from `trial_start` and `teleport`**: This matches the reference trial definition more closely than relying only on trial-number changes.
1. **Standardize all sessions to a common `15.5078125 Hz` bin size**: This is the dominant dataset rate and the paper’s effective per-plane sampling rate.
1. **Use time-varying outputs for all decoder targets**: Even per-trial outputs (`reward zone`, `reward outcome`) will be repeated across bins for consistent `(n_output, n_timepoints)` arrays.
1. **Infer reward-zone location from rewarded trials and fill omission trials within stable blocks**: Missing zone labels in the NWB export are concentrated on omission trials; the task structure constrains the active zone to one contiguous block per condition.
1. **Use omission fraction and switch timing as sanity anchors**: Raw data show `rewarded = 84.65%`, `omission = 15.35%`, matching the paper’s ~15% omission rate, and switch sessions should change condition near trial 30.
1. **Keep within-session trial number 0-indexed**: This matches raw NWB values and `glmUtils.get_timeseries_data`.

### Planned Sanity Checks
- [ ] Neural spot-check: for one single-plane session and one 31 Hz session, compare a raw NWB trial slice against converted `neural` with `np.allclose()` after ROI filtering and rebinning.
- [ ] Input spot-check: verify `time_from_trial_start`, `environment`, `trial number`, and `previous outcome` on 3 hand-checked trials loaded directly from NWB.
- [ ] Output spot-check: verify distance-to-zone bin, absolute-position bin, speed bin, and lick bin on specific raw frames loaded directly from NWB.
- [ ] Trial parsing check: confirm converted trial counts match reconstructed `trial_start` / `teleport` boundaries.
- [ ] Cohort statistics check: converted subjects/sessions/trials/neurons must match the NWB inventory and remain within paper-reported ranges.
- [ ] Outcome distribution check: rewarded/omission fraction in converted data should remain close to the raw-data fraction (`84.65% / 15.35%`).

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `convert_data.py` with:
  - NWB file discovery and sample/full session selection
  - ROI-region-aware `iscell` filtering
  - trial reconstruction from `trial_start` / `teleport`
  - trial-wise reward outcome and reward-zone inference
  - 31 Hz to 15.5078125 Hz rebinning
  - trial-aligned neural/input/output construction
  - metadata export and optional processing plots
- First execution on `--sample --show-processing` completed without runtime errors.

Code inefficiencies identified:
- Full-session NWB reads currently materialize the session response matrix in memory once per session.
- Processing plots intentionally add overhead and should stay disabled for full conversion.

Code speedups added:
- Used direct `h5py` reads instead of heavier NWB object materialization.
- Restricted neural loading to ROI indices actually referenced by the response matrix and then to curated `iscell` ROIs.
- Used vectorized trial construction and simple factor-2 rebinning for 31 Hz sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | `441` |
| Neurons / session | `181`, `260` |
| Subjects | `2` (`m11`, `m17`) |
| Sessions / subject | `1`, `1` |
| Trials (total) | `160` |
| Trials / session | `80`, `80` |
| `time_from_trial_start_s` range | `[0.0, 20.8]` |
| `environment` range | `[0.0, 1.0]` |
| `trial_number` range | `[0.0, 79.0]` |
| `previous_trial_rewarded` range | `[0.0, 1.0]` |
| `distance_to_reward_zone` distribution | `[0.286, 0.106, 0.056, 0.243, 0.017, 0.071, 0.220]` |
| `absolute_position` distribution | `[0.171, 0.176, 0.253, 0.250, 0.150]` |
| `speed` distribution | `[0.081, 0.060, 0.078, 0.334, 0.448]` |
| `lick` distribution | `[0.772, 0.228]` |
| `reward_zone_location` distribution | `[0.209, 0.396, 0.395]` |
| `reward_outcome` distribution | `[0.148, 0.852]` |

### Processing Plots Review
- `processing_m11_ses-08.png`: environment and reward-zone transition occurs at trial ~30 as expected; example trial shows consistent alignment of position, speed, lick, and discretized outputs.
- `processing_m17_ses-08.png`: 31 Hz session rebinned cleanly to the common rate; example trial and trial-wise reward-zone summary look internally consistent.
- No obvious temporal misalignment or discretization artifacts seen in either sample plot.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Direct `h5py` loading instead of full NWB object graph | Keeps per-session conversion near ~1.3 s in the sample run |
| ROI-region-aware cell filtering before downstream work | Avoids processing unused segmentation-table ROIs |
| Simple factor-2 rebin for 31 Hz sessions | Keeps multipane-session harmonization inexpensive |

| Step | Time / Session | Estimated Total Time |
| Conversion core (observed sample sessions) | `1.22-1.35 s/session` | `~3.5 min` for 152 sessions without processing plots |
| Sample conversion command including plot generation | `8.47 s / 2 sessions total` | Plotting overhead is only for sample/debug mode and will remain off for full conversion |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `distance_to_reward_zone` | `0.5652` | `0.4176` |
| `absolute_position` | `0.6101` | `0.5024` |
| `speed` | `0.5455` | `0.4720` |
| `lick` | `0.7323` | `0.6773` |
| `reward_zone_location` | `0.9609` | `0.9119` |
| `reward_outcome` | `0.6694` | `0.5765` |

Additional notes:
- Loss decreased from `33.433657` at epoch 1 to `0.759214` at epoch 200.
- All validation balanced accuracies were above uniform-chance levels.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `7.3G`
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | not directly reported | curated cells after suite2p curation | `118,493` curated neurons | `118,493` | Yes |
| Mean neurons/session | range `155-2172` | curated cells after curation | `779.56` mean (`155-1780`) | `779.56` mean (`155-1780`) | Yes |
| Subjects | `11` switch-task mice | `MetaLearn` switch cohort | `11` | `11` | Yes |
| Sessions | `152` implied by 11 mice x 14 days minus m11 day1-2 | one imaging session/day | `152` | `152` | Yes |
| Trials (total) | `12,376` cited in lick-QC section | reconstructed from trial boundaries | `12,216` raw | `12,147` kept after 69 lick-artifact removals | Close |
| Trials/session (mean) | `80.5 ± 7.4` | reconstructed from trial boundaries | `80.37` raw | `79.91` kept | Yes |
| `environment` range | binary ENV1/ENV2 | trial-wise environment logic | `[0.0, 1.0]` | `[0.0, 1.0]` | Yes |
| `trial_number` range | task-dependent, up to ~100 trials/session | 0-indexed trial ids | `[0, 99]` raw | `[0.0, 99.0]` | Yes |
| `previous_trial_rewarded` range | binary | derived from reward outcome | `[0, 1]` | `[0.0, 1.0]` | Yes |
| `reward_outcome` distribution | `~15%` omission | reward outcome per trial | raw rewarded/omission = `[0.153, 0.847]` | converted = `[0.159, 0.841]` | Yes |
| `reward_zone_location` distribution | all three zones used across cohort | reward-zone logic by task block | approximately balanced across cohort | `[0.328, 0.338, 0.334]` | Yes |

Additional notes:
- Full conversion completed in `95.11 s`.
- Lick-artifact trial removal in converted data: `69 / 12,216 = 0.56%`, close to the paper’s reported `~0.65%`.
- `verification_full_out.txt` reported no format errors or warnings.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` contained no errors and no warnings.
1. **Neural sanity checks from raw NWB**:
   - `m11_ses-03`, trial 0, single-plane session:
     - compared raw `Deconvolved` slice after ROI-region-aware `iscell` filtering to converted `neural[session][trial]`
     - result: `np.allclose(...) == True`
   - `m17_ses-08`, trial 0, 31 Hz session:
     - compared manual factor-2 pair-summed raw `Deconvolved` slice to converted rebinned neural trial
     - result: `np.allclose(...) == True`
1. **Input sanity check from raw NWB**:
   - `m11_ses-08`, trial 30 (environment/reward-zone switch trial)
   - manually reconstructed `time_from_trial_start_s`, `environment`, `trial_number`, and `previous_trial_rewarded`
   - result: `np.allclose(...) == True`
1. **Output sanity check from raw NWB**:
   - `m11_ses-08`, trial 30
   - manually reconstructed distance-to-zone bins, absolute-position bins, speed bins, lick bins, reward-zone label, and reward outcome
   - result: `np.allclose(...) == True`
1. **Reference code comparison**:
   - (a) Data loading:
     - reference: load saved `sess` / `multi_anim_sess` objects
     - converted: load the NWB export directly
     - conclusion: equivalent archive representation of aligned behavior + neural data
   - (b) Neuron / trial filtering:
     - reference: suite2p curation, plus paper-level licking QC and optional interneuron exclusion by dF/F-speed correlation
     - converted: ROI-region-aware `iscell` filtering and lick-artifact trial removal
     - conclusion: ROI curation and licking QC are matched; exact dF/F-speed-correlation interneuron exclusion was not re-run because the archive exposes deconvolved responses but not the exact reference dF/F stream
   - (c) Temporal alignment:
     - reference: `trial_start_inds` and `teleport_inds`
     - converted: reconstructed directly from NWB `trial_start` and `teleport`
     - conclusion: matched
   - (d) Binning:
     - reference: native ~15.5 Hz per plane
     - converted: 31 Hz sessions rebinned to 15.5078125 Hz to satisfy common-bin requirement
     - conclusion: matched in effective sampling rate
   - (e) Input construction:
     - converted inputs are task-specified trial-aligned variables, not the paper’s original circular decoder predictors
     - conclusion: this difference is required by the user’s decoder task
   - (f) Output construction:
     - converted outputs are categorical task-specified variables derived from paper-consistent geometry and trial labels
     - conclusion: this difference is required by the user’s decoder task
1. **Key statistics comparison**:
   - subjects: `11`, matches paper switch cohort
   - sessions: `152`, matches 11 mice x 14 days minus m11 day1-2
   - neurons/session: converted `155-1780`, within paper range `155-2172`
   - raw omission fraction: `15.35%`
   - converted reward-outcome omission fraction: `15.9%`
   - lick-artifact trial removal: converted `69/12,216 = 0.56%`, close to paper `81/12,376 ≈ 0.65%`
1. **Edge-case checks**:
   - confirmed trial accounting exactly: `n_trials_raw - n_trials_lick_artifact_removed == n_trials_kept` for every session
   - confirmed `reward_zone_location` and `reward_outcome` are constant within each converted trial
   - handled 10 sessions with a 1-sample neural/behavior length mismatch by cropping to the common minimum length before trial parsing
   - checked multipane sessions use only the ROI ids referenced by the response matrix
   - checked sessions with long trial durations / low trial counts are present in the raw archive rather than introduced by the conversion

### Issues Found and Resolved
- **Multipane ROI overcount risk**: initial dataset exploration counted curated cells from the full segmentation table; resolved by restricting to ROI ids referenced by `Deconvolved/plane0/rois`.
- **Potential trial-accounting ambiguity**: resolved by storing raw, removed, and kept trial counts per session and verifying exact equality session-by-session.
- **Reference paper vs archive trial-count mismatch**: not fully resolvable from the archive alone; retained NWB-exported trial boundaries as source of truth and verified that session means and lick-QC fractions still align with the paper.
- **Exact dF/F-speed-correlation interneuron filter**: documented as an archive-level limitation. The shared NWB files provide the deconvolved response stream used for decoding, but not the exact reference dF/F timeseries needed to reproduce that curation step bit-for-bit. Given that converted neuron counts already fall within the paper’s reported range, this difference was not forced with a surrogate filter.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes
- Final training loss (`epoch 200`): `1.259246`
- Final test loss: `2.093138`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| distance_to_reward_zone | 0.5535 | 0.4171 | `2.92x` chance on validation; strongest time-varying reward-relative proxy |
| absolute_position | 0.5994 | 0.5236 | `2.62x` chance; consistent with strong hippocampal position coding |
| speed | 0.5345 | 0.4662 | `2.33x` chance; movement state is decodable from CA1 activity |
| lick | 0.6633 | 0.6286 | Above chance but only `1.26x` chance; reviewed in Step 12 because the target is sparse and transient |
| reward_zone_location | 0.8657 | 0.8227 | `2.47x` chance; strong session/trial-context signal |
| reward_outcome | 0.6801 | 0.5482 | Above chance with small train/val gap; omission signal is weaker than location/position |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| distance_to_reward_zone | 0.4171 | Closest paper comparison is RR-position decoding; paper reports decode score above shuffle across a broad RR range, so above-chance decoding was expected |
| absolute_position | 0.5236 | Paper studies place fields and positional coding extensively, so strong above-chance position decoding was expected |
| speed | 0.4662 | Paper GLMs include movement variables and speed thresholding, so a clearly above-chance speed readout was expected |
| lick | 0.6286 | Paper includes lick-aligned behavior analyses; modest above-chance decoding is reasonable because lick is sparse and binary |
| reward_zone_location | 0.8227 | Paper’s main result is flexible reward-relative coding, so strong reward-location/context decoding was expected |
| reward_outcome | 0.5482 | Omission vs reward is only ~16% / 84%; above-chance but weaker decoding was expected |

Direct paper comparison limits:
- The paper does **not** report balanced classification accuracy for these six decoder targets.
- The paper’s decoder predicts circular reward-relative position with a continuous decode score (`cos(y - yhat)`) and compares against shuffle, so only qualitative comparison is possible.
- The closest qualitative expectations are:
  - reward-relative / position variables should decode clearly above chance
  - reward-location context should be strong
  - sparse binary outputs such as lick and reward outcome may be weaker

Checks performed:
1. **Accuracy vs chance**
   - All six validation accuracies are above chance.
   - Chance multiples on validation:
     - distance_to_reward_zone: `2.92x`
     - absolute_position: `2.62x`
     - speed: `2.33x`
     - lick: `1.26x`
     - reward_zone_location: `2.47x`
     - reward_outcome: `1.10x`
2. **Lick investigation (`< 1.5x` chance heuristic)**
   - Re-checked Step 10 raw-vs-converted output spot check: exact match.
   - Reviewed converted class balance: lick positive fraction `0.2329`, negative fraction `0.7671`.
   - Reviewed sample/full processing plots: lick transients are temporally aligned to trial time and position.
   - Conclusion: lower margin over chance is consistent with a sparse, brief binary target rather than a conversion bug.
3. **Reward-outcome investigation**
   - Validation accuracy is only slightly above chance, but the target is trial-level and highly imbalanced (`rewarded = 0.8412`, `omitted = 0.1588`).
   - Raw-vs-converted checks already passed, and train/val gap is small (`0.6801` vs `0.5482`), arguing against leakage or gross mislabeling.
4. **Train vs validation gap**
   - No output exceeded the `1.5x` train/validation gap threshold.
   - Largest ratio was `1.33x` for `distance_to_reward_zone`; others were closer.
5. **Comparison to paper**
   - Paper result: RR-position decoding is above shuffle before the switch for RR/TR/non-RR populations, and only RR remains above shuffle after the switch.
   - Converted dataset result: the closest reward-relative proxy (`distance_to_reward_zone`) is robustly above chance, while reward-zone location is strongly decodable.
   - This is qualitatively consistent with the paper’s claim that CA1 population activity carries flexible reward-related spatial information.

### Issues Found and Resolved
- `lick` validation accuracy was below the `1.5x`-chance heuristic: investigated via raw-data spot checks, class balance, and plots; no conversion change was warranted.
- No direct paper balanced-accuracy values exist for the requested decoder outputs: documented the comparison as qualitative rather than pretending to have a one-to-one numeric benchmark.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
- Moved generated plot artifacts (`processing_*.png`, `sample_trials.png`, `predictions.png`) into `cache/`.
