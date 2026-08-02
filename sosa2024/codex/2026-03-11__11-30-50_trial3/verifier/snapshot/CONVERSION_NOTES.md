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

Python environment:
- `python3` available
- `numpy 2.3.5`
- `torch 2.6.0+cu124`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `get_sess_pkl_path` | `code/src/reward_relative/utilities.py` | LOADING | Resolves the per-animal/per-day preprocessed `sess` pickle path from session metadata. |
| `load_sess_pickle` | `code/src/reward_relative/utilities.py` | LOADING | Loads an already aligned `sess` object from pickle. |
| `multi_anim_sess` | `code/src/reward_relative/utilities.py` | PROCESSING | Main post-processing entry point: loads `sess`, computes dF/F and optionally deconvolved `events`, then adds per-trial behavioral annotations. |
| `dff` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Computes dF/F from `F` and `Fneu` within valid trial epochs, optionally excluding teleport samples and optionally deconvolving to `events`. |
| `get_trial_types` | `code/src/reward_relative/behavior.py` | PROCESSING | Derives per-trial reward outcome (`isreward`) and environment identity (`morph`) from frame-aligned VR data between each trial start and teleport. |
| `get_reward_zones` | `code/src/reward_relative/behavior.py` | PROCESSING | Maps scene names to reward zone coordinates and labels (`A`, `B`, `C`), including switch days with a default change at trial 30 unless `sess.change_reward_trial` overrides it. |
| `define_trial_subsets` | `code/src/reward_relative/behavior.py` | CURATION | Defines pre/post-switch or split-half trial subsets used by the paper’s analyses. |
| `get_timeseries_data` | `code/src/reward_relative/glmUtils.py` | PROCESSING | Builds continuously sampled behavioral series aligned to neural `events`: position, reward-relative position, speed, licks, rewards, and trial ids, masking invalid samples. |
| `CircularRegression` | `code/src/reward_relative/decode.py` | PROCESSING | Circular decoder model used in Fig. 3 for reward-relative position decoding. |
| `train_vs_test_blocks` | `code/src/reward_relative/decode.py` | PROCESSING | Cross-validation helper that trains/tests decoder on repeated blocks of sample indices. |

### Notes
- Shared analysis starts from prebuilt `sess` pickles, not raw NWB or raw imaging files.
- `sess` already contains VR data aligned to imaging frames. The repo documentation says the actual frame alignment is done by `TwoPUtils.preprocessing.vr_align_to_2P` in an external repo; within this project the aligned result is consumed from `sess.vr_data`.
- The shared analysis pipeline is:
  1. load `sess`
  2. compute `dff` from `F` and `Fneu`
  3. optionally deconvolve to `events`
  4. add position-binned trial matrices
  5. derive per-trial variables (`isreward`, `morph`, reward zone, trial subsets)
- `multi_anim_sess` uses default dF/F settings:
  - neuropil subtraction with coefficient `0.7`
  - baseline method `maximin`
  - `keep_teleports=False`
  - deconvolution optional; the paper’s decoder notebook later uses `sess.timeseries['events']`
- `dff` keeps fluorescence only from `trial_start_inds` to `teleport_inds` with slices `start-1:stop-1`; teleport samples are excluded unless explicitly requested otherwise.
- `get_timeseries_data` is especially important for consistency:
  - uses `sess.timeseries['events']` as neural data
  - loops over trial epochs defined by `trial_start_inds` and `teleport_inds`
  - computes reward-relative position from the reward-zone start for each trial
  - stores trial id as a per-sample series
  - sets `was_reward` to 1 from reward delivery to trial end, else 0
  - clips licks to binary with `licks[licks > 1] = 1`
  - marks a trial’s lick samples as invalid if more than 35% of samples exceed cumulative lick count 2
  - masks out samples where speed is below `use_speed_thr=2` cm/s in the Fig. 3 decoder workflow
  - additionally masks any samples where neural `events` are NaN
- `multi_anim_sess_README.md` documents the saved fields of the shared processed data:
  - `timeseries` keys include `F`, `Fneu`, `licks`, `rewards`, `speed`, `events`, `dff`
  - one sample corresponds to one imaging frame at about 15.5 Hz (~64.5 ms)
  - `trial_start_inds` and `teleport_inds` delimit trials
  - `morph` is binary environment identity: `0=Env1`, `1=Env2`
  - `isreward` is per-trial reward outcome: `1=rewarded`, `0=omission`
  - `rz label` is per-trial reward zone label in `{A,B,C}`
- The decoder notebook `code/notebooks/Fig3_Decoder.md` decodes reward-relative position from continuously sampled `events`, with sample selection restricted to running periods above 2 cm/s and occupancy matching across trial sets for the published figure.
- For this conversion, the most relevant reference processing appears to be the continuously sampled, frame-aligned representation from `get_timeseries_data`, plus per-trial labels from `behavior.py`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains one NWB file per subject-session plus `dandiset.yaml`.
- File layout is `data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`.
- There are 152 NWB files total across 11 subjects.
- Each NWB file has:
  - `processing/behavior/BehavioralTimeSeries`
    - time series keys are identical across all files:
      - `Reward` (sparse reward deliveries with timestamps)
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
  - `processing/ophys`
    - `Deconvolved/plane0/data`: neural activity, shape `(n_timepoints, n_rois)`
    - `Fluorescence/plane0/data`: fluorescence, same shape
    - `Neuropil/plane0/data`: neuropil fluorescence, same shape
    - `ImageSegmentation/PlaneSegmentation/iscell`: suite2p-style `(n_rois, 2)` array
    - `ImageSegmentation/PlaneSegmentation/planeIdx`: imaging plane per ROI
- In the NWB files, neural data are stored as time-by-ROI, so conversion will need to transpose to neuron-by-time per trial.
- Accepted cells are not pre-filtered out of the matrices. The `iscell` dataset stores suite2p curation output; the first column is the binary accepted-cell flag and should be used for neuron filtering.
- `identifier` strings include the original scene name (for example `Env1_LocationC`), which is needed to infer reward-zone labels consistently with the reference code.
- All sessions are from brain region `hippocampus, CA1`.
- All files share the same behavior variable set and contain no NWB `trials` table; trial boundaries are represented by the continuous `trial_start` and `teleport` time series.
- Observed sentinel values match the code documentation:
  - before valid synchronization, `environment` and `trial number` can be `-1`
  - `position` can be `-500`
- Most sessions are single-plane, but 28 sessions have ROIs from two planes (`planeIdx` has values `0` and `1`).
- Behavior timestamps have spacing about `0.0645 s` (~15.5 Hz). Some multiplane files report a `Deconvolved` rate attribute of `31.015625`, but the behavior series length matches the deconvolved series length, so the aligned sample grid needs to be taken from timestamps/array length rather than trusting the ophys `rate` attribute blindly.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138678 accepted cells (`iscell[:,0] == 1`) |
| Neurons / session | mean 912.36, range 155-2341 |
| Subjects | 11 |
| Sessions / subject | mean 13.82, range 12-14 |
| Trials (total) | 12216 |
| Trials / session | mean 80.37, range 41-100 |

Additional size notes:
- Sessions total: 152
- Subject IDs: `m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19`
- Session counts by subject:
  - `m11`: 12
  - all others: 14
- ROI totals before `iscell` filtering: 312110
- Frames per session: mean 23755.77, range 14164-51520

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | not stated directly for the released switch-task dataset | not directly reported for all released sessions | 
| Neurons / session | 155-2172 putative pyramidal neurons per session | “155–2172 putative pyramidal neurons per session” |
| Subjects | 11 switch-task mice in the main dataset | “counterbalanced across mice (n = 11 mice)” |
| Sessions / subject | 14 planned task days; `m11` began imaging on day 3 | “for a total of 14 days”; “except m11, for whom imaging started on day 3” |
| Trials (total) | 12,376 imaged trials across 11 switch mice before lick-artifact removals mentioned in Methods | “81 out of 12,376 trials removed across 11 switch mice” |
| Trials / session | targeted 80-100; mean 80.5 ± 7.4 across all imaging days in the full cohort | “We targeted 80–100 trials per session”; “80.5 ± 7.4 trials” |
| Neural data time bin | ~15.5 Hz (~0.0645 s per sample) | “sampled at ~15.5 Hz”; “0.0645 s imaging frame samples” |
| Behavior data time bin | same ~15.5 Hz frame-aligned sampling | “All behavioral and neural time series were sampled at ~15.5 Hz” |
| Reward rate | ~85% rewarded, ~15% omitted | “Reward was randomly omitted on approximately 15% of trials” | 
| Reward switch point | after 30 trials on each switch day | “Each switch occurred after 30 trials” | 
| Reward zones | A: 80-130 cm, B: 200-250 cm, C: 320-370 cm | “zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm” |
| Teleport zone | ~50 cm plus jitter | “teleport zone (~50 cm + temporal jitter up to 10 s)” |
| Lick artifact trial fraction | ~0.65% of imaged trials removed from licking analyses | “~0.65% of all imaged trials” |
| Additional speed-correlated cell exclusion | 0.42 ± 0.85% of cells | “excluding 0.42 ± 0.85% of cells” |


### Processing Details
- Task structure:
  - 450 cm linear VR track in two environments (`ENV 1`, `ENV 2`)
  - hidden 50 cm reward zone at one of three positions A/B/C
  - day 3 switch after 30 trials, day 5 switch again, day 7 return, day 8 switches into the other environment and reverses order over a total of 14 days
  - automatic reward on the first 10 trials of a new condition if the mouse failed to lick in the new zone
- Temporal sampling and alignment:
  - imaging frame rate ~15.5 Hz
  - behavioral and neural time series analyzed on the same frame-aligned sampling grid
  - teleport periods exist and are variably imaged depending on session/mouse
- Neural processing:
  - Suite2p motion correction and ROI extraction
  - manual curation of ROIs
  - dF/F baseline computed independently within each trial using a maximin procedure with a 20 s sliding window
  - dF/F then smoothed with a two-sample Gaussian kernel (~0.129 s s.d.)
  - deconvolution performed with OASIS / Suite2p-style canonical calcium kernel
  - multi-plane animals `m17` and `m18` pooled across planes for most analyses
- Decoder-related details from the paper:
  - RR position decoder uses deconvolved calcium events
  - includes only samples at running speeds `> 2 cm/s`
  - tenfold cross-validation with 90% train / 10% held-out test
  - for the published decode comparison, occupancy was downsampled to match RR position bins (`2π/45`, about 10 cm)
  - shuffle control circularly shifts each cell’s timeseries independently
- GLM-related details from the paper/code:
  - task variables: linear position, reward-relative position, rewarded
  - movement variables: speed, acceleration, licking
  - 45 cosine bases for position variables; 7 B-spline bases per movement variable
  - training/test split grouped by trial identity (85%/15%)

### Curation Steps

**Neuron curation rules**:
- Manual suite2p curation removed ROIs with multiple somata/dendrites, weak/no obvious transients, overexpression, or putative interneuron-like continuous fluorescence.
- Additional putative interneurons were excluded if Pearson correlation between dF/F and running speed exceeded `0.5`.
- Multi-plane sessions pooled cells across planes for main analyses.

**Trial curation rules**:
- Reward was omitted on about 15% of trials by design.
- Sessions targeted 80-100 imaging trials and could terminate early if behavior or imaging quality degraded.
- Licking analyses removed trials with apparent lick-sensor failures.
- Important discrepancy to resolve later:
  - Methods text says bad lick trials were detected when `>30%` of frame samples had cumulative lick count `>2`
  - `glmUtils.get_timeseries_data` in the code uses a threshold of `>35%`
- Some analyses in the paper further restricted to subsets, for example:
  - rewarded-vs-omission analyses required at least 3 omission trials in a trial set
  - decoder figure used 7 switch days and occupancy-matched sample subsets

### Decoders Trained
| Decoded variable | Accuracy |
| Reward-relative position (circular decoder, Fig. 3) | Paper reports mean decode score versus shuffle, not classifier accuracy; RR cells decode above shuffle before and after the switch |
| RR position decoder spatial extent | z-scored decode `>2` from `-104.5 ± 20.1 cm` to `+152.7 ± 22.9 cm` relative to reward start |
| Deconvolved neural activity from task/movement variables (GLM, Fig. 7 / Ext. Data 9) | FDE `0.10 ± 0.19` all place cells; `0.32 ± 0.13` TR; `0.29 ± 0.11` RR; `0.29 ± 0.11` non-RR |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Storage format | Analysis code loads `sess` / `multi_anim_sess` pickles | Release contains NWB files only | Paper describes processed imaging/behavior data, not NWB structure | Treat NWB as the released representation of the same processed content. Map NWB `Deconvolved` to reference `events`, behavior time series to `sess.vr_data`, and `iscell[:,0]` to curated-cell inclusion. |
| Cohort size | Main analyses use switch-task mice; decoder figure is `77 sessions, 11 mice, 7 switch days` | 11 subjects, 152 session files | Paper also mentions a separate fixed-condition cohort of 3 mice | Released `data/` contains the 11 switch-task mice only. This matches the main RR decoder/GLM analyses and explains why the full 14-mouse cohort is not present. |
| Session count | One imaging session/day; `m11` started on day 3 | 152 sessions total | 14 task days; `m11` started imaging on day 3 | Consistent: `10 mice * 14 + m11 * 12 = 152`. |
| Trial boundaries | Reference code uses `trial_start_inds` and `teleport_inds` | NWB stores continuous `trial_start` and `teleport` vectors, no trials table | Paper describes 30-trial switch point and lap structure | Use `trial_start > 0` and `teleport > 0` to define trial epochs; these match one-for-one in all files. |
| Trial totals | Code does not provide one published total directly | 12216 trials from `trial_start`, 12217 from unique valid `trial number` because one session (`m11` day 3) has a clipped first start marker | Methods mention `12,376` imaged trials across 11 switch mice, with 81 lick-artifact trials removed in licking analyses | Use actual released-data trial counts for conversion and validation. Note the paper total is ~1.3% higher; likely a manuscript-level count from an earlier/pre-release processing state. |
| Lick artifact threshold | `glmUtils.get_timeseries_data` invalidates a trial if `>35%` of samples have cumulative lick count `>2` | NWB stores cumulative lick counts per frame | Methods text states `>30%` | Prefer the released code threshold (`35%`) when reproducing continuous-sample processing, and document the text/code mismatch. |
| Cell count range | Code uses curated `iscell`; later analyses may also exclude putative interneurons | `iscell[:,0]==1` gives 155-2341 accepted cells/session; only three `m18` sessions exceed 2172 | Methods report 155-2172 putative pyramidal neurons/session | Accept `iscell[:,0]==1` as the primary shared-data curation. The slight excess occurs only in pooled multi-plane `m18` sessions and likely reflects release-version differences or later exclusions not baked into NWB. |
| Putative interneuron exclusion | `dayData` documents exclusion by speed correlation threshold `0.5`; helper `spatial.is_putative_interneuron` defaults to `0.3` unless overridden | NWB does not contain a ready-made interneuron mask | Methods report exclusion for Pearson `r > 0.5` | If interneuron exclusion is needed, use the paper/dayData threshold `0.5`, not the helper’s default `0.3`. |
| Sampling rate in multi-plane sessions | Code/paper interpret two-plane imaging as ~15.5 Hz per plane | 28 NWB files have `planeIdx` in `{0,1}` and `Deconvolved.rate = 31.015625`, but behavior timestamps have ~0.0645 s spacing and same sample count as neural traces | Paper states two-plane scans were interleaved at 31 Hz for ~15.5 Hz per plane | Use behavior timestamps / shared sample length as the aligned time base (~15.5 Hz effective). Do not trust the NWB ophys `rate` attribute blindly in multi-plane files. |
| Reward-zone labels | Code infers A/B/C from scene names via string matching | NWB identifiers include scene strings such as `Env1_LocationA_to_C` and cross-environment names like `Env1_A_to_Env2_B` | Paper describes A/B/C zones and environment switches | Code-style scene parsing still works because switch names contain `A_to/B_to/C_to` patterns and the final character identifies the destination zone. |

Final understanding:
- The released NWB files are processed, frame-aligned session-level data equivalent in content to what the repo stores inside `sess`.
- Conversion should therefore avoid re-deriving alignment from raw files and instead preserve the shared frame-aligned time base already present in NWB.
- The core reference semantics to preserve are:
  - neuron inclusion from curated `iscell`
  - trial segmentation from `trial_start` / `teleport`
  - environment identity from `environment`
  - reward outcome from reward deliveries within each trial
  - reward-zone identity from scene/reward-zone metadata consistent with `behavior.get_reward_zones`
  - running-period masking and lick correction logic from `glmUtils.get_timeseries_data` where relevant

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane0/data` + `iscell[:,0]` | `neural` | Filter ROIs where `iscell[:,0] == 1`; transpose from `(T, ROI)` to `(ROI, T)`; segment by complete trials | `multi_anim_sess(..., calc_spks=True)`, `glmUtils.get_timeseries_data` | Use deconvolved activity directly as the NWB equivalent of reference `sess.timeseries['events']`. |
| behavior timestamps within each trial | `input[0]` (`time_from_trial_start_sec`) | `timestamps[start:stop] - timestamps[start]` | frame-aligned continuous sampling in `glmUtils.get_timeseries_data` | Time-varying continuous input. |
| `processing/behavior/BehavioralTimeSeries/environment/data` | `input[1]` (`environment`) | Take per-trial modal value in `{0,1}` and repeat across timepoints | `behavior.get_trial_types` (`morph`) | Binary per-trial input, repeated across time. |
| `processing/behavior/BehavioralTimeSeries/trial number/data` or trial index | `input[2]` (`trial_number`) | Use per-trial integer index (0-based within session), repeated across timepoints | `glmUtils.get_timeseries_data` (`trial_ids`) | Prefer complete-trial order used in segmentation. |
| prior trial reward outcome | `input[3]` (`previous_trial_rewarded`) | Compute from previous complete trial, `0` for first kept trial in a session; repeat across timepoints | derived from `behavior.get_trial_types` logic | Binary per-trial input, repeated across time. |
| `position` + trial-specific reward-zone start/end | `output[0]` (`distance_to_reward_zone`) | Signed nearest distance to zone: `<start => pos-start`, `inside => 0`, `>end => pos-end`, then discretize to 7 bins | reward-zone semantics from `behavior.get_reward_zones`; relative-position logic from `glmUtils.get_timeseries_data` | Bins: `<-50`, `[-50,-10]`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50` cm. |
| `position` | `output[1]` (`absolute_position_bin`) | Discretize absolute track position on `[0,450]` into 5 equal bins | task description in paper; `glmUtils` position handling | Time-varying categorical output. |
| `speed` | `output[2]` (`speed_bin`) | Discretize cm/s into bins `<2`, `2-10`, `10-20`, `20-40`, `>40` | `glmUtils.get_timeseries_data` uses the same speed signal | Keep low-speed samples because the target task explicitly includes the `<2` class. |
| `lick` | `output[3]` (`lick`) | Convert cumulative lick count per frame to binary (`>0 => 1`), after trial-level lick-error filtering | `glmUtils.get_timeseries_data` | Time-varying binary output. |
| session scene / switch structure | `output[4]` (`reward_zone_location`) | Map trial to zone label `A/B/C`, encode as `0/1/2`, repeat across timepoints | `behavior.get_reward_zones` | For switch sessions, use pre-switch zone for trials `<30`, post-switch zone for trials `>=30`. |
| reward delivery within trial | `output[5]` (`reward_outcome`) | Trial is rewarded if any reward delivery occurs within the trial and the trial has an `rzone` entry event; encode `0/1`, repeat across timepoints | `behavior.get_trial_types` | Per-trial categorical output, repeated across time. |

### Key Decisions
1. **Use NWB deconvolved traces directly**: The paper’s RR decoder uses deconvolved calcium events, and the NWB `Deconvolved` data are the released equivalent of reference `sess.timeseries['events']`.
2. **Filter neurons with `iscell[:,0] == 1` only**: This is the shared, explicit curated-cell mask available in NWB and matches the baseline neuron curation used before downstream analyses.
3. **Do not recompute dF/F from fluorescence**: The release already provides frame-aligned deconvolved traces; recomputation would add unnecessary divergence from the released data product.
4. **Segment trials from `trial_start` to `teleport` and exclude teleport periods**: This matches the reference code’s trial-epoch logic and the target requirement to align to trial start.
5. **Keep only complete trials with explicit start and end markers**: This avoids fabricating incomplete boundaries; for example the clipped first `m11` day-3 trial will be excluded.
6. **Represent all inputs and outputs as 2D `(d, T)` arrays**: Although some labels are per-trial, repeating them across timepoints makes dimensions uniform and matches `train_decoder.py` well.
7. **Use actual timestamps for time-from-start and nominal frame size for metadata**: Behavior timestamps are the reliable aligned time base, especially in multi-plane sessions where the ophys `rate` attribute can be misleading.
8. **Use paper/code reward-zone semantics rather than inferring from sparse `reward_zone` events alone**: The `reward_zone` series marks zone entry events, not the full zone extent, so zone A/B/C must come from session condition metadata.
9. **Keep low-speed in-trial samples**: The paper’s Fig. 3 RR decoder filtered to `>2 cm/s`, but the requested target outputs explicitly include a `<2 cm/s` class, so full trial-resolved behavior must be retained.
10. **Drop lick-artifact trials entirely rather than leaving NaNs**: The validator forbids NaNs, and the reference code already treats these trials as invalid for licking analyses.
11. **Set first-trial previous outcome to 0**: There is no previous within-session trial, so `0` is the least assumption-laden sentinel compatible with the requested binary variable.
12. **Sort sessions by file path (`subject`, then `session`)**: This gives stable ordering for `subjects`, `subject_idx`, and reproducible outputs.

### Planned Sanity Checks
- [ ] Neural spot check: for selected session/trial/neuron/timepoint, confirm converted `neural` value equals the raw NWB `Deconvolved` sample after `iscell` filtering and trial slicing using `np.allclose()`.
- [ ] Input time-base check: confirm converted `time_from_trial_start_sec` equals raw behavior timestamps minus trial-start timestamp for several trials using `np.allclose()`.
- [ ] Output position check: confirm converted absolute-position bins and reward-zone-distance bins match raw `position` and manually computed discretization for selected trials using `np.allclose()`.
- [ ] Output reward check: confirm converted trial reward outcome matches raw reward events within trial bounds for selected trials using `np.allclose()`.
- [ ] Session-count check: confirm converted session/trial totals match kept complete trials from NWB after lick-artifact exclusion.
- [ ] Subject/session consistency check: confirm subject list, session ordering, and `subject_idx` match file paths exactly.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `convert_data.py` with the required CLI:
- `python -u convert_data.py <outpicklefile>`
- `--full` / `--sample`
- `--show-processing`

Implementation notes:
- Reads NWB directly with `h5py` for speed.
- Filters curated ROIs using `iscell[:,0] == 1`.
- Uses released deconvolved traces as the neural signal.
- Segments complete trials from `trial_start` to `teleport`.
- Drops lick-artifact trials using the reference-code-style `>35%` rule.
- Builds time-varying `(d, T)` input and output arrays for every kept trial.
- Repeats per-trial labels across timepoints for validator compatibility.
- Saves diagnostic plots for up to 2 sessions in `--show-processing` mode.
- Stores session-level summary metadata for later consistency checks.

Code inefficiencies identified:
- Full-session deconvolved matrices are still loaded into memory one session at a time.
- Reward-zone labels are parsed from scene metadata rather than cached lookup tables.

Code speedups added:
- Uses `h5py` instead of higher-level NWB objects for the main conversion path.
- Processes sessions sequentially and writes only the final pickle, avoiding intermediate I/O.
- Filters trials before building output arrays to avoid wasted allocations for dropped trials.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 323 |
| Neurons / session | [155, 168] |
| Subjects | 1 (`m11`) |
| Sessions / subject | 2 |
| Trials (total) | 160 |
| Trials / session | [80, 80] |
| `time_from_trial_start_sec` range | [0.0, 30.6] |
| `environment` range | [0.0, 0.0] |
| `trial_number` range | [0.0, 79.0] |
| `previous_trial_rewarded` range | [0.0, 1.0] |
| `distance_to_reward_zone` distribution | [0.088, 0.092, 0.058, 0.285, 0.022, 0.071, 0.384] |
| `absolute_position_bin` distribution | [0.312, 0.207, 0.203, 0.146, 0.133] |
| `speed_bin` distribution | [0.121, 0.062, 0.066, 0.233, 0.518] |
| `lick` distribution | [0.820, 0.180] |
| `reward_zone_location` distribution | [0.792, 0.208, 0.000] |
| `reward_outcome` distribution | [0.161, 0.839] |

### Processing Plots Review
- `processing_sub-m11_ses-03_behavior+ophys.png` and `processing_sub-m11_ses-04_behavior+ophys.png` were created.
- Visual spot-check showed:
  - position increases monotonically from trial start to near track end
  - reward-zone shading aligns with the plateau / crossing region in position
  - reward-event markers appear within the active reward zone on rewarded trials
  - lick raster and speed traces line up with trial progression
  - discretized outputs transition at plausible positions/times with no visible off-by-one shift
- No anomalies found in the sample plots.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| Sample conversion | ~0.42 s/session on first 2 sessions | ~64 s for 152 sessions, plus pickle write overhead |
| Verification-only summary | negligible relative to conversion on sample | likely a few minutes on full data due pickle load/summary |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `distance_to_reward_zone` | 0.4211 | 0.3212 |
| `absolute_position_bin` | 0.5267 | 0.4615 |
| `speed_bin` | 0.3524 | 0.3118 |
| `lick` | 0.6754 | 0.6343 |
| `reward_zone_location` | 0.9650 | 0.9669 |
| `reward_outcome` | 0.5940 | 0.6411 |

Additional notes:
- Training loss decreased from `59.82` at epoch 1 to `0.93` at epoch 200.
- All validation balanced accuracies exceeded uniform chance:
  - distance to reward zone: chance `0.1429`
  - absolute position bin: chance `0.2000`
  - speed bin: chance `0.2000`
  - lick: chance `0.5000`
  - reward zone location: chance `0.3333`
  - reward outcome: chance `0.5000`

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9.0G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | not reported directly | curated `iscell` cells, optionally with later exclusions | 138678 accepted cells | 138678 | Yes vs released data |
| Mean neurons/session | range 155-2172/session | curated `iscell` cells per session | 912.36 | 912.36 | Yes vs released data; slight range mismatch to paper max |
| Subjects | 11 switch-task mice | switch-task animal set | 11 | 11 | Yes |
| Sessions | 152 implied (`14 days`, `m11` starts day 3) | one session/day | 152 | 152 | Yes |
| Trials (total) | 12376 imaged trials before manuscript-level lick exclusions | complete trial markers plus lick filtering logic | 12216 complete trials; 12147 after code-style lick trial drops | 12147 | Close; minor paper/release discrepancy remains |
| Trials/session (mean) | 80.5 ± 7.4 | ~80-target sessions | 79.91 kept trials/session | 79.91 | Yes, close to paper mean |
| `time_from_trial_start_sec` range | variable with long omission/slow trials | trial-aligned continuous time | [0.0, 216.5] | [0.0, 216.5] | Yes |
| `environment` range | binary `ENV1/ENV2` | `morph` 0/1 | [0.0, 1.0] | [0.0, 1.0] | Yes |
| `trial_number` range | up to ~100 trials/session | 0-based trial ids in code | [0.0, 99.0] | [0.0, 99.0] | Yes |
| `previous_trial_rewarded` range | binary | derived | [0.0, 1.0] | [0.0, 1.0] | Yes |
| `reward_outcome` distribution | ~15% omissions / ~85% rewarded | reward from trial events | [0.158, 0.842] | [0.158, 0.842] | Yes, close to paper |
| `reward_zone_location` distribution | balanced A/B/C across counterbalanced task | scene-derived A/B/C | [0.332, 0.336, 0.332] | [0.332, 0.336, 0.332] | Yes |
| `absolute_position_bin` distribution | not explicitly reported | N/A | [0.212, 0.177, 0.231, 0.226, 0.154] | [0.212, 0.177, 0.231, 0.226, 0.154] | Yes |
| `speed_bin` distribution | not explicitly reported | N/A | [0.117, 0.087, 0.134, 0.319, 0.342] | [0.117, 0.087, 0.134, 0.319, 0.342] | Yes |

Additional notes:
- Full conversion runtime: `3.36 min` for all 152 sessions.
- Full verification reported: `Data format is valid, no errors or warnings.`
- Kept trials total (`12147`) reflects dropping lick-artifact trials using the code-consistent `>35%` threshold.
- Omission fraction (`15.8%`) matches the paper’s stated `~15%` omission design closely.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. `verification_full_out.txt` review: passed cleanly with `Data format is valid, no errors or warnings.`
2. Raw-to-converted neural sanity checks with `np.allclose()`: passed on 5 sessions spanning single-plane and multi-plane data (`m11`, `m17`, `m18`, `m3`, `m4`).
3. Raw-to-converted input sanity checks with `np.allclose()`: passed for `time_from_trial_start_sec` on the same sessions.
4. Raw-to-converted output sanity checks with `np.allclose()`: passed for discretized outputs on the same sessions.
5. Reference-code comparison:
   - loading/curation matches `iscell`-filtered `sess` content
   - trial segmentation matches `trial_start_inds` / `teleport_inds`
   - reward outcome matches `behavior.get_trial_types`
   - reward-zone labels match `behavior.get_reward_zones`
   - continuous trial-aligned features match `glmUtils.get_timeseries_data` semantics except for the intentional retention of low-speed samples required by the target task
6. Key statistics comparison: subject count, session count, reward omission fraction, and trial/session mean are all consistent or very close to the paper and released data.
7. Edge-case checks:
   - `m11` day 3 has 81 raw trial numbers but only 80 complete `trial_start`/`teleport` trial epochs; converter correctly keeps the 80 complete trials and starts at trial number 0.
   - Longest kept trial is session `sub-m4_ses-04`, trial 39, `T=3359`, `216.5 s`; raw position still spans only `0.18-449.14 cm` with no teleport samples, so this is a genuine slow/paused track traversal rather than a segmentation bug.
   - Multi-plane sessions (`m17`, `m18`) now pass raw-to-converted checks after concatenating plane-wise deconvolved series in pooled ROI order.

### Issues Found and Resolved
- Initial full-conversion issue: multi-plane NWB sessions store `Deconvolved/plane0` and `Deconvolved/plane1` separately, while `iscell` indexes pooled ROIs.
  Resolution: updated `convert_data.py` to reconstruct the accepted-cell deconvolved matrix by concatenating per-plane series into pooled ROI order before trial slicing.
- No additional issues were found during Step 10 after the multi-plane fix.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (`168.3061` at epoch 1 -> `1.1868` at epoch 200; test loss `1.0654`)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| distance_to_reward_zone | 0.4635 | 0.4137 | 2.90x chance (`0.1429`) |
| absolute_position_bin | 0.5223 | 0.4961 | 2.48x chance (`0.2000`) |
| speed_bin | 0.4487 | 0.4287 | 2.14x chance (`0.2000`) |
| lick | 0.6562 | 0.6407 | 1.28x chance (`0.5000`) |
| reward_zone_location | 0.8451 | 0.8146 | 2.44x chance (`0.3333`) |
| reward_outcome | 0.5662 | 0.5182 | 1.04x chance (`0.5000`); weakest decode |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| distance_to_reward_zone | Validation balanced accuracy `0.4137` (`2.90x` chance) | Qualitatively consistent with the paper's strong above-shuffle reward-relative position decoding from CA1 RR activity (Fig. 3), though the paper reports circular decode score / z-score rather than classifier balanced accuracy. |
| absolute_position_bin | Validation balanced accuracy `0.4961` (`2.48x` chance) | Consistent with strong spatial coding in CA1 place-cell activity. |
| speed_bin | Validation balanced accuracy `0.4287` (`2.14x` chance) | Consistent with movement variables contributing to the GLM in the paper. |
| lick | Validation balanced accuracy `0.6407` (`1.28x` chance) | Modest but sensible; licking is sparse and the paper models lick as a smoothed movement regressor rather than a separately reported decoder target. |
| reward_zone_location | Validation balanced accuracy `0.8146` (`2.44x` chance) | Strongly consistent with reward-centered remapping and environment/reward-location coding in the paper. |
| reward_outcome | Validation balanced accuracy `0.5182` (`1.04x` chance) | Weak but above chance. The paper does not decode a per-trial reward-outcome label from trial start; instead it models a time-varying `rewarded` regressor that switches from `0` to `1` only after reward delivery. This task definition is therefore harder than the paper's formulation. |

Chance analysis:
- All six outputs are above chance.
- Four outputs exceed `2x` chance (`distance_to_reward_zone`, `absolute_position_bin`, `speed_bin`, `reward_zone_location`).
- The two binary outputs are weaker in ratio-to-chance terms because balanced-accuracy chance is already `0.5`.

Train-vs-validation gap analysis:
- `distance_to_reward_zone`: train `0.4635`, val `0.4137`, gap `0.0498`, train/val `1.12`
- `absolute_position_bin`: train `0.5223`, val `0.4961`, gap `0.0262`, train/val `1.05`
- `speed_bin`: train `0.4487`, val `0.4287`, gap `0.0200`, train/val `1.05`
- `lick`: train `0.6562`, val `0.6407`, gap `0.0155`, train/val `1.02`
- `reward_zone_location`: train `0.8451`, val `0.8146`, gap `0.0305`, train/val `1.04`
- `reward_outcome`: train `0.5662`, val `0.5182`, gap `0.0480`, train/val `1.09`
- No output shows the `>1.5x` train/validation gap that would suggest serious overfitting or leakage.

Paper comparison details:
- Fig. 3 reports reward-relative position decoding as decode-score-vs-shuffle, not balanced accuracy.
- The paper's strongest quantitative decoder-like statistic is the RR decode z-score range `>2` from `-104.5 ± 20.1 cm` to `+152.7 ± 22.9 cm` relative to reward start.
- The GLM section reports fraction deviance explained (`0.10 ± 0.19` overall) and uses a time-varying `rewarded` regressor, not the per-trial reward outcome label required here.
- Direct numeric equality is therefore not possible; the relevant consistency criterion is qualitative: strong position/reward-zone decoding, moderate movement-variable decoding, and weaker reward/omission decoding.

Low-accuracy investigation performed:
- Spot-checked raw NWB against converted data on `sub-m11_ses-03_behavior+ophys.nwb` trials `0`, `1`, `2`, and `6`.
- For all four trials, `reward_outcome`, `lick`, `reward_zone_location`, and time vectors matched exactly between `process_session(...)` and `converted_data.pkl` (`np.allclose=True`).
- Direct raw reward counts within those trial windows were:
  - trial `0`: `1` reward event
  - trial `1`: `0` reward events
  - trial `2`: `1` reward event
  - trial `6`: `0` reward events
- This confirms that the weak `reward_outcome` decoder result is not caused by a labeling or alignment bug in the converted dataset.

### Issues Found and Resolved
- No conversion bugs were identified in Step 12.
- Potential concern: `reward_outcome` balanced accuracy is only slightly above chance.
  Resolution: investigated directly against raw NWB trials and the paper's task-variable definition. The output is correct as specified, but it is inherently harder than the paper's time-varying `rewarded` regressor, so no code change was made.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Cleanup notes:
- Created `README.md` with dataset summary, format description, and loading example.
- Created `cache/README_CACHE.md` to document temporary-artifact policy.
- Required deliverables (`converted_data.pkl`, decoder logs, processing plots) were intentionally retained at the project root because the task specification requires them as named outputs.
