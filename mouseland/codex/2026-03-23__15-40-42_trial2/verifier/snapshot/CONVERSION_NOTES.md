# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks dataset and code bundle
- **Date started**: 2026-03-23
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
| `load_exp_beh(root, exp_type)` | `code/utils.py` | LOADING | Loads behavior dictionary from `beh/Beh_<exp_type>.npy`. |
| `load_retino(db, root='')` | `code/utils.py` | LOADING | Loads retinotopy transform and area labels from `<mouse>_<date>_trans.npz`; provides area masks `V1`, `mHV`, `lHV`, `aHV`. |
| `load_spk(db, root='')` | `code/utils.py` | LOADING | Loads session neural data from `<mouse>_<date>_<blk>_neural_data.npy` and concatenates the list in `['spks']` into a neuron x frame matrix. |
| `load_interp_spk(db, root='')` | `code/utils.py` | LOADING | Loads saved spatially interpolated neural activity from `process_data/*_interpolate_spk.npy`. |
| `interp_value(v, vind, tind)` | `code/utils.py` | PROCESSING | 1D interpolation helper used for remapping framewise activity onto position coordinates. |
| `spk_pos_interp(raw_spk, accum_pos, corridorLen, new_shape)` | `code/utils.py` | PROCESSING | Interpolates each neuron's activity from framewise cumulative position to trial x position-bin form. |
| `get_interpPos_spk(spk, spk_culm_pos, ntrial, n_bins=60, lengths=60, save_path='')` | `code/utils.py` | PROCESSING | Canonical conversion from neuron x moving-frame activity to neuron x trial x 60 position-bin activity. |
| `get_cat_id(WallName, isRew)` | `code/utils.py` | PROCESSING | Infers reward/non-reward stimulus category IDs from wall names and rewarded trials. |
| `lickCount(dats, def_range=[])` | `code/utils.py` | PROCESSING | Computes binary per-trial lick summaries (`befRew`, `aftRew`, `inRange`). |
| `Get_dprime_selective_neuron(db, Beh, stim_ID, root='')` | `code/utils.py` | CURATION | Computes stimulus selectivity (`d'`) using moving corridor frames only. |
| `Get_dprime_rewPred_neuron(...)` | `code/utils.py` | CURATION | Defines reward-prediction neurons using interpolated activity, reward-stimulus trials, and early/late cue split. |
| `Get_coding_direction(...)` | `code/utils.py` | PROCESSING | Main population analysis: odd trials define selective neurons; interpolated activity normalized relative to gray space and stimulus-specific corridor variability; appends 10 bins from previous trial gray space. |
| `Get_sort_spk(...)` | `code/utils.py` | PROCESSING | Uses odd trials for selective-neuron definition and odd/even test splits for sequence sorting. |
| `get_kfold_reward_response(root, db, Beh)` | `code/utils.py` | PROCESSING | k-fold reward decoding pipeline aligned to cue and first lick; uses z-scored activity and interpolated position activity. |

### Notes
- `code/README.md` states `data_process_script.ipynb` is the reference processing notebook and `Figures.ipynb` reproduces figures.
- `data_process_script.ipynb` documents the behavior fields explicitly and states calcium imaging frame rate is `3.17 Hz`.
- The notebook loads session metadata from `beh/Imaging_Exp_info.npy`; this indicates the dataset used in the paper is calcium imaging, not electrophysiology.
- The reference code does **not** compute delta-F-over-F inside this repository. `load_spk()` directly loads `*_neural_data.npy` and concatenates `['spks']`, so the saved neural signal is already the processed neural activity used downstream.
- The canonical spatial representation is `60` bins across a `6 m` corridor, i.e. `1 dm` spatial bins (`data_process_script.ipynb`, cell 9).
- Interpolation uses only frames with `ft_move > 0` (mouse moving). The notebook saves `process_data/*_interpolate_spk.npy` produced by `utils.get_interpPos_spk(spk[:, VRmove], ft_AcumPos[VRmove], ntrials, n_bins=60, lengths=CL)`.
- Stimulus selectivity / coding analyses use moving corridor frames only: `corr_fr = beh['ft_CorrSpc'][:nfr] & (beh['ft_move'][:nfr] > 0)`.
- Trial splitting in the reference code is by parity of trial index:
  - `ft_trInd % 2 == 0` is used as the "odd trials" training split in comments/code.
  - `trInd % 2 == 1` is used as the held-out/even split for projection or sorting outputs.
- `Get_coding_direction()` performs a specific normalization on interpolated activity:
  - keep texture area `:40` bins,
  - compute gray baseline from bins `42:52`,
  - normalize by combined std from reference stimulus trials,
  - prepend `n_bef=10` bins from the previous trial gray segment.
- Neuron curation is analysis-specific rather than a universal dataset filter:
  - retinotopy areas are derived from `iarea`,
  - some analyses exclude neurons outside visual cortex (`iarea == -1` or `7`),
  - selective-neuron analyses require corridor responsiveness (`corr_neu`) and d-prime percentile thresholds.
- Behavioral variables directly represented in code and likely important for conversion: trial start/end, cue time/position/frame, reward time/position/frame, lick times/positions/frames, framewise movement, framewise position, trial stimulus identity, reward status, and run speed.
- Reward-related analyses explicitly treat cue position and reward position as highly correlated in reward corridors, using `SoundDelPos`/`SoundFr` as alignment anchors in several functions.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- No README or text documentation exists inside `data/`.
- Top-level organization:
  - `data/beh/`: behavior dictionaries and experiment metadata.
  - `data/spk/`: raw session neural activity files `*_neural_data.npy`.
  - `data/retinotopy/`: per-session retinotopy transforms `*_trans.npz` plus `areas.npz`.
- File counts:
  - `beh/`: 25 files
  - `spk/`: 89 files
  - `retinotopy/`: 90 files
- Approximate storage:
  - `beh/`: `6.6G`
  - `spk/`: `405G`
  - `retinotopy/`: `170M`
- `Imaging_Exp_info.npy` is the central metadata index. It is a dict with 23 experiment types:
  - `naive_test1`, `naive_test2`, `naive_test3`
  - `sup_test1`, `sup_test2`, `sup_test3`
  - `sup_train1_after_learning`, `sup_train1_before_learning`
  - `sup_train2_after_learning`, `sup_train2_before_learning`
  - `test1_after_grating`, `test1_before_grating`
  - `test2_after_grating`, `test2_before_grating`
  - `train1_after_grating`, `train1_before_grating`
  - `unsup_test1`, `unsup_test2`, `unsup_test3`
  - `unsup_train1_after_learning`, `unsup_train1_before_learning`
  - `unsup_train2_after_learning`, `unsup_train2_before_learning`
- Each `exp_info[exp_type]` entry is a list of session dicts with fields drawn from:
  - `mname`, `datexp`, `blk`, `sess#`, `stim`, `stim_id`, `Gender`, `depth`, `rewType`, `Note`, `is2p`
  - some entries also include `stimtype`, `2pblk`, `artLick`, `days`, `exptype`, `isDR`
- Session identity convention:
  - standard session key: `<mouse>_<YYYY_MM_DD>_<block>`
  - `test3` experiments append stimulus swap subtype: `<mouse>_<YYYY_MM_DD>_<block>_swap1` or `_swap2`
- Behavior files `Beh_<exp_type>.npy` are dicts keyed by session key. Each session contains trialwise arrays and framewise arrays.
- Representative behavior structure from `Beh_naive_test1.npy`:
  - `ntrials = 453`
  - `Corridor_Length = 60.0`, `Texture_Length = 40.0`, `Gray_Space_length = 20.0`
  - `Reward_Mode = "Active after cue"`, `Reward_Delay_ms = 1000`
  - `WallName.shape = (453,)`
  - `ft.shape = (23194,)`
  - `run_pos.shape = (453, 60)`
  - `LickFr.shape = (0,)` in the sampled session, showing that no-lick sessions/trials exist
- Representative behavior variables available per session:
  - trialwise: `WallName`, `UniqWalls`, `stim_id`, `isRew`, `Trial_start_time`, `Trial_end_time`, `SoundPos`, `SoundTime`, `RewPos`, `RewTime`, `StartFr`, `SoundFr`, `RewardFr`, `GrayFr`, `EndFr`
  - lick-related: `LickTrind`, `LickTime`, `LickPos`, `LickFr`
  - framewise: `ft`, `ft_trInd`, `ft_Pos`, `ft_PosCum`, `ft_WallID`, `ft_move`, `ft_RunSpeed`, `ft_CorrSpc`, `ft_GraySpc`
  - already position-binned behavior: `run_pos` with shape `(ntrials, 60)`
- Retinotopy files contain `A`, `xpos`, `ypos`, `xy_t`, `iarea`; neuron count is `len(iarea)`.
- Example retinotopy file `VR2_2021_03_20_trans.npz` has `iarea.shape = (81473,)`.
- `test3` behavior confirms swap-specific split keys:
  - example keys: `TX119_2023_12_13_1_swap1`, `TX119_2023_12_13_1_swap2`
  - example `UniqWalls`: `['circle1', 'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2']`
  - example `stim_id`: `[0., 2., 5., nan, 3.]` or `[0., 2., nan, 6., 3.]`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 |
| Neurons / session | min 20,547; mean 52,708.25; max 89,577 |
| Subjects | 19 |
| Sessions / subject | min 1; mean 4.68; max 8 |
| Trials (total) | 38,110 |
| Trials / session | min 84; mean 428.20; max 789 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not directly reported | "89 recordings in 19 mice" + "20,547 to 89,577 neurons in each recording" |
| Neurons / session | 20,547 to 89,577 | "20,547 to 89,577 neurons in each recording" |
| Subjects | 19 | "89 recordings in 19 mice" |
| Sessions / subject | Not directly reported in text | "each mouse can have more than one imaging session" |
| Trials (total) | Not directly reported in text | Not directly stated |
| Trials / session | Not directly reported in text | Not directly stated |
| Neural data time bin | Framewise deconvolved traces; exact Hz not stated in paper/methods excerpt | "All our analyses were based on deconvolved fluorescence traces" |
| Behavior data time bin | Running speed interpolated at 0.1 m position bins for some analyses | "0–6 m, with a 0.1-m step size" |
| Reward rate | Approximately 50% by task design; one rewarded corridor of two | "water in rewarded trials only" |
| Corridor structure | 4 m texture corridor + 2 m grey space | "corridors were each 4 m long, with 2 m of grey space" |
| Cue position | Uniform 0.5 m to 3.5 m from corridor entry | "uniform distribution between positions 0.5 m and 3.5 m" |
| Reward zone / reward position | Begins at cue in imaging task; reward positions approx. uniform 0.5 m to 3.5 m | "reward delivery locations were also approximately uniformly distributed between 0.5 m and 3.5 m" |
| Running threshold | VR moves when speed exceeds 6 cm/s | "running faster than a threshold of 6 cm s−1" |
| Selective neuron threshold | d′ >= 0.3 or d′ <= -0.3 | "d′ ≥ 0.3 or d′ ≤ −0.3" |
| Coding-direction neuron selection | Top 5% selective neurons from train trials for each reference stimulus | "top 5% selective neurons each" |


### Processing Details
- Task structure:
  - Linear VR corridor with textures in the first `0–4 m` and grey space in `4–6 m`.
  - Trials are temporally aligned in the paper to corridor progression and, for reward-prediction analyses, additionally to sound cue and first lick.
  - Sound cue is present in all imaging trial types; for task mice it signals reward availability in the rewarded corridor.
- Neural signal:
  - Two-photon calcium imaging processed with Suite2p.
  - Analyses use deconvolved fluorescence traces, not raw fluorescence and not freshly computed dF/F in this repository.
- Core inclusion rule used throughout the paper:
  - "We only considered timepoints during running for analysis."
  - For selectivity/coding analyses, only the texture area (`0–4 m`) is used for d-prime calculations.
- Sequence similarity analysis:
  - Use half of leaf1/circle1 trials as train trials to compute d′ and select neurons.
  - Use the other half as test trials, then split test trials into odd/even to compute preferred positions and correlations.
- Coding direction / similarity index:
  - Select leaf1- and circle1-selective neurons from train trials using the top 5% d′ tails.
  - Interpolate neural activity by position.
  - Normalize each neuron by subtracting grey-space baseline and dividing by the average of leaf1/circle1 response standard deviations.
  - Evaluate coding direction only on held-out trials.
  - Average projections within the texture area (`0–4 m`) to obtain similarity indices.
- Reward-prediction analysis:
  - Interpolate neural activity into trial x position matrices.
  - Restrict to leaf1 trials and split into early-cue vs late-cue trials using sound cue position.
  - Use cue position rather than reward position because cue and reward are highly correlated in rewarded trials.
  - Use 10-fold cross-validation: 9 folds to identify reward-prediction neurons, 1 fold to evaluate held-out trial activity.
- Running-speed analysis:
  - Keep epochs with speed > `6 cm/s` for at least `66 ms`.
  - Interpolate running speed both to imaging-frame times and to position bins (`0–6 m`, step `0.1 m`).

### Curation Steps

**Neuron curation rules**:
- Upstream curation is handled by Suite2p: motion correction, ROI detection, cell classification, neuropil correction, spike deconvolution.
- For selectivity-based analyses, neurons are filtered by d′ thresholds (`>= 0.3` or `<= -0.3`) or by top-5% selectivity tails depending on analysis.
- Reward-prediction neuron selection uses `d′late vs early >= 0.3`.
- Area-based analyses aggregate neurons into `V1`, medial, lateral, and anterior visual regions.

**Trial curation rules**:
- Running-only timepoints are analyzed; non-running periods are excluded.
- Texture-based selectivity uses only positions within `0–4 m`.
- Sequence/coding analyses use held-out trials, with explicit train/test and odd/even splits.
- First-lick-aligned reward-prediction analysis includes only rewarded leaf1 trials with first lick later than `2 m` from corridor entry.
- Leaf2 lick/no-lick comparison excludes a mouse with only one no-lick leaf2 trial.

### Decoders Trained
| Decoded variable | Accuracy |
| No directly comparable generic decoder reported | The paper does not report balanced/classification accuracy for the custom decoder task in this project |
| Visual category separation (coding direction / similarity index) | Strong separation reported qualitatively; not reported as classifier accuracy |
| Reward prediction (late vs early cue leaf1 trials) | Identified via `d′late vs early`, not reported as classifier accuracy |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Recording count | Notebook iterates all `exp_info` sessions | `89` unique `spk/*_neural_data.npy` sessions | "89 recordings in 19 mice" | Consistent. Use all 89 unique imaging sessions indexed by `Imaging_Exp_info.npy`. |
| Subject count | `Imaging_Exp_info.npy` contains repeated mouse IDs across experiments | `19` unique `mname` values | "19 mice" | Consistent. Use `mname` as subject ID. |
| Corridor geometry / units | Code uses `n_bins=60`, texture area `:40`, previous/grey bins near `40:60` | `Corridor_Length=60`, `Texture_Length=40`, `Gray_Space_length=20`, `run_pos.shape=(ntrials, 60)` | "4 m long, with 2 m of grey space" | Consistent after unit conversion: raw data are in decimeters over a 6 m total corridor. |
| Running-only analysis | Code uses `ft_move > 0` and `ft_CorrSpc` masks | Behavior files contain `ft_move`, `ft_CorrSpc`, `ft_GraySpc`, `ft_RunSpeed` | "We only considered timepoints during running" | Consistent. Conversion should restrict frame-based analyses to running periods when mirroring reference logic. |
| Trial parity naming | Code comments say "odd trials" but uses `ft_trInd % 2 == 0` and `trInd % 2 == 1` | `trInd_odd` is `True` on raw trial indices `0,2,4,...` | Paper says train/test split with odd/even trials | Resolved as a naming convention issue: arrays are zero-based, so "odd trials" in the paper/comments means 1st/3rd/5th... trials. Mirror the exact boolean logic from the reference code, not the literal wording. |
| Cue position field | Code sometimes uses `SoundPos`; reward-prediction code often uses `np.mod(SoundDelPos, 60)` | Raw `SoundDelPos` can be cumulative and exceed corridor length; modulo-60 yields within-corridor positions. `SoundPos` and `RewPos` are also within-corridor scales. | Paper describes cue positions within the corridor | Resolved by following the reference code: treat cue/reward position variables in within-corridor coordinates, using modulo corridor length where needed. |
| Cue-position range | Raw data ranges are slightly broader than the simplified paper wording (`SoundPos` roughly `1.1` to `39.0` dm; `RewPos` up to `43.8` dm) | Empirical ranges extend a bit beyond the paper’s idealized `0.5–3.5 m` statement | Paper says cue positions were uniform between `0.5 m` and `3.5 m` | Minor discrepancy likely from delayed-cue / passive-reward session details and sampling noise. Keep raw within-corridor values from data, not clipped values. |
| Swap stimuli handling | Code keeps `leaf1_swap1` and `leaf1_swap2` as separate IDs/files | `test3` behavior keys are split into `_swap1` / `_swap2` sessions | Paper says swap types are pooled for statistics | Resolve by preserving raw swap subtype identity in converted trial labels; pooling, if needed, should happen only in downstream statistics. |
| Behavior-only versus imaging data | Code notebook centers on `Imaging_Exp_info.npy` and matching `spk/` + `retinotopy/` | `data/beh/example_bef_and_aft_learning_behavior.npy` exists without matching neural files | Paper includes both imaging and behavior-only experiments | For the decoder conversion, use only imaging sessions that have matched neural and retinotopy data. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk/*_neural_data.npy` -> `['spks']` via `utils.load_spk()` | `neural` | Load deconvolved traces; slice per unique session and trial using frame indices between `StartFr` and `GrayFr`; retain only running frames (`ft_move > 0`) within the texture area; keep curated neurons only | `load_spk`, `Get_dprime_selective_neuron`, `Get_coding_direction`, `get_kfold_reward_response` | Running-only and texture-only follows the paper; `GrayFr` end keeps exactly the 0–4 m texture interval. |
| `retinotopy/*_trans.npz` -> `iarea` | `brain_region_idx` | Map neurons to `V1`, `mHV`, `lHV`, `aHV`, else `other` | `load_retino`, `neu_area_ID` | Preserve all curated neurons and record their broad visual-region label. |
| Session key / recording date | `subjects`, `subject_idx` | `subjects = sorted(unique mname)`; `subject_idx` indexes subject per session | `Imaging_Exp_info.npy` | Sessions are deduplicated by base session ID `<mouse>_<date>_<blk>`. |
| `beh['SoundTime']`, `beh['ft']`, `StartFr:GrayFr` running frames | `input[0]` = `time_to_sound_cue_sec` | For each retained frame, `(SoundTime[trial] - ft[frame]) * 86400`; can be positive before cue and negative after cue | Paper Methods, `get_kfold_reward_response`, `spk_2_cue` | Continuous time-varying input in seconds. |
| Session recording date / optional `db['days']` | `input[1]` = `day_of_training` | Use elapsed calendar days since the mouse’s first recording date; repeat across retained frames in the trial | `Imaging_Exp_info.npy` | Chosen because a complete per-session training-day label is not available for all sessions. |
| `beh['ft']`, `StartFr` | `input[2]` = `time_since_trial_start_sec` | `(ft[frame] - ft[StartFr[trial]]) * 86400` for each retained frame | Behavior frame timing fields | Continuous time-varying input in seconds. |
| `beh['isRew']` | `input[3]` = `reward_available` | Trialwise 0/1 repeated across retained frames | Behavior session dict | Uses rewarded-corridor identity even in unsupervised sessions, matching reference use of `isRew`. |
| `beh['WallName']` | `output[0]` = `visual_stimulus_category` | Map exact wall name to integer category; repeat across retained frames | Behavior session dict | Use `WallName` directly so swap sessions and non-leaf/circle stimuli are preserved correctly. |
| `beh['LickFr']`, `beh['LickTrind']` | `output[1]` = `licking` | Binary per retained frame: 1 if one or more licks occur on that imaging frame, else 0 | `spk_2_firstLick`, `spk_2_cue`, behavior docs | Time-varying categorical output. |
| `beh['ft_Pos']` on retained frames | `output[2]` = `position_bin` | Discretize `0–40 dm` into 4 equal bins: `[0,10), [10,20), [20,30), [30,40]` | Paper Methods (0–4 m texture area) | Exactly matches the requested 4 one-meter bins. |
| `beh['ft_RunSpeed']` on retained frames | `output[3]` = `running_speed_bin` | Compute global quartile edges over all retained running frames in all sessions; assign bin 0–3 | Paper running-speed interpolation + user decoder spec | Quartiles computed on the included running-only texture frames. |

### Key Decisions
1. **Deduplicate 142 behavior entries down to 89 unique neural sessions**: The same recording appears under multiple analysis labels in `Imaging_Exp_info.npy`. Conversion will use one canonical copy per base session ID `<mouse>_<date>_<blk>`, preferring a non-`stimtype` entry when present. For `test3` swap sessions, either suffix copy is sufficient because `WallName` contains the true trial label and the framewise data are duplicated.
2. **Use the texture segment only (`StartFr:GrayFr`)**: The decoder output requires exactly four 1 m position bins. Ending trials at grey-space entry also matches the paper’s `0–4 m` texture-area analyses.
3. **Keep running frames only inside the texture segment**: This matches the reference paper/code rule that analyses use running timepoints and removes extremely long paused trials. Actual elapsed time is still preserved through the continuous time inputs.
4. **Curate neurons using paper-defined task relevance**: Keep neurons in retinotopically assigned visual regions that are either:
   - corridor-responsive and stimulus-selective with `|d′| >= 0.3` between the primary rewarded and non-rewarded exemplar-1 corridors, or
   - reward-prediction neurons with `d′late_vs_early >= 0.3` on the rewarded exemplar-1 corridor.
   This follows the two main neuron-selection motifs in the paper and keeps the conversion computationally tractable.
5. **Use exact `WallName` strings as stimulus labels**: This avoids ambiguity from `stim_id` NaNs in swap sessions and preserves non-leaf/circle texture pairs (`rock*`, `wood*`).
6. **Represent all decoder inputs as time-varying arrays**: Even trial-constant variables (`day_of_training`, `reward_available`) will be repeated across timepoints so the dataset is uniform and easy for `train_decoder.py`.
7. **Use actual frame times in seconds, not just frame index**: Because non-running frames are dropped, elapsed time inputs must come from `ft` timestamps rather than assuming contiguous native-frame spacing.
8. **Use day-since-first-recording as the training-day proxy**: The paper does not provide a complete per-session training-day label, while calendar date is available for every recording. The small subset of sessions with `db['days']` can be used later as a sanity check.

### Planned Sanity Checks
- [ ] For a sampled session/trial/neuron, verify the converted neural trace matches the raw `spk` values at the retained running frames using `np.allclose()`.
- [ ] For sampled trials, verify `time_since_trial_start_sec + time_to_sound_cue_sec` equals the trial’s cue time-from-start at every retained frame.
- [ ] For sampled trials, verify licking output is exactly the binary projection of raw `LickFr` events onto retained imaging frames.
- [ ] For sampled trials, verify position bins come directly from raw `ft_Pos` thresholds at retained frames.
- [ ] For sampled trials, verify running-speed bins correspond to the stored global quartile edges applied to raw `ft_RunSpeed`.
- [ ] Verify session/subject/trial counts after deduplication equal `89` sessions, `19` subjects, and `38,110` trials.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `convert_data.py` with the required CLI:
  - `python -u convert_data.py <outpicklefile>`
  - `--full`
  - `--sample`
  - `--show-processing`
- The script reuses reference loaders from `code/utils.py` for neural data loading and broad area definitions.
- Implemented deduplication from 142 analysis entries to 89 unique neural recordings.
- Implemented trial slicing on running-only texture frames (`StartFr:GrayFr`, `ft_move > 0`).
- Implemented stimulus-pair detection using `stim_id == 2` versus `stim_id == 0`, with fallbacks for edge cases.
- Implemented paper-style neuron selection:
  - stimulus-selective neurons with `|d′| >= 0.3`
  - reward-prediction neurons with `d′late_vs_early >= 0.3`
  - fallback top-|d′| selection if too few neurons pass
- Implemented time-varying input/output construction and global running-speed quartile binning.
- Implemented per-session processing plots for up to 2 sessions.
- Step 10 iteration 1 updated neuron curation to better match the reference code:
  - removed the extra gray-space response gate from the generic stimulus-selective pool because `utils.Get_dprime_selective_neuron()` does not apply that gate;
  - changed reward-neuron selection to require cue-frame stimulus selectivity `d′ > 0.3` plus late-vs-early cue `d′` above the session-specific 95th percentile within aHV;
  - restricted the reward-prediction mean response to running frames in the `0.5–4.0 m` texture segment (`ft_Pos` in `[5, 40]` dm), matching the raw-data intent of `interp_spk[:, :, 5:40].mean(2)`.

Code inefficiencies identified:
- Full-mode metadata currently recomputes frame-interval medians by reloading behavior files.
- Stimulus vocabulary collection reloads behavior dictionaries once more after catalog creation.

Code speedups added:
- `--sample` processes the two smallest neural recordings rather than the first two sessions.
- Session processing loads each large spike file only once, computes the neuron mask, then immediately drops the full matrix after subsetting.
- Speed quantiles are computed from concatenated retained running frames only, not all raw frames.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 8,361 curated neurons |
| Neurons / session | 5,906; 2,455 |
| Subjects | 2 |
| Sessions / subject | 1; 1 |
| Trials (total) | 563 |
| Trials / session | 84; 479 |
| `time_to_sound_cue_sec` range | [-13.5, 11.0] |
| `day_of_training` range | [11.0, 73.0] |
| `time_since_trial_start_sec` range | [0.0, 16.0] |
| `reward_available` range | [0.0, 1.0] |
| `visual_stimulus_category` distribution | [circle1 0.350, leaf1 0.390, leaf2 0.118, leaf3 0.142] |
| `licking` distribution | [no_lick 0.873, lick 0.127] |
| `position_bin` distribution | [0.247, 0.245, 0.250, 0.258] |
| `running_speed_bin` distribution | [0.250, 0.250, 0.250, 0.250] |

### Processing Plots Review
- `processing_TX109_2023_03_27_1.png` and `processing_TX60_2021_06_22_1.png` were generated successfully.
- First sample attempt used the two smallest sessions globally and produced a degenerate sample with `reward_available == 0` and `licking == 0` for all timepoints; this was fixed by changing `--sample` to choose the two smallest canonical sessions that contain both reward variation and lick events.
- Updated sample plots were regenerated after this fix.
- Step 10 iteration 1 reran the sample conversion after tightening neuron curation to match the reference reward-prediction code more closely; `TX109_2023_03_27_1` dropped from `6,273` to `5,906` curated neurons and `TX60_2021_06_22_1` from `2,612` to `2,455`.
- No structural anomalies were reported by `train_decoder.py --verify-only`.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Representative-session sample selection | Faster debugging and avoids meaningless all-zero licking/reward sample |
| Removed repeated full-catalog rebuilds and repeated behavior reloads in `main()` | Small but measurable startup reduction |
| Session-level processing loads each spike file once, subsets neurons, then drops the full array | Reduces peak memory during conversion |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion session 1 (`TX109_2023_03_27_1`) | 3.7 s | |
| Sample conversion session 2 (`TX60_2021_06_22_1`) | 5.3 s | |
| Full conversion estimate (size-weighted over 404.36 GB raw spike files) | ~2.10 s / GB | ~870 s (~14.5 min) plus modest validation overhead |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings:
  - `train_decoder.py` completed, but `sklearn.metrics` emitted `y_pred contains classes not in y_true` during balanced-accuracy computation on the sample split. This is a split-specific evaluation warning rather than a data-format warning; the sample has only 2 sessions and some stimulus classes appear only in one session.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| visual_stimulus_category | 0.8812 | 0.8755 |
| licking | 0.6563 | 0.6454 |
| position_bin | 0.6377 | 0.6104 |
| running_speed_bin | 0.5345 | 0.5046 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `13448882052` bytes (`~12.5 GiB`)
- `verification_full_out.txt`: created (`26178` bytes)
- `conversion_full_out.txt`: created (`12562` bytes); corrected full conversion completed in `501.4 s`

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not reported as a dataset total in the paper excerpt | Uses selective / reward-prediction subsets rather than all cells for main analyses | `4,691,034` raw neurons across all `retinotopy/*_trans.npz` files | `357,328` curated neurons retained for decoder | Partial: curated count intentionally differs from raw total, but now follows the corrected paper-style curation more closely |
| Mean neurons/session | Imaging recordings reported in the tens of thousands of neurons per recording before analysis-specific selection | Analysis code repeatedly selects subsets by `d'` threshold or top-5% tails | `52,708.25` raw neurons / session | `4,014.92` curated neurons / session | Partial: expected after analysis-style neuron curation |
| Subjects | `19 mice` | `19` unique `mname` values in `Imaging_Exp_info.npy` | `19` | `19` | Yes |
| Sessions | `89 recordings` | Notebook iterates `89` unique imaging recordings after deduplication | `89` unique matched neural sessions | `89` | Yes |
| Trials (total) | Not explicitly tabulated in paper excerpt | Derived from behavior session files | `38,110` deduplicated trials across the `89` canonical sessions | `38,110` | Yes |
| Trials/session (mean) | Not explicitly reported | Derived from behavior session files | `428.2` | `428.2` | Yes |
| `time_to_sound_cue_sec` range | Not directly reported in paper | Cue-aligned analyses imply negative/positive values around cue time | Derived from running texture-frame axis in raw behavior files | `[-51.3, 23.3]` | Yes |
| `day_of_training` range | Not directly reported in paper | Relative day index implied by longitudinal analysis | `0` to `92` from canonical session order per mouse | `[0.0, 92.0]` | Yes |
| `time_since_trial_start_sec` range | Not directly reported in paper | Trial-aligned framewise analyses from corridor entry to gray entry | Derived from retained running-frame intervals in raw behavior files | `[0.0, 56.0]` | Yes |
| `reward_available` range | Binary rewarded vs unrewarded corridors | Binary by experiment design and code logic | `0/1` from `WallType` mapping | `[0.0, 1.0]` | Yes |
| `visual_stimulus_category` distribution | Multiple wall textures including swap conditions; exact fractions not reported | Code distinguishes canonical and swap wall identities | Raw `WallName` labels span `15` categories in matched sessions | `circle1 0.252`, `circle2 0.049`, `circle3 0.010`, `leaf1 0.264`, `leaf1_swap1 0.019`, `leaf1_swap2 0.019`, `leaf2 0.128`, `leaf3 0.040`, `rock1 0.071`, `rock2 0.013`, `wood1 0.065`, `wood1_swap1 0.008`, `wood1_swap2 0.010`, `wood2 0.038`, `wood5 0.014` | Yes |
| `licking` distribution | Not directly reported as a frame fraction | Code uses framewise lick events from `LickFr` | Sparse lick events after running-only texture-frame filtering | `no_lick 0.963`, `lick 0.037` | Yes |
| `position_bin` distribution | Cue/reward positions approximately span the texture corridor | Texture-only analyses cover `0–4 m` | Near-uniform after equal-length binning of retained frames | `0-1m 0.249`, `1-2m 0.248`, `2-3m 0.249`, `3-4m 0.254` | Yes |
| `running_speed_bin` distribution | Not reported; quartile binning is a task-specific decoder requirement | Quartiles are consistent with the decoder spec, not the paper | Defined from all retained running speeds | `0.250/0.250/0.250/0.250` by construction | Yes |

Notes:
- `train_decoder.py --verify-only` reported: `Data format is valid, no errors or warnings.`
- Overall curated brain-region counts in the corrected converted dataset are `V1 169,829`, `mHV 134,418`, `lHV 35,593`, `aHV 17,488`, `other 0`.
- Step 10 resolved the main curation mismatch that affected the original full conversion. The corrected dataset preserves all `89` sessions and all `38,110` kept trials while bringing neuron selection into closer agreement with the reference reward-prediction code.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Verification log review:
   - The pre-fix `verification_full_out.txt` was clean (`Data format is valid, no errors or warnings.`), so the Step 10 issue was not a basic format or shape problem.
2. Raw-data sanity checks with `np.allclose()`:
   - Independent spot-checks were run directly from raw `Beh_*.npy`, `spk/*_neural_data.npy`, and `retinotopy/*_trans.npz` for `(session, kept-trial)` pairs `(0,0)`, `(34,10)`, and `(88,20)`.
   - These checks were rerun after the curation fix on the corrected `converted_data.pkl`.
   - For all three checks, `np.allclose()` passed for the full neural matrix, all four decoder inputs, all four decoder outputs, the brain-region index vector, and the session-to-subject mapping.
3. Reference code comparison:
   - `convert_data.py` was compared directly against `utils.Get_dprime_selective_neuron()` and `utils.get_kfold_reward_response()`.
   - This found two mismatches: the original converter used an extra gray-space response gate for generic stimulus selectivity, and its reward-neuron rule was looser than the reference code.
4. Key statistics comparison:
   - Converted-data statistics were recomputed directly from the corrected `converted_data.pkl`: `89` sessions, `19` subjects, `38,110` trials, `357,328` curated neurons, mean `4,014.92` neurons/session, licking fraction `0.03656`, and near-uniform position bins.
   - Raw-data statistics were recomputed independently from canonical sessions: `89` sessions, `19` subjects, `38,110` kept trials with at least one running frame, and `4,691,034` raw neurons (`52,708.25` mean/session; min `20,547`, max `89,577`).
5. Edge-case review:
   - Sessions with no finite aHV reward `d′` values are now handled explicitly by assigning an infinite reward threshold, preventing accidental over-selection.
   - The sample-selection guard from earlier steps still prevents degenerate all-zero reward/licking sample runs.

### Issues Found and Resolved
- Iteration 1:
  - Issue: reward-prediction neuron curation was looser than the reference code (`d′late_vs_early >= 0.3` instead of the paper-style aHV 95th-percentile rule plus cue-frame stimulus selectivity).
  - Resolution: patched `compute_neuron_selection()` to require cue-frame `dp_sound > 0.3` and `reward_dp >= percentile_95(aHV reward_dp)`, using running frames in the `0.5–4.0 m` texture segment as the raw-data approximation to the reference interpolated-position response.
  - Issue: stimulus-selective neuron curation used an extra gray-space response gate that belongs to the coding-direction analysis, not the generic d-prime selection function.
  - Resolution: removed the gray-space gate from the generic `|d′| >= 0.3` stimulus-selective pool.
  - Re-check: reran Steps 7-8 and Step 9 after the patch. `sample_data.pkl`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`, `converted_data.pkl`, `conversion_full_out.txt`, and `verification_full_out.txt` were regenerated successfully.
  - Re-check result: the corrected full dataset again passes `train_decoder.py --verify-only` with no errors or warnings, and post-fix raw-data `np.allclose()` spot-checks still pass on representative sessions/trials.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes
- Full run completed on `cuda` without needing `--cpu`.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| visual_stimulus_category | 0.4748 | 0.4709 | Strongly above chance (`0.0667`) |
| licking | 0.7790 | 0.7883 | Strongly above chance (`0.5000`) |
| position_bin | 0.3131 | 0.3127 | Above chance (`0.2500`), but only modestly |
| running_speed_bin | 0.2914 | 0.2927 | Above chance (`0.2500`), but only modestly |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| visual_stimulus_category | `0.4709` validation balanced accuracy | No directly comparable classifier accuracy in paper; strong visual discrimination is expected |
| licking | `0.7883` validation balanced accuracy | No directly comparable classifier accuracy in paper; lick-related signals should be decodable in task sessions |
| position_bin | `0.3127` validation balanced accuracy | No direct paper accuracy; moderate above-chance performance is plausible for this decoder-specific discretization |
| running_speed_bin | `0.2927` validation balanced accuracy | No direct paper accuracy; modest above-chance performance is plausible for this decoder-specific quartile target |

Analysis:
- Chance levels are `0.0667`, `0.5000`, `0.2500`, and `0.2500` for the four outputs respectively. All achieved validation accuracies are above chance.
- `visual_stimulus_category` and `licking` are comfortably above both chance and the `1.5x chance` heuristic.
- `position_bin` (`1.25x` chance) and `running_speed_bin` (`1.17x` chance) are below the `1.5x chance` heuristic, so they were investigated rather than accepted blindly.
- The investigation did not reveal a conversion bug:
  - raw-data `np.allclose()` spot-checks on three representative sessions/trials already verified exact reconstruction of both `position_bin` and `running_speed_bin` from the original files;
  - the full dataset distributions are sensible: position bins are nearly uniform, and speed bins are exactly quartiles by construction, so neither variable is degenerate;
  - the processing plots and time-variable spot-checks show no temporal lag between retained neural frames and decoder targets;
  - train/validation gaps are tiny for all outputs, so there is no sign of leakage or severe overfitting.
- The paper does not report balanced-accuracy results for this custom decoder task. Its closest analyses use d-prime selectivity, coding-direction similarity, and reward-prediction neurons rather than the same categorical targets used here. Because `position_bin` and especially `running_speed_bin` are decoder-specific reformulations imposed by this project, only modest above-chance performance is expected after restricting to running-only texture frames and paper-style visual-task neuron curation.

### Issues Found and Resolved
- No additional conversion issues were found after the Step 10 curation fix. The weaker decoder outputs (`position_bin`, `running_speed_bin`) remain modest but above chance, and their correctness is supported by the raw-data sanity checks rather than by decoder performance alone.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
- Generated plot artifacts (`processing_*.png`, `sample_trials.png`, `predictions.png`) were moved into `cache/`.
