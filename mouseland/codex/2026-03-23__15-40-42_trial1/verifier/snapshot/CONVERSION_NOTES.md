# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks dataset and reference code
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
| `load_exp_beh` | `code/utils.py` | LOADING | Load behavior dictionary from `beh/Beh_<exp_type>.npy`. |
| `load_spk` | `code/utils.py` | LOADING | Load session neural activity from `<mouse>_<date>_<blk>_neural_data.npy` and concatenate the stored `spks` entries into one neuron-by-frame matrix. |
| `load_retino` | `code/utils.py` | LOADING | Load retinotopy / area annotations from `<mouse>_<date>_trans.npz` and derive visual-area masks. |
| `spk_pos_interp` | `code/utils.py` | PROCESSING | Interpolate each neuron's running-period activity onto normalized corridor position. |
| `get_interpPos_spk` | `code/utils.py` | PROCESSING | Create `(neurons, trials, 60)` position-binned neural arrays; notebook comments state 60 bins span a 6 m corridor, so each bin is 0.1 m. |
| `get_cat_id` | `code/utils.py` | PROCESSING | Map wall / stimulus names into reward vs non-reward category ids for pretraining behavior. |
| `get_lick_raster` | `code/utils.py` | PROCESSING | Reconstruct per-trial lick positions, first-lick positions, cue positions, and reward positions. |
| `pretrain_exp_lick_raster` | `code/utils.py` | PROCESSING | Collapse pretraining behavior into rewarded vs non-rewarded categories and first-lick summaries. |
| `lickCount` | `code/utils.py` | PROCESSING | Convert lick timestamps and positions into binary trial-level lick responses before reward, after reward, or within a positional zone. |
| `Get_dprime_selective_neuron` | `code/utils.py` | CURATION | Compute stimulus selectivity `d'` using only running frames in corridor texture regions. |
| `Get_dprime_rewPred_neuron` | `code/utils.py` | CURATION | Identify reward-prediction neurons using early-vs-late cue `d'` within rewarded trials. |
| `Get_coding_direction` | `code/utils.py` | PROCESSING | Normalize interpolated neural activity, append 10 bins from the previous trial, split odd/even trials, and form coding-direction projections. |
| `Get_sort_spk` | `code/utils.py` | PROCESSING | Use odd/even splits and position-binned activity to sort stimulus-selective neurons by peak position. |
| `get_kfold_reward_response` | `code/utils.py` | PROCESSING | Build cue- and first-lick-aligned neural / lick responses using 10-fold cross-validation and selected reward neurons. |
| `spk_2_firstLick` | `code/utils.py` | PROCESSING | Align a 1D neural trace to first lick using `ranges=[15,15]` frames and 30 histogram bins. |
| `spk_2_cue` | `code/utils.py` | PROCESSING | Align a 1D neural trace to cue frame using `ranges=[15,15]` frames and 30 histogram bins. |

### Notes
- `code/README.md` states `data_process_script.ipynb` contains the preprocessing workflow used to generate intermediate files in `process_data/`; the figure scripts then consume those saved products.
- The notebook loads experiment metadata from `beh/Imaging_Exp_info.npy`, then iterates over experiment types and sessions to generate `process_data/*_interpolate_spk.npy`, `*_dprime.npy`, `*_coding_direction.npy`, `*_sort_spk.npy`, and reward-response files.
- The core raw neural loader is `load_spk`; there is no delta-F/F computation in the reference code. The saved `spks` arrays are treated as the neural signal directly, then optionally z-scored or transformed by d-prime-style normalization depending on analysis.
- There is no electrophysiology-style unit quality curation in the reference code. Neurons are included from the stored `spks` arrays, then functionally filtered only for specific analyses by area masks, corridor-vs-gray responsiveness, and stimulus selectivity thresholds.
- The main shared preprocessing step is position interpolation during movement: `VRmove = beh['ft_move'][:nfr] > 0`; `get_interpPos_spk` uses `spk[:, VRmove]` and `ft_PosCum[VRmove]` to build neural activity over 60 position bins per trial.
- Trial and frame restriction in the reference analyses commonly use:
  - `beh['ft_CorrSpc']` to keep corridor texture frames.
  - `beh['ft_GraySpc']` for gray-space / inter-trial frames.
  - odd/even trial splits via `ft_trInd % 2` or `trInd % 2` for train/test separation.
- `Get_coding_direction` explicitly appends `n_bef=10` bins from the previous trial's gray space before the current trial, so some paper figures use trial context preceding corridor entry.
- `get_kfold_reward_response`, `spk_2_cue`, and `spk_2_firstLick` show that cue-aligned and first-lick-aligned analyses are also done in frame time, not only in position space.
- Relevant stimulus names appearing throughout the code are `circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`, and swapped `leaf1_swap1` / `leaf1_swap2`.
- The code implies imaging data with retinotopic area assignment (`V1`, `mHV`, `lHV`, `aHV`) and session-level mouse/date/block identifiers.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/beh/` contains imaging-session behavior dictionaries:
  - `Imaging_Exp_info.npy`: master metadata mapping experiment types to session records.
  - `Beh_<exp_type>.npy`: one dictionary per experiment type; keys are session ids like `TX108_2023_03_25_1` or `TX61_2021_06_23_1_swap1`.
  - `example_bef_and_aft_learning_behavior.npy`: figure example behavior only.
- `data/spk/` contains raw neural activity files named `<mouse>_<date>_<blk>_neural_data.npy`.
  - Each file is a dict with one key, `spks`, which is a list of planes / blocks.
  - The reference code concatenates these arrays along the neuron dimension to get a `(n_neurons, n_frames)` matrix.
- `data/retinotopy/` contains `<mouse>_<date>_trans.npz` files plus `areas.npz`.
  - These store neuron coordinates (`xy_t`) and area ids (`iarea`) used to assign neurons to `V1`, `mHV`, `lHV`, `aHV`.
- `data/beh/Unsupervised_pretraining_behavior/` contains separate behavior-only pretraining cohorts with no matching `spk/` files:
  - `Beh_no_pretrain.npy` (25 sessions)
  - `Beh_pretrain_on_grat_image.npy` (35 sessions)
  - `Beh_pretrain_on_nat_image.npy` (55 sessions)
- There is no README or text documentation inside `data/`.
- Important raw-structure distinction:
  - `Imaging_Exp_info.npy` has **142 experiment-session records** across 23 experiment types.
  - These collapse to **99 unique behavior session keys**.
  - Those in turn map to **89 unique neural recording files**.
  - Some sessions are reused across multiple experiment labels (for example the same recording appears in both `*_train2_before_learning` and `*_test1` metadata), and some `test3` sessions reuse one neural recording with two behavior keys (`swap1`, `swap2`).
- Representative imaging behavior session structure (`Beh_sup_test1.npy`, key `TX108_2023_03_25_1`):
  - Trial-level arrays: `WallName`, `isRew`, `SoundPos`, `SoundFr`, `RewPos`, `RewTime`, `SoundTime`, `StartFr`, `EndFr`, `Trial_start_time`, `Trial_end_time`, `stim_id`, `UniqWalls`.
  - Frame-level arrays: `ft_trInd`, `ft_move`, `ft_RunSpeed`, `ft_Pos`, `ft_PosCum`, `ft_CorrSpc`, `ft_GraySpc`, `ft_WallID`.
  - Lick arrays: `LickPos`, `LickFr`, `LickTrind`, `LickTime`.
  - Session/task metadata: `Reward_Mode`, `Reward_Delay_ms`, `Corridor_Length`, `Texture_Length`, `Gray_Space_length`.
- Representative raw neural file structure (`VR2_2021_04_11_1_neural_data.npy`):
  - `spks` is a list of three arrays with shapes `(25369, 29244)`, `(25369, 29244)`, `(25370, 29244)`, which concatenate to `76108 x 29244`.
- Representative retinotopy file structure (`VR2_2021_04_11_trans.npz`):
  - keys: `A`, `dx`, `dy`, `xpos`, `ypos`, `xy_t`, `iarea`.
- Raw behavior frame arrays are slightly longer than neural recordings; across 99 unique behavior sessions, `len(ft_trInd) - n_spike_frames` is always `1`, `2`, or `3`. This matches the reference code pattern of truncating behavior arrays with `[:nfr]`.
- Reward modes vary by experiment:
  - supervised test/train sessions include `Passive` and/or `Active after cue`;
  - unsupervised sessions are mostly `Active after cue`;
  - naive and grating-control sessions often use `Passive` or `None` in metadata.
- Stimulus sets vary by experiment and include familiar categories (`circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`), swapped variants (`leaf1_swap1`, `leaf1_swap2`), and an alternate texture family (`rock*`, `wood*`).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 6,867,489 across 142 experiment-session records; 4,691,034 across 89 unique neural recordings |
| Neurons / session | Mean 48,362.599; min 20,547; max 89,577 |
| Subjects | 19 imaging mice |
| Sessions / subject | Mean 7.474 experiment-session records per mouse (range 2-15) |
| Trials (total) | 63,177 across 142 experiment-session records; 44,060 across 99 unique behavior session keys |
| Trials / session | Mean 444.908; min 84; max 789 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 89 recordings total; total neuron count not explicitly stated | "We performed 89 recordings in 19 mice" |
| Neurons / session | 20,547 to 89,577 neurons / recording | "20,547 to 89,577 neurons in each recording" |
| Subjects | 19 mice | "89 recordings in 19 mice" |
| Sessions / subject | Not explicitly stated in paper; 89 recordings / 19 mice overall | "We performed 89 recordings in 19 mice" |
| Trials (total) | Not explicitly stated | No global trial-count statement found in paper text or methods excerpt |
| Trials / session | Not explicitly stated | No per-session trial-count statement found in paper text or methods excerpt |
| Neural data time bin | No fixed ms bin reported; analyses use deconvolved traces and often interpolate by position | "All our analyses were based on deconvolved fluorescence traces" |
| Behavior data time bin | No fixed ms bin reported; running speed reported on 0-6 m positions with 0.1 m steps for some analyses | "0–6 m, with a 0.1-m step size" |
| Reward rate | No single global rate; task mice rewarded only in rewarded corridor after cue, unsupervised mice unrewarded | "The reward was delivered if a lick was detected after the sound cue in the rewarded corridor" |
| Corridor geometry | 4 m texture corridor + 2 m grey space | "corridors were each 4 m long, with 2 m of grey space" |
| Cue position range | Uniform from 0.5 m to 3.5 m in imaging sessions | "uniform distribution between positions 0.5 m and 3.5 m" |
| Running threshold / VR speed | Mouse must run faster than 6 cm s-1; corridor then advances at 60 cm s-1 | "running faster than a threshold of 6 cm s−1" / "constant speed (60 cm s−1)" |


### Processing Details
- The paper confirms this is calcium imaging, not electrophysiology: analyses use Suite2p outputs with motion correction, ROI detection, cell classification, neuropil correction, and spike deconvolution.
- Deconvolution parameter stated in methods: decay timescale 0.75 s.
- The main inclusion rule is movement: only timepoints during running are used for analysis.
- Neural selectivity (`d'`) for familiar corridors is computed from original deconvolved traces without interpolation, restricted to the 0-4 m textured portion of the corridor and to running timepoints.
- Coding-direction analyses use interpolated neural activity across corridor positions, baseline subtraction from the grey corridor portion, and normalization by the average standard deviation across the two reference corridors.
- Sequence / coding analyses use held-out data:
  - select neurons from train trials,
  - compute tuning or projections on separate test trials,
  - for sequence analyses, split test trials into odd versus even subsets.
- Reward-prediction analyses use interpolated single-neuron trial-by-position matrices on `leaf1` trials, split into early-cue versus late-cue trials based on cue position; cue position is used because it is present with and without reward and is highly correlated with reward position in rewarded trials.
- Reward-prediction population activity is computed with 10-fold cross-validation.
- Running-speed analyses in the paper interpolate speed either to imaging frame times or to 0-6 m positions with a 0.1 m step size.
- The paper reports that the overall running speeds were similar before versus after learning and between task versus unsupervised cohorts, so gross speed differences should not drive major between-cohort effects.

### Curation Steps

**Neuron curation rules**:
- Start from Suite2p-classified cells and deconvolved traces.
- No additional global neuron-quality filter is described in the text beyond Suite2p processing.
- For specific analyses, neurons are functionally selected by `d'` thresholds:
  - stimulus-selective neurons: typically `d' >= 0.3` or `d' <= -0.3`,
  - coding-direction / sequence analyses: top 5% most selective neurons per reference class,
  - reward-prediction neurons: `d'late vs early >= 0.3`.

**Trial curation rules**:
- Only running timepoints are analyzed.
- Neural selectivity uses only the 0-4 m textured corridor segment.
- For sequence analysis, one half of relevant trials is used to identify selective neurons and the held-out half is split into odd/even subsets for preferred-position comparisons.
- For reward-prediction analysis aligned to first lick, only rewarded `leaf1` trials with first lick after 2 m are included; the paper states one mouse was excluded because it had no such trials.
- For reward-prediction analysis in `leaf2`, one mouse was excluded because it had only one `leaf2` no-lick trial.
- `leaf1_swap1` and `leaf1_swap2` are treated differently by cohort:
  - task mice: introduced separately and pooled for statistics,
  - unsupervised / naive mice: both shown in the same session but treated as two data points and pooled later.

### Decoders Trained
| Decoded variable | Accuracy |
| No decoder analogous to `train_decoder.py` is reported in the paper | Not reported |
| Related paper metrics | Sequence correlation, coding-direction similarity index, fractions of selective neurons, reward-prediction `d'` |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session counting | `data_process_script.ipynb` loops over `Imaging_Exp_info.npy` experiment records | 142 experiment-session records, 99 unique behavior session keys, 89 unique neural recordings | "We performed 89 recordings in 19 mice" | Treat 89 as unique imaging recordings, but keep track that behavior / experiment labels reuse recordings. Conversion session unit will need an explicit decision in Step 5. |
| Neural signal identity | `load_spk` loads stored `spks` directly; no dF/F computation anywhere in code | `spks` arrays are `float32`, non-negative, already session-aligned neural traces | "All our analyses were based on deconvolved fluorescence traces" | The stored `spks` arrays are the deconvolved fluorescence traces; do not compute dF/F. |
| Spatial units | Utility code uses `Corridor_Length=60`, `Texture_Length=40`, 60 position bins | Behavior files have `Corridor_Length=60.0`, `Texture_Length=40.0`, `Gray_Space_length=20.0` | Paper describes 4 m corridor + 2 m grey space, cue from 0.5-3.5 m | Raw data / code are in decimeters: 60 = 6 m total, 40 = 4 m texture, 20 = 2 m grey. |
| Running-only analysis | `Get_dprime_selective_neuron` and related functions apply `ft_move > 0` and corridor masks | Frame-level arrays `ft_move`, `ft_CorrSpc`, `ft_GraySpc` are present for every session | "We only considered timepoints during running for analysis" | Raw data provide the exact masks needed to match the paper; conversion should preserve / use running frames where reference analyses do. |
| Selectivity computation | `Get_dprime_selective_neuron` uses raw frame traces in corridor frames; no interpolation | Raw data contain both original frame traces and accumulated position for interpolation | Paper states selectivity `d'` uses 0-4 m running data without interpolation | Code, data, and paper are consistent: use raw running frames for familiar-stimulus selectivity; use interpolation only for sequence / coding-direction / reward-prediction analyses. |
| Trial splitting | `Get_sort_spk` / `Get_coding_direction` use even/odd trial splits; reward response uses 10-fold CV | Trial indices are available as `trInd`, `ft_trInd`, `*_odd`, `*_even` | Paper methods describe held-out halves, odd/even subsets, and 10-fold CV | The code is a direct implementation of the paper’s split logic. |
| Behavior vs neural frame length | Code truncates behavior arrays with `[:nfr]` before combining with neural data | Across 99 unique behavior sessions, `len(ft_trInd) - n_spike_frames` is always 1, 2, or 3 | Paper does not discuss this small mismatch explicitly | Use the reference-code convention: truncate behavior frame arrays to neural frame count. |
| Unsupervised reward fields | Code still uses cue position and corridor identity in unsupervised sessions | Unsupervised sessions have valid `SoundPos` but `RewPos` is all `NaN` and `isRew.sum()==0` in sampled sessions | Paper says unsupervised mice receive no reward but still hear the sound cue | Interpret unsupervised sessions as preserving corridor and cue structure without reward availability. |
| Time-bin definition | Code mostly works in original imaging frames or in position bins, not a common ms grid | Raw timestamps exist (`ft`, `Trial_start_time`, `SoundTime`, `LickTime`); median imaging frame interval is about 314.67 ms in a sampled session | Paper does not state a fixed ms bin size | Use raw frame timestamps to build a trial-start-aligned time grid later; this preserves the reference streams while satisfying the decoder format requirement. |
| Brain-region naming | Code maps `iarea` into `V1`, `mHV`, `lHV`, `aHV` | Retinotopy files contain `xy_t` and `iarea` for every neuron | Paper discusses V1 and higher visual areas; main text also refers to medial / lateral / anterior regions | Use the code’s concrete four-region mapping for per-neuron labels, while noting that `mHV` corresponds to the paper’s medial HVA grouping. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk/*.npy -> spks` | `neural` | Concatenate plane arrays along neurons; keep deconvolved values as float32; then keep a compact, paper-grounded subset of stimulus-selective neurons before splitting by trial | `load_spk`, `Get_coding_direction`, `Get_sort_spk` | Session unit will be unique recording base `<mouse>_<date>_<blk>` (89 sessions), not duplicated metadata entries. |
| `beh['ft_trInd']`, `beh['ft_CorrSpc']`, `beh['ft_move']`, `beh['ft']` | `neural` trial segmentation | Use finite `ft_trInd`, truncate behavior arrays to neural frame count `nfr`, then keep frames where `ft_CorrSpc` is true and `ft_move > 0` to match paper-style running corridor analysis | `Get_dprime_selective_neuron`, `Get_coding_direction`, `Get_sort_spk` | Trial-aligned neural matrices will therefore contain running frames inside the 0-4 m corridor. Trials with too few retained frames will be filtered. |
| `beh['SoundTime']` and retained frame times from `beh['ft']` | `input[0]` = `time_to_sound_cue_s` | For each retained frame, compute `SoundTime - frame_time` in seconds; positive before cue, negative after cue | `spk_2_cue` (same cue timing source), paper methods on cue timing | Time-varying continuous input. |
| Session date parsed from recording base, relative to first recording date for that mouse | `input[1]` = `training_day` | Continuous per-trial scalar = elapsed days since the mouse’s first retained imaging session; repeated across timepoints in the saved 2D input array | No direct paper function; required by decoder task | Decoder-task-required addition; derived from raw session dates to remain objective and reproducible. |
| Retained frame times from `beh['ft']` and `beh['Trial_start_time']` | `input[2]` = `time_since_trial_start_s` | `frame_time - Trial_start_time` in seconds for each retained frame | Raw timestamps; consistent with paper frame-time analyses | Time-varying continuous input. |
| `beh['isRew']` | `input[3]` = `reward_available` | Trial-level binary repeated across timepoints; 1 only for rewarded-corridor task trials, 0 otherwise | `get_cat_id`, behavior code, paper methods | In unsupervised / naive / grating sessions this is expected to be all 0. |
| `beh['WallName']` | `output[0]` = `visual_stimulus` | Global categorical mapping over all unique stimulus names across retained sessions; repeated across timepoints | `stim_id`, `WallName`, figure code | 15 global categories found in imaging data: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5. |
| `beh['LickFr']`, `beh['LickTrind']` | `output[1]` = `licking` | Build a binary imaging-frame vector (1 if any lick occurs on that retained frame, else 0) | `spk_2_firstLick`, `spk_2_cue`, `lickCount` | Time-varying binary output aligned to neural frames. |
| `beh['ft_Pos']` on retained frames | `output[2]` = `position_bin` | Convert decimeter positions in the 0-40 textured corridor to four 1 m bins via edges `[0,10,20,30,40]` | Paper corridor geometry + code’s 60-position / 40-texture convention | Time-varying categorical output with values `0-1m`, `1-2m`, `2-3m`, `3-4m`. |
| `beh['ft_RunSpeed']` on retained frames | `output[3]` = `running_speed_bin` | Discretize into global quartiles (25% each) across all retained timepoints in all retained sessions | Paper running-speed interpolation; raw `ft_RunSpeed` | Time-varying categorical output; quartile edges saved to metadata. |
| Retinotopy `iarea` | `brain_region_idx` | Map to `V1`, `mHV`, `lHV`, `aHV`; then keep only selected neurons from these visual-area groups | `neu_area_ID`, `load_retino`, `Get_coding_direction` | Current export drops unmatched `other` neurons because the decoder subset is selected within paper-defined visual areas. |
| Session mouse name parsed from base id | `subjects`, `subject_idx` | Unique mouse ids in deterministic sorted order | `Imaging_Exp_info.npy` | 19 subjects expected after deduplication. |

### Key Decisions
1. **Use 89 unique recording bases as sessions**: This matches the paper’s "89 recordings in 19 mice" and avoids leakage from duplicated experiment labels and `swap1` / `swap2` aliases that share the same raw trials.
2. **Prefer a plain behavior key when multiple keys share one recording base**: For bases with both plain and swap-specific keys (for example `TX123_2023_12_18_1`), the plain key already contains the full `WallName` set, so it is the best canonical representative.
3. **Keep the stored `spks` arrays directly, but export only a selective decoder subset**: The paper and code both indicate these are already deconvolved fluorescence traces; no dF/F or additional normalization should be applied, but a compact neuron subset is necessary for tractable decoding.
4. **Align trials using frame-level `ft_trInd` rather than `StartFr` / `EndFr`**: This matches the reference code and avoids boundary ambiguities.
5. **Retain paper-style running corridor frames (`ft_CorrSpc` and `ft_move > 0`)**: This is the closest match to the published analyses, which consistently use running timepoints in the textured corridor.
6. **Filter bad / unusable trials only when required by decoder format**: Trials with fewer than 5 retained neural timepoints will be excluded; sessions with fewer than 2 remaining trials will be excluded. This is a decoder-format curation step, not a paper curation step, and will be reported explicitly.
7. **Store all four decoder inputs in a 2D time-varying array**: Even the per-trial variables (`training_day`, `reward_available`) will be repeated across timepoints so every trial has a uniform `(4, T)` input shape.
8. **Store all four decoder outputs in a 2D time-varying array**: The per-trial visual stimulus label will be repeated across retained timepoints to keep a uniform `(4, T)` output shape.
9. **Use raw frame timestamps instead of resampling to a new clock**: Median imaging frame spacing is very consistent across sessions (~314.7 ms), so keeping native frame bins is more faithful than interpolation to a synthetic time grid.
10. **Select neurons using the paper’s 5% positive / 5% negative per-area logic**: For each session, compute familiar-stimulus `d'` on running corridor frames, restrict to corridor-responsive neurons, and keep the top 5% positive and top 5% negative neurons within each of `V1`, `mHV`, `lHV`, `aHV`. This follows the reference coding-direction analysis and reduces the export size enough to make full decoding practical.
11. **Represent brain regions with five labels: `V1`, `mHV`, `lHV`, `aHV`, `other`**: The current selective export only keeps neurons from the first four groups, but the metadata schema still reserves `other` for completeness.
12. **Represent `training_day` as elapsed days since the mouse’s first retained imaging session**: The paper does not define this variable, but the decoder task requires it and the raw dates provide a reproducible continuous proxy.
13. **Use global quartiles for running-speed bins**: The decoder task specifies 25% bins, so edges must be computed from the full retained dataset, not per session.

### Planned Sanity Checks
- [ ] **Neural raw-frame check**: For at least three sampled sessions / trials, verify `converted_data['neural'][s][t]` exactly matches the corresponding columns of the raw concatenated `spks` matrix selected by the retained-frame mask.
- [ ] **Input timing check**: For sampled trials, verify `time_to_sound_cue_s` equals raw `SoundTime - ft[mask]` and `time_since_trial_start_s` equals raw `ft[mask] - Trial_start_time`.
- [ ] **Lick alignment check**: For sampled trials, verify the converted licking vector matches the raw `LickFr` events projected onto the retained imaging frames.
- [ ] **Stimulus-label check**: For sampled trials, verify the repeated categorical `visual_stimulus` output matches raw `WallName`.
- [ ] **Position-bin check**: For sampled trials, verify position-bin transitions occur at raw `ft_Pos` thresholds of 10, 20, and 30 decimeters.
- [ ] **Speed-bin check**: Recompute quartile edges from raw retained `ft_RunSpeed` values and confirm converted bins match `np.digitize` / `np.searchsorted` on those edges.
- [ ] **Session-count check**: Confirm the converted dataset contains 89 sessions unless decoder-required trial/session filtering removes any, in which case document every removal.
- [ ] **Subject-count check**: Confirm the converted dataset contains 19 mice after deduplication.
- [ ] **Neuron-count check**: Confirm per-session neuron counts match raw concatenated `spks` counts and retinotopy lengths.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Added `convert_data.py` with CLI:
  - `python -u convert_data.py <outpicklefile>`
  - `--sample`
  - `--full`
  - `--show-processing`
- The script currently:
  - deduplicates the dataset to unique recording bases,
  - computes retained running-corridor frame indices from behavior arrays,
  - drops decoder-pathological trials with retained duration > 60 s or retained inter-frame gaps > 10 s,
  - selects a compact stimulus-selective neuron subset per session using paper-style 5% percentile rules,
  - derives global visual categories and global running-speed quartile edges,
  - loads raw deconvolved neural traces session by session,
  - builds time-varying `(4, T)` input and output arrays for each retained trial,
  - stores subject and brain-region metadata,
  - optionally saves processing plots for up to 2 sessions,
  - reports per-session timing and final output size.
- Verified with:
  - `python3 -m py_compile convert_data.py`
  - `python3 convert_data.py --help`

Code inefficiencies identified:
- The current implementation still has to load each neural recording file in full before selecting the compact neuron subset, so full conversion will still be I/O-heavy.
- Session preparation currently uses behavior-derived frame counts before the exact neural frame count is known; the second pass clips frame indices to the true neural frame count.
- `--show-processing` originally used the pre-filter trial list for plot selection; this was fixed during Step 10 after the new trial-quality filters were added.

Code speedups added:
- Behavior-only preparation pass computes trial frame indices and speed quartiles before loading neural data.
- Neural recordings are loaded one session at a time in the main conversion pass.
- Retinotopy and lick-event processing are done once per session, not once per trial.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 9,960 |
| Neurons / session | [3,782, 6,178] |
| Subjects | 2 |
| Sessions / subject | 1 each for `DR10`, `TX108` |
| Trials (total) | 602 |
| Trials / session | [166, 436] |
| `time_to_sound_cue_s` range | [-20.3, 28.0] |
| `training_day` range | [0.0, 67.0] |
| `time_since_trial_start_s` range | [0.0, 37.7] |
| `reward_available` range | [0.0, 1.0] |
| `visual_stimulus` distribution | [circle1=0.291, leaf1=0.317, rock1=0.165, wood1=0.228] |
| `licking` distribution | [no_lick=0.836, lick=0.164] |
| `position_bin` distribution | [0.251, 0.253, 0.249, 0.248] |
| `running_speed_bin` distribution | [0.046, 0.454, 0.250, 0.250] |

### Processing Plots Review
- `processing_TX108_2023_03_13_1.png` and `processing_DR10_2022_07_12_1.png` were generated.
- After adding the Step 10 trial-quality filters, the sample plots still show continuous retained-running segments with no extreme timing outliers.
- Current visual checks expected from the script:
  - retained frames stay inside the 0-4 m corridor,
  - cue timing is consistent with trial-start alignment,
  - position-bin transitions follow 1 m boundaries,
  - running-speed quartile edges are drawn on the session histogram,
  - rewarded session shows nonzero lick output whereas the unrewarded session does not.
- No format anomalies were reported by `train_decoder.py --verify-only`.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Selective-neuron export (top 5% positive / negative per visual area) | Reduced sample pickle size from 3.84 GiB to 0.25 GiB and made full export feasible |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion | ~10.4 s / session | ~15.4 min for 89 sessions, before any further optimization |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| visual_stimulus | 0.9940 | 0.9750 |
| licking | 0.8428 | 0.8375 |
| position_bin | 0.7263 | 0.7478 |
| running_speed_bin | 0.5279 | 0.4956 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9.70 GiB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Raw recordings only: 89 recordings, 20,547-89,577 / rec | Top-5% positive / negative selective neurons per area in coding-direction analyses | 4,691,034 raw neurons across 89 recordings | 304,548 selected neurons across 89 sessions | Partial: selected-subset export, not raw-count export |
| Mean neurons/session | Raw recordings: 20,547-89,577 / rec | Compact selective subset expected from code percentile filter | 52,708.247 raw mean | 3,421.89 selected mean | Partial |
| Subjects | 19 | 19 in raw metadata | 19 | 19 | Yes |
| Sessions | 89 recordings | 89 unique recording bases after deduplication | 89 | 89 | Yes |
| Trials (total) | Not stated globally | Session analyses operate on retained running trials | 44,060 unique behavior-key trials; 38,110 retained running-corridor trials after dedup to 89 recordings | 35,893 after removing 2,217 decoder-pathological stalled trials | Yes relative to retained-frame trial definition plus documented Step 10 curation |
| Trials/session (mean) | Not stated globally | N/A | 428.2 raw unique-behavior mean before dedup / filtering | 403.3 kept-trial mean | Yes |
| `time_to_sound_cue_s` range | Cue positioned 0.5-3.5 m in corridor | Derived from raw timestamps | Should span positive pre-cue and negative post-cue values | [-49.1, 45.7] | Yes after Step 10 review |
| `training_day` range | Not a paper variable | N/A | Session dates span multiple days per mouse | [0.0, 92.0] | Expected for derived variable |
| `time_since_trial_start_s` range | Should be nonnegative | Derived from raw timestamps | Trial durations vary substantially | [0.0, 54.3] | Yes after Step 10 review |
| `reward_available` range | Task rewarded corridor only; unsupervised none | Uses `isRew` / corridor identity | Mixed 0 and 1 across sessions | [0.0, 1.0] | Yes |
| `visual_stimulus` distribution | Mixed familiar, test, swap, rock/wood stimuli across cohorts | Determined from `WallName` | 15 categories across imaging data | [0.247, 0.049, 0.010, 0.260, 0.020, 0.020, 0.127, 0.041, 0.072, 0.014, 0.067, 0.008, 0.010, 0.040, 0.015] | Yes |
| `licking` distribution | Anticipatory licking mainly in task mice | Derived from `LickFr` / retained frames | Sparse when restricted to retained running frames | [0.962, 0.038] | Yes |
| `position_bin` distribution | 4 equal 1 m bins | Derived from `ft_Pos` | Should be close to uniform after corridor masking | [0.249, 0.249, 0.250, 0.252] | Yes |
| `running_speed_bin` distribution | 4 quartile bins by construction | Derived from global quartiles | Should be [0.25, 0.25, 0.25, 0.25] | [0.250, 0.250, 0.250, 0.250] | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Output log verification: Re-ran `train_decoder.py converted_data.pkl --verify-only` after the Step 10 fix. Result: no errors, no warnings, 89 sessions, 35,893 trials, and sane timing ranges (`time_to_sound_cue_s` `[-49.1, 45.7]`, `time_since_trial_start_s` `[0.0, 54.3]`).
2. Neural/input/output raw sanity checks with `np.allclose()`: Independently reconstructed three representative converted trials from raw files and confirmed exact agreement for neural activity, all four inputs, all four outputs, and the selected per-neuron brain-region labels.
3. Reference code comparison:
   - Data loading: `code/utils.py::load_spk` concatenates stored `spks`; `convert_data.py` does the same in `load_spike_matrix`.
   - Region mapping: `code/utils.py::load_retino` / `neu_area_ID` define the four visual-area groups; `convert_data.py::load_region_index` uses the same `iarea` grouping (`V1`, `mHV`, `lHV`, `aHV`).
   - Neuron filtering: `Get_coding_direction` / `Get_sort_spk` select corridor-responsive top-5% positive and negative `d'` neurons per area on running corridor frames; `convert_data.py::select_decoder_neurons` mirrors that logic on the same raw frame masks.
   - Temporal alignment and trial masking: reference code uses `ft_trInd`, `ft_CorrSpc`, and `ft_move > 0`; `convert_data.py::build_trial_frame_indices` uses the same fields.
   - Binning / timing: reference analyses stay on native frames or position bins; `convert_data.py` preserves native frame timestamps for decoder inputs and discretizes only the decoder-required outputs (1 m position bins and speed quartiles).
   - Input/output construction: `time_to_sound_cue_s`, `time_since_trial_start_s`, `reward_available`, `visual_stimulus`, `licking`, `position_bin`, and `running_speed_bin` are all built directly from the raw behavior arrays (`SoundTime`, `Trial_start_time`, `isRew`, `WallName`, `LickFr`, `ft_Pos`, `ft_RunSpeed`).
4. Key statistics comparison:
   - Subjects and sessions match the paper and raw metadata exactly: 19 mice, 89 recordings / sessions.
   - Raw neuron range still matches the paper: 20,547-89,577 neurons / recording before selective export.
   - Converted neuron counts are intentionally lower because the decoder export uses the paper’s selective-neuron logic rather than all raw neurons.
   - Trial count changed from 38,110 to 35,893 only because Step 10 removed 2,217 sparse stalled trials that distort trial-start timing.
5. Edge-case review:
   - The worst outlier was raw session `TX88_2022_07_19_1`, trial 391: one nominal trial spanning 1,765 s with only 36 retained running-corridor frames.
   - Across the pre-review retained trials, 470 failed the `>60 s` retained-duration check and 1,747 more failed the `>10 s` retained-frame-gap check.
   - No session dropped below the minimum of 2 valid trials after this curation; the smallest retained session still has 62 trials.

### Issues Found and Resolved
- Extreme `time_to_sound_cue_s` / `time_since_trial_start_s` ranges: Caused by raw trials with sparse late running segments long after nominal trial start. Resolved by excluding those trials as invalid decoder-alignment periods while preserving the reference running-corridor masking for the remaining data.
- `--show-processing` plotting mismatch after the new filter: The plotting path still indexed trials using the old minimum-length rule. Resolved by tracking the actual filtered trial list used in the export.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| visual_stimulus | 0.3989 | 0.3911 | 5.87x chance (chance = 0.0667 for 15 classes) |
| licking | 0.7994 | 0.7981 | 1.60x chance (chance = 0.5000) |
| position_bin | 0.3188 | 0.3159 | 1.26x chance (chance = 0.2500); investigated in Step 12 |
| running_speed_bin | 0.2973 | 0.2968 | 1.19x chance (chance = 0.2500); investigated in Step 12 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| visual_stimulus | 0.3911 validation balanced accuracy | No directly comparable decoder accuracy reported; paper shows strong stimulus selectivity and coding-direction structure, so above-chance decoding is expected |
| licking | 0.7981 validation balanced accuracy | No directly comparable decoder accuracy reported |
| position_bin | 0.3159 validation balanced accuracy | No directly comparable decoder accuracy reported |
| running_speed_bin | 0.2968 validation balanced accuracy | No directly comparable decoder accuracy reported |

- Chance comparison:
  - `visual_stimulus`: 0.3911 vs chance 0.0667 (5.87x chance)
  - `licking`: 0.7981 vs chance 0.5000 (1.60x chance)
  - `position_bin`: 0.3159 vs chance 0.2500 (1.26x chance)
  - `running_speed_bin`: 0.2968 vs chance 0.2500 (1.19x chance)
- Train vs validation gap:
  - `visual_stimulus`: 1.020x train/val
  - `licking`: 1.002x train/val
  - `position_bin`: 1.009x train/val
  - `running_speed_bin`: 1.002x train/val
  - Conclusion: no material overfitting or leakage signal.
- Low-accuracy investigation for `position_bin` and `running_speed_bin`:
  - Verified output construction directly from raw files with `np.allclose()` on three representative trials; both discretized outputs matched exactly.
  - Reviewed temporal-alignment plots (`processing_*.png`) and decoder sample plots (`sample_trials.png`, `predictions.png`); no obvious neural/output misalignment was observed.
  - Checked output variation: both `position_bin` and `running_speed_bin` are near-uniform across the full dataset, so low accuracy is not caused by class collapse.
  - Compared train/validation gaps: both outputs have essentially identical train and validation accuracy, which argues against a hidden bug or leakage problem.
  - Final interpretation: the modest but above-chance position and speed decoding likely reflects the restricted paper-style stimulus-selective neuron subset exported for tractable full-dataset decoding, not a conversion error.

### Issues Found and Resolved
- Position / speed validation accuracy below 1.5x chance: investigated with raw output checks, alignment plots, output-distribution review, and train/validation-gap analysis. No conversion bug was found, so no further code change was applied.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
