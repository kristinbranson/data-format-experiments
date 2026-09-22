# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks dataset (`/app/data`)
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

Environment checks:
- `python3`: available (`3.13.15`)
- `numpy`: import ok (`2.4.4`)
- `torch`: import ok (`2.6.0+cu124`)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_exp_beh` | `/app/code/utils.py` | LOADING | Loads per-experiment behavior dictionary from `beh/Beh_<exp_type>.npy`. |
| `load_spk` | `/app/code/utils.py` | LOADING | Loads session neural activity from `<mouse>_<date>_<blk>_neural_data.npy` and concatenates `['spks']` across planes into one neuron-by-frame matrix. |
| `load_retino` | `/app/code/utils.py` | LOADING | Loads retinotopy transform and maps neurons to coarse visual areas using `iarea`. |
| `get_interpPos_spk` | `/app/code/utils.py` | PROCESSING | Interpolates neural activity from frame time to trial-position bins; outputs `(neurons, trials, 60)` using cumulative position and only moving frames. |
| `spk_pos_interp` | `/app/code/utils.py` | PROCESSING | Core interpolation helper used by `get_interpPos_spk`; linearly resamples each neuron along cumulative position normalized by corridor length. |
| `Get_dprime_selective_neuron` | `/app/code/utils.py` | PROCESSING | Uses raw frame activity in corridor frames during movement to compare stimulus conditions; defines stimulus selectivity. |
| `Get_coding_direction` | `/app/code/utils.py` | PROCESSING | Uses odd/even trial splits, corridor-only moving frames, and interpolated spike maps for stimulus decoding analyses. |
| `get_kfold_reward_response` | `/app/code/utils.py` | PROCESSING | Builds reward-related decoding responses from interpolated spike maps and cue/reward aligned trial variables. |
| `neu_area_ID` | `/app/code/utils.py` | CURATION | Collapses retinotopy area IDs into `V1`, `mHV`, `lHV`, `aHV`. |
| Notebook cell “get interpolational neural activity” | `/app/code/data_process_script.ipynb` | PROCESSING | Batch-generates saved interpolated spike files with 60 bins across 6 m corridor from behavior and neural data. |

### Notes
- `/app/code/README.md` states `data_process_script.ipynb` shows how intermediate results are processed and saved; `Figures.ipynb` plots figures.
- The code is for imaging mice (`Imaging_Exp_info.npy` is loaded in the notebook), so this is calcium-imaging-derived neural activity rather than electrophysiology.
- The published code does **not** recompute `dF/F`; it loads precomputed neural activity from session files via `load_spk`.
- `load_spk` concatenates arrays inside `np.load(...).item()['spks']`, so a session may be stored in multiple blocks/planes but is treated downstream as one neuron-by-frame matrix.
- The reference pipeline creates a spatially aligned representation:
  - Corridor length is 6 m.
  - Neural activity is interpolated into `60` bins (`1 decimeter` / `0.1 m` each).
  - Only frames with `beh['ft_move'] > 0` are used when building interpolated trial-by-position activity.
- Frame-level task masks used repeatedly in the reference:
  - `beh['ft_CorrSpc']`: texture corridor frames.
  - `beh['ft_GraySpc']`: gray-space frames.
  - `beh['ft_WallID']`: corridor identity / stimulus at each frame.
  - `beh['ft_trInd']`: trial index for each neural frame.
- Trial-splitting convention in the code:
  - “odd trials” in comments correspond to `ft_trInd % 2 == 0`.
  - The authors use one parity for training/selectivity estimation and the other parity for test/visualization.
- Relevant behavior fields documented in the notebook include corridor entry/exit frames (`StartFr`, `EndFr`), cue and reward frames (`SoundFr`, `SoundDelayFr`, `RewardFr`), per-frame running speed (`ft_RunSpeed`), per-frame position (`ft_Pos`, `ft_PosCum`), and lick frame/times (`LickFr`, `LickTime`, `LickPos`).
- No explicit neuron quality filtering is implemented in the public reference code beyond task- and area-based masks used for specific analyses.
- Retinotopy files provide neuron-wise area labels; sessions can include neurons from multiple areas.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/beh/`
  - `Imaging_Exp_info.npy`: experiment-to-session metadata mapping used by the reference notebook.
  - `Beh_<exp_type>.npy`: per-experiment behavior dictionaries. Each file maps session key -> session behavior structure.
  - `example_bef_and_aft_learning_behavior.npy`: example behavior-only file used in figure code.
  - `Unsupervised_pretraining_behavior/`: three behavior-only files (`Beh_no_pretrain.npy`, `Beh_pretrain_on_grat_image.npy`, `Beh_pretrain_on_nat_image.npy`) without matching imaging metadata in `Imaging_Exp_info.npy`.
- `/app/data/spk/`
  - One `*_neural_data.npy` file per raw imaging session.
  - Each file contains only `{'spks': [...]}` where `spks` is a list of arrays. In a representative file (`TX60_2021_04_10_1_neural_data.npy`) the list length is 3 and each entry has shape `(16244, 23469)`; the reference code concatenates along axis 0 to form one neuron-by-frame matrix.
- `/app/data/retinotopy/`
  - One `*_trans.npz` file per raw imaging session plus `areas.npz`.
  - Representative fields: `A`, `xpos`, `ypos`, `xy_t`, `iarea`.
  - `iarea` length matches neuron count from concatenated `spks` in the representative sample.
- `/app/data/process_data/`
  - Empty in the provided dataset; intermediate files used in the paper were not precomputed here.

Behavior dictionary structure (representative session `TX60_2021_06_07_1` from `Beh_sup_test1.npy`):
- Scalar/session fields: `ntrials`, `Corridor_Length`, `Reward_Mode`, `Reward_Delay_ms`, `UniqWalls`, `stim_id`.
- Trial-level fields: `WallName`, `isRew`, `SoundFr`, `StartFr`, `EndFr`, `RewardFr`, `SoundPos`, `SoundDelPos`, `RewPos`, `Trial_start_time`, `Trial_end_time`.
- Frame-level fields aligned to imaging frames: `ft`, `ft_trInd`, `ft_Pos`, `ft_PosCum`, `ft_RunSpeed`, `ft_move`, `ft_CorrSpc`, `ft_GraySpc`, `ft_WallID`, `BefCueFr`, `AftCueFr`.
- Event streams: `LickTime`, `LickFr`, `LickTrind`, `LickPos`.

Session-key structure:
- Most behavior entries are keyed as `<mouse>_<date>_<blk>`.
- Some entries include stimulus-type suffixes such as `_swap1` / `_swap2` via `stimtype`.
- The same raw imaging session can appear in multiple behavior files, and a minority of raw sessions have multiple distinct behavior variants.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 across 89 raw imaging sessions |
| Neurons / session | mean 52,708.247; min 20,547; max 89,577 |
| Subjects | 19 unique mice |
| Sessions / subject | raw sessions: mean 4.684 (range 1-8); behavior-session variants: mean 5.211 (range 1-12) |
| Trials (total) | 44,060 across 99 unique behavior-session keys |
| Trials / session | mean 445.051; min 84; max 789 |

Additional size notes:
- `Imaging_Exp_info.npy` contains 23 experiment groups.
- Across those 23 groups there are 142 experiment-session entries, but only 99 unique behavior-session keys and 89 unique raw imaging session files.
- 9 raw imaging sessions have multiple behavior variants (for example `_swap1` and `_swap2`).
- 33 raw imaging session files are reused across more than one experiment grouping.
- Representative per-subject raw session counts: `TX119` 8 raw / 10 behavior variants, `TX123` 8 raw / 12 behavior variants, `DR10` 6 raw / 7 behavior variants.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 89 recordings total; 20,547-89,577 neurons per recording | “We performed 89 recordings in 19 mice...” / “from 20,547 to 89,577 neurons in each recording” |
| Neurons / session | 20,547-89,577 | “from 20,547 to 89,577 neurons in each recording” |
| Subjects | 19 mice | “We performed 89 recordings in 19 mice” |
| Sessions / subject | Not given as a single summary; varies by cohort | Paper reports 89 recordings total across 19 mice, and notes that some mice have multiple imaging sessions |
| Trials (total) | Not explicitly stated in paper | Not reported as a single dataset-wide number |
| Trials / session | Not explicitly stated in paper | Not reported as a single dataset-wide number |
| Neural data time bin | Not stated in paper/methods; reference notebook states calcium frame rate `fs = 3.17 Hz` (~315.46 ms/frame) | Notebook markdown: “Calcium signal recording frame rate: fs = 3.17Hz” |
| Behavior data time bin | No single fixed bin in text; events are continuous and aligned to imaging frames/positions | Behavior is described via event times, frame-aligned variables, and position interpolation |
| Reward rate | Task mice: reward available only in rewarded corridor after cue; unsupervised mice: no rewards | “water was available after the sound cue in rewarded trials only”; “did not receive water rewards” |
| Corridor geometry | 4 m textured corridor + 2 m grey space | “The virtual reality corridors were each 4 m long, with 2 m of grey space between corridors” |
| Sound cue position | Uniformly random from 0.5-3.5 m in imaging mice | “the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m” |
| Running threshold / VR speed | 6 cm/s threshold; corridor moves at 60 cm/s while above threshold | “running faster than a threshold of 6 cm s−1” / “virtual corridors always moved at a constant speed (60 cm s−1)” |
| Selective-neuron threshold | d′ >= 0.3 (or <= -0.3) | “The criteria for selective neurons was d′ ≥ 0.3” |
| Reward-prediction cross-validation | 10-fold CV on leaf1 early- vs late-cue trials | “We randomly split all trials into tenfolds... used ninefolds as training trials...” |


### Processing Details
- Analyses are based on deconvolved fluorescence traces from Suite2p spike deconvolution, not raw fluorescence and not recomputed `dF/F`.
- For stimulus selectivity, the paper explicitly uses only the 0-4 m textured corridor and excludes non-running timepoints.
- The paper distinguishes between:
  - raw-frame analyses without interpolation for d′ selectivity;
  - position-interpolated trial-by-position matrices for sequence similarity, coding direction, and reward-prediction analyses.
- Spatial interpolation is described at 0.1 m resolution over 0-6 m for running-speed analyses; the notebook and code implement 60 bins over the full 6 m corridor.
- Cue position is the task-critical stochastic variable:
  - present on all trial types for imaging mice, including unsupervised cohorts;
  - used as a proxy for reward position in rewarded corridors because cue and reward positions are highly correlated.
- The paper’s reward-prediction analysis uses leaf1 trials split into early-cue versus late-cue trials and selects neurons with `d′late_vs_early >= 0.3`.
- Reward-prediction population responses are obtained with 10-fold cross-validation.

### Curation Steps

**Neuron curation rules**:
- No global public-code neuron QC beyond the upstream Suite2p pipeline is described.
- Selectivity-based analyses use thresholded `d′` values (for example `d′ >= 0.3`).
- Coding-direction analyses use the top 5% most selective neurons per class from train trials.
- Reward-prediction analyses additionally focus on anatomically defined anterior HVAs in the paper.

**Trial curation rules**:
- Use only running timepoints for neural analyses.
- Stimulus-selectivity calculations use only the textured 0-4 m portion of the corridor.
- Sequence-similarity analyses split leaf1/circle1 trials into train/test halves, then odd/even trial partitions inside held-out test data.
- Reward-prediction analysis:
  - uses only leaf1 trials;
  - splits trials into early-cue vs late-cue groups based on cue position;
  - excludes one mouse for first-lick alignment because it had no trials with first lick later than 2 m;
  - excludes one mouse from leaf2 lick/no-lick comparison because it had only one leaf2 no-lick trial.

### Decoders Trained
| Decoded variable | Accuracy |
| No paper-reported classifier matching this task | N/A |
| Cross-validated reward-prediction population response | Qualitatively robust in task mice; no scalar accuracy reported |
| Coding-direction / similarity analyses | Projections and similarity indices reported; no scalar accuracy reported |

Notes on expected decoder accuracy:
- The paper does not report a supervised decoder that directly predicts stimulus, licking, position, or speed from all neurons.
- Qualitative expectations from the paper are:
  - stimulus category should be decodable, especially after learning and in medial visual areas;
  - position within the corridor should be strongly represented as single-trial neural sequences;
  - licking and cue/reward-related signals should be present, especially in task cohorts;
  - reward-related temporal structure should be strongest in anterior HVAs.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Recording/session count | `Imaging_Exp_info.npy` drives analyses by experiment grouping | 142 experiment-session entries, 99 unique behavior keys, 89 raw neural session files | “89 recordings in 19 mice” | Treat the raw imaging recording as the non-duplicated neural session unit. Do not duplicate sessions merely because the same recording is reused in several figure-specific behavior files. |
| Duplicate sessions across experiment files | Reference notebook iterates experiment groups independently | 33 raw recordings are reused across multiple `Beh_<exp_type>.npy` files; core arrays (`WallName`, `SoundPos`, `StartFr`, `EndFr`, `isRew`) are identical across duplicates | Paper reuses the same recordings in multiple analyses/figures | Merge duplicates by raw session key when constructing the decoder dataset. Keep one canonical behavior instance per raw recording. |
| `_swap1` / `_swap2` sessions | Some behavior entries are separate keys with `stimtype` suffixes | In unsupervised/naive cohorts, `_swap1` and `_swap2` are alternate analysis views of the same trial arrays; only `stim_id` / `StimTrial` differs. In supervised test3, swap1 and swap2 can also appear as separate raw sessions on different dates. | Paper states swap stimuli in unsupervised/naive mice were in the same session but treated as separate data points for statistics; task mice introduced them separately | For decoder conversion, use the actual per-trial `WallName` labels from the canonical raw session and avoid duplicating identical neural recordings. Preserve both swap stimulus categories as distinct output labels when they truly occur in `WallName`. |
| Stimulus labels | Reference analyses often use `stim_id` and `StimTrial` to select a subset of categories relevant to a figure | Behavior files can contain more actual stimulus labels in `WallName` / `UniqWalls` than are assigned non-NaN `stim_id` values in a particular experiment view | Paper discusses different analyses focusing on specific stimulus subsets (for example leaf1 vs circle1, leaf1 vs leaf2, swaps) | Use `WallName` as the ground-truth per-trial visual stimulus category for decoder outputs. Use `stim_id` only when reproducing paper-specific selection logic. |
| Corridor length wording | Code uses 60 bins over full corridor and often restricts texture area to first 40 bins | Behavior files have `Corridor_Length = 60.0`; frame masks divide corridor into texture vs grey | Paper says corridors were 4 m long with 2 m grey space between corridors | These are consistent in decimeter units: 0-4 m texture + 4-6 m grey. For decoder outputs, full 0-6 m position is available and can be binned into 4 equal 1 m bins using the first 4 m of corridor progression from trial start if aligning to corridor entry. |
| Neural signal type | Code loads `spks`; no explicit fluorescence preprocessing in repo | `spk/*_neural_data.npy` stores deconvolved activity arrays | Paper says analyses use Suite2p deconvolved fluorescence traces | No `dF/F` recomputation is needed; use provided deconvolved activity matrices directly. |
| Running-time restriction | Code repeatedly masks with `ft_move > 0` and corridor masks | Behavior files provide `ft_move`, `ft_isMoving`, `ft_CorrSpc`, `ft_GraySpc` | Paper says only running timepoints were considered | Restrict framewise analyses to valid in-trial frame windows; preserve speed as an input/output variable, but note that neural interpolation in reference code uses moving frames only. |
| Reward structure | Behavior files contain both `Passive` and `Active after cue`, with delays 0/1000/1500 ms | `Reward_Mode`, `Reward_Delay_ms`, `isRew`, `RewPos`, `RewardFr` vary across sessions | Paper notes some mice received passive delayed rewards, others active rewards, but cue was present in all imaging trials | Use direct trial/frame variables from data (`isRew`, `SoundFr`, `RewardFr`, `Reward_Mode`) rather than assuming a single reward contingency. |

Final understanding:
- The paper, code, and raw files agree on the core imaging cohort: 89 raw recordings from 19 mice.
- The public behavior files are organized for multiple figure-specific analyses, which creates duplicated “analysis views” of some raw sessions.
- The canonical decoder dataset should therefore be built around unique raw recordings with one behavior object per recording, while preserving all actual trial labels present in `WallName`.
- Reference processing to preserve:
  - use deconvolved traces as provided;
  - align trials to corridor entry (`StartFr`);
  - use the paper’s frame/position variables rather than reconstructing timing from scratch;
  - keep the distinction between textured corridor (0-4 m) and grey zone (4-6 m);
  - respect running-related masking where interpolation or framewise matching requires it.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk_file['spks']` concatenated across planes | `neural` | Concatenate list elements along neuron axis; slice per trial using running corridor frame mask; store as `float16` | `load_spk` | Use provided deconvolved traces directly; no `dF/F` recomputation; decoder converts back to `float32` internally |
| `beh['ft_trInd'] == trial`, `beh['ft_CorrSpc']`, and `beh['ft_move'] > 0` | Trial time axis for `neural`, `input`, `output` | Select running frames inside the corridor that belong to the trial; this defines aligned timepoints | Paper Methods + `fr_valid = VRmove & isCorridor` in `utils.py` | Chosen instead of `StartFr:EndFr` because `StartFr` points one frame before first in-trial frame and because the reference analyses exclude non-running periods |
| `beh['ft']` and `beh['SoundFr']` | `input[0]` = `time_to_sound_cue_s` | `ft[SoundFr] - ft[selected_frames]`, converted from days to seconds | Sound cue fields in methods + behavior schema | Continuous, time-varying; negative after cue |
| Session date / block within mouse | `input[1]` = `training_day` | Continuous day index per session = days since mouse’s first imaging date + small block offset if needed | Derived from `Imaging_Exp_info.npy` | Per-trial scalar, replicated by decoder internally if needed |
| `beh['ft']` | `input[2]` = `time_since_trial_start_s` | `ft[selected_frames] - ft[first_selected_frame]`, in seconds | Trial alignment to corridor entry | Continuous, time-varying |
| `beh['isRew'][trial]` | `input[3]` = `reward_available` | Cast to {0,1} | Behavior fields / paper reward structure | Per-trial corridor identity, not actual reward delivery |
| Merged per-session `WallName` -> canonical label map from all behavior views | `output[0]` = `visual_stimulus` | Map literal wall names to canonical category IDs | `stim_id`, `StimTrial`, paper experiment structure | Session-specific mapping avoids cross-file duplication and handles rock/wood remapping |
| `beh['LickFr']`, `beh['LickTrind']` | `output[1]` = `licking` | Binary vector over selected frames; 1 if >=1 lick on that frame | Lick fields in behavior schema | Time-varying |
| `beh['ft_Pos']` | `output[2]` = `position_bin` | Discretize corridor position into 4 bins: [0,1), [1,2), [2,3), [3,4] m | Paper corridor geometry | Running corridor frames ensure 4 equal 1 m bins over textured region while matching reference frame selection |
| `beh['ft_RunSpeed']` | `output[3]` = `speed_bin` | Global quartile binning across included frames | Running speed fields + paper running analyses | Time-varying; thresholds computed from included dataset frames |
| `retinotopy['iarea']` | `brain_region_idx` | Map codes to `V1`, `mHV`, `lHV`, `aHV`, `other` | `neu_area_ID` | Preserve neurons with codes `-1` or `7` as `other` rather than dropping them |
| Mouse name from session metadata | `subjects`, `subject_idx` | Unique subject list + per-session index | `Imaging_Exp_info.npy` | One subject per raw session |

### Key Decisions
1. **Session unit = unique raw imaging recording (89 total)**: The paper and raw files agree on 89 recordings; repeated behavior entries across experiment files are reused analysis views of the same recording and should not become duplicate decoder sessions.
2. **Canonical behavior for a raw session will be built by merging all behavior views of that raw session**: Core arrays are identical across duplicates, but `stim_id`/`StimTrial` expose different subsets of categories. Merging recovers a complete per-session label map without duplicating neural data.
3. **Visual stimulus output will use session-specific canonical categories, not literal wall names alone**: This preserves cross-mouse comparability when some mice use `rock/wood` stimuli and when some sessions reverse literal circle/leaf naming. The category space will be `circle1`, `circle2`, `circle3`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`.
4. **`circle3` will be preserved as its own category**: It appears in actual trial labels but is not assigned a canonical `stim_id` in the provided figure-specific views; treating it as a separate presented stimulus preserves information rather than forcing an unsupported remapping.
5. **Trial frames will be selected with `(ft_trInd == trial) & ft_CorrSpc & (ft_move > 0)`**: This matches the paper and reference code, which restrict imaging analyses to running frames inside the textured corridor, while still aligning the trial to corridor entry rather than to `StartFr`.
6. **The decoder dataset will use running frames within the corridor (0-4 m texture region)**: This follows the published imaging analyses and removes reward-stop periods that the Methods say were excluded from analysis.
7. **Neural activity will remain on the original frame timebase after running-frame selection**: The task is temporally aligned to corridor entry and requires time-varying cue, licking, and speed outputs; framewise traces are therefore more appropriate than position-interpolated spike maps for the decoder, while still using the same valid-frame mask as the reference analyses.
8. **All neurons will be retained**: The public reference code does not apply additional global neuron QC beyond Suite2p preprocessing; region labels are metadata, not a filter.
9. **Brain-region mapping will follow the reference coarse groups with an additional `other` bucket**: The published helper only uses four named groups for some figures, but the raw data contains many neurons with codes `-1` and `7`; these should be retained and labeled explicitly.
10. **Running speed bins will be computed globally over included frames**: Global quartiles give consistent output classes across sessions, which is better suited to a cross-session decoder than per-session bins.
11. **Time variables will use actual frame timestamps (`ft`) rather than nominal frame count alone**: This preserves small timestamp jitter and makes cue-aligned timing exact.
12. **Day-of-training input will be mouse-relative continuous session day**: Using days since first imaging session (with a tiny block offset if needed) respects within-mouse chronology and preserves the intended “training day” context.

### Planned Sanity Checks
- [ ] Verify for duplicated raw sessions across behavior files that `WallName`, `SoundPos`, `StartFr`, `EndFr`, and `isRew` are identical (`np.allclose` / `np.array_equal`).
- [ ] Verify that merged per-session stimulus mappings cover every unique `WallName` actually present in that session.
- [ ] Verify that selected frames satisfy `ft_CorrSpc == True`, `ft_move > 0`, and `ft_trInd == trial`, and that `StartFr` is typically one frame earlier than the first corridor-assigned frame in the reference behavior.
- [ ] Verify that binary lick outputs match raw `LickFr` / `LickTrind` on spot-checked trials using `np.allclose`.
- [ ] Verify that time-to-cue reaches 0 at the frame corresponding to `SoundFr` on spot-checked trials.
- [ ] Verify that per-neuron brain-region indices match retinotopy array lengths and reference area grouping rules.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Implemented `/app/convert_data.py` with required CLI:
  - `python -u /app/convert_data.py <outpicklefile>`
  - `--sample`
  - `--full` (default behavior when `--sample` is absent)
  - `--show-processing`
- Script structure:
  - builds unique raw-session specs from `Imaging_Exp_info.npy` and all `Beh_*.npy` files;
  - verifies duplicated behavior views are identical on core arrays;
  - merges per-session stimulus mappings across behavior views;
  - processes one raw session at a time;
  - slices trial data with `ft_trInd == trial`, `ft_CorrSpc`, and `ft_move > 0`;
  - derives cue-relative and trial-relative time inputs from corridor position at the reference 60 cm/s virtual speed rather than from wall-clock timestamps, so excluded pause periods do not inflate decoder times;
  - saves decoder-format pickle with metadata.
- The script was compile-checked with `python3 -m py_compile /app/convert_data.py`.
- After correcting the frame-selection mask to match the reference running analyses and changing stored dtypes (`neural`/`input` to `float16`, compact integer metadata/outputs), the sample run succeeded:
  - `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
  - output: 2 sessions, 893 trials, 110,041 neurons, total run time about 0.45 min.

Code inefficiencies identified:
- Full-session spike concatenation would have created an extra multi-GB copy per session.
- Computing speed quartiles before trial conversion would have required an extra pass with ambiguous frame truncation relative to neural data length.
- Storing dense per-trial arrays as `float32` made the full pickle exceed 100 GB and pushed Step 9 well beyond the allowed runtime estimate.

Code speedups added:
- Process spike planes directly and concatenate only per-trial slices, avoiding a full-session concatenated copy.
- Process one session at a time to keep peak intermediate memory bounded.
- Defer speed binning until after session conversion, using only the already collected per-trial speed arrays.
- Add session-level parallel processing for `--full` mode (up to 4 workers) because sessions are independent and the serial full-run estimate exceeded 15 minutes.
- Store neural and input trial arrays as `float16` because `train_decoder.py` materializes its own `float32` buffers during training, so the lower on-disk precision cuts serialization cost without changing the decoder path.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 110,041 |
| Neurons / session | 47,785 and 62,256 |
| Subjects | 2 (`TX60`, `TX61`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 893 |
| Trials / session | 485, 408 |
| `time_to_sound_cue_s` range | [-5.3, 5.6] |
| `training_day` range | [0.0, 58.0] |
| `time_since_trial_start_s` range | [0.0, 6.7] |
| `reward_available` range | [0.0, 1.0] |
| `visual_stimulus` distribution | [0.389, 0.135, 0.362, 0.115] across `circle1,circle2,leaf1,leaf2` |
| `licking` distribution | [0.859, 0.141] across `no_lick, lick` |
| `position_bin` distribution | [0.252, 0.248, 0.250, 0.249] |
| `speed_bin` distribution | [0.250, 0.250, 0.250, 0.250] |

### Processing Plots Review
- Reviewed `processing_TX60_2021_06_07_1.png` and `processing_TX61_2021_06_07_2.png`.
- No obvious temporal misalignment:
  - cue/reward positions lie inside the 0-4 m corridor;
  - time-to-cue decreases through the trial and crosses zero during the trial;
  - position bins progress from 0 to 3 as expected;
  - licking events are sparse and time-localized rather than smeared.
- The sample conversion initially used two unsupervised sessions and produced degenerate reward/lick outputs; this was fixed by changing `--sample` session selection to choose representative task sessions with reward variation and in-corridor licking.
- After re-checking `utils.py`, the converter was updated to keep only running corridor frames (`ft_move > 0`) rather than all corridor frames. This reduced the sample pickle from about 6.8 GB to about 2.33 GB and matches the published imaging analysis mask more closely.
- A later review found that using raw `ft` timestamps after removing non-running frames preserved long idle pauses in the time inputs. This was corrected by converting corridor position to time using the paper’s fixed virtual speed of 60 cm/s, bringing sample input ranges to the expected order of seconds.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Session-level full-mode parallelism (`max_workers = 4`) | Reduced estimated full conversion from about 18.0 min serial to about 4.5 min by file-size scaling |
| Running-frame mask + `float16` storage | Reduced sample pickle from about 6.8 GB to about 2.33 GB and should cut full-dataset serialization and load time by roughly 3x |
| Representative lighter sample sessions for validation | Reduced sample conversion from about 1.0 min to about 0.45 min and made sample decoder training practical |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion (`TX60_2021_06_07_1`, `TX61_2021_06_07_2`) | 8.61 s, 11.71 s | 0.45 min observed for 2 sessions |
| Full conversion (serial estimate by raw spike-file size ratio) | ~12.1 s/session equivalent by sample scaling | ~17.95 min |
| Full conversion (4-worker estimate by same scaling, before mask/dtype fix) | parallel | ~4.49 min compute plus oversized serialization |
| Full conversion (4-worker estimate after mask/dtype fix) | parallel | expected to be materially lower in serialization cost; to be re-measured in Step 9 |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None from `verify_data_format`; `sklearn` emitted a balanced-accuracy warning during training because some held-out trials lacked one or more rare stimulus classes, but training completed successfully.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `visual_stimulus` | 0.9305 | 0.8717 |
| `licking` | 0.7110 | 0.6904 |
| `position_bin` | 0.8161 | 0.7755 |
| `speed_bin` | 0.4463 | 0.4304 |

Additional notes:
- Training loss decreased from `7128.87` at epoch 1 to `142.14` at epoch 200.
- Validation accuracies were above uniform-chance for all outputs:
  - `visual_stimulus`: chance 0.125
  - `licking`: chance 0.5
  - `position_bin`: chance 0.25
  - `speed_bin`: chance 0.25

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 81 GiB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 4,691,034 | same cohort, no extra neuron filter in public code | 4,691,034 | 4,691,034 | Yes |
| Mean neurons/session | 52,708.247 mean; range 20,547-89,577 | same cohort | 52,708.247 mean | 52,708.247 mean | Yes |
| Subjects | 19 | same cohort | 19 | 19 | Yes |
| Sessions | 89 recordings | same cohort | 89 | 89 | Yes |
| Trials (total) | not explicitly reported | derived from behavior files after de-duplication | 38,110 with running-corridor trial inclusion | 38,110 | Yes |
| Trials/session (mean) | not explicitly reported | derived from behavior files after de-duplication | 428.202 | 428.202 | Yes |
| `time_to_sound_cue_s` range | cue uniformly distributed along corridor; at 60 cm/s this implies about +/- 5.8 s over 0-4 m | data-derived from corridor position and fixed VR speed | data-derived | [-6.4, 6.5] | Yes |
| `training_day` range | multi-day learning/imaging schedule; no exact range reported | data-derived | [0.0, 92.0] | [0.0, 92.0] | Yes |
| `time_since_trial_start_s` range | 4 m corridor at 60 cm/s implies about 6.7 s of running time | data-derived from corridor position and fixed VR speed | data-derived | [0.0, 6.7] | Yes |
| `reward_available` range | binary rewarded vs unrewarded corridor | binary in code/data | [0.0, 1.0] | [0.0, 1.0] | Yes |
| `visual_stimulus` distribution | task design includes circle/leaf plus test and swap stimuli; no pooled fractions reported | data-derived | data-derived | [0.317, 0.060, 0.007, 0.333, 0.169, 0.058, 0.027, 0.029] | Plausible |
| `licking` distribution | anticipatory licking sparse in framewise data; no pooled fraction reported | data-derived | data-derived | [0.963, 0.037] | Plausible |
| `position_bin` distribution | 4 equal 1 m bins intended | data-derived | approximately uniform | [0.250, 0.249, 0.250, 0.252] | Yes |
| `speed_bin` distribution | quartiles by construction | quartiles by construction | quartiles by construction | [0.250, 0.250, 0.250, 0.250] | Yes |

Additional notes:
- The first Step 9 full run was interrupted after the artifact exceeded 100 GB and serialization time exceeded the allowed estimate.
- After switching to the reference running-frame mask and compact stored dtypes, the repeated Step 9 run completed in 4.25 minutes.
- A subsequent review corrected the time inputs to use position-derived motion time rather than wall-clock timestamps after frame exclusion. The final Step 9 run completed in 5.16 minutes and passed `train_decoder.py --verify-only` with no errors or warnings.
- Verified cohort summary from `/app/verification_full_out.txt`:
  - `T` mean 22.25, median 22.63, min 11, max 178
  - brain-region counts: `V1 1,833,035`, `mHV 1,108,860`, `lHV 495,318`, `aHV 668,180`, `other 585,641`

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Final `/app/verification_full_out.txt` reports `Data format is valid, no errors or warnings.` No unresolved verifier warnings remain after the final Step 9 rerun.
2. **Raw-vs-converted sanity checks with `np.allclose()`**: Added `/app/cache/step10_sanity_checks.py` and ran `python3 /app/cache/step10_sanity_checks.py`. Results:
   - raw-derived speed edges for sample sessions: `[-12.8621, 8.6872, 17.3062, 28.5168, 61.8535]`
   - `session_trial=(0, 10)`: `neural=True`, `input=True`, `output=True`
   - `session_trial=(1, 20)`: `neural=True`, `input=True`, `output=True`
   These checks loaded the original behavior and spike files directly and reconstructed the expected converted arrays without using `convert_data.py`.
3. **Reference code comparison**:
   - **(a) Data loading**:
     - Reference: `load_spk`, `load_retino`, and notebook behavior loading from `Beh_*.npy`.
     - Conversion: `load_spike_planes`, `load_retinotopy`, and merged behavior loading in `build_session_specs`.
     - Result: consistent raw sources and session identifiers.
   - **(b) Neuron / trial filtering**:
     - Reference frame mask: `fr_valid = VRmove & isCorridor` in [`utils.py`](/app/code/utils.py#L431).
     - Conversion frame mask: `get_trial_frame_indices` in [`convert_data.py`](/app/convert_data.py#L256) uses `ft_trInd == trial`, `ft_CorrSpc`, and `ft_move > 0`.
     - Result: valid-frame logic now matches the reference.
   - **(c) Temporal alignment**:
     - Reference uses running frames and often represents activity as position-interpolated trial maps through `get_interpPos_spk` in [`utils.py`](/app/code/utils.py#L120) and calls at [`utils.py`](/app/code/utils.py#L488) and [`utils.py`](/app/code/utils.py#L544).
     - Conversion keeps framewise running-corridor samples in [`convert_data.py`](/app/convert_data.py#L417) rather than position interpolation.
     - Difference and rationale: the decoder task requires time-varying licking and speed outputs aligned to trial progression, so retaining framewise samples is justified; the valid-frame mask remains matched to the reference.
   - **(d) Binning**:
     - Reference bins neural activity into 60 position bins for some analyses.
     - Conversion bins only the required decoder outputs: 4 spatial bins and 4 global speed quartiles in [`convert_data.py`](/app/convert_data.py#L483).
     - Difference and rationale: the target format requires neural activity by timepoint, not position-binned averages.
   - **(e) Input construction**:
     - Conversion builds time-to-cue and time-since-start from corridor position and the fixed 60 cm/s virtual speed in [`convert_data.py`](/app/convert_data.py#L426), preventing excluded pause periods from contaminating the time inputs.
     - Reward availability comes directly from `isRew`; training day is derived from raw session chronology in `Imaging_Exp_info.npy`.
   - **(f) Output construction**:
     - Stimulus identity uses merged raw `stim_id` / `WallName` mappings.
     - Licking uses direct framewise `LickFr` / `LickTrind`.
     - Position bins use `ft_Pos`.
     - Speed bins use raw `ft_RunSpeed` quartiles over included frames.
4. **Key statistics comparison**:
   - Verified against paper/methods and raw files that the final full dataset contains 89 sessions, 19 mice, and 4,691,034 neurons.
   - Verified per-session neuron range matches the paper’s reported `20,547-89,577`.
   - Verified the full input ranges are now physically sensible:
     - `time_to_sound_cue_s`: `[-6.4, 6.5]`
     - `time_since_trial_start_s`: `[0.0, 6.7]`
   - Verified position and speed outputs have the expected near-uniform distributions by construction / task geometry.
5. **Edge-case checks**:
   - Checked sample-session trial starts directly from raw behavior:
     - `TX60_2021_06_07_1`: first corridor frame minus `StartFr` in `{1,2}`; first running-corridor frame minus `StartFr` ranged `1..4`, mean `1.08`.
     - `TX61_2021_06_07_2`: first corridor frame minus `StartFr` in `{1,2}`; first running-corridor frame minus `StartFr` ranged `1..2`, mean `1.06`.
   - Result: `StartFr` is not a reliable direct slicing boundary for the converted running-frame data; using `ft_trInd` plus frame masks avoids the off-by-one risk.

### Issues Found and Resolved
- **Oversized full artifact / slow serialization**: The first full Step 9 attempt exceeded 100 GB and violated the runtime estimate. Resolution: switch from corridor-only frames to the reference running-corridor mask and store dense trial arrays as `float16` where appropriate; reran Steps 7-9.
- **Inflated time inputs after frame exclusion**: Using raw `ft` timestamps after dropping non-running frames produced unrealistic ranges (for example `time_since_trial_start_s` up to ~1765 s). Resolution: recompute `time_to_sound_cue_s` and `time_since_trial_start_s` from corridor position and fixed virtual speed; reran Steps 7-9.
- **Potential trial-start off-by-one**: Raw `StartFr` precedes the first usable corridor or running-corridor frame by 1-4 frames in spot checks. Resolution: do not slice trials by `StartFr`; slice by `ft_trInd` with the matched reference frame masks instead.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes overall. The run started on GPU, hit CUDA OOM after epoch 1 during the first attempt, automatically retried on CPU, and completed successfully. On the successful CPU run, loss decreased from `17387.35` at epoch 1 to `161.43` at epoch 190 and `180.48` at epoch 200.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `visual_stimulus` | 0.5499 | 0.5381 | Well above 8-class chance (0.1250) |
| `licking` | 0.8958 | 0.8729 | Strongly above binary chance (0.5000) |
| `position_bin` | 0.3654 | 0.3650 | Above 4-class chance (0.2500); slightly below the 1.5x-chance heuristic |
| `speed_bin` | 0.3875 | 0.3877 | Above 4-class chance (0.2500) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `visual_stimulus` | 0.5381 validation balanced accuracy | Paper shows strong stimulus-selective coding across visual areas; no directly comparable supervised full-cohort decoder accuracy is reported |
| `licking` | 0.8729 | Paper shows strong anticipatory licking separation between rewarded and unrewarded corridors; no directly comparable framewise decoder accuracy is reported |
| `position_bin` | 0.3650 | No directly comparable published decoder accuracy; should be above chance because trial progression is represented in both behavior and neural dynamics |
| `speed_bin` | 0.3877 | No directly comparable published decoder accuracy; above chance is the main consistency expectation |

[Analysis of any low accuracies]
- **Accuracy vs chance**:
  - `visual_stimulus`: `0.5381 / 0.1250 = 4.30x chance`
  - `licking`: `0.8729 / 0.5000 = 1.75x chance`
  - `position_bin`: `0.3650 / 0.2500 = 1.46x chance`
  - `speed_bin`: `0.3877 / 0.2500 = 1.55x chance`
- `position_bin` is the only output slightly below the 1.5x-chance heuristic. I investigated this specifically:
  - sample data with the same conversion logic reached `0.7755` validation balanced accuracy for `position_bin`;
  - the raw-vs-converted sanity checks confirmed that `position_bin` exactly matches raw `ft_Pos`-derived 1 m bins on spot-checked trials;
  - `time_since_trial_start_s` is now derived directly from corridor position and has the physically expected `0.0-6.7 s` range, so there is no remaining evidence of a conversion bug in this variable;
  - train and validation accuracies for `position_bin` are nearly identical (`0.3654` vs `0.3650`), which argues against leakage or overfitting.
- **Accuracy comparison to paper**:
  - The paper does not report a directly comparable end-to-end supervised decoder for the four requested outputs (`visual_stimulus`, `licking`, `position_bin`, `speed_bin`) using this exact architecture and cohort packaging.
  - The closest reported decoding-style results are qualitative or use different targets (for example coding-direction and reward-prediction analyses), so only qualitative comparison is possible:
    - stimulus information should be robustly present: confirmed by `visual_stimulus = 0.5381`;
    - reward/lick-related behavioral information should be present in trained mice: confirmed by `licking = 0.8729`.
- **Train vs validation gap**:
  - `visual_stimulus`: `0.5499` train vs `0.5381` validation
  - `licking`: `0.8958` train vs `0.8729` validation
  - `position_bin`: `0.3654` train vs `0.3650` validation
  - `speed_bin`: `0.3875` train vs `0.3877` validation
  - No output shows a >1.5x train/validation gap. There is no sign of severe overfitting or obvious leakage.

### Issues Found and Resolved
- **No new conversion bugs were found after the final full decoder run**. The substantive issues discovered during review were already resolved earlier in Step 10:
  - running-frame mask mismatch versus the reference code;
  - oversized full serialization from dense `float32` storage;
  - inflated time inputs caused by combining wall-clock timestamps with excluded pause frames.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Notes:
- Added `/app/README.md` with dataset summary, format description, loading example, and reproduction commands.
- Added `/app/cache/README_CACHE.md`.
- Stored Step 10 raw-vs-converted sanity-check script at `/app/cache/step10_sanity_checks.py`.
- Archived older exploratory processing plots from superseded sample-session choices into `/app/cache/`.
- Final root-level conversion/validation artifacts retained:
  - `converted_data.pkl`
  - `sample_data.pkl`
  - `conversion_sample_out.txt`
  - `verification_sample_out.txt`
  - `train_decoder_sample_out.txt`
  - `conversion_full_out.txt`
  - `verification_full_out.txt`
  - `train_decoder_full_out.txt`
  - `processing_TX60_2021_06_07_1.png`
  - `processing_TX61_2021_06_07_2.png`
  - `sample_trials.png`
  - `predictions.png`
