# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
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

Environment checks:
- `python3` runs successfully
- `numpy` import succeeds
- `torch` import succeeds

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk` | `code/utils.py` | LOADING | Loads one session of neural data from `<mouse>_<date>_<blk>_neural_data.npy` and concatenates `spks` across sublists into a single neuron x frame matrix. |
| `load_exp_beh` | `code/utils.py` | LOADING | Loads one behavior dictionary file `Beh_<exp_type>.npy`. |
| `load_retino` | `code/utils.py` | LOADING | Loads retinotopy / area assignments from `<mouse>_<date>_trans.npz` and converts area IDs into named masks (`V1`, `mHV`, `lHV`, `aHV`). |
| `get_interpPos_spk` | `code/utils.py` | PROCESSING | Interpolates running-period neural activity into trial x position bins; notebook states 60 bins spanning the 6 m corridor (1 decimeter bins). |
| `Get_dprime_selective_neuron` | `code/utils.py` | PROCESSING | Computes stimulus selectivity `d'` using raw neural frames restricted to running frames inside the corridor texture zone. |
| `Get_dprime_rewPred_neuron` | `code/utils.py` | PROCESSING | Computes reward-prediction selectivity from interpolated activity, using mean activity in corridor bins 5:40 and early-vs-late cue trials. |
| `Get_coding_direction` | `code/utils.py` | PROCESSING | Splits trials into train/test by odd/even index, builds normalized interpolated population activity, appends previous-trial gray bins, and projects onto stimulus-selective populations. |
| `Get_sort_spk` | `code/utils.py` | PROCESSING | Uses train/test trial splits plus interpolated neural activity to sort selective neurons by peak position within the corridor. |
| `get_stimNeu_and_sorted` | `code/utils.py` | PROCESSING | Builds example stimulus-selective neuron responses from train/test splits. |
| `get_kfold_reward_response` | `code/utils.py` | PROCESSING | Performs 10-fold reward-response analysis, aligning neural activity to cue and first lick after selecting reward-prediction neurons. |
| `get_reward_neuorns` | `code/utils.py` | PROCESSING | Extracts example reward-prediction neurons from a session. |
| `get_cat_id` / `get_lick_raster` / `lickCount` / `lick_response` | `code/utils.py` | PROCESSING | Defines reward vs non-reward stimulus categories and behavioral licking summaries from raw lick timestamps / positions. |
| `data_process_script.ipynb` main interpolation loop | `code/data_process_script.ipynb` | PROCESSING | Iterates over experiment groups from `Imaging_Exp_info.npy`, loads `Beh_<exp_type>.npy`, filters to running frames, and saves per-session interpolated neural activity under `process_data/`. |

### Notes
- The reference code is for calcium imaging data, not electrophysiology. There is no delta-F-over-F computation in the provided code; neural activity is loaded from saved `spks` arrays.
- The notebook explicitly states imaging frame rate `fs = 3.17 Hz`.
- The notebook documents behavior/frame alignment fields including `StartFr`, `EndFr`, `SoundFr`, `SoundDelayFr`, `RewardFr`, `LickFr`, `ft_trInd`, `ft_WallID`, `ft_PosCum`, `ft_move`, `ft_CorrSpc`, and `ft_GraySpc`.
- Position-normalized neural activity is central to the reference analyses: `get_interpPos_spk` resamples neural traces onto 60 position bins across a 6 m corridor, using only frames where `ft_move > 0`.
- Stimulus selectivity in the reference code is computed from raw frame activity restricted to running frames in the corridor (`ft_CorrSpc & (ft_move > 0)`), not from trial averages.
- Several analyses split trials by parity: frames/trials with `ft_trInd % 2 == 0` are labeled “odd trials” in comments and used for training / neuron selection; the complementary trials are used for testing.
- Neuron curation in the provided code is task-specific rather than a single global quality filter: analyses often require neurons to be in named visual areas and to respond more in corridor than gray space (`corr_neu` criterion).
- Brain-region mapping is determined through retinotopy area IDs with grouped labels `V1`, `mHV`, `lHV`, `aHV`.
- Figure loaders indicate the paper draws from multiple experiment groups (`sup_*`, `unsup_*`, `naive_*`, `test*`, pretraining behavior cohorts), but the shared low-level loading/processing functions above are the core reference logic to match.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/beh/Imaging_Exp_info.npy`: master session index for imaging experiments; 23 experiment groups referencing 142 experiment entries but only 89 unique imaging recordings.
- `data/beh/Beh_<exp_type>.npy`: per-experiment dictionaries keyed by session ID strings such as `<mouse>_<date>_<blk>` or `<mouse>_<date>_<blk>_<stimtype>`.
- `data/spk/<mouse>_<date>_<blk>_neural_data.npy`: per-session neural activity file; stores a dictionary with key `spks`, where `spks` is a list of neuron-block arrays that must be concatenated across axis 0.
- `data/retinotopy/<mouse>_<date>_trans.npz`: per-session retinotopy / area mapping with at least `xy_t` and `iarea`.
- `data/retinotopy/areas.npz`: area outlines for plotting.
- `data/beh/example_bef_and_aft_learning_behavior.npy`: example behavior file used by figure code.
- `data/beh/Unsupervised_pretraining_behavior/*.npy`: behavior-only pretraining cohorts; these do not have matching neural data files and therefore are not candidate decoder sessions.
- No README or documentation files are present inside `data/`.
- Representative behavior fields in session dictionaries include trial-level arrays (`WallName`, `isRew`, `SoundPos`, `RewPos`, `stim_id`), event-frame arrays (`StartFr`, `EndFr`, `SoundFr`, `RewardFr`, `LickFr`), frame-level aligned signals (`ft_trInd`, `ft_WallID`, `ft_PosCum`, `ft_move`, `ft_CorrSpc`, `ft_GraySpc`, `ft_RunSpeed`), and raw movement traces under `SubjMove`.
- Representative neural file structure: sample session `TX108_2023_01_05_2` contains `spks` split into 3 chunks with shapes `(14686, 20428)`, `(14686, 20428)`, `(14687, 20428)`; concatenation yields 68,553 neurons and matches the retinotopy `iarea` length.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 across 89 unique imaging recordings (counted from retinotopy `iarea` lengths) |
| Neurons / session | Mean 52,708.2; range 20,547 to 89,577 |
| Subjects | 19 imaging mice |
| Sessions / subject | Mean 4.68 unique recordings / mouse; range 2 to 8 |
| Trials (total) | 38,110 across 89 unique recordings |
| Trials / session | Mean 428.2; range 84 to 789 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not explicitly stated in paper; constrained by 89 recordings and 20,547 to 89,577 neurons / recording | “We performed 89 recordings in 19 mice…” and “We ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording.” |
| Neurons / session | 20,547 to 89,577 | “We ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording.” |
| Subjects | 19 imaging mice | “We performed 89 recordings in 19 mice…” |
| Sessions / subject | 89 / 19 = 4.68 recordings per mouse on average (inferred) | “We performed 89 recordings in 19 mice…” |
| Trials (total) | Not explicitly stated | Not explicitly stated in provided text excerpts. |
| Trials / session | Not explicitly stated | Not explicitly stated in provided text excerpts. |
| Neural data time bin | Imaging frames from deconvolved traces; exact frame rate not stated in paper excerpt, reference notebook documents 3.17 Hz (~315.5 ms / frame) | “All our analyses were based on deconvolved fluorescence traces.” |
| Behavior data time bin | Aligned to imaging / frame indices in analyses; exact bin width not explicitly stated in paper excerpt | Not explicitly stated in provided text excerpts. |
| Reward rate | Approximately 50% rewarded vs non-rewarded by task design (inferred from two pseudo-random corridors with one rewarded corridor) | “Mice had to discriminate between visual texture patterns in two corridors; these corridors were repeated in pseudo-random order.” and “Water was available after the sound cue in the rewarded corridor.” |
| Corridor geometry | 4 m corridor + 2 m grey space | “The virtual reality corridors were each 4 m long, with 2 m of grey space between corridors.” |
| Running threshold / VR speed | 6 cm/s threshold; VR advanced at 60 cm/s while above threshold | “The mice moved forward… by running faster than a threshold of 6 cm s−1, but the virtual corridors always moved at a constant speed (60 cm s−1)…” |
| Sound cue position | Uniform between 0.5 m and 3.5 m in imaging mice | “the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m.” |
| Reward zone start (behavior-only) | Uniform between 2 m and 3 m | “The beginning of the reward zone was randomly chosen per trial from a uniform distribution between 2 m and 3 m…” |
| Reward delivery position (imaging) | Approximately uniform between 0.5 m and 3.5 m | “the reward delivery locations were also approximately uniformly distributed between 0.5 m and 3.5 m” |


### Processing Details
- Imaging analyses use Suite2p-processed, deconvolved fluorescence traces; the paper explicitly states all analyses are based on deconvolved traces.
- Timepoints are restricted to running periods for analysis, matching the reference code’s `ft_move > 0` logic.
- The task is organized around corridor entry, random sound cue position inside the corridor, and reward availability only in rewarded trials.
- Imaging mice always receive the sound cue in all trial types; in rewarded task trials the cue marks reward-zone onset.
- The paper’s core selectivity analyses pool responses across positions / timepoints during running and compare stimulus conditions with a `d'`-style selectivity index.
- The paper distinguishes visual plasticity from spatial / reward effects using introduced stimuli `leaf2`, `circle2`, `leaf3`, and shuffled `leaf1` variants after initial training.

### Curation Steps

**Neuron curation rules**:
- Paper-level curation is through Suite2p motion correction, ROI detection, cell classification, neuropil correction, and spike deconvolution.
- Analyses are performed on deconvolved fluorescence traces during running; the paper does not describe an additional global neuron-quality threshold beyond Suite2p / running restriction in the provided excerpts.

**Trial curation rules**:
- Trials are defined by repeated corridor traversals in pseudo-random order.
- For imaging mice, the sound cue is present in all trial types; reward availability differs by corridor and cohort.
- The paper’s behavioural measure of anticipatory licking explicitly counts licks occurring inside the corridor before the sound cue, to avoid confounds from reward-delivery artifacts.
- No explicit global trial-rejection rule is described in the provided paper/methods excerpts beyond restricting analyses to running timepoints.

### Decoders Trained
| Decoded variable | Accuracy |
| No direct decoder / classification accuracy matching the requested conversion task is reported in the paper | N/A; closest reported neural summaries are selectivity `d'`, correlation of position sequences, similarity index, and reward-prediction analyses rather than decoder accuracy |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | `Imaging_Exp_info.npy` is iterated by experiment group; same recording can appear in multiple groups | 142 experiment entries but only 89 unique recording IDs `<mouse>_<date>_<blk>` | “We performed 89 recordings in 19 mice…” | Treat unique recording ID as the true session unit for conversion. Experiment-group duplicates are analysis references, not distinct neural recordings. |
| Subject / session indexing | Some entries include `stimtype` and appear multiple times in behavior files | Raw files contain behavior keys both with and without `stimtype`, but neural files are indexed only by `<mouse>_<date>_<blk>` | Paper discusses multiple test contexts per mouse / recording | Use `<mouse>_<date>_<blk>` for neural session identity; use behavior key with `stimtype` only when needed to pick the correct trial annotations. |
| Corridor units | Notebook/code use 60 position bins and often slice `:40` for corridor, `40:` for gray | `Corridor_Length = 60`, `Texture_Length = 40`, `Gray_Space_length = 20` in all sampled / aggregated behavior files | Paper states 4 m corridor + 2 m gray space | Data are stored in decimeter units: 60 dm total = 6 m, with 40 dm texture corridor + 20 dm gray space. This is fully consistent with paper + code. |
| Reward fraction | Many analyses assume one rewarded corridor versus one non-rewarded corridor inside a task | Overall unique-recording `isRew` fraction is ~0.114; by experiment group it is ~0.48 to 0.51 in `sup_train1_*`, ~0.30 to 0.38 in several supervised later/test sessions, and 0.0 in unsupervised / naive / grating cohorts | Paper’s task design has one rewarded corridor, but also includes unrewarded tests and unsupervised cohorts | No contradiction: pooled dataset reward fraction is low because many sessions are entirely unrewarded or have multiple unrewarded test stimuli. Reward availability must be carried per trial from raw `isRew`, not assumed from task design. |
| Cue position range | Paper says cue position uniform from 0.5 m to 3.5 m; code uses both `SoundPos` and `SoundDelPos` depending on analysis | Most sessions have `SoundPos` roughly 4 to 36 dm; one passive session ranges ~1.1 to 39.0 dm. `SoundDelPos % 60` can extend to ~44.6 dm when delay is present | Paper states sound cue chosen between 0.5 m and 3.5 m; reward delivery approx. 0.5 m to 3.5 m | Use `SoundPos` as actual cue location and `SoundDelPos` only when matching reference reward-prediction analyses. Delayed cue / reward-aligned positions can enter gray space when reward delay is non-zero, explaining values > 40 dm. Minor deviations from the nominal 0.5–3.5 m range appear in a few sessions and should be preserved. |
| Neural signal type | `load_spk` loads `spks`; all downstream code treats these as analysis-ready activity traces | Neural files store only `spks`, split across chunks | Paper says all analyses use Suite2p deconvolved fluorescence traces | The saved `spks` arrays are the deconvolved activity traces from Suite2p, not raw fluorescence; no additional dF/F computation is needed. |
| Brain-region labels | Code groups retinotopy IDs into `V1`, `mHV`, `lHV`, `aHV` | Raw retinotopy file stores numeric `iarea` IDs and `xy_t` coordinates | Paper discusses V1 plus medial / lateral / anterior higher visual areas | Use the grouped labels from reference code for `brain_regions`; this is the paper-consistent abstraction actually used in analyses. |
| Time domain used for analyses | Code alternates between frame-aligned activity and position-interpolated activity depending on figure / analysis | Raw behavior provides both frame indices (`StartFr`, `SoundFr`, `ft_*`) and position variables; raw neural arrays are frame-based | Paper describes running-only analyses and corridor-based computations, but not a single universal binning representation | Final conversion must preserve frame-aligned trial structure for the requested decoder task while matching reference filtering/alignment logic; position-derived outputs can be computed from frame-aligned position traces, and reference position interpolation remains an important sanity check rather than the primary exported neural representation. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `data/spk/<mouse>_<date>_<blk>_neural_data.npy` `['spks']` + retinotopy `iarea` | `neural` | Concatenate neuron chunks along axis 0; keep only neurons in grouped visual areas `V1`, `mHV`, `lHV`, `aHV`; slice frames per trial using `(ft_trInd == trial) & ft_CorrSpc & (ft_move > 0)` | `load_spk`, `load_retino`, running restriction in `Get_dprime_selective_neuron`, `Get_coding_direction`, `Get_sort_spk` | Session unit is unique recording ID `<mouse>_<date>_<blk>`; duplicate experiment-group references are merged. |
| Behavior key fields `ft`, `SoundTime` | `input[0]` | For each retained frame, compute signed seconds to cue as `(SoundTime[trial] - ft[frame]) * 86400` | Paper/methods cue timing + frame-aligned behavior structure | `time_to_sound_cue`; positive before cue, negative after cue. |
| Session date / mouse chronology | `input[1]` | Compute per-session day as calendar-day offset from each mouse’s first unique imaging recording; repeat over trial frames | Session metadata in `Imaging_Exp_info.npy` | `day_of_training`; continuous per-trial covariate required by user task. |
| Behavior field `ft` + `Trial_start_time` | `input[2]` | For each retained frame, compute elapsed seconds since corridor entry as `(ft[frame] - Trial_start_time[trial]) * 86400` | Trial alignment fields documented in notebook / behavior structure | `time_since_trial_start`. |
| Behavior field `isRew` | `input[3]` | Trial-constant 0/1, repeated over frames | Paper task design + raw behavior | `reward_available`; must come from raw per-trial value because many sessions are unrewarded or partially rewarded. |
| Behavior field `WallName` | `output[0]` | Map string categories to integer class IDs; repeat per frame within trial | `WallName` usage throughout utils / figure code | `visual_stimulus_category`; use `WallName` rather than `stim_id`. |
| Behavior fields `LickFr`, `LickTrind` | `output[1]` | Binary frame vector: 1 if one or more licks fall in retained frame, else 0 | `spk_2_firstLick`, `spk_2_cue`, lick utilities | `licking`; time-varying and frame-aligned. |
| Behavior field `ft_Pos` | `output[2]` | Discretize corridor position into four 1 m bins over the 4 m texture corridor: `[0,10)`, `[10,20)`, `[20,30)`, `[30,40]` in raw decimeter units | Corridor/gray split in code (`:40` corridor, `40:` gray) | `position_bin`; exported only on retained corridor-running frames. |
| Behavior field `ft_RunSpeed` | `output[3]` | Compute global quartile edges across all retained frames in all sessions, then discretize each retained frame into 4 bins | Raw behavior frame-aligned running speed | `running_speed_bin`; quartiles should each contain ~25% of retained samples. |
| Session metadata `mname` | `subjects`, `subject_idx` | Unique subject list plus per-session index | Raw metadata in `Imaging_Exp_info.npy` | Use unique recording order after deduplication. |
| Retinotopy `iarea` | `brain_regions`, `brain_region_idx` | Map numeric areas to grouped labels `V1`, `mHV`, `lHV`, `aHV`; drop unassigned / outside neurons | `neu_area_ID`, `load_retino` | Planned neuron curation reduces mean neurons / session from ~52.7k to ~46.1k; all retained neurons have valid visual-area labels. |

### Key Decisions
1. **Session identity = unique recording ID**: Use `<mouse>_<date>_<blk>` as the session key because this resolves the 142 experiment references down to the paper-consistent 89 unique recordings.
2. **Choose one canonical behavior entry per unique recording after validating duplicates**: Repeated experiment-group entries have matching `WallName`, `isRew`, `SoundPos`, and trial counts; `swap1` / `swap2` duplicates differ only in `stim_id`, so `WallName` is the reliable stimulus label.
3. **Neural activity will stay frame-aligned rather than position-interpolated in the exported dataset**: The requested decoder outputs include frame-resolved licking and running speed, so exported trials should preserve native behavioral alignment; position interpolation from the reference code will instead be used as a sanity check.
4. **Use only running corridor frames**: This matches the paper statement “We only considered timepoints during running for analysis” and the reference-code masks built from `ft_CorrSpc` and `ft_move > 0`.
5. **Trial extent for export = corridor portion only, not gray space**: The requested outputs are corridor variables (cue timing, corridor position bins, reward availability), and the reference code usually analyzes corridor frames separately from gray frames. Gray-space frames are excluded from exported trials.
6. **Keep only neurons assigned to grouped visual regions**: Reference analyses operate on `V1`, `mHV`, `lHV`, and `aHV`; neurons outside these groups are not useful for `brain_region_idx` and are excluded.
7. **Use actual behavioral timestamps for continuous time covariates**: `ft` and `Trial_start_time` / `SoundTime` provide trial-aligned times in days; converting to seconds preserves real elapsed time even when non-running frames are removed.
8. **Use `WallName` string labels for stimulus decoding targets**: This preserves all naturalistic / grating / swap categories present in the raw trials and avoids ambiguity from NaNs in `stim_id`.
9. **Discretize position with paper-consistent meter bins**: Raw positions are in decimeters; use 4 bins across the 40 dm texture corridor to match the requested 4 equal 1 m bins.
10. **Discretize running speed with global quartiles over retained samples**: The user explicitly requests 25% data bins, so quartiles will be computed after all filtering decisions are applied.
11. **Repeat per-trial constants across frames**: `day_of_training`, `reward_available`, and `visual_stimulus_category` will be broadcast across time so every trial has consistent `(n_features, n_timepoints)` input/output arrays.

### Planned Sanity Checks
- [ ] Confirm the exported dataset contains 89 sessions and 19 subjects after deduplication, matching the paper’s recording count.
- [ ] For at least one hand-checked session / trial, verify `neural_trial == raw_spk_filtered[:, mask]` with `np.allclose()` using the direct raw frame mask.
- [ ] For a hand-checked trial, verify `time_since_trial_start` and `time_to_sound_cue` against raw `ft`, `Trial_start_time`, and `SoundTime` values with `np.allclose()`.
- [ ] For a hand-checked trial, verify the binary licking vector equals the raw `LickFr` assignments restricted to retained frames.
- [ ] For a hand-checked trial, verify position-bin assignments match `ft_Pos` thresholding at 10 / 20 / 30 / 40 decimeters.
- [ ] After full conversion, verify running-speed quartile occupancy is close to 25% per bin on retained samples.
- [ ] Verify every retained neuron has a valid grouped visual-area label and that dropped neurons correspond exactly to non-visual / unassigned `iarea` values.

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `convert_data.py` with CLI `python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]`.
- The script deduplicates `Imaging_Exp_info.npy` to 89 unique recordings, validates duplicate behavior references, computes global speed quartiles, and exports frame-aligned trial arrays.
- Trial frames are filtered to `ft_CorrSpc & (ft_move > 0)` to match the paper/code running-only analysis.
- Initial mapping plan was revised during implementation because the provided decoder concatenates all session data in memory. Exporting all visual-area neurons would be intractable. The implemented compromise uses a reference-style neuron subset:
  - mHV familiar-stimulus selective neurons defined from odd running corridor frames using the same top/bottom 5% `d'` logic as `Get_coding_direction`.
  - aHV reward-prediction neurons defined using the same `d'late-vs-early >= 0.3` logic as `Get_dprime_rewPred_neuron`.
- Neural arrays are currently stored as `float16` to control output size while preserving continuous-valued activity.

Code inefficiencies identified:
- Full-session interpolation for reward-prediction selection can be expensive if applied to all neurons; implementation restricts this step to aHV neurons only.

Code speedups added:
- Per-session processing only loads one neural recording at a time.
- Trial construction concatenates only the selected neuron subset, not the full recording.
- Global speed-bin estimation uses a lightweight first pass over behavior only; neural files are not touched until conversion.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,855 selected neurons across 2 sessions |
| Neurons / session | Mean 2,427.5; range 1,952 to 2,903 |
| Subjects | 1 (`TX108`) |
| Sessions / subject | 2 |
| Trials (total) | 510 |
| Trials / session | 210, 300 |
| `time_to_sound_cue` range | [-209.4, 199.8] s |
| `day_of_training` range | [1.0, 10.0] |
| `time_since_trial_start` range | [0.0, 212.3] s |
| `reward_available` range | [0.0, 1.0] |
| `visual_stimulus_category` distribution | [0.515, 0.485] for `[rock1, wood1]` |
| `licking` distribution | [0.732, 0.268] for `[no_lick, lick]` |
| `position_bin` distribution | [0.246, 0.252, 0.251, 0.250] |
| `running_speed_bin` distribution | [0.250, 0.250, 0.250, 0.250] |

### Processing Plots Review
- `processing_TX108_2023_03_13_1.png` and `processing_TX108_2023_03_22_1.png` were created.
- No obvious temporal misalignment in the sample-trial panels: retained frames progress monotonically through corridor position, cue timing falls within the trial, lick events align to retained frames, and speed quartile edges partition the kept speed samples sensibly.
- The neuron-curation panel shows both mHV and aHV contributions in the rewarded sample sessions, which is expected from the implemented reference-style union.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Restrict reward-prediction interpolation to aHV neurons only | Avoids interpolating the full recording for supervised sessions |
| Select reference-style neuron subset before export | Keeps sample file size at 63 MB and keeps validator-trainable memory in range |
| Behavior-only first pass for speed quartiles and metadata | Avoids touching neural files until conversion stage |

| Step | Time / Session | Estimated Total Time |
| | | |
| Rewarded sample sessions (`TX108_2023_03_13_1`, `TX108_2023_03_22_1`) | 15.2 s, 19.0 s | 26.1 min if all 89 sessions behaved like these rewarded sessions |
| Weighted full-dataset estimate | ~11 to 18 s / session depending on reward interpolation branch | Roughly 13 to 18 minutes for 89 sessions; 28 unique sessions have rewards and 61 are unrewarded, so actual full runtime should be lower than the worst-case 26.1 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `visual_stimulus_category` | 0.9976 | 0.9785 |
| `licking` | 0.7001 | 0.6957 |
| `position_bin` | 0.9542 | 0.8853 |
| `running_speed_bin` | 0.3922 | 0.3650 |

- After the Step 10 neural-export bug fix, the sample decoder was re-run on the corrected `sample_data.pkl`.
- Loss decreased steadily from `2151.144344` at epoch 1 to `9.104101` at epoch 200.
- All validation balanced accuracies are above uniform-chance levels (`0.5`, `0.5`, `0.25`, `0.25` respectively).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 1.93 GB (`1,934,733,385` bytes)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not explicitly stated; 89 recordings with 20,547–89,577 neurons each | Loads full-session `spks` arrays | 4,691,034 raw neurons across 89 unique recordings | 102,541 selected neurons across 89 sessions | No; deliberate reference-style subset for tractability |
| Mean neurons/session | 20,547–89,577 range reported | Full-session neurons used before task-specific selection | 52,708.2 raw neurons/session mean | 1,152.2 selected neurons/session mean | No; deliberate reference-style subset for tractability |
| Subjects | 19 | 19 unique mice in `Imaging_Exp_info.npy` | 19 | 19 | Yes |
| Sessions | 89 recordings | 89 unique recording IDs after deduplication | 89 | 89 | Yes |
| Trials (total) | Not explicitly stated | Determined by behavior dictionaries per recording | 38,110 | 38,110 | Yes |
| Trials/session (mean) | Not explicitly stated | Determined by behavior dictionaries per recording | 428.2 | 428.2 | Yes |
| `time_to_sound_cue` range | Paper nominal cue range 0.5–3.5 m; actual time range session-dependent | Frame-/cue-aligned behavior fields used throughout | Session-dependent from raw `SoundTime` / `ft` | [-1763.3, 723.5] s | Consistent with variable trial durations and running-only frame retention |
| `day_of_training` range | Not explicitly stated | Not in reference code | Calendar span from raw session dates | [1.0, 93.0] | Derived quantity |
| `time_since_trial_start` range | Trial-start alignment implied; no explicit range | Trial-start fields available in raw behavior | Session-dependent from raw `Trial_start_time` / `ft` | [0.0, 1765.2] s | Consistent with running-only frame retention and long rewarded trials |
| `reward_available` range | Rewarded vs unrewarded trials present by design | Uses `isRew` / reward-specific trial logic | [0, 1] | [0, 1] | Yes |
| `visual_stimulus_category` distribution | Multiple categories including familiar, test, and swap stimuli | Uses `WallName`, `stim_id`, and `UniqWalls` across figures | 15 category strings present across raw sessions | 15 category strings present; fractions: `[0.252, 0.049, 0.010, 0.264, 0.019, 0.019, 0.128, 0.040, 0.071, 0.013, 0.065, 0.008, 0.010, 0.038, 0.014]` | Yes |
| `licking` distribution | Anticipatory licking present in supervised sessions, absent/low in unrewarded cohorts | Lick utilities operate on raw lick frames and times | Raw lick events are sparse overall | `[0.963, 0.037]` for `[no_lick, lick]` | Yes |
| `position_bin` distribution | Corridor spans four 1 m bins | Code treats 0–40 dm as corridor | Raw position covers the 4 m corridor | `[0.250, 0.249, 0.250, 0.252]` | Yes |
| `running_speed_bin` distribution | Not explicitly stated | Derived from raw running speed | Global quartiles requested by user | `[0.250, 0.250, 0.250, 0.250]` | Yes |

- `verification_full_out.txt` reports “Data format is valid, no errors or warnings.”
- After the Step 10 neural-export bug fix, the full dataset was rebuilt and re-verified.
- Corrected full conversion completed in `1,013.7 s` (`16.9 min`). This was above the nominal 15 min guideline but remained within the Step 7 estimate band (13 to 18 min) and did not exceed the estimate by 1.5x, so the run was retained.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` was re-run after the neural-export fix and again reported “Data format is valid, no errors or warnings.” No validator issues remained to address.
2. **Raw-data sanity checks with `np.allclose()`**: Independently reconstructed exported arrays directly from raw `Beh_*.npy`, `*_neural_data.npy`, and `*_trans.npz` files for two sessions spanning unrewarded (`DR10_2022_07_12_1`) and rewarded (`TX108_2023_03_13_1`) regimes. For each session, checked three trials (early, middle, late) and verified:
   - `brain_region_idx` matches independently reconstructed selected-neuron region labels.
   - `neural[session][trial]` matches raw selected-neuron activity on retained frames with `np.allclose(..., equal_nan=False)`.
   - `input[session][trial]` matches independently reconstructed `time_to_sound_cue`, `day_of_training`, `time_since_trial_start`, and `reward_available` with `np.allclose()`.
   - `output[session][trial]` matches independently reconstructed `visual_stimulus_category`, `licking`, `position_bin`, and `running_speed_bin` exactly.
   - All checked `time_since_trial_start` traces are monotonic within trial.
3. **Reference code comparison**:
   - **(a) Data loading**: `convert_data.py` loads the same raw sources as `load_spk` and `load_retino`: concatenated `spks` chunks plus retinotopy `iarea`. Session deduplication to 89 unique recordings is a conversion-level wrapper around the reference experiment index.
   - **(b) Neuron / trial filtering**: Trial filtering uses the same running-corridor mask as the reference code, `ft_CorrSpc & (ft_move > 0)`. mHV neuron selection uses the same odd-trial familiar-stimulus `d'` percentile logic as `Get_coding_direction`. aHV reward-prediction selection uses the same interpolated early-vs-late cue `d' >= 0.3` logic as `Get_dprime_rewPred_neuron`.
   - **(c) Temporal alignment**: Exported trials are aligned to corridor entry (`Trial_start_time`), while cue timing is computed from raw `SoundTime` and frame timestamps `ft`. This preserves the native frame grid while matching the reference running-frame restriction.
   - **(d) Binning**: Neural data stay on the native imaging frame bins (median `314.69 ms`), which is consistent with the reference frame-based deconvolved traces. Position output uses the paper-consistent 4 x 1 m bins over the 40 dm texture corridor; running speed uses user-requested global quartiles.
   - **(e) Input construction**: All four decoder inputs come directly from raw behavioral fields (`SoundTime`, `Trial_start_time`, `ft`, `isRew`) or recording date metadata.
   - **(f) Output construction**: All four decoder outputs are built from raw behavioral fields (`WallName`, `LickFr`/`LickTrind`, `ft_Pos`, `ft_RunSpeed`) on the same retained frame indices used for neural export.
4. **Key statistics comparison**:
   - Session, subject, and trial totals remain exactly consistent with the paper/reference data: 89 sessions, 19 mice, 38,110 trials.
   - Stimulus-category counts, licking sparsity, position-bin occupancy, and speed-bin occupancy match raw-data expectations and the verification report.
   - The only deliberate mismatch is neuron count: the converted export contains 102,541 selected neurons rather than all 4,691,034 raw neurons. This reflects the documented reference-style mHV/aHV subset needed to keep `train_decoder.py` tractable; the decoder implementation concatenates all session tensors in memory and cannot practically train on the full raw-neuron matrix.
5. **Edge-case scan**:
   - Every converted trial is non-empty (`min T = 11`, `max T = 178`).
   - `reward_available` is constant within every trial.
   - `visual_stimulus_category` is constant within every trial.
   - `time_since_trial_start` is monotonic within every converted trial.
   - No off-by-one issues were found at trial starts/ends in the checked raw-vs-converted comparisons.

### Issues Found and Resolved
- **Incorrect chunk-local neuron indexing in neural export**: The original Step 6/7/9 conversion code used `np.flatnonzero(local_keep)` when slicing each `spk` chunk. This indexed rows by position within `kept_idx` rather than by the true chunk-local neuron IDs, so exported neural matrices did not correspond to the intended selected neurons. Fixed `convert_data.py` to compute chunk-local rows as `kept_idx[in_chunk] - start`, then re-ran sample conversion, sample verification, sample decoder training, full conversion, full verification, and all Step 10 checks. After the fix, all raw-vs-converted neural/input/output spot checks passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `visual_stimulus_category` | 0.9129 | 0.8029 | 15-way classification; 12.0x above uniform chance (`0.0667`) on validation. |
| `licking` | 0.8819 | 0.8724 | Strongly above chance (`0.5000`); no meaningful train/validation gap. |
| `position_bin` | 0.6899 | 0.6807 | 4-way corridor-position decoding remains robust on held-out trials. |
| `running_speed_bin` | 0.4365 | 0.4361 | Above 4-way chance (`0.2500`) and above the Step 12 1.5x-chance threshold. |

- Full decoder training completed successfully on `cuda`.
- Loss decreased from `685.803365` at epoch 1 to `2.978092` at epoch 200.
- Final test loss: `3.526753`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `visual_stimulus_category` | Validation balanced accuracy `0.8029` | No directly comparable decoder accuracy reported; paper instead reports strong familiar / novel visual selectivity (`d'`) in higher visual cortex. |
| `licking` | Validation balanced accuracy `0.8724` | No directly comparable decoder accuracy reported; paper reports robust anticipatory licking differences between rewarded and unrewarded conditions. |
| `position_bin` | Validation balanced accuracy `0.6807` | No directly comparable decoder accuracy reported; paper reports reliable position- and sequence-related spatial organization in neural responses. |
| `running_speed_bin` | Validation balanced accuracy `0.4361` | No directly comparable decoder accuracy reported; behavior is strongly structured by the VR task, so above-chance decoding is expected. |

- **Accuracy vs chance**:
  - `visual_stimulus_category`: `0.8029 / 0.0667 = 12.0x chance`
  - `licking`: `0.8724 / 0.5000 = 1.74x chance`
  - `position_bin`: `0.6807 / 0.2500 = 2.72x chance`
  - `running_speed_bin`: `0.4361 / 0.2500 = 1.74x chance`
- No output is below chance.
- No output is below the Step 12 “1.5x chance” warning threshold.
- **Accuracy comparison to paper**: The reference paper and provided methods do not report decoder accuracies for this exact neural-decoder task. The closest reported analyses are stimulus-selectivity `d'`, position-sequence correlations, and reward-prediction summaries rather than held-out classification accuracy. Therefore the paper provides qualitative, not numeric, expectations for this conversion task.
- **Train vs validation gap**:
  - `visual_stimulus_category`: `0.9129 / 0.8029 = 1.14x`
  - `licking`: `0.8819 / 0.8724 = 1.01x`
  - `position_bin`: `0.6899 / 0.6807 = 1.01x`
  - `running_speed_bin`: `0.4365 / 0.4361 = 1.00x`
- No output exceeds the Step 12 overfitting flag of `1.5x` train-vs-validation gap.
- Because all outputs are above chance with small train/validation gaps, no further conversion changes were required after Step 11.

### Issues Found and Resolved
- **No new conversion issues found in Step 12**: Accuracy review after the Step 10 bug fix did not reveal below-chance outputs, suspiciously weak outputs, or overfitting patterns that would justify another conversion revision.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

- Added [`README.md`](/app/README.md) with dataset description, usage instructions, converted-data caveats, and key statistics.
- Added [`cache/raw_sanity_checks.py`](/app/cache/raw_sanity_checks.py) so the Step 10 raw-vs-converted spot checks are reproducible.
- Added [`cache/README_CACHE.md`](/app/cache/README_CACHE.md) documenting cached artifacts.
- Moved stale DR10 processing plots and temporary memmap experiment files into `cache/` to keep the top-level directory focused on the final conversion products and current validation outputs.
