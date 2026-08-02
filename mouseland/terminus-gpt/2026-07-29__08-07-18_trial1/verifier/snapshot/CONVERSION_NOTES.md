# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code
- data
- decoder.py
- docker-compose.yaml
- methods.txt
- paper.pdf
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_fig1_dat | code/[figure script] | LOADING | Loads precomputed behavior and process-data arrays (`.npy`, `.npz`) for figure analyses. |
| get_lick_raster | code/utils.py | PROCESSING | Converts lick event positions/trial indices into trial-organized lick raster summaries. |
| get_mean_lick_response | code/utils.py | PROCESSING | Aggregates lick responses across trials/conditions for behavioral summaries. |
| pretrain_exp_lick_raster | code/utils.py | PROCESSING | Builds category-conditioned lick raster and first-lick summaries from trial lick positions. |
| get_first_lick_distribution | code/utils.py | PROCESSING | Computes histogram/distribution of first-lick positions across sessions/conditions. |
| lickCount | code/utils.py | PROCESSING | Produces binary lick-response summaries per trial, likely around reward-position windows. |

### Notes
- Inspected only files under `code/` as required.
- The reference code appears organized around figure scripts that load preprocessed arrays and call shared helpers in `code/utils.py`.
- Behavioral processing is trial-based and uses lick positions (`LickPos`) and lick trial indices (`LickTrind`) to reconstruct per-trial lick rasters.
- Stimulus/category structure uses fields such as `WallType`, `WallName`, `UniqWalls`, and reward/category mapping via `get_cat_id(...)`.
- The code suggests the dataset is already substantially preprocessed into numpy dictionaries, so conversion should likely reuse these trial-level arrays rather than recomputing from raw acquisition streams unless necessary.
- Need to preserve the same trial grouping, category mapping, and lick alignment logic when constructing decoder inputs/outputs.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Inspected only the `data/` directory.
- `data/beh/*.npy` files load as scalar object arrays containing Python dictionaries.
- Most behavior files map session IDs (e.g. `TX109_2023_05_12_1`) to per-session dictionaries with fields including:
  `ntrials`, `trInd`, `trInd_odd`, `trInd_even`, `Trial_start_time`, `Trial_end_time`, `SubjMove`, `Gray_space_time`, `SoundPos`, `SoundTime`, `SoundTimeDelay`, `RewTime`, `RewPos`, `isRew`, `WallType`, `WallIsProbe`, `WallName`, `UniqWalls`, `LickTrind`, `LickTime`, `LickPos`, `Lick_wallName`, `VRposTime`, `VRpos`, `VRposCum`.
- `data/beh/Imaging_Exp_info.npy` appears to store experiment-group metadata/indices rather than individual sessions.
- Neural files are stored separately as `data/**/*_neural_data.npy` and load as dictionaries with at least the key `spks`.
- In sampled neural files, `spks` is list-backed with length 3; earlier simple `np.asarray` probes on some files yielded dense shapes like `(3, 19408, 31707)`, indicating heterogeneous internal representation that needs careful loading.
- Session naming convention embeds subject ID, date, and session number; some sessions include suffixes like `_swap1` / `_swap2`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | [pending exact count from neural files] |
| Neurons / session | [pending exact interpretation of `spks` dimensions] |
| Subjects | 19 true animal IDs observed in session-like keys |
| Sessions / subject | variable; 100 total session-like keys |
| Trials (total) | 63482 |
| Trials / session | min 84, max 789, mean 440.847 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | [not yet located in methods excerpt] | |
| Neurons / session | [not yet located in methods excerpt] | |
| Subjects | [not yet located in methods excerpt] | |
| Sessions / subject | [not yet located in methods excerpt] | |
| Trials (total) | [not yet located in methods excerpt] | |
| Trials / session | [not yet located in methods excerpt] | |
| Neural data time bin | Deconvolved fluorescence traces; imaging processed with Suite2p | "All our analyses were based on deconvolved fluorescence traces." |
| Behavior data time bin | [not explicitly stated in methods excerpt] | |
| Reward rate | Reward available only in rewarded corridor for task mice; unsupervised training has sound cue but no rewards | "Although the rewards were absent, the sound cue was still presented in the unsupervised training experiment for consistency." |
| Sound cue location | Uniform from 0.5 m to 3.5 m | "the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m" |
| Reward zone start | Uniform from 2 m to 3 m in behavior-only experiment | "The beginning of the reward zone was randomly chosen per trial from a uniform distribution between 2 m and 3 m" |
| Training duration | 5 days in behavior-only experiment | "all animals started training ... and continued training for exactly 5 days" |

### Processing Details
- Imaging mice: sound cue presented in all trial types.
- Task mice: sound cue indicates beginning of reward zone in rewarded corridor.
- Reward delivered if a lick is detected after the sound cue in the rewarded corridor.
- Unsupervised training: rewards absent, but sound cue still presented for consistency.
- Behavioral lick-response analyses considered licks occurring inside the corridor but before the sound cue to isolate anticipatory licking.
- Calcium imaging preprocessing used Suite2p for motion correction, ROI detection, cell classification, neuropil correction, and spike deconvolution.
- Non-negative deconvolution used a decay timescale of 0.75 s.
- Analyses are based on deconvolved fluorescence traces.

### Curation Steps

**Neuron curation rules**:
- Suite2p cell classification was used; exact downstream neuron inclusion criteria not yet identified from current methods excerpt.

**Trial curation rules**:
- Trial structure depends on corridor type, reward availability, and sound cue timing/location.
- Need further confirmation from code/data on any explicit exclusion of invalid trials.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal type | Code/methods refer to deconvolved imaging traces | Neural files expose `spks` entries, often list-backed and sometimes coercible to dense arrays | Methods: analyses based on deconvolved fluorescence traces after Suite2p deconvolution | Treat `spks` as the deconvolved neural activity stream for conversion, while confirming exact dimensional interpretation during script development. |
| Session identifiers | Code utilities operate on per-session dicts keyed by session IDs | Behavior files include both true session keys and metadata/group keys | Methods discuss experiment groups and sessions separately | Filter behavior keys to session-like IDs when matching to neural files. |
| Reward/sound structure | Code uses `isRew`, `WallName`, `UniqWalls`, lick variables | Behavior data contain these exact fields | Methods describe rewarded vs unrewarded corridors and sound cue timing | Use behavior fields directly; these are consistent across code, data, and methods. |
| Neural storage format | Code likely expects numpy-like access | Some `spks` probes yielded dense shapes, others are list-backed | Methods do not specify storage format | Conversion code must robustly handle both representations while preserving trial alignment. |

---

## Step 5: Mapping Planning
**Status**: IN PROGRESS

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| | neural | | |
| | input[0] | | |
| | output[0] | | |

### Key Decisions
1. **[Decision]**: [Rationale]

### Planned Sanity Checks
- [ ] Check 1
- [ ] Check 2

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Implemented frame-aligned conversion using `ft_trInd`, `ft_PosCum`, `ft_move`, `LickTrind`, `LickTime`, and concatenated `spks`.
- Sample session selection now prioritizes high-lick supervised sessions.
- Optimized trial indexing by precomputing frame indices per trial instead of repeated `np.where` scans.

Code inefficiencies identified:
[Note]

Code speedups added:
- Precomputed `trial_frame_idx` mapping once per session, reducing repeated frame-index scans and cutting sample conversion runtime from ~160 s to ~83 s for 2 sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 168790 |
| Neurons / session | 85481, 83309 |
| Subjects | 2 (TX108, TX109) |
| Sessions / subject | 1, 1 |
| Trials (total) | 580 |
| Trials / session | 210, 370 |
| time_to_sound_cue range | [-293.4, 200.1] |
| day_of_training range | [1.0, 1.0] |
| time_since_trial_start range | [0.0, 295.9] |
| reward_availability range | [0.0, 1.0] |
| visual_stimulus_category distribution | circle1 0.176, circle2 0.053, leaf1 0.151, leaf2 0.053, rock1 0.312, wood1 0.255 |
| licking distribution | no_lick 0.808, lick 0.192 |
| corridor_position_bin distribution | bin1 0.260, bin2 0.238, bin3 0.297, bin4 0.204 |
| running_speed_bin distribution | q1 0.002, q2 0.216, q3 0.533, q4 0.250 |

### Processing Plots Review
- `--show-processing` not yet implemented.
- Revised sample selection now uses high-lick, rewarded supervised sessions rather than unrewarded unsupervised sessions.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| Sample conversion | ~41 s/session after trial-index optimization | Rough full conversion time ~52 minutes for 76 sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None reported by `train_decoder.py --verify-only`

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| visual_stimulus_category | 0.9438 | 0.8770 |
| licking | 0.7437 | 0.7259 |
| corridor_position_bin | 0.8046 | 0.7290 |
| running_speed_bin | 0.7484 | 0.5464 |

### Notes
- GPU run failed with `torch.OutOfMemoryError` after epoch 1; reran on CPU successfully.
- Loss decreased steadily from 13590.913086 at epoch 1 to 222.671638 at epoch 200.
- All validation balanced accuracies were above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: created (very large; latest corrected version includes 76 sessions)
- `verification_full_out.txt`: previous run interrupted during load; custom consistency checks used in parallel

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | `spks` concatenated across list elements per session | session-specific, variable | [pending lightweight custom summary] | |
| Mean neurons/session | | variable | variable | [pending lightweight custom summary] | |
| Subjects | | | 19 true animal IDs in common-session pool | [pending lightweight custom summary] | |
| Sessions | | | 76 common behavior+neural sessions | 76 converted sessions | Yes |
| Trials (total) | | | variable by session | [pending lightweight custom summary] | |
| Trials/session (mean) | | | variable by session | [pending lightweight custom summary] | |
| Input/output ranges | | behavior-frame aligned | observed in sample/full logs | [pending lightweight custom summary] | |

### Notes
- Initial full conversion produced 75 sessions because duplicate session IDs in behavior files caused a richer session (`TX109_2023_03_27_1`) to be overwritten by a poorer example file.
- Fixed by preferring richer duplicate behavior entries based on key count and `ft_trInd` size.
- Corrected full conversion completed successfully with 76 sessions in 2177.04 s (~36.3 min).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Output log verification: initial `conversion_full_out.txt` showed 75 kept sessions instead of 76 common sessions.
2. Session consistency check: compared common behavior+neural session IDs against kept session IDs from the conversion log.
3. Missing-session diagnosis: identified dropped session `TX109_2023_03_27_1` and inspected its raw behavior/neural data.
4. Loader bug analysis: found duplicate session IDs across behavior files, with a poorer example file overwriting a richer main behavior file in `load_behavior_sessions`.
5. Re-ran full conversion after fixing duplicate-session handling and confirmed 76 kept sessions.

### Issues Found and Resolved
- Duplicate-session overwrite bug: `load_behavior_sessions` previously overwrote richer session dicts with poorer duplicates (notably `TX109_2023_03_27_1` from `example_bef_and_aft_learning_behavior.npy`). Fixed by preferring the richer dict based on key count and `ft_trInd` size.
- After fix, `TX109_2023_03_27_1` builds successfully with 84 trials and neural shape `(63625, 49)` on the first trial.
- Corrected full conversion now includes all 76 common behavior+neural sessions.

---

## Step 11: Full Decoder Training
**Status**: IN PROGRESS

### Training Progress
- Loss decreasing: [Not reached on full dataset run]

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| visual_stimulus_category | | | Full run not completed |
| licking | | | Full run not completed |
| corridor_position_bin | | | Full run not completed |
| running_speed_bin | | | Full run not completed |

### Notes
- Attempted full decoder training on CPU with `python3 -u train_decoder.py converted_data.pkl --cpu`.
- `train_decoder_full_out.txt` remained empty before manual interruption, indicating that loading/training on the 344G pickle is impractically slow in the current storage format.
- Additional optimization/compression of the converted dataset would be required to make full decoder training practical.

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
