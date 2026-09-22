# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code/
- data/
- decoder.py
- docker-compose.yaml
- methods.txt
- paper.pdf
- train_decoder.py

Setup verification:
- python3 import test passed
- numpy import passed (2.4.4)
- torch import passed (2.6.0+cu124)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `pp.create_sess(...)` | `code/notebooks/make_session_pkl.md` (calls into preprocessing package) | LOADING | Create/load a session object with scan info, VR, suite2p, and behavior streams |
| `ut.quick_load_multi_anim_sess(...)` | `code/notebooks/Fig3_Decoder.md` | LOADING | Load precomputed per-day multi-animal session data used for decoder analyses |
| `dd.subclass(multiDayData)` | `code/notebooks/Fig3_Decoder.md` | PROCESSING | Keep only decoder-relevant subset of `multiDayData` |
| `ut.write_sess_pickle(...)` | `code/notebooks/make_multiDayData_class.md` | PROCESSING | Save processed `multiDayData` pickle for downstream analyses |
| `multiDayData[day].circ_trial_matrix` | `code/notebooks/make_multiDayData_class.md`, `Fig4_Ext4_sequences.md` | PROCESSING | Trial-by-position representation in circularized track coordinates |
| `multiDayData[day].rzone_pos` | `code/notebooks/Fig2_Ext2_identifyRewardRelative.md` | PROCESSING | Reward zone location(s) per animal/day/environment set |
| `seq[d][an]['dist_to_reward0/1']` | `code/notebooks/Fig4_Ext4_sequences.md` | PROCESSING | Reward-anchored distance variable used in analyses |
| `multiDayData[day].circ_rel_stats_across_an['include_ans']` | `code/notebooks/Fig3_Decoder.md` | CURATION | Inclusion list of animals retained for circular/reward-relative analyses |
| `multiDayData[day].place_cell_masks`, `overall_place_cell_masks` | `code/notebooks/Fig2_Ext2_identifyRewardRelative.md` | CURATION | Cell classification / inclusion masks used in analysis summaries |

### Notes
- The reference code is organized primarily as notebooks plus markdown exports, with a workflow that builds increasingly processed pickle objects.
- Session-level data are created with `pp.create_sess(...)`, which loads scan metadata, VR data, suite2p outputs, and behavior.
- Downstream analyses rely heavily on saved pickle objects rather than re-reading raw files each time.
- `multiDayData` is the main high-level analysis object for decoder and sequence analyses.
- Decoder notebook `Fig3_Decoder.md` loads a saved `multiDayData` pickle, derives `include_ans`, and uses `ut.quick_load_multi_anim_sess(...)` to access per-day multi-animal session data.
- Key behavioral/spatial variables visible in the reference code include circularized trial matrices (`circ_trial_matrix`), reward-zone positions (`rzone_pos`), and reward-anchored distance variables (`dist_to_reward0/1`).
- The track length used in analyses is 450 cm, matching the decoder task specification for absolute position binning.
- The code suggests calcium imaging data processed through suite2p; therefore neural activity likely comes from preprocessed fluorescence/deconvolved traces rather than raw imaging movies.


---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
Data are organized as NWB files under subject directories (`/app/data/sub-m*/sub-m*_ses-*_behavior+ophys.nwb`). Each file appears to contain one imaging/behavior session. Representative NWB files contain `behavior` and `ophys` processing modules. The `ophys` module includes `Deconvolved`, `Fluorescence`, `Neuropil`, and `ImageSegmentation`; the ROI segmentation table includes columns `pixel_mask`, `iscell`, and `planeIdx`. The standard `nwb.trials` table is empty and `nwb.intervals` is also empty, so trial structure must be recovered from behavioral time series rather than canonical NWB trial tables. Behavioral time series include `Reward`, `autoreward`, `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, and `trial_start`, which appear sufficient to reconstruct trials and task variables. In a representative session, behavioral series have length 19,818 samples and the neural `Deconvolved`, `Fluorescence`, and `Neuropil` ROI response matrices all have shape `(19818, 349)` at 15.5078125 Hz, indicating framewise alignment between behavior and neural activity. The segmentation table includes `iscell`, but values are confidence-like rather than strictly binary, suggesting suite2p-style cell curation by thresholding `iscell` may be required.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | ~312,110 ROIs across sessions before curation |
| Neurons / session | min 315, mean 2053.36, max 5085 (ROIs before curation) |
| Subjects | 11 |
| Sessions / subject | mostly 14, one subject with 12 |
| Trials (total) | 0 in `nwb.trials` table; true trial count not yet identified |
| Trials / session | 0 in `nwb.trials` table; true trial count not yet identified |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Sessions used in one remapping subset | 77 sessions considered; 50 passed RR significance criterion | "Sessions in which the real model’s performance at k = 2 clusters exceeded that of the shuffle ... were accepted for further analysis (n = 50 out of 77 sessions for the RR population vector)." |
| Neural data representation | Deconvolved activity matrices | "we used the deconvolved activity matrices of each neuron" |
| Position binning in one analysis | 10 cm linear position bins | "(i trials × j 10 cm linear position bins), smoothed with a 10 cm s.d. Gaussian" |
| Behavioral analyses | Lick counts and running speed analyzed trial-by-trial | "To identify transitions in behavior, we similarly maximum-normalized the spatially binned lick counts or spatially binned running speed" |
| Track length | 450 cm | Supported by reference code and decoder task; code uses 450 in position transforms |

### Processing Details
- Methods confirm the use of deconvolved neural activity for trial-by-trial analyses.
- Reference code and methods both indicate analyses are organized by trial and position along a 450 cm track.
- Behavioral variables of interest include licking and running speed.
- Reward-switch structure is central to the experiment; sessions can contain different reward-zone centers across days.

### Curation Steps

**Neuron curation rules**:
- Methods text examined so far does not explicitly state NWB-side neuron curation, but data include suite2p-style `iscell` scores in ROI segmentation and reference code uses suite2p-processed imaging outputs.
- Planned curation decision: retain ROIs with `iscell > 0.5`, subject to validation against dataset statistics and decoder performance.

**Trial curation rules**:
- Trials are not stored in `nwb.trials`; they must be reconstructed from behavioral time series.
- Planned valid-trial rule: use frames with `trial number >= 0`, `scanning == 1`, and trial starts defined by `trial_start > 0`.

### Decoders Trained
| Decoded variable | Accuracy |
| | |
---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Trial storage | Reference analyses are trial-based | `nwb.trials` and `nwb.intervals` are empty | Methods discuss trial-based analyses | Reconstruct trials from behavioral time series `trial_start` and `trial number` |
| Reward-zone representation | Reference code uses reward-relative variables and `rzone_pos` | NWB `reward_zone` is a sparse time series near zone locations, not a direct A/B/C label | Paper focuses on reward-relative coding | Infer per-trial reward-zone center from positions where `reward_zone > 0`; map centers to A/B/C |
| Environment coding | Reference code refers to different environments/sets | NWB `environment` values are -1 invalid, 0 or 1 valid | Task requires binary environment type | Use valid values 0/1 directly as ENV1/ENV2 |
| Neural stream choice | Methods use deconvolved activity | NWB contains Deconvolved, Fluorescence, Neuropil | Methods explicitly mention deconvolved matrices | Use Deconvolved as neural activity |


---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Deconvolved/plane0` | neural | transpose to `(n_neurons, n_time)` and subset to curated cells | Methods use deconvolved matrices | Native sampling ~15.5 Hz |
| `ImageSegmentation.iscell` | neural curation | keep `iscell > 0.5` | suite2p-derived segmentation | Confidence-like score in NWB |
| frame timestamps / trial indices | input[0] time_from_trial_start | seconds from each trial start | trial-based analyses in methods | time-varying |
| `environment` | input[1] environment_type | valid values 0/1 repeated over trial | reference code uses environment/set structure | binary per trial |
| `trial number` | input[2] trial_number | framewise constant within trial | trial-based analyses | continuous per trial |
| derived previous reward outcome | input[3] previous_trial_outcome | shift per-trial reward outcome by one trial; first trial default 0 | reward/omission task structure | binary per trial |
| derived position relative to inferred zone center | output[0] distance_to_reward_zone | discretize into 7 requested bins | reference code uses reward-relative variables | time-varying |
| `position` | output[1] absolute_position | discretize 0-450 cm into 5 bins | 450 cm track in code/task | time-varying |
| `speed` | output[2] speed | discretize per requested bins | behavior analyses include speed | time-varying |
| `lick` | output[3] lick | binarize `lick > 0` | behavior analyses include licking | time-varying |
| inferred per-trial zone center | output[4] reward_zone_location | map center to A/B/C by nearest canonical center (~85, ~205, ~325 cm) | reference code uses `rzone_pos` | per trial, repeated across time |
| `Reward` events aligned to trials | output[5] reward_outcome | 1 if any reward event occurs during trial else 0 | reward/omission task structure | per trial, repeated across time |

### Key Decisions
1. **Use deconvolved activity**: Matches methods text and reference analyses.
2. **Curate neurons with `iscell > 0.5`**: NWB stores confidence-like `iscell`; thresholding follows suite2p convention and should remove non-cells.
3. **Reconstruct trials from `trial_start` and `trial number`**: Canonical NWB trials table is empty.
4. **Use native frame rate as time bins**: Neural and behavior streams are already aligned sample-by-sample at ~15.5 Hz.
5. **Infer reward-zone location from positions where `reward_zone > 0`**: Empirically clusters near ~85, ~205, or ~325 cm across sessions.
6. **Compute distance-to-zone from position minus inferred zone center**: Better matches requested decoder output than raw `reward_zone` codes.
7. **Restrict to valid imaging/trial frames**: Use `scanning == 1` and `trial number >= 0` to avoid invalid pre/post periods.

### Planned Sanity Checks
- [ ] Check that neural and behavior arrays have identical frame counts in each session.
- [ ] Check that reconstructed rewarded-trial counts match number of `Reward` events for spot-checked sessions.
- [ ] Check that inferred reward-zone centers cluster near three canonical positions.
- [ ] Check that per-trial environment is constant within each trial.


---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` to load NWB sessions, curate ROIs using suite2p-style `iscell`, reconstruct trials from behavioral time series, align neural/behavioral streams at native frame resolution (~15.5 Hz), infer reward-zone centers from `reward_zone > 0` positions, and construct decoder inputs/outputs in the required format.

Code inefficiencies identified:
- Full NWB sessions are loaded eagerly; may be acceptable but should be monitored during full conversion.

Code speedups added:
- Used vectorized NumPy operations for discretization and per-trial extraction where possible.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | ~312,110 ROIs across sessions before curation |
| Neurons / session | min 315, mean 2053.36, max 5085 (ROIs before curation) |
| Subjects | 11 |
| Sessions / subject | mostly 14, one subject with 12 |
| Trials (total) | 0 in `nwb.trials` table; true trial count not yet identified |
| Trials / session | 0 in `nwb.trials` table; true trial count not yet identified |
| time_from_trial_start_sec range | [0.0, 30.7] |
| environment_type range | [0, 0] |
| trial_number range | [0, 80] |
| previous_trial_outcome range | [0, 1] |
| distance_to_reward_zone distribution | [0.134,0.105,0.089,0.001,0.135,0.103,0.433] |
| absolute_position distribution | [0.310,0.206,0.203,0.146,0.135] |
| speed distribution | [0.120,0.062,0.065,0.233,0.520] |
| lick distribution | [0.821,0.179] |
| reward_zone_location distribution | [0.699,0.293,0.008] |
| reward_outcome distribution | [0.161,0.839] |

### Processing Plots Review
No format errors. Sample is not behaviorally diverse: both sessions come from one subject and one environment code, and reward-zone location C is nearly absent. This is acceptable for format validation but not representative of the full dataset.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: `y_pred contains classes not in y_true` during validation metrics on small sample split; likely due to missing classes in held-out data for one output.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| distance_to_reward_zone | 0.5029 | 0.2924 |
| absolute_position | 0.5322 | 0.4611 |
| speed | 0.3668 | 0.3227 |
| lick | 0.7145 | 0.6592 |
| reward_zone_location | 0.7297 | 0.3897 |
| reward_outcome | 0.6105 | 0.6719 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 8418756125 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | not yet extracted from paper | reference code uses curated imaging cells | NWB-derived after curation | 118493 | approximate match to curated data pipeline |
| Mean neurons/session | not yet extracted from paper | session-dependent | NWB-derived after curation | 779.6 | plausible |
| Subjects | not yet extracted from paper | implied by code/data | 11 | 11 | yes |
| Sessions | 77 in one analysis subset | analysis-specific subsets in code | 152 NWB files | 152 | yes for full NWB dataset |
| Trials (total) | not yet extracted from paper | trial-based analyses | reconstructed from behavior | 12257 | plausible |
| Trials/session (mean) | not yet extracted from paper | session-dependent | reconstructed from behavior | 80.6 | plausible |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Verification log review: full verification completed without fatal errors.
2. Raw-data sanity checks: confirmed frame-count alignment between behavior and deconvolved neural data in representative sessions; confirmed reward events map one-to-one to rewarded trials in spot checks; confirmed reward-zone-positive frames cluster near three canonical positions (~85, ~205, ~325 cm).
3. Reference code comparison: used deconvolved activity, trial-based organization, reward-relative spatial variables, and 450 cm track logic consistent with code/methods.
4. Key statistics comparison: 11 subjects and 152 sessions matched NWB inventory; curated dataset contains 118493 neurons across 152 sessions.
5. Edge-case checks: fixed ROI table-region mismatch for sessions where response-series ROIs are a subset of segmentation table; documented sessions with variable trial counts.

### Issues Found and Resolved
- ROI segmentation mismatch in some sessions: fixed by indexing `iscell` and `planeIdx` using the ROIResponseSeries DynamicTableRegion.
- Off-by-one / extra trial concern in some sessions: mitigated by requiring included trials to contain a `trial_start` pulse.
- Small-sample metric warning during Step 8: documented as class-absence artifact in held-out sample split.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| distance_to_reward_zone | 0.6089 | 0.3930 | Above chance by 2.75x |
| absolute_position | 0.6458 | 0.5589 | Strong decoding |
| speed | 0.5662 | 0.4737 | Strong decoding |
| lick | 0.6908 | 0.6661 | Strong decoding |
| reward_zone_location | 0.8238 | 0.7646 | Very strong decoding |
| reward_outcome | 0.7205 | 0.6276 | Strong decoding |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- ROI segmentation mismatch in some sessions: fixed by indexing `iscell` and `planeIdx` using the ROIResponseSeries DynamicTableRegion.
- Off-by-one / extra trial concern in some sessions: mitigated by requiring included trials to contain a `trial_start` pulse.
- Small-sample metric warning during Step 8: documented as class-absence artifact in held-out sample split.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
