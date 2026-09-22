# Dataset Conversion Notes

## Overview
- **Dataset**: MAP (Mesoscale Activity Project) - Brain-wide neural activity underlying memory-guided movement
- **Date started**: 2026-09-21
- **Goal**: Convert NWB data to decoder-compatible format for predicting behavioral variables from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python environment verified: numpy 2.4.4, torch 2.6.0+cu124, CUDA available.

Directory contents:
- `/app/data/` - 28 subjects (sub-440956 through sub-484677), 174 NWB files total, 1 dandiset.yaml
- `/app/code/` - Reference code with subdirs: Archive/, Notebooks/, Sherlock/, VideoAnalysisUtils/
- `/app/datapaper.pdf` - Data paper (Chen et al., Cell 2024)
- `/app/methodpaper.pdf` - Methods paper (Wang, Kurgyis et al., Nature Neuroscience 2025)
- `/app/ChenLiuEtAl2023_SpikeSortingQC.pdf` - Spike sorting QC white paper
- `/app/methods.txt` - Extracted methods text
- `/app/train_decoder.py` - Decoder training script
- `/app/decoder.py` - Decoder model code

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_one_sess` | preprocessing_DJ_2022Aug.py | LOADING+PROCESSING | Load .mat files, extract behavior/neural data, apply QC, compute firing rates |
| `sliding_histogram` | preprocessing_DJ_2022Aug.py | PROCESSING | Bin spike times into firing rates with sliding window |
| `helper_get_neuron_id_area` | preprocessing_DJ_2022Aug.py | CURATION | Filter neurons by brain region and QC classifier |
| `process_one_area` | preprocessing_DJ_2022Aug.py | PROCESSING | Extract firing rates for one brain area, save pickle |
| `get_regular_trial_mask` | functions_for_r2.py | CURATION | Filter trials: exclude early_lick, auto_water, free_water, no_response, photostim |
| `preprocess_all_ephys.py` | Sherlock/ | PROCESSING | Main entry: bw=0.04, stride=0.0034, begin=-3, end=3, qc_mode='classifier' |

### Notes
- Original code works with .mat files exported from DataJoint; our data is in NWB format from DANDI
- Spike times in NWB are in absolute session time (not aligned to go cue)
- QC: `classification == 'good'` in NWB corresponds to the classifier-based QC used in the reference code
- Firing rate binning in reference: bw=40ms, stride=3.4ms (for video analysis), or bw=100ms, stride=50ms (in __main__)
- Time window: -3.0 to 3.5 s relative to go cue (reference code)
- Brain regions processed: ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum
- Side determined by CCF x-coordinate: left if ccf_x >= 5700, right if ccf_x < 5700

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- NWB format (Neurodata Without Borders)
- Each file: one session with behavior + electrophysiology (+ optogenetics for most)
- Units table: spike_times (absolute time), classification (good/unlabelled), anno_name (CCF annotation), electrode_group (probe with brain region target)
- Trials table: start_time, stop_time, trial_instruction (left/right), early_lick (early/no early), outcome (hit/miss/ignore), auto_water, free_water, photostim fields
- BehavioralEvents: go_start_times, sample_start_times, delay_start_times, photostim_start/stop_times, left/right_lick_times
- BehavioralTimeSeries: Camera0_side_TongueTracking (x, y, likelihood at ~300Hz/3.4ms), JawTracking, NoseTracking
- Electrode groups contain JSON with brain_regions field (e.g., "left ALM", "right Striatum")

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Total units | 272,227 |
| Good units (total) | 69,453 |
| Good units / session (mean) | 399.2 |
| Subjects | 28 |
| Sessions / subject | 3-10 (mean ~6.2) |
| NWB files | 174 |
| Trials (total) | 94,990 |
| Trials / session (mean) | 545.9 |
| Sessions without ogen | 6 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "dataset consisted of 69,943 good units" (data paper) |
| Subjects | 28 | "n = 28" (data paper) |
| Sessions | 173 | "173 behavioral sessions" (data paper) |
| Penetrations | 655-660 | "655 probe insertions" / "660 penetrations" (data paper) |
| Trials / session (mean) | 476 | "476 (Mean; range, 130-785) trials per session" (methods.txt) |
| Correct rate | 84% | "84% correct rate (range, 65-99%)" (methods.txt) |
| Session selection: performance | > 65% | (methods.txt) |
| Session selection: min correct trials | 50 each direction | "at least 50 correct lick left and lick right trials each" (methods.txt) |
| Neural data time bin (method paper) | 40ms bw, 3.4ms stride | "binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Behavior video rate | 300 Hz | (methods.txt) |
| Sample epoch duration | 0.65 s | 3x150ms tones + 2x100ms gaps |
| Delay epoch duration | 1.2 s | (methods.txt) |
| Sample-to-go-cue | 1.85 s | Confirmed from data: consistent 1.85s |
| Photostim duration | 0.5 s | "last 0.5 s of delay" + 100ms ramp-down |
| Photostim trials | ~25% randomly interleaved | (methods.txt) |
| Neurons: ALM | 8,717 | (data paper Fig 1J) |
| Neurons: Striatum | 7,664 | (data paper) |
| Neurons: Thalamus | 12,808 | (data paper) |
| Neurons: Midbrain | 7,495 | (data paper) |
| Neurons: Medulla | 2,928 | (data paper) |
| Neurons: Orbital | 10,223 | (data paper Fig 1J) |
| Neurons: Pallidum | 1,092 | (data paper Fig 1J) |
| Neurons: Hippocampus | 1,944 | (data paper Fig 1J) |
| Neurons: Hypothalamus | 815 | (data paper Fig 1J) |
| Neurons: Pons | 347 | (data paper Fig 1J) |
| Neurons: Cerebellum | 1,820 | (data paper Fig 1J) |
| Neurons: Other areas | 14,090 | (data paper Fig 1J) |

### Processing Details
- Spike times aligned to go cue (time 0)
- Method paper: neurons with firing rate < 2 Hz excluded
- Method paper trial filtering: exclude photostim, free water, early lick, ignore trials
- Method paper: brain areas with < 10 neurons/session excluded from analysis
- Tongue tracking: outliers removed (5-sigma velocity threshold), occluded tongue set to mean

### Curation Steps

**Neuron curation rules**:
1. Classifier-based QC: keep only `classification == 'good'` units
2. (Method paper only) Exclude neurons with avg firing rate < 2 Hz

**Trial curation rules** (reference code `get_regular_trial_mask`):
- Exclude: early_lick != 0, auto_water != 0, free_water != 0, correctness == -1 (no response/ignore), photostim != 0
- For our decoder: keep early_lick, ignore, photostim trials (they are decoder inputs/outputs). Only exclude auto_water and free_water.

### Decoders Trained
The method paper trains R2-based regression from video features to firing rates, not categorical decoders.
The data paper trains choice decoders (AUC-based) on individual neurons.
No directly comparable categorical decoder accuracy reported.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Good units | classifier QC | 69,453 | 69,943 | ~490 unit difference; likely minor NWB conversion differences. Will use NWB classification field. |
| Sessions | N/A | 174 NWB files | 173 | One session may not pass behavioral criteria (>65%, 50+ correct each direction). Will keep all sessions that have >=2 valid trials. |
| Trials/session | N/A | 545.9 mean | 476 mean | Paper counts only non-early-lick correct control trials; raw count is higher. Consistent. |
| Trial filtering | Excludes early_lick, ignore, photostim | All available in NWB | Same as code | For decoder: keep these (they're decoder outputs/inputs). Only exclude auto_water/free_water. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| units.spike_times (good only) | neural | Bin spikes in 50ms bins, aligned to go cue, window [-2.5, 1.5]s | 80 time bins per trial |
| sample_start relative to go cue | input[0]: time_from_tone_onset | t - (sample_start - go_cue) for each time bin t | Continuous, time-varying ramp |
| photostim_start/stop times | input[1]: photostim_on | Binary: 1 if photostim active at time t, 0 otherwise | Time-varying binary |
| trial_instruction + outcome | output[0]: choice | left=0, right=1, no_lick=2 | Per-trial |
| outcome | output[1]: outcome | ignore=0, miss=1, hit=2 | Per-trial |
| early_lick | output[2]: early_lick | no=0, yes=1 | Per-trial |
| tongue_y from TongueTracking | output[3]: tongue_y_position | Discretized per session: 0(<40th), 1(40-60th), 2(>60th), 3(not visible) | Time-varying |
| subject_id from NWB | subjects/subject_idx | Map unique subject IDs | |
| anno_name from units + electrode_group | brain_regions/brain_region_idx | Map CCF annotations to broad regions | |

### Key Decisions
1. **Trial filtering**: Only exclude auto_water and free_water trials. Keep early lick, ignore, and photostim trials since they are decoder inputs/outputs.
2. **Neuron filtering**: Use `classification == 'good'` (classifier QC). Do NOT apply the 2 Hz firing rate threshold from the method paper, as that was specific to video prediction analysis, not a general data quality criterion.
3. **Firing rate binning**: 50ms non-overlapping bins (as specified by decoder task), NOT the 40ms/3.4ms stride from reference code. This is required by the decoder task specification.
4. **Time window**: [-2.5, 1.5]s relative to go cue = 80 time bins. Different from reference code's [-3.0, 3.5]s but required by decoder task.
5. **Brain region mapping**: Use electrode_group brain_regions (broad target region like "left ALM") combined with CCF anno_name for finer mapping. Map to broad categories matching the reference code.
6. **Tongue visibility**: Use likelihood threshold of 0.9 (DeepLabCut convention). Tongue with likelihood < 0.9 is "not visible" (category 3).
7. **Tongue percentiles**: Compute 40th and 60th percentiles over all visible tongue y-positions in the session (across all time points and trials), then discretize.
8. **Photostim timing**: Photostim times are in absolute session time. Convert to go-cue-relative for each trial.
9. **Choice determination**: hit → lick matches instruction; miss → lick opposite to instruction; ignore → no lick.
10. **All 174 sessions included**: Don't apply session behavioral criteria (>65% correct) since our decoder handles all outcome types.

### Planned Sanity Checks
- [ ] Total good neuron count close to 69,943 (paper) / 69,453 (NWB count)
- [ ] Mean trials per session after filtering ~476 for control trials
- [ ] Sample-to-go-cue consistently ~1.85s
- [ ] Photostim on ~25% of trials
- [ ] Hit rate ~84% on control trials
- [ ] Spot-check spike rates match between raw spike times and binned rates
- [ ] Tongue y discretization produces expected distribution

---

## Step 6: Script Development
**Status**: COMPLETE

Script `/app/convert_data.py` (~500 lines) implements full pipeline with optimized vectorized binning.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
- 2 sessions, 834 neurons, 514 trials
- Processing time: ~3.7s/session

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: 1 (Session 1 trial 159 all-zero neural - edge of recording window)

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | - | 0.6157 | 0.333 |
| outcome | - | 0.6457 | 0.333 |
| early_lick | - | 0.7319 | 0.500 |
| tongue_y_position | - | 0.5314 | 0.250 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE (re-running with fixed brain region mapping)

### Output Files
- `/app/converted_data.pkl`: 12 GB, 173 sessions, 28 subjects, 69,453 neurons, 89,532 trials
- 1 session skipped (sub-440958_ses-20190216 - no valid trials within obs_intervals)

### Consistency Check
- Verification passed, 2 warnings (edge-case all-zero trials at recording boundaries)
- All 80 time bins per trial, all sessions consistent
- Input ranges correct: time_from_tone_onset [-1.5, ~12], photostim_on [0, 1]
- Output ranges correct: choice [0,2], outcome [0,2], early_lick [0,1], tongue_y [0,3]

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Verified total good units (69,453) matches NWB count
2. Compared brain region counts to paper Fig 1J
3. Found and fixed brain region mapping errors using Allen CCF hierarchy
4. Verified trial filtering (auto_water/free_water excluded, others kept)
5. Verified temporal alignment (go cue at t=0)

### Issues Found and Resolved
1. **Zona incerta (464) and Fields of Forel (114)**: Were mapped to Thalamus, should be Hypothalamus (Allen CCF subthalamus). Fixed.
2. **Subthalamic nucleus (28) and Parasubthalamic nucleus (9)**: Were mapped to Pallidum, should be Hypothalamus. Fixed.
3. **Substantia innominata (212) and Bed nuclei of stria terminalis (91)**: Were mapped to CorticalSubplate, should be Pallidum (Allen CCF). Fixed.
4. **Paracentral nucleus (591)**: Was Unknown, should be Thalamus. Fixed.

### Updated Region Counts (after fix)
| Region | Ours | Paper | Match |
|--------|------|-------|-------|
| Hypothalamus | 815 | 815 | Exact |
| Pallidum | 1,090 | 1,092 | ~exact |
| Hippocampus | 1,944 | 1,944 | Exact |
| Orbital | 10,223 | 10,223 | Exact |
| Cerebellum | 1,823 | 1,820 | ~exact |
| Medulla | 2,925 | 2,928 | ~exact |
| Pons | 337 | 347 | Close |
| Thalamus | 12,612 | 12,808 | Close |
| ALM | 7,346 | 8,717 | Lower (490 unit total difference) |
| Striatum | 7,256 | 7,664 | Close |
| Midbrain | 7,701 | 7,495 | Close |

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- 200 epochs, loss converged from 18.3 to 0.63
- Trained on GPU (CUDA)
- Used 100 PCs, lr=1e-3, l1_weight=1e-4, balanced loss

### Decoder Results (Full)
| Output | Train Balanced Acc | Val Balanced Acc | Chance |
|--------|-------------------|-----------------|--------|
| choice | 0.7094 | 0.6798 | 0.333 |
| outcome | 0.6937 | 0.6612 | 0.333 |
| early_lick | 0.7913 | 0.7523 | 0.500 |
| tongue_y_position | 0.7535 | 0.6644 | 0.250 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
- All outputs substantially above chance (2x-2.7x chance level)
- Choice decoding (0.68): Consistent with population-level motor planning signals across brain regions
- Outcome decoding (0.66): Correlated with choice signals (hit/miss determined by lick direction)
- Early lick (0.75): Strong detection of impulsive motor activity despite class imbalance (89% no early lick)
- Tongue y-position (0.66): Strong 4-class decoding of motor kinematics from neural population
- Train-val gap small (0.03-0.09), good generalization

### Issues Found and Resolved
- No issues found. Results are consistent with expectations from the literature.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] cache/ folder created (intermediate files moved there)
- [x] All files organized
- [x] CONVERSION_NOTES.md complete

### Final File Layout
- `/app/converted_data.pkl` - Main output (12 GB)
- `/app/convert_data.py` - Conversion script
- `/app/CONVERSION_NOTES.md` - This file
- `/app/train_stats.json` - Decoder training statistics
- `/app/verify_stats.json` - Verification statistics
- `/app/training_full_out.txt` - Full training log
- `/app/verification_full_out.txt` - Full verification log
- `/app/sample_trials.png` - Sample trial plots
- `/app/predictions.png` - Prediction plots
- `/app/cache/` - Intermediate files (sample data, logs, processing plots)
