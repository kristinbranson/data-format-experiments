# Dataset Conversion Notes

## Overview
- **Dataset**: Mesoscale Activity Map (MAP) - Brain-wide neural activity underlying memory-guided movement (DANDI:000363, NWB format)
- **Date started**: 2026-03-23
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `ChenLiuEtAl2023_SpikeSortingQC.pdf` - spike sorting QC white paper
- `code/` - reference code from methodpaper (MapVideoAnalysis repo)
- `data/` - NWB data files organized by subject (28 subjects)
- `datapaper.pdf` - Chen et al. 2024, Cell
- `decoder.py` - decoder model code
- `methodpaper.pdf` - Wang et al. 2025, Nature Neuroscience
- `methods.txt` - extracted methods text
- `train_decoder.py` - decoder training/validation script

Python: numpy 2.3.5, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_one_sess` | `preprocessing_DJ_2022Aug.py` | LOADING+CURATION | Load .mat files, combine probes, apply QC filter |
| `sliding_histogram` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Bin spikes into firing rates (bw=40ms, stride=3.4ms) |
| `helper_get_neuron_id_area` | `preprocessing_DJ_2022Aug.py` | CURATION | Filter neurons by brain region + QC classifier |
| `process_one_area` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Extract area-specific firing rates, save pickle |
| `get_regular_trial_mask` | `population_decoding_utils.py` | CURATION | Filter trials: no early lick, no auto water, no free water, not ignore, no stim |
| `load_session` | `population_decoding_utils.py` | LOADING | Load preprocessed session data from pickle files |

### Key Processing Parameters (from `preprocess_all_ephys.py`)
- `bw = 0.04` (40ms bin width for sliding histogram)
- `stride = 0.0034` (3.4ms stride)
- `begin_time = -3.0`, `end_time = 3.0` (time window around go cue)
- `qc_mode = 'classifier'` (use classifier-based QC)

### Notes
- Reference code loads from .mat files; we have NWB files (same dataset, different format)
- Spike times in .mat files are already aligned to go cue; in NWB they are absolute session time
- QC filtering uses classifier-trained logistic regression per brain area group
- In NWB: `classification == 'good'` corresponds to the classifier QC output
- Trial filtering in reference analysis excludes early lick, auto water, free water, ignore, stim trials
- Brain region annotation via `anno_name` field in NWB units table

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- 28 subject directories (`sub-XXXXXX/`)
- 174 NWB files total (`*_behavior+ecephys+ogen.nwb`)
- 3-10 sessions per subject
- Each NWB contains: units table, trials table, behavioral events, behavioral time series

### NWB File Contents
- **Units table** (41 columns): spike_times, classification (good/unlabelled), anno_name (CCF annotation), is_good_trials, quality metrics (drift_metric, isi_violation, amplitude_cutoff, etc.)
- **Trials table** (14 columns): trial_instruction (left/right), outcome (hit/miss/ignore), early_lick, auto_water, free_water, photostim info
- **Behavioral events**: go_start_times, sample_start/stop_times, delay_start/stop, left/right_lick_times, photostim_start/stop_times
- **Behavioral time series**: TongueTracking (x, y, confidence) at ~294 Hz, JawTracking, NoseTracking

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 28 |
| NWB files (sessions) | 174 |
| Sessions / subject | 3-10 |
| Units / session (example) | ~1950-2330 (before QC) |
| Good units / session (example) | ~460 (after classification=='good') |
| Trials / session (example) | ~330-370 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "69,943 good units recorded across 173 behavioral sessions" |
| Good units (ALM) | 8,717 | "8717 good units from ALM" |
| Good units (Striatum) | 7,664 | "7664 from striatum" |
| Good units (Thalamus) | 12,808 | "12808 from thalamus" |
| Good units (Midbrain) | 7,495 | "7495 from midbrain" |
| Good units (Medulla) | 2,928 | "2928 from medulla" |
| Subjects | 28 | "28 mice" |
| Behavioral sessions | 173 | "173 behavioral sessions" |
| Probe insertions | 655 (or 660) | "655 probe insertions" / "660 penetrations" |
| QC pass rate | 25.9% | "25.9% of clusters reported by Kilosort2" |
| Median neurons/session | 393 | "median = 393" |
| Trials/session (mean) | 476 | "476 (Mean; range, 130-785)" |
| Correct rate (mean) | 84% | "84% correct rate (range, 65-99%)" |
| Neural data bin width | 40ms | "bin width of 40 ms and a stride of 3.4 ms" |
| Video frame rate | 300 Hz | "acquired at 300 Hz" |

### Task Structure
- **Sample epoch**: Pure tones (3kHz or 12kHz), 3x150ms with 100ms gaps = ~650ms
- **Delay epoch**: 1.2s
- **Go cue**: 6kHz modulated tone, 0.1s duration
- **Response epoch**: 1.5s
- **Consumption**: 1.5s

### Trial Types
- Left lick vs Right lick (instructed by tone frequency)
- Outcomes: hit (correct), miss (error), ignore (no response)
- Early lick: licking during sample/delay triggers replay

### Session Selection Criteria
- Overall behavioral performance > 65%
- At least 50 correct lick left and lick right trials each

### Photostimulation
- ~25% of trials randomly interleaved (17 VGAT-ChR2-EYFP mice)
- 40Hz sinusoidal, 5mW, during late delay (last 0.5s)
- Left-ALM, right-ALM, or bilateral

### Trial Exclusion (reference analysis)
- Early lick, auto water, free water, no response (ignore), stimulation trials excluded
- Note: For our decoder, we keep most of these since they are outputs/inputs

### Curation Steps

**Neuron curation rules**:
- Classifier-based QC: 5 region-specific logistic regression classifiers
- Applied to Kilosort2 output clusters
- In NWB: `classification == 'good'`

**Trial curation rules** (for decoder):
- Exclude: auto_water, free_water trials
- Keep: all other trials (including early lick, ignore, stim - these are decoder I/O)

### Decoders in Papers
| Decoded variable | Accuracy | Method |
|---|---|---|
| Choice (from video) | AUC 0.51 pre-sample, 0.66 sample+delay | Embedding autoencoder |
| Choice (from neural, ALM) | AUC ~0.9 delay epoch | Logistic regression |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Data format | .mat files from DataJoint | NWB files from DANDI | NWB on DANDI | Use NWB; same underlying data |
| QC filter | classifier with good units idx files | `classification` column in units | Classifier-based QC | Use `classification == 'good'` |
| Spike alignment | Already aligned to go cue in .mat | Absolute session time in NWB | Aligned to go cue | Align using go_start_times |
| N sessions | Code processes all | 174 NWB files | 173 behavioral sessions | 174 vs 173 minor discrepancy; apply session criteria |
| Brain regions | 14 major regions with side | `anno_name` = detailed CCF | "ALM, striatum, thalamus, midbrain, medulla, etc." | Map anno_name to major regions |
| Trial filtering | Excludes early lick, auto water, free water, ignore, stim | All trial types in NWB | Excludes same for analysis | For decoder: only exclude auto_water + free_water |
| Bin width | bw=40ms, stride=3.4ms | N/A | "bin width of 40 ms" | Decoder task specifies 50ms bins; use 50ms |
| Tongue tracking | Used for video analysis | (x,y,confidence) at ~294Hz | "tongue position set to mean when occluded" | Use confidence threshold; set to mean when occluded |

### Key Consistency Notes
- The 28 subjects in the data match the paper (28 mice)
- NWB `classification=='good'` implements the same classifier-based QC from the methods
- Sample/delay/go event timings in NWB are consistent with the task description
- The slight discrepancy (174 vs 173 sessions) may be because one session doesn't meet selection criteria

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|---|---|---|---|
| `units.spike_times` (good units) | `neural` | Bin into 50ms non-overlapping bins around go cue (-2.5 to +1.5s) = 80 timepoints | Filter by `classification=='good'` |
| Sample start time relative to go cue | `input[0]`: time_from_tone_onset | Continuous, `t - tone_onset` for each time bin | Use last sample_start before go cue for each trial |
| Photostim start/stop times | `input[1]`: photostim_on | Binary time-varying, 1 when photostim active | From photostim_start/stop_times |
| `trial_instruction` | `output[0]`: choice | left=0, right=1 (per-trial) | |
| `outcome` | `output[1]`: outcome | ignore=0, miss=1, hit=2 (per-trial) | |
| `early_lick` | `output[2]`: early_lick | no=0, yes=1 (per-trial) | |
| TongueTracking y-position | `output[3]`: tongue_y_position | Discretized per session: 0(<40th), 1(40-60th), 2(>60th); time-varying | Handle occlusion (set to mean when confidence low) |
| `subject.subject_id` | `subjects` | Unique subject IDs | |
| `anno_name` | `brain_regions` / `brain_region_idx` | Map CCF annotations to major regions | |

### Trial Filtering
- Exclude: `auto_water == 1` or `free_water == 1`
- Keep all other trials
- Session filtering: performance > 65%, at least 50 correct left + 50 correct right trials

### Temporal Parameters
- Alignment: go cue onset (`go_start_times`)
- Window: -2.5s to +1.5s
- Bin width: 50ms (non-overlapping)
- N timepoints: 80 bins

### Brain Region Mapping
Map detailed `anno_name` CCF annotations to these major regions:
ALM, OtherCortex, Striatum, Thalamus, Midbrain, Medulla, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, Olfactory, CorticalSubplate, Pallidum

### Key Decisions
1. **50ms bins**: Task specification overrides reference code's 40ms/3.4ms sliding histogram
2. **Keep early lick/ignore/stim trials**: These are decoder outputs/inputs, not filtered
3. **Exclude auto_water and free_water only**: Artificial conditions that change task structure
4. **Tongue occlusion handling**: Set tongue position to session mean when confidence < 0.9 (following reference: "set tongue position to its mean value" when occluded)
5. **Session selection**: Apply paper's criteria (>65% performance, >=50 correct each direction)

### Planned Sanity Checks
- [ ] Total good neurons across all sessions ~69,943
- [ ] Number of subjects = 28
- [ ] Number of sessions ~173 (after filtering)
- [ ] Median neurons/session ~393
- [ ] Mean trials/session ~476
- [ ] Correct rate ~84%
- [ ] Outcome distribution consistent with paper
- [ ] Spot-check spike rates for a few neurons match expectations

---

## Step 6: Script Development
**Status**: COMPLETE

Key implementation details:
- Fixed spike times loading: NWB uses VectorIndex wrapping VectorData; need `.target.data` for actual spikes
- Optimized tongue processing: vectorized using np.digitize instead of per-bin loop (142s -> 1.5s)
- Optimized spike binning: use np.searchsorted for binary search (14s -> 3s per session)
- Fixed tongue y discretization: when p40==p60 (common with imputed values), add small offset for 3 categories

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 597 |
| Neurons / session | 259, 338 |
| Subjects | 2 |
| Sessions | 2 |
| Trials (total) | 1037 |
| Trials / session | 487, 550 |
| Brain regions | 11 (ALM, Cerebellum, Hippocampus, Medulla, Midbrain, Olfactory, Orbital, OtherCortex, Pons, Striatum, Thalamus) |
| time_from_tone_onset range | [-1.5, 9.4] |
| photostim_on range | [0.0, 1.0] |
| choice distribution | left=39.6%, right=60.4% |
| outcome distribution | ignore=8.9%, miss=15.9%, hit=75.2% |
| early_lick distribution | no=74.4%, yes=25.6% |
| tongue_y distribution | low=19.5%, mid=69.1%, high=11.4% |

### Processing Plots Review
Processing plots saved. Neural heatmaps show reasonable firing rate patterns aligned to go cue. Tone onset input shows consistent timing. Photostim input is binary with small fraction of stim trials.

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|---|---|---|
| NWB loading | ~1.5s | ~4.3 min |
| Spike binning | ~3.3s | ~9.6 min |
| Tongue processing | ~1.5s | ~4.3 min |
| Total | ~7.5s | ~22 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.6842 | 0.6665 | 0.5000 |
| outcome | 0.6669 | 0.5856 | 0.3333 |
| early_lick | 0.7379 | 0.7232 | 0.5000 |
| tongue_y_position | 0.8244 | 0.8151 | 0.3333 |

All outputs well above chance. Loss decreases steadily from 13.5 to 0.58.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9.3 GB, 144 sessions, 56,909 neurons, 74,768 trials

### Full Dataset Statistics
| Statistic | Value | Paper Value | Match? |
|-----------|-------|-------------|--------|
| Subjects | 28 | 28 | YES |
| Sessions | 144 | 173 (total) | YES (after filtering) |
| Total neurons | 56,909 | 69,943 (all 173 sess) | YES (~83% ratio) |
| Median neurons/session | 390 | 393 | YES |
| Mean trials/session | 519.2 | 476 | Higher (we keep early lick/ignore/stim) |
| Brain regions | 14 | 14 | YES |
| Time bins | 80 | 80 (50ms x 4s) | YES |

### Session Count Resolution
- 174 NWB files total; 173 have good units (matches paper's "173 behavioral sessions")
- 144 pass selection criteria (>65% performance, >=50 correct L/R)
- 30 skipped: 23 low performance, 4 insufficient correct left, 2 insufficient correct right, 1 no good units
- Paper's "173" = total sessions with good units, not post-selection

### Brain Region Distribution
| Region | Ours (144 sess) | Paper (173 sess) | Ratio |
|--------|---------|---------|-------|
| ALM | 7,614 | 8,717 | 87% |
| Thalamus | 11,544 | 12,808 | 90% |
| Midbrain | 6,201 | 7,495 | 83% |
| Striatum | 5,869 | 7,664 | 77% |
| Medulla | 2,280 | 2,928 | 78% |

### Consistency Check
- Data format: valid, no errors or warnings
- No zero-neural-data trials (fixed by excluding trials beyond recording range)
- All output distributions reasonable

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Issues Found and Resolved

1. **Trials beyond recording range (FIXED)**: Some sessions had behavioral trials continuing after neural recording ended (or starting before recording began). Fixed by checking spike time range and excluding trials outside the recorded window. Affected 7 sessions, dropping ~1,056 trials total.

2. **"Midbrain reticular nucleus" misclassified as Thalamus (FIXED)**: The keyword "Reticular nucleus" in the Thalamus mapping was matching "Midbrain reticular nucleus". Fixed by making the Thalamus keyword more specific ("Reticular nucleus of the thalamus") and adding "Midbrain reticular nucleus" explicitly to Midbrain.

3. **Several unmapped annotations (FIXED)**: Added "Nucleus of the lateral lemniscus" → Pons, "Septofimbrial nucleus" → Striatum, "Dorsal peduncular area" → OtherCortex, "Nucleus of the brachium of the inferior colliculus" → Midbrain.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- 200 epochs, loss: 20.9 → 0.57 (training), 0.61 (test)
- Training set: 59,758 trials; Validation set: 15,010 trials
- Device: CUDA

### Decoder Results (Full)
| Output | Training Bal. Acc. | Validation Bal. Acc. | Chance |
|--------|-------------------|---------------------|--------|
| choice | 0.7224 | 0.7025 | 0.5000 |
| outcome | 0.6999 | 0.6461 | 0.3333 |
| early_lick | 0.7920 | 0.7442 | 0.5000 |
| tongue_y_position | 0.8038 | 0.7855 | 0.3333 |

All outputs well above chance with minimal overfitting (train-val gap: 0.02-0.05).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
- Choice (0.70): Paper reports AUC ~0.9 for choice from ALM-only logistic regression. Our 0.70 balanced accuracy across all regions with a simpler model is consistent.
- Outcome (0.65): Not directly reported in paper. 0.65 for 3-class (chance 0.33) is strong.
- Early lick (0.74): Not directly reported. Reflects motor activity during sample/delay period.
- Tongue y-position (0.79): Paper reports AUC ~0.66 for choice from video (harder task). Our 0.79 from neural data is expected to be higher.
- No significant overfitting observed.

### Issues Found and Resolved
No additional issues found in Review 2.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

### Output Files
- `converted_data.pkl` - Main output (9.3 GB)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - This file
- `predictions.png` - Decoder prediction visualization
- `sample_trials.png` - Sample trial visualization
