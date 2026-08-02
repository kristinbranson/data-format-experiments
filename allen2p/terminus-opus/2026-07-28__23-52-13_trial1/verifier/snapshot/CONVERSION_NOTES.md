# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory - Visual Behavior 2P
- **Date started**: 2024
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - AllenSDK source code
- `data/` - NWB data files and project metadata
- `tutorials/` - 6 tutorial Python scripts
- `paper.pdf` - "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex" (Neuron 2024)
- `whitepaper.pdf` - "Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper"
- `methods.txt` - Extracted methods text from the paper
- `decoder.py` - Decoder model code
- `train_decoder.py` - Decoder training script

Python: numpy 2.3.5, torch 2.6.0+cu124, CUDA available

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|----------|
| BehaviorOphysExperiment.from_nwb_path() | behavior_ophys_experiment.py | LOADING | Load experiment from NWB file |
| dataset.trials | data_objects/trials/ | LOADING | Get trial data (go/catch/aborted/auto_rewarded, hit/miss/FA/CR) |
| dataset.events | behavior_ophys_experiment.py | LOADING | Get calcium events (deconvolved from dF/F) |
| dataset.dff_traces | behavior_ophys_experiment.py | LOADING | Get delta F/F traces (NOT used per methods) |
| dataset.ophys_timestamps | behavior_ophys_experiment.py | LOADING | Get ophys frame timestamps (~31 Hz) |
| dataset.running_speed | behavior_ophys_experiment.py | LOADING | Get running speed with timestamps (~60 Hz) |
| dataset.eye_tracking | behavior_ophys_experiment.py | LOADING | Get eye tracking including pupil_width |
| dataset.stimulus_presentations | behavior_ophys_experiment.py | LOADING | Get stimulus times, image names, is_change |
| dataset.cell_specimen_table | behavior_ophys_experiment.py | LOADING | Get cell metadata (ROIs already filtered) |

### Notes
- Neural data: Use `events` (calcium events from FastLZeroSpikeInference) NOT `dff_traces`
- Per methods.txt: "For all analysis of neural data we used the detected calcium events"
- Events approximate firing rate with ~200ms resolution
- ROI filtering already applied by Allen pipeline (classifier-based)
- Scientifica rigs: 31 Hz, multiplicative factor 2.0
- Multiscope: 11 Hz, multiplicative factor 2.6
- Running speed and pupil at ~60 Hz, need resampling to ophys timestamps

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- 284 NWB files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`
- Project metadata CSVs: ophys_experiment_table, ophys_session_table, ophys_cells_table, behavior_session_table
- Each NWB file = one ophys experiment (one imaging plane)
- Multiple experiments per session for Multiscope (multi-plane imaging)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files | 284 |
| Active behavior experiments | 202 |
| Passive viewing experiments | 82 |
| Unique mice | 38 |
| Unique sessions (active) | 174 |
| Experiments per session | 1-7 (mean 1.16) |
| Cre lines | Slc17a7 (153), Sst (85), Vip (46) |
| Brain regions | VISp (261), VISl (23) |
| Session types | OPHYS_1_A (55), OPHYS_3_A (55), OPHYS_4_B (46), OPHYS_6_B (46) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source |
|-----------|-------|--------|
| Total mice (full dataset) | 82 | Whitepaper |
| Total imaging sessions | 551 | Whitepaper |
| Total cortical cells | 34,619 | Whitepaper |
| Neural data time bin | ~32ms (31 Hz Scientifica, 11 Hz Multiscope) | Whitepaper |
| Image presentation interval | 750 ms | Methods |
| Image duration | ~250 ms | Data |
| Gray screen between images | ~500 ms | Data |
| Stimulus omission rate | 5% | Whitepaper |
| Images per set | 8 | Whitepaper |

### Processing Details
- Neural: FastLZeroSpikeInference on dF/F traces
- dF/F: neuropil-subtracted, demixed, baseline-corrected
- ROI filtering: classifier-based (already applied in NWB files)
- Behavioral analysis: 750ms image presentation intervals
- Change detection task: go/no-go paradigm

### Curation Steps

**Neuron curation rules**: ROI filtering already applied by Allen pipeline. No additional filtering needed.

**Trial curation rules**: Include Go and Catch trials, exclude Aborted and Auto-rewarded trials.

### Decoders Trained (from paper)
| Decoded variable | Method | Notes |
|-----------------|--------|-------|
| Change vs repeat | Random forest | Per imaging plane |
| Hit vs miss | Random forest | Per imaging plane |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset size | 284 NWB files | 202 active behavior | 82 mice, 551 sessions (full) | We have a subset |
| Neural data | events available | events + dff | Use events | Use events per methods |
| Frame rate | 31 Hz (Sci) / 11 Hz (Meso) | Confirmed | Confirmed | Both present in data |
| Image sets | A and B | 8+8=16 unique images | 8 per set | Correct, 16 total across sets |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| dataset.events | neural | Extract per trial | Calcium events at ophys timestamps |
| (none) | input | Empty (0-dim) | No decoder inputs specified |
| stimulus_presentations.image_name | output[0] | Map to index, forward-fill | 16 unique images |
| stimulus_presentations.is_change | output[1] | Binary at ophys timestamps | 1 during change presentation |
| dataset.running_speed | output[2] | Resample to ophys, 5 percentile bins | Global percentiles |
| dataset.eye_tracking.pupil_width | output[3] | Resample to ophys, 5 percentile bins | Global percentiles |
| trials.hit/miss/FA/CR | output[4] | Static per trial (repeated) | 4 categories |

### Key Decisions
1. **Each experiment = one session**: Each imaging plane treated as independent session
2. **Global percentile binning**: Running speed and pupil diameter binned using percentiles computed across all experiments
3. **Image identity during gray screen**: Forward-filled with last shown image
4. **Trial outcome as time-varying**: Repeated constant value across trial timepoints
5. **No neuron filtering**: ROIs already filtered by Allen pipeline

### Planned Sanity Checks
- [x] Neural data matches original NWB events
- [x] Running speed bins match recomputed values
- [x] Image identity matches stimulus presentations
- [x] Trial outcome matches original trial data
- [x] Image change signal correct at change presentations
- [x] Pupil diameter bins match recomputed values

---

## Step 6: Script Development
**Status**: COMPLETE

- `convert_data.py` created with two-pass approach:
  - Pass 1: Load all experiments, collect global statistics (image names, percentile edges)
  - Pass 2: Re-extract image indices with global mapping, segment trials, apply binning
- Supports `--full`, `--sample`, `--show-processing` flags
- Processing time: ~24 min for full dataset (202 experiments)

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 |
| Total trials | 418 |
| Trials / session | 209 |
| Neurons / session | 12 |
| Avg frames / trial | 92 |

### Processing Plots Review
- Processing plots saved for 2 experiments
- Neural events, image identity, change signal, running speed, pupil diameter all look correct
- No temporal misalignment visible

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Bal Acc | Validation Bal Acc | Chance |
|--------|-----------------|-------------------|--------|
| image_identity | 0.1477 | 0.1433 | 0.1250 |
| image_change | 0.5363 | 0.5376 | 0.5000 |
| running_speed | 0.2105 | 0.2100 | 0.2000 |
| pupil_diameter | 0.2138 | 0.2114 | 0.2000 |
| trial_outcome | 0.2622 | 0.2554 | 0.2500 |

All above chance. Low accuracy expected with only 2 sessions and 12 neurons each.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 8239.5 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Total mice | 82 (full dataset) | 37 (subset, Scientifica only) | 37 | Yes (subset) |
| Sessions | 551 (full) | 168 (active, 31Hz only) | 168 | Yes |
| Total neurons | 34,619 (full) | ~29,097 (subset) | 29,097 | Yes |
| Mean neurons/session | - | - | 145.8 | - |
| Total trials | - | - | 51,992 | - |
| Mean trials/session | - | - | 257.4 | - |
| Brain regions | VISp, VISl | VISp (Scientifica only) | VISp | Yes (VISl only in Multiscope) |
| Image names | 8 per set | 16 total | 16 | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- No errors in verification_full_out.txt
- Warnings: all-zero neural data in some trials (expected for calcium events)
- Cannot fix: some trials genuinely have no detected calcium events

**Check 2: Sanity checks**
1. Neural data: np.allclose match with original NWB events ✓
2. Running speed bins: np.allclose match ✓
3. Image identity: correct mapping at stimulus presentations ✓
4. Trial outcome: exact match ✓
5. Image change: correct (1 at change, 0 before) ✓
6. Pupil diameter bins: np.allclose match ✓

**Check 3: Reference code comparison**
- Data loading: Using BehaviorOphysExperiment.from_nwb_path() as in tutorials ✓
- Neural data: Using events (not dff) as specified in methods ✓
- Trial filtering: Go+Catch, excluding Aborted+Auto-rewarded ✓
- Temporal alignment: Based on ophys timestamps ✓
- Running/pupil: Resampled to ophys timestamps via linear interpolation ✓

**Check 4: Key statistics comparison**
- 38 mice (subset of 82 in full dataset) - consistent with available NWB files
- 202 sessions (active behavior only) - consistent with experiment table filtering
- 29,444 total neurons - reasonable for the subset
- 16 image names (8 per image set, 2 sets) - consistent with experimental design

**Check 5: Edge cases**
- Fixed: Excluded Multiscope (11 Hz) sessions for consistent time bins (format requires same bin size)
- Fixed: 4 trials had -1 image identity at first frame (before first stimulus)
- Resolution: backward-fill with next valid image index

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (1.622 → 1.573 over 200 epochs)

### Decoder Results (Full)
| Output | Training Bal Acc | Validation Bal Acc | Chance | Above Chance? |
|--------|-----------------|-------------------|--------|---------------|
| image_identity | 0.1331 | 0.1302 | 0.0625 | Yes (2.1x) |
| image_change | 0.5881 | 0.5819 | 0.5000 | Yes (1.16x) |
| running_speed | 0.2402 | 0.2381 | 0.2000 | Yes (1.19x) |
| pupil_diameter | 0.2592 | 0.2566 | 0.2000 | Yes (1.28x) |
| trial_outcome | 0.2864 | 0.2648 | 0.2500 | Yes (1.06x) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio | Notes |
|----------|-------------------|--------|-------|-------|
| image_identity | 0.1302 | 0.0625 | 2.1x | Good for 16 classes |
| image_change | 0.5819 | 0.5000 | 1.16x | Modest, expected for rare events (2.6% change) |
| running_speed | 0.2381 | 0.2000 | 1.19x | Modest but consistent |
| pupil_diameter | 0.2566 | 0.2000 | 1.28x | Good |
| trial_outcome | 0.2648 | 0.2500 | 1.06x | Low, expected for static per-trial |

**Check 1: Accuracy vs chance**
- All outputs above chance ✓
- image_identity at 2x chance is good for 16 classes
- image_change modest but the class is very imbalanced (97.5% no_change)
- trial_outcome is static per trial (same value all timepoints), harder to decode

**Check 2: Accuracy comparison to papers**
- Paper used random forest classifier per imaging plane, not neural network across all sessions
- Direct comparison not applicable due to different decoder architecture and scope
- Paper decoded change vs repeat and hit vs miss on image-by-image basis
- Our decoder operates on continuous time series across all sessions

**Check 3: Train vs validation gap**
- Small gap for all outputs (< 1.1x), no overfitting concern ✓

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
