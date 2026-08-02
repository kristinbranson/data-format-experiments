# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory - Visual Behavior 2P (Visual Behavior Ophys)
- **Date started**: 2025-07-29
- **Goal**: Convert to decoder-compatible format for neural decoding

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - AllenSDK source code
- `data/` - NWB data files and metadata
- `tutorials/` - Tutorial scripts
- `methods.txt`, `paper.pdf`, `whitepaper.pdf` - Reference materials
- `train_decoder.py`, `decoder.py` - Decoder scripts

Python: numpy 2.3.5, torch 2.6.0, pandas 2.3.3

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|--------|
| BehaviorOphysExperiment.from_nwb() | behavior_ophys_experiment.py | LOADING | Load experiment from NWB |
| dataset.events | BehaviorOphysExperiment | LOADING | Deconvolved calcium events |
| dataset.ophys_timestamps | BehaviorOphysExperiment | LOADING | Ophys frame timestamps |
| dataset.trials | BehaviorOphysExperiment | LOADING | Trial table with outcomes |
| dataset.stimulus_presentations | BehaviorOphysExperiment | LOADING | Stimulus timing |
| dataset.running_speed | BehaviorOphysExperiment | LOADING | Running speed |
| dataset.eye_tracking | BehaviorOphysExperiment | LOADING | Pupil area |

### Notes
- Paper uses "events" (deconvolved calcium events) for neural analysis
- Cell filtering (valid_roi) already applied in NWB files
- ROI filtering uses multi-label classifier

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- 284 NWB files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`
- Metadata CSVs in `data/visual-behavior-ophys-1.1.0/project_metadata/`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files available | 284 |
| Active experiments | 202 |
| Neurons (total, active) | 29,444 |
| Neurons / session | mean 146, range 4-666 |
| Subjects | 38 |
| Active sessions | 174 unique ophys sessions |
| Trials / session | mean 257, range 39-409 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Excitatory cells | 8,619 | "8,619 excitatory cells (21 imaging sessions, 9 mice)" |
| Sst cells | 470 | "470 Sst cells (15 imaging sessions, 6 mice)" |
| Vip cells | 1,239 | "1,239 Vip cells (21 imaging sessions, 9 mice)" |
| Image duration | 250ms | "250 ms stimulus duration" |
| ISI | 500ms | "500 ms inter-stimulus duration" |
| Decoding window | 400ms | "first 400 ms after each stimulus presentation" |
| MESO frame rate | 11 Hz | Whitepaper |
| CAM2P frame rate | 31 Hz | Whitepaper |

Note: Paper cell counts are for familiar+MESO subset only. Our dataset includes all active experiments.

### Processing Details
- Paper uses familiar image set sessions on multi-plane imaging rig
- Calcium events extracted from raw fluorescence
- 750ms image presentation intervals

### Curation Steps
**Neuron curation**: ROI filtering already applied (valid_roi=True)
**Trial curation**: Include Go + Catch, exclude Aborted + Auto-rewarded

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code/Data | Papers | Resolution |
|-------|-----------|--------|------------|
| Cell counts | 29,444 active | 10,328 (familiar+MESO) | Paper used subset; we use all active |
| Sessions | 202 active exps | 57 sessions | Paper used familiar MESO only |
| Frame rate | 11Hz + 31Hz | 11Hz (MESO) | Resample all to ~93ms bins |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source | Target | Transform |
|--------|--------|----------|
| events | neural | Resample to 93ms bins |
| (none) | input | No inputs |
| stimulus_presentations.image_name | output[0]: image_identity | Categorical, time-varying |
| stimulus_presentations.is_change | output[1]: image_change | Binary, time-varying |
| running_speed | output[2]: running_speed | Interpolate + 5 percentile bins |
| eye_tracking.pupil_area | output[3]: pupil_diameter | Interpolate + 5 percentile bins |
| trials outcome | output[4]: trial_outcome | Per-trial categorical |

---

## Step 6: Script Development
**Status**: COMPLETE

Key optimizations:
- h5py for fast stats collection (0.1s vs 5s per experiment with SDK)
- Session-wide resampling then trial extraction (vs per-trial resampling)
- Memory-efficient: load, process, release each experiment

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Trials | 229 |
| Neurons | 231 (89 + 142) |
| Bins/trial | 85-89 |
| Time bin | 93ms |

### Run Time Estimates
| Step | Time / Experiment | Total (202 exps) |
|------|------------------|------------------|
| Stats (h5py) | 0.1s | 20s |
| Load (SDK) | 5.5s avg | 18 min |
| Process | 0.5s avg | 1.5 min |
| Total | | ~20 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Train Acc | Val Acc | Chance |
|--------|-----------|---------|--------|
| image_identity | 0.192 | 0.132 | 0.0625 |
| image_change | 0.605 | 0.578 | 0.500 |
| running_speed | 0.308 | 0.295 | 0.200 |
| pupil_diameter | 0.290 | 0.233 | 0.200 |
| trial_outcome | 0.311 | 0.265 | 0.250 |

All outputs above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 2987.8 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference | Converted | Match? |
|-----------|-----------|-----------|--------|
| Sessions | 202 active | 202 | Yes |
| Trials | ~51K expected | 51,992 | Yes |
| Neurons | 29,444 | 29,444 | Yes |
| Subjects | 38 | 38 | Yes |
| Brain regions | VISp, VISl | VISp, VISl | Yes |
| Image names | 16 unique | 16 | Yes |
| Trial outcomes | hit/miss/fa/cr | 4 categories | Yes |

### Warnings
- 2,621 trials with all-zero neural data (expected for sparse calcium events)

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: No errors. 2,621 warnings about zero neural data - expected for sparse events in MESO experiments with few neurons.
2. **Neural sanity check**: Session 0, Trial 0 - original mean 0.000421, converted mean 0.000431 (close match after resampling). Neuron count matches (89).
3. **Temporal alignment**: Original trial has 225 frames at 32.3ms, converted has 78 bins at 93ms. Ratio 225/78 = 2.88 ≈ 32.3ms/93ms. Correct.
4. **Trial filtering**: Go+Catch only, excluding Aborted and Auto-rewarded. Verified on sample experiment.
5. **Image identity**: 16 unique images across image sets A and B. Correctly mapped.
6. **Reference code comparison**: Uses events (not dff), matches paper specification.

### Issues Found and Resolved
- Fixed output dtype from float32 to int64 (required for decoder indexing)
- Fixed pupil area collection in h5py (use area key, not width*height)
- Optimized memory usage to prevent slowdowns with large datasets

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (1.622 → 1.565)
- Test loss: 1.564

### Decoder Results (Full)
| Output | Train Acc | Val Acc | Chance | Val/Chance |
|--------|-----------|---------|--------|------------|
| image_identity | 0.172 | 0.167 | 0.0625 | 2.67x |
| image_change | 0.541 | 0.534 | 0.500 | 1.07x |
| running_speed | 0.257 | 0.254 | 0.200 | 1.27x |
| pupil_diameter | 0.309 | 0.304 | 0.200 | 1.52x |
| trial_outcome | 0.304 | 0.278 | 0.250 | 1.11x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Val Acc | Chance | Ratio | Assessment |
|----------|---------|--------|-------|------------|
| image_identity | 0.167 | 0.0625 | 2.67x | Good - neural activity encodes image identity |
| image_change | 0.534 | 0.500 | 1.07x | Marginal - change is brief event, hard to decode |
| running_speed | 0.254 | 0.200 | 1.27x | Moderate - running correlates with neural activity |
| pupil_diameter | 0.304 | 0.200 | 1.52x | Good - arousal state encoded in neural activity |
| trial_outcome | 0.278 | 0.250 | 1.11x | Marginal - outcome is behavioral, weakly encoded |

### Accuracy vs Papers
- Paper reports change decoder accuracy but uses random forest on individual imaging planes, not the same architecture
- Paper uses image-by-image analysis (first 400ms), our decoder uses full trial time series
- Direct comparison not possible due to different decoder architectures and analysis approaches
- All outputs above chance indicates correct data formatting

### Train vs Validation Gap
- Small gap for all outputs (< 1.1x ratio), no overfitting

### Issues Found and Resolved
- No issues requiring fixes

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
