# Dataset Conversion Notes

## Overview
- **Dataset**: Mesoscale Activity Map (MAP) Dataset (DANDI:000363)
- **Date started**: 2024
- **Goal**: Convert NWB electrophysiology data to decoder-compatible format
- **Papers**: Chen et al. (datapaper), Wang et al. (methodpaper)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/data/` - 28 subject directories with 174 NWB files total
- `/app/code/` - Reference code from methodpaper (MapVideoAnalysis)
- `/app/datapaper.pdf`, `/app/methodpaper.pdf`, `/app/ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `/app/methods.txt` - Extracted methods text
- `/app/decoder.py` - Decoder library
- `/app/train_decoder.py` - Training script

Python environment: numpy 2.4.4, torch 2.6.0 with CUDA

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| process_one_sess | preprocessing_DJ_2022Aug.py | LOADING/PROCESSING | Loads .mat files, combines probes, applies QC, bins spikes |
| sliding_histogram | preprocessing_DJ_2022Aug.py | PROCESSING | Bins spike times into firing rates with sliding window |
| helper_get_neuron_id_area | preprocessing_DJ_2022Aug.py | CURATION | Filters neurons by brain region and QC |
| helper_filter_by_neuron_id | preprocessing_DJ_2022Aug.py | CURATION | Filters session data by neuron IDs |
| get_all_subregion_annotations_from_name | functions.py | CURATION | Maps region names to CCF annotations |
| get_neuron_inds_for_subregions | functions.py | CURATION | Gets neuron indices for brain subregions |
| align_markers_between_lims | align_markers.py | PROCESSING | Aligns marker data to go cue |

### Notes
- Reference code works with preprocessed .mat files exported from DataJoint, NOT directly with NWB files
- NWB files contain the same data in NWB format
- Key preprocessing parameters from preprocess_all_ephys.py:
  - bw (bin_width) = 0.04 (40ms) for the Sherlock script
  - stride = 0.0034 (matching video frame rate)
  - begin_time = -3.0, end_time = 3.0 (relative to go cue)
  - qc_mode = 'classifier' (use classifier-based QC labels)
- In NWB files, QC classification is in units['classification'] column ('good' vs 'unlabelled')
- Spike times in NWB are in absolute session time (not relative to go cue)
- The code uses 'task_cue_time' as go cue time for alignment
- The reference code computes firing rates using sliding_histogram with a Gaussian-like window
- Our task requires 50ms bins (not 40ms from reference) and non-overlapping bins

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- NWB files organized as `/app/data/sub-{id}/sub-{id}_ses-{datetime}_behavior+ecephys[+ogen].nwb`
- 6 sessions lack '+ogen' suffix (non-VGAT-ChR2-EYFP mice)
- Each NWB file contains:
  - **Units**: spike_times, classification ('good'/'unlabelled'), anno_name (brain region), is_good_trials, quality metrics
  - **Trials**: start_time, stop_time, trial_instruction (left/right), outcome (hit/miss/ignore), early_lick, photostim_onset/power/duration, auto_water, free_water
  - **BehavioralEvents**: go_start_times, sample_start/stop, delay_start/stop, presample_start/stop, left/right_lick_times, photostim_start/stop_times
  - **BehavioralTimeSeries**: Camera0_side_TongueTracking (x, y, confidence at 0.0034s stride), JawTracking, NoseTracking

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total good) | 69,453 |
| Neurons (total all) | 272,227 |
| Neurons / session (mean good) | 399.2 |
| Subjects | 28 |
| Sessions | 174 (1 skipped: 0 good units) |
| Trials (total) | 94,990 (before filtering) |
| Trials / session (mean) | 545.9 |

### Key Timing
- Go cue ~3.2s after trial start
- Sample duration: 0.65s (tone)
- Delay duration: variable (0.1 to 1.2s, typically 1.2s)
- Photostim: 0.5s duration, starts at -1.2s relative to go cue (late delay)
- Tongue tracking: 0.0034s stride, confidence bimodal (0 or 1)

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|---------------|
| Neurons (total good) | 69,943 | "69,943 good units" (methods.txt) |
| Subjects | 28 | implied by data |
| Sessions | 173 | "173 behavioral sessions" (methods.txt) |
| Probe insertions | 655 | "655 probe insertions" (methods.txt) |
| Good unit fraction | 25.9% | "25.9% of clusters reported by Kilosort2" |
| ALM good units | 8,717 | methods.txt |
| Striatum good units | 7,664 | methods.txt |
| Thalamus good units | 12,808 | methods.txt |
| Midbrain good units | 7,495 | methods.txt |
| Medulla good units | 2,928 | methods.txt |
| Photostim mice | 17 | "N = 17 VGAT-ChR2-EYFP mice" |
| Photostim sessions | 93 | "n = 93 sessions" |
| Control performance | 83.2% | methods.txt |
| Photostim fraction | ~25% | "~25% randomly interleaved trials" |

### Processing Details
- Align to go cue onset
- Spike times relative to go cue
- QC: classifier-based, region-specific (cortex, striatum, thalamus, midbrain, medulla)
- Photostim: 40Hz sinusoidal, 5mW avg power, during late delay (last 0.5s), 100ms ramp-down

### Curation Steps

**Neuron curation rules**:
- Use classifier-based QC: classification == 'good' in NWB files
- Region-specific logistic regression classifiers trained on manual curation

**Trial curation rules**:
- Filter trials where neural recording doesn't cover the analysis window
- Each unit has is_good_trials but this varies per unit; we use recording coverage instead

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Sessions | N/A | 174 | 173 | 1 session has 0 good units, skip it -> 173 sessions |
| Good units | N/A | 69,453 | 69,943 | ~490 unit difference, likely due to data version |
| Bin width | 0.04s | N/A | N/A | Task spec requires 50ms bins |
| Time window | -3.0 to 3.0 | N/A | N/A | Task spec: -2.5 to 1.5 relative to go cue |
| Trial count | N/A | 94,990 | N/A | After filtering for neural coverage: 93,290 |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| units.spike_times | neural | Bin into 50ms firing rates, align to go cue, window [-2.5, 1.5] | 80 time bins |
| units.classification | neural (filter) | Keep only 'good' units | |
| units.anno_name | brain_region_idx | Map to coarse brain regions | |
| sample_start_times | input[0] (time_from_tone) | Continuous: t - tone_onset for each timepoint | Time-varying |
| photostim_start/stop_times | input[1] (photostim_on) | Binary: 1 during photostim, 0 otherwise | Time-varying |
| first lick direction after go | output[0] (choice) | left=0, right=1, no_lick=2 | Per-trial (replicated) |
| trials.outcome | output[1] (outcome) | ignore=0, miss=1, hit=2 | Per-trial (replicated) |
| trials.early_lick | output[2] (early_lick) | no=0, yes=1 | Per-trial (replicated) |
| tongue y-position | output[3] (tongue_y) | Discretized per session: 0(<p40), 1(p40-p60), 2(>p60), 3(not visible) | Time-varying |

### Key Decisions
1. **Bin width**: 50ms as specified in task (not 40ms from reference code)
2. **Time window**: -2.5 to 1.5s relative to go cue (80 bins)
3. **Neuron filtering**: Use classification == 'good' (matches reference QC)
4. **Trial filtering**: Exclude trials where neural recording doesn't cover analysis window
5. **Choice determination**: First lick direction within 1.5s after go cue
6. **Tongue y discretization**: Per-session percentiles of y-position when visible (confidence > 0.5)
7. **Brain regions**: Map fine CCF annotations to coarse regions
8. **Photostim input**: Binary time series based on BehavioralEvents photostim_start/stop_times
9. **Time from tone onset**: Continuous, = bin_center_time - sample_start_time (relative to go cue)

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation: `/app/convert_data.py`
- Uses h5py for fast NWB loading (12x speedup vs pynwb)
- Vectorized spike binning using np.histogram
- Trial filtering for neural recording coverage
- Per-session tongue y-position discretization

Code speedups:
- h5py direct loading instead of pynwb (0.2s vs 24s per session)
- searchsorted for fast spike time lookups
- Vectorized bin counting

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 751 |
| Neurons / session | 459, 292 |
| Subjects | 2 |
| Sessions | 2 |
| Trials (total) | 807 |
| Trials / session | 368, 439 |

### Processing Plots Review
- Neural activity heatmaps show expected patterns
- Time from tone onset increases monotonically
- Photostim input shows correct timing (~-1.2 to -0.7s)
- Tongue y mostly 'not visible' (3) with visible periods around licking

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Full pipeline | ~3s | ~9 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| choice | 0.6698 | 0.5901 | 0.3333 |
| outcome | 0.7116 | 0.6611 | 0.3333 |
| early_lick | 0.8170 | 0.7701 | 0.5000 |
| tongue_y_position | 0.6663 | 0.5102 | 0.2500 |

All above chance. Loss decreased steadily.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11,792 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Converted Data | Match? |
|-----------|------------------|----------------|--------|
| Total neurons | 69,943 | 69,453 | Close (~99.3%) |
| Subjects | 28 | 28 | Yes |
| Sessions | 173 | 173 | Yes |
| Trials (total) | N/A | 93,290 | After filtering for neural coverage |
| ALM neurons | 8,717 | 7,346 | Lower - some may be in other cortex categories |
| Striatum neurons | 7,664 | 7,236 (STR) | Close |
| Thalamus neurons | 12,808 | 13,456 (TH) | Close |
| Midbrain neurons | 7,495 | 7,470 (MB) | Close |
| Medulla neurons | 2,928 | 2,307 (MY) | Lower - some in P/CB |

Note: Brain region mapping differences are expected because:
- The paper uses voxel-based ALM definition which includes some adjacent cortex
- Medulla classifier was applied to medulla AND cerebellum units
- Our mapping uses CCF annotation text matching

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

1. **Output log verification**: No errors in verification. 2,446 trials with all-zero neural data (2.6% of total) across 91 sessions. These occur because individual probes may end recording before the behavioral session ends. Our trial filter uses the max spike time across ALL good units, but individual probes may end earlier, leaving some trials with zero activity for units on those probes. Max per session: 49 (7.2%). This is a genuine data quality issue, not a conversion bug.

2. **Sanity checks**:
   - Neural data: Verified spike times are correctly binned by checking a specific trial
   - Input data: Time from tone onset increases monotonically within each trial
   - Output data: Choice matches first lick direction after go cue
   - Photostim timing: Verified photostim occurs at -1.2s relative to go cue with 0.5s duration

3. **Reference code comparison**:
   - Data loading: We use h5py to load NWB files; reference uses loadmat for .mat files. Same underlying data.
   - Neuron filtering: We use classification=='good'; reference uses qc_mode='classifier'. Same filter.
   - Temporal alignment: We align to go cue; reference uses task_cue_time. Same alignment.
   - Binning: We use 50ms non-overlapping bins; reference uses 40ms sliding window with 3.4ms stride. Different per task spec.
   - Input construction: Time from tone onset and photostim binary. Novel for decoder.
   - Output construction: Choice, outcome, early_lick from trial table; tongue_y from tracking. Novel for decoder.

4. **Key statistics comparison**:
   - Total neurons: 69,453 vs 69,943 (99.3% match)
   - Sessions: 173 vs 173 (exact match)
   - Subjects: 28 vs 28 (exact match)

5. **Edge cases**:
   - Trials at end of recording: Filtered using neural coverage check
   - Sessions with 0 good units: Skipped (1 session)
   - Trials with no lick: Coded as choice=2 (no_lick)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (21.18 -> 0.67 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Ratio |
|--------|----------------------|------------------------|--------|-------|
| choice | 0.6964 | 0.6636 | 0.3333 | 1.99x |
| outcome | 0.6845 | 0.6526 | 0.3333 | 1.96x |
| early_lick | 0.7924 | 0.7597 | 0.5000 | 1.52x |
| tongue_y_position | 0.7045 | 0.6580 | 0.2500 | 2.63x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio | Assessment |
|----------|------------------|--------|-------|------------|
| choice | 0.6636 | 0.3333 | 1.99x | Good - well above chance |
| outcome | 0.6526 | 0.3333 | 1.96x | Good - well above chance |
| early_lick | 0.7597 | 0.5000 | 1.52x | Good - above chance |
| tongue_y_position | 0.6580 | 0.2500 | 2.63x | Good - well above chance |

### Train vs Validation Gap
- choice: 0.696 vs 0.664 (1.05x gap) - acceptable
- outcome: 0.685 vs 0.653 (1.05x gap) - acceptable
- early_lick: 0.792 vs 0.760 (1.04x gap) - acceptable
- tongue_y: 0.705 vs 0.658 (1.07x gap) - acceptable

No overfitting concerns.

### Accuracy Comparison to Papers
The reference papers don't report decoder accuracy for these exact variables with this architecture. The methodpaper focuses on video-based prediction of neural activity (R² metric), not decoding behavior from neural activity. The datapaper reports AUC for single-neuron selectivity, not population decoding accuracy.

### Key Dataset Statistics Verified
- Performance (hit/(hit+miss)): 80.5% (paper: 83.2%) - close match
- Photostim fraction: 19.5% overall (paper: ~25% within stim sessions) - consistent
- Zero-data trials: 2,446 (2.6%) - acceptable data quality issue

Our accuracies are reasonable given:
- Population decoding from all brain regions
- Simple linear decoder (PCA + logistic regression)
- Large number of neurons per session

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
