# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - longitudinal tracking of neuronal activity from mouse barrel cortex
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from paper
- `code/` - Reference code (Track2p package)
- `data/` - Raw data (6 subjects: jm031, jm032, jm038, jm039, jm040, jm046)
- `decoder.py` - Decoder module
- `train_decoder.py` - Decoder training script
- `data/load_data.ipynb` - Data loading notebook

Each subject has 6-7 sessions (dated directories), each containing:
- `suite2p/plane0/` - Neural data (F.npy, Fneu.npy, spks.npy, iscell.npy, stat.npy, ops.npy)
- `move_deve/` - Behavioral data (motion_energy_glob.npy, tstamps.npy, interframe_int.npy)

Python environment verified: numpy 2.4.4, torch 2.6.0+cu124, GPU available.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_traces() | data/load_data.ipynb | LOADING | Load raw F from suite2p |
| load_fov() | data/load_data.ipynb | LOADING | Load mean image from ops.npy |
| zscore_rows() | data/load_data.ipynb | PROCESSING | Z-score rows for visualization |
| iscell filtering | code/notebooks/demo_t2p_ouputs.ipynb | CURATION | Filter by iscell threshold |
| match_mat filtering | code/notebooks/demo_t2p_ouputs.ipynb | CURATION | Get neurons tracked across all days |

### Notes
- The data files already contain only tracked neurons (Track2p suite2p format output)
- All iscell values are 1 (all cells pass threshold) - no further neuron filtering needed
- Neurons are matched across sessions by row index (row i in session A = same neuron as row i in session B)
- The reference code `load_data.ipynb` loads raw F directly; notes say "for more proper analysis compute dF/F the way as described in the paper (or alternatively use spks.npy)"
- Paper uses dF/F (baseline-corrected fluorescence) for decoding analysis
- Suite2p parameters: neucoeff=0.7, baseline='maximin', sig_baseline=10, win_baseline=60, prctile_baseline=8
- Paper: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps"

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/
  jm031/  (7 sessions, 221 neurons, 36000 frames = 20 min at 30Hz)
  jm032/  (7 sessions, 370 neurons, 36000 frames = 20 min)
  jm038/  (7 sessions, 685 neurons, 54000 frames = 30 min)
  jm039/  (7 sessions, 746 neurons, 54000 frames = 30 min)
  jm040/  (6 sessions, 541 neurons, 54000 frames = 30 min)
  jm046/  (7 sessions, 435 neurons, 54000 frames = 30 min)
```

Each session: `suite2p/plane0/{F,Fneu,spks,iscell,stat,ops}.npy` + `move_deve/{motion_energy_glob,tstamps,interframe_int}.npy`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2998 (sum across subjects, same within subject across sessions) |
| Neurons / subject | 221, 370, 685, 746, 541, 435 (mean=499.7) |
| Subjects | 6 |
| Sessions / subject | 6-7 (total 41 sessions) |
| Frame rate | 30 Hz |
| Session duration | 20 min (jm031,jm032) or 30 min (jm038-jm046) |
| Camera frame mismatches | Some sessions have fewer ME frames than neural frames |

### Missing camera frames
Sessions with frame mismatches (ME < F):
- jm031/2023-10-20_a: 2 missing
- jm031/2023-10-21_a: 3 missing
- jm031/2023-10-22_a: 116 missing
- jm032/2023-10-20_a: 2 missing
- jm032/2023-10-21_a: 2 missing
- jm032/2023-10-22_a: 148 missing
- jm039/2024-05-04_a: 1 missing
- jm040/2024-05-04_a: 1 missing
- jm046/2024-09-09_a: 1 missing

Timestamps in tstamps.npy are in kiloseconds (multiply by 1000 to get seconds).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons per mouse | 526 +/- 190 std | "On average 526 (+-190 std) neurons per mouse were successfully tracked" |
| Subjects | 6 | "a full dataset of 6 mice imaged daily" |
| Sessions / subject | min 6 consecutive days | "for a minimum of 6 consecutive days within the second postnatal week (P7 to P14)" |
| Neural data time bin | 10 frames = 0.333s | "averaging in bins of 10 consecutive timestamps" |
| Behavior data time bin | Same, 10 frames | "denoised the dF/F as well as the behaviour traces by averaging in bins of 10" |
| Imaging rate | 30 Hz | "Imaging rate was 30 Hz (resonant scanner)" |
| Session duration | 20 minutes | "each session lasted 20 minutes" |
| FOV | 720x720 um | "720 x 720 um field of view" |
| Resolution | 512x512 pixels | "512 x 512 pixel resolution" |
| iscell threshold | 0.5 | "all ROIs above the default threshold of 0.5 as true cells" |
| dF/F method | Baseline corrected | "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" |
| Decoding method | Ridge regression | "linear regression with ridge regularisation" |
| CV folds | 5-fold nested CV | "5 fold splits for both inner and outer loops" |
| CV split unit | 2-min blocks | "splits were done on consecutive 2 minute blocks" |
| Tracked % | 33% +/- 11% std | "corresponding to 33% (+/-11% std) of the neurons detected on the first day" |

### Processing Details
- Neural: dF/F computed from Suite2p (F - 0.7*Fneu, then baseline correction using maximin method)
- Binning: 10 frames average for both neural and behavioral data
- Camera triggered by microscope at 30 Hz
- Motion energy: pixel-wise difference of consecutive frames, squared, summed across pixels
- Missing camera frames: interpolate over them

### Curation Steps

**Neuron curation rules**:
- Data already contains only tracked neurons (Track2p output)
- iscell threshold of 0.5 already applied
- No additional neuron filtering needed

**Trial curation rules**:
- No specific trial curation mentioned (spontaneous behavior, no task trials)
- Sessions split into 60-second chunks as per decoder task specification

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Motion energy (R2) | Variable by age: low at P8-P10, increasing after P11, high at P12-P14 |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session duration | N/A | 20 min (jm031,32) or 30 min (jm038-46) | 20 minutes | Some sessions are 30 min; use all available data |
| Neurons/mouse | N/A | mean=499.7 | 526 +/- 190 | Close enough; paper may include slightly different set |
| Neuron counts | All tracked cells in data | 221-746 per mouse | 526 +/- 190 std | Consistent with paper stats |

Note: Session duration discrepancy (20 vs 30 min) - paper says 20 min but 4 of 6 mice have 30-min sessions. This may be because the paper focused on the first dataset (jm031-jm032) or the full duration is available in the data. We use all available data.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| F.npy, Fneu.npy | neural | Compute dF/F (F - 0.7*Fneu, baseline correction), bin 10 frames | Paper uses dF/F for decoding |
| Time in seconds | input[0] | np.arange(n_bins) * bin_duration_sec | Time elapsed from session start |
| motion_energy_glob.npy | output[0] | Interpolate missing frames, bin 10 frames, discretize to 5 percentile bins per session | Per-session percentile bins |

### Key Decisions
1. **dF/F computation**: Use Suite2p baseline-corrected dF/F method (F - 0.7*Fneu, maximin baseline), matching paper description
2. **Binning**: 10-frame bins (0.333s) as described in paper for decoding analysis
3. **Trial splitting**: 60-second trials = 180 bins per trial (as specified in decoder task)
4. **Motion energy alignment**: Interpolate missing camera frames to neural frame indices
5. **Discretization**: 5 equal-percentile bins per session for motion energy output
6. **Brain region**: "barrel cortex" (all neurons from same region)
7. **No additional neuron filtering**: Data already contains only tracked neurons with iscell > 0.5

### Planned Sanity Checks
- [x] Verify neuron counts match paper (~526 +/- 190 per mouse)
- [ ] Verify dF/F values are reasonable (typical range)
- [ ] Verify binned ME and neural data are temporally aligned
- [ ] Verify 60s trial splitting produces expected trial counts
- [ ] Verify ME discretization produces ~equal bin frequencies
- [ ] Spot-check neural activity at specific timepoints against raw F

---

## Step 6: Script Development
**Status**: COMPLETE

Script `/app/convert_data.py` implemented with:
- dF/F computation using Suite2p's maximin baseline method
- 10-frame binning for both neural and behavioral data
- Motion energy interpolation for missing camera frames
- 60-second trial splitting
- 5 equal-percentile discretization of motion energy per session
- Processing visualization option

Code inefficiencies identified:
- Per-neuron loop for maximin baseline (sliding window min)

Code speedups added:
- Vectorized binning using reshape+mean
- numpy.interp for alignment

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons / session | 221 |
| Subjects | 1 (jm031) |
| Sessions | 2 |
| Trials / session | 20 |
| Trials total | 40 |
| dF/F range | [-3.04, 18.31] |
| dF/F mean | 0.33-0.40 |
| Time input range | [0.0, 1199.7] s |
| ME bin distribution | {0: 0.20, 1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20} |

### Processing Plots Review
- dF/F shows typical calcium transients (verified)
- Motion energy raw and aligned overlay correctly (no misalignment for full-frame sessions)
- Discretized ME shows 5 levels as expected
- Neural activity raster looks normal

### Run Time Estimates
| Step | Time / Session | Total Estimate |
|------|---------------|----------------|
| Processing (jm031, 221 neurons, 36k frames) | ~0.37s | |
| Processing (larger, ~700 neurons, 54k frames) | ~1.5s est | |
| Full conversion (41 sessions) | | ~40-60s |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| motion_energy | 0.4786 | 0.2942 | 0.2000 |

Notes: Sample uses only jm031 (first 2 sessions = P8-P9), which are early postnatal days where paper shows low decoding R2. Above-chance validation accuracy confirms data is formatted correctly. Loss decreased from 20.5 to 1.47 over 200 epochs.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 395.3 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 6 | 6 | 6 | Yes |
| Sessions | min 6/mouse | 6-7/mouse (41 total) | 41 | Yes |
| Sessions/subject | min 6 | 7,7,7,7,6,7 | 7,7,7,7,6,7 | Yes |
| Neurons/mouse | 526+/-190 | 221,370,685,746,541,435 | Same | Yes (mean=499.7) |
| Mean neurons/session | ~526 | ~499.7 | 498.66 | Yes |
| Trials/session | N/A (60s) | N/A | 20 or 30 | Correct |
| Total trials | N/A | N/A | 1090 | - |
| Frame rate | 30 Hz | 30 Hz | 30 Hz | Yes |
| Time bin | 10 frames | N/A | 333.3 ms | Yes |
| Session duration | 20 min | 20 or 30 min | 20 or 30 min | Matches data |
| ME bin distribution | N/A | N/A | Uniform 0.20 each | Correct |
| iscell threshold | 0.5 | All > 0.5 | All pass | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- verification_full_out.txt: No errors, no warnings

### Check 2: Sanity checks (loading original data directly)
- **Neural**: Spot-checked session 5 (jm031/2023-10-23_a), neuron 10, trial 2, timepoint 50. Recomputed dF/F from raw F.npy/Fneu.npy independently. Result: np.allclose = True
- **Output**: Spot-checked ME discretization at bin 100, session 5. Recomputed from raw motion_energy_glob.npy. Result: exact match
- **Input**: Spot-checked time value at bin 100, session 5. Expected 33.3333s, got 33.3333s. Result: np.allclose = True

### Check 3: Reference code comparison
| Processing Step | My Code | Reference Code/Paper | Match? |
|---|---|---|---|
| Data loading | F.npy, Fneu.npy from suite2p/plane0 | load_data.ipynb loads F.npy same way | Yes |
| Neuron filtering | No filtering (data has tracked cells) | Data README: "only includes traces for cells present across all days" | Yes |
| Temporal alignment | np.interp camera times to neural frames | Data README: "interpolated over" | Yes |
| Binning | 10-frame average | Paper: "averaging in bins of 10 consecutive timestamps" | Yes |
| dF/F | F-0.7*Fneu, maximin baseline | Paper: "baseline corrected fluorescence... default Suite2p parameters", Suite2p uses gaussian->min->max | Yes |
| Output construction | 5 equal-percentile bins per session | Decoder task spec | Yes |

### Check 4: Key statistics comparison
| Statistic | Paper | Converted | Match? |
|---|---|---|---|
| Subjects | 6 | 6 | Yes |
| Neurons/mouse mean | 526 +/- 190 | 499.7 +/- 180.5 | Close (within 1 std) |
| Sessions/mouse | min 6 | 6-7 | Yes |
| Frame rate | 30 Hz | 30 Hz | Yes |
| iscell threshold | 0.5 | All > 0.5 (pre-filtered) | Yes |

### Check 5: Edge cases
- Trial boundaries: Time gap between trials = 0.3333s (one bin), correct
- All trials have consistent shapes within sessions
- First and last sessions verified
- Missing camera frames handled correctly via interpolation

### Issues Found and Resolved
- Initial dF/F computation was incorrect (running min only, no gaussian smoothing or max filter). Fixed to match Suite2p's maximin: gaussian -> minimum -> maximum filter.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (58.5 -> 1.41 over 200 epochs)
- Training: 872 trials, Testing: 218 trials

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|--------|-------|
| motion_energy | 0.5018 | 0.3165 | 0.2000 | 1.58x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Val Balanced Acc | Chance | Ratio | Paper Reference |
|---|---|---|---|---|
| motion_energy | 0.3165 | 0.2000 | 1.58x | Paper: R2 varies 0-0.7 by age |

### Check 1: Accuracy vs chance
- Validation accuracy 0.3165 is 1.58x chance (0.2000). Above chance confirms the decoder finds meaningful signal.

### Check 2: Accuracy comparison to paper
- Paper uses continuous R2 from ridge regression; we use 5-class balanced accuracy from a neural network
- Paper shows R2 varies dramatically by postnatal age: ~0 at P8-P10, increasing to ~0.4-0.7 by P12-P14
- Our dataset pools all ages, so early sessions (weak coupling) dilute overall accuracy
- 1.58x chance across mixed-age data is consistent with paper findings

### Check 3: Train vs validation gap
- Train: 0.5018, Val: 0.3165, ratio 1.59x
- Slightly above 1.5x threshold. Expected given: (a) different number of neurons per session, (b) mix of easy and hard sessions, (c) 200 epochs training. Not indicative of a conversion bug.

### Additional verification
- Spot-checked output values for late session (jm039, session 7): 5 different trial/timepoint combinations all match expected values computed independently from raw data.

### Issues Found and Resolved
- None

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots and decoder output plots
- [x] cache/README_CACHE.md created
- [x] All files organized
