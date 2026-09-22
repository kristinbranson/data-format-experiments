# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - Longitudinal tracking of neuronal activity in developing mouse barrel cortex
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment:
- Python 3 with numpy 2.4.4, torch 2.6.0+cu124, CUDA available

Directory contents:
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Extracted methods from paper
- `/app/code/` - Reference code (Track2p package with notebooks)
- `/app/data/` - Data files (6 mice: jm031, jm032, jm038, jm039, jm040, jm046 + load_data.ipynb + README.md)
- `/app/decoder.py` - Decoder implementation
- `/app/train_decoder.py` - Decoder training script

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_traces() | data/load_data.ipynb | LOADING | Load raw F.npy from session dir |
| load_fov() | data/load_data.ipynb | LOADING | Load meanImg from ops.npy |
| zscore_rows() | data/load_data.ipynb | PROCESSING | Z-score for visualization |
| iscell filtering | code/notebooks/demo_t2p_ouputs.ipynb | CURATION | Filter by iscell_thr (default 0.5) |
| match matrix filtering | code/notebooks/demo_t2p_ouputs.ipynb | CURATION | Select neurons tracked across all days |

### Notes
- Track2p outputs data in Suite2p format with only tracked neurons
- The provided data already contains only tracked neurons (matched across all days)
- All cells in the provided data pass iscell > 0.5 (already filtered)
- Suite2p files: F.npy (raw fluor), Fneu.npy (neuropil), spks.npy (deconvolved), iscell.npy, stat.npy, ops.npy
- Behavior data: motion_energy_glob.npy, tstamps.npy, interframe_int.npy in move_deve/
- Paper uses dF/F (baseline corrected, Suite2p defaults) for decoding, NOT raw F or spks
- Decoding bins data by 10 frames (from 30 Hz to 3 Hz)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/{subject}/{session_date}_a/suite2p/plane0/{F,Fneu,spks,iscell,stat,ops}.npy
data/{subject}/{session_date}_a/move_deve/{motion_energy_glob,tstamps,interframe_int}.npy
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 (jm031, jm032, jm038, jm039, jm040, jm046) |
| Sessions / subject | jm031:7, jm032:7, jm038:7, jm039:7, jm040:6, jm046:7 (total 41) |
| Neurons / subject | jm031:221, jm032:370, jm038:685, jm039:746, jm040:541, jm046:435 |
| Neurons (total) | 2998 (summed across subjects; constant within subject across sessions) |
| Mean neurons/subject | 499.7 |
| Frames/session | jm031,jm032: 36000 (20min); jm038-jm046: 54000 (30min) |
| Frame rate | 30 Hz |
| ME length mismatches | Several sessions have ME shorter than neural frames (missing camera frames) |

### Suite2p ops parameters (relevant for dF/F)
- fs=30, neucoeff=0.7, baseline='maximin', win_baseline=60.0, sig_baseline=10.0, prctile_baseline=8.0

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 | "full dataset of 6 mice" |
| Mean neurons/mouse | 526 +/- 190 std | "On average 526 (± 190 std) neurons per mouse" |
| % tracked | 33% +/- 11% std | "33% (± 11% std) of neurons detected on first day" |
| Sessions/mouse | >=6 consecutive days | "imaged daily for a minimum of 6 consecutive days" |
| Age range | P7 to P14 | "within the second postnatal week (P7 to P14)" |
| Session duration | 20 minutes (but data shows some are 30 min) | "each session lasted 20 minutes" |
| Imaging rate | 30 Hz | "Imaging rate was 30 Hz" |
| Camera rate | 30 Hz | "Videos were recorded at 30 Hz" |
| Neuron iscell threshold | 0.5 | "default threshold of 0.5 as true cells" |
| Neural signal | dF/F (baseline corrected, Suite2p defaults) | "baseline corrected fluorescence traces as our dF/F" |
| Decoding bin size | 10 frames (333.33 ms) | "averaging in bins of 10 consecutive timestamps" |
| Decoding method | Ridge regression, nested 5-fold CV | "5 fold splits...consecutive 2 minute blocks" |
| Decoding target | Motion energy (continuous, R² metric) | "predict a behavioural variable (y, mouse motion)" |

### Processing Details
1. **dF/F computation**: F_corrected = F - 0.7*Fneu, then baseline correction (maximin filter, 60s window), dF/F = (Fc - baseline) / baseline
2. **Binning**: Average in bins of 10 consecutive timestamps for both dF/F and motion energy
3. **Camera sync**: Camera triggered by microscope acquisition; missing frames may occur (check tstamps.npy)
4. **Motion energy**: Pixel-wise difference of consecutive video frames, squared and summed across pixels

### Curation Steps

**Neuron curation rules**:
- Only neurons tracked across ALL days by Track2p are included (already done in provided data)
- iscell probability > 0.5 (Suite2p default; already applied in provided data)

**Trial curation rules**:
- No explicit trial curation mentioned (spontaneous behavior, no task trials)
- We will split sessions into 60-second trials as specified by decoder task

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Motion energy (continuous) | R² values shown in Fig 7C, varying by age; higher at later ages (P11+) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session duration | N/A | jm031,jm032: 36000 frames (20min); others: 54000 frames (30min) | "20 minutes" | Paper's "20 minutes" may refer to example mouse or earlier mice. Data shows variable session lengths. Use actual frame counts. |
| Mean neurons/mouse | N/A | (221+370+685+746+541+435)/6 = 499.7 | 526 +/- 190 | Close but not exact. The paper's number likely includes rounding or slightly different filtering. Our data mean is within 1 std. Acceptable. |
| ME length mismatches | README: "missing frames from camera" | Several sessions: ME shorter than neural frames (e.g., jm031 day3: 35998 vs 36000) | N/A | Interpolate missing ME frames to match neural frame count |
| Neuron filtering | demo notebook: filter by iscell_thr then by match matrix | All 221 cells in jm031 have iscell>0.5 | iscell > 0.5 | Data already filtered; no additional filtering needed |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| F.npy, Fneu.npy | neural | Compute dF/F (neuropil subtraction + baseline correction), bin by 10 frames | Suite2p defaults | (n_neurons, n_timebins) per trial |
| Time index | input[0] | time_elapsed = bin_index * (10/30) seconds from session start, adjusted per trial | N/A | Continuous, time-varying |
| motion_energy_glob.npy | output[0] | Interpolate missing frames, bin by 10 frames, discretize into 5 equal-percentile bins per session | N/A | Categorical (5 classes), time-varying |

### Key Decisions
1. **dF/F computation**: Use Suite2p default parameters (neucoeff=0.7, maximin baseline, win_baseline=60s). Paper explicitly states "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)".
2. **Binning by 10 frames**: Paper states "averaging in bins of 10 consecutive timestamps" for decoding. Time bin = 10/30 = 333.33 ms.
3. **Trial splitting**: Split each session into 60-second trials. At 3 Hz (after binning), each trial = 180 timebins. 20-min sessions → 20 trials, 30-min sessions → 30 trials.
4. **Motion energy interpolation**: For sessions with missing camera frames, interpolate ME to match neural frame count before binning.
5. **Motion energy discretization**: 5 equal-percentile bins per session (quintiles), as specified in decoder task. Each bin has ~20% of data.
6. **Input = time elapsed**: Time from session start in seconds. Per trial, this ranges from trial_start_time to trial_end_time.
7. **Brain region**: "barrel cortex" (all recordings from same region)
8. **Each session is a separate entry**: Sessions from same mouse are separate sessions in the output (neurons are tracked across days = same neuron ordering).

### Planned Sanity Checks
- [ ] Verify neuron counts match across sessions within each subject
- [ ] Verify dF/F values look reasonable (not all zeros, sensible range)
- [ ] Verify ME binning produces expected number of timebins
- [ ] Verify 5 ME percentile bins each contain ~20% of values per session
- [ ] Verify trial counts: 20-min sessions → 20 trials, 30-min sessions → 30 trials
- [ ] Compare total neuron count to paper (526 +/- 190 mean)

---

## Step 6: Script Development
**Status**: COMPLETE

### Implementation notes
- Script at `/app/convert_data.py`
- Uses Suite2p's maximin baseline for dF/F (reimplemented using PyTorch)
- Bins by 10 frames as per paper's decoding analysis
- Splits sessions into 60s trials (180 bins each)
- Motion energy interpolated for missing camera frames, then binned and discretized into 5 quintile bins per session

Code inefficiencies identified:
- dF/F computation uses CPU PyTorch (could use GPU but CPU is more reliable)

Code speedups added:
- Batch processing for dF/F baseline computation (100 neurons at a time)
- Vectorized binning operation

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons / session | 221 |
| Subjects used | jm031 |
| Sessions | 2 |
| Trials (total) | 40 |
| Trials / session | 20 |
| Time input range | [0.0, 1199.7] seconds |
| ME output distribution | ~0.2 per bin (5 bins) |

### Processing Plots Review
- Raw F/Fneu: normal calcium imaging traces
- dF/F: baseline-corrected, reasonable range
- Motion energy: sparse (mostly quiet), some movement bouts
- Discretization: 5 bins with equal fractions (~0.2 each)
- No anomalies detected

### Run Time Estimates
| Step | Time / Session | Estimated Total Time (41 sessions) |
|------|---------------|-------------------------------------|
| Full conversion | ~1.4s | ~57s |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| motion_energy | 0.5228 | 0.2883 |

- Loss decreases monotonically from 56.96 to 1.30
- Validation accuracy (0.2883) above chance (0.2000)
- Training-validation gap expected with only 2 sessions (40 trials)

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 395.3 MB
- `verification_full_out.txt`: created, no errors/warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 6 | 6 | 6 | Yes |
| Sessions | >=6/mouse | 41 total | 41 total | Yes |
| Sessions/subject | >=6 consecutive | 7,7,7,7,6,7 | 7,7,7,7,6,7 | Yes |
| Mean neurons/mouse | 526 +/- 190 | 499.7 | 499.7 | Close (within 1 std) |
| Neurons per subject | N/A | 221,370,685,746,541,435 | 221,370,685,746,541,435 | Yes |
| Imaging rate | 30 Hz | 30 Hz | 30 Hz | Yes |
| Bin size | 10 frames (333ms) | N/A | 333.33 ms | Yes |
| Trials/session (20min) | N/A | N/A | 20 | Yes (1200s / 60s) |
| Trials/session (30min) | N/A | N/A | 30 | Yes (1800s / 60s) |
| Total trials | N/A | N/A | 1090 | Expected |
| ME distribution | Equal bins | N/A | 0.2 per bin | Yes (quintiles) |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- No errors or warnings in verification_full_out.txt
- All sessions have correct dimensions
- Output distribution is 0.2 per bin for all sessions

**Check 2: Sanity checks against raw data**
- Neural: dF/F computed independently using Suite2p's preprocess function matches exactly (max abs diff = 0.0, np.allclose=True) for session 0, trials 0 and 5
- ME discretization: Manually computed discretization matches output exactly (0 differences) for session 0
- Input (time): First trial starts at 0.0s, ends at 59.67s; second trial starts at 60.0s. Correct.
- Missing ME frames: Session with 2 missing frames (jm031 day 3) handled correctly via interpolation; output bins still ~0.2 each

**Check 3: Reference code comparison**
| Step | My Code | Reference | Match? |
|------|---------|-----------|--------|
| (a) Data loading | Load F.npy, Fneu.npy | load_traces loads F.npy; demo loads F.npy, Fneu.npy, iscell.npy | Yes (we use F+Fneu for dF/F) |
| (b) Neuron filtering | No additional filtering (data pre-filtered) | Filter by iscell_thr then match matrix | Yes (provided data already filtered) |
| (c) Temporal alignment | Interpolate missing ME frames to neural frame count | Camera triggered by microscope, handle missing frames | Yes |
| (d) Binning | Average 10 consecutive frames (333.33 ms bins) | "averaging in bins of 10 consecutive timestamps" | Yes |
| (e) Input construction | Time from session start in seconds | N/A (decoder task specifies this) | N/A |
| (f) Output construction | ME binned then discretized into 5 quintile bins per session | N/A (decoder task specifies this) | N/A |

**Check 4: Key statistics comparison**
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Subjects | 6 | 6 | Yes |
| Sessions | >=6/mouse | 41 (7+7+7+7+6+7) | Yes |
| Mean neurons/mouse | 526 +/- 190 | 499.7 | Within 1 std |
| Imaging rate | 30 Hz | 30 Hz | Yes |
| Session duration | 20 min | 20 or 30 min | Partial (4 of 6 mice have 30-min sessions) |

**Check 5: Edge cases**
- Trial boundaries: first trial starts at t=0.0s, last trial ends at session end. No off-by-one errors.
- Sessions with missing ME frames: interpolation handles correctly (verified for jm031 day 3, 2 missing frames)
- All trials have exactly 180 bins (T=180)

### Issues Found and Resolved
- None found. All checks pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (103.88 -> 1.15 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| motion_energy | 0.6116 | 0.3097 | Chance: 0.2000, 1.55x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

**Check 1: Accuracy vs chance**
| Variable | Val Accuracy | Chance | Ratio |
|----------|-------------|--------|-------|
| motion_energy | 0.3097 | 0.2000 | 1.55x |

Accuracy is above 1.5x chance. This is reasonable given the dataset characteristics (see below).

**Check 2: Accuracy comparison to paper**
The paper reports R² values for same-day continuous regression of motion energy:
- Early days (P8-P11): R² near 0 for most mice
- Late days (P12-P14): R² up to ~0.4 for the best sessions
- Figure 7C shows high variability across mice and ages

Our decoder uses classification (5 bins) across ALL sessions pooled. The modest accuracy (0.31) is expected because:
1. Early developmental sessions (P8-P11) have weak neural-behavior coupling (paper Fig 7C,D)
2. Our decoder pools all sessions together, including many where R² ~ 0
3. Classification into 5 bins is a different task than continuous regression
4. The decoder must generalize across different mice with different neuron counts

**Check 3: Train vs validation gap**
- Training: 0.6116, Validation: 0.3097 (ratio: 1.97x)
- This gap is expected because:
  - Different sessions have vastly different amounts of neural-behavior information
  - The model can memorize patterns specific to training sessions
  - High-dimensional neural data with relatively few training samples
  - Not indicative of a conversion bug

**Verification of raw data accuracy for 3 specific trials:**
- Session 0 trial 0: Output values verified against manual computation (Check 2 in Step 10)
- Neural data verified exact match against independent Suite2p dF/F computation

### Issues Found and Resolved
- No issues found. Accuracy is consistent with expectations from the paper.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots
- [x] cache/README_CACHE.md created
- [x] All files organized
