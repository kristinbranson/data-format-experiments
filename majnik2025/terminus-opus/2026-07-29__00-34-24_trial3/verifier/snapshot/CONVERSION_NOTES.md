# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - Longitudinal tracking of neuronal activity from the same cells in the developing brain (Majnik et al. 2025)
- **Date started**: 2025-07-29
- **Goal**: Convert to decoder-compatible format for decoding motion energy from neural activity in mouse barrel cortex

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Track2p code repository
- `data/` - Neural and behavioral data for 6 mice
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from paper
- `decoder.py` - Decoder library
- `train_decoder.py` - Decoder training script

Python environment verified: numpy 2.3.5, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_traces | data/load_data.ipynb | LOADING | Load raw F.npy traces from session directory |
| F_processing | code/track2p/gui/data_management.py | PROCESSING | Compute dF/F with neuropil correction and baseline |
| load_all_ds_stat_iscell | code/track2p/io/s2p_loaders.py | LOADING | Load stat arrays filtered by iscell threshold |
| load_all_ds_ops | code/track2p/io/s2p_loaders.py | LOADING | Load ops dictionaries for all sessions |

### Notes
- The Track2p code is primarily for cell tracking across sessions, not for decoding
- No decoding analysis code is provided in the repository
- The provided data has ALREADY been processed by Track2p: neurons are tracked and matched across sessions
- F.npy files have the same number of rows (neurons) across all sessions for each mouse
- All iscell values are 1.0 (already filtered)
- The demo notebook (demo_t2p_ouputs.ipynb) shows how to load tracked data
- dF/F computation in F_processing uses: neucoeff=0.0 (no neuropil subtraction), baseline=maximin
- BUT the paper says "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)"
- Suite2p defaults from ops.npy: neucoeff=0.7, baseline=maximin, sig_baseline=10.0, win_baseline=60.0
- Therefore, dF/F should use Suite2p defaults (neucoeff=0.7), NOT the Track2p GUI defaults (neucoeff=0.0)

### dF/F Computation (Suite2p defaults)
1. Fc = F - 0.7 * Fneu (neuropil correction)
2. Flow = gaussian_filter(Fc, [0, sig_baseline=10]) -> minimum_filter1d(win=60*30=1800) -> maximum_filter1d(win=1800)
3. dF/F = (Fc - Flow) / Flow

### Decoding Processing (from methods.txt)
- "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps"
- "splits were done on consecutive 2 minute blocks of the recording"
- Ridge regression with nested cross-validation

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/
  jm031/  (mouse A)
    2023-10-18_a/ ... 2023-10-24_a/  (7 sessions)
      suite2p/plane0/  F.npy, Fneu.npy, iscell.npy, ops.npy, spks.npy, stat.npy
      move_deve/  interframe_int.npy, motion_energy_glob.npy, tstamps.npy
  jm032/  (mouse B) - 7 sessions
  jm038/  (mouse C) - 7 sessions + ground_truth.csv
  jm039/  (mouse D) - 7 sessions + ground_truth.csv
  jm040/  (mouse E) - 6 sessions
  jm046/  (mouse F) - 7 sessions + ground_truth.csv
  load_data.ipynb
  README.md
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 (jm031-jm046) |
| Sessions total | 41 |
| Sessions / subject | 6-7 |
| Neurons jm031 | 221 |
| Neurons jm032 | 370 |
| Neurons jm038 | 685 |
| Neurons jm039 | 746 |
| Neurons jm040 | 541 |
| Neurons jm046 | 435 |
| Neurons total | 2998 |
| Mean neurons/mouse | 499.7 |
| Frame rate | 30 Hz |
| Frames jm031/jm032 | 36000 (20 min) |
| Frames jm038-jm046 | 54000 (30 min) |

### Key Data Properties
- Neural: F.npy shape (n_neurons, n_frames), float32
- Motion energy: motion_energy_glob.npy shape (n_frames,), uint64
- Timestamps: tstamps.npy in units of kiloseconds (ksec), so multiply by 1000 for seconds
- Motion energy sometimes has slightly fewer frames than neural data (missing camera frames)
- All iscell[:,0] == 1.0 (pre-filtered)

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 | "a full dataset of 6 mice" |
| Sessions/subject | ≥6 | "at least 6 consecutive days" |
| Mean neurons/mouse | 526 ± 190 | "On average 526 (± 190 std) neurons per mouse" |
| % of first day cells | 33% ± 11% | "corresponding to 33 % (± 11 % std) of the neurons detected on the first day" |
| Frame rate | 30 Hz | "Imaging rate was 30 Hz" |
| Session duration | 20 min | "each session lasted 20 minutes" |
| FOV | 720x720 μm | "720 × 720 μm field of view" |
| Resolution | 512x512 px | "512 × 512 pixel resolution" |
| iscell threshold | 0.5 | "above the default threshold of 0.5 as true cells" |
| Bin size for decoding | 10 frames | "averaging in bins of 10 consecutive timestamps" |
| CV splits | 5-fold | "5 fold splits" |
| CV block size | 2 min | "consecutive 2 minute blocks" |
| Video frame rate | 30 Hz | "Videos were recorded at 30 Hz" |
| Brain region | barrel cortex L2/3 | "layer 2/3" |
| Age range | P7-P14 | "within the second postnatal week" |

### Processing Details
- dF/F: Suite2p baseline correction with default parameters (neucoeff=0.7, baseline=maximin)
- Denoising: average in bins of 10 consecutive timestamps for both dF/F and behavior
- Motion energy: pixel-wise difference of consecutive video frames, squared, summed across pixels
- Decoding: Ridge regression, nested 5-fold cross-validation on 2-minute blocks

### Curation Steps
**Neuron curation rules**: iscell probability > 0.5 (already applied in provided data)
**Trial curation rules**: None explicitly mentioned - continuous recordings

### Decoders Trained
| Decoded variable | Metric |
|---|---|
| Motion energy (behavioral state) | R² (coefficient of determination) |

Note: Paper uses continuous R² regression. Our task requires categorical classification (5 bins).
R² values shown in figures only (Fig 7C, 7G, Fig S7E, S7F) - not reported as specific numbers in text.
Same-day R² appears low early (≤P11) and higher late (>P11).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neurons/mouse | N/A | 499.7 mean (221-746) | 526 ± 190 | Close match. Data mean is within range. |
| neucoeff for dF/F | Track2p GUI: 0.0 | ops.npy: 0.7 | "Suite2p defaults" | Use 0.7 per paper |
| Session duration | N/A | 36000 frames (20min) for jm031/32, 54000 (30min) for others | "20 minutes" | Paper says 20min but some mice have 30min recordings. Use all available data. |
| Number of sessions | N/A | 41 total (6-7/mouse) | "at least 6 consecutive days" | Consistent |
| Motion energy frames | N/A | Sometimes fewer than neural frames | README: missing camera frames | Interpolate or truncate to common length |

### Key Resolution: Session Duration Discrepancy
The paper states "each session lasted 20 minutes" but 4 of 6 mice (jm038, jm039, jm040, jm046) have 54000 frames = 30 minutes. This may be because the paper focused on a subset or the 20-minute statement applies to the example mouse. We will use all available data.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference | Notes |
|-----------------|--------------|-----------|-----------|-------|
| F.npy, Fneu.npy | neural | dF/F (Suite2p defaults: neucoeff=0.7, maximin baseline), bin by 10 frames | methods.txt: "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" | (n_neurons, n_timepoints) per trial |
| time index | input[0] | Time elapsed from start of session in seconds | Task spec: "Time elapsed from the beginning of the experiment" | (1, n_timepoints) per trial |
| motion_energy_glob.npy | output[0] | Normalize per-session, discretize into 5 equal-percentile bins | Task spec: "Motion energy, normalized and discretized into five equal-percentile bins" | (1, n_timepoints) per trial |

### Key Decisions
1. **Trial segmentation**: Use 2-minute blocks (3600 frames at 30Hz). Paper uses "consecutive 2 minute blocks" for CV splits. After binning by 10: 360 timepoints per trial.
2. **dF/F computation**: Use Suite2p defaults (neucoeff=0.7) per paper, not Track2p GUI defaults (neucoeff=0.0).
3. **Binning**: Average in bins of 10 consecutive timestamps for both neural and behavioral data, per methods.txt.
4. **Motion energy normalization**: Per-session min-max normalization before percentile binning.
5. **Missing ME frames**: Interpolate to match neural frame count.
6. **Session = recording day**: Each session is one recording day for one mouse.
7. **Time bin size**: 333.33 ms (10 frames / 30 Hz).

### Planned Sanity Checks
- [ ] Verify neuron counts match across sessions for same mouse
- [ ] Verify dF/F values are reasonable (mean ~0, std ~1)
- [ ] Verify motion energy bin distribution is approximately uniform (20% each)
- [ ] Verify trial counts: 10 for 20-min sessions, 15 for 30-min sessions
- [ ] Compare neuron counts to paper (526 ± 190 mean)
- [ ] Spot-check raw F values against loaded data

---

## Step 6: Script Development
**Status**: COMPLETE

Script: convert_data.py
- Implements dF/F computation using Suite2p defaults
- Bins data by 10 frames
- Segments into 2-minute trials
- Normalizes and discretizes motion energy per session
- Supports --sample, --full, and --show-processing flags

Code inefficiencies identified: None significant - vectorized operations used throughout.
Code speedups added: Using float64 for baseline computation to avoid precision issues.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 2 (jm031, jm032) |
| Sessions | 4 |
| Trials/session | 10 |
| Total trials | 40 |
| Neurons jm031 | 221 |
| Neurons jm032 | 370 |
| Timepoints/trial | 360 |
| Time bin size | 333.33 ms |
| Input range | [0.0, 1199.7] s |
| Output distribution | [0.2, 0.2, 0.2, 0.2, 0.2] (uniform) |

### Processing Plots Review
Plots saved as processing_session_0.png and processing_session_1.png. No anomalies detected.

### Run Time Estimates
| Step | Time/Session | Est. Total Time |
|------|-------------|------------------|
| Full conversion | ~0.8s | ~33s for 41 sessions |

### Bug Fixed
- dF/F computation was using division by baseline (Fc-Flow)/Flow which caused huge values
- Fixed to use subtraction only (Fc-Flow) matching Suite2p's actual implementation

---

## Step 8: Sample Decoder Training
**Status**: NOT STARTED

---

## Step 9: Full Conversion and Validation
**Status**: NOT STARTED

---

## Step 10: Critical Review 1
**Status**: NOT STARTED

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED
