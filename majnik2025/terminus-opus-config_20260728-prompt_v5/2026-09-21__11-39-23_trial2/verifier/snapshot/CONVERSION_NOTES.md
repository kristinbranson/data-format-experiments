# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - Longitudinal tracking of neuronal activity from barrel cortex (Majnik et al. 2025)
- **Date started**: 2025
- **Goal**: Convert to decoder-compatible format for decoding motion energy from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Extracted methods text
- `/app/code/` - Track2p code repository
- `/app/data/` - Dataset with 6 subjects (jm031-jm046)
- `/app/decoder.py` - Decoder module
- `/app/train_decoder.py` - Decoder training script
- `/app/data/load_data.ipynb` - Data loading notebook

Python environment: numpy 2.4.4, torch 2.6.0+cu124, scipy 1.18.0, suite2p available.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_traces | data/load_data.ipynb | LOADING | Load raw F from suite2p |
| load_fov | data/load_data.ipynb | LOADING | Load mean image from ops.npy |
| zscore_rows | data/load_data.ipynb | PROCESSING | Z-score for visualization |
| preprocess | suite2p.extraction.dcnv | PROCESSING | Baseline correction (maximin) |
| baseline_maximin | suite2p.extraction.dcnv | PROCESSING | Gaussian smooth -> min pool -> max pool -> subtract |
| demo_t2p_outputs | code/notebooks/ | LOADING | Shows how to load tracked cells across days |

### Notes
- Data has already been processed through Track2p - all neurons are tracked and matched across sessions
- All ROIs in provided data have iscell=1 (all pass the default 0.5 threshold)
- F.npy rows correspond to the same neuron across sessions for a given subject
- Suite2p preprocess does: Gaussian smooth -> min_pool -> max_pool -> subtract baseline (NOT divide)
- Paper says "baseline corrected fluorescence traces as our dF/F"
- Decoding in the paper uses bins of 10 frames and ridge regression with nested 5-fold CV on 2-minute blocks
- The demo notebook (demo_t2p_ouputs.ipynb) filters by iscell_thr from track_ops; in our data all cells already pass
- The eval notebooks focus on evaluating Track2p tracking performance (F1 scores), not neural decoding

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
/app/data/
  {subject}/                     # jm031, jm032, jm038, jm039, jm040, jm046
    {date}_a/                    # session directory
      suite2p/plane0/
        F.npy                    # Raw fluorescence (n_neurons, n_frames) float32
        Fneu.npy                 # Neuropil fluorescence (n_neurons, n_frames) float32
        iscell.npy               # Cell classification (n_neurons, 2) - all 1.0
        ops.npy                  # Suite2p options dict
        spks.npy                 # Deconvolved spikes (n_neurons, n_frames) float32
        stat.npy                 # ROI statistics (array of dicts)
      move_deve/
        motion_energy_glob.npy   # Global motion energy (n_me_frames,) uint64
        tstamps.npy              # Timestamps in kiloseconds (n_frames,) float64
        interframe_int.npy       # Inter-frame intervals (n_frames-1,) float64
  {subject}/ground_truth.csv     # Manual tracking ground truth (jm038, jm039, jm046 only)
  load_data.ipynb                # Data loading notebook
  README.md                      # Data documentation
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 (jm031, jm032, jm038, jm039, jm040, jm046) |
| Sessions total | 41 |
| Sessions/subject | jm031:7, jm032:7, jm038:7, jm039:7, jm040:6, jm046:7 |
| Neurons/subject | jm031:221, jm032:370, jm038:685, jm039:746, jm040:541, jm046:435 |
| Mean neurons/subject | 499.7 |
| Total neuron-sessions | 20,445 |
| Frame rate | 30 Hz (all sessions) |
| Frames/session | 36,000 (jm031, jm032: 20 min) or 54,000 (others: 30 min) |
| All iscell | 1.0 (all ROIs classified as cells) |
| ME length mismatches | Some sessions have ME slightly shorter than F (up to 116 frames) |

### Timestamp units
- tstamps.npy values span 0 to ~1.21 (for 36000 frames) or ~1.82 (for 54000 frames)
- These are in kiloseconds: multiply by 1000 to get seconds
- Verified: ts[-1] * 1000 ≈ 1200s (20min) for jm031/jm032, ≈ 1800s (30min) for others
- For our conversion, we use frame_index / fs for time in seconds instead

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|---------------|
| Subjects | 6 | "full dataset of 6 mice" |
| Sessions/subject | >=6 consecutive days | "imaged daily for a minimum of 6 consecutive days" |
| Neurons/subject (mean) | 526 ± 190 std | "On average 526 (± 190 std) neurons per mouse" |
| Fraction of day-1 neurons | 33% ± 11% std | "corresponding to 33% (± 11% std) of the neurons detected on the first day" |
| Imaging rate | 30 Hz | "Imaging rate was 30 Hz (resonant scanner)" |
| Session duration | 20 minutes | "each session lasted 20 minutes" |
| FOV | 720×720 µm, 512×512 pixels | "720 × 720 µm field of view and 512 × 512 pixel resolution" |
| Video rate | 30 Hz | "Videos were recorded at 30 Hz" |
| Neural data time bin | 10 frames (333ms) | "averaging in bins of 10 consecutive timestamps" |
| Behavior data time bin | 10 frames (333ms) | same |
| Cell threshold | 0.5 (default) | "all ROIs above the default threshold of 0.5 as true cells" |
| Neuropil coefficient | 0.7 | Suite2p default, confirmed in ops |
| Baseline method | maximin | Suite2p default, confirmed in ops |
| win_baseline | 60.0 seconds | confirmed in ops |
| sig_baseline | 10.0 frames | confirmed in ops |
| CV splits | 5-fold, 2-min blocks | "5 fold splits...on consecutive 2 minute blocks" |
| Peak detection | SciPy, bin=10 frames, height/prominence >= 1 std | For calcium event rate analysis |

### Processing Details
1. **Neuropil correction**: Fc = F - 0.7 * Fneu
2. **Baseline correction (dF/F)**: Suite2p maximin method:
   - Gaussian smooth with sigma=10 frames
   - Min pool with window = 60s × 30Hz = 1800 frames (made odd: 1801)
   - Max pool with same window
   - Subtract baseline: dF/F = Fc - Flow (subtraction only, NOT division)
3. **Binning**: Average in bins of 10 consecutive frames (both neural and behavioral)
4. **Motion energy**: Pixel-wise difference of consecutive frames, squared, summed across pixels (already computed in data)
5. **Decoding**: Ridge regression, nested 5-fold CV on 2-min blocks

### Curation Steps

**Neuron curation rules**:
- Suite2p iscell probability > 0.5 (default threshold)
- In our data, all ROIs already pass this threshold (all iscell=1)
- Track2p matching: only neurons tracked across all days are included
- In our data, this is already done (same n_neurons per subject across all sessions)

**Trial curation rules**:
- No explicit trial curation mentioned (spontaneous activity, no task trials)
- Sessions split into 60-second segments for our decoder task

### Decoders Trained (in paper)
| Decoded variable | Method | Notes |
|---|---|---|
| Motion energy | Ridge regression | Nested 5-fold CV, 2-min blocks, bins of 10 frames |
| (cross-day decoding also performed) | Ridge regression | Fit on one day, evaluate on others |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neurons/subject mean | N/A | 499.7 | 526 ± 190 | Within 1 std, consistent |
| Session duration | N/A | 20 min (jm031,jm032) or 30 min (others) | 20 min | Some sessions are longer; use all available data |
| ME length | N/A | Some ME arrays shorter than F (up to 116 frames) | N/A | Pad with last ME value to match F length |
| iscell filtering | threshold 0.5 | All iscell=1 | threshold 0.5 | Data already filtered by Track2p |
| dF/F computation | Suite2p preprocess subtracts baseline | N/A | "baseline corrected fluorescence traces as our dF/F" | Consistent: use Suite2p preprocess |

### Notes on session duration
- Paper says "each session lasted 20 minutes" but 4 out of 6 subjects have 54000 frames = 30 minutes
- The paper may be describing the primary cohort or a general statement; we use all available data
- This does not affect our processing since we split into 60-second trials regardless

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code | Notes |
|-----------------|--------------|-----------|----------------|-------|
| F.npy, Fneu.npy | neural | Fc = F - 0.7*Fneu → Suite2p preprocess (maximin) → bin 10 frames | suite2p.extraction.dcnv.preprocess | dF/F per Suite2p defaults |
| frame index | input[0] (time_in_session) | (bin_idx + 0.5) × (10/30) seconds | N/A | Time from session start, center of bin |
| motion_energy_glob.npy | output[0] (motion_energy) | align → bin 10 frames → 5 equal-percentile bins/session | N/A | Per task specification |
| subject folder name | subjects | Direct mapping | N/A | jm031-jm046 |
| N/A | brain_regions | All barrel_cortex | Paper: "barrel cortex" | Single region |

### Key Decisions
1. **dF/F computation**: Use Suite2p's preprocess with maximin baseline, matching the paper exactly. This is baseline subtraction (not division), which is what Suite2p calls "baseline corrected fluorescence".
2. **Binning**: 10 frames as specified in paper for decoding analysis. Both neural and behavioral data are binned identically.
3. **Trial duration**: 60 seconds = 180 bins per trial as specified in decoder task.
4. **ME discretization**: 5 equal-percentile bins per session as specified in decoder task. Using np.digitize with percentile edges.
5. **ME alignment**: When ME is shorter than F, pad with last ME value. This handles minor camera trigger drops.
6. **Time input**: Center of each bin (offset by 0.5 × time_per_bin). This represents the midpoint of each temporal bin.
7. **Brain region**: barrel_cortex for all neurons (paper: "barrel cortex development").
8. **No additional neuron filtering**: All neurons in the data are already Track2p-matched and iscell-filtered.

### Planned Sanity Checks
- [x] Neuron counts match paper statistics (499.7 vs 526 ± 190)
- [x] ME bin distribution is uniform (20% each)
- [x] Time input range matches session duration
- [x] dF/F values computed correctly (spot-checked against raw recomputation)
- [x] Neural data spot-checked across multiple sessions
- [x] Output data spot-checked across multiple sessions

---

## Step 6: Script Development
**Status**: COMPLETE

Script: `/app/convert_data.py`
- Supports --full, --sample, --show-processing flags
- Uses Suite2p's preprocess for dF/F computation
- Processing pipeline: load → neuropil correction → baseline subtraction → bin → discretize → split trials
- Handles ME length mismatches by padding

Code inefficiencies identified:
- Suite2p preprocess uses CPU torch operations (could use GPU but CPU is fast enough)

Code speedups added:
- Vectorized binning using reshape + mean
- No unnecessary I/O

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (jm031/2023-10-18_a, jm038/2023-04-30_a) |
| Subjects | 2 (jm031, jm038) |
| Trials total | 50 (20 + 30) |
| Neurons | 221, 685 |
| Timepoints/trial | 180 |
| Time bin size | 333.33 ms |
| Input range | [0.2, 1799.8] seconds |
| ME bin distribution | 20% each (perfect) |

### Processing Plots Review
Plots saved for both sessions showing raw F, dF/F, ME, discretized ME, neural rasters, and trial structure. No anomalies observed.

### Run Time Estimates
| Step | Time / Session | Notes |
|------|---------------|-------|
| Load | 0.0-0.2s | Depends on file size |
| dF/F | 0.2-0.8s | Depends on neuron count |
| Bin+discretize | <0.1s | Fast vectorized ops |
| Total | 0.3-1.1s | ~0.8s average |
| **Full estimate** | | **~33s for 41 sessions** |

Actual full conversion time: 31.7s ✓

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| motion_energy | 0.5149 | 0.2713 |

Chance level: 0.20 (5 classes). Both training and validation above chance.
Validation accuracy modest (27.1%) but expected with only 2 sessions / 50 trials.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 396.0 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 6 | 6 | 6 | ✓ |
| Sessions | >=6/subject | 41 total | 41 | ✓ |
| Sessions/subject | >=6 | 7,7,7,7,6,7 | 7,7,7,7,6,7 | ✓ |
| Neurons/subject mean | 526 ± 190 | 499.7 | 498.7 | ✓ (within 1 std) |
| Neurons range | N/A | 221-746 | 221-746 | ✓ |
| Total neuron-sessions | N/A | 20,445 | 20,445 | ✓ |
| Frame rate | 30 Hz | 30 Hz | 30 Hz (via bin size) | ✓ |
| Trials total | N/A | N/A | 1,090 | N/A |
| Trials/session | N/A | N/A | 20 or 30 | N/A |
| Timepoints/trial | N/A | N/A | 180 | N/A |
| Time bin size | 10 frames (333ms) | N/A | 333.33 ms | ✓ |
| ME bins | N/A | N/A | 5 (20% each) | ✓ |
| Input range | N/A | N/A | [0.2, 1799.8]s | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- `/app/verification_full_out.txt`: "Data format is valid, no errors or warnings."
- No errors, no warnings. All statistics look correct.

### Check 2: Sanity checks

**Neural data sanity check (3 spot checks):**
1. Session 0 (jm031/2023-10-18_a), trial 0, neuron 5, tp 10:
   - Converted: 2.4826254844665527
   - Recomputed from raw: 2.4826254844665527
   - Match: True ✓

2. Session 4 (jm031/2023-10-22_a, ME mismatch session), trial 5, neuron 10, tp 50:
   - Converted: 96.149658203125
   - Recomputed from raw: 96.149658203125
   - Match: True ✓

3. Session 40 (jm046/2024-09-09_a, last session), trial 29, neuron 0, tp 179:
   - Converted: 33.924468994140625
   - Recomputed from raw: 33.924468994140625
   - Match: True ✓

**Input data sanity check:**
- Session 0, trial 0, tp 10: converted=3.5, expected=3.5 ✓

**Output data sanity check:**
- Session 0, trial 0, tp 10: converted=3, expected=3 ✓
- Session 40, trial 29, tp 179: converted=0, expected=0 ✓
- Session 0 ME distribution: exactly 720 per bin (20% each) ✓

### Check 3: Reference code comparison

| Processing Step | My Code | Reference Code/Paper | Match? |
|----------------|---------|---------------------|--------|
| (a) Data loading | np.load F.npy, Fneu.npy, ops.npy, motion_energy_glob.npy | load_data.ipynb: np.load F.npy | ✓ |
| (b) Neuron filtering | All iscell=1, no additional filtering needed | iscell_thr=0.5, Track2p match matrix | ✓ (data pre-filtered) |
| (c) Neuropil correction | Fc = F - 0.7 * Fneu | neucoeff=0.7 from ops | ✓ |
| (d) Baseline correction | suite2p.extraction.dcnv.preprocess with maximin | Paper: "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" | ✓ |
| (e) Binning | Average 10 frames | Paper: "averaging in bins of 10 consecutive timestamps" | ✓ |
| (f) ME processing | Already computed in data | Paper: pixel-wise diff, squared, summed | ✓ |
| (g) ME binning | Average 10 frames | Paper: same binning for both neural and behavioral | ✓ |
| (h) ME discretization | 5 equal-percentile bins per session | Decoder task specification | ✓ |

### Check 4: Key statistics comparison
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Subjects | 6 | 6 | ✓ |
| Neurons/subject mean | 526 ± 190 | 499.7 | ✓ (within 1 std) |
| Sessions/subject | >=6 | 6-7 | ✓ |
| Frame rate | 30 Hz | 30 Hz | ✓ |
| Bin size | 10 frames | 10 frames | ✓ |

### Check 5: Edge cases
- ME length mismatches: handled by padding with last value (up to 116 frames = 0.3% of session)
- Trial boundaries: clean division (36000/10/180 = 20, 54000/10/180 = 30, no remainders)
- First/last bin time: 0.1667s to 1199.8s (20min) or 1799.8s (30min) - correct

### Issues Found and Resolved
- No issues found. All checks pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (102.7 → 1.09 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| motion_energy | 0.6211 | 0.3048 | Chance: 0.20, 1.52x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio |
|----------|------------------|--------|-------|
| motion_energy | 0.3048 | 0.20 | 1.52x |

### Check 1: Accuracy vs chance
- Validation accuracy 0.3048 is 1.52x chance (0.20)
- This is above the 1.5x threshold, indicating the decoder is learning meaningful patterns

### Check 2: Accuracy comparison to paper
- The paper uses ridge regression for decoding motion energy, not a neural network
- The paper does not report specific accuracy numbers for motion energy decoding in a comparable format
- The paper focuses on cross-day decoding stability rather than absolute accuracy
- The paper shows that motion energy can be decoded from neural activity, which our results confirm
- Given that this is spontaneous activity (not task-driven), moderate accuracy is expected

### Check 3: Train vs validation gap
- Training: 0.6211, Validation: 0.3048
- Ratio: 2.04x (training is ~2x validation)
- This gap suggests some overfitting, which is expected given:
  - High-dimensional neural data (221-746 neurons)
  - Relatively short trials (180 timepoints)
  - 5-class classification
  - The decoder architecture may be more complex than needed for this data

### Issues Found and Resolved
- No critical issues. Accuracy is above chance and the model is learning.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
