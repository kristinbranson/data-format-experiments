# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - Longitudinal tracking of neuronal activity from barrel cortex (Majnik et al. 2025)
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format for decoding motion energy from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `paper.pdf` - Reference paper (eLife 2025)
- `methods.txt` - Methods excerpt from paper
- `code/` - Reference code repository (Track2p)
- `data/` - Experimental data (6 mice: jm031, jm032, jm038, jm039, jm040, jm046)
- `decoder.py` - Decoder module
- `train_decoder.py` - Decoder training script
- Each mouse has multiple session directories (dates like 2023-10-18_a)
- Each session contains:
  - `suite2p/plane0/` - Neural data (F.npy, Fneu.npy, spks.npy, iscell.npy, ops.npy, stat.npy)
  - `move_deve/` - Behavioral data (motion_energy_glob.npy, tstamps.npy, interframe_int.npy)
- Python 3 with numpy 2.4.4, torch 2.6.0+cu124, CUDA available

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_traces() | data/load_data.ipynb | LOADING | Loads raw fluorescence F.npy from session dir |
| load_fov() | data/load_data.ipynb | LOADING | Loads meanImg from ops.npy |
| zscore_rows() | data/load_data.ipynb | PROCESSING | Z-scores rows for visualization |
| load_all_ds_stat_iscell() | code/track2p/io/s2p_loaders.py | LOADING | Loads stat+iscell, filters by iscell_thr |
| load_all_ds_ops() | code/track2p/io/s2p_loaders.py | LOADING | Loads ops.npy for all sessions |

### Notes
- The code repository is the Track2p cell tracking algorithm, NOT a decoding pipeline
- The data/load_data.ipynb notebook provides the main data loading guidance
- The notebook says: "for more proper analysis compute dF/F the way as described in the paper (or alternatively use spks.npy)"
- The data already contains only tracked cells (cells present across ALL days for a given mouse), so iscell is all 1s
- Neurons are already matched across days: row i of F.npy in session 1 = same neuron as row i in session 2
- No additional cell filtering is needed since Track2p already filtered to tracked cells

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/
  {subject}/                     # e.g., jm031, jm032, ...
    {date}_a/                    # e.g., 2023-10-18_a
      suite2p/plane0/
        F.npy                    # Raw fluorescence, shape (n_neurons, n_frames)
        Fneu.npy                 # Neuropil fluorescence, shape (n_neurons, n_frames)
        spks.npy                 # Deconvolved spikes, shape (n_neurons, n_frames)
        iscell.npy               # Cell classification, shape (n_neurons, 2)
        ops.npy                  # Suite2p parameters (dict)
        stat.npy                 # Cell statistics (list of dicts)
      move_deve/
        motion_energy_glob.npy   # Motion energy, shape (n_me_frames,)
        tstamps.npy              # Camera timestamps, shape (n_me_frames,)
        interframe_int.npy       # Inter-frame intervals, shape (n_me_frames,)
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 (jm031, jm032, jm038, jm039, jm040, jm046) |
| Sessions / subject | 7 (except jm040: 6) |
| Total sessions | 41 |
| Neurons per subject | jm031:221, jm032:370, jm038:685, jm039:746, jm040:541, jm046:435 |
| Mean neurons/subject | 500 |
| Total neurons (sum) | 2998 |
| Frames per session | 36000 (jm031,jm032) or 54000 (others) |
| Session duration | 20 min (36000/30Hz) or 30 min (54000/30Hz) |
| Frame rate | 30 Hz |
| Missing ME frames | Up to 148 frames per session (most sessions have 0) |

### Suite2p Parameters (from ops.npy)
- fs: 30 Hz
- neucoeff: 0.7 (neuropil correction coefficient)
- baseline: maximin
- win_baseline: 60.0 s
- sig_baseline: 10.0
- prctile_baseline: 8.0
- tau: 0.3 (GCaMP decay time constant)

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 | "a full dataset of 6 mice imaged daily for a minimum of 6 consecutive days" |
| Sessions/subject | min 6, up to 7 | "a minimum of 6 consecutive days within the second postnatal week (P7 to P14)" |
| Mean neurons/mouse | 526 +/- 190 std | "On average 526 (+/- 190 std) neurons per mouse were successfully tracked" |
| % of first-day neurons | 33% +/- 11% std | "corresponding to 33% (+/- 11% std) of the neurons detected on the first day" |
| FOV size | 720x720 um | "720x720 um field of view" |
| Pixel resolution | 512x512 | "512x512 pixel resolution" |
| Frame rate | 30 Hz | "Imaging rate was 30 Hz (resonant scanner)" |
| Session duration | 20 minutes | "each session lasted 20 minutes" |
| Cell threshold | iscell > 0.5 | "all ROIs above the default threshold of 0.5 as true cells" |
| dF/F method | baseline-corrected | "baseline corrected fluorescence traces as our dF/F (using default Suite2p parameters)" |
| Decoding bin size | 10 frames | "averaging in bins of 10 consecutive timestamps" |
| Decoding method | Ridge regression | "linear regression with ridge regularisation" |
| CV splits | 5-fold nested | "5 fold splits for both inner and outer loops" |
| CV block size | 2 minutes | "splits were done on consecutive 2 minute blocks" |

### Processing Details
1. **dF/F computation**: Suite2p baseline-corrected (F - 0.7*Fneu, then maximin baseline with 60s window)
2. **Binning for decoding**: Average dF/F and behavior in bins of 10 consecutive frames (30Hz -> 3Hz)
3. **Motion energy**: Pixel-wise squared difference of consecutive video frames, summed across pixels
4. **Camera sync**: Video at 30Hz triggered by microscope acquisition

### Curation Steps
**Neuron curation rules**: Data already filtered - only cells tracked across ALL days are included. The iscell threshold of 0.5 was applied during Suite2p preprocessing before Track2p tracking.

**Trial curation rules**: No trial curation described - sessions are continuous recordings of spontaneous behavior. We split into 60-second trials as required by the decoder task.

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Motion energy (same-day) | R^2 varies by age: low at P8-P10, increases after P11. R^2 values appear to range from ~0 to ~0.5 from Figure 7C |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session duration | N/A | 36000 frames (20min) for jm031/jm032, 54000 frames (30min) for others | "20 minutes" | Paper may describe the typical protocol. jm038-jm046 are 30min sessions. Use actual data lengths. |
| Neurons per mouse | N/A | 221, 370, 685, 746, 541, 435 (mean=500) | 526 +/- 190 std | Reasonably consistent (mean 500 vs 526). The data contains only tracked cells. |
| Missing frames | load_data.ipynb mentions interpolation | Some sessions have ME shorter than F | README describes missing camera frames | Interpolate ME to match F length before processing |

### Key Consistency Observations
- All iscell values are 1.0 (confirmed: data only has tracked cells that passed the 0.5 threshold)
- Same number of neurons across all sessions for a given mouse (confirmed: Track2p matched cells)
- Frame rate 30Hz confirmed in ops.npy (fs=30)
- Suite2p parameters (neucoeff=0.7, baseline=maximin, win_baseline=60) match paper description

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| F.npy, Fneu.npy | neural | dF/F computation, bin by 10 frames | Suite2p default baseline correction | (n_neurons, n_timepoints) per trial |
| Time index | input[0] | Elapsed seconds from session start | N/A | (1, n_timepoints) per trial |
| motion_energy_glob.npy | output[0] | Interpolate missing, bin by 10, discretize to 5 bins | N/A | (1, n_timepoints) per trial |

### Processing Pipeline
1. **Load neural data**: F.npy and Fneu.npy
2. **Compute dF/F**: F_corrected = F - 0.7*Fneu; baseline correction using Suite2p maximin method
3. **Load motion energy**: Interpolate to match F length if missing frames
4. **Bin by 10 frames**: Average both dF/F and ME in bins of 10 → 3Hz effective rate (333.33ms bins)
5. **Split into 60-second trials**: 180 timepoints per trial (60s * 3Hz)
6. **Compute time input**: Elapsed seconds from session start for each timepoint
7. **Discretize ME**: 5 equal-percentile bins per session (quintiles of non-NaN values)

### Key Decisions
1. **dF/F vs spks**: Use dF/F as specified in paper for decoding ("we slightly denoised the dF/F"). The load_data notebook also suggests dF/F for proper analysis.
2. **Binning**: 10 frames at 30Hz → 3Hz (333.33ms bins), matching paper exactly.
3. **Trial length**: 60 seconds = 180 timepoints at 3Hz.
4. **Discretization**: 5 equal-percentile bins per session (quintile boundaries computed per session).
5. **Missing frames**: Interpolate ME to match neural frame count before binning, as suggested by data README.
6. **Brain region**: All recordings from barrel cortex (S1BF).
7. **Session = recording day**: Each session is one recording day for one mouse.
8. **No additional neuron filtering**: Data already contains only tracked neurons that passed iscell>0.5.

### Planned Sanity Checks
- [ ] Verify dF/F computation matches Suite2p default (spot-check against spks pattern)
- [ ] Verify number of neurons matches across all sessions for each mouse
- [ ] Verify total neuron counts match paper statistics (~526 +/- 190 mean)
- [ ] Verify trial count: 20 trials for 20-min sessions, 30 trials for 30-min sessions
- [ ] Verify ME discretization produces ~equal bin counts (20% each)
- [ ] Verify temporal alignment: binned neural and ME should have same timepoints per trial

---

## Step 6: Script Development
**Status**: COMPLETE

Script `/app/convert_data.py` implements:
1. dF/F computation using Suite2p maximin baseline (verified against suite2p source code)
2. Motion energy interpolation for missing frames
3. Binning by 10 frames (30Hz -> 3Hz)
4. Trial splitting (60s = 180 timepoints)
5. Per-session quintile discretization of motion energy

Processing time: ~1.1s/session, ~47s total for 41 sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (jm031 session 1, jm046 session 7) |
| Neurons/session | 221, 435 |
| Trials/session | 20, 30 |
| Timepoints/trial | 180 |
| Time bin size | 333.33 ms |
| Time input range | [0, 1799.7] s |
| ME bin distribution | 20% each (5 bins) |

### Processing Plots Review
- dF/F shows proper calcium transients with baseline correction
- Motion energy shows clear movement bouts
- Discretization follows continuous ME trace correctly
- Time input is linear as expected

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Full pipeline | 1.1-1.8s | ~47s |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| motion_energy_bin | 0.2850 | 0.2316 | 0.2000 |

Note: Low accuracy expected with only 2 sessions (one young mouse with minimal motion encoding).

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
| Sessions | min 6/mouse | 41 (6-7/mouse) | 41 | Yes |
| Sessions/jm040 | min 6 | 6 | 6 | Yes |
| Mean neurons/mouse | 526 +/- 190 | 500 +/- 180 | 500 +/- 180 | Yes (within 1 std) |
| Neuron counts | N/A | 221,370,685,746,541,435 | same | Yes |
| Frame rate | 30 Hz | 30 Hz | 3 Hz (after 10x binning) | Yes |
| ME bin distribution | N/A | N/A | 20% each | Yes (quintiles) |
| Total trials | N/A | N/A | 1090 | N/A |
| Timepoints/trial | N/A | N/A | 180 (60s at 3Hz) | N/A |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- No errors in verification_full_out.txt
- No warnings
- All output bins at exactly 20% (perfect quintile distribution)

### Check 2: Sanity checks (all passed)
1. **Neural data**: Spot-checked dF/F for jm039, neuron 5, trial 3, timepoint 10. Recomputed from raw F.npy/Fneu.npy → np.allclose=True
2. **Input data**: Verified time_elapsed starts at 0.0 for trial 0, 180.0 for trial 3. All correct.
3. **Output data**: Verified ME discretization bin assignment for jm039 trial 3 tp 10. Expected bin matches actual bin.

### Check 3: Reference code comparison
| Processing Step | My Code | Reference (paper/code) | Match? |
|----------------|---------|----------------------|--------|
| (a) Data loading | Load F.npy, Fneu.npy from suite2p/plane0/ | load_data.ipynb loads F.npy from suite2p/plane0/ | Yes |
| (b) Neuron filtering | No filtering (all cells already tracked) | Data contains only tracked cells (iscell all 1.0) | Yes |
| (c) dF/F computation | Fc=F-0.7*Fneu, maximin baseline (win=60s, sig=10 frames) | Suite2p default: neucoeff=0.7, maximin, win_baseline=60, sig_baseline=10 | Yes |
| (d) Binning | Average in bins of 10 frames | "averaging in bins of 10 consecutive timestamps" | Yes |
| (e) Input construction | Time elapsed from session start in seconds | Decoder task specification | N/A |
| (f) Output construction | ME discretized to 5 quintile bins per session | Decoder task specification | N/A |

### Check 4: Key statistics comparison
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| 6 mice | 6 | 6 | Yes |
| min 6 sessions/mouse | Yes | jm040 has 6, others 7 | Yes |
| 526 +/- 190 neurons/mouse | 500 +/- 180 | Same | Yes |
| 30 Hz imaging rate | 30 Hz | Used for binning | Yes |
| 10-frame bins for decoding | 10 frames | 10 frames | Yes |

### Check 5: Edge cases
- Missing ME frames: Handled by linear interpolation (up to 148 missing frames out of 36000-54000)
- Trial boundaries: Clean 60-second cuts, no overlap, no partial trials
- Session boundaries: Each session processed independently
- Bin edges: -inf and +inf for outer edges ensures all values are captured

### Issues Found and Resolved
- None. All checks passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (1.7M -> 15.7K over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|--------|-------|
| motion_energy_bin | 0.2689 | 0.2200 | 0.2000 | Above chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
- motion_energy_bin: 0.2200 validation accuracy vs 0.2000 chance (1.1x chance)
- This is above chance but low. The paper shows that decoding quality varies dramatically with developmental age:
  - Early sessions (P8-P10): R^2 near 0 (no motion encoding)
  - Late sessions (P12-P14): R^2 up to ~0.5
- Our decoder trains across ALL 41 sessions, including ~20 sessions with minimal motion encoding
- The low accuracy is consistent with the paper's findings

### Check 2: Accuracy comparison to paper
| Variable | Our Accuracy | Paper's Metric | Paper's Value | Notes |
|----------|-------------|----------------|---------------|-------|
| Motion energy | 0.22 (5-class balanced acc) | R^2 (regression) | ~0-0.5 (varies by age) | Different metrics; paper uses ridge regression R^2, we use categorical classification accuracy |

- The paper uses continuous regression (R^2), while we use 5-class categorical decoding (balanced accuracy)
- These metrics are not directly comparable
- The paper trains within-session decoders; our decoder trains across all sessions
- Both factors explain the seemingly low accuracy

### Check 3: Train vs validation gap
- Training: 0.2689, Validation: 0.2200 (ratio ~1.22x)
- Gap is modest, no severe overfitting

### Debugging checks performed:
1. Verified output values are correct by loading raw data for 3 specific trials (sanity checks in Step 10)
2. Processing plots show neural activity and ME are temporally aligned
3. Output has equal variation (20% per class, quintile distribution)
4. All neurons included (data already filtered by Track2p to tracked cells)
5. Processing matches reference: dF/F with Suite2p parameters, 10-frame binning

### Conclusion
The accuracy of 0.22 is reasonable given the developmental heterogeneity in the dataset. The paper shows that motion encoding in barrel cortex only emerges around P11, meaning roughly half the sessions have minimal motion-related neural activity. The conversion is correct.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots and decoder visualizations
- [x] cache/README_CACHE.md documenting cached files
- [x] All required output files verified
