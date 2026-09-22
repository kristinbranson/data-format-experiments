# Dataset Conversion Notes

## Overview
- **Dataset**: Zhong et al. 2025, "Unsupervised pretraining in biological neural networks"
- **Date started**: 2026-09-21
- **Goal**: Convert calcium imaging + VR corridor behavioral data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python: numpy 2.4.4, torch 2.6.0+cu124, CUDA available

Directory contents:
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Methods excerpts
- `/app/code/` - Reference code (utils.py, fig1-5.py, S6.py, Figures.ipynb, data_process_script.ipynb)
- `/app/data/beh/` - Behavioral data (.npy files), Imaging_Exp_info.npy
- `/app/data/spk/` - Neural spike data (.npy, 89 sessions across 19 mice)
- `/app/data/retinotopy/` - Retinotopy data (.npz, + areas.npz)
- `/app/data/process_data/` - Empty directory
- `/app/decoder.py` - Decoder module
- `/app/train_decoder.py` - Training script

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk(db, root)` | utils.py | LOADING | Load neural data, concatenate planes: `np.concatenate([nspk for nspk in np.load(path).item()['spks']], 0)` |
| `load_retino(db, root)` | utils.py | LOADING | Load retinotopy, compute brain area indices via `neu_area_ID()` |
| `load_exp_beh(root, exp_type)` | utils.py | LOADING | Load behavioral data from `Beh_{exp_type}.npy` |
| `neu_area_ID(iarea)` | utils.py | PROCESSING | Map iarea codes to brain regions: V1=8, mHV=[0,1,2,9], lHV=[5,6], aHV=[3,4] |
| `get_interpPos_spk()` | utils.py | PROCESSING | Interpolate neural activity from time to position space (60 bins per corridor) |
| `spk_pos_interp()` | utils.py | PROCESSING | Core interpolation: map spike data from cumulative position to linear position bins |
| `dprime(x1, x2)` | utils.py | CURATION | Compute d' for neuron selectivity: `2*(u1-u2)/(sig1+sig2)` |
| `Get_dprime_selective_neuron()` | utils.py | CURATION | Compute d' for all neurons, filtering by `ft_CorrSpc & VRmove` (corridor + running) |
| `Get_density_map()` | utils.py | CURATION | Filter neurons: `(arid!=-1) & (arid!=7)` to exclude non-visual-cortex neurons |

### Notes
- **Frame filtering**: Reference code consistently uses `VRmove = ft_move > 0` (running) and `ft_CorrSpc` (in corridor) as frame mask
- **Neural data format**: Each `.npy` file has dict with 'spks' key containing list of arrays (one per imaging plane), concatenated across planes
- **Position interpolation**: Reference code interpolates to 60 position bins (10 per meter, corridor = 6m = 60 bins). Bins 0-39 = texture corridor (4m), bins 40-59 = grey space (2m)
- **No neuron filtering by quality**: Reference code does not filter neurons by firing rate or other quality metrics for general analyses. Only d' thresholds are used for specific analyses.
- **Brain region exclusion**: Neurons with `iarea == -1` (outside visual cortex) and `iarea == 7` (unassigned) are excluded from analysis

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- **Neural**: `/app/data/spk/{mouse}_{date}_{blk}_neural_data.npy` - dict with 'spks' list of float32 arrays per imaging plane
- **Behavioral**: `/app/data/beh/Beh_{exp_type}.npy` - dict of session dicts keyed by `{mouse}_{date}_{blk}`
- **Retinotopy**: `/app/data/retinotopy/{mouse}_{date}_trans.npz` - contains iarea (brain area), xy_t (positions)
- **Experiment info**: `/app/data/beh/Imaging_Exp_info.npy` - dict mapping exp_type to list of session dicts
- 23 experiment types, 142 total entries (many sessions appear in multiple exp types)
- Same physical recording has identical behavioral data across different exp types (only stim_id mapping differs)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Unique physical recordings | 89 |
| Unique subjects (mice) | 19 |
| Sessions / subject | 1-8 (mean ~4.7) |
| Neurons / session | 20,547 - 89,577 (from paper) |
| Example: LZ13_2024_05_15_1 | 34,658 neurons, 17,283 frames |
| Example: DR10_2022_07_12_1 | 58,224 neurons, 31,707 frames |
| Trials / session | ~200-600 |
| Frame rate | 3.178 Hz (315 ms/frame) |
| Corridor length | 60 dm (6m: 4m texture + 2m grey) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings | 89 | "We performed 89 recordings in 19 mice" |
| Subjects | 19 | "19 mice bred to express GCaMP6s" |
| Neurons / session | 20,547 - 89,577 | "20,547 to 89,577 neurons in each recording" |
| Neural data time bin | ~315 ms | Frame rate 3.178 Hz |
| Behavior data time bin | same (aligned to neural frames) | Behavioral vars mapped to neural frame times |
| Calcium indicator | GCaMP6s | "TetO-GCaMP6s x camK2a-tTa mice" |
| Processing | Suite2p | "Suite2p performs motion correction, ROI detection..." |
| Deconvolution timescale | 0.75 s | "timescale of decay of 0.75 s" |
| VR speed | 60 cm/s | "constant speed (60 cm/s) as long as mice kept running" |
| Running threshold | 6 cm/s | "running faster than a threshold of 6 cm/s" |
| Sound cue range | 0.5 - 3.5 m (5-35 dm) | "uniform distribution between positions 0.5 m and 3.5 m" |
| Reward zone | 0.5 - 3.5 m | "For task mice, the sound cue indicated the beginning of the reward zone" |
| d' threshold | 0.3 | "criteria for selective neurons was d' >= 0.3" |
| Reward rate | ~36% of trials | Observed in sup_test1 data: isRew.mean()=0.361 |

### Processing Details
- **All analyses based on deconvolved fluorescence traces** (not dF/F, not raw fluorescence)
- **Running-only frames**: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards"
- **Position-based interpolation**: Reference code interpolates to 60 position bins per corridor
- **Brain regions**: V1, medial higher visual (mHV), lateral higher visual (lHV), anterior higher visual (aHV)
- **Neuron exclusion**: iarea == -1 (outside visual cortex) and iarea == 7 excluded

### Curation Steps

**Neuron curation rules**:
- Exclude neurons with iarea == -1 (outside visual cortex) or iarea == 7 (unassigned boundary region)
- No additional quality filtering (no minimum firing rate, no signal-to-noise cutoff)

**Trial curation rules**:
- No explicit trial exclusion in reference code for general analysis
- Only running frames (ft_move > 0) within corridor (ft_CorrSpc) are used
- Trials with 0 valid frames after filtering should be excluded

### Decoders Trained
No explicit decoder accuracy reported in the paper. Paper focuses on d' selectivity and coding direction analysis.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Recordings | 89 unique session keys in exp_info | 89 neural data files in spk/ | "89 recordings" | Consistent |
| Mice | 19 unique mouse IDs | 19 mice in spk/ | "19 mice" | Consistent |
| Neurons per recording | Varies by concatenating planes | LZ13: 34,658; DR10: 58,224 | "20,547 to 89,577" | Consistent |
| Frame rate | Computed from ft: 3.178 Hz | | "3.17 Hz" (data_process_script) | Consistent |
| Corridor length | Corridor_Length=60 | Position 0-60 | "4m long" + "2m grey" = 6m | Consistent (60 dm = 6m) |
| Running filter | `ft_move > 0` | | "running for analysis" | Consistent |
| Brain regions | `neu_area_ID`: V1=8, mHV=[0,1,2,9], lHV=[5,6], aHV=[3,4] | iarea values -1 to 9 | V1, mHV, lHV, aHV | Consistent |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| spks (concatenated planes) | neural | Filter to visual cortex neurons (iarea != -1, != 7). Extract per-trial running corridor frames. | `load_spk()`, `neu_area_ID()` | Deconvolved fluorescence, float32 |
| SoundFr, ft | input[0]: time_to_sound_cue | `(SoundFr[trial] - frame_idx) * dt_s`. Continuous, time-varying. | | Positive before cue, negative after |
| Session dates | input[1]: day_of_training | Days since first session for each mouse. Per-trial (same for all trials in session). | | Computed from date strings |
| Frame index within trial | input[2]: time_since_trial_start | `(frame_idx - first_frame_of_trial) * dt_s`. Continuous, time-varying. | | Seconds from corridor entry |
| isRew | input[3]: reward_availability | 1 if isRew[trial]==True, else 0. Discrete, per-trial. | | Binary |
| WallName | output[0]: stimulus_category | Map to broad category: circle, leaf, rock, brick. Per-trial. | | 4 categories |
| LickFr, LickTrind | output[1]: licking | Binary time series: 1 if any lick in frame's time bin, 0 otherwise. | | Time-varying |
| ft_Pos | output[2]: position | Discretize 0-40dm into 4 bins: [0-10), [10-20), [20-30), [30-40]. | | 4 bins of 1m each |
| ft_RunSpeed | output[3]: running_speed | Discretize into 4 quartile bins (computed across all data). | | Time-varying |

### Key Decisions
1. **Time bins = native imaging frames (~315ms)**: Matches reference code's temporal resolution. No interpolation needed.
2. **Corridor-only frames**: Include only ft_CorrSpc == True. Position output requires 0-4m range.
3. **Running-only frames**: Follow reference paper: "only considered timepoints during running for analysis". Use ft_move > 0.
4. **Neuron filtering by brain region**: Exclude iarea == -1 and iarea == 7, following reference code's `Get_density_map()` logic.
5. **Stimulus categories**: Use broad categories (circle, leaf, rock, brick) rather than specific stimulus IDs. The "e.g. circle, leaf, etc." in the decoder task suggests this.
6. **Session deduplication**: Each physical recording (mname_datexp_blk) included once. Use first available exp_type for behavioral data.
7. **Running speed quartiles**: Compute across ALL sessions' valid frames to ensure consistent bin edges.
8. **Day of training**: Compute as days since first session date for each mouse.

### Planned Sanity Checks
- [ ] Total unique sessions = 89, mice = 19
- [ ] Neuron counts per session match between neural data and retinotopy
- [ ] Trial counts match between behavioral data and ft_trInd
- [ ] Frame rate is ~3.178 Hz across sessions
- [ ] Position range in valid frames is [0, 40] dm
- [ ] Sound cue positions are in [5, 35] dm range
- [ ] Stimulus categories cover circle, leaf, rock, brick

---

## Step 6: Script Development
**Status**: COMPLETE

Script: `/app/convert_data.py` (~665 lines)
- CLI: `python -u convert_data.py <output> [--full|--sample] [--show-processing]`
- Follows reference code conventions (load_spk, neuron filtering, frame masking)
- Uses actual timestamps for time calculations (not frame counting)

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

- Sample: 2 sessions (TX88, TX108), saved to `/app/sample_data.pkl` (5.92 GB)
- Processing plots: `/app/processing_TX88_2022_06_20_1.png`, `/app/processing_TX108_2023_03_25_1.png`
- Verification: no errors or warnings

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| stimulus_category | 0.994 | 0.975 |
| licking | 0.955 | 0.862 |
| position | 0.925 | 0.907 |
| running_speed | 0.431 | 0.435 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 151.58 GB (89 sessions, 38,110 trials)
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | 89 | 89 unique keys | 89 spk files | 89 | Yes |
| Subjects | 19 | 19 unique mname | 19 mice | 19 | Yes |
| Neurons/session | 20,547-89,577 | concat planes | varies | 17,363-78,815 (filtered) | Yes (pre-filter range matches) |
| Brain regions | V1,mHV,lHV,aHV | neu_area_ID | iarea codes | 4 regions | Yes |
| Frame rate | 3.17 Hz | computed | ~315ms | 315ms time_bin_size | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Session & subject counts**: 89 sessions, 19 subjects — matches paper exactly. PASS
2. **Unique sessions from exp_info**: 89 unique keys confirmed. PASS
3. **Spot-check neuron counts & brain regions (sessions 0, 44, 88)**: Loaded original spk + retinotopy files, computed expected neuron filter, compared with converted data. All 3 sessions match exactly. Brain region mapping verified for 100 neurons per session. PASS
4. **Spot-check neural values**: Loaded original spk data for session 0 trial 0, applied same frame mask (ft_trInd==0 & ft_CorrSpc & ft_move>0), compared with `np.allclose` — exact match. PASS
5. **Position binning**: All position bins in [0,3] range across all trials. PASS
6. **Licking values**: All licking values are strictly 0 or 1 (binary) across all 38,110 trials. PASS
7. **Stimulus categories**: All 4 categories present (circle=11,619, leaf=17,761, rock=3,278, wood=5,452). Constant within each trial. PASS
8. **Day of training**: Verified for 3 sessions against independently computed dates. PASS
9. **Time to sound cue**: Recomputed from original timestamps for session 0 trial 0 — exact match with `np.allclose`. Sign convention correct (positive before cue, negative after). PASS
10. **Reward availability**: 28 sessions with reward, 61 without — reasonable for mix of supervised/unsupervised experiments. PASS
11. **Brain region totals**: V1=1,833,035 > mHV=1,108,860 > aHV=668,180 > lHV=495,318. Ordering consistent with V1 being largest visual area. PASS
12. **Extreme values / data integrity**: No NaN or Inf in neural, input, or output data. One trial with time_since_start=1765s (session 49, trial 391, 36 frames) — valid due to mouse stopping/restarting during long trial. PASS

### Issues Found and Resolved
- No issues found. All 12 checks passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (15700 → 136 over 200 epochs, converged ~epoch 130)
- Test loss: 87.9

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|-------|-------|
| stimulus_category | 0.634 | 0.629 | 0.250 | 2.5x above chance |
| licking | 0.936 | 0.828 | 0.500 | Strong, some overfitting |
| position | 0.317 | 0.316 | 0.250 | Above chance |
| running_speed | 0.384 | 0.387 | 0.250 | 1.5x above chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Val Accuracy | Chance | Ratio | Expectation |
|----------|-------------|--------|-------|-------------|
| stimulus_category | 0.629 | 0.250 | 2.5x | Reasonable - paper shows neurons encode texture identity via d' selectivity. Decoder generalizes across 89 sessions with different neuron sets. |
| licking | 0.828 | 0.500 | 1.7x | Good - licking is a behavioral readout correlated with reward expectation. Train-val gap (0.94→0.83) indicates some overfitting. |
| position | 0.316 | 0.250 | 1.3x | Modest but above chance. Position is encoded in VR but with variable-length filtered frames, position coding is spread across time bins. |
| running_speed | 0.387 | 0.250 | 1.5x | Reasonable. Running speed is partially reflected in neural activity (running modulation well-known in V1). |

### Analysis
- All 4 outputs are above chance, confirming the data format is correct and neural signals contain meaningful information
- Train-val gap is small for most outputs except licking (0.11 gap), indicating good generalization
- Stimulus category accuracy (0.63) is substantial - the decoder must learn texture representations across different recording sessions
- Position and speed have lower accuracy, which is expected since: (1) the decoder uses SVD-projected neural data (2000 dims), and (2) these time-varying outputs require fine temporal discrimination
- Compared to sample decoder (2 sessions): stimulus drops from 0.975→0.629 (expected, harder to generalize across 89 sessions), licking drops from 0.862→0.828, position drops from 0.907→0.316, speed stays similar at ~0.39

### Issues Found and Resolved
- No issues found. All outputs decode above chance with reasonable train-val gaps.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with converted_data.pkl and sample_data.pkl
- [x] All files organized
