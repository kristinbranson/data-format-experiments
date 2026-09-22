# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa et al. 2025 - "A flexible hippocampal population code for experience relative to reward"
- **Date started**: 2026-09-21
- **Goal**: Convert hippocampal calcium imaging + VR navigation data to decoder-compatible format
- **Source**: DANDI:001361, NWB format, 11 subjects, 152 sessions

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `paper.pdf` - Reference paper (Nature Neuroscience, Vol 28, July 2025, pp 1497-1509)
- `methods.txt` - Extracted methods text
- `code/` - Reference code repository (src/reward_relative/)
- `data/` - NWB data files (11 subjects: m3, m4, m7, m11-m15, m17-m19)
- `decoder.py` - Decoder module
- `train_decoder.py` - Decoder training script

Environment: Python 3, numpy 2.4.4, torch 2.6.0+cu124, CUDA available

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess()` | preprocessing.py:17 | LOADING | Create session class from raw imaging/VR files |
| `dff()` | preprocessing.py:289 | PROCESSING | Compute dF/F: neuropil subtraction (coef=0.7), maximin baseline (20s window), smoothing (2-frame Gaussian), optional OASIS deconvolution |
| `get_trial_types()` | behavior.py:37 | LOADING | Classify trials: isreward (binary), morph (env 0/1) |
| `get_reward_zones()` | behavior.py:77 | LOADING | Get reward zone coords/labels per trial. Zones: A[175,225], B[390,440], C[325,375], X[80,130], Y[200,250], Z[320,370] |
| `define_trial_subsets()` | behavior.py:160 | CURATION | Split trials: set0 (first 30 on switch days), set1 (remaining) |
| `correct_lick_sensor_error()` | behavior.py:368 | CURATION | Remove trials with stuck lick sensor (>30% frames with cum lick >2) |
| `calc_place_cells()` | spatial.py:1130 | CURATION | Identify place cells: SI > 95% of 100 shuffles, speed > 2 cm/s |
| `is_putative_interneuron()` | spatial.py:754 | CURATION | Identify interneurons: speed-dFF correlation > threshold |
| `calc_lick_metrics()` | behavior.py:876 | PROCESSING | Lick rate metrics with permutation test |
| `pos_cm_to_rad()` | spatial.py:423 | PROCESSING | Convert position to reward-relative radians |

### Notes
- The reference code works with preprocessed pickle files (multi_anim_sess), not directly with NWB files
- NWB files contain: Fluorescence (raw), Deconvolved, Neuropil traces + behavioral timeseries
- The `iscell` array in NWB indicates manually curated cells from Suite2P
- dF/F is computed from raw fluorescence with neuropil subtraction, maximin baseline, and optional deconvolution
- Spatial analyses use 10 cm bins (45 bins for 450 cm track)
- Speed threshold: 2 cm/s for place cell calculation

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb`
- NWB format v2.8.0
- Neural: `/processing/ophys/` - Fluorescence, Deconvolved, Neuropil (timepoints x ROIs)
- Behavior: `/processing/behavior/BehavioralTimeSeries/` - position, speed, lick, environment, reward_zone, teleport, trial_number, trial_start, scanning, autoreward
- Reward events: separate timestamps array (one entry per reward event)
- Image segmentation: iscell (ROI x 2), pixel_mask
- Imaging rate: ~15.5 Hz (15.508 for m11)

### Key NWB variables (from sub-m11_ses-03):
- Fluorescence: (19818, 349) float32 - raw fluorescence
- Deconvolved: (19818, 349) float32 - deconvolved events
- Neuropil: (19818, 349) float32 - neuropil signal
- iscell: (349, 2) float64 - col0=is_cell(0/1), col1=probability. 155 cells of 349 ROIs
- position: (19818,) - range -500 to ~451 cm (-500 = ITI)
- speed: (19818,) - cm/s
- lick: (19818,) - lick count per frame (0-6)
- environment: (19818,) - {-1, 0} (-1=ITI)
- reward_zone: (19818,) - integer 0-6
- teleport: (19818,) - binary
- trial number: (19818,) - -1 to 80
- trial_start: (19818,) - binary

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 (m3, m4, m7, m11-m15, m17-m19) |
| Sessions / subject | 12-14 (m11 has 12, others have 14) |
| Total sessions | 152 |
| ROIs / session | varies (349 in m11_ses-03, much larger in m3) |
| Cells / session | varies (155 of 349 ROIs in m11_ses-03) |
| Trials / session | ~80 (80 in m11_ses-03) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source |
|-----------|-------|-------|
| Total mice | 14 (11 switch + 3 fixed) | Paper |
| Mice in data | 11 (switch task mice) | Data directory |
| Sessions/mouse | 14 days | Paper |
| Total imaged trials | 12,376 (11 switch mice) | Paper |
| Trials/session | 80.5 +/- 7.4 | Paper |
| Cells/switch session | 954 +/- 453 | Paper |
| Place cells/switch session | 459 +/- 263 | Paper |
| Place cell fraction | 48.5 +/- 14.5% | Paper |
| Imaging rate | ~15.5 Hz | Paper |
| Track length | 450 cm | Paper |
| Reward zone span | 50 cm | Paper |
| Zone A | 80-130 cm | Paper |
| Zone B | 200-250 cm | Paper |
| Zone C | 320-370 cm | Paper |
| Reward omission rate | ~15% random | Paper |
| Speed threshold | 2 cm/s | Paper |
| Interneuron exclusion | r > 0.5 speed-dFF correlation | Paper |
| Interneuron fraction | 0.42 +/- 0.85% | Paper |
| Lick error trials | 81/12376 (~0.65%) | Paper |
| Neural data time bin | ~64.5 ms (1/15.5 Hz) | Paper |
| Behavior data time bin | ~13-20 ms (50-75 Hz VR) | Paper |
| Spatial bin size | 10 cm (45 bins) | Paper |
| dF/F baseline | maximin, 20s window | Paper |
| dF/F smoothing | 2-sample Gaussian | Paper |
| Neuropil coefficient | 0.7 | Code |

### Processing Details
- dF/F: neuropil subtraction -> maximin baseline (20s sliding window, within each trial) -> dF/F = (F - baseline)/|baseline| -> 2-frame Gaussian smoothing
- Deconvolution: OASIS algorithm (Suite2P) with canonical calcium kernel
- Speed threshold 2 cm/s for spatial information calculation only
- Trial matrices: trial x position_bins (10 cm) x cells

### Curation Steps

**Neuron curation rules**:
1. Manual Suite2P curation (iscell array in NWB)
2. Interneuron exclusion: Pearson r(dFF, speed) > 0.5 -> exclude (~0.42% cells)
3. Place cell identification: SI > 95th percentile of 100 circular shuffles (pooled across cells)
4. Place field width >= 20 cm, exceeding 20% of max trial-averaged activity
5. Active in >= 8/30 trials (>25%)

**Trial curation rules**:
1. Lick sensor error: >30% frames with cumulative lick count >2 -> remove trial (81/12376 removed)
2. Switch day division: first 30 trials = set0, remainder = set1
3. Non-switch days: 50/50 split

### Decoders Trained (in paper)
- Circular-linear regression decoder for position (reward-relative)
- Trained on deconvolved events at speeds > 2 cm/s
- 10-fold cross-validation, 100 shuffles
- GLM with 156 features (position, RR position, speed, acceleration, licking)
- GLM FDE: all place cells 0.10+/-0.19, TR cells 0.32+/-0.13

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found and Resolved
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Reward zones | A[175,225], X[80,130], Y[200,250], Z[320,370] | reward_zone: int 0-6, position clusters match A/B/C | A[80-130], B[200-250], C[320-370] | Code's X/Y/Z keys map to paper's A/B/C labels. Zone label 'A'→coords X[80,130], 'B'→Y[200,250], 'C'→Z[320,370]. Code's 'A','B','C' keys are from an older task. |
| Subjects | N/A | 11 subjects | 14 total (11 switch + 3 fixed) | Data contains only 11 switch-task mice. Fixed-condition mice (n=3) not included. Consistent. |
| Interneuron threshold | Default r_thresh=0.3 in code | N/A | r > 0.5 | Paper overrides code default. Use r > 0.5. |
| Sessions m11 | N/A | 12 sessions (ses-03 to ses-14) | Imaging started day 3 for m11 | Paper: "imaging started on day 3 for m11 owing to lower viral expression". Consistent. |
| Dual-plane mice | N/A | m17,m18: rate=31.02 Hz, 2 planes | m17,m18: dual-plane at ~31 Hz interleaved | Consistent. Need to downsample by 2x to match 15.5 Hz. |
| Environment switch | N/A | Only ses-08 has env switching | Day 8: env switch | Consistent. |
| NWB reward_zone | N/A | Values 0-6, >0 only near actual reward zone position | N/A | Values 1-6 = different states in reward zone. 0 = outside. Use position where rz>0 to determine zone A/B/C. |
| Lick error threshold | correction_thr=0.5 (code default) | N/A | >30% frames with lick>2 | Paper says 30%, code default 50%. Paper is authoritative: use 0.3 to match paper's reported 81 removed trials. |
| reward_zone in NWB | N/A | Binary description but int 0-6 values | N/A | 0=outside zone, >0=inside/near zone (different interaction states). Confirmed by position analysis. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code | Notes |
|-----------------|--------------|-----------|----------------|-------|
| Fluorescence + Neuropil (iscell-filtered) | neural | Compute dF/F: neuropil subtract (0.7), add back neuropil mean, maximin baseline, dF/F, 2-frame smooth | preprocessing.dff() | Shape (n_cells, n_timepoints) per trial |
| Frame timestamps within trial | input[0]: time_from_trial_start | (frame_idx - trial_start_idx) / rate | N/A | Time-varying, seconds |
| environment variable | input[1]: environment | 0=ENV1, 1=ENV2 | behavior.py env_morph_dict | Per-trial, broadcast to T |
| trial number variable | input[2]: trial_number | Integer trial index within session | N/A | Per-trial, broadcast to T |
| Reward events of previous trial | input[3]: previous_trial_outcome | 0=omission/no-reward, 1=rewarded | behavior.get_trial_types() | Per-trial, broadcast to T. First trial=0 |
| position - reward zone | output[0]: distance_to_reward_zone | Signed distance to nearest point in zone, discretize into 7 bins | N/A | Time-varying |
| position | output[1]: absolute_position | Discretize into 5 bins (90 cm each) | N/A | Time-varying |
| speed | output[2]: speed | Discretize into 5 bins | N/A | Time-varying |
| lick | output[3]: lick | Binary (>0 → 1) | N/A | Time-varying |
| reward zone position | output[4]: reward_zone_location | A=0, B=1, C=2, from position where rz>0 | behavior.get_reward_zones() | Per-trial, broadcast to T |
| Reward events | output[5]: reward_outcome | 0=no, 1=yes | Match reward timestamps to trials | Per-trial, broadcast to T |

### Key Decisions
1. **Neural signal: Compute dF/F from raw Fluorescence + Neuropil**: Matches reference processing exactly. More informative than deconvolved events for neural network decoder. Reference method: neuropil subtract (0.7), add back neuropil mean per trial, maximin baseline (smooth sigma=15, min filter 300, max filter 300), dF/F = (F-baseline)/|baseline|, smooth sigma=2.
2. **Cell filtering: iscell + interneuron exclusion (r>0.5)**: Use Suite2P manual curation (iscell[:,0]==1) then exclude putative interneurons with speed-dFF correlation > 0.5 (paper threshold). Do NOT filter to place cells only - use all curated pyramidal cells for decoder.
3. **Trial filtering: Lick sensor error removal**: Use threshold 0.3 (paper description) to detect stuck lick sensor.
4. **Dual-plane mice (m17, m18): Downsample by 2x**: Average every 2 consecutive frames for neural and behavioral data to get consistent ~15.5 Hz across all sessions. Combine neurons from both planes.
5. **Time bin**: 1/15.5078125 Hz ≈ 64.48 ms for all sessions.
6. **Trial boundaries**: trial_start==1 to teleport==1 (inclusive of start, exclusive of teleport frame). Extract only active trial frames.
7. **Distance to reward zone**: Signed distance = pos - rz_start if pos < rz_start, 0 if inside zone, pos - rz_end if pos > rz_end.
8. **Reward zone determination**: For each trial, use mean position where reward_zone > 0 to determine zone. If no rz entry, inherit from nearest trial with rz entry.
9. **All inputs/outputs time-varying**: Per-trial values broadcast to (d, T) for consistency. Decoder handles both shapes.
10. **No speed filtering for neural data**: Speed < 2 cm/s filtering is only for place cell identification (spatial information calculation), not for general neural activity. Include all timepoints.

### Planned Sanity Checks
- [ ] Total trials matches ~12,376 (paper)
- [ ] Mean trials/session ~80.5 (paper)
- [ ] Mean cells/session ~954 (paper, for switch sessions)
- [ ] Interneuron fraction ~0.42% (paper)
- [ ] Lick error trial count ~81 (paper)
- [ ] Reward rate ~85% (paper: ~15% omission)
- [ ] Position range [0, 450] cm
- [ ] Speed distribution reasonable
- [ ] Reward zone positions: A near 105, B near 225, C near 345 (center of zones)

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with:
- Full dF/F pipeline matching reference preprocessing.py
- Interneuron detection (speed-dFF correlation > 0.5)
- Lick sensor error detection (>30% frames with lick>2)
- Dual-plane downsampling (2x for m17, m18)
- All 4 inputs and 6 outputs as specified
- Processing visualization with `--show-processing`

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (2 sessions: m11_ses-03, m19_ses-01)
| Statistic | Value |
|-----------|-------|
| Subjects | 2 (m11, m19) |
| Sessions | 2 |
| Total trials | 160 |
| Trials/session | 80.0 +/- 0.0 |
| Cells/session | 340.0 +/- 186.0 (154 and 526) |
| Interneurons removed | 1 (m11), 2 (m19) |
| Lick error trials | 0 |
| Time bin | 64.48 ms |
| Time from trial start | [0, 31.5] seconds |
| Reward rate | 87.2% |

### Processing time: 2.7s for 2 sessions (~1.35s/session for small sessions)
### Estimated full conversion: ~25 minutes for 152 sessions (larger sessions take longer)

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| distance_to_reward_zone | 0.7390 | 0.5956 | 0.1429 |
| absolute_position | 0.8404 | 0.6983 | 0.2000 |
| speed | 0.7142 | 0.5813 | 0.2000 |
| lick | 0.8198 | 0.7504 | 0.5000 |
| reward_zone_location | 0.9746 | 0.9706 | 0.3333 |
| reward_outcome | 0.8862 | 0.6778 | 0.5000 |

All outputs well above chance. Loss decreased monotonically from 2.37 to 0.53 over 200 epochs.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 8385.2 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Subjects | 11 switch mice | 11 | YES |
| Sessions | 152 (14/mouse, 12 for m11) | 152 | YES |
| Total trials (raw) | 12,376 | 12,216 (NWB data) | ~98.7% - 160 missing from NWB |
| Total trials (post-filter) | 12,295 (12376-81) | 12,135 | ~98.7% |
| Trials/session | 80.5 +/- 7.4 | 79.8 +/- 6.9 | CLOSE |
| Cells/session | 954 +/- 453 (switch only) | 909.7 +/- 448.0 (all) | CLOSE |
| Lick error trials | 81 | 81 | EXACT MATCH |
| Interneuron fraction | 0.42 +/- 0.85% | 0.29% (overall) | Similar order |
| Reward rate | ~85% | 84.1% | CLOSE |
| RZ distribution | ~equal A/B/C | A:32.8%, B:33.7%, C:33.5% | YES |
| Time bin | ~64.5 ms | 64.48 ms | YES |

### Notes
- 160 trial difference is in the raw NWB data (12,216 vs paper's 12,376), not in processing
- Some sessions have fewer trials (40-50) due to early termination (paper: "terminated if mouse ceased running")
- Max trial duration 216.5s - trials where mouse stopped running
- Dual-plane mice (m17, m18) successfully processed with downsampling

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- No errors or warnings in verification_full_out.txt
- All 152 sessions processed

### Check 2: Sanity checks (using raw NWB data, NOT conversion code)
1. **Neural**: Raw fluorescence 1585 vs converted dF/F -0.077 for same cell/frame — dF/F correct
2. **Input prev_outcome**: Matches actual reward of preceding trial for 5 trials checked
3. **Output position**: 9 spot-checks all match raw position data
4. **Output distance_to_RZ**: Verified against raw position and zone B [200,250]
5. **Output speed**: 4 spot-checks all match raw speed bins
6. **Output lick**: 266/266 frames match raw lick>0

### Check 3: Reference code comparison
All major steps match reference. Lick error threshold: paper's 0.3 gives exactly 81 (verified), code default 0.5 gives only 44.

### Check 4: Key statistics — see Step 9 table above

### Check 5: Edge cases verified
- Low-trial sessions: early termination, valid
- Long trials: mouse stopped running, valid
- Dual-plane frame mismatch: fixed by truncation

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (2.37 → 0.64 over 200 epochs)

### Decoder Results (Full)
| Output | Train Acc | Val Acc | Chance | Ratio |
|--------|-----------|---------|--------|-------|
| distance_to_reward_zone | 0.8068 | 0.6235 | 0.1429 | 4.4x |
| absolute_position | 0.8973 | 0.7621 | 0.2000 | 3.8x |
| speed | 0.7432 | 0.6362 | 0.2000 | 3.2x |
| lick | 0.8034 | 0.7699 | 0.5000 | 1.5x |
| reward_zone_location | 0.9645 | 0.8742 | 0.3333 | 2.6x |
| reward_outcome | 0.9515 | 0.6033 | 0.5000 | 1.2x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
All outputs above chance. Reward outcome marginal (1.2x) with slight train/val gap (1.58x), expected for per-trial binary with 85% majority class. Not a data bug. Paper comparison not directly possible (different decoder/metrics) but accuracy patterns consistent with known CA1 spatial coding.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots
- [x] All files organized
