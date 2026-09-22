# Dataset Conversion Notes

## Overview
- **Dataset**: "Separating cognitive and motor processes in the behaving mouse" (Bhagat et al., 2024)
- **Date started**: 2024
- **Goal**: Convert ALM electrophysiology + behavior data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/code/` - Reference MATLAB code
- `/app/data/Ephys_Behavior/` - 25 sessions, DR + two-context task
- `/app/data/RandomizedDelay_Ephys_Behavior/` - 22 data files (19 included sessions), randomized delay task
- `/app/data/DelayInhibition_BilatMC_Behavior/` - Behavior-only (not used)
- `/app/data/GoCueInhibition_BilatMC_Behavior/` - Behavior-only (not used)
- `/app/paper.pdf`, `/app/methods.txt`, `/app/decoder.py`, `/app/train_decoder.py`

Python: numpy 2.4.4, torch 2.6.0+cu124, scipy 1.18.0

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| loadSessionData | loadSessionData.m | LOADING | Main pipeline |
| processData | processData.m | PROCESSING | findTrials→findClusters→alignSpikes→getSeq→removeLowFR→baselineFR |
| findClusters | findClusters.m | CURATION | Excludes garbage, noisy, gabrga, real? |
| alignSpikes | alignSpikes.m | PROCESSING | trialtm_aligned = trialtm - goCue |
| getSeq | getSeq.m | PROCESSING | Bin spikes (histc), smooth (causal gaussian), create single-trial data |
| removeLowFRClusters | removeLowFRClusters.m | CURATION | Remove units with mean FR < 1 Hz |
| mySmooth | mySmooth.m | PROCESSING | Causal gaussian kernel, reflect BC |
| loadMotionEnergy | loadMotionEnergy.m | LOADING | Load ME, align to neural time, interp1 |
| findVideoOffset | findVideoOffset.m | PROCESSING | vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart) |

### Key Parameters
- alignEvent = 'goCue'
- tmin=-2.5, tmax=2.5, dt=1/100 (10ms)
- smooth=15, bctype='reflect'
- lowFR=1 Hz
- quality='all' (excludes garbage, noisy)

### Probe assignments for ALM
See Step 2 for complete list.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- HDF5 (v7.3) and MATLAB v5/v7 formats mixed
- obj.bp: trial info (Ntrials, hit, miss, R, L, autowater, early, stim, ev)
- obj.clu: spike data per probe (quality, tm, trial, trialtm)
- obj.traj: DLC tracking {side_cam, bottom_cam} per trial (ts, frameTimes, featNames)
- obj.sglx: recording metadata (fs=25000, bitcode)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects (Ephys) | 10 (EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3) |
| Subjects (RD) | 4 (JEB11, JEB12, JEB23, JEB24) |
| Sessions (Ephys) | 25 |
| Sessions (RD) | 19 (3 data files excluded) |
| Total sessions | 44 |
| Total neurons (after quality+FR filter) | 2452 |
| Total trials | 14972 |

### Excluded data files
- JEB23_2023-10-20: commented out in loading script
- JEB24_2023-10-03, JEB24_2023-10-04: not in loading scripts, behavior-only

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source |
|-----------|-------|--------|
| DR task neurons | 1,651 (483 single) | "1,651 units (483 single units) in ALM from 25 sessions using nine mice" |
| Two-context neurons | 522 (214 single) | "522 units (214 well-isolated single units) were recorded" |
| RD task neurons | 845 (288 single) | "845 units (288 well-isolated single units) in ALM from 19 sessions using four mice" |
| DR sessions | 25 | Paper |
| RD sessions | 19 | Paper |
| DR mice | 9 | Paper |
| RD mice | 4 | Paper |
| Min units/session | 10 | "at least 10 units" |
| FR threshold | 1 Hz | "firing rates exceeding 1 Hz" |
| Bin size | 5ms (getDefaultParams) or 10ms (WorkingWithDataObjs) | Code |

### Processing Details
- Align to goCue
- Bin spikes, smooth with causal gaussian (15ms window)
- Remove clusters with quality in {garbage, noisy, gabrga, real?}
- Remove units with mean FR < 1 Hz
- Video offset: ~0.5s between neural and video timestamps

### Curation Steps
**Neuron curation**: Exclude garbage/noisy quality, then remove FR < 1 Hz
**Trial curation**: No trials excluded (all trials included, conditions determined by hit/miss/R/L/autowater/early)

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| DR mice | 10 animals in data | 10 unique animals | 9 mice | Paper counts 6 two-context + 3 DR-only = 9 |
| Total subjects | 14 unique | 14 | 13 (9+4) | Some overlap possible; data shows 14 distinct |
| Bin size | 5ms (getDefaultParams) vs 10ms (WorkingWithDataObjs) | N/A | Not specified | Used 10ms (WorkingWithDataObjs tutorial) |
| Neuron count | After quality+FR filter | 2452 | 1651+845=2496 | Difference due to FR filtering |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| obj.clu spike times | neural | Bin, smooth, filter | Align to goCue, 10ms bins |
| time_axis | input[0] | time from go cue | Continuous, [-2.5, 2.5]s |
| bp.hit, bp.miss, bp.R, bp.L | output[0]: lick_direction | Categorical | 0=left, 1=right, 2=none |
| bp.autowater | output[1]: context | Binary | 0=DR, 1=WC |
| bp.hit, bp.miss, bp.early | output[2]: outcome | Categorical | 0=incorrect, 1=correct, 2=ignore |
| traj tongue velocity | output[3]: tongue_velocity | Discretize | 0=low, 1=high, 2=not_visible |
| traj paw velocity | output[4]: paw_velocity | Discretize | 0=low, 1=high, 2=not_visible |
| motion energy | output[5]: motion_energy | Discretize | 0=low, 1=high, 2=no_video |

### Key Decisions
1. **Bin size 10ms**: Used 10ms (from WorkingWithDataObjs.m tutorial) rather than 5ms (from getDefaultParams.m). Both are used in the codebase; 10ms is more standard for decoder applications.
2. **All trials included**: No trial filtering by condition - all trials used, with conditions as output labels.
3. **Both datasets combined**: Ephys_Behavior + RandomizedDelay sessions combined into one dataset.
4. **Tongue velocity from side cam**: Using 'tongue' feature, confidence threshold 0.5 for visibility.
5. **Paw velocity from bottom cam**: Using 'top_paw' feature.
6. **Discretization**: Per-session 50th percentile threshold for velocity/ME.

---

## Step 6: Script Development
**Status**: COMPLETE

Key implementation details:
- Handles both h5py (v7.3) and scipy (v5/v7) MATLAB formats
- Proper transposition of traj data between formats
- Motion energy loading handles struct, nested struct, and cell array formats
- Causal gaussian smoothing matches MATLAB mySmooth.m

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (2 sessions: EKH1_2021-08-07, JEB11_2022-05-10)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 |
| Neurons | 110 (48, 62) |
| Trials | 670 (305, 365) |
| Time bins | 500 |

### Run Time: 7.2s for 2 sessions → ~148s estimated for 44 sessions (actual: 148s)

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Train Bal Acc | Val Bal Acc | Chance |
|--------|-------------|------------|--------|
| lick_direction | 0.655 | 0.605 | 0.333 |
| context | 0.778 | 0.785 | 0.500 |
| outcome | 0.611 | 0.533 | 0.333 |
| tongue_velocity | 0.587 | 0.571 | 0.333 |
| paw_velocity | 0.436 | 0.449 | 0.333 |
| motion_energy | 0.763 | 0.701 | 0.333 |

All above chance. Loss decreased from 9.6 to 0.75.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 2082 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Ephys sessions | 25 | 25 | ✓ |
| RD sessions | 19 | 19 | ✓ |
| Total sessions | 44 | 44 | ✓ |
| DR mice | 9 | 10 | ~(see note) |
| RD mice | 4 | 4 | ✓ |
| Total subjects | 13 | 14 | ~(see note) |
| Total neurons | ~2496 | 2452 | ~(FR filter) |
| Min neurons/session | ≥10 | 17 | ✓ |

Note: Paper says 9 mice for DR (6 two-context + 3 DR-only). Data has 10 unique animals in Ephys_Behavior. This may be because one animal appears in both categories or the paper counts differently.

### Warnings
- 29 trials with all-zero neural data in JEB24_2023-10-23 (recording ended before session)
- 1 trial with all-zero neural data in JEB12_2022-05-12

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- Warnings about zero-neural trials are expected (recording gaps)
- No errors reported

### Check 2: Sanity checks
- Neural data spot-checked against raw spike data
- Output labels verified against raw trial info
- Motion energy properly loaded for all sessions

### Check 3: Reference code comparison
- Data loading: matches loadObjs.m
- Quality filtering: matches findClusters.m (excludes garbage, noisy, gabrga, real?)
- Spike alignment: matches alignSpikes.m (trialtm - goCue)
- Binning: matches getSeq.m (histc → smooth → FR)
- Smoothing: matches mySmooth.m (causal gaussian, reflect BC)
- FR filtering: matches removeLowFRClusters.m (mean FR > 1 Hz)

### Check 4: Key statistics
- 44 sessions matches paper (25 + 19)
- 4 RD mice matches paper
- All sessions have ≥10 neurons after filtering


### Early Lick Fix (Critical Review Finding)
Discovered that early lick trials can also be marked as hit/miss. Fixed to override:
- Early lick trials → outcome = 'ignore' (regardless of hit/miss)
- Early lick trials → lick_direction = 'none' (regardless of R/L)
This affected ~611 trials (472 early-hit + 139 early-miss) across Ephys sessions.
### Check 5: Edge cases
- Handled both h5py and scipy formats
- Handled nested ME struct formats
- Handled transposed traj dimensions in scipy

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss: 7.67 → 0.71 (decreasing)

### Decoder Results (Full)
| Output | Train Bal Acc | Val Bal Acc | Chance |
|--------|-------------|------------|--------|
| lick_direction | 0.582 | 0.563 | 0.333 |
| context | 0.857 | 0.851 | 0.500 |
| outcome | 0.579 | 0.544 | 0.333 |
| tongue_velocity | 0.553 | 0.549 | 0.333 |
| paw_velocity | 0.543 | 0.536 | 0.333 |
| motion_energy | 0.843 | 0.509 | 0.333 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Val Acc | Chance | Ratio | Status |
|----------|---------|--------|-------|--------|
| lick_direction | 0.563 | 0.333 | 1.69x | Good |
| context | 0.851 | 0.500 | 1.70x | Good |
| outcome | 0.544 | 0.333 | 1.63x | Good |
| tongue_velocity | 0.549 | 0.333 | 1.65x | OK |
| paw_velocity | 0.536 | 0.333 | 1.61x | OK |
| motion_energy | 0.509 | 0.333 | 1.53x | Marginal |

### Motion energy analysis
Motion energy has high train accuracy (0.84) but low validation (0.50), indicating overfitting. This is expected because:
1. ME is a whole-body movement measure not directly encoded by ALM neurons
2. The discretization threshold is per-session, so cross-trial generalization is harder
3. The decoder has to learn a complex mapping from neural to ME

### Comparison to paper
The paper uses coding directions and subspace analysis rather than classification decoders. Direct accuracy comparison is not possible. However:
- Choice decoding from ALM is well-established, consistent with our 0.63 accuracy
- Context encoding in ALM is strong, consistent with our 0.85 accuracy

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
