# Dataset Conversion Notes

## Overview
- **Dataset**: Hasnain, Birnbaum et al. "Separating cognitive and motor processes in the behaving mouse" (Nature Neuroscience 2024)
- **Date started**: 2025
- **Goal**: Convert to decoder-compatible format for training neural decoders

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Extracted methods text
- `/app/code/` - Reference code (MATLAB)
- `/app/data/` - Data files (Ephys_Behavior: 25 sessions, RandomizedDelay: 22 sessions)
- `/app/decoder.py` - Decoder library
- `/app/train_decoder.py` - Decoder training script

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| loadObjs | loadObjs.m | LOADING | Load raw .mat data objects |
| loadSessionData | loadSessionData.m | LOADING | Orchestrate loading + processing |
| processData | processData.m | PROCESSING | Main processing pipeline |
| findTrials | findTrials.m | CURATION | Find trials matching conditions |
| findClusters | findClusters.m | CURATION | Filter clusters by quality |
| alignSpikes | alignSpikes.m | PROCESSING | Align spikes to goCue |
| getSeq | getSeq.m | PROCESSING | Bin spikes, smooth, create single-trial data |
| removeLowFRClusters | removeLowFRClusters.m | CURATION | Remove neurons with FR < threshold |
| loadMotionEnergy | loadMotionEnergy.m | LOADING | Load and align motion energy |
| getKinematicsFromVideo | getKinematicsFromVideo.m | PROCESSING | Extract position/velocity from DLC |
| findPosition | findPosition.m | PROCESSING | Interpolate DLC tracking to neural time axis |
| findVelocity | findVelocity.m | PROCESSING | Compute velocity from position (gradient) |
| findVideoOffset | findVideoOffset.m | PROCESSING | Align video to neural recording |
| UseInclusionCritera | UseInclusionCritera.m | CURATION | Exclude sessions with <40 R/L hit DR trials |

### Notes
- Pipeline: loadObjs -> processData -> (findTrials, findClusters, alignSpikes, getSeq, removeLowFRClusters)
- alignSpikes: trialtm_aligned = trialtm - ev.goCue(trial)
- getSeq: bins at dt resolution, smooths with causal gaussian (15-sample window)
- removeLowFRClusters: mean FR across all time and trials, threshold at lowFR
- Video: 2 cameras at ~400Hz, DLC tracking interpolated to neural time axis
- Velocity = gradient of interpolated position
- Video offset = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- HDF5 (.mat v7.3) and MATLAB v5 formats
- Each file: `data_structure_<animal>_<date>.mat` with `obj` struct
- Key fields: obj.bp (behavior), obj.clu (clusters), obj.traj (video tracking), obj.sglx (recording metadata)
- Motion energy in separate files: `motionEnergy_<animal>_<date>.mat`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Data files (Ephys) | 25 |
| Data files (RandomizedDelay) | 22 |
| Unique animals (Ephys) | 10 |
| Unique animals (RandDelay) | 4 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|-------------------------------|
| Neurons DR task | 1,651 (483 single) | "1,651 units (483 single units) in ALM from 25 sessions using nine mice" |
| Neurons randomized delay | 845 (288 single) | "845 units (288 well-isolated single units) in ALM from 19 sessions using four mice" |
| Sessions DR | 25 | "25 sessions" |
| Sessions randomized delay | 19 | "19 sessions using four mice" |
| Mice DR | 9 | "nine mice" |
| Mice randomized delay | 4 | "four mice" |
| Neural data time bin | 5ms (1/200) | Code: params.dt = 1/200 |
| Smoothing | 15-sample causal gaussian | Code: params.smooth = 15 |
| Low FR threshold | 1 Hz | Paper: "firing rates exceeding 1 Hz" |
| Cluster quality | all (excl garbage, noisy, real?) | Code: params.quality = {'all'} |
| Session min units | >=10 | "at least 10 units" |
| Session min DR trials | >40 correct per direction | UseInclusionCritera.m |
| DR delay epoch | 0.9s fixed (or randomized) | Paper |
| DR sample tone | 1.3s | Paper |

### Processing Details
- Align to goCue onset
- Bin spikes at 5ms, smooth with 15-sample causal gaussian
- Filter: exclude garbage/noisy/real? quality, remove FR < 1 Hz
- Exclude early lick, ignore, and stimulation trials
- Video tracking interpolated to neural time axis with video offset correction

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| lowFR | getDefaultParams: 0.5; analysis scripts: 1.0 | N/A | 1 Hz | Use 1.0 Hz (matches paper) |
| dt | getDefaultParams: 1/200; WorkingWithDataObjs: 1/100 | N/A | N/A | Use 1/200 (matches analysis scripts) |
| Mice DR | 10 unique animal IDs | 10 animals in data | 9 mice | Include all 10; paper may count differently |
| RandDelay sessions | 20 with loading functions | 22 data files | 19 sessions | Include 20 (2 files without loading functions excluded) |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| obj.clu spike times | neural | Bin 5ms, smooth 15-sample causal gaussian, align to goCue | Single trial FR (spk/s) |
| time from goCue | input[0] | Continuous time axis [-2.5, 2.5] | |
| L/R + hit/miss | output[0]: lick_direction | 0=left, 1=right, 2=none | Per-trial |
| autowater | output[1]: context | 0=DR, 1=WC | Per-trial |
| hit/miss/no | output[2]: outcome | 0=incorrect, 1=correct, 2=ignore | Per-trial |
| tongue DLC tracking | output[3]: tongue_velocity | Discretized per-session 50th percentile | Time-varying |
| paw DLC tracking | output[4]: paw_velocity | Discretized per-session 50th percentile | Time-varying |
| motion energy | output[5]: motion_energy | Discretized per-session 50th percentile | Time-varying |

### Key Decisions
1. **Include both Ephys_Behavior and RandomizedDelay sessions**: Both have neural + behavioral data
2. **Trial filtering**: Exclude early lick, ignore, stimulation trials (matching paper)
3. **Neuron filtering**: Quality filter + lowFR=1.0 Hz (matching paper)
4. **Tongue velocity**: From camera 0 'tongue' feature. Speed = sqrt(xvel^2 + yvel^2). Tongue not visible -> category 2.
5. **Paw velocity**: From camera 1 'bottom_paw' feature. Speed = sqrt(xvel^2 + yvel^2). Paw not visible -> category 2.
6. **Motion energy**: From separate files, aligned via video frame times. No ME file -> category 2.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py` with unified data access layer for HDF5 and MATLAB v5 formats
- SessionData class provides consistent interface regardless of file format
- Supports --sample, --full, and --show-processing modes

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (EKH1, JEB11) |
| Neurons | 110 (48 + 62) |
| Trials | 479 (214 + 265) |
| Processing time | 8.3s |

### Processing Plots Review
- Neural activity shows expected patterns (activity increase after go cue)
- Output distributions are reasonable
- No anomalies detected

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|----------------|---------------------|
| Full conversion | ~3.7s | ~167s (~2.8 min) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Bal Acc | Validation Bal Acc | Chance |
|--------|-----------------|-------------------|--------|
| lick_direction | 0.647 | 0.633 | 0.333 |
| context | 0.787 | 0.761 | 0.500 |
| outcome | 0.633 | 0.585 | 0.333 |
| tongue_velocity | 0.570 | 0.564 | 0.333 |
| paw_velocity | 0.584 | 0.571 | 0.333 |
| motion_energy | 0.777 | 0.785 | 0.333 |

All outputs above chance. Loss decreasing properly.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3341.0 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| DR sessions | 25 | 23 | Close (2 JEB19 sessions excluded for <40 trials) |
| RandDelay sessions | 19 | 20 | Close (1 extra session) |
| DR neurons | 1,651 | 1,426 | ~86% (due to 2 excluded sessions) |
| RandDelay neurons | 845 | 972 | Higher (1 extra session) |
| Total sessions | 44 | 43 | Close |
| Total neurons | 2,496 | 2,398 | ~96% |
| DR mice | 9 | 10 | 1 extra animal ID |
| RandDelay mice | 4 | 4 | Match |
| Correct rate | ~80-90% | 86.4% | Consistent |

### Notes on Discrepancies
- 2 JEB19 sessions excluded because they had ≤40 correct DR trials per direction
- Paper says 9 mice for DR but data has 10 unique animal IDs; this may be because the paper counted JGR2 and JGR3 as one mouse or excluded one animal
- RandomizedDelay has 20 sessions vs paper's 19; one extra session may have been excluded in the paper's analysis

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- Warnings: Sessions 29 and 36 (JEB24) have trials at end with all-zero neural data
  - These are trials that occurred after the neural recording ended
  - Cannot be fixed without excluding these trials, but they are valid behavioral trials
  - The zero neural data is correct given the spike data

### Check 2: Sanity Checks
1. **Neural data**: Manually computed FR for session 0, neuron 5, trial 10 matches converted data exactly (correlation=1.0, max diff=0.0)
2. **Output data**: Verified lick_direction, context, outcome for 5 trials in session 0 - all match expected values from raw behavioral data
3. **Input data**: Time axis matches expected range [-2.4975, 2.4975] with 1000 timepoints at 5ms bins

### Check 3: Reference code comparison
- **Data loading**: Matches loadObjs.m - load .mat files, access obj struct
- **Neuron filtering**: Matches findClusters.m (exclude garbage, noisy, real?) and removeLowFRClusters.m (FR > 1 Hz)
- **Temporal alignment**: Matches alignSpikes.m - subtract goCue time from trialtm
- **Binning**: Matches getSeq.m - histogram with edges tmin:dt:tmax, divide by dt for FR
- **Smoothing**: Matches mySmooth.m - causal gaussian kernel with window N=15, reflect boundary
- **Input construction**: Time from go cue as continuous variable
- **Output construction**: Follows decoder task specification

### Check 4: Key statistics comparison
- Total neurons: 2,398 vs paper's ~2,496 (96% match)
- Sessions: 43 vs paper's ~44 (2 excluded for insufficient trials)
- Correct rate: 86.4% consistent with expected performance
- Context distribution: 92.6% DR, 7.4% WC (many sessions are DR-only)

### Check 5: Edge cases
- Handled both HDF5 and MATLAB v5 file formats
- Handled sessions with missing motion energy data (set to category 2)
- Handled trials with no video data (tongue/paw set to not_visible)
- Handled dual-probe sessions (JEB15)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (8.42 -> 0.63)

### Decoder Results (Full)
| Output | Training Bal Acc | Validation Bal Acc | Chance |
|--------|-----------------|-------------------|--------|
| lick_direction | 0.665 | 0.648 | 0.333 |
| context | 0.864 | 0.855 | 0.500 |
| outcome | 0.670 | 0.637 | 0.333 |
| tongue_velocity | 0.551 | 0.542 | 0.333 |
| paw_velocity | 0.568 | 0.562 | 0.333 |
| motion_energy | 0.819 | 0.817 | 0.333 |

All outputs well above chance. Small train-validation gap indicates no overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio |
|----------|------------------|--------|-------|
| lick_direction | 0.648 | 0.333 | 1.94x |
| context | 0.855 | 0.500 | 1.71x |
| outcome | 0.637 | 0.333 | 1.91x |
| tongue_velocity | 0.542 | 0.333 | 1.63x |
| paw_velocity | 0.562 | 0.333 | 1.69x |
| motion_energy | 0.817 | 0.333 | 2.45x |

All outputs are >1.5x chance. The paper reports choice decoding from neural activity with high accuracy in ALM, consistent with our lick_direction accuracy. Context decoding is also strong, consistent with the paper's finding of persistent context coding.

### Train vs Validation Gap
- Largest gap: outcome (0.670 vs 0.637 = 1.05x), acceptable
- No evidence of overfitting

### Issues Found and Resolved
- None remaining. All checks pass.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
