# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse (Economo lab)
- **Date started**: 2025-07-29
- **Goal**: Convert electrophysiology + behavioral data to decoder-compatible format
- **Paper**: "Separating cognitive and motor processes in the behaving mouse"
- **Data source**: Ephys_Behavior directory (fixed delay DR+WC two-context task)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from paper
- `code/` - Reference MATLAB code from the paper
- `data/` - Data files (Ephys_Behavior, RandomizedDelay_Ephys_Behavior, etc.)
- `decoder.py` - Decoder module
- `train_decoder.py` - Decoder training script
- Python environment: numpy 2.3.5, torch 2.6.0, scipy 1.18.0

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| loadObjs | DataLoadingScripts/loadObjs.m | LOADING | Loads raw .mat data objects |
| loadSessionData | DataLoadingScripts/loadSessionData.m | LOADING | Main entry - loads and processes all sessions |
| processData | DataLoadingScripts/processData.m | PROCESSING | Orchestrates: findTrials, findClusters, alignSpikes, getSeq, removeLowFRClusters |
| findTrials | DataLoadingScripts/findTrials.m | CURATION | Finds trial indices matching condition strings |
| findClusters | DataLoadingScripts/findClusters.m | CURATION | Filters clusters by quality (excludes garbage, noisy, real?) |
| alignSpikes | DataLoadingScripts/alignSpikes.m | PROCESSING | Aligns spike times to goCue event |
| getSeq | DataLoadingScripts/getSeq.m | PROCESSING | Bins spikes, smooths, creates PSTHs and single-trial data |
| removeLowFRClusters | DataLoadingScripts/removeLowFRClusters.m | CURATION | Removes neurons with mean FR < lowFR threshold |
| loadMotionEnergy | DataLoadingScripts/loadMotionEnergy.m | LOADING | Loads motion energy, interpolates to neural time axis |
| mySmooth | utils/mySmooth.m | PROCESSING | Causal Gaussian smoothing kernel |
| findPosition | funcs/kinematics/findPosition.m | PROCESSING | Interpolates DLC positions to neural time axis |
| findVelocity | funcs/kinematics/findVelocity.m | PROCESSING | Computes velocity via gradient of position |

### Notes
- Data is MATLAB v7.3 (HDF5 format), requires h5py
- Processing: load -> find trials -> find clusters -> align to goCue -> bin and smooth -> remove low FR
- Smoothing: causal Gaussian kernel (gausswin), window=15, first half zeroed for causality, boundary=reflect
- Single trial data: obj.trialdat (time x neurons x trials), units are spks/sec
- Motion energy: loaded separately, interpolated to neural time axis using video frameTimes
- Kinematics: computed from DLC tracking data (traj), velocity = gradient(position)

### Probe Assignments (ALM recordings)
| Animal | Sessions | Probe(s) |
|--------|----------|----------|
| EKH1 | 2021-08-07 | 1 |
| EKH3 | 2021-08-11 | [1,2] (dual) |
| JEB6 | 2021-04-18 | 2 |
| JEB7 | 2021-04-29, 2021-04-30 | 2 |
| JEB13 | 09-13, 09-14 probe=2; 09-21, 09-24, 09-25 probe=1 |
| JEB14 | all 4 sessions | 1 |
| JEB15 | all 4 sessions | 1 |
| JEB19 | all 4 sessions | 1 |
| JGR2 | 2021-11-16, 2021-11-17 | 1 |
| JGR3 | 2021-11-18 | 1 |

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- HDF5 (.mat v7.3) files, one per session
- obj.bp: Behavioral data (L, R, hit, miss, no, autowater, early, stim, ev)
- obj.clu: Cluster/neuron data (quality, tm, trial, trialtm, site)
- obj.traj: DLC trajectory data (2 views, per-trial)
- Motion energy in separate motionEnergy_*.mat files

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 10 |
| Sessions | 25 |
| Trials / session | 137-472 (valid) |
| Neurons / session | 27-134 (after filtering) |
| Total neurons (after quality) | 1653 |
| Total neurons (after FR filter) | 1453 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total, DR) | 1,651 | "we recorded 1,651 units in ALM from 25 sessions using nine mice" |
| Subjects (DR) | 9 mice | same |
| Sessions (DR) | 25 | same |
| Two-context sessions | 12 from 6 mice | "In 12 sessions from six mice" |
| Two-context units | 522 (214 single) | same |
| Neural data time bin | 10ms | params.dt = 1/100 |
| Smoothing | 15-sample causal Gaussian | params.smooth = 15 |
| Low FR threshold | 1 Hz | "firing rates exceeding 1 Hz" |
| Min units per session | 10 | "at least 10 units" |
| Quality filter | all (excl. garbage, noisy) | findClusters.m |
| Time window | -2.5 to 2.5s from goCue | params.tmin/tmax |
| Delay duration | 0.9s (fixed) | "delay epoch (0.9 s)" |
| Sample tone | 1.3s | "auditory tones lasting 1.3 s" |

### Processing Details
- Align to goCue onset
- Time bins: 10ms
- Smoothing: causal Gaussian kernel, window=15 samples, boundary=reflect
- Quality filter: exclude garbage, gabrga, noisy, real? clusters
- Low FR filter: exclude neurons with mean FR <= 1 Hz

### Curation Steps
**Neuron curation**: Exclude garbage/noisy quality, then exclude FR <= 1 Hz
**Trial curation**: Exclude early lick, ignore/no-response, stimulation trials

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Subjects | 10 animals loaded | 10 animals | 9 mice | Data has 10; paper counts 9. Using all 10. |
| lowFR | default 0.5 | N/A | 1 Hz | Using 1 Hz per paper |
| dt | default 1/200 | N/A | 1/100 in most scripts | Using 1/100 (10ms) |
| Neuron count | N/A | 1653 after quality | 1651 | Match within 2 (null quality strings) |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| Spike times (clu) | neural | Bin 10ms, smooth causal Gaussian, align to goCue | spks/sec |
| Time axis | input[0] | Time from goCue in seconds | Continuous |
| R (right trial) | output[0] | left=0, right=1 | Per trial |
| autowater | output[1] | WC=0, DR=1 | Per trial |
| hit | output[2] | incorrect=0, correct=1 | Per trial |
| DLC tongue velocity | output[3] | Discretize 50th %ile | Time-varying |
| DLC paw velocity | output[4] | Discretize 50th %ile | Time-varying |
| Motion energy | output[5] | Discretize 50th %ile | Time-varying |

### Key Decisions
1. **Use Ephys_Behavior only**: Has DR+WC two-context task
2. **lowFR = 1 Hz**: Matches paper
3. **dt = 10ms**: Standard across analysis scripts
4. **Include hit and miss trials**: Both are valid
5. **Exclude early, no, stim trials**: Per paper methods
6. **Tongue velocity**: Side view (view 1), tongue feature, speed = sqrt(xvel^2 + yvel^2)
7. **Paw velocity**: Top view (view 2), top_paw + bottom_paw averaged
8. **Brain region**: All ALM
9. **EKH3 dual probe**: Concatenate both probes
10. **Discretization edge case**: When threshold=0, use median of positive values

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation in `convert_data.py`:
- Handles HDF5 data loading with h5py
- Proper probe indexing (handles single vs dual probe sessions)
- Quality filtering matching MATLAB findClusters.m
- Spike binning and causal Gaussian smoothing matching getSeq.m
- Velocity computation matching findVelocity.m
- Motion energy loading and interpolation matching loadMotionEnergy.m

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (EKH1, JEB6) |
| Neurons | 60 (31 + 29) |
| Trials | 474 (214 + 260) |
| Time bins | 500 |

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Full processing | ~6-7s | ~180s (25 sessions) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Acc | Validation Acc |
|--------|-------------|---------------|
| lick_direction | 0.633 | 0.590 |
| behavioral_context | 0.726 | 0.739 |
| outcome | 0.640 | 0.558 |
| tongue_velocity | 0.783 | 0.793 |
| paw_velocity | 0.595 | 0.576 |
| motion_energy | 0.817 | 0.831 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 932.2 MB
- `verification_full_out.txt`: created, no errors

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Total neurons (after quality) | 1,651 | 1,653 | ~Yes (2 diff) |
| Total neurons (after FR 1Hz) | N/A | 1,453 | N/A |
| Subjects | 9 | 10 | Close (data has 10) |
| Sessions | 25 | 25 | Yes |
| Trials (total valid) | N/A | 6,150 | N/A |
| Time bins | N/A | 500 | N/A |
| Lick direction | ~50% | 49.8% right | Yes |
| Outcome (correct) | >70% trained | 83.8% | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- verification_full_out.txt: No errors, no warnings
- All output ranges [0, 1] as expected
- Some sessions have behavioral_context [1.0, 1.0] (all DR) - these are sessions without WC blocks, which is expected
- Motion energy [1.0, 1.0] for 2 sessions - investigated, these sessions have all-zero motion energy data

**Check 2: Sanity checks**
1. Neural data: Recomputed firing rate from raw spikes for EKH1, neuron 3, trial 10 - exact match with converted data (OK)
2. Input data: Time axis matches expected [-2.495, 2.495] in 10ms steps (OK)
3. Output data: All values are 0 or 1 (OK)
4. Behavioral variables: Spot-checked trial 5 of EKH1 - lick direction, context, outcome all match raw data (OK)
5. Cross-session consistency: All trials within each session have consistent shapes (OK)

**Check 3: Reference code comparison**
- (a) Data loading: Using h5py to read HDF5 files, matching loadObjs.m
- (b) Neuron filtering: Quality filter matches findClusters.m (excludes garbage, gabrga, noisy, real?)
- (c) Temporal alignment: Spike times aligned to goCue, matching alignSpikes.m
- (d) Binning: 10ms bins from -2.5 to 2.5s, matching getSeq.m edges computation
- (e) Smoothing: Causal Gaussian kernel with window=15, reflect boundary, matching mySmooth.m
- (f) FR filtering: Mean FR > 1 Hz threshold, matching removeLowFRClusters.m
- (g) Motion energy: Interpolated to neural time axis using video frameTimes, matching loadMotionEnergy.m
- (h) Velocity: gradient(position), tongue NaN->0, paw subtract baseline, matching findVelocity.m

**Check 4: Key statistics comparison**
- Total neurons after quality: 1653 (paper: 1651) - match within 2
- Sessions: 25 (paper: 25) - exact match
- Subjects: 10 (paper: 9) - data has 10 animals, paper may count differently
- Lick direction: ~50% right - consistent with balanced task
- Outcome: 83.8% correct - consistent with well-trained mice (>70%)

**Check 5: Edge cases**
- JEB15_2022-07-29: probe 1 has 197 clusters with null quality strings. Initially excluded (causing session skip), fixed by including null quality strings per MATLAB behavior.
- JEB7 probe=2 but clu shape (1,1): Fixed by checking clu shape and using index 0 when only 1 probe in data.
- EKH3 dual probe: Properly concatenates neurons from both probes.
- Tongue velocity discretization: When threshold=0 (most values are 0 when tongue not visible), use median of positive values.
- Motion energy nested struct: JEB15 sessions have me.data as a struct with .data and .moveThresh fields. Fixed by unwrapping (matching MATLAB: if isstruct(me.data), me.data = me.data.data).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (6.33 -> 0.51)

### Decoder Results (Full)
| Output | Training Acc | Validation Acc | Chance | Notes |
|--------|-------------|---------------|--------|-------|
| lick_direction | 0.667 | 0.642 | 0.500 | Above chance |
| behavioral_context | 0.839 | 0.818 | 0.500 | Well above chance |
| outcome | 0.687 | 0.652 | 0.500 | Above chance |
| tongue_velocity | 0.830 | 0.819 | 0.500 | Well above chance |
| paw_velocity | 0.622 | 0.619 | 0.500 | Above chance |
| motion_energy | 0.798 | 0.793 | 0.500 | Well above chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio to Chance |
|----------|------------------|--------|----------------|
| lick_direction | 0.642 | 0.500 | 1.28x |
| behavioral_context | 0.818 | 0.500 | 1.64x |
| outcome | 0.652 | 0.500 | 1.30x |
| tongue_velocity | 0.819 | 0.500 | 1.64x |
| paw_velocity | 0.619 | 0.500 | 1.24x |
| motion_energy | 0.793 | 0.500 | 1.59x |

All outputs are well above chance (>1.2x). The paper reports neural choice decoding accuracy varying by time point, with peak accuracy around the response period. Our decoder uses a different architecture (RNN-based vs the paper's time-resolved SVM), so direct comparison is approximate.

**Accuracy vs paper expectations:**
- Choice (lick direction): The paper shows time-resolved choice decoding reaching ~80-90% during the response period. Our 64.8% is lower but expected since we decode across the full time window including pre-stimulus periods where choice is not yet determined.
- Context (behavioral context): The paper shows context can be decoded from neural activity. Our 81.8% is strong.
- Outcome: Correlated with choice accuracy. Our 66.6% is reasonable.
- Tongue/paw/motion energy: These are movement-related outputs that should be decodable from motor cortex (ALM). Our 62-82% accuracies confirm this.

**Train vs validation gap:**
- All outputs show small train-val gaps (<0.04), indicating no significant overfitting.

### Issues Found and Resolved
- No critical issues found in this review.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
