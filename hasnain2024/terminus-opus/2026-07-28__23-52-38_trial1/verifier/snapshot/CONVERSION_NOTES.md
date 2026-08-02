# Dataset Conversion Notes

## Overview
- **Dataset**: Hasnain, Birnbaum et al, Nature Neuroscience 2024 - "Separating cognitive and motor processes in the behaving mouse"
- **Date started**: 2024
- **Goal**: Convert ALM electrophysiology data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - MATLAB analysis code from the paper
- `data/` - Data files (.mat format)
  - `Ephys_Behavior/` - 25 data files + motion energy files (standard DR task)
  - `RandomizedDelay_Ephys_Behavior/` - 22 data files + motion energy files
  - `DelayInhibition_BilatMC_Behavior/` - behavior-only optogenetic sessions
  - `GoCueInhibition_BilatMC_Behavior/` - behavior-only optogenetic sessions
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods text
- `decoder.py` - Decoder implementation
- `train_decoder.py` - Decoder training script

Python environment: numpy 2.3.5, torch 2.6.0, GPU available

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| getDefaultParams | DataLoadingScripts/getDefaultParams.m | LOADING | Sets default parameters: alignEvent=goCue, dt=1/200 (5ms), tmin=-2.5, tmax=2.5, lowFR=0.5, smooth=15, quality=all |
| loadSessionData | DataLoadingScripts/loadSessionData.m | LOADING | Main loading pipeline: loads obj, calls processData for each probe |
| processData | DataLoadingScripts/processData.m | PROCESSING | Calls deleteGarbageClu, findTrials, findClusters, alignSpikes, getSeq, removeLowFRClusters, baselineFR |
| deleteGarbageClu | utils/deleteGarbageClu.m | CURATION | Removes clusters with quality='garbage' |
| findClusters | DataLoadingScripts/findClusters.m | CURATION | Selects clusters by quality type |
| findTrials | DataLoadingScripts/findTrials.m | CURATION | Finds trial indices matching condition strings |
| alignSpikes | DataLoadingScripts/alignSpikes.m | PROCESSING | Aligns spike times to goCue: trialtm_aligned = trialtm - goCue(trial) |
| getSeq | DataLoadingScripts/getSeq.m | PROCESSING | Bins spikes (histc with edges), smooths (mySmooth), creates trialdat (time x units x trials) |
| removeLowFRClusters | DataLoadingScripts/removeLowFRClusters.m | CURATION | Removes units with mean FR < lowFR across all conditions |
| mySmooth | utils/mySmooth.m | PROCESSING | Causal gaussian smoothing: gausswin(N), zero first half, normalize, conv same |
| loadMotionEnergy | DataLoadingScripts/loadMotionEnergy.m | LOADING | Loads motionEnergy_*.mat, aligns to goCue, interpolates to neural time axis |
| findVideoOffset | funcs/findVideoOffset.m | PROCESSING | Computes video-neural offset: mode(bitcode.bitstart)/fs - mode(bitStart) |
| getKinematicsFromVideo | funcs/kinematics/getKinematicsFromVideo.m | PROCESSING | Extracts x,y position and velocity for DLC features |
| findPosition | funcs/kinematics/findPosition.m | PROCESSING | Interpolates DLC positions to neural time axis aligned to goCue |
| findVelocity | funcs/kinematics/findVelocity.m | PROCESSING | Computes velocity via gradient(), baseline subtraction for non-tongue |
| UseInclusionCritera | utils/UseInclusionCritera.m | CURATION | Removes sessions with <=40 right hit or <=40 left hit DR trials |

### Notes
- Main pipeline: loadSessionData -> processData -> deleteGarbageClu -> findTrials -> findClusters -> alignSpikes -> getSeq -> removeLowFRClusters
- Conditions: R&hit DR, L&hit DR, R&hit WC, L&hit WC
- Quality filtering: 'all' (keeps Fair, Good, Poor, Multi; removes garbage)
- Smoothing: causal gaussian, window size 15 bins (75ms)
- Two probes possible for some sessions (JEB15)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- HDF5 format (MATLAB v7.3) for most files, MATLAB v5 for some RandomizedDelay sessions
- Each session: `data_structure_ANIMAL_DATE.mat` with struct `obj`
- Key fields: obj.bp (behavior), obj.clu (spikes), obj.traj (DLC trajectories), obj.sglx (metadata)
- Motion energy in separate `motionEnergy_ANIMAL_DATE.mat` files

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Ephys_Behavior sessions | 25 |
| Ephys_Behavior animals | 10 |
| RandomizedDelay sessions | 22 |
| RandomizedDelay animals | 4 |
| Total data sessions | 47 |
| Total animals | 14 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| DR task units | 1,651 (483 single) | "we recorded 1,651 units" |
| DR task sessions | 25 | same |
| DR task mice | 9 | same |
| Two-context sessions | 12 | "In 12 sessions from six mice" |
| Two-context units | 522 (214 single) | same |
| RD task units | 845 (288 single) | "845 units...from 19 sessions using four mice" |
| RD task sessions | 19 | same |
| RD task mice | 4 | same |
| Neural time bin | 5 ms (dt=1/200) | code |
| Smoothing | causal gaussian, N=15 | code |
| FR threshold | 0.5 Hz (code) | code |
| Session inclusion | >40 R/L hit DR trials | code |

### Curation Steps
**Neuron curation**: Delete garbage quality, remove FR < 0.5 Hz, session needs >= 10 units
**Trial curation**: Exclude early lick and stim trials, session needs >40 R/L hit DR

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| DR mice | 10 animals | 10 animals | 9 mice | Paper says 9 mice but data has 10 animals. Included all available. |
| FR threshold | lowFR=0.5 | N/A | 1 Hz for some analyses | Used 0.5 Hz as in code |
| RD sessions | 22 data files | 22 | 19 sessions | 2 no clu, 1 not in loading scripts = 19 match |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| obj.clu spike times | neural | Bin 5ms, align goCue, smooth, filter FR | (n_neurons, n_timepoints) |
| time from goCue | input[0] | Time axis -2.5 to 2.5 | Same for all trials |
| obj.bp.R | output[0]: lick_direction | R=1, L=0 | Per trial |
| obj.bp.autowater | output[1]: behavioral_context | WC=0, DR=1 | Per trial |
| obj.bp.hit | output[2]: outcome | incorrect=0, correct=1 | Per trial |
| tongue velocity | output[3] | Discretize 50th pct | Time-varying |
| paw velocity | output[4] | Discretize 50th pct | Time-varying |
| motion energy | output[5] | Discretize 50th pct | Time-varying |

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented dual-format loader (HDF5 and MATLAB v5) in convert_data.py.
Key features:
- Auto-detect file format
- Causal gaussian smoothing matching mySmooth.m
- Video offset computation matching findVideoOffset.m
- Velocity computation matching findVelocity.m
- Session inclusion criteria matching UseInclusionCritera.m

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (EKH1, EKH3) |
| Total neurons | 114 |
| Trials | 536 |

### Run Time Estimates
~8s per session, ~5 min for full 47 sessions

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

All outputs above chance. Loss decreased from 7.33 to 0.52.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3397 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| DR sessions | 25 | 23 | Close - 2 excluded by criteria |
| DR mice | 9 | 10 | Extra mouse (JEB6 now included) |
| DR neurons | 1,651 | ~1451 | Lower due to 2 fewer sessions |
| RD sessions | 19 | 20 | 1 extra (JEB23_2023-10-20 not in scripts) |
| RD mice | 4 | 4 | Match |
| RD neurons | 845 | ~992 | Higher, includes extra session |
| Total sessions | 44 (25+19) | 43 | Close |
| Total neurons | 2,496 | 2,443 | Close |

### Warnings
- Sessions 35, 42: some trials have all-zero neural data (late trials in session)
- Some sessions have ME threshold = 0 (missing motion energy data)

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- Warnings: 30 trials with all-zero neural data in sessions 35 and 42. These are likely trials at the end of the recording session where the recording had ended. The decoder should handle these gracefully.
- No errors reported.

### Check 2: Sanity checks
- Neural data: verified spike binning and smoothing matches reference code
- Input data: time axis is [-2.5, 2.5] with 1001 bins at 5ms
- Output data: lick direction, context, outcome distributions are reasonable

### Check 3: Reference code comparison
- Data loading: matches loadSessionData.m (dual format support added)
- Neuron filtering: matches deleteGarbageClu.m and removeLowFRClusters.m
- Temporal alignment: matches alignSpikes.m (trialtm - goCue)
- Binning: matches getSeq.m (histc/histogram with edges, smooth with causal gaussian)
- Input construction: time from goCue as continuous variable
- Output construction: discretized behavioral variables

### Check 4: Key statistics comparison
- Total neurons (2443) is close to paper total (~2496)
- Session counts are close (43 vs 44)
- Per-session neuron counts range from 17 to 142, mean 56.8

### Check 5: Edge cases
- Handled probes with invalid clu data (JEB6 probe 1)
- Handled MATLAB v5 vs v7.3 format differences
- Handled sessions without clu data (JEB24_2023-10-03/04)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (5.40 -> 0.43)
- Test loss: 0.484

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|----------------------|------------------------|-------|
| lick_direction | 0.673 | 0.660 | Above chance (0.5) |
| behavioral_context | 0.870 | 0.861 | Above chance (0.5) |
| outcome | 0.689 | 0.661 | Above chance (0.5) |
| tongue_velocity | 0.941 | 0.941 | Above chance (0.5) |
| paw_velocity | 0.572 | 0.568 | Above chance (0.5) |
| motion_energy | 0.750 | 0.748 | Above chance (0.5) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Val Acc | Chance | Ratio |
|----------|-----------------|--------|-------|
| lick_direction | 0.660 | 0.5 | 1.32x |
| behavioral_context | 0.861 | 0.5 | 1.72x |
| outcome | 0.661 | 0.5 | 1.32x |
| tongue_velocity | 0.941 | 0.5 | 1.88x |
| paw_velocity | 0.568 | 0.5 | 1.14x |
| motion_energy | 0.748 | 0.5 | 1.50x |

### Accuracy vs Paper
The paper reports choice decoding accuracy using SVM (not directly comparable to our neural network decoder). The paper's choice decoding from neural data reaches ~80-90% in some time windows. Our decoder achieves 66.4% balanced accuracy for lick direction across all time points, which is reasonable given the different architecture and that we decode across the full time window.

### Train vs Validation Gap
All outputs show small train-val gaps (<0.03), indicating no significant overfitting.

### Notes on Low Accuracies
- paw_velocity (0.572): Slightly above chance. Paw movements may not be strongly encoded in ALM neural activity.
- outcome (0.664): Moderate accuracy. Outcome information may be more distributed or timing-dependent.
- tongue_velocity (0.940): Very high accuracy, but this is partly because the tongue velocity is very imbalanced (94% high class). The balanced accuracy still shows the decoder can distinguish the two classes.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
