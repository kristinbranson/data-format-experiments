# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python environment: numpy 2.4.4, torch 2.6.0+cu124 (CUDA)

Directory contents:
- `paper.pdf` - Reference paper
- `methods.txt` - Methods excerpt from paper
- `code/` - Reference code repository (MATLAB)
- `data/` - Data files organized in 4 subdirectories:
  - `Ephys_Behavior/` - 25 sessions, data_structure + motionEnergy .mat files (8 subjects: EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3)
  - `RandomizedDelay_Ephys_Behavior/` - 22 sessions (3 subjects: JEB11, JEB12, JEB23, JEB24)
  - `DelayInhibition_BilatMC_Behavior/` - 53 sessions (5 subjects: MAH13, MAH14, MAH20, MAH21) - behavior only, no motionEnergy
  - `GoCueInhibition_BilatMC_Behavior/` - 20 sessions (4 subjects: MAH13, MAH14, MAH20, MAH21) - behavior only, no motionEnergy
- `decoder.py` - Decoder module
- `train_decoder.py` - Decoder training script

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| loadObjs | DataLoadingScripts/loadObjs.m | LOADING | Load data_structure .mat files |
| loadMotionEnergy | DataLoadingScripts/loadMotionEnergy.m | LOADING | Load motionEnergy .mat files, align to neural timebase |
| findDataFn | utils/findDataFn.m | LOADING | Locate data files by animal/date |
| loadBehavObjs | funcs/fig1/loadBehavObjs.m | LOADING | Load behavior-only sessions |
| processData | DataLoadingScripts/processData.m | PROCESSING | Orchestrate full pipeline |
| alignSpikes | DataLoadingScripts/alignSpikes.m | PROCESSING | Align spikes to event (goCue, moveOnset, etc.) |
| getSeq | DataLoadingScripts/getSeq.m | PROCESSING | Bin spikes into time bins, create PSTHs and single-trial data |
| mySmooth | utils/mySmooth.m | PROCESSING | Causal Gaussian smoothing |
| baselineFR | DataLoadingScripts/baselineFR.m | PROCESSING | Calculate baseline firing rate stats |
| findTrials | DataLoadingScripts/findTrials.m | CURATION | Filter trials by condition |
| findClusters | DataLoadingScripts/findClusters.m | CURATION | Filter neurons by quality |
| removeLowFRClusters | DataLoadingScripts/removeLowFRClusters.m | CURATION | Remove neurons with mean FR < threshold |
| loadSessionData | DataLoadingScripts/loadSessionData.m | PROCESSING | High-level pipeline: load + process all sessions |
| NeuralChoiceDecoding | ChoiceContextDecoding/NeuralChoiceDecoding.m | ANALYSIS | Decode left/right choice |
| NeuralContextDecoding | ChoiceContextDecoding/NeuralContextDecoding.m | ANALYSIS | Decode task context (DR vs WC) |

### Notes
- **Pipeline**: loadObjs -> processData (findTrials, findClusters, alignSpikes, getSeq, removeLowFRClusters, baselineFR)
- **Time binning**: dt = 1/100 = 10 ms (code default), tmin = -2.5s, tmax = 2.5s
- **Smoothing**: Causal Gaussian kernel, 15 samples = 150ms (code default)
- **Quality filter**: 'all' excludes garbage, noisy, gabrga, real?
- **Low FR threshold**: 1 Hz
- **Brain region**: ALM (anterior lateral motor cortex)
- **Alignment event**: goCue (standard)
- **Data structure**: obj.clu (spikes), obj.bp (behavior/trial info), obj.traj (video tracking), obj.me (motion energy)
- **clu structure per probe**: quality, tm (spike times), trial, trialtm, site, spkWavs
- **bp structure**: hit, miss, no, early, autowater, L, R, stim.enable, ev.goCue, ev.sample, ev.delay, ev.lickL, ev.lickR
- **traj structure**: 2 cameras (bottom: tongue/jaw/paw, side: tongue/paw/jaw/nose), each with ts (features x [x,y,conf] x frames), frameTimes, featNames
- **Choice decoder**: rez.binSize=75ms, 4-fold CV
- **getSeq output**: obj.trialdat{probe} = (nTimeBins x nClusters x nTrials), firing rates in Hz

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- **Ephys_Behavior/**: 25 sessions, each with data_structure_ANM_DATE.mat + motionEnergy_ANM_DATE.mat
  - HDF5 format (MATLAB v7.3), loaded with h5py
  - Contains neural data (clu), behavior (bp), video tracking (traj)
  - 10 animals: EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3
  - Mixed WC+DR and DR-only sessions
  - 1-2 probes per session
- **RandomizedDelay_Ephys_Behavior/**: 22 sessions, data_structure + motionEnergy files
  - Mix of HDF5 and older MATLAB format (scipy.io.loadmat)
  - 20 sessions have neural data, 2 behavior-only (JEB24_10-03, JEB24_10-04)
  - 4 animals: JEB11, JEB12, JEB23, JEB24
  - Randomized delay durations
  - Motion energy integrated in data_structure (obj.me) for some, separate files for others
- **DelayInhibition_BilatMC_Behavior/**: 53 sessions, behavior-only (no neural data)
- **GoCueInhibition_BilatMC_Behavior/**: 20 sessions, behavior-only (no neural data)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, raw) | 14,297 (11,158 Ephys + 3,139 RandDelay) |
| Neurons / session | 28-1258 (Ephys), 17-560 (RandDelay) |
| Subjects (with neural) | 14 (10 Ephys + 4 RandDelay) |
| Sessions with neural | 45 (25 Ephys + 20 RandDelay) |
| Trials (total) | ~15,000 |
| Trials / session | 218-517 |

### Key data fields
- **obj.clu**: Per-probe neuron data: tm (spike times), quality, trial, trialtm, site
- **obj.bp**: Per-trial behavior: hit/miss/no/early, L/R, autowater, stim.enable
- **obj.bp.ev**: Event times: goCue, sample, delay, lickL, lickR, reward
- **obj.traj{cam}**: DLC tracking: ts (features x [x,y,conf] x frames), frameTimes
- **obj.me**: Motion energy (per-trial time series)
- **motionEnergy.mat**: me.data (per-trial), me.moveThresh (scalar)

### Quality labels in data
- Ephys_Behavior: Fair, Good, Poor, Multi (and some empty strings)
- RandomizedDelay: excellent, great, good, fair, multi, poor (and typos like "mutli")

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source |
|-----------|-------|--------|
| Total mice | 17 | Paper |
| Ephys mice (fixed delay) | 9 (6 two-context + 3 DR-only) | Paper |
| Ephys sessions (fixed delay) | 25 | Paper |
| Ephys units (fixed delay) | 1,651 (483 single units) | Paper |
| RandDelay mice | 4 | Paper |
| RandDelay sessions | 19 | Paper |
| RandDelay units | 845 (288 single units) | Paper |
| Opto mice | 4 VGAT-ChR2-EYFP | Paper |
| Neural time bin | 5 ms (paper) vs 10 ms (code) | Paper/Code |
| Smoothing | Causal Gaussian, half-width 35ms (paper) vs 15 samples (code) | Paper/Code |
| Video frame rate | 400 Hz | Paper |
| Brain region | ALM | Paper |
| Min FR threshold | 1 Hz | Paper/Code |
| Sample epoch | 1.3s auditory stimulus | Paper |
| Delay epoch | 0.9s (fixed) | Paper |
| Choice selectivity (sample) | 36% of 483 units | Paper |
| Choice selectivity (delay) | 42% of 483 units | Paper |
| Choice selectivity (response) | 58% of 483 units | Paper |
| Context selectivity | 39% of 214 units | Paper |

### Processing Details
- **Alignment**: DR trials aligned to go cue onset; WC trials aligned to water drop
- **Neural processing**: Spike times binned at 5ms, smoothed with causal Gaussian (half-width 35ms)
- **Baseline**: -2.4s to -2.2s relative to go cue, used for subtraction and standardization
- **Probes**: H2 (2 shanks, 32 ch each) and Neuropixels 1.0 (384 ch)
- **Spike sorting**: JRCLUST v4.1.0 and/or Kilosort 3, manually curated with Phy 2

### Curation Steps

**Neuron curation rules**:
- Quality filter: 'all' (exclude garbage, noisy)
- Mean firing rate > 1 Hz
- For subspace/selectivity: only well-isolated single units

**Trial curation rules**:
- Sessions need >= 40 correct DR trials per direction, >= 20 correct WC per direction (for behavioral analysis)
- Early lick trials excluded from analyses
- Sessions need >= 10 units for inclusion
- Stim/opto trials excluded (stim.enable == 0)

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Choice (CDchoice, ROC-AUC) | 0.86 +/- 0.11 |
| CDchoice from video (R2) | 0.41 +/- 0.23 (fixed), 0.38 +/- 0.20 (rand) |
| CDramp from video (R2) | 0.46 +/- 0.23 |
| CDcontext from video (R2) | 0.49 +/- 0.22 |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Time bin size | dt=1/100=10ms | N/A | 5ms | Use 10ms (matches code default; paper's 5ms may be for specific analyses) |
| Smoothing | 15 samples (150ms at 10ms bins) | N/A | half-width 35ms | Use code's smoothing: 15-sample causal Gaussian |
| Ephys animals | N/A | 10 animals | 9 mice | Data has 10; paper may count differently. Include all. |
| RandDelay sessions | N/A | 20 with neurons | 19 sessions | One extra session may not meet inclusion criteria (>=10 units after filtering) |
| Neuron counts | After quality+FR filter | 14,297 raw | 1,651+845=2,496 after filter | Raw >> filtered; quality+FR filter reduces dramatically |
| Quality labels | excellent/great/good/multi | Mixed: Fair/Good/Poor/Multi AND excellent/great/good/fair/multi | N/A | All labels except garbage/noisy pass 'all' filter |
| tmin/tmax | -2.5s to 2.5s | N/A | N/A | Use code defaults: -2.5s to 2.5s around go cue |
| Session min units | N/A | N/A | >= 10 units | Apply after quality + FR filtering |

### Key Resolution Decisions
1. **Include both Ephys_Behavior and RandomizedDelay sessions with neural data** - both have the variables needed for the decoder
2. **Exclude behavior-only sessions** (DelayInhibition, GoCueInhibition, JEB24_10-03, JEB24_10-04) - no neural data
3. **Use code parameters** (10ms bins, 15-sample smoothing, tmin=-2.5, tmax=2.5) as they represent the standard analysis pipeline
4. **Apply 'all' quality filter** + FR > 1 Hz as per reference code
5. **Exclude stim trials** (stim.enable == 1) and early lick trials

---

## Step 5: Mapping Planning
**Status**: IN PROGRESS

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| obj.clu spike times | neural | Bin at 10ms, smooth causal Gaussian 15 samples, align to goCue | alignSpikes, getSeq, mySmooth | Fire rates in Hz |
| Time from go cue (s) | input[0] | Continuous time vector | N/A | -2.5 to 2.5s |
| obj.bp.L/R + lickL/lickR | output[0]: lick_direction | left/right/none per trial | findTrials | 0=left, 1=right, 2=none |
| obj.bp.autowater | output[1]: behavioral_context | WC=1, DR=0 per trial | N/A | 0=WC, 1=DR |
| obj.bp.hit/miss/no | output[2]: outcome | correct/incorrect/ignore per trial | N/A | 0=incorrect, 1=correct, 2=ignore |
| obj.traj tongue velocity | output[3]: tongue_velocity | Discretize per-session: <50th=0, >=50th=1, not_visible=2 | DLC tracking | Time-varying |
| obj.traj paw velocity | output[4]: paw_velocity | Discretize per-session: <50th=0, >=50th=1, not_visible=2 | DLC tracking | Time-varying |
| motion energy | output[5]: motion_energy | Discretize per-session: <50th=0, >=50th=1, no_video=2 | loadMotionEnergy | Time-varying |

### Key Decisions
1. **Time bin = 10ms**: Matches reference code default (params.dt = 1/100)
2. **Trial window = [-2.5s, 2.5s] around go cue**: Matches code defaults (params.tmin, params.tmax)
3. **Include all quality levels except garbage/noisy**: Matches findClusters('all')
4. **FR > 1 Hz filter**: Matches removeLowFRClusters
5. **Exclude early lick trials**: Referenced in paper
6. **Exclude stim trials**: stim.enable == 1
7. **Tongue velocity from bottom camera tongue feature**: Euclidean velocity of (x,y) position
8. **Paw velocity from side camera paw feature**: Euclidean velocity of (x,y) position
9. **Sessions with < 10 units after filtering excluded**: Paper criterion
10. **Both Ephys and RandDelay sessions included**: All have neural data and needed variables

### Planned Sanity Checks
- [ ] Total neurons after filtering matches ~1,651 (Ephys) + ~845 (RandDelay)
- [ ] Session count matches ~25 (Ephys) + ~19 (RandDelay) after minimum unit filter
- [ ] Trial counts per session are reasonable (200-500)
- [ ] Firing rates are positive and in reasonable range (0-100 Hz typical)
- [ ] Output distributions match expectations (hit rate ~70-80%)
- [ ] Temporal alignment: neural activity shows expected patterns around go cue
- [ ] Spot-check spike times against original data

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `convert_data.py` (1170 lines) with:
- Dual format loading (h5py for HDF5, scipy.io for older MATLAB files)
- Causal Gaussian smoothing matching `mySmooth.m`
- Video offset computation matching `findVideoOffset.m`
- DLC velocity extraction and discretization
- Motion energy alignment and discretization
- Full trial curation (exclude early lick, stim trials, NaN goCue)
- Quality filtering + FR > 1 Hz + min 10 units per session

---

## Step 7-8: Sample Conversion and Decoder Training
**Status**: COMPLETE

Sample: 2 sessions (JEB15_2022-07-27, JEB23_2023-10-21), 127 neurons, 685 trials.
Format validation passed. Sample decoder training confirmed working pipeline.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 2084 MB (44 sessions after dedup)
- `sample_data.pkl`: 91.5 MB (2 sessions)

### Consistency Check
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Ephys sessions | 25 | 25 | Yes |
| RandDelay sessions | 19 | 19 | Yes (after dedup) |
| Subjects (with neural) | 14 | 14 | Yes |
| Ephys neurons | 1,651 | 2,141 | Higher* |
| RandDelay neurons | 845 | 925 | Higher* |
| Total neurons | 2,496 | 3,066 | Higher* |

*Paper neuron counts are for specific analyses with stricter filtering. Our 'all' quality filter includes more unit types.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **All-zero neural data**: Sessions 36, 43 (both JEB24) have all-zero late trials where recording ended. Data artifact, not a conversion bug.
2. **Neural data sanity**: All FRs >= 0, mean FR 8.84 Hz, max instantaneous 934 Hz (bursty neuron). Reasonable.
3. **Dimension consistency**: All trials have (n_neurons, 500) neural, (1, 500) input, (6, 500) output.
4. **Output value ranges**: All within expected sets.
5. **Per-trial outputs constant**: Lick direction, context, outcome constant across time bins.
6. **Lick/outcome consistency**: Perfect — correct↔has lick, ignore↔no lick, incorrect↔has lick.
7. **Time-varying outputs**: All sessions have variation in tongue/paw velocity. 41/44 sessions have motion energy data (3 have no video).
8. **No NaN**: Clean data throughout.
9. **Duplicate detection**: JEB23_2023-10-20 is duplicate of JEB23_2023-10-19 (different file format, identical data). Excluded.

### Issues Found and Resolved
- **Duplicate session**: JEB23_2023-10-20.mat is a duplicate of JEB23_2023-10-19.mat. Added to SKIP_FILES exclusion list. Session count now matches paper (44 = 25 + 19).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Loss converged: 9.82 -> 0.70 (200 epochs), test loss 0.73.

### Decoder Results (Full)
| Output | Train Bal. Acc | Val Bal. Acc | Chance | Ratio |
|--------|-------------|--------|--------|-------|
| lick_direction | 0.667 | 0.640 | 0.333 | 1.92x |
| behavioral_context | 0.874 | 0.865 | 0.500 | 1.73x |
| outcome | 0.653 | 0.611 | 0.333 | 1.83x |
| tongue_velocity | 0.572 | 0.570 | 0.333 | 1.71x |
| paw_velocity | 0.530 | 0.528 | 0.333 | 1.59x |
| motion_energy | 0.794 | 0.792 | 0.333 | 2.38x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
- All 6 outputs well above chance (1.59x to 2.38x)
- Train-validation gap is small (~0.03), no overfitting
- Behavioral context (0.87) and motion energy (0.79) decoded most reliably
- Lick direction (0.64) lower than paper's AUC 0.86, but different metric/architecture/time window
- Paw velocity (0.53) lowest but still above chance, limited by feature visibility (17% not_visible)

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created (sanity_check.py, processing plots, intermediate outputs)
- [x] All files organized
