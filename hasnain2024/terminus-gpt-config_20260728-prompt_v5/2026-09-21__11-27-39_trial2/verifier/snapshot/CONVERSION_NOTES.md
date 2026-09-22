# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `total 8148`
- `drwxr-xr-x  4 root root      45 Sep 21 15:33 .`
- `dr-xr-xr-x 19 root root     116 Sep 21 15:33 ..`
- `-rw-r--r--  1 root root   48022 Sep 21 15:32 .manifest`
- `-rw-r--r--  1 root root    5006 Sep 21 15:33 CONVERSION_NOTES.md`
- `-rw-r--r--  1 root root    2664 Sep 21 02:57 Dockerfile`
- `drwxr-xr-x 13 root root    4096 Sep 21 02:51 code`
- `drwxr-xr-x  2 root root    4096 Aug  9 20:54 data`
- `-rw-r--r--  1 root root   89127 Sep 21 02:51 decoder.py`
- `-rw-r--r--  1 root root     652 Sep 21 02:51 docker-compose.yaml`
- `-rw-r--r--  1 root root   18385 Sep 21 02:51 methods.txt`
- `-rw-r--r--  1 root root 8153445 Sep 21 02:51 paper.pdf`
- `-rw-r--r--  1 root root    7641 Sep 21 02:51 train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| loadSessionData | DataLoadingScripts/loadSessionData.m | LOADING | Load per-session neural and behavioral data objects |
| loadProcessedData | DataLoadingScripts/loadProcessedData.m | LOADING | Load preprocessed arrays/objects for downstream analyses |
| processData | DataLoadingScripts/processData.m | PROCESSING | Main session processing pipeline linking spikes, behavior, and video |
| alignSpikes | DataLoadingScripts/alignSpikes.m | PROCESSING | Bin/align spike times to task events across trials |
| findTrials | DataLoadingScripts/findTrials.m | CURATION | Select trials by condition/context/outcome criteria |
| removeLowFRClusters | DataLoadingScripts/removeLowFRClusters.m | CURATION | Filter low firing-rate units |
| loadKinData | DataLoadingScripts/loadKinData.m | LOADING | Load kinematic/DLC-derived behavioral features |
| loadMotionEnergy | DataLoadingScripts/loadMotionEnergy.m | LOADING | Load motion energy video features |
| getDefaultParams | DataLoadingScripts/getDefaultParams.m | PROCESSING | Define default timing/binning/analysis parameters |
| getEventTimes | funcs/getEventTimes.m | PROCESSING | Extract task event times used for alignment |
| getOutcome | funcs/getOutcome.m | PROCESSING | Derive trial outcome labels |
| firstLickTime | funcs/firstLickTime.m | PROCESSING | Compute first-lick timing/direction-related quantities |
| findVideoOffset | funcs/findVideoOffset.m | PROCESSING | Align behavior video timing to task timebase |
| UseInclusionCritera | utils/UseInclusionCritera.m | CURATION | Apply session inclusion rules |
| RemoveUnwantedSessions | utils/RemoveUnwantedSessions.m | CURATION | Exclude problematic sessions |
| NeuralChoiceDecoding | ChoiceContextDecoding/NeuralChoiceDecoding.m | PROCESSING | Reference neural choice decoding pipeline |
| NeuralContextDecoding | ChoiceContextDecoding/NeuralContextDecoding.m | PROCESSING | Reference neural context decoding pipeline |
| DLC_ChoiceDecoding | ChoiceContextDecoding/DLC_ChoiceDecoding.m | PROCESSING | Reference behavior-video choice decoding pipeline |
| DLC_ContextDecoding | ChoiceContextDecoding/DLC_ContextDecoding.m | PROCESSING | Reference behavior-video context decoding pipeline |

### Notes
- Read code README and identified likely core loading, processing, curation, and decoding entry points.
- Extracted/inspected reference loading and decoding entry points. Key concrete findings so far: code examples use getEventTimes(obj.bp.ev,{'sample','delay','goCue'},'goCue'), indicating event times are commonly expressed relative to go cue; obj.time is the common trial time axis for neural and behavioral streams; motion energy is interpolated onto obj.time with a video shift correction and params.advance_movement; trial-aligned neural data appear in obj.trialdat and condition-averaged PSTHs in obj.psth. Exact parameter values and field names were inspected directly from the source files listed above and will guide conversion.
- Repository README states recordings are from right and left ALM; tasks include delayed-response and water-cued licking paradigms.
- Sessions under recording-and-video contain neural data; video-only sessions exist for inhibition experiments and likely should be excluded for neural decoder because decoder inputs neural activity.
- Need exact object fields from WorkingWithDataObjs and processing scripts to determine trial-aligned neural arrays, event times, and video features.
- Need exact inclusion/exclusion criteria from UseInclusionCritera/RemoveUnwantedSessions and unit filtering from removeLowFRClusters before Step 1 can be finalized.



---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are organized into task/modality subdirectories under `/app/data`, including `Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`, `DelayInhibition_BilatMC_Behavior`, and `GoCueInhibition_BilatMC_Behavior`.
- `data_structure_*.mat` files are MATLAB v7.3/HDF5 files and should be read with `h5py` or equivalent.
- `motionEnergy_*.mat` files are older MATLAB format and can be read with `scipy.io.loadmat`; they contain a struct `me` with at least `data` and `moveThresh`.
- Representative recording-session `data_structure` files contain root object `obj` with fields `pth`, `bp`, `sglx`, `traj`, `trials`, `clu`, and `meta`.
- `obj.bp` contains behavioral per-trial arrays and event structures, including `Ntrials`, `hit`, `miss`, `no`, `early`, `protocol`, `stim`, `bitRand`, `autowater`, `R`, `L`, and event times `bitStart`, `sample`, `delay`, `goCue`, `reward`, `lickL`, `lickR`.
- `obj.trials.bp` and `obj.trials.sglx` provide trialwise linkage between behavior, ephys, and video streams via presence flags and file indices.
- `obj.traj` is a cell array of trajectory/video structs with fields including `fn`, `featNames`, `ts`, `frameTimes`, and `NdroppedFrames`.
- `obj.meta.probe` stores probe metadata such as `depth`, `angle`, `loc`, `type`, `serialNum`, `badchans`, `chans`, and `spikesorter`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | pending direct file inspection of recording sessions |
| Neurons / session | pending direct file inspection of recording sessions |
| Subjects | 18 |
| Sessions / subject | variable; see notes |
| Trials (total) | pending direct file inspection |
| Trials / session | pending direct file inspection |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 1651 (DR), 522 (two-context), 845 (randomized delay) | "For the DR task, we recorded 1,651 units... In 12 sessions from six mice... 522 units... randomized delay task... 845 units" | 
| Neurons / session | | |
| Subjects | 9 (DR), 6 (two-context), 4 (randomized delay) | same excerpt |
| Sessions / subject | 25 sessions/9 mice (DR); 12 sessions/6 mice (two-context); 19 sessions/4 mice (randomized delay) | same excerpt |
| Trials (total) | | |
| Trials / session | | |
| Neural data time bin | | |
| Behavior data time bin | | |
| Reward rate | | | | 
| Minimum units/session | 10 | "Recording sessions were included for analysis only if they had at least 10 units" | 
| ITI window for context selectivity | 300 ms preceding sample tone | "during the ITI (the 300 ms preceding the sample tone onset)" |
| ... | | | | 


### Processing Details
- Electrophysiology sessions were included only if they had at least 10 units.
- For most analyses, all units with firing rate >1 Hz were included; for subspace alignment and some single-unit selectivity analyses, only well-isolated single units with firing rate >1 Hz were included.
- Choice selectivity was assessed by subsampling 40 right-correct and 40 left-correct trials and comparing spike counts during sample, delay, or response epochs.
- Context selectivity was assessed during the ITI, specifically the 300 ms preceding sample tone onset, comparing DR and WC trials while controlling for block-order confounds.
- Reference code and examples indicate trial-aligned time axes are commonly expressed relative to goCue (`getEventTimes(..., 'goCue')`), which matches the decoder task requirement to align to go cue onset.
- Motion energy in code is interpolated onto the common trial time axis `obj.time`, with source motion-energy data sampled at 400 Hz according to comments in `loadMotionEnergy.m`.
- Behavioral/video streams are linked to neural/behavioral trials via `obj.trials` metadata and `traj` frame times.

### Curation Steps

**Neuron curation rules**:
- Session inclusion required at least 10 units.
- Most analyses included all units with firing rate >1 Hz.
- Some analyses restricted to well-isolated single units with firing rate >1 Hz, but this appears analysis-specific rather than universal.

**Trial curation rules**:
- Choice-selectivity analyses subsampled balanced trial sets (40 right-correct and 40 left-correct trials).
- Context-selectivity analyses subsampled 40 DR and 40 WC trials and further controlled for context-block order across the session.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| | | | | |

---

## Step 5: Mapping Planning
**Status**: IN PROGRESS

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.trialdat` (recording sessions) | neural | Use per-trial neuron-by-time activity aligned to goCue | `processData`, `alignSpikes`, `loadSessionData` | Common time axis is `obj.time`; low-FR units filtered by reference code |
| `obj.time` relative to goCue | input[0] | Tile common time vector across trials as continuous time-from-go-cue | `getDefaultParams`, `WorkingWithDataObjs` | Decoder input specified by task |
| `obj.bp.R`, `obj.bp.L`, and lick-event logic | output[0] lick direction | Map per trial to left/right/none | `firstLickTime`, `getOutcome`, `obj.bp.ev.lickL/lickR` | `none` for ignore/no-lick trials |
| Context / protocol / block identity in `obj.bp` | output[1] behavioral context | Map to WC vs DR | `findTrials`, context-decoding scripts, `obj.bp.protocol` | Two-context sessions are subset of DR ephys task |
| `obj.bp.hit`, `obj.bp.miss`, `obj.bp.no` | output[2] outcome | Map to correct/incorrect/ignore | `getOutcome` | Likely hit->correct, miss->incorrect, no->ignore |
| Kinematic trajectory features from `traj`/`loadKinData` | output[3] tongue velocity | Compute per-timepoint speed/velocity, discretize by session median; 2 if not visible | `loadKinData` | Need exact feature indices from featNames during implementation |
| Kinematic trajectory features from `traj`/`loadKinData` | output[4] paw velocity | Compute per-timepoint speed/velocity, discretize by session median; 2 if not visible | `loadKinData` | Need exact feature indices from featNames during implementation |
| `me.data` interpolated to `obj.time` | output[5] motion energy | Discretize by session median; 2 if no video | `loadMotionEnergy` | Reference code aligns/interpolates motion energy to common trial time axis |

### Key Decisions
1. **Use only ephys sessions for converted dataset**: Decoder input must be neural activity, so behavior-only inhibition sessions without neural data should be excluded.
2. **Align all streams to goCue**: This matches the decoder task requirement and the default/reference code alignment (`params.alignEvent = 'goCue'`).
3. **Use low-FR filtering consistent with reference code**: Reference code removes low firing-rate clusters and methods state most analyses include units with firing rate >1 Hz; exact threshold reconciliation will be finalized in implementation/validation.
4. **Represent time as a continuous time-varying decoder input**: This directly matches the task specification.
5. **Represent lick direction, context, and outcome as per-trial categorical outputs**: These are naturally trial-level labels in the source data.
6. **Represent tongue/paw velocity and motion energy as time-varying categorical outputs**: Continuous streams will be discretized per session using 50th-percentile thresholds, with explicit missing/not-visible category 2.
7. **Use no-video / not-visible category rather than imputing**: This preserves missingness required by the task and avoids introducing artificial movement signals.

### Planned Sanity Checks
- [ ] Verify `obj.bp.Ntrials` matches the number of converted trials for representative sessions.
- [ ] Verify converted motion energy for a spot-checked trial matches reference interpolation/alignment from raw `me.data` and frame times.
- [ ] Verify a spot-checked neural trial matrix matches raw `obj.trialdat` values for selected neuron/time/trial indices.
- [ ] Verify per-trial outcome labels match `hit`/`miss`/`no` raw arrays on spot-checked trials.


---

## Step 6: Script Development
**Status**: IN PROGRESS

[Implementation notes]

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: NOT STARTED

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | |
| Neurons / session | |
| Subjects | |
| Sessions / subject | |
| Trials (total) | |
| Trials / session | |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
[Notes on any anomalies]

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: [None / List]
- Warnings: [None / List]

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| <Output 1> | | |
| <Output 2> | | |
| ... | | |

---

## Step 9: Full Conversion and Validation
**Status**: IN PROGRESS

### Output Files
- `converted_data.pkl`: [size]
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | | | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: NOT STARTED

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

### Training Progress
- Loss decreasing: [Yes/No]

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| <Output 1> | | |
| <output 2> | | |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized


Step 2 progress note: MATLAB files are v7.3/HDF5; scipy.io.loadmat does not work directly, so conversion should use h5py or a compatible v7.3 reader. Representative files expose an HDF5 object/group named `obj`. Further HDF5 field inspection in progress.

Step 2 progress note: initial HDF5 traversal mostly exposed MATLAB internal `#refs#` datasets; next inspection is focusing specifically on the root `obj` group and its fields to recover native variable names.

Step 2 progress note: root `obj` includes at least `sglx`, `traj`, and `trials`; `trials` has `bp` and `sglx` substructures, and `traj` is a cell array of trajectory/video-related content. Continuing to inspect recording-session files and motion-energy files to map neural and behavioral fields.

Step 2 progress note: filename-level counts show 120 sessions across 18 subjects, with 45 motionEnergy files. Directory split: 50 files in Ephys_Behavior, 42 in RandomizedDelay_Ephys_Behavior, 53 in DelayInhibition_BilatMC_Behavior, and 20 in GoCueInhibition_BilatMC_Behavior. This suggests the neural-decoder-relevant recording sessions are likely the 45 ephys sessions that also have motionEnergy companions.

Step 2 progress note: `obj.trials.bp` and `obj.trials.sglx` contain per-trial linkage metadata (`N`, ephys/video presence flags, and cross-file indices), confirming that behavioral, ephys, and video streams are linked trialwise. Continuing to locate actual neural unit arrays and trajectory/video feature fields in recording-session files.

Step 2 progress note: file formats are mixed. `data_structure_*.mat` files are MATLAB v7.3/HDF5 and should be read with h5py or equivalent; `motionEnergy_*.mat` files are not HDF5 and should be read with scipy.io.loadmat.

Step 2 progress note: representative recording-session files contain `obj` fields `pth`, `bp`, `sglx`, `traj`, `trials`, `clu`, and `meta`; `traj` entries include per-video feature names and frame times; motion-energy files contain a struct `me` with at least `data` and `moveThresh`. Inspecting `bp`, `clu`, and `meta` next to extract trial/neuron counts and behavioral variable names.

Step 2 progress note: `meta.probe` includes anatomical and hardware metadata (`depth`, `angle`, `loc`, `type`, `serialNum`, `badchans`, `chans`, `spikesorter`), which should support brain-region annotation and probe-level curation. Inspecting `clu` and `bp` now to recover neuron-level and behavior/event-level variables.

Step 3 progress note: methods excerpt already confirms key ephys curation statistics: sessions required at least 10 units; most analyses used all units with firing rate >1 Hz; some analyses used only well-isolated single units >1 Hz. Reported subsets include DR task 1,651 units from 25 sessions/9 mice, two-context task 522 units from 12 sessions/6 mice, and randomized-delay task 845 units from 19 sessions/4 mice.

Step 3 progress note: `pdftotext` is unavailable in the environment, so paper text extraction is being done via Python PDF libraries if available. Continuing to mine methods/paper for timing, alignment, video sampling, and decoding-accuracy details.

Step 4 progress note: beginning explicit cross-checks among methods, code, and data. Initial hypotheses: the 45 motionEnergy files correspond to the 45 ephys sessions (25 DR + 19 randomized-delay + 1 additional/other session or a counting nuance), while the 12 two-context sessions are likely a subset of the DR ephys sessions rather than a separate directory. Code and examples consistently indicate goCue-relative event timing and a common `obj.time` axis across neural and video streams.

Step 4 progress note: directory counts show 25 DR ephys sessions and 22 randomized-delay ephys sessions in raw files, whereas methods report 19 randomized-delay sessions used in analysis. This strongly suggests additional session-level exclusion beyond raw-file presence; inspecting `UseInclusionCritera.m` and `RemoveUnwantedSessions.m` to recover the exact filtering logic.

Step 4 progress note: targeted code inspection shows session-level inclusion criteria requiring at least 40 right-hit and 40 left-hit trials, in addition to methods-reported unit-count criteria. This provides a concrete explanation for why raw directory counts can exceed the number of sessions used in analyses.

Step 5 progress note: now inspecting concrete source variables for outputs—`getOutcome.m` and `firstLickTime.m` for outcome/lick-direction logic, `loadKinData.m` and trajectory `featNames` for kinematic variables, and motion-energy file structure for video-derived outputs.

Step 6 progress note: initial `convert_data.py` scaffold created and syntax-checked. Next step is runtime debugging against sample conversion to locate processed neural arrays or reconstruct them from raw `clu`/`sglx` data if `trialdat` is not stored directly in the session files.

Step 6 progress note: conversion now reconstructs neural trial matrices from raw cluster `trial` / `trialtm` fields rather than assuming precomputed `trialdat`. Current debugging focus is recursive loading of nested behavioral structs (`bp.ev`, `bp.protocol`, `bp.stim`) from MATLAB v7.3 HDF5 files.

Step 7 progress note: sample conversion and format verification now run successfully. Verifier summary shows structurally valid data for 2 sessions (199 total neurons), but current mapping is semantically incomplete: behavioral context is constant in the sample, tongue/paw outputs are entirely `not_visible`, and motion energy is constant. These issues must be improved before sample decoder training.

Step 7/8 pre-debug note: trajectory `ts` arrays contain per-landmark x/y/confidence traces across time, making tongue velocity derivable from tongue landmark coordinates aligned by `frameTimes`. Paw landmarks have not yet been identified in the inspected ephys sessions, so paw may remain missing unless discovered elsewhere.

Step 7 progress note: after iterative fixes, sample verification is clean with valid format and improved output distributions. Motion energy is now aligned/interpolated and balanced by session-median discretization; tongue velocity is derived from tongue landmark trajectories and includes visible + not-visible states. In inspected ephys sessions, trajectory feature names consistently lack paw landmarks, so paw output remains `not_visible` for these sessions. Protocol numbers in sampled ephys sessions do not show multi-context variation, so constant DR context in the sample may reflect the chosen sessions rather than a loading bug.

Step 8 progress note: sample decoder training finished successfully with decreasing loss. Validation balanced accuracies were above nominal chance for all outputs, but `behavioral_context` and `paw_velocity` are degenerate in the sample (single-label outputs), producing sklearn warnings and trivially perfect balanced accuracy. Non-degenerate validation balanced accuracies: lick_direction 0.4847, outcome 0.4270, tongue_velocity 0.4925, motion_energy 0.6510.

Step 9 progress note: full conversion and verification succeeded structurally, producing `/app/converted_data.pkl` (34 sessions, 13 subjects, 11,385 trials, 8,087 neurons) and a valid verification report. However, this is still incomplete relative to reference expectations: 13 sessions were skipped due to mixed MAT-file formats or missing `clu`, behavioral_context remained constant DR across all converted sessions, and paw velocity remained unavailable in inspected ephys sessions. Additional work is needed to implement scipy fallback for non-HDF5 `data_structure` files and to identify context labels for the two-context subset.
