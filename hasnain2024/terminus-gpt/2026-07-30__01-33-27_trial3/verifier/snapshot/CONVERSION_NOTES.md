# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse (paper/code/data provided in project)
- **Date started**: 2026-07-30
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code
- data
- decoder.py
- docker-compose.yaml
- methods.txt
- paper.pdf
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| getDefaultParams | code/DataLoadingScripts/getDefaultParams.m | LOADING | Defines default analysis parameters including alignment-related settings and timing metadata. |
| loadProcessedData | code/DataLoadingScripts/loadProcessedData.m | LOADING | Main entry for loading preprocessed session-level data structures used in analyses. |
| processData | code/DataLoadingScripts/processData.m | PROCESSING | Builds processed analysis structures from raw/session data. |
| loadSessionData | code/DataLoadingScripts/loadSessionData.m | LOADING | Loads per-session neural/behavior/video data objects. |
| findTrials | code/DataLoadingScripts/findTrials.m | CURATION | Defines trial groupings/IDs by condition used for decoding and analyses. |
| alignSpikes | code/DataLoadingScripts/alignSpikes.m | PROCESSING | Aligns spike trains to task events on a common trial time axis. |
| loadKinData | code/DataLoadingScripts/loadKinData.m | PROCESSING | Loads/interpolates kinematic features such as tongue/paw trajectories onto trial-aligned axes. |
| loadMotionEnergy | code/DataLoadingScripts/loadMotionEnergy.m | PROCESSING | Interpolates motion energy to aligned trial time axis, fills edge NaNs, and thresholds movement. |
| removeLowFRClusters | code/DataLoadingScripts/removeLowFRClusters.m | CURATION | Removes low firing-rate neural clusters/cells. |
| NeuralChoiceDecoding | code/ChoiceContextDecoding/NeuralChoiceDecoding.m | PROCESSING | Decodes lick choice from neural activity across aligned time bins. |
| NeuralContextDecoding | code/ChoiceContextDecoding/NeuralContextDecoding.m | PROCESSING | Decodes task context from neural activity across aligned time bins. |
| DLC_ChoiceDecoding | code/ChoiceContextDecoding/DLC_ChoiceDecoding.m | PROCESSING | Decodes lick choice from video/DLC-derived features. |
| DLC_ContextDecoding | code/ChoiceContextDecoding/DLC_ContextDecoding.m | PROCESSING | Decodes context from video/DLC-derived features. |

### Notes
- Codebase is primarily MATLAB, organized around session loading, processing, and decoding analyses.
- Relevant subdirectory for conversion is `code/DataLoadingScripts`, with decoder examples in `code/ChoiceContextDecoding`.
- Decoder scripts indicate `rez.binSize = 75; % ms`, suggesting the reference decoding analyses use 75 ms time bins.
- Context/choice decoder scripts compute `rez.dt = floor(rez.binSize / (params(1).dt*1000))`, implying a finer native time base in `params.dt` that is rebinned for decoding.
- `findTrials`/`trialid` condition groupings are central for constructing per-trial labels such as context, choice, and outcome.
- `alignSpikes` is the likely reference for event alignment of neural data; we need to match its alignment event and trial window during conversion.
- `removeLowFRClusters` indicates explicit neural curation by firing-rate threshold.
- `loadKinData` and `loadMotionEnergy` suggest behavioral features are interpolated onto the same aligned trial axis as neural data.
- From `loadMotionEnergy.m`: frame times are assumed at 400 Hz, data are interpolated using `interp1(frameTimes-0.5-alignTimes(trix), ..., taxis)`, NaNs at trial starts are filled with nearest values, and binary movement is created by thresholding motion energy (`me.move = me.data > me.moveThresh`).
- Choice/context decoder scripts sample balanced trial subsets across conditions before train/test splitting, useful for understanding label definitions though not necessarily for conversion.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains four dataset families:
  - `DelayInhibition_BilatMC_Behavior`
  - `Ephys_Behavior`
  - `GoCueInhibition_BilatMC_Behavior`
  - `RandomizedDelay_Ephys_Behavior`
- Files are MATLAB `.mat` files.
- `data_structure_*.mat` files are MATLAB v7.3 / HDF5 for at least the inspected session and contain a top-level `obj` group.
- In inspected `RandomizedDelay_Ephys_Behavior/data_structure_JEB11_2022-05-10.mat`, `obj` contains keys: `bp`, `clu`, `ex`, `meta`, `pth`, `sglx`, `traj`, `trials`.
- `obj/bp/ev` contains trial event variables including `bitStart`, `delay`, `goCue`, `lickL`, `lickR`, `reward`, and `sample`. In the inspected session these event arrays span 365 trials, giving a representative trial count for one session.
- `lickL` and `lickR` are stored as object arrays, consistent with variable-length per-trial lick timestamps.
- `obj/trials` contains `bp` and `sglx` subgroups, suggesting trialized behavior and spike/ephys views coexist in the session object.
- `obj/clu` is stored through HDF5 object references, implying cluster/neuron metadata are stored in referenced structures rather than flat arrays.
- `obj/meta` contains subject/day/probe/channel metadata fields such as `anm`, `day`, `chans`, and `probe`.
- `motionEnergy_*.mat` files are standard MATLAB files with top-level variable `me`.
- Filename pairing suggests some ephys datasets have sidecar motion-energy files per session.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | pending deeper file inspection |
| Neurons / session | pending deeper file inspection |
| Subjects | 18 across all `data/` subdirectories by filename (4 + 10 + 4, with overlap possible by family) |
| Sessions / subject | varies by family; see notes |
| Trials (total) | pending deeper file inspection |
| Trials / session | pending deeper file inspection |

### Family Counts (from filenames)
- `DelayInhibition_BilatMC_Behavior`: 53 `data_structure` files, 0 `motionEnergy` files, subjects = `MAH13`, `MAH14`, `MAH20`, `MAH21`
- `Ephys_Behavior`: 25 `data_structure` files, 25 `motionEnergy` files, subjects = `EKH1`, `EKH3`, `JEB13`, `JEB14`, `JEB15`, `JEB19`, `JEB6`, `JEB7`, `JGR2`, `JGR3`
- `GoCueInhibition_BilatMC_Behavior`: 20 `data_structure` files, 0 `motionEnergy` files, subjects = `MAH13`, `MAH14`, `MAH20`, `MAH21`
- `RandomizedDelay_Ephys_Behavior`: 22 `data_structure` files, 20 `motionEnergy` files, subjects = `JEB11`, `JEB12`, `JEB23`, `JEB24`

### Notes
- Need to determine which dataset family/families correspond to the paper/task by comparing against reference code and methods.
- Need deeper inspection of `obj.bp`, `obj.clu`, `obj.trials`, `obj.traj`, and `me` fields to identify trial variables, spike data, kinematics, and alignment events.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 1,651 (DR task) | "For the DR task, we recorded 1,651 units ... from 25 sessions using nine mice." |
| Subjects | 9 (DR task) | same quote |
| Sessions | 25 (DR task) | same quote |
| Neurons (two-context task) | 522 units | "In 12 sessions from six mice, animals performed the two-context task. In total, 522 units ... were recorded in these sessions." |
| Subjects (two-context task) | 6 mice | same quote |
| Sessions (two-context task) | 12 sessions | same quote |
| Neurons (randomized delay task) | 845 units | "Finally, for the randomized delay task, we recorded 845 units ... from 19 sessions using four mice." |
| Subjects (randomized delay task) | 4 mice | same quote |
| Sessions (randomized delay task) | 19 sessions | same quote |
| Min units / included recording session | 10 units | "Recording sessions were included for analysis only if they had at least 10 units" |
| Behavioral inclusion criterion | >=40 correct DR trials/direction and >=20 correct WC trials/direction | "All sessions used for behavioral analysis had at least 40 correct DR trials for each direction ... and 20 correct WC trials for each direction ... excluding early lick and ignore trials" |
| Trial exclusions | early lick and ignore trials omitted | same quote |
| Unit FR threshold for most analyses | >1 Hz | "All units with firing rates exceeding 1 Hz were included in all other analyses." |
| Unit subset for subspace/selectivity | well-isolated single units with FR >1 Hz | "only well-isolated single units with firing rates exceeding 1 Hz were included" |

### Processing Details
- Behavioral analyses use correct-trial count thresholds split by context (DR vs WC) and lick direction.
- Early lick and ignore trials are omitted from analyses.
- Motion energy during delay epoch and tongue visibility after go cue/water drop are explicitly analyzed in the paper.
- Context selectivity is assessed during the ITI, specifically the 300 ms preceding sample tone onset, using balanced subsampling across DR and WC trials.
- Choice selectivity uses balanced subsampling of 40 right and 40 left correct trials and compares spike counts during sample, delay, or response epochs.
- The codebase decoder scripts use 75 ms bins (`rez.binSize = 75` ms), which is likely relevant for reproducing reference processing for decoder-aligned features.

### Curation Steps

**Neuron curation rules**:
- Session inclusion requires at least 10 units.
- For most analyses, include all units with firing rate >1 Hz.
- For subspace alignment and single-unit selectivity analyses, restrict to well-isolated single units with firing rate >1 Hz.

**Trial curation rules**:
- Exclude early lick and ignore trials.
- Behavioral-session inclusion requires at least 40 correct DR trials per lick direction and 20 correct WC trials per lick direction.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | not yet extracted from text/code |
| Context | not yet extracted from text/code |
| Kinematic/video features | not yet extracted from text/code |


---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| DR task session count | Codebase includes an `Ephys_Behavior` family used by ephys analyses/decoders | `Ephys_Behavior` has 25 `data_structure` files and 10 subject IDs by filename | Paper reports DR task had 25 sessions from 9 mice | Likely one raw-data subject/file is excluded in paper analyses or one mouse ID is not part of the DR-task count after curation; keep investigating exact inclusion via code/session metadata. |
| Randomized delay session count | Codebase includes `RandomizedDelay_Ephys_Behavior` family | Raw files show 22 `data_structure` sessions from 4 mice and 20 motionEnergy sidecars | Paper reports randomized delay task had 19 sessions from 4 mice | Raw data likely contain extra sessions excluded by behavioral/unit-count criteria or missing sidecar data; final conversion must apply paper/code curation. |
| Two-context task location | Context decoder scripts operate on ephys/video data with DR/WC labels rather than a separately named folder | No standalone `TwoContext_*` data directory; likely context sessions are a subset of `Ephys_Behavior` or another ephys family | Paper reports 12 sessions from 6 mice for the two-context task | Working hypothesis: two-context task is a curated subset of ephys sessions identified through trial/context variables and code parameters rather than directory name. |
| Neural curation | `processData.m` removes low firing-rate clusters after alignment/PSTH extraction | Representative RandomizedDelay session has 63 raw cluster entries in `clu` | Paper says all units with FR >1 Hz were used for most analyses; some analyses use well-isolated single units only | Conversion should follow reference low-FR filtering and session inclusion rules, then decide whether all >1 Hz units or a quality subset is appropriate for decoder task. |
| Temporal binning | Context decoder scripts explicitly use `rez.binSize = 75` ms | Raw event streams and motion-energy data are on finer native time bases | Paper methods do not yet explicitly state decoder bin size in extracted text | Use 75 ms bins unless later text/code evidence shows a different binning for the conversion target. |
- `getDefaultParams.m` defines four core trial conditions for context/choice analyses: right-hit and left-hit trials with `~autowater` plus right-hit and left-hit trials with `autowater`, all excluding stimulation and early-lick trials.
- `NeuralContextDecoding.m` and `DLC_ContextDecoding.m` define context as `afccond = [1 2]` versus `awcond = [3 4]`, i.e. non-autowater (DR/2AFC) versus autowater (WC) trial groupings.
- Therefore, the two-context task is encoded via trial labels within session data rather than via a separate top-level data directory name.


---

## Step 5: Mapping Planning
**Status**: IN PROGRESS

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Spike times / trial-aligned spike data from `obj.clu` and trial structures | neural | Align to `goCue`, bin at 75 ms, convert to neuron x time trial matrices, filter low-FR units and excluded sessions | `alignSpikes.m`, `getSeq.m`, `removeLowFRClusters.m`, `processData.m` | Use all units >1 Hz for decoder unless stronger evidence requires quality subset |
| Trial time relative to go cue | input[0] | Continuous time-from-go-cue vector repeated for each trial/time bin | alignment logic from `alignSpikes.m`; decoder binning from `NeuralContextDecoding.m`/`DLC_ContextDecoding.m` | Decoder input specification requires time from go cue onset |
| Trial direction fields (`R`/`L` or equivalent) | output[0] lick_direction | Per-trial categorical: left=0, right=1 | `getDefaultParams.m`, `findTrials.m`, choice decoder scripts | Use trial labels rather than post-hoc lick timestamps for direction |
| `autowater` trial label | output[1] behavioral_context | Per-trial categorical: WC/autowater=0 or 1 depending on final convention, DR/non-autowater as the other class | `getDefaultParams.m`, `NeuralContextDecoding.m`, `DLC_ContextDecoding.m` | Need final value convention to match task spec: WC=0, DR=1 |
| Outcome fields (`hit`/`miss`, possibly ignore/early exclusions) | output[2] outcome | Per-trial categorical: incorrect=0, correct=1 after excluding early/ignore trials | `getDefaultParams.m`, `findTrials.m` | Need to define incorrect carefully for remaining non-ignored trials |
| Tongue trajectories / DLC features from `obj.traj` or processed kin data | output[3] tongue_velocity | Compute scalar tongue speed over aligned time bins, then discretize by per-session 50th percentile | `loadKinData.m`, DLC decoder scripts | Must inspect exact raw field names and derive speed consistently |
| Paw trajectories / DLC features from `obj.traj` or processed kin data | output[4] paw_velocity | Compute scalar paw speed over aligned time bins, then discretize by per-session 50th percentile | `loadKinData.m`, DLC decoder scripts | Must inspect exact raw field names and derive speed consistently |
| Motion energy from `motionEnergy_*.mat` `me.data` | output[5] motion_energy | Interpolate/align to trial time axis, bin to 75 ms, discretize by per-session 50th percentile | `loadMotionEnergy.m`, DLC decoder scripts | Fill edge NaNs as in reference code before binning/discretization |
| Subject metadata from filename / `obj.meta` | subjects, subject_idx | Extract unique mouse IDs and per-session indices | `obj.meta`, filenames | Session order must match neural/input/output lists |
| Cluster site/region metadata | brain_regions, brain_region_idx | Map neuron sites/probe metadata to ALM/other region labels | `obj.clu.site`, `obj.meta.probe/chans` | Paper emphasizes ALM; likely all neurons are ALM in relevant subset |

### Key Decisions
1. **Primary task subset**: Use the two-context ephys sessions identified by presence of both DR/non-autowater and WC/autowater trial types, consistent with paper context analyses and context decoder scripts.
2. **Temporal alignment**: Align all modalities to `goCue`, matching the decoder task specification and raw event availability in `obj.bp.ev.goCue`.
3. **Time binning**: Use 75 ms bins because the reference context decoder scripts explicitly use `rez.binSize = 75` ms.
4. **Trial exclusion**: Exclude early-lick and ignore trials, following methods text and condition definitions in `getDefaultParams.m`.
5. **Neural curation**: Apply low firing-rate filtering consistent with `removeLowFRClusters.m`; target unit inclusion is all units >1 Hz for decoder unless later evidence requires restricting to well-isolated single units.
6. **Context coding**: Convert raw `autowater`-style labels to task-required convention `WC=0`, `DR=1`.
7. **Continuous behavior outputs**: Convert tongue velocity, paw velocity, and motion energy to binary outputs using per-session median thresholds, as required by the task.
8. **Time-varying outputs**: Keep kinematic and motion-energy outputs time-varying across bins; keep lick direction/context/outcome per-trial and broadcast across time if needed for decoder compatibility.

### Planned Sanity Checks
- [ ] Verify one trial's binned neural counts against raw spike times aligned to `goCue` using direct recomputation.
- [ ] Verify one trial's motion-energy aligned trace against raw `me.data` interpolation and binning from `loadMotionEnergy.m` logic.
- [ ] Verify context labels from converted data match raw `autowater` labels for several manually checked trials.
- [ ] Verify lick-direction labels match raw `R`/`L` trial fields for several manually checked trials.
- [ ] Verify session counts / subject counts for the converted subset match paper-reported two-context task counts after curation.

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
**Status**: COMPLETE

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


Step 6 iteration note: initial sample conversion failed because all inspected `Ephys_Behavior/data_structure_*.mat` files are MATLAB v7.3/HDF5; scipy-only loading returned zero sessions. Need h5py-based session loader before Step 6 can complete.

Step 6 iteration note: HDF5 loader patch succeeded. `convert_data.py --sample` now finds 22 context-capable raw `Ephys_Behavior` sessions (both autowater states present) and writes a 2-session sample file. This exceeds the paper's 12-session two-context subset, so additional trial/session curation remains to be implemented. Neural and tongue/paw outputs are still placeholders and must be replaced with real extracted data.

Step 7 early validation note: `train_decoder.py --verify-only` passes on the 2-session sample. Real neural data are present (32 and 53 neurons), but `tongue_velocity` and `paw_velocity` remain degenerate placeholders (all low), so conversion is not yet behaviorally complete.

Step 7 validation update: after adding trajectory-based extraction, `tongue_velocity` is now non-constant but highly imbalanced (~97% high in the 2-session sample), suggesting thresholding/alignment still needs debugging. `paw_velocity` remains degenerate because no paw-like feature has yet been identified in the inspected Ephys trajectory streams.

Step 7 sample stats update: switching the sample subset to JEB13 sessions yields 626 trials across 2 sessions and 922 ALM neurons total, with structurally valid outputs. Remaining issue: `tongue_velocity` is still highly imbalanced (~96% high), indicating sparse-visibility or NaN-handling problems in tongue feature extraction/discretization.

Step 7 completion note: after nearest-fill interpolation of aligned trajectory speeds, sample verification on two JEB13 sessions shows balanced kinematic outputs (`tongue_velocity` ~0.555/0.445, `paw_velocity` ~0.528/0.472) and valid structure with 626 trials and 922 ALM neurons total.

Step 8 results: sample decoder training completed on JEB13 sample with decreasing loss (0.782 -> 0.612). Validation balanced accuracies: lick_direction 0.5071, behavioral_context 0.7109, outcome 0.5005, tongue_velocity 0.5151, paw_velocity 0.5846, motion_energy 0.5410. All outputs are at or above chance, with strongest performance for behavioral_context and paw_velocity.

Step 9 iteration note: full conversion failed at `motionEnergy_JEB15_2022-07-26.mat` because `me.data` entries are not uniformly numeric; at least some are nested MATLAB structs. Need a more robust motion-energy loader before full conversion can complete.

Step 9 iteration note: full conversion next failed at `data_structure_JEB6_2021-04-18.mat` because the `clu` HDF5 structure differs from later sessions; current code assumes `clu` resolves to a group with `trial` and `trialtm` fields. Need robust handling or session skipping for alternative neural-storage layouts.

Step 9 verification update: current full conversion passes structural verification and yields 21 sessions, 4,837 trials, and 5,512 ALM neurons. This is still inconsistent with the paper's two-context subset (12 sessions, 6 mice, 522 units), so additional curation is required before the conversion can be considered reference-matched.

Current status note: full conversion/verification are engineering-valid, but reference-matching curation remains unresolved. Code inspection shows many analyses assemble explicit mouse/session cohorts via `load*_ALMVideo` scripts rather than including all raw context-capable sessions, so final session selection should likely be derived from those curated meta lists.

Trial-filtering tradeoff note: enforcing the exact hit-only/no-stim conditions from `getDefaultParams.m` makes the `outcome` output degenerate (all correct), which conflicts with the decoder task requirement to predict incorrect vs correct outcome. For the decoder-task conversion, miss trials likely need to remain included so `outcome` is nontrivial, even though some reference context-decoding analyses operate on hit-only subsets.

Selection-debug note: current `select_context_sessions()` still uses only the presence of both autowater states, yielding 21 converted sessions across 9 used subjects. Next refinement should compute paper-style per-session correct DR/WC counts by lick direction and log them during selection.
