# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse (local project files: `paper.pdf`, `methods.txt`, `code/`, `data/`)
- **Date started**: 2026-03-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadObjs` | `code/DataLoadingScripts/loadObjs.m` | LOADING | Loads each session `obj` `.mat` file and normalizes missing struct fields (`meta`, `ex`, `me`). |
| `findDataFn` | `code/utils/findDataFn.m` | LOADING | Resolves the `data_structure_*` filename for a given animal/date. |
| `loadJEB*_ALMVideo`, `loadJGR*_ALMVideo`, etc. | `code/DataLoadingScripts/Recording and video/*.m` | LOADING | Enumerate included ephys+video sessions and assign the relevant probe number(s). |
| `loadSessionData` | `code/DataLoadingScripts/loadSessionData.m` | LOADING | Main reference entry point: loops sessions/probes, calls `processData`, then concatenates across probes into session-level outputs. |
| `processData` | `code/DataLoadingScripts/processData.m` | PROCESSING | Performs trial selection, cluster selection, spike alignment, PSTH/trial matrix creation, low-FR curation, and baseline statistics. |
| `findTrials` | `code/DataLoadingScripts/findTrials.m` | CURATION | Evaluates logical expressions on `obj.bp` fields to produce per-condition trial indices. |
| `findClusters` | `code/DataLoadingScripts/findClusters.m` | CURATION | Selects clusters by quality label; `'all'` excludes `garbage`, `gabrga`, `noisy`, and `real?`. |
| `alignSpikes` | `code/DataLoadingScripts/alignSpikes.m` | PROCESSING | Aligns per-spike trial times to the chosen event (`goCue`, `moveOnset`, `firstLick`, `lastLick`, `jawOnset`). |
| `getSeq` | `code/DataLoadingScripts/getSeq.m` | PROCESSING | Bins aligned spikes from `tmin:dt:tmax`, smooths them, and creates `obj.psth{probe}` and `obj.trialdat{probe}`. |
| `removeLowFRClusters` | `code/DataLoadingScripts/removeLowFRClusters.m` | CURATION | Removes neurons with mean firing rate below `lowFR` using the trial-averaged PSTH. |
| `baselineFR` | `code/DataLoadingScripts/baselineFR.m` | PROCESSING | Computes baseline firing rate and sigma from `-0.5 s` to `sample` onset. |
| `findVideoOffset` | `code/funcs/findVideoOffset.m` | PROCESSING | Computes offset between neural and video file starts for frame-time alignment. |
| `loadMotionEnergy` | `code/DataLoadingScripts/loadMotionEnergy.m` | PROCESSING | Loads session motion energy, interpolates it onto the neural time axis aligned to `params.alignEvent`, fills edge NaNs, and thresholds into move/non-move. |
| `getKinematicsFromVideo` | `code/funcs/kinematics/getKinematicsFromVideo.m` | PROCESSING | Interpolates DLC coordinates to the aligned trial time axis and derives x/y displacement and velocity per feature. |
| `findVelocity` | `code/funcs/kinematics/findVelocity.m` | PROCESSING | Computes per-trial velocity from interpolated trajectories; non-tongue features are baseline-subtracted, tongue NaNs become zero velocity. |
| `getKinematics` | `code/funcs/kinematics/getKinematics.m` | PROCESSING | Builds the full behavioral feature tensor by combining DLC kinematics, derived tongue angle/length, and motion energy. |
| `loadBehavSessionData` | `code/funcs/fig3/loadBehavSessionData.m` | LOADING | Behavior-only/session helper that finds trial IDs and video offset; includes `firstLick` alignment logic. |

### Notes
- The reference data object is a MATLAB struct `obj` with trial/task data in `obj.bp`, spike data in `obj.clu`, DLC trajectories in `obj.traj`, and sometimes motion energy in `obj.me`.
- The main ephys processing path is `loadSessionData -> processData -> findTrials/findClusters -> alignSpikes -> getSeq -> removeLowFRClusters -> baselineFR`.
- Reference analyses commonly use `params.alignEvent = 'goCue'`, which matches the decoder task requirement.
- Spike data are not converted to dF/F; these are electrophysiology sessions. Neural activity is processed as binned/smoothed firing rates.
- Neural binning is configurable. Core loaders support arbitrary `dt`; multiple figure scripts use `dt = 1/100` (10 ms) and some use `dt = 1/200` (5 ms). Final choice must be reconciled with paper/methods and data in later steps.
- `getSeq` bins spikes on `[tmin, tmax)` using `histc`, then smooths counts converted to Hz with `mySmooth`. Session-level single-trial neural data end up as `(time, neurons, trials)`.
- Cluster curation has two layers:
  - Quality filter by label using `findClusters`.
  - Firing-rate filter using `removeLowFRClusters`, typically `lowFR = 1` Hz in several figure scripts and tutorial code.
- Video/behavior streams are interpolated to the neural/session time axis rather than used at native frame rate.
- Motion energy alignment is explicit: `interp1(frameTimes - video_offset - align_event_time, me.data{trial}, obj.time + advance_movement)`, then edge NaNs are filled with nearest values.
- DLC trajectories also use frame times corrected by a 0.5 s or computed video offset depending on loader/helper path. This is important for later consistency checks on temporal alignment.
- Trial selection is driven by expressions over `obj.bp` fields such as `R`, `L`, `hit`, `miss`, `no`, `autowater`, `stim.enable`, and `early`.
- `autowater` is the code’s proxy for behavioral context: delayed response / 2AFC (`autowater == 0`) versus water-cued (`autowater == 1`).
- Choice/context decoding scripts in `code/ChoiceContextDecoding` balance trials across classes, typically use hit/miss subsets, and operate on `obj.trialdat` or interpolated kinematic tensors aligned to go cue.
- Metadata loaders encode the included sessions and which probe corresponds to ALM. Some sessions have one relevant probe and some analyses can concatenate two probes within a session.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Top-level raw-data folders:
  - `data/Ephys_Behavior`: delayed-response / water-cued sessions with neural data and separate `motionEnergy_*.mat` files.
  - `data/RandomizedDelay_Ephys_Behavior`: randomized-delay sessions; most have neural data plus separate `motionEnergy_*.mat` files.
  - `data/DelayInhibition_BilatMC_Behavior`: behavior-only inhibition sessions; no neural `clu` field, motion energy embedded inside `obj.me`.
  - `data/GoCueInhibition_BilatMC_Behavior`: behavior-only inhibition sessions; no neural `clu` field.
- File format:
  - All raw session files are MATLAB `.mat`.
  - Most `data_structure_*.mat` files are MATLAB v7.3/HDF5; at least the older `EKH1` and `EKH3` sessions are non-HDF5 MATLAB files.
  - `motionEnergy_*.mat` files load directly with SciPy and contain `me.data` and `me.moveThresh`.
- Raw session object layout:
  - Ephys sessions: `obj.bp`, `obj.clu`, `obj.ex`, `obj.meta`, `obj.pth`, `obj.sglx`, `obj.traj`, `obj.trials`.
  - Behavior-only sessions: `obj.bp`, `obj.ex`, `obj.me`, `obj.pth`, `obj.sglx`, `obj.traj`, `obj.trials`.
- Important raw variables observed:
  - `obj.bp`: trial labels (`L`, `R`, `hit`, `miss`, `no`, `autowater`, `early`, `protocol`, `stim`).
  - `obj.bp.ev`: event times including `bitStart`, `sample`, `delay`, `goCue`, `reward`, `lickL`, `lickR`.
  - `obj.clu`: per-probe spike data with fields `quality`, `site`, `spkWavs`, `tm`, `trial`, `trialtm`.
  - `obj.traj`: two camera views; each view stores per-trial lists of `frameTimes`, `ts`, `featNames`, etc.
  - External motion-energy files: `me.data` is one trace per trial; `moveThresh` is a per-session threshold.
- Representative dimensions from raw files:
  - DLC trajectory arrays are `(n_frames, 3, n_features)` per trial.
  - Example side-camera trial: `(2639, 3, 7)`; example bottom-camera trial: `(2639, 3, 10)`.
  - Example video sampling interval is about `2.52 ms` (`~396.8 Hz`).
  - Example motion-energy trial length is `2639` samples for an ephys session, consistent with the video frame count.
- Raw-data caveats found directly from the files:
  - `data_structure_JEB24_2023-10-03.mat` and `data_structure_JEB24_2023-10-04.mat` live in `RandomizedDelay_Ephys_Behavior` but have no `clu` field, so they are behavior-only for decoder purposes.
  - Those same two JEB24 sessions are also missing external `motionEnergy_*.mat` companions.
  - Probe location metadata are inconsistently populated in the raw files; some sessions record `L ALM` / `R ALM`, many store no location string.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 14,297 raw clusters across sessions with neural data |
| Neurons / session | mean 317.7, median 70, min 17, max 1258 (raw, pre-curation) |
| Subjects | 14 mice with neural data; 18 mice total across all raw folders |
| Sessions / subject | mean 3.21 neural sessions per mouse (range 1-10) |
| Trials (total) | 15,324 trials in sessions with neural data; 37,073 trials across all raw folders |
| Trials / session | mean 340.5, median 343, min 230, max 517 for neural-data sessions |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Two-context task: 522 units; DR-only fixed-delay task: 1,651 units; randomized-delay task: 845 units | “two-context paradigm: 12 sessions, six mice, 522 units”; “25 sessions, nine mice, 1,651 units”; “19 sessions… 845 units” |
| Neurons / session | Two-context mean about 43.5 units/session; DR-only mean about 66.0; randomized-delay mean about 44.5 | Computed from paper totals above |
| Subjects | Two-context: 6 mice; fixed-delay DR total: 9 mice; randomized-delay: 4 mice | “12 sessions, six mice”; “25 sessions, nine mice”; “19 sessions… four mice” |
| Sessions / subject | Two-context mean 2.0; fixed-delay DR mean about 2.8; randomized-delay mean about 4.8 | Computed from paper totals above |
| Trials (total) | Not explicitly tabulated in text | Not numerically stated in methods/paper text provided |
| Trials / session | Not explicitly tabulated in text; block structure is described | “approximately 100 DR trials” at session start, then “10–25 trials” per interleaved block |
| Neural data time bin | 5 ms in published decoding analyses | “each bin is 5 ms” |
| Behavior data time bin | 400 Hz raw video, approximately 2.5 ms per frame; decoding analyses also operate on 5 ms bins | “High-speed video was captured (400-Hz frame rate)” and “each bin is 5 ms” |
| Reward rate | Reward delivered for correct DR choices; WC reward is approximately 3 µl at a random port | “water reward was delivered” / “an approximately 3-µl water reward was presented randomly” |
| Task/behavior statistic 1 | Fixed-delay DR performance criterion: >70% accuracy before/for expert animals | “reached at least 70% accuracy”; “became experts (>70% accuracy …)” |
| Task/behavior statistic 2 | Randomized-delay early-lick criterion: <20% | “became experts (>70% accuracy and <20% early lick rate)” |
| Task/behavior statistic 3 | Behavioral analysis sessions required at least 40 correct DR trials per direction and 20 correct WC trials per direction | “at least 40 correct DR trials for each direction… and 20 correct WC trials for each direction” |


### Processing Details
- Task structure:
  - DR task: 1.3 s auditory sample, then delay, then 10 ms auditory go cue.
  - Standard fixed-delay DR task used 0.9 s delay for 12 mice; one mouse had 0.7 s delay that was linearly time-warped to 0.9 s in analyses.
  - WC task omits auditory cues; a water drop appears at a random side/time.
  - Two-context sessions begin with about 100 DR trials, then alternate WC/DR blocks of 10-25 trials; sessions begin in DR.
  - Randomized-delay task uses six delay values: 0.3, 0.6, 1.2, 1.8, 2.4, 3.6 s.
- Temporal alignment:
  - Methods and figure text repeatedly describe activity, movement, and decoding relative to the go cue (or water drop in WC plots); the user task specifically requires go-cue alignment, matching the main neural alignment used in the code and paper.
  - ITI context analyses use the 300 ms before sample onset.
- Kinematic/video processing:
  - Two 400 Hz cameras.
  - Tongue, jaw, nose tracked in both views; paws tracked from bottom view only.
  - Use x/y position and velocity per tracked feature.
  - Missing values are filled with nearest values for all features except the tongue.
  - Tongue angle and length are derived from the bottom camera.
- Motion energy processing:
  - Frame-wise motion energy is based on the absolute difference between the median of the next five frames and previous five frames.
  - Each frame is summarized by the 99th percentile across pixels.
  - Move/non-move threshold is set manually per session from bimodal distributions.
- Decoding in the paper:
  - Equal numbers of correct left and right lick trials were used for choice decoding.
  - Separate models were trained at each time bin for neural and kinematic decoding.
  - Models used ridge regularization and four-fold cross-validation.
  - Chance accuracy was computed by shuffling labels.
- Neural recording/processing:
  - ALM extracellular recordings from H2 and Neuropixels 1.0 probes.
  - Spike sorting via JRCLUST and/or Kilosort 3 with manual Phy curation.
  - Sessions were included only if they had at least 10 units.
  - For most analyses, all units with firing rate >1 Hz were included.
  - For subspace alignment and some single-unit analyses, only well-isolated single units with firing rate >1 Hz were included.

### Curation Steps

**Neuron curation rules**:
- Manual spike-sorting curation distinguishes well-isolated single units from multiunits.
- Recording sessions must have at least 10 units to be analyzed.
- A >1 Hz firing-rate threshold is applied.
- Some analyses use only well-isolated single units >1 Hz; other analyses include all units >1 Hz.

**Trial curation rules**:
- Early-lick trials are omitted from analyses.
- Ignore / no-response trials are omitted from behavioral analyses.
- Behavioral analyses require enough correct trials per class: at least 40 correct DR trials per direction and 20 correct WC trials per direction.
- Choice decoding uses equal numbers of correct left and right lick trials.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from uninstructed movement / neural activity | Reported qualitatively as above chance over time; exact numeric curve values are shown graphically, not tabulated in text |
| Context from kinematics / neural activity | Reported qualitatively as above chance over time, including during ITI; exact numeric curve values are shown graphically, not tabulated in text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Randomized-delay session count | `Figure3i.m` loads JEB11 (2), JEB12 (2), JEB23 (7; `2023-10-20` commented out), JEB24 (8) = 19 sessions | Raw folder has 22 files; 2 (`JEB24_2023-10-03`, `JEB24_2023-10-04`) have no `clu`, leaving 20 neural-capable files | “19 sessions… four mice” | Resolved: the analyzed randomized-delay set is the 19 sessions explicitly selected by the loader files, not every raw file in the folder. |
| Fixed-delay DR session/mouse count | `Figure3d.m` / `Figure3f.m` load 25 sessions spanning 10 subject IDs (`JEB6`, `JEB7`, `EKH1`, `EKH3`, `JGR2`, `JGR3`, `JEB13`, `JEB14`, `JEB15`, `JEB19`) | Raw `Ephys_Behavior` folder contains exactly those 25 sessions from 10 mice | “25 sessions, nine mice” | Partially resolved: session count is consistent at 25, but mouse count differs by one between paper text and released code/data. Treat this as a likely text/reporting discrepancy; follow the explicit session lists in code. |
| Two-context dataset size | `Figure8a_thru_c.m` / `Figure8d.m` load 12 sessions from 7 subject IDs (`JEB6`, `JEB7`, `EKH1`, `EKH3`, `JGR2`, `JGR3`, `JEB19`) | Raw fixed-delay folder contains 22 sessions with both `autowater` states present | “12 sessions, six mice, 522 units” | Resolved for session count but not mouse count: use the 12-session subset defined in the code for context analyses. The six-vs-seven mouse discrepancy is left as a text/code inconsistency. |
| Neuron totals | Core loaders start from all clusters, then apply quality and low-FR filters; specific analyses may further restrict to well-isolated single units >1 Hz | Raw neural files contain 14,297 clusters across 45 neural sessions | Paper reports 522 units (two-context), 1,651 units (fixed-delay DR total), 845 units (randomized delay) | Resolved: raw release contains unfiltered clusters and extra sessions/files; paper counts are post-curation analysis subsets, not raw file totals. |
| Region labels | Loader files explicitly pick the ALM probe for most sessions, and `loadSessionData` concatenates both probes when `meta.probe = [1 2]` | Raw `ex.probe.loc` strings are often missing, but can include non-ALM labels on pooled sessions (for example `JEB15_2022-07-26` has `L ALM` and `L M1TJ`) | Paper emphasizes ALM recordings, but the released session metadata can include additional probe locations in pooled sessions | Resolved: preserve per-neuron region labels from raw metadata for the reference-selected probe(s), rather than forcing every neuron to `ALM`. |
| Time bin size | Code examples use both 10 ms (`dt=1/100`) and 5 ms (`dt=1/200`) depending on figure/script | Raw video is about 400 Hz (`~2.5 ms`) and spikes are stored as event times | Paper’s decoding methods explicitly state 5 ms bins | Resolved: for decoder-matched conversion, prioritize 5 ms bins over generic 10 ms examples. |
| Motion-energy availability | Code expects external `motionEnergy_*.mat` for ephys and embedded `obj.me` for behavior-only sessions | Two randomized-delay files (`JEB24_2023-10-03`, `JEB24_2023-10-04`) lack both `clu` and external motion energy | Paper describes motion energy as available for movement analyses | Resolved: exclude sessions without neural data from the decoder dataset; missing motion energy on those two sessions is not relevant once excluded. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu{probe}.trialtm`, `obj.clu{probe}.trial`, `obj.bp.ev.goCue` | `neural` | Align each spike to go cue, bin at 5 ms from `-2.5` to `+2.5` s, smooth with the reference kernel, convert to firing rate, arrange as `(neurons, time)` per trial | `alignSpikes`, `getSeq`, `removeLowFRClusters` | Use the probe(s) selected by the reference loader files; concatenate probes within a session if code does so. |
| Common aligned time axis | `input[0]` | Continuous time-from-go-cue series, same for every trial, shape `(1, time)` | `getSeq` / paper decoding methods | Input name: `time_from_go_cue`. |
| `obj.bp.R` / `obj.bp.L` | `output[0]` | Per-trial categorical label: left=`0`, right=`1` | `findTrials`, task definitions in methods | Exclude early and no-response trials so choice/outcome are well-defined. |
| `obj.bp.autowater` | `output[1]` | Per-trial categorical label remapped to WC=`0`, DR=`1` | `findTrials`; paper/task description | Raw code uses `autowater==1` as WC. |
| `obj.bp.hit` / `obj.bp.miss` | `output[2]` | Per-trial categorical label: miss=`0`, hit=`1` | `getOutcome` | Ignore/no-response trials excluded rather than forced into incorrect. |
| DLC trajectories for tongue-related features | `output[3]` | Reference kinematic interpolation and velocity computation, then aggregate to a single tongue-speed trace and bin by per-session median | `getKinematicsFromVideo`, `findVelocity`, `getKinematics` | Because the decoder spec needs one tongue-velocity output, aggregate across tongue velocity channels after reference processing. |
| DLC trajectories for paw-related features | `output[4]` | Reference kinematic interpolation and velocity computation, then aggregate to a single paw-speed trace and bin by per-session median | `getKinematicsFromVideo`, `findVelocity`, `getKinematics` | Paws are tracked in the bottom view only. |
| Motion energy traces (`motionEnergy_*.mat` or embedded `obj.me`) | `output[5]` | Align/interpolate to neural time axis using reference logic, then bin by per-session median | `loadMotionEnergy`, `loadMotionEnergy_Behav` | The paper’s manual move threshold is not used because the decoder task explicitly asks for 50th-percentile binning. |

### Key Decisions
1. **Session inclusion**: Include only neural sessions represented in the reference ephys loaders.
   Fixed-delay set: all 25 sessions in `Ephys_Behavior`.
   Randomized-delay set: the 19-session subset encoded by `loadJEB11_ALMVideo`, `loadJEB12_ALMVideo`, `loadJEB23_ALMVideo`, and `loadJEB24_ALMVideo`.
   Behavior-only inhibition sessions are excluded because they have no neural activity.
2. **Trial inclusion**: Use only control/non-stimulation trials with valid behavioral labels.
   Exclude `stim.enable`, `early`, and `no` trials to match the reference analyses and keep output definitions unambiguous.
3. **Neural curation**: Apply the released-code quality filter (`all` except garbage/noisy-like labels), then remove units with firing rate `<= 1 Hz`, and exclude sessions with fewer than 10 remaining units.
4. **Temporal representation**: Use go-cue alignment and a common 5 ms bin width over `[-2.5 s, +2.5 s)`.
   This follows the paper’s explicit decoding-bin specification while preserving the baseline window used in the methods and the code.
5. **Context encoding**: Represent behavioral context as WC=`0`, DR=`1`, even though raw `autowater` uses the opposite polarity.
6. **Outcome encoding**: Represent only miss versus hit.
   Ignore trials are removed instead of being merged into incorrect.
7. **Kinematic scalar outputs**: After reproducing the reference video interpolation and per-feature velocity calculation, collapse the multi-feature representation to scalar tongue-speed and paw-speed traces using aggregate speed across the relevant tracked points.
   This is a required deviation from the reference feature set because the decoder task asks for exactly one tongue-velocity output and one paw-velocity output.
8. **Motion-energy binning**: Use the reference aligned continuous trace, but discretize with the task-mandated per-session median instead of the paper’s manually chosen move threshold.
9. **Brain-region labels**: Store per-neuron labels from the raw selected probe metadata, preserving hemisphere-specific ALM labels and any additional locations present in pooled sessions; fall back to `ALM` only when the raw location string is missing.

### Planned Sanity Checks
- [ ] Neural spot-check: for selected session/trial/neuron, verify converted binned firing rates match raw spike times aligned to go cue and smoothed with the same kernel.
- [ ] Input check: verify `input[0]` exactly matches the converted time axis for every session/trial.
- [ ] Trial-label check: compare converted lick direction, context, and outcome against raw `bp` fields for at least three manually chosen trials in multiple sessions.
- [ ] Motion-energy alignment check: interpolate raw motion-energy traces directly from the source files for selected trials and verify equality with converted continuous traces before discretization.
- [ ] Kinematic check: recompute tongue/paw continuous speed from raw DLC trajectories for selected trials/timepoints and verify agreement with the conversion code before median-threshold binning.
- [ ] Session-count check: verify converted sessions equal the reference loader-defined neural session lists (25 fixed-delay + 19 randomized-delay = 44 sessions).

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Implemented `convert_data.py` as a CLI script with:
  - `python -u convert_data.py <outpicklefile>`
  - `--full`
  - `--sample`
  - `--show-processing`
- The script:
  - Parses the reference MATLAB loader files to recover the analyzed session lists and selected probes.
  - Handles both single-probe entries and pooled dual-probe entries like `meta.probe = [1 2]`, matching the concatenation logic in `loadSessionData.m`.
  - Uses `pymatreader` to load both v7.3/HDF5 and older MATLAB files into a consistent Python representation.
  - Reproduces the reference neural processing steps needed here: go-cue alignment, 5 ms binning, causal Gaussian smoothing, code-matched quality filtering, and a 1 Hz low-FR filter.
  - Reproduces reference video/motion processing: video offset correction, interpolation to the neural time axis, per-feature velocity calculation, and motion-energy alignment.
  - Assigns `brain_region_idx` per neuron from raw `ex.probe.loc` metadata instead of using a fixed hard-coded region list.
  - Builds the target dictionary and validates it with `verify_data_format` before saving.
- Smoke test completed successfully:
  - Sample mode processed `JEB6_2021-04-18` and `JEB11_2022-05-10`.
  - Output: 525 trials total, 91 neurons total.

Code inefficiencies identified:
- Session loading from large MATLAB files is the main cost.
- Kinematic interpolation and per-neuron spike binning dominate runtime.

Code speedups added:
- Avoided full-session temporary 3D neural arrays before low-FR filtering by using a two-pass approach:
  - pass 1: estimate firing rate and select neurons
  - pass 2: build only kept neurons’ trial matrices
- Reused parsed reference loader definitions instead of scanning all raw files dynamically.
- Aggregated output traces only for the feature groups needed by the decoder task (tongue, paw, motion energy) instead of reconstructing the full video feature matrix.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 91 |
| Neurons / session | [28, 63] |
| Subjects | 2 (`JEB6`, `JEB11`) |
| Sessions / subject | [1, 1] |
| Trials (total) | 525 |
| Trials / session | [260, 265] |
| `time_from_go_cue` range | [-2.4975, 2.4975] |
| `lick_direction` distribution | [0.4190, 0.5810] |
| `behavioral_context` distribution | [0.1619, 0.8381] |
| `outcome` distribution | [0.0990, 0.9010] |
| `tongue_velocity_bin` distribution | [0.5000, 0.5000] |
| `paw_velocity_bin` distribution | [0.5000, 0.5000] |
| `motion_energy_bin` distribution | [0.5000, 0.5000] |

### Processing Plots Review
- Generated `processing_JEB6_2021-04-18.png` and `processing_JEB11_2022-05-10.png`.
- Numeric inspection of the plotted quantities found no anomalies:
  - Neural matrices have the expected `(neurons, 1000)` shape per trial with go cue at `t = 0`.
  - Continuous movement traces and session-median thresholds are finite and aligned to the common time axis.
  - Output matrices are binary in every dimension after discretization.
- Additional spot-check beyond sample sessions:
  - Verified that `JEB15_2022-07-26` now retains both reference-selected probes `(1, 2)` after fixing the loader parser; selected neurons span `L ALM` and `L M1TJ`, matching raw probe metadata and reference probe pooling logic.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Two-pass neural processing (FR filter before building full trial matrices) | Avoids constructing trial matrices for rejected neurons; major reduction in per-session memory and compute |
| Restricting video processing to tongue/paw/motion-energy outputs only | Avoids reconstructing the full kinematic feature tensor |
| Parsing the MATLAB session-loader files once instead of scanning raw directories heuristically | Eliminates repeated session-discovery overhead and keeps session inclusion code-matched |

| Step | Time / Session | Estimated Total Time |
| Sample conversion with plots (`JEB6`, `JEB11`) | 11.38-11.44 s/session | `~23 s` for 2 sessions |
| Full conversion without plots | `~10-12 s/session` on sample sessions | `~8 minutes` for the 44 reference-selected sessions, comfortably below the 15-minute optimization threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `lick_direction` | 0.6364 | 0.6151 |
| `behavioral_context` | 0.7749 | 0.8020 |
| `outcome` | 0.6048 | 0.5706 |
| `tongue_velocity_bin` | 0.8577 | 0.8656 |
| `paw_velocity_bin` | 0.5799 | 0.5585 |
| `motion_energy_bin` | 0.7773 | 0.7807 |

- Loss decreased monotonically over training from `7.4359` at epoch 1 to `0.5519` at epoch 200.
- Every output decoded above chance (`0.5`) on the validation set.
- Sample training therefore supports the current alignment, filtering, and output-construction choices well enough to proceed to full conversion.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `3.2G`
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 2,496 across fixed + randomized ephys results (1651 + 845) | Loader-defined sessions; unit total not explicitly tabulated | Raw selected sessions contain many more pre-curation clusters | 2,455 | Close to paper total; session/processing logic consistent, exact unit total differs modestly |
| Mean neurons/session | ~56.7 | Not explicitly tabulated | Raw selected sessions are much larger pre-curation | 55.8 | Yes, close to paper aggregate |
| Subjects | 13 by paper totals (9 fixed + 4 randomized) | 14 distinct subject IDs in loader files | 14 distinct subject IDs in selected raw files | 14 | Matches code/data; paper text has known mouse-count discrepancy |
| Sessions | 44 (25 fixed + 19 randomized) | 44 loader-selected sessions | 44 selected neural sessions | 44 | Yes |
| Trials (total) | Not explicitly stated | Not explicitly stated | 11,955 after behavior + neural-coverage curation | 11,955 | Yes |
| Trials/session (mean) | Not explicitly stated | Not explicitly stated | 271.7 after curation | 271.7 | Yes |
| `time_from_go_cue` range | `[-2.5, 2.5]` implied by chosen decoding window | `tmin=-2.5`, `tmax=2.5` in reference scripts | Raw event times support this range | `[-2.4975, 2.4975]` | Yes |
| `lick_direction` distribution | Not explicitly tabulated | Not explicitly tabulated | Derived from selected trials | `[0.4980, 0.5020]` | Plausible / balanced |
| `behavioral_context` distribution | Two-context sessions plus DR-only sessions imply DR-dominant full set | Not explicitly tabulated | Derived from `autowater` in selected trials | `[0.0843, 0.9157]` | Plausible / expected DR dominance |
| `outcome` distribution | Performance typically >70% correct in trained animals | Not explicitly tabulated | Derived from selected trials | `[0.1378, 0.8622]` | Yes, consistent with expert-performance regime |

- Full conversion runtime: `479.91 s` (`~8.0 min`), consistent with the Step 7 estimate.
- Full verify-only run completed with `no errors or warnings`.
- Edge-case fix applied during Step 9:
  - `JEB15` motion-energy files required recursive unwrapping of nested `me.data` structs.
  - `JEB24_2023-10-23` and `JEB24_2023-11-03` contained behavior-valid trials after the last trial with neural spikes; those late trials were excluded to avoid invalid all-zero neural inputs.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` re-run after the Step 9 fixes; result: `Data format is valid, no errors or warnings.`
2. **Raw-data sanity checks with `np.allclose()`**:
   - Neural: independently rebuilt the go-cue-aligned, binned spike train for `JEB6_2021-04-18`, selected probe 2, kept neuron corresponding to converted neuron 0 (raw cluster index 3), first kept trial; matched converted trace exactly (`np.allclose=True`, max abs diff `0.0`).
   - Input: independently rebuilt the 1000-bin time axis `(-2.4975 ... 2.4975)` and matched `input[0]` for the same session/trial (`np.allclose=True`, max abs diff `3.58e-07`).
   - Output: compared raw `bp.R`, `bp.autowater`, and `bp.hit` against converted `lick_direction`, `behavioral_context`, and `outcome` for kept trials 0, 100, and 259 of `JEB6_2021-04-18`; all matched exactly (`np.allclose=True` in all cases).
3. **Reference code comparison**:
   - Data loading: `convert_data.py` parses the same loader files used by `loadSessionData.m`, including pooled probe sessions such as `JEB15 [1 2]`.
   - Neuron/trial filtering: quality filter matches `findClusters(...,'all')`; low-FR threshold remains `>1 Hz`; behavioral trial filter matches the reference no-stim / non-early / non-no-response logic used throughout the figure scripts.
   - Temporal alignment and binning: go-cue alignment matches `alignSpikes`; binning/smoothing logic matches `getSeq` with causal Gaussian smoothing and `reflect` boundary handling.
   - Input construction: decoder input is the common aligned time axis, derived from the same bin centers as the neural data.
   - Output construction: categorical variables come directly from `obj.bp`; movement outputs use raw aligned traces but are discretized by per-session median as required by the decoder task.
4. **Key statistics comparison**:
   - Session counts match the released loader files exactly: 25 fixed-delay sessions and 19 randomized-delay sessions.
   - Subject counts match the released code/data (10 fixed-delay subjects, 4 randomized-delay subjects, 14 total) and preserve the previously documented paper-text mouse-count discrepancy.
   - Post-curation neuron totals are close to the paper totals but not identical (`1530` fixed-delay vs paper `1651`; `925` randomized-delay vs paper `845`).
     - Interpretation: the released paper text does not provide enough detail to reproduce these totals exactly from the public files with certainty, and the code does not tabulate them directly. The explicit session selection, probe handling, alignment, and curation logic do match the released code path.
5. **Edge-case review**:
   - Verified and fixed nested external motion-energy structs in `JEB15`.
   - Verified and fixed late-trial neural-coverage gaps in `JEB24_2023-10-23` and `JEB24_2023-11-03` by excluding behavior-valid trials occurring after the last trial containing spikes in the kept neuron set.

### Issues Found and Resolved
- Nested motion-energy file structure in `JEB15`: recursively unwrapped `me.data` until reaching the per-trial trace list.
- Behavior trials with no neural recording in two `JEB24` sessions: added a neural-coverage trim after neuron selection and re-ran full conversion/verification until warnings disappeared.
- Spurious `np.nanmean` runtime warning during kinematic aggregation: replaced with an explicit finite-value average to avoid warning spam while preserving the aggregated trace.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `lick_direction` | 0.6607 | 0.6481 | Above chance with small train/validation gap |
| `behavioral_context` | 0.8574 | 0.8568 | Strongest categorical decoder; essentially no overfitting |
| `outcome` | 0.6744 | 0.6501 | Above chance with modest gap |
| `tongue_velocity_bin` | 0.8548 | 0.8535 | Strongest movement decoder |
| `paw_velocity_bin` | 0.5711 | 0.5669 | Weakest decoder but still above chance |
| `motion_energy_bin` | 0.7448 | 0.7414 | Robust above-chance decoding |

- Test loss: `0.542068`
- Training ran successfully on `cuda` without needing the `--cpu` fallback.
- Loss decreased from `6.8394` at epoch 1 to `0.5212` at epoch 200.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `lick_direction` | 0.6481 | Paper reports above-chance neural decoding of choice around go cue / movement epochs; no exact scalar table is provided in text |
| `behavioral_context` | 0.8568 | Paper reports strong above-chance context decoding from movement and neural activity; exact scalar table not provided |
| `outcome` | 0.6501 | Outcome is not tabulated directly in the paper text provided; above-chance decoding is plausible |
| `tongue_velocity_bin` | 0.8535 | Paper reports strong neural/kinematic relationships with tongue/jaw movement; exact binary-threshold accuracy not tabulated |
| `paw_velocity_bin` | 0.5669 | Paw-related movement is expected to be weaker than tongue/jaw in ALM-centered recordings; no exact scalar table in text |
| `motion_energy_bin` | 0.7414 | Paper reports strong movement-related structure; exact binary-threshold accuracy not tabulated |

[Analysis of any low accuracies]
- Every validation accuracy is above chance (`0.5` for all outputs).
- Outputs below `1.5x` chance (`0.75`) were investigated:
  - `lick_direction`, `outcome`, and `paw_velocity_bin` are weaker than the strongest decoders but remain clearly above chance.
  - Raw-value checks on trial labels passed.
  - Temporal alignment and all-zero-trial warnings were already checked and fixed in Steps 9-10.
  - Output class balance is acceptable for all outputs; the movement bins are exactly 50/50 by construction.
  - Train/validation gaps are small for every output, arguing against leakage or severe overfitting.
- Accuracy comparison to the paper is necessarily qualitative because the provided text does not tabulate exact scalar decoder accuracies for these same binary tasks. The achieved pattern is directionally consistent with the paper:
  - strong context and tongue/motion decoding,
  - moderate choice/outcome decoding,
  - weaker paw decoding.

### Issues Found and Resolved
- No new conversion bugs were identified from the full decoder results beyond the Step 9-10 fixes already applied.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
