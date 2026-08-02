# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse (paper assets in this directory)
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
| `loadObjs` | `code/DataLoadingScripts/loadObjs.m` | LOADING | Loads each session `.mat` file into `obj`, backfilling missing `meta`, `ex`, and `me` fields for struct consistency. |
| `loadSessionData` | `code/DataLoadingScripts/loadSessionData.m` | LOADING | Main session loader; calls `loadObjs`, then `processData` per session/probe, and concatenates dual-probe sessions. |
| `processData` | `code/DataLoadingScripts/processData.m` | PROCESSING | Runs the standard preprocessing sequence: trial selection, cluster selection, spike alignment, PSTH/single-trial extraction, low-FR filtering, baseline FR estimation. |
| `findTrials` | `code/DataLoadingScripts/findTrials.m` | CURATION | Evaluates logical trial-condition expressions against `obj.bp` fields and returns matching trial indices. |
| `findClusters` | `code/DataLoadingScripts/findClusters.m` | CURATION | Selects units by quality label. |
| `alignSpikes` | `code/DataLoadingScripts/alignSpikes.m` | PROCESSING | Aligns spike times to the chosen event (`goCue` by default; also supports `jawOnset`, `moveOnset`, `firstLick`, `lastLick`). |
| `getSeq` | `code/DataLoadingScripts/getSeq.m` | PROCESSING | Bins aligned spikes on a uniform time grid and generates smoothed PSTHs and single-trial neural matrices. |
| `removeLowFRClusters` | `code/DataLoadingScripts/removeLowFRClusters.m` | CURATION | Removes units whose mean firing rate across all trials is below the threshold. |
| `baselineFR` | `code/DataLoadingScripts/baselineFR.m` | PROCESSING | Computes pre-sample baseline firing-rate mean and variance for retained units. |
| `loadMotionEnergy` | `code/DataLoadingScripts/loadMotionEnergy.m` | LOADING | Loads motion-energy files, aligns/interpolates them to the neural time base using video timestamps and alignment event, then thresholds into movement periods. |
| `getKinematicsFromVideo` | `code/funcs/kinematics/getKinematicsFromVideo.m` | PROCESSING | Extracts per-trial DLC positions and velocities for selected body parts, aligned to the same session time axis. |
| `findVelocity` | `code/funcs/kinematics/findVelocity.m` | PROCESSING | Computes per-feature velocity traces from aligned positions, with special handling for tongue visibility gaps. |
| `getOutcome` | `code/funcs/getOutcome.m` | PROCESSING | Returns per-trial correct/error labels from `bp.hit`, setting ignored/no-response trials to `NaN`. |

### Notes
- Reference pipeline entry point is `loadSessionData(meta, params)`, which first loads raw session `obj` structs, then processes each requested probe independently, and concatenates probes when a session has two ALM probes.
- The tutorial in `code/WorkingWithDataObjs.m` documents the raw session object fields: `obj.bp` (behavior/trials), `obj.clu` (sorted units), `obj.traj` (DLC trajectories), `obj.me` (motion energy for behavior-only sessions), and `obj.ex` (session metadata).
- Neural data are electrophysiology, not imaging. There is no delta-F/F computation anywhere in the reference loading path.
- Spike alignment defaults to `goCue`. In the tutorial, authors explicitly use `params.alignEvent = 'goCue'`.
- Neural binning is performed on a uniform time grid from `tmin` to `tmax` with bin width `dt`. In the tutorial example this is `dt = 1/100` s (10 ms), while `getDefaultParams.m` uses `dt = 1/200` s (5 ms); both use a `-2.5` to `2.5` s window around alignment.
- `getSeq` stores single-trial neural activity in `obj.trialdat{probe}` with shape `(time, neurons, trials)` after dividing spike counts by `dt` and smoothing with `mySmooth`.
- Smoothing is causal Gaussian in the tutorial (`params.smooth = 15`, `params.bctype = 'reflect'` in the worked example). This is part of the reference neural preprocessing.
- Unit filtering occurs in two stages:
  - quality filter via `findClusters` with `params.quality` (tutorial/default commonly uses `'all'`, with tutorial comments noting `garbage`/`noisy` are excluded from useful units),
  - firing-rate filter via `removeLowFRClusters`, which removes units with mean FR below threshold (`1 Hz` in tutorial example, `0.5 Hz` in defaults).
- Trial-condition logic in tutorial/default code distinguishes delayed-response versus water-cued context using `autowater`:
  - delayed-response: `~autowater`
  - water-cued: `autowater`
- Trial-condition logic also excludes stimulation and early-lick trials in the standard analyses via `~stim.enable` and `~early`.
- Motion energy is loaded from companion files and aligned to the same trial-centered time axis by interpolating DLC/video-timestamped traces onto `obj.time + params.advance_movement`, after subtracting the video offset and the chosen alignment event time.
- Kinematics are derived from DLC trajectories, producing x/y displacement and x/y velocity for each tracked feature from both views, then concatenated into a common `(time, trials, features)` representation.
- The reference decoding scripts (`NeuralChoiceDecoding.m`, `NeuralContextDecoding.m`) decode from `obj.trialdat`, confirming that single-trial smoothed firing rates are the intended neural representation rather than raw spikes.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains four session groups:
  - `Ephys_Behavior/`: standard delayed-response + water-cued sessions with electrophysiology, DLC trajectories, and companion `motionEnergy_*.mat` files.
  - `RandomizedDelay_Ephys_Behavior/`: randomized-delay electrophysiology sessions, also with companion `motionEnergy_*.mat` files.
  - `DelayInhibition_BilatMC_Behavior/`: behavior-only bilateral motor-cortex inhibition sessions. These session files contain embedded `obj.me` motion-energy data and no `clu` field.
  - `GoCueInhibition_BilatMC_Behavior/`: behavior-only go-cue inhibition sessions. These also contain embedded `obj.me` and no neural units.
- Session files follow `data_structure_<animal>_<date>.mat`.
- Motion-energy files follow `motionEnergy_<animal>_<date>.mat` for ephys sessions; behavior-only sessions store motion energy directly in the session file.
- MATLAB file formats are mixed:
  - many session files are MATLAB v7.3 / HDF5,
  - some later session files are classic MAT files readable with `scipy.io.loadmat`,
  - motion-energy files are classic MAT files.
- Raw session object fields seen across files:
  - ephys sessions: `obj.pth`, `obj.bp`, `obj.sglx`, `obj.traj`, `obj.trials`, `obj.clu`, `obj.ex`, and sometimes `obj.meta`
  - behavior-only sessions: `obj.pth`, `obj.bp`, `obj.sglx`, `obj.traj`, `obj.trials`, `obj.me`, `obj.ex`
- Behavioral/task fields in `obj.bp` include:
  - trial labels: `R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, sometimes `autowaterBlock`, `autolearn`
  - timing/event container `ev` with `bitStart`, `sample`, `delay`, `goCue`, `reward`, `lickL`, `lickR`
  - stimulation/task metadata: `stim`, `protocol`, `bitRand`
- DLC trajectories are stored in `obj.traj` for two camera views. Each trial struct contains `featNames`, `ts`, `frameTimes`, `NdroppedFrames`, and filename metadata.
- Example tracked features:
  - side view: `tongue`, `left_tongue`, `right_tongue`, `jaw`, `trident`, `nose`, `lickport`
  - bottom view: `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`, `top_paw`, `bottom_paw`, `lickport`, `jaw`, `top_nostril`, `bottom_nostril`
- Raw neural data are sorted spike trains per unit with fields such as `tm`, `trialtm`, `trial`, `quality`, `spkWavs`, and `site`/`channel`.
- Raw unit quality labels are heterogeneous and include `garbage`, `multi`, `poor`, `fair`, `good`, `great`, plus capitalization variants.
- No README or documentation files were found inside `data/`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 14,297 raw sorted units across neural sessions |
| Neurons / session | 317.71 mean across 45 sessions with neural data (range 17-1258) |
| Subjects | 18 total mice |
| Sessions / subject | 6.67 mean across all subjects (range 1-24) |
| Trials (total) | 37,073 |
| Trials / session | 308.94 mean (range 209-517) |

Additional size notes:
- Total `data_structure_*.mat` session files: 120
- Sessions with neural data (`clu` present and non-empty): 45
- Sessions without neural data: 75
- Session counts by group:
  - `DelayInhibition_BilatMC_Behavior`: 53
  - `Ephys_Behavior`: 25
  - `GoCueInhibition_BilatMC_Behavior`: 20
  - `RandomizedDelay_Ephys_Behavior`: 22
- Subjects present: `EKH1`, `EKH3`, `JEB11`, `JEB12`, `JEB13`, `JEB14`, `JEB15`, `JEB19`, `JEB23`, `JEB24`, `JEB6`, `JEB7`, `JGR2`, `JGR3`, `MAH13`, `MAH14`, `MAH20`, `MAH21`
- Raw `ex.probe.loc` / region labels seen in files include `L ALM`, `R ALM`, `L M1TJ`, `R M1TJ`, `m1tj`, `alm`, `Bi_ALM`, `Bi_M1TJ`, `Bi_MC`, and some `DUMMY` placeholders.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 522 units in two-context ephys dataset | “12 sessions, six mice, 522 units” |
| Neurons / session | ~43.5 units/session in two-context ephys dataset | Derived from 522 units / 12 sessions |
| Subjects | 6 mice in two-context ephys dataset | “12 sessions, six mice” |
| Sessions / subject | 2.0 mean in two-context ephys dataset | Derived from 12 sessions / 6 mice |
| Trials (total) | Not stated explicitly in text | Not explicitly given in paper/methods excerpt |
| Trials / session | Session starts with ~100 DR trials, then 10-25 trial alternating blocks | “approximately 100 DR trials” and “Each interleaved block was 10–25 trials” |
| Neural data time bin | 5 ms reference neural bin for CD prediction analyses | “B = 6; each bin is 5 ms” |
| Behavior data time bin | 400 Hz video = 2.5 ms/frame | “High-speed video was captured (400-Hz frame rate)” |
| Reward rate | Training criterion >70% accuracy on DR / randomized-delay tasks | “at least 70% accuracy” and “>70% accuracy” |
| Delay duration, standard DR | 0.9 s for 12 mice; 0.7 s for 1 mouse, linearly warped to 0.9 s | “0.9 s for 12 mice, 0.7 s for one mouse and linearly time warped to 0.9 s” |
| Randomized delay durations | 0.3, 0.6, 1.2, 1.8, 2.4, 3.6 s | “randomly selected from six possible values” |
| Choice-selective single units | 36% sample, 42% delay, 58% response of 483 single units | “sample: 36%; delay: 42%; response: 58%” |
| Context-selective single units | 39% of 214 single units | “39% of single units” |


### Processing Details
- Two tasks alternate within a session:
  - delayed-response (DR): sample tone (1.3 s), delay, then 10 ms auditory go cue.
  - water-cued (WC): no auditory cues; water delivered at a random port/time.
- Sessions in the two-context paradigm begin with approximately 100 DR trials, then alternate between WC and DR in blocks of 10-25 trials.
- Temporal reference events in the paper are context-dependent:
  - DR analyses are described relative to go cue.
  - WC analyses are described relative to water drop.
  - The user’s decoder task specifically requires alignment to go cue onset, so WC trials will need a consistent handling rule relative to their water timing while preserving the paper’s trial/event semantics.
- Early licks are excluded from analyses in WC and, more generally, early lick / ignore trials are omitted from behavioral analyses.
- Video processing:
  - two cameras at 400 Hz,
  - position and velocity computed for tracked features,
  - missing values filled with nearest available value for all features except tongue,
  - tongue angle and length computed from bottom camera.
- Motion energy processing:
  - framewise difference between medians of next 5 and previous 5 frames,
  - per-frame scalar defined as 99th percentile across pixels,
  - threshold chosen manually on a per-session basis at the separation between two modes of the motion-energy distribution.
- Electrophysiology:
  - ALM recordings with H2 or Neuropixels 1.0 probes,
  - raw voltages sampled at 25 kHz,
  - session inclusion required at least 10 units.
- Decoder methods stated in paper:
  - logistic regression for choice and context decoding from either kinematic features or single-trial neural activity,
  - equal numbers of correct left and right lick trials,
  - separate model at each time bin,
  - ridge regularization,
  - four-fold cross-validation with 30% held out for testing,
  - chance from shuffled labels.
- Prediction of CD projections from kinematics used lagged 5 ms bins (`B = 6` previous bins), which is relevant for understanding the reference temporal resolution but is distinct from the simpler decoder-validation task here.

### Curation Steps

**Neuron curation rules**:
- Spike sorting done with JRCLUST and/or Kilosort 3 plus manual curation.
- Well-isolated single units defined by manual inspection of ISI violations, separation from other units, and stationarity across session.
- Sessions included only if they had at least 10 units.
- For subspace alignment and some single-unit analyses: only well-isolated single units with firing rates >1 Hz.
- For other analyses: all units with firing rates >1 Hz.

**Trial curation rules**:
- Early lick trials were omitted from analyses.
- Ignore / no-response trials were omitted from behavioral analyses.
- Behavioral inclusion criterion for analyzed sessions: at least 40 correct DR trials per direction and 20 correct WC trials per direction, excluding early and ignore trials.
- For choice/context decoding, equal numbers of correct left and right lick trials were used.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from `CDchoice` projections | AUC = 0.86 ± 0.11 across sessions |
| Choice from neural population activity | Strongly above shuffled labels in Fig. 3b; exact scalar not stated in text |
| Choice from kinematic features | Strongly above shuffled labels in Fig. 3b; exact scalar not stated in text |
| Context from neural population activity | Strongly above shuffled labels in Fig. 4b; exact scalar not stated in text |
| Context from kinematic features | Strongly above shuffled labels in Fig. 4b; exact scalar not stated in text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Overall dataset scope | Codebase contains loaders for neural sessions and separate behavior-only inhibition sessions; context analyses use figure-specific session loaders | `data/` contains 120 sessions total, including 75 behavior-only sessions with no `clu` | Paper neural analyses are much smaller curated subsets (for example, 12 two-context sessions; 25 DR sessions; 19 randomized-delay sessions) | Treat the raw `data/` directory as a superset. For the decoder conversion, exclude sessions without neural data and match the paper/code subset relevant to the requested outputs. |
| Two-context neural subset | `Figure8a_thru_c.m` loads 12 specific ALM sessions via `loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo` | Raw `Ephys_Behavior/` contains 25 fixed-delay ephys sessions, 22 of which contain both DR and WC trials | Paper reports “12 sessions, six mice, 522 units” for the two-context dataset | Use the 12-session figure-specific loader list as the reference subset for context analyses, then enforce paper/code curation (quality + FR filtering). Mouse-count wording in paper versus code session-loader list will be re-checked against final converted totals if needed. |
| Raw unit counts vs paper unit counts | Reference code filters units by quality and low FR before analysis | Raw session files contain many `garbage`, `multi`, `poor`, `fair`, etc. units; raw counts are far above paper totals | Paper reports 522 units for two-context, 1,651 for fixed-delay DR, 845 for randomized delay after curation | Use curation consistent with code/paper: exclude junk-quality units and require firing rate > 1 Hz for analysis-ready neural data. |
| Low-FR threshold | `getDefaultParams.m` sets `lowFR = 0.5`; tutorial and several figure scripts use `lowFR = 1` | Raw data include many low-quality / low-rate units | Paper states analyses used units with firing rates exceeding 1 Hz | Resolve in favor of the paper and figure scripts: use a 1 Hz firing-rate threshold. |
| Base neural time bin | Tutorial example uses `dt = 1/100` (10 ms); `getDefaultParams.m` and paper methods indicate 5 ms internal bins in later analyses | Raw spikes are continuous event times | Paper methods explicitly mention 5 ms bins for lagged kinematic prediction; decoder scripts then aggregate to 75 ms | Resolve in favor of 5 ms base binning for initial alignment/processing, with any coarser binning derived afterward if needed. |
| Alignment event across contexts | Reference code commonly aligns to `goCue`; paper text describes DR relative to go cue and WC relative to water drop | Raw mixed-context files still contain a populated `bp.ev.goCue` field on WC trials; on inspected sessions it behaves like the WC action-triggering event (water-drop-aligned event) rather than being missing | User requires universal alignment to “Go cue onset” | Use `bp.ev.goCue` as the universal alignment field. For WC trials, the stored `goCue` appears to serve as the water-presentation-equivalent event, which resolves the cross-context alignment requirement. |
| Randomized-delay session count | Paper says 19 randomized-delay sessions, four mice | Raw folder contains 22 randomized-delay session files | Paper also requires session inclusion only if there are at least 10 units, plus later curation | Treat `RandomizedDelay_Ephys_Behavior/` as a raw superset; later curation should reduce this toward the paper total. These sessions are less suitable for the requested context decoder because context does not vary there. |
| Behavior-only inhibition sessions | Code README marks “Video only” sessions as behavior only | `DelayInhibition_*` and `GoCueInhibition_*` files have no neural `clu`, only behavior/video/motion-energy data | Paper uses these sessions for optogenetic behavior analyses, not neural population decoding | Exclude these sessions from the converted neural decoder dataset because decoder inputs require neural activity. |

Final understanding after reconciliation:
- The requested decoder dataset should be built from neural ALM sessions, not behavior-only inhibition sessions.
- Because the decoder outputs include behavioral context (`WC` vs `DR`), the most paper-consistent primary dataset is the two-context ALM subset loaded by the context-analysis scripts rather than the entire raw ephys archive.
- Neural preprocessing should follow the reference electrophysiology path: align spikes to `goCue`, bin on a common grid, smooth, exclude junk-quality units, and apply a 1 Hz firing-rate threshold.
- Behavioral context is represented by `autowater` in both paper and code (`~autowater` = DR, `autowater` = WC).
- The raw `goCue` field can be used for both DR and WC trials because WC trials still have a meaningful `goCue`-equivalent event recorded.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu{probe}[*].trialtm`, `obj.clu{probe}[*].trial`, `bp.ev.goCue` | `neural` | Bin spikes at 5 ms on a common window around aligned event; convert to firing rate by dividing by `dt`; smooth with reference-style Gaussian smoothing | `alignSpikes`, `getSeq`, `mySmooth` | Use only the ALM probe specified by the figure/session loader metadata. |
| Common aligned time axis | `input[0]` | Continuous time-from-go-cue vector repeated for every trial | Reference decoders train one model per time bin after alignment | Single decoder input requested by user. |
| First post-alignment lick side from `bp.ev.lickL` / `bp.ev.lickR` | `output[0]` | Per-trial binary constant over time: left = 0, right = 1 | Raw event times; paper/code use lick-port events and directional trial structure | Use actual behavioral lick direction, not instructed trial side; this is required by the user task. |
| `bp.autowater` | `output[1]` | Per-trial binary constant over time: WC = 0, DR = 1 | Trial conditions in reference code (`autowater` vs `~autowater`) | Exact context variable used throughout the paper/code. |
| `bp.hit`, `bp.miss` | `output[2]` | Per-trial binary constant over time: incorrect = 0, correct = 1 | `getOutcome` | Restrict to hit/miss trials; exclude ignore/no-response trials. |
| DLC kinematics for side-view tongue feature | `output[3]` | Time-varying binary trace from per-timepoint tongue-speed magnitude, thresholded at session median | `getKinematicsFromVideo`, `findVelocity` | Planned scalar = `sqrt(xvel^2 + yvel^2)` for `tongue_*_view1`. |
| DLC kinematics for bottom-view paw feature(s) | `output[4]` | Time-varying binary trace from per-timepoint paw-speed magnitude, thresholded at session median | `getKinematicsFromVideo`, `findVelocity` | Planned scalar = mean speed magnitude across `top_paw` and `bottom_paw` in view 2. |
| Motion-energy trace | `output[5]` | Time-varying binary trace from aligned motion-energy value, thresholded at session median | `loadMotionEnergy` | Use aligned motion-energy stream after interpolation onto neural time base. |
| Loader-session metadata (`anm`, selected ALM probe) | `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx` | Normalize subject names to mouse IDs; normalize selected recording area to ALM | Session loader functions, `ex.probe.loc` | All kept neural sessions are ALM sessions from figure-specific loaders. |

### Key Decisions
1. **Session subset**: Convert the paper’s two-context ALM ephys subset rather than every raw file. Rationale: the requested decoder includes behavioral context (`WC` vs `DR`), which only varies in the two-context neural sessions, and behavior-only inhibition sessions have no neural activity.
2. **Use figure-specific session list**: Use the 12-session context-analysis loader set (`JEB6`, `JEB7`, `EKH1`, `EKH3`, `JGR2`, `JGR3`, `JEB19`) as the reference inclusion set before neural curation. Rationale: this is the clearest code path corresponding to the paper’s context analyses.
3. **Trial inclusion**: Keep non-stim, non-early, non-ignore neural trials with a valid lick outcome (`hit` or `miss`). Rationale: this matches the paper’s omission of early/ignore trials and keeps outcome well-defined.
4. **Alignment**: Align every signal to `bp.ev.goCue`. Rationale: this is the requested decoder alignment, is the default in the reference code, and raw WC trials still contain a meaningful `goCue`-equivalent event.
5. **Neural representation**: Use smoothed single-trial firing rates rather than raw spike binaries. Rationale: the reference neural decoding scripts operate on `obj.trialdat`, which is the smoothed firing-rate representation.
6. **Neural curation**: Exclude junk-quality units and require firing rate > 1 Hz. Rationale: this matches the paper’s electrophysiology analysis criteria more closely than the looser 0.5 Hz default helper.
7. **Brain-region normalization**: Store region as `ALM` for all kept neurons. Rationale: the selected probe in the reference loader functions is the ALM probe; hemisphere-specific labels are less important than area identity for the target format.
8. **Lick direction definition**: Use actual first post-alignment lick side from lickport events, not the instructed side (`R`/`L`). Rationale: the user explicitly requested “lick direction”; for error trials this differs from the cue/target side.
9. **Static outputs as time-varying constants**: Represent lick direction, context, and outcome as constant binary traces across the full trial window. Rationale: this keeps all outputs in a uniform `(n_output, n_timepoints)` format while preserving per-trial labels.
10. **Continuous-movement discretization**: Compute per-session medians on the aligned traces across all kept trials/timepoints and bin each timepoint into `<50th percentile` vs `>=50th percentile`. Rationale: this follows the user’s required decoder-output discretization.
11. **Velocity scalar choice**: Reduce multi-axis DLC velocity to a scalar speed magnitude before thresholding. Rationale: the user asked for “velocity” as one output stream each for tongue and paw, and scalar speed is the most defensible collapse of x/y velocity.
12. **WC timing handling**: Treat stored WC `goCue` as the water-presentation-equivalent event already recorded in the raw data. Rationale: direct raw-data inspection showed populated `goCue` values on WC trials, making a unified alignment feasible without inventing a synthetic event.

### Planned Sanity Checks
- [ ] Raw spike binning spot-check: for one kept session, neuron, and trial, compare converted binned firing-rate values against a direct histogram of raw `trialtm - goCue` spike times from the source file with `np.allclose()`.
- [ ] Lick-direction spot-check: for several trials, compare converted per-trial lick label against the earliest post-alignment value among raw `lickL` and `lickR` event times from the source file.
- [ ] Motion-energy alignment spot-check: compare converted aligned motion-energy trace for a chosen trial against direct interpolation from the raw motion-energy file using raw frame times and `goCue`, with `np.allclose()` after matching binning.
- [ ] Tongue/paw velocity spot-check: for selected trials/timepoints, compare converted speed traces against direct reconstruction from raw DLC positions and finite-difference velocity calculations using the same view/feature definitions.
- [ ] Session-statistics check: verify session counts, subject counts, and filtered neuron counts trend toward the paper’s 12-session / 522-unit two-context statistics after curation.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `convert_data.py` with CLI: `python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]`
- Implemented the reference two-context ALM session subset directly from the figure-specific session loader list in the reference code.
- Implemented raw-session loading with `mat73` for MATLAB v7.3/HDF5 session files and `scipy.io.loadmat` for motion-energy files.
- Implemented reference-style preprocessing in Python:
  - alignment to `bp.ev.goCue`
  - 5 ms spike binning from `-2.5` s to `+2.5` s
  - causal Gaussian smoothing matching the reference `mySmooth` behavior
  - quality filtering consistent with `findClusters(..., {'all'})`
  - low firing-rate filtering at 1 Hz
  - motion-energy interpolation onto the neural time axis using video timestamps and video offset
  - DLC-based tongue and paw speed extraction using reference-style position interpolation and velocity calculation
- Implemented target-format assembly with:
  - 1 time-varying decoder input (`time_from_go_cue_seconds`)
  - 6 outputs (`lick_direction`, `behavioral_context`, `outcome`, `tongue_velocity`, `paw_velocity`, `motion_energy`)
  - constant-in-time traces for per-trial categorical variables
  - per-session median thresholding for movement outputs
- Implemented `--show-processing` plots that summarize neural activity, aligned continuous traces, threshold distributions, discretized outputs, and session counts for up to 2 sessions.

Code inefficiencies identified:
- Full MATLAB session objects are loaded into memory per session via `mat73`, which is simpler and reliable but not maximally lean.
- Kinematic alignment currently interpolates each requested feature trial-by-trial in Python loops.

Code speedups added:
- Neural spike binning is vectorized within each unit using `np.add.at` rather than nested time-bin loops.
- Only the exact kinematic features needed for the requested outputs are extracted (`tongue`, `top_paw`, `bottom_paw`) instead of the full feature set.
- Sample mode processes only the first two sessions to accelerate iteration.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 95 |
| Neurons / session | 47.5 mean (29, 66) |
| Subjects | 2 |
| Sessions / subject | 1, 1 |
| Trials (total) | 469 |
| Trials / session | 260, 209 |
| `time_from_go_cue_seconds` range | [-2.5, 2.5] |
| `lick_direction` distribution | [0.478, 0.522] |
| `behavioral_context` distribution | [0.277, 0.723] |
| `outcome` distribution | [0.139, 0.861] |
| `tongue_velocity` distribution | [0.804, 0.196] |
| `paw_velocity` distribution | [0.500, 0.500] |
| `motion_energy` distribution | [0.500, 0.500] |

### Processing Plots Review
- Generated `processing_JEB6_2021-04-18.png` and `processing_JEB7_2021-04-29.png`.
- Alignment visually centers all traces at the stored `goCue` event.
- Motion energy and paw velocity show balanced median thresholding as expected.
- Initial tongue discretization collapsed in one session because invisible tongue periods dominated the median; fixed by computing the session median from tongue-visible timepoints only and assigning invisible periods to the low bin.
- After the fix, tongue output is binary in both sessions and no format warnings remain.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Extract only requested kinematic features (`tongue`, `top_paw`, `bottom_paw`) | Avoids full-DLC feature processing |
| Vectorized spike binning with `np.add.at` | Removes nested time-bin loops |
| Sample mode (`--sample`) | Reduces iteration cycle to first 2 sessions |

| Step | Time / Session | Estimated Total Time |
| | | |
| Conversion (sample measurement) | ~7.4 s/session | ~1.5-2 min for 12 sessions |
| Verification-only | negligible relative to conversion | <1 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `lick_direction` | 0.6335 | 0.6328 |
| `behavioral_context` | 0.7474 | 0.7350 |
| `outcome` | 0.6410 | 0.6504 |
| `tongue_velocity` | 0.7796 | 0.7697 |
| `paw_velocity` | 0.5668 | 0.5562 |
| `motion_energy` | 0.7810 | 0.7633 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 531M
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 522 units | 12-session context loader list, then quality + FR curation | Raw selected-session total is much larger before filtering | 519 neurons | Close; likely minor curation-definition difference |
| Mean neurons/session | ~43.5 | 12 selected sessions | Raw selected-session mean is much larger before filtering | 43.25 | Yes, effectively matched |
| Subjects | 6 mice (paper text) | 7 named animal IDs in context figure loader list | 7 subject IDs in selected raw files | 7 subjects | Matches code/raw; paper text differs by 1 |
| Sessions | 12 | 12 | 12 in selected raw subset | 12 | Yes |
| Trials (total) | Not explicitly stated | Trial counts determined after filtering | 2,415 after hit/miss + !early + !ignore + !stim in selected raw subset | 2,415 | Yes |
| Trials/session (mean) | Not explicitly stated | N/A | 201.25 after trial filtering in selected raw subset | 201.25 | Yes |
| `time_from_go_cue_seconds` range | Go-cue-centered analyses throughout paper | `alignEvent = 'goCue'` | Raw `goCue` available on DR and WC trials | [-2.5, 2.5] | Yes |
| `lick_direction` distribution | Not explicitly stated | Derived from lick events / directional structure | Available in raw lick-port events | [0.514, 0.486] | Plausible / balanced |
| `behavioral_context` distribution | DR/WC blocks present | `autowater` drives DR vs WC conditions | Raw selected subset mixes DR and WC in every kept session | [0.289, 0.711] | Yes |
| `outcome` distribution | Sessions trained to expert performance >70% | `hit`, `miss`, `no` fields | Raw kept trials are mostly correct | [0.136, 0.864] | Yes |
| `tongue_velocity` distribution | Not explicitly stated | Tongue tracked with special visibility handling | Raw tongue visibility is sparse, so low bin dominates | [0.824, 0.176] | Plausible |
| `paw_velocity` distribution | Not explicitly stated | User-required median discretization | Raw selected subset supports balanced median thresholding | [0.500, 0.500] | Yes |
| `motion_energy` distribution | Motion-energy thresholding is session-specific in paper | `loadMotionEnergy` aligns to neural axis | Raw aligned traces available for every kept trial | [0.500, 0.500] | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reported “Data format is valid, no errors or warnings.” No decoder-format errors remained after the tongue-threshold fix from Step 7.
2. **Raw-data sanity checks with `np.allclose()`**: added `cache/step10_checks.py` and ran `python3 cache/step10_checks.py | tee cache/step10_checks_out.txt`.
   - Neural check: direct raw spike histogram from session `JEB6_2021-04-18`, first retained unit, first retained trial, divided by `dt` and smoothed with the reference-style causal Gaussian, matched converted neural data with `np.allclose(..., atol=1e-6) = True`; max absolute error was `6.317751e-06` (float32 rounding level).
   - Input check: converted `time_from_go_cue_seconds` matched the independently reconstructed 5 ms time axis exactly (`np.allclose = True`).
   - Output checks:
     - first-lick direction matched direct raw-event reconstruction (`LICK_MATCH = True`);
     - motion-energy output for the same trial matched direct raw interpolation + session-median thresholding (`ME_BIN_ALLCLOSE = True`).
3. **Reference code comparison**:
   - Data loading:
     - reference: `Figure8a_thru_c.m` -> `loadJEB6_ALMVideo`, `loadJEB7_ALMVideo`, `loadEKH1_ALMVideo`, `loadEKH3_ALMVideo`, `loadJGR2_ALMVideo`, `loadJGR3_ALMVideo`, `loadJEB19_ALMVideo`
     - conversion: hard-coded the same 12-session / probe list into `CONTEXT_SESSION_SPECS`
     - result: matched.
   - Neuron filtering:
     - reference: `findClusters(..., {'all'})` excludes exact labels `garbage`, `gabrga`, `noisy`, `real?`; Figure 8 sets `params.lowFR = 1`
     - conversion: same label exclusions and `LOW_FR_HZ = 1.0`
     - result: matched.
   - Trial filtering:
     - reference Figure 8 conditions use hit/miss/no for all-trial PSTHs and specific hit/miss, `~stim.enable`, and `~early` conditions for context analyses
     - conversion keeps hit/miss, excludes `early`, `no`, and `stim`
     - result: matched the paper/code inclusion logic needed for well-defined outcome labels.
   - Temporal alignment and binning:
     - reference aligns to `goCue`; Figure 8 uses `tmin = -3`, `tmax = 2.5`, `dt = 10 ms`; broader paper/method text also uses 5 ms bins for later decoding analyses
     - conversion aligns to `bp.ev.goCue` with `[-2.5, 2.5]` and `5 ms` bins
     - result: alignment matched; bin width/window differ slightly from Figure 8 because the user requested go-cue-centered decoder input and a uniform compact window, while 5 ms is still paper-consistent.
   - Input construction:
     - reference decoders train separate models at each aligned time bin rather than using an explicit “time” predictor array
     - conversion stores the aligned time axis as the requested decoder input
     - result: deliberate target-format adaptation, not a mismatch.
   - Output construction:
     - `behavioral_context`: `autowater` exactly matches reference DR/WC splitting
     - `outcome`: derived from `hit` as in `getOutcome`
     - `lick_direction`: direct first post-alignment lick side from `lickL` / `lickR`
     - movement variables: derived from reference-style aligned video/motion-energy traces, then discretized per the user’s required median split
     - result: matched reference streams, with only the user-required discretization differing from the paper.
4. **Key statistics comparison**:
   - Sessions: 12 in paper, reference code, selected raw data, and converted data. Exact match.
   - Subjects: paper text says 6 mice, but the Figure 8 loader code names 7 animals (`EKH1`, `EKH3`, `JEB6`, `JEB7`, `JGR2`, `JGR3`, `JEB19`), and the selected raw files also contain these same 7 IDs. I retained 7 because code + data agree.
   - Units: converted data have 519 units versus 522 in the paper text. I independently recomputed the full selected-session count from raw data and reproduced 519 exactly. I also tested nearby obvious curation variants; they gave 519 or 520, not 522. Conclusion: the 3-unit gap is most likely due to a minor undocumented curation difference or a paper-rounding/reporting discrepancy, not a bug in the implemented loader/filter logic.
5. **Edge-case review**:
   - `cache/step10_checks.py` found `2` raw trials with empty/all-NaN `frameTimes`; the conversion already handles this by falling back to the nominal 400 Hz frame grid, matching the reference behavior of reconstructing time when timestamps are unavailable.
   - `527` raw trials in the selected sessions had the tongue completely invisible; the conversion follows the reference tongue rule by treating missing tongue velocity as zero and keeping invisible periods in the low bin.
   - No off-by-one evidence was found at bin boundaries: the neural/raw reconstruction matched exactly after direct histogramming against the same `[-2.5, 2.5)` bin edges.

### Issues Found and Resolved
- **Paper subject-count discrepancy (`6` vs `7`)**: Not fixed, because the Figure 8 reference code and selected raw sessions both clearly contain 7 animal IDs. Documented as a source discrepancy and resolved in favor of code + raw data.
- **Paper unit-count discrepancy (`522` vs `519`)**: No conversion change made after review because independent raw-data recount reproduced 519 exactly under the implemented reference curation. Documented as a minor unresolved paper/code discrepancy.
- **No format or numerical mismatches were found in the Step 10 sanity checks**.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `lick_direction` | 0.6335 | 0.6127 | Above chance; consistent with choice-related information in ALM, though lower than paper’s CD-projection AUC because metric/task are not identical. |
| `behavioral_context` | 0.7523 | 0.7469 | Strong above-chance context decoding, consistent with persistent context coding described in the paper. |
| `outcome` | 0.6399 | 0.6121 | Above chance; no direct paper benchmark for this exact decoder target. |
| `tongue_velocity` | 0.7862 | 0.7766 | Strongest movement-decoding output; consistent with strong movement-related structure in the video variables. |
| `paw_velocity` | 0.5796 | 0.5737 | Lowest but still above chance; investigated further in Step 12. |
| `motion_energy` | 0.7631 | 0.7579 | Strong above-chance movement decoding, consistent with paper’s finding that uninstructed movements are tightly coupled to neural signals. |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `lick_direction` | 0.6127 validation balanced accuracy | Paper reports strong above-shuffled neural choice decoding and `AUC = 0.86 ± 0.11` for delay-epoch `CDchoice` projections; qualitatively consistent but not directly the same metric/model/input. |
| `behavioral_context` | 0.7469 validation balanced accuracy | Paper reports strong above-shuffled neural context decoding across epochs (Fig. 4b); qualitatively consistent. |
| `outcome` | 0.6121 validation balanced accuracy | No direct paper decoder for this exact output. |
| `tongue_velocity` | 0.7766 validation balanced accuracy | No direct paper decoder, but paper shows strong movement-neural coupling; consistent. |
| `paw_velocity` | 0.5737 validation balanced accuracy | No direct paper decoder; above chance but weaker than tongue/motion-energy outputs. |
| `motion_energy` | 0.7579 validation balanced accuracy | No direct paper decoder, but paper shows movement kinematics strongly predict neural choice/context-related dynamics; consistent. |

Checks performed:
- **Chance analysis**: ran `cache/step12_checks.py` and saved `cache/step12_checks_out.txt`.
  - `lick_direction`: validation / chance = `1.2254`
  - `behavioral_context`: validation / chance = `1.4938`
  - `outcome`: validation / chance = `1.2242`
  - `tongue_velocity`: validation / chance = `1.5532`
  - `paw_velocity`: validation / chance = `1.1474`
  - `motion_energy`: validation / chance = `1.5158`
  - No output was below chance.
- **Train vs validation gap**:
  - `lick_direction`: train / val = `1.0339`
  - `behavioral_context`: `1.0072`
  - `outcome`: `1.0454`
  - `tongue_velocity`: `1.0124`
  - `paw_velocity`: `1.0103`
  - `motion_energy`: `1.0069`
  - No output approached the `>1.5x` train/validation overfitting threshold; there is no evidence of leakage or severe overfitting.
- **Targeted checks for lower-accuracy outputs**:
  - Verified raw-vs-converted labels for the first three retained trials of `JEB6_2021-04-18`; `lick_direction`, `behavioral_context`, and `outcome` all matched exactly.
  - Reconstructed `paw_velocity` bins directly from raw DLC trajectories for the first three retained trials of `JEB6_2021-04-18`; all three matched the converted output exactly.
  - Reviewed the existing processing plots (`processing_JEB6_2021-04-18.png`, `processing_JEB7_2021-04-29.png`) and decoder sample plots (`sample_trials.png`, `predictions.png`); no temporal misalignment or class-collapse anomaly was evident.
  - Output distributions remained non-degenerate (`paw_velocity` and `motion_energy` were exactly median-balanced; other outputs were not near 99/1 splits).

Interpretation of lower accuracies:
- `lick_direction` and `outcome` are clearly above chance but below the arbitrary `1.5x chance` heuristic for binary variables. After direct raw-label checks and the Step 10 neural/alignment sanity checks, I found no evidence of a conversion bug driving these values.
- `paw_velocity` is the weakest decoded output, but it is still above chance, exactly balanced by construction, and its raw-bin reconstruction matched the converted data in all spot checks. This supports the interpretation that paw movement is simply less decodable from the selected ALM neural activity than tongue speed or motion energy under this target-format task.
- The paper does not report scalar balanced accuracies for these exact six outputs under this exact decoder formulation, so only the choice/context comparisons are directly informative. Those comparisons are qualitatively consistent with the paper: both are robustly above chance and context is especially well decoded.

### Issues Found and Resolved
- **No conversion bugs were uncovered in Step 12**.
- **Lower-than-`1.5x chance` outputs (`lick_direction`, `outcome`, `paw_velocity`, and marginally `behavioral_context`) were investigated and retained** because direct raw-data spot checks, alignment plots, class-balance checks, and train/validation-gap analysis all supported the current conversion.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
