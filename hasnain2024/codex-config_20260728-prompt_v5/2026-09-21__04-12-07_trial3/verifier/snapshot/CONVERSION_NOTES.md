# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse
- **Date started**: 2026-09-21
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

Environment checks:
- `python3` works
- `numpy==2.4.4`
- `torch==2.6.0+cu124`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadObjs` | `code/DataLoadingScripts/loadObjs.m` | LOADING | Load each session `.mat` into a MATLAB `obj` struct, padding missing fields (`meta`, `ex`, `me`) for compatibility. |
| `loadSessionData` | `code/DataLoadingScripts/loadSessionData.m` | LOADING | Load all requested sessions, process each requested probe, concatenate dual-probe sessions, and package per-session params. |
| `processData` | `code/DataLoadingScripts/processData.m` | PROCESSING | Main per-probe pipeline: find trials, find clusters, align spikes, bin/smooth trial data, remove low-FR clusters, compute baseline firing stats. |
| `findTrials` | `code/DataLoadingScripts/findTrials.m` | CURATION | Evaluate trial-condition expressions on `obj.bp` fields and return trial indices for each behavioral condition. |
| `findClusters` | `code/DataLoadingScripts/findClusters.m` | CURATION | Select clusters by quality label; special case `all` excludes `garbage`, `gabrga`, `noisy`, `real?`. |
| `alignSpikes` | `code/DataLoadingScripts/alignSpikes.m` | PROCESSING | Align spike times to the requested event by subtracting per-trial event times; supports `goCue`, `moveOnset`, `firstLick`, `lastLick`, `jawOnset`. |
| `getSeq` | `code/DataLoadingScripts/getSeq.m` | PROCESSING | Bin spike trains on a shared time grid, smooth them, and produce `obj.time`, condition PSTHs, and single-trial neural matrices. |
| `removeLowFRClusters` | `code/DataLoadingScripts/removeLowFRClusters.m` | CURATION | Remove units whose mean smoothed firing rate across conditions is below `lowFR`. |
| `baselineFR` | `code/DataLoadingScripts/baselineFR.m` | PROCESSING | Compute pre-sample baseline mean and SD for each unit from `-0.5 s` to sample onset. |
| `loadMotionEnergy` | `code/DataLoadingScripts/loadMotionEnergy.m` | LOADING | Load session motion-energy file, align/interpolate to the neural timebase, fill initial NaNs, and threshold into movement mask. |
| `getKinematicsFromVideo` | `code/funcs/kinematics/getKinematicsFromVideo.m` | PROCESSING | Interpolate DLC positions to the common trial timebase and derive displacement and velocity features for selected tracked body parts. |
| `findVelocity` | `code/funcs/kinematics/findVelocity.m` | PROCESSING | Compute per-frame gradients for x/y position; for tongue features, missing values are set to zero velocity, while non-tongue features are nearest-filled. |
| `getKinematics` | `code/funcs/kinematics/getKinematics.m` | PROCESSING | Combine DLC kinematics with motion energy, standardize features, and optionally reduce dimensionality. |
| `UseInclusionCritera` | `code/utils/UseInclusionCritera.m` | CURATION | Example session filter retaining sessions with more than 40 trials in each of two key conditions. |
| `loadJEB*_ALMVideo`, etc. | `code/DataLoadingScripts/Recording and video/*.m` | LOADING | Hard-code the session list and the ALM probe(s) to include for each animal/date. |

### Notes
- The reference data are electrophysiology plus behavior/video, not calcium imaging. No `dF/F` computation is applicable.
- Neural alignment is done by subtracting per-trial event times from `obj.clu{probe}(clu).trialtm`, creating `trialtm_aligned`.
- The canonical time window in the default loader is `tmin=-2.5 s`, `tmax=2.5 s`, with `dt=1/200 s` (5 ms bins) in `getDefaultParams.m`. The tutorial in `WorkingWithDataObjs.m` uses `dt=1/100 s` (10 ms), so exact bin width must be resolved against paper/method details later.
- Binning uses histogram counts divided by bin width to yield firing rate, then smoothed by `mySmooth(...)`.
- Low-quality unit handling in code is two-stage:
  - quality-label filter keeps all clusters except `garbage`, `gabrga`, `noisy`, `real?` when `quality={'all'}`;
  - low-firing-rate filter removes units below `lowFR` after PSTH construction.
- `getDefaultParams.m` uses `lowFR=0.5 Hz`, while the tutorial script shows `lowFR=1 Hz`; this is a code-level discrepancy to resolve in later steps.
- Trial-condition logic is expression-based over `obj.bp` fields. The core paper-related conditions distinguish delayed-response (`~autowater`) from water-cued (`autowater`) and exclude `stim.enable` and `early` trials.
- Motion energy is aligned to the same time axis as neural data by interpolating each trial to `obj.time + params.advance_movement` after compensating for the video offset.
- Kinematic features come from DLC-tracked points in side and bottom views. For tongue visibility gaps, the code records NaN intervals, baseline-fills position, and sets missing tongue velocities to zero in `findVelocity`.
- Session inclusion is encoded in the per-animal metadata-loader files under `Recording and video`, including which probe corresponds to ALM in each session.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains four experiment-group directories:
  - `Ephys_Behavior/`: delayed-response + water-cued sessions with neural recordings and separate motion-energy files.
  - `RandomizedDelay_Ephys_Behavior/`: randomized-delay sessions; mostly neural recording sessions, but two files (`JEB24_2023-10-03`, `JEB24_2023-10-04`) contain behavior/video without a `clu` field.
  - `DelayInhibition_BilatMC_Behavior/`: behavior/video-only inhibition sessions, no neural data.
  - `GoCueInhibition_BilatMC_Behavior/`: behavior/video-only inhibition sessions, no neural data.
- Native session files are `data_structure_<ANM>_<DATE>.mat`.
- Motion energy is stored:
  - as separate `motionEnergy_<ANM>_<DATE>.mat` files for `Ephys_Behavior` and many `RandomizedDelay_Ephys_Behavior` sessions;
  - within the session file itself for behavior-only datasets.
- File-format mix:
  - Most `data_structure_*.mat` files are MATLAB v7.3 / HDF5.
  - 11 files in `RandomizedDelay_Ephys_Behavior` are MATLAB v5.
  - `data_structure_JEB6_2021-04-18.mat` is v7.3 but has a non-standard `clu` layout (probe 1 empty, probe 2 populated), which required special handling during inspection.
- Representative native variable layout:
  - `obj.bp`: trial-level behavior/task variables (`Ntrials`, `R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim`, `protocol`, `bitRand`; some files also include `autowaterBlock`, `autolearn`).
  - `obj.bp.ev`: event times (`bitStart`, `sample`, `delay`, `goCue`, `reward`, `lickL`, `lickR`; some session variants may omit/add fields).
  - `obj.traj`: 2 camera views (side, bottom), each carrying per-trial `frameTimes`, DLC `ts`, feature names, and dropped-frame info.
  - `obj.clu` for neural sessions: raw sorted-unit information including `tm`, `trialtm`, `trial`, `quality`, and waveform/channel/site metadata.
  - `me`: motion energy, with at least `data` and `moveThresh`.
- For the decoder task, the relevant source pool is the neural subset:
  - `Ephys_Behavior`
  - `RandomizedDelay_Ephys_Behavior`
- Raw neural data are not yet filtered to the paper’s ALM-session list or ALM probe selection at this step.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 14,297 raw clusters across neural-data files before reference-session filtering |
| Neurons / session | Mean 304.19 raw clusters/session (median 68; min 0; max 1,258) |
| Subjects | 14 subjects in neural-data files (`EKH1`, `EKH3`, `JEB11`, `JEB12`, `JEB13`, `JEB14`, `JEB15`, `JEB19`, `JEB23`, `JEB24`, `JEB6`, `JEB7`, `JGR2`, `JGR3`) |
| Sessions / subject | 47 neural-data files total; per-subject counts range 1-10 |
| Trials (total) | 15,843 across neural-data files before reference-session filtering |
| Trials / session | Mean 337.09 (median 337; min 218; max 517) |

Additional raw inventory:
- `DelayInhibition_BilatMC_Behavior`: 53 data files, 4 subjects, 15,079 total trials, no separate motion-energy files, no neural data.
- `GoCueInhibition_BilatMC_Behavior`: 20 data files, 4 subjects, 6,151 total trials, no separate motion-energy files, no neural data.
- `Ephys_Behavior`: 25 data files, 25 motion-energy files, 10 subjects, 8,260 total trials.
- `RandomizedDelay_Ephys_Behavior`: 22 data files, 20 motion-energy files, 4 subjects, 7,583 total trials.
- Important edge case: `data_structure_JEB24_2023-10-03.mat` and `data_structure_JEB24_2023-10-04.mat` live in `RandomizedDelay_Ephys_Behavior` but have no `clu` field, so they cannot contribute to a neural decoder dataset.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Two-context task: 522 units; DR-only task: 1,651 units; randomized-delay task: 845 units | “two-context paradigm: 12 sessions, six mice, 522 units… an additional three mice were trained only on the DR task: 25 sessions, nine mice, 1,651 units… randomized delay task… 845 units” |
| Neurons / session | Two-context: 43.5 mean; DR-only: 66.0 mean; randomized-delay: 44.5 mean | Derived from text totals: 522/12, 1651/25, 845/19 |
| Subjects | 17 total mice across all cohorts; 6 two-context; 9 fixed-delay ephys; 4 randomized-delay; 4 optogenetics cohort | “This study used data collected from 17 mice…” plus cohort counts in Methods |
| Sessions / subject | Two-context: 12/6 = 2.0; DR-only fixed delay: 25/9 = 2.78; randomized delay: 19/4 = 4.75 | Derived from Methods cohort totals |
| Trials (total) | Not stated explicitly in paper text | No aggregate trial count reported in copied text |
| Trials / session | Not stated explicitly in paper text | No aggregate trial count reported in copied text |
| Neural data time bin | 5 ms | “single-trial neural activity was first binned in 5-ms intervals” |
| Behavior data time bin | 400 Hz video (2.5 ms per frame) | “High-speed video was captured (400-Hz frame rate)” |
| Reward rate | Exact overall reward rate not reported; training criterion was >70% accuracy | “trained on the DR task… until they reached at least 70% accuracy” |
| Task/behavior statistic 1 | Randomized-delay expert criterion: >70% accuracy and <20% early lick rate | “trained on this version of the task until they became experts (>70% accuracy and <20% early lick rate)” |
| Task/behavior statistic 2 | Session inclusion for behavior analyses: at least 40 correct DR left + 40 correct DR right + 20 correct WC left + 20 correct WC right, excluding early/ignore | “All sessions used for behavioral analysis had at least 40 correct DR trials for each direction… and 20 correct WC trials for each direction, excluding early lick and ignore trials” |
| ... | Choice-selective single units in fixed-delay DR: sample 36%, delay 42%, response 58% of 483 single units; context-selective in two-context: 39% of 214 single units | “sample: 36%; delay: 42%; response: 58% of 483 single units” and “39% of single units, 12 sessions, six mice, 214 single units” |


### Processing Details
- Task timing:
  - DR sample tone lasts 1.3 s.
  - Standard fixed delay is 0.9 s, except one mouse with 0.7 s delay that was linearly time-warped to 0.9 s in analyses.
  - Go cue is a 10 ms chirp.
  - Ignore trials are trials without a response within 3 s of the go cue.
- Two-context sessions:
  - Sessions begin with about 100 DR trials, then alternate WC and DR blocks.
  - Block lengths are 10-25 trials.
  - All sessions start with DR.
  - WC trials omit auditory cues and present ~3 µl water at a random side/time.
- Randomized-delay sessions:
  - Delay length is drawn from {0.3, 0.6, 1.2, 1.8, 2.4, 3.6} s with an exponential-like distribution (tau = 0.9 s).
- Video and movement:
  - Two cameras at 400 Hz.
  - Tongue, jaw, and nose tracked in both views; paws only in bottom view.
  - Missing values are nearest-filled for all features except the tongue.
  - Velocity is the first derivative of position.
  - Motion energy is built from framewise temporal-median differences over +/-5 frames and reduced to the 99th percentile per frame.
  - Motion threshold is manually set per session from a bimodal distribution.
- Neural processing:
  - Single-trial data are binned at 5 ms and smoothed with a causal Gaussian kernel (half width 35 ms).
  - Baseline standardization uses -2.4 s to -2.2 s relative to go cue (ITI).
- Decoding details reported in the paper:
  - Choice/context logistic-regression models are trained separately at each time bin.
  - Inputs are either kinematics or single-trial neural activity.
  - Equal numbers of correct left/right trials are used for choice decoding.
  - Models use ridge regularization, four-fold cross-validation, and 30% held-out test data.
- Decoder-aligned event for this project should therefore be compatible with the paper’s dominant alignment convention: go cue (or water drop in WC-specific paper analyses). For this conversion task, I will align to go cue as required.

### Curation Steps

**Neuron curation rules**:
- Units are spike sorted with JRCLUST and/or Kilosort 3 plus manual curation in Phy.
- “Well-isolated single units” are defined by manual inspection of ISI violations, separation from other units, and stationarity.
- Multiunits are manually curated units with higher ISI violation rates.
- Recording sessions are included only if they have at least 10 units.
- For subspace-alignment and single-unit selectivity analyses, only well-isolated single units with firing rate >1 Hz are included.
- For all other analyses, all units with firing rate >1 Hz are included.

**Trial curation rules**:
- Early-lick trials are omitted from analyses.
- Ignore trials are omitted from behavioral-session inclusion counts and many paper analyses.
- Behavioral-session inclusion requires enough correct trials per direction/context (40 correct DR per direction, 20 correct WC per direction).
- Choice decoding in the paper uses equalized correct left/right trial counts.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from `CDchoice` (single-session logistic regression on delay-epoch CD projections) | AUC = 0.86 ± 0.11 across sessions |
| Choice from kinematics or neural population (time-resolved logistic regression) | Clearly above shuffled-label chance; exact scalar summary not reported in text |
| Context from kinematics or neural population (time-resolved logistic regression) | Clearly above shuffled-label chance, including during ITI; exact scalar summary not reported in text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural time bin | `getDefaultParams.m` uses `dt=1/200` (5 ms); tutorial script in `WorkingWithDataObjs.m` uses `dt=1/100` (10 ms) | Raw data are unbinned spike times, so either binning choice is possible downstream | Methods explicitly state single-trial neural activity was binned in 5 ms intervals | Use 5 ms bins in the converter. Treat the 10 ms tutorial as illustrative, not authoritative. |
| Low firing-rate threshold | `getDefaultParams.m` uses `lowFR=0.5`, but figure/analysis scripts overwhelmingly set `lowFR=1` before `loadSessionData()` | Raw clusters have no precomputed low-FR exclusion | Methods say analyses generally used units with firing rates exceeding 1 Hz | Use `>1 Hz` low-FR filtering to match the paper and most figure scripts. |
| Session lists in randomized-delay data | Raw directory has 22 `data_structure_*.mat` files in `RandomizedDelay_Ephys_Behavior` | Two files (`JEB24_2023-10-03`, `JEB24_2023-10-04`) have no `clu`; one `JEB23` date is commented out in the loader code | Paper reports 19 randomized-delay recording sessions from 4 mice | Follow the reference code session list / usable neural-session set: exclude the two no-neural JEB24 files and the extra commented-out JEB23 date, yielding 19 sessions. |
| Fixed-delay subject count | Raw/code session loaders enumerate 25 fixed-delay sessions across 10 mouse IDs | Data support those 25 sessions directly | Paper text says 25 sessions from 9 mice | Prioritize the explicit code/data session list over the one-number mouse-count discrepancy in text. Document this as a likely manuscript/reporting inconsistency unless later counts show one mouse is fully excluded by inclusion criteria. |
| Two-context versus DR-only sessions | Code session loaders mix sessions with substantial `autowater` blocks and sessions with few/zero `autowater` trials | Raw files confirm that some fixed-delay sessions contain many WC trials, others contain almost none, and some contain none | Paper discusses both two-context sessions and additional DR-only fixed-delay sessions | Keep both session types available; later mapping will label context per trial (`DR`/`WC`) and allow DR-only sessions to carry a constant `DR` context label. |
| Go-cue alignment for WC trials | Code defaults to aligning spikes to `goCue` | Raw event fields include non-NaN `goCue` timestamps even on `autowater` trials, with reward occurring shortly after `goCue` | Paper often discusses go-cue alignment for DR and water-drop alignment for some WC behavioral analyses | For this project, align all trials to `goCue` as instructed by the task. This is feasible because WC trials still carry a usable `goCue` field in the raw data. |
| Tongue missing-value handling | Methods say missing values are nearest-filled for all features except tongue; code baseline-fills tongue position after bookkeeping and sets missing tongue velocities to 0 | Raw DLC trajectories contain missing tongue samples | Methods and code differ slightly in wording but both preserve tongue visibility as a special case | Treat “tongue not visible” explicitly in decoder outputs; use the raw visibility/missingness rather than pretending tongue velocity is observed during invisibility. |
| Motion-energy thresholding | Code loads session-specific `moveThresh`; methods say threshold was set manually per session using bimodal motion-energy distributions | Raw motion-energy files include per-session threshold metadata | Paper and code are consistent | Use provided per-session motion-energy values directly and create decoder classes from per-session medians as required by the task, while retaining note that original move/not-move threshold was manual. |
| Behavioral inclusion criteria | Utility filters show trial-count criteria; not every script applies them automatically | Raw sessions vary widely in WC availability and trial counts | Methods require at least 40 correct DR and 20 correct WC trials per direction for behavioral analyses; recording sessions need at least 10 units | Implement task-appropriate filtering explicitly in the converter rather than assuming every raw file should be included unchanged. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu` spike times from ALM probe(s) | `neural` | Select ALM probe(s), drop excluded qualities, align `trialtm` to `bp.ev.goCue`, bin in 5 ms from -2.5 s to +2.5 s, convert to firing rate, smooth with causal Gaussian like `mySmooth(...,15,'reflect')`, remove units with mean FR <= 1 Hz | `loadSessionData`, `processData`, `alignSpikes`, `getSeq`, `removeLowFRClusters`, `mySmooth` | All kept units are ALM units; multiunits are allowed if FR > 1 Hz, matching “all units >1 Hz” analyses. |
| Shared time grid relative to go cue | `input[0]` | 1 x T vector of bin centers in seconds, repeated for every trial | `getSeq` (`obj.time`) | Decoder input is the required continuous time-from-go-cue signal. |
| First post-go-cue lick side from `bp.ev.lickL` / `bp.ev.lickR` plus `bp.no` fallback | `output[0]` (`lick_direction`) | Per-trial categorical label repeated across all time bins: left / right / none | `firstLickTime` (related logic) | Use actual first post-go-cue lick side, not instructed side. Ignore/no-response trials map to `none`. |
| `bp.autowater` | `output[1]` (`behavioral_context`) | Per-trial categorical label repeated across time bins: `DR` if 0, `WC` if 1 | `findTrials` condition logic | DR-only sessions remain valid with constant `DR` label. |
| `bp.hit`, `bp.miss`, `bp.no` | `output[2]` (`outcome`) | Per-trial categorical label repeated across time bins: correct / incorrect / ignore | `findTrials` condition logic | Early trials are excluded entirely rather than encoded as an outcome. |
| Raw DLC tongue trajectories (`obj.traj`) | `output[3]` (`tongue_velocity`) | Build aligned tongue speed trace from tracked tongue points using reference interpolation/velocity logic; per-session median threshold on visible samples gives class 0/1; class 2 when tongue not visible | `findPosition`, `findVelocity`, `getKinematicsFromVideo` | Visibility comes from raw NaNs before tongue-specific filling; output must explicitly preserve “not visible”. |
| Raw DLC paw trajectories (`obj.traj`) | `output[4]` (`paw_velocity`) | Build aligned paw speed trace from paw-tracked bottom-view points; per-session median threshold on visible samples gives class 0/1; class 2 when paw not visible | `findPosition`, `findVelocity`, `getKinematicsFromVideo` | Use visibility mask from raw aligned positions before nearest-fill. |
| Motion-energy trace (`motionEnergy_*.mat` when available, else `obj.me`) | `output[5]` (`motion_energy`) | Align/interpolate to neural time grid; per-session median threshold on valid samples gives class 0/1; class 2 for sessions lacking motion-energy/video data | `loadMotionEnergy`, `findVideoOffset` | For neural sessions with valid video, output is time-varying 0/1 after discretization. |
| Animal/session identity from filename / `obj.ex` | `subjects`, `subject_idx`, metadata | Map each included session to its mouse ID and session date | session-loader files under `Recording and video/` | Session order will be deterministic and documented. |
| Probe location metadata (`obj.ex.probe`) / loader-selected probe IDs | `brain_regions`, `brain_region_idx` | Keep only ALM units, assign region index `ALM` | `load<Animal>_ALMVideo.m` session loaders | Final `brain_regions` is expected to be `['ALM']` unless an edge case appears during implementation. |

### Key Decisions
1. **Include only sessions with neural data**: Use `Ephys_Behavior` plus the 19 usable randomized-delay recording sessions; exclude the behavior-only directories and any file with no `clu`, because the decoder requires neural input.
2. **Use the reference ALM-session selection when available**: Follow the paper code’s `Recording and video/load*_ALMVideo.m` session/probe logic for the randomized-delay dataset and ALM probe assignment. For fixed-delay files, raw directory contents already match the 25-session reference session set.
3. **Filter trials conservatively but keep decoder-relevant outcomes**: Exclude `early` trials and `stim.enable` trials to match the paper/reference conditions. Keep `hit`, `miss`, and `no` trials so the required `outcome` output can include correct / incorrect / ignore.
4. **Align every trial to `goCue`**: This is mandated by the task and is supported by the raw event structure, including WC trials.
5. **Use 5 ms neural bins and causal smoothing matching the reference pipeline**: This matches Methods and the default/reference scripts more closely than the 10 ms tutorial example.
6. **Keep all ALM units above 1 Hz, not only single units**: The paper explicitly uses all >1 Hz units for most population analyses; restricting to single units would throw away substantial signal and diverge from most reference analyses.
7. **Represent outputs as time-by-category sequences where needed, but keep per-trial labels constant over time**: Lick direction, context, and outcome are conceptually per-trial, yet storing them as constant 1 x T traces keeps output shapes uniform alongside time-varying kinematic outputs.
8. **Define lick direction from actual behavior, not instructed side**: First post-go-cue lick side is the cleanest behavioral output and naturally gives `none` for ignore trials.
9. **Preserve “not visible” / “no video” explicitly as class 2**: This is required by the task and is preferable to silently imputing those outputs into low/high velocity classes.
10. **Threshold continuous movement outputs by per-session medians on valid samples**: This follows the task specification exactly and avoids cross-session scale differences.

### Planned Sanity Checks
- [ ] Neural check: for a chosen session / unit / trial, manually histogram aligned raw spike times into 5 ms bins, apply the same smoothing kernel, and verify equality with the converted trial matrix using `np.allclose()`.
- [ ] Input check: verify that every trial’s input vector equals the expected bin-center time axis from -2.5 s to +2.5 s relative to go cue.
- [ ] Output check (choice/outcome/context): for selected trials, recompute first post-go-cue lick side, `autowater`, and `hit`/`miss`/`no` directly from raw fields and compare to converted labels with `np.allclose()`.
- [ ] Output check (motion/video): for selected trials, interpolate raw motion-energy and DLC-derived speed traces directly from source files and compare discretized class traces to converted outputs.
- [ ] Dataset-size check: compare converted session counts, subject counts, and ALM unit counts against paper-reported cohort sizes and the reference code’s session lists.
- [ ] Edge-case check: confirm that no-`clu` files, `early` trials, and `stim.enable` trials are excluded exactly as intended.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Wrote `/app/convert_data.py` with CLI interface:
  - `python -u /app/convert_data.py <outpicklefile>`
  - `--full` to process the full reference session list
  - `--sample` to process exactly 2 test sessions (`JEB13_2022-09-13` and `JEB23_2023-10-18`) so both HDF5 and MATLAB v5 loading paths are exercised
  - `--show-processing` to save `processing_<session_id>.png` for up to 2 sessions
- Implemented mixed MATLAB loader support:
  - HDF5 / v7.3 files via `mat73`
  - MATLAB v5 files via `scipy.io.loadmat`
  - recursive normalization to a shared Python structure for `bp`, `traj`, `clu`, and motion-energy files
- Hard-coded the reference ALM session/probe manifest from `Recording and video/load*_ALMVideo.m`:
  - 25 fixed-delay sessions
  - 19 randomized-delay sessions
  - randomized-delay exclusion of commented-out `JEB23_2023-10-20`
- Implemented neural processing to match the reference pipeline:
  - align spike times to `goCue`
  - bin from `-2.5 s` to `+2.5 s` in 5 ms bins
  - smooth with a Python port of `mySmooth(..., 15, 'reflect')`, including the causal Gaussian kernel and the same prepend-style boundary handling
  - exclude cluster qualities `garbage`, `gabrga`, `noisy`, `real?`
  - remove units with mean FR `<= 1 Hz`
  - exclude sessions with fewer than 10 remaining units
- Implemented trial filtering:
  - exclude `early` trials
  - exclude `stim.enable` trials
  - keep `hit`, `miss`, and `no` trials
  - require finite `goCue`
- Implemented outputs:
  - per-trial `lick_direction`, `behavioral_context`, `outcome` repeated across time bins
  - time-varying `tongue_velocity`, `paw_velocity`, `motion_energy`
  - explicit class `2` for not-visible / no-video outputs, preserving missingness before any reference-style filling
- Implemented processing plots showing:
  - mean neural activity across trials
  - per-trial output distributions
  - example aligned/discretized tongue, paw, and motion traces
- Smoke-test run completed successfully:
  - `python3 -u /app/convert_data.py /app/_step6_test.pkl --sample`
  - output: 2 sessions, 793 kept trials, 91 kept ALM units, elapsed `10.3 s`
  - initial run revealed one loader bug for top-level MATLAB v5 dict recursion; fixed immediately
  - initial run also emitted a harmless all-NaN visibility warning from absent video frames; fixed by guarding the baseline-derivative calculation

Code inefficiencies identified:
- A naive cluster-by-trial histogram loop would have been too slow and would not scale to full-dataset conversion.
- Re-loading or re-interpolating movement data repeatedly per output would add unnecessary overhead.

Code speedups added:
- Spike binning is vectorized with `np.bincount` over flattened `(time_bin, trial)` indices instead of nested Python loops over spikes/trials.
- Firing-rate filtering reuses already computed single-trial neural matrices to build condition PSTHs instead of re-binning spikes a second time.
- Trial-level motion / kinematic alignment is done once per session per feature and then reused for discretization and plotting.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 91 |
| Neurons / session | 48, 43 (mean 45.5) |
| Subjects | 2 (`JEB13`, `JEB23`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 793 |
| Trials / session | 389, 404 |
| Time input range | [-2.5, 2.5] s |
| Lick direction distribution | [0.405 left, 0.494 right, 0.101 none] |
| Behavioral context distribution | [0.026 WC, 0.974 DR] |
| Outcome distribution | [0.148 incorrect, 0.752 correct, 0.101 ignore] |
| Tongue velocity distribution | [0.159 low, 0.159 high, 0.682 not visible] |
| Paw velocity distribution | [0.477 low, 0.477 high, 0.046 not visible] |
| Motion energy distribution | [0.479 low, 0.480 high, 0.041 no video] |

### Processing Plots Review
- Created:
  - `/app/processing_JEB13_2022-09-13.png`
  - `/app/processing_JEB23_2023-10-18.png`
- Review:
  - Neural trial-mean activity is centered on the go-cue alignment point with no obvious truncation or time-base mismatch.
  - Tongue velocity shows expected high missingness because visibility is intermittent; this is intentional and now explicitly encoded as class `2`.
  - Paw velocity now preserves a small but non-zero not-visible fraction after fixing the initial visibility bug caused by nearest-filling before class assignment.
  - Motion energy has a small `no_video` fraction near trial edges / missing video samples, which is expected from interpolation to a common neural timebase.
  - No plotting anomalies suggesting obvious neural/video temporal misalignment were observed.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Vectorized spike binning with `np.bincount` instead of nested trial loops | Keeps per-session processing in the ~3-8 s range in sample runs rather than scaling with Python-level spike/trial loops |
| Reuse single-trial rate matrices for low-FR filtering / condition PSTHs | Avoids a second spike-binning pass per unit |
| Single-pass motion / kinematic interpolation per feature | Avoids repeated frame-time interpolation during discretization / plotting |

| Step | Time / Session | Estimated Total Time |
| Sample conversion (`--sample`) | 11.7 s / 2 sessions = 5.85 s / session overall | |
| Fixed-delay sample session (`JEB13_2022-09-13`) | 8.4 s | |
| Randomized-delay sample session (`JEB23_2023-10-18`) | 3.2 s | |
| Full conversion estimate | weighted by observed per-session times and paper-scale unit counts | approximately 4-6 minutes for 44 sessions |

Step 7 checks:
- Sample conversion command completed successfully:
  - `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing 2>&1 | tee /app/conversion_sample_out.txt`
- Verification command completed successfully:
  - `python -u /app/train_decoder.py /app/sample_data.pkl --verify-only 2>&1 | tee /app/verification_sample_out.txt`
- `train_decoder.py --verify-only` reported:
  - `Data format is valid, no errors or warnings.`
- Issues found and fixed during Step 7:
  - Top-level MATLAB v5 dict recursion bug in the loader prevented reading motion-energy files; fixed by recursively converting dict values as well as `mat_struct` values.
  - Paw not-visible class was initially lost because nearest-filled trajectories were being used to infer visibility; fixed by preserving a pre-fill visibility mask and only using reference-style filled traces for velocity computation.
  - A harmless empty-slice warning while averaging across two paw markers was fixed by replacing `nanmean` with explicit sum/count logic.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `lick_direction` | 0.5495 | 0.5505 |
| `behavioral_context` | 0.8113 | 0.8145 |
| `outcome` | 0.5455 | 0.5688 |
| `tongue_velocity` | 0.6797 | 0.6746 |
| `paw_velocity` | 0.6046 | 0.6220 |
| `motion_energy` | 0.7173 | 0.7117 |

Step 8 training checks:
- Command run:
  - `python -u /app/train_decoder.py /app/sample_data.pkl 2>&1 | tee /app/train_decoder_sample_out.txt`
- Training completed successfully on GPU (`cuda`).
- Loss decreased monotonically in the logged checkpoints:
  - epoch 1: `7.530454`
  - epoch 50: `0.939621`
  - epoch 100: `0.777258`
  - epoch 150: `0.763428`
  - epoch 200: `0.755776`
  - test loss: `0.786340`
- All validation balanced accuracies were above chance:
  - 3-class outputs chance = `0.3333`
  - 2-class output (`behavioral_context`) chance = `0.5000`
- No sample-decoding symptom currently suggests a gross alignment or labeling error.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3.0G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 1,651 fixed-delay + 845 randomized = 2,496 recorded units | Session/probe list implies 44 included sessions but does not print one combined unit count in the loader files | 44 selected sessions before trial filtering; raw selected-probe unit count not directly tabulated in source files | 2,364 units after quality exclusion + `>1 Hz` firing-rate filter | Partial / explained: converted count is expected to differ from paper’s recorded-unit count because the converter applies the reference low-FR analysis filter. |
| Mean neurons/session | Fixed: 66.0; randomized: 44.5 | Not explicitly tabulated | Not explicitly tabulated | 53.73 overall; fixed 59.76; randomized 45.79 | Partial / explained: randomized closely matches paper, fixed is lower after filtering. |
| Subjects | Fixed 9 mice + randomized 4 mice in text; 17 total across all cohorts in paper | 14 unique neural-session subjects in provided loader/session files | 14 unique subjects in the selected neural files | 14 | Match to provided code/data; differs from one paper sentence as already documented in Step 4. |
| Sessions | 25 fixed-delay + 19 randomized = 44 | 25 fixed-delay + 19 randomized = 44 | 44 selected neural sessions | 44 | Yes |
| Trials (total) | Not stated explicitly | Not stated explicitly | 14,972 raw trials across the 44 selected session files | 13,762 kept trials after excluding early, stim, and uncovered post-recording tail trials | Yes / task-appropriate filtering applied |
| Trials/session (mean) | Not stated explicitly | Not stated explicitly | 340.27 raw | 312.77 kept | Yes / filtered |
| Time input range | N/A | `obj.time` from `-2.5 s` to `+2.5 s` in 5 ms bins | Supported by raw `goCue` events and spike times | `[-2.5, 2.5]` | Yes |
| Lick direction distribution | Not tabulated | Not tabulated | Derivable only after conversion | `[0.423 left, 0.446 right, 0.132 none]` | Plausible |
| Behavioral context distribution | Two-context subset described qualitatively; not tabulated over all sessions | Not tabulated | Mixed DR/WC in fixed-delay cohort; DR-only randomized cohort | `[0.097 WC, 0.903 DR]` | Plausible |
| Outcome distribution | Ignore trials expected to be relatively infrequent; exact aggregate not tabulated | Not tabulated | Derivable only after conversion | `[0.120 incorrect, 0.749 correct, 0.131 ignore]` | Plausible |
| Tongue velocity distribution | Not tabulated | Not tabulated | Derivable only after conversion | `[0.172 low, 0.172 high, 0.657 not visible]` | Plausible |
| Paw velocity distribution | Not tabulated | Not tabulated | Derivable only after conversion | `[0.479 low, 0.479 high, 0.042 not visible]` | Plausible |
| Motion energy distribution | Not tabulated | Not tabulated | Derivable only after conversion | `[0.481 low, 0.481 high, 0.038 no video]` | Plausible |

Full-conversion notes:
- Command run:
  - `python -u /app/convert_data.py /app/converted_data.pkl --full 2>&1 | tee /app/conversion_full_out.txt`
- Final conversion runtime:
  - `214.6 s` total for 44 sessions
- Verification command run:
  - `python -u /app/train_decoder.py /app/converted_data.pkl --verify-only 2>&1 | tee /app/verification_full_out.txt`
- Final verification result:
  - `Data format is valid, no errors or warnings.`
- Important issue found during Step 9 and fixed:
  - Two JEB24 randomized-delay sessions (`2023-10-23`, `2023-11-03`) contained more behavioral trials than were covered by the neural cluster trial indices.
  - Initial full conversion therefore produced blocks of all-zero neural trials at the session tail, which `train_decoder.py --verify-only` warned about.
  - Fix: added an explicit `last_neural_trial` filter so valid trials must lie within the neural recording coverage of the selected probe(s).
  - After rebuilding, warnings were eliminated and the kept-trial counts for those sessions changed from:
    - `JEB24_2023-10-23`: 320 -> 292
    - `JEB24_2023-11-03`: 334 -> 301
- Cohort summaries in the final converted dataset:
  - Fixed-delay cohort: 25 sessions, 10 subjects, 7,426 kept trials, 1,494 kept units
  - Randomized-delay cohort: 19 sessions, 4 subjects, 6,336 kept trials, 870 kept units
- Spot checks performed during Step 9:
  - confirmed that the sample sessions from Step 7 remained unchanged in structure after the full-dataset fixes
  - confirmed that the JEB24 late-trial all-zero neural warnings disappeared after the coverage filter was added

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` now reports `Data format is valid, no errors or warnings.` Initial warnings about all-zero neural trials in two JEB24 sessions were traced to behavior trials extending past the final recorded neural trial and were fixed by adding the `last_neural_trial` filter.
2. **Constructed raw-data sanity checks with `np.allclose()`**:
   - Neural check:
     - Session: `JEB13_2022-09-13`
     - Raw source: `/app/data/Ephys_Behavior/data_structure_JEB13_2022-09-13.mat`
     - Manual reconstruction:
       - selected reference probe 2
       - excluded `early` and `stim.enable`
       - enforced neural-trial coverage cutoff
       - aligned raw `trialtm` to `goCue`
       - binned into 5 ms bins from `-2.5` to `+2.5`
       - applied the same causal Gaussian smoothing as `mySmooth(...,15,'reflect')`
       - selected the first cluster surviving quality and `>1 Hz` filtering
     - Comparison: full 1,000-bin neural trace for the first kept neuron on the first kept trial
     - Result: `np.allclose(...) == True`
   - Input check:
     - Session: `JEB13_2022-09-13`
     - Manual reconstruction: expected shared time axis `[-2.4975, ..., 2.4975]`
     - Comparison: converted first trial input vector
     - Result: `np.allclose(...) == True`
   - Output behavior-label check:
     - Session: `JEB13_2022-09-13`
     - Raw source fields: `lickL`, `lickR`, `autowater`, `hit`, `miss`, `no`, `goCue`
     - Comparison: first 5 kept trials, using actual first lick within 3 s after go cue
     - Result: converted `(lick_direction, behavioral_context, outcome)` tuples exactly matched the manually derived labels; `np.allclose(...) == True`
   - Output motion-energy check:
     - Session: `JEB23_2023-10-10`
     - Raw sources:
       - `/app/data/RandomizedDelay_Ephys_Behavior/data_structure_JEB23_2023-10-10.mat`
       - `/app/data/RandomizedDelay_Ephys_Behavior/motionEnergy_JEB23_2023-10-10.mat`
     - Manual reconstruction:
       - excluded `early` and `stim.enable`
       - enforced neural-trial coverage cutoff
       - computed `vidshift = median(bitstart_video)/fs - median(bitStart_bpod)`
       - interpolated raw motion energy to the common neural time axis relative to `goCue`
       - thresholded by the per-session median across valid aligned samples
     - Comparison: full 1,000-bin motion-energy class trace for the first kept trial
     - Result: `np.allclose(...) == True`
3. **Reference code comparison**:
   - Data loading:
     - Reference: `loadObjs`, `loadSessionData`, `Recording and video/load*_ALMVideo.m`
     - Converter: hard-coded the same session/probe manifest; supports both HDF5/v7.3 and MATLAB v5 container variants found in the provided data.
   - Neuron / cluster filtering:
     - Reference: `findClusters(..., {'all'})` excludes `garbage`, `gabrga`, `noisy`, `real?`; `removeLowFRClusters(..., lowFR)`
     - Converter: same quality exclusions; uses `>1 Hz` low-FR threshold per Methods / figure scripts.
   - Temporal alignment:
     - Reference: `alignSpikes` subtracts `bp.ev.goCue` per trial; `findVideoOffset` aligns video to neural start time
     - Converter: same go-cue subtraction and same `vidshift` formula.
   - Binning / smoothing:
     - Reference: `getSeq` bins from `tmin=-2.5` to `tmax=2.5` in `dt=1/200`; `mySmooth(...,15,'reflect')`
     - Converter: same time window, same bin width, same causal Gaussian kernel and prepend-style reflect boundary handling.
   - Input construction:
     - Reference: `obj.time` is the shared bin-center axis
     - Converter: uses that shared time axis directly as the only decoder input.
   - Output construction:
     - Reference code does not produce decoder targets in this exact format, but it uses the same source fields and aligned motion / video streams.
     - Converter difference from reference:
       - keeps explicit categorical missingness for tongue/paw/motion outputs because the decoder task requires class `2` rather than silent imputation.
       - repeats per-trial outputs across time bins for uniform `(n_output, T)` trial shapes.
4. **Key statistics comparison**:
   - Session counts match the provided reference code/data exactly: 25 fixed-delay + 19 randomized-delay = 44 sessions.
   - Subject count matches the provided session list / data exactly: 14 unique subjects.
   - Converted kept-unit counts are:
     - fixed-delay: 1,494
     - randomized-delay: 870
     - total: 2,364
   - These differ from paper-reported recorded-unit counts (1,651 fixed-delay; 845 randomized-delay) but are explainable:
     - the converter applies the reference `>1 Hz` analysis filter
     - the paper and provided code/data already contain a documented mouse-count inconsistency for the fixed-delay cohort
     - provided data include container / session-format edge cases that required explicit handling
   - No statistic in the final converted dataset currently contradicts the provided reference code or the actually provided session files.
5. **Edge-case review**:
   - MATLAB container edge cases:
     - top-level MATLAB v5 dict recursion bug for motion-energy files: fixed
     - randomized-delay motion-energy files stored as raw cell arrays instead of structs: fixed
   - Neural coverage edge case:
     - behavior continued after neural recording ended in two JEB24 sessions; fixed by dropping post-recording trials
   - Missing-data handling:
     - paw visibility now uses the raw pre-fill visibility mask so class `2` reflects real invisibility
     - tongue missingness remains explicit and is not silently converted into a low/high observed-velocity class
   - Session eligibility:
     - files without `clu` are excluded by construction
     - sessions with fewer than 10 kept units are excluded; final dataset retains 44 valid sessions, so no reference-listed session fell below threshold after the final fixes

### Issues Found and Resolved
- **MATLAB v5 motion-energy recursion bug**: top-level dict values were not being recursively converted, which broke sample conversion on fixed-delay motion-energy files. Fixed by extending recursive conversion to ordinary dicts.
- **Randomized-delay motion-energy container mismatch**: some motion-energy files were raw cell arrays instead of structs with `.data`. Fixed by accepting both representations in the loader.
- **Paw visibility class bug**: nearest-filled paw trajectories were initially being used to infer visibility, erasing class `2`. Fixed by preserving a raw pre-fill visibility mask and only using the filled traces for velocity estimation.
- **All-zero neural tail trials in JEB24 sessions**: `verification_full_out.txt` initially flagged blocks of zero neural activity. Investigation showed `bp.Ntrials` exceeded the largest neural `trial` index for those sessions. Fixed by enforcing `trial <= last_neural_trial` for valid converted trials.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `lick_direction` | 0.6066 | 0.5917 | Above 1.5x chance (`0.3333`) |
| `behavioral_context` | 0.8400 | 0.8317 | Above 1.5x chance (`0.5000`) |
| `outcome` | 0.6085 | 0.5880 | Above 1.5x chance (`0.3333`) |
| `tongue_velocity` | 0.6315 | 0.6261 | Above 1.5x chance (`0.3333`) |
| `paw_velocity` | 0.5807 | 0.5764 | Above 1.5x chance (`0.3333`) |
| `motion_energy` | 0.7067 | 0.7007 | Above 1.5x chance (`0.3333`) |

Step 11 training notes:
- Command run:
  - `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`
- Training completed successfully on GPU (`cuda`); no CPU fallback was needed.
- Loss curve summary:
  - epoch 1: `8.281474`
  - epoch 10: `5.020575`
  - epoch 20: `3.328373`
  - epoch 50: `1.242383`
  - epoch 100: `0.793187`
  - epoch 150: `0.761020`
  - epoch 200: `0.746707`
  - test loss: `0.791778`
- All full-dataset validation accuracies were comfortably above chance.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `lick_direction` | Validation balanced accuracy `0.5917` | Paper reports choice decoding from `CDchoice` projections at `AUC = 0.86 ± 0.11`; not directly numerically comparable because the paper metric, decoder input space, and target definition differ, but both indicate strong above-chance choice information. |
| `behavioral_context` | Validation balanced accuracy `0.8317` | Paper reports context decoding as strongly above shuffled chance across time, including in the ITI; no exact scalar summary was found in the provided text. Result is qualitatively consistent. |
| `outcome` | Validation balanced accuracy `0.5880` | No direct scalar decoder metric reported in the provided paper text. Above-chance performance is expected if neural activity captures response and reward-related information. |
| `tongue_velocity` | Validation balanced accuracy `0.6261` | No direct paper metric for this exact categorical output. Above-chance performance is consistent with the paper’s strong movement-related signal in ALM and video variables. |
| `paw_velocity` | Validation balanced accuracy `0.5764` | No direct paper metric for this exact categorical output. Above-chance performance is expected but somewhat lower than tongue/motion, which is reasonable for a coarser movement feature. |
| `motion_energy` | Validation balanced accuracy `0.7007` | No direct scalar paper metric for this exact discretized output. The paper reports strong neural/kinematic coupling and substantial movement information; result is qualitatively consistent. |

Accuracy-vs-chance review:
- Chance levels:
  - 3-class outputs: `0.3333`
  - 2-class output (`behavioral_context`): `0.5000`
- Validation accuracies:
  - `lick_direction`: `0.5917` (`1.78x` 3-class chance)
  - `behavioral_context`: `0.8317` (`1.66x` 2-class chance)
  - `outcome`: `0.5880` (`1.76x` 3-class chance)
  - `tongue_velocity`: `0.6261` (`1.88x` 3-class chance)
  - `paw_velocity`: `0.5764` (`1.73x` 3-class chance)
  - `motion_energy`: `0.7007` (`2.10x` 3-class chance)
- No output fell below chance or below the `1.5x chance` review threshold.

Train-vs-validation gap review:
- `lick_direction`: train `0.6066` vs val `0.5917` -> ratio `1.03`
- `behavioral_context`: train `0.8400` vs val `0.8317` -> ratio `1.01`
- `outcome`: train `0.6085` vs val `0.5880` -> ratio `1.03`
- `tongue_velocity`: train `0.6315` vs val `0.6261` -> ratio `1.01`
- `paw_velocity`: train `0.5807` vs val `0.5764` -> ratio `1.01`
- `motion_energy`: train `0.7067` vs val `0.7007` -> ratio `1.01`
- No output shows the `>1.5x` train/validation gap that would indicate serious overfitting or leakage.

Interpretation:
- Choice / context / outcome decoding are all well above chance, supporting the correctness of the trial alignment and behavioral labeling.
- Movement-related outputs are also well above chance, supporting the video/motion alignment and discretization logic.
- The closest paper-number comparison is the paper’s `CDchoice` AUC, but that is not a like-for-like metric with the current decoder task. The appropriate comparison here is mainly qualitative: the converted dataset supports robust above-chance decoding of the same broad variable classes discussed in the paper, which it does.

### Issues Found and Resolved
- **No new issues were revealed by the full-decoder accuracy review.** All outputs exceeded chance substantially, none showed a problematic train/validation gap, and no additional conversion changes were required after Step 11.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Step 13 notes:
- Created `/app/README.md` summarizing the converted dataset, format, processing rules, and validation results.
- Created `/app/cache/README_CACHE.md`.
- Moved the non-deliverable Step 6 smoke-test pickle to `/app/cache/step6_smoketest.pkl`.
- Kept all required deliverables at the top level of `/app`.
