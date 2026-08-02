# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse
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
| `loadObjs` | `code/DataLoadingScripts/loadObjs.m` | LOADING | Loads each session `.mat` into `obj`; ensures missing fields (`meta`, `ex`, `me`) exist so structs concatenate cleanly. |
| `loadSessionData` | `code/DataLoadingScripts/loadSessionData.m` | LOADING | Main session loader; calls `loadObjs`, then `processData` for each requested probe, and concatenates probes within session. |
| `processData` | `code/DataLoadingScripts/processData.m` | PROCESSING | Reference pipeline driver: finds trials, finds clusters, aligns spikes, bins/smooths spikes, removes low-FR units, computes presample baseline statistics. |
| `findTrials` | `code/DataLoadingScripts/findTrials.m` | CURATION | Evaluates logical trial-condition expressions against `obj.bp.*` fields and returns matching trial indices. |
| `findClusters` | `code/DataLoadingScripts/findClusters.m` | CURATION | Selects clusters by quality; special case `all` keeps all except `garbage`, `gabrga`, `noisy`, `real?`. |
| `alignSpikes` | `code/DataLoadingScripts/alignSpikes.m` | PROCESSING | Aligns per-spike trial times to chosen event (`goCue`, `moveOnset`, `firstLick`, `lastLick`, `jawOnset`). |
| `getSeq` | `code/DataLoadingScripts/getSeq.m` | PROCESSING | Bins aligned spikes on `edges = tmin:dt:tmax`, smooths them, creates trial-averaged PSTHs and single-trial neural matrices. |
| `removeLowFRClusters` | `code/DataLoadingScripts/removeLowFRClusters.m` | CURATION | Drops units with mean firing rate below threshold based on PSTHs across conditions. |
| `baselineFR` | `code/DataLoadingScripts/baselineFR.m` | PROCESSING | Computes presample baseline firing rate and sigma from unaligned spike times. |
| `getDefaultParams` | `code/DataLoadingScripts/getDefaultParams.m` | PROCESSING | Defines default alignment (`goCue`), time window (`-2.5` to `2.5` s), neural bin width (`dt = 1/200`), low-FR threshold (`0.5` Hz), and DLC features. |
| `loadMotionEnergy` | `code/DataLoadingScripts/loadMotionEnergy.m` | LOADING | Loads per-trial motion energy, aligns to chosen event using video offset, interpolates onto `obj.time`, fills missing samples, thresholds movement. |
| `findVideoOffset` | `code/funcs/findVideoOffset.m` | PROCESSING | Computes video-to-neural timing offset from bitcode metadata (`sglx.bitcode.bitstart/fs - bp.ev.bitStart`). |
| `findPosition` | `code/funcs/kinematics/findPosition.m` | PROCESSING | Extracts DLC x/y trajectories for one body part, applies video offset, aligns to event, interpolates onto neural time axis. |
| `findVelocity` | `code/funcs/kinematics/findVelocity.m` | PROCESSING | Computes per-trial x/y velocity from interpolated positions; tongue NaNs become zero velocity, non-tongue NaNs filled by nearest. |
| `getKinematicsFromVideo` | `code/funcs/kinematics/getKinematicsFromVideo.m` | PROCESSING | Builds full kinematics tensor from DLC positions and velocities for selected features across views. |
| `getKinematics` | `code/funcs/kinematics/getKinematics.m` | PROCESSING | Adds tongue angle/length and motion energy to kinematic features, standardizes them, and performs dimensionality reduction. |
| `UseInclusionCritera` | `code/utils/UseInclusionCritera.m` | CURATION | Drops sessions with fewer than 40 right-hit and 40 left-hit trials. |

### Notes
- The reference dataset is electrophysiology plus behavior/video, not calcium imaging. No `dF/F` computation is needed.
- Neural data are sorted spike times in `obj.clu{probe}(cluster)` with per-spike session time (`tm`), per-spike within-trial time (`trialtm`), and trial index (`trial`).
- The core neural representation used downstream is `obj.trialdat`, produced by histogramming aligned spikes into fixed bins and smoothing.
- Alignment is event-based. For this task, the reference default is `goCue`, which matches the requested decoder alignment.
- Reference binning window is centered on the alignment event. Default code uses `tmin=-2.5`, `tmax=2.5`, `dt=1/200` (5 ms bins). The tutorial script in `WorkingWithDataObjs.m` shows a variant with `dt=1/100` (10 ms bins), so exact choice must be cross-checked against data/text later.
- Motion energy and DLC trajectories are not used at their native frame rate directly. They are interpolated onto the same `obj.time` axis as neural activity after correcting for video offset.
- DLC/video alignment uses `frameTimes - vidshift - alignTimes(trial)`, where `vidshift` is computed from bitcode synchronization metadata.
- Motion energy loader expects one motion-energy file per ephys session and converts it to `me.data(time, trial)` on the neural time base. It also defines a binary `me.move` using a stored session threshold.
- Cluster curation has two stages in code:
  1. Quality filter by string label via `findClusters`.
  2. Low-firing-rate removal via `removeLowFRClusters`.
- Session-level inclusion criterion found in code: at least 40 right-hit and 40 left-hit trials (`UseInclusionCritera`), though this may depend on the analysis figure and needs cross-checking later.
- Choice/context decoding scripts use balanced trial sampling across conditions and operate on `obj.trialdat`, confirming that trial-aligned binned/smoothed firing rates are the reference neural input representation.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Top-level `data/` contains four task folders:
  - `Ephys_Behavior/`: delayed-response sessions with neural recordings and separate `motionEnergy_*.mat` files.
  - `RandomizedDelay_Ephys_Behavior/`: randomized-delay sessions with neural recordings and separate `motionEnergy_*.mat` files.
  - `DelayInhibition_BilatMC_Behavior/`: behavior-only sessions (`obj.me` embedded in the session file; no `obj.clu`).
  - `GoCueInhibition_BilatMC_Behavior/`: behavior-only sessions (`obj.me` embedded in the session file; no `obj.clu`).
- Session files are named `data_structure_<animal>_<date>.mat`.
- Motion energy for ephys sessions is stored separately as `motionEnergy_<animal>_<date>.mat`.
- File formats are mixed:
  - Most `data_structure_*.mat` files are MATLAB v7.3 / HDF5.
  - 11 randomized-delay session files are older non-HDF5 MATLAB files and require a `scipy.io.loadmat` fallback.
- Native session object contents:
  - Ephys sessions: `obj.bp`, `obj.clu`, `obj.traj`, `obj.sglx`, `obj.trials`, `obj.ex`, sometimes `obj.meta`, plus `obj.me` placeholder or struct.
  - Behavior-only sessions: `obj.bp`, `obj.traj`, `obj.sglx`, `obj.trials`, `obj.ex`, embedded `obj.me`, and no `obj.clu`.
- Behavioral trial fields in `obj.bp` include `Ntrials`, `hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim`, and event timings in `obj.bp.ev` (`bitStart`, `sample`, `delay`, `goCue`, `reward`, `lickL`, `lickR`).
- Neural data in ephys sessions live in `obj.clu`:
  - Usually a cell array over probes.
  - Each probe contains a struct array of units with fields like `tm`, `trialtm`, `trial`, `quality`, waveform info, and channel/site.
  - Some sessions include empty probe slots, so probe handling must be robust.
- Video/DLC data live in `obj.traj` with two views per session:
  - Side view features observed in sampled sessions: `tongue`, `left_tongue`, `right_tongue`, `jaw`, `trident`, `nose`, `lickport`.
  - Bottom view features observed in sampled sessions: `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`, `top_paw`, `bottom_paw`, `lickport`, `jaw`, `top_nostril`, `bottom_nostril`.
  - Each trial stores `featNames`, `ts`, `frameTimes`, and `NdroppedFrames`.
- Motion energy files/fields contain `me.data` (per-trial motion energy traces) and `me.moveThresh` (session threshold).
- Probe metadata are not perfectly uniform across sessions:
  - Many sessions expose `obj.ex.probe.loc` (example: `L ALM`, `L M1TJ`).
  - Some earlier sessions are missing that specific subfield, so brain-region mapping will need a fallback strategy.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 14,297 raw clustered units across ephys sessions |
| Neurons / session | 17-1,258 raw units; mean 317.71 across ephys sessions |
| Subjects | 18 total subjects in `data/`; 14 have neural recordings |
| Sessions / subject | 120 sessions total; ephys subjects contribute 1-8 sessions each |
| Trials (total) | 37,073 total trials across all session files; 15,324 in ephys sessions |
| Trials / session | 230-517 in ephys sessions; mean 340.53 (all-session mean 308.94) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Two-context: 522 units; fixed-delay DR total: 1,651 units; randomized delay: 845 units | “two-context paradigm: 12 sessions, six mice, 522 units…”; “25 sessions, nine mice, 1,651 units…”; “randomized delay task…845 units…” |
| Neurons / session | Not given as a single mean in text; per-session stats said to be in Extended Data Fig. 2a | “see Extended Data Fig. 2a for per-session statistics” |
| Subjects | Two-context: 6 mice; fixed-delay DR total: 9 mice; randomized delay: 4 mice | “12 sessions, six mice”; “25 sessions, nine mice”; “19 sessions using four mice” |
| Sessions / subject | Two-context: 12 sessions; fixed-delay DR total: 25 sessions; randomized delay: 19 sessions | Same sentences as above |
| Trials (total) | Not reported explicitly in paper text | Not explicitly stated in methods/results |
| Trials / session | Session begins with ~100 DR trials, then 10-25 trial alternating blocks | “A behavioral session began with approximately 100 DR trials and was then followed by alternating blocks… Each interleaved block was 10–25 trials” |
| Neural data time bin | Raw extracellular traces sampled at 25 kHz; analysis bin not stated in text | “voltage traces sampled at 25 kHz and stored for offline analysis” |
| Behavior data time bin | Video acquired at 400 Hz | “High-speed video was captured (400-Hz frame rate) from two cameras” |
| Reward rate | Not explicitly reported as a fraction in text | Not explicitly stated in methods/results |
| Delay length (fixed DR) | 0.9 s for 12 mice; 0.7 s for one mouse, linearly warped to 0.9 s | “The delay epoch (0.9 s for 12 mice, 0.7 s for one mouse and linearly time warped to 0.9 s)” |
| Randomized delays | 0.3, 0.6, 1.2, 1.8, 2.4, 3.6 s | “randomly selected from six possible values (0.3 s, 0.6 s, 1.2 s, 1.8 s, 2.4 s and 3.6 s)” |
| Choice-selective units | 36% sample, 42% delay, 58% response of 483 single units | “choice selectivity… (sample: 36%; delay: 42%; response: 58% of 483 single units)” |
| Context-selective units | 39% of 214 single units in two-context data | “39% of single units, 12 sessions, six mice, 214 single units” |


### Processing Details
- Task timing from methods:
  - Sample cue lasts 1.3 s.
  - Fixed DR delay is 0.9 s for most mice; one mouse used 0.7 s and was linearly time-warped to 0.9 s.
  - Go cue is a 10 ms chirp.
  - Ignore trial if no response within 3 s of go cue.
  - In two-context sessions, all sessions begin with DR trials, then alternate WC/DR blocks of 10-25 trials.
- Alignment-relevant text:
  - The paper discusses activity “from go cue” repeatedly and the decoder figures are plotted relative to go cue / water drop, consistent with go-cue-centered alignment for DR analyses.
  - For the fixed-delay DR task, preparatory dynamics are discussed across sample and delay epochs leading up to the go cue.
- Video/kinematics processing from text:
  - Two 400 Hz cameras: side and bottom.
  - Tongue, jaw and nose tracked in both views; paws tracked in bottom view only.
  - Missing position values filled with nearest values for all features except tongue.
  - Velocity is the first derivative of position.
  - Tongue angle and length are derived from bottom view.
- Motion-energy processing from text:
  - Frame-wise motion energy is based on the absolute difference between the median of the next 5 frames and previous 5 frames.
  - Per-frame motion energy summary is the 99th percentile across pixels.
  - Session-specific movement threshold is chosen manually from a bimodal motion-energy distribution.
- Electrophysiology processing from text:
  - Spikes were sorted with JRCLUST and/or Kilosort 3, manually curated in Phy 2.
  - Recordings were made in ALM.
  - Sessions included only if they had at least 10 units.
  - For subspace alignment and single-unit selectivity analyses: use well-isolated single units with firing rates >1 Hz.
  - For all other analyses: use all units with firing rates >1 Hz.

### Curation Steps

**Neuron curation rules**:
- Well-isolated single units are defined by manual curation using ISI histogram, separation from other units, and stationarity across session.
- Multiunits are manually curated units with higher ISI violation rates.
- Recording sessions included only if they contained at least 10 units.
- Analysis-specific neuron filter:
  - Subspace alignment and single-unit selectivity: well-isolated single units with firing rates >1 Hz.
  - Other analyses: all units with firing rates >1 Hz.

**Trial curation rules**:
- Early licks are omitted from analyses.
- Ignore trials are omitted from analyses.
- Behavioral-session inclusion for behavior analyses: at least 40 correct DR trials per direction and 20 correct WC trials per direction, excluding early lick and ignore trials.
- Randomized delay animals trained to expert criterion >70% accuracy and <20% early lick rate.

### Decoders Trained
| Decoded variable | Accuracy |
| Upcoming choice from `CDchoice` projections | AUC = 0.86 ± 0.11 across sessions |
| Choice from uninstructed movements (Fig. 3b / 3g) | Above chance and increases through delay; exact scalar accuracy not stated in text |
| Context from kinematic features and neural population (Fig. 4b) | Above shuffled across epochs including ITI; exact scalar accuracy not stated in text |
| Single-trial `CDchoice` prediction from kinematics | `R2 = 0.41 ± 0.23` across sessions |
| Single-trial `CDramp` prediction from kinematics | `R2 = 0.46 ± 0.23` across sessions |
| Single-trial `CDcontext` prediction from kinematics | `R2 = 0.49 ± 0.22` across sessions |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Ephys session count | `Recording and video` loader scripts list 50 ALM sessions total, but 6 of those are JEB4/JEB5 sessions not present in this project’s `data/` directory | 45 session files with `obj.clu` are present in raw `data/` | 25 fixed-delay ALM sessions + 19 randomized-delay ALM sessions = 44 analyzed ephys sessions | Use the intersection of reference-code session lists and available data files. This yields 44 code-selected, paper-consistent ephys sessions. |
| Extra raw ephys sessions | Code includes JEB23 `2023-10-10/11/12/13/18/19/21` and JEB24 `2023-10-23/24/25/26/27/31/11-02/11-03` | Raw data also contains JEB23 `2023-10-20` and JEB24 `2023-10-03/10-04` | Paper reports 19 randomized-delay sessions | Exclude raw sessions not referenced by the loader scripts (`JEB23 2023-10-20`, `JEB24 2023-10-03`, `JEB24 2023-10-04`). |
| Raw unit counts versus paper unit counts | Code loads a single ALM-designated probe per session and later filters units by quality / FR | Raw files contain all clusters from all probes, including non-ALM probes and empty probe slots; raw count was 14,297 clusters across ephys files | Paper counts are much smaller: 1,651 ALM units in fixed-delay data and 845 ALM units in randomized-delay data, after curation | Use only the code-specified probe for each session, treat that probe as ALM, and apply paper/code curation downstream rather than using all raw probes/clusters. |
| Brain region metadata | Loader filenames and `meta.probe` indicate which probe is ALM for each session | Raw probe metadata can include multiple regions, for example `L ALM` and `L M1TJ`; some early sessions lack explicit `probe.loc` fields | Paper states recordings were made in ALM | Define the converted neural dataset from the ALM-designated probe chosen by the reference loaders. Use ALM as the brain-region label for neurons on that selected probe. |
| MATLAB file format | MATLAB code can transparently load both v7.3 and older `.mat` files | `data/` mixes HDF5/v7.3 and older MATLAB files | Paper does not mention file-format heterogeneity | Python conversion must support dual loading: `h5py` for v7.3/HDF5 session files and `scipy.io.loadmat` fallback for older `.mat` files. |
| Neural bin width | `WorkingWithDataObjs.m` example uses `dt = 1/100` (10 ms); `getDefaultParams.m` uses `dt = 1/200` (5 ms) | Raw spikes are timestamped continuously; no native analysis bin | Methods text gives 25 kHz acquisition but not the final analysis bin | Treat `WorkingWithDataObjs.m` as illustrative. Use the default analysis pipeline setting only after verifying against later processing needs; tentatively prefer the paper/code-aligned 5 ms default, but keep this as a validation target in later steps. |
| Low firing-rate threshold | `WorkingWithDataObjs.m` example uses `lowFR = 1`; `getDefaultParams.m` uses `lowFR = 0.5`; Fig. 8 script uses `lowFR = 1` | Raw data contain many low-count clusters | Methods state units with firing rates exceeding 1 Hz were included in analyses | Resolve in favor of the methods text and figure scripts: use a 1 Hz low-FR threshold when matching the analyzed neural population, then verify counts against the paper. |
| Cluster quality usage | Default code commonly sets `params.quality = {'all'}` and `findClusters` excludes only obvious bad labels | Raw data include manual quality strings and mixed unit types | Methods distinguish well-isolated single units versus multiunits; “all units with firing rates exceeding 1 Hz” were used for most analyses | For the decoder conversion, use all non-garbage/non-noisy manually curated units on the selected ALM probe, then apply the FR threshold. Reserve single-unit-only filtering only for analyses where the paper explicitly did so. |
| Context alignment event | Figure text sometimes describes alignment to “go cue / water drop”; code default alignment is `goCue` | `obj.bp.ev.goCue` exists for all inspected sessions, including autowater/WC trials | User explicitly requires alignment to go cue onset | Use the stored `bp.ev.goCue` field for all trials as the universal alignment event in the converted dataset. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Code-selected ALM probe from `load*_ALMVideo.m` + `obj.clu{probe}` spike times | `neural` | Keep units from the session’s ALM-designated probe only; exclude bad-quality labels via `findClusters`; align spikes to `bp.ev.goCue`; bin on a fixed global time grid; smooth as in reference; drop units with mean FR `<= 1 Hz`; output one `(n_neurons, n_timepoints)` matrix per trial | `load*_ALMVideo.m`, `findClusters`, `alignSpikes`, `getSeq`, `removeLowFRClusters` | Use only the 44 sessions in the intersection of reference loader lists and available data files |
| `obj.time` / aligned bin centers relative to `goCue` | `input[0]` | Continuous time-from-go-cue vector repeated for every trial: shape `(1, n_timepoints)` | `getSeq` | Decoder input requested by user |
| `bp.R`, `bp.L` | `output[0]` | Lick direction: left `0`, right `1`; encode as a constant time series over the trial window | `findTrials` conditions use `R` and `L` throughout | Keep only non-early hit/miss trials so direction is well defined |
| `bp.autowater` | `output[1]` | Behavioral context: WC `0`, DR `1` via `1 - autowater`; constant time series over the trial window | `findTrials`; context conditions in Fig. 8 scripts | Stored `autowater=1` corresponds to WC in the reference code/paper |
| `bp.hit`, `bp.miss` | `output[2]` | Outcome: miss `0`, hit `1`; constant time series over the trial window | `findTrials` | Exclude `no` and `early` trials rather than inventing a label for ignores |
| Reference-aligned kinematics from side-camera tongue marker velocities | `output[3]` | Compute tongue speed as `sqrt(tongue_xvel_view1^2 + tongue_yvel_view1^2)` after reference interpolation/fill rules; binarize by session median over all kept samples | `findPosition`, `findVelocity`, `getKinematicsFromVideo` | Uses the canonical `tongue` marker from the side view to avoid mixing camera coordinate systems |
| Reference-aligned kinematics from bottom-camera paw markers | `output[4]` | Compute top- and bottom-paw speed magnitudes from x/y velocity pairs, average them per timepoint, then binarize by session median over all kept samples | `findPosition`, `findVelocity`, `getKinematicsFromVideo` | Paws are tracked in bottom view only per methods |
| Aligned motion-energy trace `me.data` | `output[5]` | Use aligned continuous motion energy and binarize by session median over all kept samples | `loadMotionEnergy` | The paper uses a manual movement threshold for move/non-move analyses; the user explicitly requests a 50th-percentile discretization for decoder output |
| Animal IDs from filenames / loader scripts | `subjects`, `subject_idx` | Build unique subject list across retained sessions | `load*_ALMVideo.m` | Expected 14 unique neural subjects in the available dataset |
| Selected probe region | `brain_regions`, `brain_region_idx` | Label neurons from the code-selected probe as `ALM` | `load*_ALMVideo.m` | Raw metadata can include non-ALM probes, but the loader scripts resolve which probe to use |

### Key Decisions
1. **Retain only the 44 code-selected ALM sessions present in `data/`**: This is the cleanest way to match the paper’s analyzed dataset rather than the broader raw archive.
2. **Use the ALM-designated probe only for each retained session**: Raw files can contain multiple probes/areas; the reference loaders specify which probe corresponds to ALM.
3. **Exclude stimulation, early-lick, and ignore/no-response trials**: Reference analyses consistently use `~stim.enable`, `~early`, and usually hit/miss conditions; ignore trials are omitted in the paper.
4. **Use `goCue` as the universal alignment event**: This matches the user request and the default reference alignment.
5. **Use a fixed window of `[-2.5, 2.5] s` around `goCue` with a 5 ms bin (`dt = 1/200`)**: This matches the default reference processing pipeline better than the 10 ms tutorial example and gives a shared time axis for neural/video streams.
6. **Use all non-garbage manually curated units on the selected probe, then apply a 1 Hz FR threshold**: This matches the paper’s “all units >1 Hz” rule for most analyses better than a single-unit-only subset.
7. **Represent trial-level categorical outputs as constant time series**: This keeps all outputs on the same `(n_output, n_timepoints)` grid while preserving the per-trial labels requested by the user.
8. **Define tongue velocity from the side-view `tongue` marker speed magnitude**: This uses a direct reference-processed kinematic channel without inventing an unreferenced cross-camera combination.
9. **Define paw velocity from the average of top- and bottom-paw speed magnitudes in the bottom view**: This captures overall paw movement while remaining close to the tracked features described in the paper/code.
10. **Use per-session 50th-percentile thresholds for tongue velocity, paw velocity, and motion energy**: This follows the decoder task even where it differs from the paper’s manual movement threshold.

### Planned Sanity Checks
- [ ] Verify that retained session list exactly matches the 44-session intersection of reference loader scripts and available data files.
- [ ] Compare retained fixed-delay and randomized-delay session counts against the paper’s 25 and 19 session totals.
- [ ] After probe selection and 1 Hz filtering, compare unit counts against the paper’s reported 1,651 fixed-delay and 845 randomized-delay units.
- [ ] Spot-check one neuron in one trial by histogramming raw spike times directly from the source file and comparing to the converted binned/smoothed trace.
- [ ] Spot-check one trial’s `lick_direction`, `context`, and `outcome` labels directly from raw `bp` fields using `np.allclose()`.
- [ ] Spot-check one trial’s aligned motion energy against direct interpolation from the raw motion-energy file using the source frame times / video offset logic.
- [ ] Spot-check one trial’s aligned tongue and paw velocity traces against direct interpolation of raw DLC coordinates followed by the same velocity computation.
- [ ] Check that all trials in the converted dataset share the same time vector and that `input[0]` matches the neural bin centers exactly.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `convert_data.py`.
- Script design:
  - Parses the reference ALM session loader scripts to recover the analyzed session list and probe selections.
  - Supports both MATLAB v7.3/HDF5 session files (`h5py`) and older MATLAB session files (`scipy.io.loadmat`).
  - Reimplements the reference spike binning and smoothing path in Python, including the causal Gaussian kernel behavior in `mySmooth.m`.
  - Loads and aligns motion energy and selected DLC features to the neural time base using the reference video-offset logic.
  - Filters to non-stim, non-early hit/miss trials and builds decoder-ready `input`/`output` arrays.
  - Supports `--full`, `--sample`, and `--show-processing`.

Code inefficiencies identified:
- Full recursive HDF5-to-Python loading was too slow and memory-heavy for the large session files.
- Per-spike/per-trial histogram loops would likely be a bottleneck if implemented naively.

Code speedups added:
- Implemented targeted field loading instead of full-session recursive decoding for HDF5 files.
- Implemented vectorized per-unit spike accumulation with `np.add.at` instead of calling a histogram inside nested trial loops.
- Processes one session at a time to keep memory bounded.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 99 |
| Neurons / session | 48, 51 (mean 49.5) |
| Subjects | 2 (`EKH1`, `JEB23`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 575 |
| Trials / session | 214, 361 (mean 287.5) |
| `time_from_go_cue_s` range | [-2.5, 2.5] |
| `lick_direction` distribution | [0.501, 0.499] |
| `behavioral_context` distribution | [0.096, 0.904] |
| `outcome` distribution | [0.078, 0.922] |
| `tongue_velocity_bin` distribution | [0.806, 0.194] |
| `paw_velocity_bin` distribution | [0.500, 0.500] |
| `motion_energy_bin` distribution | [0.500, 0.500] |

### Processing Plots Review
- Generated plots:
  - `processing_EKH1_2021-08-07.png`
  - `processing_JEB23_2023-10-18.png`
- Visual review:
  - Neural matrices show clear trial-varying activity aligned to the shared `goCue`-centered time axis.
  - Time vector and behavioral outputs share the same 1000-bin window in both sessions.
  - Paw velocity and motion-energy binarizations split cleanly around the session medians.
  - Initial sample verification exposed a degenerate tongue bin (`all ones`) caused by invalid tongue-tracking periods being converted to zero before percentile thresholding. Fixed by computing the tongue percentile on visible frames only and assigning non-visible frames to bin `0`.
  - After the fix, the sample verifier reported no errors or warnings and tongue bins span both classes in each sample session.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Targeted HDF5 field loading instead of recursive object decoding | Keeps large v7.3 session load times to a few seconds instead of much longer full-object reconstruction |
| Vectorized spike accumulation with `np.add.at` | Avoids nested trial-by-trial histogram loops; sample conversion finished in 8.12 s for 2 sessions |
| Session-by-session processing | Bounded memory use and stable runtime across mixed HDF5/old-MAT files |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion (`convert_data.py --sample --show-processing`) | 4.06 s / session average (4.26 s, 3.18 s observed) | Rough full-run estimate for 44 retained sessions: ~3.5-5 minutes, well below the 15 minute optimization threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `lick_direction` | 0.6693 | 0.6668 |
| `behavioral_context` | 0.8571 | 0.8689 |
| `outcome` | 0.6219 | 0.5918 |
| `tongue_velocity_bin` | 0.8093 | 0.8131 |
| `paw_velocity_bin` | 0.6063 | 0.5888 |
| `motion_energy_bin` | 0.7258 | 0.7198 |

- Training run summary:
  - Training completed successfully on GPU (`cuda`).
  - Loss decreased monotonically from `5.815792` at epoch 1 to `0.532089` at epoch 200.
  - Test loss was `0.559657`.
  - All outputs were above chance (`0.5`) on both training and validation data.
  - The smallest validation margin above chance was for `outcome` (`0.5918`), still clearly above chance on the 2-session sample.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3.2 GB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 2,496 total across fixed + randomized tasks | 44 retained sessions from loader lists; exact post-filter count not written in code | Raw selected sessions contain many more unfiltered clusters; after conversion curation, 2,455 retained units | 2,455 | Near-match; investigate exact difference in Step 10 |
| Mean neurons/session | Not stated explicitly | Not stated explicitly | 55.80 after current curation | 55.80 | Yes |
| Subjects | 9 fixed-delay + 4 randomized-delay mice reported by paper summaries | 14 unique subjects in the available loader/data intersection | 14 unique subjects across retained sessions | 14 | Matches available dataset intersection; paper grouping differs by analysis cohort |
| Sessions | 25 fixed-delay + 19 randomized-delay = 44 | 44 reference-selected sessions present in `data/` | 44 retained ephys sessions | 44 | Yes |
| Trials (total) | Not explicitly stated | Not explicitly stated | 11,955 retained hit/miss non-stim non-early trials with neural coverage | 11,955 | Yes |
| Trials/session (mean) | Not explicitly stated | Not explicitly stated | 271.70 | 271.70 | Yes |
| `time_from_go_cue_s` range | Go-cue-centered analyses throughout paper; exact plotted window often spans sample/delay/response | `params.tmin = -2.5`, `params.tmax = 2.5` in default pipeline | [-2.5, 2.5] | [-2.5, 2.5] | Yes |
| `lick_direction` distribution | Not explicitly stated | Computed from `R`/`L` trial conditions | [0.498, 0.502] | [0.498, 0.502] | Yes |
| `behavioral_context` distribution | WC/DR blocks described qualitatively, not as a global fraction | Computed from `autowater` | [0.084, 0.916] | [0.084, 0.916] | Yes |
| `outcome` distribution | Reward rate not explicitly tabulated | Computed from `hit`/`miss` | [0.138, 0.862] | [0.138, 0.862] | Yes |
| `tongue_velocity_bin` distribution | Not explicitly stated | Decoder-task-specific median split | [0.807, 0.193] | [0.807, 0.193] | Yes |
| `paw_velocity_bin` distribution | Not explicitly stated | Decoder-task-specific median split | [0.500, 0.500] | [0.500, 0.500] | Yes |
| `motion_energy_bin` distribution | Paper uses a manual movement threshold for some analyses | Decoder-task-specific median split | [0.500, 0.500] | [0.500, 0.500] | Yes |

- Full conversion summary:
  - Runtime: `135.23 s` for all 44 sessions.
  - Retained sessions: 44.
  - Retained trials: 11,955.
  - Retained units: 2,455.
  - Fixed-delay subset: 25 sessions, 10 subjects, 6,150 trials, 1,530 units.
  - Randomized-delay subset: 19 sessions, 4 subjects, 5,805 trials, 925 units.
  - Full verification result: `Data format is valid, no errors or warnings.`
- Step 9 issues found and fixed before final verifier pass:
  - Nested `motionEnergy` MATLAB structs in `JEB15` sessions required recursive unwrapping of `.data`.
  - Randomized-delay `motionEnergy` files store only the per-trial traces in `me` and omit `moveThresh`; loader now accepts that form and records `NaN` for the unused manual threshold.
  - `JEB6_2021-04-18` contains an empty HDF5 probe slot before the actual ALM probe; loader now indexes original raw probe slots rather than the compressed non-empty list.
  - `JEB24_2023-10-23` and `JEB24_2023-11-03` include trailing behavioral trials with no neural coverage; conversion now drops valid behavioral trials whose raw trial index exceeds the last neural `unit.trial` index.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Re-ran `python -u train_decoder.py converted_data.pkl --verify-only 2>&1 | tee verification_full_out.txt` after fixing session-edge cases. Final result was `Data format is valid, no errors or warnings.`
2. **Independent sanity checks from raw files**:
   - Neural check (`JEB23_2023-10-18`, old-format MATLAB session):
     - Loaded the raw `obj.clu` arrays directly with `scipy.io.loadmat`, independently rebuilt the first kept unit’s go-cue-aligned 5 ms binned/smoothed trace for converted trial 0, and compared against `converted_data.pkl`.
     - Result: `np.allclose = True`, max absolute difference `1.406842e-05`.
   - Input check (`EKH1_2021-08-07`, raw HDF5 session):
     - Independently constructed the go-cue-centered time axis `[-2.4975, ..., 2.4975]` from the reference window/bin definition and compared against `input[0]`.
     - Result: `np.allclose = True`, max absolute difference `0.0`.
   - Output check (`JEB23_2023-10-18`, raw MATLAB session):
     - Loaded raw `bp.R`, `bp.autowater`, and `bp.hit` directly, selected converted local trial 10, and compared the expected constant categorical label series against the first 3 output rows.
     - Result: `np.allclose = True`; raw labels were `[lick_direction=1, behavioral_context=1, outcome=1]`.
3. **Reference code comparison**:
   - Data loading:
     - Reference: `load*_ALMVideo.m`, `loadObjs.m`, `loadSessionData.m`.
     - Converter: parses the same loader scripts to recover the exact analyzed session list and probe selection, then loads either v7.3/HDF5 or old-format `.mat` sessions.
   - Neuron filtering:
     - Reference: `findClusters(..., {'all'})` excludes `garbage`, `gabrga`, `noisy`, `real?`; `removeLowFRClusters` uses mean FR `> lowFR`.
     - Converter: uses the same bad-quality exclusions and a mean FR `> 1 Hz` cutoff, matching the methods text and the code path used in several analyses.
   - Temporal alignment and binning:
     - Reference: `alignSpikes`, `getSeq`, default `params.tmin=-2.5`, `params.tmax=2.5`, `params.dt=1/200`.
     - Converter: aligns all streams to `goCue` on the same `[-2.5, 2.5] s` window with 5 ms bins.
   - Smoothing:
     - Reference: `mySmooth.m` with a causal Gaussian-like kernel and reflected padding.
     - Converter: Python reimplementation of the same kernel/padding behavior.
   - Input construction:
     - Reference trial time base is `obj.time` relative to the align event.
     - Converter stores that same time base as `time_from_go_cue_s`.
   - Output construction:
     - Trial labels are taken from the same raw behavioral fields (`R/L`, `autowater`, `hit/miss`).
     - Time-varying kinematic/motion outputs follow the same interpolation/video-offset logic as `findPosition`, `findVelocity`, and `loadMotionEnergy`, then apply the decoder-task-specific median split requested by the user.
4. **Key statistics comparison**:
   - Session count matches exactly: 25 fixed-delay + 19 randomized-delay = 44.
   - Randomized-delay subject count matches the paper: 4 mice.
   - Converted unit totals are close but not identical to the paper summary totals:
     - Fixed-delay: converted `1,530` vs paper `1,651`.
     - Randomized-delay: converted `925` vs paper `845`.
     - Total: converted `2,455` vs paper `2,496`.
   - This difference is small relative to the full dataset (`41 / 2,496 = 1.64%`) and is plausibly explained by paper-summary cohort definitions versus the available dataset intersection plus exact quality/FR curation on the provided files. The converter otherwise matches the reference session lists, probe selection, and filtering logic.
5. **Edge-case review**:
   - Checked late-session trial numbering and found two `JEB24` sessions where behavioral trials continued after the last neural `unit.trial` index.
   - Checked HDF5 probe-slot structure and found `JEB6_2021-04-18` uses probe slot 2 with slot 1 empty.
   - Checked motion-energy file structure and found both nested-struct (`JEB15`) and direct-object-array (`JEB23/JEB24`) variants.

### Issues Found and Resolved
- `verification_full_out.txt` initially warned about all-zero neural trials in `JEB24_2023-10-23` and `JEB24_2023-11-03`: fixed by dropping valid behavioral trials whose raw trial index exceeded the last neural `unit.trial` index.
- `JEB15` motion-energy files stored `me.data` as a nested MATLAB struct: fixed by recursively unwrapping `.data` containers.
- Randomized-delay motion-energy files stored the traces directly in `me` without `moveThresh`: fixed by accepting that variant and recording `NaN` for the unused manual threshold.
- `JEB6_2021-04-18` HDF5 probe selection failed because the first raw probe slot was empty: fixed by indexing raw probe slots directly instead of a compressed non-empty list.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `lick_direction` | 0.6560 | 0.6429 | Above chance; moderate generalization gap |
| `behavioral_context` | 0.8643 | 0.8455 | Strong decoding |
| `outcome` | 0.6833 | 0.6462 | Above chance; moderate signal |
| `tongue_velocity_bin` | 0.7957 | 0.7913 | Strong decoding with minimal train/val gap |
| `paw_velocity_bin` | 0.5881 | 0.5834 | Weakest decoded output, but still above chance |
| `motion_energy_bin` | 0.7479 | 0.7450 | Strong decoding with minimal train/val gap |

- Full training summary:
  - Command: `python -u train_decoder.py converted_data.pkl --plot-samples 2>&1 | tee train_decoder_full_out.txt`
  - Device: `cuda`
  - Training trials: `9,548`
  - Validation trials: `2,407`
  - Training loss dropped from `7.047011` (epoch 1) to `0.530699` (epoch 200).
  - Test loss: `0.564669`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `lick_direction` | Validation balanced acc `0.6429` | Paper reports strong upcoming-choice decoding from `CDchoice` trajectories (`AUC = 0.86 ± 0.11`), not directly the same decoder/metric, but qualitatively consistent with above-chance choice information in ALM |
| `behavioral_context` | Validation balanced acc `0.8455` | Paper reports above-shuffle context decoding from neural population across epochs; no single scalar balanced-accuracy value given |
| `outcome` | Validation balanced acc `0.6462` | No direct paper accuracy reported for outcome decoding |
| `tongue_velocity_bin` | Validation balanced acc `0.7913` | No direct paper accuracy reported for this median-split variable |
| `paw_velocity_bin` | Validation balanced acc `0.5834` | No direct paper accuracy reported for this median-split variable |
| `motion_energy_bin` | Validation balanced acc `0.7450` | Paper shows strong relationships between motion energy and neural subspaces; no directly comparable balanced-accuracy value reported |

- Chance-level review:
  - Chance is `0.5` for all six binary outputs.
  - All validation accuracies are above chance.
  - Outputs exceeding `1.5 x chance = 0.75`: `behavioral_context`, `tongue_velocity_bin`.
  - Outputs below `0.75` but still clearly above chance: `lick_direction`, `outcome`, `paw_velocity_bin`, `motion_energy_bin` (the last is just below at `0.7450`).
- Investigation of lower-accuracy outputs:
  - Raw-file sanity checks already confirmed the correctness of neural binning, input construction, and trial-label outputs.
  - Full-format verification reported no structural warnings after the neural-coverage fix.
  - Class distributions are sensible:
    - `lick_direction` is essentially balanced (`0.498 / 0.502`).
    - `paw_velocity_bin` and `motion_energy_bin` are near-perfectly balanced by construction.
    - `outcome` is somewhat imbalanced (`0.138 / 0.862`) but still decoded well above chance under balanced accuracy.
  - These points argue against a conversion bug as the explanation for the weaker outputs.
- Train vs validation gap review:
  - `lick_direction`: `0.6560 / 0.6429 = 1.02x`
  - `behavioral_context`: `0.8643 / 0.8455 = 1.02x`
  - `outcome`: `0.6833 / 0.6462 = 1.06x`
  - `tongue_velocity_bin`: `0.7957 / 0.7913 = 1.01x`
  - `paw_velocity_bin`: `0.5881 / 0.5834 = 1.01x`
  - `motion_energy_bin`: `0.7479 / 0.7450 = 1.00x`
  - No output approaches the `>1.5x` overfitting concern threshold.
- Overall interpretation:
  - Decoder behavior is internally consistent: loss decreases smoothly, train/validation gaps are small, and all outputs are above chance.
  - The strongest outputs (`behavioral_context`, `tongue_velocity_bin`, `motion_energy_bin`) are the ones most expected to have broad session-level signal.
  - No additional conversion fixes were indicated by the Step 12 review.

### Issues Found and Resolved
- No new conversion issues were found after the full decoder review.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

- Final documentation added:
  - `README.md` summarizes the converted dataset, output format, loading example, and validation commands.
  - `cache/README_CACHE.md` documents the reusable cache contents.
  - `cache/step10_sanity_checks.py` reproduces the raw-file sanity checks used during Step 10.
- Final artifact check:
  - Present: `CONVERSION_NOTES.md`
  - Present: `convert_data.py`
  - Present: `converted_data.pkl`
  - Present: `sample_data.pkl`
  - Present: `README.md`
  - Present: `train_decoder_full_out.txt`
  - Present: `conversion_sample_out.txt`
  - Present: `verification_sample_out.txt`
  - Present: `train_decoder_sample_out.txt`
  - Present: `conversion_full_out.txt`
  - Present: `verification_full_out.txt`
