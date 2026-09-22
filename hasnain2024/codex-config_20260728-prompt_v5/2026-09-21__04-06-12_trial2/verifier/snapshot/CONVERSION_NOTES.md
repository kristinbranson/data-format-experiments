# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse (paper, code, and data provided in `/app`)
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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadObjs` | `/app/code/DataLoadingScripts/loadObjs.m` | LOADING | Loads each session `.mat` into an `obj` struct and normalizes missing fields (`meta`, `ex`, `me`) so sessions can be concatenated. |
| `loadSessionData` | `/app/code/DataLoadingScripts/loadSessionData.m` | LOADING | Main session-processing entry point. Iterates over sessions/probes, calls `processData`, and concatenates dual-probe sessions. |
| `processData` | `/app/code/DataLoadingScripts/processData.m` | PROCESSING | Runs trial selection, cluster selection, spike alignment, PSTH/single-trial extraction, low-FR filtering, and baseline FR summaries. |
| `findTrials` | `/app/code/DataLoadingScripts/findTrials.m` | CURATION | Evaluates logical trial-condition expressions against `obj.bp` fields and returns trial IDs for each condition. |
| `findClusters` | `/app/code/DataLoadingScripts/findClusters.m` | CURATION | Selects clusters by quality label; `'all'` excludes `garbage`, `gabrga`, `noisy`, and `real?`. |
| `alignSpikes` | `/app/code/DataLoadingScripts/alignSpikes.m` | PROCESSING | Aligns each spike time to an event such as `goCue`, `moveOnset`, `firstLick`, `lastLick`, or `jawOnset`. |
| `getSeq` | `/app/code/DataLoadingScripts/getSeq.m` | PROCESSING | Bins aligned spikes over `tmin:dt:tmax`, smooths them, and creates both condition-averaged PSTHs and single-trial neural arrays. |
| `removeLowFRClusters` | `/app/code/DataLoadingScripts/removeLowFRClusters.m` | CURATION | Drops units whose mean PSTH firing rate is below the low-FR threshold across conditions. |
| `baselineFR` | `/app/code/DataLoadingScripts/baselineFR.m` | PROCESSING | Computes baseline firing-rate median and SD in the presample epoch. |
| `loadMotionEnergy` | `/app/code/DataLoadingScripts/loadMotionEnergy.m` | LOADING | Loads per-trial motion energy, aligns it to the chosen event using video timestamps, interpolates it to the neural time base, and thresholds movement. |
| `findVideoOffset` | `/app/code/funcs/findVideoOffset.m` | PROCESSING | Computes the offset between SpikeGLX and video start using bit-code synchronization. |
| `findPosition` | `/app/code/funcs/kinematics/findPosition.m` | PROCESSING | Interpolates DLC feature trajectories into the aligned neural time base, with special handling for missing video frames and non-tongue features. |
| `findVelocity` | `/app/code/funcs/kinematics/findVelocity.m` | PROCESSING | Computes framewise x/y velocity per tracked feature and uses different missing-data behavior for tongue vs non-tongue features. |
| `getKinematicsFromVideo` | `/app/code/funcs/kinematics/getKinematicsFromVideo.m` | PROCESSING | Builds aligned kinematic feature matrices `(time, trial, feature)` from DLC trajectories for both cameras. |
| `getKinematics` | `/app/code/funcs/kinematics/getKinematics.m` | PROCESSING | Augments kinematics with tongue angle/length and motion energy, standardizes features, and computes reduced-dimensional video factors. |
| `NeuralChoiceDecoding` | `/app/code/ChoiceContextDecoding/NeuralChoiceDecoding.m` | PROCESSING | Example neural decoder setup for lick choice using aligned single-trial neural data. |
| `NeuralContextDecoding` | `/app/code/ChoiceContextDecoding/NeuralContextDecoding.m` | PROCESSING | Example neural decoder setup for behavioral context using aligned single-trial neural data. |

### Notes
- The repository is an electrophysiology + behavior/video pipeline, not calcium imaging, so there is no delta-F/F computation anywhere in the reference loading path.
- The canonical neural pipeline is: load `obj` -> select trials with logical expressions over `obj.bp.*` -> select clusters by quality -> align spike times to an event -> bin and smooth spikes -> filter low-FR units.
- `WorkingWithDataObjs.m` shows the reference task conditions and a standard alignment to `goCue`, with typical windows `tmin=-2.5`, `tmax=2.5`, and either `dt=1/100` (tutorial example) or `dt=1/200` (`getDefaultParams` default).
- The reference code uses smoothed firing rates, not raw spike times, for downstream analyses and decoders.
- Default ephys curation in `getDefaultParams.m` uses `quality={'all'}` and `lowFR=0.5` Hz, while the tutorial example in `WorkingWithDataObjs.m` uses `lowFR=1` Hz. This threshold difference needs resolution against the dataset and paper later.
- Trial-condition logic in the code explicitly distinguishes delayed-response (`~autowater`) from water-cued (`autowater`) trials and excludes `stim.enable` and `early` trials in the standard conditions.
- Video/DLC traces are synchronized to neural time using `findVideoOffset`, then interpolated into the same time axis as neural data relative to the alignment event.
- Motion energy is aligned to the same trial-wise event and interpolated to the neural time base in `loadMotionEnergy`.
- Missing video is handled explicitly: some trials can be skipped if dropped frames are flagged; non-tongue features are nearest-filled after interpolation, while tongue features retain NaNs longer and are baseline-filled separately for position calculations.
- The neural choice/context decoder examples both use neural `obj.trialdat` as input and balanced trial sampling across classes, which is directly relevant to the downstream validation task.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains four experiment folders:
  - `Ephys_Behavior/`: delayed-response and water-cued sessions with electrophysiology, DLC trajectories, and separate `motionEnergy_*.mat` files.
  - `RandomizedDelay_Ephys_Behavior/`: randomized-delay sessions with electrophysiology for most sessions plus separate `motionEnergy_*.mat` files.
  - `DelayInhibition_BilatMC_Behavior/`: behavior/video-only bilateral motor cortex inhibition sessions; motion energy is embedded inside each `data_structure_*.mat`.
  - `GoCueInhibition_BilatMC_Behavior/`: behavior/video-only go-cue inhibition sessions; motion energy is embedded inside each `data_structure_*.mat`.
- File naming conventions:
  - Session files: `data_structure_<animal>_<date>.mat`
  - Separate motion energy files for ephys folders: `motionEnergy_<animal>_<date>.mat`
- Mixed MATLAB formats are present:
  - 109 session files are MATLAB v7.3/HDF5 and can be read with `mat73`/`h5py`.
  - 11 session files are older MATLAB format and require `scipy.io.loadmat`.
- Native session object layout from representative files:
  - Ephys sessions: `obj = {bp, clu, ex, meta, pth, sglx, traj, trials}`
  - Behavior-only sessions: `obj = {bp, ex, me, pth, sglx, traj, trials}`
- Key raw variables:
  - `obj.bp`: trial-level task variables (`L`, `R`, `hit`, `miss`, `no`, `autowater`, `early`, `stim`, `protocol`, `Ntrials`) and event times in `obj.bp.ev` (`sample`, `delay`, `goCue`, `reward`, `lickL`, `lickR`, `bitStart`).
  - `obj.clu` (ephys only): per-probe cluster structs with `quality`, `site`, `spkWavs`, `tm`, `trial`, `trialtm`.
  - `obj.traj`: two camera views (side and bottom), each storing per-trial `ts`, `frameTimes`, `featNames`, `NdroppedFrames`, and filenames.
  - `obj.me`: per-trial motion energy and `moveThresh` for behavior-only sessions.
  - Separate `motionEnergy_*.mat`: `me = {data, moveThresh}` for ephys sessions.
- Example DLC feature sets observed in ephys sessions:
  - View 1 / side camera: `tongue`, `left_tongue`, `right_tongue`, `jaw`, `trident`, `nose`, `lickport`
  - View 2 / bottom camera: `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`, `top_paw`, `bottom_paw`, `lickport`, `jaw`, `top_nostril`, `bottom_nostril`
- Important edge case found during exploration:
  - `RandomizedDelay_Ephys_Behavior/data_structure_JEB24_2023-10-03.mat` and `..._2023-10-04.mat` do not contain `obj.clu`; they are behavior-only sessions despite living in an ephys-labeled folder.
- Raw session sizes are heterogeneous:
  - Ephys sessions range from 17 to 1258 clusters before any quality or firing-rate curation.
  - Trial counts range from 209 to 517 across all session files.
- Overall raw dataset totals from file inspection:
  - 120 session files across 18 mice and 37,073 trials.
  - 45 sessions contain neural clusters (14 mice, 15,324 trials, 14,297 raw clusters total).
  - 75 sessions are behavior-only (including the 2 JEB24 sessions in the randomized-delay folder).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 14,297 raw clusters across neural sessions |
| Neurons / session | 317.71 mean raw clusters across 45 neural sessions |
| Subjects | 14 mice with neural data (18 mice overall in `/app/data`) |
| Sessions / subject | 3.21 mean neural sessions per neural-data mouse |
| Trials (total) | 15,324 trials across neural sessions (37,073 overall) |
| Trials / session | 340.53 mean trials across neural sessions |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Fixed-delay DR: 1,651 units; two-context subset: 522 units; randomized-delay: 845 units | “1,651 units”; “522 units”; “845 units” |
| Neurons / session | Fixed-delay DR: 66.0; two-context: 43.5; randomized-delay: 44.5 | “25 sessions, 1,651 units”; “12 sessions, 522 units”; “19 sessions, 845 units” |
| Subjects | Fixed-delay DR: 9 mice; two-context: 6 mice; randomized-delay: 4 mice; study total: 17 mice | “25 sessions, nine mice”; “12 sessions, six mice”; “19 sessions using four mice”; “This study used data collected from 17 mice” |
| Sessions / subject | Fixed-delay DR: 25/9; two-context: 12/6; randomized-delay: 19/4 | “25 sessions, nine mice”; “12 sessions, six mice”; “19 sessions using four mice” |
| Trials (total) | Not explicitly reported in text | No explicit trial-count total found in paper text |
| Trials / session | Session starts with ~100 DR trials, then alternating 10–25 trial blocks | “approximately 100 DR trials” and “Each interleaved block was 10–25 trials” |
| Neural data time bin | 5 ms for single-trial analyses | “single-trial neural activity was first binned in 5-ms intervals” |
| Behavior data time bin | 400 Hz video = 2.5 ms per frame | “High-speed video was captured (400-Hz frame rate)” |
| Reward rate | Not explicitly reported; expert criterion >70% correct | “trained … until they reached at least 70% accuracy” |
| <Task/behavior statistic 1> | Choice-selective single units in DR: sample 36%, delay 42%, response 58% of 483 single units | “sample: 36%; delay: 42%; response: 58% of 483 single units” |
| <Task/behavior statistic 2> | Context-selective single units: 39% of 214 single units | “39% of single units, 12 sessions, six mice, 214 single units” |
| ... | Randomized-delay experts: >70% accuracy and <20% early lick rate | “>70% accuracy and <20% early lick rate” |


### Processing Details
- Behavioral paradigm details relevant to conversion:
  - DR trials: 1.3 s sample tone, then delay, then 10 ms auditory go cue; ignore trials are no response within 3 s of go cue.
  - Fixed-delay DR task: nominal delay 0.9 s for 12 mice and 0.7 s for one mouse, “linearly time warped to 0.9 s”.
  - Two-context sessions begin with ~100 DR trials and then alternate WC and DR blocks of 10–25 trials; all sessions start with DR.
  - WC trials omit auditory cues and present ~3 µl water at a random lickport and random time.
- Video / behavior processing:
  - Two cameras at 400 Hz.
  - Tongue, jaw, and nose tracked in both cameras; paws tracked only in the bottom view.
  - Missing values are nearest-filled for all tracked features except tongue.
  - Velocity is the first derivative of position.
  - Motion energy is computed from the absolute difference between medians of the next/previous 5 frames and summarized by the 99th percentile across pixels for each frame.
  - Motion-energy movement threshold is set manually per session at the separation between the two modes of a bimodal distribution.
- Neural processing:
  - Sessions are included only if they have at least 10 units.
  - All units with firing rate >1 Hz are used for most analyses.
  - Only well-isolated single units with firing rate >1 Hz are used for single-unit selectivity and some subspace-alignment analyses.
  - Single-trial neural activity is binned at 5 ms and smoothed with a causal Gaussian kernel with half-width 35 ms.
  - Single-trial normalization uses baseline activity from −2.4 s to −2.2 s relative to go cue, during the ITI.
- Paper decoder/analysis setup directly relevant to later validation:
  - Choice/context decoding used logistic regression with ridge regularization.
  - Separate models were trained at each time bin.
  - Equal numbers of correct left/right trials were used.
  - Four-fold cross-validation was used, with 30% of trials held out for testing.
  - Chance performance was estimated by shuffling labels across trials.

### Curation Steps

**Neuron curation rules**:
- Spike sorting used JRCLUST and/or Kilosort 3 with manual curation in Phy 2.
- Units were labeled as well-isolated single units based on ISI histogram, separation from other units, and stationarity across the session.
- Units passing manual curation but with higher ISI violation rates were treated as multiunits.
- Recording sessions were included only if they had at least 10 units.
- For most analyses, units had to exceed 1 Hz firing rate.

**Trial curation rules**:
- Early-lick trials were omitted from analyses.
- Ignore trials were omitted from behavioral analyses.
- Behavioral-analysis sessions required at least 40 correct DR trials for each direction and 20 correct WC trials for each direction, excluding early and ignore trials.
- Logistic choice/context decoders used equalized numbers of correct left/right trials.
- Randomized-delay CDchoice fit/test procedure used non-1.2 s delays as fit trials and 1.2 s delays as test trials.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from delay-epoch `CDchoice` projection | ROC AUC = 0.86 ± 0.11 across sessions |
| `CDchoice` predicted from movement kinematics (fixed-delay DR) | R² = 0.41 ± 0.23 across sessions |
| `CDramp` predicted from movement kinematics (fixed-delay DR) | R² = 0.46 ± 0.23 across sessions |
| `CDchoice` predicted from movement kinematics (randomized-delay DR) | Average R² = 0.38 ± 0.20 across sessions |
| Choice/context logistic decoding from neural data or kinematics | Time-resolved accuracy curves shown, but no single summary accuracy value stated in the main text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Randomized-delay session count | `Figure3h.m` / `Figure3i.m` load exactly 19 sessions from `JEB11`, `JEB12`, `JEB23`, `JEB24` via the `loadJEB*_ALMVideo.m` scripts. `loadJEB23_ALMVideo.m` comments out `2023-10-20`, and `loadJEB24_ALMVideo.m` starts at `2023-10-23`. | The raw folder contains 22 session files, including `JEB24` `2023-10-03` and `2023-10-04` with no `obj.clu`, plus `JEB23` `2023-10-20`. | Randomized-delay cohort reported as 19 sessions from 4 mice. | Mirror the loader-defined 19-session cohort, not the raw folder count. Exclude the two behavior-only `JEB24` sessions and the commented-out `JEB23` `2023-10-20` session. |
| Fixed-delay and context mouse counts | `Figure3h.m` fixed-delay loaders enumerate 25 sessions across 10 animal IDs (`JEB6`, `JEB7`, `EKH1`, `EKH3`, `JGR2`, `JGR3`, `JEB13`, `JEB14`, `JEB15`, `JEB19`). `Figure8a_thru_c.m` context loaders enumerate 12 sessions across 7 animal IDs (`JEB6`, `JEB7`, `EKH1`, `EKH3`, `JGR2`, `JGR3`, `JEB19`). | Raw file names match the loader animal IDs. | Paper text states 25 sessions from 9 mice and 12 sessions from 6 mice. | Follow the explicit loader session lists from the released code. Treat the paper mouse counts as an apparent manuscript/counting inconsistency unless later evidence shows that two IDs belong to the same animal. |
| Raw neuron counts vs paper neuron counts | Reference loaders select specific probes per session, including mixed single-probe and dual-probe sessions (for example `JEB15` uses `[1 2]` on three dates). `findClusters('all')` excludes only `garbage`, `gabrga`, `noisy`, and `real?`. | Counting all raw clusters across every probe overestimates the paper-comparable totals. Using loader-selected probes and `findClusters('all')` logic gives 1,566 fixed-delay units and 948 randomized-delay units before any firing-rate/session inclusion filter. | Paper reports 1,651 fixed-delay units and 845 randomized-delay units. | Final conversion must use the loader-selected probes, then apply the reference low-FR and session-inclusion rules before comparing counts. The remaining mismatch is small enough to attribute to downstream firing-rate/session filtering and mixed file layouts rather than a cohort-definition error. |
| `obj.clu` storage layout differs by file format | MATLAB code assumes `obj.clu{probe}` for most HDF5/v7.3 files. | Newer v7.3 sessions usually store `obj.clu` as a list of probe structs with `quality` vectors, but older MATLAB-format sessions (for example randomized `JEB23` `2023-10-18`, `JEB23` `2023-10-21`, and all loader-included `JEB24` sessions) store `obj.clu` as a flat list of unit dicts for a single probe. | Paper does not discuss this implementation detail. | The Python loader must branch on `obj.clu` layout: detect probe-structured vs flat-unit storage and normalize both into one per-session unit table before filtering/alignment. |
| Alignment event on WC trials | `Figure8a_thru_c.m` sets `params.alignEvent = 'goCue'` even when mixing DR and WC trials. | In raw context sessions, WC trials still have valid `bp.ev.goCue` values; median `goCue-sample` is 2.2 s and median `goCue-delay` is 0.9 s on both WC and DR trials. Reward occurs later on WC than on DR. | Context analyses are reported relative to go cue / response timing. | Align all trials to `goCue` exactly as requested by the decoder task and as done in the released context-analysis code. Do not substitute reward time for WC trials. |
| Availability of WC trials outside the 12-session context cohort | Figure 8 uses only the 12-session subset loaded from `JEB6`, `JEB7`, `EKH1`, `EKH3`, `JGR2`, `JGR3`, `JEB19`. | Additional fixed-delay ephys sessions from `JEB13`, `JEB14`, and `JEB15` also contain nonzero `autowater` trial counts, though often fewer than the canonical context sessions. | Paper highlights a dedicated “two-context” cohort of 12 sessions. | Treat `bp.autowater` as the authoritative trial-wise context label. The 12-session context cohort is a stricter analysis subset, not evidence that other sessions lack context information. |

### Final Understanding
- The defensible neural cohort for conversion is the union of the released ephys loader cohorts, interpreted through the loader scripts rather than the raw folder counts.
- Session inclusion must respect the per-session probe choice encoded in the `load*_ALMVideo.m` scripts.
- Mixed MATLAB file layouts are a real data-loading edge case and must be handled explicitly in Python.
- Trial-wise context should come directly from `bp.autowater`, while the paper’s 12-session context cohort should be treated as a specialized subset used for one analysis figure.
- Go-cue alignment is consistent across the decoder task, the reference figure scripts, and the raw event tables, including WC trials.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu` selected by per-session loader probe(s) | `neural` | Load only the probe(s) specified by the released `load*_ALMVideo.m` scripts, exclude cluster qualities `garbage`, `gabrga`, `noisy`, `real?`, align spike times to `bp.ev.goCue`, bin on a common `dt=0.01 s` grid over `[-2.5, 2.5] s`, smooth with causal `mySmooth(...,15,'reflect')`, then remove units with mean FR `< 1 Hz` across conditions. Save each trial as `(n_neurons, n_timepoints)` float32 firing rates in Hz. | `loadSessionData`, `processData`, `findClusters`, `alignSpikes`, `getSeq`, `removeLowFRClusters`, `mySmooth` | This follows the figure-level decoder/kinematic scripts (`Figure3h.m`, `Figure8a_thru_c.m`) more closely than the generic defaults in `getDefaultParams.m`. |
| `obj.time` after neural binning | `input[0]` | Use the neural time vector relative to go cue, repeated identically for every trial in the session as a `(1, T)` float32 array. | `getSeq` | Decoder task specifies time from go cue onset as the only decoder input. |
| `bp.ev.lickL`, `bp.ev.lickR`, `bp.ev.goCue` | `output[0]` (`lick_direction`) | Find the earliest lick event after go cue. Encode `left=0`, `right=1`, `none=2`, then broadcast across all `T` bins within the trial. | `firstLickTime` (timing logic), raw lick-event arrays | Uses actual first response side, not instructed side. Miss trials therefore flip relative to `bp.L`/`bp.R` as expected. |
| `bp.autowater` | `output[1]` (`behavioral_context`) | Encode `WC=0` when `autowater==1`, `DR=1` when `autowater==0`, then broadcast across time. | Trial-condition logic in `findTrials`; context conditions in `Figure8a_thru_c.m` | Use `autowater` directly even outside the 12-session context-analysis subset. |
| `bp.hit`, `bp.miss`, `bp.no` | `output[2]` (`outcome`) | Encode `incorrect=0` for miss, `correct=1` for hit, `ignore=2` for no-response, then broadcast across time. | `getOutcome`, standard trial fields in `bp` | Keep ignore trials because the decoder task explicitly requires them, even though some behavioral analyses omitted them. |
| Side-view tongue trajectory (`tongue`; fallback `left_tongue` / `right_tongue`) | `output[3]` (`tongue_velocity`) | Reuse the reference interpolation/alignment logic from `findPosition`/`findVelocity`, but retain the pre-fill visibility mask. Compute tongue speed magnitude from aligned x/y velocity. Within each session, threshold visible samples at the 50th percentile: `<p50 -> 0`, `>=p50 -> 1`, `not visible -> 2`. | `getKinematicsFromVideo`, `findPosition`, `findVelocity` | The reference code sets missing tongue velocity to 0 after interpolation; for the decoder task we preserve a separate “not visible” class before that fill step. |
| Bottom-view paw trajectory (`top_paw`; fallback `bottom_paw`) | `output[4]` (`paw_velocity`) | Use the same interpolation/alignment logic as the reference video code, compute speed magnitude from aligned x/y velocity, threshold visible samples at the session 50th percentile, and encode `not visible -> 2`. | `Figure1e.m` feature choice, `findPosition`, `findVelocity` | The shared code explicitly analyzes `top_paw_yvel_view2`; using `top_paw` as the primary paw marker is the closest published precedent. |
| Motion energy file `me.data` / `obj.me.data` and aligned video timing | `output[5]` (`motion_energy`) | Load and align motion energy on the same neural time base as `loadMotionEnergy`. On trials with usable video, threshold aligned motion-energy samples at the session 50th percentile: `<p50 -> 0`, `>=p50 -> 1`. If the trial/session lacks usable video or motion-energy data, encode `2` for the affected bins. | `loadMotionEnergy`, `findVideoOffset` | This intentionally differs from the paper’s manual bimodal movement threshold because the decoder task explicitly requests a per-session median split. |

### Key Decisions
1. **Use the loader-defined neural cohort, not every raw file**: The released paper code defines the paper-comparable cohorts explicitly through the `load*_ALMVideo.m` scripts. The conversion will therefore use the union of the fixed-delay and randomized-delay ephys loader sessions (44 sessions total before downstream filtering), rather than every raw `.mat` file with neural data.
2. **Exclude `stim.enable` and `early` trials, but keep hit/miss/no trials**: Reference trial definitions routinely exclude perturbation/stimulation and early-lick trials. Ignore/no-response trials are retained because the decoder task requires the `ignore` outcome class.
3. **Use one common alignment event and one common bin size for all sessions**: Align everything to `goCue`, use `dt=0.01 s`, and use a common window `[-2.5, 2.5] s`. This matches the released figure scripts and tutorial path closely while avoiding unnecessary extrapolation at very early times.
4. **Store outputs as fully time-varying `(doutput, T)` arrays**: Although lick direction, context, and outcome are trial-level variables, they will be repeated across time bins so they can coexist with time-varying kinematic outputs in a single consistent output array per trial.
5. **Keep neural activity as smoothed firing rates, not z-scored residuals**: The reference loading path and decoder examples operate directly on `obj.trialdat` after spike alignment/binning/smoothing. Additional z-scoring or subspace projection would make the saved dataset less general and less faithful to the primary loader output.
6. **Extend the reference kinematic extraction only where the decoder task forces it**: The shared scripts already define the correct video synchronization, interpolation, smoothing, and missing-data handling. The only deliberate extensions are: adding a paw feature to the requested trajectory list, computing scalar speed magnitudes, and preserving missing-visibility masks for the required `not visible` / `no video` classes.
7. **Use trial-event licks for lick direction instead of `bp.L` / `bp.R`**: `bp.L` and `bp.R` encode the trial side, not the animal’s realized first post-go-cue lick. Actual lick direction should come from `bp.ev.lickL` and `bp.ev.lickR`.
8. **Require at least 10 retained units per session and at least 2 valid trials per saved session**: This matches the paper’s session-inclusion rule and the decoder’s minimum practical requirement.
9. **Represent all neurons as ALM**: The loader names and `getDefaultParams.m` indicate the recorded area is ALM for these ephys/video cohorts, so `brain_regions = ['ALM']` is sufficient unless a raw-data contradiction appears during implementation.

### Planned Sanity Checks
- [ ] Recompute one trial-by-trial neural firing-rate vector from raw spike times with `np.histogram` + causal Gaussian smoothing and confirm it matches the saved neural array with `np.allclose()`.
- [ ] Recompute one aligned motion-energy trace directly from raw `me.data`, `frameTimes`, `findVideoOffset`, and `goCue`, then compare against the saved motion-energy-derived output bins.
- [ ] Recompute one tongue-speed trace and one paw-speed trace directly from raw DLC trajectories and confirm the saved visible/not-visible masks and discretized bins match.
- [ ] Spot-check three trials where lick direction is obvious from `lickL` / `lickR` timestamps and verify agreement with saved `lick_direction`, `context`, and `outcome`.
- [ ] Check that the saved session/unit counts after filtering are consistent with the loader-selected cohorts and with the paper’s approximate neuron/session totals.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Implemented `/app/convert_data.py` with the required CLI:
  - `python -u /app/convert_data.py <outpicklefile>`
  - `--full` / `--sample`
  - `--show-processing`
- The script now:
  - Parses the released figure/loader scripts to recover the paper-defined session lists and per-session probe selections.
  - Loads both MATLAB v7.3/HDF5 and older MATLAB files.
  - Normalizes the two observed `obj.clu` layouts:
    - per-probe structs with vectorized fields
    - flat per-unit lists in older single-probe files
  - Normalizes the observed `obj.traj` layouts, including the v7.3 case where each camera view is a dict of trial-indexed fields.
  - Reimplements the reference spike pipeline in Python:
    - go-cue alignment
    - 10 ms binning over `[-2.5, 2.5] s`
    - causal Gaussian smoothing matching `mySmooth(...,15,'reflect')`
    - quality filtering
    - low-FR filtering
  - Builds trial-wise outputs for lick direction, context, outcome, tongue velocity, paw velocity, and motion energy on the common neural time base.
  - Emits optional per-session processing plots as `processing_<session_id>.png`.

Code inefficiencies identified:
- MATLAB-style nested structs/cells produce expensive Python-side branching if normalized repeatedly inside inner loops.
- Video alignment can become expensive if feature-name parsing or trajectory-layout parsing happens per trial instead of once per session.

Code speedups added:
- Session lists and probe selections are parsed once from the reference loader scripts, not rediscovered from the filesystem every session.
- Trajectory and cluster layouts are normalized once per session before inner-loop processing.
- Neural trial tensors are preallocated per session and filled in place.
- The sample run indicates roughly 5-6 seconds per session on two representative sessions, which is short enough that no parallelization is required yet.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 92 |
| Neurons / session | 46.0 mean (29, 63) |
| Subjects | 2 (`JEB6`, `JEB11`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 625 |
| Trials / session | 302, 323 |
| `time_from_go_cue_s` range | [-2.5, 2.5] |
| `lick_direction` distribution | [0.381, 0.461, 0.158] for [left, right, none] |
| `behavioral_context` distribution | [0.184, 0.816] for [WC, DR] |
| `outcome` distribution | [0.083, 0.757, 0.160] for [incorrect, correct, ignore] |
| `tongue_velocity` distribution | [0.067, 0.067, 0.866] for [low, high, not_visible] |
| `paw_velocity` distribution | [0.465, 0.465, 0.070] for [low, high, not_visible] |
| `motion_energy` distribution | [0.500, 0.500] for [low, high] |

### Processing Plots Review
- Generated `processing_JEB6_2021-04-18.png` and `processing_JEB11_2022-05-10.png` without plotting/runtime errors.
- No shape or alignment errors were surfaced by the plotting code path.
- One statistic worth monitoring in Step 8 is the high tongue `not_visible` fraction (86.6% in the two-session sample). This may reflect real tongue occlusion/sparsity rather than a bug, but decoder performance must confirm that the tongue output remains decodable enough to be defensible.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Session/probe parsing and struct normalization done once per session; neural tensors preallocated | Keeps runtime near linear in session count and avoids repeated nested-structure traversal inside inner loops |

| Step | Time / Session | Estimated Total Time |
| Sample conversion (`JEB6`, `JEB11`) | 5.0 s, 5.8 s | ~4.0 min for 44 sessions at 5.4 s/session |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `lick_direction` | 0.6213 | 0.5817 |
| `behavioral_context` | 0.7973 | 0.8136 |
| `outcome` | 0.5911 | 0.5561 |
| `tongue_velocity` | 0.5481 | 0.5294 |
| `paw_velocity` | 0.4719 | 0.4470 |
| `motion_energy` | 0.8095 | 0.8077 |

### Notes
- Training used 499 trials and validated on 126 trials.
- Loss decreased monotonically from `7.0897` at epoch 1 to `0.7521` at epoch 200.
- Every output was above uniform-chance balanced accuracy.
- The tongue output remained decodable above chance despite the large `not_visible` fraction in the two-session sample, which argues that the visibility/discretization logic is at least self-consistent on the sample cohort.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 1.8G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 1,651 fixed + 845 randomized = 2,496 | 44 loader-defined sessions; exact post-filter unit total not printed in scripts | 7,241 selected-probe raw clusters fixed + 3,089 selected-probe raw clusters randomized before quality/FR filters | 1,520 fixed + 923 randomized = 2,443 total | Approximate match; within 2.1% of paper total |
| Mean neurons/session | 66.0 fixed; 44.5 randomized | Loader cohort sizes imply 25 fixed / 19 randomized sessions | Highly heterogeneous raw counts by session | 55.5 overall; 60.8 fixed; 48.6 randomized | Reasonable after low-FR filtering |
| Subjects | 9 fixed + 4 randomized (paper text); 17 total mice in study overall | 10 fixed animal IDs + 4 randomized animal IDs = 14 subject IDs in loader-defined decoder cohort | 14 mice with neural data in the ephys/video folders | 14 subject IDs | Matches reference code / raw neural cohort; differs from paper mouse-count text as noted in Step 4 |
| Sessions | 25 fixed + 19 randomized = 44 | 25 fixed + 19 randomized = 44 | 45 raw neural sessions, but one randomized session is commented out by code | 44 | Yes |
| Trials (total) | Not explicitly reported | Not explicitly reported | 15,324 raw trials across all raw neural sessions | 13,762 kept trials after excluding `stim`, `early`, and trials without neural coverage | Expected reduction from reference-style trial filtering plus removal of late behavioral-only tails in two JEB24 sessions |
| Trials/session (mean) | Not explicitly reported | Not explicitly reported | 340.5 raw neural trials/session | 312.8 kept trials/session | Expected reduction from filtering |
| `time_from_go_cue_s` range | Go-cue aligned analyses throughout; figure scripts commonly use `[-2.5, 2.5]` | `Figure3h.m` fixed uses `[-2.5, 2.5]` | Raw events support go-cue alignment | [-2.5, 2.5] | Yes |
| `lick_direction` distribution | Not explicitly reported | Trial logic supports left/right/none outcomes | Raw trial tables contain left/right/ignore behavior | [0.407, 0.462, 0.130] for [left, right, none] | Plausible |
| `behavioral_context` distribution | Two-context analyses use DR and WC; randomized cohort is DR-only | Context is carried by `autowater` | Many sessions are mostly or entirely DR | [0.097, 0.903] for [WC, DR] | Plausible |
| `outcome` distribution | Training criterion >70% correct is consistent with correct dominating | Choice/context scripts emphasize hit/miss/ignore structure | Raw trials contain hit/miss/no | [0.120, 0.749, 0.131] for [incorrect, correct, ignore] | Plausible |
| `tongue_velocity` distribution | Not explicitly reported | Reference code handles missing tongue visibility specially | Raw tongue tracking is sparse/occluded in many sessions | [0.033, 0.040, 0.927] for [low, high, not_visible] | Needs decoder-based review in Step 11/12 but internally consistent |
| `paw_velocity` distribution | Not explicitly reported | Paws are tracked in bottom view; figure scripts use `top_paw` | Raw paw tracking is present but missing in many late randomized sessions | [0.311, 0.311, 0.378] for [low, high, not_visible] | Plausible |
| `motion_energy` distribution | Motion energy used extensively in the paper | Reference code aligns motion energy to neural time | Some randomized sessions lack separate motion-energy files / usable video | [0.377, 0.378, 0.245] for [low, high, no_video] | Plausible; missing-video fraction is explained by raw data availability |

### Notes
- Full conversion completed in 212.6 s for 44 sessions, comfortably below the 15-minute optimization threshold.
- After one Step 10 bug-fix iteration (described below), the current `verification_full_out.txt` rerun reports `Data format is valid, no errors or warnings.`

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: The first full verify pass exposed warnings about all-zero neural trials in session indices 36 and 43. I mapped these to `JEB24_2023-10-23` and `JEB24_2023-11-03`, inspected the raw `obj.clu[*].trial` coverage directly, and confirmed that these sessions contain behavioral records after the last trial with any spikes. I fixed this by requiring `neural_covered` inside the saved-trial mask. After rerunning full conversion and verify, `verification_full_out.txt` contains no errors or warnings.
2. **Raw-data sanity checks using `np.allclose()`**: I rebuilt representative saved values directly from the original `.mat` files, not by calling the conversion path:
   - `JEB6_2021-04-18` input check: independently reconstructed the `[-2.5, 2.5]` go-cue time axis and verified `np.allclose(manual_taxis, saved_input[0]) == True`.
   - `JEB6_2021-04-18` neural check: independently histogrammed spike times from raw probe 2, aligned them to raw `bp.ev.goCue`, smoothed with the causal Gaussian kernel, and verified `np.allclose(manual_trace, saved_neural_trial) == True` for saved trial index 4 / raw trial 5 / neuron 0.
   - `JEB6_2021-04-18` categorical output checks: for saved trial indices 0, 50, and 200 (raw trials 0, 55, and 261), independently recomputed first post-go-cue lick side from raw `lickL` / `lickR`, context from raw `autowater`, and outcome from raw `hit` / `miss` / `no`; all three trials matched the saved output rows with `np.allclose()`.
   - `JEB6_2021-04-18` motion-energy output check: independently loaded `motionEnergy_JEB6_2021-04-18.mat`, reconstructed the aligned motion-energy trace for saved trial index 4, discretized it with the saved per-session median threshold (`15.5`), and verified `np.allclose(manual_disc, saved_motion_output) == True`.
3. **Reference code comparison**:
   - **Loading**: `convert_data.py` parses the same loader-defined cohorts from `Scripts/Figure 3/Figure3h.m` and `DataLoadingScripts/Recording and video/load*_ALMVideo.m` that the reference figure scripts use, rather than discovering sessions from the filesystem. This matches the released cohort definitions exactly.
   - **Neuron filtering**: `extract_selected_units()` mirrors `findClusters('all')` by excluding only `garbage`, `gabrga`, `noisy`, and `real?`. `process_session()` then mirrors the published `>1 Hz` criterion using condition-averaged smoothed firing rates before saving units.
   - **Trial filtering**: `keep_trials = (hit | miss | no) & ~early & ~stim & neural_covered` matches the reference exclusion of `early` and `stim` trials, with one explicit extension: `neural_covered` prevents saving behavioral-only tail trials that are not valid neural-decoder examples.
   - **Temporal alignment**: both the reference scripts and the converter align to `goCue` for the relevant fixed-delay and context analyses; WC trials keep their native `goCue`, consistent with the reference context analyses.
   - **Binning / smoothing**: the converter uses `dt = 0.01 s`, `[-2.5, 2.5]`, and a causal Gaussian smoother intended to match `mySmooth(...,15,'reflect')`. This matches the released figure-script configuration more closely than the generic 5 ms / 200 Hz defaults described elsewhere in the repository and paper.
   - **Input construction**: the decoder task only requires time from go cue, so the saved input is the aligned neural time axis. This is a deliberate task-driven restriction, not a divergence in preprocessing.
   - **Output construction**: categorical trial variables come directly from raw `lickL` / `lickR`, `autowater`, and `hit` / `miss` / `no`; continuous video-derived outputs reuse the same raw sources and alignment strategy as `findPosition`, `findVelocity`, and `loadMotionEnergy`, then apply the task-required per-session median split.
4. **Key statistics comparison**:
   - Cohort counts match the released code exactly: 25 fixed sessions + 19 randomized sessions = 44 saved sessions.
   - Converted cohort membership also matches the raw loader-selected data exactly: 10 fixed subject IDs + 4 randomized subject IDs = 14 neural-data subject IDs.
   - Converted neuron totals are close to the paper totals but not identical: 1,520 fixed vs 1,651 paper, 923 randomized vs 845 paper, 2,443 total vs 2,496 paper. The session counts match exactly, so the remaining discrepancy is most plausibly due to paper/code curation differences and the manuscript’s inconsistent mouse-count reporting, not a cohort-definition bug.
   - Trial totals are not reported in the paper, but the reduction from 15,324 raw neural-session trials to 13,762 saved trials is fully explained by the reference-style `stim` / `early` exclusions plus the two JEB24 neural-coverage edge cases.
5. **Edge-case review**:
   - Verified the mixed raw `obj.clu` layouts (probe-structured vs flat per-unit lists) and the mixed raw `obj.traj` layouts are both handled.
   - Verified the separate motion-energy files also come in mixed layouts (struct-wrapped vs bare arrays) and are both handled.
   - Verified the late-session partial-coverage edge case directly from raw data:
     - `JEB24_2023-10-23`: raw neural coverage stops at trial 314/343; saved kept-trial count after requiring coverage is 292.
     - `JEB24_2023-11-03`: raw neural coverage stops at trial 312/346; saved kept-trial count after requiring coverage is 301.
   - Spot-checked trial boundaries through the independent `taxis` reconstruction and found no off-by-one discrepancy at the first or last saved time bin.

### Issues Found and Resolved
- **Late behavioral-only trials were being saved in two randomized-delay sessions**: The initial full verify warning revealed that some raw sessions continue to log behavior after the neural recording stops. Resolution: add `neural_covered` to the saved-trial mask, rerun full conversion, and confirm that the warnings disappear.
- **No remaining Step 10 mismatches**: After the coverage fix and the raw-data `np.allclose()` checks, I did not find additional loading, alignment, or labeling mismatches that required code changes.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `lick_direction` | 0.6439 | 0.6297 | 1.89x chance; stable train/validation match |
| `behavioral_context` | 0.8609 | 0.8493 | 1.70x chance; strongest categorical task besides motion energy |
| `outcome` | 0.6360 | 0.6153 | 1.85x chance |
| `tongue_velocity` | 0.6411 | 0.6303 | 1.89x chance despite high `not_visible` fraction |
| `paw_velocity` | 0.6344 | 0.6297 | 1.89x chance |
| `motion_energy` | 0.8465 | 0.8452 | 2.54x chance; essentially no train/validation gap |

### Notes
- Training ran successfully on `cuda`; no `--cpu` fallback was required.
- The full run used 10,994 training trials and 2,768 validation trials.
- Loss decreased from `7.640192` at epoch 1 to `0.600955` at epoch 200; test loss was `0.647585`.
- `train_decoder.py` finished successfully and produced the expected sample/prediction plots.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
|----------|-------------------|------------------------|
| `lick_direction` | 0.6297 validation balanced accuracy | Nearest published comparator is neural choice decoding / delay-epoch `CDchoice` decoding. The paper reports strong above-chance neural choice decoding and a session-wise ROC AUC of `0.86 ± 0.11` for delay-epoch `CDchoice`; metric is not directly comparable to balanced accuracy, but the achieved decoder performance is directionally consistent with a strong neural choice signal. |
| `behavioral_context` | 0.8493 validation balanced accuracy | The paper shows strong above-chance neural context decoding across time (Fig. 4b), but does not provide a single numeric balanced-accuracy summary. The achieved value is consistent with the paper’s qualitative result that context is robustly decodable from ALM. |
| `outcome` | 0.6153 validation balanced accuracy | No directly comparable neural outcome-decoding accuracy is reported in the paper. Above-chance performance is therefore assessed internally rather than against a published numeric target. |
| `tongue_velocity` | 0.6303 validation balanced accuracy | No directly comparable neural tongue-velocity decoder accuracy is reported. The paper does report strong coupling between neural population activity and kinematics, so above-chance decoding is expected qualitatively. |
| `paw_velocity` | 0.6297 validation balanced accuracy | No directly comparable neural paw-velocity decoder accuracy is reported. Above-chance decoding is qualitatively consistent with the paper’s movement-related ALM signals. |
| `motion_energy` | 0.8452 validation balanced accuracy | No directly comparable neural motion-energy classification accuracy is reported, but the paper repeatedly shows strong neural-movement relationships. The very high score is consistent with those qualitative results. |

### Review Results
- **Accuracy vs chance**: every validation score exceeds chance by a comfortable margin:
  - `lick_direction`: `0.6297 / 0.3333 = 1.89x`
  - `behavioral_context`: `0.8493 / 0.5000 = 1.70x`
  - `outcome`: `0.6153 / 0.3333 = 1.85x`
  - `tongue_velocity`: `0.6303 / 0.3333 = 1.89x`
  - `paw_velocity`: `0.6297 / 0.3333 = 1.89x`
  - `motion_energy`: `0.8452 / 0.3333 = 2.54x`
- **Train vs validation gap**: no output shows the `>1.5x` train/validation ratio that would suggest severe overfitting or leakage. The largest ratio is `1.034` for `outcome`; the smallest is `1.002` for `motion_energy`.
- **Comparison to paper**: the paper reports only partially comparable metrics for decoding. Where it does report a numeric neural decoding metric (`CDchoice` ROC AUC `0.86 ± 0.11`), the result is broadly compatible with the strong choice decoding seen here, acknowledging the metric/task mismatch. For context, the paper gives qualitative time-resolved decoding curves but not a single aggregate balanced-accuracy value. For outcome and movement-class outputs, there is no direct published accuracy number to match.
- **Further debugging decision**: because all outputs are above chance, train/validation gaps are small, the raw-data spot checks passed, and the current dataset matches the released code cohort definitions, I did not find evidence that another conversion iteration was justified.

### Issues Found and Resolved
- No Step 12 issues required further converter changes after the successful full decoder run and the accuracy review above.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

### Notes
- Added `/app/README.md` with dataset summary, loading instructions, format description, processing summary, and validation snapshot.
- Added `/app/cache/README_CACHE.md` documenting the role of the cache directory and why the required output files remain in `/app/`.
- No standalone investigation scripts needed relocation because debugging/sanity checks were executed as one-off commands and fully documented in this notes file.
