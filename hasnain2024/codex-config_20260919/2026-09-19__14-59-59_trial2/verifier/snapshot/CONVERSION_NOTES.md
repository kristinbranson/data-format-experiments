# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse (provided paper/code/data)
- **Date started**: 2026-09-19
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

Environment verification:
- Python 3.13.15
- NumPy 2.4.4 imported successfully
- PyTorch 2.6.0+cu124 imported successfully
- Checkpoint `ls -la /app/CONVERSION_NOTES.md` succeeded.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadObjs` | `code/DataLoadingScripts/loadObjs.m` | LOADING | Loads each MATLAB `obj` named by session metadata and normalizes absent top-level fields. |
| `loadSessionData` | `code/DataLoadingScripts/loadSessionData.m` | LOADING | Orchestrates per-session/per-probe processing and concatenates selected probes. |
| `findTrials` | `code/DataLoadingScripts/findTrials.m` | CURATION | Evaluates logical condition expressions against `obj.bp` and returns 1-based trial IDs. |
| `findClusters` | `code/DataLoadingScripts/findClusters.m` | CURATION | Implements quality-label selection; `all` excludes `garbage`, typo `gabrga`, `noisy`, and `real?`. |
| `alignSpikes` | `code/DataLoadingScripts/alignSpikes.m` | PROCESSING | Subtracts the per-trial alignment event (here `goCue`) from every spike's within-trial time. |
| `getSeq` | `code/DataLoadingScripts/getSeq.m` | PROCESSING | Bins aligned spikes on `[tmin,tmax)` and converts counts to Hz, then causally Gaussian-smooths trial and PSTH data. |
| `mySmooth` | `code/utils/mySmooth.m` | PROCESSING | MATLAB `gausswin(N)` with the leading half zeroed, renormalized, and convolved using `same`; optional reflected/zero padding. |
| `removeLowFRClusters` | `code/DataLoadingScripts/removeLowFRClusters.m` | CURATION | Retains clusters whose condition-averaged smoothed PSTH is strictly above `params.lowFR`. |
| `findVideoOffset` | `code/funcs/findVideoOffset.m` | PROCESSING | Computes neural/video offset as `mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`. |
| `findPosition` / `getKinematicsFromVideo` | `code/funcs/kinematics/getKinematicsFromVideo.m` | PROCESSING | Aligns DLC frame times by video offset and trial go cue, interpolates onto the neural time axis, and derives x/y velocity. |
| `loadMotionEnergy` | `code/DataLoadingScripts/loadMotionEnergy.m` | LOADING / PROCESSING | Loads session motion energy, applies video offset, aligns/interpolates onto neural time, and nearest-fills edge NaNs. |
| `getOutcome` | `code/funcs/getOutcome.m` | PROCESSING | Uses `bp.hit` for correct/incorrect and sets `bp.no` trials to NaN (ignore). |
| `NeuralChoiceDecoding` | `code/ChoiceContextDecoding/NeuralChoiceDecoding.m` | PROCESSING | Reference neural choice decoding uses go-cue-aligned `trialdat` and 75 ms decoding bins. |
| `NeuralContextDecoding` | `code/ChoiceContextDecoding/NeuralContextDecoding.m` | PROCESSING | Reference context labels distinguish delayed response (`~autowater`) from water-cued (`autowater`) trials. |

### Notes
- The repository README identifies the supplied neural recordings as electrophysiology from left/right ALM. There is no imaging or fluorescence stream, so delta-F/F is not applicable.
- Native session objects contain behavioral/task data (`obj.bp`), sorted clusters (`obj.clu`), DLC trajectories (`obj.traj`), SpikeGLX metadata (`obj.sglx`), and session metadata (`obj.ex`). Separate `motionEnergy_*.mat` files accompany electrophysiology sessions.
- Unit qualities described in `WorkingWithDataObjs.m` call `excellent`, `great`, and `good` single units, while the standard `params.quality={'all'}` path is broader and excludes only explicit bad labels. The final choice must be resolved against Methods and supplied data in Steps 3-5.
- Standard parameters align to go cue, use a five-second window (`-2.5` to `+2.5` s), and use causal Gaussian smoothing with a 15-sample window. `getDefaultParams.m` uses 5 ms bins and a 0.5 Hz threshold; the tutorial example uses 10 ms bins and a 1 Hz threshold. This discrepancy is deferred to cross-source resolution.
- The bin centers produced by `getSeq` are `tmin + dt/2` through `tmax - dt/2`; the right endpoint is excluded.
- DLC feature coordinates are interpolated using `frameTimes - video_offset - goCue`. Non-tongue gaps are nearest-filled; tongue nonvisibility is tracked from NaNs before baseline substitution. Velocity in `findVelocity` is a per-sample gradient, not explicitly converted from pixels/sample to pixels/second.
- The requested outputs differ from paper analyses in requiring per-session median discretization. Raw aligned velocity/motion-energy values and explicit visibility masks should therefore be retained through alignment and only then mapped to classes 0/1/2.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Archive size is 16 GB. The only data-side documentation is `.fetch_complete`, a Zenodo manifest for record `13941415`.
- Four experiment directories contain 120 `data_structure_<animal>_<date>.mat` sessions: `Ephys_Behavior` (25), `RandomizedDelay_Ephys_Behavior` (22), `DelayInhibition_BilatMC_Behavior` (53), and `GoCueInhibition_BilatMC_Behavior` (20).
- `Ephys_Behavior` contains 25 paired `motionEnergy_*.mat` files. `RandomizedDelay_Ephys_Behavior` contains 20 motion-energy files for 22 data objects; JEB24 2023-10-03 and 2023-10-04 also lack `obj.clu` and are therefore not neural sessions.
- Most data objects are MATLAB v7.3/HDF5. Several late randomized-delay objects are MATLAB v5 and require `scipy.io.loadmat`; motion-energy files are MATLAB v5.
- Each neural `obj` provides `bp` (trial/task fields), `bp.ev` (event times and lick contacts), `clu` (sorted spike clusters), `traj` (two-camera DLC trajectories), `sglx` (sampling and synchronization metadata), and `trials` (availability masks). Some older objects lack `meta`; subject/date remain encoded in filenames.
- Trial/task variables include `R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `bitRand`, protocol values, and events `bitStart`, `sample`, `delay`, `goCue`, `reward`, `lickL`, and `lickR`.
- Cluster fields include session time `tm`, trial-relative time `trialtm`, trial number `trial`, quality string, waveform, and site/channel. Cluster data are sorted electrophysiology, not calcium imaging.
- `traj` is a two-element camera cell array. Each camera contains one structure per trial with `ts`, `frameTimes`, `NdroppedFrames`, filenames, and feature names. In HDF5 storage, a nominal MATLAB `(frames, 3, features)` `ts` array appears reversed as `(features, 3, frames)`. Side-camera features include tongue/jaw/trident/nose; bottom-camera features include tongue landmarks, top/bottom paw, jaw, and nostrils.
- Motion-energy files contain `me.data`, a per-trial variable-length vector sampled on video frames, and a scalar author threshold `me.moveThresh`. The requested decoder instead requires a per-session 50th-percentile threshold.
- Representative Ephys session EKH1 2021-08-07 has 305 trials, two probes (32 and 48 raw clusters), 7 side-camera and 10 bottom-camera DLC features, and 305 motion-energy vectors. All 8,260 `Ephys_Behavior` trials report finite go cues plus `haveEphys=1` and `haveVid=1`.
- In `Ephys_Behavior`, aggregate task flags are: 5,711 hit, 1,168 miss, 1,381 no-response, 4,119 right, 4,141 left, 1,449 autowater/WC, 661 early, and 187 optogenetic-stimulation trials.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 14,297 raw clusters over all files containing neural data; 11,158 in `Ephys_Behavior` and 3,139 in randomized-delay data. These are pre-quality/pre-region clusters across all probes, not the final curated count. |
| Neurons / session | `Ephys_Behavior`: mean 446.32, median 243, range 27-1,258 raw all-probe clusters; randomized-delay: mean 142.68 including two zero-neuron files, median 52, range 0-560. |
| Subjects | 18 in the complete archive; 10 in `Ephys_Behavior`; 14 across the two directories labeled as ephys. |
| Sessions / subject | Complete archive: 1-24; `Ephys_Behavior`: EKH1 1, EKH3 1, JEB6 1, JEB7 2, JEB13 5, JEB14 4, JEB15 4, JEB19 4, JGR2 2, JGR3 1. |
| Trials (total) | 37,073 complete archive; 8,260 `Ephys_Behavior`; 7,583 randomized-delay; 15,079 delay-inhibition behavior-only; 6,151 go-cue-inhibition behavior-only. |
| Trials / session | `Ephys_Behavior`: mean 330.4, median 313, range 230-517; randomized-delay: mean 344.68, median 348, range 218-450. |

The requested WC/DR context output is directly represented in `Ephys_Behavior` via `autowater`; whether randomized-delay sessions belong in the target dataset remains a cross-source scope decision for Steps 3-5. Behavior-only inhibition sessions cannot be included in a neural decoder because they contain no neural activity.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Fixed-delay DR cohort: 1,651 units (483 single units); two-context subset: 522 units (214 single units); randomized delay: 845 units (288 single units). | Methods: “1,651 units (483 single units)” and “522 units (214 well-isolated single units)”. | 
| Neurons / session | Not stated as a mean; recording sessions required at least 10 included units. | Methods: sessions were included only with “at least 10 units”. |
| Subjects | 17 mice across all cohorts; fixed-delay neural cohort 9 mice; two-context subset 6 mice; randomized-delay cohort 4 mice. | Methods: “This study used data collected from 17 mice”. |
| Sessions / subject | Fixed-delay: 25/9 = 2.78 average; two-context: 12/6 = 2 average; randomized delay: 19/4 = 4.75 average. | Paper reports 25 sessions/9 mice, 12 sessions/6 mice, and 19 sessions/4 mice. |
| Trials (total) | Not reported as an aggregate. | No aggregate trial count in paper/methods. |
| Trials / session | Not reported as a mean; behavioral-analysis inclusion required at least 40 correct DR trials/direction and 20 correct WC trials/direction. | Methods states these per-condition minima. |
| Neural data time bin | 5 ms, with a causal Gaussian kernel of 35 ms half-width for single-trial analysis. | Methods: “binned in 5-ms intervals” and causal Gaussian smoothing. |
| Behavior data time bin | Raw high-speed video at 400 Hz (2.5 ms/frame); reference models use behavior resampled to neural 5 ms bins. | Methods: “High-speed video was captured (400-Hz frame rate)”. |
| Reward rate | Not reported; animals were trained to at least 70% accuracy. | Methods: animals reached “at least 70% accuracy”. |
| Two-context block design | Session begins with ~100 DR trials, then alternating WC/DR blocks of 10-25 trials. | Methods describes this block schedule. |
| Ignore definition | No response within 3 s after go cue. | Methods defines this as an ignore trial. |
| Typical response latency | Usually within 300 ms after go cue. | Methods notes responses were typically registered within 300 ms. |
| Raw movement sampling | Two cameras at 400 Hz; side and bottom views. | Videography Methods. |
| Choice selectivity | 36% sample, 42% delay, 58% response among 483 single units. | Results, neural dynamics section. |
| Context selectivity | 39% of 214 two-context single units during the ITI criterion. | Results, neural dynamics section. |


### Processing Details
- Neural activity is extracellular ALM spiking. For single-trial analysis the paper bins at 5 ms and smooths with a causal Gaussian kernel of 35 ms half-width. The reference code realizes this as `dt=1/200` and a 15-sample causal `gausswin`.
- Paper time is expressed relative to go cue on DR trials and relative to water drop on WC trials. The supplied object/reference code stores the analogous alignment in `bp.ev.goCue`; this semantic distinction must be retained in metadata.
- The standard paper analysis window is compatible with `[-2.5,+2.5)` s around alignment. Baseline standardization in coding-direction analyses uses `[-2.4,-2.2]` s.
- DLC x/y positions are used directly; missing values are nearest-filled for non-tongue features but not tongue. Feature velocity is the first derivative of position. Tongue visibility is therefore carried by the raw NaN mask.
- Motion energy per frame is the 99th percentile of pixelwise differences between median future-five-frame and past-five-frame images. The paper manually thresholds the bimodal distribution per session. The decoder task explicitly supersedes that threshold with the per-session median.
- The paper's choice/context classifiers are ridge logistic regressions, trained separately at each time bin with four-fold cross-validation and 30% held-out trials. The reference balances correct left/right trials; chance is established by shuffled labels.
- The requested conversion does not reproduce paper-specific delay warping or trial balancing in the saved dataset: alignment is fixed to the native go-cue/water-drop event and trial balancing belongs to decoder training, not data conversion.

### Curation Steps

**Neuron curation rules**:
- Manual spike-sorting labels distinguish well-isolated single units and multiunits. The paper includes all units (single and multi) with firing rates strictly exceeding 1 Hz for the analyses most analogous to this decoder; single-unit-only filtering was reserved for subspace alignment and selectivity analyses.
- Sessions require at least 10 retained units.
- Only the ALM probe is relevant to paper neural analyses; simultaneously recorded tjM1/brainstem probes must not be mixed in.

**Trial curation rules**:
- Paper analyses omit early-lick trials. Reference condition strings also exclude `stim.enable` trials.
- Correct and error trials are used by the core single-trial subspace analyses. Ignore trials are normally omitted in paper analyses, but the decoder specification explicitly requires an `ignore` outcome and `none` lick direction; retaining valid ignore trials is therefore an intentional downstream exception.
- Behavior-only photoinactivation sessions cannot enter a neural decoder. Randomized-delay sessions have no WC context and are not part of the two-context context-decoding cohort.

### Decoders Trained
| Decoded variable | Accuracy |
| Upcoming choice from delay-period `CDchoice` | ROC-AUC 0.86 ± 0.11 across sessions. |
| Choice from full neural population or kinematics | Time-resolved four-fold CV accuracy shown in Fig. 3b; no scalar aggregate reported in text. |
| Behavioral context from full neural population or kinematics | Time-resolved four-fold CV accuracy shown in Fig. 4b; no scalar aggregate reported in text. |

The paper reports related prediction performance as variance explained rather than categorical accuracy: kinematics predict `CDchoice` at R² 0.41 ± 0.23, `CDramp` at R² 0.46 ± 0.23 (25 sessions), randomized-delay `CDchoice` at R² 0.38 ± 0.20 (19 sessions), and `CDcontext` at R² 0.49 ± 0.22 (12 sessions). `pypdf` 6.19.0 was installed to read the supplied PDF because no PDF text extractor was preinstalled.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Decoder cohort | Figure 8 loads JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, and JEB19 using 12 sessions total. | These exact 12 files contain substantial WC blocks and paired neural/video/motion-energy data. Other fixed-delay sessions sometimes contain a handful of autowater trials but are not in the context-analysis loader. | Two-context paradigm: 12 sessions. | Use the exact 12-session context cohort from Figure 8, not every file with any `autowater=1`. |
| Subject count | Loader names imply 7 distinct IDs. | Seven filename IDs occur in the 12 selected files. | Paper reports six mice. | Preserve the seven native subject IDs rather than merge undocumented identities. Record the paper/code discrepancy in metadata. |
| Brain region/probe | Per-animal loaders specify one ALM probe: JEB6 p2, JEB7 p1, EKH1 p2, EKH3 p2, JGR2 p1, JGR3 p1, JEB19 p1. | Modern `meta.probe.loc` confirms ALM where present; older files rely on loaders. | Neural analyses focus on ALM. | Use author loader probe assignments and label all retained neurons `ALM`; do not include simultaneously recorded tjM1/brainstem clusters. |
| Unit count | Figure 8 uses `lowFR=1`; the all-unit analysis uses `quality={'all'}`. | Selected probes reproduce exactly 214 well-isolated labels after >1 Hz filtering; all-unit counts are approximately the paper total, with borderline behavior dependent on the exact MATLAB smoothing/condition implementation. | 522 units, including 214 well-isolated single units. | Port `findClusters`, `getSeq`, `mySmooth`, and `removeLowFRClusters` semantics and verify against 522/214. Use all non-explicitly-bad manual labels for decoder data. |
| Firing-rate threshold | Figure scripts use 1 Hz; generic defaults use 0.5 Hz and tutorial examples vary. | Several manually curated clusters lie near 1 Hz. | All analogous analyses include units exceeding 1 Hz. | Use strict `>1 Hz`, resolving the default/tutorial discrepancy in favor of paper figure scripts and Methods. |
| Neural bin size | Figure 8d and `getDefaultParams` use 5 ms; some plotting scripts use 10 or 30 ms. | Raw spikes permit arbitrary binning. | Single-trial Methods specify 5 ms. | Use 5 ms consistently across sessions. |
| Smoothing | Code uses a 15-sample causal `gausswin` with reflected-prefix handling. | Applicable to all selected spike streams. | Causal Gaussian kernel with 35 ms half-width. | At 5 ms, the 15-sample code kernel matches the stated ~35 ms half-width; port it directly. |
| Analysis window | Standard loader/default is `[-2.5,+2.5)`; context Figure 8 uses a `-3` s start. | Video/spikes cover both in most trials. | Baseline is `[-2.4,-2.2]`, and figures show roughly ±2 s. | Use `[-2.5,+2.5)` as the standard decoder window, sufficient for the stated baseline and consistent with `getDefaultParams`; record this explicitly. |
| Alignment semantics | All author scripts set `alignEvent='goCue'`; video and motion energy subtract `bp.ev.goCue`. | WC trials also populate this field, at the water-delivery response event. | Figures say go cue for DR / water drop for WC. | Align every stream to native `bp.ev.goCue`; metadata states that this is water-drop onset in WC trials. |
| Trial filtering | Author condition strings generally exclude stimulation and early trials; most analyses omit ignores. | Selected sessions contain stimulation, early, and ignore trials. | Early trials are omitted; correct and error trials are analyzed. | Exclude `early` and `stim.enable`. Retain ignore trials only because the decoder explicitly requires an ignore/none class. |
| Video processing | `findPosition` uses measured frame times, computed session video offset, go-cue subtraction, linear interpolation, and nearest fill for non-tongue features. | Raw NaNs expose tongue/paw visibility and invalid video. | Same interpolation/first-derivative approach; tongue missingness is preserved. | Match alignment and interpolation. Preserve visibility before any fill so requested class 2 can be represented. |
| Motion threshold | Data include an author manual threshold. | Every selected session has a paired motion-energy file. | Manual bimodal threshold per session. | Decoder task explicitly overrides this with a finite-sample per-session 50th percentile; keep author alignment but use requested median. |

### Final Cross-Source Understanding
- The conversion target is the paper's two-context ALM cohort: 12 sessions, native IDs from seven subjects, one author-designated ALM probe per session, and a reported 522 included units (214 well-isolated singles).
- Neural and video data will share 1,000 bins spanning centers from -2.4975 to +2.4975 s at 5 ms resolution around `bp.ev.goCue`.
- Reference scientific curation is retained (ALM, explicit bad-quality rejection, >1 Hz units, no stimulation, no early trials). The sole trial-curation exception is retaining ignores to satisfy the requested outcome and lick-none outputs.
- A direct inventory of the selected sessions after the planned trial rule gives 3,116 trials: 2,086 correct, 329 incorrect, 701 ignore; 982 WC and 2,134 DR; 1,112 left, 1,303 right, and 701 none.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Selected `obj.clu{probe}.trialtm`, `.trial`, `.quality` + `bp.ev.goCue` | `neural[session][trial]` | Select author-designated ALM probe; `quality={'all'}` semantics; align spikes by subtracting per-trial go cue; 5 ms histogram; Hz conversion; 15-sample causal Gaussian; strict >1 Hz curation; save `(neurons,1000)` float32. | `findClusters`, `alignSpikes`, `getSeq`, `mySmooth`, `removeLowFRClusters` | Window `[-2.5,+2.5)` s; centers `-2.4975...+2.4975`. |
| Fixed bin centers relative to alignment | `input[session][trial][0,:]` | Copy continuous seconds-from-go-cue vector to every trial as `(1,1000)` float32. | `getSeq` time construction | Only decoder input requested; do not leak outputs/context into inputs. |
| `bp.L`, `bp.R`, `bp.no` | `output[...,0,:]` lick direction | Code left=0, right=1, none=2; repeat per-trial code over time. | Paper task definition; Figure condition masks | `no` overrides instructed L/R to `none`. |
| `bp.autowater` | `output[...,1,:]` behavioral context | WC=0 when true, DR=1 when false; repeat over time. | `NeuralContextDecoding`; Figure 8 conditions | WC uses water-drop event stored in `goCue`. |
| `bp.miss`, `bp.hit`, `bp.no` | `output[...,2,:]` outcome | incorrect=0, correct=1, ignore=2; repeat over time. | `getOutcome` | Validate exactly one state per retained trial. |
| Side-camera DLC feature `tongue` x/y and frame times | `output[...,3,:]` tongue velocity class | Apply author video offset and go-cue alignment; interpolate; first derivative per reference; Euclidean x/y speed; session median over finite visible samples; low=0/high=1/not-visible=2. | `findVideoOffset`, `findPosition`, `findVelocity`, `getKinematicsFromVideo` | Preserve raw NaN visibility mask before tongue velocity NaNs are zeroed. Positive scaling to pixels/s is unnecessary because median classes are scale-invariant; use reference pixels/bin derivative. |
| Bottom-camera DLC feature `top_paw` x/y and frame times | `output[...,4,:]` paw velocity class | Same alignment; nearest-fill coordinate gaps for speed as reference; derivative/baseline correction; Euclidean x/y speed; session median over originally visible samples; low=0/high=1/not-visible=2. | Same kinematic functions; Figure 1 uses `top_paw_yvel_view2` | Class 2 uses pre-fill visibility, satisfying requested missingness while numerical speed follows reference. |
| Paired `motionEnergy_*.mat: me.data` + side-camera frame times | `output[...,5,:]` motion-energy class | Align/interpolate using video offset and go cue; nearest-fill finite-trial edge gaps; session median over finite values; low=0/high=1/no-video=2. | `loadMotionEnergy` | Ignore supplied manual `moveThresh` because decoder explicitly requires 50th percentile. |
| Filename/native animal ID | `subjects`, `subject_idx` | Unique IDs in first-session order; session index into that list. | Author `meta.anm` | Seven native IDs retained despite paper's six-mouse statement. |
| Author ALM probe assignment | `brain_regions`, `brain_region_idx` | `brain_regions=['ALM']`; zero vector of length retained neurons per session. | `load<animal>_ALMVideo.m` | Only ALM clusters enter conversion. |
| Task names/classes | `input_names`, `output_names`, `output_values` | Explicit ordered names/classes matching codes above. | Decoder specification | All six outputs stored time-varying for consistent shape and time-resolved decoding. |
| Events, curation counts, thresholds, offsets, filenames | `metadata` | Store task/window/alignment plus per-session audit records. | All above | Includes semantic WC water-drop note and trial/neuron exclusions. |

### Key Decisions
1. **Cohort is the 12 author-listed context sessions**: This matches Figure 8 and the paper's session/unit statistics; randomized-delay and behavior-only experiments cannot support the required WC/DR neural-decoder task.
2. **Output window and bins are `[-2.5,+2.5)` at 5 ms**: This matches paper single-trial binning, `getDefaultParams`, and the baseline interval while keeping every trial/session identical in length.
3. **Reference causal firing rates, not raw counts**: Neural values are Hz after the exact 15-sample causal Gaussian path, matching the paper rather than inventing a new preprocessing stream.
4. **All scientifically curated ALM units, not singles only**: The analogous population decoder includes multiunits as well as well-isolated units, with explicit bad labels and ≤1 Hz units removed.
5. **Exclude early and stimulation trials; retain ignores**: Early/stim exclusions match paper/code. Keeping ignores is required to produce the specified ignore and lick-none classes.
6. **Per-trial categorical variables are repeated over time**: This creates a single `(6,time)` output matrix alongside time-varying movement outputs and allows the validator/decoder to treat every output consistently.
7. **Velocity is 2-D speed from reference x/y derivatives**: A percentile of signed velocity would separate direction rather than movement magnitude. Speed is the scientifically meaningful scalar requested by “velocity,” while derivative/interpolation/visibility match the reference pipeline.
8. **Central side-view tongue and top-paw bottom-view landmarks**: These are the canonical named tongue marker and the same paw marker used by paper plotting code, avoiding arbitrary averaging across anatomically distinct landmarks.
9. **Threshold population**: Each median is computed once per session from all finite samples in retained trials and the full decoder window. Missing samples are excluded from threshold estimation and assigned class 2.
10. **No-video versus edge gaps**: Valid video trials receive nearest-filled edge values as in `loadMotionEnergy`; only trials/streams with no usable frames remain class 2.
11. **Native subject IDs are authoritative**: No undocumented merging is applied to force the paper's six-mouse statement.
12. **Additional experimental variables remain audited, not model inputs**: `early`, `stim`, task events, protocol, and lick timestamps inform curation/metadata but are not added to inputs or outputs beyond the Decoder Task, preventing leakage and target drift.

### Planned Sanity Checks
- [ ] Cohort/session/probe list equals the Figure 8 loader (12 sessions) and native IDs/probe indices match each file.
- [ ] Retained ALM unit statistics reproduce the paper's 522 total and 214 well-isolated units, or every borderline discrepancy is traced to a documented MATLAB/Python numerical detail.
- [ ] Raw trial flags reproduce converted lick direction, context, and outcome exactly for at least three fixed spot-check trials using `np.allclose`.
- [ ] Independently histogram and smooth one raw cluster/trial and compare its full converted neural vector using `np.allclose`.
- [ ] Independently construct the expected bin-center vector and compare every input with `np.allclose`.
- [ ] Independently align one tongue, paw, and motion-energy trial from original frame times and compare continuous intermediates/classes with `np.allclose`.
- [ ] Confirm all retained trials satisfy finite go cue, have ephys, `~early`, and `~stim.enable`; do not require video because class 2 encodes its absence.
- [ ] Confirm all trial matrices have 1,000 timepoints and session neuron counts are constant; every session has at least two trials and ten neurons.
- [ ] Confirm categorical domains and class distributions, and verify visible finite samples are approximately median-split per session (allowing ties).
- [ ] Confirm no NaN/Inf in neural/input/categorical output and no conversion of true missing video into movement class 0.
- [ ] Verify saved metadata counts, thresholds, offsets, and output class orders against computations before pickling.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required positional output path and `--full` (default), `--sample`, and `--show-processing` modes. The script:
- Uses direct HDF5 reference traversal to avoid loading 100-300 MB MATLAB objects wholesale.
- Encodes the exact author context-session/probe list.
- Vectorizes spike binning into `(unit,trial,time)` arrays and applies the causal FIR along the time axis.
- Aligns DLC and motion-energy streams from original frame times using the computed SpikeGLX/video offset.
- Preserves raw visibility masks before reference fill/velocity operations.
- Performs session-wide median discretization only after retained-trial alignment.
- Emits per-session timing/audit metadata and multi-panel processing plots.
- Performs internal shape, finiteness, categorical-domain, trial-count, neuron-count, and region-index validation before saving.
- Passed `python3 -m py_compile /app/convert_data.py`; CLI help confirms all required options.

Code inefficiencies identified:
- MATLAB-style per-unit/per-trial histogram and convolution loops would be expensive in Python.
- Loading full v7.3 objects through a generic MAT reader would materialize unused waveforms and both non-ALM probes.
- Reconstructing the identical decoder time input for every trial would waste pickle space.

Code speedups added:
- Directly reads only requested HDF5 datasets/references.
- Uses indexed accumulation for binned counts and a vectorized FIR filter for smoothing.
- Processes sessions sequentially to bound peak memory and calls garbage collection after each session.
- Reuses one immutable input time array per session; pickle memoization stores shared arrays efficiently.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (session-summed) | 98 |
| Neurons / session | 32, 66 (mean 49) |
| Subjects | 2 (`JEB6`, `JEB7`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 547 |
| Trials / session | 302, 245 |
| Time from go cue range | [-2.4975, 2.4975] s; verifier rounds to [-2.5, 2.5] |
| Lick direction fractions | [0.360 left, 0.497 right, 0.143 none] |
| Context fractions | [0.329 WC, 0.671 DR] |
| Outcome fractions | [0.119 incorrect, 0.739 correct, 0.143 ignore] |
| Tongue velocity fractions | [0.063 low, 0.063 high, 0.874 not visible] |
| Paw velocity fractions | [0.478 low, 0.478 high, 0.044 not visible] |
| Motion-energy fractions | [0.500 low, 0.500 high, 0 no video] |

### Processing Plots Review
`processing_JEB6_2021-04-18.png` and `processing_JEB7_2021-04-29.png` were inspected. Neural rasters/rates use the intended go-cue-centered window; event overlays are aligned; class traces have exactly 1,000 bins; session medians split all finite velocity/energy samples approximately 50/50; and missing tongue/paw samples are explicitly class 2. High tongue missingness (87.4% of bins) reflects raw likelihood/NaN visibility, concentrated outside the licking epoch, rather than a conversion loss. No anomalies or temporal shifts were seen.

The verifier reported: `Data format is valid, no errors or warnings.` All arrays have consistent 1,000-bin time axes, finite neural data, input shape `(1, 1000)`, output shape `(6, 1000)`, and neuron-by-time neural shape.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|---|---|
| Indexed spike binning plus vectorized FIR filtering | Both sample sessions converted in 5.84 s total |
| Direct selective HDF5 reads and shared input arrays | Low I/O overhead; sample pickle saved in 0.11 s |

| Step | Time / Session | Estimated Total Time |
|---|---:|---:|
| Conversion (without plotting) | about 2.9 s | about 35 s for 12 sessions |
| Sample conversion including two plots and serialization | 4.4 s | under 1 minute projected |

The measured estimate scales by session count (12/2) and remains far below the 15-minute optimization threshold. The sample pickle is 101.7 MiB.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction | 0.6015 | 0.5959 |
| Behavioral context | 0.7483 | 0.7260 |
| Outcome | 0.5978 | 0.5873 |
| Tongue velocity | 0.5502 | 0.5457 |
| Paw velocity | 0.4460 | 0.4334 |
| Motion energy | 0.7722 | 0.7779 |

Training completed successfully on CUDA. Loss decreased monotonically from 7.851989 at epoch 1 to 0.782988 at epoch 200; test loss was 0.808316. Every validation balanced accuracy exceeded the script's reported chance (1/3 for the three-valued output schemas and 1/2 for context). Train/validation results are close, with no material sample overfitting gap.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 552.4 MiB (553 MiB filesystem display)
- `verification_full_out.txt`: created; verifier reported no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 522 (214 single) | 1-Hz filter applied to archived units | 521 (213 single) after supplied code logic | 521 (213 single) | Data/code exact; paper differs by one |
| Mean neurons/session | 43.5 inferred | 43.42 on archive | 43.42 (range 27–67) | 43.42 (range 27–67) | Yes to source/code |
| Subjects | 6 stated | Seven loader calls/IDs | 7 native IDs | 7 native IDs | Source/code exact; publication grouping differs |
| Sessions | 12 | 12 exact Figure 8 loader entries | 12 | 12 | Yes |
| Trials (total) | Not reported | Exclude early/stim for decoder; preserve ignores | 3,116 retained | 3,116 | Yes |
| Trials/session (mean) | Not reported | Same mask | 259.67 (210–390) | 259.67 (210–390) | Yes |
| Time input range | Go-cue centered | -3 to +2.5 processing | output centers [-2.4975, 2.4975] | [-2.4975, 2.4975] | Yes |
| Lick distribution | Not scalar-reported | L/R/no flags | [0.357, 0.418, 0.225] | [0.357, 0.418, 0.225] | Yes |
| Context distribution | Not scalar-reported | autowater flag | [0.315 WC, 0.685 DR] | [0.315, 0.685] | Yes |
| Outcome distribution | Not scalar-reported | miss/hit/no flags | [0.106, 0.669, 0.225] | [0.106, 0.669, 0.225] | Yes |
| Tongue velocity distribution | Not reported | side-camera tongue tracking | [0.040, 0.042, 0.919] | [0.040, 0.042, 0.919] | Yes |
| Paw velocity distribution | Not reported | bottom-camera top-paw tracking | [0.435, 0.435, 0.129] | [0.435, 0.435, 0.129] | Yes |
| Motion-energy distribution | Not reported | side-camera ME | [0.500, 0.500, 0.0003] | [0.500, 0.500, 0.0003] | Yes |

Full conversion took approximately 34 seconds plus 0.57 seconds serialization, consistent with the Step 7 estimate and well below 15 minutes. The exact retained-trial class counts were 1,112 left, 1,303 right, 701 none; 982 WC, 2,134 DR; and 329 incorrect, 2,086 correct, 701 ignore. Three sessions (first, middle, last) were spot-checked: each had 1,000 time bins, finite neural values, the correct input endpoints, expected categorical domains, and matching trial/session metadata.

The one-neuron publication discrepancy is reproducible from the supplied archive: the exact case-sensitive `findClusters(..., {'all'})` exclusions followed by the supplied equal-condition `removeLowFRClusters` logic yield 521 units and 213 well-isolated units. The publication reports 522/214, consistent with a one-unit difference between the analysis-time dataset and deposited archive. The conversion preserves the current source and executable reference logic rather than fabricating a unit. Likewise, the seven native subject IDs are preserved because no defensible mapping of one ID onto another is supplied.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: Read `verification_full_out.txt` completely. Result: `Data format is valid, no errors or warnings`; dimensions, domains, session counts, region indices, and per-session distributions are valid. There were no warnings requiring waiver.
2. **Independent raw neural check**: `critical_checks.py` independently dereferenced the original HDF5 file (without importing conversion code), selected the author ALM probe/quality labels, aligned raw spikes to raw go cues, binned, causally smoothed, applied the seven-condition >1-Hz rule, and compared the entire first-session tensor. `np.allclose(rtol=1e-5, atol=1e-5)` passed, including trial 5/neuron 3/bin 10 and every first/last output bin.
3. **Independent raw input check**: Reconstructed all 1,000 5-ms bin centers from -2.5 to +2.5 s and compared them with the decoder input using `np.allclose(rtol=0, atol=1e-7)`. Passed.
4. **Independent raw output checks**: Directly loaded L/R/no, autowater, miss/hit/no, video records, and paired motion-energy files from all 12 sources. Per-trial static labels, aggregate counts, and no-video classifications all passed `np.allclose`. Session median class ordering and missing class 2 were checked for tongue, paw, and motion energy. The one raw ME trial whose aligned frames do not overlap the requested window correctly maps entirely to `no_video`.
5. **Trial curation and sizes**: Independently reconstructed `finite(goCue) & haveEphys & ~early & ~stim.enable` for all sessions. Per-session counts and total 3,116 passed `np.allclose`. No session has fewer than two trials or fewer than 10 units.
6. **Key statistics**: Confirmed 12 source/code sessions, seven deposited IDs, 521 archived qualifying units, 3,116 retained trials, static distributions, 5-ms bins, and categorical ranges. The only paper mismatches (522/214 units and six mice) are source-version/identity discrepancies described in Step 9; source and executable code match exactly.
7. **Edge cases/off-by-one review**: Verified MATLAB 1-based spike trial numbers are decremented exactly once, probes remain explicitly MATLAB-indexed in configuration, histogram bins are left-inclusive with centers at ±2.4975 s, output cropping has exactly 1,000 bins, and first/last bins independently match raw reconstruction. Ignore trials retain raw instructed-side flags, so the explicit `no` check correctly takes precedence and produces lick class `none`. One absent aligned ME trial and per-bin tongue/paw NaNs remain class 2 rather than being imputed.

### Reference Code Comparison
| Processing stage | Conversion implementation | Supplied reference | Review result |
|---|---|---|---|
| Data loading | Direct MATLAB v7.3 dereferencing plus paired `motionEnergy` MAT | `loadSessionData`, animal ALMVideo loaders, `loadMotionEnergy` | Same 12 loader entries/probes and native fields |
| Neuron/trial filtering | Case-sensitive `findClusters('all')`, seven-condition mean >1 Hz; exclude early/stim only, retain ignores | `findClusters.m`, `removeLowFRClusters.m`, Figure 8 conditions; methods omit early | Neural logic exact; ignore retention is required by decoder task |
| Temporal alignment | Subtract per-trial `goCue`; video offset from neural/behavior bit-start modes | `alignSpikes.m`, `findPosition.m`, Figure 8 `alignEvent='goCue'` | Exact event and synchronization convention |
| Binning/smoothing | 5-ms histograms, 15-sample causal `gausswin`, reference reflect prefix; crop after filtering | Figure 8d `dt=1/200`, `getSeq.m`, `mySmooth.m` | Exact relevant high-resolution neural path; crop avoids boundary artifacts |
| Input construction | Continuous bin-center seconds in one row | Decoder specification | Exact; reference experiment supplies the alignment event rather than a decoder input array |
| Output construction | Raw static flags; side tongue/bottom top-paw velocities; side ME; per-session finite medians, class 2 missing | `findVelocity.m`, video feature loaders, raw behavior flags, decoder specification | Reference kinematics plus task-required categorization |

### Issues Found and Resolved
- The first audit-script draft treated raw L/R instruction flags as a lick on ignore trials. Inspection showed ignore trials retain an instructed side; the independent expectation was fixed to give raw `no` precedence. The converter was already correct.
- The initial no-video audit relied only on `haveVid`; one trial has `haveVid=true` but no ME samples overlapping the aligned window. The audit was corrected to use the actual video/ME frame record, matching the decoder's semantic `no video` class. The converter was already correct.
- A median audit initially expected equal low/high counts. Exact ties at the median (one session has median tongue speed zero) correctly belong to `>= median`; the audit now tests median ordering while allowing ties. The converter was already correct.
- After each audit correction, all checks were rerun from the beginning. Final output ends `ALL CRITICAL CHECKS PASSED`; no conversion-code changes were required and the full pickle remains valid.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. It fell monotonically from 8.477475 (epoch 1) to 0.784114 (epoch 200); held-out loss was 0.828242. Training completed successfully on CUDA.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction | 0.5738 | 0.5561 | Above 1/3 chance |
| Behavioral context | 0.7222 | 0.7110 | Above 1/2 chance |
| Outcome | 0.5826 | 0.5455 | Above 1/3 chance |
| Tongue velocity | 0.5613 | 0.5518 | Above 1/3 chance |
| Paw velocity | 0.5359 | 0.5253 | Above 1/3 chance |
| Motion energy | 0.7805 | 0.7119 | Above 1/3 schema chance |

`sample_trials.png` and `predictions.png` were visually inspected. Neural traces have plausible event-locked activity without truncation; the continuous input spans the complete window; static labels remain constant per trial; movement classes change at video-rate-aligned times; and predictions track both static and dynamic targets. The sole sklearn warning states that predicted classes can be absent from `y_true`: the dataset contains exactly one no-video trial (1,000 of 3,116,000 motion-energy bins), and that rare class is absent from the random validation target. Retaining it is mandatory under the requested output schema, so the warning is expected and is not a format or conversion failure.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Accuracy/chance | Expectation from paper |
|---|---:|---:|---:|---|
| Lick direction | 0.5561 | 0.3333 | 1.668 | Binary correct-choice delay ROC-AUC 0.86 ± 0.11; not the same three-class metric |
| Behavioral context | 0.7110 | 0.5000 | 1.422 | Time-resolved neural/context curves in Fig. 4b; no scalar value reported |
| Outcome | 0.5455 | 0.3333 | 1.637 | Not decoded in paper |
| Tongue velocity | 0.5518 | 0.3333 | 1.655 | Not decoded in paper |
| Paw velocity | 0.5253 | 0.3333 | 1.576 | Not decoded in paper |
| Motion energy | 0.7119 | 0.3333 | 2.136 | Used as a movement signal, not reported as a decoding target |

All outputs exceed chance. Five exceed the requested 1.5×-chance screening level. Context is 1.422× chance, so it received the full low-accuracy audit: raw context flags were checked for all trials (including explicit checks of JEB6 raw trials 0, 92, and 140); the alignment and prediction plots were inspected; both classes are well represented (31.5% WC/68.5% DR overall and both occur in every session); neuron filtering was rechecked against `findClusters.m` and `removeLowFRClusters.m`; and go-cue/video synchronization was independently audited. No bug or alternative reference-consistent conversion was found. A 0.711 held-out balanced accuracy is also consistent with the paper's qualitative finding of persistent but imperfect context coding.

The paper contains one scalar neural decoding result: binary delay-epoch choice ROC-AUC 0.86 ± 0.11. The main decoder score is lower because its target includes a third `none` class, all outcomes/contexts, time bins outside the delay, and reports balanced accuracy rather than ROC-AUC. To test the converted signal on the paper's target instead of dismissing this difference, `accuracy_audit.py` trained a session-wise ridge-logistic benchmark on correct DR left/right trials and mean -0.9-to-0-s neural activity with repeated 70/30 splits. It achieved ROC-AUC **0.9238 ± 0.0844** across the 12 sessions, meeting/exceeding the paper result and confirming preserved choice information.

Train/validation ratios were 1.032 (lick), 1.016 (context), 1.068 (outcome), 1.017 (tongue), 1.020 (paw), and 1.096 (motion energy), all far below the 1.5 overfitting threshold. There is no concerning train/validation gap or evidence of leakage.

### Issues Found and Resolved
- **Context below 1.5× chance screen**: Exhaustive raw labels, variation, alignment, curation, and processing checks passed; 0.711 is a valid biological/architecture result, not a conversion defect.
- **Main lick score below paper AUC**: A like-for-like audit reached 0.9238 ± 0.0844 ROC-AUC. Difference resolved as target/metric/window composition, supported by a successful direct benchmark rather than assumption.
- **Rare ME class warning**: Exactly one trial has no aligned video/ME. It is independently confirmed in the raw paired files and must remain class 2; random validation has no true examples of that rare class. No change is appropriate.
- No code/data corrections were indicated, so reconversion or retraining was unnecessary after the full set of checks.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading example, schema, statistics, caveats, and reproduction commands
- [x] cache/ folder created with `README_CACHE.md`
- [x] Independent audit/benchmark scripts and logs moved to cache; required datasets, run logs, plots, converter, and documentation remain at the project root

Final artifact review confirmed that every required file exists: `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`, `README.md`, both conversion logs, both verification logs, both decoder logs, and the processing/decoder plots. The full pickle is 552.4 MiB and the sample pickle is 101.7 MiB. All workflow steps are complete.
