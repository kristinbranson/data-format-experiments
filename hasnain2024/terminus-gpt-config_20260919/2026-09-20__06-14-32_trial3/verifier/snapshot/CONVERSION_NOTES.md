# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse (provided paper/code/data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- numpy 2.4.4
- torch 2.6.0+cu124
- Mandatory checkpoint passed: `/app/CONVERSION_NOTES.md` exists.

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
| `getDefaultParams` | `DataLoadingScripts/getDefaultParams.m` | PROCESSING | Sets go-cue alignment, -2.5 to +2.5 s window, 5 ms bins (200 Hz), smoothing=15, low-FR=0.5 Hz, ALM probe, and canonical conditions. |
| `loadObjs` / `loadSessionData` / `processData` | `DataLoadingScripts/` | LOADING | Loads session objects and orchestrates trial selection, cluster selection, alignment, rates, video and behavior processing. |
| `findTrials` | `DataLoadingScripts/findTrials.m` | CURATION | Evaluates logical condition expressions against Bpod/task fields to return trial indices. |
| `findClusters` | `DataLoadingScripts/findClusters.m` | CURATION | Selects clusters according to requested quality labels. Default quality is `all`. |
| `alignSpikes` | `DataLoadingScripts/alignSpikes.m` | PROCESSING | Subtracts each spike's trial-specific event time; default event is `goCue`. |
| `removeLowFRClusters` | `DataLoadingScripts/removeLowFRClusters.m` | CURATION | Keeps neurons with mean PSTH firing rate strictly >0.5 spikes/s and filters PSTH/trial data consistently. |
| `baselineFR` / `getFiringRate` | `DataLoadingScripts/baselineFR.m`, `funcs/getFiringRate.m` | PROCESSING | Histogram/rate and trial-epoch firing-rate operations; bins use `dt`; PSTHs are smoothed with `mySmooth`. |
| `loadKinData` / `getKinematicsFromVideo` / `findVelocity` | `DataLoadingScripts/loadKinData.m`, `funcs/kinematics/` | PROCESSING | Aligns tracked x/y positions, creates displacement and x/y velocity features, tongue geometry, and appends motion energy. |
| `loadMotionEnergy` | `DataLoadingScripts/loadMotionEnergy.m` | PROCESSING | Loads companion motion-energy files, interpolates video values to neural/object time relative to go cue, nearest-fills gaps, retains missing sessions as NaN. |
| `UseInclusionCritera` | `utils/UseInclusionCritera.m` | CURATION | Keeps sessions with >40 right-hit and >40 left-hit trials. |
| `getOutcome` | `funcs/getOutcome.m` | PROCESSING | Derives trial outcomes from behavioral fields. |
| Neural/DLC choice and context decoders | `ChoiceContextDecoding/*.m` | PROCESSING | Balance four right/left x DR/WC conditions, construct neural or kinematic predictors, and split train/test data. |

### Notes
- Repository is MATLAB code for Hasnain, Birnbaum et al. (Nature Neuroscience 2024). Data objects are named `data_structure_Animal_SessionDate.mat`; ephys sessions have companion `motionEnergy_Animal_SessionDate.mat` files.
- Recordings are from right and left ALM. Loader scripts contain 16 recording animals and 50 active session-date entries; commented entries are excluded by the authors.
- No calcium imaging is present, so delta-F/F is not applicable. Neural data are electrophysiological spike-derived firing rates.
- Reference defaults: `alignEvent='goCue'`; `tmin=-2.5`, `tmax=2.5` s; `dt=1/200=0.005` s; smoothing parameter 15; `lowFR=0.5` Hz; time warping disabled.
- Canonical conditions are right/left hit without stimulation in delayed-response (`~autowater`) and water-cued (`autowater`) contexts, excluding early trials.
- Spike alignment is trial-specific: `trialtm_aligned = trialtm - event`.
- Default cluster quality accepts all quality labels, then low-rate clusters are removed at >0.5 Hz.
- Video features contain x/y displacement and x/y velocity for tracked parts. The code preserves invisibility intervals in `kin.nans`; tongue velocity NaNs are set to zero only for feature computation. This means target class `not visible` must be reconstructed from masks rather than treating zero as visible low velocity.
- Motion energy is aligned by frame times and go-cue times, interpolated to the common time axis, and nearest-filled. Missing files produce NaN/no-video data.
- `UseInclusionCritera` requires strictly more than 40 right-hit and 40 left-hit trials per session.
- Neural and DLC reference decoders balance the four principal conditions before decoding choice or context. This balancing is analysis-specific and should not discard other valid trials required for the requested outcome/lick-direction targets.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is 16 GB and contains 165 MAT files plus `fetch_complete`.
- Four task-family directories are present: `Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`, `DelayInhibition_BilatMC_Behavior`, and `GoCueInhibition_BilatMC_Behavior`.
- There are 120 `data_structure_*.mat` session files across 18 subjects and 45 companion `motionEnergy_*.mat` files.
- Neural conversion scope is the ephys directories. They contain 47 sessions; 45 have cluster data and 2 (`JEB24_2023-10-03`, `JEB24_2023-10-04`) lack `obj.clu` and cannot be neural-decoder sessions.
- Neural-bearing sessions span 14 subjects: EKH1, EKH3, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24, JEB6, JEB7, JGR2, JGR3.
- Native formats are heterogeneous: earlier sessions are MATLAB v7.3/HDF5; 11 later randomized-delay sessions are MATLAB v5 and require `scipy.io.loadmat`.
- Native object fields include `bp` (behavior), `clu` (spike clusters), `traj` (two camera views), `trials` (cross-stream mappings/validity), `sglx`, paths/metadata, and in some later files embedded `me`/`ex`.
- Behavior fields: `R`, `L`, `autowater`, `hit`, `miss`, `no`, `early`, `stim`, `protocol`; event fields include `goCue`, `lickL`, `lickR`, `reward`, `sample`, `delay`, and `bitStart`.
- `obj.trials.bp` supplies `haveEphys`, `haveVid`, `sglxFileNum`, and `vidFileNum`; these explicitly identify valid/matched streams and must be honored.
- Cluster fields include spike times (`tm`), trial-relative times (`trialtm`), trial indices, quality, waveform, and channel/site.
- Each trajectory view has per-trial `ts`, `frameTimes`, `featNames`, filename, and dropped-frame count. `ts` contains coordinate, likelihood, and tracked-feature axes (MATLAB/HDF5 axis order differs). Example views contain 7 and 10 tracked features.
- Motion-energy files are MATLAB v5 with struct `me`; 45 companions exist. All 45 neural-bearing sessions have motion energy; the two no-cluster JEB24 sessions are the only ephys-directory sessions without companions.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 14,297 raw clusters before reference low-FR filtering |
| Neurons / session | 17–1,258 raw clusters |
| Subjects | 18 overall; 14 with neural data |
| Sessions / subject | 1–24 overall; neural sessions vary by subject |
| Trials (total) | 15,324 across 45 neural-bearing sessions |
| Trials / session | 230–517 (mean 340.5) |

Additional native counts for the 45 neural sessions: right 7,636; left 7,688; hit 11,417; miss 1,906; no/ignore 2,001; early 988; autowater/WC 1,548. `haveEphys` sums to 15,324 and `haveVid` to 15,325 (the latter one-count discrepancy reflects native stream bookkeeping and will be checked per trial during conversion).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | DR: 1,651 units; randomized delay: 845 units | “For the DR task, we recorded 1,651 units… Finally, for the randomized delay task, we recorded 845 units…” |
| Neurons / session | At least 10 included units | “Recording sessions were included… only if they had at least 10 units.” |
| Subjects | DR: 9 mice; randomized delay: 4 mice | Methods electrophysiology analysis |
| Sessions / subject | DR: 25 sessions/9 mice; randomized: 19 sessions/4 mice | Methods electrophysiology analysis |
| Trials (total) | Not stated globally | — |
| Trials / session | Behavioral inclusion: >=40 correct DR/side and >=20 correct WC/side | Methods behavioral analysis |
| Neural data time bin | Reference code 5 ms (`dt=1/200`); paper does not state a different bin | `getDefaultParams.m` |
| Behavior data time bin | Video acquired at 400 Hz; reference code interpolates to common 200 Hz object axis | Methods videography; code |
| Reward rate | Not stated globally | — |
| Two-context subset | 522 units (214 single units), 12 sessions, 6 mice | Methods electrophysiology analysis |
| Early/ignore handling | Omitted from paper behavioral analyses | “excluding early lick and ignore trials, which were omitted from all analyses” |

### Processing Details
- Electrophysiology was recorded from ALM at 25 kHz and spike-sorted with JRCLUST and/or Kilosort 3, with manual Phy curation.
- Units were categorized as well-isolated single units or multiunits. The paper used all units above 1 Hz for most analyses; selected subspace/single-unit analyses used only well-isolated single units above 1 Hz.
- Reference code aligns spikes trial-by-trial to go cue and uses a -2.5 to +2.5 s window sampled at 200 Hz (5 ms).
- High-speed video was acquired at 400 Hz from side and bottom cameras. Tongue, jaw, and nose were tracked in both cameras; paws only in bottom view.
- Position is from DeepLabCut x/y output. Missing values are nearest-filled for all features except tongue. Velocity is the first derivative of position. Tongue angle and length use the bottom camera.
- Motion energy is the absolute difference between medians of the next and previous five frames (12.5 ms each), reduced to one value/frame using the 99th percentile across pixels. The paper’s moving/not-moving threshold was manually chosen per session, but this decoder task instead mandates a per-session median threshold.
- Choice/context decoding in the paper used ridge-regularized models at each time bin, four-fold cross-validation, 30% held-out testing trials, balanced trial conditions, and shuffled-label chance estimates.

### Curation Steps

**Neuron curation rules**:
- Sessions require at least 10 units.
- Most paper analyses include manually curated single- and multiunits with firing rate >1 Hz.
- Generic reference loader default is >0.5 Hz and quality=`all`; this conflict is investigated in Step 4.

**Trial curation rules**:
- Paper behavioral analyses exclude early and ignore trials.
- Behavioral session inclusion requires at least 40 correct DR trials per direction and 20 correct WC trials per direction.
- Decoder analyses balance relevant right/left and DR/WC conditions.
- The requested target explicitly requires an `ignore` outcome, so ignore trials may need retention despite paper analysis exclusions; early trials remain candidates for exclusion.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from neural population | Shown as four-fold CV time course in Fig. 3; no numeric accuracy stated in prose |
| Choice from kinematics | Shown as four-fold CV time course in Fig. 3; no numeric accuracy stated in prose |
| Context from neural population | Shown as four-fold CV time course in Fig. 5; no numeric accuracy stated in prose |
| Context from kinematics | Shown as four-fold CV time course in Fig. 5; no numeric accuracy stated in prose |
| Choice from CDchoice | ROC-AUC shown in Extended Data Fig. 2; no numeric aggregate stated in prose |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session scope | Active recording loaders select session/date/probe combinations | 47 ephys-directory files; 44 match active loaders; JEB23 2023-10-20 is commented out and JEB24 2023-10-03/04 lack clusters | 25 fixed-delay + 19 randomized-delay sessions | Use the 44 active, available sessions: exactly 25 + 19. |
| Missing active sessions | Loaders include six JEB4/JEB5 sessions | Those files are absent | Paper totals include 25 + 19 sessions | Process all active sessions available in supplied data; document unavailable JEB4/JEB5 files. Available replacements/session composition still exactly matches 44 paper session count. |
| Probe selection | Each loader selects probe 1, probe 2, or both; three JEB15 sessions concatenate both | Native objects often contain two probe cells | Paper reports ALM units | Follow loader probe selection exactly. JEB15 2022-07-26/27/28 use both; 2022-07-29 uses probe 2 only. |
| Garbage clusters | `quality='all'` in defaults, but curated analysis data exclude garbage | 10,931/14,297 raw clusters explicitly labeled garbage | Paper describes manually curated single/multiunits | Exclude explicit garbage/corrupted-garbage labels; retain curated multi/fair/poor/good/great/excellent labels as reference “all units.” |
| Firing-rate threshold | Executable default is >0.5 Hz | Selected sessions/probes yield 2,498 units >0.5 Hz | Paper prose says >1 Hz and reports 2,496 total units (1,651 + 845) | Use >0.5 Hz because it reproduces reported aggregate within 2 units; >1 Hz yields only 2,459. Document text/code mismatch. |
| Trial exclusions | Decoder analyses balance correct/miss conditions; behavioral analyses omit early and ignore | Raw data contain hit, miss, no, early flags | Paper omits early and ignore | Exclude early trials. Retain ignore because requested target explicitly requires an ignore class. Do not balance/drop valid trials for the converted dataset. |
| Time base | Code defaults to -2.5..+2.5 s, dt=5 ms, go-cue alignment | Raw spike `trialtm` is from trial start and go cue varies by trial | Paper aligns key results to go cue | Subtract trial-specific go cue and bin on the common reference axis. |
| Video rate | Neural/object axis is 200 Hz after interpolation | Raw video is ~400 Hz with per-trial frame times | Methods state 400 Hz acquisition | Interpolate derived behavior to the common 5 ms axis as reference code does. |
| Motion threshold | Paper manually separates bimodal movement distributions | Raw companion motion-energy stream available | Decoder task mandates 50th percentile | Use per-session median threshold, overriding the paper manual movement threshold exactly as requested. |

Final consistent understanding: load only active available ephys sessions and selected probes; remove explicit garbage clusters; compute go-cue-aligned 5-ms firing-rate matrices on -2.5 to +2.5 s; retain units above 0.5 Hz; exclude early trials but retain hit, miss, and no-response trials; align video/motion to the same axis using frame times and go-cue timestamps.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Selected `obj.clu{probe}` spike `trialtm`, `trial`, `quality` | `neural` | Exclude garbage; subtract trial go cue; histogram in 5-ms bins from -2.5 to +2.5 s; convert to spikes/s; smooth with reference `mySmooth` equivalent; retain mean rate >0.5 Hz | `alignSpikes`, `getPSTHs`, `removeLowFRClusters` | Shape neurons x 1000 time bins; selected probes follow loaders. |
| Common bin centers | `input[0]` | Broadcast signed seconds from go cue across every trial | `getDefaultParams` | Continuous time-varying decoder input, shape 1 x 1000. |
| `bp.R/L`, `bp.hit/miss/no` | `output[0]` lick direction | right=`R&hit` or `L&miss`; left=`L&hit` or `R&miss`; none=`no` | condition definitions and choice decoders | Per-trial categorical scalar. |
| `bp.autowater` | `output[1]` behavioral context | 0=DR (`false`), 1=WC (`true`) | default condition expressions | Per-trial categorical scalar. |
| `bp.hit/miss/no` | `output[2]` outcome | 0=incorrect/miss, 1=correct/hit, 2=ignore/no | `getOutcome`, behavior fields | Per-trial categorical scalar; retain ignore trials by task requirement. |
| DLC tongue x/y coordinates + visibility | `output[3]` tongue velocity | derive speed magnitude from first derivative; align/interpolate to 5-ms axis; threshold visible samples by per-session median; class 2 where tongue not visible | `getKinematicsFromVideo`, `findVelocity`, invisibility masks | Time-varying categorical series. Use tracked tongue point consistently across sessions/views; visibility from finite coordinates/likelihood-derived NaNs, never infer visibility from zero speed. |
| DLC paw x/y coordinates + visibility | `output[4]` paw velocity | derive speed magnitude, align/interpolate, median split visible samples; class 2 where paw not visible | same kinematics functions | Bottom-view paw. Time-varying categorical series. |
| companion `me.data` + frame times | `output[5]` motion energy | align to go cue and interpolate to 5-ms axis; per-session median split; class 2 only if session has no video/motion stream | `loadMotionEnergy` | Time-varying categorical series. Do not encode ordinary interpolation gaps as no-video. |
| filename animal ID | `subjects`, `subject_idx` | unique stable subject list and per-session index | loader metadata | 14 available recording subjects expected. |
| selected probe neurons | `brain_region_idx` | all neurons map to ALM | README, methods, loader `probeArea` | `brain_regions=['ALM']`. |

### Key Decisions
1. **Session inclusion**: use the 44 sessions that are both present and active in reference recording loaders. This exactly gives 25 fixed-delay and 19 randomized-delay sessions; exclude commented JEB23 2023-10-20 and no-cluster JEB24 sessions.
2. **Neuron curation**: follow loader probe choices, exclude explicit garbage labels, retain all curated quality categories, and apply strict mean-rate >0.5 Hz. This matches executable code and nearly exactly reproduces paper aggregate units.
3. **Trial curation**: require valid ephys mapping, finite positive go cue, and `~early`. Retain no/ignore trials because the target requires them. Exclude stimulation trials if any are present, matching canonical no-stimulation reference conditions.
4. **Time axis**: use 1000 half-open 5-ms bins spanning [-2.5, 2.5) s, represented by centers from -2.4975 to 2.4975 s. This avoids a 1001-point edge/center ambiguity and matches histogram bin width.
5. **Neural representation**: single-trial binned firing rates in spikes/s, float32, with the same temporal smoothing convention as reference PSTHs applied along time. No z-scoring, because reference trial data are rates and the target asks for neural activity.
6. **Per-trial vs time-varying outputs**: lick direction, context, and outcome are per-trial scalars; tongue velocity, paw velocity, and motion energy are time-varying because their timing relative to go cue is scientifically relevant.
7. **Velocity threshold population**: calculate one median per session using all finite/visible values across included trials and time bins. Values below median=0, >=median=1, not visible=2.
8. **Motion threshold population**: calculate one median per session over finite aligned motion-energy values. Values below median=0, >=median=1; class 2 is reserved for sessions/trials with no video stream as specified.
9. **Behavior feature selection**: use a consistently named tongue feature and bottom-view paw feature. If multiple paw/tongue points exist, use a fixed anatomically corresponding point and speed magnitude, not separate x/y categories. Exact names will be discovered programmatically and logged per session.
10. **Missing data**: preserve categorical missingness. Never treat missing tongue/paw values as low velocity. Interpolate only within valid visible stretches/reference behavior; outside video support remains not visible (tongue/paw) or no-video where applicable.

### Planned Sanity Checks
- [ ] Recompute one selected session/probe neuron list directly from raw quality and spike fields; `np.allclose` converted neural trial/bin values to an independently histogrammed raw trial.
- [ ] `np.allclose` converted time input to independently generated bin centers for multiple trials/sessions.
- [ ] Independently derive raw right/left/context/outcome values for at least three hit/miss/ignore trials and compare with converted outputs using `np.allclose`.
- [ ] Independently align raw video frame times to go cue for tongue/paw and compare selected converted categorical samples after applying saved session medians.
- [ ] Independently align companion motion energy and compare selected converted samples after applying saved median.
- [ ] Verify 25 fixed-delay + 19 randomized sessions, >=2 trials/session, 14 subjects, all ALM region indices, uniform 5-ms/1000-bin dimensions.
- [ ] Compare retained neuron total to reference expectation (~2,498 from code-based curation; paper 2,496 aggregate).
- [ ] Check output class distributions, visible/no-video fractions, and median-split classes (approximately balanced among visible finite samples).

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with dual MATLAB v7.3/HDF5 and v5 loaders, automatic parsing of active author session/probe loaders, neuron/trial curation, go-cue spike alignment, 5-ms firing-rate construction, video/motion alignment, session-median discretization, metadata, assertions, plots, and required CLI modes. Script passes `py_compile` and `--help`.

Implementation uses float32 neural/input arrays and int8 outputs. Bulk data are processed session-by-session; only the final converted lists are retained. HDF5 files are accessed lazily. Processing prints per-session timing.

Code inefficiencies identified:
- Raw trajectory data are large and per-trial MATLAB references require iteration.
- Single-trial spike histograms require unit/trial grouping.

Code speedups added:
- Lazy HDF5 dereferencing, float32/int8 storage, vectorized alignment, and processing one session at a time.
- Sample mode limits conversion to two sessions and plotting to at most two.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 114 |
| Neurons / session | 48, 66 |
| Subjects | 2 (EKH1, EKH3) |
| Sessions / subject | 1 each |
| Trials (total) | 642 |
| Trials / session | 252, 390 |
| Time input range | [-2.4975, 2.4975] s |
| Lick direction distribution | left 0.431, right 0.403, none 0.165 |
| Context distribution | DR 0.755, WC 0.245 |
| Outcome distribution | incorrect 0.061, correct 0.774, ignore 0.165 |
| Tongue velocity distribution | below 0.052, above 0.052, not visible 0.895 |
| Paw velocity distribution | below 0.325, above 0.325, not visible 0.350 |
| Motion energy distribution | below 0.500, above 0.500, no-video 0.000 |

### Processing Plots Review
`processing_EKH1_2021-08-07.png` and `processing_EKH3_2021-08-11.png` were created. Plots show neural activity, raw aligned tongue/paw speed and motion energy on the same go-cue axis, zero-time markers, and session thresholds. No gross alignment discontinuity or threshold inversion was found. Tongue missingness is high because the tracked tongue is physically invisible except during protrusion; visible samples are balanced exactly at the session median. Paw missingness varies by session and remains explicitly classed as not visible.

Format verification: valid with no errors or warnings. All arrays are finite after categorical missing-data encoding; neural ranges are 0–251.6 and 0–327.7 spikes/s.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Lazy HDF5 reads, session-wise processing, compact dtypes | Full run estimated in minutes rather than >15 min |

| Step | Time / Session | Estimated Total Time |
| Conversion | 2.9–4.8 s for first two sessions | approximately 3–6 min allowing larger Neuropixels sessions and v5 loading |
| Validation | <10 s for sample | <1 min estimated full summary |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| lick_direction | 0.7375 | 0.6859 |
| behavioral_context | 0.8687 | 0.8478 |
| outcome | 0.6711 | 0.5022 |
| tongue_velocity | 0.5934 | 0.5858 |
| paw_velocity | 0.5007 | 0.5020 |
| motion_energy | 0.7696 | 0.7616 |

Loss decreased steadily from >3.3 to 0.6726 over 200 epochs; test loss was 0.6849. Every validation accuracy was above the script-reported chance level. Train-validation gaps were small (largest for outcome, 0.169 absolute, still below a 1.5x ratio).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3.329 GB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; valid with no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 2,496 aggregate (1,651 fixed DR + 845 randomized) | selected probes, non-garbage, >0.5 Hz gives 2,498 | 14,297 raw; 3,366 non-garbage all probes | 2,498 | Yes, within 2 units; exact code-based count |
| Mean neurons/session | not stated | selected probe(s) | — | 56.8 (range 17–142) | Consistent with >=10/session |
| Subjects | 9 fixed-task and 4 randomized cohorts; overlap/context subsets described separately | 14 available active-loader animals | 14 neural subjects in supplied active data | 14 | Matches supplied data/code |
| Sessions | 25 fixed + 19 randomized | 44 active available sessions | 47 files; 3 excluded by loader/no clusters | 44 | Exact |
| Trials (total) | not stated | exclude early; canonical no-stim | 15,324 before curation | 13,762 | Consistent with documented filtering |
| Trials/session | behavioral minimum criteria | >=2 required | 230–517 raw | 193–474 converted | All sessions valid |
| Time input range | go-cue aligned | -2.5 to +2.5 s, 5 ms | trial-specific go cues | [-2.4975, 2.4975] centers | Exact |
| Lick distribution | not globally stated | source hit/miss mapping | raw behavior flags | [0.422686, 0.446011, 0.131304] | Plausible |
| Context distribution | not globally stated | `autowater` | raw behavior flag | DR 0.903430, WC 0.096570 | Plausible |
| Outcome distribution | ignore omitted in paper analyses | hit/miss/no source flags | raw behavior flags | incorrect 0.119677, correct 0.749019, ignore 0.131304 | Plausible |
| Tongue velocity | visibility scientifically meaningful | retain tongue NaNs/mask | tongue visible around protrusions | below 0.046681, above 0.046683, invisible 0.906636 | Median-balanced visible values |
| Paw velocity | no global fraction | nearest-fill non-tongue, preserve no-video support | variable view support | below 0.366212, above 0.366214, invisible 0.267574 | Median-balanced visible values |
| Motion energy | per-session bimodal; task requests median | aligned/interpolated | companion file all neural sessions | below 0.498746, above 0.501182, no-video 0.000073 | Median-balanced; one missing trial |

Iterations resolved before completion:
1. Legacy nested motion-energy schema was unwrapped as in reference code.
2. Incorrect symmetric sigma-15 smoothing was replaced with exact reference 15-point causal Gaussian-window convolution.
3. Sixty-one trailing trials in two JEB24 sessions had no neural samples despite incorrect native validity flags; these invalid periods were excluded. Revalidation then reported no warnings.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: corrected full `verification_full_out.txt` says “Data format is valid, no errors or warnings.” Earlier 61 all-zero warnings were fixed by excluding invalid trailing recording periods and re-running conversion/validation.
2. **Independent raw neural sanity (`np.allclose`)**: loaded EKH1 native HDF5 directly (without importing conversion code), selected probe 2/raw curated units, independently aligned spikes, histogrammed the first converted trial, and applied the 15-point causal Gaussian kernel. Full 1000-bin trace matched exactly (`True`, max difference 0.0).
3. **Independent input sanity (`np.allclose`)**: independently generated 5-ms bin centers; full input trace matched (`True`).
4. **Independent output sanity (`np.allclose`)**: loaded native R/L/hit/miss/no/autowater flags for the same raw trial, independently derived actual lick direction, context, and outcome; matched converted values (`True`). All per-trial target rows were verified constant over time.
5. **Reference code comparison**:
   - Loading: HDF5/v5 native `obj` and companion motion files; matches `loadObjs`/`loadMotionEnergy` source schemas.
   - Neuron filtering: active loader probe choices, explicit garbage exclusion, strict >0.5 Hz; reproduces 2,498 reference-code units.
   - Trial filtering: valid ephys, finite go cue, non-early, non-stim; ignore retained only because target explicitly requires it.
   - Alignment: raw `trialtm - goCue[trial]`, exactly as `alignSpikes`.
   - Binning: 5-ms `histc`-equivalent bins on [-2.5,2.5), rates in spikes/s.
   - Smoothing: exact `mySmooth` logic—`gausswin(15)`, first half zeroed, normalized, `same` convolution.
   - Input: common bin-center time series.
   - Outputs: behavior flags and reference hit/miss choice mapping; video frame-time/go-cue interpolation; explicit invisibility; median thresholds mandated by task.
6. **Key statistics**: 44 sessions = 25 fixed + 19 randomized; 14 subjects; 13,762 valid trials; 2,498 neurons; 17–142 neurons/session. Aggregate neuron count differs from paper 2,496 by only two and exactly matches executable code-based curation on supplied files.
7. **Edge cases/off-by-one**: verified 1000 bins with centers -2.4975 and +2.4975; half-open final edge at +2.5; MATLAB 1-based trial indices converted once to 0-based; all region-index lengths match; all sessions >=2 trials; all arrays finite; no all-zero neural trials.
8. **Median thresholds**: visible tongue/paw classes are balanced within rounding. Some motion sessions have unequal class counts because repeated samples equal the median; this is correct under the explicit `< median` versus `>= median` definition, not a threshold bug.

### Issues Found and Resolved
- **Nested motion-energy struct**: unwrapped legacy `me.data.data`, matching reference loader.
- **Incorrect initial smoothing interpretation**: replaced sigma=15 Gaussian filtering with exact 15-point causal Gaussian-window smoothing; repeated sample and full conversion/validation.
- **Invalid trailing neural periods**: excluded 28 trials from JEB24 2023-10-23 and 33 from JEB24 2023-11-03 after native clusters ended; repeated all checks and eliminated validator warnings.
- **Unavailable/reference-excluded sessions**: six active JEB4/JEB5 files are not supplied; JEB23 2023-10-20 is explicitly commented out; two JEB24 files lack clusters. Final scope exactly matches 25+19 paper session counts.

Independent check output is preserved for cleanup into `cache/independent_sanity_out.txt`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, from 8.9589 at epoch 1 to 0.7499 at epoch 200.
- Test loss: 0.7853.
- Full 44-session run finished successfully on GPU with `--plot-samples`.
- sklearn emitted one warning that `y_pred` contained classes absent from `y_true`; this can occur for rare classes in an individual held-out subset (especially no-video/ignore) and is not a data-format warning. Full format verification remained warning-free.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| lick_direction | 0.6133 | 0.5932 | 1.78x chance |
| behavioral_context | 0.8404 | 0.8312 | 1.66x chance |
| outcome | 0.6059 | 0.5836 | 1.75x chance |
| tongue_velocity | 0.5242 | 0.5227 | 1.57x chance |
| paw_velocity | 0.5111 | 0.5073 | 1.52x chance |
| motion_energy | 0.7632 | 0.7044 | 2.11x script-reported chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Ratio to Chance | Expectation from Paper |
|----------|---------------------|--------|-----------------|------------------------|
| lick_direction | 0.5932 | 0.3333 | 1.78x | Paper Fig. 3 shows above-chance time-resolved neural choice decoding; no numeric aggregate in prose |
| behavioral_context | 0.8312 | 0.5000 | 1.66x | Paper Fig. 5 shows above-chance neural context decoding; no numeric aggregate in prose |
| outcome | 0.5836 | 0.3333 | 1.75x | Not directly reported |
| tongue_velocity | 0.5227 | 0.3333 | 1.57x | Not directly reported as this categorical target |
| paw_velocity | 0.5073 | 0.3333 | 1.52x | Not directly reported as this categorical target |
| motion_energy | 0.7044 | 0.3333 script-reported | 2.11x | Paper analyzes/predicts movement but does not report this categorical decoder accuracy |

Checks performed:
1. **Accuracy versus chance**: every output exceeds chance and exceeds 1.5x chance. No low-accuracy debugging trigger remains.
2. **Comparison to paper**: every neural decoding accuracy reported in prose was searched. Choice/context accuracies are presented graphically as time courses, not numerical aggregate values; achieved results are qualitatively consistent with robust above-chance decoding.
3. **Train-validation gap**: ratios are lick 1.03, context 1.01, outcome 1.04, tongue 1.00, paw 1.01, motion 1.08. None approaches the >1.5x overfitting threshold.
4. **Class variation**: full verification shows all requested classes globally. The no-video class is intentionally extremely rare because all selected neural sessions have companion video/motion; one native trial lacks usable video.
5. **Alignment/label checks**: independent raw checks in Step 10 exactly matched neural bins, time input, and behavioral labels. Processing plots show all streams on the same go-cue axis.

### Issues Found and Resolved
- No conversion issue was revealed by full decoder accuracy.
- The sklearn warning (`y_pred contains classes not in y_true`) reflects a rare class absent from an evaluation subset, not globally missing labels. It cannot be fixed without changing valid fold composition or fabricating/dropping data, so it is retained and explained.
- Initial over-smoothed conversion produced artificially higher sample accuracies; exact reference smoothing lowered them but remained robustly above chance. The corrected exact-reference results are final.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with README_CACHE.md
- [x] Investigation scripts/results moved to cache; reference code directory restored
- [x] All required conversion, validation, plotting, and decoder outputs retained in `/app`
