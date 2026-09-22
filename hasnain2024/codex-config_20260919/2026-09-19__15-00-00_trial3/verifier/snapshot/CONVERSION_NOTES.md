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

Environment check: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124 import successfully. Checkpoint confirmed with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadObjs` / `loadSessionData` | `DataLoadingScripts/loadObjs.m`, `loadSessionData.m` | LOADING | Load one MATLAB `obj` per session; process each selected ALM probe and concatenate dual probes. |
| `findTrials` | `DataLoadingScripts/findTrials.m` | CURATION | Evaluate Boolean task-condition expressions against `obj.bp` and return 1-based trial indices. |
| `findClusters` | `DataLoadingScripts/findClusters.m` | CURATION | Select requested quality labels; `all` excludes garbage/gabrga/noisy/real?. |
| `alignSpikes` | `DataLoadingScripts/alignSpikes.m` | PROCESSING | Subtract each spike's trial-specific alignment-event time (here `goCue`). |
| `getSeq` | `DataLoadingScripts/getSeq.m` | PROCESSING | Bin aligned spikes in `[tmin,tmax)` and causally smooth counts/`dt`; output `trialdat` shaped time × unit × trial. |
| `removeLowFRClusters` | `DataLoadingScripts/removeLowFRClusters.m` | CURATION | Retain units whose mean trial-averaged firing rate is strictly above `params.lowFR`. |
| `findVideoOffset` | `funcs/findVideoOffset.m` | PROCESSING | Compute video/neural clock shift as `(mode(bitcode.bitstart)/fs) - mode(bp.ev.bitStart)`. |
| `loadMotionEnergy` | `DataLoadingScripts/loadMotionEnergy.m` | PROCESSING | Interpolate 400-Hz motion energy onto aligned neural time using frame times minus video shift and event time; nearest-fill edge NaNs. |
| `getKinematicsFromVideo` / `findVelocity` | `funcs/kinematics/` | PROCESSING | Interpolate DLC coordinates and calculate coordinate gradients; preserve tongue invisibility via missing tracking before downstream handling. |
| `getOutcome` | `funcs/getOutcome.m` | PROCESSING | Use hit as correctness and replace `bp.no` (ignore) with NaN. |

### Notes
- Native session files are MATLAB `data_structure_Animal_Date.mat` objects containing Bpod trial/event data (`bp`), sorted ephys (`clu`), SpikeGLX metadata (`sglx`), DLC trajectories (`traj`), and metadata (`ex`). Separate motion-energy files accompany ephys sessions.
- This is electrophysiology, not imaging: delta-F/F is not applicable.
- The tutorial describes `excellent`, `great`, and `good` as single units, while the standard current pipeline's `quality={'all'}` includes all labels except explicit bad labels. The paper-specific tutorial uses 10-ms bins (`dt=1/100`), `[-2.5,2.5]` s about go cue, 15-sample causal Gaussian smoothing with reflected boundaries, and a 1-Hz low-rate cutoff. `getDefaultParams` differs (5-ms bins, 0.5-Hz cutoff), so paper-analysis scripts and text must decide the final settings.
- Decoder scripts balance relevant trial classes and decode choice/context from `obj.trialdat`; their 75-ms windowing is analysis-specific and does not change stored neural preprocessing.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is 16 GB and contains four experiment collections: `Ephys_Behavior` (25 session objects + 25 motion-energy files), `RandomizedDelay_Ephys_Behavior` (22 objects + 20 motion-energy files), `DelayInhibition_BilatMC_Behavior` (53 behavior-only objects), and `GoCueInhibition_BilatMC_Behavior` (20 behavior-only objects).
- The decoder needs neural activity plus both behavioral contexts (WC and DR), so the applicable source collection is the 25-session `Ephys_Behavior` cohort. Inhibition collections have no ephys. Randomized-delay sessions test a different task variant and do not provide the alternating WC/DR context required here; reference loaders also exclude some files in that folder. They are therefore inventoried but not merged into the target cohort.
- Session objects are MATLAB v7.3/HDF5 files. Top-level `obj` fields are `bp`, `clu`, `meta`, `pth`, `sglx`, `traj`, `trials`. `bp` contains trial labels (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, stimulation) and event times (`sample`, `delay`, `goCue`, reward, left/right lick cells). `clu` contains per-probe arrays of unit quality/site/waveform and spike time, trial, and within-trial time. `traj` contains two camera views, one structure per trial, with `frameTimes`, feature names, and `ts` (frames × x/y/confidence × feature). `sglx` provides sampling/clock synchronization metadata.
- Each `motionEnergy_*.mat` is MATLAB v5 and contains `me.data`, an object array with one 400-Hz vector per trial, and a session movement threshold (`moveThresh`).
- Example native dimensions (EKH1 2021-08-07): 305 trials; selected ALM probe 2 has 48 raw clusters; both camera views have 305 trials; side view has 7 features and bottom view 10; trial 1 has 1,792 frames; ephys sampling is 25 kHz.
- Authors' loader-selected ALM probes were used for counts. Across these probes there are 7,241 raw clusters; 5,676 carry explicit rejected labels (`garbage`, typo `gabrga`, `noisy`, `real?`). Remaining labels include multi (499), poor (415), fair (365), good (141), great (101), excellent (40), and four malformed null labels. Final retained counts require the paper's quality and firing-rate curation (Steps 3-6).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 7,241 raw clusters on reference-selected ALM probe(s); 11,158 across every probe |
| Neurons / session | 27-740 raw selected-probe clusters (mean 289.64) |
| Subjects | 10 (`EKH1`, `EKH3`, `JEB6`, `JEB7`, `JGR2`, `JGR3`, `JEB13`, `JEB14`, `JEB15`, `JEB19`) |
| Sessions / subject | 1-5; 25 total |
| Trials (total) | 8,260 native trials |
| Trials / session | 230-517 (mean 330.4) |

Native trial flags over the 25 sessions: 5,711 hit, 1,168 miss, 1,381 no-response, 661 early, 1,449 autowater/WC, and 187 stimulation-enabled. These flags overlap (for example early trials retain an outcome flag), so curated totals are determined later.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 1,651 DR-cohort units; 522 two-context units (214 well-isolated); 845 randomized-delay units | Electrophysiology analysis states these totals after curation and >1-Hz inclusion. | 
| Neurons / session | At least 10 required | Recording sessions were included only with ≥10 units. |
| Subjects | 9 in full fixed-delay DR cohort; 6 reported for two-context cohort; 4 randomized-delay | Electrophysiology analysis. |
| Sessions / subject | 25 sessions/9 mice full DR; 12 sessions/6 mice two-context; 19 sessions/4 mice randomized delay | Electrophysiology analysis. |
| Trials (total) | Not reported | Native files provide the exact count. |
| Trials / session | Behavioral analyses required ≥40 correct DR trials/direction and ≥20 correct WC trials/direction | Behavioral analysis. |
| Neural data time bin | 10 ms in supplied choice/context and main figure pipelines; 5 ms for a specific single-trial subspace analysis | Reference code and Methods. |
| Behavior data time bin | 2.5 ms native video frames (400 Hz), interpolated to the neural time base | Videography analysis and `loadMotionEnergy`. |
| Reward rate | Not reported as a single cohort statistic | Animals were trained to ≥70% DR accuracy. | 
| Choice decoder | ROC-AUC 0.86 ± 0.11 across sessions | Main text / Extended Data Fig. 2c. | 
| Choice selectivity | 36% sample, 42% delay, 58% response (483 single units) | Main text. |
| Context selectivity | 39% during ITI (214 single units) | Main text. |


### Processing Details
- The task alternates DR and WC blocks. DR has a 1.3-s auditory sample, usually a 0.9-s delay, then go cue; WC omits tones and presents water at the response/alignment time. `autowater` is used by the supplied code as the WC indicator.
- High-speed side and bottom video was acquired at 400 Hz. DLC tracks tongue/jaw/nose in both views and paws in the bottom view. Missing position is nearest-filled except for tongue; velocity is the first derivative of position. Motion energy is a 99th-percentile pixel-change metric.
- Supplied task-analysis pipelines align to go cue, use `[-2.5, 2.5)` s, 10-ms bins, a 15-sample causal Gaussian smoother, reflected boundaries, and no movement advance. Figure 8 context analysis extends the start to -3 s, but the common choice/kinematics pipeline and tutorial use -2.5 s.
- Choice/context paper decoders used session-wise firing rates or kinematic regressors, ridge logistic regression, four-fold CV, and 30% held-out trials. They balance relevant correct-trial classes. These reported algorithms differ from the provided downstream neural decoder, but establish expected decodability.

### Curation Steps

**Neuron curation rules**:
Manual spike sorting identifies acceptable units/single units vs explicit junk. The standard loader with `quality={'all'}` excludes garbage/gabrga/noisy/real? and then retains all acceptable units whose mean firing rate is strictly >1 Hz. Well-isolated-only filtering is reserved for single-unit selectivity/subspace analyses; all >1-Hz acceptable units are used elsewhere.

**Trial curation rules**:
Early-lick trials are omitted from analyses. Reference task conditions typically also exclude stimulation-enabled trials. Behavioral analyses omit ignore trials, while the requested decoder explicitly requires an `ignore` outcome, so valid ignore trials must be retained as a task-required exception. Correct and error trials from both contexts are otherwise used in single-trial population analyses.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from delay-epoch CDchoice | ROC-AUC 0.86 ± 0.11 |
| Choice/context time-resolved logistic models | Curves reported in Figs. 3b/4b; no numeric aggregate stated in text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Cohort scope | Context scripts load JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19: 12 sessions | Those exact 12 files exist and contain substantial WC blocks | Two-context cohort: 12 sessions, reported as six mice, 522 units | Use the explicit 12-session code cohort. Files contain seven distinct subject IDs, which is preserved rather than relabeling animals. |
| Unit total | Reject explicit bad quality labels, then mean FR >1 Hz | 528 units pass quality and 518 pass an exact Python reproduction of the 10-ms, go-aligned rate criterion | 522 units | Four-unit difference is consistent with data/code-version drift; use actual files and published algorithm. Do not tune the threshold to force a paper total. |
| Full fixed-delay cohort | General choice scripts load all 25 sessions | 25 files, 8,260 trials, ten IDs | 25 sessions, nine mice, 1,651 units | Not used because 13 sessions do not contain genuine alternating WC blocks; small `autowater` counts there are assistance trials, not a WC context. |
| Time window/binning | Choice/kinematics code uses `[-2.5,2.5)`, 10 ms; context Figure 8 uses `[-3,2.5)`, 10 ms | Video/ephys cover the common window | Methods has analysis-dependent 5-ms or 10-ms processing | Use the common decoder/kinematics window `[-2.5,2.5)` and 10 ms; it matches both the tutorial and choice decoding and avoids context-specific extra ITI. |
| Trials | Paper task conditions exclude early and stimulation trials and behavioral analyses omit ignores | Raw files contain all flags | Requested output explicitly includes `ignore` | Exclude early and stimulation trials; retain valid ignore trials because predicting ignore is explicitly required. |
| Video missingness | Reference nearest-fills non-tongue features and preserves tongue missingness | Raw DLC coordinates have missing samples/trials | Requested tongue/paw class 2 encodes not visible | Compute reference-style velocity on filled coordinates, but retain the pre-fill visibility mask and assign class 2 where raw tracking is unavailable. |
| Motion threshold | Paper manually sets a bimodal movement threshold | Separate files provide `moveThresh` | Requested output explicitly mandates 50th percentile | Use the task-mandated per-session median over aligned valid motion-energy samples, not `moveThresh`; reserve class 2 for missing video. |

Final consistent understanding: convert the 12 sessions explicitly used by the paper's two-context neural analyses (3,626 native trials), select only loader-designated ALM probe(s), reject explicit junk qualities, retain units with reference-processed mean FR >1 Hz, exclude early/stimulation trials, align all streams to go cue/water delivery, and preserve valid hit/miss/ignore trials. Neural rates use 10-ms bins and the reference 15-sample causal Gaussian smoothing with reflected leading boundary. Video features are clock-corrected using `findVideoOffset`, linearly interpolated to the identical time axis, and categorized using task-specified medians.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Selected `obj.clu{probe}.trialtm`, `.trial`, `bp.ev.goCue` | `neural` | Align each spike by its trial go cue; histogram in 10-ms bins on `[-2.5,2.5)`; divide by 0.01 s; apply the reference 15-sample causal Gaussian smoother with reflected leading boundary; transpose to neuron × time; float32 | `alignSpikes`, `getSeq`, `mySmooth` | Only explicit-loader ALM probes; reject garbage/gabrga/noisy/real? and mean FR ≤1 Hz. |
| Bin centers `-2.495 ... 2.495` s | `input[0]` | Store as a 1 × 500 float32 time series on every trial | `getSeq` | Decoder input is only time from go cue. |
| `bp.R/L`, `bp.hit/miss/no` | `output[0]` lick direction | Actual response: hit uses instructed/reward side, miss uses opposite side, no-response maps to none; broadcast per-trial value across 500 bins | Task definitions / `getOutcome` | Codes: left=0, right=1, none=2. |
| `bp.autowater` | `output[1]` behavioral context | 1 → WC, 0 → DR; broadcast | Paper condition expressions | Codes: WC=0, DR=1. |
| `bp.hit/miss/no` | `output[2]` outcome | miss → incorrect, hit → correct, no → ignore; broadcast | `getOutcome` | Codes: incorrect=0, correct=1, ignore=2. |
| Side-camera DLC `tongue` x/y | `output[3]` tongue velocity | Video-clock correct; interpolate positions to bin centers; compute reference-style x/y gradients and Euclidean speed; session median over visible retained samples; preserve raw missing mask | `findVideoOffset`, `findPosition`, `findVelocity` | Codes low=0, high=1, not visible=2. |
| Bottom-camera DLC `top_paw` and `bottom_paw` x/y | `output[4]` paw velocity | Clock correct/interpolate; reference nearest-fill for velocity; baseline-gradient correction; average available marker speeds; session median over visible retained samples; restore visibility mask | same | Codes low=0, high=1, not visible=2. |
| `motionEnergy_*.mat: me.data` | `output[5]` motion energy | Clock correct and linearly interpolate exactly as `loadMotionEnergy`; session median over available retained samples | `loadMotionEnergy` | Codes low=0, high=1, no video=2. Task-required median replaces supplied manual `moveThresh`. |
| Animal token from filename/meta | `subjects`, `subject_idx` | Unique IDs in first-session order and integer indices | loader metadata | Seven IDs in selected files. |
| Selected probes | `brain_regions`, `brain_region_idx` | `brain_regions=['ALM']`; zero vector per session | loader files / paper | All converted neurons are ALM. |

### Key Decisions
1. **Cohort**: Use the 12 sessions explicitly loaded by Figure 8/two-context neural scripts. This is the only neural cohort in which `autowater` represents alternating WC blocks rather than occasional assistance.
2. **Trial curation**: Retain trials satisfying `~early & ~stim.enable` and having exactly one outcome flag and one instructed/reward-side flag. Keep ignores as explicitly required. Do not drop missing-video trials; encode their behavior outputs as class 2. This preserves neural/outcome data while truthfully marking behavior availability.
3. **Uniform output representation**: Store all six outputs as a 6 × 500 int64 array. The first three rows are constant within trial (therefore still per-trial); the last three are time varying. The provided decoder requires a single common dimensionality, so mixed 1-D/2-D rows are impossible.
4. **Time base**: `[-2.5,2.5)` at 10 ms matches the main choice/kinematics code and gives exactly 500 samples in all sessions/trials. Bin centers, rather than edges, match `obj.time` in `getSeq`.
5. **Velocity visibility**: Use missingness before reference nearest-fill to assign class 2. Filling remains necessary to reproduce derivative processing, but must not erase the requested not-visible state.
6. **Paw scalar**: Average speed magnitude of the two bottom-view paw landmarks using whichever are visible; class 2 only when neither is visible. This avoids arbitrarily favoring a landmark while remaining invariant to marker naming.
7. **Thresholding**: Compute each median once per session from all finite visible samples among retained trials. Values equal to the median are class 1, exactly matching `<50th` versus `>=50th`.
8. **Data types**: Neural/input float32 and categorical output int64 minimize size while matching decoder expectations. Metadata records source trial indices, session IDs, retained unit indices/qualities, thresholds, counts, and processing constants.

### Planned Sanity Checks
- [ ] Directly recompute a selected raw spike/bin from original HDF5 and verify converted neural with `np.allclose` after smoothing.
- [ ] Directly compare every converted time input to independently constructed bin centers with `np.allclose`.
- [ ] Directly reconstruct labels for at least three source trials and compare outputs with `np.allclose`.
- [ ] Directly interpolate one raw motion-energy trace using source frame times/clock offset and compare categories with `np.allclose`.
- [ ] Confirm 12 sessions, expected subject/session IDs, no early/stim trials, ≥2 trials/session, identical 500-bin shapes, finite neural/input arrays, and categorical ranges.
- [ ] Compare retained unit count to 522 paper units (accept documented file-version delta only) and per-session ≥10 rule.
- [ ] Verify class-0/class-1 counts around each behavior median differ only through ties/missingness and plot raw traces, aligned traces, thresholds, visibility, and neural rates.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required `--full` (default), `--sample`, and `--show-processing` interfaces. It directly reads only needed fields from v7.3/HDF5 session objects, reads v5 motion-energy files with SciPy, reproduces reference spike alignment/binning/smoothing and video clock correction, validates shapes/ranges, records source indices and thresholds, and writes pickle protocol 5. Syntax compilation and CLI help completed without errors.

Code inefficiencies identified:
Loading full 100-300 MB MATLAB objects via `mat73` would materialize waveforms and all DLC features, and smoothing one spike train/trial at a time would create excessive Python overhead.

Code speedups added:
Use HDF5 field/reference access, vectorized spike binning (`np.add.at`), 2-D convolution across all trials, single-pass per-view video reads, float32 neural storage, and sequential session release. Plots are limited to two sessions as required.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 95 session-units |
| Neurons / session | 29, 66 |
| Subjects | 2 |
| Sessions / subject | 1 each |
| Trials (total) | 547 retained (302, 245) |
| Trials / session | 245-302 |
| Time input range | [-2.495, 2.495] bin centers (validator displays [-2.5,2.5]) |
| Lick direction distribution | [0.410, 0.448, 0.143] |
| Context distribution | [0.329, 0.671] |
| Outcome distribution | [0.119, 0.739, 0.143] |
| Tongue velocity distribution | [0.063, 0.063, 0.875] |
| Paw velocity distribution | [0.495, 0.495, 0.010] |
| Motion energy distribution | [0.495, 0.495, 0.010] |

### Processing Plots Review
Both `processing_JEB6_2021-04-18.png` and `processing_JEB7_2021-04-29.png` were visually inspected. Neural rasters are finite and smoothly causal; vertical zero marks the go cue/water event. Tongue visibility begins primarily after the cue as expected, paw and motion traces share the corrected time base, motion rises around/post cue, missingness is explicit, and finite low/high categories split at the plotted session medians. No temporal shift or threshold anomaly was found.

Initial iteration: conversion reached plotting but the plot summary indexed already-retained categorical arrays by source trial numbers, causing an `IndexError`. The plot-only indexing was fixed; the required command was rerun from scratch and completed. The final sample log contains no error.

Format verification reports: valid, no errors or warnings. Manual inspection confirmed all trials are `(n_neurons,500)`, inputs `(1,500)`, outputs `(6,500)`, finite neural/input values, valid categorical ranges, and source metadata.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Direct HDF5 field access, vectorized spike binning/smoothing, float32 storage | Sample completed in 9.47 s rather than loading/processing unused multi-GB fields |

| Step | Time / Session | Estimated Total Time |
| Conversion including plot (sample) | 4.08 s mean processing; 4.74 s/session including save/plots | ~57 s for 12 sessions (plots disabled) plus final pickle write; safely <15 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction | 0.6413 | 0.6348 |
| Behavioral context | 0.7919 | 0.7692 |
| Outcome | 0.6418 | 0.6297 |
| Tongue velocity | 0.5305 | 0.4988 |
| Paw velocity | 0.5094 | 0.5135 |
| Motion energy | 0.6254 | 0.6393 |

Training completed on CUDA for all 200 epochs after the final exact-edge fix. Mean loss fell monotonically from 7.428421 (epoch 1) to 0.792822 (epoch 200); test loss was 0.811186. Every validation balanced accuracy exceeded uniform chance (1/3, or 1/2 for context). The train-validation gaps are small, providing no evidence of leakage/overfitting.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 344 MiB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 522 two-context units | quality rejection + >1 Hz | 528 quality-pass; 518 >1 Hz in supplied files | 518 | Near: documented 4-unit data-version delta |
| Mean neurons/session | Not reported | ≥10/session | 43.17 retained | 43.17 (27-67) | Yes |
| Subjects | 6 reported | seven IDs loaded by scripts | 7 unique IDs | 7 | Matches data/code; paper discrepancy documented |
| Sessions | 12 | 12 explicit sessions | 12 | 12 | Yes |
| Trials (total) | Not reported | exclude early/stim; task requires ignores | 3,626 native; 3,116 valid retained | 3,116 | Yes |
| Trials/session (mean) | Eligibility criteria only | ≥2 target requirement | 259.67 retained | 259.67 (210-390) | Yes |
| Time input range | go aligned | `[-2.5,2.5)`, dt=.01 | centers [-2.495,2.495] | [-2.495,2.495] | Yes |
| Lick distribution | Not reported | derived hit/miss/no and R/L | [0.398,0.377,0.225] | [0.398,0.377,0.225] | Yes |
| Context distribution | Not reported | `autowater` | [0.315,0.685] | [0.315,0.685] | Yes |
| Outcome distribution | Not reported | hit/miss/no | [0.106,0.669,0.225] | [0.106,0.669,0.225] | Yes |
| Tongue distribution | Not reported | task median + visibility | computed per session | [0.029,0.052,0.919] | Yes; median ties explain imbalance |
| Paw distribution | Not reported | task median + visibility | computed per session | [0.492,0.492,0.016] | Yes |
| Motion distribution | Not reported | task median + no-video | computed per session | [0.495,0.497,0.007] | Yes |

Full conversion completed in 40.78 s, faster than the ~57-s estimate. Validation reports a valid structure with no errors or warnings. Spot checks of the first, middle, and last sessions confirmed source trial indices, constant per-trial rows, finite neural/input data, correct neuron counts, and all category bounds.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Re-read the regenerated `verification_full_out.txt`; data are valid with no errors or warnings. The initial full conversion emitted one intermittent SciPy direct-convolution cast warning despite finite output. Replaced that implementation with an explicit sliding-window convolution mathematically identical to `np.convolve`; regenerated output contains no warning.
2. **Independent raw-source `np.allclose` checks** (`sanity_checks.py`, which does not import conversion code):
   - Input: independently recreated bin centers and matched all 3,116 converted trials (`atol=2e-7`, the float32 quantization bound).
   - Output labels/trial curation: independently loaded Bpod flags, recreated retained indices for every session, and matched lick/context/outcome rows for first/middle/last retained trials in every session (36 raw trials total).
   - Neural: independently loaded one raw unit's spike times/trial IDs, go-aligned, histc-binned, and causally smoothed it; the complete 500-bin trace matched converted session 0/trial 0/unit 0 (`rtol=2e-6`, `atol=2e-5`).
   - Motion output: independently loaded the raw motion file and frame timestamps, recomputed clock correction/alignment and the session median, and matched all 500 categorical samples for session 0/trial 0; threshold matched to `1e-12`.
3. **Reference code comparison**:
   - (a) Loading: `h5py`/SciPy access the same `obj` and `me` fields as `loadObjs`/`loadMotionEnergy`; authors' session loaders determine selected ALM probes.
   - (b) Filtering: `findClusters` bad labels and `removeLowFRClusters` strict >1-Hz logic are reproduced. Early/stimulation trials follow paper conditions; ignores are retained only because the target explicitly requests them.
   - (c) Alignment: spike `trialtm-goCue` matches `alignSpikes`; video `frameTimes-vidshift-goCue` matches `findPosition`/`loadMotionEnergy` and `findVideoOffset`.
   - (d) Binning: explicit edges, 10-ms `[−2.5,2.5)` bins, rates/`dt`, 15-sample causal Gaussian, and leading reflected block match `getSeq`/`mySmooth`.
   - (e) Input: bin centers match `obj.time`; the target specifically limits input to time.
   - (f) Output: Bpod fields follow paper condition semantics; reference-style DLC gradients and clock interpolation are used, with only target-required percentile discretization and missing classes differing.
4. **Key statistics**: Confirmed 12 code-selected sessions, 3,626 native/3,116 retained trials, 7 file IDs, 518 >1-Hz units (27-67/session), all output ranges/distributions, and 500 bins. Paper comparisons are fully tabulated in Step 9. The 518-versus-522 unit and seven-ID-versus-six-mouse differences remain attributable to the supplied data snapshot and are not alterable without violating the provided raw data/code.
5. **Edge cases/off-by-one audit**: Explicit histc edges exclude exactly +2.5 s and include −2.5 s; bin centers are −2.495 through +2.495 (500 bins). First and final source trials are retained where valid, all stored indices are in bounds/1-based in metadata, all trial arrays agree, missing camera periods map to class 2, session medians use finite values only, median ties correctly map to high, and minimum retained unit rate is 1.00577 Hz (strictly >1).

### Issues Found and Resolved
- **Plot retained-index bug (Step 7)**: Removed a second source-index operation on already-retained label arrays and reran sample conversion/validation.
- **Exact-edge neural binning**: Independent check found floor arithmetic could shift spikes exactly on decimal edges relative to MATLAB `histc`. Replaced floor with explicit-edge `searchsorted(..., side='right')`, reran full conversion, full validation, and every check above. All independent checks pass.
- **Convolution warning**: Replaced SciPy N-D direct convolution with an exact warning-free sliding-window/einsum implementation; verified against `np.convolve` to machine precision (`max difference 2.22e-16`). Final conversion is warning-free and faster (27.91 s).
- **Median-tie tongue sessions**: Six sessions have a tongue-speed median of zero, so there are no values strictly below the 50th percentile and class 0 is absent in those sessions. This is mathematically required by `< median` versus `>= median`, not a conversion error; visible zero-speed samples correctly enter class 1.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. CUDA training completed all 200 epochs; loss declined monotonically from 7.302714 to 0.772226. Test loss: 0.816256.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction | 0.6239 | 0.6040 | chance 0.3333 |
| Behavioral context | 0.7596 | 0.7480 | chance 0.5000 |
| Outcome | 0.6202 | 0.5769 | chance 0.3333 |
| Tongue velocity | 0.6513 | 0.6339 | chance 0.3333 |
| Paw velocity | 0.5371 | 0.5119 | chance 0.3333 |
| Motion energy | 0.6384 | 0.6028 | chance 0.3333 |

The script finished successfully and created sample/prediction plots. All six held-out balanced accuracies exceed uniform chance; four of five 3-class outputs exceed 1.5× chance, while paw velocity is just above that threshold (0.5119 versus 0.5000).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
|---|---:|---|
| Lick direction | 0.6040 balanced accuracy (1.812× chance) | Paper reports choice ROC-AUC 0.86 ± 0.11 on balanced correct L/R DR trials during delay only. Our harder 3-class target includes incorrect/ignore trials and all 500 time bins; raw labels/alignment were independently verified. |
| Behavioral context | 0.7480 (1.496× chance) | Paper Fig. 4b reports above-shuffle time-resolved context decoding but no numeric aggregate. Result is strongly above chance and matches the qualitative expectation. |
| Outcome | 0.5769 (1.731× chance) | Not decoded in paper. |
| Tongue velocity | 0.6339 (1.902× chance) | Not decoded from neural activity as this categorical target in paper. |
| Paw velocity | 0.5119 (1.536× chance) | Not decoded from neural activity as this categorical target in paper. |
| Motion energy | 0.6028 (1.808× chance) | Not decoded as this 3-class target in paper. |

No output is below chance. Context is only 0.002 below the diagnostic 1.5×-chance line. It was investigated rather than dismissed: all 12 sessions contain both contexts; full distribution is WC 31.5% / DR 68.5%; source `autowater` labels match converted values on 36 direct trial checks; an explicit JEB6 audit confirmed DR trial 1, first WC retained trial 141, final WC ignore trial 382, and seven block transitions; alignment plots show correct go/water zero; and unit filtering matches the paper pipeline. Changing to balanced correct-only trials could mimic the paper decoder and raise accuracy, but would discard the requested incorrect/ignore output classes and is therefore not an appropriate conversion change.

The paper's sole numeric classification result (choice AUC 0.86 ± 0.11) is not metric/task-matched to the provided decoder. Nevertheless, lick direction is robustly decoded above chance, and independent raw checks found no label or alignment error. Prediction plots were visually inspected: neural activity is continuous/aligned, static targets remain constant, time-varying targets change plausibly, and held-out predictions track all target types without obvious leakage.

Train/validation ratios are 1.033 (lick), 1.016 (context), 1.075 (outcome), 1.027 (tongue), 1.049 (paw), and 1.059 (motion), all far below the 1.5× overfitting criterion.

### Issues Found and Resolved
- **Context narrowly below 1.5× chance**: Completed raw-label, variation, alignment, filtering, and paper-method audits; no conversion issue found. Retaining all requested outcome classes is the justified representation.
- **Post-fix sample reproducibility**: Regenerated `sample_data.pkl`, both sample validation logs, processing plots, and sample training after the exact-edge correction. Final sample validation remains warning-free and every accuracy remains above chance.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with cohort, loading, format, outputs, reproduction commands, and final statistics
- [x] cache/ folder created; independent `sanity_checks.py` and generated bytecode moved there
- [x] `cache/README_CACHE.md` documents cached/audit files
- [x] All required files are present and non-empty
- [x] Final converter/audit syntax checks pass
- [x] Final independent audit passes all neural/input/output `np.allclose` checks
- [x] Full decoder log ends with successful completion after 200 epochs

Final deliverable summary: 12 sessions, 3,116 trials, 518 ALM session-units, 500 10-ms bins/trial, 344-MiB full pickle. The converter reproduces the applicable reference loading, curation, alignment, binning, and smoothing, with explicitly documented target-required differences for ignore trials and median behavior discretization.
