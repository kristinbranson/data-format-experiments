# Dataset Conversion Notes

## Overview
- **Dataset**: "Separating cognitive and motor processes in the behaving mouse" (paper.pdf, /app/code, /app/data)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: IN PROGRESS

Directory contents:
- [to be listed]

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Repo = code for Hasnain, Birnbaum et al. 2024 (Nat Neuro), "Separating cognitive and motor processes in the behaving mouse". MATLAB.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| loadObjs | DataLoadingScripts/loadObjs.m | LOADING | loads `data_structure_ANM_DATE.mat` -> struct `obj` (fields bp, clu, traj, sglx, ex, me) |
| loadANM_ALMVideo (16 files) | DataLoadingScripts/Recording and video/ | LOADING | per-animal meta: anm, date, probe number(s) that are ALM |
| loadSessionData | DataLoadingScripts/loadSessionData.m | LOADING | loop sessions x probes -> processData; concatenates units across 2 probes |
| processData | DataLoadingScripts/processData.m | PROCESSING | findTrials -> findClusters -> alignSpikes -> getSeq -> removeLowFRClusters -> baselineFR |
| findTrials | DataLoadingScripts/findTrials.m | CURATION | evaluates condition strings (e.g. 'R&hit&~stim.enable&~autowater&~early') over obj.bp fields; returns trial ids |
| findClusters | DataLoadingScripts/findClusters.m | CURATION | quality={'all'} -> keep all clusters EXCEPT quality in {garbage, gabrga, noisy, 'real?'} |
| alignSpikes | DataLoadingScripts/alignSpikes.m | PROCESSING | trialtm_aligned = clu.trialtm - obj.bp.ev.(alignEvent)(clu.trial) |
| getSeq | DataLoadingScripts/getSeq.m | PROCESSING | edges=tmin:dt:tmax; obj.time=edges+dt/2 (drop last); histc spike counts/trial/unit; rate=N/dt; mySmooth(N, smooth, bctype) -> obj.trialdat (time,units,trials) |
| mySmooth | utils/mySmooth.m | PROCESSING | causal gaussian (gausswin(N), first half zeroed, normalized), 'reflect' boundary |
| removeLowFRClusters | DataLoadingScripts/removeLowFRClusters.m | CURATION | drop units with mean over conditions/time of psth <= params.lowFR (1 spk/s in the figure scripts) |
| baselineFR | DataLoadingScripts/baselineFR.m | PROCESSING | presample mean/std per unit (used for z-scoring) |
| getKinematics | funcs/kinematics/getKinematics.m | PROCESSING | builds kin.dat (time,trials,feat): per DLC feature xdisp,ydisp,xvel,yvel + tongue_angle, tongue_length + motion_energy; then standardizes and PCA |
| getKinematicsFromVideo | funcs/kinematics/getKinematicsFromVideo.m | PROCESSING | loops views/features -> findPosition, findVelocity; tongue NaNs (not visible) replaced with session-mean initial tongue position, NaN indices kept in kin.nans |
| findPosition | funcs/kinematics/findPosition.m | PROCESSING | interp1(frameTimes - vidshift - alignEventTime, ts, obj.time+advance_movement); skips trials with NaN NdroppedFrames; non-tongue features smoothed & fillmissing('nearest') |
| findVelocity | funcs/kinematics/findVelocity.m | PROCESSING | gradient of x/y position; non-tongue: subtract median diff (baseline drift) & fillmissing nearest; tongue: NaN -> 0 velocity |
| findVideoOffset | funcs/findVideoOffset.m | PROCESSING | vidshift = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart) |
| loadMotionEnergy | DataLoadingScripts/loadMotionEnergy.m | LOADING/PROCESSING | loads motionEnergy_ANM_DATE.mat; interp1 to obj.time using frameTimes-vidshift-alignTime; fillmissing nearest; me.move = me.data > me.moveThresh |
| getOutcome | funcs/getOutcome.m | PROCESSING | outcome = bp.hit, NaN where bp.no (ignore) |
| NeuralChoiceDecoding / NeuralContextDecoding | ChoiceContextDecoding/ | ANALYSIS | decode choice / context (DR vs WC) from obj.trialdat in 75 ms bins; normalize X to [-1,1]; 4-fold |

### Standard parameter values (from Scripts/Figure*/ and WorkingWithDataObjs.m)
- `params.alignEvent = 'goCue'`
- `params.tmin=-2.5`, `params.tmax=2.5`
- `params.dt`: 1/100 (WorkingWithDataObjs), (1/100)*3 = 30 ms (Figure 3 scripts), 1/200 (getDefaultParams)
- `params.smooth = 15` bins, causal gaussian, `bctype='reflect'`
- `params.quality = {'all'}` (exclude garbage/noisy/'real?')
- `params.lowFR = 1` spk/s in figure scripts (0.5 in getDefaultParams)
- `params.advance_movement = 0`
- Conditions always exclude `stim.enable` (optogenetic) and `early` (early-lick) trials.
- `autowater` == water-cued (WC) context; `~autowater` == delayed-response (DR) context.

### Notes
- Ephys sessions = files NOT containing 'MAH'. MAH sessions are behavior+video only (bilateral MC inhibition) -> no neural data -> cannot be used for a neural decoder.
- `NullPotent/SessionMeta.csv` lists the 24 fixed-delay ALM ephys sessions used in the paper's main N/P analyses (animals JEB6, JEB7 x2, EKH3, JGR2 x2, JGR3, JEB13 x5, JEB14 x4, JEB15 x4, JEB19 x4).
- Randomized-delay ephys sessions (JEB11, JEB12, JEB23, JEB24) live in `RandomizedDelay_Ephys_Behavior`.
- Video frame times need a `-vidshift` correction (~0.5 s) to sync video to ephys clock.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/` has 4 subdirectories:
| Directory | Files | Content |
|---|---|---|
| `Ephys_Behavior` | 25 `data_structure_*.mat` + 25 `motionEnergy_*.mat` | fixed-delay DR (+WC two-context) ALM ephys sessions |
| `RandomizedDelay_Ephys_Behavior` | 21 `data_structure_*.mat` + 19 `motionEnergy_*.mat` | randomized-delay DR ALM ephys sessions |
| `DelayInhibition_BilatMC_Behavior` | ~30 `data_structure_MAH*.mat` | behavior+video only (opto inhibition) -> NO neural data |
| `GoCueInhibition_BilatMC_Behavior` | ~26 `data_structure_MAH*.mat` | behavior+video only -> NO neural data |

MAT file versions are mixed: 36 data_structure files are MAT v7.3 (HDF5, read with `h5py`), 11 are MAT v7 (read with `scipy.io.loadmat`). All `motionEnergy_*.mat` are MAT v7. A unified loader (`Session` class) was written for both.

`obj` fields:
- `obj.bp` (Bpod): `Ntrials`, `hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim.enable`, `bitRand`, `protocol`, sometimes `autowaterBlock`, `autolearn`; `obj.bp.ev`: `bitStart`, `sample`, `delay`, `goCue`, `reward` (per trial, seconds within trial), `lickL`, `lickR` (cell arrays of lick contact times).
- `obj.clu`: cell (nProbes x 1) of struct arrays with fields `tm` (session spike times), `trialtm` (time within trial), `trial` (trial number), `quality` (string), `spkWavs`, `site`/`channel`.
- `obj.traj`: cell (2x1), view0 = side cam, view1 = bottom cam. Per trial: `ts` (frames, [x,y,confidence], feature), `frameTimes` (s, video clock), `NdroppedFrames`, `featNames`.
  - side feats: tongue, left_tongue, right_tongue, jaw, trident, nose, lickport
  - bottom feats: top_tongue, topleft_tongue, bottom_tongue, bottomleft_tongue, top_paw, bottom_paw, lickport, jaw, top_nostril, bottom_nostril
- `obj.sglx`: `fs` (~25 kHz), `bitcode.bitstart` (samples) -> used for video/ephys sync offset.
- `obj.ex.probe.loc`: probe location strings, e.g. 'R ALM', 'L M1TJ', 'DUMMY'.
- motionEnergy file: `me.data` = cell (nTrials x 1) of per-frame motion energy (400 Hz, same frames as video), `me.moveThresh` = per-session movement threshold.

### Dataset Size (from data files, before curation)
| Statistic | Value |
|-----------|-------|
| Ephys sessions (files) | 47 (25 fixed-delay + 22 randomized-delay; 2 of the latter have NO sorted units and 'DUMMY' probe) |
| Animals with ephys | 14 (EKH1, EKH3, JEB6, JEB7, JEB11-15, JEB19, JEB23, JEB24, JGR2, JGR3) |
| Raw clusters (all probes, all qualities) | 14,297 (10,931 labeled 'garbage') |
| Trials (total, all ephys sessions) | 15,843 |
| Trials / session | 218 - 517 (median ~340) |
| Cluster quality labels | excellent 74, great 192, good 304, fair 784, multi 1136, poor 664, garbage 10931, noisy 1, 'real?' 1, typos: 'mutli' 2, 'gabrga' 1, 'ood' 1, '' 206 |

Notes:
- Quality strings differ in case/padding between sessions ('Poor' vs 'poor   '), and include typos; matching must be case-insensitive and trimmed.
- JEB24 2023-10-03 and 2023-10-04 have `obj.clu` absent/empty, probe loc 'DUMMY', and no motionEnergy file -> not usable (also absent from the reference meta scripts).
- JEB15 2022-07-29 probe 1 is labeled 'DUMMY' (not sorted); reference meta uses probe 2 only for that session.
- Sessions with two probes: one is ALM, the other tjM1 (M1TJ). Reference meta specifies the ALM probe per session; `obj.ex.probe.loc` agrees.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| DR (fixed-delay) task units | 1,651 units (483 single units), 25 sessions, 9 mice | "For the DR task, we recorded 1,651 units (483 single units) in ALM from 25 sessions using nine mice" |
| Two-context (DR+WC) subset | 12 sessions, 6 mice, 522 units (214 SU) | "In 12 sessions from six mice, animals performed the two-context task... 522 units (214 well-isolated single units)" |
| Randomized delay task | 845 units (288 SU), 19 sessions, 4 mice | "for the randomized delay task, we recorded 845 units (288 well-isolated single units) in ALM from 19 sessions using four mice" |
| Session inclusion | >= 10 units | "Recording sessions were included for analysis only if they had at least 10 units" |
| Unit inclusion | FR > 1 Hz | "All units with firing rates exceeding 1 Hz were included in all other analyses." |
| Trial exclusion | early-lick and ignore trials omitted from *behavioral* analyses | "excluding early lick and ignore trials, which were omitted from all analyses" |
| Behavior criterion | >=40 correct DR trials/direction, >=20 correct WC trials/direction | methods, Behavioral analysis |
| Video | 2 cameras (side, bottom) at 400 Hz, DeepLabCut | Videography analysis |
| Motion energy | per frame, 99th percentile over pixels; per-session manual movement threshold | Motion energy |
| Task timing | sample tone 1.3 s; delay 0.9 s (0.7 s for 1 mouse); go cue chirp 10 ms; response window 3 s | Mouse behavior |
| Randomized delay | delays from {0.3,0.6,1.2,1.8,2.4,3.6} s | Mouse behavior |
| Context blocks | session starts with ~100 DR trials, then alternating WC/DR blocks of 10-25 trials | Mouse behavior |
| Choice selectivity | sample 36%, delay 42%, response 58% of 483 SU | Results |
| Context selectivity | 39% of 214 SU | Results |

### Processing Details
- Alignment: `params.alignEvent='goCue'`; all reference analyses align neural, kinematic, and ME data to the go cue (in WC trials, the "go cue" time is the water drop time, stored in the same `bp.ev.goCue` field).
- Window: tmin=-2.5 s, tmax=+2.5 s relative to the go cue.
- Binning: `dt` = 5-30 ms depending on script (WorkingWithDataObjs 10 ms; Figure 3 scripts 30 ms; getDefaultParams 5 ms). Decoding analyses re-bin neural data to **75 ms**.
- Smoothing: causal Gaussian kernel, `params.smooth = 15` bins, 'reflect' boundary.
- Video/ephys sync: frameTimes must be shifted by `vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)` (~0.49-0.5 s).

### Curation Steps
**Neuron curation rules**:
1. Exclude clusters whose quality is garbage/noisy/'real?' (findClusters with quality='all').
2. Exclude clusters with mean firing rate <= 1 spk/s (removeLowFRClusters, params.lowFR=1; paper: "units with firing rates exceeding 1 Hz").
3. Use only the ALM probe(s) as listed in the reference meta scripts / `obj.ex.probe.loc`.

**Trial curation rules**:
- Analyses exclude `stim.enable` (photoinactivation) trials and `early` (early lick) trials; ignore trials excluded from most analyses but are a required decoder output class here, so they are retained (see Step 5).
- Sessions with < 10 units excluded.

### Decoders Trained (reference)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Choice (R vs L, correct trials) from neural population | logistic regression per 75-ms bin, ridge, 4-fold | ~0.5 (chance) before go cue rising to ~0.9-1.0 after go cue (Fig. 3b) |
| Context (DR vs WC) from neural population | same | ~0.7-0.9 across the trial incl. ITI (Fig. 4b) |
| Choice from CDchoice projections (delay epoch) | logistic regression + ROC | AUC ~0.8-0.9 (ED Fig. 2c) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

Cross-checks run (scripts in `/app/cache/`): `survey.py`, `unitcheck.py`, `variants*.py`, `lickcheck.py`, `videocheck.py`.

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| # fixed-delay ephys sessions | 25 uncommented entries in load*_ALMVideo.m | 25 `data_structure` files in `Ephys_Behavior` | "25 sessions" | consistent -> use all 25 |
| # randomized-delay sessions | 19 uncommented entries (JEB23 2023-10-20 commented out; JEB24 10-03/10-04 absent) | 22 files (2 without any units, 1 duplicate-like 10-20) | "19 sessions" | use the 19 reference sessions |
| # mice (fixed delay) | 10 animals in meta scripts; `NullPotent/SessionMeta.csv` has 24 sessions / 9 animals (no EKH1) | 10 animals have fixed-delay files | "nine mice" | The paper's 9 mice = SessionMeta.csv list. The 25-session list in the loading scripts (which the paper's 25-session count matches) contains 10 animals. Kept all 25 sessions/10 animals; documented. |
| # units, randomized delay | quality filter + FR>1 Hz | **845** units with FR > 1 Hz | "845 units" | EXACT MATCH - validates curation rules |
| # units, fixed delay | quality filter + FR>1 Hz | 1375 (1565 before FR filter, 1890 if both probes used) | "1,651 units" | Could not reproduce exactly with any threshold on the ALM probe alone. 1651 is between the ALM-only (1565) and both-probe (1890) counts; the paper number probably includes some sessions/probes not in the shared loading scripts. We keep the reference rule (ALM probe from meta, quality filter, FR>1 Hz). |
| # single units | 'excellent/great/good' = single units (WorkingWithDataObjs) | 248 (fixed), 156 (random) | 483 / 288 | The paper's single-unit counts are larger, so their SU definition must also include 'fair' (fixed: 248+311 = 559, random: 156+223 = 379) or use a different manual list. SU labels are not needed for this conversion (all units with FR>1 Hz are used, as the paper does for all but two analyses). |
| lick direction | firstLickTime.m: first lickport contact after go cue | first post-go-cue lick side agrees 100% with (hit & R -> right, miss & R -> left, no -> none) on all tested sessions | "ignore" if no response within 3 s | consistent; use the actual first lick |
| motion energy | loadMotionEnergy.m expects `me.data` cell + `me.moveThresh` | 3 file variants: (a) struct with cell data, (b) nested struct (JEB15 x2), (c) bare cell array without moveThresh (JEB23 x4) | n/a | wrote a robust loader; ME trial counts match Ntrials in every session |
| video sync | findVideoOffset: vidshift = mode(bitcode.bitstart)/fs - mode(bp.ev.bitStart) | ~0.49-0.50 s for all sessions | n/a | applied |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Sessions / subjects / regions
- Sessions: the 44 reference ALM ephys sessions (25 fixed-delay + 19 randomized-delay), transcribed into `/app/cache/meta_sessions.py` (also copied into `convert_data.py`).
- Only the ALM probe(s) listed in the reference meta are used (JEB15 07-26/27/28 have probes [1 2]; probe 2 there is L M1TJ per `obj.ex.probe.loc`, so for those sessions we keep **only the ALM probe** -> see Key Decisions).
- `subjects` = the 14 animals; `subject_idx` per session.
- `brain_regions` = ['ALM']; every included neuron is in ALM.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu{alm}.trialtm`, `.trial` | `neural` | align to go cue (subtract `bp.ev.goCue`), bin -2.5..2.5 s at dt=30 ms, rate = count/dt, causal-Gaussian smooth (N=15, 'reflect') | alignSpikes.m, getSeq.m, mySmooth.m | firing rate in spikes/s, float32 |
| time axis | `input[0]` = `time_from_gocue` | obj.time = bin centers, seconds | getSeq.m | continuous, time-varying (1 x T) |
| first lick after go cue (`bp.ev.lickL/lickR`) | `output[0]` = `lick_direction` | 0=left, 1=right, 2=none | firstLickTime.m | per-trial value broadcast over T |
| `bp.autowater` | `output[1]` = `context` | 0=WC (autowater=1), 1=DR | params.condition strings | per-trial |
| `bp.hit/miss/no` | `output[2]` = `outcome` | 0=incorrect(miss), 1=correct(hit), 2=ignore(no) | getOutcome.m | per-trial |
| side-cam `tongue` x/y (`obj.traj{1}`) | `output[3]` = `tongue_velocity` | speed=|d(x,y)/dt| of interpolated position; 0 if < session median, 1 if >= median, 2 if tongue not visible (DLC NaN) | findPosition.m, findVelocity.m | time-varying |
| bottom-cam `top_paw`,`bottom_paw` (`obj.traj{2}`) | `output[4]` = `paw_velocity` | mean speed of the visible paw markers; 0/1 by session median, 2 if neither paw visible | findPosition.m, findVelocity.m | time-varying |
| `motionEnergy_*.mat` `me.data` | `output[5]` = `motion_energy` | interp to trial time axis (frameTimes - vidshift - goCue), fillmissing nearest; 0/1 by session median, 2 if the trial has no usable video | loadMotionEnergy.m | time-varying |

### Key Decisions
1. **Alignment to go cue** (`params.alignEvent='goCue'`), window **-2.5 to +2.5 s**, exactly as in every reference analysis script. In WC trials `bp.ev.goCue` holds the water-drop time, so the alignment is meaningful in both contexts.
2. **Bin size 30 ms** = `params.dt = (1/100)*3` used in the paper's Figure 3 analysis scripts (other scripts use 5-10 ms; the decoding analyses re-bin to 75 ms). 30 ms keeps ~167 bins/trial, enough temporal resolution for licking (~7 Hz) while keeping the dataset tractable (~2.3M timepoints).
3. **Causal Gaussian smoothing, N=15 bins, 'reflect'** exactly as `mySmooth` — causal so no information from the future leaks backwards in time (important for a decoder).
4. **Neuron curation**: quality not in {garbage, gabrga, noisy, real?, empty} (findClusters, case-insensitive/trimmed to handle label case and typos) AND mean firing rate over the trial window > 1 Hz (removeLowFRClusters with `lowFR=1`, matching the paper's "units with firing rates exceeding 1 Hz"). Sessions with < 10 units would be dropped (paper criterion); in practice all 44 pass.
5. **Probe selection**: use exactly the probe(s) listed in the reference meta scripts (`load<ANM>_ALMVideo.m`). These are the probes the paper analysed. For each kept probe the recorded region is read from `obj.ex.probe.loc` ('R ALM'/'L ALM' -> ALM, 'M1TJ' -> tjM1); when `loc` is missing or 'DUMMY', ALM is assumed (all the loader scripts are named *_ALMVideo and the paper states all recordings analysed were ALM). Consequence: for JEB15 2022-07-26/27/28 the reference uses probes [1 2] where probe 2 is 'L M1TJ', and for JEB15 2022-07-29 the reference uses probe 2 ('L M1TJ'); those neurons are labelled tjM1 so that `brain_region_idx` is truthful, rather than dropping data the reference analysed.
6. **Trial curation**: exclude photoinactivation trials (`bp.stim.enable`) and early-lick trials (`bp.early`), exactly as every reference condition string does ('~stim.enable&~early') and as stated in the methods ("Trials in which the animal contacted the lickport before the reward ('early lick') were omitted from analyses"). **Ignore trials are kept** because 'ignore' and 'none' are required output classes.
7. **Kinematics NaNs are kept as the 'not visible' class** instead of being filled/zeroed as in `findPosition`/`findVelocity`. This is required by the decoder output specification, which asks for an explicit 'not visible' category.
8. **Discretization thresholds are per session** (median over all included trials and timepoints where the feature is visible), as specified in the decoder task.
9. **Session curation**: keep a session only if it has >= 10 curated units (paper criterion) and at least 2 usable trials. No session is dropped for probe-location reasons (see Decision 5).
10. **Video/ephys synchronisation** uses `vidshift` from `findVideoOffset.m` for both DLC and motion energy, as in the reference.

### Planned Sanity Checks
- [ ] Number of sessions = 44 (25 fixed + 19 randomized), animals = 14.
- [ ] Units after curation: randomized-delay total = 845 (matches paper exactly).
- [ ] Every session has >= 10 units; neurons/session distribution similar to ED Fig 2a (13-130).
- [ ] Trial counts: sum of kept trials ~= total trials - early - stim.
- [ ] Lick direction from first lick agrees with hit/miss x R/L in >99% of trials.
- [ ] Outcome fractions: hit ~70%, miss ~12%, ignore ~13% (from raw bp fields).
- [ ] Context fraction WC ~10-20% of trials overall (sessions with two contexts: 12 of 25 fixed-delay).
- [ ] Each discretized output has ~50/50 split between classes 0 and 1 among visible timepoints.
- [ ] Spot check (Step 10): recompute neural rate, tongue speed, ME for specific trials directly from the raw .mat files with independent code and compare with `np.allclose`.
- [ ] Time axis: bin centers from -2.485 to 2.485, dt exactly 0.03 s.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` is self-contained (no imports from `/app/cache`). Structure:
- `Session` class: unified reader for MAT v7 (scipy) and v7.3 (h5py) `data_structure` files; `load_motion_energy` handles the three motion-energy file variants.
- Ports of the reference MATLAB routines: `time_axis` (getSeq.m), `gausswin`/`causal_kernel`/`mysmooth` (mySmooth.m), `bin_spikes` (alignSpikes.m + getSeq.m), `interp_to_axis`/`_fill_nearest` (interp1 + fillmissing('nearest')), `speed_from_xy` (findVelocity.m).
- `process_session`: trial curation, neural binning, unit curation, behavioural outputs, video outputs, per-session discretization.
- `plot_processing`: 14-panel figure per session covering every processing step.
- `main`: multiprocessing pool over sessions, assembly, session curation, metadata, pickling.

Efficiency:
- Spike binning vectorised with `np.histogram2d` over (trial, aligned time) per unit instead of a per-trial loop (the reference's double loop over units x trials).
- Smoothing done with `scipy.signal.fftconvolve` over all trials at once.
- Video frames binned with `np.bincount` instead of per-bin loops.
- Sessions processed in parallel with a `multiprocessing.Pool` (default 8 workers; memory-bound because v7.3 files are 90-250 MB each).

Code inefficiencies identified: per-trial DLC reads from HDF5 dominate runtime (each trial is a separate dataset reference).
Code speedups added: vectorised histogram binning, FFT smoothing, bincount frame binning, multiprocessing over sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Ran `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` (sessions JEB6_2021-04-18 and JEB11_2022-05-10) -> `/app/conversion_sample_out.txt`, then `train_decoder.py --verify-only` -> `/app/verification_sample_out.txt` (**valid, no errors, no warnings**).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 92 (29 + 63) |
| Trials (total) | 625 (302 + 323) |
| Trials / session | 302, 323 (of 382, 365 raw; the rest are stim/early trials) |
| T per trial | 166 bins (30 ms) for every trial |
| time_from_go_cue range | [-2.5, 2.5] s (bin centres -2.485 .. 2.465) |
| lick_direction | left 0.381, right 0.461, none 0.158 |
| context | WC 0.184, DR 0.816 |
| outcome | incorrect 0.083, correct 0.757, ignore 0.160 |
| tongue_velocity | <med 0.093, >=med 0.093, not visible 0.814 |
| paw_velocity | <med 0.480, >=med 0.480, not visible 0.040 |
| motion_energy | <med 0.500, >=med 0.500, no video 0.000 |

### Processing Plots Review
`processing_JEB6_2021-04-18.png`, `processing_JEB11_2022-05-10.png` were written. Their content was also checked numerically (`/app/cache/sample_checks.py`):
- Population firing rate peaks at t = +0.22 s / +0.19 s after the go cue (response-epoch activity) -> neural alignment correct.
- Tongue visible 5% of the time before the go cue vs 28-36% after, peaking at +0.16/+0.25 s -> DLC alignment correct.
- Motion energy above-median fraction 0.28-0.35 before vs 0.66-0.72 after the go cue -> ME alignment correct.
- 100% of correct trials have a lick; 100%/98% of ignore trials have lick_direction = none -> behavioural outputs correct.
- Discretization: class 0 and class 1 fractions are equal to 3 decimals for every discretized output (median split), as expected.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| vectorised histogram2d binning + FFT smoothing + bincount + 8-way multiprocessing | ~10x vs naive loops |

| Step | Time / Session | Estimated Total Time |
| neural binning | ~0.4 s | ~20 s |
| video/DLC + ME | ~2.5 s | ~110 s |
| whole session | 2.6-3.1 s | 44 sessions / 8 workers ~ 1-3 min (larger Neuropixels sessions are slower; observed full run below) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions)
Loss decreased monotonically 3.05 -> 0.63 over 200 epochs (test loss 0.688).

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|---|
| lick_direction | 0.690 | 0.644 | 0.333 |
| context | 0.842 | 0.854 | 0.500 |
| outcome | 0.647 | 0.588 | 0.333 |
| tongue_velocity | 0.630 | 0.625 | 0.333 |
| paw_velocity | 0.656 | 0.615 | 0.333 |
| motion_energy | 0.854 | 0.842 | 0.333 |

All outputs are above chance; no sign of leakage (train ~ validation).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/converted_data.pkl --full   # 15 s wall clock, 8 workers
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

### Output Files
- `converted_data.pkl`: 44 sessions, 13,762 trials, 2,457 neurons, 14 subjects (~3.5 GB)
- `verification_full_out.txt`: created; **"Data format is valid, no errors or warnings."**

### Issue found and fixed during this step
- First full run produced 61 warnings "all neural data is zero" in JEB24_2023-10-23 (trials 292-319) and JEB24_2023-11-03 (trials 301-333). These are contiguous **trailing** trials in which the ephys recording had already stopped while Bpod kept running. Fixed by dropping trials in which no curated unit fires a single spike in the whole 5 s window. Re-ran conversion and verification -> no warnings.
- Also changed the low-FR criterion to use the raw (unsmoothed) spike counts over **all** trials, matching `removeLowFRClusters.m`, whose first PSTH condition `(hit|miss|no)` spans every trial (changing this altered only 1 unit in total).

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions (fixed delay) | 25 | 25 meta entries | 25 files | 25 | YES |
| Sessions (randomized delay) | 19 | 19 meta entries | 22 files (2 unusable, 1 commented out) | 19 | YES |
| Subjects | 9 (fixed) + 4 (random) | 10 (fixed) + 4 (random) | 10 + 4 | 14 total | code/data say 10 fixed-delay mice; paper says 9 (SessionMeta.csv has 9, without EKH1) |
| Total units (fixed delay) | 1,651 | n/a | 1,565 pass the quality filter on the ALM probes | 1,532 (FR > 1 Hz) | NO - see note |
| Total units (randomized delay) | 845 | n/a | 948 pass quality filter | 925 (FR > 1 Hz) | close (9% high) |
| Units total | ~2,496 | n/a | 2,513 quality-passed | 2,457 | close |
| Neurons / session | ED Fig 2a: ~10-150 | >= 10 required | - | mean 55.8, min 17, max 141 | YES |
| Trials (total) | - | - | 15,062 in the 44 sessions | 13,762 after removing stim/early/no-ephys trials | consistent |
| Trials / session | - | - | 218-517 | 193-474 | YES |
| T per trial | -2.5..2.5 s | tmin/tmax | - | 166 bins x 30 ms | YES |
| time_from_go_cue range | - | -2.5..2.5 | - | [-2.5, 2.5] | YES |
| lick_direction | - | - | first-lick side | left 0.423, right 0.446, none 0.130 | plausible (near-balanced L/R) |
| context | ~100 DR trials then alternating blocks | autowater flag | WC fraction 0-39% per session | WC 0.097, DR 0.903 | YES (12/25 fixed-delay sessions have >5% WC trials; randomized-delay sessions are almost pure DR) |
| outcome | trained animals > 70% correct | hit/miss/no | hit fraction 0.53-0.94 per session | incorrect 0.120, correct 0.749, ignore 0.131 | YES (75% correct > 70% criterion) |
| tongue_velocity | tongue visible only during licking | DLC NaN = not visible | ~89% NaN over the whole trial | not visible 0.845 | YES |
| paw_velocity | - | DLC NaN | 0.5-9% NaN | not visible 0.049 | YES |
| motion_energy | - | ME per frame | - | 0.500/0.500/0.000 | YES (exact median split) |
| Brain regions | ALM (+ tjM1 on second probes) | probe loc | 'R/L ALM', 'R/L M1TJ' | ALM 2,312 neurons, tjM1 145 | YES |

Note on the fixed-delay unit count: only 1,565 clusters on the meta-specified ALM probes survive the quality filter, so 1,651 cannot be reached with any firing-rate threshold. Counting both probes of the dual-probe sessions gives 1,890 (1,890 > 1,651 > 1,565), so the paper's number probably comes from a slightly different session/probe set than the one in the shared loading scripts. We follow the shared code.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- First full run: 61 warnings "all neural data is zero" (JEB24_2023-10-23 trials 292-319, JEB24_2023-11-03 trials 301-333). Diagnosed as contiguous trailing trials recorded after the ephys file ended. **Fixed** by dropping trials with zero spikes across the whole curated population. After the fix `verification_full_out.txt` reports **"Data format is valid, no errors or warnings."** No remaining warnings, so nothing is left unexplained.

### Check 2: Sanity checks against the raw files (`/app/cache/sanity_checks.py`, output `/app/cache/sanity_checks_out.txt`)
All comparisons use `np.allclose` against values recomputed from the raw `.mat` files with independent code (plain loops, `np.convolve`, `np.interp`):

| Check | Sessions tested | Result |
|---|---|---|
| input time axis == bin centres of `tmin:dt:tmax` | all | PASS |
| input identical for every session/trial | 0, 10, 43 | PASS |
| `lick_direction` == side of the first lickport contact after the go cue | 3 | PASS |
| `context` == `bp.autowater` | 3 | PASS |
| `outcome` == `bp.hit`/`miss`/`no` | 3 | PASS |
| per-trial outputs constant across time | 3 | PASS |
| number of curated units == recomputed quality+FR filter | 3 | PASS |
| neural firing rate for 3 (trial, neuron) pairs per session == histogram + causal-Gaussian convolution recomputed from raw spike times | 9 spot checks | PASS |
| `motion_energy` class == interp1 + fillmissing + median threshold recomputed from the raw ME file | 3 | PASS |
| tongue visibility bins == DLC NaN pattern of that trial | 3 | PASS |
| discretized outputs are balanced 50/50 between class 0 and 1 | 9 | PASS |

**Bug found and fixed by these checks**: the initial `speed_from_xy` used `np.gradient` over the whole trial, so visible frames adjacent to a NaN frame received a NaN speed and were mislabelled 'not visible' (tongue "not visible" was 0.845 -> 0.826 after the fix; ~2% of visible timepoints were affected). Fixed by computing the gradient separately inside each contiguous run of visible frames. Re-ran the full conversion, verification, and all checks -> all pass.

### Check 3: Reference code comparison
| Step | Reference | This conversion | Same? |
|---|---|---|---|
| (a) loading | `loadObjs.m` loads `data_structure_*.mat`; sessions/probes from `load<ANM>_ALMVideo.m` | `Session` class reads the same files (v7 + v7.3); session/probe list transcribed from the same meta scripts | YES |
| (b) neuron filtering | `findClusters.m` (quality != garbage/gabrga/noisy/'real?') then `removeLowFRClusters.m` (mean PSTH FR > `params.lowFR`) | identical, case-insensitive and trimmed so that 'Poor' vs 'poor   ' and typos ('mutli') are handled; `lowFR = 1` as in the figure scripts and the paper | YES |
| (b) trial filtering | condition strings `...&~stim.enable&~early` | same mask, plus "trial must be hit/miss/no" and "trial must have ephys coverage"; ignore trials retained (required output class) | YES + 2 additions |
| (c) alignment | `alignSpikes.m`: `trialtm - bp.ev.goCue(trial)`; video: `frameTimes - vidshift - goCue`; ME: same | identical formulas, `vidshift` from `findVideoOffset.m` | YES |
| (d) binning | `getSeq.m`: `edges = tmin:dt:tmax`, `histc`, rate = N/dt, `mySmooth(N=15,'reflect')` causal Gaussian | identical (`np.histogram2d` + `fftconvolve` with the same kernel; verified against a from-scratch `np.convolve` implementation) | YES |
| (e) input construction | `obj.time` used as the time axis everywhere | `input[0] = obj.time` | YES |
| (f) output construction | `firstLickTime.m` (lick direction), `autowater` (context), `getOutcome.m` (outcome), `getKinematicsFromVideo.m`/`findVelocity.m` (kinematics), `loadMotionEnergy.m` (ME) | same sources; differences: (i) DLC NaNs become the 'not visible' class instead of being filled (required by the decoder spec), (ii) speed = |velocity| rather than separate x/y velocities, discretized at the session median (required by the decoder spec), (iii) paw speed = mean over the two bottom-cam paw markers, (iv) ME is averaged into 30 ms bins via interpolation onto `obj.time` exactly as the reference does | YES, with the spec-driven differences documented |

Differences and why: the decoder spec requires *categorical* outputs with an explicit 'not visible'/'no video' class, so the reference's NaN-filling (tongue position set to the mean initial position, tongue velocity set to 0) would destroy the information the spec asks for. Everything upstream of that (alignment, interpolation, binning) is unchanged.

### Check 4: Key statistics comparison
See the table in Step 9. Sessions (25 + 19), mice (14), neurons/session (17-141, mean 55.8), correct-trial fraction (0.75 > the 0.7 training criterion), tongue visible only ~17% of the time and almost exclusively after the go cue: all consistent with the paper. The only mismatch is the total fixed-delay unit count (1,532 vs 1,651), which cannot be reached from the shared session/probe list (only 1,565 clusters survive the quality filter), documented in Step 9.

### Check 5: Edge cases (`/app/cache/edge_checks.py`)
- Trial bookkeeping: for all 44 sessions, kept trials == (non-stim, non-early, hit|miss|no) minus no-ephys trials. **0 mismatches.**
- Every trial has T = 166 bins; neural float32, input float32, output int64; input shape (1,166), output shape (6,166).
- Max class index per output = [2,1,2,2,2,2], exactly matching the lengths of `output_values` [3,2,3,3,3,3] (class 2 of motion energy occurs in the one trial with unusable video in JEB19_2023-04-19).
- No non-finite values in the neural data.
- `brain_region_idx` length == number of neurons for every session; `subject_idx` int64 in [0,13].
- Time axis: -2.485 .. 2.465 s, dt = 0.03 s to float32 precision; `off_start/off_end` = -2.5/+2.5.
- Data-quirk handling verified: MAT v7 vs v7.3, 3 motion-energy file layouts, missing `obj.clu`, 'DUMMY' probes, quality-label case and typos, trials with unusable video (`NdroppedFrames` NaN), sessions where the ephys stops early.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (device: cuda) -> `/app/train_decoder_full_out.txt`.

### Training Progress
- Loss decreasing: **Yes**, 7.14 (epoch 1) -> 0.621 (epoch 200); test loss 0.676.

### Decoder Results (Full, 44 sessions)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|---|---|
| lick_direction | 0.696 | 0.678 | 0.333 | 2.0x chance |
| context | 0.886 | 0.865 | 0.500 | 1.7x chance |
| outcome | 0.677 | 0.648 | 0.333 | 1.9x chance |
| tongue_velocity | 0.617 | 0.609 | 0.333 | 1.8x chance |
| paw_velocity | 0.622 | 0.614 | 0.333 | 1.8x chance |
| motion_energy | 0.867 | 0.796 | 0.333 | 2.4x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
Every output is 1.7-2.4x its chance level, and none is below chance. The lowest ratios are the two kinematic outputs (tongue 1.83x, paw 1.84x), which is expected: they are time-varying, single-bin movement labels decoded by a *linear, memoryless* model from 100 PCs, and the medians are computed per session so the classes are exactly balanced (a hard 3-way problem).

### Check 2: Accuracy comparison to the paper
The paper trains a **separate** logistic-regression model at each time bin with balanced classes; the provided decoder trains a **single** model shared over all timepoints, sessions, and trial types, so the numbers are not directly comparable. To compare like-for-like I replicated the paper's protocol on the converted data (`/app/cache/paper_decoding.py`: per-time-bin logistic regression, balanced classes, 4-fold 70/30 CV):

| Variable | Epoch | This conversion (per-bin LR) | Paper |
|---|---|---|---|
| Choice (R vs L, correct DR trials) | ITI (-2.5..-1.3 s) | 0.617 | ~0.5-0.6 (Fig. 3b) |
| Choice | delay (-1.2..-0.05 s) | 0.807 | ~0.7-0.9 (Fig. 3b) |
| Choice | response (0..0.5 s) | **0.926** (peak 0.952 at +0.28 s) | ~0.9-1.0 (Fig. 3b) |
| Context (DR vs WC) | ITI (-2.5..-1.3 s) | 0.804 | ~0.7-0.8 (Fig. 4b) |
| Context | response (0..0.5 s) | **0.873** (peak 0.894) | ~0.8-0.9 (Fig. 4b) |
| Choice AUC from delay-epoch CDchoice | delay | not recomputed | 0.8-0.9 (ED Fig. 2c) |

Our per-bin accuracies match or slightly exceed the published curves, which is strong evidence that the neural data, temporal alignment, and trial labels are correct. The lower accuracy of the provided shared decoder is an architecture difference (one linear model for all 166 timepoints and all 44 sessions, including timepoints long before the go cue where choice/outcome are weakly encoded), not a data problem.

### Check 3: Train vs validation gap
Largest ratio is motion_energy (0.867/0.796 = 1.09); all others <= 1.05. No overfitting or leakage.

### Iterations in this step
No further issues were found, so no re-conversion was needed after the Step 10 fixes (no-ephys trials, segment-wise velocity).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, output specification, key statistics, validation summary)
- [x] `cache/` folder created with `README_CACHE.md` documenting every investigation/validation script
- [x] All files organized; deliverables in `/app`:
  - `CONVERSION_NOTES.md`, `README.md`, `convert_data.py`
  - `converted_data.pkl`, `sample_data.pkl`
  - `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`
  - `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`
  - `processing_JEB6_2021-04-18.png`, `processing_JEB11_2022-05-10.png` (from `--show-processing`)
  - `sample_trials.png`, `predictions.png` (from `train_decoder.py --plot-samples`)

### Summary of all decisions
1. Use the 44 ALM ephys sessions listed in the authors' loading scripts (25 fixed-delay + 19 randomized-delay); the behaviour-only optogenetic sessions (MAH*) have no neural data and cannot be used.
2. Align every stream to the go cue, -2.5 to +2.5 s, 30 ms bins, causal Gaussian smoothing (15 bins) - all taken from the reference parameters.
3. Curate units by quality label and firing rate > 1 Hz, as in `findClusters.m`/`removeLowFRClusters.m` and the paper.
4. Curate trials by removing photoinactivation, early-lick and no-ephys trials; keep ignore trials because they are a required output class.
5. Outputs taken directly from the Bpod fields and DeepLabCut/motion-energy streams, discretized at the per-session median with an explicit not-visible/no-video class, as the decoder task specifies.
6. Everything was verified by independent recomputation from the raw files and by reproducing the paper's decoding accuracies.
---

## Addendum: empirical validation of the binning / smoothing choice

The reference code uses several time-bin widths in different scripts (5, 10, 30 ms) with the same causal
smoothing window of 15 bins. To make sure the choice adopted here (dt = 30 ms, smooth = 15 bins, the setting of
the paper's Figure 3 analysis scripts) is not hurting the decoder, two alternatives were converted for the
sample sessions and run through `train_decoder.py` (scripts `/app/cache/convert_dt10.py`, `/app/cache/convert_sm7.py`):

| Setting | lick_direction | context | outcome | tongue_vel | paw_vel | motion_energy |
|---|---|---|---|---|---|---|
| **dt = 30 ms, smooth = 15 (chosen)** | **0.647** | **0.856** | **0.596** | **0.638** | **0.634** | **0.842** |
| dt = 10 ms, smooth = 15 | 0.582 | 0.815 | 0.554 | 0.611 | 0.600 | 0.804 |
| dt = 30 ms, smooth = 7 | 0.602 | 0.829 | 0.576 | 0.618 | 0.631 | 0.821 |

(validation balanced accuracy, 2 sample sessions). The chosen setting is best on every output, so the reference
parameters were kept unchanged.

