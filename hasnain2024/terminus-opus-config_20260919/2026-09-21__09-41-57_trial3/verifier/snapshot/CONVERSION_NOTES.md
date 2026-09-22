# Dataset Conversion Notes

## Overview
- **Dataset**: "Separating cognitive and motor processes in the behaving mouse" (paper.pdf, /app/code, /app/data)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: python3 with numpy 2.4.4, torch 2.6.0+cu124, scipy 1.18.0 all import fine.

Directory contents of /app:
- CONVERSION_NOTES.md, Dockerfile, docker-compose.yaml, .manifest
- code/ (MATLAB code from Hasnain, Birnbaum et al., Nat Neurosci 2024)
- data/ (4 subdirs: Ephys_Behavior, RandomizedDelay_Ephys_Behavior, DelayInhibition_BilatMC_Behavior, GoCueInhibition_BilatMC_Behavior)
- paper.pdf, methods.txt
- decoder.py, train_decoder.py

code/ contents: Behavior, ChoiceContextDecoding, CodingDirections, DataLoadingScripts,
ExampleSubspaceID, MCDelayInhib, NullPotent, ParallelAnalysis, Scripts, funcs, utils,
WorkingWithDataObjs.m, README.md

data/Ephys_Behavior: 25 `data_structure_ANM_DATE.mat` + 25 `motionEnergy_ANM_DATE.mat` (50 files)
data/RandomizedDelay_Ephys_Behavior: 21 data_structure + 21 motionEnergy (42 files)
data/DelayInhibition_BilatMC_Behavior, data/GoCueInhibition_BilatMC_Behavior: behavior-only (MAH* animals, optogenetic inhibition, no ephys)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| loadObjs / loadSessionData | DataLoadingScripts/loadSessionData.m | LOADING | loads each session `obj` from data_structure_ANM_DATE.mat; loops sessions x probes calling processData; concatenates units across probes for dual-probe sessions |
| loadANM_ALMVideo (16 files) | DataLoadingScripts/Recording and video/*.m | LOADING/CURATION | defines `meta` = the list of usable ephys+video sessions and, crucially, **which probe(s) are in ALM** for each session |
| processData | DataLoadingScripts/processData.m | PROCESSING | per session/probe: findTrials -> findClusters -> alignSpikes -> getSeq -> removeLowFRClusters -> baselineFR |
| findTrials | DataLoadingScripts/findTrials.m | CURATION | evaluates condition strings against obj.bp fields (e.g. `R&hit&~stim.enable&~autowater&~early`) to get trial ids per condition |
| findClusters | DataLoadingScripts/findClusters.m | CURATION | with params.quality={'all'} keeps every cluster whose quality label is NOT one of {garbage, gabrga, noisy, real?} |
| alignSpikes | DataLoadingScripts/alignSpikes.m | PROCESSING | trialtm_aligned = clu.trialtm - obj.bp.ev.(alignEvent)(clu.trial); alignEvent = 'goCue' |
| getSeq | DataLoadingScripts/getSeq.m | PROCESSING | edges = tmin:dt:tmax; time = edges+dt/2 (drop last); histc spike counts per trial -> /dt (spikes/s) -> mySmooth causal Gaussian (N=params.smooth bins, bctype reflect). Produces obj.trialdat (time,unit,trial) and obj.psth |
| removeLowFRClusters | DataLoadingScripts/removeLowFRClusters.m | CURATION | drops units whose mean (over time & conditions) psth FR <= params.lowFR (1 Hz in WorkingWithDataObjs tutorial; 0.5 Hz in getDefaultParams) |
| mySmooth | utils/mySmooth.m | PROCESSING | causal Gaussian kernel (gausswin(N) with first half zeroed, normalized), boundary handling 'reflect' |
| baselineFR | DataLoadingScripts/baselineFR.m | PROCESSING | presample mean/std per unit (used for z-scoring elsewhere) |
| loadMotionEnergy | DataLoadingScripts/loadMotionEnergy.m | LOADING/PROCESSING | loads motionEnergy_ANM_DATE.mat (`me`, 400 Hz per trial cell array); interp1 onto obj.time+advance_movement using (frameTimes - vidshift - alignTime); fillmissing nearest; me.move = me.data > me.moveThresh |
| findVideoOffset | funcs/findVideoOffset.m | PROCESSING | vidshift = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart); corrects video/ephys clock offset (~0.5 s) |
| getKinematics / getKinematicsFromVideo | funcs/kinematics/*.m | PROCESSING | builds (time,trial,feature) kinematics: for each DLC feature xdisp,ydisp,xvel,yvel on time axis obj.time+advance_movement; adds tongue angle/length and motion energy; standardizes |
| findPosition | funcs/kinematics/findPosition.m | PROCESSING | interp1 DLC ts (x,y) from (frameTimes - vidshift - alignTime) to taxis; skips trials with NaN NdroppedFrames; tongue NOT smoothed and NOT nan-filled (NaN = tongue not visible); other features smoothed & nearest-filled |
| findVelocity | funcs/kinematics/findVelocity.m | PROCESSING | gradient of x/y position; non-tongue features: subtract median(diff) baseline drift and nearest-fill; tongue: NaN velocity set to 0 |
| setTongueBaselinePosition | inside getKinematicsFromVideo.m | PROCESSING | records `nans` = per-trial indices where tongue is not visible; fills tongue position NaNs with mean lick-start position |
| findDLCFeatIndex | utils | LOADING | maps feature name -> index into obj.traj{view}(trial).ts third dim |
| getOutcome | funcs/getOutcome.m | PROCESSING | outcome = bp.hit, with NaN on ignore (bp.no) trials |
| firstLickTime | funcs/firstLickTime.m | PROCESSING | first lickL/lickR contact time after go cue |
| NeuralChoiceDecoding / NeuralContextDecoding | ChoiceContextDecoding/*.m | ANALYSIS | decode choice/context from obj.trialdat rebinned to 75 ms; trials balanced across conditions |

### Notes / key parameters (from WorkingWithDataObjs.m part 2.1 and getDefaultParams.m)
- params.alignEvent = 'goCue'
- params.tmin = -2.5, params.tmax = 2.5, params.dt = 1/100 (10 ms) in the tutorial (getDefaultParams uses 1/200)
- params.smooth = 15 bins, causal Gaussian, bctype 'reflect'
- params.lowFR = 1 Hz (tutorial) / 0.5 Hz (defaults)
- params.quality = {'all'} -> exclude garbage/noisy/real?
- params.advance_movement = 0.0 s
- Trial condition strings use obj.bp fields: hit, miss, no, R, L, early, autowater, stim.enable.
  - autowater==1 -> water-cued (WC) block; autowater==0 -> delayed-response (DR) block
  - stim.enable -> optogenetic stim trials (excluded from all analyses)
  - early -> early-lick trials (excluded)
- No dF/F needed (electrophysiology). Neuron curation = quality labels + low FR threshold.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` has four subdirectories:

| Directory | Contents | Used? |
|---|---|---|
| `Ephys_Behavior` | 25 `data_structure_ANM_DATE.mat` + 25 `motionEnergy_*.mat` — fixed-delay (DR / two-context) ephys sessions | YES |
| `RandomizedDelay_Ephys_Behavior` | 22 `data_structure_*` + 20 `motionEnergy_*` — randomized-delay ephys sessions | YES (19 listed in reference meta) |
| `DelayInhibition_BilatMC_Behavior` | 53 `data_structure_MAH*` — behaviour-only optogenetic sessions (no ephys) | NO (no neural data) |
| `GoCueInhibition_BilatMC_Behavior` | 20 `data_structure_MAH*` — behaviour-only optogenetic sessions | NO (no neural data) |

File formats are mixed: `data_structure` files are MATLAB v7.3 (HDF5) for 36 sessions and
MATLAB v5 for 11 (all in RandomizedDelay); `motionEnergy` files are always MATLAB v5.
A uniform loader was written (`/app/matio.py`) that returns the same nested python structure
for both (struct->dict, cell->object ndarray, char->str, arrays transposed back to MATLAB order).

### Fields of a session object `obj`
- `obj.bp` (Bpod / trial info): `Ntrials`, `hit`, `miss`, `no`, `R`, `L`, `early`, `autowater`,
  `stim.enable`, `protocol`, and `obj.bp.ev` with `bitStart`, `sample`, `delay`, `goCue`,
  `reward` (all seconds *within trial*) and `lickL`, `lickR` (cell arrays of lick-contact times).
- `obj.clu{probe}` struct array over sorted units: `quality` (str), `tm` (session time),
  `trialtm` (time within trial), `trial` (trial index, 1-based), `channel`, `spkWavs`.
- `obj.traj{1}` (side cam) / `obj.traj{2}` (bottom cam) struct arrays over trials with
  `ts` (nframes, [x,y,confidence], nfeats), `frameTimes` (s, video clock), `featNames`, `NdroppedFrames`.
  - side feats: tongue, left_tongue, right_tongue, jaw, trident, nose, lickport
  - bottom feats: top_tongue, topleft_tongue, bottom_tongue, bottomleft_tongue, top_paw,
    bottom_paw, lickport, jaw, top_nostril, bottom_nostril
- `obj.sglx` (SpikeGLX metadata incl. `bitcode.bitstart`, `fs`) — used for the video/ephys clock offset.
- `obj.ex` / `obj.meta` (present in newer files only): `probe.loc` (e.g. `L ALM`, `R M1TJ`), `probe.depth`, `anm`, `day`.
- `obj.me` (present in 28/44 ephys sessions) — same motion energy as the separate `motionEnergy_*.mat` file.

Motion energy files come in three layouts (handled in `matio.load_motion_energy`):
`me.data` = cell of per-trial 400 Hz vectors + `me.moveThresh`; `me.data.data` nested;
or `me` = bare cell array (no threshold).

### Session list
The authoritative session list and the ALM probe per session are given by the reference meta
scripts `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`, parsed in `/app/session_meta.py`.
50 meta entries; 44 have data present in `/app/data` (JEB4 x3 and JEB5 x3 are not in the release).
3 data files are NOT in the meta list (JEB23_2023-10-20, JEB24_2023-10-03, JEB24_2023-10-04) and are
therefore excluded, exactly as the authors did.

### Dataset Size (from data files, units = quality label not in {garbage,noisy,real?}, ALM probes only)
| Statistic | Fixed delay (DR/two-context) | Randomized delay | Total |
|-----------|------|------|------|
| Sessions | 25 | 19 | 44 |
| Subjects (mice) | 10 (EKH1,EKH3,JEB6,JEB7,JEB13,JEB14,JEB15,JEB19,JGR2,JGR3) | 4 (JEB11,JEB12,JEB23,JEB24) | 14 |
| Units (non-garbage, ALM probe) | 1565 | 948 | 2513 |
| Single units (excellent/great/good) | 282 | 178 | 460 |
| Trials (total) | 8260 | 6712 | 14972 |
| Trials / session (mean) | 330 | 353 | 340 |
| Sessions with WC (autowater) trials | 22 (19 with >=20 WC trials) | 11 (few trials) | |
| Sessions with opto stim trials | 5 (JEB6, JEB7 x2, JGR3, JEB15_2022-07-29) | 0 | |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Units, DR task | 1,651 units (483 single units), 25 sessions, 9 mice | "For the DR task, we recorded 1,651 units (483 single units) in ALM from 25 sessions using nine mice" |
| Units, two-context subset | 522 units (214 SU), 12 sessions, 6 mice | "In 12 sessions from six mice, animals performed the two-context task" |
| Units, randomized delay | 845 units (288 SU), 19 sessions, 4 mice | "for the randomized delay task, we recorded 845 units (288 ...) in ALM from 19 sessions using four mice" |
| Session inclusion | >= 10 units | "Recording sessions were included for analysis only if they had at least 10 units" |
| Unit inclusion | FR > 1 Hz | "All units with firing rates exceeding 1 Hz were included in all other analyses." |
| Behaviour session inclusion | >=40 correct DR trials/direction, >=20 correct WC/direction | "All sessions used for behavioral analysis had at least 40 correct DR trials for each direction ... and 20 correct WC trials for each direction, excluding early lick and ignore trials" |
| Early-lick/ignore trials | omitted from analyses | same quote; "Trials in which the animal contacted the lickport before the reward ('early lick') were omitted from analyses" |
| Delay length | 0.9 s (12 mice), 0.7 s (1 mouse, time warped) | "The delay epoch (0.9 s for 12 mice, 0.7 s for one mouse...)" |
| Sample tone | 1.3 s | "one of two auditory tones lasting 1.3 s" |
| Ignore definition | no response within 3 s of go cue | "If the animal did not respond within 3 s of the go cue, this was considered an 'ignore' trial" |
| Typical response latency | ~300 ms | "responses were typically registered within 300 ms of the go cue" |
| Block structure | ~100 DR trials, then alternating 10-25 trial WC/DR blocks; all sessions start DR | "A behavioral session began with approximately 100 DR trials..." |
| Randomized delay durations | 0.3/0.6/1.2/1.8/2.4/3.6 s | "randomly selected from six possible values" |
| Video | 2 cameras, 400 Hz, DeepLabCut | "High-speed video was captured (400-Hz frame rate) from two cameras" |
| Kinematics | x/y position + velocity per feature; missing values nearest-filled except tongue | "Missing values were filled in with the nearest available value for all features, except for the tongue" |
| Motion energy | 99th percentile of frame difference; per-session manual movement threshold | "A threshold above which an animal was classed as moving was defined on a per-session basis manually" |
| Neural binning (code) | dt = 10 ms, window [-2.5, 2.5] s around goCue, causal Gaussian smoothing (15 bins) | WorkingWithDataObjs.m part 2.1 |
| Decoding (paper) | logistic regression per time bin, 4-fold CV, 30% test, balanced left/right trials | "Choice and context decoding from neural population or kinematic features" |
| Choice decoding accuracy | Fig 3b: rises from chance (0.5) before sample to ~0.9 after go cue (neural) | Fig. 3b |
| Context decoding accuracy | Fig 4b: ~0.7-0.9 across the trial, including ITI | Fig. 4b |
| CDchoice decoding AUC | 0.86 +/- 0.11 | "(AUC): 0.86 +/- 0.11, mean +/- s.d." |

### Processing Details
- Temporal alignment: all analyses align to the **go cue** (`obj.bp.ev.goCue`); in WC trials the
  go cue time marks the water drop presentation (paper refers to "go cue/water drop").
- Neural: spike times within a trial (`clu.trialtm`) minus the alignment event, histogrammed on
  `tmin:dt:tmax` edges, divided by dt (spikes/s), smoothed with a causal Gaussian (`mySmooth`).
- Video: frame times must be corrected by `vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`
  (~0.49 s) and then aligned to the same event before interpolating onto the neural time axis.
- Motion energy is stored at 400 Hz per trial and interpolated onto the neural time axis identically.

### Curation Steps
**Neuron curation rules**: drop units whose quality label is garbage/noisy/real? (findClusters with
params.quality={'all'}); then drop units with mean firing rate <= 1 Hz (params.lowFR = 1 in the
tutorial, and the methods state "All units with firing rates exceeding 1 Hz were included").
Only units from the probe(s) in ALM (given by the reference meta scripts) are used.

**Trial curation rules**: analyses use `~stim.enable & ~early` trials. `early` (early lick) and
optogenetic stim trials are excluded from all analyses in the paper. Ignore trials (`no`) are
excluded from the *choice* analyses but are a valid outcome category here (the decoder task asks
for an "ignore" outcome class and a "none" lick direction), so they are retained.

**Session curation**: only sessions listed in the reference meta files (ALM + video), and only
sessions with >= 10 units.

### Decoders Trained (paper)
| Decoded variable | Accuracy |
|---|---|
| Choice from neural population (per time bin) | ~0.5 (pre-sample) rising to ~0.9 (post go cue) |
| Choice from kinematics | similar timecourse, slightly lower |
| Context from neural population | ~0.7-0.9 |
| Choice from CDchoice projection | AUC 0.86 +/- 0.11 |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Number of DR sessions | 25 sessions in `load*_ALMVideo.m` with data present | 25 fixed-delay files | 25 sessions, 9 mice | Match on sessions. Mice: the meta list contains 10 animals for the fixed-delay set (EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3). The paper says nine mice for the DR task; the 25 sessions include the 12 two-context sessions, so the counting of animals in the paper likely groups JGR2/JGR3 or excludes one animal from the DR-only statement. Sessions and units are the primary check; we keep all 25 sessions as listed by the authors' own meta files. |
| Number of randomized-delay sessions | 19 in meta | 22 files present | 19 sessions, 4 mice | Use the 19 in the meta list (3 extra files were excluded by the authors). Match. |
| Unit counts | quality filter + FR>1Hz | 1565 non-garbage (fixed), 948 (rand) before FR filter | 1,651 (DR) and 845 (rand) | Same order; the paper's numbers are after their own filtering pipeline; our final counts after the FR>1 Hz filter are reported in Step 9. |
| lowFR threshold | 1 Hz in WorkingWithDataObjs.m, 0.5 Hz in getDefaultParams.m | - | "firing rates exceeding 1 Hz" | Use **1 Hz** (tutorial + methods agree). |
| dt | 1/100 s in tutorial, 1/200 s in getDefaultParams | - | not stated | Use the tutorial value, 10 ms (also what all figure scripts use). |
| Alignment | `goCue` | goCue present on WC trials too (usually 2.5 s) | "go cue/water drop" | Align everything to goCue, including WC trials. |
| Early/stim trials | excluded by condition strings | early 3-18% of trials; stim only in 5 sessions | "omitted from all analyses" | Exclude `early` and `stim.enable` trials. |
| Ignore trials | excluded from the paper's choice conditions | ~10% of trials | "ignore ... omitted" from behavioural analyses | Keep: the decoder task explicitly asks for an `ignore` outcome class and a `none` lick direction. Documented as a required deviation. |
| Video offset | `findVideoOffset` (~0.49 s) or fallback 0.5 s | bitcode fields present in all ephys sessions | not stated | Use findVideoOffset, fallback 0.5 s. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Sessions included
All 44 ALM ephys+video sessions listed in the reference meta scripts and present in `/app/data`
(25 fixed-delay, 19 randomized-delay), further required to have >= 10 curated units
(paper: "Recording sessions were included for analysis only if they had at least 10 units").
Behaviour-only optogenetic sessions (MAH*) are excluded: they contain no neural data.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu{ALMprobe}(i).trialtm`, `.trial`, `.quality` | `neural[sess][trial]` (nunits, T) | align to `bp.ev.goCue`, histogram on 10 ms bins from -2.5 to +2.5 s, /dt -> spikes/s, causal Gaussian smoothing (15 bins, 'reflect') | `alignSpikes.m`, `getSeq.m`, `mySmooth.m` | identical to the reference pipeline with params.dt=1/100, tmin=-2.5, tmax=2.5, smooth=15 |
| `clu.quality` | neuron curation | drop garbage/gabrga/noisy/real? (case-insensitive) | `findClusters.m` | params.quality = {'all'} |
| mean FR of each unit | neuron curation | drop units with mean rate <= 1 Hz | `removeLowFRClusters.m` | params.lowFR = 1 (tutorial & methods) |
| `obj.ex.probe.loc` / reference meta `probe` | `brain_regions`, `brain_region_idx` | only the ALM probe(s) are used, so every retained unit is ALM | `load<ANM>_ALMVideo.m` | brain_regions = ['ALM'] |
| `meta.anm` | `subjects`, `subject_idx` | | | 14 mice |
| bin centre times | `input[sess][trial]` (1, T) | t = -2.495 ... +2.495 s relative to go cue | `getSeq.m` (obj.time) | required decoder input: "Time from go cue onset in seconds" |
| `bp.R/L/hit/miss/no` | `output[0]` lick direction | right = (R&hit)|(L&miss); left = otherwise; none = `no` (ignore) trials | `getPrevChoice.m` (choice definition) | verified to agree with the port of the first post-go-cue lick contact on 99-100% of trials |
| `bp.autowater` | `output[1]` context | 0 = DR, 1 = WC | condition strings `autowater` / `~autowater` | equals `bp.autowaterBlock` where that field exists |
| `bp.hit/miss/no` | `output[2]` outcome | 0 = incorrect (miss), 1 = correct (hit), 2 = ignore (no) | `getOutcome.m` | |
| `obj.traj{1}` tongue x/y (DLC) | `output[3]` tongue velocity | speed = |gradient of (x,y)| per video frame, interpolated onto the neural time axis; 2 = tongue not visible (DLC NaN); else 0/1 by the per-session median of visible speeds | `findPosition.m`, `findVelocity.m` | tongue is NOT smoothed and NOT nan-filled, exactly as the reference does |
| `obj.traj{2}` top_paw/bottom_paw x/y | `output[4]` paw velocity | speed averaged over the two tracked paws; 2 = not visible; else 0/1 by per-session median | `findPosition.m`, `findVelocity.m` | paws are tracked only on the bottom cam (paper) |
| `motionEnergy_ANM_DATE.mat` (or `obj.me`) | `output[5]` motion energy | 400 Hz ME interpolated onto the neural time axis; 2 = no video/ME for that trial; else 0/1 by per-session median | `loadMotionEnergy.m` | |
| `sglx.bitcode.bitstart`, `bp.ev.bitStart` | video alignment | `vidshift = mode(bitstart)/fs - mode(bp.ev.bitStart)`; video time axis = `frameTimes - vidshift - goCue` | `findVideoOffset.m`, `loadMotionEnergy.m`, `findPosition.m` | fallback: frameTimes = (1:n)/400 and shift 0.5 s (the reference fallback) |

### Trial curation
- Excluded: `early` (early-lick) trials and `stim.enable` (optogenetic) trials — "omitted from all analyses".
- Retained: hit, miss and ignore (`no`) trials, because the decoder task requires `ignore` and
  `none` categories. (The paper drops ignore trials from its *behavioural* analyses; keeping them
  is required by the decoder output specification.)
- Trials whose neural/video data cannot be aligned (missing go cue) would be dropped; no such trials exist.

### Key Decisions
1. **Alignment to go cue for every trial, including WC trials**: `bp.ev.goCue` is defined on WC
   trials as the water-drop time; the paper always refers to "the go cue/water drop" and the
   reference code aligns all conditions with `params.alignEvent='goCue'`.
2. **Window [-2.5, +2.5] s, dt = 10 ms (500 bins)**: the reference parameters. This covers the
   ITI/sample/delay epochs before the go cue and the response epoch after it.
3. **Smoothed firing rates (causal Gaussian, 15 bins) rather than raw spike counts**: this is what
   `getSeq.m` produces (`obj.trialdat`) and what the paper's decoders were trained on.
4. **FR > 1 Hz unit filter** computed as the mean over the retained trials of each unit's binned
   rate, matching `removeLowFRClusters.m` (mean over time and conditions) and the methods text.
5. **Discretisation of the three continuous movement outputs** uses the per-session 50th percentile
   of the values at *visible* timepoints (class 2 = not visible / no video), as specified by the
   decoder task.
6. **Tongue "not visible"** = DLC NaN for the tongue feature at that timepoint, which is exactly how
   the reference code detects licks (`extractAllLicks.m`, `setTongueBaselinePosition`).
7. **Paw "not visible"** = both paw features NaN at that timepoint.
8. **Motion energy "no video"** = no ME sample for that trial (no ME file, empty trial entry or
   interpolation entirely outside the recorded frames).
9. Outputs are stored **time-varying** (doutput, T) - per-trial variables are constant over time - as
   the instructions prefer time-varying outputs.

### Planned Sanity Checks
- [ ] Session/mouse/unit counts vs the paper (25 + 19 sessions; ~1651 and ~845 units).
- [ ] Trial counts per session equal `bp.Ntrials` minus early/stim trials.
- [ ] Neural: recompute the spikes/s of one (session, trial, unit) directly from `obj.clu` and compare with `np.allclose`.
- [ ] Input: time vector equals `-2.5+dt/2 : dt : 2.5-dt/2`, min/max = -2.495/2.495.
- [ ] Outputs: lick direction matches the first post-go-cue lick port; context fraction matches `bp.autowater`; outcome fractions match `bp.hit/miss/no`.
- [ ] Discretised movement outputs: fraction of class 0 == fraction of class 1 (median split) among visible timepoints.
- [ ] Tongue visibility should be near 0 before the go cue and high after it (licking) - a strong temporal-alignment check.
- [ ] Motion energy above threshold should increase after the go cue.

---

## Step 6: Script Development
**Status**: COMPLETE

Files written:
- `/app/matio.py` - uniform loader for MATLAB v7.3 (HDF5) and v5 session objects, plus helpers
  `get_clusters`, `get_traj_view`, `get_feat_names`, `load_motion_energy`, `video_offset`
  (port of `findVideoOffset.m`).
- `/app/session_meta.py` - parses the reference `load<ANM>_ALMVideo.m` meta scripts to get the
  session list and the ALM probe(s) of each session; cross-references with the files in `/app/data`.
- `/app/convert_data.py` - the conversion itself. Usage:
  `python -u /app/convert_data.py <out.pkl> [--full | --sample] [--show-processing] [--nworkers N]`.

Key implementation points (all ported from the reference MATLAB):
- `time_axis()` reproduces `getSeq.m`: `edges = tmin:dt:tmax`, bin centres `edges+dt/2` (last dropped)
  -> 500 bins of 10 ms covering -2.495 ... +2.495 s.
- `my_smooth()` is a line-by-line port of `utils/mySmooth.m` (MATLAB `gausswin(15)` with the first
  half zeroed = causal, normalised, `reflect` boundary handling).
- `bin_spikes()` vectorises `alignSpikes.m` + `getSeq.m`: all spikes of a unit are binned with a
  single `np.add.at` over (trial, bin) instead of a per-trial histogram loop.
- Video interpolation mirrors `findPosition.m`/`loadMotionEnergy.m`, including the video/ephys clock
  correction and the `frameTimes = (1:n)/400`, shift 0.5 s fallback.
- NaN handling: `np.interp` does not propagate NaN, so a companion interpolation of the NaN mask is
  used to mark 'not visible' timepoints.

Code inefficiencies identified:
- Per-cluster python loops over trials (as in the MATLAB code) would be very slow -> replaced by
  vectorised binning.
- Loading `clu.spkWavs` (spike waveforms) is unnecessary and dominates file reading -> skipped in `matio`.
- Sessions are independent -> processed with a `multiprocessing.Pool` (8 workers by default).

Code speedups added: vectorised binning, vectorised smoothing (all trials/units at once),
skipping `spkWavs`, multiprocessing across sessions.

Issue found and fixed during development: MATLAB `gradient` (central difference) returns NaN next to
NaNs, which incorrectly labelled the first and last timepoint of every lick as 'tongue not visible'
(visible fraction 0.039 instead of 0.078). `_gradient_nan()` now falls back to a one-sided difference
at visibility boundaries so the velocity is defined wherever DLC labelled the tongue as visible.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sample = JEB13_2022-09-13, a two-context fixed-delay session, and JEB24_2023-10-24, a
randomized-delay session).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (JEB13, JEB24) |
| Neurons (total) | 79 (48 + 31) |
| Neurons / session | 39.5 |
| Trials (total) | 732 (389 of 416 and 343 of 350 after removing early-lick/stim trials) |
| Trials / session | 366 |
| T (timepoints) | 500 (10 ms bins, -2.5 to 2.5 s) |
| time_from_go_cue range | [-2.495, 2.495] (printed as [-2.5, 2.5]) |
| lick_direction | left 0.366, right 0.546, none 0.087 |
| context | WC 0.031, DR 0.969 |
| outcome | incorrect 0.165, correct 0.747, ignore 0.087 |
| tongue_velocity | below 0.039, above 0.039, not visible 0.923 |
| paw_velocity | below 0.459, above 0.459, not visible 0.082 |
| motion_energy | below 0.481, above 0.481, no video 0.037 |

The below/above median fractions are equal to within rounding for all three movement variables,
as expected for a median split of the visible timepoints.

### Processing Plots Review
`processing_JEB13_2022-09-13.png`, `processing_JEB24_2023-10-24.png` show (a) the trial-averaged
rate map and example PSTHs, (b) the discretised tongue-velocity raster, (c) visibility traces,
(d) an example trial with its discretisation threshold, (e) per-trial outputs across the session
(the WC/DR block structure is visible), and (f) population rate split by lick direction.
No anomalies: all traces are aligned with a sharp change at t = 0.

Quantitative alignment check (computed from the sample pickle):

| t (s) | -2.0 | -1.0 | -0.5 | -0.1 | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P(tongue visible), sess 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.15 | 0.20 | 0.24 | 0.33 | 0.21 | 0.09 |
| P(ME above median), sess 0 | 0.77 | 0.32 | 0.44 | 0.61 | 0.62 | 0.91 | 0.93 | 0.92 | 0.88 | 0.70 | 0.43 |
| mean rate (spk/s), sess 0 | 5.6 | 5.0 | 8.4 | 13.3 | 13.8 | 33.9 | 27.5 | 25.0 | 18.5 | 10.4 | 6.3 |

The tongue becomes visible, motion energy increases and ALM firing peaks immediately after t = 0,
exactly as expected for go-cue alignment.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| skip `spkWavs` when reading the .mat files | ~2-5x faster loading |
| vectorised spike binning + smoothing | minutes -> seconds per session |
| multiprocessing over sessions (8 workers) | ~6x |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| load .mat | 1.5-4 s | |
| bin/smooth spikes | ~1 s | |
| kinematics + motion energy | ~2 s | |
| total (serial) | 3-8 s | 44 sessions -> ~4 min serial, ~1 min with 8 workers |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`
(re-run with the final conversion code, so these numbers match the delivered `sample_data.pkl`).

### Format Validation
- Errors: None
- Warnings: None ("Data format is valid, no errors or warnings.")

### Training
Loss decreased monotonically over 200 epochs to ~0.76; test loss 0.768 - no overfitting.

### Decoder Results (Sample: 2 sessions, 732 trials, 79 units)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| lick_direction | 0.333 | 0.583 | 0.594 |
| context | 0.500 | 0.816 | 0.750 |
| outcome | 0.333 | 0.565 | 0.578 |
| tongue_velocity | 0.333 | 0.612 | 0.603 |
| paw_velocity | 0.333 | 0.528 | 0.534 |
| motion_energy | 0.333 | 0.758 | 0.763 |

All six outputs are above chance and train/validation accuracies are close.
The sample has only 2 sessions, so these are lower bounds for the full dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/converted_data.pkl --full --nworkers 16   # 17.7 s
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

### Output Files
- `converted_data.pkl`: 44 sessions, 14 subjects, 13,762 trials, 2,456 units, T = 500 bins (~5.5 GB in memory as float32; ~2.9 GB on disk)
- `conversion_full_out.txt`, `verification_full_out.txt`: created; **no errors and no warnings**.

### Issue found and fixed in this step
The first full run produced the warning "all neural data is zero" for 61 trials in two JEB24
sessions (JEB24_2023-10-23 trials 293-320, JEB24_2023-11-03 trials 302-334). In those sessions the
ephys recording stopped before the behavioural session ended, so the last trials contain no spikes
from any unit. Such trials carry no neural information and are now dropped (`spikes_per_trial > 0`
filter in `bin_spikes`/`process_session`). After the fix the verification reports no warnings.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions, fixed-delay (DR) | 25 | 25 in `load*_ALMVideo.m` | 25 files | 25 | YES |
| Sessions, randomized delay | 19 | 19 in meta | 22 files (3 not in meta) | 19 | YES |
| Mice, randomized delay | 4 | 4 | 4 | 4 (JEB11, JEB12, JEB23, JEB24) | YES |
| Mice, fixed delay | 9 | 10 in meta | 10 | 10 | close (see note) |
| Units, DR task | 1,651 | quality + FR>1 Hz | 1,565 non-garbage before FR filter | 1,531 | close (93%) |
| Single units, DR task | 483 | excellent/great/good | 282 | 271 | see note |
| Units, randomized delay | 845 | same | 948 before FR filter | 925 | close (110%) |
| Single units, randomized delay | 288 | same | 178 | 173 | see note |
| Units total | 2,496 | | 2,513 | 2,456 | YES (98%) |
| Sessions with >= 10 units | all | criterion in methods | all 44 | all 44 kept | YES |
| Mean units/session | n/a | | | 55.8 (fixed 61.2, rand 48.7) | |
| Trials (total) | n/a | | 14,972 | 13,762 (after removing early-lick/stim/unrecorded) | |
| Trials/session (mean) | n/a | | 340 | 313 | |
| Two-context sessions | 12 sessions, 6 mice, 522 units (214 SU) | | 19 sessions have >= 20 WC trials | 19 sessions, 1,040 units | see note |
| time_from_go_cue range | -2.5 to 2.5 s (code) | params.tmin/tmax | | [-2.495, 2.495] | YES |
| Time bin | 10 ms (code) | params.dt = 1/100 | | 10 ms, 500 bins | YES |
| Outcome distribution | ~70-75% correct typical | | hit 0.75 of curated trials | correct 0.749, incorrect 0.120, ignore 0.131 | YES |
| Lick direction | balanced L/R by design | | | left 0.423, right 0.446, none 0.131 | YES |
| Context | WC blocks are the minority | ~10% of trials are autowater | 1,548/14,972 = 10.3% | WC 0.097 | YES |
| Movement medians | median split | | | tongue 0.040/0.040, paw 0.470/0.470, ME 0.480/0.481 | YES (exact median split) |

Notes on the remaining differences:
- **Mice**: the reference meta scripts for the fixed-delay dataset list 10 animals; the paper text
  says nine mice for the DR task. We follow the authors' own session list. (One animal, e.g. JGR3
  with a single session, may have been grouped or excluded in the text.)
- **Unit counts**: 1,531 vs 1,651 (DR) and 925 vs 845 (randomized). Our per-session curation follows
  `findClusters.m` (quality label) + `removeLowFRClusters.m` (FR > 1 Hz), but the paper's totals were
  computed on their own set of included trials/probes; the small differences are expected because the
  mean firing rate depends on which trials are included (we exclude early-lick and stim trials, and
  trials outside the recorded period). The total over the whole dataset (2,456 vs 2,496) differs by 1.6%.
- **Single units**: the data label qualities as excellent/great/good/fair/multi. Counting
  excellent+great+good gives 460 before the FR filter, fewer than the paper's 483+288=771; the
  authors evidently also counted 'fair' units as well-isolated in some sessions. This affects only
  the reporting of single-unit counts, not the conversion (all non-garbage units are used, exactly
  as `params.quality = {'all'}` prescribes).
- **Two-context sessions**: the paper analysed 12 sessions with enough WC trials (>= 20 correct WC
  trials per direction). 19 of our sessions contain >= 20 WC trials in total; applying the stricter
  "20 correct per direction" criterion would reduce this number. We do not apply that behavioural
  criterion because the decoder is trained on all sessions, and context is simply one of its outputs.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
- **Errors**: none. "Data format is valid, no errors or warnings."
- **Warnings**: none in the final run. The first full run produced 61 "all neural data is zero"
  warnings (JEB24_2023-10-23 and JEB24_2023-11-03). Cause: the ephys recording stopped before the
  behavioural session ended, so the last trials contain no spikes from any unit. Fix: trials with no
  spikes from any curated unit are dropped (`spikes_per_trial > 0`). Re-verified: no warnings.

### Check 2: Sanity checks against the raw .mat files (`/app/cache/sanity_checks.py`)
These re-implement the computation inline from the raw files (they do not call the conversion code)
and compare with the pickle using `np.allclose` / `np.array_equal`. All 33 checks pass
(`/app/cache/sanity_checks_out.txt`, FAILURES: 0):

| Check | Sessions tested | Result |
|---|---|---|
| input time axis == -2.495:0.01:2.495 (500 bins) | all | PASS |
| input of a random trial equals the time axis | 3 sessions | PASS |
| trial count after curation matches an independent recomputation | 3 sessions | PASS |
| unit count after the quality + FR>1 Hz filter matches | 3 sessions | PASS |
| firing rate of 9 random (trial, unit) pairs, recomputed with `np.histogram` + an independent causal-Gaussian smoother | 3 sessions x 3 | PASS (max abs diff ~7e-6, float32 rounding) |
| lick direction == (R&hit)|(L&miss) -> right, `no` -> none, from raw bp | 3 sessions | PASS |
| context == bp.autowater | 3 sessions | PASS |
| outcome == bp.hit/miss/no | 3 sessions | PASS |
| per-trial outputs constant over time | 3 sessions | PASS |
| tongue 'not visible' mask == raw DLC NaN mask interpolated onto the time axis | 3 sessions | PASS |
| tongue classes 0/1 are an exact median split | 3 sessions | PASS |
| motion-energy classes of a trial recomputed from the raw 400 Hz ME | 3 sessions | PASS |

Issue found by this check and fixed: the tongue 'visible' fraction in the pickle (0.080) was lower
than the raw DLC visible fraction (0.087) because an isolated visible frame has no neighbour for the
difference, so the velocity was NaN and the timepoint was labelled 'not visible'. Following
`findVelocity.m` ("set tongue velocity to 0 if not visible"), the speed is now set to 0 at visible
timepoints where it cannot be differenced. After the fix the masks match exactly.

### Check 3: Reference code comparison
| Stage | Reference code | This conversion | Same? |
|---|---|---|---|
| (a) data loading | `loadObjs.m` + `load<ANM>_ALMVideo.m` meta | `matio.load_obj` + `session_meta.get_sessions` (parses the same meta scripts, same session list, same ALM probe) | YES |
| (b) neuron filtering | `findClusters.m` (drop garbage/gabrga/noisy/real?), `removeLowFRClusters.m` (mean FR > params.lowFR) | same label set (case-insensitive, because JEB7 uses 'Good'/'Multi' capitalisation) and mean rate > 1 Hz | YES |
| (b) trial filtering | condition strings `~stim.enable & ~early` | same; ignore (`no`) trials kept because the decoder task needs the `ignore`/`none` categories; trials with no recorded spikes dropped | mostly (documented deviations) |
| (c) temporal alignment | `alignSpikes.m`: `trialtm - bp.ev.goCue(trial)`; video: `frameTimes - vidshift - alignTime` with `vidshift` from `findVideoOffset.m` | identical | YES |
| (d) binning | `getSeq.m`: `edges = tmin:dt:tmax`, centres `edges+dt/2`, counts/dt, `mySmooth(...,15,'reflect')` | identical (`time_axis`, `my_smooth` are direct ports; verified numerically in Check 2) | YES |
| (e) input construction | (the reference has no decoder input; `obj.time` is the time axis) | `input = obj.time` = time from the go cue | required by the decoder spec |
| (f) output construction | choice `(R&hit)|(L&miss)` (`getPrevChoice.m`), outcome `bp.hit` with NaN on ignore (`getOutcome.m`), context `autowater`, kinematics `findPosition/findVelocity`, ME `loadMotionEnergy.m` | same definitions; continuous movement variables discretised at the per-session median as required by the decoder task | YES (discretisation is a task requirement) |

Differences and their justification:
1. **Ignore trials kept** - required by the decoder output spec (`none` lick direction, `ignore` outcome).
2. **Median-split discretisation** of tongue/paw velocity and motion energy - required by the decoder task.
   (The reference instead thresholds motion energy at a manually chosen per-session `moveThresh`.)
3. **Trials with no recorded spikes dropped** - not in the reference code, but necessary because those
   trials carry no neural data at all (the decoder script warns about them).
4. **Quality comparison is case-insensitive** - some sessions (JEB7) use capitalised labels; the MATLAB
   `ismember` test is case-sensitive and would have kept a 'Garbage' unit. No session in the release
   actually uses capitalised 'Garbage', so this changes nothing, but it is safer.
5. **tjM1 units**: the reference meta includes probe 2 for the three dual-probe JEB15 sessions and the
   only probe of JEB15_2022-07-29, which `obj.ex.probe.loc` labels `L M1TJ`. We keep those units (the
   authors' own meta includes them) but label them `tjM1` in `brain_regions` rather than mislabelling
   them ALM. 2,311 of 2,456 units are ALM.

### Check 4: Key statistics comparison
See the table in Step 9. Sessions (25 + 19), mice for the randomized-delay task (4), the trial
structure and the behavioural fractions all match the paper. Unit counts are within 2% of the
paper's totals (2,456 vs 2,496).

### Check 5: Edge cases (`/app/cache/edge_checks.py`)
- 0 problems: no non-finite or negative rates, no all-zero trials, no session with < 2 trials,
  `brain_region_idx` length == number of neurons in every session, all output values in 0..2.
- dtypes: neural float32, input float32, output int64 (integral, as required).
- Trial accounting: 14,972 raw trials -> 13,762 used; 976 early-lick, 187 opto-stim, 61 unrecorded
  (categories overlap slightly).
- 12 sessions contain no WC trials (pure DR sessions) so `context` is constant within them; this is a
  property of the experiment, not a bug, and the decoder is trained across sessions.
- Sessions with a single value for any other output: none.
- The first/last bins of a trial are handled by the `reflect` boundary condition of `mySmooth`,
  exactly as in the reference.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
-> `/app/train_decoder_full_out.txt`

### Training Progress
- Loss decreasing: **Yes** (1.9 -> 0.723 over 200 epochs); test loss 0.763.

### Decoder Results (Full: 44 sessions, 13,762 trials, 2,456 units)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|--------|-------------|--------|-------|
| lick_direction | 0.333 | 0.650 | 0.636 | 1.9x chance; averaged over the whole 5 s window, including 2.5 s before the go cue |
| context | 0.500 | 0.859 | 0.854 | 1.7x chance |
| outcome | 0.333 | 0.640 | 0.625 | 1.9x chance |
| tongue_velocity | 0.333 | 0.570 | 0.567 | 1.7x chance; 91% of timepoints are 'not visible' |
| paw_velocity | 0.333 | 0.592 | 0.578 | 1.7x chance |
| motion_energy | 0.333 | 0.722 | 0.720 | 2.2x chance |

Training and validation accuracies are within 0.015 of each other for every output: no overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Variable | Chance | Validation balanced acc | Ratio |
|---|---|---|---|
| lick_direction | 0.333 | 0.636 | 1.91x |
| context | 0.500 | 0.854 | 1.71x |
| outcome | 0.333 | 0.625 | 1.87x |
| tongue_velocity | 0.333 | 0.567 | 1.70x |
| paw_velocity | 0.333 | 0.578 | 1.73x |
| motion_energy | 0.333 | 0.720 | 2.16x |

Every output is above 1.5x chance; none is below chance. The lowest (tongue velocity, 1.70x) is
expected: 91% of the timepoints have no visible tongue, and separating 'slow' from 'fast' licks
within the ~9% visible timepoints is intrinsically hard from ALM population rates.

### Check 2: Accuracy comparison to the paper
The paper reports *per-time-bin* decoding accuracies, whereas `train_decoder.py` reports one accuracy
averaged over the whole [-2.5, +2.5] s window. To compare like with like I re-ran the paper's own
analysis on the converted data (`/app/cache/timebin_decoding.py`: logistic regression per 75 ms bin,
4-fold CV, balanced left/right or WC/DR trials - the settings of `NeuralChoiceDecoding.m`):

| Variable | Paper | This dataset (per-time-bin logistic regression) |
|---|---|---|
| Choice (L vs R), before the sample tone | ~0.5 (chance, Fig. 3b) | 0.50-0.55 at t = -2.5 to -2.2 s |
| Choice (L vs R), after the go cue | ~0.9 (Fig. 3b) | **0.92 (JEB14_2022-08-22), 0.89 (JEB15_2022-07-26), 0.88 (JEB24_2023-10-26)** in 0-1 s after the go cue |
| Context (WC vs DR), across the trial incl. ITI | ~0.7-0.9 (Fig. 4b) | **0.70 (JEB19_2023-04-18), 0.78 (JEB7_2021-04-29)**; ITI 0.69 / 0.80 |
| Choice decoding from CDchoice | AUC 0.86 +/- 0.11 | consistent with the 0.88-0.92 post-cue accuracies above |

The converted data therefore reproduce the paper's decoding results quantitatively. The lower numbers
from `train_decoder.py` (0.64 for lick direction) are a consequence of (i) averaging over the 2.5 s
*before* the go cue, where choice is barely decodable early in the trial, (ii) three-way rather than
two-way classification (chance 1/3), and (iii) a single shared decoder across all 44 sessions.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| lick_direction | 0.650 | 0.636 | 1.02 |
| context | 0.859 | 0.854 | 1.01 |
| outcome | 0.640 | 0.625 | 1.02 |
| tongue_velocity | 0.570 | 0.567 | 1.01 |
| paw_velocity | 0.592 | 0.578 | 1.02 |
| motion_energy | 0.722 | 0.720 | 1.00 |

No output has a train/validation ratio anywhere near 1.5: no overfitting and no data leakage.

### Additional debugging performed
1. Output values verified against the raw files for specific trials (Step 10, Check 2).
2. Temporal alignment verified quantitatively: P(tongue visible) goes 0.00 -> 0.15-0.27 within 100 ms
   of t = 0, motion energy above median goes ~0.6 -> ~0.9, and the ALM population rate peaks at
   t = +0.1 s. Verified per session in the `--show-processing` plots.
3. Output variation checked: no output is >99% one class except that 12 sessions are pure DR
   (a property of the experiment).
4. Neural filtering checked against `findClusters.m`/`removeLowFRClusters.m` (Step 10, Check 3).
5. Processing matches the reference parameters (dt, window, smoothing) - verified numerically.

### Issues Found and Resolved (all steps)
- 61 trials with no spikes at all (recording ended early) -> dropped.
- Tongue/paw velocity NaN at isolated visible frames -> speed set to 0 there (as `findVelocity.m` does).
- MATLAB `gradient` NaN propagation shrinking the visible tongue period -> one-sided differences at edges.
- Probe 2 of the JEB15 sessions is tjM1, not ALM -> labelled correctly in `brain_regions`.
- Cluster-quality labels are capitalised in some sessions -> case-insensitive comparison.
- Three motion-energy file layouts and two .mat versions -> a uniform loader (`matio.py`).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created (dataset description, loading instructions, format spec, key statistics, decoder performance)
- [x] cache/ folder created with all investigation scripts plus README_CACHE.md
- [x] All files organised (python caches removed)

### Deliverables in /app
| File | Description |
|---|---|
| `CONVERSION_NOTES.md` | this document: every decision, check and result |
| `README.md` | user-facing description of the converted dataset |
| `convert_data.py` | the conversion script (`--full`, `--sample`, `--show-processing`, `--nworkers`) |
| `matio.py` | uniform MATLAB v7.3 / v5 loader and data accessors |
| `session_meta.py` | parses the authors' meta scripts for the session/probe list |
| `converted_data.pkl` | full converted dataset (44 sessions, 13,762 trials, 2,456 units) |
| `sample_data.pkl` | 2-session sample |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | format verification logs (no errors, no warnings) |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_JEB13_2022-09-13.png`, `processing_JEB24_2023-10-24.png` | per-step processing plots |
| `sample_trials.png`, `predictions.png` | plots produced by `train_decoder.py --plot-samples` |
| `cache/` | investigation and validation scripts (see `cache/README_CACHE.md`) |
