# Dataset Conversion Notes

## Overview
- **Dataset**: Hasnain, Birnbaum et al., *"Separating cognitive and motor processes in the behaving mouse"*, Nature Neuroscience 2024. Data: Zenodo DOI 10.5281/zenodo.13941415 (local copy in `/app/data`), code in `/app/code`, paper in `/app/paper.pdf`, methods in `/app/methods.txt`.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (align to **go cue**; decode lick direction, context, outcome, tongue velocity, paw velocity, motion energy from ALM spiking).

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `paper.pdf` (8.2 MB), `methods.txt` (18 kB) — reference text
- `code/` — MATLAB code accompanying the paper (`Behavior`, `ChoiceContextDecoding`, `CodingDirections`, `DataLoadingScripts`, `ExampleSubspaceID`, `MCDelayInhib`, `NullPotent`, `ParallelAnalysis`, `Scripts`, `funcs`, `utils`, `WorkingWithDataObjs.m`, `README.md`)
- `data/` — 4 sub-directories:
  - `Ephys_Behavior/` (4.1 GB): 25 `data_structure_<ANM>_<DATE>.mat` + 25 `motionEnergy_<ANM>_<DATE>.mat` (fixed-delay DR task; 12 of these are the two-context DR/WC sessions)
  - `RandomizedDelay_Ephys_Behavior/` (2.9 GB): 22 `data_structure_*.mat` + motion energy (randomized-delay DR task)
  - `DelayInhibition_BilatMC_Behavior/` (5.9 GB) and `GoCueInhibition_BilatMC_Behavior/` (2.5 GB): optogenetic, **video/behavior only, no ephys**
- `decoder.py`, `train_decoder.py` — provided decoder
- `Dockerfile`, `docker-compose.yaml`, `.manifest`

Python environment verified: `numpy 2.4.4`, `torch 2.6.0+cu124`, `h5py`, `scipy`, `sklearn`, `matplotlib` all import.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

`code/README.md` + `WorkingWithDataObjs.m` are the entry points. Each session is one MATLAB struct `obj` with fields
`bp` (Bpod trial/task data), `clu` (sorted spikes, cell array 1×nProbes), `traj` (DeepLabCut output, cell 2×1 = {side cam, bottom cam}), `sglx` (SpikeGLX metadata), `ex`/`meta` (session metadata incl. probe location), `me` (motion energy, only in some objs), `pth`, `trials`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load<ANM>_ALMVideo(meta,datapth)` | `DataLoadingScripts/Recording and video/` | LOADING / CURATION | Hard-coded list of **which sessions** and **which probe** is ALM. Commented-out entries = sessions the authors excluded. |
| `loadObjs` / `loadSessionData` | `DataLoadingScripts/` | LOADING | Loads each session obj, calls `processData` per probe, concatenates probes. |
| `processData` | `DataLoadingScripts/processData.m` | PROCESSING | Pipeline: `findTrials` → `findClusters` → `alignSpikes` → `getSeq` → `removeLowFRClusters` → `baselineFR`. |
| `findTrials(obj,conditions)` | `DataLoadingScripts/findTrials.m` | CURATION | Evaluates condition strings (e.g. `'R&hit&~stim.enable&~autowater&~early'`) over `obj.bp` fields to get trial indices per condition. |
| `findClusters(qualityList,{'all'})` | `DataLoadingScripts/findClusters.m` | CURATION | `'all'` ⇒ keep every cluster whose quality is **not** `garbage`, `gabrga`, `noisy`, `real?` (case-sensitive `ismember` after `strtrim`). |
| `alignSpikes(obj,params,prbnum)` | `DataLoadingScripts/alignSpikes.m` | ALIGNMENT | `trialtm_aligned = trialtm - ev.(alignEvent)(trial)`; for us `alignEvent = 'goCue'`. |
| `getSeq(obj,params,prbnum)` | `DataLoadingScripts/getSeq.m` | PROCESSING | `edges = tmin:dt:tmax`; `obj.time = edges+dt/2` (drop last) ⇒ bin centres. `histc` spike counts per bin → `/dt` → `mySmooth(...,params.smooth,params.bctype)`. Produces `obj.psth` (time,units,conditions) and `obj.trialdat` (time,units,trials) = **single-trial smoothed firing rate**. |
| `mySmooth(x,N,'reflect')` | `utils/mySmooth.m` | PROCESSING | **Causal** Gaussian smoothing: `gausswin(15)` (MATLAB α = 2.5), first `floor(N/2)=7` taps zeroed, normalised, `conv(...,'same')`; `'reflect'` prepends `x(1:N)` and trims. |
| `removeLowFRClusters(obj,cluid,lowFR,prb)` | `DataLoadingScripts/removeLowFRClusters.m` | CURATION | `meanFRs = mean(mean(psth,3,'omitnan'),'omitnan')` (mean over conditions then over time); keep units with `meanFR > params.lowFR` (= **1 Hz** in every figure script). |
| `findVideoOffset(obj)` | `funcs/findVideoOffset.m` | ALIGNMENT | `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`; video `frameTimes` must have `vidshift` subtracted to live in the Bpod clock (≈0.5 s for old sessions, ≈0.99 s for JEB19). |
| `loadMotionEnergy(obj,meta,params,datapth)` | `DataLoadingScripts/loadMotionEnergy.m` | PROCESSING | Loads `motionEnergy_<ANM>_<DATE>.mat` (cell array, one 400 Hz trace per trial), `interp1(frameTimes − vidshift − alignTime, me, taxis)`, then `fillmissing(...,'nearest')`. `me.move = me.data > me.moveThresh`. |
| `findPosition(taxis,obj,nTrials,view,feat,alignEv)` | `funcs/kinematics/findPosition.m` | PROCESSING | DLC x/y for one feature, `interp1(frameTimes − vidshift − alignTime, ts, taxis)`. **Tongue is not smoothed and NaNs are not filled**; all other features `fillmissing('nearest')`. Skips trials with `NdroppedFrames = NaN`. |
| `findVelocity(xpos,ypos,feat)` | `funcs/kinematics/findVelocity.m` | PROCESSING | `gradient()` of the *interpolated* position; for non-tongue features subtracts the baseline drift `median(diff(pos))`; tongue NaN velocity → 0, other features `fillmissing('nearest')`. |
| `getKinematics` / `getKinematicsFromVideo` | `funcs/kinematics/` | PROCESSING | Assembles (xdisp,ydisp,xvel,yvel) per feature per view + tongue angle/length + motion energy. |
| `getPrevChoice` | `funcs/getPrevChoice.m` | OUTPUT DEF | **Lick direction**: right ⇔ `(R&hit) | (L&miss)`, left ⇔ `(L&hit) | (R&miss)`, `NaN` for ignore (`no`). |
| `getOutcome` | `funcs/getOutcome.m` | OUTPUT DEF | `hit` = correct, `miss` = incorrect, `no` = ignore (set to NaN there). |
| `getBlockNum_AltContextTask` | `CodingDirections/funcs/` | OUTPUT DEF | Context blocks are defined purely by transitions of `bp.autowater` ⇒ `autowater==1` ⇔ **WC**, `autowater==0` ⇔ **DR**. |
| `UseInclusionCritera` / `RemoveUnwantedSessions` | `utils/` | CURATION | Optional extra session filter (>40 right-hit and >40 left-hit DR trials). **Not called by any of the figure scripts** — the session lists in `load<ANM>_ALMVideo.m` are already curated. |

### Reference processing parameters (figure scripts)
| Param | Value | Where |
|---|---|---|
| `alignEvent` | `'goCue'` | every script |
| `dt` | `1/100` s = **10 ms** | every figure script (`getDefaultParams` has 1/200 but is unused by the figures) |
| `tmin, tmax` | `-2.5, 2.5` (tutorial, `getDefaultParams`, EDFig2) / `-3, 2.5` (Fig 8, Fig 3, Fig 1e) | scripts |
| `smooth` | `15` samples, causal Gaussian | every script |
| `bctype` | `'reflect'` | every figure script |
| `lowFR` | `1` Hz | every figure script |
| `quality` | `{'all'}` | every figure script |
| `advance_movement` | `0` | every figure script |

### Notes
- Electrophysiology ⇒ no ΔF/F. Neurons **are** quality-filtered (drop `garbage`/`noisy`/`real?`) and rate-filtered (>1 Hz).
- The reference **never** z-scores or normalises `trialdat` at load time (z-scoring happens later inside specific analyses); the decoder here does its own PCA/standardisation, so exported neural data = smoothed single-trial firing rate in spikes/s.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`.mat` files are a mix of MATLAB v7.3 (HDF5 → `h5py`) and v7 (→ `scipy.io.loadmat`); all `data_structure_*.mat` in `Ephys_Behavior` are v7.3, some in `RandomizedDelay_*` are v7. `motionEnergy_*.mat` are v7.

`obj.bp` (per-trial, length `Ntrials`): `L`, `R` (cued/rewarded side), `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `protocol`, and `ev.{bitStart,sample,delay,goCue,reward,lickL,lickR}` (times in seconds within the trial; `lickL/lickR` are cell arrays of contact times).

`obj.clu{probe}` is a struct array over clusters with `quality` (char), `tm` (session spike times), `trialtm` (time within trial), `trial` (1-based trial index), `site/channel`, `spkWavs`.

`obj.traj{view}(trial)`: `ts` (frames × [x,y,confidence] × bodypart), `frameTimes` (400 Hz, video clock), `featNames`, `NdroppedFrames`.
- view 1 (side): `tongue, left_tongue, right_tongue, jaw, trident, nose, lickport`
- view 2 (bottom): `top_tongue, topleft_tongue, bottom_tongue, bottomleft_tongue, top_paw, bottom_paw, lickport, jaw, top_nostril, bottom_nostril`

`motionEnergy_*.mat` → `me.data` (cell, one 400 Hz vector per trial, same length as `frameTimes`) and `me.moveThresh` (manual per-session move threshold).

Event timing (example JEB19 2023-04-18, relative to go cue): `bitStart` −2.46 s, `sample` −2.2 s, `delay` −0.9 s, `goCue` 0, reward ≈ +0.37 s (DR) / +0.57 s (WC).

### Dataset inventory (all ephys data in `/app/data`)
| Group | Sessions in folder | Sessions in reference loaders | Mice |
|---|---|---|---|
| `Ephys_Behavior` (fixed delay) | 25 | 25 (EKH1 1, EKH3 1, JEB13 5, JEB14 4, JEB15 4, JEB19 4, JEB6 1, JEB7 2, JGR2 2, JGR3 1) | 10 |
| — of which **two-context (DR+WC)** | 12 | 12 (JEB6 1, JEB7 2, EKH1 1, EKH3 1, JGR2 2, JGR3 1, JEB19 4) | 7 |
| `RandomizedDelay_Ephys_Behavior` | 22 | **19** (JEB11 2, JEB12 2, JEB23 7, JEB24 8; 3 extra files not in loaders) | 4 |
| `*_BilatMC_Behavior` | — | behaviour/video only, **no neural data** | — |

### Dataset size (12 two-context sessions, after curation — see Steps 5/9)
| Statistic | Value |
|-----------|-------|
| Neurons (total, quality-filtered, >1 Hz) | 521 |
| Neurons / session | 27–66 (mean 43.4) |
| Subjects | 7 (EKH1, EKH3, JEB19, JEB6, JEB7, JGR2, JGR3) |
| Sessions / subject | 1–4 |
| Trials (total, after `~early & ~stim`) | 3116 |
| Trials / session | 210–390 (mean 259.7) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Two-context sessions / mice / units | **12 sessions, 6 mice, 522 units, 214 well-isolated single units** | "two-context paradigm: 12 sessions, six mice, 522 units, including 214 well-isolated single units" |
| DR task (all fixed-delay) | 25 sessions, 9 mice, 1,651 units (483 single) | "For the DR task, we recorded 1,651 units (483 single units) in ALM from 25 sessions using nine mice." |
| Randomized delay | 19 sessions, 4 mice, 845 units (288 single) | "for the randomized delay task, we recorded 845 units (288 well-isolated single units) in ALM from 19 sessions using four mice" |
| Recording region | ALM (left and right) | "We recorded activity extracellularly in the ALM with high-density silicon probes" |
| Neural time bin | 10 ms (`params.dt = 1/100`) | reference code (all figure scripts) |
| Video/behaviour sampling | 400 Hz | "High-speed video was captured (400-Hz frame rate)" |
| Alignment | go cue | `params.alignEvent = 'goCue'` |
| Analysis window | −2.5…2.5 s (also −3…2.5 s in Fig 1/3/8) | reference code |
| Sample tone | 1.3 s | "one of two auditory tones lasting 1.3 s was played" |
| Delay | 0.9 s (0.7 s for one mouse, warped to 0.9) | methods |
| Ignore window | 3 s after go cue | "If the animal did not respond within 3 s of the go cue, this was considered an 'ignore' trial" |
| Block structure | ~100 DR trials first, then alternating 10–25-trial blocks; **all sessions start with DR** | methods |
| Behavioural inclusion | ≥40 correct DR trials/direction, ≥20 correct WC trials/direction | methods |
| Session inclusion | ≥10 units | "Recording sessions were included for analysis only if they had at least 10 units" |
| Unit rate inclusion | >1 Hz | "All units with firing rates exceeding 1 Hz were included in all other analyses." |
| Trials omitted | early-lick trials (and, for the paper's own analyses, ignore trials) | "Trials in which the animal contacted the lickport before the reward ('early lick') were omitted from analyses"; "excluding early lick and ignore trials, which were omitted from all analyses" |
| Photoinactivation | ~30% of trials in opto sessions | methods |
| Motion energy | 99th-percentile of |median(next 5 frames) − median(prev 5 frames)| per frame; manual per-session bimodal threshold | methods |
| Kinematics | DLC x/y per feature per camera; missing values nearest-filled **except tongue**; velocity = first derivative of position; paws only from the bottom view | methods |

### Measured statistics from the data (12 two-context sessions, `~early & ~stim`)
| Statistic | Value |
|---|---|
| Total kept trials | 3116 |
| Lick direction (left / right / none) | 0.398 / 0.377 / 0.225 |
| Outcome (incorrect / correct / ignore) | 0.106 / 0.669 / 0.225 |
| Context (WC / DR) | 0.315 / 0.685 |
| Blocks per session | 8–16 |
| First WC trial index | 91–140 (paper: "approximately 100 DR trials" first) ✔ |

### Processing Details
- Temporal alignment: spike times → `trialtm − goCue(trial)`; video → `frameTimes − vidshift − goCue(trial)` with `vidshift = mode(sglx.bitcode.bitstart)/fs − mode(bp.ev.bitStart)`; motion energy uses the same video clock.
- Temporal binning: 10 ms, causal Gaussian smoothing (15-sample window) applied to spike-count/dt.

### Curation Steps
**Neuron curation rules**: probe = the ALM probe listed in `load<ANM>_ALMVideo.m`; drop clusters with quality `garbage`/`gabrga`/`noisy`/`real?`; drop clusters with condition-averaged mean firing rate ≤ 1 Hz.

**Trial curation rules**: drop `early` (early-lick) trials and `stim.enable` (photoinactivation) trials. Keep `hit|miss|no` (ignore trials are *retained* here because the decoder task requires an "ignore" outcome class — a deliberate, documented deviation from the paper's own analyses).

**Session curation rules**: only the 12 two-context sessions, because "behavioural context (WC, DR)" is a required decoder output and only these sessions contain both contexts. All 12 have ≥10 units.

### Decoders Trained (in the paper)
| Decoded variable | Method | Accuracy reported |
|---|---|---|
| Choice (right vs left) from **DLC kinematic features** | linear decoder, 75 ms bins (`DLC_ChoiceDecoding.m`) | Fig. 3/4: rises from chance (0.5) before the go cue to ≈0.9–1.0 after the go cue |
| Context (DR vs WC) from **DLC kinematic features** | same (`DLC_ContextDecoding.m`) | ≈0.7–0.9 across the trial |
| Choice / context from **neural activity** | `NeuralChoiceDecoding.m`, `NeuralContextDecoding.m` | near-perfect after the go cue for choice; high and sustained for context |
| Context coding direction (CD~context~) | ROC/selectivity | 39% of single units context-selective during the ITI |
No single scalar "decoder accuracy" table is given in the paper; the decoding curves are the reference (chance = 0.5 for the binary decoders).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| # two-context mice | 7 animals in the Fig-8/EDFig-2a loader list (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19) | 7 distinct animal IDs | "six mice" | Paper appears to under-count by one (it also says "nine mice" for the 25 fixed-delay sessions where the code/data give 10 animals — the same off-by-one). **Session counts (12 and 25) match exactly**, so I follow the code/data and report 7 subjects. |
| # two-context units | quality filter + `lowFR=1` on the ALM probe | 522 with Fig-8 params (`tmin=-3`), 521 with `tmin=-2.5`, 519 if mean FR is taken over all trials instead of over conditions | 522 | Confirms the session list, probe choice and both filters. I use `tmin=-2.5` ⇒ **521** units (see Step 5 rationale); difference from 522 is a single unit at the window edge. |
| # single units | `excellent`/`great`/`good`/`fair` (after `strtrim`, case-insensitive) that pass the >1 Hz filter | **214** | 214 | Exact match — confirms the quality→single-unit mapping and the rate filter. (The tutorial says "excellent, great, good treated as single units"; including `fair` is what reproduces 214, and `poor`/`multi` are then the multi-units.) |
| Randomized-delay sessions | 19 in loaders | 22 files on disk | 19 | Loader list is the curated one. (Not used here anyway — no WC context.) |
| Ignore trials | excluded by every `params.condition` string | present in data | "excluded from all analyses" | **Retained**, because the decoder task defines `ignore` as an outcome class and `none` as a lick-direction class. Documented deviation. |
| `params.dt` | `1/200` in `getDefaultParams.m`, `1/100` in all figure scripts (and the tutorial comment wrongly says "5 ms") | — | — | Use **10 ms** (`1/100`), the value actually used for all published analyses. |
| Video offset | `findVideoOffset` (data-driven) vs. "subtract 0.5 s" in the tutorial | offset is 0.5 s for 2021 sessions, 0.99 s for JEB19 | — | Use `findVideoOffset` (the tutorial's 0.5 s is a special case). |
| `me` inside `obj` vs separate file | active code path loads the separate `motionEnergy_*.mat` | both exist for JEB19 | — | Use the separate file, as the reference does; verified they agree (Step 10). |

**Resolved understanding**: the reference pipeline, the data layout and the paper agree once (a) the 12 two-context sessions with their ALM probe are selected, (b) quality + 1 Hz filters are applied, and (c) early-lick and photoinactivation trials are dropped.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu{probe}(c).trialtm`, `.trial`, `obj.bp.ev.goCue` | `neural[s][t]` (n_units, 500) | `trialtm - goCue(trial)`; `histc` into 500 x 10 ms bins over [-2.5, 2.5]; `/dt`; causal Gaussian smoothing | `alignSpikes.m`, `getSeq.m`, `mySmooth.m` | spikes/s, float32; equals `obj.trialdat` of the reference pipeline |
| `obj.clu{probe}(c).quality` | neuron selection | drop `garbage`/`gabrga`/`noisy`/`real?` (after `strtrim`, case-sensitive) | `findClusters.m` (`params.quality={'all'}`) | 1757 of 2287 clusters dropped (almost all Kilosort `garbage` in the JEB19 Neuropixels sessions) |
| condition-averaged PSTH | neuron selection | mean over the 7 Fig-8 conditions (`omitnan`) then over time; keep > 1 Hz | `removeLowFRClusters.m` (`params.lowFR=1`) | 8 further clusters dropped |
| time axis | `input[s][t]` (1, 500) | bin centres `-2.495 … 2.495` s | `getSeq.m` (`obj.time`) | the single decoder input required by the task |
| `obj.bp.R/L/hit/miss/no` | `output[0]` lick_direction | `right = (R&hit)|(L&miss)`, `left = (L&hit)|(R&miss)`, `none = no` | `getPrevChoice.m` | per trial, broadcast over time |
| `obj.bp.autowater` | `output[1]` context | `1 -> WC (0)`, `0 -> DR (1)` | `getBlockNum_AltContextTask.m`, `params.condition` strings | per trial |
| `obj.bp.hit/miss/no` | `output[2]` outcome | `miss->0 incorrect`, `hit->1 correct`, `no->2 ignore` | `getOutcome.m` | per trial |
| `obj.traj{1}(t).ts[:, :, tongue]` | `output[3]` tongue_velocity | resample x,y onto the aligned 10 ms grid, `speed = |grad(x,y)|`, split at the per-session 50th percentile of the visible samples; NaN (DLC did not detect the tongue) -> class 2 | `findPosition.m`, `findVelocity.m` | time-varying |
| `obj.traj{2}(t).ts[:, :, top_paw/bottom_paw]` | `output[4]` paw_velocity | as above, averaged over the two paw markers, with the reference's baseline-drift removal; NaN -> class 2 | `findPosition.m`, `findVelocity.m` | paws are tracked only by the bottom camera (Methods) |
| `motionEnergy_<ANM>_<DATE>.mat: me.data` | `output[5]` motion_energy | `interp1` onto the aligned grid + `fillmissing('nearest')`, split at the per-session 50th percentile; class 2 only if the trial has no video | `loadMotionEnergy.m` | class 2 has no instances after the no-video trial is dropped |
| `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, `obj.bp.ev.bitStart` | video alignment | `vidshift = mode(bitstart)/fs - mode(bitStart)`; every video time becomes `frameTimes - vidshift - goCue(trial)` | `findVideoOffset.m` | 0.490 s for the 2021 sessions, 0.990 s for JEB19 |
| animal name in the filename | `subjects`, `subject_idx` | unique list of 7 animals | `load<ANM>_ALMVideo.m` | |
| `obj.ex.probe.loc` / `load<ANM>_ALMVideo.m` | `brain_regions`, `brain_region_idx` | all units are ALM | reference loaders | verified `L ALM` / `R ALM` where `obj.ex` exists |

### Key Decisions
1. **Use only the 12 two-context sessions** (JEB6 1, JEB7 2, EKH1 1, EKH3 1, JGR2 2, JGR3 1, JEB19 4).
   *Rationale*: "behavioural context (WC, DR)" is a required decoder output, and these are the only
   ephys sessions containing both contexts. They are exactly the sessions loaded by
   `Scripts/Figure 8` and `Scripts/EDFigure 2a-left`, and reproduce the paper's 522-unit /
   214-single-unit figure. The 13 DR-only fixed-delay sessions and the 19 randomized-delay
   sessions have no WC trials, so adding them would add a constant, undecodable context label.
2. **One probe per session (the ALM probe)**: taken from `load<ANM>_ALMVideo.m`. All 12
   two-context sessions are single-ALM-probe sessions. Verified against `obj.ex.probe.loc`
   where present (`R ALM` for JEB6 probe 2, `L ALM` for JEB19 probe 1).
3. **Align to the go cue** (`obj.bp.ev.goCue`), as required by the task and as in every reference
   figure script (`params.alignEvent = 'goCue'`). In WC trials this Bpod event marks the water
   presentation, which is the behavioural analogue of the go cue.
4. **Window [-2.5, +2.5] s, 10 ms bins (500 timepoints)**. `dt = 1/100` is the value used by every
   published figure. `tmin = -2.5` is the value in the tutorial, `getDefaultParams.m` and
   EDFigure 2; Figures 1/3/8 use `-3`. I chose `-2.5` because the high-speed video starts at trial
   onset (~-2.48 s relative to the go cue), so this window is fully covered by video on essentially
   every trial, whereas `-3` would require extrapolating ~0.5 s of the behavioural outputs.
   Cost: 521 instead of 522 units pass the >1 Hz filter (one unit sits on the boundary).
5. **Neural data = smoothed single-trial firing rate** (`obj.trialdat`), not raw spike counts:
   this is the quantity every reference analysis uses, and it is what the reference decoding
   scripts feed to their classifiers.
6. **Neuron curation**: quality filter + >1 Hz condition-averaged mean rate, exactly as in
   `findClusters.m` / `removeLowFRClusters.m`, using the condition list of `Figure8a_thru_c.m`
   (the script that analyses this very data set).
7. **Trial curation**: drop early-lick (`bp.early`) and photoinactivation (`bp.stim.enable`) trials,
   which every `params.condition` in the reference excludes. **Keep ignore (`bp.no`) trials** even
   though the paper excludes them — the decoder task requires "ignore" as an outcome class and
   "none" as a lick-direction class. Also drop the single trial whose video is missing
   (JEB19 2023-04-19), because three of the six outputs are undefined for it and
   `findPosition.m` skips such trials too.
8. **Per-trial outputs are broadcast across time** rather than stored as 1-D arrays, so that all
   six outputs share one `(6, T)` array (required because three of them are time-varying).
9. **Discretisation**: the 50th percentile is computed **per session over the time points at which
   the feature is available**, because "not visible"/"no video" is a separate class. This makes
   classes 0 and 1 exactly balanced within every session.
10. **"not visible" vs "no video"**: the task wording is different for the two cases and so is the
    treatment. Tongue/paw class 2 = DeepLabCut produced no position at that time point (a
    per-time-point property of tracking). Motion-energy class 2 = the trial has no video at all (a
    trial-level property); within a trial the reference's `fillmissing('nearest')` is applied, so
    the handful of samples at the trial edges that the video does not cover are filled rather than
    labelled "no video" — exactly as `loadMotionEnergy.m` does.
11. **Single input** (`time_from_go_cue`, continuous), as specified by the task.

### Planned Sanity Checks
- [x] 12 sessions, 7 animals, 521 units, 214 single units (paper: 12 / 6 / 522 / 214)
- [x] every session starts with a DR block and the first WC trial appears near trial ~100
- [x] spike raster aligned to the go cue: licking starts at t = 0
- [x] licks on trials labelled `left` hit the left port and vice versa; `none` trials are silent
- [x] `vidshift` reproduces the 0.5 s offset documented in `WorkingWithDataObjs.m` for old sessions
- [x] resampled DeepLabCut traces overlay the raw 400 Hz traces
- [x] independent re-derivation of the smoothed firing rate, and of all six outputs, from the raw
      `.mat` files with `np.allclose` / `np.array_equal`
- [x] paper-style per-session choice/context decoding on the converted data reproduces the
      published accuracy curves

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`, run as `python -u /app/convert_data.py <out.pkl> [--full|--sample] [--show-processing]`.

Structure: `matstr`/`mode_value`/`gausswin`/`make_causal_kernel`/`my_smooth`/`nearest_fill` (ports of
`mySmooth.m` and MATLAB `fillmissing`), `load_bpod`, `video_offset` (`findVideoOffset.m`),
`bin_spikes` (+`alignSpikes.m`/`getSeq.m`), `fig8_conditions` + `low_fr_mask`
(`findTrials.m`/`removeLowFRClusters.m`), `load_motion_energy` (`loadMotionEnergy.m`),
`extract_kinematics` (`findPosition.m`/`findVelocity.m`), `discretize`, `convert_session`,
`plot_processing`, `main`.

Code inefficiencies identified and removed:
- per-cluster per-trial `histc` loops (as in `getSeq.m`) replaced by one `np.histogram2d` per cluster
  over (aligned spike time x trial) — the whole binning step is 0.1-0.35 s per session;
- smoothing applied once to the reshaped `(T, nunits*ntrials)` matrix instead of per trial;
- PSTHs for the low-FR filter are obtained by averaging the *already smoothed* single-trial data
  (valid because `mySmooth` is a linear operator), so spikes are binned only once;
- `nearest_fill` is a vectorised `searchsorted` instead of an O(n^2) nearest-index search;
- DeepLabCut trajectories are read once per trial and all features resampled from that read.

Run time: **25 s for all 12 sessions** (~2 s/session; 1.5-2 s of that is HDF5 reads of the
DeepLabCut trajectories). No parallelism needed.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
-> `/app/conversion_sample_out.txt`, `/app/processing_JEB6_2021-04-18.png`,
`/app/processing_JEB7_2021-04-29.png`.

### Sample Statistics (sessions JEB6 2021-04-18 and JEB7 2021-04-29)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 97 (31, 66) |
| Single units | 45 |
| Subjects | 2 (JEB6, JEB7) |
| Sessions / subject | 1, 1 |
| Trials (total) | 547 (302, 245) |
| Timepoints / trial | 500 (10 ms bins) |
| `time_from_go_cue` range | [-2.495, 2.495] s |
| lick_direction (left, right, none) | [0.410, 0.448, 0.143] |
| context (WC, DR) | [0.329, 0.671] |
| outcome (incorrect, correct, ignore) | [0.119, 0.739, 0.143] |
| tongue_velocity (low, high, not visible) | [0.063, 0.063, 0.875] |
| paw_velocity (low, high, not visible) | [0.495, 0.495, 0.010] |
| motion_energy (low, high, no video) | [0.500, 0.500, 0.000] |

### Processing Plots Review
`processing_<session>.png` has 12 panels covering every processing step:
1. raw spike raster aligned to the go cue — spikes are dense from t = 0 onwards, no offset;
2. 10 ms binning and causal-Gaussian smoothing of one trial — the smoothed trace rises only
   *after* each spike, confirming the kernel is causal;
3. histogram of condition-averaged mean rates with the 1 Hz cut;
4. trial-averaged population rate, units sorted by peak time — the expected go-cue response;
5. **lick raster sorted by the exported `lick_direction`** — trials labelled `left` contain almost
   exclusively left-port contacts, `right` trials right-port contacts, and `none` trials are empty.
   This is the strongest alignment check: contacts start exactly at t = 0;
6. per-trial `context`/`lick_direction`/`outcome` overlaid on the raw `bp.autowater` trace — the
   context label is the exact complement of `autowater` and reproduces the block structure;
7. raw 400 Hz DeepLabCut tongue trace with the resampled 10 ms trace overlaid — they superimpose
   exactly, and the tongue only becomes visible after the go cue;
8/9. tongue speed with the session median and the resulting 0/1/2 category; category raster;
10/11/12. paw speed and motion energy with their session medians and category rasters — motion
   energy is low before and high after the go cue, as expected.
No anomalies. One quirk noted: in JEB19 2023-04-20 the last three (WC ignore) trials have the
tongue marker stuck at a fixed position with high DeepLabCut confidence, which pulls that
session's tongue-speed median down to 0.49 (other sessions: 2.5-10). This is genuine DeepLabCut
output and the reference applies no extra confidence filtering, so it was left as is.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| `histogram2d` instead of per-trial `histc` | ~20x on the binning step |
| single smoothing call on the reshaped matrix | ~10x on the smoothing step |
| PSTHs reused from the smoothed single-trial data | avoids a second full binning pass (7 conditions) |
| vectorised `nearest_fill` | ~50x on the kinematics step |

| Step | Time / Session | Estimated Total Time (12 sessions) |
|---|---|---|
| spike binning | 0.11-0.35 s | ~2.5 s |
| smoothing | 0.07-0.32 s | ~2.2 s |
| kinematics + motion energy | 1.3-2.2 s | ~20 s |
| **total** | **1.6-3.1 s** | **~25 s (measured)** |

Format verification (`/app/verification_sample_out.txt`): **no errors, no warnings**.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None

### Training
Loss decreased monotonically from 7.38 (epoch 1) to 0.720 (epoch 200); test loss 0.755.

### Decoder Results (Sample, 2 sessions)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| lick_direction | 0.333 | 0.6374 | 0.6320 |
| context | 0.500 | 0.7940 | 0.7711 |
| outcome | 0.333 | 0.6404 | 0.6286 |
| tongue_velocity | 0.333 | 0.6127 | 0.6217 |
| paw_velocity | 0.333 | 0.5070 | 0.5178 |
| motion_energy | 0.333 | 0.8067 | 0.8094 |

Every output is above chance, and train/validation are nearly identical (no overfitting).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 361 MB, 12 sessions, 3115 trials, 521 units
- `conversion_full_out.txt`, `verification_full_out.txt`: created; **no errors, no warnings**

### Per-session summary
| session | probe | clusters | after quality | after >1 Hz | single units | trials | kept | early | stim | vidshift (s) | tongue thr | paw thr | ME thr | me.moveThresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| JEB6_2021-04-18 | 2 | 33 | 33 | 31 | 13 | 382 | 302 | 25 | 58 | 0.490 | 4.12 | 0.396 | 15.5 | 10.0 |
| JEB7_2021-04-29 | 1 | 67 | 67 | 66 | 32 | 346 | 245 | 62 | 48 | 0.490 | 6.18 | 0.375 | 16.0 | 15.0 |
| JEB7_2021-04-30 | 1 | 27 | 27 | 27 | 15 | 254 | 210 | 18 | 27 | 0.490 | 8.39 | 0.368 | 13.5 | 10.0 |
| EKH1_2021-08-07 | 2 | 48 | 48 | 48 | 19 | 305 | 252 | 53 | 0 | 0.490 | 8.80 | 0.454 | 13.5 | 10.0 |
| EKH3_2021-08-11 | 2 | 66 | 66 | 66 | 31 | 426 | 390 | 36 | 0 | 0.490 | 10.21 | 0.392 | 18.9 | 10.5 |
| JGR2_2021-11-16 | 1 | 29 | 29 | 29 | 12 | 267 | 214 | 53 | 0 | 0.490 | 5.43 | 0.423 | 19.1 | 10.0 |
| JGR2_2021-11-17 | 1 | 53 | 53 | 53 | 13 | 254 | 226 | 28 | 0 | 0.490 | 5.81 | 0.520 | 15.0 | 7.5 |
| JGR3_2021-11-18 | 1 | 28 | 28 | 28 | 12 | 261 | 230 | 30 | 2 | 0.490 | 3.53 | 0.674 | 16.0 | 9.0 |
| JEB19_2023-04-18 | 1 | 436 | 36 | 36 | 12 | 301 | 284 | 17 | 0 | 0.990 | 3.93 | 0.246 | 8.5 | 8.0 |
| JEB19_2023-04-19 | 1 | 567 | 38 | 37 | 11 | 230 | 215 | 14 | 0 | 0.990 | 2.54 | 0.252 | 8.0 | 8.5 |
| JEB19_2023-04-20 | 1 | 475 | 69 | 67 | 32 | 324 | 297 | 27 | 0 | 0.990 | 0.49 | 0.291 | 7.5 | 8.0 |
| JEB19_2023-04-21 | 1 | 458 | 35 | 33 | 12 | 276 | 250 | 26 | 0 | 0.990 | 4.57 | 0.285 | 9.5 | 8.0 |

(JEB19 2023-04-19 keeps 215 of the 216 non-early/non-stim trials; one trial has no video.)

An independent cross-check of the per-session median motion-energy threshold against the
authors' manually set `me.moveThresh` shows they agree closely (e.g. 8.5 vs 8.0, 9.5 vs 8.0,
15.5 vs 10.0) — the 50th percentile of motion energy over the trial window sits at essentially
the same place as the movement threshold the authors chose by eye from the bimodal distribution.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Two-context sessions | 12 | 12 (Fig 8 / EDFig 2a loaders) | 12 files with alternating `autowater` blocks | 12 | YES |
| Subjects | "six mice" | 7 animals in the loaders | 7 animal IDs | 7 | code/data, paper appears to miscount (see Step 4) |
| Total units | 522 | 522 with Fig-8 params (`tmin=-3`); 521 with `tmin=-2.5` | 2287 raw clusters on the ALM probes | 521 | YES (1-unit window-edge difference) |
| Well-isolated single units | 214 | 214 | - | 214 | YES |
| Units / session | EDFig 2a shows 27-67 | - | - | 27-67 (mean 43.4) | YES |
| Sessions with >= 10 units | all | criterion in Methods | all | all 12 | YES |
| Neural time bin | - | `params.dt = 1/100` = 10 ms | - | 10 ms | YES |
| Window | - | [-2.5, 2.5] / [-3, 2.5] | - | [-2.5, 2.5], 500 bins | YES |
| Alignment | go cue | `params.alignEvent='goCue'` | - | go cue | YES |
| Trials | - | `~early & ~stim.enable` | 3626 total in these 12 sessions | 3115 = 3626 - 389 early - 135 stim + 14 counted twice (early & stim) - 1 no video; every trial is hit, miss or no | YES |
| Sessions start with DR | "approximately 100 DR trials" then alternating blocks | `getBlockNum_AltContextTask.m` | first WC trial at index 91-140 in all 12 | same | YES |
| Blocks per session | "10-25 trials" per block | - | 8-16 blocks/session | same | YES |
| `time_from_go_cue` range | - | `obj.time` = -2.495 … 2.495 | - | [-2.495, 2.495] | YES |
| lick_direction | - | `getPrevChoice.m` | 0.398 / 0.377 / 0.225 | 0.398 / 0.377 / 0.225 | YES |
| context | ~1/3 WC (block design) | `autowater` | 0.315 / 0.685 | 0.315 / 0.685 | YES |
| outcome | DR performance > 70% | `getOutcome.m` | 0.106 / 0.670 / 0.225 | 0.106 / 0.670 / 0.225 | YES (correct/(correct+incorrect) = 86%) |
| tongue_velocity | - | - | tongue visible 3-14% of the time | 0.041 / 0.041 / 0.919 | YES |
| paw_velocity | - | paws from bottom cam only | paw NaN 0.6-7.6% per session | 0.492 / 0.492 / 0.015 | YES |
| motion_energy | - | `fillmissing('nearest')` | 1 trial of 3116 without video | 0.499 / 0.501 / 0.000 | YES |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There are no errors and no warnings to address. Specifically:
- all 500-timepoint trials, all sessions consistent in `dinput`/`doutput`;
- no NaN/Inf anywhere in `neural`, `input` or `output`;
- neural data are float32 with real variability (not 0/1);
- `brain_region_idx` lengths match the neuron counts, `subject_idx` is an integer array in range.

One informational (not warning-level) point: `motion_energy` class 2 (`no_video`) has zero
instances, so the summary prints only two of the three declared `output_values` and
`train_decoder.py` reports chance as 1/3 rather than the effective 1/2. This is intentional —
after the single video-less trial is dropped (Step 5, decision 7) every analysed trial has video.
The third value is kept in `output_values` because the decoder task defines the encoding as
`0/1/2`, and it documents what a `2` would mean.

### Check 2 — Independent sanity checks (`/app/cache/sanity_checks.py`)
This script re-derives everything from the raw `.mat` files without importing `convert_data.py`.
Full output in `/app/cache/sanity_checks_out.txt`. **All checks pass.**

| # | Check | Result |
|---|---|---|
| A | 12 sessions, 7 subjects, 521 units, 214 single units, 3115 trials, all sessions >= 10 units, `brain_region_idx`/`subject_idx` consistent | PASS |
| B | `input` equals the 10 ms bin-centre grid `-2.495 … 2.495` for every sampled trial of every session | PASS |
| C | For 9 random (session, trial, unit) triples in 3 sessions, the exported firing rate equals an independently coded re-binning + causal-Gaussian smoothing of the raw `obj.clu` spike times (`np.allclose`, max abs diff < 4e-6) | PASS |
| C | Total smoothed "mass" (`sum(rate)*dt`) of a unit over all trials is within 2% of its raw spike count in the window (1464.4 vs 1465; 18289.7 vs 18261; 10375.7 vs 10367) | PASS |
| D | For all 12 sessions: exported `lick_direction`/`context`/`outcome` equal the values recomputed from `obj.bp` (`np.array_equal`); no early or stim trial is present; the kept-trial count equals the expected count; the three per-trial outputs are constant in time | PASS |
| E | Behavioural cross-check in 3 sessions: the side of the **first lickport contact after the go cue** matches the exported `lick_direction` on 644/649 responded trials (99.2%), and all 151 trials labelled `none` really contain no post-go-cue contact | PASS |
| F | `vidshift` reproduced exactly in 3 sessions; for 9 random trials the tongue, paw and motion-energy category time series are reproduced bin-for-bin by an independent re-implementation (0 mismatching bins out of 500) | PASS |
| G | In every session and for each of the three movement variables, classes 0 and 1 contain the same number of samples to within 2% — i.e. the split really is at the session median | PASS |
| H | Methods' behavioural inclusion criterion (>=40 correct DR trials per direction, >=20 correct WC per direction) — 7/12 sessions meet it strictly | informational (see below) |

On Check H: five sessions fall slightly short of the criterion the Methods state for *behavioural*
analyses (e.g. JEB7 2021-04-30 has 16 correct WC right trials, JEB19 2023-04-19 has 31 correct DR
left trials). Those sessions are nevertheless part of the authors' 12-session neural data set
(`Scripts/Figure 8`, `Scripts/EDFigure 2a-left`), and the paper's 522-unit / 214-single-unit
counts are only reproduced when all 12 are included. The criterion is therefore not applied here,
and `utils/UseInclusionCritera.m` / `RemoveUnwantedSessions.m` are not called by any figure script.

The 5 disagreements in Check E are trials in which the animal's *first* contact was on the
opposite port from the one the Bpod state machine scored; the reference definition
(`getPrevChoice.m`, based on `R/L` x `hit/miss`) is what is exported, and it is what the paper uses.

### Check 3 — Reference code comparison
| Stage | Reference | This script | Same? |
|---|---|---|---|
| (a) data loading | `loadObjs`/`loadSessionData` read `data_structure_<ANM>_<DATE>.mat`; session and ALM probe come from `load<ANM>_ALMVideo.m`; motion energy from `motionEnergy_*.mat` (`loadMotionEnergy.m`) | identical files, identical session/probe list, identical motion-energy file | yes |
| (b) neuron filtering | `findClusters` with `params.quality={'all'}` (drop `garbage`/`gabrga`/`noisy`/`real?`), then `removeLowFRClusters` with `params.lowFR=1` on the condition-averaged PSTH | identical, using the `params.condition` list of `Figure8a_thru_c.m` | yes |
| (b) trial filtering | every `params.condition` contains `~stim.enable` and `~early`; ignore trials are only in the `(hit|miss|no)` condition | `~early & ~stim.enable`, ignore trials **kept** (required by the task), plus the one video-less trial dropped | deliberate deviations, documented |
| (c) temporal alignment | `alignSpikes`: `trialtm - ev.goCue(trial)`; video: `frameTimes - findVideoOffset(obj) - ev.goCue(trial)`; `params.advance_movement = 0` | identical | yes |
| (d) binning | `getSeq`: `edges = tmin:dt:tmax`, `histc`, drop the last bin, `/dt`, `mySmooth(...,15,'reflect')`; `obj.time = edges+dt/2` (drop last) | identical (`np.histogram` over half-open bins; kernel and reflect padding ported 1:1 and verified numerically) | yes |
| (e) input construction | `obj.time` | `obj.time` as the single decoder input | yes (task-specified) |
| (f) output construction | `getPrevChoice`/`getOutcome`/`autowater` for the task variables; `findPosition`+`findVelocity` for kinematics; `loadMotionEnergy` for motion energy | identical formulas; three differences, all forced by the task spec (below) | mostly |

Deliberate differences from the reference, and why:
1. **Ignore trials retained.** The paper excludes them; the decoder task requires `ignore` and
   `none` classes. Without them two of the six outputs would be binary.
2. **Paw "not visible" is kept as a class.** `findPosition.m` nearest-fills every non-tongue
   feature, so the reference has no notion of an invisible paw. The task explicitly asks for
   `2: not visible`, so the pre-fill DeepLabCut NaN mask is used to define it. Velocity itself is
   still computed on the nearest-filled position exactly as the reference does.
3. **`basederiv` applied per axis.** `findVelocity.m` subtracts `basederiv(1)` (the x-axis median
   derivative) from *both* the x and y velocity, which looks like an indexing typo. I subtract the
   median derivative of each axis from that axis. The effect is negligible (the median derivative of
   a tracked feature is ~0) and it cannot change a speed percentile appreciably.
4. **Window `tmin = -2.5` rather than the `-3` used in Figure 8** (rationale in Step 5, decision 4).

### Check 4 — Key statistics comparison
See the table in Step 9. Every statistic that the paper reports for this data set is reproduced:
12 sessions; 521 vs 522 units (the single-unit difference is the `tmin` window edge — with the
Figure-8 window `tmin = -3` the script gives exactly 522); **214 vs 214** single units; sessions
that all begin with ~100 DR trials followed by 8-16 alternating blocks; ALM only. The only
statistic that does not match is the number of mice (7 vs "six"), which is a discrepancy between
the paper text and the authors' own session lists, not between the paper and this conversion
(the same off-by-one exists for the 25 fixed-delay sessions: the paper says nine mice, the
loaders contain ten).

### Check 5 — Edge cases handled
- **Cluster quality strings** come in mixed case and with trailing spaces (`'multi    '`,
  `'Poor'`, `'garbage'`) and one session has clusters whose quality is `char([0 0])`. The filter
  strips whitespace/NUL and matches case-sensitively, exactly like MATLAB `ismember` after
  `strtrim`, which keeps the one `'Noisy'` (capital N) cluster in JEB6 and the four empty-quality
  clusters — reproducing the reference counts.
- **Bin edges**: `histc(...)(1:end-1)` gives half-open bins, so spikes exactly at `t = +2.5` are
  dropped; `np.histogram` would put them in the last bin, so the code masks `aligned < tmax`
  explicitly.
- **Trial indices** in `obj.clu.trial` are 1-based; spikes with out-of-range or NaN trial
  indices are discarded before binning.
- **Cameras have independent frame counts and frame-time vectors**; the first version of the
  script used the side-camera frame times for the bottom-camera features and crashed on JEB6.
  Each view now uses its own `frameTimes`, and the number of frames is clipped to
  `min(len(frameTimes), ts.shape[2])`.
- **Trials whose video is missing** (`frameTimes` empty or NaN): detected and dropped (1 trial).
- **Trials where DeepLabCut never sees a paw/tongue**: `nanmean` over an empty set is avoided by
  an explicit count, and the whole trial simply becomes class 2 for that variable.
- **Sessions without `obj.bp.stim`** (`JEB6`): handled by a `try/except` that falls back to an
  all-false stim mask.
- **Sessions without `obj.ex`** (EKH3, JEB7, JGR2, JGR3): the probe number comes from the
  reference loader scripts rather than the data file.
- **`goCue` NaN**: excluded by the trial mask (none occur in these 12 sessions).

### Issues found and resolved during this step
1. *Bottom-camera resampling crash* (different frame counts per view) — fixed by using each view's
   own `frameTimes`; re-ran conversion, verification and all sanity checks.
2. *Motion energy "no video" defined per time point* — my first version skipped the reference's
   `fillmissing('nearest')` so that the ~0.6% of samples at the trial edges not covered by video
   became class 2. This created a class with ~0.8% of the data, which the decoder's balanced loss
   then over-weighted: 29% of true below-median samples were predicted as "no video" and
   motion-energy validation balanced accuracy fell to 0.585. Restoring the reference behaviour
   (fill within a trial; "no video" only for trials without video, which are dropped) raised it to
   **0.775** with no change to any other output. Documented as Step 5 decision 10.
3. *Session-level provenance* — `trial_ids`, `cluster_ids` and `unit_qualities` were added to
   `metadata['session_info']` so that every exported trial/unit can be traced back to the raw file
   (this is what makes the independent sanity checks possible).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
-> `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.

### Training Progress
Loss decreasing: **Yes** — 7.34 (epoch 1) -> 2.78 (epoch 20) -> 1.20 (epoch 50) -> 0.798 (epoch 100)
-> 0.750 (epoch 200); test loss 0.764. Trained on the GPU (NVIDIA L4).

### Decoder Results (Full, 12 sessions, 3115 trials, 521 units)
| Output | Chance (1/nclass) | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|--------|-------------|--------|-------|
| lick_direction | 0.333 | 0.6128 | **0.5973** | 1.79x chance; 0.72 after the go cue, chance before it |
| context | 0.500 | 0.7604 | **0.7507** | 1.50x chance; sustained across the whole trial, as in the paper |
| outcome | 0.333 | 0.6126 | **0.5791** | 1.74x chance |
| tongue_velocity | 0.333 | 0.5925 | **0.5894** | 1.77x chance |
| paw_velocity | 0.333 | 0.5423 | **0.5072** | 1.52x chance (weakest; see Step 12) |
| motion_energy | 0.333 | 0.7771 | **0.7746** | 2.32x chance (effective chance 0.5 -> 1.55x) |

Train and validation agree to within 0.035 for every output — no overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance
No output is at or below chance. Ratios to uniform chance: 1.79 / 1.50 / 1.74 / 1.77 / 1.52 / 2.32.
The two outputs closest to the 1.5x line were investigated:

**`context` (0.751, chance 0.500, ratio 1.50).** Binary variables can never exceed 2x chance, so
the ratio is not informative here; 0.75 on a balanced binary problem is a strong result and matches
the paper (context is decodable throughout the trial, Fig. 4b).

**`paw_velocity` (0.507, chance 0.333, ratio 1.52).** Three diagnostics:
   1. *Is the paw signal real?* Yes — lag-1 autocorrelation of the paw speed is 0.62-0.72 (not
      tracking noise), and mean paw speed roughly doubles after the go cue (0.55 -> 0.94 px/bin in
      JEB6; 0.42 -> 0.74 in JEB19), the expected task modulation. Its correlation with motion energy
      is low (0.10-0.16), i.e. paw movement is a genuinely different uninstructed movement from the
      tongue/jaw movement that dominates motion energy, so a lower decodability than motion energy
      is expected rather than suspicious.
   2. *Is the rare "not visible" class to blame?* Partly. A control in which the paw position is
      nearest-filled exactly as `findPosition.m` does (which removes the "not visible" class) gives
      a binary balanced accuracy of **0.592 against a chance of 0.500 (1.18x)**, versus 0.515
      against 0.333 (1.55x) for the 3-class version on the same two sessions. So the 3-class
      version is, if anything, *further* above its chance level; the underlying binary problem is
      simply hard. The "not visible" class is kept because the decoder task explicitly defines it
      and 1.5% of time points genuinely have no DeepLabCut paw detection (7.6% in EKH1).
   3. *Time course.* Paw decoding peaks at 0.70 just after the go cue and is near chance during
      the ITI — i.e. the decoder finds paw movement when paw movement actually happens.

### Check 2 — Accuracy comparison to the paper
The paper reports decoding accuracies only as curves (Fig. 3b, Fig. 4b, Extended Data Fig. 2c), and
its decoders differ from `train_decoder.py` in three ways that make the numbers not directly
comparable: they are (i) **per session**, (ii) **binary and class-balanced on correct trials only**,
and (iii) fit on **75 ms** bins. To make a real comparison I therefore re-ran *the paper's own
analysis* (`NeuralChoiceDecoding.m` / `NeuralContextDecoding.m`, Methods "Choice and context
decoding from neural population") **on the converted data** — per session, 75 ms bins, ridge
logistic regression, 4-fold CV, equal numbers of left/right (or DR/WC) correct trials
(`/app/cache/paper_style_decoding.py`):

| Quantity | Paper | Converted data (paper-style analysis) | `train_decoder.py` on converted data |
|---|---|---|---|
| Neural **choice** decoding, before the sample tone | ~chance (0.5) | 0.62 at -2.0 s | — |
| Neural **choice** decoding, end of delay | rising through the delay; CD~choice~ AUC 0.86 ± 0.11 | 0.79 at -0.23 s | — |
| Neural **choice** decoding, after the go cue | peaks near 1.0 (Fig. 3b) | **0.96** at +0.33 s | 0.72 at +0.5 s (3-class, incl. `none`) |
| Neural **context** decoding, during the ITI | well above chance (Fig. 4b) | **0.85-0.87** | 0.78-0.80 |
| Neural **context** decoding, at the go cue | peak | **0.97** | 0.84 |
| Neural **context** decoding, mean over trial | sustained | 0.81 | 0.751 |

The converted data therefore reproduce the published decoding curves closely — choice at chance
early, rising through the delay, peaking at 0.96 after the go cue and decaying; context high and
sustained from the ITI onwards with a peak at the go cue. This is strong evidence that the neural
data, the labels and the temporal alignment are all correct, and that the lower numbers from
`train_decoder.py` come from its architecture (**one** decoder shared across 12 sessions with
different neural populations, three classes including `none`/error trials, a memoryless
per-10 ms-timepoint linear readout) rather than from the conversion.

Time-resolved balanced accuracy of `train_decoder.py` (held-out trials) also shows exactly the
expected structure:

| t (s) | -2.0 | -1.5 | -1.0 | -0.5 | -0.2 | +0.1 | +0.3 | +0.5 | +1.0 | +1.5 | +2.0 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lick_direction | 0.505 | 0.517 | 0.567 | 0.583 | 0.638 | 0.590 | 0.692 | 0.721 | 0.692 | 0.647 | 0.603 | 0.597 |
| context | 0.776 | 0.800 | 0.773 | 0.801 | 0.801 | 0.838 | 0.812 | 0.782 | 0.707 | 0.712 | 0.692 | 0.758 |
| outcome | 0.549 | 0.544 | 0.572 | 0.550 | 0.599 | 0.547 | 0.606 | 0.681 | 0.663 | 0.660 | 0.635 | 0.595 |
| tongue_velocity | 0.728 | 0.594 | 0.700 | 0.509 | 0.797 | 0.487 | 0.558 | 0.486 | 0.474 | 0.445 | 0.595 | 0.578 |
| paw_velocity | 0.534 | 0.568 | 0.575 | 0.613 | 0.625 | 0.699 | 0.609 | 0.649 | 0.605 | 0.492 | 0.537 | 0.556 |
| motion_energy | 0.734 | 0.705 | 0.715 | 0.721 | 0.736 | 0.781 | 0.818 | 0.830 | 0.819 | 0.738 | 0.694 | 0.744 |

`lick_direction` and `outcome` start at chance 2 s before the go cue (the animal's choice is not
yet determined, and in WC trials the rewarded port is chosen at random at water delivery) and rise
to ~0.72/0.68 around the response epoch — precisely the pattern the paper reports. Averaging over
the whole 5 s window, as `train_decoder.py` does, necessarily dilutes the post-go-cue values; this
is a property of the task specification (decode at every time point), not a conversion problem.
(The tongue numbers early in the trial are noisy because only ~5% of pre-go-cue time points are
"visible", so each time point's balanced accuracy averages over very few class-0/1 samples.)

### Check 3 — Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| lick_direction | 0.6128 | 0.5973 | 1.03 |
| context | 0.7604 | 0.7507 | 1.01 |
| outcome | 0.6126 | 0.5791 | 1.06 |
| tongue_velocity | 0.5925 | 0.5894 | 1.01 |
| paw_velocity | 0.5423 | 0.5072 | 1.07 |
| motion_energy | 0.7771 | 0.7746 | 1.00 |

All ratios are <= 1.07, far below the 1.5 threshold: no overfitting and no data leakage. (The split
is per trial within each session, and the per-trial outputs are constant within a trial, so there is
no within-trial leakage across the split either.)

### Additional targeted debugging performed
1. **Output values verified on specific trials** — Check D/E/F of the sanity-check script
   re-derive every output for randomly chosen trials directly from the raw `.mat` and compare
   bin-for-bin; all match.
2. **Temporal alignment plotted for single trials** — `processing_<session>.png` panels 1, 5 and 7
   show spikes, lickport contacts and DeepLabCut traces on the same go-cue-aligned axis; licking
   begins at t = 0 and the resampled video overlays the raw 400 Hz trace.
3. **Class variation** — no output is dominated by a single class beyond what the data dictate;
   the most extreme is `tongue_velocity` at 92% "not visible", which is the true fraction of time
   the tongue is inside the mouth.
4. **Neural filtering re-verified** against the paper's unit counts (521/214 vs 522/214).
5. **Processing re-verified** against the reference code function by function (Step 10, Check 3).

### Issues found and resolved
- Motion-energy "no video" definition (see Step 10, issue 2): validation balanced accuracy for
  `motion_energy` improved from 0.585 to 0.775 after restoring the reference's
  `fillmissing('nearest')` and dropping the single video-less trial.
- No other issue was found that changed the data.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format spec, key statistics)
- [x] `cache/` folder holds the exploration and validation scripts, with `cache/README_CACHE.md`
- [x] All required deliverables present: `CONVERSION_NOTES.md`, `convert_data.py`,
      `converted_data.pkl`, `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`,
      `verification_sample_out.txt`, `train_decoder_sample_out.txt`, `conversion_full_out.txt`,
      `verification_full_out.txt`, `train_decoder_full_out.txt`, plus the
      `processing_<session>.png` diagnostics and `sample_trials.png` / `predictions.png`.
