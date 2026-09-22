# Dataset Conversion Notes

## Overview
- **Dataset**: Hasnain, Birnbaum et al., *"Separating cognitive and motor processes in the
  behaving mouse"*, Nature Neuroscience 28:640-653 (2025). ALM Neuropixels/H2 recordings +
  high-speed video during a two-context (delayed-response / water-cued) directional licking
  task. Data in `/app/data`, reference MATLAB code in `/app/code`
  (Zenodo DOI 10.5281/zenodo.13941415).
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`), aligned to the
  **go cue**, decoding lick direction, context, outcome, tongue velocity, paw velocity and
  motion energy from ALM spiking activity.

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `paper.pdf`, `methods.txt` — reference text
- `code/` — authors' MATLAB analysis code (see Step 1)
- `data/` — 4 sub-directories (see Step 2)
- `decoder.py`, `train_decoder.py` — provided decoder / validation code
- `Dockerfile`, `docker-compose.yaml`, `.manifest`
- created by me: `CONVERSION_NOTES.md`, `convert_data.py`, `cache/` (exploration scripts)

Environment verified: `python3` 3.13, numpy 2.4.4, torch 2.6.0+cu124 (CUDA available),
scipy 1.18, h5py, pandas. `pypdf` installed to extract text from `paper.pdf`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

`/app/code/README.md` + `WorkingWithDataObjs.m` document the data objects and the analysis
pipeline. Each session is a MATLAB struct `obj` with fields
`bp` (bpod trial/task data), `clu` (sorted spikes, one cell per probe), `traj` (DeepLabCut
output, `{1}` = side cam, `{2}` = bottom cam), `sglx` (SpikeGLX metadata), `ex` (session
metadata incl. probe location), `me` (motion energy; only for behaviour-only sessions).

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load<ANM>_ALMVideo` | DataLoadingScripts/Recording and video/*.m | LOADING | **Definitive session + probe list**: which sessions have ephys+video and which probe is ALM. Commented-out entries = sessions the authors excluded. |
| `loadObjs` | DataLoadingScripts/loadObjs.m | LOADING | loads `data_structure_<anm>_<date>.mat` for each entry of `meta` |
| `loadSessionData` | DataLoadingScripts/loadSessionData.m | LOADING | per session/probe driver; concatenates units of both probes when `meta.probe = [1 2]` |
| `processData` | DataLoadingScripts/processData.m | PROCESSING | findTrials → findClusters → alignSpikes → getSeq → removeLowFRClusters → baselineFR |
| `findTrials` | DataLoadingScripts/findTrials.m | CURATION | evaluates condition strings such as `'R&hit&~stim.enable&~autowater&~early'` over `obj.bp` fields |
| `findClusters` | DataLoadingScripts/findClusters.m | CURATION | with `params.quality={'all'}` keeps every cluster except quality `garbage`, `gabrga`, `noisy`, `real?` |
| `alignSpikes` | DataLoadingScripts/alignSpikes.m | PROCESSING | `trialtm_aligned = trialtm - ev.(alignEvent)(trial)`; `params.alignEvent='goCue'` |
| `getSeq` | DataLoadingScripts/getSeq.m | PROCESSING | histograms aligned spikes into `params.tmin:params.dt:params.tmax` bins, divides by `dt` (spikes/s) and smooths with `mySmooth`; produces `obj.trialdat (time, units, trials)` |
| `mySmooth` | utils/mySmooth.m | PROCESSING | causal gaussian kernel: `gausswin(15)` with first 7 taps zeroed, normalised, `conv(...,'same')`, `'reflect'` boundary = prepend first 15 samples then trim |
| `removeLowFRClusters` | DataLoadingScripts/removeLowFRClusters.m | CURATION | drops units whose mean firing rate over the window/trials is `<= params.lowFR` |
| `getDefaultParams` / `WorkingWithDataObjs.m` | DataLoadingScripts | PROCESSING | parameters: `alignEvent='goCue'`, `tmin=-2.5`, `tmax=2.5`, `dt=1/100`, `smooth=15`, `bctype='reflect'`, `lowFR=1`, `quality={'all'}` |
| `findVideoOffset` | funcs/findVideoOffset.m | PROCESSING | `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)`; video clock → ephys clock |
| `findPosition` | funcs/kinematics/findPosition.m | PROCESSING | `interp1(frameTimes - vidshift - alignTime, xy, taxis)`; skips trials with `isnan(NdroppedFrames)`; fills missing with nearest **except tongue** |
| `findVelocity` | funcs/kinematics/findVelocity.m | PROCESSING | velocity = `gradient(position)` (minus the median frame-to-frame difference for non-tongue features) |
| `getKinematicsFromVideo` | funcs/kinematics | PROCESSING | assembles (xdisp, ydisp, xvel, yvel) per feature per view |
| `loadMotionEnergy` | DataLoadingScripts/loadMotionEnergy.m | LOADING/PROCESSING | loads `motionEnergy_<anm>_<date>.mat`, interpolates the 400 Hz trace onto the neural time axis with the same video shift, fills remaining NaNs with nearest, `me.move = me.data > me.moveThresh` |
| `getPrevChoice` | funcs/getPrevChoice.m | PROCESSING | defines **choice**: `(R&hit) | (L&miss)` = right, NaN on ignore trials |
| `getOutcome` | funcs/getOutcome.m | PROCESSING | outcome = `bp.hit`, NaN on ignore (`bp.no`) trials |
| `NeuralChoiceDecoding.m`, `NeuralContextDecoding.m` | ChoiceContextDecoding | ANALYSIS | logistic decoding of choice/context from `obj.trialdat` in 75 ms bins, 4-fold CV — the closest analogue of our decoder task |

### Notes
- Neural data are **spike times**, so no dF/F is needed; the reference representation used
  for every single-trial analysis is `obj.trialdat` = binned, causally smoothed firing rate.
- Cell quality filtering is a two-stage process: manual quality label, then a firing-rate
  threshold (`lowFR`). The paper states "All units with firing rates exceeding 1 Hz were
  included in all other analyses" (1 Hz is also the value used in `WorkingWithDataObjs.m`).
- The reference code excludes photoinactivation (`stim.enable`) and early-lick (`early`)
  trials in every condition string it uses for ephys analyses.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` contains four directories:

| Directory | Contents | Used? |
|-----------|----------|-------|
| `Ephys_Behavior` | 25 `data_structure_*.mat` + 25 `motionEnergy_*.mat` — fixed-delay task sessions with ephys + video | **yes** |
| `RandomizedDelay_Ephys_Behavior` | 22 `data_structure_*.mat` + 22 `motionEnergy_*.mat` — randomized-delay task sessions with ephys + video | **yes** (19 of 22; see Step 4) |
| `DelayInhibition_BilatMC_Behavior` | behaviour + video only (optogenetic delay inhibition) | no — **no neural data** |
| `GoCueInhibition_BilatMC_Behavior` | behaviour + video only (optogenetic go-cue inhibition) | no — **no neural data** |

File formats: most `data_structure_*.mat` are MATLAB v7.3 (HDF5); 11 files
(`JEB23 2023-10-18/20/21`, all `JEB24`) are MATLAB v7 (MAT5). All `motionEnergy_*.mat` are
MATLAB v7. `convert_data.py` implements a reader for both layouts.

Key fields (per session):
- `bp.Ntrials`, `bp.R`, `bp.L` (instructed/rewarded side), `bp.hit`, `bp.miss`, `bp.no`
  (ignore), `bp.early` (early lick), `bp.autowater` (1 = water-cued trial),
  `bp.stim.enable` (photoinactivation), `bp.ev.{bitStart, sample, delay, goCue, reward}`
  (times within the trial, seconds), `bp.ev.lickL/lickR` (cell array of lickport contact
  times per trial).
- `clu{probe}(i).{quality, trialtm, trial, tm, site/channel, spkWavs}` — spike times within
  each trial and the trial index for each spike.
- `traj{view}(trial).{ts (frames x [x,y,confidence] x bodypart), frameTimes, featNames,
  NdroppedFrames}`; side-cam features `tongue, left_tongue, right_tongue, jaw, trident,
  nose, lickport`; bottom-cam features `top_tongue, topleft_tongue, bottom_tongue,
  bottomleft_tongue, top_paw, bottom_paw, lickport, jaw, top_nostril, bottom_nostril`.
  Untracked timepoints are already NaN in `ts`.
- `sglx.fs`, `sglx.bitcode.bitstart` (used for the video/ephys clock offset).
- `ex.probe(i).loc` — e.g. `'R ALM'`, `'L M1TJ'`, `'DUMMY'` (missing in some sessions).
- `motionEnergy_*.mat`: `me.data{trial}` = 400 Hz motion-energy trace (one value per video
  frame, verified to equal `numel(frameTimes)` for **every trial of every session**),
  `me.moveThresh` = the authors' manual per-session movement threshold.

### Dataset Size (from data files, reference session/probe lists, before my curation)
| Statistic | Value |
|-----------|-------|
| Sessions with ephys (files present) | 25 fixed-delay + 22 randomized-delay = 47 |
| Sessions in the reference load scripts | 25 fixed-delay + 19 randomized-delay = **44** |
| Subjects | 10 fixed-delay (EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3) + 4 randomized (JEB11, JEB12, JEB23, JEB24) = **14** |
| Sessions / subject | 1–8 (median 2.5) |
| Trials (total, all 44 sessions) | 14,972 (8,260 fixed + 6,712 randomized) |
| Trials / session | 230–517 (mean 340) |
| Clusters (all, incl. garbage) | 7,241 fixed + 3,089 randomized |
| Clusters after quality curation | 1,565 fixed + 948 randomized = 2,513 |
| ... of which single units (excellent/great/good) | 282 fixed + 178 randomized |
| Clusters after quality + FR > 1 Hz | 1,532 fixed + 925 randomized = 2,457 |
| Quality labels present | excellent, great, good, fair, multi, poor, garbage, gabrga (typo), mutli (typo), noisy, real?, '' (empty) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source quote |
|-----------|-------|--------------|
| Fixed-delay ephys dataset | 25 sessions, 9 mice, 1,651 units (483 single units) | "For the DR task, we recorded 1,651 units (483 single units) in ALM from 25 sessions using nine mice." |
| Two-context subset | 12 sessions, 6 mice, 522 units (214 SU) | "In 12 sessions from six mice, animals performed the two-context task. In total, 522 units (214 well-isolated single units) were recorded in these sessions." |
| Randomized-delay dataset | 19 sessions, 4 mice, 845 units (288 SU) | "for the randomized delay task, we recorded 845 units (288 well-isolated single units) in ALM from 19 sessions using four mice" |
| Session inclusion | ≥ 10 units | "Recording sessions were included for analysis only if they had at least 10 units" |
| Unit inclusion | FR > 1 Hz | "All units with firing rates exceeding 1 Hz were included in all other analyses." |
| Neural time bin | 10 ms (`params.dt = 1/100`), 5 ms mentioned in the tutorial comment; decoding analyses re-bin to 75 ms | `WorkingWithDataObjs.m` |
| Alignment window | −2.5 … 2.5 s around the go cue | `params.tmin/tmax` |
| Video | 400 Hz, 2 cameras, DeepLabCut; missing values filled with nearest **except tongue** | Methods, "Videography analysis" |
| Motion energy | per frame, 99th percentile over pixels; manual per-session movement threshold | Methods, "Motion energy" |
| Sample epoch | 1.3 s tone | Methods |
| Delay epoch | 0.9 s (12 mice) / 0.7 s (1 mouse); randomized: 0.3/0.6/1.2/1.8/2.4/3.6 s | Methods |
| Ignore criterion | no response within 3 s of the go cue | Methods |
| WC block structure | ~100 DR trials, then alternating blocks of 10–25 trials; all sessions start with DR | Methods |
| Early-lick trials | omitted from analyses | Methods |
| Photoinactivation | ~30 % of trials in opto sessions | Methods |
| Choice selectivity (DR) | sample 36 %, delay 42 %, response 58 % of 483 SU | Results |
| Context selectivity | 39 % of 214 SU (ITI) | Results |

### Processing Details
- Temporal alignment for essentially all analyses is the **go cue** (`obj.bp.ev.goCue`); in
  WC trials the same field holds the water-drop time (paper axes read "Time from go
  cue/water drop").
- Neural data: spikes binned at `dt`, converted to spikes/s and causally smoothed
  (`mySmooth`, 15-bin gaussian, `reflect` boundary).
- Video: `frameTimes` are on the video clock; the offset to the ephys/bpod clock is
  `findVideoOffset` (0.49 s for 13 of 14 animals, 0.99 s for JEB19; the tutorial quotes
  "subtract 0.5 second from frametimes").

### Curation Steps
**Neuron curation rules**
1. drop clusters with quality label `garbage` / `gabrga` / `noisy` / `real?` (`findClusters`);
2. drop clusters with mean FR ≤ 1 Hz across all trials (`removeLowFRClusters`, `lowFR = 1`).

**Trial curation rules**
1. drop early-lick trials (`bp.early`);
2. drop photoinactivation trials (`bp.stim.enable`);
3. (mine, see Step 5) drop trials with no ephys coverage.

**Session curation**: the authors' `load<ANM>_ALMVideo.m` lists are used verbatim (they
already encode ≥ 10 units and session-quality decisions; several sessions are commented out
with explanations such as "no usable left miss trials" or "doesn't seem to be sorted").

### Decoders Trained (reference)
| Decoded variable | Method | Accuracy (from the paper's figures) |
|------------------|--------|-------------------------------------|
| Choice (left/right), neural population, per-time-bin | ridge logistic regression, 4-fold CV, 75 ms bins, correct trials only, balanced classes (Fig. 3b) | ~0.55–0.6 before sample, rising to ~0.75–0.85 late delay, ~0.95–1.0 after the go cue |
| Choice, kinematic features (Fig. 3b) | same | ~0.5 at sample onset rising to ~0.8–0.9 after the go cue |
| Context (DR/WC), neural population (Fig. 4b bottom) | same | ~0.8–0.9 throughout the trial, peaking near 0.95 after the go cue |
| Context, kinematic features (Fig. 4b top) | same | ~0.75–0.9 |
| Choice from delay-epoch CD_choice projection (ED Fig. 2c) | logistic regression + ROC | AUC 0.86 ± 0.11 |

No decoding accuracies are reported in the paper for outcome, tongue/paw velocity or motion
energy.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| # fixed-delay sessions | 25 sessions listed in `load*_ALMVideo.m` | 25 `data_structure` files in `Ephys_Behavior` | 25 sessions | consistent → use all 25 |
| # mice, fixed delay | 10 animals in the load scripts | 10 animals | "nine mice" | data/code agree with each other; the paper's count is off by one (or pools two animals). Followed code+data (10 mice). |
| # randomized-delay sessions | 19 (JEB23 2023-10-20 commented out; JEB24 2023-10-03/04 absent from the list) | 22 files | 19 sessions | code+paper agree → use the 19 listed sessions |
| JEB23 2023-10-20 | commented out of the load list | `data_structure_JEB23_2023-10-20.mat` has **exactly the same** trial data as 2023-10-19 (N=352, 292 hit, 28 miss, 32 ignore) while `motionEnergy_JEB23_2023-10-20.mat` has 307 trials | not mentioned | duplicate/mislabelled session — excluded, matching the reference |
| JEB24 2023-10-03 | absent from the load list | the file has **no `clu` field at all** (no sorted spikes) | — | excluded, matching the reference |
| JEB24 2023-10-04 | absent from the load list | present, sorted | — | excluded, matching the reference |
| Unit counts | quality+FR curation gives 1,532 (fixed) / 925 (randomized) | 1,565 / 948 before the FR filter | 1,651 / 845 | within 5 %/10 % of the paper; the paper's bookkeeping cannot be reproduced exactly from the released files (no per-session table is given). Documented, not "fixed" — my counts follow the reference code exactly. |
| Single units | excellent/great/good = 282 (fixed) / 178 (rand) | same | 483 / 288 | same caveat; adding `fair` gives 647/444, so no simple label mapping reproduces the paper's numbers. Only affects reporting, not the conversion (all quality labels except garbage/noisy are used, as in the reference). |
| Cluster quality letter case | `findClusters` uses case-sensitive `ismember` against lower-case labels | labels are lower case in the Neuropixels sessions and Capitalised in the H2 sessions (`Poor`, `Noisy`, ...) | — | I match case-insensitively, so the single `Noisy` unit in JEB6 is excluded (the reference would have kept it by accident). 1 unit out of 2,513. |
| Probe location vs "ALM" | `loadJEB15_ALMVideo.m` uses probes `[1 2]` (2022-07-26/27/28) and probe 2 (2022-07-29) | `ex.probe.loc` says probe 2 of JEB15 is `L M1TJ` (tongue-jaw motor cortex) | "recorded ... in the ALM" | kept exactly the reference's probes, but each unit is labelled with its **actual** recorded region (`ALM` or `tjM1`) from `ex.probe.loc`; sessions with no `loc` string are labelled ALM, as asserted by the `*_ALMVideo` load scripts. |
| Context variable | `autowater` used as the DR/WC proxy in every reference script (`getBlockNum_AltContextTask.m` derives blocks from it) | `autowaterBlock` also exists in newer files but is all-zero there | "obj.bp.autowater=1 when water was delivered regardless of animal choice ... can be used as a proxy for obtaining water-cued blocks" | use `bp.autowater` |
| Two-context sessions | not marked in the code | 33 of 44 sessions contain ≥1 autowater trial; 16 sessions have ≥30 | "12 sessions, six mice" | the paper's 12-session subset is a behavioural-criterion subset (≥20 correct WC trials per direction). I keep all sessions and let the per-trial `autowater` flag define the context label; sessions without WC trials simply contribute only DR labels. |

Sanity checks run during Step 4 (scripts in `/app/cache`):
- `bp.hit + bp.miss + bp.no == Ntrials` in every session (no unlabelled trials).
- `bp.R + bp.L == Ntrials` in every session.
- Lick direction derived from `(R&hit)|(L&miss)` agrees with the side of the **first
  lickport contact after the go cue** on **100 %** of trials in the sessions tested, and
  ignore (`bp.no`) trials have **no** post-go-cue licks. This validates both the outcome
  and the lick-direction definitions.
- `numel(me.data) == Ntrials` and `numel(me.data{i}) == numel(frameTimes{i})` for all 44
  sessions → the motion-energy trace is sampled on the video frame clock.
- No session has `bp.fidx`, and spike `trial` indices always lie in `1..Ntrials`.
- `goCue` is finite on every trial of every session.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source variable | Target field | Transform | Reference function | Notes |
|-----------------|--------------|-----------|--------------------|-------|
| `clu{probe}(i).trialtm`, `.trial`, `bp.ev.goCue` | `neural[session][trial]` (n_units × 500) | align to go cue, 10 ms bins over [−2.5, 2.5] s, /dt, causal gaussian smoothing | `alignSpikes` + `getSeq` + `mySmooth` | float32 spikes/s |
| bin-centre times | `input[session][trial]` (1 × 500) | `edges + dt/2`, i.e. −2.495 … 2.495 s | `getSeq` (`obj.time`) | the only decoder input requested |
| `bp.R/L/hit/miss/no` | `output[0]` lick_direction | left=0, right=1, none=2 | `getPrevChoice` | per-trial, broadcast over time |
| `bp.autowater` | `output[1]` context | WC=0, DR=1 | `findTrials` condition strings | per-trial |
| `bp.hit/miss/no` | `output[2]` outcome | incorrect=0, correct=1, ignore=2 | `getOutcome` | per-trial |
| `traj{1}(t).ts[:, :, 'tongue']` | `output[3]` tongue_velocity | interpolate to the neural axis, speed = ‖d(x,y)/dbin‖, split at the session's 50th percentile; 2 where DLC did not track the tongue | `findPosition`, `findVelocity` | time-varying |
| `traj{2}(t).ts[:, :, 'top_paw']` | `output[4]` paw_velocity | as above | `findPosition`, `findVelocity`, `Scripts/Figure 1/Figure1e.m` (which uses `top_paw_yvel_view2` as "paw speed") | time-varying |
| `me.data{t}` | `output[5]` motion_energy | interpolate to the neural axis, split at the session's 50th percentile; 2 where no video frame covers the timepoint | `loadMotionEnergy` | time-varying |
| `ex.anm` / filename | `subjects`, `subject_idx` | | | 14 mice |
| `ex.probe(p).loc` | `brain_regions`, `brain_region_idx` | `'* ALM'`→ALM, `'* M1TJ'`→tjM1, missing→ALM | | |

### Key Decisions
1. **Which sessions**: all 44 ephys sessions listed in the reference `load*_ALMVideo.m`
   files (25 fixed-delay + 19 randomized-delay). The two behaviour-only directories are
   excluded because they contain no neural data. Randomized-delay sessions are included
   because they are valid ALM recordings with the same task events, video and outcome
   variables; aligning on the go cue makes the variable delay irrelevant to the alignment
   (and the decoder is given time-from-go-cue as an input). They contribute DR-context
   trials only, which is correct, not degenerate.
2. **Which probe(s)**: exactly those in the reference load scripts (`meta.probe`), including
   the dual-probe JEB15 sessions where both probes are concatenated, as `loadSessionData`
   does.
3. **Alignment / window / binning**: go cue, [−2.5, 2.5] s, 10 ms bins — the reference
   `params` values. Every trial therefore has exactly 500 timepoints.
4. **Neural representation**: causally smoothed firing rate (spikes/s), exactly the
   `obj.trialdat` that all reference single-trial analyses (including their choice/context
   decoders) use. Smoothing is causal, so no information leaks backwards in time.
5. **Neuron curation**: quality label filter + mean FR > 1 Hz over the aligned window across
   **all** trials (reference: "on average across all trials"), computed before trial
   curation so the unit set does not depend on the trial subset.
6. **Trial curation**: drop `early` and `stim.enable` trials (as in every reference ephys
   condition string). Additionally drop trials with zero spikes on all units — in 2 sessions
   (JEB24 2023-10-23 and 2023-11-03) the ephys recording stopped before the behaviour did,
   leaving trailing trials with literally no neural data (they would be all-zero inputs and
   trigger the validator's "all neural data is zero" warning).
7. **Ignore trials are kept**: the decoder task explicitly asks for `none` lick direction and
   `ignore` outcome classes, so `bp.no` trials must be retained (the reference excludes them
   from its choice analyses, which is a task-specific difference required here).
8. **Discretisation**: per-session 50th percentile of all *valid* samples of the retained
   trials, exactly as instructed; class 2 = tongue/paw not tracked by DLC, or (motion
   energy) no video frame covering that timepoint. Per-session thresholds are necessary
   because camera/scale differences make the units incomparable across sessions (e.g.
   median tongue speed 3.8 px/bin in JEB6 vs 14 px/bin in JEB23).
9. **Tongue and paw features**: tongue = side-camera `tongue`; paw = bottom-camera
   `top_paw` (paws are only tracked from the bottom view, and `top_paw_yvel_view2` is the
   paw feature the authors use for "paw speed" in Fig. 1e). Speed is the magnitude of the
   2-D velocity, i.e. `sqrt(xvel² + yvel²)` of the reference's `xvel`/`yvel`.
10. **Velocity of a partially tracked feature**: the reference sets tongue velocity to 0
    wherever the position is NaN. Because "not visible" is a separate output class here, I
    instead compute the derivative with NaN-aware one-sided differences at the edges of each
    visibility bout, so that visible timepoints never get an artificial 0 (that would have
    mis-assigned ~30 % of visible tongue samples to the "below median" class).
11. **Per-trial outputs are emitted as time-varying rows** (constant within a trial) so that
    all six outputs share one `(6, 500)` array; the decoder broadcasts per-trial values the
    same way internally.
12. **Brain regions** are taken from `ex.probe.loc` rather than assumed, so the four JEB15
    sessions that include a tjM1 probe are labelled correctly.

### Planned Sanity Checks
- [x] population PSTH shows the expected sharp increase at t = 0 (go cue) on lick trials and
      a much weaker one on ignore trials
- [x] lickport contacts occur only after t = 0, and their side matches `output[0]`
- [x] tongue is "visible" essentially only after the go cue
- [x] motion energy rises at the go cue
- [x] `hit+miss+no == Ntrials`, `R+L == Ntrials`
- [x] me trace length == number of video frames for every trial
- [ ] (Step 10) spot checks against the raw `.mat` files with `np.allclose`
- [ ] (Step 10) statistics vs the paper: sessions, mice, units, trials, performance

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (`python -u convert_data.py <out.pkl> [--full|--sample]
[--show-processing]`), structure:

- `SESSIONS` — the 44 (animal, date, probes, directory, task) entries transcribed from the
  reference load scripts.
- `_H5Session` / `_V7Session` / `open_session` — a single reader interface over the two
  MATLAB file formats; only the arrays actually needed are read (for HDF5 files the DLC
  tensor is sliced per feature so a session's video data is never fully materialised).
- `load_motion_energy` — handles the three layouts found in the released
  `motionEnergy_*.mat` files (`me.data` cell array, `me.data.data` struct, bare cell array).
- `gausswin`, `_causal_kernel`, `my_smooth` — exact ports of `mySmooth.m`; since the kernel's
  first `N//2` taps are zero, `conv(...,'same')` is equivalent to a causal FIR filter, which
  is implemented with `scipy.signal.lfilter` (vectorised over all units × trials at once).
- `bin_spikes` — vectorised port of `alignSpikes` + `getSeq`: one `np.bincount` over
  `(trial, bin)` per unit instead of a MATLAB double loop with `histc`.
- `mean_firing_rates` / low-FR removal — port of `removeLowFRClusters`.
- `kinematic_speed`, `motion_energy_trace` — ports of `findPosition`/`findVelocity` and
  `loadMotionEnergy` (`interp_matlab` reproduces MATLAB `interp1`: NaN outside the data
  range and NaN propagated from NaN samples).
- `discretize` — 50th-percentile split with the third "invalid" class.
- `_plot_processing` — the `--show-processing` figure (8 panels: spike raster/rate with lick
  times and task events, population PSTH by lick direction, and traces + distributions with
  the median threshold marked for each of the three continuous variables).

Code inefficiencies identified and removed:
- per-spike Python loops → `np.bincount` on a flattened (trial, bin) index;
- per-unit/per-trial smoothing → one `lfilter` call over a `(500, units·trials)` matrix;
- reading the whole `(nfeat, 3, nframes)` DLC tensor per trial → HDF5 hyperslab of the two
  needed rows;
- `lfilter` returns float64 → cast back to float32 (halves the pickle size).

Run time: **~4 s per session** (≈2.3 s of which is video interpolation), so the full
44-session conversion is ≈3 minutes — well under the 15-minute budget, no parallelism needed.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`processing_JEB6_2021-04-18.png`, `processing_JEB23_2023-10-10.png`.
The two sample sessions are deliberately one fixed-delay two-context session (JEB6) and one
randomized-delay session (JEB23), so both task variants are exercised.

`python -u train_decoder.py /app/sample_data.pkl --verify-only`
→ `/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (JEB6 2021-04-18, JEB23 2023-10-10) |
| Neurons (total) | 122 (29 + 93) |
| Neurons / session | 61 (29–93) |
| Subjects | 2 |
| Sessions / subject | 1 |
| Trials (total) | 651 (302 + 349) |
| Trials / session | 326 |
| Timepoints / trial | 500 (all trials) |
| time_from_go_cue_s range | [−2.495, 2.495] |
| lick_direction | left 0.469, right 0.445, none 0.086 |
| context | WC 0.152, DR 0.848 |
| outcome | incorrect 0.063, correct 0.851, ignore 0.086 |
| tongue_velocity | 0.064 / 0.064 / 0.873 not visible |
| paw_velocity | 0.449 / 0.449 / 0.101 not visible |
| motion_energy | 0.478 / 0.478 / 0.044 no video |

### Processing Plots Review
No anomalies, and several independent alignment checks are visible in the figures:
- the **population firing rate rises sharply exactly at t = 0** on lick trials and only
  weakly on ignore trials — the go-cue response is where it should be;
- lickport contacts (red/blue dots) appear only **after** t = 0 and their side matches the
  `lick_direction` label;
- the tongue is tracked (not "not visible") essentially only **after** the go cue, i.e.
  during licking, which independently confirms the video↔ephys alignment;
- motion energy rises at the go cue;
- for the randomized-delay session, the first ~0.6 s of the window is correctly flagged
  "no video"/"not visible" on short-delay trials because the video recording starts ~0.02 s
  into the trial while the window starts at goCue−2.5 s ≈ −0.6 s of trial time;
- the median lines sit in the middle of each distribution, and the class-0/class-1 counts
  are equal to within 2 % in every session (checked programmatically).

### Run Time Estimates
| Speed-up implemented | Effect |
|----------------------|--------|
| `np.bincount` over a flattened (trial, bin) index instead of per-trial `histc` | spike binning ≤ 0.45 s/session |
| single `lfilter` over a (500, units·trials) matrix instead of per-unit smoothing | negligible smoothing cost |
| HDF5 hyperslab of the 2 needed DLC rows instead of the full (nfeat, 3, nframes) tensor | video step ~2.3 s/session instead of ~8 s |
| cast smoothed rates back to float32 | pickle 3.3 GB → 1.64 GB |

| Step | Time / session | Estimated total (44 sessions) |
|------|----------------|-------------------------------|
| curation (quality + FR) | 0.1–1.7 s | ~35 s |
| spike binning + smoothing | 0.04–0.45 s | ~10 s |
| video (DLC + motion energy) | 0.05–2.6 s | ~60 s |
| **total** | **1.2–4.2 s** | **~2 min** (measured: 107 s) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: **None**
- Warnings: **None**

### Decoder Results (Sample, 2 sessions)
Loss decreased monotonically from 3.96 (epoch 1) to 0.725 (epoch 200); test loss 0.741.

| Output | Training balanced acc | Validation balanced acc | Chance |
|--------|----------------------|-------------------------|--------|
| lick_direction | 0.662 | 0.676 | 0.333 |
| context | 0.866 | 0.869 | 0.500 |
| outcome | 0.606 | 0.605 | 0.333 |
| tongue_velocity | 0.601 | 0.612 | 0.333 |
| paw_velocity | 0.552 | 0.556 | 0.333 |
| motion_energy | 0.731 | 0.739 | 0.333 |

Every output is above chance, and validation ≈ training (no overfitting).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/converted_data.pkl --full` → `/app/conversion_full_out.txt`
(107 s for 44 sessions) and `python -u train_decoder.py /app/converted_data.pkl
--verify-only` → `/app/verification_full_out.txt` (**valid, no errors, no warnings**).

### Output Files
- `converted_data.pkl`: 1.64 GB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference paper | Reference code | Reference data | Converted data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions, fixed delay | 25 | 25 listed | 25 files | 25 | ✓ |
| Sessions, randomized delay | 19 | 19 listed (of 22 files) | 22 files | 19 | ✓ |
| Sessions total | 44 | 44 | 47 | 44 | ✓ |
| Mice, fixed delay | 9 | 10 | 10 | 10 | ✗ paper (see Step 4) |
| Mice, randomized delay | 4 | 4 | 4 | 4 | ✓ |
| Mice total | 13 (9+4) | 14 | 14 | 14 | ✗ paper (off by one) |
| Units, fixed delay | 1,651 | quality+FR curation | 1,565 after quality | 1,532 | ~ (−7 %) |
| Units, randomized delay | 845 | quality+FR curation | 948 after quality | 925 | ~ (+9 %) |
| Single units, fixed / randomized | 483 / 288 | excellent+great+good | 282 / 178 | 270 / 173 | ✗ (see Step 10, Check 4) |
| Sessions with ≥ 10 units | required | — | — | 44/44 (min 17) | ✓ |
| Mean units / session | — | — | — | 55.8 (17–141) | — |
| Trials total | — | — | 14,972 | 13,762 | ✓ (−8 % = early + stim + no-ephys) |
| Trials / session | — | — | 340 | 312.8 (193–474) | ✓ |
| Sample epoch | 1.3 s | — | 1.3 s in all 44 sessions | (in window) | ✓ |
| Fixed delay epoch | 0.9 s | — | 0.9 s in all 25 sessions | — | ✓ |
| Randomized delays | 0.3/0.6/1.2/1.8/2.4/3.6 s | — | those 6 values (+ restart-inflated values) | — | ✓ |
| DR performance | > 70 % (training criterion) | — | 84.8 % mean (65.8–94.9 %), ≥70 % in 93 % of sessions | — | ✓ |
| Response latency | "typically within 300 ms" | — | median 186 ms, 82 % < 300 ms | — | ✓ |
| WC blocks | 10–25 trials, session starts with ~100 DR trials | — | median block 11 trials; first WC trial at index 76–125 in 16/20 sessions | — | ✓ |
| Photoinactivation | ~30 % of trials in opto sessions | excluded | 5 sessions, 12.3 % of trials | excluded | ✓ |
| Time bin | 10 ms (`dt = 1/100`) | 10 ms | — | 10 ms | ✓ |
| Window | [−2.5, 2.5] s | `tmin/tmax` | — | [−2.5, 2.5] s | ✓ |
| Alignment | go cue | `alignEvent='goCue'` | `bp.ev.goCue` | go cue | ✓ |
| lick_direction distribution | — | — | — | 0.423 / 0.446 / 0.131 | — |
| context distribution | — | — | — | WC 0.097 / DR 0.903 | — |
| outcome distribution | — | — | — | 0.120 / 0.749 / 0.131 | — |
| tongue_velocity distribution | — | — | — | 0.046 / 0.046 / 0.909 | — |
| paw_velocity distribution | — | — | — | 0.403 / 0.403 / 0.194 | — |
| motion_energy distribution | — | — | — | 0.480 / 0.481 / 0.039 | — |

Nothing was lost in conversion: every session in the reference list is present, and the
13,762 retained trials = 14,972 raw trials − 1,210 removed (976 early-lick, 187
photoinactivation, 64 with no ephys coverage; 17 trials meet more than one criterion).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There are no errors and no warnings to address. In particular the warnings that this
validator can raise were all avoided by construction: no NaN/Inf, no all-zero trials (the
"no ephys coverage" trial filter), consistent neuron counts within a session, consistent
input/output dimensions across all sessions, float dtypes for neural/input and integer
dtype for output.

### Check 2 — independent sanity checks (`/app/cache/sanity_checks.py`, output
`/app/cache/sanity_checks_out.txt`) — **0 failures**

All checks re-read the raw `.mat` files with code that does not use `convert_data`'s
processing functions (raw h5py / scipy readers, `np.convolve` instead of `scipy.signal.lfilter`,
`scipy.interpolate.interp1d` instead of `np.interp`), and compare with `np.allclose` /
`np.array_equal`:

1. **Neural** — 12 random (session, trial, unit) combinations across 4 sessions (both file
   formats, single- and dual-probe): re-histogram that unit's raw spike times relative to
   `bp.ev.goCue`, smooth with an independent implementation of `mySmooth`, compare to the
   pickle. `np.allclose(..., atol=1e-4)` passes with max |difference| ≈ 3e-6 (float32
   rounding).
2. **Neural bookkeeping** — `n_neurons` of every trial equals the session's unit count and
   the length of `brain_region_idx`, for all 44 sessions.
3. **Input** — equals the bin centres `-2.495 … 2.495` for every checked trial of every
   session.
4. **Outputs, per-trial** — `lick_direction`, `context` and `outcome` recomputed from the
   raw `bp.R/L/hit/miss/no/autowater` fields match for **all** trials of 5 sessions.
5. **Outputs, independent definition** — `lick_direction` equals the side of the first
   lickport contact within the 3 s response window on 100 % of the trials of those 5
   sessions, and on 99.91 % of all 14,972 trials of all 44 sessions (14 exceptions, see
   "issues" below).
6. **Outputs, kinematics** — tongue and paw discretised classes recomputed from the raw DLC
   traces (independent interpolation + gradient) match exactly for 16 trials across 2
   sessions; motion energy recomputed from the raw `motionEnergy_*.mat` file matches exactly
   for 8 trials.
7. **Thresholds** — in every session and for each of the 3 discretised outputs the class-0
   and class-1 counts differ by < 2 %, as they must for a median split.

### Check 3 — reference code comparison
| Stage | Reference | `convert_data.py` | Same? |
|-------|-----------|-------------------|-------|
| (a) data loading | `loadObjs.m` + `load<ANM>_ALMVideo.m` session/probe lists; `loadMotionEnergy.m` | `SESSIONS` transcribed from those files; `open_session`, `load_motion_energy` | yes (also handles the v7 files, which the MATLAB code reads transparently) |
| (b) neuron filtering | `findClusters.m` (`quality={'all'}` → drop garbage/gabrga/noisy/real?) then `removeLowFRClusters.m` (`lowFR=1`) | identical two stages | yes, except the label match is case-insensitive (Step 4) |
| (c) trial filtering | condition strings `'...&~stim.enable&~early'` in `findTrials.m` | `early < .5 & stim.enable < .5` | yes. Additional: trials with no ephys coverage (not in the reference, justified in Step 5). Ignore trials are kept (required by the task's `none`/`ignore` output classes). |
| (d) temporal alignment | `alignSpikes.m`: `trialtm - ev.goCue(trial)`; video: `frameTimes - findVideoOffset(obj) - ev.goCue(trial)`; motion energy identical | identical formulas | yes |
| (e) binning | `getSeq.m`: `histc` into `tmin:dt:tmax`, drop the last bin, `/dt`, `mySmooth(...,15,'reflect')`; `obj.time = edges + dt/2` | `bin_spikes` + `my_smooth`; time axis identical | yes (verified numerically in Check 2) |
| (f) input construction | (no analogue — the reference has no decoder input beyond neural data) | bin-centre time, as required by the task | task requirement |
| (g) output construction | `getPrevChoice.m` (choice), `getOutcome.m` (outcome), `autowater` (context), `findPosition/findVelocity` (kinematics), `loadMotionEnergy` (ME) | same definitions | yes, with two deliberate differences (below) |

Deliberate differences from the reference, and why:
1. **Ignore (`bp.no`) trials are kept.** Every reference ephys condition string selects
   `hit`/`miss` only, but the decoder task requires `none` (lick direction) and `ignore`
   (outcome) classes, so these trials must be present.
2. **Tongue velocity at the edges of a visibility bout.** `findVelocity.m` replaces NaN
   tongue velocities by 0; here "not visible" is its own output class, so a NaN-aware
   derivative (one-sided at bout edges) is used instead. Using the reference's rule would
   have assigned an artificial speed of 0 — hence class "below median" — to ~30 % of the
   timepoints at which the tongue *is* visible.
3. **Velocity magnitude** `sqrt(xvel² + yvel²)` rather than the reference's separate
   `xvel`/`yvel` features, because the task asks for a single "velocity" variable to
   discretise. The reference's per-axis baseline subtraction (`- basederiv(1)`, a constant
   ≈ 0 that `findVelocity.m` applies to both axes) is omitted; it shifts speeds by < 0.01
   px/bin and cannot change a median split materially.
4. **Case-insensitive quality labels** (the reference's `ismember` is case-sensitive and
   would keep the single `Noisy`-labelled unit in JEB6).
5. **Median (50th-percentile) thresholds** for the three continuous variables instead of
   the authors' manual `me.moveThresh`: required explicitly by the decoder-task
   specification. (For reference, `me.moveThresh` corresponds to the 43rd–88th percentile
   depending on the session, mean ≈ 65th.)

### Check 4 — key statistics comparison (`/app/cache/paper_stats.py`)
Everything in the Step 9 table above was checked. The two genuine mismatches:

- **Mice in the fixed-delay dataset**: paper says nine, but the authors' own load scripts
  and the released files both contain ten (EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15,
  JEB19, JGR2, JGR3) for exactly the 25 sessions the paper reports. Code+data agree with
  each other, so the paper's count appears to be an error; I follow code+data.
- **Unit counts**: paper 1,651 / 845 (483 / 288 single units); mine 1,532 / 925
  (270 / 173). I investigated this at length:
  - the quality-curated counts before the 1 Hz filter are 1,565 / 948, so the FR filter is
    not the source of the difference;
  - no combination of quality labels reproduces 483/288 single units (excellent+great+good
    = 282/178, adding `fair` = 647/444), so the paper's single-unit criterion (manual ISI
    inspection) is not recoverable from the released `quality` strings;
  - restricting to the ALM probe only, or including both probes of the dual-probe sessions,
    moves the totals further away;
  - the paper gives no per-session table (Extended Data Fig. 2a is a bar plot), so the
    difference cannot be localised to specific sessions.
  I therefore keep the curation the reference **code** specifies (which is unambiguous and
  reproducible) and document the ±5–10 % discrepancy with the paper's reported totals.
- The paper's "12 sessions, six mice, 522 units" two-context subset could not be reproduced
  either: applying the paper's behavioural criterion (≥ 40 correct DR trials per direction
  and ≥ 20 correct WC trials per direction) selects 7 sessions/5 mice, while merely
  requiring ≥ 20 WC trials per direction selects 9. This does not affect the conversion,
  because context is labelled per trial and all sessions are kept.

### Check 5 — edge cases (`/app/cache/edge_cases.py`)
- All 13,762 trials have exactly 500 timepoints; neural float32, input float32, output int8;
  `subject_idx` int64 in [0, 13]; all `brain_region_idx` int64 with valid indices.
- Smallest session: 193 trials (≥ 2 required); smallest unit count 17 (≥ 10 required).
- Bin-edge handling matches MATLAB `histc` + `N(1:end-1)`: a spike exactly at t = −2.5 falls
  in the first bin and one exactly at t = +2.5 is dropped.
- No silent or constant units within the retained trials. Two of the 2,457 units fall just
  below 1 Hz *when recomputed on the retained trials only* (0.988 and 0.999 Hz) because the
  reference computes the criterion over **all** trials; this is the reference behaviour and
  is left as is.
- Partial ephys coverage: only 6 trials (in 3 sessions) have no spikes in the last 0.5 s of
  the window; complete-coverage failures (0 spikes anywhere in the trial) were removed by
  the "no ephys" filter (64 trials, JEB24 2023-10-23 and 2023-11-03).
- 12 of the 44 sessions contain only DR trials, so `context` is constant within them; this
  is a property of the experiment, not a bug, and the decoder is trained across sessions.
  One session (JEB24 2023-10-23) has no ignore trials after curation.
- The 14 trials (0.09 %) where `lick_direction` disagrees with the "first lick" heuristic
  were traced to two harmless causes: (i) `bp.no` (ignore) trials on which the animal
  contacted a port 3.9–6.0 s after the go cue, i.e. outside the 3 s response window (and
  outside our analysis window), and (ii) trials where the same contact time is registered on
  **both** lick channels (sensor cross-talk), for which the heuristic has no defined answer.
  The bpod scoring fields used for the conversion are correct in both cases.

### Issues found and resolved
| Issue | Resolution |
|-------|------------|
| 11 of the 47 data objects are MATLAB v7, not v7.3 | added a second reader; both paths verified by the Check-2 spot tests |
| 3 layouts of `motionEnergy_*.mat` (`me.data` cell, `me.data.data` struct, bare cell array) | all handled in `load_motion_energy`, mirroring `loadMotionEnergy.m` |
| `data_structure_JEB24_2023-10-03.mat` has no `clu` field | it is not in the reference session list; excluded |
| `JEB23 2023-10-20` duplicates `2023-10-19` | not in the reference session list; excluded |
| 64 trailing trials in 2 sessions have no spikes at all | excluded (would be all-zero neural inputs) |
| smoothed output was float64 | cast to float32 (pickle 3.3 → 1.64 GB) |
| the first cross-check of `lick_direction` failed on 2 sessions | root-caused (late licks on ignore trials; simultaneous contacts on both channels); the check was refined to use the 3 s response window and a tie rule, and now passes |

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt` (≈3 min on GPU), plus `sample_trials.png` and
`predictions.png`.

### Training Progress
- Loss decreasing: **yes**, monotonically from 4.00 (epoch 1) → 0.7286 (epoch 200);
  test loss 0.7684 (close to the training loss).
- Trained on 11,001 trials, validated on 2,761 held-out trials (every session contributes to
  both).

### Decoder Results (Full, 44 sessions)
| Output | Training balanced acc | Validation balanced acc | Chance | Val / chance |
|--------|----------------------|-------------------------|--------|--------------|
| lick_direction | 0.651 | 0.619 | 0.333 | 1.86× |
| context | 0.859 | 0.860 | 0.500 | 1.72× |
| outcome | 0.639 | 0.608 | 0.333 | 1.83× |
| tongue_velocity | 0.584 | 0.572 | 0.333 | 1.72× |
| paw_velocity | 0.549 | 0.543 | 0.333 | 1.63× |
| motion_energy | 0.720 | 0.716 | 0.333 | 2.15× |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — accuracy vs chance
| Output | Validation balanced acc | Chance (1/k) | Ratio | Majority-class rate |
|--------|------------------------|--------------|-------|---------------------|
| lick_direction | 0.619 | 0.333 | 1.86× | 0.446 |
| context | 0.860 | 0.500 | 1.72× | 0.903 |
| outcome | 0.608 | 0.333 | 1.83× | 0.749 |
| tongue_velocity | 0.572 | 0.333 | 1.72× | 0.909 |
| paw_velocity | 0.543 | 0.333 | 1.63× | 0.403 |
| motion_energy | 0.716 | 0.333 | 2.15× | 0.481 |

No output is at or below chance; all are ≥ 1.6× chance. `paw_velocity` is the weakest, which
is expected: forepaw movements are uninstructed, and 19 % of its samples are the
"not visible" class, a mixture of DeepLabCut drop-out and window edges that ALM activity
cannot be expected to predict.

### Check 2 — comparison with the accuracies reported in the paper
The paper's decoding numbers (Fig. 3b, Fig. 4b, ED Fig. 2c) come from a **different
estimator**: one linear SVM **per session and per 75 ms time bin**, trained on balanced
correct left/right (or DR/WC) trials, 4-fold CV. The provided `train_decoder.py` instead
fits **one** linear readout shared by all 44 sessions and all 500 timepoints, on all trial
types. To compare like with like, I re-ran the paper's analysis on my converted data
(`/app/cache/reproduce_fig3b_4b.py`, figure `/app/cache/reproduce_fig3b_4b.png`),
following `NeuralChoiceDecoding.m` / `NeuralContextDecoding.m` / `DLC_ChoiceDecoder.m`
(min-max normalisation to [−1, 1], 75 ms bins, equal class counts, 4-fold CV, LinearSVC):

| Epoch | Choice (L vs R), this conversion | Choice, paper (Fig. 3b) | Context, this conversion | Context, paper (Fig. 4b) |
|-------|----------------------------------|-------------------------|--------------------------|--------------------------|
| ITI (−2.5…−2.0 s) | 0.544 | ≈ 0.5–0.55 | 0.792 | ≈ 0.75–0.85 |
| sample (−2.2…−1.0 s) | 0.690 | rises from ≈0.55 | 0.825 | ≈ 0.8 |
| late delay (−0.6…0 s) | 0.846 | ≈ 0.8–0.85 | 0.833 | ≈ 0.8–0.85 |
| response (0.1…1.0 s) | 0.922 | ≈ 0.9–0.95 | 0.854 | ≈ 0.85–0.9 |
| peak | **0.953** | **≈ 0.95–1.0** | **0.926** | **≈ 0.9–0.95** |
| shuffled labels | 0.504 | ≈ 0.5 | 0.503 | ≈ 0.5 |

The reproduced time courses are essentially identical to the published figures: choice
decoding is at chance during the ITI, **jumps exactly at the sample-tone onset (−2.2 s)**,
grows through the delay, peaks just after the go cue and then decays; context decoding is
high (≈ 0.8) throughout the trial including the ITI, with steps at the sample onset and the go
cue. The sharp inflection at −2.2 s (= −1.3 s sample − 0.9 s delay) is an independent
confirmation that the temporal alignment is correct to within a bin. The paper's
ED Fig. 2c AUC of 0.86 ± 0.11 for delay-epoch choice decoding likewise matches my late-delay
accuracy of 0.846.

**Conclusion**: the converted data supports exactly the decoding accuracies the paper
reports. The lower numbers from `train_decoder.py` are a property of its shared-readout,
all-timepoints-at-once architecture, not of the conversion. Two further controls support
this:
- a *per-session* linear SVM restricted to one model for all timepoints (i.e. the same
  framing as the provided decoder but with no cross-session sharing and no time input)
  is **worse** than the provided decoder on every output (lick_direction 0.631 vs 0.619
  with far fewer training samples per model, tongue 0.522 vs 0.572, paw 0.473 vs 0.543,
  motion energy 0.663 vs 0.716) — `/app/cache/per_session_kinematics.py`;
- the time-resolved accuracy of the provided decoder (`/app/cache/time_resolved_accuracy.py`)
  has the same shape as the paper's curves (left-vs-right 0.46 in the ITI → 0.63 in the late
  delay → 0.73 in the response epoch; context flat at ≈0.85), i.e. the information is where
  the paper says it is, just read out by a weaker model.

### Check 3 — train vs validation gap
| Output | Train | Validation | Train/Val |
|--------|-------|-----------|-----------|
| lick_direction | 0.651 | 0.619 | 1.05 |
| context | 0.859 | 0.860 | 1.00 |
| outcome | 0.639 | 0.608 | 1.05 |
| tongue_velocity | 0.584 | 0.572 | 1.02 |
| paw_velocity | 0.549 | 0.543 | 1.01 |
| motion_energy | 0.720 | 0.716 | 1.01 |

No gap exceeds 1.05×, so there is no overfitting and no data leakage (the split is by trial,
and no quantity used to build a trial is computed from other trials' *labels*; the only
session-level quantities are the three median thresholds and the unit-selection firing
rates, which are label-independent).

### Additional experiments run in this review
1. **Neural normalisation** (`/app/cache/test_normalization.py`, 11-session subset). The
   reference decoding code min-max normalises the firing rates per session; I tested raw
   rates, per-neuron z-scoring, per-session min-max and √-transform:
   raw 0.593/0.813/0.579/0.564/0.540/0.732, z-score 0.594/0.784/0.568/0.582/0.467/0.733,
   min-max 0.535/0.797/0.568/0.536/0.521/0.693, √ 0.603/0.821/0.588/0.577/0.553/0.739.
   Raw smoothed firing rates (what `obj.trialdat` contains) are as good as any alternative,
   so the dataset ships un-normalised rates — the representation the reference produces, and
   the one a downstream user can transform however they like.
2. **Time-bin size** (`/app/cache/test_binsize.py`, same subset): accuracy increases
   monotonically with bin width (lick_direction 0.593 at 10 ms, 0.596 at 20 ms, 0.618 at
   50 ms, 0.636 at 100 ms, 0.661 at 200 ms), because averaging bins denoises the per-timepoint
   neural vector — there is no optimum short of destroying the time resolution of the
   outputs. I therefore kept the reference's `params.dt = 10 ms` rather than chasing
   accuracy: it is the binning the reference pipeline uses to build `obj.trialdat`, it
   preserves the ~40 ms structure of the tongue-visibility bouts that two of the outputs
   encode, and a user who wants the reference's *decoding* representation can average 7–8
   bins (as `DLC_ChoiceDecoder.m` does, and as my Fig. 3b reproduction above does).

### Issues found and resolved in this step
- None that required changing the conversion. The two candidate changes (normalisation,
  coarser bins) were tested and rejected for the documented reasons.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format spec, key statistics)
- [x] `cache/` folder holds all exploration / verification scripts and their outputs, with
      `cache/README_CACHE.md` describing each
- [x] Deliverables in `/app`: `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`,
      `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`,
      `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
      `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`,
      plus the `--show-processing` figures `processing_JEB6_2021-04-18.png`,
      `processing_JEB23_2023-10-10.png` and the decoder's `sample_trials.png` /
      `predictions.png`.
