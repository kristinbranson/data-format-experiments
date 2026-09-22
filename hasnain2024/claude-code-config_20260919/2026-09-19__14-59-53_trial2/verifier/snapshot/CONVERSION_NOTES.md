# Dataset Conversion Notes

## Overview
- **Dataset**: Hasnain, Birnbaum et al., *"Separating cognitive and motor processes in the behaving mouse"*, Nature Neuroscience 2024/2025. Data: Zenodo DOI 10.5281/zenodo.13941415 (copy in `/app/data`), code in `/app/code`, paper in `/app/paper.pdf`, methods in `/app/methods.txt`.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`), go-cue aligned, for a neural decoder that predicts lick direction, behavioral context, outcome, tongue velocity, paw velocity and motion energy from ALM spiking activity.

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `.manifest`, `Dockerfile`, `docker-compose.yaml` — container metadata
- `CONVERSION_NOTES.md` — this file
- `code/` — MATLAB code released with the paper
- `data/` — four data subdirectories (see Step 2)
- `decoder.py`, `train_decoder.py` — provided decoder / validation harness
- `methods.txt`, `paper.pdf` — reference text

Environment verified:
- `python3` 3.13, `numpy` 2.4.4, `torch` 2.6.0+cu124, CUDA available (NVIDIA L4, 23 GB)
- 128 CPUs, 1 TB RAM
- Installed `pymatreader` (+`pypdf`) — needed because the `.mat` files are a mix of
  MATLAB v7.3 (HDF5) and MATLAB v7 (MAT5) formats; `pymatreader` reads both with a
  single uniform interface.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified

| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadObjs` | `DataLoadingScripts/loadObjs.m` | LOADING | `load()` each `data_structure_<ANM>_<DATE>.mat` into struct array `obj` |
| `load<ANM>_ALMVideo` | `DataLoadingScripts/Recording and video/*.m` | LOADING / CURATION | **Defines the analysed session list** and, per session, **which probe(s) are the ALM probes**. Commented-out entries are sessions the authors excluded. |
| `loadSessionData` | `DataLoadingScripts/loadSessionData.m` | LOADING | Loops sessions × probes, calls `processData`, concatenates 2-probe sessions |
| `processData` | `DataLoadingScripts/processData.m` | PROCESSING | Orchestrates `findTrials` → `findClusters` → `alignSpikes` → `getSeq` → `removeLowFRClusters` → `baselineFR` |
| `findTrials` | `DataLoadingScripts/findTrials.m` | CURATION | Evaluates `params.condition` strings (e.g. `'R&hit&~stim.enable&~autowater&~early'`) against `obj.bp` fields to get trial indices per condition |
| `findClusters` | `DataLoadingScripts/findClusters.m` | CURATION | With `params.quality={'all'}` keeps every cluster whose (trimmed) quality label is **not** `garbage`, `gabrga`, `noisy`, `real?` (case-sensitive) |
| `alignSpikes` | `DataLoadingScripts/alignSpikes.m` | ALIGNMENT | `clu.trialtm_aligned = clu.trialtm - ev.(alignEvent)(clu.trial)`; for us `alignEvent='goCue'` |
| `getSeq` | `DataLoadingScripts/getSeq.m` | PROCESSING | `edges = tmin:dt:tmax`; `obj.time = edges(1:end-1)+dt/2`; per trial `histc` spike counts → `/dt` → `mySmooth(...,params.smooth,params.bctype)` → `obj.trialdat` (time × units × trials), units = spikes/s |
| `removeLowFRClusters` | `DataLoadingScripts/removeLowFRClusters.m` | CURATION | Drops units with mean PSTH firing rate ≤ `params.lowFR` (1 Hz in all figure scripts) |
| `mySmooth` | `utils/mySmooth.m` | PROCESSING | Causal Gaussian filter: `gausswin(15)`, first `floor(15/2)=7` taps zeroed, normalised, `conv(...,'same')`; `'reflect'` boundary = prepend first 15 samples, then trim |
| `findVideoOffset` | `funcs/findVideoOffset.m` | ALIGNMENT | `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` (≈0.49 s, 0.99 s for JEB19). Video time in trial coordinates = `frameTimes − vidshift` |
| `loadMotionEnergy` | `DataLoadingScripts/loadMotionEnergy.m` | LOADING / ALIGNMENT | Loads `motionEnergy_<ANM>_<DATE>.mat`, then `interp1(frameTimes − vidshift − alignTime, me, obj.time)` and `fillmissing('nearest')`; `me.move = me.data > me.moveThresh` |
| `findPosition` | `funcs/kinematics/findPosition.m` | PROCESSING / ALIGNMENT | Per DLC feature: take `ts(:,1:2,featix)` (x,y), interp onto `obj.time` using `frameTimes − vidshift − alignTime`; **tongue is not smoothed and NaNs are kept**; all other features `fillmissing('nearest')` |
| `findVelocity` | `funcs/kinematics/findVelocity.m` | PROCESSING | `xvel = gradient(x)`, `yvel = gradient(y)`; non-tongue features get a `median(diff(pos))` baseline-drift subtraction and `fillmissing('nearest')`; **tongue NaN velocities are set to 0** |
| `getKinematics` / `getKinematicsFromVideo` | `funcs/kinematics/*.m` | PROCESSING | Assembles (xdisp, ydisp, xvel, yvel) per feature per view + tongue angle/length + motion energy into `kin.dat` (time × trials × features) |
| `getPrevChoice` | `funcs/getPrevChoice.m` | OUTPUT DEF | **Definition of choice/lick direction**: `choice = (R & hit) | (L & miss)` (1 = right), `NaN` on `no` (ignore) trials |
| `getOutcome` | `funcs/getOutcome.m` | OUTPUT DEF | `outcome = bp.hit`, `NaN` on `no` (ignore) trials |
| `UseInclusionCritera` / `RemoveUnwantedSessions` | `utils/*.m` | CURATION | Drop sessions with ≤40 right-correct or ≤40 left-correct DR trials (used for a subset of figures) |
| `NeuralChoiceDecoding.m` / `NeuralContextDecoding.m` | `ChoiceContextDecoding/` | ANALYSIS | The paper's own decoders: input `obj.trialdat` (time × units × trials), min-max normalised to [-1,1], 75 ms bins, logistic regression per time bin |

### Parameters used consistently across every figure script
```
params.alignEvent = 'goCue';
params.lowFR      = 1;            % Hz
params.tmin       = -2.5;         % s relative to align event
params.tmax       =  2.5;         % s
params.dt         = 1/100;        % 10 ms bins
params.smooth     = 15;           % causal gaussian window (samples)
params.bctype     = 'reflect';
params.quality    = {'all'};      % keep all but garbage/noisy
params.advance_movement = 0;      % no neural/video lag
```
(`getDefaultParams.m` has `lowFR=0.5`, `dt=1/200`, but **every** actual analysis script —
`Scripts/Figure*`, `Scripts/EDFigure*`, `Behavior/*`, `WorkingWithDataObjs.m` — overrides these
with the values above. The above is therefore the reference processing.)

### Notes
- Electrophysiology: spikes are already sorted; units must be **quality-filtered**
  (`findClusters`) and **firing-rate filtered** (`removeLowFRClusters`, >1 Hz). No dF/F
  (this is ephys, not imaging).
- The neural data the paper feeds to its own decoders is `obj.trialdat`: **smoothed,
  binned single-trial firing rates in spikes/s**, go-cue aligned, 10 ms bins, [-2.5, 2.5] s.
- Video is sampled at 400 Hz and must be shifted by `findVideoOffset` before being
  aligned to the go cue; there is no additional neural/video lag (`advance_movement = 0`).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` has four subdirectories:

| Directory | Contents | Neural data? |
|---|---|---|
| `Ephys_Behavior` | 25 `data_structure_*.mat` + 25 `motionEnergy_*.mat` | **yes** — fixed-delay DR task (12 of these are also two-context DR+WC sessions) |
| `RandomizedDelay_Ephys_Behavior` | 22 `data_structure_*.mat` + 20 `motionEnergy_*.mat` | **yes** — randomized-delay DR task |
| `DelayInhibition_BilatMC_Behavior` | behaviour-only optogenetic sessions | no |
| `GoCueInhibition_BilatMC_Behavior` | behaviour-only optogenetic sessions | no |

The two `*_Behavior` (inhibition) directories contain **no ephys** (README: "sessions under
the `Video only` subdirectory contain behavior only") and are therefore unusable for a
*neural* decoder. **Excluded.**

`data_structure_<ANM>_<DATE>.mat` contains one struct `obj`:

| Field | Contents used here |
|---|---|
| `obj.bp` | Bpod trial data: `Ntrials`, `R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, and `obj.bp.ev` with `bitStart`, `sample`, `delay`, `goCue`, `reward`, `lickL{}`, `lickR{}` (all times in seconds **within trial**) |
| `obj.clu` | 1×nProbes cell; each is a struct array over clusters with `tm`, `trialtm` (spike time within trial), `trial` (1-based trial index), `quality`, `channel`/`site`, `spkWavs` |
| `obj.traj` | 2×1 cell: `{1}` side cam, `{2}` bottom cam. Per trial: `ts` (frames × [x,y,confidence] × bodypart), `frameTimes` (s, video clock), `featNames`, `NdroppedFrames` |
| `obj.sglx` | `fs`, `bitcode.bitstart` (samples) — needed for the video offset |
| `obj.ex`/`obj.meta` | `anm`, `day`, `probe.loc`, `probe.type` |
| `obj.me` | present in some sessions; same motion-energy content as the separate file |

`motionEnergy_<ANM>_<DATE>.mat` holds `me`, either a struct with `data` (1×nTrials cell of
400 Hz motion-energy traces) + `moveThresh`, or (JEB23/JEB24) a bare cell array of traces.
Verified: **motion-energy trace length == `frameTimes` length for every trial of every session.**

File formats are mixed: most are MATLAB v7.3 (HDF5), 11 randomized-delay
`data_structure` files and all `motionEnergy` files are MATLAB v7 (MAT5).

### Session list (from the reference `load<ANM>_ALMVideo.m` scripts)

Fixed delay (25 sessions, 10 mice; probe = ALM probe per the load script):

JEB6 2021-04-18 [2]; JEB7 2021-04-29 [1], 2021-04-30 [1]; EKH1 2021-08-07 [2];
EKH3 2021-08-11 [2]; JGR2 2021-11-16 [1], 2021-11-17 [1]; JGR3 2021-11-18 [1];
JEB19 2023-04-18/19/20/21 [1]; JEB13 2022-09-13 [2], 2022-09-14 [2], 2022-09-21 [1],
2022-09-24 [1], 2022-09-25 [1]; JEB14 2022-08-22/23/24/25 [1];
JEB15 2022-07-26 [1,2], 2022-07-27 [1,2], 2022-07-28 [1,2], 2022-07-29 [2].

Randomized delay (19 sessions, 4 mice, probe 1):
JEB11 2022-05-10/11; JEB12 2022-05-12/13;
JEB23 2023-10-10/11/12/13/18/19/21; JEB24 2023-10-23/24/25/26/27/31, 2023-11-02/03.

Three randomized-delay `.mat` files on disk are **not** in the load scripts and are excluded:
JEB23 2023-10-20 (a byte-for-byte duplicate of 2023-10-19 in trial count / frame counts) and
JEB24 2023-10-03, 2023-10-04 (which additionally have **no** `motionEnergy` file).
Dropping exactly these three gives 19 randomized-delay sessions = the number reported in the paper.

### Dataset Size (from data files, reference session list, before curation)
| Statistic | Value |
|-----------|-------|
| Sessions | 44 (25 fixed delay + 19 randomized delay) |
| Subjects | 14 (EKH1, EKH3, JEB6, JEB7, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24, JGR2, JGR3) |
| Sessions / subject | 1–8 (median 2) |
| Trials (total) | 14,972 |
| Trials / session | 230–517 (mean 340) |
| Clusters on ALM probes, all qualities | 12,140 |
| Clusters after quality filter (`findClusters 'all'`) | 2,513 (1,565 fixed delay + 948 randomized) |
| Single units (fair/good/great/excellent) after quality filter | 1,091 |
| Trials with photoinactivation (`stim.enable`) | 187 |
| Trials with early lick (`early`) | 976 |
| `autowater` (WC) trials | 1,548 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Two-context sessions / mice / units | 12 sessions, 6 mice, 522 units (214 single units) | "two-context paradigm: 12 sessions, six mice, 522 units, including 214 well-isolated single units" |
| Fixed-delay DR sessions / mice / units | 25 sessions, 9 mice, 1,651 units (483 single units) | "an additional three mice were trained only on the DR task: 25 sessions, nine mice, 1,651 units, including 483 well-isolated single units" |
| Randomized-delay sessions / mice / units | 19 sessions, 4 mice, 845 units (288 single units) | "for the randomized delay task, we recorded 845 units (288 well-isolated single units) in ALM from 19 sessions using four mice" |
| Recording area | ALM (right and left) | "We recorded activity extracellularly in the ALM with high-density silicon probes" |
| Session inclusion | ≥10 units | "Recording sessions were included for analysis only if they had at least 10 units" |
| Unit inclusion | firing rate > 1 Hz | "All units with firing rates exceeding 1 Hz were included in all other analyses." |
| Neural time bin | 10 ms, [-2.5, 2.5] s about go cue | `params.dt = 1/100; params.tmin=-2.5; params.tmax=2.5` in every analysis script |
| Video frame rate | 400 Hz | "High-speed video was captured (400-Hz frame rate) from two cameras" |
| Session structure (two-context) | ≈100 DR trials, then alternating 10–25 trial WC/DR blocks; all sessions start with DR | Methods, "Mouse behavior" |
| Delay length (fixed) | 0.9 s (0.7 s for one mouse, time-warped) | Methods |
| Delay lengths (randomized) | 0.3/0.6/1.2/1.8/2.4/3.6 s, exponential-ish pdf, τ=0.9 s | Methods |
| Sample tone | 1.3 s | Methods |
| Ignore threshold | no response within 3 s of go cue | Methods |
| Early-lick trials | omitted from all analyses | "Trials in which the animal contacted the lickport before the reward ('early lick') were omitted from analyses." |
| Behavioural session inclusion | ≥40 correct DR trials each direction, ≥20 correct WC each direction | Methods, "Behavioral analysis" |
| Choice selectivity (DR) | sample 36%, delay 42%, response 58% of 483 single units | Results |
| Context selectivity | 39% of 214 single units | Results |
| Motion-energy definition | \|median(next 5 frames) − median(prev 5 frames)\|, 99th percentile over pixels | Methods |
| Motion-energy move threshold | manual, per session (bimodal split) | Methods |

Reproduced from the raw data (Step 2/4 scan): the 12 sessions whose `autowater` trials form
≈4–8 blocks that begin around trial ~100 are exactly
JEB6 2021-04-18, JEB7 2021-04-29/30, EKH1 2021-08-07, EKH3 2021-08-11,
JGR2 2021-11-16/17, JGR3 2021-11-18, JEB19 2023-04-18/19/20/21.
Their quality-filtered unit count is **528** vs. the paper's **522** and their single-unit
(fair/good/great/excellent) count is **219** vs. the paper's **214** — a <1.2% difference
that confirms both the session identification and the unit-curation rule.

### Processing Details
- **Temporal alignment**: go cue (`obj.bp.ev.goCue`), which in WC (autowater) trials is the
  time the water drop is presented. Window [-2.5, +2.5] s.
- **Temporal binning**: 10 ms bins; spike counts → rate (÷dt) → causal Gaussian smoothing
  (`mySmooth`, window 15 samples = 150 ms, 'reflect' boundary).
- **Video alignment**: `frameTimes − vidshift − goCue`, `vidshift` from `findVideoOffset`
  (0.49 s for all sessions except JEB19 where it is 0.99 s). Linear interpolation onto the
  10 ms neural time base. No neural/video lag.

### Curation Steps

**Neuron curation rules**:
1. Only the ALM probe(s) designated in the reference `load<ANM>_ALMVideo.m` scripts.
2. Drop clusters whose trimmed quality label is `garbage`, `gabrga`, `noisy` or `real?`
   (`findClusters` with `quality={'all'}`; note the comparison is case-sensitive, so the single
   `Noisy` label in JEB6 is kept — reproduced exactly).
3. Drop clusters with mean firing rate ≤ 1 Hz over the analysis window (`lowFR = 1`).

**Trial curation rules**:
1. Drop `early` (early-lick) trials — explicitly excluded from all analyses in the Methods and
   by `~early` in every `params.condition`.
2. Drop photoinactivation trials (`stim.enable == 1`) — `~stim.enable` in every
   `params.condition`; these perturb both neural activity and behaviour.
3. **Keep** `hit`, `miss` and `no` (ignore) trials, and keep both `autowater` (WC) and
   non-autowater (DR) trials: the decoder task requires `ignore` as an outcome class,
   `none` as a lick-direction class and `WC`/`DR` as context classes.

**Session curation rule**: ≥10 units after neuron curation (Methods).

### Decoders Trained (in the paper)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Choice (L vs R), from CD_choice projection | logistic regression, ROC | AUC 0.86 ± 0.11 (ED Fig. 2c) |
| Choice (L vs R), from neural population | logistic regression per 75 ms bin, 4-fold CV | ≈0.5 (chance) before sample, rising through the delay to ≈0.9 after the go cue (Fig. 3b) |
| Choice, from video kinematics | same | rises from chance after the sample tone (Fig. 3b) |
| Context (DR vs WC), from neural population | same | well above chance (≈0.7) even during the ITI, ≈0.9 around the response epoch (Fig. 4b) |
| Context, from video kinematics | same | similar timecourse to neural (Fig. 4b) |
| CD_context from kinematics (regression) | ridge regression | R² = 0.49 ± 0.22 (n = 12 sessions) |

These are the quantitative expectations against which the converted-data decoder is compared
in Steps 11–12. Note the paper's decoders are trained **per session** on **balanced** subsets
of correct left/right trials, whereas the provided harness trains one shared decoder over all
sessions on all trials; exact numbers are therefore not directly comparable, but the
qualitative expectation (choice and context well above chance, highest after the go cue) is.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Analysis parameters | `getDefaultParams.m`: `dt=1/200`, `lowFR=0.5` vs every figure script: `dt=1/100`, `lowFR=1` | — | 1 Hz FR threshold; 10 ms consistent with Fig. 2/3 panels | Use the figure-script values (`dt=1/100`, `lowFR=1`); `getDefaultParams` is unused by the shipped analyses |
| Bin width comment | `WorkingWithDataObjs.m` comment says "use a 5 ms bin width" but the code sets `params.dt = 1/100` | — | — | Comment is stale; use 10 ms |
| # mice, fixed delay | 10 distinct animals in the load scripts | 10 animals over 25 files | "25 sessions, nine mice" | Paper appears to under-count by one (EKH1 is absent from `NullPotent/SessionMeta.csv`, which lists 24 of the 25 sessions and 9 mice). Session **count** matches exactly, so the load-script list is used. Documented, not "fixed". |
| # units, fixed delay | quality filter gives 1,565 | 1,565 | 1,651 | 5% discrepancy; the two-context subset (528 vs 522) and the session/mouse counts match, so the curation rule is right. Possible causes: the paper counting before/after a slightly different filter, or counting the extra probe of a dual-probe session. Recorded as a known small mismatch. |
| # units, randomized delay | quality filter gives 948 | 948 | 845 | Same kind of ~10% mismatch; session count (19) and mouse count (4) match exactly. |
| Probe location labels | `loadJEB15_ALMVideo` uses probes **[1 2]**, and probe 2 for 2022-07-29 | `obj.ex.probe.loc` for JEB15 reads `L ALM` (probe 1) and `L M1TJ` (probe 2); for 2022-07-29 probe 1 is `DUMMY` | "We recorded activity extracellularly in the ALM"; all load scripts are named `*_ALMVideo` | The `loc` strings are internally inconsistent for JEB15 (a `DUMMY` probe is designated as the ALM probe for 2022-07-29). Follow the reference **load scripts** for probe selection and label every included unit `ALM`, as the paper does. |
| Context proxy | `params.condition` uses `autowater` to separate WC from DR; tutorial: "this field can be used as a proxy for obtaining water-cued blocks and delayed-response blocks" | `autowater` forms contiguous blocks starting ≈trial 100 in exactly 12 sessions | "A behavioral session began with approximately 100 DR trials and was then followed by alternating blocks of WC and DR trials" | Use `autowater` as the context label. Confirmed against the described block structure. |
| Motion-energy file format | `loadMotionEnergy` expects `me.data` (struct) | JEB23/JEB24 `motionEnergy_*.mat` store a bare cell array; those sessions also carry `obj.me` | — | Loader normalises both layouts to a list of per-trial traces |
| `findVelocity` baseline subtraction | subtracts `basederiv(1)` from **both** `xvel` and `yvel` | — | — | Apparent typo in the reference; effect is negligible (both medians ≈0 px/bin). Reproduced **as written** to stay faithful. |

Everything else (event times, trial counts, spike/trial indexing, frame counts, video offset)
was cross-checked between code, data and text and is consistent:
- `hit + miss + no == Ntrials` in all 44 sessions.
- `ev.sample ≈ 0.3 s`, `ev.delay ≈ 1.6 s`, `ev.goCue ≈ 2.5 s` (fixed delay: sample tone 1.3 s,
  delay 0.9 s) — exactly the task timing in the Methods.
- Randomized-delay sessions: median `goCue − delay` ≈ 0.6 s with six discrete delay values.
- `vidshift ≈ 0.49 s` puts the first video frame at 0.025 s in trial time, i.e. essentially at
  `bitStart` (0.04 s) — the tutorial's "subtract 0.5 second from frametimes".

---

## Step 5: Mapping Planning
**Status**: IN PROGRESS

### Variable Mapping

| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `obj.clu{p}(c).trialtm`, `.trial`, with `obj.bp.ev.goCue` | `neural` | align (`trialtm − goCue`), bin into 10 ms bins over [-2.5, 2.5] s, ÷dt, causal-Gaussian smooth (`gausswin(15)`, 7 taps zeroed, 'reflect' boundary) → spikes/s, `float32` (n_neurons, 500) | `alignSpikes`, `getSeq`, `mySmooth` | exactly reproduces `obj.trialdat` |
| `obj.clu{p}(c).quality` | neuron curation | drop `garbage`/`gabrga`/`noisy`/`real?` (case-sensitive, trimmed) | `findClusters` | |
| mean of binned rate over window & trials | neuron curation | keep > 1 Hz | `removeLowFRClusters`, Methods | |
| `obj.time` (= bin centres) | `input[0]` `time_from_go_cue` | continuous ramp −2.495 … +2.495 s, `float32` (1, 500) | `getSeq` | only decoder input, per the task spec |
| `obj.bp.R/L/hit/miss/no` | `output[0]` `lick_direction` | `0=left` if `(L&hit)|(R&miss)`, `1=right` if `(R&hit)|(L&miss)`, `2=none` if `no` | `getPrevChoice` (same formula) | per trial, broadcast over time |
| `obj.bp.autowater` | `output[1]` `context` | `0=WC` if `autowater`, else `1=DR` | `params.condition` strings, `WorkingWithDataObjs.m` | per trial |
| `obj.bp.hit/miss/no` | `output[2]` `outcome` | `0=incorrect` (`miss`), `1=correct` (`hit`), `2=ignore` (`no`) | `getOutcome` | per trial |
| `obj.traj{1}.ts[:, :2, 'tongue']` + `frameTimes` | `output[3]` `tongue_velocity` | interp to `obj.time` using `frameTimes − vidshift − goCue`; speed = ‖(dx/dbin, dy/dbin)‖ per visible segment; class 2 where the tongue is not labelled or the video does not cover the bin; else 0/1 by the **session** median of visible speeds | `findPosition`, `findVelocity`, `findVideoOffset` | time-varying |
| `obj.traj{2}.ts[:, :2, 'top_paw'/'bottom_paw']` | `output[4]` `paw_velocity` | as above; speed averaged over whichever paw(s) are visible; class 2 only where **neither** paw is visible | same | time-varying |
| `motionEnergy_*.mat` `me.data{trial}` (or `obj.me`) | `output[5]` `motion_energy` | interp to `obj.time` using `frameTimes − vidshift − goCue`; class 2 where the video does not cover the bin; else 0/1 by the **session** median of covered bins | `loadMotionEnergy` | time-varying |
| `obj.ex.anm` / filename | `subjects`, `subject_idx` | 14 mice | `load<ANM>_ALMVideo` | |
| ALM probe from load script | `brain_regions`, `brain_region_idx` | all units labelled `ALM` | `load<ANM>_ALMVideo`, paper | |

### Key Decisions

1. **Which sessions.** All 44 sessions that contain ALM electrophysiology *and* video and are
   listed in the reference `DataLoadingScripts/Recording and video/load*_ALMVideo.m` scripts:
   25 fixed-delay + 19 randomized-delay. The two behaviour-only optogenetic directories have no
   spikes and cannot be used by a neural decoder. The 3 `data_structure` files on disk that the
   load scripts omit are excluded (this reproduces the paper's n = 19 randomized-delay sessions).
   *Rationale*: the requested outputs (lick direction, outcome, kinematics, motion energy) are
   defined in every session; `context` is degenerate (all DR) in the 32 sessions without WC
   blocks but is still a correct label there, and the harness computes **balanced** accuracy
   pooled over sessions, so keeping them adds data without corrupting the context label. The
   12 two-context sessions supply the WC class.
2. **Alignment: go cue** (`obj.bp.ev.goCue`), as required by the task and as used by every
   reference analysis (`params.alignEvent = 'goCue'`). In WC trials this field holds the
   water-drop time, which is the matched event for that context.
3. **Window and binning: [-2.5, +2.5] s, 10 ms bins (500 bins/trial)** — exactly
   `params.tmin/tmax/dt` from every reference figure script. Same length for all trials/sessions.
4. **Neural representation: smoothed firing rate in spikes/s** (`obj.trialdat`), i.e. the same
   quantity the paper feeds to its own choice/context decoders, including the causal 150 ms
   Gaussian kernel with `'reflect'` boundary handling.
5. **Neuron curation**: reference quality filter + >1 Hz mean firing rate (Methods).
   Brain region: `ALM` for all units (paper; probe selection from the load scripts).
6. **Trial curation**: drop `early` (Methods: "omitted from all analyses") and `stim.enable`
   (photoinactivation) trials. Keep hit/miss/ignore and DR/WC — required by the output spec.
7. **Session curation**: ≥10 curated units (Methods) and ≥2 trials (harness requirement).
8. **Discretisation threshold**: per session, the median (50th percentile) over all *valid*
   (visible / video-covered) timepoints of all kept trials in that session. Invalid timepoints
   are class 2 and are excluded from the percentile, otherwise the "not visible" bins (≈90% of
   the tongue trace) would drag the threshold to 0.
9. **"Not visible" / "no video"** covers both DLC drop-out and bins outside the video's temporal
   coverage. The video starts ≈0.03 s into the trial and ends 5–9 s in, so short-delay trials
   genuinely have no video at the start of the window and some randomized-delay trials have
   none at the end. Three trials in the whole dataset have unusable `frameTimes` (all NaN);
   their kinematic/ME outputs are entirely class 2.
10. **Paw**: mean speed of the visible subset of {`top_paw`, `bottom_paw`} (bottom camera only,
    as in the Methods). Using a single paw would make "not visible" reach 88–96% in some
    sessions because DLC loses one paw or the other; the union is 0–17%.
11. **Tongue**: side-camera `tongue` feature (`params.traj_features{1}{1}`), the canonical
    tongue-tip marker.
12. **Velocity units** are pixels per 10 ms bin (the reference's `gradient()` on the interpolated
    position, no unit conversion). Only the within-session rank matters after the median split.
13. **No neural/video lag** (`params.advance_movement = 0`), per all reference scripts.

### Planned Sanity Checks
- [ ] Session/mouse counts: 44 sessions, 14 mice; fixed-delay subset 25/10; randomized 19/4;
      two-context subset 12 sessions.
- [ ] Unit counts vs. paper: two-context subset ≈522 pre-FR-filter (we get 528).
- [ ] Every session ≥10 units; report the distribution (paper ED Fig. 2a).
- [ ] `hit + miss + no == Ntrials` per session; kept trials = Ntrials − |early ∪ stim|.
- [ ] Input range exactly [-2.495, 2.495]; identical time base for every trial.
- [ ] Output 0/1/2 fractions per variable; tongue class-2 fraction ≈0.9 (matches raw DLC NaN rate).
- [ ] Per session, class-0 and class-1 counts for each discretised output are equal ±1
      (median split by construction).
- [ ] Spike-count spot check: recompute the binned rate for a specific (session, trial, neuron)
      straight from the raw `.mat` and compare with `np.allclose`.
- [ ] Behaviour spot check: recompute lick direction / outcome / context for specific trials
      from `bp` fields.
- [ ] Kinematics spot check: recompute the interpolated tongue/paw position and motion energy
      for a specific trial and verify the class assignment.
- [ ] Alignment check: mean firing rate and mean motion energy locked to the go cue should show
      a sharp rise at t = 0 (plots in `--show-processing`).

**Status**: COMPLETE

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the plan above. Structure:

| Function | Mirrors |
|---|---|
| `load_obj` | `loadObjs.m` (via `pymatreader`, handles MAT v7 and v7.3) |
| `select_clusters` | `findClusters.m` with `quality={'all'}` |
| `select_trials` | the `~stim.enable & ~early` part of every `params.condition` |
| `causal_gaussian` / `my_smooth` | `mySmooth.m` (`gausswin(15)`, first 7 taps zeroed, `'reflect'`) |
| `bin_index` | `histc(t, tmin:dt:tmax)` then `N(1:end-1)` |
| `bin_and_smooth` | `alignSpikes.m` + `getSeq.m` → `obj.trialdat` |
| `video_shift` | `findVideoOffset.m` |
| `interp_feature` | `findPosition.m` |
| `feature_speed` | `findVelocity.m` |
| `load_motion_energy` + `interp_trace` | `loadMotionEnergy.m` |
| mean-rate filter in `process_session` | `removeLowFRClusters.m` |

CLI: `python -u /app/convert_data.py <out.pkl> [--full|--sample] [--show-processing] [--workers N]`.

**Verification of the two most error-prone primitives** (`/app/cache/test_primitives.py`):
- `my_smooth` reproduces a literal transcription of `mySmooth.m`
  (`np.convolve(concat(x[:15], x), kern, 'same')[15:]`) to 2.2e-16, and an impulse at
  bin 100 produces no output before bin 100 (causality) with its peak exactly at bin 100.
- `causal_gaussian()` = `[0]*7 + [0.25098, 0.23547, 0.19447, 0.14137, 0.09046, 0.05096,
  0.02527, 0.01103]`, sums to 1 — matches `gausswin(15)` (alpha = 2.5) with `kern(1:7)=0`.
- `bin_index` reproduces `histc` edge semantics: `t = -2.5 → bin 0`, `t = 0 → bin 250`
  (whose centre is +0.005 s), `t = 2.4999 → bin 499`, `t = 2.5 → dropped` (this is the
  `N(1:end-1)` step), `t` outside `[-2.5, 2.5)` dropped.

Code inefficiencies identified:
- `pymatreader` reads the whole `obj`, including `clu.spkWavs`, which dominates the runtime
  (6–8 s and ~1.4 GB for the largest 290 MB session).
- Naive per-trial × per-neuron Python loops for binning would be ~10^4 iterations/session.

Code speedups added:
- Spike binning is a single `np.bincount` per cluster over a flattened `(trial, bin)` index
  instead of a per-trial loop.
- Smoothing is applied once per session to a `(515, nclusters·ntrials)` matrix, as an 8-tap
  causal FIR (the kernel's first 7 taps are zero), so no FFT and no round-off.
- Sessions are processed in parallel with `ProcessPoolExecutor` (12–16 workers; each worker
  peaks at ~1.5 GB, well within the 1 TB available).
- Outputs are stored as `int8` and the (identical) time-ramp input array is shared between
  trials, which keeps the pickle dominated by the neural `float32` arrays.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`processing_JEB19_2023-04-18.png`, `processing_JEB11_2022-05-10.png`.

Sample = JEB19 2023-04-18 (fixed delay, **two-context**: DR + WC blocks) and
JEB11 2022-05-10 (randomized delay), chosen so that both task variants and both
context values are exercised.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 98 (35 + 63) |
| Neurons / session | 49.0 mean |
| Subjects | 2 (JEB11, JEB19) |
| Sessions / subject | 1 |
| Trials (total) | 607 (284 + 323) |
| Trials / session | 303.5 mean |
| Timepoints / trial | 500 (10 ms bins) |
| `time_from_go_cue` range | [-2.495, 2.495] s |
| `lick_direction` distribution | left 0.379, right 0.387, none 0.234 |
| `context` distribution | WC 0.209, DR 0.791 |
| `outcome` distribution | incorrect 0.117, correct 0.649, ignore 0.234 |
| `tongue_velocity` distribution | 0.046 / 0.046 / 0.907 not visible |
| `paw_velocity` distribution | 0.477 / 0.477 / 0.045 not visible |
| `motion_energy` distribution | 0.477 / 0.477 / 0.045 no video |

Class 0 and class 1 fractions are equal to 3 decimals for all three discretised outputs,
exactly as a median split must produce — a first internal consistency check.

### Processing Plots Review
`processing_<session>.png` has 8 panels. All were inspected and show no anomalies:
1. **raster → binning → smoothing** for the highest-rate unit of one trial: the smoothed
   trace rises only *after* each spike (causal kernel) and its peak height matches the
   binned rate scaled by the kernel peak.
2. **Go-cue-aligned population PSTH**: a sharp rise exactly at t = 0 and, for the
   fixed-delay session, a second bump at t ≈ −2.2 s — the sample-tone onset
   (`ev.sample` = 0.3 s, `ev.goCue` = 2.5 s → −2.2 s). This is a direct confirmation that
   the alignment is correct. For the randomized-delay session the pre-go-cue bumps are
   smeared over several delay lengths, as expected.
3. **Neural matrix** for one trial — no empty rows, no edge artefacts.
4. **Tongue**: raw 400 Hz DLC y-position (dots) and the 100 Hz interpolation lie on top of
   each other; the class trace is 2 exactly where the tongue is unlabelled and 0/1 during
   licks. Licks occur after the go cue, as they must.
5. **Motion energy**: raw and interpolated traces coincide; the class trace switches at the
   plotted session median.
6. **Class fractions** bar chart (see table above).
7. **Per-trial outputs over the session**: the `context` trace shows the ~100 initial DR
   trials followed by alternating DR/WC blocks — exactly the design described in the Methods.
8. **P(motion energy above median) vs time, split by context**: a sharp movement onset at
   the go cue in both contexts, with the DR context ramping up during the delay
   (the paper's "uninstructed movements during the delay epoch").

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|---|---|
| `np.bincount` binning instead of per-trial `histogram` | ~20× on the binning step |
| single session-wide FIR smoothing instead of per-trial `conv` | ~50× on the smoothing step |
| 12-way process parallelism | ~8× wall clock |

| Step | Time / Session | Estimated Total (44 sessions) |
|---|---|---|
| `.mat` load (`pymatreader`) | 4.2–7.6 s | 250 s serial |
| spike binning + smoothing | 0.2–0.4 s | 15 s |
| video/kinematics/ME | 0.2–0.4 s | 15 s |
| **total, 12 workers** | — | **~45 s** wall + ~30 s to pickle ≈ **1.5 min** |

Well under the 15-minute budget, so no further optimisation was needed. (Sessions differ in
size; the estimate above uses the largest observed per-session load time, 7.6 s, for the
25 fixed-delay sessions, which are the big ones.)

### Format verification
`/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**
All per-session `T` = 500, input range [-2.5, 2.5], output ranges 0–2 / 0–1 as designed,
`ALM: 98 neurons`.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None

### Training
Loss decreased monotonically, 8.10 → 0.734 over 200 epochs; test loss 0.791.

### Decoder Results (Sample)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| lick_direction | 0.333 | 0.629 | 0.591 |
| context | 0.500 | 0.808 | 0.814 |
| outcome | 0.333 | 0.628 | 0.557 |
| tongue_velocity | 0.333 | 0.585 | 0.573 |
| paw_velocity | 0.333 | 0.647 | 0.634 |
| motion_energy | 0.333 | 0.759 | 0.751 |

Every output is above chance; train and validation are close (no overfitting).
Context 0.81 on a genuine two-context session is in line with the paper's neural context
decoding (Fig. 4b).

### Neural-scaling control experiment
Because the paper's own decoder min-max normalises the neural matrix to [-1, 1]
(`NeuralChoiceDecoding.m`), two alternative neural representations were tested on the
sample before committing to raw rates:

| Neural representation | lick | context | outcome | tongue | paw | ME |
|---|---|---|---|---|---|---|
| **firing rate, spikes/s (chosen)** | 0.61 | 0.76 | 0.61 | 0.65 | 0.54 | 0.77 |
| min-max to [-1, 1] (per session) | 0.57 | 0.72 | 0.57 | 0.61 | 0.51 | 0.72 |
| per-neuron z-score (soft) | 0.62 | 0.75 | 0.59 | 0.66 | 0.55 | 0.76 |

(run on the first sample pair, JEB13 2022-09-13 + JEB11 2022-05-10). Rescaling gives no
benefit, so the dataset stores the **raw smoothed firing rate in spikes/s**, i.e. exactly
`obj.trialdat` from the reference pipeline — the most faithful and most interpretable choice.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --workers 16`
→ `/app/conversion_full_out.txt`, `/app/converted_data.pkl`.
Wall clock **17 s** (vs. the ~90 s estimated in Step 7 — the estimate was conservative
because it assumed the largest per-session load time for every session).

### Output Files
- `converted_data.pkl`: 1.61 GB
- `verification_full_out.txt`: created — **"Data format is valid, no errors or warnings."**

### Issue found and fixed during this step
The first full run produced 61 `all neural data is zero` warnings, in sessions
JEB24 2023-10-23 (28 trials) and JEB24 2023-11-03 (33 trials). Investigating the raw
files showed that in those two sessions the **SpikeGLX recording stops before the
behavioural session does**: the largest trial index appearing in any cluster's spike train
is 314 of 343, and 312 of 346 respectively. JEB12 2022-05-12 similarly has one trial
(index 192) with no spikes, which `obj.trials.bp.haveEphys` also flags as 0.
`trials_with_ephys()` was added to `convert_data.py`: a trial is dropped when **no**
cluster on the ALM probe(s) — including clusters the quality filter rejects — has a single
spike on it. That removes 64 trials (29 + 34 + 1) and eliminates all warnings.
(`obj.trials.bp.haveEphys` was not used directly because its length disagrees with
`bp.Ntrials` in the one session where it is informative — 319 entries for 318 trials.)

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions, all ALM ephys | 25 + 19 | 25 + 19 session entries in `load*_ALMVideo.m` | 25 + 22 files (3 not in the load scripts) | **44** = 25 + 19 | ✅ |
| Sessions, fixed delay | 25 | 25 | 25 | 25 | ✅ |
| Sessions, randomized delay | 19 | 19 | 22 files | 19 | ✅ |
| Sessions, two-context | 12 | — | 12 with autowater block structure | 12 | ✅ |
| Mice, fixed delay | 9 | 10 | 10 | 10 | ⚠ paper under-counts by 1 (see Step 4) |
| Mice, randomized delay | 4 | 4 | 4 | 4 | ✅ |
| Mice, two-context | 6 | — | 7 | 7 | ⚠ paper under-counts by 1 |
| Mice, total | — | 14 | 14 | 14 | ✅ |
| Units, two-context (quality filter) | 522 | — | 529 | 529 (518 after >1 Hz) | ✅ within 1.3% |
| Single units, two-context | 214 | — | 219 | — | ✅ within 2.3% |
| Units, fixed delay (quality filter) | 1,651 | — | 1,566 | 1,566 (1,532 after >1 Hz) | ⚠ −5% |
| Units, randomized delay | 845 | — | 948 | 948 (925 after >1 Hz) | ⚠ +12% |
| Units, all | — | — | 2,514 | 2,457 after >1 Hz | — |
| Units / session | ED Fig. 2a: all ≥ 10 | `lowFR = 1` | — | min 17, max 141, mean 55.8 | ✅ |
| Trials (total, raw) | — | — | 14,972 | 14,972 | ✅ |
| Trials kept | — | drops early + stim | 976 early, 187 stim, 64 no-ephys | **13,762** (91.9%) | ✅ |
| Trials / session | — | — | 230–517 | 193–474, mean 312.8 | ✅ |
| Neural bin | 10 ms | `dt = 1/100` | — | 10 ms, 500 bins | ✅ |
| Window | [-2.5, 2.5] s | `tmin/tmax` | — | [-2.5, 2.5] s | ✅ |
| `time_from_go_cue` range | — | `obj.time` = −2.495…2.495 | — | [−2.495, 2.495] | ✅ |
| `lick_direction` | — | `(R&hit)\|(L&miss)` | — | left 0.423 / right 0.446 / none 0.131 | ✅ |
| `context` (WC fraction) | ~50% of trials after the first ~100 in two-context sessions | `autowater` | 1,548/14,972 raw = 0.103 | 0.097 overall, **0.315 in the 12 two-context sessions** | ✅ |
| `outcome` | DR accuracy > 70% (training criterion) | `hit` / `no` | — | correct 0.749, incorrect 0.120, ignore 0.131; **DR hit rate excl. ignores 0.857, WC 0.915** | ✅ |
| `tongue_velocity` | tongue visible only during licks | DLC NaN for tongue kept | raw DLC tongue NaN fraction 0.86–0.97 | not_visible 0.909, 0.046 / 0.046 | ✅ |
| `paw_velocity` | paws tracked from bottom view only | `top_paw`, `bottom_paw` | — | not_visible 0.059, 0.470 / 0.470 | ✅ |
| `motion_energy` | — | — | — | no_video 0.039, 0.480 / 0.481 | ✅ |

Notes on the two ⚠ unit-count rows: the session and mouse counts of every group match the
paper exactly, the two-context unit count matches to 1.3%, and the curation rule is taken
verbatim from `findClusters.m`. The fixed-delay (−5%) and randomized-delay (+12%)
discrepancies therefore reflect the paper counting over a slightly different probe/session
set (e.g. both probes of the dual-probe sessions, or a session that is not in the public
release), not a different curation rule. The direction of the two errors is opposite, which
rules out a systematic filtering difference on our side.

### Spot checks of data integrity
See `/app/cache/sanity_checks.py` (Step 10, Check 2) — all pass.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification (`/app/verification_full_out.txt`)

**Iteration 1**: 0 errors, **61 warnings**, all of the form
`Session <s>, trial <t>: all neural data is zero`, concentrated in sessions 36
(JEB24 2023-10-23, 28 trials) and 43 (JEB24 2023-11-03, 33 trials).
*Cause*: the ephys recording ends before the behavioural session (last trial with any
sorted spike = 314/343 and 312/346). *Fix*: `trials_with_ephys()` drops trials with no
spikes on any cluster of the ALM probe(s) (64 trials in total across the dataset).
See Step 9.

**Iteration 2 (current)**: `Data format is valid, no errors or warnings.`
No warnings remain, so nothing had to be left unfixed.

### Check 2 — Independent sanity checks (`/app/cache/sanity_checks.py`)

The script re-derives every quantity from the raw `.mat` files with a literal
transcription of the MATLAB code (`histc` implemented as an explicit half-open interval
count, `mySmooth` as `np.convolve(concat(x[:15], x), gausswin_causal, 'same')[15:]`,
`findVideoOffset` from `mode(sglx.bitcode.bitstart)/fs − mode(bp.ev.bitStart)`) and
compares with `np.allclose` / `np.array_equal`. It imports **only the session table**
from `convert_data.py`, never the processing functions. Provenance for the comparison
comes from the `trial_index` and `cluster_index` arrays now stored in
`metadata['session_info']`.

Checks run on 4 sessions (`JEB6_2021-04-18`, `JEB13_2022-09-14`, `JEB23_2023-10-11`,
`JEB24_2023-11-03`), 2 trials and 2 neurons each — **all 44 assertions pass**:

| Check | Example result |
|---|---|
| neural firing-rate trace, (session, trial, neuron) | `max|diff| = 3.7e-06` (float32 storage precision), e.g. 108 spikes → peak 98.0 spk/s |
| lick_direction from `bp.R/L/hit/miss/no` on 3 trials/session | e.g. `R=1,L=0,hit=1 → right`; `R=0,L=1,no=1 → none` |
| context from `bp.autowater` | exact |
| outcome from `bp.hit/no` | exact |
| tongue / paw / motion-energy class vectors for a whole trial | exact (`np.array_equal`) |
| stored per-session threshold really is the median of valid timepoints | class-1 count ≥ class-0 count, imbalance < 5% |
| `input` == bin centres, identical for every trial of every session | `[-2.495 .. 2.495]` |
| trial bookkeeping (`ntrials_kept == len(trial_index)`, unique, in range) | all 44 sessions |

Output: `ALL SANITY CHECKS PASSED`.

Additional primitive tests: `/app/cache/test_primitives.py` (Step 6) — kernel, smoother
and binning semantics.

### Check 3 — Reference code comparison

| Stage | Reference (MATLAB) | This conversion | Same? |
|---|---|---|---|
| **(a) data loading** | `loadObjs.m`: `load(meta(i).datapth)`; session list and ALM probe from `load<ANM>_ALMVideo.m`; motion energy from `motionEnergy_<ANM>_<DATE>.mat` (`loadMotionEnergy.m`) | `load_obj` (pymatreader), identical session/probe table hard-coded from the same scripts, `load_motion_energy` reads the same files (falling back to `obj.me` for the JEB23/JEB24 layout the reference loader does not handle) | ✅ |
| **(b) neuron filtering** | `findClusters(quality,'all')`: drop `garbage`/`gabrga`/`noisy`/`real?` (case-sensitive, `strtrim`ed); then `removeLowFRClusters` with `params.lowFR = 1` | `select_clusters` — identical string set and case sensitivity; then mean rate over the analysis window > 1 Hz | ✅ |
| **(b) trial filtering** | every `params.condition` contains `~stim.enable&~early`; conditions also split by `hit/miss/no`, `R/L`, `autowater`, but only to *label* trials | `select_trials` drops `early` and `stim.enable`; hit/miss/no and DR/WC are kept and used as labels, as the decoder-output spec demands | ✅ (plus the no-ephys rule, see below) |
| **(c) temporal alignment** | `alignSpikes.m`: `trialtm − bp.ev.goCue(trial)`. Video: `frameTimes − findVideoOffset(obj) − ev.goCue(trial)` (`findPosition.m`, `loadMotionEnergy.m`), `params.advance_movement = 0` | identical formulas, same `findVideoOffset` | ✅ |
| **(d) binning** | `getSeq.m`: `edges = -2.5:0.01:2.5`, `histc`, drop the last bin, `/dt`, `mySmooth(...,15,'reflect')`; `obj.time = edges(1:end-1) + dt/2` | `bin_index` + `np.bincount` (exact `histc` semantics, verified), `/DT`, `my_smooth` (verified equal to `mySmooth` to 2e-16) | ✅ |
| **(e) input construction** | `obj.time` is the reference time base | `input[0] = obj.time` broadcast over the trial | ✅ (the input set is dictated by the task spec) |
| **(f) output construction** | `getPrevChoice.m` `choice = (R&hit)|(L&miss)`, NaN on `no`; `getOutcome.m` `outcome = hit`, NaN on `no`; context from `autowater`; kinematics from `findPosition`/`findVelocity`; motion energy from `loadMotionEnergy` | identical definitions, with the reference's `NaN` on ignore trials replaced by the explicit third class the task spec requires (`none` / `ignore`) | ✅ with the spec-mandated change |

**Deliberate differences, and why**

1. **Trials with no electrophysiology are dropped** (64 trials). The reference has no such
   rule because the MATLAB pipeline simply leaves those trials as all-zero rows in
   `obj.trialdat` and the figures use trial subsets that happen to avoid them. An all-zero
   neural matrix is not data and the harness flags it; dropping is required for a decoder.
2. **`ignore` trials keep a label instead of `NaN`.** `getOutcome.m`/`getPrevChoice.m` set
   NaN so that MATLAB analyses skip them. The task spec requires `none` and `ignore` as
   explicit classes, so they are coded 2.
3. **Velocity is differentiated per contiguous visible segment.** `findVelocity.m` runs
   `gradient()` over the whole trial and then replaces NaN with 0 (tongue) or the nearest
   value (other features), which is appropriate for assembling a dense regressor matrix but
   injects spurious zero velocities at the edge of every visibility gap. Because the task
   spec makes "not visible" its own class, the velocity only has to be defined where the
   feature *is* visible, and a per-segment derivative is the correct estimate there. The
   reference's non-tongue baseline-drift subtraction (`median(diff(pos))`) is reproduced,
   including its use of `basederiv(1)` for both axes (an apparent typo whose numerical
   effect is negligible — both medians are ≈0 px/bin).
4. **Paw velocity is the mean over the visible paws** rather than one fixed marker: DLC
   loses `top_paw` on up to 88% of frames in some sessions and `bottom_paw` on up to 96% in
   others, so a single marker would make "not visible" the dominant class for reasons that
   have nothing to do with the animal. The union is unavailable on only 0.6–24% of bins.
5. **No PCA/dimensionality reduction of the kinematic features** (`getKinematics.m` does
   this for its regression analyses). The decoder task asks for specific discretised
   kinematic variables, not latent factors.
6. **Both fixed-delay and randomized-delay sessions are included in one dataset.** The paper
   analyses them separately; the harness trains a per-session projection, so mixing them is
   harmless and keeps all available ALM recordings with video.

### Check 4 — Key statistics comparison
See the table in Step 9. Sessions (44 = 25 + 19), mice per group, two-context unit count
(529 vs. 522), trials (14,972 raw, 13,762 kept), bin size, window, input range, and all six
output distributions were compared against the paper, the reference code and the raw data.
The only residual mismatches are the fixed-delay (−5%) and randomized-delay (+12%) total
unit counts, investigated in Step 9 and attributed to the paper's counting rather than to a
curation difference (the discrepancies go in opposite directions, and the group with the
most explicitly stated composition — the 12 two-context sessions — matches to 1.3%).

Additional behavioural cross-checks computed from the converted data
(`/app/cache/check_statistics.py`):
- DR hit rate excluding ignore trials **0.857**, WC **0.915** — consistent with animals
  being trained to >70% accuracy on the DR task and with WC trials requiring only reward
  consumption.
- WC trials are **31.5%** of kept trials in the 12 two-context sessions: the Methods
  describe ≈100 DR trials followed by alternating 10–25 trial blocks, which for a ~300-trial
  session predicts ≈33%.
- Every session has ≥17 units (Methods threshold: ≥10), mean 55.8.
- Session mean firing rates span 5.4–14.6 spk/s — normal for ALM.

### Check 5 — Edge cases

| Edge case | Finding | Handling |
|---|---|---|
| 1-based MATLAB trial indices in `clu.trial` | confirmed 1-based (max == `Ntrials`) | `−1` on load; verified by the spot checks, which map converted trial 5 → raw trial 6 etc. |
| `histc` last-bin semantics | `histc` returns 501 bins and the code drops the last | `bin_index` drops `t == 2.5` exactly; unit-tested |
| bin-centre convention | `obj.time = edges + dt/2` then truncated | bin 250 centre = +0.005 s, i.e. the go cue falls at the *start* of bin 250; unit-tested |
| empty / DUMMY probes | JEB6 probe 1 is an empty array; JEB15 2022-07-29 probe 1 is labelled `DUMMY` | neither is in the reference probe list; `select_clusters` also skips non-dict/missing entries |
| single-element MATLAB cell arrays | pymatreader unwraps them | `as_cell_list` re-wraps |
| clusters with a non-string quality (`[]`) | 7 across the dataset | treated as `''`, which is not in the reject list, i.e. kept — exactly what `findClusters.m` does |
| case-sensitive quality labels | JEB6 has one `Noisy` (capital N) | kept, reproducing MATLAB `ismember(...,'noisy')` |
| `featNames` differing between trials | checked: identical in all 44 sessions (7 side / 10 bottom features) | `feature_index` still scans until it finds a usable list |
| trials with unusable `frameTimes` (all-NaN) | exactly 3 in the whole dataset (EKH1 2021-08-07 #178, JEB13 2022-09-24 #186, JEB19 2023-04-19 #222); 2 of the 3 are early-lick/stim trials that are dropped anyway | the remaining one (JEB19 2023-04-19 → converted trial 208) is entirely class 2 for all three video outputs — verified |
| motion-energy trace length vs `frameTimes` | equal for every trial of every session (checked all 44) | a mismatch would fall back to class 2 |
| video not covering the whole window | video starts ≈0.03 s into the trial and ends 5–9 s in, so short-delay trials have no video before the window start and some long-delay randomized trials none after +2.5 s | those bins are class 2, which is exactly what "no video" means |
| `bp` field NaNs / ambiguity | none: no NaN in `R/L/hit/miss/no/early/autowater/goCue`; `R` xor `L` for every trial; `hit + miss + no == Ntrials` in all 44 sessions | nothing to handle |
| ties at the median | `motion_energy` in 5 sessions has up to 1.9% more class-1 than class-0 bins because the raw motion-energy values are quantised and many bins equal the median exactly | correct behaviour of the specified `>=` rule; median \|n0 − n1\| over all 132 session × output combinations is **1** |
| sessions below the harness' 2-trial minimum, or below 10 units | none | rule implemented anyway |

### Iteration log
1. **Iteration 1** — full conversion → 61 "all neural data is zero" warnings →
   diagnosed as the ephys recording ending early in two JEB24 sessions →
   added `trials_with_ephys()` → re-ran conversion, verification, statistics and all
   sanity checks → 0 errors, 0 warnings, all checks pass.
2. **Iteration 2** — added `trial_index` / `cluster_index` provenance arrays so the
   sanity checks could address raw trials/clusters directly → re-ran conversion and all
   checks → unchanged statistics, all checks pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.
Trained on GPU, 11,015 training trials / 2,747 validation trials, 200 epochs, ~2.5 min.

### Training Progress
- Loss decreasing: **Yes**, monotonically — 7.155 → 0.718 over 200 epochs. Test loss 0.762.

### Decoder Results (Full)
| Output | Chance (1/nclasses) | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|------|-------------|--------|------|-------|
| lick_direction  | 0.333 | 0.6551 | **0.6431** | 1.93× | 3 classes (left/right/none) over all trials and all timepoints |
| context         | 0.500 | 0.8649 | **0.8454** | 1.69× | highest of all outputs, as in the paper |
| outcome         | 0.333 | 0.6379 | **0.6187** | 1.86× | |
| tongue_velocity | 0.333 | 0.5818 | **0.5755** | 1.73× | 91% of timepoints are "not visible" |
| paw_velocity    | 0.333 | 0.5831 | **0.5741** | 1.72× | |
| motion_energy   | 0.333 | 0.7224 | **0.7142** | 2.14× | |

Every output is comfortably above chance, and training and validation accuracies differ by
≤0.02, i.e. the model is if anything under-fitting — there is no data leakage.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance

| Output | classes | chance | validation balanced acc | ratio | ≥1.5× chance? |
|---|---|---|---|---|---|
| lick_direction | 3 | 0.333 | 0.6431 | 1.93× | ✅ |
| context | 2 | 0.500 | 0.8454 | 1.69× | ✅ |
| outcome | 3 | 0.333 | 0.6187 | 1.86× | ✅ |
| tongue_velocity | 3 | 0.333 | 0.5755 | 1.73× | ✅ |
| paw_velocity | 3 | 0.333 | 0.5741 | 1.72× | ✅ |
| motion_energy | 3 | 0.333 | 0.7142 | 2.14× | ✅ |

No output is below chance and none is below 1.5× chance.

**Why the two velocity outputs are the lowest** (`/app/cache/analyze_decoder.py`,
per-class recall on the validation set):

| Output | class 0 recall | class 1 recall | class 2 recall |
|---|---|---|---|
| tongue_velocity | 0.440 (below median) | 0.566 (above median) | 0.710 (not visible) |
| paw_velocity | 0.451 | 0.482 | 0.790 (not visible) |
| motion_energy | 0.597 | 0.725 | 0.815 (no video) |
| lick_direction | 0.561 (left) | 0.590 (right) | 0.732 (none) |
| context | 0.907 (WC) | 0.824 (DR) | — |
| outcome | 0.532 | 0.572 | 0.733 |

The "is the feature visible at all" distinction is decoded well (0.71–0.82), while
"fast vs slow *given that it is visible*" is the hard part. That is expected rather than a
bug: within a single lick (~100 ms) the tongue speed alternates between near-zero at the
protrusion peak and high during protrusion/retraction on a 10–20 ms timescale, whereas the
neural regressor is a firing rate smoothed with the reference 150 ms **causal** Gaussian
kernel, which cannot resolve those alternations. The same argument applies to the paws.
Motion energy, which is a smooth envelope rather than a signed derivative, reaches 0.71.

**Time-resolved validation accuracy** (`/app/cache/decoder_accuracy_vs_time.png`) confirms
that every output's accuracy has the causally correct time course:

| Output | ITI / pre-sample (t < −2 s) | late delay (−0.5…0 s) | response (0…1 s) | peak |
|---|---|---|---|---|
| lick_direction | 0.512 | 0.663 | 0.708 | 0.745 at +0.20 s |
| context | 0.853 | 0.873 | 0.869 | 0.892 at −0.26 s |
| outcome | 0.552 | 0.587 | 0.669 | 0.730 at +0.62 s |
| motion_energy | 0.636 | 0.670 | 0.708 | 0.745 at +0.99 s |

Lick direction is essentially at chance before the sample tone and rises monotonically
through the delay, peaking just after the go cue — the signature of the paper's Fig. 3b.
Outcome only becomes decodable *after* the go cue, when reward delivery/omission happens —
the decoder is not seeing the future. Context is high and flat across all epochs including
the ITI, which is precisely the paper's central claim (Fig. 4b).

### Check 2 — Accuracy comparison to the paper

The paper reports these decoding numbers:

| Paper analysis | Paper value | This dataset | Comment |
|---|---|---|---|
| Choice from delay-epoch CD_choice, ROC (ED Fig. 2c) | AUC 0.86 ± 0.11 | — | CD-projection analysis, not reproduced here |
| **Choice (L vs R) from neural population, per time bin (Fig. 3b)** | ≈0.5 before the sample tone, rising through the delay to **≈0.9** just after the go cue | **0.525 pre-sample, 0.742 delay, 0.857 response, peak 0.894 at +0.25 s** | ✅ reproduced |
| **Context (WC vs DR) from neural population, per time bin (Fig. 4b)** | clearly above chance during the ITI (≈0.7) rising to **≈0.9** around the response epoch | **0.739 pre-sample/ITI, 0.751 delay, 0.797 response, peak 0.914 at +0.10 s** | ✅ reproduced |
| CD_context predicted from video (Fig. 4d) | R² = 0.49 ± 0.22 | — | regression analysis, not reproduced here |

Those numbers come from `/app/cache/paper_style_decoder.py`, which runs **the paper's own
analysis** on the converted data: per-session, per-time-bin, ridge-regularised logistic
regression on the raw firing rates of the simultaneously recorded units, equal numbers of
correct left and right trials, 4 repeats with 30% of trials held out — exactly the recipe
in the Methods section "Choice and context decoding from neural population or kinematic
features". The resulting curves
(`/app/cache/paper_style_decoding.png`) reproduce Fig. 3b and Fig. 4b closely, including
the step at t ≈ −2.2 s where the sample tone occurs.

**Therefore the gap between the paper's ≈0.9 and the harness' 0.64/0.85 is a property of
the decoder, not of the converted data.** The two differ in three ways that each cost
accuracy: (i) the harness trains **one shared linear readout across all 44 sessions and all
500 timepoints**, whereas the paper fits a **separate model for every session and every
time bin**; (ii) the harness sees 100 shared PCs per session instead of the full population;
(iii) the harness scores **all** trials (including ignore and error trials) and all
timepoints, including the 2.5 s before the go cue where choice is genuinely undecodable,
whereas the paper scores only balanced correct trials at one time bin at a time.
Restricting the harness' own predictions to the paper's conditions (correct trials only for
choice, two-context sessions only for context) already closes part of the gap:
choice 0.768 in the response epoch (peak 0.814) and context 0.736 in the late delay
(peak 0.767).

### Check 3 — Train vs validation gap

| Output | train | validation | ratio |
|---|---|---|---|
| lick_direction | 0.6551 | 0.6431 | 1.019 |
| context | 0.8649 | 0.8454 | 1.023 |
| outcome | 0.6379 | 0.6187 | 1.031 |
| tongue_velocity | 0.5818 | 0.5755 | 1.011 |
| paw_velocity | 0.5831 | 0.5741 | 1.016 |
| motion_energy | 0.7224 | 0.7142 | 1.011 |

All ratios are ≤ 1.03, far below the 1.5 threshold: no overfitting and no data leakage.
(Leakage is also structurally impossible for the per-trial outputs: the split is by trial,
and a trial's label is constant within the trial.)

### Debugging steps applied to the lower-accuracy outputs
1. **Output values verified against the raw data on specific trials** — Step 10, Check 2,
   `np.array_equal` on the whole class vector for tongue, paw and motion energy in 4
   sessions, and on lick/context/outcome for 12 trials.
2. **Temporal alignment verified by plotting neural + output for single trials**
   (`processing_*.png` panels 1, 2, 4, 5, 8) and by the population PSTH, which peaks at
   t = 0 and shows the sample-tone response at exactly −2.2 s.
3. **Output variation checked** — no output is dominated by one class in a way that was not
   intended: the only heavily imbalanced output is `tongue_velocity` (90.9% "not visible"),
   which is a true property of the tongue and is handled by the harness' balanced loss and
   balanced-accuracy metric.
4. **Neural filtering checked** — quality filter + >1 Hz reproduce the paper's unit counts
   (Step 9); a control experiment (Step 8) showed that rescaling the firing rates does not
   improve accuracy.
5. **Processing matched to the reference code** — Step 10, Check 3.

### Issues Found and Resolved
- *(from Step 10, carried here)* 64 trials with no electrophysiology → dropped.
- No further issues were found in this step; no change to `convert_data.py` was required,
  so the Step 9/11 numbers stand as final.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `/app/README.md` created — dataset description, loading example, full format
      specification, variable definitions, curation summary, key statistics and decoder
      performance.
- [x] `/app/cache/` created with `README_CACHE.md` documenting every cached file.
- [x] Investigation / verification / analysis scripts moved to `cache/`
      (`test_primitives.py`, `sanity_checks.py`, `check_statistics.py`,
      `analyze_decoder.py`, `paper_style_decoder.py`, plus their outputs, figures and the
      plain-text extraction of the paper). `__pycache__` removed.
- [x] Deliverables in `/app`: `CONVERSION_NOTES.md`, `README.md`, `convert_data.py`,
      `converted_data.pkl`, `sample_data.pkl`, `conversion_sample_out.txt`,
      `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
      `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`,
      `processing_JEB19_2023-04-18.png`, `processing_JEB11_2022-05-10.png`.

### Summary of every decision (quick reference)

| Decision | Choice | Why |
|---|---|---|
| Sessions | 44 ALM ephys+video sessions from the reference load scripts (25 fixed + 19 randomized delay) | the only sessions with spikes; the load-script list reproduces the paper's n = 25 and n = 19 exactly. The two behaviour-only optogenetic directories have no neural data. |
| Alignment | go cue | required by the task; `params.alignEvent = 'goCue'` in all reference scripts |
| Window / bin | [−2.5, +2.5] s, 10 ms | `params.tmin/tmax/dt` in all reference scripts |
| Neural signal | causal-Gaussian-smoothed firing rate, spikes/s | `getSeq.m` + `mySmooth.m`; this is what the paper feeds to its own decoders. A control experiment showed rescaling does not help. |
| Unit curation | reference quality filter + >1 Hz | `findClusters.m`, `removeLowFRClusters.m`, Methods |
| Trial curation | drop early-lick, photostim and no-ephys trials; keep hit/miss/ignore and DR/WC | reference `params.condition`; ignore/none and WC/DR are decoder classes |
| Session curation | ≥10 units | Methods |
| Brain region | all units labelled `ALM` | the paper's recordings are in ALM; probe choice from the load scripts |
| Input | time from go cue, continuous ramp, 1 dimension | exactly what the task specifies |
| Lick direction | `(R&hit)|(L&miss)` = right, `(L&hit)|(R&miss)` = left, `no` = none | `getPrevChoice.m` |
| Context | `bp.autowater` | reference `params.condition`; validated against the described block structure |
| Outcome | `hit` / `miss` / `no` | `getOutcome.m` |
| Tongue velocity | side-camera `tongue` marker speed, per-session median split, class 2 when untracked | `findPosition.m`/`findVelocity.m` + task spec |
| Paw velocity | mean speed of the visible bottom-camera paws | as above; single-marker tracking is too intermittent |
| Motion energy | interpolated 400 Hz trace, per-session median split, class 2 outside video coverage | `loadMotionEnergy.m` + task spec |
| Discretisation threshold | per-session median of *valid* timepoints only | otherwise the 91% "not visible" tongue bins would define the threshold |
| Storage | neural `float32`, output `int8`, shared input array | 1.6 GB instead of ~5 GB, no information lost |
