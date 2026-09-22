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

Environment check: Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 import successfully. The required notes-file checkpoint passed (`ls -la /app/CONVERSION_NOTES.md`).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadObjs` | `code/DataLoadingScripts/loadObjs.m` | LOADING | Loads each session MATLAB `obj` from `meta(i).datapth`; normalizes missing top-level fields for struct concatenation. |
| Animal-specific `load*_ALMVideo` functions | `code/DataLoadingScripts/Recording and video/` | LOADING/CURATION | Enumerate author-selected ephys/video sessions and the ALM probe(s); commented entries are excluded and JEB15 2022-07-29 explicitly drops unsorted probe 1. |
| `loadSessionData` / `processData` | `code/DataLoadingScripts/` | PROCESSING | Orchestrate condition selection, unit-quality selection, spike alignment/binning/smoothing, low-FR removal, and multi-probe concatenation. |
| `findTrials` | `code/DataLoadingScripts/findTrials.m` | CURATION | Evaluates Boolean trial conditions against `obj.bp` fields; preserves original 1-based trial IDs. |
| `findClusters` | `code/DataLoadingScripts/findClusters.m` | CURATION | With quality `all`, excludes labels `garbage`, typo `gabrga`, `noisy`, and `real?`; otherwise pattern-matches requested labels. |
| `alignSpikes` | `code/DataLoadingScripts/alignSpikes.m` | PROCESSING | Computes per-spike aligned time as raw within-trial spike time minus the per-trial alignment event (go cue here). |
| `getSeq` | `code/DataLoadingScripts/getSeq.m` | PROCESSING | Builds single-trial firing rates in half-open bins from `tmin` to `tmax`, divides counts by `dt`, and causally Gaussian-smooths time. |
| `mySmooth` | `code/utils/mySmooth.m` | PROCESSING | Uses MATLAB `gausswin(N)`, zeros the first half to make the kernel causal, normalizes it, convolves in `same` mode; optional leading reflect/zero padding. |
| `removeLowFRClusters` | `code/DataLoadingScripts/removeLowFRClusters.m` | CURATION | Keeps units whose mean condition-PSTH rate is strictly above `params.lowFR`. |
| `findVideoOffset` | `code/funcs/findVideoOffset.m` | PROCESSING | Computes camera/ephys offset as modal SpikeGLX bit-start time minus modal behavioral bit-start time. |
| `findPosition` / `getKinematicsFromVideo` / `findVelocity` | `code/funcs/kinematics/` | PROCESSING | Align DLC x/y trajectories to the neural time axis via interpolation; derive x/y frame-gradient velocities; retain tongue invisibility masks and fill other-feature gaps. |
| `loadMotionEnergy` | `code/DataLoadingScripts/loadMotionEnergy.m` | LOADING/PROCESSING | Loads session motion-energy files, applies camera/ephys offset, aligns/interpolates to neural time, nearest-fills edge NaNs, and thresholds movement. |
| `getOutcome` | `code/funcs/getOutcome.m` | PROCESSING | Encodes hit as 1, miss as 0, and ignore (`bp.no`) as NaN in the reference analysis. |
| `getBlockNum_AltContextTask` | `code/funcs/fig8/getBlockNum_AltContextTask.m` | PROCESSING | Treats `bp.autowater` as the WC/DR context indicator and finds context-switch blocks. |
| `NeuralChoiceDecoding` / `NeuralContextDecoding` | `code/ChoiceContextDecoding/` | PROCESSING | Decode balanced choice/context labels from 75-ms averages of the aligned single-trial neural matrix. |

### Notes
- This is extracellular electrophysiology, not calcium imaging; delta-F/F is not applicable. Neural source data are sorted spike times in `obj.clu{probe}(cluster)`.
- The tutorial states qualities `excellent`, `great`, and `good` are single units, while other labels may be multi-units/excluded. The actual default/reference pipeline uses `quality={'all'}`, whose implemented exclusions are listed above, then low-rate filtering.
- Default pipeline parameters are go-cue alignment, `tmin=-2.5 s`, `tmax=+2.5 s`, `dt=1/200 s` (5 ms), causal Gaussian smoothing with `N=15`, and `lowFR=0.5 Hz`. `WorkingWithDataObjs.m` contains an illustrative variant at 10 ms and 1 Hz, so the default function is the stronger reusable specification pending comparison with paper/data.
- `getSeq` time centers are `tmin + dt/2` through `tmax - dt/2`; therefore 1000 time points at 5 ms over `[-2.5, 2.5)`.
- Reference condition definitions normally exclude stimulation and early trials and distinguish DR (`~autowater`) from WC (`autowater`). For this requested decoder, all three outcome classes are needed, so the eventual trial filter must be justified after paper/data comparison rather than copying hit-only decoder subsets.
- Video alignment uses `frameTimes - vidshift - goCue`; when timestamps are unavailable the code assumes 400 Hz and (in several behavior helpers) a 0.5-s offset. DLC time series are interpolated onto the same aligned time centers as neural data.
- Kinematic velocity in reference code is the per-frame gradient in pixels/sample, not explicitly rescaled to pixels/s. Tongue missing points are recorded before baseline filling, allowing a distinct `not visible` class as requested. Paw gaps/dropout must likewise be derived from raw visibility/trajectory validity before non-tongue nearest filling.
- No neural-region inference is necessary: README and selected probe metadata identify the retained recordings as ALM (right and/or left anterior lateral motor cortex); dual ALM probes are concatenated within session.
- `UseInclusionCritera.m` exists and requires >40 right-hit and >40 left-hit trials, but it is not called by the central loader shown in the tutorial. Whether it applies to the supplied subset will be resolved against the paper, scripts, and data in Steps 2–4.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is 16 GB and contains four experiment folders. All session files are MATLAB `data_structure_<animal>_<YYYY-MM-DD>.mat`; neural-bearing sessions have paired `motionEnergy_<animal>_<date>.mat` files.
- `Ephys_Behavior/`: 25 go-cue/task ephys+video sessions from 10 mice, all MATLAB v7.3/HDF5, plus 25 MATLAB-v5 motion-energy files. This is the context dataset matching the required WC/DR output: 1,449/8,260 trials have `autowater=1` (WC), while DR is `autowater=0`.
- `RandomizedDelay_Ephys_Behavior/`: 22 session objects from 4 mice (mixed MATLAB v7.3 and v5); 20 contain neural clusters and 2 JEB24 pilot files contain no `obj.clu`. It has only 99/7,583 WC-labeled trials, so it is principally a randomized-delay DR dataset rather than the alternating-context dataset. It will be assessed against the paper in Steps 3–4 before final inclusion.
- `DelayInhibition_BilatMC_Behavior/`: 53 behavior/video-only sessions from 4 MAH mice; no electrophysiology and thus cannot supply the required decoder neural input.
- `GoCueInhibition_BilatMC_Behavior/`: 20 behavior/video-only sessions from the same 4 MAH mice; likewise ineligible for a neural-input decoder.
- Within an ephys session, top-level `obj` fields include `bp` (task/trial labels), `bp.ev` (event/lick times), `bp.stim`, `sglx` (recording/bitcode synchronization), `clu` (cell array of probe-specific cluster structs), `traj` (side/bottom camera DLC trajectories), `trials`, and metadata (`meta` or `ex`).
- Each cluster contains session spike times `tm`, within-trial spike times `trialtm`, 1-based spike trial IDs `trial`, `quality`, electrode site, and waveforms. These are variable-length float64 vectors stored via MATLAB object references.
- `obj.bp` has length-`Ntrials` float64/logical arrays: `R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, and events including `goCue`, `sample`, `delay`, `reward`, `lickL`, and `lickR`. Every static-context session has a finite go cue for every trial. Outcomes are mutually represented by hit/miss/no in the inspected totals.
- `obj.traj` contains two views and one struct per trial. Side-camera features are consistently `tongue`, `left_tongue`, `right_tongue`, `jaw`, `trident`, `nose`, `lickport`; bottom-camera features are consistently four tongue landmarks, `top_paw`, `bottom_paw`, `lickport`, `jaw`, and two nostril landmarks. Each `ts` is stored as `(features, x/y/confidence, frames)` in HDF5 (MATLAB view is frames × 3 × features), with matching `frameTimes` and `NdroppedFrames`.
- Both camera structs exist for all 8,260 static-context trials; all `NdroppedFrames` entries are finite. The actual frame interval is approximately 0.00251994–0.00252000 s (about 396.83 Hz). Individual DLC landmarks can still be NaN/not visible and require mask-aware handling.
- Paired motion-energy files contain `me.data`, a length-`Ntrials` object array of variable-length float64 traces, and an author threshold `moveThresh`. All 8,260 static-context trials have nonempty finite traces; trace lengths range from 1,711 to 8,138 samples and stored thresholds range 7.5–15. The requested decoder nevertheless specifies a session median threshold, not `moveThresh`.
- Static-context raw behavior totals are 5,711 hit, 1,168 miss, 1,381 no/ignore, 661 early, 187 stimulation-enabled, and 1,449 WC trials across 8,260 trials. Trial filtering remains a Step 3–5 decision.
- Static-context cluster inventory is 11,158 across every stored probe. Restricting to the author-selected probe(s) from the session metadata scripts gives 7,241 raw clusters; applying the implemented quality exclusions but not yet the low-FR criterion gives 1,565 clusters. Large garbage-cluster counts explain this difference. The all-probe quality inventory is: 8,790 garbage, 720 multi, 565 poor, 493 fair, 198 dummy `[0 0]`, 195 good, 138 great, 50 excellent, 5 unlabeled, and one each `ood`, `gabrga`, `real?`, `noisy`.
- Randomized-delay inventory: 7,583 total trials (7,064 in the 20 neural-bearing files), 3,139 raw stored clusters, and 998 passing the reference quality-label exclusions before low-FR filtering. Raw outcome totals are 6,146 hit, 814 miss, and 623 no/ignore.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | Static-context author-selected probes: 7,241 raw clusters; 1,565 quality-eligible before low-FR filtering |
| Neurons / session | Raw selected mean 289.6/session; quality-eligible mean 62.6/session (final after low-FR TBD) |
| Subjects | Static-context ephys: 10 (`EKH1`, `EKH3`, `JEB13`, `JEB14`, `JEB15`, `JEB19`, `JEB6`, `JEB7`, `JGR2`, `JGR3`) |
| Sessions / subject | 1, 1, 5, 4, 4, 4, 1, 2, 2, 1 respectively (25 total) |
| Trials (total) | Static-context ephys: 8,260 raw trials |
| Trials / session | Mean 330.4; range 230–517 |

Native-data checkpoint: file organization, formats, hierarchies, variables, data types/dimensions, and raw totals have been inspected directly. Behavior-only sources are documented but are not candidates for a decoder that requires neural activity.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | DR/fixed-delay: 1,651 units, 483 well-isolated single units; two-context subset: 522 units, 214 single units; randomized-delay: 845 units, 288 single units | Methods, electrophysiology recording analysis |
| Neurons / session | DR/fixed-delay mean 66.0; two-context mean 43.5; randomized-delay mean 44.5 (computed from stated totals) | Paper states totals and session counts |
| Subjects | DR/fixed-delay 9; two-context 6; randomized-delay 4 | “25 sessions using nine mice”; “12 sessions from six mice”; “19 sessions using four mice” |
| Sessions / subject | DR/fixed-delay 25/9; two-context 12/6; randomized-delay 19/4 | Methods |
| Trials (total) | Not stated as a total | — |
| Trials / session | Not stated; behavioral inclusion demanded sufficient balanced correct trials | At least 40 correct DR trials/direction and 20 correct WC trials/direction for behavioral analyses |
| Neural data time bin | 5 ms | “single-trial neural activity was first binned in 5-ms intervals” |
| Behavior data time bin | Native high-speed video 400 Hz (~2.5 ms); aligned analyses use neural bins | “High-speed video was captured (400-Hz frame rate)” |
| Reward rate | Mice reached at least 70% DR accuracy before advancing; no aggregate reward rate reported | Mouse behavior methods |
| Two-context task | Alternating blocks: session begins with ~100 DR trials, followed by WC/DR blocks of 10–25 trials | Mouse behavior methods |
| Ignore definition | No response within 3 s of go cue; responses usually within 300 ms | Mouse behavior methods |
| Choice selectivity | 36% sample, 42% delay, 58% response among 483 single units | Results |
| Context selectivity | 39% of 214 single units during ITI among 12 sessions | Results |


### Processing Details
- The decoder-requested alignment is exactly the paper/code convention: go cue onset (or water-drop onset for WC, stored in the same go-cue event field). The fixed analysis window in reference code is −2.5 to +2.5 s.
- Single-trial spiking is binned at 5 ms and smoothed with a causal Gaussian kernel with 35-ms half width. The code implementation’s 15-sample (75-ms full window) causal Gaussian is consistent with this description.
- The paper baseline-standardizes some downstream subspace analyses using −2.4 to −2.2 s. That normalization is analysis-specific; raw smoothed firing rates are the appropriate neural decoder input because the supplied decoder handles its own normalization/training.
- Video is acquired from side and bottom cameras at 400 Hz. DLC x/y positions are extracted; missing values are nearest-filled except tongue; velocity is the first derivative of position. Tongue visibility must therefore be preserved before any fill operation to implement the requested class 2.
- Motion energy is the per-frame 99th percentile of pixelwise absolute differences between five-frame future and past medians. The paper’s movement/no-movement threshold was manually selected per session, but this task explicitly overrides it with the per-session 50th percentile.
- Paper choice/context decoders used time-resolved ridge-logistic regression, four-fold cross-validation, 30% held-out trials, balanced correct left/right trials, and shuffled labels for chance. This differs from the supplied neural decoder architecture but supplies qualitative accuracy expectations.

### Curation Steps

**Neuron curation rules**:
- Spike sorting used JRCLUST and/or Kilosort 3 with manual Phy curation. Well-isolated single units were defined by ISI, separation, and session stationarity; curated units with higher ISI violations were multiunits.
- Sessions required at least 10 units.
- All paper analyses other than specified subspace/single-unit analyses used all units with firing rate >1 Hz. Selected subspace alignment and single-unit selectivity used only well-isolated single units with firing rate >1 Hz.
- For this broad neural decoding task, the applicable paper rule is all curated units >1 Hz, not single-units only. This agrees with the tutorial’s conceptual quality handling, though code default thresholds vary and are reconciled in Step 4.

**Trial curation rules**:
- Paper analyses omit early-lick trials. Behavioral analyses also omit ignore trials, while single-trial neural subspace analysis explicitly uses all correct and error trials from DR and WC unless stated otherwise.
- Reference condition code also removes stimulation-enabled trials. Thus the general decoder trial pool should exclude early/stimulation trials while retaining correct and incorrect trials.
- The requested output explicitly includes `ignore`, so ignore trials must be retained despite their omission from paper analyses; this is a necessary task-specific exception. They also provide the required `none` lick-direction class.
- The requested behavioral context output makes the fixed/two-context dataset the applicable source. Randomized-delay sessions are a distinct task and nearly lack WC examples in the files, so including them would change the task and heavily distort context classes.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from delay-epoch CDchoice | ROC AUC 0.86 ± 0.11 across sessions (exact text) |
| Choice from neural population, time-resolved | Figure 3b: approximately 0.65–0.8 during sample/delay, peaking ~0.88–0.90 just after go cue; shuffled ~0.5 (read from plotted curve, not a tabulated statistic) |
| Context from neural population, time-resolved | Figure 4b: approximately 0.78–0.85 before go cue/water drop, peaking ~0.9 just after alignment; shuffled ~0.5 (read from plotted curve) |
| Choice/context from kinematics | Choice peaks ~0.95 and context peaks ~0.9 in the plotted four-fold-CV curves; these are not neural-input decoder targets here |

The paper reports no decoding accuracy for outcome, tongue velocity, paw velocity, or motion-energy classes; those are new downstream tasks. Rendered figure pages used for visual reading are cached under `/app/cache/` and will be documented during cleanup.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Applicable sessions | Figure 8 context scripts load JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19: 12 sessions | Exactly those 12 have substantial WC blocks; DR-only JEB13/14/15 and randomized-delay files do not represent the requested balanced context task | Two-context paradigm: 12 sessions | Use exactly the 12 sessions enumerated by Figure 8 context code. Exclude DR-only, randomized-delay, and behavior-only sessions. |
| Subject count | Context loader names contain 7 unique IDs | Seven unique file/metadata IDs | Paper says six mice | Preserve the seven explicit source IDs in `subjects`; collapsing distinct IDs would fabricate identity. Treat the one-mouse difference as a paper/release-version reporting discrepancy. |
| Context unit count | Selected probes + `quality='all'` exclusions + `lowFR=1` | Selected 12 sessions have 528 quality-eligible units and 219 label-defined single units before FR filtering. A direct recreation of the code-style PSTH criterion retains 521 units, including exactly 214 single units. | 522 units, 214 single units | The single-unit count matches exactly and the total differs by only one unit, consistent with minor release/numerical differences. Use the executable >1-Hz rule on the released files and report the resulting count. |
| Fixed-delay inventory | All fixed-delay scripts enumerate 25 sessions | 25 source files, but 10 unique file IDs | 25 sessions, nine mice, 1,651 units/483 singles | Not the decoder cohort: only the 12-session two-context subset is relevant. Preserve actual IDs instead of forcing paper-wide counts. |
| Randomized-delay inventory | Load scripts enumerate 19 sessions; JEB23 2023-10-20 and JEB24 pilots are omitted | 20 neural-bearing files plus two no-neural pilots | 19 sessions, four mice, 845 units | Exact agreement after author curation confirms that animal-specific loader scripts define session inclusion. This set is excluded because it is a distinct randomized-delay task with almost no WC trials. |
| Firing-rate threshold | Most figure scripts use `lowFR=1`; only generic defaults use 0.5 | Rates span both thresholds; the 12 context sessions have 528 units before FR filtering | All included units exceed 1 Hz | Use strict `>1 Hz`; figure scripts and paper override the generic default. |
| Neural bin size | Some illustrative scripts use 10 ms; generic defaults and several figures use 5 ms | Spike times have sufficient precision | Single-trial analysis explicitly uses 5 ms | Use 5-ms bins. |
| Neural smoothing | `mySmooth` uses MATLAB `gausswin(15)`, zeros its first seven coefficients, normalizes, and convolves causally | Raw spike times | Causal Gaussian half width 35 ms | Reproduce the 15-bin 5-ms causal Gaussian exactly (effective history about 35 ms), with the reference leading boundary handling. |
| Analysis window | Context figure scripts sometimes start at −3 s; generic loader/default and primary single-trial workflows use −2.5 to +2.5 s | All selected trials have go cues; video covers the target interval with edge handling | Figures primarily display around −2 to +2 s; baseline is −2.4 to −2.2 s | Use bin centers over `[−2.5,+2.5)` (1000 bins), matching `getDefaultParams/getSeq` and retaining the paper baseline period. |
| Trial curation | Conditions reject stimulation and usually `early`; neural subspace workflows use correct/error trials | Selected files include early, stimulation, and ignore flags | Early trials omitted; most paper analyses omit ignores | Exclude stimulation and early trials. Retain hit, miss, and no/ignore because the requested output explicitly requires all three outcomes and `none` lick direction. |
| Video timing | `frameTimes - findVideoOffset(obj) - goCue`, then interpolate to neural time | Computed offset is 0.49004 s in older sessions and 0.9900159 s for JEB19; fixed 0.5 s would misalign JEB19 | Two synchronized ~400-Hz cameras | Use per-session bitcode-derived offset, never a universal 0.5-s shift. |
| Kinematic missingness | Reference nearest-fills non-tongue positions and retains tongue missingness separately | Both cameras/trials exist, but landmarks contain visibility gaps | Missing values nearest-filled except tongue; tongue visibility analyzed explicitly | Save visibility masks before any filling. Use class 2 at missing tongue/paw timepoints as required; fill only to calculate visible neighboring velocities consistently. |
| Motion-energy threshold | Paper/code manual bimodal threshold | Stored `moveThresh` 7.5–15 | Manual per-session threshold | Decoder specification overrides this: use the median of aligned valid motion energy per session. |
| Context encoding | `autowater=1` is WC; 0 is DR | WC blocks visible in the 12 sessions | WC omits tones and presents water; DR is instructed | Encode `[WC, DR]` as `[0,1]`. |
| Lick direction | Reference decoders sometimes use instructed R/L conditions and correct-only trials | `R/L`, hit/miss/no jointly determine actual response | Task distinguishes instructed directional response and ignore | Required “lick direction” is actual response: hit uses instructed side, miss uses opposite side, no/ignore maps to none. |

Final coherent interpretation: convert the author-curated 12-session two-context ALM cohort, retain the task-required outcomes, and otherwise reproduce the paper’s 5-ms go-cue-aligned neural and video processing. The near-exact 521-versus-522 unit and exact 214-single-unit recreation is the primary cross-source sanity check.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Selected `obj.clu{probe}.trialtm`, `.trial`, `.quality` | `neural` | Select author-listed probe(s), exclude `garbage/gabrga/noisy/real?`, apply strict reference `>1 Hz` filter, align each spike by that trial’s `bp.ev.goCue`, bin `[−2.5,2.5)` at 5 ms, divide by 0.005 s, causal-Gaussian smooth, transpose to neuron × time, float32 | `findClusters`, `alignSpikes`, `getSeq`, `mySmooth`, `removeLowFRClusters` | Expected 12 sessions and about 521 units; retain smoothed firing rates, not binary spikes. |
| Bin centers `−2.5+dt/2 : dt : 2.5−dt/2` | `input[0]` | One-row continuous float32 time series, shape `(1,1000)`, identical across trials | `getSeq` (`obj.time`) | Name: `time_from_go_cue_s`; explicitly continuous per task, not an onset indicator. |
| `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, `bp.no` | `output[0]` | Actual response direction: hit→instructed/reward side, miss→opposite side, no→none; broadcast across time | Trial-condition logic / requested task | Values `[left,right,none] = [0,1,2]`. |
| `bp.autowater` | `output[1]` | `1` (water-cued)→WC 0; `0`→DR 1; broadcast across time | `getBlockNum_AltContextTask`, context decoders | Values `[WC,DR] = [0,1]`. |
| `bp.miss`, `bp.hit`, `bp.no` | `output[2]` | Mutually exclusive categorical trial outcome; broadcast across time | `getOutcome` plus task-required explicit ignore class | Values `[incorrect,correct,ignore] = [0,1,2]`. |
| Side-view `traj.ts[:,:,tongue]`, `traj.frameTimes`, sync fields | `output[3]` | Convert stored frames×(x,y,confidence)×feature orientation; align with `frameTimes − video_offset − goCue`; interpolate x/y; compute Euclidean magnitude of x/y first derivatives on 5-ms grid; threshold visible values at session median; missing landmark→2 | `findVideoOffset`, `findPosition`, `findVelocity`, `getKinematicsFromVideo` | Values `[below_median,at_or_above_median,not_visible] = [0,1,2]`; time-varying. Main side-camera `tongue` landmark is the paper’s tongue-tip representation. |
| Bottom-view `traj.ts[:,:,top_paw]`, `traj.frameTimes`, sync fields | `output[4]` | Same alignment; preserve raw/interpolated visibility mask before reference-style nearest filling; derive Euclidean first-derivative speed; threshold visible values at session median; missing→2 | Same kinematics functions | Values `[below_median,at_or_above_median,not_visible]`; `top_paw` matches paper figure code’s paw feature. |
| Paired `motionEnergy_*.mat: me.data` and side-camera frame times | `output[5]` | Unwrap legacy nested `me.data` if needed; align/interpolate using the same per-session video offset/go cue; nearest-fill only valid-trace edges as reference; threshold aligned valid values at session median; absent trace/video→2 | `loadMotionEnergy` | Values `[below_median,at_or_above_median,no_video]`; requested median overrides stored manual `moveThresh`. |
| Session animal prefix | `subjects`, `subject_idx` | Preserve seven distinct source identifiers and index sessions deterministically | Animal-specific loader metadata | Session order follows the Figure 8 loader order then date: JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19. |
| Author-selected probe and paper recording site | `brain_regions`, `brain_region_idx` | Single region `ALM`; one zero integer per retained neuron | `load*_ALMVideo` metadata; paper | Context cohort is the ALM cohort. |
| Processing/session provenance | `metadata` | Store window/bin/smoothing/filter rules, raw and retained trial/unit counts, thresholds, source filenames, probe IDs, video offsets, and trial indices | All above | Enables exact audit and raw spot checks. |

### Key Decisions
1. **Cohort is the 12-session two-context set**: It exactly matches Figure 8’s executable session list and the paper’s session count, and supplies both WC/DR classes. DR-only, randomized-delay, and behavior-only data are not the requested decoding experiment.
2. **Trials exclude `early` or `stim.enable`, but retain hit/miss/no**: This follows paper/code quality curation while the explicit output contract requires incorrect and ignore classes. Direct pre-conversion expectation is 3,116 trials: 2,086 hit, 329 miss, 701 no; 982 WC and 2,134 DR.
3. **Rate filtering reproduces figure code, not generic defaults**: Apply quality exclusions then the context-workflow `lowFR=1` logic. A standalone code-style recreation yielded 521 units/214 label-defined singles versus paper 522/214.
4. **5-ms, 5-s window**: 1000 shared bin centers from −2.4975 to +2.4975 s. This is the paper’s single-trial binning and the generic reference pipeline’s go-cue window.
5. **Exact causal kernel**: MATLAB `gausswin(15)` corresponds to a symmetric Gaussian with standard deviation `(15−1)/(2×2.5)=2.8` samples; zero coefficients 0–6, renormalize, convolve in `same` mode after the reference leading 15-bin padding, then crop the padding.
6. **All six outputs use a single `(6,T)` int8 matrix**: Decoder code permits per-trial 1-D labels but mixed static/time-varying outputs cannot share one array. Broadcasting lick/context/outcome across T preserves their per-trial semantics; velocity/energy remain genuinely time-varying.
7. **Velocity means speed magnitude**: Median-thresholding a signed component would classify direction rather than velocity magnitude. Euclidean first-derivative magnitude is the physically meaningful scalar consistent with the paper’s “speed” plots while retaining the reference derivative-before-standardization procedure.
8. **Visibility precedes filling**: Use finite raw/interpolated x/y to assign class 2, then fill paw positions only for computing speed. This is required to avoid erasing `not visible`; tongue remains unfilled as in the paper.
9. **Session percentile is computed over retained trials and visible/valid aligned timepoints**: This defines each converted session internally, yields ~50/50 classes 0/1 among visible samples, and excludes class-2 samples from the percentile.
10. **Per-session bitcode video offsets**: Use `mode/median(bp.ev.bitStart)` and `mode/median(sglx.bitcode.bitstart)/fs`; offsets are ~0.49004 s for older sessions and ~0.990016 s for JEB19. A fixed 0.5-s correction is demonstrably wrong for JEB19.
11. **Actual lick rather than instructed side**: Miss trials invert R/L; ignores map to none. This implements the named output and avoids silently decoding stimulus side.
12. **Dtypes**: neural/input float32, outputs int8, indices integer. No NaN/Inf is permitted in target arrays.

### Available Source Variables Inventory
- Trial/task: `Ntrials`, instructed/reward side `R/L`, `hit`, `miss`, `no`, `early`, `bitRand`, `autowater`, `protocol`, stimulation `enable/num/wavParams`.
- Events: `bitStart`, `sample`, `delay`, `goCue`, `reward`, variable-length `lickL`, `lickR`.
- Neural/synchronization: per-cluster `tm`, `trialtm`, `trial`, `quality`, `site`, `spkWavs`; SpikeGLX sampling/bitcode/camera-trigger metadata.
- Video: side/bottom x, y, confidence trajectories for all listed tongue, jaw, nose/nostril, paw, trident, and lickport landmarks; frame times and dropped-frame counts.
- Motion: per-frame motion energy plus the authors’ manual movement threshold.
- Only variables mandated by the Decoder Task become input/output dimensions; remaining variables are used for curation, alignment, derivation, and provenance rather than adding unrequested decoder targets.

### Planned Sanity Checks
- [x] Exact raw-to-converted neural spot check with `np.allclose`: independently bin/smooth selected unit/trial/time bins from `trialtm − goCue`.
- [x] Exact input check with `np.allclose`: every trial input equals the analytically generated 1000 bin centers and contains zero between the two central bins as expected for bin centers.
- [x] Exact static-output checks on at least hit, miss, and no raw trials using direct `bp` arrays and `np.allclose` after broadcast.
- [x] Exact video alignment check with `np.allclose`: independently interpolate known raw tongue/paw frames using `frameTimes − offset − goCue` at selected timepoints.
- [x] Exact motion-energy alignment check against the paired original file with `np.allclose` at selected timepoints.
- [x] Confirm all retained raw trial IDs satisfy `~early & ~stim.enable`; no trial lost beyond this rule.
- [x] Confirm 12 sessions, seven explicit subject IDs, ≥2 trials/session, roughly 521 units with ≥10/session, and approximately 3,116 trials.
- [x] Confirm neural shape `(n_units,1000)`, input `(1,1000)`, output `(6,1000)` for every trial; finite values and consistent dtypes.
- [x] Confirm output ranges/classes match `output_values`; visible speed/valid energy class fractions are approximately 0.5/0.5 within every session.
- [x] Confirm unit/single-unit totals against paper (522/214 expected; release reproduction ~521/214) and explain any one-unit numerical difference.
- [x] Plot raw/aligned trajectories, velocity, median thresholds, neural rasters/rates, and class traces for up to two sessions in `--show-processing` mode.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `/app/convert_data.py` with required positional output path and `--full` (default), `--sample`, and `--show-processing` modes.
- Uses direct HDF5 field/reference access rather than materializing entire v7.3 MATLAB structs. Paired small v5 motion-energy files are read with SciPy.
- Implements the exact 12-session Figure 8 order and selected probe IDs, direct trial/outcome validation, paper/reference quality and firing-rate filters, vectorized spike accumulation, exact causal smoothing, per-session bitcode video offset, aligned DLC/motion interpolation, visibility preservation, percentile discretization, metadata, internal shape/dtype/range checks, and eight-panel processing figures.
- Syntax/help tests pass. A direct full conversion of JEB6 2021-04-18 completed without errors in 3.89 s: 382→302 trials, 32→31 units, trial shapes neural `(31,1000)`, input `(1,1000)`, output `(6,1000)`.
- The causal smoother was numerically compared to explicit `np.convolve` and matched to float32 precision (maximum absolute difference `9.54e-7`).

Code inefficiencies identified:
- Nested MATLAB references require per-cluster/per-trial reads; naive full `mat73` loading was stopped because it materialized the complete nested file and was slow/memory-heavy.
- DLC arrays are variable-length and therefore require a trial loop, but all transformations within a trace are vectorized.

Code speedups added:
- Field-level HDF5 reads, vectorized `np.add.at` spike binning, one session-sized float32 neural tensor, and `scipy.signal.lfilter` for the short causal kernel.
- Estimated full conversion from the measured first session is under one minute plus pickle I/O, well below the 15-minute optimization threshold.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 97 (31, 66 by session) |
| Neurons / session | 48.5 mean; range 31-66 |
| Subjects | 2 (`JEB6`, `JEB7`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 547 |
| Trials / session | 273.5 mean; 302, 245 |
| Time-from-go input range | [-2.4975, 2.4975] s |
| Lick distribution | [0.4095 left, 0.4479 right, 0.1426 none] |
| Context distribution | [0.3291 WC, 0.6709 DR] |
| Outcome distribution | [0.1188 incorrect, 0.7386 correct, 0.1426 ignore] |
| Tongue velocity distribution | [0.0469 below, 0.0469 high, 0.9063 not visible] |
| Paw velocity distribution | [0.4781 below, 0.4781 high, 0.0439 not visible] |
| Motion-energy distribution | [0.4998 below, 0.5002 high, 0 no-video] |

### Processing Plots Review
- Inspected both eight-panel figures at original resolution. Go cue is at zero in neural, position, velocity, and motion panels; aligned traces have no discontinuity at zero. Motion increases after go as expected. DLC gaps remain class 2 rather than being silently imputed for the output. The neural histogram shows the 1-Hz cutoff, and the categorical plots match retained-trial counts.
- Tongue visibility is only about 9.4%, but this is expected: the source tracker stores tongue coordinates mainly during protrusions. Paw visibility is 95.4-96.0%; motion energy is available throughout both sessions.
- Direct raw-file checks (first/middle/last retained trial per session) reproduced trial retention, lick direction, and outcome exactly. Every input vector matched the analytical 5-ms bin centers. Neural shapes matched region-vector lengths. Among visible samples, all three median-split outputs were 50/50 to numerical/tie precision.
- `train_decoder.py --verify-only` reported no errors or warnings. Its generated summary confirmed all fields, shapes, dtypes, ranges, and categorical values.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|---|---|
| Direct field-level HDF5 access and vectorized spike binning/smoothing | Sample completed in 9.13 s total; avoided full nested MATLAB-object materialization |

| Step | Time / Session | Estimated Total Time |
|---|---:|---:|
| Conversion compute (measured session logs) | 3.87 s | 46.4 s for 12 sessions |
| End-to-end including plots and pickle I/O | 4.57 s | about 55 s (conservative: under 2 min) |

Artifacts: `/app/sample_data.pkl` (102.7 MiB), `/app/conversion_sample_out.txt`, `/app/verification_sample_out.txt`, and two `processing_*.png` files.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Training used CUDA, 437 training trials and 110 validation trials. Loss decreased monotonically from 8.4158 (epoch 1) to 0.7791 (epoch 200); test loss was 0.8101.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction | 0.5857 | 0.5761 |
| Behavioral context | 0.7491 | 0.7263 |
| Outcome | 0.5989 | 0.5901 |
| Tongue velocity | 0.5850 | 0.5929 |
| Paw velocity | 0.4472 | 0.4409 |
| Motion energy | 0.7728 | 0.7779 |

All validation scores exceed the validator's displayed chance levels (1/3 for the five three-label specifications and 1/2 for context). Motion energy has only two observed classes in this cohort, so its empirical uniform chance is 1/2; its validation score also exceeds that stricter benchmark. The close train/validation scores provide no evidence of overfitting or leakage.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 591,599,569 bytes (564.2 MiB)
- `verification_full_out.txt`: created; no errors or warnings
- Full conversion wall time: 43.39 s, consistent with and slightly faster than the Step 7 estimate. No optimization retry was needed.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 522 units | Figure 8 cohort/loaders and >1-Hz rule | 528 quality-eligible; 521 after released calculation | 521 | Yes within the one-unit paper/data-version discrepancy; exact 214 single units matches paper |
| Mean neurons/session | 43.5 from 522/12 | session-specific filtering | 43.42 after source calculation | 43.42 (27-67) | Yes |
| Subjects | 6 reported | loader names 7 IDs | 7 explicit source IDs | 7 | Source/code exact; documented paper discrepancy |
| Sessions | 12 | 12 explicit sessions | 12 | 12 | Yes |
| Trials (total) | not reported for this subset | exclude early/stim for this analysis | 3,626 raw; 3,116 retained | 3,116 | Yes |
| Trials/session (mean) | not reported | same trial mask | 259.67 retained | 259.67 (210-390) | Yes |
| Time input range | N/A | 5-ms bins around go cue | [-2.4975, 2.4975] bin centers | [-2.4975, 2.4975] | Yes |
| Lick distribution | N/A | actual choice; ignore separately | [0.398,0.377,0.225] | [0.398,0.377,0.225] | Yes |
| Context distribution | N/A | autowater encodes WC | [0.315,0.685] | [0.315,0.685] | Yes |
| Outcome distribution | ignores omitted in paper analyses | hit/miss/no source fields | [0.106,0.669,0.225] | [0.106,0.669,0.225] | Yes; retained because requested |
| Tongue velocity distribution | N/A | side-camera tongue coordinates | [0.028,0.028,0.945] | [0.028,0.028,0.945] | Yes |
| Paw velocity distribution | N/A | bottom-camera top-paw coordinates | [0.435,0.435,0.129] | [0.435,0.435,0.129] | Yes |
| Motion-energy distribution | N/A | side-camera motion-energy files | [0.4997,0.5003,0 no-video] | [0.4997,0.5003,0] | Yes |

All 12 raw files were reopened independently. Their `Ntrials`, exact `~early & ~stim` retained indices, trial-list lengths, unit/region lengths, and first/middle/last finite neural matrices match the pickle. The audit initially contained a manually typed expected raw total of 3,826; summing the source sessions showed 3,626, the assertion constant was corrected, and the complete audit then passed. This was a test transcription error, not a conversion change.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: Read the complete regenerated `/app/verification_full_out.txt`. It explicitly reports “Data format is valid, no errors or warnings” and completes normally. There are no warnings requiring waiver.
2. **Independent raw-file sanity audit**: Created `/app/cache/audit_conversion.py`; it never imports the converter. For session 0, retained trial 5 (raw trial 6), neuron 3, it independently bins raw spikes, applies an explicit convolution, and matches all 1000 converted neural samples with `np.allclose` (maximum absolute error 0). The analytical input centers match with `np.allclose`. Direct raw behavior fields reproduce the three static outputs for every one of 3,116 trials, including the requested trial/time spot check.
3. **Independent video-output audit**: Direct HDF5 DLC/frame-time reads and the original motion-energy MAT file independently reconstruct the full 1000-bin tongue, top-paw, and motion categorical traces for the same trial. All three match with exact `np.allclose`. This jointly tests bitcode synchronization, interpolation, derivatives, missingness, medians, and discretization.
4. **Reference-code comparison**: Compared each major stage as detailed below. All applicable reference operations match; intentional differences are confined to the explicit decoder contract.
5. **Statistics comparison**: Reopened all 12 source files. Totals are 3,626 raw trials, 3,116 exact `~early & ~stim` trials, 521 retained units, 12 sessions, and seven explicit source IDs. Unit qualities are 19 excellent + 130 fair + 31 good + 34 great + 149 multi + 155 poor + 3 unlabeled. The four well-isolated labels sum to the paper's exact 214 single units. Converted distributions are reported in Step 9 and exactly reproduced by the audit.
6. **Edge cases/off-by-one audit**: Checked first/middle/last trial matrices in every session, every shape/dtype/value domain, exact first/last time centers, and all retained spike alignments for equality to −2.5 or +2.5 s. There are zero exact boundary spikes, removing last-edge ambiguity. Every session has 210–390 trials and 27–67 units. All class-0/1 median splits are within 0.006 of 50/50 per session.

### Major Processing-Step Comparison
| Stage | Reference implementation | Converter implementation | Result / rationale |
|---|---|---|---|
| Data loading | `loadObjs`, animal `load*_ALMVideo`, Figure 8 explicit cohort/probes | `SESSION_SPECS`, direct v7.3 HDF5 reads, `load_motion_energy` | Same 12 sessions/probes. Direct field reads avoid materialization but do not alter values. |
| Neuron filtering | `findClusters(...,'all')`, then `removeLowFRClusters` with strict `>1` in figure scripts | `selected_clusters` (same four exclusions), `low_fr_filter` (same seven conditions and strict `>1`) | 521 all-unit analysis set and exact 214 well-isolated singles; paper reports 522/214, consistent to one released unit. |
| Trial filtering | Figure conditions exclude `stim`; analyses exclude `early`; paper generally omits ignores | `keep_trials = ~early & ~stim`; hit/miss/no retained | Same quality filter. Retaining no/ignore is required to supply requested ignore and none classes. |
| Temporal alignment | `alignSpikes`: `trialtm - goCue`; `findPosition`/`loadMotionEnergy`: frame time − video offset − go cue | `build_neural`, `feature_speed`, `align_kinematics_and_motion` | Exact same events/sign convention. Raw-derived categorical traces match exactly. |
| Binning/smoothing | `getSeq`, 5-ms centers; `mySmooth`, causalized `gausswin(15)` with leading prefix | `TIME`, `build_neural`, `reference_smooth` | Explicit independent convolution matches converted values exactly. |
| Input construction | `obj.time` bin centers | one continuous `time_from_go_cue_s` row | Exact requested decoder input and reference time axis. |
| Output construction | behavior fields, kinematic first derivatives, aligned motion energy | actual lick/context/outcome plus requested median-discretized speed magnitudes and energy | Static fields match all raw trials. Median thresholds override manual motion thresholds as explicitly required. Missing tongue/paw masks are retained for requested class 2. |

### Issues Found and Resolved
- **Raw-total audit typo**: The first hand-written expected total was 3,826, while source-session summation is 3,626. Corrected the audit constant and reran it; all per-session values had already matched. No conversion change was involved.
- **All-NaN motion frame times**: JEB19 2023-04-19 retained trial 208/raw trial 222 has a finite 3,087-sample raw motion trace but all-NaN camera timestamps. The first converter version assigned 1,000 class-2 bins. Reference `loadMotionEnergy.m` instead catches this case and uses a nominal 400-Hz clock with an explicit 0.5-s shift. Implemented that fallback, regenerated the full pickle, reran full verification, and reran every audit above. The trace is now recovered and no motion sample is incorrectly class 2.
- **Recheck result**: Regenerated full conversion completed in 42.77 s. Verification has zero errors/warnings; every independent audit prints `ALL INDEPENDENT AUDIT CHECKS PASSED`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. CUDA training used 2,489 training and 627 validation trials. Loss decreased monotonically from 7.8210 at epoch 1 to 0.7821 at epoch 200; held-out test loss was 0.8046. Execution completed normally and sample plots were generated.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction | 0.5776 | 0.5566 | 3 specified/observed classes |
| Behavioral context | 0.7221 | 0.7143 | 2 classes |
| Outcome | 0.5820 | 0.5420 | 3 classes |
| Tongue velocity | 0.5813 | 0.5627 | 3 classes; class 2 dominates raw samples but balanced scoring/loss used |
| Paw velocity | 0.5316 | 0.5228 | 3 classes |
| Motion energy | 0.7506 | 0.7496 | 2 observed classes; output vocabulary reserves class 2 for absent video |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Full-decoder validation BA | Uniform chance / ratio | Expectation from paper | Finding |
|---|---:|---:|---|---|
| Lick direction | 0.5566 | 0.333 / 1.67× | Delay CDchoice AUC 0.86±0.11; plotted time-resolved neural accuracy ~0.65–0.90 | Above heuristic; supplemental paper-like curve peaks 0.9396 |
| Behavioral context | 0.7143 | 0.500 / 1.43× | Plotted time-resolved neural accuracy ~0.78–0.90 | Below 1.5× heuristic globally, fully investigated; paper-like curve is 0.76–0.96 over most of the trial and peaks 0.9624 |
| Outcome | 0.5420 | 0.333 / 1.63× | Not decoded in paper | Above heuristic |
| Tongue velocity | 0.5627 | 0.333 / 1.69× | Not decoded in paper | Above heuristic despite 94.5% not-visible samples; balanced loss/metric used |
| Paw velocity | 0.5228 | 0.333 / 1.57× | Not decoded in paper | Above heuristic |
| Motion energy | 0.7496 | 0.500 / 1.499× empirical | Not decoded in paper | Essentially 1.5× strict two-observed-class chance (the validator displays 1/3 because class 2 is reserved); raw alignment and labels verified exactly |

The supplied decoder's lick/context scores are averaged over every 5-ms timepoint of correct, incorrect, and ignore trials, whereas the paper balanced correct trials and trained a separate 75-ms linear model at each time window within each session. To test the data rather than assume this difference explains the scores, `/app/cache/paper_like_decoder_check.py` reproduced the paper design using balanced correct trials, 75-ms windows, four-fold CV, and session-specific linear models. Choice accuracy was 0.6157 at −2 s, 0.6764 at −1 s, 0.7773 at go, 0.9186 at +0.5 s, and peaked at 0.9396; context was 0.8360, 0.7985, 0.8123, 0.8635, and peaked at 0.9624 respectively. Shuffled means were 0.5038 and 0.5015. The shape and scale match or exceed the paper curves, strongly rejecting a temporal-alignment or label bug.

Training-to-validation ratios are lick 1.04, context 1.01, outcome 1.07, tongue 1.03, paw 1.02, and motion 1.00—far below the 1.5× overfitting threshold. The 0.7821 training versus 0.8046 test loss is similarly close; there is no evidence of leakage or material overfit.

Specific low-score debugging checks were all performed: raw behavior and motion outputs were checked for JEB6 session 0 retained trial 5/raw 6, JEB19 2023-04-20 retained trial 100/raw 121, and the repaired JEB19 2023-04-19 retained trial 208/raw 222; all matched exactly. Output variation is substantial for static variables and exactly median-balanced for valid continuous samples. `predictions.png`, both processing plots, and `cache/paper_like_decoding.png` show synchronized neural/output changes around go cue. Unit quality exclusions, strict >1-Hz filtering, and causal smoothing were independently rechecked in Step 10.

### Issues Found and Resolved
- **Context below 1.5× global chance**: No data bug. Exact raw labels, class variation, alignment, and paper-style decoding all pass; the supplied global multi-task metric averages long periods that the paper scores separately.
- **Motion at 1.499× empirical chance**: No data bug. All motion outputs derive exactly from raw aligned traces, classes are 49.97/50.03%, and 0.7496 is only 0.0004 below the rounded 0.75 heuristic. Inventing absent-video samples or altering thresholds to game this metric would violate the source and task.
- **Paper comparison**: The only exact tabulated neural decoding value is choice AUC 0.86±0.11, a different metric. Supplemental matched-design accuracies agree with the paper's time-resolved plotted choice/context curves; the four new outputs have no paper comparator.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with loading example, schema, category meanings, statistics, reproducibility commands, and decoder results.
- [x] `cache/` folder created; `README_CACHE.md` inventories independent audits, supplemental decoder, logs, figures, and paper-page renders.
- [x] All investigation scripts reside under `cache/`; required conversion code, pickles, logs, and user-facing figures remain at the project root.
- [x] All required artifacts are present and nonempty. `convert_data.py` and both cached audits pass `py_compile`; final format verification and independent raw audit both pass with no warnings/errors.
