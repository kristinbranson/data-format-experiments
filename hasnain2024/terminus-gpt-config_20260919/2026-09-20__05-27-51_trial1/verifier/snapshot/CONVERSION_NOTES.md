# Dataset Conversion Notes

## Overview
- **Dataset**: Separating cognitive and motor processes in the behaving mouse
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code`
- `data`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- Torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadProcessedData` / `loadObjs` / `loadSessionData` | `DataLoadingScripts/` | LOADING | Load saved MATLAB session objects and associated metadata. |
| `getDefaultParams` | `DataLoadingScripts/getDefaultParams.m` | PROCESSING | Defines analysis defaults, including event alignment, analysis window, bin width/smoothing, conditions, and accepted unit quality. |
| `findTrials` | `DataLoadingScripts/findTrials.m` | CURATION | Selects trial indices using behavioral condition labels and trial events. |
| `findClusters` | `DataLoadingScripts/findClusters.m` | CURATION | Matches each cluster's quality string against requested accepted qualities. |
| `removeLowFRClusters` | `DataLoadingScripts/removeLowFRClusters.m` | CURATION | Removes clusters below the configured firing-rate criterion. |
| `alignSpikes` | `DataLoadingScripts/alignSpikes.m` | PROCESSING | Subtracts the selected event time from within-trial spike times; supports go cue and movement/lick/jaw-derived events. |
| `getPSTHs` | reference utility called by preprocessing | PROCESSING | Builds trial-averaged PSTHs and binned single-trial neural activity for selected trials/units. |
| `preProcessObjs` / `processData` | `utils/preProcessObjs.m`, `DataLoadingScripts/processData.m` | PROCESSING | Orchestrates trial/unit selection, alignment, PSTH construction, and session/probe assembly. |
| `loadKinData` / `loadBehavVid` | `DataLoadingScripts/loadKinData.m`, `utils/loadBehavVid.m` | LOADING | Loads trialized behavior-video/DLC trajectories and camera timestamps. |
| `getKinematics` / `getKinematicsFromVideo` | `funcs/kinematics/` | PROCESSING | Selects named DLC features and puts video-derived kinematics onto the analysis time base. |
| `findVelocity` | `funcs/kinematics/findVelocity.m` | PROCESSING | Computes velocity from tracked position trajectories. |
| `loadMotionEnergy` | `DataLoadingScripts/loadMotionEnergy.m` | LOADING | Loads video motion-energy traces for sessions/trials. |
| `findDLCFeatIndex` | `utils/findDLCFeatIndex.m` | PROCESSING | Resolves named body-part features in per-trial DLC arrays. |

### Notes
- The code base is MATLAB. Neural data are extracellular spikes, not calcium imaging; no delta-F/F calculation applies.
- Canonical preprocessing is: load session object/metadata, choose behavioral trials with `findTrials`, choose unit qualities with `findClusters`, optionally remove low-rate clusters, align each spike by subtracting the requested event, and bin/smooth selected spikes into single-trial matrices/PSTHs.
- `alignSpikes` stores `trialtm_aligned = trialtm - event`; therefore go-cue alignment must use the session behavioral event corresponding to go cue, preserving negative pre-cue and positive post-cue times.
- Sessions with two probes are represented by concatenating neural columns/units across probes after processing; they remain one behavioral session.
- Behavioral video is trialized and linked to camera timestamps. DLC feature names are resolved rather than assumed by fixed column position. Kinematic functions interpolate/transform tracked features onto a common analysis time base; motion energy has a dedicated loader.
- Choice/context decoding scripts use behavioral trial groups (water-cued versus delayed-response contexts and left/right choices), confirming those labels originate from trial metadata rather than inferred from neural activity.
- Neural curation is quality-string based and can include a low-firing-rate cutoff. Exact qualities, threshold, time window, and bin size used for the released dataset will be cross-checked against raw metadata, analysis entry scripts, methods, and paper in Steps 2-4.
- Video streams may be absent or features may be untracked on individual frames/trials. The source code retains camera/trial correspondence and named-feature checks; missingness must not be silently converted to low velocity.
- No source files are modified or imported from outside `/app/code` during this step.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains 120 MATLAB session files in four cohorts: `DelayInhibition_BilatMC_Behavior` (53), `GoCueInhibition_BilatMC_Behavior` (20), `Ephys_Behavior` (25), and `RandomizedDelay_Ephys_Behavior` (22). Files are a mix of MATLAB v7.3/HDF5 and older MAT serialization.
- Each `data_structure_<animal>_<date>.mat` contains `obj`. Common fields are `bp` (behavior and events), `traj` (two camera views, trialized DLC trajectories/timestamps), `trials` (behavior/ephys mapping), and metadata. Ephys-capable files additionally contain `clu` with spike time, trial, channel, waveform, and quality fields. Some files include embedded `me`; separate `motionEnergy_<animal>_<date>.mat` files also exist.
- `bp` includes `Ntrials`, `R`, `L`, `hit`, `miss`, `no`, `early`, stimulation/autowater flags, and `ev`. Randomized-delay event fields are `bitStart`, `sample`, `delay`, `goCue`, `reward`, `lickL`, and `lickR`.
- `traj` has two camera views and, in checked sessions, one structure per behavioral trial. Fields include feature names, tracked coordinates, frame timestamps, and dropped-frame information. Missing motion-energy files/arrays occur in some sessions.
- Probe metadata identifies Neuropixels/Kilosort recordings and locations including left/right ALM and M1TJ. Randomized-delay neural sessions are ALM; dummy/nonrecording sessions are explicitly marked.
- Schema edge cases: cluster data can be a struct array or a cell array by probe; quality strings are space-padded and include one source typo `mutli`; older sessions omit newer fields such as `autolearn`; motion energy is represented as either a struct or trial cell array.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Native session files | 120 |
| Native trials (all supplied cohorts) | 37,073 |
| Native subjects | 18 unique IDs |
| Sessions by cohort | Delay inhibition 53; Ephys 25; go-cue inhibition 20; randomized-delay ephys/behavior 22 |
| Trials by cohort | 15,079; 8,260; 6,151; 7,583 respectively |
| Randomized-delay subjects | 4 (JEB11, JEB12, JEB23, JEB24) |
| Randomized-delay sessions with directly populated units | 11 |
| Randomized-delay trials in those sessions | 3,939 |
| Randomized-delay units in those sessions | 455 before quality curation |
| Randomized-delay trials/session | 218-450 (all 22 sessions) |
| Randomized-delay units/session | 17-62 among populated-unit sessions |
| Video views | 2 per randomized-delay session; trial counts match `Ntrials` in scan |
| Motion energy | Present for 16/22 randomized-delay sessions by embedded/separate-data scan; missingness must be retained |

### Available Variables and Quality Notes
- Neural: per-unit global and within-trial spike times, trial indices, channels, waveform summaries, and quality labels (`excellent`, `great`, `good`, `fair`, `multi`, plus typo `mutli`).
- Experimental/behavioral: instructed side R/L, hit/miss/no-response/early, stimulation, autowater/autolearn, sample/delay/go-cue/reward timing, individual left/right lick times.
- Video: named DLC trajectories (including tongue/jaw/paw features), camera times/drop counts, and motion energy where video processing exists.
- The full supplied archive includes behavior-only optogenetic cohorts that have no clusters; they cannot become decoder sessions because neural input is mandatory. Final inclusion and context mapping will be resolved against code and paper in Steps 3-5.
- Counts were obtained by loading original MAT files directly (h5py/scipy), not through conversion code.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote / interpretation |
|-----------|-------|-------------------------------|
| DR units | 1,651 total; 483 well-isolated single units | Methods: ALM, 25 sessions, nine mice. |
| DR sessions / mice | 25 / 9 | Electrophysiology recording analysis. |
| Two-context units | 522 total; 214 well-isolated single units | Methods: subset in which animals performed DR and WC. |
| Two-context sessions / mice | 12 / 6 | This is the dataset relevant to required WC/DR output. |
| Randomized-delay units | 845 total; 288 well-isolated single units | Methods: 19 sessions, four mice. |
| Session inclusion | at least 10 units | Recording-session criterion. |
| Unit inclusion for most analyses | all units with firing rate >1 Hz | Only specified subspace-alignment/single-unit analyses restrict further to well-isolated single units. |
| Behavioral trial minimum | >=40 correct DR trials/direction and >=20 correct WC trials/direction | Behavioral-analysis sessions; early-lick and ignore trials excluded for those analyses. |
| Context organization | alternating DR/WC blocks, beginning with DR | Methods discussion of context-selective units; sessions often end in WC. |
| ITI context analysis window | 300 ms before sample tone | No external cue indicates context in this interval. |
| Movement-initiation window | correct lick within 600 ms of go cue/water drop | Go cue in DR and water drop in WC are analogous response triggers. |
| Tongue visibility window | 1 s after go cue/water drop | DLC-labeled visibility fraction. |
| Neural source | extracellular Neuropixels; JRCLUST and/or Kilosort 3, manual curation | No imaging or delta-F/F. |
| Paper decoder metric | tongue-angle regression R² | Paper predicts continuous tongue angle from full/null/potent neural activity; no directly comparable categorical accuracy table for required outputs. |

### Processing Details
- DR trials contain sample, delay, go-cue, and response epochs. In WC blocks, an uncued water drop drives licking; the paper treats go cue/water drop as corresponding movement-initiation events.
- The required conversion nevertheless aligns every included trial to the source `goCue` event as explicitly requested. The native event array is the authoritative timing source.
- Neural spike sorting used JRCLUST and/or Kilosort with manual curation in Phy. PSTHs/single-trial rates in the repository are formed after event alignment; exact bin/smoothing parameters will be taken from the analysis defaults/code during mapping.
- Video features were tracked with DeepLabCut. Tongue visibility is meaningful missingness, not zero movement. Motion energy is analyzed from video in task epochs.
- The paper uses time warping only for specific across-trial/session kinematic comparisons; the decoder task requests direct go-cue temporal alignment, so time warping is not appropriate for the converted trial streams.

### Curation Steps

**Neuron curation rules**:
- Include recording sessions only if at least 10 units.
- For analyses analogous to population decoding, retain manually curated single and multiunits with session firing rate >1 Hz.
- Do not limit to well-isolated single units unless reproducing the paper's explicitly listed subspace-alignment or single-unit-selectivity analyses.

**Trial curation rules**:
- The paper omits early-lick and ignore trials from several analyses, but the requested decoder explicitly includes outcome `ignore`; therefore ignore trials must be retained when neural/video alignment is valid and encoded as an output class.
- Stimulation/autowater trials require code/data cross-checking. They should not be silently mixed with control context labels if the reference two-context analysis excludes them.
- Require valid go-cue alignment and matching neural trial mapping. Preserve absent/untracked video through the specified `not visible`/`no video` categories.

### Decoders Trained
| Decoded variable | Accuracy / metric |
|------------------|-------------------|
| Tongue angle from population activity | R² reported graphically across 25 sessions; no categorical accuracy reported |
| Context/choice coding dimensions | Population selectivity/projections rather than a directly comparable classifier accuracy |

### Reference interpretation for this task
The relevant source subset is the 12 two-context electrophysiology sessions (six mice, 522 pre-rate-filter units), because behavioral context WC/DR is required. Randomized-delay sessions cannot supply WC labels even though their schema includes `goCue`. Final session identities and post->1 Hz counts must match metadata/reference code in Step 4.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Relevant cohort | Figure 8 loads animal-specific ALM/video sessions and defines DR as `~autowater`, WC as `autowater` | `Ephys_Behavior` has the required alternating autowater blocks and neural/video streams | Two-context subset has 12 sessions and 522 units | Use the Figure 8 two-context Ephys subset, not randomized-delay or optogenetic behavior-only cohorts. |
| Raw unit counts | `findClusters` selects configured quality strings; `removeLowFRClusters` retains mean PSTH FR > threshold | Modern files contain many `garbage`/`poor` clusters and probe-nested cell arrays | 522 manually curated units; most analyses use units >1 Hz | Follow selected probe metadata, accepted quality labels, then >1 Hz filter; never count all raw Kilosort clusters. |
| Context label | Figure 8 conditions contrast hit, nonstim, non-early `~autowater` versus `autowater` | Autowater occurs in alternating blocks in context sessions | DR/WC blocks alternate and begin in DR | Map `autowater=False` to DR and `autowater=True` to WC for valid two-context trials. |
| Ignore trials | Figure 8/paper often omit ignore/early trials for specific analyses | `no` and `early` flags exist per trial | Behavioral analyses omit them | Decoder explicitly requires `ignore`; retain aligned ignore trials, map `no`/early nonresponses to ignore, and document task-required deviation. |
| Temporal processing | Figure 8a-c population context analysis uses 10 ms bins and smoothing parameter 15 around go cue (Figure 8d uses 5 ms only for its single-unit/subspace analysis) | Native spikes have per-trial times and `goCue` events; video has frame timestamps | Go cue/water drop anchors movement analyses | Use direct go-cue alignment and reference binning; interpolate video-derived outputs to the neural time grid without time warping. |


### Final Reconciliation
- Authoritative Figure 8 loader metadata selects 12 available ALM/video sessions: JEB6 (1), JEB7 (2), EKH1 (1), EKH3 (1), JGR2 (2), JGR3 (1), and JEB19 (4), using the explicitly selected probe for each session.
- Directly counting source spikes in the reference -2.5 to +2.5 s window and applying strict mean firing rate >1 Hz reproduces **214 well-isolated single units**, exactly the paper value.
- The exact 214 well-isolated-unit count is reproduced after >1 Hz filtering. The paper's 522 total lies between the raw manually categorized count (~524, depending on noisy/questionable labels) and 517 non-garbage/nonblank units above 1 Hz; it therefore describes recorded manually curated units before the population rate filter. This small label-accounting discrepancy is retained explicitly rather than forcing the count.
- The released loader/data use seven animal IDs whereas the manuscript says six mice. Session count and exact 214 single-unit statistic match; no data are duplicated. This is documented as a source-text/code discrepancy rather than silently dropping an animal.
- One loader entry, JEB7 2021-04-17, is absent from the supplied data; the loaders also include the two available JEB7 dates. The exact 12 available sessions match the manuscript session count.
- Correct reference population parameters are go-cue alignment, no time warping, -2.5 to +2.5 s, 10 ms bins, smoothing parameter 15, and mean firing rate >1 Hz. Figure 8d's 5 ms/single-unit settings are not used for this population-decoder conversion.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Selected probe `clu[].trial`, `clu[].trialtm` | `neural` | Subtract each trial's `bp.ev.goCue`; histogram into 10 ms bins from -2.5 to +2.5 s; divide by 0.01 s; smooth across time with the reference width 15 | `alignSpikes`, `getSeq`, `processData`, `removeLowFRClusters` | `(neurons, 500)` float32 per trial; no time warping. |
| Bin centers | `input[0]` | `-2.495, ..., 2.495` seconds repeated per trial | `getSeq` | Decoder input is continuous time from go cue, shape `(1,500)`. |
| `R`, `L`, `hit`, `miss`, `no` | `output[0]` lick direction | hit: instructed side; miss: opposite side; no response: none | behavioral flags / `findTrials` | Values: left=0, right=1, none=2; repeated over time. |
| `autowater` | `output[1]` behavioral context | false=DR, true=WC | Figure 8 conditions | Values WC=0, DR=1; repeated over time. |
| `hit`, `miss`, `no`, `early` | `output[2]` outcome | miss=incorrect, hit=correct, no/otherwise early nonresponse=ignore | behavioral flags | Values incorrect=0, correct=1, ignore=2; repeated over time. |
| Side-view DLC tongue landmark positions/timestamps | `output[3]` tongue velocity | Smooth position as reference, finite difference magnitude; interpolate to bin centers; per-session median split using only visible finite samples | `getKinematicsFromVideo`, `findVelocity`, `findDLCFeatIndex` | 0 below median, 1 at/above median, 2 not visible. Preserve NaN visibility before reference zero-fill. |
| DLC `top_paw` position/timestamps | `output[4]` paw velocity | Same velocity/interpolation and per-session visible-sample median split | Figure 1 feature usage; `findVelocity` | 0 below median, 1 at/above median, 2 not visible. |
| Session/trial motion energy | `output[5]` motion energy | Interpolate native trace to go-cue grid; per-session median split over available finite samples | `loadMotionEnergy`, `loadMotionEnergy_Behav` | 0 below median, 1 at/above median, 2 no video/data. |
| Loader animal ID | `subjects`, `subject_idx` | unique ID and per-session index | `load<animal>_ALMVideo` | Seven released IDs; manuscript reports six mice, documented discrepancy. |
| Probe `meta.probe.loc` | `brain_regions`, `brain_region_idx` | normalize side-specific ALM labels to `ALM` | metadata/loaders | Every retained neuron belongs to ALM. |

### Key Decisions
1. **Session subset**: Use the 12 available sessions selected by Figure 8 ALM/video loaders: JEB6 2021-04-18; JEB7 2021-04-29 and 2021-04-30; EKH1 2021-08-07; EKH3 2021-08-11; JGR2 2021-11-16 and 2021-11-17; JGR3 2021-11-18; JEB19 2023-04-18 through 2023-04-21. This matches the paper's session count and exact 214 single-unit statistic. Use each loader's selected probe.
2. **Trial filtering**: Exclude 135 stimulation trials because every Figure 8 context condition requires `~stim.enable`. Retain hit, miss, no-response, and early trials with valid go cues because the requested output explicitly includes incorrect and ignore. All selected sessions have finite go cues. Require native trial index correspondence; do not invent trials.
3. **Unit filtering**: Exclude clear artifacts (`garbage`, blank/unlabeled, `noisy`, and questionable `real?`), then require mean firing rate strictly >1 Hz over the reference five-second window and included trials. Retain poor/fair/good/great/excellent/multi categories because population analyses use all units rather than only well-isolated single units. The paper's 522 recorded-unit count is pre-rate-filter and differs by a few label-edge cases; exact 214 high-rate single-unit count was reproduced.
4. **Neural representation**: Use firing rate rather than raw counts, matching PSTH processing. Use Figure 8a-c population settings (10 ms), not Figure 8d's 5 ms single-unit/subspace settings. Smoothing width 15 is applied in bins with edge-aware normalization.
5. **Mixed outputs**: Repeat per-trial lick/context/outcome labels over 500 time points so all six outputs form a single categorical `(6,500)` array. This is semantically constant per trial and compatible with time-varying video labels.
6. **Context**: `autowater` is the code's WC/AW context indicator, not merely an outcome flag; its alternating blocks match the paper. `~autowater` is DR.
7. **Lick direction**: R/L encode instructed direction, so misses must be inverted to represent actual lick direction. `no` maps to none. This avoids incorrectly treating instructed side as behavioral choice.
8. **Missing video**: Never turn invisible tongue/paw or absent video into low velocity. The decoder specification overrides reference plotting code that zero-fills invisible tongue velocity.
9. **Thresholds**: Compute one median per session and variable from all finite available time samples after alignment/interpolation and trial filtering; equality belongs to class 1 exactly as specified. Missing samples do not affect medians.
10. **Metadata window**: `off_start=-2.5`, `off_end=2.5`, `time_bin_size=10.0` ms, alignment event `Bpod goCue onset`.

### Planned Sanity Checks
- [ ] Direct raw-vs-converted neural `np.allclose` check for selected session/trial/unit bins after independently histogramming source spikes.
- [ ] Direct input `np.allclose` against analytical bin-center vector.
- [ ] Direct output `np.allclose` for three trials covering hit, miss, and no-response, including context inversion and actual lick side.
- [ ] Verify 12 sessions, seven released IDs, 3,626 native trials, 135 stimulation trials excluded, and no invalid go cues.
- [ ] Verify every retained session has >=2 trials and >=10 post-filter units; investigate otherwise.
- [ ] Confirm output values are integer categories in declared ranges and all three lick/outcome classes and both contexts occur.
- [ ] Confirm per-session finite velocity/energy class fractions straddle 50% (ties may shift exact fraction) and missing classes correspond exactly to source missingness.
- [ ] Confirm two trajectory views have one entry per native trial and video timestamps overlap the go-cue window where classes are not missing.
- [ ] Compare retained unit totals and single-unit subset to paper (exact 214 single units expected before task-required representation changes).

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Implemented `/app/convert_data.py` with `--full`, `--sample`, and `--show-processing` modes.
- Uses targeted h5py access to MATLAB v7.3 structs, avoiding full deserialization of multi-gigabyte reference arrays.
- Supports nested probe-cell cluster structures, source quality labels, >1 Hz filtering, reference go-cue binning/smoothing, DLC trajectories, per-trial variable-length motion energy, and explicit missing-video categories.
- Assertions/shape assumptions are checked while loading; every output is stored as compact float32/int64 arrays.

Code inefficiencies identified:
- Initial `mat73` recursive loading stalled on large spike/video branches.
- Initial neural smoothing reshape could reorder trial/unit axes; found during sample review and fixed before validation.
- Motion-energy traces are ragged MATLAB cells and could not be cast to a dense array.

Code speedups added:
- Direct HDF5 reference traversal loads only selected probe/unit arrays.
- Vectorized histogram construction where possible; conversion takes about 3 seconds/session in the two-session sample, predicting under one minute for all 12 sessions.
- Ragged motion-energy traces are interpolated trial-by-trial using actual camera frame timestamps without redundant file reads.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (session-neuron total) | 95 |
| Neurons / session | [29, 66] |
| Subjects | 2 (JEB6, JEB7) |
| Sessions | 2 |
| Trials (total) | 622 |
| Trials / session | [324, 298] |
| Input time range | [-2.495, 2.495] s |
| Time points / trial | 500 |
| Output distributions | [('lick direction', {0: 0.426, 1: 0.4437, 2: 0.1302}), ('behavioral context', {0: 0.3424, 1: 0.6576}), ('outcome', {0: 0.127, 1: 0.7428, 2: 0.1302}), ('tongue velocity', {0: 0.0633, 1: 0.0633, 2: 0.8734}), ('paw velocity', {0: 0.4379, 1: 0.4379, 2: 0.1242}), ('motion energy', {0: 0.4531, 1: 0.4531, 2: 0.0938})] |

### Processing Plots Review
- `processing_JEB6_2021-04-18.png` and `processing_JEB7_2021-04-29.png` were created and are nonempty.
- Velocity and motion-energy traces overlap the go-cue window; tongue visibility is sparse outside licking, as expected. Paw and motion energy have broad coverage.
- Session medians split every finite continuous stream approximately 50/50; missing values map only to class 2.
- Manual dimensions, finite-neural values, integer outputs, and analytical input-vector checks passed.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Targeted HDF5 reference loading | Avoids stalled full MAT deserialization |
| Per-session single-pass video/neural processing | Sample completes in ~6 s |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion | ~3 s | <1 minute for 12 sessions |
| Verification | seconds | seconds |

### Validation
- `/app/sample_data.pkl` created successfully.
- `/app/verification_sample_out.txt` created; verifier output reviewed in terminal.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None requiring conversion changes

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| Lick direction | 0.7104 | 0.6809 |
| Behavioral context | 0.8414 | 0.8069 |
| Outcome | 0.7019 | 0.6309 |
| Tongue velocity | 0.6350 | 0.6135 |
| Paw velocity | 0.5630 | 0.5767 |
| Motion energy | 0.7180 | 0.7209 |

- Loss decreased steadily from >3.4 early in training to 0.6994 at epoch 200; test loss was 0.7681.
- Every validation balanced accuracy is above chance (1/3 except context at 1/2), with no concerning train-validation gap.
- `/app/train_decoder_sample_out.txt` completed with `train_decoder.py finished successfully`.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 401.93 MB
- `verification_full_out.txt`: created; `Data verification complete` with no errors

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | 12 two-context | 12 available loader-selected | 12 | 12 | Yes |
| Subjects | 6 mice stated | 7 released IDs in loaders | 7 IDs | 7 | Code/data match; manuscript discrepancy documented |
| Native trials | not stated globally | all hit/miss/no, nonstim context conditions | 3,626 | 3,491 retained after 135 stim exclusions | Exact |
| Total neurons | 522 recorded | all nonartifact, >1 Hz for population | source label accounting 517-524; 515 after strict artifact and >1 Hz filtering | 515 | Consistent with curation/filtering |
| Well-isolated units | 214 | excellent/great/good/fair and >1 Hz | independently reproduced 214 | included within population | Exact source check |
| Time bin | population Fig. 8: 10 ms | `params.dt=1/100` | spike timestamps continuous | 10 ms | Yes |
| Time window | figure/code ±2.5 s | `tmin=-2.5`, `tmax=2.5` | finite go cues all trials | [-2.5,2.5], 500 bins | Yes |
| Alignment | go cue | `alignEvent='goCue'` | `bp.ev.goCue` | Bpod goCue | Yes |
| Trials/session | not tabulated | loader sessions | 230-426 native | 227-426 retained | Consistent |
| Lick distribution | not tabulated | behavior flags | source-derived | [0.4171,0.3755,0.2074] | Sensible |
| Context distribution | block design | autowater vs non-autowater | source-derived | WC 0.3220, DR 0.6780 | Sensible |
| Outcome distribution | not tabulated | hit/miss/no | source-derived | incorrect 0.1126, correct 0.6800, ignore 0.2074 | Sensible |
| Tongue velocity | visibility sparse | DLC tongue | source-derived | low/high 0.0425 each, missing 0.9150 | Expected visibility behavior |
| Paw velocity | tracked video | DLC top_paw | source-derived | low/high 0.3834 each, missing 0.2333 | Median split correct |
| Motion energy | video-dependent | per-trial ME | source-derived | low 0.4360, high 0.4372, no video 0.1268 | Median split/missing correct |

- Spot checks of sessions 1, 7, and 12 found consistent `(trial, neuron, 500)`, `(trial,1,500)`, and `(trial,6,500)` shapes and finite neural values.
- Every session has at least two trials and at least 27 retained neurons.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Searched `conversion_full_out.txt` and `verification_full_out.txt` for errors, warnings, traceback, failure, NaN, and infinity. No actionable warnings or errors were present; verifier ended with `Data verification complete`.
2. **Independent raw-source sanity checks**: `/tmp/sanity_raw.py` loads the original JEB6 HDF5 file directly without importing conversion code. An independently histogrammed and smoothed complete neural trace matched converted session 1/trial 6/unit 1 with `np.allclose`; the analytical time vector matched with `np.allclose`; raw behavioral labels matched converted labels with `np.allclose`. Additional explicit hit, miss, and no-response trials all matched lick/context/outcome labels exactly.
3. **Reference code comparison**:
   - Loading: converter follows MATLAB cell/struct references for the loader-selected probe, equivalent to `loadSessionData`; it avoids unrelated probes.
   - Neuron filtering: source artifact labels are excluded and mean firing rate is required to be strictly >1 Hz, matching `removeLowFRClusters` and methods. Population multiunits are retained; 214 well-isolated units were independently reproduced.
   - Trial filtering: stimulation is excluded as in every Figure 8 context condition. Unlike paper plots, miss/no trials are retained because decoder outputs explicitly require incorrect/ignore.
   - Temporal alignment: source within-trial times subtract `bp.ev.goCue`, exactly matching `alignSpikes`; all 3,626 native go cues are finite.
   - Binning: -2.5 to +2.5 s at 10 ms and 15-bin centered smoothing match Figure 8a-c/getSeq. No time warping is used.
   - Input: bin centers exactly match `edges + dt/2`; direct `np.allclose` passed.
   - Outputs: context follows Figure 8 `autowater` versus `~autowater`; outcome follows hit/miss/no; actual lick direction inverts instructed side on miss trials. Video uses named DLC features/frame timestamps and per-trial motion energy. Missing visibility/video is preserved as class 2 per decoder requirements rather than reference plotting's tongue zero-fill.
4. **Key statistics**: 12 sessions, 3,626 native trials, 135 stimulation trials excluded, 3,491 converted trials, seven released subject IDs, 515 retained session-neurons, one ALM region, 500 time bins. Session/trial/unit counts and all output distributions were compared with paper/code/data in Step 9.
5. **Edge cases**: Verified nested per-probe cluster cells, variable-length video/ME traces, capitalization/padding of quality labels, absent/questionable labels, trial starts/ends, exact bin count, per-session thresholds, and missing-video classes. Every native trial equals retained plus excluded stimulation; every session has 27-67 retained neurons and 227-426 trials.

### Issues Found and Resolved
- **Full MAT loading stalled**: replaced with targeted h5py reference traversal.
- **Initial neural reshape could scramble axes**: replaced with direct `(trial, neuron, time)` stacking and time-axis smoothing; raw `np.allclose` subsequently passed.
- **Ragged motion energy failed dense conversion**: loaded per-trial cells and aligned each trace using actual frame timestamps.
- **Figure 8 bin discrepancy**: selected 10 ms population settings from Figure 8a-c rather than 5 ms single-unit Figure 8d settings.
- **Manuscript subject count**: paper states six mice, while exact 12 released loader-selected sessions use seven IDs. Exact session and 214 single-unit statistics match; no animal was arbitrarily dropped.
- **No unresolved conversion errors remain.**

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Loss fell steadily from >3.36 to 0.7567 by epoch 200; test loss was 0.7865.
- Full execution completed with `train_decoder.py finished successfully`.
- Sample plots were requested with `--plot-samples` and generated by the validator.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Lick direction | 0.6669 | 0.6443 | 1.93× chance |
| Behavioral context | 0.7956 | 0.7847 | 1.57× chance |
| Outcome | 0.6592 | 0.6099 | 1.83× chance |
| Tongue velocity | 0.5947 | 0.5885 | 1.77× chance |
| Paw velocity | 0.5426 | 0.5276 | 1.58× chance |
| Motion energy | 0.6926 | 0.6858 | 2.06× chance |

- All validation accuracies exceed chance and 1.5× chance.
- Train-validation gaps are small (0.006-0.049 absolute), with no evidence of severe overfitting or leakage.
- Results are recorded in `/app/train_decoder_full_out.txt`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Accuracy / Chance | Expectation from Paper |
|----------|---------------------|--------|-------------------|------------------------|
| Lick direction | 0.6443 | 0.3333 | 1.93× | Paper reports robust ALM choice coding, but no matching categorical classifier accuracy |
| Behavioral context | 0.7847 | 0.5000 | 1.57× | Paper reports context-selective ALM activity/coding dimensions, but no matching accuracy |
| Outcome | 0.6099 | 0.3333 | 1.83× | Correct/error neural differences are reported qualitatively; no matching accuracy |
| Tongue velocity | 0.5885 | 0.3333 | 1.77× | Paper predicts continuous tongue angle and reports R² graphically, not this median-category accuracy |
| Paw velocity | 0.5276 | 0.3333 | 1.58× | No directly comparable paper classifier |
| Motion energy | 0.6858 | 0.3333 | 2.06× | Paper analyzes motion energy and movement-related subspaces, without matching classifier accuracy |

### Checks Performed
1. **Accuracy versus chance**: Every output exceeds chance and the requested 1.5×-chance investigation threshold. Context and paw are the closest but remain above it.
2. **Accuracy versus paper**: A full-text search found no categorical accuracies for the six requested outputs. The paper's neural prediction result uses continuous tongue-angle R² and different subspace regression, so numerical comparison would be invalid. Qualitative expectations (choice, context, and movement information in ALM) agree with achieved performance.
3. **Train-validation gap**: Absolute gaps are lick 0.0226, context 0.0109, outcome 0.0493, tongue 0.0062, paw 0.0150, and motion 0.0068. No training accuracy is remotely 1.5× its validation accuracy.
4. **Class variation**: Full verification shows all declared classes occur. Session medians create balanced finite low/high video classes; missing classes reflect source visibility/video coverage rather than imbalance bugs.
5. **Raw-label/alignment debugging checks**: Step 10 independently verified hit, miss, and no-response labels and a full neural trace against source files with `np.allclose`; go-cue alignment and output construction therefore need no revision.

### Issues Found and Resolved
- No Step 12 issue required conversion changes.
- Lower paw accuracy relative to other movement variables remains above 1.5× chance and is biologically plausible because paw movement is less directly represented in ALM than lick/motion-energy signals.
- Sparse tongue visibility is explicitly represented by the required class 2 and does not create below-chance behavior.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
