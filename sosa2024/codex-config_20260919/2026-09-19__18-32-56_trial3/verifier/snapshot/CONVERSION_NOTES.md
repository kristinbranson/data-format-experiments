# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward (provided paper/code/data)
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

Environment check: Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 imported successfully. Required checkpoint `ls -la /app/CONVERSION_NOTES.md` passed (file exists).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_sess_pickle` | `code/src/reward_relative/utilities.py` | LOADING | Loads a preprocessed per-animal/per-day `sess` object with `dill`. |
| `vr_align_to_2P` (external TwoPUtils; `vr_align_to_mock_2P` documents equivalent logic) | `code/src/reward_relative/preprocessing.py` | PROCESSING | Synchronizes VR to imaging frames: linear interpolation for position, nearest interpolation for categorical state, and cumulative interpolation/differencing for lick/reward/trial events. |
| `append_session_data` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Adds aligned lick, reward, and speed time series and 10-cm spatial trial matrices. |
| `dff` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Computes per-trial dF/F after 0.7 neuropil subtraction, maximin baseline (300-frame min/max filters), 2-bin smoothing, and optional Suite2p OASIS deconvolution. Outside trial intervals activity is NaN. |
| `multi_anim_sess` | `code/src/reward_relative/utilities.py` | LOADING / PROCESSING | Loads sessions, computes dF/F and deconvolved `events`, derives trial variables, optionally identifies place cells, and assembles animal dictionaries. |
| `get_trial_types` | `code/src/reward_relative/behavior.py` | PROCESSING | For each start-to-teleport interval, derives rewarded outcome from reward and reward-zone signals and obtains environment (`morph`). |
| `get_reward_zones` | `code/src/reward_relative/behavior.py` | PROCESSING | Maps scene and switch trial to reward-zone coordinates/labels: A/X=80–130 cm, B/Y=200–250 cm, C/Z=320–370 cm. |
| `define_trial_subsets` | `code/src/reward_relative/behavior.py` | CURATION | Splits switch sessions chronologically by reward-zone epoch (normally change trial 30); can split nonswitch sessions into halves. |
| `calc_place_cells` | `code/src/reward_relative/spatial.py` | CURATION | Computes place-cell masks from deconvolved events, 10-cm position bins, speed >2 cm/s, 100 shuffles, and p<0.05 (population-shuffle mode in manuscript pipeline). |
| `get_timeseries_data` | `code/src/reward_relative/glmUtils.py` | PROCESSING / CURATION | Extracts trial-only event/behavior samples, creates reward-relative and absolute position, binarizes licks (`>1` to 1), masks sensor-fault trials, and optionally applies speed threshold. |
| `CircularRegression` / `train_vs_test_blocks` | `code/src/reward_relative/decode.py` | PROCESSING | Paper decoder uses timepoint-by-neuron events to decode circular reward-relative position with block cross-validation. |

### Notes
- The data are two-photon calcium imaging, not electrophysiology. Suite2p cell/ROI curation occurs manually before `sess` construction (`iscell.npy`). Therefore no electrophysiology quality filter applies.
- dF/F does need to be computed in the raw preprocessing pipeline, but the reference repository's shared processed products already contain both `dff` and deconvolved `events`; the manuscript's position decoder and place-cell analyses use `events`. If the provided NWB files expose these processed event traces, they should be loaded rather than recomputing dF/F from fluorescence.
- The canonical sample rate is approximately 15.5 Hz (about 64.5 ms/frame); synchronized `vr_data` has one row per imaging frame.
- Reference trial slicing is consistently `start-1:stop-1` in the original one-based-derived `sess` indices. In NWB-native intervals, slicing must instead respect the interval timestamps/index semantics discovered in Step 2, avoiding a blind extra offset.
- Trials extend from track entry/start to teleport/end and exclude intertrial teleport periods. The actual teleport sample is considered position-ambiguous.
- The reference neural stream is nonnegative deconvolved calcium `events`; NaNs outside valid trial acquisition become zero only when fitting the paper decoder. For this conversion, invalid periods will be excluded by trial slicing rather than retained and zero-filled.
- The reference decoder predicts circular position relative to the reward-zone start. The requested output differs explicitly: signed linear distance to any location in the reward zone, so Step 5 must define inside-zone distance as exactly 0 and signed distance before/after the interval.
- Repository review found no native NWB loader; the downloaded DANDI NWB representation is a distribution format for already processed/synchronized streams. NWB loading details must therefore come from the file hierarchy in Step 2 while preserving the reference processing semantics above.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/dandiset.yaml`: DANDI 001361 metadata (version `0.251124.0550`). It declares 152 NWB assets, 11 subjects, and 92,448,350,544 bytes; the local NWB files occupy 86.10 GiB.
- `data/sub-<mouse>/sub-<mouse>_ses-<day>_behavior+ophys.nwb`: one NWB/HDF5 file per mouse/day. Ten mice have sessions 01–14; m11 has sessions 03–14, giving 152 sessions.
- Neural activity is in each `processing/ophys/Deconvolved/plane*/data` float32 array, shaped `(frames, segmented_ROIs_in_plane)`. This is the reference deconvolved event stream. There is one plane in 124 sessions and two planes in all 28 m17/m18 sessions; planes must be filtered and concatenated in `rois`/segmentation-table order, consistent with the paper's pooled-plane analyses. `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell` has columns `(curated_is_cell, cell_probability)` and must be used to retain manually curated cells. Raw fluorescence and neuropil arrays are also available but dF/F/deconvolution need not be recomputed.
- Behavior is under `processing/behavior/BehavioralTimeSeries/`. Synchronized dense float64 frame streams are `position` (cm), `speed` (cm/s), `lick` (cumulative count per imaging frame), `reward_zone` (zone-entry count), `environment` (0=Env1, 1=Env2; -1 before synchronization), `trial number`, `trial_start`, `teleport`, `scanning`, and `autoreward`. `Reward` is a sparse time series of 0.004 mL reward deliveries with timestamps.
- All dense behavioral variables within a file have identical timestamps and lengths. Median frame interval is exactly 0.0644836272 s (15.5078125 Hz) in every session. Ten two-plane files advertise a 31.015625-Hz ophys acquisition rate and contain exactly one extra terminal neural row relative to behavior, matching the documented scan-stop one-frame correction; all trial spans lie within the shared behavior length.
- Trials are exactly delimited by nonzero `trial_start` and `teleport` samples. Every session has matched, ordered starts/ends, one valid trial number and one valid environment per interval, and no NaNs in requested behavior streams within intervals. Track position progresses from approximately 0 to 450 cm; presynchronization/teleport samples include -500/-50 cm and are excluded by trial boundaries.
- Session scene/protocol is encoded at the end of root `identifier` (for example, `Env1_LocationB_to_C`). Reward-zone labels parse from these identifiers. All 11 cross-environment sessions change environment exactly at zero-based trial index 30, corroborating the reference reward switch at trial 30. Zone coordinates are not stored as coordinates but follow the code-defined A=80–130, B=200–250, C=320–370 cm intervals.
- Sparse reward timestamps produce 10,342 rewarded and 1,874 unrewarded trials. Every delivered reward co-occurs with `reward_zone`; 52 unrewarded trials have zone entry without delivery (lapses), as expected. Three sparse reward events occur outside analyzed start-to-teleport intervals and are ignored.
- The lick sensor-error criterion from reference GLM code (`fraction(lick > 2) > 0.35` in a trial) identifies 69 of 12,216 trials. This is a documented data-quality edge case for mapping/curation decisions.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total; curated cells summed across sessions) | 138,678 |
| Neurons / session | mean 912.36; median 921.5; range 155–2,341 |
| Subjects | 11 (`m3`, `m4`, `m7`, `m11`–`m15`, `m17`–`m19`) |
| Sessions / subject | 14 for 10 mice; 12 for m11 (152 total) |
| Trials (total) | 12,216 |
| Trials / session | mean 80.37; median 80; range 41–100 (values 41, 50, 60, 75, 80, 90, 100) |
| Frames (total/full files) | 3,610,877 time rows per plane; 14,164–51,520/session |
| Segmented ROIs before `iscell` curation | 312,110 across sessions; 315–5,085/session |
| Rewarded trials | 10,342 / 12,216 = 84.659% |
| Environment trials | Env1: 6,226; Env2: 5,990 |
| Reward-zone labels (protocol-derived) | A: 4,186; B: 4,010; C: 4,020 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not reported as a cross-session sum | The paper reports per-session counts, not a sum over repeated imaging of cells. | 
| Neurons / session | 155–2,172 putative pyramidal neurons before the additional speed-correlation exclusion | “yielded 155–2172 putative pyramidal neurons per session” |
| Subjects | 11 switch-task mice (the provided dataset); 3 additional fixed-condition mice are described but absent here | “counterbalanced across mice (n = 11 mice)” |
| Sessions / subject | One/day for 14 task days; m11 begins imaging on day 3, implying 152 switch-cohort sessions | “one imaging session per day”; “except m11, for whom imaging started on day 3” |
| Trials (total) | 12,376 reported for lick QC across the 11 switch mice | “n = 81 out of 12,376 trials” |
| Trials / session | Target 80–100; 80.5 ± 7.4 mean ± s.d. across all 14-mouse imaging days | “targeted 80–100 trials per session” |
| Neural data time bin | ~0.0645 s (~15.5 Hz per plane) | “sampling rate of ~15.5 Hz per plane” |
| Behavior data time bin | Raw VR 50–75 Hz; synchronized neural/behavior streams ~15.5 Hz (0.0645 s) | “All behavioral and neural time series were sampled at ~15.5 Hz” |
| Reward rate | Approximately 85% (reward omitted on ~15%) | “Reward was randomly omitted on approximately 15% of trials” |
| Track / zones | 450 cm; A=80–130, B=200–250, C=320–370 cm | “hidden, unmarked 50 cm span at one of three possible locations” |
| Switch timing | After 30 trials on days 3, 5, 7, 8, 10, 12, 14 | “Each switch occurred after 30 trials.” |
| Lick QC | 81/12,376 (~0.65%), detected when >30% of trial frames have cumulative lick count >2 | “removed from subsequent licking analysis” |
| Interneuron QC | dF/F–speed Pearson r >0.5; 0.42 ± 0.85% cells excluded | “Additional putative interneurons were detected for exclusion” |


### Processing Details
- VR frames (50–75 Hz) were synchronized to imaging by Unity TTL pulses and resampled to the per-plane imaging grid (~15.5 Hz). Position is continuous; event counts must be accumulated during downsampling so lick/reward events are not lost.
- Trial analysis runs from entry at 0 cm to track end/teleport; variable-length teleport intervals are generally excluded. Alignment requested here is the first trial frame (`trial_start`).
- Fluorescence preprocessing: manual Suite2p ROI curation; 0.7 neuropil correction in code; per-trial maximin baseline with a 20-s sliding window; dF/F `(F-baseline)/abs(baseline)`; Gaussian smoothing with 2-sample (~0.129-s) s.d.; OASIS calcium-kernel deconvolution. The provided `Deconvolved` stream is the resulting manuscript neural representation.
- Paper spatial analyses exclude samples at speeds <2 cm/s and bin the 450-cm track into 45 × 10-cm bins. Those restrictions served spatial/place-cell analyses and the paper circular position decoder, but the requested decoder explicitly includes a `<2 cm/s` speed output class and time-varying full trials, so slow samples cannot be removed here.
- The paper's RR position is circular and centered on reward-zone start. The requested signed distance-to-any-zone output is different and must use linear distance to the interval (negative before, zero inside, positive after).
- The paper binarizes valid lick counts per frame. Paper lick QC uses >30% frames with cumulative count >2; the repository GLM helper currently uses >35%. The paper is authoritative for final curation.

### Curation Steps

**Neuron curation rules**:
Retain manually curated Suite2p cells (`iscell[:,0] == 1`), pooling planes in m17/m18. The paper additionally excludes putative interneurons with Pearson correlation >0.5 between dF/F and running speed (0.42 ± 0.85%); Step 4 must determine whether the distributed NWB curation already incorporates this or whether an implementable equivalent is required.

**Trial curation rules**:
Use complete ordered trial-start-to-teleport intervals. For outputs involving licking, paper-QC trials with >30% of frames having cumulative lick count >2 are invalid; because the target format has no per-output missing mask, Step 5 must decide whether to exclude those whole trials. Do not include teleport periods. Do not apply the paper's >2 cm/s sample restriction because it would delete the explicitly requested stationary speed category.

### Decoders Trained
| Decoded variable | Accuracy |
| Circular reward-relative position from deconvolved events | Decode score `cos(true-predicted)`; chance=0, perfect=1. Figure 3 reports data-vs-shuffle effect sizes for test-before: RR 2.6, TR 3.3, non-RR 3.2; test-after: RR 2.7, TR -0.5, non-RR -0.2 (77 sessions, 11 mice, seven switch days). No categorical accuracy matching the requested bins is reported. |
| RR position extent above shuffle | z-scored decode >2 from -104.5 ± 20.1 cm to +152.7 ± 22.9 cm relative to zone start across switch days. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Cohort/session coverage | `define_anim_list` uses the switch cohort on task days | 11 mice, 152 sessions; m11 lacks days 1–2 | 11 switch mice; imaging for m11 began day 3 | Fully consistent; convert all 152 available sessions. |
| Trial total | Trial intervals are paired starts/teleports | 12,216 complete intervals | Lick-QC denominator is 12,376 | Difference is exactly 160 trials (two typical 80-trial sessions), consistent with m11's behavior on days 1–2 but absence of imaging/NWB ophys files. Decoder requires neural activity, so only the 12,216 imaged trials are eligible. |
| Trial/session distribution | Typically 80–100, may stop early | mean 80.37, range 41–100 | target 80–100; 80.5 ± 7.4 across all 14 mice | Consistent given early termination and that the paper mean includes the separate fixed-condition cohort. |
| Reward omission | `get_trial_types`: delivery (and zone flag) within trial | 84.659% rewarded, 15.341% omitted/lapsed | ~15% randomly omitted | Match. All deliveries have zone signal; 52 zone-entry/no-delivery trials correctly remain outcome 0. |
| Reward-zone definitions | `get_reward_zones`: X/A=80–130, Y/B=200–250, Z/C=320–370; switch at trial 30 | Scene identifiers encode A/B/C; cross-environment state changes at trial 30 | Same coordinates; every switch after 30 trials | Match. Parse zone labels from scene; change at zero-based trial 30. |
| Imaging planes | Pool planes except anatomical analysis | 124 one-plane and 28 two-plane (m17/m18) sessions; segmentation `rois` links each plane to global ROI rows | m17/m18 two planes sampled at ~15.5 Hz/plane; pooled | Match. Concatenate curated plane streams in global ROI order. A Step 4 audit corrected an earlier plane0-only path description; curated count was already combined and unchanged. |
| Sampling rate | `frame_rate/n_planes`; aligned VR at imaging-frame samples | behavior timestamps always 15.5078125 Hz; two-plane ophys metadata says 31.015625 Hz | ~15.5 Hz per plane | Use aligned timestamps, 64.4836 ms/bin. The 31-Hz metadata is interleaved acquisition, not per-plane row spacing. |
| Terminal frame | Mock aligner explicitly handles a one-frame scan-stop correction | 10 two-plane sessions have one extra terminal neural row; no trial reaches it | Occasional one-frame correction documented by code | Trim each plane to the shared behavior length. No trial data are lost. |
| Manual neuron curation | Suite2p `iscell.npy` updated during manual curation | `iscell` yields 138,678 curated cells; 155–2,341/session | 155–2,172/session | Minimum matches. Three newer m18 NWBs exceed the paper maximum (2,281–2,341). Local DANDI is version 0.251124.0550, while the paper cites 0.250406.0045, so later curation/export differences are the supported explanation. Use the supplied current `iscell` flags rather than forcing an obsolete cap. |
| Putative interneurons | `dayData` calls dF/F–speed correlation >0.5 after place-cell calculation | NWB has raw F/Fneu and curated flags but no saved dF/F or interneuron mask | exclude r>0.5; 0.42 ± 0.85% | Recreate the exact paper dF/F preprocessing only to compute this mask, then exclude r>0.5 cells from decoder neural events. A spot implementation on the max-cell session found 18/2,341 (0.77%), plausible relative to the paper. This filter does not explain the newer paper/data max-count difference. |
| Lick sensor QC | Current GLM helper uses `>0.35` fraction | >0.30 identifies exactly 81 bad trials; >0.35 identifies 69 | >0.30; 81/12,376 (~0.65%) | Paper is authoritative. Use strict fraction >0.30; exclude 81 bad-lick trials entirely because the target has no per-output missing mask and lick is required for every trial. |
| Slow movement | Place-cell/position-decoder analyses use speed >2 cm/s | valid slow samples exist | paper excludes <2 cm/s for spatial analyses | Do not apply this sample filter: the requested output explicitly defines a `<2 cm/s` class and asks for time-varying full trials. This is a required downstream-task difference. |
| Trial slice origin | Legacy `sess` analyses use `start-1:stop-1` because stored indices came from the original pipeline | NWB has explicit `trial_start=1` at the first nonnegative track-entry frame and `teleport=1` at track end | task alignment is entry to the track | For this NWB conversion and explicit requested alignment, use `[trial_start_index:teleport_index)` so time 0 is the actual start event. This avoids including a negative teleport position one frame before alignment. |

Final consistent understanding: each target session is one provided mouse/day NWB; neural data are manuscript deconvolved events from every manually curated, non-interneuron CA1 ROI across both planes; requested behavioral streams are already on the same 64.48-ms imaging grid; complete trials run from the explicit start sample through the sample before teleport; 81 paper-defined faulty-lick trials are removed; task switches occur after trial 30 and reward outcomes derive from sparse deliveries inside each interval.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane*/data` + each plane's `rois` + segmentation `iscell` | `neural` | Keep `iscell==1`, remove recreated dF/F–speed r>0.5 masks, concatenate planes in global ROI order, slice `[trial_start:teleport)`, transpose to `(neurons,time)`, float32 | `preprocessing.dff`; `spatial.is_putative_interneuron`; paper RR decoder | Use deconvolved events without normalization/smoothing; trim only any extra terminal neural row. |
| dense behavior timestamps | `input[0]` | subtract timestamp at explicit trial-start frame; float32 seconds | `vr_align_to_2P` semantics | `time_from_trial_start_s`, time-varying. |
| `environment` | `input[1]` | verify one value/trial in {0,1}; broadcast | `behavior.get_trial_types` (`morph`) | `environment`, binary ENV1=0/ENV2=1. |
| `trial number` | `input[2]` | use native zero-based experimental lap ID; broadcast | `glmUtils.get_timeseries_data` assigns zero-based `i` | `trial_number`; preserve original numbering across any excluded lick-fault trial. |
| sparse `Reward/timestamps` in preceding raw trial | `input[3]` | any delivery in previous `[start,teleport)` and zone signal; first trial defaults 0; broadcast | `behavior.get_trial_types` | `previous_trial_outcome`; uses actual preceding experimental trial even if it is excluded for lick QC. |
| `position` + protocol zone bounds | `output[0]` | signed distance to interval: `pos-start` before, 0 inside inclusive bounds, `pos-end` after; discretize into 7 specified classes | `behavior.get_reward_zones`; requested-task override of circular RR distance | Time-varying `distance_to_reward_zone`. |
| `position` | `output[1]` | 5 requested 90-cm bins over 0–450 cm | paper track definition | Time-varying `absolute_position`. |
| `speed` | `output[2]` | requested thresholds 2,10,20,40 cm/s | synchronized VR speed | Time-varying `speed`. Negative smoothing artifacts naturally remain in `<2` class. |
| `lick` | `output[3]` | `lick > 0` to 1, otherwise 0 | `glmUtils.get_timeseries_data`; paper lick method | Time-varying `lick`; exclude paper-QC fault trials first. |
| scene suffix + trial index relative to 30 | `output[4]` | A→0, B→1, C→2; broadcast | `behavior.get_reward_zones` | Per-trial `reward_zone_location`, represented as constant time series so it can coexist with time-varying outputs. |
| sparse `Reward/timestamps` + `reward_zone` | `output[5]` | delivery and zone-entry evidence in current interval →1, else 0; broadcast | `behavior.get_trial_types` | Per-trial `reward_outcome`, represented as constant time series. |
| NWB subject ID | `subjects`, `subject_idx` | sorted natural mouse order; per-session index | NWB metadata | 11 subjects. |
| ImagingPlane `location` | `brain_regions`, `brain_region_idx` | normalize “hippocampus, CA1” to `CA1`; all retained neurons index 0 | paper imaging target | One brain region. |

### Key Decisions
1. **Session scope**: Include all 152 imaged NWB sessions. The decoder task is not limited to switch days, and every available session has the requested variables and ≥2 valid trials.
2. **Trial bounds/alignment**: Slice from the frame whose `trial_start` value is nonzero through (excluding) the frame whose `teleport` value is nonzero. This makes `input[0,0]==0`, excludes teleport, and follows the requested event rather than applying the legacy `sess` one-based offset to native NWB event indices.
3. **Common time grid**: No temporal resampling is needed because neural rows and dense behavioral samples share the imaging grid. Use timestamp differences directly; metadata bin size is 64.4836272 ms. For ten files, ignore the unused one extra terminal neural frame.
4. **Neural representation**: Use author-provided OASIS-deconvolved calcium events, the stream used by their decoder. Recomputing deconvolution would add numerical differences without benefit. Recompute dF/F only transiently to apply the paper's r>0.5 interneuron mask.
5. **Neuron filtering**: Apply manual `iscell` and paper interneuron filtering, but do not restrict to place cells: requested outputs include speed/lick/outcome as well as position, so non-place neurons can carry relevant information and the downstream decoder learns its own projections.
6. **Lick-fault trials**: Exclude whole trials when >30% of frames have `lick>2`. The paper marks lick data invalid, and the target has no missing-label mask. Expected retained total is 12,135 trials (12,216−81).
7. **Slow samples**: Retain them because `<2 cm/s` is an explicit required output class. The paper's speed>2 restriction is analysis-specific and incompatible here.
8. **Mixed variable dimensionality**: Store both inputs and outputs as dense arrays, `(4,T)` and `(6,T)`. Broadcast per-trial values across T; this is required because time is varying and one ndarray cannot mix a `(T,)` row with scalars. It also matches `train_decoder.py`'s timepoint-wise target convention.
9. **Exact output boundaries**: distance classes are `<−50`, `[−50,−10)`, `[−10,0)`, exactly 0, `(0,10]`, `(10,50]`, `>50`; position classes are `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360`; speed classes are `<2`, `[2,10)`, `[10,20)`, `[20,40]`, `>40`. Exact-boundary conventions follow every explicit strict inequality in the request; continuous data make unresolved endpoints negligible.
10. **Distance definition**: “distance to any location in the reward zone” is zero everywhere inside the 50-cm interval, negative to its near/start edge before entry, and positive to its far/end edge after exit. This uniquely gives the requested negative/zero/positive semantics.
11. **Reward outcomes/history**: Match `get_trial_types`: current outcome is any sparse delivery with reward-zone evidence in the interval. Prior outcome refers to the preceding raw experimental trial, not preceding retained trial. First-trial history is 0 because the requested binary has no missing category.
12. **Dtypes/storage**: neural and inputs float32; outputs int8 categorical; indices integer. Preserve trial lists and variable lengths to avoid padding and duplicated invalid periods.
13. **Metadata**: `off_start=0.0`; `off_end=None` because trials have variable duration. Include source version, sampling rate, zone bounds, bin rules, curation, per-session scene/file/count information, and explicit variable-length note.
14. **Available but intentionally unused variables**: raw `Fluorescence`, `Neuropil`, `scanning`, `autoreward`, `reward_zone` event magnitude, ROI pixel masks/probabilities, and background images are retained in source but are not decoder inputs/outputs requested here. `scanning` is used as an integrity check; raw fluorescence/neuropil are used only for interneuron QC.

### Planned Sanity Checks
- [ ] Directly reload raw NWB (independent of conversion helpers) and `np.allclose()` selected converted neural values for both a one-plane and two-plane session, including first/last retained neurons and trial endpoints.
- [ ] Directly reconstruct one trial's four inputs from NWB timestamps/environment/trial ID and preceding reward timestamps; compare all rows with `np.allclose()`.
- [ ] Directly reconstruct one trial's continuous signed distance, discretized position/speed/lick, zone label, and outcome; compare all six output rows with `np.allclose()`.
- [ ] Assert 152 sessions, 11 subjects, expected source trials 12,216, exactly 81 lick-QC exclusions, and 12,135 retained trials; every retained session must have ≥2 trials.
- [ ] Assert equal T across neural/input/output per trial; fixed neuron count within session; plane/global-ROI concatenation ordering; no NaN/Inf; outputs integer and within `[0,nclasses-1]`.
- [ ] Assert every trial begins at time 0 and nonnegative track entry, ends before teleport, has monotonically increasing uniform time, and contains one source trial number/environment.
- [ ] Compare reward rate, environment/zone distributions, trial/session distributions, and neuron/session range against Steps 2–4 after accounting for explicit exclusions.
- [ ] Unit-test every discretization edge with values immediately below, exactly at, and immediately above each threshold.
- [ ] Visually plot raw continuous variables, classes, start/end markers, and representative neural traces for up to two sessions, including one two-plane example when selected.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `/app/convert_data.py` with the required positional output path, default/explicit `--full`, `--sample`, and `--show-processing` modes.
- Sample mode deliberately selects m12 day 10 and m18 day 11: together they exercise all A/B/C zone labels, both environment values, switch logic, one-/two-plane loading, and both reward-outcome classes in each deterministic held-out split. The initial m12/m18 day-3 pair was replaced during Step 8 because one held-out session contained no omission trials, making sample balanced accuracy ill-posed; full-mode scope/logic were unchanged.
- Implemented explicit NWB integrity validation, natural session ordering, paper dF/F-based interneuron QC, manual-cell/plane mapping, lick-fault removal, start-to-teleport alignment, requested class construction, metadata/session provenance, edge tests, and final shape/dtype/range validation.
- `--show-processing` produces one six-panel figure per processed sample session showing source boundaries/rejections, neuron curation, aligned events, continuous vs discretized position, time-varying class traces, and all target outputs.
- `python3 -m py_compile /app/convert_data.py` and CLI help both completed successfully.

Code inefficiencies identified:
The exact paper interneuron filter requires reading raw fluorescence and neuropil and applying per-trial image filters before deconvolved events are loaded. The final pickle necessarily copies each variable-length neural trial because the target format is a list of independent arrays.

Code speedups added:
Interneuron QC is vectorized over 128-cell blocks, uses sufficient statistics rather than retaining dF/F, and processes one session at a time. Neural streams are loaded only for retained cells and only through the behavior-aligned length. Inputs/outputs are NumPy-vectorized, no resampling is performed, and timing/remaining estimates print per session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total; session-summed) | 2,700 |
| Neurons / session | 1,327 (m12 day 10); 1,373 (m18 day 11) |
| Subjects | 2 (`m12`, `m18`) |
| Sessions / subject | 1 each |
| Trials (total) | 159 retained / 160 source; one m12 lick-fault trial excluded |
| Trials / session | 79, 80 |
| Timepoints / trial | 103–402; sample mean 176.42 |
| Time input range | [0, 25.9] s |
| Environment range | [0, 1] |
| Trial-number range | [0, 79] |
| Previous-outcome range | [0, 1] |
| Distance-class distribution | [0.277, 0.102, 0.132, 0.222, 0.019, 0.066, 0.184] |
| Absolute-position distribution | [0.158, 0.195, 0.352, 0.161, 0.135] |
| Speed distribution | [0.122, 0.060, 0.106, 0.227, 0.484] |
| Lick distribution | [0.750, 0.250] |
| Reward-zone distribution | [0.139, 0.603, 0.257] |
| Reward-outcome distribution | [0.244, 0.756] |

### Processing Plots Review
Both `/app/processing_m12_ses-10.png` and `/app/processing_m18_ses-11.png` were visually inspected. Trial starts align to ~0 cm, every last included frame is at ~450 cm, teleport/ITI periods are excluded, m12's faulty-lick trial is visibly marked, neural traces begin at time 0, and continuous position/distance/speed transitions agree with their discrete classes. Initial plot review on the first sample pair found two display-only issues (ambiguous teleport-frame marker positions and int8 overflow in `position_class*90`); both were fixed before the final sample run. No remaining alignment or discretization anomaly was found.

`/app/sample_data.pkl` is 145 MiB. Manual structure/range/distribution inspection passed. `/app/verification_sample_out.txt` reports: “Data format is valid, no errors or warnings.” Every requested output class occurs in the combined representative sample. The original sample pair's deterministic test split contained no omissions for one session; Step 8 raw-label checks showed this was a sampling defect, so sample mode was changed to a class-complete representative pair and Steps 7–8 were rerun.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Blockwise dF/F QC with sufficient statistics | Avoids materializing per-session dF/F and bounds memory while matching reference filters |
| Load deconvolved data only for retained cells | Avoids copying 312,110 uncurated ROI streams into output |
| Native synchronized frame grid | Eliminates resampling/interpolation |

| Step | Time / Session | Estimated Total Time |
| Sample conversion including plots | 3.25 s (1-plane m12); 3.86 s (2-plane m18) | 7.11 s for two representative sessions |
| Full conversion (cell×trial-frame work proxy) | full/sample work ratio 62.12–62.27 | ~442 s (~7.4 min) plus full-pickle serialization; safely below 15 min |
| Expected full output size | sample 0.142 GiB; neural-value ratio 62.12 | ~8.82 GiB |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Distance to reward zone | 0.7833 | 0.4899 (chance 0.1429) |
| Absolute position | 0.8237 | 0.6521 (chance 0.2000) |
| Speed | 0.7321 | 0.4807 (chance 0.2000) |
| Lick | 0.8223 | 0.6975 (chance 0.5000) |
| Reward-zone location | 0.9362 | 0.7729 (chance 0.3333) |
| Reward outcome | 0.6422 | 0.5255 (chance 0.5000) |

Training completed successfully on GPU. Mean normalized loss decreased monotonically in the reported checkpoints from 129.1149 (epoch 1) to 1.1350 (epoch 200); test loss was 5.9394. Every validation balanced accuracy exceeded uniform chance. The first sample attempt had reward-outcome validation 0.286 because one session's held-out set contained zero omissions; three raw-trial labels matched converted labels exactly, and deterministic split inspection isolated this as unstratified sample-class coverage rather than conversion error. Selecting the final representative pair yielded 4/12 and 6/10 omission/reward counts in held-out trials and resolved the check without altering full conversion or label semantics.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9,517,305,868 bytes (8.864 GiB)
- `conversion_full_out.txt`: created; 152 sessions converted in 383.45 s and serialized in 7.79 s (391.23 s total)
- `verification_full_out.txt`: created; validator reports “Data format is valid” with no errors or warnings

The measured full run was faster than the Step 7 conservative estimate and required no further optimization. Manual direct-source spot checks passed for raw trial 5 in one-plane `m12_ses-03` (218 samples, 1,143 neurons), two-plane `m18_ses-01` (184 samples, 2,281 neurons), and one-plane `m13_ses-01` (195 samples, 686 neurons). In each case, independently loaded NWB neural samples and all input rows were `np.allclose()` to the pickle, including trial endpoints.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | No cross-session sum; extra r>0.5 cells excluded | `iscell` then dF/F–speed r>0.5 | 138,678 manual cells; 409 r>0.5 | 138,269 | Yes |
| Mean neurons/session | 155–2,172 before additional exclusion in cited version | Same two filters | 912.36 manual mean; range 155–2,341 | 909.66; range 154–2,323 | Yes, after filter; supplied DANDI revision has several larger sessions |
| Subjects | 11 switch mice | 11 switch-animal IDs | 11 | 11 | Yes |
| Sessions | 152 inferred: 14/mouse except m11 days 3–14 | available animal/day files | 152 | 152 | Yes |
| Trials (total) | 12,376 behavior trials; 81 lick-fault | complete imaged intervals | 12,216 imaged; 81 bad | 12,135 retained | Yes: absent m11 days 1–2 explain 160; explicit QC explains 81 |
| Trials/session (mean) | target 80–100; reported 80.5 ± 7.4 on broader cohort | all complete intervals | 80.37, range 41–100 | 79.84, range 41–100 after QC | Yes |
| Time input range (s) | ~15.5-Hz trial samples | imaging-aligned VR | 0–216.5 over retained trials | 0–216.5 | Yes |
| Environment / trial / previous outcome ranges | binary / zero-based lap / binary | same | [0,1] / [0,99] / [0,1] | [0,1] / [0,99] / [0,1] | Yes |
| Reward outcome | ~85% rewarded | sparse delivery + zone entry | 10,271/12,135 = 84.64% | time-weighted `[0.158, 0.842]`; trial rate 84.64% | Yes |
| Reward-zone trial counts | A/B/C protocol | scene-derived A/B/C | `[4172,3974,3989]` | `[4172,3974,3989]` | Yes |
| Distance output distribution | Not reported for requested classes | N/A (paper uses circular RR) | independently reconstructible | `[.251,.102,.073,.238,.021,.072,.243]` time-weighted | Internally consistent |
| Position output distribution | 450-cm track; paper uses 45 bins | requested five bins | independently reconstructible | `[.212,.177,.231,.226,.154]` | Internally consistent |
| Speed output distribution | <2 removed only in paper spatial analyses | synchronized speed | independently reconstructible | `[.117,.087,.134,.319,.343]` | Required task-specific retention |
| Lick output distribution | binary valid-frame lick | binarize count | independently reconstructible after QC | `[.777,.223]` | Yes |

All categorical distributions above are weighted by timepoints, matching the validator. Trial-level counts are reported separately where appropriate. No data expected by the stated curation were lost: the accounting identity is 12,216 source intervals − 81 paper-QC lick trials = 12,135 converted trials, and 138,678 curated ROIs − 409 putative interneurons = 138,269 converted neurons.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Full validator log**: Read all of `verification_full_out.txt`. It begins “Data format is valid, no errors or warnings”; therefore there are no validator warnings requiring dismissal. All shapes, ranges, class names/counts, and subject/region indices are valid.
2. **Independent raw-data `np.allclose` checks**: `cache/critical_review.py` reads NWB HDF5 datasets directly and never imports `convert_data.py`. It independently rebuilt raw trial 5 from `m12_ses-03` (one plane, 1,143 cells × 218 samples), `m18_ses-01` (two planes, 2,281 × 184), and `m18_ses-03` (two planes, 2,323 × 224 after independently finding 18 r>0.5 cells). For every trial, the full neural matrix, all four inputs, and all six outputs passed `np.allclose` against the pickle. Thus this supplies separate neural, input, and output sanity checks and covers plane pooling plus nontrivial neuron filtering.
3. **Boundary checks**: Independently tested values just below, exactly at, and just above every distance, position, and speed class boundary with `np.allclose`; all passed.
4. **Data loading comparison**: Reference `load_sess_pickle` loads earlier preprocessed dill sessions; the supplied data are the corresponding DANDI NWBs, for which the repository has no loader. `convert_data.py::convert_session` therefore reads the NWB datasets directly while selecting the same synchronized behavior and OASIS `Deconvolved` event streams. This is a format adaptation, not a processing change.
5. **Neuron/trial filtering comparison**: Reference manual `iscell`, `preprocessing.dff`, and `spatial.is_putative_interneuron` correspond to `compute_interneuron_mask` plus `load_deconvolved`. Parameters match: 0.7 neuropil, Gaussian 15, min/max 300, Gaussian 2, Pearson r>0.5. Paper lick QC (>30% frames with count>2) corresponds to `bad_trials`; the current GLM helper's stale 35% differs, so the paper threshold is intentionally authoritative. Place-cell and speed>2 filters are intentionally not used because the requested decoder covers all cells/behaviors and explicitly requires a <2 class.
6. **Temporal-alignment comparison**: Reference processed-session code slices `start-1:stop-1` because its saved event indices inherited one-based conventions. The NWB contains explicit binary start and teleport samples. `trial_bounds` and `convert_session` use `[start:teleport)`, verified directly: time starts at 0, track-entry position is nonnegative, and the ambiguous teleport sample is absent. The reference legacy slice is reproduced only for the interneuron-QC computation. Ten harmless extra terminal neural rows are outside all trial spans and trimmed.
7. **Binning comparison**: Neither neural nor behavior streams are resampled because NWB already puts both at 15.5078125 Hz; metadata correctly gives 64.483627 ms. The paper's spatial analysis uses 45 × 10-cm/circular bins, whereas `discretize_distance`, `discretize_position`, and `discretize_speed` implement the decoder task's mandatory bins. This is the required downstream-task difference.
8. **Input construction comparison**: `convert_session` obtains timestamp-relative time and native synchronized environment/trial-number streams directly. `reward_outcomes`, based on reference `get_trial_types`, supplies history from the actual preceding raw trial. Independent reconstruction passed all four rows on the three test trials.
9. **Output construction comparison**: Reference `get_reward_zones` and `get_trial_types` provide zone/outcome semantics; GLM code binarizes lick counts. Requested linear signed distance and categorical position/speed bins replace the paper's circular/continuous targets. Independent reconstruction passed all six rows on all test trials.
10. **Key-statistics comparison**: Independently asserted 152 sessions/11 mice; 12,216 source − 81 faulty-lick = 12,135 retained trials; 138,678 manual cells − 409 r>0.5 = 138,269 neurons; and ≥2 trials/session. Reward rate 84.64% matches the paper's approximately 85%, sample interval matches ~15.5 Hz, and zone counts are balanced as expected. The supplied later DANDI revision explains the documented larger maximum cell count; no forced cap is scientifically justified.
11. **Edge-case audit**: Checked two-plane global ROI sorting, an r-filtered session, exact class boundaries, first-trial previous-outcome default, raw-trial history across exclusions, variable trial lengths, early-ended sessions, missing m11 days 1–2, switch at raw index 30, 10 extra terminal neural rows, sparse reward timestamps, faulty lick trials, and position values immediately around trial endpoints. No off-by-one mismatch was found.

### Issues Found and Resolved
- No new conversion defect was found in this critical-review iteration. Every independent array reconstruction and accounting assertion passed on the first Step 10 run; output is saved in `cache/critical_review_out.txt`.
- Earlier development issues already resolved before full conversion were rechecked here: global ordering of two-plane ROIs, use of native NWB event bounds rather than the legacy offset, plot-only teleport-marker ambiguity, and integer-overflow in a visualization. None affects the final pickle.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. GPU training completed all 200 epochs; loss fell from 181.845873 (epoch 1) to 1.206677 (epoch 200), and test loss was 1.121878. The small epoch-170-to-180 fluctuation (1.2583 to 1.2609) is normal near convergence; the overall trend is unambiguous.
- Split: 9,702 training trials and 2,433 validation trials. The script finished successfully and produced `sample_trials.png` and `predictions.png`.
- Visual review: sample trials show aligned nonnegative deconvolved activity, monotonically increasing time/position classes, per-trial constant labels, and plausible speed/lick patterns. Prediction plots show strong tracking of position-derived classes and plausible noisier predictions for lick/outcome, without a systematic temporal offset.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Distance to reward zone | 0.4554 | 0.4048 | chance 0.1429; 2.83× chance validation |
| Absolute position | 0.5261 | 0.4998 | chance 0.2000; 2.50× chance |
| Speed | 0.4107 | 0.3850 | chance 0.2000; 1.93× chance |
| Lick | 0.6256 | 0.6162 | chance 0.5000; 1.23× chance |
| Reward-zone location | 0.8390 | 0.8067 | chance 0.3333; 2.42× chance |
| Reward outcome | 0.5641 | 0.5144 | chance 0.5000; 1.03× chance |

All six validation accuracies exceed chance. Training-to-validation ratios are 1.125, 1.053, 1.067, 1.015, 1.040, and 1.097 respectively; none approaches the >1.5 overfitting criterion.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Ratio to chance | Paper comparison |
|----------|------------------------------|--------|-----------------|------------------|
| Distance to reward zone | 0.4048 | 0.1429 | 2.83× | Closest requested analogue to RR position, but categorical linear interval-distance is not the paper's circular position target |
| Absolute position | 0.4998 | 0.2000 | 2.50× | Paper does not report categorical track-position accuracy |
| Speed | 0.3850 | 0.2000 | 1.93× | Paper uses speed as a covariate/filter, not a decoded target |
| Lick | 0.6162 | 0.5000 | 1.23× | Paper uses licking as behavior/covariate, not a decoded target |
| Reward-zone location | 0.8067 | 0.3333 | 2.42× | Paper does not report direct A/B/C decoding accuracy |
| Reward outcome | 0.5144 | 0.5000 | 1.03× | Paper uses outcome as a GLM covariate; no outcome-decoding accuracy is reported |

The paper's only neural decoding accuracy is its circular RR-position analysis. Its per-timepoint score is `cos(true−predicted)`, where 1 is perfect and 0 is random—not categorical balanced accuracy. Figure 3 reports data-versus-shuffle effect sizes, not raw accuracies: test-before RR=2.6, TR=3.3, non-RR=3.2; test-after RR=2.7, TR=−0.5, non-RR=−0.2 (77 switch sessions). It also reports post-switch RR decoding above z=2 from −104.5±20.1 cm to +152.7±22.9 cm relative to zone start. Therefore no paper number is numerically commensurate with any target above. Qualitatively, the strong requested distance/position/zone results agree with the paper's conclusion that CA1 population events encode position and reward-relative structure.

**Accuracy vs chance**: Every output is above chance. Distance, absolute position, speed, and zone exceed 1.5× chance. Lick and outcome do not, so all mandated debugging checks were performed:

1. Direct raw verification: `cache/accuracy_review.py` loaded the NWB independently and used `np.allclose` on lick and outcome for three concrete m12/day-3 trials: raw trial 13 (omitted, lick fraction .1420), trial 0 (rewarded, .1122), and trial 58 (omitted, .2876). All passed. The broader Step 10 audit also reconstructed all six outputs on three sessions.
2. Temporal alignment: `sample_trials.png`, `predictions.png`, and the two detailed processing plots show neural and outputs on the same start-aligned frame grid. Position/speed changes and neural traces share endpoints; no shift is visible. Direct arrays use identical `[start:teleport)` indices.
3. Variation: lick fractions are `[.7774,.2226]` and outcome fractions `[.1577,.8423]`, so neither is near a 99% single class. Balanced loss is used by the supplied trainer.
4. Neural filtering: manual `iscell`, two-plane pooling, and exact paper r>0.5 exclusion were independently repeated; m18/day-3 reproduced all 18 exclusions and all neural values.
5. Reference agreement: lick is the synchronized source count binarized at >0 after excluding exactly the paper's 81 sensor-fault trials; reward outcome matches sparse delivery plus reward-zone evidence from `get_trial_types`.

The modest lick accuracy is scientifically plausible for sparse lick events decoded from 15.5-Hz calcium signals. Reward omission is randomized at ~15%; broadcasting its required per-trial label from trial start means most pre-reward samples cannot contain causal evidence of the current random outcome, so near-chance balanced accuracy is expected. Changing this to a reward-event label or withholding pre-delivery samples would contradict the requested “reward outcome, per-trial” definition and would create an unjustified accuracy boost.

**Train/validation gap**: train/validation ratios are distance 1.125, absolute position 1.053, speed 1.067, lick 1.015, zone 1.040, and outcome 1.097. None exceeds the 1.5× threshold; there is no evidence of material overfitting or leakage.

### Issues Found and Resolved
- No conversion issue was found. The lower lick/outcome accuracies triggered the full investigation above, and all raw-label, alignment, variation, filtering, and reference-processing checks passed. Consequently no scientifically justified conversion change was made and no rerun was needed.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading example, schema, curation, statistics, validation results, and reproduction commands
- [x] cache/ folder created; `README_CACHE.md` documents independent audit scripts, outputs, superseded plots, and bytecode
- [x] All files organized: required datasets/scripts/logs and final plots remain at project root; investigation-only artifacts are under `cache/`

Final deliverable audit confirms all requested files exist. `CONVERSION_NOTES.md` records the complete ordered workflow, decisions, raw-data comparisons, full validation, decoder results, and both critical reviews.
