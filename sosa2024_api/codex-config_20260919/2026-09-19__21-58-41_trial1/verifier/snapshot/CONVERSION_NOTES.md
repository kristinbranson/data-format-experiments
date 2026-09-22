# Dataset Conversion Notes

## Overview
- **Dataset**: "A flexible hippocampal population code for experience relative to reward" NWB dataset
- **Date started**: 2026-09-20
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
- `pynwb_docs/`
- `train_decoder.py`

Environment check: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124. The required notes file was verified with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `src/reward_relative/preprocessing.py` | LOADING | Build a TwoPUtils session, align Unity VR to imaging frames, load Suite2p output, and attach lick/reward/speed time series and spatial trial matrices. |
| `vr_align_to_2P` (external TwoPUtils; local mock implementation reviewed) | `src/reward_relative/preprocessing.py` and TwoPUtils reference named in docs | PROCESSING | Interpolate continuous position to imaging timestamps, nearest-neighbor categorical variables, preserve event counts by cumulative interpolation/differencing, smooth displacement, and compute speed. |
| `dff` | `src/reward_relative/preprocessing.py` | PROCESSING | Mask activity to trial intervals; subtract 0.7× neuropil; compute per-trial maximin baseline (15-frame smoothing and 300-frame min/max filters); calculate dF/F; smooth by 2 samples; optionally OASIS-deconvolve. |
| `multi_anim_sess` | `src/reward_relative/utilities.py` | PROCESSING | Load curated session pickles, calculate dF/F/events, create trial matrices, derive reward/morph/zone labels and trial sets, and optionally calculate place cells. |
| `get_trial_types` | `src/reward_relative/behavior.py` | PROCESSING | Per trial, derive rewarded status from reward (and reward-zone) events and take the environment morph value. |
| `get_reward_zones` | `src/reward_relative/behavior.py` | PROCESSING | Map scene/trial blocks to reward-zone coordinates and labels A/B/C; switch is normally at trial 30 or session override. |
| `define_trial_subsets` | `src/reward_relative/behavior.py` | CURATION | Define chronological pre/post reward-location trial sets on switch days; optionally split fixed-zone sessions in half. |
| `correct_lick_sensor_error` | `src/reward_relative/behavior.py` | CURATION | Mark a trial's lick stream invalid when excessive cumulative counts indicate a stuck sensor. |
| `get_timeseries_data` | `src/reward_relative/glmUtils.py` | PROCESSING | Extract frame-aligned events and behavior only within `start-1:teleport-1`, calculate reward-relative position, cap licks to binary, reject bad-lick samples, and optionally require speed ≥2 cm/s. |
| `CircularRegression` / `train_vs_test_blocks` | `src/reward_relative/decode.py` | PROCESSING | Paper decoder for circular reward-relative position using deconvolved neural events and repeated blocked cross-validation. |

### Notes
- The source is two-photon calcium imaging, not electrophysiology. The rawest `sess` objects do **not** contain dF/F; the original pipeline computes dF/F and deconvolved `events` later. The distributed NWB representation must be checked in Step 2 to determine whether these processed signals are already present; recomputation is unnecessary if the authoritative exported processing module includes them.
- ROIs were manually curated in the Suite2p GUI (`iscell.npy`) before session construction. Thus the NWB ROI table/series should be treated as the curated cell set unless the data expose an explicit quality mask.
- Paper analyses use deconvolved `events` for continuously sampled decoding and place-cell detection, while dF/F is used for place-field peaks. For this neural-decoder task, the closest paper-matched neural input is deconvolved events at the native imaging-frame cadence.
- Native imaging samples are documented as approximately 15.5 Hz (~64.5 ms). VR variables have already been synchronized to those exact imaging frames; trials span the track start to immediately before teleport. Position trial matrices use 10-cm spatial bins, but the requested decoder is time-aligned, so continuously sampled frame data should be retained instead.
- The paper's Fig. 3 decoder removes samples below 2 cm/s and occupancy-matches positions for its scientific comparison. Those operations are not automatically appropriate for the requested outputs (which explicitly include a `<2 cm/s` class and full trial time courses); this required divergence will be resolved and documented in Steps 4–5.
- Reward-zone coordinates in the reference behavior code are A/X = 80–130 cm, B/Y = 200–250 cm, and C/Z = 320–370 cm. Reward-relative position is referenced to zone start in the paper decoder.
- Trial indices in reference time-series extraction use `start-1:stop-1`, reflecting MATLAB/legacy 1-based event indices. NWB trial timestamps/indices must be inspected directly rather than blindly applying this offset.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- DANDI dataset 001361, version `0.251124.0550`; local metadata report 92,448,350,544 bytes and 152 NWB assets.
- 11 subject directories (`sub-m3`, `sub-m4`, `sub-m7`, `sub-m11`–`sub-m15`, `sub-m17`–`sub-m19`) contain 152 `*_behavior+ophys.nwb` files. Ten subjects have sessions 01–14; m11 has sessions 03–14 (12 sessions).
- Every file was opened with `pynwb.NWBHDF5IO(..., "r", load_namespaces=True)`; no `h5py` access was used. All 152 opened without error.
- `processing/behavior/BehavioralTimeSeries` contains `position`, `speed`, `lick`, `environment`, `trial number`, `trial_start`, `teleport`, `reward_zone`, `autoreward`, and `scanning` sampled on a common behavior/imaging-volume clock, plus sparse `Reward` delivery events. Descriptions/units identify position in cm, speed in cm/s, cumulative licks per imaging frame, environment identity, and binary start/teleport events.
- `processing/ophys` contains `Deconvolved`, raw ROI `Fluorescence`, `Neuropil`, `ImageSegmentation`, and background images. Each neural group has `plane0`, plus `plane1` in 28 dual-plane sessions. Response matrices are time × ROI; conversion must transpose to neuron × time.
- `ImageSegmentation/PlaneSegmentation` contains `pixel_mask`, `planeIdx`, and a two-column `iscell` field. Column 0 is the curated binary Suite2p cell flag; column 1 is confidence. The response series include both accepted and rejected ROIs, so column 0 must filter neurons. `DynamicTableRegion` indices explicitly map each plane series' columns to the shared ROI table.
- All imaging planes report GCaMP7f in hippocampal CA1. There are 124 single-plane sessions and 28 dual-plane sessions. Dual-plane neural metadata reports 31.015625 Hz (scan/volume metadata), but behavior timestamps and row counts show the effective per-plane synchronized sampling interval is consistently 0.064483627204 s (15.5078125 Hz). Both planes have matching row counts and can be concatenated across neurons.
- Trial intervals are not stored as an NWB trials table. They are exactly recoverable from the 1-valued samples in `trial_start` and `teleport`; all 12,216 starts pair and alternate with 12,216 teleports. The valid track interval is `[start, teleport)`: the start sample lies near 0 cm and the teleport sample can contain interpolation between track end and the −50-cm intertrial corridor, matching the reference exclusion of the teleport sample.
- Ten dual-plane sessions have one more neural row than behavior rows (documented one-frame scan termination artifact in the reference pipeline); all valid trial endpoints lie within behavior length, so behavior-indexed trial slices safely ignore the trailing unmatched neural row.
- One m11 session has a spurious `trial number` value 80 after the final of 80 paired trial intervals. Event-defined paired intervals, rather than unique trial-number count, avoid retaining this trailing fragment.
- Across valid trials there are 2,620,514 native samples. Trial durations span 96–3,359 frames (median 190; mean 214.5), or 6.19–216.60 s (median 12.25 s). Long trials are retained because variable trial duration is expected and no reference rule rejects them.
- There are 10,342 rewarded paired trials and 1,874 omissions (84.66% rewarded when classified by sparse reward timestamps within `[start, teleport)`). Three reward events fall outside paired trial intervals and are not trial outcomes under the reference slicing rule.
- `lick` and `reward_zone` are cumulative frame counts (observed maxima 5–8), not intrinsically binary. Reference processing caps lick values >1 to 1. Sixty-nine trials meet the paper code's bad-sensor signature (>35% of trial frames with lick count >2), an edge case to resolve in mapping because dropping entire trials can alter requested coverage.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 curated cell-session instances (`iscell[:,0] > 0`); 260,091 ROIs before curation |
| Neurons / session | 155–2,341; mean 912.36; median 921.5 |
| Subjects | 11 |
| Sessions / subject | 12 (m11) or 14 (all other subjects); 152 total |
| Trials (total) | 12,216 paired start-to-teleport trials |
| Trials / session | 41–100; mean 80.37; median 80 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not stated as a single total because cells were tracked/repeated across days | Paper reports per-session ROIs and across-day tracking rather than summing cell-session instances. | 
| Neurons / session | 155–2,172 putative pyramidal neurons | Methods: manual curation “yielded 155–2172 putative pyramidal neurons per session.” |
| Subjects | 11 switch-task mice in the distributed cohort; paper also studied 3 fixed-condition mice | Methods: “counterbalanced across mice (n = 11 mice)” and fixed cohort “n = 3 mice.” |
| Sessions / subject | One per day over 14 task days; m11 imaging begins day 3 | Methods: one imaging session/day and “imaging started on day 3” for m11. |
| Trials (total) | 12,376 stated for licking QC across 11 switch mice | Methods licking section: “n = 81 out of 12,376 trials removed”; raw NWB paired intervals total 12,216 (investigated in Step 4). |
| Trials / session | 80.5 ± 7.4 (mean ± s.d. across 14 mice/all imaging days); target 80–100 | Methods: “targeted 80–100 trials per session” and reported mean ± s.d. |
| Neural data time bin | ~64.5 ms (15.5 Hz per imaging plane) | Methods: single plane ~15.5 Hz; dual planes interleaved at ~31 Hz for ~15.5 Hz/plane. |
| Behavior data time bin | ~64.5 ms | Methods GLM: behavioral and neural time series sampled at ~15.5 Hz. |
| Reward rate | ~85% | Reward randomly omitted on ~15% of trials. |
| Track / reward-zone geometry | 450-cm track; A 80–130, B 200–250, C 320–370 cm | Hidden reward task methods. |
| Switch timing | After trial 30 on switch days | Methods: “Each switch occurred after 30 trials.” |
| Bad lick trials | 81 (~0.65%), threshold >30% of frames with cumulative count >2 | Methods licking QC. Raw NWB reproduces exactly 81 flagged trials at this threshold, though its paired-trial denominator is 12,216. |
| Low-speed neural exclusion | <2 cm/s excluded from paper spatial neural analyses | Place cell identification and RR decoder methods. |
| Putative interneuron exclusion | dF/F–speed Pearson r >0.5; 0.42 ± 0.85% of cells | Neural processing methods. |


### Processing Details
- Imaging/behavior are synchronized at the ~15.5-Hz imaging-plane rate. Dual-plane acquisitions alternate at ~31 Hz but yield ~15.5 Hz per plane.
- Trials are track traversal from trial start to entry into the teleport/intertrial zone. Teleport data are normally excluded because laser power was reduced there in most sessions and teleport position is not part of the 0–450-cm task.
- dF/F is calculated independently within each trial: neuropil-corrected fluorescence uses a 20-s maximin baseline; dF/F is `(F - baseline) / abs(baseline)`; then a two-sample (~0.129-s s.d.) Gaussian smooth is applied. OASIS with a canonical calcium kernel produces deconvolved event activity. This activity is not interpreted as literal spike rate.
- Paper spatial analyses exclude samples below 2 cm/s and bin the 450-cm track into 45 × 10-cm bins. The requested time-varying speed output includes a `<2 cm/s` category, so the low-speed exclusion cannot be used for this conversion without deleting a requested class.
- The paper's RR coordinate is circular and centered at reward-zone **start**. The requested signed distance bins instead require a linear distance “to any location in the reward zone,” so distance must be zero throughout the 50-cm zone, negative before its start, and positive after its end.
- The paper's RR decoder uses deconvolved calcium events, tenfold CV, 90/10 splits, speed >2 cm/s, position-occupancy matching, and a circular cosine score (chance 0). This task's provided decoder predicts six categorical outputs with balanced accuracy; the two accuracy metrics are not directly comparable.
- Figure 3 reports strong above-shuffle RR-position decoding: before-switch train/test effect sizes RR 2.6, TR 3.3, non-RR 3.2; train before/test after RR 2.7, TR −0.5, non-RR −0.2. RR decoding exceeded shuffle over roughly −104.5 ± 20.1 to +152.7 ± 22.9 cm relative to zone start.

### Curation Steps

**Neuron curation rules**:
Suite2p ROI identification followed by manual removal of multi-soma/dendritic ROIs, ROIs without clear transients, suspected overexpression, and continuously fluctuating putative interneurons. A later dF/F–speed correlation >0.5 exclusion removes an additional 0.42 ± 0.85% for paper analyses. Planes are pooled for dual-plane animals except anatomy-specific analyses.

**Trial curation rules**:
Use valid start-to-teleport task periods. For lick-dependent analysis, set the lick vector to NaN on trials where >30% of 64.5-ms frames contain cumulative lick count >2 (81 trials). Reward/omission analyses require at least three omissions per trial set, but that specialized restriction does not apply to the requested all-trial decoder dataset.

### Decoders Trained
| Decoded variable | Accuracy |
| Circular reward-relative position from RR cells, train/test before | Above shuffle; effect size 2.6 (paper Fig. 3) |
| Circular reward-relative position from RR cells, train before/test after | Above shuffle; effect size 2.7 (paper Fig. 3) |
| Circular reward-relative position from TR / non-RR cells, train before/test after | Not above shuffle; effect sizes −0.5 / −0.2 |
| Requested categorical distance/position/speed/lick/zone/outcome | No directly comparable paper balanced accuracies reported |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal | Reference `dff` computes per-trial maximin dF/F and OASIS events from F/Fneu. | NWB `Deconvolved` is on raw Suite2p scale (example median nonzero tens–hundreds); recomputed reference events are ~0–1 and correlate only ~0.40 median across cells with NWB `Deconvolved`. F and Fneu are available. | Decode from custom deconvolved dF/F; do not interpret as spike rate. | Recompute dF/F and OASIS events from NWB `Fluorescence` and `Neuropil` using the reference logic, neuropil coefficient 0.7, maximin baseline, sigma-2 smoothing, tau 0.7, and effective 15.5078125-Hz plane rate. Do not use the incompatible raw Suite2p `Deconvolved` export. |
| ROI filtering | Manual Suite2p `iscell` curation, then speed-correlation interneuron exclusion in `dayData`; notebook sets r > 0.5. | ROI table contains all ROIs plus `iscell[:,0]`; an exact sample recomputation identified 1/155 additional r > 0.5 cell. | Manual criteria; additional dF/F–speed r > 0.5 exclusion (0.42 ± 0.85%). | First retain `iscell[:,0] > 0`, then recompute dF/F and exclude cells with Pearson r > 0.5 versus speed across valid track samples. Pool planes after filtering. |
| Cell-count range | Code operates on the current curated session masks. | 155–2,341 `iscell` cells/session; max session retains 2,323 after r > 0.5 exclusion in a direct check. | 155–2,172 per session. | Minimum agrees; the released DANDI curation/version has a higher maximum than the manuscript snapshot. Use the explicit current NWB curation plus the documented paper filter rather than inventing an undocumented cutoff to force the old maximum. |
| Trial total | Paired `trial_start_inds` / `teleport_inds`; operate only on these intervals. | 12,216 exact paired intervals in 152 ophys files. | Lick-QC denominator states 12,376 across switch mice. | Difference is exactly 160 trials, consistent with two 80-trial m11 task days 1–2 for which the paper says imaging was absent and which are not in the behavior+ophys assets. The conversion can only and should use the 12,216 paired ophys trials supplied. |
| Trial boundaries | Legacy arrays are sliced `start-1:stop-1`, excluding teleport. | NWB start sample is near 0 cm; NWB teleport-event sample may be an unreliable 450→−50 interpolation. | Track is 0–450 cm; teleport excluded from normal analyses. | NWB event arrays encode the legacy events on a 0-based time axis, so use `[start_event_index, teleport_event_index)` with no extra offset. |
| Dual-plane timing | Divide ~31-Hz scan rate by two planes to get ~15.5 Hz/plane. | Neural metadata says 31.015625 Hz, but rows align one-to-one with 15.5078125-Hz behavior samples; ten files have one extra final neural row. | Interleaved planes are sampled at ~15.5 Hz per plane. | Use behavior timestamps / 64.483627-ms bins as authoritative and align neural rows by index; ignore only unmatched trailing neural row beyond all trial intervals. Concatenate planes over neurons. |
| Reward zones | `get_reward_zones` parses scene and switches after 30 trials; A/B/C = 80–130/200–250/320–370 cm. | NWB identifiers encode initial/final zone. 98.87% of within-trial reward samples fall in the assigned zone ±1 cm; small overshoots reflect frame sampling/auto-reward after zone end. Cross-environment identity changes occur exactly at trial index 30 in all 11 day-8 files. | Same coordinates and switch timing. | Parse zone labels from the NWB identifier and use source trial ordinal `<30` versus `>=30`; use the time-varying NWB `environment` value per trial for ENV1/ENV2. |
| Reward outcome | `get_trial_types`: any reward **and** any zone-entry event within trial. | 10,342 trials have both; 1,822 have neither; 52 omissions have a zone-entry event but no reward. | ~15% random omissions. | Outcome is 1 only when sparse Reward timestamp occurs in `[start, teleport)` (equivalent to reference conjunction here), else 0. Observed reward rate 84.66% matches paper. |
| Lick QC threshold | Helper defaults vary (0.5); continuous decoder helper uses 0.35. | Threshold >0.30 reproduces exactly 81 flagged trials; >0.35 gives 69 and >0.50 gives 44. | >30% of samples with cumulative lick count >2; 81 trials. | Follow the paper's final stated >0.30 rule. Because lick is a required categorical output and cannot be represented as NaN, exclude those 81 whole trials (0.66%) from this multi-output dataset, rather than treating known sensor failure as “no lick.” |
| Low-speed samples | Paper RR decoder and spatial analyses use speed >2 cm/s. | Full trials contain the requested <2 cm/s class. | Same exclusion for paper's spatial scientific analyses. | Required decoder output explicitly includes speed class 0 (<2 cm/s); retain all track samples. This is a task-required divergence, not a reference mismatch. |
| Decoder benchmark | Circular regression/cosine score, cell-class subsets, occupancy matching. | Requested validator uses categorical balanced accuracy on all requested outputs. | Chance circular score 0; Fig. 3 effect sizes, not categorical accuracies. | Use paper results only as qualitative evidence for position decodability; compare final categorical outputs to their own class chance levels. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Fluorescence` + `ophys/Neuropil`, ROI `iscell`, behavior speed | `neural` | Filter `iscell[:,0]>0`; calculate trial-masked neuropil-corrected maximin dF/F; sigma-2 smooth; exclude dF/F–speed r>0.5; OASIS deconvolve; pool planes; slice `[start,teleport)`; transpose/retain as neuron × time float32 | `preprocessing.dff`, `spatial.is_putative_interneuron` | Closest exact match to paper's decoder stream; NWB raw Suite2p `Deconvolved` is not used. |
| Behavior timestamps | `input[0]` | `timestamps[start:stop] - timestamps[start]`, seconds, one value/frame | `glmUtils.get_timeseries_data` interval logic | Time-varying; begins at exactly 0. |
| Behavior `environment` | `input[1]` | Per-trial constant median (verified constant in every trial), 0=ENV1, 1=ENV2; repeat over frames | `behavior.get_trial_types` | Binary. Do not use session identifier alone because day-8 sessions switch environments at trial 30. |
| Paired-trial ordinal / source `trial number` | `input[2]` | Zero-based continuous trial ordinal, repeat over frames | `glmUtils.get_timeseries_data` assigns zero-based `i` | Use interval ordinal to avoid one trailing false trial-number fragment in m11 session 03. |
| Sparse `Reward.timestamps` in previous paired interval | `input[3]` | Previous raw trial reward outcome, 0=omitted, 1=rewarded; first trial defaults to 0; repeat over frames | `behavior.get_trial_types` | Previous outcome is based on the true preceding source trial even if that preceding trial is excluded for bad lick sensing. |
| Behavior `position`, parsed A/B/C interval | `output[0]` distance to reward zone | Signed distance to the nearest point in active zone: `pos-start` before zone, 0 inside inclusive interval, `pos-end` after zone. Classes: `<−50`, `[−50,−10)`, `[−10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`. | `behavior.get_reward_zones`; adapted because paper RR coordinate is circular distance to zone start | Time-varying categorical int64. This implements “distance to any location in reward zone,” so the entire 50-cm zone is class 3. |
| Behavior `position` | `output[1]` absolute position | Classes: `<90`, `90–180`, `180–270`, `270–360`, `>360`; deterministic shared-edge convention assigns exact 90 upward and exact internal/right endpoints to the lower stated closed range. | Direct synchronized `vr_data['pos']` equivalent | Time-varying categorical int64. Exact internal boundaries are vanishingly rare in continuous samples. |
| Behavior `speed` | `output[2]` speed | Classes: `<2`, `2–10`, `10–20`, `20–40`, `>40` cm/s; negative jitter belongs to `<2`. | VR alignment smooths displacement with sigma 5 and computes cm/s | Time-varying; retain all speeds, including paper-excluded slow frames, because class 0 is requested. |
| Behavior cumulative `lick` | `output[3]` lick | After excluding bad-sensor trials (>30% frames with count >2), binarize as `lick > 0` | Paper licking methods; reference helper caps >1 to 1 | Time-varying binary. |
| NWB identifier scene + paired trial ordinal | `output[4]` reward zone location | Parse initial/final A/B/C; switch at ordinal 30; map A/B/C→0/1/2; repeat over frames | `behavior.get_reward_zones` | Per-trial output expanded over time for a rectangular six-channel output. |
| Sparse `Reward.timestamps`, `reward_zone` | `output[5]` reward outcome | 1 if reward delivery timestamp occurs in `[start,teleport)` (and zone-entry is present), else 0; repeat over frames | `behavior.get_trial_types` | Per-trial output expanded over time. Sparse reward criterion is equivalent in supplied trials. |

### Key Decisions
1. **Native temporal bins and variable trial lengths**: retain every 64.483627-ms imaging-volume sample from start through the sample before teleport. This preserves exact neural/behavior alignment; trials need not have equal durations under the target format. `off_start=0`, `off_end=None` because end offset varies.
2. **Paper-matched neural processing**: port the relevant reference functions rather than consume the mismatched NWB raw Suite2p deconvolution. Use float32 output and session-local batch processing to control memory.
3. **Cell curation**: manual `iscell` plus dF/F–speed r>0.5, with no place-cell-only restriction. The paper decoder used selected place-cell subtypes for its scientific question, but the general downstream decoder should receive all curated pyramidal neurons and not leak output-derived place-cell selection.
4. **Trial curation**: retain all paired trials except the exactly 81 paper-identified bad-lick trials. No reward/omission balancing, occupancy matching, or minimum omission count is applied because those were analysis-specific and would distort required output distributions.
5. **Rectangular inputs/outputs**: create input arrays `(4,T)` and output arrays `(6,T)`, repeating per-trial variables across T. This makes every named dimension unambiguous to the validator and decoder.
6. **Session definition**: each NWB file is one session. Preserve deterministic lexical subject/session order. Subject list is unique sorted mouse IDs; brain region list is `['CA1']` and every retained neuron maps to index 0.
7. **Reward-zone parsing**: support both `Env#_LocationA[_to_B]` and cross-environment `Env#_A_to_Env#_B` identifiers. Validate switch point against environment stream and delivery positions.
8. **Boundary conventions**: use explicit comparisons, not implicit `np.digitize` defaults, and describe them in metadata. This eliminates off-by-one ambiguity at stated class edges.
9. **Expected post-QC raw statistics**: 12,135 trials and 2,576,026 samples; reward rate 84.639%; ENV1 51.067% of trials; reward zones A/B/C = 34.380/32.748/32.872% of trials. Expected time-varying distributions are distance `[0.251268,0.101960,0.073264,0.238460,0.020713,0.071783,0.242553]`, absolute position `[0.212008,0.176739,0.231212,0.226414,0.153626]`, speed `[0.116576,0.087350,0.134120,0.319194,0.342760]`, and lick `[0.777420,0.222580]`.

### Planned Sanity Checks
- [x] Independently load a raw NWB trial through `pynwb` and use `np.allclose` to compare recomputed neural events at selected neuron/time coordinates against the converted matrix.
- [x] Independently derive time, environment, trial ordinal, and previous outcome from raw NWB and use `np.allclose` against one converted input trial.
- [x] Independently derive all six categorical outputs from raw NWB for three trials (including switch/omission cases) and use `np.allclose` against converted outputs.
- [x] Assert paired start/teleport counts, strict event alternation, no neural/behavior under-run, constant environment within trials, and at least two retained trials/session.
- [x] Assert all arrays finite; neural/input/output time lengths identical per trial; neuron count matches brain-region index length; categorical domains exactly match declared output values.
- [x] Compare full raw-derived expected trial/sample counts and output distributions above with converted statistics.
- [x] Confirm reward-zone parser for every scene and verify cross-environment switches occur at trial 30.
- [x] Plot raw position/speed/lick and their categorical conversions alongside neural heatmaps for two sessions, including trial boundaries and class thresholds.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implemented `/app/convert_data.py`. It accepts the required output path, defaults to full conversion, supports mutually exclusive `--full` / `--sample`, and supports `--show-processing`. The script compiles and its CLI help runs successfully. It uses `pynwb.NWBHDF5IO` exclusively for NWB access; no direct `h5py` import/access is present. Core functions separately implement scene parsing, trial pairing, paper-matched dF/F, interneuron filtering, OASIS deconvolution, all requested discretizations, session validation, plotting, and aggregate statistics. Sample mode deliberately uses m11 session 03 (single-plane reward switch) and m17 session 08 (dual-plane environment/reward switch) to exercise all major branches.]

Code inefficiencies identified:
[Raw fluorescence and neuropil are large and OASIS must operate on each cell/trial; recomputing the paper signal necessarily reads both streams. Holding rejected ROIs or all raw NWB series would waste memory. Trial arrays must ultimately remain resident until pickle serialization.]

Code speedups added:
[Only `iscell` columns are loaded; processing is vectorized across neurons within each trial; dF/F–speed correlations are vectorized; rejected interneurons are removed before OASIS; planes are processed sequentially; raw arrays and diagnostics are released promptly; outputs use compact int8 and inputs/neural use float32; progress and per-session timings are printed.]

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 741 cell-session instances after all filtering |
| Neurons / session | 154 (m11 s03), 587 (m17 s08) |
| Subjects | 2 (m11, m17) |
| Sessions / subject | 1 each |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| Timepoints | 30,496; 137–338/trial |
| Time from start range | [0, 21.731] s |
| Environment range | [0, 1] |
| Trial ordinal range | [0, 79] |
| Previous outcome range | [0, 1] |
| Distance distribution | [0.137526, 0.094307, 0.042038, 0.261051, 0.022560, 0.082404, 0.360113] |
| Absolute position distribution | [0.257411, 0.192615, 0.251607, 0.160316, 0.138051] |
| Speed distribution | [0.068042, 0.069780, 0.090176, 0.325649, 0.446354] |
| Lick distribution | [0.822600, 0.177400] |
| Reward-zone distribution | [0.573354, 0.426646, 0] (sample contains A/B only) |
| Reward-outcome distribution | [0.116310, 0.883690] |

### Processing Plots Review
`processing_m11_ses-03.png` and `processing_m17_ses-08.png` show trial segmentation, switch location, raw fluorescence/neuropil correction, maximin baseline, dF/F, OASIS events, position/distance classes, speed thresholds, lick binarization, all categorical outputs, and neural heatmaps. Both were visually inspected. Track position progresses smoothly from ~0 to ~450 cm; categorical transitions coincide with the plotted continuous thresholds; distance is exactly zero through the active 50-cm zone; lick events align; neural activity and behavior span identical trial times. No temporal offset or processing anomaly was observed. The validator's `sample_trials.png` was also inspected and shows the expected aligned monotonic time/position signals and categorical outputs.

`verification_sample_out.txt` reports: “Data format is valid, no errors or warnings.” Manual pickle inspection confirmed all required keys; float32 neural/input, integer outputs; `(4,T)` inputs, `(6,T)` outputs; neuron and brain-region dimensions; and accurate metadata.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Load only curated ROI columns; vectorize trial processing/correlation; filter interneurons before OASIS | Avoids processing 121,413 rejected ROIs across the full dataset and unnecessary OASIS work |
| float32 neural/input, int8 outputs, per-plane sequential processing | Bounds memory and halves/quarters avoidable storage compared with float64/int64 |

| Step | Time / Session | Estimated Total Time |
| Sample conversion without serialization overhead | 1.95 s/session average including plot creation (3.89 s total) |
| Neural core from session metadata timing | 0.46 s (small single-plane) to 1.65 s (representative dual-plane) |
| Full conversion estimate | ~438 s (7.3 min) neural core from curated-cell×frame workload ratio 207.4; conservatively 8–10 min including 152 opens, validation, and ~8.8-GiB serialization |

Full estimate is below 15 minutes, so no risky parallel materialization is needed. Sequential processing also avoids multiplying peak memory and random I/O.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Training completed on CUDA; loss decreased monotonically from 2.588915 (epoch 1) to 0.929390 (epoch 200), with test loss 0.905912.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Distance to reward zone (chance 0.1429) | 0.4992 | 0.4554 |
| Absolute position (chance 0.2000) | 0.6584 | 0.5507 |
| Speed (chance 0.2000) | 0.5321 | 0.4884 |
| Lick (chance 0.5000) | 0.7699 | 0.7440 |
| Reward-zone location (chance 0.3333) | 0.8670 | 0.9022 |
| Reward outcome (chance 0.5000) | 0.6727 | 0.5706 |

Every validation accuracy is above uniform chance. Position-related variables substantially exceed chance, supporting neural/behavior alignment. Reward outcome is the weakest expected task because omission is randomized (~15%), but still exceeds chance without leakage from current outcome into decoder inputs (only previous outcome is supplied).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9,518,397,391 bytes (8.865 GiB)
- `verification_full_out.txt`: created; “Data format is valid, no errors or warnings.”
- `conversion_full_out.txt`: created; all 152 sessions completed in 350.08 s (5.83 min), including 8.40 s serialization.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | No summed cell-session total | `iscell`, then r>0.5 exclusion | 138,678 `iscell` instances | 138,276 after 402 (0.290%) r>0.5 exclusions | Yes; follows methods |
| Mean neurons/session | Range 155–2,172 manuscript snapshot | Current masks | 912.36 before r filter | 909.71 after r filter; range 154–2,323 | Current release matches processing; manuscript max differs by data version (documented Step 4) |
| Subjects | 11 switch mice | Session dictionaries | 11 | 11 | Yes |
| Sessions | 14 days except m11 begins day 3 | One sess/file | 152 | 152 | Yes |
| Trials (total) | 12,376 lick-QC denominator includes 160 non-ophys m11 day1–2 trials | Paired start/teleport | 12,216 paired ophys trials | 12,135 after exactly 81 paper lick-QC exclusions | Yes after defined QC |
| Trials/session (mean) | 80.5 ± 7.4 across all 14 mice | Event-paired | 80.37 ± 6.14 (11 supplied switch mice) | 79.84 ± 6.86 post-QC | Yes within cohort/QC difference |
| Time bin | ~64.5 ms (~15.5 Hz/plane) | `frame_rate/n_planes` | 64.483627 ms behavior timestamps | 64.483627 ms | Yes |
| Reward rate (per trial) | ~85% | `get_trial_types` | 84.66% before QC | 84.64% after QC | Yes |
| Environment per trial | ENV1/ENV2; switches day 8 after trial 30 | morph 0/1 | [51.067%, 48.933%] post-QC | [51.067%, 48.933%] | Exact |
| Reward zone per trial | A/B/C counterbalanced | scene mapping, switch after 30 | [34.380%, 32.748%, 32.872%] | [34.380%, 32.748%, 32.872%] | Exact |
| Distance distribution (time) | Not categorical in paper | N/A | [0.251268,0.101960,0.073264,0.238460,0.020713,0.071783,0.242553] | Same | Exact |
| Absolute-position distribution (time) | 0–450 cm | N/A | [0.212008,0.176739,0.231212,0.226414,0.153626] | Same | Exact |
| Speed distribution (time) | Includes movement; paper spatial subset >2 | N/A | [0.116576,0.087350,0.134120,0.319194,0.342760] | Same | Exact |
| Lick distribution (time) | Binary after QC | cap count >1 | [0.777420,0.222580] | Same | Exact |

Spot checks covered sessions 0 (single-plane), 75 (dual-plane environment switch), 82 (large dual-plane), and 151, at first/middle/last retained trials. All arrays were finite; neural/input/output shapes matched; inputs began at 0 with 0.064483627-s increments; dtypes were float32/float32/int8. All 152 source filenames are unique and present in metadata, so no sessions were lost.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: Searched `verification_full_out.txt` for errors, warnings, and invalid status. Its only match is line 1: “Data format is valid, no errors or warnings.” No issue required waiver.
2. **Independent raw-data `np.allclose` checks**: Created `cache/sanity_check_conversion.py`, which does not import conversion code and opens source files with `pynwb`. It independently recomputes neuropil correction, maximin baseline, dF/F, speed correlation filtering, OASIS, inputs, and outputs. Full trial matrices passed `np.allclose` for m11 s03 raw trials 1, 5, 30, 79 and m17 s08 raw trials 1, 0, 30, 79, covering omissions, rewarded trials, pre/post switch, first/last session periods, and both single/dual planes. Explicit neural coordinate `[neuron 3, time 10]` also passed in each checked trial. Output saved to `cache/sanity_check_out.txt`.
3. **Reference-code comparison**:
   - Loading: conversion `process_session` (line 260) uses `pynwb` behavior and ophys containers; reference `create_sess`/`append_session_data` loads aligned `sess` streams. NWB streams are the already-aligned exported equivalents.
   - Neuron filtering: conversion `process_plane` (line 147) applies NWB `iscell`, then `correlations_with_speed` r>0.5; reference `spatial.is_putative_interneuron` (line 754) uses dF/F–speed Pearson correlation and notebook threshold 0.5. Logic matches. Place-cell filtering is intentionally omitted because target is a general neural decoder, not paper cell-class comparison.
   - Trial filtering: `trial_events` (line 76) uses paired start/teleport and excludes teleport; reference `glmUtils.get_timeseries_data` slices the equivalent `start-1:stop-1`. Paper lick threshold >30% is used; specialized occupancy/speed/omission-set filters are intentionally not used because requested outputs require these samples/trials.
   - Temporal alignment: behavior timestamps define 64.483627-ms samples; neural rows use identical indices and planes are pooled over neurons. This matches the reference session's imaging-frame synchronization and dual-plane per-plane rate.
   - Neural binning/processing: `reference_dff` (line 86) and `reference_events` (line 119) port `preprocessing.dff` (line 289): coefficient 0.7, within-trial mean neuropil restoration, sigma-15 smoothing, 300-frame min then max, `(F-F0)/abs(F0)`, sigma-2 smoothing, OASIS tau 0.7 at 15.5078125 Hz. No temporal rebinning is applied.
   - Input construction: raw timestamp, environment, interval ordinal, previous raw outcome are repeated over the matching T. Reference zero-based trial assignment and behavior extraction are followed; these exact requested predictors were not all used by the paper decoder.
   - Output construction: synchronized position/speed/lick and reference scene/outcome mappings are used. Discretization and signed interval-distance are task-required additions.
4. **Key-statistics comparison**: Confirmed 11 subjects, 152 unique source files/sessions, 12,216 raw paired trials, exactly 81 paper-QC trial exclusions, 12,135 retained trials, 138,678 manually accepted ROI instances, 402 r>0.5 exclusions, 138,276 retained neurons, 64.483627-ms bins, 84.64% rewarded retained trials, and exact raw/converted class fractions. Differences from manuscript totals/ranges are fully reconciled in Steps 4 and 9.
5. **All-session edge audit**: Created `cache/edge_case_checks.py` and checked all 152 original NWBs. Every retained converted trial length equals raw `teleport-start`; zero-based input ordinal preserves gaps when bad trials are skipped; first-trial previous outcome is 0; all starts/stops strictly alternate; all 11 cross-environment sessions change exactly at trial 30. Exactly ten dual-plane files have one extra neural row, and each is safely truncated to behavior length beyond its final valid endpoint. Output saved to `cache/edge_case_checks_out.txt`.
6. **Boundary/shape checks**: `validate_session` enforces finite `(neurons,T)/(4,T)/(6,T)` arrays, constant session neuron counts, categorical domains, and ≥2 trials. Full spot checks at sessions 0, 75, 82, 151 and first/middle/last trials confirmed float32/float32/int8, `t[0]=0`, correct dt, and matching region indices. Processing and validator plots show no one-frame shift at trial starts/ends.

### Issues Found and Resolved
- **NWB raw deconvolution mismatch (found before conversion, rechecked here)**: raw Suite2p-scale `Deconvolved` did not match the manuscript signal. Resolved by recomputing reference dF/F/OASIS; independent checks pass.
- **Ten one-row neural overruns**: resolved by behavior-length truncation; row is after the valid behavior range and never part of a trial.
- **One trailing false trial-number value**: resolved by paired start/teleport intervals rather than unique trial-number values.
- **Known bad lick sensing**: exactly reproduces paper's 81 trials at >30%; these trials are excluded so no invalid lick labels enter training.
- No new issue was revealed by this review, so no conversion rerun was necessary. All checks passed on their first post-conversion execution.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Training loss decreased monotonically from 2.907916 (epoch 1) to 0.943137 (epoch 200), a 67.6% reduction; final held-out loss was 0.834703. The exact required command completed successfully on CUDA and its complete output is in `train_decoder_full_out.txt`.
- `sample_trials.png` and `predictions.png` were visually reviewed. Position and reward-distance class changes are synchronous, zone/outcome are constant within trials, lick/speed transitions are plausible, and predictions follow the major behavioral transitions without evidence of a time shift.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Distance to reward zone | 0.6095 | 0.5411 | 3.79x validation chance (1/7) |
| Absolute position | 0.7226 | 0.6646 | 3.32x validation chance (1/5) |
| Speed | 0.6045 | 0.5656 | 2.83x validation chance (1/5) |
| Lick | 0.7679 | 0.7492 | 1.50x validation chance (1/2; 1.4984 before rounding) |
| Reward-zone location | 0.8908 | 0.8440 | 2.53x validation chance (1/3) |
| Reward outcome | 0.7783 | 0.5607 | 1.12x validation chance (1/2); investigated in Step 12 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Expectation from paper |
|----------|------------------------------|--------|--------------------|------------------------|
| Distance to reward zone | 0.5411 | 0.1429 | 3.79x | Paper's related circular RR-position decoder is strongly above shuffle; metric/classes differ. |
| Absolute position | 0.6646 | 0.2000 | 3.32x | Track and reward-relative positions strongly explain population activity; no matching five-class accuracy reported. |
| Speed | 0.5656 | 0.2000 | 2.83x | Speed is an explicit paper GLM predictor; no matching categorical accuracy reported. |
| Lick | 0.7492 | 0.5000 | 1.4984x | Licking is an explicit paper GLM predictor; no matching binary accuracy reported. |
| Reward-zone location | 0.8440 | 0.3333 | 2.53x | Reward-relative remapping predicts strong zone information; no matching three-class accuracy reported. |
| Reward outcome | 0.5607 | 0.5000 | 1.12x | Omissions are randomized (~15%); paper analyzes reward responses but reports no binary outcome-decoding accuracy. |

**Every paper decoder number located:** Figure 3 reports circular reward-relative-position decoder effect sizes rather than balanced accuracies. When trained/tested before the switch they are RR 2.6, TR 3.3, non-RR 3.2; when trained before/tested after they are RR 2.7, TR -0.5, non-RR -0.2. Chance is a circular cosine score of 0. The requested decoder uses all curated cells, discrete classes, different samples, and balanced accuracy, so numerical equality is not meaningful. The appropriate qualitative comparison is met: distance and absolute position are respectively 3.79x and 3.32x their categorical chance levels.

**Train/validation gap check:** train/validation ratios are 1.13, 1.09, 1.07, 1.03, 1.06, and 1.39 in output order. None exceeds the specified 1.5x overfitting threshold. Reward outcome has the largest gap, consistent with a subtle post-delivery response and randomized per-trial omissions, not leakage.

**Low-accuracy investigation:** lick (1.4984x chance before rounding) and outcome (1.12x chance) were investigated under the prescribed protocol. `cache/sanity_check_conversion.py` independently verified raw-NWB labels and complete neural/input/output matrices for more than three specific rewarded/omission trials via `np.allclose`. The processing and decoder plots show correct neural/output synchronization. Lick has substantial variation (22.258% positive frames after bad-sensor-trial removal); outcome has 1,864 omissions and 10,271 rewards, so neither is a degenerate 99% class. Reference neural processing and filtering were rechecked in Step 10 and match the manuscript. `cache/accuracy_review.py` independently re-opened all source NWBs with `pynwb`: current versus previous outcome correlation is only 0.0234 and reward rates conditional on previous omission/reward are 82.74%/85.01%, confirming that the specified previous-outcome input cannot reveal the next randomized omission. Only 50.62% of all retained frames occur at or after reward delivery; before delivery the per-trial outcome is intentionally not causally observable. The modest whole-trial outcome score is therefore expected. Restricting labels to post-reward time or supplying current reward as input would improve accuracy by changing the requested task or leaking the target, so no such change was made.

All six validation scores are above chance. The five behavior/context outputs that are neurally observable throughout most of a trial are at approximately 1.5--3.8x chance; randomized outcome is modestly but genuinely above chance.

### Issues Found and Resolved
- No temporal alignment, label, class-balance, curation, or overfitting issue was found. The only threshold concern was reward outcome below 1.5x chance; exhaustive raw-label, timing, variation, and reference-processing checks explain this as a causal limitation of a randomized per-trial label. No conversion revision or rerun was warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with `README_CACHE.md`
- [x] All files organized; investigation scripts and runtime bytecode are in `cache/`, while required data, logs, and user-facing plots remain at project root

Final artifact audit confirmed that every required file is nonempty. `verification_full_out.txt` begins “Data format is valid, no errors or warnings”; `train_decoder_full_out.txt` contains all six accuracy results and ends successfully. All workflow steps are complete.
