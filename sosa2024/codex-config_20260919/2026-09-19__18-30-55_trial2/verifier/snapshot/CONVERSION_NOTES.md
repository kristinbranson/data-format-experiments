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

Environment check: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124. Required imports succeeded. Checkpoint confirmed with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `code/src/reward_relative/preprocessing.py` | LOADING | Construct a TwoP session; synchronize VR to imaging; add lick, reward, and speed streams and spatial trial matrices. |
| `vr_align_to_2P` (external TwoPUtils; documented by repository) / `vr_align_to_mock_2P` | `code/docs/preprocessing_guide.md`, `preprocessing.py` | PROCESSING | Align VR samples to imaging frames; linearly interpolate time/position, nearest-neighbor interpolate categorical state, and cumulative-interpolate event counts so events are not lost; smooth displacement over 5 frames to derive speed. |
| `dff` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Restrict fluorescence to valid trial periods, subtract 0.7 neuropil, estimate per-trial maximin baseline (300-frame min/max filters), calculate dF/F, smooth by 2 frames, and optionally OASIS-deconvolve to `events`. |
| `multi_anim_sess` | `code/src/reward_relative/utilities.py` | LOADING / PROCESSING | Load each session; recompute dF/F and deconvolved events; derive trial outcome, environment, reward-zone coordinates/labels, trial subsets, and optional place-cell masks. |
| `get_trial_types` | `code/src/reward_relative/behavior.py` | PROCESSING | For each start-to-teleport interval, define rewarded outcome from reward and reward-zone events and obtain the unique environment morph. |
| `get_reward_zones` | `code/src/reward_relative/behavior.py` | PROCESSING | Map scene and switch trial to per-trial zone A/B/C coordinates. Canonical zones are A/X=[80,130], B/Y=[200,250], C/Z=[320,370] cm. |
| `define_trial_subsets` | `code/src/reward_relative/behavior.py` | CURATION | Split switch sessions chronologically by reward-zone identity (default switch trial 30, or session-specific override). |
| `calc_place_cells` | `code/src/reward_relative/spatial.py` | CURATION | Calculate spatial-information place-cell masks from deconvolved events with 100 shuffles, p<0.05, and (for the paper pipeline) speed >2 cm/s. |
| `get_timeseries_data` | `code/src/reward_relative/glmUtils.py` | PROCESSING / CURATION | Extract synchronized continuous events and behavior only within start-to-teleport trial spans; compute reward-relative position; binarize lick counts; exclude invalid/NaN lick samples and, in Fig. 3, speed <2 cm/s. |
| `CircularRegression` / `train_vs_test_blocks` | `code/src/reward_relative/decode.py` | PROCESSING | Paper decoder: regularized circular regression from timepoints × neurons to reward-relative circular position with repeated held-out folds. |

### Notes
- The data are two-photon calcium imaging, not electrophysiology. Raw `sess` objects do not contain dF/F; the author pipeline computes it later. The provided higher-level/NWB data may already contain the resulting processed `events`; this will be checked in Step 2 before deciding whether recomputation is needed.
- Manual Suite2p curation writes `iscell.npy` before session creation. The analysis objects operate on those curated ROIs; no additional generic firing-quality threshold is applied by the Fig. 3 decoder. Place-cell masks are scientific cell classes, not a global recording-quality filter.
- The paper's Fig. 3 decoder explicitly uses continuous deconvolved `events`, replaces remaining neural NaNs by zero, restricts to running speed >2 cm/s, and occupancy-matches reward-relative position. Those extra restrictions serve its cell-class comparison. For the requested trial-aligned decoder, retaining the complete valid start-to-teleport trajectory is necessary to represent stops, lick behavior, speed class 0, and every requested time-varying output; any divergence will be documented.
- Indexing convention is important: reference processing generally writes/reads imaging samples `[trial_start - 1 : teleport - 1]`, reflecting the stored 1-based event indices. Native NWB interval timestamps/row indices must be inspected to avoid copying that offset blindly.
- Reference spatial trial matrices use 10 cm position bins over 0–450 cm, but the requested output is temporally aligned to trial start. Therefore continuous imaging-frame samples, rather than spatially binned matrices, are the applicable source.
- Fig. 3 reports a circular-position similarity score rather than categorical accuracy; it is not directly comparable to the six requested classification heads.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/dandiset.yaml` describes DANDI:001361, version `0.251124.0550`.
- There are 152 NWB 2.8.0 files organized as `data/sub-m<id>/sub-m<id>_ses-<day>_behavior+ophys.nwb`: one file per subject/day session. Subjects m3, m4, m7, m12, m13, m14, m15, m17, m18, and m19 have sessions 1–14; m11 has sessions 3–14.
- Each NWB stores synchronized behavior under `processing/behavior/BehavioralTimeSeries`. Dense float64 series are `autoreward`, `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, and `trial_start`, each with explicit timestamps. `Reward` is a sparse float64 event series with delivery timestamps and 0.004 mL values.
- Dense behavior timestamps have an invariant 0.064483627204 s interval (15.5078125 Hz) in every file. This is the authoritative sampling interval. For the two-plane m17/m18 files, the NWB `imaging_rate`/RoiResponseSeries `rate` says 31.015625 Hz, but dense behavior timestamps and array durations demonstrate one sample per 0.06448 s; the attribute is the volume/scan rate before division by `n_planes` and must not be used as the time-bin rate.
- Ophys series are `processing/ophys/{Fluorescence,Neuropil,Deconvolved}/plane0/data`, stored as time × ROI float32. `Fluorescence` and `Neuropil` are raw fluorescence-like signals; `Deconvolved` is the available spike-like activity. Plane segmentation provides `iscell` (manual Suite2p label and probability) and `planeIdx`.
- In two-plane files, PlaneSegmentation includes both planes but the `plane0` response matrices include exactly the `planeIdx == 0` rows. Therefore the correct column curation mask is `iscell[planeIdx == 0, 0] > 0`. In single-plane files all segmentation rows correspond to plane0.
- Neural and dense behavior lengths match in 142/152 files. In 10 two-plane files neural data have one extra trailing sample; conversion must truncate all streams to their common minimum before trial extraction. Sparse Reward intentionally has a different length.
- Trial starts and teleports are paired exactly in all sessions. Slicing the dense NWB arrays as `[start_flag_index:teleport_flag_index]` retains the on-track samples (approximately 0 through 450 cm) and excludes the teleport sample/ITI; this is equivalent to the reference object's stored 1-based indices sliced as `[start-1:stop-1]`.
- Behavioral native ranges over full recordings: position −500 to 452.47 cm (−500 is pre-sync invalid and −50 is teleport); speed −6.45 to 144.52 cm/s; lick cumulative count 0–8/frame; environment −1/0/1 (−1 invalid/pre-sync); trial number −1–99. Within paired trial spans, environment is always binary 0/1.
- All NWB imaging locations are hippocampus, CA1.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 118,493 manually curated plane0 cells (260,091 plane0 ROI candidates before `iscell`) |
| Neurons / session | 155–1,780 curated; mean 779.56 (315–3,934 candidates) |
| Subjects | 11 |
| Sessions / subject | 12 for m11; 14 for each of the other 10 subjects |
| Sessions (total) | 152 |
| Trials (total) | 12,216 paired start-to-teleport trials before lick-sensor curation |
| Trials / session | 41–100; mean 80.37 |
| Samples (dense behavior/neural duration basis) | 3,610,877 frames before trial slicing |
| Trial duration | 96–3,359 frames; median 190 (12.25 s), mean 214.51 (13.83 s) |
| Rewarded trials | 10,342/12,216 (84.66%); 10,345 sparse reward events, of which 3 lie outside paired on-track spans |

Data-quality finding: applying the reference lick-sensor criterion (trial is invalid when more than 35% of its samples have cumulative lick count >2) flags 69 trials across 17 sessions, leaving 12,147 trials. This is documented here as a source-data property; the final curation decision is deferred to Steps 4–5.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not stated as a dataset-wide sum | Paper reports session-level counts rather than summing cells repeatedly imaged across days. |
| Neurons / session | 155–2,172 manually curated putative pyramidal neurons; switch-day mean 954 ± 453 | “155–2172 putative pyramidal neurons per session”; Results: “954 ± 453 cells imaged” across 11 mice × 7 switch days. |
| Subjects | 11 switch-task mice (plus 3 fixed-condition mice not present in the provided NWB set) | “counterbalanced across mice (n = 11 mice)” |
| Sessions / subject | 14 task days; m11 imaging begins day 3 | “total of 14 days”; “imaging started on day 3” for m11. |
| Trials (total) | 12,376 cited for 11 switch mice, including the two m11 behavior days not in provided imaging NWBs | Lick-curation denominator: “n = 81 out of 12,376 trials.” |
| Trials / session | Target 80–100; 80.5 ± 7.4 over all 14 mice/days | “mean ± s.d., 80.5 ± 7.4 trials” |
| Neural data time bin | ~0.0645 s (~15.5 Hz per plane) | Two-plane acquisition was 31 Hz interleaved, “sampling rate of ~15.5 Hz per plane.” |
| Behavior data time bin | Resampled to imaging frames, ~0.0645 s | Lick methods explicitly refer to “0.0645 s imaging frame samples”; GLM says all streams were ~15.5 Hz. |
| Reward rate | ~85% rewarded / ~15% omitted | “Reward was randomly omitted on approximately 15% of trials.” |
| Track and zones | 450 cm; A 80–130, B 200–250, C 320–370 cm | Hidden 50 cm reward-zone description. |
| Switch schedule | Days 3, 5, 7, 8, 10, 12, 14; switch after trial 30 | Paper task schedule; “Each switch occurred after 30 trials.” |
| Lick sensor exclusions | 81/12,376 (~0.65%); >30% frames with count >2 | Paper Quantification of licking behavior. |
| Place cells on switch days | 459 ± 263 / 954 ± 453 cells (48.5 ± 14.5%) | Paper Results. |


### Processing Details
- Unity VR frames were synchronized to imaging with TTLs; all neural/behavior streams used for modeling were sampled at the ~15.5 Hz imaging-frame rate.
- Trials are laps from entry at 0 cm through the 450 cm track; the variable-length gray teleport period occurs after the on-track trial and is normally excluded.
- dF/F: subtract 0.7× neuropil (per reference code), calculate a maximin baseline separately within each trial using a 20 s/300-sample window, `(F-baseline)/abs(baseline)`, then Gaussian-smooth with 2-sample (~0.129 s) s.d. Deconvolve with OASIS/canonical GCaMP kernel to obtain event activity.
- The paper RR decoder uses deconvolved calcium events, only speed >2 cm/s, and position-occupancy matching at ~10 cm resolution. The requested decoder instead explicitly predicts the <2 cm/s speed category and temporally varying lick output, so removing slow samples would erase required targets and break regular time bins.
- Reward-relative position in the paper is referenced to reward-zone start and circularly wrapped. The requested output is linear signed distance to any point in the whole reward zone, requiring a different (specified) transform.

### Curation Steps

**Neuron curation rules**:
1. Manual Suite2P curation removed multi-soma/dendritic ROIs, ROIs without clear transients, overexpression, and continuously fluctuating putative interneurons (`iscell`).
2. The paper then excluded additional putative interneurons whose dF/F correlated >0.5 with running speed (0.42 ± 0.85% of cells across mice/days).
3. Place-cell and remapping-class restrictions are analysis-specific scientific selections and are not a generic quality filter. Using them for this six-output decoder would discard legitimate neural information and is not planned unless required after consistency review.

**Trial curation rules**:
1. Use complete paired on-track start-to-teleport intervals; exclude pre-sync samples and teleport/ITI.
2. For lick-dependent analysis, remove trials with erroneous sensor activity: >30% of frames have cumulative lick count >2. (The repository `glmUtils.py` currently uses 35%; the published final Methods use 30%, and the provided data reproduce the published count of 81 only at 30%.)
3. Binarize valid lick counts (`count > 0`).

### Decoders Trained
| Decoded variable | Accuracy |
| Circular reward-relative position from RR, TR, or non-RR-remapping events | Paper reports cosine decode score (1 perfect, 0 shuffled expectation), not categorical accuracy. Before→after generalization exceeded shuffle only for RR neurons; effect size 2.7, P=7.54×10⁻14 across 77 switch sessions. |
| Requested six categorical outputs | No directly comparable paper decoder. Position-related outputs should be strongly decodable; outcome-related outputs may be weaker. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Lick sensor threshold | `glmUtils.get_timeseries_data`: >35% frames with count >2 | 69 trials at 35%; 81 trials at 30% | >30%; 81/12,376 | Use published >30%. The 81 flags are reproduced exactly in the available 12,216 NWB trials. |
| Total trials | Code operates on available session start/teleport arrays | 12,216 across 152 files | 12,376 across 11 mice | The exact 160-trial difference is the two ~80-trial m11 days 1–2; paper states m11 imaging began day 3, and those sessions are absent from NWB. Use every available paired imaging trial. |
| Sampling rate | Reference uses frame rate / number of planes; target ~15.5 Hz | Behavior timestamps are uniformly 0.064483627 s. m17/m18 NWB rate attribute is 31.015625 despite one row per 0.06448 s. | ~31 Hz interleaved acquisition, ~15.5 Hz per plane | Use explicit behavior timestamps: 15.5078125 Hz and 64.4836 ms bins for every session. No temporal resampling is needed. |
| Multi-plane cells | Paper pools both planes for most analyses | PlaneSegmentation contains two planes for m17/m18, but all response matrices are named/stored as `plane0` and have columns only for `planeIdx==0`; plane1 response data are absent | 155–2,172 cells/session, planes pooled | It is impossible to recover plane1 activity. Use curated plane0 cells only and disclose this source limitation. Switch-day curated plane0 mean is 821±371 versus 954±453 in paper; using segmentation labels from both planes gives 970±482, confirming why the paper count is higher. |
| ROI curation | Suite2P manual `iscell`; later dF/F–speed correlation >0.5 exclusion | 260,091 plane0 candidates; 118,493 `iscell` plane0 cells | Manual curation gives putative pyramidal cells; speed-correlation removal is 0.42±0.85% | Apply the available `iscell[planeIdx==0]` mask. Recompute dF/F and apply the published >0.5 speed-correlation exclusion before saving neural activity. |
| Neural signal | Paper pipeline recomputes per-trial dF/F then OASIS `events`; Fig. 3 uses these events | NWB `Deconvolved` has large Suite2P-scale values and correlates only median r≈0.50 with a reference-style recomputation in 20 spot-checked cells | Decoder uses deconvolved dF/F events | Do not use NWB `Deconvolved` directly. Recompute dF/F from NWB Fluorescence/Neuropil with the paper algorithm, then OASIS-deconvolve. A spot check gave recomputed event range 0–0.81 versus NWB series 0–4,466, confirming they are different processing stages. |
| Trial slicing/indexing | Stored `sess` indices are sliced `[start-1:teleport-1]` | Event flags at row `s/e`; row `s` is first on-track position and row `e` is teleport/ITI | Trials are on-track laps, teleport excluded | Slice NWB rows `[s:e]`; it is the same physical interval after accounting for reference 1-based indices. |
| Reward-zone identity | Scene mapping A/B/C and switch at trial 30 | Scene names encode one/two zones; dense zone-entry position is available | Zones A/B/C at 80/200/320 cm starts; every switch after 30 trials | Parse labels from scene and assign the second label from trial index 30 onward. Validated against 10,394 raw reward-zone entries: 0 nearest-zone mismatches. |
| Reward outcome | Reward within start-to-teleport and a zone entry | Sparse Reward series: 10,345 events; 10,342 map one-to-one to paired on-track trials; 3 fall outside | ~15% omissions | A trial is rewarded iff it has a sparse reward timestamp within `[start, teleport)` and a zone-entry flag, matching `get_trial_types`; ignore the 3 out-of-trial events. Result is 84.66% rewarded. |
| Speed threshold | Fig. 3/place analyses keep >2 cm/s | Data contain stopping and small negative smoothed speeds | Neural spatial analyses exclude <2 cm/s | Do not remove slow frames: the requested decoder explicitly requires classifying speed `<2 cm/s`, and all outputs must remain on a regular trial-start time axis. This is a task-required divergence. |
| Place-cell filtering | Paper decoder compares predefined RR/TR/non-RR place-cell classes | NWB has no saved place-cell masks | Place-cell classification is based on stochastic SI shuffles | Do not introduce a new stochastic place-cell filter. The downstream task asks for neural population activity, not cell-class comparisons; all quality-curated, non-interneuron cells retain potentially useful signal. |

Final consistent understanding: rows of every dense NWB behavior series and plane0 neural series are already synchronized at 64.4836 ms. Valid trials are paired `trial_start`→`teleport` rows. Neural processing must start from raw Fluorescence/Neuropil and reproduce trial-wise dF/F/OASIS rather than accepting the NWB Suite2P `Deconvolved` stream. Environment comes from the synchronized raw stream; zone identity comes from the validated scene/switch schedule; reward comes from sparse event timestamps. Published final-method thresholds take precedence where the repository snapshot differs.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `Fluorescence/plane0/data`, `Neuropil/plane0/data`, `iscell`, `planeIdx` | `neural` | Select `planeIdx==0 & iscell`; per paired trial subtract 0.7×neuropil, restore 0.7×trial mean neuropil, 15-sample Gaussian → 300-sample minimum then maximum baseline, `(F-F0)/abs(F0)`, 2-sample Gaussian, OASIS at 15.5078125 Hz; exclude dF/F–speed r>0.5 cells; transpose to neuron×time float32 | `preprocessing.dff`, `utilities.multi_anim_sess`, `spatial.is_putative_interneuron` | Nonfinite events become 0 exactly as Fig. 3 replaces neural NaNs. No per-cell normalization. |
| dense behavior timestamps | `input[0]` time from trial start | `timestamp[s:e] - timestamp[s]`, float32 seconds | NWB alignment / `glmUtils.get_timeseries_data` | Time-varying, begins at exactly 0. |
| dense `environment` | `input[1]` environment type | Confirm a single 0/1 in each trial, then repeat it across trial frames | `behavior.get_trial_types` | 0=ENV1, 1=ENV2. |
| dense `trial number` | `input[2]` trial number | Read at start row and repeat original zero-based value across frames | `glmUtils.get_timeseries_data` | Continuous per-trial input; do not renumber after exclusions. |
| prior raw trial's sparse Reward outcome | `input[3]` previous trial outcome | Previous original trial outcome, repeated over frames; first raw trial gets 0 | `behavior.get_trial_types` | 0=omitted/no predecessor, 1=rewarded. A dropped bad-lick trial still counts as the actual previous trial. |
| position + per-trial zone bounds | `output[0]` distance to reward zone | Signed distance to nearest point in zone: `position - clip(position,start,end)`; categorical bins below | `behavior.get_reward_zones`; task specification | Negative before zone, exactly 0 anywhere inside, positive after. |
| dense `position` | `output[1]` absolute position | Explicit classes `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360` | Spatial track conventions; task specification | Positions are on-track because teleport is excluded. |
| dense `speed` | `output[2]` speed | Explicit classes `<2`, `[2,10)`, `[10,20)`, `[20,40]`, `>40` | Synchronized `sess.vr_data['speed']`; task specification | Small negative smoothed values remain class 0; no clipping or extra smoothing. |
| dense cumulative `lick` | `output[3]` lick | `(lick > 0).astype(int)` | `glmUtils.get_timeseries_data`; paper lick methods | Trials failing >30% count>2 criterion are excluded before binarization. |
| scene ID and raw trial index | `output[4]` reward zone location | Parse chronological A/B/C labels; second label begins at raw trial index 30; repeat class A=0/B=1/C=2 | `behavior.get_reward_zones` | Validated against raw zone-entry positions. |
| sparse Reward timestamps + dense zone entry | `output[5]` reward outcome | 1 iff ≥1 Reward timestamp is within `[start_time,teleport_time)` and trial has a zone-entry event; else 0; repeat over frames | `behavior.get_trial_types` | 0=no/omitted, 1=rewarded. |
| NWB `subject_id` | `subjects`, `subject_idx` | Sorted unique IDs; per-session index | — | Session order is subject then numeric session day. |
| ImagingPlane location | `brain_regions`, `brain_region_idx` | One region `CA1`; zeros for all retained neurons | NWB metadata/paper | All recordings are hippocampal CA1. |

### Key Decisions
1. **All signals are two-dimensional time series**: Inputs have shape `(4,T)` and outputs `(6,T)`. Per-trial variables are repeated in time so static and dynamic variables coexist consistently and the validator/trainer can concatenate them.
2. **Trial alignment/window**: Alignment is trial-start entry to the linear track. Include rows `[trial_start, teleport)`, so `off_start=0` and `off_end=None` because lap duration varies. Teleport/ITI is excluded, matching core reference analyses and the requested 450 cm corridor.
3. **Time bins**: Keep native 64.483627 ms bins for every session. The apparent 31 Hz two-plane NWB attribute is not used because timestamps and the paper both establish 15.5 Hz per plane.
4. **Neural representation**: Recompute the author's dF/F-derived OASIS events rather than use NWB Suite2P `Deconvolved`. This is the paper's actual decoder input and empirically differs from the NWB stream.
5. **Neuron curation**: Apply manual `iscell` and the published dF/F–speed r>0.5 interneuron exclusion. Do not select place cells or remapping classes because this general decoder requires the neural population and those stochastic masks are not present in NWB.
6. **Unavailable plane 1**: m17/m18 plane1 segmentation exists but response data do not. Only plane0 can be converted; metadata will disclose this.
7. **Bad lick trials**: Drop all 81 trials identified by the published >30% criterion. The target requires a valid lick label at every frame and forbids NaN; dropping entire corrupt trials is safer than fabricating lick outputs. Every session still has ≥2 trials.
8. **Distance definition**: “Distance to any location in the reward zone” is the signed point-to-interval distance, zero throughout the 50 cm zone. This directly creates the specified zero class instead of measuring only from zone start.
9. **Discretization boundaries**: Use explicit inequalities for every output. Position is `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360`; speed is `<2`, `[2,10)`, `[10,20)`, `[20,40]`, `>40`. Signed distance is class 0 `<−50`; 1 `[-50,−10)`; 2 `[-10,0)`; 3 `==0`; 4 `(0,10]`; 5 `(10,50]`; 6 `>50`. This obeys the task's strict final-bin `>` wording and its positive-distance “to +10/+50” wording.
10. **Reward/previous outcome**: Outcomes are calculated for every raw trial before bad-lick removal so previous outcome preserves experimental chronology. The three sparse rewards outside an on-track interval are ignored, matching start-to-teleport reference logic.
11. **Dtypes**: neural/input float32; output integer int8 (accepted categorical integer). Subject/region indices integer arrays.
12. **Metadata/session provenance**: Store source filename, subject, day, scene, raw/kept/dropped trial indices, source/retained neuron counts, timing, signal-processing parameters, and the plane limitation.

### Planned Sanity Checks
- [ ] Every dense stream is truncated to the common neural/behavior length; every start has a later teleport; all retained trial matrices share exactly the same T across neural/input/output.
- [ ] Reproduce 152 sessions, 11 subjects, 12,216 raw trials, 81 bad-lick exclusions, and 10,342 rewarded raw trials; confirm ≥2 kept trials/session.
- [ ] Confirm all retained environment values are 0/1 and constant per trial; parsed zone labels agree with every available raw zone-entry position (expected zero mismatches).
- [ ] Verify zone switch occurs between raw trials 29 and 30 and that trial number is not silently renumbered.
- [ ] Confirm all inputs finite, all neural events finite/nonnegative, and output classes are complete subsets of their declared ranges.
- [ ] Confirm time begins at 0 and `np.diff(time)` is allclose to 0.064483627 s.
- [ ] Independently recompute a raw F/Fneu→dF/F→OASIS segment for specific saved cells/trials and require `np.allclose` to converted neural activity.
- [ ] Independently load original timestamp/environment/trial/outcome fields for specific trials and require `np.allclose` to converted inputs.
- [ ] Independently bin original position/speed/lick and construct zone/outcome labels for specific trials and require `np.allclose` to converted outputs.
- [ ] Check reward rate, trial/frame distributions, neuron counts, and switch-day neuron mean against data/paper, explaining unavoidable plane1 and missing-m11-day discrepancies.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required positional output, default/explicit `--full`, `--sample`, and `--show-processing` modes. Syntax compilation and CLI help completed without errors. The script:

- discovers and naturally sorts all subject/session NWBs;
- validates timestamp identity, fixed sampling, trial pairing, environment constancy, and plane/response correspondence;
- reproduces per-trial neuropil correction, maximin baseline, dF/F smoothing, OASIS events, manual-cell selection, and dF/F–speed interneuron filtering;
- applies the published lick-trial curation and constructs all four inputs and six categorical outputs;
- retains detailed session provenance and processing parameters;
- runs internal shape/range/time-axis checks before pickling; and
- creates ten-panel diagnostic plots covering raw fluorescence, neuropil subtraction, baseline, dF/F, OASIS, alignment, distance/speed/lick discretization, inputs, and neuron curation for up to two sessions.

Code inefficiencies identified:
NWB datasets are contiguous rather than chunked. Repeated fancy-column reads can trigger expensive HDF5 selection overhead, while loading full multi-thousand-ROI sessions creates high peak memory. OASIS must operate trial-by-trial to match the reference.

Code speedups added:
Read each short contiguous trial row block once and subset curated columns in memory; avoid loading unused NWB `Deconvolved` and image datasets; process a single session at a time; accumulate interneuron correlations from sufficient statistics; skip OASIS on trials already rejected for lick corruption; keep arrays float32; and avoid temporal resampling because streams are already aligned.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 1,895 session-neurons after curation |
| Neurons / session | 1,013; 882 (mean 947.5) |
| Subjects | 1 (`m3`) |
| Sessions / subject | 2 |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| Trial T | min 162, session-mean 315.12, max 1,546 frames |
| Time input range | [0, 99.6] s |
| Environment range | [0, 0] (both sample sessions are ENV1 stay days) |
| Trial number range | [0, 79] |
| Previous outcome range | [0, 1] |
| Distance-class distribution | [0.222, 0.055, 0.169, 0.394, 0.037, 0.066, 0.057] |
| Position-class distribution | [0.075, 0.066, 0.081, 0.601, 0.177] |
| Speed-class distribution | [0.433, 0.089, 0.081, 0.143, 0.254] |
| Lick distribution | [0.767, 0.233] |
| Zone distribution | [0, 0, 1.000] (both sample sessions use C) |
| Outcome distribution | [0.113, 0.887] |

### Processing Plots Review
`processing_m3_ses-01.png` and `processing_m3_ses-02.png` show the full pipeline. Fluorescence transients remain after neuropil subtraction; the maximin baseline tracks the lower envelope; smoothed dF/F and OASIS events are aligned; position begins near 0 and ends near 450; the zone-C zero-distance plateau matches 320–370 cm; speed and lick category steps follow the raw traces; time begins at zero. No temporal shift or discretization anomaly is visible.

Iteration: the first plot exposed numerically unstable raw-sum Pearson accumulation and a shared-axis plotting artifact. Pearson accumulation was replaced with batch-Welford centered covariance (unit-tested against `np.corrcoef`, exact agreement); correlations are now bounded [−1,1]. Plot axes were separated so the speed-correlation histogram is legible. The entire sample conversion and verification were rerun after both fixes.

The apparent 3.7–3.8% removal by speed correlation in these two all-cell sessions is higher than the paper's 0.42% statistic because that reported exclusion occurs inside its place-cell-selected analysis population. The implemented threshold and dF/F signal exactly match the published rule; the general decoder otherwise retains all manually curated non-interneuron cells.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Contiguous trial-row reads, float32 arrays, online centered correlation, skip OASIS for rejected trials | Sample completes in 6.27 s plus 0.21 s pickle write; no redundant full-session signal copies |

| Step | Time / Session | Estimated Total Time |
| Neural + behavior conversion | 2.40 s mean in sample (2.77, 2.02 s) | Work-scaled estimate ~258 s for 152 sessions |
| Full pickle write | 0.105 s/session equivalent | ~8–15 s for estimated 7.3–7.7 GiB neural payload |
| Total | 3.14 s/session including plots/sample overhead | ~4.5–6 min, safely below 15 min |

Validation: `/app/sample_data.pkl` is 185 MiB. `/app/verification_sample_out.txt` reports “Data format is valid, no errors or warnings.” Manual loading confirmed all arrays finite, neural nonnegative, dtypes float32/float32/int8, exact `(neuron,4,6)×T` agreement, and preserved raw trial numbers 0–79. Available disk (3.4 TiB) and RAM (~1 TiB) comfortably exceed the projected full payload.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: Format verification has none. Training emits sklearn's “y_pred contains classes not in y_true” for reward-zone location because the deliberately small sample's two stay sessions contain only zone C while the declared task has A/B/C. This cannot be fixed without violating `--sample`'s deterministic two-session selection; it is not present as a data-format warning and full data contain all zones.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Distance to reward zone | 0.7903 | 0.6829 |
| Absolute corridor position | 0.8750 | 0.8250 |
| Speed | 0.6568 | 0.6389 |
| Lick | 0.8039 | 0.7736 |
| Reward zone location | 0.9974 | 0.9975 |
| Reward outcome | 0.7623 | 0.7106 |

Training completed on CUDA in 200 epochs. Loss decreased monotonically from 3.7733 (epoch 1) to 0.6137 (epoch 200); test loss was 0.5660. Every validation balanced accuracy exceeds uniform chance (0.1429, 0.2, 0.2, 0.5, 0.3333, 0.5 respectively). This strongly supports correct synchronization and category construction. The final sample artifacts were regenerated after the exact-boundary metadata correction; numeric class assignments were unchanged because no raw frame lies exactly on a position/speed edge.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 8,081,552,766 bytes (7.527 GiB)
- `conversion_full_out.txt`: created; conversion took 221.67 s plus 7.99 s to pickle
- `verification_full_out.txt`: created; validator reported "Data format is valid, no errors or warnings."

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 155--2,172 cells/session; 954 +/- 453 pooled around switches | manual `iscell`, then speed-correlation exclusion | 118,493 manual plane-0 cells | 118,169 session-neurons; 777.43/session (154--1,780); switch days 778.6 +/- 375.2 | Yes after documented plane-0/source-data and speed-filter restrictions |
| Mean neurons/session | above | session-specific | 779.56 manual plane-0 | 777.43 | Yes |
| Subjects | 11 imaged mice | 11 mouse IDs | 11 | 11 | Yes |
| Sessions | 14 days/mouse, with m11 imaging absent days 1--2 | available behavior+ophys NWBs | 152 | 152 | Yes |
| Trials (total) | 12,376 behavioral trials, including 160 m11 trials without imaging | per-session paired trials | 12,216 paired imaging trials | 12,135 after 81 published lick-quality exclusions | Yes |
| Trials/session (mean) | about 80 | native session count | 80.37 paired | 79.84 retained | Yes |
| Input: time (s) | trial start through pre-teleport frame | `[start-1:stop-1]` equivalent | [0, 216.5] | [0, 216.5] | Yes |
| Input: environment | ENV1/ENV2 | scene/day mapping | [0, 1] | [0, 1] | Yes |
| Input: trial number | within-session number | trial table order | [0, 99] | [0, 99] | Yes |
| Input: previous outcome | preceding raw trial reward | sparse reward event | [0, 1] | [0, 1] | Yes |
| Distance bins | task-specified | n/a | derivable from position/zone | [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243] | Yes |
| Position bins | task-specified | n/a | derivable from position | [0.212, 0.177, 0.231, 0.226, 0.154] | Yes |
| Speed bins | task-specified | speed stream | derivable from speed | [0.117, 0.087, 0.134, 0.319, 0.343] | Yes |
| Lick | binary `count > 0` | lick stream | derivable from lick count | [0.777, 0.223] | Yes |
| Reward zone | A/B/C | scene parser and switch at trial 30 | raw fractions approximately balanced | [0.332, 0.336, 0.333] | Yes |
| Reward outcome | sparse event in trial | sparse reward event | 10,342/12,216 paired raw trials (84.66%) | timepoint-weighted [0.158, 0.842] | Yes |

Manual spot checks loaded the pickle independently of the validator for sessions 0, 42, 82, 124, and 151 (covering the first/last files, m11's reduced session set, a switch session, and m18's two-plane segmentation files). Every neural/input/output trial had identical time length, all checked neural values were finite, raw-trial indices spanned the expected range, and zone labels agreed with the scene/day structure. Aggregate metadata exactly reconciled 12,216 raw paired trials to 12,135 retained trials (`12,216 - 81`) and 118,493 manually curated plane-0 cells to 118,169 retained neurons (`118,493 - 324`). No data disappeared beyond these declared filters.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: read all of `verification_full_out.txt`. It begins "Data format is valid, no errors or warnings" and contains no later error/warning text. All 152 sessions have at least 40 retained trials, all dimensions/ranges are valid, and all six output classes cover their declared range.
2. **Independent raw-data `np.allclose` tests**: `cache/independent_raw_checks.py` does not import the converter. It reads raw `m11_ses-03`, independently recomputes trial-wise maximin dF/F, the speed-correlation mask, OASIS events, reward outcomes, inputs, and outputs. Raw trials 0, 1, 30, 70, and 79 span both sides of the within-session B-to-A switch and include rewarded and omission trials. Whole neural matrices, all four input rows, and all six output rows passed `np.allclose` for every tested trial. Exact distance, position, and speed bin boundaries also passed synthetic `np.allclose` tests. Output is in `cache/independent_raw_checks_out.txt`.
3. **Reference-code comparison**:

| Major step | Reference implementation | Converter implementation | Result / justified difference |
|------------|--------------------------|--------------------------|-------------------------------|
| Data loading | Session objects load synchronized Suite2P fluorescence and VR streams; NWB exposes the same source arrays | `convert_session` reads `Fluorescence/plane0`, `Neuropil/plane0`, dense behavior, sparse Reward timestamps, and manual `iscell` directly with h5py | Same available source content; direct NWB loading avoids depending on unpublished session pickles |
| Neuron filtering | manual Suite2P `iscell`; `spatial.is_putative_interneuron` correlates full valid dF/F with speed and excludes `r > 0.5` (paper) | manual plane-0 `iscell`; stable batch-Welford dF/F-speed Pearson correlation; exclude `r > 0.5` | Same. No place-cell/SI filter because those are analysis-specific, not recording-quality criteria |
| Trial filtering | paper: invalidate trials if >30% of frames have cumulative lick count >2 (81/12,376); code snapshot contains older 0.35 default in places | drop the whole trial at the published >30% criterion because lick is a required target and NaNs are invalid | Published methods take precedence; exactly 81 paired NWB trials are found |
| Temporal alignment | reference slices `trial_start-1:teleport-1` because its MATLAB-derived indices are one-based | NWB flags are zero-based and sliced `[trial_start:teleport)` | Equivalent rows; excludes teleport/ITI and aligns t=0 to track entry |
| Neural processing | `preprocessing.dff`: 0.7 neuropil subtraction, restore 0.7 trial mean neuropil, Gaussian sigma 15, 300-sample minimum then maximum, divide by absolute baseline, Gaussian sigma 2, Suite2P OASIS tau 0.7 at per-plane rate | same operations and constants, performed trial-by-trial from raw F/Fneu | Same; raw recomputation is used because NWB `Deconvolved` is not the paper's final trial-wise processing |
| Binning | paper spatial analyses use 10 cm bins and often remove speed <2 cm/s | retain native 64.4836 ms imaging bins; categorical boundaries are those mandated by this decoder task | Required difference: time-varying decoder outputs, including a `<2 cm/s` class, require regular temporal samples rather than spatial averaging/removal |
| Input construction | no directly corresponding paper decoder inputs | time from trial start; raw ENV; zero-based raw trial number; preceding raw-trial reward | Task-mandated. Per-trial values are repeated across time as target schema requires |
| Output construction | raw position/speed/lick/zone/reward streams; zone A/B/C bounds in methods; switch after trial 29 | exact task inequalities; lick is `count > 0`; scene-derived A/B/C; reward event within trial | Same source variables; categorical transforms are task-mandated |

4. **Key-statistics comparison**: independently scanning all NWBs (not via converter) reproduced 152 files, 12,216 paired imaging trials, 81 bad-lick trials, 118,493 manually curated plane-0 cells, 10,345 sparse reward events, and 10 files with a neural/behavior trailing-length mismatch. The conversion has 12,135 trials and 118,169 neurons after its two declared filters. The 11 subjects, m11's missing days 1--2, 15.5078125 Hz sampling, three 50 cm zones, switch at raw trial 30, ~15% omissions (raw paired reward rate 84.66%), and 80--100 typical trials all match the paper. The paper's 12,376 trials include the 160 m11 days 1--2 behavioral trials without imaging; those cannot be neural-decoder trials. The paper pools both planes for m17/m18, but these NWBs provide RoiResponseSeries arrays only for plane 0, a source limitation stated in metadata. The converted speed-correlation exclusion is 324/118,493 = 0.273%, compatible with the paper's 0.42 +/- 0.85% across mice/days.
5. **Edge-case audit**: verified paired and ordered start/teleport flags; ten NWBs with extra trailing rows are safely truncated to the common synchronized length; three sparse reward events outside paired trial windows are ignored; the first trial's previous outcome is zero; after a bad-lick trial, "previous" still means the preceding raw trial rather than preceding retained trial; zone changes occur at raw trial 30 even if a prior trial is dropped; all session IDs and raw trial numbers are unique; all retained trials have finite equal-length neural/input/output arrays; all sessions exceed the two-trial requirement; and exact class boundaries were explicitly tested. m17/m18's NWB `rate=31.015625` describes interleaved acquisition while behavior timestamps and the paper confirm 15.5078125 Hz per plane, which is used.

### Issues Found and Resolved
- **Iteration 1 — unstable raw-sum Pearson calculation during development**: a processing plot exposed impossible correlation magnitudes from cancellation in raw sums. Replaced it with a centered batch-Welford merge; all correlations are now bounded and the independent `np.corrcoef` mask matches exactly.
- **Iteration 2 — categorical upper-edge semantics**: review found ordinary `np.digitize` would classify exactly 360 cm and 40 cm into the strict `>360` and `>40` classes. Replaced it with explicit inequalities, corrected labels, and added boundary tests. Independent scanning found zero raw samples exactly at any position/speed edge, so distributions did not change. Nevertheless, the full conversion (241.82 s + 8.85 s write), full validator, and every independent check were rerun. All counts/distributions remained identical and all checks passed.
- **Reference discrepancies resolved without code changes**: the repository snapshot's older 35% lick threshold is superseded by the published 30%/81-trial rule; speed <2 cm/s samples are retained because speed class 0 is explicitly required; missing m17/m18 plane-1 responses and m11 days 1--2 imaging cannot be reconstructed and are transparently excluded. These are source/task constraints, not unexplained loss.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Training loss decreased smoothly from 2.695586 (epoch 1) to 0.831655 (epoch 200); held-out test loss was 0.805185. Training completed on CUDA without fallback or errors.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Distance to reward zone | 0.6073 | 0.5206 | chance 0.1429 |
| Absolute corridor position | 0.6724 | 0.6187 | chance 0.2000 |
| Speed | 0.6291 | 0.5721 | chance 0.2000 |
| Lick | 0.7838 | 0.7591 | chance 0.5000 |
| Reward zone location | 0.8833 | 0.8423 | chance 0.3333 |
| Reward outcome | 0.8101 | 0.5799 | chance 0.5000 |

The command `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` finished successfully and its complete output is in `train_decoder_full_out.txt`. Sample/prediction plots were also produced by the supplied trainer.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Train / validation | Expectation from paper |
|----------|------------------------------|--------|--------------------|--------------------|------------------------|
| Distance to reward zone | 0.5206 | 0.1429 | 3.64x | 1.167 | Strong reward-relative/position coding expected; direct metric not reported |
| Absolute corridor position | 0.6187 | 0.2000 | 3.09x | 1.087 | CA1 spatial coding predicts strong performance; direct metric not reported |
| Speed | 0.5721 | 0.2000 | 2.86x | 1.100 | Speed-correlated activity is documented, though the strongest `r>0.5` cells are appropriately excluded; no speed decoder reported |
| Lick | 0.7591 | 0.5000 | 1.52x | 1.033 | No lick decoder reported |
| Reward zone location | 0.8423 | 0.3333 | 2.53x | 1.049 | Flexible reward-relative maps predict strong zone information; no zone-class accuracy reported |
| Reward outcome | 0.5799 | 0.5000 | 1.16x | 1.397 | Random ~15% omissions make current outcome intrinsically weak before reward; no outcome decoder reported |

All validation accuracies exceed chance. Five outputs meet or exceed the requested 1.5x-chance review threshold. No train/validation ratio exceeds the 1.5 overfitting threshold; the maximum is 1.397 for reward outcome.

### Comparison to Every Paper Decoding Result

The paper reports one decoder family: circular-linear regression of continuous circular reward-relative position from selected RR, TR, or non-RR-remapping cell populations. It reports cosine decode score (1 perfect, 0 random), not categorical balanced accuracy, and therefore none of its values is numerically commensurate with the six supplied-decoder heads.

| Paper result | Paper value | Closest requested head | This decoder | Interpretation |
|--------------|-------------|------------------------|--------------|----------------|
| Train/test before switch, real versus shuffled | Effect sizes: RR 2.6, TR 3.3, non-RR 3.2 | Distance / position | 0.5206 / 0.6187 balanced accuracy | Qualitatively consistent strong spatial decoding; metrics and neuron selection differ |
| Train before/test after switch | RR effect size 2.7; TR -0.5; non-RR -0.2; RR versus shuffle P=7.54e-14 across 77 switch sessions | Distance to zone | 0.5206 (3.64x chance) | Qualitatively consistent preserved reward-relative information |
| Spatial extent of above-shuffle RR decode | z-score >2 from -104.5 +/- 20.1 cm to +152.7 +/- 22.9 cm relative to zone start | Distance to zone | Seven classes over the entire corridor | This is a spatial extent, not an accuracy value |

The figures display session mean cosine scores but the article text/caption provides no numeric mean decode scores or categorical accuracies. Speed, lick, zone identity, and reward outcome were not decoded from neural activity in the paper. Thus there is no omitted like-for-like paper accuracy and no evidence of an accuracy shortfall.

### Low-Accuracy Investigation: Reward Outcome

1. **Raw values**: independently loaded raw `m11_ses-03` trials 1 (omission), 30 (rewarded), and 70 (omission). In each, the entire converted output row exactly matches independently detected sparse reward events by `np.allclose`; extra rewarded trials 0 and 79 also pass.
2. **Temporal alignment**: `cache/outcome_alignment_check.png` overlays normalized converted neural events, the outcome target, and independently located raw sparse reward events. The omission has no reward marker and target 0; the rewarded trial has a raw event at 2.4 s and target 1. Both start exactly at raw `trial_start`; visual review found no offset.
3. **Variation**: converted timepoint distribution is 15.8% omission / 84.2% rewarded, and the independently computed raw paired-trial reward rate is 84.66%. This matches the paper's randomized approximately 15% omission design and is far from 99% one-class.
4. **Filtering/processing**: the same manually curated, OASIS-deconvolved, speed-correlation-filtered activity passed the Step 10 raw checks. Rewarded and omitted trials undergo identical processing; only invalid lick-sensor trials are removed.
5. **Interpretation**: current outcome was randomized by the task and is specified as a per-trial label. Repeating it over the full trial necessarily labels pre-delivery neural samples with a future random event; these samples cannot contain causal information about the omission. Recasting the target as post-delivery-only would violate the required per-trial output. The modest above-chance held-out score is therefore expected, not evidence of misalignment. The paper likewise analyzes outcome effects mainly after reward and reports no outcome-decoder accuracy.

No conversion change is warranted from decoder performance. Predictions plots were produced, loss decreased continuously, all outputs beat chance, raw labels and alignment are exact, and no leakage/overfitting threshold is crossed.

### Issues Found and Resolved
- Reward outcome was below 1.5x chance: completed all prescribed raw-label, alignment, variation, filtering, and reference-processing checks. No conversion error was found; weakness follows from randomized omissions and the mandated per-trial label across pre-reward samples.
- No other output triggered a Step 12 issue. Consequently no iteration of Steps 9--11 was required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading example, row order, processing/caveats, statistics, commands, and full decoder results.
- [x] `cache/` created; `README_CACHE.md` inventories the independent audit script/output, outcome-alignment plot, and generated bytecode.
- [x] All files organized. Investigation/audit code is under `cache/`; required conversion, logs, plots, and deliverable pickles remain at project root.

Final completeness audit found every required file. Final sizes include `converted_data.pkl` 8,081,552,758 bytes and `sample_data.pkl` 193,492,882 bytes. Both final sample and full validator logs begin "Data format is valid, no errors or warnings." Every workflow step is marked COMPLETE, the full 200-epoch decoder run ends successfully, and the independent raw check ends `ALL INDEPENDENT RAW CHECKS PASSED`.
