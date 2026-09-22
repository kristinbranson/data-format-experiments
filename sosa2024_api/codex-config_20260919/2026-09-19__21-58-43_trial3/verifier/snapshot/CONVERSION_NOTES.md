# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward (provided NWB data)
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

Runtime verified: Python 3.13.15, NumPy 2.4.4, PyTorch 2.6.0+cu124. Mandatory checkpoint passed: `/app/CONVERSION_NOTES.md` exists.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `src/reward_relative/preprocessing.py` | LOADING | Construct a TwoP session, synchronize VR to imaging, load Suite2p-curated ROIs, and add aligned behavioral time series/trial matrices. |
| `vr_align_to_mock_2P` | `src/reward_relative/preprocessing.py` | PROCESSING | Demonstrates synchronization policy: linear interpolation for position, nearest interpolation for categorical values, and cumulative interpolation/differencing for events. Actual imaging uses TwoPUtils `vr_align_to_2P`. |
| `dff` | `src/reward_relative/preprocessing.py` | PROCESSING | Restrict fluorescence to trial intervals; subtract 0.7× neuropil; estimate per-trial maximin baseline (15-frame smoothing then 300-frame min/max filters); calculate dF/F; smooth by 2 frames; optionally OASIS-deconvolve. |
| `multi_anim_sess` | `src/reward_relative/utilities.py` | LOADING / PROCESSING | Load curated session pickles, calculate dF/F and deconvolved events, add 10-cm spatial trial matrices, trial types/zones, and place-cell statistics. |
| `get_trial_types` | `src/reward_relative/behavior.py` | PROCESSING | Derive per-trial reward outcome (reward and zone-entry when available) and environment morph from start-to-teleport samples. |
| `get_reward_zones` | `src/reward_relative/behavior.py` | PROCESSING | Assign per-trial reward zones A/B/C using scene/switch identity; canonical zones are A/X=80–130, B/Y=200–250, C/Z=320–370 cm. |
| `define_trial_subsets` | `src/reward_relative/behavior.py` | CURATION | Split switch sessions chronologically by reward-zone identity; optionally split constant-zone sessions into halves. |
| `calc_place_cells` | `src/reward_relative/spatial.py` | CURATION | Identify place cells from deconvolved events at speed >2 cm/s, with 100 population shuffles and p<0.05, separately for trial sets. |
| `get_timeseries_data` | `src/reward_relative/glmUtils.py` | PROCESSING | Select only start-to-teleport samples, align position relative to reward-zone start, preserve absolute position/events, clean lick sensor errors, and optionally retain speed≥2 cm/s. |
| `CircularRegression` / `train_vs_test_blocks` | `src/reward_relative/decode.py` | PROCESSING | Decode circular reward-relative position from timepoint×neuron deconvolved activity with repeated block cross-validation. |

### Notes
- The data are two-photon calcium imaging, not electrophysiology. The repository explicitly says raw `sess` objects lack dF/F, but the published NWB export may already contain processed fluorescence/deconvolved activity; this must be checked in Step 2 before recomputing anything.
- Manual neuron curation occurs in the Suite2p GUI (`iscell.npy`) before session creation. Published analysis then identifies place cells, but the downstream task asks for neural activity generally; place-cell-only filtering is not automatically appropriate and will be resolved against the NWB content and paper.
- Reference continuous decoding uses deconvolved `events`, zeros residual NaNs, and excludes samples below 2 cm/s. That speed threshold is an analysis-specific occupancy/control decision; because speed itself is a required decoder output, excluding stationary samples would destroy a specified class and therefore should not be copied without justification.
- Trials run from `trial_start_inds` to `teleport_inds`; reference functions consistently use the Python slice `start-1:stop-1` because stored event indices are effectively one-based. NWB interval timestamps may remove this legacy indexing issue.
- Trial matrices are spatially binned at 10 cm from 0–450 cm, but the requested target is temporally aligned to trial start, so native imaging-frame samples (about 15.5 Hz per repository docs) are the relevant representation.
- Licks are cumulative counts per imaging frame in the original synchronized stream; reference code clips values >1 to 1 for event presence. Reward outcome is whether reward occurred in the trial (and, where available, reward-zone entry also occurred).
- Paper decoder target differs from this task: it predicts circular reward-relative position and reports cosine similarity, whereas the requested outputs are six categorical targets and will be validated by a provided neural decoder.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/dandiset.yaml` describes DANDI:001361 version 0.251124.0550 (92,448,350,544 bytes), two-photon CA1 imaging plus behavior.
- 152 files are organized as `sub-<mouse>/sub-<mouse>_ses-<day>_behavior+ophys.nwb`. Subjects m3, m4, m7, m12–m15, and m17–m19 each have days 1–14; m11 has days 3–14.
- All inspection used `pynwb.NWBHDF5IO(..., load_namespaces=True)`. No `h5py` access was used.
- Each NWB file contains processing modules `behavior` and `ophys`. `behavior/BehavioralTimeSeries` has aligned series: `position` (cm), `speed` (cm/s), cumulative-per-frame `lick`, `environment`, `trial number`, binary/event-count `trial_start`, `teleport`, `reward_zone`, `autoreward`, `scanning`, plus irregular event series `Reward` (mL at delivery timestamps).
- `ophys` contains `Fluorescence` (raw ROI F), `Neuropil` (Fneu), `Deconvolved`, `ImageSegmentation`, and background images. Each response matrix is time × ROI, float32. Behavioral arrays are float64.
- `PlaneSegmentation` has `pixel_mask`, two-column Suite2p `iscell` (`[:,0]` is the curated 0/1 label; `[:,1]` is probability), and `planeIdx`. It is essential not to interpret the probability column as another cell label.
- 124 files are single-plane (`planeIdx={0}`) and advertise 15.5078125 Hz. The 28 m17/m18 files are two-plane (`planeIdx={0,1}`) and advertise 31.015625 Hz in `Deconvolved`, but their row count matches behavior sampled at 15.5078125 Hz; hence behavior timestamps provide the authoritative aligned timebase. Ten two-plane sessions have one more neural row than behavior and require truncation to the common length.
- There is no NWB `trials` table. Trial intervals are encoded by matched `trial_start` and `teleport` events (all 152 files have equal counts); identifiers encode scene/environment/reward-location schedules. Valid on-track position spans approximately 0–450 cm.
- Full schema spot checks showed all expected streams in representative single- and multi-plane files, response-series ROI regions matching segmentation rows, and neural/behavior lengths identical except the ten documented +1-neural-row cases.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 manually curated cells (`iscell[:,0] == 1`); 312,110 candidate ROIs before curation |
| Neurons / session | curated: mean 912.36, median 921.5, range 155–2,341; all ROIs: mean 2,053.36, range 315–5,085 |
| Subjects | 11: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19 |
| Sessions / subject | 14 each except m11=12; 152 total |
| Trials (total) | 12,216 start-to-teleport intervals |
| Trials / session | mean 80.37, median 80, range 41–100 |

Additional native totals: 3,610,877 neural samples across session time axes and 10,345 reward-delivery events (84.68% of trials if one delivery per rewarded trial, to be explicitly verified during conversion). Session environment sets are Env1-only in 73 files, Env2-only in 68, and both in 11 environment-switch files. Reward-zone schedules are recoverable from identifiers (A/B/C and switches) and the reference zone definitions.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 73,512 cell-sessions pooled over seven switch days | Extended Data Fig. 9: “11935/73512 … of all cells” |
| Neurons / session | 954 ± 453 imaged; 459 ± 263 place cells on switch days; manual-curation range 155–2,172 | Main text: “459 ± 263 place cells out of 954 ± 453 cells imaged”; Methods: “155–2172 putative pyramidal neurons per session” |
| Subjects | 11 switch-task mice (plus 3 fixed-condition mice not present in NWB data) | “counterbalanced across mice (n = 11 mice)” |
| Sessions / subject | 14 task days, one imaging session/day except m11 began imaging day 3; seven switch days (3,5,7,8,10,12,14) | Task and 2P imaging Methods |
| Trials (total) | 12,376 imaged-task trials cited for lick QC; NWB omission of m11 days 1–2 predicts 12,216 available trials | “n = 81 out of 12,376 trials removed across 11 switch mice”; m11 imaging started day 3 |
| Trials / session | 80.5 ± 7.4 across all 14 mice in paper; target 80–100 | “mean ± s.d., 80.5 ± 7.4 trials” |
| Neural data time bin | ~64.5 ms (15.5 Hz); two-plane acquisition interleaved at ~31 Hz gives ~15.5 Hz/plane | 2P Imaging and GLM Methods |
| Behavior data time bin | ~64.5 ms, synchronized to imaging | “All behavioral and neural time series were sampled at ~15.5 Hz” |
| Reward rate | ~85% rewarded / ~15% random omissions | “Reward was randomly omitted on approximately 15% of trials” |
| Track and zones | 450 cm; A=80–130, B=200–250, C=320–370 cm | Hidden reward zone task Methods |
| Switch timing | Reward/environment condition changes after first 30 trials; ten initial new-condition trials may autoreward | Hidden reward zone task Methods |
| Bad lick trials | 81/12,376 (~0.65%), detected when >30% of frames have cumulative lick count >2 | Quantification of licking behavior Methods |
| Place-cell analysis | 45 × 10-cm bins; exclude speed <2 cm/s; 100 within-trial circular shuffles; pooled-animal p<0.05 | Place cell identification Methods |


### Processing Details
- Trial activity is defined from entry at 0 cm through track completion/teleport onset. Teleport activity is normally excluded because laser power was often reduced there and because the requested task concerns the 0–450 cm corridor.
- Manual Suite2p ROI curation excludes non-somatic/multi-soma/dendritic/no-transient/overexpressing/high-continuous-fluctuation ROIs. Raw F is neuropil corrected, then a per-trial baseline is estimated by maximin with a 20 s window. dF/F is `(F-baseline)/abs(baseline)`, smoothed with a two-sample (~0.129 s s.d.) Gaussian, then OASIS-deconvolved with a canonical calcium kernel.
- Additional putative interneurons are excluded when Pearson correlation between dF/F and running speed exceeds 0.5 (0.42 ± 0.85% of cells across mice/days).
- Behavior and neural streams are synchronized at imaging frames. Lick count is converted to binary per frame. Reference lick-only analysis invalidates corrupt trials, but for the requested joint decoder those trials can retain other outputs while lick samples/trials need explicit handling.
- Reward-relative paper analyses center circular coordinates on reward-zone **start**. The requested signed distance “to any location in the reward zone” instead requires zero throughout the entire 50 cm zone, negative distance before its start, and positive distance after its end.
- Switch sessions are days 3, 5, 7, 8, 10, 12, and 14; reward moves after trial 30. Day 8 also changes environments. Stay days preserve the previous condition.

### Curation Steps

**Neuron curation rules**:
Manual Suite2p `iscell` curation followed by speed-correlation exclusion (`r > 0.5`); paper place-cell-only restrictions apply to spatial/remapping analyses, but not necessarily to a general neural decoder. If using all curated pyramidal neurons, record that as a decoder-task exception.

**Trial curation rules**:
Use start-to-teleport track intervals. Paper lick analysis sets erroneous lick trials to NaN when >30% of frames have cumulative lick count >2. Reward omission analyses require at least three omissions/set, but that restriction does not apply to this decoder. No generic trial exclusion beyond invalid/missing synchronized streams is described.

### Decoders Trained
| Decoded variable | Accuracy |
| Circular reward-relative position from RR/TR/non-RR cells | Cosine decode score (1 perfect, 0 random), not categorical accuracy. Within-before data exceeded shuffle for all populations; train-before/test-after exceeded shuffle only for RR cells. Reported standardized effect sizes: within-before RR=2.6, TR=3.3, non-RR=3.2; cross-switch RR=2.7, TR=−0.5, non-RR=−0.2 (77 sessions, 11 mice). |

The paper reports no directly comparable class accuracy for the six requested outputs. Figure 3 establishes that aligned deconvolved activity carries strong position information; direct numerical comparison must acknowledge the different target, cell-subpopulation selection, metric, and model.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session scope | Main analyses use seven switch days (3,5,7,8,10,12,14) | 77 switch sessions plus 75 stay/acquisition sessions = 152 | Main decoder uses 77 sessions; experiment contains 14 days | Convert all 152 provided sessions because the requested “full dataset” and task outputs are defined for every condition; identify switch/stay status in metadata. This is a task-scope extension, not a processing difference. |
| Curated cells | `sess.iscell[:,0]` is the manually curated cell mask | 138,678 curated ROI-session observations overall; 74,652 on 77 switch sessions | 73,512 switch-day cells; 954 ± 453/session; range 155–2,172 | All single-plane per-mouse switch counts exactly match Extended Data Fig. 9. The 1,140-cell difference is confined to later multi-plane NWBs (m17 +118, m18 +1,022), consistent with an updated DANDI curation/export relative to manuscript analysis. Use the current NWB `iscell[:,0]`, then apply the documented interneuron rule. Do not discard arbitrary multi-plane cells to force an old count. |
| Multi-plane timebase | ~31 Hz interleaved acquisition, ~15.5 Hz per plane and for synchronized behavior | RRS metadata says 31.015625 Hz for m17/m18, yet one row per 15.5078125-Hz behavior sample; 10 files have one extra neural row | ~15.5 Hz per plane; all neural/behavior streams sampled ~15.5 Hz | Use behavioral timestamps as authoritative and trim every stream to its common length. This preserves the actual one-row-per-aligned-frame representation. |
| Neural signal | Recompute per-trial maximin dF/F and OASIS `events` from F and Fneu | NWB `Deconvolved` has raw-scale values (example max 2,484) and only modest correlation (0.25–0.67 in five cells) to reconstructed reference events (max 0.53) | Decoder uses the custom deconvolved calcium event timeseries | Recompute reference dF/F/events from NWB `Fluorescence` and `Neuropil`; do not treat NWB `Deconvolved` as the final paper signal. This is the major resolved processing discrepancy. |
| Neuron filtering | Manual `iscell`; `is_putative_interneuron` excludes dF/F–speed Pearson r>0.5 | Required raw F/Fneu/speed and manual labels are present | 0.42 ± 0.85% excluded; place-cell filtering used for spatial/remapping analyses | Apply manual cell and r>0.5 filters. Do not restrict to place cells: the new multi-output task includes movement, licking, and outcome, for which non-place pyramidal cells remain relevant; place-cell classification is analysis-specific and computationally stochastic. |
| Trial bounds | Start-to-teleport; legacy session code often slices `start-1:stop-1` | Explicit 0-based aligned frame events; trial starts have median position 1.75 cm, final included frames median 448.94 cm | Analyze 0–450 cm track; teleport excluded | Slice NWB directly as `[start_event_index:teleport_event_index]`. Applying the legacy −1 offset to explicit NWB indices would introduce a pre-start frame. |
| Trials | Switch after trial 30; ~80–100/session | 12,216 intervals; all files have matched start/end events; all 11 environment switches occur exactly at trial index 30 | 12,376 trials cited; m11 lacked imaging on days 1–2 (160 trials) | Exact reconciliation: 12,376−160=12,216. Use all valid NWB intervals. |
| Reward outcomes | Any reward in start-to-teleport interval (and reward-zone entry when frame series available) | 10,342 trials contain a mapped delivery; three extra `Reward` events occur during unusually long inter-trial gaps in two m4 sessions | ~15% random omissions | Define outcome only from deliveries inside each trial: 84.66% rewarded / 15.34% omission, matching paper. Exclude the three ITI deliveries from outcomes. |
| Lick corruption | Reference utilities use 0.35 in `dayData`; manuscript specifies >30% frames with count>2 | Exact >30% rule identifies 81/12,216 trials; >35% identifies 69 | 81/12,376 (~0.65%) trials removed from lick analysis | Prefer the explicitly reported >30% criterion. Exclude these trials from the joint decoder, because retaining them with fabricated lick labels would corrupt one output; loss is only 0.66% and is exactly reproduced in available sessions. |
| Reward zones | `behavior.get_reward_zones` derives A/B/C from scene, switches at trial 30 | Identifiers contain scene schedules; `reward_zone` stream marks entries but not labels | A=80–130, B=200–250, C=320–370 cm | Parse scene identifier with reference zone coordinates; switch after 30 trials. Validate environment stream and zone-entry positions. |

Final consistent understanding: the NWBs store synchronized raw Suite2p fluorescence/neuropil, a non-reference deconvolution, curated ROI labels, and behavior at imaging-frame resolution. Conversion must recreate reference per-trial dF/F and OASIS events, exclude manual non-cells and speed-correlated putative interneurons, remove only the 81 documented corrupt-lick trials, and retain every other start-to-teleport trial from all 152 available sessions.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Fluorescence/plane0`, `Neuropil/plane0`, segmentation `iscell` | `neural` | Keep `iscell[:,0]==1`; per-trial 0.7 neuropil subtraction, 20-s (300-sample) maximin baseline, dF/F, Gaussian σ=2 samples, OASIS (`tau=0.7`, 15.5078125 Hz); exclude dF/F–speed r>0.5; slice trials; float32 neurons×time | `preprocessing.dff`, `spatial.is_putative_interneuron` | Recreates paper `events`; pool planes as paper does. |
| behavior timestamps | `input[0]` | `timestamp - trial_start_timestamp`, seconds, repeated as time row | synchronized VR processing | Continuous time-varying. |
| `environment` | `input[1]` | Mode within trial, preserve 0=ENV1 / 1=ENV2, repeated through trial | `behavior.get_trial_types` | Binary per-trial context. |
| `trial number` | `input[2]` | Mode within trial (native zero-based number), repeated through trial | `behavior.get_trial_types` pattern | Continuous per-trial covariate. |
| `Reward.timestamps` | `input[3]` | Map deliveries to all original start-to-teleport intervals; for trial i use outcome i−1; first trial=0; repeat through trial | `behavior.get_trial_types` | 0=previous omitted/no preceding trial, 1=previous rewarded. Excluding a corrupt-lick trial does not alter the next trial’s true predecessor. |
| `position` + active reward-zone bounds | `output[0]` | Signed distance to nearest zone location: `pos-start` before, 0 inside, `pos-end` after; discretize into 7 specified bins | paper RR logic adapted to requested linear zone distance | Boundary policy: −50 starts bin1; −10 remains bin1; +10 remains bin4; +50 remains bin5; zero is exactly bin3. |
| `position` | `output[1]` | 5 bins: <90, [90,180), [180,270), [270,360], >360 | requested task | Time-varying. Tiny synchronized endpoint overshoots remain in extreme bins. |
| `speed` | `output[2]` | 5 bins: <2, [2,10), [10,20), [20,40], >40 cm/s | requested task; paper uses continuous speed | Time-varying; stationary samples retained because class 0 is required. |
| `lick` | `output[3]` | `lick>0` → 1 else 0 | `glmUtils.get_timeseries_data`, lick Methods | Time-varying. Exclude trials meeting manuscript corruption rule (>30% frames have cumulative count>2). |
| NWB identifier scene + trial index | `output[4]` | Parse active zone; switch after 30 original trials; A/B/C → 0/1/2; repeat through trial | `behavior.get_reward_zones` | A=80–130, B=200–250, C=320–370 cm. |
| current-trial mapped reward delivery | `output[5]` | 0=no delivery in interval, 1=delivery; repeat through trial | `behavior.get_trial_types` | Three ITI deliveries are intentionally not trial rewards. |
| `subject.subject_id` | `subjects`, `subject_idx` | Sorted unique IDs and per-session index | — | 11 mice. |
| CA1 experiment anatomy | `brain_regions`, `brain_region_idx` | `['CA1']`; all retained cells index 0 | paper/DANDI metadata | Both imaging planes are CA1 and are pooled. |

### Key Decisions
1. **Sessions**: Convert all 152 NWBs in deterministic path order; `--sample` selects m11 days 3 and 5 so two small switch sessions jointly contain reward zones A/B/C.
2. **Neural representation**: Reconstruct paper deconvolved events rather than use NWB `Deconvolved`, based on the direct mismatch established in Step 4.
3. **No place-cell restriction**: Retain all curated non-interneurons because position, speed, licking, and outcome are all requested; place-cell selection would discard relevant non-spatial cells and is specific to the paper’s remapping decoder.
4. **Native temporal sampling**: Do not resample. All behavior timestamps have a 64.483627 ms median interval, already synchronized to one neural row; trim ten +1-row neural sessions to the common length.
5. **Trial validity**: Require ordered start/teleport events, at least two samples, scanning=1 for retained samples, finite required behavior, and exclude the 81 corrupt-lick trials. No speed filtering because stationary speed is a requested output class.
6. **Mixed time-varying/per-trial variables**: Store all four inputs and all six outputs as 2-D arrays with per-trial labels repeated over time. This is necessary because one NumPy array cannot mix 1-D and 2-D rows and makes alignment explicit.
7. **Categorical storage**: Outputs use integer class IDs (`int16`); neural and inputs use `float32`. Neural arrays own their trial slices so the pickle is self-contained.
8. **Timing metadata**: `time_bin_size=64.483627204...` ms, alignment=`trial_start / entry to 0-cm corridor`, `off_start=0.0`, `off_end=None` because trial durations vary.
9. **Output labels**: `input_names=['time_from_trial_start_s','environment','trial_number','previous_trial_outcome']`; `output_names=['distance_to_reward_zone','absolute_position','speed','lick','reward_zone_location','reward_outcome']`. Value labels will explicitly reproduce all thresholds.
10. **Unused but checked variables**: `scanning`, `trial_start`, `teleport`, `reward_zone`, `autoreward`, segmentation plane IDs, and all timestamps are used for validity/provenance checks; raw pixel masks/background/acquisition placeholder are not decoder variables.

### Planned Sanity Checks
- [ ] Independently load an NWB with `pynwb` and use `np.allclose` to compare three raw F/Fneu-derived dF/F/event samples against converted neural values.
- [ ] Independently reconstruct time/environment/trial/previous-outcome for a named trial and compare full input matrix with `np.allclose`.
- [ ] Independently reconstruct position, speed, lick, zone, and reward classes for three trials spanning a switch and compare full output matrices with `np.allclose`.
- [ ] Assert every stream’s time length matches per trial; no NaN/Inf; uniform 64.483627-ms source interval; ≥2 trials/session.
- [ ] Reconcile 152 sessions, 12,216 raw intervals, exactly 81 lick-corrupt exclusions, and retained totals; compare switch-day cell and trial/reward statistics to paper.
- [ ] Confirm all 11 day-8 environment transitions occur at original trial index 30 and all reward-zone switches follow identifier schedules.
- [ ] Check discretization edges with synthetic exact-boundary values and assert class ranges 0–K−1.
- [ ] Visually overlay continuous position/speed/lick with their categorical bins and neural events in `--show-processing` plots for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `/app/convert_data.py` with required positional output, default/explicit `--full`, `--sample`, and `--show-processing` options.
- The script uses only `pynwb.NWBHDF5IO` for NWB access, recreates reference dF/F and OASIS events, filters manual cells/interneurons, validates aligned trial shapes/classes, and writes the exact requested dictionary.
- Built-in boundary unit tests cover every discretizer. Syntax compilation and CLI help completed without errors.
- `--show-processing` generates eight-panel figures covering raw F/Fneu, neuropil correction, baseline, dF/F, OASIS, trial alignment/zones, all inputs, source behavior, and all output discretizations.

Code inefficiencies identified:
Loading 92 GB of NWB assets and applying filters cell×frame is intrinsically expensive. Building full-session event matrices would also duplicate final trial storage.

Code speedups added:
Filter manual cells before signal processing; vectorize speed correlations and all class transforms; process baseline trial-wise; OASIS only retained non-interneurons; immediately store final float32 trial events; process one session at a time; avoid image acquisition reads.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 343 retained cell-sessions (344 manual; one speed-correlated exclusion) |
| Neurons / session | [154, 189] |
| Subjects | 1 (m11) |
| Sessions / subject | 2 (days 3 and 5) |
| Trials (total) | 160 retained / 160 source |
| Trials / session | [80, 80] |
| Time input range | [0.0, 26.8] s |
| Environment / trial / previous outcome | [0,0] / [0,79] / [0,1] |
| Distance output distribution | [0.236,0.092,0.047,0.275,0.022,0.071,0.257] |
| Position output distribution | [0.254,0.182,0.210,0.209,0.145] |
| Speed output distribution | [0.109,0.057,0.060,0.273,0.501] |
| Lick output distribution | [0.805,0.195] |
| Zone output distribution | [0.464,0.207,0.328] |
| Outcome output distribution | [0.105,0.895] |

### Processing Plots Review
`processing_sub-m11_ses-03.png` and `processing_sub-m11_ses-05.png` were visually inspected. Raw F/Fneu, corrected F/baseline, smooth dF/F/OASIS, monotonic 0→450 cm trial trajectories, zone overlays, frame-aligned inputs, source speed/lick, and every discretized output are mutually aligned. Distance is exactly class 3 throughout the zone. No temporal shift or discretization anomaly was visible.

The validator reported “Data format is valid, no errors or warnings.” All six outputs cover their expected global class ranges in the two-session sample.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Filter candidate ROIs before dF/F and OASIS | Avoids processing 55.6% of candidate ROIs globally |
| Deconvolve only non-interneurons and store float32 per trial | Avoids full-session event duplication and halves final bytes versus float64 |
| Vectorized correlation/discretization | Removes per-cell/per-frame Python loops |
| Multi-plane plane-wise loading | Reads only selected ROIs while correctly concatenating global ROI regions |

| Step | Time / Session | Estimated Total Time |
| Small sample sessions (plots included) | 1.33–1.47 s | — |
| Representative 1,143-cell session | 2.30 s | — |
| Large 2,323-cell/two-plane session | 7.90 s | — |
| Full conversion estimate | ~3 s typical, ≤8 s observed large | ~8–10 min for 152 sessions plus pickle write (below 15 min) |

Iteration: a performance probe revealed that m17/m18 response series are split into `plane0` and `plane1`. The initial loader assumed only `plane0`; it was fixed to concatenate all plane series through each `DynamicTableRegion`. The sample conversion and all Step 7 checks were rerun successfully after the fix.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Distance to reward zone | 0.4260 | 0.4072 |
| Absolute position | 0.5842 | 0.4816 |
| Speed | 0.4760 | 0.3847 |
| Lick | 0.7846 | 0.7729 |
| Reward zone location | 0.8451 | 0.8447 |
| Reward outcome | 0.7198 | 0.7152 |

Training completed on CUDA. Loss decreased monotonically from 2.232173 (epoch 1) to 0.958377 (epoch 200); test loss was 0.951147. Every validation balanced accuracy exceeded uniform chance (respectively 0.1429, 0.2, 0.2, 0.5, 0.3333, 0.5), with no train/validation gap above 1.21×.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9,533,041,126 bytes (9.53 GB)
- `conversion_full_out.txt`: created; conversion completed in 442.9 s (7.38 min)
- `verification_full_out.txt`: created; no errors or warnings

The complete conversion contains 152 sessions from all 11 mice. There are 12,216 source
trials, of which exactly 81 are removed by the paper's lick-sensor failure criterion, leaving
12,135 trials and 2,576,026 behavioral time bins. The current NWB files contain 138,678
manually accepted (`iscell`) ROIs; 402 speed-correlated putative interneurons are removed,
leaving 138,276 neurons. Every session retains at least 40 trials (well above the required two).

Spot checks covered the first and last sessions, a day-8 environment-switch session, a
two-plane session with 2,323 cells, and a session with lick-corrupt trials. Neural, input, and
output time dimensions agreed in every case. Plane membership and source-trial indices in
`session_info` also agreed with their NWB sources.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 73,512 all cells on 77 switch days | manual `iscell`, then speed-correlation exclusion | 138,678 manual all sessions; 74,652 on switch days | 138,276 all sessions; 74,496 on switch days | Yes, with documented updated multi-plane files |
| Mean neurons/session | 954 +/- 453 on switch days | same curation | 969.5 manual on switch days | 967.5 +/- 481.0 retained on switch days; 909.7 all sessions | Yes |
| Subjects | 11 switch mice | 11 analyzed mice | 11 | 11 | Yes |
| Sessions | 77 switch sessions | days 3,5,7,8,10,12,14 | 152 total; 77 switch | 152 total; 77 switch | Yes |
| Trials (total) | 12,376 before exclusions | exclude lick-failure trials | 12,216 (m11 day 1-2 absent) | 12,135 after exactly 81 exclusions | Yes |
| Trials/session (mean) | approximately 80 | source trial table | 80.37 | 79.84 | Yes |
| Time input range (s) | trial start to teleport | timestamps relative to start | [0, 216.5] | [0, 216.5] | Yes |
| Environment range | ENV1/ENV2 | binary scene mapping | [0, 1] | [0, 1] | Yes |
| Trial-number range | session trial index | source trial index | [0, 99] | [0, 99] | Yes |
| Previous-outcome range | omitted/rewarded | one-trial shift | [0, 1] | [0, 1] | Yes |
| Distance-bin distribution | not reported | task cut points | source-derived | [.251,.102,.073,.238,.021,.072,.243] | Yes |
| Position-bin distribution | track is 450 cm | 90-cm bins | source-derived | [.212,.177,.231,.226,.154] | Yes |
| Speed-bin distribution | not reported | task cut points | source-derived | [.117,.087,.134,.319,.343] | Yes |
| Lick distribution | lick events used; 81 failures | cumulative-sensor difference | source-derived | [.777,.223] | Yes |
| Reward-zone distribution | zones A/B/C | trial reward location | trial counts [4172,3974,3989] | [.332,.336,.333] | Yes |
| Outcome distribution | about 15% omissions | in-trial reward delivery | trial counts [1864,10271] | [.154,.846] by trial; [.158,.842] by time bin | Yes |

Three reward events occur after teleport during unusually long inter-trial intervals and are not
assigned to either adjacent trial, matching explicit trial boundaries and the paper's definition
of trial outcome. The validator's timepoint-weighted outcome distribution consequently differs
slightly from the trial-weighted 84.64% reward rate.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: searched `verification_full_out.txt` for errors and
   warnings. The only match is the validator's statement “Data format is valid, no errors or
   warnings.” Thus there are no unresolved validator messages.
2. **Independent raw-neural equivalence**: loaded
   `sub-m11_ses-03_behavior+ophys.nwb` directly with `pynwb` in a standalone check that does
   not import `convert_data.py`. Independently selected `iscell[:,0]`, combined ROI response
   series, computed neuropil-corrected per-trial maximin dF/F, applied the speed-correlation
   filter, and ran OASIS. `np.allclose(rtol=1e-6, atol=1e-6)` passed for the complete
   neuron-by-time matrices of source trials 0, 5, 29, 30, and 79. Maximum absolute differences
   were respectively 3.58e-7, 2.38e-7, 1.79e-7, 1.79e-7, and 9.22e-8: float32 rounding only.
3. **Independent raw-input equivalence**: for the same five trials, independently reconstructed
   time since raw `trial_start`, modal environment, modal source trial number, and preceding
   source-trial reward outcome. All four full time series passed `np.allclose`; this includes
   the first-trial default and trials immediately before/after the switch.
4. **Independent raw-output equivalence**: independently reconstructed signed distance to the
   active zone, position, speed, binary lick, zone identity, and in-trial reward outcome from
   raw NWB behavior. All six outputs passed exact `np.allclose(rtol=0, atol=0)` on the five
   trials. Trials 29/30 explicitly establish the absence of a zone-switch off-by-one error.
5. **Whole-dataset invariants**: across 12,135 converted trials, zero neural/input/output shape
   failures, zero nonzero/negative trial-start times, zero nonmonotonic time vectors, zero class
   range failures, zero failures of per-trial variable constancy, and zero preceding-outcome
   mismatches for adjacent source trials. The exact timepoint count was 2,576,026 and observed
   categorical minima/maxima were `[0,0,0,0,0,0]`/`[6,4,4,1,2,1]`.
6. **Multi-plane edge case**: independently inspected m18 day 3 through `pynwb`. Its two
   fluorescence series jointly cover all 2,341 manually accepted global ROI IDs exactly once
   as a set; the independent speed filter removes 18, agreeing with 2,323 converted neurons.
7. **Neuron-curation statistic**: 402/138,678 cells were excluded. The session-wise percentage
   is 0.348 +/- 0.609% (range 0–3.817%) over all days and 0.260 +/- 0.375% on switch days,
   reasonably consistent with the paper's 0.42 +/- 0.85% given its mouse/day aggregation and
   earlier multi-plane curation snapshot.
8. **Reference size/statistic reconciliation**: 11 mice, 77 switch sessions, 12,216 source
   trials, 81 lick-corrupt trials, ~15.36% omissions, 15.5-Hz sampling, 450-cm track, switch
   after 30 trials, and zone coordinates all agree exactly or to rounding with the paper. The
   only cell-count difference is the already localized updated m17/m18 multi-plane curation;
   discarding legitimate current-NWB cells would be less defensible than retaining them.

### Reference Code Comparison
| Processing step | Reference implementation | Conversion implementation | Comparison / justification |
|-----------------|--------------------------|---------------------------|----------------------------|
| Data loading | `preprocessing.create_sess`; synchronized Suite2p F/Fneu and VR | `NWBHDF5IO`, `load_behavior`, `load_selected_roi_series` | Same streams; NWB timestamps are the authoritative already-synchronized imaging-frame clock. No direct HDF5 access is used. |
| Neuron/trial filtering | `iscell[:,0]`; dF/F-speed r>0.5; lick failure >30% | manual ROI mask, `speed_correlations`, trial QC in `convert_session` | Same curation. Place-cell filtering is deliberately not used because it is specific to spatial analyses and would remove signals relevant to five other requested outputs. Exactly 81 lick trials are removed. |
| Temporal alignment | `trial_start_inds` to `teleport_inds`; legacy arrays are one-based | direct NWB event positions, `[start:stop]` | Same conceptual interval. No legacy -1 shift because NWB event arrays are explicit zero-based samples; raw equivalence and start/end position checks confirm this. |
| Binning | ~15.5-Hz events; spatial analyses separately use 10-cm bins | native ~64.484-ms imaging frames | Correct for the requested temporal decoder; spatial 10-cm binning would conflict with trial-start time alignment and specified categorical cut points. |
| Input construction | no identical paper decoder inputs | raw timestamp/environment/trial/outcome streams | Task-mandated mapping; modal categorical values are robust to frame representation, and previous outcome is shifted in source-trial rather than retained-trial order. |
| Output construction | `get_timeseries_data`, `get_trial_types`, `get_reward_zones` | raw position/speed/lick/reward plus scene-derived zones | Same raw variables and zone schedule. Requested seven/ five/five-class discretizations replace the paper's circular continuous coordinate as explicitly required. |

### Edge-Case Review
- Trial bounds are start-inclusive/teleport-exclusive; raw trial 0 begins at time zero in the
  conversion and the last included sample precedes teleport. No trial has fewer than two bins.
- Distance/position/speed boundary unit tests cover every exact cut point. In particular,
  distance -10 belongs to class 1, 0 to class 3, +10 to class 4, +50 to class 5; position 360
  and speed 40 remain in their inclusive classes as specified.
- The current source trial number is retained after exclusions, while previous outcome comes
  from the immediately preceding **source** trial. Thus removal of a corrupt-lick trial does
  not silently relabel task history.
- Ten multi-plane files and one single-plane file have one extra neural row; common-length
  trimming removes only that unmatched trailing row. Every included trial remains inside the
  common synchronized interval.
- Three rewards outside all start-to-teleport intervals remain unmapped rather than leaking into
  an adjacent outcome. All NaN/invalid/scanning checks pass on retained intervals.

### Issues Found and Resolved
- The first independent neural comparison used an unnecessarily strict absolute tolerance of
  1e-7 and flagged a maximum 3.58e-7 difference. Rechecking with 1e-6, appropriate for stored
  float32 OASIS output, passed all five complete matrices; inputs and outputs were unchanged.
- The first multi-plane assertion confused 2,341 manual cells with the 2,323 post-filter count.
  Direct inspection confirmed all 2,341 manual ROIs are covered and the documented 18-cell
  speed filter exactly explains the difference. This was a test expectation error, not a data
  conversion error.
- No conversion defect was found in this review, so no regeneration was required. All checks
  were rerun after clarifying the two test expectations and passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Training loss fell monotonically from 2.026457 at epoch 1 to
  0.872331 at epoch 200; test loss was 0.777774. The CUDA run completed successfully without
  memory fallback and created the requested sample plots.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Distance to reward zone | 0.6623 | 0.5741 | 4.02x validation chance |
| Absolute position | 0.7624 | 0.6960 | 3.48x validation chance |
| Speed | 0.6518 | 0.6025 | 3.01x validation chance |
| Lick | 0.7721 | 0.7514 | 1.50x validation chance |
| Reward-zone location | 0.9015 | 0.8488 | 2.55x validation chance |
| Reward outcome | 0.8326 | 0.5794 | 1.16x validation chance; investigated in Step 12 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Train / validation | Expectation from paper |
|----------|------------------------------|--------|--------------------|--------------------|------------------------|
| Distance to reward zone | 0.5741 | 0.1429 | 4.02x | 1.15x | No matching categorical accuracy; circular RR position is significantly decodable and cross-switch RR-cell standardized effect size is 2.7 vs shuffle. |
| Absolute position | 0.6960 | 0.2000 | 3.48x | 1.10x | No absolute-position class accuracy reported; within-condition circular RR-position standardized effect sizes are 2.6/3.3/3.2 for RR/TR/non-RR populations. |
| Speed | 0.6025 | 0.2000 | 3.01x | 1.08x | Not decoded in paper. |
| Lick | 0.7514 | 0.5000 | 1.50x | 1.03x | Not decoded in paper. |
| Reward-zone location | 0.8488 | 0.3333 | 2.55x | 1.06x | Not decoded in paper. |
| Reward outcome | 0.5794 | 0.5000 | 1.16x | 1.44x | Not decoded in paper; reward omissions are random on about 15% of trials. |

Every output is above chance; five of six meet or exceed 1.5x chance. No train/validation
ratio exceeds the specified 1.5x overfitting trigger. The paper's only decoder is circular-linear
RR position regression scored by mean cosine similarity (1 perfect, 0 random), not categorical
accuracy. Its reported numerical comparisons are standardized effects against shuffled data,
not accuracies, so there is no honest one-to-one numerical paper accuracy to substitute in this
table. The strong distance and position results are nevertheless qualitatively consistent with
Fig. 3.

### Focused Reward-Outcome Investigation
Reward outcome was above chance but below 1.5x chance, so all requested diagnostics were run:

1. **Three raw trials**: direct `pynwb` inspection identified m11 day-3 trials 0 and 2 as
   rewarded (delivery 9.286 s and 4.772 s after start) and trial 6 as omitted. These labels,
   along with complete independently reconstructed outputs for trials 0, 5, 29, 30, and 79,
   exactly match the pickle (`np.allclose` with zero tolerance for outputs).
2. **Temporal alignment plot**: visually reviewed `processing_sub-m11_ses-03.png`. Raw F/Fneu,
   maximin baseline, dF/F/OASIS, position and active zone, trial-start time, speed/lick, and all
   discretized outputs share the same samples. Position begins near 0 cm and ends near 450 cm;
   class transitions occur at the plotted spatial boundaries with no visible shift.
3. **Variation**: retained trial outcomes are 1,864 omitted and 10,271 rewarded (15.36% vs
   84.64%); the validator's timepoint distribution is 15.8%/84.2%. This is expected from the
   paper and far from a 99% class collapse. Balanced class loss was used by the validator.
4. **Causal timing**: of 2,169,754 bins in rewarded trials, 865,799 (39.90%) occur before the
   first reward delivery. The required per-trial label marks those bins “rewarded” even though
   current neural activity cannot yet reveal a random future omission. Omitted trials never
   contain a delivery. This task-required representation places a principled ceiling on
   timepoint decoding and explains why outcome is the weakest variable.
5. **Filtering/reference match**: the manual ROI, r>0.5 interneuron, start-to-teleport, and lick
   corruption filters were rechecked in Step 10. Notably, the paper's GLM uses a different
   *causal* predictor that switches from 0 to 1 only after reward delivery; changing this task's
   explicitly per-trial outcome to that signal would violate the requested output definition.
6. **Prediction plot**: visually reviewed `predictions.png`; neural streams are nonempty and
   time-varying, and position/speed/lick predictions track their targets without a systematic
   lag. Per-trial outcome errors are consistent with random omissions rather than an alignment
   artifact.

Conclusion: the 0.5794 outcome accuracy reflects the random, partly future event requested as
a whole-trial label, not a conversion bug. Relabeling pre-delivery timepoints or leaking current
outcome into the inputs would improve accuracy artificially and is therefore rejected.

### Issues Found and Resolved
- No below-chance output, >1.5x train/validation gap, class collapse, filtering mismatch, or
  temporal offset was found. The only diagnostic flag was reward outcome below 1.5x chance;
  raw-label, timing, variation, plotting, and reference checks above resolve it as an intrinsic
  task limitation. No conversion change or retraining iteration was warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading example, format, processing,
  statistics, validation commands, and decoder results.
- [x] `cache/` folder created; `README_CACHE.md` inventories its contents.
- [x] All files organized. Generated Python bytecode was moved under `cache/`; no standalone
  investigation scripts were created. Required data, logs, processing plots, and decoder plots
  remain at the project root for direct inspection.

Final audit confirmed that all 14 workflow steps are COMPLETE and every required deliverable
exists: `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
`README.md`, both conversion logs, both verification logs, both decoder logs, and all requested
processing/decoder plots. The final full validator reports no errors or warnings, and full
training completed all 200 epochs successfully.
