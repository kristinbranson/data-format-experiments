# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward dataset, from the provided paper/code/data bundle
- **Date started**: 2026-09-21
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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` | `code/src/reward_relative/preprocessing.py` | LOADING | Builds a `TwoPUtils.sess.Session` object from imaging and VR paths, then appends aligned scan/VR/behavior data. |
| `append_session_data` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Calls `sess.align_VR_to_2P()`, adds behavior timeseries, and creates position-binned trial matrices. |
| `dff` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Computes trial-restricted deltaF/F using neuropil subtraction and `maximin` baseline; optionally deconvolves to `events`. |
| `multi_anim_sess` | `code/src/reward_relative/utilities.py` | PROCESSING | Loads saved session pickles, computes dFF / events, place-cell outputs, and per-trial behavioral annotations for each animal. |
| `get_trial_types` | `code/src/reward_relative/behavior.py` | PROCESSING | Creates per-trial rewarded/omission labels and environment morph identity from aligned VR samples. |
| `get_reward_zones` | `code/src/reward_relative/behavior.py` | PROCESSING | Infers reward-zone coordinates and labels (`A/B/C`) from the scene name and switch day structure. |
| `define_trial_subsets` | `code/src/reward_relative/behavior.py` | PROCESSING | Splits trials into set 0 / set 1, using reward-switch boundaries or first-vs-second half when needed. |
| `get_omission_inds` / `get_omission_trials` | `code/src/reward_relative/rewardAnalysis.py` | PROCESSING | Finds omission trials and reward-zone-entry indices for omission labeling. |
| `get_timeseries_data` | `code/src/reward_relative/glmUtils.py` | PROCESSING | Produces continuously sampled decoder inputs/outputs from aligned framewise data, masking low-speed and invalid samples. |
| `quick_load_multi_anim_sess` | `code/src/reward_relative/utilities.py` | LOADING | Loads pre-saved processed pickles for an experimental day. |
| `get_sess_pkl_path` / `load_sess_pickle` | `code/src/reward_relative/utilities.py` | LOADING | Resolves and loads the per-session pickle used as the core analysis object. |
| `sessions_dict.single_plane` / `multi_plane` | `code/src/reward_relative/sessions_dict.py` | CURATION | Declares which animal/day/session combinations belong to the experiment and their scene metadata. |

### Notes
- The reference repo is built around `sess` pickle objects generated upstream with `TwoPUtils`, not raw NWB readers inside this repo.
- Behavioral data are aligned to imaging frames before most downstream analysis. The aligned framewise table is `sess.vr_data`, with one row per imaging sample.
- Framewise variables used downstream include `pos`, `trialnum`, `speed`, `lick`, `reward`, `rzone`, `morph`, `tstart`, and `teleport`.
- Trial boundaries are represented by `sess.trial_start_inds` and `sess.teleport_inds`. Reference code commonly iterates over trials using slices `start-1:stop-1`, so I need to watch for 1-based vs 0-based boundary conventions when converting.
- The reference pipeline computes dF/F after session creation. `sess` itself does not initially contain dF/F; `multi_anim_sess` computes it with `preprocessing.dff(...)`.
- The default dF/F settings in `utilities.default_dff_method` are: neuropil subtraction with coefficient 0.7, `maximin` baseline, and `keep_teleports=False`.
- `preprocessing.dff` keeps only samples inside trial intervals unless `keep_teleports=True`, computes baseline within each trial, then smooths dF/F and can deconvolve with suite2p OASIS to produce `events`.
- Decoder-related code in `Fig3_Decoder.md` does not use raw fluorescence. It uses `glmUtils.get_timeseries_data(...)`, which extracts `sess.timeseries['events']` as neural activity and aligned behavior at imaging-frame resolution.
- `glmUtils.get_timeseries_data(...)` masks out NaNs, licks flagged by a sensor-error heuristic, and optionally samples with speed below threshold (`use_speed_thr=2` in the decoder notebook).
- Reward outcome is defined per trial from any reward delivery during that trial, usually requiring reward-zone occupancy as well.
- Reward-zone identity/location is inferred from `sess.scene` using `behavior.get_reward_zones`, with switch days split at `change_reward_trial` if present, otherwise default switch at trial 30.
- The notebook decoder in the repo targets circular position relative to reward and uses subsets of cell classes; for this conversion task I will preserve the same framewise alignment/processing logic where applicable, but use all available neurons unless the data or validation suggests reference curation excludes some cells.
- Cell curation is manual upstream through Suite2p `iscell.npy`; I have not yet seen additional automatic neuron-quality filtering inside this repo beyond using the curated `sess` objects and downstream place-cell classifications.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains one NWB file per subject-session plus `dandiset.yaml`.
- Subject folders present: `sub-m3`, `sub-m4`, `sub-m7`, `sub-m11`, `sub-m12`, `sub-m13`, `sub-m14`, `sub-m15`, `sub-m17`, `sub-m18`, `sub-m19`.
- Total NWB session files: 152.
- Session file naming pattern: `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`.
- Each NWB has:
  - `processing/behavior/BehavioralTimeSeries` with frame-aligned behavior streams:
    - `environment`
    - `position`
    - `speed`
    - `lick`
    - `reward_zone`
    - `scanning`
    - `trial number`
    - `trial_start`
    - `teleport`
    - `autoreward`
    - `Reward` (sparse reward-delivery event series, not framewise)
  - `processing/ophys` with:
    - `Fluorescence/plane0`
    - `Neuropil/plane0`
    - `Deconvolved/plane0`
    - `ImageSegmentation/PlaneSegmentation`
  - ROI metadata columns in `PlaneSegmentation`: `pixel_mask`, `iscell`, `planeIdx`.
- Neural arrays are stored as time x ROI matrices. Example single-plane session: `(26279, 2999)` for deconvolved activity. Example multi-plane session: `(22634, 936)`.
- `iscell` has shape `(n_rois, 2)` and matches Suite2p convention: first column is the 0/1 cell mask, second column is the classifier score/probability.
- Brain region metadata are uniform across files: imaging plane location is `hippocampus, CA1`.
- There is no NWB trials table; trial structure must be reconstructed from the framewise `trial_start`, `teleport`, and `trial number` streams.
- Important structural differences from the reference `sess` objects:
  - Reward delivery is sparse (`Reward`) rather than framewise binary.
  - dF/F is not explicitly stored; raw fluorescence, neuropil, and deconvolved activity are stored.
  - Two animals (`m17`, `m18`) are multi-plane, marked by `planeIdx` values `[0, 1]`.
- Sampling / alignment observations:
  - 124 sessions are single-plane with deconvolved `starting_time.rate = 15.5078125 Hz`.
  - 28 sessions are two-plane with deconvolved `starting_time.rate = 31.015625 Hz`, but the behavior timestamps advance at ~`0.06448 s` per sample (effectively ~15.5 Hz). This metadata mismatch must be handled carefully in conversion.
  - In 10 multi-plane sessions, ophys matrices are one sample longer than behavior matrices.
- Edge case already identified:
  - `sub-m11_ses-03` has 81 unique nonnegative `trial number` labels (`0..80`) but only 80 `trial_start` and 80 `teleport` markers. Later inspection showed the extra label is a trailing post-teleport tunnel fragment with no on-track `trial_start`, so complete trials should be defined from `trial_start` / `teleport`, not by unique trial labels.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 curated cells (`iscell[:,0] == 1`) |
| Neurons / session | mean 912.36, median 921.5, range 155-2341 |
| Subjects | 11 |
| Sessions / subject | 12-14 (152 total sessions; all 14 except `sub-m11` with 12) |
| Trials (total) | 12,216 by `trial_start` markers; 12,217 by unique nonnegative `trial number` labels |
| Trials / session | mean 80.37, median 80, range 41-100 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not explicitly stated | Paper/methods do not give a total neuron count across the released switch-task NWB bundle. | 
| Neurons / session | 155-2172 putative pyramidal neurons/session | "This approach yielded 155–2172 putative pyramidal neurons per session" |
| Subjects | 11 switch-task mice in this experiment; 14 mice if including the fixed-condition cohort | "Each mouse encountered a different starting reward zone and sequence of reward zone switches... (n = 11 mice)." and "An additional 'fixed-condition' cohort (n = 3 mice)..." |
| Sessions / subject | 14 task days planned; m11 imaging starts on day 3 | "a total of 14 days" and "The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Trials (total) | Not explicitly stated for switch-task NWB subset | Methods give per-session target/mean, not an exact total for the released switch cohort. |
| Trials / session | Target 80-100; mean 80.5 +/- 7.4 imaging trials/session | "We targeted 80–100 trials per session... (mean ± s.d., 80.5 ± 7.4 trials across 14 mice, all imaging days)." |
| Neural data time bin | ~15.5 Hz imaging frame rate (~0.0645 s/sample); m17/m18 interleaved at ~31 Hz total, ~15.5 Hz/plane | "All behavioral and neural time series were sampled at ~15.5 Hz" and "frames bidirectionally imaged at ~31 Hz interleaved in the scan for a sampling rate of ~15.5 Hz per plane" |
| Behavior data time bin | ~15.5 Hz (~0.0645 s/sample) | "All behavioral and neural time series were sampled at ~15.5 Hz" and lick QC refers to "0.0645 s imaging frame samples" |
| Reward rate | ~85% rewarded, ~15% omitted | "Reward was randomly omitted on approximately 15% of trials" |
| Reward zone locations | A: 80-130 cm; B: 200-250 cm; C: 320-370 cm | "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" |
| Switch timing | Switch occurs after 30 trials; first 10 trials of a new condition can be autorewarded if mouse fails to lick | "On day 3... the zone was moved after 30 trials" and "On the first ten trials of any new condition... the reward was automatically delivered..." |
| Decoder evaluation set | 77 sessions, 11 mice, 7 switch days | "Mean decode score compared to the mean of shuffles per session (n = 77 sessions, 11 mice, seven switch days)" |
| Decoder accuracy expectation | RR cells decode above shuffle both pre-switch and cross-switch; TR and non-RR decode above shuffle only within pre-switch set | "when we tested on trials after the reward switch, only the RR population decoded the animals' RR position better than shuffle" |
| RR decoder spatial extent | Above-shuffle RR decoding spans roughly -104.5 +/- 20.1 cm to +152.7 +/- 22.9 cm relative to reward start | "range of z-scored decode >2: -104.5 ± 20.1 cm to +152.7 ± 22.9 cm relative to reward zone start" |


### Processing Details
- Task structure:
  - 450 cm linear track.
  - Two environments (ENV1, ENV2).
  - Three hidden 50 cm reward zones at A/B/C.
  - Switches occur on days 3, 5, 7, 8, 10, 12, 14; each switch occurs after 30 trials.
- Imaging / alignment:
  - Single-plane imaging at ~15.5 Hz.
  - m17 and m18 are two-plane, interleaved at ~31 Hz total, corresponding to ~15.5 Hz per plane.
  - Reference analyses treat behavioral and neural streams as synchronized framewise time series.
- Decoder-related processing in the paper:
  - Neural signal for decoding is deconvolved calcium events.
  - Samples with running speed <2 cm/s are excluded.
  - RR position is represented in circular coordinates from `-pi` to `pi`, centered on reward-zone start.
  - Decoder uses tenfold cross-validation and compares against circularly time-shifted shuffles.
- Other reference processing relevant for consistency checks:
  - Spatial analyses use 45 position bins of 10 cm each across the 450 cm track.
  - Licks are converted to binary after quality control and then smoothed / binned depending on analysis.
  - Teleport periods are usually excluded from standard spatial analyses unless explicitly analyzing teleport activity.

### Curation Steps

**Neuron curation rules**:
- Suite2P ROIs were manually curated.
- Excluded ROIs include:
  - multiple somata or dendrites,
  - no visually obvious transients,
  - suspected overexpression,
  - high continuous fluorescence fluctuation suggestive of interneurons.
- In multi-plane mice, ROIs were identified separately by plane and pooled for most analyses.

**Trial curation rules**:
- Imaging sessions targeted 80-100 trials but could end early if the mouse stopped licking/running consistently or imaging exceeded 50 min.
- Reward omissions occur randomly on ~15% of trials.
- For licking analyses, trials with capacitive-sensor artifacts were removed if >30% of 0.0645 s samples had cumulative lick count >2; methods report 81 / 12,376 imaged trials removed (~0.65%) across 11 switch mice.
- For standard neural spatial analyses, samples with speed <2 cm/s are excluded.
- For omission analyses, only trial sets with at least three omission trials were included.

### Decoders Trained
| Decoded variable | Accuracy |
| Reward-relative position from RR cells | Above shuffle both when testing within pre-switch trials and when testing post-switch; main text states only RR cells remain above shuffle cross-switch |
| Reward-relative position from TR cells | Above shuffle when trained/tested within pre-switch trials, not above shuffle when tested post-switch |
| Reward-relative position from non-RR remapping cells | Above shuffle when trained/tested within pre-switch trials, not above shuffle when tested post-switch |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session inventory | `sessions_dict.py` includes exp_day 0 running-training entries for mice m12-m19 and 14 task days for switch mice; m11 imaging starts day 3 | NWB bundle contains only task sessions `ses-01..14` for most mice, and `ses-03..14` for m11 | m11 imaging started on day 3; switch task has 14 days | Treat NWB `ses-01..14` as task day / experimental day. Ignore code-only exp_day 0 running-training sessions because they are not in the NWB bundle and are outside the main task. m11 missing days 1-2 is expected. |
| Scene metadata | `sessions_dict.py` scene names define reward zone / environment structure | NWB `identifier` suffixes exactly match code scene names for all data sessions checked | Paper describes same day-by-day switch structure | Use NWB `identifier` scene names with the same logic as `behavior.get_reward_zones`. |
| Reward representation | Reference `sess.vr_data` has framewise `reward` and `rzone` columns | NWB stores `reward_zone` framewise but `Reward` as sparse event timestamps | Paper defines reward outcome per trial and rewarded state switching on reward delivery | Reconstruct framewise/per-trial reward information from `Reward` timestamps aligned to behavior timestamps; use reward-zone labels from scene metadata. |
| Trial boundaries | Reference code uses `trial_start_inds` and `teleport_inds`, slicing `start-1:stop-1` to exclude teleport/tunnel | NWB has framewise `trial_start` and `teleport`; `trial number` changes at teleport and includes the tunnel before track entry | Paper defines trials as laps of the 450 cm track followed by teleport | Build trials from `trial_start` to `teleport` (on-track segment only), not from `trial number` alone. Use `trial number` only as a consistency check / label source. |
| Extra trial label in `sub-m11_ses-03` | Reference code defines trials by `trial_start_inds` and `teleport_inds`, not by `trial number` alone | This NWB has 81 nonnegative `trial number` labels (`0..80`) but only 80 `trial_start` and 80 `teleport` markers | Paper expects ~80 trials/session and uses lap structure, not label count, as the true trial count | Inspection shows the final label-only segment (`trial 80`) is a trailing post-teleport tunnel fragment with no `trial_start`. Therefore the correct complete-trial count is 80, and start/teleport pairs should be treated as authoritative. |
| Multi-plane sampling rate | Reference methods: m17/m18 scanned at ~31 Hz interleaved, ~15.5 Hz per plane | NWB deconvolved series has `rate=31.015625`, but behavior timestamps advance by ~0.06448 s (15.5078125 Hz) and match sample count | Paper explicitly says effective sampling is ~15.5 Hz per plane | Treat sample spacing as 0.0644836272 s from behavior timestamps and ignore the misleading deconvolved `rate` attribute for multi-plane sessions. |
| Neural signal choice | Reference decoder notebook uses `sess.timeseries['events']` from deconvolution | NWB contains `Fluorescence`, `Neuropil`, and `Deconvolved`, but no explicit dF/F | Paper decoder uses "deconvolved calcium event timeseries" | Use NWB `Deconvolved` for neural input, with curated cells only (`iscell[:,0]==1`). |
| Cell filtering | Reference preprocessing relies on manual Suite2p curation, then uses curated sess objects | NWB exposes `iscell` per ROI and contains many non-cell ROIs if unfiltered | Paper describes manual curation yielding putative pyramidal neurons | Keep only ROIs with `iscell[:,0] == 1`; no extra automatic neuron-quality filter unless later validation exposes a problem. |
| Lick artifact threshold | Methods text says >30% of frame samples with cumulative lick count >2; code implements thresholds around 0.35 in downstream analysis | NWB lick stream contains cumulative counts >1 on some frames | Methods and code are close but not identical | Follow the actual reference code logic (`>35%` of samples in a trial with lick count >2) because the conversion should match released analysis behavior more closely than a rounded prose threshold. |

### Final understanding
- The released NWB files are a downstream export of the same frame-aligned data structure that the reference code calls `sess`, but not the same file format.
- Standard on-track analyses in the paper exclude the teleport/tunnel period; the appropriate trial window is from `trial_start` to `teleport`.
- Reward-zone identity should be derived from session scene metadata using the same reward-zone mapping as the reference code and methods:
  - A = 80-130 cm
  - B = 200-250 cm
  - C = 320-370 cm
- Environment identity is already encoded framewise in NWB `environment` and matches the within-session switch structure (for example, trial 30 is the first switched trial on switch days).
- Sparse reward timestamps can be aligned cleanly to framewise behavior; across the full dataset, this yields an omission fraction of `0.1535`, consistent with the paper's "~15%".
- `trial number` is useful as a contextual label but not as the authoritative trial-boundary definition because it includes tunnel fragments before/after on-track laps.
- The task-relevant neural signal for conversion is curated deconvolved activity, not raw fluorescence and not recomputed dF/F, because the NWB export already contains the deconvolved trace used by the reference decoder analyses.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane0/data` + `iscell[:,0]` | `neural` | Trim to behavior length if needed, keep curated cells only, transpose to `(n_neurons, n_timepoints)`, slice per trial from `trial_start` to `teleport` | `glmUtils.get_timeseries_data`, `utilities.multi_anim_sess`, `preprocessing.dff` | Use deconvolved activity directly because it matches the paper's decoder signal. |
| Behavior timestamps (`position/timestamps`) | `input[0]` (`time_from_trial_start_sec`) | For each trial, subtract the trial start timestamp to get elapsed seconds per sample | Reference `sess.vr_data['time']` usage throughout repo | Time-varying continuous input. |
| `environment` | `input[1]` (`environment_type`) | Map valid values `0 -> ENV1`, `1 -> ENV2`; repeat over trial timepoints | `behavior.get_trial_types` / `morph` concept | Constant within a trial; some sessions switch environment after trial 30. |
| `trial number` | `input[2]` (`trial_number`) | Use per-session trial index (0-based), repeated over timepoints | `glmUtils.get_timeseries_data` writes 0-based `trial_ids` | Use actual trial index as continuous context input. |
| Previous trial reward outcome | `input[3]` (`previous_trial_outcome`) | Derive reward outcome for each trial from aligned reward events, then shift by one trial; set first trial to 0 | Consistent with paper's rewarded/omission logic | Repeat over timepoints. |
| Position + inferred reward-zone interval | `output[0]` (`distance_to_reward_zone`) | Signed distance to nearest point in current reward-zone interval; negative before zone, 0 inside zone, positive after zone; discretize to 7 bins per task spec | `behavior.get_reward_zones`; methods reward-zone coordinates | Time-varying categorical output. |
| `position` | `output[1]` (`absolute_position`) | Discretize 0-450 cm into 5 equal 90 cm bins | Methods: 450 cm track | Time-varying categorical output. |
| `speed` | `output[2]` (`speed`) | Discretize into `<2`, `2-10`, `10-20`, `20-40`, `>40` cm/s bins | Methods mention speed thresholding and framewise speed | Time-varying categorical output. |
| `lick` | `output[3]` (`lick`) | Apply lick QC, then binarize cumulative counts as `>0 -> 1` | `glmUtils.get_timeseries_data`, methods licking QC | Time-varying categorical output. |
| Scene-derived reward-zone label | `output[4]` (`reward_zone_location`) | Map `A/B/C -> 0/1/2`, repeat across timepoints | `behavior.get_reward_zones` | Per-trial categorical output repeated across timepoints for consistent array shapes. |
| Sparse `Reward` event times within trial | `output[5]` (`reward_outcome`) | `1` if any reward event falls in trial, else `0`; repeat across timepoints | `behavior.get_trial_types`; methods omission definition | Per-trial categorical output repeated across timepoints. |
| NWB subject id | `subjects`, `subject_idx` | Normalize to strings like `m3`, `m11`, ... | Session metadata | One session per NWB file. |
| Imaging plane location | `brain_regions`, `brain_region_idx` | All neurons map to single region `hippocampus, CA1` | NWB imaging-plane metadata | Plane depth info can go into metadata if useful; target `brain_regions` stays region-level. |

### Key Decisions
1. **Use all curated neurons, not only place-cell or RR/TR/non-RR subsets**: The user-specified decoder is different from the paper's RR-position decoder and needs general-purpose neural inputs for all requested outputs. Manual Suite2p curation is the reference neuron-quality filter that remains applicable.
2. **Use curated deconvolved activity rather than recomputing dF/F**: The paper's decoder and many downstream analyses use deconvolved events, and the NWB export already contains that signal. Recomputing dF/F from fluorescence/neuropil would add avoidable mismatches.
3. **Slice trials from `trial_start` to `teleport`**: This matches the reference code's `trial_start_inds` to `teleport_inds` logic and excludes the variable-length tunnel/teleport period from the standard on-track trial representation.
4. **Trust behavior timestamps for sample spacing**: Multi-plane sessions have misleading 31 Hz ophys metadata, while the behavior timestamps and paper both support an effective ~15.5 Hz sample grid per plane.
5. **Represent all inputs and outputs as time-varying arrays**: Per-trial variables will be repeated across the trial so every trial has a consistent `(n_features, n_timepoints)` structure, which is simpler and less error-prone for downstream decoding.
6. **Infer reward-zone identity from scene metadata instead of framewise `reward_zone` counts**: The framewise `reward_zone` series indicates occupancy/entry events, not the zone identity itself; the scene metadata and reference code provide the intended A/B/C mapping.
7. **Apply lick sensor QC at the trial level and drop invalid trials if needed**: Because lick is one of the decoder targets, trials with known sensor artifacts should not be preserved as if they were valid. I will first try excluding those trials; if this causes undesirable data loss or structural issues, I will revisit with explicit documentation.
8. **Ignore label-only tunnel fragments that lack a `trial_start`**: Complete on-track trials are defined by `trial_start` to `teleport` pairs. This automatically resolves the `sub-m11_ses-03` extra-label edge case without any ad hoc repair.

### Planned Sanity Checks
- [ ] Check 1
- [ ] Check 2
- [ ] Compare one converted neural trial against the raw NWB `Deconvolved` slice after curated-cell filtering using `np.allclose()`.
- [ ] Compare one converted position / speed / lick trial against raw NWB behavior arrays on the same frame indices using `np.allclose()`.
- [ ] Compare per-trial reward outcome derived from sparse `Reward` timestamps against the presence/absence of aligned reward events in three raw trials.
- [ ] Compare inferred reward-zone labels and reward-zone distance bins against scene metadata plus raw position for three raw trials.
- [ ] Verify dataset-level counts after conversion against raw NWB statistics: 11 subjects, 152 sessions, 138,678 curated cells, ~15.35% omission trials.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Wrote `/app/convert_data.py` with CLI:
  - `python -u /app/convert_data.py <outpicklefile>`
  - `--full`
  - `--sample`
  - `--show-processing`
- Loader uses direct `h5py` access instead of `pynwb` for speed and lower overhead.
- Multi-plane sessions are handled by reading every deconvolved response series (`plane0`, `plane1`, ...), applying the `iscell` mask within each series' ROI table region, then concatenating curated neurons across planes.
- Trial extraction uses `trial_start` to `teleport` pairs, which correctly excludes the tunnel / teleport period and automatically ignores trailing label-only fragments in `trial number`.
- Reward outcomes are reconstructed from sparse NWB `Reward` timestamps aligned onto the behavior timestamp grid.
- Per-trial variables are repeated across time so every trial has a consistent time-varying input/output matrix.
- Processing plots are saved as `processing_<session_id>.png` for up to 2 sessions.
- The script prints per-session timing and summary counts.

Code inefficiencies identified:
- Reading full raw segmentation tables when only response-series ROIs are used in multi-plane sessions would be wasteful or incorrect.
- Loading all sessions into memory at once would be unnecessary.

Code speedups added:
- Process sessions sequentially and release each session before moving to the next.
- Use raw HDF5 slicing and load only the curated deconvolved traces actually used.
- Avoid expensive per-trial fancy indexing into HDF5 by loading each session's curated neural matrix once, then slicing in-memory.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 746 |
| Neurons / session | [155, 591] |
| Subjects | 2 |
| Sessions / subject | 1, 1 |
| Trials (total) | 160 |
| Trials / session | [80, 80] |
| `time_from_trial_start_sec` range | [0.0, 30.3] |
| `environment_type` range | [0.0, 1.0] |
| `trial_number` range | [0.0, 79.0] |
| `previous_trial_outcome` range | [0.0, 1.0] |
| `distance_to_reward_zone` distribution | [0.222, 0.095, 0.040, 0.223, 0.021, 0.087, 0.313] |
| `absolute_position` distribution | [0.218, 0.194, 0.265, 0.175, 0.148] |
| `speed` distribution | [0.062, 0.059, 0.086, 0.375, 0.417] |
| `lick` distribution | [0.808, 0.192] |
| `reward_zone_location` distribution | [0.269, 0.731, 0.000] |
| `reward_outcome` distribution | [0.142, 0.858] |

### Processing Plots Review
- Generated:
  - `processing_sub-m11_ses-03.png`
  - `processing_sub-m17_ses-01.png`
- No structural anomalies were detected in the underlying data checks that feed the plots:
  - trial starts align with position entering the track,
  - trial ends align with teleport markers,
  - multi-plane session data were trimmed to the behavior length and remained time-aligned,
  - discretized outputs span the expected class ranges.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Direct `h5py` loading + per-session processing | Avoids heavy NWB object construction and whole-dataset memory duplication |
| Load curated traces only | Reduces I/O and memory substantially relative to raw ROI tables |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion (2 sessions) | 2.30 s / 2 sessions = 1.15 s/session average on sampled sessions | Raw session-count scaling: ~175 s for 152 sessions |
| Cost-weighted full estimate | Using total `(timepoints x curated neurons)` ratio of 202.48x sample workload | ~466 s total (~7.8 min) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `distance_to_reward_zone` | 0.5006 | 0.3886 |
| `absolute_position` | 0.5790 | 0.5260 |
| `speed` | 0.5119 | 0.4232 |
| `lick` | 0.6981 | 0.6699 |
| `reward_zone_location` | 0.9788 | 0.9877 |
| `reward_outcome` | 0.6729 | 0.6320 |

- Loss decreased steadily from `45.17` at epoch 1 to `0.82` by epoch 200.
- Every output validated above uniform-chance performance on the sample dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9.1G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not explicitly stated | Uses curated neurons from session objects | 138,678 curated cells | 138,678 curated cells | Yes |
| Mean neurons/session | Range stated as 155-2172/session | Curated neurons only | 912.36 mean; 155-2341 range | 912.36 mean; 155-2341 range | Raw match; release max exceeds paper max |
| Subjects | 11 switch-task mice | 11 switch-task mice in task subset | 11 | 11 | Yes |
| Sessions | Expected 152 in released switch-task NWB bundle (14 days x 11 mice, minus 2 non-imaged m11 days) | 152 task sessions when excluding exp_day0 running sessions and missing m11 day1-2 | 152 | 152 | Yes |
| Trials (total) | Mean only, not exact total; licking QC context mentions 12,376 raw imaged trials | No exact total in repo | 12,216 complete trials from `trial_start`; 12,147 after lick-QC exclusion | 12,147 | Yes relative to intended QC filtering |
| Trials/session (mean) | 80.5 +/- 7.4 | No exact mean in repo | 80.37 raw; 79.91 after lick-QC exclusion | 79.91 | Yes, close to paper |
| `time_from_trial_start_sec` range | Variable; trial-aligned framewise data at ~15.5 Hz | Trial-aligned framewise data | [0.0, 216.5] after slicing complete trials | [0.0, 216.5] | Yes |
| `environment_type` range | ENV1 / ENV2 | `morph` 0 / 1 | [0, 1] | [0, 1] | Yes |
| `previous_trial_outcome` range | Omission / rewarded | Rewarded vs omission logic in code | [0, 1] | [0, 1] | Yes |
| `reward_zone_location` distribution | Balanced by design across A/B/C permutations | Scene metadata encodes A/B/C by session | Approx. balanced across released sessions | [0.332, 0.336, 0.332] | Yes |
| `reward_outcome` distribution | ~15% omissions / ~85% rewarded | Reward/omission logic used throughout repo | Raw omission fraction 0.1535 before lick-QC filtering | [0.158, 0.842] | Yes, within expectation |
| `distance_to_reward_zone` distribution | Not explicitly stated | Consistent with track and reward geometry | Derived from raw position + scene metadata | [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243] | Yes |

- Full conversion runtime: 53.79 s for all 152 sessions.
- 69 trials were intentionally excluded by lick-sensor QC, leaving 12,147 converted trials.
- 10 sessions had a one-sample neural/behavior length mismatch in the NWB files; these were trimmed to the behavior length before trial extraction.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reported no errors and no warnings.
2. **Neural sanity checks from raw NWB with `np.allclose()`**:
   - `m11`, session `03`, kept trial `0`: converted neural matrix exactly matched the raw curated deconvolved slice.
   - `m17`, session `01`, kept trial `0`: converted neural matrix exactly matched the concatenated curated multi-plane raw slice.
   - `m18`, session `03`, kept trial `10`: converted neural matrix exactly matched the concatenated curated multi-plane raw slice.
3. **Input sanity checks from raw NWB with `np.allclose()`**:
   - For the same three raw trials above, converted `time_from_trial_start_sec`, `environment_type`, `trial_number`, and `previous_trial_outcome` all exactly matched independently recomputed raw values.
4. **Output sanity checks from raw NWB with `np.allclose()`**:
   - For the same three raw trials above, converted `distance_to_reward_zone`, `absolute_position`, `speed`, `lick`, `reward_zone_location`, and `reward_outcome` all exactly matched independently recomputed raw values.
5. **Reference code comparison**:
   - Data loading: reference uses prebuilt `sess` objects; conversion uses NWB export but recovers the same core aligned streams.
   - Neuron filtering: reference uses manual Suite2p curation; conversion uses `iscell[:,0] == 1`.
   - Temporal alignment: reference uses `trial_start_inds` to `teleport_inds`; conversion uses NWB `trial_start` to `teleport`.
   - Binning: reference uses 10 cm track structure and ~15.5 Hz framewise sampling; conversion preserves framewise data and task-specified output binning.
   - Inputs/outputs: conversion differs from the paper's RR-position decoder only where required by the user-specified decoder task.
6. **Key statistics comparison**:
   - Raw curated-cell count matched converted exactly: `138,678`.
   - Raw complete-trial count (`trial_start` markers) was `12,216`; converted kept-trial count was `12,147`, exactly explained by `69` lick-QC trial exclusions.
   - Converted omission fraction was `0.1581`, close to the paper's reported ~15% omissions.
   - Subject/session counts matched the released NWB bundle and task subset of the reference code exactly.
7. **Edge-case checks**:
   - Multi-plane ROI response series required per-plane ROI table region handling; fixed in loader.
   - 10 sessions had neural arrays one sample longer than behavior arrays; trimming resolved these without downstream warnings.
   - `trial number` contains tunnel fragments and cannot define complete trials alone; `trial_start` / `teleport` pairs are authoritative.
   - `sub-m11_ses-03` extra `trial number` label was confirmed to be an incomplete trailing tunnel fragment, not a missing start marker.

### Issues Found and Resolved
- **Issue**: Initial loader assumed a single deconvolved response series and 1:1 correspondence between response columns and the full segmentation table.
  **Resolution**: Updated loader to iterate across all deconvolved plane series, apply the `iscell` mask within each series' ROI table region, and concatenate curated neurons across planes.
- **Issue**: Initial interpretation of `sub-m11_ses-03` suggested a missing first trial start.
  **Resolution**: Direct raw inspection showed the anomaly was actually a trailing incomplete tunnel fragment labeled as trial `80`; the trial-boundary logic was kept as `trial_start` to `teleport`, which already handles this correctly.
- **Issue**: Multi-plane NWB files expose `rate=31.015625 Hz` on the neural series, inconsistent with the effective per-plane time grid.
  **Resolution**: Conversion uses behavior timestamps, which agree with the paper's effective ~15.5 Hz per plane.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes
- Full command completed successfully on GPU:
  - `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`
- Decoder sample plots generated by the validator:
  - `sample_trials.png`
  - `predictions.png`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `distance_to_reward_zone` | 0.4572 | 0.4146 | 2.90x uniform chance; strongest time-varying spatial target. |
| `absolute_position` | 0.5301 | 0.5024 | 2.51x chance; consistent with strong position coding. |
| `speed` | 0.4401 | 0.4203 | 2.10x chance; stable train/validation match. |
| `lick` | 0.6161 | 0.6063 | Above chance with minimal overfitting; sparse pointwise event target. |
| `reward_zone_location` | 0.8428 | 0.8122 | 2.44x chance; easy per-trial context variable. |
| `reward_outcome` | 0.5158 | 0.5048 | Near chance; see Step 12 investigation. |

- Loss decreased monotonically from `188.9175` at epoch 1 to `1.1957` at epoch 200.
- Test loss at completion: `1.0608`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `distance_to_reward_zone` | 0.4146 validation balanced accuracy | Closest analogue to the paper's RR-position decoder. Paper reports RR-position decoding above shuffle, especially for RR cells, rather than a balanced-accuracy percentage. Strongly above chance here is consistent with preserved position/reward alignment. |
| `absolute_position` | 0.5024 | Paper does not report this exact decoder, but hippocampal CA1 position decoding should be robust; 2.51x chance is consistent with correct spatial alignment. |
| `speed` | 0.4203 | Paper does not report a speed decoder. Above-chance performance is expected if framewise behavior is aligned to neural activity. |
| `lick` | 0.6063 | Paper does not report a lick decoder. Above-chance but modest performance is plausible because lick is sparse and instantaneous. |
| `reward_zone_location` | 0.8122 | Paper does not report this exact decoder. High accuracy is expected because reward-zone identity is stable within a trial and strongly tied to session context. |
| `reward_outcome` | 0.5048 | Paper does not report reward-outcome decoding. Because omissions are randomly assigned on ~15% of trials, near-chance accuracy under trial-start alignment is biologically plausible after label verification. |

- **Check 1: Accuracy vs chance**
  - `distance_to_reward_zone`: `0.4146` vs chance `0.1429` (`2.90x` chance)
  - `absolute_position`: `0.5024` vs chance `0.2000` (`2.51x` chance)
  - `speed`: `0.4203` vs chance `0.2000` (`2.10x` chance)
  - `lick`: `0.6063` vs chance `0.5000` (`1.21x` chance)
  - `reward_zone_location`: `0.8122` vs chance `0.3333` (`2.44x` chance)
  - `reward_outcome`: `0.5048` vs chance `0.5000` (`1.01x` chance)
- Outputs below `1.5x` chance were `lick` and `reward_outcome`, so I investigated both for conversion bugs.
- `lick` investigation:
  - Raw-vs-converted `np.allclose()` sanity checks had already shown exact agreement for three independently recomputed raw trials.
  - The target is sparse (`22.3%` positive frames) and pointwise, so modest but above-chance decoding is expected.
  - Train/validation gap was only `1.016x`, which argues against leakage or overfitting artifacts.
- `reward_outcome` investigation:
  - Raw reward timestamps were re-checked directly from NWB files for three concrete converted trials:
    - `m11`, `ses-03`, kept trial `0`: raw `1`, converted `1`
    - `m11`, `ses-03`, kept trial `1`: raw `0`, converted `0`
    - `m18`, `ses-03`, kept trial `10`: raw `1`, converted `1`
  - Broader raw-vs-converted `np.allclose()` checks in Step 10 already showed exact equality for reward-outcome vectors on three raw trials.
  - The paper states reward omission was random on approximately `15%` of trials. Under the user-required alignment to trial start, this makes current-trial reward outcome intrinsically hard to predict from most of the trial, so near-chance performance is plausible and not evidence of a conversion bug.
  - Train/validation gap was only `1.022x`, so there is no sign of leakage or instability.
- **Check 2: Accuracy comparison to paper**
  - The paper's decoder is not the same task: it predicts circular reward-relative position using RR/TR/non-RR cell subsets and evaluates cosine decode score relative to shuffle, not balanced accuracy on categorical variables.
  - The paper therefore does not provide directly comparable balanced-accuracy values for any of this task's six outputs.
  - The closest qualitative comparison is that position-like variables should decode well if alignment and processing are correct. Both `distance_to_reward_zone` and `absolute_position` are strongly above chance, matching that expectation.
  - The paper's main qualitative result is that reward-relative position decoding is above shuffle, especially for reward-relative cells across reward switches. Nothing in the converted dataset or full decoder run contradicts that qualitative expectation.
- **Check 3: Train vs validation gap**
  - `distance_to_reward_zone`: `0.4572 / 0.4146 = 1.10x`
  - `absolute_position`: `0.5301 / 0.5024 = 1.05x`
  - `speed`: `0.4401 / 0.4203 = 1.05x`
  - `lick`: `0.6161 / 0.6063 = 1.02x`
  - `reward_zone_location`: `0.8428 / 0.8122 = 1.04x`
  - `reward_outcome`: `0.5158 / 0.5048 = 1.02x`
  - No output exceeded the `1.5x` train/validation-gap threshold, so I found no evidence of overfitting or leakage.
- **Conclusion**
  - I did not identify any Step 12 issue that justified changing `convert_data.py`.
  - The two weaker outputs are weak for task-design reasons (`lick` sparsity; `reward_outcome` random omissions under trial-start alignment), not because of a formatting or alignment error.

### Issues Found and Resolved
- No new conversion issues were found in Step 12, so no additional code changes were required after the Step 11 full decoder run.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

- Created `/app/README.md` with dataset description, processing summary, loading instructions, and validation command lines.
- Created `/app/cache/README_CACHE.md` documenting the cache directory and explaining that no standalone investigation scripts were retained on disk.
- Verified presence of all required deliverables:
  - `/app/CONVERSION_NOTES.md`
  - `/app/convert_data.py`
  - `/app/converted_data.pkl`
  - `/app/sample_data.pkl`
  - `/app/README.md`
  - `/app/train_decoder_full_out.txt`
  - `/app/conversion_sample_out.txt`
  - `/app/verification_sample_out.txt`
  - `/app/train_decoder_sample_out.txt`
  - `/app/conversion_full_out.txt`
  - `/app/verification_full_out.txt`
