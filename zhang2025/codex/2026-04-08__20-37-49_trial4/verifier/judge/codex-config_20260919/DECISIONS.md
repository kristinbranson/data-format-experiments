# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the canonical Brain-Wide Map release CSV, deduplicates it to session rows, groups its probe rows by `eid`, and processes every listed session (unless sample mode is selected). It locates revisioned ALF files first in the shipped read-only ONE cache and then in a writable cache, downloading a missing dataset through ONE. Trials, wheel, camera, spike, cluster, and channel arrays are opened directly from the resolved paths. Full mode uses per-session worker processes and cacheable `SessionRecord`s; failed/incompatible sessions are omitted.

ii.
```python
bwm = pd.read_csv(RELEASE_CSV, index_col=0)
session_rows = bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
probe_rows = {eid: grp[["pid", "probe_name"]] for eid, grp in bwm.groupby("eid", sort=False)}
```
```python
for root in (READONLY_CACHE, WRITABLE_CACHE):
    matches = sorted((root / rel).glob(relative_glob))
one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
```

iii. The notes say the 459-session release list is canonical and direct ALF loading was used because optional dependencies needed by the reference helper stack were unavailable. Revision-aware lookup, a writable download fallback, parallelism, and session caching were intended to make the complete conversion robust and restartable.

## 1-b. How are the data split into subjects?

i. Subject identity is taken directly from the release CSV. During final assembly, subjects are registered on first occurrence and each session receives the corresponding integer `subject_idx`.

ii.
```python
if rec.subject not in subject_to_idx:
    subject_to_idx[rec.subject] = len(subjects)
    subjects.append(rec.subject)
subject_idx.append(subject_to_idx[rec.subject])
```

iii. The notes treat the release metadata's subject label as the authoritative identifier and report 136 retained subjects after session exclusions.

## 1-c. How are the data split into sessions?

i. The release CSV's unique `eid` values define sessions. Each successful `process_session` call creates one `SessionRecord`, and each record becomes one element of the top-level `neural`, `input`, and `output` lists.

ii.
```python
session_rows = bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
```
```python
for rec in session_records:
    neural.append(rec.neural)
    input_data.append(rec.input_trials)
    output.append(session_output)
```

iii. The release already supplies unique session IDs, so the agent preserves that native unit and merges probes only within a session.

## 1-d. How are the data split into trials?

i. Each row of `_ibl_trials.table.pqt` defines a trial. Stimulus-centered intervals are built from each row's `stimOn_times`; neural and behavioral streams are sliced/interpolated over those intervals. Only indices passing the combined mask are appended as per-trial arrays.

ii.
```python
intervals = np.vstack([trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
                       trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]]).T
```
```python
valid_idx = np.flatnonzero(combined_mask)
for idx in valid_idx:
    neural_trial = compact_neural_trial(binned_spikes[idx].T)
```

iii. The trial table is already one-row-per-trial; the notes say shared validity across neural, wheel, and whisker streams was enforced.

## 1-e. How are trials filtered based on quality controls?

i. The base mask excludes missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`; reaction times below 80 ms or above 2 s; trials longer than 10 s from go cue to feedback; and no-choice trials. It retains unbiased trials. This is intersected with masks requiring finite, complete wheel and whisker traces whose samples cover both window edges within 20 ms. A session must retain at least two trials.

ii.
```python
trial_mask = build_trial_mask(trials_df)
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
if len(valid_idx) < 2:
    raise RuntimeError(...)
```
```python
query += f" | (feedback_times - goCue_times > {max_trial_len})"
query += " | (choice == 0)"
```

iii. The notes attribute the event, reaction-time, no-choice, and 10 s rules to the papers/reference code. Coverage filtering is justified because both time-varying decoder outputs need a complete common window. Fifteen sessions were ultimately excluded for missing/inadequate whisker or overlapping behavioral data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each probe's `spikes.times.npy` and `spikes.clusters.npy`. Cluster quality and anatomical association are derived from cluster channels/depths/metrics and channel CCF region IDs; all retained probes in a session are merged.

ii.
```python
spike_times = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
cluster_metrics = pd.read_parquet(cluster_metrics_path)
channel_region_ids = np.load(channel_regions_path).astype(np.int64)
```

iii. The notes say direct ALF fields reproduce the reference loader's required information while avoiding unavailable optional dependencies.

## 2-b. How is the `neural` data processed?

i. Probes are merged after remapping their retained clusters to consecutive session-wide IDs and sorting all spikes by time. Spikes are counted in 100 consecutive 20 ms bins from -0.5 to 1.5 s around stimulus onset. The resulting per-trial neuron-by-time count matrices are stored as `uint8`; they are not divided by bin width and therefore remain spike counts rather than firing rates.

ii.
```python
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr,
                                         xbin=binsize, xlim=[t_beg, t_end])
binned_spikes[interval_idx, idxs_tmp, :] = binned_tmp[:, :n_bins]
```
```python
return neural_trial.astype(np.uint8, copy=False)
```

iii. The notes explicitly plan “bin counts” and justify `uint8` as necessary to reduce the full pickle to about 3.1 GB and avoid training-memory problems. They do not justify the departure from the reference's conversion of counts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters with missing/low quality labels are removed, retaining only `label >= 1`. Probes with no such units are skipped. Regions are mapped to Beryl, but the agent does not remove units mapped to `void` (or `root`).

ii.
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
spike_keep = np.isin(spike_clusters, good_cluster_ids)
```
```python
cluster_regions_beryl = brain_regions.acronym2acronym(
    clusters.loc[cluster_ids, "acronym"].to_numpy(), mapping="Beryl")
```

iii. The agent found that `label >= 1` gives exactly the paper's 75,708 well-isolated neurons and chose that paper-level QC despite a reference helper path that can load all clusters. It states that Beryl labels are retained, but does not discuss or implement the reference solution's removal of `void` units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural interval begins at `stimOn_times - 0.5` s and ends at `stimOn_times + 1.5` s, using timestamps already expressed on the session clock.

ii.
```python
ALIGN_EVENT = "stimOn_times"
interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
```

iii. The notes choose stimulus onset because it is explicitly required by the decoder task and agrees with the executable reference pipeline.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural spikes are binned directly into 20 ms bins, producing 100 bins over the 2 s interval. There is no subsequent temporal rebinning or smoothing.

ii.
```python
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
```

iii. The agent chose 20 ms to match the executable reference and papers and uses it consistently for neural binning and behavior sampling.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a fixed grid defined from the chosen `stimOn_times` alignment window and bin size rather than measured from another raw column. The same grid is repeated for every trial.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS,
                        dtype=np.float32)
```

iii. The notes describe this as the stimulus-aligned interpolation grid and report its range as approximately -0.48 to 1.5 s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent generates 100 equally spaced bin-end values from -0.48 through 1.50 s and stores them as `float16` in each input matrix. It does not use bin centers.

ii.
```python
TIME_GRID = np.linspace(-0.5 + 0.02, 1.5, 100, dtype=np.float32)
input_trial = np.vstack([TIME_GRID, np.full(N_BINS, trial_num)]).astype(np.float16)
```

iii. The notes deliberately identify the grid as bin-end times, though they offer no substantive rationale for choosing ends instead of the human reference's centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Both have 100 positions over the same nominal [-0.5, 1.5] s window, but the labels are bin ends (-0.48 to 1.50) whereas neural counts represent intervals beginning at -0.5. Thus array indices correspond, although timestamps are shifted +10 ms from the reference's bin-center convention.

ii.
```python
xlim=[t_beg, t_end]
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS)
```

iii. The agent says the executable reference uses this interpolation grid and its plots showed no apparent misalignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from sequential values of the trials table's `probabilityLeft`; a change or a value after missing data starts a new block.

ii.
```python
block_trial_number = compute_trial_number_in_block(
    trials_df["probabilityLeft"].to_numpy())
```

iii. The notes describe recovering block structure from the prior probability because it is constant within blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The full unfiltered trial sequence is scanned. A new block starts at 1 and same-prior trials increment the counter, so numbering is one-based. Missing prior values reset the state and receive NaN. The retained trial's number is repeated across all 100 time bins.

ii.
```python
if prev is None or np.isnan(prev) or not np.isclose(prev, value):
    count = 1
else:
    count += 1
out[idx] = count
```

iii. Computing before filtering preserves the animal's actual position in the block. The notes report the resulting range beginning at 1, but do not justify differing from the reference's zero-based count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the trials table's `choice` field.

ii.
```python
choice_code = map_choice(trials_df.iloc[idx]["choice"])
```

iii. The agent uses the established IBL choice convention and excludes no-response trials so the requested output remains binary.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Raw `+1` is mapped to 0 (left), raw `-1` to 1 (right), and the per-trial code is repeated through 100 bins as `uint8`. Raw zero choices were removed earlier.

ii.
```python
if np.isclose(choice_value, 1.0): return 0
if np.isclose(choice_value, -1.0): return 1
np.full(N_BINS, choice_code, dtype=np.uint8)
```

iii. This is the requested categorical mapping; repetition provides a uniform time-varying output shape.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trials table's `probabilityLeft` value.

ii.
```python
prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
```

iii. The notes identify this field as the task's block prior and the direct source requested by the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 map to categories 0, 1, and 2 respectively, then the per-trial category is repeated through 100 bins.

ii.
```python
if np.isclose(prob_left, 0.2): return 0
if np.isclose(prob_left, 0.5): return 1
if np.isclose(prob_left, 0.8): return 2
```

iii. The mapping exactly follows the decoder specification, with time repetition used for a uniform target matrix.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps = np.load(ts_path)
position = np.load(pos_path)
```

iii. The notes say this matches the reference behavior source.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1000 Hz, filtered/differentiated with `velocity_filtered`, converted to absolute velocity, linearly interpolated/extrapolated to the 100 trial grid points, stored temporarily in `float16`, and later discretized.

ii.
```python
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
```
```python
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The agent states that absolute wheel velocity and the loader-equivalent interpolation/filtering match the reference. The shared grid supports direct neural/behavior alignment.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained wheel samples from all retained sessions are concatenated. Dataset-wide 1/3 and 2/3 quantiles define two global thresholds; `np.digitize` maps values to low/mid/high. Degenerate thresholds receive fallback evenly spaced or epsilon-adjusted edges.

ii.
```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records])
wheel_edges = safe_quantile_edges(wheel_all)
discretize(wheel_vals, wheel_edges)
```

iii. The notes explicitly choose global tertiles so category semantics and thresholds are consistent across sessions. This differs from the human reference's per-session tertiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each trial, wheel samples inside the same stimulus-centered interval are interpolated at 20 ms-spaced points from onset-0.48 through onset+1.5 s. These indices match the 100 neural bins, but represent bin ends rather than bin centers.

ii.
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE,
                       interval_ends[interval_idx], n_bins)
```

iii. The notes say all streams share the session clock and the same stimulus-aligned grid; plots were used as a visual check.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`, trying the left camera first and falling back to the right.

ii.
```python
sides = [("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
         ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy")]
```

iii. The agent says left-first/right-fallback follows the reference and paper.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released ROI motion-energy samples are cast to float, checked for finite/full-window coverage, and linearly interpolated/extrapolated onto the 100 trial grid points. No filtering or normalization is applied before global discretization.

ii.
```python
values = np.load(energy_path).astype(np.float32)
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False)
```

iii. The notes say the released trace is the reference source and no extra signal processing is needed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Every retained whisker sample across the entire converted dataset is concatenated. Global 1/3 and 2/3 quantile edges then produce low/mid/high categories, with safeguards for equal edges.

ii.
```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records])
whisker_edges = safe_quantile_edges(whisker_all)
discretize(whisker_vals, whisker_edges)
```

iii. The agent chose global tertiles for cross-session semantic consistency, rather than the reference's session-specific equal-sized classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera samples in the stimulus-centered window are evaluated at the same 100 bin-end grid points used for wheel and inputs. The resulting indices line up with the neural matrix, with the same +10 ms center-versus-end discrepancy.

ii.
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE,
                       interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The notes rely on shared session-clock timestamps and diagnostic plots to justify the alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing expected trial columns or cluster metadata raise explicit errors. Missing datasets are downloaded if possible. Trials with missing required events, NaNs in behavioral segments, inadequate edge coverage, or invalid interpolation are removed. A missing camera side falls back to the other side; an empty good-unit probe is skipped. Sessions with no good probe, missing both cameras, or fewer than two valid trials fail and are omitted in the full workflow. Atomic temporary files protect cached and final pickles.

ii.
```python
if missing:
    raise ValueError(f"Missing expected trial columns: {missing}")
```
```python
except Exception as exc:
    errors.append(f"{side}: {exc}")
```
```python
if len(valid_idx) < 2:
    raise RuntimeError(...)
```

iii. The notes document 14 sessions without whisker traces and one without valid overlap, and consider exclusion necessary because whisker is a required output. They also document fixing the empty-good-unit-probe edge case.

## 10-a. What are the most time-consuming steps of the code?

i. Raw dataset resolution/loading, probe spike binning, behavior interpolation, and the full conversion/serialization dominate. The implementation parallelizes sessions and caches compact per-session records; the observed full cache fill took about 429 seconds.

ii.
```python
with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
    futures = {...}
```
```python
binned_tmp, _, cluster_idxs = bincount2D(...)
```

iii. The notes emphasize that expensive raw loading should happen once, and that session cache reuse and four workers reduce retries and wall time.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. `compute_trial_number_in_block`, the per-interval loops in neural binning and behavior interpolation, the per-valid-trial construction loop, and final per-trial output assembly could potentially be vectorized or batched. Probe and session loops are structurally appropriate because their arrays differ in size and sessions are already parallelized.

ii.
```python
for idx, value in enumerate(probability_left): ...
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(...): ...
for interval_idx, (ib, ie) in enumerate(...): ...
for idx in valid_idx: ...
```

iii. The notes do not explicitly inventory vectorizable loops; their efficiency work instead focuses on coarse session-level parallelism, compact dtypes, and caching.

## 10-c. What processing does the code repeat multiple times?

i. Each behavior stream independently runs the same coverage/interpolation routine. Trial arrays and outputs are repeatedly allocated per trial. In diagnostic mode some values are copied/cast again. Across fresh runs, dataset lookup and raw processing repeat unless session-cache options are used.

ii.
```python
wheel_values, wheel_mask = get_behavior_per_interval(...)
whisker_values, whisker_mask = get_behavior_per_interval(...)
```
```python
np.full(N_BINS, choice_code, dtype=np.uint8)
np.full(N_BINS, prior_code, dtype=np.uint8)
```

iii. The notes recognize repeated raw work across retries and added cached per-session intermediates specifically to avoid it, but otherwise do not discuss repeated processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads cluster depths and builds a full cluster metadata DataFrame although downstream output needs only QC and region labels. It bins neural data for every raw trial before applying the combined validity mask, so counts for rejected trials are discarded. Continuous wheel/whisker traces are retained in session records only to compute global thresholds and are absent from the final dataset. Optional diagnostics copy substantial raw/interpolated signals solely for plots.

ii.
```python
cluster_depths = np.load(cluster_depths_path).astype(np.float32)
clusters_df["depths"] = cluster_depths
```
```python
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)
valid_idx = np.flatnonzero(combined_mask)
```

iii. The notes justify the continuous intermediate traces because global discretization requires a second assembly stage and justify diagnostics as validation aids, but do not explicitly identify discarded trial binning or unused depth metadata.
