# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `bwm_release.csv`, groups its probe rows by session `eid`, constructs each local ONE-cache path from lab/subject/date/session number, and loads trials, behavior, and every listed probe directly from files. It optionally truncates the session list with `session_limit`.

ii.
```python
release_df = pd.read_csv(BWM_RELEASE_CSV)
for eid, group in release_df.groupby("eid", sort=False):
    session_rows.append({"eid": eid, "subject": first["subject"],
                         "probe_names": list(group["probe_name"])})
session_path = ONE_CACHE_DIR / session["lab"] / "Subjects" / session["subject"] / session["date"] / f"{session['session_number']:03d}"
```

iii. The trajectory says this was chosen as an offline conversion using the release manifest and local cache, avoiding dependency on live ONE/Alyx access while still merging all probes belonging to a session.

## 1-b. How are the data split into subjects?

i. Subject identifiers come directly from the release CSV. After conversion, first-occurrence order defines `subjects`, and each retained session receives an index into that list.

ii.
```python
subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_idx = np.asarray([subject_lookup[s["subject"]] for s in kept_sessions])
```

iii. The agent treated the release manifest's subject field as the authoritative identifier; no filename parsing was needed.

## 1-c. How are the data split into sessions?

i. Rows are grouped by `eid`; each group is one session and supplies its probes. Session order is the first-occurrence order in the release CSV.

ii.
```python
for eid, group in release_df.groupby("eid", sort=False):
    ...
```

iii. The trajectory identifies `eid` as the unique session key and notes that probes in the same session must be merged rather than treated independently.

## 1-d. How are the data split into trials?

i. Each trials-table row is treated as one trial. Retained rows are looped over individually, using `stimOn_times - 0.5` and `stimOn_times + 1.5` as boundaries.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows():
    if not trial_mask[trial_idx]:
        continue
    trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
```

iii. The agent relied on the ALF trials table already being row-per-trial and used the requested stimulus-onset alignment to cut every stream.

## 1-e. How are trials filtered based on quality controls?

i. It requires seven finite trial fields, reaction time in `[0.08, 2.0]` s, trial length at most 10 s, and nonzero choice. Later it drops trials without nearly complete wheel/camera coverage, nonfinite interpolated behavior, or any spike in the window. Sessions with fewer than two surviving trials are dropped.

ii.
```python
for column in required:
    mask &= np.isfinite(trials_df[column].to_numpy())
mask &= rt >= MIN_RT_S
mask &= rt <= MAX_RT_S
mask &= trial_len <= MAX_TRIAL_LEN_S
mask &= trials_df["choice"].to_numpy() != 0
...
if wheel_trial is None or whisker_trial is None: continue
if not np.any(neural_trial): continue
```

iii. The trajectory says the finite-field, RT, no-choice, and 10 s rules were taken from `load_trials_and_mask`/`prepare_data`. The zero-spike exclusion was added after verification warned about all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices derive from `spikes.times.npy` and `spikes.clusters.npy`; cluster metrics, cluster-channel assignments, and channel atlas IDs determine which units and region labels survive.

ii.
```python
spike_times = np.load(spike_times_path)
spike_clusters = np.load(spike_clusters_path)
metrics = pd.read_parquet(metrics_path, columns=["label"])
cluster_channels = np.load(cluster_channels_path)
channel_region_ids = np.load(channel_regions_path)
```

iii. The agent followed the spike-sorting files described in the supplied code and used ancillary arrays only for curation and anatomy.

## 2-b. How is the `neural` data processed?

i. Good units from all probes are renumbered and merged, spikes are time-sorted, and each trial is binned into 100 20-ms bins with `bincount`. The saved values are raw spike counts cast to `float16`; they are not divided by bin width or smoothed.

ii.
```python
flat_idx = clusters[valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
return counts.astype(np.float16)
```

iii. The trajectory interpreted the reference caching routine as using unsmoothed binned spikes and selected `float16` to control the size of the multi-gigabyte output. It also explicitly justified merging probes because they share behavior and are not independent sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters must have `metrics.label >= 1`; clusters assigned to `void` or `root` are removed. Region IDs are converted directly to BrainRegions acronyms (without Beryl remapping). Trials with no spikes across all retained units are also removed.

ii.
```python
keep_mask = metrics["label"].to_numpy() >= 1.0
cluster_acronyms = region_ids_to_acronyms(cluster_region_ids, brain_regions)
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
...
if not np.any(neural_trial): continue
```

iii. The agent described this as retaining only well-isolated units and excluding non-brain/root assignments; the trial-level filter was a late response to verifier warnings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike timestamps are sliced from `stimOn_times - 0.5` (inclusive) to `stimOn_times + 1.5` (exclusive), then floored into bins relative to that start.

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
start_idx = np.searchsorted(spike_times, trial_start, side="left")
bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
```

iii. The trajectory cites the reference configuration `align_time='stimOn_times'` and `time_window=(-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, with 100 bins across two seconds. Spikes are newly histogrammed into these bins; there is no subsequent rebinning or smoothing.

ii.
```python
BIN_SIZE_S = 0.02
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

iii. The agent matched the reference parameters and method-paper description of 2-s trials with 20-ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is analytically generated from the fixed offsets and bin size, with stimulus onset serving only as the alignment event; no sampled raw trace is used.

ii.
```python
time_since_stim = np.linspace(OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS)
```

iii. The agent regarded time since onset as the common trial grid defined by the requested alignment window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It creates 100 evenly spaced values from -0.48 through 1.50 s, inclusive, and stores them as `float32` for every trial.

ii.
```python
time_since_stim = np.linspace(-0.5 + 0.02, 1.5, 100, dtype=np.float32)
```

iii. The trajectory explains this as the behavioral sampling grid, although it is the right edges of neural bins rather than their centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It has the same 100 columns as neural data, but labels them by neural-bin right edges (`-0.48 ... 1.50`) rather than centers (`-0.49 ... 1.49`). Thus dimensions align, while timestamps are shifted 10 ms relative to neural-bin centers.

ii.
```python
input_trial = np.vstack([time_since_stim, np.full(N_BINS, ...)])
```

iii. The agent asserted that the streams shared a 20-ms stimulus-aligned grid, but did not discuss the center-versus-edge mismatch.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived solely from the sequence of `probabilityLeft` values; a changed or nonfinite value starts a new block.

ii.
```python
same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
```

iii. The agent inferred blocks from the prior because the trials table does not provide a separate block identifier.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Trials are counted before filtering, starting at 1 in each block and incrementing while the prior is unchanged. The resulting scalar is repeated across all 100 time bins.

ii.
```python
count = count + 1 if same_block else 1
numbers[idx] = float(count)
...
np.full(N_BINS, trial_number_in_block[trial_idx])
```

iii. Counting before filtering preserves the trial's actual position in the experiment. The trajectory did not justify choosing one-based rather than zero-based numbering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials table's `choice` column.

ii.
```python
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. The agent used the IBL choice encoding documented in the supplied sources.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `+1` is mapped to class 0 (left), `-1` to class 1 (right), and no-choice trials are filtered. The per-trial class is broadcast across 100 bins.

ii.
```python
if np.isclose(value, 1.0): return 0
if np.isclose(value, -1.0): return 1
np.full(N_BINS, choice_class, dtype=np.int8)
```

iii. This directly follows the requested left/right mapping and enables a uniform time-varying output matrix.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. The agent identified this as the task's blockwise prior variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 are mapped to 0, 1, and 2 and broadcast across time.

ii.
```python
mapping = {0.2: 0, 0.5: 1, 0.8: 2}
np.full(N_BINS, prior_class, dtype=np.int8)
```

iii. The mapping exactly follows the decoder-task instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps = np.load(timestamps_path)
position = np.load(position_path)
```

iii. The agent followed the reference wheel loader rather than treating position itself as speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, low-pass filtered/differentiated with an order-8 20-Hz Butterworth operation, converted to absolute velocity, and linearly interpolated per trial at 100 requested times.

ii.
```python
position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity)
y_interp = np.interp(x_interp, trial_times, trial_values)
```

iii. The trajectory says these parameters reproduce `SessionLoader`/IBL's recommended wheel-velocity processing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are the 1/3 and 2/3 quantiles of all retained wheel samples pooled across every session, trial, and time bin. `np.digitize` produces low/medium/high classes.

ii.
```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]])
```

iii. The agent chose global tertiles to make three balanced dataset-wide categories; it recorded the learned edges in metadata.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is interpolated at `trial_start + 0.02 ... trial_end`, i.e. -0.48 through +1.50 s from stimulus onset. This is 10 ms later than neural-bin centers and includes the right endpoint.

ii.
```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins)
y_interp = np.interp(x_interp, trial_times, trial_values)
```

iii. The agent intended these as the same 100 20-ms aligned samples, but did not account for bin centers.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `<left/right>Camera.ROIMotionEnergy.npy` and matching camera timestamps, preferring left and falling back to right.

ii.
```python
for view in ("left", "right"):
    me_path = latest_revision_file(...)
    times_path = latest_revision_file(...)
```

iii. This camera selection mirrors the available release data and the reference's left-first behavior.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Motion energy is loaded without filtering or normalization. If timestamps are longer, their leading excess is dropped; the trace is linearly interpolated to each trial's 100 behavior-grid samples.

ii.
```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
...
y_interp = np.interp(x_interp, trial_times, trial_values)
```

iii. The agent considered released ROI motion energy already processed and only performed temporal resampling and length repair.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Global 1/3 and 2/3 quantiles pooled over all retained sessions/trials/times define the three classes.

ii.
```python
whisker_flat = np.concatenate(whisker_continuous_all)
_, whisker_edges = discretize_three_bins(whisker_flat)
```

iii. As for wheel speed, the agent selected global tertiles to balance the complete converted dataset.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera samples are interpolated at -0.48 through +1.50 s relative to stimulus onset. The output has 100 columns but is shifted 10 ms from neural-bin centers.

ii.
```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. The trajectory described this as sharing the neural time grid, without recognizing the edge/center offset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Latest revision files are selected. Camera timestamp excess is trimmed from the front; shorter timestamps, missing files, shape mismatches, and other per-session exceptions cause the whole session to be logged and skipped. Missing behavioral window coverage drops a trial. Probes with no retained units are ignored; sessions with no good units or fewer than two trials are dropped.

ii.
```python
except Exception as exc:
    dropped_sessions.append((session["eid"], f"error:{type(exc).__name__}:{exc}"))
...
if len(session_neural) < 2: ... continue
```

iii. The agent prioritized producing a complete valid dataset despite isolated missing/corrupt artifacts, and the trajectory reports using verifier warnings to tighten curation.

## 10-a. What are the most time-consuming steps of the code?

i. Conversion is dominated by loading large per-probe spike arrays, sorting/merging them, trial-wise spike binning and behavior interpolation, and finally serializing the roughly 6-GB pickle. The trajectory also found downstream full verification/training expensive, especially loading/summarizing the pickle and per-session SVD initialization, though those are outside `convert_data.py`.

ii.
```python
spike_times = np.load(spike_times_path)
order = np.argsort(spike_times, kind="stable")
for trial_idx, trial_row in trials_df.iterrows(): ...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Runtime observations in the trajectory explicitly identify large-file I/O/writeout and later 439-session model initialization as bottlenecks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loops over trial rows, per-trial spike histogramming/interpolation, cluster acronym conversion, trial-number calculation, and final per-trial output construction could be partly vectorized or batched. Session/probe loops are less naturally vectorizable because files and array sizes differ.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows(): ...
for idx, value in enumerate(values): ...
for static_values, wheel_trial, whisker_trial in zip(...): ...
```

iii. The trajectory focused on correctness and memory stability rather than refactoring these heterogeneous per-trial operations.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches for latest-revision files, performs `searchsorted` and interpolation independently for wheel and whisker on every trial, bins spikes trial by trial, and later walks all retained trials again to discretize outputs after global thresholds are known.

ii.
```python
wheel_trial = interpolate_behavior_trial(...)
whisker_trial = interpolate_behavior_trial(...)
...
for static_values, wheel_trial, whisker_trial in zip(...):
```

iii. The second pass is required by the chosen global-tertile decision; thresholds cannot be known until all continuous traces have been accumulated.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `velocity_filtered` computes acceleration but the caller discards it. The converter collects detailed statistics/drop reasons only for printing or optional JSON. It also retains continuous wheel/whisker traces until global binning, then only categorical versions are saved in the target dataset.

ii.
```python
acc = np.insert(np.diff(vel), 0, 0.0) * fs
velocity, _ = velocity_filtered(...)
...
return data, stats
```

iii. Acceleration is inherited from the reference-style helper API; continuous traces and statistics support global threshold estimation and validation but are not downstream decoder fields.
