# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the BWM release CSV, optionally restricts it with `DATALIMIT_SUBSET.csv`, initializes a local ONE cache only to resolve each `eid` to a session path, and then directly reads parquet/NumPy files. Sessions are processed concurrently and checkpointed before final assembly.

ii.
```python
release = pd.read_csv(BWM_TABLE, index_col=0)
release = release[release["eid"].astype(str).isin(selected)]
One.load_cache(one, CACHE_DIR / "Brainwidemap")
session_path = Path(one.eid2path(str(eid)))
```

iii. The trajectory says the cache contains the full BWM release and that direct file access avoids ambiguities caused by multiple tagged revisions while still using the curated release table.

## 1-b. How are the data split into subjects?

i. Subject names come from the release table. At assembly, unique names are sorted and each session receives an integer `subject_idx`.

ii.
```python
subject=str(rows.iloc[0]["subject"])
subjects = sorted({payload["subject"] for payload in payloads})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
```

iii. The agent treated the release table's subject field as the authoritative identifier; no subject parsing was necessary.

## 1-c. How are the data split into sessions?

i. Rows in the release table are grouped by `eid`; all probes sharing an `eid` form one session.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    probes=tuple(rows["probe_name"].astype(str))
```

iii. The trajectory identifies `eid` as the unique session identifier and notes that probes in the same session share behavior and should be merged.

## 1-d. How are the data split into trials?

i. Each row of `_ibl_trials.table.pqt` is a trial. Retained rows index per-trial neural, input, and output arrays.

ii.
```python
trials = pd.read_parquet(table)
keep_idx = np.flatnonzero(keep)
interval_begins = stimulus_times[keep] + OFF_START
```

iii. The trials table already provides the trial boundaries/events, so the agent did not infer trials from continuous streams.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have required events, a nonzero choice, reaction time from 0.08 to 2 s, go-cue-to-feedback duration at most 10 s, and complete finite wheel and camera coverage. Sessions with fewer than two retained trials are skipped.

ii.
```python
return (trials[needed].notna().all(axis=1).to_numpy()
        & (trials["choice"].to_numpy() != 0)
        & (reaction_time.to_numpy() >= 0.08)
        & (reaction_time.to_numpy() <= 2.0)
        & (trial_duration.to_numpy() <= 10.0))
keep = paper_mask & wheel_good & motion_good
```

iii. The agent explicitly attributed these exclusions to the BWM paper/repository and added stream-coverage checks so every retained target has a full aligned trace.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each probe's `spikes.times.npy` and `spikes.clusters.npy`. Cluster-channel and channel-atlas arrays are used only to attach Beryl region labels.

ii.
```python
times = np.load(times_path, mmap_mode="r")
clusters = np.load(clusters_path, mmap_mode="r")
cluster_atlas_ids = channel_atlas_ids[cluster_channels[used]]
```

iii. The trajectory traced the repository's spike loader and concluded that spike timestamps and cluster assignments are the sufficient raw variables for binning.

## 2-b. How is the `neural` data processed?

i. For every probe and retained trial, spikes are counted per cluster in 100 non-overlapping 20 ms bins. Probe arrays are concatenated by neuron. Counts are stored as `float32`; they are not divided by bin width or smoothed.

ii.
```python
flat = cluster_ids[valid] * N_BINS + spike_bins[valid]
binned[trial].flat[:] = np.bincount(flat, minlength=len(used) * N_BINS)
neural = np.concatenate(probe_bins, axis=1)
```

iii. The agent stated that Zhang et al.'s decoder pipeline uses spike counts and no smoothing, and verified its binning against `bincount2D`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No cluster-quality or anatomical exclusion is applied. Every cluster represented in the spike train is retained, including `root`/`void` regions.

ii.
```python
used = np.unique(clusters)
binned = np.zeros((len(interval_begins), len(used), N_BINS), dtype=np.float32)
```

iii. The agent chose all Kilosort clusters because the supplied methods-paper `prepare_data` path calls the spike loader without a QC argument; metadata records this choice explicitly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural windows begin 0.5 s before `stimOn_times` and end 1.5 s after it. Absolute spike times are converted to bin offsets from each window start.

ii.
```python
interval_begins = stimulus_times[keep] + OFF_START
spike_bins = np.floor((times[lo:hi] - interval_begins[trial]) / BIN_SIZE)
```

iii. The agent followed the reference's two-second stimulus-aligned window and relied on the streams' shared synchronized session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, producing 100 bins over two seconds. Spikes are binned once; no later rebinning occurs.

ii.
```python
BIN_SIZE = 0.02
N_BINS = 100
```

iii. The agent cited the methods paper's 20 ms, 100-step configuration.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the fixed window relative to `trials.stimOn_times`, rather than from another recorded stream.

ii.
```python
stimulus_times = trials["stimOn_times"].to_numpy(dtype=np.float64)
relative_time = OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE
```

iii. The agent regarded stimulus onset as time zero and the requested window/bin width as defining the coordinate.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It constructs `[-0.50, -0.48, ..., 1.48]` seconds, i.e. the left edge of each neural bin, and repeats it for every trial.

ii.
```python
relative_time = OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE
inputs[:, 0, :] = relative_time
```

iii. Metadata expressly calls this the “left edge of each 20 ms neural bin”; the trajectory does not provide a deeper justification for choosing edges rather than centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Each value labels the left edge of the corresponding neural count bin, using the same start and 20 ms step.

ii.
```python
spike_bins = np.floor((times[lo:hi] - interval_begins[trial]) / BIN_SIZE)
inputs[:, 0, :] = relative_time
```

iii. The agent intended a one-to-one shared bin index between input time and neural activity.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full, unfiltered `probabilityLeft` sequence; a value change marks a new block.

ii.
```python
probability_left = trials["probabilityLeft"].to_numpy(dtype=np.float64)
trial_in_block = _trial_number_in_block(probability_left)
```

iii. The trials table has no separate block identifier, so the agent inferred blocks from the prior held constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based run length is computed before filtering; it increments when the prior matches the preceding trial and resets to zero when it changes. The scalar is broadcast across time.

ii.
```python
out[trial] = out[trial - 1] + 1 if probability_left[trial] == probability_left[trial - 1] else 0
inputs[:, 1, :] = trial_in_block[keep, None]
```

iii. Computing it before filtering preserves the animal's true position even when an intervening trial is later excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice`, whose retained values are +1 (left) and -1 (right).

ii.
```python
raw_choice = trials["choice"].to_numpy()
choice = (raw_choice == -1).astype(np.int64)
```

iii. The agent confirmed IBL's sign convention in repository code/data.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nonresponses are filtered; +1 becomes 0 (left), -1 becomes 1 (right), and the scalar label is repeated over all 100 bins.

ii.
```python
& (trials["choice"].to_numpy() != 0)
outputs[:, 0, :] = choice[keep, None]
```

iii. This implements the task's required left=0/right=1 categorical coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials.probabilityLeft`.

ii.
```python
probability_left = trials["probabilityLeft"].to_numpy(dtype=np.float64)
```

iii. The agent identified it as the block prior specified by the decoder task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to labels 0, 1, and 2 with `isclose`; unexpected retained values raise an error. Labels are repeated over time.

ii.
```python
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
prior[np.isclose(probability_left, value)] = label
outputs[:, 1, :] = prior[keep, None]
```

iii. This is the mapping explicitly required by the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
position = np.load(position_path)
timestamps = np.load(timestamps_path)
```

iii. The agent traced `SessionLoader.load_wheel` to these raw arrays.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, filtered/differentiated with the IBL wheel utility (20 Hz, order 8), converted to absolute velocity, linearly sampled into each trial, and discretized by within-session tertiles.

ii.
```python
position_1khz, times_1khz = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(position_1khz, fs=1000, corner_frequency=20, order=8)
return _interpolate_trials(times_1khz, np.abs(velocity), stimulus_times)
```

iii. The agent aimed to reproduce `SessionLoader`'s recommended/default wheel processing exactly.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained samples in a session are pooled; the 1/3 and 2/3 quantiles define low, medium, and high labels via `digitize`.

ii.
```python
q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
```

iii. The trajectory explicitly considered the ambiguity and chose session-specific equal-frequency bins to avoid cross-session scale differences.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel values are linearly interpolated at 100 points from window start +20 ms through window end, so they correspond to neural-bin right edges, not centers.

ii.
```python
target_times = np.linspace(begins[trial] + BIN_SIZE, ends[trial], N_BINS)
values[trial] = interp1d(times, vals, kind="linear", fill_value="extrapolate")(target_times)
```

iii. The agent described this as matching repository endpoint interpolation and checked that stream endpoints were within one bin.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `<view>Camera.ROIMotionEnergy.npy` and matching `_ibl_<view>Camera.times.npy`, preferring left and falling back to right.

ii.
```python
for view in ("left", "right"):
    energy = np.load(energy_path)
    times = np.load(times_path)
```

iii. The agent says this follows the reference's camera selection logic and handles sessions lacking the preferred view.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Released motion energy is used without filtering/normalization. Extra leading camera timestamps are trimmed, values are linearly interpolated to trial points, then discretized into within-session tertiles.

ii.
```python
if len(times) > len(energy):
    times = times[-len(energy):]
values, good = _interpolate_trials(times, energy, stimulus_times)
```

iii. The agent mirrored `SessionLoader._check_video_timestamps` for known camera timestamp mismatches and applied no unmentioned signal processing.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same pooled within-session 1/3 and 2/3 quantiles and `digitize` labels as wheel speed.

ii.
```python
motion_labels, motion_edges = _three_bins(motion)
outputs[:, 3, :] = motion_labels
```

iii. The agent chose equal-frequency session-specific classes for both continuous outputs consistently.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is linearly interpolated at the same 100 right-edge timestamps used for wheel speed, relative to the same stimulus onset.

ii.
```python
target_times = np.linspace(begins[trial] + BIN_SIZE, ends[trial], N_BINS)
```

iii. The agent relied on synchronized camera/neural clocks and repository-style endpoint interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Required files/columns and length consistency are validated. Extra leading camera timestamps are trimmed; too few timestamps cause fallback/failure. Nonfinite or incompletely covered behavior trials are removed. Sessions with fewer than two valid trials or conversion errors are skipped and logged. Atomic, versioned checkpoints support safe resume.

ii.
```python
if len(times) > len(energy): times = times[-len(energy):]
good[trial] = np.isfinite(values[trial]).all()
if len(keep_idx) < 2: raise ValueError(...)
except Exception as exc: failures[spec.eid] = repr(exc)
```

iii. The trajectory reports deriving the usable-session subset from actual availability/alignment rather than hard-coding it, while preserving zero-spike trials because the reference binning produces valid zero arrays.

## 10-a. What are the most time-consuming steps of the code?

i. Loading very large spike arrays, binning them across trials, serializing the multi-gigabyte session/final arrays, and downstream full-decoder validation dominate runtime.

ii.
```python
times = np.load(times_path, mmap_mode="r")
for trial, (lo, hi) in enumerate(zip(start_idx, end_idx)):
    binned[trial].flat[:] = np.bincount(...)
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory's audits estimate hundreds of thousands of units and a very large output; the agent introduced memory mapping, concurrency, and checkpoints for these costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial behavior interpolation, per-trial spike binning, trial-number run length, prior mapping, region-index construction, and payload assembly all use Python loops. The first two offer the clearest vectorization opportunity, though ragged time slices complicate it.

ii.
```python
for trial, (beg_idx, end_idx) in enumerate(zip(idx_beg, idx_end)):
for trial, (lo, hi) in enumerate(zip(start_idx, end_idx)):
for trial in range(1, len(probability_left)):
```

iii. The agent favored bounded contiguous slices and `bincount` within each trial, then parallelized independent sessions; the trajectory does not claim the loops are fully optimal.

## 10-c. What processing does the code repeat multiple times?

i. `_interpolate_trials` repeats nearly identical slicing/interpolation for wheel and motion energy; file discovery repeatedly searches glob patterns; each session creates a new `BrainRegions`; checkpoints are written and later reread during assembly.

ii.
```python
return _interpolate_trials(times_1khz, np.abs(velocity), stimulus_times)
values, good = _interpolate_trials(times, energy, stimulus_times)
atlas = BrainRegions()
```

iii. The shared interpolation helper intentionally consolidates repeated logic. Checkpoint rereading was accepted to make the large conversion resumable and memory-bounded.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes/stores diagnostics such as tertile edges, elapsed time, counts and camera view; loads channel atlas information solely for region metadata; writes then rereads checkpoints; and generates a failures sidecar. These are not decoder features but support validation/provenance. It also computes paper masks and behavior traces for sessions later skipped.

ii.
```python
"wheel_speed_tertile_edges": list(wheel_edges),
"elapsed_seconds": float(time.time() - started),
with open(path, "rb") as stream: payloads.append(pickle.load(stream))
```

iii. The trajectory emphasizes reproducibility, resumability, and auditing at full-release scale, which explains these otherwise downstream-unused computations.
