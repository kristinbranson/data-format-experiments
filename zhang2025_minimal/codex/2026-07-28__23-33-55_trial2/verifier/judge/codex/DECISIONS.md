# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the reference ONE API loading path. Instead, it read `/app/code/code_zhang2025/data/bwm_release.csv`, grouped rows by `eid`, constructed each session path directly under `/app/data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf`, and then loaded parquet and `.npy` files from disk with manual revision selection.

ii. ```python
BWM_RELEASE_CSV = Path("/app/code/code_zhang2025/data/bwm_release.csv")
ONE_CACHE_DIR = Path("/app/data/one_cache")
```

```python
release_df = pd.read_csv(BWM_RELEASE_CSV)
for eid, group in release_df.groupby("eid", sort=False):
    ...
    session_path = (
        ONE_CACHE_DIR
        / session["lab"]
        / "Subjects"
        / session["subject"]
        / session["date"]
        / f"{session['session_number']:03d}"
    )
    alf_path = session_path / "alf"
```

iii. `CONVERSION_NOTES.md` says this was done for “offline loader around the local ONE cache” using `bwm_release.csv` to resolve session paths. The trajectory says the agent checked whether “local ONE metadata is sufficient or whether I need to resolve session paths directly from the filesystem,” then chose the direct-cache approach.

## 1-b. How are the data split into subjects?

i. Sessions are assigned to subjects using the `subject` column from `bwm_release.csv`. The final `subjects` list preserves first-seen session order rather than sorting, and `subject_idx` is built from that ordered list.

ii. ```python
session_rows.append({
    "eid": eid,
    "subject": first["subject"],
    ...
})
```

```python
subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_lookup[session["subject"]] for session in kept_sessions],
    dtype=np.int64,
),
```

iii. The notes justify this by saying sessions were discovered from `bwm_release.csv`, which already contains the session-to-subject mapping. No additional reasoning beyond using the release metadata is given.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. All probe rows with the same `eid` are grouped into one session record, and the converter iterates session-by-session over those grouped rows.

ii. ```python
for eid, group in release_df.groupby("eid", sort=False):
    first = group.iloc[0]
    session_rows.append({
        "eid": eid,
        ...
        "probe_names": list(group["probe_name"]),
    })
```

iii. `CONVERSION_NOTES.md` says session discovery came from `bwm_release.csv`, and the trajectory says the agent used it to map `eid -> session path / probes`.

## 1-d. How are the data split into trials?

i. Trials are taken from the trials table `_ibl_trials.table.pqt`, one table row per trial. The code iterates through `trials_df.iterrows()` and keeps only rows that pass the mask and later coverage checks.

ii. ```python
def load_trials_table(alf_path: Path):
    trials_path = latest_revision_file(alf_path, "#*/_ibl_trials.table.pqt")
    ...
    return pd.read_parquet(trials_path)
```

```python
for trial_idx, trial_row in trials_df.iterrows():
    if not trial_mask[trial_idx]:
        continue
    ...
```

iii. The notes describe the trial table as the source of trial-level variables and filtering. There is no separate justification beyond following the released ALF trial table structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a broader mask than the reference. It requires finite `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, and `goCue_times`; requires reaction time in `[0.08, 2.0]`; requires `feedback_times - goCue_times <= 10`; excludes `choice == 0`; then drops any trial whose wheel or whisker trace cannot be interpolated over the full window, and finally drops trials with no spikes in any retained neuron.

ii. ```python
def compute_trial_mask(trials_df: pd.DataFrame):
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
        "goCue_times",
    ]
    ...
    rt = trials_df["firstMovement_times"].to_numpy() - trials_df["stimOn_times"].to_numpy()
    trial_len = trials_df["feedback_times"].to_numpy() - trials_df["goCue_times"].to_numpy()
    mask &= rt >= MIN_RT_S
    mask &= rt <= MAX_RT_S
    mask &= trial_len <= MAX_TRIAL_LEN_S
    mask &= trials_df["choice"].to_numpy() != 0
    return mask
```

```python
wheel_trial = interpolate_behavior_trial(...)
whisker_trial = interpolate_behavior_trial(...)
if wheel_trial is None or whisker_trial is None:
    continue
...
if not np.any(neural_trial):
    continue
```

iii. `CONVERSION_NOTES.md` explicitly lists the finite-column checks, the RT range, `choice != 0`, and the 10 s trial-length limit, claiming this follows `load_trials_and_mask(...)` plus `prepare_data(...)`. The notes also explicitly justify dropping trials with no spikes as a practical curation step for the dense output format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived from `spikes.times.npy` and `spikes.clusters.npy` after filtering clusters using `clusters.metrics.pqt`. Region labels are assigned using `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`, but the actual binned neural values come from spike times and cluster assignments.

ii. ```python
metrics_path = latest_revision_file(sorter_base, "#*/clusters.metrics.pqt")
spike_times_path = latest_revision_file(sorter_base, "#*/spikes.times.npy")
spike_clusters_path = latest_revision_file(sorter_base, "#*/spikes.clusters.npy")
cluster_channels_path = latest_revision_file(sorter_base, "#*/clusters.channels.npy")
channel_regions_path = latest_revision_file(
    sorter_base,
    "#*/channels.brainLocationIds_ccf_2017.npy",
)
```

```python
return {
    "spike_times": spike_times[spike_keep],
    "spike_clusters": remap[spike_clusters[spike_keep]],
    "cluster_acronyms": cluster_acronyms[kept_clusters],
}
```

iii. The notes say spikes were loaded from `spikes.times.npy` and `spikes.clusters.npy`, cluster QC came from `clusters.metrics.pqt`, and cluster regions came from `channels.brainLocationIds_ccf_2017.npy`.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, bins spikes into 20 ms bins over a 2 s stimulus-aligned window, and stores dense spike-count matrices as `float16`. It does not divide counts by bin width, so the result is spike counts per bin rather than firing rates.

ii. ```python
for info in probe_infos:
    if info is None:
        continue
    merged_times.append(info["spike_times"])
    merged_clusters.append(info["spike_clusters"] + cluster_offset)
    ...
```

```python
def bin_spikes_for_trial(...):
    ...
    flat_idx = clusters[valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
    return counts.astype(np.float16)
```

iii. The notes justify merging probes as matching the reference and state explicitly that “Neural arrays are stored as dense `float16` spike-count matrices to keep the full pickle size manageable.” No justification is given for departing from firing-rate scaling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are kept only if `clusters.metrics.label >= 1.0`. The AI additionally excludes clusters whose derived region acronym is `root` or `void`, skips sessions with no retained units, and drops individual trials if the retained neurons have zero spikes across the full 2 s window.

ii. ```python
keep_mask = metrics["label"].to_numpy() >= 1.0
...
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
...
if kept_clusters.size == 0:
    return None
```

```python
merged = merge_session_probes(probe_infos)
if merged is None:
    dropped_sessions.append((session["eid"], "no_good_units"))
    continue
...
if not np.any(neural_trial):
    continue
```

iii. The notes justify `label >= 1` using the paper’s “well-isolated neurons” criterion and a release-wide neuron-count sanity check. They also justify excluding `root/void` and zero-spike trials as practical curation for a dense decoder-format dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For trial `t`, the code defines the neural window as `[stimOn_times[t] - 0.5, stimOn_times[t] + 1.5)` and bins spikes relative to that window start, so the neural matrices are stimulus-onset aligned.

ii. ```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
```

```python
neural_trial = bin_spikes_for_trial(
    merged["spike_times"],
    merged["spike_clusters"],
    n_clusters,
    trial_start,
)
```

iii. The notes and trajectory both explicitly say the alignment event is `stimOn_times`, with a `[-0.5 s, +1.5 s]` window to match the reference configuration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 20 ms bins over 100 time bins per trial. No additional temporal rebinning is applied after binning/interpolation onto this grid.

ii. ```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

```python
bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
...
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
```

iii. The notes cite the reference defaults of a 2 s interval, 20 ms bins, and `stimOn_times` alignment.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read as its own raw data stream. It is constructed from the chosen alignment convention around `stimOn_times`, using the fixed window bounds and bin size.

ii. ```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
```

iii. The notes say this input is “time-varying” and represented from `-0.48 s` to `1.50 s`, tied to stimulus-onset alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI constructs a deterministic 100-point vector with `np.linspace` from `-0.48` to `1.50` seconds and repeats that same vector for every retained trial. This represents the right edge of each 20 ms bin rather than the bin center.

ii. ```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
input_trial = np.vstack([
    time_since_stim,
    np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32),
]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as “represented at the right edge of each bin, from `-0.48 s` to `1.50 s`.”

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is paired with the same 100-bin trial window as the neural data, but the input time values are the right edges of those bins rather than the centers. So the intended alignment is one value per neural bin, using the same stimulus-aligned window boundaries.

ii. ```python
neural_trial = bin_spikes_for_trial(..., trial_start)
```

```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. The notes explicitly say the time input is represented at the right edge of each 20 ms bin and is aligned to `stimOn_times`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trial table. Consecutive trials with the same `probabilityLeft` are treated as belonging to the same block.

ii. ```python
def compute_trial_number_in_block(probability_left):
    values = np.asarray(probability_left, dtype=np.float64)
    ...
    for idx, value in enumerate(values):
        same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
        count = count + 1 if same_block else 1
```

iii. The notes say this is “computed on the original session trial order before masking” and define the block by the current `probabilityLeft` value.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a 1-based within-block counter on the full unmasked trial sequence, incrementing while `probabilityLeft` stays the same and resetting to `1` when it changes. For retained trials, that scalar is then repeated across all 100 time bins.

ii. ```python
count = 0
previous = np.nan
for idx, value in enumerate(values):
    same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
    count = count + 1 if same_block else 1
    numbers[idx] = float(count)
    previous = value
```

```python
np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32)
```

iii. The notes justify computing it before masking so dropped trials still advance the block count, but they explicitly describe it as a “1-based count.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived directly from the trial-table `choice` column.

ii. ```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0
    if np.isclose(value, -1.0):
        return 1
```

```python
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. The notes explicitly say `choice` is mapped from IBL `choice`, with `1 -> left -> 0` and `-1 -> right -> 1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The only processing is recoding the signed IBL choice into the requested binary labels, and then repeating that class across all 100 time bins of the retained trial.

ii. ```python
choice_class = choice_to_class(float(trial_row["choice"]))
...
output_trial = np.vstack([
    np.full(N_BINS, choice_class, dtype=np.int8),
    ...
])
```

iii. The notes justify repeating static outputs across time so every output has the same `(d_output, T)` shape.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trial-table `probabilityLeft` column.

ii. ```python
def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
    ...
```

```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. The notes explicitly describe the `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` mapping.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The only processing is recoding the scalar prior probability into the requested class label and repeating it across all 100 time bins for each retained trial.

ii. ```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
...
output_trial = np.vstack([
    np.full(N_BINS, choice_class, dtype=np.int8),
    np.full(N_BINS, prior_class, dtype=np.int8),
    ...
])
```

iii. The notes justify repeating static outputs across time to keep the output tensor shape uniform.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. ```python
timestamps_path = latest_revision_file(alf_path, ["_ibl_wheel.timestamps.npy", "#*/_ibl_wheel.timestamps.npy"])
position_path = latest_revision_file(alf_path, ["_ibl_wheel.position.npy", "#*/_ibl_wheel.position.npy"])
```

```python
position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity).astype(np.float32)
```

iii. The notes say the AI followed the same `SessionLoader.load_wheel(...)` processing as the reference, with 1000 Hz interpolation and Butterworth filtering.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to 1000 Hz, applies the Butterworth-based velocity computation, takes absolute velocity, then linearly interpolates each trial onto the decoder’s 100-bin time grid. Unlike the reference, the interpolation targets right bin edges and uses only samples from inside the requested interval.

ii. ```python
def interpolate_position(re_ts, re_pos, freq=WHEEL_FS):
    t = np.arange(re_ts[0], re_ts[-1], 1.0 / freq, dtype=np.float64)
    ...
    yinterp = np.interp(t, re_ts, re_pos)
```

```python
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
...
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. The notes justify the interpolation frequency, filter cutoff, and order by saying they match `SessionLoader.load_wheel(...)`. The notes also say trial values were resampled onto 20 ms decoder bins using the same interval coverage checks as `get_behavior_per_interval(...)`, although the code is not an exact copy of that helper.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes two global quantile thresholds from all retained wheel-speed samples across all sessions, trials, and time bins, then digitizes each trial’s wheel-speed trace into three classes using those global thresholds.

ii. ```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
```

```python
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
```

iii. `CONVERSION_NOTES.md` explicitly says `wheel_speed_bin` used “global tertiles across all retained wheel-speed time bins.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is sliced over the same stimulus-aligned trial interval as the neural data and interpolated onto 100 points intended to correspond to the neural bins. In this implementation those 100 points are the right bin edges, not the bin centers.

ii. ```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
```

```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
```

iii. The notes justify this as resampling onto the “20 ms decoder bins” within the same `stimOn_times`-aligned window as neural activity.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, or `rightCamera.ROIMotionEnergy.npy` plus `_ibl_rightCamera.times.npy`, with left preferred when available.

ii. ```python
for view in ("left", "right"):
    me_path = latest_revision_file(alf_path, f"#*/{view}Camera.ROIMotionEnergy.npy")
    ...
    times_path = latest_revision_file(
        alf_path,
        [f"_ibl_{view}Camera.times.npy", f"#*/_ibl_{view}Camera.times.npy"],
    )
```

```python
return timestamps, motion_energy, view
```

iii. The notes explicitly say the left camera was used when available and the right camera otherwise.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion-energy trace is used directly after timestamp repair: if the timestamp array is longer than the motion-energy array, the extra leading timestamps are trimmed. Each retained trial is then linearly interpolated onto the 100-bin decoder grid, again using right bin edges rather than bin centers.

ii. ```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
```

```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. The notes justify the timestamp trimming by claiming it matches `_check_video_timestamps(...)`, and say the values were linearly interpolated onto the 20 ms decoder bins.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI computes two global quantile thresholds from all retained whisker-motion-energy samples across all sessions, trials, and time bins, then digitizes each trial trace into three classes using those global thresholds.

ii. ```python
whisker_flat = np.concatenate(whisker_continuous_all)
_, whisker_edges = discretize_three_bins(whisker_flat)
```

```python
whisker_bins = np.digitize(
    whisker_trial,
    [whisker_edges[0], whisker_edges[1]],
    right=False,
).astype(np.int8)
```

iii. `CONVERSION_NOTES.md` explicitly says `whisker_motion_energy_bin` used “global tertiles across all retained whisker-motion-energy time bins.”

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is taken over the same stimulus-aligned interval as the neural data and interpolated to 100 trial time points intended to match the neural bins. As with wheel speed, those time points are the right edges of the bins.

ii. ```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
```

iii. The notes justify this as using the same `stimOn_times`-aligned 20 ms decoder bins as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or inconsistent data mostly by dropping affected trials or sessions. It manually picks the latest revisioned file, trims overlong camera timestamp arrays, raises errors for missing required files, drops sessions with no retained units or fewer than two retained trials, skips trials whose behavior traces cannot cover the full window, and skips trials with no spikes in retained neurons.

ii. ```python
def latest_revision_file(base: Path, patterns):
    ...
    matches.sort(key=sort_key)
    return matches[-1]
```

```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
...
if wheel_trial is None or whisker_trial is None:
    continue
...
if merged is None:
    dropped_sessions.append((session["eid"], "no_good_units"))
    continue
...
if len(session_neural) < 2:
    dropped_sessions.append((session["eid"], "fewer_than_two_trials"))
    continue
```

iii. The notes frame this as matching mixed-layout ALF revisions and `_check_video_timestamps(...)`, and as practical curation for sessions/trials that cannot support the decoder format.

## 10-a. What are the most time-consuming steps of the code?

i. The code is dominated by repeatedly loading large spike-sorting arrays from disk and then iterating trial-by-trial to interpolate behavior and bin spikes. The global concatenation and quantile calculation for wheel/whisker traces is smaller by comparison.

ii. ```python
for probe_name in session["probe_names"]:
    probe_infos.append(load_probe_good_units(alf_path / probe_name, brain_regions))
```

```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    wheel_trial = interpolate_behavior_trial(...)
    whisker_trial = interpolate_behavior_trial(...)
    neural_trial = bin_spikes_for_trial(...)
```

iii. The trajectory shows the agent spent time investigating practical loading performance and local-cache layout, but there is no explicit written performance analysis in the final notes beyond the note that the dense target format would otherwise be large.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial loop over `trials_df.iterrows()`, which repeatedly slices/interpolates behavior and bins spikes, and the Python loop in `compute_trial_number_in_block`. The per-session probe loop is also serial and could only be parallelized at a higher level.

ii. ```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
```

```python
for idx, value in enumerate(values):
    same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
    count = count + 1 if same_block else 1
```

iii. The notes do not discuss vectorization. This is an inference from the implemented loops rather than a stated design rationale.

## 10-c. What processing does the code repeat multiple times?

i. The code rebuilds the same `time_since_stim` vector inside every retained trial instead of once per dataset or session. It also re-runs `searchsorted` and interpolation separately for wheel and whisker on every trial, and bins spikes trial-by-trial rather than reusing precomputed session-level window indices.

ii. ```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    time_since_stim = np.linspace(
        OFF_START_S + BIN_SIZE_S,
        OFF_END_S,
        N_BINS,
        dtype=np.float32,
    )
```

```python
wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
neural_trial = bin_spikes_for_trial(...)
```

iii. The notes do not acknowledge this repeated work. This is inferred from the code.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `velocity_filtered` computes acceleration even though only velocity is used. `discretize_three_bins` computes a full digitized array for the global flattened traces, but the caller throws away that digitized output and keeps only the thresholds. The converter also computes `trial_end` as a separate scalar even though only the interpolation helper uses it immediately.

ii. ```python
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
acc = np.insert(np.diff(vel), 0, 0.0) * fs
return vel, acc
```

```python
wheel_flat = np.concatenate(wheel_continuous_all)
whisker_flat = np.concatenate(whisker_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
_, whisker_edges = discretize_three_bins(whisker_flat)
```

iii. The notes do not discuss these discarded intermediate computations. This is inferred from the implementation.
