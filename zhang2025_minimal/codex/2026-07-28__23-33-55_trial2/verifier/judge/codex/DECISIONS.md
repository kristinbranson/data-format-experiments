# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the release manifest from `bwm_release.csv`, groups rows by `eid` to recover one session per experiment, resolves each session to a local path inside `/app/data/one_cache`, and then manually loads all required ALF objects from disk: the trials table, wheel arrays, whisker motion-energy arrays, and spike-sorting outputs for each probe.

ii. 
```python
release_df = pd.read_csv(BWM_RELEASE_CSV)
session_rows = []
for eid, group in release_df.groupby("eid", sort=False):
    first = group.iloc[0]
    session_rows.append({
        "eid": eid,
        "subject": first["subject"],
        "lab": first["lab"],
        "date": first["date"],
        "session_number": int(first["session_number"]),
        "probe_names": list(group["probe_name"]),
    })

session_path = (
    ONE_CACHE_DIR
    / session["lab"]
    / "Subjects"
    / session["subject"]
    / session["date"]
    / f"{session['session_number']:03d}"
)
alf_path = session_path / "alf"

trials_df = load_trials_table(alf_path)
wheel_times, wheel_speed = load_wheel_speed(alf_path)
whisker_times, whisker_me, motion_view = load_motion_energy(alf_path)
for probe_name in session["probe_names"]:
    probe_infos.append(load_probe_good_units(alf_path / probe_name, brain_regions))
```

iii. The notes say session discovery was driven by `code/code_zhang2025/data/bwm_release.csv`, and the trajectory explains that the agent intentionally used the local ONE cache directly so the conversion would be reproducible offline.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the release table. After session conversion, the agent builds an ordered list of unique retained subjects and stores, for each kept session, an index into that list.

ii.
```python
session_rows.append({
    "eid": eid,
    "subject": first["subject"],
    ...
})

subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}

"subject_idx": np.asarray(
    [subject_lookup[session["subject"]] for session in kept_sessions],
    dtype=np.int64,
),
```

iii. The notes state that subjects were taken from the release manifest. No separate deeper justification was documented beyond matching the target format.

## 1-c. How are the data split into sessions?

i. One session is one unique `eid` from the release table. All probes listed for that `eid` are merged into a single session-level recording before trialization and output formatting.

ii.
```python
for eid, group in release_df.groupby("eid", sort=False):
    ...
    session_rows.append({
        "eid": eid,
        ...
        "probe_names": list(group["probe_name"]),
    })

for probe_name in session["probe_names"]:
    probe_infos.append(load_probe_good_units(alf_path / probe_name, brain_regions))
merged = merge_session_probes(probe_infos)
```

iii. The notes explicitly justify probe merging as matching the decoder paper and repository logic that probes from the same session are not independent and should not be treated as separate sessions.

## 1-d. How are the data split into trials?

i. Trials are taken directly from rows of `_ibl_trials.table.pqt`. For each retained trial row, the code constructs a fixed 2 s window relative to `stimOn_times`, bins spikes in that interval, and interpolates wheel and whisker traces over the same interval.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows():
    if not trial_mask[trial_idx]:
        continue
    trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
    trial_end = float(trial_row["stimOn_times"] + OFF_END_S)

    wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
    whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
    neural_trial = bin_spikes_for_trial(
        merged["spike_times"],
        merged["spike_clusters"],
        n_clusters,
        trial_start,
    )
```

iii. The notes and trajectory say the agent followed the repo defaults from `0_data_caching.py`: alignment to `stimOn_times`, a `[-0.5, 1.5]` window, and 20 ms bins.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies a trial mask requiring finite `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, and `goCue_times`; then it excludes trials with reaction times outside 0.08-2.0 s, trials longer than 10 s from `goCue_times` to `feedback_times`, and no-choice trials. It further drops trials if wheel or whisker coverage is insufficient for the aligned window, and drops all-zero neural trials.

ii.
```python
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

if wheel_trial is None or whisker_trial is None:
    continue
...
if not np.any(neural_trial):
    continue
```

iii. The notes say this was chosen to follow `load_trials_and_mask(...)` plus the reference preparation logic, and the trajectory later explains that the extra zero-spike-trial removal was added after verification surfaced all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from spike times and cluster assignments in each probe’s `pykilosort` output, with cluster QC from `clusters.metrics.pqt` and brain-region labels derived from `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
metrics_path = latest_revision_file(sorter_base, "#*/clusters.metrics.pqt")
spike_times_path = latest_revision_file(sorter_base, "#*/spikes.times.npy")
spike_clusters_path = latest_revision_file(sorter_base, "#*/spikes.clusters.npy")
cluster_channels_path = latest_revision_file(sorter_base, "#*/clusters.channels.npy")
channel_regions_path = latest_revision_file(
    sorter_base,
    "#*/channels.brainLocationIds_ccf_2017.npy",
)

metrics = pd.read_parquet(metrics_path, columns=["label"])
spike_times = np.load(spike_times_path).astype(np.float64)
spike_clusters = np.load(spike_clusters_path).astype(np.int64)
cluster_channels = np.load(cluster_channels_path).astype(np.int64)
channel_region_ids = np.load(channel_regions_path).astype(np.int64)
```

iii. The notes explicitly describe those files as the source of spikes, QC labels, and anatomical labels.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, remaps cluster ids into a single session-wide index space, sorts spikes by time, and bins spike counts into dense `(n_neurons, 100)` matrices using 20 ms bins over each 2 s trial window. The result is stored as `float16`.

ii.
```python
merged_clusters.append(info["spike_clusters"] + cluster_offset)
...
order = np.argsort(spike_times, kind="stable")
return {
    "spike_times": spike_times[order],
    "spike_clusters": spike_clusters[order],
    "cluster_acronyms": np.asarray(merged_regions, dtype=object),
}

bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
flat_idx = clusters[valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
return counts.astype(np.float16)
```

iii. The notes say this was meant to match the reference session-level probe merge and the 20 ms stimulus-aligned binning used in the reference code path.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only clusters with `label >= 1`, excludes units whose derived acronyms are `root` or `void`, and removes trials whose retained neural matrix is all zeros. It does not apply the paper’s later region-level constraints such as a minimum of five well-isolated neurons per region per session.

ii.
```python
keep_mask = metrics["label"].to_numpy() >= 1.0
...
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
kept_clusters = np.flatnonzero(keep_mask)
...
if not np.any(neural_trial):
    continue
```

iii. The notes justify `label >= 1` by citing the paper’s “well-isolated neurons” analysis set and by saying a release-wide scan gave unit counts close to the paper. The trajectory shows this was an explicit choice after comparing raw cluster counts against expected counts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every neural trial is aligned to stimulus onset. The binning window starts 0.5 s before `stimOn_times` and ends 1.5 s after it.

ii.
```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5

trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
neural_trial = bin_spikes_for_trial(..., trial_start)
```

iii. The notes quote the reference repo defaults from `0_data_caching.py`, and the trajectory says the agent treated that script as the decisive reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins throughout, producing 100 bins for a 2 s trial window. There is no second-stage temporal rebinning after spike binning or behavior interpolation.

ii.
```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

iii. The notes say this matches the `binsize=0.02` and `time_window=(-.5, 1.5)` settings in the reference code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is not taken from a raw sampled variable. It is synthesized from the chosen alignment event `stimOn_times` and the fixed binning window definition.

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. The notes describe it as a time-varying decoder input “represented at the right edge of each bin.” No further justification beyond matching the aligned-bin representation was documented.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates a fixed length-100 ramp from `-0.48` s to `1.50` s, corresponding to the right edges of the 20 ms bins in the stimulus-aligned window. This vector is identical for every retained trial.

ii.
```python
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

iii. The notes explicitly say the variable is one value per 20 ms bin and uses the right edge of each bin.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned bin-for-bin with the neural matrices: the same 100 bins, the same 20 ms spacing, and the same `stimOn_times`-centered `[-0.5, 1.5]` trial window.

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
...
neural_trial = bin_spikes_for_trial(..., trial_start)
...
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. The justification is implicit in the notes’ single shared alignment-and-binning scheme for neural and behavioral streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` values in the trials table.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```

iii. The notes explicitly identify this variable as a 1-based count within the current `probabilityLeft` block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the original session trial order, increments a counter while `probabilityLeft` stays the same, resets the counter when the block value changes, and assigns the resulting 1-based count to each trial. For retained trials, that scalar is then repeated across all time bins.

ii.
```python
def compute_trial_number_in_block(probability_left):
    values = np.asarray(probability_left, dtype=np.float64)
    numbers = np.zeros(values.shape[0], dtype=np.float32)
    count = 0
    previous = np.nan
    for idx, value in enumerate(values):
        same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
        count = count + 1 if same_block else 1
        numbers[idx] = float(count)
        previous = value
    return numbers

np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32)
```

iii. The notes say it was computed “on the original session trial order before masking,” which is the explicit justification recorded by the agent.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived directly from the trials-table `choice` column.

ii.
```python
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. The notes explicitly say the source variable is IBL `choice`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps `choice == 1` to left class `0` and `choice == -1` to right class `1`, then repeats that class across all 100 bins for the trial so every output has the same `(d_output, T)` shape.

ii.
```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0
    if np.isclose(value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value: {value}")

output_trial = np.vstack([
    np.full(N_BINS, choice_class, dtype=np.int8),
    ...
])
```

iii. The notes document exactly this `1 -> left -> 0`, `-1 -> right -> 1` mapping and the repetition across time bins.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trials-table `probabilityLeft` column.

ii.
```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. The notes explicitly identify `probabilityLeft` as the source.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code encodes `probabilityLeft` categorically as `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeats that class across all 100 bins of the trial.

ii.
```python
def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
    for key, encoded in mapping.items():
        if np.isclose(value, key):
            return encoded
    raise ValueError(f"Unexpected probabilityLeft value: {value}")

output_trial = np.vstack([
    np.full(N_BINS, choice_class, dtype=np.int8),
    np.full(N_BINS, prior_class, dtype=np.int8),
    ...
])
```

iii. The notes document this mapping exactly.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps_path = latest_revision_file(alf_path, ["_ibl_wheel.timestamps.npy", "#*/_ibl_wheel.timestamps.npy"])
position_path = latest_revision_file(alf_path, ["_ibl_wheel.position.npy", "#*/_ibl_wheel.position.npy"])
...
timestamps = np.load(timestamps_path).astype(np.float64)
position = np.load(position_path).astype(np.float64)
```

iii. The notes explicitly say those arrays were the source.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The code linearly interpolates wheel position to 1000 Hz, computes low-pass-filtered velocity with an 8th-order 20 Hz Butterworth filter, takes its absolute value to obtain speed, and then linearly interpolates the speed trace onto the 20 ms trial bins.

ii.
```python
position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity).astype(np.float32)

wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
```

iii. The notes say this was intentionally matched to `SessionLoader.load_wheel(...)` from the IBL stack and to `get_behavior_per_interval(...)` for interval resampling.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent pools all retained wheel-speed samples across sessions, trials, and time bins, computes the global 1/3 and 2/3 quantiles, and uses those two thresholds to assign each sample to `low`, `medium`, or `high`.

ii.
```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
...
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
```

iii. The notes explicitly justify this as “global tertiles across all retained wheel-speed time bins.” No stronger reference-based justification was recorded because the task, not the paper, required categorical outputs.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset-centered `[-0.5, 1.5]` window as the neural data. The resampled wheel values are placed at the right edges of the shared 20 ms bins, and trials are rejected if wheel coverage does not span the interval closely enough.

ii.
```python
wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)

def interpolate_behavior_trial(...):
    ...
    if abs(interval_start - trial_times[0]) > binsize:
        return None
    if abs(interval_end - trial_times[-1]) > binsize:
        return None
    x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
```

iii. The notes say this used “the same interval coverage checks as `get_behavior_per_interval(...)`.” The trajectory also says the agent treated the repo’s stimulus-onset-aligned caching script as the decisive reference.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy`, together with the corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy` timestamps.

ii.
```python
for view in ("left", "right"):
    me_path = latest_revision_file(alf_path, f"#*/{view}Camera.ROIMotionEnergy.npy")
    ...
    times_path = latest_revision_file(
        alf_path,
        [f"_ibl_{view}Camera.times.npy", f"#*/_ibl_{view}Camera.times.npy"],
    )
```

iii. The notes explicitly document left-camera preference with right-camera fallback.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the whisker ROI motion-energy trace, fixes the common timestamp-length mismatch by trimming excess leading timestamps when needed, chooses the left camera if available otherwise the right camera, and linearly interpolates the resulting trace onto the shared 20 ms decoder bins.

ii.
```python
motion_energy = np.load(me_path).astype(np.float32)
timestamps = np.load(times_path).astype(np.float64)

if timestamps.shape[0] < motion_energy.shape[0]:
    raise ValueError(f"{view} camera timestamps shorter than motion energy")
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]

return timestamps, motion_energy, view
...
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. The notes explicitly say the timestamp trimming was chosen to match `SessionLoader._check_video_timestamps(...)`.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The agent computes global tertile thresholds across all retained whisker-motion-energy samples and digitizes every interpolated trial trace into `low`, `medium`, or `high`.

ii.
```python
whisker_flat = np.concatenate(whisker_continuous_all)
_, whisker_edges = discretize_three_bins(whisker_flat)
...
whisker_bins = np.digitize(
    whisker_trial,
    [whisker_edges[0], whisker_edges[1]],
    right=False,
).astype(np.int8)
```

iii. The notes explicitly describe “global tertiles across all retained whisker-motion-energy time bins.”

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned exactly like wheel speed: the same `stimOn_times`-anchored `[-0.5, 1.5]` interval, the same 20 ms bins, and the same coverage checks before interpolation.

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. The notes justify this by reference to the shared decoder binning scheme and the repo’s interval interpolation logic.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code is defensive. It searches for the latest matching ALF revision, accepts wheel/camera files whether they live at the ALF root or in a revision folder, trims extra camera timestamps, falls back from left to right whisker camera, drops individual trials when behavior coverage is inadequate, drops sessions when required files are missing or no good units remain, and records those session-level failures in `dropped_sessions`.

ii.
```python
def latest_revision_file(base: Path, patterns):
    ...
    matches.sort(key=sort_key)
    return matches[-1]

if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]

for view in ("left", "right"):
    ...
    return timestamps, motion_energy, view

if wheel_trial is None or whisker_trial is None:
    continue
...
except Exception as exc:
    dropped_sessions.append((session["eid"], f"error:{type(exc).__name__}:{exc}"))
```

iii. The trajectory explicitly mentions that the agent added mixed root-plus-revision file resolution after debugging a real cache-layout bug. The notes justify the timestamp trimming and camera fallback as matching IBL loader behavior.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is the nested session and trial processing in `convert_dataset`: loading large ALF arrays, looping over every retained trial, rebinning spikes for each trial, and separately interpolating wheel and whisker traces for each trial. The final global concatenations for discretization are also large but secondary.

ii.
```python
for session_idx, session in enumerate(session_rows, start=1):
    ...
    for probe_name in session["probe_names"]:
        probe_infos.append(load_probe_good_units(alf_path / probe_name, brain_regions))
    ...
    for trial_idx, trial_row in trials_df.iterrows():
        ...
        wheel_trial = interpolate_behavior_trial(...)
        whisker_trial = interpolate_behavior_trial(...)
        neural_trial = bin_spikes_for_trial(...)
```

iii. No explicit efficiency justification was documented. The trajectory instead shows the agent mainly tracking runtime and memory during long full-dataset passes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python-level loops could have been reduced or vectorized: the per-trial `iterrows()` loop, the manual loop in `compute_trial_number_in_block`, repeated probe loading/appending, and the final per-trial output-construction loop that digitizes wheel and whisker traces trial by trial.

ii.
```python
for idx, value in enumerate(values):
    same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
    count = count + 1 if same_block else 1
    numbers[idx] = float(count)
    previous = value

for trial_idx, trial_row in trials_df.iterrows():
    ...

for static_values, wheel_trial, whisker_trial in zip(
    session["static_outputs"],
    session["wheel_continuous"],
    session["whisker_continuous"],
):
    ...
```

iii. No explicit justification for keeping these loops was documented.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes a constant `time_since_stim` vector inside every retained trial, runs nearly identical interpolation logic separately for wheel and whisker on every trial, and makes two passes over dynamic outputs: once to store continuous traces and again later to digitize them into categories.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
    whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
    ...
    time_since_stim = np.linspace(
        OFF_START_S + BIN_SIZE_S,
        OFF_END_S,
        N_BINS,
        dtype=np.float32,
    )
    ...
    wheel_continuous_all.append(wheel_trial)
    whisker_continuous_all.append(whisker_trial)

wheel_flat = np.concatenate(wheel_continuous_all)
whisker_flat = np.concatenate(whisker_continuous_all)
...
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
whisker_bins = np.digitize(...)
```

iii. No explicit justification for the repeated work was documented. The notes only justify the final semantics, not the implementation efficiency.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes wheel acceleration even though acceleration is never used; it computes full continuous wheel and whisker trial traces only to discard them after turning them into categories in the final dataset; and `discretize_three_bins` computes full bin assignments once during threshold estimation but those initial assignments are ignored and only the edges are kept.

ii.
```python
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
acc = np.insert(np.diff(vel), 0, 0.0) * fs
return vel, acc

wheel_continuous_all.append(wheel_trial)
whisker_continuous_all.append(whisker_trial)
...
_, wheel_edges = discretize_three_bins(wheel_flat)
_, whisker_edges = discretize_three_bins(whisker_flat)
```

iii. No explicit justification for these discarded intermediates was documented, beyond the general note that the full dataset had to stay manageable and pass decoder verification.
