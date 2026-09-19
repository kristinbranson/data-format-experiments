# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the ONE API. It loaded the release index from `code/code_zhang2025/data/bwm_release.csv`, grouped rows by `eid`, and built filesystem paths directly under `/app/data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf`. It then opened trial tables with `pd.read_parquet(...)` and other arrays with `np.load(...)`, resolving the newest revision folder by globbing `#...#` directories.

ii. 
```python
BWM_RELEASE_CSV = Path("/app/code/code_zhang2025/data/bwm_release.csv")
ONE_CACHE_DIR = Path("/app/data/one_cache")
```

```python
release_df = pd.read_csv(BWM_RELEASE_CSV)
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
```

```python
session_path = (
    ONE_CACHE_DIR / session["lab"] / "Subjects" / session["subject"]
    / session["date"] / f"{session['session_number']:03d}"
)
alf_path = session_path / "alf"
trials_df = load_trials_table(alf_path)
```

iii. In the trajectory, the AI said it wanted the conversion to be "reproducible offline against the cache" and therefore chose "an offline loader around the local ONE cache using `bwm_release.csv` to map `eid -> session path / probes`" instead of relying on ONE metadata resolution.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column in `bwm_release.csv`. Each kept session stores its subject string, and at the end the AI builds `subjects` with first-seen order and `subject_idx` by indexing into that ordered list.

ii. 
```python
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

iii. The trajectory does not contain a separate justification beyond using the release table as the source of session metadata. The implied justification is that subject identity is already present in `bwm_release.csv`, so no extra derivation is needed.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. The AI groups all probe rows with the same `eid` into one session record, then processes one such record at a time.

ii. 
```python
for eid, group in release_df.groupby("eid", sort=False):
    first = group.iloc[0]
    session_rows.append({
        "eid": eid,
        ...
        "probe_names": list(group["probe_name"]),
    })
```

```python
for session_idx, session in enumerate(session_rows, start=1):
    ...
```

iii. In the trajectory, the AI explicitly described `bwm_release.csv` as the mapping from `eid` to session path and probes, so it treated one `eid` as one session.

## 1-d. How are the data split into trials?

i. The AI loads `_ibl_trials.table.pqt` and iterates row-by-row through the resulting DataFrame. Each retained row becomes one trial in the converted dataset.

ii. 
```python
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

iii. The trajectory does not give a separate justification here. The code follows the raw trials table structure directly.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters. First, `compute_trial_mask(...)` requires finite `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, and `goCue_times`; reaction time in `[0.08, 2.0]` s; trial length `feedback_times - goCue_times <= 10` s; and nonzero choice. Later, each kept trial must also have wheel and whisker data spanning the window closely enough for interpolation, and must contain at least one spike in the retained neurons. Sessions with fewer than two retained trials are dropped.

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
```

```python
wheel_trial = interpolate_behavior_trial(...)
whisker_trial = interpolate_behavior_trial(...)
if wheel_trial is None or whisker_trial is None:
    continue
...
if not np.any(neural_trial):
    continue
...
if len(session_neural) < 2:
    dropped_sessions.append((session["eid"], "fewer_than_two_trials"))
    continue
```

iii. In the trajectory, the AI said it would reproduce trial masking from the papers/code using "`stimOn`, `choice`, `probabilityLeft`, `feedback`, `firstMovement`, RT in `[0.08, 2.0]`". Later it added a further curation step after verification: "The full verifier passed, but it surfaced a small number of all-zero neural trials... I’m dropping those trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are built from per-spike timestamps and cluster assignments (`spikes.times.npy`, `spikes.clusters.npy`). The AI also uses `clusters.metrics.pqt` for QC and `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` to determine each retained unit's brain region.

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
```

```python
return {
    "spike_times": spike_times[spike_keep],
    "spike_clusters": remap[spike_clusters[spike_keep]],
    "cluster_acronyms": cluster_acronyms[kept_clusters],
}
```

iii. In the trajectory, the AI said the remaining technical point was recovering brain-region names because "cluster QC tables don’t carry acronyms directly", so it chose to derive region labels from cached channel location IDs.

## 2-b. How is the `neural` data processed?

i. For each session, the AI merges retained clusters from all probes into a single unit index space, sorts all surviving spikes by time, then bins spikes trial-by-trial into `100` bins of `20` ms over the window `[stimOn-0.5, stimOn+1.5]`. It stores dense spike-count matrices as `float16`. It does not divide counts by bin width, so the saved `neural` signal is spike counts per bin, not firing rate in Hz.

ii. 
```python
for info in probe_infos:
    ...
    merged_clusters.append(info["spike_clusters"] + cluster_offset)
    ...
order = np.argsort(spike_times, kind="stable")
return {
    "spike_times": spike_times[order],
    "spike_clusters": spike_clusters[order],
    ...
}
```

```python
bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
flat_idx = clusters[valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
return counts.astype(np.float16)
```

iii. In the trajectory, the AI said it would reproduce "20 ms spike bins" in a 2 s `stimOn`-aligned window and merge probes within session. There is no trajectory evidence that it intended to convert counts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1.0`, maps cluster channels to region acronyms, excludes both `void` and `root`, skips sessions where no clusters survive, and later drops any retained trial whose binned neural matrix is all zeros.

ii. 
```python
keep_mask = metrics["label"].to_numpy() >= 1.0
...
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
kept_clusters = np.flatnonzero(keep_mask)
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

iii. The AI documented this as "clusters.metrics.label >= 1, excluding root/void regions" and justified `label >= 1` in its notes by citing the paper's well-isolated-neuron language and practical dataset size. In the trajectory it also explicitly justified dropping zero-spike trials as an extra curation step after verifier warnings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to `stimOn_times`. For each trial it defines `trial_start = stimOn_times - 0.5` and bins spikes over the next 2 s, so the window is `[-0.5, 1.5]` around stimulus onset.

ii. 
```python
OFF_START_S = -0.5
OFF_END_S = 1.5
...
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
...
neural_trial = bin_spikes_for_trial(..., trial_start)
```

iii. In the trajectory, the AI repeatedly said the decisive reference used "2 s `stimOn`-aligned windows `[-0.5, 1.5]`" and that it wanted to reproduce that alignment exactly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a fixed `20` ms bin size, producing `100` bins over the 2 s window. It does not apply any temporal rebinning beyond this initial binning/interpolation onto the 20 ms grid.

ii. 
```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

iii. In the trajectory, the AI explicitly committed to "20 ms bins" because that was what it found in `0_data_caching.py`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI derives this input from the trial's `stimOn_times` together with the fixed conversion window. It does not use another raw time series; instead it constructs a standard relative time grid for every retained trial.

ii. 
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
```

```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. In the trajectory, the AI said the relevant reference quantity was a "2 s `stimOn`-aligned window" and listed "time since stimulus onset" as the first decoder input it would construct.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates it synthetically with `np.linspace(...)`, using the right edge of each 20 ms bin: `[-0.48, -0.46, ..., 1.50]`. It repeats the same vector for every retained trial.

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

iii. The trajectory does not explicitly justify using right edges rather than bin centers. The closest justification is the AI's general claim that it would use 20 ms `stimOn`-aligned bins on the same window.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The AI aligns this input using the same `stimOn_times` window and the same `N_BINS=100` grid as the neural data. In its implementation the representative times are the right edges of the bins rather than their centers, but they are still tied to the same per-trial interval as the neural counts.

ii. 
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
...
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

iii. In the trajectory, the AI justified this only at a high level: it planned to put neural, wheel, and whisker variables on the same 20 ms `stimOn`-aligned window.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trials table. The AI treats consecutive trials with the same `probabilityLeft` as belonging to the same block.

ii. 
```python
def compute_trial_number_in_block(probability_left):
    values = np.asarray(probability_left, dtype=np.float64)
    ...
    same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
```

iii. The trajectory did not separately justify this, but the choice is implicit in the code and matches the task structure the AI identified from the reference repo.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI scans the full session's `probabilityLeft` values in order, resets the count when the value changes, and otherwise increments it. The resulting counter is 1-based, not 0-based. It computes the counter before masking and then repeats the per-trial value across all time bins in the retained trial.

ii. 
```python
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

iii. The trajectory only says that "trial number in block" would be one of the required inputs. There is no explicit justification for the 1-based choice.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column in the trials table.

ii. 
```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0
    if np.isclose(value, -1.0):
        return 1
```

```python
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. In the trajectory, the AI listed `choice` among the trial-mask and output variables it intended to reproduce from the reference code.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `choice == 1` to class `0` ("left") and `choice == -1` to class `1` ("right"). Trials with `choice == 0` are removed earlier by the trial mask.

ii. 
```python
if np.isclose(value, 1.0):
    return 0
if np.isclose(value, -1.0):
    return 1
raise ValueError(f"Unexpected choice value: {value}")
```

```python
mask &= trials_df["choice"].to_numpy() != 0
```

iii. The trajectory does not separately justify this mapping beyond treating `choice` as a required output and excluding no-choice trials.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

ii. 
```python
def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
```

```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. In the trajectory, the AI explicitly referred to this as the "prior-left block code" it would convert into the target format.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`.

ii. 
```python
mapping = {0.2: 0, 0.5: 1, 0.8: 2}
for key, encoded in mapping.items():
    if np.isclose(value, key):
        return encoded
```

iii. The trajectory does not contain a separate justification for this mapping beyond identifying it as a required output encoding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. 
```python
timestamps_path = latest_revision_file(alf_path, ["_ibl_wheel.timestamps.npy", "#*/_ibl_wheel.timestamps.npy"])
position_path = latest_revision_file(alf_path, ["_ibl_wheel.position.npy", "#*/_ibl_wheel.position.npy"])
...
timestamps = np.load(timestamps_path).astype(np.float64)
position = np.load(position_path).astype(np.float64)
```

iii. In the trajectory, the AI said it was drilling into "the exact wheel ... preprocessing so the converter reproduces those traces rather than approximating them."

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI linearly interpolates wheel position to 1000 Hz, applies the same style of Butterworth low-pass filtering and differentiation used by the IBL wheel helper, takes absolute velocity as speed, then linearly interpolates that speed into each trial's 20 ms decoder bins.

ii. 
```python
position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity).astype(np.float32)
```

```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. In the trajectory, the AI explicitly said wheel speed should be "derived from interpolated/filtered wheel velocity" to match `SessionLoader`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes two global quantile cut points from all retained wheel-speed samples across the entire converted dataset, then applies those edges to every trial in every session. This yields global tertiles, not session-specific tertiles.

ii. 
```python
def discretize_three_bins(values):
    values = np.asarray(values, dtype=np.float64)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    ...
    return bins, (float(q1), float(q2))
```

```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
...
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
```

iii. In the trajectory, the AI initially described the goal as "wheel-speed tertiles". In its final notes it explicitly documented the implemented rule as "global tertiles over all retained session/trial/time bins."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is resampled trial-by-trial over the same `stimOn-0.5` to `stimOn+1.5` interval as the neural data, with 100 samples per trial. As implemented, the resampling grid uses the right edge of each 20 ms bin rather than the bin center.

ii. 
```python
wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
```

```python
start_idx = np.searchsorted(sample_times, interval_start, side="right")
end_idx = np.searchsorted(sample_times, interval_end, side="left")
...
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
```

iii. In the trajectory, the AI justified this at a high level by saying wheel speed should be binned "on that same window" as the neural data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy` plus the corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`. The AI prefers the left camera and falls back to the right camera.

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

iii. In the trajectory, the AI said whisker motion energy should come from "the side camera traces with the same timestamp-fix logic as `SessionLoader`."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads the released motion-energy trace, trims extra leading timestamps if the time array is longer than the motion-energy array, performs no filtering or normalization, and linearly interpolates the trace into the 20 ms decoder bins for each trial.

ii. 
```python
motion_energy = np.load(me_path).astype(np.float32)
timestamps = np.load(times_path).astype(np.float64)

if timestamps.shape[0] < motion_energy.shape[0]:
    raise ValueError(f"{view} camera timestamps shorter than motion energy")
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
```

```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. In the trajectory, the AI said it wanted to reproduce the side-camera preprocessing rather than approximate it, and later fixed a bug around mixed root-level and revisioned camera timestamps.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI uses the same global-tertile rule as for wheel speed: it concatenates all retained whisker motion-energy samples across the whole dataset, computes two global quantile edges, and digitizes each trial against those edges.

ii. 
```python
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

iii. In its final notes, the AI explicitly described both dynamic outputs as "global tertiles".

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, whisker motion energy is resampled over each trial's `stimOn-0.5` to `stimOn+1.5` interval with 100 points. The time grid matches the implementation's right-edge grid rather than the reference bin centers.

ii. 
```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. In the trajectory, the AI justified this only generally, by saying whisker motion energy should be put into 20 ms `stimOn`-aligned bins on the same window as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly drops problematic data. Missing session files raise exceptions and the whole session is dropped. Missing wheel or whisker coverage within a trial causes that trial to be skipped. Sessions with no surviving good units or fewer than two surviving trials are dropped. If camera timestamps are longer than the motion-energy array, the AI trims the extra timestamps from the front. It also drops all-zero neural trials after verification flagged them.

ii. 
```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
```

```python
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

```python
except Exception as exc:
    dropped_sessions.append((session["eid"], f"error:{type(exc).__name__}:{exc}"))
```

iii. The trajectory explicitly justifies two such choices: fixing mixed root/revision camera layouts, and dropping zero-spike trials because the verifier surfaced them as a curation issue.

## 10-a. What are the most time-consuming steps of the code?

i. The code is likely dominated by loading large spike-sorting arrays for every probe and by per-trial repeated interpolation/binning work across all sessions. The AI also accumulates all retained wheel and whisker traces in memory before discretizing them globally.

ii. 
```python
spike_times = np.load(spike_times_path).astype(np.float64)
spike_clusters = np.load(spike_clusters_path).astype(np.int64)
```

```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    wheel_trial = interpolate_behavior_trial(...)
    whisker_trial = interpolate_behavior_trial(...)
    neural_trial = bin_spikes_for_trial(...)
```

iii. The trajectory does not explicitly rank hotspots. It does, however, emphasize matching the offline cache layout and repeatedly rerunning the full conversion, which implies the cost sits in session-scale I/O and per-trial processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the per-trial loop that repeatedly interpolates wheel/whisker traces and bins spikes, and the loop inside `compute_trial_number_in_block(...)`. The probe-merging and assembly loops are less important.

ii. 
```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    wheel_trial = interpolate_behavior_trial(...)
    whisker_trial = interpolate_behavior_trial(...)
    neural_trial = bin_spikes_for_trial(...)
```

```python
for idx, value in enumerate(values):
    same_block = ...
    count = count + 1 if same_block else 1
```

iii. The trajectory does not discuss vectorization directly. This conclusion is inferred from the implementation.

## 10-c. What processing does the code repeat multiple times?

i. The AI recomputes the same `time_since_stim` vector inside every retained trial even though it is constant. It also stores continuous wheel and whisker traces trial-by-trial, concatenates them later to compute global thresholds, and then digitizes each trial again in a second pass instead of discretizing in one pass.

ii. 
```python
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
session_wheel.append(wheel_trial)
session_whisker.append(whisker_trial)
wheel_continuous_all.append(wheel_trial)
whisker_continuous_all.append(whisker_trial)
...
wheel_flat = np.concatenate(wheel_continuous_all)
whisker_flat = np.concatenate(whisker_continuous_all)
...
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False)
```

iii. The trajectory does not explicitly acknowledge these repeats. They are visible only in the final implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes wheel acceleration in `velocity_filtered(...)` but never uses it. It also retains large intermediate continuous wheel/whisker buffers only to estimate global thresholds, then discards them from the final dataset. The `dropped_sessions` accounting and summary statistics are also not used downstream by the decoder itself.

ii. 
```python
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
acc = np.insert(np.diff(vel), 0, 0.0) * fs
return vel, acc
```

```python
wheel_continuous_all = []
whisker_continuous_all = []
...
wheel_continuous_all.append(wheel_trial)
whisker_continuous_all.append(whisker_trial)
...
wheel_flat = np.concatenate(wheel_continuous_all)
whisker_flat = np.concatenate(whisker_continuous_all)
```

iii. The trajectory does not justify these extra computations. They appear to be implementation conveniences rather than requirements of the reference pipeline.
