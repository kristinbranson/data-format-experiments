# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the release inventory from `bwm_release.csv`, groups rows by `eid` to define sessions, resolves each session to a local `ONE_CACHE_DIR/<lab>/Subjects/<subject>/<date>/<session_number>/alf` path, then loads trials, wheel, whisker-motion, and per-probe spike-sorting files from that session directory.

ii.
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

session_path = (
    ONE_CACHE_DIR / session["lab"] / "Subjects" / session["subject"]
    / session["date"] / f"{session['session_number']:03d}"
)
alf_path = session_path / "alf"
trials_df = load_trials_table(alf_path)
wheel_times, wheel_speed = load_wheel_speed(alf_path)
whisker_times, whisker_me, motion_view = load_motion_energy(alf_path)
```

iii. `CONVERSION_NOTES.md` says session discovery comes from `code/code_zhang2025/data/bwm_release.csv` and local session paths under `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/`. Trajectory step 11 shows the reference `0_data_caching.py` also starts from `data/bwm_release.csv`.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the release CSV. After session conversion, the code takes the ordered unique subject names and stores a `subject_idx` per kept session.

ii.
```python
subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.asarray(
    [subject_lookup[session["subject"]] for session in kept_sessions],
    dtype=np.int64,
),
```

iii. The justification is implicit: the release table already carries `subject`, and trajectory step 11 shows the reference code sampling or grouping sessions by `bwm_df.subject`.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. All probes listed for the same `eid` are merged into one session before decoding.

ii.
```python
for eid, group in release_df.groupby("eid", sort=False):
    ...
    "probe_names": list(group["probe_name"]),

probe_infos = []
for probe_name in session["probe_names"]:
    probe_infos.append(load_probe_good_units(alf_path / probe_name, brain_regions))
merged = merge_session_probes(probe_infos)
```

iii. `CONVERSION_NOTES.md` explicitly says probes from the same session were merged, and trajectory steps 11 and 15 show the reference pipeline calling `prepare_data(...)` per `eid` and merging probes via `merge_probes(...)`.

## 1-d. How are the data split into trials?

i. Within each kept session, the code iterates over rows of the session trial table. Each retained row becomes one 2 s trial window aligned to `stimOn_times`, with neural and behavior sliced/interpolated onto that window.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows():
    if not trial_mask[trial_idx]:
        continue
    trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
    trial_end = float(trial_row["stimOn_times"] + OFF_END_S)

    wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
    whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
    neural_trial = bin_spikes_for_trial(..., trial_start)
```

iii. `CONVERSION_NOTES.md` says trials use a `stimOn_times`-aligned `[-0.5 s, +1.5 s]` window. Trajectory steps 11 and 14 show the reference code building trial intervals as `trials_df[align_time] + time_window`.

## 1-e. How are trials filtered based on quality controls?

i. The explicit trial mask requires finite `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, and `goCue_times`; reaction time between `0.08` and `2.0` s; trial length `feedback_times - goCue_times <= 10 s`; and `choice != 0`. During trial extraction, the code also drops trials whose wheel or whisker trace cannot cover the whole interval, and trials with zero spikes across all retained neurons in the 2 s window.

ii.
```python
required = [
    "stimOn_times", "choice", "feedback_times", "probabilityLeft",
    "firstMovement_times", "feedbackType", "goCue_times",
]
...
mask &= rt >= MIN_RT_S
mask &= rt <= MAX_RT_S
mask &= trial_len <= MAX_TRIAL_LEN_S
mask &= trials_df["choice"].to_numpy() != 0
...
if wheel_trial is None or whisker_trial is None:
    continue
...
if not np.any(neural_trial):
    continue
```

iii. `CONVERSION_NOTES.md` ties the main mask to `load_trials_and_mask(...)` and `prepare_data(...)` from the reference code, and trajectory step 13 shows the reference default mask fields and RT limits. The extra `goCue_times` finite check and zero-spike trial drop are the agent’s additions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from spike-sorting outputs: `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
metrics_path = latest_revision_file(sorter_base, "#*/clusters.metrics.pqt")
spike_times_path = latest_revision_file(sorter_base, "#*/spikes.times.npy")
spike_clusters_path = latest_revision_file(sorter_base, "#*/spikes.clusters.npy")
cluster_channels_path = latest_revision_file(sorter_base, "#*/clusters.channels.npy")
channel_regions_path = latest_revision_file(
    sorter_base, "#*/channels.brainLocationIds_ccf_2017.npy"
)
```

iii. `CONVERSION_NOTES.md` explicitly lists the spike and cluster files used, and trajectory steps 13 and 15 show the reference code loading spiking data from `SpikeSortingLoader` and carrying cluster acronyms forward.

## 2-b. How is the `neural` data processed?

i. For each session, the code keeps selected clusters, merges probes, sorts spikes by time, then bins spikes into 20 ms counts across a 2 s window per trial. The saved trial matrices are dense `(n_neurons, 100)` spike-count arrays stored as `float16`.

ii.
```python
merged = merge_session_probes(probe_infos)
...
trial_end = trial_start + binsize * n_bins
start_idx = np.searchsorted(spike_times, trial_start, side="left")
end_idx = np.searchsorted(spike_times, trial_end, side="left")
...
counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
return counts.astype(np.float16)
```

iii. `CONVERSION_NOTES.md` says probes were merged before decoding and neural arrays were stored as dense `float16` spike-count matrices. Trajectory steps 11, 13, and 14 show the reference pipeline also merging probes and binning trial-aligned spikes into 20 ms bins over 2 s.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only clusters with `label >= 1`, excludes clusters whose region acronym is `void` or `root`, and drops trials with no spikes from the retained neurons.

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

iii. `CONVERSION_NOTES.md` justifies `label >= 1` by citing the paper’s “well-isolated neurons” language and a release-wide sanity check near 108 good units per probe. Trajectory steps 21 and 58 show the agent explicitly debating this because the reference Zhang code path appeared to load all clusters, not only QC-passing ones.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus onset by defining each trial window as `stimOn_times + [-0.5, 1.5]` and binning spikes relative to that window.

ii.
```python
OFF_START_S = -0.5
OFF_END_S = 1.5
...
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
neural_trial = bin_spikes_for_trial(..., trial_start)
```

iii. `CONVERSION_NOTES.md` says the alignment event is `stimOn_times` with `[-0.5 s, +1.5 s]`. Trajectory step 11 shows the same defaults in the reference `0_data_caching.py`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins over a 2 s interval, yielding 100 bins per trial. No additional temporal rebinning is applied after that.

ii.
```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

iii. `CONVERSION_NOTES.md` and trajectory step 11 both cite the reference parameters `interval_len = 2`, `binsize = 0.02`, and `time_window = (-.5, 1.5)`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a raw timeseries. It is synthesized from the trial alignment event `stimOn_times` plus the fixed window constants `OFF_START_S`, `OFF_END_S`, and `BIN_SIZE_S`.

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. `CONVERSION_NOTES.md` says this input is “represented at the right edge of each bin, from `-0.48 s` to `1.50 s`.” There is no separate raw variable for it in the trajectory or notes.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code builds a fixed 100-sample ramp using `np.linspace`, one value per 20 ms bin, at the right edge of each bin. The same ramp is reused for every retained trial.

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

iii. The justification comes from `CONVERSION_NOTES.md`, which explicitly says the representation is time-varying, one value per 20 ms bin, and uses the bin right edges.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction: it uses the same number of bins, the same 20 ms spacing, and the same stimulus-onset-centered interval as the neural trial matrix.

ii.
```python
neural_trial = bin_spikes_for_trial(..., trial_start)
...
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. `CONVERSION_NOTES.md` states that the variable is aligned to the same `stimOn_times` window used for neural activity.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial-table column `probabilityLeft`.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(
    trials_df["probabilityLeft"].to_numpy()
)
```

iii. `CONVERSION_NOTES.md` says it is a “1-based count of the trial index within the current `probabilityLeft` block.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans `probabilityLeft` in original session order, increments a counter while consecutive values stay the same, resets the counter when the block value changes, then repeats that scalar across all 100 time bins of each retained trial.

ii.
```python
for idx, value in enumerate(values):
    same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
    count = count + 1 if same_block else 1
    numbers[idx] = float(count)
    previous = value
...
np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says this was computed on the original session trial order before masking and then repeated across time within retained trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived directly from the trial-table `choice` column.

ii.
```python
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. `CONVERSION_NOTES.md` says the output is mapped from IBL `choice`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps IBL choice coding `1 -> left -> 0` and `-1 -> right -> 1`, then repeats that class across all 100 time bins for the trial.

ii.
```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0
    if np.isclose(value, -1.0):
        return 1
...
np.full(N_BINS, choice_class, dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` documents the mapping explicitly. The mapping is also consistent with the raw trial examples the agent inspected in trajectory step 50.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trial-table `probabilityLeft` column.

ii.
```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. `CONVERSION_NOTES.md` says the output comes from the per-trial prior-left probability block.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code discretizes `probabilityLeft` by exact-value matching with `np.isclose`: `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. It then repeats that class across all 100 time bins for the trial.

ii.
```python
mapping = {0.2: 0, 0.5: 1, 0.8: 2}
for key, encoded in mapping.items():
    if np.isclose(value, key):
        return encoded
...
np.full(N_BINS, prior_class, dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` documents the same class mapping and says static outputs were repeated across time so all outputs share shape `(d_output, T)`.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from raw wheel timestamps and wheel position arrays: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps_path = latest_revision_file(alf_path, ["_ibl_wheel.timestamps.npy", "#*/_ibl_wheel.timestamps.npy"])
position_path = latest_revision_file(alf_path, ["_ibl_wheel.position.npy", "#*/_ibl_wheel.position.npy"])
...
timestamps = np.load(timestamps_path).astype(np.float64)
position = np.load(position_path).astype(np.float64)
```

iii. `CONVERSION_NOTES.md` explicitly lists those two files as the wheel source.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code linearly interpolates wheel position to 1000 Hz, computes low-pass-filtered velocity with an 8th-order 20 Hz Butterworth filter, takes the absolute value to make speed, then linearly interpolates that speed onto the 20 ms trial bins.

ii.
```python
position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity).astype(np.float32)
...
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the wheel processing matches `SessionLoader.load_wheel(...)` and uses absolute velocity as the decoded quantity. Trajectory step 48 shows the same filter parameters in `brainbox.io.one.SessionLoader.load_wheel`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained wheel-speed values across all sessions, trials, and time bins are pooled. The code computes global 1/3 and 2/3 quantiles and bins each wheel-speed sample into `low`, `medium`, or `high` with `np.digitize`.

ii.
```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
...
q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
bins = np.digitize(values, [q1, q2], right=False).astype(np.int8)
...
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
```

iii. `CONVERSION_NOTES.md` says `wheel_speed_bin` uses global tertiles across all retained wheel-speed time bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each retained trial, the code extracts the wheel segment over the same `stimOn_times + [-0.5, 1.5]` interval and linearly interpolates it to the same 100 bin centers used for neural data.

ii.
```python
wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
...
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says trial values were resampled onto the 20 ms decoder bins using the same interval-coverage checks as `get_behavior_per_interval(...)`. Trajectory step 14 shows the reference interpolation rule uses bin right edges and start/end coverage checks.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy` plus the matching camera timestamp files `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

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

iii. `CONVERSION_NOTES.md` explicitly lists those files and says left camera is preferred with right-camera fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the first available side camera, trims extra timestamps from the front if the timestamp vector is longer than the motion-energy array, then linearly interpolates motion-energy values onto the 20 ms trial bins.

ii.
```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
...
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. `CONVERSION_NOTES.md` says the timestamp trimming matches `_check_video_timestamps(...)` and interpolation is linear onto the decoder bins. Trajectory step 48 shows `SessionLoader.load_motion_energy` calling `_check_video_timestamps(...)`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, all retained whisker-motion samples are pooled globally, thresholded at the 1/3 and 2/3 quantiles, and assigned to `low`, `medium`, or `high`.

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

iii. `CONVERSION_NOTES.md` says `whisker_motion_energy_bin` is derived from global tertiles across all retained whisker-motion time bins.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. For each retained trial, the code uses the same stimulus-onset-aligned 2 s interval as the neural data and interpolates whisker motion to the same 100 time bins.

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. `CONVERSION_NOTES.md` says whisker values were linearly interpolated onto the same 20 ms decoder bins used for neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing required trial fields cause trial exclusion. Missing wheel or whisker coverage for a trial causes that trial to be skipped. Missing entire wheel or whisker streams, or missing spike-sorting files, cause the session to be dropped. If camera timestamps are longer than motion-energy data, the code trims the leading timestamps. If timestamps are shorter than motion-energy data, it raises an error. Unknown `probabilityLeft` or `choice` values also raise errors.

ii.
```python
for column in required:
    mask &= np.isfinite(trials_df[column].to_numpy())
...
if wheel_trial is None or whisker_trial is None:
    continue
...
if timestamps.shape[0] < motion_energy.shape[0]:
    raise ValueError(f"{view} camera timestamps shorter than motion energy")
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
...
except Exception as exc:
    dropped_sessions.append((session["eid"], f"error:{type(exc).__name__}:{exc}"))
```

iii. `CONVERSION_NOTES.md` says missing motion energy or wheel files were major session-drop reasons and explicitly notes the timestamp-trimming rule. The rest is implicit in the implementation.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive steps are the outer loop over all sessions, the inner loop over all retained trials, per-trial behavior interpolation, and especially per-trial spike binning across all retained neurons. Global concatenation and quantile computation for wheel and whisker outputs are also nontrivial.

ii.
```python
for session_idx, session in enumerate(session_rows, start=1):
    ...
    for trial_idx, trial_row in trials_df.iterrows():
        ...
        wheel_trial = interpolate_behavior_trial(...)
        whisker_trial = interpolate_behavior_trial(...)
        neural_trial = bin_spikes_for_trial(...)
...
wheel_flat = np.concatenate(wheel_continuous_all)
whisker_flat = np.concatenate(whisker_continuous_all)
```

iii. There is no explicit efficiency justification in `CONVERSION_NOTES.md`; this assessment is inferred from the implementation and the scale reported there (`439` sessions, `186,953` trials).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The session loop must stay, but several inner loops could be vectorized: the `compute_trial_mask` finite-column loop, the `compute_trial_number_in_block` scan, the per-trial `iterrows()` extraction loop, repeated `np.full(...)` construction for static outputs, and repeated `bin_spikes_for_trial(...)` calls that could instead bin all trial intervals in one pass like the reference `bin_spiking_data(...)`.

ii.
```python
for column in required:
    mask &= np.isfinite(trials_df[column].to_numpy())
...
for idx, value in enumerate(values):
    ...
for trial_idx, trial_row in trials_df.iterrows():
    ...
for static_values, wheel_trial, whisker_trial in zip(...):
    ...
```

iii. No explicit justification was recorded. This is an audit observation based on the code and on trajectory step 14, where the reference helper bins spikes across many intervals together.

## 10-c. What processing does the code repeat multiple times?

i. It recreates the same `time_since_stim` vector for every retained trial, performs separate `np.searchsorted` work for wheel and whisker on every trial, constructs repeated `np.full(...)` arrays for static outputs and trial-number inputs on every trial, and repeatedly scans the revisioned filesystem for each file lookup.

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS, dtype=np.float32
)
...
start_idx = np.searchsorted(sample_times, interval_start, side="right")
end_idx = np.searchsorted(sample_times, interval_end, side="left")
...
np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32)
np.full(N_BINS, choice_class, dtype=np.int8)
np.full(N_BINS, prior_class, dtype=np.int8)
```

iii. No explicit justification was recorded. The repetition is visible directly in the implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps full continuous wheel-speed and whisker-motion trial traces in memory only to compute global quantile thresholds and then discard them from the saved dataset. It also expands static trial labels (`choice`, `prior`) to 100-bin time series even though they are constant within trial.

ii.
```python
wheel_continuous_all.append(wheel_trial)
whisker_continuous_all.append(whisker_trial)
...
wheel_flat = np.concatenate(wheel_continuous_all)
whisker_flat = np.concatenate(whisker_continuous_all)
...
output_trial = np.vstack([
    np.full(N_BINS, choice_class, dtype=np.int8),
    np.full(N_BINS, prior_class, dtype=np.int8),
    wheel_bins,
    whisker_bins,
])
```

iii. `CONVERSION_NOTES.md` justifies the time-expanded static outputs as a shape-unification choice, not as reference processing. There is no explicit efficiency justification for retaining the continuous traces after discretization thresholds are computed.
