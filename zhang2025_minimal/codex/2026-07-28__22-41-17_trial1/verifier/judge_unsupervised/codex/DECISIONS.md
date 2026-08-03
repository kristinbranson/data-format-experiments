# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent starts from the full `bwm_release.csv` release table, groups rows by `eid` to form sessions, resolves each session to a local ONE-cache path, and then loads the trial table, wheel files, whisker motion-energy files, and spike-sorting outputs from ALF/pykilosort files. It does not call the reference `SessionLoader`/`SpikeSortingLoader`; it reimplements the same loading against the cached files.

ii. 
```python
release = pd.read_csv(RELEASE_CSV, index_col=0)
sessions = (
    release.groupby("eid", sort=False)
    .agg(
        {
            "subject": "first",
            "date": "first",
            "session_number": "first",
            "lab": "first",
            "probe_name": lambda x: tuple(sorted(x)),
        }
    )
    .reset_index()
)
...
session_path = find_session_path(row)
trials = load_trials_table(session_path)
wheel_times, wheel_speed = load_wheel_speed(session_path)
whisker_times, whisker_me, whisker_view = load_whisker_motion_energy(session_path)
spike_times, spike_clusters, region_labels = load_good_units(session_path, row["probe_name"], br)
```

iii. In `CONVERSION_NOTES.md` the agent says it used the full provided `bwm_release.csv` table and resolved sessions directly as `lab / Subjects / subject / date / session_number`. In the trajectory it explicitly says it chose a “standalone converter against the cached ALF files” rather than reusing the higher-level reference loaders.

## 1-b. How are the data split into subjects?

i. Subjects are split using the `subject` column from `bwm_release.csv`. After session conversion, the agent constructs a unique `subjects` list in first-seen order and a `subject_idx` array mapping each included session to that subject list.

ii. 
```python
subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])
...
"subjects": subjects,
"subject_idx": np.array([subject_to_idx[rec["subject"]] for rec in records], dtype=np.int16),
```

iii. The justification is implicit rather than argued at length: the release table already carries subject IDs, and the target decoder format requires subject names plus a per-session subject index.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `eid`. Multiple probes belonging to the same `eid` are merged into one pooled session population, and each included `eid` becomes one session in `neural`, `input`, and `output`.

ii. 
```python
sessions = (
    release.groupby("eid", sort=False)
    .agg(
        {
            "subject": "first",
            "date": "first",
            "session_number": "first",
            "lab": "first",
            "probe_name": lambda x: tuple(sorted(x)),
        }
    )
    .reset_index()
)
...
for probe_name in probe_names:
    ...
    all_times.append(spikes_times[valid][keep])
    all_clusters.append(mapped[keep])
```

iii. `CONVERSION_NOTES.md` explicitly says the agent merged all probes from the same session into a single pooled population. That mirrors the reference utility’s `merge_probes(...)` behavior.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. For each trial row, the agent defines a 2 s interval centered on stimulus onset using `stimOn_times + [-0.5, 1.5]`, bins spikes in that interval, interpolates wheel and whisker streams onto that interval, and then keeps the selected trial indices.

ii. 
```python
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
...
selected_idx = np.flatnonzero(final_mask)
neural_selected = [neural_trials[i] for i in selected_idx]
```

iii. The notes say all trials are aligned to `trials.stimOn_times` with a `[-0.5, 1.5]` s window, chosen to satisfy the task’s stimulus-aligned decoder format.

## 1-e. How are trials filtered based on quality controls?

i. The agent first applies a reference-style trial mask: required non-NaN events, reaction time in `[0.08, 2.0]` s, trial duration `<= 10 s`, and `choice != 0`. It then adds extra exclusions not present in the reference trial mask: trials whose aligned wheel or whisker streams do not fully cover the interval, and trials whose binned neural matrix is entirely zero.

ii. 
```python
def make_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
    ]
    ...
    mask &= rt >= 0.08
    mask &= rt <= 2.0
    mask &= trial_len <= 10.0
    mask &= trials["choice"].to_numpy() != 0
    for col in required:
        mask &= ~trials[col].isna().to_numpy()
    return mask
...
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
...
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. `CONVERSION_NOTES.md` says the core trial mask “matches the reference `load_trials_and_mask(...)` logic,” but also admits two additional cleanup rules: drop spike-silent trials and keep only trials with full wheel and whisker coverage. The trajectory shows the zero-spike-trial drop was added specifically to eliminate validator warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is derived from spike timestamps and spike cluster assignments from each probe, after selecting “good” clusters using `clusters.metrics.pqt`. Brain-region metadata are derived separately from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii. 
```python
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
...
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
...
cluster_channels = np.load(probe_base / "clusters.channels.npy")[good_mask].astype(np.int64)
channel_region_ids = np.load(probe_base / "channels.brainLocationIds_ccf_2017.npy")
```

iii. The notes list exactly these files under “Neuron QC and probe merging,” and explain that probe data were pooled session-wise after selecting QC-passed clusters.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, remaps cluster IDs into one pooled neuron index, sorts spikes by time, and bins spikes into per-trial count matrices of shape `(n_neurons, 100)` using 20 ms bins over a 2 s stimulus-aligned window. Counts are finally stored as `float16`.

ii. 
```python
cluster_map[good_cluster_ids] = np.arange(offset, offset + good_cluster_ids.shape[0], dtype=np.int32)
...
spike_times = np.concatenate(all_times)
spike_clusters = np.concatenate(all_clusters)
order = np.argsort(spike_times, kind="stable")
return spike_times[order], spike_clusters[order], all_regions
```

```python
times = spike_times[lo:hi] - start
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
valid = (bins >= 0) & (bins < NBINS)
flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
trials.append(counts.astype(np.float16))
```

iii. The agent justified this as matching the reference session-level probe merge and 2 s / 20 ms binned neural representation needed for the requested unified decoder dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. At the neuron level, only clusters with `label >= 1.0` are retained. Sessions with zero retained neurons are dropped. At the trial level, the agent additionally removes trials whose final binned neural matrix is all zero, even though that is not part of the reference unit QC.

ii. 
```python
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]
if good_cluster_ids.size == 0:
    continue
...
if len(region_labels) == 0:
    skip_reasons["no_good_units"] += 1
    continue
...
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The notes explicitly justify `label >= 1.0` by matching the reference good-unit count of 75,708. The extra all-zero-trial exclusion is justified only as validator cleanup.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to stimulus onset (`stimOn_times`) and cut to `[-0.5, 1.5]` s around that event, matching the instruction to align the decoder to stimulus onset.

ii. 
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)
...
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
```

iii. The notes explicitly say this was chosen to “preserve the requested stimulus alignment for all decoder variables.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins over a 2 s window, giving 100 bins per trial. There is no further temporal rebinning after spike counting. This same 20 ms grid is used for inputs and outputs.

ii. 
```python
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))
...
counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
```

iii. The notes justify this as matching the 2 s / 20 ms setup mentioned in the methods text and as a single common grid for all decoder variables. The agent did not adopt the 50 ms choice/prior binning described in the trial-aligned paper section.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a measured input stream. It is derived from the chosen alignment event `stimOn_times` plus the fixed conversion window and bin size.

ii. 
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)
BIN_SIZE_S = 0.02
...
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The notes describe this row as “time since stimulus onset in seconds” with bin end times from `-0.48` to `1.50`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent constructs a synthetic time vector of bin-end times, not a binary series or an event-only marker. The exact values are identical for every kept trial.

ii. 
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
...
input_trials = [
    np.vstack(
        [
            time_input,
            np.full(NBINS, trial_number[i], dtype=np.float32),
        ]
    ).astype(np.float32)
    for i in selected_idx
]
```

iii. The notes justify this as storing a continuous time-since-stimulus row over the shared 100-bin stimulus-aligned grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned exactly to the neural bins by using the same `WINDOW`, `BIN_SIZE_S`, and `NBINS`, with time values representing the bin end times corresponding to each neural count bin.

ii. 
```python
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))
...
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly states that the row uses “bin end times,” which mirrors the interpolation grid used for behavior and the bin count grid used for neural data.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial-wise `probabilityLeft` column in the trial table.

ii. 
```python
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
```

iii. The notes say trial number within block “resets to `1` whenever `probabilityLeft` changes.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent computes the run length of consecutive trials with the same `probabilityLeft`, resetting to `1` whenever the block value changes. The resulting scalar is then repeated across all 100 time bins for that trial.

ii. 
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.ones(prob_left.shape[0], dtype=np.float32)
    curr = 1.0
    for i in range(1, prob_left.shape[0]):
        if np.isclose(prob_left[i], prob_left[i - 1]):
            curr += 1.0
        else:
            curr = 1.0
        trial_num[i] = curr
    return trial_num
```

```python
np.full(NBINS, trial_number[i], dtype=np.float32)
```

iii. The justification in the notes is minimal but clear: the target format asks for “trial number within the current `probabilityLeft` block.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii. 
```python
choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The notes explicitly map IBL `choice == -1` to left and `choice == +1` to right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps raw choices from `{-1, +1}` to `{0, 1}` as requested, rejects unexpected values, and then repeats the per-trial category across all 100 time bins in the output tensor.

ii. 
```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0
    out[np.isclose(values, 1.0)] = 1
    if np.any(out < 0):
        raise ValueError("Unexpected choice values encountered")
    return out
```

```python
out = np.vstack(
    [
        np.full((1, NBINS), choice, dtype=np.int8),
        ...
    ]
)
```

iii. The notes justify the value remapping directly from the decoder specification. They do not separately justify repeating the value across time; that appears to be a format choice so every output trial has shape `(4, 100)`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. 
```python
prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The notes describe this as mapping trial-wise `probabilityLeft` values into the requested categorical labels.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent rounds `probabilityLeft` to one decimal place, maps `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, errors on anything else, and repeats that category across all 100 time bins.

ii. 
```python
def map_probability_left(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values.astype(np.float64), 1)
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(rounded, 0.2)] = 0
    out[np.isclose(rounded, 0.5)] = 1
    out[np.isclose(rounded, 0.8)] = 2
    if np.any(out < 0):
        raise ValueError("Unexpected probabilityLeft values encountered")
    return out
```

iii. The notes justify the mapping from the decoder task. As with choice, they do not explicitly justify the decision to repeat a static trial variable across all bins.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. 
```python
timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
```

iii. The notes say the wheel target follows the same IBL wheel preprocessing path as the provided code.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent interpolates wheel position to 1000 Hz using IBL wheel code, computes filtered velocity with `velocity_filtered(...)`, takes the absolute value to make speed, and then linearly interpolates the continuous speed trace into each stimulus-aligned trial interval.

ii. 
```python
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

```python
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
wheel_cont = [np.asarray(x, dtype=np.float32) for x in wheel_selected]
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as following the provided IBL wheel implementation: interpolate to 1000 Hz, low-pass/Butterworth velocity, then absolute value.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent computes one pair of global tercile thresholds across all included wheel-speed time bins, then applies `np.digitize` per trial to label each time bin as low, medium, or high.

ii. 
```python
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)
...
def digitize_tertiles(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.int8)
...
wheel_disc = digitize_tertiles(wheel, wheel_thresholds)
```

iii. The notes justify global tertiles as an invented discretization needed by the task: preserve rank information, avoid session-specific drift, and keep classes roughly balanced.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The continuous wheel-speed trace is sliced into the same stimulus-aligned intervals as the neural data and linearly interpolated to the same 100 bin-end timestamps. The final categorical wheel output therefore has one value per neural bin.

ii. 
```python
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
...
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The notes say all decoder variables were put on the same stimulus-aligned 2 s / 20 ms grid to satisfy the instruction, even though the methods paper used first movement for the dynamic behaviors.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from camera motion-energy timestamps and motion-energy arrays, preferring left whisker camera files and falling back to right whisker camera files.

ii. 
```python
times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
...
times, values = _load_camera_stream(session_path, "left")
...
times, values = _load_camera_stream(session_path, "right")
```

iii. The notes justify the left-preferred, right-fallback policy as matching the provided code and materially affecting session retention.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the whisker motion-energy trace, trims timestamps if they are longer than the values array, rejects the stream if timestamps are shorter, chooses left or right camera per the fallback rule, and then linearly interpolates the continuous trace into each stimulus-aligned trial interval.

ii. 
```python
if times.shape[0] < values.shape[0]:
    raise ValueError(f"{view} camera timestamps shorter than motion-energy array")
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
return times.astype(np.float32), values.astype(np.float32)
```

```python
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
whisker_cont = [np.asarray(x, dtype=np.float32) for x in whisker_selected]
```

iii. `CONVERSION_NOTES.md` gives this exact justification, including why the timestamp trim and camera fallback were necessary.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The agent computes one pair of global tercile thresholds over all included whisker-motion-energy time bins, then digitizes each interpolated time series into low, medium, or high.

ii. 
```python
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
...
whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
```

iii. The notes give the same justification as for wheel speed: global tertiles were an added discretization choice required by the categorical decoder output format.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The continuous whisker signal is interpolated into the same stimulus-aligned trial intervals and onto the same 100 bin-end timestamps as the neural data, then discretized per time bin.

ii. 
```python
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
...
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The notes explicitly say the agent forced whisker motion energy onto the same stimulus-aligned grid as the neural data because the task required stimulus alignment for all decoder variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or malformed data by exclusion and fallback: missing required files skip the session, left whisker-camera failures trigger right-camera fallback, NaNs in required trial events drop the trial, NaNs or incomplete coverage in interpolated wheel/whisker intervals drop the trial, and timestamp/value length mismatches in whisker data are either trimmed or rejected.

ii. 
```python
except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1
```

```python
try:
    times, values = _load_camera_stream(session_path, "left")
    return times, values, "left"
except Exception:
    times, values = _load_camera_stream(session_path, "right")
    return times, values, "right"
```

```python
if t.shape[0] == 0:
    outputs.append(None)
    continue
if np.isnan(y).any():
    outputs.append(None)
    continue
if np.abs(beg - t[0]) > BIN_SIZE_S or np.abs(end - t[-1]) > BIN_SIZE_S:
    outputs.append(None)
    continue
```

iii. The notes justify these choices as necessary to produce a clean stimulus-aligned decoder dataset from the local cache. The trajectory also shows the agent chose exclusion rather than downloading missing sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the per-session loading of large spike and behavior arrays, the per-trial spike binning loop over all aligned intervals, and the per-trial interpolation of wheel and whisker streams. The logs show the full conversion takes roughly 14 to 17 minutes for 459 candidate sessions.

ii. 
```python
for idx, row in enumerate(sessions.to_dict("records"), start=1):
    ...
    spike_times, spike_clusters, region_labels = load_good_units(session_path, row["probe_name"], br)
    ...
    neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
    wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
    whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. This is inferred from the structure of the code and the runtime logs in `conversion_full_out.txt`, rather than from an explicit justification note.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearly vectorizable Python loops are `compute_trial_number_in_block`, the per-trial construction of `input_trials`, the per-trial `build_output_trials` loop, and likely parts of `interpolate_behavior_into_trials`. `bin_spikes` is already partly vectorized internally with `np.bincount`, but it still loops over trials in Python.

ii. 
```python
for i in range(1, prob_left.shape[0]):
    if np.isclose(prob_left[i], prob_left[i - 1]):
        curr += 1.0
    else:
        curr = 1.0
```

```python
input_trials = [
    np.vstack([...]).astype(np.float32)
    for i in selected_idx
]
```

```python
for choice, prior, wheel, whisker in zip(...):
    wheel_disc = digitize_tertiles(wheel, wheel_thresholds)
    whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
    ...
```

iii. This is an evaluation of the code structure; the agent did not document these optimization opportunities itself.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly performs path globbing and array loads per session and per probe, repeatedly rebuilds the same interpolation grid inside `interpolate_behavior_into_trials`, and separately interpolates wheel and whisker using identical logic. It also stores continuous wheel/whisker trial arrays once for global threshold estimation and then traverses them again when building final outputs.

ii. 
```python
matches = sorted(session_path.glob(pattern))
...
timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
```

```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

```python
wheel_pool.append(np.concatenate(wheel_cont))
whisker_pool.append(np.concatenate(whisker_cont))
...
"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

iii. This is again an evaluation from reading the code, not something the agent explicitly documented.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is that the converter keeps continuous per-trial `wheel_cont` and `whisker_cont` arrays in `records`, concatenates them into global pools to compute tertile thresholds, and then drops those continuous values from the final saved dataset. It also computes and stores large metadata structures that the downstream decoder does not use.

ii. 
```python
records.append(
    {
        ...
        "wheel_cont": wheel_cont,
        "whisker_cont": whisker_cont,
        ...
    }
)
...
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
...
"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

iii. This is not described in the notes as unnecessary, but it is apparent from the implementation: the final pickle contains only discretized wheel/whisker outputs, not the continuous traces used to derive them.
