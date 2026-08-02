# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter starts from the release table `bwm_release.csv`, groups rows by `eid` so all probes for one session are merged, resolves each session to a local ONE-cache path, and then reads ALF trial, wheel, camera, and spike-sorting files directly from disk.

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

def find_session_path(row: pd.Series) -> Path:
    return DATA_ROOT / row["lab"] / "Subjects" / row["subject"] / row["date"] / f"{int(row['session_number']):03d}"

trials = load_trials_table(session_path)
wheel_times, wheel_speed = load_wheel_speed(session_path)
whisker_times, whisker_me, whisker_view = load_whisker_motion_energy(session_path)
spike_times, spike_clusters, region_labels = load_good_units(session_path, row["probe_name"], br)
```

iii. `CONVERSION_NOTES.md` says the agent intentionally used the full local `bwm_release.csv` release and resolved session paths as `lab / Subjects / subject / date / session_number` so it could reproduce the reference processing from cached ALF files without remote ONE calls.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the session records' `subject` field. The output `subjects` list stores unique subject IDs in first-seen order, and `subject_idx` maps each included session back to that list.

ii.
```python
subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])

"subjects": subjects,
"subject_idx": np.array([subject_to_idx[rec["subject"]] for rec in records], dtype=np.int16),
```

iii. The notes emphasize release-level subject counts and report that the included dataset ends with 135 subjects after session exclusions, so the subject split is driven by included sessions rather than by a separate mouse-level file.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the release table. Multiple probe rows for the same `eid` are collapsed into one session record, and all probes for that session are merged later into one pooled neural population.

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
```

iii. `CONVERSION_NOTES.md` explicitly says the converter used the full release table and that all probes from the same session were merged into a single pooled population.

## 1-d. How are the data split into trials?

i. Trials are the rows of the session trial parquet table. For each session the code builds one aligned 2 s interval per trial around `stimOn_times`, then keeps only the trials passing the final mask.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    return pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))

stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)

selected_idx = np.flatnonzero(final_mask)
neural_selected = [neural_trials[i] for i in selected_idx]
```

iii. The trajectory says the agent settled on “stimulus-aligned 2 s trials” as the unified trial format, and the notes repeat that all decoder variables were organized into `[-0.5, 1.5]` s trials around stimulus onset.

## 1-e. How are trials filtered based on quality controls?

i. The converter first applies the same reference-style trial mask as `load_trials_and_mask(...)`: required trial events must be non-NaN, reaction time must be 0.08 to 2.0 s, trial duration must be at most 10 s, and `choice == 0` trials are dropped. It then adds extra filters not in the reference mask: trials are removed if wheel coverage fails, whisker coverage fails, or the final QC-passed neural population has zero spikes in the aligned window.

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
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    trial_len = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    mask &= rt >= 0.08
    mask &= rt <= 2.0
    mask &= trial_len <= 10.0
    mask &= trials["choice"].to_numpy() != 0
    for col in required:
        mask &= ~trials[col].isna().to_numpy()
    return mask

neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The notes state that the main trial mask was chosen to match `ibl_data_utils.py`, but they also justify extra cleanup as necessary to avoid validator problems and to ensure wheel and whisker are present on every retained trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from spike times and cluster identities for QC-passed units, plus cluster-to-channel and channel-to-region mappings. The relevant raw arrays are `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
cluster_channels = np.load(probe_base / "clusters.channels.npy")[good_mask].astype(np.int64)
channel_region_ids = np.load(probe_base / "channels.brainLocationIds_ccf_2017.npy")

spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

iii. The notes list exactly these files under “Neuron QC and probe merging” and say the goal was to reproduce the release-level good-unit accounting while pooling all probes within a session.

## 2-b. How is the `neural` data processed?

i. Spike trains from all good clusters in all probes of a session are merged, sorted by time, and binned into per-trial spike-count matrices with shape `(n_neurons, 100)` using 20 ms bins across a 2 s stimulus-aligned window.

ii.
```python
BIN_SIZE_S = 0.02
WINDOW = (-0.5, 1.5)
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))

spike_times = np.concatenate(all_times)
spike_clusters = np.concatenate(all_clusters)
order = np.argsort(spike_times, kind="stable")

times = spike_times[lo:hi] - start
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
```

iii. The trajectory says the agent deliberately chose a “stimulus-aligned 2 s trial, 20 ms bin” unified representation, and the notes say this was meant to preserve the requested stimulus alignment across all decoder variables.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The converter retains only clusters with `clusters.metrics.label >= 1.0`, skips sessions with no surviving units, and then further rejects individual trials whose aligned neural matrix is all zeros.

ii.
```python
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]
if good_cluster_ids.size == 0:
    continue

if len(region_labels) == 0:
    skip_reasons["no_good_units"] += 1
    continue

neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
```

iii. `CONVERSION_NOTES.md` says the agent chose `label >= 1.0` to match the reference good-unit logic and explicitly notes that it did not apply the paper’s later region-level grey-matter and minimum-neuron filters because the target dataset pooled neurons session-wide.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `trials.stimOn_times`. The neural window runs from 0.5 s before to 1.5 s after that event.

ii.
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)

stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
```

iii. Both the trajectory and the notes explicitly say the unified decoder dataset was built around stimulus onset because the task instructions required stimulus-based alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 20 ms bins, giving 100 bins over 2 s. No later rebinning is applied; the 20 ms count matrices are stored directly.

ii.
```python
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))

counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
trials.append(counts.astype(np.float16))
```

iii. The notes justify 20 ms bins as matching the unified 2 s / 20 ms setup the agent adopted for the decoder-format dataset.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a dedicated raw field. It is generated from the alignment choice (`stimOn_times`) plus the fixed conversion constants `WINDOW`, `BIN_SIZE_S`, and `NBINS`.

ii.
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))

time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The notes describe this input as “bin end times” relative to stimulus onset, so the agent treated it as a constructed decoder covariate rather than a raw measurement stream.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates a length-100 vector of bin end times from `-0.48` to `1.50` s and uses it unchanged for every kept trial.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)

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

iii. `CONVERSION_NOTES.md` explicitly says row 0 of each input array is “bin end times: `[-0.48, -0.46, ..., 1.50]`”.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses exactly the same 100 bins as the neural matrices, so each time entry corresponds to one neural count bin in the stimulus-aligned window.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)

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

iii. The notes say the input arrays are `2 x 100` and are stored for every kept trial in the same stimulus-aligned frame as the neural data.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived entirely from `trials["probabilityLeft"]`.

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

iii. The notes justify this as “trial number within the current `probabilityLeft` block” and say it resets whenever `probabilityLeft` changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code counts consecutive trials with the same `probabilityLeft` value, resets to 1 when the value changes, and then repeats the resulting scalar across all 100 time bins in each kept trial.

ii.
```python
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))

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

iii. The notes explicitly describe this reset-on-change rule and present the trial number as a session-context decoder input rather than a reference paper variable.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials["choice"]`.

ii.
```python
choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The notes say the output “choice” comes from the IBL `choice` field and is converted to the requested left/right coding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The converter maps raw IBL choices with `choice == -1` to decoder class `0` (“left”) and `choice == +1` to decoder class `1` (“right”), then repeats that categorical value across all 100 bins of the trial.

ii.
```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0
    out[np.isclose(values, 1.0)] = 1
    if np.any(out < 0):
        raise ValueError("Unexpected choice values encountered")
    return out

np.full((1, NBINS), choice, dtype=np.int8),
```

iii. `CONVERSION_NOTES.md` states this mapping explicitly. There is no additional justification beyond matching the requested output coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials["probabilityLeft"]`.

ii.
```python
prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The notes say the converter used the block prior already present in the trials table and mapped it into the three requested categories.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code rounds `probabilityLeft` to one decimal place, maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, and then repeats that categorical value across all 100 bins of the trial.

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

np.full((1, NBINS), prior, dtype=np.int8),
```

iii. The notes justify this as the requested decoder-output discretization for the block prior.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from raw wheel timestamps and wheel positions: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
```

iii. The notes say wheel preprocessing was intentionally based on the same IBL wheel-loading path as the reference code.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The code interpolates wheel position to 1000 Hz, computes filtered velocity with `velocity_filtered(...)`, takes the absolute value to obtain speed, interpolates that continuous speed trace into each stimulus-aligned trial window, and stores it temporarily before discretization.

ii.
```python
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)

wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
wheel_cont = [np.asarray(x, dtype=np.float32) for x in wheel_selected]
```

iii. The notes justify this as reuse of the IBL wheel-processing path and say the decoder target specifically uses absolute velocity.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, the converter pools every retained wheel-speed sample from every retained trial, computes global one-third and two-third quantiles, and uses `np.digitize` to assign each time bin to low, medium, or high.

ii.
```python
wheel_pool.append(np.concatenate(wheel_cont))
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)

def digitize_tertiles(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.int8)

wheel_disc = digitize_tertiles(wheel, wheel_thresholds)
```

iii. The notes say this was a deliberate adaptation because the reference code treats wheel speed as continuous, while the task required a 3-class categorical output. The agent justified global tertiles as preserving rank information and avoiding session-specific label drift.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Continuous wheel speed is interpolated onto the same 100 stimulus-aligned bin end times used for neural data, and only trials with full wheel coverage over that `[-0.5, 1.5]` s window are kept.

ii.
```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
interp = np.interp(x_interp, rel_t, y).astype(np.float32)

wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The notes say this was done to preserve the requested stimulus alignment across all variables in the unified decoder dataset.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from precomputed camera motion-energy arrays and their timestamps: preferred left-camera `*_ibl_leftCamera.times.npy` with `leftCamera.ROIMotionEnergy.npy`, or right-camera equivalents if the left stream is unavailable.

ii.
```python
times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))

def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        times, values = _load_camera_stream(session_path, "left")
        return times, values, "left"
    except Exception:
        times, values = _load_camera_stream(session_path, "right")
        return times, values, "right"
```

iii. The notes say the agent deliberately preferred the left whisker camera and fell back to the right camera because the provided code uses that policy and otherwise some sessions would be lost.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The converter uses the precomputed motion-energy stream directly rather than recomputing motion energy from video frames. If the timestamp array is longer than the motion-energy array, it trims timestamps from the front so the lengths match; if timestamps are shorter than values, it rejects that stream. The resulting continuous trace is then interpolated into each stimulus-aligned trial.

ii.
```python
def _load_camera_stream(session_path: Path, view: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
    values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
    if times.shape[0] < values.shape[0]:
        raise ValueError(f"{view} camera timestamps shorter than motion-energy array")
    if times.shape[0] > values.shape[0]:
        times = times[-values.shape[0]:]
    return times.astype(np.float32), values.astype(np.float32)

whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. The notes justify the left/right fallback explicitly and also say the timestamp trimming was chosen to handle the cached files in the same way the agent believed the provided code behaved.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is pooled over all retained time bins in all retained sessions, split at the global 1/3 and 2/3 quantiles, and digitized into low, medium, and high.

ii.
```python
whisker_pool.append(np.concatenate(whisker_cont))
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)

whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
```

iii. The notes give the same justification as for wheel speed: the reference target is continuous, but the requested decoder dataset demanded 3 categorical bins, so the agent chose global tertiles.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The continuous whisker motion-energy trace is linearly interpolated onto the same 100 stimulus-aligned bin end times used for neural and wheel data, and trials without full whisker coverage are excluded.

ii.
```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
interp = np.interp(x_interp, rel_t, y).astype(np.float32)

whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The notes explicitly say all decoder variables were forced into the common stimulus-aligned window, and whisker trials were kept only if the stream covered the full aligned interval.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles data issues by choosing the latest file revision in a session folder, skipping sessions with missing required files, falling back from left to right whisker camera, trimming overlong timestamp arrays for motion energy, rejecting streams with NaNs or incomplete coverage, and dropping trials or sessions that become too sparse after these checks.

ii.
```python
def pick_latest(session_path: Path, pattern: str) -> Path:
    matches = sorted(session_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {session_path}")
    return matches[-1]

if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]

except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1

if final_mask.sum() < 2:
    skip_reasons["too_few_trials_after_stream_alignment"] += 1
    continue
```

iii. The notes justify these choices pragmatically: preserve as many sessions as possible while still satisfying the fixed-format decoder validator. They also document the skipped sessions rather than downloading missing assets.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive work is the full session loop over all release sessions, especially per-trial spike binning, per-trial wheel interpolation, and per-trial whisker interpolation. Each of these processes every trial window for every included session.

ii.
```python
for idx, row in enumerate(sessions.to_dict("records"), start=1):
    ...
    neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
    wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
    whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. The agent did not justify runtime hotspots explicitly in the notes. This is inferred from the implementation and from the trajectory comment that full runs were dominated by long validation/training passes after conversion.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops are still scalar or per-trial: `compute_trial_number_in_block`, the trial loop in `bin_spikes`, the trial loop in `interpolate_behavior_into_trials`, and the output-construction loop in `build_output_trials`.

ii.
```python
for i in range(1, prob_left.shape[0]):
    if np.isclose(prob_left[i], prob_left[i - 1]):
        curr += 1.0
    else:
        curr = 1.0

for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...

for i, (ib, ie, beg, end) in enumerate(zip(idx_beg, idx_end, starts, ends, strict=True)):
    ...

for choice, prior, wheel, whisker in zip(
    record["choice"],
    record["prior"],
    record["wheel_cont"],
    record["whisker_cont"],
    strict=True,
):
    ...
```

iii. The agent did not discuss vectorization in its notes or trajectory. This is a direct reading of the implementation.

## 12-c. What processing does the code repeat multiple times?

i. The converter repeatedly glob-searches for “latest” files, repeats nearly the same interpolation logic for wheel and whisker, and performs two passes over continuous behavioral data: once to collect global pools for quantiles and again to digitize each trial during output construction.

ii.
```python
timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))

times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))

wheel_pool.append(np.concatenate(wheel_cont))
whisker_pool.append(np.concatenate(whisker_cont))

"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

iii. The notes do not mention this explicitly. It is inferred from the code structure.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins spikes and interpolates behavior for every trial before applying the final mask, so some work is thrown away. It also stores full continuous wheel and whisker traces in `records` only to use them for global quantiles and then discard them in the final dataset, which keeps only the discretized categories.

ii.
```python
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid

records.append(
    {
        ...
        "wheel_cont": wheel_cont,
        "whisker_cont": whisker_cont,
        ...
    }
)

"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

iii. The agent did not justify these extra passes explicitly. This is inferred from the converter’s control flow and data structures.
