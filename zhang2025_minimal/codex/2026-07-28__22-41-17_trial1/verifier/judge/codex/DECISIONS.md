# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the release index from `bwm_release.csv`, groups rows by `eid` to get one row per session, resolves each session to a cache path under `data/one_cache`, and then opens ALF files directly from disk with `np.load` and `pd.read_parquet`. It does not use the ONE API or `SessionLoader`/`SpikeSortingLoader` for session discovery.

ii. 
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

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
```

```python
def find_session_path(row: pd.Series) -> Path:
    return DATA_ROOT / row["lab"] / "Subjects" / row["subject"] / row["date"] / f"{int(row['session_number']):03d}"
```

iii. `CONVERSION_NOTES.md` says the AI "used the full provided `bwm_release.csv` release table" and "resolved directly from the local ONE cache using `lab / Subjects / subject / date / session_number`."

## 1-b. How are the data split into subjects (mice)?

i. Subjects come from the `subject` column in the release table. Sessions keep that subject label, and `subjects` / `subject_idx` are assembled after conversion.

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

```python
subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])
```

iii. The notes justify this as using the release table as the session source; there is no separate derivation step for subject identity.

## 1-c. How are the data split into sessions?

i. Sessions are identified by grouping the release table on `eid`. Each grouped row is processed as one session, with its probe names collected into a tuple.

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

```python
for idx, row in enumerate(sessions.to_dict("records"), start=1):
    session_path = find_session_path(row)
```

iii. `CONVERSION_NOTES.md` treats the release table as the authoritative session list: "459 sessions, 699 probe insertions, 139 subjects."

## 1-d. How are the data split into trials?

i. Trials are taken from rows of the session’s `_ibl_trials.table.pqt` table. After masking, one converted trial is produced for each surviving row.

ii. 
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    return pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))
```

```python
selected_idx = np.flatnonzero(final_mask)
neural_selected = [neural_trials[i] for i in selected_idx]
...
for choice, prior, wheel, whisker in zip(
    record["choice"],
    record["prior"],
    record["wheel_cont"],
    record["whisker_cont"],
    strict=True,
):
```

iii. No separate justification was given beyond following the trials table and then applying the trial mask.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials only if they have non-missing required events, reaction time between 0.08 s and 2.0 s, trial length `feedback_times - goCue_times <= 10.0`, nonzero choice, valid wheel coverage, valid whisker coverage, and at least one spike in the final QC-passed neural population during the 2 s window.

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
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    trial_len = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    mask &= rt >= 0.08
    mask &= rt <= 2.0
    mask &= trial_len <= 10.0
    mask &= trials["choice"].to_numpy() != 0
```

```python
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The notes say this "matches the reference `load_trials_and_mask(...)` logic" and add "trials with no spikes at all in the final QC-passed population within the 2 s window were dropped to avoid avoidable validator warnings."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is derived from `spikes.times.npy` and `spikes.clusters.npy` on each probe, with `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy` used for QC filtering and brain-region labels.

ii. 
```python
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
...
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

```python
cluster_channels = np.load(probe_base / "clusters.channels.npy")[good_mask].astype(np.int64)
channel_region_ids = np.load(probe_base / "channels.brainLocationIds_ccf_2017.npy")
```

iii. The notes explicitly list those files under "Neuron QC and probe merging."

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, sorts spikes by time, bins spikes into 100 bins of 20 ms over the 2 s aligned window, and stores the resulting spike counts as `float16`. It does not divide by bin width to convert counts to firing rates.

ii. 
```python
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

iii. The notes justify the 2 s / 20 ms setup, but do not justify leaving the neural signal as counts rather than converting to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps only clusters with `label >= 1.0` from `clusters.metrics.pqt`, then keeps only spikes assigned to those good clusters.

ii. 
```python
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]
```

```python
cluster_map[good_cluster_ids] = np.arange(offset, offset + good_cluster_ids.shape[0], dtype=np.int32)
...
mapped = cluster_map[spikes_clusters[valid]]
keep = mapped >= 0
all_times.append(spikes_times[valid][keep])
all_clusters.append(mapped[keep])
```

iii. The notes say this reproduces the paper’s `75,708` good units exactly and therefore matches the paper’s "well-isolated units" QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code builds one interval per trial from `stimOn_times + [-0.5, 1.5]` and bins spikes within that interval. Internally, spikes are expressed relative to the interval start (`stimOn_times - 0.5`), so the event is encoded through the fixed stimulus-centered window rather than by subtracting stimulus onset directly.

ii. 
```python
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
```

```python
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...
    times = spike_times[lo:hi] - start
    bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
```

iii. The notes justify alignment by saying "All trials are aligned to `trials.stimOn_times`" with the `[-0.5, 1.5]` window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 20 ms bins across a 2 s window, yielding 100 bins per trial. No additional temporal rebinning is applied.

ii. 
```python
WINDOW = (-0.5, 1.5)
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))
```

```python
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
```

iii. The notes explicitly state "Bin size: `20 ms`" and "Number of bins per trial: `100`."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the stimulus-alignment convention rather than measured from a separate raw signal. The code uses `trials.stimOn_times` to define trial windows and then constructs a fixed 100-point time vector for every kept trial.

ii. 
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))
```

```python
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The notes describe this as a stimulus-aligned decoder and say row 0 is "time since stimulus onset in seconds."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates a fixed vector `[-0.48, -0.46, ..., 1.50]` and copies it into every kept trial. This is a bin-end-time vector, not a bin-center vector.

ii. 
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

```python
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

iii. The notes explicitly justify this choice: "row 0: time since stimulus onset in seconds, bin end times: `[-0.48, -0.46, ..., 1.50]`."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The AI uses the same fixed 100-bin grid for the input time axis that it uses when binning spikes and interpolating behavior in each stimulus-aligned window.

ii. 
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

```python
times = spike_times[lo:hi] - start
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
```

iii. No separate justification was given beyond the general claim that all variables are stimulus aligned with 20 ms bins.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

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

iii. The notes say this variable is "trial number within the current `probabilityLeft` block."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans through the full `probabilityLeft` vector, increments a counter while the value stays the same, resets the counter when `probabilityLeft` changes, and assigns counts starting at `1`. Filtering happens afterward by indexing `selected_idx`, so removed trials still advance the counter.

ii. 
```python
trial_num = np.ones(prob_left.shape[0], dtype=np.float32)
curr = 1.0
for i in range(1, prob_left.shape[0]):
    if np.isclose(prob_left[i], prob_left[i - 1]):
        curr += 1.0
    else:
        curr = 1.0
    trial_num[i] = curr
```

```python
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
...
np.full(NBINS, trial_number[i], dtype=np.float32)
```

iii. The notes justify the reset rule and explicitly state the value "resets to `1` whenever `probabilityLeft` changes."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column of the trials table.

ii. 
```python
choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])
```

```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0
    out[np.isclose(values, 1.0)] = 1
```

iii. The notes say the choice output comes directly from IBL `choice` after recoding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code first removes no-choice trials via `choice != 0`, then maps `-1 -> 0` and `+1 -> 1`, and repeats that category across all 100 time bins of a kept trial.

ii. 
```python
mask &= trials["choice"].to_numpy() != 0
```

```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0
    out[np.isclose(values, 1.0)] = 1
```

```python
np.full((1, NBINS), choice, dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this mapping, stating "IBL `choice == -1` -> left -> `0`; `choice == +1` -> right -> `1`."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. 
```python
prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])
```

```python
def map_probability_left(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values.astype(np.float64), 1)
    out = np.full(values.shape[0], -1, dtype=np.int8)
```

iii. The notes say this output is the block prior from the trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code rounds `probabilityLeft` to one decimal place, maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, and repeats the resulting class across all 100 bins of the trial.

ii. 
```python
out[np.isclose(rounded, 0.2)] = 0
out[np.isclose(rounded, 0.5)] = 1
out[np.isclose(rounded, 0.8)] = 2
```

```python
np.full((1, NBINS), prior, dtype=np.int8)
```

iii. The notes justify this directly from the requested decoder mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. 
```python
timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
```

```python
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

iii. The notes justify this as reusing the provided IBL wheel preprocessing path.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code interpolates wheel position to 1000 Hz, computes Butterworth-filtered velocity, takes the absolute value to get speed, then linearly interpolates that continuous speed trace into each aligned 100-bin trial.

ii. 
```python
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

```python
rel_t = t - beg
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The notes say this exactly follows the IBL wheel-processing functions and uses `abs(velocity)` as the decoder target.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI pools all included wheel-speed samples across all sessions and all kept trials, computes global 1/3 and 2/3 quantiles, and digitizes each trial’s wheel-speed trace with those shared thresholds.

ii. 
```python
wheel_pool.append(np.concatenate(wheel_cont))
...
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)
```

```python
def digitize_tertiles(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.int8)
```

```python
wheel_disc = digitize_tertiles(wheel, wheel_thresholds)
```

iii. The notes justify this as a deliberate deviation: "I used a single global discretization over all included time bins" to "avoid session-specific label drift."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is sliced within each `stimOn_times + [-0.5, 1.5]` interval and interpolated onto the same 100-bin trial grid used for the neural data.

ii. 
```python
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
```

```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
...
rel_t = t - beg
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The notes justify this only at a high level by saying the decoder is stimulus aligned and uses 20 ms bins throughout.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `<view>Camera.ROIMotionEnergy.npy` and `_ibl_<view>Camera.times.npy`, preferring the left camera and falling back to the right camera if needed.

ii. 
```python
def _load_camera_stream(session_path: Path, view: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
    values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
```

```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        times, values = _load_camera_stream(session_path, "left")
        return times, values, "left"
    except Exception:
        times, values = _load_camera_stream(session_path, "right")
        return times, values, "right"
```

iii. The notes explicitly justify the left-preferred, right-fallback camera policy and say it preserves six additional sessions.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the raw released motion-energy trace, trims leading timestamps if the timestamp array is longer than the value array, rejects the stream if timestamps are shorter than values, and interpolates the per-frame signal into each aligned 100-bin trial.

ii. 
```python
if times.shape[0] < values.shape[0]:
    raise ValueError(f"{view} camera timestamps shorter than motion-energy array")
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
```

```python
rel_t = t - beg
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The notes justify this as matching the provided code behavior for camera selection and timestamp/value-length handling.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, it pools all included whisker-motion samples across the converted dataset, computes global 1/3 and 2/3 quantiles, and digitizes each trial’s whisker trace with those global thresholds.

ii. 
```python
whisker_pool.append(np.concatenate(whisker_cont))
...
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
```

```python
whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
```

iii. The notes give the same global-tertile justification as for wheel speed: preserve rank information and avoid session-specific label drift.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera trace is sliced in the same `stimOn_times + [-0.5, 1.5]` window and interpolated onto the same 100-bin grid used for the neural data and wheel signal.

ii. 
```python
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
...
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The notes justify this only through the common stimulus-aligned 20 ms binning scheme.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly drops invalid data. Missing required files skip a session. No-good-unit sessions and sessions with too few surviving trials are skipped. Trials with missing required events, failed stream coverage, NaNs in interpolated behavior, or all-zero neural bins are removed. For camera data, it prefers left but catches any exception and falls back to right.

ii. 
```python
if times.shape[0] < values.shape[0]:
    raise ValueError(f"{view} camera timestamps shorter than motion-energy array")
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
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

```python
except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1
except Exception as exc:
    skip_reasons[type(exc).__name__] += 1
```

iii. The notes justify the extra all-zero-spike drop as a validator-driven cleanup and describe the camera fallback and missing-file exclusions explicitly.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is the per-session loop over 459 sessions, especially loading and scanning large probe-level spike arrays, then binning spikes and interpolating behavior trial by trial.

ii. 
```python
for idx, row in enumerate(sessions.to_dict("records"), start=1):
    ...
    spike_times, spike_clusters, region_labels = load_good_units(session_path, row["probe_name"], br)
```

```python
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

```python
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. There is no explicit justification in the notes. This is inferred from the code structure and the progress logging during conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several explicit Python loops could have been vectorized: the block-counter loop in `compute_trial_number_in_block`, the per-trial spike-binning loop, the per-trial behavior interpolation loop, the list-comprehension build of `input_trials`, and the per-trial output discretization loop in `build_output_trials`.

ii. 
```python
for i in range(1, prob_left.shape[0]):
    if np.isclose(prob_left[i], prob_left[i - 1]):
        curr += 1.0
```

```python
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...
    counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
```

```python
for i, (ib, ie, beg, end) in enumerate(zip(idx_beg, idx_end, starts, ends, strict=True)):
    ...
    interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The AI did not explicitly justify keeping these loops; this is inferred from the implementation.

## 10-c. What processing does the code repeat multiple times?

i. It computes neural bins and behavioral interpolants for every trial before applying `final_mask`, so work is repeated on trials that will later be discarded. It also stores continuous wheel/whisker traces, pools them for global thresholds, and then loops over them again to discretize outputs.

ii. 
```python
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

```python
wheel_cont = [np.asarray(x, dtype=np.float32) for x in wheel_selected]
whisker_cont = [np.asarray(x, dtype=np.float32) for x in whisker_selected]
wheel_pool.append(np.concatenate(wheel_cont))
whisker_pool.append(np.concatenate(whisker_cont))
...
"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

iii. There is no explicit justification for these repeated passes in the notes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It keeps continuous `wheel_cont` and `whisker_cont` traces only to compute thresholds and build discrete outputs, but those continuous traces are not saved in the final dataset. It also bins spikes and interpolates behavior for trials that are later discarded by `final_mask`.

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
```

```python
"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

```python
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
selected_idx = np.flatnonzero(final_mask)
```

iii. The notes partially justify the all-zero-neural exclusion as validator cleanup, but they do not justify the extra continuous-trace bookkeeping or the work spent on discarded trials.
