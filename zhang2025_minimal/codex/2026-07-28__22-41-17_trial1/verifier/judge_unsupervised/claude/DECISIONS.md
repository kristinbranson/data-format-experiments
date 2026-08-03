# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the local ONE cache filesystem. It reads `bwm_release.csv` (the BWM release table with 699 probe insertions across 459 sessions) to enumerate all sessions. For each session, it constructs a filesystem path from lab/subject/date/session_number and loads trial tables, spike data, wheel data, and whisker motion energy from ALF-format files (parquet, npy).

ii.
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

release = pd.read_csv(RELEASE_CSV, index_col=0)
sessions = (
    release.groupby("eid", sort=False)
    .agg({
        "subject": "first",
        "date": "first",
        "session_number": "first",
        "lab": "first",
        "probe_name": lambda x: tuple(sorted(x)),
    })
    .reset_index()
)
```

iii. The AI used the same release CSV used by the reference code (`bwm_release.csv`) and read data from the local ONE cache, bypassing the ONE API for efficiency while preserving the same data source.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. After all sessions are processed, unique subjects are collected in order of first appearance across successfully processed sessions.

ii.
```python
subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])
```

iii. The AI builds the subjects list from the successfully converted sessions, assigning each subject a unique index. This matches the target data format requirement for `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`. The release table has 699 rows (one per probe insertion), which are grouped by `eid` to yield 459 unique sessions. Multiple probes from the same session are aggregated into a tuple.

ii.
```python
sessions = (
    release.groupby("eid", sort=False)
    .agg({
        "subject": "first",
        "date": "first",
        "session_number": "first",
        "lab": "first",
        "probe_name": lambda x: tuple(sorted(x)),
    })
    .reset_index()
)
```

iii. The AI followed the same approach as the reference code, which groups probe insertions by session `eid` and merges multi-probe data within each session.

## 1-d. How are the data split into trials?

i. For each session, trials are loaded from the `_ibl_trials.table.pqt` parquet file. Each trial corresponds to a row in this table. Trials are then aligned to `stimOn_times` and the 2-second window is divided into 100 time bins of 20ms each.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    return pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))

stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
```

iii. Trial splitting follows the standard IBL approach: each row of the trials table is one trial. The AI then constructs per-trial time intervals for binning neural and behavioral data.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using a multi-criterion mask matching the reference `load_trials_and_mask()` function: required non-NaN events, reaction time bounds (0.08-2.0s), max trial length (10s), no-choice exclusion. Additionally, trials must have valid wheel and whisker coverage for the full aligned window, and non-zero spike counts.

ii.
```python
def make_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times", "choice", "feedback_times",
        "probabilityLeft", "firstMovement_times", "feedbackType",
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

# Additional filtering:
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The AI documented that it matched the reference code's `load_trials_and_mask()` logic with default parameters (min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True, exclude_unbiased=False). The additional neural_valid mask drops trials with zero spikes after binning.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times and cluster assignments: `spikes.times.npy` and `spikes.clusters.npy` for each probe, filtered by cluster quality labels from `clusters.metrics.pqt`.

ii.
```python
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
```

iii. The AI loads raw spike trains (times and cluster IDs) from the ALF cache, matching the reference code's use of `SpikeSortingLoader` but reading files directly for efficiency.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 20ms bins within each trial's 2-second window ([-0.5, 1.5] relative to stimulus onset), producing spike count matrices of shape (n_neurons, 100) per trial. Multiple probes from the same session are merged by concatenating and re-indexing cluster IDs.

ii.
```python
def bin_spikes(spike_times, spike_clusters, intervals, n_neurons):
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    start_idx = np.searchsorted(spike_times, starts, side="left")
    end_idx = np.searchsorted(spike_times, ends, side="left")
    trials = []
    for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
        times = spike_times[lo:hi] - start
        bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
        valid = (bins >= 0) & (bins < NBINS)
        flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
        counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
        trials.append(counts.astype(np.float16))
    return trials
```

iii. The AI implemented a vectorized binning approach using `np.floor` and `np.bincount` rather than the reference code's `bincount2D` multiprocessing approach, but achieves the same result: spike count matrices per trial.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1.0` in `clusters.metrics.pqt` are retained (well-isolated units). The AI verified this reproduces 75,708 good units across all 699 probes. Sessions with zero good units are skipped. Additionally, trials with all-zero spike counts after binning are excluded.

ii.
```python
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]
if good_cluster_ids.size == 0:
    continue

# Later:
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
```

iii. The AI matched the reference code's QC threshold (`label >= 1.0`, equivalent to `qc=1` in the reference `load_spiking_data`). The all-zero trial exclusion is an extra safeguard not in the reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). Each trial window spans from 0.5s before to 1.5s after stimulus onset.

ii.
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)

stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
```

iii. The instructions specify "Temporally align based on stimulus onset." The AI used stimulus onset alignment for all variables, consistent with the instructions but differing from the reference paper which uses first-movement alignment for wheel and whisker.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms, producing 100 bins per trial over the 2-second window. No temporal rebinning is applied.

ii.
```python
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))  # = 100
```

iii. The 20ms bin size matches the reference paper's description for dynamic behaviors (wheel speed and whisker motion energy). The reference paper uses 50ms bins for choice and prior decoding, but since the instructions require a unified dataset, 20ms was chosen consistently. This matches the reference code's bin size for dynamic behavioral targets.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` in the trials table, combined with the window parameters and bin size.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The time input is computed as bin-end times relative to stimulus onset, ranging from -0.48s to 1.50s in steps of 0.02s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linearly spaced array of bin-end times is computed. The values represent the right edge of each 20ms bin relative to stimulus onset. This is constant across all trials.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
# Results in [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. The AI chose bin-end times rather than bin-start or bin-center times, consistent with the reference code's behavior interpolation: `x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input array is the same for every trial since all trials have the same window and bin structure. It is directly aligned with the neural spike count bins by construction.

ii.
```python
input_trials = [
    np.vstack([
        time_input,
        np.full(NBINS, trial_number[i], dtype=np.float32),
    ]).astype(np.float32)
    for i in selected_idx
]
```

iii. Since both the neural bins and time input use the same window and bin size, alignment is implicit.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table. Block boundaries are detected by changes in `probabilityLeft`.

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

iii. The AI uses `probabilityLeft` as the block identifier, incrementing a counter when consecutive trials share the same value and resetting on changes. This correctly captures block structure.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Starting from 1, the trial number increments by 1 for each consecutive trial with the same `probabilityLeft` value. It resets to 1 when `probabilityLeft` changes. The computation is done over ALL trials (before filtering), and then the values for selected trials are picked.

ii.
```python
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
# ...
np.full(NBINS, trial_number[i], dtype=np.float32)  # broadcast per-trial value across time bins
```

iii. The trial number is computed before trial masking, which is correct since it represents position within the experimental block sequence. The value is constant across all time bins within a trial.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii.
```python
choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The IBL `choice` column contains -1 (left) and +1 (right), with 0 indicating no choice.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL choice values are remapped: -1 (left) becomes 0, +1 (right) becomes 1. No-choice trials (choice=0) are already excluded by the trial mask. The per-trial value is broadcast across all 100 time bins.

ii.
```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0
    out[np.isclose(values, 1.0)] = 1
    if np.any(out < 0):
        raise ValueError("Unexpected choice values encountered")
    return out

# In build_output_trials:
np.full((1, NBINS), choice, dtype=np.int8),
```

iii. This matches the instructions: "left = 0, right = 1". The AI correctly validates that no unexpected values remain.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The `probabilityLeft` column contains the block-level probability (0.2, 0.5, or 0.8).

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to 1 decimal place and mapped to categorical indices: 0.2->0, 0.5->1, 0.8->2. The per-trial value is broadcast across all 100 time bins.

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

# In build_output_trials:
np.full((1, NBINS), prior, dtype=np.int8),
```

iii. This matches the instructions: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
    position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
```

iii. The raw wheel data consists of position measurements at irregular timestamps from a rotary encoder.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000 Hz, then velocity is computed using a Butterworth low-pass filter (20 Hz corner frequency, order 8). The absolute value of velocity gives wheel speed. This is then linearly interpolated into trial-aligned 20ms bins.

ii.
```python
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

iii. The AI imported and used the exact same IBL library functions (`interpolate_position`, `velocity_filtered`) from `brainbox.behavior.wheel`, matching the reference code's wheel processing. The reference `load_target_behavior` for 'wheel-speed' also uses `np.abs(velocity)`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global tertile thresholds are computed across all included time bins from all sessions, then applied using `np.digitize` to produce 3 categories (0=low, 1=medium, 2=high).

ii.
```python
wheel_pool.append(np.concatenate(wheel_cont))
# After all sessions:
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)

def digitize_tertiles(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.int8)
```

iii. The instructions say "Wheel speed discretized into 3 bins". The AI chose global tertiles to ensure approximately equal class frequencies, which is a reasonable discretization strategy.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Continuous wheel speed is linearly interpolated into the same bin-end time grid as the neural data, using `np.interp` within each trial's stimulus-aligned window.

ii.
```python
def interpolate_behavior_into_trials(times, values, intervals):
    x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
    # ...
    rel_t = t - beg
    interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The AI aligns wheel speed to stimulus onset, matching the neural data alignment. The reference paper uses first-movement alignment for wheel, but the instructions specify stimulus onset alignment for the unified dataset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from camera timestamps (`_ibl_{view}Camera.times.npy`) and motion energy values (`{view}Camera.ROIMotionEnergy.npy`), preferring the left camera with right camera fallback.

ii.
```python
def _load_camera_stream(session_path, view):
    times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
    values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
```

iii. The whisker motion energy is a pre-computed feature from video frames, measuring whisker pad movement.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly (no additional filtering). If camera timestamps are longer than the motion energy array, timestamps are trimmed from the front. The values are then linearly interpolated into trial-aligned 20ms bins.

ii.
```python
def _load_camera_stream(session_path, view):
    times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
    values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
    if times.shape[0] > values.shape[0]:
        times = times[-values.shape[0]:]
    return times, values
```

iii. The timestamp trimming from the front matches the IBL `_check_video_timestamps()` behavior for pre-GPIO sessions. The left-first/right-fallback policy matches the reference code.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile thresholds computed across all included time bins, then applied with `np.digitize` to produce 3 categories.

ii.
```python
whisker_pool.append(np.concatenate(whisker_cont))
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
```

iii. Same justification as wheel speed discretization: global tertiles for uniform class balance.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Continuous whisker motion energy is linearly interpolated into the same bin-end time grid as the neural data, within each trial's stimulus-aligned window. Same interpolation function as wheel speed.

ii.
```python
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. Aligned to stimulus onset, same as neural data and wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple mechanisms handle missing data:
- Sessions with missing files (wheel, whisker, spike data) are skipped with a `FileNotFoundError` catch
- Trials with NaN in required columns are excluded by the trial mask
- Trials where behavioral data doesn't cover the full window are marked invalid by the interpolation function
- Trials with all-zero spike counts are excluded
- Camera timestamp length mismatches are handled by trimming
- Any unexpected exceptions during session processing result in the session being skipped and logged

ii.
```python
except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1
except Exception as exc:
    skip_reasons[type(exc).__name__] += 1
    print(f"Skipped session {row['eid']} due to {type(exc).__name__}: {exc}")

# In interpolate_behavior_into_trials:
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

iii. The AI took a conservative approach, excluding any trial or session with data quality issues rather than imputing. This is documented in CONVERSION_NOTES.md with a full list of excluded sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading and filtering spike data per session (reading large npy files for spikes.times and spikes.clusters)
2. Spike binning across all trials per session (the `bin_spikes` function)
3. Wheel processing (interpolating to 1000 Hz and filtering)
4. Behavior interpolation into trial bins

ii.
```python
# Spike loading - reads large arrays
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)

# Wheel processing - interpolation to 1kHz + filtering
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
```

iii. The AI's conversion took significant wall time processing 438 sessions with large spike arrays.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could benefit from vectorization is the trial-level spike binning loop in `bin_spikes`, which iterates over each trial sequentially. The `compute_trial_number_in_block` function also uses a sequential loop. The `interpolate_behavior_into_trials` function loops over trials one at a time.

ii.
```python
# bin_spikes trial loop
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    # ...per-trial binning...

# compute_trial_number_in_block
for i in range(1, prob_left.shape[0]):
    if np.isclose(prob_left[i], prob_left[i - 1]):
        curr += 1.0
    else:
        curr = 1.0

# interpolate_behavior_into_trials
for i, (ib, ie, beg, end) in enumerate(...):
    # ...per-trial interpolation...
```

iii. The reference code uses multiprocessing for these loops. The AI's sequential approach is simpler but slower.

## 10-c. What processing does the code repeat multiple times?

i. The behavior interpolation function `interpolate_behavior_into_trials` is called twice per session (once for wheel, once for whisker) with the same trial intervals. The `x_interp` array is recomputed inside the function each time it's called despite being constant.

ii.
```python
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)

# Inside the function:
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The repeated `x_interp` computation is negligible in cost but could be precomputed once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes neural data as spike counts (integers stored as float16) rather than firing rates. The reference code's `bincount2D` also produces raw counts that are later used. The code also stores comprehensive metadata and session info that goes beyond what the decoder needs. The `time_input` is a deterministic function of the window and bin size, so storing it per trial is redundant (though required by the format). The code computes `trial_number_in_block` for ALL trials including filtered-out ones.

ii.
```python
# trial_number computed for all trials, only used for selected ones
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
# ...
np.full(NBINS, trial_number[i], dtype=np.float32)  # only selected_idx used
```

iii. The extra computation on filtered-out trials is minor. The extensive metadata is useful for reproducibility even if not consumed by the decoder.
