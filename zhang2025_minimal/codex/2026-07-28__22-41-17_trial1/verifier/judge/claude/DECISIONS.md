# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads data directly from cached ALF files on disk rather than using the ONE API. It loads a release CSV (`bwm_release.csv`) from the reference code directory to enumerate sessions and their probe insertions. For each session, it constructs a file path from the lab, subject, date, and session number, then reads trial tables (parquet), spike sorting files (npy), wheel data (npy), and camera motion energy (npy) directly from those paths.

ii.
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
release = pd.read_csv(RELEASE_CSV, index_col=0)
sessions = (
    release.groupby("eid", sort=False)
    .agg({
        "subject": "first", "date": "first",
        "session_number": "first", "lab": "first",
        "probe_name": lambda x: tuple(sorted(x)),
    })
    .reset_index()
)
```

```python
def find_session_path(row: pd.Series) -> Path:
    return DATA_ROOT / row["lab"] / "Subjects" / row["subject"] / row["date"] / f"{int(row['session_number']):03d}"
```

iii. The agent noted (step 36-40) that the local ALF cache had enough raw data to bypass the online APIs, and decided to read cached ALF files directly rather than using the ONE client, since the `ibllib` packages were not fully wired into `PYTHONPATH`.

## 1-b. How are the data split into subjects?

i. The subject name comes from the `subject` column of the release CSV. Subjects are collected in encounter order as sessions are processed, and a `subject_to_idx` mapping is built at the end.

ii.
```python
subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])
```

iii. The release CSV already contains a subject column per session row, so no additional parsing is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV represents one session. The code groups the release rows by `eid` to get one row per session with aggregated probe names.

ii.
```python
sessions = (
    release.groupby("eid", sort=False)
    .agg({...})
    .reset_index()
)
```

iii. Sessions are already the unit of organization in the release CSV.

## 1-d. How are the data split into trials?

i. The trials table (parquet) has one row per trial. Each row contains timing events, choice, and block probability. Trials are indexed and filtered by a mask.

ii.
```python
trials = load_trials_table(session_path)
# returns pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))
```

iii. The trials table is already one row per trial, so no splitting logic is needed.

## 1-e. How are trials filtered based on quality controls?

i. Several filters are applied: (1) reaction time between 0.08s and 2.0s, (2) trial length (feedback_times - goCue_times) <= 10s, (3) choice != 0 (no-response excluded), (4) required columns must be non-NaN (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (5) wheel and whisker behavioral streams must cover the trial window, (6) neural trials with all-zero spike counts after binning are excluded.

ii.
```python
def make_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    trial_len = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    mask &= rt >= 0.08
    mask &= rt <= 2.0
    mask &= trial_len <= 10.0
    mask &= trials["choice"].to_numpy() != 0
    for col in required:
        mask &= ~trials[col].isna().to_numpy()
    return mask
```

```python
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The agent noted (step 16) it was following the paper's trial exclusions: "80 ms to 2 s reaction times" and excluding no-choice trials. The trial length filter and neural validity check are additional filters not from the reference paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike times) and `spikes.clusters.npy` (cluster assignments). Cluster quality labels come from `clusters.metrics.pqt`, and brain regions are derived from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
```

iii. The agent identified spikes.times and spikes.clusters as the core neural data arrays, consistent with the reference approach.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20ms bins over a 2s trial window (-0.5 to 1.5s around stimulus onset). The counts are stored as float16 without conversion to firing rates. When a session has multiple probes, units are pooled with continuous numbering.

ii.
```python
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
valid = (bins >= 0) & (bins < NBINS)
flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
trials.append(counts.astype(np.float16))
```

iii. The agent followed the reference's 20ms bin size and 2s window. However, it stored raw spike counts rather than converting to firing rates (Hz).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1.0` in the metrics table are kept. Unlike the reference, the AI does NOT filter out `void` brain regions (channels placed outside the brain by histology). All QC-passed clusters are included regardless of atlas label.

ii.
```python
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]
# No void filtering
```

iii. The agent noted (step 68) verifying neuron counts against the paper's 75,708 well-isolated units and used the same QC threshold, but did not implement void region filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are aligned to stimulus onset. The interval is [stimOn_times + WINDOW[0], stimOn_times + WINDOW[1]] = [stimOn - 0.5, stimOn + 1.5]. Spike times within this interval are offset by subtracting the interval start (stimOn + WINDOW[0]).

ii.
```python
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
# In bin_spikes:
times = spike_times[lo:hi] - start  # start = stim_on + WINDOW[0]
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
```

iii. The agent aligned to stimulus onset as specified in the instructions and reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms, giving 100 bins over the 2s window. No rebinning or interpolation is applied to neural data.

ii.
```python
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))  # 100
```

iii. This matches the reference code's `binsize: 0.02` and the method paper's description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The input is a time axis computed from the window parameters and bin size.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The agent used stimulus onset as the alignment event, consistent with the instructions.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time input is computed as bin END times relative to stimulus onset, starting at -0.48s and ending at 1.5s, with 20ms spacing. This differs from the reference which uses bin CENTERS.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
# This gives [-0.48, -0.46, ..., 1.48, 1.50]
# Reference gives [-0.49, -0.47, ..., 1.47, 1.49] (bin centers)
```

iii. The agent documented this as "bin end times relative to stimulus onset" in the metadata.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses bin end times while the neural data is binned from t=0 to t=BIN for bin 0, so the time values represent the right edge of each bin. This is internally consistent within the AI's code but differs from the reference's bin-center approach.

ii.
```python
# Neural bins: floor(times / BIN_SIZE_S), so bin 0 covers [0, 0.02)
# Time input: WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS+1), so first value is -0.48
```

iii. The agent explicitly noted "bin_time_reference: bin end times relative to stimulus onset" in metadata.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Changes in this value indicate block boundaries.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    for i in range(1, prob_left.shape[0]):
        if np.isclose(prob_left[i], prob_left[i - 1]):
            curr += 1.0
        else:
            curr = 1.0
        trial_num[i] = curr
    return trial_num
```

iii. The trials table has no explicit block identifier, so blocks are recovered from changes in probabilityLeft, matching the reference approach.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number starts at 1 for the first trial in each block and increments by 1. This is 1-indexed, unlike the reference which uses 0-indexed counting (cumcount starts at 0). The count is computed before trial filtering, so excluded trials still advance the counter.

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

iii. The agent computed trial numbers before applying the trial mask, consistent with the reference's approach of counting against the original block structure.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which is +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The agent identified choice as a direct column in the trials table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps -1 -> 0 and +1 -> 1. This is REVERSED from the instructions which specify left=0, right=1. In the IBL convention, +1 is left and -1 is right, so the AI's mapping gives right=0 and left=1.

ii.
```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0  # right -> 0 (should be left -> 0)
    out[np.isclose(values, 1.0)] = 1   # left -> 1 (should be right -> 1)
```

iii. The agent did not provide explicit reasoning about the direction of the choice mapping in the trajectory.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The agent correctly identified probabilityLeft as the source variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The values are mapped to integers: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, matching the instructions.

ii.
```python
def map_probability_left(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values.astype(np.float64), 1)
    out[np.isclose(rounded, 0.2)] = 0
    out[np.isclose(rounded, 0.5)] = 1
    out[np.isclose(rounded, 0.8)] = 2
```

iii. The mapping matches the instructions exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The position is interpolated to 1000 Hz and differentiated into velocity using a Butterworth filter, then absolute value gives speed.

ii.
```python
timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

iii. The agent imported the IBL wheel processing functions directly from `brainbox.behavior.wheel`, matching the reference's approach.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000 Hz, differentiated with a 20 Hz Butterworth low-pass filter (order 8), and absolute value taken. The continuous speed trace is then interpolated onto trial time bins (bin end times) and discretized into 3 categories using GLOBAL tertile thresholds computed across all sessions.

ii.
```python
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

```python
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1/3, 2/3]).astype(np.float32)
wheel_disc = digitize_tertiles(wheel, wheel_thresholds)
```

iii. The agent chose global tertile thresholds across the entire dataset rather than per-session percentiles as the reference does.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The continuous wheel speed is discretized into 3 bins (low=0, medium=1, high=2) using GLOBAL 33rd and 67th percentile thresholds computed across all sessions' concatenated wheel speed values.

ii.
```python
wheel_pool.append(np.concatenate(wheel_cont))
# After all sessions:
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1/3, 2/3])
wheel_disc = digitize_tertiles(wheel, wheel_thresholds)
```

iii. The agent documented this as "global tertiles over all included time bins in the converted dataset" in the metadata.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same time grid as the behavioral data, using bin end times relative to the trial window start. This differs slightly from the reference which uses bin centers.

ii.
```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
rel_t = t - beg  # beg = stim_on + WINDOW[0]
interp = np.interp(x_interp, rel_t, y)
```

iii. The wheel is on the same session clock as the spikes, so interpolation onto the time grid provides alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`. The left camera is preferred; the right camera is used as fallback.

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple:
    try:
        times, values = _load_camera_stream(session_path, "left")
        return times, values, "left"
    except Exception:
        times, values = _load_camera_stream(session_path, "right")
        return times, values, "right"
```

iii. The agent followed the reference's preference for left camera with right-camera fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is with no additional filtering. It is interpolated onto bin end times and discretized using GLOBAL tertile thresholds across all sessions.

ii.
```python
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1/3, 2/3])
whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
```

iii. Same global discretization approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: GLOBAL 33rd and 67th percentile thresholds across all sessions, producing 3 categories (low=0, medium=1, high=2).

ii.
```python
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1/3, 2/3])
```

iii. The agent chose global thresholds for consistency across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: interpolated onto bin end times within each trial window.

ii.
```python
interp = np.interp(x_interp, rel_t, y)
```

iii. Camera frame times are on the same session clock as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data handlers: (1) sessions with missing required files are skipped via FileNotFoundError, (2) sessions with no QC-passed units are skipped, (3) sessions with fewer than 2 valid trials are skipped, (4) camera timestamp/value length mismatches are handled by trimming timestamps from the front, (5) NaN values in behavioral streams cause trial exclusion, (6) generic exceptions during session processing cause the session to be skipped with a logged reason.

ii.
```python
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
```

```python
except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1
except Exception as exc:
    skip_reasons[type(exc).__name__] += 1
```

iii. The agent tracked skip reasons in a Counter for diagnostic purposes and documented 20 sessions missing required files and 1 failing stream alignment.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (reading large npy files for spikes.times and spikes.clusters per probe) is the most expensive step. The full conversion took about 14 minutes for 438 sessions.

ii.
```python
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

iii. The agent noted (step 95) the conversion was multi-minute and the dominant cost was I/O.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops could be vectorized: (1) the spike binning loop in `bin_spikes` iterates over each trial, and (2) the behavioral interpolation loop in `interpolate_behavior_into_trials` iterates per trial. Both could be vectorized with offset indexing.

ii.
```python
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    # per-trial spike binning
```

```python
for i, (ib, ie, beg, end) in enumerate(zip(idx_beg, idx_end, starts, ends, strict=True)):
    # per-trial behavioral interpolation
```

iii. The agent did not discuss vectorization opportunities in the trajectory.

## 10-c. What processing does the code repeat multiple times?

i. The code processes each session sequentially in a single loop, computing wheel and whisker data once per session. However, the discretization thresholds require a second pass: behavioral data is first stored as continuous values, then after all sessions are processed, global thresholds are computed and discretization is applied when building the output.

ii.
```python
# First pass: collect continuous values
wheel_pool.append(np.concatenate(wheel_cont))
whisker_pool.append(np.concatenate(whisker_cont))
# After loop: compute thresholds and build outputs
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1/3, 2/3])
"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

iii. This two-pass approach is necessitated by the global thresholding design choice.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores continuous wheel and whisker values in each record (`wheel_cont`, `whisker_cont`) which are only used to compute global thresholds and then build discretized outputs. These continuous arrays are not included in the final pickle but consume memory during processing. The code also computes and stores extensive metadata and session info that is not used by the decoder.

ii.
```python
records.append({
    ...
    "wheel_cont": wheel_cont,
    "whisker_cont": whisker_cont,
    ...
})
```

iii. The two-pass design means all continuous behavioral data must be held in memory until thresholds are computed.
