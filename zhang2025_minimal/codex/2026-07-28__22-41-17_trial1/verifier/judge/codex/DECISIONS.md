# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the ONE API or `SessionLoader` / `SpikeSortingLoader`. It loaded the release index from `bwm_release.csv`, grouped rows by `eid` to define sessions, reconstructed each session path inside the local `one_cache`, then opened ALF files directly with `pandas.read_parquet` and `numpy.load`. When multiple matching files existed, it chose the lexicographically latest path with `pick_latest(...)`.

ii.
```python
DATA_ROOT = ROOT / "data" / "one_cache"
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def pick_latest(session_path: Path, pattern: str) -> Path:
    matches = sorted(session_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {session_path}")
    return matches[-1]

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

iii. In the trajectory, the AI explicitly justified bypassing the API because the local cache already contained the needed files: "The local cache has the trial tables and motion-energy arrays, so I can bypass the online APIs" (step 36). It also said the cache had all 459 release sessions and treated direct ALF access as sufficient (steps 65 and 68).

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. After session conversion, `subjects` is built in first-seen order, and `subject_idx` maps each included session to that order.

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

subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])
```

iii. The trajectory does not show an explicit debate here. The AI treated the release table metadata as authoritative session metadata and carried subject IDs through from there.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from `bwm_release.csv`. Probe rows are merged into one session record by grouping on `eid` and collecting probe names.

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

iii. In the trajectory, the AI described the release table as the session source and repeatedly reasoned in terms of all 459 release sessions, then keeping a usable subset after filtering (steps 63, 65, 68, 107).

## 1-d. How are the data split into trials?

i. Trials are taken directly from the rows of `_ibl_trials.table.pqt` for each session. The code does not reconstruct trial boundaries from another stream.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    return pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))

trials = load_trials_table(session_path)
```

iii. The trajectory treats the trial table as present in the cache and as the canonical source of trial information (step 36). No alternative split was discussed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a two-stage trial filter. First, `make_trial_mask(...)` requires non-NaN `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`; enforces `0.08 <= firstMovement_times - stimOn_times <= 2.0`; enforces `feedback_times - goCue_times <= 10.0`; and removes `choice == 0`. Second, it keeps only trials with wheel coverage, whisker coverage, and at least one spike in the final QC-passed neural population.

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

iii. The trajectory shows explicit justification for the extra silent-trial filter: after a subset validation run, the AI said it saw "one fully silent trial" and decided to drop "any completely spike-silent trial after QC/alignment so the final dataset validates without avoidable warnings" (steps 83 and 86). The trial-mask shape otherwise follows its reading of the reference code, as reflected later in its conversion notes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` arrays come from `spikes.times.npy` and `spikes.clusters.npy` on each probe. The code also loads `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy` to decide which clusters survive QC and to assign region labels.

ii.
```python
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
cluster_channels = np.load(probe_base / "clusters.channels.npy")[good_mask].astype(np.int64)
channel_region_ids = np.load(probe_base / "channels.brainLocationIds_ccf_2017.npy")

spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

iii. In the trajectory, the AI said it was verifying "how cluster QC labels, channel-to-region mapping, and wheel/video timestamps are stored" before writing the converter (step 36), then later said it wanted to match the paper's well-isolated-neuron count instead of guessing (step 68).

## 2-b. How is the `neural` data processed?

i. The AI merges all probes in a session into one pooled population, renumbers clusters across probes, sorts all spikes by time, then bins spikes into 100 bins of 20 ms over each 2 s trial window. The stored `neural` signal is spike count per bin cast to `float16`; it is not divided by bin width into firing rate.

ii.
```python
cluster_map[good_cluster_ids] = np.arange(offset, offset + good_cluster_ids.shape[0], dtype=np.int32)
all_times.append(spikes_times[valid][keep])
all_clusters.append(mapped[keep])
offset += good_cluster_ids.shape[0]

spike_times = np.concatenate(all_times)
spike_clusters = np.concatenate(all_clusters)
order = np.argsort(spike_times, kind="stable")

for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...
    times = spike_times[lo:hi] - start
    bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
    flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
    counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
    trials.append(counts.astype(np.float16))
```

iii. The trajectory explicitly said the converter would use "stimulus-aligned 20 ms spike bins" and a "single pooled population" per session (steps 72 and 74). It did not justify storing counts instead of rates; that appears to have been an unstated implementation choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps clusters with `clusters.metrics.label >= 1.0`, merges them across probes, and retains their Beryl-mapped region labels. It does not filter out Beryl `void` labels; its notes explicitly say labels such as `root`, `void`, `x`, and `y` are preserved when present.

ii.
```python
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]

cluster_region_ids = channel_region_ids[cluster_channels]
allen_regions = br.id2acronym(cluster_region_ids)
beryl_regions = br.acronym2acronym(allen_regions, mapping="Beryl")
all_regions.extend(str(x) for x in beryl_regions.tolist())
```

iii. The trajectory shows that the AI noticed "`root`/`void` atlas labels" in the subset output, then checked decoder behaviour before deciding whether to filter them (step 83). Its written notes justify keeping them to preserve release-level good-unit accounting, but the trajectory itself only shows that it saw the issue and chose not to remove those labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window is anchored on `stimOn_times`, with `[stimOn - 0.5, stimOn + 1.5]`. Within `bin_spikes(...)`, spike times are expressed relative to the start of that window, so the first bin corresponds to `[-0.5, -0.48)` relative to stimulus onset.

ii.
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)

stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)

times = spike_times[lo:hi] - start
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
```

iii. The trajectory consistently states that the conversion is "stimulus-aligned" and uses "2 s trials" around `stimOn_times` (steps 16, 72, 74).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms (`BIN_SIZE_S = 0.02`) with 100 bins per trial. No later temporal rebinning is applied.

ii.
```python
WINDOW = (-0.5, 1.5)
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))
```

iii. The trajectory explicitly cites the "2 s / 20 ms setup described in the methods text" as a design anchor (step 72).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read as a raw stream. The AI synthesizes it from the configured alignment event `stimOn_times`, the fixed window `[-0.5, 1.5]`, and the 20 ms bin grid.

ii.
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)
BIN_SIZE_S = 0.02
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The trajectory repeatedly frames the dataset as stimulus-aligned and describes the first decoder input as part of that fixed aligned grid rather than something loaded from disk (steps 16, 72, 74).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates a fixed 100-element vector of bin end times, `[-0.48, -0.46, ..., 1.50]`, and reuses that row for every trial. It does not use bin centres.

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

iii. The trajectory does not explicitly justify choosing bin ends. The later notes state this directly as "bin end times," so the choice appears to have been intentional but not argued in the live reasoning trace.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The same 100-bin trial structure is used for `input` and `neural`. The first input row is broadcast into each trial as the per-bin time coordinate for the spike-count matrix.

ii.
```python
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
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

iii. The trajectory consistently describes the whole dataset as a unified stimulus-aligned 100-bin representation (steps 72 and 74).

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` trial column. A new block starts whenever `probabilityLeft` changes.

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

iii. The trajectory does not show separate debate on this point. The AI’s later notes explain that block membership is recovered from `probabilityLeft` because that value is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts trials within each `probabilityLeft` block with a Python loop. The count resets to `1` at each block change, so this is a one-based counter. It is computed on the full trial table before `final_mask` is applied, so dropped trials still advance the count.

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

trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
selected_idx = np.flatnonzero(final_mask)
```

iii. The trajectory does not explicitly justify the one-based indexing. Its notes do say the counter "resets to 1 whenever `probabilityLeft` changes."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `Choice` is derived from the trial-table `choice` column and then remapped into a binary label.

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

iii. The trajectory does not contain an explicit justification for this sign convention. The AI’s written notes later assert that `-1` means left and `+1` means right, so this appears to have been an assumption rather than a trajectory-justified decision.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI converts the trial-table `choice` values into binary labels with `-1.0 -> 0` and `1.0 -> 1`, then repeats the resulting per-trial value across all 100 time bins in the output array.

ii.
```python
choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])

out = np.vstack(
    [
        np.full((1, NBINS), choice, dtype=np.int8),
        np.full((1, NBINS), prior, dtype=np.int8),
        wheel_disc[None, :],
        whisker_disc[None, :],
    ]
)
```

iii. No explicit justification for the remapping appears in the trajectory.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` trial-table column.

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

iii. The trajectory does not show a separate debate; this follows the task instruction directly.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI rounds the raw float values to one decimal place, maps `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, and repeats the per-trial category across all 100 bins.

ii.
```python
rounded = np.round(values.astype(np.float64), 1)
out[np.isclose(rounded, 0.2)] = 0
out[np.isclose(rounded, 0.5)] = 1
out[np.isclose(rounded, 0.8)] = 2

prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. The trajectory contains no special justification beyond following the requested output coding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
```

iii. The trajectory explicitly says the AI inspected the `ibllib` wheel implementation so it could match the reference processing rather than approximate it (step 40), then summarized the result as 1 kHz interpolation plus filtered differentiation (step 51).

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to 1000 Hz using `interpolate_position`, computes low-pass filtered velocity with `velocity_filtered(..., corner_frequency=20, order=8)`, takes the absolute value, and then interpolates the speed trace into each trial window with `np.interp`.

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

iii. The trajectory directly justifies the preprocessing choice: "wheel is linearly interpolated to 1 kHz then low-pass differentiated" (step 51). It also said it wanted to reuse the IBL implementation rather than invent one (step 40).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI pools all kept wheel samples from all included sessions, computes one pair of global tertile thresholds over the whole dataset, and digitizes each trial’s wheel-speed trace against those two thresholds.

ii.
```python
wheel_pool.append(np.concatenate(wheel_cont))
...
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)

def digitize_tertiles(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.int8)

wheel_disc = digitize_tertiles(wheel, wheel_thresholds)
```

iii. The trajectory does not argue for global tertiles in the live messages. The later conversion notes justify them as a way to avoid session-specific label drift and keep class balance.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI intended the wheel trace to be sampled on the same 100-bin, stimulus-aligned grid as the neural data. In the code, however, the interpolation query points are `[-0.48, ..., 1.50]` while `rel_t` is measured from the start of the trial window, so the wheel trace is sampled on a shifted coordinate system.

ii.
```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
...
rel_t = t - beg
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The trajectory repeatedly says the outputs are stimulus-aligned and share the same aligned bins as the neural data (steps 72 and 74). It does not acknowledge the coordinate-frame mistake visible in the implementation.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `<view>Camera.ROIMotionEnergy.npy` and the corresponding `_ibl_<view>Camera.times.npy` stream, preferring the left camera and falling back to the right camera.

ii.
```python
def _load_camera_stream(session_path: Path, view: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
    values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
    ...

def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        times, values = _load_camera_stream(session_path, "left")
        return times, values, "left"
    except Exception:
        times, values = _load_camera_stream(session_path, "right")
        return times, values, "right"
```

iii. The trajectory explicitly says the AI was checking the "right-camera whisker fallback" to explain the session count gap and that six included sessions relied on that fallback (steps 107 and 110).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy values directly, trims leading timestamps if the timestamp array is longer than the motion-energy array, rejects the stream if timestamps are shorter than values, and interpolates the trace into each trial window with `np.interp`.

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

iii. The trajectory says video timestamps may need trimming and that the AI wanted to replicate the reference camera handling instead of approximating it (steps 40 and 51).

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the AI uses one set of global tertile thresholds pooled across all included sessions and trials, then digitizes each whisker trace against those thresholds.

ii.
```python
whisker_pool.append(np.concatenate(whisker_cont))
...
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
...
whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
```

iii. The live trajectory does not justify global thresholds. The later notes justify them as preserving rank information without session-specific label drift.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI intended whisker motion energy to share the same 100-bin, stimulus-aligned grid as the neural data. In the implementation, it uses the same shifted interpolation pattern as wheel speed: `rel_t` is measured from window start while `x_interp` is expressed as `[-0.48, ..., 1.50]`.

ii.
```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
...
rel_t = t - beg
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The trajectory describes whisker motion energy as stimulus-aligned in the same unified format as the other streams (steps 72, 74, 107, 110), but it does not notice the coordinate mismatch in the code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data by dropping trials or skipping sessions. Missing wheel or whisker files raise `FileNotFoundError` and skip the session. Trials with missing required events, missing wheel/whisker coverage, NaNs in interpolated behavior, or zero spikes in the QC-passed neural population are removed. Sessions with no good units or fewer than two final trials are skipped.

ii.
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
if trial_mask.sum() < 2:
    skip_reasons["too_few_trials_after_reference_mask"] += 1
    continue
...
if len(region_labels) == 0:
    skip_reasons["no_good_units"] += 1
    continue
...
if final_mask.sum() < 2:
    skip_reasons["too_few_trials_after_stream_alignment"] += 1
    continue
...
except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1
```

iii. The trajectory explicitly says the converter needs to "tolerate or skip" incomplete sessions (step 63), identifies missing behavior streams as a major exclusion source (steps 65, 130, 174), and justifies dropping silent trials to avoid validator warnings (step 86).

## 10-a. What are the most time-consuming steps of the code?

i. The code is likely dominated by per-session raw-data loading and per-trial spike / behavior processing: opening spike arrays and metrics for every probe, binning spikes for every trial, and interpolating wheel and whisker traces for every trial. The full conversion loop is sequential over sessions.

ii.
```python
for idx, row in enumerate(sessions.to_dict("records"), start=1):
    ...
    wheel_times, wheel_speed = load_wheel_speed(session_path)
    whisker_times, whisker_me, whisker_view = load_whisker_motion_energy(session_path)
    spike_times, spike_clusters, region_labels = load_good_units(session_path, row["probe_name"], br)
    ...
    neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
    wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
    whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. The trajectory gives operational evidence rather than profiling: it reports the full conversion taking about 14 minutes (step 107) and later describes validation as spending time loading and summarizing a 6.2 GB dataset (steps 122 and 134).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious Python loops are the block-counter loop, the per-trial spike-binning loop, the per-trial interpolation loop, the per-trial output-assembly loop, and the outer per-session loop. The inner three are the main candidates for vectorization or parallelization.

ii.
```python
for i in range(1, prob_left.shape[0]):
    ...
```

```python
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...
```

```python
for i, (ib, ie, beg, end) in enumerate(zip(idx_beg, idx_end, starts, ends, strict=True)):
    ...
```

```python
for choice, prior, wheel, whisker in zip(
    record["choice"],
    record["prior"],
    record["wheel_cont"],
    record["whisker_cont"],
    strict=True,
):
    ...
```

iii. The trajectory never discusses vectorization explicitly. Its plan focused on getting a correct standalone converter running first, then validating on subset and full data (step 72).

## 10-c. What processing does the code repeat multiple times?

i. The same interpolation routine is run separately for wheel and whisker on every session, and the code processes all trials through spike binning and behavior interpolation before masking out many of them with `trial_mask` and `final_mask`. It also assembles continuous wheel/whisker traces into `records`, then revisits them later to discretize outputs.

ii.
```python
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

```python
wheel_pool.append(np.concatenate(wheel_cont))
whisker_pool.append(np.concatenate(whisker_cont))
...
"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

iii. The trajectory does not call this out explicitly. It emphasizes correctness and validation rather than optimizing repeated work (step 72).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins spikes and interpolates wheel / whisker for every trial in a session before filtering with `final_mask`, so work on excluded trials is discarded. It also retains continuous `wheel_cont` and `whisker_cont` traces in `records` only so it can compute global thresholds and then convert them to discrete outputs; those continuous traces are not saved in the final dataset.

ii.
```python
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

```python
"wheel_cont": wheel_cont,
"whisker_cont": whisker_cont,
...
"output": [build_output_trials(rec, wheel_thresholds, whisker_thresholds) for rec in records],
```

iii. The trajectory does not justify this extra work directly. The only explicit rationale near it is the decision to keep continuous traces long enough to compute pooled tertiles, which the AI later defended in its written notes rather than in the live trajectory.
