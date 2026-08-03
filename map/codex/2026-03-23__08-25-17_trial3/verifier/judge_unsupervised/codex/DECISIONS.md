# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `data/sub-*/*.nwb`, sorts the file list, and opens each NWB with `h5py`. It first filters the session list to NWB files that contain at least one unit with `units/classification == "good"`, then re-opens each retained file and loads its trials, events, units, and tongue-tracking streams inside `process_session()`.

ii.
```python
def get_nwb_files(sample_only: bool) -> list[Path]:
    files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
    valid = []
    for path in files:
        with h5py.File(path, "r") as f:
            good = decode_str_array(f["units/classification"]) == "good"
            if np.any(good):
                valid.append(path)
    if sample_only:
        return valid[:SAMPLE_SESSION_COUNT]
    return valid

for i, path in enumerate(files, start=1):
    result = process_session(path)
    results.append(result)
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as matching the paper’s analyzed 173-session set by excluding the single raw NWB session with zero good units. It also explicitly chose `h5py` for lower overhead and direct control over ragged NWB arrays.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is taken from the NWB file’s parent directory name such as `sub-440956`. Later, `build_dataset()` deduplicates those subject strings and constructs `subject_idx` so each processed session points back to its mouse.

ii.
```python
session_id = path.stem
subject_id = path.parent.name

if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The notes say to use the exact NWB subject IDs (`sub-xxxxx`) and to keep session order aligned with the sorted NWB paths.

## 1-c. How are the data split into sessions?

i. Each retained NWB file is treated as one session. `process_session()` returns one `SessionResult`, and `build_dataset()` appends its `neural`, `input`, `output`, and `brain_region_idx` payloads as one session entry each.

ii.
```python
def process_session(path: Path) -> SessionResult:
    session_id = path.stem
    ...
    return SessionResult(
        session_id=session_id,
        subject_id=subject_id,
        neural_trials=neural_trials,
        input_trials=input_trials,
        output_trials=output_trials,
        ...
    )

for session_idx, result in enumerate(results):
    data["neural"].append(result.neural_trials)
    data["input"].append(result.input_trials)
    data["output"].append(result.output_trials)
```

iii. The notes state that session order follows sorted NWB file paths and that this preserves one session per NWB file, which is how the raw dataset is organized.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table and aligned to the one-per-trial `go_start_times` event stream. The code masks all per-trial variables with `valid_trial_mask`, then later turns the session arrays into Python lists where each list element is one trial.

ii.
```python
trials = f["intervals/trials"]
n_trials_raw = len(trials["id"])
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)

valid_trial_mask = compute_valid_trial_mask(obs_intervals=obs_intervals, go_times=go_times_all)

start_times = start_times_all[valid_trial_mask]
stop_times = stop_times_all[valid_trial_mask]
go_times = go_times_all[valid_trial_mask]

neural_trials = [firing_rates[:, i, :].copy() for i in range(n_trials)]
...
for trial_idx in range(n_trials):
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. The agent’s notes say `go_start_times` was chosen as the unique per-trial anchor because sample and delay event streams can contain replayed epochs after early licks, so they are not one-to-one with trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages. First, `compute_valid_trial_mask()` keeps only trials whose requested `[-2.5, 1.5)` go-aligned window fits inside `units/obs_intervals`. Second, after spike binning, the code drops any trial whose entire neural tensor is zero across all units and bins. It intentionally keeps stimulation, early-lick, ignore, and miss trials if they pass the neural-coverage filter.

ii.
```python
def compute_valid_trial_mask(
    obs_intervals: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    trial_start = go_times + REL_START_S
    trial_end = go_times + REL_END_S
    starts = obs_intervals[:, 0][None, :]
    ends = obs_intervals[:, 1][None, :]
    return np.any((trial_start[:, None] >= starts) & (trial_end[:, None] <= ends), axis=1)

valid_trial_mask = compute_valid_trial_mask(obs_intervals=obs_intervals, go_times=go_times_all)
...
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
if not np.all(nonzero_trial_mask):
    firing_rates = firing_rates[:, nonzero_trial_mask, :]
    ...
```

iii. In the notes, the agent says this differs from the paper’s stricter “regular trial” mask because the decoder task explicitly requires photostimulation, outcome, and early-lick labels. It also describes the all-zero removal as a fix for behavior trials that extended beyond valid neural recording coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural output is derived from raw `units/spike_times`, `units/spike_times_index`, `units/classification`, and go-cue timestamps from `acquisition/BehavioralEvents/go_start_times/timestamps`. `units/anno_name` is also loaded to label each retained unit’s brain region.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)

anno_name = decode_str_array(f["units/anno_name"])
region_labels = anno_name[good_unit_indices]

spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The notes map neural data to NWB spike times plus good-unit labels and go-cue timing, with `classification == "good"` used as the NWB-native proxy for the reference pipeline’s QC-selected units.

## 2-b. How is the `neural` data processed?

i. The code bins each retained unit’s absolute spike times into the go-aligned window and converts counts to firing rates in Hz by dividing by the 50 ms bin width. The stored trial matrices therefore have shape `(n_good_units, 80)` and dtype `float16`.

ii.
```python
BIN_SIZE_S = 0.05
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)

def bin_spikes_to_firing_rates(...):
    ...
    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
        counts = np.diff(edge_idx, axis=1)
        firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
    return firing_rates
```

iii. The agent justified this as preserving the reference pipeline’s go-centered spike binning logic while changing the actual bin width to 50 ms because that was an explicit decoder-task requirement.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is reduced to session exclusion if there are zero good units, plus per-unit retention only when `units/classification == "good"`. The script does not use `unit_quality`, `is_good_trials`, or the external QC/histology intersection described in the reference code notes.

ii.
```python
good = decode_str_array(f["units/classification"]) == "good"
if np.any(good):
    valid.append(path)

classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    raise ValueError(f"Session {session_id} has no good units")
```

iii. The notes explicitly say this was a deliberate approximation: the reference code used externally generated `goodunits` files and additional anatomical intersection, but the agent chose the NWB `classification` field as the closest directly available analogue.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset. Trial edges are built by adding the fixed relative edge vector `[-2.5, 1.5)` to each trial’s absolute go time.

ii.
```python
REL_START_S = -2.5
REL_END_S = 1.5
go_times = go_times_all[valid_trial_mask]
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The notes repeatedly justify go-cue alignment as both the instruction requirement and the common alignment convention in the reference code and papers.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms non-overlapping bins, giving 80 time bins per trial over the 4 s window. No secondary temporal rebinning is applied after spike counting.

ii.
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
```

iii. The notes explicitly call this a required deviation from the method paper’s 40 ms width and 3.4 ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final code, this input is not derived from a raw per-trial event variable. It is derived from the fixed constant `TONE_ONSET_REL_GO_S = -1.85` together with the common relative time-bin centers. Raw sample events are not used.

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The notes say the agent intentionally abandoned the raw `sample_start_times` stream because replayed sample epochs after early licks made that stream ambiguous, and instead used the canonical task timing of sample onset at `-1.85 s` relative to go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code computes a single session-wide vector `REL_CENTERS - (-1.85)` and tiles it identically across every trial. There is no per-trial correction for replayed sample epochs, missing sample events, or observed sample-event jitter.

ii.
```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. In the notes and trajectory, the agent justified this as “canonical sample onset at -1.85 s relative to go cue from task structure,” explicitly preferring fixed task structure over noisy raw sample-event streams.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by construction to the same go-centered bin centers used for neural activity. Every trial uses the same `REL_CENTERS` vector, so the time-from-tone input has the same 80 time points as the neural matrix.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The justification in the notes is that all decoded streams should share the go-cue-centered binning grid. The agent treated a common go-centered timeline as more reliable than trial-specific sample timestamps.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from trial-table columns `photostim_onset`, `photostim_duration`, and `start_time`, together with the go-cue timestamps used to convert the onset into go-centered coordinates.

ii.
```python
photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
start_times = start_times_all[valid_trial_mask]
go_times = go_times_all[valid_trial_mask]
```

iii. The notes say this mirrors the reference preprocessing, which subtracts go time from stimulation timing so stimulation can be represented in go-centered coordinates.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code parses trial-table strings, turns `"N/A"` into `NaN`, converts onset and offset from trial-start coordinates into go-centered coordinates, and fills a binary per-bin matrix where bin centers inside the stimulation interval are set to `1.0`.

ii.
```python
def build_photostim_matrix(...):
    onset_trial = parse_optional_float_array(photostim_onset_str)
    duration = parse_optional_float_array(photostim_duration_str)
    go_minus_start = go_times - start_times
    onset_rel_go = onset_trial - go_minus_start
    offset_rel_go = onset_rel_go + duration

    stim = np.zeros((len(go_times), N_BINS), dtype=np.float16)
    valid = np.isfinite(onset_rel_go) & np.isfinite(offset_rel_go)
    for trial_idx in np.where(valid)[0]:
        mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
        stim[trial_idx, mask] = 1.0
    return stim
```

iii. The agent justified keeping stimulation trials because photostimulation is an explicit decoder input, even though many reference analyses exclude stimulation trials from control-only analyses.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is represented on the same go-centered 50 ms time grid as the neural data. The binary mask is defined directly over the shared `REL_CENTERS`.

ii.
```python
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
stim[trial_idx, mask] = 1.0
```

iii. The notes describe this as converting raw stimulation timing into go-cue coordinates so it lands on the same per-trial time axis as spikes and tongue position.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `intervals/trials/trial_instruction`, `intervals/trials/outcome`, and absolute lick event times from `BehavioralEvents/left_lick_times` and `right_lick_times`. Trial `start_time`, `go_time`, and `stop_time` are used for the ignore-trial fallback logic.

ii.
```python
trial_instruction = decode_str_array(trials["trial_instruction"])[valid_trial_mask]
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]

left_lick_times = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()].astype(np.float64)
right_lick_times = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()].astype(np.float64)
choice_code = build_choice_array(
    trial_instruction=trial_instruction,
    outcome_code=outcome_code,
    start_times=start_times,
    go_times=go_times,
    stop_times=stop_times,
    left_lick_times=left_lick_times,
    right_lick_times=right_lick_times,
)
```

iii. The notes justify this as reconstructing a left/right choice label from trial type, outcome, and licks because the decoder task requires only two choice classes, even for ignore trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps instructed left/right to `0/1`. Hits inherit the instructed side, misses become the opposite side, and ignore trials use the first lick after go if present, otherwise the first lick anywhere in the trial, otherwise the instructed side. The chosen label is then repeated across all 80 bins.

ii.
```python
instructed = np.where(trial_instruction == "left", 0, 1).astype(np.int16)

hit_mask = outcome_code == 2
miss_mask = outcome_code == 1
ignore_mask = outcome_code == 0

choice[hit_mask] = instructed[hit_mask]
choice[miss_mask] = 1 - instructed[miss_mask]

for trial_idx in ignore_trials:
    choice[trial_idx] = lick_choice_with_fallback(...)

output_trial = np.vstack(
    [
        np.full(N_BINS, choice_code[trial_idx], dtype=np.int16),
        ...
    ]
)
```

iii. The notes say this fallback was necessary because most ignore trials have no post-go lick, and the trajectory explicitly records that the agent chose instructed side as the last fallback when no lick could be found.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table string column `intervals/trials/outcome`.

ii.
```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
```

iii. The notes treat this as a direct categorical variable already present in the NWB trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps strings to integer labels `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the integer across all time bins in each trial.

ii.
```python
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
...
np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16)
```

iii. This follows the exact output coding requested in the instructions, which the agent copied into the mapping notes.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table string column `intervals/trials/early_lick`.

ii.
```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
```

iii. The notes describe this as a direct trial-table label needed both for the decoder output and for understanding the paper’s regular-trial exclusions.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `no early -> 0` and `early -> 1`, then repeats that per-trial label across all 80 bins in the output tensor.

ii.
```python
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
...
np.full(N_BINS, early_code[trial_idx], dtype=np.int16)
```

iii. The mapping is exactly what the decoder task requested, and the agent’s notes explicitly highlight early lick as a category preserved rather than filtered out.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the side-view tongue tracking time series `BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the `x`, `y`, and `likelihood` columns, plus that time series’ timestamps and the trial go-cue times.

ii.
```python
tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. The notes state that all sessions had this side-view tongue stream, and that the agent intentionally used the side-view tongue `y` coordinate because it matches the requested output and the reference paper’s side-view video processing.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The script performs two cleanup steps before alignment. First, it computes frame-to-frame speed from tongue `x` and `y`, flags non-finite points or points above a `mean + 5*std` speed threshold, and linearly interpolates through those outliers. Second, it treats low-likelihood frames as occlusions and replaces their `y` values with the mean of high-likelihood `y` frames. After cleaning, it aligns the cleaned `y` values to trial bins.

ii.
```python
speed = np.zeros_like(x)
if len(x) > 1:
    speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
...
x[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], x[keep_mask])
y[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], y[keep_mask])
...
visible_mask = np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
mean_y = float(np.nanmean(y[visible_mask])) if np.any(visible_mask) else float(np.nanmean(y))
y[occluded_mask] = mean_y
```

iii. The notes and trajectory say this was chosen to mirror the reference paper’s five-sigma velocity-outlier cleanup and mean-imputation of occluded tongue positions, while specializing the logic to side-view tongue `y`.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After alignment, the code computes the 40th and 60th percentiles over all aligned tongue-`y` samples that were retained for that session, not over the full raw video stream. It initializes all bins to category `1`, sets bins below `q40` to `0`, and bins above `q60` to `2`.

ii.
```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

iii. The notes explicitly justify this as computing percentiles from “the aligned tongue-y samples that actually enter the converted dataset, not from unrelated off-trial video periods.”

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Alignment uses last-frame-carried-forward sampling onto the same go-centered bin centers used for neural activity. For each trial and bin center, the code finds the most recent tongue-tracking timestamp at or before that absolute time and copies the cleaned `y` value from that frame.

ii.
```python
def align_tongue_y(
    timestamps: np.ndarray,
    cleaned_y: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

iii. The trajectory and notes say the agent explicitly checked the reference alignment logic and chose last-frame-carried-forward because it believed that best matched the reference marker-alignment scripts.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses pragmatic fallbacks throughout. `"N/A"` photostimulation strings become `NaN` and then simply produce all-zero photostim rows. Non-finite tongue points and high-velocity outliers are interpolated; if too few good frames remain, the code falls back to the mean. Low-likelihood tongue frames are replaced by the mean visible `y`. Ignore trials without post-go licks fall back to within-trial licks and then to the instructed side. Sessions with no good units raise an error and are excluded upstream. Trials with no valid neural coverage are removed before conversion, and trials that still bin to all-zero neural matrices are dropped afterward.

ii.
```python
def parse_optional_float_array(strings: np.ndarray) -> np.ndarray:
    out = np.full(strings.shape, np.nan, dtype=np.float64)
    for i, value in enumerate(strings):
        if value == "N/A":
            continue
        out[i] = float(value)
    return out

if np.sum(keep_mask) >= 2:
    x[outlier_mask] = np.interp(...)
    y[outlier_mask] = np.interp(...)
else:
    x[outlier_mask] = np.nanmean(x)
    y[outlier_mask] = np.nanmean(y)

y[occluded_mask] = mean_y
...
return instructed_choice
```

iii. The notes justify these choices as conservative fallbacks needed to preserve required decoder inputs and outputs while still keeping the converted dataset fully populated and verifier-clean.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is spike binning in `bin_spikes_to_firing_rates()`, which loops over every retained unit and performs `searchsorted` over all trial edges. Reading large NWB arrays, especially spike arrays and tongue-tracking arrays, is the next major cost. The conversion log shows per-session binning often taking a large fraction of total session time.

ii.
```python
for i, unit_idx in enumerate(good_unit_indices):
    spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
    edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
    counts = np.diff(edge_idx, axis=1)
    firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)

print(
    f"[session] {session_id} - done "
    f"({n_trials} trials, {len(good_unit_indices)} good units, "
    f"{diagnostics['session_time_s']:.1f}s total, {neural_time_s:.1f}s binning)"
)
```

iii. The notes and run logs both point to spike binning as the heavy step; the notes describe the script as keeping “all heavy operations vectorized in NumPy” except for unavoidable per-unit loops during binning.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `bin_spikes_to_firing_rates()` is the main vectorization target. Smaller candidates are the per-trial loop in `build_photostim_matrix()`, the per-ignore-trial loop in `build_choice_array()`, the Python loops in `decode_str_array()` and `parse_optional_float_array()`, the per-trial construction of `input_trials` and `output_trials`, and the per-unit region-index mapping loop in `build_dataset()`.

ii.
```python
for i, unit_idx in enumerate(good_unit_indices):
    ...

for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
    stim[trial_idx, mask] = 1.0

for trial_idx in ignore_trials:
    choice[trial_idx] = lick_choice_with_fallback(...)

for trial_idx in range(n_trials):
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. The notes emphasize that the agent used NumPy where practical, but they also record several explicit Python loops left in place for simplicity.

## 10-c. What processing does the code repeat multiple times?

i. The script repeats several computations. It opens every NWB once during session discovery and again during real processing. Photostimulation onset relative to go is computed once for the actual input matrix and again for diagnostics. The identical `time_from_tone_onset` vector is tiled for every trial in every session. Constant trial labels for choice, outcome, and early lick are expanded into 80-bin vectors separately for every trial. It also builds both `region_names` and `region_labels_per_unit`, but only the latter is ultimately used to populate the dataset.

ii.
```python
for path in files:
    with h5py.File(path, "r") as f:
        good = decode_str_array(f["units/classification"]) == "good"
        ...

onset_rel_go = onset_trial - go_minus_start
...
photostim_onsets_rel_go = onset_rel_go[np.isfinite(onset_rel_go)]

input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
...
np.full(N_BINS, choice_code[trial_idx], dtype=np.int16)
```

iii. The notes mention the two-pass session handling and call out some duplicated diagnostic computations as acceptable overhead for a one-shot conversion script.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is the diagnostic path: preview traces, histograms, percentile values, timing fields, and plot support arrays are computed even though they are not saved into `converted_data.pkl`. `SessionResult.region_names` is also built but never used when assembling the final dataset. In addition, trials that end up all-zero are fully binned before being thrown away.

ii.
```python
diagnostics = {
    "tracking_preview_time": tongue_timestamps[:preview_n] - go_times[0],
    "tracking_preview_raw": tongue_y[:preview_n],
    "tracking_preview_clean": cleaned_tongue_y[:preview_n],
    "aligned_tongue_y_trial": aligned_tongue_y[0],
    "photostim_onsets_rel_go": photostim_onsets_rel_go,
    ...
}

return SessionResult(
    ...
    region_names=sorted(set(region_labels.tolist())),
    region_labels_per_unit=region_labels.copy(),
    ...
)
```

iii. The notes frame this extra work as sanity-check support rather than output-critical processing, especially because the instructions required conversion notes, validation, and optional processing plots.
