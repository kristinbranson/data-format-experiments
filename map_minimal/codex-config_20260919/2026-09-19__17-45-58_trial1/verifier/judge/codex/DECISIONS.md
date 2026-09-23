# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the dataset by globbing every NWB file under `sub-*/*.nwb`, sorting that list, and iterating through it one file at a time. Each file is opened directly with `h5py`, then the converter reads datasets from `units`, `intervals/trials`, `acquisition/BehavioralEvents`, and `acquisition/BehavioralTimeSeries`.

ii.
```python
def convert(data_dir: Path, output_path: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")

    converted_sessions = []
    skipped_sessions = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:3d}/{len(paths)}] {path.name}", flush=True)
        session = _convert_session(path)
```

```python
def _convert_session(path: Path) -> dict | None:
    with h5py.File(path, "r") as nwb:
        classification = _decode_array(nwb["units/classification"])
        all_go_times = nwb[
            "acquisition/BehavioralEvents/go_start_times/timestamps"
        ][:].astype(np.float64)
        n_source_trials = len(nwb["intervals/trials/id"])
```

iii. In the trajectory, the agent said it would first inspect the preprocessing choices and raw NWB schema, then preserve the source pipeline where possible. It later reported that the dataset boundary was 174 NWB files and that the conversion should iterate over those files once, dropping only sessions that failed curation.

## 1-b. How are the data split into subjects?

i. The agent treats the parent folder name as the subject id. For each session it stores `path.parent.name.removeprefix("sub-")`, then builds `subjects` as the sorted unique set of those ids and `subject_idx` as the per-session lookup into that set.

ii.
```python
subject_id = path.parent.name.removeprefix("sub-")
return {
    "subject": subject_id,
    ...
}
```

```python
subjects = sorted({session["subject"] for session in converted_sessions})
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_lookup[session["subject"]] for session in converted_sessions],
    dtype=np.int32,
),
```

iii. The trajectory does not record a separate argument for using the folder name instead of the NWB subject field. The closest justification is the agent’s schema audit message that the dataset consisted of 174 files across 28 mice and that the file hierarchy itself cleanly exposed those session and mouse boundaries.

## 1-c. How are the data split into sessions?

i. The agent uses one NWB file as one session. Session order is the sorted file order; each surviving `_convert_session(path)` result becomes one output session, and sessions with no good units are skipped entirely.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
for index, path in enumerate(paths, start=1):
    session = _convert_session(path)
    if session is None:
        skipped_sessions.append(path.name)
    else:
        converted_sessions.append(session)
```

iii. In the trajectory, the agent explicitly said the dataset boundary was 174 files and later summarized the final artifact in numbers of sessions kept versus skipped. That indicates it interpreted file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. The agent starts from the NWB trial table length (`intervals/trials/id`) and the go-cue timestamp array, requiring them to match. When the per-unit `is_good_trials` matrix has fewer columns than the full trial table, it maps those compact ephys trials back onto source-trial indices using the common observation interval across retained good units. The surviving indices are then used to index all per-trial variables.

ii.
```python
all_go_times = nwb[
    "acquisition/BehavioralEvents/go_start_times/timestamps"
][:].astype(np.float64)
n_source_trials = len(nwb["intervals/trials/id"])
if len(all_go_times) != n_source_trials:
    raise ValueError(
        f"{path.name}: {len(all_go_times)} go cues for "
        f"{n_source_trials} trials"
    )
```

```python
good_trial_matrix = nwb["units/is_good_trials"][good_rows, :]
n_ephys_trials = good_trial_matrix.shape[1]
if n_ephys_trials == n_source_trials:
    ephys_source_idx = np.arange(n_source_trials)
else:
    ...
    ephys_source_idx = np.flatnonzero(
        (all_go_times + OFF_START >= common_start - 1e-6)
        & (all_go_times + OFF_END <= common_stop + 1e-6)
    )
```

iii. The trajectory says the agent found sessions where the behavioral table extended beyond ephys acquisition and tightened curation by mapping compact ephys trial masks back to the behavioral rows. That was its explicit rationale for not using only the raw trial table rows as-is.

## 1-e. How are trials filtered based on quality controls?

i. The agent filters trials by taking the intersection of `units/is_good_trials` across all retained good units, mapping those columns back to source trials, and then removing `auto_water` and `free_water` trials. After neural binning it also drops any remaining trial whose entire population firing matrix is all zeros. Sessions with fewer than two valid trials raise an error and are skipped.

ii.
```python
common_good_mask = np.all(good_trial_matrix, axis=0)
common_good_idx = ephys_source_idx[common_good_mask]
auto_water_all = nwb["intervals/trials/auto_water"][:].astype(bool)
free_water_all = nwb["intervals/trials/free_water"][:].astype(bool)
water_mask = auto_water_all[common_good_idx] | free_water_all[common_good_idx]
valid_trial_idx = common_good_idx[~water_mask]
if valid_trial_idx.size < 2:
    raise ValueError(f"{path.name}: fewer than two common valid trials")
```

```python
population_recorded = np.any(rates != 0, axis=(1, 2))
n_population_dropouts = int(np.count_nonzero(~population_recorded))
if n_population_dropouts:
    rates = rates[population_recorded]
    valid_trial_idx = valid_trial_idx[population_recorded]
    go_times = go_times[population_recorded]
```

iii. This is the best-justified part of the trajectory. The agent wrote that all retained trials had to be inside every retained unit’s `is_good_trials` coverage so a recording gap would not be encoded as biological silence. It later said eight NWBs extended past ephys acquisition, that it therefore tightened curation to the intersection of `is_good_trials`, and that free-water trials had no spikes in this NWB release. It then added `auto_water` removal because the repository’s `get_regular_trial_mask` excludes both auto-water and free-water trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `units/spike_times_index`, restricted to rows where `units/classification == "good"`. Trial go-cue times supply the alignment windows.

ii.
```python
classification = _decode_array(nwb["units/classification"])
good_rows = np.flatnonzero(classification == "good")
...
all_spikes = nwb["units/spike_times"][:]
spike_ends = nwb["units/spike_times_index"][:]
rates = _bin_good_units(all_spikes, spike_ends, good_rows, go_times)
```

iii. In the trajectory, the agent explicitly said the source pipeline confirmed that spikes should come from the NWB spike timestamps and that only classifier-labeled good units should be retained. It also said go-cue timestamps were the common clock for alignment.

## 2-b. How is the `neural` data processed?

i. The agent converts spikes into non-overlapping 50 ms firing-rate bins in Hz. For each good unit it assigns each spike to the unique trial window that contains it, computes a within-window bin index by flooring the spike’s offset from the window start, counts spikes with `np.bincount`, reshapes the result into `(trial, time)`, and divides by `0.05`.

ii.
```python
def _bin_good_units(
    all_spikes: np.ndarray,
    spike_ends: np.ndarray,
    good_rows: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    n_trials = len(go_times)
    n_units = len(good_rows)
    rates = np.zeros((n_trials, n_units, N_BINS), dtype=np.float32)
    window_starts = go_times + OFF_START
    window_stops = go_times + OFF_END
```

```python
for out_unit, unit_row in enumerate(good_rows):
    ...
    trial = np.searchsorted(window_starts, spikes, side="right") - 1
    ...
    time_bin = np.floor(
        (spikes - window_starts[trial]) / BIN_SIZE_S + 1e-10
    ).astype(np.int64)
    valid = (time_bin >= 0) & (time_bin < N_BINS)
    flat_bin = trial[valid] * N_BINS + time_bin[valid]
    counts = np.bincount(flat_bin, minlength=n_trials * N_BINS)
    rates[:, out_unit, :] = counts.reshape(n_trials, N_BINS) / BIN_SIZE_S
```

iii. The trajectory says the agent confirmed from the source code that firing rate should be spike count divided by bin width, and later noted that go-cue separations exceeded the four-second analysis window, which justified assigning each spike to a unique trial by window start.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The main neural QC filter is `units/classification == "good"`. Sessions with zero such units are skipped. The agent also raises an error if a retained good unit lacks an anatomical `anno_name`, and it only keeps trials that all retained good units mark as valid in `units/is_good_trials`.

ii.
```python
classification = _decode_array(nwb["units/classification"])
good_rows = np.flatnonzero(classification == "good")
if good_rows.size == 0:
    return None
```

```python
annotations = _decode_array(nwb["units/anno_name"])[good_rows]
if np.any(annotations == ""):
    raise ValueError(f"{path.name}: a good unit lacks an Allen annotation")
```

```python
good_trial_matrix = nwb["units/is_good_trials"][good_rows, :]
common_good_mask = np.all(good_trial_matrix, axis=0)
```

iii. The trajectory repeatedly states that classifier-labeled good units were the published curation target and that one file with no such units should be skipped. It separately argues that `is_good_trials` should be used so missing acquisition is not treated as real zero firing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset. Each trial’s window is `[go_time - 2.5 s, go_time + 1.5 s)`, and spike offsets inside that window determine the 50 ms bin index.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE_S))
BIN_EDGES = OFF_START + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
window_starts = go_times + OFF_START
window_stops = go_times + OFF_END
...
time_bin = np.floor(
    (spikes - window_starts[trial]) / BIN_SIZE_S + 1e-10
).astype(np.int64)
```

iii. The trajectory explicitly says the go cue is the common alignment event and that all NWB clocks are already on the same absolute time base, so no separate cross-stream synchronization step is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 80 non-overlapping 50 ms bins from `-2.5` to `+1.5` seconds relative to the go cue. There is no additional temporal rebinning or smoothing.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE_S))
BIN_EDGES = OFF_START + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The trajectory says the converter produced exactly 80 left-closed bins centered from `-2.475` to `+1.475` seconds and was intentionally matched to the task’s required 50 ms bin width.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from the go-cue timestamps and the `sample_start_times` timestamps. The agent chooses the final sample/tone onset before each go cue.

ii.
```python
def _tone_onsets(go_times: np.ndarray, sample_starts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sample_starts, go_times, side="right") - 1
    if np.any(idx < 0):
        raise ValueError("A go cue has no preceding sample/tone onset")
    return sample_starts[idx]
```

```python
sample_starts = nwb[
    "acquisition/BehavioralEvents/sample_start_times/timestamps"
][:]
tone_onset = _tone_onsets(go_times, sample_starts)
```

iii. The trajectory states that early licks can replay the sample epoch, so the agent intentionally used the last tone before the go cue rather than the first one in the trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the agent computes `go_time + BIN_CENTERS - tone_onset`, so the result is a continuous time-varying signal expressed at every neural-bin center. The array is cast to `float32`.

ii.
```python
for trial in range(n_trials):
    time_from_tone = (
        go_times[trial] + BIN_CENTERS - tone_onset[trial]
    ).astype(np.float32)
    ...
    inputs.append(np.stack((time_from_tone, photo_on), axis=0))
```

iii. In the trajectory, the agent summarized this as using the last replayed tone before the go cue and evaluating time on the same 80 bin centers used for neural data.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated exactly on the neural-bin centers relative to each trial’s go cue, so it shares the neural time axis directly.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = (
    go_times[trial] + BIN_CENTERS - tone_onset[trial]
).astype(np.float32)
```

iii. The trajectory says the same 80-bin go-cue-centered grid was used across inputs, outputs, and neural activity to avoid any extra alignment transform.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table columns `start_time`, `photostim_onset`, and `photostim_duration`, together with the trial go-cue times used to place those intervals on the aligned time axis.

ii.
```python
trial_start = nwb["intervals/trials/start_time"][:][valid_trial_idx]
stim_onset_raw = nwb["intervals/trials/photostim_onset"][:][valid_trial_idx]
stim_duration_raw = nwb["intervals/trials/photostim_duration"][:][valid_trial_idx]
stim_onset = np.asarray([_optional_float(x) for x in stim_onset_raw])
stim_duration = np.asarray([_optional_float(x) for x in stim_duration_raw])
```

iii. The trajectory says the agent preserved photostimulation trials because they are required decoder inputs, but needed to convert the stored per-trial stimulation timing onto the go-cue-centered grid.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The agent converts onset and duration values from strings to floats, leaves missing values as `NaN`, then creates a binary time series per trial. A bin center is assigned `1.0` when its absolute time falls within `[photo_start, photo_stop)`, otherwise `0.0`.

ii.
```python
def _optional_float(value) -> float:
    text = _decode(value)
    return np.nan if text in {"", "N/A", "nan"} else float(text)
```

```python
photo_on = np.zeros(N_BINS, dtype=np.float32)
if np.isfinite(stim_onset[trial]) and np.isfinite(stim_duration[trial]):
    photo_start = trial_start[trial] + stim_onset[trial]
    photo_stop = photo_start + stim_duration[trial]
    absolute_centers = go_times[trial] + BIN_CENTERS
    photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
```

iii. The trajectory justifies keeping photostimulation as a time-varying decoder input even though some source analyses excluded those trials.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The code compares photostimulation start and stop times to `go_times + BIN_CENTERS`, so photostimulation is aligned to the same per-trial neural-bin centers.

ii.
```python
photo_start = trial_start[trial] + stim_onset[trial]
photo_stop = photo_start + stim_duration[trial]
absolute_centers = go_times[trial] + BIN_CENTERS
photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
```

iii. The trajectory repeatedly states that the go cue is the common clock reference across all data streams, and this code directly applies that decision.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a dedicated choice column. The agent derives it from the trial-table fields `trial_instruction` and `outcome`.

ii.
```python
instruction = _decode_array(
    nwb["intervals/trials/trial_instruction"]
)[valid_trial_idx]
outcome_text = _decode_array(
    nwb["intervals/trials/outcome"]
)[valid_trial_idx]
```

```python
def _trial_choice(instruction: str, outcome: str) -> int:
    if outcome == "ignore":
        return 2
    if outcome == "hit":
        return 0 if instruction == "left" else 1
    if outcome == "miss":
        return 1 if instruction == "left" else 0
```

iii. The agent’s top-level docstring says this was deliberate: it used the instruction and outcome labels as the authoritative source because direct lick-event timing in NWB could be missing or mistimestamped on rare trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps `hit` to the instructed side, `miss` to the opposite side, and `ignore` to a third `no lick` class. It stores those values as `0 = left`, `1 = right`, `2 = no lick`, then repeats the per-trial label across all 80 time bins.

ii.
```python
trial_output = np.empty((4, N_BINS), dtype=np.uint8)
trial_output[0, :] = _trial_choice(
    instruction[trial], outcome_text[trial]
)
```

```python
"output_values": [
    ["left", "right", "no lick"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ...
],
```

iii. The trajectory says the agent kept ignored/no-response trials specifically because they are required decoder classes, which is why `ignore` becomes its own `no lick` choice label instead of being dropped.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the NWB trial-table field `intervals/trials/outcome`.

ii.
```python
outcome_text = _decode_array(
    nwb["intervals/trials/outcome"]
)[valid_trial_idx]
```

iii. The trajectory does not add a separate justification here beyond the general decision to reuse authoritative trial labels from the NWB tables whenever available.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps outcome strings with `{"ignore": 0, "miss": 1, "hit": 2}` and repeats the resulting categorical value across all 80 bins in output row 1.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
...
trial_output[1, :] = outcome_map[outcome_text[trial]]
```

iii. The trajectory’s general justification is that ignored/no-response trials must remain in the dataset because `outcome` is one of the requested decoder outputs.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the NWB trial-table field `intervals/trials/early_lick`.

ii.
```python
early_text = _decode_array(
    nwb["intervals/trials/early_lick"]
)[valid_trial_idx]
```

iii. The trajectory states that early-lick trials were intentionally preserved, even though many source analyses excluded them, because early lick is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent maps the string label to a binary category with `1 if early_text[trial] == "early" else 0`, then repeats that label across all 80 bins in output row 2.

ii.
```python
trial_output[2, :] = 1 if early_text[trial] == "early" else 0
```

iii. The trajectory’s explicit reasoning is the same as for 7-a: the code retains early-lick trials so that the requested target variable can be learned rather than filtered away.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and the matching `timestamps`. The agent uses column 1 as `y` and column 2 as the DeepLabCut likelihood.

ii.
```python
tongue_path = (
    "acquisition/BehavioralTimeSeries/"
    "Camera0_side_TongueTracking"
)
tongue_data = nwb[f"{tongue_path}/data"][:]
camera_times = nwb[f"{tongue_path}/timestamps"][:]
...
y = tongue_data[:, 1]
likelihood = tongue_data[:, 2]
```

iii. The agent’s docstring and trajectory both say it intentionally used the side-camera DLC tongue trace and treated it as sharing the same NWB session clock as neural data.

## 8-b. How is the `output` *Tongue y-position* processed?

i. The agent first defines session-visible frames as those with finite `y`, finite likelihood, and likelihood `>= 0.9`. It computes the 40th and 60th percentiles from all such visible raw frames over the session. It then samples one camera frame per neural-bin center using the immediately preceding frame, discards samples where the preceding frame is more than 20 ms old, and classifies visible sampled values against the two session cut points. No 50 ms averaging within bins is performed.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
visible_session = (
    np.isfinite(y)
    & np.isfinite(likelihood)
    & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
)
if not np.any(visible_session):
    q40 = q60 = np.nan
else:
    q40, q60 = np.percentile(y[visible_session], [40, 60])
```

```python
target_times = go_times[:, None] + BIN_CENTERS[None, :]
frame = np.searchsorted(camera_times, target_times, side="right") - 1
valid_frame = frame >= 0
frame = np.clip(frame, 0, len(camera_times) - 1)
valid_frame &= (target_times - camera_times[frame]) <= 0.020
sampled_y = y[frame]
sampled_likelihood = likelihood[frame]
```

iii. The docstring records the main justification: the agent wanted to “sample the DLC tongue trace at each neural-bin center” and took the preceding camera frame “as in the repository’s marker-alignment code.” It also chose a stricter visibility threshold of `0.9`.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code uses session-wide percentiles `q40` and `q60` computed from raw visible frames. Sampled visible values below `q40` become class `0`, values from `q40` through `q60` become class `1`, values above `q60` become class `2`, and anything not visible becomes class `3`.

ii.
```python
categories = np.full(target_times.shape, 3, dtype=np.uint8)
if np.isfinite(q40):
    categories[visible & (sampled_y < q40)] = 0
    categories[visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
    categories[visible & (sampled_y > q60)] = 2
```

iii. The trajectory does not add much beyond the code comments and top-level docstring. The explicit recorded rationale is that the percentiles should come from “all visible frames” in the session, not from later bin means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Instead of bin-averaging tongue samples over the same 50 ms intervals as the neural data, the agent aligns tongue position by evaluating one preceding camera frame at each neural-bin center `go_time + BIN_CENTERS`. It also refuses to carry a frame forward if it is more than 20 ms old.

ii.
```python
target_times = go_times[:, None] + BIN_CENTERS[None, :]
frame = np.searchsorted(camera_times, target_times, side="right") - 1
valid_frame = frame >= 0
frame = np.clip(frame, 0, len(camera_times) - 1)
valid_frame &= (target_times - camera_times[frame]) <= 0.020
```

iii. The recorded justification is the same one given in the docstring: sample the tongue trace at neural-bin centers using the preceding frame, matching the agent’s reading of the repository’s marker-alignment logic.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several data-quality issues explicitly. Non-string string-like fields and `NaN` text entries are decoded to empty strings; optional photostimulation strings like `"N/A"` become `NaN`; sessions with no good units are skipped; trial/go-cue count mismatches and missing observation intervals raise errors; all-zero population trials are treated as acquisition dropouts and removed; and missing or stale tongue samples are assigned the `not visible` class.

ii.
```python
def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return str(value)

def _optional_float(value) -> float:
    text = _decode(value)
    return np.nan if text in {"", "N/A", "nan"} else float(text)
```

```python
good_rows = np.flatnonzero(classification == "good")
if good_rows.size == 0:
    return None
...
if len(all_go_times) != n_source_trials:
    raise ValueError(...)
...
if n_population_dropouts:
    rates = rates[population_recorded]
    valid_trial_idx = valid_trial_idx[population_recorded]
```

```python
valid_frame &= (target_times - camera_times[frame]) <= 0.020
...
categories = np.full(target_times.shape, 3, dtype=np.uint8)
```

iii. The trajectory is explicit that all-zero trials created by recording gaps should not be treated as biological silence, and that non-recorded or unreliable tongue samples should become an explicit missing/hidden class rather than be silently imputed.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work in this code is opening and reading each NWB file, materializing full `units/spike_times` and tongue-tracking arrays, and looping over good units in `_bin_good_units` to count spikes into bins. The per-trial input/output construction loop is smaller but still repeated for every trial of every session.

ii.
```python
with h5py.File(path, "r") as nwb:
    ...
    all_spikes = nwb["units/spike_times"][:]
    spike_ends = nwb["units/spike_times_index"][:]
    rates = _bin_good_units(all_spikes, spike_ends, good_rows, go_times)
    ...
    tongue_data = nwb[f"{tongue_path}/data"][:]
    camera_times = nwb[f"{tongue_path}/timestamps"][:]
```

```python
for out_unit, unit_row in enumerate(good_rows):
    ...

for trial in range(n_trials):
    ...
```

iii. The trajectory does not provide a detailed profile, but it repeatedly describes the full 50 GB conversion as the dominant workload and highlights the full-data neural binning and session rewrites as the heavy steps.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been reduced or vectorized further: the per-good-unit spike-binning loop in `_bin_good_units`, the per-good-unit observation-interval scan used to infer `common_start`/`common_stop`, and the per-trial loop that separately builds `inputs` and `outputs`.

ii.
```python
for out_unit, unit_row in enumerate(good_rows):
    ...
```

```python
for unit_row in good_rows:
    obs_start = (
        0 if unit_row == 0 else int(observation_ends[unit_row - 1])
    )
    obs_stop = int(observation_ends[unit_row])
    ...
```

```python
for trial in range(n_trials):
    time_from_tone = (
        go_times[trial] + BIN_CENTERS - tone_onset[trial]
    ).astype(np.float32)
    ...
    trial_output = np.empty((4, N_BINS), dtype=np.uint8)
```

iii. The trajectory does not discuss vectorization directly. This answer is inferred from the implementation the agent chose and from its general emphasis on getting a correct full-dataset pass working under time constraints.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several small computations per trial and per unit. It recalculates `go_times[trial] + BIN_CENTERS` inside the per-trial input loop, recomputes and fills per-trial output arrays with labels that are constant across all 80 bins, and scans the retained units twice in separate loops, once for trial-coverage mapping and once for spike binning.

ii.
```python
for trial in range(n_trials):
    time_from_tone = (
        go_times[trial] + BIN_CENTERS - tone_onset[trial]
    ).astype(np.float32)
    ...
    absolute_centers = go_times[trial] + BIN_CENTERS
```

```python
for trial in range(n_trials):
    trial_output = np.empty((4, N_BINS), dtype=np.uint8)
    trial_output[0, :] = _trial_choice(
        instruction[trial], outcome_text[trial]
    )
    trial_output[1, :] = outcome_map[outcome_text[trial]]
    trial_output[2, :] = 1 if early_text[trial] == "early" else 0
```

iii. The trajectory does not explicitly call out repeated work. This is visible from the structure of the final implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent does some bookkeeping that is not needed by the downstream decoder itself: it computes and stores many session-level curation counters, stores `source_file`, `skipped_sessions`, and tongue percentile metadata, and spends work inferring common observation intervals and population-dropout counts purely to support conservative curation and metadata. It also computes `absolute_centers` per trial only as a temporary intermediate for photostimulation alignment.

ii.
```python
"session_info": {
    "source_file": str(path.relative_to(path.parents[1])),
    "n_source_trials": int(n_source_trials),
    "n_ephys_mask_trials": int(n_ephys_trials),
    "n_common_good_trials": int(len(common_good_idx)),
    "n_excluded_water_trials": int(np.count_nonzero(water_mask)),
    "n_population_recording_dropouts": n_population_dropouts,
    "n_trials": int(n_trials),
    "n_good_units": int(len(good_rows)),
    "tongue_y_40th_percentile": tongue_q40,
    "tongue_y_60th_percentile": tongue_q60,
},
```

```python
"metadata": {
    ...
    "bin_centers_relative_to_go_s": BIN_CENTERS.tolist(),
    ...
    "session_info": [
        session["session_info"] for session in converted_sessions
    ],
    "skipped_sessions": skipped_sessions,
},
```

iii. The trajectory frames this extra work as defensive bookkeeping rather than a core requirement of the decoder task. The agent wanted to explain conservative curation choices and keep audit metadata, even though those values are not used by the downstream decoder model.
