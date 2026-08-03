# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates the NWB files by walking `/app/data` with `os.listdir`, sorting subject directories and filenames, and opening each `.nwb` file directly with `h5py`. Inside each file it reads HDF5 datasets by path, such as `units/classification`, `intervals/trials/*`, `acquisition/BehavioralEvents/*`, and `acquisition/BehavioralTimeSeries/*`.

ii. 
```python
def load_sorted_nwb_paths(data_root: str) -> list[str]:
    paths = []
    for subject in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for filename in sorted(os.listdir(subject_path)):
            if filename.endswith(".nwb"):
                paths.append(os.path.join(subject_path, filename))
    return paths
```

```python
with h5py.File(path, "r") as f:
    classifications = decode_array(f["units/classification"][:]).astype(str)
    ...
    trial_group = f["intervals/trials"]
    ...
    behavioral_events = f["acquisition/BehavioralEvents"]
```

iii. `CONVERSION_NOTES.md` says the released dataset is a collection of NWB sessions and that session inclusion follows the QC labels visible in those NWB files. The notes emphasize that the agent verified the full collection had 174 NWB files and 173 kept sessions after QC, so the direct HDF5 traversal was treated as a complete read of the dataset.

## 1-b. How are the data split into subjects?

i. The AI uses the parent directory name of each NWB file, such as `sub-440956`, as the subject identifier. It never reads `nwb.subject.subject_id`; instead it derives subject membership from the filesystem layout and then builds `subjects` and `subject_idx` from those path-based IDs.

ii.
```python
def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

```python
return {
    "session_id": get_session_id_from_path(path),
    "subject": get_subject_from_path(path),
    ...
}
```

```python
subjects = sorted({session["subject"] for session in session_results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session["subject"]])
```

iii. There is no explicit separate justification in the code comments. The closest justification appears in the notes and README: the dataset is stored under one subject directory per animal, so the agent treated the directory structure as authoritative for grouping sessions by mouse.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list, and the session ID is taken from the filename stem rather than from `nwb.identifier`.

ii.
```python
def get_session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
```

```python
all_paths = load_sorted_nwb_paths(args.data_root)
...
for idx, path in enumerate(all_paths, start=1):
    session_id = get_session_id_from_path(path)
    ...
```

iii. The README states that the workspace "converts the NWB sessions in `data/`", which implies the file boundary is the session boundary. The notes also report exactly one excluded NWB file and 173 kept sessions, reinforcing that one file equals one session.

## 1-d. How are the data split into trials?

i. The AI does not use all rows of the behavioral trials table as output trials. Instead, it first reads the first good unit's `obs_intervals`, assumes those intervals define the ephys-covered trials, matches those start/stop pairs back to `intervals/trials/start_time` and `stop_time`, and uses the matched behavioral rows as the session's trial list.

ii.
```python
n_ephys_trials = int(f["units/is_good_trials"].shape[1])
trial_group = f["intervals/trials"]
behavior_trial_start = np.asarray(trial_group["start_time"][:], dtype=np.float64)
behavior_trial_stop = np.asarray(trial_group["stop_time"][:], dtype=np.float64)
...
obs_intervals = np.asarray(f["units/obs_intervals"][:], dtype=np.float64)
obs_index = np.asarray(f["units/obs_intervals_index"][:], dtype=np.int64)
first_good_unit = int(good_unit_indices[0])
obs_start = 0 if first_good_unit == 0 else int(obs_index[first_good_unit - 1])
obs_stop = int(obs_index[first_good_unit])
first_unit_obs = obs_intervals[obs_start:obs_stop]
```

```python
trial_lookup = {
    (round(float(start), 4), round(float(stop), 4)): idx
    for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
}
trial_indices = []
for start, stop in first_unit_obs:
    key = (round(float(start), 4), round(float(stop), 4))
    ...
    trial_indices.append(trial_lookup[key])
trial_indices = np.asarray(trial_indices, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says that when the behavior table contains more trials than the ephys recording, the conversion "matches from the first good unit's `obs_intervals` back to the behavioral trial table" to avoid behavior-only trials with no neural recording. That is the agent's stated reason for defining trials from `obs_intervals` rather than from the full trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by keeping only the behavioral trials that match the first good unit's `obs_intervals`. It does not apply the reference solution's extra `free_water == 0` exclusion. The agent explicitly keeps early-lick, ignore, and photostim trials if they are ephys-covered.

ii.
```python
trial_start = behavior_trial_start[trial_indices]
trial_stop = behavior_trial_stop[trial_indices]
instructions = decode_array(trial_group["trial_instruction"][:])[trial_indices].astype(str)
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
```

iii. The notes say, "All ephys-covered trials are retained, including photostim, early-lick, and ignore trials," and "the conversion uses the first good unit's `obs_intervals` to select only the trials actually present in the recording." The full-dataset summary reports 93,310 kept trials, which reflects this looser filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived from `units/spike_times` plus `units/spike_times_index`, filtered by `units/classification == "good"`. Trial start/stop times and go-cue times are also used so spikes can be assigned to trials and aligned to the go cue.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
...
spike_counts = bin_spike_counts_for_good_units(
    spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
    spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
    good_unit_mask=good_unit_mask,
    trial_start=trial_start,
    trial_stop=trial_stop,
    go_times=go_times,
)
```

iii. The notes state, "Neural data: spike counts per 50 ms bin, converted to firing rates by dividing by `0.05 s`," and "Units are filtered with `classification == "good"`." That is the stated basis for the neural data construction.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 50 ms trial-relative bins, counts spikes with `np.add.at`, then converts counts to firing rates by multiplying by `1 / 0.05`. Spikes are first assigned to behavioral trials using `trial_start` and `trial_stop`, then shifted by the corresponding go-cue time to place them on the `[-2.5, 1.5)` go-aligned window.

ii.
```python
trial_idx = np.searchsorted(trial_start, spikes, side="right") - 1
valid = (trial_idx >= 0) & (trial_idx < n_trials)
if np.any(valid):
    valid_trial_idx = trial_idx[valid]
    valid &= spikes <= trial_stop[valid_trial_idx]
...
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
...
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
```

```python
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))
```

iii. The notes justify this as "spike counts per 50 ms bin, converted to firing rates by dividing by `0.05 s`." No smoothing, baseline subtraction, or normalization is described anywhere in the agent's materials.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `"good"` are kept. A session is dropped entirely if it has zero such units.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

iii. `CONVERSION_NOTES.md` explicitly says, "Units are filtered with `classification == "good"`, matching the published classifier-based QC workflow," and that exactly one NWB file with zero good units is removed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural activity to go-cue onset. For each spike assigned to a retained trial, it subtracts that trial's `go_time` and only keeps spikes whose relative times fall inside `[-2.5, 1.5)` before binning.

ii.
```python
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
...
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
...
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
```

iii. The notes and README both state that all trials are aligned to go cue onset and restricted to `[-2.5 s, +1.5 s)` using 50 ms bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins from `-2.5 s` to `+1.5 s`, giving 80 bins per trial. Neural spikes are rebinned into this fixed grid; no other temporal smoothing or resampling is applied.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
```

iii. The notes call out "50 ms bins" and "non-overlapping bins from `-2.5 s` to `+1.5 s`" as a core processing choice.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `sample_start_times` in `acquisition/BehavioralEvents`, together with the per-trial go cues. The agent takes the last sample/tone onset before each trial's go cue, constrained to lie after the matched trial start.

ii.
```python
sample_events = np.asarray(behavioral_events["sample_start_times"]["timestamps"][:], dtype=np.float64)
...
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
```

iii. The notes state, "Input `time_from_tone_onset_s`: bin centers expressed relative to the final sample/tone onset before the aligned go cue." That is the agent's stated rule.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding the last tone onset before each go cue, the AI computes the time value at each bin center as `BIN_CENTERS_S - sample_rel`, where `sample_rel = sample_start_times - go_times`. This yields go-aligned bin centers expressed as seconds since tone onset.

ii.
```python
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
...
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. The notes justify this as expressing the aligned bin centers relative to the final sample/tone onset before the go cue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI places this input on exactly the same 80 go-aligned bin centers used for neural firing rates. It is a `(n_trials, 80)` array built directly from `BIN_CENTERS_S`.

ii.
```python
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
...
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. The justification in the notes is implicit: the bins are defined around the go cue for all modalities, and `time_from_tone_onset_s` is just those same bins re-expressed on a tone-relative axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the per-trial `photostim_onset` and `photostim_duration` columns in `intervals/trials`, together with `start_time` and `go_times` so onset and offset can be converted into go-relative time.

ii.
```python
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
...
photostim, stim_trial_count = bin_photostim_series(
    trial_start=trial_start,
    go_times=go_times,
    onset_values=photostim_onset,
    duration_values=photostim_duration,
)
```

iii. The notes summarize this as "Input `photostim_on`: binary per bin, 1 if the photostim interval overlaps the bin."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI decodes trial-level onset and duration values, converts them from trial-start-relative time into go-relative time, and then marks a bin as 1 whenever the photostim interval overlaps any part of that 50 ms bin. Non-stim trials stay all zero because `"N/A"` becomes `None`.

ii.
```python
def as_float_or_none(value: Any) -> float | None:
    value = decode_scalar(value)
    if value in ("N/A", "", None):
        return None
    return float(value)
```

```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
photostim[trial_idx] = overlap.astype(np.float32)[0]
```

iii. The notes explicitly justify the binary series as an overlap test: "1 if the photostim interval overlaps the bin." No alternative rule is mentioned in the trajectory or README.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI converts each stim interval into go-relative coordinates by subtracting the trial's `go_time`, then compares that interval to the same fixed bin edges used for neural activity.

ii.
```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
```

iii. The notes and README both state that the whole dataset is go-cue aligned, so photostim is aligned by putting its onset and offset on that same go-cue axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice primarily from lick event timestamps, not from the trial outcome. It uses `left_lick_times`, `right_lick_times`, `trial_start`, `trial_stop`, `go_times`, and `trial_instruction`. It only falls back to the instructed side if no lick is found anywhere in the trial.

ii.
```python
def compute_choice_labels(
    trial_start: np.ndarray,
    trial_stop: np.ndarray,
    go_times: np.ndarray,
    instructions: np.ndarray,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> tuple[np.ndarray, Counter]:
```

```python
left_lick_times = np.asarray(behavioral_events["left_lick_times"]["timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(behavioral_events["right_lick_times"]["timestamps"][:], dtype=np.float64)
...
choice, choice_sources = compute_choice_labels(
    trial_start=trial_start,
    trial_stop=trial_stop,
    go_times=go_times,
    instructions=instructions,
    left_lick_times=left_lick_times,
    right_lick_times=right_lick_times,
)
```

iii. `CONVERSION_NOTES.md` justifies this directly: "Output `choice`: first post-go lick side when available; otherwise first lick anywhere in the trial; otherwise instructed side if the trial has no licks at all." The trajectory summary repeats that the conversion counted trials by those three source rules.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI scans left and right lick timestamps trial by trial. It first looks for the earliest post-go lick, then the earliest lick anywhere in the trial, then uses the instructed side if the trial is completely lick-free. It encodes only two classes, left `0` and right `1`, and repeats that per-trial code across all 80 bins.

ii.
```python
OUTPUT_VALUES = [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
]
```

```python
if left_post.size or right_post.size:
    left_first = left_post[0] if left_post.size else np.inf
    right_first = right_post[0] if right_post.size else np.inf
    choice[trial_idx] = 0 if left_first < right_first else 1
    ...
if left_all.size or right_all.size:
    ...
    choice[trial_idx] = 0 if left_first < right_first else 1
    ...
choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
```

```python
np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8)
```

iii. The notes provide the full justification quoted above and further report the full-dataset counts for the three fallback sources: 79,585 post-go, 1,063 any-trial, and 12,662 instruction fallback trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the `outcome` column of `intervals/trials`.

ii.
```python
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
```

iii. The notes state, "Output `outcome`: `ignore=0`, `miss=1`, `hit=2`," implying no extra derivation beyond reading the trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, and `hit` to integer codes `0`, `1`, and `2`, then repeats the per-trial code across all 80 bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
```

```python
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8)
```

iii. The notes explicitly say the outcome coding is `ignore=0`, `miss=1`, `hit=2`.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the `early_lick` column of `intervals/trials`.

ii.
```python
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
```

iii. The notes state, "Output `early_lick`: `no=0`, `yes=1`," which implies this comes directly from the trial-level early-lick label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `"no early"` to `0` and `"early"` to `1`, then repeats that per-trial value across all 80 bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
```

```python
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8)
```

iii. The notes explicitly document the `no=0`, `yes=1` coding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives this output from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` column 1 (`tongue_y`) and the corresponding `timestamps`. Unlike the reference solution, it does not use the likelihood column when constructing the categories.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
q40 = float(np.percentile(tongue_y_raw, 40))
q60 = float(np.percentile(tongue_y_raw, 60))
```

iii. The notes describe the output as "side-camera tongue `y` position" and say it is discretized "using the 40th and 60th percentiles of the raw session-wide tongue `y` trace." They do not mention the likelihood channel as an input to the categorization.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes session-wide 40th and 60th percentiles from the raw `tongue_y` trace. For each go-aligned bin it chooses the last sample inside the bin, or if the bin contains no sample it falls back to the latest sample before the bin end. It then thresholds that selected sample into 0/1/2 classes. There is no averaging within bins, no likelihood filtering, and no explicit missing-tongue class.

ii.
```python
q40 = float(np.percentile(tongue_y_raw, 40))
q60 = float(np.percentile(tongue_y_raw, 60))
```

```python
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
...
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
y_binned = tongue_y[sample_idx]
```

```python
tongue_disc = np.ones_like(y_binned, dtype=np.int8)
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
```

iii. `CONVERSION_NOTES.md` justifies this as matching "the reference marker-alignment style" by using "the last frame inside each 50 ms bin." The notes also say that previous-frame fallback was used to preserve alignment without introducing NaNs.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses three categories only: below the raw-session 40th percentile is `0`, between the 40th and 60th percentile is `1`, and above the 60th percentile is `2`. It does not create a fourth class for bins with no visible tongue.

ii.
```python
OUTPUT_VALUES = [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
]
```

```python
tongue_disc = np.ones_like(y_binned, dtype=np.int8)
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
```

iii. The notes justify the thresholds as "the 40th and 60th percentiles of the raw session-wide tongue `y` trace." They do not discuss a hidden or not-visible category; instead they state that fallback sampling avoids NaNs.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same go-aligned bin edges as the neural data. For each bin it finds the last tongue-tracking timestamp before the bin end, which may be a frame inside the bin or, if the bin has no sample, a frame from earlier than the bin. The resulting 80-bin sequence is therefore on the same nominal grid as neural activity, but some bins use previous-frame fallback rather than within-bin aggregation.

ii.
```python
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
...
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
y_binned = tongue_y[sample_idx]
```

iii. The notes state that this "preserves alignment without introducing NaNs" and that it was chosen to match the "reference marker-alignment style."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several irregularities explicitly. Sessions with zero good units are dropped. Behavioral trials not present in the first good unit's `obs_intervals` are excluded. `photostim_onset` values such as `"N/A"` are treated as no stimulation. Missing tongue samples inside a bin are handled by falling back to the previous frame instead of marking the bin missing. Duplicate or unordered `obs_intervals` mappings raise errors.

ii.
```python
if n_good_units == 0:
    return None
```

```python
if value in ("N/A", "", None):
    return None
```

```python
if np.unique(trial_indices).size != trial_indices.size:
    raise ValueError(f"obs_intervals map to duplicate behavioral trials in {path}")
if np.any(np.diff(trial_indices) <= 0):
    raise ValueError(f"obs_intervals are not strictly ordered in {path}")
```

```python
fallback_missing = int(np.sum(right_idx <= left_idx))
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
```

iii. The notes justify the main policies: `obs_intervals` matching is used to remove behavior-only trials, and previous-frame fallback is used for tongue bins "without introducing NaNs." There is no evidence that the agent treated missing tongue visibility as a separate categorical state.

## 10-a. What are the most time-consuming steps of the code?

i. The code suggests that the dominant costs are opening each NWB/HDF5 file, reading large spike and tongue arrays, and looping over units to bin spikes. Secondary costs come from repeated per-trial loops for choice, photostim, tongue fallback indexing, and final per-trial object assembly.

ii.
```python
with h5py.File(path, "r") as f:
    ...
    tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
    ...
    spike_counts = bin_spike_counts_for_good_units(
        spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
        spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
        ...
    )
```

```python
for unit_idx, unit_end in enumerate(spike_times_index):
    ...
```

iii. The agent did not include a dedicated performance analysis section, but the code structure and the full-run notes imply that session I/O and spike binning were the intended heavy steps.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the per-trial loop in `compute_choice_labels`, the per-trial loop in `bin_photostim_series`, the Python loop that matches `obs_intervals` back to the behavioral trial table, and the final per-trial loop that builds `neural`, `input`, and `output` lists. The per-unit spike loop is harder to remove because spike trains are ragged.

ii.
```python
for trial_idx in range(len(trial_start)):
    ...
```

```python
for trial_idx in range(n_trials):
    ...
```

```python
for start, stop in first_unit_obs:
    ...
```

```python
for trial_idx in range(n_trials):
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)
```

iii. There is no explicit justification for leaving these loops in Python. The notes only discuss correctness-oriented fallbacks, not vectorization choices.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans lick-event arrays trial by trial in `compute_choice_labels`, repeatedly performs per-trial photostim overlap checks, and repeatedly allocates full-length constant arrays with `np.full` for trial-level outputs. It also constructs both a full dataset and a sample dataset from overlapping session results.

ii.
```python
left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
right_all = right_lick_times[(right_lick_times >= start) & (right_lick_times <= stop)]
```

```python
for trial_idx in range(n_trials):
    onset = as_float_or_none(onset_values[trial_idx])
    ...
```

```python
np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8),
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8),
```

```python
full_data, full_summary = build_dataset(session_results)
...
sample_results = session_results[:sample_count]
sample_data, sample_summary = build_dataset(sample_results)
```

iii. The notes do not justify these repeated passes beyond reporting summary statistics for the full and sample datasets.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores many bookkeeping statistics that the downstream decoder does not use, including choice-source counts, stimulus counts, tongue fallback counts, per-session summaries, `n_behavior_trials`, `n_ephys_trials`, and percentile diagnostics. It also builds a separate sample dataset and summary as part of the same run. In addition, the tongue previous-frame fallback produces values specifically to avoid missing bins, even though those fallback details are not consumed downstream.

ii.
```python
"stats": {
    "n_trials": int(n_trials),
    "n_good_units": int(len(brain_region_labels)),
    "n_behavior_trials": int(n_behavior_trials),
    "n_ephys_trials": int(n_ephys_trials),
    "stim_trial_count": int(stim_trial_count),
    "choice_counts": dict(choice_counter),
    "choice_sources": dict(choice_sources),
    "outcome_counts": dict(outcome_counter),
    "early_counts": dict(early_counter),
    "tongue_q40": q40,
    "tongue_q60": q60,
    "tongue_bin_fallback_count": int(tongue_fallback_missing),
},
```

```python
metadata = {
    ...
    "choice_source_counts": dict(total_choice_sources),
    "stim_trial_count_total": int(total_stim_trials),
    "tongue_bin_fallback_count_total": int(total_tongue_fallback),
    "session_summary": session_summaries,
}
```

```python
sample_results = session_results[:sample_count]
sample_data, sample_summary = build_dataset(sample_results)
write_pickle(args.sample_out, sample_data)
```

iii. The notes show that these computations were mainly used for reporting and validation: they summarize choice-source counts, tongue fallback counts, and sample-dataset statistics in `CONVERSION_NOTES.md`. They were not required for the decoder inputs or outputs themselves.
