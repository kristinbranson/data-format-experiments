# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every session by globbing `/app/data/sub-*/*.nwb`, sorting the paths, and opening each NWB file with `h5py` inside `process_session`. Within each file it reads the `units` table, the `intervals/trials` table, behavioral event timestamps, and tongue-tracking time series.

ii. 
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))

for idx, file_path in enumerate(target_files, start=1):
    result = process_session(file_path, show_processing=args.show_processing and plotted < 2)
```

```python
with h5py.File(file_path, "r") as h5:
    units = h5["units"]
    trials = h5["intervals"]["trials"]
    go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials]
    tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md`, the agent says it switched from PyNWB to direct `h5py` reads for speed and to avoid object-construction overhead. The trajectory shows it explored the reference code first, then intentionally implemented a direct NWB/HDF5 loading path for the conversion script.

## 1-b. How are the data split into subjects?

i. Subjects are split by the parent folder name of each NWB file. The code strips the `sub-` prefix to produce a subject id, and later deduplicates and sorts subject ids when building the final dataset.

ii. 
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id
```

```python
subjects = sorted({r.subject_id for r in results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes explicitly say the dataset is DANDI-style with one `sub-<subject_id>/` directory per mouse, and that `subjects` / `subject_idx` should be derived from those folder names.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session id is the file stem, and one `SessionResult` is produced per successfully processed file.

ii. 
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id
```

```python
for idx, file_path in enumerate(target_files, start=1):
    result = process_session(file_path, show_processing=args.show_processing and plotted < 2)
    if result is None:
        continue
    results.append(result)
```

iii. The notes say the raw dataset contains one NWB file per session and one subject folder per mouse, so the agent adopted the obvious one-file-per-session split.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table and associated event series. The code first reads all behavioral trials, then selects a subset of trial indices and slices every per-trial array with the same `selected_trial_idx`.

ii. 
```python
trials = h5["intervals"]["trials"]
n_behavior_trials = int(len(trials["id"]))
trial_start_all = trials["start_time"][:].astype(np.float64)
trial_stop_all = trials["stop_time"][:].astype(np.float64)
trial_instruction_all = np.char.lower(decode_str_array(trials["trial_instruction"][:]))
```

```python
selected_trial_idx, session_obs_start, session_obs_stop = select_trial_indices(
    go_times_all=go_times_all,
    good_unit_obs_intervals=unit_obs_intervals,
)
trial_start = trial_start_all[selected_trial_idx]
trial_stop = trial_stop_all[selected_trial_idx]
go_times = go_times_all[selected_trial_idx]
```

iii. In the notes, the agent says it originally assumed the ephys-backed trials were just the first `N` behavioral trials, then changed to explicit trial-index selection after finding offset sessions in the raw data.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply the reference `regular_trial_mask` that excludes early-lick, no-response, and stimulation trials. Instead, it keeps those trial types because they are decoder targets/inputs, and filters trials by whether the full `[-2.5, +1.5] s` go-cue window is supported by session observation intervals. It also optionally uses `units/is_good_trials` when its column count matches the selected trial block, zeros invalid unit-trial activity, and finally drops trials whose neural tensor is all zeros.

ii. 
```python
def select_trial_indices(
    go_times_all: np.ndarray,
    good_unit_obs_intervals: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    session_obs_start = float(np.min(good_unit_obs_intervals[:, 0]))
    session_obs_stop = float(np.max(good_unit_obs_intervals[:, 1]))
    full_window_mask = (
        (go_times_all + WINDOW_START_S >= session_obs_start)
        & (go_times_all + WINDOW_END_S <= session_obs_stop)
    )
    trial_idx = np.flatnonzero(full_window_mask)
    return trial_idx, session_obs_start, session_obs_stop
```

```python
is_good_trials_raw = units["is_good_trials"][good_unit_idx, :n_recorded_trials].astype(bool)
uses_direct_is_good_trials = is_good_trials_raw.shape[1] == len(selected_trial_idx)
if uses_direct_is_good_trials:
    is_good_trials = is_good_trials_raw.copy()
else:
    is_good_trials = np.ones((len(good_unit_idx), len(selected_trial_idx)), dtype=bool)
```

```python
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
    input_trials = [trial for keep, trial in zip(nonzero_trial_mask, input_trials) if keep]
    output_trials = [trial for keep, trial in zip(nonzero_trial_mask, output_trials) if keep]
```

iii. The notes say the agent intentionally retained early-lick and photostim trials because `early_lick` is an output and `photostim_on` is an input. Its Step 10 review says it replaced a simpler prefix-of-trials assumption with an `obs_intervals`-based validity rule after examining offset sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw spike times in `units/spike_times`, restricted to units whose `classification` is `"good"`. The code also uses `spike_times_index` to unpack the ragged spike array and uses `is_good_trials`, `obs_intervals`, and `anno_name` as supporting QC/metadata fields.

ii. 
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
brain_region_names = decode_str_array(units["anno_name"][good_unit_idx])
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)
```

iii. The notes repeatedly justify `classification == good` by the papers’ and reference code’s use of classifier-approved good units as the analysis population.

## 2-b. How is the `neural` data processed?

i. For each good unit, the agent bins spikes into non-overlapping 50 ms bins over `[-2.5, 1.5]` seconds relative to go cue, converts counts to firing rates in Hz, and stores the result as `float16`. The tensor is built as `(n_trials, n_units, n_bins)` and then converted to a list of per-trial `(n_units, n_bins)` matrices.

ii. 
```python
bin_edges_rel, bin_centers_rel = bin_edges_and_centers()
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)

for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
    fr = counts / bin_width
    neural_session[:, unit_pos, :] = fr.astype(np.float16)
```

iii. In the notes, the agent explicitly maps this to the reference `sliding_histogram` / `process_one_area` logic, while noting the task-specific deviation to 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only units with `classification == "good"`, skips any session with zero such units, and tries to zero unit-trial firing rates for invalid trials using `is_good_trials` and `obs_intervals`.

ii. 
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None
```

```python
obs_start = unit_obs_intervals[unit_pos, 0]
obs_stop = unit_obs_intervals[unit_pos, 1]
valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
invalid_trials = ~valid_obs
if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
```

iii. The notes justify the good-unit filter by the paper’s classifier-based QC. They also say the agent wanted an explicit raw-data validity rule from `is_good_trials` / `obs_intervals`, although that is not how the reference MATLAB preprocessing code performs trial curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go cue onset. For each selected trial, the code constructs absolute bin edges by adding the relative `[-2.5, 1.5]` window to that trial’s `go_time`.

ii. 
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5

abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)
```

iii. The notes say the paper/code align to go cue and that the decoder instructions explicitly require go-cue alignment over `[-2.5, +1.5] s`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins. The code bins directly from raw spikes into that resolution; there is no second-stage rebinning.

ii. 
```python
BIN_WIDTH_S = 0.05

def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
    centers = edges[:-1] + BIN_WIDTH_S / 2.0
    return edges, centers
```

```python
"time_bin_size": 50.0,
"time_bin_size_s": BIN_WIDTH_S,
```

iii. The agent notes this is a deliberate deviation from the reference preprocessing windows because the task instructions explicitly asked for 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` for tone/sample onset, `go_start_times` for alignment, and the trial `start_time` / `stop_time` fields to determine which sample events belong to which trial.

ii. 
```python
sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
trial_start_all = trials["start_time"][:].astype(np.float64)
trial_stop_all = trials["stop_time"][:].astype(np.float64)
```

iii. The notes describe this mapping explicitly and say the raw `sample_start_times` series was chosen because replayed early-lick trials can contain multiple sample epochs.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the code finds sample-start events within that trial window, takes the earliest one, subtracts the trial go cue to get `sample_onset_rel_go`, then fills the first input channel with `bin_center_rel_go - sample_onset_rel_go`. If no sample event is found, it falls back to a hard-coded `-1.85 s` relative onset.

ii. 
```python
sample_onset_abs = np.empty(n_trials, dtype=np.float64)
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

```python
sample_onset_rel_go = sample_onset_abs - go_times
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The notes say the agent intentionally used the earliest sample start to preserve replay-induced timing shifts in early-lick trials, and added the `-1.85 s` fallback only for missing-event edge cases.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same `bin_centers_rel` used for neural activity, so the tone-onset channel is a per-bin time series aligned to the same go-cue-centered time axis as `neural`.

ii. 
```python
inp = np.zeros((2, n_bins), dtype=np.float32)
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The notes say the decoder inputs and neural data share a common go-cue-aligned binning grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the per-trial `photostim_onset` and `photostim_duration` fields in the NWB trial table, together with `trial_start` and `go_times` to convert the onset/offset into go-cue-relative time.

ii. 
```python
photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
photostim_duration_all = decode_str_array(trials["photostim_duration"][:])
```

```python
stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
stim_rel_off = stim_rel_on + float(photostim_duration[trial])
```

iii. The notes say trial-table photostim onset/duration matched the event-series timestamps in spot checks, so the agent used the trial table as the primary source.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The agent turns photostimulation into a binary time-varying input. Trials with `photostim_onset == "N/A"` stay all zeros; otherwise the second input row is `1` on bin centers between the computed onset and offset and `0` elsewhere.

ii. 
```python
inp = np.zeros((2, n_bins), dtype=np.float32)
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
    inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The notes say the agent chose a time-varying binary series because the decoder specification asked for whether photostimulation is on at every time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned on the same go-cue-centered 50 ms bins used for neural firing rates.

ii. 
```python
inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The notes explicitly say the photostim input is represented on the shared neural bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the left-lick and right-lick event timestamp streams, trial start/stop, and go cue. If no lick is found, the code falls back to `trial_instruction`.

ii. 
```python
def infer_choice_for_trial(
    trial_start: float,
    trial_stop: float,
    go_time: float,
    instruction: str,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> tuple[int, str]:
```

```python
choice_val, source = infer_choice_for_trial(
    trial_start=trial_start[trial],
    trial_stop=trial_stop[trial],
    go_time=go_times[trial],
    instruction=str(trial_instruction[trial]),
    left_lick_times=left_lick_times,
    right_lick_times=right_lick_times,
)
```

iii. The notes say this was the main ambiguous field: responded trials can use lick timing, but ignore trials have no post-go lick, so the agent added a documented fallback hierarchy.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code first looks for the first post-go lick and labels left as `0`, right as `1`. If no post-go lick exists, it uses the first lick anywhere in the trial. If the trial has no lick at all, it uses the instructed side as a fallback. The chosen scalar is then repeated across all time bins in the output matrix.

ii. 
```python
left_post = left_lick_times[left_go:left_stop]
right_post = right_lick_times[right_go:right_stop]
if len(left_post) or len(right_post):
    first_left = left_post[0] if len(left_post) else np.inf
    first_right = right_post[0] if len(right_post) else np.inf
    return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")
```

```python
left_any = left_lick_times[left_start:left_stop]
right_any = right_lick_times[right_start:right_stop]
if len(left_any) or len(right_any):
    ...
return (0 if instruction == "left" else 1, "instruction_fallback")
```

iii. `CONVERSION_NOTES.md` says the agent chose this fallback chain because the target format still required a binary choice output even on ignore trials with no lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` field.

ii. 
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
outcome = outcome_all[selected_trial_idx]
```

iii. The notes say this field could be used directly from the raw trial labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent lowercases the strings, maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the categorical value across all bins for that trial.

ii. 
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
```

```python
out = np.empty((4, n_bins), dtype=np.int8)
out[1] = outcome_trials[trial]
```

iii. The notes explicitly document this mapping as matching the decoder specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` field.

ii. 
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
early_lick = early_lick_all[selected_trial_idx]
```

iii. The notes say this field is exposed directly in NWB and was retained because `early_lick` itself is a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent maps `no early -> 0` and `early -> 1`, then repeats the per-trial categorical value across all bins.

ii. 
```python
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
```

```python
out = np.empty((4, n_bins), dtype=np.int8)
out[2] = early_trials[trial]
```

iii. The notes say the agent intentionally did not exclude early-lick trials because this variable is one of the requested outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data[:, 1]` for the y-coordinate and the corresponding `timestamps`.

ii. 
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The notes say the side-camera tongue marker was chosen because the reference marker-alignment code explicitly uses `tongue_y` from side-view tracking.

## 8-b. How is `output` *Tongue y-position* derived from?

i. The agent aligns tongue timestamps to each trial’s go cue, bins them into the same 50 ms windows as neural data, and within each bin takes the last available tongue-y sample rather than averaging.

ii. 
```python
def bin_tongue_y(
    tongue_timestamps: np.ndarray,
    tongue_y: np.ndarray,
    go_times: np.ndarray,
    bin_edges_rel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
    end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
```

```python
valid = end_idx >= start_idx
binned = np.full(end_idx.shape, np.nan, dtype=np.float32)
binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
```

iii. The notes explicitly justify the “last frame in bin” rule by citing the reference `align_markers_between_lims` implementation.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After session-level binning and alignment, the code pools all finite binned tongue-y values for the session, computes the 40th and 60th percentiles, then discretizes each time point into three categories.

ii. 
```python
valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
tongue_p40 = float(np.percentile(valid_values, 40))
tongue_p60 = float(np.percentile(valid_values, 60))
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```

iii. The notes say this follows the task specification exactly: per-session discretization with 40th and 60th percentile thresholds.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is aligned to the same go-cue-centered bin edges used for neural firing rates, so tongue bins and neural bins share the exact time axis.

ii. 
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
tongue_y_binned, tongue_valid = bin_tongue_y(
    tongue_timestamps=tongue_timestamps,
    tongue_y=tongue_y,
    go_times=go_times,
    bin_edges_rel=bin_edges_rel,
)
```

iii. The notes say the converted tongue signal should follow the same shared time axis as neural and other decoder variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles edge cases with a mix of skipping, fallback values, and hard errors. Sessions with zero good units are skipped; empty `anno_name` or completely missing aligned tongue data cause an exception; missing sample onset falls back to `-1.85 s`; `photostim_onset == "N/A"` means no stimulation; trials with no post-go lick fall back to any lick or trial instruction for choice; and all-zero neural trials are removed after neural construction.

ii. 
```python
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None

if np.any(brain_region_names == ""):
    raise ValueError(f"{session_id}: found kept good units with empty anno_name")
```

```python
if len(events):
    sample_onset_abs[trial] = events[0]
else:
    sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

```python
if len(valid_values) == 0:
    raise ValueError(f"{session_id}: no valid tongue_y values after alignment")
```

iii. The notes describe these as pragmatic fixes for raw-data irregularities, especially replayed sample epochs, no-lick ignore trials, and sessions lacking usable units.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is the per-unit spike binning loop, which does a `searchsorted` over all trial/bin edges for every kept neuron in every session. Large HDF5 reads of `spike_times` and tongue-tracking arrays are also substantial.

ii. 
```python
spike_times = units["spike_times"][:].astype(np.float64)
...
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
```

iii. In Step 6 notes, the agent explicitly identifies full-object loading and spike binning as the main performance bottlenecks and says it chose `h5py` plus vectorized `searchsorted` to keep runtime manageable.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several trial-level loops remain vectorizable: the sample-onset extraction loop, the per-trial input construction loop, the choice inference loop, and the per-trial output assembly loop. The per-unit spike loop is harder to remove entirely because the spike trains are ragged, but it is still the main scalar loop left in the code.

ii. 
```python
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    ...
```

```python
for trial in range(n_trials):
    inp = np.zeros((2, n_bins), dtype=np.float32)
    ...
```

```python
for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(...)
```

iii. The notes mention that earlier Python-heavy loops would have been too slow on the full dataset, and the remaining loops are visible directly in the current script.

## 10-c. What processing does the code repeat multiple times?

i. The code recomputes several quantities more than once: photostim windows are computed once for inputs and again for plotting; lick-time searches happen during choice inference and again in plot construction; and many arrays are re-sliced when `nonzero_trial_mask` drops zero-neural trials.

ii. 
```python
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
```

```python
if str(photostim_onset[trial_plot_index]) != "N/A":
    stim_rel_on = trial_start[trial_plot_index] + float(photostim_onset[trial_plot_index]) - go
    stim_window = (stim_rel_on, stim_rel_on + float(photostim_duration[trial_plot_index]))
```

```python
if dropped_zero_trials:
    sample_onset_rel_go = sample_onset_rel_go[nonzero_trial_mask]
    tongue_y_binned = tongue_y_binned[nonzero_trial_mask]
    ...
    output_trials = [trial for keep, trial in zip(nonzero_trial_mask, output_trials) if keep]
```

iii. This follows from direct inspection of `convert_data.py`; the notes also mention that additional raw-data sanity checks repeatedly re-ran parts of the conversion during development.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes or loads several values that do not survive into the final converted dataset: `tongue_likelihood` is only used for plotting, `tongue_valid` is computed but not otherwise used, delay/sample slices are mainly for plot payloads, and large `stats` / `plot_payload` structures are built for logging rather than decoder inputs.

ii. 
```python
tongue_likelihood = tongue_values[:, 2]
...
tongue_y_binned, tongue_valid = bin_tongue_y(...)
```

```python
plot_payload = {
    ...
    "tongue_likelihood_window": tongue_likelihood[tongue_window],
}
```

```python
stats = {
    "choice_source_counts": choice_source_counts,
    "sample_onset_fallbacks": sample_onset_fallbacks,
    "tongue_thresholds": [tongue_p40, tongue_p60],
    ...
}
```

iii. This is mostly visible from the code itself. The notes frame plotting and statistics as diagnostics and validation aids rather than required downstream decoder content.
