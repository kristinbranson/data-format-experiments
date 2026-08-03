# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing `sub-*/*.nwb` under `/app/data`, then opening each NWB/HDF5 file directly with `h5py`. Inside each file it reads the `units` table, the `intervals/trials` table, `BehavioralEvents`, and `Camera0_side_TongueTracking`.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
```

```python
with h5py.File(file_path, "r") as h5:
    units = h5["units"]
    trials = h5["intervals"]["trials"]
    go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
    tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. In Step 6 of `CONVERSION_NOTES.md`, the agent says it switched from PyNWB to direct `h5py` reads to reduce object-construction overhead and warning spam while still reading the same NWB contents.

## 1-b. How are the data split into subjects?

i. The AI treats the parent folder name `sub-<id>` as the subject identifier for each session. After all sessions are processed, it deduplicates and sorts those folder-derived ids to build `subjects`, then maps each session into `subject_idx`.

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
...
"subjects": subjects,
"subject_idx": np.array([subject_to_idx[r.subject_id] for r in results], dtype=np.int64),
```

iii. The notes describe the DANDI/NWB layout as one folder per subject and one file per session, so the agent used the folder-derived id instead of reading `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session ordering is the sorted file order, and the session id is the file stem rather than the NWB `identifier`.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
```

```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id
```

iii. In Step 2 and Step 5, the notes describe the dataset as one NWB file per session under subject folders, so the agent used file boundaries directly as session boundaries.

## 1-d. How are the data split into trials?

i. The AI starts from the full behavioral trial table (`intervals/trials`) and the `go_start_times` event series. It then applies a selected-trial index mask and slices every per-trial array with that mask, so the kept trials are whichever behavioral trials survive the neural-support filter.

ii.
```python
trials = h5["intervals"]["trials"]
n_behavior_trials = int(len(trials["id"]))
trial_start_all = trials["start_time"][:].astype(np.float64)
trial_stop_all = trials["stop_time"][:].astype(np.float64)
...
go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
```

```python
selected_trial_idx, session_obs_start, session_obs_stop = select_trial_indices(
    go_times_all=go_times_all,
    good_unit_obs_intervals=unit_obs_intervals,
)
trial_start = trial_start_all[selected_trial_idx]
trial_stop = trial_stop_all[selected_trial_idx]
trial_instruction = trial_instruction_all[selected_trial_idx]
...
go_times = go_times_all[selected_trial_idx]
```

iii. The Step 10 notes say the agent discovered that the ephys-backed trials were not always a simple behavioral-trial prefix, so it switched to selecting behavioral trials from raw go-cue timing plus observation-interval support.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials whose full `[-2.5, +1.5] s` go-aligned window falls inside the session-wide minimum/maximum of the selected `obs_intervals` rows. It also uses `units/is_good_trials` when its column count matches the selected trial count, otherwise it zeros invalid unit-trials based on per-unit observation bounds. After neural binning it drops any trial whose entire neural tensor is zero.

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
```

```python
is_good_trials_raw = units["is_good_trials"][good_unit_idx, :n_recorded_trials].astype(bool)
uses_direct_is_good_trials = is_good_trials_raw.shape[1] == len(selected_trial_idx)
...
if uses_direct_is_good_trials:
    invalid_trials = ~is_good_trials[unit_pos]
else:
    obs_start = unit_obs_intervals[unit_pos, 0]
    obs_stop = unit_obs_intervals[unit_pos, 1]
    valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
    invalid_trials = ~valid_obs
if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
```

```python
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
    ...
```

iii. The notes justify this as a fix for a trial-alignment bug: the agent says many sessions had offset ephys-backed trial blocks, so it preferred a raw go-time plus `obs_intervals` validity rule over assuming the first `N` behavioral trials matched the recorded ones.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units/spike_times` for units whose `classification` is `"good"`. It also uses `spike_times_index` to recover each unit’s spike train and `go_start_times` to define the trial-aligned bin edges.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
...
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)
```

```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)
```

iii. The notes state that the reference ephys path uses raw spike times with classifier-selected good units, and that the decoder task requires rebinning those spikes around go cue.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into non-overlapping 50 ms bins over `[-2.5, 1.5]` s relative to go cue, computes counts by `searchsorted` plus `np.diff`, divides by bin width to get Hz, and stores the per-trial matrices as `float16`.

ii.
```python
def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
    centers = edges[:-1] + BIN_WIDTH_S / 2.0
    return edges, centers
```

```python
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
    fr = counts / bin_width
    ...
    neural_session[:, unit_pos, :] = fr.astype(np.float16)
```

iii. In Step 5 and Step 6, the notes say the agent intentionally kept the reference principle of go-cue alignment and spike-based firing rates, but changed the binning to 50 ms because the task explicitly required that output format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == "good"` and drops sessions with zero such units. It also zeros neural activity for unit-trial pairs judged invalid by `is_good_trials` or the fallback observation-window check.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None
```

```python
if uses_direct_is_good_trials:
    invalid_trials = ~is_good_trials[unit_pos]
else:
    obs_start = unit_obs_intervals[unit_pos, 0]
    obs_stop = unit_obs_intervals[unit_pos, 1]
    valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
    invalid_trials = ~valid_obs
if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
```

iii. The notes repeatedly justify `classification == good` as the intended NWB representation of the classifier-based QC used in the papers. The extra unit-trial masking is justified as protection against partially unsupported neural windows.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go cue by adding fixed relative bin edges to each trial’s `go_start_times` timestamp. Spike times are already in the same absolute session clock, so they are binned directly against those absolute edges.

ii.
```python
go_times = go_times_all[selected_trial_idx]
...
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)
```

iii. The notes say the conversion should preserve the reference’s go-cue alignment; the agent treated all streams as already on the NWB session clock and therefore only needed to place a go-relative bin grid on that clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins and 80 timepoints per trial across the `[-2.5, 1.5]` s window. This is a fresh rebinning from raw spike times; there is no secondary temporal smoothing or resampling step.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
```

```python
def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
    centers = edges[:-1] + BIN_WIDTH_S / 2.0
    return edges, centers
```

iii. The notes explicitly call this an intentional deviation from the paper’s published decoder settings, justified only by the task instruction that demanded 50 ms bins over this exact window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times`, `trial_start`, `trial_stop`, and `go_start_times`. It first finds all sample-start events falling inside each trial window, then uses the earliest such event for that trial; if none are found, it uses a fixed fallback of `-1.85 s` relative to go cue.

ii.
```python
sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
...
sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
```

```python
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

iii. In Step 5, the notes say the agent chose the earliest sample event because it thought that was the most faithful raw-data representation of replay-extended early-lick trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After computing a per-trial sample onset, the AI subtracts that onset from each shared bin center to create a continuous time-since-tone vector. If no sample event is found within the trial window, it inserts a fallback onset at `go - 1.85 s`.

ii.
```python
EXPECTED_SAMPLE_ONSET_REL_GO = -1.85
...
sample_onset_rel_go = sample_onset_abs - go_times
```

```python
inp = np.zeros((2, n_bins), dtype=np.float32)
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The notes justify the fallback as a safeguard for missing events and describe the resulting quantity as a time-varying decoder input rather than a one-hot event.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI places this input on the same per-trial 50 ms bin centers used for neural data. The only trial-specific shift is the estimated tone onset relative to go cue.

ii.
```python
bin_edges_rel, bin_centers_rel = bin_edges_and_centers()
...
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The notes say all streams are put on the common go-cue-aligned bin grid so that bin `k` means the same interval for neural activity and inputs.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `photostim_onset` and `photostim_duration` in the trial table, together with `trial_start` and `go_times` so that the stimulation window can be re-expressed relative to go cue.

ii.
```python
photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
photostim_duration_all = decode_str_array(trials["photostim_duration"][:])
...
photostim_onset = photostim_onset_all[selected_trial_idx]
photostim_duration = photostim_duration_all[selected_trial_idx]
```

```python
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
```

iii. The notes say the agent checked ambiguous timing fields directly in the NWB files and chose the trial-table onset/duration representation as the clean per-trial source.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts stimulation into a binary time series: each bin is `1` if its center lies in `[stim_rel_on, stim_rel_off)`, otherwise `0`. Non-stimulated trials remain all zero because the code only writes the vector when onset is not `"N/A"`.

ii.
```python
inp = np.zeros((2, n_bins), dtype=np.float32)
...
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
    inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. In Step 5, the notes explicitly justify a time-varying binary series rather than a per-trial flag because the decoder specification asked whether photostimulation is on at every time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI converts the stimulation onset and offset from trial-relative times to go-relative times, then compares those go-relative times against the same bin centers used for the neural matrices.

ii.
```python
stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
stim_rel_off = stim_rel_on + float(photostim_duration[trial])
inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The notes treat go cue as the common alignment anchor, so any trial-relative input first gets converted onto that axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI does not use `outcome` to derive choice. Instead it infers choice from `left_lick_times`, `right_lick_times`, `trial_start`, `trial_stop`, `go_time`, and `trial_instruction`. It uses the first post-go lick if present, otherwise the first lick anywhere in the trial, and finally falls back to the instructed side on no-lick trials.

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
    ...
    if len(left_post) or len(right_post):
        ...
        return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")
    ...
    if len(left_any) or len(right_any):
        ...
        return (0, "any_lick") if first_left < first_right else (1, "any_lick")
    return (0 if instruction == "left" else 1, "instruction_fallback")
```

```python
for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(
        trial_start=trial_start[trial],
        trial_stop=trial_stop[trial],
        go_time=go_times[trial],
        instruction=str(trial_instruction[trial]),
        left_lick_times=left_lick_times,
        right_lick_times=right_lick_times,
    )
```

iii. Step 5 says the agent preferred “actual lick direction” from lick events, and introduced the fallback hierarchy because the task still required a binary choice label on ignore trials that have no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes left as `0` and right as `1`, with no separate “no lick” class. The chosen label is then repeated across all 80 bins for that trial.

ii.
```python
choice_trials = np.empty(n_trials, dtype=np.int8)
...
choice_trials[trial] = choice_val
```

```python
out = np.empty((4, n_bins), dtype=np.int8)
out[0] = choice_trials[trial]
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_p40", "p40_to_p60", "gt_p60"],
],
```

iii. The notes justify the binary coding by the decoder task statement (`left = 0`, `right = 1`) and treat the instruction-based fallback as the least-bad way to fill ignore trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI reads outcome directly from the trial-table `outcome` column after trial selection.

ii.
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
...
outcome = outcome_all[selected_trial_idx]
```

iii. The notes describe the raw trial labels as already carrying the needed `hit` / `miss` / `ignore` categories, so no derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI lowercases the strings, maps `ignore -> 0`, `miss -> 1`, `hit -> 2`, and repeats the per-trial code across all bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
```

```python
out[1] = outcome_trials[trial]
```

iii. The mapping follows the task specification exactly, and the notes say this was one of the unambiguous direct mappings.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI reads early-lick labels directly from the trial-table `early_lick` column after trial selection.

ii.
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
...
early_lick = early_lick_all[selected_trial_idx]
```

iii. The notes say early lick is exposed directly in the NWB trial table, so the converter retains it as a decoder target rather than filtering those trials away.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI lowercases the labels, maps `no early -> 0` and `early -> 1`, and repeats the result across all bins for the trial.

ii.
```python
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
```

```python
out[2] = early_trials[trial]
```

iii. The notes say this follows the task specification directly and is why the converter does not apply the reference “regular-trial” exclusion mask wholesale.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue position from `Camera0_side_TongueTracking/data[:, 1]` and its `timestamps`. It also reads `data[:, 2]` (`tongue_likelihood`) but does not use it in the discretization itself.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_likelihood = tongue_values[:, 2]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The Step 5 notes identify the side-camera tongue tracker as the source and explicitly say likelihood is kept only for diagnostics, not for filtering, to stay closer to the reference marker-alignment code path the agent inspected.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI aligns tongue timestamps into the go-cue-centered 50 ms bins and takes the last observed frame in each bin. It then pools all finite binned tongue values across the session, takes the 40th and 60th percentiles of that pooled set, and digitizes each trial/bin value into three classes.

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
    ...
    binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
    return binned, valid
```

```python
tongue_y_binned, tongue_valid = bin_tongue_y(...)
valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
tongue_p40 = float(np.percentile(valid_values, 40))
tongue_p60 = float(np.percentile(valid_values, 60))
```

iii. Step 5 says the agent intentionally used the “last frame within bin” rule because the reference marker-alignment code used a last-sample convention. It also says the per-session percentile split was taken directly from the decoder task.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI initializes every bin to class `0`, assigns class `1` to values in `[p40, p60]`, and class `2` to values above `p60`. Because NaNs never satisfy the comparisons, bins with no tongue sample remain in class `0`; there is no separate hidden/not-visible category.

ii.
```python
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_p40", "p40_to_p60", "gt_p60"],
],
```

iii. The notes justify the percentile boundaries from the task instructions, but do not give a separate justification for collapsing empty/not-visible bins into the lowest class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data on the same go-cue-centered bin grid as the neural data by converting relative bin edges to absolute times for each trial and then using `searchsorted` on camera timestamps.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
```

iii. The notes say the reference marker-processing path also aligned behavior to go cue, so the agent reused the same go-relative bin grid across modalities.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly: it skips sessions with zero recorded trials or zero good units, raises an error if a kept good unit has an empty `anno_name`, fills missing tone-onset events with a fixed `-1.85 s` offset, zeros invalid unit-trials, drops trials with all-zero neural activity, and leaves missing tongue bins as NaN until discretization, where they effectively become class `0`.

ii.
```python
if n_recorded_trials == 0:
    print(f"Skipping {session_id}: zero recorded trials in units/is_good_trials")
    return None
...
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None
...
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
if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
...
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
```

iii. The notes justify most of these as practical safeguards found during debugging: missing sample events get a task-timing fallback, bad or unsupported neural windows get masked or removed, and session-level structural problems cause the session to be skipped.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s notes indicate that the slowest work is reading large raw HDF5 arrays, especially `spike_times` and tongue tracking, then running the per-unit `searchsorted` neural binning loop. Optional plotting and pickling also add overhead.

ii.
```python
tongue_values = tongue_group["data"][:].astype(np.float64)
...
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
```

```python
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
```

iii. Step 6 and Step 7 say the agent moved to direct `h5py` reads and vectorized `searchsorted` specifically because these I/O-heavy and spike-binning steps dominated runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several remaining Python loops are still vectorizable: the per-trial sample-onset loop, the per-trial input-construction loop, the per-trial choice-inference loop, the per-trial output assembly loop, and the per-unit neural loop.

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

```python
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    ...
```

iii. The notes acknowledge runtime pressure and emphasize the vectorizations already added, but the final code still leaves these loops in Python for clarity or because they were not the biggest bottlenecks.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work. It computes stimulation windows once when building `input_trials` and again when making the optional plot payload. It also builds full input/output/trial arrays before applying `nonzero_trial_mask`, then re-slices many of those arrays and lists after all-zero trials are detected.

ii.
```python
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
    inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

```python
if str(photostim_onset[trial_plot_index]) != "N/A":
    stim_rel_on = trial_start[trial_plot_index] + float(photostim_onset[trial_plot_index]) - go
    stim_window = (stim_rel_on, stim_rel_on + float(photostim_duration[trial_plot_index]))
```

```python
if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
    sample_onset_rel_go = sample_onset_rel_go[nonzero_trial_mask]
    ...
    input_trials = [trial for keep, trial in zip(nonzero_trial_mask, input_trials) if keep]
    output_trials = [trial for keep, trial in zip(nonzero_trial_mask, output_trials) if keep]
```

iii. There is no strong explicit justification in the notes beyond debugging and visualization convenience. The notes do emphasize producing processing plots and raw-data sanity checks, which explains some of the repeated calculations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does some processing that is not needed for the final saved dataset: it loads `delay_start_times` and `tongue_likelihood`, computes `tongue_valid`, builds optional plotting payloads, keeps detailed `stats` for diagnostics, and carries several plotting-only slices and event arrays that are not part of the final pickle.

ii.
```python
delay_start_times = h5["acquisition"]["BehavioralEvents"]["delay_start_times"]["timestamps"][:].astype(np.float64)
...
tongue_likelihood = tongue_values[:, 2]
...
tongue_y_binned, tongue_valid = bin_tongue_y(...)
```

```python
plot_payload = None
if show_processing:
    ...
    plot_payload = {
        ...
        "delay_events_rel_go": delay_events,
        ...
        "tongue_likelihood_window": tongue_likelihood[tongue_window],
    }
```

```python
stats = {
    "choice_source_counts": choice_source_counts,
    "sample_onset_fallbacks": sample_onset_fallbacks,
    "fraction_invalid_unit_trials": float(np.mean(~is_good_trials)),
    ...
}
```

iii. The notes justify these extras as part of the required sanity-check and visualization workflow, not as necessary ingredients of the final decoder dataset itself.
