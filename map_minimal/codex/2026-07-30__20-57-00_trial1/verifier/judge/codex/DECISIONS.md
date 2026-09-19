# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data` by subject directory, collects all `.nwb` files in sorted order, and opens each session file directly with `h5py`. Within each file it reads the trial table from `intervals/trials`, event streams from `acquisition/BehavioralEvents`, tongue tracking from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, and unit data from `units/*`.

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
def convert_session(path: str) -> dict[str, Any] | None:
    with h5py.File(path, "r") as f:
        trial_group = f["intervals/trials"]
        behavioral_events = f["acquisition/BehavioralEvents"]
        tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
```

iii. In the trajectory, the AI said it first inspected the NWB schema so it could “mirror the original loading and filtering rather than inventing a format,” then confirmed the dataset as a collection of NWB session files and proceeded with deterministic sorted loading.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the directory name containing each NWB file, for example `sub-440956`. The final `subjects` list is the sorted set of those directory names, and `subject_idx` maps each kept session to that list.

ii.
```python
def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

```python
subjects = sorted({session["subject"] for session in session_results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session["subject"]])
```

iii. The trajectory does not show a separate argument for this choice beyond relying on the DANDI directory layout. The AI treated the folder structure as the subject partition.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. The session identifier is the filename without the `.nwb` extension, and session order follows the sorted file list.

ii.
```python
def get_session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
```

```python
for idx, path in enumerate(all_paths, start=1):
    session_id = get_session_id_from_path(path)
    result = convert_session(path)
```

iii. The AI’s notes state that session inclusion starts from all 174 NWB files, then excludes only one zero-good-unit file. That implies the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. Trials are not taken as the full behavioral table directly. Instead, the AI uses the first good unit’s `obs_intervals` as the set of recorded trials, matches those `[start, stop]` intervals back to rows in the behavioral trial table, and then asserts that each retained trial contains exactly one go-cue event.

ii.
```python
first_unit_obs = obs_intervals[obs_start:obs_stop]
trial_lookup = {
    (round(float(start), 4), round(float(stop), 4)): idx
    for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
}
trial_indices = []
for start, stop in first_unit_obs:
    key = (round(float(start), 4), round(float(stop), 4))
    if key not in trial_lookup:
        raise ValueError(...)
    trial_indices.append(trial_lookup[key])
```

```python
trial_start = behavior_trial_start[trial_indices]
trial_stop = behavior_trial_stop[trial_indices]
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
```

iii. In the trajectory, the AI first tried using the behavioral table more directly, then found later “behavior-only trials” with all-zero neural data. It justified the switch to `obs_intervals`-matched trials as the “correct reference behavior” because those are the trials actually covered by ephys.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters sessions to those with at least one `classification == "good"` unit, and filters trials to those covered by the first good unit’s `obs_intervals` / `units/is_good_trials` count. It deliberately keeps all such recorded trials, including photostimulation, early-lick, ignore, and `free_water` trials.

ii.
```python
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

```python
n_ephys_trials = int(f["units/is_good_trials"].shape[1])
...
first_unit_obs = obs_intervals[obs_start:obs_stop]
...
trial_start = behavior_trial_start[trial_indices]
trial_stop = behavior_trial_stop[trial_indices]
```

```python
"inclusion_rules": [
    "Include NWB sessions with at least one unit whose classification is 'good'.",
    "Keep all trials after session inclusion so outcome=ignore, early-lick, and photostim conditions remain available for decoding.",
]
```

iii. The trajectory explicitly says the AI chose to retain all ephys-covered trials because the decoder outputs require `ignore`, early-lick, and photostim conditions. It separately justified dropping only the single zero-good-unit session because that brought the session count down to the paper-level 173.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `units/spike_times` and `units/spike_times_index`, restricted to units with `classification == "good"`. Trial-aligned binning is anchored by per-trial `go_start_times`.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
```

```python
spike_counts = bin_spike_counts_for_good_units(
    spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
    spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
    good_unit_mask=good_unit_mask,
    trial_start=trial_start,
    trial_stop=trial_stop,
    go_times=go_times,
)
```

iii. The AI’s notes summarize this as “spike counts per 50 ms bin, converted to firing rates,” and repeatedly tie neural construction to the classifier-based good-unit QC.

## 2-b. How is the `neural` data processed?

i. The AI bins spike times into 50 ms counts by unit and trial, using trial start/stop to assign spikes to trials, converting those spikes to go-cue-relative times, and then using `np.add.at` into fixed bins. The counts are then converted to firing rates by dividing by `0.05`, and each trial matrix is stored as `float16`.

ii.
```python
for unit_idx, unit_end in enumerate(spike_times_index):
    ...
    trial_idx = np.searchsorted(trial_start, spikes, side="right") - 1
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

iii. In the trajectory and notes, the AI framed this as matching the requested go-cue alignment with non-overlapping 50 ms bins and converting spike counts to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units labeled `classification == "good"` are used. If a session has zero such units, the entire session is dropped.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

iii. The AI explicitly justified this with the published classifier-based QC workflow and noted that excluding the one zero-good-unit session produced the expected 173-session dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go-cue onset. The AI finds one `go_start_times` timestamp per retained trial, converts spike times to time relative to that go cue, and bins only the `[-2.5, 1.5)` second window.

ii.
```python
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
...
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
```

iii. The trajectory repeatedly says the conversion should use “go-cue alignment and 50 ms non-overlapping bins,” and the AI described the raw NWB times as being on a common session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins from `-2.5 s` to `+1.5 s` around go cue, yielding 80 time bins. There is no second-stage temporal rebinning beyond that fixed binning.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
```

iii. This follows the user instruction directly, and the AI’s notes restate the same window and bin size as one of the core processing choices.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` and `go_start_times`, after matching one sample/tone onset to each retained trial.

ii.
```python
sample_events = np.asarray(behavioral_events["sample_start_times"]["timestamps"][:], dtype=np.float64)
...
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
```

iii. The AI justified this in its notes as using “the final sample/tone onset before the aligned go cue.”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI picks the last sample/tone onset before the go cue, computes its offset from go cue, and adds that offset to the shared bin centers. The result is a continuous time-varying signal in seconds.

ii.
```python
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. The trajectory and notes describe this as expressing the 50 ms go-cue-centered bins relative to the tone onset used by that trial.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the exact same 80 go-cue-centered bin grid as the neural data. Only the values change from “time from go cue” to “time from tone onset.”

ii.
```python
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
...
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

```python
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))
```

iii. The AI explicitly aimed to put all streams on “one shared 50 ms grid so neural, photostim, and tongue labels stay exactly aligned per trial.”

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from trial-table `photostim_onset` and `photostim_duration`, together with behavioral-trial `start_time` and the per-trial go cue.

ii.
```python
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
```

```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
```

iii. The AI said in the trajectory that the NWB files expose the same photostim fields used by the reference pipeline and that those trials should be retained because photostim is a required decoder input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI decodes `photostim_onset` / `photostim_duration`, converts them to go-cue-relative onset and offset times, and marks a bin as `1` whenever the stimulation interval overlaps any part of that 50 ms bin. Non-stim trials stay all zeros.

ii.
```python
def as_float_or_none(value: Any) -> float | None:
    value = decode_scalar(value)
    if value in ("N/A", "", None):
        return None
    return float(value)
```

```python
onset = as_float_or_none(onset_values[trial_idx])
duration = as_float_or_none(duration_values[trial_idx])
...
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
photostim[trial_idx] = overlap.astype(np.float32)[0]
```

iii. The trajectory does not give a long separate defense of “overlap” versus “center-in-bin”; the justification is mainly pragmatic: turn photostim into a binary time-varying signal on the shared 50 ms grid.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim is aligned by converting onset/offset into time relative to the trial’s go cue, then evaluating those intervals on the same 50 ms bins used for neural data.

ii.
```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
```

iii. The AI’s stated goal was one shared go-cue-centered binning convention across neural, inputs, and time-varying outputs.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI does not derive choice from `outcome`. Instead, it derives it from `left_lick_times`, `right_lick_times`, per-trial `start_time` / `stop_time`, per-trial `go_start_times`, and `trial_instruction` as a fallback.

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
left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
right_all = right_lick_times[(right_lick_times >= start) & (right_lick_times <= stop)]
left_post = left_all[left_all >= go_time]
right_post = right_all[right_all >= go_time]
```

iii. The AI justified this explicitly in its notes: “first post-go lick side when available; otherwise first lick anywhere in the trial; otherwise instructed side if the trial has no licks at all.”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is assigned as left/right only. The AI uses the first post-go lick if possible, otherwise the first lick anywhere in the trial, and if there is no lick at all it falls back to the instructed side. It then repeats that per-trial code across all 80 bins. There is no separate “no lick” class in `OUTPUT_VALUES`.

ii.
```python
if left_post.size or right_post.size:
    ...
    choice[trial_idx] = 0 if left_first < right_first else 1
    source_counter["post_go_lick"] += 1
    continue

if left_all.size or right_all.size:
    ...
    choice[trial_idx] = 0 if left_first < right_first else 1
    source_counter["any_trial_lick"] += 1
    continue

choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
source_counter["instruction_fallback"] += 1
```

```python
OUTPUT_VALUES = [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
]
...
np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8)
```

iii. The trajectory shows the AI explicitly preferred retaining no-response trials and still assigning them a left/right label, rather than emitting a third no-lick category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the behavioral trial-table `outcome` column.

ii.
```python
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
```

iii. No additional justification appears in the trajectory beyond using the NWB trial fields directly when they already match the requested label.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, `hit` to `0`, `1`, `2` and repeats the resulting per-trial label across all bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
```

```python
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8)
```

iii. This is a direct encoding choice; the trajectory does not show any alternative being considered.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the behavioral trial-table `early_lick` column.

ii.
```python
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
```

iii. The AI’s broader justification was to keep early-lick trials in the dataset because `early_lick` is an explicit decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to `0` and `early` to `1`, then repeats that label across all 80 bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
```

```python
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8)
```

iii. The trajectory ties this to the decision to retain early-lick trials rather than apply the paper’s “regular trial” mask.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI uses the side-camera tongue tracking series `Camera0_side_TongueTracking`, specifically its `timestamps` and the second column of `data` (`tongue_y`). It does not use the likelihood column when constructing the output classes.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
```

iii. The notes describe this as “side-camera tongue y position,” and the only explicit methodological justification in the trajectory is to match the repo’s marker-alignment style rather than use a visibility threshold.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes session-specific 40th and 60th percentiles directly from the raw session-wide `tongue_y` trace. For each trial/bin, it picks the last camera sample before the bin end; if a bin has no sample strictly inside it, it falls back to the most recent sample before the bin end. It then discretizes that scalar sample into three classes and never averages frames within a bin.

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

iii. The AI explicitly justified this as matching the reference repo’s marker-alignment convention: “use the last sample available inside each bin” and, when empty, fall back to the latest prior sample to avoid NaNs.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses only three categories: `0` for `< q40`, `1` for `q40` to `q60`, and `2` for `> q60`. It does not create the required fourth “not visible” class.

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

iii. The notes frame the discretization as a three-way thresholding of raw `tongue_y`, not as a four-class visible/not-visible representation.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue output is aligned to the same go-cue-centered 50 ms bins as the neural data. The AI forms bin edges at `go_times + BIN_EDGES_S`, uses `searchsorted` on camera timestamps to locate each bin, and takes one representative frame per bin.

ii.
```python
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
```

iii. The trajectory states that the AI wanted a “shared 50 ms grid” and deliberately reconciled the repo’s video-marker alignment style with the neural bin grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing photostim values by converting `"N/A"` / empty strings to `None` and leaving those trials as all-zero photostim. It drops sessions with zero good units, and raises errors if expected NWB groups or trial matches are missing. It does not mark missing tongue visibility as a separate class; instead it carries forward the most recent frame before a bin end.

ii.
```python
def as_float_or_none(value: Any) -> float | None:
    value = decode_scalar(value)
    if value in ("N/A", "", None):
        return None
    return float(value)
```

```python
if "BehavioralTimeSeries" not in f["acquisition"]:
    raise ValueError(f"Missing BehavioralTimeSeries in {path}")
...
if key not in trial_lookup:
    raise ValueError(...)
```

```python
fallback_missing = int(np.sum(right_idx <= left_idx))
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
```

iii. The trajectory shows the AI prioritizing hard failures for structural mismatches, excluding behavior-only trials with no neural recording, and avoiding NaNs in tongue bins by previous-frame fallback.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive steps are session-by-session NWB I/O, reading the large spike-time and tongue-tracking arrays, the per-unit spike-binning loop, the per-trial output assembly, and serializing the large pickle. The trajectory also shows the full conversion spending most of its wall time in the multi-gigabyte full build.

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

```python
write_pickle(args.full_out, full_data)
```

iii. In the trajectory, the AI repeatedly commented that the full run was dominated by large late sessions and a “multi-gigabyte build,” and waited on the long conversion rather than changing the algorithm.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are left scalar or per-trial: the `compute_choice_labels` trial loop, the `bin_photostim_series` trial loop, the trial-by-trial assembly of `neural` / `input` / `output`, and the region-to-index mapping loop. The per-unit spike loop is partly unavoidable because spikes are ragged by unit, but it is also the largest remaining non-vectorized block.

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
for unit_idx, unit_end in enumerate(spike_times_index):
    ...
```

iii. The trajectory does not explicitly discuss vectorization opportunities beyond debugging correctness. The code itself shows where the remaining Python loops are.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans the full left/right lick arrays once per trial in `compute_choice_labels`, repeatedly constructs constant-length per-trial arrays with `np.full` for per-trial outputs, and repeatedly stacks per-trial `input` / `output` matrices in Python loops after already computing session-level arrays.

ii.
```python
left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
right_all = right_lick_times[(right_lick_times >= start) & (right_lick_times <= stop)]
```

```python
for trial_idx in range(n_trials):
    neural_trials.append(...)
    input_trials.append(np.vstack([...]).astype(np.float32))
    output_trials.append(
        np.vstack(
            [
                np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8),
                np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
                np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8),
                tongue_disc[trial_idx],
            ]
        )
    )
```

iii. No explicit justification for these repeated operations appears in the trajectory; they appear to be straightforward implementation choices rather than deliberate optimizations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores substantial bookkeeping that is not needed by the decoder itself: per-session stats, multiple counters, fallback-source counts for choice labels, tongue fallback counts, sample-dataset export, and summary metadata. These are useful for debugging but not used by downstream decoding on `converted_data.pkl`.

ii.
```python
choice, choice_sources = compute_choice_labels(...)
...
outcome_counter = Counter(int(x) for x in outcome.tolist())
early_counter = Counter(int(x) for x in early.tolist())
choice_counter = Counter(int(x) for x in choice.tolist())
```

```python
"stats": {
    "n_trials": int(n_trials),
    "n_good_units": int(len(brain_region_labels)),
    "n_behavior_trials": int(n_behavior_trials),
    "n_ephys_trials": int(n_ephys_trials),
    "stim_trial_count": int(stim_trial_count),
    ...
}
```

```python
sample_count = min(args.sample_sessions, len(session_results))
sample_results = session_results[:sample_count]
sample_data, sample_summary = build_dataset(sample_results)
write_pickle(args.sample_out, sample_data)
```

iii. The trajectory shows these additions being used for validation and reporting rather than for the final decoder input/output tensors themselves.
