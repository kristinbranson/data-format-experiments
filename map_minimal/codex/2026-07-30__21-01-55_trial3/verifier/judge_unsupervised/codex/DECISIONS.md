# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter walks the NWB tree with `sorted(data_dir.glob("sub-*/*.nwb"))`, treats each `.nwb` file as one session, and reads the needed tables/streams directly with `h5py`. Inside each session it loads unit metadata, the trial table, behavioral event timestamps, and side-camera tongue tracking.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
with h5py.File(path, "r") as f:
    units = f["units"]
    trials = f["intervals/trials"]
    go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
    sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
    left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
    right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
    video_timestamps = np.asarray(
        f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][()],
        dtype=np.float64,
    )
```

iii. `CONVERSION_NOTES.md` says the goal was a session-by-session NWB conversion that “included every NWB session with at least one unit whose `units/classification == "good"`.” The trajectory also shows the agent deliberately mirrored the NWB schema rather than inventing an alternate loader.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the parent directory name (`sub-<id>`). The saved dataset then stores the unique sorted subject list plus a per-session integer index into that list.

ii.
```python
stats = {
    "session_id": path.stem,
    "subject": path.parent.name.replace("sub-", ""),
    ...
}
```

```python
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_to_idx[session["subject"]] for session in sessions], dtype=np.int16),
```

iii. `CONVERSION_NOTES.md` states that “subjects come from the NWB subject directory names.” No different subject-mapping logic appeared in the trajectory.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The file stem becomes `session_id`, and the session is skipped only if it yields no usable session-level output.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
    session_stats.append(stats)
    if session is not None:
        sessions.append(session)
```

```python
session = {
    "session_id": path.stem,
    "subject": stats["subject"],
    ...
}
```

iii. `CONVERSION_NOTES.md` says this yielded 173 included sessions and 1 skipped session, with the skipped file having zero good units. The trajectory shows the agent explicitly checking that this matched the paper’s 173-session count.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`, but all trial-level arrays are first truncated to `units["is_good_trials"].shape[1]` so behavioral rows do not exceed ephys-covered rows. The kept trial indices are then iterated one by one.

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])

trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_stop = np.asarray(trials["stop_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_instruction = decode_bytes_array(trials["trial_instruction"][()])[:ephys_trial_count]
...
keep_idx = np.flatnonzero(keep_mask)
```

```python
for trial_idx in keep_idx:
    go_abs = float(go_times[trial_idx])
    abs_edges = go_abs + BIN_EDGES
    ...
    valid_trials.append(trial_idx)
```

iii. `CONVERSION_NOTES.md` calls this truncation a “critical correction” because some NWBs contain more trial-table rows than are covered by the ephys trial matrix; otherwise late behavioral trials became all-zero neural trials.

## 1-e. How are trials filtered based on quality controls?

i. The converter excludes only `auto_water` and `free_water` trials from the behavioral trial table. It deliberately keeps early-lick, ignore/no-response, and photostimulation trials because those are required decoder targets/inputs. It also mechanically drops trials whose full aligned window falls outside the side-camera timestamps and trials that end up with all-zero neural activity after binning.

ii.
```python
keep_mask = (auto_water == 0) & (free_water == 0)
stats["n_trials_dropped_auto_free"] = int(np.sum(~keep_mask))
keep_idx = np.flatnonzero(keep_mask)
```

```python
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
```

```python
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if not np.all(nonzero_mask):
    stats["n_trials_dropped_all_zero_neural"] = int(np.sum(~nonzero_mask))
    ...
```

iii. `CONVERSION_NOTES.md` and trajectory step 18 both say the reference “regular trial” mask excluded early lick, no-response, and stimulation trials, but the agent treated those as task-driven exceptions because `early_lick`, `outcome`, and `photostim_on` are required decoder variables.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` and `units/spike_times_index`, restricted to units where `units/classification == "good"`. Brain-region labels come from `units/anno_name`.

ii.
```python
classification = decode_bytes_array(units["classification"][()])
anno_name = decode_bytes_array(units["anno_name"][()])
good_unit_idx = np.flatnonzero(classification == "good")
```

```python
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
...
spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
```

iii. `CONVERSION_NOTES.md` says the agent used the NWB-provided QC label `classification == "good"` because the papers describe downstream analyses as using units labeled “good” by the QC classifier.

## 2-b. How is the `neural` data processed?

i. For each kept trial, the code constructs absolute bin edges for a fixed `[-2.5, 1.5)` s window around the go cue. For each retained unit it counts spikes in each 50 ms bin using `np.searchsorted`, divides by `0.05`, and stores firing rates in spikes/s.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_WIDTH = 0.05
...
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

```python
flat_abs_edges = abs_edge_matrix.reshape(-1)
...
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
counts = np.diff(edge_idx, axis=1)
rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
```

iii. `CONVERSION_NOTES.md` explicitly says the converter uses “exact-width bins over `[start, end)`” and stores neural activity as `spike_count_in_bin / 0.05`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is applied at the unit level with `classification == "good"`. Sessions with zero such units are dropped. After binning, trials with all-zero neural matrices are also dropped. The converter does not apply any additional firing-rate/waveform thresholds and does not use the per-unit/per-trial `units/is_good_trials` mask except to determine `ephys_trial_count`.

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
stats["n_units_good"] = int(good_unit_idx.size)
if good_unit_idx.size == 0:
    return None, stats
ephys_trial_count = int(units["is_good_trials"].shape[1])
```

```python
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if not np.all(nonzero_mask):
    stats["n_trials_dropped_all_zero_neural"] = int(np.sum(~nonzero_mask))
    ...
```

iii. `CONVERSION_NOTES.md` says “No extra firing-rate or waveform filtering was added on top of the NWB `good` label.” The trajectory shows the agent believed `classification == "good"` was the paper’s main QC output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to go-cue onset. The code takes the absolute go-cue timestamp for that trial and adds the fixed relative bin edges to get absolute neural-bin edges.

ii.
```python
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
...
go_abs = float(go_times[trial_idx])
abs_edges = go_abs + BIN_EDGES
...
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
```

iii. This follows both the task instructions and `CONVERSION_NOTES.md`, which state “Alignment event: go cue onset.” The recovered reference preprocessing code also described spikes as already aligned to go cue in the raw processed data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use non-overlapping 50 ms bins, giving 80 bins over the 4 s window. No later rebinning is applied.

ii.
```python
BIN_WIDTH = 0.05
...
n_bins = int(round((end_time - start_time) / width))
...
"time_bin_size": 50.0,
"time_bin_size_sec": BIN_WIDTH,
"binning_note": "Exact 50-ms bins spanning [-2.5, 1.5) s relative to the go cue.",
```

iii. `CONVERSION_NOTES.md` states “Bin width: `0.05` s” and “Number of bins per trial: `80`.”

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times/timestamps`, together with `trial_start` and `go_start_times/timestamps` to identify the relevant sample/tone onset for each trial.

ii.
```python
sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
...
tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, float(trial_start[trial_idx]), go_abs)
```

```python
def last_sample_before_go(sample_starts: np.ndarray, trial_start: float, go_time: float):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
```

iii. `CONVERSION_NOTES.md` says the tone onset was taken as the last `sample_start_times` timestamp between trial start and go cue, because replay can create multiple sample starts within a trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code finds the last sample-start before go within the trial, converts that absolute tone time to a go-relative offset, and then computes `bin_center_relative_to_go - tone_onset_relative_to_go` for every neural bin. If no sample start is found it falls back to `go_time - 1.85`.

ii.
```python
if hi > lo:
    return float(sample_starts[hi - 1]), False
return float(go_time - 1.85), True
```

```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. `CONVERSION_NOTES.md` says this choice was made to handle replayed sample epochs after early licks. The notes also report that the fallback path was never actually used (`Tone onset fallbacks: 0`).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed directly on the same 80 go-relative bin centers used for neural activity, so each neural bin gets a same-time-bin scalar value.

ii.
```python
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
...
input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. `CONVERSION_NOTES.md` says “For each neural bin, the value is `bin_center_relative_to_go - tone_onset_relative_to_go`. That makes zero correspond to tone onset.”

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from trial-table fields `photostim_onset`, `photostim_duration`, and `photostim_power`, plus `trial_start` and `go_start_times` to convert from trial-start time to go-relative time.

ii.
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

```python
go_rel = go_abs - float(trial_start[trial_idx])
on_rel = float(stim_onset[trial_idx] - go_rel)
off_rel = on_rel + float(stim_duration[trial_idx])
```

iii. `CONVERSION_NOTES.md` says the converter follows the reference code pattern where stimulation times are stored relative to trial start and then shifted into go-cue coordinates.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. String-valued trial-table entries like `N/A` are parsed into `NaN`. A trial is considered stimulated only if power, onset, and duration are all finite and `stim_power > 0`. The stimulation interval is then converted to go-relative time and binarized on the neural time grid.

ii.
```python
if value in {"N/A", "nan", "NaN", ""}:
    out[i] = np.nan
else:
    out[i] = float(value)
```

```python
if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
    go_rel = go_abs - float(trial_start[trial_idx])
    on_rel = float(stim_onset[trial_idx] - go_rel)
    off_rel = on_rel + float(stim_duration[trial_idx])
else:
    on_rel = np.nan
    off_rel = np.nan
```

```python
if np.isfinite(stim_on_rel[row_idx]):
    stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
else:
    stim_on = np.zeros(BIN_CENTERS.shape[0], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says “Bins are marked 1 when the bin center lies within the stimulation interval.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation trace is generated on the same go-relative `BIN_CENTERS` as the neural activity and is stored as the second row of each per-trial `input` matrix.

ii.
```python
stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
...
input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. The notes explicitly describe this as a binary, time-varying input “aligned with the neural bins.”

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the global `left_lick_times/timestamps` and `right_lick_times/timestamps` event streams, with `trial_stop` used to limit the search to the current trial. If no post-go lick is found, the converter falls back to `trial_instruction`.

ii.
```python
left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
```

```python
choice_code, choice_used_fallback = first_post_go_choice(
    left_licks,
    right_licks,
    go_abs,
    float(trial_stop[trial_idx]),
    trial_instruction[trial_idx],
)
```

```python
return (0 if instruction == "left" else 1), True
```

iii. The trajectory says the agent considered choice ambiguous because the NWB trial table did not expose an explicit per-trial reported side, so it decided to recover choice from lick events “rather than silently substituting the instructed side.” `CONVERSION_NOTES.md` then documents the instructed-side fallback for ignore trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code finds the first left lick and first right lick at or after go cue and before trial end; whichever occurs earlier sets the choice (`left = 0`, `right = 1`). If neither exists, it uses the instructed side. The resulting scalar is then repeated across all 80 time bins.

ii.
```python
left_idx = np.searchsorted(left_licks, go_time, side="left")
right_idx = np.searchsorted(right_licks, go_time, side="left")
...
if np.isfinite(left_time) and np.isfinite(right_time):
    return (0, False) if left_time <= right_time else (1, False)
if np.isfinite(left_time):
    return 0, False
if np.isfinite(right_time):
    return 1, False
return (0 if instruction == "left" else 1), True
```

```python
np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` says “Choice is the first post-go lick side within the trial,” and that the instructed-side fallback was used specifically to satisfy the required binary label format on ignore trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table field `intervals/trials/outcome`.

ii.
```python
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
```

```python
if outcome[trial_idx] == "ignore":
    outcome_code = 0
elif outcome[trial_idx] == "miss":
    outcome_code = 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
```

iii. `CONVERSION_NOTES.md` documents this as a direct mapping from the raw NWB outcome labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string-valued trial outcome is mapped to `ignore = 0`, `miss = 1`, `hit = 2`, then repeated across all neural time bins as a per-trial constant categorical output.

ii.
```python
if outcome[trial_idx] == "ignore":
    outcome_code = 0
elif outcome[trial_idx] == "miss":
    outcome_code = 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
...
outcome_codes.append(outcome_code)
```

```python
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8)
```

iii. The notes list the exact mapping and describe `outcome` as a per-trial label expanded across bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table field `intervals/trials/early_lick`.

ii.
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
```

```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
```

iii. `CONVERSION_NOTES.md` says the output is derived directly from the NWB early-lick label and kept because `early_lick` is itself a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code binarizes the trial-table label as `no = 0`, `yes = 1` via the raw strings `"no early"` and `"early"`, then repeats the resulting scalar across all 80 bins.

ii.
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
early_codes.append(early_code)
```

```python
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8)
```

iii. The notes document the mapping as `no = 0`, `yes = 1`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from the side-camera tongue tracking stream: `Camera0_side_TongueTracking/data` for values and `Camera0_side_TongueTracking/timestamps` for timing. The converter uses column 1 of the data array as `tongue_y`.

ii.
```python
video_timestamps = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][()],
    dtype=np.float64,
)
tongue_data = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][()],
    dtype=np.float64,
)
tongue_y = tongue_data[:, 1]
```

iii. `CONVERSION_NOTES.md` explicitly says “Source: side-camera tongue tracking `Camera0_side_TongueTracking`” and “Used the `tongue_y` coordinate only.”

## 8-b. How is `output` *Tongue y-position* processed?

i. The converter computes session-level 40th and 60th percentiles over all raw `tongue_y` samples. For each neural bin of each trial it uses the last video frame inside that bin; if no frame falls inside the bin, it uses the most recent available frame at or before the bin end.

ii.
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
```

```python
starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
...
if end_idx > start_idx:
    values[i] = tongue_y[end_idx - 1]
else:
    fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
    values[i] = tongue_y[fallback_idx]
```

iii. `CONVERSION_NOTES.md` gives this exact justification: take the last frame in each neural bin, otherwise the most recent frame before the bin end, then discretize using session-wide percentiles.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code uses per-session thresholds. Category `0` is below the 40th percentile, category `1` is between the 40th and 60th percentiles inclusive, and category `2` is above the 60th percentile.

ii.
```python
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
mid = (values >= p40) & (values <= p60)
cats[mid] = 1
```

iii. This matches both the task instructions and the mapping written in `CONVERSION_NOTES.md`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Neural trial windows are turned into absolute time-bin edges with `go_abs + BIN_EDGES`. Those absolute edges are then used to sample the tongue trace into the same 80 per-trial bins.

ii.
```python
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
...
tongue_cat = trial_tongue_categories(
    video_timestamps=video_timestamps,
    tongue_y=tongue_y,
    abs_edges=abs_edge_matrix[row_idx],
    p40=tongue_p40,
    p60=tongue_p60,
)
```

iii. `CONVERSION_NOTES.md` describes the tongue output as time-varying and sampled into “each neural bin.”

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter uses several ad hoc repairs and guards: it parses `N/A`/`nan` strings into `NaN`; truncates trial-table arrays to `ephys_trial_count`; falls back to `go_time - 1.85` if no sample-start is found; falls back to instructed side if no post-go lick is found; reuses the most recent tongue frame if a bin has no new frame; and drops trials whose full window is outside the video range or whose neural matrix is all zeros.

ii.
```python
if value in {"N/A", "nan", "NaN", ""}:
    out[i] = np.nan
```

```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
...
return float(go_time - 1.85), True
...
return (0 if instruction == "left" else 1), True
```

```python
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
...
if not np.all(nonzero_mask):
    stats["n_trials_dropped_all_zero_neural"] = int(np.sum(~nonzero_mask))
```

iii. The trajectory says the agent wanted every fallback to be “auditable in the notes.” `CONVERSION_NOTES.md` then documents the major repairs, especially the `ephys_trial_count` truncation and the choice fallback on ignore trials.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant costs are session-wide NWB reads, the per-unit spike binning loop, and repeated per-trial tongue binning. The trajectory also shows the full conversion/decoder run was CPU-bound and memory-heavy.

ii.
```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

```python
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    if end_idx > start_idx:
        values[i] = tongue_y[end_idx - 1]
    else:
        fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
        values[i] = tongue_y[fallback_idx]
```

iii. There is no explicit efficiency analysis in `CONVERSION_NOTES.md`, but trajectory step 222 says the full run was “CPU-bound and memory-heavy,” which is consistent with these loops.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the inner `for trial_row` assignment in neural construction, the per-bin loop in `trial_tongue_categories`, and the Python-level string parsing loops in `decode_bytes_array` and `parse_optional_float_array`.

ii.
```python
for value in arr:
    ...
```

```python
for i, value in enumerate(arr):
    ...
```

```python
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    ...
```

```python
for trial_row in range(n_trials):
    neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. No explicit justification for these inefficiencies appears in the notes or trajectory; this is an inference from the implementation.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly computes per-trial input/output arrays in Python loops, repeatedly constructs constant 80-bin label vectors with `np.full`, and repeatedly performs `searchsorted`-based alignment separately for tone, choice, tongue, and spikes.

ii.
```python
for row_idx, trial_idx in enumerate(valid_trials):
    ...
    input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
    ...
    output_trials.append(
        np.vstack(
            [
                np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8),
                np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8),
                np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8),
                tongue_cat,
            ]
        )
    )
```

```python
left_idx = np.searchsorted(left_licks, go_time, side="left")
right_idx = np.searchsorted(right_licks, go_time, side="left")
...
starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
...
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
```

iii. No separate efficiency rationale was documented; this repeated work is visible directly in the code.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `session_output_counts` is populated but never used. Several bookkeeping fields are also stored only for auditability, not for decoder consumption (`trial_indices_source`, `tongue_percentiles`, many `stats` counters). More broadly, scalar outputs like choice/outcome/early-lick are expanded to full 80-bin vectors even though they do not vary within the trial.

ii.
```python
session_output_counts = Counter()
...
session_output_counts[f"outcome_{outcome[trial_idx]}"] += 1
```

```python
session = {
    ...
    "trial_indices_source": valid_trials,
    "tongue_percentiles": (float(tongue_p40), float(tongue_p60)),
}
```

```python
np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8),
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8),
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8),
```

iii. The notes justify the extra counters and fallback accounting as audit/debug support, but they are not needed by the downstream decoder itself.
