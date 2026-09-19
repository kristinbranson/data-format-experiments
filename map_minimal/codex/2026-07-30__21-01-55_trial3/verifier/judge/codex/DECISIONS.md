# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the dataset by iterating over every `sub-*/*.nwb` file under the data directory, opening each file directly with `h5py`, and then reading HDF5 groups for `units`, `intervals/trials`, `acquisition/BehavioralEvents`, and `BehavioralTimeSeries`. It does not use `pynwb`.

ii. 
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
with h5py.File(path, "r") as f:
    units = f["units"]
    trials = f["intervals/trials"]
    go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)
```

iii. In the trajectory, the agent said it would "build `convert_data.py` around session-by-session NWB reads" (step 76) and that the "NWB files already carry the paper’s derived trial table and QC/unit metadata" (step 24), so it chose to read each NWB session file directly.

## 1-b. How are the data split into subjects?

i. Subjects are split by the parent directory name of each NWB file. The agent strips the `sub-` prefix from the folder name and later builds `subjects` and `subject_idx` from those strings.

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
"subject_idx": np.asarray([subject_to_idx[session["subject"]] for session in sessions], dtype=np.int16),
```

iii. The trajectory does not give a separate justification for using the folder name, but the post-run notes and step 24 show the agent trusted the NWB release structure and metadata rather than adding a separate subject-mapping layer.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session identifier is `path.stem`, and session order follows the sorted file list.

ii. 
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
"session_id": path.stem,
```

iii. In step 76, the agent explicitly planned a "session-by-session" converter. The trajectory shows no attempt to merge or subdivide files, so the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. The agent first truncates all trial-level arrays to `units["is_good_trials"].shape[1]`, treating that width as the ephys-covered trial count. It then iterates over the retained row indices from the trial table after filtering. Trials are therefore the first `ephys_trial_count` rows of `intervals/trials`, not the full NWB trial table.

ii. 
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])

trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_stop = np.asarray(trials["stop_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_instruction = decode_bytes_array(trials["trial_instruction"][()])[:ephys_trial_count]
...
go_times = np.asarray(
    f["acquisition/BehavioralEvents/go_start_times/timestamps"][()],
    dtype=np.float64,
)[:ephys_trial_count]
```

iii. The agent justified this directly in the trajectory: in step 105 it concluded that "`units/is_good_trials` matrix width is the actual ephys-covered trial count" and said it would slice behavior/video arrays to that exact count so "spikes and labels live on the same trial set."

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes `auto_water` and `free_water` trials, drops trials whose aligned `[-2.5, 1.5)` window falls outside the side-camera timestamps, and later drops trials whose binned neural data are all zero. It keeps early-lick, ignore, and photostimulation trials.

ii. 
```python
keep_mask = (auto_water == 0) & (free_water == 0)
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
    ...
```

iii. The agent discussed this several times: step 18 says the paper code excludes `early_lick`, `auto_water`, `free_water`, and `correctness == -1`, but the decoder task requires early/ignore/photostim trials; step 76 repeats the plan to exclude only `auto_water`/`free_water`; step 191 says it would keep early/no-response trials because they are decoder targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index`, using `go_start_times` to place the per-trial bin edges. Only units with `classification == "good"` are used.

ii. 
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
...
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
flat_abs_edges = abs_edge_matrix.reshape(-1)
```

iii. In step 24 the agent said the NWB files already contain QC/unit metadata, and in step 236 it summarized the converter as using NWB `classification == "good"` units with go-cue alignment.

## 2-b. How is the `neural` data processed?

i. For each good unit, the agent reads that unit’s ragged spike train, counts spikes in 50 ms bins using `np.searchsorted`, divides by bin width to get firing rates in spikes/s, and stores each trial’s matrix as `float16`.

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

iii. Step 76 says the agent intended to "build 50 ms neural bins" on the go-cue-aligned grid. Step 236 summarizes the result as "go-cue alignment, exact 50 ms bins over `[-2.5, 1.5)`."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by `classification == "good"`. If a session has zero such units, the session is dropped.

ii. 
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
stats["n_units_good"] = int(good_unit_idx.size)
if good_unit_idx.size == 0:
    return None, stats
```

iii. The trajectory explicitly supports this. Step 24 says the agent was checking whether "`classification == 'good'` ... is the correct neuron filter," step 28 says that field is "almost certainly the paper’s QC output," and step 236 says the finished converter uses NWB `classification == "good"` units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go cue onset by adding a fixed vector of relative bin edges to each trial’s `go_start_times` timestamp, then binning spikes against those absolute edges.

ii. 
```python
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
...
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
flat_abs_edges = abs_edge_matrix.reshape(-1)
```

iii. The trajectory repeatedly states this choice: step 76 says the agent would "match the reference alignment logic at go cue," step 86 says it switched to literal 50 ms bins over `[-2.5, 1.5)`, and step 236 confirms "go-cue alignment."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use non-overlapping 50 ms bins from `-2.5 s` to `+1.5 s` relative to go cue, giving 80 bins per trial. No additional rebinning or smoothing is applied.

ii. 
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_WIDTH = 0.05
...
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

iii. In step 86 the agent says it is "switching the converter to literal 50 ms bins over `[-2.5, 1.5)` so the window matches the task specification," and step 236 repeats that exact summary.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The agent derives this input from `sample_start_times`, `trial_start`, and `go_start_times`. For each kept trial, it finds the last sample-start timestamp between trial start and the go cue.

ii. 
```python
sample_starts = np.asarray(
    f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()],
    dtype=np.float64,
)
...
tone_time_abs, tone_used_fallback = last_sample_before_go(
    sample_starts,
    float(trial_start[trial_idx]),
    go_abs,
)
```

```python
def last_sample_before_go(sample_starts: np.ndarray, trial_start: float, go_time: float):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        return float(sample_starts[hi - 1]), False
    return float(go_time - 1.85), True
```

iii. Step 76 states that the agent would "derive tone onset from the last `sample_start` inside each trial." Step 77 adds that fallback decisions like a missing last-sample event would be counted and reported.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding a tone onset time per trial, the agent converts it to go-relative coordinates and subtracts that offset from every neural bin center. If no sample start is found in `[trial_start, go]`, it falls back to `go_time - 1.85`.

ii. 
```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

```python
if hi > lo:
    return float(sample_starts[hi - 1]), False
return float(go_time - 1.85), True
```

iii. In step 76 the agent said it would use the "last `sample_start` inside each trial," and in step 77 it said missing-last-sample fallback decisions would be made explicit and auditable.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The input is evaluated exactly at the same 80 go-cue-aligned bin centers used for the neural data, so both arrays share the same time axis.

ii. 
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
...
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. Step 76 says the agent would "build 50 ms neural bins and photostim/tongue labels on the same grid." The time-from-tone input is built from `BIN_CENTERS`, so it follows that same alignment plan.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation input is derived from `photostim_onset`, `photostim_duration`, `photostim_power`, `trial_start`, and `go_start_times`.

ii. 
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

```python
if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
    go_rel = go_abs - float(trial_start[trial_idx])
    on_rel = float(stim_onset[trial_idx] - go_rel)
    off_rel = on_rel + float(stim_duration[trial_idx])
```

iii. Step 18 notes that the decoder explicitly needs photostimulation as an input, and step 76 says the agent would "build ... photostim ... labels on the same grid."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The agent converts stimulation onset from trial-start coordinates into go-cue-relative coordinates, adds duration to get the offset, and marks bins whose centers fall inside `[stim_on, stim_off)`. Trials without finite onset/duration or with non-positive power are all-zero.

ii. 
```python
if np.isfinite(stim_on_rel[row_idx]):
    stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
else:
    stim_on = np.zeros(BIN_CENTERS.shape[0], dtype=np.float32)
```

iii. The explicit trajectory justification is limited, but step 76 says the agent planned to build photostimulation "on the same grid" as neural data, and step 18 says stimulation trials were retained because photostim is a required decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned to neural data by expressing the stimulation interval relative to the go cue and then comparing it to the same bin centers used for neural firing rates.

ii. 
```python
go_rel = go_abs - float(trial_start[trial_idx])
on_rel = float(stim_onset[trial_idx] - go_rel)
off_rel = on_rel + float(stim_duration[trial_idx])
...
stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
```

iii. Step 76 explicitly says photostim would be built "on the same grid" as the aligned neural bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives choice from the raw lick event streams `left_lick_times` and `right_lick_times`, plus `trial_instruction` for fallback when there is no post-go lick in the trial.

ii. 
```python
left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
...
choice_code, choice_used_fallback = first_post_go_choice(
    left_licks,
    right_licks,
    go_abs,
    float(trial_stop[trial_idx]),
    trial_instruction[trial_idx],
)
```

iii. The trajectory makes this explicit. In step 69, the agent says choice is ambiguous because the trial table lacks an explicit side, so it would use lick-event streams "so I can label choice from behavior itself rather than silently substituting the instructed side."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the agent finds the first left and right licks after go cue and before trial stop, picks the earlier side as the choice, and if neither exists, falls back to the instructed side. The result is encoded as a binary class (`0` left, `1` right) and repeated across all bins.

ii. 
```python
def first_post_go_choice(left_licks, right_licks, go_time, stop_time, instruction):
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
"output_values": [
    ["left", "right"],
    ...
]
```

```python
np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8)
```

iii. Step 69 gives the main rationale for reading actual lick events. Step 76 says the agent would "label choice from the first post-go lick with a defined fallback for ignore trials," and step 77 says those fallback decisions would be counted and documented.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the `outcome` column in the trials table.

ii. 
```python
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
```

iii. The trajectory does not discuss outcome separately; the agent simply uses the NWB trial-table field already carrying the behavioral label.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps `'ignore' -> 0`, `'miss' -> 1`, and `'hit' -> 2`, then repeats that code across all 80 bins for the trial.

ii. 
```python
if outcome[trial_idx] == "ignore":
    outcome_code = 0
elif outcome[trial_idx] == "miss":
    outcome_code = 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
```

```python
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8)
```

iii. The trajectory does not show a separate justification; this follows the task’s categorical outcome labels directly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the `early_lick` column of the trials table.

ii. 
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
```

iii. The trajectory discusses early-lick trials at the filtering level rather than label derivation. Steps 18, 76, and 191 all say early-lick trials were kept because `early_lick` is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string label is mapped to `1` for `"early"` and `0` otherwise, then repeated across all bins of the trial.

ii. 
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
...
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8)
```

iii. The trajectory justification is indirect: the agent kept early-lick trials specifically because early lick is one of the outputs (steps 18, 76, 191).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The agent derives tongue y-position from `Camera0_side_TongueTracking/timestamps` and the second column of `Camera0_side_TongueTracking/data` (`tongue_y`). It does not use the likelihood channel.

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

iii. Step 18 says the agent inspected the NWB schema to "map ... the tongue trace without guessing field names." Step 236 summarizes the result as "session-percentile `tongue_y`."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent computes the 40th and 60th percentiles over the session’s raw `tongue_y` values, then for each bin takes the last frame inside the bin; if the bin has no frame, it reuses the most recent available frame before the bin end. The resulting scalar is discretized into three categories.

ii. 
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
```

```python
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    if end_idx > start_idx:
        values[i] = tongue_y[end_idx - 1]
    else:
        fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
        values[i] = tongue_y[fallback_idx]
```

iii. The explicit trajectory justification is minimal. Step 76 says only that tongue labels would be built on the same grid as neural bins. The post-run notes clarify that the agent intentionally used "the last video frame within that bin" and a carry-forward fallback.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent thresholds tongue y into three classes only: `< 40th percentile`, `40th to 60th percentile`, and `> 60th percentile`. There is no separate "not visible" class.

ii. 
```python
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
mid = (values >= p40) & (values <= p60)
cats[mid] = 1
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
],
```

iii. Step 236 summarizes this only as "session-percentile `tongue_y`." The trajectory does not mention a visibility class; the code and notes show the agent chose not to represent invisibility explicitly.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue output is aligned by evaluating each trial on the same absolute go-cue-centered bin edges as the neural data. For each bin, the agent uses video frames whose timestamps fall between consecutive neural bin edges.

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

```python
starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
```

iii. Step 76 says the agent would build tongue labels "on the same grid" as neural bins. Step 91 adds that it removed an artificial trial-start/trial-stop gate and would instead "only check whether the continuous video stream actually covers the requested aligned window."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several missing-data cases with fallbacks or exclusions: text and optional-float fields are decoded with generic helpers; sessions with zero `good` units are dropped; trials outside video coverage are dropped; trials with all-zero neural data are dropped; missing tone events fall back to `go_time - 1.85`; missing post-go choices fall back to the instructed side; empty tongue bins fall back to the most recent frame.

ii. 
```python
def decode_bytes_array(arr) -> np.ndarray:
    ...

def parse_optional_float_array(arr) -> np.ndarray:
    ...
```

```python
if good_unit_idx.size == 0:
    return None, stats
...
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    ...
...
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

```python
return float(go_time - 1.85), True
...
return (0 if instruction == "left" else 1), True
...
values[i] = tongue_y[fallback_idx]
```

iii. Step 77 says the agent wanted fallback decisions like missing last-sample events or ignore-trial choices to be "auditable." Step 91 justifies dropping some trials by requiring that the video stream cover the aligned window. Step 105 justifies trial truncation as a guard against misaligned behavior and ephys.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive parts are the per-session NWB reads, the per-unit spike binning loop, and the tongue/video binning loops. The neural code repeatedly reads ragged spike trains, performs `searchsorted` against all aligned edges, and then copies each unit’s rates back into per-trial arrays.

ii. 
```python
with h5py.File(path, "r") as f:
    ...
```

```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    ...
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. The trajectory does not explicitly profile runtime inside the converter, but step 156 shows the agent regarded the full run and decoder validation as the long-running steps, and the code structure makes the neural and video loops the dominant work inside conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized further: `decode_bytes_array`, `parse_optional_float_array`, the per-trial filtering loop, the per-bin loop in `trial_tongue_categories`, the per-trial input/output construction loop, and especially the nested per-unit/per-trial neural writeback loop.

ii. 
```python
for i, value in enumerate(arr):
    ...
```

```python
for trial_idx in keep_idx:
    ...
```

```python
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    ...
```

```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    ...
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. The trajectory does not justify leaving these loops in Python; step 77 suggests the agent prioritized transparency and auditability over a tighter vectorized implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some processing across multiple passes. It loops once over trials to decide which ones are valid and collect labels, then loops again over the same retained trials to build inputs/outputs, and then iterates over all units while repeatedly writing rates trial by trial. It also performs several summary passes over the converted outputs.

ii. 
```python
for trial_idx in keep_idx:
    ...
```

```python
for row_idx, trial_idx in enumerate(valid_trials):
    ...
```

```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    ...
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. The trajectory does not call this out explicitly. The closest justification is step 77, where the agent says it wants fallback choices and counts to be auditable, which helps explain the separate bookkeeping pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter performs some processing that is not used downstream: it loads `photostim_power` only to gate stimulation, computes `session_output_counts` and never uses it, collects many `stats` counters only for reporting, stores `trial_indices_source` and `tongue_percentiles` in intermediate session dicts but not in the final output, and does a full nonzero-neural scan after constructing all neural trials.

ii. 
```python
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
...
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

iii. The trajectory does not justify these extra computations directly. Step 77’s emphasis on counting and documenting fallback decisions explains part of the extra bookkeeping.
