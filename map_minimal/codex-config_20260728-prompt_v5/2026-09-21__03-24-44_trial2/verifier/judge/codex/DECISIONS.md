# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session data by globbing every NWB file under `/app/data/sub-*`, sorting the paths, and opening each file once with `NWBHDF5IO`. Within each session it reads the NWB trials table, units table, and behavioral acquisitions directly from the NWB object.

ii. 
```python
def iter_session_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
```

```python
for i, path in enumerate(session_paths, 1):
    session_record, brain_region_to_idx = convert_session(path, brain_region_to_idx)
```

```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    trials_df = nwb.trials.to_dataframe()
    units = nwb.units
```

iii. In trajectory steps 4, 21, and 54, the agent says it wanted to use the NWB files as the canonical source and mirror the published preprocessing directly from that format rather than from older derived exports.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.subject_id` as the subject identifier for each session, with a fallback to the parent folder name if the NWB subject field were missing. It then assigns each session a subject index in first-seen order while iterating through the sorted sessions.

ii. 
```python
subject_id = str(getattr(nwb.subject, "subject_id", path.parent.name.replace("sub-", "")))
```

```python
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
...
subject_idx.append(subject_to_idx[subj])
```

iii. The trajectory does not dwell on subject ordering, but step 21 says the agent mapped the exact NWB fields for session metadata, and the final summary in step 151 treats the NWB `subject_id` as the canonical subject key.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order is the sorted file order, and each session is processed independently and appended once to the output lists.

ii. 
```python
def iter_session_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
```

```python
for i, path in enumerate(session_paths, 1):
    print(f"[{i}/{len(session_paths)}] converting {path.name}", flush=True)
```

iii. In steps 4 and 21 the agent explicitly notes that the raw data are organized as NWB sessions and that it wanted to convert directly from that session-level organization.

## 1-d. How are the data split into trials?

i. The AI starts from `nwb.trials.to_dataframe()`, but then redefines the usable trial set by taking the minimum length of `units["is_good_trials"][u]` across good units and truncating the trials table to the first `n_recorded_trials` rows. It then takes only the first `n_recorded_trials` go-cue and go-stop events and uses those truncated tables as the session’s trials.

ii. 
```python
trials_df = nwb.trials.to_dataframe()
...
recorded_trial_counts = [len(np.asarray(units["is_good_trials"][u])) for u in good_unit_indices]
n_recorded_trials = int(min(recorded_trial_counts)) if recorded_trial_counts else n_trials
...
if n_recorded_trials < n_trials:
    trials_df = trials_df.iloc[:n_recorded_trials].copy()
    n_trials = n_recorded_trials
```

```python
go_times, response_ends = get_go_times_and_response_ends(nwb, n_trials)
...
return go_times_all[:n_trials], response_ends_all[:n_trials]
```

iii. In steps 70 and 151, the agent says it would “trim each session to the recorded trial count implied by `units.is_good_trials`.” Step 80 shows the motivation: a smoke test revealed many all-zero trials late in a session, and the agent chose to treat that as recording coverage rather than as a bug in its assignment logic.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three effective trial filters. First, it truncates each session to the minimum `is_good_trials` length across good units. Second, after neural binning, it drops any trial whose neural matrix is entirely zero. Third, it drops sessions with fewer than two remaining trials. It does not explicitly filter `free_water` trials before conversion.

ii. 
```python
recorded_trial_counts = [len(np.asarray(units["is_good_trials"][u])) for u in good_unit_indices]
n_recorded_trials = int(min(recorded_trial_counts)) if recorded_trial_counts else n_trials
...
if n_recorded_trials < 2:
    return None, brain_region_to_idx
```

```python
valid_trial_indices = [i for i, trial in enumerate(neural_trials) if np.any(trial)]
if len(valid_trial_indices) < len(neural_trials):
    neural_trials = [neural_trials[i] for i in valid_trial_indices]
    input_trials = [input_trials[i] for i in valid_trial_indices]
    output_trials = [output_trials[i] for i in valid_trial_indices]
```

iii. Steps 80 and 151 provide the stated rationale: the agent found sessions where later trials had no spikes and concluded those should be filtered as missing neural coverage. Step 17 shows it knew that “regular trial” masks existed in prior analyses, but it chose to keep behaviorally diverse trials for the decoder task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units["spike_times"]` for the units it keeps, together with `BehavioralEvents/go_start_times` to place trial windows around the go cue.

ii. 
```python
spike_ends = np.asarray(units["spike_times"].data[:], dtype=np.int64)
spike_values = np.asarray(units["spike_times"].target.data[:], dtype=np.float64)
```

```python
events = nwb.acquisition["BehavioralEvents"].time_series
go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
```

iii. In steps 17, 21, and 54, the agent says it wanted go-cue-aligned neural activity straight from the NWB source data and to keep only quality-controlled units.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 50 ms bins from -2.5 s to +1.5 s around each go cue. For each good unit it assigns each spike to a trial by comparing against per-trial window ends, checks that the spike falls after that trial’s window start, converts spike times to go-cue-relative offsets, bins them with `np.floor`, accumulates counts with `np.add.at`, and divides by bin width to produce firing rates.

ii. 
```python
window_starts = go_times + WINDOW_START_S
window_ends = go_times + WINDOW_END_S
...
trial_idx = np.searchsorted(window_ends, spikes, side="right")
...
rel_spikes = spikes - go_times[trial_idx]
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
...
np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)
```

```python
trial_rates = counts[:, trial_idx, :].astype(np.float16) / np.float16(BIN_SIZE_S)
```

iii. Steps 54, 70, and 151 say the agent had “go-cue alignment, `-2.5s` to `+1.5s`, `50 ms` bins” locked in before writing, and that the resulting neural activity should match the decoder specification.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == "good"` and a non-empty `anno_name`. If no units pass, it drops the session. In practice, the extra `anno_name` condition is bundled into the good-unit mask used for all downstream neural processing.

ii. 
```python
classifications = get_vector_strings(units, "classification")
anno_names = get_vector_strings(units, "anno_name")
good_mask = (classifications == "good") & (anno_names != "")
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    return None, brain_region_to_idx
```

iii. Step 70 says the agent’s unit-level choice was to “keep only classifier-`good` units with nonempty histology labels,” and step 151 repeats that same summary as one of the key processing choices.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset. Each trial’s spike window is defined by `go_time + [-2.5, 1.5]`, so the binned neural matrices are already expressed on a common go-cue-relative grid.

ii. 
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
...
window_starts = go_times + WINDOW_START_S
window_ends = go_times + WINDOW_END_S
...
rel_spikes = spikes - go_times[trial_idx]
```

iii. Steps 54, 70, and 151 all explicitly state that go-cue alignment is the organizing event for the conversion.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins throughout. There are 80 non-overlapping bins over the 4 s window, and the code does not do any additional smoothing or temporal rebinning after that binning step.

ii. 
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
BIN_EDGES_S = np.linspace(WINDOW_START_S, WINDOW_END_S, N_BINS + 1, dtype=np.float64)
BIN_CENTERS_S = (BIN_EDGES_S[:-1] + BIN_EDGES_S[1:]) / 2.0
```

iii. Step 151 lists the go-cue, `-2.5` to `+1.5`, `50 ms` binning choice as a deliberate core decision to satisfy the decoder spec.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times` plus per-trial `start_time`, `stop_time`, and `go_start_times`. Its main rule is “last sample onset before the go cue,” but it also falls back to the last sample onset anywhere inside the trial, and finally to an imputed median go-minus-sample delay if it finds no sample event at all.

ii. 
```python
sample_starts = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
...
for i, (trial_start, go_time) in enumerate(zip(trial_starts, go_times)):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        per_trial[i] = sample_starts[hi - 1]
```

```python
for i in np.where(~valid)[0]:
    ...
    if hi > lo:
        per_trial[i] = sample_starts[hi - 1]
    else:
        per_trial[i] = go_times[i] - median_go_minus_sample
```

iii. Step 28 is the clearest rationale: the agent identified that early licks can replay sample epochs and concluded the right tone reference was the final sample onset before the go cue. Step 70 repeats that this was the locked-in choice.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding a per-trial sample onset, the AI converts it to a go-relative offset and subtracts that offset from each go-relative bin center. That produces one continuous time-from-tone value per bin.

ii. 
```python
sample_onsets = get_sample_onsets(nwb, trial_starts, go_times, trial_stops)
sample_onsets_rel = sample_onsets - go_times
...
trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
```

iii. Step 70 says the input should be “time from the last sample/tone onset before go,” and the final summary in step 151 presents that as a deliberate implemented feature.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI uses the same `BIN_CENTERS_S` grid that defines the neural bins, so the time-from-tone input is a bin-by-bin value on the exact same go-cue-aligned timeline as the neural data.

ii. 
```python
BIN_CENTERS_S = (BIN_EDGES_S[:-1] + BIN_EDGES_S[1:]) / 2.0
...
trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
```

iii. Steps 54 and 151 frame the whole converter around a shared go-cue-aligned binning grid, and the time-from-tone feature is built directly on that grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trials-table fields `photostim_onset`, `photostim_duration`, and `start_time`, then expresses the resulting interval relative to the trial’s `go_time`.

ii. 
```python
for i, trial in enumerate(trials_df.itertuples()):
    onset = maybe_float(trial.photostim_onset)
    duration = maybe_float(trial.photostim_duration)
    ...
    start_abs = float(trial.start_time) + onset
    end_abs = start_abs + duration
    stim_starts[i] = start_abs - go_times[i]
    stim_ends[i] = end_abs - go_times[i]
```

iii. Step 21 says the agent mapped the exact NWB fields for photostimulation timing before coding, and step 151 says it implemented per-bin photostim on/off from those fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts the onset and duration into a per-bin binary time series. For each trial, bins whose centers lie within `[stim_start_rel, stim_end_rel)` are assigned 1; non-stimulated trials are all zeros.

ii. 
```python
if np.isnan(stim_starts_rel[trial_idx]) or np.isnan(stim_ends_rel[trial_idx]):
    trial_input[1] = 0.0
else:
    trial_input[1] = (
        (BIN_CENTERS_S >= stim_starts_rel[trial_idx]) & (BIN_CENTERS_S < stim_ends_rel[trial_idx])
    ).astype(np.float32)
```

iii. Step 151 summarizes this as building “per-bin photostim on/off,” which matches the code.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI places photostimulation on the same go-cue-relative axis as neural activity by subtracting `go_time` from the absolute stimulation interval and then comparing that interval with the shared bin centers.

ii. 
```python
stim_starts[i] = start_abs - go_times[i]
stim_ends[i] = end_abs - go_times[i]
...
(BIN_CENTERS_S >= stim_starts_rel[trial_idx]) & (BIN_CENTERS_S < stim_ends_rel[trial_idx])
```

iii. Steps 54 and 151 make clear that all time-varying streams were intended to live on the common go-cue-aligned bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI does not derive choice from the trials table. Instead it uses `BehavioralEvents/left_lick_times`, `BehavioralEvents/right_lick_times`, `go_start_times`, and `go_stop_times`, defining choice as the first lick side in the post-go response window and `no lick` when neither side licks before `go_stop_time`.

ii. 
```python
left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
```

```python
for i, (go_time, response_end) in enumerate(zip(go_times, response_ends)):
    left_idx = np.searchsorted(left_licks, go_time, side="left")
    right_idx = np.searchsorted(right_licks, go_time, side="left")
    ...
    if left_time < right_time:
        choices[i] = CHOICE_TO_INT["left"]
    elif right_time < left_time:
        choices[i] = CHOICE_TO_INT["right"]
    else:
        choices[i] = CHOICE_TO_INT["no lick"]
```

iii. Step 70 says the agent deliberately chose to “derive choice from the first response-window lick,” and step 151 repeats that as one of the defining output choices.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps the first-response rule to codes `0 = left`, `1 = right`, `2 = no lick`, and then repeats that scalar category across all 80 bins of the trial’s output array.

ii. 
```python
CHOICE_TO_INT = {"left": 0, "right": 1, "no lick": 2}
```

```python
trial_output = np.empty((4, N_BINS), dtype=np.int8)
trial_output[0] = choices[trial_idx]
```

iii. The explicit rationale in steps 70 and 151 is that the first response-window lick was the agent’s chosen operational definition of choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derives outcome directly from the `outcome` column of the trials table.

ii. 
```python
outcome_strings = trials_df["outcome"].astype(str).to_numpy()
```

iii. The trajectory does not debate this field; once the agent mapped the NWB trials columns in step 21, the outcome variable was read directly from the trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome strings to integers with `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeats the per-trial code across all bins.

ii. 
```python
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
...
outcome_labels = np.asarray([OUTCOME_TO_INT[x] for x in outcome_strings], dtype=np.int8)
...
trial_output[1] = outcome_labels[trial_idx]
```

iii. Step 151 lists outcome as one of the final outputs and does not suggest any extra derivation beyond this direct mapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI derives early lick directly from the `early_lick` column of the trials table.

ii. 
```python
early_strings = trials_df["early_lick"].astype(str).to_numpy()
```

iii. The trajectory treats early lick as a straightforward trial-table output once the relevant NWB columns had been mapped.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats that category across all bins of the trial output.

ii. 
```python
EARLY_TO_INT = {"no early": 0, "early": 1}
...
early_labels = np.asarray([EARLY_TO_INT[x] for x in early_strings], dtype=np.int8)
...
trial_output[2] = early_labels[trial_idx]
```

iii. Step 151 lists early lick as one of the outputs the converter writes directly into the decoder dataset.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `BehavioralTimeSeries/Camera0_side_TongueTracking`, using `timestamps`, `data[:, 1]` as tongue y, and `data[:, 2]` as the DeepLabCut likelihood.

ii. 
```python
ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
timestamps = np.asarray(ts.timestamps[:], dtype=np.float64)
data = np.asarray(ts.data[:], dtype=np.float32)
y = data[:, 1]
likelihood = data[:, 2]
```

iii. Step 21 says the agent specifically mapped the exact NWB fields for tongue tracking before coding, and step 151 says tongue position was then discretized session-wise from those data.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI first marks frames with `likelihood < 0.9` as not visible for threshold computation. It computes session-level 40th and 60th percentiles from the raw y-values of visible frames, not from 50 ms bin means. For each trial and each 50 ms bin, it takes at most one frame: the last frame before the bin end, provided that frame lies within the bin. If that frame is visible, the bin is labeled by comparing its y-value to the session quantiles; otherwise the bin stays in class `3`.

ii. 
```python
TONGUE_VISIBLE_THRESHOLD = 0.9
...
visible = likelihood >= likelihood_threshold
if np.any(visible):
    q40, q60 = np.quantile(y[visible], [0.4, 0.6]).astype(np.float32)
```

```python
frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
labels = np.full(N_BINS, 3, dtype=np.int8)
...
y_valid = y[idx]
like_valid = likelihood[idx] >= likelihood_threshold
...
labels[visible_bins[y_vis < q40]] = 0
mid = (y_vis >= q40) & (y_vis <= q60)
labels[visible_bins[mid]] = 1
labels[visible_bins[y_vis > q60]] = 2
```

iii. Step 70 says the agent intended to “discretize tongue position from session-wide visible-frame percentiles using DLC confidence,” and step 151 repeats that session-wise discretization choice. The stricter `0.9` threshold and frame-based quantiles come from the code rather than an explicit trajectory discussion.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI thresholds using two session-wide cut points `q40` and `q60` computed from visible-frame y-values. Values `< q40` become class `0`, values `q40 <= y <= q60` become class `1`, values `> q60` become class `2`, and bins with no accepted visible frame remain class `3`.

ii. 
```python
if np.any(visible):
    q40, q60 = np.quantile(y[visible], [0.4, 0.6]).astype(np.float32)
...
labels[visible_bins[y_vis < q40]] = 0
mid = (y_vis >= q40) & (y_vis <= q60)
labels[visible_bins[mid]] = 1
labels[visible_bins[y_vis > q60]] = 2
```

iii. The only explicit trajectory justification is step 70’s phrase “session-wide visible-frame percentiles using DLC confidence”; the exact thresholding details are encoded in the script itself.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue labels on the same 50 ms go-cue-relative bins as the neural data. For each trial it builds `bin_starts` and `bin_ends` from `go_time + BIN_EDGES_S`, then searches the camera timestamps against those per-bin edges.

ii. 
```python
for go_time in go_times:
    bin_starts = go_time + BIN_EDGES_S[:-1]
    bin_ends = go_time + BIN_EDGES_S[1:]
    frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
```

iii. Steps 54 and 151 make the shared go-cue-aligned 50 ms grid the central alignment device for all time-varying streams, including tongue position.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several kinds of missingness explicitly. It maps non-numeric or `"N/A"` photostim values to `None`/`NaN`, maps missing string columns to empty strings, drops sessions with no good units, drops trials whose neural matrices are entirely zero, and assigns tongue bins with no accepted visible frame to class `3`. It also imputes a sample onset using the within-session median go-minus-sample delay if no suitable sample event is found for a trial.

ii. 
```python
def maybe_float(value) -> float | None:
    text = str(value)
    if text == "N/A":
        return None
```

```python
def get_vector_strings(table, column: str) -> np.ndarray:
    raw = table[column].data[:]
    return np.asarray(["" if x is None else str(x) for x in raw], dtype=object)
```

```python
if hi > lo:
    per_trial[i] = sample_starts[hi - 1]
else:
    per_trial[i] = go_times[i] - median_go_minus_sample
```

```python
valid_trial_indices = [i for i, trial in enumerate(neural_trials) if np.any(trial)]
...
labels = np.full(N_BINS, 3, dtype=np.int8)
```

iii. Step 80 shows the agent explicitly reacting to missing-spike coverage by deciding to drop all-zero neural trials. Steps 28 and 70 explain the fallback tone-onset logic as protection against ambiguous or repeated sample events.

## 10-a. What are the most time-consuming steps of the code?

i. The code’s likely bottlenecks are repeated NWB file reads across all sessions, the per-unit spike binning loop, and the per-trial tongue-label loop. The script also builds many per-trial arrays in Python lists and writes a large pickle at the end.

ii. 
```python
for i, path in enumerate(session_paths, 1):
    session_record, brain_region_to_idx = convert_session(path, brain_region_to_idx)
```

```python
for unit_pos, unit_idx in enumerate(good_unit_indices):
    ...
    np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)
```

```python
for go_time in go_times:
    ...
    per_trial.append(labels)
```

iii. The trajectory only addresses runtime indirectly. Step 112 says the full pass was proceeding file-by-file faster than expected, and step 80 identifies neural coverage issues during a smoke test; it does not contain a detailed profiling analysis.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain unvectorized: the per-trial sample-onset search loop, the per-trial photostim conversion loop, the per-trial first-lick choice loop, the per-trial tongue-label loop, the per-unit spike loop, and the final per-trial assembly loop for input/output arrays.

ii. 
```python
for i, (trial_start, go_time) in enumerate(zip(trial_starts, go_times)):
    ...
```

```python
for i, trial in enumerate(trials_df.itertuples()):
    ...
```

```python
for i, (go_time, response_end) in enumerate(zip(go_times, response_ends)):
    ...
```

```python
for unit_pos, unit_idx in enumerate(good_unit_indices):
    ...
```

iii. The trajectory does not justify these loops explicitly. Step 73 says the agent wanted “explicit helpers” and “inspectable” preprocessing logic, which suggests readability was favored over more aggressive vectorization.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several search-and-assembly passes over the same trial structure: one pass to infer sample onsets, one for photostim intervals, one for choices, one for tongue labels, one for neural binning, and another to assemble `input_trials` and `output_trials`. It also calls `iter_session_paths(data_dir)` twice, once for conversion and once again when filling metadata.

ii. 
```python
sample_onsets = get_sample_onsets(...)
stim_starts_rel, stim_ends_rel = get_photostim_relative_intervals(...)
choices = get_first_response_choices(...)
tongue_labels, tongue_quantiles = build_tongue_binned_labels(...)
neural_trials = build_neural_trials(...)
```

```python
for trial_idx in range(n_trials):
    ...
    input_trials.append(trial_input)
    ...
    output_trials.append(trial_output)
```

```python
"n_source_sessions": len(iter_session_paths(data_dir)),
```

iii. The trajectory does not call this out as a concern. Step 73 instead emphasizes code inspectability, so the repeated passes appear to be a consequence of decomposing the converter into separate helper functions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is `compute_control_performance`, which computes behavioral performance statistics, left/right hit counts, and regular-trial counts that are stored only in metadata and are not used to construct `neural`, `input`, or `output`. The script also stores tongue quantiles in metadata even though downstream decoding uses only the discretized labels.

ii. 
```python
def compute_control_performance(trials_df) -> tuple[float, int, int, int]:
    ...
    return performance, left_hits, right_hits, n_regular
```

```python
performance, left_hits, right_hits, n_regular = compute_control_performance(trials_df)
...
"control_performance_non_early_no_auto_free": performance,
"control_hit_left": left_hits,
"control_hit_right": right_hits,
"control_trials_responded": n_regular,
"tongue_visible_q40": tongue_quantiles[0],
"tongue_visible_q60": tongue_quantiles[1],
```

iii. The trajectory does not justify this extra metadata computation. Step 17 notes the concept of “regular trial” performance from the papers, so this looks like an added diagnostic carried into metadata rather than something required by the decoder dataset.
