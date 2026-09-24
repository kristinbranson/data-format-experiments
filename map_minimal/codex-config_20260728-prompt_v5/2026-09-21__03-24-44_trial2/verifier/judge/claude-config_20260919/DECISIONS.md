# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is read directly from the published NWB files (one file per session) under `/app/data/sub-<id>/`. A single sorted glob enumerates every session file; each file is opened once with `pynwb.NWBHDF5IO` inside a `with` block and everything needed (trials table, units table, behavioral events, video tracking, subject id) is pulled from that one open handle. Sessions are processed sequentially in sorted path order and accumulated into per-session lists. All 174 files are visited; 173 survive curation. The agent deliberately chose the NWB files over the older `.mat` export referenced by `/app/code`.

ii.
```python
DATA_DIR = Path("/app/data")

def iter_session_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
```
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    trials_df = nwb.trials.to_dataframe()
    ...
    units = nwb.units
    ...
    events = nwb.acquisition["BehavioralEvents"].time_series
```
```python
for i, path in enumerate(session_paths, 1):
    print(f"[{i}/{len(session_paths)}] converting {path.name}", flush=True)
    try:
        session_record, brain_region_to_idx = convert_session(path, brain_region_to_idx)
    except Exception as exc:
        skipped_sessions.append({"session_path": str(path), "reason": repr(exc)})
```

iii. From the trajectory (steps 16–20): after reading `/app/code`, the agent noted the reference pipeline was written against a cluster-specific `.mat` export and that "the NWB files expose trials and units directly, not just the old `.mat` export", so the conversion should be done "from the canonical source rather than guessing around the old pipeline". It also avoided `nwb.units.to_dataframe()` because that "may silently drag in ragged spike-time columns and make the conversion much slower than it needs to be" (step 65), reading the ragged `spike_times` buffer via the `.data`/`.target.data` offset representation instead.

## 1-b. How are the data split into subjects?

i. Each session's animal is read from `nwb.subject.subject_id` (a numeric string, e.g. `'440956'`), falling back to the containing directory name with the `sub-` prefix stripped if that attribute is missing. `subjects` is built as the list of unique ids in order of first appearance across the sorted file list, and `subject_idx` records each session's index into that list. This yields 28 subjects with 3–10 sessions each.

ii.
```python
subject_id = str(getattr(nwb.subject, "subject_id", path.parent.name.replace("sub-", "")))
```
```python
subj = session_record["subject_id"]
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
...
subject_idx.append(subject_to_idx[subj])
```
```python
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int32),
```

iii. Not discussed explicitly in the trajectory; the agent treated `subject_id` as the canonical animal identifier exposed by NWB and the directory layout (`sub-<id>/`) as derived from it, so no separate grouping step was needed. The directory-name fallback is a defensive path in case `nwb.subject` is absent.

## 1-c. How are the data split into sessions?

i. One NWB file is taken to be one session; no grouping or splitting is performed. Session order in the output is the sorted file-path order, which (because the filename embeds the acquisition timestamp, `ses-20190207T120657`) is chronological within each subject. Each session is identified in the output by its file path rather than by `nwb.identifier`; the path, trial count, good-unit count and per-session behavioral/tongue statistics are stored in `metadata['session_info']`, and any dropped session is recorded with a reason in `metadata['skipped_sessions']`.

ii.
```python
session_record = {
    "session_path": str(path),
    "subject_id": subject_id,
    ...
    "session_info": {
        "session_path": str(path),
        "n_trials": n_trials,
        "n_good_units": int(len(good_unit_indices)),
        "control_performance_non_early_no_auto_free": performance,
        ...
    },
}
```
```python
"n_source_sessions": len(iter_session_paths(data_dir)),
"n_kept_sessions": len(neural),
"session_info": session_info,
"skipped_sessions": skipped_sessions,
```

iii. Steps 38/58/106: the agent explicitly tried to reconcile the 174 files on disk against the paper's "173 sessions / 69,943 good units". It first tested the data paper's behavioral performance criterion as a session filter and rejected it because it "keeps only 145 of 174 sessions, so that criterion was clearly analysis-specific rather than the dataset-wide inclusion rule". It then found that dropping the single file with no quality-controlled units "lands at 173 sessions and 69,453 good labeled units, which is very close to the paper's 173-session / 69,943-good-unit scale", and adopted that as the session rule.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials.to_dataframe()`), one row per behavioral trial. Per-trial event times are taken from the `BehavioralEvents` streams by positional index: `go_start_times` and `go_stop_times` are sliced to the first `n_trials` entries and assumed to be in one-to-one row order with the trials table. The code raises if there are fewer go events than trial rows (so a session with a genuine count mismatch is skipped and logged), but tolerates more. A session with fewer than 2 trials is dropped, as required by the target format.

ii.
```python
def get_go_times_and_response_ends(nwb, n_trials):
    events = nwb.acquisition["BehavioralEvents"].time_series
    go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
    response_ends_all = np.asarray(events["go_stop_times"].timestamps[:], dtype=np.float64)
    if len(go_times_all) < n_trials or len(response_ends_all) < n_trials:
        raise ValueError(...)
    return go_times_all[:n_trials], response_ends_all[:n_trials]
```
```python
trials_df = nwb.trials.to_dataframe()
n_trials = len(trials_df)
if n_trials < 2:
    return None, brain_region_to_idx
```

iii. Step 89: the agent originally asserted that the go-cue event count equalled the trial count, and relaxed it after hitting a session where "the go-cue event stream still has all 480 behavioral events even when the neural recording only covers the first 160 trials" — "I'm relaxing that check so the converter slices the event arrays to the recorded-trial count instead of treating it as an error." It also verified in step 29 that, unlike `sample_start_times`/`delay_start_times` (which replay on early-lick trials), there is exactly one go event per trial.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at trials with no spike data; no behavioral quality filter is applied and early-lick / ignore trials are deliberately kept.

1. **Recorded-trial trim.** For every good unit the length of its ragged `units['is_good_trials']` entry is read; the session is truncated to the *first* `min(len(is_good_trials))` trials, on the assumption that the ephys-covered trials are the leading ones.
2. **All-zero trial drop.** After binning, any trial whose entire (n_neurons × 80) firing-rate matrix is zero is removed, together with its matching input/output entries and trials-table row.

A session is dropped if fewer than 2 trials survive either stage. Net result: 93,310 trials after the trim, 90,734 after the all-zero drop (reference: 90,860).

ii.
```python
recorded_trial_counts = [len(np.asarray(units["is_good_trials"][u])) for u in good_unit_indices]
n_recorded_trials = int(min(recorded_trial_counts)) if recorded_trial_counts else n_trials
n_recorded_trials = min(n_recorded_trials, n_trials)
if n_recorded_trials < 2:
    return None, brain_region_to_idx
if n_recorded_trials < n_trials:
    trials_df = trials_df.iloc[:n_recorded_trials].copy()
    n_trials = n_recorded_trials
```
```python
valid_trial_indices = [i for i, trial in enumerate(neural_trials) if np.any(trial)]
if len(valid_trial_indices) < len(neural_trials):
    neural_trials = [neural_trials[i] for i in valid_trial_indices]
    input_trials = [input_trials[i] for i in valid_trial_indices]
    output_trials = [output_trials[i] for i in valid_trial_indices]
    trials_df = trials_df.iloc[valid_trial_indices].reset_index(drop=True)
    n_trials = len(valid_trial_indices)
```

iii. Steps 79–83: a smoke test surfaced "one session has valid shapes but no spikes at all for the back half of its trials". The agent checked whether this was its own bug or the data and concluded "The issue is in the source data: that second NWB file has 480 behavioral trials, but every good unit's spikes end at about 1107 s, exactly around trial 158. So the right fix is to drop unrecorded late trials per session, not to keep hundreds of behavior-only trials with zero neural data." It inspected `units['obs_intervals']` (step 95) but ultimately used the `is_good_trials` length as the "recorded trial count" proxy, and added the all-zero-trial drop as a catch-all for any remaining trial with no spikes. It kept every behavioral trial type, having explicitly rejected the reference pipeline's "regular trial" mask (early-lick / auto-water / free-water / no-response / stim removal) as analysis-specific: "this task does require keeping photostim trials" (step 16).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `units/spike_times` (session-absolute spike times, stored ragged as a values buffer plus per-unit end offsets) for the units that pass quality control, and from `BehavioralEvents/go_start_times`, which places the bin grid.

ii.
```python
units = nwb.units
spike_ends = np.asarray(units["spike_times"].data[:], dtype=np.int64)
spike_values = np.asarray(units["spike_times"].target.data[:], dtype=np.float64)
...
for unit_pos, unit_idx in enumerate(good_unit_indices):
    start = 0 if unit_idx == 0 else spike_ends[unit_idx - 1]
    end = spike_ends[unit_idx]
    spikes = spike_values[start:end]
```

iii. Step 65: the agent chose the raw offset/target buffers over `units.to_dataframe()` for speed. Spike times are the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin are converted to firing rates in Hz by dividing by the bin width. No smoothing, normalisation, baseline subtraction or z-scoring. Counts are accumulated in a `uint16` (n_units, n_trials, 80) array and each trial is sliced out and cast to **float16** before division. Spikes are assigned to a trial via `searchsorted` on the window end times and then to a bin by flooring the go-cue-relative time.

ii.
```python
counts = np.zeros((n_units, n_trials, N_BINS), dtype=np.uint16)
window_starts = go_times + WINDOW_START_S
window_ends = go_times + WINDOW_END_S
...
    trial_idx = np.searchsorted(window_ends, spikes, side="right")
    valid = trial_idx < n_trials
    ...
    in_window = spikes >= window_starts[trial_idx]
    ...
    rel_spikes = spikes - go_times[trial_idx]
    bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
    good = (bin_idx >= 0) & (bin_idx < N_BINS)
    if np.any(good):
        np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)

neural_trials = []
for trial_idx in range(n_trials):
    trial_rates = counts[:, trial_idx, :].astype(np.float16) / np.float16(BIN_SIZE_S)
    neural_trials.append(trial_rates)
```

iii. Not discussed in detail in the trajectory beyond the plan statement to "align spikes and behavior to go cue" with the specified window and bin size. The float16 choice appears in step 102, where the agent sized the output (`est_float16_bytes = good_total * trial_total / kept_sessions * 80 * 2`) before committing to the full export; the resulting pickle is 5.97 GB (the reference's float32 version is 11.9 GB).

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is kept only if `units['classification'] == 'good'` **and** its `anno_name` (histological annotation) is non-empty. No per-metric thresholds (presence ratio, ISI violation, amplitude cutoff) are applied, even though those columns exist. A session with zero surviving units is dropped. This keeps 69,453 of 272,227 units, a mean of 401.46 per session — identical to the expert reference.

ii.
```python
classifications = get_vector_strings(units, "classification")
anno_names = get_vector_strings(units, "anno_name")
good_mask = (classifications == "good") & (anno_names != "")
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    return None, brain_region_to_idx
```
```python
def get_vector_strings(table, column):
    raw = table[column].data[:]
    return np.asarray(["" if x is None else str(x) for x in raw], dtype=object)
```

iii. Step 69: "keep only classifier-`good` units with nonempty histology labels". The agent validated the rule dataset-wide (steps 102–106) and reported that it "lands at 173 sessions and 69,453 good labeled units, which is very close to the paper's 173-session / 69,943-good-unit scale", treating agreement with the spike-sorting QC white paper as the criterion. It explicitly checked the alternative `unit_quality` column earlier (step 21) and did not use it. The `anno_name` requirement additionally guarantees every retained unit can be assigned to a `brain_regions` entry.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset, taken from `BehavioralEvents/go_start_times`. Spikes and events share one session-absolute clock, so no resampling or offset correction is needed: for each trial an absolute window `[go - 2.5 s, go + 1.5 s)` is formed, each spike is mapped to the first trial whose window end exceeds it, checked against that window's start, and binned by its time relative to that trial's go cue.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
...
window_starts = go_times + WINDOW_START_S
window_ends = go_times + WINDOW_END_S
...
trial_idx = np.searchsorted(window_ends, spikes, side="right")
in_window = spikes >= window_starts[trial_idx]
rel_spikes = spikes - go_times[trial_idx]
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
```
```python
"temporal_alignment_event": "go cue onset",
"off_start": WINDOW_START_S,
"off_end": WINDOW_END_S,
```

iii. Step 16: the agent first noted from the reference code that "spikes are already go-cue aligned in the raw session data", then confirmed the NWB event streams directly (steps 21–22, 29), printing per-trial `presample/sample/delay/go` times to check the epoch structure before writing the binning code. The alignment event and window are taken straight from the Decoder Task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 per trial, spanning -2.5 s to +1.5 s about the go cue — the same fixed grid for every trial and session, so all trials have exactly 80 timepoints. There is no rebinning: spike times are histogrammed once directly onto this grid from the raw event times. `metadata['time_bin_size']` is reported as 50.0 ms and the bin centres are also stored.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))   # 80
BIN_EDGES_S = np.linspace(WINDOW_START_S, WINDOW_END_S, N_BINS + 1, dtype=np.float64)
BIN_CENTERS_S = (BIN_EDGES_S[:-1] + BIN_EDGES_S[1:]) / 2.0
```
```python
"time_bin_size": BIN_SIZE_S * 1000.0,
"time_bin_centers_s": BIN_CENTERS_S.astype(float).tolist(),
```

iii. Step 150: "Go-cue alignment, `-2.5s` to `+1.5s`, `50 ms` bins are defined in convert_data.py#L15" — taken directly from the Decoder Task specification. Defining the grid once at module level guarantees the constant-timepoint requirement of the target format.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (the sample-epoch tone onsets) together with each trial's go-cue time and the trials table's `start_time`/`stop_time`. For each trial the tone is the **last** `sample_start_times` event that falls between the trial's start time and its go cue. Two fallbacks exist: if no sample event is found in `[trial_start, go]`, the last event in `[trial_start, trial_stop]` is used; failing that, the tone is synthesised as `go − median(go − tone)` over the session.

ii.
```python
def get_sample_onsets(nwb, trial_starts, go_times, trial_stops):
    sample_starts = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
    per_trial = np.full(len(go_times), np.nan, dtype=np.float64)
    for i, (trial_start, go_time) in enumerate(zip(trial_starts, go_times)):
        lo = np.searchsorted(sample_starts, trial_start, side="left")
        hi = np.searchsorted(sample_starts, go_time, side="right")
        if hi > lo:
            per_trial[i] = sample_starts[hi - 1]
    valid = ~np.isnan(per_trial)
    median_go_minus_sample = float(np.median(go_times[valid] - per_trial[valid])) if np.any(valid) else 1.85
    for i in np.where(~valid)[0]:
        ...
        per_trial[i] = go_times[i] - median_go_minus_sample
    return per_trial
```

iii. Step 27: "'tone onset' is not a single NWB trial column because early-lick trials can replay sample epochs... I'll inspect one session's event timings against the trial table to decide whether the correct per-trial tone reference is the final sample onset before the go cue, which is what the reference preprocessing strongly suggests." Step 29 confirmed this empirically by printing trials where an early lick produced two `sample_start_times` (e.g. trial 2: sample starts `[7.8919, 8.6411]`, go at `10.4911`, "last sample before go 8.6411"). Step 69: "derive tone onset from the final sample onset before each go cue".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is expressed relative to the go cue (`sample_onsets − go_times`) and subtracted from the go-cue-relative bin centres, giving a continuous, monotonically increasing per-bin value equal to seconds elapsed since that trial's tone. It is stored as `float32` in row 0 of the input array. It is *not* converted to a binary onset indicator.

ii.
```python
sample_onsets = get_sample_onsets(nwb, trial_starts, go_times, trial_stops)
sample_onsets_rel = sample_onsets - go_times
...
trial_input = np.empty((2, N_BINS), dtype=np.float32)
trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
```
```python
"input_names": ["time_from_tone_onset_s", "photostimulation_on"],
```

iii. Step 150: "Built decoder inputs as time from the last sample/tone onset before go". The instructions specify this input as "continuous, time-varying", so the elapsed-time encoding was used rather than the binary-onset representation the format notes offer for event times.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same go-cue-relative grid used for the spike binning — `BIN_CENTERS_S` are the centres of the same 80 bins whose edges (`BIN_EDGES_S`) define the spike histogram — so bin *k* of the input covers the same interval as bin *k* of the firing rates by construction. No interpolation or resampling.

ii.
```python
BIN_EDGES_S = np.linspace(WINDOW_START_S, WINDOW_END_S, N_BINS + 1, dtype=np.float64)
BIN_CENTERS_S = (BIN_EDGES_S[:-1] + BIN_EDGES_S[1:]) / 2.0
```
```python
trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
```

iii. Not separately discussed; alignment falls out of using the single module-level grid for every stream.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials-table columns `photostim_onset` and `photostim_duration` (stored as strings, with `'N/A'` on unstimulated trials, and measured relative to the trial's `start_time`), plus `start_time` and the trial's go cue to re-express them on the go-cue axis. The `photostim_start_times` / `photostim_stop_times` event streams are not used.

ii.
```python
def get_photostim_relative_intervals(trials_df, go_times):
    stim_starts = np.full(len(trials_df), np.nan, dtype=np.float64)
    stim_ends = np.full(len(trials_df), np.nan, dtype=np.float64)
    for i, trial in enumerate(trials_df.itertuples()):
        onset = maybe_float(trial.photostim_onset)
        duration = maybe_float(trial.photostim_duration)
        if onset is None or duration is None:
            continue
        start_abs = float(trial.start_time) + onset
        end_abs = start_abs + duration
        stim_starts[i] = start_abs - go_times[i]
        stim_ends[i] = end_abs - go_times[i]
    return stim_starts, stim_ends
```
```python
def maybe_float(value):
    text = str(value)
    if text == "N/A":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out
```

iii. Step 16: the agent noted that the reference analysis discards photostimulation trials but that "this task does require keeping photostim trials", so the stimulation had to be represented as an input rather than used as an exclusion criterion. The `'N/A'`-to-`None` handling follows directly from the string encoding it observed in the trials table (step 21).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary per-bin time series: a bin is 1 if its centre lies in `[stim_onset, stim_offset)` relative to the go cue, else 0. Trials with no stimulation (NaN bounds) are explicitly set to all-zero. Stored as `float32` in row 1 of the input array.

ii.
```python
if np.isnan(stim_starts_rel[trial_idx]) or np.isnan(stim_ends_rel[trial_idx]):
    trial_input[1] = 0.0
else:
    trial_input[1] = (
        (BIN_CENTERS_S >= stim_starts_rel[trial_idx]) & (BIN_CENTERS_S < stim_ends_rel[trial_idx])
    ).astype(np.float32)
```

iii. Step 150: "per-bin photostim on/off". The instructions specify this input as "Whether photostimulation is on at every time point (discrete, time-varying)", so a per-bin binary indicator rather than a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are converted from trial-start-relative to go-cue-relative (`start_time + onset − go_time`) and then compared against the same `BIN_CENTERS_S` grid used for the firing rates, so the stimulation window lands on the same 80 bins as the neural data.

ii.
```python
stim_starts[i] = start_abs - go_times[i]
stim_ends[i] = end_abs - go_times[i]
```
```python
trial_input[1] = ((BIN_CENTERS_S >= stim_starts_rel[trial_idx]) & (BIN_CENTERS_S < stim_ends_rel[trial_idx])).astype(np.float32)
```

iii. Not separately discussed; the re-referencing to the go cue is the necessary consequence of the trials table storing onsets relative to trial start.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column, so choice is derived from the raw lick event streams `BehavioralEvents/left_lick_times` and `BehavioralEvents/right_lick_times`, combined with `go_start_times` and **`go_stop_times`**, which the agent treats as the end of the response window. For each trial, the first left lick and the first right lick at or after the go cue are found; whichever is earlier and falls before `go_stop_times` sets the choice; if neither qualifies, the trial is labelled "no lick". `trial_instruction` and `outcome` are not used for this output.

ii.
```python
def get_go_times_and_response_ends(nwb, n_trials):
    go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
    response_ends_all = np.asarray(events["go_stop_times"].timestamps[:], dtype=np.float64)
    ...
    return go_times_all[:n_trials], response_ends_all[:n_trials]
```
```python
def get_first_response_choices(nwb, go_times, response_ends):
    left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
    right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
    for i, (go_time, response_end) in enumerate(zip(go_times, response_ends)):
        left_idx = np.searchsorted(left_licks, go_time, side="left")
        right_idx = np.searchsorted(right_licks, go_time, side="left")
        left_time = left_licks[left_idx] if left_idx < len(left_licks) else np.inf
        right_time = right_licks[right_idx] if right_idx < len(right_licks) else np.inf
        if left_time >= response_end:
            left_time = np.inf
        if right_time >= response_end:
            right_time = np.inf
```

iii. Step 69: "derive choice from the first response-window lick". The agent verified the event streams on the *first* session file only (steps 21–22), where `go_stop_times − go_start_times` is 1.5 s and therefore does coincide with the response window; it generalised that reading to the whole dataset without re-checking. Using actual lick events rather than `instruction × outcome` was presumably chosen as the more direct measurement of the animal's lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The earlier of the two candidate lick times determines the class: `0` left, `1` right; a tie (which occurs only when both are `inf`, i.e. no qualifying lick) gives `2` "no lick". The per-trial integer is broadcast across all 80 bins into row 0 of an `int8` (4, 80) output array, and `output_values[0] = ['left', 'right', 'no lick']`.

ii.
```python
CHOICE_TO_INT = {"left": 0, "right": 1, "no lick": 2}
...
        if left_time < right_time:
            choices[i] = CHOICE_TO_INT["left"]
        elif right_time < left_time:
            choices[i] = CHOICE_TO_INT["right"]
        else:
            choices[i] = CHOICE_TO_INT["no lick"]
```
```python
trial_output = np.empty((4, N_BINS), dtype=np.int8)
trial_output[0] = choices[trial_idx]
```
```python
"output_names": ["choice", "outcome", "early_lick", "tongue_y_position"],
"output_values": [
    ["left", "right", "no lick"],
    ...
],
```

iii. Step 150: "Built outputs as choice from first response-window lick". Broadcasting a per-trial value across bins follows the format note that outputs should be time-varying "if at all possible" and keeps all four outputs in one `(n_output, n_timepoints)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials table's `outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome_strings = trials_df["outcome"].astype(str).to_numpy()
```

iii. Step 21 printed the unique values of `outcome` and confirmed they are exactly the three categories the instructions ask for, so no derivation is required.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped through a fixed dictionary to `0` ignore, `1` miss, `2` hit, and the per-trial code is broadcast across all 80 bins into row 1 of the output array. `output_values[1] = ['ignore', 'miss', 'hit']`, matching the order given in the instructions.

ii.
```python
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
...
outcome_labels = np.asarray([OUTCOME_TO_INT[x] for x in outcome_strings], dtype=np.int8)
...
trial_output[1] = outcome_labels[trial_idx]
```

iii. The code assignment follows the instructions' ordering ("ignore, miss, hit"); no further justification given in the trajectory.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the trials table's `early_lick` column, which holds `'no early'` / `'early'`.

ii.
```python
early_strings = trials_df["early_lick"].astype(str).to_numpy()
```

iii. Step 21 confirmed the two unique values. The flag is stored per trial; the lick that sets it occurs during the sample or delay epoch, which lies inside the -2.5 s pre-go window.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped through a fixed dictionary to `0` no, `1` yes and broadcast across all 80 bins into row 2 of the output array; `output_values[2] = ['no', 'yes']`.

ii.
```python
EARLY_TO_INT = {"no early": 0, "early": 1}
...
early_labels = np.asarray([EARLY_TO_INT[x] for x in early_strings], dtype=np.int8)
...
trial_output[2] = early_labels[trial_idx]
```

iii. Follows the instructions' "(no, yes)" ordering. Note that because the mapping is a strict dictionary lookup, an unexpected string would raise and the whole session would be logged as skipped rather than silently mislabelled.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, an `(n_frames, 3)` DeepLabCut array of `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` at ~295 Hz. Column 1 (`tongue_y`) is the value; column 2 (likelihood) determines visibility. The bottom-view camera, jaw and nose series are not used.

ii.
```python
ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
timestamps = np.asarray(ts.timestamps[:], dtype=np.float64)
data = np.asarray(ts.data[:], dtype=np.float32)
y = data[:, 1]
likelihood = data[:, 2]
```

iii. Steps 22–23 and 50: the agent enumerated the `BehavioralTimeSeries` members, confirmed the reference code contains no tongue-visibility logic to copy (`rg` over `/app/code` found only unrelated "marker" hits), and read in `/app/methods.txt` that "We trained DeepLabCut to track the movement of tongue, jaw and nose", confirming the third column is a DLC confidence.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) A frame counts as visible if `likelihood >= 0.9`. (2) The 40th and 60th percentiles of `tongue_y` are computed over **all visible frames in the session** (falling back to all frames if none are visible). (3) For each of the 80 bins of each trial, a **single representative frame** is selected — the last frame whose timestamp is `< bin_end`, kept only if that timestamp is also `>= bin_start` — and its y-value and likelihood decide the bin's class. Within-bin averaging is not performed, so ~14 of the ~15 frames in each bin are discarded.

ii.
```python
TONGUE_VISIBLE_THRESHOLD = 0.9
...
visible = likelihood >= likelihood_threshold
if np.any(visible):
    q40, q60 = np.quantile(y[visible], [0.4, 0.6]).astype(np.float32)
else:
    q40, q60 = np.quantile(y, [0.4, 0.6]).astype(np.float32)
```
```python
bin_starts = go_time + BIN_EDGES_S[:-1]
bin_ends = go_time + BIN_EDGES_S[1:]
frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
labels = np.full(N_BINS, 3, dtype=np.int8)
valid = (frame_idx >= 0) & (timestamps[np.clip(frame_idx, 0, len(timestamps) - 1)] >= bin_starts)
```

iii. Step 52: the agent swept the likelihood threshold over `[0.001 … 0.95]` on one session and found the visible fraction essentially flat (0.1034 → 0.0991) and the 40th/60th percentiles stable to ~1 px, i.e. the likelihood is effectively bimodal, and then picked 0.9 as a conservative value. Step 69: "discretize tongue position from session-wide visible-frame percentiles using DLC confidence". The per-bin single-frame sampling is not discussed or justified anywhere in the trajectory.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the four classes the instructions specify, using the per-session 40th/60th percentiles: `0` if `y < q40`, `1` if `q40 <= y <= q60`, `2` if `y > q60`, and `3` ("not visible") for any bin whose representative frame is missing or has `likelihood < 0.9`. Class 3 is the default the label array is initialised to. Result: 84.0% of bins are class 3 (reference: 75.0%); among visible bins the 0/1/2 split is ~39/20/41%.

ii.
```python
labels = np.full(N_BINS, 3, dtype=np.int8)
...
    idx = frame_idx[valid]
    y_valid = y[idx]
    like_valid = likelihood[idx] >= likelihood_threshold
    valid_bins = np.where(valid)[0]
    visible_bins = valid_bins[like_valid]
    if len(visible_bins):
        y_vis = y_valid[like_valid]
        labels[visible_bins[y_vis < q40]] = 0
        mid = (y_vis >= q40) & (y_vis <= q60)
        labels[visible_bins[mid]] = 1
        labels[visible_bins[y_vis > q60]] = 2
```
```python
"output_values": [..., ["lt_40th_percentile", "p40_to_p60", "gt_60th_percentile", "not_visible"]],
```
```python
"tongue_visible_q40": tongue_quantiles[0],
"tongue_visible_q60": tongue_quantiles[1],
```

iii. The class edges and the per-session scope are taken verbatim from the Decoder Task specification ("0: < 40th percentile of y-position over the session … 3: not visible"). The per-session percentiles are also written into `session_info` so the discretisation is auditable.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and events, so the tongue labels are placed on the identical go-cue-relative grid: the bin boundaries are formed as `go_time + BIN_EDGES_S`, exactly the edges used for the spike histogram, and frames are looked up against those absolute boundaries by `searchsorted`. No interpolation or offset correction. Bins that fall outside the video coverage (e.g. before the trial's video started) have no frame and become class 3.

ii.
```python
for go_time in go_times:
    bin_starts = go_time + BIN_EDGES_S[:-1]
    bin_ends = go_time + BIN_EDGES_S[1:]
    frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    ...
    valid = (frame_idx >= 0) & (timestamps[np.clip(frame_idx, 0, len(timestamps) - 1)] >= bin_starts)
```

iii. Not separately discussed; the shared `BIN_EDGES_S` grid is reused for every stream so bin *k* of the tongue output covers the same interval as bin *k* of the firing rates.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several distinct mechanisms:
- **Non-string / missing unit labels**: `get_vector_strings` coerces every entry to a string (`None` → `''`), so the un-annotated session whose `classification` is NaN yields no `'good'` units and the session is dropped and recorded in `skipped_sessions`.
- **`'N/A'` photostim fields**: `maybe_float` returns `None` for `'N/A'`, unparseable values and NaN, and those trials get an all-zero photostim input.
- **Missing spike coverage**: handled by the recorded-trial trim and the all-zero-trial drop (1-e); sessions left with < 2 trials or 0 units are dropped.
- **Missing tone events**: two fallbacks, the last of which synthesises the tone as `go − median(go − tone)` for the session.
- **Missing / low-confidence video frames**: represented as an explicit fourth class (`not visible`) rather than imputed.
- **Anything else**: each session is converted inside a `try/except`; a failure is logged to `skipped_sessions` with the exception text and the run continues.

ii.
```python
def get_vector_strings(table, column):
    raw = table[column].data[:]
    return np.asarray(["" if x is None else str(x) for x in raw], dtype=object)
```
```python
def maybe_float(value):
    text = str(value)
    if text == "N/A":
        return None
    ...
    if math.isnan(out):
        return None
    return out
```
```python
for i in np.where(~valid)[0]:
    ...
    else:
        per_trial[i] = go_times[i] - median_go_minus_sample
```
```python
except Exception as exc:
    skipped_sessions.append({"session_path": str(path), "reason": repr(exc)})
    print(f"  skipped: {exc}", flush=True)
    continue
```
```python
if session_record is None:
    skipped_sessions.append({"session_path": str(path), "reason": "fewer than 2 trials or 0 good units"})
```

iii. Step 111: the agent watched the full export and reported "one session correctly dropped for having no usable good units", i.e. it treated the drop as the intended outcome of the QC rule rather than an error. The general philosophy visible in the trajectory (step 83) is to exclude data that was never recorded rather than emit it as zeros — "the right fix is to drop unrecorded late trials per session, not to keep hundreds of behavior-only trials with zero neural data" — while genuinely-absent measurements (retracted tongue) become an explicit category.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant costs are (1) reading each NWB file — in particular materialising the whole ragged `spike_times` buffer (up to ~11.5 M doubles) and the `(n_frames, 3)` tongue array (~680 k × 3) per session; (2) the per-unit spike-binning loop, whose accumulation step uses `np.add.at`, the slowest NumPy scatter-add path; (3) an extra ragged HDF5 read of `units['is_good_trials']` for *every* good unit (~400 per session) purely to measure its length; and (4) pickling the ~6 GB result. From the trajectory, ~34 of 174 sessions converted in the first 30 s, so the full pass took roughly 2.5 minutes — faster than the expert reference's 247 s, largely because of the float16 payload.

ii.
```python
recorded_trial_counts = [len(np.asarray(units["is_good_trials"][u])) for u in good_unit_indices]
```
```python
for unit_pos, unit_idx in enumerate(good_unit_indices):
    ...
    np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)
```
```python
with args.output.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 65: the agent explicitly considered I/O cost and avoided `units.to_dataframe()` because it "may silently drag in ragged spike-time columns and make the conversion much slower than it needs to be". Step 102 shows it pre-estimating the output size in bytes before committing to the full export, and steps 109–122 show it monitoring throughput during the run ("The full pass is moving faster than expected").

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five per-trial Python loops remain, all of which are vectorizable:
- `get_sample_onsets`: the per-trial `searchsorted` pair could be one vectorised call over all trials.
- `get_first_response_choices`: fully vectorizable (`searchsorted` of `go_times` into the two lick arrays, then a masked comparison).
- `get_photostim_relative_intervals`: a `df.itertuples()` loop where the columns could be converted and masked in bulk.
- `build_tongue_binned_labels`: the per-trial loop over go cues; the frame→bin lookup could be done once for all trials.
- The per-trial assembly loop that fills `trial_input` / `trial_output` and the final loop that slices `counts` into per-trial arrays.

The per-unit loop in `build_neural_trials` cannot be collapsed further (each unit has a ragged spike array), but its `np.add.at` accumulation could be replaced by a flat `np.bincount`, which is typically an order of magnitude faster.

ii.
```python
for i, (go_time, response_end) in enumerate(zip(go_times, response_ends)):
    left_idx = np.searchsorted(left_licks, go_time, side="left")
    right_idx = np.searchsorted(right_licks, go_time, side="left")
```
```python
for i, trial in enumerate(trials_df.itertuples()):
    onset = maybe_float(trial.photostim_onset)
```
```python
for trial_idx in range(n_trials):
    trial_input = np.empty((2, N_BINS), dtype=np.float32)
    trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
```

iii. Step 72: the agent chose the structure deliberately — "I'm editing the converter as a standalone script with explicit helpers for spike binning, event extraction, and tongue discretization so the preprocessing logic is inspectable rather than buried in one long loop." Readability was prioritised over vectorisation, and since these loops run over trials (hundreds) rather than samples (millions), the cost is small relative to I/O.

## 10-c. What processing does the code repeat multiple times?

i. Little is recomputed, but three items are:
- `iter_session_paths(data_dir)` is called a second time at assembly purely to report `n_source_sessions`, re-running the glob over the whole tree.
- `units['is_good_trials'][u]` is read from disk once per good unit (~400 reads/session) and only its length is used; one read would suffice.
- Inside the per-trial tongue loop, `np.clip(frame_idx, 0, len(timestamps) - 1)` and the bin-boundary arrays `go_time + BIN_EDGES_S[...]` are rebuilt for every trial instead of once per session.
- `trials_df` columns are converted with `.astype(str).to_numpy()` separately in the main body and again inside `compute_control_performance`.

Everything else — file open, spike buffer read, tongue array read, tongue percentiles — happens exactly once per session, and the bin grid is built once at module level.

ii.
```python
"n_source_sessions": len(iter_session_paths(data_dir)),
```
```python
recorded_trial_counts = [len(np.asarray(units["is_good_trials"][u])) for u in good_unit_indices]
```
```python
for go_time in go_times:
    bin_starts = go_time + BIN_EDGES_S[:-1]
    bin_ends = go_time + BIN_EDGES_S[1:]
    ...
    valid = (frame_idx >= 0) & (timestamps[np.clip(frame_idx, 0, len(timestamps) - 1)] >= bin_starts)
```

iii. Not discussed in the trajectory. The conversion is otherwise a single streaming pass over the files, which is possible because the tongue percentiles are per-session and can be computed inside that same pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three items, all small:
- `compute_control_performance()` runs a full pass over six trials-table columns per session to compute behavioral performance and left/right hit counts. These end up only in `metadata['session_info']` and are never used by the decoder. (The agent had investigated the paper's behavioral criterion as a possible session filter, rejected it, and kept the statistic as diagnostic metadata.)
- The `units['is_good_trials']` reads (see 10-a/10-c): ~400 ragged array reads per session, of which only the minimum length is retained.
- `metadata['time_bin_centers_s']`, `tongue_visible_q40/q60`, `skipped_sessions` and the full `session_info` list are informational only and are not consumed by the decoder — although `off_start`/`off_end`/`time_bin_size` next to them *are* required by the format.

Also worth noting: `response_ends` (`go_stop_times`) is computed for every session but is only used by the choice derivation, and the neural counts are accumulated for all units × all trials into one `(n_units, n_trials, 80)` array which is then sliced trial by trial — a transient that roughly doubles peak per-session memory.

ii.
```python
def compute_control_performance(trials_df):
    early = trials_df["early_lick"].astype(str).to_numpy() == "early"
    auto = trials_df["auto_water"].to_numpy().astype(int) == 1
    free = trials_df["free_water"].to_numpy().astype(int) == 1
    control = trials_df["photostim_duration"].astype(str).to_numpy() == "N/A"
    ...
    return performance, left_hits, right_hits, n_regular
```
```python
"control_performance_non_early_no_auto_free": performance,
"control_hit_left": left_hits,
"control_hit_right": right_hits,
"control_trials_responded": n_regular,
```
```python
"time_bin_centers_s": BIN_CENTERS_S.astype(float).tolist(),
"n_source_sessions": len(iter_session_paths(data_dir)),
```

iii. Step 58: the agent computed the published behavioral-performance criterion dataset-wide, found "it keeps only 145 of 174 sessions, so that criterion was clearly analysis-specific rather than the dataset-wide inclusion rule", and did not apply it — but retained the statistic per session as provenance. The instructions ask for extra descriptive fields in `metadata`, so the session-level records are intentional rather than accidental.
