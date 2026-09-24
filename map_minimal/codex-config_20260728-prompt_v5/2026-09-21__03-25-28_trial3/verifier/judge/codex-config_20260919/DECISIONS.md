# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds sorted `sub-*/*.nwb` paths under the data directory, opens each with `NWBHDF5IO`, and reads the NWB trials, units, acquisition events, and behavioral time series. Each file is processed as one session.

ii.
```python
session_paths = sorted(Path(data_dir).glob(NWB_GLOB))
for session_idx, path in enumerate(session_paths, start=1):
    session_data = convert_session(path, brain_region_to_idx)
```
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    trials = nwb.trials.to_dataframe()
    units_df = nwb.units.to_dataframe()
```

iii. The trajectory says the agent inspected the available files and NWB schema, concluded that the NWB schema directly exposed the needed trials, units, events, and tracking, and used the standard `pynwb` reader. Sorting provides deterministic traversal.

## 1-b. How are the data split into subjects?

i. The subject is read from `nwb.subject.subject_id`. Unique subjects are registered in first-session encounter order, and each retained session receives the corresponding integer `subject_idx`.

ii.
```python
"subject": str(nwb.subject.subject_id),
```
```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
data["subject_idx"].append(subject_to_idx[subject])
```

iii. The trajectory identified the NWB subject field as the direct subject identifier. No additional inference or parsing was considered necessary.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Retained session data are appended once per file in sorted path order, and the filename stem is stored as the session id in metadata.

ii.
```python
session_paths = sorted(Path(data_dir).glob(NWB_GLOB))
session_data = convert_session(path, brain_region_to_idx)
```
```python
"session_id": path.stem,
"source_file": str(path),
```

iii. The agent observed the dataset layout and described it as one NWB session per file. It expected 173 analyzable sessions after dropping the file with no good units.

## 1-d. How are the data split into trials?

i. Rows of `nwb.trials.to_dataframe()` define trials. The same boolean trial mask is applied positionally to the go-cue array, and retained rows are later converted one at a time into input/output trials.

ii.
```python
trials = nwb.trials.to_dataframe()
go_times = np.asarray(
    nwb.acquisition["BehavioralEvents"].time_series["go_start_times"].timestamps[:],
    dtype=np.float64,
)
go_times_kept = go_times[keep_trials]
```

iii. The trajectory states that the NWB trials and events were rich enough to reproduce trial curation directly. Unlike the reference, the agent did not explicitly assert that trial-row and go-cue counts agree.

## 1-e. How are trials filtered based on quality controls?

i. Trials must match a good unit's `obs_intervals`, must have neither auto-water nor free-water, and must not be all-zero across every retained neuron in the aligned neural window. Sessions with fewer than two trials after filtering are discarded. Early-lick and ignore trials are intentionally retained.

ii.
```python
keep_trials = (
    recorded_trials
    & (trials["auto_water"].to_numpy() == 0)
    & (trials["free_water"].to_numpy() == 0)
)
```
```python
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if nonzero_trial_mask.sum() < 2:
    return None
```

iii. The agent retained early-lick and ignore trials because they are explicit decoder targets. It discovered all-zero trials during verification, traced most to behavioral trials outside ephys `obs_intervals`, and then removed two residual all-zero trials as likely boundary artifacts. It also followed the paper's “regular trial” convention for auto/free-water, although the reference conversion only excludes free-water.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from each retained unit's `spike_times`, with `classification` selecting units and `go_start_times` setting trial-relative bin edges.

ii.
```python
good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    spikes = np.asarray(spike_times, dtype=np.float64)
```

iii. The trajectory examined unit fields and the spike-sorting paper, then chose the published `classification == "good"` verdict rather than inventing metric thresholds.

## 2-b. How is the `neural` data processed?

i. For each good unit, sorted spike times are searched at every absolute trial/bin edge. Adjacent cumulative indices are differenced into counts and divided by 0.05 s to produce Hz. Results are stored as float16 neuron-by-time arrays.

ii.
```python
edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
counts = np.diff(edge_idx, axis=1)
neural[:, unit_idx, :] = (counts / BIN_SIZE_S).astype(np.float16, copy=False)
```

iii. The agent explicitly selected 50-ms firing rates with no smoothing or normalization, matching the task and reference method. Float16 was chosen for the large output's storage footprint.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose NWB `classification` string is `good` are retained. A session with zero good units is skipped. No per-unit quality-metric thresholds or `is_good_trials` stability filter is applied.

ii.
```python
good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
if good_unit_mask.sum() == 0:
    return None
```

iii. The agent investigated `classification`, quality metrics, and `is_good_trials`, and concluded that the white-paper classifier's `good` verdict was the relevant published QC. The single session with no good units was expected to be dropped.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 s are added to each trial's absolute go-cue time. Spikes are binned directly against these absolute edges.

ii.
```python
abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]
edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
```

iii. The agent recognized that NWB spikes and events share a common clock, so adding the go time to a common relative grid is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 50 ms: 80 nonoverlapping bins over `[-2.5, 1.5)` s. Raw spike times are binned into firing rates; no further temporal rebinning, smoothing, or interpolation is applied.

ii.
```python
BIN_SIZE_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
bin_edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S / 2, BIN_SIZE_S)
```

iii. These values directly follow the decoder instructions, and the trajectory repeatedly states that the intended output is go-aligned 50-ms firing-rate bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `delay_start_times`, trial `start_time`/`stop_time`, go-cue times, and the known 0.65-s sample duration. The last delay event within each trial is selected and tone onset is inferred as 0.65 s earlier; if none exists, it falls back to go minus 1.85 s.

ii.
```python
delay_last = last_event_per_trial(delay_times, trials_kept["start_time"].to_numpy(...),
                                  trials_kept["stop_time"].to_numpy(...))
tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
```

iii. The agent correctly noticed that early licks can replay task epochs, so a fixed tone offset is insufficient. It investigated repeated sample/delay events and chose the last event in the trial, though the reference directly uses the last `sample_start_times` event before go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The inferred tone onset is subtracted from each absolute bin center, yielding a continuous seconds-since-tone value at each of the 80 bins.

ii.
```python
time_from_tone = (go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]) - tone_onset[:, None]
```

iii. The trajectory says the final replayed sample is behaviorally relevant and that time from tone should be continuous on the same grid as neural data.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the go-cue-aligned bin centers corresponding to the neural bin intervals.

ii.
```python
abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
time_from_tone = abs_centers - tone_onset[:, None]
```

iii. The agent intentionally reused the shared 80-bin go-cue grid so input and neural columns correspond.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses each trial's `photostim_onset`, `photostim_duration`, and `start_time`, plus the go-cue time for alignment.

ii.
```python
onset = trial["photostim_onset"]
duration = trial["photostim_duration"]
onset_abs = float(trial["start_time"]) + float(onset)
offset_abs = onset_abs + float(duration)
```

iii. The agent found these explicit fields in the NWB trials table and used them instead of reconstructing stimulation from other events.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. `N/A` trials remain all zero. Otherwise a bin is one when its center is at or after stimulation onset and before stimulation offset, and zero elsewhere.

ii.
```python
if onset == "N/A" or duration == "N/A":
    continue
photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) &
                (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
```

iii. This implements the requested time-varying binary “on at every time point” input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute trial-start-relative stimulation times are converted to go-relative onset/offset values and compared with the same relative bin centers used for neural data.

ii.
```python
onset_rel = onset_abs - go_times_kept[i]
offset_rel = offset_abs - go_times_kept[i]
```

iii. The agent chose the common go-relative axis to ensure matching time columns.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial `trial_instruction` and `outcome`: ignore means no lick, hit means the instructed side, and miss means the opposite side.

ii.
```python
def choice_from_instruction_and_outcome(instruction, outcome):
    if outcome == "ignore": return 2
    if instruction == "left": return 0 if outcome == "hit" else 1
    if instruction == "right": return 1 if outcome == "hit" else 0
```

iii. The trajectory recognized that actual direction is not directly stored but is fully determined by instruction and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0 left, 1 right, or 2 no lick, then repeated over all 80 time bins as a per-trial label.

ii.
```python
np.full(len(BIN_CENTERS_REL_S), choice, dtype=np.int8)
```

iii. The agent checked the decoder and chose broadcasting so all outputs can share a time-indexed array while preserving a per-trial target.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials table's `outcome` value.

ii.
```python
outcome = outcome_to_int(str(trial["outcome"]))
```

iii. The raw categories already exactly match the requested output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to 0 ignore, 1 miss, and 2 hit, then repeated across all time bins.

ii.
```python
mapping = {"ignore": 0, "miss": 1, "hit": 2}
np.full(len(BIN_CENTERS_REL_S), outcome, dtype=np.int8)
```

iii. The mapping follows `output_values`, and broadcasting matches the agent's chosen common output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is read directly from the trials table's `early_lick` field.

ii.
```python
early = early_lick_to_int(str(trial["early_lick"]))
```

iii. The agent noted that early licking is an explicit requested decoder target, which is why these trials are retained.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` maps to 1; the value is repeated across all bins.

ii.
```python
mapping = {"no early": 0, "early": 1}
np.full(len(BIN_CENTERS_REL_S), early, dtype=np.int8)
```

iii. This directly realizes the requested no/yes categories in the common time-indexed output format.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking`: column 1 is y-position, column 2 is tracking likelihood, and the series timestamps locate frames in time.

ii.
```python
tracking = np.asarray(tongue_ts.data[:], dtype=np.float32)
timestamps = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
y_all = tracking[:, 1]
likelihood_all = tracking[:, 2]
```

iii. The trajectory inspected the tracking container and its three columns, identifying it as the available side-camera tongue measurement.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood at least 0.9 define visible y values and session percentile thresholds. At each neural bin center, the most recent camera frame is selected; its y and likelihood determine a class, with class 3 used when not visible or when no valid preceding frame exists. There is no averaging of frames within a 50-ms bin.

ii.
```python
visible_all = likelihood_all >= TONGUE_VISIBILITY_THRESHOLD
visible_y = y_all[visible_all]
p40, p60 = np.percentile(visible_y, [40, 60])
frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
```

iii. The agent probed likelihood distributions and selected 0.9 as a visibility threshold. It aimed to create a time-varying output efficiently, but did not justify using one held frame per bin rather than the reference's mean of visible frames in each bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles are calculated over all visible raw-frame y values. Visible samples are classified as 0 for `< p40`, 1 for `p40 <= y <= p60`, and 2 for `> p60`; invisible samples are 3.

ii.
```python
state[visible & (y < p40)] = 0
state[visible & (y >= p40) & (y <= p60)] = 1
state[visible & (y > p60)] = 2
```

iii. The per-session 40/60 split follows the task. The agent used visible raw frames, whereas the reference uses percentiles of session-wide 50-ms visible-frame bin means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each absolute neural bin center (`go + relative center`), the code chooses the latest camera frame at or before that center and assigns its category to that neural column.

ii.
```python
abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
```

iii. The agent relied on shared NWB timestamps and the common go-cue clock. This aligns sample times, but differs from applying the same 50-ms intervals and averaging all camera frames within each interval.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units and sessions with fewer than two usable trials are skipped. Malformed `obs_intervals` default to retaining all trials. Trials with all-zero neural windows are removed. No visible tongue values yield `None` thresholds and all tongue states remain “not visible”; out-of-range camera indices also remain “not visible.” Unexpected categorical strings raise `ValueError`.

ii.
```python
if obs_intervals.ndim != 2 or obs_intervals.shape[1] != 2:
    return np.ones(len(trials), dtype=bool)
```
```python
if visible_y.size == 0:
    p40 = np.nan
    p60 = np.nan
tongue_state = np.full(abs_centers.shape, 3, dtype=np.int8)
```

iii. Verification drove the handling of missing ephys coverage: the agent diagnosed zero trials, added `obs_intervals` filtering, and removed residual all-zero trials. It treated absent tongue measurements as an explicit requested category rather than imputing them.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant work is reading all NWB session data, materializing dataframes and full tongue arrays, iterating over every good unit to search all trial/bin edges, building the very large in-memory result, and pickling it. The full conversion was repeatedly run during debugging.

ii.
```python
tracking = np.asarray(tongue_ts.data[:], dtype=np.float32)
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left")
```

iii. The trajectory reports stable per-session conversion throughput and a multi-gigabyte pickle; it treated NWB reading/neural binning and export as the substantive costs, then separately ran verification and training.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The photostimulation trial loop, `obs_intervals` loop, region-label loop, input/output construction loop, and nonzero-trial list scan could be vectorized. The neural loop is already vectorized across trials/bins but remains per unit because spike arrays are ragged.

ii.
```python
for i, (_, trial) in enumerate(trials_kept.iterrows()):
for obs_start, obs_stop in obs_intervals:
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
```

iii. The agent prioritized correctness and completed within budget. Its spike search deliberately flattens all trial edges per unit, capturing the most important vectorization opportunity.

## 10-c. What processing does the code repeat multiple times?

i. It iterates over retained trials once to build photostimulation and again to build inputs/outputs. Trial masks are applied separately to each derived array. Full dataset conversion was also executed three times during debugging, though a normal invocation performs one pass.

ii.
```python
for i, (_, trial) in enumerate(trials_kept.iterrows()):  # photostim
...
for i, (_, trial) in enumerate(trials_kept.iterrows()):  # inputs/outputs
```

iii. The trajectory shows repetition arose mainly from validation-driven fixes: initial conversion, conversion after `obs_intervals`, and conversion after all-zero filtering. Within the final program, repeated trial iteration favors clarity.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes/stores metadata percentile values and multiple trial counts not used by decoder training, copies filtered trial dataframes, converts brain-region labels through Python loops, and computes neural arrays for trials later discarded solely because the whole aligned window is zero. Broadcasting three per-trial labels across 80 bins also duplicates data, although it suits the common output shape.

ii.
```python
trials_kept = trials.loc[keep_trials].copy()
"tongue_visible_p40": p40,
"tongue_visible_p60": p60,
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. The trajectory does not identify these as problematic; most metadata was added for auditability, and broadcasting was chosen after inspecting the decoder. The all-zero scan was added specifically to eliminate verifier warnings.
