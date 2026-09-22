# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing all NWB files under `sub-*/*.nwb`, sorting them, then opening each file with `NWBHDF5IO`. Within each session file it reads the NWB trial table, unit table, and behavioral acquisitions directly from the NWB object.

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

iii. The trajectory says the AI first inspected the repository and NWB structure so it could “mirror the reference processing” and then treated the dataset as “173 NWB sessions” that could be processed directly from the NWB schema (steps 4, 9, 19).

## 1-b. How are the data split into subjects?

i. Each kept session contributes `str(nwb.subject.subject_id)` as its subject ID. The global `subjects` list is built in first-seen order as sessions are processed, and `subject_idx` stores the matching integer index for each session.

ii.
```python
return {
    "subject": str(nwb.subject.subject_id),
    "neural": neural_trials,
    "input": inputs,
    "output": outputs,
    "brain_region_idx": brain_region_idx,
    "session_info": session_info,
}
```

```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)

data["subject_idx"].append(subject_to_idx[subject])
```

iii. The trajectory treats the NWB subject field as the canonical split by animal and reports final subject counts from that schema, but it does not give further justification for using insertion order rather than a sorted subject list (steps 19, 90, 128).

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Sessions are processed in sorted path order, and session metadata records `path.stem` as the session identifier.

ii.
```python
session_paths = sorted(Path(data_dir).glob(NWB_GLOB))
for session_idx, path in enumerate(session_paths, start=1):
    session_data = convert_session(path, brain_region_to_idx)
```

```python
session_info = {
    "session_id": path.stem,
    "source_file": str(path),
    "subject": str(nwb.subject.subject_id),
    ...
}
```

iii. The trajectory repeatedly refers to the dataset as a collection of NWB session files and later reports “174 files” reduced to “173 sessions” after one session is skipped for zero good units (steps 9, 78, 82, 90, 128).

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table as the trial definition. After building a Boolean `keep_trials` mask, it subsets both the trial dataframe and the per-trial go-cue vector with the same mask, so each remaining row is one trial.

ii.
```python
trials = nwb.trials.to_dataframe()
...
trials_kept = trials.loc[keep_trials].copy()
...
go_times = np.asarray(
    nwb.acquisition["BehavioralEvents"].time_series["go_start_times"].timestamps[:],
    dtype=np.float64,
)
go_times_kept = go_times[keep_trials]
```

iii. The trajectory says the NWB “trials already expose” the needed behavioral fields and that the NWB “trials and events are rich enough to reproduce the paper’s session/trial curation directly” (steps 19, 24).

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they fall inside the first good unit’s `obs_intervals`, are not `auto_water`, and are not `free_water`. After neural binning, the AI applies a second filter that drops any trial whose full neural matrix is all zeros, and it drops sessions with fewer than two surviving trials.

ii.
```python
recorded_trials = recorded_trial_mask_from_obs_intervals(trials, first_good_unit["obs_intervals"])
keep_trials = (
    recorded_trials
    & (trials["auto_water"].to_numpy() == 0)
    & (trials["free_water"].to_numpy() == 0)
)
trials_kept = trials.loc[keep_trials].copy()
if len(trials_kept) < 2:
    return None
```

```python
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if nonzero_trial_mask.sum() < 2:
    return None

trials_kept = trials_kept.iloc[nonzero_trial_mask].copy()
```

iii. The trajectory initially planned to exclude only `auto_water` and `free_water` while retaining early-lick and ignore trials because those are decoder targets (steps 29, 67, 128). After the verifier exposed all-zero neural trials, the AI justified intersecting with `obs_intervals` and then explicitly trimming remaining all-zero terminal trials as neural-coverage cleanup (steps 95, 100, 102, 105, 110, 112).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from each good unit’s `spike_times` and each trial’s go-cue timestamp. Only units with `classification == "good"` are used.

ii.
```python
good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
...
go_times = np.asarray(
    nwb.acquisition["BehavioralEvents"].time_series["go_start_times"].timestamps[:],
    dtype=np.float64,
)
...
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    spikes = np.asarray(spike_times, dtype=np.float64)
```

iii. The trajectory says the AI would use “`classification == "good"` units” and “go-aligned 50 ms firing-rate bins” from the NWB schema and the published QC label (steps 62, 67, 69, 128).

## 2-b. How is the `neural` data processed?

i. For each good unit, the AI bins spikes into 50 ms windows around the go cue using `np.searchsorted`, converts per-bin spike counts to firing rates by dividing by `BIN_SIZE_S`, stores the result as `float16`, and returns one `(n_neurons, n_timepoints)` matrix per trial.

ii.
```python
abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]

for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    spikes = np.asarray(spike_times, dtype=np.float64)
    edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
    counts = np.diff(edge_idx, axis=1)
    neural[:, unit_idx, :] = (counts / BIN_SIZE_S).astype(np.float16, copy=False)
```

iii. The trajectory says the intended representation was “go-aligned 50 ms firing-rate bins over `[-2.5, 1.5)`” and later describes the verifier passing once these binned neural matrices were filtered for actual recording coverage (steps 69, 78, 102, 128).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is based on the NWB unit-level `classification` field only: the AI keeps units whose classification string is `"good"`. It does not apply `is_good_trials` as a hard mask, and it drops any session with zero such units.

ii.
```python
good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
if good_unit_mask.sum() == 0:
    return None
```

iii. The trajectory explicitly says the “published session-level unit QC is the `classification == "good"` label” and that `is_good_trials` was used only as a sanity check, not as a hard exclusion rule (step 62).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset. The AI constructs trial-specific absolute bin edges by adding the fixed relative window `[-2.5, 1.5]` s to each kept trial’s go time, then bins spikes against those edges.

ii.
```python
BIN_EDGES_REL_S, BIN_CENTERS_REL_S = make_time_grid()
...
abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]
```

iii. The trajectory repeatedly states that the conversion should be “go-cue aligned” and later summarizes the final alignment/binning exactly that way (steps 14, 67, 69, 128).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins over a 4 s window from -2.5 s to +1.5 s relative to the go cue, giving 80 bins per trial. No additional temporal rebinning or smoothing is applied.

ii.
```python
BIN_SIZE_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
...
bin_edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S / 2, BIN_SIZE_S)
bin_centers = bin_edges[:-1] + BIN_SIZE_S / 2
```

iii. The trajectory explicitly planned and later summarized “50 ms bins” and the `[-2.5, 1.5)` go-cue-aligned window (steps 67, 69, 128).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives tone onset from `delay_start_times`, plus trial start/stop boundaries and go-cue times. It finds the last delay start inside each trial and subtracts `0.65` s to infer the tone onset; if no delay event is found it falls back to `go_time - 1.85`.

ii.
```python
delay_times = np.asarray(
    nwb.acquisition["BehavioralEvents"].time_series["delay_start_times"].timestamps[:],
    dtype=np.float64,
)
delay_last = last_event_per_trial(
    delay_times,
    trials_kept["start_time"].to_numpy(dtype=np.float64),
    trials_kept["stop_time"].to_numpy(dtype=np.float64),
)
tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
```

iii. The trajectory says the AI first considered deriving tone from the “final valid pre-go task structure” and concluded that the final sample onset is usually `1.85 s` before the go cue, but then hard-coded the delay-based derivation unless that event is missing (steps 48, 53, 67).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After inferring one tone onset per trial, the AI computes absolute bin-center times for each go-aligned trial and subtracts the inferred tone onset to obtain a continuous time-from-tone vector for every bin in that trial.

ii.
```python
tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
time_from_tone = (go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]) - tone_onset[:, None]
```

iii. The trajectory says the bins are go aligned and the tone input should be a continuous, time-varying quantity built on that grid rather than a per-trial scalar (steps 43, 53, 67).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The tone-time input uses the same bin centers as the neural data. For each trial, the absolute neural bin centers are computed from the same go-aligned grid, and the tone input is simply the absolute bin-center time minus the inferred tone onset.

ii.
```python
abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
...
time_from_tone = (go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]) - tone_onset[:, None]
```

iii. The trajectory explicitly checked the decoder so that per-trial outputs and time-varying inputs would share a common aligned time axis, then kept the tone input on the same 80-bin go-cue-aligned grid as the neural data (step 43).

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from trial-table columns `photostim_onset`, `photostim_duration`, and `start_time`, together with each trial’s go-cue time.

ii.
```python
onset = trial["photostim_onset"]
duration = trial["photostim_duration"]
...
onset_abs = float(trial["start_time"]) + float(onset)
offset_abs = onset_abs + float(duration)
onset_rel = onset_abs - go_times_kept[i]
offset_rel = offset_abs - go_times_kept[i]
```

iii. The trajectory says the AI checked “photostim timing” in the NWB trial table before writing the converter and then summarized the final input as binary “photostim-on” on the aligned time grid (steps 53, 54, 69, 128).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts onset and duration from per-trial metadata into an on/off time series. For each trial it marks bins whose centers fall within `[onset_rel, offset_rel)` as `1` and leaves all other bins at `0`.

ii.
```python
photostim = np.zeros((len(trials_kept), len(BIN_CENTERS_REL_S)), dtype=np.float32)
for i, (_, trial) in enumerate(trials_kept.iterrows()):
    onset = trial["photostim_onset"]
    duration = trial["photostim_duration"]
    if onset == "N/A" or duration == "N/A":
        continue
    ...
    photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) & (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
```

iii. The trajectory frames this as a time-varying decoder input rather than a per-trial flag and says the single-session sanity checks confirmed photostim landed in the expected late-delay bins (steps 67, 73, 78).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation onset and offset are converted to times relative to each trial’s go cue and compared against the same relative bin centers used for the neural matrices.

ii.
```python
onset_rel = onset_abs - go_times_kept[i]
offset_rel = offset_abs - go_times_kept[i]
photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) & (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
```

iii. The trajectory says the implementation uses go-cue alignment throughout and verified that photostim occupied the expected bins after conversion (steps 67, 73, 78).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not taken from a single raw column. The AI derives it from `trial_instruction` and `outcome`.

ii.
```python
def choice_from_instruction_and_outcome(instruction, outcome):
    if outcome == "ignore":
        return 2
    if instruction == "left":
        return 0 if outcome == "hit" else 1
    if instruction == "right":
        return 1 if outcome == "hit" else 0
```

iii. The trajectory explicitly checked the decoder’s representation expectations and then chose to broadcast per-trial categorical variables, including derived choice, across aligned bins (steps 43, 67, 128).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps trials to three classes: `0` for left, `1` for right, and `2` for no lick on `ignore` trials. That per-trial class is repeated across all 80 bins in output row 0.

ii.
```python
choice = choice_from_instruction_and_outcome(
    str(trial["trial_instruction"]),
    str(trial["outcome"]),
)
...
np.full(len(BIN_CENTERS_REL_S), choice, dtype=np.int8)
```

iii. The trajectory says that if the task spec makes a variable per-trial, the cleanest implementation is to broadcast it across bins so all outputs share one time axis (step 43).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` column.

ii.
```python
outcome = outcome_to_int(str(trial["outcome"]))
```

iii. The trajectory treats outcome as an already available trial-level NWB field that can be converted into decoder labels directly (steps 19, 67).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats that category across all bins in output row 1.

ii.
```python
def outcome_to_int(outcome):
    mapping = {"ignore": 0, "miss": 1, "hit": 2}
    ...
```

```python
outcome = outcome_to_int(str(trial["outcome"]))
...
np.full(len(BIN_CENTERS_REL_S), outcome, dtype=np.int8)
```

iii. The trajectory says the converter should express these requested decoder outputs as categorical labels on the shared aligned grid (steps 43, 67, 128).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trials-table `early_lick` column.

ii.
```python
early = early_lick_to_int(str(trial["early_lick"]))
```

iii. The trajectory explicitly calls out retaining early-lick trials because early lick itself is one of the requested decoder outputs (steps 29, 67, 128).

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats that class across all 80 bins in output row 2.

ii.
```python
def early_lick_to_int(value):
    mapping = {"no early": 0, "early": 1}
    ...
```

```python
early = early_lick_to_int(str(trial["early_lick"]))
...
np.full(len(BIN_CENTERS_REL_S), early, dtype=np.int8)
```

iii. The trajectory says per-trial categorical outputs should be broadcast across the aligned bins while preserving early-lick trials in the dataset (steps 43, 67, 128).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the tracking data columns for y position and DeepLabCut likelihood together with their timestamps.

ii.
```python
tracking = np.asarray(tongue_ts.data[:], dtype=np.float32)
timestamps = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
y_all = tracking[:, 1]
likelihood_all = tracking[:, 2]
```

iii. The trajectory says the AI was “probing the behavior containers” to recover “tongue visibility” from the tracking stream and later decided to treat visibility through the DeepLabCut confidence channel (steps 19, 53, 69).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first marks frames as visible when `likelihood >= 0.9`. It computes the 40th and 60th percentiles over all visible raw `y` values from the session. Then, for each neural bin center in each trial, it finds the most recent camera frame at or before that bin center and assigns a tongue state from that single frame.

ii.
```python
TONGUE_VISIBILITY_THRESHOLD = 0.9
...
visible_all = likelihood_all >= TONGUE_VISIBILITY_THRESHOLD
visible_y = y_all[visible_all]
...
p40, p60 = np.percentile(visible_y, [40, 60])
```

```python
abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
...
y = y_all[frame_idx[valid]]
likelihood = likelihood_all[frame_idx[valid]]
visible = likelihood >= TONGUE_VISIBILITY_THRESHOLD
```

iii. The trajectory says the remaining implementation choice was to “treat tongue visibility from the DeepLabCut confidence channel rather than raw coordinates,” and the final summary states the converter used “DLC confidence `>= 0.9` for visibility and per-session visible-frame `y` percentiles” (steps 53, 67, 128).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible frames are categorized as `0` if `y < p40`, `1` if `p40 <= y <= p60`, and `2` if `y > p60`, where `p40` and `p60` are percentiles of all visible raw `y` values in the session. Non-visible bins remain class `3`.

ii.
```python
tongue_state = np.full(abs_centers.shape, 3, dtype=np.int8)
...
if visible_y.size > 0:
    state[visible & (y < p40)] = 0
    state[visible & (y >= p40) & (y <= p60)] = 1
    state[visible & (y > p60)] = 2
tongue_state[valid] = state
```

iii. The trajectory justifies the categorization as a session-specific percentile discretization after thresholding visibility by confidence, but it does not mention averaging within 50 ms bins before applying percentiles (steps 53, 67, 128).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue output is aligned by evaluating the camera stream at each neural bin center. For each trial and each go-cue-aligned bin center, the AI finds the latest camera frame at or before that time and uses that frame’s y/likelihood to assign the tongue class.

ii.
```python
abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
frame_idx = frame_idx.reshape(abs_centers.shape)
```

iii. The trajectory says the AI wanted tongue alignment to share the same aligned grid as neural activity, but implemented that by sampling the tracking stream at the neural bin centers rather than bin-averaging frames over the full 50 ms interval (steps 43, 69, 128).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several failure cases by filtering or assigning fallback values. A session with zero `classification == "good"` units is dropped. Trials outside `obs_intervals`, `auto_water`, `free_water`, and later all-zero neural trials are dropped. Missing tone-event cases fall back to `go_time - 1.85`. Tongue bins default to class `3` (“not visible”), and if no visible tongue frames exist then `p40`/`p60` are returned as `None`.

ii.
```python
good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
if good_unit_mask.sum() == 0:
    return None
```

```python
if obs_intervals.ndim != 2 or obs_intervals.shape[1] != 2:
    return np.ones(len(trials), dtype=bool)
```

```python
tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
...
tongue_state = np.full(abs_centers.shape, 3, dtype=np.int8)
```

iii. The trajectory explicitly describes dropping the single zero-good-unit session, trimming trials to the recording intervals when verifier checks found all-zero neural trials, and using confidence-thresholded tongue visibility as the basis for the “not visible” class (steps 38, 78, 95, 100, 102, 112, 128).

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work in the AI code is opening every NWB file, reading full trial/unit/tracking arrays, and looping over every good unit to bin spikes with `np.searchsorted`. The trajectory also shows that full-dataset conversion and repeated rebuilds dominated runtime.

ii.
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

```python
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    spikes = np.asarray(spike_times, dtype=np.float64)
    edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
```

iii. The trajectory notes that the AI ran a “full-dataset NWB pass,” then a full conversion, then multiple rebuilds after verifier findings, which implies session I/O and neural binning were the dominant costs (steps 38, 78, 82, 102, 112).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops remain and could have been vectorized further: iterating over `obs_intervals` to build `recorded_trials`, iterating over trials with `iterrows()` for photostim, iterating over units for neural binning, and iterating over trials again to build `inputs` and `outputs`.

ii.
```python
for obs_start, obs_stop in obs_intervals:
    keep |= np.isclose(trial_starts, obs_start, atol=atol) & np.isclose(trial_stops, obs_stop, atol=atol)
```

```python
for i, (_, trial) in enumerate(trials_kept.iterrows()):
    ...
    photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) & (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
```

```python
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    ...
```

iii. The trajectory does not discuss vectorization tradeoffs directly, but it does show the AI prioritized getting a structurally valid converter working and then fixing verifier issues, rather than optimizing these loops away (steps 67, 71, 78, 95, 102, 112).

## 10-c. What processing does the code repeat multiple times?

i. Inside the converter, the code repeats several per-trial operations in separate loops: it iterates once to build photostim, again to build per-trial input/output arrays, and separately scans every neural trial afterward to detect all-zero trials. Outside the file itself, the trajectory shows the AI reran the entire conversion multiple times after patching trial filters.

ii.
```python
photostim = np.zeros((len(trials_kept), len(BIN_CENTERS_REL_S)), dtype=np.float32)
for i, (_, trial) in enumerate(trials_kept.iterrows()):
    ...
```

```python
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
...
for i, (_, trial) in enumerate(trials_kept.iterrows()):
    ...
    inputs.append(input_trial)
    outputs.append(output_trial)
```

iii. The trajectory confirms the code path was rerun repeatedly during debugging: original export, rebuild after the `obs_intervals` fix, and a second rebuild after trimming residual zero-neural trials (steps 100, 102, 110, 112).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores some metadata that is not needed by the downstream decoder itself, including `source_file`, `n_trials_total`, `n_trials_recorded`, `n_trials_nonzero_neural`, and the tongue percentile values `p40`/`p60`. It also builds detailed laminar brain-region labels instead of coarser region names, which increases label cardinality without helping the decoder inputs/outputs.

ii.
```python
session_info = {
    "session_id": path.stem,
    "source_file": str(path),
    "subject": str(nwb.subject.subject_id),
    "n_trials_total": int(len(trials)),
    "n_trials_recorded": int(recorded_trials.sum()),
    "n_trials_kept": int(len(trials_kept)),
    "n_trials_nonzero_neural": int(nonzero_trial_mask.sum()),
    "n_good_units": int(good_unit_mask.sum()),
    "tongue_visible_p40": p40,
    "tongue_visible_p60": p60,
}
```

```python
anno_names = units_df.loc[good_unit_mask, "anno_name"].astype(str).str.strip().replace("", "Unknown")
```

iii. The trajectory frames these additions as diagnostics and sanity-checking rather than core decoder requirements; they were added while the AI was debugging coverage and tongue discretization choices (steps 62, 69, 95, 100, 128).
