# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script glob-loads every NWB file under `/app/data/sub-*/*.nwb`, sorts them, filters out sessions with zero `classification == "good"` units, then opens each remaining NWB file with `h5py`. Within each session it reads unit tables, trial tables, behavioral event timestamps, and side-camera tongue tracking data.

ii. 
```python
def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))

def select_files(all_files: list[Path], sample_mode: bool) -> tuple[list[Path], list[str]]:
    selected: list[Path] = []
    excluded: list[str] = []
    for path in all_files:
        with h5py.File(path, "r") as f:
            classification = decode_strings(f["units"]["classification"])
            if np.sum(classification == "good") == 0:
                excluded.append(path.name)
                continue
        selected.append(path)

def process_session(path: Path, make_plot: bool = False) -> SessionResult | None:
    with h5py.File(path, "r") as f:
        classification = decode_strings(f["units"]["classification"])
        anno_name = decode_strings(f["units"]["anno_name"])
        spike_times_flat = f["units"]["spike_times"][:]
        trials = f["intervals"]["trials"]
        events = f["acquisition"]["BehavioralEvents"]
        tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md`, the agent says the local dataset is a DANDI/NWB archive with 174 session files and that direct `h5py` reads are faster than PyNWB. In Step 4/5 it justifies excluding the single zero-good-unit session so the loaded session count matches the paper’s 173 analyzed sessions.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `general/subject/subject_id` inside each NWB file. The code prefixes that value with `sub-`, de-duplicates subjects while building the final dataset, and stores a per-session `subject_idx`.

ii. 
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"

for result in results:
    if result.subject_id not in subject_to_idx:
        subject_to_idx[result.subject_id] = len(subjects)
        subjects.append(result.subject_id)
    subject_idx.append(subject_to_idx[result.subject_id])
```

iii. `CONVERSION_NOTES.md` Step 5 says subject metadata should come directly from the NWB subject block and that session order should follow the sorted NWB file list after exclusions.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session ID is derived from the filename stem, with `_behavior+ecephys+ogen` or `_behavior+ecephys` removed. Sessions with zero classifier-good units are excluded before conversion.

ii. 
```python
session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)

if np.sum(classification == "good") == 0:
    excluded.append(path.name)
    continue
```

iii. In Step 4 of `CONVERSION_NOTES.md`, the agent explicitly resolves the 174-vs-173 discrepancy by excluding `sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`, because all its unit classifications are effectively missing and it has zero local good units.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each row, the code reads `start_time` and `stop_time`, finds the go cue event that falls within that interval, and constructs one output trial record per kept row.

ii. 
```python
trials = f["intervals"]["trials"]
trial_start = trials["start_time"][:].astype(np.float64)
trial_stop = trials["stop_time"][:].astype(np.float64)

for trial_idx in range(len(trial_start)):
    start = trial_start[trial_idx]
    stop = trial_stop[trial_idx]
    go_candidates = interval_values(go_times, start, stop)
    if len(go_candidates) == 0:
        raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
    go_time = float(go_candidates[-1])
    ...
    trial_records.append({...})
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent says it will use event-in-trial matching rather than row-order assumptions, because the reference code aligns behavioral variables to go cue on a per-trial basis.

## 1-e. How are trials filtered based on quality controls?

i. The script does not apply the paper’s downstream “regular trial” mask. It keeps early-lick, ignore, and photostim trials because those variables are decoder targets/inputs. It only drops trials whose full `[-2.5, 1.5] s` window is not covered by the unit observation intervals, and after binning it drops trials whose neural tensor is entirely zero.

ii. 
```python
covered = np.any(
    (window_start >= session_obs_intervals[:, 0])
    & (window_end <= session_obs_intervals[:, 1])
)
if not covered:
    n_trials_dropped_outside_obs += 1
    continue

nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
n_trials_dropped_all_zero = int((~nonzero_trial_mask).sum())
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
    input_tensor = input_tensor[nonzero_trial_mask]
    output_tensor = output_tensor[nonzero_trial_mask]
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 4, the agent notes that the reference repo’s `get_regular_trial_mask` excludes early lick, ignore, auto/free water, and stimulation trials, but argues those exclusions are analysis-specific and conflict with this task because early lick, outcome, and photostimulation must be preserved.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` and `units/spike_times_index` for units whose `units/classification` equals `"good"`. `units/obs_intervals` and `units/obs_intervals_index` are also used to decide whether each trial window is covered.

ii. 
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"

spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]

obs_intervals = f["units"]["obs_intervals"][:].astype(np.float64)
obs_intervals_index = f["units"]["obs_intervals_index"][:]
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `units.spike_times` from classifier-good units directly onto `neural`, and Step 4 says `classification == "good"` is the closest local equivalent to the paper’s external QC good-unit lists.

## 2-b. How is the `neural` data processed?

i. For each trial, the script creates 50 ms bins from 2.5 s before to 1.5 s after go cue. For each good unit, it counts spikes in each bin using `np.searchsorted`, divides by `0.05` to convert to firing rate in Hz, and stores the result as `float16`.

ii. 
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)

trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL

neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. `CONVERSION_NOTES.md` Step 5 says the reference pipeline bins spikes into firing rates, but the final bin width/window are intentionally changed to 50 ms and `[-2.5, +1.5] s` because the decoder task explicitly requires that deviation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are kept. Sessions with no such units are removed entirely. The code does not additionally use `unit_quality`, `is_good_trials`, or the reference repo’s external good-unit index files.

ii. 
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
...
if np.sum(classification == "good") == 0:
    excluded.append(path.name)
    continue
```

iii. The main justification is in `CONVERSION_NOTES.md` Step 4: `unit_quality == "good"` gives about 155k units and badly misses the white paper’s 69,943 good-unit count, whereas `classification == "good"` yields 69,453 units across 173 sessions and is much closer to the classifier-based QC described in the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to go cue onset. The code finds a go cue timestamp inside the trial interval, uses that as time zero, and bins spikes over `go_time + [-2.5, 1.5] s`.

ii. 
```python
go_candidates = interval_values(go_times, start, stop)
if len(go_candidates) == 0:
    raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
go_time = float(go_candidates[-1])
window_start = go_time + WINDOW_START_S
window_end = go_time + WINDOW_END_S
...
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. Step 3 and Step 5 of `CONVERSION_NOTES.md` say both the papers and reference code are go-cue aligned, and the task instructions also explicitly require go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, yielding 80 bins across the 4 s window. There is no finer intermediate representation followed by rebinning; the script bins directly at 50 ms.

ii. 
```python
BIN_WIDTH_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_WIDTH_S / 2.0
N_BINS = len(BIN_CENTERS_REL)
...
metadata = {
    ...
    "time_bin_size": 50.0,
```

iii. The agent documents this as an intentional task-driven deviation from the method-paper code, which used 40 ms bins and 3.4 ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps`, plus the per-trial go cue time used to create aligned bin centers.

ii. 
```python
sample_start_times = events["sample_start_times"]["timestamps"][:].astype(np.float64)
...
sample_candidates = interval_values(sample_start_times, start, go_time)
...
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the agent interpreted the last `sample_start` before go cue as the effective tone/sample onset, because that is the final replay leading into the observed go cue on early-lick trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the script chooses the last `sample_start` event between trial start and go cue. If none exists, it falls back to `go_time - 1.85`, based on the canonical sample-plus-delay duration. It then subtracts that onset from every go-aligned bin center to produce a continuous time-from-tone vector.

ii. 
```python
sample_candidates = interval_values(sample_start_times, start, go_time)
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
else:
    sample_onset = float(sample_candidates[-1])

bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. The rationale is in trajectory step 126 and `CONVERSION_NOTES.md` Step 5: early-lick trials can replay the sample epoch, so the first sample event is not always the right one. The 1.85 s fallback comes from the methods summary: 0.65 s sample epoch plus 1.2 s delay.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same per-trial 50 ms bin centers as the neural firing rates, using `bin_centers_abs = go_time + BIN_CENTERS_REL`.

ii. 
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
...
input_tensor[keep_idx, 0, :] = rec["time_from_tone"]
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. `CONVERSION_NOTES.md` Step 5 says all decoder inputs should be placed on the same trial/time grid as the neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from `intervals/trials/photostim_onset`, `photostim_duration`, and `photostim_power`, together with `trial.start_time` and the go-aligned bin centers.

ii. 
```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
...
stim_onset = parse_optional_float(trial_photostim_onset[trial_idx])
stim_dur = parse_optional_float(trial_photostim_duration[trial_idx])
stim_power = parse_optional_float(trial_photostim_power[trial_idx])
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent maps the NWB trial-table photostim fields onto a binary per-bin photostim input, noting that the reference `.mat` code used a `task_stimulation` table with power and on/off times relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code parses optional numeric strings, treats missing values (`N/A`, `nan`, etc.) as no stimulation, converts the trial-relative onset/duration to absolute timestamps, then marks each 50 ms bin center as 1 if it falls inside the photostim interval and 0 otherwise.

ii. 
```python
def parse_optional_float(value: str) -> float | None:
    if value in {"N/A", "", "nan", "None"}:
        return None
    return float(value)

photostim_row = np.zeros(N_BINS, dtype=np.float32)
if stim_power is not None and stim_onset is not None and stim_dur is not None:
    stim_start_abs = start + stim_onset
    stim_stop_abs = stim_start_abs + stim_dur
    photostim_row = (
        (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
    ).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says this is the target-format analogue of the reference code’s practice of subtracting go-cue time from stimulation on/off times.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned on the same go-cue-centered 50 ms bin-center grid used for neural firing rates.

ii. 
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
...
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
...
input_tensor[keep_idx, 1, :] = rec["photostim_on"]
```

iii. In Step 5, the agent says all task and behavioral streams should be projected onto the neural time axis, preserving go-cue alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived primarily from `intervals/trials/trial_instruction` and `intervals/trials/outcome`. For ignore trials, the fallback uses `BehavioralEvents/left_lick_times` and `right_lick_times`; if no lick is found, it falls back again to the instructed side.

ii. 
```python
trial_instruction = decode_strings(trials["trial_instruction"])
trial_outcome = decode_strings(trials["outcome"])
left_lick_times = events["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = events["right_lick_times"]["timestamps"][:].astype(np.float64)

choice_code, choice_source = infer_choice(
    instruction=trial_instruction[trial_idx],
    outcome=trial_outcome[trial_idx],
    trial_start=start,
    trial_stop=stop,
    left_lick_times=left_lick_times,
    right_lick_times=right_lick_times,
)
```

iii. `CONVERSION_NOTES.md` Step 5 and trajectory step 130 say that hit/miss choice is recoverable from instruction plus outcome, but ignore trials do not carry an explicit choice label in the source data, so a fallback label had to be invented for decoder compatibility.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Hits are mapped to the instructed side; misses are mapped to the opposite side. For ignore trials, the earliest left/right lick within the trial decides choice if one exists; otherwise the code uses the instructed side as a placeholder. The final choice code is then repeated across all 80 bins.

ii. 
```python
def infer_choice(...):
    if outcome == "hit":
        return CHOICE_MAP[instruction], "instruction+outcome"
    if outcome == "miss":
        opposite = "right" if instruction == "left" else "left"
        return CHOICE_MAP[opposite], "instruction+outcome"
    ...
    return CHOICE_MAP[instruction], "ignore:instruction_fallback"

output_row[0, :] = choice_code
```

iii. The justification is explicit in `CONVERSION_NOTES.md`: this is a task-driven compromise because the reference analyses usually exclude ignore trials rather than assigning them a choice label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `intervals/trials/outcome`.

ii. 
```python
trial_outcome = decode_strings(trials["outcome"])
...
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. `CONVERSION_NOTES.md` Step 5 says this output is a direct raw-to-target mapping and follows the user-specified categorical order.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String labels are mapped as `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeated across all time bins for the trial.

ii. 
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
...
output_row[1, :] = outcome_code
```

iii. The notes state that this exactly matches the decoder specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from `intervals/trials/early_lick`.

ii. 
```python
trial_early = decode_strings(trials["early_lick"])
...
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. `CONVERSION_NOTES.md` Step 5 says this is another direct trial-table mapping.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are mapped as `no early -> 0` and `early -> 1`, then repeated across all 80 bins of the trial.

ii. 
```python
EARLY_MAP = {"no early": 0, "early": 1}
...
early_code = EARLY_MAP[trial_early[trial_idx]]
...
output_row[2, :] = early_code
```

iii. The notes say this matches the requested output coding exactly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and its `timestamps`. The code uses column 1 as `y` and column 2 as the tracking likelihood.

ii. 
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
processed_tongue_y, tongue_info = process_tongue_trace(tongue_data, tongue_timestamps)
...
y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
```

iii. `CONVERSION_NOTES.md` Step 2/5 says the local NWB files provide side-view tongue tracking directly, which is sufficient because the target output only asks for tongue y-position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The script masks low-likelihood frames, detects frame-to-frame velocity outliers using a 5-sigma rule, fills low-likelihood frames with the session mean tongue y, linearly interpolates outlier/non-finite samples from neighboring valid points, and returns the processed continuous y trace.

ii. 
```python
LIKELIHOOD_THRESHOLD = 0.1
...
velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
vel_threshold = vel_mean + 5.0 * vel_std
...
low_likelihood_mask = likelihood < LIKELIHOOD_THRESHOLD
...
processed_y[low_likelihood_mask] = session_mean_y
interp_mask = outlier_mask | ~np.isfinite(processed_y)
if np.any(interp_mask):
    processed_y[interp_mask] = np.interp(
        tongue_timestamps[interp_mask],
        tongue_timestamps[keep_mask],
        processed_y[keep_mask],
    )
```

iii. Step 3 and Step 5 of `CONVERSION_NOTES.md` cite the methods description that marker outliers are removed with a five-sigma velocity threshold and occluded tongue positions are set to the mean value.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After processing the continuous tongue y trace for a session, the code computes the 40th and 60th percentiles over the full processed session trace. Per-bin values are then assigned class 0 if below `p40`, class 1 if between `p40` and `p60`, and class 2 if above `p60`.

ii. 
```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
...
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
...
output_row[3, :] = tongue_bin
```

iii. This follows the user’s decoder specification exactly, and `CONVERSION_NOTES.md` Step 5 explicitly planned that same discretization.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each neural bin center, the code chooses the closest preceding tongue-tracking frame (`searchsorted(..., side="right") - 1`) and uses that processed y value to assign the tongue category. So tongue is sampled on the neural bin-center grid rather than averaged over the whole 50 ms interval.

ii. 
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
...
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent says this is meant to mirror the reference marker-alignment logic, which also uses the last frame within each small time step.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses several ad hoc fallbacks. String datasets are decoded with a backup path; `N/A` photostim entries become `None`; low-likelihood tongue frames are filled with the session mean; velocity outliers and non-finite tongue samples are interpolated; if no valid tongue frame exists the trace is flattened to a mean or zero; missing sample-onset events fall back to `go - 1.85 s`; ignore-trial missing choices fall back to instructed side; and missing go cues or empty region annotations raise errors instead of being repaired.

ii. 
```python
def decode_strings(dataset: h5py.Dataset) -> np.ndarray:
    try:
        return dataset.asstr()[:]
    except Exception:
        ...

def parse_optional_float(value: str) -> float | None:
    if value in {"N/A", "", "nan", "None"}:
        return None

if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85

if not np.any(good_mask):
    session_mean_y = float(np.nanmean(y[np.isfinite(y)]))
    if not math.isfinite(session_mean_y):
        session_mean_y = 0.0

return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. The agent documents most of these in `CONVERSION_NOTES.md` Step 5 and metadata notes, especially the sample-onset fallback, ignore-trial choice placeholder, and tongue preprocessing rules.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant costs are session-by-session NWB I/O and per-unit spike binning across all retained trials. Tongue preprocessing is cheap by comparison, and the code’s own notes identify neural binning as the main bottleneck that was optimized.

ii. 
```python
for i, path in enumerate(session_files, start=1):
    result = process_session(path, make_plot=make_plot)
...
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. In `CONVERSION_NOTES.md` Step 6, the agent says “Per-trial Python loops over neurons would be too slow” and highlights vectorized neural binning as the key speedup.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python loops that could still be vectorized are the trial loop that builds `trial_records`, the loop that copies each record into `trial_edge_matrix` / `input_tensor` / `output_tensor`, the loop over sessions in `build_dataset`, and the per-unit spike-binning loop if spike times were packed differently. The script already vectorized binning across all trials for one unit at a time.

ii. 
```python
for trial_idx in range(len(trial_start)):
    ...
    trial_records.append({...})

for keep_idx, rec in enumerate(trial_records):
    trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
    input_tensor[keep_idx, 0, :] = rec["time_from_tone"]
    input_tensor[keep_idx, 1, :] = rec["photostim_on"]
    output_tensor[keep_idx] = rec["output"]

for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
```

iii. The notes say the main neuron-by-trial nested loop was intentionally avoided, but they do not claim the rest of the script is fully vectorized.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches sorted event arrays trial by trial (`interval_values` for go and sample events, ignore-trial lick checks), repeatedly rebuilds per-trial dictionaries before packing them into arrays, and calls `select_files(all_files, sample_mode=False)` again at the end only to estimate runtime. It also converts packed arrays back into Python lists of trials after having already built dense tensors.

ii. 
```python
go_candidates = interval_values(go_times, start, stop)
sample_candidates = interval_values(sample_start_times, start, go_time)
...
trial_records.append({...})
...
neural_trials = [neural_tensor[i] for i in range(n_trials)]
input_trials = [input_tensor[i] for i in range(n_trials)]
output_trials = [output_tensor[i] for i in range(n_trials)]
...
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

iii. These are not called out in the notes as bugs, but they are visible in the implementation and are the main repeated work that is not essential to the final data representation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds dense trial tensors and then immediately converts them back to Python trial lists for pickling; repeats constant per-trial labels (`choice`, `outcome`, `early_lick`) across all 80 bins even though they are not time-varying; optionally constructs plot payloads and PNGs that are not part of downstream decoding; and computes `go_per_trial` / `sample_on_per_trial` arrays mainly for plotting and stats.

ii. 
```python
output_row = np.empty((4, N_BINS), dtype=np.int16)
output_row[0, :] = choice_code
output_row[1, :] = outcome_code
output_row[2, :] = early_code
...
neural_trials = [neural_tensor[i] for i in range(n_trials)]
input_trials = [input_tensor[i] for i in range(n_trials)]
output_trials = [output_tensor[i] for i in range(n_trials)]

if make_plot:
    plot_payload = {...}
```

iii. `CONVERSION_NOTES.md` Step 5 says constant outputs were intentionally expanded to `(4, 80)` “to satisfy the decoder format cleanly,” so this redundancy was a deliberate convenience choice rather than an accidental bug.
