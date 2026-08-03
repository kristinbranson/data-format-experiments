# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all sessions by globbing `"/app/data/sub-*/*.nwb"` and sorting the results. It reads NWB files directly with `h5py` rather than `pynwb`, first filtering candidate files to those with at least one `units/classification == "good"` unit, then reopening each kept file and reading the needed HDF5 groups: `units`, `intervals/trials`, `acquisition/BehavioralEvents`, and `acquisition/BehavioralTimeSeries`.

ii.
```python
def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
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
```

```python
with h5py.File(path, "r") as f:
    classification = decode_strings(f["units"]["classification"])
    ...
    trials = f["intervals"]["trials"]
    events = f["acquisition"]["BehavioralEvents"]
    tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md`, the AI says it used direct HDF5 reads "for speed and lower overhead" and filtered to the 173 sessions with at least one classifier-good unit because that matched the paper's reported session count better than keeping all 174 files.

## 1-b. How are the data split into subjects?

i. The AI reads each session's subject from `general/subject/subject_id`, converts it to a string, prepends `"sub-"`, and then builds `subjects` and `subject_idx` in first-seen session order rather than sorting unique IDs up front.

ii.
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
```

```python
subjects: list[str] = []
subject_to_idx: dict[str, int] = {}
...
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
subject_idx.append(subject_to_idx[result.subject_id])
```

iii. The notes say `subject.subject_id` is the source field and that session order follows the sorted NWB file list. There is no separate rationale for the `"sub-"` prefix beyond mirroring the directory naming.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. It uses the filtered, sorted file list as session order and derives `session_id` from the filename stem by stripping `_behavior+ecephys(+ogen)` suffixes.

ii.
```python
def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
```

iii. The notes explicitly say "Base session set: Use the 173 NWB sessions with at least one classifier-labeled good unit" and "Session order follows sorted NWB file list after excluding the zero-good-unit session."

## 1-d. How are the data split into trials?

i. The AI iterates row-by-row through the NWB `intervals/trials` table. For each trial row, it finds the go cue by taking the last `go_start_times` timestamp that falls inside `[trial.start_time, trial.stop_time]`, and treats that as the trial's alignment event and effective trial instance.

ii.
```python
trial_start = trials["start_time"][:].astype(np.float64)
trial_stop = trials["stop_time"][:].astype(np.float64)
...
for trial_idx in range(len(trial_start)):
    start = trial_start[trial_idx]
    stop = trial_stop[trial_idx]
    go_candidates = interval_values(go_times, start, stop)
    if len(go_candidates) == 0:
        raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
    go_time = float(go_candidates[-1])
```

iii. In the notes, the AI justifies this as safer than assuming row-wise alignment: "Use event-in-trial matching rather than row order assumptions."

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in two stages. First, it drops any trial whose full go-aligned `[-2.5, 1.5] s` window is not fully contained in the first good unit's `obs_intervals`. Second, after binning spikes, it drops any trial whose entire neural tensor is all zeros. It does not explicitly filter `free_water` trials.

ii.
```python
obs_intervals = f["units"]["obs_intervals"][:].astype(np.float64)
...
session_obs_intervals = obs_intervals[obs_start:obs_stop]
...
covered = np.any(
    (window_start >= session_obs_intervals[:, 0])
    & (window_end <= session_obs_intervals[:, 1])
)
if not covered:
    n_trials_dropped_outside_obs += 1
    continue
```

```python
nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
n_trials_dropped_all_zero = int((~nonzero_trial_mask).sum())
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
    input_tensor = input_tensor[nonzero_trial_mask]
    output_tensor = output_tensor[nonzero_trial_mask]
```

iii. The AI's notes say this filter was introduced after the verifier found "zero activity across every neuron" on some trials, and that it used observation-interval coverage plus a final all-zero drop to remove those invalid windows. It also states this reduces the mean kept trials/session below the paper's number because the decoder task requires a complete aligned neural window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` for units whose `units/classification` is `"good"`, together with per-trial go cue times from `BehavioralEvents/go_start_times`.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
...
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
...
go_times = events["go_start_times"]["timestamps"][:].astype(np.float64)
```

iii. The notes explicitly map "`units.spike_times` for units with `classification == 'good'`" to `neural` and describe `go_start_times` as the alignment event used to place bin edges.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times into per-trial, per-unit firing rates by building one `trial_edge_matrix` of absolute bin edges, counting spikes in each bin with `np.searchsorted`, taking bin-wise differences, and dividing by the 50 ms bin width. It stores the final rates as `float16`.

ii.
```python
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
...
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. The notes say the AI chose firing rates rather than counts because the task asked for 50 ms bins "for computing firing rates", and that vectorizing across all trials per unit was a deliberate speedup.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are kept. Sessions with zero such units are excluded before processing.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

iii. The notes justify this as the local equivalent of the classifier-based QC used by the paper, and specifically reject `units.unit_quality == "good"` as too permissive.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the go cue. For each trial, it finds `go_time` within the trial interval, adds the relative bin edges `[-2.5, 1.5]` s to that absolute time, and bins spikes against those absolute edges.

ii.
```python
go_candidates = interval_values(go_times, start, stop)
go_time = float(go_candidates[-1])
...
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. The notes say this preserves the reference "time to go" convention and uses go-cue alignment because the decoder instructions required it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms non-overlapping bins over `[-2.5, 1.5]` s relative to go, giving 80 bins per trial. It does not do additional temporal rebinning after that.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_WIDTH_S / 2.0
N_BINS = len(BIN_CENTERS_REL)
```

iii. The notes state this was an intentional deviation from the paper's 40 ms / 3.4 ms reference preprocessing to satisfy the task-specific decoder format.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, the trial's `start_time`, and the per-trial `go_time`. The AI uses the last sample-start event between trial start and the go cue.

ii.
```python
sample_start_times = events["sample_start_times"]["timestamps"][:].astype(np.float64)
...
sample_candidates = interval_values(sample_start_times, start, go_time)
...
sample_onset = float(sample_candidates[-1])
```

iii. The notes justify using the last sample event because early-lick replay trials can contain repeated sample epochs, and the last one before go is the effective tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes absolute bin centers around the go cue, subtracts the selected tone/sample onset, and stores the result as a length-80 time series. If no sample-start event is found within `[trial.start, go]`, it falls back to `go_time - 1.85`.

ii.
```python
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
else:
    sample_onset = float(sample_candidates[-1])

bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. The notes describe the main logic as "last `sample_start` before the go cue." Later notes say no trial actually required the `go - 1.85 s` fallback (`n_missing_sample_onset_fallback = 0`), so the fallback was precautionary rather than exercised on the final dataset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by using the same go-aligned bin centers as the neural data. For each trial, `time_from_tone` is evaluated exactly at `go_time + BIN_CENTERS_REL`.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. The notes describe this as constructing tone timing from the same raw event streams and on the same go-aligned grid used for neural binning.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `trials.photostim_onset`, `trials.photostim_duration`, `trials.photostim_power`, and `trials.start_time`, then compares the resulting absolute interval to the go-aligned bin centers.

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

iii. The notes say the intended transform is from trial-relative onset/duration to a go-aligned binary series, and note that trials with `photostim_power == N/A` are treated as unstimulated.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts the trial-relative onset and duration into an absolute stimulation interval `[start + onset, start + onset + duration)`, then marks each 50 ms bin center as 1 if it falls inside that interval and 0 otherwise.

ii.
```python
photostim_row = np.zeros(N_BINS, dtype=np.float32)
...
if stim_power is not None and stim_onset is not None and stim_dur is not None:
    stim_start_abs = start + stim_onset
    stim_stop_abs = stim_start_abs + stim_dur
    photostim_row = (
        (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
    ).astype(np.float32)
```

iii. The notes describe `photostim_on` as a binary, time-varying input built from the same raw timing fields used in the reference preprocessing.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by evaluating stimulation on the same absolute bin-center timestamps used for the neural bins: `go_time + BIN_CENTERS_REL`.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
...
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
```

iii. The notes say the photostimulation epochs are "constructed from the same raw event streams" and placed on the go-aligned neural time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. For `hit` and `miss` trials, the AI derives choice from `trials.trial_instruction` and `trials.outcome`. For `ignore` trials, where it judged choice to be missing in the source, it additionally looks at `BehavioralEvents/left_lick_times` and `right_lick_times` inside the full trial interval and, if still unresolved, falls back to the instructed side.

ii.
```python
def infer_choice(
    instruction: str,
    outcome: str,
    trial_start: float,
    trial_stop: float,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> tuple[int, str]:
    if outcome == "hit":
        return CHOICE_MAP[instruction], "instruction+outcome"
    if outcome == "miss":
        opposite = "right" if instruction == "left" else "left"
        return CHOICE_MAP[opposite], "instruction+outcome"
```

```python
left_trial = interval_values(left_lick_times, trial_start, trial_stop)
right_trial = interval_values(right_lick_times, trial_start, trial_stop)
...
return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. The notes explicitly call this "the main task-specific edge case": the AI says ignore trials do not have an explicit choice label, so it uses earliest lick side in the trial when present, otherwise the instructed side, and documents the resulting label noise.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI codes only two choice classes, `left=0` and `right=1`, repeats the per-trial choice across all 80 bins, and records provenance statistics on whether the choice came from `instruction+outcome` or an ignore-trial fallback path.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
```

```python
choice_code, choice_source = infer_choice(...)
choice_sources[choice_source] += 1
...
output_row = np.empty((4, N_BINS), dtype=np.int16)
output_row[0, :] = choice_code
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["<40th_pct", "40th_to_60th_pct", ">60th_pct"],
],
```

iii. The notes justify the two-class output as matching the user-facing task specification, while acknowledging that 13,258 ignore trials required fallback assignment.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table column `trials.outcome`.

ii.
```python
trial_outcome = decode_strings(trials["outcome"])
...
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. The notes say this variable is taken directly from the raw trial table and matches the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats that single per-trial value across all bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
output_row[1, :] = outcome_code
```

iii. The notes say this exactly matches the task specification and that outcome is repeated across bins so all outputs share a common `(n_output, n_timepoints)` shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes directly from the trials-table column `trials.early_lick`.

ii.
```python
trial_early = decode_strings(trials["early_lick"])
...
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. The notes describe this as a direct mapping from the trial table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats that code across all 80 bins.

ii.
```python
EARLY_MAP = {"no early": 0, "early": 1}
...
output_row[2, :] = early_code
```

iii. The notes say this follows the requested coding and uses the same repeated-across-bins output shape as other trial-level outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue position from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the `y` coordinate in column 1 and the tracking likelihood in column 2, together with the corresponding camera timestamps.

ii.
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
...
y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
```

iii. The notes identify the side-camera tongue stream as the directly available source variable and say no other marker reconstruction was needed because the target output only required tongue y-position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI preprocesses the full-session tongue trace before trial extraction. It computes frame-to-frame velocity in `(x, y)`, marks frames above a 5-sigma velocity threshold as outliers, marks frames with likelihood below `0.1` as low-confidence, fills low-confidence frames with the session mean y-value, linearly interpolates outliers and non-finite values, and then computes session-level 40th and 60th percentiles on the fully processed per-frame y trace.

ii.
```python
LIKELIHOOD_THRESHOLD = 0.1
...
velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
...
vel_threshold = vel_mean + 5.0 * vel_std
...
low_likelihood_mask = likelihood < LIKELIHOOD_THRESHOLD
...
processed_y[low_likelihood_mask] = session_mean_y
...
processed_y[interp_mask] = np.interp(
    tongue_timestamps[interp_mask],
    tongue_timestamps[keep_mask],
    processed_y[keep_mask],
)
...
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
```

iii. The notes say this follows the method paper's description of low-likelihood mean fill and five-sigma velocity outlier handling, and that the AI intentionally resampled from the processed continuous trace at bin centers.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses the session-wide 40th and 60th percentiles of the processed per-frame tongue y trace as thresholds. Each bin gets class 0 by default, class 1 if `y >= p40`, and class 2 if `y > p60`. There is no explicit "not visible" or missing-data class.

ii.
```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
...
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["<40th_pct", "40th_to_60th_pct", ">60th_pct"],
],
```

iii. The notes describe this as "session-level 40th and 60th percentiles into classes `0/1/2`" and note that the conservative preprocessing produced strong middle-bin dominance.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output by sampling the processed continuous tongue trace at the closest preceding camera frame for each neural bin center, rather than averaging all tongue frames within the 50 ms bin.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
...
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. The notes explicitly justify this as closer to the reference marker-alignment logic than "averaging over long windows."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses a mix of exclusion and imputation. It excludes sessions with no classifier-good units, excludes trials whose full aligned window lies outside the recorded observation intervals, and drops fully zero neural trials. For missing or low-confidence behavioral data, it imputes: missing tone onset would fall back to `go - 1.85 s`; ignore-trial choice is assigned heuristically from lick times or instruction; low-confidence tongue frames are filled with the session mean and outliers are linearly interpolated.

ii.
```python
if n_good_units == 0:
    return None
...
if not covered:
    n_trials_dropped_outside_obs += 1
    continue
...
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
```

```python
if len(left_trial) and len(right_trial):
    ...
return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

```python
processed_y[low_likelihood_mask] = session_mean_y
...
processed_y[interp_mask] = np.interp(...)
```

iii. The notes frame the trial exclusions as necessary to avoid invalid all-zero neural windows, and frame the ignore-choice and tongue imputations as task-driven compromises for variables that lacked a clean label or were partially missing.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are opening and scanning all NWB files, reading and splitting the ragged spike-time buffers, the per-trial trial-building loop, the per-unit `searchsorted` neural binning loop, and optionally loading and preprocessing full tongue-tracking arrays. The script also makes an extra full-dataset pass through `select_files`.

ii.
```python
for path in all_files:
    with h5py.File(path, "r") as f:
        classification = decode_strings(f["units"]["classification"])
```

```python
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
```

```python
for trial_idx in range(len(trial_start)):
    ...
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
```

iii. The notes emphasize direct HDF5 reads, vectorized per-unit neural binning, and sessionwise processing as speed-focused choices, which implies those areas were expected to dominate runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several Python-level loops that could be reduced further: the trial-construction loop over all trials, the second loop that copies `trial_records` into dense tensors, the per-unit neural loop, the region-label indexing loop, and the file-filtering loop in `select_files`. The AI only partially vectorized the neural step by vectorizing across trials within each unit.

ii.
```python
for trial_idx in range(len(trial_start)):
    ...
for keep_idx, rec in enumerate(trial_records):
    ...
for unit_idx, spikes in enumerate(good_spike_times):
    ...
for i, label in enumerate(result.brain_region_labels):
    ...
for path in all_files:
    with h5py.File(path, "r") as f:
```

iii. The notes explicitly call out vectorizing neural binning across all trials at once as a speedup, which implies the remaining loops were left as pragmatic rather than fully optimized choices.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work. It opens every NWB file once in `select_files` and again in `process_session`, rereads `units/classification` in both places, calls `select_files(all_files, sample_mode=False)` again at the end just to estimate full-runtime, and materializes ragged spike times for all units before subsetting to good units.

ii.
```python
session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)
...
for i, path in enumerate(session_files, start=1):
    result = process_session(path, make_plot=make_plot)
...
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

```python
classification = decode_strings(f["units"]["classification"])
...
classification = decode_strings(f["units"]["classification"])
```

```python
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
```

iii. The notes do not present these as intentional, but the code structure shows they are repeated for convenience or bookkeeping rather than required by the target format.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does extra work for diagnostics and plotting that is not used in the final converted dataset: it computes `choice_sources`, `tongue_info`, per-session timing stats, and optional `plot_payload`; it reads `photostim_power` only to gate stimulation trials but does not store it; it computes velocity and interpolation state for tongue preprocessing even though only the final categorical tongue bins are emitted; and it builds metadata fields such as excluded sessions and fallback counters that are not downstream decoder features.

ii.
```python
choice_sources = Counter()
...
stats = {
    "n_trials": n_trials,
    "n_good_units": n_good_units,
    "choice_sources": dict(choice_sources),
    "n_missing_sample_onset_fallback": n_missing_sample_onset_fallback,
    "n_trials_dropped_outside_obs": n_trials_dropped_outside_obs,
    "n_trials_dropped_all_zero": n_trials_dropped_all_zero,
    "tongue_info": tongue_info,
    "stim_trials": int(np.sum(input_tensor[:, 1, :].any(axis=1))),
    "session_seconds": float(time.time() - session_start),
}
```

```python
if make_plot:
    plot_payload = {
        "tongue_y_raw": tongue_data[:, 1].astype(np.float32),
        "tongue_y_processed": processed_tongue_y.copy(),
        ...
        "neural_trial": neural_tensor[trial_idx].astype(np.float32),
    }
```

```python
stim_power = parse_optional_float(trial_photostim_power[trial_idx])
```

iii. The notes explicitly mention `--show-processing`, session statistics, and detailed sanity checking, so these extra computations were deliberate observability features rather than part of the core converted representation.
