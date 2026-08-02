# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `/app/data/sub-*/*.nwb`, opens each NWB file with `h5py`, and reads trial tables from `intervals/trials`, event times from `acquisition/BehavioralEvents`, tongue tracking from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, unit metadata from `units`, and spikes from the ragged `units/spike_times` arrays.

ii. 
```python
def convert_dataset(data_dir, exclude_auto_free=True):
    data_dir = Path(data_dir)
    nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
    ...
    for session_number, path in enumerate(nwb_paths, start=1):
        with h5py.File(path, "r") as f:
            trials = f["intervals/trials"]
            ...
            be = f["acquisition/BehavioralEvents"]
            ...
            tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
            ...
            units = f["units"]
```

iii. `CONVERSION_NOTES.md` says the input files are all NWB files under `/app/data/sub-*/*.nwb`. The trajectory shows the agent deliberately mapped the paper/code concepts onto the local NWB release rather than the original MATLAB export.

## 1-b. How are the data split into subjects?

i. Subjects are split by NWB `general/subject/subject_id`. The converter builds a unique `subjects` list and stores one `subject_idx` per kept session.

ii. 
```python
subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
subject = str(subject)
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The agent notes report 28 subjects and describe subject IDs as coming from NWB subject metadata.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Sessions with no `classification == "good"` units are skipped, and sessions that end up with fewer than two kept trials after filtering are also skipped.

ii. 
```python
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        ...
        good_unit_indices = np.flatnonzero(classification == "good")

        if len(good_unit_indices) == 0:
            continue
        ...
        if n_keep_trials < 2:
            continue

        neural_sessions.append([np.ascontiguousarray(session_neural[i]) for i in range(n_keep_trials)])
```

iii. `CONVERSION_NOTES.md` explicitly says the release has 174 NWB files and that one zero-good-unit file was dropped, leaving 173 sessions.

## 1-d. How are the data split into trials?

i. Trials are defined by the rows of `intervals/trials`. Trial-level arrays such as `start_time`, `stop_time`, `trial_instruction`, `outcome`, `early_lick`, and photostim fields are read row-wise, and go cues are required to have the same count as trials.

ii. 
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])

trial_starts = np.asarray(trials["start_time"], dtype=np.float64)
trial_stops = np.asarray(trials["stop_time"], dtype=np.float64)
trial_instruction = _decode_str_array(trials["trial_instruction"]).reshape(-1)
outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
if len(go_starts) != n_trials:
    raise RuntimeError(...)
```

iii. The notes say the “base trial table comes from `intervals/trials`,” and the trajectory shows the agent verified `go_start_times` counts against trial counts before implementing the converter.

## 1-e. How are trials filtered based on quality controls?

i. The converter excludes `auto_water` and `free_water` trials, intersects the remaining trials with a “common good ephys” mask derived from `units/is_good_trials` and `units/obs_intervals`, and then drops any residual all-zero neural trials. It keeps early-lick, ignore, and photostimulation trials.

ii. 
```python
trial_keep = np.ones(n_trials, dtype=bool)
if exclude_auto_free:
    trial_keep &= auto_water == 0
    trial_keep &= free_water == 0
...
is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
...
recorded_trial_mask[mapped_idx[common_obs_mask]] = True
...
trial_keep &= recorded_trial_mask
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
session_input = [session_input[i] for i in np.flatnonzero(nonzero_trial_mask)]
session_output = [session_output[i] for i in np.flatnonzero(nonzero_trial_mask)]
```

iii. `CONVERSION_NOTES.md` says exactly this and the trajectory shows the agent added `is_good_trials` and then `obs_intervals` after finding late-session all-zero trials. The agent justified keeping early/ignore/photostim trials because the decoder specification asked for those variables.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` for units whose `units/classification` is `"good"`. Region labels come from `units/anno_name`, and trial-validity support comes from `units/is_good_trials` and `units/obs_intervals`.

ii. 
```python
units = f["units"]
classification = _decode_str_array(units["classification"]).reshape(-1)
anno_name = _decode_str_array(units["anno_name"]).reshape(-1)
good_unit_indices = np.flatnonzero(classification == "good")
...
unit_spikes = _extract_unit_spikes(units, int(unit_idx))
```

iii. The notes say “keep units where `units/classification == "good"`” and cite the paper’s classifier-based QC as the basis.

## 2-b. How is the `neural` data processed?

i. For each kept unit, absolute spike times are counted in non-overlapping 50 ms bins spanning `[-2.5, +1.5]` seconds around each trial’s go cue, then divided by `0.05` to convert counts to firing rates in Hz.

ii. 
```python
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
...
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
...
spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
session_neural[:, pos, :] = spike_counts
```

iii. `CONVERSION_NOTES.md` states that neural data are firing rates in Hz from non-overlapping 50 ms spike-count bins. The trajectory shows this was a deliberate adaptation to the decoder instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps only classifier-good units, requires kept trials to lie in the common per-session good-ephys support, drops sessions with zero good units, and removes residual all-zero neural trials. It does not apply the paper’s behavioral session-selection criteria.

ii. 
```python
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    continue
...
trial_keep &= recorded_trial_mask
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
...
if n_keep_trials < 2:
    continue
```

iii. The notes emphasize classifier-good units and common ephys-valid trials. The trajectory shows the common-trial mask was added specifically to remove trials outside valid recording intervals.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to go cue onset from `acquisition/BehavioralEvents/go_start_times/timestamps`; bin edges and centers are defined relative to each trial’s go time.

ii. 
```python
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
...
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
```

iii. The notes say the alignment event is go cue onset and that the go cue count matches the trial count in each converted session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, yielding 80 bins per trial over a 4 s window. The code does not first recreate the paper’s 3.4 ms stride/40 ms smoothing pipeline; it directly bins spikes into the requested 50 ms bins.

ii. 
```python
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
N_BINS = len(BIN_CENTERS_REL)
```

iii. `CONVERSION_NOTES.md` states “80 bins of width 50 ms.” The trajectory repeatedly refers to “50 ms non-overlapping go-aligned bins” as a task-driven choice.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from trial `start_time`, go cue timestamps, and `sample_start_times` from `BehavioralEvents`; the inferred tone onset for each trial is the relevant sample-start timestamp.

ii. 
```python
trial_starts = np.asarray(trials["start_time"], dtype=np.float64)
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
sample_starts = np.asarray(be["sample_start_times"]["timestamps"], dtype=np.float64)
tone_onsets = _resolve_sample_onsets(trial_starts, go_starts, sample_starts)
```

iii. The notes say the requested tone input is resolved from `sample_start_times` because the NWB files can include replayed sample epochs after early licks.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code resolves one sample onset per trial by taking the last `sample_start_times` event before that trial’s go cue, with a fallback constrained to the trial’s own `[start_time, go_time]` interval. It then computes “time since tone onset” at every neural bin center.

ii. 
```python
def _resolve_sample_onsets(trial_starts, go_starts, sample_starts):
    sample_idx = np.searchsorted(sample_starts, go_starts, side="right") - 1
    ...
    invalid = np.isnan(sample_onsets) | (sample_onsets < (trial_starts - 1e-9))
    if np.any(invalid):
        for i in np.where(invalid)[0]:
            mask = (sample_starts >= (trial_starts[i] - 1e-9)) & (sample_starts <= (go_starts[i] + 1e-9))
            ...
            sample_onsets[i] = candidates[-1]
```
```python
tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. The notes and trajectory both justify this as a repair for replayed sample epochs in NWB. This was an inferred conversion decision rather than something copied directly from the paper code.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled at the same 80 go-aligned bin centers used for neural firing rates, so each trial’s `time_from_tone_onset_s` vector is one-to-one with the neural time axis.

ii. 
```python
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
...
input_trial = np.vstack([time_from_tone, stim_on]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says this input is “computed at each bin center.”

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the trial table fields `photostim_onset` and `photostim_duration`, with `start_time` used to convert onset offsets into absolute timestamps.

ii. 
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
...
stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
```

iii. The notes say trial photostim timing uses `intervals/trials/photostim_onset` and `photostim_duration`, interpreted relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. String placeholders like `"N/A"` are converted to `NaN`. For trials with finite onset and duration, the code marks a bin as `1` if the bin center falls inside the absolute photostim interval; otherwise it stays `0`.

ii. 
```python
def _parse_optional_float_array(dataset):
    ...
    if value in ("N/A", "", None):
        continue
    out[i] = float(value)
```
```python
stim_on = np.zeros(N_BINS, dtype=np.float32)
if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
    ...
    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says `photostimulation_on` is 1 when the bin center falls inside the photostim interval.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned on the same go-centered bin centers used for `neural`. The code compares the photostim interval against `abs_centers`, which are the neural bin centers expressed in absolute time.

ii. 
```python
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. The notes describe this as a bin-center-aligned binary channel.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The script derives `choice` from `intervals/trials/trial_instruction`, not from actual lick events or lick-direction behavior variables.

ii. 
```python
trial_instruction = _decode_str_array(trials["trial_instruction"]).reshape(-1)
...
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
```

iii. `CONVERSION_NOTES.md` says this was chosen to match the paper code’s `trial_type` label and to preserve ignore trials. The trajectory repeats this rationale.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps `left -> 0` and `right -> 1` from `trial_instruction` and then broadcasts that per-trial value across all 80 time bins.

ii. 
```python
CHOICE_MAP = {"left": 0, "right": 1}
...
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
...
np.full(N_BINS, choice_value, dtype=np.int16)
```

iii. The notes explicitly document the map and justify it as using the paper’s left/right trial label instead of a realized choice variable.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `Outcome` is derived directly from `intervals/trials/outcome`.

ii. 
```python
outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
...
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
```

iii. The notes say outcome is mapped directly from the NWB outcome field.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats that categorical label across all time bins in the trial.

ii. 
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` gives exactly this mapping.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. No such output is created. The script only creates `choice`, `outcome`, `early_lick`, and `tongue_y_position`. For `outcome`, the chosen implementation is to broadcast a single per-trial label across all neural time bins.

ii. 
```python
"output_names": ["choice", "outcome", "early_lick", "tongue_y_position"],
...
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_value, dtype=np.int16),
        np.full(N_BINS, outcome_value, dtype=np.int16),
        np.full(N_BINS, early_value, dtype=np.int16),
        tongue_cat,
    ]
)
```

iii. The notes only discuss `outcome`, not distance to reward zone. This question appears to be a template typo relative to the actual task.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. `Early lick` is derived from `intervals/trials/early_lick`.

ii. 
```python
early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
...
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
```

iii. The notes say it is mapped directly from the NWB `early_lick` field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `no early -> 0` and `early -> 1`, then repeats the result across all 80 bins for the trial.

ii. 
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
...
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` documents the same mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from the side-camera tongue-tracking time series `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the second coordinate column `data[:, 1]` plus its timestamps.

ii. 
```python
tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
```

iii. The notes say side-camera tongue tracking comes from `Camera0_side_TongueTracking`.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each go-aligned 50 ms bin, the code samples the most recent tongue y value whose timestamp falls within that bin. Bins without a valid sample remain zero.

ii. 
```python
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    ...
    sampled = np.zeros(N_BINS, dtype=np.float32)
    if np.any(valid):
        sampled[valid] = y_values[idx[valid]].astype(np.float32)
    return sampled
```

iii. The notes say the reference alignment code uses the most recent marker sample within each time window rather than interpolation.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The script computes the 40th and 60th percentiles from the full-session tongue y distribution, then assigns category 0 below the 40th percentile, category 1 from the 40th percentile up to and including the 60th percentile, and category 2 above the 60th percentile.

ii. 
```python
tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
...
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
```

iii. `CONVERSION_NOTES.md` describes the same 40/60 percentile discretization over the full session.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue categories are produced on exactly the same 80 go-aligned 50 ms bins as the neural data, because the raw tongue signal is first reduced to one sampled y value per neural bin and then discretized.

ii. 
```python
tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
...
output_trial = np.vstack(
    [
        ...,
        tongue_cat,
    ]
)
```

iii. The notes say tongue y is sampled “for each 50 ms neural bin.”

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles string placeholders like `"N/A"` by converting them to `NaN`; resolves ambiguous tone onsets with a fallback constrained to the trial interval; zero-fills tongue bins with no in-bin sample; falls back to a simpler `is_good_trials` mapping if `obs_intervals` do not map cleanly; and removes residual all-zero neural trials.

ii. 
```python
if value in ("N/A", "", None):
    continue
```
```python
if np.any(invalid):
    for i in np.where(invalid)[0]:
        ...
        if len(candidates) == 0:
            raise RuntimeError(...)
        sample_onsets[i] = candidates[-1]
```
```python
sampled = np.zeros(N_BINS, dtype=np.float32)
```
```python
if np.sum(valid) == n_recorded_trials:
    recorded_trial_mask[mapped_idx[common_obs_mask]] = True
else:
    recorded_trial_mask[:n_recorded_trials] = common_obs_mask
```

iii. The trajectory documents these as reactive fixes after validation found edge cases, especially for trial coverage and all-zero trials.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is opening 174 NWB files, extracting ragged spike arrays unit-by-unit, binning spikes across all kept trials, and then looping over kept trials again to build time-varying input/output arrays and tongue-bin samples.

ii. 
```python
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        ...
        for pos, unit_idx in enumerate(good_unit_indices):
            unit_spikes = _extract_unit_spikes(units, int(unit_idx))
            spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
            ...
        for local_idx, trial_idx in enumerate(keep_trial_indices):
            ...
            tongue_y_trial = _bin_tongue_y(...)
```

iii. The trajectory shows the agent repeatedly reran full conversion and had to optimize only by improving correctness, which implies these session/unit/trial loops dominate runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop over `good_unit_indices`, the per-trial loop that builds each `input_trial`/`output_trial`, the Python loop in `_parse_optional_float_array`, and the fallback loop over invalid tone onsets could all be vectorized or partially vectorized.

ii. 
```python
for i, value in enumerate(raw):
    ...
for pos, unit_idx in enumerate(good_unit_indices):
    ...
for local_idx, trial_idx in enumerate(keep_trial_indices):
    ...
for i in np.where(invalid)[0]:
    ...
```

iii. The converter is written in a clear session-by-session imperative style; the notes do not claim any runtime optimization.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly recomputes per-session absolute bin edges/centers, repeatedly searches through spike times for each unit, repeatedly samples tongue traces bin-by-bin for each trial, and repeatedly constructs constant-across-time output rows with `np.full`.

ii. 
```python
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
...
tongue_y_trial = _bin_tongue_y(...)
...
np.full(N_BINS, choice_value, dtype=np.int16)
```

iii. This follows directly from the code structure; there is no caching beyond per-session arrays.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `photostim_power` but never uses it; computes per-session summary statistics that are only written to JSON; writes sample-data artifacts unrelated to the final full dataset; and removes all-zero trials only after fully constructing their neural/input/output arrays. It also broadcasts per-trial labels across all 80 bins even though those labels are constant within trial.

ii. 
```python
photostim_power = _parse_optional_float_array(trials["photostim_power"])
```
```python
summary["per_session"].append(
    {
        ...,
        "mean_trial_duration_s": float(np.mean(...)),
        "mean_tone_onset_rel_go_s": float(np.mean(...)),
    }
)
```
```python
sample_data = _subset_data(data, sample_sessions, args.sample_trials_per_session)
```
```python
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
```

iii. The notes emphasize reporting and reproducibility outputs, but those computations are not used by `train_decoder.py` itself.
