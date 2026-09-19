# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script loads all session files by globbing `/app/data/sub-*/*.nwb`, then opens each NWB file directly with `h5py`. Within each file it reads the trials table, behavioral events, tongue tracking time series, units table, and subject metadata from fixed HDF5 paths.

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

iii. In the trajectory and `CONVERSION_NOTES.md`, the agent justifies this as the complete NWB release layout: “Input files are all NWB files under `/app/data/sub-*/*.nwb`,” and it treats one file as one session to process.

## 1-b. How are the data split into subjects?

i. Subjects are read from `general/subject/subject_id` in each NWB file. The script builds `subjects` in first-seen file order and stores `subject_idx` per session from a dictionary mapping.

ii.
```python
subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
subject = str(subject)
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The trajectory notes say the canonical mouse identifier is the NWB `subject_id`, so the agent reused that field directly rather than inferring subjects from filenames or session ids.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. A session is included only if it has at least one `classification == "good"` unit and at least two kept trials after filtering.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
if len(good_unit_indices) == 0:
    continue
...
if n_keep_trials < 2:
    continue
```

iii. `CONVERSION_NOTES.md` states that the release contains 174 NWB files and that one file with zero good units is dropped, leaving 173 sessions. The agent therefore used file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table, one row per trial, and aligned to the corresponding `go_start_times` event sequence. The script checks that the number of go cues equals the number of trial rows, then applies a boolean `trial_keep` mask.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
if len(go_starts) != n_trials:
    raise RuntimeError(f"{path.name}: expected {n_trials} go cues, found {len(go_starts)}")
...
keep_trial_indices = np.flatnonzero(trial_keep)
```

iii. In the trajectory the agent describes the alignment event as the go cue and explicitly calls out that “Go cue count matches trial count in each converted session.”

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes `auto_water == 1` and `free_water == 1` trials, then further restricts trials to those with “common good-ephys support” using both `units/is_good_trials` across all good units and `obs_intervals` from the first good unit. After neural binning it removes any remaining trials whose neural array is entirely zero. Sessions with fewer than two surviving trials are dropped.

ii.
```python
trial_keep = np.ones(n_trials, dtype=bool)
if exclude_auto_free:
    trial_keep &= auto_water == 0
    trial_keep &= free_water == 0
...
is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
...
common_obs_mask = np.all(is_good_trials[:, :n_recorded_trials], axis=0)
...
trial_keep &= recorded_trial_mask
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
session_input = [session_input[i] for i in np.flatnonzero(nonzero_trial_mask)]
session_output = [session_output[i] for i in np.flatnonzero(nonzero_trial_mask)]
if n_keep_trials < 2:
    continue
```

iii. The trajectory states the “main implementation choices” included exclusion of `auto_water` and `free_water` trials, plus later “intersecting with the valid ephys trial masks.” It also says the last two all-zero trials were pruned as a final sanity filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` for each kept unit, with go-cue times from `acquisition/BehavioralEvents/go_start_times/timestamps` used to place the bin edges.

ii.
```python
def _extract_unit_spikes(units_group, unit_idx):
    spike_times = units_group["spike_times"]
    spike_index = units_group["spike_times_index"]
    end = int(spike_index[unit_idx])
    start = 0 if unit_idx == 0 else int(spike_index[unit_idx - 1])
    return np.asarray(spike_times[start:end], dtype=np.float64)
...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
```

iii. The agent’s notes say neural activity is firing rates from non-overlapping spike-count bins, using classifier-good units and go-cue alignment.

## 2-b. How is the `neural` data processed?

i. Each good unit’s spike times are binned into 50 ms non-overlapping bins around each trial’s go cue. The script uses `np.searchsorted` on absolute bin edges, differences adjacent counts, and divides by the bin size to convert spike counts to firing rates in Hz.

ii.
```python
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
...
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
    spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
    session_neural[:, pos, :] = spike_counts
```

iii. The trajectory justification is that the decoder instructions require go-aligned 50 ms bins and that the script should emit firing rates in Hz from spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only when `units/classification == "good"`. Sessions with no such units are dropped. In addition, trials are filtered by a “common good-ephys” rule requiring all good units to mark the trial as good in `is_good_trials`, constrained by `obs_intervals`, and then any fully zero neural trials are removed.

ii.
```python
classification = _decode_str_array(units["classification"]).reshape(-1)
good_unit_indices = np.flatnonzero(classification == "good")

if len(good_unit_indices) == 0:
    continue
...
common_obs_mask = np.all(is_good_trials[:, :n_recorded_trials], axis=0)
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
```

iii. The trajectory repeatedly emphasizes `classification == "good"` unit inclusion and says the remaining all-zero trials were removed because they “carry no neural information.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to go cue onset by adding the fixed relative bin edges to each trial’s go-cue time to obtain absolute bin edges, then binning spikes against those edges.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
...
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
```

iii. The trajectory and notes both explicitly state that all trial data are aligned to go cue onset over `[-2.5 s, +1.5 s]`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, producing 80 bins per trial from -2.5 s to +1.5 s relative to the go cue. No additional temporal rebinning or smoothing is applied.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
N_BINS = len(BIN_CENTERS_REL)
```

iii. The agent cites the decoder instructions as the reason for the 50 ms non-overlapping binning scheme.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times`, `go_start_times`, and trial start times. For each trial it chooses the last sample-start timestamp before the go cue, with a fallback restricted to the current trial’s start-to-go interval.

ii.
```python
def _resolve_sample_onsets(trial_starts, go_starts, sample_starts):
    sample_idx = np.searchsorted(sample_starts, go_starts, side="right") - 1
    ...
    invalid = np.isnan(sample_onsets) | (sample_onsets < (trial_starts - 1e-9))
    if np.any(invalid):
        for i in np.where(invalid)[0]:
            mask = (sample_starts >= (trial_starts[i] - 1e-9)) & (sample_starts <= (go_starts[i] + 1e-9))
            candidates = sample_starts[mask]
            ...
            sample_onsets[i] = candidates[-1]
```

iii. The trajectory says `sample_start_times` can include replayed sample epochs, so the agent intentionally took the final sample onset before the trial’s go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After resolving the tone onset for each trial, the AI computes the time at each decoder bin center relative to that tone onset. This is done as go-aligned bin centers minus the tone-to-go offset.

ii.
```python
tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. The notes say the requested input is time from tone onset, so the agent computes it directly at each bin center rather than storing a per-trial scalar.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same 80 go-aligned bin centers used for the neural firing-rate bins.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
...
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. The trajectory justification is that all decoder variables should share the same go-aligned time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset` and `photostim_duration`, together with each trial’s `start_time` and the go-aligned bin-center times.

ii.
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
...
stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
```

iii. In the notes the agent says `photostim_onset` is interpreted relative to trial start and then converted onto the same time axis used for decoding.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The onset and duration strings are parsed into floats with `N/A` converted to `NaN`. For stimulated trials, the script marks a bin as 1 when that bin center lies within the photostim interval; otherwise it is 0.

ii.
```python
def _parse_optional_float_array(dataset):
    raw = _decode_str_array(dataset).reshape(-1)
    out = np.full(raw.shape[0], np.nan, dtype=np.float64)
    for i, value in enumerate(raw):
        if value in ("N/A", "", None):
            continue
        out[i] = float(value)
    return out
...
stim_on = np.zeros(N_BINS, dtype=np.float32)
if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
    ...
    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. The trajectory notes describe `photostimulation_on` as a time-varying binary input and justify the `N/A -> NaN` conversion as the representation for unstimulated trials.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation signal is compared against the same absolute go-aligned bin centers used for the neural data, so its 80 timepoints match the neural timepoints one-for-one.

ii.
```python
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. The agent’s justification is that all decoder streams should live on the same go-aligned bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI does not derive actual lick direction from `outcome`. Instead it uses only `trial_instruction` and treats that left/right trial label as the `choice` output.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
...
kept_trial_instruction = trial_instruction[trial_keep]
...
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
```

iii. In `CONVERSION_NOTES.md`, the agent explicitly justifies this by saying the reference code uses the paper’s left/right trial label and that, to preserve ignore trials, `choice` was encoded directly from `trial_instruction`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `trial_instruction` is mapped as `left -> 0` and `right -> 1`, then repeated across all 80 bins. No third “no lick” class is created for ignore trials.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
...
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_value, dtype=np.int16),
        np.full(N_BINS, outcome_value, dtype=np.int16),
        np.full(N_BINS, early_value, dtype=np.int16),
        tongue_cat,
    ]
)
...
"output_values": [
    ["left", "right"],
```

iii. The trajectory justification is the same as above: the agent decided to match the paper’s left/right trial label rather than infer the animal’s actual lick direction.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial table’s `outcome` field.

ii.
```python
outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
...
kept_outcomes = outcomes[trial_keep]
```

iii. The notes say this was mapped directly from the NWB `outcome` column.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are mapped as `ignore -> 0`, `miss -> 1`, `hit -> 2`, and repeated across all 80 bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
...
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. The trajectory says outcome was mapped directly from the NWB trial outcome field into the requested categorical codes.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial table’s `early_lick` field.

ii.
```python
early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
...
kept_early = early_lick[trial_keep]
```

iii. The notes say this output was mapped directly from the NWB `early_lick` field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are mapped as `no early -> 0` and `early -> 1`, then repeated across all 80 bins.

ii.
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
...
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
...
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. The trajectory justification is that early lick is already represented in the NWB trials table and only needs categorical recoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps` and the second column of `data` (`data[:, 1]`). It does not use the likelihood/confidence column in the conversion logic.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md`, the agent says tongue tracking comes from `Camera0_side_TongueTracking` and that its alignment uses the most recent marker sample within each window.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. First, session-wide 40th and 60th percentiles are computed on the full raw `tongue_y_all` trace. Then, for each trial and bin, the script samples the most recent tongue-y frame that falls inside that 50 ms bin. It does not average frames within bins and does not mask low-likelihood frames.

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
...
tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
...
tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
```

iii. The trajectory justification is explicit: the agent says “the reference alignment code uses the most recent marker sample within each time window rather than interpolation,” and therefore chose last-sample-in-bin sampling.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The sampled tongue-y value defaults to category 0, is promoted to 1 if it is at least the session 40th percentile, and to 2 if it is greater than the session 60th percentile. There is no separate “not visible” category.

ii.
```python
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
...
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40th_pct", "40th_to_60th_pct", "gt_60th_pct"],
],
```

iii. The notes justify this as a per-session discretization using the full-session tongue-y distribution at the 40th and 60th percentiles.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial, the tongue output uses the same go-aligned 50 ms bins as the neural data. A tongue sample is assigned to a bin if the latest timestamp before that bin’s end also falls after that bin’s start.

ii.
```python
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    valid = idx >= 0
    if np.any(valid):
        valid_idx = idx[valid]
        valid[valid] &= timestamps[valid_idx] >= bin_starts[valid]
```

iii. The trajectory says the tongue output was intentionally put on the same 50 ms go-aligned decoder grid as the neural data, using the latest marker sample inside each time window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles byte/string decoding explicitly, converts optional numeric trial fields such as photostim timing from strings with `N/A` to `NaN`, uses a fallback search if the naive sample-onset lookup would cross a trial boundary, drops sessions with no good units, and removes any all-zero neural trials at the end. For tongue data, bins without a valid sampled frame remain at the default zero value rather than becoming a distinct missing-data category.

ii.
```python
def _decode_scalar(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value
...
def _parse_optional_float_array(dataset):
    ...
    if value in ("N/A", "", None):
        continue
...
invalid = np.isnan(sample_onsets) | (sample_onsets < (trial_starts - 1e-9))
...
if len(good_unit_indices) == 0:
    continue
...
sampled = np.zeros(N_BINS, dtype=np.float32)
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
```

iii. The trajectory justifies the sample-onset fallback as protection against replayed sample epochs, and justifies removing all-zero trials as a final sanity check because they “carry no neural information.”

## 10-a. What are the most time-consuming steps of the code?

i. The code’s main runtime costs are per-session NWB/HDF5 reads, the per-unit spike extraction and `searchsorted` neural binning loop, and the per-trial loop that builds inputs/outputs and bins tongue data. Writing the full pickle and summary artifacts is also substantial.

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
            tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
...
with open(args.full_out, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory does not contain a detailed runtime analysis; it mainly shows the agent focusing on correctness and post-hoc validation rather than optimizing these loops.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: the per-value loop in `_parse_optional_float_array`, the per-unit region-label loop, the per-unit spike-extraction/binning loop, the per-trial input/output assembly loop, the per-trial tongue binning, and the sample-subset copying loops.

ii.
```python
for i, value in enumerate(raw):
    ...
for i, unit_idx in enumerate(good_unit_indices):
    region_name = str(anno_name[unit_idx])
    ...
for pos, unit_idx in enumerate(good_unit_indices):
    ...
for local_idx, trial_idx in enumerate(keep_trial_indices):
    ...
for sess_idx in session_indices:
    ...
```

iii. The trajectory does not show the agent explicitly discussing vectorization; it accepted these loops while prioritizing getting a verifier-clean output.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several passes and copies: it parses three optional photostim fields separately, builds session input/output trial lists before filtering zero-neural trials and then copies them again after filtering, and creates a second sample dataset by copying slices out of the full dataset.

ii.
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
photostim_power = _parse_optional_float_array(trials["photostim_power"])
...
session_input.append(input_trial)
session_output.append(output_trial)
...
session_input = [session_input[i] for i in np.flatnonzero(nonzero_trial_mask)]
session_output = [session_output[i] for i in np.flatnonzero(nonzero_trial_mask)]
...
sample_data = _subset_data(data, sample_sessions, args.sample_trials_per_session)
```

iii. The trajectory contains no explicit justification for this repetition; it appears to be a convenience-oriented implementation rather than a deliberately minimal one-pass design.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI script computes and stores several things not used by the downstream decoder on the full dataset path: `photostim_power` is parsed but never used, `trial_stops` is only used for summary statistics, `region_names` in `_subset_data` is unused, and the script spends time generating sample/summary artifacts in addition to the required full pickle.

ii.
```python
trial_stops = np.asarray(trials["stop_time"], dtype=np.float64)
...
photostim_power = _parse_optional_float_array(trials["photostim_power"])
...
region_names = []
...
"mean_trial_duration_s": float(np.mean(trial_stops[trial_keep] - trial_starts[trial_keep])),
...
sample_data = _subset_data(data, sample_sessions, args.sample_trials_per_session)
...
with open(args.summary_json, "w") as f:
    json.dump(summary, f, indent=2)
```

iii. The trajectory suggests these extra artifacts were added for validation and reproducibility, not because the decoder itself needed them.
