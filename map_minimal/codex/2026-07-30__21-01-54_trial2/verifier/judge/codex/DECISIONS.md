# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code loads all `.nwb` files under `sub-*/*.nwb` with `Path.glob`, sorts them, and opens each session file directly with `h5py`. Within each file it reads the trial table, behavioral event groups, tongue-tracking time series, and units tables from raw HDF5 paths rather than using `pynwb`.

ii. 
```python
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

iii. The notes say the input files are “all NWB files under `/app/data/sub-*/*.nwb`” and the trajectory shows the agent explicitly decided to map the NWB layout onto raw dataset fields before implementing the converter. No separate justification was given for using `h5py` instead of `pynwb`.

## 1-b. How are the data split into subjects?

i. Subjects are split using each file’s `general/subject/subject_id`. The output `subjects` list is built in first-seen order while iterating through the sorted session files, and `subject_idx` stores the corresponding integer for each kept session.

ii.
```python
subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
subject = str(subject)
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The notes state that the final dataset contains 28 subjects and use the NWB subject field as the canonical identifier. No further grouping logic or mouse-name remapping is documented.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order is the sorted file order, and sessions with no good units or fewer than two kept trials are skipped rather than represented as empty sessions.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
if len(good_unit_indices) == 0:
    continue
...
if n_keep_trials < 2:
    continue
...
neural_sessions.append([np.ascontiguousarray(session_neural[i]) for i in range(n_keep_trials)])
```

iii. The notes explicitly say “One file, `sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`, has zero units with `units/classification == "good"` and is dropped,” so the file boundary is the session boundary and unusable files are discarded.

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table. The code assumes one trial per table row and checks that the number of go-cue timestamps equals the number of trial rows before continuing.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
if len(go_starts) != n_trials:
    raise RuntimeError(f"{path.name}: expected {n_trials} go cues, found {len(go_starts)}")
```

iii. The notes say the “Base trial table comes from `intervals/trials`” and that “Go cue count matches trial count in each converted session.”

## 1-e. How are trials filtered based on quality controls?

i. The AI applies four trial filters: it excludes `auto_water == 1`, excludes `free_water == 1`, intersects trials with a “common good-ephys support” mask built from `units/obs_intervals` and `units/is_good_trials`, and finally removes any kept trials whose neural tensor is entirely zero after binning. Sessions with fewer than two remaining trials are dropped.

ii.
```python
trial_keep = np.ones(n_trials, dtype=bool)
if exclude_auto_free:
    trial_keep &= auto_water == 0
    trial_keep &= free_water == 0
...
is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
...
trial_keep &= recorded_trial_mask
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
session_input = [session_input[i] for i in np.flatnonzero(nonzero_trial_mask)]
session_output = [session_output[i] for i in np.flatnonzero(nonzero_trial_mask)]
...
if n_keep_trials < 2:
    continue
```

iii. The notes justify this as excluding `auto_water` and `free_water` trials, restricting to trials with “common good-ephys support,” and pruning two residual all-zero neural trials. The trajectory shows the agent added the `is_good_trials`/`obs_intervals` logic after verifier warnings about all-zero trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, `units/spike_times_index`, the unit-level `classification` field used to select “good” units, and trial go-cue timestamps used to place the bins.

ii.
```python
classification = _decode_str_array(units["classification"]).reshape(-1)
good_unit_indices = np.flatnonzero(classification == "good")
...
unit_spikes = _extract_unit_spikes(units, int(unit_idx))
spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
```

iii. The notes say “Unit inclusion follows the NWB classifier output: keep units where `units/classification == "good"`,” and describe the neural signal as firing rates in Hz from 50 ms spike-count bins.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the AI bins spike times into non-overlapping 50 ms windows aligned to go cue and converts counts to firing rates by dividing by bin width. There is no smoothing, normalization, or baseline subtraction.

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

iii. The notes describe the neural output as “firing rates in Hz from non-overlapping spike-count bins,” and the trajectory confirms the agent deliberately chose 50 ms non-overlapping go-aligned bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. At the unit level, only `classification == "good"` units are kept. At the trial level, the neural data is also restricted to trials that pass the `obs_intervals`/`is_good_trials` mask and to trials whose neural tensor is not all zero.

ii.
```python
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    continue
...
trial_keep &= recorded_trial_mask
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
```

iii. The notes justify unit inclusion with the NWB classifier output and say trials are restricted to those with “common good-ephys support.” The trajectory shows this trial-level neural QC was added reactively to suppress all-zero-trial verifier failures.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. The code constructs absolute bin edges by adding a fixed relative window `[-2.5, 1.5]` s to each trial’s go-cue timestamp.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
...
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
```

iii. The notes explicitly say the alignment event is “go cue onset from `acquisition/BehavioralEvents/go_start_times/timestamps`.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins over a 4 s window from `-2.5` to `+1.5` s relative to go cue, giving 80 bins. No extra temporal rebinning or smoothing is applied.

ii.
```python
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
N_BINS = len(BIN_CENTERS_REL)
```

iii. The notes say each kept trial is represented with “80 bins of width 50 ms, aligned to go cue onset, spanning `-2.5 s` to `+1.5 s`.”

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `intervals/trials/start_time`, `BehavioralEvents/go_start_times/timestamps`, and `BehavioralEvents/sample_start_times/timestamps`.

ii.
```python
trial_starts = np.asarray(trials["start_time"], dtype=np.float64)
...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
sample_starts = np.asarray(be["sample_start_times"]["timestamps"], dtype=np.float64)
tone_onsets = _resolve_sample_onsets(trial_starts, go_starts, sample_starts)
```

iii. The notes explain that `sample_start_times` can include replayed sample epochs after early licks, so the tone onset must be resolved from the event stream rather than read per trial directly.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI resolves each trial’s tone onset as the last `sample_start_times` entry before its go cue, with a fallback restricted to that trial’s own start-to-go interval. It then computes time-from-tone at each bin center as the go-aligned bin center minus the tone’s offset relative to go cue.

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

iii. The notes explicitly justify the “last sample start before go cue” rule because early licks can replay the sample epoch, and say the input is computed at each bin center.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-aligned bin centers as the neural firing rates, one scalar per bin.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
...
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. The notes say “Input channel `time_from_tone_onset_s` is computed at each bin center,” which matches the neural binning grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset`, `intervals/trials/photostim_duration`, and trial start/go times. The code also loads `photostim_power` but does not use it.

ii.
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
photostim_power = _parse_optional_float_array(trials["photostim_power"])
```

```python
stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
```

iii. The notes justify this by stating that `photostim_onset` is interpreted relative to trial start and that the binary input should indicate whether stimulation is on at each time point.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts onset and duration into an absolute stimulation interval for each trial and marks each bin center with `1` if it lies within that interval, else `0`. Non-stimulated trials remain all zeros.

ii.
```python
stim_on = np.zeros(N_BINS, dtype=np.float32)
if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
    stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
    stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. The notes say “`photostimulation_on` is 1 when the bin center falls inside the photostim interval.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned on the same go-cue-centered time grid as the neural data by comparing the neural bin centers, expressed in absolute time, against the stimulation interval.

ii.
```python
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. The notes justify this by saying the photostimulation interval is put on the same time axis as the go-aligned bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives `choice` only from `intervals/trials/trial_instruction`, not from `outcome`. It treats the paper’s left/right trial label as the decoder choice target.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
...
kept_trial_instruction = trial_instruction[trial_keep]
...
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
```

iii. The notes explicitly say the reference code uses the paper’s left/right trial label and that, to preserve ignore trials, `choice` is encoded directly from `trial_instruction` as `left -> 0`, `right -> 1`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The chosen left/right label is mapped to `0/1` and repeated across all 80 bins for each trial. There is no extra “no lick” class for ignore trials.

ii.
```python
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
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

iii. The notes justify this as preserving ignore trials while keeping the paper’s trial-label notion of “choice,” rather than reconstructing actual lick direction from `outcome`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `intervals/trials/outcome`.

ii.
```python
outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
...
kept_outcomes = outcomes[trial_keep]
```

iii. The notes state that outcome is “Mapped directly from NWB `outcome`.”

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are mapped as `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeated across the 80 bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
...
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. The notes list the same mapping and present outcome as a per-trial output.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from `intervals/trials/early_lick`.

ii.
```python
early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
...
kept_early = early_lick[trial_keep]
```

iii. The notes state that early lick is “Mapped directly from NWB `early_lick`.”

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two trial labels are mapped as `no early -> 0` and `early -> 1`, then repeated across bins.

ii.
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
...
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
...
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. The notes list the same mapping and treat early lick as a per-trial output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps` and column 1 of the `data` array. It does not use the tracking-likelihood column when constructing the output.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
```

iii. The notes say side-camera tongue tracking comes from `Camera0_side_TongueTracking` and describe the output as using the “most recent marker sample within each time window rather than interpolation.” They do not mention using likelihood-based visibility filtering.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes per-session 40th and 60th percentiles from the full session’s raw `tongue_y` values, with no likelihood filter. For each trial and 50 ms bin, it samples the last camera frame that falls inside the bin, uses zero if no frame qualifies, and then discretizes that sampled value into categories.

ii.
```python
tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
```

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

iii. The notes justify this by saying the reference alignment code used the most recent marker sample in each time window and that discretization should use the full-session tongue-y distribution.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI thresholds each binned sample using session-level `q40` and `q60`: values below `q40` stay `0`, values `>= q40` become `1`, and values `> q60` become `2`. It uses only these three classes.

ii.
```python
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
```

iii. The notes state the intended category scheme as `< 40th percentile -> 0`, `40th to 60th percentile -> 1`, `> 60th percentile -> 2`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue values are aligned to the same go-cue-centered 50 ms bins as the neural data. For each bin, the code selects the last camera sample whose timestamp falls within that bin’s absolute time interval.

ii.
```python
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    ...
```

iii. The notes justify this as following the same windowing used for the neural data while avoiding interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data issues explicitly: text-like missing values in HDF5 string datasets are decoded and optional numeric strings such as photostim timing treat `"N/A"` as `NaN`; unresolved sample onsets trigger a fallback search constrained to the trial’s own interval and then raise an error if still unresolved; sessions with no good units are skipped; trials failing the recorded-ephys mask or with all-zero neural data are dropped. Missing tongue samples within a bin are implicitly filled with `0` before discretization rather than assigned a separate missing-data category.

ii.
```python
def _parse_optional_float_array(dataset):
    raw = _decode_str_array(dataset).reshape(-1)
    out = np.full(raw.shape[0], np.nan, dtype=np.float64)
    for i, value in enumerate(raw):
        if value in ("N/A", "", None):
            continue
        out[i] = float(value)
```

```python
if np.any(invalid):
    for i in np.where(invalid)[0]:
        ...
        if len(candidates) == 0:
            raise RuntimeError(f"Could not resolve sample onset for trial {i}.")
```

```python
sampled = np.zeros(N_BINS, dtype=np.float32)
if np.any(valid):
    sampled[valid] = y_values[idx[valid]].astype(np.float32)
```

iii. The notes and trajectory justify the trial-level pruning as a response to verifier-discovered all-zero trials. The notes do not document the zero-fill behavior for tongue bins; that behavior is only visible in the code.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are likely opening every NWB file, extracting spike trains one unit at a time, running `np.searchsorted` over all trial/bin edges for each unit, and repeatedly binning tongue data trial by trial.

ii.
```python
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        ...
        for pos, unit_idx in enumerate(good_unit_indices):
            unit_spikes = _extract_unit_spikes(units, int(unit_idx))
            spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
```

```python
for local_idx, trial_idx in enumerate(keep_trial_indices):
    ...
    tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
```

iii. No explicit runtime breakdown is documented in the notes, so this is inferred from the structure of the code and the large per-session loops.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike extraction/binning loop, the per-trial input/output assembly loop, the invalid-trial fallback loop in `_resolve_sample_onsets`, and the per-neuron region remapping loop in `_subset_data` could all be vectorized or at least reduced.

ii.
```python
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    ...
```

```python
for local_idx, trial_idx in enumerate(keep_trial_indices):
    ...
```

```python
for i in np.where(invalid)[0]:
    ...
```

iii. No explicit justification was documented for leaving these loops in place. The code simply implements them directly.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly slices ragged spike arrays one unit at a time instead of reading the spike buffer once, repeatedly searches the full tongue timestamp array once per kept trial, and repeatedly constructs per-trial constant output rows with `np.full` inside the trial loop.

ii.
```python
def _extract_unit_spikes(units_group, unit_idx):
    spike_times = units_group["spike_times"]
    spike_index = units_group["spike_times_index"]
    end = int(spike_index[unit_idx])
    start = 0 if unit_idx == 0 else int(spike_index[unit_idx - 1])
    return np.asarray(spike_times[start:end], dtype=np.float64)
```

```python
for local_idx, trial_idx in enumerate(keep_trial_indices):
    ...
    tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
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

iii. The notes do not describe this as an intentional design choice; it is visible from the implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `photostim_power` but never uses it; it also computes some statistics only for `conversion_summary.json` such as mean trial duration, mean tone-to-go offset, and per-session tongue percentiles, which are not needed by the downstream decoder. The helper `_subset_data` also remaps subjects and regions only for the sample artifact, not for the full converted dataset used downstream.

ii.
```python
photostim_power = _parse_optional_float_array(trials["photostim_power"])
```

```python
summary["per_session"].append(
    {
        ...
        "tongue_q40": float(tongue_q40),
        "tongue_q60": float(tongue_q60),
        "mean_trial_duration_s": float(np.mean(trial_stops[trial_keep] - trial_starts[trial_keep])),
        "mean_tone_onset_rel_go_s": float(np.mean(kept_tone_onsets - go_starts[trial_keep])),
    }
)
```

iii. No explicit justification was documented for these extra computations beyond producing summary artifacts.
