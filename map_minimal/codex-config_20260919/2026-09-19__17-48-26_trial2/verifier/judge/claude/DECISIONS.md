# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files under `data/sub-*/` using a sorted glob pattern, then opens each file with `h5py` (not `pynwb`) and processes it in `convert_session()`. Subjects, trials, and units are read from HDF5 paths such as `units/classification`, `intervals/trials`, and `acquisition/BehavioralEvents`.

ii.
```python
paths = sorted(glob.glob(os.path.join(data_dir, "sub-*", "*.nwb")))
...
for path in paths:
    session = convert_session(path)
```

```python
with h5py.File(path, "r") as nwb:
    units = nwb["units"]
    trials = nwb["intervals/trials"]
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
```

iii. The agent verified that both `pynwb` and `h5py` were available and chose `h5py` for direct HDF5 access. It noted there are 174 NWB files across 28 subjects, with one session dropped for lacking classifier QC labels, leaving 173.

## 1-b. How are the data split into subjects?

i. Each NWB file's subject is read from `general/subject/subject_id` (a numeric string like `'440956'`). After processing, `subjects` is the sorted set of unique IDs and `subject_idx` indexes each session into that list.

ii.
```python
subject = _scalar_string(nwb["general/subject/subject_id"])
...
subjects = sorted({session["subject"] for session in converted_sessions})
subject_to_index = {subject: i for i, subject in enumerate(subjects)}
"subject_idx": np.array(
    [subject_to_index[session["subject"]] for session in converted_sessions],
    dtype=np.int16,
),
```

iii. The agent used the NWB `subject_id` field, which is the canonical animal identifier in the file. This yielded 28 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each session is identified by `nwb["identifier"]`. Session order follows the sorted file list.

ii.
```python
paths = sorted(glob.glob(os.path.join(data_dir, "sub-*", "*.nwb")))
...
identifier = _scalar_string(nwb["identifier"])
```

iii. The dandiset stores one session per file, so the file boundary is the session boundary. 173 of 174 files are retained (one dropped for lacking QC labels).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioural trial. The row count is checked against the number of go-cue events.

ii.
```python
trials = nwb["intervals/trials"]
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
if len(go_times) != len(trials["id"]):
    raise ValueError(f"{path}: go cue and trial counts differ")
```

iii. The agent validated that go cue count matches trial count in every session.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in three stages:
1. **Neural validity**: Uses `units["is_good_trials"]` matrix. For each trial, ALL good units must have `is_good_trials == True`. Trials beyond the ephys recording range are excluded.
2. **Assistance trials**: Trials with `auto_water != 0` OR `free_water != 0` are excluded.
3. **Empty neural**: After binning, trials where all firing rates are zero are excluded.
4. Sessions with fewer than 2 surviving trials are dropped.

Early-lick, ignore, and photostimulation trials are deliberately kept.

ii.
```python
unit_trial_validity = units["is_good_trials"][:][good_indices]
n_ephys_trials = unit_trial_validity.shape[1]
neural_valid = np.zeros(len(go_times), dtype=bool)
neural_valid[:n_ephys_trials] = np.all(unit_trial_validity, axis=0)

assistance = (trials["auto_water"][:] != 0) | (trials["free_water"][:] != 0)
keep = neural_valid & ~assistance
...
neural_nonzero = np.any(firing_rates_all != 0, axis=(1, 2))
keep &= neural_nonzero
```

iii. The agent stated: "Trials with automatic or free water are training/assistance trials and are excluded, as in the reference analysis. Ordinarily the paper also excludes early-lick, no-response, and photostimulation trials. Those three exclusions are deliberately not made here because the requested decoder variables explicitly require those trials." The agent also added an empty-neural check to catch trials at the end of recordings where spike data may be absent.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (sorted spike times of each unit in session-absolute seconds), with `units/spike_times_index` for ragged indexing. Only units with `classification == 'good'` AND non-empty `anno_name` contribute.

ii.
```python
all_spikes = units["spike_times"][:]
spike_ends = units["spike_times_index"][:].astype(np.int64)
spike_starts = np.r_[0, spike_ends[:-1]]
```

iii. `spike_times` is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit, spikes are assigned to trial-time bins using vectorized searchsorted and bincount, then divided by bin width (0.05s). No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
for out_unit, unit_index in enumerate(good_indices):
    spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
    trial_index = np.searchsorted(window_starts, spikes, side="right") - 1
    valid = trial_index >= 0
    trial_index = trial_index[valid]
    spikes = spikes[valid]
    valid = spikes < window_ends[trial_index]
    trial_index = trial_index[valid]
    spikes = spikes[valid]
    time_index = np.floor(
        (spikes - window_starts[trial_index]) / BIN_S + 1e-12
    ).astype(np.int64)
    valid = (time_index >= 0) & (time_index < N_TIME)
    flat_index = trial_index[valid] * N_TIME + time_index[valid]
    counts = np.bincount(flat_index, minlength=n_trials * N_TIME)
    rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_S
```

iii. The agent computed firing rate from binned spike counts divided by the 50ms bin width, consistent with the instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` AND a non-empty `anno_name` (CCF annotation) are kept. A session with no such units is dropped. This retains 69,453 units across 173 sessions.

ii.
```python
class_values = classification.asstr()[:]
annotations = units["anno_name"].asstr()[:]
annotation_present = np.array([bool(x.strip()) for x in annotations])
good = (class_values == "good") & annotation_present
good_indices = np.flatnonzero(good)
if len(good_indices) == 0:
    return None
```

iii. The agent stated: "The source is 173 NWB sessions, and the reference analysis uses classifier-based unit QC plus explicit trial exclusions." The additional `anno_name` filter ensures every retained unit has a valid brain region annotation for the `brain_region_idx` field.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB times share one session-absolute clock. The bin edges are computed as go_cue_time + OFF_START to go_cue_time + OFF_END. Spikes are assigned to trials via searchsorted on window_starts, then to time bins by flooring the offset from window start.

ii.
```python
window_starts = go_times + OFF_START
window_ends = go_times + OFF_END
...
trial_index = np.searchsorted(window_starts, spikes, side="right") - 1
...
time_index = np.floor(
    (spikes - window_starts[trial_index]) / BIN_S + 1e-12
).astype(np.int64)
```

iii. Everything in the NWB file is timestamped on one global clock, so aligning to the go cue only requires looking up each trial's go-cue time and taking the window around it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins spanning -2.5s to +1.5s relative to the go cue, giving 80 timepoints per trial. The bin grid is defined once and reused for all trials/sessions.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_S = 0.050
N_TIME = int(round((OFF_END - OFF_START) / BIN_S))  # 80
BIN_EDGES = OFF_START + np.arange(N_TIME + 1) * BIN_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The 50ms bin width and [-2.5, 1.5]s window are directly from the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `presample_stop_times` timestamps (the end of the presample period / start of the sample tone), together with the go cue of each trial. The AI chose `presample_stop_times` because it has exactly one entry per trial, unlike `sample_start_times` which can fire multiple times when early licks replay epochs.

ii.
```python
tone_onsets = nwb[
    "acquisition/BehavioralEvents/presample_stop_times/timestamps"
][:][keep]
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. The agent stated: "'tone onset' is the initial sample onset, the one event that remains one-to-one with trials." The code comment says: "presample_stop is the first tone/sample onset and, unlike sample_start, has exactly one entry per trial even when an early lick replays epochs."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute bin center times are subtracted by the tone onset time, giving time-from-tone in seconds.

ii.
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. Straightforward subtraction; no additional processing needed.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers are computed from the same go-cue-relative grid used for neural data. `absolute_bin_times` = go_time + BIN_CENTERS, which is the same grid used for spike binning. So each time-from-tone value corresponds exactly to its matching neural time bin.

ii.
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. Both neural and input data are computed on the same go-cue-aligned time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` used to convert to absolute times.

ii.
```python
onset = _numeric_or_nan(trials["photostim_onset"].asstr()[:])[keep]
duration = _numeric_or_nan(trials["photostim_duration"].asstr()[:])[keep]
trial_starts = trials["start_time"][:][keep]
stimulation_start = trial_starts + onset
stimulation_end = stimulation_start + duration
```

iii. The onsets are stored as strings relative to trial start with `'N/A'` for non-stimulated trials, converted to floats with NaN for N/A.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 where its center falls between stimulation onset and offset, 0 elsewhere. NaN comparisons yield False, so non-stimulated trials stay all-zero.

ii.
```python
photostim = (
    (absolute_bin_times >= stimulation_start[:, None])
    & (absolute_bin_times < stimulation_end[:, None])
)
```

iii. Binary time series computed from onset/offset intervals.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Stimulation onset/offset are in absolute session time. `absolute_bin_times` is the same grid used for neural binning (go_time + BIN_CENTERS). Comparison is direct.

ii.
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
photostim = (
    (absolute_bin_times >= stimulation_start[:, None])
    & (absolute_bin_times < stimulation_end[:, None])
)
```

iii. Same go-cue-aligned time grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') in the trials table. Choice is not stored directly.

ii.
```python
instruction = trials["trial_instruction"].asstr()[:][keep]
outcome_text = trials["outcome"].asstr()[:][keep]
...
choice = np.full(len(kept_indices), 2, dtype=np.int8)  # no lick
responded = outcome_text != "ignore"
reported_left = ((outcome_text == "hit") & (instruction == "left")) | (
    (outcome_text == "miss") & (instruction == "right")
)
choice[responded & reported_left] = 0
choice[responded & ~reported_left] = 1
```

iii. The agent's code comment states: "Outcome and instruction encode the report robustly even when the raw lick-event stream contains a brief lick at the other port. Ignore means no reported lick; miss means the report was opposite the instruction."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no lick and repeated across all 80 time bins.

ii.
```python
choice = np.full(len(kept_indices), 2, dtype=np.int8)  # no lick
responded = outcome_text != "ignore"
reported_left = ((outcome_text == "hit") & (instruction == "left")) | (
    (outcome_text == "miss") & (instruction == "right")
)
choice[responded & reported_left] = 0
choice[responded & ~reported_left] = 1
...
np.full(N_TIME, choice[trial_index], dtype=np.int8),
```

iii. Per-trial value repeated across bins to create time-varying output array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', and 'hit'.

ii.
```python
outcome_text = trials["outcome"].asstr()[:][keep]
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int8)
```

iii. The trials table stores outcome explicitly with the three required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0=ignore, 1=miss, 2=hit and repeated across all 80 time bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int8)
...
np.full(N_TIME, outcome[trial_index], dtype=np.int8),
```

iii. Direct mapping, one value per trial repeated across bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
early_text = trials["early_lick"].asstr()[:][keep]
early = (early_text == "early").astype(np.int8)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes and repeated across all 80 time bins.

ii.
```python
early = (early_text == "early").astype(np.int8)
...
np.full(N_TIME, early[trial_index], dtype=np.int8),
```

iii. Direct boolean mapping, one value per trial repeated across bins.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is (n_frames, 3) = x, y, likelihood. Column index 1 (y) is used for position; column index 2 (likelihood) determines visibility.

ii.
```python
tracking = behavior["Camera0_side_TongueTracking"]
tracking_data = tracking["data"][:, 1:3]  # y, DeepLabCut likelihood
tracking_times = tracking["timestamps"][:]
y_all = tracking_data[:, 0]
likelihood_all = tracking_data[:, 1]
```

iii. This is the only tongue measurement in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Processing steps:
1. Frames with `likelihood < 0.90` are marked invisible.
2. Percentiles (40th, 60th) are computed from **raw visible frames** across the full session (not binned).
3. For each bin center time, the **preceding** video frame is sampled via searchsorted.
4. Sampled frames are classified: 0 if y < p40, 1 if p40 <= y <= p60, 2 if y > p60, 3 if not visible.

ii.
```python
TONGUE_LIKELIHOOD_CUTOFF = 0.90

visible_all = (
    np.isfinite(y_all)
    & np.isfinite(likelihood_all)
    & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
)
percentile_40, percentile_60 = np.percentile(y_all[visible_all], [40, 60])

frame_index = np.searchsorted(
    tracking_times, absolute_bin_times.ravel(), side="right"
) - 1
...
result[visible & (y < percentile_40)] = 0
result[visible & (y >= percentile_40) & (y <= percentile_60)] = 1
result[visible & (y > percentile_60)] = 2
```

iii. The agent stated the tongue visibility rule uses "side-camera DeepLabCut likelihood >= 0.90; percentiles computed from visible frames over each full session."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four categories: 0 (below 40th percentile), 1 (40th-60th percentile), 2 (above 60th percentile), 3 (not visible). The percentiles are computed per-session from raw visible frames. The boundary conditions are: strictly less than p40 for class 0, inclusive on both sides for class 1 (>= p40 AND <= p60), strictly greater than p60 for class 2.

ii.
```python
result[visible & (y < percentile_40)] = 0
result[visible & (y >= percentile_40) & (y <= percentile_60)] = 1
result[visible & (y > percentile_60)] = 2
```

iii. Follows the instruction's 40th/60th percentile split with a fourth "not visible" class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center in `absolute_bin_times` (the same go-cue-aligned grid used for neural data), the preceding video frame is found via `searchsorted(tracking_times, bin_time, side="right") - 1`. This samples a single frame per bin rather than averaging all frames within each bin.

ii.
```python
frame_index = np.searchsorted(
    tracking_times, absolute_bin_times.ravel(), side="right"
) - 1
in_range = (frame_index >= 0) & (frame_index < len(tracking_times))
safe_index = np.clip(frame_index, 0, len(tracking_times) - 1)
sampled = tracking_data[safe_index]
```

iii. The camera timestamps share the global clock with spikes and events, so no interpolation or offset correction is needed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Session without classifier QC**: The `classification` column has non-string dtype (NaN values). Detected by checking `classification.dtype.kind not in "OSU"` and session is dropped.
- **Trials without valid neural data**: Excluded via `is_good_trials` matrix and the zero-firing-rate check.
- **Invisible tongue frames**: Frames with likelihood < 0.90 are excluded from percentile computation; bins with no visible frame get class 3 ("not visible").

ii.
```python
if classification.dtype.kind not in "OSU":
    return None

neural_nonzero = np.any(firing_rates_all != 0, axis=(1, 2))
keep &= neural_nonzero

visible_all = (
    np.isfinite(y_all) & np.isfinite(likelihood_all)
    & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
)
result = np.full(len(frame_index), 3, dtype=np.int8)
```

iii. Missing data is handled by exclusion (dropping sessions/trials with no valid data) or by explicit categorization (tongue "not visible" class).

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file and the spike binning loop dominate. The per-unit loop in `_bin_good_units` performs searchsorted and bincount for each good unit. The full conversion processes 173 sessions.

ii.
```python
for out_unit, unit_index in enumerate(good_indices):
    spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
    trial_index = np.searchsorted(window_starts, spikes, side="right") - 1
    ...
    counts = np.bincount(flat_index, minlength=n_trials * N_TIME)
    rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_S
```

iii. I/O and per-unit binary search are the dominant costs, both scaling with data size.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `_bin_good_units` iterates over each good unit to extract and bin its spikes. This loop exists because each unit has a different number of spikes (ragged storage). The tongue classification uses vectorized searchsorted for frame lookup, with no explicit trial loop.

ii.
```python
for out_unit, unit_index in enumerate(good_indices):
    spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
    ...
```

iii. The per-unit loop is inherent to ragged spike storage and cannot be easily collapsed.

## 10-c. What processing does the code repeat multiple times?

i. The code bins firing rates for ALL trials (including ones later excluded by the assistance or empty-neural filter) because `_bin_good_units` is called before the final `keep` mask is applied. This means spikes are binned for auto_water and free_water trials unnecessarily.

ii.
```python
firing_rates_all = _bin_good_units(units, good_indices, go_times)  # ALL trials
...
neural_nonzero = np.any(firing_rates_all != 0, axis=(1, 2))
keep &= neural_nonzero
...
firing_rates = firing_rates_all[keep]  # then filter
```

iii. Binning all trials first then filtering is simpler but does redundant work for excluded trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Firing rates are computed for all trials (including assistance trials and empty-neural trials) then filtered afterward. The brain region lookup table is decompressed from a base85/zlib blob at import time. Additionally, the code computes detailed per-session statistics (excluded trial counts by category, tongue visibility fraction) that are stored in metadata but not used by the decoder.

ii.
```python
firing_rates_all = _bin_good_units(units, good_indices, go_times)
...
firing_rates = firing_rates_all[keep]

REGION_LOOKUP = json.loads(
    zlib.decompress(base64.b85decode(_REGION_LOOKUP_B85)).decode("utf-8")
)
```

iii. Computing rates for all trials before filtering is a design choice that simplifies the logic at the cost of some wasted computation.
