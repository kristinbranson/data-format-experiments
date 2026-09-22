# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files via `sorted(glob.glob('/app/data/sub-*/*.nwb'))`, iterates over them, and opens each with `pynwb.NWBHDF5IO`. Within each file, it reads `nwb.units`, `nwb.trials`, `nwb.acquisition['BehavioralEvents']`, and `nwb.acquisition['BehavioralTimeSeries']`. Each file is one session.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
# ...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, bin_edges, n_bins, ...)
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
subject_id = nwb.subject.subject_id
classification = nwb.units['classification'][:]
# ...
be = nwb.acquisition['BehavioralEvents']
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
```

iii. From CONVERSION_NOTES.md: "NWB format (Neurodata Without Borders) ... Each file: one session with behavior + electrophysiology." The AI recognized NWB as the published format and used pynwb as the standard reader.

## 1-b. How are the data split into subjects?

i. Each NWB file contains `nwb.subject.subject_id` (a numeric string like `'440956'`). The AI collects unique subject IDs across all sessions and creates a sorted list of subjects with corresponding index mapping.

ii.
```python
subject_id = nwb.subject.subject_id
# ...
subject_ids = sorted(set(r['subject_id'] for r in all_results))
subjects = [str(sid) for sid in subject_ids]
subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
```

iii. From CONVERSION_NOTES.md: "28 subjects (sub-440956 through sub-484677)". The AI used `subject_id` from the NWB file directly.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The AI processes each file independently and collects results into a list. 174 files are found, 173 sessions survive (one dropped for having no good units).

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
# ...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
    if result is not None:
        all_results.append(result)
```

iii. From CONVERSION_NOTES.md: "174 NWB files total" and "1 session skipped (sub-440958_ses-20190216 - no valid trials within obs_intervals)".

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials` which provides one row per behavioral trial. Go cue times are read from `BehavioralEvents/go_start_times`.

ii.
```python
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
trial_instruction = nwb.trials['trial_instruction'][:]
# ...
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI used the NWB trials table directly, which defines trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials with three criteria: (1) `auto_water == 0`, (2) `free_water == 0`, and (3) the trial's analysis window (`go_cue + [-2.5, 1.5]`) must fall within the obs_intervals recording range (start of first interval to end of last interval for the first good unit). Sessions with fewer than 2 valid trials are dropped.

ii.
```python
valid_trial_mask = (auto_water == 0) & (free_water == 0)

obs_intervals = nwb.units['obs_intervals'][good_indices[0]]
obs_start = obs_intervals[0, 0]
obs_end = obs_intervals[-1, 1]

begin_time = bin_edges[0]
end_time = bin_edges[-1]
within_obs = ((go_cue_times_all + begin_time) >= obs_start) & \
             ((go_cue_times_all + end_time) <= obs_end)
valid_trial_mask = valid_trial_mask & within_obs
```

iii. From CONVERSION_NOTES.md: "Only exclude auto_water and free_water trials. Keep early lick, ignore, and photostim trials since they are decoder inputs/outputs." The AI justified keeping early_lick and ignore trials because they are decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for each unit classified as `'good'`. Go cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
all_spike_times_obj = nwb.units['spike_times']
for i, unit_idx in enumerate(good_indices):
    st = all_spike_times_obj[unit_idx]
    fr_all[i] = bin_spikes_all_trials(st, go_cue_times, bin_edges)
```

iii. The AI identified spike_times as the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. For each good unit, spike times are binned using `np.histogram` with 50ms bins aligned to the go cue. Spike counts are divided by bin width to convert to firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_spikes_all_trials(spike_times_unit, go_cue_times, bin_edges):
    bin_width = bin_edges[1] - bin_edges[0]
    fr = np.zeros((n_trials, n_bins), dtype=np.float32)
    for t in range(n_trials):
        gc = go_cue_times[t]
        mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
        if np.any(mask):
            aligned = spike_times_unit[mask] - gc
            counts, _ = np.histogram(aligned, bins=bin_edges)
            fr[t] = counts / bin_width
    return fr
```

iii. From CONVERSION_NOTES.md: "Bin spikes in 50ms bins, aligned to go cue, window [-2.5, 1.5]s - 80 time bins per trial."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. Sessions with zero good units are dropped. No additional firing rate threshold or other quality metric is applied.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)

if n_good == 0:
    io.close()
    return None
```

iii. From CONVERSION_NOTES.md: "Use `classification == 'good'` (classifier QC). Do NOT apply the 2 Hz firing rate threshold from the method paper, as that was specific to video prediction analysis, not a general data quality criterion."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time and then histogramming into the relative bin edges `[-2.5, 1.5]s`.

ii.
```python
for t in range(n_trials):
    gc = go_cue_times[t]
    mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
    if np.any(mask):
        aligned = spike_times_unit[mask] - gc
        counts, _ = np.histogram(aligned, bins=bin_edges)
```

iii. Go cue alignment follows the instructions directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins spanning [-2.5, 1.5]s relative to the go cue, yielding 80 time bins per trial. The bin edges are computed with `np.linspace`.

ii.
```python
BEGIN_TIME = -2.5
END_TIME = 1.5
BIN_WIDTH = 0.05
n_bins = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
bin_edges = np.linspace(BEGIN_TIME, END_TIME, n_bins + 1)
```

iii. From CONVERSION_NOTES.md: "50ms non-overlapping bins (as specified by decoder task)."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times` (tone onset times) and `go_start_times` (go cue times). The last sample_start before each trial's go cue is taken as the tone onset for that trial.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
# ...
for t, trial_idx in enumerate(valid_trial_indices):
    gc = go_cue_times_all[trial_idx]
    candidates = sample_start_times[sample_start_times <= gc + 0.01]
    if len(candidates) > 0:
        tone_onsets_rel[t] = candidates[-1] - gc
    else:
        tone_onsets_rel[t] = -1.85  # fallback
```

iii. The AI takes the last sample_start_time before the go cue (with a 0.01s tolerance), recognizing that early licks can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is expressed relative to the go cue. Then for each time bin, the time from tone onset is computed as the bin center minus the tone-relative-to-go offset.

ii.
```python
tone_onsets_rel[t] = candidates[-1] - gc  # negative value
# ...
time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
```

iii. `bin_centers - tone_onsets_rel` = `bin_centers - (tone - go)` = `bin_centers + (go - tone)`, which is equivalent to the reference computation.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for the neural data are used for the time_from_tone computation. Both share the same go-cue-relative time grid.

ii.
```python
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
# Used for both neural binning and time_from_tone
time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
```

iii. Alignment is inherent because both use the same bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `BehavioralEvents/photostim_start_times` and `photostim_stop_times` (absolute session times of photostimulation events).

ii.
```python
has_photostim = 'photostim_start_times' in be.time_series
if has_photostim:
    ps_start_times = be.time_series['photostim_start_times'].timestamps[:]
    ps_stop_times = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The AI used the BehavioralEvents photostim timestamps rather than the per-trial `photostim_onset`/`photostim_duration` fields in the trials table. Both encode the same information but in different form.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, each photostim event's start/stop is converted to go-cue-relative time. A bin is 1.0 if its center falls between the start and stop of any photostim event, 0 otherwise. The result is clipped to [0, 1].

ii.
```python
for t in range(n_valid_trials):
    gc = go_cue_times[t]
    for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
        ps_s_rel = ps_s - gc
        ps_e_rel = ps_e - gc
        if ps_s_rel < bin_edges[-1] and ps_e_rel > bin_edges[0]:
            photostim_all[t] += ((bin_centers >= ps_s_rel) & (bin_centers < ps_e_rel)).astype(np.float32)
    photostim_all = np.clip(photostim_all, 0, 1)
```

iii. The AI uses a binary time-varying representation. However, it iterates over ALL photostim events for ALL trials (a many-to-many search), which is less efficient and different from the reference approach (which uses per-trial onset/duration from the trials table).

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim start/stop times are converted to go-cue-relative time and compared against the same bin centers used for neural data.

ii.
```python
ps_s_rel = ps_s - gc
ps_e_rel = ps_e - gc
photostim_all[t] += ((bin_centers >= ps_s_rel) & (bin_centers < ps_e_rel)).astype(np.float32)
```

iii. Uses the same go-cue-relative bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table.

ii.
```python
instr = trial_instruction[trial_idx]
out = outcome_arr[trial_idx]
if out == 'hit':
    choices[t] = 0 if instr == 'left' else 1
elif out == 'miss':
    choices[t] = 1 if instr == 'left' else 0
else:
    choices[t] = 2  # no lick
```

iii. Choice is derived from instruction and outcome since the actual lick direction is not stored directly. Hit means licked the instructed side; miss means licked the opposite side; ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Per-trial value is repeated across all 80 time bins.

ii.
```python
output_data = np.stack([
    np.full(n_bins, int(choices[t]), dtype=np.int64),
    # ...
], axis=0)
```

iii. From CONVERSION_NOTES.md: "left=0, right=1, no_lick=2".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains strings 'ignore', 'miss', 'hit'.

ii.
```python
outcome_arr = nwb.trials['outcome'][:]
# ...
outcomes[t] = outcome_map.get(out, 0)
```

iii. The outcome is stored directly in the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped as ignore=0, miss=1, hit=2. Per-trial value repeated across all 80 time bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes[t] = outcome_map.get(out, 0)
# ...
np.full(n_bins, int(outcomes[t]), dtype=np.int64)
```

iii. Direct categorical encoding matching the instruction categories.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains 'early' and 'no early'.

ii.
```python
early_lick_arr = nwb.trials['early_lick'][:]
# ...
early_vals[t] = 1 if early_lick_arr[trial_idx] == 'early' else 0
```

iii. The early lick flag is stored directly in the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped as 0=no early, 1=early. Per-trial value repeated across all 80 time bins.

ii.
```python
early_vals[t] = 1 if early_lick_arr[trial_idx] == 'early' else 0
# ...
np.full(n_bins, int(early_vals[t]), dtype=np.int64)
```

iii. Simple binary encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has columns (tongue_x, tongue_y, tongue_likelihood) with timestamps. Column 1 is y-position, column 2 is likelihood.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. From CONVERSION_NOTES.md: "tongue_y from TongueTracking".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes per-session percentiles (40th and 60th) of the tongue y-position for all frames with likelihood >= 0.9 (across the whole session, on raw frames not bin means). Then for each trial and time bin, frames are assigned to bins, and if the average likelihood of frames in a bin is >= 0.9, the mean y of ALL frames in that bin (regardless of individual frame likelihood) is discretized into 0 (<p40), 1 (p40-p60), 2 (>p60). Otherwise the bin is 3 (not visible).

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9

visible_mask_all = tongue_data[:, 2] >= TONGUE_LIKELIHOOD_THRESH
visible_y = tongue_data[visible_mask_all, 1]
tongue_y_p40 = np.percentile(visible_y, 40)
tongue_y_p60 = np.percentile(visible_y, 60)
# ...
for b in range(n_bins):
    frame_mask = bin_idx == b
    frames = trial_data[frame_mask]
    avg_likelihood = np.mean(frames[:, 2])
    if avg_likelihood >= likelihood_thresh:
        avg_y = np.mean(frames[:, 1])
        if avg_y < tongue_y_p40:
            result[t, b] = 0
        elif avg_y <= tongue_y_p60:
            result[t, b] = 1
        else:
            result[t, b] = 2
```

iii. From CONVERSION_NOTES.md: "Use likelihood threshold of 0.9 (DeepLabCut convention). Tongue with likelihood < 0.9 is 'not visible' (category 3)." and "Compute 40th and 60th percentiles over all visible tongue y-positions in the session."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles (40th and 60th) of visible y-positions define two thresholds. Each bin's mean y is compared against these thresholds: 0 if below p40, 1 if between p40 and p60 (inclusive), 2 if above p60, 3 if not visible.

ii. (Same as 8-b above)

iii. The AI uses `<` for the p40 boundary and `<=` for p60, which means the p40 boundary itself falls in category 1. The reference uses `np.digitize` with edges at [p40, p60], which gives: 0 for values < p40, 1 for p40 <= x < p60, 2 for x >= p60.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are in the same session-absolute time as spikes and go cues. Frames within the trial window are found via `searchsorted`, assigned to bins using the same go-cue-relative bin edges, and averaged within each bin.

ii.
```python
abs_edges = gc + bin_edges
i_start = np.searchsorted(tongue_timestamps, abs_edges[0])
i_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
# ...
bin_idx = np.searchsorted(abs_edges, trial_ts, side='right') - 1
```

iii. Aligned via the same go-cue-relative bin grid used for neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good units (classification is NaN or non-'good') are dropped. (2) Trials outside obs_intervals or with auto_water/free_water are excluded. (3) Tongue bins where average likelihood is below 0.9 get category 3 (not visible).

ii.
```python
if n_good == 0:
    io.close()
    return None
# ...
valid_trial_mask = (auto_water == 0) & (free_water == 0)
# ...
if avg_likelihood >= likelihood_thresh:
    # discretize
else:
    # remains 3 (not visible)
```

iii. From CONVERSION_NOTES.md: "1 session skipped... no valid trials within obs_intervals" and the tongue handling uses the not-visible category.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file (especially spike_times and tongue tracking data) and the spike binning loop (per-unit, per-trial histogram computation). The AI reports ~3.7s/session for sample and processes 173 sessions.

ii. N/A (timing is reported in print statements)

iii. From CONVERSION_NOTES.md: "Processing time: ~3.7s/session".

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two nested loops remain: (1) the spike binning loops over units AND trials (`for t in range(n_trials)` inside `bin_spikes_all_trials`, called per unit); (2) the tongue binning loops over trials AND bins (`for b in range(n_bins)` inside the trial loop). The reference vectorizes the trial dimension for spikes by flattening all trial edges into one `searchsorted` call.

ii.
```python
# Spike binning: per-trial loop
for t in range(n_trials):
    gc = go_cue_times[t]
    mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
    if np.any(mask):
        aligned = spike_times_unit[mask] - gc
        counts, _ = np.histogram(aligned, bins=bin_edges)
        fr[t] = counts / bin_width

# Tongue binning: per-trial, per-bin loop
for t in range(n_trials):
    # ...
    for b in range(n_bins):
        frame_mask = bin_idx == b
```

iii. The per-trial spike loop could be vectorized by computing all edges at once as done in the reference.

## 10-c. What processing does the code repeat multiple times?

i. The photostim computation iterates over ALL session photostim events for every trial, even though each trial has at most one photostim event. This is redundant work.

ii.
```python
for t in range(n_valid_trials):
    gc = go_cue_times[t]
    for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
        # checks every photostim event for every trial
```

iii. The reference avoids this by using per-trial photostim_onset/duration from the trials table.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `classify_brain_region` function performs elaborate hierarchical brain region mapping (~170 lines of classification logic) that maps CCF annotations to broad categories. The reference simply takes the first part of the annotation name before any comma/period/slash. Additionally, the AI reads `electrode_group` data for fallback region classification, which adds I/O overhead.

ii.
```python
def classify_brain_region(anno_name):
    """Map a CCF annotation name to a broad brain region."""
    # ~140 lines of if/elif logic
    ...

def get_brain_region_for_unit(anno_name, electrode_group_location):
    # Falls back to electrode_group JSON parsing
    ...
```

iii. The broad region categories are stored but the downstream decoder uses PCA on the neural data directly, so the brain region labels serve only as metadata/documentation.
