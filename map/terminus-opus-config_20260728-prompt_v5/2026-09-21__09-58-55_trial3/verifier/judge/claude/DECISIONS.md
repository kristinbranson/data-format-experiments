# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by globbing `sub-*/*.nwb` under the data directory, sorting them, and iterating over each file. It uses `h5py` (not `pynwb`) to directly read HDF5 groups for speed. Each session's units, trials, behavioral events, and tongue tracking are extracted from the HDF5 structure.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
def load_session_h5py(nwb_path):
    with h5py.File(nwb_path, 'r') as f:
        data['subject_id'] = f['general']['subject']['subject_id'][()].decode() ...
        units_grp = f['units']
        data['classification'] = np.array([x.decode() ...])
        spike_times_data = units_grp['spike_times'][()]
        spike_times_index = units_grp['spike_times_index'][()]
        trials_grp = f['intervals']['trials']
        ...
```

iii. The AI chose h5py over pynwb for a reported 12x speedup. CONVERSION_NOTES.md states: "h5py direct loading instead of pynwb (0.2s vs 24s per session)".

## 1-b. How are the data split into subjects?

i. Each NWB file's `subject_id` is read from `f['general']['subject']['subject_id']`. As sessions are processed, unique subject IDs are accumulated in order of first appearance. A mapping from subject ID to index is maintained.

ii.
```python
data['subject_id'] = f['general']['subject']['subject_id'][()].decode() ...

subj_id = result['subject_id']
if subj_id not in subject_to_idx:
    subject_to_idx[subj_id] = len(subjects_list)
    subjects_list.append(subj_id)
```

iii. The AI uses the numeric subject_id from the NWB file. Subjects are accumulated in order of first encounter rather than sorted.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The AI iterates over sorted NWB file paths. Sessions that have 0 good units or fewer than 2 valid trials are skipped. This results in 173 sessions from 174 files.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
    if result is None:
        continue
```

iii. CONVERSION_NOTES.md states: "174 NWB files total" and "1 session has 0 good units, skip it -> 173 sessions".

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`f['intervals']['trials']`). Go cue times come from `BehavioralEvents/go_start_times`. The AI reads all trial columns (start_time, stop_time, trial_instruction, outcome, early_lick) and go_times from behavioral events.

ii.
```python
trials_grp = f['intervals']['trials']
data['trial_start'] = trials_grp['start_time'][()]
data['trial_stop'] = trials_grp['stop_time'][()]
...
data['go_times'] = be['go_start_times']['timestamps'][()]
```

iii. No explicit assertion that the number of go cues matches the number of trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on neural recording coverage only: a trial is valid if its analysis window `[go_time + align_start, go_time + align_end]` falls within the overall min/max spike times across all good units (with a 1-bin buffer). No `obs_intervals` filter is used. No `free_water` filter is applied. Sessions with fewer than 2 valid trials are dropped.

ii.
```python
def get_valid_trial_mask(spike_times_list, go_times, ...):
    max_spike = 0
    min_spike = float('inf')
    for spikes in spike_times_list:
        if len(spikes) > 0:
            max_spike = max(max_spike, spikes[-1])
            min_spike = min(min_spike, spikes[0])
    valid = (go_times + align_start >= min_spike - BIN_WIDTH) & \
            (go_times + align_end <= max_spike + BIN_WIDTH)
    return valid
```

iii. CONVERSION_NOTES.md states: "Filter trials where neural recording doesn't cover the analysis window" and "Each unit has is_good_trials but this varies per unit; we use recording coverage instead". The AI did not use `obs_intervals` or filter `free_water` trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` (the ragged array). Only units with `classification == 'good'` contribute. Go cue times from `BehavioralEvents/go_start_times` define the alignment.

ii.
```python
spike_times_data = units_grp['spike_times'][()]
spike_times_index = units_grp['spike_times_index'][()]
...
good_mask = data['classification'] == 'good'
spike_times_good = [data['all_spike_times'][i] for i in good_indices]
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins spanning -2.5s to +1.5s relative to go cue (80 bins). For each trial and each neuron, `np.histogram` computes spike counts per bin, then divides by bin width to get firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start

for t in range(n_trials):
    go_time = go_times[t]
    abs_bin_edges = bin_edges_rel + go_time
    for i, spikes in enumerate(spike_times_list):
        ...
        counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
        fr[i] = counts / bin_width
    results.append(fr)
```

iii. CONVERSION_NOTES.md states: "Vectorized spike binning using np.histogram". The processing matches the instructions (50ms bins, Hz rates).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. No additional metric thresholds. Sessions with 0 good units are dropped. This yields 69,453 good units across 173 sessions.

ii.
```python
good_mask = data['classification'] == 'good'
n_good = np.sum(good_mask)
if n_good == 0:
    print(f"  Skipping {basename}: no good units")
    return None
good_indices = np.where(good_mask)[0]
```

iii. CONVERSION_NOTES.md: "Use classifier-based QC: classification == 'good' in NWB files". Matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are defined relative to go cue onset. For each trial, absolute bin edges are computed as `bin_edges_rel + go_time`. Spikes are histogrammed against these absolute edges.

ii.
```python
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
...
abs_bin_edges = bin_edges_rel + go_time
counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
```

iii. Alignment to go cue matches the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins spanning -2.5s to +1.5s relative to go cue. No rebinning from a finer resolution; spikes are binned directly at 50ms.

ii.
```python
BIN_WIDTH = 0.05  # 50ms bins
ALIGN_START = -2.5
ALIGN_END = 1.5
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
```

iii. Matches the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onset timestamps) in BehavioralEvents, and the go cue times. The AI finds the **first** sample_start_time within each trial's `[trial_start, trial_stop]` window.

ii.
```python
data['sample_starts'] = be['sample_start_times']['timestamps'][()]
...
def find_sample_starts_for_trials(trial_starts, trial_stops, sample_starts, go_times):
    for t in range(n_trials):
        idx_start = np.searchsorted(sample_starts, trial_starts[t])
        idx_end = np.searchsorted(sample_starts, trial_stops[t])
        if idx_start < idx_end:
            tone_rel_go[t] = sample_starts[idx_start] - go_times[t]
    return tone_rel_go
```

iii. The AI uses the first sample_start within the trial, not the last one before the go cue as the reference does. On early-lick trials where the sample epoch replays, the first onset is the original tone; the reference takes the last onset (after replay).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset time relative to go cue is computed for each trial. Then for each time bin, the value is `bin_center_time - tone_relative_to_go`, giving time from tone onset in seconds.

ii.
```python
bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
...
time_from_tone = (bin_centers_rel - tone_rel_go[t]).astype(np.float32)
```

iii. The formula is correct: `bin_center_relative_to_go - tone_relative_to_go = time_from_tone_onset`.

## 3-c. How is `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and the time-from-tone input use the same bin grid defined relative to the go cue. Bin centers are at `align_start + bin_width/2, align_start + 3*bin_width/2, ...`.

ii.
```python
bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
```

iii. Alignment is inherent — same time grid for neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `photostim_stop_times` timestamps. If these time series don't exist, empty arrays are used.

ii.
```python
if 'photostim_start_times' in be:
    data['photostim_starts'] = be['photostim_start_times']['timestamps'][()]
    data['photostim_stops'] = be['photostim_stop_times']['timestamps'][()]
else:
    data['photostim_starts'] = np.array([])
    data['photostim_stops'] = np.array([])
```

iii. The AI uses the event-based photostim timestamps from BehavioralEvents, not the per-trial `photostim_onset`/`photostim_duration` from the trials table.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI checks all session-wide photostim start/stop events. A bin center is marked as 1.0 if it falls within any `[photostim_start, photostim_stop)` interval.

ii.
```python
photostim = np.zeros(n_bins, dtype=np.float32)
abs_bin_centers = bin_centers_rel + go_time
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
    mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
    photostim[mask] = 1.0
```

iii. This iterates over ALL photostim events for each trial (not just the trial's own stimulation), but since photostim events typically only overlap with the correct trial window, the result is functionally equivalent.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The same bin center grid (relative to go cue) is used for photostim as for neural data, with absolute bin centers computed as `bin_centers_rel + go_time`.

ii.
```python
abs_bin_centers = bin_centers_rel + go_time
```

iii. Same time grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the actual lick event timestamps: `left_lick_times` and `right_lick_times` from BehavioralEvents. The first lick within 1.5s after the go cue determines the choice.

ii.
```python
data['left_lick_times'] = be['left_lick_times']['timestamps'][()]
data['right_lick_times'] = be['right_lick_times']['timestamps'][()]
...
def get_lick_choices(go_times, left_lick_times, right_lick_times, response_window=1.5):
    for t, go_time in enumerate(go_times):
        left_idx = np.searchsorted(left_lick_times, go_time)
        first_left = left_lick_times[left_idx] if left_idx < len(left_lick_times) and \
                     left_lick_times[left_idx] < go_time + response_window else float('inf')
        right_idx = np.searchsorted(right_lick_times, go_time)
        first_right = right_lick_times[right_idx] if right_idx < len(right_lick_times) and \
                      right_lick_times[right_idx] < go_time + response_window else float('inf')
        if first_left < first_right:
            choices[t] = 0
        elif first_right < float('inf'):
            choices[t] = 1
```

iii. CONVERSION_NOTES.md: "Choice determination: First lick direction within 1.5s after go cue". This differs from the reference, which derives choice from `trial_instruction` x `outcome`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no lick. The value is per-trial and replicated across all 80 time bins.

ii.
```python
choices = np.full(len(go_times), 2, dtype=np.int64)  # default: no lick
...
output_data[0, :] = choices[t]
```

iii. The coding scheme matches the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which contains 'ignore', 'miss', 'hit'.

ii.
```python
data['outcome'] = np.array([x.decode() ... for x in trials_grp['outcome'][()]])
...
outcome_codes = get_outcome_codes(outcome_valid)
```

iii. Direct from trials table, same as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Per-trial value replicated across all 80 bins.

ii.
```python
def get_outcome_codes(outcomes):
    mapping = {'ignore': 0, 'miss': 1, 'hit': 2}
    return np.array([mapping.get(o, 0) for o in outcomes], dtype=np.int64)
...
output_data[1, :] = outcome_codes[t]
```

iii. Matches the reference encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
data['early_lick'] = np.array([x.decode() ... for x in trials_grp['early_lick'][()]])
```

iii. Same source as reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: 'no early'=0, otherwise=1. Per-trial value replicated across all 80 bins.

ii.
```python
def get_early_lick_codes(early_licks):
    return np.array([0 if e == 'no early' else 1 for e in early_licks], dtype=np.int64)
...
output_data[2, :] = early_lick_codes[t]
```

iii. Matches the reference encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, which has columns (x, y, confidence) and timestamps.

ii.
```python
data['tongue_data'] = bts['Camera0_side_TongueTracking']['data'][()]
data['tongue_timestamps'] = bts['Camera0_side_TongueTracking']['timestamps'][()]
...
tongue_y = tongue_data[:, 1]
tongue_conf = tongue_data[:, 2]
```

iii. Same source as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with confidence > 0.5 are considered visible. Session-wide percentiles (40th, 60th) are computed on **raw visible frames** (not bin means). For each trial bin, if >50% of frames are visible, the mean y of visible frames is discretized: <p40 → 0, p40-p60 → 1, >p60 → 2. Otherwise → 3 (not visible).

ii.
```python
visible_mask = tongue_conf > TONGUE_CONFIDENCE_THRESHOLD
y_visible = tongue_y[visible_mask]
p40 = np.percentile(y_visible, 40)
p60 = np.percentile(y_visible, 60)
...
for b in range(n_bins):
    ...
    visible_frac = np.mean(bin_conf > TONGUE_CONFIDENCE_THRESHOLD)
    if visible_frac > 0.5:
        vis_mask = bin_conf > TONGUE_CONFIDENCE_THRESHOLD
        mean_y = np.mean(bin_y[vis_mask])
        if mean_y < p40:
            trial_tongue_y[b] = 0
        elif mean_y < p60:
            trial_tongue_y[b] = 1
        else:
            trial_tongue_y[b] = 2
```

iii. Two differences from reference: (1) percentiles are computed on raw frames not 50ms bin means, (2) bins require >50% visible frames (reference: any visible frame counts). Also uses `<` for threshold boundaries rather than `np.digitize`.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses explicit comparisons: `mean_y < p40` → 0, `mean_y < p60` → 1, else → 2, with 3 for not-visible bins. This means the boundary cases differ slightly from using `np.digitize` (which the reference uses).

ii.
```python
if mean_y < p40:
    trial_tongue_y[b] = 0
elif mean_y < p60:
    trial_tongue_y[b] = 1
else:
    trial_tongue_y[b] = 2
```

iii. Values exactly at p40 go to class 1 (not 0), and values exactly at p60 go to class 2 (not 1). This is equivalent to `np.digitize` with `right=False` (the default), so the boundary behavior matches the reference.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The same bin grid is used. For each trial and bin, frames within `[abs_bin_edge[b], abs_bin_edge[b+1])` are found using `searchsorted` on the tongue timestamps.

ii.
```python
for b in range(n_bins):
    idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
    idx_end = np.searchsorted(tongue_timestamps, abs_bin_edges[b + 1])
```

iii. Same temporal grid ensures alignment with neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Sessions with 0 good units**: Skipped (returns None).
- **Trials without neural coverage**: Filtered by checking if the analysis window falls within the recording period (min/max spike times).
- **Tongue not visible**: Bins with <50% visible frames default to category 3 (not visible).
- **Missing photostim data**: If photostim time series don't exist in the file, empty arrays are used.

ii.
```python
if n_good == 0:
    return None
...
valid = (go_times + align_start >= min_spike - BIN_WIDTH) & \
        (go_times + align_end <= max_spike + BIN_WIDTH)
...
if 'photostim_start_times' in be:
    ...
else:
    data['photostim_starts'] = np.array([])
```

iii. The AI does not handle the case of NaN/non-string values in `classification` (which the reference handles with `_text()`). For the one session where classification is NaN, h5py may decode it differently than pynwb.

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion takes 547s (~9.1 minutes) for 174 sessions. Loading data is fast (~0.2-0.6s per session with h5py). The neural firing rate computation dominates (~1-3s per session), since it loops over trials and neurons. Tongue processing adds ~0.1-0.3s. Pickle output is also substantial for the 11.8 GB file.

ii. N/A

iii. CONVERSION_NOTES.md: "h5py direct loading instead of pynwb (0.2s vs 24s per session)".

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two nested loops could be vectorized:
1. The firing rate computation loops over trials (outer) and neurons (inner) with `np.histogram` per neuron per trial.
2. The tongue discretization loops over trials (outer) and bins (inner).

ii.
```python
for t in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
```

```python
for t in range(n_trials):
    ...
    for b in range(n_bins):
        idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
```

iii. The reference vectorizes over trials by flattening all trial edges into one array per neuron (one `searchsorted` per neuron for all trials at once). The AI's double loop (trials x neurons) is significantly slower.

## 10-c. What processing does the code repeat multiple times?

i. The bin edge computation `bin_edges_rel + go_time` is repeated for neural, input, and tongue processing per trial, rather than being computed once. The searchsorted for tongue timestamps is done per-bin rather than per-trial.

ii.
```python
# In compute_firing_rates_all_trials:
abs_bin_edges = bin_edges_rel + go_time
# In compute_inputs_all_trials:
abs_bin_centers = bin_centers_rel + go_time
# In compute_tongue_y_all_trials:
abs_bin_edges = bin_edges_rel + go_time
```

iii. Minor redundancy; the absolute bin edges for each trial are computed multiple times across different functions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `subject_desc`, `tongue_data` (all 3 columns including x), `left_lick_times`, and `right_lick_times` from every session. The lick times are used for choice computation, which is a design choice. The subject description and tongue x-position are loaded but not used in the final output. The coarse brain region mapping is an elaborate function that could be simplified.

ii.
```python
data['subject_desc'] = f['general']['subject']['description'][()].decode() ...
```

iii. Minor overhead from loading unused fields.
