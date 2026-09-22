# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all sessions by globbing `'/app/data/sub-*/*.nwb'`, sorting the file list, and reading each NWB file directly with `h5py`. Within each file it pulls subject metadata, unit tables, trial tables, behavioral event timestamps, and tongue-tracking time series into a plain Python dict.

ii.
```python
def load_session_h5py(nwb_path):
    with h5py.File(nwb_path, 'r') as f:
        data['subject_id'] = f['general']['subject']['subject_id'][()]
        units_grp = f['units']
        trials_grp = f['intervals']['trials']
        be = f['acquisition']['BehavioralEvents']
        data['go_times'] = be['go_start_times']['timestamps'][()]
        data['sample_starts'] = be['sample_start_times']['timestamps'][()]
        data['tongue_data'] = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][()]
```

```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI explicitly justifies this as a speed optimization over `pynwb`, stating that `h5py` gave about a 12x speedup and still exposed the same NWB contents.

## 1-b. How are the data split into subjects?

i. Subjects are split using each file’s `general/subject/subject_id`. The AI builds `subjects_list` incrementally in first-seen session order and records a session-level `subject_idx`.

ii.
```python
data['subject_id'] = f['general']['subject']['subject_id'][()]
...
if subj_id not in subject_to_idx:
    subject_to_idx[subj_id] = len(subjects_list)
    subjects_list.append(subj_id)
...
'subjects': subjects_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. The notes treat the NWB subject field as the canonical mouse identifier and report that this yields 28 subjects, matching the dataset summary.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Sessions are processed in sorted path order, and each surviving file contributes one session entry to `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. `CONVERSION_NOTES.md` Step 2 describes the dataset as one NWB file per session under each subject directory, so the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table as the trial structure and indexes per-trial arrays by the trial row order. It assumes `go_times` are aligned row-for-row with the trial table and then subsets all trial-level arrays with `valid_indices`.

ii.
```python
data['trial_start'] = trials_grp['start_time'][()]
data['trial_stop'] = trials_grp['stop_time'][()]
...
n_trials = len(data['trial_start'])
go_times = data['go_times']
...
valid_indices = np.where(valid_mask)[0]
go_times_valid = go_times[valid_indices]
trial_starts_valid = data['trial_start'][valid_indices]
trial_stops_valid = data['trial_stop'][valid_indices]
```

iii. The trajectory shows the AI inspected the trial table and behavioral event counts and concluded trials were naturally represented by the NWB trial rows.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials using a session-wide neural-coverage heuristic: it keeps trials whose aligned window `[go-2.5, go+1.5]` fits between the earliest and latest spike time across all good units. It does not use `obs_intervals`, does not filter `free_water`, and only drops a session if fewer than 2 trials survive.

ii.
```python
def get_valid_trial_mask(spike_times_list, go_times, align_start=ALIGN_START, align_end=ALIGN_END):
    max_spike = 0
    min_spike = float('inf')
    for spikes in spike_times_list:
        if len(spikes) > 0:
            max_spike = max(max_spike, spikes[-1])
            min_spike = min(min_spike, spikes[0])
    valid = (go_times + align_start >= min_spike - BIN_WIDTH) & (go_times + align_end <= max_spike + BIN_WIDTH)
    return valid
...
if n_valid < 2:
    return None
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 67-68, the AI says it chose this rule to exclude trials whose neural recording did not cover the full decoder window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times` after filtering units by `classification == 'good'`, with `go_start_times` used to place the aligned bins.

ii.
```python
data['classification'] = ...
spike_times_data = units_grp['spike_times'][()]
spike_times_index = units_grp['spike_times_index'][()]
...
good_mask = data['classification'] == 'good'
spike_times_good = [data['all_spike_times'][i] for i in good_indices]
go_times = data['go_times']
```

iii. The notes repeatedly describe the neural stream as spike times in absolute session time plus go-cue timestamps for alignment.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times to 50 ms firing rates by histogramming each good unit’s spikes into trial-specific bins aligned to the go cue and dividing counts by bin width. It does not smooth, normalize, or baseline-subtract.

ii.
```python
for t in range(n_trials):
    abs_bin_edges = bin_edges_rel + go_time
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    for i, spikes in enumerate(spike_times_list):
        left = np.searchsorted(spikes, abs_bin_edges[0])
        right = np.searchsorted(spikes, abs_bin_edges[-1])
        spikes_in_window = spikes[left:right]
        counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
        fr[i] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` Step 5 says the neural target is “Bin into 50ms firing rates, align to go cue, window [-2.5, 1.5]”.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit curation is just `classification == 'good'`. Sessions with zero such units are skipped. No additional metric thresholds are applied.

ii.
```python
good_mask = data['classification'] == 'good'
n_good = np.sum(good_mask)
if n_good == 0:
    print(f"  Skipping {basename}: no good units")
    return None
```

iii. The notes explicitly tie this to the reference classifier-based QC and contrast it with other possible quality fields.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset by adding the fixed relative bin edges `[-2.5, 1.5]` to each trial’s absolute go-cue timestamp, then binning spikes against those absolute edges.

ii.
```python
BIN_WIDTH = 0.05
ALIGN_START = -2.5
ALIGN_END = 1.5
...
go_time = go_times[t]
abs_bin_edges = bin_edges_rel + go_time
counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
```

iii. The notes say the task requires go-cue alignment and describe NWB spike and event times as already sharing the same absolute session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 80 non-overlapping bins of width 50 ms from -2.5 s to +1.5 s around go cue. This is a rebinning of spike times into fixed-width rate bins.

ii.
```python
BIN_WIDTH = 0.05
ALIGN_START = -2.5
ALIGN_END = 1.5
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
```

iii. `CONVERSION_NOTES.md` Step 5 says this was chosen to follow the decoder specification rather than the 40 ms / 3.4 ms video stride used in the reference preprocessing code.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times` together with each trial’s `start_time`, `stop_time`, and `go_time`. It searches for sample starts that fall within the trial bounds.

ii.
```python
def find_sample_starts_for_trials(trial_starts, trial_stops, sample_starts, go_times):
    ...
    idx_start = np.searchsorted(sample_starts, trial_starts[t])
    idx_end = np.searchsorted(sample_starts, trial_stops[t])
    if idx_start < idx_end:
        tone_rel_go[t] = sample_starts[idx_start] - go_times[t]
```

iii. In Step 5 of the notes, the AI frames this as “continuous: t - tone_onset for each timepoint,” and the trajectory shows it reasoned from `sample_start_times` as the tone source.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI picks the first `sample_start_times` event inside that trial, converts it to a go-relative offset, defaults missing trials to `-1.85`, and then computes time-from-tone as `bin_center_rel - tone_rel_go`.

ii.
```python
tone_rel_go = np.full(n_trials, -1.85, dtype=np.float64)
...
if idx_start < idx_end:
    tone_rel_go[t] = sample_starts[idx_start] - go_times[t]
...
time_from_tone = (bin_centers_rel - tone_rel_go[t]).astype(np.float32)
```

iii. The notes justify the overall representation as a continuous time-varying input. The trajectory indicates the AI focused on identifying a trial-local tone onset, but it did not document the consequences of repeated sample epochs after early licks.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI uses the same 50 ms go-cue-centered bin centers as the neural data. `time_from_tone` is evaluated once per neural time bin per trial.

ii.
```python
bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
...
time_from_tone = (bin_centers_rel - tone_rel_go[t]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says this variable is “time-varying,” and the code computes it on the same aligned bin grid used for the neural rates.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the global `BehavioralEvents` streams `photostim_start_times` and `photostim_stop_times`, not from the trial-table `photostim_onset` / `photostim_duration` fields.

ii.
```python
if 'photostim_start_times' in be:
    data['photostim_starts'] = be['photostim_start_times']['timestamps'][()]
    data['photostim_stops'] = be['photostim_stop_times']['timestamps'][()]
else:
    data['photostim_starts'] = np.array([])
    data['photostim_stops'] = np.array([])
```

iii. The notes say photostim is a binary time-varying input based on behavioral-event photostim start/stop times, and trajectory step 43 describes it as “whether photostim is active.”

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI computes absolute bin centers and marks a bin as 1 if its center falls inside any session-level photostim interval; otherwise it stays 0. This yields a binary time series per trial.

ii.
```python
photostim = np.zeros(n_bins, dtype=np.float32)
abs_bin_centers = bin_centers_rel + go_time
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
    mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
    photostim[mask] = 1.0
```

iii. The notes justify photostim as a binary per-timepoint decoder input, and the AI chose absolute event times to avoid parsing the trial-table string onset values.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim is aligned by comparing absolute photostim start/stop times to the absolute centers of the go-cue-aligned neural bins for that trial.

ii.
```python
abs_bin_centers = bin_centers_rel + go_time
mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
```

iii. The same bin-center axis is used for `time_from_tone`, photostim, and neural rates, so the AI’s stated intent was a shared aligned timeline.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `left_lick_times` and `right_lick_times` in `BehavioralEvents`, not from `trial_instruction` plus `outcome`.

ii.
```python
data['left_lick_times'] = be['left_lick_times']['timestamps'][()]
data['right_lick_times'] = be['right_lick_times']['timestamps'][()]
...
def get_lick_choices(go_times, left_lick_times, right_lick_times, response_window=LICK_RESPONSE_WINDOW):
```

iii. In trajectory steps 42-43, the AI explicitly states that choice should be “determined by first lick direction after go cue.”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the AI finds the first left and right lick after the go cue within a 1.5 s response window, chooses the earlier one as left/right, and assigns `2` if neither occurs. That categorical choice is then repeated across all 80 bins.

ii.
```python
choices = np.full(len(go_times), 2, dtype=np.int64)
...
first_left = ...
first_right = ...
if first_left < first_right:
    choices[t] = 0
elif first_right < float('inf'):
    choices[t] = 1
...
output_data[0, :] = choices[t]
```

iii. The notes and trajectory justify this as the most direct behavioral readout of actual lick direction after the response cue.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trial-table `outcome` column.

ii.
```python
data['outcome'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in trials_grp['outcome'][()]])
...
outcome_valid = data['outcome'][valid_indices]
```

iii. The AI recognized this as an already-categorical trial variable matching the requested decoder target.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the resulting per-trial code across all time bins.

ii.
```python
def get_outcome_codes(outcomes):
    mapping = {'ignore': 0, 'miss': 1, 'hit': 2}
    return np.array([mapping.get(o, 0) for o in outcomes], dtype=np.int64)
...
output_data[1, :] = outcome_codes[t]
```

iii. `CONVERSION_NOTES.md` Step 5 lists exactly this mapping for the decoder output.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` column.

ii.
```python
data['early_lick'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in trials_grp['early_lick'][()]])
...
early_lick_valid = data['early_lick'][valid_indices]
```

iii. The notes identify `trials.early_lick` as the source and treat it as an already-annotated categorical output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats the resulting per-trial code across all bins.

ii.
```python
def get_early_lick_codes(early_licks):
    return np.array([0 if e == 'no early' else 1 for e in early_licks], dtype=np.int64)
...
output_data[2, :] = early_lick_codes[t]
```

iii. This mapping is explicitly listed in `CONVERSION_NOTES.md` Step 5.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `Camera0_side_TongueTracking`: column 1 of `data` is tongue y, column 2 is confidence, and `timestamps` give frame times.

ii.
```python
data['tongue_data'] = bts['Camera0_side_TongueTracking']['data'][()]
data['tongue_timestamps'] = bts['Camera0_side_TongueTracking']['timestamps'][()]
...
tongue_y = tongue_data[:, 1]
tongue_conf = tongue_data[:, 2]
```

iii. The notes say the side-camera tongue tracker provides `(x, y, confidence)` at video-frame resolution and that low confidence means the tongue is not visible.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI computes session-wide 40th and 60th percentiles from all raw visible tongue-y frames (`confidence > 0.5`). For each trial and each 50 ms bin, it checks whether more than half the frames are visible; if so it averages only visible-frame y values in that bin and classifies the mean against those percentile thresholds. Otherwise the bin is set to `3` (“not visible”).

ii.
```python
visible_mask = tongue_conf > TONGUE_CONFIDENCE_THRESHOLD
y_visible = tongue_y[visible_mask]
if len(y_visible) > 0:
    p40 = np.percentile(y_visible, 40)
    p60 = np.percentile(y_visible, 60)
...
visible_frac = np.mean(bin_conf > TONGUE_CONFIDENCE_THRESHOLD)
if visible_frac > 0.5:
    mean_y = np.mean(bin_y[vis_mask])
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory step 42, the AI justifies this with the confidence threshold and per-session percentile discretization, emphasizing that visible tongue frames are rare.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses `0` for mean y below `p40`, `1` for `p40 <= y < p60`, `2` for `>= p60`, and `3` when visibility in the bin is not strong enough.

ii.
```python
trial_tongue_y = np.full(n_bins, 3, dtype=np.int64)
...
if mean_y < p40:
    trial_tongue_y[b] = 0
elif mean_y < p60:
    trial_tongue_y[b] = 1
else:
    trial_tongue_y[b] = 2
```

iii. The AI states in the notes that `3` means “not visible” and that the visible categories should be based on per-session percentiles.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output by using the same trial-wise go-cue-centered 50 ms bins as the neural data. For each trial, it searches the camera timestamps between each pair of absolute bin edges.

ii.
```python
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
...
abs_bin_edges = bin_edges_rel + go_time
idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
idx_end = np.searchsorted(tongue_timestamps, abs_bin_edges[b + 1])
```

iii. The notes describe tongue tracking as sampled at 0.0034 s stride and intended to be discretized onto the decoder’s 50 ms go-aligned bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data ad hoc. Sessions with no good units are skipped. Trials outside the inferred global recording coverage are dropped. Tongue bins with insufficient visible frames stay at class `3`. If no sample-start event is found inside a trial, `tone_rel_go` falls back to a hard-coded `-1.85` s default.

ii.
```python
if n_good == 0:
    return None
...
valid_mask = get_valid_trial_mask(spike_times_good, go_times)
...
tone_rel_go = np.full(n_trials, -1.85, dtype=np.float64)
...
trial_tongue_y = np.full(n_bins, 3, dtype=np.int64)
```

iii. The notes frame these choices as practical handling for incomplete neural coverage and invisible tongue frames. The trajectory shows the AI focused on keeping decoder training viable rather than preserving a stricter provenance for every missing value.

## 10-a. What are the most time-consuming steps of the code?

i. The AI explicitly optimized file loading and describes direct NWB access as the main bottleneck it addressed. In the final code, the heaviest steps are session loading, per-trial/per-neuron spike histogramming, and per-trial/per-bin tongue processing.

ii.
```python
data = load_session_h5py(nwb_path)
...
neural_trials = compute_firing_rates_all_trials(spike_times_good, go_times_valid)
...
tongue_y_list, p40, p60 = compute_tongue_y_all_trials(
    data['tongue_data'], data['tongue_timestamps'], go_times_valid
)
```

iii. `CONVERSION_NOTES.md` Step 6 says `h5py` was introduced specifically for speed, and the per-session timing printout breaks runtime into `load`, `neural`, `input`, and `output`.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Although the AI describes several functions as “vectorized,” the code still contains important loops that could have been vectorized further: a trial loop around the neural histogramming, a neuron loop inside each trial, a nested trial-by-bin loop for tongue classification, and a trial-by-photostim-interval loop for photostim construction.

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

iii. The AI’s justification, from Step 6 of the notes, is performance-oriented: it believed `h5py`, `searchsorted`, and some batching were enough to meet the runtime budget.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it re-creates aligned bin edges separately for neural, tongue, and input functions; re-scans all photostim intervals for every trial; and re-runs `searchsorted` on tongue timestamps for every bin of every trial.

ii.
```python
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
...
for t in range(n_trials):
    abs_bin_edges = bin_edges_rel + go_time
```

```python
for t in range(n_trials):
    abs_bin_centers = bin_centers_rel + go_time
    for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
        mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
```

iii. The notes do not present these repeats as a deliberate design choice; they follow from the AI’s priority on a working implementation that finished within time.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and/or carries several things that are not kept in the final output or used by downstream decoding: `subject_desc`, `anno_names`, tongue percentiles `p40/p60`, `dt`, `t0_tongue`, and optional processing plots. It also returns `anno_names` and `subject_desc` from `process_session` but drops them when assembling the saved dataset.

ii.
```python
data['subject_desc'] = ...
...
dt = np.median(np.diff(tongue_timestamps[:100]))
t0_tongue = tongue_timestamps[0]
...
return {
    'coarse_regions': coarse_regions,
    'anno_names': anno_names,
    'subject_id': data['subject_id'],
    'subject_desc': data['subject_desc'],
```

```python
output_data = {
    'neural': all_neural,
    ...
    'brain_regions': brain_regions_list,
    'brain_region_idx': all_brain_region_idx,
```

iii. The notes emphasize validation, visualization, and dataset exploration, so some of this extra work appears to come from debugging and inspection rather than the final decoder-format requirements.
