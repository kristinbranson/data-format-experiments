# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent finds every NWB file under `/app/data/sub-*/*.nwb`, sorts the paths, and processes each file once. `load_session_h5py` opens each NWB file directly with `h5py` and reads subject metadata, ragged unit spike times, trial columns, behavioral-event timestamps, and tongue tracking arrays.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, show_processing=show, session_idx=i)
```
```python
with h5py.File(nwb_path, 'r') as f:
    units_grp = f['units']
    trials_grp = f['intervals']['trials']
    acq = f['acquisition']
```

iii. The notes say the dataset contains 174 NWB files in 28 subject directories and describe direct `h5py` loading as a roughly 12-fold speedup over `pynwb` while accessing the same underlying NWB data.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file. During assembly, IDs are added in first-seen file order to `subjects_list`; each retained session receives the corresponding integer in `subject_idx`.

ii.
```python
data['subject_id'] = f['general']['subject']['subject_id'][()].decode() ...
```
```python
if subj_id not in subject_to_idx:
    subject_to_idx[subj_id] = len(subjects_list)
    subjects_list.append(subj_id)
all_subject_idx.append(subject_to_idx[subj_id])
```

iii. The agent treated the NWB subject field as authoritative and checked that the final result had 28 subjects, matching the dataset and papers.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. A session is appended to the output unless it has no classifier-good units or fewer than two trials passing the agent's coverage filter.

ii.
```python
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, show_processing=show, session_idx=i)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The notes report 174 source files and explain that the one file with zero good units is skipped, producing the 173 sessions reported in the paper.

## 1-d. How are the data split into trials?

i. Trial rows come from `/intervals/trials`; go cues come from `BehavioralEvents/go_start_times`. After filtering, the selected trial indices are applied to trial starts, stops, outcome, early-lick labels, and go times, and each retained go cue defines one neural/input/output trial.

ii.
```python
data['trial_start'] = trials_grp['start_time'][()]
data['trial_stop'] = trials_grp['stop_time'][()]
data['go_times'] = be['go_start_times']['timestamps'][()]
...
valid_indices = np.where(valid_mask)[0]
go_times_valid = go_times[valid_indices]
```

iii. The notes identify the NWB trials table and go-cue stream as the trial definitions. The code assumes their ordering and lengths agree; unlike the reference, it does not assert that equality.

## 1-e. How are trials filtered based on quality controls?

i. A trial is retained when its entire `[-2.5, +1.5]` go-cue window falls between the earliest and latest spike observed across any good unit, with a 50-ms allowance. Sessions with fewer than two retained trials are removed. It does not use `obs_intervals` and does not remove `free_water` trials, so it retains 2,446 all-zero trials and produces 93,290 trials versus the reference's 90,860.

ii.
```python
for spikes in spike_times_list:
    if len(spikes) > 0:
        max_spike = max(max_spike, spikes[-1])
        min_spike = min(min_spike, spikes[0])
valid = (go_times + align_start >= min_spike - BIN_WIDTH) & \
        (go_times + align_end <= max_spike + BIN_WIDTH)
```

iii. The agent justified this as filtering trials whose analysis windows lack neural recording coverage. Its later investigation found many zero-spike trials and found `free_water` trials, but concluded the zeroes were genuine quiet periods and explicitly decided not to exclude free-water trials. This conflicts with the reference's use of `obs_intervals` plus `free_water == 0`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the ragged `/units/spike_times` and `/units/spike_times_index` arrays. `units/classification` selects units and go-cue timestamps position the trial bins.

ii.
```python
spike_times_data = units_grp['spike_times'][()]
spike_times_index = units_grp['spike_times_index'][()]
good_mask = data['classification'] == 'good'
neural_trials = compute_firing_rates_all_trials(spike_times_good, go_times_valid)
```

iii. The notes state that spike times are absolute session times and that classifier labels should provide the paper's classifier-based QC.

## 2-b. How is the `neural` data processed?

i. For every retained trial and good neuron, spikes in the go-aligned window are histogrammed into non-overlapping bins, and counts are divided by 0.05 seconds to obtain Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
fr[i] = counts / bin_width
```

iii. The notes say the task's required 50-ms bins supersede the paper code's 40-ms sliding window and 3.4-ms stride. They describe the result as 50-ms binned firing rates aligned to go cue.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `'good'` are retained. A session with no such units is skipped. No additional metric thresholds or `unit_quality` filter are used.

ii.
```python
good_mask = data['classification'] == 'good'
n_good = np.sum(good_mask)
if n_good == 0:
    return None
```

iii. The agent identified `classification == 'good'` as the NWB equivalent of the reference code's `qc_mode='classifier'`. It checked 69,453 retained units against the paper's 69,943 and attributed the small discrepancy to data versioning.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 seconds are added to each trial's absolute go-cue timestamp, and spikes are histogrammed against those absolute edges.

ii.
```python
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
abs_bin_edges = bin_edges_rel + go_time
counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
```

iii. The notes say all NWB spike/event times share the session clock and identify `go_start_times` as the required alignment signal.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins covering four seconds. Raw spike times are binned directly at this resolution; there is no subsequent rebinning.

ii.
```python
BIN_WIDTH = 0.05
ALIGN_START = -2.5
ALIGN_END = 1.5
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)
```

iii. The agent explicitly chose the task-mandated 50 ms instead of the different bin/stride settings in the original analysis code.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times`, trial start/stop times, and go-cue times. The agent chooses the first sample start within each trial; if none is found, it fabricates a default tone time of -1.85 seconds relative to go cue.

ii.
```python
tone_rel_go = np.full(n_trials, -1.85, dtype=np.float64)
idx_start = np.searchsorted(sample_starts, trial_starts[t])
idx_end = np.searchsorted(sample_starts, trial_stops[t])
if idx_start < idx_end:
    tone_rel_go[t] = sample_starts[idx_start] - go_times[t]
```

iii. The notes describe this as using sample onset relative to go cue, but do not justify selecting the first onset or the fallback. The reference instead selects the last sample onset before go because early licking can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The selected tone's go-relative time is subtracted from every go-relative bin center, yielding seconds elapsed since tone onset as a continuous `float32` series.

ii.
```python
bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
time_from_tone = (bin_centers_rel - tone_rel_go[t]).astype(np.float32)
```

iii. The agent states that the variable is continuous and time-varying, and verifies that it increases monotonically.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses exactly the same 80 go-relative bin centers as the neural histograms; the tone offset merely shifts the value at each center.

ii.
```python
bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
time_from_tone = bin_centers_rel - tone_rel_go[t]
```

iii. The notes say both neural and input streams are go-cue aligned on the required -2.5-to-1.5-second window.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It comes from the session-wide `BehavioralEvents/photostim_start_times` and `photostim_stop_times` timestamp arrays. If the streams are absent, empty arrays are used.

ii.
```python
if 'photostim_start_times' in be:
    data['photostim_starts'] = be['photostim_start_times']['timestamps'][()]
    data['photostim_stops'] = be['photostim_stop_times']['timestamps'][()]
else:
    data['photostim_starts'] = np.array([])
    data['photostim_stops'] = np.array([])
```

iii. The notes identify these event streams as the source and say observed stimulation occurs around -1.2 to -0.7 seconds from go cue. This differs from the reference's equivalent derivation from trial-table onset/duration fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each trial begins with an all-zero series. For every session photostimulation interval, bin centers within `[start, stop)` are set to 1.

ii.
```python
photostim = np.zeros(n_bins, dtype=np.float32)
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
    mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
    photostim[mask] = 1.0
```

iii. The agent chose a binary time series, as required, and reports sanity-checking the expected 0.5-second stimulation timing.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The common go-relative bin centers are converted to absolute session time by adding that trial's go cue, then compared directly with absolute photostimulation timestamps.

ii.
```python
abs_bin_centers = bin_centers_rel + go_time
mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
```

iii. The notes say these timestamps use the same NWB clock and confirm photostimulation appears at its expected go-relative time.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from the left- and right-lick timestamp streams and each trial's go cue. The earliest left or right lick in the 1.5 seconds after go determines choice; no lick gives class 2.

ii.
```python
left_idx = np.searchsorted(left_lick_times, go_time)
right_idx = np.searchsorted(right_lick_times, go_time)
...
if first_left < first_right:
    choices[t] = 0
elif first_right < float('inf'):
    choices[t] = 1
```

iii. The notes call this “first lick direction within 1.5s after go cue” and report verifying it. The reference instead reconstructs choice from `trial_instruction` and `outcome`; direct lick events are a plausible alternative source for the requested behavioral choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Values are encoded left=0, right=1, no lick=2, then repeated across all 80 time bins in output row 0.

ii.
```python
choices = np.full(len(go_times), 2, dtype=np.int64)
...
output_data[0, :] = choices[t]
```

iii. The agent followed the requested three classes and repeated the per-trial label so all outputs share a time-varying array shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials table's `outcome` column.

ii.
```python
data['outcome'] = np.array([x.decode() ... for x in trials_grp['outcome'][()]])
outcome_valid = data['outcome'][valid_indices]
```

iii. The notes identify `trials.outcome` as the direct source.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to ignore=0, miss=1, hit=2, and the per-trial class is repeated across all bins. Unexpected strings silently map to ignore.

ii.
```python
mapping = {'ignore': 0, 'miss': 1, 'hit': 2}
return np.array([mapping.get(o, 0) for o in outcomes], dtype=np.int64)
...
output_data[1, :] = outcome_codes[t]
```

iii. The mapping and repetition are documented as matching the requested categorical output. No justification is given for silently coercing unknown values to `ignore`.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials table's `early_lick` column.

ii.
```python
data['early_lick'] = np.array([x.decode() ... for x in trials_grp['early_lick'][()]])
early_lick_valid = data['early_lick'][valid_indices]
```

iii. The notes identify this trial-table field as the source.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'no early'` maps to 0 and every other value maps to 1; the result is repeated over all 80 bins.

ii.
```python
return np.array([0 if e == 'no early' else 1 for e in early_licks], dtype=np.int64)
...
output_data[2, :] = early_lick_codes[t]
```

iii. The agent documents the intended no=0/yes=1 mapping and repetition of a per-trial label.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 (y) and column 2 (tracking confidence) of `Camera0_side_TongueTracking/data`, plus that series' timestamps and trial go cues.

ii.
```python
data['tongue_data'] = bts['Camera0_side_TongueTracking']['data'][()]
data['tongue_timestamps'] = bts['Camera0_side_TongueTracking']['timestamps'][()]
...
tongue_y = tongue_data[:, 1]
tongue_conf = tongue_data[:, 2]
```

iii. The notes identify the camera series as x, y, and confidence sampled at about 3.4 ms and use confidence >0.5 for visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent computes 40th/60th percentiles over all visible raw frames in the session. For each 50-ms trial bin, it requires more than half of frames to be visible, averages only visible-frame y values, and otherwise emits `not visible`.

ii.
```python
visible_mask = tongue_conf > TONGUE_CONFIDENCE_THRESHOLD
y_visible = tongue_y[visible_mask]
p40 = np.percentile(y_visible, 40)
p60 = np.percentile(y_visible, 60)
...
visible_frac = np.mean(bin_conf > TONGUE_CONFIDENCE_THRESHOLD)
if visible_frac > 0.5:
    mean_y = np.mean(bin_y[vis_mask])
```

iii. The notes justify a 0.5 confidence threshold based on bimodal confidence and describe “per-session percentiles of y-position when visible.” They do not justify the extra majority-visible rule. The reference instead calculates percentiles from session-wide 50-ms bin means and labels a bin visible whenever it has at least one visible frame.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Mean visible y below p40 is class 0, from p40 (inclusive) to p60 (exclusive) is class 1, p60 or above is class 2, and bins failing the majority-visible requirement are class 3.

ii.
```python
trial_tongue_y = np.full(n_bins, 3, dtype=np.int64)
if mean_y < p40:
    trial_tongue_y[b] = 0
elif mean_y < p60:
    trial_tongue_y[b] = 1
else:
    trial_tongue_y[b] = 2
```

iii. The class boundaries follow the requested 40th/60th percentile categories, but their thresholds are based on raw visible frames rather than the same binned statistic being categorized, unlike the reference.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial, the same absolute 50-ms edges used for neural data are searched in the camera timestamp array; camera frames between each adjacent pair are assigned to that neural bin.

ii.
```python
abs_bin_edges = bin_edges_rel + go_time
idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
idx_end = np.searchsorted(tongue_timestamps, abs_bin_edges[b + 1])
```

iii. The agent states that camera and neural streams share the NWB session clock and uses the same go-aligned window and resolution.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units and sessions with fewer than two coverage-filtered trials are skipped. Absent photostimulation streams become empty arrays; bins without sufficient visible tongue frames become class 3. A trial with no discovered sample onset receives a hard-coded -1.85-second tone offset. Unknown outcomes become `ignore`, and any early-lick string other than `'no early'` becomes `yes`. The code does not use `obs_intervals` or remove free-water/no-spike trials.

ii.
```python
if n_good == 0:
    return None
tone_rel_go = np.full(n_trials, -1.85, dtype=np.float64)
mapping.get(o, 0)
np.array([0 if e == 'no early' else 1 for e in early_licks])
trial_tongue_y = np.full(n_bins, 3, dtype=np.int64)
```

iii. The notes frame skipped sessions/trials and explicit tongue invisibility as data-quality handling. After investigating zero-neural trials and free-water flags, the agent chose to retain them. Several silent fallbacks are not documented or validated and can turn missing/unknown data into apparently valid labels.

## 10-a. What are the most time-consuming steps of the code?

i. In the agent's implementation, neural histogramming dominates most per-session timings (often seconds versus tenths of a second for direct-HDF5 loading); tongue/output processing is usually next. The final 11.8-GB pickle write is also substantial, though its duration is not separately timed.

ii.
```python
t_neural = time.time()
...
print(f"neural={t_neural-t_filter:.1f}s, ... output={t_output-t_input:.1f}s")
...
pickle.dump(output_data, f)
```

iii. The notes report roughly 3 seconds per session and about 9 minutes total, emphasizing the `h5py` speedup. The conversion log confirms that the nested neural binning is generally the largest measured component.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-neuron neural loop could be reorganized as one `searchsorted` over all trial edges per neuron, as in the reference. The trial-by-bin tongue loop could use global/per-trial bin indices and `bincount`. The input code unnecessarily tests every session photostimulation interval for every trial. Per-trial output construction and region-index assembly are also vectorizable, though comparatively cheap.

ii.
```python
for t in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
```
```python
for t in range(n_trials):
    for b in range(n_bins):
        idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
```
```python
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
```

iii. The agent repeatedly calls these routines “vectorized” because operations within each loop use NumPy, but it does not discuss the remaining nested loops. The human reference removes the neural trial loop by flattening all trial edges for each unit.

## 10-c. What processing does the code repeat multiple times?

i. Neural bin edges and histograms are rebuilt for every trial; tongue timestamp searches are performed twice for every one of 80 bins in every trial; and all session photostimulation intervals are scanned for every trial. Similar bin-center arrays are separately constructed in the neural, input, and tongue functions.

ii.
```python
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
```
appears independently in neural and tongue processing, while input processing separately creates:
```python
bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
```

iii. The notes emphasize fast loading and “vectorized” processing but do not acknowledge these repeated computations. The reference defines the common grid once and performs spike edge searches across all trials at once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `subject_desc`, computes tongue timestamp `dt` and `t0_tongue` without using them, carries fine `anno_names` out of `process_session` but discards them at assembly, and optionally generates extensive diagnostic plots that are not part of the converted dataset. The neural bounds checks and slicing also precede `np.histogram`, which could operate via edge searches without materializing each trial slice.

ii.
```python
data['subject_desc'] = f['general']['subject']['description'][()]...
dt = np.median(np.diff(tongue_timestamps[:100]))
t0_tongue = tongue_timestamps[0]
...
'anno_names': anno_names,
```

iii. The agent used descriptions, fine annotations, and plots during exploration/validation, but the final output retains only subject IDs and coarse region indices. The two tongue timing variables are entirely unused.
