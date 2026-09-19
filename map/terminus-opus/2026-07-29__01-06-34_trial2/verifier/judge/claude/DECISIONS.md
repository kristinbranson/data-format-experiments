# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data from NWB files found via a glob pattern `data/sub-*/*.nwb`. Each file is opened with `pynwb.NWBHDF5IO` and processed one at a time in `process_session()`. Trial metadata, spike times, behavioral events, and tongue tracking are all read within the same `with` block for each file. 174 NWB files are found.

ii.
```python
def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
```
```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    subject_id = nwb.subject.subject_id
    ...
    trial_instruction = nwb.trials['trial_instruction'][:]
    outcome = nwb.trials['outcome'][:]
    ...
    go_times = be.time_series['go_start_times'].timestamps[:]
    sample_start_times = be.time_series['sample_start_times'].timestamps[:]
```

iii. The AI notes in CONVERSION_NOTES.md that pynwb is the standard reader for NWB files and that all 174 files are found matching the expected count from the dandiset.

## 1-b. How are the data split into subjects?

i. Each NWB file's `nwb.subject.subject_id` is read and stored per session. At assembly, unique subject IDs are collected and sorted, and each session is assigned a subject index.

ii.
```python
subject_id = nwb.subject.subject_id
...
subjects = sorted(set(s['subject_id'] for s in all_sessions))
...
subj_idx.append(subjects.index(s['subject_id']))
```

iii. The AI uses the numeric subject_id from the NWB file, resulting in 28 subjects, matching the reference.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The sorted file list determines session order. 173 sessions survive (one dropped for having no good neurons).

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
```

iii. The AI correctly identifies one session per NWB file. 173 sessions match the reference.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`). Each trial is identified by its index in this table.

ii.
```python
n_trials = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
...
```

iii. Trials are directly taken from the NWB trials table, which is the standard source.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the reference code's `get_regular_trial_mask` logic, which filters out: early lick trials (`early_lick != 'no early'`), auto water trials (`auto_water != 0`), free water trials (`free_water != 0`), ignore/no-response trials (`outcome == 'ignore'`), and photostimulation trials (`has_photostim`). Additionally, only trials within `obs_intervals` (recorded trials) are kept. Sessions with fewer than 2 valid trials are dropped.

ii.
```python
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim

recorded_set = set(recorded_trials)
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. The AI explicitly states in CONVERSION_NOTES.md: "Trial filtering: no early lick, no auto water, no free water, no ignore, no photostim" — matching the reference code's `get_regular_trial_mask`. This results in 52,990 valid trials across 173 sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for each good unit. The go cue times from `BehavioralEvents/go_start_times` provide the alignment reference.

ii.
```python
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
```

iii. The AI uses spike_times from the units table, the only neural representation available in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins spanning [-2.5, 1.5]s relative to the go cue, then divided by bin size to get firing rates in Hz. This is done per trial with `np.histogram`.

ii.
```python
def bin_spikes_all_trials(spike_times_list, go_times_arr, t_start, t_end, bin_size):
    n_bins = int(round((t_end - t_start) / bin_size))
    ...
    for go in go_times_arr:
        bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, st in enumerate(spike_times_list):
            ...
            counts, _ = np.histogram(st_w, bins=bin_edges)
            fr[i] = counts.astype(np.float32) / bin_size
        result.append(fr)
    return result
```

iii. The AI uses `np.linspace` for bin edges and `np.histogram` for binning. No smoothing or normalization is applied, matching the reference approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. If no good units exist, the session is dropped. This matches the QC classifier approach.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```

iii. The AI uses the `classification` field from the QC classifier, consistent with the spike sorting QC paper. This gives 69,453 good neurons, close to the paper's 69,943.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. Bin edges span from `go_time + t_start` to `go_time + t_end` for each trial.

ii.
```python
for go in go_times_arr:
    bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The go cue times and spike times share the same session-absolute clock, so alignment is done by simply offsetting the bin edges by the go cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins spanning [-2.5, 1.5]s, giving 80 time bins per trial. No rebinning is applied — spike times are directly binned at 50ms resolution.

ii.
```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
n_bins = int(round((t_end - t_start) / bin_size))  # 80
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
```

iii. The 50ms bin width and [-2.5, 1.5]s window match the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onsets) in `BehavioralEvents`, together with the go cue time and trial start/stop times. The AI selects the **first** sample_start within the trial window.

ii.
```python
def get_sample_start_for_trial(sample_start_times, trial_start, trial_stop):
    mask = (sample_start_times >= trial_start) & (sample_start_times <= trial_stop)
    return sample_start_times[mask][0] if np.any(mask) else None
```

iii. The AI finds tone onsets within each trial's time window and takes the first one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `bin_centers - (sample_start - go_time)`, giving the time elapsed since the tone onset at each bin center. If no sample_start is found within the trial, NaN is used.

ii.
```python
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The computation is straightforward: bin_centers are relative to the go cue, and the offset `(ss - go)` shifts them to be relative to the tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both the neural data and the time-from-tone input use the same bin centers relative to the go cue, ensuring they share the same temporal grid.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
```

iii. The bin centers are defined once and used consistently for both neural binning and input computation.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset`, `photostim_duration`, and `photostim_power` in the trials table, plus `start_time` and go cue time for alignment.

ii.
```python
photostim_power_raw = nwb.trials['photostim_power'][:]
photostim_onset_raw = nwb.trials['photostim_onset'][:]
photostim_dur_raw = nwb.trials['photostim_duration'][:]
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
```

iii. The AI checks photostim_power to determine whether photostimulation occurred.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For trials with photostimulation, the onset is converted from trial-start-relative to go-cue-relative coordinates, and a binary time series is created where 1 indicates the photostim is active. However, because all photostim trials are filtered out by `regular_mask`, the photostim input is always 0 in practice.

ii.
```python
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    d = float(photostim_dur_raw[trial_idx])
    ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The AI notes in CONVERSION_NOTES.md Step 12: "Photostim input: Always 0 because photostim trials are filtered out. This is correct per reference code."

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The same bin_centers are used, so alignment is inherent. But since photostim is always 0 after filtering, this is moot.

ii. Same bin_centers as neural data.

iii. N/A — always zero.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from `trial_instruction` only. The AI maps 'left' to 0 and 'right' to 1. It does NOT derive actual lick direction from the combination of instruction and outcome.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. The AI does not justify using trial_instruction as choice. Since ignore trials are filtered out and only hit/miss trials remain, and since miss means licking the wrong side, using trial_instruction as choice is incorrect for miss trials — it records the instructed side rather than the actual lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A simple mapping: 'left' -> 0, 'right' -> 1. Only two categories (no "no lick" category). The value is repeated across all 80 time bins.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
...
out[0, :] = choice
```
```python
'output_values': [['left','right'], ...]
```

iii. The AI defines only 2 choice categories instead of 3. There is no "no lick" category because ignore trials are filtered out.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table. However, only 'miss' and 'hit' values appear because 'ignore' trials are filtered out.

ii.
```python
outcome = nwb.trials['outcome'][:]
...
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
```

iii. The mapping exists for all three values but ignore never occurs in practice.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Repeated across all 80 bins. In practice, only miss=1 and hit=2 appear.

ii.
```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
out[1, :] = out_val
```

iii. The output_values list has only 3 categories: `['ignore','miss','hit']`. The range in the data is [1, 2] since ignore trials are filtered.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table. However, all early lick trials are filtered out, so this output is always 0.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
...
early_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The AI notes in CONVERSION_NOTES.md: "early_lick is trivial because all early lick trials are filtered out by get_regular_trial_mask."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early) or 1 (early). In practice always 0 since early lick trials are filtered.

ii.
```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
out[2, :] = early_val
```

iii. The output_values list is `['no','yes']`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `BehavioralTimeSeries`. Column 1 is tongue_y and column 2 is tongue_likelihood.

ii.
```python
tt_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_ts = tt_obj.timestamps[:]
td = tt_obj.data[:]
tongue_y_arr = td[:, 1]
tongue_lk_arr = td[:, 2]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI filters tongue frames by likelihood > 0.9, computes percentiles (40th, 60th) over all visible frames within valid trial windows (not the whole session, not bin means), and discretizes into 3 classes (0: < p40, 1: p40-p60, 2: > p60). Note: default is class 1 (mid) when no visible frames in a bin, not a separate "not visible" class.

ii.
```python
# Percentile computation - over raw frames from valid trials only
for ti in valid_trials:
    go = go_times[ti]
    i_lo = np.searchsorted(tongue_ts, go + t_start)
    i_hi = np.searchsorted(tongue_ts, go + t_end)
    if i_hi > i_lo:
        lk = tongue_lk_arr[i_lo:i_hi]
        good = lk > 0.9
        if np.any(good):
            all_ty.append(tongue_y_arr[i_lo:i_hi][good])
if all_ty:
    concat = np.concatenate(all_ty)
    tongue_y_p40 = np.percentile(concat, 40)
    tongue_y_p60 = np.percentile(concat, 60)
```
```python
# Per-bin discretization
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
    if ih > il:
        lk = tongue_lk_arr[il:ih]
        gd = lk > 0.9
        if np.any(gd):
            my = np.mean(tongue_y_arr[il:ih][gd])
            ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. The AI uses likelihood > 0.9 threshold (vs reference's 0.5), computes percentiles over raw visible frames from valid trials (not session-wide bin means), and defaults unvisible bins to class 1 (mid) rather than a separate "not visible" class 3.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories only: 0 (< 40th pctl), 1 (40th-60th pctl or not visible), 2 (> 60th pctl). There is no separate "not visible" (class 3) category.

ii.
```python
ty = np.ones(n_bins, dtype=np.int64)  # default to 1 (mid)
...
ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```
```python
'output_values': [..., ['low','mid','high']]
```

iii. The AI defaults non-visible bins to class 1 (mid) instead of a 4th class. The output_values list only has 3 categories.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin, the AI searches for tongue frames within `[bin_center - half_bin, bin_center + half_bin]` using `np.searchsorted` on the camera timestamps. The bin centers are the same as for neural data.

ii.
```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The alignment approach uses the same bin centers as the neural data, ensuring temporal correspondence.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good neurons are dropped. (2) Trials without neural recordings (outside obs_intervals) are excluded. (3) Tongue bins with no visible frames (likelihood <= 0.9) default to class 1 (mid). When no sample_start is found in a trial, NaN is used for time_from_tone_onset.

ii.
```python
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```
```python
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```
```python
ty = np.ones(n_bins, dtype=np.int64)  # default to 1 (mid)
```

iii. The AI handles the NaN classification session by implicit comparison (`'good' != NaN`). Missing tongue data defaults to "mid" class rather than an explicit "not visible" class.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies loading spike times and per-trial spike binning as the bottlenecks. Processing is ~10-20ms per trial, with total conversion taking about 15-30 minutes for full data.

ii.
```python
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
print(f"  Loaded spike times ({time.time()-t_load:.1f}s)")
```

iii. The conversion output shows ~14-23ms per trial, with full processing completing in ~700-900 seconds.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two nested loops exist: the outer loop over trials in `bin_spikes_all_trials` and the inner loop over neurons. The per-bin tongue discretization also loops over each bin for each trial.

ii.
```python
# Outer loop over trials
for go in go_times_arr:
    ...
    # Inner loop over neurons
    for i, st in enumerate(spike_times_list):
        counts, _ = np.histogram(st_w, bins=bin_edges)
```
```python
# Tongue loop: per-bin per-trial
for b in range(n_bins):
    bc = go + bin_centers[b]
    ...
```

iii. The reference vectorizes the trial dimension by flattening all bin edges and using a single `searchsorted` per neuron. The AI's approach loops over both trials and neurons, making it significantly slower.

## 10-c. What processing does the code repeat multiple times?

i. Each trial independently computes bin edges via `np.linspace`, which could be precomputed once. The tongue percentile computation reads the same tongue data arrays multiple times.

ii.
```python
for go in go_times_arr:
    bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The bin edge computation is simple but repeated for each trial rather than being offset from a single precomputed template.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session-level performance statistics (hit rate, correct L/R counts) that are only printed and not used in the output. It also computes photostim detection that is rendered moot by trial filtering.

ii.
```python
# Performance computation - printed but not used in output
ctrl_out = outcome[control]
n_hit = np.sum(ctrl_out == 'hit')
n_miss = np.sum(ctrl_out == 'miss')
perf = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0.0
```
```python
# Photostim detection - always filtered out
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
```

iii. These are diagnostic computations that don't affect the output data.
