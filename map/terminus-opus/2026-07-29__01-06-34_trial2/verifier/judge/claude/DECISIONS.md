# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all NWB files matching `data/sub-*/*.nwb` using glob, sorted alphabetically. Each file is opened with `pynwb.NWBHDF5IO` and read into memory. In `--sample` mode, only 2 sessions (indices 2 and 4 from the sorted list) are used. Each NWB file contains one session's worth of data including trials, units, behavioral events, and behavioral time series.

ii.
```python
def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

# In main():
nwb_files = get_nwb_files()
if args.sample:
    nwb_files = [nwb_files[2], nwb_files[4]]

for i, path in enumerate(nwb_files):
    r = process_session(path, T_START, T_END, BIN_SIZE, ...)
```

iii. The AI noted that the reference code uses .mat files exported from DataJoint, but the provided data is in NWB format. NWB is the native format of the DANDI dataset. The AI reads all 174 NWB files and processes each session sequentially.

---

## 1-b. How are the data split into subjects (mice)?

i. Subject ID is extracted from `nwb.subject.subject_id` within each NWB file. After processing all sessions, unique subject IDs are collected, sorted, and stored in a list. Each session is mapped to its subject via an index array.

ii.
```python
# Inside process_session():
subject_id = nwb.subject.subject_id

# In main():
subjects = sorted(set(s['subject_id'] for s in all_sessions))
subj_idx.append(subjects.index(s['subject_id']))
```

iii. The AI identified 28 unique subjects, matching the dandiset metadata and the paper.

---

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The 174 NWB files yield 174 candidate sessions. Sessions with no good neurons are skipped (1 session), and sessions with fewer than 2 valid trials are also skipped, yielding 173 sessions.

ii.
```python
for i, path in enumerate(nwb_files):
    r = process_session(path, ...)
    if r: all_sessions.append(r)
    else: skipped += 1
```

iii. The AI noted that the paper reports 173 sessions, and one NWB file had no good neurons, producing the matching count.

---

## 1-d. How are the data split into trials?

i. Trials are read from the `nwb.trials` table. Each row is one trial. The AI also determines which trials have neural recordings by comparing `obs_intervals` with trial start times.

ii.
```python
n_trials = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
# ...
recorded_trials = get_recorded_trial_indices(nwb, good_indices)
```

iii. The AI recognized that some sessions only record a subset of trials (obs_intervals may not cover all trials) and implemented logic to match observation intervals to trial indices.

---

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a "regular trial mask" that filters out: (1) early lick trials, (2) auto water trials, (3) free water trials, (4) ignore/no-response trials, (5) photostimulation trials. Additionally, only trials within neural recording intervals (obs_intervals) are retained. Sessions with fewer than 2 valid trials are skipped. No session-level performance filtering (>65%, ≥50 correct per side) is applied.

ii.
```python
mask_no_early = (early_lick == 'no early')
mask_no_auto = (auto_water == 0)
mask_no_free = (free_water == 0)
mask_no_ignore = (outcome != 'ignore')
mask_no_photostim = ~has_photostim
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim

recorded_set = set(recorded_trials)
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. The AI states this matches `get_regular_trial_mask` from the reference code. The CONVERSION_NOTES mention that session-level performance filtering was initially implemented but removed to match the paper's 173 sessions count.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']`, specifically for units where `nwb.units['classification']` equals `'good'`.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
```

iii. The AI identified that the reference code uses classifier-based QC (logistic regression classifiers trained per brain region), and the NWB `classification` field stores the result of this QC.

---

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins using `np.histogram`, then divided by bin_size (0.05s) to convert to firing rates in Hz. The time window is -2.5s to 1.5s relative to go cue onset, yielding 80 time bins per trial.

ii.
```python
def bin_spikes_all_trials(spike_times_list, go_times_arr, t_start, t_end, bin_size):
    n_bins = int(round((t_end - t_start) / bin_size))
    for go in go_times_arr:
        bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, st in enumerate(spike_times_list):
            counts, _ = np.histogram(st_w, bins=bin_edges)
            fr[i] = counts.astype(np.float32) / bin_size
        result.append(fr)
```

iii. The AI noted that the task instructions specify 50ms bins. The reference code uses a 100ms sliding window with 50ms stride (`bw=0.1, stride=0.05`), but the AI followed the task instructions for bin width.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are included. No additional neural quality filtering (e.g., firing rate thresholds, ISI violation thresholds) is applied beyond the classifier-based QC stored in the NWB file.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```

iii. The AI identified that the reference code uses classifier-based QC where region-specific logistic regression classifiers are trained on manual curation labels. The NWB `classification` field stores the classifier output directly.

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. The go cue times are extracted from `nwb.acquisition['BehavioralEvents'].time_series['go_start_times'].timestamps[:]`. For each trial, bin edges are computed relative to the trial's go cue time.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
# ...
go_valid = go_times[valid_trials]
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)

# Inside bin_spikes_all_trials:
for go in go_times_arr:
    bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The AI noted that the reference code already has spike times aligned to go cue in the .mat files (`gocue_time` is subtracted). In NWB, spike times are in absolute time, so the go cue timestamp is used to compute bin edges.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (0.05s). No rebinning is applied. The data is binned directly from spike times into 50ms non-overlapping bins. The reference code uses `stride=0.05, bw=0.1` (100ms sliding window with 50ms stride), but the AI follows the task instruction of 50ms bins.

ii.
```python
BIN_SIZE = 0.05  # 50ms
n_bins = int(round((t_end - t_start) / bin_size))  # 80 bins
bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The task instructions specify "Use 50-ms-width bins for computing firing rates." The AI follows this directly rather than the reference code's 100ms sliding window.

---

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (from `BehavioralEvents`) and `go_start_times` (also from `BehavioralEvents`). The sample start time is the onset of the tone stimulus.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI identified `sample_start_times` as the tone onset times from the NWB behavioral events.

---

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the sample start time (tone onset) is found within the trial's time window. The time from tone onset is computed as `bin_centers - (sample_start - go_cue)`, giving seconds elapsed since tone onset at each bin center. If no sample start time is found within the trial, NaN values are used.

ii.
```python
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The AI reasoned that the tone onset time needs to be expressed relative to the go cue (since neural data is aligned to go cue), then subtracted from bin centers to get elapsed time since tone onset.

---

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and time-from-tone-onset use the same `bin_centers` array, which is computed relative to go cue onset. The input has the same number of time bins (80) as the neural data.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
tft = bin_centers - (ss - go)  # same bin_centers as neural
input_trials.append(np.stack([tft.astype(np.float32), ps_on]))
```

iii. Implicit alignment through shared bin_centers array.

---

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `photostim_power`, `photostim_onset`, and `photostim_duration` columns in the trials table.

ii.
```python
photostim_power_raw = nwb.trials['photostim_power'][:]
photostim_onset_raw = nwb.trials['photostim_onset'][:]
photostim_dur_raw = nwb.trials['photostim_duration'][:]
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
```

iii. The AI identified three NWB trial columns for photostimulation parameters.

---

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is created. Photostim onset is converted from trial-start-relative to go-cue-relative time. For each time bin, the photostim indicator is 1 if the bin center falls within the photostim window (onset to onset + duration), else 0. However, since photostim trials are filtered out by `regular_mask`, this input is always 0 for all valid trials.

ii.
```python
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    d = float(photostim_dur_raw[trial_idx])
    ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES that photostim is always 0 because photostim trials are filtered out by the regular trial mask, consistent with the reference code.

---

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Same `bin_centers` array as neural data. The photostim onset is converted to go-cue-relative coordinates before comparison with bin centers.

ii.
```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. Since photostim trials are filtered out, this alignment code never produces non-zero values.

---

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` in the NWB trials table. This encodes the **instructed** (correct) lick direction, NOT the animal's actual choice.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
# ...
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. The AI did not explicitly discuss the distinction between instructed direction and actual lick direction in the CONVERSION_NOTES.

---

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Simple mapping: `trial_instruction == 'right'` maps to 1, else maps to 0. This is a per-trial value replicated across all time bins. The mapping uses the instructed direction rather than the actual lick direction, which is incorrect for miss trials (where the animal licked the opposite side).

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
out[0, :] = choice
```

iii. No explicit justification provided. The reference code similarly stores `trial_type` (instruction) rather than actual lick direction, but calls it `trial_type` not `choice`.

---

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `outcome` in the NWB trials table, which contains string values `'hit'`, `'miss'`, or `'ignore'`.

ii.
```python
outcome = nwb.trials['outcome'][:]
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
```

iii. The AI followed the task spec mapping: ignore=0, miss=1, hit=2.

---

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String-to-integer mapping using the dictionary `{'ignore': 0, 'miss': 1, 'hit': 2}`. Since ignore trials are filtered out by `regular_mask`, outcome only takes values 1 (miss) or 2 (hit) in practice. The value is per-trial and replicated across all time bins.

ii.
```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
out[1, :] = out_val
```

iii. The AI followed the exact mapping from the task spec.

---

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. There is no "Distance to reward zone" output in this dataset. This question appears to be a template artifact. For the Outcome variable: it is a per-trial scalar replicated identically across all 80 time bins.

ii.
```python
out[1, :] = out_val  # per-trial value broadcast to all time bins
```

iii. N/A - this output variable does not exist in this dataset.

---

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `early_lick` in the NWB trials table, which contains string values like `'early'` or `'no early'`.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
early_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The AI identified the correct NWB field for early lick status.

---

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Maps `'early'` to 1, anything else to 0. However, since early lick trials are filtered out by `regular_mask` (`mask_no_early`), this output is always 0 for all valid trials. The decoder achieves trivial 100% accuracy because there is no variation.

ii.
```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
out[2, :] = early_val
```

iii. The AI explicitly acknowledged in CONVERSION_NOTES that early_lick is trivial because early lick trials are filtered by `get_regular_trial_mask`, and this is consistent with the reference code.

---

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `Camera0_side_TongueTracking` in `nwb.acquisition['BehavioralTimeSeries']`. Specifically, column index 1 (y-position) and column index 2 (DLC likelihood) of the tracking data.

ii.
```python
tt_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_ts = tt_obj.timestamps[:]
td = tt_obj.data[:]
tongue_y_arr = td[:, 1]
tongue_lk_arr = td[:, 2]
```

iii. The AI identified DeepLabCut (DLC) tongue tracking data from the side camera.

---

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Tongue y-position values are filtered by DLC likelihood > 0.9 (high-confidence detections only). The mean y-position within each time bin is computed from high-confidence samples falling within a half-bin-width window around each bin center.

ii.
```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
    if ih > il:
        lk = tongue_lk_arr[il:ih]
        gd = lk > 0.9
        if np.any(gd):
            my = np.mean(tongue_y_arr[il:ih][gd])
```

iii. The AI used standard DLC likelihood thresholding (>0.9) to select reliable tracking frames.

---

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentile-based discretization: <40th percentile → 0 (low), 40th-60th percentile → 1 (mid), >60th percentile → 2 (high). Percentiles are computed over all high-confidence tongue y values across all valid trials in the session. When no high-confidence detection exists in a bin, the default value is 1 (mid).

ii.
```python
# Computing percentiles:
all_ty = []
for ti in valid_trials:
    go = go_times[ti]
    i_lo = np.searchsorted(tongue_ts, go + t_start)
    i_hi = np.searchsorted(tongue_ts, go + t_end)
    if i_hi > i_lo:
        lk = tongue_lk_arr[i_lo:i_hi]
        good = lk > 0.9
        if np.any(good):
            all_ty.append(tongue_y_arr[i_lo:i_hi][good])
tongue_y_p40 = np.percentile(concat, 40)
tongue_y_p60 = np.percentile(concat, 60)

# Categorizing:
ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. The AI followed the task spec: "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile of y-position over the session."

---

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each of the 80 time bins (matching neural data), the tongue y-position is averaged within a window of `[bin_center - bin_size/2, bin_center + bin_size/2]` (absolute time), then categorized. This provides a time-varying categorical output aligned with the neural data bins.

ii.
```python
half_bin = bin_size / 2.0
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The AI uses the same bin_centers as the neural data to ensure alignment.

---

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Sessions with no good neurons**: Skipped entirely (1 session).
- **Sessions with <2 valid trials**: Skipped.
- **Missing sample_start_times**: Time-from-tone-onset is set to NaN for that trial.
- **Missing tongue tracking data**: Tongue y is defaulted to category 1 (mid).
- **All-zero neural data**: Kept (noted as occurring at recording boundary for 1 trial).
- **obs_intervals mismatch**: Multiple matching strategies (direct count, timestamp proximity) to determine which trials have recordings.

ii.
```python
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)

# Default tongue y:
ty = np.ones(n_bins, dtype=np.int64)  # default mid category
```

iii. The AI documented these edge cases in CONVERSION_NOTES and noted that defaults were chosen to minimize impact.

---

## 10-a. What are the most time-consuming steps of the code?

i. Based on the timing output in the code:
1. **Loading spike times from NWB** (disk I/O): Reported per-session.
2. **Spike binning** (`bin_spikes_all_trials`): Iterates over all trials and neurons.
3. **Tongue y-position computation**: Inner loop over 80 bins per trial with `np.searchsorted` calls.

ii.
```python
t_load = time.time()
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
print(f"  Loaded spike times ({time.time()-t_load:.1f}s)")
# ...
print(f"  Processed {len(valid_trials)} trials ({time.time()-t_fr:.1f}s, ...)")
```

iii. The AI reported ~10s/session and ~30 minutes total in CONVERSION_NOTES.

---

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized:
1. **Tongue y-position bin loop** (`for b in range(n_bins)`): The inner loop over 80 bins per trial does individual `searchsorted` calls that could be vectorized.
2. **Spike binning neuron loop** (`for i, st in enumerate(spike_times_list)`): Each neuron is binned separately with `np.histogram`. This could potentially be vectorized using advanced indexing.

ii.
```python
# Tongue y - iterates over 80 bins per trial:
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)

# Spike binning - iterates over neurons:
for i, st in enumerate(spike_times_list):
    counts, _ = np.histogram(st_w, bins=bin_edges)
```

iii. The AI did use `np.searchsorted` and `np.histogram` for partial vectorization but kept outer loops over bins and neurons.

---

## 10-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each session is processed once independently. The tongue y-position data is read once per session and used for both percentile computation and per-bin categorization (two passes over the data, but for different purposes). The `go_times` array is accessed multiple times but this is negligible.

ii.
```python
# Pass 1: Compute percentiles over valid trials
for ti in valid_trials:
    # ... collect tongue y values
tongue_y_p40 = np.percentile(concat, 40)

# Pass 2: Categorize per bin per trial
for t_idx, trial_idx in enumerate(valid_trials):
    for b in range(n_bins):
        # ... categorize tongue y
```

iii. No explicit discussion of repeated processing in CONVERSION_NOTES.

---

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations produce degenerate or unused results:
1. **Photostim input calculation**: Always 0 because photostim trials are filtered out. The photostim onset/duration conversion code executes but the `if has_photostim[trial_idx]` branch is never taken for valid trials.
2. **Session performance calculation**: Computed for each session (lines 117-129) but not used for any filtering or downstream processing.
3. **Outcome ignore class (0)**: Defined in `output_values` but never occurs after filtering.
4. **Early lick output**: Always 0 after filtering; decoder trivially gets 100% accuracy.

ii.
```python
# Photostim - always 0 for valid trials:
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:  # never True for valid trials
    ...

# Performance - computed but unused:
perf = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0.0
print(f"  Performance: {perf:.1%} ...")

# Early lick - always 0:
early_val = 1 if early_lick[trial_idx] == 'early' else 0  # always 0
```

iii. The AI acknowledged in CONVERSION_NOTES that photostim and early_lick are trivial/degenerate due to the trial filtering matching the reference code's `get_regular_trial_mask`.
