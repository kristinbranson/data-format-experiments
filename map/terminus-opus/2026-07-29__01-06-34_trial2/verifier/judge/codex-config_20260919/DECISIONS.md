# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed and sorted every NWB file under `data/sub-*/*.nwb`, opened each with `pynwb.NWBHDF5IO`, and extracted subject, trials, units, behavioral events, and tracking data. Full mode processes all files; sample mode selects two fixed files.

ii.
```python
def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
...
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
```

iii. The notes justify NWB/pynwb as the available equivalent of the reference repository's MATLAB loading and report 174 discovered NWB files, 173 with good neurons.

## 1-b. How are the data split into subjects?

i. Each session obtains its subject from `nwb.subject.subject_id`; unique IDs are sorted and each session is mapped to its list index.

ii.
```python
subject_id = nwb.subject.subject_id
...
subjects = sorted(set(s['subject_id'] for s in all_sessions))
subj_idx.append(subjects.index(s['subject_id']))
```

iii. The agent regarded the NWB subject field as authoritative and validated that the result contained 28 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Surviving per-file results are appended in sorted file order; a file is skipped if it has no good units or fewer than two valid trials.

ii.
```python
for i, path in enumerate(nwb_files):
    r = process_session(path, T_START, T_END, BIN_SIZE, ...)
    if r: all_sessions.append(r)
```

iii. This follows the dataset layout. The notes explain the 174-to-173 difference as one file with no good neurons and report agreement with the paper's session count.

## 1-d. How are the data split into trials?

i. Trial-indexed columns come from `nwb.trials`; go cues are indexed with the same trial indices. Neural recording coverage is inferred from one good unit's `obs_intervals`, matched to trial starts when necessary. Only indices in both the recorded set and the regular-trial mask are processed.

ii.
```python
recorded_trials = get_recorded_trial_indices(nwb, good_indices)
recorded_set = set(recorded_trials)
valid_trials = np.array([i for i in range(n_trials)
                         if i in recorded_set and regular_mask[i]])
```

iii. The notes say `obs_intervals` is needed because some sessions record only a subset of behavioral trials and claim this handling was manually checked.

## 1-e. How are trials filtered based on quality controls?

i. The agent retains recorded trials only when there is no early lick, auto water, free water, ignore outcome, or photostimulation. Sessions with fewer than two survivors are dropped.

ii.
```python
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
valid_trials = np.array([i for i in range(n_trials)
                         if i in recorded_set and regular_mask[i]])
if len(valid_trials) < 2:
    return None
```

iii. The agent explicitly justified this as an exact application of the reference repository's `get_regular_trial_mask`. Its notes acknowledge that this makes early lick trivial and photostimulation always zero, but still call it reference-consistent.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for units whose `classification` is `good`, using `go_start_times` to define trial windows.

ii.
```python
good_indices = np.where(np.array(classifications) == 'good')[0]
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The notes identify classifier-based QC as the paper/white-paper method and state that NWB spike times share the event time base.

## 2-b. How is the `neural` data processed?

i. For each retained trial and good neuron, spikes within the four-second window are histogrammed into non-overlapping bins and divided by bin width to produce Hz. No smoothing or normalization is applied.

ii.
```python
counts, _ = np.histogram(st_w, bins=bin_edges)
fr[i] = counts.astype(np.float32) / bin_size
```

iii. The notes say this implements the reference `sliding_histogram` firing-rate concept with the task-required 50 ms width; manual spike/bin checks reportedly matched.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained; sessions with zero good units are skipped. No additional unit-quality metric thresholds are applied.

ii.
```python
classifications = nwb.units['classification'][:]
good_indices = np.where(np.array(classifications) == 'good')[0]
if len(good_indices) == 0:
    return None
```

iii. The agent attributes this classification to the specified classifier QC and reports 69,453 retained units, close to the paper's 69,943.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges are created from each trial's go-cue timestamp plus offsets from -2.5 to +1.5 seconds; absolute spike timestamps are histogrammed against them.

ii.
```python
for go in go_times_arr:
    bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The agent notes that spike and event timestamps use the same NWB clock, so no further clock correction is necessary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50 ms bins over [-2.5, +1.5) seconds. Raw event-time spikes are binned directly; no subsequent temporal rebinning occurs.

ii.
```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
n_bins = int(round((t_end - t_start) / bin_size))
```

iii. The agent chose 50 ms because it is explicitly required, noting that it supersedes the reference video's 40 ms processing.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial `start_time`/`stop_time`, and `go_start_times`. The helper selects the first sample start within a trial.

ii.
```python
mask = (sample_start_times >= trial_start) & (sample_start_times <= trial_stop)
return sample_start_times[mask][0] if np.any(mask) else None
```

iii. Documentation describes this as time since the first tone onset. No justification is given for choosing the first rather than the last replayed tone on early-lick trials; those trials are filtered anyway.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Go-relative bin centers are shifted by the go-minus-tone interval, yielding seconds since tone at each bin center. Missing tone times yield all NaNs.

ii.
```python
tft = bin_centers - (ss - go)
...
tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The notes say continuous seconds from sample/tone onset are the requested representation and report a manual value check.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 go-cue-relative bin centers used to delimit the neural bins.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
tft = bin_centers - (ss - go)
```

iii. The shared centers and go timestamp put the input and neural arrays on the same time axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_power`, `photostim_onset`, `photostim_duration`, and `start_time`, plus the go cue. Power determines whether a trial is stimulated.

ii.
```python
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
```

iii. The agent describes onset as trial-start-relative and therefore converts it to the go-relative decoder axis.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary vector is one when a bin center lies in the half-open stimulation interval. However, all photostimulation trials are removed beforehand, so every saved vector is zero.

ii.
```python
ps_on = ((bin_centers >= o_rel) &
         (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The notes recognize the all-zero result and defend it as a consequence of applying the reference regular-trial mask.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-start-relative onset is converted to go-relative time and compared with the same bin centers as the neural grid.

ii.
```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The agent says this conversion aligns stimulation to the go cue. The alignment formula is sound despite stimulated trials being absent downstream.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is taken directly from `trial_instruction`: right maps to 1 and every other value maps to left/0. It does not combine instruction with outcome to recover actual lick direction.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. The mapping plan equates `trial_instruction` with choice. A manual miss-trial check in the notes incorrectly claims instructed left/outcome miss validates choice=left.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The binary code (0 left, 1 right) is repeated across all 80 time bins. There is no no-lick category because ignore trials are filtered.

ii.
```python
out[0, :] = choice
...
'output_values': [['left','right'], ...]
```

iii. The agent describes the output as a per-trial left/right class and treats exclusion of ignores as reference filtering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial-table `outcome` column.

ii.
```python
outcome = nwb.trials['outcome'][:]
```

iii. The raw data already provides the requested categorical outcome labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2 and the code is repeated over time. Yet ignore trials are filtered, so class 0 is absent.

ii.
```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
out[1, :] = out_val
```

iii. The notes say the category mapping follows the task, while defending removal of ignores via the paper's analysis mask.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial-table `early_lick` column.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
```

iii. The raw field explicitly contains `early`/`no early`, so no behavioral event reconstruction was attempted.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `early` maps to 1 and anything else to 0, repeated over time. All early-lick trials are filtered first, so the saved output is always zero.

ii.
```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
out[2, :] = early_val
```

iii. The notes explicitly call its perfect decoding “trivial” because early trials were filtered, but accept that as reference-code consistency.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y) and 2 (likelihood) of `Camera0_side_TongueTracking`.

ii.
```python
tt_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_ts = tt_obj.timestamps[:]
td = tt_obj.data[:]
tongue_y_arr, tongue_lk_arr = td[:, 1], td[:, 2]
```

iii. The notes identify this NWB time series as the side-camera tongue measurement used by the reference analysis.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Likelihood must exceed 0.9. Percentile thresholds are computed from all qualifying raw frames inside retained trial windows. For each output bin, qualifying frames are averaged and categorized; bins without qualifying frames default to class 1 (`mid`).

ii.
```python
good = lk > 0.9
concat = np.concatenate(all_ty)
tongue_y_p40 = np.percentile(concat, 40)
...
ty = np.ones(n_bins, dtype=np.int64)
```

iii. The notes justify >0.9 as high-confidence DLC tracking and explicitly choose mid as the default when no detection exists.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of qualifying raw frame y-values form thresholds: below p40=0, p40 through p60=1, above p60=2. There is no required fourth `not visible` category.

ii.
```python
tongue_y_p40 = np.percentile(concat, 40)
tongue_y_p60 = np.percentile(concat, 60)
ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. The agent cites the requested per-session percentile split but omits the specified category 3 and computes thresholds over frames rather than the values actually classified (bin means).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Each neural bin center is converted to absolute time; camera frames within center ±25 ms are selected via `searchsorted` and averaged.

ii.
```python
bc = go + bin_centers[b]
il = np.searchsorted(tongue_ts, bc - half_bin)
ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The common absolute timestamp clock and identical 50 ms intervals align video bins with neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units and sessions with fewer than two valid trials are skipped; unrecorded trials are excluded using `obs_intervals`; missing tone gives NaNs; missing/low-confidence tongue data is silently imputed as mid; one all-zero boundary neural trial was retained.

ii.
```python
if n_good == 0: return None
if len(valid_trials) < 2: return None
tft = np.full(n_bins, np.nan, dtype=np.float32)
ty = np.ones(n_bins, dtype=np.int64)
```

iii. The notes call the single all-zero boundary trial acceptable, and describe the mid-class tongue default as an edge-case policy. They report fixing partial-recording handling after discovering `obs_intervals` mismatches.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant work is nested per-trial/per-neuron spike filtering and histogramming, followed by per-trial/per-bin tongue searches. NWB array loading and writing the multi-gigabyte pickle are also substantial.

ii.
```python
for go in go_times_arr:
    for i, st in enumerate(spike_times_list):
        counts, _ = np.histogram(st_w, bins=bin_edges)
...
for b in range(n_bins):
    il = np.searchsorted(tongue_ts, bc - half_bin)
```

iii. The notes report roughly 10–12 seconds per session and about 30 minutes total, with trial processing explicitly timed.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural bin edges could be flattened across all trials so only a per-unit `searchsorted` loop remains. Tongue bins could be assigned in bulk rather than searching timestamps for every trial/bin. Trial-to-observation matching and list-index assembly could also use vectorized/dictionary lookups.

ii.
```python
for go in go_times_arr:
    for i, st in enumerate(spike_times_list): ...
for b in range(n_bins): ...
subj_idx.append(subjects.index(s['subject_id']))
```

iii. The agent called `np.histogram` and `np.searchsorted` efficient, but did not discuss the avoidable outer loops; the much slower reported runtime reflects these choices.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly masks each unit's complete spike vector for every trial, constructs edges per trial, searches camera timestamps twice per bin, and linearly searches subject/region lists during assembly. Tongue data is traversed once for percentiles and again for output bins.

ii.
```python
mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
...
il = np.searchsorted(tongue_ts, bc - half_bin)
ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The notes do not identify these repetitions; they emphasize successful validation rather than profiling or eliminating redundant passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes session performance and left/right correct counts only for logging/metadata; loads and processes photostimulation fields although all stimulated trials are discarded; loads `trial_stops` chiefly for first-tone lookup; and optional plots sort/aggregate arrays without affecting conversion.

ii.
```python
perf = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0.0
correct_left = np.sum(...)
correct_right = np.sum(...)
```

iii. Performance was used as a sanity statistic after the agent decided not to session-filter. The agent acknowledges photostimulation is always zero but retained its construction to satisfy the requested schema.
