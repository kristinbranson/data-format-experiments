# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files found via glob pattern `data/sub-*/*.nwb`. Each file is opened with `pynwb.NWBHDF5IO` and processed individually. Trials, units, behavioral events, and tongue tracking are extracted within each file's context.

ii.
```python
def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
```

```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    subject_id = nwb.subject.subject_id
    trial_instruction = nwb.trials['trial_instruction'][:]
    outcome = nwb.trials['outcome'][:]
    ...
    go_times = be.time_series['go_start_times'].timestamps[:]
    sample_start_times = be.time_series['sample_start_times'].timestamps[:]
```

iii. The AI's CONVERSION_NOTES.md states: "NWB files: data/sub-{id}/sub-{id}_ses-{datetime}_behavior+ecephys[+ogen].nwb". The AI identified 174 NWB files across 28 subject directories, consistent with the dandiset. This approach is standard for NWB data.

## 1-b. How are the data split into subjects?

i. Each NWB file provides `nwb.subject.subject_id`. The AI collects all unique subject IDs, sorts them, and maps each session to its subject index.

ii.
```python
subject_id = nwb.subject.subject_id
...
subjects = sorted(set(s['subject_id'] for s in all_sessions))
subj_idx.append(subjects.index(s['subject_id']))
```

iii. The AI uses the numeric subject_id from the NWB file (e.g., '440956') rather than the mouse name (e.g., 'SC015'). The CONVERSION_NOTES.md confirms 28 subjects were found, matching the dandiset.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. The sorted glob produces a deterministic session order. Sessions with no good neurons are skipped.

ii.
```python
nwb_files = get_nwb_files()
...
for i, path in enumerate(nwb_files):
    r = process_session(path, ...)
    if r: all_sessions.append(r)
    else: skipped += 1
```

iii. The AI's notes confirm 173 sessions with good neurons out of 174 total, matching the reference paper's count.

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials`, read as individual column arrays. The AI reads go_times from `BehavioralEvents/go_start_times`. Trials are indexed by position in the trials table.

ii.
```python
n_trials = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
...
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI does not explicitly assert that len(go_times) == n_trials, but uses go_times indexed by trial index throughout, implying a 1:1 correspondence.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies `get_regular_trial_mask` from the reference code, filtering out: (1) early lick trials, (2) auto water trials, (3) free water trials, (4) ignore (no response) trials, and (5) photostimulation trials. Additionally, only trials within `obs_intervals` are kept. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
mask_no_early = (early_lick == 'no early')
mask_no_auto = (auto_water == 0)
mask_no_free = (free_water == 0)
mask_no_ignore = (outcome != 'ignore')
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
recorded_set = set(recorded_trials)
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. The AI's CONVERSION_NOTES.md states: "Trial filtering: Apply get_regular_trial_mask (no early lick, no auto water, no free water, no ignore, no photostim)". The AI followed the reference *analysis* code's trial filter, which was designed for the paper's analyses, not for the decoder task specified in the instructions. The instructions explicitly list early lick, outcome (including ignore), and photostimulation as decoder outputs/inputs, meaning those trials should have been retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for units classified as 'good'. The go cue times from `BehavioralEvents/go_start_times` are used for temporal alignment.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
...
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
```

iii. The AI correctly identifies spike_times as the source of neural data and classification == 'good' as the quality filter.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins using `np.histogram` for each neuron and each trial. Counts are divided by bin_size to get firing rates in Hz.

ii.
```python
def bin_spikes_all_trials(spike_times_list, go_times_arr, t_start, t_end, bin_size):
    n_bins = int(round((t_end - t_start) / bin_size))
    n_neurons = len(spike_times_list)
    result = []
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

iii. The AI uses `np.linspace` for bin edges rather than an additive approach. With `np.linspace(-2.5+go, 1.5+go, 81)`, the edges should be equivalent to `go + T_START + BIN*np.arange(81)` up to floating point differences.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. Sessions with 0 good units are skipped.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```

iii. The AI's CONVERSION_NOTES.md states: "QC: classifier-based, stored as classification field in NWB units ('good'/'unlabelled')". This matches the reference approach of using the QC classifier verdict.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, bin edges are constructed as `np.linspace(t_start + go, t_end + go, n_bins + 1)`, centering the window around the go cue time.

ii.
```python
go_valid = go_times[valid_trials]
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

```python
for go in go_times_arr:
    bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The AI correctly aligns to the go cue as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins spanning -2.5s to +1.5s relative to go cue, giving 80 time bins per trial. No rebinning is applied; spikes are directly histogrammed into the final bins.

ii.
```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
n_bins = int(round((t_end - t_start) / bin_size))  # 80
```

iii. The temporal parameters match the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset events) and `go_start_times`. The AI finds the first `sample_start_times` event that falls within the trial's time window (`trial_start` to `trial_stop`).

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
def get_sample_start_for_trial(sample_start_times, trial_start, trial_stop):
    mask = (sample_start_times >= trial_start) & (sample_start_times <= trial_stop)
    return sample_start_times[mask][0] if np.any(mask) else None
```

iii. The AI's approach finds the first sample_start within the trial window, rather than the last one before the go cue. Since early lick trials can have multiple sample onsets (the sample epoch replays on early lick), the first sample_start within the trial is not necessarily the correct one to use. However, since the AI filters out early lick trials, each remaining trial should only have one sample start, making first vs. last equivalent in practice for the filtered data.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Time from tone onset is computed as `bin_centers - (sample_start - go)`, where bin_centers are the centers of the 50ms bins relative to the go cue.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
...
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The computation `bin_centers - (ss - go)` = `bin_centers + (go - ss)` gives the time from tone onset for each bin center, which is correct. If no sample start is found, NaN is used.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and time_from_tone_onset use the same bin_centers array defined relative to the go cue, so they are inherently aligned.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
```

iii. The bin centers used for time_from_tone_onset are the same as used for neural binning, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_power`, `photostim_onset`, and `photostim_duration` in the trials table, plus `start_time` and go cue times.

ii.
```python
photostim_power_raw = nwb.trials['photostim_power'][:]
photostim_onset_raw = nwb.trials['photostim_onset'][:]
photostim_dur_raw = nwb.trials['photostim_duration'][:]
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
```

iii. The AI uses `photostim_power` to determine whether photostimulation occurred, checking both for 'N/A' and for power > 0. The reference code uses `photostim_onset != 'N/A'` instead.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For trials with photostimulation, the onset is converted from trial-start-relative to go-cue-relative coordinates, and a binary time series is created where bin centers fall within [onset, onset+duration).

ii.
```python
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    d = float(photostim_dur_raw[trial_idx])
    ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. However, since photostim trials are filtered out by `regular_mask`, this code never actually produces non-zero values. The AI's CONVERSION_NOTES.md confirms: "Photostim input: Always 0 because photostim trials are filtered out."

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim onset is expressed relative to the go cue, using the same bin_centers as the neural data. However, since all photostim trials are filtered out, the alignment is moot.

ii.
```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The alignment logic is correct in principle but never exercised due to trial filtering.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from `trial_instruction` ('left'/'right'), mapped to 0/1. It does NOT use `outcome` to derive the actual lick direction.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. Because the AI filters out ignore trials, every remaining trial has a lick response. However, choice is mapped from the *instructed* side, not the *actual* lick direction. On miss trials, the animal licked the opposite side from the instruction, so this gives the wrong lick direction for miss trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as left=0, right=1 with only 2 classes. The value is per-trial, broadcast across all time bins.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
...
out[0, :] = choice
...
'output_values': [['left','right'], ...]
```

iii. The AI defines only 2 output values for choice (['left', 'right']), missing the 'no lick' category that the reference has. Since ignore trials are filtered out, the AI did not need a third class, but the filtering itself is problematic.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcome = nwb.trials['outcome'][:]
...
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
```

iii. The AI reads outcome directly from the trials table. However, since ignore trials are filtered out, outcome only ever takes values 'hit' or 'miss' in the final data.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are mapped to integers: ignore=0, miss=1, hit=2, and broadcast across time bins. But since ignore trials are filtered, outcome is effectively binary (miss=1 or hit=2).

ii.
```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
out[1, :] = out_val
```

iii. The encoding matches the instructions (ignore=0, miss=1, hit=2), but the data will never contain ignore=0 due to trial filtering.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing 'early' or 'no early'.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
...
early_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The variable is read correctly from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1 and broadcast across time bins. However, since early lick trials are filtered out, this output is always 0.

ii.
```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
out[2, :] = early_val
```

iii. The AI's CONVERSION_NOTES.md confirms: "early_lick is trivial because all early lick trials are filtered out by get_regular_trial_mask." The decoder achieves 100% accuracy because it's a constant.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 (tongue_y) and column 2 (likelihood).

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
if 'Camera0_side_TongueTracking' in bts.time_series:
    tt_obj = bts.time_series['Camera0_side_TongueTracking']
    tongue_ts = tt_obj.timestamps[:]
    td = tt_obj.data[:]
    tongue_y_arr = td[:, 1]
    tongue_lk_arr = td[:, 2]
```

iii. The same tracking data as the reference, correctly identified from the NWB file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood > 0.9 are considered valid. For percentile computation, the AI collects all valid tongue y values from valid trials only (not the whole session), concatenates them, and computes the 40th and 60th percentiles on these raw frame values. For each trial bin, frames within [bin_center - half_bin, bin_center + half_bin] with likelihood > 0.9 are averaged, then classified.

ii.
```python
# Percentile computation - across valid trials only, on raw frames
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
if all_ty:
    concat = np.concatenate(all_ty)
    tongue_y_p40 = np.percentile(concat, 40)
    tongue_y_p60 = np.percentile(concat, 60)
```

iii. Key differences from reference: (1) likelihood threshold is 0.9 vs reference's 0.5, (2) percentiles are computed on raw frame values rather than 50ms bin means, (3) percentiles use only valid trial data vs the whole session, (4) per-trial binning uses [center-half_bin, center+half_bin] rather than full bin edges.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Tongue y is classified as: 0 if below 40th percentile, 1 if between 40th and 60th (using strict inequality for >p60), 2 if above 60th percentile. When no valid frames exist in a bin, the default class is 1 (mid).

ii.
```python
ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

Default initialization:
```python
ty = np.ones(n_bins, dtype=np.int64)  # default to 1 (mid)
```

Output values:
```python
'output_values': [..., ['low','mid','high'], ...]
```

iii. The AI uses only 3 classes ['low','mid','high'] and defaults invisible bins to class 1 ('mid'). The reference uses 4 classes with an explicit 'not visible' class (value 3). The AI's approach biases the tongue_y distribution toward the middle class and loses information about tongue visibility.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each time bin, frames are selected by their timestamp falling within [bin_center - half_bin, bin_center + half_bin] in absolute time. This uses the same bin_centers as the neural data, ensuring alignment.

ii.
```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The bin selection approach (center ± half_bin) differs slightly from the reference's approach of using bin edges directly, but should select approximately the same frames.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good neurons are skipped. (2) Trials outside `obs_intervals` are excluded via `get_recorded_trial_indices`. (3) Tongue bins with no high-confidence frames default to class 1 (mid). (4) Trials with no matching sample_start get NaN for time_from_tone_onset.

ii.
```python
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None

# obs_intervals handling
recorded_trials = get_recorded_trial_indices(nwb, good_indices)

# Tongue default
ty = np.ones(n_bins, dtype=np.int64)  # default to 1

# Missing tone
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The AI handles the session with no good neurons correctly. The obs_intervals handling uses a more complex matching approach than the reference. Defaulting invisible tongue to class 1 loses information compared to the reference's explicit "not visible" class.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's code has a per-trial, per-neuron inner loop in `bin_spikes_all_trials`, making spike binning the dominant cost. The CONVERSION_NOTES estimate ~10s/session, ~30min total. The tongue per-bin loop is also slow.

ii.
```python
for go in go_times_arr:           # per trial
    for i, st in enumerate(spike_times_list):  # per neuron
        counts, _ = np.histogram(st_w, bins=bin_edges)
```

iii. The AI recognized the processing time (~10s/session) but did not optimize the nested loop structure.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning has a doubly-nested loop (trials × neurons) that could be vectorized. The reference code vectorizes the trial dimension by flattening all bin edges and using one `searchsorted` per unit. The tongue y per-bin loop could also be vectorized.

ii.
```python
# AI's double loop:
for go in go_times_arr:
    for i, st in enumerate(spike_times_list):
        counts, _ = np.histogram(st_w, bins=bin_edges)

# Reference's vectorized approach:
edges = (go[:, None] + REL_EDGES[None, :]).ravel()
for r, u in enumerate(good):
    pos = np.searchsorted(s, edges).reshape(n_trials, N_BINS + 1)
    rates[r] = np.diff(pos, axis=1)
```

iii. The trial dimension could be vectorized as in the reference code. The tongue y computation loops over both trials and bins, which could be improved.

## 10-c. What processing does the code repeat multiple times?

i. Spike times are loaded individually per unit via `nwb.units['spike_times'][idx]`, which may involve repeated reads from the underlying HDF5 ragged array. The reference reads the full spike_times buffer once.

ii.
```python
# AI reads per unit:
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]

# Reference reads buffer once:
offs = np.asarray(units['spike_times'].data)
allst = np.asarray(units['spike_times'].target.data)
```

iii. Reading spike times individually is less efficient than the reference's bulk read, though functionally equivalent.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session-level performance statistics (hit rate, correct left/right counts) that are stored in metadata but not used by the decoder. It also checks for photostimulation (has_photostim) and processes photostim timing, but since photostim trials are filtered out, the photostim input is always zero.

ii.
```python
# Performance computation not needed for decoder
ctrl_out = outcome[control]
n_hit = np.sum(ctrl_out == 'hit')
n_miss = np.sum(ctrl_out == 'miss')
perf = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0.0

# Photostim processing that always produces zeros
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    ...
```

iii. The performance statistics are informational but not used downstream. The photostim processing is entirely wasted since photostim trials are filtered out.
