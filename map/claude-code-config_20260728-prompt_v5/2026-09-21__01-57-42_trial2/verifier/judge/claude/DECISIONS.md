# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored under `/app/data/sub-*/`. It uses `h5py` to directly read HDF5/NWB files rather than `pynwb`. All NWB files are found via `glob.glob('/app/data/sub-*/*.nwb')`, sorted alphabetically. Each file is opened with `h5py.File()` and processed one at a time in `process_session()`.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, show_processing=args.show_processing)
```

```python
def process_session(nwb_path, show_processing=False):
    f = h5py.File(nwb_path, 'r')
    subject_id = f['general']['subject']['subject_id'][()].decode()
    ...
```

iii. The AI chose h5py over pynwb for direct HDF5 access. CONVERSION_NOTES.md notes the dataset contains 28 subjects and 174 NWB files. The approach processes each file sequentially.

## 1-b. How are the data split into subjects?

i. Subject identity is read from `general/subject/subject_id` in each NWB file. Unique subject IDs are collected across all sessions, sorted, and indexed.

ii.
```python
subject_id = f['general']['subject']['subject_id'][()].decode()
```

```python
all_subject_ids = sorted(set(s['subject_id'] for s in all_sessions))
subject_to_idx = {sid: i for i, sid in enumerate(all_subject_ids)}
```

iii. The AI uses the numeric subject_id from the NWB file (e.g. '440956') as the subject identifier, consistent with the NWB standard.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. No grouping or splitting is needed. Sessions are identified by the NWB filename.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
```

```python
return {
    'subject_id': subject_id,
    'session_name': basename,
    ...
}
```

iii. CONVERSION_NOTES states 174 NWB files found, with 173 sessions surviving after filtering. The one dropped session has no good units.

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table in each NWB file. The number of trials is determined from the `id` field. Go cue times are verified to match the trial count.

ii.
```python
n_trials_total = f['intervals']['trials']['id'].shape[0]
...
go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
assert len(go_start_times) == n_trials_total
```

iii. The AI verifies that go cue count matches trial count as a consistency check.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out trials where `auto_water == 1` or `free_water == 1`. It does NOT check `obs_intervals` (which records which trials have spike data). Sessions with fewer than 2 valid trials are dropped.

ii.
```python
is_auto_or_free = (auto_water == 1) | (free_water == 1)
trial_mask = ~is_auto_or_free
trial_indices = np.where(trial_mask)[0]
n_trials = len(trial_indices)

if n_trials < 2:
    print(f"  SKIP {basename}: only {n_trials} valid trials")
    f.close()
    return None
```

iii. CONVERSION_NOTES states: "Remove only auto_water and free_water trials (not genuine behavioral trials). KEEP early lick trials (early_lick is a decoder output). KEEP ignore/no response trials (outcome is a decoder output). KEEP photostim trials (photostim is a decoder input)." The AI notes 1,061/90,605 (1.2%) zero-neural trials in the verification output, which are trials that would have been excluded by an obs_intervals check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` in the NWB file. Only units with `classification == 'good'` are used.

ii.
```python
spike_times_data = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
unit_spike_times = extract_unit_spike_times(spike_times_data, spike_times_index, good_indices)
```

iii. CONVERSION_NOTES identifies `units/spike_times` as the source for neural activity, consistent with the NWB format.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins aligned to the go cue (-2.5s to +1.5s, 80 bins). For each unit and each trial, spikes within the trial window are histogrammed using `np.histogram`. Counts are divided by bin width to get firing rates in Hz.

ii.
```python
def compute_firing_rates_fast(unit_spike_times_list, go_times, bin_edges):
    for ni in range(n_units):
        st = unit_spike_times_list[ni]
        for ti in range(n_trials):
            go_t = go_times[ti]
            t_lo = go_t + bin_edges[0]
            t_hi = go_t + bin_edges[-1]
            i_lo = np.searchsorted(st, t_lo)
            i_hi = np.searchsorted(st, t_hi)
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
            all_rates[ni, ti] = counts
    all_rates /= bin_width
```

iii. The AI uses 50ms bins as specified in the instructions. No smoothing, normalization, or baseline subtraction is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == b'good'` are kept. Sessions with no good units are dropped entirely (1 session dropped).

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)

if n_good == 0:
    print(f"  SKIP {basename}: no good units")
    f.close()
    return None
```

iii. CONVERSION_NOTES states: "Use classifier-based QC: keep units with classification == 'good'" and notes this yields 69,453 good units across 173 sessions, close to the paper's 69,943.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are binned relative to each trial's go cue onset time. Bin edges are computed as `BIN_EDGES + go_time` for each trial, so go cue corresponds to time 0.

ii.
```python
go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
...
go_times = go_start_times[trial_indices]
...
counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

iii. The AI aligns to go cue onset as specified in the instructions, using the go_start_times event timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins spanning -2.5s to +1.5s relative to go cue, yielding 80 time bins. No rebinning is applied; spike times are directly binned at this resolution.

ii.
```python
BIN_WIDTH = 0.05       # 50 ms bins
T_START = -2.5         # seconds before go cue
T_END = 1.5            # seconds after go cue
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

iii. Matches the instruction specification of 50ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI uses a FIXED constant offset (`TONE_ONSET_REL_GO = -1.85`) derived from the task protocol (sample epoch 0.65s + delay 1.2s = 1.85s before go cue). It does NOT use the actual `sample_start_times` event data.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # seconds before go cue
```

iii. CONVERSION_NOTES explains: "Initially used event-based lookup (find_tone_onset_for_trial), but discovered 12.5% of trials had incorrect timing due to early lick replay events in NWB. Fixed by using hardcoded TONE_ONSET_REL_GO = -1.85 based on task protocol."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes time_from_tone as `BIN_CENTERS - TONE_ONSET_REL_GO`, which yields a constant array for ALL trials: `BIN_CENTERS + 1.85`. This is the same for every trial in every session.

ii.
```python
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO  # same for all trials
```

```python
trial_input = np.stack([time_from_tone.astype(np.float32), photostim_ts], axis=0)
```

iii. Because the tone onset is fixed at -1.85s relative to go cue, the time_from_tone input is identical across all trials. The range is [0.625, 3.35] seconds.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same BIN_CENTERS array, so they are trivially aligned. Since time_from_tone is computed from the same bin centers used for neural binning, they share the same temporal grid.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO
```

iii. Alignment is automatic since the same bin grid is used.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`, with `start_time` and go cue times used to convert to go-cue-relative coordinates.

ii.
```python
photostim_onset = f['intervals']['trials']['photostim_onset'][:]
photostim_dur = f['intervals']['trials']['photostim_duration'][:]
```

iii. CONVERSION_NOTES identifies these as the source fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, photostim_onset (relative to trial start) is converted to go-cue-relative time. A binary time series is created where 1 indicates photostim is active (bin center falls within [onset, offset]) and 0 otherwise. Trials with `N/A` get all zeros.

ii.
```python
def get_photostim_intervals(f, n_trials, go_times, trial_start_times):
    for i in range(n_trials):
        if photostim_onset[i] == b'N/A':
            intervals.append(None)
        else:
            onset_rel_trial = float(photostim_onset[i])
            dur = float(photostim_dur[i])
            onset_abs = onset_rel_trial + trial_start_times[i]
            onset_rel_go = onset_abs - go_times[i]
            offset_rel_go = onset_rel_go + dur
            intervals.append((onset_rel_go, offset_rel_go))
```

```python
photostim_ts[(BIN_CENTERS >= onset_rel) & (BIN_CENTERS < offset_rel)] = 1.0
```

iii. The conversion from trial-relative to go-cue-relative timing matches the reference approach.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onset/offset are expressed relative to the go cue, and the binary series is evaluated at the same BIN_CENTERS used for neural data, ensuring alignment.

ii.
```python
photostim_ts[(BIN_CENTERS >= onset_rel) & (BIN_CENTERS < offset_rel)] = 1.0
```

iii. Same temporal grid as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `intervals/trials/outcome` and `intervals/trials/trial_instruction`. Choice is not stored directly in the NWB file.

ii.
```python
trial_outcomes = outcome[trial_indices]
trial_instructions = trial_instruction[trial_indices]
```

iii. CONVERSION_NOTES documents the derivation logic.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Hit + left instruction = left (0), hit + right instruction = right (1), miss + left instruction = right (1), miss + right instruction = left (0), ignore = no lick (2). The per-trial value is broadcast to all time bins.

ii.
```python
if out == 'hit':
    choice = 0 if inst == 'left' else 1
elif out == 'miss':
    choice = 1 if inst == 'left' else 0
else:  # ignore
    choice = 2
...
trial_output = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
])
```

iii. The logic correctly derives the animal's actual lick direction from the combination of instruction and outcome.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcome = np.array([x.decode() for x in f['intervals']['trials']['outcome'][:]])
```

iii. Direct mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. The per-trial value is broadcast to all time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]
...
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. Standard categorical encoding matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `intervals/trials/early_lick`, which contains 'early' or 'no early'.

ii.
```python
early_lick = np.array([x.decode() for x in f['intervals']['trials']['early_lick'][:]])
```

iii. Direct from trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: no early=0, early=1. Broadcast to all time bins.

ii.
```python
early_val = 0 if trial_early_lick[ti] == 'no early' else 1
...
np.full(N_BINS, early_val, dtype=np.int64)
```

iii. Binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains (x, y, likelihood) per frame. Column 1 is y-position, column 2 is likelihood.

ii.
```python
tongue_ts = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
tongue_data_raw = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
```

iii. Same source variable as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Visible y-values (likelihood >= 0.9) are collected from within trial windows only (not the whole session). 40th and 60th percentiles are computed on raw frame values (not bin means). Per trial, for each 50ms bin, the mean of ALL frames in the bin (including low-likelihood) is computed, and the mean likelihood is checked against the 0.9 threshold. If mean likelihood >= 0.9, the mean y is categorized; otherwise the bin is "not visible" (3).

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9

# First pass: collect visible y values for percentile computation
visible_y_all = []
for ti in range(n_trials):
    ...
    vis_mask = tongue_lh[i_lo:i_hi] >= likelihood_thresh
    if vis_mask.any():
        visible_y_all.append(tongue_y[i_lo:i_hi][vis_mask])

all_visible = np.concatenate(visible_y_all)
p40 = np.percentile(all_visible, 40)
p60 = np.percentile(all_visible, 60)

# Second pass: discretize per bin
for b in range(n_bins):
    b_mask = bin_idx_v == b
    avg_lh = trial_lh_v[b_mask].mean()
    if avg_lh >= likelihood_thresh:
        avg_y = trial_y_v[b_mask].mean()
        if avg_y < p40: trial_bins[b] = 0
        elif avg_y <= p60: trial_bins[b] = 1
        else: trial_bins[b] = 2
```

iii. CONVERSION_NOTES states: "session-level percentile discretization (40th/60th) on visible tongue data" with likelihood threshold of 0.9.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Categories are: 0 (below 40th percentile), 1 (40th to 60th percentile), 2 (above 60th percentile), 3 (not visible). The thresholds are the 40th and 60th percentiles of visible tongue y-values collected from trial windows.

ii.
```python
if avg_y < p40: trial_bins[b] = 0
elif avg_y <= p60: trial_bins[b] = 1
else: trial_bins[b] = 2
```

iii. The categorization follows the instructions' specification.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are used to find frames within each trial's time window (go_time + T_START to go_time + T_END). Frames are assigned to bins using `np.searchsorted` on the same bin edges used for neural data.

ii.
```python
rel_ts = trial_ts - go_t
bin_idx = np.searchsorted(bin_edges, rel_ts, side='right') - 1
```

iii. Uses the same go-cue-aligned bin grid as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled:
- Sessions with no good units (classification not 'good'): session is skipped entirely.
- Sessions with fewer than 2 valid trials: session is skipped.
- Tongue tracking with no visible frames: returns all bins as "not visible" (class 3).
- Exceptions during processing are caught and the session is skipped with an error message.

The AI does NOT handle trials without spike data (no obs_intervals check), resulting in ~1,061 zero-neural trials.

ii.
```python
if n_good == 0:
    print(f"  SKIP {basename}: no good units")
    f.close()
    return None
```

```python
try:
    result = process_session(nwb_file, ...)
except Exception as e:
    print(f"  ERROR processing {nwb_file}: {e}")
```

```python
if len(visible_y_all) == 0:
    return [np.full(n_bins, 3, dtype=np.int64) for _ in range(n_trials)], p40, p60
```

iii. CONVERSION_NOTES acknowledges 1,061 zero-neural trials but does not address them.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reports ~2.9s per session for the full conversion (500s total for 173 sessions). The firing rate computation with its nested unit x trial loop and per-trial `np.histogram` calls is likely the dominant cost within each session.

ii. N/A (timing is logged but the nested loop structure is the bottleneck)

iii. CONVERSION_NOTES states: "Processing time: ~500s (2.9s/session)"

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The firing rate computation has a nested loop over units AND trials:

ii.
```python
for ni in range(n_units):
    st = unit_spike_times_list[ni]
    for ti in range(n_trials):
        counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
        all_rates[ni, ti] = counts
```

The inner trial loop could be vectorized by flattening all trial bin edges into a single array and using one `np.searchsorted` call per unit (as the reference solution does). The tongue discretization also has a per-bin inner loop.

iii. The AI's CONVERSION_NOTES does not identify this inefficiency.

## 10-c. What processing does the code repeat multiple times?

i. The photostim intervals are computed for ALL trials (including filtered ones) in `get_photostim_intervals()`, then only the filtered subset is used. The tongue y visible values are collected in a first pass, then percentiles computed, then a second pass discretizes -- this two-pass approach is necessary.

ii.
```python
photostim_intervals = get_photostim_intervals(f, n_trials_total, go_start_times, trial_start_times)
```
This processes all `n_trials_total` trials, but only `trial_indices` (filtered subset) are used later.

iii. Not documented by the AI.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session performance statistics (`performance`) that are stored in metadata but not used by the decoder. The brain region mapping uses an elaborate keyword-based classification system that maps to abbreviated region names, which is more processing than needed (the reference simply lowercases and takes the first comma-separated component of the annotation name).

ii.
```python
is_control = (photostim_power == b'N/A') & (early_lick == 'no early') & ~is_auto_or_free
n_control_responded = np.sum((outcome[is_control] == 'hit') | (outcome[is_control] == 'miss'))
performance = n_correct / n_control_responded if n_control_responded > 0 else 0
```

iii. Performance computation is documented as metadata but not essential for the conversion task.
