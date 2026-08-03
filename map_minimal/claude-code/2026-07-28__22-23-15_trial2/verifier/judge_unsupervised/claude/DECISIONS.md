# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (Neurodata Without Borders) files using the `pynwb` library. It globs for all NWB files matching the pattern `data/sub-*/sub-*_ses-*.nwb`, sorts them, and iterates through each file. Each NWB file represents one session. The trial table, behavioral events, unit spike times, and tongue tracking are all extracted from the NWB file structure. The reference code was designed for .mat files exported from DataJoint, but the AI correctly adapted to the NWB format available on DANDI.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
# ...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
```

```python
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
trials = nwb.trials.to_dataframe()
be = nwb.acquisition['BehavioralEvents']
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI recognized that the data was in NWB format (from DANDI) rather than the .mat format used by the reference code, and adapted accordingly. The trajectory shows it explored the NWB file structure to identify the correct fields for trials, units, behavioral events, and tongue tracking.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `nwb.subject.description` (e.g., 'SC015') from each NWB file. An `OrderedDict` tracks unique subjects in encounter order. A `subject_idx` array maps each session to its subject index.

ii.
```python
subject_desc = nwb.subject.description  # e.g. 'SC015'
# ...
all_subjects = OrderedDict()
if sub_desc not in all_subjects:
    all_subjects[sub_desc] = result['subject_id']
# ...
subjects = list(all_subjects.keys())
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. The AI used the subject description field from NWB metadata, which contains the mouse ID (e.g., SC015). The final dataset has 28 subjects, consistent with the data.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed independently, and those passing quality filters are added to the output list.

ii.
```python
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
    if result is not None:
        all_sessions.append(result)
```

iii. The AI identified 174 NWB files. After filtering, 144 sessions were retained. The paper reports 173 behavioral sessions (one fewer NWB file than sessions).

## 1-d. How are the data split into trials?

i. Trials are extracted from the NWB trials table (`nwb.trials.to_dataframe()`). Each row is one trial. The go cue time for each trial comes from `BehavioralEvents/go_start_times`. Trials are filtered by obs_intervals coverage and quality criteria.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_total = len(trials)
go_start_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_start_times) == n_trials_total
```

iii. The AI verified that the number of go cue timestamps matches the number of trials.

## 1-e. How are trials filtered based on quality controls?

i. Three levels of filtering are applied:
1. **Session-level**: Performance >65% on control trials, and >=50 correct left + right control trials. Performance is computed on ALL session trials (not just recorded ones). Control trials exclude photostim, auto_water, free_water, early lick, and ignore trials.
2. **obs_intervals coverage**: Only trials covered by the neural recordings (determined by matching obs_intervals to trial start times).
3. **Trial-level**: auto_water and free_water trials are excluded. Early lick and ignore (no-response) trials are kept.

ii.
```python
# Session filtering
all_valid = (auto_water == 0) & (free_water == 0)
control_mask_all = (all_valid &
                    (early_lick == 'no early') &
                    (outcome != 'ignore') &
                    no_photostim)
n_correct = ((outcome == 'hit') & control_mask_all).sum()
performance = n_correct / n_control

# obs_intervals coverage
obs_0 = units.get_unit_obs_intervals(good_indices[0])
first_obs_start = obs_0[0, 0]
first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))
recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)

# Trial-level filtering
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. The AI followed the paper's criteria: ">65% performance, at least 50 correct lick left and lick right trials each". It correctly included early lick and ignore trials (needed as decoder outputs) while excluding auto_water and free_water trials. The methods text says "Early lick trials and no response trials were excluded for analysis" but the decoder task requires these as output variables, so keeping them is correct.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from unit spike times obtained via `units.get_unit_spike_times(ui)` for each good unit in the NWB file.

ii.
```python
units = nwb.units
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. The AI used the standard pynwb interface to access spike times from the units table.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins spanning -2.5s to +1.5s relative to go cue (80 bins total). Spike counts per bin are converted to firing rates (spikes/s) by dividing by bin width. For each trial, spikes are filtered to the time window using `searchsorted`, then histogrammed.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)

# Per trial:
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
for i, st in enumerate(all_spike_times):
    lo = np.searchsorted(st, abs_start)
    hi = np.searchsorted(st, abs_end)
    if hi > lo:
        rel_spikes = st[lo:hi] - go_time
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
        trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The reference code uses 40ms bins with 3.4ms stride (sliding/overlapping histogram), but the decoder task instructions specify 50ms bins. The AI correctly followed the decoder instructions for bin size while adapting the approach to NWB spike time data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by two criteria: (1) `classification == 'good'` from the NWB units table, and (2) a valid (non-empty) CCF annotation (`anno_name`) that maps to one of 14 brain regions.

ii.
```python
classification = units['classification'].data[:]
anno_names = units['anno_name'].data[:]

good_indices = []
unit_regions = []
for ui in range(n_units_total):
    if classification[ui] != 'good':
        continue
    region = map_anno_to_region(anno_names[ui])
    if region is not None:
        good_indices.append(ui)
        unit_regions.append(region)
```

iii. The `classification` field in NWB corresponds to the output of the region-specific QC classifiers described in the methods text and spike sorting white paper. The reference code uses separate QC files (`goodunits/*.mat`) that contain lists of good unit IDs — the NWB classification field is the equivalent information. The AI obtained 57,935 good units across 144 sessions (paper reports 69,943 across 173 sessions; difference due to fewer sessions passing filters).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, the go cue time is obtained from `BehavioralEvents/go_start_times`, and spikes are extracted in the window [go_time - 2.5s, go_time + 1.5s].

ii.
```python
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME  # go_time - 2.5
abs_end = go_time + END_TIME      # go_time + 1.5
rel_spikes = st[lo:hi] - go_time
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The instructions specify "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue". The AI followed both requirements.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins. No rebinning is applied — spikes are directly histogrammed into these bins. The reference code uses 40ms bins with 3.4ms stride for a sliding histogram, but the AI used the decoder task parameters instead.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

iii. The decoder instructions specify "50-ms-width bins". The AI correctly used this value rather than the reference code's 40ms bins. No temporal rebinning is needed since the data is binned directly from raw spike times.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from the task structure: the tone onset (sample start) is computed as a fixed offset of -1.85s relative to the go cue. This offset comes from the sample period (0.65s) + delay period (1.2s) = 1.85s.

ii.
```python
TONE_ONSET_REL = -1.85  # tone onset relative to go cue (sample 0.65s + delay 1.2s)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. The AI verified this by comparing `sample_start_times` to `go_start_times` across sessions (trajectory step 30) and found it was consistently 1.85s. The task structure confirms this: 3 tones x 150ms + 2 gaps x 100ms = 650ms sample + 1200ms delay = 1850ms.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `bin_center - (-1.85)` for each bin center. This is a deterministic function of the bin structure and is the same for all trials. The resulting range is [-0.6, 3.3] seconds.

ii.
```python
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
# Precomputed once, reused for all trials
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. Since the tone onset relative to go cue is fixed (1.85s), and all trials are aligned to go cue, the time from tone onset is identical for all trials. This was precomputed once as a constant array.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time from tone onset uses the same bin centers as the neural data (80 bins from -2.5s to +1.5s relative to go cue). The first bin center is at -2.475s relative to go cue, corresponding to -0.625s relative to tone onset.

ii.
```python
# Same BIN_CENTERS used for neural data and input
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. Perfect alignment is achieved by using the same bin structure for both neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `photostim_start_times` and `photostim_stop_times` timestamps in the NWB `BehavioralEvents` acquisition.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. These timestamps contain the absolute start and stop times of photostimulation events across the entire session.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is created for each trial. For each photostim event that overlaps with the trial's time window, bin centers falling within [ps_start, ps_stop) are set to 1.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float32)
trial_start_abs = go_time + BEGIN_TIME
trial_end_abs = go_time + END_TIME

for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        rel_start = ps_start - go_time
        rel_stop = ps_stop - go_time
        mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
        photostim_binary[mask] = 1.0
```

iii. The AI created a time-varying binary signal rather than a per-trial scalar, which matches the instruction "Whether photostimulation is on at every time point (discrete, time-varying)".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation binary signal uses the same bin centers as the neural data. Photostim timestamps are converted to go-cue-relative times and compared to bin centers.

ii.
```python
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. Same bin structure ensures alignment with neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from the `trial_instruction` column (left/right) and the `outcome` column (hit/miss/ignore) from the NWB trials table.

ii.
```python
trial_instruction = trials['trial_instruction'].values
outcome = trials['outcome'].values
instr = trial_instruction[trial_idx]
outc = outcome[trial_idx]
```

iii. The AI inferred the lick direction from the combination of instruction and outcome rather than using raw lick direction data directly.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The choice is inferred:
- Hit (correct): choice matches instruction (left instruction -> 0, right -> 1)
- Miss (wrong lick): choice is opposite of instruction (left instruction -> 1, right -> 0)
- Ignore (no response): choice is set to instruction direction (left -> 0, right -> 1)

ii.
```python
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. For hit and miss trials, this correctly infers the actual lick direction. For ignore trials (no response), the animal didn't actually lick, so assigning instruction direction is arbitrary. The reference code has `lick_directions` data available per trial, which could provide the actual lick direction when available.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from the `outcome` column of the NWB trials table, which contains values 'ignore', 'miss', and 'hit'.

ii.
```python
outcome = trials['outcome'].values
outc = outcome[trial_idx]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

iii. The mapping matches the decoder instructions (ignore=0, miss=1, hit=2).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping from string values to integers: 'ignore' -> 0, 'miss' -> 1, 'hit' -> 2.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

iii. No additional processing. The mapping matches the instructions exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from the `early_lick` column of the NWB trials table, which contains values like 'early' and 'no early'.

ii.
```python
early_lick = trials['early_lick'].values
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The early_lick field in NWB directly indicates whether an early lick occurred.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: 'early' -> 1, anything else -> 0.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. Matches the decoder instructions (no=0, yes=1).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `Camera0_side_TongueTracking` in the NWB `BehavioralTimeSeries` acquisition. Column index 1 (y-coordinate) is used.

ii.
```python
bt = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. The tongue tracking data comes from DeepLabCut video analysis of a side-view camera.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per-session percentile thresholds are computed from ALL tongue y data in the session (no confidence/likelihood filtering). The 40th and 60th percentiles are used as thresholds.

ii.
```python
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
```

iii. The CONVERSION_NOTES incorrectly state "33rd/67th percentile thresholds on valid (likelihood > 0.9) tongue_y values", but the actual code uses 40th/60th percentiles with no likelihood filtering. The 40th/60th matches the decoder instructions. However, no confidence filtering is applied, which means low-confidence tracking points (e.g., when tongue is not visible) are included in both the percentile computation and the per-bin values.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories based on per-session percentiles:
- 0 (low): y < 40th percentile
- 1 (mid): 40th <= y <= 60th percentile
- 2 (high): y > 60th percentile

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. This matches the decoder instructions. The sequential assignment logic correctly creates the three categories: values start at 0, those >= p40 become 1, then those > p60 become 2.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each time bin center, the closest tongue tracking timestamp is found using `searchsorted` with a nearest-neighbor correction. The tongue y-value at that closest timestamp is used.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]
tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. The nearest-neighbor approach ensures each 50ms neural bin is matched to the closest tongue tracking sample (collected at 300 Hz, i.e., ~3.3ms resolution).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- Sessions with no good units are skipped.
- Sessions with fewer than 2 valid trials are skipped.
- Trials outside obs_intervals coverage are excluded.
- Units with unmapped brain regions are excluded.
- Errors during session processing are caught and the session is skipped.

However, some issues are NOT handled:
- Session 34 has many all-zero neural trials (likely from obs_intervals mismatch) that are not filtered out.
- Session 131 has highly skewed tongue_y distribution (99.1% "low") suggesting poor tracking, but is included.
- No confidence filtering on tongue tracking data.

ii.
```python
# Error handling
try:
    result = process_session(nwb_file, verbose=verbose)
except Exception as e:
    print(f"  ERROR: {e}")
    continue

# Min trials check
if n_trials < 2:
    if verbose:
        print(f"  SKIP: only {n_trials} valid trials")
    io.close()
    return None
```

iii. The AI documented known issues in CONVERSION_NOTES (session 0 trial 159, session 131 tongue distribution) but did not implement fixes for them.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files (`NWBHDF5IO` and `io.read()`)
2. Extracting spike times per unit (`units.get_unit_spike_times(ui)`) — this is called for each good unit individually
3. Computing firing rates: looping over trials and units to bin spikes

ii.
```python
# Per-unit spike time extraction (called for each good unit)
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)

# Per-trial firing rate computation
for trial_idx in valid_indices:
    for i, st in enumerate(all_spike_times):
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
```

iii. The trajectory (step 50) notes the conversion was slow, with each session taking significant time due to per-unit spike time extraction.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The inner loop over units for firing rate computation (lines 312-319) — could use a vectorized histogram for all units at once.
2. The loop over photostim events for each trial (lines 350-356) — could be vectorized across all events.
3. The `map_anno_to_region` function is called per unit with string matching — could be done as a batch lookup.

ii.
```python
# Could be vectorized: per-unit firing rate loop
for i, st in enumerate(all_spike_times):
    lo = np.searchsorted(st, abs_start)
    hi = np.searchsorted(st, abs_end)
    if hi > lo:
        rel_spikes = st[lo:hi] - go_time
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
        trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The firing rate computation involves a nested loop (trials x units). While `searchsorted` and `histogram` are individually efficient, the Python-level loop over units adds overhead.

## 10-c. What processing does the code repeat multiple times?

i.
1. `go_start_times[trial_idx]` is accessed in both the neural and input/output loops (lines 307 and 342).
2. The obs_intervals are read per unit to find the minimum coverage (`units.get_unit_obs_intervals(ui)` in lines 266-272), even though all units tend to have the same intervals.
3. The photostim event loop iterates over ALL session events for every trial, even though most events are irrelevant.

ii.
```python
# Neural loop
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    # ... compute firing rates

# Input/output loop (same trials, same go_times)
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    # ... compute inputs/outputs
```

iii. The two main loops over trials could be merged into a single loop to avoid redundant indexing, though the computational cost of this redundancy is minimal.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
1. The full tongue tracking data (`tongue_y_all_data`, `tongue_timestamps`) is loaded for the entire session but only queried at 80 bin centers per trial.
2. The brain region mapping function (`map_anno_to_region`) performs extensive string matching for every unit, even though many units share the same annotation.
3. All photostim events are checked against every trial, even though only a subset of trials have photostim.
4. The code computes firing rates as float32 (continuous values), but these are only used as-is — no additional smoothing or normalization is applied that might require full precision.

ii.
```python
# Full tongue data loaded
tongue_y_all_data = tongue_ts.data[:, 1]  # ALL timepoints
tongue_timestamps = tongue_ts.timestamps[:]
# But only 80 values per trial are used
```

iii. These are minor efficiency concerns. The main potential source of wasted computation is loading and processing all tongue tracking samples when only a small fraction are needed per trial.
