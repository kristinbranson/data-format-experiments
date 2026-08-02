# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (Neurodata Without Borders) files stored in the `data` directory. It uses `glob` to find all files matching `data/sub-*/sub-*_ses-*.nwb`, then iterates over each NWB file using `pynwb.NWBHDF5IO`. Each NWB file corresponds to one session. The AI found 174 NWB files total.

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
```

iii. The AI noted that the data was in NWB format (DANDI archive version), not the .mat format used by the reference code. The NWB files contain the same data but in a standardized neuroscience format. The AI correctly identified 174 NWB files from 28 subjects.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from each NWB file's `nwb.subject.description` field (e.g., 'SC015'). An `OrderedDict` tracks unique subjects in order of first appearance. A `subject_idx` array maps each session to its subject index.

ii.
```python
subject_desc = nwb.subject.description  # e.g. 'SC015'
# ...
all_subjects = OrderedDict()
# ...
sub_desc = result['subject_desc']
if sub_desc not in all_subjects:
    all_subjects[sub_desc] = result['subject_id']
```

iii. The AI identified 28 unique subjects. Subject identity is extracted per-session from the NWB metadata, consistent with the reference approach.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed independently. After filtering, 144 of 174 sessions are retained.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
    if result is not None:
        all_sessions.append(result)
```

iii. The AI noted that the paper reports 173 behavioral sessions but found 174 NWB files. 30 sessions were filtered out (21 low performance, 6 insufficient correct trials, 1 no good units, 2 other errors), yielding 144 valid sessions.

## 1-d. How are the data split into trials?

i. Trials are extracted from `nwb.trials.to_dataframe()`. Go cue times come from `BehavioralEvents['go_start_times']`. Only trials covered by neural recordings (`obs_intervals`) and passing trial-level filters (no `auto_water`, no `free_water`) are included. The AI aligns obs_intervals to trials by matching the first obs_interval start time to the closest trial start time.

ii.
```python
trials = nwb.trials.to_dataframe()
go_start_times = be.time_series['go_start_times'].timestamps[:]
# ...
obs_0 = units.get_unit_obs_intervals(good_indices[0])
n_obs_trials = len(obs_0)
# ...
trial_starts = trials['start_time'].values
first_obs_start = obs_0[0, 0]
first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))
recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
```

iii. The AI discovered during development that obs_intervals don't always start at trial 0 — some sessions start recording mid-session. The fix matches obs_interval start times to trial start times via nearest-neighbor lookup. However, verification output shows session 34 still has ~125 trials with all-zero neural data, suggesting the alignment may be imperfect for some sessions.

## 1-e. How are trials filtered based on quality controls?

i. Trials with `auto_water == 1` or `free_water == 1` are excluded. All other trials (including early lick and ignore/no-response) are kept. Only trials covered by obs_intervals are included.

ii.
```python
valid_trial_mask = np.zeros(n_trials_total, dtype=bool)
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. The AI noted that the reference code also excludes early lick and no-response trials for its analyses, but the decoder task instructions explicitly require early_lick and outcome (ignore) as decoder outputs, so these trials must be kept. This is consistent with the instructions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from individual unit spike times, accessed via `units.get_unit_spike_times(ui)` from each NWB file. Only units passing quality control filters are used.

ii.
```python
units = nwb.units
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. The NWB units table contains spike times for all sorted units. The AI extracts absolute spike times for each good unit and later bins them relative to go cue onset.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins spanning [-2.5, 1.5]s relative to go cue onset (80 bins). Spike counts per bin are converted to firing rates by dividing by bin width (0.05s). The result is a (n_neurons, 80) matrix per trial.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
# ...
for i, st in enumerate(all_spike_times):
    lo = np.searchsorted(st, abs_start)
    hi = np.searchsorted(st, abs_end)
    if hi > lo:
        rel_spikes = st[lo:hi] - go_time
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
        trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The reference code uses a sliding histogram with 40ms width and 3.4ms stride, but the instructions explicitly specify 50ms bins. The AI uses non-overlapping 50ms bins via `np.histogram`, which matches the instruction's "50-ms-width bins" requirement. The conversion to firing rates (spikes/s) is standard.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` (from automated quality control classifiers) and a valid CCF annotation that maps to one of 14 brain regions are included.

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

iii. The AI uses the `classification` field from the NWB units table, which corresponds to the automated classifier-based QC described in the spike sorting white paper. The reference code uses a separate QC file (`goodunits/*.mat`) with per-region classifiers. The AI's approach relies on the NWB-embedded classification labels, which should be equivalent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial, the go cue time is obtained from `BehavioralEvents['go_start_times']`, and spike times are referenced to this time. Bins span from go_time - 2.5s to go_time + 1.5s.

ii.
```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
# ...
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    abs_start = go_time + BEGIN_TIME
    abs_end = go_time + END_TIME
    # ...
    rel_spikes = st[lo:hi] - go_time
    counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The instructions specify alignment to "Go cue onset" and extraction of "2.5 s before to 1.5 s after". The AI correctly uses go_start_times from the NWB BehavioralEvents and offsets of -2.5s to +1.5s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms non-overlapping bins, yielding 80 time bins per trial. No overlapping/sliding window is used (unlike the reference code's 40ms/3.4ms sliding approach). No rebinning is applied — data is binned directly from raw spike times.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

iii. The instructions explicitly specify "50-ms-width bins for computing firing rates." The AI follows this specification. The reference code uses a sliding histogram (40ms width, 3.4ms stride for some analyses, 100ms width/50ms stride for others), but the decoder task overrides these with 50ms non-overlapping bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Time from tone onset is computed as a fixed offset from bin centers. The tone onset is assumed to be at a constant -1.85s relative to go cue, derived from the task structure: sample epoch (650ms) + delay epoch (1200ms) = 1850ms.

ii.
```python
TONE_ONSET_REL = -1.85  # tone onset relative to go cue (sample 0.65s + delay 1.2s)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. The AI verified that sample_start_times relative to go_start_times are consistently ~1.85s across all sessions (trajectory step 30). The task structure is: 3 tones x 150ms + 2 gaps x 100ms = 650ms sample + 1200ms delay = 1850ms. This is used as a constant offset rather than per-trial lookup.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The processing is a simple subtraction: for each bin center, compute `bin_center - (-1.85)`, yielding a continuous time-from-tone variable. This is precomputed once as a constant array since it's the same for all trials.

ii.
```python
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
# Range: [-0.6, 3.3]
```

iii. The resulting range is [-0.625, 3.325]s (from the first bin center at -2.475 to the last at 1.475, minus -1.85). This is a continuous time-varying input as specified.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Since time from tone onset is computed from the same bin centers as the neural data, alignment is inherent. Both share the same 80-bin time axis centered on go cue onset.

ii.
```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
# TIME_FROM_TONE uses the same BIN_CENTERS as neural data
```

iii. The constant offset assumption ensures perfect alignment. Since tone onset is always 1.85s before go cue and bins are defined relative to go cue, the time-from-tone values are automatically aligned to neural bins.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `BehavioralEvents['photostim_start_times']` and `BehavioralEvents['photostim_stop_times']` timestamps from the NWB file.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. These timestamps record the absolute times of photostimulation onset and offset during the session.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, a binary array (80 bins) is created. For each photostim start/stop pair, any bin whose center falls within [photostim_start, photostim_stop) relative to go cue is marked as 1.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float32)
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        rel_start = ps_start - go_time
        rel_stop = ps_stop - go_time
        mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
        photostim_binary[mask] = 1.0
```

iii. The AI treats photostimulation as a time-varying binary input, as specified by the instructions ("Whether photostimulation is on at every time point"). The code checks overlap between each photostim interval and the trial window.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned by using the same bin centers as the neural data. Photostim start/stop times are converted to go-cue-relative times, then bin centers within those intervals are marked.

ii.
```python
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. The alignment is inherent since both use the same go-cue-relative time bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) columns from the NWB trials table.

ii.
```python
trial_instruction = trials['trial_instruction'].values
outcome = trials['outcome'].values
# ...
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. For hit trials, the animal licked correctly so choice matches the instruction. For miss trials, the animal licked the wrong side so choice is opposite to instruction. For ignore trials (no response), choice defaults to the instruction direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The choice is encoded as 0 (left) or 1 (right), then broadcast to all 80 time bins as a per-trial constant.

ii.
```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    # ...
], axis=0)
```

iii. Choice is a per-trial categorical variable. For hit and ignore trials, choice = instruction direction. For miss trials, choice = opposite direction. Broadcasting to all time bins follows the instruction's preference for time-varying format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `outcome` column of the NWB trials table, which contains 'hit', 'miss', or 'ignore' string values.

ii.
```python
outcome = trials['outcome'].values
# ...
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

iii. The encoding matches the instructions: ignore=0, miss=1, hit=2.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The outcome string is mapped to integer values (ignore=0, miss=1, hit=2) and broadcast to all 80 time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
# ...
np.full(N_BINS, outcome_val, dtype=np.int64),
```

iii. Direct string-to-integer mapping as specified in the instructions.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This question refers to *Outcome*, not *Distance to reward zone* (which is not part of this dataset). Outcome is a per-trial scalar that is broadcast to all 80 time bins, so it is trivially aligned with neural data.

ii.
```python
np.full(N_BINS, outcome_val, dtype=np.int64),
```

iii. No temporal alignment is needed since outcome is constant across the trial.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the `early_lick` column of the NWB trials table, which contains 'early' or 'no early' string values.

ii.
```python
early_lick = trials['early_lick'].values
# ...
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The encoding matches the instructions: no early lick=0, early lick=1.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The early_lick string is compared to 'early' to produce a binary value (0 or 1), then broadcast to all 80 time bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
# ...
np.full(N_BINS, early_lick_val, dtype=np.int64),
```

iii. Direct binary encoding as specified.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from DeepLabCut tracking data stored in `BehavioralTimeSeries['Camera0_side_TongueTracking']`. Specifically, column index 1 (y-coordinate) of the tracking data is used.

ii.
```python
bt = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. The tongue tracking data comes from DeepLabCut analysis of side-view camera video at 300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session-wide percentiles (40th and 60th) of tongue y-position are computed over ALL tongue tracking data in the session (no likelihood filtering). For each trial, tongue y values are looked up at each bin center using nearest-neighbor interpolation on timestamps.

ii.
```python
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
# ...
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

iii. The CONVERSION_NOTES state "Computed per-session using 33rd/67th percentile thresholds on valid (likelihood > 0.9) tongue_y values" but the code actually uses 40th/60th percentiles without likelihood filtering. The code matches the instructions; the CONVERSION_NOTES are inaccurate.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Tongue y values are discretized into 3 categories using session-wide 40th and 60th percentiles: 0 (< 40th), 1 (40th to 60th), 2 (> 60th).

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. The thresholding matches the instructions: 0 = below 40th percentile, 1 = between 40th and 60th, 2 = above 60th percentile. The boundary conditions are: values exactly at the 40th percentile get category 1; values exactly at the 60th percentile also get category 1 (since the condition for category 2 is strictly greater than p60).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each neural time bin center, the closest tongue tracking timestamp is found via `searchsorted` and nearest-neighbor comparison. The tongue y value at that closest timestamp is used.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
# nearest-neighbor refinement...
tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. Since tongue tracking is at 300 Hz (3.3ms intervals) and neural bins are 50ms, each bin center will have a tongue measurement within ~1.7ms, providing good temporal alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases:
- Sessions with no good units or insufficient trials are skipped entirely.
- If obs_intervals differ across units, the minimum count is used.
- Trials outside recorded intervals produce zero firing rates (no explicit handling).
- The verification output shows session 34 has ~125 trials with all-zero neural data, and session 0 has 1 such trial, indicating imperfect obs_interval alignment.
- Session 131 has nearly all "low" tongue y-position (99.1%), suggesting poor tongue tracking.
- No likelihood filtering is applied to tongue tracking data, so invalid DLC predictions (tongue not visible) are included at their raw values.

ii.
```python
# obs_intervals: take minimum across units
for ui in good_indices[1:]:
    oi = units.get_unit_obs_intervals(ui)
    if len(oi) != n_obs_trials:
        n_obs_trials = min(n_obs_trials, len(oi))

# Missing/empty spikes default to zero firing rate
trial_fr = np.zeros((n_good, N_BINS), dtype=np.float32)
```

iii. The AI documented some data quality issues in CONVERSION_NOTES.md but did not implement fixes for all of them. The zero-neural-data trials in session 34 represent a bug in obs_interval alignment that was not fully resolved.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Reading NWB files and extracting spike times for each unit (`units.get_unit_spike_times(ui)`) — requires HDF5 reads.
2. Computing firing rates by binning spike times per unit per trial — involves nested loops over units and trials.
3. Computing tongue y nearest-neighbor lookup per trial.

ii.
```python
# Spike time extraction (O(n_good_units) HDF5 reads per session)
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)

# Firing rate computation (O(n_good_units * n_trials))
for trial_idx in valid_indices:
    for i, st in enumerate(all_spike_times):
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
```

iii. The agent noted in the trajectory (step 50) that processing was slow, identifying spike time extraction as the main bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over units for firing rate computation could potentially be vectorized by concatenating all spike times and using vectorized histogram operations. The photostim loop over all start/stop pairs per trial could be vectorized.

ii.
```python
# This loop over units could be vectorized:
for i, st in enumerate(all_spike_times):
    lo = np.searchsorted(st, abs_start)
    hi = np.searchsorted(st, abs_end)
    if hi > lo:
        rel_spikes = st[lo:hi] - go_time
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
        trial_fr[i, :] = counts / BIN_WIDTH

# This loop over photostim events could be vectorized:
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        # ...
```

iii. The unit loop uses searchsorted which is already efficient per-unit. True vectorization would require restructuring the data (e.g., a single sorted spike array with unit labels). The photostim loop is typically short (few stimulation events per session).

## 10-c. What processing does the code repeat multiple times?

i. The code iterates over `valid_indices` twice: once for neural data (lines 306-321) and once for inputs/outputs (lines 341-401). These could be combined into a single pass. The go_time lookup `go_start_times[trial_idx]` is repeated in both loops.

ii.
```python
# First loop: neural data
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    # ... compute firing rates ...

# Second loop: inputs/outputs
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    # ... compute inputs and outputs ...
```

iii. Splitting the loops doesn't cause correctness issues but doubles the iteration overhead and go_time lookups. In practice, the neural data computation dominates runtime.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code broadcasts per-trial scalar outputs (choice, outcome, early_lick) to all 80 time bins using `np.full(N_BINS, ...)`. If the decoder treats these as per-trial constants, the temporal replication is unnecessary memory overhead. The code also computes tongue y-position for all time bins including pre-trial periods (before -2.5s relative to go cue) where the tongue may not be active.

ii.
```python
# Per-trial scalars broadcast to 80 time bins
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_lick_val, dtype=np.int64),
    tongue_y_disc.astype(np.int64),
], axis=0)
```

iii. The instructions state outputs "Can be time-varying or discrete values per trial. If at all possible, make it time-varying." Since choice, outcome, and early_lick are inherently per-trial and not time-varying, broadcasting them adds memory usage without information content. However, the format is valid per the instructions.
