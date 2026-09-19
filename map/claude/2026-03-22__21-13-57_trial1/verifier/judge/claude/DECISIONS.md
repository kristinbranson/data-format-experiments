# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files in `/app/data/sub-*/`. It uses `glob.glob` to find all NWB files, sorts them, and processes each with `pynwb`. Each file is opened, and units, trials, behavioral events, and tongue tracking are extracted. The NWB file handle is NOT closed inside a `with` block — instead `io.close()` is called manually after extracting all data into memory.

ii.
```python
def get_nwb_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))
```
```python
def load_nwb_session(nwb_path):
    import pynwb
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    # ... extract all fields ...
    io.close()
    return { ... }
```

iii. The AI documented in CONVERSION_NOTES.md that the data is organized as 28 subject directories with 174 NWB files total. The approach of loading all data from NWB files using pynwb is standard and correct.

## 1-b. How are the data split into subjects?

i. The AI reads `nwb.subject.subject_id` for each session and uses an `OrderedDict` to build the subjects list, maintaining insertion order. `subject_idx` maps each session to its subject's index in this list.

ii.
```python
subject_id = nwb.subject.subject_id
```
```python
subjects_set = OrderedDict()
# ...
sid = result['subject_id']
if sid not in subjects_set:
    subjects_set[sid] = len(subjects_set)
```

iii. The AI uses `subject_id` (numeric ID like `'440956'`), which is the standard NWB subject identifier. The reference also uses `nwb.subject.subject_id`. Both approaches give 28 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. The AI processes each file independently and collects results. Session order follows the sorted glob output.

ii.
```python
nwb_files = get_nwb_files()
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
```

iii. This is the natural way to split sessions given the NWB file organization.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The AI extracts all trial columns into a dictionary. Go cue times are read from `BehavioralEvents/go_start_times`. No explicit assertion checks that the number of go cues matches the number of trials.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trials_data = {
    'start_time': trials['start_time'][:],
    'stop_time': trials['stop_time'][:],
    'trial_instruction': trials['trial_instruction'][:],
    # ...
}
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The trials table provides the natural trial boundaries. The reference code includes an assertion `assert len(go) == len(trials)` to verify consistency; the AI does not.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies **session-level selection criteria** (performance > 65%, >=50 correct left, >=50 correct right) and **trial-level filtering** (exclude `auto_water == 1` and `free_water == 1` trials). It also excludes trials beyond the neural recording range by checking min/max spike times. It does NOT use `obs_intervals` from the NWB file.

ii.
```python
# Session selection
performance, correct_left, correct_right = compute_session_performance(data['trials_data'])
if performance < MIN_PERFORMANCE:  # 0.65
    return None
if correct_left < MIN_CORRECT_LEFT:  # 50
    return None
if correct_right < MIN_CORRECT_RIGHT:  # 50
    return None

# Trial filtering
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False

# Recording range filter
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
```

iii. The AI documents in CONVERSION_NOTES.md Step 5 that it keeps early lick, ignore, and stim trials since they are decoder inputs/outputs, and only excludes auto_water and free_water as "artificial conditions." The session selection criteria come from the data paper. However, the reference code does NOT apply session selection criteria — it only filters using `obs_intervals` and `free_water`. The AI's session filtering reduces sessions from 173 to 144, dropping 29 sessions and losing ~13,000 neurons and ~16,000 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`. The AI reads all spike times and the VectorIndex end indices, then extracts spike times for each unit classified as `'good'`.

ii.
```python
spike_times_vi = units['spike_times']
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])

for ui in good_indices:
    start_idx = 0 if ui == 0 else int(all_st_idx[ui - 1])
    end_idx = int(all_st_idx[ui])
    good_spike_times.append(all_spike_times[start_idx:end_idx])
```

iii. Same source variable as the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins and converted to firing rates (Hz) by dividing counts by bin width. The AI uses `np.histogram` per neuron per trial with `np.linspace`-based bin edges.

ii.
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
for n in range(n_neurons):
    st = spike_times_list[n]
    for t in range(n_trials):
        go_t = go_times[t]
        abs_start = go_t + align_start
        abs_end = go_t + align_end
        idx_lo = np.searchsorted(st, abs_start, side='left')
        idx_hi = np.searchsorted(st, abs_end, side='left')
        if idx_hi > idx_lo:
            rel_spikes = st[idx_lo:idx_hi] - go_t
            counts, _ = np.histogram(rel_spikes, bins=bin_edges)
            all_matrices[t, n, :] = counts / bin_width
```

iii. The approach is functionally correct — spike counts divided by bin width give Hz. The reference uses a more vectorized approach (`searchsorted` on all trial edges at once), but the result should be equivalent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by `classification == 'good'`. Additionally, the AI applies a second filter: units must have a valid brain region mapping (their `anno_name` must map to one of the 14 predefined major regions). Units whose annotation doesn't match any keyword in `REGION_MAPPING` are dropped.

ii.
```python
good_mask = classifications == 'good'
good_anno = data['good_anno_names']
region_labels = []
neuron_mask = []
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
    if region is not None:
        region_labels.append(region)
        neuron_mask.append(i)
```

iii. The reference only filters by `classification == 'good'` and does NOT drop units with unmappable brain regions. The reference uses a simple string split to get the region label. The AI's additional filtering could drop neurons that don't match any keyword.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to the go cue. For each trial, the absolute go cue time is used to define the window `[go_time + ALIGN_START, go_time + ALIGN_END]`, and spikes in that window are converted to relative times and binned.

ii.
```python
go_t = go_times[t]
abs_start = go_t + align_start
abs_end = go_t + align_end
rel_spikes = st[idx_lo:idx_hi] - go_t
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
```

iii. Correct alignment to go cue as specified in instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins spanning -2.5 s to +1.5 s relative to go cue. No rebinning — spikes are binned directly into 50 ms bins.

ii.
```python
BIN_WIDTH = 0.050
ALIGN_START = -2.5
ALIGN_END = 1.5
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80
```

iii. Matches the instruction requirements.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (the tone onset events) and `go_start_times` (the go cue). The AI finds the last `sample_start_times` entry within each trial's `[start_time, go_time]` range.

ii.
```python
sample_start_ts = be.time_series['sample_start_times'].timestamps[:]
# ...
def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    for i in range(n_trials):
        mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
        matching = sample_start_ts[mask]
        if len(matching) > 0:
            tone_onsets[i] = matching[-1]
```

iii. The reference uses `np.searchsorted(sample, go, side='left') - 1` to find the last sample start before each go cue. The AI's approach constrains the search to within each trial's time range, which is more conservative but achieves the same result since the last sample start before the go cue should always be within the trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset at each bin center is computed as `bin_center - (tone_onset - go_cue)`, giving seconds since the tone at each timepoint.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
tone_rel = tone_t - go_t
tone_input = (bin_centers - tone_rel).astype(np.float32)
```

iii. This is equivalent to the reference's `CENTERS + (go - tone)`. The formula is correct.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers aligned to the go cue. The bin centers are defined as `np.linspace(ALIGN_START + BIN_WIDTH/2, ALIGN_END - BIN_WIDTH/2, N_BINS)`.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```

iii. Aligned with neural data by using the same go-cue-relative time axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI reads the global `photostim_start_times` and `photostim_stop_times` event streams from `BehavioralEvents`, rather than the per-trial `photostim_onset` and `photostim_duration` columns in the trials table.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The reference uses the per-trial `photostim_onset` (relative to trial start) and `photostim_duration` from the trials table. Both approaches should give the same result since the event streams record the same photostimulation events. The AI's approach is less efficient because it iterates over ALL photostim events for each trial.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial and each photostim event pair (start, stop), the AI checks if the event overlaps the trial window, then marks bins where the bin center falls between start and stop as 1.

ii.
```python
def compute_photostim_input(go_times, photostim_start_ts, photostim_stop_ts, ...):
    for t in range(n_trials):
        go_t = go_times[t]
        ps = np.zeros(n_bins, dtype=np.float32)
        for si in range(len(photostim_start_ts)):
            ps_start = photostim_start_ts[si] - go_t
            ps_stop = photostim_stop_ts[si] - go_t
            if ps_stop < align_start or ps_start > align_end:
                continue
            for b in range(n_bins):
                bc = bin_centers[b]
                if bc >= ps_start and bc < ps_stop:
                    ps[b] = 1.0
```

iii. The result is a binary time series (0/1), same as the reference. However, the implementation is O(n_trials * n_photostim_events * n_bins) with nested Python loops, which is very inefficient. The reference uses vectorized comparisons.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim start/stop times are converted to go-cue-relative times by subtracting the go time. The same bin centers used for neural data are used to determine which bins are active.

ii. See 4-b code.

iii. Correctly aligned via the shared go-cue-relative time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI uses `trial_instruction` only, mapping 'left' to 0 and 'right' to 1. It does NOT derive the actual lick direction from the combination of instruction and outcome.

ii.
```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. This is INCORRECT. The instructions say "Lick direction choice (left, right, no lick, per-trial)". The AI codes `trial_instruction` (the instructed side), not the actual lick direction. On miss trials, the animal licked the opposite side, so the choice is wrong. On ignore trials, there was no lick, which should be coded as "no lick" but the AI codes it as left or right based on instruction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Simple mapping from `trial_instruction` to integer: left=0, right=1. No "no lick" category is included. The `output_values` only lists `['left', 'right']` with 2 values instead of 3.

ii.
```python
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
# output_values:
['left', 'right'],  # choice: 0=left, 1=right
```

iii. The reference derives choice from instruction × outcome: hit → same side as instruction, miss → opposite side, ignore → "no lick" (code 2). The AI's approach gives the wrong value for miss and ignore trials and is missing the "no lick" category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column in the trials table, which has values `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcomes_raw = td['outcome'][trial_indices]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```

iii. Same as the reference. Correct.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String-to-integer mapping: ignore=0, miss=1, hit=2. Broadcast across all 80 bins.

ii.
```python
np.full(N_BINS, outcomes[t], dtype=np.int64),  # outcome (per-trial, broadcast)
```

iii. Matches reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column in the trials table, with values `'no early'` and `'early'`.

ii.
```python
early_lick_raw = td['early_lick'][trial_indices]
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
```

iii. Same as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: 'early' → 1, anything else → 0. Broadcast across all 80 bins.

ii.
```python
np.full(N_BINS, early_licks[t], dtype=np.int64),  # early lick (per-trial, broadcast)
```

iii. Matches reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, columns: x (col 0), y (col 1), confidence (col 2), plus timestamps.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]  # (n_frames, 3): x, y, confidence
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI imputes low-confidence tongue positions with the session mean y-value (using threshold 0.9), then bins and computes percentiles over ALL values (including imputed). Percentiles are computed over the concatenation of all trial bin values.

ii.
```python
TONGUE_CONFIDENCE_THRESHOLD = 0.9
visible_mask = tongue_conf >= confidence_threshold
session_mean_y = np.mean(tongue_y[visible_mask])
tongue_y_imputed = tongue_y.copy()
tongue_y_imputed[~visible_mask] = session_mean_y

# Percentiles over ALL values including imputed
all_values = np.concatenate([t for t in tongue_y_trials])
p40 = np.percentile(all_values, 40)
p60 = np.percentile(all_values, 60)
```

iii. The reference uses confidence threshold 0.5, sets low-confidence values to NaN (not session mean), computes percentiles over session-wide 50ms bin means of only the visible frames, and uses a "not visible" class (3) for bins with no visible frames. The AI's approach of imputing with session mean and then computing percentiles over all values (including imputed) is quite different — it means the percentiles are dominated by the session mean value since ~90% of frames are occluded.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses 3 categories only: 0 (< 40th percentile), 1 (40th–60th), 2 (> 60th). There is NO "not visible" class. When p40 == p60, the AI adds a small epsilon to create distinct thresholds. The `output_values` lists `['low', 'mid', 'high']` — only 3 values.

ii.
```python
def discretize_tongue_y(tongue_y_trials, session_mean_y):
    all_values = np.concatenate([t for t in tongue_y_trials])
    p40 = np.percentile(all_values, 40)
    p60 = np.percentile(all_values, 60)
    if np.isclose(p40, p60):
        eps = max(1e-6, abs(p40) * 1e-4)
        p40 = p40 - eps
        p60 = p60 + eps
    discretized = []
    for trial_y in tongue_y_trials:
        d = np.zeros(len(trial_y), dtype=np.int64)
        d[trial_y >= p40] = 1
        d[trial_y >= p60] = 2
        discretized.append(d)
    return discretized
```

iii. The instructions specify 4 categories including "3: not visible". The reference implements all 4 classes. The AI only implements 3, missing the "not visible" class entirely because it imputes occluded values with the session mean instead of treating them as a separate category. The `output_values` has only 3 entries for tongue_y_position.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial, tongue frames within `[go_time + ALIGN_START, go_time + ALIGN_END)` are found, converted to go-cue-relative timestamps, and binned into the same 80 bins as neural data using `np.digitize`.

ii.
```python
window_start = go_t + align_start
window_end = go_t + align_end
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
bin_indices = np.digitize(trial_ts, bin_edges) - 1
```

iii. Aligned via the same go-cue-relative time axis. When no tongue data is available for a trial, the session mean is used for all bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **No good units**: Session skipped (returns None) — same as reference.
- **Trials beyond recording range**: Excluded by checking min/max spike times — reference uses `obs_intervals` instead.
- **Occluded tongue**: Imputed with session mean — reference uses NaN and a "not visible" class.
- **Missing tone onset**: Set to 0 (shouldn't happen per AI's comment).
- **No tongue data in trial**: Fill all bins with session mean.
- **Unmapped brain regions**: Neurons dropped.

ii.
```python
if len(good_indices) == 0:
    return None  # no good neurons → skip session

# Recording range check
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)

# Tongue imputation
tongue_y_imputed[~visible_mask] = session_mean_y
```

iii. The AI's approach to handling missing tongue data (imputation with mean) differs significantly from the reference (NaN + explicit "not visible" class). The recording range filter is a reasonable alternative to `obs_intervals` but may not be as precise.

## 10-a. What are the most time-consuming steps of the code?

i. According to CONVERSION_NOTES.md Step 7, the estimated times are: spike binning ~3.3s/session (~9.6 min total), NWB loading ~1.5s/session (~4.3 min total), tongue processing ~1.5s/session (~4.3 min total). Total estimated ~22 minutes for 174 sessions.

ii. N/A

iii. The spike binning is the dominant cost. The AI's nested loop approach (per-neuron × per-trial) is slower than the reference's vectorized approach.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The photostim computation has a triple-nested Python loop (trials × photostim events × bins) that could be fully vectorized. The spike binning has a double loop (neurons × trials) where the trial loop could be vectorized (as the reference does). The tongue processing has a per-trial loop with per-bin inner loop.

ii.
```python
# Photostim: triple nested loop
for t in range(n_trials):
    for si in range(len(photostim_start_ts)):
        for b in range(n_bins):
            if bc >= ps_start and bc < ps_stop:
                ps[b] = 1.0

# Spike binning: double loop
for n in range(n_neurons):
    for t in range(n_trials):
        counts, _ = np.histogram(rel_spikes, bins=bin_edges)
```

iii. The reference vectorizes the trial dimension in spike binning by flattening all trial edges into one array and using a single `searchsorted`. The photostim computation in the reference is fully vectorized.

## 10-c. What processing does the code repeat multiple times?

i. The NWB file is opened and read once per session — no redundant I/O. However, the photostim computation iterates over ALL session photostim events for every trial, checking overlap, which is redundant since most events won't overlap most trials.

ii. See 4-b code.

iii. The reference avoids this by using per-trial photostim columns from the trials table.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session performance metrics (correct left, correct right, overall performance) to apply session selection criteria. This computation involves iterating over all trials with control/no-early/non-ignore masks. The reference does NOT apply session selection criteria, so all this processing is absent in the reference. The AI also loads several data fields that aren't used in the final output (e.g., `left_lick_times`, `right_lick_times`, `subject_desc`).

ii.
```python
def compute_session_performance(trials_data):
    # ... complex filtering logic ...
    return performance, int(correct_left), int(correct_right)
```

iii. The session selection filtering itself is the unnecessary processing — it removes 29 sessions that the reference keeps, losing data.
