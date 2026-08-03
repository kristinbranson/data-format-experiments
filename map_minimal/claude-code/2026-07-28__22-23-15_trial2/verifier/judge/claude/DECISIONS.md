# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `glob.glob` to find all NWB files matching `sub-*/sub-*_ses-*.nwb` under the data directory, sorts them, and processes each with `pynwb.NWBHDF5IO`. Each file is opened, and trial, unit, and behavioral data are read from within (`nwb.trials`, `nwb.units`, `nwb.acquisition`). The NWB file handle is kept open throughout processing (not using a `with` block).

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
...
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
trials = nwb.trials.to_dataframe()
be = nwb.acquisition['BehavioralEvents']
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. NWB is the published format for this dataset. The glob pattern matches the DANDI directory structure. The AI found 174 NWB files.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.description` (e.g., `'SC015'`) as the subject identifier rather than `nwb.subject.subject_id` (a numeric string like `'440956'`). Subjects are collected in an `OrderedDict` keyed by description, preserving insertion order.

ii.
```python
subject_desc = nwb.subject.description  # e.g. 'SC015'
...
all_subjects = OrderedDict()
...
sub_desc = result['subject_desc']
if sub_desc not in all_subjects:
    all_subjects[sub_desc] = result['subject_id']
...
subjects = list(all_subjects.keys())
```

iii. The agent used the mouse name from `description` rather than the numeric `subject_id`, reasoning these are the identifiers used in the papers. The CONVERSION_NOTES reference "sub=SC015" style names.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each session is identified by the basename of the NWB file. Session order follows sorted file listing.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
...
sess_name = os.path.basename(nwb_file)
```

iii. The DANDI dataset stores one session per NWB file, so no splitting or grouping is needed.

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials.to_dataframe()`, one row per trial. The number of go-cue events is asserted to match the number of trial rows.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_total = len(trials)
go_start_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_start_times) == n_trials_total
```

iii. The trials table provides a clear trial-level structure, confirmed by matching go-cue counts.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies **two levels** of filtering:

**Session-level filtering**: Sessions must have >65% correct performance and >=50 correct lick-left and >=50 correct lick-right trials on control trials (no photostim, no auto_water, no free_water, no early lick, not ignore). This drops 30 of 174 sessions, keeping 144.

**Trial-level filtering**: Within kept sessions, trials with `auto_water != 0` or `free_water != 0` are excluded. Trials outside the neural recording window (`obs_intervals`) are also excluded. A minimum of 2 valid trials per session is required.

ii.
```python
# Session filtering
control_mask_all = (all_valid &
                    (early_lick == 'no early') &
                    (outcome != 'ignore') &
                    no_photostim)
performance = n_correct / n_control
if performance < MIN_PERFORMANCE:
    ...
    return None
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    ...
    return None

# Trial filtering
if auto_water[ti] == 0 and free_water[ti] == 0:
    valid_trial_mask[ti] = True
```

iii. The agent adopted session filtering criteria from the published paper's methods: "overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." The auto_water and free_water filters remove training/hint trials where outcomes are artificially forced. The CONVERSION_NOTES state "144 of 174 sessions passed (30 skipped)."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units.get_unit_spike_times(ui)` for each good unit, which returns per-unit spike times in session-absolute seconds.

ii.
```python
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. Spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. For each trial and each good unit, spike times within the trial window are extracted using `searchsorted`, then binned using `np.histogram` with 50ms bin edges. Counts are divided by the bin width (0.05s) to convert to firing rates in Hz. This is a **nested loop**: outer loop over trials, inner loop over units.

ii.
```python
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    trial_fr = np.zeros((n_good, N_BINS), dtype=np.float32)
    for i, st in enumerate(all_spike_times):
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
        if hi > lo:
            rel_spikes = st[lo:hi] - go_time
            counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
            trial_fr[i, :] = counts / BIN_WIDTH
    neural_data.append(trial_fr)
```

iii. The 50ms bin width and firing rate conversion match the instructions. No smoothing or normalization is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must satisfy TWO criteria: `classification == 'good'` AND have a valid (non-empty, mappable) `anno_name` that maps to one of 14 brain regions. If `map_anno_to_region` returns `None`, the unit is excluded.

ii.
```python
for ui in range(n_units_total):
    if classification[ui] != 'good':
        continue
    region = map_anno_to_region(anno_names[ui])
    if region is not None:
        good_indices.append(ui)
        unit_regions.append(region)
```

iii. The AI filters on `classification == 'good'` (from the QC classifier). The additional `anno_name` filter is applied because every unit must be assigned to one of the 14 brain regions; units without a mappable annotation cannot be included. The CONVERSION_NOTES report 57,935 good units across 144 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to the go cue. For each trial, the absolute start and end times of the window are computed as `go_time + BEGIN_TIME` and `go_time + END_TIME`. Relative spike times are computed as `spike_time - go_time` and binned with `np.histogram` using edges centered on the go cue.

ii.
```python
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
rel_spikes = st[lo:hi] - go_time
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. All NWB times share one session-absolute clock, so alignment to the go cue requires only subtracting the go cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins spanning -2.5s to +1.5s relative to the go cue, producing 80 time bins. No rebinning is applied; spike times are binned directly.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

iii. These parameters match the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI does NOT use any raw data variable for per-trial tone onset. Instead, it uses a **fixed constant** `TONE_ONSET_REL = -1.85` (seconds relative to go cue), derived from the task structure: 0.65s sample + 1.2s delay = 1.85s.

ii.
```python
TONE_ONSET_REL = -1.85  # tone onset relative to go cue (sample 0.65s + delay 1.2s)
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. The agent reasoned that since the task structure is fixed (sample=0.65s, delay=1.2s), tone onset is always 1.85s before the go cue. The agent noted that `sample_start_times` has more entries than trials due to early-lick replays and decided a constant offset was simpler and more reliable. From trajectory: "The sample and delay periods are fixed across trials at 650ms and 1.2s respectively, so I can use the constant offset of -1.85s."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time-from-tone-onset is computed as a single constant array: `BIN_CENTERS - (-1.85)`, giving the same values for every trial in every session. The result is a 1D array of 80 values.

ii.
```python
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
...
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. Because a fixed offset is used, no per-trial computation is needed. The same `TIME_FROM_TONE` array is reused for every trial.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers relative to the go cue. Since `TIME_FROM_TONE` is computed from `BIN_CENTERS`, they share the same time axis by construction.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. Same bin grid ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents` (session-level absolute timestamps of all photostimulation events).

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. These provide exact onset/offset timestamps for all photostim events in the session.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code iterates through ALL photostim start/stop pairs to find any that overlap with the trial window. For overlapping events, bin centers falling within the photostim interval are set to 1.0; otherwise 0.0.

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

iii. The approach produces a binary time series matching the bin grid. The iteration over all events is inefficient but functionally correct.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onsets/offsets are converted to go-cue-relative times (`ps_start - go_time`), then compared against `BIN_CENTERS` which are also relative to the go cue.

ii.
```python
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. Using go-cue-relative coordinates for both ensures alignment with the neural bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') columns of the trials table.

ii.
```python
instr = trial_instruction[trial_idx]
outc = outcome[trial_idx]
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. The lick direction is not stored directly but can be inferred from instruction and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For hit trials, choice equals the instructed side (correct lick). For miss trials, choice is the opposite side (wrong lick). For ignore trials (no lick), the AI assigns the **instructed side** as the choice. Choice is coded as 0=left, 1=right with **no third category for "no lick"**. The value is broadcast to all 80 time bins.

ii.
```python
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
...
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
])
```

```python
output_values = [
    ['left', 'right'],  # only 2 values, no 'no lick'
    ...
]
```

iii. The agent reasoned that for ignore trials, "the neural activity likely still encoded the instructed side" since the mouse failed to lick rather than choosing differently. This preserves a binary choice variable.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which contains 'ignore', 'miss', and 'hit'.

ii.
```python
outc = outcome[trial_idx]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

iii. The trials table stores outcome explicitly with the three categories needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. The value is broadcast to all 80 time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
...
np.full(N_BINS, outcome_val, dtype=np.int64),
```

iii. Matches the instructions: "ignore = 0, miss = 1, hit = 2".

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains 'no early' and 'early'.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early lick) or 1 (early lick). Broadcast to all 80 time bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
...
np.full(N_BINS, early_lick_val, dtype=np.int64),
```

iii. Matches the instructions: "no = 0, yes = 1".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `BehavioralTimeSeries`. Column 1 is tongue y-position. The corresponding timestamps are used for temporal alignment.

ii.
```python
bt = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. This is the only tongue tracking measurement in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes the 40th and 60th percentiles of **all raw tongue y values** across the entire session (no likelihood filtering). Then for each trial, the tongue y at each bin center is obtained via **nearest-neighbor interpolation** (finding the closest timestamp). The y-value is discretized into 3 categories: 0 (below 40th pctl), 1 (40th-60th pctl), 2 (above 60th pctl).

ii.
```python
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
...
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]

tongue_y_values = tongue_y_all_data[tongue_indices]
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. The CONVERSION_NOTES state "Computed per-session using 33rd/67th percentile thresholds on valid (likelihood > 0.9) tongue_y values." However, the actual code uses 40th/60th percentiles with NO likelihood filtering. The notes are inconsistent with the code. The agent initially considered likelihood filtering ("should probably filter for high-likelihood points") but the final code omits it.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories only: 0 (< 40th pctl), 1 (40th-60th pctl), 2 (> 60th pctl). There is NO "not visible" category. Bins where the tongue is retracted still get a y-value from the nearest camera frame (which may have low tracking confidence) and are classified into one of the three categories.

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

```python
output_values = [
    ...
    ['low', 'mid', 'high'],  # only 3 categories, no 'not visible'
]
```

iii. The instructions specify 3 categories (0, 1, 2) based on 40th/60th percentile. The AI follows this literally without adding a "not visible" class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center time, the nearest camera frame timestamp is found via `searchsorted` and distance comparison. The tongue y-value at that nearest frame is used.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
# ... nearest-neighbor selection ...
tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. Nearest-neighbor interpolation assigns one camera frame's y-value to each 50ms bin, rather than averaging all frames within the bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Sessions with no good units or failing performance criteria**: Dropped entirely (return None).
- **Trials outside obs_intervals or with auto_water/free_water**: Excluded.
- **Tongue tracking with no visible tongue**: No special handling; nearest-neighbor always returns a y-value regardless of tracking confidence, so bins with retracted tongue get classified using low-confidence tracker output.
- **Processing errors**: Caught with a try/except that prints the traceback and skips the session.

ii.
```python
try:
    result = process_session(nwb_file, verbose=verbose)
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    continue
```

iii. The try/except approach prevents a single bad file from crashing the entire pipeline. Missing tongue data is not handled separately since the nearest-neighbor approach always finds a frame.

## 10-a. What are the most time-consuming steps of the code?

i. The **nested loop** over trials (outer) and units (inner) for computing firing rates is the most computationally expensive step. For each trial, for each unit, it runs `searchsorted` to find spikes in the window, then `np.histogram` to bin them. This results in `n_trials * n_units` histogram calls per session. Loading the NWB file and reading spike times also contribute significantly.

ii.
```python
for trial_idx in valid_indices:        # outer: trials
    for i, st in enumerate(all_spike_times):  # inner: units
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. Each session with ~500 trials and ~400 units performs ~200,000 histogram operations.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The **per-trial loop** in the neural binning could be vectorized by flattening the bin edges across all trials (as the reference solution does). Instead of calling `np.histogram` once per trial per unit, one could use `np.searchsorted` on all trial edges simultaneously for each unit.

The **per-trial loop** for photostim could also be optimized: the inner loop iterates over ALL session photostim events for EACH trial, which is O(n_trials * n_photostim_events).

ii.
```python
# Current: nested loop
for trial_idx in valid_indices:
    for i, st in enumerate(all_spike_times):
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)

# Reference approach: vectorized over trials
edges = (go[:, None] + REL_EDGES[None, :]).ravel()
for r, u in enumerate(good):
    pos = np.searchsorted(s, edges).reshape(n_trials, N_BINS + 1)
    rates[r] = np.diff(pos, axis=1)
```

iii. The reference solution's approach of flattening edges across trials converts the O(n_trials * n_units) loop into O(n_units), a significant speedup.

## 10-c. What processing does the code repeat multiple times?

i. The **photostim event iteration** is repeated for every trial: the code loops through ALL session photostim start/stop pairs for each trial, even though most events don't overlap with the current trial window. Loading spike times via `units.get_unit_spike_times(ui)` is done separately for each unit rather than reading the ragged buffer once.

ii.
```python
for trial_idx in valid_indices:
    for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
        if ps_stop > trial_start_abs and ps_start < trial_end_abs:
            ...
```

iii. A more efficient approach would match each trial to its photostim event once using the trials table's `photostim_onset` and `photostim_duration` columns.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The **brain region mapping** (`map_anno_to_region`) implements a complex ~150-line function to map CCF annotations to 14 broad categories. This grouping is not required by the target format, which just needs a list of brain region names and per-neuron indices. The raw annotation names (or a simpler transformation) would suffice.

The **session performance statistics** (correct_left, correct_right, performance) are computed for ALL sessions but only used for filtering; they are not included in the output data structure.

ii.
```python
def map_anno_to_region(anno_name):
    """Map a CCF annotation name to one of 14 broad brain regions."""
    # ~120 lines of string matching
    ...
```

iii. The reference solution uses a simple `re.split(r'[,./]', anno)[0].strip().lower()` to get region labels, producing fine-grained region names directly from the CCF annotation.
