# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data from NWB files in the `/app/data` directory. It iterates over subject directories (`sub-*`) and lists `.nwb` files within each. Each file is opened with `pynwb.NWBHDF5IO` and processed individually. Trials, units, behavioral events, and behavioral time series are extracted from each NWB file.

ii.
```python
def list_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sub-')])
    all_files = []
    for sub in subjects:
        sub_dir = os.path.join(data_dir, sub)
        nwb_files = sorted([os.path.join(sub_dir, f)
                           for f in os.listdir(sub_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append((sub, nwb_file))
    return all_files
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The AI uses pynwb to read NWB files, which is the standard reader for this format. The directory listing approach finds all 174 NWB files across 28 subjects.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from `nwb.subject.subject_id` for each session. Unique subjects are collected in order of first appearance during processing.

ii.
```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
```

```python
subjects = []
for sess in session_results:
    if sess['subject_id'] not in subjects:
        subjects.append(sess['subject_id'])
    subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The AI uses the NWB subject field directly. Subjects are collected in order of first appearance rather than sorted.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is taken from `nwb.identifier`. Sessions are processed in sorted file order within subject directories.

ii.
```python
session_id = nwb.identifier
```

iii. One file per session is the standard layout for this DANDI dataset.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`). The number of trials is verified against the number of go cue events.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
...
go_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_times) == n_trials, f"Go times ({len(go_times)}) != trials ({n_trials})"
```

iii. The trials table defines each behavioral trial unambiguously.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple trial filters: (1) excludes `auto_water` and `free_water` trials, (2) excludes trials where tone onset could not be found, (3) excludes trials beyond the neural recording period (using max_recording_time from obs_intervals of a representative unit), and (4) applies **session-level behavioral selection criteria** from the paper: >65% correct rate and >=50 correct left/right trials on "regular" trials (excluding early lick, stim, auto/free water, and ignore). Sessions failing these criteria are dropped entirely.

ii.
```python
MIN_CORRECT_RATE = 0.65
MIN_CORRECT_LEFT = 50
MIN_CORRECT_RIGHT = 50

behav_valid = (auto_water == 0) & (free_water == 0)
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)

for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False

if correct_rate < MIN_CORRECT_RATE:
    print(f'  SKIP: correct rate {correct_rate:.2f} < {MIN_CORRECT_RATE}')
    io.close()
    return None
if correct_left < MIN_CORRECT_LEFT or correct_right < MIN_CORRECT_RIGHT:
    ...
    return None
```

iii. The AI applies session selection criteria from the paper (>65% correct, >=50 correct L/R), reasoning that the paper used these criteria. The AI does NOT use `obs_intervals` per-trial matching -- instead it only uses the max recording time from a representative unit. This approach misses the proper per-trial obs_intervals filtering. Additionally, the AI filters out `auto_water` trials, which the reference does not (the reference only filters `free_water`).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']`, accessed per unit by indexing into the spike_times column for each good unit index.

ii.
```python
spike_times_all = units['spike_times']
spike_times_good = [spike_times_all[idx] for idx in good_indices]
```

iii. Spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. For each trial, spike times within the window [go_time + T_START, go_time + T_END] are binned into 50ms bins using floor division. Spike counts per bin are converted to firing rates by dividing by bin width.

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_time, t_start, t_end, bin_width, n_bins):
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
    bin_edges_end = bin_edges_start + bin_width
    for i, spk in enumerate(spike_times_list):
        mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
        spk_window = spk[mask]
        bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(fr[i], bin_idx, 1)
    fr /= bin_width
    return fr
```

iii. Firing rates are computed as spike counts per 50ms bin divided by bin width (0.05s), giving rates in Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by two criteria: (1) `classification == 'good'` and (2) `anno_name` must not be empty or None. Sessions with zero good units are dropped.

ii.
```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]

if len(good_indices_units) == 0:
    print(f'  SKIP: no good neurons')
    io.close()
    return None
```

iii. The AI filters by `classification == 'good'` (the QC classifier verdict) AND additionally requires non-empty `anno_name`. The anno_name filter removes units that lack brain region annotations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. For each trial, the go cue time is used to define the window [go_time + T_START, go_time + T_END], and spikes within that window are binned relative to the go cue.

ii.
```python
go_time = go_times[trial_idx]
fr = compute_firing_rates_vectorized(
    spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
)
```

iii. Spike times and go cue times are on the same absolute clock in the NWB file, so alignment just requires computing the window around each trial's go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50ms (0.05s), producing 80 bins over the [-2.5, 1.5] second window. No rebinning is applied -- spikes are binned directly at the target resolution.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins (decoder task spec)
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

iii. The 50ms bin width and [-2.5, 1.5] window are specified by the decoder task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset times) in `BehavioralEvents`, and the go cue time for each trial. For each trial, the last `sample_start` within the trial window [trial_start, go_time] is used as the tone onset.

ii.
```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]

tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]
```

iii. The AI finds the last sample onset within each trial's window, accounting for early-lick replays that can produce multiple sample events per trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center, time from tone onset is computed as `bin_center - tone_relative`, where `tone_relative = tone_time - go_time` (the tone onset relative to the go cue).

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2

tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time  # negative value
time_from_tone = bin_centers - tone_relative  # time since tone onset
```

iii. This is mathematically equivalent to `bin_center_abs - tone_time`, giving seconds since tone onset at each bin center.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for neural data binning are used for the time-from-tone computation, so they are inherently aligned.

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
```

iii. Both neural and input data share the same time grid, defined by the bin centers relative to the go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `trial_starts` (trial start times) used to convert to absolute times.

ii.
```python
photostim_onset = trials['photostim_onset'][:]
photostim_duration = trials['photostim_duration'][:]
```

iii. Photostim onset is stored as a string relative to trial start, with `'N/A'` for non-stimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The onset is converted from trial-relative to go-cue-relative coordinates. A bin is set to 1 if its center falls within the [onset, onset+duration) window, otherwise 0. This is done via a per-bin loop.

ii.
```python
if photostim_onset[trial_idx] != 'N/A':
    ps_onset = float(photostim_onset[trial_idx])
    ps_duration = float(photostim_duration[trial_idx])
    ps_onset_abs = trial_starts[trial_idx] + ps_onset
    ps_end_abs = ps_onset_abs + ps_duration
    ps_onset_rel = ps_onset_abs - go_time
    ps_end_rel = ps_end_abs - go_time

    for b in range(N_BINS):
        bc = bin_centers[b]
        if ps_onset_rel <= bc < ps_end_rel:
            photostim[b] = 1.0
```

iii. The per-bin loop is functionally correct but less efficient than a vectorized comparison.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onset/offset are converted to go-cue-relative coordinates and compared against the same bin centers used for neural data.

ii.
```python
ps_onset_rel = ps_onset_abs - go_time
ps_end_rel = ps_end_abs - go_time
```

iii. Using the same go-cue-relative time axis ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore). Hit means the animal licked the instructed side. Miss means the animal licked the opposite side. For ignore trials, the AI assigns the instruction direction as the "choice."

ii.
```python
instr = instructions[trial_idx]
outcome = outcomes[trial_idx]
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1  # assign instruction direction
```

iii. The AI derives choice from instruction and outcome, which is correct for hit and miss trials. However, for ignore trials (no lick), the AI assigns the instruction direction rather than a "no lick" category.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left) or 1 (right) -- only 2 categories. There is no "no lick" category. The value is repeated across all 80 time bins.

ii.
```python
'output_values': [
    ['left', 'right'],           # choice
    ...
]
```

```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
])
```

iii. The instructions specify "Lick direction choice (left, right, no lick, per-trial)" with 3 categories. The AI only uses 2 categories and assigns ignore trials to the instruction direction, which does not match the specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which contains 'ignore', 'miss', and 'hit'.

ii.
```python
outcomes = trials['outcome'][:]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
```

iii. The trials table stores outcome explicitly with the required three categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is mapped to integers: ignore=0, miss=1, hit=2. The value is repeated across all 80 time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
output_data = np.array([
    ...
    np.full(N_BINS, outcome_val, dtype=np.int64),
    ...
])
```

iii. Straightforward categorical encoding matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'early' or 'no early'.

ii.
```python
early_licks = trials['early_lick'][:]
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

iii. The trials table stores early lick status directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early) or 1 (early). Repeated across all 80 time bins.

ii.
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
output_data = np.array([
    ...
    np.full(N_BINS, early_val, dtype=np.int64),
    ...
])
```

iii. Binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `BehavioralTimeSeries`, which contains (x, y, likelihood) per frame with timestamps.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
if has_tongue:
    tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]
    tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
    tongue_y = tongue_data[:, 1]
    tongue_likelihood = tongue_data[:, 2]
```

iii. This is the only tongue measurement available in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes session-wide percentiles (40th and 60th) of tongue y values where `likelihood > 0.5`. Per trial, for each bin, the nearest frame to the bin center is found via `searchsorted`, and the y-value at that frame is discretized into 3 classes (low/mid/high) regardless of whether the tongue is visible. There is no "not visible" class.

ii.
```python
tongue_visible = tongue_likelihood > 0.5
if np.sum(tongue_visible) > 100:
    visible_y = tongue_y[tongue_visible]
    p40 = np.percentile(visible_y, 40)
    p60 = np.percentile(visible_y, 60)

for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
    if ty < p40:
        tongue_y_trial[b] = 0
    elif ty < p60:
        tongue_y_trial[b] = 1
    else:
        tongue_y_trial[b] = 2
```

iii. The AI computes percentiles only over visible frames (likelihood > 0.5), but during per-trial discretization, it uses the nearest frame's y-value regardless of likelihood. This means low-likelihood frames (when the tongue is retracted) are still assigned to a class rather than being marked "not visible."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories based on per-session 40th and 60th percentile thresholds of visible tongue y values: 0 (below 40th), 1 (40th to 60th), 2 (above 60th). There is no "not visible" (class 3) category.

ii.
```python
'output_values': [
    ...
    ['low', 'mid', 'high'],      # tongue_y
]
```

iii. The instructions specify 4 categories including "3: not visible". The AI only implements 3 categories. Additionally, the percentiles are computed over raw visible frames rather than over 50ms bin means as the reference does.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center, the nearest camera frame is found using `searchsorted` on the camera timestamps. The y-value of that nearest frame is used directly (nearest-neighbor interpolation).

ii.
```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
```

iii. This nearest-neighbor approach differs from the reference, which averages all frames falling within each 50ms bin (bin-mean approach). The nearest-neighbor approach uses a single frame per bin rather than averaging ~15 frames (at 300 Hz) per 50ms bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Sessions with no good neurons are skipped. (2) Trials where no tone onset is found (NaN) are excluded. (3) Trials beyond the neural recording period are excluded using max_recording_time. (4) Sessions failing behavioral criteria are skipped. (5) Errors during processing are caught with try/except and the session is skipped. (6) If tongue tracking is absent, tongue_y defaults to 1 (middle class).

ii.
```python
valid_mask &= ~np.isnan(tone_onset_per_trial)

for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

```python
try:
    result = process_session(nwb_path, ...)
except Exception as e:
    print(f'  ERROR: {e}')
    n_skipped += 1
    continue
```

iii. The AI handles missing data by skipping sessions/trials rather than by using explicit NaN or missing-data categories. The recording coverage check uses max_recording_time + 1.0s buffer rather than per-trial obs_intervals matching, which is a coarser approximation.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the per-trial firing rate computation, which loops over every trial and every neuron. The AI code processes each trial independently, calling `compute_firing_rates_vectorized` once per trial (which itself loops over neurons). NWB file I/O is also significant.

ii.
```python
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

iii. The per-trial outer loop combined with the per-neuron inner loop makes this O(n_trials * n_neurons), compared to the reference which vectorizes the trial dimension. The AI's CONVERSION_NOTES estimates ~5-10s per session, with full conversion taking ~30 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be improved: (1) The per-trial loop for firing rates -- the reference processes all trials at once per neuron using a single flattened edge array and `searchsorted`. (2) The per-bin loop for photostim could be replaced with vectorized comparison. (3) The per-trial tongue discretization loop with its per-bin inner loop.

ii.
```python
# Per-trial loop (could process all trials at once)
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(spike_times_good, go_time, ...)

# Per-bin photostim loop (could be vectorized)
for b in range(N_BINS):
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. The reference code vectorizes the trial dimension by computing all trial bin edges at once and using a single searchsorted per neuron, which is substantially faster.

## 10-c. What processing does the code repeat multiple times?

i. The spike_times for good units are loaded once per session (via list comprehension), but the firing rate computation repeats the window masking and binning for each trial independently, rather than processing all trials at once.

ii.
```python
spike_times_good = [spike_times_all[idx] for idx in good_indices]

for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(spike_times_good, go_time, ...)
```

iii. Each call to `compute_firing_rates_vectorized` re-masks spikes to the trial window and re-bins them, when all trials could be processed together.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI applies session selection criteria (correct rate >65%, >=50 correct L/R trials) that are not required by the decoder task and cause 30 sessions to be unnecessarily dropped. It also computes the session-level `correct_rate` and stores it per session result but it is not part of the output format. The large `REGION_MAPPING` dictionary maps detailed annotations to broad categories, which is extra processing not needed if the raw annotation names are used directly (as the reference does).

ii.
```python
MIN_CORRECT_RATE = 0.65
MIN_CORRECT_LEFT = 50
MIN_CORRECT_RIGHT = 50

if correct_rate < MIN_CORRECT_RATE:
    return None
if correct_left < MIN_CORRECT_LEFT or correct_right < MIN_CORRECT_RIGHT:
    return None
```

iii. The session selection criteria come from the paper's analysis methods but are not part of the decoder task requirements. They cause a significant reduction in usable data (144 vs 173 sessions).
