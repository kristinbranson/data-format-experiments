# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads data by scanning `/app/data` for subject directories whose names start with `sub-`, then scanning each subject directory for `.nwb` files. Each NWB file is treated as one session and processed in a top-level loop in `main()`.

ii. ```python
def get_nwb_files(data_dir):
    """Get list of all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for fname in files:
            nwb_files.append((subj, os.path.join(subj_dir, fname)))
    return nwb_files

# Get all NWB files
nwb_files = get_nwb_files(DATA_DIR)

for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset is “NWB files organized by subject” and explicitly chooses to include all 173 sessions with good units after skipping the one NWB file with zero good units.

## 1-b. How are the data split into subjects?

i. Subjects are split by the directory structure under `/app/data`: each `sub-XXXXXX` directory is a subject, and the subject ID string is carried through into `subjects` and `subject_idx`.

ii. ```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])

for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for fname in files:
        nwb_files.append((subj, os.path.join(subj_dir, fname)))

if subject_id not in subjects_seen:
    subjects_seen[subject_id] = len(subjects_seen)
    subjects_list.append(subject_id)
subj_idx = subjects_seen[subject_id]
```

iii. The notes say “NWB files organized by subject (sub-XXXXXX directories)” and list 28 subjects. No other subject-splitting logic appears in the script.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. `process_session()` reads one NWB file and returns one session dictionary; the top-level `data['neural']`, `data['input']`, and `data['output']` are lists over these processed sessions.

ii. ```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)
    if result is None:
        print("SKIPPED (no good units or trials)")
        continue

    all_sessions.append(result)

data = {
    'neural': [s['neural'] for s in all_sessions],
    'input': [s['input'] for s in all_sessions],
    'output': [s['output'] for s in all_sessions],
```

iii. `CONVERSION_NOTES.md` says “Each NWB file = one session” and the full run output shows 174 files found, 173 sessions retained.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `trials` table. After building a `trial_mask`, the script uses the surviving row indices as trial IDs and processes each index independently, using the same index into `trials` columns and `go_start_times`.

ii. ```python
trials = nwb.trials
outcomes = trials['outcome'][:]
instructions = trials['trial_instruction'][:]
early_lick = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]

trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]

for ti in trial_indices:
    go_cue = go_start_times[ti]
```

iii. The notes describe the NWB `trials` table as the source of per-trial variables and say trial filtering is applied before processing.

## 1-e. How are trials filtered based on quality controls?

i. The script excludes only `auto_water` and `free_water` trials, then further restricts to trials considered covered by all good units according to `obs_intervals`. It deliberately keeps early-lick, ignore, and photostim trials because those variables are required as decoder outputs or inputs.

ii. ```python
def get_valid_trial_indices(nwb, good_unit_indices, n_trials):
    trial_starts = nwb.trials['start_time'][:]
    obs_sets = {}
    for idx in good_unit_indices:
        obs = nwb.units['obs_intervals'][idx]
        n_obs = len(obs)
        if n_obs not in obs_sets:
            obs_sets[n_obs] = obs

    valid_trials = np.ones(n_trials, dtype=bool)
    for n_obs, obs in obs_sets.items():
        obs_starts = obs[:, 0]
        diffs = np.abs(trial_starts[:, None] - obs_starts[None, :])
        min_diffs = diffs.min(axis=1)
        covered = min_diffs < 1.0
        valid_trials &= covered
    return np.where(valid_trials)[0]

trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
```

iii. The agent justifies this directly in `CONVERSION_NOTES.md`: reference code excludes early lick, ignore, and stimulation in its regular-trial mask, but the agent chose to keep them because they are required by the decoder specification. It also notes a later fix: “Added `obs_intervals` filtering to exclude trials without valid neural recording.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `nwb.units['spike_times']` after filtering units by `nwb.units['classification'] == 'good'`. Region labels are derived separately from `nwb.units['anno_name']`.

ii. ```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])

anno_names = nwb.units['anno_name'][:][good_mask]

good_unit_indices = np.where(good_mask)[0]
good_unit_indices = good_unit_indices[valid_neuron_mask]

spike_times_per_unit = []
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

iii. The notes say “In NWB: `classification == 'good'` corresponds to QC classifier pass” and “Spike times are ABSOLUTE.”

## 2-b. How is the `neural` data processed?

i. For each kept trial, the script takes each good unit’s absolute spike times, crops them to a `[-2.5, 1.5)` second window around the trial’s go cue, bins the spikes into non-overlapping 50 ms bins, and divides by bin width to convert counts to firing rates in Hz.

ii. ```python
BIN_SIZE_S = 0.05
T_START = -2.5
T_END = 1.5

def compute_firing_rates(spike_times_list, go_cue_time, t_start, t_end, bin_size):
    n_neurons = len(spike_times_list)
    n_bins = int((t_end - t_start) / bin_size)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)

    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end

    for i, st in enumerate(spike_times_list):
        mask = (st >= abs_start) & (st < abs_end)
        spikes_in_window = st[mask]
        bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
        np.add.at(fr[i], bin_indices, 1.0)

    fr /= bin_size
    return fr
```

iii. The agent’s notes explicitly say it is following the task override: reference preprocessing uses 40 ms width and 3.4 ms stride, but this conversion intentionally uses 50 ms non-overlapping bins because the decoder task requires that.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by the NWB unit-level quality label `classification == 'good'`. After that, units whose `anno_name` cannot be mapped into the hard-coded 14 coarse regions are dropped. No firing-rate threshold is applied.

ii. ```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])

anno_names = nwb.units['anno_name'][:][good_mask]

region_indices = []
valid_neuron_mask = np.ones(n_good, dtype=bool)
for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    if region is None:
        valid_neuron_mask[i] = False
        region_indices.append(-1)
    else:
        region_indices.append(COARSE_REGIONS.index(region))

good_unit_indices = np.where(good_mask)[0]
good_unit_indices = good_unit_indices[valid_neuron_mask]
```

iii. `CONVERSION_NOTES.md` says the reference QC mode is the classifier-based “good units” list and that the 2 Hz firing-rate filter in the method paper was analysis-specific and intentionally not applied here.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to that trial’s go cue onset from `BehavioralEvents/go_start_times`. The neural window is `go_cue + [-2.5, 1.5]`.

ii. ```python
events = nwb.acquisition['BehavioralEvents']
go_start_times = events.time_series['go_start_times'].timestamps[:]

for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The notes repeatedly say the target alignment event is go cue onset and that NWB spike times are absolute, so alignment must be performed during conversion.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins and 80 total bins over the `[-2.5, 1.5]` s window. No additional rebinning is done beyond this direct binning step.

ii. ```python
BIN_SIZE_S = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80
```

iii. The notes say the reference code uses 40 ms width / 3.4 ms stride but the decoder task requires 50 ms bins, so the agent explicitly chose the task-specified binning.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and `BehavioralEvents/go_start_times`.

ii. ```python
go_start_times = events.time_series['go_start_times'].timestamps[:]
sample_start_times = events.time_series['sample_start_times'].timestamps[:]

tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
time_from_tone = compute_time_from_tone(go_cue, tone_onset, T_START, T_END, BIN_SIZE_S)
```

iii. The notes say “Tone onset: Last `sample_start_time` before each trial’s go cue (accounting for early lick replays).”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the script finds the last sample-start event before the trial’s go cue, treats that as tone onset, converts it to a time relative to the go cue, then subtracts that offset from every bin center. If no sample-start event exists, it falls back to `go_cue - 1.85`.

ii. ```python
def find_last_sample_before_go(sample_start_times, go_cue_time):
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]

def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    tone_rel = tone_onset_time - go_cue_time
    time_from_tone = bin_centers - tone_rel
    return time_from_tone.astype(np.float32)

tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
if tone_onset is None:
    tone_onset = go_cue - 1.85
```

iii. The justification in the notes is that `sample_start_times` has more entries than trials because early licks cause replays, so the last sample event before go cue is the best proxy for the effective tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled at the same 80 bin centers used for the neural data, with bin centers defined relative to the same go cue.

ii. ```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
tone_rel = tone_onset_time - go_cue_time
time_from_tone = bin_centers - tone_rel

input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The notes describe this input as time-varying and aligned to go cue in the same `[-2.5, 1.5]` window as neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, using trial-relative overlap with the analysis window.

ii. ```python
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]

trial_abs_start = go_cue + T_START
trial_abs_end = go_cue + T_END
stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
ps_starts = photostim_start_abs[stim_mask]
ps_stops = photostim_stop_abs[stim_mask]
```

iii. The notes say the trial table stores `photostim_onset/power/duration`, but the agent chose the absolute event timestamps for actual temporal alignment.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, all photostim intervals overlapping the trial window are collected and turned into a binary time series. Any bin whose center lies between a photostim start and stop time is set to 1, otherwise 0.

ii. ```python
def compute_photostim_timeseries(go_cue_time, photostim_starts, photostim_stops,
                                  t_start, t_end, bin_size):
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    abs_centers = bin_centers + go_cue_time

    stim = np.zeros(n_bins, dtype=np.float32)
    for start, stop in zip(photostim_starts, photostim_stops):
        mask = (abs_centers >= start) & (abs_centers <= stop)
        stim[mask] = 1.0
    return stim
```

iii. The notes say “Photostim input: Binary time series - 1 during photostim, 0 otherwise.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned to the same go cue and sampled at the same bin centers as the neural data.

ii. ```python
photostim_ts = compute_photostim_timeseries(go_cue, ps_starts, ps_stops,
                                            T_START, T_END, BIN_SIZE_S)
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The notes explicitly say the photostim time series is “aligned to go cue.”

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The script does not derive choice from actual lick behavior. It derives “choice” from the per-trial instruction field `trials['trial_instruction']`.

ii. ```python
instructions = trials['trial_instruction'][:]

# Output 0: choice (left=0, right=1)
choice = 0 if instructions[ti] == 'left' else 1
```

iii. The notes say “choice: left=0, right=1” and map it from `trial_instruction`. No justification beyond convenience is given, and no lick-time-based choice extraction appears in the code.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The processing is just a binary remapping of `trial_instruction`: `left -> 0`, everything else -> `1`.

ii. ```python
# Output 0: choice (left=0, right=1)
choice = 0 if instructions[ti] == 'left' else 1
```

iii. The agent treated trial instruction as synonymous with lick choice. The reference-code notes, however, distinguish `trial_type` from `lick_directions`, and the script contains no logic to inspect actual licks.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `trials['outcome']`.

ii. ```python
outcomes = trials['outcome'][:]
outcome_str = outcomes[ti]
```

iii. The notes describe the NWB `trials` table as containing `outcome (hit/miss/ignore)`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. It remaps the outcome strings to integer categories `ignore=0`, `miss=1`, `hit=2`, with unknown strings defaulting to `0`.

ii. ```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
```

iii. This mapping is stated directly in both the task instructions and the agent’s notes.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `trials['early_lick']`.

ii. ```python
early_lick = trials['early_lick'][:]
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. The notes identify `early_lick` as a trial-table field and say it is kept because the decoder must predict it.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. It maps `'no early' -> 0` and any other label to `1`.

ii. ```python
# Output 2: early lick (no=0, yes=1)
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. The agent’s notes say “Output 2: early lick (no=0, yes=1).”

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 for `tongue_y`, column 2 for `tongue_likelihood`, plus the associated timestamps.

ii. ```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. The notes state that `Camera0_side_TongueTracking` has `(x, y, likelihood)` columns and explicitly choose column 1 as tongue y-position.

## 8-b. How is `output` *Tongue y-position* processed?

i. For each trial, the script takes tongue samples in the `[-2.5, 1.5]` s window around go cue, discards low-confidence frames by setting `likelihood < 0.9` to `NaN`, and averages the remaining y-values within each 50 ms bin.

ii. ```python
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
if mask.sum() == 0:
    return np.full(n_bins, np.nan, dtype=np.float32)

t_in_window = tongue_timestamps[mask]
y_in_window = tongue_data[mask]
like_in_window = tongue_likelihood[mask]

y_in_window = y_in_window.copy()
y_in_window[like_in_window < 0.9] = np.nan

bin_indices = np.digitize(t_in_window, bin_edges) - 1
valid_mask = ~np.isnan(y_in_window)
if valid_mask.any():
    sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
    counts = np.bincount(valid_bins, minlength=n_bins)
    tongue_y_binned[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
```

iii. The notes justify the column choice and binning, but there is no code implementing the 5-sigma outlier correction the notes mention from the reference methods.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The script pools all valid tongue-y values across kept trials within a session, computes the 40th and 60th percentiles, and discretizes each bin to `0`, `1`, or `2` according to those thresholds.

ii. ```python
if len(tongue_y_session) > 0:
    tongue_y_arr = np.array(tongue_y_session)
    p40 = np.percentile(tongue_y_arr, 40)
    p60 = np.percentile(tongue_y_arr, 60)

tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

iii. The notes say this follows the decoder specification exactly: per-session 40th/60th percentile thresholds over all valid time points.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue data are windowed using the same `go_cue + [-2.5, 1.5]` interval as the neural data and then binned into the same 80 bins.

ii. ```python
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
```

iii. The notes explicitly describe tongue y as a time-varying output aligned to go cue.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing or awkward data with simple fallbacks:
   - if no tone onset is found, it uses `go_cue - 1.85`;
   - if no tongue data exist in the window, it returns all-`NaN`;
   - low-likelihood tongue frames are set to `NaN`;
   - if a unit’s region annotation cannot be mapped, the unit is dropped;
   - if a session has zero retained units or fewer than two kept trials, the session is skipped;
   - after discretization, `NaN` tongue bins are effectively converted to class `0` because the discretized array is initialized with zeros.

ii. ```python
if tone_onset is None:
    tone_onset = go_cue - 1.85

if mask.sum() == 0:
    return np.full(n_bins, np.nan, dtype=np.float32)

y_in_window[like_in_window < 0.9] = np.nan

if region is None:
    valid_neuron_mask[i] = False

if len(trial_indices) < 2:
    io.close()
    return None

tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
```

iii. The notes mention the 1.85 s tone fallback and the later `obs_intervals` fix. No explicit justification is given for converting missing tongue bins into the lowest class.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is the nested session -> trial -> neuron spike-binning loop in `compute_firing_rates()`, because every trial rescans every retained unit’s spike train. Secondary costs are full-session NWB I/O and tongue-tracking windowing/binning.

ii. ```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)

for ti in trial_indices:
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)

for i, st in enumerate(spike_times_list):
    mask = (st >= abs_start) & (st < abs_end)
    spikes_in_window = st[mask]
    bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
    np.add.at(fr[i], bin_indices, 1.0)
```

iii. The full conversion output reports about 2300.8 seconds total for 173 sessions, and the notes estimate roughly 6.5 seconds per session in sample mode. That runtime is consistent with the spike-binning loops being the main bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are:
   - the per-neuron loop in `compute_firing_rates()`;
   - the per-trial loop that recomputes neural, input, and output features independently;
   - the per-unit region-mapping loop over `anno_name`;
   - the repeated `find_last_sample_before_go()` scan over all sample starts for every trial.

ii. ```python
for i, st in enumerate(spike_times_list):
    ...

for ti in trial_indices:
    ...

for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    ...

valid = sample_start_times[sample_start_times < go_cue_time]
```

iii. The script is straightforward but largely scalar at the session/trial/neuron level. The notes themselves call out only one vectorized improvement for tongue binning; the more expensive spike and event logic remain loop-heavy.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes the same bin centers or edges in multiple helper functions, repeatedly rescans `sample_start_times` to find the last pre-go sample, and repeatedly filters absolute photostim events against each trial window.

ii. ```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time

tone_onset = find_last_sample_before_go(sample_start_times, go_cue)

stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
```

iii. This repetition follows from the helper-per-feature structure in `convert_data.py`; the notes do not claim any caching or precomputation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does several things that are not necessary for downstream decoding:
   - it replicates static per-trial labels (`choice`, `outcome`, `early_lick`) across all 80 time bins;
   - it loads the full tongue tracking matrix and then discards tongue `x`;
   - it stores raw tongue traces in `output_trials_raw` only to overwrite them with discretized labels later.

ii. ```python
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)

output_trials_raw.append({
    'choice': choice,
    'outcome': outcome_val,
    'early_lick': early_val,
    'tongue_y_raw': tongue_y_trial
})

full_output = np.zeros((4, N_TIMEBINS), dtype=np.int64)
full_output[0, :] = out_dict['choice']
full_output[1, :] = out_dict['outcome']
full_output[2, :] = out_dict['early_lick']
```

iii. These choices come from the target-format convenience and the script’s two-stage tongue processing. The notes do not describe a downstream need for the repeated static labels or full raw tongue matrix.
