# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (Neurodata Without Borders) files stored in `/app/data/`. It iterates over subject directories (`sub-XXXXXX/`), listing all `.nwb` files within each. Each NWB file corresponds to one behavioral session. Files are opened using `pynwb.NWBHDF5IO` and read sequentially in a loop. The original reference code works with `.mat` files exported from DataJoint, not NWB files directly.

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

iii. The AI noted in CONVERSION_NOTES.md that "Reference code works with .mat files from DataJoint export, NOT NWB directly" and that the data needed to be mapped from NWB fields to equivalent .mat fields. The NWB files are from the DANDI archive (000363).

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB file's `subject.subject_id` field. Each NWB file is associated with one subject directory. The AI collects unique subject IDs and builds a `subjects` list and `subject_idx` array mapping each session to its subject.

ii.
```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
```

```python
# In build_dataset():
for sess in session_results:
    if sess['subject_id'] not in subjects:
        subjects.append(sess['subject_id'])
    subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The AI documented 28 subjects in the dataset, matching the paper's stated 28 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file represents one session. Sessions are processed one at a time in a loop. Sessions that do not meet selection criteria are skipped. The final dataset contains 144 sessions (out of 174 NWB files).

ii.
```python
for i, (subject, nwb_path) in enumerate(all_files):
    result = process_session(nwb_path, show_processing=args.show_processing, session_idx=i)
    if result is None:
        n_skipped += 1
        continue
    session_results.append(result)
```

iii. The AI noted a discrepancy: the paper reports 173 sessions but 174 NWB files exist. After applying session selection criteria, only 144 sessions pass. The AI attributed the difference to strict application of behavioral criteria.

## 1-d. How are the data split into trials?

i. Trials are extracted from the NWB `trials` table. Each trial has start_time, stop_time, instruction, outcome, early_lick, auto_water, free_water, and photostim fields. Go cue times are taken from `BehavioralEvents.go_start_times`. Valid trials are iterated over and neural/input/output data are computed per trial.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
instructions = trials['trial_instruction'][:]
outcomes = trials['outcome'][:]
early_licks = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
```

iii. Trial information is directly available in the NWB trial table.

## 1-e. How are trials filtered based on quality controls?

i. Two levels of filtering are applied:
1. **Session selection**: Sessions must have >65% correct rate and ≥50 correct left + ≥50 correct right trials. Correct rate is computed on "regular" trials (excluding early lick, photostim, auto/free water, and ignore trials).
2. **Trial filtering**: Auto-water and free-water trials are excluded. Trials with missing tone onset (`NaN`) or beyond recording coverage are also excluded. Early lick, photostim, and ignore trials are KEPT (needed for decoder outputs/inputs).

ii.
```python
# Session selection:
behav_valid = (auto_water == 0) & (free_water == 0)
regular_mask = behav_valid.copy()
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
    if early_licks[i] == 'early':
        regular_mask[i] = False
regular_mask &= (outcomes != 'ignore')
correct_rate = correct_regular / n_regular

# Trial filtering:
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

iii. The AI documented: "Decoder task requires photostim as input and early_lick/outcome(ignore) as outputs, so we keep those trials. Only exclude auto_water and free_water." The reference code's `get_regular_trial_mask()` excludes early lick, stim, auto/free water, and no-response trials, but the AI intentionally keeps more trials for the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units.spike_times` in the NWB file, filtered to units where `classification == 'good'` and `anno_name` is not empty.

ii.
```python
units = nwb.units
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]
spike_times_all = units['spike_times']
spike_times_good = [spike_times_all[idx] for idx in good_indices]
```

iii. The AI noted that `classification == 'good'` in NWB is equivalent to passing the QC classifier used in the reference code.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins to compute firing rates (Hz). For each trial, spikes within the window [go_cue - 2.5s, go_cue + 1.5s] are assigned to bins. Spike counts per bin are divided by bin width to get firing rates.

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_time, t_start, t_end, bin_width, n_bins):
    n_neurons = len(spike_times_list)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
    bin_edges_end = bin_edges_start + bin_width
    for i, spk in enumerate(spike_times_list):
        if len(spk) == 0:
            continue
        mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
        spk_window = spk[mask]
        if len(spk_window) == 0:
            continue
        bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(fr[i], bin_idx, 1)
    fr /= bin_width
    return fr
```

iii. The AI chose 50ms bins as specified in the decoder task instructions (different from the reference code's 40ms bins with 3.4ms stride). This is a deliberate difference per the instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by two criteria: `classification == 'good'` (QC classifier output) and non-empty `anno_name` (must have histological annotation). Sessions with zero good neurons are skipped.

ii.
```python
good_mask_units = classification == 'good'
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]
if len(good_indices_units) == 0:
    print(f'  SKIP: no good neurons')
    io.close()
    return None
```

iii. The AI documented: "QC classifier: classification == 'good' in NWB" and "Must have both ephys and histology (CCF coordinates), anno_name must not be empty."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the Go cue onset. For each trial, the go cue time is obtained from `BehavioralEvents.go_start_times`. Spikes are binned in a window of [-2.5, +1.5] seconds relative to the go cue.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
# In compute_firing_rates_vectorized:
bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width  # t_start = -2.5
```

iii. The instructions specify "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms bins, producing 80 time bins per trial over the [-2.5, 1.5]s window. No rebinning is applied — spike times are directly binned at 50ms. The reference code uses 40ms bins with 3.4ms stride, but the decoder instructions specify 50ms.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins (decoder task spec)
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

iii. The AI noted: "Decoder task specifies 50ms bins, different from reference code's 40ms bins. This is allowed per instructions."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents.sample_start_times` (tone onset timestamps) and `BehavioralEvents.go_start_times` (go cue timestamps).

ii.
```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]
# Map sample_start to each trial:
tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]
```

iii. The AI takes the LAST sample onset before the go cue within each trial's window, accounting for potential replays due to early licking.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the tone onset time is found (last `sample_start` event within the trial). This is converted to go-cue-relative coordinates. Then for each time bin, the time from tone onset is computed as: `bin_center - tone_onset_relative_to_go_cue`.

ii.
```python
tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time  # tone onset relative to go cue (negative)
time_from_tone = bin_centers - tone_relative  # time since tone onset at each bin
```

iii. The AI documented: "Continuous variable = current_time - first_sample_start_in_trial (in go-cue-relative coords)."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time from tone onset is computed at the same bin centers as the neural data (50ms bins from -2.5 to +1.5s relative to go cue), ensuring perfect temporal alignment.

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
time_from_tone = bin_centers - tone_relative
```

iii. Both neural and input use the same bin_centers array.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from the trials table fields `photostim_onset` and `photostim_duration`, plus `trial_starts` (trial start times) for absolute time conversion.

ii.
```python
photostim_onset = trials['photostim_onset'][:]
photostim_duration = trials['photostim_duration'][:]
```

iii. The AI noted these values are stored as strings in the NWB file ('N/A' when no photostim).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series (0/1) is created for each trial. If `photostim_onset != 'N/A'`, the onset is interpreted as relative to trial start and converted to absolute time. Then for each bin, if the bin center falls within [ps_onset, ps_onset + ps_duration], photostim is set to 1.

ii.
```python
photostim = np.zeros(N_BINS, dtype=np.float32)
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

iii. The AI interprets `photostim_onset` as relative to trial start time. This is a key assumption.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is evaluated at the same bin centers as the neural data, ensuring alignment.

ii.
```python
for b in range(N_BINS):
    bc = bin_centers[b]  # same bin_centers used for neural data
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. Same bin_centers array as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table.

ii.
```python
instr = instructions[trial_idx]
outcome = outcomes[trial_idx]
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. The AI documented: "hit: choice=instruction; miss: choice=opposite; ignore: choice=instruction."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For hit trials, choice matches the instruction direction. For miss trials, choice is the opposite of instruction (wrong lick). For ignore trials (no lick), choice is assigned the instruction direction. Left=0, right=1. The output is per-trial (broadcast to all time bins).

ii.
```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
], dtype=np.int64)
```

iii. The AI chose to assign instruction direction for ignore trials since there's no actual lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived directly from the `outcome` field in the trials table.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
```

iii. Direct mapping from the NWB trial outcome values.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct categorical mapping: ignore=0, miss=1, hit=2. Per-trial value broadcast to all time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
output_data = np.array([
    ...
    np.full(N_BINS, outcome_val, dtype=np.int64),
    ...
])
```

iii. Follows the decoder task specification exactly.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This question references "Distance to reward zone" but the AI's dataset does not include this variable. The AI implemented *Outcome* as a per-trial categorical variable broadcast to all time bins, so it is trivially aligned.

ii. N/A — the output is per-trial, not time-varying.

iii. The decoder task specification lists Outcome as per-trial, not as a spatially-varying distance metric.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from the `early_lick` field in the trials table.

ii.
```python
early_licks = trials['early_lick'][:]
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

iii. Direct mapping from NWB field values.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: 'no early' → 0, 'early' → 1. Per-trial value broadcast to all time bins.

ii.
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
output_data = np.array([
    ...
    np.full(N_BINS, early_val, dtype=np.int64),
    ...
])
```

iii. Matches decoder task specification: no=0, yes=1.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `BehavioralTimeSeries.Camera0_side_TongueTracking` which contains (x, y, likelihood) per frame at ~294 Hz.

ii.
```python
tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]  # (n_frames, 3): x, y, likelihood
tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The AI identified Camera0_side_TongueTracking as the source for tongue y-position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per-session percentile thresholds are computed from tongue y-positions where likelihood > 0.5 (tongue visible). The 40th and 60th percentiles are used as thresholds.

ii.
```python
tongue_visible = tongue_likelihood > 0.5
if np.sum(tongue_visible) > 100:
    visible_y = tongue_y[tongue_visible]
    p40 = np.percentile(visible_y, 40)
    p60 = np.percentile(visible_y, 60)
else:
    p40 = np.percentile(tongue_y, 40)
    p60 = np.percentile(tongue_y, 60)
```

iii. The AI uses a likelihood threshold of 0.5 to determine visibility, and falls back to using all data if too few visible frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (< 40th percentile), 1 (40th to 60th percentile), 2 (> 60th percentile). Thresholds computed per session.

ii.
```python
if ty < p40:
    tongue_y_trial[b] = 0
elif ty < p60:
    tongue_y_trial[b] = 1
else:
    tongue_y_trial[b] = 2
```

iii. Matches decoder task specification: "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each time bin, the closest tongue tracking frame is found using `np.searchsorted` on the tongue timestamps. The tongue y value at that frame is then discretized.

ii.
```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
```

iii. Nearest-frame lookup aligns tongue data to the same bin centers as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Trials with no tone onset (NaN) are excluded via `valid_mask &= ~np.isnan(tone_onset_per_trial)`.
- Trials beyond neural recording coverage are excluded (go_time + T_END > max_recording_time + 1.0).
- Neurons with empty `anno_name` are excluded.
- Sessions with zero good neurons are skipped.
- Sessions with fewer than 2 valid trials are skipped.
- For sessions without tongue tracking data, tongue_y defaults to 1 (middle category).
- Unmapped brain region annotations fall back to 'OtherCortex'.
- 126/74894 trials (0.17%) have all-zero neural data, concentrated in session 34.

ii.
```python
valid_mask &= ~np.isnan(tone_onset_per_trial)
if trial_end_abs > max_recording_time + 1.0:
    valid_mask[i] = False
if len(good_indices_units) == 0:
    return None
if not has_tongue:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)  # default to middle
print(f'  WARNING: Unmapped annotation: "{anno_name}"')
return 'OtherCortex'
```

iii. The AI documented zero-neural-data trials as "negligible edge cases from recording coverage boundaries."

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is computing firing rates per trial (`compute_firing_rates_vectorized`), which loops over all neurons and all trials. Loading spike times from NWB files is also slow due to I/O. The AI estimated ~5-10s per session, ~30 min total.

ii.
```python
for i, spk in enumerate(spike_times_good):  # loop over all neurons
    # ... for each trial, bin spikes
```

iii. The AI noted timing estimates in CONVERSION_NOTES.md: "~5-10s per session, ~30 min total for full conversion."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The neuron loop in `compute_firing_rates_vectorized` iterates over each neuron individually.
2. The photostim bin assignment uses a Python for-loop over bins.
3. The tongue y-position uses a Python for-loop over bins with `np.searchsorted` per bin.
4. The trial filtering loop for regular_mask checks photostim and early_lick per trial.

ii.
```python
# Photostim loop (could be vectorized):
for b in range(N_BINS):
    bc = bin_centers[b]
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0

# Tongue loop (could be vectorized):
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
```

iii. The AI named the firing rate function "vectorized" but it still loops over neurons.

## 10-c. What processing does the code repeat multiple times?

i. The `compute_firing_rates_vectorized` function is called separately for each trial, reloading and filtering spike times for all neurons each time. The spike window masking (`spk >= go_time + t_start`) is repeated for every neuron-trial combination. The tongue timestamp lookup is done per-bin per-trial rather than being precomputed.

ii.
```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS)
```

iii. Each call to `compute_firing_rates_vectorized` re-processes all neurons' full spike trains to find the relevant window.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things:
1. The `max_recording_time` is computed from `obs_intervals` of only one representative unit, which may not be representative.
2. The `make_processing_plots` function generates detailed visualizations only used in `--show-processing` mode.
3. Brain region counts and output distributions are printed to stdout but not stored in the output.
4. The entire tongue tracking timeseries is loaded into memory even though only small windows per trial are used.

ii.
```python
# Full tongue data loaded but only bin centers queried:
tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]
tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
```

iii. Loading the full tongue tracking timeseries is memory-inefficient but functionally correct.
