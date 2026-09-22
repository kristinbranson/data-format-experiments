# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by scanning `/app/data` for `sub-*` subject directories, collecting every `.nwb` file, and opening each file with `pynwb.NWBHDF5IO`. Inside each session file it reads `nwb.subject`, `nwb.identifier`, `nwb.trials`, `nwb.units`, `nwb.acquisition['BehavioralEvents']`, and `nwb.acquisition['BehavioralTimeSeries']`.

ii.
```python
def get_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
        for f in nwb_files:
            all_files.append((subj, f))
    return all_files
```

```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    subject_id = nwb.subject.subject_id
    session_id = nwb.identifier
    trials = nwb.trials
    units = nwb.units
```

iii. `CONVERSION_NOTES.md` says the data are NWB files organized as subject directories containing 174 session files, and the trajectory repeatedly treats one NWB file as one session to be processed once.

## 1-b. How are the data split into subjects?

i. Subjects are split first by the `sub-*` directory layout when files are enumerated, and then stored using `nwb.subject.subject_id`. The final `subjects` list is the sorted unique set of subject IDs, and `subject_idx` maps each session to that list.

ii.
```python
subject_id = nwb.subject.subject_id
```

```python
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The notes state there are 28 subject directories and 28 subjects in the dataset, and the trajectory treats `subject_id` from NWB as the canonical animal identifier.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session ID comes from `nwb.identifier`, and the output session order follows the sorted per-subject file list returned by `get_nwb_files`.

ii.
```python
session_id = nwb.identifier
```

```python
for i, (subj, fpath) in enumerate(all_files):
    result = process_session(fpath, show_processing=show, session_idx=i)
```

iii. `CONVERSION_NOTES.md` explicitly says there are 174 NWB files and 173 sessions with good units, so the AI used file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. Trials are taken from `nwb.trials`, and trial-level arrays are indexed by trial row position. Go-cue timestamps are read from `BehavioralEvents/go_start_times`, and valid trials are selected by a boolean mask over the trial rows.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
valid_trial_indices = np.where(valid_trial_mask)[0]
for i, trial_idx in enumerate(valid_trial_indices):
    go_time = go_times[trial_idx]
```

iii. The trajectory says `go_start_times` has one event per trial and that go cue onset is the alignment event, so the AI indexed trial processing by trial-table row and matched rows to `go_times` by shared order.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials only if `auto_water == 0`, `free_water == 0`, and the full `[-2.5 s, +1.5 s]` go-cue-aligned window lies inside a recording range derived from the first good unit’s `obs_intervals`. Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
valid_trial_mask = (auto_water == 0) & (free_water == 0)
for i in range(n_trials):
    if valid_trial_mask[i]:
        window_start = go_times[i] + ALIGN_START
        window_end = go_times[i] + ALIGN_END
        if window_start < rec_min or window_end > rec_max:
            valid_trial_mask[i] = False

valid_trial_indices = np.where(valid_trial_mask)[0]
if n_valid_trials < 2:
    return None
```

iii. The notes say “exclude auto_water and free_water; exclude trials outside recording range,” and the trajectory justifies this by partial recordings and use of `obs_intervals` to find which parts of a behavioral session actually contain neural data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units['spike_times']` for units where `units['classification'] == 'good'`, with `BehavioralEvents/go_start_times` used to define the per-trial bin edges.

ii.
```python
classification = units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
```

```python
good_spike_times = []
for idx in good_indices:
    st = units['spike_times'][idx]
    good_spike_times.append(np.sort(st))
```

iii. The notes say neuron curation uses `classification == "good"` to match classifier-based QC, and the trajectory identifies `spike_times` as the neural source variable.

## 2-b. How is the `neural` data processed?

i. For each valid trial and each good unit, the AI bins spike times into 50 ms bins from `-2.5` to `+1.5` s around the go cue, counts spikes with `np.histogram`, and divides by bin width to get firing rates in Hz.

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_cue_times, n_neurons,
                                     bin_width=BIN_WIDTH, align_start=ALIGN_START,
                                     align_end=ALIGN_END):
    n_trials = len(go_cue_times)
    n_bins = int((align_end - align_start) / bin_width)
    bin_offsets = align_start + np.arange(n_bins + 1) * bin_width
```

```python
for t in range(n_trials):
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    bin_edges = go_cue_times[t] + bin_offsets
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes_window, bins=bin_edges)
        fr[i] = counts / bin_width
```

iii. The notes describe the neural transform as “50 ms bin firing rates, -2.5 to +1.5 s from go cue,” and the trajectory says the code was updated to vectorize firing-rate computation as much as needed to stay under the runtime budget.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. A session is skipped if it has zero such units. No extra per-unit metric thresholds are applied.

ii.
```python
classification = units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
...
if n_good == 0:
    print(f"  Session {session_id}: No good units, skipping")
    return None
```

iii. `CONVERSION_NOTES.md` says classifier-based QC from the papers should map to `units/classification == "good"`, and the trajectory explicitly identifies this as the intended neuron curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go-cue onset. For each trial the code adds fixed offsets from `-2.5` to `+1.5` s to that trial’s `go_time` to define absolute bin edges for spike counting.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
bin_offsets = align_start + np.arange(n_bins + 1) * bin_width
bin_edges = go_cue_times[t] + bin_offsets
```

iii. The notes and trajectory both say the decoder task requires alignment to go cue onset, and the AI reports using `go_start_times` as time zero.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, giving 80 time bins over a 4 s window. No secondary rebinning or smoothing is applied.

ii.
```python
BIN_WIDTH = 0.050
ALIGN_START = -2.5
ALIGN_END = 1.5
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)
BIN_CENTERS = ALIGN_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
```

iii. The notes say the task requirement overrode the reference code’s 100 ms width / 50 ms stride preprocessing, so the AI adopted 50 ms bins directly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and `go_start_times`. For each trial the AI uses the most recent sample-start event before the go cue as tone onset.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
prev_samples = sample_start_times[sample_start_times < go_time + 0.01]
if len(prev_samples) > 0:
    tone_onset = prev_samples[-1]
```

iii. The trajectory states that sample onset is about `-1.85 s` relative to the go cue and that replayed sample epochs on early-lick trials require choosing the last sample event before go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding the trial’s tone onset, the AI subtracts it from each absolute bin center `(go_time + BIN_CENTERS)` to create a continuous time-varying signal. If no preceding sample event is found, it falls back to `go_time - 1.85`.

ii.
```python
if len(prev_samples) > 0:
    tone_onset = prev_samples[-1]
else:
    tone_onset = go_time - 1.85
time_from_tone = (go_time + BIN_CENTERS) - tone_onset
```

iii. The trajectory justifies the main step by the observed `~ -1.85 s` tone-to-go timing and replayed sample epochs. The fallback was not separately justified beyond that empirical timing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same 80 bin centers used for neural firing rates, so the time-from-tone value at each index matches the neural activity in that bin.

ii.
```python
BIN_CENTERS = ALIGN_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
...
time_from_tone = (go_time + BIN_CENTERS) - tone_onset
```

iii. The notes say the decoder inputs should share the same go-cue-aligned 50 ms grid as the neural data, and the code implements that directly.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, while using `trials['photostim_power']` to skip non-stim trials.

ii.
```python
has_photostim_events = 'photostim_start_times' in be.time_series
if has_photostim_events:
    photostim_event_starts = be.time_series['photostim_start_times'].timestamps[:]
    photostim_event_stops = be.time_series['photostim_stop_times'].timestamps[:]
```

```python
photostim_power_str = trials['photostim_power'][:]
```

iii. The trajectory says the AI investigated the timing fields and concluded that the event streams give absolute photostim times, while `photostim_onset` in the trials table is relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each stimulated trial, the AI builds a binary 80-bin series and sets a bin to `1` if the absolute bin interval overlaps any photostim event interval; otherwise it stays `0`.

ii.
```python
photostim_binary = np.zeros(N_TIMEBINS, dtype=np.float32)
if has_photostim_events and photostim_power_str[trial_idx] != 'N/A':
    for ps_idx in range(len(photostim_event_starts)):
        ps_start = photostim_event_starts[ps_idx]
        ps_stop = photostim_event_stops[ps_idx]
        ...
        for b in range(N_TIMEBINS):
            bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH
            bin_end_abs = bin_start_abs + BIN_WIDTH
            if bin_start_abs < ps_stop and bin_end_abs > ps_start:
                photostim_binary[b] = 1.0
```

iii. The trajectory argues that photostim occurs during the late delay and is best represented as an on/off time series on the aligned bins. No explicit justification was given for using interval-overlap rather than bin-center tests.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is done on the same absolute go-cue-centered time axis as the neural data: each trial’s bins are converted to absolute time and compared to absolute photostim event start/stop times.

ii.
```python
bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH
bin_end_abs = bin_start_abs + BIN_WIDTH
if bin_start_abs < ps_stop and bin_end_abs > ps_start:
    photostim_binary[b] = 1.0
```

iii. The trajectory explicitly says the AI used absolute photostim event times relative to the go cue so they would share the neural alignment frame.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a dedicated raw variable. It is derived from the trial-table fields `trial_instruction` and `outcome`.

ii.
```python
trial_instruction = trials['trial_instruction'][:]
outcome = trials['outcome'][:]
...
choice = determine_choice(trial_instruction[trial_idx], outcome[trial_idx])
```

iii. The notes map “instruction + outcome” to choice, and the trajectory states that hits correspond to instructed-side licks, misses to opposite-side licks, and ignores to no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `hit` to the instructed side, `miss` to the opposite side, and `ignore` to a third `no_lick` class. That scalar choice is then repeated across all 80 bins in the output tensor.

ii.
```python
def determine_choice(trial_instruction, outcome):
    if outcome == 'ignore':
        return 2
    elif outcome == 'hit':
        return 0 if trial_instruction == 'left' else 1
    elif outcome == 'miss':
        return 1 if trial_instruction == 'left' else 0
    return 2
```

```python
per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
```

iii. The trajectory contains a detailed choice-mapping plan based on `trial_instruction` plus `outcome`, and the README documents the resulting code mapping as left/right/no_lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table field `trials['outcome']`.

ii.
```python
outcome = trials['outcome'][:]
```

iii. The notes describe outcome as a direct trial-table mapping with categories `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped as `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeated across all 80 bins in each trial output array.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome[trial_idx], 0)
```

```python
per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
```

iii. The notes and README both document this fixed categorical mapping for the decoder.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table field `trials['early_lick']`.

ii.
```python
early_lick = trials['early_lick'][:]
```

iii. The notes describe early lick as a direct mapping from the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `early` to `1` and `no early` to `0`, then repeats that per-trial value across all 80 bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

```python
per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
```

iii. The notes and README both describe early lick as a binary categorical output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of the `(x, y, likelihood)` data is used as `y`, and column 2 is used as a visibility/confidence signal.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
```

```python
visible = frames[:, 2] >= TONGUE_LIKELIHOOD_THRESHOLD
...
tongue_y_binned[b] = frames[visible, 1].mean()
```

iii. The trajectory identifies `Camera0_side_TongueTracking` as the relevant 300 Hz behavioral time series with three columns corresponding to tracked tongue position and likelihood.

## 8-b. How is `output` *Tongue y-position* derived?

i. The AI bins tongue frames within each trial window, averages `y` only over frames with likelihood `>= 0.9`, and records for each 50 ms bin both the mean visible `y` value and the fraction of visible frames. After processing all valid trials in a session, it pools visible-bin `y` values across the session and takes the 40th and 60th percentiles for discretization.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
```

```python
for b in range(n_bins):
    mask = bin_idx == b
    ...
    visible = frames[:, 2] >= TONGUE_LIKELIHOOD_THRESHOLD
    visible_binned[b] = visible.mean()
    if visible.sum() > 0:
        tongue_y_binned[b] = frames[visible, 1].mean()
```

```python
visible_mask = tongue_vis >= 0.5
if visible_mask.sum() > 0:
    tongue_y_all_visible.extend(tongue_y[visible_mask].tolist())
...
p40 = np.percentile(tongue_y_arr, 40)
p60 = np.percentile(tongue_y_arr, 60)
```

iii. The notes list “Tongue visibility: 0.9 DLC likelihood threshold” as a key decision. The trajectory says the AI wanted bins where the tongue was confidently visible and then used per-session percentiles over those visible values.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Each time bin starts as class `3` (`not visible`). Bins with at least 50% visible frames are compared against session-wide `p40` and `p60`: values `< p40` become `0`, values `<= p60` become `1`, and values `> p60` become `2`.

ii.
```python
tongue_discrete = np.full(N_TIMEBINS, 3, dtype=np.int64)
visible_mask = tongue_vis >= 0.5
if visible_mask.sum() > 0 and len(tongue_y_all_visible) > 0:
    ty = tongue_y[visible_mask]
    tongue_discrete[visible_mask] = np.where(ty < p40, 0,
                                             np.where(ty <= p60, 1, 2))
```

iii. The notes say the decoder output should use the task’s 40th/60th percentile categories, and the AI added a majority-visible rule before applying those thresholds.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial the AI takes camera timestamps from `go_time - 2.5` to `go_time + 1.5`, bins them onto the same 50 ms go-cue-aligned grid, and outputs one tongue class per neural time bin.

ii.
```python
bin_edges = go_cue_time + align_start + np.arange(n_bins + 1) * bin_width
left_idx = np.searchsorted(tongue_ts, bin_edges[0])
right_idx = np.searchsorted(tongue_ts, bin_edges[-1])
bin_idx = np.digitize(ts_window, bin_edges) - 1
```

iii. The trajectory describes using the same go-cue-relative time frame for tongue tracking and neural activity so the time-varying output is directly aligned to the neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing `pynwb` triggers an on-the-fly `pip install`. Sessions with no good units are dropped. Trials outside the inferred recording range are dropped. If no sample event is found before go cue, tone onset is imputed as `go_time - 1.85`. Missing or low-confidence tongue frames become `NaN`, and bins with insufficient visibility remain class `3` (`not visible`).

ii.
```python
try:
    import pynwb
except ImportError:
    os.system('pip install pynwb')
    import pynwb
```

```python
if n_good == 0:
    return None
...
if window_start < rec_min or window_end > rec_max:
    valid_trial_mask[i] = False
```

```python
if len(prev_samples) > 0:
    tone_onset = prev_samples[-1]
else:
    tone_onset = go_time - 1.85
```

iii. The trajectory says these choices were meant to handle partial recordings, replay-related event timing, and tongue-tracking visibility; the notes mention partial recordings and the tongue visibility threshold explicitly.

## 10-a. What are the most time-consuming steps of the code?

i. The AI treated session processing as dominated by reading unit/tongue data from NWB and by firing-rate computation. The code prints timings for “Reading units,” “Reading tongue,” “Computing firing rates,” and “Processing I/O,” and the notes estimate about 3.5 s per session and about 10 minutes total.

ii.
```python
t_units_start = time.time()
...
t_units_end = time.time()
print(f"    Reading units: {t_units_end - t_units_start:.1f}s")
```

```python
t_fr_start = time.time()
neural_trials = compute_firing_rates_vectorized(
    good_spike_times, valid_go_times, n_good)
t_fr_end = time.time()
print(f"    Computing firing rates: {t_fr_end - t_fr_start:.1f}s")
```

iii. The notes and trajectory repeatedly discuss timing output and use it to argue the code is within the runtime budget, with firing-rate computation highlighted as the main computational hotspot.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain unvectorized: the trial loop and nested neuron loop in `compute_firing_rates_vectorized`, the nested photostim-event and bin loops, the per-bin loop in `get_tongue_y_for_trial`, and the session-level loops that build region indices and summary distributions.

ii.
```python
for t in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes_window, bins=bin_edges)
```

```python
for ps_idx in range(len(photostim_event_starts)):
    ...
    for b in range(N_TIMEBINS):
```

```python
for b in range(n_bins):
    mask = bin_idx == b
```

iii. The trajectory says the AI optimized enough to fit the 15-minute target but did not describe these remaining loops as fully optimized.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans all photostim events for every stimulated trial, rebuilds per-trial bin edges in several places, computes subject and region indices with repeated linear searches, and expands per-trial categorical outputs across 80 bins for every trial.

ii.
```python
for i, trial_idx in enumerate(valid_trial_indices):
    ...
    for ps_idx in range(len(photostim_event_starts)):
        ...
        for b in range(N_TIMEBINS):
```

```python
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
...
idx = np.array([all_region_names.index(r) for r in regions])
```

iii. The trajectory focuses on runtime budget rather than eliminating repeated work; it treats the code as “optimized enough” once full conversion finished in time.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints `correct_rate` only for reporting, carries `n_total_trials` and `rec_range` in per-session intermediate results without saving them to the final dataset, optionally generates diagnostic plots, and spends time computing summary distributions purely for console output.

ii.
```python
control_mask_all = (photostim_power_str == 'N/A') & (early_lick == 'no early')
ctrl_out = outcome[control_mask_all]
...
correct_rate = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0
```

```python
return {
    'subject_id': subject_id,
    ...
    'correct_rate': correct_rate,
    'n_total_trials': n_trials,
    'rec_range': (rec_min, rec_max),
}
```

```python
if show_processing:
    ...
    plt.savefig(f'/app/processing_{safe_id}.png', dpi=100)
```

iii. The notes emphasize validation, timing, and plotting for sanity checks, so this extra work appears to have been done for diagnostics rather than for the final decoder dataset.
