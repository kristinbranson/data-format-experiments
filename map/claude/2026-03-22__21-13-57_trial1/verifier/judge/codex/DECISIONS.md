# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing every NWB file under `/app/data/sub-*/sub-*_ses-*.nwb`, then iterating through those files one session at a time. For each NWB file it opens the file with `pynwb.NWBHDF5IO` and extracts subject metadata, units/spike times, the trials table, behavioral event timestamps, and tongue-tracking time series into a session dictionary.

ii. ```python
def get_nwb_files():
    """Get list of all NWB files sorted by subject then session."""
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
...
units = nwb.units
trials = nwb.trials
...
be = nwb.acquisition['BehavioralEvents']
...
bts = nwb.acquisition['BehavioralTimeSeries']
```

```python
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by noting that the dataset is distributed as 174 NWB files across 28 subject directories and that each NWB contains the needed units, trials, behavioral events, and behavioral time series.

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from each NWB file’s `nwb.subject.subject_id`. During assembly, the AI accumulates unique subject ids in insertion order using an `OrderedDict`, then records each surviving session’s subject index into `subject_idx`.

ii. ```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description
```

```python
subjects_set = OrderedDict()
...
sid = result['subject_id']
if sid not in subjects_set:
    subjects_set[sid] = len(subjects_set)
...
subjects = list(subjects_set.keys())
subject_idx.append(subjects_set[sess['subject_id']])
```

iii. The notes say the NWB files are organized into 28 subject directories and that `subject.subject_id` should populate `subjects`; the AI therefore treated the NWB subject field as the authoritative mouse identity.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. It processes each file independently with `process_session`, but then drops entire sessions that fail session-level criteria such as low performance, insufficient correct left/right trials, no usable neurons, or fewer than two valid trials.

ii. ```python
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
    if result is None:
        continue
    all_sessions.append(result)
```

```python
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_LEFT:
    return None
if correct_right < MIN_CORRECT_RIGHT:
    return None
```

iii. In the notes, the AI explicitly framed the paper’s behavioral criteria as session-selection rules and concluded that 144 of 174 files should survive, with 30 sessions skipped for failing those criteria or having no good units.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table. The AI first reads all trial columns into `trials_data`, then builds a boolean `trial_mask` from trial-level filters, converts it to `trial_indices`, and uses those indices to subset `go_times` and the other trial arrays. The go-cue timestamps are therefore used as the per-trial alignment anchors after filtering.

ii. ```python
trials = nwb.trials
n_trials = len(trials)
trials_data = {
    'start_time': trials['start_time'][:],
    'stop_time': trials['stop_time'][:],
    'trial_instruction': trials['trial_instruction'][:],
    'outcome': trials['outcome'][:],
    ...
}
```

```python
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False

trial_indices = np.where(trial_mask)[0]
go_times = data['go_times'][trial_indices]
```

iii. In the trajectory, the AI noted that `go_start_times` mapped 1:1 with trials and used the trials table as the base trial structure, with additional subsetting only for exclusion rules and recording-range fixes.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial/session filters. At the trial level it excludes `auto_water == 1` and `free_water == 1`. It also removes trials whose aligned neural window extends beyond the minimum/maximum spike times of the retained neurons. At the session level it requires performance at least 65%, at least 50 correct left trials, at least 50 correct right trials, and at least two remaining trials.

ii. ```python
MIN_CORRECT_LEFT = 50
MIN_CORRECT_RIGHT = 50
MIN_PERFORMANCE = 0.65
```

```python
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False
```

```python
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
...
if n_valid_trials < 2:
    return None
```

iii. `CONVERSION_NOTES.md` says the AI kept early-lick, ignore, and stimulation trials because they are decoder outputs/inputs, but excluded auto-water and free-water trials as artificial conditions. Later notes say the AI added a recording-range filter after finding some behavioral trials continued outside the neural recording. The same notes also say it intentionally applied the paper’s session performance criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` output is derived from `units['spike_times']` for neurons with `classification == 'good'`, plus `anno_name` for brain-region filtering/labeling, and `go_start_times` for trial alignment.

ii. ```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
anno_names = units['anno_name'][:]
```

```python
spike_times_vi = units['spike_times']
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])
```

```python
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The notes and trajectory state that the classifier-based QC in the papers corresponds to `classification == 'good'` in NWB, and that spike times in NWB are stored on an absolute session clock and therefore must be aligned to the go cue.

## 2-b. How is the `neural` data processed?

i. The AI converts each retained unit’s absolute spike times into per-trial firing rates by taking spikes within `[go + ALIGN_START, go + ALIGN_END)`, subtracting the trial’s go cue to get relative times, binning those spikes into 80 non-overlapping 50 ms bins with `np.histogram`, and dividing counts by the bin width to produce Hz.

ii. ```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
all_matrices = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)
```

```python
idx_lo = np.searchsorted(st, abs_start, side='left')
idx_hi = np.searchsorted(st, abs_end, side='left')
if idx_hi > idx_lo:
    rel_spikes = st[idx_lo:idx_hi] - go_t
    counts, _ = np.histogram(rel_spikes, bins=bin_edges)
    all_matrices[t, n, :] = counts / bin_width
```

iii. In the notes, the AI justified this as the NWB analogue of the reference code’s go-cue-aligned spike processing, but with the task’s required 50 ms bins rather than the reference code’s 40 ms / 3.4 ms sliding histogram.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == 'good'`, then further drops any of those good units whose `anno_name` cannot be mapped into one of its 14 coarse brain-region categories. If a session ends up with no such neurons, the entire session is dropped.

ii. ```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
...
good_indices = np.where(good_mask)[0]
```

```python
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
    if region is not None:
        region_labels.append(region)
        neuron_mask.append(i)
...
if n_neurons < 1:
    return None
```

iii. The notes say the AI intentionally used classifier-based QC via `classification == 'good'` and then mapped `anno_name` into the “14 major regions” used in the paper code, treating unmapped annotations as unusable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go-cue onset. For each trial, the AI builds an absolute spike window from `go_t + ALIGN_START` to `go_t + ALIGN_END`, then converts spikes in that window to relative time by subtracting `go_t` before binning.

ii. ```python
ALIGN_START = -2.5
ALIGN_END = 1.5
...
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
abs_start = go_t + align_start
abs_end = go_t + align_end
...
rel_spikes = st[idx_lo:idx_hi] - go_t
```

iii. The notes repeatedly state that NWB timestamps are absolute session times, unlike the already aligned `.mat` files in the paper code, so the AI aligned all streams by converting them into go-cue-relative time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins over a 4 s window, from -2.5 s to +1.5 s relative to go cue, yielding 80 time bins. No additional smoothing or rebinning is applied after the initial histogramming.

ii. ```python
BIN_WIDTH = 0.050
ALIGN_START = -2.5
ALIGN_END = 1.5
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)
```

```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
all_matrices[t, n, :] = counts / bin_width
```

iii. The notes explicitly say the AI chose 50 ms non-overlapping bins because the decoder-task instructions overrode the reference code’s 40 ms bin width and 3.4 ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `sample_start_times` in the behavioral events, together with per-trial `start_time`, `stop_time`, and `go_times`. For each trial, the AI looks for sample-start events within the trial and at or before the go cue, then takes the last such event as the tone onset.

ii. ```python
sample_start_ts = be.time_series['sample_start_times'].timestamps[:]
```

```python
def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    ...
    mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
    matching = sample_start_ts[mask]
    if len(matching) > 0:
        tone_onsets[i] = matching[-1]
```

iii. The trajectory says the AI found repeated sample-start events due to early-lick trial replays and therefore decided to use the last sample-start event before the go cue for each trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes bin centers on the go-cue-relative grid, then for each trial subtracts the tone onset from the absolute time of each bin center. If no tone onset is found for a trial, it fills the input with zeros.

ii. ```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```

```python
if np.isnan(tone_t):
    tone_input = np.zeros(n_bins, dtype=np.float32)
else:
    tone_rel = tone_t - go_t
    tone_input = (bin_centers - tone_rel).astype(np.float32)
```

iii. The trajectory records the AI’s rationale that the tone starts about 1.85 s before go cue and that “time from tone onset” should be a continuous time-varying variable on the same bin grid as the neural data.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI uses the exact same 80 go-cue-centered bin centers that define the neural data, so `time_from_tone_onset` is sampled on the same per-trial time axis as the neural bins.

ii. ```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
...
tone_input = (bin_centers - tone_rel).astype(np.float32)
```

```python
neural_trials = bin_spikes(spike_times_list, go_times, ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
tone_onset_input = compute_tone_onset_input(go_times, tone_onsets, ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
```

iii. The notes treat all decoded variables as sharing one common go-cue-aligned 50 ms grid; this input follows that same grid directly.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation input is derived from the behavioral event streams `photostim_start_times` and `photostim_stop_times`. The AI also reads trial-table `photostim_onset` / `photostim_duration`, but only uses those for performance filtering, not for the decoder input itself.

ii. ```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

```python
photostim_onset = trials_data['photostim_onset']
...
if photostim_onset[i] != 'N/A':
    control_mask[i] = False
```

iii. In `CONVERSION_NOTES.md`, the AI’s mapping plan says `Photostim start/stop times -> input[1]: photostim_on`, so it intentionally chose the event timestamps as the source of the time-varying decoder input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI initializes an all-zero vector, iterates over every photostimulation event in the session, converts each event to time relative to that trial’s go cue, and marks each bin center as 1 when it falls inside any stimulation interval.

ii. ```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
...
ps = np.zeros(n_bins, dtype=np.float32)
```

```python
for si in range(len(photostim_start_ts)):
    ps_start = photostim_start_ts[si] - go_t
    ps_stop = photostim_stop_ts[si] - go_t
    ...
    for b in range(n_bins):
        bc = bin_centers[b]
        if bc >= ps_start and bc < ps_stop:
            ps[b] = 1.0
```

iii. The notes say the photostimulation input should be a binary time-varying signal rather than a per-trial flag, so the AI marked bins where stimulation was active.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is done by subtracting each trial’s `go_t` from the absolute photostimulation event timestamps and then testing those relative intervals against the same go-cue-relative bin centers used for neural data.

ii. ```python
ps_start = photostim_start_ts[si] - go_t
ps_stop = photostim_stop_ts[si] - go_t
```

```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```

iii. The AI’s notes consistently frame go cue as the common alignment event for all streams, and this function implements that by re-expressing photostim on the neural time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives `choice` only from the trials-table `trial_instruction` field. It does not use `outcome` or lick events to infer actual lick direction.

ii. ```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. In the notes’ variable-mapping table, the AI explicitly mapped `trial_instruction` to `output[0]: choice` and justified keeping ignore trials separately via the outcome output instead of modifying the choice variable.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes `left` as 0 and `right` as 1, then repeats the per-trial value across all 80 time bins. It defines only two output labels for choice.

ii. ```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

```python
out = np.array([
    np.full(N_BINS, choices[t], dtype=np.int64),
    np.full(N_BINS, outcomes[t], dtype=np.int64),
    np.full(N_BINS, early_licks[t], dtype=np.int64),
    tongue_y_discrete[t].astype(np.int64),
], dtype=np.int64)
```

iii. The notes state `choice` should be left/right per trial, and the AI implemented that as a broadcast per-trial categorical variable.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` column.

ii. ```python
outcomes_raw = td['outcome'][trial_indices]
```

iii. The notes’ mapping table lists `outcome` as a direct trial-table mapping to decoder output, with the instruction-specified ignore/miss/hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the per-trial code across all 80 bins.

ii. ```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```

```python
np.full(N_BINS, outcomes[t], dtype=np.int64)
```

iii. The notes say this mapping comes directly from the decoder specification and that the variable is per-trial rather than genuinely time-varying.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes directly from the trials-table `early_lick` column.

ii. ```python
early_lick_raw = td['early_lick'][trial_indices]
```

iii. The notes’ variable map treats early lick as a direct trial-table output and explicitly says those trials should be kept because early lick itself is one of the decoder targets.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI encodes `'early'` as 1 and everything else as 0, then repeats the per-trial label across all 80 bins.

ii. ```python
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
```

```python
np.full(N_BINS, early_licks[t], dtype=np.int64)
```

iii. The notes say early lick should be a binary categorical output (`no`/`yes`) and broadcast across time like the other per-trial outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the `Camera0_side_TongueTracking` behavioral time series: `data[:, 1]` is the y position and `data[:, 2]` is the confidence/visibility score. The matching `timestamps` define the frame times.

ii. ```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]  # (n_frames, 3): x, y, confidence
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

```python
tongue_y = tongue_data[:, 1].astype(np.float64)
tongue_conf = tongue_data[:, 2].astype(np.float64)
```

iii. The trajectory and notes say the AI identified the side-camera tongue tracking stream as the relevant video variable and interpreted its three columns as x, y, and confidence.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first marks frames with confidence below 0.9 as not visible and imputes their y value with the session mean of visible frames. It then extracts frames inside each trial’s aligned window, averages the (possibly imputed) y values within each 50 ms bin, and fills bins with no frames at all using the same session mean.

ii. ```python
TONGUE_CONFIDENCE_THRESHOLD = 0.9
```

```python
visible_mask = tongue_conf >= confidence_threshold
if np.sum(visible_mask) > 0:
    session_mean_y = np.mean(tongue_y[visible_mask])
else:
    session_mean_y = np.mean(tongue_y)

tongue_y_imputed = tongue_y.copy()
tongue_y_imputed[~visible_mask] = session_mean_y
```

```python
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
trial_y = tongue_y_imputed[mask]
...
trial_tongue_y = np.full(n_bins, session_mean_y, dtype=np.float64)
for b in range(n_bins):
    in_bin = trial_y[bin_indices == b]
    if len(in_bin) > 0:
        trial_tongue_y[b] = np.mean(in_bin)
```

iii. `CONVERSION_NOTES.md` says the AI followed a paper statement that when the tongue was occluded its position should be set to the mean value, and later notes explain that this imputation caused many values to pile up at the mean.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After constructing all per-bin tongue y traces, the AI concatenates every bin from every retained trial in the session, computes the 40th and 60th percentiles over those values including imputed bins, perturbs the thresholds slightly if they are equal, and then assigns three classes: below p40, between p40 and p60, and at/above p60. It does not use a separate hidden/not-visible category.

ii. ```python
all_values = np.concatenate([t for t in tongue_y_trials])

p40 = np.percentile(all_values, 40)
p60 = np.percentile(all_values, 60)
```

```python
if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
    p40 = p40 - eps
    p60 = p60 + eps
```

```python
d = np.zeros(len(trial_y), dtype=np.int64)
d[trial_y >= p40] = 1
d[trial_y >= p60] = 2
```

iii. The notes explicitly say the percentiles are computed “over ALL time bins (including imputed values)” and that the code adds a small offset when `p40 == p60` so that all three categories exist.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data to go cue by selecting camera frames in `[go + ALIGN_START, go + ALIGN_END)`, subtracting `go_t` from their timestamps, and binning those relative times into the same 80 50 ms bins used for the neural data.

ii. ```python
window_start = go_t + align_start
window_end = go_t + align_end
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
```

```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
bin_indices = np.digitize(trial_ts, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
```

iii. The notes and plotting code treat go cue as the common temporal reference for neural data and all behavioral variables, so tongue y is put on that same grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses a mix of dropping and imputation. Missing or low-confidence tongue values are replaced with session mean. Trials with no tongue frames in the aligned window are filled entirely with session mean. Trials outside the neural recording range are dropped. Sessions with no usable neurons, too few valid trials, or failing session criteria are dropped. If no tone onset is found, the time-from-tone input is filled with zeros. If the tongue percentiles collapse, the thresholds are artificially separated with a small epsilon.

ii. ```python
if np.isnan(tone_t):
    tone_input = np.zeros(n_bins, dtype=np.float32)
```

```python
tongue_y_imputed[~visible_mask] = session_mean_y
...
if len(trial_ts) == 0:
    tongue_y_trials.append(np.full(n_bins, session_mean_y, dtype=np.float32))
```

```python
if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
    p40 = p40 - eps
    p60 = p60 + eps
```

```python
if n_neurons < 1:
    return None
...
if n_valid_trials < 2:
    return None
```

iii. The notes describe these as pragmatic fixes: recording-range exclusion was added after the AI found zero-neural-data trials, tongue occlusion was handled by mean imputation, and equal percentile thresholds were manually separated so the output had three tongue categories.

## 10-a. What are the most time-consuming steps of the code?

i. According to the AI’s own notes, the main costs are NWB loading, spike binning, and tongue processing. It estimated about 1.5 s per session for NWB loading, 3.3 s for spike binning, 1.5 s for tongue processing, and about 7.5 s total per session.

ii. ```python
t0 = time.time()
data = load_nwb_session(nwb_path)
t_load = time.time() - t0
...
t1 = time.time()
neural_trials = bin_spikes(...)
t_bin = time.time() - t1
...
t2 = time.time()
tongue_y_trials, session_mean_y = compute_tongue_y_per_trial(...)
t_tongue = time.time() - t2
```

iii. Step 7 of `CONVERSION_NOTES.md` gives explicit runtime estimates and says those three stages dominate the per-session runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain non-vectorized: the nested neuron-by-trial spike-binning loop, the trial-by-event-by-bin photostimulation loop, the per-trial tone-onset search, the per-trial/per-bin tongue averaging loop, and the per-trial output assembly loop.

ii. ```python
for n in range(n_neurons):
    ...
    for t in range(n_trials):
        ...
        counts, _ = np.histogram(rel_spikes, bins=bin_edges)
```

```python
for t in range(n_trials):
    ...
    for si in range(len(photostim_start_ts)):
        ...
        for b in range(n_bins):
```

```python
for i in range(n_trials):
    mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
```

iii. The AI described parts of the implementation as “vectorized,” but the code still contains several explicit Python loops over trials, neurons, bins, and stimulation events that could have been reduced further.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans the same arrays in separate passes. It scans all trials once for session-performance filtering and again for conversion; it scans all sample-start events separately for every trial to find tone onset; it scans all photostimulation events separately for every trial; and it first computes continuous tongue traces and then makes another full pass to discretize them.

ii. ```python
performance, correct_left, correct_right = compute_session_performance(data['trials_data'])
...
trial_mask = np.ones(data['n_trials'], dtype=bool)
```

```python
for i in range(n_trials):
    mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
```

```python
for t in range(n_trials):
    for si in range(len(photostim_start_ts)):
```

```python
tongue_y_trials, session_mean_y = compute_tongue_y_per_trial(...)
tongue_y_discrete = discretize_tongue_y(tongue_y_trials, session_mean_y)
```

iii. This follows directly from the implemented helper structure in `convert_data.py`; the notes also mention a separate optimization pass for tongue processing, which implies the AI was aware this area was expensive.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI performs and stores some work that is not needed for the decoder arrays themselves. It loads `left_lick_ts`, `right_lick_ts`, `photostim_power`, `subject_desc`, and `n_total_units` without using them in conversion; computes session performance statistics mainly to decide whether to keep a session and for metadata/logging; includes optional plotting code; and carries auxiliary fields like `subject_desc` and `performance` only into metadata rather than the decoder inputs/outputs.

ii. ```python
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
...
'photostim_power': trials['photostim_power'][:],
...
'n_total_units': n_units,
```

```python
'subject_desc': subject_desc,
...
'performance': performance,
'correct_left': correct_left,
'correct_right': correct_right,
```

```python
def plot_processing(...):
    ...
```

iii. The notes show that the AI invested significant effort in paper-style session statistics, plotting, and sanity-check metadata in addition to the minimal converted dataset required for downstream decoding.
