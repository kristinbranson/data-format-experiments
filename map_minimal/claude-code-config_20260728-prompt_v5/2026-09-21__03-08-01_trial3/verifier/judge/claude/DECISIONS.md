# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files using `h5py` (not `pynwb`). It iterates over subject directories (`sub-*`) in `/app/data`, then iterates over `.nwb` files within each subject directory. Each file is opened with `h5py.File` and data is accessed via HDF5 paths like `f['intervals/trials/outcome']`, `f['units/classification']`, etc.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    n_trials = len(f['intervals/trials/id'])
    outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])
    ...
```

```python
for sub in subjects:
    sub_dir = os.path.join(DATA_DIR, sub)
    nwb_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.nwb')])
    for nf in nwb_files:
        nwb_path = os.path.join(sub_dir, nf)
        result = process_session(nwb_path)
```

iii. The agent used `h5py` from the start when exploring the data files. It treated the NWB files as plain HDF5 files, accessing groups and datasets directly. No explicit reasoning was given for choosing h5py over pynwb.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the directory names (`sub-*`) under the data directory. The AI iterates over sorted subject directories and tracks which subjects have at least one valid session, building a `subject_names` list. Subject index is determined by position in this list.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
...
for sub in subjects:
    ...
    if not sub_has_session:
        subject_names.append(sub)
        sub_has_session = True
    sub_idx = subject_names.index(sub)
    all_subject_idx.append(sub_idx)
```

iii. The agent used the directory structure to identify subjects, with each `sub-*` directory containing all sessions for one subject.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI iterates over NWB files within each subject directory. Sessions that fail selection criteria (performance thresholds) are skipped.

ii.
```python
nwb_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.nwb')])
for nf in nwb_files:
    nwb_path = os.path.join(sub_dir, nf)
    result = process_session(nwb_path)
    if result is None:
        sessions_skipped += 1
        continue
```

iii. The agent recognized that each NWB file is one session and processed them individually.

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table in each NWB file. The number of trials is determined by `len(f['intervals/trials/id'])`. Each trial has associated metadata (outcome, early_lick, trial_instruction, etc.).

ii.
```python
n_trials = len(f['intervals/trials/id'])
outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])
early_lick = np.array([x.decode() for x in f['intervals/trials/early_lick'][:]])
trial_instruction = np.array([x.decode() for x in f['intervals/trials/trial_instruction'][:]])
```

iii. The agent used the trials table directly from the NWB file. No explicit verification was done that the number of go cue events matches the number of trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two levels of filtering. First, **session-level**: sessions must have >65% correct on control trials (no stim, no early lick, no auto/free water, response present) and >=50 correct left and >=50 correct right. Second, **trial-level**: `auto_water` and `free_water` trials are excluded. Early-lick, no-response, and photostim trials are kept.

ii.
```python
# Session selection
control_mask = (
    (auto_water == 0) & (free_water == 0) &
    (early_lick == 'no early') & (~has_stim) &
    (outcome != 'ignore')
)
performance = correct_control.sum() / n_control
if performance <= 0.65 or n_correct_left < 50 or n_correct_right < 50:
    return None

# Trial filtering
trial_mask = (auto_water == 0) & (free_water == 0)
```

iii. The agent reasoned: "Session selection criteria from the methods require over 65% correct performance on control trials and at least 50 correct trials per lick direction." For trial filtering, the agent argued: "The decoder's output categories (no lick, ignore, early_lick yes/no) explicitly require these trials to be present, so excluding them would prevent the model from learning to predict them." Auto_water and free_water trials were excluded as "they fall outside the normal task structure."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the flat spike time array) and `units/spike_times_index` (the ragged array index), accessed via h5py. Only units with `classification == 'good'` are used.

ii.
```python
spike_times_flat = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
...
for uid in good_indices:
    start_idx = 0 if uid == 0 else spike_times_index[uid - 1]
    end_idx = spike_times_index[uid]
    unit_spikes.append(spike_times_flat[start_idx:end_idx])
```

iii. The agent accessed spike times from the HDF5 file structure directly. This is equivalent to reading `nwb.units['spike_times']` with pynwb.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue and binned into 50ms non-overlapping bins spanning -2.5s to +1.5s. The `bin_spike_times` function uses `np.histogram` to count spikes per bin, then divides by bin width to get firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
def bin_spike_times(spike_times, t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    edges = np.linspace(t_start, t_end, n_bins + 1)
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts / bin_width
```

```python
for u, spks in enumerate(unit_spikes):
    aligned = spks - go_t
    fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```

iii. The agent used standard histogram binning to compute firing rates. The approach is functionally equivalent to the reference's `np.searchsorted`-based approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions where classification data is all NaN (non-string dtype) are skipped entirely. No additional quality metric thresholds are applied.

ii.
```python
clf_raw = f['units/classification'][:]
if clf_raw.dtype == object:
    classification = np.array([x.decode() if isinstance(x, bytes) else str(x)
                               for x in clf_raw])
    good_mask = classification == 'good'
else:
    # No classifier QC available (all NaN) -- skip session
    return None
```

iii. The agent reasoned: "The reference code uses `classification == 'good'` for QC." When encountering NaN classification values, it decided: "Sessions missing classification may simply not have been part of that analysis, so falling back to unit_quality == 'good' might not match intent." It chose to skip such sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time from each spike time (`aligned = spks - go_t`), then binning in the window [-2.5, 1.5] relative to the go cue.

ii.
```python
go_t = go_times[trial_idx]
for u, spks in enumerate(unit_spikes):
    aligned = spks - go_t
    fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```

iii. The agent aligned all data to the go cue onset as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50ms (0.05s), with 80 non-overlapping bins spanning -2.5s to +1.5s relative to the go cue. No rebinning or smoothing is applied — spikes are binned directly from the raw spike times.

ii.
```python
BIN_WIDTH = 0.05       # 50 ms
T_START = -2.5
T_END = 1.5
```

```python
def get_bin_centers(t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    edges = np.linspace(t_start, t_end, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2
```

iii. The 50ms bin width and [-2.5, 1.5] window are directly from the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset timestamps) from `acquisition/BehavioralEvents` and the go cue times. For each trial, the last `sample_start` before the go cue is taken as the tone onset.

ii.
```python
sample_start_ts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
tone_onset_rel_go = np.full(n_trials, np.nan)
for i in range(n_trials):
    before = sample_start_ts[sample_start_ts < go_times[i]]
    if len(before) > 0:
        tone_onset_rel_go[i] = before[-1] - go_times[i]
```

iii. The agent reasoned: "The sample_start_times has 405 entries but go_start_times has 368. The extra ones are due to early lick replays." It took the last sample_start before each go cue as the tone onset. If no sample_start is found, it falls back to -1.85s.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `bin_centers - tone_rel`, where `tone_rel` is the tone onset time relative to the go cue (negative, since tone precedes go cue). This gives a continuous, time-varying value for each bin: the elapsed time since the tone onset.

ii.
```python
if np.isnan(tone_onset_rel_go[trial_idx]):
    tone_rel = -1.85
else:
    tone_rel = tone_onset_rel_go[trial_idx]
time_from_tone = bin_centers - tone_rel
```

iii. The agent noted: "The tone onset (sample start) is consistently 1.85 seconds before the go cue." The fallback value of -1.85 is used when no sample_start is found before the go cue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers are defined relative to the go cue, and the tone onset is also expressed relative to the go cue, so the time-from-tone values are naturally aligned with the neural data bins.

ii.
```python
bin_centers = get_bin_centers(T_START, T_END, BIN_WIDTH)
time_from_tone = bin_centers - tone_rel
```

iii. Both the neural data and the time-from-tone input share the same bin centers defined relative to the go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `photostim_onset` (onset time relative to trial start, as a string) and `photostim_duration` (duration as a string) from the trials table, along with `go_times` and `trial_start_times` for alignment.

ii.
```python
photostim_onset_str = np.array(
    [x.decode() for x in f['intervals/trials/photostim_onset'][:]])
photostim_dur_str = np.array(
    [x.decode() for x in f['intervals/trials/photostim_duration'][:]])
```

iii. The agent examined the actual timestamps from the NWB photostim data and computed on/off times relative to the go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The photostim onset (relative to trial start) is converted to be relative to the go cue. Then a binary time series is created: 1 where bin centers fall between onset and offset, 0 elsewhere. Trials with `'N/A'` onset get all-zero photostim.

ii.
```python
for i in range(n_trials):
    if photostim_onset_str[i] != 'N/A':
        onset_from_trial_start = float(photostim_onset_str[i])
        duration = float(photostim_dur_str[i])
        go_from_trial_start = go_times[i] - trial_start_times[i]
        photostim_on_rel[i] = onset_from_trial_start - go_from_trial_start
        photostim_off_rel[i] = onset_from_trial_start + duration - go_from_trial_start
```

```python
photostim_binary = ((bin_centers >= on_t) & (bin_centers < off_t)).astype(float)
```

iii. The agent constructed a binary time series for photostimulation as specified by the instructions.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim onset and offset are expressed relative to the go cue (same reference as the neural bin centers), so the binary time series is naturally aligned.

ii.
```python
go_from_trial_start = go_times[i] - trial_start_times[i]
photostim_on_rel[i] = onset_from_trial_start - go_from_trial_start
```

iii. The agent converted photostim times to the same go-cue-relative coordinate system used for neural binning.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) columns of the trials table. Choice is inferred: hit means the animal chose the instructed side, miss means the opposite, ignore means no lick.

ii.
```python
out = outcome[trial_idx]
instr = trial_instruction[trial_idx]
if out == 'ignore':
    choice = 2  # no lick
elif out == 'hit':
    choice = 0 if instr == 'left' else 1
else:  # miss
    choice = 1 if instr == 'left' else 0
```

iii. The agent reasoned: "For determining choice, I can infer it from trial_instruction and outcome: hits mean the mouse chose the instructed side, misses mean it chose the opposite side, and ignore trials mean no lick occurred at all."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no_lick. It is a per-trial scalar repeated across all 80 time bins.

ii.
```python
output_data[0, :] = choice
```

iii. The agent made choice a per-trial constant value broadcast across time bins.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains strings: 'ignore', 'miss', 'hit'.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]
```

iii. The trials table stores outcome explicitly with the three required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers: 0=ignore, 1=miss, 2=hit. The value is per-trial, repeated across all time bins.

ii.
```python
output_data[1, :] = outcome_val
```

iii. Straightforward mapping following the instruction categories.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains 'no early' or 'early'.

ii.
```python
early_val = 0 if early_lick[trial_idx] == 'no early' else 1
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Per-trial value repeated across all time bins.

ii.
```python
output_data[2, :] = early_val
```

iii. Simple binary encoding as specified.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data: tongue_x, tongue_y, tongue_likelihood, with timestamps.

ii.
```python
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The agent identified the tongue tracking data from the NWB file structure.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Tongue y-position is discretized using per-session percentiles. Visible frames are those with `likelihood > 0.9`. The 40th and 60th percentiles are computed over **all visible raw frames** in the session (not binned means). For each trial bin, the **nearest single video frame** to the bin center is found; if that frame is visible, the y value is classified as 0 (<40th pct), 1 (40-60th pct), or 2 (>60th pct). Bins with no visible frame get class 3.

ii.
```python
visible_mask = tongue_likelihood > TONGUE_LIKELIHOOD_THRESH  # 0.9
y_visible = tongue_y[visible_mask]
pct40 = np.percentile(y_visible, 40)
pct60 = np.percentile(y_visible, 60)
```

```python
for b, tc in enumerate(bin_centers):
    abs_t = go_t + tc
    frame_idx = np.searchsorted(tongue_ts, abs_t)
    frame_idx = min(frame_idx, len(tongue_ts) - 1)
    if tongue_likelihood[frame_idx] > TONGUE_LIKELIHOOD_THRESH:
        y_val = tongue_y[frame_idx]
        if y_val < pct40:
            tongue_y_disc[b] = 0
        elif y_val <= pct60:
            tongue_y_disc[b] = 1
        else:
            tongue_y_disc[b] = 2
```

iii. The agent chose 0.9 as the likelihood threshold after empirically observing a bimodal distribution (values near 0 or near 1). It computed percentiles on all visible raw video frames across the session, then applied nearest-frame lookup for discretization.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session thresholds at the 40th and 60th percentiles of visible tongue y values. Categories: 0 = below 40th percentile, 1 = between 40th and 60th, 2 = above 60th, 3 = not visible. The boundary condition uses `<` for lower and `<=` for middle class.

ii.
```python
if y_val < pct40:
    tongue_y_disc[b] = 0
elif y_val <= pct60:
    tongue_y_disc[b] = 1
else:
    tongue_y_disc[b] = 2
```

iii. Follows the instruction specification for discretization categories.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each time bin, the absolute time is computed as `go_t + bin_center`. The nearest video frame to this time is found via `searchsorted`. This gives one tongue value per neural time bin.

ii.
```python
for b, tc in enumerate(bin_centers):
    abs_t = go_t + tc
    frame_idx = np.searchsorted(tongue_ts, abs_t)
```

iii. The tongue data is aligned to the same go-cue-relative bins as the neural data by converting bin centers to absolute times and finding the nearest frame.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Sessions with NaN classification data (non-string dtype) are skipped. (2) Trials where no sample_start is found before the go cue use a fallback tone onset of -1.85s. (3) Tongue bins where the nearest frame is not visible get class 3 ("not visible"). (4) Sessions failing performance criteria are skipped.

ii.
```python
if clf_raw.dtype == object:
    ...
else:
    return None  # No classifier QC available
```

```python
if np.isnan(tone_onset_rel_go[trial_idx]):
    tone_rel = -1.85  # Fallback
```

iii. The agent handled NaN classification by skipping sessions, and missing tone onsets with a hardcoded fallback value.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading each NWB file with h5py and reading the full spike_times array. (2) The per-trial, per-unit spike binning loop (nested loop over trials and units). (3) The per-trial, per-bin tongue tracking lookup (triply nested loop).

ii.
```python
for trial_idx in trial_indices:
    for u, spks in enumerate(unit_spikes):
        aligned = spks - go_t
        fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```

iii. The nested trial-unit loop calling `np.histogram` for each unit-trial pair is significantly less efficient than the reference's vectorized approach.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be improved: (1) The per-trial loop over units for spike binning — the reference vectorizes across trials by building all bin edges at once and using `np.searchsorted`. (2) The per-trial, per-bin loop for tongue tracking — the reference averages all frames in each bin rather than doing nearest-frame lookup.

ii.
```python
# Current: per-trial, per-unit
for trial_idx in trial_indices:
    for u, spks in enumerate(unit_spikes):
        aligned = spks - go_t
        fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```

iii. The per-unit loop is partly inherent to ragged spike time storage, but the per-trial dimension could be vectorized.

## 10-c. What processing does the code repeat multiple times?

i. The tone onset computation iterates over all trials with a Python loop, filtering `sample_start_ts` for each trial rather than using a vectorized approach. Similarly, the photostim timing computation uses a per-trial Python loop. These could be computed once with vectorized operations.

ii.
```python
for i in range(n_trials):
    before = sample_start_ts[sample_start_ts < go_times[i]]
    if len(before) > 0:
        tone_onset_rel_go[i] = before[-1] - go_times[i]
```

iii. While not strictly repeating the same computation, the per-trial loops are inefficient compared to vectorized alternatives.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The brain region mapping uses an elaborate keyword-based system with ~190 entries mapping CCF annotations to 14 broad categories. This is significantly more complex than necessary — the reference simply takes the first component of the annotation string. Additionally, the `auto_water` filtering is applied but auto_water trials may already be handled by obs_intervals filtering in the reference approach.

ii.
```python
_REGION_KEYWORDS = OrderedDict([
    ('Secondary motor area', 'ALM'),
    ('Orbital area', 'Orbital'),
    ... # ~190 keyword mappings
])
```

iii. The elaborate region mapping is not strictly "unnecessary" but represents over-engineering compared to the reference approach.
