# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all NWB files found under `/app/data/sub-*/`, one file per session. It opens each file using `pynwb.NWBHDF5IO` (without a context manager), reads units, trials, behavioral events, and tongue tracking data, then closes the file handle manually.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
all_files = []
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for f in nwb_files:
        all_files.append((subj, os.path.join(subj_dir, f)))
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The AI's CONVERSION_NOTES document finding 28 subjects and 174 NWB files under the data directory. The approach of iterating over `sub-*` directories is consistent with the NWB/DANDI file layout.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject ID from the directory name (e.g., `sub-440956`), not from the NWB file's `nwb.subject.subject_id`. The directory name string (e.g., `'sub-440956'`) is used as the subject identifier throughout.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for f in nwb_files:
        all_files.append((subj, os.path.join(subj_dir, f)))
```

```python
result = process_session(fpath, subj, ...)
```

```python
return {
    ...
    'subject_id': subject_id,  # this is the directory name like 'sub-440956'
    ...
}
```

iii. The CONVERSION_NOTES indicate 28 subjects were found, matching the paper. The AI uses directory names as subject IDs rather than reading `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. Files are discovered by listing `*.nwb` files in each subject directory, sorted alphabetically. The session ID is read from `nwb.identifier`.

ii.
```python
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

```python
session_id = nwb.identifier
```

iii. The AI found 174 NWB files total, with one session (0 good units) being skipped, leaving 173 sessions. This matches the reference papers.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB trials table. Go cue times are extracted from `BehavioralEvents/go_start_times`. The AI indexes into the trials table using `valid_trial_indices` (those passing the trial filter).

ii.
```python
trials = nwb.trials
n_trials_total = len(trials)
trial_instructions = trials['trial_instruction'][:]
outcomes = trials['outcome'][:]
early_licks = trials['early_lick'][:]
```

```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
go_cues_valid = go_start_times[valid_trial_indices]
```

iii. The AI does not verify that the number of go cue events matches the number of trials (the reference code asserts this). Trials are indexed by their position in the trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out trials where `auto_water == 1` or `free_water == 1`. It does NOT use `obs_intervals` to exclude trials without neural coverage. This results in 1,061 trials with all-zero neural data being included.

ii.
```python
auto_water = np.array(trials['auto_water'][:])
free_water = np.array(trials['free_water'][:])
trial_mask = (auto_water == 0) & (free_water == 0)
valid_trial_indices = np.where(trial_mask)[0]
```

iii. The CONVERSION_NOTES document the decision: "For decoder: keep all except auto_water/free_water, since early_lick and outcome are decoder outputs." The notes also acknowledge 1,061 zero-neural trials but treat them as acceptable, stating "the decoder should learn that zero neural activity provides no information." The reference instead uses `obs_intervals` to exclude trials without spike data and does not filter `auto_water`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']`, reading individual spike time arrays per good unit. Go cue times from `BehavioralEvents/go_start_times` determine the bin edges.

ii.
```python
for i in good_indices:
    st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
    spike_times_list.append(st)
```

iii. The AI correctly identifies spike_times as the source of neural data and uses go cue times for alignment.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins using `np.histogram` to get spike counts, then divided by `BIN_WIDTH` (0.05 s) to convert to Hz. This is done per trial, per neuron in nested loops.

ii.
```python
for t_idx in range(n_trials):
    gc = go_cue_times[t_idx]
    edges = BIN_EDGES + gc
    fr_matrix = np.zeros((n_neurons, N_BINS), dtype=np.float32)
    for n_idx in range(n_neurons):
        st = spike_times_list[n_idx]
        i_start = np.searchsorted(st, t_start, side='left')
        i_end = np.searchsorted(st, t_end, side='left')
        spikes_in_window = st[i_start:i_end]
        if len(spikes_in_window) > 0:
            counts = np.histogram(spikes_in_window, bins=edges)[0]
            fr_matrix[n_idx, :] = counts / BIN_WIDTH
    trial_data.append(fr_matrix)
```

iii. The approach is functionally correct (spike counts / bin width = Hz), though much less efficient than the reference's vectorized approach which processes all trials for one unit at once.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with 0 good units are skipped.

ii.
```python
classification = np.array(units['classification'][:])
good_mask = classification == 'good'
n_good = int(np.sum(good_mask))
if n_good == 0:
    print(f"  Skipping {session_id}: 0 good units")
    io.close()
    return None
```

iii. The CONVERSION_NOTES confirm: "Neuron filtering: Use classification=='good' from NWB (matches reference classifier QC)." This matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as `BIN_EDGES + gc` where `gc` is the go cue time for each trial, so spikes are binned relative to the go cue onset. The constant `BIN_EDGES` spans -2.5 to +1.5 seconds.

ii.
```python
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
```

```python
gc = go_cue_times[t_idx]
edges = BIN_EDGES + gc
```

iii. Alignment to the go cue is correct and matches the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin width is 50 ms, with 80 non-overlapping bins spanning [-2.5, 1.5] seconds relative to the go cue. No rebinning is applied; spikes are directly binned at 50 ms.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins
TIME_BEFORE = 2.5
TIME_AFTER = 1.5
N_BINS = int((TIME_BEFORE + TIME_AFTER) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
```

iii. The CONVERSION_NOTES state: "Task requires 50ms bins; use non-overlapping 50ms bins." This matches the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI does NOT derive tone onset from any raw data variable. Instead, it uses a hardcoded constant `TONE_OFFSET = -1.85` (tone onset is assumed to always be 1.85 s before the go cue, based on sample epoch 0.65s + delay epoch 1.2s).

ii.
```python
TONE_OFFSET = -1.85  # tone onset relative to go cue (sample=0.65s + delay=1.2s)
```

```python
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)
```

iii. The CONVERSION_NOTES state: "Tone onset = go_cue - 1.85s" and the mapping table lists "Continuous ramp: bin_center + 1.85". The AI assumed a fixed delay between tone and go cue, not accounting for early-lick trials where the sample epoch is replayed and the actual last tone onset differs from the nominal timing.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Time from tone is computed as `BIN_CENTERS - TONE_OFFSET`, which equals `BIN_CENTERS + 1.85`. This produces the SAME values for every trial in every session — a constant array.

ii.
```python
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)
```

```python
trial_inputs = []
for t_idx in range(n_valid):
    input_data = np.stack([time_from_tone, photostim_inputs[t_idx]], axis=0)
    trial_inputs.append(input_data)
```

iii. The reference computes `CENTERS[None, :] + (go - tone)[:, None]` where `tone` is the last `sample_start_times` event before each trial's go cue. For trials with early licks (which replay the sample epoch), the go-to-tone interval varies, so the reference produces different values per trial. The AI's constant value is incorrect for early-lick trials.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same `BIN_CENTERS` array is used for both the time-from-tone computation and the neural data binning, so alignment is inherent.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. Since the same bin grid is used, alignment with neural data is correct.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents` time series — these are session-level absolute timestamps of all photostim events.

ii.
```python
if 'photostim_start_times' in be.time_series:
    photostim_starts = be.time_series['photostim_start_times'].timestamps[:]
    photostim_stops = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The reference uses per-trial `photostim_onset` and `photostim_duration` columns from the trials table (with `'N/A'` for non-stimulated trials). The AI instead uses session-level event timestamps and matches them to trials by time overlap. The CONVERSION_NOTES observe that 167/173 sessions have photostim events, noting this includes "masking flash" events for non-transgenic mice.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI loops over ALL photostim events in the session and checks if they overlap with the trial time window. Bins where `BIN_CENTERS` fall between the start and stop of any overlapping event are set to 1.

ii.
```python
for t_idx in range(n_trials):
    gc = go_cue_times[t_idx]
    photostim_binary = np.zeros(N_BINS, dtype=np.float32)
    trial_start = gc - TIME_BEFORE
    trial_end = gc + TIME_AFTER
    for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
        if ps_stop < trial_start or ps_start > trial_end:
            continue
        ps_start_rel = ps_start - gc
        ps_stop_rel = ps_stop - gc
        active = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_stop_rel)
        photostim_binary[active] = 1.0
    results.append(photostim_binary)
```

iii. The approach should produce correct results for genuine photostim trials but may incorrectly mark masking flash events as photostim for non-transgenic mice, since `photostim_start_times` in BehavioralEvents includes both actual photostim and masking flashes. The reference avoids this by using per-trial `photostim_onset` which is `'N/A'` for non-stim trials.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The same `BIN_CENTERS` array defines the time grid for both neural binning and photostim labeling, so alignment is inherent.

ii.
```python
active = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_stop_rel)
```

iii. Alignment is correct since the same go-cue-relative time grid is used.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) columns of the trials table.

ii.
```python
def get_choice(trial_instruction, outcome):
    if outcome == 'ignore':
        return 2  # no_lick
    elif outcome == 'hit':
        return 0 if trial_instruction == 'left' else 1
    elif outcome == 'miss':
        return 1 if trial_instruction == 'left' else 0
    return 2
```

iii. The logic is correct: hit means the animal licked the instructed side, miss means it licked the opposite side, ignore means no lick. This matches the reference.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0=left, 1=right, 2=no_lick and repeated across all 80 time bins as a per-trial constant.

ii.
```python
choice_val = get_choice(instruction, outcome)
output_combined[0, :] = choice_val
```

iii. Encoding matches the reference approach.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table (strings 'ignore', 'miss', 'hit').

ii.
```python
outcomes = trials['outcome'][:]
outcome_val = outcome_map.get(outcome, 0)
```

iii. Straightforward mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is mapped to integers (0=ignore, 1=miss, 2=hit) and repeated across all 80 time bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
output_combined[1, :] = outcome_val
```

iii. Matches the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table (strings 'no early' and 'early').

ii.
```python
early_licks = trials['early_lick'][:]
early_val = early_map.get(early_licks[trial_idx], 0)
```

iii. Direct from trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers (0=no, 1=yes) and repeated across all 80 time bins.

ii.
```python
early_map = {'no early': 0, 'early': 1}
early_val = early_map.get(early_licks[trial_idx], 0)
output_combined[2, :] = early_val
```

iii. Matches the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains columns for x, y, and likelihood, along with timestamps.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_timestamps_all = tongue_ts_obj.timestamps[:]
tongue_y_all = tongue_data_all[:, 1]
tongue_lik_all = tongue_data_all[:, 2]
```

iii. Same data source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood >= 0.9 (the AI's threshold) are considered visible. Session-level 40th and 60th percentiles are computed over the raw visible y-values (not binned means). Per-trial, frames are assigned to bins, and the mean visible y per bin is compared against the percentile thresholds.

ii.
```python
DLC_LIKELIHOOD_THRESH = 0.9

visible_mask = tongue_lik_all >= DLC_LIKELIHOOD_THRESH
if np.sum(visible_mask) >= 10:
    visible_y = tongue_y_all[visible_mask]
    p40 = float(np.percentile(visible_y, 40))
    p60 = float(np.percentile(visible_y, 60))
```

```python
if np.any(visible):
    mean_y = np.mean(y_bin[visible])
    if mean_y < p40:
        tongue_y_binned[b] = 0
    elif mean_y <= p60:
        tongue_y_binned[b] = 1
    else:
        tongue_y_binned[b] = 2
```

iii. Two differences from the reference: (1) The likelihood threshold is 0.9 vs the reference's 0.5. (2) Percentiles are computed over raw visible frames rather than over 50ms bin means. The reference first bins the whole session into 50ms bins, computes the mean y per bin, then takes percentiles of those bin means. This can produce different class edges.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three visible classes (0: below 40th percentile, 1: between 40th and 60th inclusive, 2: above 60th) plus class 3 for bins with no visible frames.

ii.
```python
if mean_y < p40:
    tongue_y_binned[b] = 0
elif mean_y <= p60:
    tongue_y_binned[b] = 1
else:
    tongue_y_binned[b] = 2
```

iii. The boundary handling differs slightly from the reference's `np.digitize`: at exactly `p60`, the AI assigns class 1 while the reference assigns class 2. This is a minor difference.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps and go cue times are on the same absolute session clock. For each trial, frames within the trial window are found by searchsorted, assigned to bins using digitize, and averaged per bin.

ii.
```python
t_start = gc + BIN_EDGES[0]
t_end = gc + BIN_EDGES[-1]
idx_start = np.searchsorted(tongue_ts, t_start, side='left')
idx_end = np.searchsorted(tongue_ts, t_end, side='right')
t_rel = t_slice - gc
bin_assignments = np.digitize(t_rel, BIN_EDGES) - 1
```

iii. Alignment is correct — the same go-cue-relative bin grid is used for both neural and tongue data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with 0 good units are skipped. (2) Trials where `auto_water == 1` or `free_water == 1` are excluded. (3) Tongue frames with low likelihood are excluded; bins with no visible frames get class 3. However, the AI does NOT handle the case of trials without neural coverage (from `obs_intervals`), resulting in 1,061 all-zero neural trials.

ii.
```python
if n_good == 0:
    print(f"  Skipping {session_id}: 0 good units")
    return None

trial_mask = (auto_water == 0) & (free_water == 0)
```

iii. The CONVERSION_NOTES acknowledge the zero-neural trials but treat them as acceptable, rather than excluding them via `obs_intervals` as the reference does.

## 10-a. What are the most time-consuming steps of the code?

i. The firing rate computation is the most expensive, with nested loops over trials and neurons. The AI reports ~3.4s per session. The overall conversion takes about 582 seconds for 173 sessions.

ii.
```python
for t_idx in range(n_trials):
    ...
    for n_idx in range(n_neurons):
        ...
        counts = np.histogram(spikes_in_window, bins=edges)[0]
```

iii. The CONVERSION_NOTES estimate ~4s/session initially, with ~12 minutes total.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The firing rate computation has a double loop (trials x neurons) that could have been partially vectorized. The reference vectorizes across all trials for each neuron by flattening the edge array. The tongue y computation also loops over trials and bins.

ii.
```python
for t_idx in range(n_trials):
    for n_idx in range(n_neurons):
        counts = np.histogram(spikes_in_window, bins=edges)[0]
```

```python
for t_idx in range(n_trials):
    for b in range(N_BINS):
        in_bin = bin_assignments == b
```

iii. The double loop in firing rate computation is a significant inefficiency compared to the reference's approach.

## 10-c. What processing does the code repeat multiple times?

i. Nothing is obviously recomputed. Bin edges and centers are computed once as module-level constants.

ii. N/A

iii. The code does a single pass over files.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The elaborate `get_high_level_region` function with ~100 lines of keyword matching produces high-level brain region categories, but the brain region labels are only used as metadata (index into `brain_regions`). The reference uses a simple one-line function. Additionally, the AI reads `electrode_group` and parses JSON location data for the region mapping, which is unnecessary complexity.

ii.
```python
def get_high_level_region(anno_name, electrode_group_target):
    # ~100 lines of keyword matching
    ...
```

iii. The region mapping is more elaborate than needed but does not produce incorrect results per se — it just adds complexity.
