# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by listing subject directories (`sub-*`) under `/app/data`, then iterating over sorted `.nwb` files within each. Each file is opened with `pynwb.NWBHDF5IO` and processed by `process_session()`. This is essentially the same approach as the reference, which uses `glob.glob('sub-*/*.nwb')`.

ii.
```python
def get_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for fname in files:
            nwb_files.append((subj, os.path.join(subj_dir, fname)))
    return nwb_files
```

```python
io = NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The AI documented in CONVERSION_NOTES.md that the dataset contains 28 subjects and 174 NWB files, matching the dandiset metadata. The approach of iterating over sorted subject directories and NWB files within them is standard for this dataset.

## 1-b. How are the data split into subjects?

i. The AI reads the subject ID from the directory name (e.g., `sub-440956`) rather than from `nwb.subject.subject_id`. Subjects are tracked in insertion order as sessions are processed, and `subject_idx` maps each session to its subject.

ii.
```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    ...
    result = process_session(nwb_path, subject_id)
    ...
    if subject_id not in subjects_seen:
        subjects_seen[subject_id] = len(subjects_seen)
        subjects_list.append(subject_id)
    subj_idx = subjects_seen[subject_id]
```

iii. The AI uses the directory name as the subject identifier, which gives IDs like `sub-440956`. The reference uses `nwb.subject.subject_id` which gives just `440956`. Both yield 28 unique subjects. The subject IDs differ in format but are functionally equivalent.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session, same as the reference. The AI identifies sessions by the NWB filename rather than `nwb.identifier`.

ii.
```python
return {
    ...
    'session_name': os.path.basename(nwb_path),
}
```

iii. The AI documented that 174 NWB files were found, with 173 having good units (one excluded). This matches the reference's 173 sessions.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table, same as the reference. The AI reads trial data using `nwb.trials['outcome'][:]` and related fields. Go cue times come from `BehavioralEvents/go_start_times`.

ii.
```python
trials = nwb.trials
outcomes = trials['outcome'][:]
instructions = trials['trial_instruction'][:]
early_lick = trials['early_lick'][:]
...
go_start_times = events.time_series['go_start_times'].timestamps[:]
```

iii. The AI does not explicitly assert that the number of go cues matches the number of trials, unlike the reference which has `assert len(go) == len(trials)`. However, the trials are indexed by trial index, so the mapping is implicit.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by excluding `auto_water` and `free_water` trials, and also excludes trials outside `obs_intervals`. The reference excludes only `free_water` and trials outside `obs_intervals` — it does NOT exclude `auto_water`.

ii.
```python
trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
```

iii. The AI justified excluding `auto_water` and `free_water` as they "confound behavior, not decoder variables". The reference only excludes `free_water`. The `auto_water` exclusion is an additional filter not present in the reference.

Additionally, the AI's `obs_intervals` matching logic uses an approximate distance-based approach (`min_diffs < 1.0`) rather than exact matching of trial start times to observation interval starts, which is less precise than the reference's `np.isin` approach.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Same as the reference: `units/spike_times` for sorted spike times, and `go_start_times` for temporal alignment. Only units with `classification == 'good'` are used.

ii.
```python
spike_times_per_unit = []
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

iii. The AI correctly identified `spike_times` as the source of neural data and `classification == 'good'` as the QC filter.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms non-overlapping bins spanning -2.5 s to 1.5 s relative to the go cue (80 bins), then converted to firing rates in Hz. This matches the reference approach.

ii.
```python
def compute_firing_rates(spike_times_list, go_cue_time, t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end
    for i, st in enumerate(spike_times_list):
        mask = (st >= abs_start) & (st < abs_end)
        spikes_in_window = st[mask]
        bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        np.add.at(fr[i], bin_indices, 1.0)
    fr /= bin_size
    return fr
```

iii. The AI noted this matches the 50 ms bins specified in the instructions. The implementation is per-trial (called once per trial) rather than vectorized across trials like the reference, which computes all trials at once using `searchsorted` on a flattened edge array.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons by `classification == 'good'`, same as the reference. However, the AI also drops neurons whose `anno_name` cannot be mapped to one of 14 coarse brain regions. The reference keeps all `classification == 'good'` neurons regardless of annotation mapping.

ii.
```python
good_mask = np.array([c == 'good' for c in classification])
...
anno_names = nwb.units['anno_name'][:][good_mask]
region_indices = []
valid_neuron_mask = np.ones(n_good, dtype=bool)
for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    if region is None:
        valid_neuron_mask[i] = False
```

iii. The AI documented that all good units have `anno_name`, but the mapping function `map_anno_to_region` returns `None` for annotations that don't match any of the 14 coarse regions. This means some neurons with valid QC are dropped because of incomplete region mapping, resulting in fewer neurons than the reference (which retains all good units with their fine-grained region labels).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Same as the reference: neural data is aligned to the go cue onset. The AI computes absolute bin edges from `go_cue_time + t_start` and bins spikes accordingly. This is functionally identical to the reference's approach.

ii.
```python
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
...
bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
```

iii. The go cue is the alignment event specified in the instructions. Both AI and reference use the same approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins per trial (-2.5 s to 1.5 s relative to go cue). Same as the reference. No rebinning — spike times are directly binned at 50 ms.

ii.
```python
BIN_SIZE_S = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80
```

iii. This matches the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Same as the reference: from `sample_start_times` (the tone onset events) and the go cue time. The AI finds the last sample onset before the go cue.

ii.
```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
...
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
```

iii. The AI correctly identifies that early lick replays can cause multiple sample onsets per trial and takes the last one before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Same as the reference: compute bin centers relative to go cue, then subtract the tone onset time relative to go cue. The result is time since tone onset at each bin center.

ii.
```python
def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    tone_rel = tone_onset_time - go_cue_time
    time_from_tone = bin_centers - tone_rel
    return time_from_tone.astype(np.float32)
```

iii. The math is equivalent to the reference's `CENTERS + (go - tone)`. Both produce time from tone onset at each bin center.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Same as the reference: both use the same bin centers relative to the go cue, so neural data and this input are inherently aligned.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```

iii. The bin centers define both the neural binning grid and the time-from-tone values, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents`, which are absolute event timestamps. The reference uses `photostim_onset` and `photostim_duration` from the trials table, which are per-trial strings relative to trial start.

ii.
```python
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The AI chose the event-based timestamps rather than the trial-table fields. Both sources encode the same underlying photostimulation events, but the AI's approach requires matching events to trials by time overlap, while the reference's approach directly indexes per-trial values.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary time series (1 if photostim on, 0 if off) by checking which event time intervals overlap the trial window, then checking if each bin center falls within any photostim interval. The reference does the same but uses per-trial onset/duration from the trials table.

ii.
```python
def compute_photostim_timeseries(go_cue_time, photostim_starts, photostim_stops,
                                  t_start, t_end, bin_size):
    ...
    for start, stop in zip(photostim_starts, photostim_stops):
        mask = (abs_centers >= start) & (abs_centers <= stop)
        stim[mask] = 1.0
    return stim
```

iii. The AI uses `<=` for the stop boundary (`abs_centers <= stop`), while the reference uses `<` (`CENTERS < stim_off`). This is a minor difference — for 50 ms bins it's unlikely a bin center would fall exactly on the stop time, but `<` is more mathematically correct for a half-open interval.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The AI converts bin centers to absolute time, then compares against absolute photostim event times. The reference converts photostim onset to go-cue-relative time and compares against relative bin centers. Both achieve the same alignment.

ii.
```python
abs_centers = bin_centers + go_cue_time
...
mask = (abs_centers >= start) & (abs_centers <= stop)
```

iii. Both approaches are equivalent — the AI works in absolute time while the reference works in go-cue-relative time.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice ONLY from `trial_instruction` (left/right), mapping the instructed direction directly to the choice. The reference derives choice from BOTH `trial_instruction` AND `outcome`, using the logic that a hit means the animal licked the instructed side, a miss means it licked the other side, and ignore means no lick.

ii.
```python
# Output 0: choice (left=0, right=1)
choice = 0 if instructions[ti] == 'left' else 1
```

iii. The AI's approach maps the instructed direction to choice, which is incorrect. The instructions ask for "Lick direction choice", which is the animal's actual lick direction, not the instruction. On miss trials, the animal licks the opposite direction from the instruction. The AI also has only 2 classes (left, right) and omits the "no lick" class for ignore trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI directly maps `trial_instruction` to 0 (left) or 1 (right) and replicates across time bins. The reference derives actual lick direction from instruction x outcome and includes a third class (2 = no lick).

ii.
```python
choice = 0 if instructions[ti] == 'left' else 1
...
full_output[0, :] = out_dict['choice']
```

The AI's `output_values[0]` is `['left', 'right']` — only 2 values, missing `'no lick'`.

iii. The AI treats choice as the trial instruction rather than the animal's actual behavioral choice. This is incorrect: on miss trials, the animal licks the opposite direction, and on ignore trials, the animal doesn't lick at all. The reference correctly handles these cases.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Same as the reference: directly from the `outcome` column of the trials table, which contains `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome_str = outcomes[ti]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
```

iii. The mapping is identical to the reference: ignore=0, miss=1, hit=2.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Same as the reference: map the three string values to integers 0/1/2 and replicate across time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
...
full_output[1, :] = out_dict['outcome']
```

iii. Identical to reference. The `.get(outcome_str, 0)` fallback to 0 is a minor difference — the reference would raise a KeyError on unknown outcomes, which is arguably better for catching data issues.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Same as the reference: from the `early_lick` column of the trials table.

ii.
```python
early_lick = trials['early_lick'][:]
...
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. Identical to reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Same as the reference: map `'no early'` to 0 and `'early'` to 1, replicate across time bins.

ii.
```python
early_val = 0 if early_lick[ti] == 'no early' else 1
...
full_output[2, :] = out_dict['early_lick']
```

iii. Functionally identical to the reference's `EARLY_CODE = {'no early': 0, 'early': 1}`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Same as the reference: from `Camera0_side_TongueTracking` in `BehavioralTimeSeries`, using columns for y-position and likelihood.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. Identical source data as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI filters frames with likelihood < 0.9 (sets to NaN), bins tongue y into 50 ms bins within each trial window, then computes the mean per bin. The reference uses a likelihood threshold of 0.5 instead. The AI computes session percentiles (40th/60th) over the pooled valid bin means from trial windows. The reference computes percentiles over bin means of the entire session timeline.

ii.
```python
y_in_window[like_in_window < 0.9] = np.nan
```

```python
tongue_y_arr = np.array(tongue_y_session)  # pooled from trial windows
p40 = np.percentile(tongue_y_arr, 40)
p60 = np.percentile(tongue_y_arr, 60)
```

iii. The AI uses a much stricter likelihood threshold (0.9 vs reference's 0.5). The reference notes the likelihood is effectively binary (89% < 0.01, 10.5% >= 0.99), so the practical difference is small. The percentile computation scope differs: AI pools valid bin means from trial windows; reference bins the entire session timeline.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI discretizes into 3 classes: 0 (< 40th percentile), 1 (40th to 60th percentile), 2 (> 60th percentile). Bins with NaN tongue data remain at 0 (the default). The reference has 4 classes, with class 3 = "not visible" for bins with no valid frames.

ii.
```python
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

`output_values[3]` is `['below_40th', '40th_to_60th', 'above_60th']` — only 3 values.

iii. The AI is missing the "not visible" class (class 3). Bins where the tongue is not visible default to class 0, which conflates "below 40th percentile" with "not visible". Since ~75% of bins have no visible tongue (per the reference), this is a significant issue that would corrupt the output distribution.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Same approach as the reference: bin camera frames into the same 50 ms bins aligned to the go cue, using the same time window (-2.5 s to 1.5 s).

ii.
```python
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
...
bin_indices = np.digitize(t_in_window, bin_edges) - 1
```

iii. The alignment is correct — same bin grid as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units are skipped (returns None). Trials outside `obs_intervals` are excluded. Tongue frames with low likelihood are set to NaN. However, the AI does not handle the "not visible" tongue case with a separate class — NaN bins become class 0. Also, neurons with unmappable brain region annotations are dropped, which loses data.

ii.
```python
if n_good == 0:
    io.close()
    return None
...
if n_neurons == 0:
    io.close()
    return None
...
y_in_window[like_in_window < 0.9] = np.nan
```

iii. The AI handles the major cases (no good units, unrecorded trials) but misses the tongue "not visible" class.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's code is significantly slower than the reference (~38 minutes vs ~4 minutes). The main bottleneck is the per-trial firing rate computation, which iterates over trials and neurons individually, rather than vectorizing across trials.

ii.
```python
for ti in trial_indices:
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The AI documented processing time of ~2300s (38 min) for the full dataset. The reference processes in ~247s (4 min). The per-trial loop with per-neuron spike filtering is the primary cause.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the per-trial firing rate computation. The reference vectorizes across trials by flattening all trial bin edges into one array and doing a single `searchsorted` per neuron. The AI calls `compute_firing_rates` once per trial, with an inner loop over neurons.

ii.
```python
for ti in trial_indices:
    fr = compute_firing_rates(spike_times_per_unit, go_cue, ...)
    # Inside compute_firing_rates:
    for i, st in enumerate(spike_times_list):
        mask = (st >= abs_start) & (st < abs_end)
        ...
```

iii. The double loop (trials x neurons) is the main inefficiency. The reference's approach of vectorizing across trials reduces this to a single loop over neurons.

## 10-c. What processing does the code repeat multiple times?

i. The AI reads spike times for each unit once, but applies them per-trial in a loop. The tongue tracking data is read once for the session but processed per-trial. No data is re-read unnecessarily.

ii. N/A

iii. The code does not repeat file I/O, but the per-trial processing is inherently repetitive where it could be batch-computed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes the full brain region mapping (with an extensive rule-based function) for 14 coarse regions, which maps every fine CCF annotation through ~50 string matching rules. The reference simply takes the first component of the annotation name, which is simpler and retains all neurons. The coarse mapping adds complexity and drops some neurons whose annotations don't match any rule.

ii.
```python
def map_anno_to_region(anno):
    # ~140 lines of string matching rules
    ...
    return None  # drops neurons that don't match
```

iii. The brain region mapping is not discarded per se, but the elaborate mapping to 14 coarse regions is unnecessary complexity — the reference just uses fine-grained labels.
