# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all data by globbing every NWB file under `data/sub-*/sub-*.nwb`, then calling `process_session()` once per file with `h5py`. Within each session file it reads trial-table fields, go-cue times, sample-start times, unit tables, spike times, observation intervals, and tongue-tracking data.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))

for nwb_path in nwb_files:
    result = process_session(nwb_path)
```

```python
f = h5py.File(nwb_path, 'r')
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
sample_start_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
spike_times_data = f['units/spike_times'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
```

iii. `CONVERSION_NOTES.md` says the source is 174 NWB files, one per recording session, and explicitly lists `/units/spike_times`, `/intervals/trials`, behavioral event timestamps, and tongue-tracking datasets as the loading sources. The trajectory shows the agent chose NWB loading after inspecting one file’s structure.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the NWB filename prefix. After processing all kept sessions, the agent builds a sorted unique subject list and a `subject_idx` array mapping each session to its subject.

ii.
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
```

```python
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The notes state that the dataset consists of many sessions across multiple subjects and that the subject ID is taken from the `sub-...` path component.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions that fail filtering return `None` from `process_session()` and are skipped; all other sessions become one element in each top-level list (`neural`, `input`, `output`, `brain_region_idx`).

ii.
```python
def process_session(nwb_path, verbose=True):
    ...
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'subject_id': subject_id,
        'unit_regions': unit_regions,
        'session_file': basename,
    }
```

```python
for nwb_path in nwb_files:
    result = process_session(nwb_path)
    if result is None:
        n_skipped += 1
        continue
    all_neural.append(result['neural'])
```

iii. `CONVERSION_NOTES.md` says each NWB file represents one recording session. The trajectory shows the agent explicitly decided to use one file = one session.

## 1-d. How are the data split into trials?

i. Trials are taken from rows of the NWB trial table. After computing a boolean `trial_mask`, the kept row indices become `trial_indices`, and the code loops over those indices to build one neural/input/output sample per trial.

ii.
```python
n_trials = len(f['intervals/trials/id'][:])
...
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
trial_indices = np.where(trial_mask)[0]
```

```python
for ti in trial_indices:
    gc = go_cue_times[ti]
    ...
    neural_trials.append(fr_matrix)
    input_trials.append(input_trial)
    ...
    output_trials.append(out)
```

iii. The notes say trials come from `/intervals/trials/` and that only trials within the neural recording window are kept. The trajectory shows the agent used go-cue-centered trial extraction from the NWB trial rows.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps trials only if `auto_water == 0`, `free_water == 0`, and a session-wide `recording_valid` mask says the entire go-cue-aligned window falls within the intersection of all selected units’ observation windows. It does not exclude early-lick, ignore, or photostimulation trials at the trial stage. Session-level behavioral QC is applied separately using control-trial performance and counts of correct left/right trials.

ii.
```python
is_control = ((early_lick == 'no early') &
              (auto_water == 0) &
              (free_water == 0) &
              (ps_onset_str == 'N/A'))
...
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
if performance <= MIN_PERFORMANCE:
    return None
if control_hits_left < MIN_CORRECT_PER_DIRECTION or control_hits_right < MIN_CORRECT_PER_DIRECTION:
    return None
```

```python
recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
```

iii. `CONVERSION_NOTES.md` says the agent intentionally kept early-lick, ignore, and photostim trials because the decoder outputs/inputs require them, while still using paper-style session criteria of `>65%` control-trial performance and at least `50` correct left and right trials. Trajectory step 18 shows the agent explicitly overrode the reference trial mask for this reason.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw spike times plus go-cue timestamps, after filtering units by `classification` and region annotation. Observation intervals are also read to decide which trials are considered to have valid neural coverage.

ii.
```python
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
anno_names = np.array([a.decode() if isinstance(a, bytes) else a
                      for a in f['units/anno_name'][:]])
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
obs_intervals = f['units/obs_intervals'][:]
obs_intervals_index = f['units/obs_intervals_index'][:]
```

iii. The notes explicitly identify `/units/spike_times` and `/units/spike_times_index` as the spike source and `/units/obs_intervals` as the recording-validity source. The trajectory also notes that the reference preprocessing starts from spike times aligned to go cue.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each kept unit, the agent subtracts the trial’s go-cue time from all spike times, clips to the `[-2.5, 1.5)` window, bins spikes into non-overlapping 50 ms bins, and divides counts by bin width to yield firing rates in Hz.

ii.
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    return counts.astype(np.float64) / bin_width
```

iii. `CONVERSION_NOTES.md` says the agent chose 50 ms non-overlapping bins because the decoder task explicitly asked for 50 ms bins, while acknowledging the reference code used a 100 ms bandwidth with 50 ms stride.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `classification == 'good'` and their `anno_name` maps into one of the agent’s 14 high-level brain-region categories. Sessions with zero kept units are dropped.

ii.
```python
good_mask = classification == 'good'
...
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)
```

```python
if len(unit_indices) == 0:
    return None
```

iii. The notes say the agent matched the paper’s classifier-based QC by using the NWB `classification='good'` field and dropping unmappable/empty anatomical labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to go-cue onset. Spike times are converted to `rel_times = spike_times - go_cue_time`, and only spikes in the fixed `[-2.5, 1.5)` window are binned.

ii.
```python
gc = go_cue_times[ti]
fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

```python
rel_times = spike_times - go_cue_time
mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
```

iii. Both the notes and the trajectory state that the decoder format is aligned to go-cue onset, matching the task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins. There is no later rebinning or smoothing; the firing rates are the direct per-bin spike counts divided by 0.05 s.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

```python
return counts.astype(np.float64) / bin_width
```

iii. The notes explicitly call this out as a deliberate divergence from the paper’s reference bandwidth/stride, justified by the decoder-task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` and `go_start_times`. The code finds the most recent sample-start timestamp before the trial’s go cue and uses that as tone onset.

ii.
```python
sample_start_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

```python
def get_tone_onset_relative_to_go(sample_start_times, go_cue_time):
    before = sample_start_times[sample_start_times < go_cue_time]
    if len(before) > 0:
        return before[-1] - go_cue_time
    return None
```

iii. `CONVERSION_NOTES.md` says the agent treated tone onset as the first tone of the successful sample epoch and used the last `sample_start` before go cue to handle early-lick replays.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The agent computes a single tone-onset time relative to go cue per trial, falls back to `-1.85` seconds if none is found, and converts every time bin center into “seconds since tone onset” via `BIN_CENTERS - tone_onset_rel`.

ii.
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. The notes justify this with the typical sample-plus-delay duration (`~1.85 s`) and say replayed sample epochs motivated taking the last sample start before go cue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the exact same 80 go-cue-centered bin centers used for the neural matrix, so its time axis is shared with the neural data.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
time_from_tone = BIN_CENTERS - tone_onset_rel
input_trial = np.stack([time_from_tone, photostim_binary], axis=0)
```

iii. The trajectory plan and the notes both say all input streams were built in the same go-cue-centered bins as the spike rates.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table fields `photostim_onset` and `photostim_duration`, plus `start_time` and `go_start_times` to convert those values into go-cue-relative time.

ii.
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The notes say the agent used per-trial photostimulation timing from the NWB trial table and interpreted onset as relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code initializes a zero vector, parses onset and duration if the trial is not `N/A`, converts the onset from trial-relative to absolute time and then to go-cue-relative time, and marks bins whose centers fall in the photostimulation interval as `1`.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float64)
if ps_onset_str[ti] != 'N/A':
    try:
        ps_onset_val = float(ps_onset_str[ti])
        ps_dur = float(ps_dur_str[ti])
        ps_abs_onset = trial_start_times[ti] + ps_onset_val
        ps_start_rel = ps_abs_onset - gc
        ps_end_rel = ps_start_rel + ps_dur
        photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                           (BIN_CENTERS < ps_end_rel)).astype(np.float64)
    except (ValueError, TypeError):
        pass
```

iii. `CONVERSION_NOTES.md` says the agent wanted a binary time-varying photostimulation input and describes the expected late-delay timing and 0.5 s duration.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is converted into go-cue-relative time and then sampled on the same bin centers used for the neural firing rates, so the photostim input has one value per neural time bin.

ii.
```python
ps_start_rel = ps_abs_onset - gc
ps_end_rel = ps_start_rel + ps_dur
photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                   (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The notes explicitly say the binary photostim series is aligned to the same go-cue-centered bins as the other streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives choice from the raw trial-table field `trial_instruction`, not from an actual behavioral response variable.

ii.
```python
instructions = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in f['intervals/trials/trial_instruction'][:]])
```

```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
```

iii. `CONVERSION_NOTES.md` explicitly says choice is based on `trial_instruction` and treats that field as “which port the animal should lick.”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent maps `left -> 0` and `right -> 1`, then broadcasts that scalar across all 80 time bins of the trial.

ii.
```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
output_choice.append(choice)
...
np.full(N_BINS, output_choice[i], dtype=np.int64)
```

iii. The notes justify this by treating `trial_instruction` as the lick direction variable, with no further inference from lick events or outcome.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial-table `outcome` string for each trial.

ii.
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. The notes explicitly list NWB `outcome` as the source and define the desired category mapping.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps `ignore -> 0`, `miss -> 1`, and everything else (implicitly `hit`) to `2`, then broadcasts the per-trial label across time bins.

ii.
```python
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:  # hit
    outcome = 2
output_outcome.append(outcome)
```

```python
np.full(N_BINS, output_outcome[i], dtype=np.int64)
```

iii. The notes describe exactly this categorical remapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the trial-table `early_lick` string.

ii.
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
```

iii. The notes say early-lick labels come directly from the NWB trial table and are retained because the decoder must predict them.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent maps `'early'` to `1` and everything else to `0`, then broadcasts the result across the trial’s time bins.

ii.
```python
el = 1 if early_lick[ti] == 'early' else 0
output_early_lick.append(el)
```

```python
np.full(N_BINS, output_early_lick[i], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` states the output uses `0 = no early lick, 1 = early lick during sample/delay`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the side-view DeepLabCut tongue-tracking series: `Camera0_side_TongueTracking/data[:, 1]` for y values and the matching timestamps array.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]  # y-position
```

iii. The notes explicitly identify `/acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` as the source and say column 1 is y-position.

## 8-b. How is `output` *Tongue y-position* processed?

i. The agent extracts tongue-tracking samples in the go-cue-aligned trial window, averages y-position within each 50 ms bin, stores `NaN` for bins with no samples, concatenates the kept-trial binned values within a session, computes session percentiles on the non-NaN values, and then converts each trial’s binned y trace to categories.

ii.
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y, go_cue_time, bin_edges):
    tongue_y_binned = np.full(n_bins, np.nan)
    ...
    ts_rel = ts_window - go_cue_time
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])
```

```python
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]
if len(valid_tongue_y) > 0:
    p40 = np.percentile(valid_tongue_y, 40)
    p60 = np.percentile(valid_tongue_y, 60)
```

iii. The notes say the agent intentionally used all tracking data with no confidence filtering. Trajectory steps 57 to 60 show the agent considered confidence-thresholding, then removed it and chose this “use all DLC values” approach.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent uses session-specific 40th and 60th percentile thresholds. For valid bins, `< p40` becomes `0`, `p40 <= y <= p60` becomes `1`, and `> p60` becomes `2`. Bins with `NaN` stay at the default middle category `1`; if a session has no valid values at all, the whole session is assigned middle.

ii.
```python
ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
valid_mask = ~np.isnan(ty)
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
```

```python
else:
    tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]
```

iii. The notes describe the intended percentile thresholds, while the trajectory shows the default-to-middle behavior was retained after the agent investigated missing/low-confidence bins.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue data are aligned to go cue by subtracting each trial’s go-cue time from the tongue timestamps and then binning into the same `BIN_EDGES` as the neural data.

ii.
```python
t_abs_start = go_cue_time + bin_edges[0]
t_abs_end = go_cue_time + bin_edges[-1]
...
ts_rel = ts_window - go_cue_time
```

```python
ty = extract_tongue_y_for_trial(tongue_ts, tongue_y, gc, BIN_EDGES)
```

iii. The notes say tongue position is time-varying and go-cue aligned in the same analysis window as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses several fallbacks instead of raising errors: if no sample start is found, it assumes `tone_onset_rel = -1.85`; if tongue bins have no data they become `NaN` and later default to middle category; if an entire session has no valid tongue data it assigns the whole session to middle; if photostim parsing fails it silently leaves the vector all zeros. Sessions are skipped if there are no responding control trials, no good units, or fewer than two kept trials.

ii.
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
```

```python
photostim_binary = np.zeros(N_BINS, dtype=np.float64)
...
except (ValueError, TypeError):
    pass
```

```python
ty_disc = np.ones(N_BINS, dtype=np.int64)
...
else:
    tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]
```

iii. The notes mention the “typical value” fallback for tone onset and the use of all tongue data without confidence filtering. The trajectory shows the agent explicitly accepted these imputations after seeing skewed tongue distributions.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant costs are the nested per-trial, per-unit firing-rate computation using `np.histogram`, the repeated observation-window scans over all kept units, and the per-trial per-bin tongue-averaging loop. Loading the full spike-time and tongue arrays from each NWB file is also expensive.

ii.
```python
for ti in trial_indices:
    fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

```python
for ui in unit_indices:
    ...
    unit_obs = obs_intervals[oi_start:oi_end]
```

```python
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
    if np.any(bin_mask):
        tongue_y_binned[b] = np.mean(y_window[bin_mask])
```

iii. This is visible directly from the code structure. The agent’s run logs also show full conversion produced a very large output file and long processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial, per-unit spike histogram loop could be batched or partially vectorized; the per-bin tongue loop could be replaced by vectorized bin assignment and grouped means; the loops building `unit_regions`, `unit_spike_times`, `brain_region_idx`, and `region_counts` are also straightforward vectorization candidates.

ii.
```python
for j, st in enumerate(unit_spike_times):
    fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

```python
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

```python
for i in range(len(classification)):
    ...
for ui in unit_indices:
    ...
for regions in all_unit_regions:
    ...
```

iii. These are direct consequences of the code’s explicit Python loops rather than any justification the agent documented.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly re-histograms global spike trains separately for every trial, repeatedly scans observation intervals per unit to compute a session-level valid window, repeatedly decodes NWB string arrays field-by-field, and repeatedly computes bin masks for tongue tracking for every bin of every trial.

ii.
```python
for ti in trial_indices:
    ...
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

```python
for ui in unit_indices:
    ...
```

```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
instructions = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in f['intervals/trials/trial_instruction'][:]])
```

iii. This follows directly from the implementation; the agent did not optimize these repeated steps.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the full tongue tracking matrix even though it only uses the y column; computes and prints extensive dataset-summary and region-count statistics that are not used in the saved dataset; constructs large metadata strings that downstream decoder code does not consume; and keeps per-trial broadcast copies of static labels (`choice`, `outcome`, `early_lick`) at every time bin even though they are trial-constant.

ii.
```python
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]
```

```python
print(f"\n=== Dataset Summary ===")
...
region_counts = {}
for regions in all_unit_regions:
    for r in regions:
        region_counts[r] = region_counts.get(r, 0) + 1
```

```python
np.full(N_BINS, output_choice[i], dtype=np.int64)
np.full(N_BINS, output_outcome[i], dtype=np.int64)
np.full(N_BINS, output_early_lick[i], dtype=np.int64)
```

iii. These choices are evident from the code itself. The notes present some of the summary/statistical processing as sanity checks rather than something needed by the decoder format.
