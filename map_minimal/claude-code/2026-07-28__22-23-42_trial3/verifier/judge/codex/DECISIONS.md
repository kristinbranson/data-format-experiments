# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all data by globbing every NWB file under `data/sub-*/sub-*.nwb`, then opening each file with `h5py`. Within each file it reads the trials table, behavioral event timestamps, unit table, observation intervals, spike times, and side-camera tongue tracking.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))

for nwb_path in nwb_files:
    result = process_session(nwb_path)
```

```python
f = h5py.File(nwb_path, 'r')

n_trials = len(f['intervals/trials/id'][:])
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
spike_times_data = f['units/spike_times'][:]
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
```

iii. In `CONVERSION_NOTES.md`, the agent says the source format is 174 NWB files across 28 subjects and that each NWB file is one recording session. The trajectory summary repeats that design.

## 1-b. How are the data split into subjects (mice)?

i. The agent derives the subject ID from the NWB filename prefix before `_ses-`, strips `sub-`, then builds a sorted unique subject list and a per-session `subject_idx`.

ii.
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
```

```python
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The notes say the data are organized as `data/sub-*/sub-*.nwb`, so the filename path is the agent's basis for mouse identity.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file as one session. `process_session()` returns one session-level block, and sessions that pass filtering are appended once to the top-level `neural`, `input`, and `output` lists.

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
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. `CONVERSION_NOTES.md` explicitly says each NWB file represents one recording session, even when the session contains multiple probe insertions.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trial table. After session-level filtering, the agent builds `trial_indices` from a boolean mask and then loops over those indices to create one neural matrix, one input matrix, and one output matrix per kept trial.

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

iii. The notes say trial data come from `/intervals/trials/` and that trials outside the neural recording window are excluded.

## 1-e. How are trials filtered based on quality controls?

i. The agent uses two layers of filtering. At the session level it keeps only sessions with control-trial performance above 65% and at least 50 correct left plus 50 correct right trials. At the trial level it excludes `auto_water`, `free_water`, and trials outside a conservative neural recording window; it keeps early-lick, ignore, and photostimulation trials.

ii.
```python
is_control = ((early_lick == 'no early') &
              (auto_water == 0) &
              (free_water == 0) &
              (ps_onset_str == 'N/A'))

control_hits_left = np.sum(is_control & (outcomes == 'hit') & (instructions == 'left'))
control_hits_right = np.sum(is_control & (outcomes == 'hit') & (instructions == 'right'))
control_responding = is_control & (outcomes != 'ignore')
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
```

```python
if performance <= MIN_PERFORMANCE:
    return None

if control_hits_left < MIN_CORRECT_PER_DIRECTION or control_hits_right < MIN_CORRECT_PER_DIRECTION:
    return None

trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
```

iii. The notes justify the session thresholds by citing the methods text. They also justify keeping early-lick, ignore, and photostim trials because those variables are required decoder outputs or inputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from raw unit spike times and go-cue times. It also uses `classification`, `anno_name`, and `obs_intervals` to decide which units and trials are valid before building neural matrices.

ii.
```python
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
anno_names = np.array([a.decode() if isinstance(a, bytes) else a
                      for a in f['units/anno_name'][:]])
obs_intervals = f['units/obs_intervals'][:]
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
```

iii. The notes say spike times come from the ragged NWB unit datasets and that observation windows are used to determine whether trials have valid neural data.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each kept unit, the agent subtracts the trial's go-cue time from absolute spike times, counts spikes in 50 ms bins from -2.5 s to +1.5 s, and divides by bin width to convert counts to Hz.

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    return counts.astype(np.float64) / bin_width
```

```python
fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
for j, st in enumerate(unit_spike_times):
    fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The notes say the decoder task requires 50 ms bins, so the agent chose simple non-overlapping bin counts converted to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only units whose NWB `classification` is `'good'`, then drops any unit whose `anno_name` cannot be mapped into one of 14 high-level brain regions. It also drops whole sessions with no remaining units. Separately, it removes trials whose full -2.5 s to +1.5 s window does not lie within the shared recording window of the selected units.

ii.
```python
good_mask = classification == 'good'

unit_regions = []
unit_indices = []
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)
```

```python
min_obs_end = np.inf
max_obs_start = -np.inf
for ui in unit_indices:
    oi_start = 0 if ui == 0 else obs_intervals_index[ui - 1]
    oi_end = obs_intervals_index[ui]
    unit_obs = obs_intervals[oi_start:oi_end]
    if len(unit_obs) > 0:
        min_obs_end = min(min_obs_end, unit_obs[-1, 1])
        max_obs_start = max(max_obs_start, unit_obs[0, 0])

recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
```

iii. The notes justify this as classifier-based QC plus exclusion of units lacking valid region annotations, and they say observation windows are needed because some sessions contain trials outside the neural recording period.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns each trial to go-cue onset. It defines `t = 0` as the trial's go cue, then bins neural activity from 2.5 s before to 1.5 s after that event.

ii.
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

```python
rel_times = spike_times - go_cue_time
counts, _ = np.histogram(rel_times, bins=bin_edges)
```

iii. The notes say temporal alignment is to go cue onset because that is the explicit decoder-task requirement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, yielding 80 bins over the 4 s window. No further temporal smoothing or rebinning is applied after the histogram step.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
return counts.astype(np.float64) / bin_width
```

iii. The notes say the agent deliberately used 50 ms non-overlapping bins because the decoder task overrode the reference preprocessing resolution.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The agent derives this input from `sample_start_times` and `go_start_times`. It looks for the last sample-start timestamp before the trial's go cue.

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

iii. The notes say this is intended to recover the first tone of the successful sample epoch and to handle early-lick replays by choosing the last sample-start before go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the agent finds the last `sample_start_time` before the go cue, converts that to go-cue-relative time, and then computes a continuous time-since-tone value at each 50 ms bin center. If no such sample-start exists, it falls back to `-1.85` s.

ii.
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. The notes justify the last-sample-start rule by citing early-lick replay trials. The hard-coded `-1.85` fallback is only implicit in the code; the agent describes it as a typical sample-plus-delay duration.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on the same go-cue-centered 50 ms bin centers used for the neural firing rates, so every trial gets a length-80 time series aligned to the neural matrix columns.

ii.
```python
time_from_tone = BIN_CENTERS - tone_onset_rel
input_trial = np.stack([time_from_tone, photostim_binary], axis=0)
```

iii. The notes say the input is aligned to go cue because the whole converted dataset is go-cue-centered.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The agent derives photostimulation from the trial table's `photostim_onset` and `photostim_duration`, together with `start_time` and go-cue timestamps to convert the stimulation interval into go-cue-relative time.

ii.
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. In the trajectory, the agent explicitly says it checked that `photostim_onset` is relative to trial start and fixed the code accordingly.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The agent initializes an all-zero 80-bin vector, parses the onset and duration when present, converts onset from trial-relative time to absolute time, subtracts the go cue to get a go-cue-relative interval, and marks bins whose centers fall inside the stimulation interval as 1.

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

iii. The notes justify this as matching the trial-table representation of photoinhibition during late delay, and the trajectory shows the agent fixing an earlier absolute-time bug.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by converting the stimulation interval into go-cue-relative time and then sampling it on the same 50 ms bin centers as the neural data.

ii.
```python
ps_start_rel = ps_abs_onset - gc
ps_end_rel = ps_start_rel + ps_dur
photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                   (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The notes say photostimulation typically occupies late-delay bins before the go cue, which is why the agent aligned it into the same go-cue-centered time base.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives the choice output from the trial-table field `trial_instruction`, not from actual lick events or a behavioral choice variable.

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

iii. `CONVERSION_NOTES.md` explicitly says choice is based on the `trial_instruction` field, described there as “which port the animal should lick.”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent maps `trial_instruction == 'left'` to 0 and everything else to 1, then broadcasts that trial-constant label across all 80 time bins.

ii.
```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
output_choice.append(choice)
```

```python
np.full(N_BINS, output_choice[i], dtype=np.int64)
```

iii. The notes justify the left/right coding by the decoder-task label convention, but the source variable chosen was still `trial_instruction`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The outcome output is derived directly from the NWB trial-table field `outcome`.

ii.
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. The notes explicitly define the mapping from the NWB `outcome` strings to the requested decoder categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent converts outcome strings into integer categories: `ignore -> 0`, `miss -> 1`, and `hit -> 2`. It then broadcasts the per-trial outcome label across all bins.

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

iii. The notes say this mapping was chosen to match the decoder-task specification exactly.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The code does not create a distance-to-reward-zone output. For the implemented `Outcome` output, the alignment is trial-constant broadcasting across the same 80 go-cue-centered bins as the neural data.

ii.
```python
out = np.array([
    np.full(N_BINS, output_choice[i], dtype=np.int64),
    np.full(N_BINS, output_outcome[i], dtype=np.int64),
    np.full(N_BINS, output_early_lick[i], dtype=np.int64),
    tongue_y_discrete_trials[i],
], dtype=np.int64)
```

iii. The notes only discuss `Outcome`, not distance to reward zone. The question text appears mismatched to the implemented task, so the agent's effective alignment rule is the per-bin broadcast shown above.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The early-lick output comes directly from the NWB trial-table field `early_lick`.

ii.
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
```

iii. The notes describe this as the trial-level indicator of whether an early lick occurred during sample or delay.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent maps `early` to 1 and all other values to 0, stores the result per trial, and broadcasts it across all 80 bins.

ii.
```python
el = 1 if early_lick[ti] == 'early' else 0
output_early_lick.append(el)
```

```python
np.full(N_BINS, output_early_lick[i], dtype=np.int64)
```

iii. The notes say early-lick trials were kept specifically because early lick is a requested decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The tongue output is derived from the side-view DeepLabCut tongue-tracking series: the raw tracking timestamps and the second column of the tracking data array, which the agent interprets as y-position.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]  # y-position
```

iii. The notes say the source is `/acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` with columns `[x, y, likelihood]`.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each kept trial, the agent extracts side-camera tongue timestamps that fall within the go-cue-centered window, converts them to go-cue-relative time, and computes the mean y-position within each 50 ms bin. It does not threshold by DeepLabCut likelihood.

ii.
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y,
                                go_cue_time, bin_edges):
    ...
    idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
    ts_window = tongue_timestamps[idx[0]:idx[1]]
    y_window = tongue_y[idx[0]:idx[1]]
    ...
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])
```

iii. The notes say the agent first tried likelihood filtering, found it left too little data, and then switched to “all tracking data used (no confidence filtering).”

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After collecting all per-bin tongue y values from the session, the agent computes the 40th and 60th percentiles of valid values only. Each valid bin is then labeled 0 if below `p40`, 1 if between `p40` and `p60`, and 2 if above `p60`. Empty bins default to the middle category, and sessions with no valid tongue data become all-middle.

ii.
```python
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]

if len(valid_tongue_y) > 0:
    p40 = np.percentile(valid_tongue_y, 40)
    p60 = np.percentile(valid_tongue_y, 60)
```

```python
ty_disc = np.ones(N_BINS, dtype=np.int64)
valid_mask = ~np.isnan(ty)
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
```

iii. The notes say this percentile split is exactly the decoder-task discretization rule, with the practical choice to treat missing bins as the middle category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The agent aligns tongue positions by selecting raw tongue timestamps inside each trial's absolute go-cue-centered window, converting them to times relative to that trial's go cue, and binning them on the same 50 ms bins used by the neural data.

ii.
```python
t_abs_start = go_cue_time + bin_edges[0]
t_abs_end = go_cue_time + bin_edges[-1]
idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
...
ts_rel = ts_window - go_cue_time
```

```python
ty = extract_tongue_y_for_trial(tongue_ts, tongue_y, gc, BIN_EDGES)
tongue_y_all_trials.append(ty)
```

iii. The notes describe tongue y-position as a time-varying output aligned to the same go-cue-centered decoder window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or malformed data mostly with defaults or exclusion. Missing valid region names drop units. Sessions with no good units, no responding control trials, or fewer than two valid trials are skipped. Trials outside a shared observation window are dropped. Missing tone-onset recovery falls back to `-1.85` s. Photostim parse failures silently leave an all-zero stim vector. Missing tongue bins become `NaN` and later default to the middle tongue category; sessions with no valid tongue data become all-middle.

ii.
```python
if region is not None:
    unit_regions.append(region)
...
if len(unit_indices) == 0:
    return None
```

```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
```

```python
except (ValueError, TypeError):
    pass
```

```python
if len(valid_tongue_y) > 0:
    ...
else:
    tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]
```

iii. The trajectory shows the agent explicitly revising its tongue handling and photostim parsing. The remaining fallbacks are justified only implicitly by “typical value,” “leave zeros,” or “default middle category” behavior in code.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the nested per-session, per-trial, per-unit histogramming of spike times into firing rates; the scan through observation intervals for all selected units; and the per-trial tongue extraction loop that iterates over all 80 bins.

ii.
```python
for ti in trial_indices:
    gc = go_cue_times[ti]
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

iii. This is not stated in the notes, but it follows directly from the structure of the code the agent wrote.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-unit firing-rate loop, the per-bin tongue loop, the repeated per-trial search through `sample_start_times`, and some repeated list-to-index conversions for brain regions and subjects.

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
before = sample_start_times[sample_start_times < go_cue_time]
```

iii. This is an inference from the implementation. The code is straightforward but mostly scalar-loop based.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans the full `sample_start_times` array once per trial, repeatedly parses photostim strings per trial, makes one pass to collect continuous tongue traces and a second pass to discretize them, and repeatedly computes list-index mappings for regions and subjects.

ii.
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
```

```python
if ps_onset_str[ti] != 'N/A':
    try:
        ps_onset_val = float(ps_onset_str[ti])
        ps_dur = float(ps_dur_str[ti])
```

```python
tongue_y_all_trials.append(ty)
...
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
...
for ty in tongue_y_all_trials:
    ty_disc = np.ones(N_BINS, dtype=np.int64)
```

iii. This is again a direct description of repeated work visible in the code rather than a separate statement in the notes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes continuous tongue y-position arrays only to discard them after discretization, loads all three tongue-tracking columns but ultimately uses only y, builds verbose metadata and summary statistics that the decoder does not consume, and computes conservative observation-window limits even though only the final boolean `recording_valid` mask is used downstream.

ii.
```python
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]
...
ty = extract_tongue_y_for_trial(tongue_ts, tongue_y, gc, BIN_EDGES)
tongue_y_all_trials.append(ty)
```

```python
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
...
tongue_y_discrete_trials.append(ty_disc)
```

```python
region_counts = {}
for regions in all_unit_regions:
    for r in regions:
        region_counts[r] = region_counts.get(r, 0) + 1
```

iii. The notes emphasize documentation and sanity checks, but these intermediate continuous arrays and summary calculations are not consumed by the final decoder dataset itself.
