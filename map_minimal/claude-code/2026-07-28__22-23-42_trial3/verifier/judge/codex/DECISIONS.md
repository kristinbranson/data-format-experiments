# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing all NWB files under `data/sub-*/sub-*.nwb`, sorting the list, and processing each file once with `h5py`. Inside each file it reads trial fields from `intervals/trials`, event timestamps from `acquisition/BehavioralEvents`, unit fields from `units`, and tongue tracking from `acquisition/BehavioralTimeSeries`.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
```

```python
f = h5py.File(nwb_path, 'r')
...
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
```

iii. `CONVERSION_NOTES.md` says the source is “174 NWB files across 28 subjects in `data/sub-*/sub-*.nwb`” and that each NWB file is one recording session. No separate justification for using `h5py` instead of `pynwb` was recorded.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the NWB filename/path, not from the NWB subject metadata. For each kept session, the code extracts the `sub-<id>` prefix from the basename, stores that as `subject_id`, then builds `subjects` as the sorted unique IDs and `subject_idx` as per-session indices into that list.

ii. 
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
```

```python
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The notes justify this indirectly by relying on the published directory layout: “174 NWB files across 28 subjects in `data/sub-*/sub-*.nwb`.” No trajectory step shows the agent validating against `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session ordering follows the sorted file list, and session identity is represented by the basename stored in `session_file`, not by `nwb.identifier`.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
```

```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'subject_id': subject_id,
    'unit_regions': unit_regions,
    'session_file': basename,
}
```

iii. `CONVERSION_NOTES.md` states “Each NWB file represents one recording session.” That is the recorded justification for using file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table length (`intervals/trials/id`) and paired with one go-cue timestamp per trial. The code then filters those trial indices with `trial_mask` and iterates one retained trial at a time.

ii. 
```python
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_cue_times) == n_trials, f"Go cue count ({len(go_cue_times)}) != trial count ({n_trials})"
```

```python
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
trial_indices = np.where(trial_mask)[0]
...
for ti in trial_indices:
    gc = go_cue_times[ti]
```

iii. The notes say trial data come from `/intervals/trials/` and go cues from `/acquisition/BehavioralEvents/go_start_times/timestamps`. The assert is the code-level sanity check that these two trial definitions agree.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters. It excludes `auto_water` and `free_water` trials, and it keeps only trials whose entire `[-2.5, +1.5]` window around the go cue falls inside a session-wide recording window derived from the latest start and earliest end across selected units’ `obs_intervals`. It also drops whole sessions for low performance (`>65%` required), too few correct control trials per direction (`>=50`), no responding control trials, or fewer than 2 surviving trials.

ii. 
```python
is_control = ((early_lick == 'no early') &
              (auto_water == 0) &
              (free_water == 0) &
              (ps_onset_str == 'N/A'))
...
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

iii. `CONVERSION_NOTES.md` says the AI intentionally used session filtering from the methods text: “>65% correct on control trials” and “>=50 correct left AND >=50 correct right trials,” and excluded `auto_water`/`free_water` while keeping early-lick, ignore, and photostim trials because the decoder needs those labels. Trajectory step 43 also shows the agent deciding to use recording windows to discard trials outside neural coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` for selected units, together with go-cue timestamps from `BehavioralEvents/go_start_times` to define trial-relative windows.

ii. 
```python
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

```python
for ui in unit_indices:
    start_idx = 0 if ui == 0 else spike_times_index[ui - 1]
    end_idx = spike_times_index[ui]
    unit_spike_times.append(spike_times_data[start_idx:end_idx])
```

iii. The notes explicitly say “Loaded from `/units/spike_times`” and that observation windows from `/units/obs_intervals` are checked to determine valid trials.

## 2-b. How is the `neural` data processed?

i. For each retained trial and each retained unit, the AI subtracts the trial’s go cue from the unit’s absolute spike times, keeps spikes in `[-2.5, +1.5)`, bins them with `np.histogram` into 50 ms bins, and divides counts by bin width to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

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

iii. `CONVERSION_NOTES.md` says “50ms non-overlapping bins” and “Spike counts divided by bin width (0.05s) to get rates in Hz.” Trajectory step 18 records the same plan.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are first filtered by classifier QC with `classification == 'good'`. The AI then further filters those good units to ones whose `anno_name` can be mapped into one of 14 hand-written high-level brain-region categories. Sessions with no such units are dropped.

ii. 
```python
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
good_mask = classification == 'good'
```

```python
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)

if len(unit_indices) == 0:
    return None
```

iii. The notes justify this as “classifier-based QC (`classification='good'` in NWB units table)” plus a 14-category region mapping matching the paper’s high-level grouping. They also state “Units with empty or unmappable annotation names excluded.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is done directly to go-cue onset. The code converts absolute spike times to go-cue-relative times by subtracting `gc`, then bins using `BIN_EDGES` defined relative to the go cue.

ii. 
```python
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

```python
rel_times = spike_times - go_cue_time
counts, _ = np.histogram(rel_times, bins=bin_edges)
```

iii. The notes state “Aligned to go cue onset (t=0)” with the specified `-2.5s` to `+1.5s` window. No additional cross-stream offset correction is described.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms non-overlapping bins across an 80-bin window from `-2.5 s` to `+1.5 s`. There is no additional temporal rebinning or smoothing.

ii. 
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

iii. The AI justified this as following the decoder task even though its notes acknowledge that the reference analysis code used a different smoothing/binning scheme.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from sample-epoch tone onset timestamps in `acquisition/BehavioralEvents/sample_start_times/timestamps` and from go-cue timestamps in `acquisition/BehavioralEvents/go_start_times/timestamps`.

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

iii. The notes and trajectory both justify using the last sample start before the go cue because early licks can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each retained trial, the AI finds the last `sample_start_time` before the go cue and stores it relative to go. If none is found, it falls back to `-1.85` seconds as a typical tone-to-go interval. It then computes time from tone onset at each bin center as `BIN_CENTERS - tone_onset_rel`.

ii. 
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. Trajectory step 24 records the justification: the normal tone-to-go gap is about `1.85 s`, but early-lick trials can replay sample epochs, so the “last sample_start before go” is the meaningful tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 bin centers as the neural data, within the same go-cue-aligned per-trial loop. The only difference is the variable stored at each bin center.

ii. 
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = BIN_CENTERS - tone_onset_rel
input_trial = np.stack([time_from_tone, photostim_binary], axis=0)
```

iii. The trajectory explicitly reasons in go-cue-relative coordinates, then converts each bin center into elapsed time since the trial’s tone onset.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table fields `photostim_onset` and `photostim_duration`, together with `start_time` and the trial’s go-cue time.

ii. 
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
```

iii. The notes say `photostim_onset` is relative to trial start and must be converted to go-cue-relative time. Trajectory steps 80 to 83 show the agent discovering and fixing exactly that issue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI initializes a zero vector for each trial. For stimulated trials it parses the onset and duration strings, converts onset to an absolute time by adding `trial_start_times[ti]`, converts that to go-cue-relative start and end times, and marks bins as `1` when their centers fall within `[start, end)`. If parsing fails, the code silently leaves the trial all zeros.

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

iii. The notes justify the binary time-series representation and say photostimulation occurs during late delay. The trajectory documents the debugging step that established `photostim_onset` as trial-start-relative.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned by converting trial-start-relative onset into go-cue-relative onset and comparing that interval against the same `BIN_CENTERS` used for the neural time axis.

ii. 
```python
ps_abs_onset = trial_start_times[ti] + ps_onset_val
ps_start_rel = ps_abs_onset - gc
ps_end_rel = ps_start_rel + ps_dur
photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                   (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The trajectory explicitly records this fix as the resolution to an alignment bug.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. In the final code, choice is derived only from the `trial_instruction` field. The code does not use `outcome` to infer whether the animal licked the instructed side, the opposite side, or did not lick.

ii. 
```python
instructions = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in f['intervals/trials/trial_instruction'][:]])
...
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
```

iii. `CONVERSION_NOTES.md` states: “Based on `trial_instruction` field (which port the animal should lick).” The earlier trajectory considered deriving choice from `instruction x outcome`, but the final implementation and notes did not keep that plan.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes `left` as `0` and everything else as `1`, then broadcasts that per-trial value across all 80 time bins. It exposes only two allowed values in `output_values`.

ii. 
```python
output_choice.append(choice)
...
out = np.array([
    np.full(N_BINS, output_choice[i], dtype=np.int64),
    ...
], dtype=np.int64)
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'middle', 'high'],
],
```

iii. The recorded justification is just the decoder spec’s `left = 0`, `right = 1` phrasing; no separate rationale was recorded for omitting a no-lick category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from the NWB trials-table `outcome` column.

ii. 
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. The notes say outcome uses the NWB trial table categories directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to `0` for `ignore`, `1` for `miss`, and `2` for `hit`, then repeated across all 80 bins.

ii. 
```python
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:
    outcome = 2
```

```python
np.full(N_BINS, output_outcome[i], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly records that coding scheme.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the NWB trials-table `early_lick` column.

ii. 
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
```

iii. The notes say the AI kept early-lick trials because the decoder needs early-lick labels as an output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `early` to `1` and everything else to `0`, then broadcasts that single trial label across all 80 bins.

ii. 
```python
el = 1 if early_lick[ti] == 'early' else 0
output_early_lick.append(el)
```

```python
np.full(N_BINS, output_early_lick[i], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` records `0 = no early lick, 1 = early lick during sample/delay`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the `Camera0_side_TongueTracking` time series, specifically the timestamps and the second data column (`y-position`). The final code does not use the likelihood column in the computation.

ii. 
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]  # y-position
```

iii. The notes say this array is DeepLabCut tracking with shape `(N, 3): [x, y, likelihood]`, but they explicitly justify the final decision as “All tracking data used (no confidence filtering).”

## 8-b. How is `output` *Tongue y-position* processed?

i. For each retained trial, the AI extracts frames in the trial window, bins them into the 80 go-cue-aligned bins, and averages raw `y` values within each bin. After processing all trials in a session, it concatenates all non-NaN per-bin values across trials, computes the 40th and 60th percentiles of that set, and digitizes each trial’s binned tongue trace using those cutoffs. It does not confidence-filter frames and does not create an explicit hidden/not-visible class.

ii. 
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y,
                                go_cue_time, bin_edges):
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

iii. The notes and trajectory justify this as a deliberate change: high-confidence tracking covered only about 14% of frames, and the AI chose to “use all tongue data for percentile computation and discretization (no confidence filtering), which gives a natural 40/20/40 split.”

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses two thresholds, `p40` and `p60`, computed from all valid per-trial binned tongue values in the session. Bins with `ty < p40` become `0`, `p40 <= ty <= p60` become `1`, and `ty > p60` become `2`. Bins with no tongue samples stay at the default middle category `1`, and sessions with no valid tongue bins at all are filled entirely with `1`.

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

iii. The notes say tongue outputs are “0 = below 40th percentile, 1 = 40th-60th percentile, 2 = above 60th percentile” and that “All DLC tracking data [are] used (no confidence filtering).” The trajectory adds that this was chosen to force a better overall category balance.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same go-cue-centered trial window as the neural data. For each trial it finds camera frames whose absolute timestamps fall in `[go + T_START, go + T_END)`, subtracts the go cue to get relative time, and assigns frames to the same 50 ms bins.

ii. 
```python
t_abs_start = go_cue_time + bin_edges[0]
t_abs_end = go_cue_time + bin_edges[-1]
idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
ts_window = tongue_timestamps[idx[0]:idx[1]]
...
ts_rel = ts_window - go_cue_time
```

iii. No separate written justification was given beyond the general notes that tongue y-position is a time-varying decoder output and that all streams were aligned to go cue.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several ad hoc fallbacks. Sessions with no `classification == 'good'` units that also map into the hand-written region list are dropped. Trials outside the estimated common recording window are removed. Missing tone-onset lookup falls back to `-1.85` seconds. Photostim parsing failures are silently ignored, leaving all-zero stimulation vectors. Missing tongue bins default to the middle category, and sessions with no valid tongue bins at all are filled with the middle category.

ii. 
```python
if len(unit_indices) == 0:
    return None
```

```python
if tone_onset_rel is None:
    tone_onset_rel = -1.85
```

```python
except (ValueError, TypeError):
    pass
```

```python
ty_disc = np.ones(N_BINS, dtype=np.int64)
...
tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]
```

iii. The notes justify the session/trial dropping as QC and observation-window handling. The trajectory justifies the `-1.85` fallback as the “typical value” and the tongue middle-category fill as a way to avoid a skewed or degenerate output distribution.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive parts of the AI code are likely reading each NWB file, constructing `unit_spike_times`, then running the nested per-trial/per-unit firing-rate computation using `np.histogram`, followed by the per-trial/per-bin tongue extraction loop.

ii. 
```python
for ti in trial_indices:
    gc = go_cue_times[ti]
    fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

```python
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
    if np.any(bin_mask):
        tongue_y_binned[b] = np.mean(y_window[bin_mask])
```

iii. No explicit runtime analysis was recorded in the notes. This is inferred from the code structure and from the large nested loops over trials, units, and bins.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization opportunities are the nested trial-by-unit neural loop, the bin-by-bin tongue loop inside `extract_tongue_y_for_trial`, and the repeated `list.index` lookups used when building `subject_idx` and `brain_region_idx`.

ii. 
```python
for ti in trial_indices:
    ...
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

```python
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

```python
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
...
idx = np.array([brain_regions.index(r) for r in regions])
```

iii. No explicit justification was recorded for leaving these loops unvectorized. The code appears to prioritize directness over efficiency.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several computations per trial that could have been hoisted or vectorized: tone-onset lookup, photostim string parsing and bin masking, per-unit histogramming for each trial independently, and list-index lookups while assembling subject and region indices.

ii. 
```python
for ti in trial_indices:
    tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
    ...
    if ps_onset_str[ti] != 'N/A':
        ...
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

```python
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
...
idx = np.array([brain_regions.index(r) for r in regions])
```

iii. No explicit justification was recorded. This is an inference from the implementation structure.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints several summary statistics and per-region counts that are not needed by the saved decoder dataset. It also computes control-trial performance diagnostics only to decide whether to skip sessions, not to populate the output arrays.

ii. 
```python
print(f"\n=== Dataset Summary ===")
print(f"Sessions: {n_processed}")
...
region_counts = {}
for regions in all_unit_regions:
    for r in regions:
        region_counts[r] = region_counts.get(r, 0) + 1
print(f"\nNeurons per region:")
```

```python
control_hits_left = np.sum(is_control & (outcomes == 'hit') & (instructions == 'left'))
control_hits_right = np.sum(is_control & (outcomes == 'hit') & (instructions == 'right'))
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
```

iii. No explicit justification was recorded. The summaries appear to be for operator inspection rather than for downstream decoding.
