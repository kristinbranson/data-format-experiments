# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files with a sorted glob over `sub-*/sub-*_ses-*.nwb`, then opens each NWB file with `pynwb.NWBHDF5IO`. Within each file it reads the subject metadata, trial table, unit table, behavioral events, and behavioral time series needed later for neural/activity alignment and tongue tracking.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
...
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()

trials = nwb.trials.to_dataframe()
units = nwb.units
be = nwb.acquisition['BehavioralEvents']
bt = nwb.acquisition['BehavioralTimeSeries']
```

iii. In the trajectory, the AI first established that there were 174 session files and 28 subjects, then decided to iterate one NWB file per session. It justified this as the natural dataset organization and as the place where all required streams already coexist.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses `nwb.subject.description` as the subject label used in the output `subjects` list and `subject_idx`, while also reading `nwb.subject.subject_id` and storing it only in the per-session intermediate result. Subjects are therefore grouped by labels like `SC015`, not by numeric NWB `subject_id`.

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description  # e.g. 'SC015'
...
sub_desc = result['subject_desc']
if sub_desc not in all_subjects:
    all_subjects[sub_desc] = result['subject_id']
...
subjects = list(all_subjects.keys())
...
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. The trajectory summary says the dataset contains 28 mice and repeatedly refers to mouse names like `SC015` and `SC016`. There is no explicit justification for preferring `subject.description` over the canonical NWB `subject_id`; the choice appears to have been made because the mouse-name field looked more interpretable.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list, and each file is processed once by `process_session`.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
```

iii. The trajectory explicitly notes 174 session files and builds the converter around one-file-per-session processing. The AI used file boundaries directly rather than trying to infer sessions from timestamps or trial structure.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table and indexed against `go_start_times`, with an assertion that the number of go cues matches the number of trial rows. After trial filtering, only the retained trial indices are used to build neural, input, and output arrays.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_total = len(trials)
...
go_start_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_start_times) == n_trials_total
...
valid_indices = np.where(valid_trial_mask)[0]
...
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
```

iii. In the trajectory, the AI treated `go_start_times` as the per-trial anchor and used the trial table as the master source of behavioral trial identity. It then restricted trial use to indices deemed recorded by `obs_intervals`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters. It keeps only a consecutive block of trials inferred from `obs_intervals`, matched by the first observed interval start time. Within those recorded trials it excludes `auto_water` and `free_water` trials. It also drops entire sessions if overall control-trial performance is below 65%, if either direction has fewer than 50 correct control trials, if no control trials exist, or if fewer than two trials remain.

ii.
```python
control_mask_all = (all_valid &
                    (early_lick == 'no early') &
                    (outcome != 'ignore') &
                    no_photostim)
...
if performance < MIN_PERFORMANCE:
    return None

if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
...
obs_0 = units.get_unit_obs_intervals(good_indices[0])
...
first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))
recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
...
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. The trajectory shows the AI inferring that many sessions contain only partial neural coverage and deciding to restrict trials to those covered by recording. It also explicitly states that session filtering should use all control trials in the session, while actual trial use should exclude `auto_water` and `free_water`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from unit spike times in `nwb.units` together with go-cue timestamps from `BehavioralEvents/go_start_times`. Unit inclusion also depends on `classification == 'good'` and on having a non-null brain-region mapping from `anno_name`.

ii.
```python
classification = units['classification'].data[:]
anno_names = units['anno_name'].data[:]
...
go_start_times = be.time_series['go_start_times'].timestamps[:]
...
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. The trajectory repeatedly identifies absolute-session spike times and go-cue times as the core neural ingredients. It also states that only `'good'` units should be used and that `anno_name` is needed for brain-region assignment.

## 2-b. How is the `neural` data processed?

i. For each retained trial, the AI takes a `[-2.5, 1.5]` s window around the go cue, clips each unit’s spike times to that window, bins spikes with `np.histogram` into 80 non-overlapping 50 ms bins, and divides by bin width to convert counts to firing rates in spikes/s. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
BIN_WIDTH = 0.05
BEGIN_TIME = -2.5
END_TIME = 1.5
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
...
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    trial_fr = np.zeros((n_good, N_BINS), dtype=np.float32)
    abs_start = go_time + BEGIN_TIME
    abs_end = go_time + END_TIME

    for i, st in enumerate(all_spike_times):
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
        if hi > lo:
            rel_spikes = st[lo:hi] - go_time
            counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
            trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The trajectory says the AI would align spikes to go cue, extract `-2.5 s` to `+1.5 s`, and bin at `50 ms`. Later it identifies per-unit spike extraction and firing-rate computation as the main bottleneck, confirming that this binned-rate pipeline was the intended processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered to those with `classification == 'good'` and with an `anno_name` that the custom region-mapping function can map to one of 14 broad brain regions. If no such units remain, the whole session is dropped.

ii.
```python
classification = units['classification'].data[:]
anno_names = units['anno_name'].data[:]
...
good_indices = []
unit_regions = []
for ui in range(n_units_total):
    if classification[ui] != 'good':
        continue
    region = map_anno_to_region(anno_names[ui])
    if region is not None:
        good_indices.append(ui)
        unit_regions.append(region)

if n_good == 0:
    return None
```

iii. The trajectory explicitly states that only `'good'` classified units should be kept and that `anno_name` should be mapped to 14 broad regions. It also notes concern when unmapped annotations reduced the unit count, showing that region mappability was part of the effective unit filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go-cue onset by subtracting each trial’s go-cue timestamp from spikes in that trial window and binning relative times from `-2.5` to `+1.5` seconds.

ii.
```python
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
...
rel_spikes = st[lo:hi] - go_time
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The trajectory says spike times are in absolute session time and therefore need to be aligned to go-cue times from behavioral events. That is exactly what the final code does by windowing around `go_time` and expressing spikes relative to it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins over a 4 s window, yielding 80 time bins per trial. No additional temporal rebinning or smoothing is applied beyond this single histogramming step.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

iii. The trajectory repeatedly cites the instruction-mandated `50 ms` resolution and `-2.5 s` to `+1.5 s` window. No later reasoning introduces any secondary resampling.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final code it is not derived from a raw per-trial event variable. Instead, it is derived from a fixed constant `TONE_ONSET_REL = -1.85` seconds relative to the go cue, combined with the shared bin centers.

ii.
```python
TONE_ONSET_REL = -1.85  # tone onset relative to go cue (sample 0.65s + delay 1.2s)
...
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. The trajectory shows the AI first inspecting `sample_start_times`, then concluding that tone onset is always 1.85 s before the go cue because the sample and delay durations are fixed. It therefore chose a constant-offset derivation rather than looking up a raw event per trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI precomputes one shared time series for all trials: each bin center, expressed in seconds relative to a tone assumed to occur exactly 1.85 s before the go cue. No per-trial tone lookup is done, and negative values before tone onset are preserved.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
...
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The trajectory explicitly says the AI decided that, for any bin time `t` relative to go, time from tone onset should be `t + 1.85 s`, and that values before tone onset should remain negative rather than being clamped.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by construction to the same 80 go-cue-centered bin centers used for the neural firing rates. The time-from-tone vector is reused unchanged for every trial.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
...
go_time = go_start_times[trial_idx]
...
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The trajectory treats the go-cue bin grid as the master time base for every stream. Because `TIME_FROM_TONE` is defined on those same bin centers, no separate alignment step is performed.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the absolute session timestamps `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, not from `photostim_onset` and `photostim_duration` in the trials table.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The trajectory treats photostimulation as a binary time-varying variable and later mentions vectorizing its computation. It does not give a detailed justification for preferring event streams over the trial-table onset/duration fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI initializes an 80-bin zero vector, scans all session photostimulation intervals, and marks a bin as 1 when that bin center falls between a photostim start and stop time that overlaps the trial window.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float32)
trial_start_abs = go_time + BEGIN_TIME
trial_end_abs = go_time + END_TIME

for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        rel_start = ps_start - go_time
        rel_stop = ps_stop - go_time
        mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
        photostim_binary[mask] = 1.0
```

iii. The trajectory describes photostimulation as a binary variable and later calls photostim computation one of the operations worth vectorizing. No separate branch for unstimulated trials is needed because the vector starts at all zeros.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned by expressing session-absolute photostim start/stop times relative to each trial’s go cue and then testing the shared neural bin centers against those relative intervals.

ii.
```python
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. The trajectory’s general alignment strategy is to put all streams on the go-cue-centered time axis. The code follows that by subtracting `go_time` before applying the same bin centers used for neural activity.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from two trial-table variables: `trial_instruction` and `outcome`.

ii.
```python
instr = trial_instruction[trial_idx]
outc = outcome[trial_idx]
```

iii. The trajectory explicitly recognized that early-lick and ignore trials complicate choice coding, and it considered different ways to represent no-response trials. The final code still uses only instruction and outcome as the inputs to the choice label.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For hits, the choice equals the instructed side; for misses, it is the opposite side; for ignores, the code assigns the instructed side again rather than a separate “no lick” category. The resulting per-trial scalar is repeated across all 80 bins. `output_values` only defines two classes, `left` and `right`.

ii.
```python
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
...
output_values = [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
]
...
np.full(N_BINS, choice, dtype=np.int64)
```

iii. The trajectory shows the AI wrestling with how to encode no-response (`ignore`) trials. It considered excluding them or using the instructed side as a proxy, and the final implementation chooses the instructed side without recording a stronger justification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` column.

ii.
```python
outcome = trials['outcome'].values
...
outc = outcome[trial_idx]
```

iii. The trajectory treats `outcome` as already present in the NWB trial table and required as a decoder output, so no additional derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `ignore`, `miss`, and `hit` to `0`, `1`, and `2`, respectively, and repeats that per-trial value across all 80 bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
...
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The trajectory keeps outcome as a per-trial categorical output because the decoder spec requires it. No time-varying transformation is attempted; the scalar is simply broadcast across bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trials-table `early_lick` column.

ii.
```python
early_lick = trials['early_lick'].values
...
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The trajectory explicitly notes that early-lick trials must be retained because `early_lick` itself is one of the required decoder outputs.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string label is reduced to a binary code, `1` for `'early'` and `0` otherwise, and the resulting scalar is repeated across all 80 bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
...
np.full(N_BINS, early_lick_val, dtype=np.int64)
```

iii. The trajectory justifies keeping early-lick trials specifically so this output remains meaningful. The final code uses the simplest binary encoding compatible with that goal.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue position from `BehavioralTimeSeries/Camera0_side_TongueTracking`, using only the `y` column (`data[:, 1]`) and the corresponding timestamps. Although the trajectory recognized that the stream contains `(x, y, confidence)`, the final code ignores the confidence/likelihood column.

ii.
```python
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. Earlier in the trajectory, the AI noted that the tongue tracking stream has `(x, y, confidence)` columns. However, the final implementation drops the confidence column and no explicit justification for that simplification is recorded.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes the 40th and 60th percentiles over all raw session `tongue_y` samples. For each neural bin center in each trial, it picks the closest tongue frame in time and uses that single raw `y` value rather than bin-averaging all frames in the 50 ms window. It does not mask low-confidence frames.

ii.
```python
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
...
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
...
tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. The trajectory originally suggested filtering low-likelihood frames and aggregating the ~300 Hz tongue signal into 50 ms bins, but the final code does not implement that plan. No later trajectory entry explains why the implementation was simplified to nearest-frame sampling.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code thresholds the nearest-frame `y` values into three visible-state categories: `<40th percentile -> 0`, `40th percentile to 60th percentile -> 1`, and `>60th percentile -> 2`. It does not create the requested fourth “not visible” category.

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
...
output_values = [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
]
```

iii. The trajectory mentions percentile-based discretization but does not justify omitting the “not visible” state from the instruction. The final code therefore implements only three categories.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue position is aligned by evaluating the tongue tracking signal at the nearest camera timestamp to each neural bin center (`go_time + BIN_CENTERS`). The alignment is therefore one sampled frame per neural bin, not a per-bin average over all camera frames in the bin.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
...
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]
```

iii. The trajectory says the tongue data must be aligned to the 50 ms neural bins, but the final implementation operationalizes that as nearest-frame lookup at each bin center. No separate justification is given for preferring nearest-frame sampling over within-bin averaging.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data mostly by dropping units, trials, or sessions. Units with missing or unmappable `anno_name` are excluded. Sessions are dropped if they fail performance thresholds, have no control trials, have no acceptable units, or have fewer than two retained trials. Missing/`N/A` photostim values are treated as “no photostim” for control-trial filtering. `obs_intervals` are matched to trial indices by nearest start time rather than exact equality. The code does not explicitly model missing tongue visibility.

ii.
```python
no_photostim = np.array([str(p) == 'N/A' or str(p) == 'nan' or
                         (isinstance(p, (int, float)) and p == 0)
                         for p in photostim_power])
...
region = map_anno_to_region(anno_names[ui])
if region is not None:
    good_indices.append(ui)
...
first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))
...
if n_control == 0:
    return None
if n_good == 0:
    return None
if n_trials < 2:
    return None
```

iii. The trajectory shows the AI treating partial neural coverage as a data-integrity issue that requires trial filtering, and later using session dropping for performance-based curation. It also noticed missing or unmapped annotations and adjusted region mapping to recover some units, but it did not implement an explicit missing-data category for tongue visibility.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s own trajectory identifies per-unit spike-time extraction and firing-rate computation as the dominant cost. In the final code, the expensive parts are reading spike trains, looping over trials and units to histogram spikes, scanning all photostim intervals per trial, and doing per-trial nearest-frame tongue lookup.

ii.
```python
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)

for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        ...
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The trajectory explicitly says the “main bottleneck” is `units.get_unit_spike_times(ui)` plus firing-rate computation, and later says the script is slow enough that optimization might matter. It also lists tongue and photostim computation as secondary optimization targets.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized: the nested trial-by-unit neural histogram loop, the loop over all photostim events for every trial, the loop that preloads unit spike times one unit at a time, and the repeated nearest-frame tongue lookup performed independently for each trial.

ii.
```python
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
...
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        ...
    for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
        ...
```

iii. The trajectory explicitly calls out three optimization ideas: load all spike times at once, vectorize tongue computation, and vectorize photostim computation. The final code still contains those loops, so the optimization opportunities remained.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations across trials. It scans the full photostimulation event list for every single trial, recomputes nearest tongue timestamps for each trial separately, and redoes per-unit spike histogramming independently for every trial instead of using a session-level vectorized spike/binning pass.

ii.
```python
for trial_idx in valid_indices:
    ...
    for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
        ...
    abs_times = go_time + BIN_CENTERS
    tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
    ...
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        ...
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The trajectory mentions wanting to vectorize photostim and tongue processing and identifies the per-unit spike loop as the main performance issue. Those comments are consistent with the final code still recomputing these quantities trial by trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several values that are only used for filtering, logging, or metadata rather than as decoder inputs or outputs: session performance, `correct_left`, `correct_right`, `n_control`, the `subject_id` field kept only in intermediates, and the full region keyword-mapping machinery. It also repeatedly scans photostim intervals and tongue timestamps in ways that are implementation overhead rather than required output.

ii.
```python
n_control = control_mask_all.sum()
n_correct = ((outcome == 'hit') & control_mask_all).sum()
performance = n_correct / n_control
correct_left = ((outcome == 'hit') & (trial_instruction == 'left') & control_mask_all).sum()
correct_right = ((outcome == 'hit') & (trial_instruction == 'right') & control_mask_all).sum()
...
return {
    ...
    'subject_id': subject_id,
    'performance': performance,
    'correct_left': correct_left,
    'correct_right': correct_right,
}
```

iii. The trajectory shows that these computations were introduced mainly to reproduce paper-style curation and to monitor whether sessions should be kept. They are not used by the decoder itself after dataset construction.
