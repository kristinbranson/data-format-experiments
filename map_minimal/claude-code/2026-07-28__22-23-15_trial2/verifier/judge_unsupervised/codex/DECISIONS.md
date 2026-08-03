# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The code loads all data by globbing every NWB file under `data/sub-*/sub-*_ses-*.nwb`, sorting the filenames, and processing each file as one session with `pynwb.NWBHDF5IO`. Inside each session it reads the NWB `trials` table, `BehavioralEvents`, `BehavioralTimeSeries`, and `units`.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
print(f"Found {len(nwb_files)} NWB files")

for i, nwb_file in enumerate(nwb_files):
    print(f"[{i+1}/{len(nwb_files)}] {os.path.basename(nwb_file)}")
    result = process_session(nwb_file, verbose=verbose)
```

```python
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
trials = nwb.trials.to_dataframe()
be = nwb.acquisition['BehavioralEvents']
bt = nwb.acquisition['BehavioralTimeSeries']
units = nwb.units
```

iii. The trajectory shows the agent first explored `/app/data`, found 174 NWB files, and then built the converter around direct NWB access. `CONVERSION_NOTES.md` also states that the source data are 174 NWB files from the MAP dataset.

## 1-b. How are the data split into subjects?

i. Subjects are split using one NWB subject record per session. The exported `subjects` list is built from unique `nwb.subject.description` values such as `SC015`, while `subject_id` is kept only in per-session intermediate results and is not exported as the canonical subject label.

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description  # e.g. 'SC015'
```

```python
if result is not None:
    all_sessions.append(result)
    sub_desc = result['subject_desc']
    if sub_desc not in all_subjects:
        all_subjects[sub_desc] = result['subject_id']

subjects = list(all_subjects.keys())
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. The code suggests the agent chose the human-readable mouse code in `description` as the exported subject ID. The trajectory shows it inspected the NWB subject fields and saw both `subject_id` and `description`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are kept in sorted filename order, and only sessions that pass the code's filtering logic are appended to the final dataset.

ii.
```python
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
    if result is not None:
        all_sessions.append(result)
```

```python
neural.append(sess['neural'])
inputs.append(sess['input'])
outputs.append(sess['output'])
```

iii. This matches the structure the agent found in the data directory: one NWB file per recording session. `CONVERSION_NOTES.md` records that 144 of 174 sessions passed the filters.

## 1-d. How are the data split into trials?

i. Trial boundaries come from the NWB `trials` table and the per-trial `go_start_times`. The code then restricts to a subset of "recorded" trials by using `obs_intervals` from the first good unit, truncating to the minimum number of intervals across good units, inferring the first recorded trial by nearest `start_time`, and assuming the recorded trials are consecutive from there. It finally drops `auto_water` and `free_water` trials.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_total = len(trials)
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

```python
obs_0 = units.get_unit_obs_intervals(good_indices[0])
n_obs_trials = len(obs_0)

for ui in good_indices[1:]:
    oi = units.get_unit_obs_intervals(ui)
    if len(oi) != n_obs_trials:
        n_obs_trials = min(n_obs_trials, len(oi))

trial_starts = trials['start_time'].values
first_obs_start = obs_0[0, 0]
first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))
recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
recorded_trial_indices = recorded_trial_indices[recorded_trial_indices < n_trials_total]
```

```python
valid_trial_mask = np.zeros(n_trials_total, dtype=bool)
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True

valid_indices = np.where(valid_trial_mask)[0]
```

iii. The notes and trajectory both justify this as a fix for sessions where recording starts mid-session and `obs_intervals[0]` is not trial 0. The trajectory summary explicitly mentions that this was added after the agent saw all-zero neural trials.

## 1-e. How are trials filtered based on quality controls?

i. There are two layers of filtering. Session QC is computed on all control trials in the session: `auto_water == 0`, `free_water == 0`, `early_lick == 'no early'`, `outcome != 'ignore'`, and no photostimulation. A session is kept only if performance is at least 65% hit rate on those control trials and there are at least 50 correct left and 50 correct right control trials. After a session passes, trial-level filtering keeps only recorded trials that are not `auto_water` and not `free_water`; early-lick and ignore trials are retained for decoder outputs.

ii.
```python
all_valid = (auto_water == 0) & (free_water == 0)
control_mask_all = (all_valid &
                    (early_lick == 'no early') &
                    (outcome != 'ignore') &
                    no_photostim)
```

```python
n_correct = ((outcome == 'hit') & control_mask_all).sum()
performance = n_correct / n_control

correct_left = ((outcome == 'hit') & (trial_instruction == 'left') & control_mask_all).sum()
correct_right = ((outcome == 'hit') & (trial_instruction == 'right') & control_mask_all).sum()

if performance < MIN_PERFORMANCE:
    return None

if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
```

```python
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. The methods excerpt in the trajectory states `> 65%` performance and `>= 50` correct lick-left and lick-right trials. The notes also say the agent intentionally changed performance calculation to use all session trials rather than only recorded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is derived from raw spike times in `units.get_unit_spike_times(ui)`, aligned to the trial-level `go_start_times`. Unit selection also depends on the raw `classification`, `anno_name`, and `obs_intervals` unit fields.

ii.
```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
classification = units['classification'].data[:]
anno_names = units['anno_name'].data[:]
```

```python
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. The trajectory shows the agent inspected NWB unit columns and found `classification`, `anno_name`, `spike_times`, and `obs_intervals`, then built the neural conversion directly from those fields.

## 2-b. How is the `neural` data processed?

i. For each valid trial, the code takes every selected unit's absolute spike times, clips them to `[go_time - 2.5 s, go_time + 1.5 s)`, subtracts `go_time` to make them trial-relative, bins them with a simple histogram into 80 non-overlapping 50 ms bins, and divides by `0.05` to convert counts to firing rates in spikes/s. It does not implement the reference preprocessing parameters `bw = 0.04` and `stride = 0.0034`.

ii.
```python
BIN_WIDTH = 0.05
BEGIN_TIME = -2.5
END_TIME = 1.5
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

```python
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

iii. The notes say the neural output is spike counts per bin converted to firing rates. The trajectory also shows the agent read `preprocess_all_ephys.py`, saw the original paper code used `bw=0.04` and `stride=0.0034`, but still chose simple 50 ms histogramming to satisfy the decoder instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code keeps only units with `classification == 'good'` and a non-empty `anno_name` that its own `map_anno_to_region` string rules can map to one of 14 regions. Sessions with no such units are dropped. Neural trials are then restricted to the recorded subset inferred from `obs_intervals`, and sessions must also pass the behavioral QC described above.

ii.
```python
good_indices = []
unit_regions = []
for ui in range(n_units_total):
    if classification[ui] != 'good':
        continue
    region = map_anno_to_region(anno_names[ui])
    if region is not None:
        good_indices.append(ui)
        unit_regions.append(region)
```

```python
if n_good == 0:
    return None
```

iii. The methods excerpt says the paper uses classifier-labeled `good` units. The trajectory summary notes that the original codebase relied on pre-organized QC files and Allen hierarchy utilities, so the agent's custom string matcher is its own replacement rather than the original region assignment path.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to go cue onset. For each trial the code subtracts that trial's `go_start_times[trial_idx]` from spike times and bins the resulting relative times from `-2.5 s` to `+1.5 s`.

ii.
```python
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
rel_spikes = st[lo:hi] - go_time
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. This follows the user instructions directly. The notes also describe the temporal alignment event as go cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, giving 80 bins over the 4 s window. No further temporal rebinning or smoothing is applied after histogramming.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

iii. This comes from the decoder task. The trajectory shows the agent knew the reference preprocessing used different parameters, but chose not to reproduce them here.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final code it is not derived from a per-trial raw NWB variable. It is computed from the fixed bin centers and a hard-coded constant `TONE_ONSET_REL = -1.85`. The notes say that `-1.85 s` was inferred from comparing `sample_start_times` to `go_start_times` and from the known task structure.

ii.
```python
TONE_ONSET_REL = -1.85
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. The justification in `CONVERSION_NOTES.md` is that tone onset is consistently 1.85 s before the go cue because the sample period is 650 ms and the delay is 1.2 s.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code precomputes one shared 80-element vector equal to each bin center minus `-1.85 s`, so the resulting value is time since tone onset at each go-aligned bin center. The same vector is reused for every trial.

ii.
```python
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
...
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The notes justify using a constant offset across all trials instead of a per-trial event lookup.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is represented on exactly the same 80 go-aligned time bins as the neural data, so there is no extra alignment step beyond reusing `BIN_CENTERS`.

ii.
```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The agent's reasoning was that tone onset timing is deterministic once the go-aligned bins are fixed.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The implemented code derives photostimulation from the `BehavioralEvents` event streams `photostim_start_times` and `photostim_stop_times`. Separately, it uses the trial table's `photostim_power` column to detect control trials during session filtering. The notes incorrectly say this input comes from a `photo_stim_type` trial column, which does not exist in the NWB files that the code reads.

ii.
```python
photostim_power = trials['photostim_power'].values
no_photostim = np.array([str(p) == 'N/A' or str(p) == 'nan' or
                         (isinstance(p, (int, float)) and p == 0)
                         for p in photostim_power])
```

```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The trajectory shows the agent explored the trial columns and event streams in the NWB files. The written notes, however, misstate the source variable.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code creates an all-zero length-80 vector, scans every photostimulation interval in the session, converts each interval into times relative to that trial's go cue, and marks bins whose centers fall inside an overlapping interval as `1.0`.

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

iii. The likely justification was the decoder requirement that photostimulation be represented as a time-varying on/off input rather than a per-trial scalar.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is done by subtracting the current trial's `go_time` from the absolute photostimulation start and stop times, then using the same `BIN_CENTERS` grid as the neural data to mark on-bins.

ii.
```python
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
photostim_binary[mask] = 1.0
```

iii. The methods excerpt says photoinhibition ends before the go cue, so a go-aligned binary trace naturally places the stimulation in the pre-go bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The code does not derive choice from actual lick-event streams. Instead it derives choice from the trial table fields `trial_instruction` and `outcome`: hit means choice equals instructed side, miss means the opposite side, and ignore means choice defaults back to the instructed side.

ii.
```python
instr = trial_instruction[trial_idx]
outc = outcome[trial_idx]

if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. `CONVERSION_NOTES.md` simplifies this and says choice is from `trial_instruction`, but the code clearly also uses `outcome`. The agent appears to have made this choice because the trial table has no explicit `choice` column.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. After inferring a binary left/right choice from `trial_instruction` and `outcome`, the code repeats that scalar choice across all 80 bins of the trial.

ii.
```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_lick_val, dtype=np.int64),
    tongue_y_disc.astype(np.int64),
], axis=0)
```

iii. There is no explicit justification in the notes beyond matching the required output format and forcing all outputs onto a common `(n_output, n_timepoints)` shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The implemented code derives outcome directly from the `outcome` column of the NWB trial table. The notes say it is derived from `outcome` and `early_lick`, but `early_lick` is not used in the actual mapping.

ii.
```python
outc = outcome[trial_idx]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

iii. The likely justification was that the NWB trial table already stores exactly the three needed labels: `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The processing is a direct categorical remap: `ignore -> 0`, `miss -> 1`, `hit -> 2`, then broadcasting that scalar across the 80 time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
...
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. This was done to match the decoder specification exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is derived from the NWB trial table's `early_lick` column.

ii.
```python
early_lick = trials['early_lick'].values
...
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The trajectory shows the agent inspected the trial columns and found `early_lick` already encoded there.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `early -> 1` and anything else, effectively `no early`, to `0`, then repeats that scalar across all 80 time bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
...
np.full(N_BINS, early_lick_val, dtype=np.int64)
```

iii. This directly follows the decoder task.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The code derives tongue position from `BehavioralTimeSeries['Camera0_side_TongueTracking']`, specifically `data[:, 1]` for y-position and `timestamps[:]` for time alignment. It ignores the third tracking column, which is the DeepLabCut likelihood/confidence.

ii.
```python
bt = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. The notes describe the source as DeepLabCut tongue tracking and claim a later likelihood filter, but the code never uses the likelihood column.

## 8-b. How is `output` *Tongue y-position* processed?

i. First, the code computes session-wide 40th and 60th percentiles over all y-values in the session. Then for each trial it finds the nearest tongue-tracking timestamp to each go-aligned bin center, reads the corresponding y-values, and discretizes each sampled y-value into low/mid/high. There is no likelihood filtering, interpolation, or smoothing.

ii.
```python
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
```

```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]

tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. The likely motivation was to keep tongue output on the same 50 ms grid as the neural data. The notes claim a different implementation: 33rd/67th percentile thresholds on likelihood-filtered data with a default low class for invalid data.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code thresholds it into three categories using the 40th and 60th percentiles of session-wide tongue y-values: below the 40th percentile stays `0`, values at or above the 40th percentile become at least `1`, and values above the 60th percentile become `2`.

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. This matches the decoder task text. The notes conflict with the code here by describing 33rd/67th percentile thresholds instead.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The code aligns tongue position to the neural data by querying tongue-tracking samples nearest to `go_time + BIN_CENTERS`, so each of the 80 neural bins receives one nearest-neighbor tongue category on the same go-aligned time axis.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
...
tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. The alignment choice is implicit in the code and reflects the common-bin target format.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles data issues mainly by dropping or skipping. Units with bad `classification` or unmappable/empty `anno_name` are excluded. Sessions with no control trials, low performance, too few correct left/right trials, no usable units, or fewer than two valid trials are skipped. `obs_intervals` misalignment is handled with a heuristic nearest-start match. Trials with no spikes simply get all-zero firing-rate rows. There is no explicit NaN handling for tongue traces, no likelihood filtering, and no code path that implements the notes' claimed "default low class for no valid tongue data."

ii.
```python
if classification[ui] != 'good':
    continue
region = map_anno_to_region(anno_names[ui])
if region is not None:
    good_indices.append(ui)
```

```python
if n_control == 0:
    return None
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
if n_trials < 2:
    return None
```

```python
first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))
```

iii. The trajectory summary explicitly mentions two bug-fix decisions: session performance must be computed on all trials, and `obs_intervals` cannot be assumed to start at trial 0. Beyond that, most missing-data handling is implicit rather than documented.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is the nested session -> trial -> unit loop that bins spikes with `searchsorted` and `np.histogram`. Secondary expensive steps are the per-trial scan over all photostimulation intervals and the per-trial nearest-neighbor tongue alignment.

ii.
```python
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
        if hi > lo:
            rel_spikes = st[lo:hi] - go_time
            counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    ...
```

```python
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
```

iii. This is inferred from the loop structure and data sizes, not explicitly justified in the notes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are easy vectorization candidates: the unit-filtering loop, the recorded-trial validity loop, the per-session region-count summary loop, the per-trial scan over all photostim intervals, and especially the repeated per-unit histogramming inside the per-trial loop.

ii.
```python
for ui in range(n_units_total):
    if classification[ui] != 'good':
        continue
    region = map_anno_to_region(anno_names[ui])
```

```python
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

```python
for sess_br in brain_region_idx:
    for idx in sess_br:
        region_counts[brain_regions[idx]] += 1
```

iii. This is an implementation analysis rather than a paper-derived decision.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans all session photostim intervals for every trial, recomputes nearest tongue timestamps for every trial, and performs `searchsorted` plus histogramming for every unit in every trial. It also does repeated `subjects.index(...)` lookups while assembling the final output.

ii.
```python
for trial_idx in valid_indices:
    ...
    for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
        ...
```

```python
for trial_idx in valid_indices:
    ...
    tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
```

```python
for sess in all_sessions:
    ...
    subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. This is visible directly from the implementation; the notes do not discuss efficiency.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code broadcasts the scalar per-trial outputs `choice`, `outcome`, and `early_lick` to all 80 time bins even though they carry no within-trial temporal variation. It duplicates the first five sessions into a separate `sample_data` object, builds and prints a region-count summary used only for logging, and stores intermediate per-session stats such as `performance`, `correct_left`, and `correct_right` only to discard them from the final dataset.

ii.
```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_lick_val, dtype=np.int64),
    tongue_y_disc.astype(np.int64),
], axis=0)
```

```python
region_counts = {r: 0 for r in brain_regions}
for sess_br in brain_region_idx:
    for idx in sess_br:
        region_counts[brain_regions[idx]] += 1
```

```python
sample_data = {
    'neural': neural[:n_sample],
    'input': inputs[:n_sample],
    'output': outputs[:n_sample],
    ...
}
```

iii. This is an implementation-level observation. The agent's notes focus on correctness and validation rather than downstream efficiency.
