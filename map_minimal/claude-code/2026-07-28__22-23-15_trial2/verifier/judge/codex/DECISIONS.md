# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing all NWB files under `sub-*`, then opening each file with `NWBHDF5IO` and reading the session-level NWB object. Within each session it reads trials from `nwb.trials`, events from `nwb.acquisition['BehavioralEvents']`, units from `nwb.units`, and behavioral time series from `nwb.acquisition['BehavioralTimeSeries']`.

ii. 
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
print(f"Found {len(nwb_files)} NWB files")
```

```python
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()

trials = nwb.trials.to_dataframe()
be = nwb.acquisition['BehavioralEvents']
units = nwb.units
bt = nwb.acquisition['BehavioralTimeSeries']
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset consists of 174 NWB files and that NWB is the source format for the MAP dataset.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.description` as the subject label in the final dataset, while also reading `nwb.subject.subject_id` separately. It builds the final `subjects` list from the first-seen `subject_desc` values and assigns each session a `subject_idx` by lookup into that list.

ii. 
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description  # e.g. 'SC015'
```

```python
sub_desc = result['subject_desc']
if sub_desc not in all_subjects:
    all_subjects[sub_desc] = result['subject_id']
...
subjects = list(all_subjects.keys())
...
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. The notes justify this implicitly by reporting mouse-style labels such as `SC015` and by framing the output as “144 sessions from 28 mice,” but they do not explain why `description` was preferred over the canonical `subject_id`.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Sessions are processed in sorted filename order, and each surviving file contributes one session entry to `neural`, `input`, `output`, and `brain_region_idx`.

ii. 
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
```

```python
return {
    'neural': neural_data,
    'input': input_data,
    'output': output_data,
    ...
    'sess_name': sess_name,
}
```

iii. The notes justify this by describing the source as 174 NWB files, effectively one file per session.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table as the source of behavioral trials and asserts that the number of `go_start_times` equals the number of rows in `nwb.trials`. Later processing loops over selected trial indices from that table.

ii. 
```python
trials = nwb.trials.to_dataframe()
n_trials_total = len(trials)
...
go_start_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_start_times) == n_trials_total
```

```python
valid_indices = np.where(valid_trial_mask)[0]
...
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
```

iii. No separate justification is given beyond relying on the NWB trial structure and the go-cue count check.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial and session filters. At the trial level it keeps only trials inferred to be covered by `obs_intervals`, then excludes `auto_water` and `free_water` trials, and requires at least two remaining trials. At the session level it further drops sessions below 65% correct on control trials or with fewer than 50 correct left and 50 correct right control trials.

ii. 
```python
control_mask_all = (all_valid &
                    (early_lick == 'no early') &
                    (outcome != 'ignore') &
                    no_photostim)
...
if performance < MIN_PERFORMANCE:
    ...
    return None

if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    ...
    return None
```

```python
obs_0 = units.get_unit_obs_intervals(good_indices[0])
...
recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
...
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. `CONVERSION_NOTES.md` explicitly justifies the behavioral session filters from paper language about session inclusion, says `obs_intervals` identifies trials with neural coverage, and says `auto_water` and `free_water` are excluded while early-lick and ignore trials are kept.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from per-unit spike times in `nwb.units`, with go-cue timestamps from `BehavioralEvents/go_start_times` used to define the extraction window and bins.

ii. 
```python
units = nwb.units
...
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
...
go_time = go_start_times[trial_idx]
```

iii. The notes justify this as “Spike counts per bin, converted to firing rates (spikes/s)” after go-cue alignment.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each kept unit, the AI windows spike times to `[go-2.5 s, go+1.5 s]`, shifts spikes into trial-relative time by subtracting the go cue, bins with `np.histogram` into 50 ms bins, and divides counts by bin width to produce firing rates in spikes/s.

ii. 
```python
trial_fr = np.zeros((n_good, N_BINS), dtype=np.float32)
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
```

```python
lo = np.searchsorted(st, abs_start)
hi = np.searchsorted(st, abs_end)
if hi > lo:
    rel_spikes = st[lo:hi] - go_time
    counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
    trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The notes justify this as using 50 ms non-overlapping bins and converting spike counts to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == 'good'` and additionally requires a non-empty `anno_name` that can be mapped into one of 14 broad regions by a custom string-matching function. Sessions with no such units are dropped.

ii. 
```python
classification = units['classification'].data[:]
anno_names = units['anno_name'].data[:]
...
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
    ...
    return None
```

iii. The notes justify `classification == 'good'` as QC-based unit curation and add the extra requirement that units must map to one of 14 broad brain regions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset by using each trial’s `go_start_times` timestamp as time zero and extracting spikes in a fixed `[-2.5, 1.5]` s window relative to that event.

ii. 
```python
BEGIN_TIME = -2.5
END_TIME = 1.5
...
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
...
rel_spikes = st[lo:hi] - go_time
```

iii. The notes explicitly state “Temporal alignment: Go cue onset (t=0).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms non-overlapping bins, yielding 80 bins over the 4 s window. No later temporal rebinning is applied.

ii. 
```python
BIN_WIDTH = 0.05
BEGIN_TIME = -2.5
END_TIME = 1.5
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The notes justify this directly from the requested decoder format: 50 ms non-overlapping bins from -2.5 s to +1.5 s around go cue.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the code, this input is not derived from a per-trial raw variable at all. It is derived from a hard-coded constant `TONE_ONSET_REL = -1.85` and the shared bin centers. The notes say this constant was inferred by comparing `sample_start_times` to `go_start_times`, but the code does not use `sample_start_times`.

ii. 
```python
TONE_ONSET_REL = -1.85  # tone onset relative to go cue (sample 0.65s + delay 1.2s)
...
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The justification in `CONVERSION_NOTES.md` is that tone onset was found to be consistently 1.85 s before go cue based on comparing `sample_start_times` and `go_start_times`, so the same vector could be reused for all trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI precomputes one 80-element vector of bin-center times relative to an assumed fixed tone onset at `-1.85 s` from go cue, and then reuses that identical vector for every trial.

ii. 
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The notes justify this by saying the tone-to-go interval is deterministic from task structure.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by construction to the same 80 go-cue-centered bins used for neural data. The AI does not compute a separate tone-aligned grid; it simply pairs the shared per-bin time-from-tone vector with each trial’s neural matrix.

ii. 
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
...
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The notes justify this by saying tone onset is fixed relative to the go-cue-aligned bin structure.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. In the code, photostimulation is derived from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`. The notes instead describe it as coming from a trial-level photostimulation indicator column, so the documentation and code are inconsistent.

ii. 
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The notes justify photostimulation only at a high level, saying it is a binary input marking whether stimulation is on.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI initializes an all-zero 80-bin vector and scans every photostimulation interval in the session. If a global photostim interval overlaps that trial’s window, it converts the interval to go-relative time and marks bins whose centers fall within `[start, stop)` as 1.

ii. 
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float32)
trial_start_abs = go_time + BEGIN_TIME
trial_end_abs = go_time + END_TIME
```

```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        rel_start = ps_start - go_time
        rel_stop = ps_stop - go_time
        mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
        photostim_binary[mask] = 1.0
```

iii. The notes justify the output conceptually as a time-varying binary photostimulation signal, but they do not explain why session-global event timestamps were preferred over trial-table onset and duration fields.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation to neural data by shifting each photostim event interval into time relative to the current trial’s go cue and comparing those relative times to the same `BIN_CENTERS` used for the neural bins.

ii. 
```python
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. The justification is implicit in the code: the same go-cue-centered bin grid is used for both neural data and the photostimulation indicator.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The code derives choice from the trial-level `trial_instruction` and `outcome` columns. It does not use any direct lick-side measurement.

ii. 
```python
trial_instruction = trials['trial_instruction'].values
outcome = trials['outcome'].values
```

```python
instr = trial_instruction[trial_idx]
outc = outcome[trial_idx]
```

iii. The notes conflict with the code here: they describe choice as binary and “from `trial_instruction`,” but the actual code also branches on `outcome`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `hit` to the instructed side, `miss` to the opposite side, and `ignore` to the instructed side rather than to a separate “no lick” category. It then repeats that per-trial scalar across all 80 bins.

ii. 
```python
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_lick_val, dtype=np.int64),
    tongue_y_disc.astype(np.int64),
], axis=0)
```

iii. The notes justify this only weakly by documenting the output as binary `0=left, 1=right`, with no explicit discussion of how `ignore` trials should be encoded.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the trials-table `outcome` column.

ii. 
```python
outcome = trials['outcome'].values
...
outc = outcome[trial_idx]
```

iii. The notes describe this as the three trial outcomes ignore, miss, and hit.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps the string outcomes to integer classes with `ignore=0`, `miss=1`, and `hit=2`, then repeats the resulting per-trial scalar across all 80 bins.

ii. 
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

```python
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The notes explicitly document the same 0/1/2 mapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trials-table `early_lick` column.

ii. 
```python
early_lick = trials['early_lick'].values
...
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The notes state that early lick comes from the NWB `early_lick` column.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts `'early'` to 1 and everything else to 0, then repeats the per-trial class across all 80 bins.

ii. 
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

```python
np.full(N_BINS, early_lick_val, dtype=np.int64)
```

iii. The notes explicitly document the same binary mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The code derives tongue position from the side-camera tongue tracking time series, using `tongue_ts.data[:, 1]` as tongue y-position and `tongue_ts.timestamps` for timing. It does not use the tracking likelihood column.

ii. 
```python
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. The notes say tongue y-position is from DeepLabCut tracking, but they incorrectly claim likelihood-based filtering that the code does not perform.

## 8-b. How is `output` *Tongue y-position* derived from those variables?

i. The AI computes 40th and 60th percentiles from all session-wide raw tongue-y samples, then for each trial and bin picks the single nearest camera frame to the bin center and discretizes that frame’s y-value against those percentile thresholds. There is no likelihood filtering, no within-bin averaging, and no explicit missing-data class.

ii. 
```python
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
```

```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
...
tongue_y_values = tongue_y_all_data[tongue_indices]
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. The notes claim a different procedure: valid tongue frames only, high-likelihood filtering, and session percentiles, but they do not match the implementation.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. In the code, bins start in class 0, then are promoted to class 1 if the selected y-value is at or above the session 40th percentile, and to class 2 if it is strictly above the 60th percentile. No fourth “not visible” class is created.

ii. 
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. The notes describe three classes low/mid/high, but their percentile numbers and handling of invalid frames conflict with the actual code.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue position to neural time bins by computing absolute times for each neural bin center (`go_time + BIN_CENTERS`) and assigning each bin the nearest camera frame in time.

ii. 
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
...
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]
```

iii. No detailed justification is documented beyond wanting a time-varying tongue signal aligned to the same bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles missingness by exclusion. Sessions are dropped if behavioral performance is too low or if no units survive curation. Units are dropped if they are not `classification == 'good'` or if their annotation cannot be mapped to one of 14 broad regions. Trials are dropped if they fall outside inferred `obs_intervals`, or if they are `auto_water` or `free_water`. For tongue data, however, there is no explicit handling of low-likelihood or missing frames in code.

ii. 
```python
if performance < MIN_PERFORMANCE:
    ...
    return None
...
if n_good == 0:
    ...
    return None
```

```python
region = map_anno_to_region(anno_names[ui])
if region is not None:
    good_indices.append(ui)
```

```python
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. The notes explicitly justify exclusion-based handling for low-performing sessions, unmapped units, and filtered trials. They also claim “Trials with no valid tongue data default to class 0 (low),” but the code never actually checks tongue validity.

## 10-a. What are the most time-consuming steps of the code?

i. The code structure implies that the most time-consuming steps are opening and reading every NWB file, preloading spike times for all kept units, the nested trial-by-unit neural histogram loop, the per-trial scan over all photostimulation intervals, and the per-trial nearest-frame tongue lookup.

ii. 
```python
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

```python
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        ...
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    ...
```

iii. There is no explicit runtime analysis in the notes. This is inferred from the code path the AI chose.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized or reduced: the unit-filtering loop over all units, the trial loop that builds `valid_trial_mask`, the per-trial neural loop with an inner per-unit histogram, the per-trial photostimulation loop over every event in the session, and parts of the per-trial tongue alignment.

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
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        ...
```

iii. The notes do not discuss vectorization opportunities.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several computations across trials: it scans the entire session’s photostimulation event list for every trial, recomputes nearest camera-frame assignment for every trial, and fills the same per-trial scalar outputs across all 80 bins every time. It also uses `subjects.index(...)` repeatedly during assembly.

ii. 
```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    ...
```

```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
...
```

```python
np.full(N_BINS, choice, dtype=np.int64)
np.full(N_BINS, outcome_val, dtype=np.int64)
np.full(N_BINS, early_lick_val, dtype=np.int64)
```

```python
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. No justification for the repeated work is documented in the notes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session performance metrics and left/right correct counts for every session mainly to enforce its extra session filter and to print/report them, even though those values are not decoder variables. It also performs a broad 14-region remapping that discards original anatomical specificity in `anno_name`.

ii. 
```python
n_correct = ((outcome == 'hit') & control_mask_all).sum()
performance = n_correct / n_control
correct_left = ((outcome == 'hit') & (trial_instruction == 'left') & control_mask_all).sum()
correct_right = ((outcome == 'hit') & (trial_instruction == 'right') & control_mask_all).sum()
```

```python
region = map_anno_to_region(anno_names[ui])
if region is not None:
    good_indices.append(ui)
    unit_regions.append(region)
```

iii. The notes explicitly justify the extra session filter and region remapping as matching the paper’s analysis choices, even though both go beyond the human reference conversion pipeline used for this task.
