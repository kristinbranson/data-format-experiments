# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every NWB session with a sorted glob, opens each with `NWBHDF5IO`, reads the NWB object, and catches per-file exceptions so processing can continue. It also supports limiting the number of files.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
for nwb_file in nwb_files:
    result = process_session(nwb_file, verbose=verbose)
```
```python
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
trials = nwb.trials.to_dataframe()
```

iii. The trajectory identifies NWB as the supplied format, reports 174 files and 28 subjects, and treats one file as one recording session. It chose sorted discovery for deterministic full-dataset processing.

## 1-b. How are the data split into subjects?

i. Each session reads both the numeric `subject_id` and descriptive mouse name. The output uses first-seen unique `subject.description` values as `subjects`, with `subject_idx` locating each session in that list.

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description
...
if sub_desc not in all_subjects:
    all_subjects[sub_desc] = result['subject_id']
subjects = list(all_subjects.keys())
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. The trajectory recognized 28 animals and used human-readable mouse labels such as `SC015`, while retaining the numeric ID internally.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Surviving session dictionaries are appended in sorted-file order; sessions failing behavioral criteria, unit criteria, or the two-trial minimum are omitted.

ii.
```python
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
    if result is not None:
        all_sessions.append(result)
```

iii. The agent inferred the session boundary from the DANDI layout. The trajectory additionally says it applied paper-reported performance criteria, reducing 174 files to 144 output sessions.

## 1-d. How are the data split into trials?

i. Rows of `nwb.trials` define trials and are paired one-to-one with `go_start_times`. Valid trial indices are used to create one neural, input, and output array per trial.

ii.
```python
trials = nwb.trials.to_dataframe()
go_start_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_start_times) == n_trials_total
...
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
```

iii. The trajectory found one go cue per trial and discovered that repeated sample events from early licks cannot safely define trial boundaries, so it used the trials table.

## 1-e. How are trials filtered based on quality controls?

i. Before trial selection, the agent filters whole sessions to at least 65% performance and at least 50 correct control trials on each side. Within retained sessions it keeps trials covered by `obs_intervals`, excludes both `auto_water` and `free_water`, and requires at least two trials. Early-lick and ignore trials are retained.

ii.
```python
control_mask_all = (all_valid & (early_lick == 'no early') &
                    (outcome != 'ignore') & no_photostim)
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

iii. The agent cited the method paper's behavioral-session criteria. It deliberately retained early-lick and no-response trials because they are requested decoder targets, and fixed an initial `obs_intervals` bug after finding recordings that began partway through behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each selected unit's `units/spike_times`, with `go_start_times` supplying trial alignment and unit classification/annotation supplying curation and region metadata.

ii.
```python
st = units.get_unit_spike_times(ui)
...
go_time = go_start_times[trial_idx]
```

iii. The trajectory established that spike and event timestamps use absolute session time and that spike times are the raw neural representation available in NWB.

## 2-b. How is the `neural` data processed?

i. For every trial and good unit, spikes in the four-second window are located with `searchsorted`, histogrammed into 80 bins, and divided by 0.05 s to yield firing rates in Hz. There is no smoothing or normalization.

ii.
```python
lo = np.searchsorted(st, abs_start)
hi = np.searchsorted(st, abs_end)
rel_spikes = st[lo:hi] - go_time
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The agent selected nonoverlapping 50-ms rate bins to implement the explicit decoder requirement and verified the arrays with the supplied decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and an annotation that the custom string rules map into one of 14 broad regions. A session with no such units is dropped.

ii.
```python
if classification[ui] != 'good':
    continue
region = map_anno_to_region(anno_names[ui])
if region is not None:
    good_indices.append(ui)
```

iii. The trajectory connects `classification` to the white-paper classifier and built a 14-region mapping to resemble the analysis repository. It noted 69,453 classifier-good units but retained only 57,935 after its session and annotation choices.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute go-cue time is added to the requested relative window. Spikes are converted to go-relative times and binned on the common relative edges.

ii.
```python
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
rel_spikes = st[lo:hi] - go_time
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The agent determined that NWB streams share a clock, so subtraction of each trial's go time is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins from -2.5 to +1.5 seconds. Raw spikes are binned directly; no later rebinning is applied.

ii.
```python
BIN_WIDTH = 0.05
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
```

iii. These values directly implement the task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The implemented input is not derived from a per-trial raw tone event. It assumes tone onset is always 1.85 seconds before the go cue and combines that constant with bin centers.

ii.
```python
TONE_ONSET_REL = -1.85
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. The trajectory observed that ordinary sample-to-go timing is 0.65 s plus 1.2 s and chose the constant offset, despite also observing extra `sample_start_times` caused by early-lick replay.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. It subtracts the fixed relative tone time from each bin center. The resulting identical 80-value vector is reused for every trial.

ii.
```python
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The agent justified this by the nominal fixed sample and delay durations and did not resolve replayed epochs from the event stream.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the same go-relative bin centers as the neural histograms and have the same 80 columns.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. The shared bin grid was chosen to keep all time-varying inputs aligned with firing rates.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses the session event streams `photostim_start_times` and `photostim_stop_times`.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The agent chose the absolute event timestamps because they share the neural clock and directly describe light-on intervals.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each trial begins as zeros. Every session photostimulation interval overlapping its window marks bin centers in the half-open interval `[start, stop)` as one.

ii.
```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        mask = (BIN_CENTERS >= ps_start - go_time) & (BIN_CENTERS < ps_stop - go_time)
        photostim_binary[mask] = 1.0
```

iii. The trajectory describes photostimulation as a binary time-varying decoder input and used bin centers for consistency.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation bounds are shifted by the same trial go time and compared with the neural bin centers.

ii.
```python
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. The common NWB timestamp clock and bin grid provide the alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trials-table `trial_instruction` and `outcome`: hit uses the instructed side, miss uses the opposite side, and ignore incorrectly uses the instructed side.

ii.
```python
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:
    choice = 0 if instr == 'left' else 1
```

iii. The trajectory explicitly wrestled with no-response choice, but ultimately forced ignored trials into left/right because it interpreted the requested output as having only those two classes.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left/right are coded 0/1 and the per-trial code is broadcast across 80 bins. `output_values` contains only two choice labels.

ii.
```python
np.full(N_BINS, choice, dtype=np.int64)
...
['left', 'right']
```

iii. The agent made all categorical outputs integer arrays after validation exposed problems with floating indices, and broadcast scalars so outputs share a uniform time dimension.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is read directly from the trials-table `outcome` column.

ii.
```python
outcome = trials['outcome'].values
outc = outcome[trial_idx]
```

iii. The raw field already contains exactly the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to ignore=0, miss=1, hit=2 and broadcast over all bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The mapping follows the requested coding; broadcasting provides consistent output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is read from the trials-table `early_lick` field.

ii.
```python
early_lick = trials['early_lick'].values
```

iii. The agent retained these trials specifically because early lick is a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` becomes 1 and everything else becomes 0, then the value is broadcast over 80 bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
np.full(N_BINS, early_lick_val, dtype=np.int64)
```

iii. This implements the requested no/yes codes and common output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 of `Camera0_side_TongueTracking.data` and its timestamps. Although the stream also contains tracking likelihood, the code does not read or use it.

ii.
```python
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. The trajectory identified the stream as `(x, y, confidence)` but implemented only y and timestamps.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. It calculates session percentiles over every raw y frame, selects the single nearest camera frame to each neural bin center, and discretizes that value. It neither averages frames within 50-ms bins nor filters low-confidence tracking.

ii.
```python
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
...
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. The agent described nearest-frame lookup as vectorized and noticed at least one extremely skewed tongue distribution, but did not revise this processing.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values below the 40th raw-frame percentile are 0, values from the 40th through 60th percentile are 1, and values above the 60th are 2. There is no not-visible category.

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. The agent followed the three percentile bands literally, but omitted visibility despite discovering a confidence channel.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For every go-relative neural bin center, the code finds the nearest timestamp anywhere in the camera stream and uses that frame's y value.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
...
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]
```

iii. The agent relied on the shared absolute clock. Nearest-frame sampling was chosen for speed, rather than matching the neural bin interval.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Files that raise any exception are logged and skipped. Sessions with no mapped good units, no control trials, failed behavioral criteria, or fewer than two valid trials are skipped. Indices for tongue timestamps are clipped at stream ends, effectively filling out-of-range requests with the nearest endpoint. No explicit missing/low-confidence tongue class is created.

ii.
```python
try:
    result = process_session(nwb_file, verbose=verbose)
except Exception as e:
    print(f"  ERROR: {e}")
    continue
```
```python
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
```

iii. The trajectory shows iterative diagnostics for zero-neural trials and fixes trial coverage by matching observation timestamps. Other anomalies, including skewed tongue tracking and the last all-zero trial, were noted but retained.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant work is the nested trial-by-unit spike histogramming, repeated scans of photostimulation events for every trial, NWB reads, and writing the roughly 9.3-GB pickle. Full conversion logs were used to monitor progress.

ii.
```python
for trial_idx in valid_indices:
    for i, st in enumerate(all_spike_times):
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The trajectory calls preloading spike arrays faster than per-access NWB reads and validates first on sample data before running the full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-unit neural loop could vectorize all trials for each unit, as the reference does. Trial coverage construction can be vectorized, and the trial-by-all-photostim-events loop could instead use per-trial fields or vectorized interval comparisons. Region mapping and session assembly loops are comparatively minor.

ii.
```python
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```
```python
for trial_idx in valid_indices:
    for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
```

iii. The agent explicitly optimized nearest-frame tongue lookup and preloaded spike times, but left the largest neural computation nested by trial and unit.

## 10-c. What processing does the code repeat multiple times?

i. It scans the complete photostimulation event arrays for every trial, constructs repeated scalar output arrays per trial, performs separate spike histogram setup for every unit/trial pair, repeatedly retrieves all good units' observation intervals, and uses linear list lookup for subject assembly.

ii.
```python
for ui in good_indices[1:]:
    oi = units.get_unit_obs_intervals(ui)
...
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. The trajectory focused on correctness fixes and dataset validation; it did not document these remaining repeated computations as a concern.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes `photostim_power` only to decide session-level control trials, stores both subject identifiers although only the description reaches the output, records several summary fields used only for logs/metadata, builds and writes an additional five-session sample dataset, and imports unused `sys` and `json`. The broad-region remapping also discards fine annotation detail.

ii.
```python
photostim_power = trials['photostim_power'].values
...
return {'subject_id': subject_id, 'subject_desc': subject_desc,
        'performance': performance, 'correct_left': correct_left,
        'correct_right': correct_right, ...}
```

iii. The sample output and summaries were intentionally added for verification. The trajectory reports them as required workflow artifacts, although they are not used by the downstream full-dataset decoder.
