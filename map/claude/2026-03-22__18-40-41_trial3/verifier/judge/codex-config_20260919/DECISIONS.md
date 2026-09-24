# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `sub-*` directory and every `.nwb` file within it, then opens each file with `pynwb.NWBHDF5IO`. It processes files sequentially, catches per-file exceptions, and only appends sessions that pass `process_session`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sub-')])
...
nwb_files = sorted([os.path.join(sub_dir, f)
                    for f in os.listdir(sub_dir) if f.endswith('.nwb')])
...
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The notes identify the layout as 28 subject directories and 174 session NWBs and justify NWB as the available form of the published data. They report that behavioral session-selection rules subsequently reduced the result to 144 sessions.

## 1-b. How are the data split into subjects?

i. Subject directories provide the initial grouping; each session's definitive identifier is read from `nwb.subject.subject_id` (with a filename fallback). During assembly, subjects are retained in first-seen order and `subject_idx` is built by list lookup.

ii.
```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
...
if sess['subject_id'] not in subjects:
    subjects.append(sess['subject_id'])
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The notes treat the 28 `sub-*` directories and NWB subject metadata as the dataset's mouse identities; the final count of 28 is used as a sanity check.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session and identified by `nwb.identifier`. However, sessions are discarded if behavioral performance is below 65%, either side has fewer than 50 correct regular trials, no eligible neurons exist, fewer than two valid trials remain, or processing raises an exception. Thus only 144/174 files enter the output.

ii.
```python
session_id = nwb.identifier
...
if correct_rate < MIN_CORRECT_RATE:
    return None
if correct_left < MIN_CORRECT_LEFT or correct_right < MIN_CORRECT_RIGHT:
    return None
...
if result is None:
    n_skipped += 1
    continue
session_results.append(result)
```

iii. The agent inferred from the paper's reported performance range and trial-count criteria that these were session inclusion rules and tried to reconcile 174 archive files with 173 reported paper sessions. Its own notes acknowledge that this inference instead left only 144 sessions.

## 1-d. How are the data split into trials?

i. Rows of `nwb.trials` define trials. The agent reads each trial column by row index and asserts a one-to-one correspondence between trial rows and go-cue timestamps. It constructs one neural/input/output array per retained trial.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
...
go_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_times) == n_trials
...
for trial_idx in valid_indices:
    ...
    neural_trials.append(fr)
```

iii. The NWB trials table supplies the explicit trial boundary, while the go-event count check is the agent's sanity check that row-wise alignment is valid.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps trials with neither auto nor free water and with a discoverable tone onset. It additionally drops trials whose `go + 1.5 s` exceeds the last observation interval end by more than one second. It keeps early-lick, photostimulation, and ignore trials because they are needed as decoder variables. This is not the reference's exact `obs_intervals` membership filter.

ii.
```python
behav_valid = (auto_water == 0) & (free_water == 0)
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

iii. The notes say auto/free-water trials are nonstandard, whereas early lick, stimulation, and ignore must remain to satisfy the requested decoder targets. The extra end-time tolerance was intended to avoid windows beyond neural recording coverage; the notes later found 126 retained all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from each selected unit's `units['spike_times']`; `BehavioralEvents/go_start_times` supplies the absolute alignment time.

ii.
```python
spike_times_all = units['spike_times']
spike_times_good = [spike_times_all[idx] for idx in good_indices]
...
fr = compute_firing_rates_vectorized(spike_times_good, go_time, ...)
```

iii. The notes identify NWB spike times as absolute, unlike the already aligned MAT representation in the paper code, so each trial window must be positioned with its go cue.

## 2-b. How is the `neural` data processed?

i. For every trial and neuron, spikes in the four-second window are selected, assigned to nonoverlapping bins by floor division, counted with `np.add.at`, and divided by 0.05 s to obtain float32 firing rates in Hz. There is no smoothing or normalization.

ii.
```python
mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
spk_window = spk[mask]
bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
np.add.at(fr[i], bin_idx, 1)
fr /= bin_width
```

iii. The agent cites the reference's spike-histogram firing-rate processing, but uses the task-mandated 50 ms nonoverlapping bins rather than the paper code's 40 ms sliding window/3.4 ms stride.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and a nonempty/non-`None` `anno_name`. A session with none is dropped. No individual QC metric is thresholded.

ii.
```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
```

iii. The notes equate `classification == 'good'` with the paper's classifier QC and state that histology/CCF annotation is also required. They use agreement of mean neurons per retained session with the paper as validation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each trial, the absolute go timestamp is added to the fixed `[-2.5, 1.5)` window; absolute spikes are selected relative to that timestamp.

ii.
```python
go_time = go_times[trial_idx]
mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
```

iii. The notes explicitly distinguish NWB's absolute timestamps from the reference MAT files and say subtraction/offset by `go_start_time` is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The result has 80 nonoverlapping 50 ms bins over four seconds. Raw spike times are histogrammed directly into this grid; no subsequent rebinning occurs.

ii.
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)
```

iii. The agent correctly prioritizes the decoder specification's 50 ms bins over the reference analysis's 40 ms sliding window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial start times, and go-cue times. For each trial it selects the last sample onset between trial start and go cue.

ii.
```python
in_trial = sample_starts[(sample_starts >= trial_starts[i]) &
                         (sample_starts <= go_times[i])]
tone_onset_per_trial[i] = in_trial[-1]
```

iii. The agent recognized that early licks can replay the sample epoch, so the final tone before the go cue is the relevant onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. It computes each neural bin center relative to go and subtracts the tone's go-relative time, yielding seconds elapsed since tone onset.

ii.
```python
tone_relative = tone_time - go_time
time_from_tone = bin_centers - tone_relative
```

iii. The notes define the continuous input as current absolute/bin-center time minus tone onset; no other transformation is applied.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the exact same 80 go-relative bin centers as the neural firing rates.

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
time_from_tone = bin_centers - tone_relative
```

iii. The common go-cue-relative bin grid makes input column `b` contemporaneous with neural column `b`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus the trial go time.

ii.
```python
ps_onset_abs = trial_starts[trial_idx] + float(photostim_onset[trial_idx])
ps_end_abs = ps_onset_abs + float(photostim_duration[trial_idx])
ps_onset_rel = ps_onset_abs - go_time
```

iii. Exploration showed small onset values such as 1.8, which the agent interpreted as offsets from trial start rather than absolute times.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Nonstimulated trials remain all zero. On stimulated trials, a bin is set to 1 when its center is in the half-open stimulation interval; this is implemented with an explicit bin loop.

ii.
```python
photostim = np.zeros(N_BINS, dtype=np.float32)
...
for b in range(N_BINS):
    bc = bin_centers[b]
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. The task requests whether stimulation is on at each time point, so the agent chose a binary time series rather than a trial-level flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-start-relative onset is converted to absolute time and then go-relative time, and tested at the same bin centers used for neural data.

ii.
```python
ps_onset_rel = ps_onset_abs - go_time
ps_end_rel = ps_end_abs - go_time
bc = bin_centers[b]
```

iii. The shared go-relative centers align each binary photostimulation value with the corresponding firing-rate bin.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from `trial_instruction` and `outcome`: hit uses the instructed side, miss uses the opposite side, and—incorrectly—ignore also uses the instructed side despite there being no lick.

ii.
```python
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. The notes explicitly say that ignore has no actual lick but nevertheless choose the instruction direction as a proxy. This contradicts the requested `no lick` category.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left and right are coded 0/1 and repeated over all 80 time bins. Only two output labels are declared; there is no no-lick code.

ii.
```python
np.full(N_BINS, choice, dtype=np.int64)
...
'output_values': [
    ['left', 'right'],
```

iii. The agent repeated the per-trial value to combine it with the time-varying tongue output in a uniform 2-D output array. It justified instruction-side assignment on ignore trials only as a fallback.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is read directly from the trial-table `outcome` column.

ii.
```python
outcomes = trials['outcome'][:]
outcome = outcomes[trial_idx]
```

iii. The raw categories already match the requested ignore/miss/hit output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to `ignore=0`, `miss=1`, `hit=2`; the per-trial code is repeated across 80 bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. This fixed categorical map follows the requested ordering; repetition supplies a consistent time dimension.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` string.

ii.
```python
early_licks = trials['early_lick'][:]
```

iii. NWB already provides the trial-level early-lick flag, so the agent did not reconstruct it from lick timestamps.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'no early'` maps to 0 and every other value maps to 1, then the value is repeated across bins.

ii.
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The notes describe this as the direct no/yes mapping requested by the task.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 (y) and column 2 (tracking likelihood) of `Camera0_side_TongueTracking.data`, together with that series' timestamps. Likelihood affects session thresholds, but not whether individual output bins are visible.

ii.
```python
tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]
tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The notes identify side-camera y as the required measurement and likelihood >0.5 as visibility confidence.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session thresholds are calculated from raw visible-frame y values when more than 100 frames have likelihood >0.5; otherwise all raw y values are used. For each neural bin, the code takes the first camera frame at or after the bin center (described as “closest,” though it is not truly nearest), and discretizes its y. It neither averages frames over 50 ms nor emits a not-visible value.

ii.
```python
tongue_visible = tongue_likelihood > 0.5
visible_y = tongue_y[tongue_visible]
p40 = np.percentile(visible_y, 40)
p60 = np.percentile(visible_y, 60)
...
t_idx = np.searchsorted(tongue_ts, bc_abs)
ty = tongue_y[t_idx]
```

iii. The agent says per-session percentiles should use visible tongue positions. It interpreted temporal sampling as nearest-frame lookup and later treated a heavily skewed low class as expected tongue retraction, overlooking the mandated not-visible class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles define codes 0/1/2 with comparisons `<p40`, `<p60`, and otherwise. The implementation declares only low/mid/high; category 3 (`not visible`) is missing.

ii.
```python
if ty < p40:
    tongue_y_trial[b] = 0
elif ty < p60:
    tongue_y_trial[b] = 1
else:
    tongue_y_trial[b] = 2
...
['low', 'mid', 'high']
```

iii. The 40/60 percentiles and per-session scope follow the instructions, but the notes incorrectly conclude that three classes completely satisfy the specification.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Each neural bin center is converted to absolute time (`go + center`), and `searchsorted` selects the camera frame immediately at or after it. This aligns samples approximately at bin centers rather than aggregating the same 50 ms intervals as neural activity.

ii.
```python
bc_abs = go_time + bin_centers[b]
t_idx = np.searchsorted(tongue_ts, bc_abs)
t_idx = min(t_idx, len(tongue_ts) - 1)
```

iii. The agent relied on camera, spike, and event timestamps sharing the NWB clock. It called the selected frame “closest,” although the code does not compare the preceding frame.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing tone trials are removed; no-good-unit and too-few-trial sessions are skipped; file errors are caught and the session skipped. Missing tongue series defaults every bin to middle class 1, and too few visible frames causes thresholds to be computed from all frames. Missing annotations exclude units; unknown annotations fall back to `OtherCortex`.

ii.
```python
valid_mask &= ~np.isnan(tone_onset_per_trial)
...
except Exception as e:
    ...
    continue
...
else:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)
```

iii. The agent favored keeping conversion running and supplying fallbacks. Its notes call unknown-region fallback acceptable and characterize residual all-zero neural trials as negligible, but do not document the scientific risk of imputing missing tongue data as visible middle position.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant computation is neural binning: for every retained trial, it loops over every selected neuron and scans/masks that unit's complete spike array. NWB reading, camera-array loading, plotting when requested, training, and pickling the roughly 10 GB output also cost time.

ii.
```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(spike_times_good, go_time, ...)
...
for i, spk in enumerate(spike_times_list):
    mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
```

iii. The notes estimated 5–10 seconds per session and about 30 minutes for full conversion. They do not provide a profiler breakdown, but the nested trial/unit spike scans are visibly the principal algorithmic cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Tone lookup can use one `searchsorted` over all go cues; recording-coverage filtering is a vector comparison; photostimulation can compare all bin centers at once; and tongue lookup can search all centers at once. Most importantly, spike counts can be computed per unit for all trials at once using flattened edges and `searchsorted`, eliminating repeated full-spike masking. Region index lookup can use a dictionary.

ii.
```python
for i in range(n_trials): ...                 # tone and coverage
for trial_idx in valid_indices: ...           # trial processing
for i, spk in enumerate(spike_times_list): ...# neurons inside each trial
for b in range(N_BINS): ...                   # photostim and tongue
brain_regions.index(r)
```

iii. The function is named “vectorized,” but only counting within one neuron/trial uses NumPy; it does not vectorize across trials. The agent did not discuss these remaining loops in its notes.

## 10-c. What processing does the code repeat multiple times?

i. Each unit's entire spike vector is range-filtered again for every trial. Absolute bin boundaries and camera searches are rebuilt for every trial/bin. `brain_regions.index` is repeated for every neuron, and full-session summary lists traverse every output again after conversion.

ii.
```python
mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
...
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
```

iii. The notes do not acknowledge repeated processing; their runtime estimate implicitly accepts it. The reference implementation demonstrates that all trial edges can instead be searched together once per unit.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `trial_stops` but never uses it; computes `bin_edges_start` and `bin_edges_end` but never uses either; and computes/returns session `correct_rate` and detailed summary lists primarily for reporting. The large brain-region mapping and optional diagnostic plots support metadata/inspection but not decoder features.

ii.
```python
trial_stops = trials['stop_time'][:]
...
bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
bin_edges_end = bin_edges_start + bin_width
```

iii. The agent justifies diagnostics as sanity checks, but does not identify the unused edge arrays or trial-stop read. Optional plots are intentionally for validation and are not part of the saved dataset.
