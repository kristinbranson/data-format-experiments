# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates sorted `sub-*` directories and every sorted `.nwb` file beneath each, opens each with `NWBHDF5IO`, and reads trials, units, behavioral events, behavioral time series, and subject/session metadata. Only sessions surviving later selection enter the output.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The trajectory shows the agent inspected the NWB layout and treated one NWB file as one session. It reported a full result of 105 sessions, 25 subjects, and 48,356 trials after its selection rules.

## 1-b. How are the data split into subjects?

i. Directory names identify which files to visit, but the saved subject label is `nwb.subject.description` (mouse names such as `SC015`). Unique labels are retained in first-surviving-session order, and each session gets an integer index.

ii.
```python
subject_id = nwb.subject.description
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions])
```

iii. The code comment says the description contains the paper-style mouse name. The trajectory indicates this was chosen after inspecting subject metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; a returned `process_session` result becomes one output-session entry. Sessions failing performance, trial-count, or unit QC are omitted.

ii.
```python
for fname in files:
    result = process_session(fpath, sample_mode=sample_mode)
    if result is not None:
        all_sessions.append(result)
```

iii. The agent followed the file-level NWB organization, then justified additional session selection as matching the method paper: performance above 65% and at least 50 correct control trials per side.

## 1-d. How are the data split into trials?

i. Rows of `nwb.trials` are trials. Neural observation intervals from the first good unit are mapped to rows by position when counts match, otherwise by nearest trial start within 0.5 s; go cues are indexed by trial-row index.

ii.
```python
if n_obs == n_trials:
    return list(range(n_trials))
diffs = np.abs(trial_starts - obs[i, 0])
if diffs[min_idx] < 0.5:
    obs_to_trial.append(int(min_idx))
```

iii. The agent reasoned that `obs_intervals` identifies trials having neural data and added a tolerant start-time match for sessions where counts differ.

## 1-e. How are trials filtered based on quality controls?

i. Trials must map to an observation interval, must have `early_lick == 'no early'`, and must not have `outcome == 'ignore'`. Before this, whole sessions must exceed 65% performance on control/non-early trials and contain at least 50 correct left and right trials. Fewer than two survivors drops a session.

ii.
```python
if perf <= MIN_PERF or correct_left < 50 or correct_right < 50:
    return None
if i not in trials_with_neural: continue
if early != 'no early': continue
if outcome == 'ignore': continue
```

iii. The agent cited the method statement that early-lick and no-response trials were excluded for analysis and adopted the method paper’s session criteria. It did not reconcile those exclusions with the requested early-lick and no-lick decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `nwb.units['spike_times']` for units whose `classification` is `good`, with `obs_intervals` used to limit each trial and `go_start_times` used for alignment.

ii.
```python
all_spike_times = [nwb.units['spike_times'][uid] for uid in good_indices]
obs_intervals = units['obs_intervals'][good_indices[0]]
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The agent identified sorted spike times as the available neural signal and classifier-good units as the paper’s QC population.

## 2-b. How is the `neural` data processed?

i. For each trial and neuron, spikes are restricted to its observation interval, histogrammed over 80 go-aligned bins, and divided by 0.05 s to yield Hz. There is no smoothing or normalization.

ii.
```python
counts, _ = np.histogram(st_window, bins=bin_edges)
fr[i, :] = counts.astype(np.float32) / bin_width
```

iii. The agent explicitly chose spike counts divided by bin width, matching the requested firing-rate computation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units labeled `classification == 'good'` are retained; sessions with none are skipped. No individual metric thresholds or `unit_quality` filter are applied.

ii.
```python
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)
if len(good_indices) == 0:
    return None
```

iii. The agent connected this classification to the classifier-based QC in the supplied spike-sorting paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges span -2.5 to +1.5 s around each indexed go-cue timestamp.

ii.
```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

iii. The code and trajectory explicitly state alignment to go-cue onset, as requested.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins over four seconds. Raw spikes are histogrammed directly into these bins; no later rebinning is applied.

ii.
```python
BIN_WIDTH = 0.05
n_bins = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
```

iii. The agent selected the exact resolution specified in the decoder task.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial `start_time`, the trial go cue, and neural bin centers.

ii.
```python
valid_samples = sample_start_times[(sample_start_times >= t_start) & (sample_start_times < go_cue)]
tone_onset = valid_samples[-1]
time_from_tone = bin_centers - (tone_onset - go_cue)
```

iii. The agent interpreted the last sample-start event before go as the actual tone onset and supplied a typical-duration fallback if none exists.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The last in-trial sample onset is subtracted from each absolute bin center through a go-relative offset. If no sample event is found, tone onset is imputed as 1.85 s before go.

ii.
```python
tone_onset = valid_samples[-1] if len(valid_samples) else go_cue - 1.85
time_from_tone = bin_centers - (tone_onset - go_cue)
```

iii. The agent said the final sample start was the real sample epoch and called 1.85 s the typical sample-plus-delay duration.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the exact centers of the same 80 go-aligned bins used for spike histograms.

ii.
```python
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
time_from_tone = bin_centers - tone_onset_rel
```

iii. Using the shared bin centers was intended to guarantee one input value per neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It comes from trial `photostim_onset`, `photostim_duration`, `start_time`, and the go-cue timestamp.

ii.
```python
photostim_start_abs = t_start + float(photostim_onset_trial)
photostim_stop_abs = photostim_start_abs + float(trials['photostim_duration'][trial_idx])
```

iii. The agent recognized that stored onset is relative to trial start and must be converted to the go-aligned clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Nonstimulated (`'N/A'`) trials remain all zero. Otherwise bins whose centers fall in the half-open stimulation interval are set to one.

ii.
```python
photostim_on = np.zeros(len(bin_centers), dtype=np.float32)
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The agent chose the requested binary, time-varying representation.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation bounds are expressed relative to the same go cue, then compared with neural bin centers.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
```

iii. The shared go-relative grid was the agent’s alignment rationale.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial `trial_instruction` and `outcome`: a hit uses the instructed side and a miss flips it. Ignore trials have already been removed.

ii.
```python
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
```

iii. The agent reasoned that hit/miss plus instructed side determines the actual lick direction, but omitted the required no-lick class by filtering ignores.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left and right are encoded 0 and 1, then repeated across all 80 output bins. A fallback maps any unexpected outcome to left, though filtered data should not reach it.

ii.
```python
output_trial[0, :] = base_output[0]
'output_values': [['left', 'right'], ...]
```

iii. The trajectory describes choice as a binary decoder target after the agent’s no-response exclusion.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` value, although ignore rows are excluded before conversion.

ii.
```python
outcome = trials['outcome'][trial_idx]
```

iii. The agent noted that NWB already provides outcome categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2 and the value is repeated across 80 bins. Because ignore trials are filtered, the saved dataset only contains codes 1 and 2.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
output_trial[1, :] = base_output[1]
```

iii. The mapping follows the requested label order; the missing ignore observations are a consequence of the agent’s trial-curation decision.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from trial `early_lick`, but only `no early` trials survive.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. The agent found the explicit field but prioritized the method paper’s exclusion over preserving this requested target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and anything else to 1, repeated across all bins. In practice every saved value is zero.

ii.
```python
output_trial[2, :] = base_output[2]
```

iii. The binary coding matches the requested ordering, while the upstream filter makes the target degenerate.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking` timestamps and data columns 1 (y) and 2 (likelihood), plus go cues for trial windows.

ii.
```python
tongue_data = tongue_ts.data[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The agent identified the side-camera tracking series as the relevant behavioral stream.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood at most 0.5 are discarded; remaining y values are averaged in each 50-ms trial bin. Session percentiles are computed from valid bin means across retained trial windows, not from a whole-session grid.

ii.
```python
lh_mask = tw_lh > 0.5
tongue_y_bins[b] = np.mean(tw_y_good[bin_assignments == b])
p40 = np.percentile(valid_tongue, 40)
p60 = np.percentile(valid_tongue, 60)
```

iii. The agent sought to exclude low-confidence tracking and define thresholds on the same binned quantity ultimately classified.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible values below p40 map to 0, p40 through p60 to 1, and above p60 to 2. Critically, NaN/no-visible bins also map to 0; only three labels (`low`, `mid`, `high`) are declared, so the required category 3 is absent.

ii.
```python
if np.isnan(val): result[i] = 0
elif val < p40: result[i] = 0
elif val <= p60: result[i] = 1
else: result[i] = 2
```

iii. The function docstring explicitly says NaN is treated as low position, despite the task’s explicit separate `not visible` class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps in the go-centered window are digitized into the same 80 absolute bin edges used for neural data, with a one-bin padding only for the initial frame query.

ii.
```python
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

iii. The agent intentionally used the identical go-aligned bin grid for video and spikes.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with zero classifier-good units and sessions/trials failing selection are dropped. Missing tone events are imputed at go-1.85 s. Missing tongue bins are initially NaN but later converted to low class 0. Missing annotations become `unknown`. NWB warnings are globally suppressed.

ii.
```python
warnings.filterwarnings('ignore')
tone_onset = go_cue - 1.85
if anno is None or anno == '' or anno == 'nan': anno = 'unknown'
if np.isnan(val): result[i] = 0
```

iii. The trajectory shows pragmatic fallbacks intended to keep conversion running, but it did not preserve missing tongue visibility as the requested explicit class.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is repeated per-trial/per-unit spike masking and histogramming, loading large NWB spike/video arrays, per-bin tongue masks, writing the multi-gigabyte pickle, and later decoder training. The agent optimized by preloading spike arrays and reported the full conversion completing successfully.

ii.
```python
for trial_idx in valid_trial_indices:
    spike_times_trial = get_spike_times_for_trial_from_cache(...)
    fr, bin_centers = compute_firing_rates(...)
```

iii. Trajectory notes show the agent ran an “optimized” full conversion and emphasized preloading spike times; its final report gives a 5.9-GB product.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Unit classification, performance counting, observation matching, per-trial spike extraction, per-trial/per-neuron histograms, per-bin tongue means, tongue discretization, region mapping, and output construction are Python loops. In particular, neural binning could search all trial edges per unit once, as the reference does.

ii.
```python
for trial_idx in valid_trial_indices:
    for st in all_spike_times:
        mask = (st >= t_start) & (st < t_stop)
    for i, st in enumerate(spike_times_by_neuron):
        counts, _ = np.histogram(...)
```

iii. The agent vectorized some camera filtering and used `digitize`, but left the dominant nested spike loops despite its speed-focused rewrite.

## 10-c. What processing does the code repeat multiple times?

i. Each unit’s complete spike array is scanned once per retained trial to crop to `obs_intervals`, then scanned again to crop to the go window. Bin edges are rebuilt per trial; tongue masks scan session timestamps per trial; list searches repeatedly derive subject and region indices.

ii.
```python
mask = (st >= t_start) & (st < t_stop)
mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

iii. The trajectory’s optimization rationale focused on caching HDF5 reads, but it did not eliminate repeated in-memory scans.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It builds an observation-index lookup and slices spikes to the entire observation interval before immediately re-slicing to the smaller decoder window. It also computes session performance and correct-side counts solely to discard sessions, and computes/returns session IDs and performance that are not placed in the final dictionary.

ii.
```python
obs_idx = trial_to_obs[trial_idx]
spike_times_trial = get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx)
return {..., 'session_id': session_id, 'perf': perf}
```

iii. The session statistics were deliberate selection checks; the intermediate trial spike slicing was introduced as a caching/organization strategy rather than justified as necessary downstream data.
