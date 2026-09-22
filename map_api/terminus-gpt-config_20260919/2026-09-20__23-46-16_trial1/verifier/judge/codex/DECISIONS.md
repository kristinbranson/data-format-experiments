# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every NWB file under `/app/data`, sorts the paths, opens each file once with `pynwb.NWBHDF5IO`, and reads the session’s trials, units, events, and behavioral time series. Full mode processes all files; sample mode stops after two usable sessions.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
for path in files:
    r=process_file(path, make_plot=args.show_processing and len(results)<2)
```
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read(); trials = nwb.trials; units = nwb.units
```

iii. The agent justified this as exhaustive deterministic loading of the one-NWB-per-session archive, while complying with the requirement to use `pynwb`. It reported 174 source NWBs and 173 usable sessions.

## 1-b. How are the data split into subjects?

i. Each session’s subject is read from `nwb.subject.subject_id`. After processing, unique IDs are sorted and each session receives an integer index into that vocabulary.

ii.
```python
subject=str(nwb.subject.subject_id)
subjects=sorted({r['subject'] for r in results})
subject_idx=np.asarray([subject_map[r['subject']] for r in results],dtype=np.int64)
```

iii. The agent treated the NWB subject field as canonical and verified that it produced 28 mice.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Output session order is sorted path order; `nwb.identifier` and source-file metadata preserve session identity.

ii.
```python
session_id = nwb.identifier
return dict(..., session_id=session_id, source_file=str(path), ...)
```

iii. The notes state that the archive’s file boundary is the session boundary. The sole file with no classifier-good units is skipped.

## 1-d. How are the data split into trials?

i. Trial rows come from `nwb.trials`; go-cue events are required to have the same length. Retained source row indices are used consistently for neural, input, and output arrays.

ii.
```python
all_go = np.asarray(ev['go_start_times'].timestamps[:], dtype=np.float64)
if len(all_go) != len(trials):
    raise ValueError(f'{session_id}: go/trial count mismatch')
trial_idx = np.flatnonzero(trial_keep)
```

iii. The agent considered the NWB trials table authoritative and used the one-go-cue-per-row check to ensure unambiguous alignment.

## 1-e. How are trials filtered based on quality controls?

i. It excludes auto-water and free-water trials, then intersects `is_good_trials` over every retained classifier-good unit. Short validity vectors are mapped to trials whose go cues lie in each unit’s `obs_intervals`. It additionally removes trials whose entire retained neural cube is zero and drops sessions with fewer than two trials.

ii.
```python
trial_keep = ~(auto | free)
for ui in neuron_idx:
    stored = np.asarray(units['is_good_trials'][ui], dtype=bool)
    ...
    trial_keep &= valid
nonzero_neural = np.any(rate_cube != 0, axis=(1, 2))
```

iii. The notes describe assisted-water trials as nonstandard/confounded, require every retained neuron to be valid, and call the all-zero test a recording-boundary safeguard discovered during review. Early-lick, ignore, miss, hit, control, and stimulation trials are intentionally retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from `units['spike_times']` for units whose `classification` is `good`, with `BehavioralEvents/go_start_times` defining absolute bin edges.

ii.
```python
neuron_idx = np.flatnonzero(classes == 'good')
spikes = [np.asarray(units['spike_times'][int(i)], dtype=np.float64) for i in neuron_idx]
abs_edges = go[:, None] + EDGES_REL[None, :]
```

iii. The agent identified spike times as the raw neural representation and the published classifier verdict as the appropriate unit QC.

## 2-b. How is the `neural` data processed?

i. For each neuron, cumulative spike indices at all flattened bin edges are computed with `searchsorted`; adjacent differences yield counts, which are divided by 0.05 s to produce float32 Hz. No smoothing or normalization is applied.

ii.
```python
cumulative = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, n_edges)
rates[:, j, :] = np.diff(cumulative, axis=1) / BIN_S
```

iii. The notes say this is an exact, vectorized-across-trials histogram matching the paper’s conversion from counts to firing rate while using the task-required bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units.classification == 'good'` are retained; a session with no such units is skipped. The older `unit_quality` field and individual metric thresholds are not used.

ii.
```python
classes = as_strings(units['classification'])
neuron_idx = np.flatnonzero(classes == 'good')
if len(neuron_idx) == 0:
    return None
```

iii. The agent says this matches the region-specific classifier QC used in the papers and avoids the broader legacy label. It retained 69,453 session-neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 s are added to each trial’s absolute go-cue timestamp, then absolute spike times are histogrammed within those edges.

ii.
```python
go = all_go[trial_idx]
abs_edges = go[:, None] + EDGES_REL[None, :]
rate_cube = bin_spikes(spikes, abs_edges)
```

iii. The agent notes that spikes and events already share the NWB session clock, so no clock correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins over `[-2.5, 1.5)` s. Raw spike timestamps are binned directly; no subsequent temporal rebinning is performed.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
N_TIME = 80
```

iii. This directly follows the decoder specification and intentionally replaces the paper code’s 100-ms sliding window/50-ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses trial start and go time plus `sample_start_times`, `sample_stop_times`, `delay_start_times`, and `delay_stop_times` to choose a tone onset from the state-machine sequence.

ii.
```python
all_tone = choose_tone_onsets(starts, all_go, sample_starts, sample_stops,
                              delay_starts, delay_stops)
```

iii. The agent reasoned that early licks restart sample/delay states, so the tone belonging to the final pre-go delay should be selected explicitly from the state chain.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, it selects the delay whose stop is nearest the go cue, then the sample whose stop is nearest that delay’s start. It subtracts the resulting sample onset from every absolute bin center.

ii.
```python
dk = didx[np.argmin(np.abs(delay_stops[didx] - go))]
sk = sidx[np.argmin(np.abs(sample_stops[sidx] - delay_starts[dk]))]
out[i] = sample_starts[sk]
tone_time = abs_centers - tone[:, None]
```

iii. The notes say this handles repeats and variable delays semantically and found 183 trials where this differed from a simpler nominal-onset rule.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same absolute 50-ms bin centers as the go-aligned neural bins.

ii.
```python
abs_centers = go[:, None] + CENTERS_REL[None, :]
tone_time = abs_centers - tone[:, None]
```

iii. The shared bin-center grid guarantees a one-to-one timepoint correspondence with neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, plus trial start/stop boundaries to associate at most one interval with each trial.

ii.
```python
ps = np.asarray(ev['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(ev['photostim_stop_times'].timestamps[:], dtype=np.float64)
hits = np.flatnonzero((ps >= a) & (ps <= b))
```

iii. The agent chose the explicit event intervals and validates that a trial never has multiple intervals.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each bin is 1 when its center is at or after the stimulation start and before its stop, otherwise 0. Nonstimulated trials remain all zero.

ii.
```python
stim[k] = ((abs_centers[k] >= ps[h]) &
           (abs_centers[k] < pe[h])).astype(np.float32)
```

iii. Center sampling was justified as consistent with representing each 50-ms interval by one timepoint.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute photostimulation intervals are compared with the same go-aligned absolute centers used by neural bins.

ii.
```python
abs_centers = go[:, None] + CENTERS_REL[None, :]
stim[k] = ((abs_centers[k] >= ps[h]) & (abs_centers[k] < pe[h]))
```

iii. All streams share the NWB session clock, so the agent applied no extra offset.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trials.trial_instruction` and `trials.outcome`.

ii.
```python
instruction = as_strings(trials['trial_instruction'])
outcome = as_strings(trials['outcome'])
arr[0] = trial_choice(instruction[src_i], outcome[src_i])
```

iii. The agent reasoned that hit means the instructed side, miss the opposite side, and ignore no response.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Hit maps to the instructed direction, miss to its opposite, and ignore to no lick; codes are left=0, right=1, no lick=2 and are repeated over 80 bins.

ii.
```python
if outcome == 'ignore': return 2
if outcome == 'hit': return 0 if instruction == 'left' else 1
if outcome == 'miss': return 1 if instruction == 'left' else 0
```

iii. Repetition lets trial-level outputs share a `(4,80)` array with time-varying tongue position.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is read directly from `trials['outcome']`.

ii.
```python
outcome = as_strings(trials['outcome'])
```

iii. The raw column already contains the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped as ignore=0, miss=1, hit=2 and repeated across all time bins.

ii.
```python
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
arr[1] = omap[outcome[src_i]]
```

iii. The ordering follows the requested output-value order; repetition supports the common output matrix.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `trials['early_lick']`.

ii.
```python
early = as_strings(trials['early_lick'])
```

iii. The explicit NWB flag is used, and early-lick trials are retained because early lick is a requested target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and every other retained value maps to 1; the label is repeated across 80 bins.

ii.
```python
arr[2] = 0 if early[src_i] == 'no early' else 1
```

iii. The agent intended the requested no/yes coding and used repetition for output-shape consistency.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and data from `Camera0_side_TongueTracking`; data column 1 is y-position and column 2 is tracking likelihood.

ii.
```python
tongue = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
video_t = np.asarray(tongue.timestamps[:], dtype=np.float64)
video_d = np.asarray(tongue.data[:], dtype=np.float64)
```

iii. The notes identify this as the side-camera tongue marker and use likelihood to distinguish visible tongue from placeholder coordinates.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood at least 0.9 define the visible session distribution. The 40th/60th percentiles are computed from their raw y-values. For every trial-bin center, the nearest camera frame is selected; its y and likelihood determine the class.

ii.
```python
visible_session = video_d[:, 2] >= VISIBILITY_THRESHOLD
q40, q60 = np.percentile(video_d[visible_session, 1], [40, 60])
frame_idx = nearest_indices(video_t, abs_centers.ravel())
ty = video_d[frame_idx, 1]; tl = video_d[frame_idx, 2]
```

iii. The agent chose 0.9 because likelihood is strongly bimodal, and nearest-frame sampling because the camera is much faster than the 50-ms output grid.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. A nearest frame below likelihood 0.9 is class 3. Visible y below q40 is 0, q40 through q60 inclusive is 1, and above q60 is 2. Thresholds are per session over raw visible frames.

ii.
```python
tongue_class = np.full((len(trial_idx), N_TIME), 3, dtype=np.int64)
tongue_class[vis & (ty < q40)] = 0
tongue_class[vis & (ty >= q40) & (ty <= q60)] = 1
tongue_class[vis & (ty > q60)] = 2
```

iii. This implements the requested four categories while excluding low-confidence marker locations from percentile estimation.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The nearest camera timestamp is found for each absolute go-aligned neural bin center.

ii.
```python
frame_idx = nearest_indices(video_t, abs_centers.ravel()).reshape(len(trial_idx), N_TIME)
```

iii. The shared absolute clock and high camera rate were considered sufficient; no interpolation or within-bin averaging was applied.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Zero-good-unit sessions and sessions falling below two trials are skipped. Short `is_good_trials` vectors are mapped through `obs_intervals`; mapping inconsistencies, missing tone/delay states, multiple stimulation intervals, absent visible tongue frames, and missing anatomy raise errors. Residual all-zero neural trials are removed, while invisible tongue samples receive category 3.

ii.
```python
if len(represented_idx) != len(stored): raise ValueError(...)
if not np.any(visible_session): raise ValueError(...)
nonzero_neural = np.any(rate_cube != 0, axis=(1, 2))
```

iii. The agent preferred explicit failure for structural inconsistencies, exclusion where neural data are unusable, and an explicit not-visible category where tongue absence is meaningful.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large spike/video arrays, per-neuron spike binning across all edges, accumulation of the roughly 11-GiB result, and pickle serialization are the principal costs. Full conversion took about 194 s, including about 14 s to serialize.

ii.
```python
spikes = [np.asarray(units['spike_times'][int(i)], dtype=np.float64) for i in neuron_idx]
for j, spikes in enumerate(spike_times):
    cumulative = np.searchsorted(spikes, flat_edges, side='left')
```

iii. The notes identify I/O and spike histogramming as dominant and report that implemented vectorization kept runtime well below the allotted threshold.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining loops include tone selection per trial, validity processing per neuron and per observation interval, spike search per neuron, stimulation association per trial, output construction per trial, and conversion of the rate cube to a list. Some trial loops could be vectorized; ragged per-neuron spikes make the neuron loop less straightforward.

ii.
```python
for i, (start, go) in enumerate(zip(trial_starts, go_times)):
for ui in neuron_idx:
for k, src_i in enumerate(trial_idx):
for j, spikes in enumerate(spike_times):
```

iii. The agent emphasized that the expensive trial dimension of spike binning and all trial-bin video lookup were already vectorized; it favored clear loops for ragged/stateful operations.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly scans event arrays for each trial during tone and stimulation association, repeatedly maps validity for each good unit, and builds per-trial neural/input/output objects after constructing session-level arrays. Data validation subsequently loops through every saved trial again.

ii.
```python
didx = np.flatnonzero((delay_starts >= start) & (delay_starts <= go))
hits = np.flatnonzero((ps >= a) & (ps <= b))
for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
```

iii. The notes do not identify these as problematic repetition; they present the converter as a single pass per NWB and regard validation as an intentional sanity check.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes/stores diagnostic provenance (`source_file`, source trial indices, q40/q60, elapsed time) for metadata and optionally creates plotting intermediates. It also calculates `abs_edges`/`abs_centers` before the all-zero filter and slices them afterward. Most scientific quantities are used downstream; plot-only arrays are only used when requested.

ii.
```python
return dict(..., source_trial_idx=trial_idx.tolist(),
            q40=float(q40), q60=float(q60), elapsed=elapsed)
if make_plot:
    plot_session(...)
```

iii. The agent considered provenance and plots valuable for reproducibility and sanity checking, and otherwise claimed that computed fields feed the output or validation.
