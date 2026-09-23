# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All `.nwb` files under `/app/data` are discovered with a single recursive glob (`Path('/app/data').rglob('*.nwb')`), sorted lexicographically, and each file is converted independently in a sequential loop. Every file is opened once with `pynwb.NWBHDF5IO(..., load_namespaces=True)` and closed before the next one. From each file the AI reads (a) `nwb.subject.subject_id`, (b) `nwb.identifier` (used as the "scene" string), (c) the eight frame-aligned series of `processing['behavior']['BehavioralTimeSeries']`, (d) the event-based `Reward` series, and (e) the `processing['ophys']['Deconvolved']` `RoiResponseSeries` (one per imaging plane). `--sample` restricts the file list to the first two files; `--full` (default) processes all 152. No `h5py` access is used anywhere.

ii.
```python
DATA_ROOT = Path('/app/data')
...
files = sorted(DATA_ROOT.rglob('*.nwb'))
if args.sample: files = files[:2]
if not files: raise FileNotFoundError('No NWB files found')
...
for j, path in enumerate(files):
    sn, si, so, info = convert_session(path, args.show_processing and j < 2)
```
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
    subject = str(nwb.subject.subject_id)
    scene = str(nwb.identifier).rstrip('/').split('/')[-1]
    bts = nwb.processing['behavior']['BehavioralTimeSeries']
    needed = ['environment', 'lick', 'position', 'scanning', 'speed',
              'teleport', 'trial number', 'trial_start']
    arrays = {k: np.asarray(bts.time_series[k].data[:]).squeeze() for k in needed}
    deconv_container = nwb.processing['ophys']['Deconvolved']
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` contains 152 NWB files arranged as `sub-<id>/sub-<id>_ses-<nn>_behavior+ophys.nwb` (approximately one file per imaging session). All inspection used `pynwb.NWBHDF5IO`; `h5py` was not used." Step 9 records the census as a consistency check: 152 sessions, 11 subjects, 12,217 native trial IDs, 138,678 `iscell` ROIs. Efficiency rationale (Step 6): "Each NWB is opened once and closed after one session. Only required behavior and deconvolved activity are loaded."

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB metadata field `nwb.subject.subject_id` of each file rather than from the directory name. The unique ids are sorted to build `data['subjects']`, and `data['subject_idx']` is the index of each session's subject in that sorted list. Result: 11 subjects (m11–m15, m17–m19, m3, m4, m7), with 12 sessions for m11 and 14 for every other mouse.

ii.
```python
subject = str(nwb.subject.subject_id)
...
subjects.append(info['subject'])
...
subject_names = sorted(set(subjects))
data = {
  ...
  'subjects': subject_names,
  'subject_idx': np.asarray([subject_names.index(x) for x in subjects], dtype=np.int32),
```

iii. CONVERSION_NOTES Step 2: "Subjects | 11 (`m3`, `m4`, `m7`, `m11`–`m15`, `m17`–`m19`)"; Step 9 consistency table: "Subjects | 11 in reported analyses | named session dictionaries | 11 | 11 | Yes". The AI used the in-file subject metadata as the authoritative identifier so that the grouping does not depend on parsing file paths.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are never merged or split; each `convert_session()` call produces one element of `data['neural']` / `data['input']` / `data['output']`, and the session order is the sorted file order. No cross-session neuron alignment (ROI tracking across days) is attempted.

ii.
```python
for j, path in enumerate(files):
    sn, si, so, info = convert_session(path, args.show_processing and j < 2)
    neural.append(sn); inputs.append(si); outputs.append(so)
    infos.append(info); subjects.append(info['subject'])
```

iii. CONVERSION_NOTES Step 2 describes the layout `sub-<id>_ses-<nn>_behavior+ophys.nwb` as "approximately one file per imaging session", and Step 9 verifies "Sessions | ... | 152 NWBs | 152 | Yes for full release". Per-session provenance (file path, scene, plane count, cell count) is stored in `metadata['session_info']`.

## 1-d. How are the data split into trials?

i. Trials are defined **by grouping contiguous samples with the same non-negative value of the `trial number` behavioural series** (`-1` marks pre-session samples). `trial_start` and `teleport` are deliberately *not* used to cut the trial: the AI found the resampled `teleport` marker could give 0, 1 or 2 pulses per trial and judged trial-number grouping more robust. A consequence the AI did not detect is that a numbered trial in this dataset *begins at the teleport of the previous lap*: the first samples of each trial are the gray "jitter" period at position −50 cm, the 50 cm gray tunnel follows, and the `trial_start` pulse occurs only ~70 samples (~4.5 s, jittered 1–10 s) later. Roughly 32% of the samples inside the AI's trials precede the actual trial start.

ii.
```python
trial_ids = np.unique(arrays['trial number'][np.isfinite(arrays['trial number']) &
                                             (arrays['trial number'] >= 0)]).astype(int)
...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    nt = len(ix)
    pos = arrays['position'][ix].astype(np.float32)
    ...
    neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```
(`'teleport'` is read into `arrays` but never used.)

iii. CONVERSION_NOTES Step 4: "Teleport marker | Reference code uses teleport/end indices | Resampling yields 0, 1, or 2 positive teleport samples per otherwise complete trial | ... | Do not require exactly one pulse. Trial-number grouping plus start pulse and track completion is robust; include the whole numbered trial as the native start-to-next-boundary interval." Step 4 also states "Numbered trials usually span about −50 to 450 cm" and resolves: "Preserve complete numbered-trial samples for temporal fidelity". Step 10 Check 3 claims this is "Equivalent native trial segmentation" to the reference's `zip(trial_starts, teleports)` loop.

## 1-e. How are trials filtered based on quality controls?

i. A numbered trial is kept only if it satisfies four conditions simultaneously: at least 20 samples; at least one `trial_start` pulse inside it; maximum `position` ≥ 440 cm (the lap was actually completed); and `scanning == 1` at every sample (imaging was on throughout). Sessions with fewer than two surviving trials raise an error (none occurred). In addition, every kept trial is re-checked for finite neural/input values and matching time lengths. This removed exactly one trial across the whole dataset — a malformed 3-frame terminal fragment in an m11 session — leaving 12,216 trials (the same total the human reference obtained).

ii.
```python
valid_ids = []
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    complete = (len(ix) >= 20 and np.sum(arrays['trial_start'][ix] > 0) >= 1
                and np.nanmax(arrays['position'][ix]) >= 440
                and np.all(arrays['scanning'][ix] == 1))
    if complete:
        valid_ids.append(tid)
...
if len(valid_ids) < 2:
    raise ValueError(f'Fewer than two valid trials in {path}')
```

iii. CONVERSION_NOTES Step 4: "Trial count | ... | 12,217 IDs, but one terminal ID has only 3 frames, no start/teleport, and max position 154 cm | Analyses use completed trials | Exclude only this malformed terminal trial, leaving 12,216 valid trials. All others have one start pulse and reach >=440 cm." Step 3 curation rules add: "Preserve rewarded and omitted trials because reward outcome and previous outcome are required variables. Do not apply the paper's 'at least three omissions per trial set' restriction, which is specific to omission-comparison analyses."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are slices of the stored NWB `processing['ophys']['Deconvolved']` `RoiResponseSeries` (one series per imaging plane). The `Fluorescence` (F) and `Neuropil` (Fneu) series present in the same files are explicitly *not* used, and dF/F is not recomputed. For multi-plane sessions the per-plane series are concatenated along the ROI axis after each series' `DynamicTableRegion` (`r.rois.data`) is resolved back to segmentation-table rows.

ii.
```python
deconv_container = nwb.processing['ophys']['Deconvolved']
series = list(deconv_container.roi_response_series.values())
common_n = min([r.data.shape[0] for r in series] + [len(v) for v in arrays.values()])
...
for r in series:
    linked_rows = np.asarray(r.rois.data[:], dtype=np.int64)
    iscell = np.asarray(r.rois.table['iscell'][:])
    binary = iscell[:, 0] if iscell.ndim == 2 else iscell
    local_cell_mask = (binary[linked_rows] == 1)
    neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
neural_all = np.ascontiguousarray(np.concatenate(neural_parts, axis=1), dtype=np.float32)
```

iii. CONVERSION_NOTES Step 1: "Neural analyses use Suite2p-derived deconvolved event/activity arrays (`sess.timeseries['events']`); therefore no spike sorting quality filter is applicable and delta-F/F should not be recomputed when the released NWB already supplies processed neural activity." Step 4 discrepancy table: "Neural signal | Reference analyses use `sess.timeseries['events']` / deconvolved activity | NWB `ophys/Deconvolved` is time × all segmented ROIs | Calcium data processed with Suite2p | Use Deconvolved and retain only `iscell[:,0] == 1`; do not recompute dF/F." The AI equated the stored `Deconvolved` array with the paper's `events` without testing that equivalence.

## 2-b. How is the `neural` data processed?

i. No signal processing at all. The stored deconvolved values are truncated to the common frame length, column-masked by `iscell`, cast to `float32`, sliced per trial and transposed to (n_neurons, n_timepoints). There is no neuropil subtraction, no maximin baseline, no dF/F, no Gaussian smoothing, no OASIS deconvolution, and no per-neuron normalisation or z-scoring.

ii.
```python
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
neural_all = np.ascontiguousarray(np.concatenate(neural_parts, axis=1), dtype=np.float32)
...
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 3: "The released NWB provides fluorescence, neuropil, deconvolved activity, segmentation, and Suite2p `iscell`; the reference analyses use deconvolved calcium activity. Recomputing delta-F/F is unnecessary for the requested neural decoder." Step 3 also argues against normalisation: "Trial-by-trial remapping analyses normalize deconvolved activity per neuron ... but this normalization is specific to those analyses. ... conversion should preserve deconvolved event amplitudes and let decoder preprocessing handle scaling." Step 5 Key Decision 1 keeps the native temporal sampling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: Suite2p's binary classification `iscell[:, 0] == 1`, applied through each `RoiResponseSeries`' linked segmentation rows. 138,678 of 312,110 segmented ROIs survive (155–2,341 per session, mean 912.36). No other neuron-level filter is applied — in particular the Methods' additional exclusion of putative interneurons (Pearson r > 0.5 between a cell's dF/F and running speed) is not implemented, and analysis-level place-cell / RR / TR labels are deliberately not used.

ii.
```python
iscell = np.asarray(r.rois.table['iscell'][:])
binary = iscell[:, 0] if iscell.ndim == 2 else iscell
local_cell_mask = (binary[linked_rows] == 1)
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
```

iii. CONVERSION_NOTES Step 3 neuron-curation rules: "Apply Suite2p's binary `iscell[:,0] == 1` classification to exclude non-cell ROIs. Do not restrict the general decoder to place cells, RR cells, TR cells, or cells passing shuffled spatial-information tests. Those are downstream scientific subpopulation labels and would discard neural information relevant to other requested outputs." Step 3 conclusion: "the defensible release-level curation is Suite2p cell classification plus valid trials/samples". The interneuron-exclusion step is never mentioned anywhere in the notes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The stated alignment event is the start of the trial (`metadata['temporal_alignment_event'] = 'start of numbered trial (trial_start; first aligned imaging frame)'`, `off_start = 0.0`, `off_end = None`). Operationally, however, the trial's first sample is the first frame whose `trial number` equals the trial id, which (see 1-d) is the teleport frame ending the *previous* lap. No offsetting, cropping to the `trial_start` pulse, or windowing is performed. Neural, input and output rows are all sliced with the same index array `ix`, so the three streams are mutually aligned; what is misaligned is the reference event itself, by a variable 1–10 s jitter period.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
...
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
if neu.shape[1] != nt or inp.shape[1] != nt or out.shape[1] != nt:
    raise AssertionError('Within-trial temporal mismatch')
...
'temporal_alignment_event': 'start of numbered trial (trial_start; first aligned imaging frame)',
'off_start': 0.0, 'off_end': None,
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "Alignment is start of each numbered trial; `off_start=0.0`. Trial durations vary, so `off_end=None`". Step 10 Check 3(c): "contiguous numbered-trial samples aligned to first/start frame ... Equivalent native trial segmentation; robust to resampled teleport marker having 0/2 pulses."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging-frame resolution is retained: `DT = 0.06448362720402656` s (64.4836 ms, ~15.508 Hz), stored as `metadata['time_bin_size'] = 64.4836...` ms. No rebinning, resampling, interpolation, smoothing or spatial binning is applied, and the bin size is identical for every trial and session. The AI verified from the behavioural timestamps that this interval holds for all files — including the 28 two-plane sessions, whose `RoiResponseSeries.rate` field reads 31.0 Hz (scanner rate) but whose per-plane frames are still ~15.5 Hz.

ii.
```python
DT = 0.06448362720402656
...
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
...
'time_bin_size': DT*1000.0,
'sampling_rate_hz': 1.0/DT,
```

iii. CONVERSION_NOTES Step 2: "All frame-aligned streams use a 0.064483627204 s interval (~15.508 Hz)." Step 3: "This conversion should retain native temporal bins (~64.48 ms) because the requested decoder has time-varying outputs (time, position, speed, lick), rather than collapse samples into spatial bins." Step 10 Check 3(d): "Required difference: decoder outputs are time-varying speed/lick/position, so spatial aggregation would destroy targets."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not derived from a stored variable at all: it is synthesised as the sample index within the trial multiplied by the hard-coded constant frame interval `DT`. The actual `position` timestamps are read (as `frame_times`) but used only to map `Reward` event times onto frames, not to build the time input.

ii.
```python
pts = bts.time_series['position']
if pts.timestamps is not None:
    frame_times = np.asarray(pts.timestamps[:common_n], dtype=np.float64)
else:
    frame_times = float(pts.starting_time) + np.arange(common_n)/float(pts.rate)
...
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. CONVERSION_NOTES Step 5 mapping table: "frame timestamps / fixed interval | `input[0]` time from trial start | `arange(n_time) * 0.064483627204`, float32 | ... | Continuous and time-varying, starting at exactly 0." Justified by the Step 2 finding that the frame interval is constant (0.0644836 s) in every session.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `arange(n_timepoints) * DT`, i.e. the clock is reset to exactly 0.0 at the first sample of each trial and increments by one frame interval. Because that first sample is the teleport frame of the previous lap (1-d), the reported "time from trial start" is systematically offset by the ITI/jitter duration: the value is ~1–10 s (randomly varying trial to trial) at the moment the animal actually re-enters the corridor at 0 cm. The resulting range over the whole dataset is 0–707 s, versus 0–216.5 s for the human reference.

ii.
```python
inp = np.empty((4, nt), dtype=np.float32)
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. CONVERSION_NOTES Step 5: "Continuous and time-varying, starting at exactly 0." Step 4 resolution for the position domain: "Preserve complete numbered-trial samples for temporal fidelity". No further justification for the offset is given — the AI believed the first numbered-trial sample *was* the trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Trivially: the input array is allocated with exactly `nt = len(ix)` columns, the same `ix` used for the neural slice, so element *k* of the time vector corresponds to frame *k* of the neural matrix. An explicit assertion rejects any trial where the neural, input and output column counts disagree. No interpolation or resampling is needed because behaviour and imaging share the frame clock in the released files.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
nt = len(ix)
inp = np.empty((4, nt), dtype=np.float32)
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
...
if neu.shape[1] != nt or inp.shape[1] != nt or out.shape[1] != nt:
    raise AssertionError('Within-trial temporal mismatch')
```

iii. CONVERSION_NOTES Step 4: "The release is already temporally aligned at a fixed 64.4836 ms imaging-frame interval; no interpolation is needed." Ten sessions with one extra terminal neural frame are handled by truncating every stream to `common_n` before any indexing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The frame-aligned `environment` behavioural time series, with 0 = ENV1 and 1 = ENV2. The per-trial value is cross-checked against the environment(s) named in the session's scene string (parsed from `nwb.identifier`), and a mismatch raises an error.

ii.
```python
env_values = arrays['environment'][ix]
u, c = np.unique(env_values[np.isfinite(env_values)], return_counts=True)
if not len(u):
    raise ValueError(f'No finite environment in {path}, trial {tid}')
env_mode = u[np.argmax(c)]
env, zone = parse_scene(scene, tid, env_mode)
...
if env != expected_env:
    raise ValueError(f'Environment mismatch scene={scene}, trial={trial_id}: data={env}, expected={expected_env}')
```

iii. CONVERSION_NOTES Step 4: "Environment | Scene metadata names ENV1/ENV2 | Frame stream is 0 in ENV1 and 1 in ENV2, including environment-switch sessions | Two visual environments can switch independently of reward | Use per-trial modal frame value, 0=ENV1 and 1=ENV2." The scene-vs-stream assertion is described as a sanity check that "every scene parser result agrees with frame environment codes".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The modal (most frequent) finite value of `environment` within the trial is taken and broadcast as a constant across all timepoints of that trial, as `float32`. The mode is used rather than the raw per-sample stream so that a single transitional sample cannot corrupt a trial's label; in practice the stream is constant within a trial.

ii.
```python
u, c = np.unique(env_values[np.isfinite(env_values)], return_counts=True)
env_mode = u[np.argmax(c)]
env, zone = parse_scene(scene, tid, env_mode)
...
inp[1] = env
```
```python
def parse_scene(scene, trial_id, env_mode):
    env = int(round(float(env_mode)))
```

iii. CONVERSION_NOTES Step 5 mapping: "`environment` | `input[1]` environment type | Modal value over trial, repeat over time; 0=ENV1, 1=ENV2 | session scene/task handling | Native coding agrees with scene names." Verified range in the full data is [0, 1].

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The native `trial number` behavioural series: the trial's own integer id is used directly, not a re-indexed loop counter. Ids run 0–79 in most sessions (0–99 in the seven 100-trial sessions).

ii.
```python
trial_ids = np.unique(arrays['trial number'][np.isfinite(arrays['trial number']) &
                                             (arrays['trial number'] >= 0)]).astype(int)
...
for tid in valid_ids:
    ...
    inp[2] = tid
```

iii. CONVERSION_NOTES Step 5 mapping: "`trial number` | `input[2]` trial number | Native nonnegative trial ID, repeat over time, float32 | trial dictionaries | Preserve zero-based native numbering; continuous per-trial predictor." The AI treated the stored id as authoritative after checking in Step 4 that every retained trial contains exactly one `trial_start` pulse.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the integer id across the trial's timepoints as `float32`. Because exactly one trial (a malformed m11 fragment) is dropped dataset-wide, the retained ids are contiguous 0…N−1 in every session except that one, where the final id is simply absent rather than the remaining ids being renumbered.

ii.
```python
inp[2] = tid
```

iii. Same as 5-a; the verifier confirms the input range is [0, 99]. CONVERSION_NOTES Step 9: "Input ranges | task design | aligned behavior | env 0–1, native trial IDs 0–99, previous outcome 0–1 | same | Yes".

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the event-based `Reward` time series. Its event timestamps are mapped to the nearest frame of the `position` timestamp vector, each such frame is read off the `trial number` stream, and the set of trial ids that received at least one reward is built. `autoreward` is examined and rejected (it is identically zero in the release).

ii.
```python
reward = bts.time_series['Reward']
reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
event_frames = nearest_frame_indices(frame_times, reward_times) if len(reward_times) else np.array([], dtype=int)
event_trial_ids = arrays['trial number'][event_frames] if len(event_frames) else np.array([])
rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)
```
```python
def nearest_frame_indices(frame_times, event_times):
    idx = np.clip(np.searchsorted(frame_times, event_times), 0, len(frame_times)-1)
    left = np.maximum(idx-1, 0)
    return np.where(np.abs(frame_times[left]-event_times) < np.abs(frame_times[idx]-event_times), left, idx)
```

iii. CONVERSION_NOTES Step 4: "Reward outcome | Omission logic is event/trial based | `Reward` has timestamped 0.004-volume events; all map cleanly to numbered trials. `autoreward` is always zero. | Rewarded and omission trials are compared | Outcome=1 if at least one Reward timestamp maps within that trial, else 0. Do not use `autoreward`." Step 5 Key Decision 7: "Timestamped Reward events are authoritative and map cleanly to trials."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A running variable `prev_outcome`, initialised to 0, is broadcast across the current trial's timepoints and then updated to the current trial's outcome at the end of each loop iteration. So the first retained trial of every session gets 0, and every later trial gets the outcome of the *previous retained* trial (identical to the previous native trial except in the single session where a trial was dropped).

ii.
```python
prev_outcome = 0
for tid in valid_ids:
    ...
    outcome = int(tid in rewarded_ids)
    ...
    inp[3] = prev_outcome
    ...
    prev_outcome = outcome
```

iii. CONVERSION_NOTES Step 4: "Previous outcome | Not a primary reference predictor | First trial has no preceding outcome | Task requires binary previous outcome | Encode first valid trial as 0 (no prior rewarded trial available); subsequent trials copy prior valid trial outcome." Step 10 Check 5: "First valid trial previous outcome is 0; later values derive from the preceding retained trial."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the frame-aligned `position` series together with a reward-zone interval that is derived **from the session's scene name** (`nwb.identifier`, e.g. `Env1_LocationC_to_B`, `Env1_B_to_Env2_C`) plus the trial id and the `environment` stream. The frame-aligned `reward_zone` series is *not* used to assign the zone; the AI inspected it, found it takes transient state codes 0–7, and used it only as an informal cross-check of zone centres. Two constants encode the mapping: `ZONE_BOUNDS = {'A': (80.0, 100.0), 'B': (200.0, 220.0), 'C': (320.0, 340.0)}` — i.e. 20 cm wide zones — and a switch rule that changes the zone at trial 40 for reward-only switch scenes and at the environment transition (trial 30) for combined environment+reward scenes.

ii.
```python
ZONE_BOUNDS = {'A': (80.0, 100.0), 'B': (200.0, 220.0), 'C': (320.0, 340.0)}
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}

def parse_scene(scene, trial_id, env_mode):
    env = int(round(float(env_mode)))
    m = re.fullmatch(r'Env([12])_([ABC])_to_Env([12])_([ABC])', scene)
    if m:
        first_env, first_zone = int(m.group(1))-1, m.group(2)
        second_env, second_zone = int(m.group(3))-1, m.group(4)
        if env == first_env:  return env, first_zone
        if env == second_env: return env, second_zone
        raise ValueError(...)
    m = re.search(r'Env([12])_Location([ABC])(?:_to_([ABC]))?', scene)
    ...
    zone = second if second is not None and trial_id >= 40 else first
    return env, zone
...
low, high = ZONE_BOUNDS[zone]
dist = signed_distance_to_interval(pos, low, high)
```

iii. CONVERSION_NOTES Step 4: "Reward-zone source | ... | Native `reward_zone` values 0–7 are transient state/event combinations, not A/B/C labels | ... | Derive per-trial A/B/C from scene/task schedule: scene's first location for trials <40 and second for trials >=40 in reward-only `LocationX_to_Y`; stable scenes retain X. Combined `EnvN_X_to_EnvM_Y` sessions switch at trial 30 and use the frame environment to select the corresponding location. Cross-check: native nonzero-state positions center near A≈82, B≈202, C≈322 cm." Step 5 Key Decision 5: "Use methods-defined 20 cm zones A=80–100, B=200–220, C=320–340 cm." Step 9 Iteration 1 records that the trial-40 rule broke on combined-switch sessions and was replaced there by the environment stream, while "Reward-only switches retain the validated trial-40 rule."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest point of the active zone interval: negative before the zone, exactly 0.0 anywhere inside it, positive after it. Computed vectorised over the whole trial with `np.where`, then discretised (7-c). The zone interval used is the 20 cm `ZONE_BOUNDS` entry for the trial's scene-derived letter.

ii.
```python
def signed_distance_to_interval(position, low, high):
    """Negative before, zero within, positive after a closed interval."""
    return np.where(position < low, position-low,
                    np.where(position > high, position-high, 0.0))
...
low, high = ZONE_BOUNDS[zone]
dist = signed_distance_to_interval(pos, low, high)
out[0] = discretize_distance(dist)
```

iii. CONVERSION_NOTES Step 5 mapping: "Signed distance to nearest point in interval: negative before zone, 0 inside, positive after; discretize prescribed 7 bins ... Exact 0 represents every location inside zone ('distance to any location in zone')." Key Decision 5: "Signed distance is zero anywhere in the active interval, satisfying 'distance to any location in the reward zone,' unlike distance to a center/start."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks that follow the task specification literally, including the exact-boundary conventions: 0 `d < −50`; 1 `−50 ≤ d ≤ −10`; 2 `−10 < d < 0`; 3 `d == 0` (anywhere inside the zone); 4 `0 < d ≤ 10`; 5 `10 < d ≤ 50`; 6 `d > 50`. Stored as `int8`.

ii.
```python
def discretize_distance(d):
    """Task-prescribed seven classes, including exact-boundary conventions."""
    y = np.empty(d.shape, dtype=np.int8)
    y[d < -50] = 0
    y[(d >= -50) & (d <= -10)] = 1
    y[(d > -10) & (d < 0)] = 2
    y[d == 0] = 3
    y[(d > 0) & (d <= 10)] = 4
    y[(d > 10) & (d <= 50)] = 5
    y[d > 50] = 6
    return y
```

iii. CONVERSION_NOTES Step 10 Check 5: "Equality boundaries unit-tested: distance −50/−10/0/+10/+50". `output_values` labels are written to match: `['< -50 cm', '-50 to -10 cm', '-10 to < 0 cm', 'inside reward zone (0 cm)', '>0 to +10 cm', '+10 to +50 cm', '> +50 cm']`. A `validate_complete()` assertion checks every value lies in [0, 7).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No separate alignment step: `position` is sliced with the same frame-index array `ix` as the neural matrix, so the distance class at column *k* is contemporaneous with neural column *k*. The shared-length assertion applies to the output block as well. (The alignment is exact within the trial window; the window itself starts at the previous lap's teleport — see 1-d/2-d.)

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
pos = arrays['position'][ix].astype(np.float32)
...
out = np.empty((6, nt), dtype=np.int8)
out[0] = discretize_distance(dist)
...
if neu.shape[1] != nt or inp.shape[1] != nt or out.shape[1] != nt:
    raise AssertionError('Within-trial temporal mismatch')
```

iii. CONVERSION_NOTES Step 4: "The release is already temporally aligned at a fixed 64.4836 ms imaging-frame interval; no interpolation is needed." Step 10 Check 2 reports an independent raw-NWB audit that reconstructed "signed zone distance classes" for selected trials and compared them with `np.allclose`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The frame-aligned `position` behavioural series, in centimetres, read directly with no unit conversion or offset.

ii.
```python
arrays = {k: np.asarray(bts.time_series[k].data[:]).squeeze() for k in needed}  # includes 'position'
arrays = {k: v[:common_n] for k, v in arrays.items()}
...
pos = arrays['position'][ix].astype(np.float32)
out[1] = discretize_position(pos)
```

iii. CONVERSION_NOTES Step 2 variable table: "position | frame-aligned numeric | Absolute forward position on the 450 cm corridor."

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None apart from the `float32` cast and discretisation — the raw values are used as-is. Because the AI's trials include the preceding teleport/jitter/tunnel period, a large share of the samples have position pinned at −50 cm or ramping from −50 to 0; these are assigned to class 0 rather than excluded. The resulting class-0 fraction is 0.421 in the converted data (0.211 in the human reference, which excludes the teleport period).

ii.
```python
pos = arrays['position'][ix].astype(np.float32)
out[1] = discretize_position(pos)
```

iii. CONVERSION_NOTES Step 4: "Position domain | Reference analyses may include teleport/tunnel position down to −50 cm | Numbered trials usually span about −50 to 450 cm ... | Preserve complete numbered-trial samples for temporal fidelity, but absolute-position category follows required thresholds (<90, 90–180, ..., >360), naturally assigning negative teleport samples to bin 0." Step 5 mapping: "Values outside corridor naturally enter edge classes."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm classes spanning the 450 cm track with open ends: 0 `x < 90`; 1 `90 ≤ x < 180`; 2 `180 ≤ x < 270`; 3 `270 ≤ x ≤ 360`; 4 `x > 360`. Exactly 360 is deliberately kept in class 3 because the spec writes class 4 as "> 360 cm".

ii.
```python
def discretize_position(x):
    # Spec says >360 for class 4, hence exact 360 remains class 3.
    y = np.zeros(x.shape, dtype=np.int8)
    y[(x >= 90) & (x < 180)] = 1
    y[(x >= 180) & (x < 270)] = 2
    y[(x >= 270) & (x <= 360)] = 3
    y[x > 360] = 4
    return y
```

iii. CONVERSION_NOTES Step 10 Check 5: "position 90/180/270/360 (exact 360 remains class 3)". Step 5 mapping records the intended convention "`<90`, `[90,180)`, `[180,270)`, `[270,360]` ... `>=360` class 4" (the note's final clause is loosely worded; the code implements strict `> 360`).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame-index slice `ix` as the neural matrix; no resampling or shifting. Verified by the per-trial shape assertion and by the independent raw-NWB audit.

ii.
```python
pos = arrays['position'][ix].astype(np.float32)
...
out[1] = discretize_position(pos)
...
if neu.shape[1] != nt or inp.shape[1] != nt or out.shape[1] != nt:
    raise AssertionError('Within-trial temporal mismatch')
```

iii. CONVERSION_NOTES Step 10 Check 2 lists "absolute-position classes" among the quantities independently reconstructed from raw NWB and compared with `np.allclose` for the audited trials.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The frame-aligned `lick` behavioural series, whose native values are counts in the range 0–8.

ii.
```python
lick = arrays['lick'][ix]
...
out[3] = (lick > 0).astype(np.int8)
```

iii. CONVERSION_NOTES Step 2: "lick | frame-aligned numeric | Lick sensor signal"; Step 4: "Lick | GLM code clips values >1 after sensor-error handling | Native lick ranges 0–8 | Lick behavior is event/count-like | Required output is binary: 0 if value <=0, 1 if >0. Do not smooth."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation only: `(lick > 0)` cast to `int8`. No smoothing, no spatial lick-rate computation, and none of the reference GLM's trial-level sensor-failure NaN-masking is applied.

ii.
```python
out[3] = (lick > 0).astype(np.int8)
```

iii. CONVERSION_NOTES Step 1: "Lick preprocessing in `glmUtils.py` corrects likely sensor failures ... GLM-specific smoothing and zero-centering are analysis transforms and should not be imposed on the decoder's required binary lick output." Step 3: "The requested output is binary lick, so valid native frame-level lick values should be thresholded/clipped to 0/1 rather than spatially smoothed." Resulting frame-level lick rate is 0.182 (human reference: 0.230).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame-index slice `ix`; the lick stream shares the imaging-frame clock, so no additional alignment is performed.

ii.
```python
lick = arrays['lick'][ix]
...
out[3] = (lick > 0).astype(np.int8)
```

iii. Step 4 conclusion: "The release is already temporally aligned at a fixed 64.4836 ms imaging-frame interval; no interpolation is needed." Step 10 Check 2 audited "binary lick" against an independent raw-NWB reconstruction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The scene string (`nwb.identifier`) plus the trial id and, for combined environment+reward switch sessions, the `environment` stream — the same `parse_scene()` result used for the distance output. The letter is mapped through `ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}`. The frame-aligned `reward_zone` series is not used for the assignment.

ii.
```python
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}
...
env, zone = parse_scene(scene, tid, env_mode)
...
out[4] = ZONE_INDEX[zone]
```

iii. CONVERSION_NOTES Step 5 mapping: "scene reward schedule | `output[4]` reward zone location | A=0, B=1, C=2, repeated over time | scene dictionaries and task methods | Stable `LocationX`: X all trials. Reward-only `LocationX_to_Y`: X for IDs <40, Y for IDs >=40. Combined `EnvN_X_to_EnvM_Y` sessions switch both variables at native trial 30; use the aligned environment stream to select the matching X/Y side." Step 4 cross-check: "native nonzero-state positions center near A≈82, B≈202, C≈322 cm".

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex parsing of the scene name into (environment, letter), application of the switch rule, mapping of the letter to 0/1/2, and broadcasting across the trial's timepoints as `int8`. For combined scenes the switch is data-driven (whichever scene side matches the observed environment); for reward-only switch scenes it is the hard-coded threshold `trial_id >= 40`; for stable scenes the single letter is used throughout. A mismatch between the parsed and observed environment raises an error.

ii.
```python
    zone = second if second is not None and trial_id >= 40 else first
    return env, zone
...
out[4] = ZONE_INDEX[zone]
```

iii. Step 5 Key Decision 6: "Reward-only sessions switch after 40 trials (IDs 0–39 versus >=40). All combined environment+reward sessions switch at native trial 30; the aligned environment stream robustly selects the corresponding scene side. Native reward-state medians validate both conventions." Step 9 Iteration 1 documents fixing the combined-switch case only. The resulting zone class fractions are A 0.336 / B 0.330 / C 0.334.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The event-based `Reward` time series (its `timestamps`), mapped to frames via the `position` timestamps and then to trial ids via the `trial number` stream — the identical machinery used for the previous-trial-outcome input (6-a). `autoreward` is examined and rejected as identically zero.

ii.
```python
reward = bts.time_series['Reward']
reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
event_frames = nearest_frame_indices(frame_times, reward_times) if len(reward_times) else np.array([], dtype=int)
event_trial_ids = arrays['trial number'][event_frames] if len(event_frames) else np.array([])
rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)
```

iii. CONVERSION_NOTES Step 4: "`Reward` has timestamped 0.004-volume events; all map cleanly to numbered trials. `autoreward` is always zero. ... Outcome=1 if at least one Reward timestamp maps within that trial, else 0. Do not use `autoreward`."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial: `outcome = int(tid in rewarded_ids)` — 1 if any reward event was delivered during that trial, else 0 — broadcast across all of the trial's timepoints as `int8`. Dataset-wide this gives 10,342 rewarded and 1,874 omitted trials (84.66% rewarded), matching the paper's "~15% omissions" design.

ii.
```python
outcome = int(tid in rewarded_ids)
...
out[5] = outcome
...
prev_outcome = outcome
```

iii. CONVERSION_NOTES Step 12: "Outcome is per-trial and highly imbalanced: 1,874 omitted and 10,342 rewarded trials (15.34%/84.66%)." Step 5 Key Decision 7: "Timestamped Reward events are authoritative and map cleanly to trials. This is superior to `autoreward`, which is invariant zero." Step 5 Key Decision 8: "Repeat per-trial zone/outcome across time, yielding integer `(6, n_time)` outputs."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive cases are handled:
- **Stream-length mismatch** (ten sessions have one extra terminal neural frame): every neural series and every behavioural array is truncated to `common_n`, the minimum length across all of them, before any indexing.
- **Multi-plane ROI/response-column mapping**: each `RoiResponseSeries` resolves its own `DynamicTableRegion` (`r.rois.data`) into segmentation rows before applying `iscell`, with a hard error if the column count and link count disagree. (This was a bug found and fixed during the full run.)
- **Malformed trial**: the one 3-frame terminal fragment with no start pulse and max position 154 cm fails the completeness test and is dropped.
- **Missing/odd teleport pulses**: no requirement of exactly one teleport pulse per trial.
- **Non-finite values**: trials with any non-finite neural or input value raise an error (none occurred); trials with no finite `environment` sample raise an error.
- **Too-few trials**: a session with fewer than two valid trials raises an error (none occurred).
- **Unparseable scene / environment mismatch**: raise rather than guess.

ii.
```python
common_n = min([r.data.shape[0] for r in series] + [len(v) for v in arrays.values()])
arrays = {k: v[:common_n] for k, v in arrays.items()}
...
if r.data.shape[1] != len(linked_rows):
    raise ValueError(f'RoiResponseSeries/DynamicTableRegion mismatch in {path}: {r.name}')
...
if not (np.all(np.isfinite(neu)) and np.all(np.isfinite(inp))):
    raise ValueError(f'Nonfinite neural/input values in {path}, trial {tid}')
if len(valid_ids) < 2:
    raise ValueError(f'Fewer than two valid trials in {path}')
```

iii. CONVERSION_NOTES Step 4: "Stream lengths | ... | 142 sessions match exactly; ten have one extra terminal neural frame | ... | Truncate to the common time length before trial indexing; mismatch is terminal and one frame only." Step 9 Iteration 2 documents the multi-plane `iscell` bug and its fix ("124 single-series and 28 two-series sessions ... All 152 mappings had valid, unique, in-range links"). Step 10 Check 5 lists the edge cases re-verified after the fixes.

## 13-a. What are the most time-consuming steps of the code?

i. The AI identified NWB file I/O as the dominant cost — reading the deconvolved activity array out of 87 GB of source files — followed by serialising the 13.33 GB output pickle. Per-session wall time is measured and printed (`info['elapsed_s']`), together with a running total, so the bottleneck is visible in `conversion_full_out.txt`. The full conversion of 152 sessions took 112 s, well inside the 15-minute budget, so no further optimisation or parallelism was pursued.

ii.
```python
def convert_session(path, make_plot=False):
    t0 = time.time()
    ...
    'elapsed_s': time.time()-t0,
...
print(f"[{j+1}/{len(files)}] {path.relative_to(DATA_ROOT)} scene={info['scene']} cells={info['n_cells']} "
      f"trials={info['n_trials']} rewarded={info['rewarded_trials']} time={info['elapsed_s']:.2f}s total={time.time()-t0:.1f}s", flush=True)
...
print(f'Wrote {out} ({out.stat().st_size/1e9:.3f} GB) in {time.time()-t0:.1f}s; ...')
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: The 87 GB source requires avoiding repeated reads and retaining full-session float64 arrays." Step 9: "152 sessions processed in 112.0 s including 13.33 GB pickle serialization. This was faster than the <5 min estimate and did not require parallel I/O." Step 7: "The estimate is far below 15 minutes, so no parallel I/O (which could increase memory pressure on 87 GB of NWB sources) is needed."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not enumerate specific vectorisable loops; its efficiency notes are about I/O and dtypes rather than loop structure. The two candidates actually present are (a) the trial-membership scan `np.flatnonzero(arrays['trial number'] == tid)`, which does a full-length pass over the trial-number vector once per trial and is executed twice per trial (once in the validity loop, once in the conversion loop) — it could be replaced by a single `np.searchsorted`/boundary computation on the contiguous runs; and (b) the per-trial construction of `inp`/`out`, where `discretize_distance`, `discretize_position`, `discretize_speed` and the lick threshold could be applied once to the whole session before splitting. Both are already fast relative to HDF5 reads (112 s total), so the AI's decision not to pursue them is consistent with its own profiling.

ii.
```python
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)          # pass 1 (validity)
    ...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)          # pass 2 (conversion)
    ...
    out[0] = discretize_distance(dist)
    out[1] = discretize_position(pos)
    out[2] = discretize_speed(speed)
```
The discretisers themselves are already fully vectorised over a trial:
```python
y[(x >= 90) & (x < 180)] = 1
```

iii. CONVERSION_NOTES Step 6 lists only "Code speedups added: Each NWB is opened once and closed after one session. Only required behavior and deconvolved activity are loaded; neural data are immediately cell-filtered and cast to float32. Outputs use int8 and session processing is sequential to control memory. Processing plots are restricted to two sessions and four representative trials per session." No loop-level vectorisation analysis is documented.

## 13-c. What processing does the code repeat multiple times?

i. Three repetitions: (1) the `np.flatnonzero(trial number == tid)` membership scan is recomputed for every trial in the validity pass and then again in the conversion pass; (2) `np.nanmax(position[ix])` in the validity pass duplicates work on data that is re-sliced immediately afterwards; (3) `plot_info` keeps a second copy of position/speed/lick/distance/output arrays for the first four trials when `--show-processing` is on. Notably, the code does **one** pass over each NWB file — there is no separate survey pass — so the source files are each read exactly once.

ii.
```python
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    complete = (len(ix) >= 20 and np.sum(arrays['trial_start'][ix] > 0) >= 1
                and np.nanmax(arrays['position'][ix]) >= 440
                and np.all(arrays['scanning'][ix] == 1))
...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
...
    if len(plot_info) < 4:
        plot_info.append((tid, pos.copy(), speed.copy(), (lick > 0).copy(), dist.copy(), out.copy()))
```

iii. CONVERSION_NOTES Step 6: "Each NWB is opened once and closed after one session"; Step 9: "152 sessions processed in 112.0 s". The single-pass design is the AI's explicit answer to "avoiding repeated reads". The duplicated in-memory trial scans are not mentioned in the notes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI documents no such analysis (it asserts only that required data are loaded). Inspecting the script, the unnecessary work is: (1) the `teleport` series is read into `arrays` and truncated but never used anywhere; (2) `n_rois` and `neural_native_lengths` are computed purely for the `session_info` metadata block, which the decoder ignores; (3) the full 152-entry `session_info` list of per-session dicts is pickled with the data; (4) under `--show-processing`, extra copies of per-trial arrays and two PNG renders. Most consequentially, because trials are cut at `trial number` boundaries the converted `neural` arrays carry the ~32% of frames belonging to the preceding inter-trial teleport/jitter period — roughly 4 GB of the 13.33 GB pickle is data the paper's own trial definition would exclude.

ii.
```python
needed = ['environment', 'lick', 'position', 'scanning', 'speed',
          'teleport', 'trial number', 'trial_start']   # 'teleport' never used again
...
info = {
    'file': str(path), 'subject': subject, 'scene': scene,
    'n_cells': n_cells, 'n_rois': int(n_rois), 'n_planes': len(series),
    'n_trials': len(valid_ids), 'native_trial_ids': valid_ids,
    'common_n': common_n, 'neural_native_n': neural_native_lengths,
    'reward_events': len(reward_times), 'rewarded_trials': len(set(valid_ids) & rewarded_ids),
    'elapsed_s': time.time()-t0,
}
...
'session_info': infos,
```

iii. CONVERSION_NOTES Step 6 claims the opposite of (1): "Only required behavior and deconvolved activity are loaded". Step 5 Key Decision 9 justifies the memory strategy: "Process NWBs sequentially, materialize only deconvolved activity and required behavior for one session, immediately split/cast into compact trial arrays, then close the file. Use float32 neural/input and compact integer outputs." The retention of the teleport period is justified in Step 4 as "Preserve complete numbered-trial samples for temporal fidelity", not identified as waste.
