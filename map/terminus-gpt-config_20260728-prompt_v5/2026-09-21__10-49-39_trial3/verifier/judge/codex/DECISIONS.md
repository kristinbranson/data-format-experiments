# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by recursively finding every `*.nwb` file under `/app/data`, opening each file directly with `h5py`, then reading the trial table from `intervals/trials`, the units table from `units`, and selected event/behavior streams from `acquisition/...`. It does not use `pynwb`.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
```

```python
with h5py.File(path, 'r') as f:
    trial_table = load_trial_table(f)
    units = load_units_table(f)
    ...
    go_times = infer_go_cue_times(trial_table, f)
    ...
    left_licks = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:] if 'acquisition/BehavioralEvents/left_lick_times/timestamps' in f else np.array([])
```

iii. The notes say the script intentionally used "h5py direct reads instead of full pynwb object loading for conversion" as a speedup. The trajectory and notes also describe the goal as loading NWB files efficiently while reconstructing trials, units, and behavior streams from the HDF5 layout.

## 1-b. How are the data split into subjects?

i. The AI treats the parent folder of each NWB file as the subject id, e.g. `sub-440956`. It builds `subjects` incrementally in first-seen order and stores `subject_idx` from that list.

ii.
```python
subj = path.parent.name
...
if subj not in subjects:
    subjects.append(subj)
subject_idx.append(subjects.index(subj))
```

iii. The notes explicitly map "Subject folder / NWB subject metadata" to `subjects`/`subject_idx`, and the implementation chose the folder name as the concrete identifier. No trajectory evidence shows use of `nwb.subject.subject_id`; the apparent rationale was that the directory structure already groups sessions by subject.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The output session order follows the sorted file list.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
...
for f in files:
    sn, si, so, subj, regions = process_session(f, show_processing=args.show_processing)
```

iii. The notes describe the dataset as "session-level NWB files" under subject folders, so the AI used file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. Within each session, the AI iterates over `go_times` and treats each index `i` as one trial, using row `i` from the trial table and the corresponding `go_times[i]` window. It does not assert that the go-cue array length exactly matches the trials table length.

ii.
```python
go_times = infer_go_cue_times(trial_table, f)
...
for i, go in enumerate(go_times):
    ...
    inp0 = build_time_from_tone_vector(sample_starts[i], go)
    ...
    out = np.vstack([
        np.full(N_BINS, choice[i], dtype=np.int64),
        np.full(N_BINS, outcome[i], dtype=np.int64),
        np.full(N_BINS, early[i], dtype=np.int64),
        tongue_trial.astype(np.int64),
    ])
```

iii. The notes frame the go cue as the alignment event and describe the NWB data as trial-structured. The trajectory shows the AI initially used heuristic go-cue inference, then switched to raw `go_start_times`; no separate trial-boundary reconstruction was documented.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not use `obs_intervals` or `free_water`. Instead, it drops trials whose extracted neural matrix is entirely zero and drops sessions with fewer than 2 surviving trials.

ii.
```python
if start_idx < 0 or end_idx > session_fr.shape[1]:
    continue
...
if np.allclose(trial_mats, 0):
    continue
if np.allclose(trial_mats, 0):
    continue
...
if len(sn) < 2:
    continue
```

iii. The notes justify this as fixing "all-zero neural trials" after verification warnings. Step 10 says those warnings were "resolved by skipping trials whose aligned neural matrix was entirely zero," and the Step 12 notes say these dropped trials were considered invalid and could harm decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `units/spike_times` ragged array for units passing the AI's QC rule, together with go-cue times used to extract each trial window.

ii.
```python
def get_spike_times_for_unit(units_group, idx):
    st = units_group['spike_times']
    if 'spike_times_index' in units_group:
        ind = units_group['spike_times_index'][:]
        end = ind[idx]
        start = 0 if idx == 0 else ind[idx - 1]
        return st[start:end]
```

```python
good_inds = np.flatnonzero(good_mask)
...
for j, u in enumerate(good_inds):
    st = get_spike_times_for_unit(f['units'], int(u))
```

iii. The Step 5 mapping says "NWB units spike times / curated unit table -> neural," with go-cue alignment over the decoder window.

## 2-b. How is the `neural` data processed?

i. The AI pre-bins each good unit once over a session-wide 50 ms grid using `np.histogram`, converts counts to firing rates in Hz by dividing by `BIN_SIZE`, and then slices the pre-binned session matrix into per-trial windows.

ii.
```python
session_edges = np.arange(session_start, session_stop + BIN_SIZE, BIN_SIZE)
session_fr = np.zeros((len(good_inds), len(session_edges) - 1), dtype=np.float32)
for j, u in enumerate(good_inds):
    st = get_spike_times_for_unit(f['units'], int(u))
    m = (st >= session_start) & (st < session_stop)
    counts, _ = np.histogram(st[m], bins=session_edges)
    session_fr[j] = counts.astype(np.float32) / BIN_SIZE
```

```python
start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
end_idx = start_idx + N_BINS
trial_mats = session_fr[:, start_idx:end_idx].copy()
```

iii. The notes say the initial implementation's bottleneck was spike binning and that a speedup was added to "Pre-bin spikes once per session instead of histogramming per trial/unit." The trajectory explicitly records this as a runtime optimization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI uses a heuristic QC routine. It prefers text fields like `classification`, `unit_quality`, `quality`, `label`, or `cluster_quality`, accepting labels such as `good`, `single`, `single_unit`, or `single unit`. If those are unavailable, it falls back to boolean flags or metric thresholds (`presence_ratio`, `amplitude_cutoff`, `isi_violation(s)`, `nn_hit_rate`). Sessions with no surviving trials are later dropped implicitly.

ii.
```python
for key in ['classification', 'unit_quality', 'quality', 'label', 'cluster_quality']:
    if key in units:
        vals = units[key]
        sval = np.array([str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() for v in vals], dtype=object)
        good_words = {'good', 'single', 'single_unit', 'single unit'}
        if np.isin(sval, list(good_words)).any():
            return np.isin(sval, list(good_words))
...
if 'presence_ratio' in units:
    mask &= np.asarray(units['presence_ratio'], dtype=float) >= 0.9
...
if mask.sum() == 0:
    mask = np.ones(n, dtype=bool)
```

iii. The notes say the AI intended to "filter to good units only" and match the paper's curated-unit counts, but Step 6 also admits that the script used "heuristic field detection for ... unit QC." The trajectory and notes show flexibility/speed were prioritized over strict use of the NWB `classification == 'good'` convention.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset by taking `go_start_times` when present, then slicing an already binned session-rate matrix from `go + T_START` to `go + T_END` using rounded bin indices.

ii.
```python
def infer_go_cue_times(table, f=None):
    if f is not None and 'acquisition/BehavioralEvents/go_start_times/timestamps' in f:
        return np.asarray(f['acquisition/BehavioralEvents/go_start_times/timestamps'][:], dtype=float)
```

```python
for i, go in enumerate(go_times):
    start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
    end_idx = start_idx + N_BINS
    ...
    trial_mats = session_fr[:, start_idx:end_idx].copy()
```

iii. The notes explicitly say the AI switched from heuristic go-cue inference to raw `go_start_times` after validation warnings. The alignment rationale in the notes is that the decoder requires go-cue alignment and all streams should share the same binning window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins over a fixed `[-2.5, 1.5]` s window, for 80 time bins per trial. No secondary temporal rebinning is applied after the session-wide histogramming.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The notes repeatedly cite the decoder instructions as the reason for 50 ms bins over the go-cue-centered 4 s window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from sample/tone onset times when available, but with fallbacks. It initializes `sample_starts` from `trial_table['start_time']`, then replaces those with `sample_start_times` timestamps if available. If the number of sample events differs from the number of trials, it chooses the first sample event between `start_time` and `stop_time` for each trial.

ii.
```python
sample_starts = np.asarray(trial_table['start_time'], dtype=float)
if 'acquisition/BehavioralEvents/sample_start_times/timestamps' in f:
    sample_event_times = np.asarray(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:], dtype=float)
    if len(sample_event_times) == len(go_times):
        sample_starts = sample_event_times.copy()
    else:
        for i in range(len(go_times)):
            cand = sample_event_times[(sample_event_times >= start_times[i]) & (sample_event_times <= stop_times[i])]
            if len(cand):
                sample_starts[i] = cand[0]
```

iii. The Step 5 mapping says this variable should come from "BehavioralEvents sample/tone onset times relative to go cue." The code's fallback to `start_time` and first-in-trial sample events appears to be a robustness heuristic; no notes justify the "first event" choice specifically.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes a continuous, time-varying value at each bin center by expressing the bin center relative to the chosen `sample_start` for that trial.

ii.
```python
def build_time_from_tone_vector(sample_start, go_time):
    return (BIN_CENTERS - (sample_start - go_time)).astype(np.float32)[None, :]
```

iii. The notes state that this input should be "time-varying continuous input: time from tone onset in seconds at each bin," and the implementation directly realizes that formula.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI aligns this input to the same go-cue-centered 80-bin grid used for neural data by evaluating the time-from-tone value at the same `BIN_CENTERS`.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
inp0 = build_time_from_tone_vector(sample_starts[i], go)
```

iii. The notes say all time-varying streams should use the same 50 ms, go-cue-aligned window.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI primarily derives photostimulation from the trial-table fields `photostim_onset` and `photostim_duration`. If those fields are unavailable, it falls back to event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`.

ii.
```python
ps_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:] if 'acquisition/BehavioralEvents/photostim_start_times/timestamps' in f else np.array([])
ps_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:] if 'acquisition/BehavioralEvents/photostim_stop_times/timestamps' in f else np.array([])
...
if 'photostim_onset' in trial_table and 'photostim_duration' in trial_table:
    onset = safe_float(trial_table['photostim_onset'][i])
    dur = safe_float(trial_table['photostim_duration'][i])
```

iii. The notes first planned to use event timing, then Step 10 says photostim parsing was fixed by "treating trial-table `photostim_onset` as trial-relative and combining with `photostim_duration`." So the final justification was that the trial table gave the correct per-trial timing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts photostimulation timing into a binary time series over the 80 bins: bins whose centers fall between stimulation start and stop are set to 1, otherwise 0.

ii.
```python
def build_photostim_vector(starts, stops, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(starts, stops):
        rs = s - go_time
        re = e - go_time
        on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
        x[on] = 1.0
    return x
```

iii. The Step 5 notes explicitly say the AI decided to "Construct photostim as a binary time series" to match the decoder specification.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation to neural data by converting absolute stim start/stop times into go-relative time inside `build_photostim_vector`, then comparing those times to the shared `BIN_CENTERS`.

ii.
```python
rs = s - go_time
re = e - go_time
on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
```

iii. The notes describe photostimulation as a pre-go, time-varying input that should be represented across the same aligned window as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from behavioral lick-event timestamps after the go cue, not from `trial_instruction`. It uses `left_lick_times` and `right_lick_times` within the post-go response window, plus `stop_time` to cap that window. `outcome` is only used later as a separate decoded target.

ii.
```python
left_licks = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:] if 'acquisition/BehavioralEvents/left_lick_times/timestamps' in f else np.array([])
right_licks = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:] if 'acquisition/BehavioralEvents/right_lick_times/timestamps' in f else np.array([])
...
for i in range(n):
    l_post = np.any((left_licks >= go_times[i]) & (left_licks < min(stop[i], go_times[i] + 1.5)))
    r_post = np.any((right_licks >= go_times[i]) & (right_licks < min(stop[i], go_times[i] + 1.5)))
    if l_post and not r_post:
        choice[i] = 0
    elif r_post and not l_post:
        choice[i] = 1
    else:
        choice[i] = 2
```

iii. The Step 5 notes said choice could come from "Trial lick direction / no response fields" and the trajectory shows the AI preferred to infer labels from behavioral events when possible. No explicit justification was found for preferring lick timestamps over `trial_instruction x outcome`; the implied rationale was to use a more direct behavioral signal.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI codes choice as `0=left`, `1=right`, `2=no lick`, then repeats the per-trial value across all 80 time bins in the output tensor.

ii.
```python
if l_post and not r_post:
    choice[i] = 0
elif r_post and not l_post:
    choice[i] = 1
else:
    choice[i] = 2
```

```python
out = np.vstack([
    np.full(N_BINS, choice[i], dtype=np.int64),
    np.full(N_BINS, outcome[i], dtype=np.int64),
    np.full(N_BINS, early[i], dtype=np.int64),
    tongue_trial.astype(np.int64),
])
```

iii. The notes say the decoder output should include a no-lick class and that per-trial categorical outputs can be repeated across bins so all outputs share one `(n_output, n_timepoints)` shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` column when present, with a fallback derivation from choice if the field is missing.

ii.
```python
if 'outcome' in table:
    ov = np.array([str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() for v in table['outcome']], dtype=object)
    ...
else:
    outcome[:] = np.where(choice == 2, 0, 2)
```

iii. The Step 5 notes identify "Trial outcome fields" as the source. The extra fallback is another robustness heuristic rather than a paper- or code-derived choice.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI lowercases the `outcome` strings and maps them to `0=ignore`, `1=miss`, `2=hit`, allowing substring matches like `correct` or `incorrect`. It then repeats the per-trial value across bins.

ii.
```python
for i, s in enumerate(ov):
    if s == 'hit' or 'correct' in s:
        outcome[i] = 2
    elif s == 'miss' or 'error' in s or 'incorrect' in s:
        outcome[i] = 1
    elif s == 'ignore' or 'no' in s:
        outcome[i] = 0
    else:
        outcome[i] = 0 if choice[i] == 2 else 2
```

iii. The notes say outcome should map to the requested three categories and remain a per-trial decoded variable.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken from the trial-table `early_lick` field when available.

ii.
```python
if 'early_lick' in table:
    ev = np.asarray(table['early_lick'])
    early = np.array([
        1 if str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() == 'early' else 0
        for v in ev
    ], dtype=np.int64)
```

iii. The Step 5 notes explicitly map "Early lick indicator" to a per-trial categorical output and say these trials should be retained because early lick is itself a target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `early` to 1 and everything else to 0, then repeats the per-trial value across all 80 bins.

ii.
```python
early = np.array([
    1 if str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() == 'early' else 0
    for v in ev
], dtype=np.int64)
```

```python
np.full(N_BINS, early[i], dtype=np.int64)
```

iii. The notes say this output should be binary no/yes and retained as part of the decoder task.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `Camera0_side_TongueTracking` timestamps and data, specifically the second column of the tracking data array (`data[:, 1]`) as `y`. It does not use the likelihood column to determine visibility.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:] if 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps' in f else np.array([])
tongue_xy = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:] if 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data' in f else np.zeros((0,2))
```

```python
def discretize_tongue_y(data_xy):
    y = np.asarray(data_xy[:, 1], dtype=float)
    visible = np.isfinite(y)
```

iii. The Step 5 notes identify "Camera0_side_TongueTracking y coordinate" as the source and mention that low-confidence samples should become not visible, but that justification was not carried into the final code. The final implementation implicitly treats finite `y` as visible.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes session-wide 40th and 60th percentiles from all finite raw tongue-y frames, discretizes each frame into classes 0/1/2, then for each trial/bin assigns the majority class among frames in that bin; bins with no valid class become 3 (`not visible`).

ii.
```python
q40, q60 = np.nanpercentile(y[visible], [40, 60])
out = np.full(len(y), 3, dtype=np.int64)
out[visible & (y < q40)] = 0
out[visible & (y >= q40) & (y <= q60)] = 1
out[visible & (y > q60)] = 2
```

```python
for b in range(N_BINS):
    m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
    if np.any(m):
        vals = labels[m]
        vals = vals[vals != 3]
        out[b] = 3 if len(vals) == 0 else np.bincount(vals, minlength=3).argmax()
```

iii. The Step 5 notes justify per-session percentile discretization with a not-visible class, but the trajectory/notes do not justify the final choices to threshold raw frames rather than 50 ms means, or to use majority vote within bins. Those appear to be implementation simplifications.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. It is thresholded using the 40th and 60th percentiles of all finite session-wide tongue-y frame values, with classes `0` for below 40th, `1` for 40th-60th inclusive, `2` for above 60th, and `3` for bins with no assigned visible class.

ii.
```python
q40, q60 = np.nanpercentile(y[visible], [40, 60])
out[visible & (y < q40)] = 0
out[visible & (y >= q40) & (y <= q60)] = 1
out[visible & (y > q60)] = 2
```

iii. The notes justify the 40th/60th per-session thresholds because that is what the decoder instructions requested. No explicit note justifies using raw-frame percentiles rather than binned means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial, the AI selects tongue frames in `[go + T_START, go + T_END)`, converts their timestamps to go-relative time, and bins them on the same `BIN_EDGES` used elsewhere.

ii.
```python
if len(tongue_ts):
    m = (tongue_ts >= go + T_START) & (tongue_ts < go + T_END)
    tongue_trial = bin_tongue_for_trial(tongue_ts[m], tongue_disc[m], go)
```

```python
rel = timestamps - go_time
for b in range(N_BINS):
    m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
```

iii. The Step 5 notes say tongue output should be time-varying and aligned to the shared 50 ms go-cue-centered window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missingness with permissive fallbacks and heuristics. Examples: `safe_float` converts strings like `N/A` to `NaN`; missing sample-start information falls back to `trial_table['start_time']`; missing photostim tables fall back to event streams; missing tongue tracking yields an all-`not_visible` output; unknown brain regions become `'unknown'`; sessions/trials that produce empty or all-zero neural extractions are skipped.

ii.
```python
def safe_float(x):
    ...
    if s.lower() in {'n/a', 'na', 'none', 'nan', ''}:
        return np.nan
```

```python
sample_starts = np.asarray(trial_table['start_time'], dtype=float)
...
tongue_trial = np.full(N_BINS, 3, dtype=np.int64)
...
return np.array(['unknown'] * n_units, dtype=object)
...
if np.allclose(trial_mats, 0):
    continue
```

iii. The notes describe this generally as "Handle missing data appropriately (consult references, use sensible defaults, document)." The concrete justifications that appear later are mainly practical: fix validation warnings, keep the pipeline running across files, and preserve decoder-compatible outputs even when streams are absent.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's notes identify spike binning as the major bottleneck in the initial implementation, then describe session-wide pre-binning of spikes as the key speedup. In the final code, the likely dominant costs are reading each NWB file, histogramming spikes for every retained unit, and the per-trial loops that slice trial windows and bin tongue labels.

ii.
```python
session_fr = np.zeros((len(good_inds), len(session_edges) - 1), dtype=np.float32)
for j, u in enumerate(good_inds):
    st = get_spike_times_for_unit(f['units'], int(u))
    m = (st >= session_start) & (st < session_stop)
    counts, _ = np.histogram(st[m], bins=session_edges)
    session_fr[j] = counts.astype(np.float32) / BIN_SIZE
```

iii. Step 6 says "Per-trial per-unit spike binning is likely the main bottleneck," and Step 7 says pre-binning spikes once per session reduced runtime dramatically.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain unvectorized: scanning sample events per trial when counts do not match, deriving choice/outcome trial by trial, histogramming one unit at a time, iterating through all trials to build outputs, and iterating through all bins in `bin_tongue_for_trial`.

ii.
```python
for i in range(len(go_times)):
    cand = sample_event_times[(sample_event_times >= start_times[i]) & (sample_event_times <= stop_times[i])]
```

```python
for i in range(n):
    l_post = np.any((left_licks >= go_times[i]) & (left_licks < min(stop[i], go_times[i] + 1.5)))
```

```python
for j, u in enumerate(good_inds):
    ...
for b in range(N_BINS):
    ...
```

iii. The notes explicitly discuss vectorization as a goal and record one major speedup, but they also acknowledge that some inference and processing remained heuristic and iterative.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work. It checks `np.allclose(trial_mats, 0)` twice in a row. It repeatedly scans `sample_event_times` against per-trial windows if sample-event counts do not match trial counts. It also recomputes per-trial photostim and tongue bin assignments inside the trial loop.

ii.
```python
if np.allclose(trial_mats, 0):
    continue
if np.allclose(trial_mats, 0):
    continue
```

```python
for i in range(len(go_times)):
    cand = sample_event_times[(sample_event_times >= start_times[i]) & (sample_event_times <= stop_times[i])]
```

iii. No explicit justification was documented for the duplicated zero check. The repeated per-trial scans appear to be straightforward implementation choices rather than deliberate algorithmic decisions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some values that are not used downstream: `tongue_thr` is returned by `discretize_tongue_y` but never used; `ps_starts`/`ps_stops` are loaded even when the trial table already supplies photostim timing; `TRIAL_CANDIDATES` and helpers support many heuristic fallbacks that are mostly bypassed on this dataset; `get_table_dict` is defined but unused.

ii.
```python
tongue_disc, tongue_thr = discretize_tongue_y(tongue_xy) if len(tongue_ts) else (np.array([], dtype=np.int64), (np.nan, np.nan))
```

```python
ps_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:] if 'acquisition/BehavioralEvents/photostim_start_times/timestamps' in f else np.array([])
ps_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:] if 'acquisition/BehavioralEvents/photostim_stop_times/timestamps' in f else np.array([])
```

```python
def get_table_dict(group):
    ...
```

iii. The trajectory and notes justify the extra fallbacks as robustness during development, especially while the AI was still inferring the dataset schema and fixing validation issues. The final script still carries some of that exploratory machinery even though it is not needed for the final dataset path.
