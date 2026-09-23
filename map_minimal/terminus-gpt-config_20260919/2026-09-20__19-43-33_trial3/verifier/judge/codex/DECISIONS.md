# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by recursively finding every `*.nwb` file under `/app/data`, sorting the paths, and then opening each file once with `pynwb.NWBHDF5IO`. Within each file it reads trials from `nwb.trials`, units from `nwb.units`, and event/time-series data from `nwb.acquisition`.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
...
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
    sid = str(nwb.subject.subject_id)
    tr = nwb.trials
    ...
    events = nwb.acquisition['BehavioralEvents'].time_series
```

iii. In the trajectory, the agent said it would inspect the NWB layout and then “stream one NWB at a time to control memory” and later described the dataset as “174 NWB sessions” that should be processed sequentially.

## 1-b. How are the data split into subjects?

i. The AI splits subjects using the subject id embedded in each NWB filename, not from the NWB subject field itself. It takes the `sub-<id>` prefix from each filename, builds the sorted unique subject list once, and assigns each session a `subject_idx`.

ii.
```python
subjects = sorted({p.name.split('_')[0].replace('sub-', '') for p in files})
subject_to_idx = {s:i for i,s in enumerate(subjects)}
...
sid = str(nwb.subject.subject_id)
...
subject_idx.append(subject_to_idx[sid])
```

iii. In the trajectory, the agent repeatedly referred to the NWB release being organized one file per session under `sub-<id>` naming and treated those encoded ids as the subject split.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Sessions are ordered by the sorted file list, and each processed file contributes one session to `neural`, `input`, `output`, and `session_info`.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
...
for si, path in enumerate(files):
    ...
    neural.append(sess_n); inputs.append(sess_i); outputs.append(sess_o)
    session_info.append({'file': path.name, 'subject': sid, 'n_trials': ntr,
                         'n_neurons': len(kept_idx), ...})
```

iii. The trajectory consistently described the data as “174 NWB sessions” and planned a “streaming converter” that processes “one NWB at a time.”

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table rows as the trial list via `start_time` and `stop_time`. It then assigns go-cue and tone events to each trial by finding the first event timestamp that falls inside the trial interval.

ii.
```python
starts = np.asarray(tr['start_time'][:], float)
stops = np.asarray(tr['stop_time'][:], float)
ntr = len(starts)

go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
tone = first_event_in_trial(events['sample_start_times'].timestamps[:], starts, stops)
```

```python
def first_event_in_trial(times, starts, stops):
    ans = np.full(len(starts), np.nan)
    for i, (a, b) in enumerate(zip(starts, stops)):
        j = np.searchsorted(times, a, side='left')
        if j < len(times) and times[j] < b:
            ans[i] = times[j]
    return ans
```

iii. In the trajectory, the agent said “Go events are one timestamp per trial” and that “tone/sample events can include extra transitions,” so it chose to assign events by trial interval rather than array position.

## 1-e. How are trials filtered based on quality controls?

i. The AI effectively does not apply behavioral trial QC filtering. It keeps all trials in each retained session, provided each trial has a go cue and tone onset; its metadata explicitly says the trial filter is “all trials with go cue and tone onset.”

ii.
```python
if np.any(~np.isfinite(go)):
    raise ValueError(f'{path}: {np.sum(~np.isfinite(go))} trials lack go cue')
...
if np.any(~np.isfinite(tone)):
    raise ValueError(f'{path}: {np.sum(~np.isfinite(tone))} trials lack tone onset')
...
'trial_filter': 'all trials with go cue and tone onset',
```

iii. The trajectory said the plan was to retain “all released behavioral sessions and trials,” and after counting the NWBs the agent wrote that it would “likely retain all available sessions unless an explicit exclusion is documented.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `nwb.units['spike_times']`, with per-trial alignment coming from go-cue timestamps in `BehavioralEvents/go_start_times`.

ii.
```python
spike_col = nwb.units['spike_times']
spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
spike_ends = np.asarray(spike_col.data[:], dtype=np.int64)
...
go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
```

iii. In the trajectory, the agent stated that spikes should be represented as “firing rates” aligned to “go-cue onset,” and later optimized around bulk-loading the ragged `spike_times` vector.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times into firing rates in Hz by counting spikes in 80 nonoverlapping 50 ms bins from `-2.5` to `+1.5` seconds around go cue. For each trial it forms absolute bin edges `go + EDGES`, uses `np.searchsorted` on each unit’s sorted spike train, differences the insertion indices, and divides by `DT`.

ii.
```python
START, END, DT = -2.5, 1.5, 0.05
EDGES = np.linspace(START, END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

```python
def bin_spikes(spike_times, go):
    out = np.empty((len(spike_times), 80), dtype=np.float32)
    abs_edges = go + EDGES
    for u, st in enumerate(spike_times):
        st = np.asarray(st)
        out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
    return out
```

iii. The trajectory says the agent matched the reference “half-open bins and firing rates” and explicitly switched from `np.histogram` to `np.searchsorted` to preserve the same bin convention while improving speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `unit_quality == 'good'` and a nonempty, non-`nan` `anno_name`. Sessions with no such units are skipped entirely.

ii.
```python
quality = np.asarray(nwb.units['unit_quality'][:], dtype=str)
annotation = np.char.strip(np.asarray(nwb.units['anno_name'][:], dtype=str))
keep = (quality == 'good') & (np.char.str_len(annotation) > 0) & (np.char.lower(annotation) != 'nan')
kept_idx = np.flatnonzero(keep)
if len(kept_idx) == 0:
    print('  skipped: no good annotated units', flush=True)
    continue
```

iii. In the trajectory, the agent argued that the preprocessing repository “retains only units with both ephys and histology,” concluded that this could be implemented as “classifier-good units with nonempty anatomical annotations,” and then decided to use `unit_quality == 'good'` plus histology.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s neural data is aligned to go-cue onset. The AI computes bin centers and edges relative to go cue, converts them to absolute times by adding the trial’s go time, and bins spikes against those absolute edges.

ii.
```python
go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
...
abs_edges = go + EDGES
...
bt = go[ti] + CENTERS
```

iii. The trajectory repeatedly says the decoder data should be “aligned to go-cue onset” and that the output should have “80 half-open 50 ms bins spanning [-2.5, 1.5) around go cue.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins and no further temporal rebinning. Each trial has exactly 80 time bins covering the requested 4 s window.

ii.
```python
START, END, DT = -2.5, 1.5, 0.05
EDGES = np.linspace(START, END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. The trajectory explicitly states the plan to use “80 half-open 50 ms bins spanning [-2.5, 1.5)” and treats this as matching the instruction and reference histogram semantics.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and the trial’s go cue. The AI picks the first sample/tone event that falls inside each trial interval.

ii.
```python
tone = first_event_in_trial(events['sample_start_times'].timestamps[:], starts, stops)
...
time_from_tone = (bt - tone[ti]).astype(np.float32)
```

iii. In the trajectory, the agent said that sample/tone events can have extra transitions and therefore “tone onset should be assigned as the first sample-start timestamp within each trial.”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI subtracts the selected tone onset from each neural-bin center in absolute session time, yielding a continuous time-varying value in seconds.

ii.
```python
bt = go[ti] + CENTERS
time_from_tone = (bt - tone[ti]).astype(np.float32)
sess_i.append(np.vstack((time_from_tone, photo)).astype(np.float32, copy=False))
```

iii. The trajectory described this as “time from tone onset … at bin centers,” using the same 80-bin grid as the neural data.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled at the exact same 80 bin centers used for the neural representation. The AI first builds `bt = go + CENTERS` on the neural timeline and then shifts that timeline by the tone time.

ii.
```python
bt = go[ti] + CENTERS
sess_n.append(bin_spikes(spikes, go[ti]))
time_from_tone = (bt - tone[ti]).astype(np.float32)
```

iii. The trajectory’s plan explicitly grouped “firing rates in Hz” and “time from tone onset … at bin centers” under one shared go-cue-aligned 80-bin representation.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, not from the per-trial columns in the trials table.

ii.
```python
ps = np.asarray(events['photostim_start_times'].timestamps[:], float)
pe = np.asarray(events['photostim_stop_times'].timestamps[:], float)
```

iii. In the trajectory, the agent inspected the event streams and concluded that photostimulation should be represented continuously using the recorded onset and offset times on the same session clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary time series per trial. For each neural bin center, it marks the bin as 1 if that absolute time falls inside any photostimulation interval `[start, stop)`, otherwise 0.

ii.
```python
photo = np.zeros(80, dtype=np.float32)
if len(ps):
    photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. The trajectory says the implementation should represent “photostimulation state at bin centers” as one of the decoder inputs.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by evaluating stimulation on/off state at the same absolute bin centers `bt = go + CENTERS` used for neural data, so it shares the neural time base directly.

ii.
```python
bt = go[ti] + CENTERS
...
photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. The trajectory framed photostimulation as another time-varying input “at bin centers” on the go-cue-aligned grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a direct choice column. The AI derives it from the trials-table `trial_instruction` and `outcome` fields.

ii.
```python
instruction = np.asarray(tr['trial_instruction'][:], dtype=str)
outcome = np.asarray(tr['outcome'][:], dtype=str)
...
if outcome[ti] == 'ignore':
    choice = 2
elif outcome[ti] == 'hit':
    choice = 0 if instruction[ti] == 'left' else 1
else:
    choice = 1 if instruction[ti] == 'left' else 0
```

iii. In the trajectory, the agent stated that “choice can be mapped exactly from outcome and instruction: hit = instructed side, miss = opposite side, ignore = no lick.”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0=left`, `1=right`, `2=no lick` and writes that same per-trial value into all 80 output time bins.

ii.
```python
if outcome[ti] == 'ignore':
    choice = 2
elif outcome[ti] == 'hit':
    choice = 0 if instruction[ti] == 'left' else 1
else:
    choice = 1 if instruction[ti] == 'left' else 0
...
o[0] = choice
```

iii. The trajectory explicitly chose a 3-class categorical choice output and said the first three outputs would be per-trial labels repeated across time to keep a rectangular output array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trials-table `outcome` column.

ii.
```python
outcome = np.asarray(tr['outcome'][:], dtype=str)
```

iii. The trajectory treated outcome as a direct per-trial label already present in NWB and did not propose any further derivation.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, and `hit` to `0`, `1`, and `2`, then repeats the coded value across all 80 bins.

ii.
```python
out_map = {'ignore':0, 'miss':1, 'hit':2}
...
o[1] = out_map[outcome[ti]]
```

iii. The trajectory grouped outcome with the per-trial categorical labels that are repeated over time for decoder formatting.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` column.

ii.
```python
early = np.asarray(tr['early_lick'][:], dtype=str)
```

iii. The trajectory treated this as a direct NWB trial label alongside instruction and outcome.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI encodes `early` as `1` and everything else as `0`, then repeats that per-trial value across all 80 bins.

ii.
```python
o[2] = 1 if early[ti] == 'early' else 0
```

iii. The trajectory again treated early lick as one of the per-trial categorical outputs repeated over the decoder time axis.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 as `tongue_y`, column 2 as `tongue_likelihood`, and the corresponding camera timestamps.

ii.
```python
tts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
cam_t = np.asarray(tts.timestamps[:], dtype=float)
cam = np.asarray(tts.data[:], dtype=float)
...
y = cam[idx,1]
vis = np.isfinite(y) & np.isfinite(cam[idx,2]) & (cam[idx,2] >= DLC_THRESHOLD)
```

iii. In the trajectory, the agent noted that TongueTracking columns are explicitly `(tongue_x, tongue_y, tongue_likelihood)` and that visibility should be determined from likelihood.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a visibility threshold `DLC_THRESHOLD = 0.9`, computes the 40th and 60th percentiles of all session-wide visible `tongue_y` samples, then for each neural bin chooses the nearest camera sample and assigns its y-value to one of three percentile-based classes. Invisible bins get class `3`.

ii.
```python
DLC_THRESHOLD = 0.9
...
visible_all = np.isfinite(cam[:,1]) & np.isfinite(cam[:,2]) & (cam[:,2] >= DLC_THRESHOLD)
if not np.any(visible_all):
    q40 = q60 = np.nan
else:
    q40, q60 = np.percentile(cam[visible_all,1], [40, 60])
```

```python
idx = np.searchsorted(cam_t, bt)
...
y = cam[idx,1]
vis = np.isfinite(y) & np.isfinite(cam[idx,2]) & (cam[idx,2] >= DLC_THRESHOLD)
tongue_cat = np.full(80, 3, dtype=np.int8)
tongue_cat[vis & (y < q40)] = 0
tongue_cat[vis & (y >= q40) & (y <= q60)] = 1
tongue_cat[vis & (y > q60)] = 2
```

iii. The trajectory said tongue categories should use “session-wide 40th/60th percentiles over DLC-visible samples (likelihood >= 0.9), with category 3 for invisible bins,” and earlier argued that each neural bin should receive the nearest camera sample.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The categories are `0` for visible values below the session’s 40th percentile, `1` for visible values between the 40th and 60th percentiles, `2` for visible values above the 60th percentile, and `3` for bins treated as not visible.

ii.
```python
tongue_cat = np.full(80, 3, dtype=np.int8)
tongue_cat[vis & (y < q40)] = 0
tongue_cat[vis & (y >= q40) & (y <= q60)] = 1
tongue_cat[vis & (y > q60)] = 2
...
'output_values': [['left','right','no lick'], ['ignore','miss','hit'],
                  ['no','yes'], ['below 40th percentile','40th to 60th percentile',
                                 'above 60th percentile','not visible']],
```

iii. The trajectory explicitly planned a four-class scheme with 40th/60th session percentiles and a separate invisible class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output by evaluating one camera sample per neural bin: for each go-cue-aligned neural bin center, it picks the nearest camera frame in absolute time and uses that frame’s tongue y and likelihood.

ii.
```python
bt = go[ti] + CENTERS
idx = np.searchsorted(cam_t, bt)
idx = np.clip(idx, 1, len(cam_t)-1)
prev = idx - 1
idx = np.where(np.abs(cam_t[prev]-bt) <= np.abs(cam_t[idx]-bt), prev, idx)
```

iii. In the trajectory, the agent decided that “each neural bin receives the nearest camera sample,” using the shared global session timestamps for alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles a few missing-data cases, but narrowly. It excludes units with empty or literal `nan` `anno_name`, skips sessions with no kept units, raises an error if any trial lacks a go cue or tone onset, and assigns tongue class `3` to bins without a sufficiently confident visible sample.

ii.
```python
keep = (quality == 'good') & (np.char.str_len(annotation) > 0) & (np.char.lower(annotation) != 'nan')
if len(kept_idx) == 0:
    print('  skipped: no good annotated units', flush=True)
    continue
```

```python
if np.any(~np.isfinite(go)):
    raise ValueError(...)
if np.any(~np.isfinite(tone)):
    raise ValueError(...)
...
tongue_cat = np.full(80, 3, dtype=np.int8)
```

iii. The trajectory shows that the agent noticed literal `nan` region labels after verification and patched the filter to remove them. It also treated missing go/tone as a hard error and missing tongue visibility as an explicit “not visible” output class.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts are loading each NWB session, bulk-loading spike and camera arrays, repeatedly binning spikes for every trial by looping over all units, and writing the final large pickle.

ii.
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
...
spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
...
cam = np.asarray(tts.data[:], dtype=float)
...
for ti in range(ntr):
    sess_n.append(bin_spikes(spikes, go[ti]))
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In the trajectory, the agent first identified repeated histogramming and per-unit HDF5 reads as bottlenecks, then replaced them with `searchsorted` and bulk ragged-array loading. It still described the full conversion as dominated by reading 174 NWB files, camera streams, and serializing an approximately 11 GB pickle.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining vectorization opportunity is the per-trial loop that calls `bin_spikes` separately for every trial, causing another inner loop over all units each time. The per-trial photostim and tongue alignment work is also still looped at trial granularity.

ii.
```python
for ti in range(ntr):
    bt = go[ti] + CENTERS
    sess_n.append(bin_spikes(spikes, go[ti]))
    ...
```

```python
def bin_spikes(spike_times, go):
    ...
    for u, st in enumerate(spike_times):
        ...
```

iii. The trajectory explicitly recognized earlier that the initial implementation was too slow because “each trial calls `np.histogram` on every unit's full-session spike train.” It optimized the inner method but did not eliminate the repeated per-trial recomputation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly re-bins the same unit spike trains from scratch once per trial, and re-tests each trial’s bin centers against all photostimulation intervals. It also redoes the nearest-camera-frame search on every trial separately.

ii.
```python
for ti in range(ntr):
    bt = go[ti] + CENTERS
    sess_n.append(bin_spikes(spikes, go[ti]))
    ...
    if len(ps):
        photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
    ...
    idx = np.searchsorted(cam_t, bt)
```

iii. The trajectory’s optimization steps show the agent repeatedly diagnosing this kind of repeated work, first for `np.histogram`, then for HDF5 spike access, but it left the per-trial recomputation pattern in place.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Most computed outputs are retained, but the code does add session-level metadata fields such as `tongue_q40`, `tongue_q60`, and `dlc_likelihood_threshold` that are not used by the downstream decoder. It also imports `json` but does not use it.

ii.
```python
import json, pickle, gc
...
session_info.append({'file': path.name, 'subject': sid, 'n_trials': ntr,
                     'n_neurons': len(kept_idx), 'tongue_q40': float(q40),
                     'tongue_q60': float(q60), 'dlc_likelihood_threshold': DLC_THRESHOLD})
```

iii. The trajectory indicates these fields were kept for sanity checking and verification rather than because they were required for the decoder interface.
