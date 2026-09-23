# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every published `*_behavior+ophys.nwb` file under `/app/data/sub-*` with `glob`, then opens each file directly with `h5py`. Within each file it reads behavior arrays from `processing/behavior/BehavioralTimeSeries/...` and neural data from `processing/ophys/Deconvolved/plane0/data`.

ii.
```python
files = sorted(glob.glob(os.path.join(ROOT, 'sub-*', '*_behavior+ophys.nwb')))
...
def convert_session(path):
    with h5py.File(path, 'r') as f:
        def beh(name):
            return f[B + name + '/data'][:]
```

iii. The trajectory repeatedly says the converter should use "all 152 published behavior+ophys sessions" and process "all deposited behavior+ophys sessions" (steps 11 and 19). It also justified direct HDF5 access because the NWBs already expose frame-aligned behavior and processed deconvolved activity (steps 3, 7, 19).

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory names `sub-<id>` of the NWB files. The unique directory-derived IDs are sorted and stored in `data['subjects']`.

ii.
```python
subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-', '') for p in files},
                  key=lambda x: int(x[1:]) if x.startswith('m') and x[1:].isdigit() else x)
submap = {s: i for i, s in enumerate(subjects)}
```

iii. The trajectory says there are 11 subjects and 152 sessions, with files grouped under per-subject directories (steps 7, 9, 10). That was the basis for using directory names as mouse IDs.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The converter loops over the sorted NWB path list and appends one session entry to `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx` per file.

ii.
```python
for k, path in enumerate(files):
    neu, inp, out, nc, nt = convert_session(path)
    subj = os.path.basename(os.path.dirname(path)).replace('sub-', '')
    data['neural'].append(neu); data['input'].append(inp); data['output'].append(out)
    data['subject_idx'].append(submap[subj])
```

iii. The trajectory states that "all 152 NWB sessions" are the published imaging sessions and should be included as sessions (steps 9, 10, 19).

## 1-d. How are the data split into trials?

i. Trials are defined by explicit `trial_start` and `teleport` markers. The code finds all start indices where `trial_start > 0`, all end indices where `teleport > 0`, then pairs each start with the first later teleport before the next start. The emitted trial interval is `[trial_start, teleport)`.

ii.
```python
starts = np.flatnonzero(beh('trial_start') > 0)
ends = np.flatnonzero(beh('teleport') > 0)
pairs = trial_pairs(starts, ends)
...
def trial_pairs(start, end):
    ...
    if b < next_a and b > a:
        pairs.append((int(a), b))
```

iii. The trajectory explicitly rejects splitting on the `trial number` stream because it "changes during ITIs in a few files" and says explicit start/teleport markers are authoritative (steps 7, 8, 11, 19).

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a per-trial duration filter. It only raises an error if a session has fewer than two complete trial pairs.

ii.
```python
pairs = trial_pairs(starts, ends)
if len(pairs) < 2:
    raise ValueError(f'fewer than two complete trials: {path}')
```

iii. The trajectory says most sessions have valid trial counts and "no fixed trial-count filter should be imposed" (step 9). It does not mention any minimum-length trial removal.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the deposited Suite2p deconvolved signal in `processing/ophys/Deconvolved/plane0/data`, filtered by `iscell` after mapping ROI columns through the NWB ROI table-region.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] == 1
dset = f['processing/ophys/Deconvolved/plane0/data']
...
cell_idx = np.flatnonzero(iscell[roi_rows])
deconv = dset[:, cell_idx]
```

iii. The trajectory says the NWBs contain "processed deconvolved neural activity" and repeatedly states the converter will use "native frame-aligned Suite2p deconvolved activity" (steps 3, 7, 9, 11, 19).

## 2-b. How is the `neural` data processed?

i. The AI does not recompute neural events from fluorescence. It takes the deposited deconvolved values as-is, subsets them to curated cells, converts each trial slice to `float32`, transposes to `(neurons, time)`, and stores those trial matrices.

ii.
```python
deconv = dset[:, cell_idx]
...
neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())
```

iii. The trajectory explicitly argues for using the already deposited deconvolved activity at the "native imaging rate" rather than reprocessing raw fluorescence (steps 9, 11, 19).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter is Suite2p's `iscell` flag, applied after ROI remapping for sessions where the time series columns are a subset of the segmentation table.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] == 1
...
roi_rows = np.asarray(f[rois_path][:], dtype=int)
...
cell_idx = np.flatnonzero(iscell[roi_rows])
```

iii. The trajectory says "`iscell[:,0] == 1` is the likely neuron filter" and later adds the ROI remapping fix to preserve correct `iscell` application in m17/m18 (steps 5, 13, 14, 16, 19).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned by slicing the deconvolved matrix with the same `[trial_start, teleport)` frame indices used for the trial definition, so trial time 0 is the `trial_start` frame.

ii.
```python
for i, (a, b) in enumerate(pairs):
    n = b - a
    neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())
```

iii. The trajectory says temporal alignment should be based on trial start and that the explicit `trial_start` marker should define complete trials (steps 7, 11, 19).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native imaging frame rate. The code hardcodes `RATE = 15.5078125`, so each bin is `1000 / RATE` milliseconds, and no rebinning or interpolation is applied.

ii.
```python
RATE = 15.5078125
DT_MS = 1000.0 / RATE
...
'time_bin_size': DT_MS,
...
inp[0] = np.arange(n, dtype=np.float32) / RATE
```

iii. The trajectory says the imaging rate is `15.5078125 Hz`, behavior is already aligned to those frames, and the converter should use the "native imaging rate" without extra resampling (steps 5, 11, 19).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI derives trial timing from the fixed frame rate and trial length, not from per-trial NWB timestamps. It only uses the `position` timestamps to place reward events into trials.

ii.
```python
ts = f[B + 'position/timestamps'][:]
...
inp[0] = np.arange(n, dtype=np.float32) / RATE
```

iii. The trajectory says the behavior streams are already frame-aligned and share the imaging rate, so direct frame counting was treated as sufficient for trial time (steps 3, 5, 11).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For a trial with `n` frames, the code generates `0, 1/RATE, 2/RATE, ...` with `np.arange(n) / RATE`. It does not subtract the actual per-trial timestamp vector.

ii.
```python
inp = np.empty((4, n), dtype=np.float32)
inp[0] = np.arange(n, dtype=np.float32) / RATE
```

iii. The trajectory justifies this by saying the data are already aligned to a constant imaging frame rate and can therefore use native frame counts as time from trial start (steps 5, 11, 19).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time is aligned by construction: both the time vector and the neural matrix use the same trial length `n = b - a` and the same trial frame slice `[a:b]`.

ii.
```python
for i, (a, b) in enumerate(pairs):
    n = b - a
    neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())
    ...
    inp[0] = np.arange(n, dtype=np.float32) / RATE
```

iii. The trajectory repeatedly says the behavior streams are already frame-aligned to the imaging data, so no extra synchronization step is needed (steps 3, 7, 19).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
environment = beh('environment')
...
ev = environment[a:b]
```

iii. The trajectory notes that environment is encoded as 0/1 in-session, with `-1` outside valid periods, and therefore can be read directly from the NWB behavior stream (steps 5 and 7).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial the code drops negative values, takes the median of the remaining samples, rounds it to an integer, and broadcasts that constant value across all time bins of the trial.

ii.
```python
ev = environment[a:b]
ev = ev[ev >= 0]
env = int(np.rint(np.median(ev))) if len(ev) else 0
inp[1] = env
```

iii. The trajectory says environment is sessionwise 0 or 1, with occasional `-1` values outside valid scanning, so a within-trial summary avoids those invalid markers (steps 5, 7, 11).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from the NWB `trial number` time series. It is the loop index `i` over the paired `[trial_start, teleport)` trials.

ii.
```python
for i, (a, b) in enumerate(pairs):
    ...
    inp[2] = i
```

iii. The trajectory explicitly rejects the NWB `trial number` stream because it changes during inter-trial intervals in some files and says explicit start/teleport markers should define trials instead (steps 7, 8, 11).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing is applied beyond using the sequential trial index and broadcasting it across the trial.

ii.
```python
inp[2] = i
```

iii. The trajectory treats trial number as a per-trial identifier attached to each paired trial interval (steps 8, 11, 19).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` event timestamps together with the trial boundaries and the shared behavior timestamps used to delimit each trial interval.

ii.
```python
ts = f[B + 'position/timestamps'][:]
reward_times = f[B + 'Reward/timestamps'][:]
rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)
```

iii. The trajectory says reward outcome should be derived from the sparse reward-event stream rather than trial-number changes, and that reward events are separate timestamped events that can be matched to trial intervals (steps 4, 7, 11).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first computes a per-trial `rewarded` flag from the reward timestamps. It then stores `0` for the first trial and, for each later trial, copies the previous trial's reward flag into every time bin.

ii.
```python
prev = 0
for i, (a, b) in enumerate(pairs):
    ...
    inp[3] = prev
    ...
    prev = int(rewarded[i])
```

iii. The trajectory says the decoder input should be previous trial outcome as a binary per-trial context variable, and using the already computed per-trial reward flags is the direct way to do that (steps 11 and 19).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` time series plus a per-trial reward-zone identity inferred from the `reward_zone` event stream. The code finds positions where `reward_zone > 0`, assigns the trial to the nearest nominal zone start among 80, 200, and 320 cm, and fills missing trial labels from the nearest labeled trial.

ii.
```python
position = beh('position')
zone_event = beh('reward_zone')
...
p = position[a:b][zone_event[a:b] > 0]
...
zones[i] = int(np.argmin(np.abs(ZONE_STARTS - np.median(p))))
zones = nearest_fill(zones)
```

iii. The trajectory says reward-zone entry positions cluster near three starts (~80, ~200, ~320 cm), and omission trials need label imputation from neighboring trials; it chose nearest-zone assignment plus nearest-neighbor fill rather than session metadata or a more global model (steps 8, 10, 11).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial the code converts the zone label to a start `z0` and end `z1 = z0 + 50`. It then sets distance to `pos - z0` before the zone, `0` inside the zone, and `pos - z1` after the zone.

ii.
```python
z0 = float(ZONE_STARTS[zones[i]])
z1 = z0 + ZONE_WIDTH
dist = np.where(pos < z0, pos - z0, np.where(pos > z1, pos - z1, 0.0))
```

iii. The trajectory says the requested output is naturally represented as signed distance to the 50-cm reward zone: negative before, zero inside, positive after (step 11).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized with `np.digitize` using the requested boundaries `-50, -10, 0, nextafter(0,+1), 10, 50`, yielding seven categories `0..6`.

ii.
```python
out[0] = np.digitize(dist, [-50, -10, 0, np.nextafter(0.0, 1.0), 10, 50])
```

iii. The trajectory says this preserves the instruction's special treatment of exactly `0 cm` while keeping the other signed-distance bins unchanged (step 11).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance is computed from `position[a:b]` inside the same per-trial loop and written into an output array of the same trial length `n`, so it is frame-aligned to the neural slice `deconv[a:b, :]`.

ii.
```python
neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())
...
pos = position[a:b]
...
out = np.empty((6, n), dtype=np.int8)
out[0] = np.digitize(dist, [-50, -10, 0, np.nextafter(0.0, 1.0), 10, 50])
```

iii. The trajectory says the behavior streams are already resampled to imaging frames, so slicing the same `[a:b]` interval gives direct alignment (steps 3, 7, 19).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii.
```python
position = beh('position')
...
pos = position[a:b]
```

iii. The trajectory identifies the NWB `position` stream as the frame-aligned corridor position variable to decode (steps 3, 4, 11).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No continuous preprocessing is applied beyond per-trial slicing. The per-frame position values are immediately discretized.

ii.
```python
pos = position[a:b]
...
out[1] = np.digitize(pos, [90, 180, 270, 360])
```

iii. The trajectory treats position as already aligned and already expressed in corridor centimeters, so discretization is the only required downstream processing (steps 4 and 11).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is binned into five equal-width 90-cm bins using boundaries `90, 180, 270, 360`, producing categories `0..4`.

ii.
```python
out[1] = np.digitize(pos, [90, 180, 270, 360])
```

iii. The trajectory says the track is 450 cm long and the decoder spec asks for five equal bins, so 90-cm bin edges are used (step 11).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same `[a:b]` trial slice and same trial length `n` as the neural data, so it is directly frame-aligned.

ii.
```python
for i, (a, b) in enumerate(pairs):
    n = b - a
    neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())
    pos = position[a:b]
```

iii. The trajectory says the behavior arrays are already frame-aligned to the imaging frames and therefore require no extra interpolation (steps 3 and 19).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii.
```python
lick = beh('lick')
...
out[3] = (lick[a:b] > 0).astype(np.int8)
```

iii. The trajectory notes that the NWB lick stream contains per-frame event counts and should therefore be thresholded to binary lick/no-lick (step 7).

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code binarizes the lick counts: any positive value becomes `1`, otherwise `0`.

ii.
```python
out[3] = (lick[a:b] > 0).astype(np.int8)
```

iii. The trajectory explicitly says the lick stream is cumulative/event-count-like and should be reduced to a binary output for the decoder (step 7).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is taken from the same `[a:b]` trial slice as the neural data and written into the same-length output matrix for that trial.

ii.
```python
for i, (a, b) in enumerate(pairs):
    n = b - a
    ...
    out = np.empty((6, n), dtype=np.int8)
    out[3] = (lick[a:b] > 0).astype(np.int8)
```

iii. The trajectory says behavior is already frame-aligned to the imaging frames, so the shared slice suffices for temporal alignment (steps 3 and 19).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the `reward_zone` event stream and the `position` time series. The event-positive positions within each trial are mapped to the nearest of the nominal zone starts `[80, 200, 320]`.

ii.
```python
zone_event = beh('reward_zone')
position = beh('position')
...
p = position[a:b][zone_event[a:b] > 0]
...
zones[i] = int(np.argmin(np.abs(ZONE_STARTS - np.median(p))))
```

iii. The trajectory says the `reward_zone` stream is not directly an A/B/C label and that reward-zone location should instead be inferred from where the event occurs in the corridor (steps 7, 8, 10, 11).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code computes a per-trial zone label by taking the median position of `reward_zone > 0` samples, assigning that median to the nearest nominal zone start, then filling missing labels from the nearest labeled trial in the same session. The final categorical output is `0, 1, 2` for A/B/C.

ii.
```python
zones = np.full(len(pairs), -1, dtype=np.int8)
for i, (a, b) in enumerate(pairs):
    p = position[a:b][zone_event[a:b] > 0]
    ...
    if len(p):
        zones[i] = int(np.argmin(np.abs(ZONE_STARTS - np.median(p))))
zones = nearest_fill(zones)
...
out[4] = zones[i]
```

iii. The trajectory says omission trials can have no zone-entry event, so they need imputation from neighboring trials, and the three physical reward-zone locations are stable enough for nearest-zone classification (steps 8 and 11).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward` timestamps together with the behavior timestamps and trial boundaries.

ii.
```python
ts = f[B + 'position/timestamps'][:]
reward_times = f[B + 'Reward/timestamps'][:]
rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)
```

iii. The trajectory says reward is represented as sparse timestamped events in the NWB and should be matched to trial intervals to decide rewarded versus omitted trials (steps 4, 7, 11).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code marks the outcome as `1` if any reward timestamp falls in `[ts[a], ts[b])`, otherwise `0`. That per-trial value is then broadcast across all time bins of the trial.

ii.
```python
rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)
...
out[5] = rewarded[i]
```

iii. The trajectory says the decoder output should be a binary per-trial reward outcome and that interval membership of sparse reward events is the direct criterion (steps 11 and 19).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles a small number of specific issues. It fills missing reward-zone labels from the nearest labeled trial, errors out if a session has fewer than two complete trials, and fixes sessions where the ROI time-series columns do not directly match the segmentation table by applying the NWB ROI mapping before `iscell` filtering. It does not include broader cropping or timestamp-consistency checks.

ii.
```python
def nearest_fill(labels):
    ...
    for i in missing:
        labels[i] = labels[known[np.argmin(np.abs(known - i))]]

if len(pairs) < 2:
    raise ValueError(f'fewer than two complete trials: {path}')

if rois_path in f:
    roi_rows = np.asarray(f[rois_path][:], dtype=int)
...
cell_idx = np.flatnonzero(iscell[roi_rows])
```

iii. The trajectory says omission or aborted trials can lack reward-zone entries and should be imputed from neighbors (steps 8 and 11). It also documents the m17/m18 ROI-mapping failure and the subsequent fix as a necessary correction for valid cell filtering (steps 13, 14, 16).

## 13-a. What are the most time-consuming steps of the code?

i. The dominant costs are reading large NWB files, slicing and storing large neural trial matrices for all sessions, and serializing the final pickle.

ii.
```python
for k, path in enumerate(files):
    neu, inp, out, nc, nt = convert_session(path)
...
with open(OUT, 'wb') as fh:
    pickle.dump(data, fh, protocol=4)
```

iii. The trajectory says conversion "reads and stores roughly 13 GB of selected neural activity before serializing the pickle" and later notes that after all sessions were converted in memory, the process was still busy serializing the large pickle (steps 12, 17, 18).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization opportunities are the per-trial loops over `pairs`: zone inference, reward-outcome membership tests, and trial-by-trial construction of neural/input/output lists. `nearest_fill` is also a scalar loop over missing trials.

ii.
```python
for i, (a, b) in enumerate(pairs):
    p = position[a:b][zone_event[a:b] > 0]
    ...

rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)

for i in missing:
    labels[i] = labels[known[np.argmin(np.abs(known - i))]]
```

iii. The trajectory does not discuss vectorization directly, but it does emphasize that the conversion is dominated by per-session NWB reading and per-trial extraction, which are the places where loop restructuring would matter (steps 11, 12, 19).

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans trial slices. It first loops over all trials to infer `zones`, then loops again over all trials to build `neural`, `input`, and `output`. Reward timestamps are also rescanned against every trial interval inside the list comprehension.

ii.
```python
for i, (a, b) in enumerate(pairs):
    p = position[a:b][zone_event[a:b] > 0]
    ...

rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)

for i, (a, b) in enumerate(pairs):
    ...
    pos = position[a:b]
    spd = speed[a:b]
```

iii. The trajectory says the pipeline infers zones first and then constructs the final per-trial decoder arrays, so the same trial boundaries are reused in multiple passes (steps 11 and 19).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is little explicit throwaway processing. The clearest extra work is metadata bookkeeping such as `session_info` and the returned `len(cell_idx), len(pairs)` summary values, which are not used by downstream decoding itself. The `.copy()` on each neural trial also adds an extra memory copy purely for storage convenience.

ii.
```python
session_info = []
...
session_info.append({'file': os.path.basename(path), 'subject': subj,
                     'n_neurons': nc, 'n_trials': nt})
...
neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())
...
return neural_trials, input_trials, output_trials, len(cell_idx), len(pairs)
```

iii. The trajectory focuses on getting a valid converted dataset and does not justify extra analysis-specific bookkeeping; these parts appear to be convenience metadata rather than decoder-required computation (steps 18 and 19).
