# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by directly reading NWB files from `/app/data` using h5py, bypassing the AllenSDK high-level API. It discovers files via glob pattern matching for `behavior_ophys_experiment_*.nwb`, reads metadata from `ophys_experiment_table.csv`, and processes each experiment file individually with `h5py.File()`.

ii.
```python
def find_files():
    files = sorted(glob.glob(str(DATA_ROOT / '**' / 'behavior_ophys_experiment_*.nwb'), recursive=True))
    return {int(re.search(r'(\d+)\.nwb$', f).group(1)): f for f in files}

def metadata_table(name):
    paths = glob.glob(str(DATA_ROOT / '**' / name), recursive=True)
    ...
    return pd.read_csv(paths[0])

def select_experiments(files):
    meta = metadata_table('ophys_experiment_table.csv')
    meta = meta[meta.ophys_experiment_id.isin(files)].copy()
    ...
```

iii. The AI chose direct h5py reads over the AllenSDK API for efficiency, avoiding the overhead of constructing full PyNWB objects. The CONVERSION_NOTES.md states: "Full PyNWB/AllenSDK object construction would load unnecessary images/templates and incur high overhead for 199 large NWBs."

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata table and sorted as strings.

ii.
```python
subjects=sorted({z['mouse_id'] for z in infos}); smap={x:i for i,x in enumerate(subjects)}
```

iii. The `mouse_id` field uniquely identifies each animal. The AI found 38 mice in the local subset.

## 1-c. How are the data split into sessions?

i. Each ophys experiment (single imaging plane) is treated as a separate session. Multi-plane sessions are NOT merged; each plane is its own session with its own neuron population. The AI explicitly decided this in CONVERSION_NOTES Step 5, Key Decision 1.

ii.
```python
for k,row in meta.iterrows():
    eid=int(row.ophys_experiment_id)
    n,x,y,info=process_experiment(row,files[eid],imap,args.show_processing and len(infos)<2)
    neural.append(n); inputs.append(x); outputs.append(y); infos.append(info)
```

iii. From CONVERSION_NOTES: "Session unit = ophys experiment/imaging plane: each file has a distinct simultaneous neuron population, matching paper decoding by plane. Multi-plane behavioral rows are intentionally repeated across distinct neural sessions."

## 1-d. How are the data split into trials?

i. Trials are defined using NWB `intervals/trials`, extracting `start_time` to `stop_time` for each retained trial. Trial duration is variable, and the time axis is rebinned into 100ms bins. The number of bins per trial is `floor((stop_time - start_time) / 0.1)`.

ii.
```python
for ti in tids:
    nbin = int(np.floor((stops[ti] - starts[ti]) / BIN_S + 1e-9))
    if nbin < 1:
        continue
    centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
```

iii. The AI used the SDK/NWB trial boundaries (start_time to stop_time) which define full trial windows including pre-change flashes and post-change response period.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only go and catch trials, excluding aborted and auto-rewarded. Additionally, passive session experiments are excluded, experiments without eye tracking data are excluded, and trials where running or pupil has no finite support are dropped. Sessions with fewer than 2 usable trials raise an error and are excluded.

ii.
```python
keep = (flags['go'] | flags['catch']) & ~flags['aborted'] & ~flags['auto_rewarded']
tids = np.flatnonzero(keep)
...
if not (np.isfinite(r).all() and np.isfinite(p).all()):
    continue
...
if len(neural) < 2:
    raise ValueError(f'{eid}: fewer than 2 usable trials')
```

Also in `select_experiments`:
```python
meta = meta[~meta.session_type.str.contains('passive', case=False, na=False)].copy()
eye_map = {eid: has_eye(files[eid]) for eid in meta.ophys_experiment_id.astype(int)}
meta = meta[meta.ophys_experiment_id.map(eye_map)].copy()
```

iii. The AI excluded passive sessions because "Passive rows are almost entirely miss/correct-reject because no task responses occur" and excluded experiments without eye tracking because "pupil is mandatory and cannot be reconstructed." The AI also drops individual trials where running or pupil interpolation yields non-finite values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from **detected calcium events** (`processing/ophys/event_detection/data`) rather than dF/F traces.

ii.
```python
ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
ots = np.asarray(h['processing/ophys/event_detection/timestamps'], dtype=float)
```

iii. From CONVERSION_NOTES Step 4: "Paper neural analyses use detected calcium events" and "Use detected event magnitude at ophys timestamps; do not recompute dF/F." The AI chose events over dF/F to match the paper's neural representation.

## 2-b. How is the `neural` data processed?

i. Event magnitudes are averaged into 100ms bins. For each bin, the mean of all native frames falling within that bin is computed. If a bin contains no native frames (possible at ~11 Hz), linear interpolation from adjacent frames is used as a fallback. The data is transposed from (time, neurons) to (neurons, time).

ii.
```python
BIN_S = 0.100

def event_bin_means(events, ts, starts, centers):
    nbin = len(centers); nc = events.shape[1]
    out = np.empty((nc, nbin), dtype=np.float32)
    edges = starts + np.arange(nbin + 1) * BIN_S
    lo = np.searchsorted(ts, edges[:-1], side='left')
    hi = np.searchsorted(ts, edges[1:], side='left')
    for j, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[:, j] = np.asarray(events[a:b], dtype=np.float32).mean(axis=0)
        else:
            k = np.searchsorted(ts, centers[j])
            k = min(max(k, 1), len(ts)-1)
            t0, t1 = ts[k-1], ts[k]
            w = 0.0 if t1 == t0 else (centers[j]-t0)/(t1-t0)
            out[:, j] = (1-w)*events[k-1] + w*events[k]
    return out
```

iii. The 100ms bin size was chosen as a common temporal resolution that works for both 31 Hz and 11 Hz experiments (about 3 and 1 native frames/bin respectively). The AI noted this "preserves 250 ms image flashes and 400 ms reference decoding windows."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All released post-QC ROIs present in the NWB event detection data are included.

ii. N/A (no filtering code)

iii. From CONVERSION_NOTES: "supplied NWBs are already post-release-QC, including experiment and ROI filtering. No extra neuron activity threshold is warranted."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. Bin centers begin at `start_time + 50ms` (half-bin offset) and extend in 100ms increments until `stop_time`.

ii.
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
n = event_bin_means(ev, ots, starts[ti], centers)
```

iii. The metadata records `temporal_alignment_event: 'trial start time on the ophys session clock'` and `off_start: 0.0`. The AI aligned to trial start rather than change_time because trials have variable pre-change periods.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100ms (0.1s). The native ophys frames (~32ms at 31 Hz or ~93ms at 11 Hz) are rebinned into fixed 100ms bins by averaging event magnitudes within each bin.

ii.
```python
BIN_S = 0.100
...
'time_bin_size':100.0
```

iii. From CONVERSION_NOTES Step 5: "Common bin = 100 ms: this is supported by both 31 Hz and 11 Hz experiments (about 3 and 1 native frames/bin), preserves 250 ms image flashes and 400 ms reference decoding windows, and keeps variable trial lengths manageable."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from stimulus presentation timestamps and template image names stored in the NWB file (`stimulus/presentation` and `stimulus/templates`). A per-bin image is assigned based on whether each 100ms bin center falls within a 250ms flash window after a presentation onset.

ii.
```python
flash_t, flash_name = presentation_streams(h)
...
image = np.zeros(nbin, dtype=np.int64)
left = np.searchsorted(flash_t, centers - 0.250, side='left')
right = np.searchsorted(flash_t, centers, side='right')
for j, (a, b) in enumerate(zip(left, right)):
    if b > a:
        ft = flash_t[b-1]
        if ft <= centers[j] < ft + 0.250:
            image[j] = image_to_idx.get(str(flash_name[b-1]), 0)
```

iii. The AI used the stimulus presentation timing directly rather than the trial table's `initial_image_name`/`change_image_name` fields. This includes "gray" (code 0) as a category for inter-stimulus intervals, because the instructions say "Image identity (of the image presented during the non-grey screen)."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global image vocabulary is built from all stimulus templates across all experiments. Each image name maps to an integer code, with "gray" as code 0. For each 100ms bin, the most recent flash onset within 250ms is identified; if a flash is active, its image code is assigned; otherwise the bin is labeled "gray" (0).

ii.
```python
def image_vocabulary(meta, files):
    names = set()
    for eid in meta.ophys_experiment_id.astype(int):
        with h5py.File(files[eid], 'r') as h:
            if 'stimulus/templates' in h:
                for g in h['stimulus/templates'].values():
                    if 'control_description' in g:
                        names.update(decode_strings(g['control_description'][:]).tolist())
    return ['gray'] + sorted(n for n in names if n and n.lower() not in {'gray', 'omitted'})
```

iii. The image vocabulary is deterministic and global. Gray periods are explicitly represented because the instruction mentions "non-grey screen," implying gray is a valid category.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100ms bin centers as the neural data, using stimulus presentation timestamps on the same session clock.

ii. See 3-a code snippet -- `centers` is the same array used for neural binning.

iii. All streams share the same session clock, so bin centers align neural and stimulus data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `change_time` in the NWB trial intervals and the `go` flag.

ii.
```python
changes = np.asarray(tr['change_time'], float)
...
if flags['go'][ti] and np.isfinite(changes[ti]):
    j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
    if 0 <= j < nbin:
        change[j] = 1
```

iii. Only go trials have an actual image change; catch trials have a sham change where the same image is repeated.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A single 100ms bin is marked as 1 at the change time; all other bins are 0. The bin is identified as `ceil((change_time - start_time) / 0.1)`, i.e., the first bin whose center is at or after the change onset.

ii. See 4-a code snippet.

iii. The AI chose a single-bin impulse rather than a sustained window, interpreting "right after a change" literally as an instantaneous event marker.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. It is already categorical.

ii. `change = np.zeros(nbin, dtype=np.int64)` with selective `change[j] = 1`.

iii. The instructions specify "binary variable" so no further thresholding is required.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 100ms bin grid as neural data. The change bin index is computed from `change_time` relative to trial `start_time`.

ii. See 4-a code snippet.

iii. Alignment is guaranteed by using the same bin grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
run_t = np.asarray(h['processing/running/speed/timestamps'], float)
run_x = np.asarray(h['processing/running/speed/data'], float)
```

iii. This is the standard running speed data path in the NWB files, equivalent to the SDK's `running_speed` attribute.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to 100ms bin centers using `interp_valid()`, which filters for finite values and uses `np.interp` (which extrapolates using nearest endpoint). It is then discretized into 5 quintile bins computed **per session/experiment**.

ii.
```python
r = interp_valid(centers, run_t, run_x)
...
run_lab, run_edges = quintile_labels([q[2] for q in prelim])

def quintile_labels(values):
    allv = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
    edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
    labels = [np.searchsorted(edges, v, side='right').astype(np.int64) for v in values]
    return labels, edges
```

iii. The AI interpolated running speed to 100ms bin centers. Quintile edges are computed from all valid samples within each experiment, producing balanced categories per session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using `np.quantile` at [0.2, 0.4, 0.6, 0.8] percentiles, then `np.searchsorted` to assign bin labels 0-4. Edges are computed **per experiment/session**, not globally.

ii. See 5-b code snippet.

iii. From CONVERSION_NOTES Step 5: "compute quintile edges independently for each experiment/session from valid aligned values."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly to the same 100ms bin centers used for neural data.

ii.
```python
r = interp_valid(centers, run_t, run_x)
```

iii. Using the same `centers` array ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/data` (which contains ellipse axes) and `acquisition/EyeTracking/likely_blink/data`. The equivalent circular diameter is computed as `sqrt(width * height)`.

ii.
```python
axes = np.asarray(h['acquisition/EyeTracking/pupil_tracking/data'], float)
blink = np.asarray(h['acquisition/EyeTracking/likely_blink/data']).astype(bool)
pupil = np.sqrt(np.maximum(axes[:, 0] * axes[:, 1], 0.0))
pupil_good = (~blink) & np.isfinite(pupil) & (pupil > 0)
```

iii. The AI computed equivalent circular diameter from both pupil axes rather than using just pupil width. From CONVERSION_NOTES: "Equivalent circular diameter uses both pupil axes and is robust to ellipse orientation."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames and non-finite/zero values are excluded. Valid pupil diameter values are linearly interpolated to 100ms bin centers using `interp_valid()`. Then discretized into 5 per-session quintile bins.

ii.
```python
p = interp_valid(centers, eye_t, pupil, pupil_good)
...
pup_lab, pup_edges = quintile_labels([q[3] for q in prelim])
```

iii. Same approach as running speed. Blink removal before interpolation prevents artifacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same per-session quintile discretization as running speed, using `np.quantile` at [0.2, 0.4, 0.6, 0.8].

ii. See `quintile_labels` function in 5-b.

iii. Per-session edges ensure balanced categories within each session.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed -- interpolated to the same 100ms bin centers.

ii. `p = interp_valid(centers, eye_t, pupil, pupil_good)`

iii. Using the same `centers` array ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean flags `hit`, `miss`, `false_alarm`, `correct_reject` in the NWB trial intervals.

ii.
```python
flags = {k: np.asarray(tr[k]).astype(bool) for k in
         ['go','catch','aborted','auto_rewarded','hit','miss','false_alarm','correct_reject']}
...
if flags['hit'][ti]: outcome = 0
elif flags['miss'][ti]: outcome = 1
elif flags['false_alarm'][ti]: outcome = 2
elif flags['correct_reject'][ti]: outcome = 3
else: raise ValueError(...)
```

iii. These four outcomes are mutually exclusive for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped to integer codes 0-3 (hit=0, miss=1, false_alarm=2, correct_reject=3) and broadcast across all time bins in the trial.

ii.
```python
outputs.append(np.vstack([image, change, rl, pl,
                          np.full(len(centers), outcome, dtype=np.int64)]).astype(np.int64))
```

iii. Broadcasting is required because the output array must have shape (5, n_timepoints) to match the time-varying outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing eye tracking**: Experiments without eye tracking data are excluded entirely (3 active experiments).
- **Passive sessions**: Excluded because trial outcomes are not genuine.
- **Non-finite running/pupil**: Trials where interpolated running or pupil data has any non-finite values are dropped.
- **Empty bins in neural data**: Interpolated from neighboring frames.
- **Too few trials**: Experiments with <2 usable trials raise an error and are skipped.

ii.
```python
if not (np.isfinite(r).all() and np.isfinite(p).all()):
    continue
...
if len(neural) < 2:
    raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. The AI took a strict approach: rather than filling missing behavioral data with defaults, it drops the affected trials entirely to avoid introducing artifacts.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading each NWB file with h5py and extracting the neural event data, running speed, eye tracking, and stimulus presentation data. The AI noted per-experiment processing takes 0.34-0.61s in sample mode.

ii. N/A

iii. From CONVERSION_NOTES: estimated full conversion at ~2 minutes for 199 sessions; actual full conversion took 149.3s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin stimulus assignment loop in `process_experiment` iterates over each time bin to check for active image flashes. This could potentially be vectorized.

ii.
```python
for j, (a, b) in enumerate(zip(left, right)):
    if b > a:
        ft = flash_t[b-1]
        if ft <= centers[j] < ft + 0.250:
            image[j] = image_to_idx.get(str(flash_name[b-1]), 0)
```

Also the per-bin loop in `event_bin_means`:
```python
for j, (a, b) in enumerate(zip(lo, hi)):
    if b > a:
        out[:, j] = np.asarray(events[a:b], dtype=np.float32).mean(axis=0)
    else:
        ...
```

iii. These loops are not major bottlenecks given the relatively small number of bins per trial.

## 9-c. What processing does the code repeat multiple times?

i. The image vocabulary scan (`image_vocabulary()`) reads all NWB files once to discover image names before the main processing loop, which reads each NWB file again for conversion. This means each NWB file is opened twice.

ii.
```python
def image_vocabulary(meta, files):
    for eid in meta.ophys_experiment_id.astype(int):
        with h5py.File(files[eid], 'r') as h:
            ...
# Then later:
for k,row in meta.iterrows():
    ...process_experiment(row,files[eid],imap,...)
```

iii. The first pass is lightweight (only reading template names), so the overhead is minimal.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The per-session quintile edge computation and storage in `info` dicts is not directly used by the downstream decoder (which only sees the integer bin labels). The `session_info` list in metadata stores detailed per-experiment information that the decoder doesn't use.

ii.
```python
info = {'ophys_experiment_id': eid, ..., 'running_quintile_edges': run_edges.tolist(),
        'pupil_quintile_edges': pup_edges.tolist(), 'source_trial_indices': kept_ids, ...}
```

iii. This metadata is useful for debugging and interpretation but not consumed by `train_decoder.py`.
