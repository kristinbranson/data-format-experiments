# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading NWB files directly from disk using `BehaviorOphysExperiment.from_nwb_path()`. It first builds an experiment table from the CSV metadata (`ophys_experiment_table.csv`), filters to available NWB files, then filters by `project_code == 'VisualBehavior'`, `passive == False`, and `experience_level == 'Familiar'`. Each experiment is loaded in parallel using `multiprocessing.Pool`.

ii.
```python
def select_experiments():
    available = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                       for f in os.listdir(NWB_DIR) if f.endswith('.nwb'))
    tbl = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    sel = tbl[tbl.ophys_experiment_id.isin(available)
              & (tbl.project_code == 'VisualBehavior')
              & (~tbl.passive)
              & (tbl.experience_level == 'Familiar')]
    return sel.sort_values('ophys_experiment_id').reset_index(drop=True)

# Loading:
expt = BehaviorOphysExperiment.from_nwb_path(path)
```

iii. The AI chose to load NWB files directly rather than use the S3 cache, since the data was already available locally. Filtering by `VisualBehavior` selects single-plane sessions. Filtering by `Familiar` and active (non-passive) follows the reference paper's analysis criteria.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment table. Each unique mouse_id becomes a subject entry.

ii.
```python
mouse = str(row['mouse_id'])
if mouse not in subject_list:
    subject_list.append(mouse)
subject_idx.append(subject_list.index(mouse))
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as one session. For the `VisualBehavior` project, each session has exactly one imaging plane, so this is equivalent to one session per `ophys_session_id`.

ii.
```python
for s, row in zip(sessions, meta_rows):
    # Each s corresponds to one experiment (one ophys_experiment_id)
    ...
```

iii. Since VisualBehavior sessions have exactly one plane each, treating each experiment as a session is functionally equivalent to grouping by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `trials` table. Each trial spans from `start_time` to `stop_time` (variable length). Only `go` and `catch` trials are kept; `aborted` and `auto_rewarded` trials are excluded.

ii.
```python
trials = expt.trials
keep = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
trials = trials[keep]

for tid, tr in trials.iterrows():
    i0 = np.searchsorted(ts, tr['start_time'], side='left')
    i1 = np.searchsorted(ts, tr['stop_time'], side='left')
    if i1 - i0 < 2:
        continue
```

iii. The trial definition follows the instructions to include Go and Catch trials and exclude Aborted and Auto-rewarded trials. The trial window uses the SDK's built-in `start_time` and `stop_time`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) requiring `go` or `catch` flag, (2) excluding `aborted` and `auto_rewarded`, (3) requiring at least 2 ophys frames (`i1 - i0 < 2`), and (4) requiring exactly one outcome flag to be True. Sessions with fewer than 2 valid trials are skipped. Sessions without usable eye-tracking data are also skipped entirely (since pupil diameter is a required output).

ii.
```python
keep = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
...
if i1 - i0 < 2:
    continue
outcome = [tr['hit'], tr['miss'], tr['false_alarm'], tr['correct_reject']]
if sum(bool(x) for x in outcome) != 1:
    continue
...
if res.get('skip') or len(res.get('neural', [])) < 2:
    skipped.append(...)
```

iii. The filtering ensures only well-defined, scoreable trials are included. The eye-tracking requirement ensures pupil data is available for all sessions. The 2-trial minimum prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `filtered_events` in the SDK's `events` table — detected calcium events convolved with a half-Gaussian kernel. This is NOT the dF/F traces.

ii.
```python
events = expt.events
cell_ids = list(events.index)
traces = np.stack([np.asarray(x, dtype=np.float32)
                   for x in events['filtered_events'].values])
```

iii. The AI followed the reference paper: "For all analysis of neural data we used the detected calcium events... thus removing the slow decay dynamics of the calcium indicator." Raw events were rejected because they are nonzero on only ~0.1% of frames. `filtered_events` preserves event information in a decoder-usable form.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The `filtered_events` traces are used as-is from the SDK, stacked into a `(n_neurons, n_frames)` matrix.

ii.
```python
traces = np.stack([np.asarray(x, dtype=np.float32)
                   for x in events['filtered_events'].values])
```

iii. The events have already been processed by the Allen SDK pipeline (event detection from dF/F traces, half-Gaussian convolution).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. All ROIs in the released NWB files already passed the pipeline's QC (`valid_roi == True` for every cell).

ii. N/A — no filtering code.

iii. The AI verified that all released ROIs have `valid_roi == True`, so no further filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial start (`start_time`). Ophys frames from `start_time` to `stop_time` are extracted using `np.searchsorted`, giving variable-length trials.

ii.
```python
i0 = np.searchsorted(ts, tr['start_time'], side='left')
i1 = np.searchsorted(ts, tr['stop_time'], side='left')
...
out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The ophys timestamps are the time base. Using `searchsorted` with `side='left'` finds the first frame at or after the boundary time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Data is kept at the native ophys frame rate (~31 Hz for VisualBehavior). The time bin size is computed as the mean of median inter-frame intervals across sessions (~32.3 ms).

ii.
```python
out['dt'] = float(np.median(np.diff(ts)))
...
dt_ms = float(np.mean([s['dt'] for s in sessions]) * 1000.0)
```

iii. VisualBehavior single-plane sessions all run at ~31 Hz, so no resampling is needed.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, using the `image_name`, `start_time`, and `end_time` columns. The table is filtered to active (non-passive) and non-omitted presentations.

ii.
```python
stim = expt.stimulus_presentations
stim = stim[stim['active'].astype(bool)]
shown = stim[~stim['omitted'].astype(bool)]
image_names = sorted(set(shown['image_name'].dropna()))
```

iii. Using `stimulus_presentations` provides frame-accurate stimulus timing (start and end of each flash), rather than relying on the coarser trial-level `initial_image_name`/`change_image_name` fields.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each unique image name is mapped to an integer code (1 through N). Code 0 is reserved for "gray screen" (inter-stimulus intervals and omitted flashes). For each stimulus presentation, the corresponding ophys frames are assigned that image's code. A global remapping is applied to ensure consistent codes across sessions.

ii.
```python
img_lookup = {name: i + 1 for i, name in enumerate(image_names)}
image_id = np.zeros(nframes, dtype=np.int8)
starts = np.searchsorted(ts, shown['start_time'].to_numpy(dtype=float), side='left')
stops = np.searchsorted(ts, shown['end_time'].to_numpy(dtype=float), side='left')
codes = shown['image_name'].map(img_lookup).to_numpy(dtype=np.int8)
for i0, i1, c in zip(starts, stops, codes):
    image_id[i0:i1] = c

# Global remapping:
for s in sessions:
    remap = np.zeros(len(s['image_names']) + 1, dtype=np.int8)
    for i, name in enumerate(s['image_names']):
        remap[i + 1] = image_names.index(name) + 1
```

iii. The 0-indexed gray screen category captures inter-stimulus intervals and omitted flashes as a distinct category.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame using `np.searchsorted` on stimulus `start_time` and `end_time`, then indexed using the same frame range as neural data.

ii.
```python
starts = np.searchsorted(ts, shown['start_time'].to_numpy(dtype=float), side='left')
stops = np.searchsorted(ts, shown['end_time'].to_numpy(dtype=float), side='left')
for i0, i1, c in zip(starts, stops, codes):
    image_id[i0:i1] = c
...
out['image_id'].append(image_id[i0:i1])  # same i0:i1 as neural
```

iii. Both neural and image identity data are indexed by ophys frame indices, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in `stimulus_presentations`, combined with the `start_time` and `end_time` of each flash.

ii.
```python
change = np.zeros(nframes, dtype=np.int8)
is_chg = shown['is_change'].astype(bool).to_numpy()
for i0, i1 in zip(starts[is_chg], stops[is_chg]):
    change[i0:i1] = 1
```

iii. `is_change` from stimulus_presentations identifies the actual change flash. Sham changes on catch trials are not flagged because the image identity doesn't actually change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary indicator is set to 1 on the ophys frames during which the changed image flash is displayed (~250 ms), and 0 otherwise. Only true changes (where image identity differs) are marked.

ii.
```python
change = np.zeros(nframes, dtype=np.int8)
is_chg = shown['is_change'].astype(bool).to_numpy()
for i0, i1 in zip(starts[is_chg], stops[is_chg]):
    change[i0:i1] = 1
```

iii. The AI tested both a 750 ms window (flash + grey ISI) and a ~250 ms window (flash only) and found the 250 ms window decoded better and was more literally consistent with "right after a change in image identity."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
change = np.zeros(nframes, dtype=np.int8)
# ...
change[i0:i1] = 1
```

iii. The binary encoding directly represents the two categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity — computed per ophys frame and indexed with the same frame range.

ii.
```python
out['change'].append(change[i0:i1])
```

iii. Uses the same ophys frame indices as the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `expt.running_speed`, which provides speed and timestamps from the running wheel encoder.

ii.
```python
run = expt.running_speed
running = np.interp(ts, run['timestamps'].to_numpy(dtype=float),
                    run['speed'].to_numpy(dtype=float)).astype(np.float32)
```

iii. `running_speed` is the SDK's standard locomotion data interface.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the ophys timebase using `np.interp`, then discretized into 5 equal-count (percentile) bins computed globally across all sessions.

ii.
```python
running = np.interp(ts, run['timestamps'].to_numpy(dtype=float),
                    run['speed'].to_numpy(dtype=float)).astype(np.float32)
...
all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
run_edges = quantile_bins(all_run, NQUANTILES)
...
out[2] = np.digitize(s['running'][k], run_edges)
```

iii. `np.interp` provides linear interpolation. Global percentile bins ensure consistent categories across sessions with roughly equal class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using `np.digitize` with interior quantile edges. The bin indices range from 0 to 4.

ii.
```python
def quantile_bins(values, nbins):
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)
...
out[2] = np.digitize(s['running'][k], run_edges)
```

iii. `quantile_bins` computes only the interior edges (4 edges for 5 bins). `np.digitize` with these edges returns values 0 through 4.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto ophys timestamps before trial segmentation, then indexed with the same frame range as neural data.

ii.
```python
running = np.interp(ts, ...).astype(np.float32)
...
out['running'].append(running[i0:i1])
```

iii. Using the same ophys-frame indices guarantees alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the SDK's `eye_tracking` table. It is converted to diameter via `2 * sqrt(area / pi)`.

ii.
```python
area = eye_tracking['pupil_area'].to_numpy(dtype=float)
area[eye_tracking['likely_blink'].to_numpy(dtype=bool)] = np.nan
...
diam = 2.0 * np.sqrt(area[good] / np.pi)
```

iii. The AI chose `pupil_area` (which the SDK computes from the fitted ellipse) and converted it to diameter, rather than using `pupil_width` directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (flagged by `likely_blink`) are set to NaN. Area is converted to diameter. The cleaned signal is linearly interpolated to fill NaN gaps and resampled onto ophys timestamps using `np.interp`. Then discretized into 5 global percentile bins. Sessions with <50% usable data are dropped.

ii.
```python
area[eye_tracking['likely_blink'].to_numpy(dtype=bool)] = np.nan
good = np.isfinite(area)
if good.sum() < 0.5 * len(area):
    return None, None
t = eye_tracking['timestamps'].to_numpy(dtype=float)
diam = 2.0 * np.sqrt(area[good] / np.pi)
...
pupil = np.interp(ts, et, ed).astype(np.float32)
...
out[3] = np.digitize(s['pupil'][k], pupil_edges)
```

iii. Blink removal prevents corrupting the signal. The interpolation approach fills gaps caused by blinks rather than leaving NaN values. The 50% threshold ensures sessions with too much missing data are excluded.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — discretized into 5 bins using `np.digitize` with global percentile-based interior edges.

ii.
```python
pupil_edges = quantile_bins(all_pupil, NQUANTILES)
...
out[3] = np.digitize(s['pupil'][k], pupil_edges)
```

iii. Same percentile-based approach as running speed for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto ophys timestamps, then indexed with the same frame range as neural data.

ii.
```python
pupil = np.interp(ts, et, ed).astype(np.float32)
...
out['pupil'].append(pupil[i0:i1])
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
outcome = [tr['hit'], tr['miss'], tr['false_alarm'], tr['correct_reject']]
if sum(bool(x) for x in outcome) != 1:
    continue
out['outcome'].append(int(np.argmax(outcome)))
```

iii. These four columns are the SDK's canonical trial outcome labels. The code requires exactly one to be True, skipping ambiguous trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is encoded as an integer (0-3) via `np.argmax` of the four boolean flags. The outcome is static per trial but broadcast to all time bins in the output array.

ii.
```python
out['outcome'].append(int(np.argmax(outcome)))
...
out[4] = s['outcome'][k]  # broadcast scalar to all frames
```

iii. The argmax gives: 0=hit, 1=miss, 2=false_alarm, 3=correct_reject.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without eye-tracking data (or with <50% usable data) are skipped entirely.
- **Blinks in pupil data**: Set to NaN and linearly interpolated over.
- **Short trials**: Trials with fewer than 2 ophys frames are skipped.
- **Ambiguous outcomes**: Trials without exactly one outcome flag True are skipped.
- **Few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Failed extraction**: Exceptions during session extraction cause the session to be skipped.
- **Running speed NaN**: `np.interp` handles edge extrapolation by holding constant at edges (no NaN produced).

ii.
```python
if good.sum() < 0.5 * len(area):
    return None, None
...
if i1 - i0 < 2:
    continue
if sum(bool(x) for x in outcome) != 1:
    continue
...
if res.get('skip') or len(res.get('neural', [])) < 2:
    skipped.append(...)
```

iii. The approach is conservative — problematic data is excluded rather than imputed, to avoid introducing artifacts.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`. The AI mitigated this by using multiprocessing (`Pool` with up to 22 workers) and on-disk caching of extracted results.

ii.
```python
if workers > 1:
    with Pool(min(workers, len(rows))) as pool:
        results = pool.map(extract_cached, rows, chunksize=1)
```

iii. NWB file parsing is I/O-bound and involves reading large neural trace arrays.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulus-presentation loop for assigning image identity codes could potentially be vectorized, though it operates on relatively few entries (~2000 per session). The per-trial loop is inherently sequential due to variable-length trial outputs.

ii.
```python
for i0, i1, c in zip(starts, stops, codes):
    image_id[i0:i1] = c
```

iii. These loops are not bottlenecks compared to data loading.

## 9-c. What processing does the code repeat multiple times?

i. The AI's code avoids repeating processing by caching extracted session results to disk. On re-runs, cached results are loaded instead of re-extracting from NWB files.

ii.
```python
def extract_cached(args):
    cache = os.path.join(CACHE_DIR, f"{int(row['ophys_experiment_id'])}.pkl")
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    res = extract_session(row)
    ...
```

iii. The caching mechanism prevents redundant I/O on repeated runs.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores extensive metadata per session (cell_specimen_ids, trial_ids, equipment_name, genotype, etc.) that is not used by the decoder. The image identity global remapping is computed even when session-local ordering already matches.

ii.
```python
session_info.append({
    'ophys_experiment_id': ..., 'behavior_session_id': ...,
    'ophys_container_id': ..., 'cre_line': ..., 'full_genotype': ...,
    'equipment_name': ..., 'cell_specimen_ids': ..., 'trial_ids': ...,
    ...
})
```

iii. This metadata is useful for provenance and debugging but is not consumed by the decoder.
