# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache path used by the reference. It reads the local metadata CSV, filters it to locally available NWB files, and loads each selected experiment directly from disk with `BehaviorOphysExperiment.from_nwb_path()`. It also narrows the dataset to `project_code == 'VisualBehavior'`, `passive == False`, and `experience_level == 'Familiar'`.

ii.
```python
available = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                   for f in os.listdir(NWB_DIR) if f.endswith('.nwb'))
tbl = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
sel = tbl[tbl.ophys_experiment_id.isin(available)
          & (tbl.project_code == 'VisualBehavior')
          & (~tbl.passive)
          & (tbl.experience_level == 'Familiar')]
...
path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{oeid}.nwb')
expt = BehaviorOphysExperiment.from_nwb_path(path)
```

iii. In the trajectory, the agent justified this as a deliberate departure from the reference pipeline: it wanted single-plane 31 Hz sessions, active behavior only, familiar-image sessions only, and local NWB loading instead of S3 cache loading. It said this gave one simultaneously recorded population per session and a common frame rate across sessions, and that one session with no eye tracking was dropped because pupil was required.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id`. During assembly, the AI builds `subject_list` from the unique `mouse_id` values of the selected experiment rows and records one `subject_idx` per session.

ii.
```python
mouse = str(row['mouse_id'])
if mouse not in subject_list:
    subject_list.append(mouse)
subject_idx.append(subject_list.index(mouse))
```

iii. The trajectory repeatedly refers to mice by `mouse_id` and summarizes the final dataset in counts of mice, sessions, trials, and neurons, which is consistent with using `mouse_id` as the subject identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each selected `ophys_experiment_id` NWB file as one session. It does not regroup multiple experiments that share the same `ophys_session_id`; this follows from its choice to restrict to the single-plane `VisualBehavior` subset.

ii.
```python
def extract_session(row):
    oeid = int(row['ophys_experiment_id'])
    path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{oeid}.nwb')
    expt = BehaviorOphysExperiment.from_nwb_path(path)
...
for res, (_, row) in zip(results, rows):
    ...
    sessions.append(res)
    meta_rows.append(row)
```

iii. In the trajectory, the agent explicitly said it chose single-plane `VisualBehavior` because “a session = one simultaneously recorded population” and excluded `VisualBehaviorMultiscope`.

## 1-d. How are the data split into trials?

i. Trials are taken from the experiment’s SDK `trials` table. Each kept trial spans `[trials.start_time, trials.stop_time)` on the ophys timestamp axis, so trial lengths are variable.

ii.
```python
trials = expt.trials
keep = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
trials = trials[keep]

for tid, tr in trials.iterrows():
    i0 = np.searchsorted(ts, tr['start_time'], side='left')
    i1 = np.searchsorted(ts, tr['stop_time'], side='left')
```

iii. The trajectory says the AI used “the experiment’s own trial definition (SDK `trials` table)” and that each trial spans `start_time` to `stop_time`.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes aborted and auto-rewarded trials, skips trials with fewer than 2 ophys frames, skips trials whose outcome booleans are not exactly one-hot across `hit/miss/false_alarm/correct_reject`, and later drops sessions with fewer than 2 usable trials.

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
    skipped.append((int(row['ophys_experiment_id']),
                    res.get('skip', 'fewer than 2 usable trials')))
    continue
```

iii. In the trajectory, the agent said it was following the instruction to keep go and catch trials while excluding aborted and auto-rewarded trials, and it added the one-hot outcome check so only scoreable go/catch trials remained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the AllenSDK `events` table, specifically the `filtered_events` column, not from `dff_traces.dff`.

ii.
```python
events = expt.events
cell_ids = list(events.index)
traces = np.stack([np.asarray(x, dtype=np.float32)
                   for x in events['filtered_events'].values])
```

iii. In the trajectory, the agent explicitly justified this as following the paper’s wording about “detected calcium events.” It also said raw sparse events were too empty at single-frame resolution, so it preferred `filtered_events`.

## 2-b. How is the `neural` data processed?

i. The AI stacks the per-cell `filtered_events` traces into a `(n_neurons, n_frames)` matrix and slices that matrix into trials. It does not apply additional normalization or denoising beyond the AllenSDK’s event extraction.

ii.
```python
traces = np.stack([np.asarray(x, dtype=np.float32)
                   for x in events['filtered_events'].values])
...
out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The trajectory says the AI chose `filtered_events` because the half-Gaussian-convolved traces preserve event timing and magnitude in a way the decoder can use at each time bin, while remaining closer to the paper than dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no extra neuron-level filtering. It includes all cells present in the NWB `events` table and relies on Allen’s upstream ROI filtering. It also keeps trials with all-zero neural activity rather than filtering them out.

ii.
```python
events = expt.events
cell_ids = list(events.index)
traces = np.stack([np.asarray(x, dtype=np.float32)
                   for x in events['filtered_events'].values])
```

iii. The trajectory says “valid_roi is True for every cell,” so the AI judged that additional neuron curation was unnecessary. It also noted that some trials had all-zero activity but kept them as genuine.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps and trial starts. For each trial, the AI finds frame indices at `start_time` and `stop_time` on `ophys_timestamps` and slices that interval.

ii.
```python
ts = np.asarray(expt.ophys_timestamps, dtype=float)
...
i0 = np.searchsorted(ts, tr['start_time'], side='left')
i1 = np.searchsorted(ts, tr['stop_time'], side='left')
...
out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The trajectory explicitly states that `ophys_timestamps` are the master time base and that each trial is `[trials.start_time, trials.stop_time)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps one bin per ophys frame, about 32.3 ms per bin for the selected single-plane data. It does not temporally rebin neural data. In metadata it reports the mean session frame interval in milliseconds.

ii.
```python
ts = np.asarray(expt.ophys_timestamps, dtype=float)
...
out['dt'] = float(np.median(np.diff(ts)))
...
dt_ms = float(np.mean([s['dt'] for s in sessions]) * 1000.0)
...
'time_bin_size': dt_ms,
```

iii. The trajectory says the AI chose single-plane sessions partly so “all are 31 Hz,” and that one 2-photon frame is the common time bin across trials and sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the full-session `stimulus_presentations` table, specifically `image_name`, `start_time`, `end_time`, `active`, and `omitted`, rather than from `initial_image_name` and `change_image_name` in the trial table.

ii.
```python
stim = expt.stimulus_presentations
stim = stim[stim['active'].astype(bool)]
shown = stim[~stim['omitted'].astype(bool)]
image_names = sorted(set(shown['image_name'].dropna()))
starts = np.searchsorted(ts, shown['start_time'].to_numpy(dtype=float), side='left')
stops = np.searchsorted(ts, shown['end_time'].to_numpy(dtype=float), side='left')
codes = shown['image_name'].map(img_lookup).to_numpy(dtype=np.int8)
```

iii. In the trajectory, the agent said it wanted “the identity of the image on the screen,” so it used stimulus presentations directly and reserved a separate gray-screen category outside image flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI creates a full-session per-frame image label array. It assigns code `0` to gray-screen / omitted periods and uses positive integer codes for non-omitted flashed images. It later remaps per-session codes onto a global image ordering shared across sessions.

ii.
```python
image_id = np.zeros(nframes, dtype=np.int8)
for i0, i1, c in zip(starts, stops, codes):
    image_id[i0:i1] = c
...
image_names = sorted({n for s in sessions for n in s['image_names']})
for s in sessions:
    remap = np.zeros(len(s['image_names']) + 1, dtype=np.int8)
    for i, name in enumerate(s['image_names']):
        remap[i + 1] = image_names.index(name) + 1
    if not np.array_equal(remap, np.arange(len(remap), dtype=np.int8)):
        s['image_id'] = [remap[a] for a in s['image_id']]
```

iii. The trajectory justifies this by saying the output should reflect the image currently on screen, including gray intervals between flashes. It also says restricting to familiar sessions keeps the same 8 images across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The AI first builds `image_id` on the full ophys frame axis, then slices the same `[i0:i1]` frame range used for neural data when constructing each trial.

ii.
```python
image_id = np.zeros(nframes, dtype=np.int8)
for i0, i1, c in zip(starts, stops, codes):
    image_id[i0:i1] = c
...
out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
out['image_id'].append(image_id[i0:i1])
```

iii. The trajectory repeatedly states that all streams are aligned onto `ophys_timestamps`, so image labels and neural activity share the same frame index within each trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentation stream, using `is_change` together with each flashed image’s `start_time` and `end_time` after filtering to active, non-omitted flashes.

ii.
```python
shown = stim[~stim['omitted'].astype(bool)]
starts = np.searchsorted(ts, shown['start_time'].to_numpy(dtype=float), side='left')
stops = np.searchsorted(ts, shown['end_time'].to_numpy(dtype=float), side='left')
...
is_chg = shown['is_change'].astype(bool).to_numpy()
for i0, i1 in zip(starts[is_chg], stops[is_chg]):
    change[i0:i1] = 1
```

iii. In the trajectory, the agent says sham changes on catch trials are not true image-identity changes, so they remain 0, and that deriving changes from the flashed stimulus itself is more literal than using trial metadata alone.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI constructs a full-session binary vector and sets it to 1 only during the flashed presentation whose identity differs from the previous one. It does not mark the following gray interval.

ii.
```python
change = np.zeros(nframes, dtype=np.int8)
is_chg = shown['is_change'].astype(bool).to_numpy()
for i0, i1 in zip(starts[is_chg], stops[is_chg]):
    change[i0:i1] = 1
```

iii. The trajectory shows the AI initially considered a 750 ms window but changed to the ~250 ms changed-flash window after a decoder comparison. It said the shorter window both decoded better and matched “right after a change in image identity” more literally.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical variable with values `0 = no_change` and `1 = change`; there is no additional thresholding beyond assigning those two labels.

ii.
```python
change = np.zeros(nframes, dtype=np.int8)
...
change[i0:i1] = 1
...
'output_values': [
    ['gray_screen'] + list(image_names),
    ['no_change', 'change'],
```

iii. The trajectory treats image change as a binary event marker and never discusses any further discretization.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The AI builds the binary `change` vector on the full ophys frame axis, then slices the same trial frame range used for neural data.

ii.
```python
for i0, i1 in zip(starts[is_chg], stops[is_chg]):
    change[i0:i1] = 1
...
out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
out['change'].append(change[i0:i1])
```

iii. The trajectory says stimulus, running, pupil, and neural data are all aligned onto `ophys_timestamps`, which is why the same frame indices can be reused here.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the AllenSDK `running_speed` table, using its `timestamps` and `speed` columns.

ii.
```python
run = expt.running_speed
running = np.interp(ts, run['timestamps'].to_numpy(dtype=float),
                    run['speed'].to_numpy(dtype=float)).astype(np.float32)
```

iii. The trajectory refers to running speed as a behavior stream recorded on its own clock and aligned onto the ophys time base.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto ophys frame times with `np.interp`, pools all trial time points across sessions, computes 5 global quantile edges, and finally discretizes each trial with `np.digitize`.

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

iii. The trajectory says the AI wanted globally pooled equal-count bins so the same discrete category meaning would hold across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 equal-count bins using global quantile edges computed from all retained trial time points.

ii.
```python
NQUANTILES = 5
...
def quantile_bins(values, nbins):
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)
...
out[2] = np.digitize(s['running'][k], run_edges)
```

iii. The trajectory and metadata both describe this as 5 globally pooled quintile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto `ophys_timestamps` before trial segmentation, then trial slices are taken with the same frame boundaries used for neural data.

ii.
```python
running = np.interp(ts, run['timestamps'].to_numpy(dtype=float),
                    run['speed'].to_numpy(dtype=float)).astype(np.float32)
...
out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
out['running'].append(running[i0:i1])
```

iii. The trajectory explicitly describes `ophys_timestamps` as the common time base for all streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking['pupil_area']` plus `eye_tracking['timestamps']` and `eye_tracking['likely_blink']`; it is converted to an equivalent diameter rather than using `pupil_width`.

ii.
```python
area = eye_tracking['pupil_area'].to_numpy(dtype=float)
area[eye_tracking['likely_blink'].to_numpy(dtype=bool)] = np.nan
...
t = eye_tracking['timestamps'].to_numpy(dtype=float)
diam = 2.0 * np.sqrt(area[good] / np.pi)
```

iii. The trajectory says the AI intentionally used fitted pupil area and converted it to diameter in pixels, removing blink frames first.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI removes blink frames by setting them to NaN, requires at least 50% finite pupil samples in the session, converts area to equivalent diameter, linearly interpolates that diameter to the ophys time base with `np.interp`, then discretizes globally into 5 quantile bins.

ii.
```python
area = eye_tracking['pupil_area'].to_numpy(dtype=float)
area[eye_tracking['likely_blink'].to_numpy(dtype=bool)] = np.nan
good = np.isfinite(area)
if good.sum() < 0.5 * len(area):
    return None, None
...
diam = 2.0 * np.sqrt(area[good] / np.pi)
...
pupil = np.interp(ts, et, ed).astype(np.float32)
...
all_pupil = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
pupil_edges = quantile_bins(all_pupil, NQUANTILES)
...
out[3] = np.digitize(s['pupil'][k], pupil_edges)
```

iii. The trajectory says the AI dropped one entire session for missing eye tracking because pupil diameter was a required output and said blink-contaminated samples should be removed before interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 global equal-count bins using quantile edges pooled across all retained sessions and trials.

ii.
```python
all_pupil = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
pupil_edges = quantile_bins(all_pupil, NQUANTILES)
...
out[3] = np.digitize(s['pupil'][k], pupil_edges)
```

iii. The trajectory and metadata describe the pupil labels as 5 equal-count bins shared across the whole dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is converted to a session-long trace on the ophys frame axis before trialing, then trial slices are taken with the same `[i0:i1]` indices as neural data.

ii.
```python
pupil = np.interp(ts, et, ed).astype(np.float32)
...
out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
out['pupil'].append(pupil[i0:i1])
```

iii. The trajectory says the eye-tracking stream is resampled onto `ophys_timestamps`, which is why the same trial indices align pupil and neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the SDK `trials` table.

ii.
```python
outcome = [tr['hit'], tr['miss'], tr['false_alarm'], tr['correct_reject']]
if sum(bool(x) for x in outcome) != 1:
    continue
```

iii. The trajectory says passive sessions were excluded because these trial outcome categories are only meaningful for active behavior, and that only scoreable go/catch trials were kept.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI orders the outcome categories as `['hit', 'miss', 'false_alarm', 'correct_reject']`, converts each valid trial to an integer by `argmax`, and broadcasts that scalar across all time bins of the trial when building the final `(5, T)` output matrix.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
out['outcome'].append(int(np.argmax(outcome)))
...
out = np.empty((5, T), dtype=np.int8)
...
out[4] = s['outcome'][k]
```

iii. The trajectory repeatedly refers to trial outcome as one of hit/miss/false alarm/correct reject and describes it as tiled across the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or imperfect data by restricting to locally present NWB files, dropping sessions with no usable eye tracking, dropping sessions with fewer than 2 usable trials, skipping trials shorter than 2 frames, skipping trials with ambiguous outcomes, using interpolation to fill running and pupil values onto the ophys clock, and using on-disk caching so expensive NWB reads only happen once.

ii.
```python
if eye_tracking is None or len(eye_tracking) == 0:
    return None, None
...
if good.sum() < 0.5 * len(area):
    return None, None
...
if et is None:
    return {'oeid': oeid, 'skip': 'no eye tracking'}
...
if i1 - i0 < 2:
    continue
...
if sum(bool(x) for x in outcome) != 1:
    continue
...
if os.path.exists(cache):
    try:
        with open(cache, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass
```

iii. The trajectory specifically mentions dropping one experiment for missing eye tracking, keeping all-zero neural trials as genuine rather than errors, and adding disk caching because NWB reads were the slow part.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading and parsing each NWB file, especially extracting large neural and behavior arrays. The AI explicitly added multiprocessing and an on-disk cache to reduce that overhead.

ii.
```python
from multiprocessing import Pool
...
def extract_cached(args):
    """extract_session + an on-disk cache, so the (slow) NWB reads happen only once."""
...
with Pool(min(workers, len(rows))) as pool:
    results = pool.map(extract_cached, rows, chunksize=1)
```

iii. The trajectory says the “slow NWB reads happen only once” because of the cache and reports end-to-end conversion time after parallel loading.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops remain vectorizable: the loop over flashed stimuli when filling `image_id`, the loop over changed flashes when filling `change`, the loop over trials in each session, the loop used to remap image IDs to the global ordering, and the per-trial output assembly loop.

ii.
```python
for i0, i1, c in zip(starts, stops, codes):
    image_id[i0:i1] = c
...
for i0, i1 in zip(starts[is_chg], stops[is_chg]):
    change[i0:i1] = 1
...
for tid, tr in trials.iterrows():
    ...
for i, name in enumerate(s['image_names']):
    remap[i + 1] = image_names.index(name) + 1
...
for k in range(len(s['neural'])):
    ...
```

iii. The trajectory focuses on I/O as the main bottleneck, so the agent did not prioritize vectorizing these loops.

## 9-c. What processing does the code repeat multiple times?

i. The AI repeats some processing that could have been centralized: it repeatedly does `image_names.index(name)` inside a remapping loop, repeatedly calls `np.digitize` trial by trial instead of sessionwise, and repeatedly checks/serializes per-session cache files even though the retained sessions are later processed again during assembly.

ii.
```python
for s in sessions:
    remap = np.zeros(len(s['image_names']) + 1, dtype=np.int8)
    for i, name in enumerate(s['image_names']):
        remap[i + 1] = image_names.index(name) + 1
...
for k in range(len(s['neural'])):
    ...
    out[2] = np.digitize(s['running'][k], run_edges)
    out[3] = np.digitize(s['pupil'][k], pupil_edges)
```

iii. The trajectory does not flag these as problems; its optimization focus was instead on caching and multiprocessing the file loads.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores some information that the downstream decoder does not use: large `session_info` metadata entries, `cell_specimen_ids`, `trial_ids`, and `excluded_experiments`. It also computes `n_frames_session` inside each extracted session but never uses that field later.

ii.
```python
out = {'oeid': oeid, 'cell_ids': cell_ids, 'image_names': image_names,
       'neural': [], 'running': [], 'pupil': [], 'image_id': [], 'change': [],
       'outcome': [], 'trial_ids': [], 'n_frames_session': nframes}
...
session_info.append({
    'ophys_experiment_id': int(row['ophys_experiment_id']),
    ...
    'cell_specimen_ids': [int(c) for c in s['cell_ids']],
    'trial_ids': s['trial_ids'],
    'ophys_frame_interval_s': s['dt'],
})
...
'excluded_experiments': skipped,
'session_info': session_info,
```

iii. The trajectory emphasizes rich metadata and traceability, but none of this is required by the decoder itself.
