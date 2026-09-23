# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads metadata from the local CSV `project_metadata/ophys_experiment_table.csv`, intersects it with the NWB files physically present in `behavior_ophys_experiments/`, filters to `project_code == 'VisualBehavior'` and non-passive sessions, and then opens each session directly from disk with `BehaviorOphysExperiment.from_nwb_path(...)`. It does not use the Allen SDK project cache or reconstruct multi-experiment sessions.

ii.
```python
et = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
available = set()
for fn in os.listdir(EXPERIMENT_DIR):
    if fn.endswith('.nwb'):
        available.add(int(fn.split('_')[-1].split('.')[0]))
et = et[et.ophys_experiment_id.isin(available)]
et = et[(et.project_code == PROJECT_CODE) & (~et.passive.astype(bool))]
...
path = os.path.join(EXPERIMENT_DIR,
                    f'behavior_ophys_experiment_{oeid}.nwb')
ds = BehaviorOphysExperiment.from_nwb_path(path)
```

iii. In step 54 and again in the final summary at step 93, the AI justified this as using the local released single-plane `VisualBehavior` NWB files only, excluding multiscope data because its slower frame rate would violate the uniform-bin requirement and excluding passive sessions because the requested behavioral outputs would be ill-defined there.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the filtered session table, converted to sorted strings.

ii.
```python
subjects = sorted(table.mouse_id.astype(str).unique().tolist())
subject_idx = np.array([subjects.index(str(m)) for m in table.mouse_id],
                       dtype=np.int64)
```

iii. The trajectory does not contain a separate argument for this beyond using Allen metadata. The implicit justification is that `mouse_id` is the per-animal identifier in the session table.

## 1-c. How are the data split into sessions?

i. The AI treats each remaining `ophys_experiment_id` row as one session after asserting that, in the filtered dataset, there is exactly one experiment per `ophys_session_id`.

ii.
```python
et = et[(et.project_code == PROJECT_CODE) & (~et.passive.astype(bool))]
# single plane imaging: one imaging plane (experiment) per session
assert et.ophys_session_id.nunique() == len(et)
et = et.sort_values('ophys_experiment_id').reset_index(drop=True)
...
oeids = list(table.ophys_experiment_id.values)
```

iii. In step 54 the AI explicitly justified this as a single-plane dataset choice: one imaging plane per session for `VisualBehavior`, while multiscope experiments were excluded.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`, but the AI does not use the SDK trial boundaries directly as the saved trial window. Instead, it keeps `go` or `catch` trials and extracts a fixed-length window of ophys frames from `change_time - 2 s` to `change_time + 2 s`.

ii.
```python
trials = ds.trials
keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
        & ~trials.aborted.astype(bool)
        & ~trials.auto_rewarded.astype(bool)
        & trials.change_time.notna())
trials = trials[keep]
...
change_time = float(tr.change_time)
i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
...
tt = ts[i0:i0 + T]
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. Steps 33, 47, 54, 79, and 93 show the AI’s rationale: `change_time` was verified to coincide with the onset of the real or sham change flash, every go/catch trial had at least about 3 s before and 4.23 s after the change, and therefore a fixed `[-2, +2] s` window around `change_time` would fit all usable trials and yield a common trial length.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only `go` or `catch` trials, excludes `aborted`, `auto_rewarded`, and missing-`change_time` trials, skips trials whose fixed window would run off the recording or precede the first valid flash, and drops sessions with fewer than two usable trials.

ii.
```python
keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
        & ~trials.aborted.astype(bool)
        & ~trials.auto_rewarded.astype(bool)
        & trials.change_time.notna())
...
if i0 + T > len(ts):
    n_dropped_edge += 1
    continue
...
if np.any(k < 0):
    n_dropped_edge += 1
    continue
...
if len(neural) < 2:
    return {'oeid': oeid, 'error': f'only {len(neural)} usable trials'}
```

iii. In steps 54 and 93 the AI says this follows the task instructions to keep go/catch and drop aborted/auto-rewarded trials. Step 79 reports that all 42,470 retained trials fit inside the chosen fixed window, so the edge filter did not remove additional trials in the final run.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the Allen SDK `events` table, specifically the `filtered_events` column.

ii.
```python
events = ds.events
neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
```

iii. Steps 21, 45, 54, 77, and 93 show the justification: the AI inspected both `events` and `filtered_events`, noted that raw events were extremely sparse, and chose the half-Gaussian-filtered deconvolved events because it believed they matched the reference paper’s neural analysis and were more decodable at 32 ms resolution.

## 2-b. How is the `neural` data processed?

i. The AI stacks all cells’ `filtered_events` for a session into a `cells x time` array, then slices fixed `[-2, +2] s` trial windows directly from that matrix without additional normalization or denoising in code.

ii.
```python
neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
...
neural.append(neural_all[:, i0:i0 + T])
...
'neural': np.stack(neural),  # (ntrials, ncells, T)
```

iii. The trajectory justification is explicit in steps 54, 77, and 93: the AI viewed `filtered_events` itself as the desired processed signal and chose not to apply extra processing beyond trial slicing because the SDK signal had already been deconvolved and smoothed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No per-neuron quality filtering is applied beyond using the cells already present in the released NWB file. The AI assumes those ROIs already passed the Allen ROI/QC pipeline. It does, however, reject whole sessions with bad frame rates or no cells.

ii.
```python
if not (MIN_FRAME_RATE <= frame_rate <= MAX_FRAME_RATE):
    return {'oeid': oeid, 'error': f'frame rate {frame_rate:.2f} Hz'}
...
if neural_all.shape[0] == 0:
    return {'oeid': oeid, 'error': 'no cells'}
...
# all cells contained in the released NWB file are used.
# The published files only contain ROIs that passed the Allen ROI-filtering /
# QC pipeline (valid_roi == True), so no further neuron curation is applied.
```

iii. Step 21 confirmed that `valid_roi` was true for inspected cells, and steps 54 and 93 state the AI’s justification directly: all released ROIs were treated as already QC-filtered, so no additional neuron curation was added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `change_time` (real change on go trials, sham change on catch trials), not to the experiment-defined trial start. The saved trial starts 2 s before `change_time` and ends 2 s after it.

ii.
```python
OFF_START = -2.0
OFF_END = 2.0
...
change_time = float(tr.change_time)
i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
tt = ts[i0:i0 + T]
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. Steps 47, 54, 79, and 93 provide the stated rationale: `change_time` exactly matched the relevant flash onset, and a change-centered window gave identical trial lengths at the ophys frame rate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native ophys frame interval for the selected sessions, about 32.3 ms per bin (31 Hz). No temporal rebinning of neural data is applied.

ii.
```python
dt = float(np.median(np.diff(ts)))
frame_rate = 1.0 / dt
...
T = int(round((OFF_END - OFF_START) / dt))
...
'time_bin_size': float(np.mean(dts) * 1000.0),
```

iii. In steps 42, 54, 69, and 93 the AI emphasizes that the kept sessions were all single-plane 31 Hz recordings, so the native ophys frame times themselves served as the common time bins.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, using the non-omitted `image_name` values and their `start_time`s within the `change_detection_behavior` block.

ii.
```python
sp = ds.stimulus_presentations
sp = sp[sp.stimulus_block_name == 'change_detection_behavior']
sp = sp.sort_values('start_time')
shown = sp[~sp.omitted.astype(bool)]
flash_start = shown.start_time.values.astype(np.float64)
flash_image = shown.image_name.values.astype(str)
```

iii. Steps 33 and 47 show why the AI used this source: it inspected `stimulus_presentations`, confirmed 750 ms flash cycles, verified omitted flashes, and then described image identity in steps 54 and 93 as the flashed image held over the whole image-presentation cycle.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each ophys frame in the fixed trial window, the AI assigns the most recent non-omitted flashed image, so identity is held through the gray period and through omitted flashes. It then creates a global sorted list of unique image names and maps them to integer IDs.

ii.
```python
k = np.searchsorted(flash_start, tt, side='right') - 1
...
image_name.append(flash_image[k])
...
images = sorted({str(im) for s in sessions
                 for im in np.unique(s['image_name'])})
image_to_idx = {im: i for i, im in enumerate(images)}
...
img = np.vectorize(image_to_idx.__getitem__)(s['image_name'])
```

iii. Steps 47, 54, 81, and 93 justify this as matching the flashed-image cycle: the AI explicitly says the identity should persist over the 250 ms image, the following 500 ms gray screen, and omitted flashes because omissions continue the current image state rather than introducing a new image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the exact per-trial ophys frame times `tt` used for neural slicing, so it is frame-aligned to the neural matrix within the fixed `[-2, +2] s` window.

ii.
```python
tt = ts[i0:i0 + T]
...
k = np.searchsorted(flash_start, tt, side='right') - 1
image_name.append(flash_image[k])
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. The AI’s justification in steps 54 and 93 is that all non-neural streams should be resampled or indexed onto the ophys frame times, which it uses as the common alignment base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.start_time` for rows where `is_change` is true, together with the fixed ophys-frame time grid of each trial.

ii.
```python
change_start = sp[sp.is_change.astype(bool)].start_time.values.astype(np.float64)
...
k = np.searchsorted(change_start, tt, side='right') - 1
ch = np.zeros(T, dtype=np.int16)
valid = k >= 0
ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
```

iii. In step 47 the AI verified that `change_time` aligned exactly to change-flash onsets and that catch trials corresponded to sham changes. Steps 54 and 93 then describe the output as the 750 ms cycle that starts with a real image change, with sham changes staying zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI builds a binary vector per trial and marks frames as `1` when they fall within `IMAGE_CYCLE = 0.75` s of the most recent real change flash; all other frames are `0`.

ii.
```python
IMAGE_CYCLE = 0.75
...
k = np.searchsorted(change_start, tt, side='right') - 1
ch = np.zeros(T, dtype=np.int16)
valid = k >= 0
ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
change.append(ch)
```

iii. The stated justification in steps 54, 81, and 93 is that a “change” should label the full 750 ms image-presentation cycle after a real change, not just the single changed frame, and that catch-trial sham changes should stay zero.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary and is stored as integer categories `0 = no_change`, `1 = change`.

ii.
```python
ch = np.zeros(T, dtype=np.int16)
...
'output_values': [
    images,
    ['no_change', 'change'],
    ...
]
```

iii. The trajectory does not show a separate debate about thresholding; the AI treated image change as inherently binary throughout the implementation.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same ophys frame times `tt` used for the per-trial neural matrix, inside the fixed `[-2, +2] s` change-centered trial window.

ii.
```python
tt = ts[i0:i0 + T]
...
ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. Steps 54 and 93 give the general alignment rationale: all outputs should live on the same ophys-frame grid as neural activity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, specifically its `timestamps` and `speed` columns.

ii.
```python
run = ds.running_speed
run_t = run.timestamps.values.astype(np.float64)
run_v = run.speed.values.astype(np.float64)
```

iii. The trajectory contains no special argument here beyond inspecting the Allen object in step 21 and then using the standard running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI removes non-finite samples, linearly interpolates running speed onto each trial’s ophys frame times, concatenates all trial values across all sessions to compute global percentile edges, and then bins each frame with `np.searchsorted`.

ii.
```python
good = np.isfinite(run_t) & np.isfinite(run_v)
run_t, run_v = run_t[good], run_v[good]
...
running.append(np.interp(tt, run_t, run_v))
...
all_running = np.concatenate([s['running'].ravel() for s in sessions])
run_edges = percentile_bins(all_running, NBINS_BEHAVIOR)
...
run_bin = np.searchsorted(run_edges, s['running'], side='right')
```

iii. Steps 49, 54, 69, and 93 show the rationale: running speed is continuous, should be resampled to the ophys time base, and should be discretized with global equal-count percentile bins so category meanings stay consistent across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five percentile bins computed globally across all retained sessions and trials.

ii.
```python
NBINS_BEHAVIOR = 5
...
def percentile_bins(values, nbins):
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)
...
run_edges = percentile_bins(all_running, NBINS_BEHAVIOR)
run_bin = np.searchsorted(run_edges, s['running'], side='right')
```

iii. The AI states in steps 54 and 93 that one shared output head across sessions motivates global percentile bins rather than per-session bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the same per-trial ophys frame vector `tt` used for neural slicing.

ii.
```python
tt = ts[i0:i0 + T]
...
running.append(np.interp(tt, run_t, run_v))
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. The AI’s explicit justification in steps 54 and 93 is that the ophys frame times are the universal time base for the converted dataset.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, using `pupil_area`, `timestamps`, and `likely_blink`.

ii.
```python
eye = ds.eye_tracking
...
pupil_t = eye.timestamps.values.astype(np.float64)
pupil_d = 2.0 * np.sqrt(eye.pupil_area.values.astype(np.float64) / np.pi)
good = (np.isfinite(pupil_t) & np.isfinite(pupil_d)
        & ~eye.likely_blink.values.astype(bool))
```

iii. Steps 21 and 49 show the AI inspected both `pupil_area` and `pupil_width`, then in steps 54 and 93 explicitly chose the diameter derived from area as a geometric pupil-diameter estimate, removing blink frames.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts pupil area to diameter as `2*sqrt(area/pi)`, removes blink and non-finite samples, linearly interpolates to the per-trial ophys frame times, computes global percentile edges across all trial values, and bins frames with `np.searchsorted`.

ii.
```python
pupil_d = 2.0 * np.sqrt(eye.pupil_area.values.astype(np.float64) / np.pi)
good = (np.isfinite(pupil_t) & np.isfinite(pupil_d)
        & ~eye.likely_blink.values.astype(bool))
pupil_t, pupil_d = pupil_t[good], pupil_d[good]
...
pupil.append(np.interp(tt, pupil_t, pupil_d))
...
all_pupil = np.concatenate([s['pupil'].ravel() for s in sessions])
pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
...
pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
```

iii. Steps 49, 54, 69, and 93 provide the justification: blink-corrupted frames should be dropped before interpolation, and the derived diameter should then be discretized with global equal-count bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five percentile bins computed globally across all retained trial values.

ii.
```python
pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
...
[f'pupil_diameter_bin{i}' for i in range(NBINS_BEHAVIOR)]
```

iii. The stated rationale is the same as for running speed in steps 54 and 93: fixed global bins make the categorical labels consistent across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto the same per-trial ophys frame vector `tt` used for neural slicing.

ii.
```python
tt = ts[i0:i0 + T]
...
pupil.append(np.interp(tt, pupil_t, pupil_d))
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. The trajectory rationale is the same general one from steps 54 and 93: all streams are put on the ophys frame clock before saving.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `ds.trials`.

ii.
```python
if tr.hit:
    o = 0
elif tr.miss:
    o = 1
elif tr.false_alarm:
    o = 2
elif tr.correct_reject:
    o = 3
```

iii. The trajectory justification is implicit: these are the Allen trial outcome flags already defined for the change-detection task, and step 81 confirms they matched the saved output labels in spot checks.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four boolean outcome flags to integer codes `0..3` in a fixed order and then repeats that code across all time bins of the trial.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
if tr.hit:
    o = 0
elif tr.miss:
    o = 1
elif tr.false_alarm:
    o = 2
elif tr.correct_reject:
    o = 3
...
np.full(T, s['outcome'][i], dtype=np.int16)
```

iii. In steps 54, 81, and 93 the AI treats trial outcome as a static per-trial output that should be constant over the whole saved window.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data mostly by dropping the affected session or trial. It skips sessions with missing running data, missing eye tracking, no valid pupil samples, no cells, or unexpected frame rates. It skips trials whose fixed windows run off the recording or start before any valid flashed image. It also rejects sessions that end up with fewer than two usable trials.

ii.
```python
if not (MIN_FRAME_RATE <= frame_rate <= MAX_FRAME_RATE):
    return {'oeid': oeid, 'error': f'frame rate {frame_rate:.2f} Hz'}
...
if len(run_t) == 0:
    return {'oeid': oeid, 'error': 'no running data'}
...
if len(eye) == 0:
    return {'oeid': oeid, 'error': 'no eye tracking'}
...
if len(pupil_t) == 0:
    return {'oeid': oeid, 'error': 'no valid pupil data'}
...
if i0 + T > len(ts):
    n_dropped_edge += 1
    continue
...
if np.any(k < 0):
    n_dropped_edge += 1
    continue
...
if len(neural) < 2:
    return {'oeid': oeid, 'error': f'only {len(neural)} usable trials'}
```

iii. Steps 69 and 85 explicitly note that three sessions were dropped because they had no eye-tracking rows at all. Step 93 summarizes this as necessary because pupil diameter was a required decoder output. The trajectory does not show a separate strategy for imputation beyond interpolation within valid samples.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is opening each NWB file and extracting its session arrays; the AI parallelizes that stage across processes and caches the results to disk.

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for i, res in enumerate(pool.map(extract_and_cache, oeids)):
        ...
```

iii. Step 45 measured roughly 7.3 s just to open one NWB file, versus milliseconds to pull tables already in memory. That led the AI to parallelize per-session extraction and add an on-disk cache in steps 54 and 93.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized loops are the per-trial extraction loop in `extract_session(...)` and the per-trial output assembly loop in `main()`. The AI did not vectorize those; it relied on NumPy inside each loop and parallelized across sessions instead.

ii.
```python
for _, tr in trials.iterrows():
    ...
    image_name.append(flash_image[k])
    ...
    running.append(np.interp(tt, run_t, run_v))
    pupil.append(np.interp(tt, pupil_t, pupil_d))
    ...
    neural.append(neural_all[:, i0:i0 + T])
...
for i in range(ntrials):
    out.append(np.stack([
        img[i].astype(np.int16),
        s['change'][i].astype(np.int16),
        run_bin[i].astype(np.int16),
        pupil_bin[i].astype(np.int16),
        np.full(T, s['outcome'][i], dtype=np.int16),
    ]))
```

iii. The trajectory does not contain an explicit “could be vectorized” discussion, but steps 45 and 93 show the AI considered session loading the dominant bottleneck and chose process-level parallelism as the optimization focus.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats a full pass over session outputs after extraction: it first writes each extracted session to a cache `.npz`, then reloads all cached sessions to assemble the final dataset. It also traverses all running and pupil samples once to compute global bin edges and again to bin the saved trials.

ii.
```python
res = extract_session(oeid)
...
np.savez(tmp, **res)
...
for oeid in oeids:
    with np.load(cache_path(oeid), allow_pickle=True) as f:
        ...
        sessions.append({k: f[k] for k in f.files})
...
all_running = np.concatenate([s['running'].ravel() for s in sessions])
all_pupil = np.concatenate([s['pupil'].ravel() for s in sessions])
...
run_bin = np.searchsorted(run_edges, s['running'], side='right')
pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
```

iii. The trajectory justification is indirect: steps 45, 54, and 93 show the AI added caching and a two-pass assembly to manage large files, parallel workers, and global discretization bins.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores some intermediate values that are not used in the final decoder-facing arrays, such as `cell_specimen_ids`, `n_dropped_edge`, `n_outside_trial`, and `n_trials_available` in the per-session cache files. It also performs the cache write/read round trip even though the final pickle does not expose those caches.

ii.
```python
return {
    'oeid': oeid,
    ...
    'cell_specimen_ids': cell_specimen_ids,
    'frame_rate': frame_rate,
    'dt': dt,
    'T': T,
    'n_dropped_edge': n_dropped_edge,
    'n_outside_trial': n_outside_trial,
    'n_trials_available': int(len(trials)),
}
...
np.savez(tmp, **res)
...
with np.load(cache_path(oeid), allow_pickle=True) as f:
    ...
```

iii. The trajectory does not label this as unnecessary; the closest justification is in steps 54 and 93, where the AI presents the cache as a practical engineering choice for extraction speed and recoverability rather than as analysis-essential output.
