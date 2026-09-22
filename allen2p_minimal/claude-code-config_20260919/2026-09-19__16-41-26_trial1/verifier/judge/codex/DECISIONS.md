# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local `ophys_experiment_table.csv`, discovers locally available NWB experiment IDs, retains available active single-plane `VisualBehavior` rows, and loads each retained NWB directly with `BehaviorOphysExperiment.from_nwb_path`. Extraction is parallelized and cached as one NPZ per experiment. It excludes passive sessions and the `VisualBehaviorMultiscope` project.

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
ds = BehaviorOphysExperiment.from_nwb_path(path)
...
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for i, res in enumerate(pool.map(extract_and_cache, oeids)):
```

iii. The agent said `VisualBehavior` provides uniform ~31 Hz single-plane data, whereas multiscope data have a different per-plane rate. Passive sessions were excluded because trial outcome is behaviorally undefined there. Direct local loading avoids fetching absent data, and caching/parallelism addresses the large I/O cost.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id` entries among retained sessions. Each session gets the index of its mouse in a sorted global subject list.

ii.
```python
subjects = sorted(table.mouse_id.astype(str).unique().tolist())
subject_idx = np.array([subjects.index(str(m)) for m in table.mouse_id],
                       dtype=np.int64)
```

iii. The agent relied on `mouse_id` as the dataset’s animal identifier and used a deterministic sorted mapping.

## 1-c. How are the data split into sessions?

i. Each retained `VisualBehavior` ophys experiment is treated as one session. The code asserts a one-to-one relationship between `ophys_session_id` and experiment rows after filtering, orders them by `ophys_experiment_id`, and does not combine planes.

ii.
```python
assert et.ophys_session_id.nunique() == len(et)
et = et.sort_values('ophys_experiment_id').reset_index(drop=True)
...
oeids = list(table.ophys_experiment_id.values)
```

iii. The agent reasoned that this project contains single-plane recordings, so one experiment represents one imaging plane/session. This also avoids mixing the slower multiscope recordings.

## 1-d. How are the data split into trials?

i. Trial definitions come from `ds.trials`. The agent keeps go and catch trials and makes every converted trial a fixed four-second window, from 2 seconds before to 2 seconds after real or sham `change_time`, sampled as `T` consecutive ophys frames.

ii.
```python
trials = ds.trials
keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
        & ~trials.aborted.astype(bool)
        & ~trials.auto_rewarded.astype(bool)
        & trials.change_time.notna())
trials = trials[keep]
...
i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
tt = ts[i0:i0 + T]
neural.append(neural_all[:, i0:i0 + T])
```

iii. The agent chose the experiment’s canonical trial table but used a fixed change-centered window to make all trials and sessions have identical temporal dimensions while retaining pre- and post-change information. It reported verifying that these windows lie inside the experiment-defined trials.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials with a non-null change time are retained; aborted and auto-rewarded trials are removed. Trials that hit the recording edge, lack a preceding shown image, or have no recognized outcome are dropped. A session is rejected if fewer than two usable trials remain. Entire sessions are also rejected for invalid frame rate, no cells, no running data, or no valid pupil data.

ii.
```python
keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
        & ~trials.aborted.astype(bool)
        & ~trials.auto_rewarded.astype(bool)
        & trials.change_time.notna())
...
if i0 + T > len(ts):
    continue
if np.any(k < 0):
    continue
...
if len(neural) < 2:
    return {'oeid': oeid, 'error': f'only {len(neural)} usable trials'}
```

iii. The required exclusions remove early-lick and free-reward trials. Additional checks prevent malformed alignment and missing required outputs from entering the dataset; the two-trial minimum is required by the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `ds.events['filtered_events']`, the Allen SDK’s filtered detected calcium-event traces, not from `dff_traces`.

ii.
```python
events = ds.events
neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
```

iii. The agent said the paper uses detected L0-regularized calcium events and chose the half-Gaussian-filtered event signal because raw events are extremely sparse at 32 ms resolution; its probe reportedly improved image decoding relative to unfiltered events.

## 2-b. How is the `neural` data processed?

i. The precomputed filtered-event vectors are stacked into a cells-by-time matrix, cast to `float32`, checked against ophys timestamp length, and sliced into fixed trial windows. There is no further normalization or rebinning.

ii.
```python
neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
if neural_all.shape[1] != len(ts):
    return {'oeid': oeid, 'error': 'events / timestamps length mismatch'}
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. The filtering/deconvolution is already supplied by the SDK, and retaining its native ophys sampling preserves the processed neural signal without another transform.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra per-neuron filtering is performed. The code includes every cell in `events`, but rejects a session if it has zero cells or if event-vector length differs from the timestamp count.

ii.
```python
if neural_all.shape[1] != len(ts):
    return {'oeid': oeid, 'error': 'events / timestamps length mismatch'}
if neural_all.shape[0] == 0:
    return {'oeid': oeid, 'error': 'no cells'}
```

iii. The agent stated that released NWB files already contain ROIs that survived Allen’s ROI/QC filtering (`valid_roi`), so additional curation was unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to real or sham image `change_time`. The first bin is the first ophys timestamp at or after `change_time - 2 s`, followed by exactly `T` frames through approximately `change_time + 2 s`.

ii.
```python
change_time = float(tr.change_time)
i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
tt = ts[i0:i0 + T]
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. The agent identified `change_time` as the meaningful common event for go and catch trials and used ophys timestamps as explicitly required. A fixed window enables direct comparisons and uniform tensor sizes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data remain at the native single-plane ophys rate, approximately 31 Hz or 32.3 ms per bin. No temporal rebinning is applied. The code checks for 30–32 Hz, calculates 124 samples for the four-second window, asserts equal `T` across sessions, and records the mean session `dt`.

ii.
```python
dt = float(np.median(np.diff(ts)))
frame_rate = 1.0 / dt
if not (MIN_FRAME_RATE <= frame_rate <= MAX_FRAME_RATE):
    return {'oeid': oeid, 'error': f'frame rate {frame_rate:.2f} Hz'}
T = int(round((OFF_END - OFF_START) / dt))
...
assert np.all(Ts == Ts[0])
```

iii. The agent excluded recordings with incompatible frame rates so native ophys samples could be used uniformly without resampling neural activity.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`: `stimulus_block_name`, `omitted`, `start_time`, and `image_name`.

ii.
```python
sp = ds.stimulus_presentations
sp = sp[sp.stimulus_block_name == 'change_detection_behavior']
sp = sp.sort_values('start_time')
shown = sp[~sp.omitted.astype(bool)]
flash_start = shown.start_time.values.astype(np.float64)
flash_image = shown.image_name.values.astype(str)
```

iii. The agent used the actual presentation stream rather than only trial-level initial/change labels so identity follows every presentation and can handle omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each ophys sample, the most recent non-omitted flash is found and its image is held until the next shown flash, including gray periods and omissions. All observed names are sorted globally and converted to integer category indices.

ii.
```python
k = np.searchsorted(flash_start, tt, side='right') - 1
image_name.append(flash_image[k])
...
images = sorted({str(im) for s in sessions
                 for im in np.unique(s['image_name'])})
image_to_idx = {im: i for i, im in enumerate(images)}
img = np.vectorize(image_to_idx.__getitem__)(s['image_name'])
```

iii. The agent interpreted image identity as the current image over the full 750 ms presentation cycle; omissions continue the current gray/image context. A single global deterministic mapping keeps class codes consistent.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the same trial ophys timestamps `tt` used to slice neural data.

ii.
```python
tt = ts[i0:i0 + T]
k = np.searchsorted(flash_start, tt, side='right') - 1
image_name.append(flash_image[k])
neural.append(neural_all[:, i0:i0 + T])
```

iii. Looking up each presentation at each neural-frame timestamp puts stimulus categories and neural samples on exactly the same time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from the `is_change` and `start_time` fields of behavior-block stimulus presentations.

ii.
```python
change_start = sp[sp.is_change.astype(bool)].start_time.values.astype(np.float64)
```

iii. `is_change` distinguishes real image changes from sham catch events, so catch trials remain zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. At every trial timestamp, the code finds the latest real-change onset and labels the sample 1 if fewer than 0.75 seconds have elapsed; otherwise it labels 0.

ii.
```python
k = np.searchsorted(change_start, tt, side='right') - 1
ch = np.zeros(T, dtype=np.int16)
valid = k >= 0
ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
```

iii. The 750 ms interval represents the changed image’s 250 ms flash plus the following 500 ms gray interval and is consistent with the agent’s presentation-cycle convention.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: category 1 applies from a real change onset until elapsed time reaches 0.75 seconds; category 0 applies otherwise. There is no percentile thresholding.

ii.
```python
ch = np.zeros(T, dtype=np.int16)
ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
...
['no_change', 'change']
```

iii. The requested variable is binary, and sham changes are intentionally not classified as real changes.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change labels are computed directly at the same `tt` ophys samples used for the neural slice.

ii.
```python
tt = ts[i0:i0 + T]
k = np.searchsorted(change_start, tt, side='right') - 1
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. This frame-by-frame lookup gives the change output the same length and clock as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed.timestamps` and `ds.running_speed.speed`.

ii.
```python
run = ds.running_speed
run_t = run.timestamps.values.astype(np.float64)
run_v = run.speed.values.astype(np.float64)
```

iii. This is the SDK’s processed wheel-speed stream in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite timestamp/value pairs are removed, speed is linearly interpolated at trial ophys timestamps, and global 20th/40th/60th/80th percentile edges across all kept samples define five bins.

ii.
```python
good = np.isfinite(run_t) & np.isfinite(run_v)
run_t, run_v = run_t[good], run_v[good]
running.append(np.interp(tt, run_t, run_v))
...
run_edges = percentile_bins(all_running, NBINS_BEHAVIOR)
run_bin = np.searchsorted(run_edges, s['running'], side='right')
```

iii. Interpolation aligns speed with imaging frames, while global equal-count bins balance categories and give every session the same physical category boundaries.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global percentile cut points create five integer categories 0–4; values equal to an edge enter the higher bin because `side='right'` is used.

ii.
```python
qs = np.linspace(0, 100, nbins + 1)[1:-1]
return np.percentile(values, qs)
...
run_bin = np.searchsorted(run_edges, s['running'], side='right')
```

iii. Five equal-percentile bins were explicitly required, and global edges make labels comparable across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated directly at `tt`, the ophys timestamp vector corresponding to the exact neural trial slice.

ii.
```python
tt = ts[i0:i0 + T]
running.append(np.interp(tt, run_t, run_v))
neural.append(neural_all[:, i0:i0 + T])
```

iii. Both arrays therefore have one value per identical imaging frame.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking.timestamps`, `pupil_area`, and `likely_blink`.

ii.
```python
eye = ds.eye_tracking
pupil_t = eye.timestamps.values.astype(np.float64)
pupil_d = 2.0 * np.sqrt(eye.pupil_area.values.astype(np.float64) / np.pi)
good = (np.isfinite(pupil_t) & np.isfinite(pupil_d)
        & ~eye.likely_blink.values.astype(bool))
```

iii. The agent used the equivalent-circle diameter from the fitted pupil area and the SDK blink flag to remove artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to diameter as `2*sqrt(area/pi)`, blink and non-finite samples are removed, remaining samples are linearly interpolated at ophys timestamps, and global percentile edges discretize the result into five bins.

ii.
```python
pupil_d = 2.0 * np.sqrt(eye.pupil_area.values.astype(np.float64) / np.pi)
pupil_t, pupil_d = pupil_t[good], pupil_d[good]
pupil.append(np.interp(tt, pupil_t, pupil_d))
...
pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
```

iii. Removing blinks before interpolation prevents artifacts, and the global equal-count scheme provides balanced, consistent decoder classes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global percentile edges (20%, 40%, 60%, 80%) create categories 0–4, with equality assigned to the higher bin.

ii.
```python
pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
```

iii. This implements the requested five equal-percentile categories consistently across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Valid pupil observations are interpolated directly at the trial’s ophys timestamps `tt`, which correspond to the neural slice.

ii.
```python
tt = ts[i0:i0 + T]
pupil.append(np.interp(tt, pupil_t, pupil_d))
neural.append(neural_all[:, i0:i0 + T])
```

iii. Hardware-synchronized timestamps and evaluation on `tt` put pupil and neural data on a common frame-level clock.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. These are the canonical mutually exclusive behavioral outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four booleans map to integer codes 0–3 in the declared order. The selected scalar is repeated across every time bin of that trial.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
np.full(T, s['outcome'][i], dtype=np.int16)
```

iii. Outcome is static per trial, but repetition gives it the common `(output, time)` representation expected by the assembled output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite running and pupil observations and blink samples are removed before interpolation. Sessions with no required stream, invalid frame rate, zero cells, timestamp mismatch, or fewer than two trials are stored as errors and skipped. Edge/malformed trials and trials without a canonical outcome are skipped. Worker exceptions are caught and cached. The agent does not impute a wholly missing pupil stream; it removes that session.

ii.
```python
if len(run_t) == 0:
    return {'oeid': oeid, 'error': 'no running data'}
...
if len(pupil_t) == 0:
    return {'oeid': oeid, 'error': 'no valid pupil data'}
...
try:
    res = extract_session(oeid)
except Exception:
    res = {'oeid': oeid, 'error': traceback.format_exc(limit=3)}
...
if 'error' in f:
    skipped.append((oeid, str(f['error'])))
    continue
```

iii. The agent prioritized complete, aligned required outputs and allowed one bad session/trial not to abort the conversion. It reported that three sessions were excluded because eye tracking was entirely absent.

## 9-a. What are the most time-consuming steps of the code?

i. Reading and parsing large NWB files and extracting full-session event/behavior arrays are the dominant work; serialization of the multi-gigabyte final pickle is also substantial. The agent mitigates extraction with multiprocessing and reusable per-session NPZ caches.

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for i, res in enumerate(pool.map(extract_and_cache, oeids)):
...
np.savez(tmp, **res)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory reports a 3.94 GB result across 165 sessions, so large-file I/O and SDK/NWB parsing naturally dominate. Caching avoids repeating extraction on reruns.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trials.iterrows()` extraction loop and per-trial output assembly loop could be partly vectorized because all trials have fixed `T`. Subject lookup via repeated `subjects.index`, region lookup, image mapping via Python `np.vectorize`, and filename discovery could also use dictionary/vectorized mappings. Some per-trial slicing still naturally produces list elements required by the target format.

ii.
```python
for _, tr in trials.iterrows():
    ...
for i in range(ntrials):
    out.append(np.stack([...]))
...
subject_idx = np.array([subjects.index(str(m)) for m in table.mouse_id])
img = np.vectorize(image_to_idx.__getitem__)(s['image_name'])
```

iii. The agent did not explicitly discuss these loops in its final rationale. Its use of fixed-length stacked arrays makes more batch processing possible, although NWB I/O is likely the larger bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. Extracted trials are written to NPZ and then read back for assembly, so freshly extracted data incur an extra serialization round trip. Session arrays are traversed once to concatenate global running/pupil samples, again to collect image names, and again to assemble output. Each trial is also transformed into the final list representation individually.

ii.
```python
np.savez(tmp, **res)
...
with np.load(cache_path(oeid), allow_pickle=True) as f:
    sessions.append({k: f[k] for k in f.files})
...
all_running = np.concatenate([s['running'].ravel() for s in sessions])
images = sorted({str(im) for s in sessions for im in np.unique(s['image_name'])})
for s, (_, row) in zip(sessions, table.iterrows()):
```

iii. The cache round trip is deliberate for restartability and bounded extraction workflow; repeated passes are used to calculate global bins and category mappings before final encoding.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `cell_specimen_ids` are extracted and cached but never placed in the final dataset. Diagnostic counters (`n_dropped_edge`, `n_outside_trial`, `n_trials_available`) and some cache metadata are retained only for reporting. Extensive `session_info` metadata is serialized but is not required by the decoder. Pupil diameter is computed and interpolated in continuous form before only categorical bins are retained, although that transform is necessary to construct the requested category.

ii.
```python
cell_specimen_ids = np.asarray(events.index.values)
...
'cell_specimen_ids': cell_specimen_ids,
'n_dropped_edge': n_dropped_edge,
'n_outside_trial': n_outside_trial,
'n_trials_available': int(len(trials)),
```

iii. The agent did not identify discarded work explicitly. These fields mainly support validation, provenance, and diagnostics rather than decoder training; `cell_specimen_ids` in particular never reaches the final object.
