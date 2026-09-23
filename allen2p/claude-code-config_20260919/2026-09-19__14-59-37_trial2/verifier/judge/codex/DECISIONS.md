# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache/project table path from the reference solution. Instead, it reads the local `ophys_experiment_table.csv`, keeps only experiment IDs whose NWB files are present in `/app/data`, drops passive sessions up front, groups experiments by `ophys_session_id`, and loads each NWB directly with `BehaviorOphysExperiment.from_nwb_path`. Within each selected session it loads all imaging planes, then takes trials/running/eye/stimulus data from the first plane (with fallback to later planes if a behavior stream is empty).

ii.
```python
def select_sessions():
    exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    have = set()
    for f in os.listdir(EXP_DIR):
        if f.endswith('.nwb'):
            have.add(int(f.split('_')[-1].split('.')[0]))
    exp = exp[exp.ophys_experiment_id.isin(have)]
    exp = exp[~exp.passive]
    exp = exp.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    return exp

experiments = [load_experiment(e) for e in eids]
ds0 = experiments[0]
trials = ds0.trials
stim = ds0.stimulus_presentations
running = ds0.running_speed
eye = ds0.eye_tracking
```

iii. The notes say the SDK `BehaviorOphysExperiment` object is the canonical loader, but the agent chose direct NWB loading from local files because the task environment already contains the NWBs. The notes also justify restricting to active sessions and sessions with eye tracking because passive sessions lack meaningful outcomes and pupil is a required decoder output.

## 1-b. How are the data split into subjects?

i. Subjects are unique mouse IDs taken from the converted sessions' metadata and sorted globally at assembly time.

ii.
```python
subjects = sorted({r['mouse_id'] for r in ok})
'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok], dtype=np.int64),
```

iii. The notes map `ophys_experiment_table.mouse_id` to `subjects`/`subject_idx` and describe mouse ID as the subject identifier.

## 1-c. How are the data split into sessions?

i. Sessions are unique `ophys_session_id` groups. If a session has multiple simultaneously recorded imaging planes, those planes are treated as one session and later concatenated along the neuron axis.

ii.
```python
groups = exp.groupby('ophys_session_id')
...
jobs.append(dict(ophys_session_id=int(sid),
                 experiment_ids=g.ophys_experiment_id.tolist(),
                 trace=args.trace,
                 show_processing=args.show_processing and i < 2,
                 plot_dir=args.plot_dir))
```

iii. The notes explicitly say “Session = `ophys_session_id` (not NWB file)” because the whitepaper defines a session as one continuous recording, including Multiscope sessions with multiple planes.

## 1-d. How are the data split into trials?

i. Trials are rows of the Allen SDK `trials` table filtered to `go | catch`. For each selected trial, the AI uses the full `[start_time, stop_time)` interval, but instead of slicing native ophys frames directly it creates a fixed-width 1/31 s bin grid anchored at trial start.

ii.
```python
sel = trials[(trials.go | trials.catch)].sort_values('start_time')
...
for tid, tr in sel.iterrows():
    t_beg, t_stop = float(tr.start_time), float(tr.stop_time)
    nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
    if nbins < 2:
        continue
    tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
```

iii. The notes justify `go | catch` as exactly matching the task’s inclusion rule. They justify the fixed bins as a way to keep one common time bin size across all sessions, especially to combine 31 Hz single-plane and 11 Hz Multiscope sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only `go` or `catch` trials, which it treats as implicitly excluding aborted and auto-rewarded trials. It also requires at least 2 bins, requires the trial bin centers to be fully covered by every plane’s ophys timestamps, and later drops sessions with fewer than 2 usable trials.

ii.
```python
sel = trials[(trials.go | trials.catch)].sort_values('start_time')
assert not sel.aborted.any() and not sel.auto_rewarded.any()
assert sel.change_time.notna().all()
...
if nbins < 2:
    continue
if any(tc[0] < ts[0] or tc[-1] > ts[-1] for ts in plane_ts):
    continue
...
if len(neural_trials) < 2:
    return {'ophys_session_id': sid, 'error': 'fewer than 2 usable trials'}
```

iii. The notes say `go | catch` was chosen because it exactly excludes aborted and auto-rewarded trials, and that full ophys coverage plus the 2-trial minimum are defensive checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the final code, the default neural signal is `dff_traces.dff`, though the script also allows `events` or `filtered_events` via `--trace`. The notes are internally inconsistent: an earlier section argues for `filtered_events`, but later sections and the final summary say the agent changed the default to dF/F after benchmarking.

ii.
```python
ap.add_argument('--trace', default='dff',
                choices=['dff', 'filtered_events', 'events'])
...
if trace_name == 'dff':
    tbl = ds.dff_traces
    mat = np.stack(tbl['dff'].values).astype(np.float32)
else:
    tbl = ds.events
    mat = np.stack(tbl[trace_name].values).astype(np.float32)
```

iii. The final notes and trajectory say dF/F beat `filtered_events` and raw `events` in decoder benchmarks, so dF/F was made the default despite the paper’s use of detected events.

## 2-b. How is the `neural` data processed?

i. For each plane in a session, the AI loads the chosen trace matrix, casts it to `float32`, and samples it at trial-specific 1/31 s bin centers by taking the nearest ophys frame. If a session has multiple planes, the per-plane binned matrices are concatenated along the neuron axis.

ii.
```python
def nearest_index(sorted_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_times, query)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left = sorted_times[idx - 1]
    right = sorted_times[idx]
    idx = np.where(query - left <= right - query, idx - 1, idx)
    return np.clip(idx, 0, len(sorted_times) - 1)

mats = [mat[:, nearest_index(ts, tc)]
        for mat, ts in zip(plane_traces, plane_ts)]
neural = np.concatenate(mats, axis=0) if len(mats) > 1 else mats[0]
```

iii. The notes justify this as creating a common time base across 31 Hz and 11 Hz sessions while preserving ophys-timestamp alignment. Later notes describe the 31 Hz case as almost identity and the 11 Hz case as zero-order hold.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra neuron-level filtering is applied. The AI keeps all ROIs/cells returned by the Allen pipeline.

ii.
```python
'curation':
    '... All ROIs returned by the SDK are used: ROI quality control ... is already applied by '
    'the Allen pipeline.',
```

iii. The notes state that ROI QC is already applied upstream and that `cell_specimen_table`, `events`, and `dff_traces` already contain only valid ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to `trials.start_time`. Each time bin is centered at `start_time + (k + 0.5) * BIN_SIZE`, and the neural value is the nearest ophys frame to that bin center.

ii.
```python
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
mats = [mat[:, nearest_index(ts, tc)]
        for mat, ts in zip(plane_traces, plane_ts)]
```

iii. The notes justify trial-start alignment because the task says to use ophys timestamps and the target format requires one common time bin size. Metadata also describes the alignment event as trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a fixed bin size of `1/31 s` (`32.258 ms`) for every session and trial. Yes: it rebins/resamples neural and behavioral streams onto that common grid.

ii.
```python
BIN_SIZE = 1.0 / 31.0
...
nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
...
'time_bin_size': 1000.0 * BIN_SIZE,
```

iii. The notes argue this was necessary because the target format requires one common time bin size, and the dataset mixes 31 Hz single-plane sessions with 11 Hz Multiscope sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from `stimulus_presentations` in the `change_detection_behavior` block, using `start_time`, `end_time`, `image_name`, and `omitted`. It does not use `initial_image_name`/`change_image_name` from the trials table.

ii.
```python
beh = stim[stim.stimulus_block_name == BEHAVIOR_BLOCK].sort_values('start_time')
flash_start = beh.start_time.values.astype(float)
flash_end = beh.end_time.values.astype(float)
flash_img = beh.image_name.values.astype(str)
flash_omitted = beh.omitted.values.astype(bool)
```

iii. The notes say this lets image identity reflect the actual flashed stimulus stream, including omitted flashes and the blank inter-flash periods, rather than only the trial-level pre/post change image names.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each session, the AI builds a local lookup for the non-omitted image names, reserves code `0` for `"gray"`, and labels each time bin according to which flash interval contains the bin center. After session conversion, it remaps each session’s local image codes into a global codebook across all sessions.

ii.
```python
image_names = sorted(set(flash_img[~flash_omitted]) - {'omitted'})
img_lookup = {name: i + 1 for i, name in enumerate(image_names)}  # 0 = gray
...
j = np.searchsorted(flash_start, tc, side='right') - 1
j = np.clip(j, 0, len(flash_start) - 1)
on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
img = np.zeros(nbins, dtype=np.int64)
if np.any(on_screen):
    img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]
...
global_img = {name: i + 1 for i, name in enumerate(image_names)}
for r in ok:
    remap = np.zeros(len(r['image_names']) + 1, dtype=np.int64)
    for local, name in enumerate(r['image_names']):
        remap[local + 1] = global_img[name]
    for out in r['output']:
        out[0] = remap[out[0]]
```

iii. The notes justify the extra `"gray"` class because two-thirds of the task is the blank inter-flash interval and omitted flashes are also blank. They also note that the two image sets are disjoint, so the union across sessions gives 16 images plus gray.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated at the exact same trial bin centers used for neural alignment. A bin gets the image code only if its center falls within a stimulus flash interval; otherwise it gets the gray code.

ii.
```python
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
...
j = np.searchsorted(flash_start, tc, side='right') - 1
on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
img = np.zeros(nbins, dtype=np.int64)
if np.any(on_screen):
    img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]
```

iii. The notes explicitly describe all output streams as sampled at the same bin centers as the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the flash table rather than only the trial table: it uses `stimulus_presentations.is_change` to find the changed-image flash, while the trial table’s `go` flag determines whether a real change occurred.

ii.
```python
flash_is_change = beh.is_change.values.astype(bool)
...
chg = np.zeros(nbins, dtype=np.int64)
if bool(tr.go):
    k = int(np.searchsorted(flash_start, float(tr.change_time), side='right') - 1)
    assert flash_is_change[k], 'change_time does not match an is_change flash'
    chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
```

iii. The notes say this is meant to mark the actual changed flash shown on the screen, not just a broader post-change epoch.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For go trials, the AI sets `image_change = 1` only for bins whose centers fall inside the 250 ms changed-image flash. Catch trials remain all zeros.

ii.
```python
chg = np.zeros(nbins, dtype=np.int64)
if bool(tr.go):
    k = int(np.searchsorted(flash_start, float(tr.change_time), side='right') - 1)
    chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
```

iii. The notes justify the 250 ms window by saying it covers the stimulus event itself, whereas a single-bin pulse would be too sparse and a longer 750 ms window would extend beyond the changed flash.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` for all non-change bins and `1` for bins within the changed-image flash on go trials.

ii.
```python
chg = np.zeros(nbins, dtype=np.int64)
...
chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
```

iii. The notes describe this as a 2-category output: change versus no change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned to the same fixed trial bin centers as the neural data and image identity. The change bins are those whose centers lie within the flash interval identified from `change_time`.

ii.
```python
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
...
chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
```

iii. The notes repeatedly say all streams are sampled at common bin centers aligned to ophys time.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `running_speed.timestamps` and `running_speed.speed`.

ii.
```python
run_t = running['timestamps'].values.astype(float)
run_v = running['speed'].values.astype(float)
```

iii. The notes say the AI chose the AllenSDK’s filtered `running_speed` rather than `raw_running_speed`, matching the tutorials and whitepaper processing.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto each trial’s bin centers using `np.interp` via `interp_nonan`, concatenates all trial values within a session, computes session-specific quintile edges, and then assigns each bin to one of five labels.

ii.
```python
def interp_nonan(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(x)
    if not np.any(ok):
        return np.full(xq.shape, np.nan)
    return np.interp(xq, x[ok], y[ok])
...
run_trials.append(interp_nonan(run_t, run_v, tc))
...
run_all = np.concatenate(run_trials)
run_lab, run_edges = quantile_bin(run_all)
...
out[2] = run_lab[pos:pos + n]
```

iii. The notes justify interpolation to the common bin centers and session-level quintiles by arguing that running distributions differ substantially across animals/sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-frequency bins computed separately within each session.

ii.
```python
def quantile_bin(values: np.ndarray, nbins: int = N_QUANTILE_BINS):
    edges = np.quantile(values, np.arange(1, nbins) / nbins)
    labels = np.searchsorted(edges, values, side='right').astype(np.int64)
    return np.clip(labels, 0, nbins - 1), edges
```

iii. The notes explicitly defend per-session quintiles rather than pooled global quintiles, saying this gives balanced bins in every session and avoids mixing incomparable running distributions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is evaluated at the same trial bin centers as the neural data, not first on full-session `ophys_timestamps`.

ii.
```python
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
run_trials.append(interp_nonan(run_t, run_v, tc))
```

iii. The notes frame this as common-bin-center alignment of every stream to the ophys-based trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_width` and `eye_tracking.timestamps`.

ii.
```python
eye_t = eye['timestamps'].values.astype(float)
eye_v = eye['pupil_width'].values.astype(float)
```

iii. The notes say the agent treated `pupil_width` as pupil diameter, following the tutorials and metadata descriptions.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI interpolates `pupil_width` over finite samples only, so blink/outlier NaNs are skipped automatically. It then samples pupil at each trial’s bin centers, concatenates trial values within a session, computes session-specific quintile edges, and labels each bin accordingly.

ii.
```python
def interp_nonan(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(x)
    if not np.any(ok):
        return np.full(xq.shape, np.nan)
    return np.interp(xq, x[ok], y[ok])
...
pup_trials.append(interp_nonan(eye_t, eye_v, tc))
...
pup_all = np.concatenate(pup_trials)
pup_lab, pup_edges = quantile_bin(pup_all)
...
out[3] = pup_lab[pos:pos + n]
```

iii. The notes justify interpolating across blink NaNs because `pupil_width` is NaN on blink/outlier frames, and justify per-session binning because the absolute scale of pupil width in pixels is not comparable across rigs/sessions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five equal-frequency bins computed separately within each session.

ii.
```python
pup_lab, pup_edges = quantile_bin(pup_all)
...
['pupil_q%d' % i for i in range(N_QUANTILE_BINS)]
```

iii. The notes explicitly defend per-session quintiles for pupil because the raw values are in camera pixels and vary across zoom/eye position.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto the same trial bin centers used for neural data.

ii.
```python
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
pup_trials.append(interp_nonan(eye_t, eye_v, tc))
```

iii. The notes describe all streams as sampled on one ophys-based bin grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
if bool(tr.hit):
    outcome = 0
elif bool(tr.miss):
    outcome = 1
elif bool(tr.false_alarm):
    outcome = 2
elif bool(tr.correct_reject):
    outcome = 3
else:
    continue
```

iii. The notes map those Allen trial-table flags directly to the four output classes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome booleans to integer codes 0-3 once per trial, then broadcasts the chosen code across all time bins in that trial’s output matrix.

ii.
```python
if bool(tr.hit):
    outcome = 0
elif bool(tr.miss):
    outcome = 1
elif bool(tr.false_alarm):
    outcome = 2
elif bool(tr.correct_reject):
    outcome = 3
...
out[4] = outcome_trials[i]
```

iii. The notes say trial outcome is the only static-per-trial target and is broadcast over time because the output format is one `(5, T)` array per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases defensively: it drops sessions with no eye tracking or running data; treats omitted flashes as gray by forcing their `end_time` to `-inf`; skips trials shorter than 2 bins or not fully covered by ophys timestamps; rejects sessions if interpolated running or pupil contain non-finite values; and logs session-level errors rather than crashing the whole run.

ii.
```python
if len(eye) == 0 or not np.any(np.isfinite(eye['pupil_width'].values)):
    return {'ophys_session_id': sid, 'error': 'no eye tracking data'}
if len(running) == 0:
    return {'ophys_session_id': sid, 'error': 'no running data'}
...
flash_end = np.where(flash_omitted | ~np.isfinite(flash_end), -np.inf, flash_end)
...
if nbins < 2:
    continue
if any(tc[0] < ts[0] or tc[-1] > ts[-1] for ts in plane_ts):
    continue
...
if not np.all(np.isfinite(pup_all)):
    return {'ophys_session_id': sid, 'error': 'non-finite pupil after interpolation'}
if not np.all(np.isfinite(run_all)):
    return {'ophys_session_id': sid, 'error': 'non-finite running speed'}
...
return {'ophys_session_id': sid, 'error': repr(exc), 'traceback': traceback.format_exc()}
```

iii. The notes justify dropping no-eye sessions because pupil is a required output and “cannot be fabricated.” They also present the remaining checks as defensive data-integrity measures.

## 9-a. What are the most time-consuming steps of the code?

i. The AI treats NWB loading as the main bottleneck, especially opening each experiment file and extracting session data.

ii.
```python
experiments = [load_experiment(e) for e in eids]
...
with ProcessPoolExecutor(max_workers=nworkers) as ex:
    for i, r in enumerate(ex.map(convert_session, jobs)):
        ...
```

iii. The notes explicitly say the bottleneck is NWB reading (~250 MB per experiment) and motivate the use of a process pool over sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI mostly argues that the inner alignment work already is vectorized (`searchsorted`, `interp`, bulk array indexing). The only inefficiency it explicitly calls out is that the planes of a Multiscope session are loaded serially inside one worker. The code still retains a Python loop over trials for per-trial assembly.

ii.
```python
for tid, tr in sel.iterrows():
    ...
    mats = [mat[:, nearest_index(ts, tc)]
            for mat, ts in zip(plane_traces, plane_ts)]
    ...
    run_trials.append(interp_nonan(run_t, run_v, tc))
    pup_trials.append(interp_nonan(eye_t, eye_v, tc))
```

iii. The notes say vectorized `searchsorted` and `np.interp` already cover the expensive inner work and that serial plane loading in Multiscope sessions was not worth additional complexity.

## 9-c. What processing does the code repeat multiple times?

i. The AI tries to avoid repeated full-session processing: each NWB is loaded once, the traces are cast once, and per-session quantile edges are computed once then reused. One repeated step that remains is building session-local image codes and then remapping them again to a global codebook after all sessions are processed.

ii.
```python
image_names = sorted(set(flash_img[~flash_omitted]) - {'omitted'})
img_lookup = {name: i + 1 for i, name in enumerate(image_names)}
...
global_img = {name: i + 1 for i, name in enumerate(image_names)}
for r in ok:
    remap = np.zeros(len(r['image_names']) + 1, dtype=np.int64)
    for local, name in enumerate(r['image_names']):
        remap[local + 1] = global_img[name]
    for out in r['output']:
        out[0] = remap[out[0]]
```

iii. The notes explicitly claim the code avoids repeated loading and computes quantile edges once per session; they do not separately justify the later image-code remapping beyond needing global consistency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and carries several diagnostic or bookkeeping structures that are not used in the final output pickle: `trial_info`, `timing`, `cell_specimen_ids`, and some per-session metadata fields only used for logging/notes. Optional diagnostic plotting is also extra work outside the final decoder dataset.

ii.
```python
trial_info = []
...
plane_cellid.append(np.asarray(tbl.index.values))
...
result = dict(
    ophys_session_id=int(sid),
    neural=neural_trials,
    input=inputs,
    output=outputs,
    image_names=image_names,
    brain_region_per_neuron=brain_region_per_neuron,
    mouse_id=str(md0['mouse_id']),
    cell_specimen_ids=np.concatenate(plane_cellid),
    info=info,
    trial_info=trial_info,
    timing=timing,
)
...
'metadata': {
    ...
    'session_info': [r['info'] for r in ok],
    'excluded_sessions': [{'ophys_session_id': r['ophys_session_id'],
                           'reason': r['error']} for r in bad],
}
```

iii. The notes do not give a dedicated justification for this as part of the final dataset; the extra structures were mainly used for validation, logging, and debugging.
