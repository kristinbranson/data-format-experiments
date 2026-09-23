# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads per-subject joblib files from `/app/data` using a hard-coded list of 7 animal IDs. For each animal file, it reads the per-day `trace`, `position`, `envs`, and `blocked` entries, then processes each day/session into trials inside `process_session(...)`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for ai, animal in enumerate(animals):
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    t_load = time.time() - t0
    traces = dat['trace']
    positions = dat['position']
    envs = np.array(dat['envs']).ravel()
    blocked = dat['blocked']
    ndays = positions.shape[0]
```

iii. In `CONVERSION_NOTES.md`, the AI says the joblib and `.mat` versions contain the same content, and that it chose the joblib files because they are “the repo default” and faster to load.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is treated as one subject. The `subjects` field is initialized from the same `ANIMALS` list, and `subject_idx` stores the animal index for each processed session.

ii.
```python
data = {
    'neural': [], 'input': [], 'output': [],
    'subjects': ANIMALS if not sample else ANIMALS[:1],
    'subject_idx': [],
    ...
}
...
for ai, animal in enumerate(animals):
    ...
    data['subject_idx'].append(ai)
```

iii. In the notes, the AI states that `/app/data` contains one joblib file per mouse and lists the 7 mouse IDs explicitly, so it uses those IDs directly as subject identities.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject file is treated as one session. The code iterates over the first axis of `position`/`trace`, and each day contributes one session to the output if at least two trials remain after processing.

ii.
```python
ndays = positions.shape[0]
...
for d in range(ndays):
    trace_day = np.asarray(traces[d])
    position_day = np.asarray(positions[d])
    neural_trials, input_trials, output_trials, stats, aux = process_session(
        trace_day, position_day, blocked[d])

    if len(neural_trials) < 2:
        print(f'  WARNING session {animal} day {d} has < 2 trials, skipped')
        continue

    data['neural'].append(neural_trials)
    data['input'].append(input_trials)
    data['output'].append(output_trials)
```

iii. `CONVERSION_NOTES.md` says “Session = one recording day” and notes that this matches the reference code’s per-day analyses.

## 1-d. How are the data split into trials?

i. The AI first rebins the continuous session to 100 ms bins, then divides the rebinned session into consecutive non-overlapping 1-minute windows of 600 bins each. After that, each trial keeps only bins whose speed exceeds the movement threshold, so trial lengths become variable.

ii.
```python
TRIAL_SECONDS = 60.0
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TEMPORAL_BIN_FRAMES))   # 600 bins
...
neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS
pos_binned = bin_frames(position_day)
speed_binned = bin_frames(speed[np.newaxis, :])[0]
...
n_full_trials = nbins // TRIAL_BINS
for t in range(n_full_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    m = moving_bins[sl]
    if m.sum() < MIN_TRIAL_BINS:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
    output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
    input_trials.append(geometry.copy())
```

iii. The notes justify this as a combination of the task requirement (“1-minute trials”) plus the AI’s choice to follow the paper’s decoder preprocessing and exclude immobile time bins.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by movement: within each 1-minute window, only bins with speed above threshold are kept. If fewer than 30 moving bins remain in a 1-minute window, that trial is dropped. Entire sessions are skipped if fewer than two trials survive.

ii.
```python
MIN_TRIAL_BINS = 30
...
moving_bins = speed_binned > V_THRESH
...
for t in range(n_full_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    m = moving_bins[sl]
    if m.sum() < MIN_TRIAL_BINS:
        continue
    ...
...
if len(neural_trials) < 2:
    print(f'  WARNING session {animal} day {d} has < 2 trials, skipped')
    continue
```

iii. In the notes, the AI says this was motivated by the reference position-decoding analysis, which excludes immobility (`v_thresh=5 cm/s`), plus a decoder-specific requirement to keep sessions with at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-day `trace` array loaded from each animal file.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
traces = dat['trace']
...
trace_day = np.asarray(traces[d])
```

iii. The notes say the released `trace` data are already “rise-extracted calcium traces, binary” and are the neural signal used in the paper’s analyses.

## 2-b. How is the `neural` data processed?

i. The AI keeps selected cells, smooths each cell’s continuous trace with `gaussian_filter1d(..., sigma=3)` along time, average-pools non-overlapping 3-frame windows, and multiplies by 30 so the resulting values are event rates in Hz. Trials then inherit only the bins passing the movement filter.

ii.
```python
TRACE_SIGMA_FRAMES = 3
TEMPORAL_BIN_FRAMES = 3
...
tr = trace_day[keep_cells].astype(np.float32)
...
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SIGMA_FRAMES, axis=1)
neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS  # event rate in Hz
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
```

iii. In the notes, the AI explicitly justifies this as matching the paper’s decoder preprocessing (`fit_decoder` / `decode_position_within`), except that it smooths before removing immobile bins because it considers that “strictly more correct.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells that are registered on that day (not all-NaN) and that have more than 5 events during moving frames.

ii.
```python
nan_any = np.isnan(trace_day).any(axis=1)
nan_all = np.isnan(trace_day).all(axis=1)
assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
registered = ~nan_all
...
speed = compute_speed(position_day)
moving = speed > V_THRESH
...
events_moving = np.nansum(trace_day[:, moving], axis=1)
keep_cells = registered & (events_moving > CELL_THRESHOLD)
```

iii. `CONVERSION_NOTES.md` says this follows the reference within-session decoder’s curation rule: drop cells not registered that day and cells with `<= 5` events during moving periods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not use an experimental event. Instead, it defines the start of each artificial 1-minute window as the alignment event, and stores `off_start=0`, `off_end=60` in metadata.

ii.
```python
'metadata': {
    'task_description': (...),
    'time_bin_size': TIME_BIN_MS,
    'temporal_alignment_event': (
        'Start of each 1-minute trial window, measured from the start of the '
        'continuous 40-min recording session (there are no experimenter-defined '
        'trials in this free-foraging task).'),
    'off_start': 0.0,
    'off_end': TRIAL_SECONDS,
    ...
}
```

iii. The notes justify this by stating that the task has no experimenter-defined trials, so the 1-minute decoder windows are artificial segments of a continuous session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. The AI rebins from the original 30 Hz stream by smoothing and average-pooling 3 frames into one time bin.

ii.
```python
FPS = 30.0
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100 ms
...
def bin_frames(x, nframes_per_bin=TEMPORAL_BIN_FRAMES):
    T = x.shape[-1]
    nbins = T // nframes_per_bin
    x = x[..., :nbins * nframes_per_bin]
    return x.reshape(*x.shape[:-1], nbins, nframes_per_bin).mean(axis=-1)
```

iii. The notes say this is meant to reproduce the reference decoder’s `temporal_bin_size=3` setting exactly.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment-geometry input is derived from each session’s `blocked` entry.

ii.
```python
blocked = dat['blocked']
...
neural_trials, input_trials, output_trials, stats, aux = process_session(
    trace_day, position_day, blocked[d])
```

iii. The notes say the AI chose `blocked` rather than `get_env_mat(env)` because `blocked` is the per-session ground truth and captures vertically flipped environment variants.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The `blocked` entry is converted into a 9-element binary vector with `1` for blocked partitions and `0` for open ones. The same vector is copied into every trial from that session.

ii.
```python
def blocked_to_vector(blocked_day):
    b = np.atleast_1d(np.asarray(blocked_day[0]).ravel()).astype(float)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if b.size and b[0] >= 0:
        vec[b.astype(int)] = 1.0
    return vec
...
geometry = blocked_to_vector(blocked_day)
...
input_trials.append(geometry.copy())
```

iii. The AI’s notes justify this as the correct per-session geometry encoding for the decoder task, and specifically note that it verified the blocked-index convention against occupancy.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from each session’s `position` array.

ii.
```python
positions = dat['position']
...
position_day = np.asarray(positions[d])
```

iii. The notes describe `position` as synchronized 2D `x,y` position in centimeters and treat it as the source for decoder outputs.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first average-pools position into 100 ms bins, then converts each binned `(x, y)` sample into a 3x3 spatial-bin class. Later, it keeps only bins that pass the movement filter and slices them into 1-minute trial windows.

ii.
```python
pos_binned = bin_frames(position_day)                            # (2, nbins), cm
speed_binned = bin_frames(speed[np.newaxis, :])[0]               # (nbins,), cm/s
moving_bins = speed_binned > V_THRESH
out_class = position_to_class(pos_binned)
...
output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
```

iii. In the notes, the AI says this mirrors the paper’s decoder pipeline, except that it reduces the spatial grid from the paper’s 15x15 to the task-required 3x3.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI divides the 75 cm arena into 3 bins per axis (25 cm each), clips indices to `0..2`, and encodes the categorical output as `3 * ybin + xbin`, yielding 9 classes.

ii.
```python
def position_to_class(position_binned_xy):
    bin_down = (ARENA_SIZE + BUFFER) / N_SPATIAL_BINS
    xy = np.floor(position_binned_xy / bin_down).astype(int)
    xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. The notes say this uses the same spatial-binning formula as the reference decoder, with `n_bins=3` instead of `15`, and that the 25 cm edges match the physical arena partitions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns position and neural activity by applying the same 100 ms temporal binning, the same per-bin movement mask, and the same trial-window slices to both streams.

ii.
```python
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SIGMA_FRAMES, axis=1)
neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS
pos_binned = bin_frames(position_day)
speed_binned = bin_frames(speed[np.newaxis, :])[0]
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
```

iii. The notes repeatedly justify this by saying the paper’s `position` and `trace` streams are already frame-aligned at 30 Hz, so identical rebinning and slicing preserve alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI assumes cells are either fully present or fully absent on a session and asserts there are no partially-NaN cells. It removes absent cells, drops leftover bins that do not make a full 3-frame bin or full 1-minute trial, clips position bin indices to stay within `0..2`, and leaves very small blocked-partition occupancy caused by tracking jitter in place while only warning about it.

ii.
```python
nan_any = np.isnan(trace_day).any(axis=1)
nan_all = np.isnan(trace_day).all(axis=1)
assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
...
nbins = T // nframes_per_bin
x = x[..., :nbins * nframes_per_bin]
...
n_full_trials = nbins // TRIAL_BINS
...
xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
...
if bad:
    print(f'  WARNING session {si_i}: output classes {bad} are blocked partitions ...')
```

iii. The notes explain these choices as sanity-check-driven handling: partial-NaN cells were not expected, exact `75.0` cm positions needed clipping, trailing partial windows were dropped for uniform trial duration, and blocked-bin leakage was attributed to DeepLabCut jitter and intentionally not corrected.

## 6-a. What are the most time-consuming steps of the code?

i. The AI treats file loading as the dominant cost, with per-animal joblib loads taking roughly 9 to 16 seconds and per-session processing taking roughly 0.3 to 1 second.

ii.
```python
for ai, animal in enumerate(animals):
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    t_load = time.time() - t0
    ...
    print(f'{animal}: loaded in {t_load:.1f} s, {ndays} sessions, '
          f'{traces.shape[1]} registered-or-not cells', flush=True)
...
per_session_times.append(time.time() - ts)
...
print(f'mean processing time per session: {np.mean(per_session_times):.2f} s')
```

iii. In the notes, the AI explicitly says “Each animal file is loaded exactly once (9-16 s each); the per-session processing is 0.3-1 s.”

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s main design decision was to vectorize frame binning and discretization instead of using per-frame Python loops. The remaining explicit Python loops are over sessions and 1-minute trial windows.

ii.
```python
def bin_frames(x, nframes_per_bin=TEMPORAL_BIN_FRAMES):
    T = x.shape[-1]
    nbins = T // nframes_per_bin
    x = x[..., :nbins * nframes_per_bin]
    return x.reshape(*x.shape[:-1], nbins, nframes_per_bin).mean(axis=-1)
...
for t in range(n_full_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    m = moving_bins[sl]
    ...
```

iii. The notes explicitly say that naive per-frame loops like those in some reference functions would be “far too slow” and that all binning/discretization was vectorized with NumPy.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat the main data load or session conversion unnecessarily, but it does repeat some summary computations after conversion, especially repeated concatenation of per-session outputs during blocked-partition warnings and final class-distribution checks.

ii.
```python
for si_i, sess in enumerate(data['neural']):
    ...
    classes = np.unique(np.concatenate([o.ravel() for o in data['output'][si_i]]))
    bad = [c for c in classes if geom[c] > 0]
    if bad:
        print(f'  WARNING session {si_i}: output classes {bad} are blocked partitions '
              f'(occupancy {[float(np.mean(np.concatenate([o.ravel() for o in data["output"][si_i]]) == c)) for c in bad]})')
allout = np.concatenate([o.ravel() for s in data['output'] for o in s])
```

iii. The AI does not call out this repetition directly in the notes. The notes instead emphasize that each animal is loaded only once, so the repeated work is limited to post-conversion sanity checks rather than the main conversion path.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and returns auxiliary arrays (`speed`, `moving`, `speed_binned`, `pos_binned`, `out_class`, `registered`, `keep_cells`) for every session even though they are only needed for optional plotting/debugging. It also performs extensive summary/sanity-check printing that is not used by downstream decoding.

ii.
```python
aux = dict(speed=speed, moving=moving, speed_binned=speed_binned,
           moving_bins=moving_bins, pos_binned=pos_binned, out_class=out_class,
           neural_binned=neural_binned, keep_cells=keep_cells, registered=registered)
return neural_trials, input_trials, output_trials, stats, aux
...
if args.show_processing and nplotted < 2:
    plot_processing(f'{animal}_day{d}', position_day, trace_day, aux, stats)
...
print('output class distribution:',
      np.round(np.bincount(allout, minlength=9) / allout.size, 4))
print(f'mean neural value (Hz): {allneural_mean:.4f}')
```

iii. The AI’s notes justify these extras as validation aids: it says the processing plots and sanity checks were added to “convince the reader” and to verify that loading, alignment, and binning matched the paper.
