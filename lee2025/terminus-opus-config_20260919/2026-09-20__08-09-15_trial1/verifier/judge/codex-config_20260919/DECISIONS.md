# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes seven animal IDs, loads each extensionless joblib file once, indexes the top-level dictionary by animal ID, and obtains all session-wise traces, positions, environment labels, and blocked-partition entries. Full mode uses every animal and every day; sample mode uses two days from the first animal.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
traces = dat['trace']
positions = dat['position']
envs = np.array(dat['envs']).ravel()
blocked = dat['blocked']
```

iii. The notes say this reproduces the repository's `load_dat(..., format="joblib")` path and avoids loading an animal repeatedly. The full run accounted for all 7 mice and 207 sessions.

## 1-b. How are the data split into subjects?

i. Each named joblib file is one mouse. The animal's position in `ANIMALS` is stored as the session's `subject_idx`.

ii.
```python
for ai, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
data['subject_idx'].append(ai)
```

iii. The agent relied on the reference repository's seven-animal list and verified the expected 31/31/31/21/31/31/31 sessions per mouse.

## 1-c. How are the data split into sessions?

i. Each recording day is one output session. The code iterates over the first axis of `position`, takes the corresponding trace, position, blocked geometry, and environment, and appends the processed day if it has at least two retained trials.

ii.
```python
ndays = positions.shape[0]
for d in range(ndays):
    trace_day = np.asarray(traces[d])
    position_day = np.asarray(positions[d])
    neural_trials, input_trials, output_trials, stats, aux = process_session(
        trace_day, position_day, blocked[d])
```

iii. A day is the natural recording-session unit in the released data. The agent reports that all 207 days passed the two-trial requirement.

## 1-d. How are the data split into trials?

i. After converting the session to 100 ms bins, the agent makes consecutive, non-overlapping 600-bin windows corresponding to one minute of original session time. It drops the trailing incomplete window; after defining a window it removes its nonmoving bins, so retained trials have variable numbers of samples.

ii.
```python
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TEMPORAL_BIN_FRAMES))
n_full_trials = nbins // TRIAL_BINS
for t in range(n_full_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    m = moving_bins[sl]
    neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
```

iii. The task explicitly requires one-minute trials. The notes justify dropping the remainder so all windows span the same amount of real time.

## 1-e. How are trials filtered based on quality controls?

i. Windows with fewer than 30 retained moving bins (less than 3 seconds at 100 ms/bin) are dropped. Any session left with fewer than two trials is skipped.

ii.
```python
if m.sum() < MIN_TRIAL_BINS:
    continue
...
if len(neural_trials) < 2:
    continue
```

iii. The agent calls such windows too short or unreliable and notes that every retained session still had at least 22 trials. This minimum is agent-chosen rather than specified by the task or human conversion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace` array in the per-animal joblib object. These are released binary rising-phase calcium-event traces, with NaN rows for cells not registered on a day.

ii.
```python
traces = dat['trace']
...
trace_day = np.asarray(traces[d])
```

iii. The paper describes the binary rising-phase vector as the firing rate, so the agent uses this released trace rather than recomputing fluorescence or deconvolution.

## 2-b. How is the `neural` data processed?

i. The code selects retained cells, smooths each continuous trace with a Gaussian of sigma 3 frames, averages non-overlapping groups of 3 frames, casts to float32, multiplies by 30 to express event rate in Hz, and finally retains only moving bins in each trial.

ii.
```python
tr = trace_day[keep_cells].astype(np.float32)
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SIGMA_FRAMES, axis=1)
neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS
...
neural_binned[:, sl][:, m]
```

iii. The agent took smoothing and 3-frame pooling from `fit_decoder`, arguing that smoothing before deleting immobile samples avoids mixing temporally nonadjacent frames. Multiplication by 30 was intended to give interpretable Hz units.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell must be registered on that day (not an all-NaN row) and have more than five events during frames whose smoothed speed exceeds 5 cm/s. The code also asserts that no cell is only partially NaN.

ii.
```python
nan_any = np.isnan(trace_day).any(axis=1)
nan_all = np.isnan(trace_day).all(axis=1)
assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
registered = ~nan_all
events_moving = np.nansum(trace_day[:, moving], axis=1)
keep_cells = registered & (events_moving > CELL_THRESHOLD)
```

iii. The notes identify this as the reference `decode_position_within` rule (`cell_threshold=5`) and report that it removes 1.3% of registered cell-sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Alignment is to the start of each artificial one-minute window in the continuous recording; neural and behavior share frame indices, and identical pooled-bin and moving-bin selections are applied.

ii.
```python
sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
m = moving_bins[sl]
neural_trials.append(neural_binned[:, sl][:, m])
output_trials.append(out_class[sl][m][np.newaxis, :])
```

iii. The task is continuous free foraging, so the agent correctly states that no stimulus or behavioral event exists for alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms (10 Hz). The native 30 Hz data are average-pooled in non-overlapping groups of three frames after neural smoothing.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
...
return x.reshape(*x.shape[:-1], nbins, nframes_per_bin).mean(axis=-1)
```

iii. The agent selected the `temporal_bin_size=3` used by the paper repository's decoder functions. The human conversion instead preserves the native 33.33 ms samples.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. It is derived from the per-session `blocked` field, rather than inferred only from the string environment name.

ii.
```python
blocked = dat['blocked']
...
geometry = blocked_to_vector(blocked_day)
```

iii. The agent found that some geometries were presented flipped, so the session-specific blocked indices are more reliable than `get_env_mat(env)`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices become a length-9 float32 multi-hot vector, where 1 means blocked. A negative sentinel produces all zeros. A copy of the session-static vector is stored for each trial.

ii.
```python
vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
if b.size and b[0] >= 0:
    vec[b.astype(int)] = 1.0
...
input_trials.append(geometry.copy())
```

iii. This directly represents the requested static environment geometry and was validated against occupancy and the reference geometry definitions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output comes from each session's two-row `position` array containing x and y coordinates in centimeters.

ii.
```python
positions = dat['position']
...
position_day = np.asarray(positions[d])
```

iii. The position and imaging streams are both recorded at 30 Hz and synchronized in the released arrays.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code averages x and y over each 3-frame temporal bin, maps the mean coordinate to a 3-by-3 class, and deletes the same low-speed bins deleted from neural data.

ii.
```python
pos_binned = bin_frames(position_day)
out_class = position_to_class(pos_binned)
...
output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
```

iii. The agent used the same temporal bins as neural activity to preserve alignment and changed the reference decoder's 15-by-15 spatial grid to the task-required 3-by-3 grid.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is floor-divided by 25 cm, clipped to 0 through 2, then encoded as `3*ybin + xbin`, yielding classes 0 through 8.

ii.
```python
bin_down = (ARENA_SIZE + BUFFER) / N_SPATIAL_BINS
xy = np.floor(position_binned_xy / bin_down).astype(int)
xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. The 25 cm thresholds coincide with the physical partition walls. Clipping handles coordinates exactly at 75 cm.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and traces begin frame-aligned. Both are pooled into the same non-overlapping 3-frame bins, sliced by the same one-minute boundaries, and indexed by the identical `moving_bins` mask.

ii.
```python
neural_binned = bin_frames(tr_smooth)
pos_binned = bin_frames(position_day)
...
neural_binned[:, sl][:, m]
out_class[sl][m]
```

iii. The agent checked equal raw frame counts and exact per-trial neural/output lengths, and its plots showed no apparent lag.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Entirely NaN cell rows are treated as unregistered and removed; partially NaN rows trigger an assertion. The first undefined speed sample is set to zero. Temporal remainders are dropped, coordinates are clipped at valid class bounds, `blocked=-1` becomes no blocked partitions, short trials are removed, and sessions with fewer than two trials would be skipped.

ii.
```python
assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
registered = ~nan_all
speed = np.zeros(position_xy.shape[1], dtype=np.float64)
...
x = x[..., :nbins * nframes_per_bin]
xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
```

iii. The notes say these cases were explicitly checked: no partial-NaN cells occurred, all sessions survived, and clipping prevents the 75 cm boundary from creating an invalid class. The agent deliberately left rare apparent occupancy in blocked partitions unchanged as tracking jitter rather than filtering on the label.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the large per-animal joblib objects takes roughly 9–16 seconds each. Across all sessions, Gaussian smoothing/binning and session processing also contribute materially (reported as about 0.62 seconds per session); serialization of the 3.41 GB pickle is another substantial operation, though it is not separately timed.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
t_load = time.time() - t0
...
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SIGMA_FRAMES, axis=1)
...
pickle.dump(data, f, protocol=4)
```

iii. The notes characterize loading as 9–16 seconds per animal and processing as 0.3–1 second per session, estimating roughly 90 seconds of loading and two minutes of processing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The expensive frame-wise operations are already vectorized. The remaining loop over one-minute trials could be partially vectorized by reshaping full bins into `(n_trials, 600)`, although variable moving masks and conditional trial rejection still require per-trial construction. Animal/session loops are appropriate because dimensions vary.

ii.
```python
for t in range(n_full_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    m = moving_bins[sl]
    if m.sum() < MIN_TRIAL_BINS:
        continue
```

iii. The agent explicitly avoided the reference rate-map style of per-frame Python loops and used reshape/mean operations, claiming an approximately 100-fold improvement for binning/discretization.

## 6-c. What processing does the code repeat multiple times?

i. It copies the same static geometry for every trial. During final checks it repeatedly concatenates each session's output arrays to find classes and blocked-class occupancy, and later concatenates all outputs again for the global distribution. Neural means are also calculated once per trial before averaging.

ii.
```python
input_trials.append(geometry.copy())
...
classes = np.unique(np.concatenate([o.ravel() for o in data['output'][si_i]]))
...
allout = np.concatenate([o.ravel() for s in data['output'] for o in s])
allneural_mean = np.mean([tr.mean() for s in data['neural'] for tr in s])
```

iii. The agent does not flag these as problems; the repeated geometry is required by the target trial-wise structure, while the repeated concatenations are validation work and small relative to neural processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `process_session` always builds and returns a large `aux` dictionary (including full binned neural data, positions, speeds, and masks), but full conversion discards it unless plotting is requested. The code also loads `envs` only for metadata/logging, and performs several full-dataset sanity-statistic passes that do not affect saved arrays.

ii.
```python
aux = dict(speed=speed, moving=moving, speed_binned=speed_binned,
           moving_bins=moving_bins, pos_binned=pos_binned, out_class=out_class,
           neural_binned=neural_binned, keep_cells=keep_cells, registered=registered)
return neural_trials, input_trials, output_trials, stats, aux
...
neural_trials, input_trials, output_trials, stats, aux = process_session(...)
```

iii. `aux` exists to support optional diagnostic plots, and the sanity passes were used to validate conversion. They are defensible during development but could be conditional or omitted in production to reduce peak memory and runtime.
