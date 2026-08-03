# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not scan `*.mat` files dynamically. It hard-coded the seven animal IDs, loaded each animal from the extensionless joblib file in `data/`, and then iterated through all days in that animal record.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]

for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    envs = np.array(rec['envs']).reshape(-1)
    trace = np.asarray(rec['trace'])
    position = np.asarray(rec['position'])
    blocked = rec['blocked']
```

iii. In `CONVERSION_NOTES.md`, Step 1 says the original `load_dat` utility can load "per-animal joblib or mat data," and Step 5 explicitly states, "Use native joblib files as source." The trajectory also says the agent concluded the "correct source is the joblib files in `data/`, not recomputing from MATLAB."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the hard-coded `ANIMALS` list. Each animal ID becomes one subject, and `subject_idx` is assigned from the enumeration order of that list.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
data = {
    ...
    'subjects': animals.copy(),
    'subject_idx': [],
    ...
}

for subj_idx, animal in enumerate(animals):
    ...
    data['subject_idx'].append(subj_idx)
```

iii. Step 2 of `CONVERSION_NOTES.md` says `data/` contains seven animal datasets and treats those animal files as the native subject split. Step 5 then maps those animal IDs directly to `subjects`.

## 1-c. How are the data split into sessions?

i. Each day within one animal record is treated as a separate session. The AI reads the per-day arrays in `envs`, `trace`, `position`, and `blocked`, and iterates over `day` from `0` to `n_days - 1`.

ii.
```python
envs = np.array(rec['envs']).reshape(-1)
trace = np.asarray(rec['trace'])
position = np.asarray(rec['position'])
blocked = rec['blocked']
n_days = len(envs)

for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
```

iii. Step 2 says "Sessions correspond to recording days / environments within each animal file," and Step 5 says native data have no trial structure but do have one continuous session per day.

## 1-d. How are the data split into trials?

i. Each continuous session is split into contiguous, non-overlapping 1-minute trials of `1800` frames at `30 Hz`. The AI uses the shorter of the neural and position streams, drops any trailing remainder, and slices both streams with the same boundaries.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS

def split_session_into_trials(trace_day, pos_day, blocked_vec):
    n_frames = min(trace_day.shape[1], pos_day.shape[1])
    n_trials = n_frames // FRAMES_PER_TRIAL
    used = n_trials * FRAMES_PER_TRIAL
    trace_day = trace_day[:, :used]
    pos_day = pos_day[:, :used]
    ...
    for i in range(n_trials):
        s = i * FRAMES_PER_TRIAL
        e = (i + 1) * FRAMES_PER_TRIAL
        neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
        input_trials.append(blocked_vec.copy())
        output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. Step 3 of `CONVERSION_NOTES.md` says sessions are 40 minutes at 30 Hz, and Step 5 states, "Derive 1-minute trials from continuous 40 min sessions" while preserving frame alignment.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit per-trial quality-control filter. It only keeps complete 1-minute windows and skips an entire session if it would yield fewer than two full trials.

ii.
```python
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
...
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The trajectory says this guard was added because the target format requires at least two trials per session. The notes also state that native data have no trial concept, so no paper-derived trial QC exists to reuse.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural activity from the per-day `trace` arrays inside each joblib animal record.

ii.
```python
rec = load_animal(data_dir, animal)
trace = np.asarray(rec['trace'])
...
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. Step 1 and Step 3 of `CONVERSION_NOTES.md` say the source neural signal is the rise-extracted calcium event trace, not raw fluorescence or dF/F, and Step 5 says "`trace` per animal/day" maps directly to `neural`.

## 2-b. How is the `neural` data processed?

i. The AI converts each day's trace to `float32`, removes rows deemed invalid, zero-fills any remaining non-finite values, then splits the daily matrix into trials and stores each trial as `uint8`. It does not compute dF/F, deconvolution, smoothing, or temporal rebinning.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The notes justify using the native "binarized rising-phase calcium event vector treated as firing rate." The trajectory adds a second justification: after a failed verification and a 12 GB sample file, the agent changed neural storage to `uint8` and added non-finite handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only neurons whose entire daily trace is finite. Any neuron with any `NaN` or `Inf` value in that session is dropped, and the count of dropped neurons is recorded in metadata.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
...
session_info.append({
    ...
    'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
    ...
})
```

iii. The trajectory says this was added after `train_decoder.py --verify-only` reported `NaN`/`Inf` neural values. The agent interpreted those as "unregistered cells" and decided to drop non-finite neurons on a per-session basis.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no task event. The AI aligns each trial to session start and then uses contiguous 1-minute windows within the continuous recording.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Continuous session split into contiguous 1-minute windows from session start.',
    'off_start': 0.0,
    'off_end': float(TRIAL_SECONDS),
    ...
}
```

iii. Step 5 in `CONVERSION_NOTES.md` says to "preserve native 30 Hz frame alignment" and to derive trials from continuous sessions rather than around a stimulus or behavior event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native `30 Hz` frame rate, with `time_bin_size = 1000 / 30` ms. No temporal rebinning is applied before saving.

ii.
```python
FPS = 30
...
'metadata': {
    ...
    'time_bin_size': 1000.0 / FPS,
    'fps': FPS,
    ...
}
```

iii. Step 3 and Step 5 of `CONVERSION_NOTES.md` both say the behavioral and neural streams are already aligned at 30 Hz and should be preserved framewise.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input from each day's `blocked` entry, treating that as the static geometry/block configuration for the trial.

ii.
```python
blocked = rec['blocked']
...
blocked_vec = blocked_to_vec(blocked[day])
```

iii. Step 2 says `blocked` stores per-day blocked partition indices, and Step 5 explicitly maps "`blocked` per animal/day" to the decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI flattens the nested `blocked` representation, ignores `NaN`s, interprets `-1` as "no blocked partition," and converts the remaining indices into a 9-element binary vector. That vector is then copied once per trial as a static input.

ii.
```python
def blocked_to_vec(blocked_entry):
    vec = np.zeros(9, dtype=np.float32)
    flat = np.array(blocked_entry, dtype=object).reshape(-1)
    vals = []
    for item in flat:
        arr = np.array(item).reshape(-1)
        for v in arr:
            if np.isnan(v):
                continue
            vals.append(int(v))
    if len(vals) == 1 and vals[0] == -1:
        return vec
    for v in vals:
        if 0 <= v <= 8:
            vec[v] = 1.0
    return vec
...
input_trials.append(blocked_vec.copy())
```

iii. Step 5 says the decoder input should be a "9-d binary vector over 3x3 arena partitions" and should stay constant within each trial because the blocked geometry is session-level context.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives the decoder output from the per-day `position` array in each animal record.

ii.
```python
position = np.asarray(rec['position'])
...
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. Step 2 says `position` contains the aligned 2D behavioral coordinates, and Step 5 maps `position` directly to the decoder output after discretization.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI converts position to `float32`, zero-fills any non-finite values, truncates position to the same used frame span as the neural data, discretizes x and y into a 3-by-3 grid over the 75 cm arena, flattens that to one label per frame, and stores each trial as a `(1, time)` `uint8` array.

ii.
```python
def position_to_bins_3x3(position_xy):
    pos = np.asarray(position_xy, dtype=np.float32)
    x = pos[0]
    y = pos[1]
    eps = 1e-6
    xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
    ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
    xbin = np.clip(xbin, 0, N_POS_BINS - 1)
    ybin = np.clip(ybin, 0, N_POS_BINS - 1)
    return (ybin * N_POS_BINS + xbin).astype(np.int64)

pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
...
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. Step 5 says the task requires a 3x3 discretized position output, and the trajectory says this is an intentionally coarser version of the paper's position-decoding setup.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is thresholded into three bins by dividing the 75 cm arena into equal 25 cm spans per axis. The final category is `y_bin * 3 + x_bin`, yielding labels `0` through `8`.

ii.
```python
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
xbin = np.clip(xbin, 0, N_POS_BINS - 1)
ybin = np.clip(ybin, 0, N_POS_BINS - 1)
return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

iii. The notes derive this from the task specification: output must be 3 x 3 spatial bins, and the arena size is 75 cm.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns position to neural data by forcing both streams to the same frame span with `min(trace_len, pos_len)`, truncating both to that span, and then slicing them into trials with identical indices.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
pos_bins = position_to_bins_3x3(pos_day)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. Step 5 says the conversion should preserve native frame alignment, and the trajectory reports a raw-versus-converted spot check where session 0, trial 0 matched for both neural and output arrays.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles minor data problems by dropping neurons with any non-finite value in a session, zero-filling remaining non-finite neural and position values, truncating to the shorter of the neural and position streams, discarding leftover frames that do not fill a full minute, and skipping sessions with fewer than two full trials.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
...
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
...
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The trajectory says these fixes were added after verification exposed `NaN`/`Inf` values in one sample session. Step 8 of `CONVERSION_NOTES.md` records that the initial sample failed verification and that dropping non-finite neurons resolved it.

## 6-a. What are the most time-consuming steps of the code?

i. The AI did not write a formal bottleneck analysis in `CONVERSION_NOTES.md`, but its trajectory identifies large file I/O and serialization as the main costs: loading whole-animal joblib files, splitting many large session arrays into trial arrays, and pickling very large outputs. Optional plotting adds more work in `--show-processing` mode.

ii.
```python
def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]

for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    ...

if show_processing and plotted < 2:
    pos_bins = position_to_bins_3x3(pos_day[:, :used])
    make_processing_plot(...)

with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. In the trajectory, the agent explicitly noted that the first sample output was 12 GB, called that "an inefficiency in the conversion format," and then optimized dtypes to reduce storage cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not explicitly call out vectorization opportunities. The code still uses Python loops for flattening `blocked` entries and for assembling trial lists one trial at a time.

ii.
```python
for item in flat:
    arr = np.array(item).reshape(-1)
    for v in arr:
        if np.isnan(v):
            continue
        vals.append(int(v))

for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    input_trials.append(blocked_vec.copy())
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The trajectory focused on dtype reduction and filtering fixes rather than loop refactoring, so there is no explicit justification beyond using a straightforward implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats a few small pieces of work: it flattens environment labels both for session IDs and metadata, copies the same blocked vector once per trial, and recomputes position bins in `--show-processing` mode even though they were already computed during trial splitting.

ii.
```python
session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
...
'environment': flatten_env_name(envs[day]),
...
input_trials.append(blocked_vec.copy())
...
pos_bins = position_to_bins_3x3(pos_day)
...
if show_processing and plotted < 2:
    pos_bins = position_to_bins_3x3(pos_day[:, :used])
    make_processing_plot(...)
```

iii. This repetition was not documented explicitly. It appears to come from the agent adding bookkeeping metadata and optional visualization support around the core conversion path.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional plotting path, the plotting occupancy histogram, and the extra `session_info` bookkeeping are not used by downstream decoder training. Environment strings, runtime, and blocked-vector summaries are also saved only as metadata, not as model inputs or outputs.

ii.
```python
def make_processing_plot(session_id, pos_day, pos_bins, blocked_vec, neural_trial0, out_path):
    ...
    h = np.bincount(pos_bins, minlength=9).reshape(3, 3)
    im = ax[0, 1].imshow(h, origin='lower')
    ...

session_info.append({
    'session_id': session_id,
    'animal': animal,
    'day_index': int(day),
    'environment': flatten_env_name(envs[day]),
    'n_neurons': int(trace_day.shape[0]),
    'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
    'n_frames_raw': int(n_frames),
    'n_frames_used': int(used),
    'n_trials': int(n_trials),
    'blocked_vector': blocked_vec.astype(int).tolist(),
})

data['metadata']['session_info'] = session_info
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The prompt required `--show-processing` plots and encouraged documentation and sanity checks, so these extras were added for visibility rather than for downstream analysis.
