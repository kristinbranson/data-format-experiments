# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not scan `data/*.mat`. It hard-coded the seven animal IDs, loaded one joblib file per animal from `data/`, then iterated through every recording day in each animal object. Trials were created later by splitting each day/session into contiguous 1-minute windows.

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

iii. In `CONVERSION_NOTES.md`, the AI explicitly decided to "Use native joblib files as source" because it believed the reference repository's `load_dat` path directly consumed joblib files and because both joblib and MATLAB formats were present. Its later notes also say the conversion uses native joblib `trace`, `position`, and `blocked` fields.

## 1-b. How are the data split into subjects?

i. Subjects are the hard-coded entries in `ANIMALS`. Each joblib file is treated as one mouse, and `subjects` is initialized from that list directly.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

data = {
    ...
    'subjects': animals.copy(),
    'subject_idx': [],
    ...
}
```

iii. The notes state that `data/` contains 7 animal datasets and that per-animal files are the natural organizational unit. The AI therefore treated file identity as subject identity rather than discovering subjects dynamically from filenames.

## 1-c. How are the data split into sessions?

i. Each element of an animal's per-day arrays is treated as one session. The AI uses `envs` to determine `n_days`, then processes `trace[day]`, `position[day]`, and `blocked[day]` together as one recording session.

ii.
```python
envs = np.array(rec['envs']).reshape(-1)
trace = np.asarray(rec['trace'])
position = np.asarray(rec['position'])
blocked = rec['blocked']
n_days = len(envs)

for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
    blocked_vec = blocked_to_vec(blocked[day])
    trace_day = np.asarray(trace[day], dtype=np.float32)
    pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The notes describe sessions as "recording days / environments within each animal file" and state that native data have one per-day entry for `envs`, `position`, and `trace`. That is the AI's stated rationale for session boundaries.

## 1-d. How are the data split into trials?

i. Trials are artificial, non-overlapping 60-second chunks. The AI computes `FRAMES_PER_TRIAL = 30 * 60 = 1800`, trims the session to an integer number of full trials, and appends one neural/input/output trial triple per chunk.

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

iii. The notes say native data have no trial concept and that the task specification requires 1-minute trials. They also state that a 40-minute session should become about 40 contiguous trials while preserving alignment.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality-control filter. The only related rule is session-level: if a session yields fewer than two full 1-minute trials, the whole session is skipped.

ii.
```python
neural_trials, input_trials, output_trials, n_frames, used, n_trials = split_session_into_trials(
    trace_day, pos_day, blocked_vec
)
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The justification is implicit in the target-format requirement from the instructions: each session must have at least two trials for decoder evaluation. The AI's notes repeatedly mention the "at least two trials within each session" requirement and do not describe any additional trial QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived directly from the per-day `trace` arrays in the animal joblib files.

ii.
```python
trace = np.asarray(rec['trace'])
...
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. The notes say "`trace` per animal/day -> neural" and describe `trace` as the native binarized rising-phase calcium event signal used by the original analysis code.

## 2-b. How is the `neural` data processed?

i. The AI treats `trace` as already-processed event activity. It converts each day's trace to `float32`, removes non-finite neurons, zero-fills any remaining non-finite values, splits into 1-minute trials, and stores each trial as `uint8`. It does not perform additional deconvolution, smoothing, or temporal rebinning.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))

'source_signal': 'Binarized rising-phase calcium event vector treated as firing rate',
```

iii. The notes explicitly justify using `trace` because the methods/code treat the binarized rising-phase vector as the firing-rate-like signal for downstream analyses. The trajectory shows that after a verification failure, the AI additionally switched neural storage to `uint8` to shrink very large output files.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI drops any neuron whose per-day trace contains any non-finite value anywhere along the time axis. It records how many such neurons were dropped for each session.

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

iii. The trajectory explains this as a repair after `train_decoder.py --verify-only` reported NaN/Inf neural values. The notes then frame the choice as handling "unregistered neurons with non-finite values" by dropping them before trial splitting.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. Trials are aligned to session start and then cut into contiguous 1-minute windows from that origin.

ii.
```python
'temporal_alignment_event': 'Continuous session split into contiguous 1-minute windows from session start.',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The notes explicitly state that native data are continuous 40-minute recordings with no trial event, so the AI preserved framewise alignment and used session start as the practical alignment reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native 30 Hz frame rate, so one time bin is `1000 / 30` ms. No rebinning or downsampling is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The notes repeatedly say to "Preserve native 30 Hz frame alignment" and describe both imaging and behavior as already aligned at 30 Hz. The AI therefore kept the original temporal resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from the raw `blocked` field, not from a full geometry mask. The AI interprets the blocked partition indices as the task-relevant geometry descriptor.

ii.
```python
blocked = rec['blocked']
...
blocked_vec = blocked_to_vec(blocked[day])
```

iii. In the notes, the AI explicitly decided to "Use blocked geometry as decoder input" because this information is directly available from `blocked` and matches the decoder-task specification better than reproducing the full geometric mask logic from the paper.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked` is flattened from its nested object-array representation, NaNs are ignored, `-1` is treated as "no blocked partition," and the remaining indices are one-hot encoded into a 9-element binary vector. That vector is copied once per trial and is constant within a session.

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

iii. The notes say "`blocked` per animal/day -> input[0:9]" and explicitly justify it as a static 9-dimensional binary encoding of blocked partitions, with `-1` meaning all zeros.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The decoder output is derived from the raw `position` arrays for each day.

ii.
```python
position = np.asarray(rec['position'])
...
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The notes map "`position` per animal/day -> output[0]" and describe it as the frame-aligned 2D animal position signal.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI clips `x` and `y` into the 75 cm arena, divides each axis into three equal 25 cm bins, computes integer `xbin` and `ybin`, then converts each frame into a single category `ybin * 3 + xbin`.

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
```

iii. The notes justify this as the required task output: a coarser 3x3 version of the paper's position decoding over the 75x75 cm arena.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholding is done by fixed spatial bin edges implied by the 3x3 equal-width partitioning of the 75 cm arena. The resulting categories are integers 0 through 8.

ii.
```python
ARENA_SIZE_CM = 75.0
N_POS_BINS = 3
...
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
...
return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

iii. The notes state that the AI chose 3x3 discretized position because the decoder task explicitly requires 9 spatial bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI assumes the raw `trace` and `position` streams are already frame-aligned, trims both to the same minimum frame count, and slices them with identical trial boundaries.

ii.
```python
def split_session_into_trials(trace_day, pos_day, blocked_vec):
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

iii. The notes explicitly say that position and traces are already aligned framewise at 30 Hz and that splitting into 1-minute trials should preserve that alignment exactly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI applies several defensive fixes: it drops neurons with non-finite values, zero-fills remaining NaN/Inf values in `trace` and `position`, clips out-of-range positions into the arena bounds, truncates trace and position to their shared minimum length, discards leftover frames that do not fill a full trial, and skips sessions with fewer than two trials.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)

n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL

xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
...
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The trajectory shows that these choices were driven by verification failures caused by non-finite neural values and by a pragmatic desire to make the converted dataset pass the decoder checks. The notes later summarize this as edge-case handling for "unregistered neurons with non-finite values."

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are I/O-heavy: loading large per-animal joblib files, materializing large session/trial lists in memory, and writing the very large pickle output. Optional plotting adds extra overhead when enabled.

ii.
```python
def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]
...
for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    ...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The trajectory repeatedly discusses slow loading of large joblib files, a sample output that initially ballooned to roughly 12 GB, and a final full pickle of about 4.7 GB after dtype optimization. That is the AI's clearest performance rationale.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious candidates are the per-trial append loop in `split_session_into_trials` and the nested object-array parsing loop in `blocked_to_vec`. Trial splitting could have been handled by reshaping/slicing in bulk, and blocked parsing could have been flattened with more direct array operations.

ii.
```python
for item in flat:
    arr = np.array(item).reshape(-1)
    for v in arr:
        if np.isnan(v):
            continue
        vals.append(int(v))
...
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    input_trials.append(blocked_vec.copy())
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The AI did not explicitly justify leaving these loops unvectorized. Its efficiency work focused on reducing file size through dtype changes, not on algorithmic refactoring.

## 6-c. What processing does the code repeat multiple times?

i. The code recomputes or repeats several small transformations: `flatten_env_name` is called twice per session, position bins are computed once for trial generation and again for optional plotting, `astype` is applied at session level and again per trial, and the final assertions walk through every session/trial after the data are already built.

ii.
```python
session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
...
'environment': flatten_env_name(envs[day]),
...
pos_bins = position_to_bins_3x3(pos_day)
...
if show_processing and plotted < 2:
    pos_bins = position_to_bins_3x3(pos_day[:, :used])
...
trace_day = np.asarray(trace[day], dtype=np.float32)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. These repeated computations are visible in the code itself; the trajectory does not show the AI revisiting them. The only repeated processing it explicitly cared about was rerunning conversion/verification after the NaN fix.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several steps produce artifacts that are not used by downstream decoder training: optional plotting, environment-name flattening for `session_id`/metadata, detailed `session_info` bookkeeping, and runtime bookkeeping. Even within the saved dataset, the decoder does not use most of that metadata.

ii.
```python
def make_processing_plot(...):
    ...

session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
...
session_info.append({
    'session_id': session_id,
    'animal': animal,
    'day_index': int(day),
    'environment': flatten_env_name(envs[day]),
    ...
})
...
data['metadata']['session_info'] = session_info
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The trajectory shows that the AI created plots for inspection and added metadata for documentation and sanity checking. Those additions are useful for auditing but are not consumed by the decoder itself.
