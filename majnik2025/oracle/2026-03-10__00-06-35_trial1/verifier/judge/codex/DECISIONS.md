# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The code scans the data root for subject directories whose names start with `jm`, then scans each subject directory for session subdirectories. For each session it loads calcium traces from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`, and motion energy from `move_deve/motion_energy_glob.npy` plus `move_deve/interframe_int.npy`. Trials are not loaded directly from disk; they are created later by splitting each continuous session into fixed-size chunks.

ii. ```python
def get_subjects(base_path):
    return sorted(
        d.name for d in os.scandir(base_path)
        if d.is_dir() and d.name.startswith('jm')
    )

def get_sessions(base_path, subject):
    subject_dir = os.path.join(base_path, subject)
    sessions = [d.path for d in os.scandir(subject_dir) if d.is_dir()]
    sessions.sort()
    return sessions

F = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(session_path, 'move_deve', 'motion_energy_glob.npy'))
dt = np.load(os.path.join(session_path, 'move_deve', 'interframe_int.npy'))
```

iii. `CONVERSION_NOTES.md` and `README.md` do not contain substantive reasoning. The justification is implicit in the code structure and comments: the agent assumed the directory naming convention (`jm*` subjects, subdirectories as sessions) defines the dataset hierarchy.

## 1-b. How are the data split into subjects?

i. Subjects are defined as sorted top-level directories under the data root whose names begin with `jm`.

ii. ```python
def get_subjects(base_path):
    return sorted(
        d.name for d in os.scandir(base_path)
        if d.is_dir() and d.name.startswith('jm')
    )
```

iii. No separate written rationale was provided beyond the code. The decision appears to rely on the dataset’s folder naming convention.

## 1-c. How are the data split into sessions?

i. Sessions are defined as sorted subdirectories inside each subject directory. Each such directory is treated as one recording session.

ii. ```python
def get_sessions(base_path, subject):
    subject_dir = os.path.join(base_path, subject)
    sessions = [d.path for d in os.scandir(subject_dir) if d.is_dir()]
    sessions.sort()
    return sessions
```

iii. No detailed justification is documented. The code and comments indicate the agent treated each subject subdirectory as one daily recording.

## 1-d. How are the data split into trials?

i. Trials are artificial 60-second, non-overlapping chunks of each continuous session. At 30 Hz this is `1800` frames per trial. Any remainder at the end of a session that does not fill a full chunk is discarded.

ii. ```python
TRIAL_DUR = 60
FS = 30

trial_frames = TRIAL_DUR * FS
n_trials = n_frames // trial_frames
remainder = n_frames - n_trials * trial_frames

for ti in range(n_trials):
    s = ti * trial_frames
    e = s + trial_frames
    neural_trials.append(Fc[:, s:e])
    inp_trials.append(t[np.newaxis, :])
    output_trials.append(out[np.newaxis, s:e])
```

iii. The only explicit rationale is in the code comment `split into 1-min trials`. There is no richer note in `CONVERSION_NOTES.md`; the decision seems to have been that the recordings are continuous and therefore need fixed-length pseudo-trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. The only exclusion is that trailing remainder frames that do not fill a complete 60-second segment are dropped.

ii. ```python
remainder = n_frames - n_trials * trial_frames
if remainder > 0:
    print(f'  session {idx}: discarding last {remainder} frames '
          f'({remainder/FS:.1f}s) that do not fill a full trial')
```

iii. No detailed justification is documented. The code treats partial final chunks as unusable because the output format expects consistent trial lengths.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii. ```python
F = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The docstring for `preprocess_calcium` says it loads and preprocesses calcium data for one session. No further prose justification is provided elsewhere.

## 2-b. How is the `neural` data processed?

i. The code applies neuropil subtraction using coefficient `0.7`, then runs `suite2p.extraction.dcnv.preprocess` with `baseline='maximin'`, `win_baseline=60.0`, `sig_baseline=10`, `fs=30`, and `prctile_baseline=8.0`. The resulting baseline-corrected traces are used directly as `neural`.

ii. ```python
NEUCOEFF = 0.7

Fc = F - NEUCOEFF * Fneu
Fc = dcnv.preprocess(
    F=Fc,
    baseline='maximin',
    win_baseline=60.0,
    sig_baseline=10,
    fs=FS,
    prctile_baseline=8.0,
    batch_size=BATCH_SIZE,
    device=DEVICE,
)
```

iii. The justification is implicit in the use of suite2p preprocessing defaults. No narrative explanation beyond the function name and comments is present in the notes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filtering. The code does not consult `iscell.npy`, ROI statistics, deconvolved spikes, or other QC metadata. Every row in `F.npy` is retained.

ii. ```python
F = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'Fneu.npy'))
...
brain_region_idx.append(np.zeros(Fc.shape[0], dtype=np.int64))
```

iii. No justification is documented. By omission, the agent appears to have assumed the suite2p outputs were already sufficiently curated.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to session start rather than to an experimentally defined event. Each trial begins at the start of its 60-second chunk within the continuous session.

ii. ```python
for ti in range(n_trials):
    s = ti * trial_frames
    e = s + trial_frames
    neural_trials.append(Fc[:, s:e])

'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': None,
    'off_end': None,
}
```

iii. The justification is only implicit: the agent treated the recording as continuous and therefore used chunk boundaries rather than an event marker.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native 30 Hz frame rate, corresponding to `1000 / 30 = 33.333... ms` per bin. No rebinning or resampling is applied.

ii. ```python
FS = 30  # Hz

'metadata': {
    'time_bin_size': 1.0 / FS * 1000,
}
```

iii. No explicit rationale is written. The code directly uses the native sampling rate throughout.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from a timestamp variable in the raw data. It is synthesized from frame indices and the assumed constant frame rate `FS = 30`.

ii. ```python
t = (np.arange(trial_frames) / FS).astype(np.float32)
inp_trials.append(t[np.newaxis, :])
```

iii. No separate justification is documented. The implementation implies that frame number is sufficient to represent elapsed time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For every trial, the code constructs a vector `0, 1/30, 2/30, ...` up to 60 seconds and stores it as a single-row array of type `float32`.

ii. ```python
t = (np.arange(trial_frames) / FS).astype(np.float32)
inp_trials.append(t[np.newaxis, :])
```

iii. No explicit reasoning is documented. The choice is a direct consequence of representing time as seconds from the start of each fixed-length chunk.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It is sample-aligned frame-by-frame with the neural chunks because the time vector has exactly one entry per neural frame in each trial. However, it resets to 0 at the start of every 60-second pseudo-trial, so it is actually time from trial chunk start, not from the start of the full experiment.

ii. ```python
s = ti * trial_frames
e = s + trial_frames
t = (np.arange(trial_frames) / FS).astype(np.float32)

neural_trials.append(Fc[:, s:e])
inp_trials.append(t[np.newaxis, :])
```

iii. No explicit written justification was found. The code structure shows the agent wanted matching lengths between `input` and `neural`.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `output` motion energy is derived from `move_deve/motion_energy_glob.npy`, with `move_deve/interframe_int.npy` used to detect dropped video frames.

ii. ```python
me = np.load(os.path.join(session_path, 'move_deve', 'motion_energy_glob.npy'))
dt = np.load(os.path.join(session_path, 'move_deve', 'interframe_int.npy'))
```

iii. The function docstring and code indicate that `motion_energy_glob.npy` is the primary signal and `interframe_int.npy` supports dropped-frame repair. No fuller note is present.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code repairs short motion-energy arrays by inserting interpolated values at detected dropped-frame locations, normalizes each session’s motion energy by its standard deviation, concatenates all sessions to compute global percentile edges, then discretizes each time point into one of five bins with `np.digitize`.

ii. ```python
if me.shape[0] < expected_len:
    drop_indices = np.where(dt * 1000 > 0.04)[0]
    for offset, idx in enumerate(drop_indices):
        insert_pos = idx + 1 + offset
        interp_val = (me[insert_pos - 1] + me[insert_pos]) / 2.0
        me = np.insert(me, insert_pos, interp_val)

me = me / me.std()

concatenated = np.concatenate(all_me_flat)
percentiles = np.linspace(0, 100, n_levels + 1)
bin_edges = np.percentile(concatenated, percentiles)
output = np.digitize(me, bin_edges[1:-1])
```

iii. The code comments provide the only rationale: interpolate dropped frames, normalize by standard deviation, and discretize using percentile bins across all data.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. It is thresholded into five global percentile bins. The bin edges are the 0th, 20th, 40th, 60th, 80th, and 100th percentiles of the concatenated normalized motion-energy values from all sessions; `np.digitize` maps each sample to class `0` through `4`.

ii. ```python
N_LEVELS = 5

percentiles = np.linspace(0, 100, n_levels + 1)
bin_edges = np.percentile(concatenated, percentiles)

for me in all_me_flat:
    output = np.digitize(me, bin_edges[1:-1])
```

iii. No separate prose explanation is documented beyond the helper name `discretize_motion_energy` and the percentile-bin implementation itself.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code enforces frame-by-frame alignment within each session. If motion energy is shorter than the calcium trace, it infers dropped video frames from `interframe_int.npy`, inserts interpolated values, asserts the repaired length matches the neural frame count, and then slices both arrays with the same `[s:e]` trial boundaries.

ii. ```python
me = preprocess_motion_energy(session_path, expected_len=Fc.shape[1])

assert me.shape[0] == expected_len, (
    f'motion energy length {me.shape[0]} != expected {expected_len}'
)

for ti in range(n_trials):
    s = ti * trial_frames
    e = s + trial_frames
    neural_trials.append(Fc[:, s:e])
    output_trials.append(out[np.newaxis, s:e])
```

iii. The only explicit justification is functional: repair dropped frames so the motion signal can be indexed identically to the neural data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing video frames by interpolation and rejects unresolved length mismatches with an assertion. It also drops trailing remainder frames that do not fill a full pseudo-trial. There is no broader missing-data handling for neural traces or metadata.

ii. ```python
if me.shape[0] < expected_len:
    drop_indices = np.where(dt * 1000 > 0.04)[0]
    ...
    me = np.insert(me, insert_pos, interp_val)

assert me.shape[0] == expected_len, (
    f'motion energy length {me.shape[0]} != expected {expected_len}'
)

if remainder > 0:
    print(f'  session {idx}: discarding last {remainder} frames '
          f'({remainder/FS:.1f}s) that do not fill a full trial')
```

iii. No detailed rationale is documented. The behavior suggests the agent preferred simple repair for dropped video frames and hard failure for any remaining mismatch.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive computation is calcium preprocessing with `dcnv.preprocess`, which runs baseline correction over every neuron and frame, potentially on GPU. Large `.npy` loads and full-session motion-energy concatenation are secondary costs.

ii. ```python
Fc = dcnv.preprocess(
    F=Fc,
    baseline='maximin',
    win_baseline=60.0,
    sig_baseline=10,
    fs=FS,
    prctile_baseline=8.0,
    batch_size=BATCH_SIZE,
    device=DEVICE,
)

concatenated = np.concatenate(all_me_flat)
```

iii. The only clue to intended performance reasoning is `DEVICE = torch.device('cuda')`, which implies the agent expected `dcnv.preprocess` to dominate runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped-frame interpolation loop repeatedly calls `np.insert`, causing repeated reallocations. The trial-construction loop also repeatedly appends slices and recreates the same time vector per trial, which could be partially vectorized or hoisted out of the loop.

ii. ```python
for offset, idx in enumerate(drop_indices):
    insert_pos = idx + 1 + offset
    interp_val = (me[insert_pos - 1] + me[insert_pos]) / 2.0
    me = np.insert(me, insert_pos, interp_val)

for ti in range(n_trials):
    s = ti * trial_frames
    e = s + trial_frames
    t = (np.arange(trial_frames) / FS).astype(np.float32)
    neural_trials.append(Fc[:, s:e])
    inp_trials.append(t[np.newaxis, :])
    output_trials.append(out[np.newaxis, s:e])
```

iii. No explicit efficiency discussion is documented. This assessment is inferred directly from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code rebuilds the identical per-trial time vector `t` inside every trial loop, reloads and prints per-session array statistics for every session, and later recomputes concatenated per-session output counts purely for summary printing.

ii. ```python
for ti in range(n_trials):
    ...
    t = (np.arange(trial_frames) / FS).astype(np.float32)
    inp_trials.append(t[np.newaxis, :])

print(f'    F: {F.shape}  range [{F.min():.2f}, {F.max():.2f}]')
print(f'    Fneu: {Fneu.shape}  range [{Fneu.min():.2f}, {Fneu.max():.2f}]')
print(f'    Fc: {Fc.shape}  range [{Fc.min():.2f}, {Fc.max():.2f}]  mean={Fc.mean():.2f}')

all_o = np.concatenate([data['output'][si][ti].ravel() for ti in range(n_trials)])
levels, counts = np.unique(all_o, return_counts=True)
```

iii. No explicit justification is documented. The repetition appears to be for simplicity and verbose diagnostics rather than necessity.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It performs extensive console reporting of array ranges, means, dropped-frame indices, and per-session summary counts that are not stored in the output dataset. It also computes `all_o`, `levels`, and `counts` only to print session summaries after saving.

ii. ```python
print(f'    F: {F.shape}  range [{F.min():.2f}, {F.max():.2f}]')
print(f'    ME raw: {me.shape}  range [{me.min():.2f}, {me.max():.2f}]  (expected {expected_len})')
print(f'    missing {n_missing} frames, drops at indices: {drop_indices}')
print(f'    ME normalized: range [{me.min():.4f}, {me.max():.4f}]  mean={me.mean():.4f}  std={me.std():.4f}')

all_o = np.concatenate([data['output'][si][ti].ravel() for ti in range(n_trials)])
levels, counts = np.unique(all_o, return_counts=True)
print(f'  session {si} [{subj}]: {n_trials} trials  '
      f'neural ({data["neural"][si][0].shape[0]}, {trial_frames})  '
      f'region_idx {br.shape}  output_counts={dict(zip(levels, counts))}')
```

iii. No explicit justification is documented. These steps are evidently for manual inspection only and do not affect the saved pickle consumed downstream.
