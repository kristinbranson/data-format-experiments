# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans every top-level directory under `data/`, then every subdirectory under each subject directory. It only treats a directory as a valid session if it contains `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, `spks.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. Session contents are then loaded with `np.load`.

ii.
```python
def discover_sessions(root=ROOT):
    sessions = []
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            suite = sess_dir / 'suite2p' / 'plane0'
            move = sess_dir / 'move_deve'
            req = [suite/'F.npy', suite/'Fneu.npy', suite/'iscell.npy',
                   suite/'ops.npy', suite/'spks.npy',
                   move/'motion_energy_glob.npy', move/'tstamps.npy']
            if all(p.exists() for p in req):
                sessions.append((subj_dir.name, sess_dir.name, sess_dir))
    return sessions

def load_session(sess_dir):
    suite = sess_dir / 'suite2p' / 'plane0'
    move = sess_dir / 'move_deve'
    F = np.load(suite / 'F.npy').astype(np.float32)
    Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
    iscell = np.load(suite / 'iscell.npy')
    ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
    spks = np.load(suite / 'spks.npy').astype(np.float32)
    return F, Fneu, spks, iscell, ops, motion, tstamps
```

iii. The notes describe the dataset as `data/<subject>/<session>/...` with Suite2p outputs in `suite2p/plane0/` and behavior in `move_deve/`. In the trajectory and notes, the AI justified this as a continuous calcium-imaging dataset with motion energy and timestamps already organized per session.

## 1-b. How are the data split into subjects?

i. Subjects are the unique subject-directory names collected from the discovered sessions, sorted alphabetically. A mapping from subject name to integer index is then used to build `subject_idx`.

ii.
```python
sessions = discover_sessions()
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[subj])
```

iii. The notes say subject folder names such as `jm031` should populate `subjects` and `subject_idx`, and that session order should follow conversion order.

## 1-c. How are the data split into sessions?

i. Each valid session is one subject subdirectory returned by `discover_sessions`. After loading and processing, each session contributes one entry to `neural`, `input`, `output`, `brain_region_idx`, and `subject_idx`.

ii.
```python
for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        ...
        if all(p.exists() for p in req):
            sessions.append((subj_dir.name, sess_dir.name, sess_dir))

for subj, sess_name, sess_dir in sessions:
    ...
    neural_all.append(neural_trials)
    input_all.append(input_trials)
    motion_all.append(motion_trials)
    subject_idx.append(subject_to_idx[subj])
```

iii. The AI's notes state that sessions are continuous frame-aligned recordings stored per day inside each subject folder.

## 1-d. How are the data split into trials?

i. The AI assumes there are no native trials. It rebins each continuous session by averaging every 10 frames, then cuts the rebinned session into consecutive 2-minute windows. Any trailing partial window is discarded. Sessions producing fewer than two windows are skipped entirely.

ii.
```python
BIN_FRAMES = 10
WINDOW_SECONDS = 120.0

def segment_session(neural_b, motion_b, time_b, fs_binned, window_seconds=WINDOW_SECONDS):
    bins_per_window = max(1, int(round(window_seconds * fs_binned)))
    n_windows = neural_b.shape[1] // bins_per_window
    neural_trials, input_trials, motion_trials = [], [], []
    for i in range(n_windows):
        sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
        neural_trials.append(neural_b[:, sl].astype(np.float32))
        input_trials.append(make_time_input(time_b[sl]))
        motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
    return neural_trials, input_trials, motion_trials, bins_per_window

...
if len(neural_trials) < 2:
    continue
```

iii. The notes explicitly justify pseudo-trials as consecutive fixed windows because the recordings are continuous, and they cite the paper's use of consecutive 2-minute blocks and 10-timestamp averaging.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter. The code only keeps complete 2-minute windows, drops leftover partial windows by floor division, and skips entire sessions if they would yield fewer than two windows.

ii.
```python
n_windows = neural_b.shape[1] // bins_per_window
...
for i in range(n_windows):
    ...
if len(neural_trials) < 2:
    continue
```

iii. No trial-level quality-control rule is justified in the notes. The only stated rationale is satisfying the decoder requirement that sessions contain at least two trials/windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the final code, `neural` is derived from Suite2p `spks.npy`, filtered by `iscell.npy`. `F.npy` and `Fneu.npy` are loaded and a `compute_df_f` helper exists, but that helper is never used in the final pipeline.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
iscell = np.load(suite / 'iscell.npy')
spks = np.load(suite / 'spks.npy').astype(np.float32)
...
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The notes record a late change: the AI says an initial dF/F-like signal gave poor decoding, so it "switched to Suite2p `spks.npy`" instead.

## 2-b. How is the `neural` data processed?

i. The final neural pipeline is: select ROIs with `iscell > 0.5`, trim neural/motion/time to a common minimum length, average neural activity in 10-frame bins, and split into consecutive 2-minute windows. No neuropil subtraction or dF/F computation is applied in the saved output.

ii.
```python
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
...
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(
    neural_b, motion_b, t_b, fs_binned
)
```

iii. The AI originally planned to use fluorescence/dF/F, but the final notes say it changed to `spks.npy` after a decoder-performance check. The 10-frame averaging and 2-minute windows were justified from the methods text.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter is keeping ROIs whose `iscell` probability exceeds `0.5`.

ii.
```python
CELL_THRESHOLD = 0.5
...
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The notes justify this by quoting the paper's statement that ROIs above the default Suite2p threshold of `0.5` were considered true cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to session start, not to a stimulus event. Each trial/window is just a consecutive chunk of the session-start-aligned binned recording.

ii.
```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(
    neural_b, motion_b, t_b, fs_binned
)
...
'metadata': {
    'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
    'off_start': 0.0,
    'off_end': WINDOW_SECONDS,
}
```

iii. The notes repeatedly describe the recordings as continuous and justify session-start timing as the decoder input.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are rebinned from the original frame stream by averaging every 10 frames. With `ops['fs']` assumed to be 30 Hz, the saved time bin size is about `333.33 ms`.

ii.
```python
BIN_FRAMES = 10
...
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
...
'time_bin_size': float(1000.0 * np.median([1.0 / fs for fs in fs_binned_values]))
```

iii. The notes justify 10-frame temporal binning by citing the methods statement that dF/F and behavior were averaged in bins of 10 consecutive timestamps for decoding analyses.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. In the final code, time is derived from the binned frame index and the sampling rate from `ops['fs']`. Although `tstamps.npy` is loaded, its binned version is computed and discarded.

ii.
```python
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
...
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. The final notes say an initial timestamp-based input had the wrong scale, so the AI replaced it with frame-rate-derived elapsed seconds from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code first bins the recording to 10-frame time steps, computes a rebinned sampling rate, then generates an absolute session-time vector with `np.arange(...) / fs_binned`. Each window receives the corresponding slice of that absolute time vector.

ii.
```python
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
...
input_trials.append(make_time_input(time_b[sl]))
```

iii. The notes explicitly justify this as a fix for the earlier timestamp-scaling bug, and as the intended "time-from-session-start" decoder input.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction: after the neural data are trimmed and rebinned, `t_b` is generated at the same rebinned length and then sliced with the same window indices used for neural data.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
...
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
...
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
input_trials.append(make_time_input(time_b[sl]))
```

iii. The notes say the corrected input "matches absolute session-start timing" across pseudo-trials.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`. `tstamps.npy` is loaded only to support common-length trimming; `interframe_int.npy` is not used.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
...
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
```

iii. The notes justify `motion_energy_glob.npy` as the motion variable described in the paper and treat the streams as already synchronized enough to trim to a common valid length.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code trims motion to the shortest of neural/motion/timestamp lengths, averages motion over 10-frame bins, and later discretizes the rebinned motion globally. It does not normalize motion by session standard deviation before binning.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]

motion_b = bin_array_1d(motion, BIN_FRAMES)

def percentile_bin_outputs(all_motion_trials, n_classes=5):
    vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
    edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
```

iii. The notes justify the 10-frame averaging from the methods text and justify global percentile discretization from the decoder task. There is no explicit justification for omitting the reference normalization/interpolation procedure.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates all motion values from all sessions/trials, computes five equal-quantile bins, forces the first and last edges to `-inf` and `inf`, and assigns category labels with `np.digitize`.

ii.
```python
def percentile_bin_outputs(all_motion_trials, n_classes=5):
    vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
    edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    ...
    b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
```

iii. The notes explicitly justify global five-way equal-percentile discretization as the decoder task requirement.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion is aligned to neural activity by truncating both streams to a common minimum length, binning both by the same 10-frame factor, and slicing them with the same consecutive 2-minute window boundaries.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
...
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The notes justify this with the claim that some sessions have neural/behavior length mismatches and that trimming to a shared minimum length is an acceptable fix.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles mismatched stream lengths by truncating all streams to the shortest length. Leftover frames that do not fit into 10-frame bins or full 2-minute windows are silently dropped via floor division. Sessions with fewer than two resulting windows are skipped.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]

def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)

def bin_array_1d(x, bin_frames):
    n_bins = x.shape[0] // bin_frames
    x = x[:n_bins * bin_frames]
    return x.reshape(n_bins, bin_frames).mean(axis=1)

if len(neural_trials) < 2:
    continue
```

iii. The only explicit justification in the notes is that some sessions had neural/behavior length mismatches, so the conversion trims to a shared minimum length before binning.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps in the final code are likely loading large per-session arrays from disk, rebinnig whole-session neural and motion arrays with `reshape(...).mean(...)`, and concatenating all motion trials to compute global quantile edges. Optional plotting also adds cost when enabled.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
spks = np.load(suite / 'spks.npy').astype(np.float32)
...
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
...
vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
...
if show_processing and len(session_info) <= 2:
    plot_processing(...)
```

iii. The notes do not explicitly analyze runtime in Step 6, so this is inferred from the final code structure.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-window loop in `segment_session` and the nested session/trial loop in `percentile_bin_outputs` could be replaced by more vectorized reshaping/indexing. The directory-discovery loop is I/O-driven and less relevant.

ii.
```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    input_trials.append(make_time_input(time_b[sl]))
    motion_trials.append(motion_b[sl].astype(np.float32)[None, :])

for sess in all_motion_trials:
    sess_out = []
    for m in sess:
        b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
        sess_out.append(b[None, :])
    out.append(sess_out)
```

iii. The notes contain no explicit vectorization discussion; this is inferred from the final implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats type conversions and slicing/copying work several times. Arrays are cast to `float32` at load time and then cast again while building trials. It also performs three separate but structurally identical binning passes for neural, motion, and timestamps.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
spks = np.load(suite / 'spks.npy').astype(np.float32)
...
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
...
neural_trials.append(neural_b[:, sl].astype(np.float32))
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The notes do not justify these repeated operations; this is visible directly in the code.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are unused in the final dataset: `F.npy` and `Fneu.npy` are loaded but never used, `compute_df_f` is defined but never called, and `_t_b_raw` is computed from `tstamps.npy` and then discarded. Optional plotting also does not affect the saved dataset.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
...
def compute_df_f(F, Fneu, ops):
    ...
    return dff.astype(np.float32)
...
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
```

iii. The notes explain why the dF/F path was abandoned: the AI says it switched to `spks.npy` after poor decoder performance. They do not justify keeping the now-unused loads and helper code.
