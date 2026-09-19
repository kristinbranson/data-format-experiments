# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every subject directory under `data/`, then every session directory under each subject, and keeps only sessions that contain a required set of files. For each kept session it loads `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. Trials are not loaded from disk because the dataset is treated as continuous; trial-like windows are created later in code.

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

iii. In `CONVERSION_NOTES.md`, Step 2 describes the dataset as `data/<subject>/<session>/...` with Suite2p and `move_deve` subfolders, and Step 10 says the final processing uses Suite2p-derived activity plus motion energy and timestamps. The trajectory shows the agent explicitly checking for these files and treating recordings as continuous sessions that would be segmented later.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the first element of each tuple returned by `discover_sessions`, then deduplicated and sorted. This means subjects are defined by top-level directory names that contain at least one valid session.

ii. 
```python
sessions = discover_sessions()
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. `CONVERSION_NOTES.md` Step 2 reports subject folders such as `jm031`, `jm032`, etc., and Step 9 reports six subjects. The trajectory shows the agent relying on this folder structure rather than any subject metadata file.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject directory, provided that the required Suite2p and motion files exist. Sessions are sorted lexicographically within each subject.

ii. 
```python
for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        ...
        if all(p.exists() for p in req):
            sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

iii. `CONVERSION_NOTES.md` Step 2 states that data are organized as `data/<subject>/<session>/...` and that each valid session contains `suite2p/plane0/` and `move_deve/`. The trajectory repeatedly refers to one session as one continuous daily recording.

## 1-d. How are the data split into trials?

i. The agent does not use native trials. It segments each continuous session into consecutive non-overlapping 120-second windows after temporal binning. The number of windows is `floor(n_binned_timepoints / bins_per_window)`, and any leftover tail is implicitly dropped.

ii. 
```python
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
```

iii. `CONVERSION_NOTES.md` Step 3 says the paper used consecutive 2-minute blocks, Step 5 calls for pseudo-trials from continuous recordings, and the README says “Pseudo-trials: consecutive 2-minute windows from continuous recordings.” The trajectory also shows the agent choosing 2-minute windows because sample decoder accuracy improved with that representation.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filter. The only related filter is at the session level: sessions that produce fewer than two windows are skipped entirely. Partial trailing data that do not fill a full bin or full 120-second window are dropped by integer truncation.

ii. 
```python
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
...
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(...)
if len(neural_trials) < 2:
    continue
```

iii. The trajectory shows the agent focusing on the decoder requirement that each session must have at least two trials. There is no note in `CONVERSION_NOTES.md` describing any additional per-trial artifact rejection or behavioral-quality filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the final code, the saved neural data are derived from `suite2p/plane0/spks.npy`, filtered by `iscell.npy`. Although `F.npy` and `Fneu.npy` are loaded and a dF/F helper exists, those traces are not used in the final exported dataset.

ii. 
```python
F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 10 explicitly says the agent “switched to Suite2p `spks.npy`,” and the README lists “Neural signal: Suite2p `spks.npy` deconvolved activity.” The trajectory says this change was made after an earlier dF/F-like approach gave below-chance sample decoding.

## 2-b. How is the `neural` data processed?

i. Final neural processing is: filter ROIs by `iscell > 0.5`, trim the neural stream to the minimum common length shared with motion and timestamps, average over non-overlapping 10-frame bins, and then segment into 120-second windows. The defined `compute_df_f` function is not used.

ii. 
```python
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
...
neural_trials.append(neural_b[:, sl].astype(np.float32))
```

```python
def compute_df_f(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile = float(ops.get('prctile_baseline', 8.0))
    baseline = np.percentile(Fc, prctile, axis=1, keepdims=True).astype(np.float32)
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fc - baseline) / baseline
    return dff.astype(np.float32)
```

iii. The trajectory says the agent first tried a fluorescence-based dF/F-like signal, then replaced it with `spks.npy` because decoder performance improved. `CONVERSION_NOTES.md` Step 10 records that this was a deliberate change to improve validation accuracy.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural quality filter is `iscell > 0.5`, which keeps ROIs classified as cells by Suite2p. No additional filtering by activity level, missingness, or stability is applied.

ii. 
```python
CELL_THRESHOLD = 0.5
...
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Steps 3 to 5 repeatedly state that the paper considered ROIs above the default `iscell` threshold of 0.5 to be true cells, and the metadata records `"cell_inclusion_rule": "iscell probability > 0.5"`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to session start in the sense that windows are cut sequentially from the beginning of the session and the companion time input is measured from session start. There is no stimulus or behavioral event alignment beyond this.

ii. 
```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(
    neural_b, motion_b, t_b, fs_binned
)
...
'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
'off_start': 0.0,
'off_end': WINDOW_SECONDS,
```

iii. `CONVERSION_NOTES.md` Step 5 says the input should be elapsed time from session start and that sessions are continuous recordings segmented into fixed windows. The trajectory confirms the agent checked that later trials continue from 120 s onward rather than resetting within each session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping temporal bins. With `ops['fs']` defaulting to 30 Hz, this gives a 3 Hz series, i.e. about 333.33 ms per bin. The code averages both neural and motion streams within each bin and does not apply any additional temporal rebinning.

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

iii. `CONVERSION_NOTES.md` Step 3 and Step 5 say the paper averaged both dF/F and behavior in bins of 10 timestamps, and the trajectory shows the agent deliberately preserving that 10-frame denoising step.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The final time input is not derived from a stored time column. It is derived from the binned frame index plus the imaging sampling rate `ops['fs']`, converted to elapsed seconds from session start. `tstamps.npy` is loaded, and even binned into `_t_b_raw`, but not used for the saved input.

ii. 
```python
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
...
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 10 says the agent replaced an earlier timestamp-based input because the scale was wrong, and switched to frame-rate-derived elapsed seconds from session start. The trajectory documents that same correction.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The processing is: compute a binned sampling rate `fs_binned = ops['fs'] / 10`, create `t_b = np.arange(n_bins) / fs_binned`, cast to `float32`, and slice that session-long time vector into the same windows used for neural and motion data.

ii. 
```python
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
...
input_trials.append(make_time_input(time_b[sl]))
```

iii. The trajectory explicitly says the agent changed from raw timestamps to frame-rate-derived elapsed seconds because the original time scale was inconsistent with the task. `CONVERSION_NOTES.md` Step 10 records the same rationale.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: `t_b` is created at the same binned length as `neural_b`, and each trial uses the same slice `sl` for time, motion, and neural data. Within a session, later windows keep increasing absolute session time rather than resetting.

ii. 
```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
...
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
input_trials.append(make_time_input(time_b[sl]))
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The trajectory shows the agent explicitly checking trial 0, trial 1, and the last trial within several sessions to confirm that the input range advanced from `0-119.7` to `120-239.7`, etc., so the alignment and absolute timing were intentional.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`. `tstamps.npy` is also loaded, but only to support common-length trimming; it does not directly determine the output values.

ii. 
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
...
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `move_deve/motion_energy_glob.npy` to the decoder output, and the metadata records `"behavior_source": "global motion energy from move_deve/motion_energy_glob.npy"`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent trims the motion trace to the minimum shared length with neural data and timestamps, averages it in non-overlapping 10-frame bins, slices it into 120-second windows, concatenates all windows across all sessions, computes five global quantile edges, and digitizes each time bin into a class label.

ii. 
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
motion_b = bin_array_1d(motion, BIN_FRAMES)
...
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
...
vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
...
b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent planned to “discretize motion energy globally into 5 equal-percentile bins,” and the README repeats that choice. The trajectory shows the agent keeping this global binning after full verification.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. It is thresholded with five global equal-quantile bins computed from the concatenated motion values across all sessions and trials, not separately per session. The first and last edges are expanded to `-inf` and `inf` before digitization.

ii. 
```python
vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
edges[0] = -np.inf
edges[-1] = np.inf
...
b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly justifies global discretization as preserving “relative motion-state occupancy,” and Step 7 notes that the resulting global output distribution is exactly balanced even though per-session distributions are not.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion and neural data are aligned by trimming both streams, plus timestamps, to the shortest common raw length, then binning them with the same 10-frame averaging and slicing them with the same window boundaries.

ii. 
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]

...
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
...
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. `CONVERSION_NOTES.md` Step 4 resolves neural/behavior mismatches by “trimming to the minimum shared length,” and Step 10 repeats that “some sessions have neural/behavior length mismatches; conversion trims to shared minimum length before binning.”

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The final code handles mismatched stream lengths by silent truncation to the minimum common length across neural, motion, and timestamps. Trailing partial bins and trailing partial 120-second windows are also silently discarded. There is no explicit interpolation or assertion-based repair of missing motion frames.

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
```

iii. `CONVERSION_NOTES.md` Step 4 and Step 10 both say the agent chose to trim streams to shared duration when lengths mismatched. The trajectory shows this as a deliberate simplification after encountering alignment differences across sessions.

## 6-a. What are the most time-consuming steps of the code?

i. The agent did not explicitly document a final bottleneck analysis. From the implemented code, the heaviest steps are the per-session array loading and repeated per-session binning/segmentation, plus optional plotting in `--show-processing` mode. Because the final code does not use Suite2p baseline preprocessing, the reference pipeline’s major `dcnv.preprocess` cost is absent.

ii. 
```python
for subj, sess_name, sess_dir in sessions:
    ...
    F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
    ...
    neural_b = bin_array_2d(neural, BIN_FRAMES)
    motion_b = bin_array_1d(motion, BIN_FRAMES)
    ...
    if show_processing and len(session_info) <= 2:
        plot_processing(...)
    print(f'processed {subj}/{sess_name} in {time.time()-st:.2f}s: ...')
```

iii. `CONVERSION_NOTES.md` Step 6 leaves its efficiency subsection largely blank, so there is no strong explicit justification. The trajectory focuses on correctness bugs and decoder accuracy rather than profiling.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still uses Python loops over sessions, over windows within each session, and over every session/trial during percentile discretization. These are the main places where the agent could have reduced Python overhead. There is no evidence the agent implemented or documented further vectorization beyond using NumPy reshapes for binning.

ii. 
```python
for subj, sess_name, sess_dir in sessions:
    ...
    neural_trials, input_trials, motion_trials = [], [], []
    for i in range(n_windows):
        sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
        neural_trials.append(neural_b[:, sl].astype(np.float32))
        input_trials.append(make_time_input(time_b[sl]))
        motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

```python
out = []
for sess in all_motion_trials:
    sess_out = []
    for m in sess:
        b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
        sess_out.append(b[None, :])
    out.append(sess_out)
```

iii. The trajectory mentions performance only briefly and mainly in the context of fixing broken segmentation logic. `CONVERSION_NOTES.md` does not contain a substantive vectorization analysis.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats a few stages unnecessarily or redundantly: it loads `F.npy` and `Fneu.npy` for every session even though the final output uses only `spks.npy`; it bins timestamps into `_t_b_raw` and then ignores them; and it iterates over motion data once to build session/window lists and again to digitize them later.

ii. 
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
...
spks = np.load(suite / 'spks.npy').astype(np.float32)
return F, Fneu, spks, iscell, ops, motion, tstamps
```

```python
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
...
motion_all.append(motion_trials)
...
output_all, edges = percentile_bin_outputs(motion_all, n_classes=5)
```

iii. There is no explicit discussion of repeated work in `CONVERSION_NOTES.md`. This is inferred from the final code structure and is consistent with the trajectory’s pattern of patching correctness rather than revisiting code cleanup.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of work are computed but not used downstream: `F.npy` and `Fneu.npy` are loaded but ignored in the final pipeline, `compute_df_f` is defined but never called, and `_t_b_raw` is computed and discarded. The plotting titles also still refer to “dF/F” even though the final neural signal is `spks.npy`.

ii. 
```python
def compute_df_f(F, Fneu, ops):
    ...
    return dff.astype(np.float32)
```

```python
F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
...
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
...
neural = spks[keep].astype(np.float32)
```

iii. The trajectory shows the agent originally planning a fluorescence-based dF/F pipeline and later switching to `spks.npy` for decoder performance. The leftover unused function and file loads are artifacts of that earlier plan and were never cleaned up.
