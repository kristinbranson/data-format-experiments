# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively scans `data/`, treating each top-level directory as a subject and each nested directory as a session. A session is only included if it contains Suite2p files plus motion/timestamp files. For each included session it loads `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy` with `np.load`.

ii. Code snippets

```python
def discover_sessions(root=ROOT):
    sessions = []
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            suite = sess_dir / 'suite2p' / 'plane0'
            move = sess_dir / 'move_deve'
            req = [suite/'F.npy', suite/'Fneu.npy', suite/'iscell.npy', suite/'ops.npy', suite/'spks.npy', move/'motion_energy_glob.npy', move/'tstamps.npy']
            if all(p.exists() for p in req):
                sessions.append((subj_dir.name, sess_dir.name, sess_dir))
    return sessions
```

```python
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

iii. The notes say the agent found the dataset organized as `data/<subject>/<session>/...` and identified valid sessions by the presence of Suite2p neural outputs and `move_deve` behavior files. The trajectory shows it intentionally summarized the directory tree first, then wrote loaders around that structure.

## 1-b. How are the data split into subjects?

i. Subjects are defined entirely by the sorted names of the top-level directories under `data/`. Later, `subjects` is the sorted unique set of subject names found in the discovered sessions, and `subject_idx` maps each kept session back to that list.

ii. Code snippets

```python
for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        ...
        sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

```python
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[subj])
```

iii. The notes explicitly state that data are organized as `data/<subject>/<session>/...` and list six subjects (`jm031` ... `jm046`). The data README, which the agent inspected, also says the six subject folders correspond to the mice.

## 1-c. How are the data split into sessions?

i. Sessions are defined by the sorted subdirectories inside each subject directory. Each such folder is treated as one recording day/session if all required files exist. The converted dataset keeps one output session per discovered session, unless it is later dropped for having fewer than two 2-minute windows.

ii. Code snippets

```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    suite = sess_dir / 'suite2p' / 'plane0'
    move = sess_dir / 'move_deve'
    req = [suite/'F.npy', suite/'Fneu.npy', suite/'iscell.npy', suite/'ops.npy', suite/'spks.npy', move/'motion_energy_glob.npy', move/'tstamps.npy']
    if all(p.exists() for p in req):
        sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

```python
for subj, sess_name, sess_dir in sessions:
    ...
    if len(neural_trials) < 2:
        continue
    neural_all.append(neural_trials)
    input_all.append(input_trials)
    motion_all.append(motion_trials)
```

iii. The notes say each subject folder contains a number of session folders, one per recording day. The trajectory shows the agent adopted that folder layout directly rather than inferring sessions from timestamps or metadata inside files.

## 1-d. How are the data split into trials?

i. The raw recordings are treated as continuous sessions with no native trials. The script bins the full session, then cuts each session into consecutive non-overlapping 120-second windows, which it treats as pseudo-trials.

ii. Code snippets

```python
WINDOW_SECONDS = 120.0
```

```python
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
```

iii. In the notes the agent quotes the methods: “splits were done on consecutive 2 minute blocks of the recording.” It used that as justification for creating pseudo-trials from continuous recordings.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control step. Instead, the script drops any leftover partial window at the end of a session, and it skips any session that would produce fewer than two windows/trials.

ii. Code snippets

```python
n_windows = neural_b.shape[1] // bins_per_window
...
for i in range(n_windows):
    ...
```

```python
if len(neural_trials) < 2:
    continue
```

iii. The notes repeatedly state that there are no native trials in the source data, only continuous recordings split into 2-minute blocks. No separate trial-quality rule is documented in the notes or trajectory beyond meeting the decoder requirement of at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the final code, `neural` is derived from `suite2p/plane0/spks.npy` after filtering rows by `iscell.npy`. Although `F.npy` and `Fneu.npy` are loaded and a `compute_df_f` helper exists, that path is unused in the final conversion.

ii. Code snippets

```python
F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
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

iii. The notes initially planned to use fluorescence-derived dF/F from `F.npy` and `Fneu.npy`, matching the methods. The trajectory shows the agent later changed course: “switched to Suite2p `spks.npy`, which improved sample and full decoding substantially,” and Step 30 says it kept `spks.npy` because decoder accuracy improved.

## 2-b. How is the `neural` data processed?

i. The final pipeline filters cells with `iscell > 0.5`, truncates the neural stream to the minimum common length shared with motion and timestamps, averages the neural values in non-overlapping 10-frame bins, and then segments the binned stream into consecutive 120-second windows. No dF/F is actually computed in the final path.

ii. Code snippets

```python
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
```

```python
def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)
```

```python
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(neural_b, motion_b, t_b, fs_binned)
```

iii. The notes justify 10-frame averaging from the methods text: both neural and behavior traces were “averaging in bins of 10 consecutive timestamps.” The justification for the `spks.npy` substitution came later in the trajectory: it was retained because it improved decoder metrics, even though the notes acknowledged the methods had specified dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered by the Suite2p cell-classification score: only rows with `iscell[:,0] > 0.5` are kept. There is no additional neuron QC in the script.

ii. Code snippets

```python
CELL_THRESHOLD = 0.5
...
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

```python
'cell_inclusion_rule': 'iscell probability > 0.5',
```

iii. The notes quote the methods directly: “We considered all ROIs above the default threshold of 0.5 as true cells.” This rule was consistent from the planning notes through the final metadata.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script does not align to a discrete behavioral event. Instead it aligns everything to session start, builds a continuous time axis from there, and then cuts the session into consecutive 2-minute windows.

ii. Code snippets

```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(neural_b, motion_b, t_b, fs_binned)
```

```python
'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
'off_start': 0.0,
'off_end': WINDOW_SECONDS,
```

iii. The notes state that the source recordings are continuous and that the reference decoding used consecutive 2-minute blocks rather than event-locked trials. The agent therefore documented “session start” as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are rebinned by averaging every 10 original frames. Since `ops['fs']` is 30 Hz in all sessions, the final bin size is about 333.3 ms.

ii. Code snippets

```python
BIN_FRAMES = 10
...
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
```

```python
'time_bin_size': float(1000.0 * np.median([1.0 / fs for fs in fs_binned_values])) if fs_binned_values else float('nan'),
'binning_frames': BIN_FRAMES,
```

iii. The notes cite the methods: “we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps.” The trajectory also recorded a check that `ops['fs']` was 30 Hz across sessions.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. In the final code, the input time is not taken from `tstamps.npy`. It is derived from `ops['fs']` plus the number of retained/binned neural samples, using `np.arange` to build an evenly spaced time axis from zero.

ii. Code snippets

```python
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

```python
def make_time_input(t_binned):
    return t_binned[None, :].astype(np.float32)
```

iii. The trajectory says the agent originally built time from raw timestamps but changed it because the “raw timestamps produced incorrect scale,” replacing them with a frame-rate-derived elapsed-time axis. The notes later describe the input as “time-from-session-start.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. After trimming and binning the session, the script computes a synthetic regularly spaced time vector with spacing `1 / (ops['fs'] / 10)`, converts it to `float32`, and wraps each window as a `1 x T` array. The binned raw timestamps are computed into `_t_b_raw` and then discarded.

ii. Code snippets

```python
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

```python
input_trials.append(make_time_input(time_b[sl]))
```

iii. The trajectory explicitly justifies the change away from timestamps: the agent says it “replaced [them] with frame-rate-derived elapsed seconds from session start.” No further biological or reference-based justification is given.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is constructed on the same binned sample axis as the neural data and is segmented using the exact same slices. Later windows preserve absolute session time rather than resetting to zero.

ii. Code snippets

```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(neural_b, motion_b, t_b, fs_binned)
```

```python
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
input_trials.append(make_time_input(time_b[sl]))
```

iii. The trajectory shows the agent verified this after conversion by checking that later trials had ranges like `120.0` to `239.67` and `1680.0` to `1799.67`, concluding the time input was absolute from session start across windows.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes from `move_deve/motion_energy_glob.npy`. The script does not recompute motion from video frames; it uses the precomputed scalar motion-energy trace supplied in the dataset.

ii. Code snippets

```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
```

```python
'behavior_source': 'global motion energy from move_deve/motion_energy_glob.npy',
```

iii. The notes justify this directly from the data layout and methods: they say `motion_energy_glob.npy` is the processed behavioral variable corresponding to the paper’s global movement measure computed from frame differences.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script trims motion to the same common length as neural/timestamps, averages it in non-overlapping 10-frame bins, segments it into 120-second windows, and later digitizes the continuous values into classes.

ii. Code snippets

```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
motion_b = bin_array_1d(motion, BIN_FRAMES)
```

```python
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The notes cite the methods for the 10-frame averaging of behavior traces. The rest of the processing is the agent’s pseudo-trial packaging needed for the decoder format.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After gathering all windowed motion traces across all sessions, the script computes global quintile edges across every motion sample in the converted dataset, then uses `np.digitize` to assign each binned timepoint to one of five classes.

ii. Code snippets

```python
def percentile_bin_outputs(all_motion_trials, n_classes=5):
    vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
    edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
```

```python
for m in sess:
    b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
    sess_out.append(b[None, :])
```

iii. The notes justify this from the decoder task rather than the reference analysis: they planned to “discretize motion energy globally into 5 equal-percentile bins” because the decoder output had to be categorical.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion is aligned to neural activity by truncating both streams to the minimum shared length with timestamps, binning them with the same 10-frame rule, and slicing them with the same 2-minute window boundaries.

ii. Code snippets

```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]
```

```python
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The notes justify this as a practical resolution to length mismatches: “align by shared valid length.” The data README the agent read had noted that some sessions have missing camera frames and suggested using `tstamps.npy`/`interframe_int.npy` to mark or interpolate missing values; the final script did not do that and instead truncated.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles mismatched stream lengths only by truncating neural, motion, and timestamps to the shortest common prefix. It also drops any leftover partial 10-frame bin and any leftover partial 2-minute window. It does not interpolate missing camera frames, mark missing values, or otherwise repair data.

ii. Code snippets

```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]
```

```python
def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)
```

```python
n_windows = neural_b.shape[1] // bins_per_window
```

iii. The notes acknowledge the issue: some sessions had neural/behavior length mismatches. The chosen justification was simply to “trim to shared minimum length before binning,” even though the data README said missing camera frames could instead be treated as missing values or interpolated over.

## 6-a. What are the most time-consuming steps of the code?

i. The heavy work is the per-session loading of full dense arrays from disk (`F`, `Fneu`, `spks`, motion, timestamps), the full-session 10-frame binning of neural and motion traces, and the global pass over all motion samples to concatenate and compute quantiles before digitization.

ii. Code snippets

```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
...
spks = np.load(suite / 'spks.npy').astype(np.float32)
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
```

```python
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
```

```python
vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
```

iii. The notes do not give a detailed performance analysis. This is inferred from the code structure and from the conversion logs showing per-session processing dominates runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious Python-level loops remain: the session-segmentation loop that appends one trial/window at a time, and the nested session/trial loop inside `percentile_bin_outputs` that digitizes each trial separately. Both could be handled more vectorially once arrays are already stacked per session.

ii. Code snippets

```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    input_trials.append(make_time_input(time_b[sl]))
    motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

```python
for sess in all_motion_trials:
    sess_out = []
    for m in sess:
        b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
        sess_out.append(b[None, :])
    out.append(sess_out)
```

iii. There is no explicit justification in the notes for keeping these loops. The implementation favors simplicity over vectorization.

## 6-c. What processing does the code repeat multiple times?

i. The script traverses all motion trials twice during output creation: once to concatenate every sample and compute quantiles, and again to digitize each trial. It also re-slices session data window-by-window after already having contiguous binned arrays in memory.

ii. Code snippets

```python
vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
...
for sess in all_motion_trials:
    sess_out = []
    for m in sess:
        b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
        sess_out.append(b[None, :])
```

```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    ...
```

iii. The notes do not discuss this repetition. It is a direct consequence of how the output binning function is written.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of work are unused in the final output: loading `F.npy` and `Fneu.npy` for every session, defining `compute_df_f` but never calling it, binning timestamps into `_t_b_raw` and then discarding them, and retaining plotting labels that still call the neural signal “dF/F” even though the final signal is `spks.npy`.

ii. Code snippets

```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
...
return F, Fneu, spks, iscell, ops, motion, tstamps
```

```python
def compute_df_f(F, Fneu, ops):
    ...
    return dff.astype(np.float32)
```

```python
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
```

```python
axes[1].set_title('raw neural dF/F example cell (first 2000 frames)')
axes[3].set_title('binned neural dF/F example cell')
```

iii. The trajectory explains why the unused dF/F path remained: the agent originally implemented a fluorescence-based route, then switched to `spks.npy` after decoder performance improved. It never removed the now-dead dF/F-related code or the unused timestamp binning.
