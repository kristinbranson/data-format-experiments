# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code scans `data/` for subject directories, scans each subject for session directories, and keeps only sessions that contain a fixed required file set: `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, `spks.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. For each kept session it loads fluorescence, neuropil, deconvolved spikes, cell labels, Suite2p ops, motion energy, and timestamps. Trials are not loaded from disk because the code assumes the recordings are continuous and creates pseudo-trials later.

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

iii. The justification is spread across the notes and trajectory rather than stated in one place. In `CONVERSION_NOTES.md` Step 2, the AI records that valid sessions contain Suite2p outputs plus `move_deve` motion files. In the trajectory, it repeatedly treats the dataset as continuous calcium-imaging sessions with synchronized motion traces, and later keeps `spks.npy` in the required file set after deciding to use deconvolved spikes instead of fluorescence-derived signals.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the subject name attached to each discovered session. After discovery, the code forms a sorted set of subject names and maps each subject to an integer index.

ii.
```python
sessions = discover_sessions()
if sample:
    sessions = sessions[:2]
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI's notes justify subject splitting only implicitly: `CONVERSION_NOTES.md` Step 2 states that the data are organized as `data/<subject>/<session>/...` and lists six subject IDs. The trajectory does not show a separate argument for subject handling beyond following the directory structure.

## 1-c. How are the data split into sessions?

i. Each session is one subject subdirectory under `data/<subject>/`, but only if the required neural and motion files all exist. Sessions are yielded in sorted directory order and kept as `(subject, session_name, session_path)` tuples.

ii.
```python
for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        suite = sess_dir / 'suite2p' / 'plane0'
        move = sess_dir / 'move_deve'
        req = [suite/'F.npy', suite/'Fneu.npy', suite/'iscell.npy',
               suite/'ops.npy', suite/'spks.npy',
               move/'motion_energy_glob.npy', move/'tstamps.npy']
        if all(p.exists() for p in req):
            sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

iii. The AI justifies this with the observed data organization. In `CONVERSION_NOTES.md` Step 2 it writes that valid sessions contain `suite2p/plane0/` neural files and `move_deve/` behavioral files, so the discovery rule mirrors that assumption.

## 1-d. How are the data split into trials?

i. The AI assumes there are no native trials. It first averages every 10 frames, then splits each continuous session into consecutive non-overlapping 120 second windows. At the binned sampling rate this produces `bins_per_window = round(120 * fs_binned)` time bins per pseudo-trial.

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

fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(
    neural_b, motion_b, t_b, fs_binned
)
```

iii. This choice is explicitly justified in the notes and trajectory. `CONVERSION_NOTES.md` Step 3 cites the methods text saying the reference decoding used "consecutive 2 minute blocks of the recording," and Step 5 says there are no native trials so the code should create pseudo-trials from continuous recordings. The trajectory repeats that 2-minute windows after 10-frame averaging were chosen to stay close to the paper while satisfying the decoder format.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control rule such as dropping low-quality motion or neural trials. Instead, the code implicitly drops incomplete data in three ways: it truncates leftover frames that do not make a full 10-frame bin, truncates leftover bins that do not make a full 120 second window, and skips entire sessions that yield fewer than two windows.

ii.
```python
def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)

def bin_array_1d(x, bin_frames):
    n_bins = x.shape[0] // bin_frames
    x = x[:n_bins * bin_frames]
    return x.reshape(n_bins, bin_frames).mean(axis=1)

def segment_session(neural_b, motion_b, time_b, fs_binned, window_seconds=WINDOW_SECONDS):
    bins_per_window = max(1, int(round(window_seconds * fs_binned)))
    n_windows = neural_b.shape[1] // bins_per_window
    ...

neural_trials, input_trials, motion_trials, bins_per_window = segment_session(...)
if len(neural_trials) < 2:
    continue
```

iii. The trajectory shows the justification for skipping sessions with fewer than two windows: after an early failure, the AI added a guard because the decoder requires at least two trials per session. The notes do not describe any more detailed QC for individual trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the final implemented pipeline, `neural` is derived from Suite2p's deconvolved spike output `spks.npy`, after filtering ROIs by `iscell`. Although the code also loads `F.npy` and `Fneu.npy` and defines a `compute_df_f` helper, those fluorescence arrays are not used to build the saved neural data.

ii.
```python
def load_session(sess_dir):
    ...
    F = np.load(suite / 'F.npy').astype(np.float32)
    Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
    ...
    spks = np.load(suite / 'spks.npy').astype(np.float32)
    return F, Fneu, spks, iscell, ops, motion, tstamps

...

F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The AI explicitly changed this decision during debugging. `CONVERSION_NOTES.md` Step 10 says the initial dF/F-like fluorescence signal gave below-chance sample decoding, so the AI "switched to Suite2p `spks.npy`, which improved sample and full decoding substantially." The README also states that the neural signal is `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The final code does not compute the saved neural data from fluorescence. Instead it filters `spks.npy` by `iscell`, trims neural/motion/timestamp arrays to a common minimum length, averages the neural signal in 10-frame bins, and segments the result into consecutive 2-minute windows. A `compute_df_f` function exists but is dead code in the final pipeline.

ii.
```python
def compute_df_f(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile = float(ops.get('prctile_baseline', 8.0))
    baseline = np.percentile(Fc, prctile, axis=1, keepdims=True).astype(np.float32)
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fc - baseline) / baseline
    return dff.astype(np.float32)

...

keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(
    neural_b, motion_b, t_b, fs_binned
)
```

iii. The trajectory shows the reason for this shift. The AI originally planned to use fluorescence-derived dF/F, citing the methods text, but after sample decoder training underperformed it switched to `spks.npy` because that produced above-chance decoding. The notes preserve that rationale and acknowledge the discrepancy with the paper-derived interpretation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The neural data are filtered by keeping only ROIs whose `iscell` score exceeds `0.5`. No additional neuron QC is applied after that.

ii.
```python
CELL_THRESHOLD = 0.5

...

keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)

...

'cell_inclusion_rule': 'iscell probability > 0.5',
```

iii. This was justified from the methods text. `CONVERSION_NOTES.md` Steps 3 to 5 repeatedly cite the paper statement that ROIs above the default `0.5` threshold are considered true cells, and the trajectory uses that to motivate the filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to session start. After 10-frame binning, each pseudo-trial is a consecutive window from the continuous session, and the metadata describes the alignment event as session start rather than a behavioral event.

ii.
```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(
    neural_b, motion_b, t_b, fs_binned
)

...

'metadata': {
    ...
    'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
    'off_start': 0.0,
    'off_end': WINDOW_SECONDS,
    ...
}
```

iii. The notes justify this from the continuous nature of the recordings. `CONVERSION_NOTES.md` Step 3 says there are no native trials and the reference analysis used continuous 2-minute blocks, so the AI treated session start as the relevant anchor. Later trajectory steps confirm that the input time variable is meant to remain absolute from session start across windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are rebinned by averaging every 10 original frames. Given the code's use of `ops['fs']` with a default of `30.0`, the saved time bin is effectively `10 / 30` seconds, about `333.3 ms`.

ii.
```python
BIN_FRAMES = 10

def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)

fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)

...

'time_bin_size': float(1000.0 * np.median([1.0 / fs for fs in fs_binned_values])),
'binning_frames': BIN_FRAMES,
```

iii. The AI explicitly justified 10-frame binning from the methods text. `CONVERSION_NOTES.md` Step 3 says the reference decoding averaged neural and behavioral traces in bins of 10 consecutive timestamps, and the trajectory says this was chosen to stay close to the reference decoding setup.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. In the final code, the saved time input is not derived from a retained raw timestamp variable. The code loads `tstamps.npy`, but the final input is computed from the number of binned samples and the sampling rate `ops['fs'] / BIN_FRAMES`. The binned raw timestamps are computed into `_t_b_raw` and then discarded.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)

...

motion_b = bin_array_1d(motion, BIN_FRAMES)
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. The trajectory explicitly explains this. After verifying the sample output, the AI concluded that using raw timestamps gave the wrong scale for the decoder input and replaced them with frame-rate-derived elapsed seconds from session start. `CONVERSION_NOTES.md` Step 10 records the same fix.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code constructs a uniformly spaced time vector after neural/motion trimming and 10-frame binning. It uses `np.arange(n_bins) / fs_binned` to get elapsed seconds from session start on the binned grid, then slices that vector into the same 2-minute windows as the neural and motion data and stores each window as a `1 x T` array.

ii.
```python
def make_time_input(t_binned):
    return t_binned[None, :].astype(np.float32)

...

fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(
    neural_b, motion_b, t_b, fs_binned
)

...

input_trials.append(make_time_input(time_b[sl]))
```

iii. The AI justified this in the trajectory after discovering that the first version of the input time series had the wrong scale. It explicitly says the task requires elapsed time from the beginning of the experiment, so it rebuilt the time input from frame index and sampling rate instead of using `tstamps`.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction. The code first trims all streams to a common length, bins them on the same 10-frame grid, then uses the same window slice `sl` to cut the neural, time, and motion arrays into pseudo-trials.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
...
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    input_trials.append(make_time_input(time_b[sl]))
    motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The trajectory shows the AI checking this explicitly: it verified that later trials continue from `120 s`, `240 s`, and so on rather than resetting inside each session. That check was used as the justification that the input remained aligned to the same session-start timeline as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`. The code also loads `tstamps.npy`, but only to help trim streams to a shared minimum length. It does not use `interframe_int.npy`.

ii.
```python
def load_session(sess_dir):
    ...
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
    ...
    return F, Fneu, spks, iscell, ops, motion, tstamps

def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]
```

iii. The notes justify `motion_energy_glob.npy` from the dataset exploration and methods text: `CONVERSION_NOTES.md` Steps 2 to 5 identify this file as the synchronized global movement variable. The AI's alignment choice was to use timestamps only for consistency checks and trimming.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The implemented processing is: trim the motion signal to the common neural/motion/timestamp duration, average it in 10-frame bins, segment it into 2-minute windows, and later convert the continuous binned values to discrete bins. The final code does not normalize motion energy by standard deviation and does not interpolate dropped frames.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]

def bin_array_1d(x, bin_frames):
    n_bins = x.shape[0] // bin_frames
    x = x[:n_bins * bin_frames]
    return x.reshape(n_bins, bin_frames).mean(axis=1)

...

neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
motion_b = bin_array_1d(motion, BIN_FRAMES)
neural_trials, input_trials, motion_trials, bins_per_window = segment_session(
    neural_b, motion_b, t_b, fs_binned
)
```

iii. The AI justified 10-frame averaging and continuous-window segmentation from the methods text. Its choice to trim to common length rather than repair dropped frames is justified in `CONVERSION_NOTES.md` Step 4 and Step 10 as a pragmatic way to handle mismatched stream lengths.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. All binned motion values from all sessions and windows are concatenated, global quantile edges are computed at 0%, 20%, 40%, 60%, 80%, and 100%, and each trial is digitized into five categories `0` to `4`. The code forces the first and last edges to `-inf` and `inf`.

ii.
```python
def percentile_bin_outputs(all_motion_trials, n_classes=5):
    vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
    edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    out = []
    for sess in all_motion_trials:
        sess_out = []
        for m in sess:
            b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
            sess_out.append(b[None, :])
        out.append(sess_out)
    return out, edges

...

output_all, edges = percentile_bin_outputs(motion_all, n_classes=5)
```

iii. The AI justifies this directly in `CONVERSION_NOTES.md` Step 5 and in the README: the task requires motion energy to be discretized into five equal-percentile bins, so the code uses global percentile thresholds across the converted dataset.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by trimming all streams to the shared minimum length, rebinned with the same 10-frame averaging used for neural data, and cut with the same 2-minute window slices. There is no explicit dropped-frame interpolation; the alignment strategy is to force all arrays to the common overlap.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]

...

neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)

for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The justification is explicit in the notes. `CONVERSION_NOTES.md` Step 4 says some sessions show frame-count mismatches between neural and behavior arrays, and the AI resolves this by aligning streams "by common valid duration, trimming to the minimum shared length." Step 10 repeats that rationale.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI does not repair missing motion frames. Instead it handles inconsistencies by trimming neural, motion, and timestamp arrays to their shared minimum length, dropping incomplete trailing bins or windows, and skipping sessions that produce fewer than two windows after segmentation.

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

...

if len(neural_trials) < 2:
    continue
```

iii. `CONVERSION_NOTES.md` Step 4 and Step 10 justify trimming as the chosen response to neural/behavior length mismatches. The trajectory also shows the AI adding the two-trial guard after a failed sample run, because the downstream decoder requires at least two trials per session.

## 6-a. What are the most time-consuming steps of the code?

i. The AI did not explicitly document a bottleneck analysis in the final notes. From the implemented code, the most expensive steps are likely per-session loading of large `.npy` arrays, repeated frame-binning across all neurons and time points, session-wise segmentation, and optional plotting when `--show-processing` is enabled. The final pipeline does not include the heavier Suite2p-style preprocessing that the unused `compute_df_f` helper suggests.

ii.
```python
def load_session(sess_dir):
    ...
    F = np.load(suite / 'F.npy').astype(np.float32)
    Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
    ...
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
    spks = np.load(suite / 'spks.npy').astype(np.float32)
    return F, Fneu, spks, iscell, ops, motion, tstamps

for subj, sess_name, sess_dir in sessions:
    ...
    neural_b = bin_array_2d(neural, BIN_FRAMES)
    motion_b = bin_array_1d(motion, BIN_FRAMES)
    ...
    if show_processing and len(session_info) <= 2:
        plot_processing(...)
    print(f'processed {subj}/{sess_name} in {time.time()-st:.2f}s: neurons={neural.shape[0]}, trials={len(neural_trials)}')
```

iii. The code prints per-session timing, but `CONVERSION_NOTES.md` Step 6 leaves the speed-analysis placeholders largely unfilled. So the justification here is mostly an inference from the implementation rather than an explicit claim by the AI.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The binning functions are already vectorized, but two notable Python-level loops remain. `segment_session` loops over windows and appends one sliced trial at a time, and `percentile_bin_outputs` loops over every session and every trial when digitizing motion values. Those steps could be reshaped and batched more aggressively.

ii.
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

def percentile_bin_outputs(all_motion_trials, n_classes=5):
    ...
    for sess in all_motion_trials:
        sess_out = []
        for m in sess:
            b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
            sess_out.append(b[None, :])
        out.append(sess_out)
```

iii. The AI did not explicitly write this analysis in its notes. This is an inference from the final code structure. The trajectory only mentions speed issues indirectly when it fixed the sample run and later filled in timing placeholders.

## 6-c. What processing does the code repeat multiple times?

i. The final code still carries remnants of earlier approaches. It repeatedly loads fluorescence arrays `F.npy` and `Fneu.npy` even though the final neural output uses `spks.npy`; it also computes a binned timestamp array `_t_b_raw` and never uses it. Multiple arrays are cast to `float32` more than once as they move through the pipeline.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
...
spks = np.load(suite / 'spks.npy').astype(np.float32)

...

neural = spks[keep].astype(np.float32)
...
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
...
neural_trials.append(neural_b[:, sl].astype(np.float32))
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The justification comes from the debugging history. The trajectory and `CONVERSION_NOTES.md` Step 10 show that the AI first pursued dF/F and raw timestamps, then switched to `spks.npy` and synthetic elapsed time. The unused loads and computations are leftovers from that change in direction.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest dead work is the unused `compute_df_f` function, plus loading `F.npy` and `Fneu.npy` solely to support that abandoned path. The code also computes `_t_b_raw` from `tstamps.npy` and discards it immediately. In `plot_processing`, the labels still describe "raw neural dF/F" even though the plotted neural signal is the spike-derived signal from `spks.npy`.

ii.
```python
def compute_df_f(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile = float(ops.get('prctile_baseline', 8.0))
    baseline = np.percentile(Fc, prctile, axis=1, keepdims=True).astype(np.float32)
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fc - baseline) / baseline
    return dff.astype(np.float32)

...

_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)

...

axes[1].set_title('raw neural dF/F example cell (first 2000 frames)')
axes[3].set_title('binned neural dF/F example cell')
```

iii. The trajectory makes this understandable: the AI explicitly changed from a dF/F-like signal to `spks.npy`, and from raw timestamps to frame-rate-derived elapsed time, after debugging decoder performance and input scaling. The final code keeps pieces of the earlier discarded approach.
