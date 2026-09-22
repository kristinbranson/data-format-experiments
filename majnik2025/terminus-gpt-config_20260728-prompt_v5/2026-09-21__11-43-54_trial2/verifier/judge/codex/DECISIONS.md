# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subjects by scanning `/app/data` for directories whose names start with `jm`, then scans every subject directory for session subdirectories. For each session it loads Suite2p calcium files and motion files. Trial structure is not loaded directly from disk; it is created later from the continuous session arrays.

ii.
```python
def discover_sessions(data_root):
    data_root = Path(data_root)
    sessions = []
    for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            plane = sess_dir / 'suite2p' / 'plane0'
            move = sess_dir / 'move_deve'
            if plane.exists() and move.exists():
                sessions.append({
                    'subject': subj_dir.name,
                    'session': sess_dir.name,
                    'session_dir': sess_dir,
                    'plane_dir': plane,
                    'move_dir': move,
                })
    return sessions

def load_session_arrays(sess):
    plane = sess['plane_dir']
    move = sess['move_dir']
    F = np.load(plane / 'F.npy').astype(np.float32)
    Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
    iscell = np.load(plane / 'iscell.npy')
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
    interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
    return F, Fneu, iscell, ops, motion, tstamps, interframe
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI says the dataset is organized as `jm*` subject folders with per-session `suite2p/plane0` and `move_deve` subdirectories, and that all such sessions should be used.

## 1-b. How are the data split into subjects?

i. Subjects are defined as the top-level directories in `/app/data` whose names begin with `jm`, sorted lexicographically.

ii.
```python
for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
```

iii. In Step 2 of `CONVERSION_NOTES.md`, the AI identifies the six subject folders `jm031, jm032, jm038, jm039, jm040, jm046` and treats each as one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are defined as subject subdirectories, again sorted lexicographically. A directory is only accepted as a session if it contains both `suite2p/plane0` and `move_deve`.

ii.
```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    plane = sess_dir / 'suite2p' / 'plane0'
    move = sess_dir / 'move_deve'
    if plane.exists() and move.exists():
        sessions.append({
            'subject': subj_dir.name,
            'session': sess_dir.name,
            'session_dir': sess_dir,
            'plane_dir': plane,
            'move_dir': move,
        })
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI describes each dated subdirectory as one daily recording session and notes that the dataset contains 41 such sessions.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions with no native trial structure. It bins the time series into non-overlapping 10-frame averages, then cuts the binned session into non-overlapping 60-second trials. Each trial has `180` bins at `30 Hz / 10 = 3 Hz`. Any leftover tail shorter than a full trial is discarded. Sessions with fewer than two full trials would be dropped.

ii.
```python
trial_len_bins = int(round(60.0 / (bin_size / float(ops.get('fs', 30.0)))))
neural_trials, input_trials, output_trials = split_into_trials(neural_b, inp, out, trial_len_bins)

def split_into_trials(neural, inp, out, trial_len_bins):
    n_time = neural.shape[1]
    n_trials = n_time // trial_len_bins
    if n_trials < 2:
        return [], [], []
    keep = n_trials * trial_len_bins
    neural = neural[:, :keep]
    inp = inp[:, :keep]
    out = out[:, :keep]
    neural_trials = [neural[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.float32) for i in range(n_trials)]
    input_trials = [inp[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.float32) for i in range(n_trials)]
    output_trials = [out[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.int64) for i in range(n_trials)]
    return neural_trials, input_trials, output_trials
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI explicitly says there are no native trials and that continuous sessions must be split into 60 s windows for the decoder task.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit per-trial quality-control filter. The only effective filtering is structural: incomplete trailing bins are discarded when a session is cut into full 60-second windows, and a session would be skipped if it produced fewer than two full trials.

ii.
```python
if n_trials < 2:
    return [], [], []
keep = n_trials * trial_len_bins
neural = neural[:, :keep]
inp = inp[:, :keep]
out = out[:, :keep]
```

iii. In `CONVERSION_NOTES.md` Step 3 and Step 5, the AI notes that the source data have no native trials, so trial curation mainly reduces to handling missing behavior frames and creating fixed-length windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from `F.npy` and `Fneu.npy`, using `ops.npy` for parameters such as `fs`, `neucoeff`, and `prctile_baseline`. Although it loads `iscell.npy`, it does not use it in the neural transform.

ii.
```python
F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
iscell = np.load(plane / 'iscell.npy')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
...
neural = compute_dff(
    F, Fneu,
    neuropil_coeff=float(ops.get('neucoeff', 0.7)),
    baseline_percentile=float(ops.get('prctile_baseline', 8.0))
)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps `suite2p/plane0/F.npy (+ Fneu.npy, ops.npy)` to the `neural` field and justifies using fluorescence-based signals rather than `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction and then computes an approximate dF/F signal using a single per-neuron low-percentile baseline. It does not call Suite2p's `dcnv.preprocess`; instead it uses a simpler approximation and then averages the result into 10-frame bins.

ii.
```python
def compute_dff(F, Fneu, neuropil_coeff=0.7, baseline_percentile=8.0):
    Fc = F - neuropil_coeff * Fneu
    baseline = np.percentile(Fc, baseline_percentile, axis=1, keepdims=True).astype(np.float32)
    baseline = np.where(np.abs(baseline) < 1e-3, 1e-3, baseline)
    dff = (Fc - baseline) / np.abs(baseline)
    return dff.astype(np.float32)

...
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI says the paper used baseline-corrected fluorescence as dF/F but no explicit dF/F file exists, so it chose to "reconstruct an approximate Suite2p-style baseline-corrected fluorescence signal" from `F` and `Fneu`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no neuron filtering in the conversion script. It loads `iscell.npy` but does not use it. All rows in `F.npy` are kept.

ii.
```python
F, Fneu, iscell, ops, motion, tstamps, interframe = load_session_arrays(sess)
...
brain_region_idx.append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI argues that the Track2p outputs already contain tracked neurons and that all provided ROIs pass the Suite2p `iscell > 0.5` criterion, so no extra filtering should be applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to session start, not to any stimulus or behavior event. Each trial is a contiguous 60-second segment of the session after 10-frame binning.

ii.
```python
'metadata': {
    'task_description': 'Decode spontaneous animal motion energy from barrel cortex calcium activity in continuous sessions split into 60-second trials.',
    'time_bin_size': 1000.0 * (10.0 / 30.0),
    'temporal_alignment_event': 'session start',
    'off_start': 0.0,
    'off_end': 60.0,
    ...
}
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI describes the data as continuous spontaneous-behavior recordings and states that 60 s windows should be created from session start because there is no native event-based trial structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping mean bins. With `fs = 30 Hz`, this gives a bin width of `10 / 30 = 0.333... s`, i.e. `333.33 ms`.

ii.
```python
bin_size = 10
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
...
'time_bin_size': 1000.0 * (10.0 / 30.0),
```

iii. In `CONVERSION_NOTES.md` Step 3 and Step 5, the AI cites the paper's statement that neural and behavior traces were averaged in bins of 10 consecutive timestamps and says it is following that denoising step.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI derives the decoder input time from `tstamps.npy` when those timestamps are present and plausible after conversion from days to seconds. Otherwise it falls back to frame index and `ops['fs']`.

ii.
```python
def get_neural_frame_times_sec(n_frames, ops, tstamps=None):
    fs = float(ops.get('fs', 30.0))
    if tstamps is not None and len(tstamps) >= n_frames:
        dt = np.diff(tstamps[:n_frames])
        if len(dt) > 0:
            dt_sec = float(np.median(dt) * 86400.0)
            if 0.5 / fs < dt_sec < 1.5 / fs:
                t = (tstamps[:n_frames] - tstamps[0]) * 86400.0
                return np.asarray(t, dtype=np.float32)
    return (np.arange(n_frames, dtype=np.float32) / fs).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI says `tstamps.npy` appears to use day-like units and should be converted with `86400`, while `ops['fs']=30` is used as a cross-check and fallback.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI converts timestamps from day units to seconds when possible, or computes time from frame number and frame rate otherwise. It then bins the resulting time vector with the same 10-frame averaging used for neural and motion data, and stores it as a 1 x time array per trial.

ii.
```python
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
inp = time_b[None, :].astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly plans to "create time-elapsed-in-seconds vector per binned sample, reset at session start" and to bin it in the same way as the other streams.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction: the input time vector is computed on the same frame grid as the neural data, binned with the same 10-frame bins, and cut into the same trial windows.

ii.
```python
neural_times_sec = get_neural_frame_times_sec(n_frames, ops, tstamps)
...
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
...
neural_trials, input_trials, output_trials = split_into_trials(neural_b, inp, out, trial_len_bins)
```

iii. In `CONVERSION_NOTES.md` Step 10, the AI states that a raw-data sanity check reproduced the converted `time_elapsed_sec` values exactly for one session.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy primarily from `motion_energy_glob.npy` plus `tstamps.npy` for alignment. It also loads `interframe_int.npy`, but the conversion code does not use it in the final alignment logic.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
...
motion_aligned = align_motion_to_neural(motion, tstamps, neural_times_sec, ops)
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI says missing camera frames can be handled using timestamps or interframe intervals, and it chose timestamp-based alignment/interpolation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates the motion trace onto the neural time grid using timestamps when possible. If not, it truncates or rescales the motion trace to match neural length. It then averages the aligned trace into 10-frame bins and discretizes the binned values into 5 quantile bins within each session.

ii.
```python
def align_motion_to_neural(motion, tstamps, neural_times_sec, ops):
    fs = float(ops.get('fs', 30.0))
    n_motion = len(motion)
    if len(tstamps) >= n_motion:
        mt = (tstamps[:n_motion] - tstamps[0]) * 86400.0
        good = np.isfinite(mt) & np.isfinite(motion)
        mt = mt[good]
        mv = motion[good]
        if len(mt) >= 2 and np.all(np.diff(mt) > 0):
            return np.interp(neural_times_sec, mt.astype(np.float32), mv.astype(np.float32)).astype(np.float32)
    motion = motion[: min(len(motion), len(neural_times_sec))]
    if len(motion) == len(neural_times_sec):
        return motion.astype(np.float32)
    x_old = np.linspace(0, 1, num=len(motion), dtype=np.float32)
    x_new = np.linspace(0, 1, num=len(neural_times_sec), dtype=np.float32)
    return np.interp(x_new, x_old, motion.astype(np.float32)).astype(np.float32)

motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
motion_disc, edges = discretize_into_quantile_bins(motion_b, n_bins=5)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies timestamp-based interpolation by pointing to missing camera frames and saying this preserves neural data rather than dropping samples.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes each session's binned motion values into 5 equal-percentile bins. It uses `np.quantile` to get the edges, forces the first and last edges to `-inf` and `inf`, and perturbs ties upward with `np.nextafter` so `np.digitize` produces valid labels.

ii.
```python
def discretize_into_quantile_bins(values, n_bins=5):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    labels = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    return labels, edges
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says the decoder output should be "5 session-specific quantile bins after alignment and binning" to satisfy the task while preserving session-specific distributions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural data by interpolation onto the neural frame-time grid. If timestamps are unavailable or insufficient, it falls back to truncation or a full-length resampling based on normalized position. After alignment it bins motion and neural data with the same 10-frame bins and slices them into the same trials.

ii.
```python
neural_times_sec = get_neural_frame_times_sec(n_frames, ops, tstamps)
motion_aligned = align_motion_to_neural(motion, tstamps, neural_times_sec, ops)
...
motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
...
neural_trials, input_trials, output_trials = split_into_trials(neural_b, inp, out, trial_len_bins)
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI notes that some sessions have dropped behavior frames and that interpolation onto neural frame times is its chosen way to handle that mismatch.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles motion/neural length mismatches by interpolation with timestamps when possible, and otherwise by truncating or resampling the motion trace to the neural length. It also prevents division by near-zero baselines in the neural transform by clamping the baseline magnitude to `1e-3`. Trailing bins that do not fill a full trial are implicitly dropped.

ii.
```python
baseline = np.where(np.abs(baseline) < 1e-3, 1e-3, baseline)
...
if len(tstamps) >= n_motion:
    ...
    return np.interp(neural_times_sec, mt.astype(np.float32), mv.astype(np.float32)).astype(np.float32)
motion = motion[: min(len(motion), len(neural_times_sec))]
if len(motion) == len(neural_times_sec):
    return motion.astype(np.float32)
x_old = np.linspace(0, 1, num=len(motion), dtype=np.float32)
x_new = np.linspace(0, 1, num=len(neural_times_sec), dtype=np.float32)
return np.interp(x_new, x_old, motion.astype(np.float32)).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI explicitly flags missing camera frames as a known data issue and chooses interpolation rather than dropping the affected neural samples.

## 6-a. What are the most time-consuming steps of the code?

i. The AI does not explicitly name a single bottleneck in its notes, but its implementation spends most work on full-session array operations: percentile-based baseline estimation for all neurons, motion interpolation/resampling, and 10-frame binning. The notes also mention full-session array loading as a resource concern.

ii.
```python
baseline = np.percentile(Fc, baseline_percentile, axis=1, keepdims=True).astype(np.float32)
...
return np.interp(neural_times_sec, mt.astype(np.float32), mv.astype(np.float32)).astype(np.float32)
...
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
```

iii. The clearest explicit justification is in `CONVERSION_NOTES.md` Step 6, where the AI says full-session fluorescence arrays are loaded into memory and that it tried to keep the main numerical work vectorized.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized most of the numerically heavy steps. The remaining Python loops are mostly outer control-flow loops over subjects/sessions and list-based trial materialization. The code no longer has a per-dropped-frame insertion loop; it uses vectorized interpolation instead.

ii.
```python
for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        ...

neural_trials = [neural[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.float32) for i in range(n_trials)]
input_trials = [inp[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.float32) for i in range(n_trials)]
output_trials = [out[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.int64) for i in range(n_trials)]
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI explicitly says it "vectorized neuropil correction, percentile baseline estimation, binning, interpolation, and trial segmentation."

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated recomputation across sessions in the main pipeline. Each session is processed once. Minor repetition remains in applying the same binning and slicing logic separately to neural, motion, and time arrays, and in building trial lists with separate comprehensions for each stream.

ii.
```python
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
...
neural_trials = [ ... for i in range(n_trials)]
input_trials = [ ... for i in range(n_trials)]
output_trials = [ ... for i in range(n_trials)]
```

iii. The AI does not call out repeated processing as a concern in its notes. The Step 6 notes instead emphasize that the main transformations were implemented once per session and kept vectorized.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is loading `iscell.npy` and `interframe_int.npy` without using them in the final computation. Optional plotting for `--show-processing` is also extra work that is not used downstream by the decoder.

ii.
```python
iscell = np.load(plane / 'iscell.npy')
...
interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
return F, Fneu, iscell, ops, motion, tstamps, interframe
...
if show_processing:
    import matplotlib.pyplot as plt
    ...
    fig.savefig(outpng, dpi=150)
```

iii. The notes justify the plots as a sanity-check aid, but they do not justify keeping `iscell` and `interframe` in the runtime path once the final processing no longer uses them.
