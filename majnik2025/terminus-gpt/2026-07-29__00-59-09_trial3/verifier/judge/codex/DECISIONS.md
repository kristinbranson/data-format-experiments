# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every top-level directory under `data/` as a subject, then every subdirectory under each subject as a session. For each session it loads Suite2p neural files from `suite2p/plane0/` and behavior files from `move_deve/`. Trials are not loaded from disk; they are created later by splitting each continuous session into fixed blocks.

ii. ```python
root = Path('data')
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
session_paths = []
session_subjects = []
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)
```

```python
suite = sess_path / 'suite2p' / 'plane0'
move = sess_path / 'move_deve'
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
```

iii. In `CONVERSION_NOTES.md`, the AI says the raw dataset is organized as one folder per subject, each containing session/day folders with `suite2p/plane0` and `move_deve` subfolders. The notes say conversion should therefore load suite2p neural arrays and motion-energy behavior arrays per session.

## 1-b. How are the data split into subjects?

i. Subjects are defined as the sorted list of all directories directly under `data/`. The AI does not restrict them to names beginning with `jm`; it assumes every directory there is a subject.

ii. ```python
root = Path('data')
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI notes that `data/` contains one folder per subject, with names like `jm031`. Its subject split follows that observed directory structure.

## 1-c. How are the data split into sessions?

i. Sessions are defined as the sorted subdirectories within each subject directory. Each such folder is treated as one session.

ii. ```python
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI describes each subject folder as containing multiple session/day folders such as `2023-10-23_a`, and uses that one-folder-per-recording convention directly.

## 1-d. How are the data split into trials?

i. The AI does not use the instruction’s 60-second trials. It bins each continuous session into 10-frame time bins, then splits the binned session into consecutive non-overlapping 2-minute blocks. Those blocks become the trials.

ii. ```python
bin_size_frames = 10
raw_fs = 30.0
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)  # 2 minutes after binning
```

```python
def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    T2 = n_blocks * block_bins
    dff_binned = dff_binned[:, :T2]
    motion_binned = motion_binned[:T2]
    time_binned = time_binned[:T2]
    neural_trials, input_trials, output_cont = [], [], []
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
        neural_trials.append(dff_binned[:, sl].astype(np.float32))
        input_trials.append(time_binned[sl][None, :].astype(np.float32))
        output_cont.append(motion_binned[sl][None, :].astype(np.float32))
    return neural_trials, input_trials, output_cont
```

iii. In `CONVERSION_NOTES.md` Steps 3-5 and the trajectory, the AI argues that the paper’s decoding analyses used consecutive 2-minute blocks, so pseudo-trials should follow that block structure rather than 60-second windows.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Instead, the AI drops any leftover tail shorter than a full 2-minute block and skips sessions that produce fewer than two blocks.

ii. ```python
T = dff_binned.shape[1]
n_blocks = T // block_bins
T2 = n_blocks * block_bins
dff_binned = dff_binned[:, :T2]
motion_binned = motion_binned[:T2]
time_binned = time_binned[:T2]
```

```python
neural_trials, input_trials, output_cont = make_blocks(...)
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The trajectory and notes state that the decoder requires at least two trials per session, so the AI used “fewer than 2 blocks” as a session-level exclusion rule. No other trial-quality screen is described.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from `F.npy` and `Fneu.npy` from Suite2p `plane0`, using `ops.npy` for parameters and `iscell.npy` to decide which ROIs to keep.

ii. ```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says the methods imply fluorescence-based traces should be used rather than deconvolved spikes, with `iscell[:,1] > 0.5` matching the paper’s cell-inclusion criterion.

## 2-b. How is the `neural` data processed?

i. The AI computes an approximate dF/F: neuropil subtraction first, then a single per-neuron low-percentile baseline, then `(Fc - F0) / F0`. After that, it averages the resulting traces into non-overlapping 10-frame bins.

ii. ```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile_baseline = float(ops.get('prctile_baseline', 8.0))
    F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
    F0 = np.maximum(F0, 1e-3)
    dff = (Fc - F0) / F0
    return dff.astype(np.float32)
```

```python
dff_binned = bin_time_series(dff, bin_size_frames)
```

iii. In `CONVERSION_NOTES.md` Step 6 and the trajectory, the AI says it wanted a “practical reference-consistent approach” to Suite2p-style baseline-corrected dF/F, but replaced a slower moving-baseline implementation with this faster approximation because the original approach was too slow.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters ROIs by keeping only rows where the second column of `iscell.npy` exceeds `0.5`. No additional neural-quality filters are applied after that.

ii. ```python
iscell = np.load(suite / 'iscell.npy')
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. `CONVERSION_NOTES.md` repeatedly states that the paper and Track2p reference code use an `iscell` threshold of `0.5`, so the AI carried that rule into the conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are not aligned to an external task event. The AI keeps the session’s chronological order, bins in place, and chops the session into consecutive blocks. In metadata it labels the alignment event as `session start`, with `off_start = 0.0` and `off_end = 120.0`.

ii. ```python
neural_trials, input_trials, output_cont = make_blocks(dff_binned, motion_binned, time_binned, block_bins)
```

```python
'metadata': {
    'temporal_alignment_event': 'session start',
    'off_start': 0.0,
    'off_end': 120.0,
    ...
}
```

iii. In `CONVERSION_NOTES.md` Steps 4-5, the AI says the recordings are spontaneous continuous sessions with no natural stimulus event, so block-based segmentation is the closest analogue for decoder trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping bins at a 30 Hz raw frame rate, giving a bin size of about 333.33 ms. The AI applies this rebinning to neural data, motion energy, and time.

ii. ```python
bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs
```

```python
dff_binned = bin_time_series(dff, bin_size_frames)
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. In `CONVERSION_NOTES.md` Step 3, the AI cites the methods statement that both dF/F and behavior traces were averaged in bins of 10 consecutive timestamps for decoding analyses.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The final time input is not derived from `tstamps.npy`. Instead, it is derived from the frame index and the assumed 30 Hz acquisition rate. `tstamps.npy` is loaded, but only used when trimming all arrays to a common minimum length.

ii. ```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. The trajectory says the AI first tried to use `tstamps.npy`, then concluded it was “not in seconds and unsuitable as the decoder input,” and switched to `frame index / 30 Hz`, which it considered more consistent with the methods text.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI creates a frame-time vector in seconds from `np.arange(...) / 30`, averages it with the same 10-frame binning used elsewhere, then slices the binned time vector into the same fixed blocks as the neural data.

ii. ```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The trajectory says the time input should represent elapsed time from session start, and after the `tstamps` bug fix the AI accepted the full-session range `[0.2, 1199.8]` or `[0.2, 1799.8]` as correct.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: the AI trims all arrays to a common length, bins them with the same helper, and slices the same block ranges for `neural` and `input`.

ii. ```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

```python
dff_binned = bin_time_series(dff, bin_size_frames)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(dff_binned, motion_binned, time_binned, block_bins)
```

iii. The AI’s notes emphasize a shared time base for neural and behavior streams. The trajectory also shows it explicitly debugged the time trace until the temporal range matched full-session elapsed time.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived primarily from `move_deve/motion_energy_glob.npy`. `tstamps.npy` is loaded only to help enforce a common truncated length with the neural traces; `interframe_int.npy` is not used.

ii. ```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps `motion_energy_glob.npy` to the decoder output and treats `tstamps.npy` as part of the alignment information.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI converts motion energy to `float32`, truncates it to a common length with the neural data, averages it into non-overlapping 10-frame bins, slices it into the same consecutive blocks as the neural data, and only later discretizes it.

ii. ```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(dff_binned, motion_binned, time_binned, block_bins)
```

iii. In `CONVERSION_NOTES.md`, the AI says motion energy should be aligned framewise to neural data, averaged in the same 10-frame bins as the dF/F, and then discretized for decoding.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses one global set of percentile thresholds computed from the concatenated motion-energy values across all sessions. It uses the 20th, 40th, 60th, and 80th percentiles, then converts each binned value into an integer label from 0 to 4.

ii. ```python
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
```

```python
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
out_trials.append(labels[None, :])
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says it chose full-dataset percentile thresholds because it wanted to “avoid per-session label drift” while keeping the overall output distribution balanced.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion with neural data by truncating all streams to the minimum common raw length, applying the same 10-frame binning, and slicing the same block boundaries for all modalities.

ii. ```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

```python
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(dff_binned, motion_binned, time_binned, block_bins)
```

iii. In `CONVERSION_NOTES.md` Step 4, the AI says the modalities appear length-matched and video was triggered by the microscope, so framewise alignment followed by common binning should be sufficient.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles length mismatches by truncating all modalities to the minimum available length. It also silently drops any tail shorter than a full block and skips sessions with fewer than two resulting blocks. It does not interpolate missing camera frames.

ii. ```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

```python
n_blocks = T // block_bins
T2 = n_blocks * block_bins
dff_binned = dff_binned[:, :T2]
motion_binned = motion_binned[:T2]
time_binned = time_binned[:T2]
```

```python
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The notes and trajectory frame this as a pragmatic way to keep modalities aligned and satisfy the decoder’s “at least two trials” requirement. The AI does not document any stronger missing-data treatment than truncation.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified dF/F baseline estimation as the main bottleneck during development. In the final script, the remaining expensive steps are session I/O plus `compute_dff`, especially the percentile baseline computation over every neuron’s full trace.

ii. ```python
F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
dff = (Fc - F0) / F0
```

```python
for i, (sess_path, subj) in enumerate(zip(session_paths, session_subjects)):
    loaded = load_session(sess_path)
    dff = loaded['dff']
    motion = loaded['motion']
    ...
```

iii. `CONVERSION_NOTES.md` Step 6 says an initial sliding-window percentile baseline was “too slow for 36,000-frame sessions,” and the trajectory records that this bottleneck motivated a simpler approximation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit per-block loop in `make_blocks` and the nested loop used to discretize each session’s output trials could both have been vectorized or restructured to reduce Python overhead.

ii. ```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
    input_trials.append(time_binned[sl][None, :].astype(np.float32))
    output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

```python
for sess_trials in all_output_cont:
    out_trials = []
    for arr in sess_trials:
        x = arr.squeeze(0)
        labels = np.digitize(x, edges, right=False).astype(np.int64)
        labels = np.clip(labels, 0, 4)
        out_trials.append(labels[None, :])
    all_output.append(out_trials)
```

iii. The AI’s notes focus more on the removed slow baseline loop than on these remaining loops, but the current implementation still uses straightforward Python iteration for block construction and label assignment.

## 6-c. What processing does the code repeat multiple times?

i. The code first stores continuous binned motion traces in `all_output_cont`, then concatenates them to compute one global threshold set, then iterates over those traces again to discretize them into final categorical outputs. It also repeatedly looks up `subjects.index(subj)` inside the session loop.

ii. ```python
all_output_cont.append(output_cont)
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
...
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
...
for sess_trials in all_output_cont:
    ...
        labels = np.digitize(x, edges, right=False).astype(np.int64)
```

```python
subject_idx.append(subjects.index(subj))
```

iii. This repeated motion pass follows directly from the AI’s decision to compute one global set of motion-energy thresholds after all sessions have already been processed.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` even though the final decoder input is computed from frame index rather than timestamps. It also returns `keep` and `ops` from `load_session()` even though downstream code only uses `dff`, `motion`, and `tstamps`, and it keeps continuous motion traces in `all_output_cont` only to replace them with categorical labels later.

ii. ```python
return {
    'dff': dff,
    'motion': motion,
    'tstamps': tstamps,
    'keep': keep,
    'ops': ops,
}
```

```python
loaded = load_session(sess_path)
dff = loaded['dff']
motion = loaded['motion']
tstamps = loaded['tstamps']
```

```python
all_output_cont.append(output_cont)
...
for sess_trials in all_output_cont:
    ...
all_output.append(out_trials)
```

iii. The trajectory shows that `tstamps.npy` was kept after an earlier failed attempt to use it as the decoder input. The final code no longer uses timestamps as the actual time signal, so that load is mostly retained for trimming/alignment only.
