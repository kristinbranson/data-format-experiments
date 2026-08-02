# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script starts from `data/`, treats every top-level directory as a subject, then treats every subdirectory inside each subject as a session. For each session it loads Suite2p fluorescence files (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) and behavior files (`motion_energy_glob.npy`, `tstamps.npy`). Trials are not loaded directly from raw data; they are created later by block-splitting the continuous recordings.

ii. 
```python
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
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
```

iii. In `CONVERSION_NOTES.md`, the AI says the raw dataset is organized as subject folders containing session folders, each with `suite2p/plane0` and `move_deve/`. It also notes that the data are continuous recordings with no native trial structure, so later “trialization” must be imposed during conversion.

## 1-b. How are the data split into subjects?

i. Subjects are defined as all top-level directories under `data/`, sorted alphabetically. The code does not restrict subjects to names starting with `jm`; it assumes every directory in `data/` is a mouse.

ii. 
```python
root = Path('data')
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
```

iii. The AI's notes say that `data/` contains one folder per subject, with examples such as `jm031`, and its “Final Understanding” states that one subject corresponds to one top-level folder.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories inside each subject directory, sorted alphabetically. Each session becomes one entry in the output dataset before later block-splitting into pseudo-trials.

ii. 
```python
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)
```

iii. The AI's notes describe each session/day folder as containing `suite2p/plane0/` and `move_deve/`, and treat each such day folder as one recording session.

## 1-d. How are the data split into trials?

i. The AI decides that the continuous recordings should be segmented into consecutive 2-minute pseudo-trials after temporally averaging into non-overlapping 10-frame bins. Each binned session is split into fixed-size blocks; leftover bins at the end are discarded.

ii. 
```python
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
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly justifies 2-minute blocks by citing the paper’s decoding analyses and says this choice “creates at least 10 pseudo-trials per 20-minute session.” The notes also call the raw data “continuous recordings rather than explicit trials.”

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Instead, the script drops incomplete trailing bins when making 2-minute blocks, and it skips entire sessions that produce fewer than two blocks.

ii. 
```python
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

iii. The AI’s notes do not describe any trial-wise QC rule beyond the general decoder-format requirement that there be at least two trials per session. Its curation focus is on neuron filtering and continuous-to-block segmentation, not trial rejection.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural signal is derived primarily from `F.npy` and `Fneu.npy`, with `ops.npy` providing baseline-related parameters and `iscell.npy` determining which ROIs are retained.

ii. 
```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The AI’s notes say it chose fluorescence-based traces rather than `spks.npy`, because the methods text says later analyses used baseline-corrected fluorescence as dF/F. It also says `iscell[:,1] > 0.5` matches both the methods and the Track2p code it inspected.

## 2-b. How is the `neural` data processed?

i. Neural processing consists of neuropil subtraction followed by a simplified dF/F computation: subtract `neucoeff * Fneu`, estimate a single per-neuron low-percentile baseline, then compute `(Fc - F0) / F0`. After that, the neural traces are averaged in non-overlapping 10-frame bins.

ii. 
```python
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

iii. The AI’s notes say it wanted to match “Suite2p-style fluorescence signal” but replaced a slower sliding baseline with “a fast per-neuron low-percentile baseline approximation” so the script would run quickly on full sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons by keeping only ROIs with `iscell[:,1] > 0.5`. No additional neuron QC is applied after this filter.

ii. 
```python
iscell = np.load(suite / 'iscell.npy')
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. In `CONVERSION_NOTES.md`, the AI repeatedly states that this threshold matches the methods text and Track2p defaults, and calls it a “reference-consistent” curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns the neural data to session start, not to an external event. After binning, each pseudo-trial is just a contiguous 2-minute segment cut from the session-long trace, so alignment is inherited from the continuous session time base.

ii. 
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins
)
```

```python
'metadata': {
    'temporal_alignment_event': 'session start',
    'off_start': 0.0,
    'off_end': 120.0,
}
```

iii. The AI’s notes say the recordings are continuous, video was triggered by microscope acquisition, and there is no natural trial event, so session-start alignment plus consecutive blocks is the appropriate choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted dataset uses 10 raw frames per bin at 30 Hz, so each time bin is about 333.3 ms. Yes, the code applies temporal rebinning by averaging each non-overlapping 10-frame chunk.

ii. 
```python
bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs
```

```python
def bin_time_series(x, bin_size):
    T = x.shape[-1]
    T2 = (T // bin_size) * bin_size
    x = x[..., :T2]
    new_shape = x.shape[:-1] + (T2 // bin_size, bin_size)
    return x.reshape(new_shape).mean(axis=-1)
```

iii. The AI’s notes justify this by quoting the paper’s statement that decoding analyses averaged both dF/F and behavior into bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. In the final code, the time input is not derived from any stored raw timestamp variable. It is reconstructed from frame index and the assumed 30 Hz frame rate. The script still loads `tstamps.npy`, but does not use it to build the final decoder input.

ii. 
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
...
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. The trajectory shows that the AI first tried to use `tstamps.npy`, discovered that it did not span seconds as expected, and then changed to frame index divided by 30 Hz. Its notes say that this is “consistent with the methods.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code creates a frame-by-frame time vector from `0, 1/raw_fs, 2/raw_fs, ...`, then averages it in the same non-overlapping 10-frame bins used for neural and motion data. The result is a session-continuous time series, not a within-block reset.

ii. 
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

```python
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The AI’s notes explicitly say the decoder input should be “elapsed seconds from the beginning of the session within each block,” and the trajectory later says the full-session range is appropriate because the task asked for time from the beginning of the experiment.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is built from the same frame count as the neural traces, binned with the same 10-frame averaging, truncated to the same multiple of the block size, and sliced into the same block boundaries.

ii. 
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

```python
dff_binned = bin_time_series(dff, bin_size_frames)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins
)
```

iii. The AI’s notes say neural and motion were synchronized framewise and that time should share the same common binned time base, so the implementation uses identical trimming, binning, and block slicing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion signal is taken from `move_deve/motion_energy_glob.npy`. The script also loads `tstamps.npy` so it can truncate all streams to a common minimum length, but it does not use `interframe_int.npy`.

ii. 
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
...
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
```

iii. In the notes, the AI describes `motion_energy_glob.npy` as the relevant behavioral stream and emphasizes framewise synchronization with neural data. It chose not to rely on `tstamps.npy` for the decoder input after inspecting its scale.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code truncates motion energy to a common shared length with neural and timestamps, averages it in non-overlapping 10-frame bins, stores those continuous binned values session by session, and later discretizes them using global percentile thresholds pooled across all retained sessions. It does not normalize by standard deviation and does not interpolate dropped frames.

ii. 
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
...
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
...
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
...
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
```

iii. The AI’s notes say the paper averaged behavior traces in 10-frame bins and that the output should be discretized into 5 equal-percentile bins computed globally over the full converted dataset.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes global 20th/40th/60th/80th percentile edges over all retained binned motion-energy values, then labels each time bin with `np.digitize`, clipping labels to the range 0 to 4.

ii. 
```python
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
```

```python
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
out_trials.append(labels[None, :])
```

iii. The AI’s notes explicitly justify five global equal-percentile bins as a way to satisfy the decoder-output specification while keeping categories balanced.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by truncating all streams to the same minimum raw length, averaging motion and neural traces with the same 10-frame binning, and slicing them into identical 2-minute blocks.

ii. 
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

```python
dff_binned = bin_time_series(dff, bin_size_frames)
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins
)
```

iii. The AI’s notes say that video was triggered by the microscope so streams should be framewise aligned, and its “Final Understanding” says to align them frame-by-frame and then apply the paper’s 10-frame averaging.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles mismatches conservatively by truncating neural, motion, and timestamp arrays to the shared minimum length. It also discards incomplete trailing bins when block-splitting and skips sessions with fewer than two resulting blocks. It does not repair dropped motion frames or assert exact expected lengths.

ii. 
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

```python
T2 = n_blocks * block_bins
dff_binned = dff_binned[:, :T2]
motion_binned = motion_binned[:T2]
time_binned = time_binned[:T2]
...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The AI’s notes describe synchronization and block-based segmentation, but they do not mention a dropped-frame repair procedure. The trajectory instead emphasizes practical robustness and fast execution, which matches the choice to trim everything to a common usable extent.

## 6-a. What are the most time-consuming steps of the code?

i. According to the AI’s notes and trajectory, the most time-consuming part was the original attempt at sliding-window baseline estimation for dF/F. The final script keeps a much faster constant-percentile approximation, so the remaining costly steps are session loading and dF/F computation per session.

ii. 
```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile_baseline = float(ops.get('prctile_baseline', 8.0))
    F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
    ...
```

```python
loaded = load_session(sess_path)
dff = loaded['dff']
motion = loaded['motion']
```

iii. `CONVERSION_NOTES.md` says “Initial sliding-window percentile baseline estimation was too slow for 36,000-frame sessions” and that the AI replaced it with a faster approximation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still uses Python loops where vectorization was possible: the loop over blocks in `make_blocks`, the nested loop that digitizes output trials, and the per-session outer loop. The AI’s notes also mention that an earlier sliding-window baseline loop was too slow and had to be removed.

ii. 
```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
    input_trials.append(time_binned[sl][None, :].astype(np.float32))
    output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

```python
all_output = []
for sess_trials in all_output_cont:
    out_trials = []
    for arr in sess_trials:
        x = arr.squeeze(0)
        labels = np.digitize(x, edges, right=False).astype(np.int64)
        labels = np.clip(labels, 0, 4)
        out_trials.append(labels[None, :])
    all_output.append(out_trials)
```

iii. The AI’s notes frame performance as a major concern and explicitly describe replacing a slower baseline method with a faster approximation; the remaining loops are consistent with that tradeoff-driven design.

## 6-c. What processing does the code repeat multiple times?

i. The script repeatedly trims, casts, bins, and slices related time series. It bins neural, motion, and time separately with the same helper, then loops again to split them into blocks, and then loops again to digitize motion values after first collecting them continuously.

ii. 
```python
dff_binned = bin_time_series(dff, bin_size_frames)
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins
)
```

```python
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
...
for sess_trials in all_output_cont:
    out_trials = []
    for arr in sess_trials:
        ...
```

iii. This follows the AI’s two-stage design from the notes: first generate continuous/binned session-level traces, then compute global motion percentiles and only afterward produce categorical outputs.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is loading `tstamps.npy`, truncating it, and returning it from `load_session`, even though the final decoder input is built from frame index instead. The function also returns `keep` and `ops`, but `main()` does not use them after the dF/F computation is finished.

ii. 
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
...
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
...
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
```

iii. The trajectory shows the AI originally tried to use `tstamps.npy`, found that it was unsuitable for seconds-from-start, and switched to frame-based time. The final code still carries timestamp loading and truncation even though downstream analyses no longer use that array.
