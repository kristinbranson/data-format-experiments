# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for all top-level directories, treats each as a subject, scans all subdirectories as sessions, and loads per-session Suite2p neural files plus motion/timestamp files. It then converts each continuous session into block-based pseudo-trials later in the pipeline.

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

iii. In `CONVERSION_NOTES.md`, the AI says the raw data are organized as subject folders containing session folders with `suite2p/plane0/` and `move_deve/`, and that these are the relevant sources to load. In the trajectory it justified this as matching the observed directory structure and Track2p/Suite2p conventions.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories under `data/`, sorted alphabetically. The AI does not restrict them to names starting with `jm`; it assumes every top-level directory is a subject.

ii.
```python
root = Path('data')
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
```

iii. In the notes, the AI documented that `data/` contains one folder per subject, with examples like `jm031`, and used that observed layout as justification.

## 1-c. How are the data split into sessions?

i. Each subject's immediate subdirectories are treated as sessions, sorted alphabetically. One session corresponds to one daily recording.

ii.
```python
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)
```

iii. The notes say each subject folder contains multiple session/day folders, each holding one recording with `suite2p` and `move_deve` data.

## 1-d. How are the data split into trials?

i. The AI treats each continuous session as a sequence of consecutive 2-minute pseudo-trials after temporal averaging into 10-frame bins. It keeps only complete 2-minute blocks.

ii.
```python
bin_size_frames = 10
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)  # 2 minutes after binning
```

```python
def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    T2 = n_blocks * block_bins
    ...
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
        neural_trials.append(dff_binned[:, sl].astype(np.float32))
        input_trials.append(time_binned[sl][None, :].astype(np.float32))
        output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

iii. In `CONVERSION_NOTES.md` Step 3-5 and the trajectory, the AI justified this from the methods text: decoding analyses used 10-frame averaging and consecutive 2-minute blocks, so it treated those blocks as the decoder trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply an explicit trial-quality filter. It drops incomplete tail segments that do not fill a whole 2-minute block, and it skips entire sessions if they would yield fewer than 2 blocks.

ii.
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

iii. The trajectory says this was done to satisfy the decoder-format requirement that each session contain at least two trials. No separate trial-quality rationale was documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `F.npy` and `Fneu.npy`, while also using `iscell.npy` to select ROIs and `ops.npy` to get preprocessing parameters such as `neucoeff` and `prctile_baseline`.

ii.
```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32), np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The notes justify this by citing the methods text that downstream analyses used fluorescence-based dF/F and the Track2p code/methods convention that cells are curated with `iscell[:,1] > 0.5`.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil correction, computes a per-neuron low-percentile baseline, and converts the result to an approximate dF/F. It then averages over non-overlapping 10-frame bins.

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

iii. In the trajectory and notes, the AI explicitly says this is a fast approximation to Suite2p baseline-corrected fluorescence/dF/F. It also says it replaced a slower sliding-window baseline with this constant-percentile approximation for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only ROIs with Suite2p cell probability greater than 0.5.

ii.
```python
iscell = np.load(suite / 'iscell.npy')
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32), np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The notes repeatedly justify this choice from Track2p defaults and from the methods statement that ROIs above the default 0.5 threshold were considered true cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to session start. Each pseudo-trial is a contiguous 2-minute block within the session, and metadata describes the alignment event as session start.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(dff_binned, motion_binned, time_binned, block_bins)
```

```python
'metadata': {
    ...
    'temporal_alignment_event': 'session start',
    'off_start': 0.0,
    'off_end': 120.0,
    ...
}
```

iii. The trajectory says the input should be time elapsed from the beginning of the experiment, so the AI anchored blocks to session start rather than to a stimulus or behavioral event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at a 30 Hz raw sampling rate, giving a bin size of about 333.3 ms. Non-overlapping temporal rebinning is applied to neural, input, and output traces.

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

iii. The notes justify this from the methods text saying decoding analyses averaged dF/F and behavior in bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. In the final implementation, the input time trace is derived from the neural frame index and the assumed 30 Hz frame rate. `tstamps.npy` is loaded but not used in the final time computation.

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
...
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. The trajectory documents a bug fix: the AI first tried `tstamps.npy`, found that its range was compressed and unsuitable, and switched to frame index divided by 30 Hz because that matched the methods text.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes elapsed time in seconds from frame number, then averages the time values over non-overlapping 10-frame bins.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

```python
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The trajectory says this was chosen because the decoder input was defined as time elapsed from experiment start and because frame index / 30 Hz was more reliable than raw timestamps.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The AI derives time from the same frame axis as neural data, bins it with the same 10-frame averaging, and slices it into the same 2-minute blocks as the neural traces.

ii.
```python
dff_binned = bin_time_series(dff, bin_size_frames)
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(dff_binned, motion_binned, time_binned, block_bins)
```

iii. The AI's notes and trajectory both describe this as a common synchronized time base shared across neural, input, and output streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives the output from `move_deve/motion_energy_glob.npy`. It also loads `tstamps.npy`, but the final output computation does not use timestamps or `interframe_int.npy`.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
...
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
```

iii. The notes justify the variable choice from the methods text, which identifies global motion energy as the behavioral quantity to decode. No explicit justification is given for ignoring `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI truncates motion to a common length with neural/timestamp arrays, averages it into non-overlapping 10-frame bins, pools all binned motion values across sessions, and then discretizes them globally. It does not normalize each session by its standard deviation.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
```

```python
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
...
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
```

iii. The notes say 10-frame averaging was intended to match the decoding analysis in the paper, and the trajectory says global percentile binning was chosen to create balanced classes.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes global 20th, 40th, 60th, and 80th percentile thresholds from all binned motion values and assigns each time bin to one of five categories using `np.digitize`.

ii.
```python
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
...
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. In the notes, the AI explicitly justifies this from the decoder requirement that motion energy be normalized/discretized into five equal-percentile bins.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by truncating both streams to the minimum common length, binning them with the same 10-frame averaging, and slicing them into the same blocks.

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
neural_trials, input_trials, output_cont = make_blocks(dff_binned, motion_binned, time_binned, block_bins)
```

iii. The notes describe the streams as synchronized framewise recordings and treat shared frame count plus common binning/blocking as sufficient alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles minor mismatches by silently truncating neural, motion, and timestamp arrays to their shortest common length. It also discards partial trailing data that do not fill a complete 10-frame bin or a complete 2-minute block, and skips sessions with fewer than two blocks.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

```python
T2 = (T // bin_size) * bin_size
x = x[..., :T2]
```

```python
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The trajectory shows one documented data issue: `tstamps.npy` was found to have unusable units, so the AI replaced it with frame-derived time. For missing-frame mismatches, no interpolation rationale was documented; the implemented policy is truncation.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified neural baseline computation as the main bottleneck during development, specifically an earlier sliding-window baseline estimate. In the final code, the remaining expensive work is loading large arrays, computing per-neuron percentiles in `compute_dff`, and per-session binning/block assembly.

ii.
```python
F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
```

```python
for i, (sess_path, subj) in enumerate(zip(session_paths, session_subjects)):
    ...
    dff_binned = bin_time_series(dff, bin_size_frames)
    motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
    ...
```

iii. `CONVERSION_NOTES.md` Step 6 says the initial sliding-window percentile baseline was too slow and was replaced with a faster percentile-baseline approximation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loops are the per-block list-building loop in `make_blocks` and the nested loops that discretize outputs trial by trial. Both could be reshaped or batch-processed instead of appended in Python.

ii.
```python
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
```

iii. The AI did not explicitly document these particular loops, but the instructions in the trajectory show that it was trying to speed up bottlenecks and simplify expensive Python-level operations.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several small pieces of processing: it repeatedly casts arrays to `float32`, repeatedly concatenates trial outputs for each session and then again globally, and repeatedly looks up subject indices with `subjects.index(subj)` inside the session loop.

ii.
```python
neural_trials.append(dff_binned[:, sl].astype(np.float32))
input_trials.append(time_binned[sl][None, :].astype(np.float32))
output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

```python
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
...
all_motion_values = np.concatenate(all_motion_values)
```

```python
subject_idx.append(subjects.index(subj))
```

iii. No explicit justification was documented for these repeated operations; they appear to be incidental to the implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The most obvious unnecessary processing is loading and truncating `tstamps.npy` even though timestamps are not used in the final exported dataset. The code also stores continuous motion blocks in `all_output_cont` only to discretize them later, instead of emitting final labels directly.

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
...
tstamps = tstamps[:T]
```

```python
all_output_cont.append(output_cont)
...
for sess_trials in all_output_cont:
    ...
    labels = np.digitize(x, edges, right=False).astype(np.int64)
```

iii. The trajectory explains why timestamps remained in the code: the AI initially tried to use them for elapsed-time input and later replaced them with frame-derived time after finding a units problem. No separate justification was given for retaining them afterward.
