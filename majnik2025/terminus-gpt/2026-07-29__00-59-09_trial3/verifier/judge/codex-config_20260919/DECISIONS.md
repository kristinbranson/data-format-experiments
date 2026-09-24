# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans every top-level directory under `data` as a subject and every directory beneath it as a session. For each session, `load_session` memory-maps `F.npy`, `Fneu.npy`, motion energy, and timestamps, and loads `iscell.npy` and `ops.npy`. Full mode uses all 41 discovered sessions; sample mode keeps the first two.

ii.
```python
root = Path('data')
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
for subj in subjects:
    for sess in sorted([p for p in (root / subj).iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)

F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
```

iii. The notes say the hierarchy is subject/session/`suite2p/plane0` plus `move_deve`, and that these variables provide fluorescence, cell curation, processing parameters, behavior, and synchronization information. Memory mapping was used to reduce I/O/memory cost.

## 1-b. How are the data split into subjects?

i. Each sorted top-level directory is treated as one mouse. Sessions are mapped back to the index of their parent directory in the complete sorted `subjects` list.

ii.
```python
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
session_subjects.append(subj)
subject_idx.append(subjects.index(subj))
```

iii. The AI states that folders such as `jm031` are one subject each. It retained the global subject list even in sample mode.

## 1-c. How are the data split into sessions?

i. Every sorted child directory of a subject is a daily recording session and becomes one target-format session, unless it has fewer than two generated blocks.

ii.
```python
for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
    session_paths.append(sess)
...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
all_neural.append(neural_trials)
```

iii. The notes describe each child folder as a day/session and cite the target requirement of at least two trials per session as the reason for the skip guard.

## 1-d. How are the data split into trials?

i. The AI creates consecutive, non-overlapping **120-second** blocks after temporal binning. Each block contains 360 bins, and an incomplete tail is silently discarded.

ii.
```python
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)
n_blocks = T // block_bins
T2 = n_blocks * block_bins
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
```

iii. The AI chose two-minute blocks because the paper's decoding analyses used consecutive two-minute blocks. It overlooked the task's explicit instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no within-session trial quality filter. Sessions with fewer than two blocks are skipped, and incomplete final blocks are dropped by integer division.

ii.
```python
n_blocks = T // block_bins
...
if len(neural_trials) < 2:
    continue
```

iii. The notes identify continuous recordings with no native trials or invalid-period indicators. The only stated rationale is satisfying the decoder's minimum-two-trials requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy`, with `iscell.npy` selecting ROIs and `ops.npy` supplying `neucoeff` and the baseline percentile.

ii.
```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The AI preferred fluorescence over `spks.npy` because the methods describe baseline-corrected fluorescence/dF/F for downstream analysis.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil (`F - neucoeff*Fneu`), computes one low-percentile baseline per neuron across the entire session, floors that baseline at `1e-3`, converts to `(Fc-F0)/F0`, averages every 10 frames, and slices into blocks.

ii.
```python
Fc = F - neucoeff * Fneu
F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
F0 = np.maximum(F0, 1e-3)
dff = (Fc - F0) / F0
...
return x.reshape(new_shape).mean(axis=-1)
```

iii. The AI calls this a fast approximation to Suite2p baseline-corrected dF/F. It replaced an initially planned moving-percentile baseline because that computation was too slow.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose Suite2p cell probability is greater than 0.5 are retained.

ii.
```python
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The notes say both the paper and Track2p defaults consider `iscell` probability above 0.5 a true cell.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-triggered realignment. Continuous session data is divided from session start, and metadata declares `session start` with each 120-second block represented as offsets 0 to 120 seconds.

ii.
```python
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 120.0,
```

iii. The AI judged session start to be the natural alignment because the data are spontaneous continuous recordings without stimulus events.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Raw 30 Hz data are averaged in non-overlapping groups of 10 frames, producing 333.33 ms bins (3 Hz) for neural, motion, and time signals.

ii.
```python
bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs
dff_binned = bin_time_series(dff, bin_size_frames)
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
```

iii. This follows the methods statement that decoding analyses denoised dF/F and behavior by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Although `tstamps.npy` is loaded and used when choosing the common length, the saved input is constructed from neural frame indices and the fixed 30 Hz sampling rate.

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
```

iii. The notes report that using `tstamps.npy` initially yielded an implausible 0–1.21 range, so the AI switched to frame index divided by 30 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A time value is generated for each retained raw frame, then each group of 10 times is averaged. Thus the values are bin centers (0.15, 0.4833, … seconds), not bin left edges.

ii.
```python
frame_time = np.arange(dff.shape[1], dtype=np.float32) / raw_fs
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. The AI intended a continuous elapsed-seconds input and applied the same temporal averaging as the other streams.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is truncated/binned with the same nominal frame count and sliced with the same block slices as neural data.

ii.
```python
time_binned = time_binned[:T2]
sl = slice(i * block_bins, (i + 1) * block_bins)
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The AI verified one trial with `np.allclose` against its frame-derived construction and argued that the shared frame clock preserves alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived from `move_deve/motion_energy_glob.npy`. `tstamps.npy` affects only common-length truncation; `interframe_int.npy` is not used.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
```

iii. The notes identify the precomputed global motion-energy trace as the target behavior and timestamps as synchronization information.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion is truncated to the shortest stream, averaged over 10-frame bins, truncated to complete 120-second blocks, concatenated across retained blocks and all sessions to find thresholds, then digitized.

ii.
```python
motion = motion[:T]
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
labels = np.digitize(x, edges, right=False).astype(np.int64)
```

iii. The AI wanted the paper's 10-frame denoising and globally balanced categorical classes. It explicitly argued that global thresholds avoid session-specific label drift.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20/40/60/80 percentile edges are computed once from all retained binned motion values across the full dataset. `np.digitize` yields labels 0–4, which are clipped to that range.

ii.
```python
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. The notes say global equal-percentile bins ensure an overall 20% distribution per class and consistent thresholds between sessions. This conflicts with the instruction to select bins per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes synchronous 30 Hz acquisition, truncates all streams to their minimum length, bins them identically, and applies identical block slices. It does not detect or interpolate dropped video frames.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
...
sl = slice(i * block_bins, (i + 1) * block_bins)
neural_trials.append(dff_binned[:, sl].astype(np.float32))
output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

iii. The AI reasoned that video was microscope-triggered and therefore framewise synchronization plus common truncation was adequate; its sanity checks only confirmed consistency with its own slicing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unequal neural, motion, or timestamp lengths are handled by truncating every stream to the shortest length. Short tails are dropped during 10-frame binning and again during block creation. Sessions with fewer than two blocks are skipped; there is no interpolation or explicit NaN handling.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
...
T2 = (T // bin_size) * bin_size
x = x[..., :T2]
...
if len(neural_trials) < 2:
    continue
```

iii. The AI treated small length discrepancies as removable tails. It did not document the dropped-camera-frame repair used by the human reference.

## 6-a. What are the most time-consuming steps of the code?

i. Neural preprocessing dominates: materializing filtered `F`/`Fneu`, computing per-neuron percentiles, dF/F, and temporal means. File loading and optional plotting are secondary. The full log reports roughly 0.2–0.96 seconds per session and 25.45 seconds total.

ii.
```python
F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True)
dff = (Fc - F0) / F0
dff_binned = bin_time_series(dff, bin_size_frames)
```

iii. The notes identify the originally attempted moving-percentile baseline as the bottleneck and say the session-wide percentile approximation reduced conversion to under a second per session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Directory/session iteration and variable-sized session assembly are naturally loop-based. The block loop could be replaced by reshaping/transposing binned arrays, and the nested output-label loop could digitize each whole session before reshaping. `subjects.index(subj)` could be replaced by a dictionary lookup.

ii.
```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
...
for sess_trials in all_output_cont:
    for arr in sess_trials:
        labels = np.digitize(arr.squeeze(0), edges, right=False)
```

iii. The AI claimed to prioritize vectorization, but did not specifically document these remaining loops. Their cost is modest relative to neural preprocessing.

## 6-c. What processing does the code repeat multiple times?

i. Each session independently repeats file loading, ROI filtering, baseline estimation, dF/F conversion, three separate calls to the same temporal-binning routine (neural, motion, time), and block-list construction. Motion is also concatenated once per session and again globally for percentile calculation.

ii.
```python
dff_binned = bin_time_series(dff, bin_size_frames)
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
all_motion_values = np.concatenate(all_motion_values)
```

iii. The AI did not identify repeated processing as a concern; the repeated per-session transformations are part of its uniform pipeline.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and converts `tstamps.npy` but never uses timestamp values, only its length. It retains `keep` and `ops` in `load_session`'s returned dictionary although the caller only uses `dff`, `motion`, and `tstamps`. Optional plots are diagnostic only. It also computes continuous per-trial motion arrays that are later replaced by labels, though these are needed temporarily to obtain global thresholds.

ii.
```python
return {'dff': dff, 'motion': motion, 'tstamps': tstamps,
        'keep': keep, 'ops': ops}
...
loaded = load_session(sess_path)
dff = loaded['dff']; motion = loaded['motion']; tstamps = loaded['tstamps']
```

iii. The AI loaded timestamps for alignment and diagnostics before discovering their scale was unsuitable, but left that I/O in place. It did not document the unused returned fields.
