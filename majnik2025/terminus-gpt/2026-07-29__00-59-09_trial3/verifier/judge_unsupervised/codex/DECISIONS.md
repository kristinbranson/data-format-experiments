# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `data/` for subject directories, scans each subject directory for session directories, and then loads each session from `suite2p/plane0` plus `move_deve`. Within each session it loads `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy`.

ii. ```python
root = Path('data')
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)

suite = sess_path / 'suite2p' / 'plane0'
move = sess_path / 'move_deve'
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent says the source variables are Suite2p traces plus `move_deve` motion arrays. The trajectory also shows it intentionally mirrored the raw `data/<subject>/<session>/suite2p` and `move_deve` layout.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the top-level folders under `data/`. The script sorts those names, stores them in `subjects`, and uses the folder name as the mouse identifier.

ii. ```python
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
```

iii. `CONVERSION_NOTES.md` Step 2 says `data/` contains one folder per subject and gives examples like `jm031`. Step 5 maps the subject folder name directly to `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Sessions are the immediate subdirectories inside each subject folder. They are sorted chronologically by folder name and each one becomes one session in the output lists.

ii. ```python
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)
```

iii. `CONVERSION_NOTES.md` Step 2 says each subject folder contains multiple day folders and Step 5 maps each session/day folder to one session entry.

## 1-d. How are the data split into trials?

i. The raw recordings are treated as continuous sessions, then trialized into consecutive non-overlapping 2-minute pseudo-trials after 10-frame temporal averaging. Each session therefore becomes 9 to 15 blocks depending on session length and truncation.

ii. ```python
bin_size_frames = 10
raw_fs = 30.0
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)

def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    ...
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
        neural_trials.append(dff_binned[:, sl].astype(np.float32))
        input_trials.append(time_binned[sl][None, :].astype(np.float32))
        output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

iii. In Step 3 and Step 5 of `CONVERSION_NOTES.md`, the agent justifies block segmentation by citing the paper’s decoding analyses as using 10-frame averaging and consecutive 2-minute blocks.

## 1-e. How are trials filtered based on quality controls?

i. There is almost no explicit trial-level QC. The script only truncates each modality to the shortest available length, discards any leftover partial block at the end of a session, and skips a session if it yields fewer than two 2-minute blocks.

ii. ```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
...
n_blocks = T // block_bins
T2 = n_blocks * block_bins
...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. Step 5 in `CONVERSION_NOTES.md` says continuous sessions would be split into blocks and Step 10 documents only a minimum-trial requirement plus length matching. The trajectory does not record any additional bad-trial criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`, with parameters from `ops.npy`, after filtering rows by `iscell.npy`.

ii. ```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly says the neural source variables are `F.npy`, `Fneu.npy`, `iscell.npy`, and `ops.npy`, and that the agent chose fluorescence-based traces instead of `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The script applies a custom approximate dF/F computation: neuropil correction `F - neucoeff * Fneu`, then one constant low-percentile baseline per neuron, then `(Fc - F0) / F0`. After that it averages non-overlapping 10-frame bins.

ii. ```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile_baseline = float(ops.get('prctile_baseline', 8.0))
    F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
    F0 = np.maximum(F0, 1e-3)
    dff = (Fc - F0) / F0
    return dff.astype(np.float32)

dff_binned = bin_time_series(dff, bin_size_frames)
```

iii. Step 5 and Step 6 in `CONVERSION_NOTES.md` say the goal was to approximate Suite2p’s baseline-corrected fluorescence because the paper used dF/F. The trajectory states an initial slower moving-baseline version was replaced with this faster percentile-baseline approximation for runtime reasons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by keeping only ROIs whose Suite2p cell probability exceeds 0.5.

ii. ```python
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. Step 1, Step 3, and Step 5 in `CONVERSION_NOTES.md` repeatedly justify this as matching both the Track2p default `iscell_thr = 0.50` and the methods text saying ROIs above 0.5 were treated as cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script aligns neural data to session start, not to a within-session event. After loading and truncating, it keeps the native frame order, bins in time, and slices consecutive 2-minute windows. Metadata labels the alignment event as `"session start"`.

ii. ```python
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins
)
...
'metadata': {
    'temporal_alignment_event': 'session start',
    'off_start': 0.0,
    'off_end': 120.0,
}
```

iii. Step 5 in `CONVERSION_NOTES.md` says the decoder input is time from the beginning of the experiment and therefore the natural alignment event is session start for these continuous recordings.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use non-overlapping 10-frame bins at 30 Hz, giving 333.33 ms per bin. No further temporal rebinning is applied.

ii. ```python
bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs

def bin_time_series(x, bin_size):
    T2 = (T // bin_size) * bin_size
    ...
    return x.reshape(new_shape).mean(axis=-1)
```

iii. Step 3 in `CONVERSION_NOTES.md` cites the paper’s statement that both dF/F and behavior traces were averaged in bins of 10 consecutive timestamps at a 30 Hz acquisition rate.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. In the final code it is not derived from `tstamps.npy`. It is derived from an implicit frame index and a hard-coded 30 Hz frame rate.

ii. ```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. Step 10 in `CONVERSION_NOTES.md` says the original attempt used `tstamps.npy` but produced a compressed range, so the agent switched to frame index divided by 30 Hz because the methods described 30 Hz acquisition.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script creates a monotonically increasing frame-time vector in seconds from 0 to session length, averages it in 10-frame bins, and then slices it into the same 2-minute blocks used for neural data.

ii. ```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
...
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The trajectory records that this was an intentional fix after `tstamps.npy` was found not to be in seconds. Step 7 of `CONVERSION_NOTES.md` then accepts the full-session range `[0.2, 1199.8]` or `[0.2, 1799.8]` as the intended representation.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It is generated on the neural frame count after truncation and then binned and blocked with the same functions as neural activity, so it is exactly index-aligned to the neural matrices produced by this script.

ii. ```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
...
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins
)
```

iii. Step 10 in `CONVERSION_NOTES.md` says the input sanity check recomputed frame-derived elapsed time from the raw source and matched the converted data with `np.allclose`.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output starts from `move_deve/motion_energy_glob.npy`.

ii. ```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
```

iii. Step 3 and Step 5 of `CONVERSION_NOTES.md` identify `motion_energy_glob.npy` as the processed behavioral motion-energy variable supplied with the dataset.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script truncates motion to the common session length, averages it in non-overlapping 10-frame bins, stores per-block continuous traces, then later discretizes them.

ii. ```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins
)
```

iii. Step 5 in `CONVERSION_NOTES.md` says motion should be aligned framewise to neural data and averaged over the same 10-frame bins because the methods described that denoising for behavior traces.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent pools all binned motion values from all sessions and blocks, computes global 20th/40th/60th/80th percentile edges, and uses `np.digitize` to assign five labels `0..4`.

ii. ```python
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
...
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
out_trials.append(labels[None, :])
```

iii. Step 5 in `CONVERSION_NOTES.md` explicitly justifies using five equal-percentile bins computed on the full converted dataset to avoid per-session label drift.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code aligns motion by simple array truncation to the shortest stream length and then applying the same 10-frame averaging and 2-minute block slicing as for neural data. It does not use `tstamps.npy` or `interframe_int.npy` to repair internal dropped camera frames.

ii. ```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
...
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins
)
```

iii. Step 4 and Step 5 in `CONVERSION_NOTES.md` say the agent planned to align frame-by-frame using matched lengths and timestamps. In practice the implemented code only uses matched lengths.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data are handled minimally. The code truncates all streams to the shortest session length, clips very small baselines to `1e-3` to avoid divide-by-zero in dF/F, drops trailing partial bins/blocks, and skips sessions with fewer than two blocks. It does not interpolate or insert missing values for internal dropped motion frames.

ii. ```python
F0 = np.maximum(F0, 1e-3)
...
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
...
T2 = (T // bin_size) * bin_size
...
if len(neural_trials) < 2:
    ...
```

iii. The data README notes that camera frame drops can be identified from `tstamps.npy` or `interframe_int.npy` and treated as missing or interpolated. The trajectory shows the agent noticed only the timestamp-unit problem, not the internal dropped-frame issue.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant expensive step is the per-session dF/F reconstruction, especially the baseline computation inside `compute_dff`. Loading large arrays and session-wise binning are secondary costs.

ii. ```python
def compute_dff(F, Fneu, ops):
    ...
    F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
    ...

for i, (sess_path, subj) in enumerate(zip(session_paths, session_subjects)):
    loaded = load_session(sess_path)
    ...
    dff_binned = bin_time_series(dff, bin_size_frames)
```

iii. Step 6 in `CONVERSION_NOTES.md` and trajectory step 16 explicitly say the initial sliding-window baseline was the bottleneck and was replaced with a faster approximation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-block Python loop in `make_blocks`, the nested per-trial discretization loop for outputs, and the repeated `subjects.index(subj)` lookup inside the session loop could all have been vectorized or precomputed.

ii. ```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
    input_trials.append(time_binned[sl][None, :].astype(np.float32))
    output_cont.append(motion_binned[sl][None, :].astype(np.float32))

for sess_trials in all_output_cont:
    out_trials = []
    for arr in sess_trials:
        x = arr.squeeze(0)
        labels = np.digitize(x, edges, right=False).astype(np.int64)
        ...

subject_idx.append(subjects.index(subj))
```

iii. The agent documented the main dF/F bottleneck but did not explicitly discuss these smaller loops. This section is inferred from the code structure rather than from an explicit note.

## 6-c. What processing does the code repeat multiple times?

i. It computes continuous motion blocks first, stores them in `all_output_cont`, then makes a second full pass to discretize them. It also repeatedly casts arrays to `float32` inside the block-building loop and repeatedly scans `subjects` to recover each subject index.

ii. ```python
all_output_cont.append(output_cont)
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
...
for sess_trials in all_output_cont:
    out_trials = []
    for arr in sess_trials:
        ...
        out_trials.append(labels[None, :])

neural_trials.append(dff_binned[:, sl].astype(np.float32))
input_trials.append(time_binned[sl][None, :].astype(np.float32))
output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

iii. There is no explicit justification for this two-pass structure in the notes beyond the need to compute global percentile thresholds before final discretization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and slices `tstamps.npy` but does not use the timestamp values in the final dataset. It also stores continuous motion blocks only to throw them away after binning into labels, and returns `keep` and `ops` from `load_session` although only `ops` is used transiently during processing.

ii. ```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
...
return {
    'dff': dff,
    'motion': motion,
    'tstamps': tstamps,
    'keep': keep,
    'ops': ops,
}
...
all_output_cont.append(output_cont)
...
data = {
    'output': all_output,
}
```

iii. The trajectory shows `tstamps.npy` was investigated and then abandoned because its units were not seconds. No further justification is given for keeping the unused timestamp-loading path or the temporary continuous-output storage.
