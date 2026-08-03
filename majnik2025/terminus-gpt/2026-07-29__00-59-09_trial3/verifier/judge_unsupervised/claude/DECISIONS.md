# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all subject directories under `data/`, then over all session subdirectories within each subject. For each session, it loads Suite2p neural arrays (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) from `suite2p/plane0/` and behavioral arrays (`motion_energy_glob.npy`, `tstamps.npy`) from `move_deve/`. Files are loaded using `np.load` with memory mapping where possible. There is no separate trial structure in the raw data -- sessions are continuous recordings.

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
def load_session(sess_path):
    suite = sess_path / 'suite2p' / 'plane0'
    move = sess_path / 'move_deve'
    F = np.load(suite / 'F.npy', mmap_mode='r')
    Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(suite / 'iscell.npy')
    ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
```

iii. The AI identified that the data is organized as `data/<subject>/<session>/` directories, with Suite2p outputs and motion energy arrays. This mirrors the reference notebook's `load_traces()` function and the Track2p code's session-by-session loading approach. The AI documented this data structure in CONVERSION_NOTES.md Steps 1-2.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by top-level directory names under `data/` (e.g., `jm031`, `jm032`, etc.). A sorted list of unique subject names is built. Each session is mapped to its parent subject directory name.

ii.
```python
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
# ...
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_subjects.append(subj)
# ...
subject_idx.append(subjects.index(subj))
```

iii. The AI recognized the one-subject-per-folder organization from exploring the data directory structure. The `subjects` list and `subject_idx` mapping are directly derived from the directory hierarchy. This is consistent with the reference notebook which similarly uses subject directory names.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder corresponds to one session (e.g., `2023-10-18_a`). Sessions are loaded in sorted order per subject. The final dataset contains 41 sessions across 6 subjects.

ii.
```python
for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
    session_paths.append(sess)
    session_subjects.append(subj)
```

iii. The AI identified sessions from the directory structure. The reference notebook similarly iterates over subdirectories sorted chronologically. The AI verified 41 sessions across 6 subjects, consistent with the data.

## 1-d. How are the data split into trials?

i. There are no explicit trials in the raw data -- sessions are continuous recordings (~20 or ~30 minutes). The AI segments each session into consecutive non-overlapping 2-minute blocks (pseudo-trials). After 10-frame temporal binning (360 bins per 2 minutes), each block has 360 time bins.

ii.
```python
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)  # 2 minutes after binning

def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    T2 = n_blocks * block_bins
    # ...
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
        neural_trials.append(dff_binned[:, sl].astype(np.float32))
        input_trials.append(time_binned[sl][None, :].astype(np.float32))
        output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

iii. The AI justified this by noting the paper's methods describe "consecutive 2-minute blocks" for decoding analyses. The resulting block count per session varies: ~10 blocks for 20-min sessions, ~15 for 30-min sessions (536 total blocks). The AI documented this reasoning in CONVERSION_NOTES.md Steps 3-5.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 blocks are skipped. No other trial-level quality filtering is applied. Any residual frames that don't complete a full 2-minute block are discarded (truncated).

ii.
```python
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The AI noted that the raw data are continuous recordings rather than discrete trials, so traditional trial-level quality filtering is not applicable. The minimum-2-blocks check ensures the decoder can have at least 2 trials per session for evaluation. The AI did not find explicit trial curation rules in the reference paper or code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (Suite2p options containing parameters like `neucoeff` and `prctile_baseline`). The `iscell.npy` array is used for cell filtering.

ii.
```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
```

iii. The AI chose fluorescence-based traces over deconvolved spikes (`spks.npy`) because the methods text states: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses." This is documented in CONVERSION_NOTES.md Step 4.

## 2-b. How is the `neural` data processed?

i. Processing involves: (1) Cell filtering by `iscell[:,1] > 0.5`, (2) Neuropil correction: `Fc = F - neucoeff * Fneu` with `neucoeff` from ops (default 0.7), (3) Baseline normalization: compute 8th percentile of corrected fluorescence per neuron, then `dF/F = (Fc - F0) / F0`, (4) Temporal binning: average over non-overlapping 10-frame bins.

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

def bin_time_series(x, bin_size):
    T = x.shape[-1]
    T2 = (T // bin_size) * bin_size
    x = x[..., :T2]
    new_shape = x.shape[:-1] + (T2 // bin_size, bin_size)
    return x.reshape(new_shape).mean(axis=-1)
```

iii. The AI noted that Suite2p's default baseline-corrected dF/F uses neuropil subtraction and baseline normalization. The AI used a "fast approximation" with a single whole-session percentile rather than Suite2p's actual sliding-window baseline estimation, documented as a speed optimization in CONVERSION_NOTES.md Step 6. The 10-frame binning matches the paper's description.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the Suite2p cell probability threshold: only ROIs with `iscell[:,1] > 0.5` are retained. No other neural quality filtering is applied.

ii.
```python
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32), np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The AI matched both the Track2p reference code default (`iscell_thr = 0.50`) and the paper's statement: "We considered all ROIs above the default threshold of 0.5 as true cells." This is documented in CONVERSION_NOTES.md Steps 1, 3, and 4.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No alignment to a specific event is performed. The temporal alignment is simply "session start" -- neural data begins from the start of the session recording. Since data is continuous spontaneous activity, trials (2-minute blocks) are consecutive segments from the session start.

ii.
```python
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 120.0,
```

iii. The AI recognized that there is no task event (stimulus onset, etc.) to align to -- this is spontaneous activity. The paper describes sessions as continuous recordings of freely behaving mice. Blocks start at time 0 of the session and are consecutive.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw data is at 30 Hz (both imaging and video). The AI applies 10-frame binning, resulting in a time bin size of ~333.33 ms (1000 * 10 / 30). Each 2-minute block has 360 time bins.

ii.
```python
bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs
```

iii. The AI cited the paper: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." This is documented in CONVERSION_NOTES.md Step 3.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is derived from the frame index and the known frame rate (30 Hz). It is NOT derived from `tstamps.npy`. The AI originally used `tstamps.npy` but found it contained values in an unexpected range (0 to ~1.21, not seconds), so switched to computing time from frame indices.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. The AI documented in CONVERSION_NOTES.md Step 10 that using `tstamps.npy` produced a compressed range (~0 to 1.21 s for a 20-min session), which was incorrect. The fix constructs elapsed time from frame index / 30 Hz, which matches the known acquisition rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Processing: (1) Generate frame-by-frame time as `frame_index / 30.0` seconds, (2) Apply the same 10-frame binning as neural data (averaging time within each bin), (3) Segment into 2-minute blocks. Time within each block represents elapsed time from session start, not block start.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
# ... then in make_blocks:
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The AI's rationale was to provide elapsed time from session start as a time-varying decoder input. The time increases across blocks within a session (not reset per block).

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time is constructed from the same frame indices as the neural data and goes through the same binning procedure, ensuring exact alignment. Both share the same number of time bins per block (360).

ii. Both neural and time data use `bin_time_series` with the same `bin_size_frames=10`, and both are sliced with the same block indices in `make_blocks`.

iii. Since both streams are derived from the same frame indices and undergo identical binning and segmentation, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory of each session.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
```

iii. The AI identified this as the global motion energy computed from video frames. The paper describes motion energy as "computed their pixelwise difference... squared... summed across pixels."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load raw framewise motion energy, (2) Truncate to match neural data length, (3) Apply 10-frame binning (same as neural), (4) Segment into 2-minute blocks, (5) Discretize into 5 equal-percentile bins using global percentile edges computed across all sessions.

ii.
```python
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
# ...
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
# ...
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. The AI followed the instruction to produce "Motion energy, normalized and discretized into five equal-percentile bins" and ensured the percentile edges are computed globally across the full dataset to achieve balanced bin distribution.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five equal-percentile bins are created using the 20th, 40th, 60th, and 80th percentile edges of the global motion energy distribution. `np.digitize` assigns each value to a bin (0-4). The result is globally balanced at ~20% per bin.

ii.
```python
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. The AI chose global percentiles (rather than per-session) to avoid label drift across sessions, producing exactly 20% per bin globally as confirmed in the verification output.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy array is loaded framewise, truncated to the minimum length across neural and motion streams, then binned with the same 10-frame averaging and segmented into the same 2-minute blocks as the neural data. This ensures temporal alignment.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
# Same binning:
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
```

iii. The paper states that video was triggered by microscope acquisition at the same 30 Hz rate, enabling frame-by-frame alignment. The AI confirmed this from the data structure.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles length mismatches by truncating all streams (neural, motion, timestamps) to the minimum common length. Sessions with fewer than 2 blocks are skipped. Residual frames that don't fill a complete 10-frame bin or 2-minute block are truncated. The baseline denominator F0 is clipped to a minimum of 1e-3 to avoid division by zero.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
# ...
F0 = np.maximum(F0, 1e-3)
```

iii. The AI noted from exploration that some sessions have slightly mismatched array lengths (e.g., 35997 vs 36000 frames). The truncation approach ensures no IndexError and minimal data loss. The AI documented the tstamps bug fix in CONVERSION_NOTES.md Step 10.

## 6-a. What are the most time-consuming steps of the code?

i. Loading large numpy arrays from disk (F.npy, Fneu.npy) and computing the dF/F are the primary bottlenecks. The full conversion runs in ~25 seconds total, so efficiency is adequate. The AI initially had a slow sliding-window percentile baseline that was a significant bottleneck.

ii. From conversion_full_out.txt: session processing times range from 0.19s to 0.96s per session, with larger neuron counts taking longer.

iii. The AI documented in CONVERSION_NOTES.md Step 6 that the initial moving-percentile baseline was too slow and was replaced with a fast whole-session percentile approximation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `make_blocks` function uses a Python loop over blocks to create trial slices. This could be vectorized using `np.reshape` to split the array into blocks without explicit looping. The motion energy discretization also loops over sessions and trials.

ii.
```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
```

```python
for sess_trials in all_output_cont:
    out_trials = []
    for arr in sess_trials:
        x = arr.squeeze(0)
        labels = np.digitize(x, edges, right=False).astype(np.int64)
```

iii. Given the total runtime of ~25 seconds for 41 sessions, these loops are not significant bottlenecks, but they could be vectorized for cleaner code.

## 6-c. What processing does the code repeat multiple times?

i. The `bin_time_series` function is called separately for neural, motion, and time data, but each operates on different arrays so no computation is truly duplicated. The `tstamps` array is loaded but not actually used in the final code (the time input is constructed from frame indices instead).

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
# tstamps is loaded but only used for truncation (T = min(..., tstamps.shape[0]))
```

iii. The leftover loading of `tstamps.npy` is a minor inefficiency from the earlier bug fix. It's still used for length truncation, so not entirely wasted, but could be removed if neural and motion array lengths are sufficient.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `tstamps.npy` array is loaded and used only for length computation. The continuous motion energy values (`all_output_cont`) are computed and stored temporarily before being discretized -- only the discretized version is kept in the final output. The raw `tstamps` values are not used in the output.

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
# Only used for: T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
```

iii. The code also accumulates `all_motion_values` for global percentile computation, which is necessary for discretization but is discarded after edges are computed.
