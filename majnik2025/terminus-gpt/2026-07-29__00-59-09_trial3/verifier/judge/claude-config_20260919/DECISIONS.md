# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks the `data/` directory (hard-coded relative path `Path('data')`): every top-level directory is treated as a subject, every sub-directory of a subject as a session. It builds two parallel lists (`session_paths`, `session_subjects`) and then loops over sessions, loading, for each one, five files: `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, `suite2p/plane0/iscell.npy`, `suite2p/plane0/ops.npy`, and `move_deve/motion_energy_glob.npy` plus `move_deve/tstamps.npy`. `F`, `Fneu`, `motion_energy_glob` and `tstamps` are opened with `mmap_mode='r'`. `move_deve/interframe_int.npy` and `spks.npy` are never loaded. All 41 sessions / 6 subjects / 20,445 tracked neurons are processed in the full run. There is no pre-existing trial structure in the raw data, so trials are created afterwards by segmentation (see 1-d).

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
if args.sample:
    session_paths = session_paths[:2]
    session_subjects = session_subjects[:2]
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

iii. From CONVERSION_NOTES.md Step 2: "`data/` contains one folder per subject (e.g. `jm031`). Each subject folder contains multiple session/day folders... Each session contains at least two relevant subfolders: `suite2p/plane0/` and `move_deve/`." The AI's Step 1 exploration of the Track2p reference code established that the pipeline consumes standard suite2p outputs (`F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `stat.npy`, `ops.npy`), which is why it loads the suite2p arrays directly. It chose fluorescence over `spks.npy` because the methods state "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses."

## 1-b. How are the data split into subjects?

i. Every directory directly under `data/` is one subject (mouse); names are sorted alphabetically and stored verbatim in `data['subjects']`. There is no `jm*` name filter — any directory would be accepted (in this dataset only the six `jm0xx` folders are directories, so the result is the six mice in order `jm031, jm032, jm038, jm039, jm040, jm046`). Each session records `subject_idx` by looking its subject name up in that list.

ii.
```python
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
...
subject_idx.append(subjects.index(subj))
...
'subjects': subjects,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "Subject folder name (e.g. `jm031`) → `subjects`, `subject_idx`; Unique sorted subject IDs; map each session to subject index... One subject per top-level folder." The data README confirms "For each subject there is a folder corresponding to the subject id".

## 1-c. How are the data split into sessions?

i. Each sub-directory of a subject folder (a recording day, e.g. `2023-10-18_a`) is one session, sorted alphabetically (= chronologically). Sessions are flattened into a single ordered list across subjects, giving 41 sessions (7/7/7/7/6/7). A session identifier `subject__date` is stored in `metadata['session_ids']`. A session is dropped from the output entirely if it yields fewer than 2 trials (never triggered on this dataset).

ii.
```python
for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
    session_paths.append(sess)
    session_subjects.append(subj)
...
sess_id = f'{sess_path.parent.name}__{sess_path.name}'
...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
...
session_ids.append(sess_id)
```

iii. CONVERSION_NOTES.md Step 2: "Each subject folder contains multiple session/day folders (e.g. `2023-10-23_a`)"; the data README states each session folder corresponds to one recording day. The `< 2 blocks` guard follows the instruction that "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 1-d. How are the data split into trials?

i. There is no natural trial structure, so the AI cuts each continuous session into consecutive, non-overlapping **2-minute (120 s) blocks** — `block_bins = 2*60*30/10 = 360` bins of 333.33 ms. Any trailing bins that do not fill a complete 120 s block are discarded. This gives 10 (or 15) trials for 20-min (30-min) sessions, 536 trials total, each of T = 360 bins. Note that this contradicts the explicit task instruction "Split sessions into 60-second trials"; the AI instead followed the paper's cross-validation block length. Because the block is 120 s rather than 60 s, five sessions that are a few frames short (e.g. 35,998 of 36,000 frames) lose a whole final block: they yield 9 rather than 10 trials, discarding ~2 min of data each. Metadata records `off_start = 0.0`, `off_end = 120.0`, `block_duration_s = 120.0`.

ii.
```python
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)  # 2 minutes after binning
...
def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    T2 = n_blocks * block_bins
    dff_binned = dff_binned[:, :T2]
    ...
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
        neural_trials.append(dff_binned[:, sl].astype(np.float32))
        input_trials.append(time_binned[sl][None, :].astype(np.float32))
        output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

iii. CONVERSION_NOTES.md Step 3: "Decoding splits in the paper were based on consecutive 2-minute blocks, relevant as a sanity check for later trial segmentation decisions", and Step 5 Key Decision 4: "**Continuous sessions will be segmented into consecutive 2-minute blocks**: This follows the paper's decoding split unit and creates at least 10 pseudo-trials per 20-minute session." The trajectory shows the AI planned 2-min blocks from Step 5 onward; the "60-second trials" line of the Decoder Task section is never mentioned anywhere in the notes or trajectory.

## 1-e. How are trials filtered based on quality controls?

i. No quality-based trial filtering is applied — every complete 120 s block of every session is kept. The only exclusions are structural: (a) trailing bins that do not fill a full block are dropped, (b) any session producing fewer than 2 blocks would be skipped (never occurs), and (c) the common-length truncation in `load_session` (see 4-d/5) can shorten a session enough to remove its last block.

ii.
```python
n_blocks = T // block_bins
T2 = n_blocks * block_bins
...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The notes contain no trial-quality criterion; CONVERSION_NOTES.md Step 3 "Trial curation rules" only says "Raw data are continuous recordings rather than explicit trials. The paper's decoding analyses split recordings into consecutive 2-minute blocks..." The recordings are spontaneous behaviour in the dark with no task events, so there is no behavioural criterion for rejecting a segment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from suite2p `plane0` outputs: `F.npy` (raw ROI fluorescence) and `Fneu.npy` (neuropil fluorescence), with `iscell.npy` used to select ROIs and `ops.npy` used to read `neucoeff` (0.7) and `prctile_baseline` (8.0). Deconvolved `spks.npy` is deliberately not used.

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

iii. CONVERSION_NOTES.md Step 4 discrepancy table: "Code preserves `F`, `Fneu`, and `spks` from suite2p; no explicit dF/F recomputation found in Track2p ... Paper: 'We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses.' → Construct neural data from Suite2p fluorescence-based traces, not raw deconvolved `spks`."

## 2-b. How is the `neural` data processed?

i. Three steps: (1) neuropil subtraction `Fc = F - neucoeff*Fneu` with `neucoeff` read from `ops` (0.7); (2) a **single static baseline per neuron** — the 8th percentile of `Fc` over the whole session, floored at 1e-3 — and normalisation `dff = (Fc - F0)/F0`; (3) averaging over non-overlapping 10-frame bins. Step (2) is explicitly labelled in the code as a "fast approximation to suite2p baseline-corrected fluorescence"; it replaces suite2p's default rolling `maximin` baseline (`dcnv.preprocess`, 60 s window) that the reference solution uses. Consequences: slow baseline drift is *not* removed, and for 415 of 20,445 neurons (2%) the 8th-percentile baseline is ≤ 1e-3 and gets clipped, so their dF/F reaches ~1e5–1e6, several orders of magnitude above the rest of the data (per-session max dF/F is ~3.7e5 in the median session). Per-neuron correlation with the suite2p maximin-preprocessed trace is nevertheless high (median r ≈ 0.99 in a spot check of jm031/2023-10-18).

ii.
```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile_baseline = float(ops.get('prctile_baseline', 8.0))
    # Fast approximation to suite2p baseline-corrected fluorescence:
    # per-neuron low-percentile baseline after neuropil correction.
    F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
    F0 = np.maximum(F0, 1e-3)
    dff = (Fc - F0) / F0
    return dff.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: Initial sliding-window percentile baseline estimation was too slow for 36,000-frame sessions. Code speedups added: Replaced expensive moving-percentile baseline with a fast per-neuron low-percentile baseline approximation, reducing sample conversion to <1 second for 2 sessions." Trajectory step 16: "The bottleneck is the naive per-timepoint moving percentile loop... A practical fix is to replace the expensive moving percentile baseline with a simpler, much faster per-neuron constant percentile baseline approximation, which is still consistent with the methods' use of baseline-corrected fluorescence and can be documented." The AI wrote its own sliding-percentile baseline rather than calling suite2p's `dcnv.preprocess`, and then abandoned it for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are kept if `iscell[:, 1] > 0.5` (suite2p cell probability above the default threshold). No other neuron-level curation (SNR, activity level, etc.) is applied. On this dataset the filter is a no-op: the released suite2p folders contain only Track2p-tracked cells, and all 20,445 ROIs have probability > 0.5 (and `iscell[:,0] == 1`), so exactly the same neurons are kept as in the reference, which applies no filter at all.

ii.
```python
iscell = np.load(suite / 'iscell.npy')
...
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```
```python
'cell_filter': 'iscell_probability_gt_0.5',
```

iii. CONVERSION_NOTES.md Step 4: "Track2p default `iscell_thr = 0.50`; loaders/savers filter by `iscell[:,1] > 0.5` or `iscell[:,0] == 1` ... Paper: 'We considered all ROIs above the default threshold of 0.5 as true cells.' → Use `iscell[:,1] > 0.5` to match both code default and methods text." Step 5 Key Decision 2 repeats this.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus event: trials are contiguous segments of a continuous recording, so alignment is to **session start**. Trial *k* covers bins `[k*360, (k+1)*360)` counted from the first imaging frame of the session, and the trials tile the session without gaps or overlap. Metadata states `temporal_alignment_event = 'session start'`, `off_start = 0.0`, `off_end = 120.0` (times relative to each block's own start).

ii.
```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
...
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 120.0,
```

iii. CONVERSION_NOTES.md Step 4/5: the recordings are continuous spontaneous-behaviour sessions with no task events ("All experiments were performed in the dark, under sensory-minimised conditions"), so the only meaningful reference point is the start of the recording; segmentation into consecutive blocks is what the paper does for cross-validation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning from the 30 Hz acquisition rate to 3 Hz by averaging non-overlapping groups of 10 consecutive frames, i.e. a **333.33 ms** bin, stored in `metadata['time_bin_size']`. The same `bin_time_series` helper is applied to the dF/F matrix, the motion-energy trace and the time vector, so all three streams share one bin grid; the tail shorter than a full bin is dropped. Binning is done *before* motion energy is discretised.

ii.
```python
def bin_time_series(x, bin_size):
    T = x.shape[-1]
    T2 = (T // bin_size) * bin_size
    x = x[..., :T2]
    new_shape = x.shape[:-1] + (T2 // bin_size, bin_size)
    return x.reshape(new_shape).mean(axis=-1)

bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs
...
dff_binned = bin_time_series(dff, bin_size_frames)
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. CONVERSION_NOTES.md Step 3: "Neural data time bin | 10 frames (~0.333 s) for decoding analyses; raw acquisition 30 Hz | 'For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps.'" Key Decision 3: "**Common time base will be 10-frame bins (~0.333 s)**: This matches the paper's decoding preprocessing for both neural and behavior traces."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from any stored variable: time is reconstructed from the imaging frame index divided by the 30 Hz frame rate, `frame_time = arange(n_frames)/30`, then binned with the same 10-frame averaging (so each value is the centre of its bin: 0.15 s, 0.483 s, …). The AI first tried `move_deve/tstamps.npy` but found it is not in seconds (the full session spans only ~1.21 units, ~3.36e-5 per frame) and abandoned it; `tstamps` is still loaded but now only participates in the common-length truncation. The input is named `time_from_session_start_s` and runs continuously across trials within a session (0.15 → up to 1799.8 s), resetting at each new session.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
...
input_trials.append(time_binned[sl][None, :].astype(np.float32))
...
'input_names': ['time_from_session_start_s'],
```

iii. CONVERSION_NOTES.md Step 10: "Time input bug: initial use of `tstamps.npy` produced a compressed range (~0 to 1.21). Resolved by constructing elapsed time from frame index / 30 Hz, matching the methods." Trajectory step 20: "`tstamps.npy` spans only ~1.21 units with increments ~3.36e-05, so it is not in seconds and is unsuitable as the decoder input. The script was patched to use frame index / 30 Hz instead, which is consistent with the methods."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Only two operations: divide the 0-based frame index by 30 Hz, and average within the same 10-frame bins used for the neural data (yielding bin-centre times). No offset, rescaling, or per-trial reset is applied — the value is absolute seconds since the first frame of the session, so trial *k* spans 120 s·k … 120 s·(k+1). Values are stored as float32, shape (1, 360) per trial.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. Trajectory step 21: "the input range is [0.2, 1199.8], which spans the full 20-minute session rather than resetting within each 2-minute block. This is actually consistent with the decoder task wording: input is time elapsed from the beginning of the experiment, time-varying." The verification output confirms the per-session ranges [0.2, 1199.8] / [0.2, 1799.8] (printed rounded; the first bin centre is 0.15 s).

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: the time vector is built from `dff.shape[1]` (the same truncated frame count as the neural matrix), binned with the same `bin_time_series` call and sliced with the *same* `slice(i*block_bins, (i+1)*block_bins)` inside `make_blocks`, so element *t* of the input is the timestamp of element *t* of the neural matrix in every trial. No interpolation or resampling is needed.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins)
...
sl = slice(i * block_bins, (i + 1) * block_bins)
neural_trials.append(dff_binned[:, sl].astype(np.float32))
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. CONVERSION_NOTES.md Step 10 sanity check 3: "Input sanity check from raw source: recomputed frame-derived elapsed time for session 0 trial 0 and confirmed `np.allclose(...) == True`." The alignment follows trivially from deriving time from the neural frame index rather than from a separate clock.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. From `move_deve/motion_energy_glob.npy`, the pre-computed global motion-energy trace from the behaviour video (one value per video frame at 30 Hz). `move_deve/tstamps.npy` is also loaded, but only to compute the common minimum length; `move_deve/interframe_int.npy` — the file the data README points to for locating dropped camera frames — is never loaded.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
...
'output_names': ['motion_energy_bin'],
```

iii. CONVERSION_NOTES.md Step 3: "Motion metric | Global motion energy from squared pixelwise differences of consecutive video frames | 'computed their pixelwise difference... squared... summed across pixels'", and Step 5 maps "`move_deve/motion_energy_glob.npy` → `output[0]`". Step 2 notes that `move_deve/` "Contains the processed behavioural data".

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. (1) The raw trace is truncated to the common length with the neural data; (2) it is averaged in the same non-overlapping 10-frame bins as the neural data (3 Hz); (3) the binned continuous values for all sessions are pooled and converted to integer class labels by `np.digitize` against four global percentile edges, clipped to [0, 4]. Binning precedes discretisation (correct order — averaging class labels would be meaningless). No smoothing, log transform, z-scoring or dropped-frame interpolation is applied.

ii.
```python
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
...
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
...
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
for sess_trials in all_output_cont:
    for arr in sess_trials:
        x = arr.squeeze(0)
        labels = np.digitize(x, edges, right=False).astype(np.int64)
        labels = np.clip(labels, 0, 4)
```

iii. CONVERSION_NOTES.md Step 5 mapping: "Align framewise to neural data; average over the same 10-frame bins; discretize globally into 5 equal-percentile bins using full converted dataset; store as 1 x T categorical integer trace", justified by the methods statement that behaviour traces were denoised "by averaging in bins of 10 consecutive timestamps".

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into five bins by **global** equal-percentile edges: after all sessions are processed, every binned motion-energy value from the whole dataset is pooled, the 20/40/60/80th percentiles are taken once, and those four edges are applied to every session. The instruction asks for "five equal-percentile bins, **selected per session**"; the AI explicitly rejected per-session edges. The consequence is that the classes are uniform only in aggregate (0.2/0.2/0.2/0.2/0.2 globally) and strongly unbalanced per session: e.g. session 0 has 0.0% in class 0 and 79.7% in class 1, while the last seven sessions have 0% in classes 0 and 1. Because class prior then covaries with session identity (and hence with the neural population), this also inflates apparent decoder accuracy (0.3865 validation balanced accuracy vs. 0.3026 for the reference solution). Edges are stored in `metadata['motion_bin_edges_percentiles_20_40_60_80']`; value names are `bin_0`…`bin_4`.

ii.
```python
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
print('Motion percentile edges:', edges)

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

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "**Percentile thresholds for discretization should be computed on the full binned output distribution, then applied consistently to all sessions**: This best matches the task requirement for normalized/discretized motion energy and avoids per-session label drift." Step 9 records the resulting global distribution as "[0.2, 0.2, 0.2, 0.2, 0.2] globally — Yes" match. Trajectory step 26 notices the consequence but dismisses it: "some individual sessions lack lower bins, which is acceptable."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is assumed to be frame-for-frame (both streams were acquired at 30 Hz with the camera triggered by the microscope). The only reconciliation performed is truncation of all three streams to the shortest length, `T = min(n_imaging_frames, len(motion), len(tstamps))`; afterwards the motion trace is binned and sliced with the same indices as the neural matrix. Dropped camera frames are **not** detected or interpolated. In 8 of 41 sessions the motion file is shorter than the imaging data (1–148 frames missing), and in those sessions truncation removes samples from the *end* while the dropped frames occur *throughout* the session, so motion energy is progressively shifted relative to the neural data: e.g. jm031/2023-10-22 is missing 116 frames, the first at frame 653, so from ~22 s into the recording onward the behaviour trace leads the neural trace by 1–116 frames (up to 3.9 s ≈ 12 bins). jm032/2023-10-22 is missing 148 frames. The remaining 6 affected sessions are off by only 1–3 frames (< 1 bin).

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```
```python
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
neural_trials, input_trials, output_cont = make_blocks(
    dff_binned, motion_binned, time_binned, block_bins)
```

iii. CONVERSION_NOTES.md Step 4: "Video was triggered by microscope acquisition; both modalities recorded at 30 Hz → Align neural and motion streams frame-by-frame using timestamps/length consistency; then apply the paper's 10-frame averaging", and Step 4 Final Understanding asserts the streams are "length-matched to neural recordings in inspected sessions". Planned sanity Check 1 ("verify neural and motion traces have matched raw frame counts (~36,000)") was never executed; the Step 10 checks that were run all recompute quantities with the conversion script's own truncation logic, so they could not detect the problem.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms, all implicit and none documented as data-quality handling:
- **Missing camera frames**: handled only by truncating every stream to the common minimum length (see 4-d). The data README explicitly warns "In some recordings there might be some missing frames from the camera... The indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for motion energy or they can be interpolated over" — neither option is taken, and `interframe_int.npy` is never read. Beyond the misalignment, this also costs whole trials: a session short by 2 frames (3,599 usable bins instead of 3,600) drops from 10 to 9 two-minute blocks, discarding ~120 s of good data; this happens in 5 sessions, and overall the converted set contains 64,320 s of data against 65,400 s in the reference.
- **Degenerate baselines**: `F0 = np.maximum(F0, 1e-3)` prevents division by zero/negatives, but silently turns the 415 neurons (2%) with non-positive 8th-percentile baseline into traces with values up to ~1e6 instead of excluding or flagging them.
- **Incomplete trials**: trailing bins that do not fill a 120 s block are discarded.
- **Degenerate sessions**: sessions with fewer than 2 blocks are skipped with a printed message (never triggered).
No assertions or explicit validation of stream lengths exist in the script.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```
```python
F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
F0 = np.maximum(F0, 1e-3)
```
```python
n_blocks = T // block_bins
T2 = n_blocks * block_bins
...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The only documented data problem is the timestamp-unit bug (CONVERSION_NOTES.md Step 10: "Time input bug: initial use of `tstamps.npy` produced a compressed range (~0 to 1.21). Resolved by constructing elapsed time from frame index / 30 Hz"). The notes otherwise assert the opposite of the truth — Step 4: motion arrays are "framewise and length-matched to neural recordings in inspected sessions" — and trajectory step 24 interprets the short sessions as benign: "some sessions are slightly short of 36,000 frames and thus yield 9 blocks after truncation."

## 6-a. What are the most time-consuming steps of the code?

i. The script instruments itself with `time.time()` per session and in total. After the baseline approximation was introduced the whole 41-session conversion takes well under a minute (0.19–0.24 s/session for 221 neurons, ~0.9 s for 685 neurons, scaling with neurons × frames). What remains dominant is, per session, (1) reading and materialising `F.npy`/`Fneu.npy` into float32 arrays, (2) the `np.percentile` over the full (n_neurons × 36,000–54,000) matrix, and (3) the elementwise neuropil subtraction and division; and, once at the end, (4) pickling the ~409 MB dictionary to disk. The originally written sliding-window percentile baseline was the true bottleneck (it did not finish within seconds on one session) and was removed rather than optimised — the reference instead calls suite2p's vectorised `dcnv.preprocess`, which performs the proper maximin baseline in seconds.

ii.
```python
t0 = time.time()
...
st = time.time()
...
print(f'Processed {sess_id}: neurons={dff.shape[0]} raw_frames={dff.shape[1]} '
      f'bins={dff_binned.shape[1]} blocks={len(neural_trials)} time={time.time()-st:.2f}s')
...
print(f'Total time: {time.time()-t0:.2f}s')
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: Initial sliding-window percentile baseline estimation was too slow for 36,000-frame sessions. Code speedups added: Replaced expensive moving-percentile baseline with a fast per-neuron low-percentile baseline approximation, reducing sample conversion to <1 second for 2 sessions." Step 7: "Conversion | ~1.1 s/session | < 1 minute for full dataset".

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remaining Python loops, all cheap but avoidable:
- `make_blocks` copies each block one at a time; the whole session could be reshaped to `(n_neurons, n_blocks, block_bins)` in one operation (and the per-trial `.astype(np.float32)` copies are redundant since the arrays are already float32).
- The discretisation double loop applies `np.digitize` trial-by-trial; it could be applied once per session (or once to the whole concatenated array) before splitting.
- `subjects.index(subj)` performs a linear list search per session instead of a dict lookup (negligible with 6 subjects).
The AI did not identify any of these; the only loop it vectorised away was its own sliding-window baseline.

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

iii. No justification is given in CONVERSION_NOTES.md for these loops; the notes state only that the conversion already runs in "< 1 minute for full dataset", which is why no further vectorisation was pursued.

## 6-c. What processing does the code repeat multiple times?

i. Little is recomputed, but there is some duplication: (1) the binned motion energy is materialised twice — once as per-trial arrays in `all_output_cont` and again as a flat copy per session in `all_motion_values` — and the whole dataset is then iterated a second time to convert those same values into labels; (2) `bin_time_series` is invoked three times per session on three separate streams that share the same bin grid; (3) the frame-time vector `arange(T)/30` is rebuilt for every session even though sessions share only two distinct lengths; (4) `.astype(np.float32)` is applied to arrays that are already float32 (in `make_blocks` and on `motion`), causing extra copies. The dataset is also held entirely in memory (~409 MB) while these copies are made.

ii.
```python
all_output_cont.append(output_cont)
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
...
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
for sess_trials in all_output_cont:
    for arr in sess_trials:
        ...
```

iii. The two-pass structure is a direct consequence of Key Decision 6 (global percentile edges): the edges cannot be known until every session has been binned, so the continuous values must be retained and revisited. CONVERSION_NOTES.md does not discuss this cost.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- `tstamps.npy` is loaded, cast to float64 and truncated in every session, but after the timestamp bug fix it is used only to compute `min(...)`; its values never reach the output.
- `iscell.npy` and `ops.npy` are loaded per session and a boolean mask is applied, but the mask selects all ROIs in every session (all 20,445 have probability > 0.5), so the fancy-indexed copies of `F` and `Fneu` are pure overhead; `load_session` also returns `keep` and `ops`, which the caller never uses.
- The continuous per-trial motion arrays in `all_output_cont` are built and stored only to be replaced by integer labels; only the labels are saved.
- The final `.astype(np.float32)` casts in `make_blocks` duplicate already-float32 data, and `neural` trial slices are stored as copies rather than views of the session matrix.
- `--show-processing` plotting is optional and not part of the saved product.
None of this materially affects runtime (< 1 min total), but it is work whose result is discarded.

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
tstamps = tstamps[:T]
return {'dff': dff, 'motion': motion, 'tstamps': tstamps, 'keep': keep, 'ops': ops}
```
```python
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32),
                  np.asarray(Fneu[keep], dtype=np.float32), ops)
```
```python
output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

iii. CONVERSION_NOTES.md gives no justification for retaining `tstamps`; it was kept after trajectory step 20 replaced it as the time source ("The script was patched to use frame index / 30 Hz instead"). The `iscell` filtering is justified as matching the Track2p default and the methods ("We considered all ROIs above the default threshold of 0.5 as true cells"), i.e. it is defensive rather than effective on this already-curated release.
