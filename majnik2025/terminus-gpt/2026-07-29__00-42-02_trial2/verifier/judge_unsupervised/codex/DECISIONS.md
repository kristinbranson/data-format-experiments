# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `data/` for subject directories whose names start with `jm`, scans each subject for dated session directories whose first four characters are digits, then loads each session from fixed Suite2p and `move_deve` file paths. It computes transformed neural and motion signals per session, bins them, and finally splits each session into 2-minute trial blocks.

ii. ```python
def discover_subjects(data_root: Path):
    subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
    sessions_by_subject = {}
    for subj in subjects:
        sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        sessions_by_subject[subj.name] = sessions
    return subjects, sessions_by_subject

def load_session(session_dir: Path):
    pl0 = session_dir / 'suite2p' / 'plane0'
    move = session_dir / 'move_deve'
    F = np.load(pl0 / 'F.npy', allow_pickle=True)
    Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
    ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
    stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
    motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
    tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
    interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
    return F, Fneu, ops, stat, motion, tstamps, interframe

for idx, (subj, sessdir) in enumerate(selected):
    F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
    dff = compute_dff(F, Fneu, ops)
    aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
    neural_binned = bin_neural(dff, BIN_FRAMES)
    motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
    neural_blocks, motion_blocks, time_blocks = session_to_blocks(neural_binned, motion_binned)
```

iii. In `CONVERSION_NOTES.md`, the agent justified this from the native dataset layout: one `jm*` directory per mouse, one dated directory per recording day, and one `suite2p/plane0` plus one `move_deve` folder per session. The notes say the mapping should follow this native organization directly.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories starting with `jm`. The output `subjects` list is the sorted list of those directory names.

ii. ```python
def discover_subjects(data_root: Path):
    subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
    sessions_by_subject = {}
    for subj in subjects:
        sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        sessions_by_subject[subj.name] = sessions
    return subjects, sessions_by_subject

subjects, sessions_by_subject = discover_subjects(data_root)
subject_names = [p.name for p in subjects]
```

iii. The justification in the notes is that each `jm*` directory corresponds to one mouse, matching `data/README.md`.

## 1-c. How are the data split into sessions?

i. Sessions are split by subject subdirectories whose names begin with a four-digit year, then sorted lexicographically within each subject.

ii. ```python
for subj in subjects:
    sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    sessions_by_subject[subj.name] = sessions

selected = []
for subj in subject_names:
    for sess in sessions_by_subject[subj]:
        selected.append((subj, sess))
```

iii. The justification in the notes is that each dated subdirectory is one daily recording session, consistent with the data README.

## 1-d. How are the data split into trials?

i. The script treats each continuous session as a set of consecutive non-overlapping 2-minute blocks. Trials are created only after neural and motion traces have first been binned into 10-frame bins.

ii. ```python
FS = 30.0
BIN_FRAMES = 10
BLOCK_SECONDS = 120.0
BLOCK_FRAMES = int(FS * BLOCK_SECONDS)
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES

def session_to_blocks(neural_binned: np.ndarray, motion_binned: np.ndarray):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    neural_binned = neural_binned[:, :n_bins]
    motion_binned = motion_binned[:n_bins]
    n_blocks = n_bins // BLOCK_BINS
    neural_blocks = []
    motion_blocks = []
    time_blocks = []
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
        motion_blocks.append(motion_binned[s:e].astype(np.float32))
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
    return neural_blocks, motion_blocks, time_blocks
```

iii. The notes say the data have no native trial markers and that the paper’s decoder used consecutive 2-minute blocks, so the agent chose those blocks as the target-format “trials.”

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only after motion has been discretized. A block is dropped if fewer than 80% of its time bins have valid motion labels, or if it has fewer than 10 valid bins total. A session is kept only if at least two trial blocks survive.

ii. ```python
for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
    y = np.digitize(mb, edges, right=False).astype(np.int64)
    y[np.isnan(mb)] = -1
    valid = y >= 0
    if valid.sum() < max(10, int(0.8 * len(y))):
        continue
    ...
    sess_neural.append(nb.astype(np.float32))
    sess_input.append(tb.astype(np.float32))
    sess_output.append(y[None, :].astype(np.int64))
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. The notes justify this only indirectly: the agent wanted to preserve missing-camera-frame information and also satisfy the decoder requirement that each session contain at least two trials. The specific 80% threshold is not justified in the notes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, and baseline parameters read from `suite2p/plane0/ops.npy`.

ii. ```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()

def compute_dff(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    ...
```

iii. The justification in the notes and trajectory is that the notebook helper says `F.npy` contains raw fluorescence and suggests computing dF/F (or using `spks.npy`) for more proper analysis, while the methods summary described baseline-corrected fluorescence traces.

## 2-b. How is the `neural` data processed?

i. Neural processing is: neuropil subtraction using `F - neucoeff * Fneu`; then, for each neuron separately, a reflected moving average over a 60 s window; then a single 8th-percentile scalar baseline per neuron; then `(Fcorr - baseline) / baseline`; then 10-frame temporal averaging.

ii. ```python
def compute_dff(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    win_seconds = float(ops.get('win_baseline', 60.0))
    win = max(1, int(round(win_seconds * float(ops.get('fs', FS)))))
    prct = float(ops.get('prctile_baseline', 8.0))
    baseline = np.empty_like(Fcorr, dtype=np.float32)
    for i in range(Fcorr.shape[0]):
        smooth = moving_average_reflect(Fcorr[i], win)
        base = np.percentile(smooth, prct)
        baseline[i] = base
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fcorr - baseline) / baseline
    return dff.astype(np.float32)

def bin_neural(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)
```

iii. The agent’s justification in the notes and trajectory was that raw `F.npy` should not be used directly, so it attempted a “Suite2p-consistent” dF/F-like transform using `ops.npy` defaults, then matched the paper’s stated 10-frame denoising.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not apply any explicit neuron-quality filtering in `convert_data.py`. It uses all rows already present in the released `F.npy`/`Fneu.npy` arrays.

ii. ```python
def load_session(session_dir: Path):
    ...
    stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
    ...
    return F, Fneu, ops, stat, motion, tstamps, interframe

for idx, (subj, sessdir) in enumerate(selected):
    F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
    dff = compute_dff(F, Fneu, ops)
```

iii. The notes justify this by citing `data/README.md`: the released Suite2p folders already contain only successfully tracked neurons present across all days, with matched row order across sessions. The notes also say all inspected `iscell` rows already passed the default threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of each artificial 2-minute recording block, not to a task event. Each trial block starts at bin 0 of that block.

ii. ```python
'metadata': {
    ...
    'temporal_alignment_event': 'start of each 2-minute continuous recording block',
    'off_start': 0.0,
    'off_end': BLOCK_SECONDS,
    ...
}

for b in range(n_blocks):
    s = b * BLOCK_BINS
    e = (b + 1) * BLOCK_BINS
    neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
```

iii. The notes justify this by saying the recordings are continuous and the paper’s decoding split them into consecutive 2-minute blocks, so block starts were used as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, so each time bin is `10 / 30 = 0.333...` s, or 333.33 ms. Yes, temporal rebinning is applied to both neural and motion data.

ii. ```python
FS = 30.0
BIN_FRAMES = 10

def bin_neural(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)

'metadata': {
    'time_bin_size': 1000.0 * BIN_FRAMES / FS,
}
```

iii. The justification in the notes is that the paper’s decoder denoised both neural and behavioral traces by averaging over bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not taken from a dedicated raw data array. The time input is synthesized from the constants `FS`, `BIN_FRAMES`, and the number of bins in a 2-minute block.

ii. ```python
FS = 30.0
BIN_FRAMES = 10
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES

time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The notes justify this as a constructed decoder input required by the task, using elapsed time after 10-frame binning.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script creates the same block-local time vector for every trial: `0, 10/30, 20/30, ...` seconds up to the end of a 2-minute block. Time resets to zero at the start of every block.

ii. ```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. In Step 5 of the notes, the agent explicitly said it was “operationalizing” the requested decoder input as elapsed time within each 2-minute block.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Each neural block gets one time vector of identical length, constructed on the same 10-frame-binned grid. Alignment is therefore by shared block index and shared bin count, with time resetting at each block start.

ii. ```python
neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
...
sess_input.append(tb.astype(np.float32))
```

iii. The agent’s rationale was that the decoder needs a time-varying contextual input on the same temporal grid as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`. The script also loads `tstamps.npy` and `interframe_int.npy`, but only `motion_energy_glob.npy` is actually used to build the final output signal.

ii. ```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
...
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
```

iii. The notes justify this from `data/README.md`, which identifies `motion_energy_glob.npy` as the processed behavioral motion-energy signal.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script preserves motion sample order, pads or truncates it to imaging length with `NaN`s if needed, computes a 10-frame `NaN`-aware mean, pools all valid binned values across all sessions to compute global quantile thresholds, and later converts each block to categorical bins.

ii. ```python
def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned

def nanbin_mean_1d(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (len(x) // bin_frames) * bin_frames
    x = x[:n].reshape(-1, bin_frames)
    valid = np.isfinite(x)
    sums = np.where(valid, x, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    out = np.full(x.shape[0], np.nan, dtype=np.float32)
    nz = counts > 0
    out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    return out

all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
```

iii. The trajectory says the agent originally misused `tstamps`, then decided `tstamps` are seconds rather than frame indices, so the correct alignment should preserve sample order and leave missing frames as `NaN`. The notes also say 10-frame averaging was chosen to match the paper’s denoising.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. All valid binned motion values from all sessions are concatenated, the 20th/40th/60th/80th percentiles are computed, and `np.digitize` maps each time bin into one of five categories. Missing bins are temporarily set to `-1`, then filled from neighboring valid categories if the block is retained.

ii. ```python
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

y = np.digitize(mb, edges, right=False).astype(np.int64)
y[np.isnan(mb)] = -1
...
if not np.all(valid):
    yy = y.copy()
    last = None
    for i in range(len(yy)):
        if yy[i] >= 0:
            last = yy[i]
        elif last is not None:
            yy[i] = last
    nxt = None
    for i in range(len(yy)-1, -1, -1):
        if yy[i] >= 0:
            nxt = yy[i]
        elif nxt is not None:
            yy[i] = nxt
    yy[yy < 0] = 0
    y = yy
```

iii. The notes justify the global quantile thresholds as a way to satisfy the decoder requirement for five equal-percentile bins and to keep class frequencies balanced.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by assuming `motion_energy_glob.npy` is already framewise, preserving sample order, padding shorter recordings with trailing `NaN`s, truncating longer ones, then binning both streams on the same 10-frame grid and trimming to the shared minimum length before block splitting.

ii. ```python
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
neural_binned = bin_neural(dff, BIN_FRAMES)
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)

def session_to_blocks(neural_binned: np.ndarray, motion_binned: np.ndarray):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    neural_binned = neural_binned[:, :n_bins]
    motion_binned = motion_binned[:n_bins]
    ...
```

iii. The trajectory explains the justification clearly: after discovering that `tstamps.npy` contains seconds rather than frame indices, the agent switched to order-preserving alignment because the data README says motion energy is already framewise with occasional missing camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or mismatched motion frames are handled by padding with `NaN`s or truncating to neural length, then `NaN`-aware binning. Blocks with too many missing labels are dropped. Remaining missing categorical labels are forward-filled, then backward-filled, then set to class 0 if still unresolved. Partial trailing bins and partial trailing 2-minute blocks are discarded by integer floor division.

ii. ```python
aligned = np.full(nframes, np.nan, dtype=np.float32)
m = min(nframes, motion.shape[0])
aligned[:m] = motion[:m]

n = (len(x) // bin_frames) * bin_frames
x = x[:n].reshape(-1, bin_frames)

n_blocks = n_bins // BLOCK_BINS

y[np.isnan(mb)] = -1
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
yy[yy < 0] = 0
```

iii. The notes say missing camera frames should be treated as missing values or interpolated. The trajectory shows the agent chose missing-value handling instead of interpolation after fixing its earlier timestamp bug.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive step is the per-session neural preprocessing in `compute_dff`, especially the per-neuron loop that smooths each full-length trace and computes a percentile baseline. Full-dataset loading and storing all motion values for global quantiles also contributes materially. Optional plotting adds more cost when enabled.

ii. ```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base

for idx, (subj, sessdir) in enumerate(selected):
    F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
    dff = compute_dff(F, Fneu, ops)
    ...
    for mb in motion_blocks:
        all_motion_values.append(mb[np.isfinite(mb)])
```

iii. The notes explicitly mention the baseline computation as a likely bottleneck and estimate full-runtime primarily from this session-by-session preprocessing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization candidates are the per-neuron baseline loop in `compute_dff`, the per-block loop in `session_to_blocks`, and the forward/backward fill loops for missing motion categories.

ii. ```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base

for b in range(n_blocks):
    s = b * BLOCK_BINS
    e = (b + 1) * BLOCK_BINS
    neural_blocks.append(neural_binned[:, s:e].astype(np.float32))

for i in range(len(yy)):
    if yy[i] >= 0:
        last = yy[i]
    elif last is not None:
        yy[i] = last
for i in range(len(yy)-1, -1, -1):
    if yy[i] >= 0:
        nxt = yy[i]
    elif nxt is not None:
        yy[i] = nxt
```

iii. The notes mention the baseline loop as a known inefficiency. The other loops are not discussed explicitly there, but they are visible from the code.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats time-vector construction for every block even though all blocks use the same vector. It also performs a two-stage workflow where sessions are first fully prepared and motion values collected globally, then iterated again to build the final output arrays after quantile edges are known. It also repeats several `astype(np.float32)` conversions on arrays that are already float-like.

ii. ```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])

prepared.append({
    'subject': subj,
    'session': sessdir.name,
    'neural_blocks': neural_blocks,
    'motion_blocks': motion_blocks,
    'time_blocks': time_blocks,
    'n_neurons': dff.shape[0],
})
for mb in motion_blocks:
    all_motion_values.append(mb[np.isfinite(mb)])
...
for item in prepared:
    sess_neural = []
    sess_input = []
    sess_output = []
```

iii. The notes explicitly say all sessions are prepared in memory before global motion discretization, which is the main repeated/two-pass part the agent noticed.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `stat.npy` and `interframe_int.npy` but never uses them. It also accepts `tstamps` in `align_motion_to_frames` but the final implementation ignores it. More broadly, it builds per-block time arrays repeatedly even though the same array is reused conceptually for every block.

ii. ```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
return F, Fneu, ops, stat, motion, tstamps, interframe

def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned
```

iii. The trajectory shows this came from an earlier, abandoned attempt to use `tstamps` for alignment. After the bug fix, those arrays were still loaded but no longer used in the final processing.
