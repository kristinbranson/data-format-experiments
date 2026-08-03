# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code discovers subjects as directories under `data/` whose names start with `jm`, then discovers sessions as subdirectories whose first four characters are digits. It flattens all selected `(subject, session)` pairs into one list, loads each session's Suite2p calcium files and movement files, preprocesses them, and only later splits continuous recordings into block-based "trials."

ii.
```python
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

selected = []
for subj in subject_names:
    for sess in sessions_by_subject[subj]:
        selected.append((subj, sess))
```

iii. The notes say the dataset has one subject directory per mouse and one dated recording directory per day, and the trajectory says the AI intentionally mapped "one session per recording day and one subject per mouse" from the native organization.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories named like `jm031`, `jm032`, etc., sorted lexicographically.

ii.
```python
subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
subject_names = [p.name for p in subjects]
```

iii. In `CONVERSION_NOTES.md`, Step 2 documents six subject folders and Step 5 states "Use one session per recording day and one subject per mouse."

## 1-c. How are the data split into sessions?

i. Sessions are split by subdirectories inside each subject folder whose names begin with four digits, and are sorted lexicographically.

ii.
```python
for subj in subjects:
    sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    sessions_by_subject[subj.name] = sessions
```

iii. The notes describe each subject as containing dated session folders, and the trajectory says the AI chose a direct mapping from native subject/day organization.

## 1-d. How are the data split into trials?

i. The AI decided the raw recordings have no native trials and therefore segmented each session into consecutive 2-minute blocks after temporal binning by 10 frames. Each block is treated as one trial.

ii.
```python
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

iii. Step 3 and Step 5 of the notes say the recordings are continuous and that the paper's decoder used consecutive 2-minute blocks, so the AI treated those blocks as the natural trial unit.

## 1-e. How are trials filtered based on quality controls?

i. After block construction, the AI filters out any block whose motion labels are less than 80% valid, with a floor of 10 valid bins. It also drops entire sessions unless at least two blocks survive.

ii.
```python
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

iii. The justification appears in the trajectory rather than the final notes: the AI initially lost sessions because of motion-alignment failures and then kept this valid-label threshold to guard against blocks dominated by missing motion values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `F.npy` and `Fneu.npy`, using parameters from `ops.npy` to choose the neuropil coefficient and baseline settings. It also loads `stat.npy` but does not use it in the computation.

ii.
```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)

def compute_dff(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
```

iii. Step 5 in the notes says the neural field should come from `F.npy + Fneu.npy + ops.npy`, because the notebook indicated `F.npy` was raw fluorescence and more proper analysis should compute a dF/F-like signal.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction, then computes a simplified dF/F-like normalization: for each neuron it smooths the corrected trace with a reflected moving average, takes one global low percentile as the baseline for that neuron, divides by that baseline, and finally averages the result in non-overlapping 10-frame bins.

ii.
```python
def moving_average_reflect(x: np.ndarray, win: int) -> np.ndarray:
    pad = win // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode='reflect')
    kernel = np.ones(win, dtype=np.float32) / win
    y = np.convolve(xp, kernel, mode='valid')
    return y[: x.shape[0]].astype(np.float32)

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

iii. The notes say the AI wanted a "Suite2p-consistent neuropil-corrected / baseline-corrected fluorescence (dF/F-like signal)" and the trajectory shows it chose this custom approximation after concluding that raw `F.npy` should not be used directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies no explicit neuron-quality filtering at conversion time. It does not inspect `iscell.npy` or use `stat.npy` to exclude cells; it relies on the released dataset already containing tracked cells.

ii.
```python
F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
dff = compute_dff(F, Fneu, ops)
...
data['brain_region_idx'].append(np.zeros(item['n_neurons'], dtype=np.int64))
```

iii. Step 5 in the notes says the released Suite2p folders already contain only tracked neurons present across all days, with matched row order across sessions within a subject.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of each 2-minute block, not to a session-wide event. Time is reset to zero at the start of every block, and metadata labels that block onset as the alignment event.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])

'metadata': {
    'temporal_alignment_event': 'start of each 2-minute continuous recording block',
    'off_start': 0.0,
    'off_end': BLOCK_SECONDS,
}
```

iii. The notes explicitly state that the AI chose 2-minute blocks as the trial unit and then operationalized elapsed time within each block.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at a 30 Hz sampling rate, so each time bin is 333.33 ms. The AI explicitly rebins both neural and motion data by averaging over non-overlapping 10-frame windows.

ii.
```python
FS = 30.0
BIN_FRAMES = 10

def bin_neural(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)

'time_bin_size': 1000.0 * BIN_FRAMES / FS,
```

iii. The notes justify this by citing the methods statement that decoding used denoising by averaging in bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not read from a raw file. The AI constructs it synthetically from the bin index within each 2-minute block, using the constants `BIN_FRAMES` and `FS`.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
...
'input_names': ['time_elapsed_s'],
```

iii. Step 5 in the notes says this was "constructed to satisfy decoder input specification" and operationalized as elapsed time within each 2-minute block.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one identical per-block time vector of length `BLOCK_BINS`, starting at 0 and stepping by `10/30` seconds. It does not preserve cumulative elapsed time across the whole session.

ii.
```python
BIN_FRAMES = 10
FS = 30.0
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES

time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The notes say the AI chose elapsed time within block after 10-frame binning, because it treated each block as the aligned trial.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The input time series is aligned by construction with each neural block: the code creates one time vector per 2-minute neural block, with exactly the same number of binned time points.

ii.
```python
neural_blocks, motion_blocks, time_blocks = session_to_blocks(neural_binned, motion_binned)
...
for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
    sess_neural.append(nb.astype(np.float32))
    sess_input.append(tb.astype(np.float32))
```

iii. The AI's rationale is implicit in its block-based trial design: once neural data are segmented into 2-minute blocks after 10-frame binning, the time input is generated on the same grid.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In practice, the output is derived from `motion_energy_glob.npy`. The code also loads `tstamps.npy` and `interframe_int.npy`, but the final alignment routine does not use those arrays beyond accepting `tstamps` as an unused argument.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)

def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned
```

iii. The trajectory shows the AI originally tried to use `tstamps` as frame indices, then abandoned that and switched to sample-order alignment after reading the data README.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI pads or truncates the framewise motion array to the imaging length, leaves unmatched tail positions as `NaN`, averages motion within 10-frame bins while ignoring `NaN`s, pools all valid binned values across sessions to compute global quantile thresholds, discretizes each block, and fills any remaining missing output labels by nearest valid neighbors.

ii.
```python
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
...
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
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

iii. Step 5 of the notes says the plan was to preserve missingness information and discretize motion into five equal-percentile bins after 10-frame averaging. The trajectory later records that the AI changed from timestamp-based placement to sample-order padding/truncation because `tstamps.npy` contained seconds, not frame indices.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded into five categories using global 20th, 40th, 60th, and 80th percentiles computed over all valid binned motion values from all sessions.

ii.
```python
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
y = np.digitize(mb, edges, right=False).astype(np.int64)
'output_values': [[f'bin_{i}' for i in range(5)]],
```

iii. The notes explicitly justify this as satisfying the decoder instruction to discretize motion energy into five equal-percentile bins while keeping class frequencies balanced.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural data by preserving sample order and padding or truncating to the number of imaging frames, then rebins both streams into the same 10-frame bins and finally splits them into the same 2-minute blocks.

ii.
```python
def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned

aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
neural_binned = bin_neural(dff, BIN_FRAMES)
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
neural_blocks, motion_blocks, time_blocks = session_to_blocks(neural_binned, motion_binned)
```

iii. The trajectory explains the final rationale: video was already framewise and microscope-triggered, so after discovering that `tstamps.npy` stored seconds rather than frame indices, the AI chose order-preserving alignment with padding/truncation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or mismatched motion samples by padding the motion trace with trailing `NaN`s or truncating it to imaging length, averaging bins over finite values only, dropping blocks with too few valid motion labels, forward/backward filling remaining missing labels inside retained blocks, and silently discarding leftover frames that do not fill full 10-frame bins or full 2-minute blocks.

ii.
```python
aligned = np.full(nframes, np.nan, dtype=np.float32)
m = min(nframes, motion.shape[0])
aligned[:m] = motion[:m]

def nanbin_mean_1d(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (len(x) // bin_frames) * bin_frames
    x = x[:n].reshape(-1, bin_frames)
    valid = np.isfinite(x)
    ...

if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
yy[yy < 0] = 0
```

iii. Step 10 of the notes says the AI initially misused `tstamps.npy`, then resolved that issue by aligning motion by sample order and pad/truncate behavior based on the data README.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive step is the custom neural preprocessing in `compute_dff`, especially the per-neuron moving-average smoothing and percentile baseline calculation over long continuous traces. Optional plotting and storing all prepared sessions in memory are secondary costs.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
...
prepared.append({
    'subject': subj,
    'session': sessdir.name,
    'neural_blocks': neural_blocks,
    'motion_blocks': motion_blocks,
    'time_blocks': time_blocks,
    'n_neurons': dff.shape[0],
})
```

iii. Step 6 of the notes explicitly identifies the baseline computation loop over neurons as a potential inefficiency and notes that all sessions are prepared in memory before global discretization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious non-vectorized loops are the neuron loop in `compute_dff`, the per-block loop in `session_to_blocks`, and the forward/backward passes used to fill missing motion labels.

ii.
```python
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
...
for i in range(len(yy)-1, -1, -1):
    if yy[i] >= 0:
        nxt = yy[i]
    elif nxt is not None:
        yy[i] = nxt
```

iii. The notes only call out the baseline loop explicitly, but the trajectory also shows the AI was aware that its custom preprocessing and missing-data handling were less efficient than using Suite2p internals.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly constructs the exact same within-block time vector for every block, repeatedly casts arrays to `float32`, and repeatedly performs subject-name lookup with `subject_names.index(...)` while assembling sessions.

ii.
```python
for b in range(n_blocks):
    ...
    time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])

neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
motion_blocks.append(motion_binned[s:e].astype(np.float32))
...
sess_neural.append(nb.astype(np.float32))
sess_input.append(tb.astype(np.float32))

data['subject_idx'].append(subject_names.index(item['subject']))
```

iii. There is no explicit justification in the notes beyond implementation convenience; these repeated operations are side effects of the chosen blockwise assembly strategy.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `stat.npy`, `tstamps.npy`, and `interframe_int.npy` but does not use `stat.npy` at all and does not meaningfully use `tstamps.npy` or `interframe_int.npy` in the final alignment. It also stores session names in the temporary `prepared` structure even though they are not written to the output dataset, and optional plotting produces figures that are not used by downstream decoder training.

ii.
```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
return F, Fneu, ops, stat, motion, tstamps, interframe

prepared.append({
    'subject': subj,
    'session': sessdir.name,
    'neural_blocks': neural_blocks,
    'motion_blocks': motion_blocks,
    'time_blocks': time_blocks,
    'n_neurons': dff.shape[0],
})

if args.show_processing and idx < 2:
    plot_processing(...)
```

iii. The notes and trajectory show that `tstamps.npy` and `interframe_int.npy` were originally part of an abandoned alignment strategy, while `stat.npy` seems to have been loaded for possible QC work that never made it into the final pipeline.
