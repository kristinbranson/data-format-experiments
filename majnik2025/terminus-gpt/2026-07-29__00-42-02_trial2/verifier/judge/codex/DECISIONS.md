# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers subjects by scanning `data/` for directories whose names start with `jm`. It discovers sessions by scanning each subject directory for dated subdirectories whose first four characters are digits. For each selected session it loads `F.npy`, `Fneu.npy`, `ops.npy`, `stat.npy`, `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`. It then processes sessions one by one and later splits them into fixed-length blocks.

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
```

iii. In `CONVERSION_NOTES.md`, Step 2 documents the dataset as 6 `jm*` subject folders with dated session folders, each containing `suite2p/plane0` and `move_deve`. Step 5 says this directory structure maps directly to subjects and sessions, and Step 10 says the agent changed motion alignment to “align motion by sample order and pad/truncate to imaging length.”

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories whose names start with `jm`, sorted lexicographically, and stored as `subject_names`.

ii.
```python
subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
subject_names = [p.name for p in subjects]
```

iii. `CONVERSION_NOTES.md` Step 2 states that `data/` contains 6 subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`). Step 5 says “Use one subject per mouse: direct mapping from native organization.”

## 1-c. How are the data split into sessions?

i. Sessions are split as dated subdirectories within each subject directory. The agent sorts them and treats each such directory as one session.

ii.
```python
for subj in subjects:
    sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    sessions_by_subject[subj.name] = sessions
```

iii. `CONVERSION_NOTES.md` Step 2 says each subject contains dated session folders and reports 41 sessions total. Step 5 says “Use one session per recording day.”

## 1-d. How are the data split into trials?

i. The agent does not use the requested 60-second trial structure. It bins the session first, then splits each session into consecutive non-overlapping 2-minute blocks (`BLOCK_SECONDS = 120.0`), treating each block as a trial.

ii.
```python
BLOCK_SECONDS = 120.0
BLOCK_FRAMES = int(FS * BLOCK_SECONDS)
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES

def session_to_blocks(neural_binned: np.ndarray, motion_binned: np.ndarray):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    ...
    n_blocks = n_bins // BLOCK_BINS
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
        motion_blocks.append(motion_binned[s:e].astype(np.float32))
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 5 explicitly justify this by citing the paper’s decoding procedure: “splits were done on consecutive 2 minute blocks of the recording.” The notes say these 2-minute blocks are the “natural trial units” because the data are continuous rather than natively trial-structured.

## 1-e. How are trials filtered based on quality controls?

i. The agent filters block-trials based on motion-label completeness. It drops any block with fewer than `max(10, 0.8 * len(y))` valid motion bins, fills remaining missing labels in surviving blocks, and drops any session that ends up with fewer than two kept blocks.

ii.
```python
y[np.isnan(mb)] = -1
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
    data['input'].append(sess_input)
    data['output'].append(sess_output)
```

iii. `CONVERSION_NOTES.md` Step 10 says the agent decided missing motion frames should remain missing before binning. Given that choice, it added block-level filtering to avoid heavily missing outputs and also relied on the decoder requirement that each session must contain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives neural data from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`, using `ops.npy` only to read preprocessing parameters such as `neucoeff`, `win_baseline`, `fs`, and `prctile_baseline`.

ii.
```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
...
neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
win_seconds = float(ops.get('win_baseline', 60.0))
win = max(1, int(round(win_seconds * float(ops.get('fs', FS)))))
prct = float(ops.get('prctile_baseline', 8.0))
```

iii. `CONVERSION_NOTES.md` Step 5 says the released Suite2p folders contain tracked neurons with matched row order across days, but that `F.npy` contains raw fluorescence traces and therefore should be transformed rather than used directly.

## 2-b. How is the `neural` data processed?

i. The agent computes a custom dF/F-like signal. It subtracts neuropil (`F - neucoeff * Fneu`), smooths each neuron with a reflected moving average over a baseline window, takes a single low percentile from that smoothed trace as a constant baseline for the whole neuron, clamps the baseline at `1e-3`, and returns `(Fcorr - baseline) / baseline`. After that it averages into non-overlapping 10-frame bins.

ii.
```python
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

neural_binned = bin_neural(dff, BIN_FRAMES)
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 6 say the agent chose to compute a “Suite2p-consistent” baseline-normalized neural signal because the notes from the dataset notebook said the provided `F.npy` traces are raw fluorescence and “for more proper analysis compute dF/F.” The notes also acknowledge this is a simplified approximation rather than Suite2p internals.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neural quality filtering is applied in `convert_data.py`. The agent loads `stat.npy` but does not use `stat.npy` or `iscell.npy` to remove neurons, and it keeps all rows present in `F.npy` for each session.

ii.
```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
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

iii. `CONVERSION_NOTES.md` Step 2 says all observed `iscell[:,0] == 1`, and Step 5 says the released session folders already contain only tracked cells present across all days. On that basis the agent treated the provided arrays as already curated enough for conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the start of each artificial 2-minute block, not to session start and not to any natural behavioral event.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'start of each 2-minute continuous recording block',
    'off_start': 0.0,
    'off_end': BLOCK_SECONDS,
}
...
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent chose 2-minute blocks because the paper’s decoder used consecutive 2-minute blocks from continuous recordings, so it treated block start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame averaging at 30 Hz, yielding `10 / 30 = 0.333...` s bins, or 333.33 ms. The agent rebins both neural and motion streams into non-overlapping 10-frame means before splitting into trials.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
...
neural_binned = bin_neural(dff, BIN_FRAMES)
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
...
'time_bin_size': 1000.0 * BIN_FRAMES / FS,
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 5 cite the paper/methods statement that decoding “slightly denoised” both dF/F and behavior traces by averaging in bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time series is not derived from a stored raw data variable. It is synthesized from the binned sample index inside each 2-minute block using the known frame rate and bin size.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
...
'input_names': ['time_elapsed_s'],
```

iii. `CONVERSION_NOTES.md` Step 5 says elapsed time had to be constructed to satisfy the decoder input specification. The agent operationalized that specification as elapsed time within each block rather than reading any experimental timestamp field.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The only processing is deterministic construction from the index of each 10-frame bin: `0, 1, 2, ...` multiplied by `BIN_FRAMES / FS`. The time resets at the start of every 2-minute block.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. `CONVERSION_NOTES.md` Step 5 describes this as creating a `1 x T` time series after 10-frame binning. Step 10’s sanity checks say the recomputed time vectors matched the converted input arrays exactly.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The input is aligned by construction: `time_blocks`, `motion_blocks`, and `neural_blocks` are created from the same binned session slices, so each input time bin corresponds to the same block-local neural time bin.

ii.
```python
neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
motion_blocks.append(motion_binned[s:e].astype(np.float32))
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. `CONVERSION_NOTES.md` Step 10 reports that recomputed raw-vs-converted input and neural blocks matched exactly for checked sessions, which is the agent’s evidence that this blockwise alignment is internally consistent.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy output is derived primarily from `move_deve/motion_energy_glob.npy`. The agent also loads `tstamps.npy` and `interframe_int.npy` as auxiliary timing files for alignment and missing-frame handling.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
```

iii. `CONVERSION_NOTES.md` Step 2 and Step 3 say `motion_energy_glob.npy` stores processed motion energy from spontaneous behavior, while `tstamps.npy` and `interframe_int.npy` identify missing camera frames. Step 10 says the agent eventually chose sample-order alignment with missing values rather than timestamp-indexed insertion.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent converts motion to `float32`, pads/truncates it to imaging length, leaving any missing tail as `NaN`, averages it into 10-frame bins with a NaN-aware mean, and later digitizes the binned values into classes. It does not interpolate dropped frames and does not use `interframe_int.npy` in the final computation.

ii.
```python
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
```

iii. `CONVERSION_NOTES.md` Step 4 notes that the README says missing camera frames “should be treated as missing values or interpolated over.” Step 10 says the agent initially misread `tstamps.npy`, then resolved this by aligning motion by sample order and padding/truncating to imaging length.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent computes one set of global 20/40/60/80% quantile edges from all finite motion values across all blocks from all selected sessions, then uses `np.digitize` to create five categories. Missing binned motion values are first labeled `-1` and then imputed from neighboring valid labels or set to `0`.

ii.
```python
all_motion_values = []
...
for mb in motion_blocks:
    all_motion_values.append(mb[np.isfinite(mb)])
...
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
y = np.digitize(mb, edges, right=False).astype(np.int64)
y[np.isnan(mb)] = -1
...
yy[yy < 0] = 0
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says the plan was to “discretize motion energy into 5 equal-percentile bins using valid binned samples across the full converted dataset,” with the stated goal of keeping class frequencies globally balanced.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent aligns motion to neural data by truncating or padding the motion vector to the number of neural frames, binning both streams, truncating the two binned streams to a shared minimum length, and then slicing corresponding block indices from each.

ii.
```python
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
neural_binned = bin_neural(dff, BIN_FRAMES)
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)

def session_to_blocks(neural_binned: np.ndarray, motion_binned: np.ndarray):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    neural_binned = neural_binned[:, :n_bins]
    motion_binned = motion_binned[:n_bins]
    ...
```

iii. `CONVERSION_NOTES.md` Step 3 says the video and imaging are synchronized but motion arrays can be shorter because of dropped camera frames. Step 10 says the final alignment choice was sample-order matching plus pad/truncate, rather than explicit frame insertion from timing gaps.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or short motion traces are handled by padding with `NaN` up to imaging length, then using NaN-aware bin averaging. Blocks with too many missing motion bins are dropped. Remaining missing class labels are filled by propagating the nearest valid previous label forward, then nearest valid next label backward, and finally defaulting any unresolved values to class `0`. Partial trailing bins and trailing partial 2-minute blocks are implicitly discarded by integer division.

ii.
```python
aligned = np.full(nframes, np.nan, dtype=np.float32)
m = min(nframes, motion.shape[0])
aligned[:m] = motion[:m]

valid = np.isfinite(x)
...
y[np.isnan(mb)] = -1
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
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
yy[yy < 0] = 0
```

iii. `CONVERSION_NOTES.md` Step 3 cites the data README’s statement that missing camera frames should be treated as missing values or interpolated over. Step 10 states the agent intentionally preserved missingness before binning, then added label-filling and block filtering so the decoder would still receive categorical outputs.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step in the agent’s implementation is the custom `compute_dff` baseline calculation because it loops over neurons, performs a large moving-average convolution per neuron, and computes a percentile for each neuron. Optional plotting also adds avoidable overhead.

ii.
```python
baseline = np.empty_like(Fcorr, dtype=np.float32)
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
...
if args.show_processing and idx < 2:
    plot_processing(...)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly identifies baseline computation as a potential inefficiency because it “loops over neurons and uses a simplified approximation.” Step 7 also reports runtime per session, indicating this preprocessing dominates execution more than simple loading and binning.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the per-neuron loop inside `compute_dff`. Additional small loops that could be vectorized are the per-label forward/backward fill loops for missing motion classes and the repeated Python-level block-splitting loop.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base

for i in range(len(yy)):
    ...

for i in range(len(yy)-1, -1, -1):
    ...

for b in range(n_blocks):
    ...
```

iii. `CONVERSION_NOTES.md` Step 6 already flags the baseline loop as a likely inefficiency. The other loops are not called out explicitly there, but they follow directly from the implementation and represent avoidable Python-level iteration.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly rebuilds the same block-local time vector for every block, repeatedly casts arrays to `float32` during block construction and storage, and repeatedly computes subject indices with `subject_names.index(...)` while assembling sessions.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
...
neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
motion_blocks.append(motion_binned[s:e].astype(np.float32))
...
sess_neural.append(nb.astype(np.float32))
sess_input.append(tb.astype(np.float32))
sess_output.append(y[None, :].astype(np.int64))
...
data['subject_idx'].append(subject_names.index(item['subject']))
```

iii. The agent does not explicitly discuss these repetitions in the notes. They are implementation artifacts visible in `convert_data.py`, likely accepted because they are simple and the dataset is modest in size.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `stat.npy` and `interframe_int.npy` but never uses them. It also loads `tstamps.npy` but only passes it through an alignment function that ignores timestamp values. Optional plotting and saving processing figures are also unnecessary for the final converted dataset.

ii.
```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
...
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
return F, Fneu, ops, stat, motion, tstamps, interframe
...
def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    ...
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned
...
if args.show_processing and idx < 2:
    plot_processing(...)
```

iii. `CONVERSION_NOTES.md` shows that the agent investigated tracked-cell metadata and missing-frame timing files during development, but the final code does not use most of that information. The plotting path was kept as a debugging aid rather than a conversion requirement.
