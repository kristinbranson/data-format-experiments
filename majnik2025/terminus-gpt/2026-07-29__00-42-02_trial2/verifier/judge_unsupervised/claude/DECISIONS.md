# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subject directories (folders starting with 'jm') under `data/`, then discovers all session directories (folders whose names start with 4 digits) within each subject. For each session, it loads `F.npy`, `Fneu.npy`, `ops.npy`, `stat.npy` from `suite2p/plane0/` and `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` from `move_deve/`. All sessions are iterated over in a single loop, processed, and stored in a list before assembling the final data dictionary.

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

iii. The AI followed the data directory structure described in the data README. It loads Suite2p outputs and behavioral data for all sessions. The notebook (`load_data.ipynb`) also loads data the same way (scanning subject directories, loading F.npy from suite2p/plane0/).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the top-level directories under `data/` whose names start with 'jm'. Each subject folder contains multiple session folders. The AI maintains a `subject_names` list and maps each session to its subject via `subject_names.index(item['subject'])`.

ii.
```python
subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
# ...
subject_names = [p.name for p in subjects]
# ...
data['subject_idx'].append(subject_names.index(item['subject']))
```

iii. This follows the data organization described in the README: "For each subject there is a folder corresponding to the subject id." The 6 subjects (jm031-jm046) match the paper's description of 6 mice.

## 1-c. How are the data split into sessions?

i. Sessions are identified as subdirectories within each subject folder whose names begin with 4 digits (date-based naming like `2023-10-18_a`). Each session directory represents one recording day. Sessions are sorted alphabetically (which gives chronological order).

ii.
```python
sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

iii. The data README states: "Each subject folder contains a number of session folders, each corresponding to one recording day." The filter `name[:4].isdigit()` correctly identifies date-formatted session folders. The 41 total sessions match the data (7+7+7+7+6+7).

## 1-d. How are the data split into trials?

i. Since the raw recordings are continuous (no native trial structure), the AI segments each session into consecutive 2-minute blocks. With a 30 Hz sampling rate and 10-frame binning, each block has 360 time bins (`BLOCK_BINS = 3600 frames / 10 = 360 bins`). Any remaining data at the end of a session that doesn't fill a complete 2-minute block is discarded.

ii.
```python
BLOCK_SECONDS = 120.0
BLOCK_FRAMES = int(FS * BLOCK_SECONDS)  # 3600
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES  # 360

def session_to_blocks(neural_binned, motion_binned):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    n_blocks = n_bins // BLOCK_BINS
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e])
        motion_blocks.append(motion_binned[s:e])
        time_blocks.append((np.arange(BLOCK_BINS) * (BIN_FRAMES / FS))[None, :])
```

iii. The paper states "splits were done on consecutive 2 minute blocks of the recording" for decoding analyses. The 20-minute sessions yield 10 blocks and 30-minute sessions yield 15 blocks, giving 545 total trials across all sessions.

## 1-e. How are trials filtered based on quality controls?

i. Blocks (trials) are filtered based on the fraction of valid (non-NaN) motion energy values. A block is discarded if it has fewer than max(10, 80%) valid time bins. Additionally, sessions with fewer than 2 valid blocks are excluded entirely.

ii.
```python
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
# ...
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. The AI applied quality filtering to ensure blocks have sufficient valid motion energy data. The 80% threshold is a reasonable heuristic. In practice, no blocks were dropped (all 545 trials retained), suggesting missing motion data was minimal after binning.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` (raw fluorescence traces), `Fneu.npy` (neuropil fluorescence traces), and `ops.npy` (Suite2p parameters including neucoeff, baseline parameters). These are loaded from `suite2p/plane0/` for each session.

ii.
```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
```

iii. The data README and notebook indicate F.npy contains raw fluorescence traces and note that "for more proper analysis compute dF/F." The methods text states "baseline-corrected fluorescence traces" (dF/F) were used for analyses. The AI correctly chose F.npy + Fneu.npy over spks.npy to compute dF/F.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) neuropil subtraction: `Fcorr = F - neucoeff * Fneu` using neucoeff from ops.npy (default 0.7); (2) baseline correction: smooth each neuron's trace with a moving average, compute a single global 8th-percentile baseline, then compute `dff = (Fcorr - baseline) / baseline`; (3) temporal binning: average in non-overlapping bins of 10 frames to produce ~0.333s time bins.

ii.
```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))  # 0.7
    Fcorr = F - neucoeff * Fneu
    win = max(1, int(round(win_seconds * float(ops.get('fs', FS)))))
    prct = float(ops.get('prctile_baseline', 8.0))
    for i in range(Fcorr.shape[0]):
        smooth = moving_average_reflect(Fcorr[i], win)
        base = np.percentile(smooth, prct)
        baseline[i] = base
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fcorr - baseline) / baseline
    return dff

def bin_neural(x, bin_frames):
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2)
```

iii. The AI based these decisions on: (1) the paper's methods describing neuropil-corrected baseline-normalized fluorescence; (2) Suite2p parameters from ops.npy (neucoeff=0.7, win_baseline=60, prctile_baseline=8); (3) the paper stating "averaging in bins of 10 consecutive timestamps." However, the baseline computation uses a single global percentile per neuron rather than Suite2p's sliding-window `maximin` method (running minimum of running maximum), which is a simplification.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI uses all neurons present in the pre-released `F.npy` files.

ii.
```python
# No iscell filtering code exists in convert_data.py
# All rows of F.npy are used directly
dff = compute_dff(F, Fneu, ops)
```

iii. The AI verified that the released data already contains only successfully tracked neurons (as stated in the data README: "the data only includes traces for the cells present across all days"). All iscell values were confirmed to be 1 with probabilities > 0.5, so no additional filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to the start of each 2-minute block. Each block begins at a fixed offset from the start of the recording (block_index * 3600 frames). There is no alignment to an external behavioral event -- the blocks are simply consecutive non-overlapping segments of the continuous recording.

ii.
```python
for b in range(n_blocks):
    s = b * BLOCK_BINS
    e = (b + 1) * BLOCK_BINS
    neural_blocks.append(neural_binned[:, s:e])
```

iii. The paper describes decoding using "consecutive 2 minute blocks of the recording" for cross-validation splits. The AI treats each block start as the alignment event. The metadata reports `temporal_alignment_event = 'start of each 2-minute continuous recording block'` with `off_start = 0.0` and `off_end = 120.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has a time bin size of ~333.33 ms (10 frames at 30 Hz). This is achieved by averaging 10 consecutive imaging frames into one bin. Each 2-minute block has 360 time bins.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
# time_bin_size = 1000.0 * BIN_FRAMES / FS = 333.33 ms
```

iii. The paper's methods state "averaging in bins of 10 consecutive timestamps" for decoding, confirming the 10-frame binning. The raw imaging rate is 30 Hz, so 10-frame bins yield ~333 ms resolution.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from any raw data variable. It is synthetically constructed as an evenly-spaced time vector based on the bin index within each 2-minute block, computed from the known frame rate (30 Hz) and bin size (10 frames).

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The AI constructed the time input as time elapsed within each block, ranging from 0 to ~119.67 seconds. This is based on the task specification requiring "time elapsed from the beginning of the experiment" as a decoder input.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as `bin_index * (10 / 30)` seconds, creating a linear ramp from 0 to ~119.67 seconds within each 2-minute block. No raw data variables are used; it is purely computed from the bin structure.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
# Produces: [0.0, 0.333, 0.667, ..., 119.667] for each block
```

iii. The AI interpreted "time from start of experiment" as time within each trial/block rather than cumulative time from the start of the recording session. This resets to 0 at the start of each block.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data because it is constructed from the same bin indices. Each bin in the time vector corresponds one-to-one with each bin in the neural data matrix. Both have exactly 360 time bins per trial.

ii.
```python
for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
    # tb shape: (1, 360), nb shape: (n_neurons, 360)
    sess_input.append(tb.astype(np.float32))
```

iii. Since the time vector is constructed from the same binning parameters, it is guaranteed to be temporally aligned with the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory of each session. This contains the global motion energy computed from videography of spontaneous behavior.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
```

iii. The data README describes this as "processed behavioural data (motion energy extracted from videography of spontaneous behaviour)." The paper describes motion energy as computed from "pixelwise difference... squared... summed across pixels."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves: (1) alignment to imaging frames by direct sample-order mapping (first N motion samples map to first N imaging frames); (2) temporal binning by averaging 10 consecutive frames with NaN handling; (3) discretization into 5 equal-percentile bins using quantile edges computed across all valid motion values from all sessions.

ii.
```python
def align_motion_to_frames(motion, tstamps, nframes):
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned

def nanbin_mean_1d(x, bin_frames):
    n = (len(x) // bin_frames) * bin_frames
    x = x[:n].reshape(-1, bin_frames)
    valid = np.isfinite(x)
    sums = np.where(valid, x, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    out[nz] = (sums[nz] / counts[nz])
    return out

# Discretization:
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8])
y = np.digitize(mb, edges, right=False)
```

iii. The AI aligned motion using simple order-based mapping after discovering that tstamps.npy contains timestamps in seconds (not frame indices). Binning matches the paper's 10-frame averaging. The 5-bin equal-percentile discretization follows the task specification for "five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using `np.digitize` with edges at the 20th, 40th, 60th, and 80th percentiles of all valid motion energy values across all sessions. This produces balanced bins (each ~20% of data). NaN values in motion are handled by nearest-neighbor filling from valid values.

ii.
```python
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
# ...
y = np.digitize(mb, edges, right=False).astype(np.int64)
y[np.isnan(mb)] = -1
# Forward-fill then backward-fill NaN positions
```

iii. The task instructions specify "five equal-percentile bins," which the AI implements using global quantile edges. The overall distribution is perfectly balanced at 20% per bin. The NaN handling uses nearest valid label filling to avoid gaps.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned with neural data through: (1) order-based frame alignment (motion samples placed sequentially into imaging frame positions); (2) both signals are binned with the same 10-frame bins; (3) both are segmented into the same 2-minute blocks using identical bin indices.

ii.
```python
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
neural_binned = bin_neural(dff, BIN_FRAMES)
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
# Both are then split into blocks using the same bin indices
n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
```

iii. The data README states the video was "recorded at 30 Hz" and synchronized to microscope acquisition. The simple order-based alignment is justified by the fact that the motion energy is already framewise, with missing camera frames handled as NaN values.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled in several ways: (1) Motion energy shorter than imaging frames: trailing frames are filled with NaN; (2) NaN values in binned motion: bins containing any valid data use the mean of valid samples; bins with all NaN remain NaN; (3) NaN motion energy labels: filled using forward-then-backward nearest valid value propagation; (4) Blocks with >20% NaN in motion after discretization are discarded; (5) Sessions with fewer than 2 valid blocks are excluded.

ii.
```python
# Motion alignment padding
aligned = np.full(nframes, np.nan, dtype=np.float32)
m = min(nframes, motion.shape[0])
aligned[:m] = motion[:m]

# NaN-aware binning
out = np.full(x.shape[0], np.nan, dtype=np.float32)
nz = counts > 0
out[nz] = (sums[nz] / counts[nz])

# Label filling for remaining NaN
if not np.all(valid):
    last = None
    for i in range(len(yy)):
        if yy[i] >= 0: last = yy[i]
        elif last is not None: yy[i] = last
    # backward fill...

# Block quality threshold
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
```

iii. The data README notes that "In some recordings there might be some missing frames from the camera" and suggests treating them "as missing values for motion energy or they can be interpolated over." The AI's approach of NaN preservation through binning and nearest-neighbor label filling is a reasonable interpretation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation, specifically the `compute_dff` function which loops over all neurons to compute a smoothed trace and percentile baseline. The full conversion takes ~130 seconds for all 41 sessions.

ii.
```python
def compute_dff(F, Fneu, ops):
    # ...
    for i in range(Fcorr.shape[0]):
        smooth = moving_average_reflect(Fcorr[i], win)
        base = np.percentile(smooth, prct)
        baseline[i] = base
```

iii. The CONVERSION_NOTES note "Potential inefficiency: baseline computation loops over neurons and uses a simplified approximation." The full conversion took ~130 seconds, which is within acceptable limits.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the neuron-by-neuron dF/F baseline computation in `compute_dff`. Since the baseline is a single scalar percentile per neuron, the smoothing and percentile computation could be vectorized across neurons. The NaN label filling loop (forward-backward fill) could also be vectorized using numpy operations.

ii.
```python
# Per-neuron loop in compute_dff:
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base

# Per-element NaN filling loop:
for i in range(len(yy)):
    if yy[i] >= 0: last = yy[i]
    elif last is not None: yy[i] = last
```

iii. The AI acknowledged the per-neuron loop as a potential inefficiency but did not optimize it further since the total runtime was acceptable.

## 6-c. What processing does the code repeat multiple times?

i. The code does not significantly repeat processing. Each session is processed once in the main loop. The motion bin edges are computed once after all sessions are prepared. One area of minor repetition is that the motion values are collected into `all_motion_values` during the session loop and then concatenated and processed again for edge computation.

ii.
```python
# During session processing:
for mb in motion_blocks:
    all_motion_values.append(mb[np.isfinite(mb)])
# After all sessions:
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8])
```

iii. The two-pass approach (collect all motion values, then compute edges, then discretize) is necessary because the percentile bins are computed globally. This is not true repetition but a design requirement.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `stat.npy`, `tstamps.npy`, and `interframe_int.npy` but does not use them in any meaningful computation. `stat.npy` contains ROI spatial statistics but is never referenced after loading. `tstamps.npy` and `interframe_int.npy` are loaded but not used for the actual motion alignment (which uses simple order-based mapping instead).

ii.
```python
def load_session(session_dir: Path):
    stat = np.load(pl0 / 'stat.npy', allow_pickle=True)  # loaded but never used
    tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)  # loaded but not used for alignment
    interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)  # loaded but not used
    return F, Fneu, ops, stat, motion, tstamps, interframe
```

iii. These files are loaded because the AI initially intended to use timestamps for frame alignment. After discovering that simple order-based alignment was more appropriate, the loading was left in place but the data is unused. The `stat.npy` loading is entirely unnecessary for the conversion.
