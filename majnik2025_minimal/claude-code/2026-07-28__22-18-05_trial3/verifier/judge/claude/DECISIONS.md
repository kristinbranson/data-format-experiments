# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories starting with `jm` in the data directory, sorted alphabetically. Sessions are sorted subdirectories within each subject folder. For each session, calcium data is loaded from suite2p output files (`F.npy`, `Fneu.npy` in `suite2p/plane0/`), and motion energy from `motion_energy_glob.npy` and interframe intervals from `interframe_int.npy` in `move_deve/`.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The agent explored the directory structure and identified subject directories by the `jm` prefix, matching the naming convention in the dataset. It examined the suite2p output structure and motion energy files to determine what to load.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data directory, sorted alphabetically.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
subjects = mice
```

iii. The agent identified the naming convention from directory listing. Each `jm*` directory represents one mouse.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a sorted subdirectory within a subject's folder.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. Each subdirectory contains suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. Trials are defined as **120-second (2-minute)** non-overlapping segments of the continuous recording. With 30 Hz imaging and 10-frame bins, this gives 360 bins per trial. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_S = 120  # 2-minute blocks (paper: "splits were done on consecutive 2 minute blocks")
TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360 bins per trial

n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The agent chose 2-minute blocks based on the paper's description of cross-validation splits: "splits were done on consecutive 2 minute blocks." The agent reasoned this matched the paper's analysis scheme, despite the instructions explicitly stating "Split sessions into 60-second trials."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All trials that fill a complete trial duration are kept.

ii. N/A (no filtering code)

iii. No justification given; there is no trial filtering in either the paper or the instructions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), from `plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The agent identified these as standard suite2p output files from examining the directory structure.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - 0.7 * Fneu`), followed by suite2p's `dcnv.preprocess` which performs baseline estimation and correction using the `maximin` method with a 60s window. The agent initially tried implementing dF/F as (F - baseline) / baseline but found it hurt decoder performance, and reverted to using suite2p's preprocess directly (which returns F - baseline).

ii.
```python
def compute_dff_suite2p(F, Fneu, fs=30.0, neucoeff=0.7,
                        win_baseline=60.0, sig_baseline=10.0):
    Fc = (F - neucoeff * Fneu).astype(np.float32)
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                     device=torch.device('cpu'))
    return dff
```

iii. The agent's reasoning (step 23): "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The agent experimented with dividing by baseline but found suite2p's preprocess (subtraction only) gave better decoder performance. Note: the agent omits `prctile_baseline=8.0` and `batch_size` parameters that the reference passes, and forces CPU device.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in `F.npy` are included.

ii. N/A (no filtering code)

iii. The agent noted (step 39) that iscell values are all marked as true cells, so no further filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'Start of imaging session',
'off_start': 0.0,
'off_end': None,
```

iii. The agent reasoned there is no stimulus event to align to — the recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10  # frames per bin (paper: "averaging in bins of 10 consecutive timestamps")

def bin_data(data, bin_size):
    if data.ndim == 1:
        n = len(data)
        n_bins = n // bin_size
        trimmed = data[:n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
    elif data.ndim == 2:
        n_neurons, n_time = data.shape
        n_bins = n_time // bin_size
        trimmed = data[:, :n_bins * bin_size]
        return trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. The agent cited the paper's methods: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the bin index and bin duration, giving seconds from the start of the session. It is not derived from any raw data variable. The agent uses the **bin center** (adding 0.5 to the bin index).

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)  # (1, n_timepoints)
```

iii. The agent computed time from bin indices since the frame rate is constant at 30 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `(bin_index + 0.5) * (BIN_SIZE / FS)`, using the center of each bin rather than the left edge. The time runs continuously across trials within a session (not reset per trial).

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
```

iii. No explicit justification was given for using bin centers vs. left edges.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time array is computed from the same bin indices used for neural data slicing, so it is inherently aligned.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
    trial_neural = dff_binned[:, start:end]
```

iii. N/A — alignment is trivially ensured by using the same indices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect and interpolate dropped frames.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The agent identified these files from examining the data directory structure.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) dropped frames are detected via interframe intervals exceeding 1.5x the median IFI and interpolated using linear interpolation between neighboring values, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 percentile-based bins with edges computed per session.

ii.
```python
# Dropped frame detection
median_ifi = np.median(ifi)
threshold = median_ifi * 1.5
gap_indices = np.where(ifi > threshold)[0]
missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1

# Discretization
def discretize_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges[-1] = edges[-1] + 1e-10
    bins = np.digitize(values, edges[1:-1])
    return bins, edges
```

iii. The agent examined the interframe interval data and found missing frames at scattered locations. It used a median-based threshold to detect gaps, and linear interpolation to fill them.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session using `np.percentile` with `np.linspace(0, 100, 6)` edges. The agent adds a small epsilon (`1e-10`) to the last edge to handle ties, then uses `np.digitize` with inner edges to produce labels 0-4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(values, percentiles)
edges[-1] = edges[-1] + 1e-10
bins = np.digitize(values, edges[1:-1])  # 0 to n_bins-1
```

iii. The agent followed the instruction to discretize into five equal-percentile bins per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Missing video frames are detected using interframe intervals exceeding 1.5x the median IFI, then filled using linear interpolation. After interpolation, the motion energy is truncated or padded to match the neural frame count. Both streams are then binned and sliced identically.

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)

# Interpolation approach:
median_ifi = np.median(ifi)
threshold = median_ifi * 1.5
gap_indices = np.where(ifi > threshold)[0]
missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
# Linear interpolation at gaps, then pad remaining
```

iii. The agent examined interframe intervals and found occasional dropped frames. It used a relative threshold (1.5x median) rather than an absolute threshold.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected and interpolated (see 4-d). If the motion energy array is longer than neural data, it is truncated. If interpolation doesn't fully fill the gap, the last value is repeated. Remainder bins at the end of a session that don't fill a complete trial are discarded.

ii.
```python
if len(me) == n_neural_frames:
    return me
n_missing = n_neural_frames - len(me)
if n_missing <= 0:
    return me[:n_neural_frames]
# ... interpolation ...
while dst_idx < n_neural_frames:
    result[dst_idx] = result[dst_idx - 1]
    dst_idx += 1
return result[:n_neural_frames]
```

iii. The agent handled edge cases (ME longer, shorter, or equal to neural) but does not have a strict assertion to verify alignment after interpolation — it silently pads if needed.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `dcnv.preprocess` baseline correction, run on CPU. Loading `.npy` files is also significant for larger sessions.

ii. N/A

iii. The agent noted that using CPU (forced via `torch.device('cpu')`) rather than GPU would be slower for baseline correction.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation function uses a per-frame loop that iterates through every source frame, inserting interpolated values one at a time. This could be vectorized by pre-computing insertion positions and using array operations.

ii.
```python
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    dst_idx += 1
    if gap_pos < len(gap_indices) and src_idx == gap_indices[gap_pos]:
        n_insert = missing_per_gap[gap_pos]
        for k in range(n_insert):
            ...
```

iii. No explicit justification. The loop processes every frame individually, which is inefficient for large arrays.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. The code makes a single pass through all sessions, processing each once.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code creates a separate sample dataset (`sample_data.pkl`) with the first two mice, which is not required by the instructions. It also computes and prints detailed summary statistics that are not saved.

ii.
```python
if sample_output_path:
    sample_mice = [0, 1]
    # ... builds and saves sample_data ...
    with open(sample_output_path, 'wb') as f:
        pickle.dump(sample_data, f)
```

iii. The agent created the sample dataset for testing purposes during development.
