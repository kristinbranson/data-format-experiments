# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by listing directories in the data folder (sorted alphabetically), then lists session subdirectories within each subject folder (also sorted). For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. All subjects and all their sessions are processed (unless `--sample` mode is used).

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
# ...
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
# ...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI noted that the directory structure follows a standard convention from the dataset. It identified the relevant files from exploring the data and reference code (load_data.ipynb). The approach is documented in CONVERSION_NOTES.md Steps 1-2.

## 1-b. How are the data split into subjects?

i. Subjects are identified as all directories in the data folder, sorted alphabetically. Unlike the reference which filters for directories starting with `jm`, the AI includes all subdirectories.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
```

iii. The AI documented finding 6 subjects in the data directory. Since all directories in the data folder happen to be subject directories (jm031-jm046), the lack of a `jm` prefix filter does not cause issues in practice.

## 1-c. How are the data split into sessions?

i. Sessions are identified as all subdirectories within each subject folder, sorted alphabetically. Each subdirectory represents one daily recording session.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. The AI noted 41 total sessions across 6 subjects (7,7,7,7,6,7 per subject), consistent with the reference paper.

## 1-d. How are the data split into trials?

i. The AI splits continuous recordings into fixed-length **2-minute (120 second)** non-overlapping trials. After binning by 10 frames, each trial contains 360 bins. Any remainder bins at the end of a session are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
BIN_SIZE = 10  # number of frames per bin
# ...
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
n_trials = n_bins // trial_frames
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The AI justified this by citing the paper: "splits were done on consecutive 2 minute blocks". The reference solution uses 60-second trials instead.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials are included; only partial trials at session ends are discarded.

ii. No filtering code — only the remainder discard:
```python
n_trials = n_bins // trial_frames
```

iii. The AI noted in CONVERSION_NOTES.md: "Trials: No explicit curation; missing video frames interpolated."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), loaded from `suite2p/plane0/` in each session directory.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are standard suite2p output files. The AI correctly identified these as the source of neural data.

## 2-b. How is the `neural` data processed?

i. The AI applies: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) maximin baseline estimation and subtraction (`dff = Fc - F0`), (3) temporal binning by averaging 10 consecutive frames. The maximin baseline is reimplemented manually using scipy's `minimum_filter1d`, `maximum_filter1d`, and `gaussian_filter1d`, rather than using suite2p's `dcnv.preprocess` directly.

ii.
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0, ...):
    Fc = F - neucoeff * Fneu
    win_frames = int(win_baseline * fs)
    F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
    dff = Fc - F0
    return dff

def _maximin_baseline(Fc, win_frames, sig_frames):
    Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
    Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
    Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
    return Flow

def bin_data(data, bin_size):
    n_bins = n // bin_size
    trimmed = data[..., :n_bins * bin_size]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The AI cited the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)". It also cited: "averaging in bins of 10 consecutive timestamps". The AI noted fixing a bug where sig_baseline was initially multiplied by fs (300 instead of 10).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons present in the suite2p output files are included.

ii. No filtering code present.

iii. The AI noted: "All iscell values are 1.0 in the provided data (pre-filtered by track2p)" — the data has already been filtered by the track2p pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each 2-minute recording block. Since the recording is continuous and trials are artificial segments, no event-based alignment is applied.

ii.
```python
'temporal_alignment_event': 'Start of 2-minute recording block',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

iii. The AI noted there is no stimulus-driven trial structure, so trials are simply contiguous segments of the recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames, producing a time bin size of 333.33 ms (3 Hz effective rate). This is a significant departure from the reference solution which keeps the native 30 Hz (33.33 ms bins).

ii.
```python
BIN_SIZE = 10  # number of frames per bin (paper: "bins of 10 consecutive timestamps")
# ...
dff_binned = bin_data(dff, bin_size)  # (n_neurons, n_bins)
me_binned = bin_data(me_interp, bin_size)  # (n_bins,)
# ...
'time_bin_size': BIN_SIZE / FS * 1000,  # in ms = 333.33
```

iii. The AI justified this by citing the paper: "averaging in bins of 10 consecutive timestamps." The reference solution does not apply this binning.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from bin indices and the effective sampling rate after binning.

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The AI noted that since the frame rate is constant, computing time from indices is equivalent to using stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (BIN_SIZE / FS)`, giving time in seconds from the start of each trial (not from the start of the session/experiment). The range is [0, ~119.67s] for 120s trials.

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. The AI described this as "Time elapsed from start of trial in seconds."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed from the same bin indices as the neural data, so they are inherently aligned. Both have the same number of timepoints per trial.

ii.
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. No explicit alignment needed since time is derived from the neural data's frame count.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect missing video frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI correctly identified these as the pre-computed motion energy and frame timing data.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI applies: (1) missing frame interpolation using interframe intervals, (2) temporal binning by averaging 10 consecutive frames, (3) global quintile discretization across all sessions into 5 bins. Notably, the AI does NOT normalize motion energy by its standard deviation before discretization, unlike the reference.

ii.
```python
# Interpolation
me_interp = interpolate_missing_frames(me, interframe, n_frames)
# Binning
me_binned = bin_data(me_interp, bin_size)
# Discretization
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
binned = np.digitize(me, bin_edges[1:-1])
```

iii. The AI documented these steps in CONVERSION_NOTES.md. It noted that quintile discretization ensures balanced class counts.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins. Bin edges are computed globally across all sessions using `np.percentile` with boundaries at 0, 20, 40, 60, 80, 100th percentiles. Values are assigned to bins using `np.digitize`. Duplicate bin edges are handled by adding a small epsilon (1e-10).

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
for i in range(1, len(bin_edges)):
    if bin_edges[i] <= bin_edges[i-1]:
        bin_edges[i] = bin_edges[i-1] + 1e-10
binned = np.digitize(me, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The AI noted that global quintile bins ensure balanced output distributions (~20% per class).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Missing video frames are interpolated to match the neural frame count. After interpolation, both neural and ME signals are binned by the same factor (10 frames) and split into trials at the same boundaries, ensuring alignment.

ii.
```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
# ...
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
# Same trial splitting applied to both
neural_trials.append(neural_binned[:, start:end])
me_trials.append(me_binned[start:end])
```

iii. The AI verified alignment via processing plots and sanity checks in CONVERSION_NOTES.md Step 10.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing video frames by detecting them through interframe interval analysis. It computes the ratio of each interframe interval to the median, rounds to identify how many frames were missed, and then uses `np.interp` to fill in missing positions. If ME has more frames than neural, it truncates. Partial trials at session ends are discarded.

ii.
```python
def interpolate_missing_frames(motion_energy, interframe_int, n_neural_frames):
    median_ifi = np.median(interframe_int)
    ratios = interframe_int / median_ifi
    missed_counts = np.round(ratios).astype(int) - 1
    me_positions = np.zeros(n_me, dtype=int)
    me_positions[0] = 0
    for i in range(1, n_me):
        me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
    neural_positions = np.arange(n_neural_frames)
    me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
    return me_interp
```

iii. The AI noted that missing video frames must be handled to ensure frame-by-frame alignment between neural and behavioral data.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified the dF/F computation (maximin baseline estimation) as the most time-consuming step, taking ~1.5s per session. The AI's manual reimplementation uses scipy filters rather than suite2p's GPU-accelerated dcnv.preprocess.

ii. N/A (timing reported in CONVERSION_NOTES.md)

iii. The AI noted total conversion time of ~60s for 41 sessions, with dF/F being the dominant cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The missing frame interpolation function contains a loop building `me_positions` array element by element, which could be replaced with a cumulative sum operation. The trial splitting loop is also sequential but has minimal overhead.

ii.
```python
me_positions = np.zeros(n_me, dtype=int)
me_positions[0] = 0
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
```
Could be: `me_positions = np.cumsum(np.concatenate([[0], 1 + missed_counts[:-1]]))`

iii. The AI did not explicitly identify this as a vectorization opportunity. The impact is minimal since the number of ME frames is small relative to the dF/F computation.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each session's data is loaded and processed once in a single pass. The discretization is done as a separate global pass, which is necessary.

ii. N/A

iii. The AI designed a two-phase approach: process all sessions first, then discretize globally, which avoids redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporal binning (averaging 10 frames) is an extra processing step that the reference solution does not perform. While justified by the paper, it reduces temporal resolution and adds computational overhead. Additionally, partial bins at session ends are discarded (minor data loss). The `_plot_processing` function generates plots only when `--show-processing` is enabled, so it is not wasteful during normal runs.

ii.
```python
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
```

iii. The AI justified binning as matching the paper's methodology for decoding analysis.
