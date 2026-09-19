# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 6 mice (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each mouse, it scans subdirectories that start with '2' (date-formatted names). For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions

# Loading:
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI identified the 6 mice from examining the data directory structure. Session directories are filtered to those starting with '2' to match the date-based naming convention (e.g., `2024-01-15`). The AI's reasoning shows it examined the directory structure and identified the suite2p and move_deve subdirectories.

## 1-b. How are the data split into subjects?

i. Subjects are hardcoded as a list of 6 mouse IDs. Each mouse directory under `data/` corresponds to one subject.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
subjects = MICE[:]
```

iii. The AI identified the mice by examining the data directory and hardcoded them directly rather than dynamically discovering directories starting with 'jm'.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a mouse folder that starts with '2' is treated as one session, sorted alphabetically (chronologically). Each session contains one daily recording.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions
```

iii. The AI observed that session directories are named with dates (starting with '2' for year 2024/2025), and used this as a filter.

## 1-d. How are the data split into trials?

i. The AI splits data into 120-second (2-minute) non-overlapping trials. After binning by 10 frames, this gives 360 bins per trial. Any remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2-minute trial blocks (as in paper's decoding)
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360 bins per trial

def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
```

iii. The AI's reasoning states: "The paper splits data into consecutive 2-minute blocks, which at 30 Hz gives 3600 frames per trial, or 360 timepoints after binning by 10." The AI chose to follow the paper's methods for trial duration rather than the instructions.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials (those filling a full trial duration) are included.

ii. N/A (no filtering code)

iii. No justification provided since no filtering was applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (F - 0.7 * Fneu), then manually reimplements the suite2p maximin baseline correction using scipy filters (gaussian_filter1d, minimum_filter1d, maximum_filter1d), and then computes dF/F by dividing (Fc - F0) / F0. This differs from the reference which uses suite2p's `dcnv.preprocess` directly (which subtracts the baseline but does not divide by it).

ii.
```python
def compute_dff(F, Fneu):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE_SEC * FS)
    # Suite2p default 'maximin' baseline
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    F0 = maximum_filter1d(Flow, size=win, axis=1)
    # Avoid division by zero/near-zero
    F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    dff = (Fc - F0) / F0_safe
    return dff
```

iii. The AI's reasoning states it is using "Suite2p's standard preprocessing for deconvolution" and "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." However, the AI reimplemented the baseline estimation manually and added a dF/F normalization step (division by baseline) that is not done by suite2p's `dcnv.preprocess`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the suite2p `F.npy` output are included.

ii. N/A (no filtering code)

iii. No justification provided. Suite2p's cell detection has already identified ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy data are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (~333.33 ms time bin). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms

def bin_array(data, bin_size):
    if data.ndim == 1:
        n_bins = len(data) // bin_size
        return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_bins = data.shape[1] // bin_size
        return data[:, :n_bins * bin_size].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)

dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. The AI's reasoning cites the paper: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the trial index, bin index, and bin duration, giving seconds from the start of the session. The AI uses bin centers (adding half a bin width offset).

ii.
```python
for t_i in range(n_trials):
    start_sec = t_i * TRIAL_DURATION_SEC
    time_bins = np.linspace(
        start_sec + BIN_DURATION_MS / 2000,  # center of first bin
        start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,  # center of last bin
        TRIAL_BINS
    )
    input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. Since the frame rate is constant at 30 Hz, time can be computed from bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes time as the center of each time bin using `np.linspace`. For each trial, the time starts at `trial_index * TRIAL_DURATION_SEC + BIN_DURATION_MS/2000` and ends at `trial_index * TRIAL_DURATION_SEC + TRIAL_DURATION_SEC - BIN_DURATION_MS/2000`.

ii. (Same as 3-a)

iii. The AI chose to use bin centers rather than bin left edges. This is a minor design choice.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed to have the same number of bins as the neural data per trial, so alignment is automatic. Each time bin center corresponds to one neural time bin.

ii. N/A (alignment is implicit by construction)

iii. Both time and neural data share the same binning structure.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect and interpolate dropped frames.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) dropped frames are detected via interframe intervals exceeding 1.5x the median interval and interpolated using `np.interp`, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 percentile-based bins with edges computed **globally** across all sessions.

ii.
```python
# Alignment (dropped frame detection)
median_ifi = np.median(interframe_int)
n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))

# Global percentile computation
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)

# Discretization
binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The AI's reasoning discusses computing quintile boundaries. It considered per-session vs global percentiles but chose global percentiles. The AI noted the skewed distribution in sample data was due to global edges being applied to sessions with different motion levels.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using **global** percentile edges computed across all sessions. `np.digitize` maps values into bins 0 through 4.

ii.
```python
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)

binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to N_OUTPUT_BINS-1
```

iii. The AI did not explicitly justify global vs per-session percentiles. The instructions to the AI said "normalized and discretized into five equal-percentile bins" without specifying per-session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Dropped video frames are detected by comparing each interframe interval to the median: if `interframe_int[i] / median_ifi - 1` rounds to a positive integer, that many frames were dropped. Dropped frames are filled by NaN and then linearly interpolated using `np.interp`. After alignment, the lengths should match.

ii.
```python
def align_motion_energy(me, interframe_int, n_neural_frames):
    median_ifi = np.median(interframe_int)
    aligned = np.full(n_neural_frames, np.nan)
    neural_idx = 0
    for i in range(len(me)):
        if neural_idx < n_neural_frames:
            aligned[neural_idx] = me[i]
        neural_idx += 1
        if i < len(interframe_int):
            n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
            neural_idx += n_dropped
    nans = np.isnan(aligned)
    if np.any(nans) and not np.all(nans):
        x = np.arange(n_neural_frames)
        aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
    return aligned
```

iii. The AI's reasoning discusses using interframe intervals to detect where frames were dropped. It uses a median-based threshold rather than a fixed threshold.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected and interpolated (see 4-d). Remainder frames at the end of a session that don't fill a complete trial are discarded. The AI also checks for NaN/Inf in the final neural data.

ii.
```python
# NaN/Inf check
nan_count = sum(np.any(np.isnan(t)) for s in neural_all for t in s)
inf_count = sum(np.any(np.isinf(t)) for s in neural_all for t in s)
```

iii. The AI performs sanity checks including verifying neuron counts are consistent across sessions for each mouse and checking for NaN/Inf values.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the manual baseline correction using scipy filters (gaussian_filter1d, minimum_filter1d, maximum_filter1d) applied to the full session fluorescence data for every neuron. The AI's trajectory shows this took ~30+ minutes, and the AI had to monitor and troubleshoot it.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
F0 = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The AI's trajectory shows it initially tried using scipy's `percentile_filter` which was even slower, then switched to the gaussian+min+max approach. The AI noted this was very slow on large arrays.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `align_motion_energy` function iterates frame-by-frame through the motion energy data in a Python loop to detect and place dropped frames. This could be vectorized.

ii.
```python
for i in range(len(me)):
    if neural_idx < n_neural_frames:
        aligned[neural_idx] = me[i]
    neural_idx += 1
    if i < len(interframe_int):
        n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
        neural_idx += n_dropped
```

iii. N/A

## 6-c. What processing does the code repeat multiple times?

i. The code processes all sessions even in sample mode (to compute global percentile bin edges), then subsets to sample sessions. This means the full computation is done regardless of whether a sample or full dataset is requested.

ii.
```python
# In convert_data(), sample_only flag only affects saving, not processing
if not sample_only:
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

# Sample is always saved
sample_data = {
    'neural': data['neural'][:n_sample],
    ...
}
```

iii. The AI's trajectory confirms this: "the `--sample-only` flag still processes ALL the data first to get global percentile bin edges, it just doesn't save the full file."

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes dF/F normalization (dividing by baseline F0) which adds processing on top of the baseline subtraction. The downstream decoder would work with either baseline-subtracted or dF/F-normalized data, but the division step is extra processing not done by the reference.

ii.
```python
dff = (Fc - F0) / F0_safe  # Division by baseline is extra
```

iii. N/A
