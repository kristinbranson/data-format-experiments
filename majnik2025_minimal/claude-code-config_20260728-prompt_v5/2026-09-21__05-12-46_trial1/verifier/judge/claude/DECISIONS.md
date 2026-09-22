# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified from a hardcoded list of mouse names (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). Sessions are subdirectories within each subject folder, filtered to those starting with a digit. For each session, calcium data is loaded from suite2p output files (`F.npy`, `Fneu.npy`) and `ops.npy`, and motion energy from `motion_energy_glob.npy`.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy')).astype(np.float64)
```

iii. The agent examined the data directory structure, finding `jm*` directories containing session subdirectories. It loaded `ops.npy` to extract processing parameters (fs, neucoeff, sig_baseline, win_baseline) rather than hardcoding them, using `ops.get()` with defaults matching suite2p's standard values.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of 6 mouse names rather than dynamically scanning the directory.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for mouse_idx, mouse in enumerate(MICE):
    mouse_dir = os.path.join(DATA_DIR, mouse)
```

iii. The agent identified the 6 mouse directories in the data folder and hardcoded them, ensuring a deterministic and explicit subject list.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject's folder, filtered to those whose name starts with a digit (to exclude non-session directories like `track2p`), sorted alphabetically.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions
```

iii. Session directories are named by date (e.g., `2023-10-18_a`) which always start with a digit. The `d[0].isdigit()` filter excludes non-session subdirectories like `track2p`.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are defined as 60-second non-overlapping segments of the continuous recording (180 bins per trial at 3 Hz). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
bins_per_trial = int(TRIAL_DURATION / bin_duration)  # 180 bins per trial
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
```

iii. Per instructions, sessions are split into 60-second trials. Since there is no stimulus-driven trial structure, fixed-length segmentation from session start is the natural approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second trials are included; only the remainder that doesn't fill a full trial is discarded.

ii. N/A (no filtering code)

iii. The agent did not discuss trial-level quality controls. The instructions and paper do not specify trial-level filtering criteria for this dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (processing parameters), from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. These are standard suite2p output files. The agent loaded ops.npy to read processing parameters rather than hardcoding them.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - 0.7 * Fneu`), followed by a manually implemented maximin baseline correction: Gaussian smoothing, minimum filter, maximum filter, then baseline subtraction (`dff = Fc - Flow`). The agent reimplemented suite2p's baseline correction using scipy.ndimage filters rather than calling `dcnv.preprocess` directly.

ii.
```python
def compute_dff(F, Fneu, fs, neucoeff=0.7, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    dff = Fc - Flow
    return dff
```

iii. The agent initially implemented dF/F as `(Fc - Flow) / Flow` (division), but after seeing poor decoder performance (barely above chance at 0.2263), it discovered extreme values caused by near-zero baselines. The agent then re-examined the Track2p `F_processing` code and found it uses subtraction only, not division. It revised to `Fc - Flow` accordingly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All neurons in `F.npy` are included. The docstring claims neurons "pass iscell > 0.5 threshold" but the code does not apply any such filter.

ii. N/A (no filtering code in the processing pipeline)

iii. The agent reasoned that the Track2p pipeline had already selected tracked neurons, so all neurons in the provided data are valid. No explicit iscell filtering was implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION),
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy data are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (333.33 ms per bin). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me[np.newaxis, :], BIN_SIZE)[0]
```

iii. The paper's Methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps". The agent applied this to both streams to keep them synchronized.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from bin indices and the bin duration (BIN_SIZE / fs seconds per bin).

ii.
```python
bin_duration = BIN_SIZE / fs
time_vec = np.arange(n_bins) * bin_duration
```

iii. Since the frame rate is constant at 30 Hz with known bin size, time can be computed directly from indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (BIN_SIZE / fs)`, giving seconds from session start. Each bin's time is the left edge of the bin.

ii.
```python
time_vec = np.arange(n_bins) * bin_duration  # (n_bins,)
session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
```

iii. Straightforward computation from known constants. No additional processing needed.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices used for the neural data, so it is inherently aligned. The time vector spans the full session and is sliced into trials at the same boundaries as the neural data.

ii.
```python
time_vec = np.arange(n_bins) * bin_duration
# ...
session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
```

iii. N/A - alignment is inherent from shared indexing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived solely from `motion_energy_glob.npy` in the `move_deve` subdirectory. Unlike the reference, the agent does NOT load `interframe_int.npy` for dropped frame detection.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy')).astype(np.float64)
```

iii. The agent used the pre-computed global motion energy file from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) truncation to common length with neural data, (2) averaging into 10-frame bins, (3) discretization into 5 percentile-based bins per session. Bin edges are computed with `np.percentile` and outer edges replaced with +/-infinity.

ii.
```python
common_len = min(n_frames, len(me))
me = me[:common_len]
me_binned = bin_data(me[np.newaxis, :], BIN_SIZE)[0]
percentiles = np.linspace(0, 100, N_ME_BINS + 1)
bin_edges = np.percentile(me_binned, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
me_discrete = np.digitize(me_binned, bin_edges[1:])
me_discrete = np.clip(me_discrete, 0, N_ME_BINS - 1)
```

iii. The agent chose truncation over interpolation to handle frame count mismatches. Discretization uses equal-percentile bins per session as specified in the instructions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session using `np.percentile` to compute bin edges at 0th, 20th, 40th, 60th, 80th, 100th percentiles, then `np.digitize` to assign bins. The outer edges are set to +/-infinity and output is clipped to [0, 4].

ii.
```python
percentiles = np.linspace(0, 100, N_ME_BINS + 1)
bin_edges = np.percentile(me_binned, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
me_discrete = np.digitize(me_binned, bin_edges[1:])
me_discrete = np.clip(me_discrete, 0, N_ME_BINS - 1)
```

iii. Per instructions: "five equal-percentile bins, selected per session." The agent implemented per-session percentile-based binning.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When the motion energy frame count differs from the neural frame count (due to dropped video frames), the agent truncates both to the minimum length. This is different from the reference approach of interpolating dropped frames.

ii.
```python
common_len = min(n_frames, len(me))
F = F[:, :common_len]
Fneu = Fneu[:, :common_len]
me = me[:common_len]
```

iii. The agent treated frame count mismatches as minor edge cases and resolved them by discarding the extra frames. No explicit reasoning was given for choosing truncation over interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Frame count mismatches between neural and motion energy data are handled by truncating both to the shorter length. Remainder bins at the end of a session that don't fill a complete trial are discarded.

ii.
```python
common_len = min(n_frames, len(me))
F = F[:, :common_len]
Fneu = Fneu[:, :common_len]
me = me[:common_len]
```

iii. The agent chose the simplest approach: discard data that doesn't align or doesn't fill complete trials.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the manual baseline correction (`compute_dff`), which applies three sequential filtering operations (Gaussian, minimum, maximum) over the full session length for every neuron. Loading `.npy` files is also I/O-bound but relatively fast.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The three filter operations each process the entire neuron-by-time matrix. Unlike the reference which uses suite2p's GPU-accelerated `dcnv.preprocess`, the agent's scipy implementation runs on CPU only.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials and appends slices one at a time. This could be replaced with array reshaping to split all trials at once. However, the performance impact is negligible.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    session_neural.append(dff_binned[:, start:end].astype(np.float32))
    session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
    session_output.append(me_discrete[start:end][np.newaxis, :].astype(np.int64))
```

iii. The loop is straightforward and the number of iterations is small (~30 trials per session), so vectorization would provide minimal benefit.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session's neural data and motion energy are processed once.

ii. N/A

iii. The code follows a single-pass structure: process each session, discretize, then assemble into trials.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No clearly unnecessary processing is performed. The code processes only what is needed for the output format.

ii. N/A

iii. The code is relatively lean, converting data directly into the target format.
