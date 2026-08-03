# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads each subject's `.mat` file using `mat73.loadmat`. It iterates over a hardcoded list of 7 animal names (`ANIMALS`), loading each `.mat` file from the data directory. Each file contains `trace`, `position`, and `envs` fields, with each field being a list of arrays indexed by session.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

for animal_idx, animal in enumerate(ANIMALS):
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
    n_sessions = len(dat['trace'])
```

iii. The AI used `mat73.loadmat` (a Python library for MATLAB v7.3 HDF5 files) rather than `h5py` directly. It hardcoded the animal names rather than discovering them from the directory. Both approaches are functionally equivalent for loading the HDF5-format `.mat` files.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject (mouse). The subject names are hardcoded in the `ANIMALS` list. All subjects are always included in `data['subjects']` regardless of how many sessions are processed.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
data = {
    ...
    'subjects': ANIMALS,
    'subject_idx': np.array(subject_idx_list, dtype=int),
    ...
}
```

iii. The AI identified the 7 mice from the data files and paper. Hardcoding ensures consistent ordering.

## 1-c. How are the data split into sessions?

i. Each recording session within a `.mat` file becomes a separate session in the output. The AI iterates over `dat['trace']` entries (one per recording session). Sessions with 0 valid neurons or fewer than 2 trials are skipped.

ii.
```python
n_sessions = len(dat['trace'])
for sess_idx in range(n_sessions):
    trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
    ...
    if n_valid == 0:
        print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
        continue
    ...
    if n_full_trials < 2:
        print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
        continue
```

iii. The AI noted the instructions require at least 2 trials per session for decoder evaluation, so it added a filter for sessions with fewer than 2 complete trials.

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into non-overlapping 1-minute (1800 frame) segments. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
...
n_full_trials = n_timepoints // FRAMES_PER_TRIAL

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. The instructions specify "1-minute trials" and the paper states sessions are 40 minutes, yielding ~39-40 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No per-trial filtering is applied. All complete 1-minute trials are kept. Sessions with fewer than 2 complete trials are skipped entirely.

ii.
```python
if n_full_trials < 2:
    print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
    continue
```

iii. The AI's CONVERSION_NOTES.md states: "No velocity filtering or place cell filtering applied, as these are analysis-specific in the paper."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the `.mat` files, which contains binary rising-phase calcium transients (0/1 values).

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
```

iii. The AI identified that `trace` contains the pre-processed binary calcium transients as described in the paper.

## 2-b. How is the `neural` data processed?

i. The AI filters out all-NaN neurons (not recorded in a session), replaces any remaining NaN values with 0 using `nan_to_num`, and casts to float32. The data is already in (neurons, timepoints) format from `mat73.loadmat`.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md: "NaN values indicate neurons not detected/tracked in that session (CellReg tracking across days). NaN neurons are excluded, and any remaining NaN values in valid neurons are set to 0."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons that are entirely NaN (not recorded in the session) are removed. No additional quality filtering (e.g., firing rate thresholds, place cell criteria) is applied.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
```

iii. The AI chose to include all valid neurons for "maximum information available to the decoder."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous and trials are artificial 60-second segments starting from the beginning of the session. Alignment is to the start of the recording session.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. There is no stimulus event to align to; the experiment is free exploration. The AI set `temporal_alignment_event` to "Start of recording session" in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The 30 Hz sampling rate matches the paper's description and no rebinning was needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the environment geometry from the `envs` field in the `.mat` file, which contains the name of the environment (e.g., 'square', 'o', 't') for each session. This is then mapped to a 3x3 binary matrix using the `get_env_mat` function.

ii.
```python
env_name = dat['envs'][sess_idx][0]
env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)

def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
```

iii. The AI replicated the `get_env_mat` function from the reference code, which encodes the environment geometry as a binary accessibility matrix. The AI stated this matches the paper: "We partitioned an open square (75 x 75 cm) into a 3 x 3 grid space."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is looked up in a dictionary mapping to 3x3 binary matrices (1=accessible, 0=blocked). The matrix is flattened to a 9-element vector and cast to float32. This is static per trial (same for all trials in a session).

ii.
```python
env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)
trial_input = env_mat.copy()
```

iii. The encoding represents the physical geometry of the arena barriers. All 10 environments have unique patterns.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the `.mat` file, which contains 2D coordinates (x, y) of the mouse in the arena.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
```

iii. The position was tracked using DeepLabCut and stored at the same 30 Hz frame rate as the neural data.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 categories). Bin edges are computed from the maximum position value in each dimension (data-driven), not from the fixed 75 cm arena size. The bin index formula is `x_bin * n_bins + y_bin` (x-major ordering).

ii.
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS):
    pos = position.copy()
    max_vals = np.nanmax(pos, axis=1, keepdims=True)
    bin_size = (max_vals + BUFFER) / n_bins
    binned = np.floor(pos / bin_size).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)
    bin_idx = binned[0] * n_bins + binned[1]
    return bin_idx
```

iii. The AI used data-driven bin edges (based on the max observed position values per session) rather than fixed 75 cm edges.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (0-8) using a 3x3 spatial grid. Each dimension is divided into 3 equal bins. Values are clipped to [0, n_bins-1]. The output is stored as int64 with shape (1, n_timepoints).

ii.
```python
bin_size = (max_vals + BUFFER) / n_bins
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]
...
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The 3x3 grid produces 9 discrete position categories as specified in the instructions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same 30 Hz frame rate, so they are aligned frame-for-frame. Both are split into trials using the same start/end indices.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. Same indexing ensures alignment between neural and position data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neurons (not recorded in a session) are removed. Any remaining NaN values in valid neurons are replaced with 0 via `nan_to_num`. Sessions with 0 valid neurons or fewer than 2 trials are skipped. Remainder frames are discarded.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
...
if n_valid == 0:
    continue
if n_full_trials < 2:
    continue
```

iii. The AI explicitly handles NaN values and edge cases in session/trial counts.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the `.mat` files using `mat73.loadmat`, which involves reading large HDF5 files. The `mat73` library loads all data at once per file, which is I/O intensive for files containing large neural trace arrays.

ii. N/A

iii. The `.mat` files contain large neural trace arrays (e.g., 942 neurons x ~72000 timepoints per session, 31 sessions). Processing (NaN filtering, discretization, trial splitting) is comparatively fast.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop creates individual trial arrays one at a time using Python for-loops, which could potentially be vectorized using `np.split` or array reshaping.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_input = env_mat.copy()
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
    session_neural.append(trial_neural)
```

iii. The loop is simple and the number of iterations is small (~40 trials per session), so the performance impact is minimal.

## 6-c. What processing does the code repeat multiple times?

i. The `env_mat.copy()` is called for each trial within a session, even though the environment geometry is constant across all trials in a session. The `discretize_position` function computes `nanmax` for position data that could be computed once per session.

ii.
```python
trial_input = env_mat.copy()  # repeated for each trial
```

iii. These are minor redundancies with negligible performance impact.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes position discretization for the full session including remainder frames that are later discarded (not split into complete trials). The `create_sample` function processes all data before creating a subset.

ii.
```python
pos_bins = discretize_position(position, N_SPATIAL_BINS)  # all timepoints
# but only frames up to n_full_trials * FRAMES_PER_TRIAL are used
```

iii. Computing position bins for discarded frames is minimal overhead.
