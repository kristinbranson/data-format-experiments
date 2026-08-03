# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads per-animal joblib files from the `data/` directory using `joblib.load()`. Each file is keyed by the animal ID string (e.g., `'QLAK-CA1-08'`). The seven animal IDs are hardcoded. From each animal's dictionary, it extracts `trace`, `position`, `envs`, and `blocked` arrays. Each animal's data is then iterated day-by-day (session-by-session) to produce trials.

ii.
```python
ANIMAL_IDS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]

# In convert_dataset:
for animal_id in animal_ids:
    animal = load_animal(animal_id, data_dir)
    envs = np.array(animal['envs']).squeeze()
    traces = np.asarray(animal['trace'])
    positions = np.asarray(animal['position'])
    blocked = animal['blocked']
```

iii. The AI identified from the reference code README and demo notebooks that `load_dat(animal, p, format="joblib")` loads per-animal data with fields including `trace`, `position`, `envs`, `blocked`, `maps`, `SFPs`, and `centroids`. The AI replicated this loading pattern directly using `joblib.load`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the hardcoded list of 7 animal IDs. Each animal's data file is loaded separately, and a `subject_idx` array maps each session to its corresponding subject index. In `--sample` mode, only the first 2 animals are processed.

ii.
```python
animal_ids = ANIMAL_IDS if mode == 'full' else ANIMAL_IDS[:2]
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# ...
data['subject_idx'].append(subject_to_idx[animal_id])
```

iii. The AI identified 7 animals from the data directory and reference code. This matches the data files available.

## 1-c. How are the data split into sessions?

i. Each recording day for each animal is treated as one session. The number of days is determined from the first dimension of the `traces` array (`traces.shape[0]`). Sessions with fewer than 2 full 1-minute trials are skipped. This produces 207 total sessions (31 days each for 6 animals + 21 days for QLAK-CA1-51).

ii.
```python
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx], ...
    )
    if len(neural_trials) < 2:
        continue
    data['neural'].append(neural_trials)
```

iii. The AI noted from the paper that "All sessions were 40 min, and one session was recorded per day" and used this to define sessions as days. The consistency check in CONVERSION_NOTES.md confirmed 207 total sessions.

## 1-d. How are the data split into trials?

i. Each session (day) is split into contiguous 1-minute trials of exactly 1800 frames (at 30 Hz). The number of full trials is computed as `n_frames // FRAMES_PER_TRIAL`. Trailing partial-minute frames are dropped.

ii.
```python
FRAME_RATE_HZ = 30.0
TRIAL_SECONDS = 60.0
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)  # = 1800

n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
```

iii. The instructions specified "split into 1-minute trials within each session." The AI implemented this as 1800-frame chunks. Sessions with ~71866 frames get 39 trials; those with ~72060+ frames get 40 trials.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial-level quality filtering is applied. The only filtering is that sessions with fewer than 2 full trials are skipped (which never actually triggers since all sessions have 39-40 trials). Trailing partial-minute frames are dropped but not flagged.

ii.
```python
if len(neural_trials) < 2:
    continue
```

iii. The AI did not identify any trial-level quality filtering criteria in the reference paper or code. The reference paper's analyses operate within full sessions without trial-level exclusion, so the AI did not add any.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field of each animal's data dictionary. This contains binary rising-phase calcium event traces where 1 indicates a significant calcium event and 0 indicates no event.

ii.
```python
traces = np.asarray(animal['trace'])
# In process_day:
day_trace  # = traces[day_idx], shape (n_neurons_registered, n_frames)
```

iii. The AI documented that `trace` contains "rise-extracted calcium traces" and that the binary rising-phase vector "was treated as the firing rate in all subsequent analyses." This matches the reference code README and methods.txt.

## 2-b. How is the `neural` data processed?

i. Processing involves: (1) filtering out neurons that are all-NaN for a given day, (2) replacing remaining NaN values with 0, and (3) casting to float32. No additional processing such as smoothing, rate map computation, or temporal binning is applied.

ii.
```python
def process_day(day_trace, day_pos, ...):
    valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
    trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The AI noted that the `trace` data is "analysis-ready" and does not need dF/F recomputation or deconvolution. The reference code uses `trace` directly for analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural filtering is removal of neurons that are entirely NaN for a given day. No place cell identification (split-half reliability), minimum firing rate, or spatial information criteria are applied. All non-all-NaN neurons are included.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The AI acknowledged the paper's split-half reliability method for identifying place cells but did not implement it as a filter. The CONVERSION_NOTES.md Step 3 mentions place cell identification via "split-half rate maps" and "99th percentile threshold" but the conversion code does not apply this filtering. The neuron curation rules section of CONVERSION_NOTES.md was left as placeholder "[Describe rules]".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to session start. Each session (day) starts from frame 0, and trials are contiguous 1-minute (1800-frame) chunks from the beginning of the session. There is no alignment to a specific behavioral event — the alignment event is simply the session/day start.

ii.
```python
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
```

iii. The AI set `temporal_alignment_event` to `'Session/day start; sessions split into contiguous 1-minute trials'` with `off_start: 0.0` and `off_end: 60.0` seconds.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of 30 Hz, giving a time bin size of ~33.33 ms. No temporal rebinning is applied — the data is kept at the original frame rate.

ii.
```python
FRAME_RATE_HZ = 30.0
# In metadata:
'time_bin_size': 1000.0 / FRAME_RATE_HZ,  # = 33.33 ms
```

iii. The AI noted from the methods that "position trajectories recorded at 30 Hz" and that trace and position share the same frame dimension. The AI chose to preserve the native framewise resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from two raw data variables: `envs` (environment shape name per day, e.g., 'square', 'o', 't') and `blocked` (blocked partition indices in a 3x3 layout per day).

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']
# Combined:
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The AI identified from the reference code that `envs` gives environment identity and `blocked` stores blocked partition locations in a 3x3 layout indexed as `[[0,1,2],[3,4,5],[6,7,8]]`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Two masks are computed and element-wise multiplied: (1) `env_to_mask()` converts the environment name to a 3x3 binary matrix indicating the shape geometry, and (2) `blocked_to_mask()` converts blocked partition indices to a 3x3 matrix with 0s at blocked locations. The combined mask is flattened to a 9-element vector, which is static (same for all timepoints within a trial).

ii.
```python
def env_to_mask(env_name):
    # Maps environment names to 3x3 binary masks
    if env == 'square': return np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.float32)
    elif env == 'o': return np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)
    # ... 10 environment types total

def blocked_to_mask(entry):
    # Converts blocked indices to 3x3 mask with 0s at blocked positions
    mask = np.ones((3, 3), dtype=np.float32)
    for v in vals:
        if v == -1: continue
        r, c = divmod(int(v), 3)
        mask[r, c] = 0.0
    return mask

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)
```

iii. The AI identified the 3x3 environment mask system from the reference code's helper functions and README documentation of the `blocked` field. The combined mask captures both the inherent shape (e.g., T-shape only has certain bins accessible) and any additionally blocked partitions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field, which contains x-y coordinates per frame per day, shape `(n_days, 2, n_frames)`.

ii.
```python
positions = np.asarray(animal['position'])
pos = np.asarray(day_pos, dtype=np.float32)  # day_pos = positions[day_idx]
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The AI noted that `position` is the x-y position per day/frame from DeepLabCut head tracking.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x-y position is discretized into a single 3x3 grid label (0-8). The x and y coordinates are independently binned into 3 equal-width bins based on the observed min/max range within each session (day). The combined bin label is `y_bin * 3 + x_bin`.

ii.
```python
def discretize_position_to_3x3(pos_xy):
    x, y = pos_xy[0], pos_xy[1]
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    xmin, xmax = np.min(xv), np.max(xv)
    ymin, ymax = np.min(yv), np.max(yv)
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]
```

iii. The AI chose to discretize position within each session/day using the observed x/y range for that session. This is documented in the metadata as "Position discretized independently within each session/day into 3x3 bins using observed x/y range."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized into 9 bins (0-8) using equal-width binning of the x and y coordinates independently into 3 bins each, then combined as `y_bin * 3 + x_bin`. Invalid (NaN) positions are labeled as -1. The bin edges are determined by the session-wide min/max of each coordinate.

ii.
```python
xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
labels = y_out * 3 + x_out
labels[~valid] = -1
```

iii. The instructions specify "Mouse position discretized into 3 x 3 = 9 spatial bins." The AI implemented this with equal-width binning based on observed position ranges per session. The reference paper uses 5cm x 5cm bins for rate maps, but the AI adapted to the required 3x3 categorization.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned frame-by-frame since both share the same frame dimension in the raw data. When trace and position have different lengths, the minimum length is used (truncation). Within each trial, the same frame indices are used for both neural and position data.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
# Then same slice for both:
neural_trials.append(trace[:, s:e])
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The AI noted that "position and trace share same frame dimension within animal" at 30 Hz, so frame-by-frame alignment is natural.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/problematic data are handled: (1) Neurons that are all-NaN for a day are excluded. (2) Remaining NaN values in neural traces are replaced with 0 via `np.nan_to_num`. (3) Non-finite position values are marked as invalid and assigned label -1 in the output. (4) If trace and position have different frame counts, the minimum is used. (5) Sessions with fewer than 2 full trials are skipped.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
# Position:
valid = np.isfinite(x) & np.isfinite(y)
labels[~valid] = -1
# Sessions:
if len(neural_trials) < 2:
    continue
```

iii. The AI documented that trailing partial-minute frames are dropped and that the code handles ragged `blocked` entries robustly via `normalize_blocked_entry()`.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each animal's joblib file, which contains large arrays (e.g., trace arrays of shape `(31, 515, 71866)` for hundreds of neurons across 31 days). The actual processing (slicing, masking, discretization) is fast since it's vectorized numpy operations.

ii.
```python
animal = load_animal(animal_id, data_dir)  # joblib.load - I/O bound
```

iii. The conversion output shows all 7 animals processed quickly. No explicit timing instrumentation was added despite the instructions requesting it.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials sequentially, appending slices to lists. This could potentially be done with `np.split` or `np.reshape` for fixed-size trials, avoiding the Python loop overhead. However, since the loop body is simple numpy slicing, the performance impact is minimal.

ii.
```python
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The `discretize_position_to_3x3` function is called once per trial instead of once per session. Since the discretization uses session-wide min/max, the position could be discretized for the entire session first and then sliced into trials. This would avoid recomputing `np.isfinite` checks for each trial slice.

## 6-c. What processing does the code repeat multiple times?

i. The `discretize_position_to_3x3` function is called once per trial (39-40 times per session), but computes validity checks and min/max over per-trial slices rather than the full session. Since position discretization uses session-level ranges but is applied per-trial, the min/max computation is redundant — it should compute the session-wide min/max once and then bin per-trial positions using those bounds.

**Wait — actually looking more closely at the code**, `discretize_position_to_3x3` is called with `pos[:, s:e]` (per-trial slice), so it computes min/max over each trial's position data independently, not the full session. This means bin edges vary per trial, which is inconsistent with the metadata claim of "per-session" discretization.

ii.
```python
# Called per trial with trial-level position data:
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))

# Inside discretize_position_to_3x3:
xmin, xmax = np.min(xv), np.max(xv)  # Computed on per-trial data
```

iii. The metadata says "Position discretized independently within each session/day" but the code actually discretizes per-trial, not per-session. This is a bug — bin edges change from trial to trial within the same session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `mask.copy()` call creates a new copy of the 9-element environment mask for each trial, even though the mask is identical across all trials within a session. Since the downstream decoder treats inputs as read-only, the copies are unnecessary — a reference or single array could be shared.

Additionally, the `blocked_to_mask` function includes handling for `-1` values meaning "no blocked partitions" and ragged nested structures, which adds some overhead but is necessary for robustness.

ii.
```python
input_trials.append(mask.copy())  # Same mask for all trials in session
```

iii. No explicit justification was given for copying the mask per trial. It's a defensive coding practice to avoid shared references.
