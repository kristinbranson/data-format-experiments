# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads one **joblib archive per animal** from `/app/data/<ANIMAL_ID>` (the `format="joblib"` files used by the paper's demo notebooks, i.e. `load_dat(animal, p, format="joblib")`), rather than the HDF5 `.mat` files. The seven animal IDs are **hard-coded** in a module-level list. Each archive is a dict `{animal_id: {'SFPs', 'blocked', 'centroids', 'envs', 'maps', 'position', 'trace'}}`; the AI reads `trace` `(n_days, n_neurons, n_frames)`, `position` `(n_days, 2, n_frames)`, `envs` `(n_days, 1)` and `blocked` (list of per-day arrays). All 7 animals × all days are iterated; no subject/day is excluded. The data directory is fixed to the relative path `'data'`, so the script must be run from `/app`.

ii.
```python
ANIMAL_IDS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]

for animal_id in animal_ids:
    animal = load_animal(animal_id, data_dir)
    envs = np.array(animal['envs']).squeeze()
    traces = np.asarray(animal['trace'])
    positions = np.asarray(animal['position'])
    blocked = animal['blocked']
    n_days = traces.shape[0]
```

iii. From CONVERSION_NOTES.md Step 1: "Demo notebook usage shows per-animal loading with `dat = load_dat(animal, p, format="joblib")`, then direct access to `dat[animal]['position']`, `dat[animal]['trace']`, `dat[animal]['maps']['smoothed']`, and `dat[animal]['envs']`." The AI therefore followed the reference code's own loading entry point. It also noted (Step 3) that "`behav_dict` and data directory contain 7 animal IDs: QLAK-CA1-08, 30, 50, 51, 56, 74, 75", which is where the hard-coded ID list comes from. Verified by me: the joblib and `.mat` copies contain identical arrays (`np.allclose` on trace and position for animal 08, day 0).

## 1-b. How are the data split into subjects?

i. One subject per animal archive / animal ID. `subjects` is the list of the 7 animal IDs, and every session appends the index of its animal to `subject_idx`. The output contains 7 subjects with 31/31/31/21/31/31/31 sessions.

ii.
```python
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subject_to_idx[animal_id])
...
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. Each file holds all the recording days of a single mouse, so the file/animal ID is the natural subject identifier (CONVERSION_NOTES Step 3/4: "7 mice in provided dataset files").

## 1-c. How are the data split into sessions?

i. One session per **recording day**: the leading axis of `trace`/`position`/`envs`/`blocked` is the day axis, and each day becomes a separate entry in `neural`/`input`/`output`. This yields 207 sessions in total (matching the 207 day-entries in the raw files).

ii.
```python
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx], ...)
    ...
    data['neural'].append(neural_trials)
```

iii. CONVERSION_NOTES Step 4: "Paper says one 40 min session per day → Treat each day as a session in target format, then split each session into 1-minute trials as required by decoder task." The reference code's analyses (`decode_position_within`, `get_shr_within`) are also within-day, confirming day = session.

## 1-d. How are the data split into trials?

i. Each session is cut into contiguous, non-overlapping **1-minute trials of 1800 frames** (30 Hz × 60 s). Only complete trials are kept; the trailing partial minute is dropped. Result: 39–40 trials per session, 8187 trials total, all with exactly T = 1800.

ii.
```python
FRAME_RATE_HZ = 30.0
TRIAL_SECONDS = 60.0
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)   # 1800
...
n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The task instructions define the trial structure ("long recording sessions, which will be split into 1-minute trials within each session"); the recording is continuous with no behavioural trial structure. Metadata records `'notes': 'Trailing partial-minute frames are dropped.'`

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control at all. The only curation at this level is (a) dropping the incomplete trailing minute and (b) dropping any **session** that would yield fewer than 2 complete trials (required by the format spec "at least two trials within each session"). In practice no session was dropped — all 207 days survive.

ii.
```python
if len(neural_trials) < 2:
    continue
```

iii. Not explicitly justified in CONVERSION_NOTES (the "Trial curation rules" section is left as an unfilled placeholder). Implicitly, the reference paper applies no trial-level exclusion — sessions are continuous free foraging — and the `< 2` guard exists only to satisfy the decoder format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Exclusively the `trace` field, indexed per day: `traces[day_idx]` of shape `(n_neurons, n_frames)`. No use of `maps`, `SFPs`, or `centroids` for the neural stream.

ii.
```python
traces = np.asarray(animal['trace'])
...
def process_day(day_trace, day_pos, env_name, blocked_entry, ...):
    valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
    trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 1/4: "Neural data are **rise-extracted calcium traces** where `1` indicates a significant event; this suggests the decoder neural input should likely use these event traces rather than recomputing dF/F"; "Use provided `trace` directly as neural activity; do not recompute dF/F or deconvolution." Metadata: `'source_signal': 'Binary rising-phase calcium trace events'`.

## 2-b. How is the `neural` data processed?

i. Essentially unprocessed: select the recorded (non-all-NaN) neurons, replace any residual NaNs with 0, cast to `float32`, truncate to the common number of frames shared with `position`, and slice into 1800-frame trials. No dF/F, no deconvolution, no z-scoring, no smoothing, no rebinning. The stored values remain the binary (0/1) event trace. The joblib layout is already `(neurons, frames)`, so no transpose is needed.

ii.
```python
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
pos = np.asarray(day_pos, dtype=np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
```

iii. The methods state the binary rising-phase vector (z-scored derivative > 2.5) "was treated as the firing rate in all subsequent analyses", so the provided `trace` is already the analysis-ready signal (CONVERSION_NOTES Step 3/4). `float32` is for decoder/memory compatibility. (Verified by me: stored trials are bit-identical to the raw `trace` slice.)

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only one rule: neurons whose trace is **all-NaN on that day** are dropped, per session. These are cells that were registered across days but not recorded/active in that session. Nothing else is filtered — no place-cell selection, no spatial-reliability (SHR) threshold, no activity-rate threshold. `brain_region_idx` is then a zero vector of length = number of kept neurons (all CA1). Resulting counts: mean 336.9 neurons/session, min 113, max 564.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
data['brain_region_idx'].append(np.zeros(int(np.sum(valid_neurons)), dtype=np.int64))
```

iii. CONVERSION_NOTES Step 6: "Current implementation ... filters neurons that are all-NaN within a day." The rationale is that cross-registered cells are NaN-padded on days where they were not recorded. The AI explicitly did not apply the paper's place-cell / split-half-reliability criterion, which is an analysis-specific selection rather than a data-quality filter (the "Neuron curation rules" section of the notes was left as a placeholder, so this is implicit rather than argued).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — recording is continuous free foraging. Trials are aligned to **session (day) start**: trial *k* covers frames `[k*1800, (k+1)*1800)` from the beginning of the day's recording. Metadata declares `temporal_alignment_event = 'Session/day start; sessions split into contiguous 1-minute trials'`, `off_start = 0.0`, `off_end = 60.0`.

ii.
```python
'temporal_alignment_event': 'Session/day start; sessions split into contiguous 1-minute trials',
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. CONVERSION_NOTES Step 4 concluded that day/session is the natural unit and 1-minute trials are an artificial subdivision imposed by the decoder task, so offsets are simply 0–60 s from the start of each segment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging/behaviour frame rate of **30 Hz**, i.e. `time_bin_size = 1000/30 = 33.33 ms`. **No** temporal rebinning, downsampling, or smoothing is applied; each trial is (n_neurons, 1800). This produces a very large dataset (≈20 GB pickle).

ii.
```python
FRAME_RATE_HZ = 30.0
...
'time_bin_size': 1000.0 / FRAME_RATE_HZ,
'frame_rate_hz': FRAME_RATE_HZ,
```

iii. CONVERSION_NOTES Step 3/4: "trajectories from each recorded session recorded at 30 Hz"; "`position` and `trace` share same frame dimension within animal → Use a common framewise time base". Keeping the native bins preserves the paper's framewise analysis convention and guarantees neural/behaviour alignment without resampling.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. **Two** sources are combined: the per-day environment **name** string `envs[day]` (one of `square, o, t, u, rectangle, +, i, l, bit donut, glenn`) and the per-day **`blocked`** array of blocked 3×3 partition indices (`[-1]` when nothing is blocked). The env name is mapped to a 3×3 occupancy mask copied from the reference code, the `blocked` indices are mapped to a second 3×3 mask, and the two are multiplied element-wise. Result: a static 9-dim binary vector per trial (1 = open, 0 = blocked), `input_names = ['geom_bin_0' ... 'geom_bin_8']`.

ii.
```python
def env_to_mask(env_name):
    env = str(env_name)
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 't':
        return np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    ...

def blocked_to_mask(entry):
    cur = normalize_blocked_entry(entry)
    ...
    mask = np.ones((3, 3), dtype=np.float32)
    if len(vals) == 1 and vals[0] == -1:
        return mask
    for v in vals:
        if v == -1:
            continue
        r, c = divmod(int(v), 3)
        mask[r, c] = 0.0
    return mask

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)
```

iii. CONVERSION_NOTES Step 1: "`blocked` stores blocked partition locations in a 3x3 layout indexed as `[[0,1,2],[3,4,5],[6,7,8]]`; this is directly relevant for constructing the decoder input representing arena geometry/blockage", and an "environment geometry helper (env->3x3 mask)" was identified in the reference source. Step 4 resolution: "Build decoder input from 3x3 geometry/block mask per trial, static within each 1-minute trial." The `env_to_mask` table is transcribed from the reference code (`square … bit donut` match the reference source verbatim); the terminal dump the AI read was truncated at `elif env == 'glenn':`, so the `glenn` mask in the AI's script was not actually read from the reference.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked_to_mask` normalizes the nested MATLAB cell/array structure, treats `[-1]` as "nothing blocked", and clears entry `divmod(v, 3)` for each blocked index. `env_to_mask` returns the hard-coded shape mask. The two masks are multiplied, flattened row-major to a 9-vector and copied into every trial of that session (static per trial, float32 0/1). The polarity is inverted relative to the human reference (AI: 1 = accessible; reference: 1 = blocked), which is immaterial for a decoder.

What is **not** immaterial: the two masks use **different row conventions**. Checking every session against the raw data, `env_to_mask` agrees with the session's own `blocked` field for only 127 of 207 sessions; for `t`, `l` and `bit donut` the reference shape mask is the vertical (y-flipped) mirror of the `blocked` mask, and the AI's guessed `glenn` mask (zeros at 2,4,6) matches neither orientation of the data (`blocked = {0, 8}`). Multiplying the two therefore zeroes the **union** of two mutually inconsistent blocked sets in **80/207 sessions (39%)**. Example verified from the saved pickle: animal 08, day 3 (`env = 't'`, `blocked = {3,5,6,8}`, so bins {0,1,2,4,7} are open) is stored as `[0,1,0, 0,1,0, 0,1,0]` — bins 0 and 2 are wrongly marked blocked. The map from environment to mask is still injective (all 10 environments get distinct vectors), so the input still identifies the context for the decoder, but it misdescribes which parts of the arena were blocked.

ii.
```python
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
...
for ti in range(n_full_trials):
    ...
    input_trials.append(mask.copy())
```

iii. CONVERSION_NOTES Step 4: the decoder input should represent "arena geometry/blockage", combining the code's env→mask helper with the documented `blocked` 3×3 indexing. Metadata: "static per-trial environment geometry/block mask". No consistency check between the two sources was ever performed — Step 10 ("Check 2: sanity checks against raw data", "Check 3: reference code comparison") was left `NOT STARTED` in the notes, and the trajectory confirms the agent stopped before Steps 10/12/13.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The per-day `position` array, `positions[day_idx]` of shape `(2, n_frames)` = (x, y) in cm from DeepLabCut head tracking. Single output variable `output_names = ['position_bin']` with `output_values = [['bin_0' ... 'bin_8']]`.

ii.
```python
positions = np.asarray(animal['position'])
...
pos = np.asarray(day_pos, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 3/4: "Position data were obtained from **DeepLabCut** head tracking"; "Raw `position` is continuous x-y trajectory per frame … For target output, discretize position into 3x3 categorical bins over the arena, preserving time variation."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. x and y are **min–max normalized to the observed range and then split into equal thirds**, rather than binned on the fixed 75 cm arena coordinates. Finite (non-NaN) samples are identified, the range is rescaled to [0,1), multiplied by 3, floored and clipped to {0,1,2}; the label is `y_bin*3 + x_bin`, stored as `(1, 1800)` int64. Non-finite samples get label −1.

Crucially, `discretize_position_to_3x3` is called **on each 1800-frame trial slice**, so the min/max — and hence the bin edges — are recomputed **per trial**, not per session as the metadata claims. I confirmed this against the saved pickle: the stored labels equal the per-trial min–max binning exactly, and differ from both per-day min–max binning and fixed 0–75 cm binning.

ii.
```python
def discretize_position_to_3x3(pos_xy):
    x = pos_xy[0]; y = pos_xy[1]
    valid = np.isfinite(x) & np.isfinite(y)
    ...
    xmin, xmax = np.min(xv), np.max(xv)
    ymin, ymax = np.min(yv), np.max(yv)
    ...
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
    ...
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]

# called per trial:
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. Metadata `notes`: "Position discretized independently within each session/day into 3x3 bins using observed x/y range." No further rationale is given anywhere in CONVERSION_NOTES or the trajectory; the agent never discussed why observed-range normalization was preferred over the arena's physical 0–75 cm extent, and never noticed that the implementation is per-trial rather than per-day.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Equal-width tertiles of the (per-trial) observed x and y ranges → 3×3 = 9 classes, labelled `y_bin*3 + x_bin` (row-major, same ordering convention as the `blocked` indices). Boundary values are clipped into {0,1,2}; NaN frames would be labelled −1 (never occurs: the raw `position` arrays contain no NaNs, and the verification log reports an output range of [0, 8]).

Because the thresholds are recomputed for every 1-minute trial, a given class label denotes a different physical region in different trials. Measured against fixed 0–75 cm arena bins, **13–14 % of all frames receive a different label** (per-session means up to 0.44 for animal 08 and 0.61 for animal 51). The realized class distribution is uneven (bin_4 = 0.050 to bin_8 = 0.192). In blocked geometries the distortion is largest: e.g. in `rectangle` sessions the animal only occupies x ∈ [25, 75] cm, which per-trial rescaling stretches back over all three x bins, so the position labels no longer line up with the geometry mask supplied as decoder input.

ii.
```python
xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
labels = y_out * 3 + x_out
labels[~valid] = -1
```

iii. Per the Decoder Task spec, "Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying." The AI's only stated rationale is the metadata note about using the "observed x/y range"; the per-trial application is undocumented and appears to be an oversight (the function is simply invoked inside the trial loop).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame: `trace` and `position` share the same frame axis within a day. The AI defensively truncates both streams to `n_frames = min(trace frames, position frames)` before splitting, and then applies the identical `[s:e]` slicing to both. No shift or lag is introduced; the output label at bin *t* is the position at the same imaging frame as the neural vector at bin *t*.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
...
neural_trials.append(trace[:, s:e])
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. CONVERSION_NOTES Step 4: "`position` and `trace` share same frame dimension within animal … Use a common framewise time base". The `min()` truncation is an explicit guard against any day where the two streams might differ in length (in the provided data they never do).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures: (a) neurons that are all-NaN within a day are dropped; (b) any residual NaNs in the kept traces are replaced with 0 via `np.nan_to_num` (a no-op here — NaNs in `trace` are always whole-neuron); (c) non-finite position samples are labelled −1 (never triggered — `position` contains no NaNs); (d) neural/position length mismatches are absorbed by truncating to the shorter stream; (e) the ragged MATLAB `blocked` cell structure (nested lists / 0-d arrays / arrays, `[-1]` sentinel for "none blocked") is normalized recursively; (f) the trailing partial minute of each session is discarded; (g) sessions producing fewer than 2 trials are skipped.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
def normalize_blocked_entry(entry):
    cur = entry
    while isinstance(cur, list) and len(cur) == 1:
        cur = cur[0]
    if isinstance(cur, np.ndarray):
        cur = cur.tolist()
    if isinstance(cur, tuple):
        cur = list(cur)
    return cur
...
if len(vals) == 1 and vals[0] == -1:
    return mask
...
labels[~valid] = -1
...
if len(neural_trials) < 2:
    continue
```

iii. CONVERSION_NOTES Step 6: "Current implementation drops trailing partial-minute frames and filters neurons that are all-NaN within a day." The trajectory (step 212) notes that `blocked` entries were inspected first specifically "to implement the mask conversion robustly". The −1 sentinel for invalid positions is not documented and would silently inject an out-of-range class label if it ever fired; the AI never checked for it (the verification log's output range [0, 8] happens to confirm it did not).

## 6-a. What are the most time-consuming steps of the code?

i. The dominant costs are I/O, not computation: (1) `joblib.load` of each animal archive, which deserializes the **entire** animal dict — including `SFPs` (e.g. 35×35×515×31 float64 ≈ 4.9 GB for animal 08), `maps` and `centroids` — even though only `trace`, `position`, `envs` and `blocked` are used; (2) writing the ~20 GB output pickle. Per-frame work (NaN masking, `nan_to_num`, digitizing 2 × 72 000 positions) is negligible. The script prints **no per-step timing** — only a per-animal "processed X: N days" line and a single total runtime stored in metadata (122.9 s for the 2-animal sample run; the full run ≈7 min plus pickle write). No profiling or optimization was done; Step 6 of CONVERSION_NOTES leaves "Code inefficiencies identified" and "Code speedups added" as empty placeholders.

ii.
```python
def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]   # loads SFPs/maps/centroids too
...
t0 = time.time()
...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. Not discussed in CONVERSION_NOTES beyond the empty placeholders; the trajectory shows the agent only remarked on the size of the outputs ("a very large `converted_data.pkl` (19G)") when deciding whether GPU training would fit, never as a conversion bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop in `process_day` is the only loop over data. `discretize_position_to_3x3` is invoked once per trial (8187 times); the whole day's position could have been discretized once (a single vectorized call) and then reshaped to `(n_trials, 1800)` — which would have been both faster and would have avoided the per-trial bin-edge bug. `mask.copy()` is likewise executed once per trial instead of broadcasting one shared array. The `blocked_to_mask` loop over blocked indices is over ≤4 elements. Overall the vectorization gains would be small compared with the I/O cost.

ii.
```python
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. Not discussed; no vectorization or parallelization was attempted despite the instructions asking for it (the notes' speedup fields are blank).

## 6-c. What processing does the code repeat multiple times?

i. (a) `discretize_position_to_3x3` recomputes the `isfinite` mask and the x/y min–max on each trial slice — 39–40 redundant passes per session, and the cause of the trial-dependent bin edges. (b) `combined_env_mask` is rebuilt per day (cheap) and the resulting 9-vector is copied per trial. (c) `np.nan_to_num` scans the full (n_neurons × 72 000) trace of every session although the NaNs have already been eliminated by the all-NaN neuron mask. (d) At the workflow level the whole conversion was run twice (sample then full), with the sample run itself processing 62 sessions.

ii.
```python
xmin, xmax = np.min(xv), np.max(xv)     # recomputed for every 1800-frame trial
ymin, ymax = np.min(yv), np.max(yv)
...
input_trials.append(mask.copy())        # one copy per trial
...
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. Not discussed in CONVERSION_NOTES.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (a) `joblib.load` materializes `SFPs`, `maps` and `centroids` for every animal even though none is used — several GB of pointless deserialization per animal (the reference's h5py approach reads only the three datasets it needs). (b) `np.nan_to_num` over the full trace array is a no-op on this dataset. (c) `mask.copy()` allocates 8187 nine-element arrays that all hold the same content. (d) `--sample` was implemented as "first **2 animals**" (62 sessions, 2418 trials, a 5 GB `sample_data.pkl`) instead of the 2 sessions the instructions asked for, so the "quick" validation run cost ~2 minutes of conversion plus a full sample decoder training. (e) The `env_to_mask` factor of the input is not merely unnecessary (the env name is in 1:1 correspondence with `blocked` across all 207 sessions, so it carries no extra information) but actively corrupts the geometry input for 39 % of sessions. (f) Neural data are kept at full 30 Hz resolution with no compression, producing a 20 GB pickle — not wrong, but it makes every downstream load expensive.

ii.
```python
def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]
...
animal_ids = ANIMAL_IDS if mode == 'full' else ANIMAL_IDS[:2]
...
return env_to_mask(env_name) * blocked_to_mask(blocked_entry)
```

iii. Not discussed. CONVERSION_NOTES Steps 10, 12 and 13 were never completed (Step 10 is still `NOT STARTED`, Step 12 `IN PROGRESS`, no `README.md` was produced), so none of the required raw-data sanity checks, reference-code comparisons or efficiency reviews that would have surfaced these issues were carried out; the trajectory ends with the agent declaring the remaining steps "documentation/review refinements".
