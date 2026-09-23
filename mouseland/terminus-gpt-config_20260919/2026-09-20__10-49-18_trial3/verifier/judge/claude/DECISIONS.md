# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the `spk/` directory for all `*_neural_data.npy` files to discover 89 sessions. Behavior is loaded from all `Beh_*.npy` files in `beh/`, keyed by session ID. Retinotopy is loaded per session from `retinotopy/`. The AI also loads `Imaging_Exp_info.npy` for metadata but uses the neural file listing (not the experiment index) as the canonical session list.

ii.
```python
def neural_files():
    return {
        os.path.basename(p).removesuffix("_neural_data.npy"): p
        for p in glob.glob(os.path.join(ROOT, "spk", "*_neural_data.npy"))
    }

def load_behavior_views(valid_ids):
    """Load all analysis views and group them by unique physical recording."""
    views = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(ROOT, "beh", "Beh_*.npy"))):
        group = os.path.basename(path)[4:-4]
        obj = np.load(path, allow_pickle=True).item()
        for key, beh in obj.items():
            sid = physical_id(key)
            if sid in valid_ids:
                views[sid].append((group, key, beh))
```

iii. The AI uses the neural file system as ground truth for which sessions exist, rather than the experiment index. This is a bottom-up approach. The AI also loads all behavior views for each session to handle duplicates across experiment types.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from the session ID by splitting on underscores (taking the first component). Unique sorted mouse names form the subjects list.

ii.
```python
subjects = sorted({sid.split("_")[0] for sid in selected})
subject_map = {v: i for i, v in enumerate(subjects)}
```

iii. The session ID format is `<mouse>_<YYYY>_<MM>_<DD>_<block>`, so the first underscore-delimited token is always the mouse name.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique neural data files in the `spk/` directory. There are 89 unique session IDs. The AI handles duplicate behavior views by merging them via `choose_behavior()` and `merged_stimuli()`, verifying that physical streams are identical across views.

ii.
```python
paths_all = neural_files()
if len(paths_all) != 89:
    raise RuntimeError(f"Expected 89 unique neural sessions, found {len(paths_all)}")
```

iii. The AI explicitly asserts 89 sessions to match the paper. Duplicate behavior views (from different experiment types) are merged rather than treated as separate sessions.

## 1-d. How are the data split into trials?

i. Trials are defined by the `[ceil(StartFr), ceil(GrayFr))` window for each trial. The AI uses `np.ceil` on the fractional start and gray frame values, creating a half-open integer interval. This captures the visual corridor portion of each trial.

ii.
```python
def trial_bounds(beh, nfr):
    starts = np.ceil(np.asarray(beh["StartFr"], dtype=np.float64)).astype(np.int64)
    ends = np.ceil(np.asarray(beh["GrayFr"], dtype=np.float64)).astype(np.int64)
    starts = np.clip(starts, 0, nfr)
    ends = np.clip(ends, 0, nfr)
    if np.any(ends <= starts):
        bad = np.flatnonzero(ends <= starts)
        raise ValueError(f"Invalid trial windows: {bad[:20].tolist()}")
    return starts, ends
```

iii. The AI uses `ceil(StartFr)` and `ceil(GrayFr)` rather than the `ft_trInd` and `ft_CorrSpc` frame masks used by the reference. The AI's justification is that StartFr marks corridor entry and GrayFr marks the transition to gray space, defining the visual corridor window.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT filter trials based on length or any other quality criterion. All 38,110 trials are retained, including extremely long stalled trials (max 5,607 frames / ~29 minutes). The only check is that `ends > starts` for valid windows.

ii.
```python
# No trial filtering code exists in the AI's implementation.
# All trials with valid [ceil(StartFr), ceil(GrayFr)) windows are included.
for i, (a, z) in enumerate(zip(starts, ends)):
    # ... processes all trials
```

iii. The AI explicitly decided: "Retain all 38,110 valid windows; do not invent a duration filter." (CONVERSION_NOTES Step 4). The AI noted that no exclusion rule is stated in the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the `*_neural_data.npy` files, which contains a list of three arrays (one per imaging plane), concatenated along the neuron axis.

ii.
```python
obj = np.load(path, allow_pickle=True).item()
parts = obj["spks"]
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```

iii. Same source as the reference: deconvolved calcium traces from Suite2p.

## 2-b. How is the `neural` data processed?

i. No normalization or additional processing is applied. The AI slices the trial window from each plane and concatenates, storing as float32. The AI avoids materializing the full session-length concatenated array by slicing each plane individually before concatenation.

ii.
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```

iii. The AI notes: "No additional normalization" and "reference load_spk concatenation of all three released Suite2p deconvolved trace arrays."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons. All neurons are kept, including those outside the four visual areas (V1, mHV, lHV, aHV). Neurons outside these areas are assigned to a 5th "unassigned" brain region. This results in 4,691,034 total neurons across sessions (vs. 4,105,393 in the reference which drops non-visual-area neurons).

ii.
```python
REGIONS = ["V1", "mHV", "lHV", "aHV", "unassigned"]

def map_regions(iarea):
    a = np.asarray(iarea)
    out = np.full(a.shape, 4, dtype=np.int16)  # default to "unassigned"
    out[a == 8] = 0
    out[np.isin(a, [0, 1, 2, 9])] = 1
    out[np.isin(a, [5, 6])] = 2
    out[np.isin(a, [3, 4])] = 3
    return out
```

iii. The AI states: "Use exact reference mapping and an explicit `unassigned` class rather than dropping ROIs." (CONVERSION_NOTES Step 4). The AI chose to keep all neurons rather than filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to corridor entry (StartFr). Each trial starts at `ceil(StartFr)` and ends at `ceil(GrayFr)`, covering only the visual corridor portion. Trials are variable length.

ii.
```python
starts = np.ceil(np.asarray(beh["StartFr"], dtype=np.float64)).astype(np.int64)
ends = np.ceil(np.asarray(beh["GrayFr"], dtype=np.float64)).astype(np.int64)
# ...
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```

iii. The AI uses the fractional StartFr rounded up via ceil, which differs from the reference's approach of using `ft_trInd` and `ft_CorrSpc` masks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning. The native imaging frame rate is used. The AI computes the per-session frame interval from the median `ft` differences and reports it in metadata. The median across sessions is ~314.7 ms.

ii.
```python
def session_dt_seconds(beh):
    d = np.diff(np.asarray(beh["ft"], dtype=np.float64))
    d = d[np.isfinite(d) & (d > 0)]
    dt = float(np.median(d) * 86400.0)
    return dt
# ...
data["metadata"] = {
    "time_bin_size": float(np.median(dt_values)),
    # ...
}
```

iii. The AI computes per-session dt rather than using a fixed 3.17 Hz constant, which is more precise but functionally equivalent.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and the frame index within the trial window.

ii.
```python
sound = np.asarray(beh["SoundFr"], dtype=np.float64)
# ...
inp[0] = (sound[i] - frames) * dt
```

iii. Uses the per-trial SoundFr value and computes the time difference to each frame.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `(SoundFr[trial] - frame_index) * dt_seconds` for each frame in the trial window. This converts the frame difference to seconds using the session-specific frame interval. The result is positive before the cue and negative after.

ii.
```python
frames = np.arange(a, z, dtype=np.float64)
inp[0] = (sound[i] - frames) * dt
```

iii. The AI uses frame indices multiplied by a constant dt rather than interpolated frame times. This is an approximation that assumes uniform frame spacing.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both use the same frame range `[a, z)` for the trial, so the input is naturally aligned with the neural data.

ii.
```python
frames = np.arange(a, z, dtype=np.float64)
T = z - a
nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
inp[0] = (sound[i] - frames) * dt
```

iii. Same frame indices are used for neural and input data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date component of each session ID, parsed as a calendar date.

ii.
```python
def subject_day_values(session_ids):
    first = {}
    dates = {}
    for sid in session_ids:
        mouse, date, _ = parse_session_id(sid)
        d = datetime.strptime(date, "%Y_%m_%d").date()
        dates[sid] = d
        first[mouse] = min(first.get(mouse, d), d)
    return {sid: float((dates[sid] - first[sid.split("_")[0]]).days) for sid in session_ids}
```

iii. The AI uses calendar days since each subject's earliest recording. This is computed across all 89 sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is found. The day of training for each session is then the number of calendar days since that earliest date. This gives values ranging from 0 to 92. The value is broadcast as a constant across all frames in a trial.

ii.
```python
return {sid: float((dates[sid] - first[sid.split("_")[0]]).days) for sid in session_ids}
# ...
inp[1] = day_value  # broadcast to all frames
```

iii. The AI justifies this choice: "Calendar days since each mouse's earliest recording is reproducible, continuous, and complete. `sess#` cannot be used because it conflicts for the same physical recording and is missing for eight sessions." This differs from the reference which counts ordinal session number (0-7 range).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (the corridor entry frame) and the frame indices within the trial window.

ii.
```python
inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. Uses StartFr as the trial start event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(frame_index - StartFr) * dt_seconds`. Starts near zero (since the first frame is `ceil(StartFr)`) and increases with each frame. Uses the session-specific dt.

ii.
```python
frames = np.arange(a, z, dtype=np.float64)
inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. Same approach as time to sound cue: frame differences multiplied by constant dt, rather than interpolated frame times.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same frame range as neural data.

ii.
```python
frames = np.arange(a, z, dtype=np.float64)
```

iii. Same frame indices for all data streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks rewarded vs unrewarded corridors.

ii.
```python
reward = np.asarray(beh["isRew"], dtype=np.int16)
# ...
inp[3] = reward[i]
```

iii. Direct use of the isRew flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing. The binary flag is broadcast across all frames in the trial.

ii.
```python
inp[3] = reward[i]
```

iii. Straightforward binary indicator.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim` merged across behavior views, with fallback to `WallName` when `TrialStim` contains the placeholder `"stimulus_of_trial"`.

ii.
```python
def merged_stimuli(session_views):
    n = int(session_views[0][2]["ntrials"])
    labels = []
    for i in range(n):
        concrete = {
            str(b["TrialStim"][i]) for _, _, b in session_views
            if str(b["TrialStim"][i]) != "stimulus_of_trial"
        }
        if concrete:
            label = next(iter(concrete))
        else:
            wall = {str(b["WallName"][i]) for _, _, b in session_views}
            label = next(iter(wall))
        labels.append(label)
    return np.asarray(labels)
```

iii. The AI merges stimulus labels across duplicate behavior views.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses 8 raw texture names as categories: circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2. These are encoded as integer indices 0-7. The AI does NOT map textures to the 4 broad categories (circle, leaf, rock, wood) used by the reference. Furthermore, the AI's `STIM_VALUES` list completely omits rock and wood textures.

ii.
```python
STIM_VALUES = [
    "circle1", "circle2", "circle3", "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
STIM_TO_ID = {v: i for i, v in enumerate(STIM_VALUES)}
# ...
out[0] = STIM_TO_ID[str(stimuli[i])]
```

iii. The AI's CONVERSION_NOTES describe "Eight integer categories" for visual stimulus. The instructions say "Visual stimulus category. e.g. circle, leaf, etc., per-trial" which suggests the broad categories. The reference uses 4 categories. The AI uses 8 fine-grained names and misses rock/wood entirely.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the fractional frame number of each lick event in the session.

ii.
```python
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
lick_session = np.zeros(nfr, dtype=np.int16)
lick_idx = np.floor(lick_frames).astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
if len(lick_idx):
    lick_session[np.unique(lick_idx)] = 1
```

iii. Direct use of LickFr, converted to binary raster.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frame numbers are floored to integer frame indices. A binary flag (0/1) is set for each frame that contains at least one lick. Licks outside the neural frame range are excluded.

ii.
```python
lick_idx = np.floor(lick_frames).astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
if len(lick_idx):
    lick_session[np.unique(lick_idx)] = 1
```

iii. The AI uses `floor` for fractional lick frames while the reference uses `.astype(int)` (truncation toward zero). For positive values these are equivalent.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick raster is computed for the full session, then sliced to the trial window `[a:z)`.

ii.
```python
out[1] = lick_session[a:z]
```

iii. Same frame-based alignment as all other streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the corridor position at each imaging frame.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. Direct use of the frame-level position variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized into 4 bins using `np.digitize` with edges at [10, 20, 30] source units (corresponding to 1m boundaries), then clipped to [0, 3].

ii.
```python
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. The AI uses `np.digitize` while the reference uses `ft_Pos // 10` (floor division). `np.digitize` with bins [10, 20, 30] assigns: values < 10 to bin 0, [10, 20) to bin 1, [20, 30) to bin 2, >= 30 to bin 3. This is equivalent to floor division by 10 for values in [0, 40).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four 1-meter bins: 0-1m (source 0-10), 1-2m (source 10-20), 2-3m (source 20-30), 3-4m (source 30-40). Bins are created via `np.digitize` with thresholds at 10, 20, 30 source units.

ii.
```python
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. Four equal 1m bins as specified in the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is a frame-level variable, sliced to the trial window `[a:z)`.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. Same frame-based alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
```

iii. Direct use of frame-level running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes GLOBAL quartile thresholds across all retained corridor frames from all selected sessions. `np.quantile` at [0.25, 0.5, 0.75] gives three thresholds. Speeds are then digitized using these thresholds.

ii.
```python
def compute_speed_thresholds(selected_ids, views, nfr_by_session):
    chunks = []
    for sid in selected_ids:
        b = choose_behavior(views[sid])
        nfr = min(int(nfr_by_session[sid]), len(b["ft_RunSpeed"]))
        starts, ends = trial_bounds(b, nfr)
        speed = np.asarray(b["ft_RunSpeed"][:nfr], dtype=np.float64)
        chunks.extend(speed[a:z] for a, z in zip(starts, ends))
    values = np.concatenate(chunks)
    q = np.quantile(values, [0.25, 0.5, 0.75]).astype(np.float64)
    return q, int(values.size), (float(values.min()), float(values.max()))
```

iii. The AI uses global quantile thresholds (q=[0, 8.38, 30.19]) rather than per-session rank-based quartiles. This means each session gets very different bin distributions. Some sessions have 0.6-75% of data in one bin.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins using `np.digitize` with the global quartile thresholds. Values below 0 are in bin 0, etc.

ii.
```python
out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. Global thresholds cause highly uneven per-session distributions: e.g., some slow sessions have 80% of frames in "25-50%" bin and only 1-2% in "0-25%" bin. The overall distribution shows 0-25%: 9.8%, 25-50%: 40.2%, 50-75%: 25.0%, 75-100%: 25.0%, which is far from the intended 25% each.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is frame-level, sliced to trial window `[a:z)`.

ii.
```python
out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. Same frame-based alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavior arrays to neural frame count (`nfr`), handles lick frames outside range by clipping, and clips trial bounds to `[0, nfr]`. The AI also handles duplicate behavior views by verifying physical stream equivalence and merging stimulus labels.

ii.
```python
nfr = min(int(nfr_by_session[sid]), len(b["ft_RunSpeed"]))
starts = np.clip(starts, 0, nfr)
ends = np.clip(ends, 0, nfr)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
```

iii. The AI handles the behavior-neural frame mismatch and validates consistency across duplicate views.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the neural data files (434 GB total). The AI reads each file once and slices trial windows during that read.

ii.
```python
obj = np.load(path, allow_pickle=True).item()
parts = obj["spks"]
nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
```

iii. I/O dominates. The AI avoids materializing the full concatenated array by slicing per-plane.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `convert_session` creates arrays one trial at a time. The concatenation of plane slices `[x[:, a:z] for x in parts]` is done per trial.

ii.
```python
for i, (a, z) in enumerate(zip(starts, ends)):
    nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```

iii. The per-trial loop is inherent to the variable-length trial structure.

## 12-c. What processing does the code repeat multiple times?

i. The AI reads behavior files twice: once in `load_behavior_views` (loading all behavior), and then `choose_behavior` / `merged_stimuli` are called again during `compute_speed_thresholds` and during `convert_session`. The `trial_bounds` function is called multiple times per session (once for speed thresholds, once for conversion). In sample mode, the AI also does `inspect_neural_shapes` which reads neural files that are then read again during conversion.

ii.
```python
# First read in load_behavior_views
all_views = load_behavior_views(paths_all.keys())
# Then used again in compute_speed_thresholds
speed_q, n_speed, speed_range = compute_speed_thresholds(selected, all_views, nfr_for_speed)
# Then again in convert_session
n, i, o, r, st = convert_session(sid, paths[sid], all_views[sid], ...)
```

iii. Behavior data is kept in memory across uses so it's not re-read from disk, but the speed threshold computation iterates over sessions separately from conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive per-session metadata (`stimulus_counts`, `rewarded_trials`, `source_neural_dtypes`, `experiment_views`), per-session dt values, and other diagnostic information that is not used by the decoder. The `inspect_neural_shapes` function (sample mode) reads all neural files just to check shapes before re-reading them for conversion.

ii.
```python
stats = {
    "session_id": sid, "n_neurons": nneu, "n_neural_frames": nfr,
    "n_trials": len(neural_trials), "retained_trial_frames": retained,
    "frame_bin_ms": dt * 1000.0,
    "stimulus_counts": dict(Counter(map(str, stimuli))),
    "rewarded_trials": int(reward.sum()),
}
```

iii. This metadata is useful for diagnostics but not consumed by the decoder.
