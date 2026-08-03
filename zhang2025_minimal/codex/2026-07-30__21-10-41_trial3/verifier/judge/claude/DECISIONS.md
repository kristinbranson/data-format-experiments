# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV file (`bwm_release.csv`) that lists 699 probe insertions across 459 sessions to identify which sessions and probes to process. It then uses the ONE API to load individual datasets (trials, spikes, clusters, wheel, camera) for each session. It does not use `one.search()` or `one.load_cache(tag=...)` to discover sessions from the release index.

ii.
```python
release_df = pd.read_csv(args.release_csv)
# ...
grouped = list(release_df.groupby("eid", sort=False))
# ...
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The AI chose to use the BWM release CSV as the authoritative source of sessions/probes, noting it matches the public release statistics (459 sessions, 699 insertions, 139 subjects).

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of the release CSV. During assembly, subjects are collected in insertion order and indexed.

ii.
```python
release_df["subject"] = release_df["subject"].astype(str)
# ...
if sess["subject"] not in subject_to_idx:
    subject_to_idx[sess["subject"]] = len(subjects)
    subjects.append(sess["subject"])
```

iii. The release CSV already contains the subject for each session/probe, so no derivation is needed.

## 1-c. How are the data split into sessions?

i. Sessions are identified by grouping the release CSV by the `eid` column. Each unique `eid` becomes one session.

ii.
```python
grouped = list(release_df.groupby("eid", sort=False))
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The CSV already lists one row per probe insertion, and sessions are identified by their `eid`.

## 1-d. How are the data split into trials?

i. Trials come from the trials table loaded via `one.load_object(eid, "trials", collection="alf")`. Each row is one trial.

ii.
```python
def load_trials(one: ONE, eid: str) -> pd.DataFrame:
    trials = one.load_object(eid, "trials", collection="alf").to_df()
```

iii. The trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Several filters are applied:
- Required non-NaN values for `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
- Reaction time (firstMovement_times - stimOn_times) must be in [0.08, 2.0] s
- No-choice trials (choice == 0) are excluded
- Trial length check: feedback_times - goCue_times <= 10.0 s
- Wheel and whisker signals must be available and interpolatable for the trial window
- Additionally, trials with all-zero neural activity after binning are dropped

ii.
```python
TRIAL_NAN_EXCLUDE = (
    "stimOn_times", "choice", "feedback_times",
    "probabilityLeft", "firstMovement_times", "feedbackType",
)

def make_base_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(trials), dtype=bool)
    for col in TRIAL_NAN_EXCLUDE:
        mask &= trials[col].notna().to_numpy()
    reaction_time = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    mask &= reaction_time >= 0.08
    mask &= reaction_time <= 2.0
    mask &= trials["choice"].to_numpy() != 0
    if "goCue_times" in trials.columns:
        go_cue = trials["goCue_times"].to_numpy()
        feedback = trials["feedback_times"].to_numpy()
        trial_len_ok = np.ones(len(trials), dtype=bool)
        good_len = np.isfinite(go_cue) & np.isfinite(feedback)
        trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
        mask &= trial_len_ok
    return mask
```

```python
# All-zero neural trial removal
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. The AI's CONVERSION_NOTES.md states these filters match the reference code and paper exclusions. The trial length check and feedbackType NaN check are additional filters not in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` and `spikes.clusters.npy`. The cluster quality comes from `clusters.metrics.pqt`, and region information comes from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. Standard spike sorting outputs from IBL.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window. Counts are stored as uint16 and later cast to float16. Critically, the AI does NOT divide by the bin size to convert to firing rates -- it stores raw spike counts, not Hz.

ii.
```python
probe_counts = np.zeros((len(trial_starts), len(probe.good_cluster_ids), NBINS), dtype=np.uint16)
# ... binning loop ...
np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)

# Later:
neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
```

iii. The AI documents this as "Spike counts are binned into non-overlapping 20 ms bins" in CONVERSION_NOTES.md, not mentioning conversion to firing rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Multiple filters are applied:
- `clusters.metrics.label >= 1.0` (good unit threshold)
- Units must be in grey matter (not "void", "root", or "grey" region)
- After Beryl mapping, a region must have at least 5 neurons within a session
- A region must appear in at least 2 sessions globally
- These are significantly more stringent than the reference, which only applies label >= 1.0

ii.
```python
good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))

# Later, region-level filtering:
MIN_NEURONS_PER_SESSION_REGION = 5
MIN_SESSIONS_PER_REGION = 2
```

iii. The AI notes these filters match the data-paper region criteria and the reference code's Beryl remapping. The grey-matter and min-neuron filters are additional beyond what the reference applies.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset. The trial start time for binning is computed as `stimOn_times + WINDOW_START`. Spikes within the 2s window are binned relative to this start time.

ii.
```python
trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
# ...
for trial_idx, start_time in enumerate(trial_starts):
    end_time = start_time + (WINDOW_END - WINDOW_START)
    # ...
    bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The AI notes stimulus onset as the common alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial spanning [-0.5, 1.5] s. No rebinning is applied.

ii.
```python
WINDOW_START = -0.5
WINDOW_END = 1.5
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))  # 100
```

iii. Matches the methods paper: "2 s trials with 20 ms bins producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table. The time variable is computed as bin right edges relative to stimulus onset.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. The AI describes this as the time axis for the decoding window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes bin RIGHT EDGES rather than bin centres. The values are `[-0.48, -0.46, ..., 1.48, 1.50]` instead of the reference's bin centres `[-0.49, -0.47, ..., 1.47, 1.49]`.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
# This produces: -0.48, -0.46, ..., 1.48, 1.50
```

iii. The CONVERSION_NOTES describe this as "linear interpolation to stimulus-aligned 20 ms bin right edges."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses bin right edges while the neural data is binned from the window start. The time values are shifted by half a bin from the bin centres that would correspond to the centre of each neural bin.

ii.
```python
# Neural binning uses start_time = stimOn + WINDOW_START
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)

# Time input uses right edges
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. The AI intended this to match the reference utility logic.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From the `probabilityLeft` column of the trials table. Block boundaries are detected by changes in this value.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return trial_num
    current = 1
    trial_num[0] = current
    for idx in range(1, len(prob_left)):
        prev = prob_left[idx - 1]
        curr = prob_left[idx]
        if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
            current += 1
        else:
            current = 1
        trial_num[idx] = current
    return trial_num
```

iii. The trials table has no block ID, so block boundaries must be inferred from changes in probabilityLeft.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number starts at 1 (not 0) and increments for each consecutive trial with the same `probabilityLeft`. The count is computed BEFORE trial filtering, so dropped trials still advance the count. The AI starts counting from 1 while the reference starts from 0.

ii.
```python
current = 1
trial_num[0] = current
for idx in range(1, len(prob_left)):
    if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
        current += 1
    else:
        current = 1
```

iii. The AI computed trial number before filtering, preserving the animal's real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, where +1 = left, -1 = right, 0 = no response.

ii.
```python
choice_val = float(trials.iloc[trial_idx]["choice"])
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
```

iii. Standard IBL convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple recoding: +1 (left) -> 0, -1 (right) -> 1. No-response trials (choice == 0) are dropped.

ii.
```python
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue
```

iii. Matches the instructions: left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, taking values 0.2, 0.5, 0.8.

ii.
```python
if prob_left == 0.2:
    prior_out = 0
elif prob_left == 0.5:
    prior_out = 1
elif prob_left == 0.8:
    prior_out = 2
else:
    continue
```

iii. Matches the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Simple recoding from the three prior values to categorical integers 0, 1, 2.

ii. Same as 6-a.

iii. No additional processing.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel position and timestamps (`_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`), loaded directly and processed through the same `interpolate_position` and `velocity_filtered` functions from brainbox.

ii.
```python
def load_wheel_speed(one: ONE, eid: str) -> tuple[np.ndarray, np.ndarray]:
    wheel = one.load_object(eid, "wheel", collection="alf")
    timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
    position = np.asarray(wheel["position"], dtype=np.float64)
    interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
    speed = np.abs(np.asarray(velocity, dtype=np.float32))
    return np.asarray(interp_time, dtype=np.float64), speed
```

iii. The AI imported `interpolate_position` and `velocity_filtered` directly from the brainbox wheel module.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) Interpolate position to 1000 Hz, (2) compute velocity with Butterworth-filtered differentiation (20 Hz corner, order 8), (3) take absolute value for speed. The speed is then interpolated onto trial bins and discretized into 3 categories.

ii.
```python
interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

iii. The Butterworth parameters (order=8 vs reference's default order=3) differ. The reference uses `SessionLoader.load_wheel()` which uses order=3 by default.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using GLOBAL tertiles across the entire converted dataset, not per-session percentiles. The bin edges are computed from quantiles at 1/3 and 2/3 of all wheel speed values pooled across all sessions.

ii.
```python
wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
wheel_edges = tuple(np.quantile(wheel_all, [1/3, 2/3]).astype(np.float32).tolist())

def discretize(values, edges):
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge]), right=False).astype(np.int8)
```

iii. The AI used global tertiles rather than per-session percentiles as the reference does.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated onto bin right edges within each trial window, using `np.interp`.

ii.
```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. The AI samples at bin right edges rather than bin centres. The reference samples at bin centres.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy from the left camera only: `leftCamera.ROIMotionEnergy` and `_ibl_leftCamera.times`. The AI does not fall back to the right camera if the left is unavailable -- sessions without left camera data are skipped.

ii.
```python
def load_whisker_motion_energy(one, eid):
    try:
        cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
        # ...
        return "left", times, values
    except Exception:
        pass
    return None, None, None
```

iii. CONVERSION_NOTES: "use left camera only."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is, linearly interpolated onto the trial bin grid and then discretized into 3 categories using global tertiles.

ii.
```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

iii. No smoothing or normalization applied.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: global tertiles across the entire dataset, not per-session percentiles.

ii.
```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1/3, 2/3]).astype(np.float32).tolist())
```

iii. Global tertiles rather than per-session percentiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel: linearly interpolated onto bin right edges within each trial window.

ii.
```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. Bin right edges, not bin centres.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling:
- NaN checks on 6 trial columns before processing
- Sessions without whisker motion energy are skipped entirely
- Trials where wheel or whisker interpolation fails (insufficient coverage) are dropped
- Sessions with fewer than 2 valid trials are skipped
- Trials with all-zero neural activity are removed after binning
- Probes with no good units are skipped
- Sessions with no valid probes after QC are skipped
- feedbackType NaN check and trial-length cap are additional protections

ii.
```python
for col in TRIAL_NAN_EXCLUDE:
    mask &= trials[col].notna().to_numpy()

if wheel_interp is None or whisker_interp is None:
    continue

nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. The AI implemented more aggressive missing-data handling than the reference.

## 10-a. What are the most time-consuming steps of the code?

i. Loading the spike sorting data from disk is the most time-consuming step, as each probe's spike arrays can be hundreds of megabytes. The AI uses memory-mapped loading (`mmap_mode="r"`) to mitigate this, but still must read and filter all spikes. Additionally, the AI calls `one.alyx.rest("brain-regions", "list")` to build region maps, which is an API call at startup.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. I/O dominated by spike sorting file reads.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops:
- The per-trial spike binning loop in `bin_probe_spikes` iterates over each trial individually
- The per-trial behavioral interpolation loop iterates over trials one at a time
Both could potentially be vectorized.

ii.
```python
for trial_idx, start_time in enumerate(trial_starts):
    # ... spike binning per trial ...
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
```

```python
for trial_idx in np.flatnonzero(base_mask):
    # ... per-trial behavioral processing ...
    wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
```

iii. These loops process trials sequentially when batch operations could be more efficient.

## 10-c. What processing does the code repeat multiple times?

i. The `load_probe_info` function loads cluster metrics and channel data for each probe separately, and the `bin_probe_spikes` function re-loads spike times and clusters for each probe. This means spike data files are loaded once per probe per session, which could theoretically be combined if probes shared data structures.

ii.
```python
# In load_probe_info:
metrics_path = one.load_dataset(eid, "clusters.metrics.pqt", ...)
cluster_channels = np.asarray(one.load_dataset(eid, "clusters.channels.npy", ...))

# In bin_probe_spikes:
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. Each probe requires separate loading calls.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things:
- The `build_region_maps` function calls the Alyx API to build a full brain region hierarchy, which is expensive and not strictly necessary if using the local atlas.
- The `trial_len_ok` filter (feedback_times - goCue_times <= 10s) is not part of the reference processing.
- The all-zero neural trial removal is an extra step not in the reference.
- The region filtering (min 5 neurons per session-region, min 2 sessions per region) removes neurons and sessions that the reference would keep.

ii.
```python
records = one.alyx.rest("brain-regions", "list", no_cache=True)

# Extra trial length filter
trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0

# Extra neural trial filter
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. These extra steps add processing time and filter out data that the reference keeps.
