# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a release manifest CSV (`bwm_release.csv`) from the reference code directory to enumerate all sessions and their probe insertions. If `DATALIMIT_SUBSET.csv` exists, sessions are restricted to that subset. For each session, a ONE client is constructed against the local cache, and `SessionLoader` loads trials, wheel, and motion energy, while `SpikeSortingLoader` loads spikes per probe. Each session is processed as an independent job, optionally cached as a pickle shard for restartability.

ii.
```python
release = pd.read_csv(release_file, index_col=0)
subset = _read_subset(ROOT / "data" / "DATALIMIT_SUBSET.csv")
if subset is not None:
    release = release[release["eid"].astype(str).isin(subset)]

for eid, rows in release.groupby("eid", sort=False):
    first = rows.iloc[0]
    jobs.append({
        "eid": str(eid),
        "subject": str(first["subject"]),
        ...
    })
```

iii. The agent explored the data directory structure and reference code, finding the `bwm_release.csv` file which lists every session and probe in the release. It chose this as the authoritative session list rather than using `one.search()`.

## 1-b. How are the data split into subjects?

i. The subject name comes from the `bwm_release.csv` file, one per session. At assembly, unique subjects are sorted and indexed.

ii.
```python
subjects = sorted({session["subject"] for session in sessions})
subject_map = {name: i for i, name in enumerate(subjects)}
```

iii. The release CSV already contains the subject name per session, so no additional parsing is needed.

## 1-c. How are the data split into sessions?

i. Each row-group in `bwm_release.csv` (grouped by `eid`) is one session. Sessions are processed independently as parallel jobs.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    ...
    jobs.append({"eid": str(eid), ...})
```

iii. Sessions are the natural unit in the release CSV and in the IBL data organization.

## 1-d. How are the data split into trials?

i. The trials table loaded by `SessionLoader` has one row per trial. Trials are filtered by quality criteria, and the surviving indices form the trial list.

ii.
```python
loader.load_trials()
trials = loader.trials.copy()
```

iii. The trials table is already one row per trial; no splitting decision is needed.

## 1-e. How are trials filtered based on quality controls?

i. Six conditions are applied: (1) all required trial columns must be finite, (2) reaction time between 0.08 and 2.0 s, (3) go-cue-to-feedback time <= 10 s, (4) choice != 0 (no-response excluded), (5) probabilityLeft in {0.2, 0.5, 0.8}, and (6) both wheel and whisker motion energy must span the trial window. Additionally, at least 2 trials must survive for a session to be kept.

ii.
```python
finite = np.all(np.isfinite(trials[required].to_numpy(dtype=float)), axis=1)
reaction_time = trials["firstMovement_times"].to_numpy(dtype=float) - trials["stimOn_times"].to_numpy(dtype=float)
trial_length = trials["feedback_times"].to_numpy(dtype=float) - trials["goCue_times"].to_numpy(dtype=float)
mask = (
    finite
    & (reaction_time >= 0.08)
    & (reaction_time <= 2.0)
    & (trial_length <= 10.0)
    & (choice_raw != 0)
    & np.isin(probability, [0.2, 0.5, 0.8])
)
```

And behavior coverage check in `_interpolate_trials`:
```python
if ib >= ie or ib >= len(times) or ie <= 0:
    good[i] = False
if len(segment_t) < 2 or abs(begin - segment_t[0]) > BIN_SIZE or abs(end - segment_t[-1]) > BIN_SIZE:
    good[i] = False
```

iii. The agent verified trial masks against the reference code's BWM exclusions on a test session (407 of 565 trials matched). The extra `trial_length <= 10.0` and finite-all-required-columns filters go beyond the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` and `spikes.clusters`, loaded by `SpikeSortingLoader`. The cluster table provides anatomical labels (mapped to Beryl), and channels provide merge information.

ii.
```python
spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
spikes, clusters, channels = spike_loader.load_spike_sorting()
merged = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. Spikes times and cluster assignments are the standard source for neural activity in the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over a 2 s window (-0.5 to 1.5 s around stimulus onset), producing 100 bins. When a session has multiple probes, their clusters are merged with offset IDs. The result is stored as **raw spike counts** (float32), NOT divided by bin width to get firing rates.

ii.
```python
BIN_SIZE = 0.020
N_BINS = 100

bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
clu = clusters[lo:hi] + offset
flat_parts.append(((trial * n_neurons + clu[valid]) * N_BINS + bins[valid]))
...
counts = np.bincount(flat, minlength=size)
return counts.reshape(len(onsets), n_neurons, N_BINS).astype(np.float32)
```

iii. The agent followed the reference code's binning approach (20 ms bins, 100 time steps) but did not convert counts to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No quality filtering is applied.** All sorted clusters are retained regardless of their QC label. The AI explicitly chose to match the methods code's `prepare_data(qc=None)` default, keeping all Kilosort-sorted units. Void and root regions are NOT excluded at the cluster level; Beryl mapping is applied for region labels but not used to filter units.

ii.
```python
# Load and retain every Kilosort cluster, exactly as prepare_data(qc=None)
# in the methods repository.
mapped = brain_regions.acronym2acronym(
    merged["acronym"].to_numpy(), mapping="Beryl"
).astype(str)
spike_parts.append((np.asarray(spikes["times"]), local_ids, offset))
region_parts.append(mapped)
offset += n_clusters
```

iii. The agent's metadata states: `"neuron_filter": "all sorted clusters (qc=None), matching the methods code"`. The agent interpreted the reference code's default behavior (no QC argument) as the correct approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window starts at `onset + OFF_START` (-0.5 s) and ends at `onset + OFF_END` (1.5 s), where onset is `stimOn_times`. Spikes are binned relative to this window start.

ii.
```python
starts = onsets + OFF_START
ends = onsets + OFF_END
for trial, (begin, end) in enumerate(zip(starts, ends)):
    lo = np.searchsorted(times, begin, side="left")
    hi = np.searchsorted(times, end, side="left")
    bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
```

iii. All data streams share the same session clock, so alignment is achieved by subtracting each trial's stimulus onset time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins total over 2 s. No rebinning or interpolation is applied to neural data.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

iii. This matches the reference code's `binsize: 0.02` and the method paper's description of 100 time steps of 20 ms.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time input is computed as evenly spaced points relative to stimulus onset.

ii.
```python
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```

iii. The window and bin size come from the reference code's decoding parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are the **right edges** of each 20 ms bin, computed as `np.linspace(-0.5 + 0.02, 1.5, 100)`. This differs from the reference which uses bin centers.

ii.
```python
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
# This produces: [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. The agent's docstring states: "behavior values correspond to the right edge of each neural count bin", following the reference code's `get_behavior_per_interval` function.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time input vector is the same set of time points used for behavior interpolation. Since neural bins are counted from the window start, each bin's count corresponds to the interval [left_edge, right_edge), and the time input gives the right edge of that interval.

ii.
```python
inp[0] = RELATIVE_TIMES
```

iii. The time points used for behavior interpolation and time input are identical, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` marks a new block boundary.

ii.
```python
probability = trials["probabilityLeft"].to_numpy(dtype=float)
trial_in_block = _trial_number_in_block(probability)
```

iii. The trials table has no explicit block identifier, so blocks are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number is **1-based**: the first trial in each block is numbered 1. A new block starts whenever `probabilityLeft` changes or is non-finite. The count is computed before trial exclusions, so dropped trials still advance the count.

ii.
```python
def _trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    out = np.empty(len(probability_left), dtype=np.float32)
    number = 0
    previous = None
    for i, value in enumerate(probability_left):
        if i == 0 or not np.isfinite(value) or value != previous:
            number = 1
        else:
            number += 1
        out[i] = number
        previous = value
    return out
```

iii. The agent's metadata notes: `"trial_number_in_block": "one-based count, computed before trial exclusions"`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column from the trials table, where IBL convention uses +1 for left and -1 for right (and 0 for no response, which is excluded).

ii.
```python
choice_raw = trials["choice"].to_numpy(dtype=float)
mask = ... & (choice_raw != 0) & ...
choice = (choice_raw == 1).astype(np.int8)  # left=-1, right=1
```

iii. No-response trials (choice=0) are dropped by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `choice_raw == 1` to int8, which gives 1 for IBL-left (+1) and 0 for IBL-right (-1). However, the comment in the code says `# left=-1, right=1`, which **incorrectly states the IBL convention**. In IBL, left=+1 and right=-1. The resulting mapping is: left -> 1, right -> 0, which is the **opposite** of the instructions (left=0, right=1).

ii.
```python
choice = (choice_raw == 1).astype(np.int8)  # left=-1, right=1
```
The instructions specify: "left = 0, right = 1".
The reference does: `CHOICE = {1.0: 0, -1.0: 1}` (left -> 0, right -> 1).

iii. The agent appears to have confused the IBL choice convention, resulting in an inverted choice mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column from the trials table, taking values 0.2, 0.5, and 0.8.

ii.
```python
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_lookup[float(x)] for x in probability[candidate]], dtype=np.int8)
```

iii. The mapping matches the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A direct lookup mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. No additional processing.

ii.
```python
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_lookup[float(x)] for x in probability[candidate]], dtype=np.int8)
```

iii. Straightforward recoding as specified in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel position and timestamps loaded by `SessionLoader.load_wheel()`, which internally interpolates and differentiates to produce velocity. The speed is the absolute value of velocity.

ii.
```python
loader.load_wheel()
wheel_times = loader.wheel["times"].to_numpy()
wheel_speed = np.abs(loader.wheel["velocity"].to_numpy())
```

iii. Same approach as the reference: absolute value of the velocity produced by `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` interpolates wheel position onto a 1000 Hz grid and applies a Butterworth low-pass filter to produce velocity. The speed (absolute velocity) is then linearly interpolated onto the trial time grid (bin right edges). Finally, the continuous speed is discretized into three categories using within-session tertiles.

ii.
```python
wheel, wheel_good = _interpolate_trials(wheel_times, wheel_speed, candidate_onsets)
wheel_cat, wheel_cuts = _session_tertiles(wheel)
```

iii. The interpolation and filtering are done by the IBL library; the discretization follows the task requirement for categorical outputs.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Within-session tertiles: the 1/3 and 2/3 quantiles of all wheel speed values across all retained trials in the session are computed, and `np.digitize` assigns each value to one of three bins (0=low, 1=medium, 2=high).

ii.
```python
def _session_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cuts = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.digitize(values, cuts, right=False).astype(np.int8)
    return labels, [float(cuts[0]), float(cuts[1])]
```

iii. This produces three roughly equal-sized categories within each session, matching the reference approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated onto the same time grid used for all behavior (`RELATIVE_TIMES`), which are the right edges of the neural bins. This ensures temporal correspondence with the neural data.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
```

iii. All behavior streams are evaluated at the same time points as the neural bin edges.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The ROI motion energy from the side camera (`leftCamera.ROIMotionEnergy` or `rightCamera.ROIMotionEnergy`) with corresponding frame times. Left camera is preferred; right is a fallback.

ii.
```python
def _load_motion_energy(session_loader):
    for view in ("left", "right"):
        try:
            session_loader.load_motion_energy(views=[view])
            key = f"{view}Camera"
            frame = session_loader.motion_energy[key]
            times = frame["times"].to_numpy()
            values = frame["whiskerMotionEnergy"].to_numpy()
            ...
```

iii. This follows the reference logic of preferring left camera and falling back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the trial time grid (bin right edges), then discretized into session-wise tertiles.

ii.
```python
motion, motion_good = _interpolate_trials(motion_times, motion_values, candidate_onsets)
motion_cat, motion_cuts = _session_tertiles(motion)
```

iii. Same approach as wheel speed: interpolate then discretize.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: within-session tertiles using 1/3 and 2/3 quantiles, producing three categories (0=low, 1=medium, 2=high).

ii.
```python
motion_cat, motion_cuts = _session_tertiles(motion)
```

iii. Identical discretization method as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The motion energy trace is interpolated onto the same `RELATIVE_TIMES` grid (bin right edges) as all other behavior streams and neural bins.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
```

iii. Same alignment approach as wheel speed, using shared time points.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple safeguards: (1) trials with non-finite values in required columns are excluded, (2) trials without sufficient wheel or motion energy coverage are excluded, (3) sessions with fewer than 2 surviving trials raise an error and are skipped, (4) sessions with no neural units raise an error and are skipped, (5) failed sessions are logged and their errors saved to `failures.json`, (6) session shards are written atomically (write to .tmp then rename).

ii.
```python
finite = np.all(np.isfinite(trials[required].to_numpy(dtype=float)), axis=1)
...
if len(candidate) < 2:
    raise ValueError("fewer than two trials pass the BWM trial criteria")
...
if offset == 0:
    raise ValueError("session has no neural units")
```

iii. The agent designed the system to be robust to missing data, with per-session error handling and restartable shards.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (hundreds of megabytes per probe) and the spike binning computation across all trials. The parallel processing architecture (ProcessPoolExecutor with 8 workers) was designed to mitigate the I/O bottleneck.

ii.
```python
spikes, clusters, channels = spike_loader.load_spike_sorting()
...
neural_array = _bin_spikes(spike_parts, onsets, offset)
```

iii. The agent noted the full conversion took significant time across 459 sessions and used parallel workers and restartable shards.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over trials and probe parts with nested Python loops. The behavior interpolation also loops per-trial. Both could potentially be vectorized.

ii.
```python
for times, clusters, offset in spike_parts:
    for trial, (begin, end) in enumerate(zip(starts, ends)):
        lo = np.searchsorted(times, begin, side="left")
        hi = np.searchsorted(times, end, side="left")
        ...
```

```python
for i, onset in enumerate(onsets):
    ...
    row = np.interp(grid, segment_t, segment_v)
```

iii. The per-trial loop structure keeps the code clear but is slower than a fully vectorized approach.

## 10-c. What processing does the code repeat multiple times?

i. The ONE client is constructed fresh in each worker process (`_one()` is called per session job), including loading the Parquet cache tables each time. This is necessary for process isolation but repeats the same cache-loading work.

ii.
```python
def _process_session(job: dict) -> dict:
    ...
    one = _one(job["cache_dir"])
```

iii. Each worker independently initializes the ONE client because database connections cannot be shared across processes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and merges ALL cluster metadata (including QC labels via `merge_clusters`) but never uses the QC labels to filter neurons. It also computes `trial_length` (feedback_times - goCue_times) for filtering, which is not part of the standard BWM trial mask.

ii.
```python
merged = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
# QC labels are loaded but never used for filtering
```

```python
trial_length = trials["feedback_times"].to_numpy(dtype=float) - trials["goCue_times"].to_numpy(dtype=float)
```

iii. The merge_clusters call provides region labels needed for brain_region_idx, but the QC information it also loads is unused.
