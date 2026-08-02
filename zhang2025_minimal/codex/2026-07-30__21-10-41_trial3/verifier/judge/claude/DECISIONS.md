# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using the ONE API connected to the IBL OpenAlyx server. It reads a release CSV file (`bwm_release.csv`) listing 699 probe insertions across 459 sessions, then iterates over sessions grouped by `eid`. For each session, it loads trials, spike data per probe, wheel data, and whisker motion energy data via ONE API calls.

ii.
```python
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    cache_dir=args.cache_dir,
    silent=True,
)
# ...
release_df = pd.read_csv(args.release_csv)
grouped = list(release_df.groupby("eid", sort=False))
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The AI documented in CONVERSION_NOTES.md that it used the BWM release table matching the data paper statistics (459 sessions, 699 insertions, 139 subjects, 12 labs). This matches the reference code's approach of using `bwm_release.csv` and ONE API.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column of the release CSV. Each session's subject is tracked, and a unique subject list is built during final assembly. Subjects are not pre-grouped; instead, each session's subject is extracted from the release CSV rows.

ii.
```python
subject = str(session_rows["subject"].iloc[0])
# ...
if sess["subject"] not in subject_to_idx:
    subject_to_idx[sess["subject"]] = len(subjects)
    subjects.append(sess["subject"])
```

iii. The AI's approach is consistent with the reference code which also groups by `eid` (session) and extracts subject info from the release CSV.

## 1-c. How are the data split into sessions?

i. Sessions are split by grouping the release CSV by `eid`. Each unique `eid` corresponds to one session. Multiple probes within the same session are merged (neurons concatenated). After processing, sessions are filtered by region criteria.

ii.
```python
grouped = list(release_df.groupby("eid", sort=False))
# ...
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. This matches the reference code's approach where sessions are identified by `eid` and probes within a session are merged.

## 1-d. How are the data split into trials?

i. Trials are loaded per session using `one.load_object(eid, "trials", collection="alf")`. Each trial is one row in the resulting DataFrame. Trials are then processed individually within the stimulus-onset-aligned window [-0.5, 1.5] seconds.

ii.
```python
def load_trials(one: ONE, eid: str) -> pd.DataFrame:
    trials = one.load_object(eid, "trials", collection="alf").to_df()
    # ...
    return trials
```

iii. This is consistent with the reference code which loads trials via `SessionLoader` and processes them per-trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial filters:
- Required non-NaN values for: `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
- Reaction time (firstMovement_times - stimOn_times) must be in [0.08, 2.0] seconds
- Excluded no-choice trials (choice == 0)
- Trial length (feedback_times - goCue_times) must be <= 10.0 seconds
- Additionally, trials where wheel or whisker interpolation fails are dropped
- Trials with all-zero neural activity are dropped

ii.
```python
TRIAL_NAN_EXCLUDE = (
    "stimOn_times", "choice", "feedback_times", "probabilityLeft",
    "firstMovement_times", "feedbackType",
)

def make_base_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(trials), dtype=bool)
    for col in TRIAL_NAN_EXCLUDE:
        mask &= trials[col].notna().to_numpy()
    reaction_time = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    mask &= reaction_time >= 0.08
    mask &= reaction_time <= 2.0
    mask &= trials["choice"].to_numpy() != 0
    # ...
    trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
    mask &= trial_len_ok
    return mask
```

iii. The AI documented these filters in CONVERSION_NOTES.md and stated they matched the reference code and paper descriptions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` files for each probe, along with `clusters.metrics.pqt` for quality filtering, `clusters.channels.npy` for channel-to-region mapping, and `channels.brainLocationIds_ccf_2017.npy` for brain region assignment.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. The AI documented using spike-sorted data from Kilosort, consistent with the reference code which also uses spike times and cluster assignments.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into non-overlapping 20ms bins within a [-0.5, 1.5] second window relative to stimulus onset (100 bins total). Multiple probes within a session are concatenated along the neuron dimension. The result is spike counts per neuron per time bin, stored as float16.

ii.
```python
def bin_probe_spikes(one, eid, probe, trial_starts):
    # ...
    probe_counts = np.zeros((len(trial_starts), len(probe.good_cluster_ids), NBINS), dtype=np.uint16)
    for trial_idx, start_time in enumerate(trial_starts):
        end_time = start_time + (WINDOW_END - WINDOW_START)
        # ...
        bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
        np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
    return probe_counts
```

iii. The AI described this in CONVERSION_NOTES.md as "Spike counts are binned into non-overlapping 20 ms bins" with probes merged within session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies multiple neuron quality filters:
1. Units must have `clusters.metrics.label >= 1.0` (well-isolated)
2. Units must be in grey matter regions (Allen CCF)
3. Region acronyms must not be "void", "root", or "grey"
4. After Beryl remapping, regions must have >= 5 neurons per session
5. Regions must appear in >= 2 sessions

ii.
```python
GOOD_LABEL_THRESHOLD = 1.0
# ...
good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))
```

iii. The AI documented these filters as matching the data paper's criteria for well-isolated neurons and the region filtering described in the paper ("restricted to regions that were designated grey matter... contained at least five well-isolated neurons per session and were recorded from in at least two such sessions").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset. For each trial, the window is [stimOn_times - 0.5, stimOn_times + 1.5] seconds. Spike times within this window are binned into 100 time bins of 20ms each.

ii.
```python
WINDOW_START = -0.5
WINDOW_END = 1.5
# ...
trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
# ...
for trial_idx, start_time in enumerate(trial_starts):
    end_time = start_time + (WINDOW_END - WINDOW_START)
```

iii. The AI documented alignment to stimulus onset with [-0.5, 1.5] window, consistent with both the instructions and the reference code (`'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)`).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s) bins, producing 100 bins per trial over the 2-second window. No rebinning is applied; spikes are directly binned at this resolution.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))  # = 100
```

iii. The AI noted in CONVERSION_NOTES.md: "Common bin size: 20 ms. This yields 100 bins per trial, consistent with the methods-paper summary that uses 2 s trials with 20 ms bins."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from a raw data variable; it is computed as a deterministic time vector based on the bin structure of the trial window.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
# produces times: -0.48, -0.46, ..., 1.48, 1.50
```

iii. This represents the right edge of each 20ms bin relative to stimulus onset, matching the reference code's interpolation time points (`np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time vector is computed as evenly spaced points from `WINDOW_START + BINSIZE` to `WINDOW_END`, representing the right edge of each bin. This yields values from -0.48s to 1.50s in 0.02s increments.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. This is a straightforward computation with no complex processing.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time vector is the same for all trials (a fixed grid), so it is inherently aligned with the neural data bins. Each element in the time vector corresponds to one time bin in the neural data.

ii.
```python
input_trial = np.vstack([
    relative_time,
    np.full(NBINS, trial_num, dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. The time vector and neural bins share the same binning structure, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from `probabilityLeft`, which indicates the block identity. Block changes are detected when `probabilityLeft` changes value between consecutive trials.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.zeros(len(prob_left), dtype=np.float32)
    # ...
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

iii. The AI tracks block transitions by detecting when `probabilityLeft` changes, then counts up from 1 within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through all trials sequentially. When `probabilityLeft` is the same as the previous trial, the counter increments. When it changes (indicating a new block), the counter resets to 1. The result is broadcast to all time bins as a per-trial constant.

ii.
```python
block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))
# ...
trial_num = float(block_trial_num[trial_idx])
np.full(NBINS, trial_num, dtype=np.float32)
```

iii. This computation is done on the full trials DataFrame before filtering, so trial numbers reflect the original position in the session. The trial number is then applied only to kept trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials data.

ii.
```python
choice_val = float(trials.iloc[trial_idx]["choice"])
```

iii. Straightforward extraction from the IBL trials table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL convention where `choice == 1` means left and `choice == -1` means right is converted to the decoder format: left = 0, right = 1. Trials with `choice == 0` (no response) are excluded.

ii.
```python
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue
# ...
np.full(NBINS, choice_out, dtype=np.int8)
```

iii. The AI documented this mapping in CONVERSION_NOTES.md, consistent with the instruction's specification (left = 0, right = 1).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of the trials data.

ii.
```python
prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
```

iii. Direct extraction from the trials table's `probabilityLeft` field.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probability value is mapped to a categorical index: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Trials with other values are excluded. The output is broadcast as a constant across all time bins.

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
# ...
np.full(NBINS, prior_out, dtype=np.int8)
```

iii. The AI documented this mapping in CONVERSION_NOTES.md, matching the instruction's specification.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps` and `_ibl_wheel.position` (loaded via `one.load_object(eid, "wheel", collection="alf")`).

ii.
```python
def load_wheel_speed(one: ONE, eid: str) -> tuple[np.ndarray, np.ndarray]:
    wheel = one.load_object(eid, "wheel", collection="alf")
    timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
    position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. The AI documented using wheel timestamps and position, consistent with the reference code's approach through `SessionLoader.load_wheel()`.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000 Hz, then velocity is computed using an 8th-order Butterworth low-pass filter with 20 Hz corner frequency (`velocity_filtered`). Speed is the absolute value of velocity. The continuous speed signal is then linearly interpolated to the right edges of the 20ms trial bins.

ii.
```python
interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

iii. The AI documented: "interpolated to 1000 Hz, velocity computed with the same Butterworth-filtered method used by brainbox.behavior.wheel.velocity_filtered, speed defined as abs(velocity)." This matches the reference code's `SessionLoader.load_wheel()` which internally uses the same `interpolate_position` and `velocity_filtered` functions.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using global tertile edges (1/3 and 2/3 quantiles) computed across ALL time points from ALL sessions. Values are categorized as low (0), medium (1), or high (2) using `np.digitize`.

ii.
```python
wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
wheel_edges = tuple(np.quantile(wheel_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
# ...
def discretize(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge], dtype=np.float32), right=False).astype(np.int8)
```

iii. The AI documented: "discretized into 3 bins using global tertiles over the converted dataset."

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. The continuous wheel speed signal is linearly interpolated to the right edges of each 20ms bin within the trial window. The interpolation targets are at `start_time + binsize * [1, 2, ..., 100]`, matching the neural time bins. If interpolation fails (insufficient data coverage), the trial is excluded.

ii.
```python
def interpolate_trial_signal(signal_times, signal_values, start_time, end_time, binsize=BINSIZE):
    # ...
    sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
    interp_vals = np.interp(sample_times, times, values)
```

iii. The AI documented: "Continuous traces are linearly interpolated onto the right edge of each 20 ms trial bin, following the reference utility logic."

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from the left camera's `ROIMotionEnergy` and camera `times`, loaded via `one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")`.

ii.
```python
def load_whisker_motion_energy(one: ONE, eid: str):
    try:
        cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
        times = np.asarray(cam["times"], dtype=np.float64)
        values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
        if len(times) and len(times) == len(values):
            return "left", times, values
    except Exception:
        pass
    return None, None, None
```

iii. The AI documented: "use left camera only" and "loaded from ROIMotionEnergy and camera times."

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy time series is linearly interpolated to the right edges of each 20ms bin within the trial window, using the same `interpolate_trial_signal` function as wheel speed.

ii.
```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

iii. No additional preprocessing is applied beyond interpolation. The motion energy values are used as-is from the camera data.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized identically to wheel speed: using global tertile edges (1/3 and 2/3 quantiles) computed across all time points from all sessions, producing 3 categories (low=0, medium=1, high=2).

ii.
```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
# ...
discretize(whisker_vals, whisker_edges)
```

iii. The AI documented: "discretized into 3 bins using global tertiles over the converted dataset, bin names are low, medium, high."

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identical to wheel speed alignment: the continuous signal is linearly interpolated to the right edges of each 20ms bin in the trial window. If interpolation fails, the trial is excluded.

ii.
```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
if wheel_interp is None or whisker_interp is None:
    continue
```

iii. Same alignment approach as wheel speed, documented in CONVERSION_NOTES.md.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data at multiple levels:
- Trials with NaN in key columns are excluded via the trial mask
- Trials where wheel or whisker interpolation fails (insufficient data coverage) are skipped
- Sessions without whisker motion energy (left camera) are skipped entirely
- Trials with all-zero neural activity after binning are dropped
- Sessions with fewer than 2 valid trials are skipped
- Non-finite values in interpolated signals cause trial exclusion

ii.
```python
# NaN-based trial exclusion
for col in TRIAL_NAN_EXCLUDE:
    mask &= trials[col].notna().to_numpy()

# Interpolation failure handling
if wheel_interp is None or whisker_interp is None:
    continue

# Zero-neural trial handling
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
if not np.all(nonzero_trial_mask):
    session_counts = session_counts[nonzero_trial_mask]
```

iii. The AI applied conservative missing-data handling, dropping trials/sessions rather than imputing values.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading spike data from disk for each probe (`np.load` with mmap, then filtering and binning)
2. Building the region maps by querying `one.alyx.rest("brain-regions", "list")` for all brain regions
3. The per-trial spike binning loop in `bin_probe_spikes`
4. Loading and interpolating wheel and whisker data per trial

ii.
```python
# Spike binning loop (per trial)
for trial_idx, start_time in enumerate(trial_starts):
    # ...
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)

# Region map building (API call)
records = one.alyx.rest("brain-regions", "list", no_cache=True)
```

iii. The spike binning loop processes each trial sequentially, and the API call for brain regions is a network operation.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The per-trial spike binning loop in `bin_probe_spikes` iterates over trials with `np.add.at`, which could use vectorized histogram operations
2. The per-trial behavior interpolation loop in `process_session` iterates over valid trials individually
3. The `compute_trial_number_in_block` loop iterates element-by-element over probability values

ii.
```python
# Per-trial spike binning (could be vectorized)
for trial_idx, start_time in enumerate(trial_starts):
    # ...
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)

# Per-trial behavior interpolation (could be vectorized)
for trial_idx in np.flatnonzero(base_mask):
    wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
    whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

iii. The reference code uses `bincount2D` and multiprocessing for spike binning and behavior interpolation respectively.

## 12-c. What processing does the code repeat multiple times?

i. The code does the following repeated work:
1. Brain region acronym lookup (`id_to_acronym.get(int(rid), "void")`) is done for every cluster in every probe
2. Region mapping to Beryl is done twice: once during initial load (raw acronyms stored) and again in `apply_region_filters` using `brainreg.acronym2acronym`
3. The region-to-index mapping is rebuilt from scratch in both `build_data_from_session_results` and `apply_region_filters`

ii.
```python
# First region mapping in load_probe_info
region_acronyms = np.array([id_to_acronym.get(int(rid), "void") for rid in region_ids], dtype=object)

# Second region mapping in apply_region_filters
session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
```

iii. The double region mapping is notable - raw Allen CCF acronyms are stored first, then remapped to Beryl later.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data that may be partially discarded:
1. The `build_region_maps` function queries ALL brain regions from the Alyx API to build grey-matter ancestry checks, even though only a subset of regions are used
2. Neural data is initially stored as float16 spike counts, then re-cast to float16 again during region filtering (`trial[final_keep].astype(np.float16, copy=False)`)
3. Session metadata (lab, date, camera_view, probe_names) is collected and stored but not used by the decoder
4. The `nonzero_trial_mask` check drops trials with zero neural activity, but these trials' behavior data was already computed

ii.
```python
# Metadata collected but unused by decoder
session_info.append({
    "eid": sess["eid"],
    "subject": sess["subject"],
    "lab": sess["lab"],
    "date": sess["date"],
    "camera_view": sess["camera_view"],
    # ...
})
```

iii. The metadata collection is not strictly unnecessary (it's useful for documentation), but represents processing that doesn't affect decoder inputs/outputs.
