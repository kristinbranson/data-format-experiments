# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the IBL Brain Wide Map (BWM) release. It reads a CSV file (`bwm_release.csv`) listing 699 probe insertions across 459 sessions, then uses the ONE API to load trial, spike, and behavioral data for each session. Sessions are iterated sequentially, grouped by `eid`.

ii.
```python
release_df = pd.read_csv(args.release_csv)
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    cache_dir=args.cache_dir,
    silent=True,
)
grouped = list(release_df.groupby("eid", sort=False))
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The AI matched the reference code's approach of using `bwm_release.csv` and the ONE API, as documented in `CONVERSION_NOTES.md`: "Used the 2025 BWM release table in bwm_release.csv." The reference code (`0_data_caching.py`) uses the same CSV and ONE API connection.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column in the BWM release CSV. Each session's subject is tracked, and a unique subject list is built during final assembly. Subject indices map sessions to subjects.

ii.
```python
subject = str(session_rows["subject"].iloc[0])
# In build_data_from_session_results:
if sess["subject"] not in subject_to_idx:
    subject_to_idx[sess["subject"]] = len(subjects)
    subjects.append(sess["subject"])
subject_idx.append(subject_to_idx[sess["subject"]])
```

iii. The subject splitting follows naturally from the BWM release CSV structure, which lists subject per probe insertion. The AI collects unique subjects as they appear across sessions.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the BWM release CSV. The CSV is grouped by `eid`, and each group (potentially containing multiple probes) forms one session. Multiple probes within a session are merged.

ii.
```python
grouped = list(release_df.groupby("eid", sort=False))
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The reference code also processes per-eid and merges probes within a session using `merge_probes()`. The AI's approach matches this session definition.

## 1-d. How are the data split into trials?

i. Trials are loaded from the ONE API's `trials` object for each session. Each trial is an entry in the resulting DataFrame. Neural data, wheel, and whisker signals are segmented into per-trial windows aligned to stimulus onset.

ii.
```python
trials = one.load_object(eid, "trials", collection="alf").to_df()
# Then iterated per trial:
for trial_idx in np.flatnonzero(base_mask):
    stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
    start_time = stim_on + WINDOW_START  # -0.5s
    end_time = stim_on + WINDOW_END      # +1.5s
```

iii. Trials are defined by the IBL trials table loaded via ONE, consistent with the reference code's use of `SessionLoader` to load trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using multiple criteria: (1) non-NaN values for key columns, (2) reaction time between 0.08s and 2.0s, (3) no-choice trials excluded, (4) max trial length 10s, (5) valid wheel and whisker interpolation, (6) valid choice and prior values, (7) nonzero neural activity.

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

iii. The AI's CONVERSION_NOTES confirm these match the reference code's `load_trials_and_mask` function, which uses the same defaults: `min_rt=0.08`, `max_rt=2.0`, `nan_exclude='default'` (same 6 columns), `exclude_nochoice=True`, and `max_trial_len=10.0`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` files for each probe, along with `clusters.metrics.pqt` for quality labels, `clusters.channels.npy` for channel assignments, and `channels.brainLocationIds_ccf_2017.npy` for region mapping.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
metrics = pd.read_parquet(metrics_path)
```

iii. These are the standard IBL spike sorting outputs. The reference code loads via `SpikeSortingLoader` which ultimately accesses the same underlying data files.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms non-overlapping bins within a [-0.5, 1.5]s window relative to stimulus onset, producing spike counts per neuron per time bin. Multiple probes within a session are concatenated along the neuron axis. The result is stored as spike counts (uint16, later converted to float16).

ii.
```python
BINSIZE = 0.02
WINDOW_START = -0.5
WINDOW_END = 1.5
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))  # 100

probe_counts = np.zeros((len(trial_starts), len(probe.good_cluster_ids), NBINS), dtype=np.uint16)
for trial_idx, start_time in enumerate(trial_starts):
    end_time = start_time + (WINDOW_END - WINDOW_START)
    # ... bin spikes
    bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)

neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
```

iii. The reference code uses `bincount2D` or `get_spike_counts_in_bins` for spike binning with the same parameters (`binsize=0.02`, `time_window=(-0.5, 1.5)`). The AI's manual binning achieves the same result.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by: (1) cluster label >= 1.0 (good units), (2) must be in grey matter, (3) region acronym not in {void, root, grey}, (4) after Beryl remapping, at least 5 neurons per region per session, and (5) region must appear in at least 2 sessions.

ii.
```python
GOOD_LABEL_THRESHOLD = 1.0
MIN_NEURONS_PER_SESSION_REGION = 5
MIN_SESSIONS_PER_REGION = 2

good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))

# In apply_region_filters:
valid_regions = set(unique_names[counts >= MIN_NEURONS_PER_SESSION_REGION].tolist())
globally_valid_regions = {
    region_name for region_name, session_ids in sessions_with_region.items()
    if len(session_ids) >= MIN_SESSIONS_PER_REGION
}
```

iii. The reference code uses `qc=1` for `load_spiking_data` (label >= 1), and the Beryl region mapping. The min-neurons and min-sessions filters are additional constraints the AI added based on the data paper's description of region criteria. However, the reference code (`0_data_caching.py`) does NOT apply these min-neuron or min-session filters - it uses `select_brain_regions` which simply masks by region name. These are filtering criteria described in the data paper but not in the methods paper's code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). The trial window starts at `stimOn_times - 0.5s` and ends at `stimOn_times + 1.5s`.

ii.
```python
trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
# WINDOW_START = -0.5
```

iii. The reference code uses `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)`, which matches exactly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), yielding 100 bins per trial. No temporal rebinning is applied - spikes are directly binned at this resolution.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))  # 100
```

iii. The reference code uses `binsize=0.02` and `interval_len=2`, giving 100 bins. This matches the AI's implementation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` in the trials table and the defined time window parameters.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
# This produces times from -0.48 to 1.5 in 0.02 steps (right bin edges)
```

iii. Time since stimulus onset is computed as a deterministic time vector based on the window parameters, not directly from raw data. It represents the right edge of each time bin relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed time vector is created using `WINDOW_START + BINSIZE * arange(1, NBINS+1)`, producing values from -0.48s to 1.50s in 0.02s steps. This represents the right edges of the 100 time bins.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
input_trial = np.vstack([
    relative_time,
    np.full(NBINS, trial_num, dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. This is a straightforward computation. The use of right bin edges (starting at -0.48 rather than -0.50) is consistent with the reference code's interpolation approach for behavioral signals.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time vector is the same length as the neural data (100 bins) and represents the same temporal grid, so they are inherently aligned.

ii.
```python
# Both have NBINS = 100 time points
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
# Neural: probe_counts shape (n_trials, n_neurons, NBINS)
```

iii. Since both neural and time input share the same binning scheme, alignment is automatic.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

ii.
```python
block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))
```

iii. Block boundaries are identified when `probabilityLeft` changes value, and trial number counts up within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through all trials sequentially. When `probabilityLeft` equals the previous trial's value, the counter increments; otherwise it resets to 1. The result is a per-trial integer that counts trials within each probability block.

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

iii. This computes trial number in block as a running counter that resets at block boundaries. The value is then broadcast across all time bins for each trial as a constant per-trial input.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the IBL trials table.

ii.
```python
choice_val = float(trials.iloc[trial_idx]["choice"])
```

iii. The IBL trials table provides choice as 1 (left), -1 (right), or 0 (no response).

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL choice values are remapped: `choice == 1` (left) becomes 0, `choice == -1` (right) becomes 1. No-choice trials (`choice == 0`) are excluded.

ii.
```python
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue
```

iii. This mapping matches the instruction specification: "left = 0, right = 1". The AI's CONVERSION_NOTES confirm this convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
```

iii. `probabilityLeft` is a standard IBL trials table column representing the prior probability of the stimulus appearing on the left.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probability value is mapped to categorical values: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Trials with other values are excluded.

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

iii. This matches the instruction specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from wheel timestamps and position data loaded via the ONE API (`_ibl_wheel.timestamps` and `_ibl_wheel.position`).

ii.
```python
def load_wheel_speed(one: ONE, eid: str) -> tuple[np.ndarray, np.ndarray]:
    wheel = one.load_object(eid, "wheel", collection="alf")
    timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
    position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. The AI loads wheel data directly from the ONE API, accessing the raw wheel position and timestamps.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1000 Hz, then velocity is computed using a Butterworth low-pass filter (corner frequency 20 Hz, order 8). Speed is the absolute value of velocity.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered

interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

iii. The AI imports and uses the `brainbox.behavior.wheel` functions directly. However, the reference code (`ibl_data_utils.py`) uses `SessionLoader.load_wheel()` which produces velocity via Gaussian smoothing (`sess_loader.wheel['velocity']`), NOT the Butterworth filter. The AI chose the Butterworth approach from `wheel.py` instead of the Gaussian approach used by `SessionLoader`. This is a different processing method.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using global tertile edges (33rd and 67th percentiles) computed across all wheel speed values from all sessions.

ii.
```python
wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
wheel_edges = tuple(np.quantile(wheel_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())

def discretize(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge], dtype=np.float32), right=False).astype(np.int8)
```

iii. The instructions specify "discretized into 3 bins" without specifying the method. Using global tertiles is a reasonable approach to produce balanced bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is linearly interpolated onto the right edges of the 20ms trial time bins, matching the neural data's temporal grid.

ii.
```python
def interpolate_trial_signal(signal_times, signal_values, start_time, end_time, binsize=BINSIZE):
    sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
    interp_vals = np.interp(sample_times, times, values)
    return np.asarray(interp_vals, dtype=np.float32)
```

iii. The reference code uses `np.linspace(interval_begs + binsize, interval_ends, n_bins)` for interpolation times. The AI's `start_time + binsize * arange(1, NBINS+1)` produces equivalent values since `linspace(-0.5 + 0.02, 1.5, 100)` equals `-0.48, -0.46, ..., 1.50`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from left camera ROI motion energy data (`leftCamera.times` and `leftCamera.ROIMotionEnergy`).

ii.
```python
def load_whisker_motion_energy(one: ONE, eid: str):
    cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
    times = np.asarray(cam["times"], dtype=np.float64)
    values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
```

iii. The reference code loads via `SessionLoader.load_motion_energy(views=['left'])` which accesses `whiskerMotionEnergy` from the left camera. The AI loads `ROIMotionEnergy` directly. These should contain the same data but the attribute names differ.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy values are used directly - no additional processing (no smoothing or filtering) is applied beyond the interpolation to trial time bins.

ii.
```python
# Direct loading, no additional processing:
values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
# Then interpolated per trial in interpolate_trial_signal()
```

iii. The reference code also uses the motion energy values directly without additional processing, just interpolating them to the trial time bins.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile edges (33rd and 67th percentiles) computed across all whisker motion energy values from all sessions.

ii.
```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
```

iii. Same justification as wheel speed discretization - tertiles produce balanced bins.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: linearly interpolated onto the right edges of the 20ms trial time bins.

ii.
```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
# Uses same sample_times = start_time + binsize * arange(1, NBINS + 1)
```

iii. This matches the reference code's behavioral signal interpolation approach.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple missing data scenarios are handled: (1) Sessions without whisker motion energy are skipped entirely. (2) Trials where wheel or whisker interpolation fails (insufficient data coverage) are excluded. (3) Trials with all-zero neural activity are removed. (4) Sessions with fewer than 2 valid trials are skipped. (5) NaN values in key trial columns cause trial exclusion. (6) Sessions with no good neural units are skipped.

ii.
```python
# Missing whisker data -> skip session
if whisker_times is None or whisker_me is None:
    print(f"Skip session {eid}: no whisker motion energy trace available.")
    return None

# Failed interpolation -> skip trial
if wheel_interp is None or whisker_interp is None:
    continue

# All-zero neural -> remove trial
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. The AI's approach is conservative - it prefers to exclude problematic data rather than impute. The reference code uses similar quality checks via `good_interval` masks in `get_behavior_per_interval`.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading spike data from disk (`np.load` with memory mapping), (2) Binning spikes per trial (`bin_probe_spikes`), (3) Loading and interpolating wheel data, (4) Loading and interpolating whisker motion energy, (5) Building region maps via Alyx REST API.

ii.
```python
# Spike binning - iterates over every trial:
for trial_idx, start_time in enumerate(trial_starts):
    # searchsorted + add.at per trial

# Region map building - REST API call:
records = one.alyx.rest("brain-regions", "list", no_cache=True)
```

iii. The spike binning loop is O(n_trials * n_spikes_per_trial) and is the computational bottleneck for each session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main spike binning loop in `bin_probe_spikes` iterates over trials sequentially. This could be vectorized by assigning spikes to trials using vectorized searchsorted, then using groupby operations. The trial-level loop in `process_session` (iterating over `base_mask` for behavioral processing) could also be partially vectorized.

ii.
```python
# Current trial-by-trial loop:
for trial_idx, start_time in enumerate(trial_starts):
    start_idx = np.searchsorted(kept_times, start_time, side="left")
    end_idx = np.searchsorted(kept_times, end_time, side="left")
    # ...
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
```

iii. The reference code uses multiprocessing (`multiprocessing.Pool`) to parallelize spike binning across trials, which the AI's code does not do.

## 10-c. What processing does the code repeat multiple times?

i. (1) The `apply_region_filters` function re-applies Beryl mapping after initial conversion, duplicating region mapping work. (2) The code loads probe spike data separately per probe, then concatenates - the reference code merges probes first. (3) The `interpolate_trial_signal` function is called separately for wheel and whisker per trial, each doing its own `searchsorted`.

ii.
```python
# Region mapping applied twice:
# First in load_probe_info (raw Allen -> acronym)
# Then in apply_region_filters (acronym -> Beryl)
session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
```

iii. The two-stage region mapping (first raw, then Beryl) is somewhat redundant but ensures correctness.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code loads and processes all probe data before applying region filters that may remove neurons, meaning some spike binning work is wasted. (2) The code computes `block_trial_num` for ALL trials (including excluded ones). (3) Trial metadata (`n_trials_total`, `n_trials_base_valid`) is tracked but only used for logging. (4) The `build_region_maps` function fetches ALL brain regions from the Alyx API, though only a subset are used. (5) The grey matter hierarchy check traverses parent chains for all regions.

ii.
```python
# block_trial_num computed for all trials, not just kept ones:
block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))

# All brain regions fetched:
records = one.alyx.rest("brain-regions", "list", no_cache=True)
```

iii. These are minor inefficiencies that don't affect correctness.
