# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the list of sessions from a release CSV file (`bwm_release.csv`) which enumerates 459 sessions with their EIDs, subjects, labs, dates, and probe information. For each session, it loads trial data from ALF parquet files (`_ibl_trials.table.pqt`), spike data from per-probe pykilosort directories (`spikes.times.npy`, `spikes.clusters.npy`, etc.), wheel data (`_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy`), and whisker motion energy from camera files. Data is first looked for in a local readonly cache (`data/one_cache`), and if not found, downloaded via the ONE API.

ii.
```python
def load_release_sessions():
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    session_rows = (
        bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
        .reset_index(drop=True)
    )
    probe_rows = {
        eid: grp[["pid", "probe_name"]].reset_index(drop=True)
        for eid, grp in bwm.groupby("eid", sort=False)
    }
    return session_rows, probe_rows
```

```python
def locate_dataset(row, relative_glob: str):
    rel = session_rel_path(row)
    for root in (READONLY_CACHE, WRITABLE_CACHE):
        base = root / rel
        matches = sorted(base.glob(relative_glob))
        if matches:
            return matches[-1]
    return None
```

iii. The AI documented in CONVERSION_NOTES.md that it uses the 459-session BWM release CSV as the canonical session list, matching the data paper's release. It resolves ALF revision paths dynamically.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column of the release CSV. Each session row carries a subject name. The AI builds a unique list of subjects as sessions are processed and assigns each session a `subject_idx` into that list.

ii.
```python
subjects = []
subject_to_idx = {}
# ...
for rec in session_records:
    if rec.subject not in subject_to_idx:
        subject_to_idx[rec.subject] = len(subjects)
        subjects.append(rec.subject)
    subject_idx.append(subject_to_idx[rec.subject])
```

iii. The AI noted 139 subjects in the raw release, with 136 retained after 15 sessions were dropped.

## 1-c. How are the data split into sessions?

i. Sessions are identified by their unique EID from the release CSV. Each row in the release CSV corresponds to one session. The AI processes each session independently, loading trials, neural data, and behavioral data per session. Sessions that fail (e.g., missing whisker data) are skipped.

ii.
```python
session_rows, probe_rows = load_release_sessions()
# ...
for row in session_rows.head(sample_limit).itertuples(index=False):
    rec = process_session(one=one, row=row, probe_df=probe_rows[row.eid], ...)
    session_records.append(rec)
```

iii. The AI documented that 444 of 459 sessions were retained, with 15 dropped due to missing whisker data (14) or zero valid trials after filtering (1).

## 1-d. How are the data split into trials?

i. Trials are loaded from the ALF trials table (`_ibl_trials.table.pqt`) which contains one row per trial. After filtering (see 1-e), valid trial indices are identified and used to extract per-trial neural and behavioral data from the binned arrays.

ii.
```python
trials_df = load_trials_table(one, row)
trial_mask = build_trial_mask(trials_df)
# ...
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
# ...
for idx in valid_idx:
    neural_trial = compact_neural_trial(binned_spikes[idx].T)
    # ...
```

iii. The AI documented that the trial table provides per-trial event times, and that valid trials are those passing all quality filters and having full behavioral coverage.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the following trial filters:
- Exclude trials with NaN in key event columns: `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
- Exclude trials with reaction time < 0.08s or > 2.0s (reaction time = `firstMovement_times - stimOn_times`)
- Exclude trials with `feedback_times - goCue_times > 10s`
- Exclude no-choice trials (`choice == 0`)
- Additionally exclude trials without full wheel or whisker behavioral coverage in the [-0.5, 1.5]s window

ii.
```python
def build_trial_mask(trials_df, min_rt=0.08, max_rt=2.0, nan_exclude="default",
                     min_trial_len=None, max_trial_len=10.0,
                     exclude_unbiased=False, exclude_nochoice=True):
    if nan_exclude == "default":
        nan_exclude = ["stimOn_times", "choice", "feedback_times",
                       "probabilityLeft", "firstMovement_times", "feedbackType"]
    if min_rt is not None:
        query = f"(firstMovement_times - stimOn_times < {min_rt})"
    # ... builds query string with all exclusion criteria ...
    if exclude_nochoice:
        query += " | (choice == 0)"
    return ~trials_df.eval(query)
```

```python
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
```

iii. The AI documented these filters match the reference code's `load_trials_and_mask` function. The reference function has default `exclude_nochoice=True` and is called from `prepare_data` with `max_trial_len=10.0`. The AI also adds behavioral coverage masks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) from the pykilosort spike sorting output for each probe. Cluster quality labels come from `clusters.metrics.pqt` (the `label` column), and brain region assignments come from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
cluster_metrics = pd.read_parquet(cluster_metrics_path)
# ...
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
```

iii. The AI documented that it uses the standard IBL spike sorting outputs from pykilosort.

## 2-b. How is the `neural` data processed?

i. The AI:
1. Loads spike times and cluster assignments per probe
2. Filters to well-isolated clusters (`label >= 1`)
3. Merges probes within a session (re-indexing cluster IDs)
4. Bins spikes into 20ms bins over a [-0.5, 1.5]s window around stimulus onset using `bincount2D`
5. Stores spike counts as uint8 per trial, shape (n_neurons, 100)

ii.
```python
def bin_spiking_data(spikes, trials_df):
    intervals = np.vstack([
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
    ]).T
    binned_array, cluster_ids = get_spike_data_per_interval(
        spikes["times"], spikes["clusters"],
        interval_begs=intervals[:, 0], interval_ends=intervals[:, 1],
        interval_len=TIME_WINDOW[1] - TIME_WINDOW[0], binsize=BIN_SIZE,
    )
    binned_trials = np.asarray([x.T for x in binned_array], dtype=np.float32)
    return binned_trials, cluster_ids
```

iii. The AI documented that this matches the reference code's `bin_spiking_data` approach: stimulus-aligned, 20ms bins, using `bincount2D`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons by requiring `label >= 1` in the cluster metrics (well-isolated units). Probes with zero good units are skipped (but the session is retained if other probes have good units). NaN labels are treated as 0 (excluded).

ii.
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
# ...
if good_cluster_ids.size == 0:
    return None, None  # skip this probe
```

iii. The AI documented that the raw release has exactly 75,708 units with `label >= 1`, matching the paper's reported count of well-isolated neurons. The reference code's `prepare_data` does NOT filter by label (it loads all clusters), but the AI chose to filter, citing the paper's emphasis on 75,708 well-isolated neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the binning window is `[stimOn_times + (-0.5), stimOn_times + 1.5]`, producing 100 time bins of 20ms each.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))  # = 100

intervals = np.vstack([
    trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
    trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
]).T
```

iii. The AI documented that stimulus onset alignment matches both the executable reference code and the instructions' "Temporally align based on stimulus onset" requirement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), yielding 100 bins per trial over the 2s window. No temporal rebinning is applied -- the 20ms bin size is used directly for both neural and behavioral data.

ii.
```python
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))  # 100
```

iii. The AI documented 20ms bins matching the reference code and papers.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time since stimulus onset input is not derived from any raw data variable. It is a deterministic time grid computed from the bin size and time window parameters.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The AI documented that this is a fixed grid representing bin-end times from -0.48s to 1.5s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time input is computed as `np.linspace(-0.5 + 0.02, 1.5, 100)`, giving bin-end times. This results in values from -0.48 to 1.5. The same grid is used identically for every trial.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
# In the per-trial construction:
input_trial = np.vstack([
    TIME_GRID,
    np.full(N_BINS, trial_num, dtype=np.float32),
]).astype(np.float16)
```

iii. The AI noted the range is `[-0.47998, 1.5]` (due to float32 precision).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time grid is constructed to match the neural binning: each time point represents the end of a 20ms bin in the [-0.5, 1.5]s window. Since the neural data is binned over the same window with the same bin size, the time input is inherently aligned.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
# Same N_BINS = 100 as the neural data
```

iii. The AI's processing plots confirmed no temporal misalignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column of the trials table. Block boundaries are detected by changes in `probabilityLeft`.

ii.
```python
def compute_trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(probability_left), dtype=np.float32)
    prev = None
    count = 0
    for idx, value in enumerate(probability_left):
        if np.isnan(value):
            out[idx] = np.nan
            prev = np.nan
            count = 0
            continue
        if prev is None or np.isnan(prev) or not np.isclose(prev, value):
            count = 1
        else:
            count += 1
        out[idx] = count
        prev = value
    return out
```

iii. The AI documented that block transitions are detected by changes in `probabilityLeft` values, and the trial number resets to 1 at each transition.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The function iterates through the `probabilityLeft` array in trial order. When the value changes (or goes from NaN to a number), the counter resets to 1. Otherwise, the counter increments. The result is computed on the full (unfiltered) trial sequence, then indexed into for valid trials. The value is replicated across all 100 time bins for each trial.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
# ... later, for each valid trial:
trial_num = float(block_trial_number[idx])
input_trial = np.vstack([
    TIME_GRID,
    np.full(N_BINS, trial_num, dtype=np.float32),
]).astype(np.float16)
```

iii. The AI documented that the block trial number is per-trial and ranges from 1 to 99 in the full dataset.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii.
```python
choice_code = map_choice(trials_df.iloc[idx]["choice"])

def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0  # left
    if np.isclose(choice_value, -1.0):
        return 1  # right
    raise ValueError(f"Unexpected choice value {choice_value}")
```

iii. The AI documented that the IBL convention is `choice=1` for left and `choice=-1` for right, mapped to 0 and 1 respectively per the instructions.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw `choice` values (+1 for left, -1 for right) are mapped to binary codes (0 for left, 1 for right). No-choice trials (`choice=0`) are excluded by the trial mask. The choice code is then replicated across all 100 time bins as a time-varying output.

ii.
```python
# In build_dataset:
output_trial = np.vstack([
    np.full(N_BINS, choice_code, dtype=np.uint8),
    np.full(N_BINS, prior_code, dtype=np.uint8),
    discretize(wheel_vals, wheel_edges),
    discretize(whisker_vals, whisker_edges),
])
```

iii. The AI documented the sign convention: `left = 0, right = 1`, matching the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])

def map_prior(prob_left: float) -> int:
    if np.isclose(prob_left, 0.2):
        return 0
    if np.isclose(prob_left, 0.5):
        return 1
    if np.isclose(prob_left, 0.8):
        return 2
    raise ValueError(f"Unexpected probabilityLeft value {prob_left}")
```

iii. The AI documented the mapping `0.2 -> 0, 0.5 -> 1, 0.8 -> 2` per the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw `probabilityLeft` values (0.2, 0.5, or 0.8) are mapped to categorical codes (0, 1, 2). The code is replicated across all 100 time bins. Trials with NaN `probabilityLeft` are excluded by the trial mask.

ii.
```python
np.full(N_BINS, prior_code, dtype=np.uint8),
```

iii. The AI documented the distribution in the full dataset: approximately 42% for 0.2, 14% for 0.5, and 44% for 0.8.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` (wheel timestamps) and `_ibl_wheel.position.npy` (wheel position).

ii.
```python
def load_wheel_speed(one: ONE, row):
    timestamps = np.load(ts_path)
    position = np.load(pos_path)
    pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
    vel, _ = velocity_filtered(pos_interp, 1000)
    return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

iii. The AI documented using wheel position and timestamps, consistent with the reference code's `load_target_behavior` for "wheel-speed".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI:
1. Loads wheel position and timestamps
2. Interpolates position to 1000 Hz using `brainbox.behavior.wheel.interpolate_position`
3. Computes filtered velocity using `velocity_filtered` at 1000 Hz
4. Takes the absolute value to get speed
5. Interpolates the speed to the 20ms trial-aligned grid using `scipy.interpolate.interp1d` (linear)

ii.
```python
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

iii. The AI documented that this matches the reference code's processing of wheel speed: `np.abs(velocity)`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using global tertile edges computed across ALL trials and sessions. The edges are the 1/3 and 2/3 quantiles of all wheel speed values. Values are then digitized using `np.digitize` with `right=False`.

ii.
```python
def safe_quantile_edges(values: np.ndarray):
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    # ... handles edge cases ...
    return np.asarray([q1, q2], dtype=np.float32)

wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)

def discretize(values: np.ndarray, edges: np.ndarray):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)
```

iii. The AI documented global tertile edges so bins have consistent semantics across sessions, achieving approximately 1/3 in each bin.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset-aligned window [-0.5, 1.5]s. The continuous wheel speed signal is interpolated to the same 100-point time grid as the neural data using linear interpolation. Trials where the wheel data doesn't cover the full window are excluded.

ii.
```python
def get_behavior_per_interval(target_times, target_vals, trials_df, allow_nans=False):
    interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
    interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
    # ... for each trial ...
    x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
    y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The AI documented that the interpolation grid matches the neural time grid, ensuring temporal alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from camera ROI motion energy files: `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
def load_whisker_motion_energy(one: ONE, row):
    sides = [
        ("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
        ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy"),
    ]
    for side, times_name, energy_name in sides:
        try:
            times = np.load(times_path).astype(np.float32)
            values = np.load(energy_path).astype(np.float32)
            return {"times": times, "values": values}, side
        except Exception:
            continue
    raise FileNotFoundError(...)
```

iii. The AI documented left-camera-first with right-camera fallback, matching the reference code's `bin_behaviors` which tries "left-whisker-motion-energy" first, then "right-whisker-motion-energy".

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly (no additional processing like smoothing or normalization). They are then interpolated to the 20ms trial-aligned grid using linear interpolation, identical to wheel speed processing.

ii.
```python
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
```

iii. The AI documented that whisker motion energy is used as-is from the camera ROI computation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile edges computed across all trials and sessions, then discretized into 3 bins using `np.digitize`.

ii.
```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
# ...
discretize(whisker_vals, whisker_edges)
```

iii. The AI documented approximately 1/3 distribution in each bin across the full dataset.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identical to wheel speed alignment: interpolated to the same stimulus-onset-aligned 20ms grid over [-0.5, 1.5]s. Trials without full whisker coverage are excluded.

ii.
```python
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
# ... combined_mask includes whisker_mask ...
```

iii. The AI confirmed alignment through processing plots.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data at multiple levels:
- **Missing whisker data**: Sessions without any whisker motion energy files (left or right) are skipped entirely (14 sessions).
- **Missing trial events**: Trials with NaN in key event columns are excluded via the trial mask.
- **Missing behavioral coverage**: Trials where wheel or whisker data doesn't fully cover the [-0.5, 1.5]s window are excluded via `wheel_mask` and `whisker_mask`.
- **Missing probes/clusters**: Probes with zero well-isolated clusters are skipped; the session is retained if other probes have good units.
- **NaN cluster labels**: Treated as 0 (excluded) via `fillna(0)`.
- **Zero-valid-trial sessions**: One session had zero valid trials after all filtering and was dropped.

ii.
```python
# Missing cluster labels:
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1

# Probe with no good clusters:
if good_cluster_ids.size == 0:
    return None, None

# Behavioral coverage check in get_behavior_per_interval:
if len(seg_v) == 0:
    good = False
elif np.isnan(interval_begs[interval_idx]) or np.isnan(interval_ends[interval_idx]):
    good = False
elif np.abs(interval_begs[interval_idx] - seg_t[0]) > BIN_SIZE:
    good = False
elif np.abs(interval_ends[interval_idx] - seg_t[-1]) > BIN_SIZE:
    good = False
```

iii. The AI documented all missing data handling in CONVERSION_NOTES.md Steps 9-10, noting that 15 sessions were excluded and explaining each case.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading and processing per-session data (spike sorting files, trials, behavior) - I/O bound
2. Binning spikes into trial-aligned windows using `bincount2D` - computationally intensive for sessions with many spikes
3. Building the final dataset by iterating over all session records and discretizing behavioral outputs

ii.
```python
# Full cache fill timing from conversion notes:
# 429.35s for 444 cached sessions (about 1s/session average)
print(f"  session time: {time.time() - session_start:.2f}s", flush=True)
```

iii. The AI estimated about 3-7 s/session for sample mode and documented that the full conversion completed in well under 15 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. `compute_trial_number_in_block` iterates trial-by-trial to compute block numbers -- could use `np.diff` and `np.cumsum` on `probabilityLeft` changes.
2. The per-trial loop in `process_session` (lines 671-694) that constructs `neural_trials`, `input_trials`, `choice_codes`, etc. could be vectorized using array indexing.
3. `get_behavior_per_interval` loops over trials for interpolation -- could potentially use vectorized interpolation.
4. The `map_choice` and `map_prior` functions are called per-trial instead of vectorized with `np.where` or lookup arrays.

ii.
```python
# Per-trial loop that could be vectorized:
for idx in valid_idx:
    neural_trial = compact_neural_trial(binned_spikes[idx].T)
    choice_code = map_choice(trials_df.iloc[idx]["choice"])
    prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
    # ...
```

iii. The AI did not explicitly identify vectorization opportunities in CONVERSION_NOTES.md but noted it added session-level parallel processing as a speedup.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats:
1. ONE API initialization -- `build_one()` is called in each worker process via `process_session_worker`
2. `BrainRegions()` instantiation -- created per worker process
3. The behavioral data is processed twice in a sense: first interpolated to continuous values, stored, and then later discretized in `build_dataset` -- the continuous values are kept as intermediates
4. `trials_df[ALIGN_EVENT].to_numpy()` is computed multiple times (once in `bin_spiking_data`, once for each behavioral variable in `get_behavior_per_interval`)

ii.
```python
def process_session_worker(row_dict, probe_records, with_diagnostic, record_cache_dir=None):
    one = build_one()  # repeated per worker
    brain_regions = BrainRegions()  # repeated per worker
```

iii. The AI documented session caching as a mitigation -- once a session is cached, it doesn't need to be reprocessed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs several operations whose results are not used in the final dataset:
1. **Diagnostic data**: Full raw wheel/whisker time series are stored in diagnostic records but only used for up to 2 processing plots, then discarded.
2. **Compact session records and repacking**: The `compact_session_record` and `repack_session_cache` functions are infrastructure for intermediate caching that doesn't affect the final output.
3. **Trial numbers in block for all trials**: `compute_trial_number_in_block` is computed for all trials but only a subset of valid trials are used.
4. **Binning spikes for all trials**: `bin_spiking_data` bins ALL trials (including those that will be filtered out), not just valid trials.
5. **Wheel and whisker processing for filtered-out trials**: Behavioral interpolation is done for all trials, then many are excluded.

ii.
```python
# Binning is done for ALL trials before filtering:
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)  # all trials
# Then only valid_idx are used:
for idx in valid_idx:
    neural_trial = compact_neural_trial(binned_spikes[idx].T)
```

iii. The AI documented that the full dataset processing was fast enough (under 15 minutes) that these inefficiencies were acceptable.
