# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI builds a composite ONE index from two staged release directories (`Brainwidemap` for spikes and `2025_Q3_IBL_et_al_BWM` for behavior), merges their dataset tables, and patches the in-memory trial revision path to point to the `#2025-03-03#` staged revision. It then intersects the session EIDs from both releases. When a `DATALIMIT_SUBSET.csv` exists, sessions are restricted to those listed. Data is loaded per session through `SessionLoader` (trials, wheel, motion energy) and `SpikeSortingLoader` (spikes per probe).

ii.
```python
def build_one(verbose=True) -> tuple[One, list[str]]:
    one = One(cache_dir=CACHE_ROOT)
    one.load_cache(SPIKE_RELEASE)
    behavior_one = One(cache_dir=CACHE_ROOT)
    behavior_one.load_cache(BEHAVIOR_RELEASE)
    datasets = pd.concat([one._cache["datasets"], behavior_one._cache["datasets"]])
    datasets = datasets[~datasets.index.duplicated(keep="last")].sort_index()
    trial_rows = datasets["rel_path"].eq("alf/_ibl_trials.table.pqt")
    datasets.loc[trial_rows, "rel_path"] = (
        f"alf/#{TRIAL_REVISION}#/_ibl_trials.table.pqt"
    )
    ...
    spike_eids = set(map(str, one.search()))
    behavior_eids = set(map(str, behavior_one.search()))
    eids = sorted(spike_eids & behavior_eids)
```

iii. The AI discovered a staging discrepancy where the trial table index pointed to unstaged paths. It resolved this by patching only the in-memory ONE index to point to the staged `#2025-03-03#` revision, then loading all data through ONE/brainbox loaders.

## 1-b. How are the data split into subjects?

i. The subject name is obtained from `one.get_details(eid)` for each session. After conversion, unique subjects are collected in first-seen order and `subject_idx` maps each session to its subject.

ii.
```python
details = one.get_details(eid, full=False)
...
subject_names = list(dict.fromkeys(subjects))
subject_lookup = {name: i for i, name in enumerate(subject_names)}
```

iii. ONE provides the subject name per session; no parsing needed.

## 1-c. How are the data split into sessions?

i. Sessions are the unit of the release. The `build_one` function returns a list of session EIDs from the intersection of two releases, one per session.

ii.
```python
eids = sorted(spike_eids & behavior_eids)
```

iii. No splitting needed; the API returns one EID per session.

## 1-d. How are the data split into trials?

i. The trials table has one row per trial. `SessionLoader.load_trials()` returns the full table, and the trial mask selects which rows to keep.

ii.
```python
sess.load_trials(revision=TRIAL_REVISION)
...
base_mask = trial_mask(trials)
```

iii. No splitting needed; trials are rows in the table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) all required trial columns must be finite (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType, goCue_times), (2) reaction time (firstMovement - stimOn) between 0.08 and 2.0 s, (3) trial duration (feedback - goCue) <= 10 s, (4) choice must be -1 or 1 (excludes no-response), (5) probabilityLeft must be one of 0.2, 0.5, 0.8, (6) wheel and whisker motion energy must cover the full trial window, (7) trials with all-zero neural activity are removed.

ii.
```python
def trial_mask(trials: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(trials), dtype=bool)
    for col in REQUIRED_TRIAL_COLUMNS:
        mask &= np.isfinite(trials[col].to_numpy(dtype=float))
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    duration = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    choice = trials["choice"].to_numpy()
    prior = trials["probabilityLeft"].to_numpy()
    mask &= (rt >= 0.08) & (rt <= 2.0)
    mask &= duration <= 10.0
    mask &= np.isin(choice, (-1, 1))
    mask &= np.isin(np.round(prior, 6), (0.2, 0.5, 0.8))
    return mask
```

And the behavioral coverage and all-zero neural filters:
```python
keep = base_mask & wheel_good & whisker_good
...
neural_valid = np.asarray([np.any(x) for x in neural], dtype=bool)
if not np.all(neural_valid):
    selected = selected[neural_valid]
```

iii. The AI states these filters match the Zhang et al. reference code's `load_trials_and_mask`, which imposes the same RT bounds, duration limit, event finiteness, and no-choice exclusion. The all-zero neural filter was added after finding trials outside valid recording intervals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe, loaded via `SpikeSortingLoader`. The cluster table supplies region acronyms and quality labels (stored as metadata but not used for filtering).

ii.
```python
loader = SpikeSortingLoader(eid=eid, pname=pname, one=one)
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
...
all_times.append(np.asarray(spikes["times"], dtype=np.float64)[valid])
all_clusters.append(spike_cluster[valid] + offset)
```

iii. Spike times and cluster IDs are the standard IBL spike sorting outputs.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 half-open 20-ms bins over the [-0.5, 1.5) s window around stimulus onset. The counts are stored directly as float32 (NOT converted to firing rates). When a session has multiple probes, their clusters are merged with offset indices and sorted by time.

ii.
```python
def bin_spikes(times, clusters, stim_times, n_clusters):
    edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
    trials = []
    for stim in np.asarray(stim_times, dtype=np.float64):
        edges = stim + edges_rel
        lo = np.searchsorted(times, edges[0], side="left")
        hi = np.searchsorted(times, edges[-1], side="left")
        st = times[lo:hi]
        sc = clusters[lo:hi]
        tb = np.searchsorted(edges, st, side="right") - 1
        valid = (tb >= 0) & (tb < N_BINS)
        flat = sc[valid] * N_BINS + tb[valid]
        count = np.bincount(flat, minlength=n_clusters * N_BINS)
        trials.append(count.reshape(n_clusters, N_BINS).astype(np.float32))
    return trials
```

iii. The AI states this matches the reference code's binning approach with 20-ms bins and stimulus-onset alignment. It explicitly chose to store counts rather than rates, saying this matches the Zhang reference cache which stores counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI retains ALL Kilosort clusters regardless of quality label. It does NOT apply the `label >= 1` quality filter. The quality labels are recorded as metadata but not used for filtering. Void neurons (outside brain) are also retained.

ii.
```python
label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
all_good.append(label >= 1)
...
# good_units is stored in info but not used for filtering
"n_good_units": int(np.sum(good_units)),
```

And after loading, Beryl mapping is applied to all clusters:
```python
beryl = atlas.acronym2acronym(np.asarray(acronyms), mapping="Beryl")
```

iii. The AI justifies retaining all clusters by citing the Zhang et al. reference code which calls `load_spiking_data(..., qc=None)`, meaning no quality filtering. The AI states: "Retain all clusters, matching the decoder paper and exact supplied cache path. Record quality in metadata statistics; do not silently substitute the stricter anatomical-analysis population."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset (`stimOn_times`). Spike bins are constructed as absolute edges `stim + edges_rel` where `edges_rel` runs from -0.5 to 1.5 s in 20-ms steps. All streams share the same session clock.

ii.
```python
edges = stim + edges_rel  # edges_rel = np.arange(N_BINS + 1) * BIN_SIZE + OFF_START
```

iii. IBL synchronizes all streams to a common clock, so alignment is just addition of relative offsets to the absolute stimulus onset time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20-ms bins (100 bins over 2 seconds from -0.5 to 1.5 s). No rebinning or smoothing is applied.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

iii. The AI states this matches the reference code's `binsize=0.02` and the 100-bin cache convention.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` in the trials table (the alignment event). The time input is a fixed vector of bin-end times relative to stimulus onset.

ii.
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
```

iii. The window and bin size come from the reference code parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data. The input is a fixed vector of 100 bin-end times: -0.48, -0.46, ..., 1.48, 1.50 s. These are the RIGHT edges of each 20-ms bin.

ii.
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
# Values: [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. The AI chose bin-end times to match the Zhang reference code's `get_behavior_per_interval`, which linearly interpolates continuous behavior at bin-end timestamps.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin-end times represent the right edges of the same 20-ms bins used for spike counting. Each time value corresponds to the end of the bin that the neural data column represents.

ii.
```python
inp = np.vstack([
    BIN_END_TIMES.astype(np.float32),
    np.full(N_BINS, block_num[trial_idx], dtype=np.float32),
])
```

iii. Both neural bins and time input use the same 100-bin grid from -0.5 to 1.5 s.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A block boundary is detected wherever `probabilityLeft` changes value.

ii.
```python
def block_trial_numbers(probability_left: np.ndarray) -> np.ndarray:
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.int32)
    for i in range(1, len(p)):
        out[i] = out[i - 1] + 1 if p[i] == p[i - 1] else 0
    return out
```

iii. The trials table carries no block ID, so blocks must be recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based ordinal is computed within each run of constant `probabilityLeft`. This is calculated on the original trial table BEFORE trial filtering, so filtered-out trials still advance the count. The value is then broadcast across all 100 time bins.

ii.
```python
block_num = block_trial_numbers(trials["probabilityLeft"].to_numpy())
...
np.full(N_BINS, block_num[trial_idx], dtype=np.float32),
```

iii. Computing before filtering preserves the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values -1, 0, or +1. The AI maps -1 to 0 (labeling it "left") and +1 to 1 (labeling it "right"), and drops trials with choice == 0. The AI states "IBL convention is choice -1 = left and +1 = right".

ii.
```python
choice = 0 if trials.iloc[trial_idx]["choice"] == -1 else 1
```

Output values:
```python
["left", "right"],
```

iii. The AI's CONVERSION_NOTES state: "IBL convention is choice -1 = left and +1 = right; map -1->0, +1->1 after excluding 0/missing." However, the standard IBL convention is +1 = left (CCW wheel turn) and -1 = right (CW wheel turn), which is the opposite of what the AI believes.

## 5-b. What processing is involved in computing `output` *Choice*?

i. A simple mapping: -1 -> 0, +1 -> 1, with 0 (no-response) excluded. The choice value is broadcast across all 100 time bins.

ii.
```python
choice = 0 if trials.iloc[trial_idx]["choice"] == -1 else 1
...
np.full(N_BINS, choice, dtype=np.int8),
```

iii. The mapping follows the instructions' specification of "left = 0, right = 1", but is applied to the wrong IBL convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
...
p = float(np.round(trials.iloc[trial_idx]["probabilityLeft"], 6))
```

iii. The three values are the block prior held constant within a block.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A direct categorical mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, with rounding to 6 decimal places for floating point safety. The value is broadcast across all 100 time bins.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
p = float(np.round(trials.iloc[trial_idx]["probabilityLeft"], 6))
out = np.vstack([
    ...
    np.full(N_BINS, prior_map[p], dtype=np.int8),
    ...
])
```

iii. Matches the instructions' explicit mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `SessionLoader.wheel`, which provides smoothed wheel velocity. The speed is the absolute value of the velocity.

ii.
```python
sess.load_wheel()
wheel_speed = np.abs(wheel["velocity"].to_numpy(dtype=float))
```

iii. The reference code derives wheel speed the same way.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` internally interpolates the raw wheel position to 1000 Hz and applies a Butterworth low-pass filter to compute velocity. The speed is `abs(velocity)`. This continuous trace is then linearly interpolated at the 100 bin-end timestamps for each trial. Finally, the interpolated values are discretized into 3 bins using session-wise tertiles (1/3 and 2/3 quantiles).

ii.
```python
wheel_speed = np.abs(wheel["velocity"].to_numpy(dtype=float))
wheel_interp, wheel_good = interpolate_trials(
    wheel["times"].to_numpy(), wheel_speed, stim
)
...
wheel_labels, wheel_q = discretize_tertiles(wheel_interp[selected])
```

```python
def discretize_tertiles(values: np.ndarray):
    finite = values[np.isfinite(values)]
    thresholds = np.quantile(finite, [1 / 3, 2 / 3])
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, thresholds
```

iii. The interpolation and filtering come from `SessionLoader`'s defaults. Session-wise tertiles avoid cross-session calibration differences.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-wise tertiles: the 1/3 and 2/3 quantiles of all finite interpolated wheel speed values from retained trials are used as thresholds. `np.digitize` with `right=False` assigns values to bins 0 (slow), 1 (medium), 2 (fast).

ii.
```python
def discretize_tertiles(values: np.ndarray):
    finite = values[np.isfinite(values)]
    thresholds = np.quantile(finite, [1 / 3, 2 / 3])
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, thresholds
```

iii. Equal-sized classes from session-specific percentiles, matching the reference approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is linearly interpolated at the same bin-end timestamps used to define the time input. These correspond to the right edges of the 100 neural bins.

ii.
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
...
result[i] = np.interp(target, local_t, local_v)
```

iii. Using the same time grid ensures temporal alignment between neural and behavioral data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The ROI motion energy from a side camera (`leftCamera.ROIMotionEnergy` preferred, with right camera fallback), loaded via `SessionLoader.load_motion_energy`. Specifically the `whiskerMotionEnergy` column.

ii.
```python
for view in ("left", "right"):
    try:
        sess.load_motion_energy(views=[view])
        key = f"{view}Camera"
        frame = sess.motion_energy[key]
        if "whiskerMotionEnergy" in frame and len(frame) > 1:
            whisker_view = view
            motion = frame
            break
    except Exception:
        continue
```

iii. Left camera preferred with right fallback, matching the reference code logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated at the 100 bin-end timestamps for each trial, then discretized into 3 bins using session-wise tertiles.

ii.
```python
whisker_interp, whisker_good = interpolate_trials(
    motion["times"].to_numpy(), motion["whiskerMotionEnergy"].to_numpy(), stim
)
...
whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: session-wise 1/3 and 2/3 quantile thresholds with `np.digitize(right=False)` producing labels 0 (low), 1 (medium), 2 (high).

ii.
```python
whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])
```

iii. Equal-sized classes from session-specific percentiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: linearly interpolated at bin-end timestamps of the same 100-bin grid used for neural data.

ii.
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
result[i] = np.interp(target, local_t, local_v)
```

iii. Same time grid as neural and wheel data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple safeguards: (1) Trials with non-finite required columns are excluded. (2) Sessions without whisker motion energy are skipped. (3) Probes with no clusters are skipped. (4) Trials where wheel or whisker trace doesn't cover the full window are excluded. (5) Trials with all-zero neural activity are removed. (6) Sessions with fewer than 2 valid trials are skipped. (7) Sessions where tertile thresholds collapse (identical values) raise errors.

ii.
```python
if not used_probes:
    raise RuntimeError("no pykilosort probe collections")
...
if len(selected) < 2:
    raise RuntimeError(f"only {len(selected)} valid trials")
...
neural_valid = np.asarray([np.any(x) for x in neural], dtype=bool)
if not np.all(neural_valid):
    selected = selected[neural_valid]
...
if not thresholds[0] < thresholds[1]:
    raise RuntimeError(f"collapsed tertile thresholds {thresholds.tolist()}")
```

iii. The AI documents 18 session exclusions: 14 missing whisker motion energy, 2 missing probabilityLeft, 1 with collapsed tertiles, 1 with zero passing trials.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (hundreds of MB per probe) and the `SessionLoader.load_wheel()` smoothing/interpolation. The full conversion took about 20.6 minutes.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. The cost is mainly file I/O for spike arrays.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes` function loops over trials to bin spikes, and `interpolate_trials` loops over trials for interpolation. The `block_trial_numbers` function uses a Python for-loop over all trials. These could potentially be vectorized.

ii.
```python
def bin_spikes(times, clusters, stim_times, n_clusters):
    ...
    for stim in np.asarray(stim_times, dtype=np.float64):
        ...
```

```python
def block_trial_numbers(probability_left: np.ndarray) -> np.ndarray:
    ...
    for i in range(1, len(p)):
        out[i] = out[i - 1] + 1 if p[i] == p[i - 1] else 0
    return out
```

iii. The AI processed sessions in parallel using ThreadPoolExecutor to compensate for per-trial loops.

## 10-c. What processing does the code repeat multiple times?

i. In the full parallel conversion, each worker thread builds its own ONE instance via `build_one()`, which re-reads and merges the release parquet tables. This is repeated for every worker thread.

ii.
```python
if not hasattr(worker_state, "one"):
    worker_state.one, _ = build_one(verbose=False)
worker_one = worker_state.one
```

iii. This was necessary because ONE is not thread-safe, but it means the index construction is repeated per worker.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and records `good_units` (label >= 1) counts and detailed session info (lab, date, probe names, tertile thresholds, retained trial indices) in metadata, which are not used by the downstream decoder. It also checks for and removes all-zero neural trials, which adds an extra pass over the data.

ii.
```python
"n_good_units": int(np.sum(good_units)),
...
"retained_trial_indices": selected.astype(int).tolist(),
```

iii. These serve documentation/debugging purposes but add overhead.
