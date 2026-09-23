# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using the ONE API with the local cache (`one_cache`). Instead of using `one.search()` to discover sessions, it reads the frozen release table (`bwm_release.csv`) from the reference code's data directory to enumerate all session EIDs and probe insertions. If `DATALIMIT_SUBSET.csv` exists, sessions are restricted to those EIDs. For each session, `SessionLoader` loads trials, wheel, and motion energy, while `SpikeSortingLoader` loads spike sorting per probe.

ii.
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
release = pd.read_csv(RELEASE_CSV)
# ...
if SUBSET_CSV.exists():
    subset = pd.read_csv(SUBSET_CSV)
    # restrict to allowed EIDs
    release = release[release["eid"].astype(str).isin(allowed)]
# ...
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    silent=True,
    cache_dir=str(CACHE_DIR),
)
```

iii. The AI chose to use the release CSV because it provides the frozen list of session/probe pairs from the paper. The agent noted: "The supplied cache is the full Brain-Wide Map release... The converter will use the authenticated offline ONE cache interface (without downloading source data), matching the repository's loaders."

## 1-b. How are the data split into subjects?

i. Subject names come from the release table's `subject` column. Each unique EID maps to a subject. At assembly, subjects are sorted alphabetically and `subject_idx` maps each session to its subject index.

ii.
```python
subjects = sorted({info["subject"] for info in session_info})
subject_lookup = {name: i for i, name in enumerate(subjects)}
subject_idx = np.asarray(
    [subject_lookup[info["subject"]] for info in session_info], dtype=np.int64
)
```

iii. The release table directly provides the subject for each session, so no parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Each unique EID in the release table is one session. The release table may have multiple rows per EID (one per probe insertion), which are grouped together as one session.

ii.
```python
eids = _ordered_unique(release["eid"].astype(str))
for number, eid in enumerate(eids, start=1):
    rows = release[release["eid"].astype(str) == eid]
    neural, decoder_input, decoder_output, regions, info = make_session(one, rows)
```

iii. Sessions are the natural unit of the release; the EID uniquely identifies a session.

## 1-d. How are the data split into trials?

i. The trials table from `SessionLoader.load_trials()` has one row per trial. Each trial is processed individually after applying the validity mask.

ii.
```python
session_loader = SessionLoader(one=one, eid=eid)
session_loader.load_trials()
trials = session_loader.trials.copy()
```

iii. The trials table is already organized one row per trial; no further splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI implements a trial mask based on the reference code's `load_trials_and_mask` defaults. Trials are filtered by: (1) all required columns must have finite values (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, `goCue_times`); (2) reaction time (firstMovement_times - stimOn_times) must be between 0.08 and 2.0 s; (3) trial duration (feedback_times - goCue_times) must be <= 10 s; (4) choice must be nonzero (no-go trials dropped); (5) wheel and whisker motion energy must have complete coverage over the trial window.

ii.
```python
def valid_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times", "choice", "feedback_times", "probabilityLeft",
        "firstMovement_times", "feedbackType", "goCue_times",
    ]
    vals = trials[required].to_numpy(dtype=float)
    mask = np.all(np.isfinite(vals), axis=1)
    reaction_time = trials["firstMovement_times"].to_numpy(dtype=float) - trials["stimOn_times"].to_numpy(dtype=float)
    duration = trials["feedback_times"].to_numpy(dtype=float) - trials["goCue_times"].to_numpy(dtype=float)
    mask &= reaction_time >= 0.08
    mask &= reaction_time <= 2.0
    mask &= duration <= 10.0
    mask &= trials["choice"].to_numpy(dtype=float) != 0
    return mask
```

iii. The agent stated it would implement "the standard valid-trial mask (including 80 ms-2 s first-movement latency)" and noted that `prepare_data` passes `max_trial_len=10.0` to `load_trials_and_mask`. The code docstring states: "In addition to the default required fields and 0.08--2.00 s reaction-time interval, prepare_data passes max_trial_len=10 s."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times` and `spikes.clusters` loaded via `SpikeSortingLoader`. The cluster table provides anatomical regions (acronyms mapped to Beryl).

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
spikes, clusters, channels = loader.load_spike_sorting()
cluster_table = SpikeSortingLoader.merge_clusters(spikes, clusters, channels, compute_metrics=False).to_df()
spike_times = np.asarray(spikes["times"], dtype=float)
spike_clusters = np.asarray(spikes["clusters"], dtype=np.int64)
```

iii. Spike times and cluster assignments are the fundamental data for building spike count matrices.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into non-overlapping 20 ms bins over the [-0.5, 1.5) s trial window, giving 100 time bins. Spike counts are stored as float32 (raw counts, not converted to firing rates). When a session has multiple probes, their clusters are concatenated into one population.

ii.
```python
bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
local_cluster = np.searchsorted(cluster_ids, spike_clusters[ib:ie])
in_range = (bin_idx >= 0) & (bin_idx < N_TIME)
flat = local_cluster[in_range] * N_TIME + bin_idx[in_range]
hist = np.bincount(flat, minlength=len(cluster_ids) * N_TIME)
counts[trial_i] = hist.reshape(len(cluster_ids), N_TIME)
# ...
neural_3d = np.concatenate(probe_counts, axis=1).astype(np.float32)
```

iii. The agent followed the reference code's binning parameters: "split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps ALL sorted clusters with no QC label filtering. Void regions are also retained (though mapped to Beryl). The metadata explicitly states: "all sorted clusters (no cluster-QC threshold)".

ii.
```python
cluster_ids = np.unique(spike_clusters)
# No filtering on cluster quality labels
counts = np.zeros((len(stimulus_times), len(cluster_ids), N_TIME), dtype=np.uint16)
# ...
regions = cluster_table.iloc[cluster_ids]["acronym"].fillna("void").astype(str)
beryl = BrainRegions().acronym2acronym(regions.to_numpy(), mapping="Beryl")
```

iii. The agent justified this by noting: "The converter is retaining all Kilosort clusters exactly as the methods code does" and observed that the reference code's `prepare_data` calls `load_spiking_data` without a `qc` argument, so it defaults to None (all clusters). The agent stated: "The resulting dataset is intentionally large because the reference method keeps all sorted clusters rather than only well-isolated units; this matches both the paper text and its released preprocessing code."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the window [-0.5, 1.5) s around stimulus onset is used. Spike times within this absolute window are binned relative to the window start.

ii.
```python
stimulus_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
# ...
begin = stimulus_time + OFF_START_S
end = stimulus_time + OFF_END_S
ib = np.searchsorted(spike_times, begin, side="left")
ie = np.searchsorted(spike_times, end, side="left")
local_time = spike_times[ib:ie]
bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
```

iii. All data streams share the same session clock, so alignment is achieved by subtracting the stimulus onset time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins over the 2 s window. No rebinning or smoothing is applied.

ii.
```python
BIN_SIZE_S = 0.020
OFF_START_S = -0.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))  # 100
```

iii. The 20 ms bin size matches the reference code's `'binsize': 0.02` parameter.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` in the trials table, which defines the alignment event. The time values are the bin edges of the 100-bin grid.

ii.
```python
TIME_FROM_STIMULUS = (
    OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
)
```

iii. The time input is defined by the binning grid used for neural data alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as the left edges of each bin: -0.50, -0.48, -0.46, ..., 1.48. This is a fixed array replicated for every trial.

ii.
```python
TIME_FROM_STIMULUS = (
    OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
)
# produces: [-0.5, -0.48, -0.46, ..., 1.48]
```

iii. No explicit discussion of the choice of left edges vs bin centers in the agent trajectory.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same 100-element grid as the neural bins. Since neural bins are defined by their left edges (spike_time - begin), the time input at left edges is aligned bin-for-bin with the neural data.

ii.
```python
decoder_input.append(
    np.vstack(
        [TIME_FROM_STIMULUS, np.full(N_TIME, trial_in_block[i], np.float32)]
    ).astype(np.float32, copy=False)
)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table, which is constant within a block. A change in its value marks a new block boundary.

ii.
```python
def trial_numbers_in_block(probability_left: np.ndarray) -> np.ndarray:
    out = np.ones(len(probability_left), dtype=np.float32)
    for i in range(1, len(out)):
        if probability_left[i] == probability_left[i - 1]:
            out[i] = out[i - 1] + 1.0
    return out
```

iii. The trials table carries no explicit block identifier, so blocks must be inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number is one-indexed within each contiguous block of constant `probabilityLeft`. The first trial in each block is 1, the second is 2, etc. This numbering is computed before trial filtering, so dropped trials still advance the count.

ii.
```python
block_number = trial_numbers_in_block(
    trials["probabilityLeft"].to_numpy(dtype=float)
)
# ...
trial_in_block = block_number[source_indices]
```

iii. The metadata records: "trial_number_indexing: one-indexed within contiguous probabilityLeft blocks". No explicit reasoning in the trajectory for choosing one-indexed over zero-indexed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

ii.
```python
choice_raw = trials["choice"].to_numpy(dtype=float)[source_indices]
if not np.all(np.isin(choice_raw, [-1.0, 1.0])):
    raise ValueError("choice contains values other than -1 and +1")
choice = (choice_raw == 1.0).astype(np.int8)
```

iii. No-response trials (choice=0) are excluded by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps the raw choice values as: `(choice_raw == 1.0)`, which gives +1 (left in IBL) → 1 and -1 (right in IBL) → 0. The output_values are listed as `["left", "right"]`, meaning index 0 = "left" and index 1 = "right".

ii.
```python
choice = (choice_raw == 1.0).astype(np.int8)
# IBL +1 (left) → 1, IBL -1 (right) → 0
```

iii. No explicit discussion in the trajectory about the mapping direction. The instructions specify "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior_raw = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
prior = np.full(len(prior_raw), -1, dtype=np.int8)
for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(prior_raw, value)] = label
```

iii. The mapping 0.2→0, 0.5→1, 0.8→2 directly follows the task specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three probability values are mapped to categorical labels 0, 1, 2 using `np.isclose` for floating-point comparison. Any unexpected values would raise an error.

ii.
```python
if np.any(prior < 0):
    raise ValueError(f"unexpected probabilityLeft values {np.unique(prior_raw)}")
```

iii. Straightforward implementation of the task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From the wheel position and timestamps loaded by `SessionLoader.load_wheel()`, which internally interpolates position to 1000 Hz and computes velocity with a Butterworth low-pass filter. The speed is the absolute value of velocity.

ii.
```python
loader = SessionLoader(one=one, eid=eid)
loader.load_wheel()
wheel_times = loader.wheel["times"].to_numpy(dtype=float)
wheel_speed = np.abs(loader.wheel["velocity"].to_numpy(dtype=float))
```

iii. The reference code derives wheel speed the same way, using `np.abs` of the velocity from `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel speed trace is interpolated onto the trial time grid using `scipy.interpolate.interp1d` with linear interpolation and extrapolation. The query points are the right bin edges (not bin centers). Then the trace is discretized into three categories using within-session tertiles.

ii.
```python
relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
# produces: [-0.48, -0.46, ..., 1.50] (right bin edges)
interp = interp1d(local_t, local_v, kind="linear", fill_value="extrapolate")(query)
# ...
thresholds = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
labels = np.digitize(values, thresholds, right=False).astype(np.int8)
```

iii. The agent stated: "interp1d with extrapolation is the exact operation used by the repository. The final query is the interval end, typically just beyond the final camera/wheel sample selected with side='left'." And: "Repository behavior bins are sampled at each bin's right edge."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Within-session tertiles: the 1/3 and 2/3 quantiles of all wheel speed values across all retained trials and time bins in the session are used as thresholds. `np.digitize` maps values to categories 0 (low), 1 (medium), 2 (high).

ii.
```python
def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    thresholds = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, [float(thresholds[0]), float(thresholds[1])]
```

iii. The agent stated: "Dynamic outputs will be session-wise tertiles, which keeps 'low/medium/high' comparable despite camera-specific motion-energy scales."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is sampled at the right edges of the neural bins, while neural data is binned from left edges. This introduces a half-bin (10 ms) offset between the behavioral and neural time points.

ii.
```python
# Behavioral query points (right edges):
relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
# Neural bin assignment (left edges):
bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
```

iii. The agent justified using right edges by noting the reference code's `get_behavior_per_interval` uses `np.linspace(interval_begs + binsize, interval_ends, n_bins)`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From the motion energy of a side camera (`leftCamera.ROIMotionEnergy` or `rightCamera.ROIMotionEnergy`) with its frame times. The left camera is preferred, with right as fallback.

ii.
```python
for view in ("left", "right"):
    try:
        loader.load_motion_energy(views=[view])
        key = f"{view}Camera"
        motion_df = loader.motion_energy[key]
        motion, motion_good = interpolate_trials(
            motion_df["times"].to_numpy(dtype=float),
            motion_df["whiskerMotionEnergy"].to_numpy(dtype=float),
            stimulus_times,
        )
        if np.any(motion_good):
            return wheel, motion, wheel_good & motion_good, view
    except Exception as exc:
        last_error = exc
```

iii. The agent noted: "The reference uses left whenever it can be loaded, with right as a fallback."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Same as wheel speed: the motion energy trace is interpolated onto right bin edges using `interp1d` with extrapolation, then discretized into three categories using within-session tertiles.

ii.
```python
motion, motion_good = interpolate_trials(
    motion_df["times"].to_numpy(dtype=float),
    motion_df["whiskerMotionEnergy"].to_numpy(dtype=float),
    stimulus_times,
)
# ...
motion_class, motion_thresholds = discretize_tertiles(motion)
```

iii. Same justification as wheel speed processing.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Within-session tertiles, identical to wheel speed: 1/3 and 2/3 quantiles as thresholds, `np.digitize` to map to 0/1/2.

ii.
```python
motion_class, motion_thresholds = discretize_tertiles(motion)
```

iii. Same tertile approach for consistency across dynamic outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: sampled at right bin edges, introducing the same half-bin offset relative to neural data.

ii.
```python
# Same interpolate_trials function used for both wheel and whisker
relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
```

iii. Consistent with the agent's interpretation that the repository code samples at right edges.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled at multiple levels: (1) trials with NaN in required columns are excluded by `valid_trial_mask`; (2) trials without complete behavioral coverage are excluded by the `behavior_good` mask from `interpolate_trials`; (3) sessions with no usable whisker motion energy from either camera are skipped entirely; (4) sessions with fewer than 2 valid trials or no sorted clusters raise exceptions and are skipped; (5) all skipped sessions are documented in metadata with specific reasons.

ii.
```python
# Trial-level: finite value check
mask = np.all(np.isfinite(vals), axis=1)

# Behavioral coverage check
if abs(begin - local_t[0]) > BIN_SIZE_S:
    continue
if abs(end - local_t[-1]) > BIN_SIZE_S:
    continue

# Session-level: skip and document
except Exception as exc:
    skipped_sessions.append({"eid": eid, "reason": repr(exc)})
```

iii. The agent stated: "sessions lacking a usable required stream will be skipped and documented rather than silently filled."

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk is the most time-consuming step, as each probe's spike arrays can be hundreds of megabytes. The agent noted processing time "depends mainly on spike volume rather than trial count." Additionally, keeping all sorted clusters (no QC filter) results in much larger data volumes.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. The agent observed: "Processing time depends mainly on spike volume rather than trial count, so some dense two-probe recordings take longer."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops could be vectorized: (1) the spike binning loop in `bin_probe_spikes` iterates over each trial; (2) the behavioral interpolation loop in `interpolate_trials` iterates over each trial. Both could potentially be written as single vectorized operations.

ii.
```python
# Spike binning loop
for trial_i, stimulus_time in enumerate(stimulus_times):
    # ...per-trial binning...

# Behavioral interpolation loop
for i, stimulus_time in enumerate(stimulus_times):
    # ...per-trial interpolation...
```

iii. No explicit discussion of vectorization opportunities in the agent trajectory.

## 10-c. What processing does the code repeat multiple times?

i. `SessionLoader` is instantiated twice per session: once in `make_session` to load trials, and once in `load_behaviors` to load wheel and motion energy. The trials are loaded in the first call but the wheel/motion energy require a second instantiation.

ii.
```python
# In make_session:
session_loader = SessionLoader(one=one, eid=eid)
session_loader.load_trials()

# In load_behaviors (called from make_session):
loader = SessionLoader(one=one, eid=eid)
loader.load_wheel()
```

iii. No explicit discussion of this redundancy in the agent trajectory.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No obvious unnecessary processing. The code computes only what is needed for the target data structure, though keeping all sorted clusters (no QC filtering) means substantially more neural data is processed and stored than a QC-filtered approach would require.

ii. N/A

iii. N/A
