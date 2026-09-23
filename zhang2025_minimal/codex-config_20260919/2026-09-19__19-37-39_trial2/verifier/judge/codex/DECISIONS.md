# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `/app/code/code_zhang2025/data/bwm_release.csv` as the master index of released insertions, optionally filters it with `/app/data/DATALIMIT_SUBSET.csv`, deduplicates `eid` values to define sessions, and then loads each session through `ONE`, `SessionLoader`, and `SpikeSortingLoader`. It does not use `one.search(...)` to discover sessions from the ONE cache.

ii. 
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
SUBSET_CSV = ROOT / "data" / "DATALIMIT_SUBSET.csv"
...
def load_release_table(max_sessions: int | None = None) -> pd.DataFrame:
    release = pd.read_csv(RELEASE_CSV)
    ...
    if SUBSET_CSV.exists():
        subset = pd.read_csv(SUBSET_CSV)
        ...
        release = release[release["eid"].astype(str).isin(allowed)]
    eids = _ordered_unique(release["eid"].astype(str))
```

```python
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    silent=True,
    cache_dir=str(CACHE_DIR),
)
...
session_loader = SessionLoader(one=one, eid=eid)
...
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
```

iii. In the trajectory the AI says it "resolved an important versioning detail in the cache" and therefore chose the authenticated ONE interface so revised ALF objects would resolve correctly. It also says the converter should be usable on both the full cache and the datalimit subset, and later reports that it examined all 459 source sessions from this release table.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the `subject` column of the release CSV for each session, then the final `subjects` list is built from the converted sessions and `subject_idx` is created by indexing into that sorted subject list.

ii. 
```python
info = {
    "eid": eid,
    "subject": str(eid_rows.iloc[0]["subject"]),
    ...
}
```

```python
subjects = sorted({info["subject"] for info in session_info})
subject_lookup = {name: i for i, name in enumerate(subjects)}
subject_idx = np.asarray(
    [subject_lookup[info["subject"]] for info in session_info], dtype=np.int64
)
```

iii. The trajectory does not give a long separate rationale here; it treats the release CSV metadata as the authoritative per-session metadata and later reports final subject counts from the converted sessions.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the release CSV. Each unique `eid` is processed once, while multiple rows with the same `eid` are used only to gather that session's probes.

ii. 
```python
eids = _ordered_unique(release["eid"].astype(str))
...
for number, eid in enumerate(eids, start=1):
    rows = release[release["eid"].astype(str) == eid]
    neural, decoder_input, decoder_output, regions, info = make_session(
        one, rows
    )
```

iii. In the trajectory the AI repeatedly reports progress as "`x`/459 sessions", showing that it treats each unique `eid` as one session while still using all matching probe rows from the release table.

## 1-d. How are the data split into trials?

i. Trials come from `SessionLoader(...).load_trials()`, and the session's trial table is treated as one row per source trial. After filtering, the retained trial indices are stored in `source_indices`, and all per-trial neural/input/output arrays are built by iterating over those retained indices.

ii. 
```python
session_loader = SessionLoader(one=one, eid=eid)
session_loader.load_trials()
trials = session_loader.trials.copy()
```

```python
source_mask = valid_trial_mask(trials)
source_indices = np.flatnonzero(source_mask)
...
for i in range(len(source_indices)):
    neural.append(np.ascontiguousarray(neural_3d[i]))
    ...
```

iii. The trajectory says the converter preserves "source trial indices" in metadata for reproducibility, which is consistent with treating the trials table as the native trial split and then subselecting rows by mask.

## 1-e. How are trials filtered based on quality controls?

i. The AI first applies `valid_trial_mask`, which requires finite values in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, and `goCue_times`; keeps only trials with reaction times between 0.08 and 2.0 s; excludes trials longer than 10 s from `goCue_times` to `feedback_times`; and drops no-choice trials. It then keeps only trials whose wheel and whisker traces pass the interpolation coverage check, and it drops sessions with fewer than two retained trials.

ii. 
```python
required = [
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
    "goCue_times",
]
...
mask &= reaction_time >= 0.08
mask &= reaction_time <= 2.0
mask &= duration <= 10.0
mask &= trials["choice"].to_numpy(dtype=float) != 0
```

```python
wheel, motion, behavior_good, motion_view = load_behaviors(
    one, eid, stimulus_times
)
source_indices = source_indices[behavior_good]
...
if len(source_indices) < 2:
    raise ValueError("fewer than two trials have complete behavior coverage")
```

iii. In the trajectory the AI says it is using the repository's "standard valid-trial mask", repeatedly describes the 80 ms to 2 s first-movement filter, and later says sessions without usable whisker traces are excluded rather than padded or extrapolated.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are built from spike times and spike cluster assignments loaded by `SpikeSortingLoader`. Cluster metadata is used to attach brain-region labels, but the per-trial neural matrices themselves come from `spikes["times"]` and `spikes["clusters"]`.

ii. 
```python
spikes, clusters, channels = loader.load_spike_sorting()
cluster_table = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False
).to_df()
```

```python
spike_times = np.asarray(spikes["times"], dtype=float)
spike_clusters = np.asarray(spikes["clusters"], dtype=np.int64)
```

iii. The trajectory emphasizes that the conversion keeps "all Kilosort clusters" and bins them in 20 ms bins, with Beryl region assignment recorded alongside them.

## 2-b. How is the `neural` data processed?

i. For each probe and each retained trial, the AI counts spikes from all sorted clusters into 100 non-overlapping 20 ms bins across the window `[-0.5, 1.5)` s around stimulus onset. Probe-specific count tensors are concatenated across probes, cast to `float32`, and stored per trial. The code does not divide by bin width, so the final neural arrays are spike counts stored as float values rather than firing rates in Hz.

ii. 
```python
counts = np.zeros(
    (len(stimulus_times), len(cluster_ids), N_TIME), dtype=np.uint16
)
for trial_i, stimulus_time in enumerate(stimulus_times):
    begin = stimulus_time + OFF_START_S
    end = stimulus_time + OFF_END_S
    ...
    bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
    ...
    hist = np.bincount(flat, minlength=len(cluster_ids) * N_TIME)
    counts[trial_i] = hist.reshape(len(cluster_ids), N_TIME)
```

```python
neural_3d = np.concatenate(probe_counts, axis=1).astype(np.float32)
...
neural.append(np.ascontiguousarray(neural_3d[i]))
```

iii. In the trajectory the AI repeatedly justifies 20 ms spike-count binning as following the methods code and paper text, and explicitly says it is retaining all Kilosort clusters rather than applying unit QC.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no cluster-quality threshold. It keeps every sorted cluster that appears in `spikes["clusters"]`, uses Beryl mapping only to name the cluster's region, and does not filter out `void`, `root`, or low-QC units.

ii. 
```python
cluster_ids = np.unique(spike_clusters)
...
# Searchsorted maps arbitrary (though usually consecutive) Kilosort IDs to
# rows in the output.  All sorted clusters are retained; no QC label filter.
counts = np.zeros(
    (len(stimulus_times), len(cluster_ids), N_TIME), dtype=np.uint16
)
```

```python
regions = cluster_table.iloc[cluster_ids]["acronym"].fillna("void").astype(str)
beryl = BrainRegions().acronym2acronym(regions.to_numpy(), mapping="Beryl")
```

iii. The trajectory explicitly says the converter is "retaining all Kilosort clusters exactly as the methods code does" and later describes the dataset as large because it keeps "all sorted clusters rather than only well-isolated units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to stimulus onset by defining each trial window as `stimulus_time + [-0.5, 1.5)` and binning spikes relative to that window start. In effect, each neural matrix covers a fixed 2 s stimulus-aligned window with stimulus onset 0.5 s after the start of the trial tensor.

ii. 
```python
begin = stimulus_time + OFF_START_S
end = stimulus_time + OFF_END_S
ib = np.searchsorted(spike_times, begin, side="left")
ie = np.searchsorted(spike_times, end, side="left")
...
bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
```

iii. The trajectory repeatedly states that all sessions are aligned to the same `-0.5` to `+1.5` s stimulus-centered window and later confirms that all successful sessions use that common alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms per bin. Each trial is represented by 100 bins spanning 2 s. No additional rebinning or temporal aggregation is applied beyond this one counting/interpolation grid.

ii. 
```python
BIN_SIZE_S = 0.020
OFF_START_S = -0.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

iii. The trajectory repeatedly cites the paper/repository convention of 100 bins at 20 ms each and treats that as a fixed design choice for both neural and behavioral streams.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI does not derive this input from a per-trial raw variable after loading the trials. Instead it creates a fixed 100-element relative-time vector from the chosen trial window and bin size, then uses `stimOn_times` only as the alignment anchor for the trial itself.

ii. 
```python
BIN_SIZE_S = 0.020
OFF_START_S = -0.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
TIME_FROM_STIMULUS = (
    OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
)
```

```python
stimulus_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
```

iii. The trajectory frames this input as part of the shared stimulus-aligned trial representation: once the trial window is defined around stimulus onset, the same relative time vector is reused for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes a simple left-edge time vector, `-0.5 + 0.02 * arange(100)`, and repeats it for every trial. There is no interpolation or recomputation from raw timestamps.

ii. 
```python
TIME_FROM_STIMULUS = (
    OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
)
...
decoder_input.append(
    np.vstack(
        [TIME_FROM_STIMULUS, np.full(N_TIME, trial_in_block[i], np.float32)]
    ).astype(np.float32, copy=False)
)
```

iii. The trajectory does not give a separate justification beyond the general choice to use one common 100-bin stimulus-aligned grid for every trial.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The same 100-bin time vector is stacked into every trial and paired with the per-trial neural tensor built on the same fixed `[-0.5, 1.5)` window. The code treats the time input as the label for those neural bins.

ii. 
```python
neural.append(np.ascontiguousarray(neural_3d[i]))
decoder_input.append(
    np.vstack(
        [TIME_FROM_STIMULUS, np.full(N_TIME, trial_in_block[i], np.float32)]
    ).astype(np.float32, copy=False)
)
```

iii. The trajectory repeatedly describes a single shared stimulus-aligned 100-bin representation for the converted dataset, so the intended alignment is that the time input and neural matrix refer to the same bins.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`, with contiguous runs of identical `probabilityLeft` values treated as blocks.

ii. 
```python
def trial_numbers_in_block(probability_left: np.ndarray) -> np.ndarray:
    out = np.ones(len(probability_left), dtype=np.float32)
    for i in range(1, len(out)):
        if probability_left[i] == probability_left[i - 1]:
            out[i] = out[i - 1] + 1.0
    return out
```

iii. The trajectory describes this input as preserving the "original trial ordinals" within contiguous `probabilityLeft` blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a one-indexed counter that resets whenever `probabilityLeft` changes. It computes the counter on the full session before trial filtering, then subsets it with `source_indices`, so excluded trials still advance the numbering.

ii. 
```python
def trial_numbers_in_block(probability_left: np.ndarray) -> np.ndarray:
    out = np.ones(len(probability_left), dtype=np.float32)
    for i in range(1, len(out)):
        if probability_left[i] == probability_left[i - 1]:
            out[i] = out[i - 1] + 1.0
    return out
```

```python
block_number = trial_numbers_in_block(
    trials["probabilityLeft"].to_numpy(dtype=float)
)
...
trial_in_block = block_number[source_indices]
```

iii. The trajectory explicitly states that the converter preserves "original trial ordinals" for this input, and the metadata records `"trial_number_indexing": "one-indexed within contiguous probabilityLeft blocks"`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived directly from the `choice` column of the trials table, after trial filtering.

ii. 
```python
choice_raw = trials["choice"].to_numpy(dtype=float)[source_indices]
if not np.all(np.isin(choice_raw, [-1.0, 1.0])):
    raise ValueError("choice contains values other than -1 and +1")
```

iii. The trajectory does not provide a separate rationale beyond requiring categorical integer outputs and excluding no-choice trials through the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI keeps only `-1` and `+1` trial values, then encodes `+1` as class `1` and `-1` as class `0` by evaluating `(choice_raw == 1.0)`. The resulting scalar is then repeated across all 100 time bins of the trial.

ii. 
```python
choice_raw = trials["choice"].to_numpy(dtype=float)[source_indices]
if not np.all(np.isin(choice_raw, [-1.0, 1.0])):
    raise ValueError("choice contains values other than -1 and +1")
choice = (choice_raw == 1.0).astype(np.int8)
```

```python
np.full(N_TIME, choice[i], np.int8)
```

iii. The trajectory does not state an explicit semantic justification for the left/right mapping; it mainly reports that the resulting decoder outputs are categorical and pass validation/smoke tests.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table, after filtering retained trials.

ii. 
```python
prior_raw = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
```

iii. The trajectory treats this as the block-probability variable specified by the task and repeatedly refers to it as a static per-trial output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2` using `np.isclose`, then repeats the resulting class label across the 100 bins of the trial.

ii. 
```python
prior = np.full(len(prior_raw), -1, dtype=np.int8)
for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(prior_raw, value)] = label
if np.any(prior < 0):
    raise ValueError(f"unexpected probabilityLeft values {np.unique(prior_raw)}")
```

```python
np.full(N_TIME, prior[i], np.int8)
```

iii. The trajectory does not give separate argumentation here; it follows the decoder-task recoding requested in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel stream loaded by `SessionLoader.load_wheel()`. The code uses the loader's `velocity` column and takes its absolute value, so the underlying raw variables are the wheel timestamps and wheel position that `SessionLoader` turns into velocity.

ii. 
```python
loader = SessionLoader(one=one, eid=eid)
loader.load_wheel()
wheel_times = loader.wheel["times"].to_numpy(dtype=float)
wheel_speed = np.abs(loader.wheel["velocity"].to_numpy(dtype=float))
```

iii. The trajectory says the converter is using repository-defined wheel behavior and repeatedly refers to dynamic outputs being derived from the standard IBL loaders.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI takes the absolute wheel velocity from `SessionLoader`, resamples it trial by trial onto a 100-sample stimulus-aligned grid using linear interpolation at the right edge of each 20 ms bin, and rejects trials lacking full-window coverage. After the final retained-trial mask, it discretizes the entire session's wheel values into tertiles.

ii. 
```python
wheel_speed = np.abs(loader.wheel["velocity"].to_numpy(dtype=float))
wheel, wheel_good = interpolate_trials(wheel_times, wheel_speed, stimulus_times)
```

```python
relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
...
query = stimulus_time + relative_endpoints
interp = interp1d(
    local_t, local_v, kind="linear", fill_value="extrapolate"
)(query)
```

```python
wheel_class, wheel_thresholds = discretize_tertiles(wheel)
```

iii. In the trajectory the AI says it is matching the repository's endpoint interpolation behavior, and it justifies session-wise tertiles as a way to keep low/medium/high classes comparable across sessions despite scale differences.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded into three categories using session-wide tertiles computed over all retained wheel-speed values from all retained trials and all time bins in that session. `np.digitize` then turns those thresholds into labels `0`, `1`, and `2`.

ii. 
```python
def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    thresholds = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    ...
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, [float(thresholds[0]), float(thresholds[1])]
```

```python
wheel_class, wheel_thresholds = discretize_tertiles(wheel)
```

iii. The trajectory explicitly states that the dynamic outputs use session-wise tertiles and later confirms the resulting class balance is roughly one-third per class.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel signal is aligned trial by trial to the same `stimOn_times` used for neural binning. For each retained trial it is interpolated onto 100 stimulus-aligned sample points that correspond to the right edge of each 20 ms bin in the `[-0.5, 1.5)` s window.

ii. 
```python
begin = stimulus_time + OFF_START_S
end = stimulus_time + OFF_END_S
...
query = stimulus_time + relative_endpoints
interp = interp1d(
    local_t, local_v, kind="linear", fill_value="extrapolate"
)(query)
```

iii. The trajectory explicitly says the converter reproduces "20 ms endpoint interpolation for wheel and whisker signals" after stimulus alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the motion-energy stream loaded by `SessionLoader.load_motion_energy(...)`: specifically the per-frame `whiskerMotionEnergy` values and the associated camera `times`. The AI prefers the left camera and falls back to the right camera.

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
```

iii. The trajectory says revised motion-energy objects are important, that the converter prefers left camera data with right as fallback, and that sessions without a usable whisker stream are excluded.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released `whiskerMotionEnergy` trace without additional filtering or normalization, interpolates it onto the same 100 stimulus-aligned sample points used for wheel speed, requires full-window coverage, and then discretizes the retained session-wide values into tertiles.

ii. 
```python
motion, motion_good = interpolate_trials(
    motion_df["times"].to_numpy(dtype=float),
    motion_df["whiskerMotionEnergy"].to_numpy(dtype=float),
    stimulus_times,
)
...
motion_class, motion_thresholds = discretize_tertiles(motion)
```

iii. In the trajectory the AI says revised camera objects are used where present, and repeatedly explains that sessions with missing or incomplete whisker traces are excluded instead of being padded across gaps.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: by computing the 1/3 and 2/3 quantiles across all retained whisker-motion values in the session and applying `np.digitize` to create classes `0`, `1`, and `2`.

ii. 
```python
def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    thresholds = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    ...
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
```

```python
motion_class, motion_thresholds = discretize_tertiles(motion)
```

iii. The trajectory says both dynamic outputs are discretized with session-wise tertiles and later reports the classes are balanced by construction.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same `stimOn_times` and 2 s stimulus window used for neural data. For each retained trial, the motion-energy trace is interpolated onto the right edge of each 20 ms bin in the shared 100-sample grid.

ii. 
```python
query = stimulus_time + relative_endpoints
interp = interp1d(
    local_t, local_v, kind="linear", fill_value="extrapolate"
)(query)
```

iii. The trajectory explicitly says the converter applies the same stimulus-aligned 20 ms endpoint interpolation to both wheel and whisker outputs.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly drops or skips unusable data rather than repairing it. Missing required trial-table columns raise an error; trials without required finite events or without complete wheel/whisker coverage are dropped; sessions with fewer than two surviving trials are skipped; sessions with no usable whisker trace are skipped; and session skip reasons are recorded in metadata. It does not impute missing values.

ii. 
```python
missing = [name for name in required if name not in trials]
if missing:
    raise ValueError(f"trial table is missing columns {missing}")
```

```python
if len(source_indices) < 2:
    raise ValueError("fewer than two trials have complete behavior coverage")
...
raise ValueError(f"no usable whisker motion-energy trace ({last_error})")
```

```python
except Exception as exc:
    skipped_sessions.append({"eid": eid, "reason": repr(exc)})
```

iii. The trajectory repeatedly says missing whisker-motion sessions are excluded and documented rather than padded, and the final report states that 15 sessions were excluded for unavailable whisker motion energy.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming work in the AI code is loading and binning large spike-sorting outputs session by session, plus the final serialization and validator pass over the very large pickle. Dense multi-probe recordings dominate runtime because they have many spikes and clusters.

ii. 
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

```python
for trial_i, stimulus_time in enumerate(stimulus_times):
    ...
    hist = np.bincount(flat, minlength=len(cluster_ids) * N_TIME)
```

```python
with temporary.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly says runtime "depends mainly on spike volume rather than trial count," mentions particularly slow dense two-probe sessions, and later calls final serialization and full validation the expensive final checks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several explicit Python loops could have been vectorized: the one-indexed block counter, the per-trial interpolation loop in `interpolate_trials`, the per-trial spike-binning loop in `bin_probe_spikes`, and the final per-trial assembly loop that appends neural/input/output arrays.

ii. 
```python
for i in range(1, len(out)):
    if probability_left[i] == probability_left[i - 1]:
        out[i] = out[i - 1] + 1.0
```

```python
for i, stimulus_time in enumerate(stimulus_times):
    ...
```

```python
for trial_i, stimulus_time in enumerate(stimulus_times):
    ...
```

```python
for i in range(len(source_indices)):
    neural.append(np.ascontiguousarray(neural_3d[i]))
    ...
```

iii. The trajectory does not give a separate optimization rationale here; these opportunities are visible directly from the code structure.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several session-local operations: it re-runs per-trial `searchsorted`/slicing/interpolation separately for wheel and motion traces; it bins spikes in a separate Python loop for every trial of every probe; it constructs per-trial arrays in another loop after already holding full session tensors; and it instantiates `BrainRegions()` inside each probe-binning call.

ii. 
```python
wheel, wheel_good = interpolate_trials(wheel_times, wheel_speed, stimulus_times)
...
motion, motion_good = interpolate_trials(
    motion_df["times"].to_numpy(dtype=float),
    motion_df["whiskerMotionEnergy"].to_numpy(dtype=float),
    stimulus_times,
)
```

```python
for trial_i, stimulus_time in enumerate(stimulus_times):
    ...
```

```python
beryl = BrainRegions().acronym2acronym(regions.to_numpy(), mapping="Beryl")
```

iii. The trajectory does not defend this repetition explicitly; it focuses on correctness and reproducibility rather than implementation efficiency.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The core neural/input/output pipeline does not do much obviously throwaway processing, but it does spend work on metadata bookkeeping that the downstream decoder does not use, such as storing retained trial indices, per-session tertile thresholds, probe names, dates, session numbers, skip reasons, and conversion timing.

ii. 
```python
info = {
    "eid": eid,
    "subject": str(eid_rows.iloc[0]["subject"]),
    "date": str(eid_rows.iloc[0]["date"]),
    "session_number": int(eid_rows.iloc[0]["session_number"]),
    "probe_names": [str(x) for x in eid_rows["probe_name"]],
    ...
    "retained_trial_indices": source_indices.astype(int).tolist(),
    "motion_energy_camera": motion_view,
    "wheel_speed_tertile_thresholds": wheel_thresholds,
    "whisker_motion_energy_tertile_thresholds": motion_thresholds,
}
```

```python
"metadata": {
    ...
    "skipped_sessions": skipped_sessions,
    "session_info": session_info,
    "conversion_seconds": float(time.time() - start),
},
```

iii. The trajectory explicitly justifies this extra metadata as reproducibility and audit information, for example by saying skip reasons and source trial indices are recorded rather than silently discarded.
