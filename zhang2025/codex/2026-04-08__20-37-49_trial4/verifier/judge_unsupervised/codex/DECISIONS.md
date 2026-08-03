# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the full Brain-Wide Map release from `bwm_release.csv`, enumerates unique sessions from that table, then loads each session's ALF files directly from local `data/one_cache` with fallback ONE downloads into `cache/one_cache`. Within each session it separately loads trials, wheel, whisker, and probe ephys files rather than calling the reference helper stack.

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

def ensure_dataset_path_any(one: ONE, row, relative_globs: list[str], dataset_name: str, collection: str):
    for relative_glob in relative_globs:
        path = locate_dataset(row, relative_glob)
        if path is not None:
            return path
    one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
```

```python
one = build_one()
brain_regions = BrainRegions()
session_rows, probe_rows = load_release_sessions()
print(f"Release sessions listed: {len(session_rows)}", flush=True)
```

iii. `CONVERSION_NOTES.md` says the agent used the 459-session release list as canonical input and implemented direct ALF loading because the reference helper stack had unavailable optional dependencies. The trajectory also shows it identified `prepare_data` in the reference code, then chose to mirror that logic manually.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column in the release CSV. During dataset assembly, the agent creates a unique subject list in first-seen session order and stores a `subject_idx` entry per retained session.

ii. 
```python
session_rows = (
    bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
    .reset_index(drop=True)
)
```

```python
subjects = []
subject_to_idx = {}
...
for rec in session_records:
    if rec.subject not in subject_to_idx:
        subject_to_idx[rec.subject] = len(subjects)
        subjects.append(rec.subject)
    subject_idx.append(subject_to_idx[rec.subject])
```

iii. The notes describe session subject labels as coming from release metadata and being indexed into `subject_idx`. There is no extra subject-level processing beyond retaining only subjects with at least one retained session.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from `bwm_release.csv`. For each session, the agent reconstructs the expected cache path from lab/subject/date/session number, loads all probes listed for that `eid`, and merges those probes into one session-level representation.

ii. 
```python
def session_rel_path(row) -> Path:
    return Path(row.lab) / "Subjects" / row.subject / str(row.date) / f"{int(row.session_number):03d}"
```

```python
probe_rows = {
    eid: grp[["pid", "probe_name"]].reset_index(drop=True)
    for eid, grp in bwm.groupby("eid", sort=False)
}
```

```python
for probe_row in probe_df.itertuples(index=False):
    spikes, clusters = load_probe_data(one, row, probe_row.probe_name, brain_regions)
    ...
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. The notes explicitly say the 459-session release list was used as input and that probes are merged within a session, matching the reference project's session-level caching workflow.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. The script bins spikes and behavior for every row in the trials table, then keeps only the row indices passing the combined trial, wheel, and whisker masks. Each retained row becomes one trial in `neural`, `input`, and `output`.

ii. 
```python
def load_trials_table(one: ONE, row):
    path = ensure_dataset_path(one, row, "alf/#*/_ibl_trials.table.pqt", "_ibl_trials.table.pqt", "alf")
    trials_df = pd.read_parquet(path)
    ...
    return trials_df
```

```python
trial_mask = build_trial_mask(trials_df)
...
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
...
for idx in valid_idx:
    ...
    neural_trials.append(neural_trial)
```

iii. The notes say the agent mirrored the reference trialized processing: load the trial table, align all streams to `stimOn_times`, and then keep only trials jointly valid across required streams.

## 1-e. How are trials filtered based on quality controls?

i. Base trial filtering excludes trials with missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`; reaction times outside 0.08 to 2.0 s; trial lengths over 10 s; and no-choice trials (`choice == 0`). After that, trials are further filtered out if wheel or whisker traces do not cover the whole aligned window.

ii. 
```python
def build_trial_mask(
    trials_df: pd.DataFrame,
    min_rt=0.08,
    max_rt=2.0,
    nan_exclude="default",
    min_trial_len=None,
    max_trial_len=10.0,
    exclude_unbiased=False,
    exclude_nochoice=True,
):
    ...
    if exclude_nochoice:
        query += " | (choice == 0)"
    ...
    return ~trials_df.eval(query)
```

```python
wheel_values, wheel_mask = get_behavior_per_interval(wheel["times"], wheel["values"], trials_df, allow_nans=False)
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
```

iii. The notes say the agent followed the reference reaction-time and trial-length curation, then added the shared wheel/whisker validity requirement because those are decoder outputs here. The notes also justify excluding no-choice trials because the requested choice target is binary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from probe-level `spikes.times.npy` and `spikes.clusters.npy`, plus `clusters.channels.npy`, `clusters.depths.npy`, `clusters.metrics.pqt`, and `channels.brainLocationIds_ccf_2017.npy` to filter clusters and assign brain regions.

ii. 
```python
times_path = ensure_dataset_path(one, row, f"{collection}/#*/spikes.times.npy", "spikes.times.npy", collection)
clu_path = ensure_dataset_path(
    one, row, f"{collection}/#*/spikes.clusters.npy", "spikes.clusters.npy", collection
)
cluster_channels_path = ensure_dataset_path(
    one, row, f"{collection}/#*/clusters.channels.npy", "clusters.channels.npy", collection
)
cluster_depths_path = ensure_dataset_path(
    one, row, f"{collection}/#*/clusters.depths.npy", "clusters.depths.npy", collection
)
cluster_metrics_path = ensure_dataset_path(
    one, row, f"{collection}/#*/clusters.metrics.pqt", "clusters.metrics.pqt", collection
)
channel_regions_path = ensure_dataset_path(
    one,
    row,
    f"{collection}/#*/channels.brainLocationIds_ccf_2017.npy",
    "channels.brainLocationIds_ccf_2017.npy",
    collection,
)
```

iii. The notes map `neural` to good-unit `spikes.times` and `spikes.clusters` from all probes, with region labels coming from channel brain-location IDs via Beryl mapping.

## 2-b. How is the `neural` data processed?

i. The agent sorts out good clusters per probe, merges probes within a session, then bins spikes into 20 ms counts over a stimulus-aligned 2 s window. Trial arrays are transposed to `(n_neurons, 100)` and compacted to `uint8`.

ii. 
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)
...
neural_trial = compact_neural_trial(binned_spikes[idx].T)
```

```python
def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    n_bins = int(np.ceil(interval_len / binsize))
    ...
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
```

iii. The notes say the agent mirrored `merge_probes` and `bin_spiking_data` from the reference code, but implemented them directly on ALF files. It also notes the `uint8` compaction was a storage optimization added for trainability.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are filtered to `label >= 1` from `clusters.metrics.pqt`, which the agent interpreted as well-isolated units. Probes with zero such units are skipped, and sessions with no remaining probes are dropped.

ii. 
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
...
if good_cluster_ids.size == 0:
    return None, None
```

```python
for probe_row in probe_df.itertuples(index=False):
    spikes, clusters = load_probe_data(one, row, probe_row.probe_name, brain_regions)
    if spikes is None or clusters is None:
        print(f"  skipping probe {probe_row.probe_name}: no well-isolated clusters", flush=True)
        continue
...
if not spikes_list:
    raise RuntimeError(f"No probes with well-isolated clusters for session {row.eid}")
```

iii. The notes justify this as matching the paper’s 75,708 “well-isolated neurons,” and explicitly record that this was a deliberate change from the executable reference path, which exposed QC labels without forcing them in `prepare_data`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns each neural trial to `stimOn_times`, using a fixed window from -0.5 s to +1.5 s around stimulus onset.

ii. 
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02
```

```python
intervals = np.vstack(
    [
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
    ]
).T
```

iii. The notes repeatedly justify this with the reference cache settings from `0_data_caching.py` and the task instruction “Temporally align based on stimulus onset.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins (`BIN_SIZE = 0.02`) and 100 bins per trial over a 2 s window. The neural stream is not rebinned from another trialized representation; it is binned directly from spike times into those 20 ms bins.

ii. 
```python
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
```

```python
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
binned_spikes[interval_idx, idxs_tmp, :] = binned_tmp[:, :n_bins]
```

iii. The notes say 20 ms was chosen to match the reference cache and cited the papers’ “20-ms bins.” No additional temporal downsampling step is documented.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not measured from a continuous raw sensor stream. It is a synthetic fixed time grid derived from the chosen stimulus-onset alignment, so the only raw trial variable it depends on is `stimOn_times` as the reference event.

ii. 
```python
ALIGN_EVENT = "stimOn_times"
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

```python
input_trial = np.vstack(
    [
        TIME_GRID,
        np.full(N_BINS, trial_num, dtype=np.float32),
    ]
).astype(np.float16)
```

iii. The notes describe this as a “fixed stimulus-aligned time grid” rather than a loaded variable, and say it was chosen to match the reference interpolation grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent precomputes a 100-point vector from -0.48 s to 1.5 s, interpreted as bin-end times for the 20 ms stimulus-aligned grid, then repeats that same vector for every retained trial.

ii. 
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

```python
input_trial = np.vstack(
    [
        TIME_GRID,
        np.full(N_BINS, trial_num, dtype=np.float32),
    ]
).astype(np.float16)
```

iii. The notes explicitly say the range is `[-0.48, 1.5]` because the grid stores bin-end times. That rationale comes from the agent’s attempt to mirror the reference interpolation grid for time-varying behavior.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction: the same fixed 100-bin grid used for trialized behavior is paired with each neural trial after spike counts are binned over the same `stimOn_times`-aligned window.

ii. 
```python
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)
...
input_trial = np.vstack(
    [
        TIME_GRID,
        np.full(N_BINS, trial_num, dtype=np.float32),
    ]
).astype(np.float16)
```

```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

iii. The notes justify this by saying the time input uses the same reference interpolation grid as the aligned behavior streams. The choice to use bin-end times rather than bin starts is also documented there.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` sequence in the trials table.

ii. 
```python
def compute_trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    ...
    for idx, value in enumerate(probability_left):
        ...
        if prev is None or np.isnan(prev) or not np.isclose(prev, value):
            count = 1
        else:
            count += 1
```

iii. The notes say this input uses the “trial-table `probabilityLeft` block structure” and is a task-specific derivation, not a field exposed by the reference cache.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script scans the full raw trial order, resets the count whenever `probabilityLeft` changes or becomes NaN, keeps the resulting count even across trials that are later filtered out, and then repeats the retained count across all 100 bins in each kept trial.

ii. 
```python
block_trial_number = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
...
trial_num = float(block_trial_number[idx])
...
np.full(N_BINS, trial_num, dtype=np.float32)
```

iii. The notes explicitly justify computing block trial number on the original trial order before masking so filtering does not renumber experimental history.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the raw `choice` column in the trials table.

ii. 
```python
def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0
    if np.isclose(choice_value, -1.0):
        return 1
```

iii. The notes map the trial-table `choice` variable directly to the output and state that no-choice trials are removed first.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps raw `choice` values `+1 -> 0` and `-1 -> 1`, interpreting those as left and right respectively, and then broadcasts that category across all 100 time bins of the trial.

ii. 
```python
choice_code = map_choice(trials_df.iloc[idx]["choice"])
...
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_code, dtype=np.uint8),
```

iii. The notes say this sign convention was verified directly on raw data and was chosen to satisfy the task’s requested left/right coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the raw `probabilityLeft` column in the trials table.

ii. 
```python
def map_prior(prob_left: float) -> int:
    if np.isclose(prob_left, 0.2):
        return 0
    if np.isclose(prob_left, 0.5):
        return 1
    if np.isclose(prob_left, 0.8):
        return 2
```

iii. The notes describe this as a direct mapping from trial-table `probabilityLeft` to the requested categorical prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script discretizes `probabilityLeft` as `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, then repeats that category across the trial’s time bins.

ii. 
```python
prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
...
np.full(N_BINS, prior_code, dtype=np.uint8),
```

iii. The notes justify this as the exact remapping requested in the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from raw wheel timestamps and wheel position arrays: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. 
```python
ts_path = ensure_dataset_path_any(
    one,
    row,
    ["alf/_ibl_wheel.timestamps.npy", "alf/#*/_ibl_wheel.timestamps.npy"],
    "_ibl_wheel.timestamps.npy",
    "alf",
)
pos_path = ensure_dataset_path_any(
    one,
    row,
    ["alf/_ibl_wheel.position.npy", "alf/#*/_ibl_wheel.position.npy"],
    "_ibl_wheel.position.npy",
    "alf",
)
```

iii. The notes say wheel outputs use “wheel timestamps + position,” matching the behavior source described in the reference code and papers.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent interpolates wheel position to 1 kHz, computes filtered velocity, takes its absolute value as speed, slices each trial’s `[-0.5, 1.5]` stimulus-aligned interval, and linearly interpolates that onto the 20 ms output grid.

ii. 
```python
timestamps = np.load(ts_path)
position = np.load(pos_path)
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The notes justify this by saying the reference behavior source is wheel velocity magnitude and that the direct ALF implementation mirrors the reference interpolation logic.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, the agent pools all retained wheel-speed samples across all sessions and trials, computes global tertile cut points, then uses `np.digitize` to map each time point into `low`, `mid`, or `high`.

ii. 
```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)
```

```python
def discretize(values: np.ndarray, edges: np.ndarray):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)
```

```python
discretize(wheel_vals, wheel_edges),
```

iii. The notes explicitly call this a global-tertile discretization policy chosen so category semantics are consistent across sessions. This is a task-specific decision because the reference code treats wheel speed as continuous.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same `stimOn_times` event and `[-0.5, 1.5]` window as the neural data, then resampled to the same 100-bin grid.

ii. 
```python
interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

```python
wheel_values, wheel_mask = get_behavior_per_interval(wheel["times"], wheel["values"], trials_df, allow_nans=False)
```

iii. The notes say the wheel trace was interpolated onto the same 20 ms stimulus-aligned grid used by the reference behavior utilities.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from camera timestamps plus ROI motion-energy arrays, preferring left camera files and falling back to right camera files when the left side is unavailable.

ii. 
```python
sides = [
    ("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
    ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy"),
]
```

```python
times = np.load(times_path).astype(np.float32)
values = np.load(energy_path).astype(np.float32)
return {"times": times, "values": values}, side
```

iii. The notes explicitly say whisker motion energy is taken from the left camera when available and otherwise from the right, matching the reference behavior loader.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the ROI motion-energy trace, checks coverage of the full trial-aligned interval, and linearly interpolates the trace onto the same 20 ms stimulus-aligned grid as the other time-varying streams.

ii. 
```python
whisker, whisker_source = load_whisker_motion_energy(one, row)
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
```

```python
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
y_interp = np.asarray(y_interp, dtype=np.float32).reshape(-1)
```

iii. The notes say this mirrors the reference `load_target_behavior` plus `get_behavior_per_interval` path for whisker motion energy.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is pooled across all retained aligned time points, split at global tertiles, and digitized into three categories.

ii. 
```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
```

```python
discretize(whisker_vals, whisker_edges),
```

iii. The notes justify this with the same global-tertile policy used for wheel speed and note that the papers/reference code used continuous whisker values, not categorical bins.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to `stimOn_times` over `[-0.5, 1.5]` s and interpolated to the same 100-bin grid used for neural counts and wheel speed.

ii. 
```python
interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

```python
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
```

iii. The notes describe the whisker stream as sharing the same stimulus-aligned 20 ms interpolation grid as the other trialized outputs.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing datasets trigger an attempted ONE download. Missing essential trial fields, missing full wheel/whisker coverage, non-finite derived block counts, probes with no good units, and sessions with fewer than two valid trials all cause filtering or skipping. The code generally drops bad trials/sessions rather than imputing missing values.

ii. 
```python
one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
...
raise FileNotFoundError(
    f"Missing dataset after download attempt: eid={row.eid} dataset={dataset_name} collection={collection}"
)
```

```python
if not good:
    vals_list.append(None)
    mask.append(False)
    continue
```

```python
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. The notes say the agent preferred explicit filtering and skip rules, and also document the fallback ONE download policy as a way to avoid silently losing partially cached sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are per-session raw loading, probe-wise spike processing, per-trial spike binning, per-trial behavior interpolation, and any on-demand downloads. The agent’s own notes also identify full verification output volume and float32 neural storage as practical runtime/memory costs.

ii. 
```python
for probe_row in probe_df.itertuples(index=False):
    spikes, clusters = load_probe_data(one, row, probe_row.probe_name, brain_regions)
```

```python
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
    ...
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
```

```python
for interval_idx, (ib, ie) in enumerate(zip(idxs_beg, idxs_end)):
    ...
    y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. `CONVERSION_NOTES.md` explicitly names per-session spike binning and remote downloads as bottlenecks, and the code structure matches that assessment.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops are obvious vectorization candidates: the block-trial-number loop, the per-interval spike-binning loop, the per-interval behavior interpolation loop, the per-trial assembly loop in `process_session`, and the per-trial output-construction loop in `build_dataset`.

ii. 
```python
for idx, value in enumerate(probability_left):
    ...
```

```python
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
    ...
```

```python
for idx in valid_idx:
    ...
```

```python
for choice_code, prior_code, wheel_vals, whisker_vals in zip(
    rec.choice_codes,
    rec.prior_codes,
    rec.wheel_values,
    rec.whisker_values,
):
```

iii. The notes mention some speedups the agent added, but these remaining loops still dominate Python-side work and could be pushed further into array operations or batched kernels.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly re-creates the same `TIME_GRID` and per-trial constant vectors, repeatedly converts arrays to compact dtypes, repeatedly loops over trials to broadcast per-trial labels into 100-bin outputs, and performs a two-pass behavior workflow where continuous wheel/whisker values are first stored for all trials and then revisited to compute global thresholds and categorical outputs.

ii. 
```python
input_trial = np.vstack(
    [
        TIME_GRID,
        np.full(N_BINS, trial_num, dtype=np.float32),
    ]
).astype(np.float16)
```

```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
```

```python
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_code, dtype=np.uint8),
        np.full(N_BINS, prior_code, dtype=np.uint8),
        discretize(wheel_vals, wheel_edges),
        discretize(whisker_vals, whisker_edges),
    ]
)
```

iii. The notes explicitly describe the implementation as a two-pass pipeline and justify it as the simplest way to get dataset-wide tertile cut points.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores diagnostic payloads and processing plots for sample runs, retains continuous wheel/whisker traces in per-session cache objects only to later convert them to categories, repeats trial-level labels across every time bin although they are constant within a trial, and stores the same time grid in every trial instead of once globally.

ii. 
```python
if with_diagnostic:
    diagnostic = {
        "neural": neural_trials[0],
        "time_input": input_trials[0][0],
        "wheel_raw_times": wheel["times"],
        "wheel_raw_values": wheel["values"],
        ...
    }
```

```python
wheel_cont.append(np.asarray(wheel_values[idx], dtype=np.float16))
whisker_cont.append(np.asarray(whisker_values[idx], dtype=np.float16))
```

```python
np.full(N_BINS, choice_code, dtype=np.uint8),
np.full(N_BINS, prior_code, dtype=np.uint8),
```

iii. The notes justify some of this as validation or storage tradeoff work, but none of those extras are required by the decoder beyond the final discrete trial arrays.
