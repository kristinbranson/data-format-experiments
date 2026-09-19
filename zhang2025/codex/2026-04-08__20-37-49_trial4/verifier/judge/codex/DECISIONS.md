# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use `one.search`, `SessionLoader`, or `SpikeSortingLoader` as the primary loading path. It treated `code/code_zhang2025/data/bwm_release.csv` as the canonical session/probe index, built session-relative paths under `data/one_cache` and `cache/one_cache`, opened ALF files directly with `pandas` and `numpy`, and only used `ONE.load_dataset(..., download_only=True)` as a fallback when a required file was missing locally.

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

```python
def ensure_dataset_path_any(one: ONE, row, relative_globs: list[str], dataset_name: str, collection: str):
    for relative_glob in relative_globs:
        path = locate_dataset(row, relative_glob)
        if path is not None:
            return path
    one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
```

iii. The stated justification in `CONVERSION_NOTES.md` is that the AI "implemented direct ALF loading because the reference helper stack depends on unavailable optional packages in this environment," while still using the local ONE cache and on-demand ONE downloads when needed.

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from the `subject` column in `bwm_release.csv`. In the final dataset, `subjects` are accumulated in first-seen session order, and `subject_idx` is built from that insertion order.

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

for rec in session_records:
    if rec.subject not in subject_to_idx:
        subject_to_idx[rec.subject] = len(subjects)
        subjects.append(rec.subject)
    subject_idx.append(subject_to_idx[rec.subject])
```

iii. The notes treat the release CSV as authoritative for subject labels, so no path parsing or derived subject inference was needed.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` rows in `bwm_release.csv`. The converter drops duplicate probe rows per `eid`, keeps one metadata row per session, and separately stores the probe rows belonging to that session.

ii.
```python
session_rows = (
    bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
    .reset_index(drop=True)
)
probe_rows = {
    eid: grp[["pid", "probe_name"]].reset_index(drop=True)
    for eid, grp in bwm.groupby("eid", sort=False)
}
```

iii. `CONVERSION_NOTES.md` says the 459-session release list in the reference CSV was used as the canonical session inventory.

## 1-d. How are the data split into trials?

i. Trials are taken from the rows of the ALF trials parquet table. The code loads the full table once per session and then keeps a subset of row indices, `valid_idx`, after trial-level filtering.

ii.
```python
def load_trials_table(one: ONE, row):
    path = ensure_dataset_path(one, row, "alf/#*/_ibl_trials.table.pqt", "_ibl_trials.table.pqt", "alf")
    trials_df = pd.read_parquet(path)
    ...
    return trials_df
```

```python
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
```

iii. There is no separate written justification beyond following the ALF trials-table structure and then retaining only valid trial rows.

## 1-e. How are trials filtered based on quality controls?

i. The AI applied a base mask over the trials table, then intersected it with wheel and whisker coverage masks. The base mask excludes trials with reaction time outside `[0.08, 2.0]` s, missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`, trial length above `10` s, and no-choice trials. After that, trials also need valid wheel and whisker interpolation windows, and sessions with fewer than two surviving trials are dropped.

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
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    for event in nan_exclude:
        query += f" | {event}.isnull()"
    ...
    if exclude_nochoice:
        query += " | (choice == 0)"
    return ~trials_df.eval(query)
```

```python
wheel_values, wheel_mask = get_behavior_per_interval(wheel["times"], wheel["values"], trials_df, allow_nans=False)
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)

combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. The notes justify this as combining the reference code’s trial-mask logic with extra wheel/whisker coverage checks required by the requested decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays come from `spikes.times.npy` and `spikes.clusters.npy` on each probe, with `clusters.metrics.pqt` supplying QC labels and `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` supplying anatomical labels.

ii.
```python
times_path = ensure_dataset_path(one, row, f"{collection}/#*/spikes.times.npy", "spikes.times.npy", collection)
clu_path = ensure_dataset_path(
    one, row, f"{collection}/#*/spikes.clusters.npy", "spikes.clusters.npy", collection
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

iii. The notes say the AI chose good-unit filtering from the raw cluster labels to match the paper’s reported `75,708` well-isolated neurons, and kept Beryl region labels for retained units.

## 2-b. How is the `neural` data processed?

i. Probe-level spikes are filtered, probes are merged into a session-wide population with renumbered cluster IDs, and spikes are binned into 20 ms stimulus-aligned bins with `bincount2D`. The saved neural arrays are spike counts, not firing rates, and they are compacted to `uint8`.

ii.
```python
def merge_probes(spikes_list, clusters_list):
    ...
    spikes_local["clusters"] = spikes_local["clusters"] + cluster_max
    ...
    sort_idx = np.argsort(merged_spikes["times"], kind="stable")
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
    return merged_spikes, merged_clusters
```

```python
def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    ...
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
    ...
    binned_spikes[interval_idx, idxs_tmp, :] = binned_tmp[:, :n_bins]
```

```python
neural_trial = compact_neural_trial(binned_spikes[idx].T)
...
return neural_trial.astype(np.uint8, copy=False)
```

iii. The written rationale is that compact spike-count storage kept the full dataset trainable in memory. The notes and README describe the final representation as stimulus-aligned spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neuron-level QC in the converter is `label >= 1` from `clusters.metrics.pqt`. Probes with zero such units are skipped, and sessions with no remaining probes fail. The code does not explicitly exclude `void` regions.

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

iii. The notes repeatedly justify `label >= 1` as reproducing the paper’s good-unit count exactly and explicitly describe that as the chosen neural-unit policy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times` by forming per-trial windows from `stimOn_times - 0.5` s to `stimOn_times + 1.5` s and binning spikes within those intervals.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
```

```python
intervals = np.vstack(
    [
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
    ]
).T
```

iii. `CONVERSION_NOTES.md` says stimulus onset was chosen because it matched both the decoder task and the executable reference cache.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural activity is binned at 20 ms resolution over a 2 s window, producing 100 bins. There is no additional temporal rebinning after this binning step.

ii.
```python
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
```

```python
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
```

iii. The notes say 20 ms bins were used everywhere to match the executable reference.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI treated this input as a fixed stimulus-aligned grid defined by `ALIGN_EVENT`, `TIME_WINDOW`, and `BIN_SIZE`, then reused that same vector for every trial. It is conceptually tied to `stimOn_times`, but not recomputed from raw per-trial timestamps.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02
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

iii. The notes describe this as a "fixed stimulus-aligned time grid" used as the first decoder input.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI generated the input with `np.linspace(-0.48, 1.5, 100)`, so it used bin-end-style sample times rather than bin centers.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The notes explicitly justify this by saying the range is `[-0.48, 1.5]` "because the grid stores bin-end times."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The AI intended the input grid to be the same stimulus-aligned 20 ms grid used throughout the converter. The same `TIME_GRID` is used for `input`, and behavioral traces are interpolated to the same grid.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

```python
input_trial = np.vstack(
    [
        TIME_GRID,
        np.full(N_BINS, trial_num, dtype=np.float32),
    ]
).astype(np.float16)
```

iii. The written rationale is that all streams were put on a shared stimulus-aligned grid, although the notes frame that grid as bin-end times rather than bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` sequence in the trials table.

ii.
```python
def compute_trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    ...
```

```python
block_trial_number = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```

iii. The notes say this came from "trial-table `probabilityLeft` block structure."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the unfiltered `probabilityLeft` sequence in original trial order, resets the counter whenever `probabilityLeft` changes or becomes `NaN`, counts starting at `1`, and then repeats the resulting scalar across all 100 time bins of a kept trial.

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

```python
np.full(N_BINS, trial_num, dtype=np.float32)
```

iii. The notes explicitly say this feature is "computed on the raw trial order before filtering."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the trials-table `choice` column.

ii.
```python
def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0
    if np.isclose(choice_value, -1.0):
        return 1
```

```python
choice_code = map_choice(trials_df.iloc[idx]["choice"])
```

iii. The notes and README both state the requested left/right recoding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `+1` to `0` for left and `-1` to `1` for right, excludes `choice == 0` earlier in the trial mask, and repeats the categorical value across all 100 time bins.

ii.
```python
if np.isclose(choice_value, 1.0):
    return 0
if np.isclose(choice_value, -1.0):
    return 1
```

```python
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_code, dtype=np.uint8),
        ...
    ]
)
```

iii. The written justification is simply adherence to the decoder specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials-table `probabilityLeft` column.

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

```python
prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
```

iii. The notes say this is the requested prior output from `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI only recodes `0.2`, `0.5`, and `0.8` to `0`, `1`, and `2`, then repeats the category across all 100 time bins.

ii.
```python
if np.isclose(prob_left, 0.2):
    return 0
if np.isclose(prob_left, 0.5):
    return 1
if np.isclose(prob_left, 0.8):
    return 2
```

```python
np.full(N_BINS, prior_code, dtype=np.uint8)
```

iii. No deeper justification was given beyond following the task spec.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

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

iii. The notes identify the same wheel source variables and say the behavior source is consistent with the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code interpolates wheel position to 1000 Hz with `interpolate_position`, differentiates and low-pass filters it with `velocity_filtered`, takes the absolute velocity, then linearly interpolates that continuous trace into each stimulus-aligned trial window.

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

iii. The notes justify this as matching the reference behavior source and recommended wheel processing, with interpolation to the stimulus-aligned 20 ms grid.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI concatenates wheel traces across all retained sessions, computes global one-third and two-third quantiles, and uses those fixed edges to discretize every wheel sample into `0/1/2`.

ii.
```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)
```

```python
def discretize(values: np.ndarray, edges: np.ndarray):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as using "global tertile edges so wheel and whisker bins have consistent semantics across sessions."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated into each trial on the same stimulus-aligned 100-sample grid the AI uses elsewhere, namely from `stimOn_times - 0.48` s through `stimOn_times + 1.5` s.

ii.
```python
interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

iii. The notes say wheel and neural data were put on the same 20 ms stimulus-aligned grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is taken from `leftCamera.ROIMotionEnergy.npy` or, if unavailable, `rightCamera.ROIMotionEnergy.npy`, along with the matching `_ibl_<side>Camera.times.npy` timestamps.

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

iii. The notes say left camera is preferred with right-camera fallback, matching the reference behavior-loading logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used directly, without additional filtering or normalization, then linearly interpolated into each stimulus-aligned trial window.

ii.
```python
times = np.load(times_path).astype(np.float32)
values = np.load(energy_path).astype(np.float32)
return {"times": times, "values": values}, side
```

```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The notes describe this as using the raw whisker motion energy with left-first fallback and interpolation to the common grid.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the AI concatenates whisker traces across all retained sessions, computes global tertile edges, and discretizes each sample into three categories with those shared thresholds.

ii.
```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
```

```python
discretize(whisker_vals, whisker_edges)
```

iii. The written justification is the same as for wheel speed: global tertiles were chosen so categories have consistent semantics across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned by interpolating it into each trial window on the same 100-point stimulus-aligned grid used for `input` and wheel speed.

ii.
```python
interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

iii. The notes say all dynamic streams were placed on a common stimulus-aligned 20 ms grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing local files trigger an on-demand `ONE.load_dataset(..., download_only=True)` attempt. Missing/invalid trial-level values are removed by the trial mask and behavior-coverage mask. Probes with no good units are skipped, sessions with no good probes fail, sessions with missing whisker traces are skipped, and sessions with fewer than two valid trials are skipped.

ii.
```python
one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
```

```python
if good_cluster_ids.size == 0:
    return None, None
```

```python
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

```python
raise FileNotFoundError(f"No whisker motion energy trace available for session {row.eid} ({'; '.join(errors)})")
```

iii. The notes explicitly justify leaving whisker-missing sessions out because whisker motion energy was a required decoder output, and mention a later fix so zero-good-unit probes are skipped without losing an otherwise usable session.

## 10-a. What are the most time-consuming steps of the code?

i. The code is structured so the expensive steps are loading large probe-level spike arrays and related metadata from disk, then trializing those arrays session by session. The notes also emphasize session-level raw loading as the runtime bottleneck and introduce per-session caching to avoid repeating it.

ii.
```python
spike_times = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
cluster_metrics = pd.read_parquet(cluster_metrics_path)
```

```python
with ProcessPoolExecutor(max_workers=max_workers) as pool:
    ...
    pool.submit(
        process_session_worker,
        row._asdict(),
        probe_rows[row.eid].to_dict("records"),
        args.show_processing and idx < 2,
        str(SESSION_RECORD_CACHE),
    )
```

iii. `CONVERSION_NOTES.md` says the session cache exists because "expensive raw loading happens once" and reports full-cache fill timing, implying raw session loading was the main cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loops are per-interval spike binning, per-interval behavioral interpolation, and per-trial assembly of session outputs. These are all still explicit Python loops.

ii.
```python
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
    times_curr = times[ib:ie]
    clust_curr = clusters[ib:ie]
    ...
```

```python
for interval_idx, (ib, ie) in enumerate(zip(idxs_beg, idxs_end)):
    seg_t = target_times[ib:ie]
    seg_v = target_vals[ib:ie]
    ...
```

```python
for idx in valid_idx:
    neural_trial = compact_neural_trial(binned_spikes[idx].T)
    ...
```

iii. The AI did not give a detailed written defense of these loops, but the notes say it addressed efficiency mostly through session-level parallelism and caching rather than by fully vectorizing everything.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some processing for caching and diagnostics. It can compact session records once during processing and again during explicit cache repacking, and it discretizes diagnostic wheel/whisker traces again after the final global edges are known even though those same traces are also discretized into the final `output` arrays.

ii.
```python
def compact_session_record(record: SessionRecord) -> SessionRecord:
    ...
    neural=[compact_neural_trial(np.asarray(trial)) for trial in record.neural],
```

```python
def repack_session_cache():
    ...
    compact = compact_session_record(record)
```

```python
for rec in session_records:
    if rec.diagnostic is not None:
        rec.diagnostic["wheel_disc"] = discretize(rec.diagnostic["wheel_interp"], wheel_edges)
        rec.diagnostic["whisker_disc"] = discretize(rec.diagnostic["whisker_interp"], whisker_edges)
```

iii. The notes justify these extra passes as part of the cache/repack workflow and the `--show-processing` diagnostic plotting path.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter loads and carries some information that is not used by the final downstream decoder dataset. Examples include `cluster_depths`, per-session continuous wheel and whisker traces retained in `SessionRecord`, and the large diagnostic payload used only for plots. The final dataset only keeps the discretized wheel and whisker outputs, not those continuous intermediates.

ii.
```python
cluster_depths_path = ensure_dataset_path(
    one, row, f"{collection}/#*/clusters.depths.npy", "clusters.depths.npy", collection
)
...
cluster_depths = np.load(cluster_depths_path).astype(np.float32)
...
clusters_df["depths"] = cluster_depths
```

```python
wheel_cont.append(np.asarray(wheel_values[idx], dtype=np.float16))
whisker_cont.append(np.asarray(whisker_values[idx], dtype=np.float16))
```

```python
diagnostic = {
    "neural": neural_trials[0],
    "time_input": input_trials[0][0],
    "wheel_raw_times": wheel["times"],
    "wheel_raw_values": wheel["values"],
    "wheel_interp": wheel_cont[0],
    "whisker_raw_times": whisker["times"],
    "whisker_raw_values": whisker["values"],
    "whisker_interp": whisker_cont[0],
    ...
}
```

iii. The notes justify this extra state as supporting cache reuse, sanity checks, and optional processing plots rather than the final decoder input itself.
