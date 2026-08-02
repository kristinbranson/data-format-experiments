# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the canonical session list from `code/code_zhang2025/data/bwm_release.csv`, deduplicates it to one row per session (`eid`), then resolves each required ALF dataset from either the read-only local cache, the writable local cache, or an on-demand ONE download. Trials, wheel, whisker, and probe ephys files are then loaded per session from those resolved paths.

ii. ```python
def load_release_sessions():
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    session_rows = (
        bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
        .reset_index(drop=True)
    )
```

```python
def ensure_dataset_path_any(one: ONE, row, relative_globs: list[str], dataset_name: str, collection: str):
    for relative_glob in relative_globs:
        path = locate_dataset(row, relative_glob)
        if path is not None:
            return path
    one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
```

iii. The notes say the release CSV is the canonical 459-session source and that session data are loaded from local `data/one_cache` plus on-demand ONE downloads. The trajectory also shows the agent deliberately reimplemented direct ALF loading because the heavier reference loader stack had missing dependencies.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `subject` column in the release CSV. Each processed session record carries a `subject` string, and the final dataset builds a unique `subjects` list plus `subject_idx` per session.

ii. ```python
return SessionRecord(
    eid=row.eid,
    subject=row.subject,
    lab=row.lab,
```

```python
for rec in session_records:
    if rec.subject not in subject_to_idx:
        subject_to_idx[rec.subject] = len(subjects)
        subjects.append(rec.subject)
    subject_idx.append(subject_to_idx[rec.subject])
```

iii. The notes explicitly state the release list has 139 subjects and that session subject labels are indexed into a sorted/unique subject list in the final output.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from `bwm_release.csv`. Each `eid` is processed independently into a `SessionRecord`, optionally cached to `cache/session_records/<eid>.pkl`, and later assembled into the final dataset as one outer list element per session.

ii. ```python
session_rows = (
    bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
    .reset_index(drop=True)
)
```

```python
record_path = cache_dir / f"{row.eid}.pkl"
...
"session_eids": [rec.eid for rec in session_records],
```

iii. The notes repeatedly describe the 459-session release as the input unit and the per-session cache as a performance aid for rebuilding the final pickle.

## 1-d. How are the data split into trials?

i. Trials are the rows of `_ibl_trials.table.pqt`. The script loads the full trials table for each session, constructs trial-wise masks and trial-wise aligned intervals, bins spikes/behavior for every trial row, then retains only the valid trial indices.

ii. ```python
def load_trials_table(one: ONE, row):
    path = ensure_dataset_path(one, row, "alf/#*/_ibl_trials.table.pqt", "_ibl_trials.table.pqt", "alf")
    trials_df = pd.read_parquet(path)
```

```python
intervals = np.vstack(
    [
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
    ]
).T
```

iii. The notes describe the conversion as “trialized” and specifically tie the per-trial arrays to the ALF trials table plus stimulus-aligned windows.

## 1-e. How are trials filtered based on quality controls?

i. Trials first pass a base mask that excludes missing key events, reaction times outside `0.08` to `2.0` s, trial length over `10.0` s, and no-choice trials. They are then intersected with wheel and whisker coverage masks, so only trials with complete neural/behavioral support over the full alignment window are kept.

ii. ```python
def build_trial_mask(
    trials_df: pd.DataFrame,
    min_rt=0.08,
    max_rt=2.0,
    ...
    exclude_nochoice=True,
):
```

```python
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. `CONVERSION_NOTES.md` lists the same trial exclusions and explicitly adds full wheel/whisker coverage as required because those outputs are part of the requested converted dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from spike times and spike cluster assignments from all probes in a session, together with cluster metadata and channel brain-location IDs used for QC and region labels.

ii. ```python
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
```

iii. The notes identify the same raw sources and explicitly map “good-unit `spikes.times` and `spikes.clusters` from all probes in session” to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. The agent loads each probe separately, filters units to “well-isolated” clusters (`label >= 1`), remaps cluster IDs, merges probes within a session, sorts spikes by time, and bins spike counts into 20 ms bins over a 2 s stimulus-aligned window. Each retained trial is stored as a `(n_neurons, 100)` count matrix and compacted to `uint8`.

ii. ```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
...
spike_keep = np.isin(spike_clusters, good_cluster_ids)
spike_times = spike_times[spike_keep]
spike_clusters = spike_clusters[spike_keep]
```

```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)
...
neural_trial = compact_neural_trial(binned_spikes[idx].T)
```

iii. The trajectory shows the agent originally used all clusters, then changed to `label >= 1` because it matched the paper’s `75,708` well-isolated-neuron count and reduced memory enough for training. The notes explicitly call this a deliberate revision away from the executable reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script applies neuron-level QC by retaining only clusters whose `clusters.metrics.pqt` `label` is at least `1`. Probes with zero such units are skipped; sessions with no remaining good-unit probes fail. There is no additional region-level filtering beyond storing Beryl labels.

ii. ```python
if "label" not in clusters_df.columns:
    raise ValueError(f"Missing cluster quality label for {row.eid} {probe_name}")

good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
...
if good_cluster_ids.size == 0:
    return None, None
```

iii. The notes and trajectory both justify this as a paper-consistent “well-isolated neurons only” policy, even though the reference helper `prepare_data()` loads all clusters and only exposes QC labels in metadata.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each trial, the script constructs a window from `stimOn_times - 0.5` s to `stimOn_times + 1.5` s and bins spikes within that interval.

ii. ```python
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

iii. The notes explicitly state that the executable reference cache uses `stimOn_times` and that the agent chose to follow that executable reference despite paper text that discusses other alignments in some analyses.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins, giving 100 bins across the 2 s window. No later temporal rebinning is applied; the per-trial matrices are stored directly at that resolution.

ii. ```python
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The notes repeatedly state that 20 ms was chosen to match the executable reference cache and the decoder-format expectation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is not read from a standalone raw signal. It is derived from the alignment choice (`stimOn_times`) plus the fixed conversion window and bin size, producing a session-invariant time grid relative to stimulus onset.

ii. ```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The notes describe `input[0]` as a fixed stimulus-aligned time grid reused for every trial rather than a separately sampled raw measurement.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script builds a fixed vector of bin-end times from `-0.48` to `1.5` s in 20 ms steps, then copies that same vector into every trial’s first input row.

ii. ```python
input_trial = np.vstack(
    [
        TIME_GRID,
        np.full(N_BINS, trial_num, dtype=np.float32),
    ]
).astype(np.float16)
```

iii. The notes justify the `[-0.48, 1.5]` range by noting that the stored grid uses bin-end times rather than left bin edges.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is perfectly aligned by construction: it uses the same `ALIGN_EVENT`, `TIME_WINDOW`, `BIN_SIZE`, and `N_BINS` as the neural spike-count arrays, so each time value corresponds to the same trial bin index as the neural data.

ii. ```python
neural_trial = compact_neural_trial(binned_spikes[idx].T)
...
input_trial = np.vstack(
    [
        TIME_GRID,
        np.full(N_BINS, trial_num, dtype=np.float32),
    ]
)
```

iii. The notes describe the time input as the fixed interpolation grid shared by all other trialized arrays.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw trial-table `probabilityLeft` sequence.

ii. ```python
block_trial_number = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```

iii. The notes explicitly map trial-table `probabilityLeft` block structure to the target field `input[1]`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script scans trials in original session order, resets the count whenever `probabilityLeft` changes or becomes NaN, and otherwise increments the count within the current block. The resulting scalar is then repeated across all 100 bins of the trial.

ii. ```python
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
```

iii. The notes say this quantity is computed “on the raw trial order before filtering,” which matches the code.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the raw trial-table `choice` column.

ii. ```python
choice_code = map_choice(trials_df.iloc[idx]["choice"])
```

iii. The notes map trial-table `choice` directly to `output[0]`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script converts IBL choice signs into categorical left/right labels by mapping `+1` to `0` and `-1` to `1`, rejects `choice == 0` trials earlier in the mask, and repeats the resulting category across all time bins in the output trial array.

ii. ```python
def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0
    if np.isclose(choice_value, -1.0):
        return 1
```

```python
np.full(N_BINS, choice_code, dtype=np.uint8),
```

iii. The notes explicitly state “`+1 -> 0` (left), `-1 -> 1` (right)” and justify excluding no-choice trials because the requested decoder output is binary.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the raw trial-table `probabilityLeft` column.

ii. ```python
prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
```

iii. The notes map trial-table `probabilityLeft` directly to the requested prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script discretizes `probabilityLeft` into three categorical codes: `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. That category is then repeated across all 100 bins in the output trial matrix.

ii. ```python
def map_prior(prob_left: float) -> int:
    if np.isclose(prob_left, 0.2):
        return 0
    if np.isclose(prob_left, 0.5):
        return 1
    if np.isclose(prob_left, 0.8):
        return 2
```

iii. The notes list the same mapping and describe the output as a static per-trial category made time-shaped for a uniform target structure.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. ```python
ts_path = ensure_dataset_path_any(
    one,
    row,
    ["alf/_ibl_wheel.timestamps.npy", "alf/#*/_ibl_wheel.timestamps.npy"],
```

```python
pos_path = ensure_dataset_path_any(
    one,
    row,
    ["alf/_ibl_wheel.position.npy", "alf/#*/_ibl_wheel.position.npy"],
```

iii. The notes identify wheel data as timestamps plus position and explicitly say wheel speed is derived as the magnitude of wheel velocity.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The script interpolates wheel position to 1 kHz, computes filtered wheel velocity, takes its absolute value as speed, extracts each trial’s stimulus-aligned interval, linearly interpolates that interval to the 20 ms trial grid, and later discretizes the resulting values.

ii. ```python
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

```python
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The notes say the agent matched the reference behavior conceptually but reimplemented it directly from ALF wheel files to avoid unavailable loader dependencies.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After collecting all retained trial-aligned continuous wheel traces across sessions, the script computes global tertile edges and bins each wheel sample with `np.digitize` into three categories: low, mid, and high.

ii. ```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)
```

```python
def discretize(values: np.ndarray, edges: np.ndarray):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)
```

iii. The notes and trajectory describe this as a deliberate two-pass design: first gather continuous traces, then compute global tertiles so category semantics are consistent across sessions.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset trial window as the neural data. For each trial, wheel samples are required to cover `[-0.5, 1.5]` s around `stimOn_times`, then interpolated onto the same 100-bin grid used for neural counts.

ii. ```python
interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

iii. The notes explicitly acknowledge that the methods text discusses first-movement alignment for dynamic behaviors, but the agent chose stimulus-onset alignment to match the executable reference cache and the user’s requested decoder format.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from camera timestamps plus ROI motion-energy arrays, preferring the left camera and falling back to the right camera if needed.

ii. ```python
sides = [
    ("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
    ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy"),
]
```

iii. The notes say whisker motion energy comes from left camera when available, otherwise right camera, mirroring the reference helper.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the selected camera’s timestamps and ROI motion-energy values, extracts each trial’s stimulus-aligned interval, linearly interpolates that interval to the common 20 ms grid, and carries the continuous values forward until the final discretization step.

ii. ```python
times = np.load(times_path).astype(np.float32)
values = np.load(energy_path).astype(np.float32)
return {"times": times, "values": values}, side
```

```python
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
```

iii. The notes and trajectory both mention that the agent had to fix left-camera failure handling so right-camera fallback would actually happen in full-mode processing.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is thresholded using global tertile edges computed over all retained continuous whisker samples across the full dataset, yielding low/mid/high categories.

ii. ```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
```

```python
discretize(whisker_vals, whisker_edges),
```

iii. The notes describe wheel and whisker binning as a shared global-tertile design choice.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned exactly like wheel speed: the script requires coverage across the same `stimOn_times + [-0.5, 1.5]` window and interpolates onto the same 100-bin grid used for neural data.

ii. ```python
interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
...
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The notes frame this as the same executable-reference-versus-paper reconciliation that affected wheel alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed trials are excluded by the base trial mask; trials without full wheel/whisker coverage are excluded by the combined behavior mask; missing left whisker data triggers right-camera fallback; probes with zero good units are skipped; sessions with no usable probes or too few valid trials are skipped entirely. The code generally raises explicit errors for structurally missing required datasets rather than silently imputing values.

ii. ```python
for side, times_name, energy_name in sides:
    try:
        ...
        return {"times": times, "values": values}, side
    except Exception as exc:
        errors.append(f"{side}: {exc}")
        continue
```

```python
if spikes is None or clusters is None:
    print(f"  skipping probe {probe_row.probe_name}: no well-isolated clusters", flush=True)
    continue
...
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. The notes explicitly call out several edge-case fixes: right-camera whisker fallback, revisioned ALF path handling, and skipping only the empty-good-unit probe rather than losing the whole session.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is session-level raw data loading and per-session processing across the full 459-session release: resolving/downloading ALF datasets, loading and merging probe spikes/clusters, trializing spike counts, interpolating wheel/whisker traces, and writing per-session cache files.

ii. ```python
with ProcessPoolExecutor(max_workers=max_workers) as pool:
    futures = {}
    for idx, row in enumerate(session_rows.itertuples(index=False)):
        ...
        pool.submit(
            process_session_worker,
```

iii. The notes quantify full cache filling as the expensive stage and justify session caching and parallel workers as the main speedups.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest non-vectorized loops are the per-interval loops in `get_spike_data_per_interval()` and `get_behavior_per_interval()`, the per-trial loop that materializes `neural`, `input`, and behavior lists in `process_session()`, and the per-session/per-trial assembly loops in `build_dataset()`. These loops are straightforward but Python-heavy.

ii. ```python
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
    ...
```

```python
for idx in valid_idx:
    neural_trial = compact_neural_trial(binned_spikes[idx].T)
    ...
```

iii. The notes mention session-level parallelism and caching, but they do not claim these inner loops were optimized; the code structure shows they remain mostly scalar/Python loops.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats path-resolution and raw dataset loading patterns across modalities; repeated interval interpolation logic for wheel and whisker; repeated array materialization when caching then rebuilding from cache; and repeated trial-wise expansion of static labels (`choice`, `prior`, `trial number`) into length-100 vectors.

ii. ```python
wheel_values, wheel_mask = get_behavior_per_interval(wheel["times"], wheel["values"], trials_df, allow_nans=False)
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
```

```python
np.full(N_BINS, choice_code, dtype=np.uint8),
np.full(N_BINS, prior_code, dtype=np.uint8),
np.full(N_BINS, trial_num, dtype=np.float32),
```

iii. The trajectory explicitly describes the overall converter as a two-pass pipeline, so some repetition is intentional: continuous traces are first collected, then revisited after global thresholds are known.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores and plots optional diagnostic data that are not used downstream, keeps continuous wheel/whisker values in per-session cache even though the final exported dataset only retains discretized categories, expands static per-trial labels across all 100 bins despite them being constant, and writes metadata for caches/revisions that do not affect decoder inputs directly.

ii. ```python
diagnostic = {
    "neural": neural_trials[0],
    "time_input": input_trials[0][0],
    "wheel_raw_times": wheel["times"],
    ...
}
```

```python
wheel_cont.append(np.asarray(wheel_values[idx], dtype=np.float16))
whisker_cont.append(np.asarray(whisker_values[idx], dtype=np.float16))
...
discretize(wheel_vals, wheel_edges),
discretize(whisker_vals, whisker_edges),
```

iii. The notes explicitly call out processing plots, diagnostic traces, compact session caches, and the two-pass continuous-to-discrete workflow as implementation choices made for validation and feasibility rather than final model input requirements.
