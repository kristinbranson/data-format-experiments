# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the reference `ONE.search` plus `SessionLoader` / `SpikeSortingLoader` path. It loaded the release index from `code/code_zhang2025/data/bwm_release.csv`, treated each unique `eid` there as a session, resolved ALF files by globbing under `data/one_cache` and `cache/one_cache`, and used `ONE.load_dataset(..., download_only=True)` as a fallback when a file was missing locally.

ii. 
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

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

iii. In `CONVERSION_NOTES.md`, the AI justified this as using the reference release list while working around unavailable optional packages in the reference helper stack, and said missing local files should be downloaded on demand instead of silently dropping partially cached sessions.

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from the `subject` column in `bwm_release.csv`. During final assembly, the AI creates `subjects` and `subject_idx` in first-seen session order rather than sorting the subject list.

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
if rec.subject not in subject_to_idx:
    subject_to_idx[rec.subject] = len(subjects)
    subjects.append(rec.subject)
subject_idx.append(subject_to_idx[rec.subject])
```

iii. The AI’s notes say the release metadata already carry the subject label, so nothing had to be inferred from paths beyond using those metadata consistently in the assembled dataset.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from the release CSV. The AI drops duplicate probe rows and processes one session per remaining `eid`.

ii.
```python
def load_release_sessions():
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    session_rows = (
        bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
        .reset_index(drop=True)
    )
```

iii. The AI treated the release file as the canonical session index, matching the session granularity of the reference release.

## 1-d. How are the data split into trials?

i. Trials come directly from rows of `_ibl_trials.table.pqt`. The AI never reconstructs trial boundaries from another stream.

ii.
```python
def load_trials_table(one: ONE, row):
    path = ensure_dataset_path(one, row, "alf/#*/_ibl_trials.table.pqt", "_ibl_trials.table.pqt", "alf")
    trials_df = pd.read_parquet(path)
    ...
    return trials_df
```

iii. The AI followed the task table structure already present in the raw data.

## 1-e. How are trials filtered based on quality controls?

i. The AI builds a base trial mask with reaction time between `0.08` and `2.0` s, excludes no-choice trials, excludes trials with NaNs in several task columns, excludes trials with `feedback_times - goCue_times > 10 s`, then intersects that mask with wheel and whisker coverage masks. Trials without full aligned wheel or whisker coverage are removed. Sessions with fewer than two valid trials are dropped.

ii.
```python
def build_trial_mask(
    trials_df: pd.DataFrame,
    min_rt=0.08,
    max_rt=2.0,
    ...
    max_trial_len=10.0,
    exclude_nochoice=True,
):
    ...
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    ...
    if exclude_nochoice:
        query += " | (choice == 0)"
    return ~trials_df.eval(query)
```

```python
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. In the notes, the AI said it was mirroring the broader reference `load_trials_and_mask` logic and then enforcing shared validity across neural, wheel, and whisker streams for the requested decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is derived from `spikes.times.npy` and `spikes.clusters.npy` for each probe. Cluster metadata and channel-region IDs are used only to filter clusters and annotate retained neurons with regions.

ii.
```python
times_path = ensure_dataset_path(one, row, f"{collection}/#*/spikes.times.npy", "spikes.times.npy", collection)
clu_path = ensure_dataset_path(
    one, row, f"{collection}/#*/spikes.clusters.npy", "spikes.clusters.npy", collection
)
...
spike_times = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
```

iii. The AI’s notes explicitly map neural data to good-unit `spikes.times` and `spikes.clusters`, with QC labels and histology metadata used only for filtering and region labeling.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 20 ms counts over the stimulus-aligned `[-0.5, 1.5]` s window, keeps those as counts rather than converting to firing rate, transposes each trial to `(n_neurons, n_timepoints)`, and then compacts each trial to `uint8`.

ii.
```python
def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    ...
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
    ...
    binned_spikes[interval_idx, idxs_tmp, :] = binned_tmp[:, :n_bins]
```

```python
def bin_spiking_data(spikes, trials_df):
    ...
    binned_trials = np.asarray([x.T for x in binned_array], dtype=np.float32)
    return binned_trials, cluster_ids
```

```python
def compact_neural_trial(neural_trial: np.ndarray) -> np.ndarray:
    ...
    return neural_trial.astype(np.uint8, copy=False)
```

iii. The AI justified this in `CONVERSION_NOTES.md` as compact spike-count storage to keep the full dataset trainable and small enough to handle; it explicitly described the final task as decoding from “stimulus-aligned spike counts.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1` from `clusters.metrics.pqt`. Probes with zero surviving clusters are skipped. If all probes in a session fail that filter, the session is dropped.

ii.
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
...
if good_cluster_ids.size == 0:
    return None, None
...
if not spikes_list:
    raise RuntimeError(f"No probes with well-isolated clusters for session {row.eid}")
```

iii. The notes repeatedly justify this as matching the paper’s “75,708 well-isolated neurons” statistic, and the trajectory shows the AI deliberately switching from all clusters to `label >= 1`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to `stimOn_times`, using a fixed `TIME_WINDOW = (-0.5, 1.5)` around each trial’s stimulus onset.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
...
intervals = np.vstack(
    [
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
    ]
).T
```

iii. The AI said this choice was driven both by the user’s decoder specification and by the executable reference configuration using `align_time='stimOn_times'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses `BIN_SIZE = 0.02`, i.e. 20 ms bins, giving `N_BINS = 100` over the 2 s window. It does not apply any additional rebinning or smoothing.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
```

iii. The notes say this matches the reference code and papers’ 20 ms trialization.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial alignment event `stimOn_times` plus the fixed decoding window; the AI does not read a separate time signal from disk.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The notes describe this input as a fixed stimulus-aligned interpolation grid repeated for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI constructs a fixed 100-point grid from `-0.48` to `1.5` seconds using `np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS)`. It repeats that same vector for every retained trial.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
...
input_trial = np.vstack(
    [
        TIME_GRID,
        np.full(N_BINS, trial_num, dtype=np.float32),
    ]
).astype(np.float16)
```

iii. The trajectory and notes describe this as the common stimulus-aligned time grid for all streams, although the chosen grid is the right bin edges rather than the bin centers used by the human reference.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI aligns it by using the same `TIME_GRID` for the first input channel and for behavior interpolation, while neural spike counts are binned over the same 100 stimulus-aligned windows.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
...
input_trial = np.vstack([TIME_GRID, np.full(N_BINS, trial_num, dtype=np.float32)]).astype(np.float16)
```

iii. The AI justified this as a single shared time axis for input and behavior streams, though it used right-edge timestamps instead of the reference bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trial table. The AI treats every change in `probabilityLeft` as the start of a new block.

ii.
```python
def compute_trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    ...
    if prev is None or np.isnan(prev) or not np.isclose(prev, value):
        count = 1
    else:
        count += 1
```

iii. The notes say block structure had to be reconstructed from `probabilityLeft` transitions because there is no explicit block-number column.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes the count on the full unfiltered trial order, resets when `probabilityLeft` changes or is NaN, and counts trials within block starting from `1`. For retained trials it repeats that scalar across all 100 time bins.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
...
trial_num = float(block_trial_number[idx])
...
np.full(N_BINS, trial_num, dtype=np.float32)
```

iii. The AI explicitly documented two justifications: compute block history before filtering so dropped trials do not renumber behavior, and start counts at `1` within each block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes directly from the trial table’s `choice` column.

ii.
```python
choice_code = map_choice(trials_df.iloc[idx]["choice"])
```

iii. The AI used the IBL choice field directly and only remapped its sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `choice == 1.0` to `0` (left) and `choice == -1.0` to `1` (right). Trials with `choice == 0` are excluded earlier by the trial mask. In the final output each choice code is repeated across all 100 bins.

ii.
```python
def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0
    if np.isclose(choice_value, -1.0):
        return 1
```

```python
np.full(N_BINS, choice_code, dtype=np.uint8)
```

iii. The AI’s notes say it verified the IBL sign convention directly and used the binary mapping requested by the task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table column `probabilityLeft`.

ii.
```python
prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
```

iii. The AI followed the task instructions and the block-probability values already present in the raw trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeats the resulting code across all 100 bins for each retained trial.

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
np.full(N_BINS, prior_code, dtype=np.uint8)
```

iii. The notes say this was a direct categorical recoding required by the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The AI interpolates wheel position to a regular 1 kHz grid, computes filtered velocity, and takes the absolute value.

ii.
```python
timestamps = np.load(ts_path)
position = np.load(pos_path)
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

iii. The AI justified this as matching the IBL wheel-processing utilities used by the reference code.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes absolute filtered wheel velocity, then for each trial interpolates that continuous trace onto the fixed 100-bin stimulus-aligned grid. The continuous interpolated traces are kept in intermediate `SessionRecord` objects until a later dataset-wide discretization step.

ii.
```python
wheel = load_wheel_speed(one, row)
wheel_values, wheel_mask = get_behavior_per_interval(wheel["times"], wheel["values"], trials_df, allow_nans=False)
...
wheel_cont.append(np.asarray(wheel_values[idx], dtype=np.float16))
```

```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The AI’s notes describe this as the same wheel source and interpolation approach as the reference, with the extra design choice of deferring discretization to a second pass.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI thresholds wheel speed using one pair of dataset-wide tertile cut points computed over all retained sessions and all retained aligned wheel samples, then applies `np.digitize` to produce categories `0`, `1`, `2`.

ii.
```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)
...
discretize(wheel_vals, wheel_edges)
```

```python
def discretize(values: np.ndarray, edges: np.ndarray):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)
```

iii. The trajectory and notes explicitly justify this as a “dataset-wide tertile” policy so wheel-speed category semantics would be consistent across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI aligns wheel speed by interpolating it on the same stimulus-aligned 100-point `TIME_GRID` used elsewhere. Because `TIME_GRID` is defined from `-0.48` to `1.5`, the alignment is to right bin edges rather than bin centers.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

iii. The AI’s justification was that wheel, whisker, and the time input should share one common aligned grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`, preferring the left camera and falling back to the right camera.

ii.
```python
sides = [
    ("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
    ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy"),
]
...
return {"times": times, "values": values}, side
```

iii. The AI said this matched the reference code’s left-first whisker-source policy.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy trace as-is, linearly interpolates it into the stimulus-aligned 100-bin grid for each trial, stores the continuous aligned traces temporarily, and discretizes them only in the final assembly pass.

ii.
```python
whisker, whisker_source = load_whisker_motion_energy(one, row)
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
...
whisker_cont.append(np.asarray(whisker_values[idx], dtype=np.float16))
```

iii. The notes describe this as using the raw motion-energy stream directly, with no extra normalization beyond interpolation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI computes one dataset-wide pair of tertile cut points across all retained whisker samples, then digitizes each aligned whisker trace into categories `0`, `1`, `2`.

ii.
```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
...
discretize(whisker_vals, whisker_edges)
```

iii. As with wheel speed, the notes justify this as keeping category meanings consistent across sessions by using dataset-wide tertiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI aligns whisker motion energy by interpolating it on the same stimulus-aligned `TIME_GRID` used for time input and wheel speed. This again means right bin edges rather than reference bin centers.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

iii. The justification was the same shared-grid rationale used for the other time-varying non-neural streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing local files trigger an on-demand `ONE.load_dataset` attempt. Trials with missing required task fields or without full wheel/whisker coverage are dropped. Probes with no good units are skipped. Sessions with no surviving good-unit probe, fewer than two valid trials, or no whisker trace available raise errors and are skipped from the final dataset.

ii.
```python
one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
```

```python
if good_cluster_ids.size == 0:
    return None, None
...
if not spikes_list:
    raise RuntimeError(f"No probes with well-isolated clusters for session {row.eid}")
...
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. The AI explicitly documented these as pragmatic fallbacks so incomplete sessions would be skipped only when the requested decoder targets could not be formed.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are probe-level spike loading, per-session spike binning, and the second pass that concatenates all retained wheel and whisker traces to compute global quantile edges. The cache fill / reread workflow also adds substantial I/O in full runs.

ii.
```python
spike_times = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
...
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
```

```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
```

iii. The notes say full conversion was bottlenecked by spike loading/binning and that the global-tertile second pass is simple but memory-hungry.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could have been vectorized: per-interval spike binning, per-interval behavior interpolation, per-trial assembly into `neural` / `input` / intermediate continuous outputs, region-index construction, and per-trial final output assembly.

ii.
```python
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
    ...
```

```python
for interval_idx, (ib, ie) in enumerate(zip(idxs_beg, idxs_end)):
    ...
```

```python
for idx in valid_idx:
    ...
for rec in session_records:
    ...
    for choice_code, prior_code, wheel_vals, whisker_vals in zip(
        rec.choice_codes,
        rec.prior_codes,
        rec.wheel_values,
        rec.whisker_values,
    ):
```

iii. The AI recognized some of this in the notes, especially the cost of spike binning and repeated behavior concatenation, but chose the simpler explicit loop structure.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several kinds of work: it checks both cache roots, then retries after a download attempt for dataset lookup; it stores continuous wheel and whisker traces in one pass and later traverses them all again to compute global quantiles and discretize them; and it supports separate cache repack and cache reread workflows.

ii.
```python
for root in (READONLY_CACHE, WRITABLE_CACHE):
    ...
one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
for relative_glob in relative_globs:
    ...
```

```python
wheel_cont.append(np.asarray(wheel_values[idx], dtype=np.float16))
whisker_cont.append(np.asarray(whisker_values[idx], dtype=np.float16))
...
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
```

iii. The trajectory explicitly describes the converter as a two-pass pipeline: first collect aligned continuous behavior, then compute global tertile edges and write categorical outputs.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores full continuous wheel and whisker traces in `SessionRecord` even though the final dataset only keeps discretized categories. It also constructs optional diagnostic payloads and plotting support that are not part of `converted_data.pkl`, and it keeps cache-oriented metadata / repacking utilities outside the downstream decoder’s actual needs.

ii.
```python
class SessionRecord:
    ...
    wheel_values: list
    whisker_values: list
    ...
    diagnostic: dict | None
```

```python
wheel_cont.append(np.asarray(wheel_values[idx], dtype=np.float16))
whisker_cont.append(np.asarray(whisker_values[idx], dtype=np.float16))
...
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_code, dtype=np.uint8),
        np.full(N_BINS, prior_code, dtype=np.uint8),
        discretize(wheel_vals, wheel_edges),
        discretize(whisker_vals, whisker_edges),
    ]
)
```

iii. The AI justified this as necessary for its global-tertile second pass and for debugging/validation plots, even though those continuous intermediates are discarded from the final assembled dataset.
