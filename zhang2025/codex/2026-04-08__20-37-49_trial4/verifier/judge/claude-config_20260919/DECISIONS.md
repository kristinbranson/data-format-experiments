# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** drive loading through the ONE search API. It reads the reference repository's release manifest `code/code_zhang2025/data/bwm_release.csv` (459 sessions, 699 probe insertions) and uses its `eid / subject / lab / date / session_number / pid / probe_name` columns as the canonical index. From each CSV row it reconstructs the ALF path `<lab>/Subjects/<subject>/<date>/<NNN>/` and globs the file it wants inside the read-only cache `data/one_cache` first and the writable cache `cache/one_cache` second, taking the *last* match so that the newest `#YYYY-MM-DD#` ALF revision wins. Only if no file is found on disk does it fall back to `one.load_dataset(..., download_only=True)` against openalyx. Six raw products are read per session: `_ibl_trials.table.pqt`, `_ibl_wheel.timestamps/position.npy`, `<side>Camera.ROIMotionEnergy.npy` + `_ibl_<side>Camera.times.npy`, and per probe `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, `clusters.depths.npy`, `channels.brainLocationIds_ccf_2017.npy`. Sessions are farmed out to a `ProcessPoolExecutor` (4 workers), each session record is pickled to `cache/session_records/<eid>.pkl`, and a second pass re-reads those records and assembles the final dictionary. Flags `--resume-session-cache`, `--from-session-cache`, `--fill-session-cache-only` and `--repack-session-cache` exist to drive the two passes separately.

ii.
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def load_release_sessions():
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    session_rows = (bwm.drop_duplicates("eid")[["eid","subject","lab","date","session_number"]]
                       .reset_index(drop=True))
    probe_rows = {eid: grp[["pid","probe_name"]].reset_index(drop=True)
                  for eid, grp in bwm.groupby("eid", sort=False)}
    return session_rows, probe_rows

def session_rel_path(row) -> Path:
    return Path(row.lab) / "Subjects" / row.subject / str(row.date) / f"{int(row.session_number):03d}"

def locate_dataset(row, relative_glob: str):
    rel = session_rel_path(row)
    for root in (READONLY_CACHE, WRITABLE_CACHE):
        matches = sorted((root / rel).glob(relative_glob))
        if matches:
            return matches[-1]          # newest ALF revision
    return None

def ensure_dataset_path_any(one, row, relative_globs, dataset_name, collection):
    for g in relative_globs:
        p = locate_dataset(row, g)
        if p is not None:
            return p
    one.load_dataset(row.eid, dataset_name, collection=collection, download_only=True)
    ...
```

iii. From CONVERSION_NOTES Step 6: *"Implemented direct ALF loading because the reference helper stack depends on unavailable optional packages in this environment."* Step 4 records that the CSV, the release metadata and the data paper all agree on 459 sessions / 699 probes / 139 mice, so the CSV was taken as canonical. Step 2 notes that ALF revisions are versioned with `#YYYY-MM-DD#`, *"so the conversion must resolve the latest matching revision instead of hardcoding one date."* The session cache was added so *"expensive raw loading happens once"*.

## 1-b. How are the data split into subjects?

i. The subject label is taken verbatim from the `subject` column of `bwm_release.csv`; no path parsing or inference is involved. The subject is carried on each `SessionRecord`. At assembly the `subjects` list is built in order of first appearance (not sorted), and `subject_idx[s]` is that session's position in the list. Result: 136 subjects over the 444 retained sessions (139 in the release; 3 subjects are lost with the 15 dropped sessions).

ii.
```python
session_rows = bwm.drop_duplicates("eid")[["eid", "subject", "lab", ...]]
...
return SessionRecord(eid=row.eid, subject=row.subject, lab=row.lab, ...)
...
for rec in session_records:
    if rec.subject not in subject_to_idx:
        subject_to_idx[rec.subject] = len(subjects)
        subjects.append(rec.subject)
    subject_idx.append(subject_to_idx[rec.subject])
```

iii. CONVERSION_NOTES Step 4 checks the subject count three ways — *"Reference release list has 139 subjects | Release metadata have 139 subjects | Data paper reports 139 mice → Consistent"* — so the CSV column was accepted as an already-unique subject id needing no derivation.

## 1-c. How are the data split into sessions?

i. A session is one `eid`. `bwm_release.csv` has one row per probe insertion, so the AI de-duplicates on `eid` to get 459 sessions and groups the remaining rows into a per-session probe table. Each session is processed independently in its own worker and becomes one element of the `neural` / `input` / `output` lists; session order in the final dictionary follows the CSV row order (restored via the `indexed_records` dict keyed by CSV index, so parallel out-of-order completion does not scramble it).

ii.
```python
probe_rows = {eid: grp[["pid", "probe_name"]].reset_index(drop=True)
              for eid, grp in bwm.groupby("eid", sort=False)}
...
for idx, row in enumerate(session_rows.itertuples(index=False)):
    futures[pool.submit(process_session_worker, row._asdict(),
                        probe_rows[row.eid].to_dict("records"), ...)] = idx
...
session_records = [indexed_records[idx] for idx in sorted(indexed_records)]
```

iii. No explicit justification is given beyond Step 4's *"Used the 459-session release list as canonical"*; the session is the natural unit of the release and the `eid` is already unique.

## 1-d. How are the data split into trials?

i. A trial is one row of `_ibl_trials.table.pqt`, so no splitting is performed. The AI validates that the eleven columns it depends on (`goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight, probabilityLeft, feedback_times, feedbackType, rewardVolume, firstMovement_times`) are present and raises otherwise. Every downstream array (neural bins, wheel trace, whisker trace, trial mask, block number) is built row-aligned to that table and indexed by the same `valid_idx`.

ii.
```python
def load_trials_table(one, row):
    path = ensure_dataset_path(one, row, "alf/#*/_ibl_trials.table.pqt",
                               "_ibl_trials.table.pqt", "alf")
    trials_df = pd.read_parquet(path)
    expected_cols = ["goCue_times","response_times","choice","stimOn_times","contrastLeft",
                     "contrastRight","probabilityLeft","feedback_times","feedbackType",
                     "rewardVolume","firstMovement_times"]
    missing = [c for c in expected_cols if c not in trials_df.columns]
    if missing:
        raise ValueError(f"Missing expected trial columns: {missing}")
    return trials_df
```

iii. Implicit: the ALF trials table is already one row per trial. The column check is the AI's guard against silently mis-reading a differently-versioned table.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are ANDed together.
   1. `build_trial_mask` is a re-implementation of the reference repo's `load_trials_and_mask`, using the reference's own default arguments: reaction time (`firstMovement_times - stimOn_times`) must lie in **[0.08 s, 2.0 s]**; `feedback_times - goCue_times` must be **≤ 10 s** (the value `prepare_data` passes as `max_trial_len`); and `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType` must all be non-NaN. `exclude_unbiased=False`, so the 0.5 prior block at the start of each session is **kept**; `exclude_nochoice=True`, so `choice == 0` trials are dropped.
   2. **Wheel coverage**: the trial's [-0.5, 1.5] s window must contain at least one wheel sample, contain no NaN, and the first/last samples inside the window must be within one bin (20 ms) of the window edges.
   3. **Whisker coverage**: the same test applied to the camera motion-energy trace.
   A session with fewer than 2 surviving trials raises and is dropped. Result: 188,925 of ~296,090 raw trials, 425.5 trials/session.
   Note: unlike the human reference, `probabilityLeft ∈ {0.2, 0.5, 0.8}` is never tested in the mask; instead `map_prior` raises on any other value, which would abort the whole session rather than drop the trial (this never fired on this release).

ii.
```python
def build_trial_mask(trials_df, min_rt=0.08, max_rt=2.0, nan_exclude="default",
                     min_trial_len=None, max_trial_len=10.0,
                     exclude_unbiased=False, exclude_nochoice=True):
    if nan_exclude == "default":
        nan_exclude = ["stimOn_times","choice","feedback_times","probabilityLeft",
                       "firstMovement_times","feedbackType"]
    query = f"(firstMovement_times - stimOn_times < {min_rt})"
    query += f" | (firstMovement_times - stimOn_times > {max_rt})"
    query += f" | (feedback_times - goCue_times > {max_trial_len})"
    for event in nan_exclude:
        query += f" | {event}.isnull()"
    if exclude_nochoice:
        query += " | (choice == 0)"
    return ~trials_df.eval(query)
```

```python
# coverage test, inside get_behavior_per_interval
if len(seg_v) == 0:                                          good = False
elif np.sum(np.isnan(seg_v)) > 0 and not allow_nans:         good = False
elif np.abs(interval_begs[i] - seg_t[0])  > BIN_SIZE:        good = False
elif np.abs(interval_ends[i] - seg_t[-1]) > BIN_SIZE:        good = False
```

```python
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. CONVERSION_NOTES Step 3 lists the curation rules as coming from the papers *and* the reference code: *"exclude missing key events; exclude reaction times outside 0.08 to 2.0 s; reference code also excludes feedback_times - goCue_times > 10 s"*. Step 10 records the one intentional addition: *"deliberate differences are limited to: enforcing binary choice by removing no-choice trials"* — required because the decoder output `Choice` is specified as binary. The coverage test is inherited from the reference `get_behavior_per_interval`. (The Step 3 note also lists `goCue_times` among the NaN-excluded events; the implemented default list does not include it. A 60-session scan shows zero trials are affected, so this is a slip in the notes rather than a different filter.)

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` per probe supply the array itself. Three further products supply the per-unit metadata: `clusters.metrics.pqt` (the `label` QC score and `cluster_id`), `clusters.channels.npy` (peak channel per cluster) and `channels.brainLocationIds_ccf_2017.npy` (Allen CCF id per channel), which are combined into an Allen acronym per cluster and then mapped to Beryl. `clusters.depths.npy` is also loaded but never used downstream.

ii.
```python
spike_times    = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
cluster_channels = np.load(cluster_channels_path).astype(np.int64)
cluster_metrics  = pd.read_parquet(cluster_metrics_path)
channel_region_ids = np.load(channel_regions_path).astype(np.int64)

cluster_region_ids = channel_region_ids[cluster_channels]
cluster_acronyms   = brain_regions.id2acronym(cluster_region_ids)
clusters_df["acronym"] = cluster_acronyms
```

iii. CONVERSION_NOTES Step 2 documents the file layout `alf/probeXX/pykilosort/#*/` and its contents; Step 5 maps *"Good-unit `spikes.times` and `spikes.clusters` from all probes in session → `neural`"*. The channel→cluster region derivation replaces `SpikeSortingLoader.merge_clusters`, which the AI could not call in this environment.

## 2-b. How is the `neural` data processed?

i. Per probe, spikes belonging to non-good clusters are discarded and the surviving clusters are renumbered 0..n-1. The probes of a session are then merged with a line-for-line copy of the reference `merge_probes`: cluster indices of probe *k* are offset by the cumulative cluster count, the spike arrays are concatenated and stably re-sorted by time. Spikes are then counted into 20 ms bins over [-0.5, +1.5] s around `stimOn_times`, one call to `iblutil.numerical.bincount2D` per trial with `xlim=[t_beg, t_end]`, the 101st edge-bin dropped by `[:, :n_bins]`. No smoothing, no z-scoring, no PSTH baseline subtraction. The result is **spike counts**, stored as `uint8` (not converted to Hz and not `float32`); `compact_neural_trial` raises if any bin exceeds 255. Each trial is shaped `(n_neurons, 100)`. Clusters that emitted no spike anywhere in the session are implicitly dropped because `cluster_ids = np.unique(clusters)`.

ii.
```python
def merge_probes(spikes_list, clusters_list):
    cluster_max = 0
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes_local = {k: np.array(v, copy=True) for k, v in spikes.items()}
        spikes_local["clusters"] = spikes_local["clusters"] + cluster_max
        cluster_max = len(clusters) + cluster_max
        ...
    sort_idx = np.argsort(merged_spikes["times"], kind="stable")
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
```

```python
def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    n_bins = int(np.ceil(interval_len / binsize))
    cluster_ids = np.unique(clusters)
    binned_spikes = np.zeros((len(interval_begs), len(cluster_ids), n_bins), dtype=np.float32)
    idxs_beg = np.searchsorted(times, interval_begs, side="left")
    idxs_end = np.searchsorted(times, interval_ends, side="left")
    for i, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
        binned_tmp, _, cluster_idxs = bincount2D(times[ib:ie], clusters[ib:ie],
                                                 xbin=binsize, xlim=[t_beg, t_end])
        _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
        binned_spikes[i, idxs_tmp, :] = binned_tmp[:, :n_bins]
    return binned_spikes, cluster_ids
```

```python
def compact_neural_trial(neural_trial):
    if int(neural_trial.max(initial=0)) > np.iinfo(np.uint8).max:
        raise ValueError(...)
    return neural_trial.astype(np.uint8, copy=False)
```

iii. Step 1/Step 10: the binning path is a direct port of the reference `get_spike_data_per_interval` / `bin_spiking_data`, and probe merging follows the reference `merge_probes` because probes in one session share behaviour. Step 6/Step 10 justify `uint8`: *"Full dataset with float32 neural arrays was too large to train comfortably: Stored neural counts as uint8. The verifier warns, but the trainer converts them and full training now completes."*

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single cut: clusters whose IBL spike-sorting `label` (0, ⅓, ⅔ or 1) is **≥ 1** are kept, NaN labels treated as 0. This is applied per probe *before* merging, so only good-unit spikes ever enter the merged arrays. A probe with zero good units is skipped rather than failing the session; a session with no good probe is dropped. No other neural QC is applied — in particular, units whose Beryl acronym is `void` (i.e. histologically placed outside the brain) are **kept**: the final dataset contains 250 `void` neurons and 266 brain regions, versus 264 regions and no `void` in the human reference. `root` units (10,069) are kept by both. Retained: 73,044 neurons over 444 sessions (164.5/session); the release-wide `label >= 1` count is 75,708 and the 2,664 missing units are exactly those in the 15 dropped sessions.

ii.
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
if "cluster_id" in clusters_df.columns:
    good_cluster_ids = clusters_df.loc[good_mask, "cluster_id"].to_numpy(dtype=np.int64)
else:
    good_cluster_ids = np.flatnonzero(good_mask).astype(np.int64)
if good_cluster_ids.size == 0:
    return None, None

spike_keep     = np.isin(spike_clusters, good_cluster_ids)
spike_times    = spike_times[spike_keep]
spike_clusters = spike_clusters[spike_keep]
remap = np.full(int(np.max(good_cluster_ids)) + 1, -1, dtype=np.int32)
remap[good_cluster_ids] = np.arange(good_cluster_ids.size, dtype=np.int32)
spike_clusters = remap[spike_clusters]
clusters_df = clusters_df.loc[good_mask].reset_index(drop=True)
```

iii. This is the AI's most-argued decision. Step 1: *"The papers emphasize 75,708 well-isolated neurons. The raw release cluster labels (label >= 1) sum to exactly 75,708, so this is the defensible neural-unit policy… The reference code exposes QC labels but does not force them in `prepare_data`; I initially mirrored that behavior, then revised to good-unit filtering after verifying the paper-level count from raw data and after the all-cluster version produced mismatched scale and impractical training behavior."* Step 10 lists the probe-skip fix as an edge case found and fixed: *"Probe with zero good units incorrectly killed the full session: Fixed by skipping only that probe."* `void` is never mentioned anywhere in the notes or the trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times`. Per trial the window `[stimOn - 0.5, stimOn + 1.5]` is formed in absolute session time, `np.searchsorted` cuts the (time-sorted) merged spike train to that window, and `bincount2D(..., xlim=[t_beg, t_end])` places each spike into bin `floor((t - t_beg)/0.02)`. So bin *i* of every trial spans `[stimOn - 0.5 + 0.02i, stimOn - 0.5 + 0.02(i+1))`. No clock correction is applied: spike times, trial event times, wheel timestamps and camera frame times are all already on the IBL session clock. Trials whose `stimOn_times` is NaN produce an empty slice and an all-zero block, but they are removed by the trial mask.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)

def bin_spiking_data(spikes, trials_df):
    intervals = np.vstack([trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
                           trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]]).T
    binned_array, cluster_ids = get_spike_data_per_interval(
        spikes["times"], spikes["clusters"],
        interval_begs=intervals[:, 0], interval_ends=intervals[:, 1],
        interval_len=TIME_WINDOW[1] - TIME_WINDOW[0], binsize=BIN_SIZE)
    return np.asarray([x.T for x in binned_array], dtype=np.float32), cluster_ids
```

iii. Step 1: *"The executable reference cache path uses `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`, and `binsize=0.02`."* Step 4 resolves the one ambiguity: *"Some paper figures discuss movement alignment → Used stimulus onset because it matches both the executable reference and the user's decoder task."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms, 100 bins per trial, uniform across every trial and session; `metadata['time_bin_size'] = 20.0` (ms) with `off_start = -0.5`, `off_end = 1.5`. Spikes go straight from raw spike times into 20 ms bins — there is no intermediate finer binning and no rebinning or resampling of the neural data. The behavioural streams are resampled once onto the same 100-point grid (the wheel is first upsampled to 1000 Hz by `interpolate_position` before velocity is taken, then decimated onto the 100 points). Every trial is verified to be `(n_neurons, 100)`.

ii.
```python
BIN_SIZE = 0.02
N_BINS   = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))     # 100
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
...
if neural_trial.shape != (len(cluster_ids), N_BINS):
    raise ValueError(f"Unexpected neural shape {neural_trial.shape}")
...
"time_bin_size": 20.0,
```

iii. Step 4 resolves a conflict explicitly: *"Code uses 20 ms | Papers mention 20 ms generally; some text sections mention 50 ms for other analyses → Used 20 ms everywhere to match the executable reference."*

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` only, and indirectly — the variable is not read off the data at all. Because every trial uses the identical window and bin size, the time axis is a module-level constant `TIME_GRID` computed once from `TIME_WINDOW` and `BIN_SIZE`, and the same 100 values are emitted for every trial of every session. The tie to `stimOn_times` is that the neural bins and the behavioural samples this grid labels are themselves cut around `stimOn_times`. It is `input_names[0] = "time_since_stimulus_onset_s"`, stored as `float16`.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
...
input_trial = np.vstack([TIME_GRID, np.full(N_BINS, trial_num, dtype=np.float32)]).astype(np.float16)
```

iii. Step 5 maps *"Fixed stimulus-aligned time grid → input[0], time since stimulus onset, repeated identically for every trial"*, with the reference for the grid being *"reference interpolation grid"*. The window and bin values are the executable reference's decoding parameters (Step 1).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The AI uses `np.linspace(T_START + BIN, T_STOP, 100)` = **-0.48 … 1.50 s** — i.e. the **right edge** of each 20 ms bin, exactly the grid the reference repo's `interpolate_behavior` builds (`np.linspace(interval_begs[i] + binsize, interval_ends[i], n_bins)`). The human expert instead used bin **centres** (-0.49 … 1.49), so the two differ by a constant 10 ms. The values are cast to `float16` for storage, which makes -0.48 read back as -0.47998.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```
and in the behaviour resampler, the identical expression in absolute time:
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

iii. Step 5, notes column: *"Range is `[-0.48, 1.5]` because the grid stores bin-end times."* No further argument is offered; the choice is inherited from the reference code so that the time input labels exactly the same instants as the resampled behavioural traces.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Bin-for-bin by construction: `input[0][i]` labels the same index *i* as `neural[:, i]`, and both are defined relative to the same `stimOn_times`. The relationship is `neural` bin *i* = spike count over `[-0.5 + 0.02i, -0.5 + 0.02(i+1))` while `input[0][i] = -0.5 + 0.02(i+1)` — the time stamp is the closing edge of the bin it labels, so it leads the bin's centre by 10 ms. That half-bin offset is inherited from the reference repo, whose spike-bin timestamps are documented as left edges while its behaviour grid is bin-end. Crucially the same grid is used for `input[0]`, the wheel and the whisker trace, so all non-neural streams are mutually consistent and none is offset relative to another.

ii.
```python
# neural: bin index = floor((t - t_beg)/0.02) inside bincount2D, t_beg = stimOn - 0.5
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
binned_spikes[interval_idx, idxs_tmp, :] = binned_tmp[:, :n_bins]
```
```python
# input[0] and every behavioural trace share this grid
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
x_interp  = np.linspace(interval_begs[i] + BIN_SIZE, interval_ends[i], n_bins)
```

iii. Step 10, Check 3: *"temporal alignment and binning match the executable reference (`stimOn_times`, `[-0.5, 1.5]`, `20 ms`)."* Step 7 reports the `--show-processing` plots *"show fixed 100-bin trial windows, monotonic time input, raw versus interpolated wheel and whisker traces… No temporal misalignment or discretization anomalies were evident."*

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table alone. The trials table carries no block id, so a block boundary is inferred wherever `probabilityLeft` changes value (compared with `np.isclose`). A NaN `probabilityLeft` both emits NaN and resets the counter.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```

iii. Step 5 describes the source as *"Trial-table `probabilityLeft` block structure"* with transform *"Trial number within the current block"*, tagged as a *"task-specific derivation"* (i.e. not present in the reference code). Step 3 records the task structure it relies on: *"90 unbiased trials first, then biased blocks with `probabilityLeft` values 0.2 or 0.8."*

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop over the raw trial order that emits 1, 2, 3, … within each run of equal `probabilityLeft`, restarting at 1 whenever the value changes. The count is **1-based** (the human reference is 0-based) and is computed on the **unfiltered** trial table, so trials later removed by the quality mask still advance the counter and the number reflects the animal's true position in the block. The scalar is then broadcast across all 100 bins of the trial and stored as `float16`. Observed range in the full dataset: [1, 99] (reference: [0, 98]).

ii.
```python
def compute_trial_number_in_block(probability_left):
    out = np.zeros(len(probability_left), dtype=np.float32)
    prev, count = None, 0
    for idx, value in enumerate(probability_left):
        if np.isnan(value):
            out[idx] = np.nan; prev = np.nan; count = 0; continue
        if prev is None or np.isnan(prev) or not np.isclose(prev, value):
            count = 1
        else:
            count += 1
        out[idx] = count
        prev = value
    return out
```
```python
trial_num = float(block_trial_number[idx])
if not np.isfinite(trial_num):
    raise ValueError("trial_number_in_block is not finite")
input_trial = np.vstack([TIME_GRID, np.full(N_BINS, trial_num, dtype=np.float32)]).astype(np.float16)
```

iii. Step 5: *"Computed on the raw trial order before filtering."* Step 10's sanity-check suite re-derives the block numbering straight from the raw parquet for sessions 0, 1 and 100 and confirms *"input block-trial number"* matches.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward turn), -1 (rightward) or 0 (no response). +1 → 0 ("left"), -1 → 1 ("right"); 0 never reaches the mapper because `exclude_nochoice=True` already removed those trials. Any other value raises.

ii.
```python
def map_choice(choice_value):
    if np.isclose(choice_value, 1.0):  return 0
    if np.isclose(choice_value, -1.0): return 1
    raise ValueError(f"Unexpected choice value {choice_value}")
```
```python
choice_code = map_choice(trials_df.iloc[idx]["choice"])
```

iii. Step 5: *"Trial-table `choice` → output[0], `+1 -> 0` (left), `-1 -> 1` (right)… reference: `bin_behaviors` plus verified sign convention. No-choice trials excluded."* The instructions specify left = 0, right = 1. The full dataset comes out at 50.8 % / 49.2 % (reference 50.7 % / 49.3 %).

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the recode. The per-trial scalar is broadcast to all 100 bins as `uint8` so that `output` is uniformly `(4, 100)`; `output_values[0] = ['left', 'right']`.

ii.
```python
output_trial = np.vstack([
    np.full(N_BINS, choice_code, dtype=np.uint8),
    np.full(N_BINS, prior_code,  dtype=np.uint8),
    discretize(wheel_vals,  wheel_edges),
    discretize(whisker_vals, whisker_edges)])
```

iii. Step 5, Key Decision 6: *"Output representation: Make all outputs time-varying so the decoder sees a uniform `(n_output, n_timepoints)` target structure"* — consistent with the instruction *"If at all possible, make it time-varying."*

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, recoded 0.2 → 0, 0.5 → 1, 0.8 → 2 exactly as the instructions specify. Anything else raises. The unbiased 0.5 block at the start of each session is retained (`exclude_unbiased=False`), giving the expected three-class distribution 0.419 / 0.141 / 0.440 (reference 0.418 / 0.141 / 0.442).

ii.
```python
def map_prior(prob_left):
    if np.isclose(prob_left, 0.2): return 0
    if np.isclose(prob_left, 0.5): return 1
    if np.isclose(prob_left, 0.8): return 2
    raise ValueError(f"Unexpected probabilityLeft value {prob_left}")
```

iii. Step 5: *"Trial-table `probabilityLeft` → output[1], `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`… This is the requested prior output."* Step 9's consistency table checks the realised distribution against the expectation that only `{0.2, 0.5, 0.8}` occur.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recode; the scalar is broadcast across the 100 bins as `uint8`, with `output_values[1] = ['0.2', '0.5', '0.8']`.

ii.
```python
prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
...
np.full(N_BINS, prior_code, dtype=np.uint8)
```

iii. Same as 5-b: all outputs are made time-varying for a uniform target tensor.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The AI calls the reference repo's own `brainbox.behavior.wheel` functions on them rather than going through `SessionLoader`, then takes the absolute value of the filtered velocity — the same definition the reference's `load_target_behavior('wheel-speed')` uses.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
...
timestamps = np.load(ts_path)
position   = np.load(pos_path)
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

iii. Step 3: *"Wheel speed is the magnitude of wheel velocity."* Step 4: *"Code uses absolute wheel velocity… Papers describe the same behavior sources → Consistent."* Step 5 cites `load_target_behavior` / `get_behavior_per_interval` as the reference functions being reproduced.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps. (1) The wheel is only sampled when it turns, so `interpolate_position(..., freq=1000)` puts position on an even 1 kHz grid. (2) `velocity_filtered(pos, 1000)` differentiates it through an 8th-order Butterworth low-pass with a 20 Hz corner — these are the shipped defaults and are identical to what `SessionLoader.load_wheel` applies, so the trace equals the reference's. (3) `np.abs` gives speed in rad/s. (4) The trace is resampled per trial onto the 100-point grid with `scipy.interpolate.interp1d(kind='linear', fill_value='extrapolate')` restricted to the samples strictly inside the window, then (5) discretized (see 7-c). Everything is carried in `float16` from the session cache onward.

ii.
```python
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)   # defaults: linear
vel, _ = velocity_filtered(pos_interp, 1000)                                   # defaults: 20 Hz, order 8
```
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
y_interp = np.asarray(y_interp, dtype=np.float32).reshape(-1)
if y_interp.shape[0] != N_BINS or np.any(~np.isfinite(y_interp)):
    vals_list.append(None); mask.append(False); continue
```

iii. Step 5 maps *"Wheel timestamps + position → output[2]: Interpolate absolute wheel velocity to the stimulus-aligned 20 ms grid"* via `load_target_behavior` / `get_behavior_per_interval`. Step 10, Check 3: *"behavior construction matches absolute wheel velocity and left-whisker-first fallback."* The `interp1d`-onto-`linspace` resampler is a direct port of the reference `interpolate_behavior`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 classes at **global** tertiles: after every session is processed, all wheel-speed samples from all 444 sessions × all trials × all 100 bins are concatenated into one vector, the 1/3 and 2/3 quantiles of that vector are taken, and the *same* two edges (0.01511, 0.40503 rad/s) are applied to every session via `np.digitize`. `safe_quantile_edges` guards the degenerate case where the two quantiles coincide by falling back to equal thirds of the value range. The edges are recorded in `metadata['wheel_speed_quantile_edges']`. The human reference instead takes percentiles **within each session**. Empirically the global choice is benign here: the overall split is 0.3333/0.3333/0.3334 and no session has a class above 0.80 or below 0.02.

ii.
```python
SPLIT via quantiles over the concatenated dataset:
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)

def safe_quantile_edges(values):
    q1, q2 = np.quantile(values, [1.0/3.0, 2.0/3.0])
    if q1 >= q2:
        vmin, vmax = float(np.min(values)), float(np.max(values))
        if np.isclose(vmin, vmax): q1, q2 = vmin + 1e-6, vmax + 2e-6
        else:                      q1, q2 = vmin + (vmax-vmin)/3.0, vmin + 2.0*(vmax-vmin)/3.0
    return np.asarray([q1, q2], dtype=np.float32)

def discretize(values, edges):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)
```

iii. Step 5, Key Decision 5: *"Behavior discretization: Use global tertile edges so wheel and whisker bins have consistent semantics across sessions."* This drove the two-pass architecture (trajectory step 131: *"a two-pass pipeline: first build stimulus-aligned per-trial neural and continuous behavioral arrays with the reference filters, then compute global tertile thresholds"*). No per-session alternative is discussed anywhere in the notes or trajectory.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at exactly the 100 grid points that `input[0]` labels, measured from the same `stimOn_times`, so index *i* of the wheel output corresponds to index *i* of the neural matrix. The wheel clock is the IBL session clock, the same one the spike times are on, so no extra synchronisation is done. A trial whose window is not spanned by wheel samples to within one bin at each edge is dropped rather than extrapolated (see 1-e). As in 3-c, the sample sits on the closing edge of the neural bin it is paired with rather than at its centre.

ii.
```python
interval_begs = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0]
interval_ends = trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1]
...
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. Step 5/Step 10: alignment for every stream is *"stimulus-aligned interpolation"* to the reference grid; Step 7's processing plots were inspected specifically for misalignment and *"No temporal misalignment… anomalies were evident."*

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with frame times `_ibl_leftCamera.times.npy`; if either is absent the right camera (`rightCamera.ROIMotionEnergy.npy` + `_ibl_rightCamera.times.npy`) is used instead. If neither exists a `FileNotFoundError` is raised and the session is dropped — this accounts for 14 of the 15 dropped sessions. The released ROI trace (one value per video frame over the whisker-pad ROI) is used as published; which camera a session used is not recorded in the final metadata.

ii.
```python
def load_whisker_motion_energy(one, row):
    sides = [("left",  "_ibl_leftCamera.times.npy",  "leftCamera.ROIMotionEnergy.npy"),
             ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy")]
    errors = []
    for side, times_name, energy_name in sides:
        try:
            times_path  = ensure_dataset_path_any(one, row, [f"alf/{times_name}", f"alf/#*/{times_name}"], times_name, "alf")
            energy_path = ensure_dataset_path(one, row, f"alf/#*/{energy_name}", energy_name, "alf")
            return {"times": np.load(times_path).astype(np.float32),
                    "values": np.load(energy_path).astype(np.float32)}, side
        except Exception as exc:
            errors.append(f"{side}: {exc}"); continue
    raise FileNotFoundError(f"No whisker motion energy trace available for session {row.eid} ...")
```

iii. Step 3: *"Whisker motion energy is taken from the left camera when available, otherwise the right camera."* Step 1 attributes the preference order to the reference `bin_behaviors`, which tries `'left-whisker-motion-energy'` and falls back to `'right-whisker-motion-energy'` on skip.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None applied to the released trace — no filtering, no per-session normalisation, no baseline subtraction. It is resampled per trial onto the same 100-point grid by the same `get_behavior_per_interval` routine used for the wheel (same coverage and NaN checks, same `interp1d` linear resampler), kept in `float16`, and then discretized (8-c).

ii.
```python
whisker, whisker_source = load_whisker_motion_energy(one, row)
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False)
```

iii. Step 5: *"Left/right whisker motion energy → output[3]: Interpolate to the stimulus-aligned 20 ms grid"*, reference functions `load_target_behavior` / `get_behavior_per_interval`. The AI records no additional processing because the reference performs none.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: 3 classes split at the **global** 1/3 and 2/3 quantiles of every whisker sample pooled across all 444 sessions (edges 2.789 and 7.844), applied uniformly to every session. The human reference instead uses each session's own percentiles. Because ROI motion energy is in arbitrary, camera- and lighting-dependent units, this makes the class boundaries mean different things in different sessions: pooled over the dataset the split is 0.333/0.333/0.334, but **per session** 24 of 444 sessions put >90 % of their samples in a single class, 35 exceed 80 %, 163 exceed 60 %, and 40 sessions have a class holding <0.05 % of their samples. The human reference has exactly 1/3 in every class of every session.

ii.
```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
print(f"Global whisker edges: {whisker_edges.tolist()}", flush=True)
...
discretize(whisker_vals, whisker_edges)
```
```python
"whisker_motion_energy_quantile_edges": whisker_edges.tolist(),
```

iii. Step 5, Key Decision 5 again: *"Use global tertile edges so wheel and whisker bins have consistent semantics across sessions."* Step 9's consistency table checks only the pooled distribution (`[0.333293, 0.333161, 0.333545]` → "Yes") and never examines the per-session distributions, so the imbalance was not surfaced. Step 12 reports the whisker decoder as the best-performing output (0.7373 validation balanced accuracy) and treats that as confirmation rather than investigating whether session-level class imbalance is contributing.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Exactly as the wheel: the camera frame times are on the same session clock as the spikes, so the trace is simply evaluated at the same 100 grid points measured from the same `stimOn_times`, index for index with the neural matrix. Trials whose window is not spanned by camera frames to within one bin at each edge, or that contain a NaN, are dropped.

ii.
```python
whisker_values, whisker_mask = get_behavior_per_interval(whisker["times"], whisker["values"], trials_df, allow_nans=False)
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
```
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. Step 10, Check 3 lists temporal alignment as matching the executable reference for all streams. The diagnostic plots overlay the raw camera trace, the interpolated samples and the discretization thresholds against a stimulus-onset-zeroed axis for visual confirmation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered, and mostly by dropping.
- **Missing/NaN trial events** — removed by the trial mask's `.isnull()` clauses.
- **Missing behavioural coverage** — a trial whose window is not spanned by the wheel or camera to within one bin, or that contains a NaN sample, is masked out (`allow_nans=False`).
- **NaN cluster QC label** — `fillna(0)`, so the unit fails `>= 1` and is dropped.
- **A probe with zero good units** — skipped, and only that probe; the session continues on its other probe. The notes record this as a bug found and fixed in Step 10 that *"recovered one otherwise-lost session."*
- **A session with no good probe, no whisker trace, or fewer than 2 usable trials** — raises inside the worker, is caught in `main`, logged to `skipped`, and omitted (15 sessions: 14 missing whisker, 1 with zero overlapping trials).
- **Versioned ALF revisions** — `sorted(...)[-1]` takes the newest.
- **Missing files on disk** — one download attempt via ONE before giving up.
- **Overflow guard** — a spike count above 255 would raise rather than silently wrap.
- **Fragility**: `map_choice` / `map_prior` *raise* on an unexpected `choice` / `probabilityLeft` value, which aborts the entire session instead of dropping the one trial. Unlike the human reference there is no `probabilityLeft.isin({0.2,0.5,0.8})` mask, so these mappers are the only guard. It never fired on this release.

ii.
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
if good_cluster_ids.size == 0:
    return None, None
...
for probe_row in probe_df.itertuples(index=False):
    spikes, clusters = load_probe_data(one, row, probe_row.probe_name, brain_regions)
    if spikes is None or clusters is None:
        print(f"  skipping probe {probe_row.probe_name}: no well-isolated clusters", flush=True)
        continue
if not spikes_list:
    raise RuntimeError(f"No probes with well-isolated clusters for session {row.eid}")
```
```python
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```
```python
except Exception as exc:
    return {"ok": False, "eid": row.eid, "error": str(exc), "traceback": traceback.format_exc()}
```

iii. Step 10: *"Edge-case review: fixed empty-good-unit probe behavior so the probe is skipped instead of the session; handled revisioned ALF paths robustly; kept explicit session drops for missing whisker traces and the single zero-valid-trial session."* Step 9 quantifies the loss and shows it is fully accounted for: *"Those 15 skipped sessions contain exactly 2,664 good units in raw data"* (75,708 − 2,664 = 73,044 retained). Step 10 also states the whisker-missing sessions were *"Left unresolved by design because the requested decoder output includes whisker motion energy and the raw data are absent for those sessions."*

## 10-a. What are the most time-consuming steps of the code?

i. Raw I/O dominates: reading `spikes.times.npy` and `spikes.clusters.npy` for each of the 699 probe insertions (hundreds of MB per probe), then the per-trial `bincount2D` binning loop, then writing the 3.1 GB output pickle. The AI instruments the code with per-session `time.time()` prints and reports 429.35 s for the 444-session cache fill with 4 parallel workers (~3.9 s/session of wall clock), and 15.75 s for the second assemble-from-cache pass. The AI's own mitigation — caching one pickle per session so that *"expensive raw loading happens once"* — identifies raw loading as the step it considered dominant. Note that the session cache also *adds* I/O: each session's arrays are written to disk and read back, and `--repack-session-cache` is a further full read-modify-write pass over all 444 records.

ii.
```python
session_start = time.time()
...
print(f"  session time: {time.time() - session_start:.2f}s", flush=True)
```
```python
spike_times    = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
```
```python
max_workers = min(max(args.max_workers, 1), os.cpu_count() or 1, len(session_rows))
with ProcessPoolExecutor(max_workers=max_workers) as pool:
```

iii. Step 6: *"Code speedups added: Session-level parallel processing in full mode. Cached per-session intermediates so expensive raw loading happens once."* Step 7's timing table reports *"Full cache fill: 429.35 s for 444 cached sessions — observed full pass stayed well under 15 minutes,"* meeting the instruction's 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four.
1. **The per-trial spike-binning loop** in `get_spike_data_per_interval` — one `bincount2D` call plus one `np.intersect1d` per trial. The whole session could be done with a single `np.bincount` over a flat `(trial, unit, bin)` index, as the human reference does. The per-trial `np.intersect1d(cluster_ids, cluster_idxs)` is pure overhead: `cluster_ids` is already `np.unique(clusters)`, so a precomputed lookup would replace it.
2. **The per-trial behaviour loop** in `get_behavior_per_interval` — it builds a fresh `scipy.interp1d` object per trial per stream (2 × n_trials objects per session). One `np.interp` over a single concatenated query vector would do the whole session.
3. **`compute_trial_number_in_block`** — a pure-Python loop over every trial; the two-line pandas idiom `(pL != pL.shift()).cumsum()` + `groupby(...).cumcount()` is the vectorized equivalent.
4. **The per-trial assembly loops** in `process_session` and `build_dataset`, and the per-neuron `for region in rec.cluster_regions_beryl.tolist()` loop in `build_dataset`, which could be a single `np.searchsorted` / factorize.
   Separately, `np.isin(spike_clusters, good_cluster_ids)` sorts and searches over the full ~10⁸-element spike array; the boolean-lookup form `is_good[spike_clusters]` (already implied by the `remap` table built two lines later) is substantially cheaper.

ii.
```python
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
    _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
```
```python
for interval_idx, (ib, ie) in enumerate(zip(idxs_beg, idxs_end)):
    ...
    y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```
```python
for idx, value in enumerate(probability_left):
    ...
    if prev is None or np.isnan(prev) or not np.isclose(prev, value):
```

iii. The AI does not identify any of these. CONVERSION_NOTES Step 6 lists under *"Code inefficiencies identified"* only *"Full verification logs are huge…"* and *"Full dataset training with float32 neural storage is memory-expensive"* — both about the verifier and the trainer, not the conversion loops. Its speed-ups were process-level (parallelism, caching, compact dtypes) rather than vectorization, and since the 429 s runtime met the stated budget it did not revisit them.

## 10-c. What processing does the code repeat multiple times?

i. Several things.
- **Binning and resampling of discarded trials.** `bin_spiking_data` and both `get_behavior_per_interval` calls run over the *entire* unfiltered `trials_df`; `valid_idx` is only applied afterwards. Roughly 36 % of trials (296,090 raw → 188,925 kept) are fully binned and interpolated and then thrown away, including trials with NaN `stimOn_times`.
- **Double transposition** of the neural array: `bin_spiking_data` transposes every trial to `(n_bins, n_clusters)`, and `process_session` transposes each one straight back to `(n_clusters, n_bins)`.
- **Session round-trip through disk.** Every `SessionRecord` is pickled to `cache/session_records/`, then re-read by `main` in the same run (`with Path(result["record_path"]).open("rb") as f: indexed_records[idx] = pickle.load(f)`), and `--repack-session-cache` can re-read and re-write all 444 of them again.
- **Repeated filesystem globbing**: `locate_dataset` globs up to two cache roots for each of ~10 datasets per session.
- **Per-worker re-instantiation** of `ONE(...)` and `BrainRegions()` inside `process_session_worker`.

ii.
```python
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)        # all trials
wheel_values,  wheel_mask  = get_behavior_per_interval(..., trials_df, allow_nans=False)
whisker_values, whisker_mask = get_behavior_per_interval(..., trials_df, allow_nans=False)
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)                               # only now
```
```python
binned_trials = np.asarray([x.T for x in binned_array], dtype=np.float32)   # transpose
...
neural_trial = compact_neural_trial(binned_spikes[idx].T)                   # transpose back
```
```python
def process_session_worker(row_dict, probe_records, with_diagnostic, record_cache_dir=None):
    one = build_one()
    brain_regions = BrainRegions()
```

iii. The AI documents the session cache as a deliberate trade-off (Step 6: *"Added per-session cache files in `cache/session_records/`"*, *"Cached per-session intermediates so expensive raw loading happens once"*), which is what made the interrupted full run resumable via `--resume-session-cache`. It does not mention the unfiltered binning, the double transpose or the repeated globbing anywhere.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- `clusters.depths.npy` is loaded and attached as a `depths` column that is never read; the whole `clusters.metrics.pqt` table is carried when only `label` and `cluster_id` are used.
- The continuous `float16` wheel and whisker traces are stored on every `SessionRecord` and re-concatenated for the global quantiles, but after `discretize` only the 3-level codes reach `converted_data.pkl`.
- `SessionRecord.lab`, `.trial_indices` and `.trial_numbers_in_block` are kept per session; only `lab` reaches the output (as `metadata['session_labs']`), the other two are dropped.
- The `diagnostic` dict for the first two sessions carries the *entire* raw wheel and whisker traces for the session plus a copy of a neural trial; used only for the `--show-processing` PNGs.
- `compact_session_record` / `--repack-session-cache` is a whole extra pass that produces nothing the final pickle needs.
- `metadata['input_time_grid_s']` duplicates information already in `input[0]` of every trial.
- ~36 % of binned/interpolated trials are discarded (see 10-c).
- The neural array is materialised as `float32` by `bincount2D` and `np.zeros(..., dtype=np.float32)` and then cast down to `uint8` per trial.

ii.
```python
cluster_depths = np.load(cluster_depths_path).astype(np.float32)
clusters_df["depths"] = cluster_depths          # never used again
```
```python
diagnostic = {"neural": neural_trials[0], "time_input": input_trials[0][0],
              "wheel_raw_times": wheel["times"], "wheel_raw_values": wheel["values"],
              "whisker_raw_times": whisker["times"], "whisker_raw_values": whisker["values"], ...}
```
```python
binned_spikes = np.zeros((len(interval_begs), n_clusters, n_bins), dtype=np.float32)
...
return neural_trial.astype(np.uint8, copy=False)
```

iii. The AI justifies the parts it is aware of. Step 6 on compact dtypes: *"Compact storage (`uint8` neural, `float16` inputs/continuous traces) to reduce `converted_data.pkl` to 3.118 GB."* Step 7 on the diagnostics: the `--show-processing` plots were required by the instructions and *"show… raw versus interpolated wheel and whisker traces; discretization thresholds overlaid on continuous traces."* The continuous behavioural traces must be retained across sessions precisely because the global tertile edges cannot be computed until every session is done. The unused `depths` column, the duplicate time grid and the unfiltered binning are not mentioned.
