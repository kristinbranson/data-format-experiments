# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API to discover or resolve data. Instead it takes the session roster straight from the reference code repository's release table, `code/code_zhang2025/data/bwm_release.csv` (459 sessions, 699 insertions, 139 mice), de-duplicating it to one row per `eid` and attaching the list of `(pid, probe_name)` pairs for that session. Every session's files are then read directly off the local ONE cache tree at a deterministically reconstructed path, `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number:03d>/alf/...`. Within a session it reads:
- trials: `alf/**/_ibl_trials.table.pqt` (parquet)
- wheel: `alf/**/_ibl_wheel.timestamps.npy`, `alf/**/_ibl_wheel.position.npy`
- whisker: `alf/**/_ibl_{left,right}Camera.times.npy`, `alf/**/{left,right}Camera.ROIMotionEnergy.npy`
- spikes: `alf/<probe>/pykilosort/**/spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, `channels.brainLocationIds_ccf_2017.npy`

Dataset revisions (the `#YYYY-MM-DD#` sub-directories) are resolved by `find_latest_file`, which globs and takes the lexicographically last match. A `ONE` client is still constructed (`build_one`) but is not used on the hot path. Sessions are processed in parallel with a `ProcessPoolExecutor` (12 workers, chosen by benchmark), with a per-worker 3-attempt retry for transient errors and a final sequential retry pass for any session whose worker failed. Result: 438 of 459 sessions converted, 135 subjects, 186,261 trials, 72,757 neurons.

ii.
```python
BWM_RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
ONE_CACHE_DIR = ROOT / "data" / "one_cache"

def load_release_sessions() -> tuple[pd.DataFrame, pd.DataFrame]:
    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    sessions_df = (
        bwm_df[["eid", "lab", "subject", "date", "session_number"]]
        .drop_duplicates(subset=["eid"], keep="first")
        .reset_index(drop=True)
    )
    probe_map = (
        bwm_df[["eid", "pid", "probe_name"]] ... .groupby("eid").apply(...).to_dict()
    )
    sessions_df["probe_info"] = sessions_df["eid"].map(probe_map)
    return bwm_df, sessions_df
```

```python
def release_session_path(row: pd.Series) -> Path:
    return (ONE_CACHE_DIR / row["lab"] / "Subjects" / row["subject"]
            / row["date"] / f"{int(row['session_number']):03d}")

def find_latest_file(base_dir: Path, pattern: str) -> Path:
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {base_dir}")
    return matches[-1]
```

```python
with ProcessPoolExecutor(max_workers=args.session_workers) as executor:
    future_map = {executor.submit(process_one_session_worker, row_dict, time_axis): row_dict
                  for row_dict in records}
```

iii. CONVERSION_NOTES Step 4/5: "*Treat the 459-session `bwm_release.csv` used by the reference code as the canonical release roster for this workspace*" and Key Decision 1: "*Start from the 459-session `bwm_release.csv` used by the reference code, then require local availability of spikes, trials, wheel, and whisker motion energy*". Step 6 justifies the direct-file path on performance grounds: "*Remote ONE/SessionLoader access introduced large startup latency and transient worker failures during full conversion attempts despite the release sessions already being cached locally*" and "*Reduced sample conversion from 132.5 s to 12.1 s while preserving counts and validation results*". Step 6 also: "*Reused probe IDs directly from `bwm_release.csv` instead of per-session Alyx `eid2pid` lookups*".

## 1-b. How are the data split into subjects?

i. The subject identity is read from the `subject` column of `bwm_release.csv` — it is not parsed out of file paths or inferred. Each `ProcessedSession` carries `subject`, and at assembly the unique subject names are collected **in session order** (`ordered_unique`, not sorted) and `subject_idx` is the index of each session's subject into that list. 135 subjects across the 438 converted sessions.

ii.
```python
sessions_df = (bwm_df[["eid", "lab", "subject", "date", "session_number"]]
               .drop_duplicates(subset=["eid"], keep="first").reset_index(drop=True))
```

```python
def ordered_unique(strings: list[str]) -> list[str]:
    seen = set(); out = []
    for item in strings:
        if item not in seen:
            seen.add(item); out.append(item)
    return out

subjects = ordered_unique([session.subject for session in processed_sessions])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.subject])
```

iii. CONVERSION_NOTES Step 5 mapping table: "*`subject` from `bwm_release.csv` and session path metadata → `subjects`, `subject_idx`; Build unique subject list and index sessions in converted order*", with the note "*Session order will be fixed and deterministic in the conversion script*" (enforced by `processed_sessions.sort(key=lambda s: s.release_index)` before assembly, so the parallel completion order does not leak into the output).

## 1-c. How are the data split into sessions?

i. A session is the unit of the release table: one row per `eid` after `drop_duplicates(subset=["eid"])` (the raw CSV has one row per probe insertion). Nothing has to be split — the `eid` defines the session, and the associated `(pid, probe_name)` pairs define which probes belong to it. Session order in the output is the release-CSV order, restored after parallel processing by sorting on `release_index`.

ii.
```python
sessions_df = sessions_df.reset_index(names="release_index")
...
processed_sessions.sort(key=lambda session: session.release_index)
```

```python
probe_map = (bwm_df[["eid", "pid", "probe_name"]].astype({"pid": str, "probe_name": str})
             .groupby("eid").apply(lambda frame: [(str(pid), str(probe_name))
                 for pid, probe_name in zip(frame["pid"], frame["probe_name"])],
                 include_groups=False).to_dict())
sessions_df["probe_info"] = sessions_df["eid"].map(probe_map)
```

iii. CONVERSION_NOTES Step 4 resolves the roster question explicitly (459-session release CSV is canonical, the newer 480-session local cache table is ignored), and Step 2 documents the on-disk layout `data/one_cache/<lab>/Subjects/<subject>/<YYYY-MM-DD>/<session_number>/alf/` from which a session path is reconstructed.

## 1-d. How are the data split into trials?

i. The split is given by the data: each row of `_ibl_trials.table.pqt` is one trial. The table is read with `pd.read_parquet` and `reset_index(drop=True)`, so trial index = row index. Trial boundaries for the converted data are *not* the IBL `intervals_0/intervals_1`; every trial is a fixed 2 s window `stimOn_times + [-0.5, 1.5]`, so trials can (and do) overlap adjacent trials' raw intervals. A per-trial boolean `mask` (see 1-e) selects which rows survive; all per-trial variables are computed on the full table first and indexed by `keep_idx` afterwards.

ii.
```python
trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
trials = pd.read_parquet(trials_path).copy()
...
return trials.reset_index(drop=True), mask
```

```python
intervals = np.vstack([trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
                       trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1]]).T
```

iii. CONVERSION_NOTES Step 1 documents the reference behaviour it is reproducing: "*per-trial intervals are computed from `stimOn_times + (-0.5, 1.5)`*", and Step 4 records the decision to keep the reference code's single common `stimOn`-aligned 2 s grid for all four outputs because the task states "Temporally align based on stimulus onset".

## 1-e. How are trials filtered based on quality controls?

i. Two masks are ANDed together.

**(a) The reference code's trial mask**, reimplemented verbatim as `load_trials_and_mask_current` with the same defaults `prepare_data` uses (`min_rt=0.08`, `max_rt=2.0`, `max_trial_len=10.0`, `exclude_nochoice=True`). A trial is **rejected** if any of:
- `firstMovement_times - stimOn_times < 0.08 s` (too fast)
- `firstMovement_times - stimOn_times > 2.0 s` (too slow)
- `feedback_times - goCue_times > 10 s` (trial too long)
- any of `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType` is null
- `choice == 0` (no response)

**(b) Behavioural coverage.** A trial is additionally dropped if either the wheel trace or the whisker trace does not span the full `[-0.5, 1.5]` s window: no samples in the window, the first sample more than one bin (20 ms) after the window start, or the last sample more than one bin before the window end. This is the reference code's `get_behavior_per_interval` validity test and it is what removes trials at the edges of a recording and in camera dropouts.

A session that ends up with fewer than 2 valid trials raises and the session is dropped (21 sessions, all reported as "Only 0 valid trials after filtering"). Net effect on the full run: 186,261 trials kept, ~425 per session.

ii.
```python
def load_trials_and_mask_current(session_path, min_rt=0.08, max_rt=2.0,
                                 max_trial_len=10.0, exclude_nochoice=True):
    query_parts = []
    if min_rt is not None:
        query_parts.append(f"(firstMovement_times - stimOn_times < {min_rt})")
    if max_rt is not None:
        query_parts.append(f"(firstMovement_times - stimOn_times > {max_rt})")
    if max_trial_len is not None:
        query_parts.append(f"(feedback_times - goCue_times > {max_trial_len})")
    for event in ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                  "firstMovement_times", "feedbackType"]:
        query_parts.append(f"{event}.isnull()")
    if exclude_nochoice:
        query_parts.append("(choice == 0)")
    mask = ~trials.eval(" | ".join(query_parts)).to_numpy()
```

```python
    if np.abs(interval_begs[idx] - t_seg[0]) > binsize:
        reasons[idx] = "target data starts too late"; continue
    if np.abs(interval_ends[idx] - t_seg[-1]) > binsize:
        reasons[idx] = "target data ends too early"; continue
```

```python
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
keep_idx = np.flatnonzero(keep_mask)
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. CONVERSION_NOTES Step 4: "*Reproduce the reference code trial mask exactly, including `max_trial_len=10.0`, because it is the executable implementation provided for this project*", noting that the papers state the missing-event and 0.08–2.0 s RT exclusions but not the 10 s rule. Step 10 confirms the 21 dropped sessions are genuine: "*all 21 consistently failed with `Only 0 valid trials after filtering`, confirming these are true exclusions rather than multiprocessing artifacts*".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike raster itself comes from exactly two arrays per probe, `spikes.times.npy` and `spikes.clusters.npy`. Three further per-probe files are used only to decide which clusters survive and where they are:
- `clusters.metrics.pqt` → the `label` column (IBL spike-sorting QC score) and `cluster_id`
- `clusters.channels.npy` → peak channel of each cluster
- `channels.brainLocationIds_ccf_2017.npy` → Allen CCF region id of each channel, mapped to an acronym via `BrainRegions.id2acronym` and later to Beryl

`spikes.clusters` is re-indexed to the row number of the surviving cluster table, and `merge_probes` (imported from the reference repo, `utils.ibl_data_utils`) pools the probes of a session into one population with a continuous cluster numbering and a time-sorted merged spike train.

ii.
```python
spikes = {"times": np.load(sort_dir / "spikes.times.npy"),
          "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32)}
clusters_labeled = pd.read_parquet(sort_dir / "clusters.metrics.pqt")
cluster_channels = np.load(sort_dir / "clusters.channels.npy").astype(np.int64)
channel_region_ids = np.load(sort_dir / "channels.brainLocationIds_ccf_2017.npy").astype(np.int64)
good_channel_idx = np.clip(cluster_channels, 0, len(channel_region_ids) - 1)
cluster_region_ids = channel_region_ids[good_channel_idx]
clusters_labeled["acronym"] = BRAIN_REGIONS.id2acronym(cluster_region_ids)
```

```python
from utils.ibl_data_utils import merge_probes
...
spikes, clusters = merge_probes(spikes_list, clusters_list)
neural_dict = {"spike_times": spikes["times"],
               "spike_clusters": spikes["clusters"],
               "cluster_regions": clusters["acronym"].to_numpy()}
```

iii. CONVERSION_NOTES Step 1: "*Reference code is for electrophysiology, not imaging. Neural data are spike times and spike-sorted cluster metadata; there is no delta-F/F computation anywhere in the pipeline*", and `merge_probes` is listed as the reference function that "*Merges spikes and clusters across probes in a session, reindexes clusters, and sorts merged spikes by time*". Step 5 records that the datapaper "*combined neurons across probes within a session rather than decoding each probe independently*".

## 2-b. How is the `neural` data processed?

i. Merged spikes are binned into **spike counts** (not converted to Hz) on a 20 ms grid spanning `stimOn_times + [-0.5, 1.5]`, using the reference code's own routine (`get_spike_data_per_interval` → `bincount2D` from `iblutil.numerical`). For each trial the spikes inside the interval are selected by a boolean mask, `bincount2D` histograms them into a `(clusters_present × bins)` array, and the rows are scattered back into the full `(n_clusters × n_bins)` grid via `np.intersect1d`. The result is truncated to exactly 100 bins, transposed to the reference's `(n_bins, n_clusters)` cached layout, and then transposed again at output time to the required `(n_neurons, n_timepoints)`. Stored as `float32`. **No smoothing, no z-scoring and no normalisation** is applied — the reference code's z-scoring (`standardize_spike_data`) lives in the decoder, not the conversion.

ii.
```python
def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends,
                                interval_len, binsize):
    n_bins = int(np.ceil(interval_len / binsize))
    cluster_ids = np.unique(clusters)
    binned_spikes = np.zeros((n_intervals, len(cluster_ids), n_bins), dtype=np.float32)
    for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr, clust_curr = times[idxs_t], clusters[idxs_t]
        if times_curr.shape[0] == 0:
            continue
        binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr,
                                                 xbin=binsize, xlim=[t_beg, t_end])
        _, target_indices, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
        binned_spikes[interval_idx, target_indices, :] = binned_tmp[:, :n_bins]
    return binned_spikes
```

```python
return np.array([x.T for x in binned_array], dtype=np.float32)     # (n_trials, n_bins, n_clusters)
...
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
```

iii. CONVERSION_NOTES Step 5 mapping: "*Merge probes per session, keep well-isolated clusters with `label >= 1`, apply trial mask, bin spike counts in `stimOn_times + [-0.5, 1.5]` using `0.02 s` bins, store per trial as `(n_neurons, 100)` after transposing from reference `(100, n_neurons)`*" with the note "*This follows the paper's neuron curation while keeping the reference loading/binning machinery*". Step 1 records that standardisation is a decoder-side step (`standardize_spike_data` "*Z-scores spike counts across trials separately for each time bin and neuron before decoding*"), so it is deliberately not applied during conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single cut: clusters with `label >= 1` in `clusters.metrics.pqt` are kept, everything else is discarded, applied per probe **before** merging. `label` is the IBL 0 / ⅓ / ⅔ / 1 spike-sorting QC score, so `>= 1` keeps only units passing every metric — the data paper's "well-isolated neurons". The surviving cluster table is re-indexed to 0..n-1 and `spikes.clusters` is remapped with `ismember` so spike ids stay consistent. A probe with zero surviving clusters is skipped; a session with no surviving clusters raises and is dropped.

No anatomical filter is applied: units whose Beryl acronym is `root` (10,028 neurons) **and** units whose acronym is `void` (250 neurons — channels the histology placed outside the brain) are both retained, so the exported `brain_regions` list contains 266 labels including `void`.

Result: 72,757 neurons across 438 sessions, mean 166.1 per session (data paper: 75,708 well-isolated units across 459 sessions ⇒ ~165/session).

ii.
```python
def load_spiking_data_current(session_path, probe_name, qc=None):
    ...
    iok = clusters_labeled["label"] >= qc
    selected_clusters = clusters_labeled[iok].copy()
    spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
    selected_clusters.reset_index(drop=True, inplace=True)
    selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    selected_spikes["clusters"] = selected_clusters.index.to_numpy()[ib].astype(np.int32)
    return selected_spikes, selected_clusters
```

```python
spikes, clusters = load_spiking_data_current(session_path, probe_name=probe_name, qc=1)
if len(clusters) == 0:
    continue
...
if not spikes_list:
    raise RuntimeError("No good clusters after QC filtering")
```

iii. CONVERSION_NOTES Step 4 discrepancy table resolves the code-vs-paper conflict explicitly: the reference `prepare_data` "*loads all spike-sorted clusters; it stores `good_clusters = label >= 1` in metadata but does not filter them out before binning*", while the data paper "*report[s] 75,708 well-isolated neurons and explicitly describe[s] neuron QC*". Resolution: "*Use the code's stored QC label to filter to `label >= 1` clusters during conversion. This preserves the paper's curation intent while still relying on the provided code path and metadata definitions.*" Step 9 checks the outcome against the paper: converted mean 166.11 neurons/session vs the ~165/session implied by 75,708/459. There is no recorded justification for retaining `void` units; Step 9 only notes the region count is "*266 (264 excluding `root`/`void`)*" against the method paper's 270.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **stimulus onset**, `trials.stimOn_times`, which is the `align_time` parameter taken from the reference caching script. All IBL streams (spikes, trial events, wheel, camera) are already expressed in seconds on one synchronised session clock, so alignment is purely a window construction: for every trial the interval `[stimOn - 0.5, stimOn + 1.5]` is formed and `bincount2D` is called with `xlim=[t_beg, t_end]`, so bin *k* covers `[stimOn - 0.5 + 0.02k, stimOn - 0.5 + 0.02(k+1))`. No time-base correction, resampling or cross-stream interpolation of the spikes is performed. Metadata records `temporal_alignment_event = "stimulus onset (stimOn_times)"`, `off_start = -0.5`, `off_end = 1.5`.

ii.
```python
PARAMS = {"interval_len": 2.0, "binsize": 0.02,
          "align_time": "stimOn_times", "time_window": (-0.5, 1.5)}
```

```python
intervals = np.vstack([trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
                       trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1]]).T
...
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr,
                                         xbin=binsize, xlim=[t_beg, t_end])
```

iii. CONVERSION_NOTES Step 4: the methods paper uses target-specific alignment (prior from a −0.6 to −0.1 s pre-stimulus window; wheel/whisker aligned to first movement), but "*Because the user explicitly requires 'Temporally align based on stimulus onset' and a single common dataset with simultaneous outputs, prioritize the executable reference code's common `stimOn`-aligned 2 s / 20 ms representation, while documenting that this is the main paper-vs-task-driven deviation*". Step 10 Check 2 verified the alignment against raw data: an independent histogram of raw spike times for 3 QC-passed neurons in the first kept trial of session `6713a4a7…` "*matched exactly*".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, `n_bins = ceil(2.0 / 0.02) = 100` per trial, identical for every trial and every session. This is the native binning — spikes are histogrammed once directly onto the 20 ms grid, so there is **no** rebinning, downsampling or re-binning of an intermediate resolution. `time_bin_size` is written into metadata as `20.0` (ms). The behavioural streams are not rebinned either; they are linearly interpolated from their native rates (1000 Hz for the wheel after `interpolate_position`, ~60 Hz for the camera) onto the same 100 sample points.

ii.
```python
PARAMS = {"interval_len": 2.0, "binsize": 0.02, ...}
n_bins = int(np.ceil(interval_len / binsize))            # 100
```

```python
"time_bin_size": 20.0,
"off_start": float(PARAMS["time_window"][0]),
"off_end": float(PARAMS["time_window"][1]),
```

iii. CONVERSION_NOTES Step 4 flags a paper-vs-code conflict — the methods text says "*Within each trial, we segment neural activity into 50-ms non-overlapping time bins*" for choice/prior, while "*Recordings are split into 2-s trials, each divided into 20-ms bins*" and the reference code sets `binsize=0.02`. Resolution: use 20 ms, because the executable reference code and the dynamic-behaviour analyses use it and a single common grid is needed for all four outputs simultaneously. Step 7/9 verify `T = 100` for every trial.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable at all — it is the conversion's own time axis, fully determined by `time_window = (-0.5, 1.5)` and `binsize = 0.02` relative to `stimOn_times`. One 100-element vector, `build_time_axis()`, is computed once in `main()` and reused for every trial of every session (hence the verifier reporting an identical `[-0.5, 1.5]` range for all 438 sessions). The only raw dependency is `stimOn_times`, which defines where zero sits.

ii.
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    # Match the reference behavior interpolation grid: bin end times.
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

```python
time_axis = build_time_axis()
...
input_trials = [np.vstack([time_axis, np.full(time_axis.shape[0], block_num_all[i],
                                              dtype=np.float32)]) ... for i in keep_idx]
```

iii. CONVERSION_NOTES Step 5 mapping: "*Derived common bin time axis relative to `stimOn_times` → `input[0]`; One continuous time channel repeated for every trial, length 100; use the same 20 ms trial grid as neural/activity outputs … User-required decoder input; not present in reference code, so derive directly from the adopted alignment grid.*"

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. It is `np.linspace(-0.5 + 0.02, 1.5, 100)`, i.e. the **right edge (end time) of each 20 ms bin**, running −0.48 … 1.50 s, cast to `float32` and broadcast unchanged to every trial. The AI chose bin-end rather than bin-centre times specifically to match the reference code's behaviour interpolation grid (`np.linspace(interval_begs + binsize, interval_ends, n_bins)`), and records the convention in metadata. The variable is kept continuous (a monotone ramp), not converted to a binary onset indicator.

ii.
```python
return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

```python
"time_axis_definition": "bin end times relative to stimulus onset",
```

iii. The inline comment "*Match the reference behavior interpolation grid: bin end times*" is the justification: the same expression defines the interpolation query points for wheel and whisker (`x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)`), so input time, wheel sample time and whisker sample time are the identical instants by construction. CONVERSION_NOTES Step 5 lists the "Metadata sanity check: confirm `time_bin_size = 20 ms`, `off_start = -0.5 s`, `off_end = 1.5 s`, and `temporal_alignment_event = stimulus onset` for every converted session", which Step 10 reports as passing.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction: element *k* of the time axis is `-0.5 + 0.02(k+1)`, which is the closing edge of neural bin *k* (`[-0.5 + 0.02k, -0.5 + 0.02(k+1))`) measured from the same `stimOn_times`. So index *k* of the input, of the neural matrix, and of both behavioural outputs all refer to the same 20 ms slice of the same trial. The only residual offset is a labelling convention — the neural bin's *centre* is 10 ms earlier than the timestamp the input reports — inherited from the reference code, which samples behaviour at bin end times while binning spikes on `[beg, end)` intervals.

ii.
```python
# neural: bin k spans [t_beg + 0.02k, t_beg + 0.02(k+1))
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
```

```python
# input / behaviour: sample point k is t_beg + 0.02(k+1)
return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
```

iii. CONVERSION_NOTES Step 10 Check 2, sanity check `CHECK_INPUT = True`: "*independently reconstructed the first kept trial of session `6713a4a7-faed-4df2-acab-ee4e63326f8d` from the raw trials table and verified the converted input matrix exactly matches the raw stimulus-aligned time axis and manually derived trial-in-block value*". The Step 7 plot review adds "*behavior traces without obvious temporal shifts relative to stimulus onset … no off-by-one edge effects or missing final bins were apparent on visual spot-check*".

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft` alone. The trials table carries no block identifier, so blocks are recovered from the fact that `probabilityLeft` is constant within a block: any change of value starts a new block. No other column is used.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
```

iii. CONVERSION_NOTES Step 5 mapping: "*`trials.probabilityLeft` block structure → `input[1]`; Compute trial number within current block on the original trial order … Reset counter whenever `probabilityLeft` changes; preserve original experimental indexing rather than recomputing after exclusions.*" Step 3 records the expected block structure from the papers (90-trial unbiased opening block, then biased blocks of 20–100 trials, empirical mean 51), which is the check applied to the result.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A sequential scan over the **unfiltered** trial order: a counter starts at 1 on the first trial, increments while `probabilityLeft` is unchanged (compared with `np.isclose`), and resets to 1 whenever it changes. The resulting per-trial integer is then subset with `keep_idx`, so trials removed by quality control still advance the counter and the exported number is the animal's true position in the block. It is **1-based**, stored as `float32`, and broadcast constant across the 100 time bins of its trial.

Observed ranges confirm the design: per-session maxima cluster at 90 (the unbiased opening block) with a global range of 1–99; one session starts at 2 because its first trial was filtered out — exactly the intended behaviour.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    out = np.zeros(probability_left.shape[0], dtype=np.float32)
    current = 0
    prev = None
    for idx, value in enumerate(probability_left):
        if idx == 0 or not np.isclose(value, prev):
            current = 1
        else:
            current += 1
        out[idx] = current
        prev = value
    return out
```

```python
np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32)   # i indexes the *unfiltered* table
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "*Trial number in block: Derive this from the original experimental trial order before dropping invalid trials, then subset. This preserves the true block progression.*" Step 9's consistency table checks the result against the papers: range `[1, 99]` versus "*unbiased 90-trial start; biased blocks thereafter*", marked as matching.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is IBL's ±1/0 encoding. No other column contributes — in particular the wheel is not used to re-derive choice.

ii.
```python
choice_all = trials_df["choice"].to_numpy()
...
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. CONVERSION_NOTES Step 5 mapping: "*`trials.choice` → `output[0]` … raw `+1 -> left -> 0`, raw `-1 -> right -> 1`; no-go (`0`) excluded by trial mask*", with the source note "*Official IBL docs indicate raw choice encodes wheel direction; for the requested dataset convert to left/right categorical choice.*"

## 5-b. What processing is involved in computing `output` *Choice*?

i. A recode only. After filtering (which has already removed `choice == 0` and null choices), the values are asserted to be exactly {−1, +1} — an unexpected value raises rather than being silently coerced — and mapped `+1 → 0` (left) and `−1 → 1` (right), i.e. `(choice == -1)`. The per-trial label is then broadcast as a constant across the 100 time bins, stored as `int64`, and `output_values[0] = ["left", "right"]`. Resulting full-dataset balance: 50.86 % left / 49.14 % right.

ii.
```python
def choice_to_label(choice_values: np.ndarray) -> np.ndarray:
    choice_values = np.asarray(choice_values)
    if not np.all(np.isin(choice_values, [-1, 1])):
        bad = np.unique(choice_values[~np.isin(choice_values, [-1, 1])])
        raise ValueError(f"Unexpected choice values after filtering: {bad.tolist()}")
    # Official IBL docs: choice == -1 means chose right, choice == +1 means chose left.
    return (choice_values == -1).astype(np.int64)
```

```python
np.full(T, session.choice_labels[trial_idx], dtype=np.int64),
```

iii. The inline comment cites the IBL documentation for the sign convention. CONVERSION_NOTES Step 5 Key Decision 4: "*Choice remapping: Convert raw IBL `choice` values to left/right semantic labels required by the task: raw `+1` becomes left (`0`), raw `-1` becomes right (`1`).*" Step 10 lists `CHECK_OUTPUT_CHOICE = True` as a passing raw-data spot check, and Step 9 records the near-balanced 0.5086/0.4914 distribution as consistent with the task being approximately balanced overall.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table — the block prior the task holds constant within a block. Same column that drives `trial_number_in_block`.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
...
prior_labels = prior_to_label(prob_left_all[keep_idx])
```

iii. CONVERSION_NOTES Step 5 mapping: "*`trials.probabilityLeft` → `output[1]`; Map `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` and repeat across time bins … This is the requested discrete prior output.*"

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A recode only. Values are rounded to one decimal to absorb float representation error, checked against the allowed set {0.2, 0.5, 0.8} (anything else raises), and mapped to 0/1/2 exactly as the task specifies. The label is broadcast constant across the 100 bins as `int64`; `output_values[1] = ["0.2", "0.5", "0.8"]`. Full-dataset distribution: 0.4188 / 0.1406 / 0.4406 — the small middle class is the 90-trial unbiased opening block, as expected.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}

def prior_to_label(probability_left: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(probability_left, dtype=float), 1)
    unknown = sorted(set(rounded.tolist()) - set(PRIOR_MAP))
    if unknown:
        raise ValueError(f"Unexpected probabilityLeft values: {unknown}")
    return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "*Prior representation: Use `probabilityLeft` directly as the prior variable and encode its three observed task values as categorical labels `0/1/2` for `0.2/0.5/0.8`.*" Step 5 also planned the "*Distribution sanity check: verify prior class distribution reflects the expected `0.5` first block followed by `0.2/0.8` biased blocks*"; Step 9 records the result as matching and Step 10 lists `CHECK_OUTPUT_PRIOR = True` as a passing raw-data spot check.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw rotary encoder stream: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. Speed is derived from them with ibllib's own wheel routines rather than from any pre-computed velocity dataset; the lengths of the two arrays are checked for agreement before use.

ii.
```python
if target == "wheel-speed":
    wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
    wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
    if wheel_position.shape[0] != wheel_timestamps.shape[0]:
        raise ValueError("Length mismatch between wheel.position and wheel.timestamps")
```

iii. CONVERSION_NOTES Step 1 records the reference behaviour being reproduced: "*Continuous behaviors used in the reference code are: `wheel-speed` as `abs(wheel velocity)`*", loaded through `load_target_behavior`/`SessionLoader`. Step 6 notes the switch to direct local reads was for speed only, not a change of variable.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps, the first two being ibllib's standard wheel pipeline (the same functions `SessionLoader.load_wheel` calls internally, with the same default parameters):
1. `interpolate_position(timestamps, position, freq=1000)` — the encoder only emits samples when the wheel moves, so position is interpolated onto an even 1000 Hz grid.
2. `velocity_filtered(position, fs=1000, corner_frequency=20, order=8)` — differentiate to velocity through a 20 Hz low-pass Butterworth filter.
3. `np.abs(velocity)` — speed in rad/s. (Cast to `float32` at this point.)
4. Per trial, linear interpolation (`scipy.interpolate.interp1d`, `fill_value="extrapolate"`) of that trace onto the 100 bin-end times of the trial window. The slice fed to the interpolator is `searchsorted(..., 'right')` to `searchsorted(..., 'left')`, i.e. strictly the samples inside the window.

Discretisation is a separate, dataset-level step (7-c). The continuous traces for all kept trials of all sessions are retained in memory until then.

ii.
```python
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {"times": np.asarray(times, dtype=np.float32),
        "values": np.abs(np.asarray(velocity, dtype=np.float32))}
```

```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. CONVERSION_NOTES Step 1: "*continuous signals are segmented into the same per-trial `stimOn` window; behavior traces are linearly interpolated onto `n_bins = ceil(interval_len / binsize)` samples*". Step 5 mapping: "*Load wheel speed as `abs(velocity)`, align to stimulus onset on the same 2 s / 20 ms grid as neural data, then discretize … This is a task-driven deviation from the paper's first-movement alignment, required because the user explicitly requests stimulus-onset alignment for the full dataset.*" Step 10 lists `CHECK_OUTPUT_WHEEL = True` as a passing raw-data spot check.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes by **global** tertiles: after every session has been processed, all aligned wheel-speed samples from all kept trials of all 438 sessions are concatenated into one vector and the ⅓ and ⅔ quantiles are taken **once**; those two numbers are then applied to every session. `np.digitize(values, [q1, q2], right=False)` gives 0 = low, 1 = medium, 2 = high. The thresholds are recorded in metadata (`wheel_speed_thresholds = (0.01514, 0.40300)` rad/s for the full run). This makes the *dataset-wide* class balance exactly 1/3 : 1/3 : 1/3 (verified: 0.3333 / 0.3333 / 0.3333) but does **not** equalise classes within a session — a session with an unusually still or unusually active animal will be skewed toward one class.

ii.
```python
def compute_thresholds(processed_sessions):
    wheel_values = np.concatenate([np.concatenate(s.wheel_continuous) for s in processed_sessions])
    ...
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    thresholds[name] = (float(q1), float(q2))
    return thresholds

def discretize(values, thresholds):
    q1, q2 = thresholds
    return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)
```

```python
"wheel_speed_thresholds": thresholds["wheel_speed"],
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "*Continuous-output discretization: Compute global tertile thresholds from all retained aligned wheel-speed samples and all retained aligned whisker-motion-energy samples, respectively, then apply those thresholds consistently across all sessions.*" Step 4 frames the discretisation itself as task-mandated: "*The user task explicitly requires categorical outputs, so discretize these continuous traces into 3 bins only after reproducing the reference loading/alignment. This is an allowed task-driven deviation, not a loading mismatch.*" Step 5's planned check — "*wheel and whisker discretization should not collapse nearly all samples into one class*" — is reported as passing in Step 9 at the dataset level only; per-session balance was not examined.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is evaluated at exactly the same 100 time points as the input time axis, `stimOn + linspace(-0.48, 1.50, 100)`, i.e. one sample per neural bin, at that bin's closing edge. Because the wheel timestamps and the spike times share the IBL session clock, subtracting nothing and simply querying at `stimOn + t` is the whole alignment. A trial is dropped outright (see 1-e) unless the wheel stream brackets the window to within one bin at both ends, so no trial is padded or extrapolated across a gap; `fill_value="extrapolate"` only ever covers sub-bin edge effects. The code asserts every aligned trace has exactly the shape of the time axis.

ii.
```python
idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
idxs_end = np.searchsorted(target_times, interval_ends, side="left")
...
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
```

```python
if not all(x.shape == time_axis.shape for x in wheel_aligned):
    raise RuntimeError("Wheel trace shape mismatch after alignment")
```

iii. CONVERSION_NOTES Step 4 Key Decision 2: "*Common alignment grid: Use one common `stimOn_times`-aligned `[-0.5, 1.5]` window with 20 ms bins for neural, inputs, and outputs because that matches the executable reference code and the user explicitly requires stimulus-onset alignment.*" Step 7's plot review states the discretised traces "*transition where the continuous traces cross the global tertile thresholds; no off-by-one edge effects or missing final bins were apparent*".

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. IBL's released per-frame motion energy for the whisker-pad ROI of a side camera, `leftCamera.ROIMotionEnergy.npy`, with its frame times `_ibl_leftCamera.times.npy`. If the left camera's files are missing or unreadable the loader falls back to `rightCamera.ROIMotionEnergy.npy` / `_ibl_rightCamera.times.npy`. If neither is available every trial of the session fails the coverage test and the session is dropped.

ii.
```python
def align_continuous_behavior(session_path, behavior_name, trials_df):
    if behavior_name == "whisker-motion-energy":
        target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
        if target.get("skip"):
            target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

```python
times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
```

iii. CONVERSION_NOTES Step 1: "*`whisker-motion-energy` from left camera whisker energy, falling back to right camera if left is unavailable*", and Step 5 Key Decision 9: "*Whisker camera fallback: Use left whisker motion energy when available and fall back to right whisker motion energy otherwise, matching the provided code.*" Step 3 records the definition from the paper: "*Whisker motion energy is defined as the mean absolute difference between adjacent video frames in a whisker-pad bounding box anchored between nose tip and eye*", i.e. it is used as released with no re-derivation.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, smoothing, normalisation or baseline subtraction. The only processing is (a) a length-consistency fix between frame times and motion-energy values, and (b) the same per-trial linear interpolation onto the 100 bin-end times used for the wheel. For (a): if there are *fewer* timestamps than values the camera is rejected (and the other camera tried); if there are *more*, the **leading** extra timestamps are dropped — this reproduces ibllib's `SessionLoader._check_video_timestamps` exactly, whose rationale is that in pre-GPIO sessions the first few frames are sometimes not recorded.

ii.
```python
if times.shape[0] < values.shape[0]:
    raise ValueError("Camera times are shorter than video data for leftCamera.")
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0] :]
```

```python
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
interpolated[idx] = np.asarray(y_interp, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 1 documents the reference alignment rules being reproduced ("*behavior traces are linearly interpolated onto `n_bins = ceil(interval_len / binsize)` samples; valid intervals require data to start and end within one bin of the requested interval boundaries*"). Step 5 mapping: "*Load left whisker motion energy when available, otherwise right; align to stimulus onset on the same 2 s / 20 ms grid, then discretize into 3 global bins … Same alignment/deviation rationale as wheel speed.*"

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: one **global** pair of tertile thresholds computed from the concatenation of every aligned whisker sample in every kept trial of every session (`(2.728, 7.860)` for the full run), applied uniformly via `np.digitize` to give 0 = low, 1 = medium, 2 = high, with `output_values[3] = ["low", "medium", "high"]`. Dataset-wide balance is exactly 1/3 each; per-session balance is not equalised, so sessions differ systematically in their class mix according to how much the animal whisked and how the camera/ROI was scaled.

ii.
```python
whisker_values = np.concatenate([np.concatenate(s.whisker_continuous) for s in processed_sessions])
for name, values in [("wheel_speed", wheel_values), ("whisker_motion_energy", whisker_values)]:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    thresholds[name] = (float(q1), float(q2))
```

```python
discretize(session.whisker_continuous[trial_idx], thresholds["whisker_motion_energy"]),
```

iii. Same as 7-c: CONVERSION_NOTES Step 5 Key Decision 7 (global tertiles "*applied consistently across all sessions*") and the Step 4 note that discretisation is a task-mandated deviation from the papers' continuous regression targets. Step 9 records the resulting 0.3333/0.3333/0.3333 split as matching the "3 bins by instruction" expectation.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Exactly as the wheel: the camera frame times are on the same session clock as the spikes, so the trace is simply queried at `stimOn + linspace(-0.48, 1.50, 100)` — one value per neural bin, at the bin's closing edge. The same coverage requirement applies, so a trial survives only if camera frames bracket the whole 2 s window to within one bin at each end; trials in camera dropouts are removed rather than extrapolated. Shape is asserted after alignment.

ii.
```python
whisker_traces, whisker_mask = align_continuous_behavior(session_path, "whisker-motion-energy", trials_df)
...
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
```

```python
if not all(x.shape == time_axis.shape for x in whisker_aligned):
    raise RuntimeError("Whisker trace shape mismatch after alignment")
```

iii. Same as 7-d — CONVERSION_NOTES Step 4 Key Decision 2 (one common `stimOn`-aligned grid for neural, inputs and outputs) — with Step 10's `CHECK_OUTPUT_WHISK = True` raw-data spot check reported as passing and the Step 7 plot review reporting no visible temporal shift.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or inconsistent data is handled at five levels, and the dominant strategy is **drop rather than impute**:

- **Camera frame-count mismatch**: fewer timestamps than values → reject that camera and fall back to the other; more timestamps than values → drop the *leading* extras, reproducing ibllib's documented pre-GPIO fix.
- **Missing files / unreadable streams**: `load_target_behavior_current` wraps its loads in `try/except BaseException` and returns `{"skip": True, "error": ...}`; a missing behaviour then yields an all-False validity mask (`["missing"] * n_intervals`) rather than crashing.
- **Missing probes**: a probe whose cluster table is empty after QC is skipped; a session with no surviving probe raises and is dropped.
- **Trials with missing/edge data**: NaN event times are excluded by the trials mask; NaN interval bounds and traces that do not span the window are excluded by the coverage test.
- **Sessions**: any session raising for any reason — including `Only N valid trials after filtering` when fewer than 2 trials survive — is recorded in `metadata['skipped_sessions']` with its error and whether its files were present, and excluded. 21 sessions were dropped this way.
- **Transient failures**: `process_one_session_worker` retries up to 3 times with jittered backoff for errors matching a transient-error signature, and `main()` re-runs every worker-failed session sequentially before giving up — this is what confirmed the 21 exclusions were real and not multiprocessing artefacts.
- **Out-of-range channel index**: `np.clip(cluster_channels, 0, len(channel_region_ids) - 1)` guards the region lookup.

Two residual behaviours are worth flagging: NaNs inside a behavioural trace are explicitly allowed (`allow_nans=True`, the reference default) and would be silently digitised into the top class; and 16 trials across 3 sessions have all-zero neural matrices, which the AI verified against raw spikes as genuine zero-spike windows rather than a bug, and therefore left in place.

ii.
```python
    except BaseException as exc:  # noqa: BLE001
        return {"times": None, "values": None, "skip": True, "error": str(exc)}
```

```python
if target_times is None or target_vals is None:
    return [None] * n_intervals, np.zeros(n_intervals, dtype=bool), ["missing"] * n_intervals
```

```python
for attempt in range(1, 4):
    try:
        if attempt > 1:
            time.sleep(0.5 * attempt + random.random() * 0.5)
        return process_one_session(row, time_axis)
    except Exception as exc:
        last_exc = exc
        if attempt >= 3 or not is_likely_transient_error(exc):
            raise
```

iii. CONVERSION_NOTES Step 10 Check 5: "*21 sessions were retried sequentially after worker failures and all 21 consistently failed with `Only 0 valid trials after filtering`, confirming these are true exclusions rather than multiprocessing artifacts*" and "*The 16 verifier warnings about all-zero neural trials correspond to genuine zero-spike windows and therefore should not be 'fixed' by altering the data values; removing them would be an analysis choice, not a data-correction step*". Check 2 documents the per-trial verification: "*for all three, the independently loaded raw QC-passed spikes contained exactly 0 spikes in the converted `[-0.5, 1.5] s` stimulus-aligned window*".

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk and binning it. The script instruments each session with `trials_s`, `spikes_s`, `behavior_s`, `total_s`; the `spikes_s` stage — `np.load` of `spikes.times.npy` + `spikes.clusters.npy` (hundreds of MB per probe, two probes on many sessions) followed by the per-trial binning loop — dominates, running 15–30 s per session on the full run against a few seconds for trials and behaviour. It is I/O-bound rather than CPU-bound: at 48 workers a session that took 3.0 s alone took 91.4 s inside the pool, which is disk contention, and a worker-count sweep found the optimum at 12 (4 → 48.9 s, 8 → 34.2 s, 12 → 31.7 s, 16 → 33.8 s for 8 sessions). The full conversion took 2071 s (34.5 min) for 438 sessions — above the 15-minute target the instructions set. Secondary costs: pickling and writing the 12.49 GB output, and `compute_thresholds`, which concatenates every aligned behavioural sample in the dataset into two large vectors before quantiling.

ii.
```python
timing={"trials_s": t_trials - t0, "spikes_s": t_spikes - t_trials,
        "behavior_s": t_behavior - t_spikes, "total_s": t_behavior - t0}
```

```python
spikes = {"times": np.load(sort_dir / "spikes.times.npy"),
          "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32)}
```

iii. CONVERSION_NOTES Step 9: "*First full-conversion attempt with 48 session workers caused severe local disk contention … Set the default `--session-workers` to 12 based on the benchmark. The successful full conversion finished in 2071.0 s (34.5 min)*". Step 6 adds the earlier, larger win: moving off the ONE/Alyx path "*Reduced sample conversion from 132.5 s to 12.1 s while preserving counts and validation results*".

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four, the first being much the most expensive:

1. **`get_spike_data_per_interval`** rescans the *entire* session spike array for every trial: `idxs_t = (times >= t_beg) & (times < t_end)` is O(n_spikes) per trial, making the loop O(n_trials × n_spikes) — tens of millions of comparisons per trial on a two-probe session. Because spike times are sorted, two `np.searchsorted` calls would give the same slice in O(log n), and the whole session could be binned in a single `np.bincount` over `unit * n_bins + bin_index` with a per-trial offset. This loop is inherited verbatim from the reference code.
2. **`get_behavior_per_interval_current`** builds a fresh `scipy.interp1d` object per trial and per stream (2 × n_trials objects per session) where a single `np.interp` over a flattened query vector, or one `interp1d` over the whole session trace, would do.
3. **`trial_number_in_block`** is an explicit Python `for` loop over trials; it is a two-line vectorisation (`change = ~np.isclose(p[1:], p[:-1])`, `block = np.concatenate([[0], np.cumsum(change)])`, then `arange - group_start[block]`).
4. **`assemble_dataset`** calls `discretize` once per trial per stream, and builds each `output` matrix with a per-trial `np.vstack`; the session's traces are already a homogeneous array, so one `np.digitize` per session per stream would suffice.

The AI did not identify any of these in its notes; it addressed throughput with process-level parallelism instead, which is why the full run landed at 34.5 min rather than under the 15-minute target.

ii.
```python
for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    idxs_t = (times >= t_beg) & (times < t_end)      # full-array scan, once per trial
```

```python
for idx, (t_seg, v_seg) in enumerate(zip(target_times_list, target_vals_list)):
    ...
    fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
```

```python
for idx, value in enumerate(probability_left):
    if idx == 0 or not np.isclose(value, prev):
```

iii. No direct justification is recorded. The nearest statements are CONVERSION_NOTES Step 5 ("*When possible, import or copy code from the reference code*", which is why the binning loop was kept verbatim) and Step 6, whose "Code speedups added" list is entirely about I/O and parallelism — "*Avoided loading unused behavioral streams during session conversion*", "*Reused probe IDs directly from `bwm_release.csv`*", "*Added session-level parallelism for `--full`*" — with the inefficiencies section discussing only ONE/Alyx latency and the vendored `ibllib` import failures, never the per-trial loops.

## 10-c. What processing does the code repeat multiple times?

i. Several things are recomputed that could be computed once:

- Inside the per-trial binning loop, `np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)` runs for every trial — a sort of the cluster id array per trial — and `bincount2D` re-derives its own bin edges per trial. `cluster_ids = np.unique(clusters)` is also recomputed on each call.
- `find_latest_file` re-globs the `alf` tree separately for each dataset (trials, wheel timestamps, wheel position, camera times, camera motion energy, and once per probe for the spike files), so the same directory tree is walked six-plus times per session.
- When the left camera is unavailable, `load_target_behavior_current` is called twice, re-running the whole load/except path for the right camera.
- `session_has_local_modalities` re-globs the filesystem a fourth time, but only to annotate the error record of an already-failed session.
- Beryl remapping (`to_beryl_acronyms`) is applied to the full per-neuron acronym list at assembly, after each session already stored the per-neuron acronyms — a single pass, but on a list that was carried through the whole pipeline uncollapsed.
- The 12.49 GB dataset is held in memory in full while also being pickled, and the continuous wheel/whisker traces for every trial are retained for the whole run purely so the global thresholds can be computed at the end.

ii.
```python
_, target_indices, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
```

```python
trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
```

```python
target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
if target.get("skip"):
    target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

iii. Not documented as repeated work. The only related note is CONVERSION_NOTES Step 6's "*Avoided loading unused behavioral streams during session conversion*" (pupil diameter and body motion energy are never loaded) and "*Reused probe IDs directly from `bwm_release.csv` instead of per-session Alyx `eid2pid` lookups*", which removed one genuinely repeated lookup.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest item is that **spike binning and behaviour interpolation are performed for every trial in the table, and the trial mask is applied only afterwards**. Roughly 36 % of trials are discarded (186,261 kept of ~290,000 processed), so about a third of the binning and interpolation work — the dominant CPU cost — is thrown away. `keep_idx` is known before `bin_spiking_data_current` is called, so the interval list could simply be pre-filtered.

Smaller items:
- `meta["cluster_qc"] = {k: np.asarray(v) for k, v in clusters.to_dict("list").items()}` materialises *every* column of the cluster metrics table (dozens of QC metrics, uuids, depths) for every session; nothing downstream reads it.
- `ProcessedSession.cluster_good` (the `label >= 1` flag) is computed and carried through, but since `qc=1` already filtered, it is identically 1 and is never used in `assemble_dataset`.
- `load_release_sessions` returns `bwm_df` alongside `sessions_df`; the caller binds it and never uses it.
- `build_one()` constructs and authenticates a `ONE` client that the conversion path never calls.
- `session_has_local_modalities` performs four filesystem globs purely to decorate the skip record of sessions that already failed.
- The full-precision continuous `wheel_continuous` / `whisker_continuous` traces are kept for the entire dataset, in addition to the discretised outputs, only to compute two threshold pairs — a consequence of the global-threshold decision rather than an oversight, but it roughly doubles peak memory for the behaviour arrays.
- Outputs are stored as `int64` (8 bytes) for what are 2- and 3-valued categoricals, and `input` as `float32` where the time axis is identical for all 186,261 trials — a large share of the 12.49 GB file.

ii.
```python
binned_spikes = bin_spiking_data_current(reg_clu_ids, neural_dict, trials_df)   # ALL trials
...
keep_idx = np.flatnonzero(keep_mask)                                           # then subset
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
```

```python
meta = {"cluster_regions": list(clusters["acronym"]),
        "good_clusters": (clusters["label"] >= 1).to_numpy(dtype=np.int8),
        "cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()}}
```

```python
bwm_df, sessions_df = load_release_sessions()      # bwm_df unused thereafter
```

iii. Not documented. CONVERSION_NOTES Step 6 lists only I/O-related inefficiencies and does not mention that binning precedes filtering; the trial mask is instead described in Step 5 as being applied to select trials, and the two-stage design is justified only by the global-threshold decision ("*Implemented a two-stage conversion: session-wise loading/alignment to collect continuous wheel/whisker traces; global threshold computation for 3-bin discretization; final assembly into the target pickle dictionary*").
