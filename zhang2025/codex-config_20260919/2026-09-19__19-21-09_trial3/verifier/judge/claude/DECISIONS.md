# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API at all. It takes the reference repository's own release freeze table, `/app/code/code_zhang2025/data/bwm_release.csv`, as the authoritative inventory (one row per probe insertion), asserts that it contains exactly 459 sessions / 699 probes / 139 subjects, groups rows by `eid`, and then constructs ALF file paths directly on disk from the `lab / subject / date / session_number` columns (`/app/data/one_cache/<lab>/Subjects/<subject>/<date>/<nnn>/alf`). Within a session directory it globs recursively for each object and resolves ONE dataset revisions itself, preferring revision folders (`#YYYY-MM-DD#`) by sorting on the revision string and preferring `#2025-03-03#` for the trials table and `#2024-05-06#` pykilosort for the spike sorting. Per session it reads: `_ibl_trials.table.pqt` (pandas), `_ibl_wheel.timestamps/position.npy`, `left/rightCamera.ROIMotionEnergy.npy` + `_ibl_<side>Camera.times.npy`, and per probe `clusters.metrics.pqt`, `clusters.channels.npy`, `channels.brainLocationIds_ccf_2017.npy`, `spikes.times.npy`, `spikes.clusters.npy`. Large spike arrays are opened with `mmap_mode="r"`. Sessions are processed concurrently in a `ThreadPoolExecutor` (24 workers in `--full`, 2 in `--sample`) and re-ordered by freeze index afterwards; `--sample` takes the first two freeze sessions.

ii.
```python
FREEZE_CSV = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def session_path(row: pd.Series) -> Path:
    return (
        DATA_ROOT / str(row["lab"]) / "Subjects" / str(row["subject"])
        / str(row["date"]) / f"{int(row['session_number']):03d}"
    )

def revision_key(path: Path) -> tuple[str, str]:
    """Sort unrevisioned paths before ISO-date revision folders."""
    revisions = [p[1:-1] for p in path.parts if p.startswith("#") and p.endswith("#")]
    return (max(revisions, default=""), str(path))

def newest_file(base: Path, pattern: str, preferred_revision: str | None = None) -> Path | None:
    paths = sorted(base.glob(f"**/{pattern}"), key=revision_key)
    ...
```
```python
    if len(sessions) != 459 or len(freeze) != 699 or freeze["subject"].nunique() != 139:
        raise RuntimeError("Freeze identity does not match the paper release (459/699/139)")
```
```python
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_session, i, info, args.show_processing and i < 2): (i, info)
            for i, info in enumerate(selected)
        }
```

iii. From CONVERSION_NOTES.md Step 4/Step 5 (decision 7): "Use the CSV freeze list and merge all probes from each EID; do not use the 21 extra broad-registry sessions"; "Resolve paths only through `/app/data/one_cache`; no network/download needed"; "Prefer staged `#2024-05-06#` pykilosort products, `#2025-03-03#` trial tables, newest 2025 motion energy and newest matching camera timestamps... This matches the staged current release; direct ALF paths avoid unavailable network access and ONE's incomplete offline cross-release revision resolution." The freeze CSV is exactly what the reference caching script (`0_data_caching.py`) uses to enumerate brain-wide-map sessions.

## 1-b. How are the data split into subjects?

i. The `subject` column of the freeze CSV is carried through on each session record. At assembly the subject list is the sorted set of unique subject names over *retained* sessions, and `subject_idx` is the index of each session's subject into that list. No path or filename parsing is used. 136 of the 139 freeze subjects survive (3 subjects appear only in excluded sessions).

ii.
```python
        sessions.append({
            "eid": str(eid),
            "lab": str(first["lab"]),
            "subject": str(first["subject"]),
            ...
        })
```
```python
    subjects = sorted({r["info"]["subject"] for r in results})
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    ...
    subject_idx = np.asarray(
        [subject_lookup[r["info"]["subject"]] for r in results], dtype=np.int32,
    )
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "Freeze CSV subject/EID metadata -> `subjects`, `subject_idx`, metadata session info | Sorted unique subject names and integer lookup". The freeze table already carries a unique subject id, so nothing has to be derived.

## 1-c. How are the data split into sessions?

i. A session is the `eid` grouping of the freeze CSV. `load_freeze` groups the probe rows by `eid` with `sort=False` so the freeze ordering (and the within-session probe ordering) is preserved, producing one record per session holding its list of `pid`s and `probe_name`s. Session order in the output follows first occurrence of the eid in the freeze CSV. All probes of a session are merged into one population (see 2-b).

ii.
```python
    sessions: list[dict] = []
    # sort=False preserves the paper freeze order and probe order.
    for eid, group in freeze.groupby("eid", sort=False):
        first = group.iloc[0]
        sessions.append({
            "eid": str(eid), ...,
            "pids": group["pid"].astype(str).tolist(),
            "probe_names": group["probe_name"].astype(str).tolist(),
        })
```

iii. CONVERSION_NOTES.md Step 4: "Exactly 699 PIDs, 459 EIDs, 139 subjects; all corresponding payloads staged... Use the CSV freeze list and merge all probes from each EID." Step 5: "Session order follows first EID occurrence in the freeze CSV."

## 1-d. How are the data split into trials?

i. The split is given by the data: `_ibl_trials.table.pqt` has one row per trial. Trial events (`stimOn_times`, `choice`, `probabilityLeft`, `firstMovement_times`, `feedback_times`, `goCue_times`, `feedbackType`) are read as columns and every per-trial quantity is indexed by row position. The raw row order is preserved (used for `trial_number_in_block`), and the retained subset is stored as `source_trial_indices` in metadata.

ii.
```python
def load_trials(info: dict) -> tuple[pd.DataFrame, Path]:
    alf = session_path(pd.Series(info)) / "alf"
    path = newest_file(alf, "_ibl_trials.table.pqt", "#2025-03-03#")
    if path is None:
        raise FileNotFoundError(f"No trial table for {info['eid']}")
    trials = pd.read_parquet(path)
    return trials, path
```

iii. No decision to make — CONVERSION_NOTES.md Step 2 records that the trials table is the native ALF per-trial object; the notes report 296,090 raw trials, mean 645.08, median 601, range 401–1,525, matching the data paper's mean 645 / median 602 / range 401–1,525.

## 1-e. How are trials filtered based on quality controls?

i. Three stacked filters.
   1. An exact reimplementation of the reference `load_trials_and_mask(one, eid, max_trial_len=10.0)` used by `prepare_data`: all six `nan_exclude` fields (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`) must be non-NaN; reaction time (`firstMovement_times - stimOn_times`) must be in [0.08, 2.0] s inclusive; `choice != 0` (no-go dropped); and go-cue-to-feedback duration must not exceed 10 s (NaN durations are kept, matching pandas `eval` semantics in the reference). The initial unbiased 0.5 block is deliberately *kept* (`exclude_unbiased=False` in the reference).
   2. Stream-coverage: each surviving trial's [-0.5, +1.5) s window must be spanned by the wheel trace and by the chosen camera's timestamps, with at least two samples and both endpoints within one bin (20 ms) of the window edges — this is the reference `get_behavior_per_interval` skip rule.
   3. Neural coverage: the window must lie inside the interval common to every probe's first and last spike.
   Sessions with fewer than 2 surviving trials, or with no usable whisker stream, are skipped entirely and recorded in `metadata.skipped_sessions`. Result: 296,090 raw → 195,781 after the code mask → 188,922 retained across 444 sessions.

ii.
```python
def trial_mask(trials: pd.DataFrame) -> np.ndarray:
    """Reproduce load_trials_and_mask(..., max_trial_len=10)."""
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    good = np.ones(len(trials), dtype=bool)
    for column in required:
        good &= trials[column].notna().to_numpy()
    reaction_time = (trials["firstMovement_times"].to_numpy(dtype=float)
                     - trials["stimOn_times"].to_numpy(dtype=float))
    good &= reaction_time >= 0.08
    good &= reaction_time <= 2.0
    good &= trials["choice"].to_numpy() != 0
    # Reference query excludes only durations >10; NaN go cues are not explicitly excluded.
    if "goCue_times" in trials:
        duration = (trials["feedback_times"].to_numpy(dtype=float)
                    - trials["goCue_times"].to_numpy(dtype=float))
        good &= ~(duration > 10.0)
    return good
```
```python
        # Retain only windows covered by every recorded probe.  Without this check,
        # trailing behavioral trials can be represented as population-wide zero
        # activity after an electrophysiology recording has already stopped.
        neural_coverage_start = max(probe_starts)
        neural_coverage_end = min(probe_ends)
        neural_good = ((stim_code + OFF_START >= neural_coverage_start)
                       & (stim_code + OFF_END <= neural_coverage_end))
        ...
        stream_good = wheel_good & motion_good & neural_good
        keep_in_code = np.flatnonzero(stream_good)
        source_indices = code_indices[keep_in_code]
        if len(source_indices) < 2:
            raise ValueError(f"Only {len(source_indices)} trials after stream coverage")
```
```python
        if hi - lo < 2:
            continue
        if abs(beginnings[trial] - tx[0]) > BIN_SIZE:
            continue
        if abs(endings[trial] - tx[-1]) > BIN_SIZE:
            continue
```

iii. CONVERSION_NOTES.md Step 5 decision 3: "Apply the exact supplied mask (finite required events, choice !=0, RT 0.08–2 s inclusive, goCue-to-feedback <=10 s). Then exclude individual valid trials lacking endpoint coverage in wheel, whisker, or the common interval recorded by every neural probe. This implements the apparent intent of the reference behavior masks instead of reproducing the list-`and` bug that can discard a whole session. Drop sessions with <2 remaining trials or no usable whisker stream." Step 9/10 document why the neural-coverage clause was added: the first full verification reported three all-zero neural trials in session `8c2f7f4d…`; raw inspection showed the probe's last spike was at 1779.36 s while those trials started at 1789.50 s or later.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Per probe: `spikes.times.npy` and `spikes.clusters.npy` (from the `#2024-05-06#` pykilosort folder when present) build the activity itself. `clusters.metrics.pqt` supplies the cluster table (row count, `cluster_id`, and the `label` QC score, which is only recorded, not used to filter). `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` supply the anatomical location, mapped Allen → Beryl for `brain_region_idx`.

ii.
```python
def load_probe_metadata(alf: Path, probe_name: str) -> dict:
    directory = probe_directory(alf, probe_name)
    metrics_path = directory / "clusters.metrics.pqt"
    channels_path = directory / "clusters.channels.npy"
    atlas_path = directory / "channels.brainLocationIds_ccf_2017.npy"
    spikes_times_path = directory / "spikes.times.npy"
    spikes_clusters_path = directory / "spikes.clusters.npy"
    ...
    regions = BrainRegions()
    allen = regions.id2acronym(channel_atlas_ids[cluster_channels.astype(int)])
    beryl = regions.acronym2acronym(allen, mapping="Beryl").astype(str)
```
```python
    spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
    spike_clusters = np.load(probe["spikes_clusters_path"], mmap_mode="r")
```

iii. CONVERSION_NOTES.md Step 5 mapping row: "Revised `spikes.times`, `spikes.clusters` from every freeze-list probe -> `neural`", with reference functions `prepare_data`, `merge_probes`, `bin_spiking_data`; and "Cluster channel index + `channels.brainLocationIds_ccf_2017` -> `brain_region_idx` | CCF ID -> Allen acronym -> Beryl acronym", matching `SpikeSortingLoader.merge_clusters` / `list_brain_regions`.

## 2-b. How is the `neural` data processed?

i. Raw spike counts in 100 half-open 20 ms bins covering [stimOn − 0.5, stimOn + 1.5). Bin index is `floor((t − start)/0.02)` with an explicit range mask. No smoothing, no rate conversion (values remain counts per bin, **not** Hz), no normalisation. Probes of a session are concatenated along the neuron axis with an offset, in freeze probe order, so a session yields a single pooled population; spike-cluster ids are mapped to cluster-table rows by `searchsorted` when they are not simply `0..n-1`. Binning is done trial-batched: trial/cluster/bin are folded into one flat code and a single `np.bincount` fills a chunk of trials at a time, with the chunk size capped so temporaries stay under ~200 MB. Output is `float32` of shape `(n_trials, n_neurons, 100)`, then sliced into a per-trial list.

ii.
```python
        for local_trial, trial in enumerate(range(first, last)):
            lo = int(np.searchsorted(spike_times, starts[trial], side="left"))
            hi = int(np.searchsorted(spike_times, ends[trial], side="left"))
            ...
            clusters = map_spike_clusters(raw_clusters, probe["cluster_ids"])
            bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
            keep = (bins >= 0) & (bins < N_BINS)
            ...
            code = ((local_trial * n_clusters + clusters) * N_BINS + bins).astype(np.int64)
            encoded.append(code)
        if encoded:
            flat = np.concatenate(encoded)
            counts = np.bincount(flat, minlength=(last - first) * n_clusters * N_BINS,
                                 ).reshape(last - first, n_clusters, N_BINS)
            output[first:last] = counts
```
```python
        n_neurons = sum(p["n_clusters"] for p in probes)
        neural = np.zeros((len(source_indices), n_neurons, N_BINS), dtype=np.float32)
        offset = 0
        for probe in probes:
            n = probe["n_clusters"]
            neural[:, offset:offset + n] = bin_probe(probe, starts, ends)
            all_regions.append(probe["regions"])
            offset += n
```

iii. CONVERSION_NOTES.md Step 1: "Spike values are raw counts per 20 ms bin, not rates. No delta-F/F applies because this is electrophysiology, not imaging." Step 4: "Use half-open 20 ms bins with bin index `floor((time-start)/0.02)`, neuron x time float32 counts", matching reference `bincount2D` in `get_spike_data_per_interval`. Probe merging follows `merge_probes`, whose stated rationale is that probes in one session share the same behaviour and are not independent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality filtering is applied.** Every Kilosort cluster in every freeze probe is kept — 599,865 units across the 444 retained sessions (mean 1,351 per session, max 3,140). The `clusters.metrics` `label` score is read and the count of `label >= 1` units is stored as metadata (`n_good_label_clusters`) purely as a cross-check against the data paper's 75,708 well-isolated neurons, but never used to select. Units whose Beryl acronym is `void` (outside the brain), `root`, `x` or `y` are also retained: the verification log shows 12,827 `void`, 85,656 `root`, 344 `x` and 18 `y` neurons in the converted data. The only neural-side exclusion is at session/trial level (probes with zero spikes raise and skip the session; trials outside the probes' common recording interval are dropped).

ii.
```python
        "n_good": int((metrics["label"].to_numpy() >= 1).sum()),
```
(the only use of `label`; no mask is derived from it)
```python
            "neuron_filter": "all Kilosort clusters (qc=None), matching decoder reference code",
```

iii. CONVERSION_NOTES.md Step 3 acknowledges the conflict explicitly: "Data-paper analyses call units well-isolated only when they pass amplitude >50 microvolts, noise cutoff <20 microvolts, and refractory-period-violation criteria... However, the method paper says it bins 'all neurons' sorted by Kilosort 2.5, and the provided caching code loads all clusters (`qc=None`)." Step 4 resolution: "The requested neural-decoder conversion follows Zhang/caching code: retain all 621,733 sorted units. Quality filtering would change the reference decoder input and discard multiunit information; record quality counts as a check." Step 4 also decides to keep `void`/`root`: "Retain `void/root` if produced because code selects all regions." Step 12 reaffirms it when investigating low choice accuracy: "applying data-paper quality filtering merely to raise a score would violate the reference decoder process."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. To stimulus onset. All ALF streams (spike times, trial event times, wheel timestamps, camera frame times) are already expressed in seconds on one synchronised session clock, so alignment is a subtraction: per trial, `start = stimOn_times − 0.5` and `end = stimOn_times + 1.5`, spikes are sliced by `searchsorted` between those bounds and their bin index computed relative to `start`. The window is half-open `[-0.5, +1.5)`, and `metadata.temporal_alignment_event` is set to `"visual stimulus onset (stimOn_times)"` with `off_start = -0.5`, `off_end = 1.5`.

ii.
```python
        stim_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
        starts = stim_times + OFF_START
        ends = stim_times + OFF_END
```
```python
            bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
            keep = (bins >= 0) & (bins < N_BINS)
```
```python
            "temporal_alignment_event": "visual stimulus onset (stimOn_times)",
            "off_start": OFF_START,
            "off_end": OFF_END,
```

iii. CONVERSION_NOTES.md Step 4: "Decoder Task explicitly requires stimulus-onset alignment and joint outputs; use caching script's common 100-bin representation"; Step 3: "The supplied caching code creates a common joint representation for all named behaviors at stimulus onset over `[-0.5, +1.5)` s with 20 ms bins." This matches `params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` in `0_data_caching.py`. Step 10 records boundary checks that spikes exactly at the left edge are included and exactly at the right edge excluded.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial over the 2 s window, identical for every trial and session; `metadata.time_bin_size = 20.0` (ms). No rebinning, resampling or smoothing of the neural data — spikes are binned once, directly from spike times, at the final resolution. (Only the *continuous behavioural* streams are resampled onto this grid; see 7-b/8-b.) The time axis is labelled at bin **right edges** (`-0.48 … +1.50`), not centres, so the time input value at index *i* is the closing edge of neural bin *i*.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```
```python
            "time_bin_size": 20.0,
            "time_bin_size_units": "ms",
            "n_timepoints": N_BINS,
            "neural_representation": "raw spike counts in half-open 20-ms bins",
            "behavior_sample_convention": "right edge of each neural time bin",
```

iii. CONVERSION_NOTES.md Step 3: "Neural data time bin | 20 ms and 100 steps for 2-s choice/caching representation"; Step 1: "Reference temporal processing is explicit: stimulus-onset alignment, window `[-0.5, +1.5)` s, 20 ms bins, hence 100 time bins." Step 5 decision 2 justifies the right-edge labelling: "Behavior and the time input use the reference interpolation grid (`start + binsize` through `end`), while each neural column counts the immediately preceding half-open bin. This makes output samples causal with the spike bin ending at the same time."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table — the alignment event — together with the fixed window/bin constants. Because every trial uses the same window and bin size, the input is a single constant vector of 100 values, `[-0.48, -0.46, …, 1.50]` s, broadcast to every trial of every session. It is named `time_since_stimulus_onset_s`.

ii.
```python
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```
```python
        stim_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
        starts = stim_times + OFF_START
```
```python
        "input_names": ["time_since_stimulus_onset_s", "trial_number_in_block"],
```

iii. CONVERSION_NOTES.md Step 5 mapping: "Fixed event-relative sample grid -> `input[0]` | `[-0.48, -0.46, ..., 1.50]` s, the 100 bin-right-edge times used by reference behavior interpolation | `get_behavior_per_interval` | Name `time_since_stimulus_onset_s`; time-varying."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond defining the grid: `np.linspace(-0.5 + 0.02, 1.5, 100)`, cast to float32 and broadcast across trials. This is deliberately the *same* expression the reference `get_behavior_per_interval` uses for its interpolation grid (`np.linspace(interval_beg + binsize, interval_end, n_bins)`), i.e. the right edge of each bin rather than its centre.

ii.
```python
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```
```python
        time_input = np.broadcast_to(REL_SAMPLE_TIMES, (len(source_indices), N_BINS))
        block_input = np.broadcast_to(block_numbers[:, None], (len(source_indices), N_BINS))
        inputs = np.stack((time_input, block_input), axis=1).astype(np.float32, copy=True)
```

iii. CONVERSION_NOTES.md Step 5 decision 2 ("Right-edge time convention") and Step 10's reference-code comparison: "Binning/interpolation | `bincount2D` half-open 20-ms spike counts; behavior at bin right edges | floor-indexed half-open 20-ms counts; identical right-edge linear sampling | Same." Step 10 edge-case check confirms "the sample grid ends exactly at +1.50 s".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Index for index: column *i* of the input is `REL_SAMPLE_TIMES[i]`, which is the closing edge of neural bin *i* (which counts spikes over `[stimOn − 0.5 + 0.02·i, stimOn − 0.5 + 0.02·(i+1))`). Both are built from the same `stimOn_times`, the same `OFF_START/OFF_END`, and the same `BIN_SIZE`, so no separate alignment step exists. The same vector also defines the sampling instants for wheel speed and whisker motion energy, so all four streams share one time axis.

ii.
```python
        query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
```
```python
            bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 1: "Continuous behavior is sampled/interpolated at `stimOn + (-0.5 + 0.02, ..., 1.5)` (100 right-edge samples), while neural count bins cover corresponding half-open 20 ms intervals." Step 5 decision 2: "This makes output samples causal with the spike bin ending at the same time."

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the raw trials table. The task holds the block prior constant within a block, so a change of `probabilityLeft` from one row to the next marks a new block. No block-id column exists in the data.

ii.
```python
        raw_block_numbers = trial_number_in_block(trials["probabilityLeft"].to_numpy())
```
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    changes = np.ones(len(probability_left), dtype=bool)
    if len(probability_left) > 1:
        changes[1:] = ~np.isclose(
            probability_left[1:], probability_left[:-1], rtol=0.0, atol=1e-8,
            equal_nan=False,
        )
```

iii. CONVERSION_NOTES.md Step 5 mapping: "Trial-table `probabilityLeft` runs -> `input[1]` | Zero-based cumulative trial number within each contiguous probabilityLeft block"; Step 3 records the block design from the data paper ("first 90 trials 0.5 prior; then alternating 0.2/0.8 probabilityLeft blocks, 20–100 trials, empirical mean 51").

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Zero-based position within the contiguous run of equal `probabilityLeft`, computed vectorially: a boolean "changes" vector marks run starts, `np.maximum.accumulate` propagates the index of the current run start, and the value is `arange − run_start`. Crucially it is computed on the **raw, unfiltered** trial order and only then indexed by `source_indices`, so a trial that is later dropped by the quality mask still advances the count and the number reflects the animal's true position in the block. Values are float32 and broadcast across all 100 bins. Observed range in the full dataset: [0, 98]. NaN `probabilityLeft` compares unequal under `np.isclose(..., equal_nan=False)` and therefore starts a new run.

ii.
```python
    starts = np.maximum.accumulate(np.where(changes, np.arange(len(probability_left)), 0))
    return (np.arange(len(probability_left)) - starts).astype(np.float32)
```
```python
        raw_block_numbers = trial_number_in_block(trials["probabilityLeft"].to_numpy())
        ...
        block_numbers = raw_block_numbers[source_indices]
        ...
        block_input = np.broadcast_to(block_numbers[:, None], (len(source_indices), N_BINS))
```

iii. CONVERSION_NOTES.md Step 5 mapping: "calculated on raw trial order before filtering, then repeated over 100 bins... preserves gaps caused by filtering rather than silently renumbering valid trials." Step 10: "block ordinal is computed before filtering to preserve experimental position"; the planned boundary check "first trial of each raw probability block has trial-number input 0" was verified by the independent raw checker in `cache/raw_sanity_checks.py`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward/CCW), −1 (rightward) or 0 (no response). No-response trials are already removed by the trial mask; the code asserts the retained values are only ±1 and then recodes +1 → 0 (left) and −1 → 1 (right), as the Decoder Task requires.

ii.
```python
        raw_choice = trials["choice"].to_numpy()[source_indices]
        if not np.all(np.isin(raw_choice, [-1, 1])):
            raise ValueError("Unexpected retained choice value")
        # IBL convention: +1 is a leftward choice and -1 is rightward.
        # Target convention required here: left=0, right=1.
        choices = (raw_choice == -1).astype(np.int8)
```

iii. CONVERSION_NOTES.md Step 4: "Map +1→0 (left), −1→1 (right)". Step 12 documents that this was an *error the AI found and fixed*: "Initial code treated raw −1 as left, but direct inspection of ibllib psychometric and PETH code establishes `+1=left`, `−1=right`. Fixed `convert_data.py`, the independent checker, mapping documentation and distributions. Regenerated both sample and 106-GB full pickles" and reran all validation. Final distribution 50.8 % left / 49.2 % right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the recoding above, plus broadcasting the per-trial value across all 100 bins (so it is stored as a constant time series) and casting to int8. `output_values[0] = ["left", "right"]`.

ii.
```python
        choice_output = np.broadcast_to(choices[:, None], (len(source_indices), N_BINS))
        ...
        outputs = np.stack(
            (choice_output, prior_output, wheel_categories, motion_categories), axis=1,
        ).astype(np.int8, copy=False)
```

iii. CONVERSION_NOTES.md Step 5 decision 1: "Per-trial variables are repeated over time because a single trial array must combine static and dynamic variables; this is explicitly supported by the validator and preserves their per-trial semantics."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, whose three task values 0.2 / 0.5 / 0.8 are mapped to classes 0 / 1 / 2 exactly as the Decoder Task prescribes. The match uses `np.isclose` with an absolute tolerance of 1e-8, and any value that matches none of the three raises rather than silently producing a bad label. The unbiased 0.5 block is retained.

ii.
```python
        raw_prior = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
        priors = np.full(len(raw_prior), -1, dtype=np.int8)
        for value, category in ((0.2, 0), (0.5, 1), (0.8, 2)):
            priors[np.isclose(raw_prior, value, rtol=0.0, atol=1e-8)] = category
        if np.any(priors < 0):
            raise ValueError(f"Unexpected probabilityLeft values: {np.unique(raw_prior[priors < 0])}")
```

iii. CONVERSION_NOTES.md Step 4: "exact mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`"; Step 3 curation: "Preserve the initial unbiased 0.5-prior block because prior is a requested output and the reference loader's `exclude_unbiased=False`." Final distribution 41.7 % / 14.1 % / 44.2 %, matching the pre-filter source distribution 0.4175 / 0.1408 / 0.4417 (Step 9 table).

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the three-way recoding, broadcast across the 100 bins, int8. `output_values[1] = ["0.2", "0.5", "0.8"]`.

ii.
```python
        prior_output = np.broadcast_to(priors[:, None], (len(source_indices), N_BINS))
```
```python
            ["0.2", "0.5", "0.8"],
```

iii. As 6-a; Step 5 decision 1 covers the broadcast of per-trial variables over time.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The raw rotary-encoder position/time pair is converted to a filtered velocity with the same ibllib primitives `SessionLoader.load_wheel` uses, and the speed is the absolute value of that velocity (rad/s).

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
...
    timestamp_path = newest_file(alf, "_ibl_wheel.timestamps.npy")
    position_path = newest_file(alf, "_ibl_wheel.position.npy")
    ...
    raw_times = np.load(timestamp_path, mmap_mode="r")
    raw_position = np.load(position_path, mmap_mode="r")
    if len(raw_times) != len(raw_position):
        raise ValueError("Wheel timestamps and position length mismatch")
```

iii. CONVERSION_NOTES.md Step 1: "`load_target_behavior` … Loads absolute wheel velocity then takes magnitude for wheel speed"; Step 5 mapping names the reference functions `SessionLoader.load_wheel`, `interpolate_position`, `velocity_filtered`, `get_behavior_per_interval`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps. (1) The irregularly sampled wheel position is linearly interpolated onto a uniform 1000 Hz grid with `interpolate_position(..., freq=1000)`. (2) Velocity is computed with `velocity_filtered(position, fs=1000, corner_frequency=20, order=8)` — an order-8 20 Hz Butterworth low pass — exactly the `SessionLoader.load_wheel` defaults. (3) Speed = `np.abs(velocity)`. (4) Per trial the trace is resampled at the 100 grid times `stimOn + REL_SAMPLE_TIMES` by linear interpolation with linear extrapolation outside the sliced support, reproducing `interp1d(..., kind='linear', fill_value='extrapolate')` in the reference. Any residual non-finite sample is replaced with the session's finite median before discretisation (count recorded in metadata).

ii.
```python
    position, times = interpolate_position(raw_times, raw_position, freq=1000)
    velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
    speed = np.abs(velocity)
    sampled, good = sample_behavior(times, speed, stim_times)
```
```python
def _linear_interp_extrapolate(x, y, query):
    """Equivalent to scipy interp1d(..., linear, extrapolate) for 1-D data."""
    out = np.interp(query, x, y).astype(np.float64, copy=False)
    left = query < x[0]
    right = query > x[-1]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        out[left] = y[0] + slope * (query[left] - x[0])
    ...
```
```python
    clean = np.where(finite, values, median)
```

iii. CONVERSION_NOTES.md Step 5 mapping: "Linear resampling of position to 1 kHz; order-8 20-Hz low-pass Butterworth filtered derivative; absolute velocity; reference interpolation at bin right edges"; Step 4: "Reproduce `SessionLoader` wheel processing and timestamp-fix rule; linearly sample bin-right edges." Decision 4 on non-finite samples: "`allow_nans=True` in the reference admits internal NaNs and its decoder later mean-imputes standardized behavior. Here, interpolate as referenced, compute thresholds on finite values, and replace remaining non-finite samples with the session finite median before categorization (therefore the middle class). Record imputation counts." Step 7 records that the plotted 1 kHz trace and the 20 ms samples "overlay exactly".

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes by the **empirical 1/3 and 2/3 quantiles of that session's own retained speed samples** (all trials × all 100 bins pooled), using `np.digitize(..., right=False)`, so classes are 0 = low, 1 = medium, 2 = high and are equally sized within a session by construction. Thresholds are stored per session in `metadata.session_info[i]['wheel_tertiles']`. Tied values are left tied (the threshold rule is kept rather than rank-splitting identical physical values), which is why the realised fractions are 0.333/0.333/0.333 for the wheel.

ii.
```python
def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Continuous behavior has no finite values")
    thresholds = np.quantile(values[finite], [1 / 3, 2 / 3]).astype(np.float64)
    median = float(np.median(values[finite]))
    imputed = int(np.size(values) - np.count_nonzero(finite))
    clean = np.where(finite, values, median)
    categories = np.digitize(clean, thresholds, right=False).astype(np.int8)
    return categories, thresholds, imputed, median
```
```python
        wheel_categories, wheel_thresholds, wheel_imputed, wheel_median = discretize_tertiles(
            wheel_values
        )
```

iii. CONVERSION_NOTES.md Step 5 decision 5: "Use 1/3 and 2/3 empirical quantiles separately per session and output. Reference decoding standardizes behavior per session, and whisker energy has camera/session-dependent arbitrary scale; session tertiles are the categorical analogue. Use `np.digitize(..., right=False)` and record thresholds/class fractions. If thresholds tie, retain the threshold rule and report imbalance rather than rank-splitting equal physical values." Step 4 flags this as a task-mandated transform: the reference regresses the continuous trace, while the Decoder Task requires three bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The speed trace is evaluated at `stimOn + REL_SAMPLE_TIMES` — the identical 100 time points used for the time input and for the closing edges of the neural bins — so column *i* of the wheel output corresponds to neural bin *i*. A trial is only kept if the wheel stream has ≥2 samples inside the window and its first/last in-window samples are within one bin of the window edges, so no sample is produced by long-range extrapolation.

ii.
```python
    beginnings = stim_times + OFF_START
    endings = stim_times + OFF_END
    ibeg = np.searchsorted(times, beginnings, side="right")
    iend = np.searchsorted(times, endings, side="left")
    ...
        query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
        sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
        good[trial] = True
```

iii. CONVERSION_NOTES.md Step 1: the reference `get_behavior_per_interval` "Selects the same event-relative window, checks endpoint coverage, then linearly interpolates behavior at bin-right-edge times" — the AI reproduces both the window checks and the grid. Step 7 plot review: "Filtered 1-kHz wheel traces and 20-ms samples overlay exactly."

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy.npy` (the IBL-released mean pixelwise frame difference over a whisker-pad ROI) with its frame times `_ibl_<side>Camera.times.npy`. The **left** camera is tried first and the **right** used as fallback; the chosen view is recorded per session in `metadata.session_info[i]['whisker_camera']`. If neither view yields a usable stream the whole session is skipped (14 sessions).

ii.
```python
def load_motion_samples(alf, stim_times):
    failures: list[str] = []
    for view in ("left", "right"):
        value_path = newest_file(alf, f"{view}Camera.ROIMotionEnergy.npy")
        time_path = newest_file(alf, f"_ibl_{view}Camera.times.npy")
        if value_path is None or time_path is None:
            failures.append(f"{view}: files missing")
            continue
        ...
    raise FileNotFoundError("No usable whisker motion stream; " + "; ".join(failures))
```

iii. CONVERSION_NOTES.md Step 1: `bin_behaviors` "prefers left camera, falls back to right" — this is exactly the reference's `if 'skip' in target_dict.keys(): … 'right-whisker-motion-energy'` logic. Step 3: "Whisker motion energy is the mean pixelwise absolute difference between adjacent frames in a whisker-pad bounding box anchored between the DLC nose tip and eye. Left video is 60 Hz; right is 150 Hz."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is: no filtering, smoothing or normalisation. One repair is applied — when there are more camera timestamps than motion-energy values (pre-GPIO sessions drop the first frames) the *leading* timestamps are trimmed, which is precisely `SessionLoader._check_video_timestamps`; the reverse case (timestamps shorter than data) is treated as unusable and falls through to the other camera. The trace is then resampled at the same 100 per-trial grid points with the same linear interpolation/extrapolation as the wheel, and non-finite samples are median-imputed.

ii.
```python
        values = np.load(value_path, mmap_mode="r")
        times = np.load(time_path, mmap_mode="r")
        if len(times) < len(values):
            failures.append(f"{view}: timestamps shorter than data")
            continue
        if len(times) > len(values):
            times = times[-len(values):]
        sampled, good = sample_behavior(times, values, stim_times)
```

iii. CONVERSION_NOTES.md Step 4: "left/right camera lengths can differ and timestamps may need leading trim… Reproduce `SessionLoader` wheel processing and timestamp-fix rule". Step 5 mapping: "Apply timestamp leading-trim fix, reference linear interpolation at bin right edges, session-specific finite-value tertile classes". Step 7 plot review: "Raw camera energy and 20-ms interpolation likewise overlay at the expected 60-Hz sampling density."

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: the same `discretize_tertiles` helper, per session, on that session's retained motion-energy samples, thresholds at the empirical 1/3 and 2/3 quantiles, `np.digitize(..., right=False)`, classes labelled low/medium/high, thresholds stored in `metadata.session_info[i]['whisker_tertiles']`. Ties (motion energy is more discretised than wheel speed) leave a small imbalance — the realised full-dataset fractions are 0.333 / 0.333 / 0.335.

ii.
```python
        motion_categories, motion_thresholds, motion_imputed, motion_median = discretize_tertiles(
            motion_values
        )
```
```python
            "dynamic_output_discretization": (
                "within-session empirical 1/3 and 2/3 quantiles over retained finite samples"
            ),
```

iii. Same as 7-c (Step 5 decision 5); Step 9 notes the tie excess explicitly: "`[0.333,0.333,0.335]` | Yes; minor tie excess documented."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: sampled at `stimOn + REL_SAMPLE_TIMES`, one value per neural bin, with the same window-coverage requirement (≥2 in-window frames and both endpoints within 20 ms of the window edges) before the trial is accepted. Camera frame times are on the same synchronised session clock as the spikes, so evaluating the trace at those instants is the whole alignment.

ii.
```python
        wheel_values, wheel_good, wheel_aux = load_wheel_samples(alf, stim_code)
        motion_values, motion_good, motion_aux = load_motion_samples(alf, stim_code)
        stream_good = wheel_good & motion_good & neural_good
```
```python
        query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
        sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
```

iii. As 7-d. Step 10 lists an independent raw-file check ("6 input and 12 output comparisons" via `np.allclose` in `cache/raw_sanity_checks.py`) that recomputes the motion interpolation and thresholding from the native `.npy` files without importing the converter.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered and explicit:
- **Whole-session failures** are caught per session; the traceback is printed and the session is recorded in `metadata.skipped_sessions` with a reason rather than crashing the run (15 of 459 skipped: 14 with no usable whisker stream, 1 with no jointly covered trial).
- **Missing whisker camera**: left → right fallback; only if both fail is the session dropped. The notes state that "inventing/imputing an entire requested target would be scientifically invalid".
- **Camera timestamp/value length mismatch**: leading timestamps trimmed (`times[-len(values):]`), matching `SessionLoader._check_video_timestamps`; the un-fixable direction is rejected.
- **Trials with incomplete stream coverage** (wheel, camera, or outside the probes' common spike interval) are dropped, not extrapolated.
- **Isolated non-finite behavioural samples** are replaced by the session's finite median (→ middle class) and the count plus the imputation value are stored (`wheel_imputed_samples`, `whisker_imputed_samples`, `*_imputation_median`).
- **Non-contiguous / non-zero-based spike cluster ids** are mapped through the cluster table by `searchsorted`, raising if an id is absent.
- **Structural guards**: duplicate cluster ids, cluster-channel length mismatch, out-of-range channel index, spike times/clusters length mismatch, wheel times/position length mismatch, unexpected choice or `probabilityLeft` values, shape assertions, non-finite neural/input, and out-of-range output categories all raise.
- **Sessions with <2 usable trials** are skipped (the format requires ≥2), and probes with zero spikes raise.

ii.
```python
    except Exception as exc:
        return {"ok": False, "index": index, "eid": info["eid"],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), ...}
```
```python
                skipped.append({"eid": result["eid"], "reason": result["error"],
                                "freeze_index": result["index"]})
```
```python
        if not np.all(np.isfinite(neural)) or not np.all(np.isfinite(inputs)):
            raise ValueError("Non-finite neural/input values")
        if outputs.min() < 0 or outputs.max() > 2:
            raise ValueError("Output category outside expected range")
```
```python
    positions = np.searchsorted(sorted_ids, raw_clusters)
    if np.any(positions >= len(sorted_ids)) or np.any(sorted_ids[positions] != raw_clusters):
        raise ValueError("Spike cluster ID absent from cluster table")
```

iii. CONVERSION_NOTES.md Step 5 decision 4 (median imputation, quoted in 7-b), Step 10 "Issues Found and Resolved" (trailing trials after electrophysiology stopped; missing whisker stream), and Step 10 check 5: "Checked first/last raw and converted trials, half-open bin endpoints, exact +1.50-s right-edge behavior sample, block resets before filtering, non-contiguous cluster IDs, multi-probe offsets, camera timestamp leading trim, left-camera preference/right fallback, tied tertiles, sessions with unavailable behavior, and common spike-recording bounds."

## 10-a. What are the most time-consuming steps of the code?

i. The script times each session and prints the split. Per session, **spike loading + binning dominates** (0.9 s to 7.3 s, scaling with trials × clusters) and behavioural loading/resampling is second (0.4–1.2 s, dominated by the whole-session 1 kHz wheel interpolation and Butterworth filter). Across the whole run, the single largest wall-clock item is **pickling and writing the 106 GB output file**: the 459-session conversion finished in 208.9 s total with 24 worker threads, of which the session work is roughly half. `BrainRegions()` construction and the atlas mapping add a per-probe cost. The AI explicitly measured these and reported the estimates in Step 7.

ii.
```python
        t0 = time.perf_counter()
        wheel_values, wheel_good, wheel_aux = load_wheel_samples(alf, stim_code)
        motion_values, motion_good, motion_aux = load_motion_samples(alf, stim_code)
        ...
        behavior_seconds = time.perf_counter() - t0
        ...
        t1 = time.perf_counter()
        ...
        neural_seconds = time.perf_counter() - t1
```
```python
                    f"{result['elapsed']:.1f}s "
                    f"(behavior {result['info']['behavior_processing_seconds']:.1f}s, "
                    f"neural {result['info']['neural_processing_seconds']:.1f}s)",
```

iii. CONVERSION_NOTES.md Step 7 run-time table: "Behavior processing | 0.2 s/session sample… | ~2 min", "Neural processing | 0.2–0.3 s/session sample… | ~2 min with 24 workers", "Pickle assembly/write | Sample implies ~0.2–0.25 GB/s; full estimated ~100–110 GB | ~7–9 min", total "~10–13 min, below 15-min optimization threshold". Step 6 lists the mitigations: "Memory-map large spike arrays; binary-search only retained trial windows; batch multiple trials into one vectorized `np.bincount`; cap batches by element count; parallelize independent sessions."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two remain:
1. `sample_behavior` loops over trials, calling `np.interp` (plus the extrapolation fix-up) once per trial. Because all trials share the same relative grid, one `np.interp` over a single concatenated query vector would produce the whole matrix.
2. `bin_probe` still loops over trials to build the flat bincount codes — two `searchsorted` calls, a cluster-id mapping and a `floor` per trial — even though the `np.bincount` itself is already batched across a chunk of trials. Both `searchsorted` calls are vectorisable over all trial starts/ends at once (as the surrounding code already does for behaviour), and the bin index could be computed in one shot with a per-spike trial offset.
Additionally, the probe loop in `process_session` opens `spikes.times` once to read only its first and last element and then `bin_probe` re-opens it, and `load_probe_metadata` runs once per probe in a Python loop.
These are modest: the AI's own timings show binning at a few seconds per session, and the full run is 209 s, so the dominant remaining cost is I/O and pickling rather than these loops.

ii.
```python
    for trial in range(len(stim_times)):
        lo, hi = int(ibeg[trial]), int(iend[trial])
        ...
        sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
```
```python
        for local_trial, trial in enumerate(range(first, last)):
            lo = int(np.searchsorted(spike_times, starts[trial], side="left"))
            hi = int(np.searchsorted(spike_times, ends[trial], side="left"))
```

iii. CONVERSION_NOTES.md Step 6 "Code speedups added": "binary-search only retained trial windows; batch multiple trials into one vectorized `np.bincount`; cap batches by element count… Vectorize target/input construction and block numbering". The AI vectorised input/output construction and the bincount but kept the per-trial slicing loops, stating that reference code's per-trial process pool was the real inefficiency ("Reference code launches a process pool for every trial-level operation and bins one trial at a time, repeatedly initializing workers and dense arrays").

## 10-c. What processing does the code repeat multiple times?

i. Little that affects the output, but three real repeats:
1. `BrainRegions()` is instantiated **inside** `load_probe_metadata`, i.e. once per probe (699 times over the full run), each time re-reading the Allen/Beryl atlas tables, even though the object is stateless and could be built once at module level (the reference solution does exactly that).
2. `map_spike_clusters` is called once **per trial** inside `bin_probe`; when cluster ids are not `0..n-1` it recomputes `np.argsort(cluster_ids)` on every call, although the mapping is a fixed per-probe property.
3. `spikes.times.npy` is memory-mapped twice per probe — once in `process_session` purely to read `spike_times[0]` and `spike_times[-1]` for the coverage bounds, and again in `bin_probe`.
Also, `load_wheel_samples`/`load_motion_samples` always build and return the full-session `aux` arrays (times/values) for plotting even when `--show-processing` is not requested. None of these change the converted values.

ii.
```python
def load_probe_metadata(alf: Path, probe_name: str) -> dict:
    ...
    regions = BrainRegions()
    allen = regions.id2acronym(channel_atlas_ids[cluster_channels.astype(int)])
```
```python
            clusters = map_spike_clusters(raw_clusters, probe["cluster_ids"])
```
```python
    order = np.argsort(cluster_ids)
    sorted_ids = cluster_ids[order]
```
```python
        for probe in probes:
            spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
            ...
            probe_starts.append(float(spike_times[0]))
            probe_ends.append(float(spike_times[-1]))
```

iii. The AI did not document these particular repeats. Its Step 6 notes claim the opposite direction of effort — "avoid redundant file loads", "Memory-map large spike arrays" — and the repeats it *did* document are the reference code's: "Reference code launches a process pool for every trial-level operation and bins one trial at a time, repeatedly initializing workers and dense arrays. It also loads spikes before discovering behavior-stream failures." The AI did reorder its own pipeline so behaviour is processed before neural data so that failures avoid loading spikes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest item follows from decision 2-c: all 599,865 Kilosort clusters are binned and stored, including **85,656 `root`, 12,827 `void`, 344 `x` and 18 `y`** units. `void` is the atlas label for a site the histology placed outside the brain, and the reference repository's own region analysis explicitly excludes both (`data_loader_utils.py`: `[roi for roi in np.unique(...) if roi not in ['root', 'void']]`), so roughly 99,000 neurons are processed and written but would be dropped by any region-resolved downstream analysis. Storing counts as `float32` rather than a small integer type multiplies the neural payload ~4× (final pickle 106.081 GB). Other minor items: `n_good` (the `label >= 1` count) is computed for every probe but never used to select anything; the full-session `aux` traces are assembled on every session even without `--show-processing`; the wheel position is interpolated to 1 kHz and Butterworth-filtered over the **entire session** although only ~2 s per trial is sampled (this one is arguably necessary, since the filter needs continuous support); `_linear_interp_extrapolate` runs a full `np.interp` and then overwrites the out-of-range entries; and `source_trial_indices` is stored as a Python list for all 188,922 trials.

ii.
```python
        "n_good": int((metrics["label"].to_numpy() >= 1).sum()),
```
```python
        neural = np.zeros((len(source_indices), n_neurons, N_BINS), dtype=np.float32)
```
```python
    aux = {"times": times, "values": speed,
           "timestamp_path": str(timestamp_path), "position_path": str(position_path)}
```
```python
    out = np.interp(query, x, y).astype(np.float64, copy=False)
    left = query < x[0]
    right = query > x[-1]
```

iii. The AI's justification for keeping everything is in Step 4/Step 5 decision 6: "Preserve 621,733 clusters because the provided neural-decoder code explicitly loads `qc=None` and the method paper says all neurons. Quality labels are checked but not used to filter", and Step 4: "Retain `void/root` if produced because code selects all regions." Step 10 acknowledges the consequence: "The paper's 279 recorded structures is based on its native anatomical reporting/curation, whereas the 281-item output is the supplied code's Beryl mapping with `root`, `void`, `x`, and `y` retained under all-region selection." Step 5 decision 8 anticipated the size: "The estimated all-cluster neural payload before pickle overhead is 109.85 GB."
