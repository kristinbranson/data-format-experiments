# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Nothing is read from disk directly. The session/insertion inventory comes from the frozen release table shipped with the methods repository, `code/code_zhang2025/data/bwm_release.csv`, which lists one row per probe insertion with `eid`, `pid`, `probe_name`, `subject`, `date`, `session_number`. If `/app/data/DATALIMIT_SUBSET.csv` exists the eids are restricted to that subset (the release table always lists the full release). Every actual array is then pulled through the IBL ONE API against the staged cache in `/app/data/one_cache`, using an authenticated `ONE` client (no password, so the staged Alyx token and cached REST replies are used) rather than a purely local `One`; the agent found that the static parquet release table alone does not resolve the newest ALF revisions of the trials table and motion-energy objects. `SessionLoader` supplies trials, wheel and camera motion energy; `SpikeSortingLoader` supplies spikes/clusters/channels, called once per `pid`. Sessions are processed one at a time, in release-table order, in a single process; any session that raises is skipped and the reason is recorded in `metadata['skipped_sessions']`. Result: 459 sessions examined, 444 converted, 188,925 trials, 136 subjects.

ii.
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
SUBSET_CSV = ROOT / "data" / "DATALIMIT_SUBSET.csv"

def load_release_table(max_sessions=None):
    release = pd.read_csv(RELEASE_CSV)
    if SUBSET_CSV.exists():
        ...
        release = release[release["eid"].astype(str).isin(allowed)]
    eids = _ordered_unique(release["eid"].astype(str))
    ...
```
```python
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        silent=True,
        cache_dir=str(CACHE_DIR),
    )
```
```python
    session_loader = SessionLoader(one=one, eid=eid)
    session_loader.load_trials()
...
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
    spikes, clusters, channels = loader.load_spike_sorting()
```

iii. From the trajectory: "The supplied cache is the full Brain-Wide Map release rather than a tiny example… use the paper's stimulus-aligned −0.5 to +1.5 s window, 20 ms bins, the standard valid-trial mask… and the repository's all-cluster spike processing" (step 13); "I've resolved an important versioning detail in the cache: the newest trial tables and motion-energy arrays live in revisions that the static release table alone does not select. The converter will use the authenticated offline ONE cache interface (without downloading source data), matching the repository's loaders, and will record every skipped session and its reason in metadata" (step 40). The release CSV is the same insertion list the methods repository itself uses, so using it reproduces the paper's session/probe set exactly.

## 1-b. How are the data split into subjects (mice)?

i. The subject name is taken from the `subject` column of the release table (one value per eid), so no path parsing is needed. After conversion, `subjects` is the sorted set of names of the successfully converted sessions and `subject_idx[i]` is the index of session *i*'s subject in that list. 136 subjects over 444 sessions.

ii.
```python
    info = {
        "eid": eid,
        "subject": str(eid_rows.iloc[0]["subject"]),
        ...
```
```python
    subjects = sorted({info["subject"] for info in session_info})
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    subject_idx = np.asarray(
        [subject_lookup[info["subject"]] for info in session_info], dtype=np.int64
    )
```

iii. Not discussed explicitly in the trajectory; the subject identity is metadata carried by the release table, so nothing has to be derived or inferred.

## 1-c. How are the data split into sessions?

i. A session is the unit the release is organised by. The release table is grouped by `eid` (`_ordered_unique` preserves release-table order), and all rows sharing an eid — i.e. all probe insertions of that session — are handed to `make_session` together, so the two probes of a session become one pooled population rather than two sessions. One converted session = one eid.

ii.
```python
    eids = _ordered_unique(release["eid"].astype(str))
...
    for number, eid in enumerate(eids, start=1):
        rows = release[release["eid"].astype(str) == eid]
        neural, decoder_input, decoder_output, regions, info = make_session(one, rows)
```
```python
    for row in eid_rows.itertuples(index=False):
        counts, regions = bin_probe_spikes(one, eid, str(row.pid), str(row.probe_name), stimulus_times)
        probe_counts.append(counts)
        probe_regions.append(regions)
    neural_3d = np.concatenate(probe_counts, axis=1).astype(np.float32)
```

iii. Trajectory step 56: the smoke test "confirms that probe merging and Beryl region mapping are working across one- and two-probe sessions". This follows the repository's `merge_probes`/`prepare_data`, which merges probes because two probes in one session are not independent recordings.

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` already has one row per trial, so no splitting is done. Each retained row becomes one entry in the per-session `neural`/`input`/`output` lists, and the trial window is `stimOn_times + [-0.5, 1.5)` s. A session is rejected outright if the source table has fewer than two rows.

ii.
```python
    session_loader.load_trials()
    trials = session_loader.trials.copy()
    if len(trials) < 2:
        raise ValueError("fewer than two source trials")
...
    stimulus_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
```

iii. No justification needed/given: the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Two stages, both at the level of the trial.

  1. `valid_trial_mask` reimplements the repository's `load_trials_and_mask` with the defaults that `prepare_data` uses: all of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType, goCue_times` must be finite (the `nan_exclude='default'` list); reaction time `firstMovement_times - stimOn_times` must lie in [0.08, 2.00] s; trial duration `feedback_times - goCue_times` must be ≤ 10 s (`max_trial_len=10.0`, as passed in `prepare_data`); `choice != 0` (`exclude_nochoice=True`). The initial unbiased (0.5) block is *kept* (`exclude_unbiased=False`).
  2. Trials whose [-0.5, 1.5) s window is not fully covered by both the wheel trace and the whisker camera trace are dropped, using the repository's coverage test (at least 2 samples in the window, and the first/last sample within one bin of the window edges, and a finite interpolation).

A session needs ≥ 2 trials after stage 1 and ≥ 2 after stage 2 or it is skipped. Retained-trial indices are written into the metadata per session. Trials with no spikes are *not* dropped (3 such trials survive and produce the validator's only warnings).

ii.
```python
def valid_trial_mask(trials):
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType", "goCue_times"]
    ...
    mask = np.all(np.isfinite(vals), axis=1)
    reaction_time = (trials["firstMovement_times"].to_numpy(dtype=float)
                     - trials["stimOn_times"].to_numpy(dtype=float))
    duration = (trials["feedback_times"].to_numpy(dtype=float)
                - trials["goCue_times"].to_numpy(dtype=float))
    mask &= reaction_time >= 0.08
    mask &= reaction_time <= 2.0
    mask &= duration <= 10.0
    mask &= trials["choice"].to_numpy(dtype=float) != 0
    return mask
```
```python
    source_mask = valid_trial_mask(trials)
    source_indices = np.flatnonzero(source_mask)
    if len(source_indices) < 2:
        raise ValueError("fewer than two trials pass the paper's trial mask")
...
    wheel, motion, behavior_good, motion_view = load_behaviors(one, eid, stimulus_times)
    source_indices = source_indices[behavior_good]
    ...
    if len(source_indices) < 2:
        raise ValueError("fewer than two trials have complete behavior coverage")
```

iii. The docstring states the intent: "Implement the repository's `load_trials_and_mask` defaults. In addition to the default required fields and 0.08–2.00 s reaction-time interval, `prepare_data` passes `max_trial_len=10` s. No-choice trials are excluded. The initial unbiased block is retained." Trajectory step 13 lists "the standard valid-trial mask (including 80 ms–2 s first-movement latency)" as a key constraint, and step 67: "The observed retained-trial counts vary as expected after the paper's reaction-time/event-completeness mask". Step 239 on coverage: a session with "camera files but no interval with valid full-window coverage… is excluded rather than padded or extrapolated across missing data."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, one pair per probe insertion (`pid`). The merged cluster table (`SpikeSortingLoader.merge_clusters(..., compute_metrics=False)`) is used only for the `acronym` column, i.e. the anatomical label of each cluster; its `label` (QC) column is loaded but never used.

ii.
```python
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
    spikes, clusters, channels = loader.load_spike_sorting()
    cluster_table = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False
    ).to_df()

    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"], dtype=np.int64)
    cluster_ids = np.unique(spike_clusters)
```
```python
    regions = cluster_table.iloc[cluster_ids]["acronym"].fillna("void").astype(str)
    beryl = BrainRegions().acronym2acronym(regions.to_numpy(), mapping="Beryl")
```

iii. Not argued separately; these are the two arrays the repository's `load_spiking_data`/`bin_spiking_data` use (`neural_dict['spike_times']`, `neural_dict['spike_clusters']`, `cluster_regions`). Trajectory step 89: "The converter is retaining all Kilosort clusters exactly as the methods code does, while assigning each neuron its Beryl region".

## 2-b. How is the `neural` data processed?

i. Spikes of a trial are histogrammed into 100 non-overlapping 20 ms bins spanning `stimOn_times + [-0.5, 1.5)` s, per cluster, using a single `np.bincount` over a flattened (cluster, bin) index. Values stay **raw spike counts** (accumulated as `uint16`, cast to `float32` at the end) — they are *not* divided by the bin width, so the stored unit is counts/20 ms rather than Hz. No smoothing, no z-scoring, no PCA. Clusters that never spike in the session are implicitly excluded (`cluster_ids = np.unique(spike_clusters)`). When a session has several probes, the per-probe count arrays are concatenated along the neuron axis so the population is pooled, and the Beryl region labels are concatenated in the same order. Each neuron's Allen acronym is mapped to the coarser Beryl parcellation for `brain_regions`/`brain_region_idx`.

ii.
```python
    counts = np.zeros((len(stimulus_times), len(cluster_ids), N_TIME), dtype=np.uint16)
    for trial_i, stimulus_time in enumerate(stimulus_times):
        begin = stimulus_time + OFF_START_S
        end = stimulus_time + OFF_END_S
        ib = np.searchsorted(spike_times, begin, side="left")
        ie = np.searchsorted(spike_times, end, side="left")
        if ie <= ib:
            continue
        local_time = spike_times[ib:ie]
        bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
        local_cluster = np.searchsorted(cluster_ids, spike_clusters[ib:ie])
        in_range = (bin_idx >= 0) & (bin_idx < N_TIME)
        flat = local_cluster[in_range] * N_TIME + bin_idx[in_range]
        hist = np.bincount(flat, minlength=len(cluster_ids) * N_TIME)
        counts[trial_i] = hist.reshape(len(cluster_ids), N_TIME)
```
```python
    neural_3d = np.concatenate(probe_counts, axis=1).astype(np.float32)
    region_labels = np.concatenate(probe_regions)
```
metadata: `"neural_measurement": "Kilosort 2.5 spike counts in non-overlapping bins"`.

iii. The method paper's wording is the justification the agent adopted: trials are "divided into 20-ms bins" and `X` is constructed "by aggregating spike counts", and the repository's `bin_spiking_data` stores counts. Trajectory step 89: "retaining all Kilosort clusters exactly as the methods code does, while assigning each neuron its Beryl region"; step 228: "All completed sessions remain aligned to the same −0.5 to +1.5 s window with 20 ms bins."

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control is applied at all.** Every sorted cluster with at least one spike is kept, including multi-unit clusters that fail the IBL single-unit metrics, and including clusters whose Beryl acronym is `void` (histology placed the channel outside the brain) or `root`. A NaN acronym is relabelled `"void"` and kept as a brain region. The code explicitly documents this: `"neuron_filter": "all sorted clusters (no cluster-QC threshold)"` and the comment "All sorted clusters are retained; no QC label filter." The consequence is 599,865 neurons over 444 sessions (mean 1,351, max 3,140 per session), of which 12,175 are `void` and 86,127 are `root`, and a 106 GB pickle.

ii.
```python
    # Searchsorted maps arbitrary (though usually consecutive) Kilosort IDs to
    # rows in the output.  All sorted clusters are retained; no QC label filter.
    counts = np.zeros((len(stimulus_times), len(cluster_ids), N_TIME), dtype=np.uint16)
```
```python
    regions = cluster_table.iloc[cluster_ids]["acronym"].fillna("void").astype(str)
    beryl = BrainRegions().acronym2acronym(regions.to_numpy(), mapping="Beryl")
```
```python
            "neuron_filter": "all sorted clusters (no cluster-QC threshold)",
```

iii. Trajectory step 13: "…and the repository's all-cluster spike processing". Step 104: "The resulting dataset is intentionally large because the reference method keeps all sorted clusters rather than only well-isolated units; this matches both the paper text and its released preprocessing code." Step 284: "That size is expected from dense float32 matrices for every retained trial and all sorted clusters." The claim is checkable: the repository's `load_spiking_data(one, pid, ..., qc=None)` default keeps all clusters and `prepare_data` never passes `qc`, and `methods.txt` for the method paper says "we bin spike counts using all neurons, sorted by Kilosort 2.5, from each session".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times` of each trial, as required ("Temporally align based on stimulus onset"). All IBL streams share one session clock, so alignment is a subtraction: for each trial the spike-time search window is `[stimOn - 0.5, stimOn + 1.5)` and the bin index of each spike is `floor((t - (stimOn - 0.5)) / 0.02)`. Bins are half-open on the right (`side="left"` searchsorted at both ends), and out-of-range indices are masked rather than clipped. Every trial therefore has exactly 100 bins with bin 25 starting at the onset.

ii.
```python
OFF_START_S = -0.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))   # 100
```
```python
        begin = stimulus_time + OFF_START_S
        end = stimulus_time + OFF_END_S
        ib = np.searchsorted(spike_times, begin, side="left")
        ie = np.searchsorted(spike_times, end, side="left")
        ...
        bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
        in_range = (bin_idx >= 0) & (bin_idx < N_TIME)
```
metadata: `"temporal_alignment_event": "visual stimulus onset (stimOn_times)"`, `off_start=-0.5`, `off_end=1.5`.

iii. Module docstring: "trials are aligned to stimulus onset, span [-0.5, 1.5) s". Trajectory step 13 identifies "the paper's stimulus-aligned −0.5 to +1.5 s window" as the constraint; this is the method paper's choice window for choice decoding and is what the instructions require.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (`time_bin_size = 20.0` ms), 100 bins per trial, identical for every trial and session. Spikes are binned once, directly at this resolution, from raw spike times — there is no intermediate binning and no resampling/rebinning of the neural data. The 50 ms bins the method paper uses for *prior* decoding are deliberately not used, because one shared representation has to serve all four targets.

ii.
```python
BIN_SIZE_S = 0.020
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))   # 100
```
```python
            "time_bin_size": BIN_SIZE_S * 1000.0,
            "time_bin_size_units": "ms",
```

iii. Module docstring: "the four requested targets can share one representation: trials are aligned to stimulus onset, span [-0.5, 1.5) s, and use non-overlapping 20 ms spike-count bins". The method paper states "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps", and the repository config uses `binsize: 0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` only, indirectly: the variable is the analytic time axis of the binning grid defined relative to each trial's `stimOn_times`. It is computed once at module level and is the same vector for every trial in every session.

ii.
```python
TIME_FROM_STIMULUS = (
    OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
)
```
```python
    stimulus_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
```

iii. Implicit: `stimOn_times` is the alignment event, so "time since stimulus onset" is fixed by the window and the bin size, both taken from the reference code.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The values are the **left edge** of each 20 ms bin, i.e. −0.50, −0.48, …, 1.48 s (the observed input range in the verification stats is [−0.5, 1.48]). It is kept continuous (not binarised), stored as `float32`, and tiled into row 0 of each trial's `(2, 100)` input array.

ii.
```python
        decoder_input.append(
            np.vstack(
                [TIME_FROM_STIMULUS, np.full(N_TIME, trial_in_block[i], np.float32)]
            ).astype(np.float32, copy=False)
        )
```
```python
            "input_names": ["time since stimulus onset", "trial number in block"],
```

iii. Not argued; the instructions list this input as "continuous, time-varying", so a ramp over the trial window is the direct implementation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the neural grid itself: entry *t* of the input labels the same bin *t* whose spikes are counted in column *t* of the neural matrix, so the two are aligned bin-for-bin by construction, in every trial and session (all arrays are `(·, 100)`). The label used is the bin's start time; the spike window for that bin is `[label, label + 0.02)`.

ii.
```python
TIME_FROM_STIMULUS = OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
```
```python
        bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
```

iii. Not argued separately. (Note the internal asymmetry: the neural/time grid is labelled by bin start, while the two dynamic behavioural outputs are sampled at the bin *end* — see 7-d/8-d — a 20 ms offset between the time label and the behaviour sample within the same bin.)

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The trials table carries no block index, so blocks are recovered as maximal runs of constant `probabilityLeft`.

ii.
```python
    block_number = trial_numbers_in_block(
        trials["probabilityLeft"].to_numpy(dtype=float)
    )
```

iii. Not argued; `probabilityLeft` is constant within a block by task design (the data paper: blocks of 20–100 trials at 20:80 or 80:20, alternating), so a change of value marks a block boundary.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A running counter that resets to 1 whenever `probabilityLeft` changes, i.e. one-indexed position within the block. Crucially it is computed on the **full, unfiltered** trials table and only then indexed by the retained-trial indices, so a trial that is later dropped still advances the counter and the value remains the animal's true ordinal in the block. It is a per-trial scalar, broadcast to all 100 bins as row 1 of the input array, stored as `float32`. Observed range 1–99.

ii.
```python
def trial_numbers_in_block(probability_left):
    """One-indexed trial ordinal within each contiguous probability block."""
    out = np.ones(len(probability_left), dtype=np.float32)
    for i in range(1, len(out)):
        if probability_left[i] == probability_left[i - 1]:
            out[i] = out[i - 1] + 1.0
    return out
```
```python
    block_number = trial_numbers_in_block(trials["probabilityLeft"].to_numpy(dtype=float))
    source_mask = valid_trial_mask(trials)
    source_indices = np.flatnonzero(source_mask)
    ...
    trial_in_block = block_number[source_indices]
```

iii. Trajectory step 89: "…preserving original trial ordinals for the 'trial number in block' input." Metadata records `"trial_number_indexing": "one-indexed within contiguous probabilityLeft blocks"`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single `choice` column of the trials table, restricted to the retained trials. `choice == 0` (no response) has already been removed by the trial mask, and the code hard-asserts that only ±1 remain (a session with any other value is skipped entirely).

ii.
```python
    choice_raw = trials["choice"].to_numpy(dtype=float)[source_indices]
    if not np.all(np.isin(choice_raw, [-1.0, 1.0])):
        raise ValueError("choice contains values other than -1 and +1")
```

iii. Not argued; `choice` is the released per-trial variable and the repository's `bin_behaviors` also takes `trials_df['choice']` directly.

## 5-b. What processing is involved in computing `output` *Choice*?

i. A binary recode: `choice == +1 → 1`, `choice == -1 → 0`, stored as `int8` and broadcast to all 100 time bins (row 0 of the output array), with `output_values[0] = ["left", "right"]` (so 0 is declared to mean *left* and 1 *right*). In the IBL convention `choice = +1` is a **leftward** choice and `choice = -1` a rightward one (`ibllib` `Choice` extractor: `choice[t] = -sign(position[t])` for correct trials, with `position < 0` = stimulus on the left; `brainbox.behavior.training`: `rightward = trials.choice == -1`). The agent's mapping therefore assigns label 1 ("right") to left choices and 0 ("left") to right choices, i.e. the two class labels are swapped relative to the instruction "left = 0, right = 1". This is visible in the statistics: the reference gets `choice` fractions {0: 0.5075, 1: 0.4925} and this conversion gets the mirror image {0: 0.4921, 1: 0.5079}.

ii.
```python
    choice = (choice_raw == 1.0).astype(np.int8)
...
        decoder_output.append(
            np.vstack([
                np.full(N_TIME, choice[i], np.int8),
                ...
```
```python
        "output_values": [
            ["left", "right"],
            ...
```

iii. No justification appears anywhere in the trajectory or the code: the sign convention is never checked, stated or tested, and the only validation is that the values are ±1. The trajectory's quality checks only assert that "All retained sessions still have both choice classes" (step 138), which the flipped mapping also satisfies.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, restricted to the retained trials.

ii.
```python
    prior_raw = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
```

iii. Not argued; this is the block prior the task holds constant within a block, and the instructions name it directly.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recode to the three ordinal categories the instructions specify — 0.2 → 0, 0.5 → 1, 0.8 → 2 — with `np.isclose` to avoid float equality problems, and a hard failure (whole session skipped) if any value matches none of the three. The unbiased 0.5 block is kept as its own class. Values are `int8` and broadcast to all 100 bins (row 1 of the output array); `output_values[1] = ["0.2", "0.5", "0.8"]`. Observed fractions {0: 0.417, 1: 0.141, 2: 0.442}, essentially identical to the reference.

ii.
```python
    prior = np.full(len(prior_raw), -1, dtype=np.int8)
    for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
        prior[np.isclose(prior_raw, value)] = label
    if np.any(prior < 0):
        raise ValueError(f"unexpected probabilityLeft values {np.unique(prior_raw)}")
```

iii. Directly from the task specification ("Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2"). Trajectory step 138 verifies "all three block-prior classes" are present in every retained session.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. `_ibl_wheel.position` / `_ibl_wheel.timestamps` after the loader's standard preprocessing; specifically the `times` and `velocity` columns of `loader.wheel`. Speed is the absolute value of the velocity.

ii.
```python
    loader = SessionLoader(one=one, eid=eid)
    loader.load_wheel()
    wheel_times = loader.wheel["times"].to_numpy(dtype=float)
    wheel_speed = np.abs(loader.wheel["velocity"].to_numpy(dtype=float))
```

iii. Not argued in prose, but this is exactly the repository's `load_target_behavior(one, eid, 'wheel-speed')`, which returns `np.abs(sess_loader.wheel['velocity'])` on `sess_loader.wheel['times']`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps. (1) `SessionLoader.load_wheel()` does the IBL-standard work: interpolate the irregularly sampled wheel position to a uniform 1000 Hz grid and differentiate with a 20 Hz Butterworth low-pass to get velocity; the absolute value is the speed in rad/s. (2) Per trial, the trace is resampled onto the trial's 100 bin **end points** by linear `interp1d` with `fill_value="extrapolate"`, restricted to the samples strictly inside the window (`searchsorted` right/left), with the repository's coverage guards (≥ 2 samples, first/last sample within one bin of the window edges, all interpolated values finite). Trials failing any guard are dropped from the session. (3) The whole session's `(n_trials, 100)` speed matrix is discretised into tertiles (see 7-c). No smoothing or normalisation beyond the loader's own filtering.

ii.
```python
    # Repository behavior bins are sampled at each bin's right edge.
    relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
    ...
        ib = np.searchsorted(sample_times, begin, side="right")
        ie = np.searchsorted(sample_times, end, side="left")
        if ie - ib < 2:
            continue
        local_t = sample_times[ib:ie]
        local_v = sample_values[ib:ie]
        if abs(begin - local_t[0]) > BIN_SIZE_S:
            continue
        if abs(end - local_t[-1]) > BIN_SIZE_S:
            continue
        query = stimulus_time + relative_endpoints
        interp = interp1d(local_t, local_v, kind="linear", fill_value="extrapolate")(query)
        if not np.all(np.isfinite(interp)):
            continue
```
```python
    wheel, wheel_good = interpolate_trials(wheel_times, wheel_speed, stimulus_times)
```

iii. The function docstring says the intent explicitly: "Match `get_behavior_per_interval`'s endpoint sampling and coverage test"; and inline, "interp1d with extrapolation is the exact operation used by the repository. The final query is the interval end, typically just beyond the final camera/wheel sample selected with `side='left'`." Trajectory step 120: "The converter continues to reproduce the repository's 20 ms endpoint interpolation for wheel and whisker signals, then applies session-level tertiles only after the final valid-trial mask." This does match the repository, whose `get_behavior_per_interval` uses `x_interp = np.linspace(beg + binsize, end, n_bins)` and the same three skip conditions.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 equal-occupancy classes per session: the 1/3 and 2/3 quantiles are computed over **all** retained trial × time-bin values of that session (pooled, not per trial), and `np.digitize(..., right=False)` maps values to 0 (low), 1 (medium), 2 (high). Thresholds are recorded per session in the metadata. Because thresholds are session-specific, the classes are balanced by construction (observed fractions 0.3333/0.3333/0.3333) and are comparable across sessions with different wheel gains.

ii.
```python
def discretize_tertiles(values):
    """Convert a session's continuous behavior into low/middle/high tertiles."""
    thresholds = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("non-finite tertile thresholds")
    # np.digitize creates exactly the required ordered labels 0, 1, 2.
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, [float(thresholds[0]), float(thresholds[1])]
```
```python
    wheel_class, wheel_thresholds = discretize_tertiles(wheel)
```
metadata: `"dynamic_output_discretization": "within-session tertiles over retained trial-time bins"`.

iii. Trajectory step 40: "Dynamic outputs will be session-wise tertiles, which keeps 'low/medium/high' comparable despite camera-specific motion-energy scales." Step 138: "balanced three-level dynamic targets". The instructions only require "discretized into 3 bins"; equal-occupancy tertiles guarantee all three classes exist in every session, which the decoder's balanced-accuracy scoring needs.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Same alignment event and same window as the neural data: the query times are `stimOn_times + (-0.5 + 0.02·k)` for k = 1…100, i.e. the end of each of the 100 neural bins, so output column *t* corresponds to neural column *t*. The wheel is on the same session clock as the spikes, so no clock correction is needed. (Relative to the neural bin's spike-accumulation interval `[start, end)`, the wheel sample is taken at `end`, and relative to the *Time since stimulus onset* input, which is labelled by the bin start, it is 20 ms later.)

ii.
```python
    relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
    ...
        query = stimulus_time + relative_endpoints
```
```python
        decoder_output.append(
            np.vstack([..., wheel_class[i], motion_class[i]])
        )
```

iii. Same as 7-b: the endpoint grid is chosen to reproduce the repository's `get_behavior_per_interval` exactly ("Repository behavior bins are sampled at each bin's right edge").

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times, loaded through `SessionLoader.load_motion_energy(views=[view])` as the `times` and `whiskerMotionEnergy` columns of `loader.motion_energy['<side>Camera']`. The **left** camera is tried first and the **right** is used as a fallback; a session where neither yields any usable trial is dropped entirely (this accounts for all 15 skipped sessions). The camera actually used is recorded per session as `metadata['session_info'][i]['motion_energy_camera']`.

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
            # The reference uses left whenever it can be loaded, with right as
            # a fallback.  A wholly unusable left trace is treated as failure.
            if np.any(motion_good):
                return wheel, motion, wheel_good & motion_good, view
        except Exception as exc:
            last_error = exc
    raise ValueError(f"no usable whisker motion-energy trace ({last_error})")
```

iii. The inline comment states the rule, which mirrors the repository's `bin_behaviors`: it calls `load_target_behavior(one, eid, 'left-whisker-motion-energy')` and falls back to the right camera on failure. Trajectory step 124: the first skip "occurred… because neither camera provides a usable whisker-motion-energy trace; this is the correct session-level exclusion for a dataset requiring that output"; step 128: "That exclusion pattern is consistent with the methods paper reporting fewer usable multimodal sessions than the 459-session BWM release."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Identical to the wheel: the released ROI motion-energy trace is used as-is (no filtering, normalisation or unit conversion), resampled per trial onto the same 100 bin end points with the same `interp1d`/coverage guards, and then discretised into session tertiles. Trials whose window is not covered by the camera are dropped (combined with the wheel mask via `wheel_good & motion_good`).

ii.
```python
            motion, motion_good = interpolate_trials(
                motion_df["times"].to_numpy(dtype=float),
                motion_df["whiskerMotionEnergy"].to_numpy(dtype=float),
                stimulus_times,
            )
```
```python
    motion_class, motion_thresholds = discretize_tertiles(motion)
```

iii. Same as 7-b: "Match `get_behavior_per_interval`'s endpoint sampling and coverage test." The released trace is already the quantity the papers define (mean absolute inter-frame difference in the whisker-pad box), so no further processing is warranted.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The same `discretize_tertiles` as the wheel: per-session 1/3 and 2/3 quantiles over all retained trial × bin values, `np.digitize` → 0/1/2, labelled `["low", "medium", "high"]`, thresholds stored per session in the metadata. Observed fractions {0: 0.3326, 1: 0.3326, 2: 0.3348}.

ii.
```python
    motion_class, motion_thresholds = discretize_tertiles(motion)
...
        "whisker_motion_energy_tertile_thresholds": motion_thresholds,
```

iii. Trajectory step 40: session-wise tertiles "keeps 'low/medium/high' comparable despite camera-specific motion-energy scales" — relevant here because the left (60 Hz, full-res) and right (150 Hz, half-res) cameras give motion energies on different scales, so a global threshold would not be comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as the wheel: the camera trace is evaluated at `stimOn_times + (-0.5 + 0.02·k)`, k = 1…100 — the end of each neural bin — on the same session clock as the spikes, giving one value per neural bin. Camera frames (60/150 Hz) are sparser than 50 Hz bins only for the left camera at 60 Hz, so the linear interpolation genuinely resamples rather than decimates.

ii.
```python
    relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
    ...
        query = stimulus_time + relative_endpoints
        interp = interp1d(local_t, local_v, kind="linear", fill_value="extrapolate")(query)
```

iii. As in 7-d: the endpoint grid reproduces the repository's behaviour binning; the camera times are already synchronised to the ephys clock by the IBL pipeline, so no further alignment is done.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is dropped, never imputed, and every drop is recorded.
  - Missing trial events (NaN in any of the seven required columns) → trial dropped by the mask.
  - Wheel/camera trace absent or not spanning the trial window, or producing non-finite interpolated values → trial dropped (`behavior_good`).
  - Neither camera usable, fewer than 2 source trials, fewer than 2 trials after the mask, fewer than 2 with behaviour coverage, a probe with no spikes, or a session with no clusters → the whole session is skipped; the exception's repr is appended to `metadata['skipped_sessions']` and printed to stderr. 15 of 459 sessions were skipped, all for unusable whisker motion energy.
  - NaN cluster acronym → relabelled `"void"` and kept.
  - Unexpected categorical values (`choice` ∉ {−1, +1}, `probabilityLeft` ∉ {0.2, 0.5, 0.8}), mismatched cluster indices, or a count exceeding `uint16` raise rather than being silently coerced — note that these raise at session level, so one anomalous trial would discard the whole session (in practice none triggered).
  - Trials with zero spikes are *not* treated as errors; 3 such trials remain and are the validator's only warnings.
  - Multiple ALF revisions are warned about and suppressed, with ONE selecting the newest revision.

ii.
```python
        except Exception as exc:
            skipped_sessions.append({"eid": eid, "reason": repr(exc)})
            print(f"[{number}/{len(eids)}] SKIP {eid}: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
```
```python
        if not np.all(np.isfinite(interp)):
            continue
        result[i] = interp.astype(np.float32)
        good[i] = True
```
```python
        warnings.filterwarnings("ignore", message="Multiple revisions:.*")
```

iii. Trajectory step 63: "sessions lacking a usable required stream will be skipped and documented rather than silently filled"; step 239: excluded "rather than padded or extrapolated across missing data"; step 296 on the zero-spike trials: "these trials are retained because the paper's trial mask does not exclude them, and the reference binning code likewise emits zero counts when no spikes occur."

## 10-a. What are the most time-consuming steps of the code?

i. Measured from the run log (444 sessions, mean 12.4 s, median 11.0 s, max 47.8 s, total ≈ 5,530 s ≈ 92 min for the conversion loop), the cost is dominated by:
  1. `SpikeSortingLoader.load_spike_sorting()` + `merge_clusters()` per probe — reading the hundreds of MB of `spikes.times`/`spikes.clusters` per insertion. Per-session time tracks cluster count (and therefore spike volume) much more than trial count, which the agent noted at step 72.
  2. The per-trial binning loop, whose `np.bincount(minlength=n_clusters*100)` allocates and fills an `n_clusters × 100` grid for every trial — with no QC filter that is up to 3,140 × 100 per trial, ~300× more work per trial than the reference's curated populations.
  3. Serialising the 106 GB pickle, which took roughly half an hour on its own (steps 284–289 track the temp file growing to ~98 GB).
  4. The whole run is **single-process**: sessions are converted in a plain `for` loop, so nothing overlaps I/O with compute.

ii.
```python
    for number, eid in enumerate(eids, start=1):
        ...
        neural, decoder_input, decoder_output, regions, info = make_session(one, rows)
```
```python
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 72: "Processing time depends mainly on spike volume rather than trial count, so some dense two-probe recordings take longer"; step 284: "Serialization is still active; the temporary pickle has reached about 98 GB. That size is expected from dense float32 matrices for every retained trial and all sorted clusters."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three.
  - `bin_probe_spikes`: the `for trial_i, stimulus_time in ...` loop. All trials could be done with one `bincount` by offsetting each spike's flat index by `trial * n_clusters * N_TIME`, since the windows are non-overlapping and the spike array is sorted.
  - `interpolate_trials`: the per-trial `interp1d` loop. A single `np.interp` over a concatenated query vector (or one `searchsorted`-based linear interpolation) would replace it; the per-trial coverage guards could be evaluated as vectorised comparisons on the `searchsorted` results, exactly as the reference's `covered()` does.
  - `trial_numbers_in_block`: a scalar Python loop over every trial of the session that is a textbook `cumcount` — `(p != np.roll(p,1)).cumsum()` plus a grouped `arange`, or `pandas.groupby(...).cumcount()`.
  The first two are the ones that matter in principle, but both are small next to spike I/O; the third is pure Python over ≤ ~1,500 elements and is negligible. The larger available speedup is not vectorisation but parallelism: the sessions are independent and could have been run in a process pool (as the reference does with 10 workers).

ii.
```python
    for trial_i, stimulus_time in enumerate(stimulus_times):
        ...
        hist = np.bincount(flat, minlength=len(cluster_ids) * N_TIME)
```
```python
    for i, stimulus_time in enumerate(stimulus_times):
        ...
        interp = interp1d(local_t, local_v, kind="linear", fill_value="extrapolate")(query)
```
```python
    for i in range(1, len(out)):
        if probability_left[i] == probability_left[i - 1]:
            out[i] = out[i - 1] + 1.0
```

iii. Not discussed in the trajectory; the agent's stated priority throughout was fidelity to the repository's per-interval operations ("interp1d with extrapolation is the exact operation used by the repository"), which the loop form makes easy to verify.

## 10-c. What processing does the code repeat multiple times?

i. Several small repetitions, none dominant:
  - `BrainRegions()` is constructed **inside** `bin_probe_spikes`, i.e. once per probe insertion (~699 times) rather than once per process; each construction re-reads and parses the Allen/Beryl region tables.
  - Two `SessionLoader` objects are built for the same eid — one in `make_session` (trials) and a second in `load_behaviors` (wheel + motion energy) — duplicating the loader's session lookup.
  - `release["eid"].astype(str) == eid` re-casts the whole release column to string on every session iteration (and `_ordered_unique` over the eid column is done twice).
  - `load_motion_energy` is attempted a second time on the right camera whenever the left camera fails, and the wheel interpolation that preceded it is kept, so nothing there is wasted, but the left-camera interpolation over all trials is thrown away.
  - `np.searchsorted` over the full spike-time array is repeated per trial instead of once for all window edges.

ii.
```python
    beryl = BrainRegions().acronym2acronym(regions.to_numpy(), mapping="Beryl")
```
```python
    session_loader = SessionLoader(one=one, eid=eid)     # in make_session
...
    loader = SessionLoader(one=one, eid=eid)             # again in load_behaviors
```
```python
            rows = release[release["eid"].astype(str) == eid]
```

iii. Not discussed. These are all small relative to spike I/O, and the duplicated `SessionLoader` keeps `load_behaviors` self-contained.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dominant item is the absence of neuron curation (see 2-c): 599,865 clusters are binned, stored and written where the data paper's curation would keep ~75,000, including 12,175 units the atlas places outside the brain (`void`) — a ~8× inflation that produces a 106 GB pickle, of which the decoder keeps only the first 100 principal components per session. Smaller items:
  - Static variables are materialised at full temporal resolution: `choice` and `prior` are tiled 100× per trial, and `trial number in block` likewise, although the target format permits per-trial vectors.
  - `TIME_FROM_STIMULUS` is identical for all 188,925 trials yet is stored as a separate row in every trial's input array.
  - Counts are accumulated as `uint16` and then converted to `float32` for a whole session at once (`neural_3d`), doubling the peak memory of the largest sessions, and an `np.bincount`-plus-`reshape` temporary is built per trial.
  - The per-trial `hist.max(initial=0) > uint16 max` overflow check is an extra full pass over each trial's histogram.
  - `merge_clusters` builds the whole cluster dataframe (channels, depths, metrics, uuids) when only the `acronym` column is used.
  - `metadata['session_info']` stores `retained_trial_indices` for every session (≈ 189k integers), which nothing downstream reads.

ii.
```python
            "neuron_filter": "all sorted clusters (no cluster-QC threshold)",
```
```python
        decoder_output.append(
            np.vstack([
                np.full(N_TIME, choice[i], np.int8),
                np.full(N_TIME, prior[i], np.int8),
                wheel_class[i],
                motion_class[i],
            ])
        )
```
```python
        if hist.max(initial=0) > np.iinfo(np.uint16).max:
            raise OverflowError("20 ms spike count exceeds uint16 capacity")
```
```python
        "retained_trial_indices": source_indices.astype(int).tolist(),
```

iii. The tiling is deliberate and follows the instructions ("If at all possible, make it time-varying"); the module docstring says "Static targets are repeated over time so that they can coexist with the two dynamic targets in one output array." The dataset size is acknowledged and defended rather than reduced: "The resulting dataset is intentionally large because the reference method keeps all sorted clusters rather than only well-isolated units" (step 104).
