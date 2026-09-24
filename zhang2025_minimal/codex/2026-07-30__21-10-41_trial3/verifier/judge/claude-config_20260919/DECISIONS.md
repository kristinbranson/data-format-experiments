# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The session/probe inventory is taken from the BWM release freeze table shipped with the methods-paper repository, `code/code_zhang2025/data/bwm_release.csv` (699 insertions, 459 sessions, 139 subjects), which the AI verified against the session counts quoted in the data paper. The table is grouped by `eid`, so one group = one session with all of its probe insertions. Data themselves are read through a single `ONE` client pointed at the local cache `/app/data` but constructed in online mode (`base_url`, `password`), so any ALF file that is not already cached is downloaded from the IBL S3 mirror. Nothing is opened by raw path. Each stream is requested as a named object/dataset for the session: `one.load_object(eid, "trials", collection="alf")`, `one.load_object(eid, "wheel", ...)`, `one.load_object(eid, "leftCamera", attribute=["times","ROIMotionEnergy"], ...)`, and per probe `one.load_dataset(eid, "<name>", collection=f"alf/{probe}/pykilosort", revision="2024-05-06")` for `clusters.metrics.pqt`, `clusters.channels.npy`, `channels.brainLocationIds_ccf_2017.npy`, `spikes.times.npy`, `spikes.clusters.npy`. Region names come from the Alyx REST endpoint `brain-regions/list`. Sessions are processed strictly serially in a single process.

ii.
```python
release_df = pd.read_csv(args.release_csv)           # code_zhang2025/data/bwm_release.csv
...
one = ONE(base_url="https://openalyx.internationalbrainlab.org",
          password="international", cache_dir=args.cache_dir, silent=True)
id_to_acronym, grey_ids = build_region_maps(one)
grouped = list(release_df.groupby("eid", sort=False))
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

```python
def load_trials(one: ONE, eid: str) -> pd.DataFrame:
    trials = one.load_object(eid, "trials", collection="alf").to_df()
```

```python
    spikes_times_path = one.load_dataset(
        eid, "spikes.times.npy", collection=probe.collection,
        revision=SPIKE_SORTING_REVISION, download_only=True)
```

iii. The AI explicitly went looking for the loading path used by the methods-paper code and mirrored it: `0_data_caching.py` reads the same `data/bwm_release.csv` freeze and builds the identical `ONE(base_url=..., password='international', silent=True, cache_dir=...)` client (trajectory step 11). It confirmed that "the release metadata under `2025_Q3_IBL_et_al_BWM` lines up with the paper's 459-session public release, which is likely the correct freeze to target" (step 23) and that "ONE can resolve sessions locally and fetch only missing ALF files, which gives us a clean route for a reproducible converter script" (step 43). It deliberately avoided `SpikeSortingLoader`/`load_spike_sorting` after one attempt "started pulling unnecessary large spike-side arrays", switching "to per-dataset loads so the converter only requests `spikes.times`, `spikes.clusters`, `clusters.metrics`, and channel metadata" (step 59). The pinned revision `2024-05-06` was added after the first single-session test failed on ambiguous revisions (step 117).

## 1-b. How are the data split into subjects?

i. No parsing or inference: the `subject` column of the release table is carried through with each session. Subjects are indexed in order of first appearance while sessions are assembled, and the index list is re-derived (and compacted, dropping subjects that lost all their sessions) after the final region filter. Result: 133 subjects over 433 sessions.

ii.
```python
subject = str(session_rows["subject"].iloc[0])
...
for sess in session_results:
    if sess["subject"] not in subject_to_idx:
        subject_to_idx[sess["subject"]] = len(subjects)
        subjects.append(sess["subject"])
    subject_idx.append(subject_to_idx[sess["subject"]])
```

```python
used_subjects = sorted(set(int(x) for x in used_subject_idx.tolist()))
subject_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_subjects)}
new_subject_names = [data["subjects"][old_idx] for old_idx in used_subjects]
```

iii. The release freeze already carries a unique subject name per session, so the AI treated this as metadata bookkeeping rather than a decision; it recorded `subject` (with `lab` and `date`) in `metadata['session_info']` for traceability and reported the subject count as a check against the paper's 139-mouse release (CONVERSION_NOTES.md, "Session / probe source").

## 1-c. How are the data split into sessions?

i. A session is the `eid` group of the release table; `release_df.groupby("eid")` yields one group per session, whose rows are that session's probe insertions. All probes of a session are merged into one population (see 2-b), so sessions, not insertions, are the unit of the output. 459 release sessions are iterated; 433 survive to the final dataset.

ii.
```python
grouped = list(release_df.groupby("eid", sort=False))
print(f"Release probes: {len(release_df)}")      # 699
print(f"Release sessions: {len(grouped)}")       # 459
...
eid = str(session_rows["eid"].iloc[0])
for _, row in session_rows.sort_values("probe_name").iterrows():
    probe = load_probe_info(one, eid, str(row["probe_name"]), id_to_acronym, grey_ids)
```

iii. The AI followed `prepare_data` in `ibl_data_utils.py`, which states "when merging probes we are interested in eids, not pids" and merges all insertions of a session because probes from one session share the same behaviour and are not statistically independent. The AI's notes record "Probe data are merged within session" as a reference-matching decision.

## 1-d. How are the data split into trials?

i. The ALF trials table has one row per trial, so no splitting is required; the row index is the trial index. Each surviving row defines a window `[stimOn_times - 0.5, stimOn_times + 1.5]` s that all streams are cut to.

ii.
```python
trials = load_trials(one, eid)
base_mask = make_base_trial_mask(trials)
for trial_idx in np.flatnonzero(base_mask):
    stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
    start_time = stim_on + WINDOW_START
    end_time = stim_on + WINDOW_END
```

iii. No justification was needed or given beyond using the released trials table as-is; this matches `load_trials_and_mask`, which also returns the full trials table plus a boolean mask.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied, reproducing `load_trials_and_mask(one, eid, max_trial_len=10.0)` from `prepare_data`, plus a behavioural-coverage test:
1. no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType` (the function's `nan_exclude='default'` list);
2. reaction time `firstMovement_times - stimOn_times` within `[0.08, 2.0]` s;
3. `choice != 0` (no-response trials dropped);
4. trial duration `feedback_times - goCue_times <= 10.0` s.
On top of that, a trial is kept only if both the wheel and the left-camera traces actually span its 2 s window (`interpolate_trial_signal` returns `None` otherwise), and only if `probabilityLeft` is exactly one of 0.2/0.5/0.8 and `choice` is ±1. Finally, trials in which every retained neuron is silent across the whole window are dropped, and a session is discarded if fewer than 2 trials survive. 184,664 trials survive over 433 sessions.

ii.
```python
def make_base_trial_mask(trials):
    mask = np.ones(len(trials), dtype=bool)
    for col in TRIAL_NAN_EXCLUDE:
        mask &= trials[col].notna().to_numpy()
    reaction_time = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    mask &= reaction_time >= 0.08
    mask &= reaction_time <= 2.0
    mask &= trials["choice"].to_numpy() != 0
    if "goCue_times" in trials.columns:
        ...
        trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
        mask &= trial_len_ok
    return mask
```

```python
    wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
    whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
    if wheel_interp is None or whisker_interp is None:
        continue
```

```python
    nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
    if nonzero_trial_mask.sum() < 2:
        print(f"Skip session {eid}: fewer than 2 nonzero neural trials after binning.")
        return None
```

iii. The AI read `load_trials_and_mask` and `prepare_data` directly (trajectory steps 13–19) and transcribed the defaults, including `max_trial_len=10.0`, which only appears at the `prepare_data` call site. CONVERSION_NOTES.md states the mask "[m]atched the trial mask used in the reference code and the exclusions described in the papers". The coverage test copies the reference utility's skip conditions verbatim (`|interval_beg - target_time[0]| > binsize` → "target data starts too late", `|interval_end - target_time[-1]| > binsize` → "target data ends too early"). Dropping all-zero neural trials was a reaction to a verifier warning: "The sample verification surfaced one concrete data-quality warning: a trial whose selected neurons are all silent across the full 2 s window. I'm filtering those zero-information trials out" (step 151).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The binned array itself comes from two per-probe arrays, `spikes.times.npy` and `spikes.clusters.npy`. Three further per-probe datasets are used only to decide which clusters to keep and how to label them anatomically: `clusters.metrics.pqt` (the `label` QC score), `clusters.channels.npy` (peak channel per cluster) and `channels.brainLocationIds_ccf_2017.npy` (CCF region id per channel), with ids resolved to acronyms through the Alyx `brain-regions` list and then remapped to Beryl.

ii.
```python
    spike_times = np.load(spikes_times_path, mmap_mode="r")
    spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

```python
    cluster_channels = np.asarray(one.load_dataset(eid, "clusters.channels.npy", ...))
    channel_region_ids = np.asarray(one.load_dataset(eid, "channels.brainLocationIds_ccf_2017.npy", ...))
    region_ids[valid_channel_idx] = channel_region_ids[cluster_channels[valid_channel_idx]]
    region_acronyms = np.array([id_to_acronym.get(int(rid), "void") for rid in region_ids], dtype=object)
    good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
```

iii. The AI chose the minimal dataset set deliberately after a broad loader call began downloading large unused spike attributes (step 59): "I'm ... switching to per-dataset loads so the converter only requests `spikes.times`, `spikes.clusters`, `clusters.metrics`, and channel metadata." It checked the cluster metadata to confirm the QC field used by the reference: "The probe metadata confirms the QC labels used by the reference code: it is filtering 'good' units by `clusters.metrics.label >= 1`" (step 72). It initially avoided an atlas dependency and derived acronyms from CCF ids via the REST endpoint (steps 76–80), installing `iblatlas` only later for the Beryl remap (step 526).

## 2-b. How is the `neural` data processed?

i. Spikes of retained clusters are counted into 100 non-overlapping 20 ms bins spanning `[stimOn - 0.5, stimOn + 1.5]` s. No smoothing, no z-scoring, and **no conversion to firing rate** — the stored values are raw spike counts per bin, cast to `float16`. All probes of a session are concatenated along the neuron axis into one population (probe order = sorted `probe_name`), with the region list extended in the same order so `brain_region_idx` stays aligned. Counts are accumulated per trial with `np.add.at` into a `uint16` array.

ii.
```python
    for trial_idx, start_time in enumerate(trial_starts):
        end_time = start_time + (WINDOW_END - WINDOW_START)
        start_idx = np.searchsorted(kept_times, start_time, side="left")
        end_idx = np.searchsorted(kept_times, end_time, side="left")
        trial_times = kept_times[start_idx:end_idx]
        trial_clusters = kept_clusters[start_idx:end_idx]
        bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
        in_bounds = (bins >= 0) & (bins < NBINS) & (trial_clusters >= 0)
        np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
```

```python
    offset = 0
    for probe in probes:
        probe_counts = bin_probe_spikes(one, eid, probe, trial_starts)
        end_offset = offset + probe_counts.shape[1]
        session_counts[:, offset:end_offset, :] = probe_counts
        session_regions.extend(probe.cluster_regions.tolist())
        offset = end_offset
    ...
    neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
```

iii. CONVERSION_NOTES.md: "Probe data are merged within session … Spike counts are binned into non-overlapping 20 ms bins." This mirrors the reference pipeline, where `merge_probes` pools insertions within a session and `bin_spiking_data`/`bincount2D` returns spike **counts** per bin (never a rate). `float16` was a memory decision: the AI checked "the decoder's memory profile against likely dataset sizes … some of them would produce a pickle that is impractical to train on" (step 99) before committing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Four cuts, applied in two stages.
Per probe, at load time: (1) `clusters.metrics.label >= 1`, the IBL "well-isolated"/stringent criterion; (2) the cluster's CCF region must be a descendant of `grey` (id 8), i.e. grey matter only, which removes fibre tracts, ventricles and out-of-brain sites; (3) the acronym must not be `void`, `root` or `grey`.
After the whole dataset is built, in `apply_region_filters`: acronyms are remapped to the **Beryl** atlas, `root`/`void` are dropped again, and a neuron is kept only if its Beryl region has ≥5 neurons **in that session** and that region appears in ≥2 sessions dataset-wide; a session with no surviving neuron is dropped.
Net effect on the full run: 437 → 433 sessions, 472 → 208 regions, and 59,751 retained neurons (mean 138/session, min 5, max 508) versus ~72,400 for the expert solution.

ii.
```python
    good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
    good &= np.isin(region_ids, list(grey_ids))
    good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))   # {"void","root","grey"}
    good_cluster_ids = np.flatnonzero(good).astype(np.int32)
```

```python
        session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
        invalid = np.isin(session_beryl, ["root", "void"])
        unique_names, counts = np.unique(valid_names, return_counts=True)
        valid_regions = set(unique_names[counts >= MIN_NEURONS_PER_SESSION_REGION].tolist())   # >= 5
        keep_mask = np.isin(session_beryl, list(valid_regions)) & ~invalid
...
    globally_valid_regions = {r for r, s in sessions_with_region.items() if len(s) >= MIN_SESSIONS_PER_REGION}  # >= 2
```

iii. The `label >= 1` cut was verified against the reference code (step 72) and against the data paper's "well-isolated units" curation; the AI also argued it on feasibility grounds — "I've narrowed the likely feasible export to well-isolated units rather than every pykilosort cluster" (step 103). The grey-matter/`void`/`root`/`grey` cut was in the first version of the script with no recorded argument beyond excluding non-brain sites. The region-count rules were a late retrofit, motivated by a mismatch between the AI's region count and the methods paper's: "the converted brain-region count is 472 versus the 270-region figure in the methods paper" (step 504) → "The reference code maps cluster acronyms to Beryl regions, and the papers add two more region curation rules: keep only regions with at least 5 well-isolated neurons within a session and present in at least 2 sessions. That explains the inflated 472-region count" (step 508). It was applied by reprocessing the finished pickle rather than re-running the crawl (step 512). The AI documented that the result (208 regions) still misses the paper's 270 and flagged this as "the main remaining reference discrepancy".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is by subtraction on the shared session clock: all IBL streams (spike times, trial event times, wheel timestamps, camera frame times) are already expressed in seconds on one synchronised timebase, so a trial's window is `[stimOn_times - 0.5, stimOn_times + 1.5]` and a spike's bin is `floor((t - (stimOn - 0.5)) / 0.02)`. No resampling or cross-clock correction. `temporal_alignment_event` is recorded as `"stimulus onset"` with `off_start = -0.5`, `off_end = 1.5`. One implementation caveat: spike times are cast to `float32` before the subtraction, which quantises absolute times to ~0.5–1 ms late in a long recording — well inside a 20 ms bin, but a small avoidable jitter relative to the `float64` arithmetic the reference uses.

ii.
```python
    trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
```

```python
    kept_times = np.asarray(spike_times[spike_keep], dtype=np.float32)
    ...
        bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The window, bin size and alignment event were taken from the reference `params` dict in `0_data_caching.py` (`'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)`), which the AI read at step 11. Its notes record: "Common alignment event: stimulus onset. Common window: `[-0.5, 1.5]` s … This yields `100` bins per trial, consistent with the methods-paper summary that uses 2 s trials with 20 ms bins." The AI had earlier weighed using different windows per target, as some paper analyses do, and rejected it because the decoder format needs "one common stimulus-aligned representation across static and dynamic variables" (step 36).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, identical for every trial and session; `metadata['time_bin_size'] = 20.0` ms. Spikes are binned once at 20 ms directly from spike times — there is no intermediate resolution and therefore no rebinning. The behavioural traces are resampled (not rebinned) onto the same 20 ms grid by linear interpolation.

ii.
```python
WINDOW_START = -0.5
WINDOW_END = 1.5
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))   # 100
```

```python
        "time_bin_size": float(BINSIZE * 1000.0),
```

iii. Directly from the reference configuration (`params = {'interval_len': 2, 'binsize': 0.02, ...}`) and the methods paper's 2 s / 20 ms / T = 100 description; the AI validated that "all sessions have a common time dimension of 100 bins" as a sanity check (CONVERSION_NOTES.md).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` of the trials table, which defines the window origin, combined with the fixed bin grid. The values are the **right edge** of each 20 ms bin relative to onset, i.e. `-0.48, -0.46, …, 1.48, 1.50`, identical for every trial, broadcast as a `(100,)` row of the `(2, 100)` input array. It is stored as a continuous ramp rather than a binary onset marker, which the Decoder Task explicitly asks for ("Time since stimulus onset, continuous, time-varying").

ii.
```python
    relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

```python
        input_trial = np.vstack([relative_time,
                                 np.full(NBINS, trial_num, dtype=np.float32)]).astype(np.float32, copy=False)
```

iii. The right-edge convention is copied from the reference utility `get_behavior_per_interval`, which builds its grid as `x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)`. Using the same grid for the time input and for the two continuous behavioural outputs keeps every time-varying channel on one axis. The AI's notes describe this as "linear interpolation to stimulus-aligned 20 ms bin right edges".

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid: the vector is defined by the window and bin size, not measured, so it is computed once per trial as a constant `arange`. No per-trial variation exists.

ii.
```python
    relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. N/A — no decision beyond the grid convention described in 3-a. (The vector is rebuilt inside `process_session` once per session, then copied into every trial's input array.)

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the label of the neural bin grid itself. Neural bin *k* covers `[stimOn - 0.5 + 0.02k, stimOn - 0.5 + 0.02(k+1))`, and `relative_time[k] = -0.5 + 0.02(k+1)` is the closing edge of that same bin, so column *k* of `input` and column *k* of `neural` describe the same 20 ms interval. No offset or interpolation is involved.

ii.
```python
        bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)     # neural bin index
```
```python
    relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)   # same bins, right edge
```

iii. N/A — the same grid constants (`WINDOW_START`, `BINSIZE`, `NBINS`) are used for both, and the verifier confirmed a common `T = 100` and input range `[-0.48, 1.5]` across all 433 sessions.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` only. The trials table carries no block id, so a block boundary is inferred wherever `probabilityLeft` changes value from one row to the next (or becomes non-finite).

ii.
```python
    block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))
```
```python
        if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
            current += 1
        else:
            current = 1
```

iii. Implicit in the code rather than argued in the trajectory: `probabilityLeft` is constant within a block by task design, so it is the only available block signal. The AI verified the resulting range in the verifier output (`trial_number_in_block: [1.0, 99.0]`), consistent with IBL blocks of 20–100 trials.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 1-based running counter over the **unfiltered** trials table: it is computed from the full `trials` DataFrame before any quality mask is applied, so a trial that is later discarded still advances the count and the value reported for a kept trial is the animal's true position in the block. The scalar is broadcast to all 100 bins so the input array is rectangular (`(2, 100)`), and it is kept as a raw count (no normalisation), matching "Trial number in block, continuous, per-trial".

ii.
```python
    trials = load_trials(one, eid)
    base_mask = make_base_trial_mask(trials)
    block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))
    ...
        trial_num = float(block_trial_num[trial_idx])
        input_trial = np.vstack([relative_time, np.full(NBINS, trial_num, dtype=np.float32)])
```

iii. Not argued explicitly in the trajectory; the ordering in the code (counter built from the full table, indexed afterwards by the surviving `trial_idx`) shows the intent to preserve the real block position. The observed range `[1, 99]` was used as the sanity check.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward wheel turn), −1 (rightward) or 0 (no response). Trials with 0 are already removed by the trial mask, and any other value causes the trial to be skipped.

ii.
```python
        choice_val = float(trials.iloc[trial_idx]["choice"])
        if choice_val == 1:
            choice_out = 0
        elif choice_val == -1:
            choice_out = 1
        else:
            continue
```

iii. The AI did not take the sign convention on trust; it checked it empirically against behaviour: "The `choice` convention is resolved from easy-trial behavior: `trials.choice == 1` means left, and `== -1` means right" (step 95), and recorded the check in CONVERSION_NOTES.md ("high-contrast left trials are mostly `choice == 1`; high-contrast right trials are mostly `choice == -1`").

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding +1 → 0 (left), −1 → 1 (right) required by the Decoder Task. The per-trial scalar is then broadcast across all 100 bins as an `int8` row so the output array is time-varying in shape, and the value names `['left', 'right']` are stored in `output_values[0]`. Resulting balance: 50.9 % left / 49.1 % right.

ii.
```python
            output_trial = np.vstack([
                np.full(NBINS, choice_out, dtype=np.int8),
                np.full(NBINS, prior_out, dtype=np.int8),
                discretize(wheel_vals, wheel_edges),
                discretize(whisker_vals, whisker_edges)])
```

iii. The instruction "Choice, binary, per-trial, left = 0, right = 1" fixes the mapping; the broadcast follows the format requirement "If at all possible, make it time-varying".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes exactly the three block values 0.2, 0.5 and 0.8. Any other value (including NaN, already excluded by the mask) causes the trial to be skipped.

ii.
```python
        prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
        if prob_left == 0.2:   prior_out = 0
        elif prob_left == 0.5: prior_out = 1
        elif prob_left == 0.8: prior_out = 2
        else:                  continue
```

iii. Directly from the Decoder Task specification ("0.2 -> 0, 0.5 -> 1, 0.8 -> 2"); the value is the block prior that the IBL task holds constant within a block.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the 0.2/0.5/0.8 → 0/1/2 recoding, broadcast across the 100 bins as `int8`, with `output_values[1] = ['0.2','0.5','0.8']`. No smoothing, no subjective/Bayesian prior estimate is substituted. Resulting distribution: 41.8 % / 14.0 % / 44.2 %, i.e. the unbiased 0.5 block at the start of each session is retained rather than excluded.

ii.
```python
                np.full(NBINS, prior_out, dtype=np.int8),
```
```python
        "output_values": [["left", "right"], ["0.2", "0.5", "0.8"], ["low","medium","high"], ["low","medium","high"]],
```

iii. Keeping the unbiased block matches the reference default (`exclude_unbiased=False` in `load_trials_and_mask`, and `prepare_data` does not override it), and the instruction requires all three levels to exist as categories.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw wheel encoder stream of the session: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, loaded as the ALF object `wheel`. Speed is derived from these two arrays only (see 7-b); the released `wheel.velocity` is not used, it is recomputed with the ibllib routines.

ii.
```python
def load_wheel_speed(one: ONE, eid: str):
    wheel = one.load_object(eid, "wheel", collection="alf")
    timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
    position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. "The reusable upstream wheel functions import cleanly once I add the bundled `ibllib` tree to `PYTHONPATH`, so I can mirror the wheel preprocessing instead of approximating it" (step 67). CONVERSION_NOTES.md records the source datasets explicitly and notes the velocity is computed "with the same Butterworth-filtered method used by `brainbox.behavior.wheel.velocity_filtered`".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) The irregularly sampled encoder position is interpolated onto an even 1000 Hz grid with `interpolate_position`, then differentiated into a velocity with `velocity_filtered(fs=1000, corner_frequency=20, order=8)` — the same 20 Hz low-pass Butterworth configuration `SessionLoader.load_wheel` uses by default — and speed is `abs(velocity)` in rad/s. (2) Per trial, the 1000 Hz trace is sliced to the window and linearly interpolated onto the 100 bin right edges, with the reference's two coverage guards (trace must start no more than one bin after the window start and end no more than one bin before the window end, else the trial is dropped). (3) The continuous value is then discretised (7-c).

ii.
```python
    interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
    speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

```python
def interpolate_trial_signal(signal_times, signal_values, start_time, end_time, binsize=BINSIZE):
    start_idx = np.searchsorted(signal_times, start_time, side="right")
    end_idx = np.searchsorted(signal_times, end_time, side="left")
    ...
    if abs(start_time - times[0]) > binsize:  return None
    if abs(end_time - times[-1]) > binsize:   return None
    sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
    interp_vals = np.interp(sample_times, times, values)
```

iii. The AI imported the actual ibllib functions rather than reimplementing them, precisely so the trace could not drift from the reference (step 67; CONVERSION_NOTES.md "Behavior traces / Wheel"). The slicing, the two `> binsize` guards and the right-edge sample grid are a line-for-line transcription of `get_behavior_per_interval` in `ibl_data_utils.py`, which the AI had read at step 19.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 classes by **global tertiles**: after every session is converted, all wheel-speed samples from every trial of every session are pooled, the 1/3 and 2/3 quantiles of that pooled distribution are taken as two fixed edges (0.01515 and 0.40699 rad/s for the full run), and every sample everywhere is assigned `low`/`medium`/`high` against those same two edges via `np.digitize`. The edges are stored in `metadata['behavior_processing']['wheel_speed_bin_edges']`. By construction the classes are exactly balanced dataset-wide (33.3 %/33.3 %/33.4 %) but not within a session — in the 8-session sample the per-session class fractions range from 0.22 to 0.47.

ii.
```python
    wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
    wheel_edges = tuple(np.quantile(wheel_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
```
```python
def discretize(values, edges):
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge], dtype=np.float32), right=False).astype(np.int8)
```

iii. CONVERSION_NOTES.md: "`wheel_speed_bin` and `whisker_motion_energy_bin`: discretized into 3 bins using global tertiles over the converted dataset; bin names are `low`, `medium`, `high`." The instructions only say "discretized into 3 bins", so the tertile rule is the AI's own choice; no explicit argument for *global* rather than per-session tertiles appears anywhere in the trajectory or notes. The AI's stated check was only that the global fractions come out at 1/3 each and that the decoder beats chance.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The speed trace is evaluated at the right edge of each of the 100 neural bins of that trial, measured from the same `stimOn_times`, so output column *k* and neural column *k* refer to the same 20 ms interval. Alignment is pure interpolation on the shared session clock; no lag or shift is introduced, and trials whose wheel trace does not cover the window are removed from all streams together.

ii.
```python
        sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
        interp_vals = np.interp(sample_times, times, values)
```
```python
        wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
        whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
        if wheel_interp is None or whisker_interp is None:
            continue
```

iii. Wheel timestamps are on the same synchronised session clock as the spikes, so interpolation onto the shared grid is all that is required; the grid is the reference utility's `linspace(beg + binsize, end, n_bins)`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with its frame times `_ibl_leftCamera.times.npy`, loaded as one ALF object. This is the IBL-released motion energy computed over a whisker-pad ROI, used as released. Only the **left** camera is accepted: if the left-camera trace is missing, malformed or length-mismatched with its timestamps, the whole session is skipped — there is no fall-back to the right camera. 22 sessions were dropped for this reason in the full run; `camera_view_counts` is `{'left': 433}`.

ii.
```python
def load_whisker_motion_energy(one, eid):
    try:
        cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
        times = np.asarray(cam["times"], dtype=np.float64)
        values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
        if len(times) and len(times) == len(values):
            return "left", times, values
    except Exception:
        pass
    return None, None, None
```
```python
    if whisker_times is None or whisker_me is None:
        print(f"Skip session {eid}: no whisker motion energy trace available.")
        return None
```

iii. The AI first implemented the reference behaviour (left, else right — `bin_behaviors` falls back to `right-whisker-motion-energy` when the left trace is missing) and ran most of a full conversion that way, then deliberately reversed it mid-task: "I'm making one substantive correction …: switching whisker motion energy to left-camera only instead of left-then-right fallback. The paper text's `60 Hz` whisker signal description points to the left camera specifically, and that choice is much more likely to recover the paper's `433-session` cohort than salvaging right-camera-only sessions" (step 244). It then restarted the full 3-hour conversion and tracked the cohort size against the target: "the corrected cohort is converging … 163 kept sessions out of 171 processed, which projects to the low-430s rather than the high-450s" (step 291).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the trace itself — no filtering, smoothing, normalisation or per-session standardisation. The released per-frame value (60 Hz left camera) is sliced to the trial window and linearly interpolated onto the same 100 bin right edges as everything else, using the same `interpolate_trial_signal` with the same two coverage guards; the trial is dropped if the camera does not span the window or if any interpolated value is non-finite. Discretisation (8-c) is the only transformation.

ii.
```python
        whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```
```python
    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]; values = values[finite]
    if len(times) < 2: return None
    ...
    if not np.all(np.isfinite(interp_vals)): return None
```

iii. CONVERSION_NOTES.md: "Whisker motion energy: use left camera only; loaded from `ROIMotionEnergy` and camera `times`", and "Continuous traces are linearly interpolated onto the right edge of each 20 ms trial bin, following the reference utility logic." The reference pipeline likewise consumes `ROIMotionEnergy` unmodified.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly the same rule as wheel speed: two **global** edges at the 1/3 and 2/3 quantiles of all whisker samples pooled over all trials and all sessions (2.7466 and 7.8613 arbitrary motion-energy units for the full run), applied to every session with `np.digitize`. Globally the three classes are balanced (33.4 %/33.3 %/33.3 %); within a session they are often extremely skewed, because motion energy is in uncalibrated, camera- and session-dependent units — in the 8-session sample, one session is 64.7 % `high` while two others contain essentially no `high` samples at all (0.3 % and 0.5 %).

ii.
```python
    whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
    whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
```
```python
                discretize(whisker_vals, whisker_edges),
```

iii. Same single line of justification as the wheel: "discretized into 3 bins using global tertiles over the converted dataset". The only recorded validation was that the global fractions are 1/3 each and that "sample decoder training beats chance for all requested outputs"; no per-session check of the class distribution was made, and the trajectory contains no discussion of the fact that motion energy is not comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identically to the wheel: the camera trace is interpolated onto the right edge of each of the 100 neural bins of the trial, measured from the same `stimOn_times` on the same session clock, so it is bin-for-bin simultaneous with the neural array. A trial is kept only if the camera covers the whole window to within one bin at each edge, and that mask is applied to the neural, input and all output channels together.

ii.
```python
        sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
        interp_vals = np.interp(sample_times, times, values)
```
```python
        if wheel_interp is None or whisker_interp is None:
            continue
```

iii. Camera frame times are synchronised to the ephys clock upstream by IBL, so no correction is needed; the coverage guards are the reference utility's "target data starts too late" / "ends too early" conditions.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Uniformly by dropping the affected unit of data, at four levels, always with a printed reason.
- **Trial**: NaN in any of the six required trial events (mask); wheel or camera trace not covering the window, fewer than 2 usable samples in the window, or non-finite interpolated values; a `probabilityLeft`/`choice` value outside the expected set; all retained neurons silent for the whole trial.
- **Probe**: an insertion with no cluster surviving QC returns `None` and is skipped; clusters whose peak channel index is out of range get region id −1 → acronym `void` → dropped.
- **Session**: wheel load failure, no left-camera motion energy, fewer than 2 surviving trials, no good grey-matter unit, no neuron surviving the Beryl region filter, or *any* unhandled exception anywhere in `process_session`.
- **Dataset**: `RuntimeError` if no session at all converts.
Missing files are not treated as errors at all in the normal case — ONE downloads them from S3 on demand.

ii.
```python
    try:
        wheel_times, wheel_speed = load_wheel_speed(one, eid)
    except Exception as exc:
        print(f"Skip session {eid}: wheel load failed: {exc}")
        return None
```
```python
        try:
            result = process_session(one, session_rows, id_to_acronym, grey_ids)
        except Exception as exc:
            print(f"Skip session {eid}: unexpected error: {exc}")
            result = None
```
```python
    if len(kept_trial_indices) < 2:
        print(f"Skip session {eid}: only {len(kept_trial_indices)} trials after alignment and behavior checks.")
        return None
    ...
    if not probes:
        print(f"Skip session {eid}: no good grey-matter units after QC.")
        return None
```

iii. The skip messages were designed to be auditable: "I'm … keep[ing] the per-session processing explicit so I can audit any session that drops out" (step 109), and the AI later mined the log for skip reasons to explain the gap between the 459-session release and the converted cohort (steps 380, 417, 504: "22 sessions were skipped during raw conversion because no left-camera whisker motion energy trace was available"). Dropping all-zero neural trials was added in response to a verifier warning (step 151).

## 10-a. What are the most time-consuming steps of the code?

i. Three, in order.
1. **Getting the spike sorting off S3/disk.** `spikes.times.npy` and `spikes.clusters.npy` are ~100 MB each per insertion and 699 insertions are read; the conversion log is dominated by S3 progress bars. The AI's own estimate was "a couple of hours … consistent with a first-time scan over a partially cached BWM release" (step 188), and the full run took roughly 3 hours.
2. **Serial session processing.** Sessions are processed one at a time in a single process; there is no multiprocessing anywhere, so download latency and CPU work never overlap (the expert solution runs 10 worker processes).
3. **Per-trial spike binning with `np.add.at`.** `np.add.at` is numpy's slow unbuffered scatter-add path, called once per trial per probe (~185k trials × ~1.6 probes); `np.bincount` on a flattened `unit * NBINS + bin` index, as the expert uses, is roughly an order of magnitude faster.
Secondary costs: `np.quantile` over ~1.8×10⁹ pooled behaviour samples, the whole-dataset copy in `apply_region_filters`, and writing/reading the 5.4 GB pickle (which had to be read back and rewritten for the region-filter retrofit).

ii.
```python
    spike_times = np.load(spikes_times_path, mmap_mode="r")
    spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
    ...
    spike_keep = keep_bool[spike_clusters]
    kept_times = np.asarray(spike_times[spike_keep], dtype=np.float32)
```
```python
    for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
        result = process_session(one, session_rows, id_to_acronym, grey_ids)
```
```python
        np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
```

iii. The AI monitored throughput and memory throughout and consciously decided not to optimise: "The rate is reasonable enough that I'm continuing with the current implementation rather than rewriting for more aggressive I/O parallelism" (step 193). It did make one targeted fix while a run was in flight — "I was deriving the cluster-map size from the full spike-cluster array, which forces an unnecessary scan. I'm switching that to the much smaller cluster-metadata length" (step 137) — and avoided a second full crawl for the region fix by adding a `--reprocess-existing-pkl` path (step 508).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four.
1. `bin_probe_spikes`'s `for trial_idx, start_time in enumerate(trial_starts)` — could be one `np.bincount` over a flat `(trial, unit, bin)` index for all trials at once; even keeping the loop, `np.add.at` should be `np.bincount`.
2. `process_session`'s `for trial_idx in np.flatnonzero(base_mask)` — the two `interpolate_trial_signal` calls per trial rebuild `searchsorted` bounds and call `np.interp` 2 × 185k times; all query points could be built as one vector and interpolated in a single call per session.
3. Inside that loop, `trials.iloc[trial_idx]["stimOn_times"]`, `["choice"]` and `["probabilityLeft"]` are three pandas scalar lookups per trial (a row `.iloc` plus a label lookup each), which is far slower than pre-extracting the three columns to numpy once.
4. `compute_trial_number_in_block`'s explicit Python `for idx in range(1, len(prob_left))` — the expert does the same thing in two vectorised pandas operations (`(p != p.shift()).cumsum()` + `groupby(...).cumcount()`).
Minor: `np.isin(region_ids, list(grey_ids))` rebuilds a ~1300-element list from a set on every probe, and the region-index assignment in `build_data_from_session_results`/`apply_region_filters` loops over neurons in Python instead of using `np.unique(..., return_inverse=True)`.

ii.
```python
    for trial_idx, start_time in enumerate(trial_starts):
        ...
        np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
```
```python
    for trial_idx in np.flatnonzero(base_mask):
        stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
        ...
        choice_val = float(trials.iloc[trial_idx]["choice"])
        prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
```
```python
    for idx in range(1, len(prob_left)):
        prev = prob_left[idx - 1]; curr = prob_left[idx]
        if np.isfinite(prev) and np.isfinite(curr) and curr == prev: current += 1
        else: current = 1
```

iii. No justification is recorded for keeping these loops; the AI judged overall throughput acceptable and chose not to restructure (step 193). The loops do keep the per-trial slicing explicit and auditable, which is consistent with its stated goal of being able to "audit any session that drops out" (step 109).

## 10-c. What processing does the code repeat multiple times?

i.
1. **Region labelling is done twice.** Acronyms are first derived per cluster from CCF ids via the Alyx `brain-regions` list during conversion, then the whole dataset is re-labelled through `BrainRegions().acronym2acronym(..., 'Beryl')` in `apply_region_filters`, and the `brain_regions` list plus every `brain_region_idx` is rebuilt from scratch a second time.
2. **The neural arrays are rebuilt twice.** `process_session` materialises one `float16` array per trial from `session_counts`, and `apply_region_filters` immediately re-slices and copies every one of those ~185k arrays again (`trial[final_keep].astype(np.float16)`), i.e. a second full pass over the entire multi-GB dataset.
3. **Per-probe re-derivation of constant objects**: `list(grey_ids)` and `list(INVALID_REGION_ACRONYMS)` are rebuilt for every probe; `relative_time` is rebuilt per session and then copied into every trial's input array (185k copies of an identical 100-value vector).
4. **Repeated pandas row lookups** for the same trial row (3 separate `.iloc` calls, see 10-b).
5. `np.isin(session_beryl, ...)` is evaluated twice per session in `apply_region_filters` (once for `keep_mask`, once for `final_keep`).

ii.
```python
    region_acronyms = np.array([id_to_acronym.get(int(rid), "void") for rid in region_ids], dtype=object)
    ...
        session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
```
```python
    neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
    ...
        filtered_trials = [trial[final_keep].astype(np.float16, copy=False) for trial in data["neural"][session_idx]]
```

iii. The duplication is a direct consequence of how the region fix was applied: rather than re-running the 3-hour crawl, the AI bolted the Beryl remap and count filters on as a post-processing pass over the finished dataset — "I'll ... reprocess the existing full pickle in-place instead of spending another three hours on a raw rerun" (step 508), exposed as `--reprocess-existing-pkl`. That was a sound time trade-off for the session, but it left the two-stage structure permanently in the script.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
1. **Spikes are binned and stored for ~12,700 neurons that are then thrown away.** Every unit passing per-probe QC is binned across all trials, materialised as `float16`, and only afterwards deleted by the Beryl `≥5 neurons/session-region` and `≥2 sessions/region` rules. That is roughly 18 % of the binning work, the peak memory and the intermediate pickle wasted.
2. **Whole sessions are fully converted and then dropped** — 4 sessions lost all their neurons at the region-filter stage after all their spikes, wheel and camera traces had already been processed; the global tertile edges are likewise computed from data including those sessions.
3. **Wheel velocity is computed for the entire session at 1000 Hz** (`interpolate_position` + `velocity_filtered` over the full recording) when only ~2 s per kept trial is ever sampled, and `velocity_filtered`'s acceleration output is computed and discarded.
4. **Metadata that no downstream step consumes**: the full `clusters.metrics.pqt` parquet is read for one column; `intervals_0`/`intervals_1` are reconstructed in `load_trials` and never used; `n_trials_total`, `n_trials_base_valid`, `lab`, `date`, `probe_names`, `region_filter_summary` and the duplicated `neuron_filters` entries are written into `metadata` but unused by the decoder; `--seed`/`np.random.seed` has no effect because nothing in the converter is stochastic.
5. **`relative_time` is stored 185k times** as row 0 of every input array although it is identical for every trial in the dataset — about 74 MB of pure redundancy (unavoidable given the required rectangular format, but worth noting).
6. Process-level: the first full conversion (437 sessions, ~3 h) was discarded entirely when the camera rule changed.

ii.
```python
    session_counts = np.zeros((len(kept_trial_indices), total_neurons, NBINS), dtype=np.uint16)
    ...
    neural_trials = [session_counts[idx].astype(np.float16, copy=True) ...]
    # ... later, after the whole dataset exists:
    data, filter_stats = apply_region_filters(data)     # deletes ~12.7k of 72.4k neurons
```
```python
    interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)   # acceleration discarded
```
```python
    if "intervals_0" not in trials.columns and "intervals" in trials.columns:
        trials["intervals_0"] = intervals[:, 0]; trials["intervals_1"] = intervals[:, 1]   # never read
```

iii. Again a consequence of the retrofit (step 508): the region rules were discovered after the crawl had finished, so they could only be applied downstream of binning. The AI accepted the waste knowingly, having checked that memory was not a constraint — "The memory ceiling is not a concern here; the machine reports hundreds of GiB available" (step 199) — and preferred it to a second three-hour crawl.
