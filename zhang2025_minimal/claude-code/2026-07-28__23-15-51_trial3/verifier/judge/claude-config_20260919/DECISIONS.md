# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI first tried the ONE API (`ONE(..., mode='local')`, `one.load_cache(tag=...)`, `one.search()`), found that the locally cached dataset registry listed only a subset of the files that are physically present in `/app/data/one_cache`, and abandoned the API entirely. Instead it reads the session/probe manifest `code/code_zhang2025/data/bwm_release.csv` (one row per probe insertion: `eid, lab, subject, date, probe_name`), reconstructs each session directory as `<cache>/<lab>/Subjects/<subject>/<date>/001` (falling back to the first sub-directory of the date folder if `001` does not exist), and then opens every file directly off the filesystem with `Path.rglob`. Each data stream is found by globbing the session tree: `_ibl_trials.table.pqt`, `_ibl_wheel.position.npy` / `.timestamps.npy`, `leftCamera.ROIMotionEnergy.npy` / `_ibl_leftCamera.times.npy` (right camera as fallback), and per probe `alf/<probe>/pykilosort/<revision>/spikes.times.npy`, `spikes.clusters.npy`, `clusters.channels.npy`, `clusters.depths.npy`, `channels.brainLocationIds_ccf_2017.npy`, `clusters.metrics.pqt`. For the spike-sorting directory the *last* revision folder is taken (`sorted(revisions)[-1]`); for every other dataset the *first* `rglob` hit is taken (`trials_files[0]`, `wheel_pos_files[0]`, …), which is an arbitrary order when a session holds two revisions of the same dataset (5 of 461 sessions have two revisions of `_ibl_trials.table.pqt`). Sessions are processed one at a time, in sorted-`eid` order, in a single process, and everything is accumulated in RAM before one final `pickle.dump`. 459 sessions were found, 439 converted, 20 skipped.

ii.
```python
def find_session_paths(cache_dir):
    """Find all session paths in the ONE cache directory."""
    bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)

    for _, row in bwm_df.iterrows():
        ...
        session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
        if not session_path.exists():
            parent = Path(cache_dir) / lab / 'Subjects' / subject / date
            if parent.exists():
                subdirs = sorted(parent.iterdir())
                if subdirs:
                    session_path = subdirs[0]
        if session_path.exists():
            if eid not in sessions:
                sessions[eid] = {'path': session_path, 'probes': [], 'subject': subject, ...}
            sessions[eid]['probes'].append(probe_name)
```

```python
    trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
    trials = pd.read_parquet(trials_files[0])
```

```python
    probe_path = session_path / 'alf' / probe_name / 'pykilosort'
    revisions = [d for d in probe_path.iterdir() if d.is_dir()]
    revision_path = sorted(revisions)[-1]  # Use latest revision
    spike_times = np.load(revision_path / 'spikes.times.npy')
    spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

iii. From the trajectory (steps 47–56): the agent ran `one.search()` under several tags and found "The ONE cache has files on disk but the dataset registry only has a subset. Let me work directly with the files." Its reasoning at step 55: *"The ONE API lists only certain datasets. The local files include trials table, wheel, spikes, clusters, etc. but they're not listed in the dataset registry. Let me work directly with the file system instead."* `bwm_release.csv` was chosen because it is the session/probe manifest shipped with the reference (Zhang 2025) repository, so it defines exactly the same session and probe set the reference analysis used (step 39–40: 459 unique eids, 699 pids, 139 subjects).

## 1-b. How are the data split into subjects (mice)?

i. The subject name is taken from the `subject` column of `bwm_release.csv` (it is also the directory name in the cache path), so no parsing or inference is needed. `subjects` is built as a list in first-encountered order (not sorted), and `subject_idx` is the position of each session's subject in that list. 135 subjects over 439 sessions.

ii.
```python
sessions[eid] = {'path': session_path, 'probes': [], 'subject': subject, 'lab': lab, ...}
```

```python
        subject = result['subject']
        if subject not in subject_set:
            subject_set.append(subject)
        subj_idx = subject_set.index(subject)
        ...
        all_subject_idx.append(subj_idx)
```

```python
        'subjects': subject_set,
        'subject_idx': np.array(all_subject_idx),
```

iii. Not discussed explicitly in the trajectory; the subject identifier is a column of the release manifest, so the agent treated it as given (CONVERSION_NOTES sanity check #8: "Expected 139 subjects (data paper); Observed 135 subjects — 4 fewer due to skipped sessions").

## 1-c. How are the data split into sessions?

i. A session is one `eid` in `bwm_release.csv`. The CSV has one row per probe insertion, so rows are grouped by `eid` into a dict whose value carries the session path and the list of probe names; the resulting dict is iterated in sorted-`eid` order, one entry per session. Sessions are never split or merged further; the two probes of a dual-probe session are pooled into that one session (see 2-b).

ii.
```python
    for i, (eid, session_info) in enumerate(sorted(sessions.items())):
        print(f"\nSession {i+1}/{len(sessions)}: {eid}")
        result = process_session(session_info, cache_dir, br)
```

iii. The agent verified the manifest structure before writing code (step 39/40: "Unique eids: 459 … Sessions with 1 probe / 2 probes"), i.e. the eid *is* the session, so nothing has to be derived.

## 1-d. How are the data split into trials?

i. The trials table `_ibl_trials.table.pqt` has one row per trial, so the split is given by the data. Each retained row becomes one trial, with its window defined as `stimOn_times + (-0.5, +1.5) s`.

ii.
```python
    trials = pd.read_parquet(trials_files[0])
    ...
    valid_trials = trials[mask].copy()
    for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
        stim_on = trial[ALIGN_TIME]
        t_start = stim_on + TIME_WINDOW[0]
        t_end = stim_on + TIME_WINDOW[1]
```

iii. The agent inspected the trials table first (step 57, printing its columns and dtypes) and treated one row = one trial; the reference code's `create_intervals`/`load_trials_and_mask` does the same.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, only one of which is documented.

**(1) The documented trial mask**, copied from the reference code's `load_trials_and_mask()` defaults: drop any trial with a NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; drop trials with reaction time (`firstMovement_times − stimOn_times`) below 0.08 s or above 2.0 s; drop no-response trials (`choice == 0`). A session is skipped when fewer than 2 trials survive.

**(2) An undocumented coverage/NaN filter inside the per-trial loop**: a trial is silently `continue`d if `interpolate_behavior_to_bins` returns `None` for the wheel or the whisker trace — i.e. if fewer than 2 samples of that stream fall in `[t_start − 20 ms, t_end + 20 ms]`, or if the first/last sample inside the window is more than one bin (20 ms) away from the window edge — or if the interpolated trace contains a NaN. Because the AI feeds this check the **raw, movement-triggered** wheel timestamps (see 7-b) rather than IBL's uniformly re-sampled wheel, this second filter is drastic: on a test session it takes the trial count from 408 (after filter 1) to **193**, i.e. every trial in which the mouse held the wheel still near the window edges is discarded. Dataset-wide the AI kept 70,778 trials (mean 161/session) where the expert kept 188,740 (mean ~428/session). The camera coverage check, by contrast, drops essentially nothing (408 → 408 on the same session).

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
...
    mask = pd.Series(True, index=trials.index)
    for event in NAN_EXCLUDE:
        if event in trials.columns:
            mask &= ~trials[event].isna()
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    mask &= (trials['choice'] != 0)
```

```python
        if ws is None or wm is None:
            continue
        # Check for NaNs in behavioral data
        if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
            continue
```

```python
    # inside interpolate_behavior_to_bins
    if len(t_sel) < 2:
        return None
    if np.abs(t_sel[0] - t_start) > binsize or np.abs(t_sel[-1] - t_end) > binsize:
        return None
```

iii. CONVERSION_NOTES §5: *"Exclude trials with NaN in required events …, reaction time < 0.08 s or > 2.0 s, no choice made (choice == 0). Justification: Matches `load_trials_and_mask()` with default parameters. Source: `ibl_data_utils.py` lines 123–213."* The coverage test is a transcription of the reference's `interpolate_behavior` guards ("target data starts too late" / "ends too early", both `> binsize`). The resulting trial loss was noticed but rationalised rather than investigated — CONVERSION_NOTES sanity check #3: *"Observed: Mean 161 valid trials per session … Status: PASS — reduction from ~400+ to ~161 after strict filtering is expected."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's latest `pykilosort` revision. `clusters.channels.npy` (peak channel of each cluster) and `channels.brainLocationIds_ccf_2017.npy` (Allen CCF id of each channel) are loaded only to assign a brain region to each unit; `clusters.metrics.pqt` (the quality `label`) and `clusters.depths.npy` are loaded, carried through `merge_probes`, and then never used.

ii.
```python
        spike_times = np.load(revision_path / 'spikes.times.npy')
        spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
        clusters_channels = np.load(revision_path / 'clusters.channels.npy')
        clusters_depths = np.load(revision_path / 'clusters.depths.npy')
        chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')
        metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
        if metrics_files:
            metrics = pd.read_parquet(metrics_files[0])
            cluster_labels = metrics['label'].values
```

```python
    cluster_acronyms = br.id2acronym(cluster_brain_ids)
    beryl_regions = br.acronym2acronym(cluster_acronyms, mapping='Beryl')
```

iii. The agent enumerated the on-disk ALF objects (steps 29–31, 58) and confirmed the cluster→channel→CCF-id chain reproduces the acronyms that `SpikeSortingLoader.merge_clusters` would attach (step 71). CONVERSION_NOTES §6: Beryl mapping is used "because the reference code uses `brainreg.acronym2acronym(neural_dict['cluster_regions'], mapping='Beryl')`".

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the 2 s trial window, giving a `(n_clusters, 100)` `float32` matrix per trial. **The counts are left as counts — they are not divided by the bin width**, so the stored values are spikes/20 ms rather than Hz. No smoothing, no normalisation, no z-scoring. All probes of a session are merged into a single population: cluster ids of the second probe are offset by the first probe's cluster count, channel ids likewise, and the concatenated spike arrays are re-sorted by time. Clusters that emit no spike in the whole session are dropped by `np.unique`, and the surviving cluster ids are renumbered 0…n-1 with a Python dict lookup applied to every spike.

ii.
```python
    for i, (times, clusters) in enumerate(spikes_list):
        n_clusters = len(clusters_channels_list[i])
        merged_clusters.append(clusters + cluster_offset)
        merged_channels.append(clusters_channels_list[i] + channel_offset)
        cluster_offset += n_clusters
        channel_offset += n_channels
    sort_idx = np.argsort(all_times, kind='stable')
```

```python
    cluster_ids = np.unique(spike_clusters)
    n_clusters = len(cluster_ids)
    cluster_id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
    spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
```

```python
def bin_spikes_trial(spike_times, spike_clusters, n_clusters, t_start, t_end, binsize, n_bins):
    i_start = np.searchsorted(spike_times, t_start, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    ...
    bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    valid = clusters_sel < n_clusters
    np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)
```

iii. CONVERSION_NOTES §4: *"Merge all probes within a session into a single neural population. Justification: the reference code explicitly merges probes … because probes in the same session share the same behavioral state and are not independent. Source: `ibl_data_utils.py merge_probes()`."* The binning parameters are taken from `0_data_caching.py` (`binsize: 0.02`). No justification is given anywhere for leaving the data as counts rather than rates; the format spec allows "firing rates, spike counts".

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control is applied at all.** Every cluster that fired at least one spike in the session is kept, regardless of its IBL `label` (0, 1/3, 2/3 or 1) and regardless of whether the histology placed it outside the brain. The `label` column is actually loaded from `clusters.metrics.pqt` and merged across probes, but never used. The consequences: mean 1,358 units/session, ~596,000 units total (expert: 164/session, 72,417 total), and 12,630 units assigned the Beryl acronym `void`, i.e. sites the atlas places outside the brain, plus 85,264 `root` units. The only unit-level removal is the implicit `np.unique(spike_clusters)` drop of silent clusters. The module docstring additionally claims a session-level criterion — "Minimum 5 good neurons per session (from data paper inclusion criteria)" — that does not exist anywhere in the code.

ii.
```python
"""
- Load ALL clusters (not filtered by quality) per the reference caching code
...
- Minimum 5 good neurons per session (from data paper inclusion criteria)
"""
```

```python
        metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
        cluster_labels = None
        if metrics_files:
            metrics = pd.read_parquet(metrics_files[0])
            cluster_labels = metrics['label'].values
        return spike_times, spike_clusters, clusters_channels, chan_brain_ids, cluster_labels
```

```python
    # Get unique cluster IDs
    cluster_ids = np.unique(spike_clusters)
    n_clusters = len(cluster_ids)
```
(`all_labels` is returned by `merge_probes` and then never referenced again.)

iii. The agent explicitly measured the cost of the alternative (step 59: "Total clusters: 898; Clusters with label >= 1: 76") and then decided against it. Reasoning at step 83: *"The reference code loads ALL clusters (not just good ones) … `prepare_data` calls `load_spiking_data(one, pid)` without `qc` parameter (defaults to None) … The cluster quality filtering happens at the analysis level, not the caching level. Since we're building a decoder dataset, loading all clusters is correct — the decoder can learn from all available neural data."* It noted the tension with the data paper ("Out of the 621,733 units collected, 75,708 were considered well-isolated neurons") and chose the code over the paper. CONVERSION_NOTES §3 repeats this, citing `ibl_data_utils.py` line 736.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to visual stimulus onset, `stimOn_times`, on the shared IBL session clock (spikes, trial events, wheel and camera times are all already synchronised upstream), so no cross-stream re-alignment is performed. For each trial the absolute window `[stimOn − 0.5 s, stimOn + 1.5 s]` is cut out of the time-sorted spike train with `searchsorted`, and the bin index is computed as `floor((t − t_start)/binsize)`, i.e. relative to the trial's own onset. The alignment event is recorded in the metadata.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
        stim_on = trial[ALIGN_TIME]
        t_start = stim_on + TIME_WINDOW[0]
        t_end = stim_on + TIME_WINDOW[1]
        neural = bin_spikes_trial(spike_times, spike_clusters_remapped, n_clusters,
                                  t_start, t_end, BINSIZE, N_BINS)
```

```python
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
```

iii. CONVERSION_NOTES §1: *"Align all trials to stimulus onset (`stimOn_times`). Justification: the decoder task specifies 'Temporally align based on stimulus onset'. The reference code (`0_data_caching.py`) also uses `align_time: 'stimOn_times'`. Window: −0.5 s to +1.5 s; Source: reference code params `'time_window': (-.5, 1.5)`."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial, identical for every trial and every session. No rebinning, re-sampling or smoothing of the neural data is applied — spikes are histogrammed once directly onto the 20 ms grid. `time_bin_size` is written to the metadata in ms (20.0). The behavioural streams are the only ones re-sampled (linear interpolation onto the same 100-point grid, see 7-d/8-d).

ii.
```python
BINSIZE = 0.02  # 20ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0s
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
...
            'time_bin_size': BINSIZE * 1000,  # in ms
            'n_time_bins': N_BINS,
```

iii. CONVERSION_NOTES §2: *"20 ms non-overlapping bins → 100 time bins per trial. Justification: reference code uses `binsize: 0.02` (20 ms). The methods paper mentions both 50 ms (for choice/prior) and 20 ms (for wheel/whisker ME) but the caching code uses 20 ms uniformly."* Step 67 shows the agent weighing the paper's 50 ms statement against the code's 20 ms and choosing the code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` (the alignment event) together with the fixed window and bin size — nothing else is read. Because the grid is identical for every trial, the same 100-element vector is regenerated inside the trial loop and stored for each trial as row 0 of `input`. Values run from −0.48 s to +1.50 s: `np.linspace(t_start + binsize, t_end, n_bins)`, i.e. the **right edge** of each 20 ms bin (the code's comment calls these "bin centers", which they are not; the expert used true centres, −0.49 to 1.49).

ii.
```python
        time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
        input_data = np.stack([
            time_since_stim,
            np.full(N_BINS, trial_num, dtype=np.float32)
        ], axis=0)
```

iii. The grid is copied verbatim from the reference code's behaviour-interpolation grid, `x_interp = np.linspace(interval_begs[i] + binsize, interval_ends[i], n_bins)` (`ibl_data_utils.py:614`), which the agent read at step 17 and re-used so that inputs, outputs and neural bins share one axis (CONVERSION_NOTES §9).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None — it is a deterministic ramp defined by the window and bin size, cast to `float32`, and is the same for all trials and sessions. It is kept continuous rather than converted to a binary event marker.

ii.
```python
        time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The decoder task specifies "Time since stimulus onset, continuous, time-varying", so the agent stored it as a continuous ramp (CONVERSION_NOTES: decoder inputs section of the module docstring).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the neural binning grid itself. Bin *i* of the neural matrix covers `[stimOn − 0.5 + 0.02·i, stimOn − 0.5 + 0.02·(i+1))`, and element *i* of the time input is `−0.5 + 0.02·(i+1)`, the right edge of that same bin. So the two are aligned bin-for-bin, with a constant +10 ms offset relative to the bin centre. The same grid is used for the wheel and whisker traces, so all four streams share one time axis.

ii.
```python
    bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)     # neural grid
```
```python
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)         # behaviour grid
```
```python
        time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)  # input grid
```

iii. Same as 3-a: the agent adopted the reference code's `x_interp` definition for every non-neural stream so that "behavioral signals match neural bin centers" (CONVERSION_NOTES §9).

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The trials table carries no block index, so block boundaries are recovered as the points where `probabilityLeft` changes value.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    trials_masked = trials_df[mask].copy()
    prob_left = trials_masked['probabilityLeft'].values
    block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
```

iii. Not discussed in the trajectory or the notes beyond the decoder-task requirement ("Trial number in block, continuous, per-trial"); `probabilityLeft` is the only block-related column in the table (the agent printed the trials columns at step 57).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The counter is computed **after** the trial mask has been applied: `compute_trial_number_in_block` first subsets the trials table with the quality mask and only then detects block changes and assigns `0, 1, 2, …` within each run of constant `probabilityLeft`. The number therefore counts *surviving* trials in the block, not the animal's true position in the block, and it is further compressed by the wheel-coverage filter of 1-e (which removes ~half the remaining trials but is applied later, so those trials still advance the counter). The resulting range is 0–84, versus 0–98 for the expert, who deliberately counted before filtering. The scalar is broadcast across all 100 bins as row 1 of `input`.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    """Compute trial number within each block (0-indexed)."""
    trials_masked = trials_df[mask].copy()
    prob_left = trials_masked['probabilityLeft'].values
    block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
    trial_in_block = np.zeros(len(prob_left), dtype=int)
    for i in range(len(block_changes)):
        start = block_changes[i]
        end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
        trial_in_block[start:end] = np.arange(end - start)
    return trial_in_block
```

```python
        trial_num = np.float32(trial_in_block[trial_idx])
        input_data = np.stack([time_since_stim, np.full(N_BINS, trial_num, dtype=np.float32)], axis=0)
```

iii. No justification is given for computing the counter on the masked rather than the full trials table; neither CONVERSION_NOTES nor the trajectory mentions the choice.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward turn), −1 (rightward) or 0 (no response). No-response trials have already been removed by the mask (`choice != 0`), so only ±1 reaches the encoder, which maps +1 → 0 (left) and everything else → 1 (right).

ii.
```python
        choice_val = 0 if trial['choice'] == 1 else 1  # IBL: 1=left, -1=right -> map: left=0, right=1
```

iii. CONVERSION_NOTES §11: *"IBL choice convention (1=left, −1=right) mapped to decoder convention (left=0, right=1). Justification: decoder task specifies 'left = 0, right = 1'."*

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the recoding. The per-trial scalar is broadcast across all 100 time bins and stored as row 0 of the `(4, 100)` `int64` output array, and `output_values[0] = ['left', 'right']`.

ii.
```python
        output = np.stack([
            np.full(N_BINS, choice_val, dtype=np.int64),
            np.full(N_BINS, prior_val, dtype=np.int64),
            wheel_disc[i].astype(np.int64),
            whisker_disc[i].astype(np.int64),
        ], axis=0)
```

iii. The format spec asks for time-varying outputs "if at all possible"; the agent broadcast the per-trial constants to the time axis so that all four outputs share the same `(d_output, T)` shape (module docstring, decoder-task section).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table (the block prior held constant within a block), recoded 0.2 → 0, 0.5 → 1, 0.8 → 2. Any other value silently falls back to 1; in practice this never fires, because NaNs are removed by the mask and the IBL task only ever uses those three values. Observed distribution 0.409 / 0.167 / 0.423 (expert: 0.418 / 0.141 / 0.442).

ii.
```python
        prob_left = trial['probabilityLeft']
        if prob_left == 0.2:
            prior_val = 0
        elif prob_left == 0.5:
            prior_val = 1
        elif prob_left == 0.8:
            prior_val = 2
        else:
            prior_val = 1  # fallback
```

iii. CONVERSION_NOTES §12: *"Map probabilityLeft values: 0.2→0, 0.5→1, 0.8→2. Justification: decoder task specifies '0.2 -> 0, 0.5 -> 1, 0.8 -> 2'."* Sanity check #6 confirms the expected shape of the distribution ("First 90 trials at 0.5, remaining in alternating 0.2/0.8 blocks").

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding; the per-trial integer is broadcast over the 100 bins as row 1 of `output`, with `output_values[1] = ['0.2', '0.5', '0.8']`.

ii.
```python
            np.full(N_BINS, prior_val, dtype=np.int64),
```
```python
        'output_values': [
            ['left', 'right'],
            ['0.2', '0.5', '0.8'],
            ['low', 'medium', 'high'],
            ['low', 'medium', 'high'],
        ],
```

iii. As 6-a — the mapping is dictated by the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw ALF wheel arrays `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, read straight off disk. Speed is defined as the absolute value of the velocity derived from those two arrays.

ii.
```python
def load_wheel_speed(session_path):
    wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
    wheel_ts_files = list(session_path.rglob('_ibl_wheel.timestamps.npy'))
    wheel_pos = np.load(wheel_pos_files[0])
    wheel_ts = np.load(wheel_ts_files[0])
```

iii. CONVERSION_NOTES §7: *"Compute wheel speed as absolute value of wheel velocity … Justification: reference code loads wheel speed via `load_target_behavior(one, eid, 'wheel-speed')` which uses `np.abs(sess_loader.wheel['velocity'])`."*

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Velocity is computed as a plain first difference of the raw encoder samples, `dp/dt`, timestamped at the midpoints, and the speed is its absolute value. **The two steps `SessionLoader.load_wheel()` performs are omitted**: the position is *not* interpolated onto a uniform 1 kHz grid, and no 20 Hz Butterworth low-pass is applied before differentiating. The resulting trace therefore lives on the raw, movement-triggered sample times, and is noisier and differently distributed than the reference trace. This has a large downstream consequence: because the raw wheel stream only emits samples when the wheel moves (on the sessions checked, ~70 % of the session's wall-clock time sits inside gaps longer than one 20 ms bin, with individual gaps up to 79 s), the coverage test in `interpolate_behavior_to_bins` rejects any trial whose window edges fall in a quiet period. On a test session this cut the trials from 408 to 193, and dataset-wide it is the main reason the AI's dataset holds 70,778 trials where the expert's holds 188,740 — and the loss is not random, it preferentially removes low-movement trials. The surviving trace is linearly interpolated onto the trial's 100-point grid and then discretized (7-c).

ii.
```python
    # Compute velocity using Gaussian-smoothed derivative (matching brainbox)
    # Simple finite difference for velocity, then take absolute value for speed
    dt = np.diff(wheel_ts)
    dp = np.diff(wheel_pos)
    vel = dp / dt
    speed = np.abs(vel)
    vel_ts = wheel_ts[:-1] + dt / 2
    return vel_ts, speed
```

```python
    mask = (beh_times >= t_start - margin) & (beh_times <= t_end + margin)
    t_sel = beh_times[mask]; v_sel = beh_values[mask]
    if len(t_sel) < 2:
        return None
    if np.abs(t_sel[0] - t_start) > binsize or np.abs(t_sel[-1] - t_end) > binsize:
        return None
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)
```

iii. The agent was aware of the deviation and accepted it. Step 62 comment: *"The reference code uses `SessionLoader.load_wheel()` which computes velocity … The brainbox SessionLoader interpolates and uses Gaussian smoothing. But for our purposes, a simple diff/dt should work."* CONVERSION_NOTES, "Known Limitations" §1: *"The reference code uses `SessionLoader.load_wheel()` which applies Gaussian smoothing. Our implementation uses simple finite differences, which is noisier but preserves the same information."* The consequent trial loss was not connected to this choice.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three equal-count classes per session. All non-NaN interpolated speed values of all retained trials of that session are pooled, the 0 / 33.3 / 66.7 / 100 percentiles are taken, the outer edges are replaced by ±inf, and each trial's trace is digitised against the two interior edges to give 0/1/2 = low/medium/high. Edges are therefore session-specific; they are computed once per session (not per trial) and returned in the per-session result dict, but never written into the saved data.

ii.
```python
def discretize_to_bins(values_list, n_bins, compute_edges=True, edges=None):
    if compute_edges:
        all_vals = np.concatenate([v[~np.isnan(v)] for v in values_list if v is not None])
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(all_vals, quantiles)
        edges[0] = -np.inf
        edges[-1] = np.inf
    for v in values_list:
        d = np.digitize(v, edges[1:-1])  # Returns 0 to n_bins-1
```

```python
    wheel_disc, wheel_edges = discretize_to_bins(wheel_speed_raw, N_WHEEL_BINS)
```

iii. CONVERSION_NOTES §10: *"Discretize wheel speed and whisker ME into 3 equal-count bins using quantiles. Justification: the decoder task specifies 'Wheel speed discretized into 3 bins' … Using quantile-based discretization ensures approximately equal class frequencies. Quantile method: computed per-session to maintain local context."* Known Limitations §3 flags that the edges vary across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. By linear interpolation onto exactly the same 100-point grid used for the neural bins and for the time input: `np.linspace(stimOn − 0.5 + 0.02, stimOn + 1.5, 100)`, evaluated on the absolute session clock (wheel timestamps and spike times share the IBL sync clock, so no further alignment is needed). Only samples within one bin of the window are used for the interpolation, and `fill_value='extrapolate'` covers the edges. Trials where the stream does not span the window are dropped rather than extrapolated (see 7-b).

ii.
```python
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
    ...
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)
```

```python
            ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
```

iii. CONVERSION_NOTES §9: *"Interpolate wheel speed and whisker ME to neural bin centers using linear interpolation. Justification: reference code uses `interp1d(..., kind='linear', fill_value='extrapolate')` in `get_behavior_per_interval()`. Source: `ibl_data_utils.py` lines 624–625."*

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with its frame times `_ibl_leftCamera.times.npy`; if the left camera is absent (or its lengths do not match), `rightCamera.ROIMotionEnergy.npy` with `_ibl_rightCamera.times.npy` is used instead. This is the same array `SessionLoader.load_motion_energy` exposes as `whiskerMotionEnergy` for the side views. A one-sample length mismatch is tolerated by truncating the timestamps.

ii.
```python
def load_whisker_motion_energy(session_path):
    """Load whisker motion energy (prefer left camera at 60Hz)."""
    left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
    left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
    if left_me_files and left_times_files:
        me = np.load(left_me_files[0]); times = np.load(left_times_files[0])
        if len(me) == len(times):
            return times, me
        elif len(me) == len(times) - 1:
            return times[:-1], me
    # Try right camera
    ...
```

iii. CONVERSION_NOTES §8: *"Use left camera (60 Hz) preferentially, fall back to right camera (150 Hz). Justification: the reference code tries left camera first, then right. Left camera has full resolution (1280×1024) at 60 Hz per the data paper."* The agent verified lengths and frame rates on a real session first (steps 59–60: left 307,375 frames at 60.1 Hz, right 769,676 at 150.4 Hz, both exactly matching their ME arrays).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None — the released per-frame ROI motion-energy values are used as they are, with no filtering, normalisation or baseline subtraction. They are linearly interpolated onto the trial's 100-point grid (same function as the wheel) and then discretized. Trials where the camera does not span the window, or where the interpolated trace contains a NaN, are dropped; in practice the camera stream is continuous, so this rejects almost nothing.

ii.
```python
            wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
        ...
        if ws is None or wm is None:
            continue
        if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
            continue
```

iii. CONVERSION_NOTES §9 (as 7-d) plus Known Limitations §2: *"ROIMotionEnergy timestamps are aligned to camera frame times. For the left camera (60 Hz), the temporal resolution is ~16.7 ms, which is close to the 20 ms bin size."*

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: three equal-count classes from the 33.3rd and 66.7th percentiles of that session's own pooled interpolated values, digitised to 0/1/2 = low/medium/high. Observed fractions 0.333 / 0.333 / 0.334.

ii.
```python
    whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```
```python
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(all_vals, quantiles)
        edges[0] = -np.inf; edges[-1] = np.inf
        ...
        d = np.digitize(v, edges[1:-1])
```

iii. Same as 7-c (CONVERSION_NOTES §10): the decoder task demands 3 bins, and quantiles give balanced classes; computed per session "to maintain local context", with the cross-session comparability caveat noted in Known Limitations §3.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Through the same `interpolate_behavior_to_bins` call on the same `np.linspace(t_start + 0.02, t_end, 100)` grid, evaluated on the absolute session clock. Camera frame times are already synchronised to the ephys clock by the IBL pipeline, so subtracting nothing and simply evaluating at the trial grid is sufficient; the trace ends up bin-for-bin aligned with the neural matrix and with the wheel trace.

ii.
```python
            wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
```
```python
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)
```

iii. As 7-d/8-b — the agent checked that `ROIMotionEnergy` is one value per camera frame and that the frame times are on the session clock (step 60), so interpolation onto the trial grid is the only alignment required.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything is handled by dropping, with a few silent fallbacks:
- **Session level**: no trials table → skip; fewer than 2 trials passing the mask → skip; no probe with loadable spikes → skip; no wheel data → skip; any exception while loading a probe is caught, printed, and that probe is skipped (the session continues with the remaining probes). A session that ends with fewer than 2 fully usable trials is skipped. 20 of 459 sessions were skipped this way.
- **Trial level**: missing/NaN trial events, out-of-range RT, no-response (1-e); no wheel or camera coverage of the window; NaNs anywhere in the interpolated behavioural traces.
- **Silent fallbacks**: an unexpected `probabilityLeft` is coded as 1 (the 0.5 class) rather than raising; a cluster whose id or peak channel falls outside the channel table is given brain id 0 (commented "root", which `id2acronym` actually resolves to `void`); a one-sample length mismatch between motion energy and camera times is fixed by truncation.
- Not handled: trials whose neural window is empty are kept — the format checker reported 4 trials in 2 sessions where the whole `neural` matrix is zero.

ii.
```python
    if trials is None:
        print(f"    Skipping: no trials data"); return None
    n_valid = mask.sum()
    if n_valid < 2:
        print(f"    Skipping: only {n_valid} valid trials"); return None
    ...
    if not spikes_list:
        print(f"    Skipping: no spike data"); return None
    if wheel_speed is None:
        print(f"    Skipping: no wheel data"); return None
    ...
    if len(neural_trials) < 2:
        print(f"    Skipping: only {len(neural_trials)} valid trials after behavioral filtering"); return None
```

```python
    except Exception as e:
        print(f"  Error loading spikes for {probe_name}: {e}")
        return None, None, None, None, None
```

```python
        else:
            prior_val = 1  # fallback
```

iii. CONVERSION_NOTES sanity check #1: *"459 sessions found, 439 processed, 20 skipped (missing behavioral data)"*, and the trajectory at step 140 records the acceptance of the zero-neural warnings: *"Verification passed with only minor warnings (4 trials with all-zero neural data in 2 sessions)."* The general philosophy is stated in the notes as dropping anything that cannot be fully populated rather than imputing.

## 10-a. What are the most time-consuming steps of the code?

i. The run is single-process and took roughly 70–80 minutes of wall clock for 439 sessions (steps 88–129 of the trajectory show the agent polling the process for over an hour). The dominant costs are:
1. **Reading the spike sorting off disk** — `spikes.times.npy` + `spikes.clusters.npy` are hundreds of MB per probe and are read in full for every session (two probes for many sessions).
2. **The per-spike Python remap loop** `[cluster_id_to_idx[c] for c in spike_clusters]` — a dict lookup executed once per spike, i.e. tens of millions of Python-level iterations per session.
3. **The per-trial loop**: for each trial, `bin_spikes_trial` allocates and fills an `(n_clusters, 100)` matrix (with ~1,358 clusters this is the bulk of the 38.6 GB of output), and `interpolate_behavior_to_bins` builds a boolean mask over the *entire* session wheel/camera array (0.3–1.2 M samples) twice per trial.
4. **Accumulating everything in RAM and the final `pickle.dump`** of a 38.6 GB object; the agent watched RSS grow past 11 GB mid-run.
The agent did optimise one hot spot mid-way: `bin_spikes_trial` was rewritten to use `searchsorted` + `np.add.at` after the first sample run (step 85/86).

ii.
```python
    spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
```
```python
        spike_times = np.load(revision_path / 'spikes.times.npy')
        spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```
```python
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
```

iii. Step 85: *"I realize the conversion is slow because of the per-spike loop in `bin_spikes_trial`. Let me optimize it."* After that the agent stopped profiling and simply waited for the background job, checking `/proc/<pid>/status` every few minutes (steps 108–129); no further performance decision was taken and no parallelism was introduced.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four places:
1. `spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])` — a pure-Python loop over every spike; `np.searchsorted(cluster_ids, spike_clusters)` or `np.unique(..., return_inverse=True)` does the same in one vectorised call and is the single biggest easy win.
2. The per-trial loop in `process_session`: both the spike binning and the two interpolations could be done for all trials at once (one `bincount` with a trial-offset flat index; one `np.interp` over a concatenated query vector), as the expert notes for the equivalent loops.
3. The trial-window selection inside `interpolate_behavior_to_bins` uses a full-array boolean mask instead of two `searchsorted` calls, so its cost is O(n_trials × n_samples) rather than O(n_trials × log n).
4. The per-cluster region lookup (`for cid in cluster_ids: ... all_brain_ids[ch]`), and the assembly-time `if r not in all_regions_flat` / `all_regions_flat.index(r)` linear scans, which are O(n_neurons × n_regions) ≈ 1.7 × 10^8 comparisons over ~596 k units — a dict would make both O(1).

ii.
```python
    cluster_id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
    spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
```
```python
    mask = (beh_times >= t_start - margin) & (beh_times <= t_end + margin)
```
```python
    for regions in all_beryl_regions_per_session:
        for r in regions:
            if r not in all_regions_flat:
                all_regions_flat.append(r)
    ...
        idx = np.array([all_regions_flat.index(r) for r in regions])
```

iii. The agent vectorised only `bin_spikes_trial` (step 86, switching to `searchsorted` + `np.add.at`) and did not revisit the others; no justification is recorded for leaving them as Python loops.

## 10-c. What processing does the code repeat multiple times?

i.
- `rglob` walks of the whole session directory tree are repeated for every dataset of every session: once in `load_trials`, twice in `load_wheel_speed`, and two to four times in `load_whisker_motion_energy`.
- The full-session boolean mask over the wheel and camera time arrays is rebuilt once per trial per stream (2 × n_trials full scans of arrays with up to 1.2 M elements), and a fresh `interp1d` object is constructed for each of those calls.
- `time_since_stim` is recomputed identically inside every trial iteration and then stored separately for each of the 70,778 trials, although it is one and the same 100-element vector everywhere.
- `np.full(N_BINS, ...)` broadcasts of the per-trial constants (choice, prior, trial-in-block) are materialised per trial.
- `subject_set.index(subject)` re-scans the subject list for every session and `all_regions_flat.index(r)` re-scans the region list for every neuron.

ii.
```python
    wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
    wheel_ts_files = list(session_path.rglob('_ibl_wheel.timestamps.npy'))
```
```python
    for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
        ...
        ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
        wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
        time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. No discussion in the trajectory; the repeated work is a by-product of structuring the per-trial loop around self-contained helper functions that each take the full-session arrays.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Spike binning for trials that are then thrown away.** `bin_spikes_trial` is called *before* the wheel/whisker coverage test, so the `(n_clusters, 100)` matrix is built and discarded for the >50 % of masked trials that fail that test.
- **Loading quality metrics and depths that are never used.** `clusters.metrics.pqt` is read, `label` extracted, merged across probes into `all_labels` — and never referenced again; `clusters.depths.npy` is loaded and dropped immediately.
- **Carrying ~596,000 unfiltered units**, of which only ~12 % would pass the IBL `label >= 1` criterion and 12,630 are `void` (outside the brain) — this is what inflates the pickle to 38.6 GB (the expert's dataset has 72,417 units) and dominates every downstream read.
- **Storage that could be constant or narrower**: the identical 100-element time ramp is stored once per trial; choice, prior and trial-in-block are broadcast to 100 bins each; outputs that only ever take values 0–2 are stored as `int64` (8× larger than `int8`).
- `wheel_edges` / `whisker_edges` are computed and returned in the per-session dict but never written into the saved data.
- `print_stats` re-walks every trial of every session at the end purely to print summaries, and a separate `sample_data.pkl` (360 MB) is produced and kept.

ii.
```python
        neural = bin_spikes_trial(spike_times, spike_clusters_remapped, n_clusters,
                                  t_start, t_end, BINSIZE, N_BINS)
        ...
        if ws is None or wm is None:
            continue                      # the binned matrix just computed is discarded
```
```python
    all_labels = np.concatenate(merged_labels) if merged_labels else None
    return all_times, all_clusters, all_channels, all_brain_ids, all_labels   # all_labels unused
```
```python
        output = np.stack([
            np.full(N_BINS, choice_val, dtype=np.int64),
            np.full(N_BINS, prior_val, dtype=np.int64),
            ...
```
```python
        'wheel_edges': wheel_edges,
        'whisker_edges': whisker_edges,   # never propagated into `data`
```

iii. The agent never addressed dataset size; it only checked that the machine had enough RAM (step 116: *"1 TB of memory — plenty"*) and reported the final size as a fact (*"The conversion is complete! The file is 38.6 GB"*). The decision to keep all clusters (2-c) is the stated reason the data is this large.
