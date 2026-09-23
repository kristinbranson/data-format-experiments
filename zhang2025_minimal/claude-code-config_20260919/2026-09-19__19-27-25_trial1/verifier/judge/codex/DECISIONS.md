# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use `one.search(...)` to enumerate sessions. Instead it read `/app/code/code_zhang2025/data/bwm_release.csv`, grouped rows by `eid`, and used that as the session/probe inventory. To make ONE able to load the on-disk files, it first rebuilt a patched local ONE tables directory by merging the shipped cache tables and repointing each dataset row to the newest revision actually present on disk. It then used `SessionLoader` for trials/wheel/motion-energy and `SpikeSortingLoader` for spikes/clusters.

ii.
```python
BWM_RELEASE = REPO / 'data' / 'bwm_release.csv'

def get_one(force_rebuild=False):
    tables_dir = build_patched_tables(force=force_rebuild)
    return ONE(..., cache_dir=str(CACHE_DIR), tables_dir=str(tables_dir), mode='local')
```

```python
bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
for eid, grp in bwm_df.groupby('eid', sort=False):
    jobs.append((eid, list(grp.pid), list(grp.probe_name),
                 grp.subject.iloc[0], grp.lab.iloc[0]))
```

```python
sess_loader = SessionLoader(one=one, eid=eid)
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
```

iii. The trajectory explicitly says the shipped ONE cache tables were stale relative to the file tree, so local ONE would silently fail to find trials/wheel/camera files for most sessions. The agent therefore justified the patched-table approach as necessary to make the release loadable while preserving released `eid`/dataset identities.

## 1-b. How are the data split into subjects?

i. Subject identity comes directly from `bwm_release.csv`. Each session job carries one `subject`, and during assembly the code builds `subjects` in first-seen order and records `subject_idx` per session.

ii.
```python
jobs.append((eid, list(grp.pid), list(grp.probe_name),
             grp.subject.iloc[0], grp.lab.iloc[0]))
```

```python
if res['subject'] not in subjects:
    subjects.append(res['subject'])
subject_idx.append(subjects.index(res['subject']))
```

iii. The trajectory does not contain a separate extended argument here; the implicit justification is that the release CSV already provides the subject ID for each session/probe row, so no path parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values. The code groups the release CSV by `eid`, creates one processing job per group, and each processed result corresponds to one session.

ii.
```python
for eid, grp in bwm_df.groupby('eid', sort=False):
    jobs.append((eid, list(grp.pid), list(grp.probe_name),
                 grp.subject.iloc[0], grp.lab.iloc[0]))
```

iii. The trajectory justification is implicit: `eid` is treated as the session identifier throughout the code and metadata, consistent with the IBL release structure.

## 1-d. How are the data split into trials?

i. Trials are taken from the session trials table loaded by `SessionLoader`; one row is one trial. After loading, trial-level masks are applied, and the surviving rows define the trial list for that session.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
trials = sess_loader.trials
keep = trials_mask(trials)
trials = trials[keep]
```

iii. The trajectory does not argue this separately. The code assumes the IBL trials table is already the authoritative trial split.

## 1-e. How are trials filtered based on quality controls?

i. The AI used a stricter mask than the human reference. First it applies the Zhang-code style `trials_mask`: reaction time in `[0.08, 2.0]`, feedback within 10 s of the go cue, non-null `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`, and `choice != 0`. After neural and behavioural loading, it further requires the wheel and whisker traces to cover the full trial window, keeping only trials where both `wheel_mask` and `whisker_mask` are true.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in NAN_EXCLUDE:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
return (~trials.eval(query)).to_numpy()
```

```python
wheel_speed, wheel_mask = bin_behavior(...)
whisker, whisker_mask = bin_behavior(...)
ok = wheel_mask & whisker_mask
```

iii. The trajectory explicitly says the agent followed the reference `load_trials_and_mask(max_trial_len=10.0)` criteria from Zhang’s code and verified the mask matched that reference. It separately justified the wheel/whisker coverage check as necessary because the decoder needs a complete behavioural label in every time bin.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is built from per-spike `times` and `clusters`, loaded probe by probe from `SpikeSortingLoader`. Cluster metadata (`label`, `acronym`) are also used to decide which units survive and which brain region each surviving unit belongs to.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
merged_times.append(spikes['times'])
merged_ids.append(spikes['clusters'].astype(np.int64) + offset)
```

```python
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
good = (clusters['label'].to_numpy() >= QC_LABEL) & ~np.isin(beryl, list(NON_GREY))
```

iii. The trajectory justification is explicit in the final summary: the agent says it followed the spike binning logic of the Zhang reference, but used the data-paper neuron inclusion criteria (`clusters.label == 1`, grey-matter regions) when deciding which clusters to keep.

## 2-b. How is the `neural` data processed?

i. Spikes from all probes in a session are merged into one population, cluster IDs are offset so probe-local IDs do not collide, spike times are concatenated and sorted, then spikes are counted into 100 bins of 20 ms for each trial. The result is left as spike counts per bin; it is not divided by bin width to convert to Hz.

ii.
```python
for spikes, clusters in zip(spikes_list, clusters_list):
    merged_times.append(spikes['times'])
    merged_ids.append(spikes['clusters'].astype(np.int64) + offset)
    offset += int(clusters.index.max()) + 1
```

```python
b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
counts = np.bincount(c * N_BINS + b, minlength=n_clusters * N_BINS)
out[k] = counts.reshape(n_clusters, N_BINS)
```

iii. The trajectory explicitly says the spike binner was verified against the Zhang reference and was “bit-identical,” but it also states in metadata that the saved neural data are “spike counts per 20 ms bin.” No explicit justification is given for not converting counts to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applied several filters. It kept only clusters with `label >= 1`, mapped cluster acronyms to Beryl regions, dropped regions mapped to `root` or `void`, required each region to have at least five such neurons within the session, and after all sessions were processed kept only regions recorded in at least two sessions.

ii.
```python
good = (clusters['label'].to_numpy() >= QC_LABEL) & ~np.isin(beryl, list(NON_GREY))
regions, counts = np.unique(beryl[good], return_counts=True)
enough = set(regions[counts >= MIN_NEURONS_PER_REGION])
good &= np.array([r in enough for r in beryl])
cluster_ids = np.nonzero(good)[0]
```

```python
keep_regions = sorted(r for r, n in n_sessions_with_region.items()
                      if n >= MIN_SESSIONS_PER_REGION)
keep = np.array([r in region_index for r in res['regions']])
spikes = res['neural'][:, keep, :]
```

iii. The trajectory explicitly justifies this as following the data paper rather than Zhang’s `qc=None` loading: well-isolated units only, grey matter only, at least five such neurons per region per session, and regions observed in at least two sessions. It also cites decoder memory pressure as part of the rationale for not keeping every sorted unit.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each trial the bin window is `[stimOn_times - 0.5 s, stimOn_times + 1.5 s]`, and spikes are placed into bins defined relative to that trial’s stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
...
b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```

iii. The trajectory explicitly says alignment/binnning was taken “verbatim” from the `params` block of `0_data_caching.py`: `stimOn_times`, window `(-0.5, 1.5)`, 20 ms bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, for 100 time bins per trial. There is no additional temporal rebinning beyond this binning.

ii.
```python
BINSIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. The trajectory explicitly states that the AI copied the 20 ms, 100-bin setup from Zhang’s code and methods description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial alignment event `stimOn_times`, but the actual values are a synthetic bin-centre grid relative to that event, not a raw signal sampled from disk.

ii.
```python
ALIGN_TIME = 'stimOn_times'
...
align_times = trials[ALIGN_TIME].to_numpy()
```

```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
inputs[:, 0, :] = bin_centres
```

iii. The trajectory justification is implicit and consistent with the final summary: time since stimulus onset is the decoder’s canonical trial grid around `stimOn_times`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No raw-data transformation is performed beyond defining the 100 bin centres from `-0.49` to `1.49` s. The same vector is copied into every trial.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
inputs[:, 0, :] = bin_centres
```

iii. No separate trajectory justification was recorded; the code comments frame this as part of the fixed reference trial grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same trial window and 20 ms binning used for neural spike counts. The input stores the centre of each neural bin, so it is aligned bin-for-bin to the neural tensor.

ii.
```python
b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```

```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
inputs[:, 0, :] = bin_centres
```

iii. The trajectory explicitly says alignment/binnning follows the reference `stimOn_times` grid and window.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trials table. A change in `probabilityLeft` marks the start of a new block.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    new_block[1:] = p[1:] != p[:-1]
```

iii. The trajectory does not justify this separately, but the implementation follows the standard inference that constant `probabilityLeft` values define blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code computes a 0-based count within each contiguous block of constant `probabilityLeft`. Importantly, this is done on the full trial table before trial filtering, then masked afterward so excluded trials still advance the count.

ii.
```python
block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
return np.arange(len(p)) - block_start
```

```python
block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
trials = trials[keep]
block_idx = block_idx[keep]
```

iii. The trajectory justification is explicit in the function docstring and consistent with the final narrative: the block count should reflect the block the mouse experienced, not a renumbered count after exclusions.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials['choice']`.

ii.
```python
choice = ((1 - trials['choice'].to_numpy()) / 2).astype(np.int8)
```

iii. The trajectory explicitly says the agent empirically verified that `trials.choice == +1` corresponds to a left report and used that to code left `0`, right `1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After trials with `choice == 0` are excluded, the code recodes `+1 -> 0` and `-1 -> 1`, then broadcasts that single class label across all 100 time bins of the trial.

ii.
```python
choice = ((1 - trials['choice'].to_numpy()) / 2).astype(np.int8)
outputs[:, 0, :] = choice[:, None]
```

iii. The trajectory explicitly gives the mapping and says it was checked empirically across sessions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials['probabilityLeft']`.

ii.
```python
PRIOR_VALUES = np.array([0.2, 0.5, 0.8])
prior = np.abs(trials['probabilityLeft'].to_numpy()[:, None]
               - PRIOR_VALUES[None, :]).argmin(axis=1).astype(np.int8)
```

iii. The trajectory justification is implicit: the decoder target is the 3-class prior, and the script keeps the unbiased `0.5` block so all three classes are represented.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The scalar `probabilityLeft` value is mapped to the nearest entry in `[0.2, 0.5, 0.8]`, yielding classes `0`, `1`, `2`, and then repeated across all time bins of the trial.

ii.
```python
prior = np.abs(trials['probabilityLeft'].to_numpy()[:, None]
               - PRIOR_VALUES[None, :]).argmin(axis=1).astype(np.int8)
outputs[:, 1, :] = prior[:, None]
```

iii. The trajectory justification is that the initial unbiased block is intentionally kept so prior remains a three-valued decoder output.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It ultimately comes from the wheel timestamps and wheel position loaded by `SessionLoader.load_wheel()`, but the code uses the processed `wheel['velocity']` exposed by the loader and then takes its absolute value.

ii.
```python
sess_loader.load_wheel()
wheel_speed, wheel_mask = bin_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
```

iii. The trajectory states that wheel speed follows the reference pipeline and uses the loader-provided wheel velocity, with `abs(...)` to get speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code relies on `SessionLoader` to produce the wheel velocity trace, takes its absolute value, resamples that continuous trace onto the decoder time grid with `bin_behavior`, and later discretizes the resampled values into tertiles.

ii.
```python
wheel_speed, wheel_mask = bin_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
```

```python
outputs[:, 2, :] = tertile_labels(wheel_speed)
```

iii. The trajectory explicitly justifies the tertile discretisation as session-specific because wheel usage differs strongly across mice/sessions, while the decoder output head is shared.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is discretized into three equal-occupancy bins using session-specific quantiles over all surviving wheel-speed samples from that session.

ii.
```python
def tertile_labels(values):
    edges = np.quantile(values, np.arange(1, N_OUTPUT_BINS) / N_OUTPUT_BINS)
    return np.searchsorted(edges, values, side='right').astype(np.int8)
```

iii. The trajectory explicitly justifies session-level tertiles instead of global thresholds because a shared global threshold would make “low” and “high” mean different behaviours in different sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is aligned by stimulus onset and sampled on the same 2 s trial window, but not at the bin centres. `bin_behavior` evaluates the continuous trace at the right edge of each 20 ms bin (`interval_start + BINSIZE` through `interval_end`), so the trace is offset by half a bin relative to a bin-centre representation.

ii.
```python
grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
...
binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. The trajectory explicitly says this was meant to match Zhang’s `get_behavior_per_interval` grid exactly, while also noting that values differ from a within-interval extrapolation only at the final bin.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the camera motion-energy stream loaded through `SessionLoader.load_motion_energy(...)`. The code prefers the left camera and falls back to the right camera if needed, then uses that camera’s `times` and `whiskerMotionEnergy` columns.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[cam]
        whisker, whisker_mask = bin_behavior(
            me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy(), align_times)
        break
```

iii. The trajectory explicitly says the agent followed the reference’s “left preferred, right fallback” logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used directly, resampled onto the decoder time grid with `bin_behavior`, and then discretized into per-session tertiles. No additional filtering or normalisation is applied in this script.

ii.
```python
whisker, whisker_mask = bin_behavior(
    me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy(), align_times)
...
outputs[:, 3, :] = tertile_labels(whisker)
```

iii. The trajectory justification mirrors wheel speed: session-level tertiles are used because motion-energy units are camera-dependent arbitrary units, so global thresholds would not be comparable across sessions.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded into three equal-occupancy classes using session-specific quantiles across all surviving whisker-motion samples in that session.

ii.
```python
def tertile_labels(values):
    edges = np.quantile(values, np.arange(1, N_OUTPUT_BINS) / N_OUTPUT_BINS)
    return np.searchsorted(edges, values, side='right').astype(np.int8)
```

iii. The trajectory explicitly justifies per-session discretisation because the motion-energy scale is not directly comparable across cameras and sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, whisker motion energy is aligned by stimulus onset and sampled over the same `[-0.5, 1.5]` s window, but on the right-edge sampling grid used by `bin_behavior` rather than on bin centres.

ii.
```python
grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. The trajectory explicitly says this grid was chosen to match Zhang’s behavioural sampling code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled missing/inconsistent data aggressively. It repaired stale ONE cache tables up front; skipped sessions with missing spike sorting or missing whisker-motion data; filtered out non-finite spike times; dropped trials failing the trial mask; and dropped any trials whose wheel or whisker stream did not cover the whole trial window or whose resampled behavioural trace was non-finite. Sessions with fewer than two surviving trials or zero surviving neurons were skipped.

ii.
```python
if PATCHED_DIR.exists() and not force:
    return PATCHED_DIR
...
datasets['rel_path'] = new_paths
datasets['exists'] = True
```

```python
finite = np.isfinite(spike_times)
spike_times, spike_clusters = spike_times[finite], spike_clusters[finite]
```

```python
if not spikes_list:
    return {'eid': eid, 'skip': 'no spike sorting available'}
...
if whisker is None:
    return {'eid': eid, 'skip': 'no whisker motion energy available'}
...
if ok.sum() < 2:
    return {'eid': eid, 'skip': f'only {int(ok.sum())} trials have complete behaviour'}
```

iii. The trajectory explicitly highlights the stale-cache repair as a necessary robustness fix, and the final summary lists the reasons sessions were dropped (no camera data, no QC neurons, or too few complete trials).

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading spike sorting for every probe/session and then binning spikes trial-by-trial. There is also a one-time full-cache filesystem scan to build patched ONE tables before session processing starts.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

```python
for k in range(len(align_times)):
    ...
    counts = np.bincount(c * N_BINS + b, minlength=n_clusters * N_BINS)
```

```python
for eid, group in datasets.groupby('eid', sort=False):
    session_dir = session_dirs.get(eid)
    ...
    index = _session_file_index(session_dir)
```

iii. The trajectory explicitly reports spike loading/binnning as the main conversion work and separately motivates the cache repair as a required up-front pass over the file tree.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining obvious per-trial loop is in `bin_spikes`. Behavioural resampling is already mostly vectorized by flattening the trial-time grid and calling `np.interp` once, so the AI eliminated the second clear per-trial loop present in the human reference. Some smaller Python loops remain for per-session/per-probe assembly and final subject/region bookkeeping.

ii.
```python
for k in range(len(align_times)):
    if i1[k] <= i0[k]:
        continue
    ...
    counts = np.bincount(c * N_BINS + b, minlength=n_clusters * N_BINS)
```

```python
grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. There is no explicit trajectory discussion of loop vectorisation beyond the final summary’s “vectorised spike binner” language. The code itself shows behaviour interpolation was vectorized across trials, while spike binning still loops trial-by-trial.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly constructs a new local ONE client inside every `_process_session` call, repeatedly instantiates `BrainRegions()`, and repeatedly recomputes per-session bin centres and quantile thresholds. It also scans the full cache directory once to build patched tables before conversion.

ii.
```python
def _process_session(...):
    from iblatlas.regions import BrainRegions
    one = get_one()
```

```python
beryl = BrainRegions().acronym2acronym(...)
...
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
...
outputs[:, 2, :] = tertile_labels(wheel_speed)
outputs[:, 3, :] = tertile_labels(whisker)
```

iii. The trajectory does not explicitly defend these repetitions. The only repeated work it directly justifies is rebuilding ONE access around the patched tables so sessions can be loaded reliably in worker processes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest extra step is the cache-table repair pass, which is needed for loading but not for the final decoder arrays themselves. Within session processing, some bookkeeping is carried only for metadata or skip reporting (`lab`, `n_probes`, `n_trials_total`, `n_trials_pass_criteria`), and `align_times` is updated after behavioural masking even though it is not used afterward. There is no large discarded analysis product analogous to an unused alternate dataset.

ii.
```python
build_patched_tables()
```

```python
align_times = align_times[ok]
...
return {
    'eid': eid, 'subject': subject, 'lab': lab, 'skip': None,
    ...
    'n_probes': len(spikes_list),
    'n_trials_total': int(len(keep)),
    'n_trials_pass_criteria': int(keep.sum()),
}
```

iii. The trajectory explicitly justifies the cache repair as operationally necessary, not scientifically necessary. For the smaller bookkeeping items, there is no separate justification beyond metadata/reporting.
