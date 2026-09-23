# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `bwm_release.csv` (the brain-wide map release table shipped with the reference code) to enumerate sessions and their probe insertions. It then builds patched ONE cache tables by merging three shipped parquet table directories and re-pointing each dataset row at the newest on-disk revision. A local-mode ONE client is pointed at these patched tables. For each session, a `SessionLoader` loads trials, wheel, and motion energy, while a `SpikeSortingLoader` loads spikes per probe. The optional `DATALIMIT_SUBSET.csv` restricts which sessions are processed.

ii.
```python
bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
subset_file = Path('/app/data/DATALIMIT_SUBSET.csv')
if subset_file.exists():
    subset = pd.read_csv(subset_file)
    bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]

for eid, grp in bwm_df.groupby('eid', sort=False):
    jobs.append((eid, list(grp.pid), list(grp.probe_name),
                 grp.subject.iloc[0], grp.lab.iloc[0]))
```

iii. The AI noted that the shipped ONE cache tables were stale relative to the file tree (files exist under revision folders not listed in the tables), and built a patched table system to resolve this. It used `bwm_release.csv` from the reference code as the session index rather than `one.search`.

## 1-b. How are the data split into subjects?

i. The subject name comes directly from the `bwm_release.csv` table's `subject` column. Subjects are collected as a list in order of first appearance. `subject_idx` maps each session to its position in the subjects list.

ii.
```python
if res['subject'] not in subjects:
    subjects.append(res['subject'])
subject_idx.append(subjects.index(res['subject']))
```

iii. The `bwm_release.csv` already contains subject identifiers, so no derivation is needed.

## 1-c. How are the data split into sessions?

i. Sessions are already the unit the `bwm_release.csv` is organized by (one row per probe insertion, grouped by `eid`). The AI groups by `eid` to get one job per session.

ii.
```python
for eid, grp in bwm_df.groupby('eid', sort=False):
    jobs.append((eid, list(grp.pid), list(grp.probe_name), ...))
```

iii. No splitting decision needed; sessions are already the unit in the source data.

## 1-d. How are the data split into trials?

i. The trials table loaded by `SessionLoader` has one row per trial, so the split is inherent in the data.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
trials = sess_loader.trials
```

iii. No splitting decision needed; trials are rows of the trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the reference code's `load_trials_and_mask` criteria: reaction time (first movement minus stimulus onset) must be between 80 ms and 2 s; feedback must occur within 10 s of the go cue (`MAX_TRIAL_LEN=10.0`); six key trial events must not be NaN (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`); choice must not be 0 (no response). Additionally, wheel and whisker motion energy traces must cover the full trial window.

ii.
```python
def trials_mask(trials):
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    return (~trials.eval(query)).to_numpy()
```

```python
ok = wheel_mask & whisker_mask
```

iii. The AI stated it followed "reference `load_trials_and_mask(max_trial_len=10.0)`" and "Verified identical to the reference mask." The `feedback_times - goCue_times > 10.0` filter and the NaN checks on six events go beyond the reference solution's simpler RT + choice + prior filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` (spike times) and `spikes.clusters` (cluster assignments). The cluster table supplies quality labels and anatomical locations for filtering.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. Spike times and cluster IDs are the standard arrays for constructing neural activity matrices.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over a 2 s trial window (-0.5 to 1.5 s around stimulus onset), giving spike counts per unit per bin. The AI stores raw spike counts (not firing rates). When a session has multiple probes, their units are merged into one population with offset cluster indices.

ii.
```python
out = np.zeros((len(align_times), n_clusters, N_BINS), dtype=np.float32)
for k in range(len(align_times)):
    ...
    b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
    np.clip(b, 0, N_BINS - 1, out=b)
    counts = np.bincount(c * N_BINS + b, minlength=n_clusters * N_BINS)
    out[k] = counts.reshape(n_clusters, N_BINS)
return out
```

iii. The AI stated: "I verified my vectorised spike binner against the reference `bin_spiking_data` on a full session: bit-identical." The counts are stored as float32 but are not divided by the bin width to convert to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1.0` (well-isolated) are kept. Additionally, units in non-grey-matter regions (Beryl `root` and `void`) are excluded. Further, the AI enforces a minimum of 5 well-isolated neurons per region per session, and later a minimum of 2 sessions per region across the whole dataset.

ii.
```python
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
good = (clusters['label'].to_numpy() >= QC_LABEL) & ~np.isin(beryl, list(NON_GREY))
regions, counts = np.unique(beryl[good], return_counts=True)
enough = set(regions[counts >= MIN_NEURONS_PER_REGION])
good &= np.array([r in enough for r in beryl])
```

```python
keep_regions = sorted(r for r, n in n_sessions_with_region.items()
                      if n >= MIN_SESSIONS_PER_REGION)
```

iii. The AI stated: "I followed the data paper rather than the Zhang code, which passes `qc=None` and keeps all ~1350 Kilosort units per session. I applied the BWM inclusion criteria." The additional filters (min 5 neurons/region, min 2 sessions/region, excluding `root`) go beyond both the reference solution and the Zhang code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are aligned to `stimOn_times` (stimulus onset). The bin window starts at `stimOn_times - 0.5` and ends at `stimOn_times + 1.5`.

ii.
```python
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
...
b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```

iii. Alignment to stimulus onset follows the reference code's `params` block.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins total over the 2 s window. No rebinning or interpolation is applied.

ii.
```python
BINSIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. Matches the reference code's `'binsize': 0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The input is the bin centres of the 100 bins spanning -0.5 to 1.5 s.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
inputs[:, 0, :] = bin_centres
```

iii. The time variable is defined by the binning grid, not derived from raw data beyond the alignment event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing; the bin centres are computed as `TIME_WINDOW[0] + (i + 0.5) * BINSIZE` for i in 0..99.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
```

iii. N/A

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin centres correspond exactly to the centres of the neural data bins, since both use the same window and bin size.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
inputs[:, 0, :] = bin_centres
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` marks a new block boundary.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    new_block[1:] = p[1:] != p[:-1]
    block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
    return np.arange(len(p)) - block_start
```

iii. The trials table has no block identifier, so blocks are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based position of each trial within its block is computed on the full trials table (before filtering), so excluded trials still advance the count. The count is then filtered to kept trials and broadcast to all time bins.

ii.
```python
block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
trials = trials[keep]
block_idx = block_idx[keep]
...
inputs[:, 1, :] = block_idx[:, None]
```

iii. Computing on the full table before filtering ensures the block count reflects the animal's actual experience.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, which is +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice = ((1 - trials['choice'].to_numpy()) / 2).astype(np.int8)
```

iii. The AI verified empirically that `trials.choice == +1` corresponds to a left report.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The mapping `(1 - choice) / 2` converts +1 to 0 (left) and -1 to 1 (right). No-response trials (choice=0) are already filtered out.

ii.
```python
choice = ((1 - trials['choice'].to_numpy()) / 2).astype(np.int8)
outputs[:, 0, :] = choice[:, None]
```

iii. N/A

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, 0.8.

ii.
```python
prior = np.abs(trials['probabilityLeft'].to_numpy()[:, None]
               - PRIOR_VALUES[None, :]).argmin(axis=1).astype(np.int8)
```

iii. Uses closest-value matching against `PRIOR_VALUES = [0.2, 0.5, 0.8]`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI finds the index of the closest value in `[0.2, 0.5, 0.8]` using argmin of absolute differences, mapping 0.2->0, 0.5->1, 0.8->2.

ii.
```python
PRIOR_VALUES = np.array([0.2, 0.5, 0.8])
prior = np.abs(trials['probabilityLeft'].to_numpy()[:, None]
               - PRIOR_VALUES[None, :]).argmin(axis=1).astype(np.int8)
```

iii. Functionally equivalent to a direct mapping but uses argmin instead.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From wheel position and timestamps (`_ibl_wheel.position`, `_ibl_wheel.timestamps`), loaded via `SessionLoader.load_wheel()` which computes velocity internally.

ii.
```python
sess_loader.load_wheel()
wheel_speed, wheel_mask = bin_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
```

iii. Same as the reference: absolute value of the velocity from `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` interpolates wheel position to 1000 Hz and differentiates with a 20 Hz Butterworth low-pass filter to get velocity. The AI takes the absolute value (speed), then interpolates it onto the trial time grid using `np.interp`. The grid uses right bin edges: `align + TIME_WINDOW[0] + (1..N_BINS) * BINSIZE`. Finally, per-session tertile labels are applied.

ii.
```python
grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
```

```python
def tertile_labels(values):
    edges = np.quantile(values, np.arange(1, N_OUTPUT_BINS) / N_OUTPUT_BINS)
    return np.searchsorted(edges, values, side='right').astype(np.int8)
```

iii. The AI stated: "Behavioural sampling matches the reference `get_behavior_per_interval` grid and trial mask exactly."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tertiles: the 1/3 and 2/3 quantiles of the session's wheel speed values define the thresholds, yielding three categories (low=0, medium=1, high=2).

ii.
```python
def tertile_labels(values):
    edges = np.quantile(values, np.arange(1, N_OUTPUT_BINS) / N_OUTPUT_BINS)
    return np.searchsorted(edges, values, side='right').astype(np.int8)
```

iii. The AI justified per-session quantiles: "a global threshold would mean 'low'/'high' referred to different behaviour in different sessions."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated onto the same trial time grid, which uses the right bin edges offset from `stimOn_times`. This is slightly different from the neural bin centres but covers the same window.

ii.
```python
grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. The AI stated the grid matches the reference `get_behavior_per_interval`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `<side>Camera.ROIMotionEnergy` and `_ibl_<side>Camera.times`, preferring the left camera, falling back to the right.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[cam]
        whisker, whisker_mask = bin_behavior(
            me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy(), align_times)
        break
    except Exception:
        continue
```

iii. Follows the reference's preference for left camera with right as fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is interpolated onto the trial time grid (right bin edges) via `np.interp`, then discretized into per-session tertiles. NaN values are dropped before interpolation; trials without sufficient coverage are masked out.

ii.
```python
whisker, whisker_mask = bin_behavior(
    me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy(), align_times)
...
outputs[:, 3, :] = tertile_labels(whisker)
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Per-session tertiles, identical to wheel speed discretization.

ii.
```python
outputs[:, 3, :] = tertile_labels(whisker)
```

iii. Same justification as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto the trial time grid (right bin edges from stimulus onset).

ii.
```python
grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
```

iii. Same alignment approach as wheel speed and neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) Patched cache tables handle stale ONE table entries. (2) NaN values in six key trial events cause trial exclusion. (3) NaN values in behavioral traces are dropped before interpolation. (4) Trials where wheel or whisker traces don't cover the full window are masked out. (5) Sessions with no spike sorting, no neurons passing QC, or fewer than 2 usable trials are skipped. (6) Sessions whose neurons all fall in regions not recorded in at least 2 sessions are dropped.

ii.
```python
if len(spikes) == 0:
    continue
...
if ok.sum() < 2:
    return {'eid': eid, 'skip': f'only {int(ok.sum())} trials have complete behaviour'}
...
finite = np.isfinite(values)
times, values = times[finite], values[finite]
```

iii. The AI reported: "19 sessions were dropped: 14 have no camera data at all, 4 have no neurons passing QC, 1 has <2 trials with complete behaviour."

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting from disk (hundreds of megabytes per probe), and the patched table construction which walks the filesystem for every session.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

```python
def build_patched_tables(force=False, verbose=True):
    ...
    for eid, group in datasets.groupby('eid', sort=False):
        ...
        index = _session_file_index(session_dir)
```

iii. The spike sorting I/O dominates per-session time; the table patching is a one-time cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates trial by trial (`for k in range(len(align_times))`), which could potentially be vectorized by offsetting spike indices across trials. The behavior binning has a similar structure but uses `np.interp` on the full flattened grid, which is already vectorized across trials.

ii.
```python
for k in range(len(align_times)):
    ...
    b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
    counts = np.bincount(c * N_BINS + b, minlength=n_clusters * N_BINS)
    out[k] = counts.reshape(n_clusters, N_BINS)
```

iii. The per-trial spike binning loop is functionally identical to the reference solution's approach.

## 10-c. What processing does the code repeat multiple times?

i. The ONE client and patched tables are rebuilt for every worker process (via `get_one()` called inside each `_process_session`). The `BrainRegions()` atlas is also instantiated fresh in every session.

ii.
```python
def _process_session(eid, pids, probe_names, subject, lab):
    ...
    one = get_one()
    ...
    beryl = BrainRegions().acronym2acronym(...)
```

iii. These are repeated due to the multiprocessing architecture (spawn context), where each worker needs its own instances.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes the `file_size` for every dataset row during table patching by stat-ing each file, which is only needed for ONE's internal consistency checks. The `hash` column is set to `None` explicitly. The `n_trials_total` and other metadata fields are computed but only stored in metadata, not used for decoding.

ii.
```python
datasets['file_size'] = [
    (session_dirs[eid] / rel).stat().st_size
    for eid, rel in zip(datasets['eid'], datasets['rel_path'])]
datasets['hash'] = None
```

iii. These are side effects of making the patched ONE tables internally consistent.
