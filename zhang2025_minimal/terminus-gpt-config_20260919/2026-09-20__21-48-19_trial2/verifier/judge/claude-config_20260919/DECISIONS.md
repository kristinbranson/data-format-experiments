# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from the locally mounted IBL ONE cache, with no network access (Alyx is blocked in the container). `ONE(cache_dir=CACHE, mode='local', silent=True)` is created and the Brain Wide Map release tables are attached in-memory with `one._cache = load_tables(CACHE/'Brainwidemap')` (the mutating `one.load_cache(tag=...)` was deliberately abandoned after it corrupted the shared parquet tables when 16 workers raced on it). The universe of sessions is **not** taken from `one.search`; it is taken from the reference repository's bundled `code/code_zhang2025/data/bwm_release.csv`, which lists 459 unique `eid`s with their `subject`, `pid` and `probe_name`. For each session: spikes/clusters are loaded per probe insertion with `SpikeSortingLoader(pid=...)` and merged with the repo's `merge_probes`; the trials table is read **directly** from the newest revision folder (`alf/#*#/_ibl_trials.table.pqt`) instead of via `SessionLoader.load_trials()`; the wheel is loaded with `SessionLoader.load_wheel()`; whisker motion energy and camera times are read **directly** as `.npy` files instead of via `SessionLoader.load_motion_energy()`. The run was sharded 16 ways with per-session resumable intermediate pickles and per-session `.failed` markers, then assembled in a single final pass. 444 of 459 sessions converted (15 excluded).

ii.
```python
one = ONE(cache_dir=CACHE, mode='local', silent=True)
one._cache = load_tables(CACHE/'Brainwidemap')
```
```python
def selected_sessions(n=None):
    """Return all release sessions, or the mounted data-limit subset when supplied."""
    bwm = pd.read_csv(RELEASE, index_col=0)
    subset_file = ROOT/'data/DATALIMIT_SUBSET.csv'
    if subset_file.exists():
        subset = pd.read_csv(subset_file)
        values = set(subset.astype(str).to_numpy().ravel())
        bwm = bwm[bwm.eid.astype(str).isin(values) | bwm.pid.astype(str).isin(values)]
    rows = []
    for eid, group in bwm.groupby('eid', sort=False):
        rows.append((str(group.subject.iloc[0]), str(eid),
                     list(group.pid.astype(str)), list(group.probe_name.astype(str))))
```
```python
def local_spiking_data(one, eid, pid, pname):
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    spikes['clusters'] = np.array(spikes['clusters'], dtype=np.int32, copy=True)
    clusters = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False).to_df()
    return spikes, clusters

def load_local_trials(one, eid):
    """Load the newest consolidated trial table (avoids obsolete ALF attributes)."""
    path = Path(one.eid2path(eid))/'alf'
    files = sorted(path.glob('#*#/_ibl_trials.table.pqt'))
    ...
    return pd.read_parquet(files[-1])
```

iii. From the trajectory: the agent first tried the reference loaders verbatim and hit three concrete, verified failures. (1) `load_spiking_data` in the repo opens a *remote* raw-ephys stream only to read the sampling frequency, which is unused downstream and requires blocked network access, so it was replaced with a local-only loader. (2) `SessionLoader.load_trials()` resolved to an obsolete standalone ALF attribute (returning only `goCueTrigger_times`) because of version skew between the installed ONE and the 2025 revisioned cache; the agent inspected the revision folders, confirmed `#2025-03-03#/_ibl_trials.table.pqt` was complete, and read it directly. (3) `load_motion_energy` combined mismatched revisions/attribute names, so camera times and ROI motion energy are matched directly by view and sample count. It also chose `bwm_release.csv` as the session universe because `stage_cache.sh` documents that the full release should be converted when `DATALIMIT_SUBSET.csv` is absent, and because the repo's own `0_data_caching.py` draws its sessions from that CSV.

## 1-b. How are the data split into subjects?

i. The subject name comes straight from the `subject` column of `bwm_release.csv`, taken as the first row of each `eid` group; no path or filename parsing. At assembly the subject list is the sorted unique set of names and `subject_idx` is each session's index into it. Result: 136 subjects over 444 sessions.

ii.
```python
rows.append((str(group.subject.iloc[0]), str(eid), ...))
```
```python
subjects = sorted(set(s['subject'] for s in sessions))
smap = {s:i for i,s in enumerate(subjects)}
...
subject_idx=np.array([smap[s['subject']] for s in sessions], np.int32),
```

iii. Not discussed at length; the release CSV already carries a unique subject id per session, so nothing has to be derived.

## 1-c. How are the data split into sessions?

i. No splitting is done. The release is organised by session, so `bwm_release.csv` is grouped by `eid` (one row per probe insertion collapses to one row per session, carrying the list of `pid`s and probe names). The list order of `neural`/`input`/`output` is the order in which the 444 successful session intermediates are loaded.

ii.
```python
for eid, group in bwm.groupby('eid', sort=False):
    rows.append((str(group.subject.iloc[0]), str(eid),
                 list(group.pid.astype(str)), list(group.probe_name.astype(str))))
```

iii. A session is the native unit of the release; the multiple probes of one session are merged into one population (see 2-b) rather than treated as separate sessions, following the repo's `merge_probes`.

## 1-d. How are the data split into trials?

i. The consolidated trials table has one row per trial, so the split is given by the data. Trial windows are built as `stimOn_times + (-0.5, 1.5)` and handed to the repo's `bin_spiking_data`, which turns the session-wide spike train into one array per trial. All per-trial quantities are indexed by the same row order, and the retained rows are `idx = np.flatnonzero(valid)` into the original table.

ii.
```python
PARAMS = dict(interval_len=2, binsize=BIN, single_region=False,
              align_time='stimOn_times', time_window=(OFF0, OFF1))
...
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
```

iii. No decision to make; the agent confirmed by reading `bin_spiking_data` that intervals are constructed directly from `trials_df[align_time] + time_window`.

## 1-e. How are trials filtered based on quality controls?

i. Three filters intersected. (1) `paper_trial_mask` reimplements the repo's `load_trials_and_mask(..., max_trial_len=10.0)` exactly: no NaN in `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`; reaction time (`firstMovement_times - stimOn_times`) in [0.08, 2.0] s; trial duration (`feedback_times - goCue_times`) ≤ 10 s; `choice != 0` (no-response trials dropped). Unbiased 0.5 blocks are kept. (2) The wheel window must be covered. (3) The whisker window must be covered. Coverage means: at least one sample strictly inside the window, no non-finite values, and the first/last in-window sample within one 20 ms bin of the window edges — the repo's own `get_behavior_per_interval` skip rules. A session left with fewer than two jointly valid trials is dropped entirely. Result: 188,925 trials over 444 sessions (≈425/session).

ii.
```python
def paper_trial_mask(trials):
    required = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                'firstMovement_times', 'feedbackType']
    valid = trials[required].notna().all(axis=1).to_numpy(dtype=bool, copy=True)
    rt = trials.firstMovement_times.to_numpy() - trials.stimOn_times.to_numpy()
    duration = trials.feedback_times.to_numpy() - trials.goCue_times.to_numpy()
    valid &= (rt >= 0.08) & (rt <= 2.0)
    valid &= duration <= 10.0
    valid &= trials.choice.to_numpy() != 0
    return valid
```
```python
if len(vx) == 0 or not np.all(np.isfinite(vx)):
    out.append(None); continue
if abs(beg-tx[0]) > BIN or abs(end-tx[-1]) > BIN:
    out.append(None); continue
```
```python
valid = paper_mask & wheel_good & whisk_good
idx = np.flatnonzero(valid)
if len(idx) < 2:
    raise RuntimeError('fewer than two valid trials')
```

iii. Trajectory step 15: "Reference neural processing ... applies the exact paper trial criteria: 0.08–2.0 s reaction time, required non-NaN events, and exclusion of no-response choice 0, while retaining unbiased prior-0.5 trials", and step 21 adds the ≤10 s trial-duration criterion from `prepare_data`. Step 28: the agent deliberately intersects the wheel and whisker masks rather than reproducing what it judged to be a bug in the repo's `bin_behaviors` (`beh_mask` is overwritten per behaviour and combined with Python `and`, so effectively only the last behaviour's mask applies), "because every retained trial must have all requested decoder outputs".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from each probe's spike sorting, merged across probes. The merged cluster table's `acronym` column supplies the anatomical label (mapped to Beryl) but contributes nothing to the array values. No other spike attributes are used.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
...
neural_df = {'spike_times': spikes['times'], 'spike_clusters': spikes['clusters']}
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials, ...)
```

iii. Step 17: "Spike binning constructs trial intervals directly from `stimOn_times + (-0.5, 1.5)`, selects all spikes belonging to all chosen clusters, and returns each trial as time-by-neuron internally; the target format requires its transpose."

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 non-overlapping 20 ms bins per trial by the repo's `bin_spiking_data`/`get_spike_data_per_interval` (left-inclusive, right-exclusive `bincount2D`). No smoothing, no rate conversion, no z-scoring: the stored values are raw spike **counts**, cast to `int16`, transposed to `(n_neurons, n_timepoints)`. Probes of one session are merged into a single population with `merge_probes`, which offsets the second probe's cluster ids and re-sorts all spikes by time. Neuron ordering within a session is `np.unique(spike_clusters)` (the `used` array returned by `bin_spiking_data`), so clusters that fired no spikes at all in the session are dropped implicitly. Cluster acronyms are mapped to the Beryl ontology for `brain_region_idx`. Mean 1351 neurons/session (min 135, max 3140).

ii.
```python
cluster_ids = np.arange(len(clusters), dtype=np.int64)
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
...
neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
...
beryl = BrainRegions().acronym2acronym(clusters.acronym.to_numpy(), mapping='Beryl')
beryl = np.asarray(beryl)[np.asarray(used, dtype=int)]
```

iii. Steps 15/18/24: the agent read the repo and recorded that it "merges every probe listed for a session, uses all clusters", "maps Allen cluster acronyms to the Beryl ontology, retains every unique Beryl region, and bins spike counts with left-inclusive/right-exclusive 20 ms intervals", and that "cluster order is sorted unique merged cluster ID, which also defines each session's neuron-region index". It kept raw counts because the methods paper describes the model input as binned spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron quality control is applied at all.** `load_spiking_data` is replaced by `local_spiking_data`, which calls `merge_clusters(..., compute_metrics=False)` and returns every cluster; `cluster_ids = np.arange(len(clusters))` selects all of them. The cluster `label` (the IBL 0/⅓/⅔/1 QC score) is never read. Units whose Beryl acronym is `void` (histologically outside the brain) are also kept. The only implicit exclusion is clusters with zero spikes in the whole session, dropped by `np.unique` inside `bin_spiking_data`. Resulting totals: 599,865 units retained, of which 85,846 are `root` and 12,759 are `void`; mean 1351 units/session. (The expert solution applies `label >= 1` plus a `void` drop and retains 72,417 units, mean 164/session.) The final pickle is 53.2 GB.

ii.
```python
def local_spiking_data(one, eid, pid, pname):
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    ...
    clusters = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False).to_df()
    return spikes, clusters
```
```python
cluster_ids = np.arange(len(clusters), dtype=np.int64)   # every cluster, no QC
```
```python
neuron_filter='all clusters; probes in the same session merged',
```

iii. Step 15: "Reference neural processing uses all Kilosort clusters (`qc=None`)". This is factually what the reference repository does — `prepare_data` calls `load_spiking_data(one, pid, eid=..., pname=...)` with no `qc` argument, so `qc=None` and every sorted cluster is binned — and the methods paper states it directly: "we bin spike counts using all neurons, sorted by Kilosort 2.5, from each session." The agent did not discuss the data paper's competing "stringent quality-control metrics ... 75,708 well-isolated neurons" criterion, and never considered dropping `void`/`root` units (the repo drops those only later, in `3_decode_multi_region.py`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times`, by constructing per-trial intervals `[stimOn - 0.5, stimOn + 1.5]` and binning spikes inside them. All IBL streams (spikes, trial events, wheel, camera) are already on one synchronised session clock, so no clock correction is applied. `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)` are passed through to `bin_spiking_data`, which does `intervals = np.vstack([trials[align_time] + tw[0], trials[align_time] + tw[1]]).T`.

ii.
```python
OFF0, OFF1 = -0.5, 1.5
PARAMS = dict(interval_len=2, binsize=BIN, single_region=False,
              align_time='stimOn_times', time_window=(OFF0, OFF1))
```
```python
metadata=dict(..., temporal_alignment_event='visual stimulus onset (stimOn_times)',
              off_start=-0.5, off_end=1.5, ...)
```

iii. Step 12: "The reference code establishes the key common representation required here: stimulus-onset alignment, -0.5 to +1.5 seconds, 20 ms bins, all brain regions, and outputs including choice, block prior, wheel speed, and whisker motion energy." The agent noted (step 4) that the methods paper uses 50 ms bins for choice/prior at stimulus onset and 20 ms bins at movement onset for the dynamic behaviours, and explicitly decided that "this task requires one jointly aligned representation at stimulus onset for all four outputs", resolving the conflict in favour of the stimulus-onset window with 20 ms bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (`time_bin_size: 20.0` ms), 100 bins per trial spanning the 2 s window, identical for every trial and session (`T_min = T_max = 100`). Spikes are histogrammed once directly at 20 ms; there is no rebinning, resampling or smoothing of the neural data. The behavioural traces are resampled (interpolated) onto the same 20 ms grid — see 7-b/8-b.

ii.
```python
BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
PARAMS = dict(interval_len=2, binsize=BIN, ...)
...
time_bin_size=20.0,
neural_representation='Kilosort 2.5 spike counts in non-overlapping 20 ms bins',
```

iii. Step 12/24: the repo's caching configuration uses `binsize=0.02` over a 2 s window, and the methods paper states "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable; it is the analysis time grid itself, defined by `stimOn_times` (the alignment event) plus the fixed window and bin size. One 100-element vector is built once and reused for every trial of every session.

ii.
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0
```
```python
input_names=['time since stimulus onset', 'trial number in block'],
```

iii. Implicit in step 22/28: the agent adopted the repo's time axis so that the input time base and the behavioural sample times coincide exactly.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The values are the **right edges** of the 100 neural bins: −0.48, −0.46, …, 1.50 s (confirmed by the verifier's reported input range [−0.48, 1.50]). It is a continuous ramp, tiled identically into row 0 of every trial's `(2, 100)` input array.

ii.
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0
...
inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. Step 22: the agent established that the repo interpolates continuous signals "at bin right edges from -0.48 through +1.5 seconds (100 values)" and used exactly that vector as the time input so that inputs, outputs and neural bins share one axis.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction: element *t* of the time vector is the right edge of neural bin *t*, and both are measured from the same `stimOn_times`. The neural bin *t* covers `[stimOn − 0.5 + 0.02t, stimOn − 0.5 + 0.02(t+1))`, and `time[t] = −0.5 + 0.02(t+1)` is its closing edge. The same vector is used to sample wheel and whisker, so all four streams are on one grid bin-for-bin.

ii.
```python
xi = np.linspace(beg+BIN, end, nbin)          # behaviour sample times, same convention
...
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0
inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. Step 22 (as above): the choice of right edges rather than bin centres was made to reproduce the repo's `get_behavior_per_interval` grid exactly.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft` alone. The trials table has no block id, so block boundaries are recovered as the positions where `probabilityLeft` changes value.

ii.
```python
block_trial = trial_number_in_block(trials.probabilityLeft.to_numpy())[idx]
```

iii. Not discussed explicitly in the trajectory; the agent noted in step 24 that the raw `probabilityLeft` values are carried "before task-specific remapping", which is what makes the block reconstruction possible.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A running counter that resets to 1 whenever `probabilityLeft` changes, so the value is the 1-based position of a trial within its block. It is computed over the **full, unfiltered** trials table and only then indexed by the retained trials, so a dropped trial still advances the counter and the number reflects the animal's true position in the block. It is stored as `float32` and tiled across all 100 bins. Observed range 1–99 (the expert's 0-based equivalent is 0–98).

ii.
```python
def trial_number_in_block(prob):
    out = np.empty(len(prob), dtype=np.float32)
    k = 0
    previous = None
    for i, value in enumerate(prob):
        if i == 0 or value != previous:
            k = 1
        else:
            k += 1
        out[i] = k
        previous = value
    return out
```
```python
block_trial = trial_number_in_block(trials.probabilityLeft.to_numpy())[idx]
...
inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. Not explicitly justified in the trajectory beyond the format requirement that per-trial inputs be broadcast over time ("static variables will be repeated across time so all outputs share the 100-bin temporal axis", step 28). The counting-before-filtering behaviour follows from where the call sits in `process_session`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single `choice` column of the trials table, which is +1, −1 or 0. Trials with 0 (no response) have already been removed by `paper_trial_mask`, and the code asserts that only ±1 remain.

ii.
```python
choice_raw = trials.choice.to_numpy()[idx]
if not np.all(np.isin(choice_raw, [-1, 1])):
    raise ValueError(f'unexpected choices {np.unique(choice_raw)}')
```

iii. Step 24: "raw IBL choice and probabilityLeft values before task-specific remapping" — the agent went looking for the repo's own encodings and found only that `bin_behaviors` passes `trials_df['choice']` through unchanged.

## 5-b. What processing is involved in computing `output` *Choice*?

i. A single boolean recode, `choice == +1 → 1`, `choice == −1 → 0`, stored as `int8` and tiled over all 100 bins, with `output_values[0] = ['left', 'right']` (so 0 is declared to mean left and 1 to mean right). **This inverts the IBL convention**: in the IBL trials table `choice == +1` is a leftward choice and `choice == −1` is a rightward choice (`brainbox/behavior/training.py:591` defines `rightward = trials.choice == -1`, and line 695 comments "choice == -1 means contrast on right hand side"). Verified empirically on `NYU-11/2020-02-18/001`: among correct trials, all 236 with a left-side stimulus have `choice = +1` and all 203 with a right-side stimulus have `choice = −1`. The agent's code therefore labels left choices as "right" and right choices as "left", i.e. the opposite of the instruction's `left = 0, right = 1`.

ii.
```python
choice = (choice_raw == 1).astype(np.int8)
...
out = np.vstack((np.full(T, s['choice'][j], np.int8),
                 np.full(T, s['prior'][j], np.int8), wb, mb))
...
output_values=[['left', 'right'], ...]
```

iii. No justification appears anywhere in the trajectory. The agent never searched for or stated the sign convention of `trials.choice`; the mapping was written directly without checking, while the neighbouring `probabilityLeft` mapping was taken explicitly from the task instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the values 0.2, 0.5 and 0.8.

ii.
```python
pleft_raw = trials.probabilityLeft.to_numpy()[idx]
```

iii. Step 24 (as for choice): the repo passes `probabilityLeft` through as `block`; the categorical mapping is specified by the task instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A dictionary recode `0.2 → 0, 0.5 → 1, 0.8 → 2` (with a `round(x, 1)` guard against float representation error), stored as `int8` and tiled over all 100 bins, with `output_values[1] = ['0.2', '0.5', '0.8']`. Unbiased 0.5 blocks are retained rather than dropped. Any unexpected value would raise a `KeyError` and fail the session. Resulting distribution 0.417 / 0.141 / 0.442, essentially identical to the expert's 0.418 / 0.141 / 0.442.

ii.
```python
pmap = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([pmap[round(float(x), 1)] for x in pleft_raw], np.int8)
```

iii. Directly from the Decoder Task specification ("Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2"); the agent noted in step 15 that the repo retains the unbiased 0.5 trials.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded through `SessionLoader.load_wheel()`, from which the absolute value of the derived `velocity` column is taken.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()
wheel_t = sl.wheel.times.to_numpy()
wheel_v = np.abs(sl.wheel.velocity.to_numpy())
```

iii. Step 20: "Wheel speed is the absolute value of SessionLoader's ... wheel velocity", copied from the repo's `load_target_behavior(one, eid, 'wheel-speed')`. Step 36: the agent deliberately kept `SessionLoader` here (rather than reading the raw npy files as it did for trials and camera) "because it also computes the paper's smoothed wheel velocity", and step 37 confirmed "wheel processing works correctly".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) Inside `SessionLoader.load_wheel`, the event-driven wheel position is interpolated onto a uniform 1000 Hz grid and differentiated into velocity with a low-pass filter (the IBL default), giving rad/s; the absolute value is taken. (2) Per trial, the samples strictly inside `[stimOn − 0.5, stimOn + 1.5]` are selected with `searchsorted(..., 'right')` / `searchsorted(..., 'left')` and linearly interpolated onto the 100 bin right edges `np.linspace(beg + 0.02, end, 100)` — a byte-for-byte reimplementation of the repo's `get_behavior_per_interval`, except that `np.interp` (edge clamping) is used instead of `interp1d(fill_value='extrapolate')`. (3) At assembly the trace is discretised into 3 classes (see 7-c). Trials failing the coverage/finiteness checks are dropped rather than imputed.

ii.
```python
def interpolate(times, values):
    out, good = [], np.zeros(len(trials), bool)
    nbin = int(np.ceil((OFF1-OFF0)/BIN))
    for i, onset in enumerate(trials.stimOn_times.to_numpy()):
        beg, end = onset + OFF0, onset + OFF1
        ib = np.searchsorted(times, beg, side='right')
        ie = np.searchsorted(times, end, side='left')
        tx, vx = times[ib:ie], values[ib:ie]
        if len(vx) == 0 or not np.all(np.isfinite(vx)):
            out.append(None); continue
        if abs(beg-tx[0]) > BIN or abs(end-tx[-1]) > BIN:
            out.append(None); continue
        xi = np.linspace(beg+BIN, end, nbin)
        out.append(np.interp(xi, tx, vx).astype(np.float32))
        good[i] = True
    return out, good
```

iii. Step 22: "Continuous behaviors are sliced strictly inside each trial window, required to cover both boundaries within one 20 ms bin, and linearly interpolated at bin right edges from -0.48 through +1.5 seconds (100 values)." The agent reproduced this deliberately rather than inventing its own resampling. (Note: the metadata string calls the filter "Gaussian-smoothed"; this wording is inherited verbatim from the repo's docstring and is inaccurate — `SessionLoader` uses a Butterworth low-pass — but it describes the loader, not code the agent wrote.)

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 classes by **global pooled tertiles**: after all 444 sessions are converted, every retained wheel sample from every trial of every session is concatenated into one vector, the 1/3 and 2/3 quantiles of that pooled vector are taken as two global cut points, and `np.digitize` is applied with the same two edges everywhere. The edges are `[0.01512, 0.40508]` rad/s and are recorded in `metadata['wheel_speed_bin_edges']`. This makes the classes exactly balanced *globally* (0.3333/0.3333/0.3333) but not within a session; measured over a random sample of 25 sessions, per-session fractions ranged from about 0.20 to 0.53 per class, i.e. still reasonably balanced because wheel speed is in physical units comparable across rigs. (The expert instead uses the 33rd/67th percentiles of each session's own trace.)

ii.
```python
def category_edges(sessions, key):
    values = np.concatenate([np.concatenate(s[key]) for s in sessions])
    edges = np.quantile(values, [1/3, 2/3])
    if not edges[0] < edges[1]:
        edges = np.array([np.nanpercentile(values, 33.333),
                          np.nanpercentile(values, 66.667)])
    return edges.astype(float)
...
wheel_edges = category_edges(sessions, 'wheel')
...
wb = np.digitize(s['wheel'][j], wheel_edges).astype(np.int8)
...
discretization='global pooled tertiles over retained samples',
```

iii. Step 28: "Dynamic variables will be discretized into global tertiles after alignment". Step 45 records the agent's check that "tertile discretization is exactly balanced" on the verifier output. The repo gives no guidance here — it decodes these targets as continuous variables — so the tertile split is the agent's own answer to the instruction "Wheel speed discretized into 3 bins".

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is sampled at the right edge of each of the 100 neural bins, measured from the same `stimOn_times`, so element *t* of the wheel row corresponds to neural bin *t*. Wheel timestamps and spike times share the session clock, so no further correction is applied. Trials where the wheel does not cover the window (to within one bin at each edge) are dropped from the session entirely, so neural and wheel rows never disagree in length.

ii.
```python
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
...
valid = paper_mask & wheel_good & whisk_good
```

iii. Step 22/28 (as above): the interpolation grid was chosen precisely so that behaviour, inputs and neural bins share one temporal axis.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<view>Camera.ROIMotionEnergy.npy` together with `_ibl_<view>Camera.times.npy`, read directly from the session's `alf` directory (newest revision first) rather than through `SessionLoader.load_motion_energy`. The left camera is preferred and the right is used as a fallback. A view is accepted only if a motion-energy array can be found whose length equals the camera times array; if neither view yields a match the whole session is excluded.

ii.
```python
for view in ('left', 'right'):
    tf = sorted(alf.glob(f'#*#/_ibl_{view}Camera.times.npy')) or sorted(alf.glob(f'_ibl_{view}Camera.times.npy'))
    mf = sorted(alf.glob(f'#*#/{view}Camera.ROIMotionEnergy.npy')) or sorted(alf.glob(f'{view}Camera.ROIMotionEnergy.npy'))
    for tfile in reversed(tf):
        t = np.load(tfile)
        match = None
        for mfile in reversed(mf):
            m = np.load(mfile)
            if len(m) == len(t):
                match = m
                break
        ...
if motion_t is None:
    raise FileNotFoundError('no matching whisker motion energy and camera times')
```

iii. Step 23: "The wrapper uses left whisker motion energy when available and falls back to right only if left loading fails" — copied from the repo's `bin_behaviors`. Step 37: the agent found that `SessionLoader.load_motion_energy` "combines mismatched revisions/attribute names" under this cache, and step 38 verified on a real session that "camera timestamps and motion-energy arrays match exactly by view and sample count, so direct local loading is straightforward and faithful". The length-equality test is its guard against pairing files from different revisions.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the raw trace — the released per-frame ROI motion energy is used as-is, with no filtering, normalisation or per-session rescaling. It goes through exactly the same `interpolate` helper as the wheel: slice the samples strictly inside the trial window, require full coverage within one bin at each edge and all-finite values, then linearly interpolate onto the 100 bin right edges. It is then discretised (see 8-c).

ii.
```python
whisk, whisk_good = interpolate(motion_t, motion_v)
```
(same `interpolate` shown in 7-b)

iii. Step 22/23: the repo applies no processing to the motion-energy trace beyond the interval interpolation, so the agent did the same.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to wheel speed: **global pooled tertiles** over every retained whisker sample from all 444 sessions, giving a single pair of cut points `[2.7383, 7.8483]` applied with `np.digitize` to every session. Globally this is exactly balanced (0.3333 each), but whisker ROI motion energy is in arbitrary, camera- and session-dependent units (the left camera is 1280×1024 @ 60 Hz and the right 640×512 @ 150 Hz, and the ROI scale differs per session), so per-session class distributions are severely skewed. Measured on a random sample of 25 sessions: `[0.996, 0.004, 0.000]`, `[0.951, 0.049, 0.000]`, `[0.333, 0.667, 0.000]`, `[0.021, 0.545, 0.434]`, `[0.072, 0.254, 0.674]` … — several sessions have an entirely empty class and one is 99.6% a single class. The expert's per-session percentile split keeps all sessions near 1/3–1/3–1/3.

ii.
```python
whisk_edges = category_edges(sessions, 'whisk')
...
mb = np.digitize(s['whisk'][j], whisk_edges).astype(np.int8)
...
whisker_motion_energy_bin_edges=whisk_edges.tolist(),
```

iii. Step 28: "Dynamic variables will be discretized into global tertiles after alignment." The agent verified balance only on the *pooled* distribution (step 45, on a single session at that point) and never checked the per-session distributions, nor discussed that motion energy has no common physical scale across sessions and cameras.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as the wheel: interpolated onto the right edges of the same 100 neural bins, relative to the same `stimOn_times`, on the shared session clock. Trials whose camera window is not fully covered are dropped from the session.

ii.
```python
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
...
valid = paper_mask & wheel_good & whisk_good
```

iii. Step 22/28 (as above).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything missing is dropped, at the appropriate granularity, and recorded.
- **Trial level:** NaN in any required trial event drops the trial (`paper_trial_mask`); a wheel or camera window that is absent, short at either edge, or contains any non-finite sample drops the trial; masks for all three streams are intersected so a retained trial has every input and output.
- **Session level:** a session with no matching whisker motion-energy/camera-times pair raises and is skipped; a session with fewer than two jointly valid trials raises and is skipped. 15 of 459 sessions were excluded (14 for missing whisker streams, 1 for <2 valid trials), and each is listed with its reason in `metadata['failed_sessions']`; a `.failed` marker file makes the exclusion persistent across reruns.
- **Infrastructure level:** any exception in a session is caught, its traceback printed, and the session skipped rather than aborting the run; the trials table is resolved to the newest revision to avoid obsolete ALF attributes; `spikes['clusters']` is copied because the mounted arrays are read-only memmaps and `merge_probes` mutates in place; intermediates are written to a `.tmp` file and `os.replace`d so a killed worker cannot leave a corrupt cache.

ii.
```python
except Exception as exc:
    import traceback; traceback.print_exc()
    print(f'SKIP {eid}: {type(exc).__name__}: {exc}', flush=True)
    failures.append((eid, repr(exc)))
    fail_file.write_text(f'{type(exc).__name__}: {exc}\n')
```
```python
tmp_file = cache_file.with_suffix('.pkl.tmp')
with open(tmp_file, 'wb') as f:
    pickle.dump(sess, f, protocol=pickle.HIGHEST_PROTOCOL)
os.replace(tmp_file, cache_file)
```
```python
spikes['clusters'] = np.array(spikes['clusters'], dtype=np.int32, copy=True)
```
```python
valid = trials[required].notna().all(axis=1).to_numpy(dtype=bool, copy=True)
```

iii. Step 77: "all for the same justified curation reason: no camera timestamp array matches an available whisker ROI motion-energy stream. Because whisker motion energy is a required decoder output, these sessions cannot be retained. This matches the reference caching script's behavior of skipping sessions when required behavior loading fails." Steps 40–43 document the read-only-memmap and read-only-pandas-mask fixes found from tracebacks; step 68 documents the parquet corruption caused by concurrent `one.load_cache(tag=...)` and the switch to the non-mutating `load_tables`; step 107 documents making the exclusions persistent. Step 108: "Reconciliation is exact: all 459 release sessions are represented by 444 successful intermediates and 15 persistent exclusion markers, with no missing, extra, or overlapping IDs."

## 10-a. What are the most time-consuming steps of the code?

i. In order of cost:
1. **Reading the spike sorting** (`load_spike_sorting` per probe): hundreds of MB of `spikes.times`/`spikes.clusters` per insertion, up to ~80 M spikes for a two-probe session. Pure I/O; this dominated the ~35 min wall clock of the 16-shard run.
2. **`bin_spiking_data`** — specifically the inherited `compute_spike_count`, which for *every* trial evaluates `idxs_t = (times >= t_beg) & (times < t_end)` over the *entire* session spike array. That is O(n_trials × n_spikes) ≈ 426 × 8×10⁷ ≈ 3.4×10¹⁰ comparisons per session, mitigated only by a 4-process pool per session.
3. **Serialisation**: writing 444 intermediate pickles (~50 GB) and then reading them all back and writing the single 53.2 GB `converted_data.pkl`. Assembly alone took several minutes and the verifier needed several more just to load the result.
4. `SessionLoader.load_wheel` (1 kHz interpolation + filtering of the whole session) and the 444-session global `np.concatenate` in `category_edges`.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```
```python
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
```
```python
with open(OUT, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent's response to cost was parallelism rather than algorithmic change: per-session resumable intermediates, an `IBL_CACHE_ONLY` mode and `IBL_SHARD_COUNT`/`IBL_SHARD_INDEX` sharding (step 60), scaled from 8 to 16 workers once it measured "throughput is roughly 20–25 sessions per minute" and 879 GiB of free memory (steps 64–66), concluding at step 73 that "throughput is I/O-bound, so further concurrency would provide little benefit."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three.
1. `trial_number_in_block` is a pure Python loop over every trial. It is exactly `trials.groupby((p != p.shift()).cumsum()).cumcount()` — one vectorised pandas expression (which is how the expert writes it). Cheap in absolute terms (~10³ iterations/session) but gratuitous.
2. The per-trial loop inside `interpolate` runs twice per session (wheel, whisker) and re-does `searchsorted` per trial. `np.searchsorted` is already vectorised over all onsets, and the whole resampling can be done with a single flattened query vector.
3. The biggest one is in the inherited `compute_spike_count`: the per-trial boolean mask over the full spike array. Replacing it with two vectorised `np.searchsorted` calls plus a single `np.bincount` on a flat `unit * n_bins + bin` index — the approach the expert uses — would turn O(n_trials × n_spikes) into O(n_spikes), removing the need for the nested 4-process pool entirely. The agent kept the repo implementation unchanged.

ii.
```python
for i, value in enumerate(prob):
    if i == 0 or value != previous:
        k = 1
    else:
        k += 1
```
```python
for i, onset in enumerate(trials.stimOn_times.to_numpy()):
    ib = np.searchsorted(times, beg, side='right')
    ie = np.searchsorted(times, end, side='left')
```
```python
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials, n_workers=4, **PARAMS)
```

iii. Not discussed. The agent's stated priority was fidelity to the reference implementation ("preserve the reference preprocessing exactly, adding only task-required category discretization", step 27), and it addressed runtime with sharding instead of rewriting the repo's binning.

## 10-c. What processing does the code repeat multiple times?

i.
- **Every session is serialised twice and read twice.** The shard pass writes ~50 GB of per-session intermediates; the assembly pass reads all 444 back and writes a second 53.2 GB copy. Both copies remain on disk.
- **`BrainRegions()` is constructed once per session** (444 times), reloading the Allen atlas tables each time, instead of once at module level as the expert does.
- **`category_edges` is run twice** over the whole dataset, each call `np.concatenate`-ing every retained sample of every session (≈1.9×10⁷ values) into one temporary array.
- **`np.isin(spike_clusters, reg_clu_ids)` inside `bin_spiking_data`** builds a full-length boolean mask even though `reg_clu_ids` is `arange(len(clusters))`, i.e. selects everything — a no-op pass over all spikes.
- The `interpolate` helper duplicates the window `searchsorted` work for the wheel and the whisker independently (unavoidable, different sample grids, but the trial-window bookkeeping is recomputed).

ii.
```python
with open(tmp_file, 'wb') as f:
    pickle.dump(sess, f, protocol=pickle.HIGHEST_PROTOCOL)
...
with open(cache_file, 'rb') as f: sess = pickle.load(f)
...
with open(OUT, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```
```python
beryl = BrainRegions().acronym2acronym(clusters.acronym.to_numpy(), mapping='Beryl')
```
```python
wheel_edges = category_edges(sessions, 'wheel')
whisk_edges = category_edges(sessions, 'whisk')
```

iii. The double serialisation is a deliberate trade for resumability and shard-safety: step 60, "Add shard/cache-only support so the many sessions can be processed in parallel into resumable intermediates without concurrent writes to the final pickle", and step 63, "The run is resumable, so no completed work will be lost." It paid off — the run survived two worker restarts and a corrupted cache table. The repeated `BrainRegions()` construction and the redundant `np.isin` are not discussed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Unfiltered units.** 85,846 `root` and 12,759 `void` units are binned, stored and shipped. `void` is the atlas label for a site the histology placed outside the brain, and the reference repo itself discards `root`/`void` before decoding (`3_decode_multi_region.py`). This is ~16% of the 599,865 stored units, and together with the absent `label >= 1` cut it is what turns a ~12 GB artifact into a 53.2 GB one.
- **`np.array(binned_list)`** materialises the whole session as a `float64` array of shape (n_trials, 100, n_clusters) — for a 3140-unit session that is ~10 GB of float64 — immediately before each trial is cast down to `int16`.
- **`int16` storage** produced 188,928 verifier warnings ("neural dtype is int16, expected float32. Will be converted during training"), i.e. the whole array is re-cast at training time anyway.
- **Per-trial constants tiled 100×.** `choice`, `prior` and `trial number in block` are each replicated across all 100 bins. This is what the target format asks for and the expert does the same, but it is a 100-fold redundancy the decoder collapses again.
- **Metadata bloat.** `source_trial_indices` is stored for all 188,925 retained trials, and `failed_sessions` carries full `repr` strings; the metadata block alone accounts for most of the 23 MB stats dump and is never used downstream.
- **`feedbackType` / `feedback_times` / `goCue_times`** are loaded and used only to build the trial mask; `duration <= 10.0` is computed for every trial although the criterion removes very few.
- **459 intermediate pickles (~50 GB)** are left in `/app/data/converted_sessions` after assembly.

ii.
```python
neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
```
```python
session_info.append(dict(eid=s['eid'], subject=s['subject'],
    n_trials=len(s['neural']), n_neurons=len(s['regions']),
    source_trial_indices=s['source_trial_idx'].tolist()))
```
```python
duration = trials.feedback_times.to_numpy() - trials.goCue_times.to_numpy()
valid &= duration <= 10.0
```
```python
out = np.vstack((np.full(T, s['choice'][j], np.int8),
                 np.full(T, s['prior'][j], np.int8), wb, mb))
```

iii. The agent's framing throughout was inclusiveness over economy: it read the instruction "Save the full converted dataset" and `stage_cache.sh` literally, reversed its own earlier 10-session decision at step 60 ("the current deterministic 10-session sample does not satisfy the requirement to save the full converted dataset"), and accepted the resulting size, reporting at step 112 only that "`/app/converted_data.pkl` is 53.17 GB and contains 444 valid sessions with 188,925 trials; no assembly errors occurred." It never revisited whether the unfiltered units were needed, and never noted that the same instruction set asks the conversion to match the papers' neuron curation.
