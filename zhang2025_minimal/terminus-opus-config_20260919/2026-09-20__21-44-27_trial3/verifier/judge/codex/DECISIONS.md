# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script rebuilds root ONE cache tables from the staged filesystem if needed, then reads a fixed session/probe manifest from `bwm_release.csv` rather than discovering sessions via `one.search`. It processes each unique `eid` in that freeze file, and for each session uses `SpikeSortingLoader` per probe plus `SessionLoader` for trials, wheel, and motion energy.

ii. ```python
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'
...
if not (Path(CACHE_DIR) / 'datasets.pqt').exists():
    rebuild_cache_tables()
...
bwm_df = pd.read_csv(FREEZE_FILE, index_col=0)
...
jobs = [(e, bwm_df[bwm_df.eid == e]) for e in eids]
```

```python
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
...
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
sess_loader.load_wheel()
sess_loader.load_motion_energy(views=[view])
```

iii. In the trajectory, the AI said the staged cache tables were stale relative to on-disk revisions, so it chose to rebuild the cache locally. It also said `bwm_release.csv` already contained the needed `eid`, `pid`, and `probe_name`, so `eid2pid` and session discovery were unnecessary.

## 1-b. How are the data split into subjects?

i. Subject identity comes from the `subject` column of `bwm_release.csv`. During assembly, the AI accumulates subject names in first-seen session order and stores `subject_idx` as the index of each session's subject in that list.

ii. ```python
meta = dict(eid=eid, subject=sub_df.subject.iloc[0], lab=sub_df.lab.iloc[0],
            date=str(sub_df.date.iloc[0]), n_probes=len(sub_df),
            motion_energy_view=me_view, ntrials=int(len(kidx)),
            ntrials_total=int(len(trials_df)), nneurons=int(nneurons))
```

```python
sub = str(meta['subject'])
if sub not in subjects:
    subjects.append(sub)
subject_idx.append(subjects.index(sub))
```

iii. The trajectory says the freeze CSV already contains subject metadata, so nothing had to be derived from paths or queried separately.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`, filtered to those present in the local ONE cache.

ii. ```python
available = set(one._cache['sessions'].index.astype(str))
eids = [e for e in bwm_df.eid.unique() if e in available]
eids.sort()
```

iii. The AI justified this by saying the brain-wide-map freeze contains 459 released `eid`s and those are the sessions intended by the reference repository.

## 1-d. How are the data split into trials?

i. Trials are taken from the session trials table loaded by `SessionLoader`; each row of `trials_df` is treated as one trial, and the kept trials are indexed by a boolean mask and `kidx`.

ii. ```python
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
...
kidx = np.flatnonzero(keep)
```

iii. The trajectory treated the trials table as the canonical source of trial boundaries, consistent with the IBL session format.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies `load_trials_and_mask(..., max_trial_len=10.0)` from the Zhang utilities, then further requires wheel coverage, whisker-motion-energy coverage, and non-NaN alignment times. This imports the Zhang/data-paper exclusions rather than the human reference's narrower custom mask.

ii. ```python
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
trials_mask = trials_mask.to_numpy()
...
keep = trials_mask & wheel_ok & me_ok & ~np.isnan(align)
if keep.sum() < 2:
    raise RuntimeError(f'only {keep.sum()} usable trials')
```

iii. In the trajectory, the AI explicitly said the key trial decision was `load_trials_and_mask(max_trial_len=10.0)` plus dropping trials whose behavioral traces did not span the decoding window, because that matched the Zhang pipeline it chose to follow.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from spike times and spike cluster assignments loaded per probe; cluster metadata are only used for region labels, not for computing the spike tensor itself.

ii. ```python
sp, cl, ch = ssl.load_spike_sorting()
...
spikes, clusters = merge_probes(spikes_list, clusters_list)
...
st = spikes['times']
sc = spikes['clusters']
```

iii. The trajectory repeatedly described the neural stream as coming from the released spike sorting and said the conversion should use all Kilosort clusters from each session.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, sorts spikes by time, bins spikes into 20 ms bins for each kept trial, and stores the result as per-bin spike counts. Unlike the human reference, it does not divide by bin width to convert counts into Hz.

ii. ```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
...
order = np.argsort(st, kind='stable')
st, sc = st[order], sc[order]
...
binned = np.zeros((len(kidx), nneurons, NBINS), dtype=np.float32)
for j, k in enumerate(kidx):
    ...
    b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
    np.clip(b, 0, NBINS - 1, out=b)
    np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)
```

iii. The trajectory says the AI followed the Zhang caching pipeline with merged probes and 20 ms bins, and it repeatedly referred to "spike counts" rather than rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI intentionally keeps all Kilosort clusters and only drops clusters that never emit a finite spike in the session, because the remapping is built from `np.unique(sc)` after spike loading. It does not apply the human reference's good-unit (`label >= 1`) or `void`-region filters.

ii. ```python
# clusters that fire at least once, as in get_spike_data_per_interval
used = np.unique(sc)
remap = np.full(len(clusters), -1, dtype=np.int64)
remap[used] = np.arange(len(used))
nneurons = len(used)
```

iii. In the trajectory and module docstring, the AI justified this by citing the Zhang methods paper line that they bin spike counts using all neurons, and by noting that `load_spiking_data` is called with `qc=None` in the reference repository.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Trials are aligned to `stimOn_times`. For each trial, the script constructs `[stimOn-0.5, stimOn+1.5]` and bins spikes relative to the start of that window, so the bins are locked to stimulus onset.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align = trials_df[ALIGN_TIME].to_numpy(dtype=float)
beg = align + TIME_WINDOW[0]
end = align + TIME_WINDOW[1]
...
b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
```

iii. The trajectory says the main alignment choice was exactly the Zhang setting `stimOn_times` with window `(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms over a 2 s window, yielding 100 bins. No additional temporal rebinning or smoothing is applied after binning.

ii. ```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

iii. The AI cited the Zhang `params` dictionary and methods-paper description of 2 s trials split into 20 ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` plus a fixed trial-relative bin grid. The stored values are not taken from another raw variable besides the stimulus-onset timestamps.

ii. ```python
ALIGN_TIME = 'stimOn_times'
...
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
...
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. The trajectory says the entire conversion should use stimulus-onset alignment with the Zhang bin grid, so this input is just that grid expressed relative to the event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI does not infer this from a measured time series; it defines a fixed vector of 100 bin times and copies it into every trial. The chosen values are bin right edges from `-0.48` to `1.5`, not the bin centers used by the human reference.

ii. ```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
...
inputs = np.empty((len(kidx), 2, NBINS), dtype=np.float32)
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. The trajectory ties this to the Zhang behavior-alignment code, which the AI interpreted as interpolating onto the right-edge bin grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same 100-bin trial grid is used for both neural binning and time input, but the input records the right edge of each neural bin rather than its center.

ii. ```python
b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
...
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. The AI's trajectory says it wanted the inputs and behaviors on the same Zhang bin grid as the spikes.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`, with block boundaries detected whenever `probabilityLeft` changes between adjacent trials.

ii. ```python
def trial_in_block(pleft):
    out = np.zeros(len(pleft), dtype=np.int64)
    count = 0
    for i in range(len(pleft)):
        if i > 0 and pleft[i] != pleft[i - 1]:
            count = 0
```

iii. The trajectory says blocks were inferred from `probabilityLeft` because the trials table does not provide a separate block counter.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a 0-based count of position within each block on the full unfiltered trial sequence, then subselects the kept trials so dropped trials still advance the counter.

ii. ```python
def trial_in_block(pleft):
    ...
    out[i] = count
    count += 1
    return out
...
tib = trial_in_block(trials_df['probabilityLeft'].to_numpy())[kidx]
inputs[:, 1, :] = tib[:, None]
```

iii. The trajectory explicitly mentioned checking block structure and said the block count should follow the original session sequence.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the `choice` column of the trials table.

ii. ```python
choice = trials_df['choice'].to_numpy()[kidx]
choice_out = (choice < 0).astype(np.int64)             # left 0, right 1
```

iii. The trajectory says the AI empirically verified the IBL sign convention and concluded `choice == +1` means left and `choice == -1` means right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps left to `0` and right to `1` by testing whether the raw choice is negative, and it relies on the earlier trial mask to have removed no-response trials.

ii. ```python
choice = trials_df['choice'].to_numpy()[kidx]
choice_out = (choice < 0).astype(np.int64)             # left 0, right 1
...
outputs[:, 0, :] = choice_out[:, None]
```

iii. The trajectory says the sign convention was checked against task variables before the final implementation.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. ```python
pleft = trials_df['probabilityLeft'].to_numpy()[kidx]
prior_out = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
```

iii. The trajectory says the AI confirmed the block priors were exactly `{0.2, 0.5, 0.8}` in the staged data.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI discretely recodes `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, and throws an error if another value appears.

ii. ```python
prior_out = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
if np.any(prior_out < 0):
    raise RuntimeError('unexpected probabilityLeft value')
```

iii. The trajectory justified this as directly matching the task specification and the observed raw values.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel timestamps and the wheel velocity generated by `SessionLoader.load_wheel`, then converted to absolute value.

ii. ```python
sess_loader.load_wheel()
wheel_speed, wheel_ok = interp_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), beg, end)
```

iii. The trajectory says the AI followed the Zhang target-behavior logic and used `abs(velocity)` from the session loader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI uses the loader's velocity trace, takes its absolute value, linearly interpolates it onto each trial's 100-bin grid, rejects trials whose wheel trace does not cover the window, and then discretizes the kept session trace into three equal-occupancy bins.

ii. ```python
wheel_speed, wheel_ok = interp_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), beg, end)
...
wheel_bin = discretize(wheel_speed[kidx])
```

```python
def discretize(values, nbins=NBEHBINS):
    edges = np.nanquantile(values, np.arange(1, nbins) / nbins)
    edges = np.unique(edges)
    return np.digitize(values, edges).astype(np.int64)
```

iii. The trajectory says the AI chose this because wheel speed and whisker motion energy are not comparable across sessions, so sessionwise tertiles keep all three classes populated.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded by computing within-session tertile cut points with `np.nanquantile` and assigning bins with `np.digitize`.

ii. ```python
def discretize(values, nbins=NBEHBINS):
    edges = np.nanquantile(values, np.arange(1, nbins) / nbins)
    edges = np.unique(edges)
    return np.digitize(values, edges).astype(np.int64)
...
wheel_bin = discretize(wheel_speed[kidx])
```

iii. The trajectory explicitly justified tertiles as a way to avoid cross-session scale mismatches.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is interpolated onto the same per-trial 100-bin grid used for neural data, with samples placed at the bin right edges defined by `BIN_TIMES`.

ii. ```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
...
x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. The trajectory says the AI wanted behavior and spikes on the same Zhang alignment grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from the `whiskerMotionEnergy` column and timestamps of the left camera when available, otherwise the right camera.

ii. ```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        v, ok = interp_behavior(df['times'].to_numpy(),
                                df['whiskerMotionEnergy'].to_numpy(), beg, end)
```

iii. The trajectory says the AI followed the reference behavior-loading logic and preferred the left camera with right as fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy trace as-is, interpolates it onto each trial grid, rejects trials without full coverage, and discretizes the kept session trace into tertiles.

ii. ```python
v, ok = interp_behavior(df['times'].to_numpy(),
                        df['whiskerMotionEnergy'].to_numpy(), beg, end)
...
me_bin = discretize(me_vals[kidx])
```

iii. The trajectory says this matched `load_target_behavior` and `get_behavior_per_interval` in the Zhang utilities.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: within-session tertiles computed with `np.nanquantile`, then assigned with `np.digitize`.

ii. ```python
def discretize(values, nbins=NBEHBINS):
    edges = np.nanquantile(values, np.arange(1, nbins) / nbins)
    edges = np.unique(edges)
    return np.digitize(values, edges).astype(np.int64)
...
me_bin = discretize(me_vals[kidx])
```

iii. The trajectory gives the same session-scale justification as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated onto the same per-trial 100-bin grid used for the spikes, again at the bin right edges.

ii. ```python
x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. The trajectory says the AI wanted the continuous behavior streams evaluated on the same Zhang time grid as the neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or inconsistent data mainly by dropping unusable trials or whole sessions. It skips sessions with no spike sorting, no whisker motion energy, fewer than two usable trials, or other loader failures; it also rebuilds cache tables when the shipped cache is stale.

ii. ```python
if len(spikes_list) == 0:
    raise RuntimeError('no spike sorting')
...
if me_vals is None:
    raise RuntimeError('no whisker motion energy')
...
keep = trials_mask & wheel_ok & me_ok & ~np.isnan(align)
if keep.sum() < 2:
    raise RuntimeError(f'only {keep.sum()} usable trials')
```

```python
if not (Path(CACHE_DIR) / 'datasets.pqt').exists():
    rebuild_cache_tables()
```

iii. The trajectory says the cache rebuild was necessary because offline cache tables no longer matched the on-disk revisions, and otherwise the AI consistently chose to skip unusable data rather than repair it.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is session processing: loading spike sorting from disk, sorting/binning large spike arrays, and then writing and later reloading large per-session `.npz` intermediates before building the final pickle.

ii. ```python
sp, cl, ch = ssl.load_spike_sorting()
...
order = np.argsort(st, kind='stable')
...
np.savez(f, neural=binned, input=inputs, output=outputs, regions=beryl)
...
z = np.load(TMP_DIR / f'{eid}.npz', allow_pickle=True)
```

iii. In the trajectory, the AI explicitly identified spike loading as heavy, estimated full-dataset size repeatedly, and later discussed the cost of assembling a very large pickle after per-session processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious Python loops remain: trial-by-trial interpolation in `interp_behavior` and trial-by-trial spike binning in `process_session`. The session-assembly loops over neurons, sessions, and subjects are also straightforward but not vectorized.

ii. ```python
for k in range(ntrials):
    ...
    values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

```python
for j, k in enumerate(kidx):
    ...
    np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)
```

iii. The trajectory says the AI first prioritized reproducing the Zhang pipeline and then scaling it with multiprocessing rather than rewriting these inner loops more aggressively.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several I/O-heavy steps: every worker constructs its own ONE client, each session is written to a temporary `.npz`, and the same session arrays are then read back during assembly. The left-camera load is also attempted before possibly retrying with the right camera.

ii. ```python
def get_one():
    return ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
               cache_dir=CACHE_DIR, mode='local')
...
one = get_one()
```

```python
np.savez(f, neural=binned, input=inputs, output=outputs, regions=beryl)
...
z = np.load(TMP_DIR / f'{eid}.npz', allow_pickle=True)
```

iii. The trajectory did not explicitly call this out as a design decision; this repetition is mainly inferred from the final implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest extra work is serializing every processed session to `.npz` and immediately reloading it only to build the final pickle. The cache-table rebuild is also extra work relative to the logical conversion itself, though it was used as a loader workaround.

ii. ```python
f = TMP_DIR / f'{eid}.npz'
np.savez(f, neural=binned, input=inputs, output=outputs, regions=beryl)
...
z = np.load(TMP_DIR / f'{eid}.npz', allow_pickle=True)
```

```python
if not (Path(CACHE_DIR) / 'datasets.pqt').exists():
    rebuild_cache_tables()
```

iii. The trajectory explicitly justified the cache rebuild as necessary for stale cache metadata, but it did not give a separate justification for the write-then-read intermediate storage pattern.
