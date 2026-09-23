# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use `one.search(...)` to define the session set. Instead it read the release/freeze CSV `bwm_release.csv`, optionally restricted it with `DATALIMIT_SUBSET.csv`, grouped rows by `eid`, and then loaded each session's trials, wheel, motion energy, and probe spike sorting through `SessionLoader` and `SpikeSortingLoader`.

ii. ```python
CACHE_DIR = '/app/data/one_cache'
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'
```

```python
bwm = pd.read_csv(FREEZE_FILE, index_col=0)
...
if os.path.exists(limit_csv):
    sub = pd.read_csv(limit_csv)
    col = 'eid' if 'eid' in sub.columns else sub.columns[0]
    keep = set(sub[col].astype(str))
    bwm = bwm[bwm.eid.astype(str).isin(keep)]
```

```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
sl.load_wheel()
...
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as iterating over the 459-session freeze in `bwm_release.csv`, while still using ONE loaders for the actual payloads. It also noted that the local ONE client had to stay offline and that `DATALIMIT_SUBSET.csv` should restrict the session list when present.

## 1-b. How are the data split into subjects (mice)?

i. The AI took the subject label directly from `bwm_release.csv`. After grouping rows by `eid`, it stored `g.subject.iloc[0]` with each session job, and later built `subjects` and `subject_idx` from those stored subject strings.

ii. ```python
for eid, g in bwm.groupby('eid', sort=False):
    jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
```

```python
subjects = sorted(set(s['subject'] for s in sessions))
subj_lut = dict((s, i) for i, s in enumerate(subjects))
...
'subject_idx': np.array([subj_lut[s['subject']] for s in sessions],
                        dtype=np.int64),
```

iii. The notes say the freeze file lists 139 subjects and that `subject` from the freeze is the source field mapped into the output structure.

## 1-c. How are the data split into sessions?

i. The AI treated each unique `eid` as one session. It grouped the freeze file by `eid`, assembled one job per `eid`, and each job was processed into one session record.

ii. ```python
for eid, g in bwm.groupby('eid', sort=False):
    jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
```

```python
session = {'eid': eid, 'subject': subject, 'neural': neural,
           'input': inputs, 'output': outputs, 'regions': reg,
           'trial_idx': idx,
           'raw': {'wheel_vals': wheel_k, 'me_vals': me_k,
                   'stim_on': stim_on[idx]}}
```

iii. The AI's notes describe the release as 459 `eid`s and explicitly say all 459 release `eid`s were attempted, so `eid` was its session unit.

## 1-d. How are the data split into trials?

i. The AI used the IBL trials table as the trial definition. It loaded `sl.trials` through `SessionLoader`, built trial windows from `stimOn_times`, and then kept selected trial indices in `idx`.

ii. ```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials, mask = load_trials_and_mask(sl)
```

```python
stim_on = trials[PARAMS['align_time']].to_numpy(dtype=float)
begs_all = stim_on + WIN[0]
ends_all = stim_on + WIN[1]
...
idx = np.where(valid)[0]
```

iii. The notes say the trials table is the source for `stimOn_times`, `choice`, `probabilityLeft`, and the other per-trial variables, and that the processing follows the reference `load_trials_and_mask`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applied a compound mask. First it used the reference-style trial mask: reaction time between 0.08 and 2.0 s, `feedback_times - goCue_times <= 10 s`, required non-NaN trial fields, and `choice != 0`. Then it required wheel and whisker traces to cover the trial window, required `probabilityLeft` to be one of `0.2/0.5/0.8`, and finally dropped trials outside the spike-time span or with all-zero neural data.

ii. ```python
query = '(firstMovement_times - stimOn_times < %s)' % MIN_RT
query += ' | (firstMovement_times - stimOn_times > %s)' % MAX_RT
query += ' | (feedback_times - goCue_times > %s)' % MAX_TRIAL_LEN
for event in NAN_EXCLUDE:
    query += ' | %s.isnull()' % event
query += ' | (choice == 0)'
mask = ~sess_loader.trials.eval(query)
```

```python
wheel_vals, wheel_ok = interpolate_behavior(wheel_t, wheel_v, begs_all, ends_all)
me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
valid = mask & wheel_ok & me_ok
...
valid &= (np.isclose(pl, 0.2) | np.isclose(pl, 0.5) | np.isclose(pl, 0.8))
```

```python
in_span = (begs_all >= st[0]) & (ends_all <= st[-1])
valid &= in_span
...
has_spikes = binned.sum(axis=(1, 2)) > 0
```

iii. The notes justify the first part as matching `load_trials_and_mask(..., max_trial_len=10.0)` and the behavior coverage checks in the reference code. The extra neural-coverage filter was justified later as removing invalid recording gaps that produced decoder warnings from all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from `spikes['times']` and `spikes['clusters']` loaded per probe, with `clusters['label']` and `clusters['acronym']` used to decide which units to keep and how to label them anatomically.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
...
cl = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

```python
st = np.asarray(spikes['times'])
sc = np.asarray(spikes['clusters'])
good = cl['label'].to_numpy(dtype=float) >= 1.0
beryl = np.asarray(br.acronym2acronym(cl['acronym'].to_numpy(),
                                      mapping='Beryl'))
```

iii. The notes say the conversion follows the reference spike loaders, then uses data-paper neuron curation fields (`label`, Beryl acronym) to filter units.

## 2-b. How is the `neural` data processed?

i. The AI merged probes within a session, renumbered surviving units across probes, sorted spikes by time, and binned spikes into 20 ms bins over each trial window. It kept the final neural arrays as binned spike counts in shape `(n_units, 100)` and did not convert them to Hz.

ii. ```python
times_l.append(st[sel])
clu_l.append(new[sel])
...
order = np.argsort(st, kind='stable')
return st[order], sc[order], reg, n_all, n_good
```

```python
t = spike_times[a:b] - begs[k]
bi = (t / BINSIZE).astype(np.int64)
np.clip(bi, 0, NBINS - 1, out=bi)
flat = spike_clusters[a:b] * NBINS + bi
counts = np.bincount(flat, minlength=n_units * NBINS)
out[k] = counts.reshape(n_units, NBINS)
```

iii. The notes explicitly say, "Neural: raw spike counts per bin," and justify the 20 ms stimulus-aligned binning as matching the reference code's trial structure. The AI also highlighted that standardization is deferred to the decoder, not done in the cache.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI kept only units with `label >= 1.0`, mapped unit acronyms to Beryl regions, dropped units in `root` or `void`, then kept only regions with at least 5 surviving units in that session, and after all sessions were processed applied an additional cross-session filter keeping only regions seen in at least 2 sessions.

ii. ```python
good = cl['label'].to_numpy(dtype=float) >= 1.0
beryl = np.asarray(br.acronym2acronym(cl['acronym'].to_numpy(),
                                      mapping='Beryl'))
keep = good & ~np.isin(beryl, EXCLUDE_REGIONS)
```

```python
counts = Counter(reg.tolist())
keep_unit = np.array([counts[r] >= MIN_UNITS_PER_REGION for r in reg])
...
reg = reg[keep_unit]
```

```python
region_session_count = Counter()
for s in sessions:
    for r in set(s['regions'].tolist()):
        region_session_count[r] += 1
...
keep_regions = set(r for r, c in region_session_count.items()
                   if c >= min_sess)
```

iii. The notes justify this as following the BWM data paper's curation: well-isolated units only, grey-matter regions only, at least 5 units per region per session, and regions present in at least 2 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligned neural data to stimulus onset. For each trial it formed the window `stimOn_times + (-0.5, 1.5)` and binned spikes within that window, so all neural trials are expressed relative to stimulus onset.

ii. ```python
PARAMS = {'interval_len': 2.0, 'binsize': 0.02,
          'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}
```

```python
stim_on = trials[PARAMS['align_time']].to_numpy(dtype=float)
begs_all = stim_on + WIN[0]
ends_all = stim_on + WIN[1]
```

```python
t = spike_times[a:b] - begs[k]
```

iii. The notes say the conversion uses the reference unified grid: `stimOn_times`, window `(-0.5, +1.5)` s, shared by all neural and behavioral streams because the decoder task requires stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI used 20 ms bins and 100 bins per 2 s trial. It did not rebin or smooth neural data beyond the initial spike binning.

ii. ```python
BINSIZE = PARAMS['binsize']
WIN = PARAMS['time_window']
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))   # 100
```

```python
bi = (t / BINSIZE).astype(np.int64)
```

iii. The notes justify this as copied directly from `0_data_caching.py` and consistent with the decoder task and the papers' 20 ms, 2 s trial description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI treated this input as the shared analysis time grid relative to the alignment event, not as a raw sensor trace. It depended on the reference alignment variable `stimOn_times` to define the trial windows, then used the fixed per-bin relative times `BIN_TIMES`.

ii. ```python
PARAMS = {'interval_len': 2.0, 'binsize': 0.02,
          'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}
...
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
```

```python
stim_on = trials[PARAMS['align_time']].to_numpy(dtype=float)
```

iii. The notes say this is a new decoder input required by the task, built on the same stimulus-aligned grid as the rest of the session.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computed a single fixed vector of 100 relative times from `-0.48` to `1.50` s, corresponding to the right edge of each 20 ms bin, and copied that row into every trial's input matrix.

ii. ```python
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
```

```python
time_row = BIN_TIMES.astype(np.float32)
...
inputs.append(np.stack([time_row,
                        np.full(NBINS, tib[k], dtype=np.float32)]))
```

iii. The notes justify the right-edge convention by pointing to the reference behavior interpolation grid `np.linspace(t_beg + binsize, t_end, n_bins)`, and argue that using the same time grid avoids temporal offsets between streams.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI aligned the time input to the same per-trial bin grid used for neural and behavior. Neural bins span the same 20 ms windows, while `BIN_TIMES` stores the corresponding right-edge time for each bin.

ii. ```python
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
```

```python
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
```

```python
inputs.append(np.stack([time_row,
                        np.full(NBINS, tib[k], dtype=np.float32)]))
```

iii. The notes include an explicit alignment proof: behavior samples are taken at `stim_on + BIN_TIMES`, and spike bin `i` ends at `BIN_TIMES[i]`, so the AI considered all streams time-aligned bin by bin.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. The AI derived this input from `trials['probabilityLeft']`. A change in `probabilityLeft` was treated as a block boundary.

ii. ```python
def trial_number_in_block(prob_left):
    ...
    new_block[1:] = p[1:] != p[:-1]
```

iii. The notes state that `probabilityLeft` is the only raw field carrying block structure, so the block identity had to be reconstructed from that column.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computed a 0-based count within each contiguous block of constant `probabilityLeft`, using the full trials table before masking, then broadcast that scalar across all 100 time bins of each kept trial.

ii. ```python
starts = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
return np.arange(len(p)) - starts
```

```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
tib = tib_all[idx].astype(np.float32)
...
np.full(NBINS, tib[k], dtype=np.float32)
```

iii. The notes explicitly justify computing it before trial filtering so excluded trials still advance the within-block counter and the value reflects the animal's real block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The AI derived choice from `trials['choice']`.

ii. ```python
choice_raw = trials['choice'].to_numpy()[idx]
```

iii. The notes document the IBL convention as `+1 = left`, `-1 = right`, `0 = no-go`, and say that this raw field is the source for the decoder's choice output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI dropped no-response trials via the trial mask, recoded left to `0` and right to `1` using `(choice_raw < 0)`, and broadcast that scalar across all 100 time bins for each retained trial.

ii. ```python
query += ' | (choice == 0)'
```

```python
# IBL convention (verified empirically): +1 = reported LEFT, -1 = RIGHT
choice = (choice_raw < 0).astype(np.int64)
```

```python
np.full(NBINS, choice[k], dtype=np.int64)
```

iii. The notes justify the remapping as required by the decoder task (`left = 0`, `right = 1`) and say the left/right semantics were empirically checked.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The AI derived prior from `trials['probabilityLeft']`.

ii. ```python
pl = trials['probabilityLeft'].to_numpy()
```

iii. The notes identify `probabilityLeft` as the raw trial-table field carrying the block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI filtered trials to the three expected values `0.2`, `0.5`, `0.8`, recoded them to class labels `0`, `1`, `2`, and broadcast the class over all 100 time bins in a kept trial.

ii. ```python
valid &= (np.isclose(pl, 0.2) | np.isclose(pl, 0.5) | np.isclose(pl, 0.8))
```

```python
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2]).astype(np.int64)
```

```python
np.full(NBINS, prior[k], dtype=np.int64)
```

iii. The notes justify this directly from the decoder task, which requires `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The AI derived wheel speed from `SessionLoader`'s wheel trace, specifically `sl.wheel['velocity']`, which itself comes from the raw wheel position/timestamp data processed by `SessionLoader`.

ii. ```python
sl.load_wheel()
wheel_t = sl.wheel['times'].to_numpy()
wheel_v = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. The notes explicitly map wheel speed to `abs(SessionLoader.wheel['velocity'])` and say this follows the reference behavior-loading logic.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI took the absolute value of the wheel velocity, linearly interpolated it onto the shared 100-bin trial grid, then later discretized the interpolated values into three categories.

ii. ```python
wheel_v = np.abs(sl.wheel['velocity'].to_numpy())
```

```python
wheel_vals, wheel_ok = interpolate_behavior(wheel_t, wheel_v,
                                            begs_all, ends_all)
```

```python
f = interp1d(tt, tv, kind='linear', fill_value='extrapolate',
             bounds_error=False, assume_sorted=True)
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
vals = f(grid)
```

iii. The notes justify this as matching the reference `get_behavior_per_interval` interpolation pattern and the reference source definition `wheel-speed = abs(velocity)`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI used per-session terciles. It computed the 1/3 and 2/3 quantiles over the full `(n_trials, n_bins)` wheel-speed array for a session, digitized values into `0/1/2`, and used a rank-based fallback if the quantiles were degenerate.

ii. ```python
def discretize_terciles(values):
    q1, q2 = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 == q2:
        flat = values.ravel()
        ranks = np.argsort(np.argsort(flat))
        d = (ranks * 3 // max(len(flat), 1)).reshape(values.shape)
        return np.clip(d, 0, 2).astype(np.int64), (float('nan'), float('nan'))
    return np.digitize(values, [q1, q2]).astype(np.int64), (float(q1), float(q2))
```

```python
wheel_d, wheel_thr = discretize_terciles(wheel_k)
```

iii. The notes justify per-session terciles because the decoder requires categorical outputs and because session-specific thresholds avoid encoding session identity from camera or behavioral scale differences; they also give balanced classes by construction.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI sampled wheel speed on the same stimulus-aligned grid as the neural data. It interpolated the wheel trace at the right edge of each 20 ms bin in the per-trial window.

ii. ```python
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
```

```python
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
vals = f(grid)
```

iii. The notes argue that this exactly matches the reference behavior grid and therefore keeps behavior and spikes synchronized with no temporal offset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The AI derived whisker motion energy from `sl.motion_energy[<view>Camera]['whiskerMotionEnergy']` and the corresponding camera timestamps, preferring the left camera and falling back to the right camera.

ii. ```python
for view in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[view])
        key = view + 'Camera'
        if key in sl.motion_energy and \
                'whiskerMotionEnergy' in sl.motion_energy[key]:
            me_t = sl.motion_energy[key]['times'].to_numpy()
            me_v = sl.motion_energy[key]['whiskerMotionEnergy'].to_numpy()
            me_view = view
            break
```

iii. The notes justify this as following the reference behavior-loading logic: left camera preferred, right as fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI used the released whisker-motion-energy trace as-is, linearly interpolated it onto the same 100-bin trial grid, and later discretized the interpolated values into three categories.

ii. ```python
me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
```

```python
f = interp1d(tt, tv, kind='linear', fill_value='extrapolate',
             bounds_error=False, assume_sorted=True)
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
vals = f(grid)
```

iii. The notes say whisker motion energy should be taken from the camera trace, aligned with the shared grid, and discretized only because the decoder requires categorical outputs.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI used the same per-session tercile procedure as for wheel speed, with a quantile-based split and a rank-based fallback if needed.

ii. ```python
me_d, me_thr = discretize_terciles(me_k)
```

```python
return np.digitize(values, [q1, q2]).astype(np.int64), (float(q1), float(q2))
```

iii. The notes justify this with the same argument as wheel speed: the decoder needs categories, and per-session thresholds avoid unwanted session-identity confounds while balancing the classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI aligned whisker motion energy to the same stimulus-aligned trial grid as the neural data, sampling the trace at the right edge of each 20 ms neural bin.

ii. ```python
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
```

```python
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
vals = f(grid)
```

iii. The notes explicitly state that behavior sample `i` and spike bin `i` end at the same instant, which is the AI's alignment rationale here too.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handled bad or missing data by filtering it away. It skipped sessions with missing whisker motion energy or no usable units, rejected trials with missing behavioral coverage, invalid `probabilityLeft`, or empty neural windows, filtered non-finite spike times and out-of-range cluster ids, and dropped sessions left with fewer than two trials.

ii. ```python
if me_t is None:
    return None, dict(info, skip='no whisker motion energy')
```

```python
finite = np.isfinite(tt) & np.isfinite(tv)
tt, tv = tt[finite], tv[finite]
...
good &= np.all(np.isfinite(vals), axis=1)
```

```python
ok = np.isfinite(st) & (sc >= 0) & (sc < len(lut))
st, sc = st[ok], sc[ok]
```

```python
if len(idx) < MIN_TRIALS:
    return None, dict(info, skip='too few trials with spikes')
```

iii. The notes justify these filters as necessary to match the reference behavior-coverage logic, avoid invalid all-zero neural trials, and satisfy the target-format requirement that each session contain at least two trials.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified loading spike sorting from disk as the dominant expensive step, and also treated per-session spike binning and behavior interpolation as the main computational hotspots it optimized.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

```python
binned = bin_spikes(st, sc, n_units, begs_all[idx], ends_all[idx])
wheel_vals, wheel_ok = interpolate_behavior(wheel_t, wheel_v, begs_all, ends_all)
```

iii. The notes say the biggest savings came from vectorized `np.bincount` binning and one `interp1d` per session, but still frame I/O from spike loading as the fundamental heavy step.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI explicitly criticized the reference code's per-trial spike-binning and behavior-resampling work and replaced much of it with session-level vectorization. In its own code, the remaining obvious per-trial loops are the `for k in range(n_trials)` loop in `bin_spikes` and the final `for k in range(n_trials)` loop that assembles Python trial lists.

ii. ```python
for k in range(n_trials):
    a, b = i0[k], i1[k]
    ...
    counts = np.bincount(flat, minlength=n_units * NBINS)
    out[k] = counts.reshape(n_units, NBINS)
```

```python
for k in range(n_trials):
    neural.append(np.ascontiguousarray(binned[k]))
    inputs.append(np.stack([time_row,
                            np.full(NBINS, tib[k], dtype=np.float32)]))
    outputs.append(np.stack([
        np.full(NBINS, choice[k], dtype=np.int64),
        np.full(NBINS, prior[k], dtype=np.int64),
        wheel_d[k], me_d[k]]))
```

iii. The notes justify the current structure as already much faster than the reference because vectorization was moved to the session level, even though these smaller per-trial loops remain.

## 10-c. What processing does the code repeat multiple times?

i. The AI did not describe any major repeated processing as a problem. In the actual code, only small repetitions remain, such as trying left then right camera views and rebuilding per-trial arrays in Python after session-level computations are already finished.

ii. ```python
for view in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[view])
        ...
        break
```

```python
for k in range(n_trials):
    neural.append(np.ascontiguousarray(binned[k]))
    ...
```

iii. No explicit separate justification was given beyond the notes' broader claim that the expensive repeated work from the reference had already been reduced substantially.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is that the AI bins and stores all session units before the later cross-session `>= 2`-session region filter, then slices some units back out. It also keeps diagnostic fields like `trial_idx` and `raw` in intermediate session dictionaries even though they are not written to the final pickle.

ii. ```python
session = {'eid': eid, 'subject': subject, 'neural': neural,
           'input': inputs, 'output': outputs, 'regions': reg,
           'trial_idx': idx,
           'raw': {'wheel_vals': wheel_k, 'me_vals': me_k,
                   'stim_on': stim_on[idx]}}
```

```python
keep_regions = set(r for r, c in region_session_count.items()
                   if c >= min_sess)
...
if not km.all():
    s['neural'] = [np.ascontiguousarray(n[km]) for n in s['neural']]
    s['regions'] = s['regions'][km]
```

iii. The notes justify the region filter itself from the data paper, but they also acknowledge it is a post-hoc cross-session filter applied after session-level processing, so some already-binned units are discarded later.
