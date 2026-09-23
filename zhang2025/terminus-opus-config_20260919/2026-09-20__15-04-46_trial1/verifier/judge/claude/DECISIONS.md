# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the freeze file `bwm_release.csv` (from the reference code directory) to get the list of all probe insertions (699 insertions, 459 eids, 139 subjects). It groups by `eid` to get the list of sessions and their associated probes and subjects. If a `DATALIMIT_SUBSET.csv` exists in the data directory, it restricts to only those eids. It then uses the ONE API and `SessionLoader`/`SpikeSortingLoader` to load the actual data for each session.

ii.
```python
bwm = pd.read_csv(FREEZE_FILE, index_col=0)
# ...
limit_csv = '/app/data/DATALIMIT_SUBSET.csv'
if os.path.exists(limit_csv):
    sub = pd.read_csv(limit_csv)
    col = 'eid' if 'eid' in sub.columns else sub.columns[0]
    keep = set(sub[col].astype(str))
    bwm = bwm[bwm.eid.astype(str).isin(keep)]
# ...
jobs = []
for eid, g in bwm.groupby('eid', sort=False):
    jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
```

iii. The AI chose to use the freeze file (`bwm_release.csv`) as the authoritative list of sessions and probes, rather than using `one.search` with dataset filters. The freeze file is the exact session list the reference code uses, which ensures consistency with the methods paper.

## 1-b. How are the data split into subjects?

i. Subjects are obtained from the `subject` column of `bwm_release.csv`. Each job tuple includes the subject name, and after processing, sessions are grouped by unique subject names.

ii.
```python
jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
# ...
subjects = sorted(set(s['subject'] for s in sessions))
subj_lut = dict((s, i) for i, s in enumerate(subjects))
```

iii. Subjects come directly from the freeze file metadata, no parsing needed.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in the freeze file. Each eid corresponds to one session. The groupby on eid naturally gives one job per session.

ii.
```python
for eid, g in bwm.groupby('eid', sort=False):
    jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
```

iii. The freeze file already organizes data by session (eid), so no additional splitting is needed.

## 1-d. How are the data split into trials?

i. Trials come from the trials table loaded by `SessionLoader.load_trials()`, which has one row per trial.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials, mask = load_trials_and_mask(sl)
```

iii. The trials table is already one row per trial; no splitting logic is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI implements the reference code's `load_trials_and_mask` function with parameters: `min_rt=0.08`, `max_rt=2.0`, `max_trial_len=10.0`. Trials are excluded if: any of stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, or feedbackType is NaN; reaction time outside 0.08-2.0s; feedback_times - goCue_times > 10s; choice == 0 (no-go). Additional filters: wheel and whisker traces must cover the trial window; probabilityLeft must be one of {0.2, 0.5, 0.8}; the trial window must lie inside the span of the spike sorting; and the binned neural matrix must contain at least one spike.

ii.
```python
def load_trials_and_mask(sess_loader):
    query = '(firstMovement_times - stimOn_times < %s)' % MIN_RT
    query += ' | (firstMovement_times - stimOn_times > %s)' % MAX_RT
    query += ' | (feedback_times - goCue_times > %s)' % MAX_TRIAL_LEN
    for event in NAN_EXCLUDE:
        query += ' | %s.isnull()' % event
    query += ' | (choice == 0)'
    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask.to_numpy()
```

```python
# Additional behavioral coverage
wheel_vals, wheel_ok = interpolate_behavior(wheel_t, wheel_v, begs_all, ends_all)
me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
valid = mask & wheel_ok & me_ok
# prior check
valid &= (np.isclose(pl, 0.2) | np.isclose(pl, 0.5) | np.isclose(pl, 0.8))
# neural coverage
in_span = (begs_all >= st[0]) & (ends_all <= st[-1])
valid &= in_span
# zero-spike trial removal
has_spikes = binned.sum(axis=(1, 2)) > 0
```

iii. The AI documents this as matching the reference code's `load_trials_and_mask` with `max_trial_len=10.0`. The neural coverage and zero-spike checks were added after finding 16 all-zero trials in the initial conversion (documented in CONVERSION_NOTES Step 10).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from each probe, loaded via `SpikeSortingLoader.load_spike_sorting()`. The cluster table provides the quality label and anatomical region.

ii.
```python
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
cl = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. These are the standard spike sorting outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window [-0.5, 1.5)s around stimulus onset, giving spike counts per unit per bin. Counts are stored as float32 (NOT converted to firing rate). When a session has multiple probes, units are pooled into one population with continuous indexing.

ii.
```python
def bin_spikes(spike_times, spike_clusters, n_units, begs, ends):
    out = np.zeros((n_trials, n_units, NBINS), dtype=np.float32)
    # ...
    t = spike_times[a:b] - begs[k]
    bi = (t / BINSIZE).astype(np.int64)
    np.clip(bi, 0, NBINS - 1, out=bi)
    flat = spike_clusters[a:b] * NBINS + bi
    counts = np.bincount(flat, minlength=n_units * NBINS)
    out[k] = counts.reshape(n_units, NBINS)
    return out
```

iii. The AI stores raw spike counts rather than firing rates. The metadata states `'neural_units': 'spike counts per 20 ms bin'`. The reference code's caching also stores raw counts (the standardization is applied at training time, not caching time).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (well-isolated units passing all three RIGOR single-unit metrics). Additionally, clusters in Beryl regions `root` or `void` are excluded. A further region-level filter requires at least 5 well-isolated units per region per session, and the region must appear in at least 2 sessions across the dataset.

ii.
```python
good = cl['label'].to_numpy(dtype=float) >= 1.0
beryl = np.asarray(br.acronym2acronym(cl['acronym'].to_numpy(), mapping='Beryl'))
keep = good & ~np.isin(beryl, EXCLUDE_REGIONS)  # EXCLUDE_REGIONS = ('root', 'void')
```

```python
# >= MIN_UNITS_PER_REGION well-isolated units per region in this session
counts = Counter(reg.tolist())
keep_unit = np.array([counts[r] >= MIN_UNITS_PER_REGION for r in reg])
```

```python
# Cross-session region filter
keep_regions = set(r for r, c in region_session_count.items() if c >= min_sess)
```

iii. The AI states these are the data paper's curation criteria: "grey matter only, >=5 units/region/session, region in >=2 sessions". The reference code caches all clusters (qc=None) but stores the label for downstream use; the AI chose to apply QC at caching time.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to stimulus onset (`stimOn_times`). The trial window is [-0.5, +1.5)s around onset. Spike times are referenced from the beginning of the window (`begs = stim_on + WIN[0]`).

ii.
```python
stim_on = trials[PARAMS['align_time']].to_numpy(dtype=float)
begs_all = stim_on + WIN[0]
ends_all = stim_on + WIN[1]
# ...
binned = bin_spikes(st, sc, n_units, begs_all[idx], ends_all[idx])
```

iii. All IBL data streams share one synchronized clock, so alignment is simply subtracting the reference time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins total covering the 2s window. No rebinning or resampling is applied.

ii.
```python
PARAMS = {'interval_len': 2.0, 'binsize': 0.02,
          'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}
BINSIZE = PARAMS['binsize']
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))   # 100
```

iii. Matches the reference code's `'binsize': 0.02` and the methods paper description of "2-s trials, each divided into 20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table. The input is the right edge of each time bin relative to stimulus onset.

ii.
```python
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
# BIN_TIMES = [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. The AI chose bin right edges to match the reference code's `get_behavior_per_interval` grid: `np.linspace(t_beg + binsize, t_end, n_bins)`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond defining the time grid. The grid is the right edge of each 20ms bin, computed once and reused for every trial.

ii.
```python
time_row = BIN_TIMES.astype(np.float32)
# ...
inputs.append(np.stack([time_row, np.full(NBINS, tib[k], dtype=np.float32)]))
```

iii. The time values are deterministic given the window and bin size.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the right edges of the same bins the neural data is counted in. Bin i of the neural data covers [stim_on - 0.5 + i*0.02, stim_on - 0.5 + (i+1)*0.02), and BIN_TIMES[i] = -0.5 + (i+1)*0.02, which is that bin's right edge.

ii.
```python
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
```

iii. The AI's CONVERSION_NOTES Step 6 documents the alignment proof: behavior sample i and spike bin i end at the same instant.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Changes in `probabilityLeft` define block boundaries.

ii.
```python
def trial_number_in_block(prob_left):
    p = np.asarray(prob_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    new_block[1:] = p[1:] != p[:-1]
    starts = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
    return np.arange(len(p)) - starts
```

iii. The trials table has no explicit block identifier, so blocks are recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based index of each trial within its block of constant `probabilityLeft`, computed on the full trials table before any filtering, so excluded trials still advance the count and the number reflects the animal's true position in the block.

ii.
```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
# ... later, after filtering:
tib = tib_all[idx].astype(np.float32)
```

iii. Computing on the full table ensures the block count is not affected by trial exclusions.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which takes values +1 (left), -1 (right), 0 (no-go).

ii.
```python
choice_raw = trials['choice'].to_numpy()[idx]
choice = (choice_raw < 0).astype(np.int64)
```

iii. IBL convention verified empirically: +1 = reported LEFT, -1 = RIGHT.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Mapped to 0 (left) and 1 (right) using `(choice_raw < 0).astype(int)`. No-go trials (choice == 0) are already excluded by the trial mask.

ii.
```python
choice = (choice_raw < 0).astype(np.int64)
```

iii. This correctly maps +1 (left) to 0 and -1 (right) to 1, matching the decoder specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pleft = pl[idx]
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2]).astype(np.int64)
```

iii. The three values are the block priors; the mapping 0.2->0, 0.5->1, 0.8->2 is given by the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapped from continuous values to categorical using `np.select` with `np.isclose` comparisons. 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

ii.
```python
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2]).astype(np.int64)
```

iii. Uses `np.isclose` for floating-point safety, which is more robust than exact equality.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel position and timestamps (`_ibl_wheel.position`, `_ibl_wheel.timestamps`), loaded via `SessionLoader.load_wheel()` which interpolates to 1kHz and computes velocity. The speed is `abs(velocity)`.

ii.
```python
sl.load_wheel()
wheel_t = sl.wheel['times'].to_numpy()
wheel_v = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. Same source as the reference code's `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` interpolates position to 1kHz and applies a 20Hz Butterworth low-pass filter to compute velocity. The absolute value gives speed. The speed trace is then linearly interpolated onto the right edge of each 20ms bin for each trial using `scipy.interp1d`. Finally, discretized into 3 categories at per-session 33.3rd and 66.7th percentiles.

ii.
```python
wheel_vals, wheel_ok = interpolate_behavior(wheel_t, wheel_v, begs_all, ends_all)
# ...
wheel_d, wheel_thr = discretize_terciles(wheel_k)
```

```python
def interpolate_behavior(target_times, target_vals, begs, ends):
    f = interp1d(tt, tv, kind='linear', fill_value='extrapolate', bounds_error=False)
    grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
    vals = f(grid)
    return vals, good
```

iii. The interpolation grid matches the reference code's `get_behavior_per_interval`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session terciles: the 33.3rd and 66.7th percentiles of all (trial x timepoint) values in the session determine two thresholds, and `np.digitize` assigns each value to one of three categories (0=low, 1=medium, 2=high).

ii.
```python
def discretize_terciles(values):
    q1, q2 = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return np.digitize(values, [q1, q2]).astype(np.int64), (float(q1), float(q2))
```

iii. Per-session terciles ensure equal class sizes, which is good for balanced classification.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated onto the right edge of each neural bin, using the same `BIN_TIMES` grid as the time input. The interpolation grid is `begs + (BIN_TIMES - WIN[0])` which equals `stim_on + BIN_TIMES`, the right edge of each bin.

ii.
```python
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
vals = f(grid)
```

iii. Both the behavior and neural data reference the same time bins, ensuring alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy from the left camera (fallback to right), specifically the `whiskerMotionEnergy` column from `SessionLoader.load_motion_energy()`.

ii.
```python
for view in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[view])
        key = view + 'Camera'
        if key in sl.motion_energy and 'whiskerMotionEnergy' in sl.motion_energy[key]:
            me_t = sl.motion_energy[key]['times'].to_numpy()
            me_v = sl.motion_energy[key]['whiskerMotionEnergy'].to_numpy()
            break
    except Exception:
        continue
```

iii. Left camera preferred, right as fallback, matching the reference code logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is with no additional filtering. It is linearly interpolated onto the bin right edges using `scipy.interp1d`, then discretized into 3 categories at per-session terciles, same as wheel speed.

ii.
```python
me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
me_d, me_thr = discretize_terciles(me_k)
```

iii. Same interpolation and discretization pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: per-session 33.3rd and 66.7th percentile thresholds, producing 3 equally-populated categories.

ii.
```python
me_d, me_thr = discretize_terciles(me_k)
```

iii. Identical method to wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto the right edge of each neural bin via the `BIN_TIMES` grid.

ii.
```python
me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
```

iii. Uses the same `interpolate_behavior` function and grid as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) Trials with NaN in key columns are excluded by `load_trials_and_mask`. (2) Sessions without whisker motion energy are skipped. (3) Probes with no spike sorting or empty data are skipped. (4) Regions with fewer than 5 units per session or present in fewer than 2 sessions are removed. (5) Trials whose window extends beyond the spike sorting span or that have all-zero neural data are excluded. (6) Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
if spikes is None or len(spikes) == 0:
    continue
# ...
if st is None:
    return None, dict(info, skip='no well-isolated grey-matter units')
# ...
in_span = (begs_all >= st[0]) & (ends_all <= st[-1])
valid &= in_span
# ...
has_spikes = binned.sum(axis=(1, 2)) > 0
```

iii. The AI documents investigating 16 all-zero trials caused by three distinct issues (ephys recording ending early, mid-recording dropout, genuine silence with very few units) and adding the neural coverage check to address them.

## 10-a. What are the most time-consuming steps of the code?

i. Loading the spike sorting data from disk (hundreds of MB per probe) and the spike binning loop.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The AI uses session-level multiprocessing (24 workers) to parallelize the I/O-heavy processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop iterates over each trial. This could potentially be vectorized by offsetting spike bin indices per trial and doing a single bincount. The interpolation uses `scipy.interp1d` evaluated on the full grid at once rather than per-trial, which is already vectorized.

ii.
```python
for k in range(n_trials):
    a, b = i0[k], i1[k]
    # ...
    flat = spike_clusters[a:b] * NBINS + bi
    counts = np.bincount(flat, minlength=n_units * NBINS)
    out[k] = counts.reshape(n_units, NBINS)
```

iii. The per-trial loop for spike binning is a potential vectorization target, though the AI already optimized the inner loop to use flat indexing and `np.bincount`.

## 10-c. What processing does the code repeat multiple times?

i. The `interpolate_behavior` function is called twice per session (once for wheel, once for whisker ME), each time building a new `scipy.interp1d` object and computing searchsorted for the trial boundaries. Also, for `--show-processing` mode, the wheel and motion energy data is loaded a second time to produce raw traces for the plots.

ii.
```python
wheel_vals, wheel_ok = interpolate_behavior(wheel_t, wheel_v, begs_all, ends_all)
me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
```

iii. The repeated interpolation setup is minor in cost. The re-loading for plotting is only in debug mode and doesn't affect production performance.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores raw wheel and whisker values in `session['raw']` for plotting/debugging, which is popped before pickling. The `n_clusters_all` and `n_clusters_good` counts are computed for sanity-check reporting but not stored in the final data. The `trial_idx` array is stored in the session dict but not included in the final pickle structure.

ii.
```python
session = {'eid': eid, ..., 'raw': {'wheel_vals': wheel_k, 'me_vals': me_k, ...}}
# ...
for s in sessions:
    s.pop('raw', None)
```

iii. These are for validation/debugging purposes and are cleaned up before the final output.
