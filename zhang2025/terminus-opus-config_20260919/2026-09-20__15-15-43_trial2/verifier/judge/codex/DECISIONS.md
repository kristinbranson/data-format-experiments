# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the release manifest from `bwm_release.csv`, uses unique `eid` values as sessions, and uses the manifest's `pid` / `probe_name` columns to know which probes belong to each session. It then uses a local `ONE` client plus `SessionLoader` / `SpikeSortingLoader` to load trials, wheel, motion energy, and spike sorting from the cache.

ii.
```python
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
ONE_KWARGS = dict(cache_dir='/app/data/one_cache',
                  tables_dir='/app/cache/one_tables',
                  mode='local', silent=True)
```

```python
bwm = pd.read_csv(BWM_CSV, index_col=0)
eids = sorted(bwm.eid.unique())
tasks = [(eid, bwm[bwm.eid == eid][['pid', 'probe_name']].copy(),
          i < n_show, out_dir, not args.no_zscore) for i, eid in enumerate(eids)]
```

```python
sess_loader = SessionLoader(one=one, eid=eid)
ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
```

iii. In `CONVERSION_NOTES.md`, the AI says it could not rely on `one.eid2pid()` or the shipped ONE tables, so it rebuilt local ONE tables and used `bwm_release.csv` as the authoritative session/probe index.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are not inferred from paths. The AI uses the `subject` column in `bwm_release.csv`, then builds `subjects` as the sorted unique names among converted sessions and `subject_idx` as the per-session lookup into that list.

ii.
```python
eid2subject = bwm.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
subjects = sorted({eid2subject[r['eid']] for r in ok})
subject_index = {s: i for i, s in enumerate(subjects)}
```

```python
'subject_idx': np.array([subject_index[eid2subject[r['eid']]] for r in ok],
                        dtype=np.int64),
```

iii. The notes justify this as using the release freeze metadata directly, rather than deriving subject identity from filenames or directory layout.

## 1-c. How are the data split into sessions?

i. A session is one unique `eid` from `bwm_release.csv`. The AI processes one `eid` at a time and later assembles one output entry per surviving `eid`.

ii.
```python
eids = sorted(bwm.eid.unique())
```

```python
def convert_session(eid, probe_rows, show_processing=False, out_dir='.', zscore=True):
```

```python
'neural': [r['neural'] for r in ok],
'input': [r['input'] for r in ok],
'output': [r['output'] for r in ok],
```

iii. The AI treated session boundaries as already defined by the release manifest.

## 1-d. How are the data split into trials?

i. Trials come from the session trials table loaded by `load_trials_and_mask`. Each retained row becomes one trial; time-varying neural, input, and output arrays are created per retained trial.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                    sess_loader=sess_loader)
align_times = trials[ALIGN_EVENT].values[mask]
```

```python
for k in range(n_trials):
    inputs.append(arr)
...
for k in range(n_trials):
    outputs.append(arr)
```

iii. The AI's notes describe the trials table as one row per trial and say the per-trial arrays are built only after filtering.

## 1-e. How are trials filtered based on quality controls?

i. The AI imports the reference helper `load_trials_and_mask(..., max_trial_len=10.0)` and uses its boolean mask. That mask removes no-choice trials, trials with invalid key events, reaction times outside 0.08-2.0 s, and trials longer than 10 s. It then additionally requires valid wheel and motion-energy coverage across the full 2 s aligned window, and skips sessions with fewer than two surviving trials.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                    sess_loader=sess_loader)
mask = np.asarray(mask, dtype=bool)
```

```python
keep = valid_wheel & valid_me
if keep.sum() < MIN_TRIALS_PER_SESSION:
    return dict(eid=eid, skip='fewer than %d trials with complete behaviour' % MIN_TRIALS_PER_SESSION,
                n_trials_raw=n_trials_raw, n_trials_mask=int(mask.sum()))
```

iii. The notes explicitly justify this as using the runnable Zhang et al. trial mask verbatim, then adding full-behavior coverage so neural and behavioral outputs share the same aligned window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The stored neural arrays are derived from `spikes['times']` and `spikes['clusters']` loaded from each probe. Cluster metadata is used only to filter units and assign region labels.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

```python
times_list.append(np.asarray(spikes['times'])[sel])
clusters_list.append(remap[np.asarray(spikes['clusters'])[sel]])
```

iii. The AI's notes describe this as mirroring `load_spiking_data` + `merge_probes`, with the cluster table only supplying QC and anatomy.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 20 ms bins over the aligned 2 s window, merges both probes of a session into one neuron population, and stores the result as standardized spike counts. It does not convert to firing rate in Hz; instead it applies per-time-bin standardization pooled over all neurons and trials of the session.

ii.
```python
bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
np.clip(bin_idx, 0, N_BINS - 1, out=bin_idx)
np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
```

```python
mu = spikes_binned.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
sd = spikes_binned.std(axis=(0, 1), keepdims=True, dtype=np.float64)
neural = ((spikes_binned.astype(np.float64) - mu)
          / np.where(sd > 0, sd, 1.0)).astype(np.float32)
```

iii. The notes first proposed per-neuron z-scoring, then later corrected that to "reference per-time-bin standardisation" after re-reading `standardize_spike_data`. The stated reason was that `/app/decoder.py` does no normalization itself.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with IBL QC `label >= 1` and drops units whose Beryl-mapped acronym is in `('root', 'void')`. It merges only the surviving clusters across probes.

ii.
```python
good = (clu['label'].values >= 1)
beryl = np.asarray(br.acronym2acronym(clu['acronym'].to_numpy(), mapping='Beryl'))
keep = good & ~np.isin(beryl, NON_GREY)
```

iii. The notes justify `label >= 1` as matching the paper's "well-isolated neurons" count, and justify dropping `root` / `void` as enforcing a grey-matter-only inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to `stimOn_times`. For each trial, the binning window is `[stimOn_times - 0.5 s, stimOn_times + 1.5 s)`, so time zero is stimulus onset.

ii.
```python
ALIGN_EVENT = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
```

iii. The notes say this follows both the decoder task and the Zhang et al. alignment parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 20 ms bins across a 2 s window, yielding 100 time bins per trial. No further temporal rebinning is applied.

ii.
```python
BIN_SIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
```

iii. The notes explicitly say 20 ms was chosen to match the runnable reference configuration `binsize=0.02`, `time_window=(-0.5, 1.5)`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time-since-stimulus input is derived from the chosen alignment event `stimOn_times` and the fixed bin grid; the stored values are the bin centers relative to stimulus onset.

ii.
```python
ALIGN_EVENT = 'stimOn_times'
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
```

```python
time_axis = BIN_CENTRES.astype(np.float32)
arr[0] = time_axis
```

iii. The notes describe this as a new decoder input required by the task, defined from the aligned bin grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No raw signal is transformed; the AI simply fills each trial with the constant vector of bin-center times `[-0.49, -0.47, ..., 1.49]`.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
...
arr[0] = time_axis
```

iii. The notes justify this as the simplest continuous time-varying representation required by the decoder task.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same 100-bin trial grid as the neural data, so each time value corresponds to the same bin index used for spike counts.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
```

```python
out = np.zeros((n_trials, n_neurons, N_BINS), dtype=np.float32)
```

iii. The notes state that this input is meant to share the neural time base exactly.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`, with block boundaries inferred whenever that value changes.

ii.
```python
def trial_number_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=float)
    new_block = np.ones(len(pl), dtype=bool)
    new_block[1:] = pl[1:] != pl[:-1]
```

iii. The notes explain that the dataset has no explicit block index, so blocks are recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a 0-based count since the start of the current block on the unmasked trials table, then applies the trial mask afterwards so dropped trials still advance the counter.

ii.
```python
block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(pl)), 0))
return np.arange(len(pl)) - block_start
```

```python
tnb_all = trial_number_in_block(trials['probabilityLeft'].values)
...
tnb = tnb_all[mask]
...
arr[2] = tnb[k]
```

iii. The notes explicitly justify computing this before masking because block position is a property of the experiment, not of the filtered dataset.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials['choice']` after masking to retained trials.

ii.
```python
choice_raw = trials['choice'].values[mask]
```

iii. The notes say the ALF sign convention was verified empirically across sessions before recoding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps positive ALF choice to decoder class `0` (left) and negative choice to class `1` (right), then broadcasts the per-trial value across all 100 time bins.

ii.
```python
choice = np.where(choice_raw > 0, 0, 1).astype(np.int8)
```

```python
arr = np.empty((4, N_BINS), dtype=np.int8)
arr[0] = choice[k]
```

iii. The notes justify this with the task requirement "left = 0, right = 1" and the empirically verified ALF sign convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials['probabilityLeft']` after trial filtering.

ii.
```python
prob_left_raw = trials['probabilityLeft'].values[mask]
```

iii. The AI treated the three released block priors as the decoder classes.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI recodes `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts the result across the 100 bins of each trial.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_map[round(float(p), 4)] for p in prob_left_raw], dtype=np.int8)
```

```python
arr[1] = prior[k]
```

iii. The notes justify retaining the unbiased `0.5` block because the decoder task explicitly asks for three prior classes.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `SessionLoader`'s wheel timestamps and wheel velocity, using the absolute value of the velocity trace.

ii.
```python
sess_loader.load_wheel()
wheel_speed_raw, valid_wheel = interp_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
```

iii. The notes say this follows the reference behavior-loading path for `wheel-speed`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI loads the precomputed wheel velocity from `SessionLoader`, takes its absolute value, linearly interpolates it onto a per-trial 100-point grid anchored to stimulus onset, and then discretizes the pooled session values into three equal-count classes.

ii.
```python
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
...
f = interp1d(beh_times, beh_values, kind='linear', bounds_error=False,
             fill_value=(beh_values[0], beh_values[-1]))
vals[valid] = f(grid[valid])
```

```python
wheel_lab, wheel_edges = discretize(wheel_speed_raw)
```

iii. The notes justify this as matching the Zhang et al. behavior helper more closely than simple per-trial `np.interp`, and as making session-specific scales comparable.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded into three classes using per-session quantiles of all retained wheel-speed samples. The thresholds are nudged apart if duplicate quantiles occur.

ii.
```python
flat = values.ravel()
qs = np.linspace(0, 1, n_bins + 1)[1:-1]
edges = np.quantile(flat, qs)
for i in range(1, len(edges)):
    if edges[i] <= edges[i - 1]:
        edges[i] = np.nextafter(edges[i - 1], np.inf)
labels = np.searchsorted(edges, flat, side='right').astype(np.int8)
```

iii. The notes justify per-session terciles because wheel speed is skewed and session scale varies, so equal-count bins give balanced decoder classes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI aligns wheel speed to the same stimulus-onset-centered trial windows as the neural data, but samples the continuous trace on `BIN_RIGHT_EDGES` rather than bin centers.

ii.
```python
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 1.0)
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
```

iii. The notes justify the right-edge sampling by referring to the reference `get_behavior_per_interval` helper, which evaluates behavior on `t_beg + binsize ... t_end`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera` or, if unavailable, `rightCamera` motion-energy traces and their timestamps. The AI prefers left camera and falls back to right.

ii.
```python
for view in ('left', 'right'):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[view + 'Camera']
```

iii. The notes explicitly say this follows the reference `bin_behaviors` preference order.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released whisker motion-energy trace as-is, linearly interpolates it onto the aligned 100-bin grid for each retained trial, and discretizes the pooled session values into three equal-count classes.

ii.
```python
me_raw, valid_me = interp_behavior(me['times'].to_numpy(),
                                   me['whiskerMotionEnergy'].to_numpy(),
                                   align_times)
```

```python
me_lab, me_edges = discretize(me_raw)
```

iii. The notes justify not adding extra filtering or normalization because the release already provides the relevant motion-energy signal.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is discretized exactly like wheel speed: per-session quantiles over all retained trial/bin samples, with duplicate thresholds nudged apart if necessary.

ii.
```python
edges = np.quantile(flat, qs)
for i in range(1, len(edges)):
    if edges[i] <= edges[i - 1]:
        edges[i] = np.nextafter(edges[i - 1], np.inf)
labels = np.searchsorted(edges, flat, side='right').astype(np.int8)
```

iii. The notes justify terciles because motion-energy units are session- and camera-dependent.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-onset-centered trial windows as the neural data, using the same 100-bin grid as wheel speed and the same right-edge sample times.

ii.
```python
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
```

iii. The notes justify this as reusing the same stimulus-onset alignment for all decoder outputs.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI drops invalid trials and skips unusable sessions rather than imputing missing data. Sessions are skipped if they lack whisker motion energy, have too few valid trials, or have no surviving neurons. Within a session, only trials with complete wheel and motion-energy coverage and finite interpolated values are kept.

ii.
```python
finite = np.isfinite(beh_times) & np.isfinite(beh_values)
...
valid &= np.isfinite(vals).all(axis=1)
```

```python
if me_raw is None:
    return dict(eid=eid, skip='no whisker motion energy from either camera', ...)
...
if spike_times is None or len(acronyms) == 0:
    return dict(eid=eid, skip='no well-isolated grey-matter neurons', ...)
```

iii. The notes repeatedly describe the strategy as "drop missing data" rather than repair it, to keep all modalities exactly aligned.

## 10-a. What are the most time-consuming steps of the code?

i. The AI explicitly times stages and reports spike loading as by far the dominant cost, with wheel loading a distant second and binning much cheaper.

ii.
```python
timing['wheel'] = time.time() - t0
...
timing['spikes'] = time.time() - t0
timing['binning'] = time.time() - t0
```

```python
for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    print('  %-14s %7.1f s  (%.0f%%)' % (k, v, 100 * v / max(tot, 1e-9)))
```

iii. `CONVERSION_NOTES.md` and `conversion_full_out.txt` both say spike loading dominates runtime because it is the heaviest disk I/O.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized most behavioral interpolation across the full `(n_trials, 100)` grid, but it still retains a per-trial loop in `bin_spikes` and per-trial loops when packaging inputs and outputs. Those loops could be further vectorized if needed.

ii.
```python
for k in range(n_trials):
    a, b = i0[k], i1[k]
    ...
    np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
```

```python
for k in range(n_trials):
    arr = np.empty((3, N_BINS), dtype=np.float32)
    ...
for k in range(n_trials):
    arr = np.empty((4, N_BINS), dtype=np.int8)
```

iii. The notes emphasize that the major optimization was moving from per-trial behavior interpolation to a single vectorized interpolation, and from rescanning spikes per trial to `searchsorted` plus `np.add.at`.

## 10-c. What processing does the code repeat multiple times?

i. The AI avoids most repeated heavy processing inside a session, but it still makes extra full-dataset passes for reporting and metadata assembly, such as stacking all outputs and all inputs again for summary statistics. It also recomputes small per-trial array allocations in the input/output packaging loops.

ii.
```python
allout = np.concatenate([np.stack(r['output'])[:, :, :] for r in ok], axis=0)
...
allin = np.concatenate([np.stack(r['input'])[:, :, :] for r in ok], axis=0)
```

```python
for k in range(n_trials):
    arr = np.empty((3, N_BINS), dtype=np.float32)
...
for k in range(n_trials):
    arr = np.empty((4, N_BINS), dtype=np.int8)
```

iii. The notes mainly criticize repeated work in the reference pipeline; for this code, the repeated work is limited to diagnostics and packaging rather than repeated reloading or rebinding of raw data.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores diagnostic quantities that the downstream decoder does not use, including timing summaries, mean firing rates, discretization edges, skipped-session reasons, and optional plotting inputs. It also makes reporting-only passes over the assembled dataset to print class balances and input ranges.

ii.
```python
raw_counts_example = spikes_binned[0].copy() if show_processing else None
...
if show_processing:
    plot_processing(...)
```

```python
return dict(
    ...
    wheel_edges=wheel_edges,
    me_edges=me_edges,
    mean_rate=float(spikes_binned.mean() / BIN_SIZE),
    timing=timing,
)
```

```python
'session_info': [
    {'eid': r['eid'], ..., 'wheel_speed_tercile_edges': [float(x) for x in r['wheel_edges']],
     'whisker_me_tercile_edges': [float(x) for x in r['me_edges']],
     'mean_firing_rate_hz': r['mean_rate']}
    for r in ok],
```

iii. The notes justify these extras as validation and documentation aids rather than part of the decoder-facing representation.
