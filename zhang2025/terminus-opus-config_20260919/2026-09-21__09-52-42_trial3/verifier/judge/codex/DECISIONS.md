# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the session list from the release freeze file `bwm_release.csv`, optionally restricts it with `DATALIMIT_SUBSET.csv`, builds a local ONE client against a rebuilt `LocalIndex`, and then loads each session with `SessionLoader` plus one `SpikeSortingLoader` call per probe. This differs from the human reference, which uses `one.search(...)` on the Brainwidemap index to discover sessions and required datasets.

ii.
```python
TABLES_DIR = '/app/data/one_cache/LocalIndex'
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'

def get_one():
    if not os.path.exists(os.path.join(TABLES_DIR, 'datasets.pqt')):
        ...
        build_one_cache.main()
    return ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local',
               cache_dir=CACHE_DIR, tables_dir=TABLES_DIR)

def session_list():
    bwm = pd.read_csv(FREEZE_FILE, index_col=0)
    if os.path.exists(DATALIMIT):
        keep = pd.read_csv(DATALIMIT)
        ...
        bwm = bwm[bwm.eid.isin(set(keep[col].astype(str)))]
    return bwm
```
```python
spike_times, spike_clusters, clusters = load_session_spikes(one, eid, probes)
trials, mask, sl = trials_and_mask(one, eid)
beh = load_behaviour(sl)
```

iii. In `CONVERSION_NOTES.md`, the agent says the shipped ONE tables were stale and did not index revision folders correctly, so it rebuilt the datasets index and used `bwm_release.csv` as the authoritative release freeze. The trajectory repeatedly justifies this as a workaround for `load_trials()` otherwise returning a truncated one-column table.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. At assembly, the script builds a sorted unique `subjects` list and maps each session to its subject index.

ii.
```python
result = dict(
    eid=eid,
    subject=str(probes.iloc[0].subject),
    ...
)
```
```python
subjects = sorted({r['subject'] for r in results})
sub_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([sub_idx[r['subject']] for r in results], dtype=int),
```

iii. The notes justify this by saying `bwm_release.csv` is the same freeze file used by the Zhang reference pipeline, and it already contains the session-to-subject mapping.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from `bwm_release.csv`. The main loop deduplicates `bwm.eid` and processes one session per `eid`.

ii.
```python
bwm = session_list()
eids = list(dict.fromkeys(bwm.eid))
jobs = [(e, bwm[bwm.eid == e], args.show_processing and i < 2, outdir)
        for i, e in enumerate(eids)]
```

iii. The agent’s notes say the freeze file contains the 459 released sessions and should be treated as the authoritative session list once the cache index problem is fixed.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the rows of the IBL trials table loaded by `SessionLoader.load_trials()`. The per-session code keeps the indices of rows that survive the mask and then extracts aligned trial-wise arrays from those rows.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
tr = sl.trials
```
```python
keep = np.where(mask)[0]
align = trials[ALIGN_TIME].to_numpy()[keep]
...
n_tr = len(keep)
```

iii. The justification is implicit: the agent follows the structure of the IBL trials table and the Zhang reference utilities, where each table row already corresponds to one trial.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies the Zhang `load_trials_and_mask` logic: remove trials with reaction time outside 0.08-2.0 s, long `feedback_times - goCue_times` (>10 s), no-go trials (`choice == 0`), or NaNs in key columns. After that, it bins wheel and whisker traces and drops trials that do not have complete behavioural coverage. If fewer than two behaviour-complete trials remain, it drops the session.

ii.
```python
rt = tr['firstMovement_times'] - tr['stimOn_times']
bad = (rt < MIN_RT) | (rt > MAX_RT)
bad |= (tr['feedback_times'] - tr['goCue_times']) > MAX_TRIAL_LEN
bad |= tr['choice'] == 0
for col in NAN_EXCLUDE:
    bad |= tr[col].isna()
return tr, (~bad).to_numpy(), sl
```
```python
wheel, ok_w = bin_behaviour(wt, wv, align)
...
whisk, ok_m = bin_behaviour(mt, mv, align)
valid = ok_w & ok_m
if valid.sum() < 2:
    raise RuntimeError(f'only {valid.sum()} trials with complete behaviour')
keep, align = keep[valid], align[valid]
```

iii. The notes explicitly cite `load_trials_and_mask(max_trial_len=10)` as the intended reference logic and add the behaviour-coverage filter as necessary because wheel speed and whisker motion energy are required decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays are derived from raw spike times and spike cluster assignments loaded for each probe. The merged cluster table is used for QC and region labels, but the actual binned signal is built from `spikes['times']` and `spikes['clusters']`.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
...
times.append(spikes['times'])
clus.append(spikes['clusters'] + offset)
...
return (np.concatenate(times), np.concatenate(clus),
        pd.concat(tables, ignore_index=True))
```

iii. The notes say the neural signal is “count spikes per 20 ms bin per neuron” from `spikes.times / spikes.clusters`, while `clusters.label` and `clusters.acronym` are only used for filtering and region annotation.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, remaps kept clusters to a contiguous unit index, sorts spikes by time, and bins spikes into 100 bins of 20 ms each for every aligned trial. It stores spike counts, not firing rates.

ii.
```python
sel = np.isin(spike_clusters, unit_idx)
st, sc = spike_times[sel], remap[spike_clusters[sel]]
order = np.argsort(st, kind='stable')
st, sc = st[order], sc[order]
```
```python
tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
np.clip(tb, 0, N_BINS - 1, out=tb)
flat = np.bincount(sc[s] * N_BINS + tb, minlength=n_units * N_BINS)
out[k] = flat.reshape(n_units, N_BINS)
```

iii. The notes justify spike counts as the direct output of binning and say the provided decoder internally z-scores, so storing counts is acceptable and more compact. The trajectory also shows the agent switched to `float32` counts after format warnings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only clusters with `label >= 1`, maps their acronyms into the Beryl atlas, removes units whose Beryl label is either `root` or `void`, and then rejects sessions with fewer than 5 surviving units.

ii.
```python
NON_GREY = ('root', 'void')
MIN_UNITS_PER_SESSION = 5
...
beryl = np.asarray(BrainRegions().acronym2acronym(
    clusters['acronym'].to_numpy(), mapping='Beryl'), dtype=object)
keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY)
return np.where(keep)[0], beryl
```
```python
unit_idx, beryl = good_grey_units(clusters)
if len(unit_idx) < MIN_UNITS_PER_SESSION:
    raise RuntimeError(f'only {len(unit_idx)} well-isolated grey-matter units '
                       f'(minimum {MIN_UNITS_PER_SESSION})')
```

iii. The notes justify `label >= 1` as reproducing the paper’s 75,708 well-isolated neurons exactly, justify dropping `root`/`void` as a grey-matter restriction, and justify the 5-unit minimum by citing the data paper’s “at least five well-isolated neurons per session” rule to eliminate degenerate near-empty sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each trial, the code defines a window from -0.5 s to +1.5 s relative to stimulus onset and bins spikes inside that window.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align = trials[ALIGN_TIME].to_numpy()[keep]
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
```

iii. The notes repeatedly say this matches both the decoder task and the Zhang caching parameters (`align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins across a 2 s window, producing 100 time bins per trial. There is no extra temporal rebinning after this.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```
```python
tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
```

iii. The notes cite the reference code and the paper’s “2-s trials, each divided into 20-ms bins, producing T = 100” statement as the reason for keeping 20 ms resolution.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the alignment variable `stimOn_times`, together with the fixed decoding window and bin size. The final values are not read from a raw column directly; they are constructed relative to stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
...
align = trials[ALIGN_TIME].to_numpy()[keep]
```
```python
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
inputs[:, 0, :] = tgrid.astype(np.float32)
```

iii. The notes say this input is a task-required constructed variable anchored to the same `stimOn_times` event used for neural alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent constructs a fixed 100-point time grid using bin end times from `-0.5 + 0.02` to `1.5` seconds, then broadcasts that same vector to every kept trial. It also adds a separate `stim_onset` binary indicator input, although that is a second input rather than part of `time_from_stim_onset` itself.

ii.
```python
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
...
inputs = np.empty((n_tr, 3, N_BINS), dtype=np.float32)
inputs[:, 0, :] = tgrid.astype(np.float32)
inputs[:, 1, :] = onset
inputs[:, 2, :] = tib[:, None]
```

iii. The notes justify bin-end times by saying they match the behaviour interpolation grid used by `get_behavior_per_interval`, and justify the extra `stim_onset` indicator by the task instruction that “a time input should be a binary time series.”

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time input is aligned trial-by-trial to the same stimulus-onset windows as the neural data. However, the script uses bin end times for the input grid, whereas neural spikes are counted over half-open 20 ms bins spanning the same intervals.

ii.
```python
counts = bin_spikes(spike_times, spike_clusters, unit_idx, align)
...
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
inputs[:, 0, :] = tgrid.astype(np.float32)
```

iii. The notes explicitly say the time grid is chosen to match the behaviour interpolation grid and that the onset indicator should mark the first bin containing `t >= 0`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`. A change in `probabilityLeft` marks the start of a new block.

ii.
```python
def trial_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=float)
    new = np.ones(len(pl), dtype=bool)
    new[1:] = pl[1:] != pl[:-1]
    idx = np.arange(len(pl))
    return idx - np.maximum.accumulate(np.where(new, idx, 0))
```

iii. The notes justify this by saying the trials table has no explicit block ID, so blocks must be reconstructed from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code computes a 0-based running index within each block over the full unfiltered trial sequence, then subsets that vector after trial filtering and broadcasts it across time bins for each kept trial.

ii.
```python
tib_all = trial_in_block(trials['probabilityLeft'].to_numpy())
...
tib = tib_all[keep].astype(np.float32)
inputs[:, 2, :] = tib[:, None]
```

iii. The notes explicitly justify computing it before masking so that removing trials does not distort a mouse’s true position within the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials['choice']`.

ii.
```python
choice = trials['choice'].to_numpy()[keep]
```

iii. The notes say the agent verified the IBL sign convention and concluded that `+1` means left and `-1` means right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code remaps `choice > 0` to class 0 (left) and everything else that survives masking to class 1 (right), then broadcasts that per-trial label across all 100 time bins.

ii.
```python
y_choice = np.where(choice > 0, 0, 1).astype(np.int16)
...
outputs[:, 0, :] = y_choice[:, None]
```

iii. The notes justify this as the task-required recoding from IBL’s `+1/-1` convention to left `0` and right `1`, after no-response trials were already removed by the mask.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from `trials['probabilityLeft']`.

ii.
```python
pl = trials['probabilityLeft'].to_numpy()[keep]
```

iii. The notes say the agent confirmed empirically that the left stimulus frequency matches the literal values 0.2, 0.5, and 0.8.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. Any trial with another value is dropped at this stage. The resulting class is broadcast across time bins.

ii.
```python
y_prior = np.select([np.isclose(pl, 0.2), np.isclose(pl, 0.5), np.isclose(pl, 0.8)],
                    [0, 1, 2], default=-1).astype(np.int16)
if (y_prior < 0).any():
    okp = y_prior >= 0
    keep, align, counts, wheel, whisk = (keep[okp], align[okp], counts[okp],
                                         wheel[okp], whisk[okp])
    y_choice, y_prior = y_choice[okp], y_prior[okp]
```

iii. The notes justify keeping the unbiased 0.5 block because it is a required decoder class and matches the reference setting `exclude_unbiased=False`.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed comes from the session wheel trace loaded by `SessionLoader.load_wheel()`, specifically the absolute value of the loader’s `velocity` field together with the wheel timestamps.

ii.
```python
sl.load_wheel()
out['wheel_speed'] = (sl.wheel['times'].to_numpy(),
                      np.abs(sl.wheel['velocity'].to_numpy()))
```

iii. The notes say this matches the Zhang helper `load_target_behavior('wheel-speed')`, which defines wheel speed as `abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script takes `abs(velocity)`, linearly interpolates it onto each trial’s 100-bin stimulus-aligned grid using `interp1d`, and then discretises the whole session’s retained wheel samples into tertiles.

ii.
```python
wheel, ok_w = bin_behaviour(wt, wv, align)
...
grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
out[k] = interp1d(t[finite], v[finite], kind='linear',
                  fill_value='extrapolate')(grid)
```
```python
y_wheel, thr_w = tertile_bins(wheel)
```

iii. The notes justify using interpolation because the Zhang helper `get_behavior_per_interval` does that, and justify session-wise tertiles because wheel speed scale is session-specific but the task requires categorical outputs.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded into three categories using the 33rd and 67th percentiles of all finite wheel-speed values from the retained trials of a session.

ii.
```python
finite = values[np.isfinite(values)]
...
lo, hi = np.percentile(finite, [100 / 3, 200 / 3])
return (np.digitize(values, [lo, hi]).astype(np.int16), (float(lo), float(hi)))
```

iii. The notes justify per-session tertiles as producing balanced classes and avoiding a global threshold that would mostly encode session identity.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset trial windows as the neural data and sampled on the same 100-bin trial grid, but at bin end times rather than bin centres.

ii.
```python
counts = bin_spikes(spike_times, spike_clusters, unit_idx, align)
wheel, ok_w = bin_behaviour(wt, wv, align)
```
```python
grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
```

iii. The notes explicitly say the behaviour traces are resampled exactly like the reference helper `get_behavior_per_interval`, which evaluates them at `linspace(beg + binsize, end, n_bins)`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy comes from the motion-energy trace and timestamps of a side camera. The code prefers the left camera and falls back to the right camera if the left is unavailable.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        df = sl.motion_energy[cam]
        vals = df['whiskerMotionEnergy'].to_numpy()
        if np.isfinite(vals).any():
            me = (df['times'].to_numpy(), vals, view)
            break
```

iii. The notes justify this as following the Zhang behaviour-loading logic and as necessary because 14 sessions have neither left nor right whisker motion energy.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used directly, linearly interpolated onto the 100-bin stimulus-aligned grid for each retained trial, and then discretised session-wise into tertiles.

ii.
```python
mt, mv, me_side = beh['whisker_motion_energy']
whisk, ok_m = bin_behaviour(mt, mv, align)
...
y_whisk, thr_m = tertile_bins(whisk)
```

iii. The notes justify using the released trace without extra normalization and say the required categorical output is obtained via the same tertile rule as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded into low, medium, and high classes using the 33rd and 67th percentiles of all finite whisker-motion-energy values from the retained trials of that session.

ii.
```python
y_whisk, thr_m = tertile_bins(whisk)
```

iii. The notes give the same justification as for wheel speed: balanced per-session classes are preferable to global thresholds because the raw scale is session-specific.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, whisker motion energy is aligned to the same per-trial stimulus-onset windows as the neural data and sampled on the same 100-bin grid, again using bin end times.

ii.
```python
whisk, ok_m = bin_behaviour(mt, mv, align)
...
grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
```

iii. The notes justify this by citing the Zhang interpolation helper as the intended behaviour-alignment rule.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing or problematic data by rebuilding the stale cache index, skipping sessions with no wheel data, no whisker motion energy, or no spike data, dropping trials that fail the mask or lack complete behaviour coverage, dropping invalid prior-probability trials, and rejecting sessions with fewer than 5 kept units or fewer than 2 valid trials.

ii.
```python
if not os.path.exists(os.path.join(TABLES_DIR, 'datasets.pqt')):
    ...
    build_one_cache.main()
```
```python
if beh['whisker_motion_energy'] is None:
    raise RuntimeError('no whisker motion energy (left or right)')
if beh['wheel_speed'] is None:
    raise RuntimeError('no wheel data')
...
if len(unit_idx) < MIN_UNITS_PER_SESSION:
    raise RuntimeError(...)
if valid.sum() < 2:
    raise RuntimeError(...)
```

iii. The notes and trajectory justify the rebuilt index as mandatory because the shipped cache tables were stale, and justify the stricter session skips as necessary to avoid silently producing malformed outputs for required decoder targets.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies spike loading as the dominant cost. The trajectory says a worst-case 2-probe session spends most of its wall time in loading spike sorting, and the final script records stage timings for spike loading, trial loading, behaviour loading, behaviour binning, and spike binning.

ii.
```python
t = time.time()
spike_times, spike_clusters, clusters = load_session_spikes(one, eid, probes)
timings['load_spikes'] = time.time() - t
```

iii. The notes explicitly say spike loading dominates runtime and that vectorised binning keeps the rest relatively cheap.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops remain: one in `bin_spikes` and one in `bin_behaviour`. The agent already vectorised some setup work, but trial-wise slicing/interpolation and per-trial `bincount` calls are still explicit Python loops that could be fused further.

ii.
```python
for k in range(len(align_times)):
    if i1[k] <= i0[k]:
        continue
    ...
    flat = np.bincount(sc[s] * N_BINS + tb, minlength=n_units * N_BINS)
    out[k] = flat.reshape(n_units, N_BINS)
```
```python
for k in range(n):
    if not np.isfinite(begs[k]) or not np.isfinite(ends[k]):
        continue
    ...
    out[k] = interp1d(...)(grid)
    ok[k] = True
```

iii. The notes argue the main efficiency gains already came from sorting spike trains once per session and parallelising across sessions rather than within-session intervals.

## 10-c. What processing does the code repeat multiple times?

i. The clearest repeated work is that `convert_session` calls `get_one()` for every session, so each session rebuilds or reopens a ONE client instead of reusing one client per worker process. The script also recomputes the same trial grid (`tgrid`) inside each session. The notes do not call these out; they focus instead on vectorised spike binning and session-level parallelism.

ii.
```python
def convert_session(eid, probes, show=False, outdir='/app'):
    t0 = time.time()
    one = get_one()
```
```python
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. There is no explicit justification in the final notes for repeating ONE setup per session. The trajectory mainly justifies correctness and robustness, not this efficiency choice.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is no large discarded computation in the main data path, but the script does collect extra bookkeeping and optional diagnostics that the decoder does not use directly: per-session timing breakdowns, lab IDs, thresholds, and diagnostic plotting support.

ii.
```python
timings = {}
...
result = dict(
    eid=eid,
    subject=str(probes.iloc[0].subject),
    lab=str(probes.iloc[0].lab),
    ...
    thr_wheel=thr_w,
    thr_whisker=thr_m,
    timings=timings,
    elapsed=time.time() - t0,
)
```
```python
if show:
    plot_processing(...)
```

iii. The notes justify these extras as traceability and sanity-check aids rather than as part of the decoder input itself.
