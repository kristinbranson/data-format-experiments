# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the BWM release table, optionally restricts it with `DATALIMIT_SUBSET.csv`, and processes every unique session ID. It uses ONE-backed `SpikeSortingLoader` and `SessionLoader`, loading all probes for each session. Per-session results are cached as pickle files and later assembled.

ii.
```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
if os.path.exists(DATALIMIT):
    sub = pd.read_csv(DATALIMIT)
    bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
eids = list(dict.fromkeys(bwm.eid.tolist()))
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
sl = SessionLoader(one=one, eid=eid)
```

iii. The trajectory says this is exactly the release freeze iterated by the paper code. ONE was chosen to resolve cached IBL data, and per-session caching/parallelism made the large conversion resumable.

## 1-b. How are the data split into subjects?

i. Subject labels come from the release-table rows. During assembly, subjects are accumulated in first-seen order and each session receives the corresponding integer index.

ii.
```python
'subject': str(rows.subject.iloc[0]),
if res['subject'] not in subjects:
    subjects.append(res['subject'])
data['subject_idx'].append(subjects.index(res['subject']))
```

iii. The trajectory treated the release table's subject as the authoritative unique mouse identifier.

## 1-c. How are the data split into sessions?

i. Each unique `eid` is one session. All probe rows sharing that `eid` form one job and their neurons are merged into one session population.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))
jobs = [(eid, bwm[bwm.eid == eid]) for eid in eids ...]
```

iii. The agent followed the BWM release organization and the reference method's merging of probes recorded during the same behavior session.

## 1-d. How are the data split into trials?

i. The trials table already has one row per trial. Each retained row defines an interval beginning 0.5 s before `stimOn_times`; neural and behavior arrays are indexed by the same retained trial indices and converted to lists during assembly.

ii.
```python
align = trials[ALIGN_TIME].to_numpy(dtype=float)
interval_begs = align + TIME_WINDOW[0]
tidx = np.nonzero(valid)[0]
binned = bin_spikes(..., interval_begs[tidx])
```

iii. The trajectory recognized that the IBL trials table supplies the trial split directly.

## 1-e. How are trials filtered based on quality controls?

i. Trials are rejected for reaction times outside 0.08–2 s, choice zero, trial length over 10 s, or NaNs in six trial columns. Trials without complete, NaN-free wheel and whisker coverage are also rejected; sessions with fewer than two surviving trials are skipped.

ii.
```python
nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
query = f'(firstMovement_times - stimOn_times < {min_rt})'
query += f' | (firstMovement_times - stimOn_times > {max_rt})'
query += f' | (feedback_times - goCue_times > {max_trial_len})'
query += ' | (choice == 0)'
valid = trials_mask & ws_good & wm_good
```

iii. The agent said this was the standard BWM `load_trials_and_mask` used by `prepare_data`, plus the behavioral coverage checks used by `get_behavior_per_interval`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each probe's spike timestamps and spike cluster assignments. Cluster metrics/anatomy select neurons and assign regions.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
spike_times = np.concatenate([s['times'] for s in ms])
spike_clusters = np.concatenate([s['clusters'] for s in ms])
```

iii. The agent followed the paper code's spike-sorting loaders and merged all probes in a session.

## 2-b. How is the `neural` data processed?

i. Probe cluster IDs are offset and merged, retained cluster IDs are remapped contiguously, and spikes are counted in 100 nonoverlapping 20 ms bins per trial. The stored values are raw spike counts (`float32`), not rates or smoothed activity.

ii.
```python
tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
flat = sc[b:e] * N_BINS + tb
counts = np.bincount(flat, minlength=n_clusters * N_BINS)
out[k] = counts.reshape(n_clusters, N_BINS)
```

iii. The trajectory states that this is equivalent to the reference `bincount2D` processing. It explicitly described the final neural units as spike counts per 20 ms bin.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with IBL `label >= 1` are retained, and Beryl regions `root` and `void` are excluded. Sessions with no retained neurons are skipped.

ii.
```python
keep = (label >= 1) & ~np.isin(beryl, ['root', 'void'])
if n_neurons < MIN_NEURONS_PER_SESSION:
    return {'eid': eid, 'skip': ...}
```

iii. The agent found that all Kilosort units would make an approximately 164 GB result, so it chose the data paper's well-isolated-neuron criterion (about 75.7k units) and characterized `root`/`void` exclusion as a grey-matter restriction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`; each neural window spans -0.5 to +1.5 s, with bin indices computed relative to the window start.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
interval_begs = align + TIME_WINDOW[0]
tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
```

iii. The agent identified these as the exact alignment event and window in `0_data_caching.py`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 100 nonoverlapping 20 ms bins across a 2 s window. Spikes are directly binned once; there is no smoothing or later rebinning.

ii.
```python
BINSIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The trajectory says this exactly matches the method paper parameters.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the configured window and bin size relative to each trial's `stimOn_times`, rather than from an additional raw signal.

ii.
```python
ALIGN_TIME = 'stimOn_times'
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The agent interpreted the reference behavior interpolation grid as using bin right edges.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed 100-element sequence from -0.48 through +1.50 s is constructed and copied into every trial.

ii.
```python
BIN_TIMES = np.linspace(-0.5 + 0.02, 1.5, 100)
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. The code comment says these are the right edges used by the reference `get_behavior_per_interval` interpolation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The fixed time sequence has 100 entries corresponding by array index to the 100 neural bins, but labels bins by their right edges.

ii.
```python
inputs[:, 0, :] = BIN_TIMES[None, :]
outputs = np.empty((n_trials, 4, N_BINS), ...)
```

iii. The agent considered right-edge timestamps to be the reference sampling times and sanity-checked behavior around the stimulus bin.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`; every change in that value starts a new block.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy(dtype=float)
new_block[1:] = pleft[1:] != pleft[:-1]
block_id = np.cumsum(new_block) - 1
```

iii. The prior is constant within task blocks and no explicit block ID was used.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Trials in each block are numbered from zero before quality filtering, and the retained trials keep their original within-block positions. The scalar is repeated across all 100 time bins.

ii.
```python
for b in np.unique(block_id):
    idx = np.nonzero(block_id == b)[0]
    trial_in_block[idx] = np.arange(len(idx))
inputs[:, 1, :] = tinb[:, None].astype(np.float32)
```

iii. This preserves the animal's true position even when intervening trials are later excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials table's `choice` column.

ii.
```python
choice = trials['choice'].to_numpy()[tidx]
```

iii. The agent used the documented IBL choice convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. No-choice trials are removed. Positive choices become left/0 and negative choices become right/1; the per-trial value is broadcast over time.

ii.
```python
out_choice = (choice < 0).astype(np.int64)
outputs[:, 0, :] = out_choice[:, None]
```

iii. The trajectory states `+1 = left`, `-1 = right`, matching the requested coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials['probabilityLeft']`.

ii.
```python
pleft_t = pleft[tidx]
```

iii. This column directly represents the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to one decimal and mapped 0.2→0, 0.5→1, 0.8→2, then broadcast over time.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
out_prior = np.array([prior_map[round(float(p), 1)] for p in pleft_t])
outputs[:, 1, :] = out_prior[:, None]
```

iii. The mapping is explicitly required by the task; rounding makes floating-point lookup robust.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is the absolute value of wheel velocity returned by `SessionLoader`, using the loader's wheel timestamps.

ii.
```python
sl.load_wheel()
wheel_speed_t = sl.wheel['times'].to_numpy()
wheel_speed_v = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. The agent relied on the IBL loader's recommended position interpolation, filtering, and differentiation.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Absolute velocity is linearly interpolated separately for every trial at 100 timestamps from window start +20 ms through the window end, then discretized by session-wide tertiles.

ii.
```python
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
out_ws = discretize3(ws)
```

iii. The agent sought to reproduce `get_behavior_per_interval`; session-wise thresholds were justified because signal scales differ across rigs/sessions.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Thresholds are the 1/3 and 2/3 quantiles of all retained wheel samples in a session. Duplicate edges are removed and `searchsorted(..., side='right')` assigns integer categories.

ii.
```python
q = np.quantile(x, [1. / 3., 2. / 3.])
edges = np.unique(q)
return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The goal was equally populated, session-relative low/medium/high bins. The trajectory noticed that constant traces can leave a class empty and accepted that.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel samples use the same trial start/end and 100-element length as neural data, but are evaluated at neural-bin right edges rather than centers.

ii.
```python
interval_ends = interval_begs + N_BINS * BINSIZE
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
```

iii. The agent believed these right-edge timestamps exactly reproduced the method code and checked that mean movement rose after stimulus onset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It comes from `whiskerMotionEnergy` and `times` in the left camera motion-energy stream when available, otherwise the right camera.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sl.load_motion_energy(views=[view])
    whisker_t = me['times'].to_numpy()
    whisker_v = me['whiskerMotionEnergy'].to_numpy()
```

iii. The agent followed the reference's side-camera fallback and used the released whisker-pad motion-energy trace.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. No normalization or filtering is added. The trace is linearly interpolated per trial at bin right edges and discretized with session-wide tertiles.

ii.
```python
wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)
out_wm = discretize3(wm)
```

iii. The trajectory states that camera-specific arbitrary units motivate session-relative thresholds.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The same 1/3 and 2/3 session quantiles and duplicate-edge removal used for wheel speed assign categories.

ii.
```python
q = np.quantile(x, [1. / 3., 2. / 3.])
edges = np.unique(q)
return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The agent intended low/medium/high classes of roughly equal session-wide size.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It uses the same stimulus-aligned trial window and number of samples, interpolated at 20 ms-spaced right edges.

ii.
```python
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

iii. The agent considered this identical to reference behavioral alignment and sanity-checked the post-stimulus rise.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing spike sorting/probes are skipped; sessions without motion energy, neurons, or two usable trials are skipped. Trials with missing required fields, NaNs in behavioral windows, or incomplete trace coverage are dropped. Exceptions become recorded skip results rather than terminating the run.

ii.
```python
except Exception:
    return {'eid': eid, 'skip': 'exception: ' + ...}
if whisker_t is None:
    return {'eid': eid, 'skip': 'no whisker motion energy'}
if np.any(np.isnan(tv)):
    continue
```

iii. The agent prioritized a complete aligned example for every retained trial and documented that 14 sessions lacked whisker energy and one lacked behavior-covered trials.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large spike-sorting arrays, processing/binnning all spikes and trials, serializing the approximately 11.7 GB result, and downstream decoder PCA/training are the expensive operations. Sessions are processed in parallel.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
with ctx.Pool(processes=args.n_workers) as pool:
    for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
```

iii. The trajectory repeatedly notes the 567 GB source scale, large spike files, several-minute assembly/verification, and long PCA/training initialization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike-binning loop, per-trial behavior-interpolation loop, per-block numbering loop, prior mapping comprehension, and assembly-time region/subject index searches could be further vectorized. Searchsorted and bincount already vectorize the expensive work inside trial loops.

ii.
```python
for k in range(n_trials): ...
for b in np.unique(block_id): ...
out_prior = np.array([prior_map[round(float(p), 1)] for p in pleft_t])
```

iii. The trajectory called spike binning “vectorised” relative to the reference because it pre-sorts and uses `searchsorted`/`bincount`, while retaining a trial loop for variable slices.

## 10-c. What processing does the code repeat multiple times?

i. Each worker constructs a new ONE client and `BrainRegions`; behavior interpolation and discretization are separately repeated for wheel and whisker; output assembly repeatedly broadcasts per-trial constants; list membership/index scans repeat during final assembly.

ii.
```python
one = get_one()
br = BrainRegions()
ws, ws_good = bin_behavior(...)
wm, wm_good = bin_behavior(...)
out_ws = discretize3(ws)
out_wm = discretize3(wm)
```

iii. These repetitions arise from independent spawned workers and applying the same generic pipeline to two behavioral signals; the agent did not explicitly discuss them as waste.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges channel information and full cluster tables mainly to select labels/anatomy, computes and stores ancillary session metadata, sorts spike times even if loader output is already ordered, bins behavior for initially invalid trials before applying the final mask, and writes intermediate per-session pickles. Some loaded session fields and `lab/date/n_probes` do not affect decoder training.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
ws, ws_good = bin_behavior(..., interval_begs)
wm, wm_good = bin_behavior(..., interval_begs)
```

iii. The intermediate pickles were intentional for resumability; the trajectory did not identify other discarded work, focusing instead on tractability and validation.
