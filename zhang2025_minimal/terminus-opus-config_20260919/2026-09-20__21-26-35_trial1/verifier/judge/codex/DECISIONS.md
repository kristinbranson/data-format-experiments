# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the session list from the release-freeze CSV `/app/code/code_zhang2025/data/bwm_release.csv`, optionally restricts it with `/app/data/DATALIMIT_SUBSET.csv`, then processes each unique `eid`. Within each session it uses `SpikeSortingLoader` for probe-level spike sorting and `SessionLoader` for session-level trials, wheel, and motion-energy data. It constructs a default `ONE` client and relies on the cached Alyx/ONE metadata rather than traversing raw files directly for the session contents.

ii.
```python
BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT = '/app/data/DATALIMIT_SUBSET.csv'
...
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
if os.path.exists(DATALIMIT):
    sub = pd.read_csv(DATALIMIT)
    col = 'eid' if 'eid' in sub.columns else sub.columns[0]
    bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
...
eids = list(dict.fromkeys(bwm.eid.tolist()))
```

```python
def get_one():
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)
```

```python
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
sp, cl, ch = ssl.load_spike_sorting()
...
sl = SessionLoader(one=one, eid=eid)
trials, trials_mask = load_trials_and_mask(sl)
sl.load_wheel()
sl.load_motion_energy(views=[view])
```

iii. In the trajectory, the AI said it chose the 459 `eid`s from `bwm_release.csv` because that is what the Zhang caching script iterates over, and that `ONE` in default/remote mode could still work offline using cached REST responses. It explicitly justified not scanning the cache by hand for the main loads once it had confirmed the cached client could resolve trials, spikes, wheel, and camera data.

## 1-b. How are the data split into subjects (mice)?

i. The AI takes the subject identity from the rows of `bwm_release.csv` that correspond to each `eid`. During assembly it builds `subjects` incrementally in first-seen order and records `subject_idx` as the position of each session’s subject in that list.

ii.
```python
return {
    ...
    'subject': str(rows.subject.iloc[0]),
    ...
}
```

```python
subjects, regions_all = [], []
...
if res['subject'] not in subjects:
    subjects.append(res['subject'])
data['subject_idx'].append(subjects.index(res['subject']))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. In the trajectory, the AI treated the release CSV as the authoritative session table and therefore used its `subject` column instead of deriving subjects from path structure. The stated reason was that the release freeze already contains the same session metadata needed to mirror the reference session list.

## 1-c. How are the data split into sessions?

i. The AI treats each unique `eid` from `bwm_release.csv` as one session. It de-duplicates the CSV’s probe rows by `eid`, then runs one session job per `eid`.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))
...
jobs = [(eid, bwm[bwm.eid == eid]) for eid in eids
        if not os.path.exists(os.path.join(args.tmp, eid + '.pkl'))]
```

iii. In the trajectory, the AI repeatedly described the public BWM release freeze as the session list used by the reference code and therefore treated `eid` as the natural session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the `SessionLoader` trials table and treats each row as one trial. Later filtering is applied with a Boolean mask, but the original split into trials is the table row structure.

ii.
```python
if sess_loader.trials.empty:
    sess_loader.load_trials()
trials = sess_loader.trials
```

```python
tidx = np.nonzero(valid)[0]
...
n_trials = len(tidx)
```

iii. In the trajectory, the AI described the trials table as the source of trial boundaries and focused its work on masking and alignment rather than inventing a new trial segmentation.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a stricter “standard BWM trial mask” copied from `load_trials_and_mask`, then additionally requires wheel and whisker traces to cover the full decoding window. Its mask excludes trials with NaNs in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`, trials with reaction times outside `[0.08, 2.0]`, `choice == 0`, and trials with `feedback_times - goCue_times > 10 s`. It then intersects that with behavioural coverage masks from `bin_behavior`.

ii.
```python
nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
...
query = f'(firstMovement_times - stimOn_times < {min_rt})'
query += f' | (firstMovement_times - stimOn_times > {max_rt})'
if max_trial_len is not None:
    query += f' | (feedback_times - goCue_times > {max_trial_len})'
for event in nan_exclude:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
mask = ~trials.eval(query)
```

```python
ws, ws_good = bin_behavior(wheel_speed_t, wheel_speed_v, interval_begs)
wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)

valid = valid & ws_good & wm_good
```

iii. In the trajectory, the AI explicitly justified this as “the standard BWM trial mask (`load_trials_and_mask`) with `max_trial_len=10 s`, exactly as `prepare_data` calls it,” plus dropping trials whose behavioural traces do not span the decoder window. It framed that as copying the Zhang pipeline literally.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from spike times and spike cluster assignments loaded per probe, with the cluster table used to decide which units survive and what Beryl region each unit belongs to.

ii.
```python
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
sp, cl, ch = ssl.load_spike_sorting()
...
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```

```python
spike_times = np.concatenate([s['times'] for s in ms])
spike_clusters = np.concatenate([s['clusters'] for s in ms])
...
beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
label = clusters['label'].to_numpy()
```

iii. In the trajectory, the AI said it would “load spikes for all probes, merge, filter good units, Beryl regions,” and treated `spikes.times`, `spikes.clusters`, and cluster metadata as the needed ingredients.

## 2-b. How is the `neural` data processed?

i. The AI merges all probes within a session, renumbers cluster IDs, bins spikes into 20 ms bins over a 2 s stimulus-aligned window, and stores the result as float32 spike counts. It does not divide by bin width, so the final neural matrices are counts per 20 ms bin rather than firing rates in Hz.

ii.
```python
for cl, sp in zip(clusters_list, spikes_list):
    sp = dict(sp)
    sp['clusters'] = sp['clusters'] + cmax
    cmax = cl.index.max() + 1
    ms.append(sp)
    mc.append(cl)
```

```python
def bin_spikes(spike_times, spike_clusters, n_clusters, interval_begs):
    ...
    out = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    ...
    tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
    ...
    counts = np.bincount(flat, minlength=n_clusters * N_BINS)
    out[k] = counts.reshape(n_clusters, N_BINS)
    return out
```

```python
return {
    ...
    'neural': binned,                       # (n_trials, n_neurons, N_BINS) float32
    ...
}
```

iii. In the trajectory, the AI said it would “bin spikes in 20 ms bins over (-0.5, 1.5) s from `stimOn`” and later summarized the dataset as “population spike counts.” It justified the neuron restriction partly on memory grounds and did not claim to convert counts to rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1` and excludes Beryl regions `root` and `void`. Sessions with fewer than one surviving neuron are skipped entirely.

ii.
```python
beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
label = clusters['label'].to_numpy()
keep = (label >= 1) & ~np.isin(beryl, ['root', 'void'])
keep_ids = np.nonzero(keep)[0]
...
if n_neurons < MIN_NEURONS_PER_SESSION:
    return {'eid': eid, 'skip': f'only {n_neurons} well-isolated neurons'}
```

iii. In the trajectory, the AI justified this as using “well-isolated neurons” from the data paper and argued that retaining every Kilosort unit would be infeasible at roughly 164 GB. It also said grey-matter-only filtering matched the paper better and that dropping `root` was part of that grey-matter restriction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to `stimOn_times`. It converts each trial to a window beginning at `stimOn_times - 0.5 s`, and spike binning is performed relative to that interval so that the neural data span `[-0.5, 1.5]` seconds around stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
align = trials[ALIGN_TIME].to_numpy(dtype=float)
...
interval_begs = align + TIME_WINDOW[0]
...
tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
```

iii. In the trajectory, the AI repeatedly described the alignment choice as identical to the Zhang caching parameters: “align to `stimOn_times`, window (-0.5,1.5) s, 20 ms bins.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 20 ms bins over a 2 s window, producing 100 time bins. No additional temporal rebinning is applied beyond this initial binning/interpolation.

ii.
```python
BINSIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

```python
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
```

iii. In the trajectory, the AI justified this as exactly matching the `params` dictionary used by the Zhang reference code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI derives this input from the fixed time grid associated with the `stimOn_times` alignment. The absolute event variable used is `stimOn_times`, but the stored input values are the same fixed relative bin times for every trial.

ii.
```python
ALIGN_TIME = 'stimOn_times'
...
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

```python
align = trials[ALIGN_TIME].to_numpy(dtype=float)
...
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. In the trajectory, the AI described this input as “time since stim onset” coming from the stimulus-aligned 20 ms grid rather than a separate raw signal.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI does not compute a binary event channel or bin centres. Instead it uses a precomputed vector of bin right-edge times, from `-0.48` s to `1.5` s in 20 ms steps, and copies that same vector into every trial.

ii.
```python
# time stamp of each bin, relative to the alignment event.  These are the right edges of
# the bins, i.e. the sample times used by `get_behavior_per_interval` in the reference
# code (x_interp = linspace(beg + binsize, end, n_bins)).
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

```python
inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. In the trajectory, the AI justified the right-edge convention by saying it wanted to match `get_behavior_per_interval`, which samples behavioural traces at `beg + binsize` through `end`, rather than using bin centres.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI uses the same trial window and the same number of bins as the neural data, so the time input is aligned bin-for-bin to the neural matrices. Its timestamp convention is the right edge of each neural bin rather than the bin centre.

ii.
```python
interval_begs = align + TIME_WINDOW[0]
...
binned = bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs[tidx])
```

```python
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. In the trajectory, the AI explicitly connected the alignment of behaviour and input time channels to the same `stimOn`-anchored 100-bin grid used for neural spike counts.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. The AI derives trial-in-block from the trial-wise `probabilityLeft` sequence. A change in `probabilityLeft` starts a new block.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy(dtype=float)
new_block = np.ones(len(pleft), dtype=bool)
new_block[1:] = pleft[1:] != pleft[:-1]
block_id = np.cumsum(new_block) - 1
```

iii. In the trajectory, the AI described block identity as recoverable from the block prior, matching the logic of the reference pipeline.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. After identifying blocks from changes in `probabilityLeft`, the AI numbers trials within each block from zero upward. This is done on the full trial table before the final `valid` mask is applied, so filtered-out trials still advance the count.

ii.
```python
trial_in_block = np.zeros(len(pleft), dtype=np.int64)
for b in np.unique(block_id):
    idx = np.nonzero(block_id == b)[0]
    trial_in_block[idx] = np.arange(len(idx))
```

```python
tinb = trial_in_block[tidx]
...
inputs[:, 1, :] = tinb[:, None].astype(np.float32)
```

iii. In the trajectory, the AI said it wanted `trial number in block` as a per-trial variable and kept the counting in the original session trial order, not after filtering, to mirror the task structure.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The AI derives choice from the trials table column `choice`.

ii.
```python
choice = trials['choice'].to_numpy()[tidx]          # +1 = left, -1 = right
```

iii. In the trajectory, the AI said it had empirically verified the sign convention and concluded that `choice = +1` means left and `choice = -1` means right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI drops `choice == 0` trials in the trial mask, then maps the surviving values to decoder labels with left `(+1)` -> `0` and right `(-1)` -> `1`. It broadcasts the per-trial category across all 100 time bins.

ii.
```python
query += ' | (choice == 0)'
```

```python
out_choice = (choice < 0).astype(np.int64)          # left = 0, right = 1
...
outputs[:, 0, :] = out_choice[:, None]
```

iii. In the trajectory, the AI justified this by noting that it had checked the task convention directly and confirmed that rightward choices should be encoded as `1` for the downstream decoder task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The AI derives prior probability of left from the trials table column `probabilityLeft`.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy(dtype=float)
...
pleft_t = pleft[tidx]
```

iii. In the trajectory, the AI described `probabilityLeft` as the block prior variable and used it both to identify blocks and to create the categorical output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2`, `0.5`, and `0.8` to decoder classes `0`, `1`, and `2`, then broadcasts that per-trial label across all 100 time bins.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
out_prior = np.array([prior_map[round(float(p), 1)] for p in pleft_t], dtype=np.int64)
...
outputs[:, 1, :] = out_prior[:, None]
```

iii. In the trajectory, the AI said it was following the decoder task’s required `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The AI derives wheel speed from the wheel time series loaded by `SessionLoader.load_wheel()`, specifically from the loader-produced `velocity` trace and its timestamps.

ii.
```python
sl.load_wheel()
wheel_speed_t = sl.wheel['times'].to_numpy()
wheel_speed_v = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. In the trajectory, the AI justified this as following the reference behaviour processing: load the wheel through `SessionLoader`, then use the absolute value of the derived velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI uses the velocity already computed by `SessionLoader`, takes its absolute value, interpolates it onto each trial’s 100-bin stimulus-aligned grid, rejects trials whose trace does not cover the full window or contains NaNs, and then discretizes the resulting session-wide values.

ii.
```python
wheel_speed_t = sl.wheel['times'].to_numpy()
wheel_speed_v = np.abs(sl.wheel['velocity'].to_numpy())
...
ws, ws_good = bin_behavior(wheel_speed_t, wheel_speed_v, interval_begs)
```

```python
if np.any(np.isnan(tv)):
    continue
if abs(interval_begs[k] - tt[0]) > BINSIZE:
    continue
if abs(interval_ends[k] - tt[-1]) > BINSIZE:
    continue
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

iii. In the trajectory, the AI said it was matching `get_behavior_per_interval` by interpolating behavioural traces onto the bin right edges, and it relied on `SessionLoader` to have already produced the filtered/interpolated wheel velocity.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI discretizes wheel speed into three session-specific tertile bins using the 1/3 and 2/3 quantiles of the full session trace values across all kept trials and bins.

ii.
```python
def discretize3(x):
    q = np.quantile(x, [1. / 3., 2. / 3.])
    edges = np.unique(q)
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

```python
out_ws = discretize3(ws)
```

iii. In the trajectory, the AI justified session-wise tertiles by arguing that wheel scale varies across rigs and that “low/medium/high for this mouse in this session” is more comparable than a global threshold.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI samples wheel speed on the same 100-bin stimulus-aligned grid used for neural data. As with the time input, the samples correspond to the right edge of each 20 ms neural bin.

ii.
```python
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

```python
binned = bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs[tidx])
```

iii. In the trajectory, the AI explicitly said its behavioural alignment matched the Zhang helper that interpolates to the bin right edges and later reported a sanity check that wheel speed rose after stimulus onset, which it used as evidence that the alignment was correct.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The AI derives whisker motion energy from the motion-energy trace and frame times of a side camera loaded through `SessionLoader`. It prefers the left camera and falls back to the right camera if needed.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        me = sl.motion_energy[cam]
        whisker_t = me['times'].to_numpy()
        whisker_v = me['whiskerMotionEnergy'].to_numpy()
        break
    except Exception:
        continue
```

iii. In the trajectory, the AI said it would use whichever whisker motion-energy stream was available, with a left-camera preference, because whisker motion energy was a required decoder output.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI takes the released whisker-motion-energy trace as-is, interpolates it onto each trial’s 100-bin aligned grid, rejects incomplete or NaN-containing trial windows, and then discretizes the resulting values session-wise.

ii.
```python
wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)
```

```python
if np.any(np.isnan(tv)):
    continue
...
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

iii. In the trajectory, the AI justified this as matching the released motion-energy signal and the same interpolation helper used for wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI discretizes whisker motion energy into three session-specific tertile bins using the same `discretize3` helper used for wheel speed.

ii.
```python
def discretize3(x):
    q = np.quantile(x, [1. / 3., 2. / 3.])
    edges = np.unique(q)
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

```python
out_wm = discretize3(wm)
```

iii. In the trajectory, the AI justified this by saying whisker motion energy is in camera-dependent arbitrary units, so per-session tertiles preserve within-session low/medium/high semantics.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI interpolates whisker motion energy onto the same stimulus-aligned 100-bin grid used for neural data, again using the right edge of each neural bin as the sample time.

ii.
```python
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

```python
binned = bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs[tidx])
```

iii. In the trajectory, the AI used the same post-stimulus rise sanity check for whisker energy as for wheel speed to support that this alignment was reasonable.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or unusable data primarily by dropping trials or skipping sessions. Trials are dropped if key trial fields are NaN, if behavioural traces contain NaNs or fail to cover the full decoding window, or if `choice == 0`. Sessions are skipped when spike sorting is absent, whisker motion energy cannot be loaded, too few neurons survive, or too few valid trials remain.

ii.
```python
if sp is None or len(sp) == 0:
    continue
...
if len(spikes_list) == 0:
    return {'eid': eid, 'skip': 'no spike sorting'}
```

```python
if whisker_t is None:
    return {'eid': eid, 'skip': 'no whisker motion energy'}
...
if valid.sum() < MIN_TRIALS_PER_SESSION:
    return {'eid': eid, 'skip': f'only {int(valid.sum())} trials with complete behaviour'}
```

```python
if np.any(np.isnan(tv)):
    continue
```

iii. In the trajectory, the AI explicitly described missing whisker data as a reason to skip a session because whisker motion energy was a required decoder output, and it described the behavioural coverage checks as a direct copy of the reference helper.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s own trajectory identifies the expensive parts as loading large spike-sorting datasets from disk and then processing the full 459-session conversion. The code structure also suggests that per-session spike loading, cluster merging, and spike binning dominate runtime.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```

```python
binned = bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs[tidx])
```

```python
with ctx.Pool(processes=args.n_workers) as pool:
    for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
```

iii. In the trajectory, the AI repeatedly emphasized dataset scale and file I/O, described spike sorting as the bottleneck, and used multiprocessing specifically because the session conversions were independent.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already partially vectorized spike binning by precomputing search ranges and using one `np.bincount` per trial, but several Python-level loops remain. The per-trial loop in `bin_behavior`, the per-trial loop in `bin_spikes`, the per-block loop for `trial_in_block`, and the assembly-time loops over trials, subjects, and regions could all be vectorized or replaced with more efficient indexed structures.

ii.
```python
for k in range(n_trials):
    b, e = beg[k], end[k]
    ...
    counts = np.bincount(flat, minlength=n_clusters * N_BINS)
    out[k] = counts.reshape(n_clusters, N_BINS)
```

```python
for k in range(n_trials):
    tt = target_times[ib[k]:ie[k]]
    tv = target_vals[ib[k]:ie[k]]
    ...
    vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

```python
for b in np.unique(block_id):
    idx = np.nonzero(block_id == b)[0]
    trial_in_block[idx] = np.arange(len(idx))
```

iii. In the trajectory, the AI explicitly said it was vectorizing the spike binning helper because of the scale of the dataset, but it did not claim to eliminate the remaining trial-wise interpolation or assembly loops.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several lightweight operations: it repeatedly slices `bwm` by `eid` when building jobs, repeatedly performs linear membership/index lookups when building `subjects` and `brain_regions`, and separately scans similar trial windows for wheel and whisker interpolation. These are not catastrophic, but they are repeated work that the reference solution avoids or handles more directly.

ii.
```python
jobs = [(eid, bwm[bwm.eid == eid]) for eid in eids
        if not os.path.exists(os.path.join(args.tmp, eid + '.pkl'))]
```

```python
if res['subject'] not in subjects:
    subjects.append(res['subject'])
data['subject_idx'].append(subjects.index(res['subject']))
for r in res['regions']:
    if r not in regions_all:
        regions_all.append(r)
data['brain_region_idx'].append(
    np.array([regions_all.index(r) for r in res['regions']], dtype=np.int64))
```

```python
ws, ws_good = bin_behavior(wheel_speed_t, wheel_speed_v, interval_begs)
wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)
```

iii. The trajectory does not contain a direct justification for these repeated passes. The AI’s justification focused on keeping the code straightforward while parallelizing at the session level.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does some extra processing and bookkeeping that are not needed by downstream decoding: it writes a temporary pickle for every session, stores lab/date/probe-count/trial-count metadata in those temporary records, and computes rich metadata fields such as `bin_times_s`, `spike_sorting`, and long descriptive strings that the decoder does not use. It also processes sessions that are later skipped, because the skip decision happens only after partial loading and preprocessing.

ii.
```python
TMP_DIR = '/app/work/sessions'
...
with open(os.path.join(args.tmp, res['eid'] + '.pkl'), 'wb') as f:
    pickle.dump(res, f, protocol=4)
```

```python
return {
    'eid': eid,
    'skip': None,
    ...
    'lab': str(rows.lab.iloc[0]),
    'date': str(rows.date.iloc[0]),
    'n_probes': int(len(rows)),
    'n_trials_total': int(len(trials)),
}
```

```python
'metadata': {
    ...
    'bin_times_s': [float(t) for t in BIN_TIMES],
    'spike_sorting': 'IBL pykilosort (Kilosort 2.5 with IBL additions)',
    'neuron_selection': (...),
    'trial_selection': (...),
    ...
}
```

iii. In the trajectory, the AI justified the temporary per-session pickles as part of a parallel, resumable workflow and later justified the extra metadata because it wanted the final file to validate cleanly and document its choices. It did not claim those fields were required by the downstream decoder itself.
