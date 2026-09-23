# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the human reference solution's Alyx-backed `ONE.search(...)` flow as the primary index. Instead, it read `sessions.pqt` and `bwm_release.csv` directly, reconstructed absolute session paths, rebuilt the ONE `datasets` cache by walking each session's `alf/` tree, then used a local-mode ONE client plus `SessionLoader` / `SpikeSortingLoader` to load per-session data. Probe ids came from `bwm_release.csv`, not `one.eid2pid`.

ii. 
```python
def get_session_table():
    sess = pd.read_parquet(SESSIONS_PQT)
    sess.index = sess.index.astype(str)
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    ...
    return bwm, sess, paths
```

```python
def build_dataset_table(session_paths_map):
    for eid, sp in session_paths_map.items():
        for dirpath, _, filenames in os.walk(os.path.join(sp, 'alf')):
            ...
    d = pd.DataFrame(rows, columns=['eid', 'id', 'file_size', 'hash',
                                    'default_revision', 'qc', 'exists', 'rel_path'])
```

```python
def make_one(session_paths_map):
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')
    one.load_cache(tables_dir=f'{ROOT}/Brainwidemap')
    one._cache['datasets'] = build_dataset_table(session_paths_map)
    return one
```

```python
sl = SessionLoader(one=one, eid=eid)
...
sp, cl, n_tot = load_spiking_data(one, r.pid, eid, r.probe_name)
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as a workaround for stale release tables: the shipped ONE cache allegedly pointed to unrevised trial paths and returned incomplete trial tables, so it rebuilt the dataset table from the filesystem and avoided `eid2pid`, which it said needed network access.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `bwm_release.csv` rows matched to each `eid`. After conversion, the output `subjects` list is the sorted unique subject names and `subject_idx` indexes sessions into that list.

ii. 
```python
sub = bwm[bwm.eid == eid]
info['subject'] = sub.subject.iloc[0]
```

```python
subjects = sorted({r['subject'] for r in kept})
subj_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. The notes say the release table already contains subject identity, so no path parsing beyond locating sessions is needed.

## 1-c. How are the data split into sessions?

i. A session is one unique `eid` from `bwm_release.csv`. The script iterates `list(bwm.eid.unique())`, converts one session at a time with `convert_session`, and stores one session entry in each top-level list.

ii. 
```python
eids = list(bwm.eid.unique())
...
for res, info in ex.map(_worker, rest, chunksize=1):
    results.append(res)
```

```python
'neural': [r['neural'] for r in kept],
'input': [r['input'] for r in kept],
'output': [r['output'] for r in kept],
```

iii. The AI treated the release's session identifier as the atomic unit, consistent with the notes' description of one conversion per `eid`.

## 1-d. How are the data split into trials?

i. Trials come from the session trials table loaded by `SessionLoader`. The script applies a trial mask, then keeps the surviving rows of `trials` as the trial axis for neural, input, and output arrays.

ii. 
```python
sl = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(sl)
...
trials_sel = trials[mask]
align_times = trials_sel[ALIGN_TIME].to_numpy()
```

```python
for k in range(ntrials):
    neural_list.append(binned[k])
    ...
    input_list.append(inp)
    ...
    output_list.append(out)
```

iii. The justification is implicit: the trials table is already organized one row per trial, so the AI used it directly after masking.

## 1-e. How are trials filtered based on quality controls?

i. The AI used a multi-stage mask. First it applied a transcription of `load_trials_and_mask`: reaction time 0.08-2.0 s, no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, `choice != 0`, and `feedback_times - goCue_times <= 10 s`. Then it dropped trials whose wheel or whisker traces did not fully cover the window or contained NaNs. Finally it dropped trials whose entire neural population was silent for the full 2 s window.

ii. 
```python
if nan_exclude == 'default':
    nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                   'firstMovement_times', 'feedbackType']
...
if max_trial_len is not None:
    query += f' | (feedback_times - goCue_times > {max_trial_len})'
...
if exclude_nochoice:
    query += ' | (choice == 0)'
mask = ~sess_loader.trials.eval(query)
```

```python
wheel_vals, wheel_good = bin_behavior(*traces['wheel-speed'], align_times)
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
beh_good = wheel_good & whisk_good
```

```python
neural_covered = binned.sum(axis=(1, 2)) > 0
keep = beh_good & neural_covered
```

iii. The notes justify the first mask as a transcription of the BWM/reference code mask, the behaviour mask as required because the target format forbids NaNs, and the no-spike mask as protection against recording dropouts that otherwise create all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays are derived from `spikes['times']` and `spikes['clusters']` loaded per probe via `SpikeSortingLoader`. The cluster table contributes QC labels and region acronyms, but the actual binned neural matrix is built from spike times plus cluster ids.

ii. 
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

```python
st = spikes['times'][finite]
sc = spikes['clusters'][finite]
...
binned = bin_spiking_data(st, sc, n_clusters, align_times)
```

iii. The AI's notes explicitly say this follows the reference loading path for ephys data.

## 2-b. How is the `neural` data processed?

i. The AI kept well-isolated spike-sorted units, merged probes within a session, removed NaN spike times, sorted spikes by time, then binned spikes into 100 non-overlapping 20 ms bins from -0.5 s to +1.5 s around stimulus onset. Unlike the human reference solution, it kept raw spike counts per bin rather than dividing by bin width to convert to Hz.

ii. 
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
...
finite = np.isfinite(spikes['times'])
st = spikes['times'][finite]
sc = spikes['clusters'][finite]
order = np.argsort(st, kind='stable')
st, sc = st[order], sc[order]
```

```python
idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
flat = c[keep].astype(np.int64) * NBINS + idx[keep]
counts = np.bincount(flat, minlength=n_clusters * NBINS)
out[k] = counts.reshape(n_clusters, NBINS)
```

```python
'neural_units': 'spike counts per 20 ms bin',
```

iii. In `CONVERSION_NOTES.md`, the AI justified counts by saying the reference caching code stores raw counts and that z-scoring belongs in the model stage, not the dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filtered clusters by `clusters['label'] >= 1.0`, i.e. well-isolated units, and dropped sessions with fewer than 5 surviving neurons. It did not implement the human reference solution's additional `Beryl != 'void'` exclusion.

ii. 
```python
QC_LABEL = 1.0
...
iok = clusters_labeled['label'] >= qc
selected_clusters = clusters_labeled[iok]
...
selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)
```

```python
MIN_NEURONS = 5
...
if n_clusters < MIN_NEURONS:
    info['skip_reason'] = (f'fewer than {MIN_NEURONS} well-isolated neurons '
                           f'({n_clusters})')
    return None, info
```

iii. The notes justify `label == 1` as reproducing the data paper's well-isolated neuron count and justify `MIN_NEURONS = 5` by analogy to the BWM region-level inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For trial `k`, the bin window starts at `stimOn_times[k] - 0.5` and ends at `stimOn_times[k] + 1.5`, so time zero is stimulus onset.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
align_times = trials_sel[ALIGN_TIME].to_numpy()
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
```

iii. The AI says this matches both the decoder instructions and the Zhang reference parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms with 100 bins over a 2 s window. No temporal rebinning beyond the initial 20 ms binning is applied.

ii. 
```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

```python
idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```

iii. The notes explicitly tie this to the reference `binsize=0.02`, `time_window=(-0.5, 1.5)` settings.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial alignment event `trials.stimOn_times`, together with the fixed 20 ms bin grid. The actual values are synthetic bin-centre times relative to stimulus onset rather than a raw column copied from file.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
...
align_times = trials_sel[ALIGN_TIME].to_numpy()
```

```python
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
inp[0] = tvec
```

iii. The notes describe this as a task-required decoder input constructed on the same trial window as the neural data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computed a fixed vector of neural-bin centres, from -0.49 s to +1.49 s in 20 ms steps, and copied that same vector into every trial.

ii. 
```python
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
...
inp = np.empty((2, NBINS), dtype=np.float32)
inp[0] = tvec
```

iii. The justification in the notes is that this input should represent time on the neural bin grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned as the centre of the same 20 ms bins used for neural spike counts. The neural bins are defined by trial-specific `begs`/`ends`; `tvec` is the relative bin-centre coordinate system for those bins.

ii. 
```python
begs = align_times + TIME_WINDOW[0]
...
idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```

```python
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
inp[0] = tvec
```

iii. The AI's rationale is that the decoder input should share the neural trial time base exactly.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the raw trials table. Block boundaries are inferred whenever `probabilityLeft` changes.

ii. 
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    ...
    same = (p[1:] == p[:-1])
    newblock[1:] = ~same
```

```python
tnib_all, block_id_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. The notes justify this by saying there is no explicit block id in the trials table, so block structure must be reconstructed from `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI detects changes in `probabilityLeft`, assigns a block id by cumulative sum, then counts trials from 0 upward within each block. This is computed on the full session before trial exclusion and then masked down to surviving trials. The resulting scalar is broadcast across all 100 time bins of a trial.

ii. 
```python
block_id = np.cumsum(newblock) - 1
out = np.zeros(len(p), dtype=np.int64)
for b in np.unique(block_id):
    m = block_id == b
    out[m] = np.arange(m.sum())
return out, block_id
```

```python
tnib_all, block_id_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
tnib = tnib_all[mask]
...
inp[1] = tnib_keep[k]
```

iii. The notes explicitly justify computing it before exclusion so it reflects the animal's true block position rather than the position among retained trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column in the trials table.

ii. 
```python
choice_raw = trials_keep['choice'].to_numpy()
```

iii. The notes state that IBL encodes left choices as `+1`, right choices as `-1`, and `0` for no response.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI filtered out no-response trials through the trial mask, then recoded the surviving values so left becomes 0 and right becomes 1 using `(choice_raw < 0)`.

ii. 
```python
if exclude_nochoice:
    query += ' | (choice == 0)'
```

```python
choice_raw = trials_keep['choice'].to_numpy()
choice = (choice_raw < 0).astype(np.int8)
```

iii. The notes justify the mapping by saying it was verified against `feedbackType`, `contrastLeft`, and `contrastRight`, and it matches the task spec `left = 0, right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. 
```python
pleft = trials_keep['probabilityLeft'].to_numpy()
```

iii. The notes say this variable is the task's block prior and only takes values 0.2, 0.5, and 0.8.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI recoded `probabilityLeft` from `{0.2, 0.5, 0.8}` to categorical labels `{0, 1, 2}` and rejected a session if any other value appeared.

ii. 
```python
prior = np.full(ntrials, -1, dtype=np.int8)
for val, lab in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(pleft, val)] = lab
if np.any(prior < 0):
    info['skip_reason'] = f'unexpected probabilityLeft values {np.unique(pleft)}'
    return None, info
```

iii. The notes justify this as the task-specified categorical coding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the session wheel stream that `SessionLoader.load_wheel()` constructs from raw wheel timestamps and positions. The script uses the absolute value of the resulting wheel velocity.

ii. 
```python
sess_loader.load_wheel()
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The AI's notes say this mirrors the reference loader for wheel behaviour.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` supplies the resampled, filtered wheel velocity; the AI takes its absolute value, linearly interpolates it onto the per-trial decoder time base, then discretizes all retained wheel samples of the session into within-session tertiles.

ii. 
```python
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

```python
wheel_vals, wheel_good = bin_behavior(*traces['wheel-speed'], align_times)
...
wheel_lab, wheel_edges = discretize_tertiles(wheel_vals)
```

iii. The notes justify tertiles because wheel-speed scale varies by session and fixed thresholds would produce poorly populated classes.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI flattened the session's retained wheel-speed values across all trials and time bins, took the 1/3 and 2/3 quantiles, and used `np.digitize` to assign categories 0, 1, 2. It included a fallback for degenerate nearly constant traces.

ii. 
```python
flat = values.ravel()
edges = np.quantile(flat, [1. / 3., 2. / 3.])
...
labels = np.digitize(values, edges).astype(np.int8)
return labels, edges
```

iii. The AI justified equal-occupancy bins as giving consistent low/medium/high semantics within session and exactly 1/3 chance level for balanced accuracy.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI aligned wheel speed to the same trial windows as the neural data, but sampled it at the right edge of each 20 ms neural bin rather than at the bin centre.

ii. 
```python
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
...
x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. The notes explicitly say this was copied from `ibl_data_utils.get_behavior_per_interval`, which samples continuous behaviour at bin right edges.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera` whisker ROI motion energy when available, otherwise `rightCamera`, along with the corresponding camera timestamps.

ii. 
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        whisker = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
        out['whisker-camera'] = view
        break
```

iii. The notes justify left-first/right-fallback as matching the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI used the released motion-energy trace without extra filtering or normalization, linearly interpolated it onto the per-trial decoder time base, then discretized the retained session values into tertiles.

ii. 
```python
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
...
whisk_lab, whisk_edges = discretize_tertiles(whisk_vals)
```

iii. The notes justify tertiles because whisker motion energy is in camera-dependent arbitrary units, so fixed thresholds would not transfer across sessions.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same within-session tertile procedure as wheel speed: flatten all retained samples, compute the 1/3 and 2/3 quantiles, and digitize into low/medium/high classes.

ii. 
```python
flat = values.ravel()
edges = np.quantile(flat, [1. / 3., 2. / 3.])
...
labels = np.digitize(values, edges).astype(np.int8)
```

iii. The justification is the same session-specific scale argument given in the notes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. As with wheel speed, whisker motion energy is aligned to each trial's `stimOn_times` window but sampled at the right edge of each neural bin rather than the bin centre.

ii. 
```python
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
...
x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. The notes say this follows the behaviour interpolation routine in the reference code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled missing or problematic data by dropping bad trials or whole sessions rather than imputing most issues. It rebuilt the ONE dataset table to fix stale cache metadata, rejected sessions without usable whisker data, dropped trials with uncovered or NaN behaviour traces, dropped spikes with NaN times, dropped all-zero-neural trials as presumed recording dropouts, and dropped sessions with fewer than 5 neurons or fewer than 2 usable trials.

ii. 
```python
one._cache['datasets'] = build_dataset_table(session_paths_map)
```

```python
if whisker is None:
    raise RuntimeError('no whisker motion energy available')
```

```python
if np.isnan(vv).any():
    continue
if np.abs(begs[k] - tt[0]) > BINSIZE:
    continue
if np.abs(ends[k] - tt[-1]) > BINSIZE:
    continue
```

```python
finite = np.isfinite(spikes['times'])
...
neural_covered = binned.sum(axis=(1, 2)) > 0
...
if n_clusters < MIN_NEURONS:
    ...
if keep.sum() < 2:
    ...
```

iii. The notes justify these as necessary to prevent malformed trials, stale-table loading errors, and recording-dropout artifacts from entering the final decoder dataset.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified dataset-table reconstruction, behaviour loading/interpolation, and especially spike-sorting I/O as the expensive steps. In the per-session conversion itself, loading spike sorting is the dominant cost, with behaviour loading and spike binning secondary.

ii. 
```python
for dirpath, _, filenames in os.walk(os.path.join(sp, 'alf')):
```

```python
spikes, clusters, channels = loader.load_spike_sorting()
```

```python
wheel_vals, wheel_good = bin_behavior(*traces['wheel-speed'], align_times)
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
...
binned = bin_spiking_data(st, sc, n_clusters, align_times)
```

iii. In the notes, the AI explicitly says the expensive part is spike-sorting file I/O and contrasts that with a relatively fast vectorized binning step.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Even after the AI's optimizations, several loops remain obvious vectorization targets: the per-trial loop in `bin_spiking_data`, the per-trial interpolation loop in `bin_behavior`, the per-block loop in `trial_number_in_block`, and the per-trial packing loop that appends `neural`, `input`, and `output` arrays.

ii. 
```python
for k in range(ntrials):
    a, b = i0[k], i1[k]
    ...
    counts = np.bincount(flat, minlength=n_clusters * NBINS)
```

```python
for k in range(ntrials):
    tt = target_times[idxs_beg[k]:idxs_end[k]]
    ...
    values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

```python
for b in np.unique(block_id):
    m = block_id == b
    out[m] = np.arange(m.sum())
```

```python
for k in range(ntrials):
    neural_list.append(binned[k])
    ...
    output_list.append(out)
```

iii. The notes emphasize that the AI already vectorized the biggest hot path relative to the original reference, but these remaining loops still exist in the final code.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats two kinds of work. First, it interpolates two behaviour streams separately with nearly identical logic via two calls to `bin_behavior`. Second, it computes some values even when they are only used for diagnostics or metadata, such as `block_id_all`, extensive `info[...]` counters, and per-session edge lists.

ii. 
```python
wheel_vals, wheel_good = bin_behavior(*traces['wheel-speed'], align_times)
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
```

```python
tnib_all, block_id_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

```python
info['n_rt_short'] = int(np.nansum(rt < 0.08))
info['n_rt_long'] = int(np.nansum(rt > 2.0))
...
info['wheel_edges'] = wheel_edges.tolist()
info['whisker_edges'] = whisk_edges.tolist()
```

iii. The notes show that the AI deliberately kept a large amount of extra accounting and diagnostics for validation, even when those values are not needed to produce the core `neural` / `input` / `output` arrays.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The most obvious unnecessary or downstream-discarded work is diagnostic bookkeeping rather than core conversion: it computes `block_id_all` even though only `tnib_all` is needed for the final dataset, adds `cl['pid']` to cluster tables but never uses it downstream, builds large `info` dictionaries and per-session tertile edges only for metadata/documentation, and optionally constructs `_plotdata` solely for plots.

ii. 
```python
tnib_all, block_id_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

```python
cl = cl.copy()
cl['pid'] = r.pid
```

```python
info['wheel_edges'] = wheel_edges.tolist()
info['whisker_edges'] = whisk_edges.tolist()
...
'session_info': session_info,
```

```python
if show_processing:
    result['_plotdata'] = {
        ...
        'block_id': block_id_all, 'tnib_all': tnib_all,
    }
```

iii. The AI's notes explicitly frame much of this as sanity-check and documentation support rather than decoder-required processing.
