# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates sessions from the release freeze CSV `/app/code/code_zhang2025/data/bwm_release.csv`, optionally restricts them with `/app/data/DATALIMIT_SUBSET.csv`, then loads each session's trials and behavior with `SessionLoader` and each probe's spikes with `SpikeSortingLoader` through a local ONE cache. This differs from the human reference, which discovers sessions from the ONE release index via `one.search`.

ii.
```python
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT_SUBSET = '/app/data/DATALIMIT_SUBSET.csv'
```

```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
if os.path.exists(DATALIMIT_SUBSET):
    subset = pd.read_csv(DATALIMIT_SUBSET)
    col = 'eid' if 'eid' in subset.columns else subset.columns[0]
    bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]
```

```python
sess_loader = SessionLoader(one=one, eid=eid)
spikes, clusters = load_spikes_and_clusters(one, eid, pids, probe_names)
```

iii. In the trajectory summary, the agent says it chose offline ONE access plus `bwm_release.csv` because probe IDs come from the BWM release freeze and because it wanted to avoid forcing network re-authentication. It also says all 459 released sessions were attempted and probes within a session were merged.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken directly from the `subject` column of `bwm_release.csv`. Sessions are grouped by `eid`, and after conversion the output `subjects` list is the sorted set of subject names while `subject_idx` maps each session to its subject.

ii.
```python
for eid, rows in bwm_df.groupby('eid', sort=False):
    sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                     list(rows.pid), list(rows.probe_name)))
```

```python
subjects = sorted({r['subject'] for r in results})
subject_lookup = {s: i for i, s in enumerate(subjects)}
```

iii. The trajectory justification is minimal here. The agent's final summary says the converted dataset contains 136 mice, which implies it trusted the release metadata and did not derive subjects from paths or filenames.

## 1-c. How are the data split into sessions?

i. The agent treats one `eid` as one session. It groups the release CSV by `eid`, keeps the associated subject, lab, probe IDs, and probe names, and processes each grouped tuple as one session.

ii.
```python
for eid, rows in bwm_df.groupby('eid', sort=False):
    sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                     list(rows.pid), list(rows.probe_name)))
```

```python
for i, sess in enumerate(sessions):
    _report(i, len(sessions), _safe_process_session(sess), results, failures)
```

iii. The trajectory summary explicitly says "All 459 released sessions were attempted", so the agent's justification was that the release freeze already defines the session set.

## 1-d. How are the data split into trials?

i. The agent uses the IBL trials table as the source of trial boundaries. Each row of `sess_loader.trials` is treated as one trial, and per-trial time windows are constructed around that row's `stimOn_times`.

ii.
```python
if sess_loader.trials.empty:
    sess_loader.load_trials()
trials = sess_loader.trials
```

```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
```

iii. The agent did not give a separate justification beyond saying it was reproducing the methods-paper preprocessing and aligning trials to stimulus onset.

## 1-e. How are trials filtered based on quality controls?

i. Trials are first filtered by a `load_trials_and_mask` reimplementation: reaction time must be between `0.08` and `2.0` s, `feedback_times - goCue_times` must be at most `10` s, specified trial fields cannot be NaN, and `choice == 0` is excluded. The agent then further requires both wheel and whisker traces to cover the full trial window and contain no NaNs. Sessions with fewer than two surviving trials are dropped.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in NAN_EXCLUDE:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
```

```python
wheel, wheel_ok = bin_behavior(wheel_times, wheel_vals, interval_begs, interval_ends)
whisker, whisker_ok = bin_behavior(whisker_times, whisker_vals, interval_begs, interval_ends)
keep = trials_mask & wheel_ok & whisker_ok
if keep.sum() < MIN_TRIALS_PER_SESSION:
    raise RuntimeError(f'only {keep.sum()} trials left after filtering')
```

iii. The trajectory summary says the agent intentionally followed `prepare_data` and `load_trials_and_mask` from the methods-paper code, then added behavior-coverage filtering because the target decoder format cannot contain NaNs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from the spike sorting outputs `spikes['times']` and `spikes['clusters']` from every probe of a session. Cluster metadata are also loaded to attach region labels, but the neural matrix itself is built from spike times and cluster assignments.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

```python
spikes = {k: np.concatenate([s[k] for s in merged_spikes])
          for k in ('times', 'clusters')}
```

iii. The trajectory summary says the agent followed the methods-paper path that bins "all neurons, sorted by Kilosort, from each session", so spike times and cluster IDs were treated as the defining neural inputs.

## 2-b. How is the `neural` data processed?

i. The agent merges all probes within a session, renumbers cluster IDs across probes, sorts spikes by time, and bins spike counts into 100 non-overlapping 20 ms bins per trial. It keeps the result as spike counts rather than dividing by the bin width to convert to Hz.

ii.
```python
spikes['clusters'] = spikes['clusters'] + cluster_max
sort_idx = np.argsort(spikes['times'], kind='stable')
spikes = {k: v[sort_idx] for k, v in spikes.items()}
```

```python
bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE).astype(np.int64)
counts = np.bincount(rows * N_BINS + bin_idx[keep], minlength=n_units * N_BINS)
binned[trial] = counts.reshape(n_units, N_BINS)
```

```python
neural.append(binned[i].astype(np.float32))
```

iii. The trajectory summary explicitly says the output contains a per-trial "spike-count matrix" and that this followed the methods-paper code path using all spike-sorted units.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply single-unit QC such as `label >= 1` and does not exclude `void` regions. It keeps all spike-sorted units that appear in `spike_clusters`, which effectively drops only units that fired zero spikes in the session because `unit_ids = np.unique(spike_clusters)`.

ii.
```python
"""Mirrors `load_spiking_data` (with qc=None, i.e. all units) followed by
`merge_probes` in the reference code."""
```

```python
unit_ids = np.unique(spike_clusters)
n_units = len(unit_ids)
```

iii. The trajectory summary says the agent deliberately chose "all spike-sorted units, no QC selection" because `prepare_data -> load_spiking_data` uses `qc=None`, and it justified this by citing the methods paper's use of all Kilosort-sorted neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each trial, the binning window is `[stimOn_times - 0.5, stimOn_times + 1.5)`, so the 100 spike-count bins are defined relative to stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
```

```python
bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE).astype(np.int64)
```

iii. The trajectory summary says alignment was set to stimulus onset because that is exactly the `params` configuration in `src/0_data_caching.py`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins over a 2 s window, giving 100 bins per trial. The agent does no temporal smoothing or rebinning beyond the initial binning.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The trajectory summary says this 20 ms, 100-bin configuration was chosen because it matches the methods-paper caching code and is needed for the time-varying decoder outputs.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is derived from the trial alignment event `stimOn_times`. The actual value sequence is then defined from the fixed bin centers of the chosen window.

ii.
```python
ALIGN_TIME = 'stimOn_times'
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
```

```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
```

iii. The trajectory summary states that the first input is "time from stimulus onset (bin centres, -0.49…1.49 s)", so the agent viewed this as directly induced by the alignment event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No raw signal is transformed. The agent simply constructs a fixed vector of 100 bin centers from `-0.49` to `1.49` seconds and reuses it for every trial.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
inp = np.empty((2, N_BINS), dtype=np.float32)
inp[0] = bin_centres
```

iii. The trajectory summary explicitly describes this input as "bin centres", with no additional justification beyond matching the chosen decoding grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time-since-stimulus input uses the same 100-bin trial grid as the neural data. The neural counts are binned over the same `TIME_WINDOW` and `BINSIZE`, and the input stores the center of those bins.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
inp[0] = bin_centres
```

```python
binned, unit_ids = bin_spikes(spikes['times'], spikes['clusters'],
                              interval_begs[keep], interval_ends[keep])
```

iii. The agent's stated justification is that the dataset is aligned to stimulus onset with 20 ms bins, so the first input is just the shared time axis of the neural representation.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`. A new block starts whenever `probabilityLeft` changes value.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()
new_block[1:] = pleft[1:] != pleft[:-1]
```

iii. The trajectory summary says the block index is computed within the current `probabilityLeft` block, so the justification is that block identity is recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent computes a 0-based counter within each constant-`probabilityLeft` block, using the full unfiltered trial table so later trial exclusions do not shift the count.

ii.
```python
new_block = np.ones(len(pleft), dtype=bool)
new_block[1:] = pleft[1:] != pleft[:-1]
block_id = np.cumsum(new_block) - 1
block_starts = np.flatnonzero(new_block)
idx_in_block = np.arange(len(pleft)) - block_starts[block_id]
```

```python
in_block = trial_number_in_block(trials)[keep]
inp[1] = in_block[i]
```

iii. The trajectory summary explicitly says the agent computed the 0-based trial index on the full trials table "so exclusions don't shift the count."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials['choice']`.

ii.
```python
choice = np.where(trials['choice'].to_numpy()[keep] > 0, 0, 1).astype(np.int64)
```

iii. The trajectory summary says the agent used the IBL convention that `+1` means left choice and `-1` means right choice.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After excluding no-choice trials (`choice == 0`), the agent recodes positive choices to `0` for left and the remaining negative choices to `1` for right, then broadcasts that per-trial label across all 100 time bins.

ii.
```python
query += ' | (choice == 0)'
```

```python
choice = np.where(trials['choice'].to_numpy()[keep] > 0, 0, 1).astype(np.int64)
out[0] = choice[i]
```

iii. The trajectory summary justifies this with the IBL sign convention for `choice`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from `trials['probabilityLeft']`.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[keep]
```

iii. The trajectory summary says this output is the current block's `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, validates that no other values appear, and broadcasts the resulting per-trial category across time bins.

ii.
```python
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int64)
if np.any(prior < 0):
    raise RuntimeError(f'unexpected probabilityLeft values: {np.unique(pleft)}')
```

```python
out[1] = prior[i]
```

iii. The trajectory summary says this mapping was chosen to match the decoder task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel trace loaded by `SessionLoader`, specifically `sess_loader.wheel['times']` and `sess_loader.wheel['velocity']`. That velocity itself comes from the wheel position/timestamp dataset handled inside the loader.

ii.
```python
if sess_loader.wheel.empty:
    sess_loader.load_wheel()
return (sess_loader.wheel['times'].to_numpy(),
        np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The trajectory summary says the agent followed the reference-code source for wheel speed, namely absolute wheel velocity returned by `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent takes the absolute value of the loader-provided wheel velocity, interpolates it linearly onto the trial bins at the right edge of each 20 ms bin, rejects trials whose wheel trace does not cover the full window or contains NaNs, then discretizes the retained session-wide values into within-session tertiles.

ii.
```python
return (sess_loader.wheel['times'].to_numpy(),
        np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

```python
wheel_class, wheel_edges = discretize(wheel[keep])
```

iii. The trajectory summary says the agent believed this exactly reproduced `get_behavior_per_interval`, and it also justified session-wise tertiles as a way to keep classes comparable and balanced across sessions.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is split into 3 classes using session-specific tertiles computed over all retained trials and time bins in that session.

ii.
```python
quantiles = np.quantile(values, np.arange(1, n_classes) / n_classes)
edges = np.maximum.accumulate(quantiles)
return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. The trajectory summary says within-session tertiles were chosen because absolute wheel-speed scales differ across animals and sessions, and because balanced classes help joint decoder training.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-locked trial window as the neural data, but the agent samples it at the right edge of each 20 ms bin rather than storing bin centers.

ii.
```python
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
```

```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. The trajectory summary says the agent did this because it believed it was matching `get_behavior_per_interval` exactly.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from the camera motion-energy stream loaded by `SessionLoader`. The agent prefers the left camera and falls back to the right camera if the left is unavailable, using each stream's `times` and `whiskerMotionEnergy`.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[key]
        return (me['times'].to_numpy(),
                me['whiskerMotionEnergy'].to_numpy())
```

iii. The trajectory summary says the left camera was preferred with right-camera fallback, following the methods-paper code path.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released whisker motion-energy trace is used as-is, linearly interpolated onto the trial bins at the right edge of each 20 ms bin, trials with missing/short/NaN traces are dropped, and the retained values are discretized into within-session tertiles.

ii.
```python
whisker, whisker_ok = bin_behavior(whisker_times, whisker_vals,
                                   interval_begs, interval_ends)
```

```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

```python
whisker_class, whisker_edges = discretize(whisker[keep])
```

iii. The trajectory summary says the agent intentionally used the motion-energy trace as released, and justified the within-session tertiles on comparability and class-balance grounds.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded into 3 categories using session-specific tertiles over all retained whisker-motion-energy samples in the session.

ii.
```python
quantiles = np.quantile(values, np.arange(1, n_classes) / n_classes)
edges = np.maximum.accumulate(quantiles)
return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. The trajectory summary says session-wise thresholds were used because motion-energy units depend on camera and illumination and are not directly comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-locked trial window as the neural data, using the same 100 bins, but the signal is evaluated at the right edge of each bin.

ii.
```python
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
```

```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. The trajectory summary says the agent believed this matched the reference behavior-binning function.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mainly handles missing data by exclusion. NaN alignment times are temporarily filled with `0.0` only so indexing code remains defined, but those trials are already excluded by the trial mask. Trials are dropped if wheel or whisker coverage is incomplete or if interpolated values are non-finite. Sessions with fewer than two good trials are dropped. Any session that raises an exception is recorded in `failed_sessions` and omitted from the final dataset.

ii.
```python
align_times_filled = np.where(np.isnan(align_times), 0.0, align_times)
```

```python
if len(v) == 0:
    continue
if np.abs(interval_begs[trial] - t[0]) > BINSIZE:
    continue
if np.abs(interval_ends[trial] - t[-1]) > BINSIZE:
    continue
if np.any(~np.isfinite(y)):
    continue
```

```python
def _safe_process_session(args):
    try:
        return process_session(args)
    except Exception as exc:
        return {'eid': args[0], 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}
```

iii. The trajectory summary says the extra trial dropping was deliberate because the target format cannot contain NaNs. It also notes that 15 sessions were excluded this way because of missing or insufficient camera data.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts are loading and merging per-probe spike sorting, then binning spikes and behavior across all trials of a session. The code structure makes spike loading the clear dominant I/O cost.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

```python
binned, unit_ids = bin_spikes(spikes['times'], spikes['clusters'],
                              interval_begs[keep], interval_ends[keep])
wheel, wheel_ok = bin_behavior(wheel_times, wheel_vals, interval_begs, interval_ends)
```

iii. The trajectory does not contain a separate explicit performance analysis, but the agent's final summary and runtime logging focus on session conversion and large spike-count matrices, indicating that spike loading and binning were the intended heavy steps.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over trials in three places: spike binning, behavior interpolation, and final packing of per-trial arrays into Python lists. The agent already vectorized some setup with `searchsorted`, but the remaining trial loops could still be reduced further.

ii.
```python
for trial in range(n_trials):
    sl = slice(i_beg[trial], i_end[trial])
    ...
```

```python
for trial in range(n_trials):
    t = target_times[idxs_beg[trial]:idxs_end[trial]]
    ...
```

```python
for i in range(n_trials):
    neural.append(binned[i].astype(np.float32))
    ...
    outputs.append(out)
```

iii. The trajectory summary says the spike and behavior binning were deliberately implemented as a vectorized reimplementation of the reference path, but it does not claim the code is fully vectorized.

## 10-c. What processing does the code repeat multiple times?

i. There is no major redundant scientific processing, but some work is repeated per session or per output stream: a fresh ONE client is created for every session worker call, the same interpolation routine is run once for wheel and once for whisker traces, and the code separately loops over trials again when building the final lists.

ii.
```python
one = get_one()
```

```python
wheel, wheel_ok = bin_behavior(wheel_times, wheel_vals, interval_begs, interval_ends)
whisker, whisker_ok = bin_behavior(whisker_times, whisker_vals,
                                   interval_begs, interval_ends)
```

iii. The trajectory gives no explicit justification beyond robustness and fidelity to the reference code path.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent does some extra bookkeeping beyond the decoder-ready arrays: it computes and stores class edges, lab names, failure records, and detailed session metadata. It also computes continuous wheel and whisker traces only to discretize them immediately, so the continuous versions are discarded once the categorical outputs are built.

ii.
```python
wheel_class, wheel_edges = discretize(wheel[keep])
whisker_class, whisker_edges = discretize(whisker[keep])
```

```python
'session_info': {
    'eid': eid,
    'subject': subject,
    'lab': lab,
    ...
    'wheel_speed_class_edges': [float(e) for e in wheel_edges],
    'whisker_motion_energy_class_edges': [float(e) for e in whisker_edges],
},
```

```python
'failed_sessions': failures,
```

iii. The trajectory summary does not defend these as necessary for downstream decoding; they appear to be convenience metadata and diagnostics added by the agent.
