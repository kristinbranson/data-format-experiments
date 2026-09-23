# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not enumerate sessions through `one.search` the way the human reference does. Instead, it reads `/app/code/code_zhang2025/data/bwm_release.csv`, uses the unique `eid` values from that freeze file to define the session list, and uses the probe rows in that CSV to know which `pid`/`probe_name` pairs to load per session. Once an `eid` is selected, it loads trials, wheel, motion energy, and spike sorting through `ONE`/`brainbox` loaders.

ii. 
```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))  # preserve freeze order
```

```python
_ONE = ONE(base_url=ONE_BASE_URL, silent=True, tables_dir=ONE_TABLES_DIR)
```

```python
ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
sess_loader = SessionLoader(one=one, eid=eid)
```

iii. In `CONVERSION_NOTES.md` Step 10, the agent says this is the same mapping as `one.eid2pid` but avoids an Alyx/REST round trip. Its overall justification is that the freeze CSV defines the public BWM release and the actual data loading still goes through `ONE` and `brainbox`.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the freeze CSV. Each processed session result carries its subject string, and the final dataset builds `subjects` as sorted unique subject names with `subject_idx` mapping each session to that list.

ii. 
```python
tasks.append((eid, rows[['pid', 'probe_name']].copy(),
              rows.subject.iloc[0], rows.lab.iloc[0], i < n_show))
```

```python
subjects = sorted({r['subject'] for r in ok})
subject_index = {s: i for i, s in enumerate(subjects)}
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent explicitly maps `bwm_release.csv` `subject` to `subjects` / `subject_idx`, justifying it as the release metadata for the session list it chose to process.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `eid`. The agent preserves the freeze-file order, processes each `eid` independently, and stores one session entry per successful `eid` in `neural`, `input`, and `output`.

ii. 
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))  # preserve freeze order
```

```python
ok.sort(key=lambda r: order[r['eid']])
data = {
    'neural': [r['neural'] for r in ok],
    'input': [r['input'] for r in ok],
    'output': [r['output'] for r in ok],
```

iii. The notes describe the session set as “sessions from the 459-eid BWM freeze,” so the justification is that the public release itself is session-indexed by `eid`.

## 1-d. How are the data split into trials?

i. Within each session, the agent loads the trials table through `load_trials_and_mask`. Candidate trials are the rows where the reference mask is true, and retained trials are the surviving indices after additional behavior-validity and zero-spike filtering.

ii. 
```python
sess_loader, trials, trials_mask = load_trials(one, eid)
```

```python
cand = np.nonzero(trials_mask)[0]
...
trial_idx = cand[keep_trial]
```

iii. The agent follows the reference code’s session-level trial table structure, then documents in `CONVERSION_NOTES.md` that later filtering stages keep neural, input, and output trial-aligned.

## 1-e. How are trials filtered based on quality controls?

i. The agent starts from the reference `load_trials_and_mask(..., max_trial_len=10.0)` output, then further requires both time-varying behavior streams to fully cover the 2 s window with no NaNs. After spike binning, it also drops trials whose entire neural population has zero spikes in the whole window.

ii. 
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
```

```python
for name, d in beh.items():
    v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs,
                                   PARAMS['binsize'], N_BINS)
...
keep_trial = np.ones(len(cand), dtype=bool)
for name in beh_vals:
    keep_trial &= beh_good[name]
```

```python
has_spikes = binned.any(axis=(1, 2))
...
binned = binned[has_spikes]
trial_idx = trial_idx[has_spikes]
```

iii. In Steps 4, 5, and 10 of `CONVERSION_NOTES.md`, the agent justifies this as: keep the reference trial mask, drop behavior windows with missing data rather than impute categorical targets, and drop all-zero-spike trials as recording dropouts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is derived from spike times and spike cluster assignments from each probe, then filtered using cluster metadata such as `label` and `acronym`. The binned neural values themselves come from `spikes['times']` and remapped `spikes['clusters']`.

ii. 
```python
sp, cl, ch = ssl.load_spike_sorting()
cl_df = SpikeSortingLoader.merge_clusters(sp, cl, ch, compute_metrics=False).to_df()
```

```python
spike_cl = remap[spikes['clusters']]
sel = spike_cl >= 0
spike_times = np.ascontiguousarray(spikes['times'][sel])
spike_cl = np.ascontiguousarray(spike_cl[sel])
```

iii. The notes say `load_session_spikes` mirrors the reference `prepare_data` / `load_spiking_data` / `merge_probes` path, with cluster metadata used for neuron curation and region labels.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, remaps kept clusters to a contiguous neuron index, and bins spikes into 20 ms non-overlapping bins over a 2 s stimulus-aligned window. Unlike the human reference solution, it stores raw spike counts per bin rather than dividing by bin width to convert to Hz.

ii. 
```python
return merge_probes(spikes_list, clusters_list)
```

```python
binned = bin_spikes(spike_times, spike_cl, n_neurons, interval_begs,
                    PARAMS['binsize'], N_BINS)
```

```python
out = np.zeros((n_intervals, n_neurons, n_bins), dtype=np.float32)
...
out[k] = counts.reshape(n_neurons, n_bins)
```

iii. In Step 5 and Step 10 of `CONVERSION_NOTES.md`, the agent explicitly says it chose raw counts because that matches the cached representation in the Zhang reference code, and leaves normalization to the decoder rather than the conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only clusters with `label >= 1`, maps Allen acronyms to Beryl, and drops units whose Beryl acronym is `root` or `void`. It also rejects sessions with fewer than 5 surviving neurons.

ii. 
```python
beryl_all = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
good = clusters['label'].to_numpy() >= 1
in_brain = ~np.isin(beryl_all, NON_REGION_ACRONYMS)
keep = good & in_brain
```

```python
if n_neurons < MIN_NEURONS_PER_SESSION:
    return {'eid': eid,
            'error': f'only {n_neurons} well-isolated grey-matter neurons '
                     f'(< {MIN_NEURONS_PER_SESSION})'}
```

iii. The notes justify this as following the data paper’s “well-isolated” neurons, restricting to grey matter, and applying a five-neuron floor to avoid degenerate decoder sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each kept trial, the binning interval begins at `stimOn_times - 0.5 s` and ends at `stimOn_times + 1.5 s`.

ii. 
```python
PARAMS = {
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
```

```python
stim_on_all = trials[PARAMS['align_time']].to_numpy()
interval_begs = stim_on_all[cand] + PARAMS['time_window'][0]
```

iii. The agent’s notes repeatedly state that stimulus-onset alignment is required both by the decoder task and by the released Zhang caching code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins over a 2 s window, giving 100 bins per trial. No later temporal rebinning is applied in the conversion itself.

ii. 
```python
PARAMS = {
    'interval_len': 2.0,
    'binsize': 0.02,
}
N_BINS = int(np.ceil(PARAMS['interval_len'] / PARAMS['binsize']))  # 100
```

iii. The notes justify this as matching the released reference code and the method paper’s 20 ms / T=100 configuration.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the alignment event `trials.stimOn_times`, plus the fixed 20 ms grid used for the session. It is not read from a dedicated raw column beyond the stimulus-onset timestamps that define zero.

ii. 
```python
stim_on_all = trials[PARAMS['align_time']].to_numpy()
```

```python
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
inputs[:, 0, :] = bin_times[None, :]
```

iii. In Step 5, the agent describes this input as “time (s) of the right edge of each 20 ms bin relative to stimulus onset,” so its justification is that the time input should use the same trial-aligned temporal grid as the rest of the converted data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent constructs a fixed vector of relative bin times and broadcasts it to every retained trial. It uses the right edge of each 20 ms bin, producing values from `-0.48` to `1.50`.

ii. 
```python
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
inputs[:, 0, :] = bin_times[None, :]
```

iii. In `CONVERSION_NOTES.md`, the agent justifies right-edge timing because the continuous behaviors are also interpolated onto the right edge of each spike bin in the Zhang pipeline.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by using one timestamp per neural bin across the same stimulus-aligned 2 s window. The agent represents the time of the right edge of each spike-count bin, not the bin center.

ii. 
```python
binned = bin_spikes(spike_times, spike_cl, n_neurons, interval_begs,
                    PARAMS['binsize'], N_BINS)
...
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
```

iii. The notes explicitly say the time input is the same grid as the behavior interpolation grid; that is the stated reason it uses the right edge instead of a center timestamp.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials.probabilityLeft`. The agent infers blocks as contiguous runs of equal `probabilityLeft`.

ii. 
```python
def trial_number_in_block(probability_left):
    """0-based index of each trial within its constant-``probabilityLeft`` block."""
```

```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. In the notes, the agent justifies this by saying there is no explicit block ID in the trials table, so block identity must be recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent computes a 0-based within-block counter on the unfiltered trials table, treating any change in `probabilityLeft` as a new block, then broadcasts each retained trial’s scalar value across all 100 time bins.

ii. 
```python
changed = np.ones(len(p), dtype=bool)
if len(p) > 1:
    same = (p[1:] == p[:-1]) | (np.isnan(p[1:]) & np.isnan(p[:-1]))
    changed[1:] = ~same
block_id = np.cumsum(changed) - 1
...
return np.arange(len(p)) - first_of_block[block_id]
```

```python
tib = tib_all[trial_idx].astype(np.float32)
inputs[:, 1, :] = tib[:, None]
```

iii. The notes justify computing it before filtering because block position is treated as a property of the experiment, not of the retained-trial subset.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `trials.choice` column.

ii. 
```python
choice = trials['choice'].to_numpy()[trial_idx]
```

iii. The notes identify `trials.choice` as the source variable and note the IBL sign convention `+1 = left`, `-1 = right`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent recodes `choice == +1` to class `0` (left) and `choice == -1` to class `1` (right), then broadcasts the class label across the 100 time bins of each retained trial.

ii. 
```python
choice_lbl = ((1 - choice) / 2).astype(np.int64)
...
outputs[:, 0, :] = choice_lbl[:, None]
```

iii. In the notes, the agent justifies the mapping from the task instructions and from spot checks of high-contrast trials confirming the sign convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from `trials.probabilityLeft`.

ii. 
```python
pleft = trials['probabilityLeft'].to_numpy()[trial_idx]
```

iii. The notes identify `probabilityLeft` as the block prior carried in the trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts that class over time for each trial. It raises an error if any other `probabilityLeft` value appears.

ii. 
```python
prior_lbl = np.full(n_trials, -1, dtype=np.int64)
for val, lbl in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_lbl[np.isclose(pleft, val)] = lbl
if np.any(prior_lbl < 0):
    bad = np.unique(pleft[prior_lbl < 0])
    return {'eid': eid, 'error': f'unexpected probabilityLeft values {bad}'}
```

```python
outputs[:, 1, :] = prior_lbl[:, None]
```

iii. The notes justify the mapping as the required decoder-output coding from the task description.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed comes from the wheel timestamps/position stream as processed by `SessionLoader.load_wheel()`, specifically the absolute value of the returned wheel velocity trace.

ii. 
```python
sess_loader.load_wheel()
beh['wheel-speed'] = {
    'times': sess_loader.wheel['times'].to_numpy(),
    'values': np.abs(sess_loader.wheel['velocity'].to_numpy()),
    'source': 'wheel.velocity (abs)',
}
```

iii. In Step 6, the agent explicitly says this matches `load_target_behavior('wheel-speed')` in the Zhang utilities.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent uses the wheel velocity that `SessionLoader` computes, takes its absolute value, interpolates it onto each trial’s 100-bin grid, then discretizes the retained-trial values into three per-session tertile classes.

ii. 
```python
v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs,
                               PARAMS['binsize'], N_BINS)
```

```python
ws = beh_vals['wheel-speed'][keep_trial]
ws_lbl, ws_thr = discretize_tertiles(ws)
```

iii. The notes justify this as following the reference wheel-loading path and using tertiles because the task requires three categorical bins and session-specific balancing avoids session-scale differences.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded with session-specific tertiles, using the 33.3% and 66.7% quantiles of all retained wheel-speed values pooled across the session.

ii. 
```python
TERTILES = (1.0 / 3.0, 2.0 / 3.0)
...
flat = values.reshape(-1)
q1, q2 = np.quantile(flat, TERTILES)
labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
```

iii. In Step 5, the agent says tertiles were chosen to satisfy the three-bin categorical requirement while keeping class frequencies balanced within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The agent aligns wheel speed to the same stimulus-centered 2 s trial window as the neural data, but samples it at the right edge of each 20 ms neural bin rather than at the bin center.

ii. 
```python
x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. The notes say this follows `get_behavior_per_interval` from the Zhang code, whose interpolation target grid is the bin right edge.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from camera whisker-pad ROI motion energy and camera frame times. The agent prefers the left camera and falls back to the right camera if left-camera motion energy is unavailable.

ii. 
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        me = {
            'times': df['times'].to_numpy(),
            'values': df['whiskerMotionEnergy'].to_numpy(),
            'source': f'{cam}.ROIMotionEnergy',
        }
        break
```

iii. The notes explicitly document “left (else right) camera whisker-pad ROI motion energy” and justify the fallback as handling sessions without left-camera data.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used directly, interpolated onto the trial grid, and then discretized into three per-session tertile classes. The agent drops trials if the behavior window has missing coverage or NaNs.

ii. 
```python
me = beh_vals['whisker-motion-energy'][keep_trial]
me_lbl, me_thr = discretize_tertiles(me)
```

```python
if np.any(np.isnan(v)):
    reasons[k] = 'nans in target data'
    continue
```

iii. The notes justify leaving the trace otherwise unprocessed because the reference behavior-loading code uses the released trace, while the categorical-output requirement forces discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same per-session tertile discretization as wheel speed.

ii. 
```python
me_lbl, me_thr = discretize_tertiles(me)
```

```python
labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
```

iii. The notes justify per-session thresholds because motion-energy units depend on camera/ROI/session scale and are not directly comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-centered 2 s windows as the neural data and sampled at the right edge of each 20 ms neural bin.

ii. 
```python
x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. The notes say this matches the behavior interpolation rule in the Zhang reference utilities.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mostly handles missing or problematic data by exclusion. It skips sessions with missing whisker motion energy, too few surviving neurons, or too few valid trials; drops trials with invalid behavior coverage, NaNs in time-varying behavior, or zero spikes across the whole population; skips probe insertions that fail during loading by returning a session-level error; and guards against degenerate tertiles and unexpected `probabilityLeft` values.

ii. 
```python
if me is None:
    raise RuntimeError('no whisker motion energy available (left or right camera)')
```

```python
if np.any(np.isnan(v)):
    reasons[k] = 'nans in target data'
    continue
```

```python
if n_neurons < MIN_NEURONS_PER_SESSION:
    return {'eid': eid, 'error': ...}
...
if keep_trial.sum() < 2:
    return {'eid': eid, 'error': ...}
```

iii. In Steps 4, 5, 9, and 10 of `CONVERSION_NOTES.md`, the agent consistently justifies exclusion over imputation because the target outputs are categorical and because degenerate sessions/trials were causing validation warnings.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading spike sorting for each session/probe. The agent’s timing summaries in the notes show spike loading dominating all other stages, with trial/behavior loading much smaller and spike binning very cheap after vectorization.

ii. 
```python
sp, cl, ch = ssl.load_spike_sorting()
```

```python
timings['load_spikes'] = time.time() - t0
```

iii. In Step 7 and the final timing report, the agent explicitly says spike loading is the bottleneck and that its main optimization work focused on binning and multiprocessing rather than I/O.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized much of the spike binning compared with the reference, but its own code still contains per-interval loops in `bin_spikes` and `interpolate_behavior` that could be pushed further toward full vectorization. It also loops over sessions and probes in Python.

ii. 
```python
for k in range(n_intervals):
    i0, i1 = i0s[k], i1s[k]
    ...
    counts = np.bincount(flat, minlength=n_neurons * n_bins)
    out[k] = counts.reshape(n_neurons, n_bins)
```

```python
for k in range(n):
    t = times[idxs_beg[k]:idxs_end[k]]
    v = values[idxs_beg[k]:idxs_end[k]]
    ...
    vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. The notes justify leaving these loops in place because the major speedups had already been achieved by replacing the reference’s per-trial multiprocessing and `bincount2D` calls with a cheaper vectorized/session-parallel structure.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly performs similar interpolation-and-validity checks for each continuous behavior stream and repeats analogous per-session loading logic for every session. It also repeats per-session quantile-based discretization separately for wheel speed and whisker motion energy.

ii. 
```python
for name, d in beh.items():
    v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs,
                                   PARAMS['binsize'], N_BINS)
    beh_vals[name], beh_good[name], beh_reasons[name] = v, g, r
```

```python
ws_lbl, ws_thr = discretize_tertiles(ws)
me_lbl, me_thr = discretize_tertiles(me)
```

iii. The agent’s notes frame this as acceptable repeated per-session/per-behavior work, with the real optimization target being the heavier spike-binning path.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion does extra bookkeeping and validation work that the downstream decoder does not use: timing collection, detailed `session_info` metadata, lab names, tertile thresholds, behavior-drop diagnostics, optional processing plots, and `n_good_units` / `n_clusters_total` summaries. It also carries diagnostic branches to handle plot generation and reporting.

ii. 
```python
timings = {}
...
'info': {
    'eid': eid,
    'subject': subject,
    'lab': lab,
    'n_probes': int(len(probe_rows)),
    'n_clusters_total': int(len(clusters)),
    'n_good_units': n_good,
    ...
    'wheel_speed_tertiles': ws_thr,
    'whisker_me_tertiles': me_thr,
    'behavior_drop_counts': {
        name: int((~beh_good[name]).sum()) for name in beh_good},
},
'timings': timings,
```

```python
if show_processing:
    try:
        make_processing_figure(...)
```

iii. The notes justify these additions as auditability and sanity checking rather than core decoder input preparation.
