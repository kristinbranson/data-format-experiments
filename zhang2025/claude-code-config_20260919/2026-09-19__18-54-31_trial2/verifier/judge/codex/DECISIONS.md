# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI builds the session list from `/app/code/code_zhang2025/data/bwm_release.csv`, optionally restricts it with `/app/data/DATALIMIT_SUBSET.csv`, and then loads each session through ONE-backed loaders. Session-level behavioral data are loaded with `SessionLoader`; probe-level spike sorting is loaded with `SpikeSortingLoader`.

ii. 
```python
BWM_FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT_FILE = '/app/data/DATALIMIT_SUBSET.csv'
```

```python
def build_session_list():
    bwm = pd.read_csv(BWM_FREEZE_FILE, index_col=0)
    if os.path.exists(DATALIMIT_FILE):
        sub = pd.read_csv(DATALIMIT_FILE)
        col = 'eid' if 'eid' in sub.columns else sub.columns[0]
        bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
```

```python
sess_loader = SessionLoader(one=one, eid=eid)
times, clu, clusters_df, collection = load_spiking_data(one, pid, eid, pname)
```

iii. In `CONVERSION_NOTES.md`, the AI says it used the same 459-session freeze as the reference code and kept ONE for actual file resolution/loading, while avoiding direct path assumptions because the staged cache uses dataset revisions.

## 1-b. How are the data split into subjects?

i. Subjects come from the `subject` column of `bwm_release.csv`. After conversion, unique subject names are sorted into `subjects`, and each session gets a `subject_idx`.

ii.
```python
sessions.append({
    'eid': str(eid),
    'pids': [str(p) for p in g.pid],
    'probe_names': list(g.probe_name),
    'subject': str(g.subject.iloc[0]),
```

```python
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
```

iii. The notes justify this as reusing the session/probe freeze used by the reference repository, where subject metadata are already attached to each session.

## 1-c. How are the data split into sessions?

i. Sessions are split by `eid`. The AI groups `bwm_release.csv` by `eid`, treating each grouped set of probe rows as one session.

ii.
```python
for eid, g in bwm.groupby('eid', sort=False):
    g = g.sort_values('probe_name')
    sessions.append({
        'eid': str(eid),
        'pids': [str(p) for p in g.pid],
        'probe_names': list(g.probe_name),
```

iii. The justification in the notes is that `bwm_release.csv` is the release freeze used by the Zhang et al. code, so `eid` is the authoritative session identifier.

## 1-d. How are the data split into trials?

i. The AI uses the session trials table as the trial split. Each row of `sess_loader.trials` is treated as one trial.

ii.
```python
if sess_loader.trials.empty:
    sess_loader.load_trials()
trials = sess_loader.trials
```

```python
trials_df, trials_mask = load_trials_and_mask(one, eid, sess_loader=sess_loader)
n_trials_raw = len(trials_df)
```

iii. The justification is implicit and matches the notes: the trials table is already one row per trial, so no further segmentation is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a boolean mask that removes trials with too-short or too-long reaction times, excessive trial length, NaNs in a fixed set of key columns, no-response choices, non-finite alignment times, and any trial whose wheel or whisker trace does not fully support the analysis window after interpolation.

ii.
```python
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
```

```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in NAN_EXCLUDE:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
mask = ~trials.eval(query)
```

```python
keep = np.asarray(trials_mask, dtype=bool).copy()
keep &= np.isfinite(interval_begs)
for name in beh_good:
    keep &= beh_good[name]
```

iii. The notes say this was intended to match `load_trials_and_mask` plus the behavior-coverage logic of `align_spike_behavior`, and to drop any remaining NaN behavior trials because the downstream decoder requires finite values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is derived from spike times and spike cluster identities, with cluster metadata used for filtering and region labels.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

```python
sel_times = spikes['times'][spike_idx]
sel_cl = sel_clusters.index.to_numpy()[ib].astype(np.int64)
```

iii. The notes explicitly map `spikes.times` and `spikes.clusters` to the final neural matrix, with `clusters.acronym` and `clusters.label` used only for curation and region bookkeeping.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 20 ms bins per trial and keeps the result as spike counts. It merges probes by concatenating their neuron axes after per-probe binning. Unlike the human reference, it does not divide by bin width to convert counts to firing rates in Hz.

ii.
```python
bin_of_spike = np.floor(t_rel / binsize).astype(np.int64)
lin = ((trial_of_spike * n_clusters) + clusters[flat_idx]) * n_bins + bin_of_spike
out += np.bincount(lin, minlength=n_trials * n_clusters * n_bins).reshape(
    n_trials, n_clusters, n_bins).astype(np.float32)
```

```python
neural = np.concatenate(neural_trials, axis=1)          # (n_trials, n_neurons, T)
```

```python
'neural_units': 'spike counts per 20 ms bin',
```

iii. The AI’s notes justify the binning arithmetic as matching the reference spike binning, but it deliberately describes the stored quantity as spike counts rather than Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1`, maps regions to Beryl, removes `root` and `void`, drops clusters that emit no spikes in the session, and drops sessions with fewer than 5 surviving neurons.

ii.
```python
QC_LABEL = 1.0
NON_GREY_MATTER = ('root', 'void')
MIN_NEURONS_PER_SESSION = 5
```

```python
iok = clusters_labeled['label'] >= qc
sel_clusters = clusters_labeled[iok]
```

```python
beryl = np.asarray(br.acronym2acronym(clusters_df['acronym'].to_numpy(),
                                      mapping='Beryl'), dtype=object)
has_spikes = np.zeros(len(clusters_df), dtype=bool)
if clu.size:
    has_spikes[np.unique(clu)] = True
sel = (~np.isin(beryl, NON_GREY_MATTER)) & has_spikes
```

```python
if neural.shape[1] < MIN_NEURONS_PER_SESSION:
    raise RuntimeError(f'only {neural.shape[1]} well-isolated grey-matter neurons '
                       f'(< {MIN_NEURONS_PER_SESSION})')
```

iii. The notes justify this by appealing to the data paper’s “well-isolated grey-matter neurons” framing and a later decision to enforce at least 5 neurons/session after seeing all-zero neural trials in very small sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times` with a fixed window from -0.5 s to +1.5 s around stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
align_times = trials_df[ALIGN_TIME].to_numpy(dtype=float)
interval_begs = align_times + TIME_WINDOW[0]
```

```python
binned = bin_spikes(times[spike_sel], row_of[clu[spike_sel]],
                    int(sel.sum()), interval_begs[keep_idx])
```

iii. The notes explicitly say this mirrors the reference `params` and the task instruction to align to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms per bin, giving 100 bins over the 2 s window. There is no additional temporal rebinning beyond this binning.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. The notes justify 20 ms as matching the reference code and the paper’s “2-s trials, 20-ms bins, T = 100” description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the stimulus-onset alignment choice (`stimOn_times`) plus the fixed decoding window and bin size; the stored values are bin centers relative to that event.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
```

```python
bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)).astype(np.float32)
inputs[:, 0, :] = bin_centres[None, :]
```

iii. The notes say this input is new for the decoder task but is tied to the same stimulus-aligned bin grid as the neural data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes a deterministic time ramp consisting of the 100 bin centers from -0.49 s to +1.49 s and copies it into every retained trial.

ii.
```python
bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)).astype(np.float32)
inputs = np.empty((n_keep, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = bin_centres[None, :]
```

iii. The notes justify this as the natural representation of “time from stimulus onset” on the chosen neural binning grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same trial window and the same 100 neural bins, but records the center of each bin rather than the bin edges.

ii.
```python
interval_begs = align_times + TIME_WINDOW[0]
```

```python
bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)).astype(np.float32)
inputs[:, 0, :] = bin_centres[None, :]
```

iii. The notes state that this input is the same stimulus-locked grid used for neural binning, expressed at bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` sequence in the trials table.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
```

```python
tnb_all = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. The notes justify this by saying block identity is not provided directly, so block structure is recovered from runs of constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a 0-based run-length count within consecutive trials having the same `probabilityLeft`, does this on the full uncurated trial table, and then broadcasts the retained-trial value across time bins.

ii.
```python
out = np.zeros(len(p), dtype=np.float64)
counter = 0
for i in range(1, len(p)):
    same = (p[i] == p[i - 1]) or (np.isnan(p[i]) and np.isnan(p[i - 1]))
    counter = counter + 1 if same else 0
    out[i] = counter
```

```python
inputs[:, 1, :] = tnb_all[keep_idx].astype(np.float32)[:, None]
```

iii. The notes explicitly justify computing this before curation so the value reflects the animal’s real position in the block even if neighboring trials are later excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials_df['choice']`.

ii.
```python
choice = trials_df['choice'].to_numpy()[keep_idx]        # +1 = left, -1 = right
```

iii. The notes say the AI verified empirically that `choice == +1` means left and `choice == -1` means right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI recodes left choices to 0 and right choices to 1, then broadcasts the trial label across all time bins of that trial.

ii.
```python
choice_cls = (choice < 0).astype(np.int64)               # left -> 0, right -> 1
```

```python
outputs[:, 0, :] = choice_cls[:, None]
```

iii. The notes justify this as the decoder-task-required coding `left = 0`, `right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials_df['probabilityLeft']`.

ii.
```python
pleft = trials_df['probabilityLeft'].to_numpy()[keep_idx]
```

iii. The notes identify `probabilityLeft` as the task’s block prior and the natural source for this output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, checks that no unexpected values remain, and broadcasts the class label across time bins.

ii.
```python
pleft_cls = np.full(len(pleft), -1, dtype=np.int64)
for val, cls in ((0.2, 0), (0.5, 1), (0.8, 2)):
    pleft_cls[np.isclose(pleft, val)] = cls
if np.any(pleft_cls < 0):
    bad = np.unique(pleft[pleft_cls < 0])
    raise RuntimeError(f'unexpected probabilityLeft values {bad}')
```

```python
outputs[:, 1, :] = pleft_cls[:, None]
```

iii. The notes justify this as the decoder-task-specified 3-class encoding of the trial prior.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel time series returned by `SessionLoader.load_wheel()`, specifically from the absolute value of `wheel['velocity']`.

ii.
```python
sess_loader.load_wheel()
traces['wheel_speed'] = (sess_loader.wheel['times'].to_numpy(),
                         np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The notes justify this as matching the reference behavior loader, which defines wheel speed as `abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI takes `abs(wheel.velocity)`, interpolates it onto each trial’s 100-bin stimulus-aligned grid using linear interpolation at the right edge of each bin, and then discretizes the session’s values into three classes.

ii.
```python
vals, good = bin_behaviour(tt, tv, interval_begs)
```

```python
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
```

```python
ws = beh_vals['wheel_speed'][keep_idx]
ws_edges = discretize_terciles(ws)
ws_cls = apply_terciles(ws, *ws_edges)
```

iii. The notes justify the right-edge interpolation as an exact port of the reference `get_behavior_per_interval`, and the sessionwise discretization as appropriate because scale differs across sessions.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes per-session tercile thresholds from all retained wheel-speed bins and assigns class 0 to `v <= e1`, class 1 to `e1 < v <= e2`, and class 2 to `v > e2`, with extra handling when the quantile edges collapse.

ii.
```python
def discretize_terciles(values):
    v = np.asarray(values, dtype=np.float64).ravel()
    e1, e2 = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    if e1 == e2:
        above = v[v > e1]
        e2 = np.quantile(above, 0.5) if above.size else e1
        if e2 == e1:
            above = v[v > e1]
            e2 = above.min() if above.size else e1
    return float(e1), float(e2)
```

```python
def apply_terciles(values, e1, e2):
    return ((values > e1).astype(np.int64) + (values > e2).astype(np.int64))
```

iii. The notes justify this as a per-session equal-frequency discretization that avoids global thresholds on session-dependent scales and repairs degenerate quantiles in near-constant traces.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is aligned trial-by-trial to the same stimulus-onset windows as the neural data, but sampled at the right edge of each 20 ms neural bin.

ii.
```python
interval_begs = align_times + TIME_WINDOW[0]
```

```python
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. The notes explicitly state that each wheel sample corresponds to the right edge of the matching spike bin, which the AI considered the reference behavior-alignment rule.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the camera motion-energy time series, preferring the left camera and falling back to the right camera if needed.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[key]
        traces['whisker_motion_energy'] = (me['times'].to_numpy(),
                                           me['whiskerMotionEnergy'].to_numpy())
        cam_used = view
        break
```

iii. The notes justify this as matching the reference behavior loader’s left-camera preference with right-camera fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released `whiskerMotionEnergy` values directly, linearly interpolates them onto each trial’s 100-bin grid at bin right edges, and then discretizes the session’s values into three classes.

ii.
```python
vals, good = bin_behaviour(tt, tv, interval_begs)
```

```python
wme = beh_vals['whisker_motion_energy'][keep_idx]
wme_edges = discretize_terciles(wme)
wme_cls = apply_terciles(wme, *wme_edges)
```

iii. The notes say no extra filtering or normalization was added; the AI treated the released motion-energy trace as the source signal and only resampled/discretized it for the decoder.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same per-session tercile scheme as wheel speed, including the degenerate-quantile repair path.

ii.
```python
wme_edges = discretize_terciles(wme)
wme_cls = apply_terciles(wme, *wme_edges)
```

iii. The notes justify sessionwise thresholds because whisker-motion-energy scale depends on the camera stream and session, so global thresholds would not be comparable.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-onset trial windows as neural data and sampled at the right edge of each 20 ms neural bin.

ii.
```python
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The notes justify this as matching the reference behavior interpolation procedure exactly.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally drops unusable data rather than imputing it. It drops trials with missing key trial fields, trials whose behavior traces do not cover the analysis window, trials whose interpolated behavior still contains non-finite values, sessions with no whisker motion-energy camera, sessions with no surviving neurons, and sessions with fewer than 2 trials or fewer than 5 neurons. Unexpected `probabilityLeft` values raise an error.

ii.
```python
if cam_used is None:
    raise RuntimeError('no whisker motion energy available (neither camera)')
```

```python
if len(tv) == 0:
    continue
if np.abs(interval_begs[k] - tt[0]) > binsize:
    continue
if np.abs(interval_ends[k] - tt[-1]) > binsize:
    continue
if not np.all(np.isfinite(y)):
    continue
```

```python
if len(keep_idx) < MIN_TRIALS_PER_SESSION:
    raise RuntimeError(f'only {len(keep_idx)} usable trials')
...
if neural.shape[1] < MIN_NEURONS_PER_SESSION:
    raise RuntimeError(f'only {neural.shape[1]} well-isolated grey-matter neurons '
                       f'(< {MIN_NEURONS_PER_SESSION})')
```

iii. The notes justify this as necessary because the downstream decoder expects finite, complete outputs for every retained trial, and because required outputs like whisker motion energy cannot be missing.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike loading and spike binning as the dominant costs, especially `SpikeSortingLoader.load_spike_sorting()` and per-session spike processing across probes.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

```python
timings['spikes'] = time.time() - t0
```

iii. The notes state that spike-file I/O dominates runtime, with behavior loading/binning much cheaper.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI says it already vectorized the expensive spike-binning logic compared with the reference, but `bin_behaviour()` still loops over trials, and there are still ordinary per-probe/per-trace loops in `convert_session()`.

ii.
```python
for k in range(n_trials):
    if np.isnan(interval_begs[k]) or np.isnan(interval_ends[k]):
        continue
    tt = target_times[idxs_beg[k]:idxs_end[k]]
```

```python
for pid, pname in zip(session_info['pids'], session_info['probe_names']):
    times, clu, clusters_df, collection = load_spiking_data(one, pid, eid, pname)
```

iii. The notes justify the main vectorization work as targeting the true bottleneck, while leaving simpler session/probe/trial loops where they were not the dominant cost.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats some setup and interpolation work per session: it constructs a new ONE client inside every worker call, re-instantiates `BrainRegions`, and recomputes behavior interpolation independently for each continuous output.

ii.
```python
def get_one():
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org',
               silent=True, cache_dir=ONE_CACHE_DIR)
```

```python
one = get_one()
...
br = BrainRegions()
```

```python
for name, (tt, tv) in traces.items():
    vals, good = bin_behaviour(tt, tv, interval_begs)
```

iii. The notes mostly frame repeated processing relative to the reference code, but the code itself still repeats these session-local setup steps because sessions are processed independently.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes optional diagnostic plotting and mandatory summary/statistics passes that are not part of the saved decoder inputs/outputs. It also collects timing and metadata fields such as sorter names and discretization edges that are not used by `train_decoder.py`.

ii.
```python
if show_processing:
    _plot_processing(plot_dir, eid, trials_df, keep_idx, interval_begs, traces,
                     beh_vals, ws_edges, wme_edges, neural, inputs, outputs,
                     regions, cam_used)
```

```python
for i, name in enumerate(OUTPUT_NAMES):
    vals = np.concatenate([np.asarray(r['output'])[:, i, :].ravel() for r in results])
    frac = np.bincount(vals, minlength=len(OUTPUT_VALUES[i])) / len(vals)
```

```python
'spike_sorter': sorted(sorters),
'wheel_speed_edges': ws_edges,
'whisker_me_edges': wme_edges,
'timings': timings,
'runtime': time.time() - t_start,
```

iii. The notes justify these extras as validation and documentation aids rather than required downstream features.
