# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates sessions by reading the release table `bwm_release.csv`, then loads per-session data through ONE and brainbox. For each session `eid`, it uses `SessionLoader` to load trials, wheel, and motion energy, and `SpikeSortingLoader` to load spike sorting for each probe.

ii.
```python
BWM_RELEASE = '/app/code/code_zhang2025/data/bwm_release.csv'
...
bwm = pd.read_csv(BWM_RELEASE, index_col=0)
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
```

```python
sess_loader = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
...
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. In `CONVERSION_NOTES.md`, the AI says it intentionally keeps all actual data access inside ONE/brainbox because the staged cache includes revised datasets that local file lookups would miss. It chose the release CSV only to enumerate the 459 released sessions.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the release table metadata. Each session row already contains a `subject`, and the final dataset stores sorted unique subjects plus a per-session index into that list.

ii.
```python
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
...
subjects = sorted({r['subject'] for r in good})
sub_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
```

iii. The notes justify this as using the release metadata directly rather than parsing paths or filenames.

## 1-c. How are the data split into sessions?

i. The AI treats each unique `eid` in the release table as one session. Session processing happens one `eid` at a time in `process_session`.

ii.
```python
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
...
jobs = [(r.eid, r.subject, r.lab, args.show_processing) for r in sess_df.itertuples()]
```

```python
def process_session(args):
    """Convert a single session. Returns a dict (or a dict with 'skip' set)."""
    eid, subject, lab, show_processing = args
```

iii. The notes say the BWM release is session-organized already, so the AI preserved that unit.

## 1-d. How are the data split into trials?

i. Trials come from the session trials table loaded by `SessionLoader`, with one row per trial. Later processing uses boolean masks over those rows.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
n_trials_raw = len(trials)
```

iii. The notes describe the trials table as the native trial segmentation, so the AI does not invent trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI starts from the reference `load_trials_and_mask(max_trial_len=10.0)` mask, then further requires finite alignment times, valid wheel coverage, valid whisker-motion-energy coverage, and nonempty neural activity in the aligned window. Sessions with fewer than two usable trials are dropped.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
...
ws, ws_valid = bin_behavior(wheel_times, wheel_speed, safe_t0)
...
camera_used, me, me_valid = best
...
keep = mask & finite_align & ws_valid & me_valid
...
nonempty = counts.sum(axis=(1, 2)) > 0
...
if n_keep < MIN_TRIALS:
    return {'eid': eid, 'skip': f'only {n_keep} usable trials'}
```

iii. The notes justify this as matching the reference trial mask, enforcing the intended conjunction of trial and behavior validity masks, and dropping trials in ephys gaps because they carry no neural information.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is derived from `spikes['times']` and `spikes['clusters']`, with `clusters['label']` and `clusters['acronym']` used for QC and region assignment.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
...
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
...
st = spikes['times'][smask]
sc = spikes['clusters'][smask]
```

iii. The notes say this follows the reference loading path: spike times and cluster assignments define the neural activity; cluster metadata only controls filtering and annotation.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, filters units, sorts spikes by time, then bins spikes into 20 ms counts in a fixed 2 s window around stimulus onset. Unlike the human reference solution, it keeps raw spike counts rather than dividing by bin width to convert to Hz.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
...
order = np.argsort(st, kind='stable')
return (st[order], sc[order], np.asarray(beryl)[keep_ids], keep_ids.size,
        skipped_probes, n_clusters_all, len(clusters), len(pids))
```

```python
def bin_spikes(spike_times, spike_clusters, t0s, n_neurons, nbins=NBINS, binsize=BINSIZE):
    ...
    for k in range(ntrials):
        ...
        idx = spike_clusters[a:b] * nbins + bins
        counts = np.bincount(idx, minlength=n_neurons * nbins)
        out[k] = counts.reshape(n_neurons, nbins).astype(np.float32)
    return out
```

iii. In the notes, the AI says it matched the reference binning geometry but deliberately stored counts because the decoder standardizes downstream and counts are what the reference caching code naturally produces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1`, maps regions to Beryl, drops Beryl `root` and `void`, skips probes with no spike sorting, and drops sessions with fewer than 5 surviving units.

ii.
```python
def load_spiking_data(one, pid, eid, pname, qc=QC_LABEL):
    ...
    iok = clusters_labeled['label'] >= qc
```

```python
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
keep = ~np.isin(beryl, NON_GREY)
...
if n_units < MIN_NEURONS:
    return {'eid': eid,
            'skip': f'only {n_units} well-isolated grey-matter units '
                    f'(< {MIN_NEURONS})'}
```

iii. The notes justify `label >= 1` as the data paper’s “well-isolated neurons” criterion, `root`/`void` removal as a grey-matter restriction, and the 5-neuron minimum as a session-level quality requirement from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to `trials.stimOn_times`. For each trial, the binning window starts 0.5 s before stimulus onset and ends 1.5 s after.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
t0s_all = align + TIME_WINDOW[0]
...
counts = bin_spikes(spike_times, spike_clusters, t0s, n_units)
```

iii. The notes cite the reference caching script’s `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)` as the reason.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 20 ms bins over a 2 s window, giving 100 bins per trial. There is no later temporal rebinning.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

```python
bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
np.clip(bins, 0, nbins - 1, out=bins)
```

iii. The notes tie this directly to the reference code and papers, both of which use 20 ms bins and 100 time steps for a 2 s trial.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the chosen alignment event `stimOn_times` plus the fixed decoding window and bin size. The actual stored values are relative bin centers, not raw timestamps copied from the trials table.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
...
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
```

iii. The notes say this input is defined by the decoder task itself once stimulus onset, window, and binning are fixed.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes the 100 bin centers from -0.49 s to 1.49 s and broadcasts that same vector to every trial.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
...
inputs.append(np.stack([bin_centers,
                        np.full(NBINS, tib[i], dtype=np.float32)]))
```

iii. The notes justify this as the natural time-varying input required by the decoder format.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same trial window and 20 ms binning as the neural data; the input values are the neural bins’ centers.

ii.
```python
t0s_all = align + TIME_WINDOW[0]
counts = bin_spikes(spike_times, spike_clusters, t0s, n_units)
...
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
```

iii. The notes state that the time input and neural matrix share one common bin grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`; block boundaries are inferred whenever that value changes.

ii.
```python
def trial_number_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=np.float64)
    changed = np.ones(len(pl), dtype=bool)
    changed[1:] = ~(pl[1:] == pl[:-1])
```

iii. The notes say the trials table has no block ID, so `probabilityLeft` must be used to reconstruct blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI assigns a block ID to each consecutive run of constant `probabilityLeft`, computes a 0-based within-block counter over all trials, then subsets that vector to the kept trials and broadcasts it across time bins.

ii.
```python
block_id = np.cumsum(changed) - 1
idx = np.zeros(len(pl), dtype=np.int64)
for b in np.unique(block_id):
    m = block_id == b
    idx[m] = np.arange(m.sum())
return idx, block_id
```

```python
tib_all, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
tib = tib_all[keep].astype(np.float32)
```

iii. The notes explicitly justify computing it before trial filtering so the index reflects the animal’s true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials['choice']`.

ii.
```python
choice = trials['choice'].to_numpy()[keep]
# trials.choice: +1 = left, -1 = right  ->  left = 0, right = 1
choice_cls = (choice < 0).astype(np.int32)
```

iii. The notes document the IBL convention `+1 = left`, `-1 = right`, `0 = no-go`, and say no-go trials are already filtered out.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI recodes choice to `0=left`, `1=right` and broadcasts the per-trial label across all 100 time bins.

ii.
```python
choice_cls = (choice < 0).astype(np.int32)
...
outputs.append(np.stack([
    np.full(NBINS, choice_cls[i], dtype=np.int32),
    ...
]))
```

iii. The notes say this matches the decoder task’s requested coding while preserving a uniform `(d_output, T)` shape.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials['probabilityLeft']`.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[keep]
prior_cls = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int32)
```

iii. The notes say these are the three block priors used by the task and required by the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, aborts a session if any other value appears, and broadcasts the class across time.

ii.
```python
prior_cls = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int32)
if np.any(prior_cls < 0):
    return {'eid': eid, 'skip': 'unexpected probabilityLeft values'}
```

```python
outputs.append(np.stack([
    ...,
    np.full(NBINS, prior_cls[i], dtype=np.int32),
    ...
]))
```

iii. The notes treat this as a direct task-driven recoding, not an additional analysis step.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It comes from the wheel stream loaded by `SessionLoader.load_wheel()`, specifically `wheel['velocity']`, which the AI converts to absolute speed.

ii.
```python
sess_loader.load_wheel()
wheel_times = sess_loader.wheel['times'].to_numpy()
wheel_speed = np.abs(sess_loader.wheel['velocity'].to_numpy())
```

iii. The notes say this follows the reference behavior loader, which defines wheel speed as `abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI uses the wheel velocity already produced by `SessionLoader`, takes its absolute value, interpolates it onto one value per 20 ms neural bin, and later discretizes it into three classes.

ii.
```python
ws, ws_valid = bin_behavior(wheel_times, wheel_speed, safe_t0)
```

```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
...
interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
```

iii. The notes justify this as matching the reference `get_behavior_per_interval` interpolation strategy while using the precomputed wheel velocity from `SessionLoader`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is discretized into three per-session tertile bins using the 33.3rd and 66.7th percentiles of the kept wheel-speed samples from that session.

ii.
```python
def discretize_tertiles(values):
    edges = np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])
    classes = np.digitize(values, edges, right=False).astype(np.int32)
    return classes, edges
...
ws_cls, ws_edges = discretize_tertiles(ws_k)
```

iii. The notes justify per-session tertiles because whisker motion energy is camera-dependent in scale and because balanced class frequencies are helpful for balanced-accuracy decoding; wheel speed uses the same rule for consistency.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated on the same trial windows as the neural data, with one wheel sample per neural bin. The AI evaluates the behavior at each bin’s right edge rather than its center.

ii.
```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
```

iii. The notes explicitly justify right-edge interpolation as matching the original reference function `get_behavior_per_interval`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It comes from camera motion-energy traces loaded with `SessionLoader.load_motion_energy`. The AI checks both left and right cameras when available and keeps the one with the best usable-trial coverage, with left preferred on ties.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        cameras[cam] = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
    except Exception:
        continue
```

```python
best = None
for cam in ('leftCamera', 'rightCamera'):
    if cam not in cameras:
        continue
    me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
    if best is None or valid_c.sum() > best[2].sum():
        best = (cam, me_c, valid_c)
camera_used, me, me_valid = best
```

iii. The notes say this deviates from the literal left-then-right fallback because one session had left camera data but only the right camera covered usable trials; the AI chose better coverage to avoid losing that session.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy trace as-is, interpolates it onto one sample per neural bin, and discretizes it into three per-session tertile bins.

ii.
```python
me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
...
me_cls, me_edges = discretize_tertiles(me_k)
```

iii. The notes say no additional filtering or normalization is applied beyond resampling and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: 3 session-specific tertile bins using the 33.3rd and 66.7th percentiles of the kept whisker-motion-energy samples.

ii.
```python
def discretize_tertiles(values):
    edges = np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])
    classes = np.digitize(values, edges, right=False).astype(np.int32)
    return classes, edges
...
me_cls, me_edges = discretize_tertiles(me_k)
```

iii. The notes emphasize that whisker motion energy has arbitrary camera-dependent scale, so per-session thresholds are preferable to global thresholds.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned exactly like wheel speed: the chosen camera’s motion-energy trace is interpolated onto the neural trial windows with one value per 20 ms bin, evaluated at the bin right edge.

ii.
```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
```

iii. The notes justify the timing grid as matching the original reference behavior-interpolation code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly drops unusable data. It skips probes without spike sorting, skips sessions with no whisker motion energy or too few surviving units, drops trials failing the reference mask, drops trials lacking wheel or camera coverage, and drops trials whose aligned neural window has zero spikes because they fall in ephys gaps or after ephys ended.

ii.
```python
if not clusters or not spikes:
    return None, None
```

```python
if not cameras:
    return {'eid': eid, 'skip': 'no whisker motion energy'}
...
if neural is None:
    return {'eid': eid, 'skip': 'no units pass QC'}
...
keep = mask & finite_align & ws_valid & me_valid
...
nonempty = counts.sum(axis=(1, 2)) > 0
```

iii. The notes frame these as practical corrections for missing or invalid data rather than attempts to impute or repair it.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike-sorting loading as the dominant cost, with binning itself comparatively cheap.

ii.
```python
t = time.time()
neural = load_session_neural(one, eid)
timings['spikes'] = time.time() - t
```

iii. In the notes, it reports 3.5-8.3 s/session for spike loading and only 0.01-0.03 s/session for binning.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized much of the behavior interpolation and spike binning compared with the reference, but it still keeps per-trial loops inside `bin_spikes`, `bin_behavior` validity checks, and the final Python-list assembly of inputs/outputs/trials.

ii.
```python
for k in range(ntrials):
    a, b = i0[k], i1[k]
    if b <= a:
        continue
    ...
```

```python
for k in range(ntrials):
    a, b = i_beg[k], i_end[k]
    if b <= a:
        continue
    ...
```

```python
for i in range(n_keep):
    inputs.append(np.stack([bin_centers,
                            np.full(NBINS, tib[i], dtype=np.float32)]))
    outputs.append(np.stack([
        np.full(NBINS, choice_cls[i], dtype=np.int32),
        ...
    ]))
    neural_trials.append(np.ascontiguousarray(counts[i]))
```

iii. The notes explicitly say the AI rewrote the main binning steps to be more vectorized than the reference implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code bins whisker motion energy for both cameras when both are present and discards one result after comparing coverage. It also repeatedly builds small per-trial arrays in Python after larger session-level arrays already exist.

ii.
```python
for cam in ('leftCamera', 'rightCamera'):
    if cam not in cameras:
        continue
    me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
    if best is None or valid_c.sum() > best[2].sum():
        best = (cam, me_c, valid_c)
```

```python
for i in range(n_keep):
    inputs.append(np.stack([bin_centers,
                            np.full(NBINS, tib[i], dtype=np.float32)]))
    outputs.append(np.stack([
        np.full(NBINS, choice_cls[i], dtype=np.int32),
        np.full(NBINS, prior_cls[i], dtype=np.int32),
        ws_cls[i].astype(np.int32),
        me_cls[i].astype(np.int32)]))
```

iii. The notes acknowledge the camera-selection pass as an extra step introduced for robustness, not because it is part of the original reference pipeline.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes diagnostic timings, optional plotting outputs, and a large amount of session-level metadata that are not used by the decoder itself. When both cameras are present, it also bins both whisker traces even though only one is kept.

ii.
```python
timings = {}
...
timings['trials'] = time.time() - t
...
timings['bin_behavior'] = time.time() - t
...
timings['bin_spikes'] = time.time() - t
```

```python
if show_processing:
    plot_processing(out, trials, keep, ws_k, me_k, wheel_times, wheel_speed,
                   cameras[camera_used][0], cameras[camera_used][1], align)
```

```python
'session_info': [
    {'eid': r['eid'], 'subject': r['subject'], 'lab': r['lab'],
     ...
     'wheel_speed_tertile_edges': r['wheel_edges'],
     'whisker_me_tertile_edges': r['whisker_edges']}
    for r in good],
```

iii. The notes present these as validation and audit aids rather than part of the downstream decoder inputs/outputs.
