# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the session list from the reference code's `bwm_release.csv` file (459 eids from the BWM release freeze), then optionally restricts to `DATALIMIT_SUBSET.csv` if present. For each session, a new `ONE` client is created (per worker process), and `SpikeSortingLoader` and `SessionLoader` are used to load spike sorting, trials, wheel, and motion energy data from the local ONE cache. Sessions are processed in parallel using `multiprocessing.Pool` with spawn context.

ii.
```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
if os.path.exists(DATALIMIT):
    sub = pd.read_csv(DATALIMIT)
    col = 'eid' if 'eid' in sub.columns else sub.columns[0]
    bwm = bwm[bwm.eid.isin(sub[col].astype(str))]

eids = list(dict.fromkeys(bwm.eid.tolist()))
```

```python
one = get_one()
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
sp, cl, ch = ssl.load_spike_sorting()
sl = SessionLoader(one=one, eid=eid)
```

iii. The agent examined the reference code's `0_data_caching.py` which reads sessions from `bwm_release.csv`, and adopted the same session list. It confirmed that the ONE API works offline against the staged cache with no network access needed.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the `bwm_release.csv` rows for each session (`rows.subject.iloc[0]`). During assembly, a list of unique subjects is built in insertion order, and `subject_idx` maps each session to its index in that list.

ii.
```python
'subject': str(rows.subject.iloc[0]),
```
Assembly:
```python
if res['subject'] not in subjects:
    subjects.append(res['subject'])
data['subject_idx'].append(subjects.index(res['subject']))
```

iii. The subject name is available directly from `bwm_release.csv` for each session, so no parsing of paths is needed.

## 1-c. How are the data split into sessions?

i. Sessions are the unique eids from `bwm_release.csv`. Each eid is processed independently, with the `bwm_release.csv` rows for that eid (containing probe information) passed along.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))
jobs = [(eid, bwm[bwm.eid == eid]) for eid in eids ...]
```

iii. Sessions are already the unit of organization in the BWM release, so no splitting is needed.

## 1-d. How are the data split into trials?

i. The trials table from `SessionLoader.load_trials()` has one row per trial; trials are the natural unit. After applying the trial mask, each trial gets one slice of neural, input, and output data.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
trials, trials_mask = load_trials_and_mask(sl)
```

iii. No decision to make; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI reimplements the reference code's `load_trials_and_mask` function with `max_trial_len=10.0`. Trials are excluded if: any of `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType` is NaN; reaction time (firstMovement_times - stimOn_times) is outside [0.08, 2.0] seconds; choice == 0 (no response); or feedback_times - goCue_times > 10 seconds. Additionally, trials where wheel or whisker traces don't cover the decoding window are dropped (via `bin_behavior`'s coverage checks).

ii.
```python
nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
query = f'(firstMovement_times - stimOn_times < {min_rt})'
query += f' | (firstMovement_times - stimOn_times > {max_rt})'
if max_trial_len is not None:
    query += f' | (feedback_times - goCue_times > {max_trial_len})'
for event in nan_exclude:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
mask = ~trials.eval(query)
```

Coverage check in `bin_behavior`:
```python
if abs(interval_begs[k] - tt[0]) > BINSIZE:      # data starts too late
    continue
if abs(interval_ends[k] - tt[-1]) > BINSIZE:     # data ends too early
    continue
```

iii. The agent stated it follows "the standard BWM trial mask (`load_trials_and_mask`) with `max_trial_len=10 s`, exactly as `prepare_data` calls it." The coverage checks mirror the reference code's `get_behavior_per_interval`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` (spike timestamps) and `spikes.clusters` (cluster assignments). The cluster table provides the quality label and anatomical location for filtering.

ii.
```python
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
sp, cl, ch = ssl.load_spike_sorting()
```

```python
spike_times = np.concatenate([s['times'] for s in ms])
spike_clusters = np.concatenate([s['clusters'] for s in ms])
```

iii. The agent identified spike times and clusters as the primary data source from examining the reference code.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window [-0.5, 1.5] s around stimulus onset, giving one count per unit per bin. The neural data is stored as **raw spike counts** (not divided by bin width), as float32. When a session has multiple probes, units are merged by offsetting cluster IDs.

ii.
```python
out = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
# ...
tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
np.clip(tb, 0, N_BINS - 1, out=tb)
flat = sc[b:e] * N_BINS + tb
counts = np.bincount(flat, minlength=n_clusters * N_BINS)
out[k] = counts.reshape(n_clusters, N_BINS)
```

Probe merging:
```python
cmax = 0
for cl, sp in zip(clusters_list, spikes_list):
    sp = dict(sp)
    sp['clusters'] = sp['clusters'] + cmax
    cmax = cl.index.max() + 1
```

iii. The agent noted the metadata says `'neural_units': 'spike counts per 20 ms bin'`. It described the binning as "Equivalent to ibl_data_utils.get_spike_data_per_interval (bincount2D with xlim=[t_beg, t_end])."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with IBL label >= 1 (well-isolated neurons) are kept. Additionally, neurons mapped to 'root' or 'void' in the Beryl atlas are excluded. This differs from the reference, which keeps 'root' but drops 'void'.

ii.
```python
beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
label = clusters['label'].to_numpy()
keep = (label >= 1) & ~np.isin(beryl, ['root', 'void'])
```

iii. The agent justified the QC filter: "well-isolated neurons only ... the conjunction of the three RIGOR single-unit metrics used by the data paper ... restricted to grey matter (Beryl acronym not 'root'/'void'), again as in the data paper." The agent also noted the reference caching script keeps every Kilosort unit, but argued that 622k units is infeasible (~164 GB) while well-isolated neurons (~75.7k) produce ~11 GB.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset (`stimOn_times`). The interval begins at `stimOn_times + TIME_WINDOW[0]` (i.e. stimOn - 0.5s). Spike times falling in this interval are binned relative to the interval start.

ii.
```python
ALIGN_TIME = 'stimOn_times'
align = trials[ALIGN_TIME].to_numpy(dtype=float)
interval_begs = align + TIME_WINDOW[0]
# In bin_spikes:
tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
```

iii. The agent stated: "align to stimOn_times, window (-0.5,1.5)s, 20ms bins (T=100) exactly as in the zhang2025 caching params."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (BINSIZE = 0.02), producing 100 time bins over the 2s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BINSIZE = 0.02                     # seconds
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

iii. The agent adopted the exact parameters from the reference code's `0_data_caching.py`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time values are the right edges of the bins, computed as `np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)`.

ii.
```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The agent commented: "These are the right edges of the bins, i.e. the sample times used by `get_behavior_per_interval` in the reference code (x_interp = linspace(beg + binsize, end, n_bins))."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are a fixed array of 100 right-edge bin times from -0.48s to 1.5s, broadcast identically to every trial.

ii.
```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. The agent described these as "the sample times used by `get_behavior_per_interval` in the reference code."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values represent the right edges of the same bins the spikes are counted into, so they are aligned by construction.

ii.
```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
# These are the right edges of the 20ms bins starting from TIME_WINDOW[0]
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A new block starts whenever `probabilityLeft` changes value.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy(dtype=float)
new_block = np.ones(len(pleft), dtype=bool)
new_block[1:] = pleft[1:] != pleft[:-1]
block_id = np.cumsum(new_block) - 1
```

iii. The agent derived blocks from changes in `probabilityLeft`, same logic as the reference.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based trial index within each block is computed on the full (unmasked) trials table, then subselected to valid trials. The value is broadcast to all 100 time bins.

ii.
```python
trial_in_block = np.zeros(len(pleft), dtype=np.int64)
for b in np.unique(block_id):
    idx = np.nonzero(block_id == b)[0]
    trial_in_block[idx] = np.arange(len(idx))
# ...
tinb = trial_in_block[tidx]
inputs[:, 1, :] = tinb[:, None].astype(np.float32)
```

iii. The trial number is computed before filtering so that a dropped trial still advances the count, preserving the animal's real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, where +1 = left and -1 = right in IBL convention. No-response trials (choice == 0) are dropped by the trial mask.

ii.
```python
choice = trials['choice'].to_numpy()[tidx]
out_choice = (choice < 0).astype(np.int64)          # left = 0, right = 1
```

iii. The agent verified the choice convention empirically by checking correct trials with left vs. right stimuli.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL +1/-1 coding is remapped to 0 (left) / 1 (right) via `(choice < 0).astype(np.int64)`. The value is broadcast to all 100 time bins.

ii.
```python
out_choice = (choice < 0).astype(np.int64)
outputs[:, 0, :] = out_choice[:, None]
```

iii. N/A

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
out_prior = np.array([prior_map[round(float(p), 1)] for p in pleft_t], dtype=np.int64)
```

iii. The three values are the block prior probabilities, and the instructions specify the mapping 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probability values are mapped to discrete categories {0, 1, 2} using a dictionary. The value is broadcast to all 100 time bins.

ii.
```python
out_prior = np.array([prior_map[round(float(p), 1)] for p in pleft_t], dtype=np.int64)
outputs[:, 1, :] = out_prior[:, None]
```

iii. N/A

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel position and timestamps loaded via `SessionLoader.load_wheel()`. Speed is the absolute value of velocity.

ii.
```python
sl.load_wheel()
wheel_speed_t = sl.wheel['times'].to_numpy()
wheel_speed_v = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. The agent noted this matches the reference's `load_target_behavior` for 'wheel-speed' which uses `np.abs` of velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` internally interpolates position to 1000 Hz and applies a Butterworth low-pass filter to compute velocity. Speed is `abs(velocity)`. The continuous trace is interpolated onto the trial bin times using `scipy.interpolate.interp1d` with linear interpolation and extrapolation. It is then discretized into 3 bins using session-wise tertiles (quantiles at 1/3 and 2/3).

ii.
```python
ws, ws_good = bin_behavior(wheel_speed_t, wheel_speed_v, interval_begs)
out_ws = discretize3(ws)
```

Discretization:
```python
def discretize3(x):
    q = np.quantile(x, [1. / 3., 2. / 3.])
    edges = np.unique(q)
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The agent justified per-session tertiles because "whisker energy is in camera-specific arbitrary units ... and wheel speed also varies in scale across rigs."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-wise tertiles: the 1/3 and 2/3 quantiles of all wheel speed values across all time bins and trials of the session define the bin edges. `np.unique` is applied to the edges, and `np.searchsorted` with `side='right'` assigns categories.

ii.
```python
def discretize3(x):
    q = np.quantile(x, [1. / 3., 2. / 3.])
    edges = np.unique(q)
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The agent noted the per-session split ensures "the three classes are equally sized within a session."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated onto the same bin times as the neural data (right edges of bins relative to stimulus onset), so they share the same time axis.

ii.
```python
x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

iii. The interpolation onto the bin times follows the reference code's `get_behavior_per_interval`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The ROI motion energy from the side camera (`leftCamera.ROIMotionEnergy` or `rightCamera.ROIMotionEnergy`), with left camera preferred. Loaded via `SessionLoader.load_motion_energy()`.

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

iii. The agent followed the reference code which loads 'left-whisker-motion-energy' and falls back to 'right-whisker-motion-energy'.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is with no additional filtering or normalization. It is interpolated onto trial bin times using `interp1d` with linear interpolation, then discretized into 3 bins using session-wise tertiles (same as wheel speed).

ii.
```python
wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)
out_wm = discretize3(wm)
```

iii. Same processing as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: session-wise tertiles at 1/3 and 2/3 quantiles.

ii.
```python
out_wm = discretize3(wm)
```

iii. Same justification as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto the right edges of the 20ms bins relative to stimulus onset.

ii.
```python
wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)
```

iii. Same approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms: (1) Sessions where spike sorting cannot be loaded are skipped. (2) Sessions with no whisker motion energy (left or right camera) are skipped. (3) Trials with NaN in key fields are excluded by the trial mask. (4) Trials where behavioral traces don't cover the decoding window are dropped. (5) Sessions with fewer than 2 valid trials or fewer than 1 neuron are skipped. (6) Exceptions during session processing are caught and the session is skipped with an error message.

ii.
```python
if sp is None or len(sp) == 0:
    continue
# ...
if whisker_t is None:
    return {'eid': eid, 'skip': 'no whisker motion energy'}
# ...
if valid.sum() < MIN_TRIALS_PER_SESSION:
    return {'eid': eid, 'skip': f'only {int(valid.sum())} trials ...'}
# ...
except Exception:
    return {'eid': eid, 'skip': 'exception: ' + traceback.format_exc().splitlines()[-1]}
```

iii. The agent noted: "13 skipped for missing whisker motion energy, 1 for no trials with complete behaviour."

## 10-a. What are the most time-consuming steps of the code?

i. Loading the spike sorting from disk (spike times and cluster arrays, hundreds of MB per probe) and binning spikes into the trial-by-neuron-by-time arrays.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
# ...
out = bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs[tidx])
```

iii. The agent noted processing took ~2.3s per session per worker, dominated by I/O.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates per trial (`for k in range(n_trials)`) with `np.bincount` inside. This could potentially be vectorized into a single bincount call over all trials by offsetting indices. Similarly, `bin_behavior` loops per trial with `interp1d`.

ii.
```python
for k in range(n_trials):
    # ...
    flat = sc[b:e] * N_BINS + tb
    counts = np.bincount(flat, minlength=n_clusters * N_BINS)
    out[k] = counts.reshape(n_clusters, N_BINS)
```

```python
for k in range(n_trials):
    # ...
    vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

iii. The agent described its spike binner as "vectorised" compared to the reference's per-trial multiprocessing approach, though it still contains a per-trial loop.

## 10-c. What processing does the code repeat multiple times?

i. The ONE client is created fresh in every worker process (`get_one()`), which rebuilds the REST cache each time. The `BrainRegions()` atlas object is also created per worker.

ii.
```python
def _process_session(eid, rows):
    one = get_one()
    br = BrainRegions()
```

iii. Necessary due to spawn-based multiprocessing; objects cannot be shared across processes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads all spike sorting data including cluster metrics and channel information (`ssl.load_spike_sorting()` returns spikes, clusters, and channels), but only uses spike times, spike clusters, and cluster labels/acronyms. The full `merge_clusters` call computes additional metrics that are not used. The `n_trials_total` and `lab`/`date` fields in session info are computed but not required by the target format.

ii.
```python
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
# Only label and acronym are used from clusters
```

iii. N/A
