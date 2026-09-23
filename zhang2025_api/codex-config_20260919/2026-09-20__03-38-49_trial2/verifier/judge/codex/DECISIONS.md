# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script builds a local `ONE` client over `/app/data/one_cache`, but it does not use `one.search()` to discover sessions. Instead it reads the release table from `/app/code/code_zhang2025/data/bwm_release.csv`, optionally filters that table with `/app/data/DATALIMIT_SUBSET.csv`, groups rows by `eid`, then loads per-session trials/behavior with `SessionLoader` and per-probe spikes with `SpikeSortingLoader`.

ii.
```python
def get_one():
    """Construct a local ONE client over the aggregate release tables."""
    return ONE(cache_dir=CACHE, tables_dir=CACHE / 'Brainwidemap', mode='local')
```

```python
def release_rows():
    rows = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
    if SUBSET.exists():
        subset = pd.read_csv(SUBSET)
        ...
        rows = rows[rows.eid.astype(str).isin(subset[key].astype(str))]
    return rows
```

```python
sl = SessionLoader(one=one, eid=eid)
...
ssl = SpikeSortingLoader(pid=str(row.pid), eid=str(row.eid), pname=row.probe_name, one=one)
```

iii. The notes say the intent was to use the exact BWM release CSV and let ONE/brainbox do the actual data loading, with the subset CSV used only to restrict the release to the staged subset. The trajectory also shows the AI believed the cache metadata were inconsistent, so it leaned on the release CSV plus ONE loaders rather than pure ONE session discovery.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the release CSV rows. At assembly time, the script makes a sorted unique subject list and encodes each retained session with `subject_idx`.

ii.
```python
info = dict(eid=str(eid), subject=str(rows.subject.iloc[0]), ...)
```

```python
subjects = sorted({x['subject'] for x in infos}); subject_lookup = {s:i for i,s in enumerate(subjects)}
...
'subject_idx': np.asarray([subject_lookup[x['subject']] for x in infos], dtype=np.int64),
```

iii. The notes justify this as using the release table's authoritative session metadata rather than parsing paths.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the release CSV. All release rows for one `eid` are grouped together, so all probes for that session are processed together.

ii.
```python
rows = release_rows(); one = get_one()
grouped = list(rows.groupby('eid', sort=False))
```

```python
for k, (eid, erows) in enumerate(grouped, 1):
    vals = process_session(one, str(eid), erows, ...)
```

iii. The notes state that the exact release EIDs and probe membership from `bwm_release.csv` should define the candidate sessions.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the table loaded by `SessionLoader.load_trials()`. Each row in `sl.trials` is treated as one trial, and valid-trial indices are selected from that table.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
tr = sl.trials.copy()
```

```python
onsets = tr.stimOn_times.to_numpy(dtype=float)
...
idx = np.flatnonzero(valid)
```

iii. This follows the standard IBL trials table semantics; the notes describe `SessionLoader.load_trials` as the source of per-trial task variables.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials only if required trial columns are present and finite, reaction time is between 0.08 s and 2.0 s, trial duration from `goCue_times` to `feedback_times` is at most 10 s, choice is nonzero, and both wheel and whisker traces can be interpolated over the full analysis window. It also drops sessions with fewer than two jointly valid trials.

ii.
```python
finite = np.ones(len(tr), dtype=bool)
for col in ['stimOn_times', 'choice', 'probabilityLeft', 'firstMovement_times',
            'feedback_times', 'feedbackType']:
    finite &= np.isfinite(tr[col].to_numpy(dtype=float))
rt = tr.firstMovement_times.to_numpy() - tr.stimOn_times.to_numpy()
duration = tr.feedback_times.to_numpy() - tr.goCue_times.to_numpy()
mask = finite & (rt >= .08) & (rt <= 2.) & (duration <= 10.) & (tr.choice.to_numpy() != 0)
```

```python
wheel, wheel_good = interpolate_trials(wt, ws, onsets)
whisk, whisk_good = interpolate_trials(mt, me, onsets)
valid = mask & wheel_good & whisk_good
...
if idx.size < 2:
    raise RuntimeError(f'only {idx.size} jointly valid trials')
```

iii. The notes say this was meant to reproduce the reference trial QC, plus full wheel/whisker coverage and the `<=10 s` duration rule from the supplied cache code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from spike times and cluster assignments loaded per probe. Region labels come from merged cluster metadata, but the actual neural arrays are built from `spikes['times']` and `spikes['clusters']`.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
```

```python
all_times.append(np.asarray(spikes['times'], dtype=float))
all_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
all_regions.extend(np.asarray(clusters['acronym']).astype(str).tolist())
```

iii. The notes explicitly map `spikes.times` and `spikes.clusters` from every release probe to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. The script merges probes within a session, sorts all spikes by time, then bins spikes into 100 stimulus-aligned 20 ms bins per trial. It stores spike counts as `float32`; it does not divide by bin width to convert them to firing rates.

ii.
```python
times = np.concatenate(all_times); clu = np.concatenate(all_clusters)
order = np.argsort(times, kind='stable')
return times[order], clu[order], np.asarray(all_regions, dtype=str)
```

```python
for onset in onsets:
    beg, end = onset + OFF_START, onset + OFF_END
    lo, hi = np.searchsorted(times, [beg, end])
    relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
    ...
    counts = np.bincount(flat, minlength=n_neurons * N_TIME).reshape(n_neurons, N_TIME)
    result.append(counts.astype(np.float32))
```

iii. The notes say the AI intentionally chose “spike counts per 20-ms bin” and described this as matching the Zhang caching code structure while satisfying the decoder format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not apply neuron QC cuts. It keeps all spike-sorted clusters returned by `SpikeSortingLoader`, does not filter on `clusters['label']`, and does not drop `void` regions before building neural arrays.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
n = len(clusters['channels'])
all_times.append(np.asarray(spikes['times'], dtype=float))
all_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
all_regions.extend(np.asarray(clusters['acronym']).astype(str).tolist())
```

```python
'cluster_filter': 'all spike-sorted clusters, matching Zhang et al. caching code',
```

iii. The notes explicitly justify this choice by saying the published decoder caching pipeline uses `qc=None`, so the AI chose “all clusters” despite the data paper's stricter QC counts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For each kept trial, the script uses a window from `stimOn_times - 0.5` s to `stimOn_times + 1.5` s and bins spikes relative to that trial-specific onset.

ii.
```python
OFF_START, OFF_END = -0.5, 1.5
...
beg, end = onset + OFF_START, onset + OFF_END
lo, hi = np.searchsorted(times, [beg, end])
relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```

iii. The notes say stimulus-onset alignment with a common `[-0.5, 1.5]` s window was chosen because the downstream task explicitly required it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins, 100 bins per 2 s trial. No further temporal rebinning is applied.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_TIME = 100
```

```python
'time_bin_size': 20.0,
```

iii. The notes repeatedly cite the reference 20 ms binning and 100-step stimulus-aligned window.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the per-trial `stimOn_times` event together with a fixed relative time grid. The input row itself is the same `TIME` vector repeated for every trial.

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
```

```python
onsets = tr.stimOn_times.to_numpy(dtype=float)
...
inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
```

iii. The notes say the first decoder input should be the common stimulus-aligned time coordinate.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script does not compute this from raw samples. It defines a fixed 100-point vector from `-0.48` s to `1.50` s in 20 ms steps, corresponding to bin right edges, and copies that vector into every trial.

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
```

```python
inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
```

iii. The notes justify this as using “bin right edges as the reference behavior code does.”

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The AI uses the same trial onset and 20 ms step size for both neural binning and the time input. Neural bins are counted over `[onset-0.5, onset+1.5)`, while the time input records the right-edge coordinate of those bins.

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
```

```python
beg, end = onset + OFF_START, onset + OFF_END
relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```

iii. The notes explicitly describe the chosen time coordinate as right edges on the same stimulus-aligned grid used for neural and behavioral traces.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from `trials.probabilityLeft`, treating consecutive equal values as belonging to the same block.

ii.
```python
def trial_number_in_block(prob):
    out = np.zeros(len(prob), dtype=np.float32)
    for i in range(1, len(prob)):
        out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
    return out
```

```python
block_number = trial_number_in_block(tr.probabilityLeft.to_numpy())
```

iii. The notes say the trials table has no explicit block ID, so block position must be reconstructed from runs of `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a zero-based count within each block by scanning the `probabilityLeft` vector and resetting to 0 whenever the prior changes. It computes this before trial filtering and then repeats each scalar value across all 100 time bins.

ii.
```python
for i in range(1, len(prob)):
    out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
```

```python
inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
```

iii. The notes explicitly justify computing block position before filtering so excluded trials still advance the block count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials.choice`.

ii.
```python
choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
```

```python
out = np.vstack([np.full(N_TIME, choice), np.full(N_TIME, prior),
                 wheel_cat[j], whisk_cat[j]]).astype(np.int64)
```

iii. The notes map `trials.choice` directly to the requested categorical choice output, but the script itself does not restate the raw sign convention beyond the recode used in code.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script converts `trials.choice == -1` to category `0` and everything else that survives trial QC to category `1`, then repeats that category across all time bins of the trial.

ii.
```python
choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
...
out = np.vstack([np.full(N_TIME, choice), np.full(N_TIME, prior),
                 wheel_cat[j], whisk_cat[j]]).astype(np.int64)
```

iii. The notes say static outputs are repeated over time so all outputs share the same `n_output x n_timepoints` shape.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from `trials.probabilityLeft`.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
...
prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
```

iii. The notes justify this as the task-specified categorical recoding of the block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script rounds the floating `probabilityLeft` value to one decimal place, maps `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, and repeats the resulting category across all 100 time bins.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
```

```python
out = np.vstack([np.full(N_TIME, choice), np.full(N_TIME, prior),
                 wheel_cat[j], whisk_cat[j]]).astype(np.int64)
```

iii. The notes frame this as a direct implementation of the decoder-task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed comes from the wheel time series loaded by `SessionLoader.load_wheel()`. The script uses `sl.wheel.times` and the absolute value of `sl.wheel.velocity`, which `SessionLoader` computes from wheel position/timestamps.

ii.
```python
sl.load_wheel()
wt = sl.wheel.times.to_numpy(dtype=float)
ws = np.abs(sl.wheel.velocity.to_numpy(dtype=float))
```

iii. The notes explicitly say wheel speed should be absolute wheel velocity from `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script takes absolute wheel velocity, interpolates it onto the common stimulus-aligned time grid for every trial, and then categorizes the interpolated values.

ii.
```python
wheel, wheel_good = interpolate_trials(wt, ws, onsets)
```

```python
def interpolate_trials(times, values, onsets):
    query = onsets[:, None] + TIME[None, :]
    ...
    out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
```

iii. The notes justify this as matching the reference behavior loading and interpolation pipeline while putting the result on the decoder grid.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is thresholded within each session using tertiles computed over all retained trial-by-time samples. If the two quantile cut points collapse, the script falls back to a deterministic rank-based three-way split.

ii.
```python
def tertiles(x):
    edges = np.quantile(x, [1 / 3, 2 / 3])
    if edges[0] < edges[1]:
        return np.digitize(x, edges, right=False).astype(np.int64), edges
    # Deterministic rank fallback for degenerate signals.
    order = np.argsort(x.ravel(), kind='stable'); labels = np.empty(order.size, dtype=np.int64)
    labels[order] = np.minimum(2, np.arange(order.size) * 3 // order.size)
    return labels.reshape(x.shape), edges
```

```python
wheel_cat, wheel_edges = tertiles(wheel[idx])
```

iii. The notes explicitly justify within-session tertiles as robust to session-to-session scaling differences and say the fallback was added to avoid degenerate categories.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at `stimOn_times + TIME`, so it shares the same trial onset and 20 ms step size as the neural bins. As implemented, `TIME` is the bin right-edge coordinate.

ii.
```python
query = onsets[:, None] + TIME[None, :]
...
wheel, wheel_good = interpolate_trials(wt, ws, onsets)
```

```python
relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```

iii. The notes say all dynamic streams were intentionally put on the same stimulus-aligned grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy comes from `SessionLoader.load_motion_energy()`, using the left camera if available and otherwise the right camera. The script uses the motion-energy timestamps and `whiskerMotionEnergy` values from that chosen view.

ii.
```python
for view in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[view])
        df = sl.motion_energy[f'{view}Camera']
        return wt, ws, df.times.to_numpy(dtype=float), df.whiskerMotionEnergy.to_numpy(dtype=float), view
```

iii. The notes explicitly state “prefer left, fall back right; never average views.”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released whisker motion-energy trace is used directly, interpolated onto the common trial grid, and then discretized. The code does not add extra filtering or normalization.

ii.
```python
whisk, whisk_good = interpolate_trials(mt, me, onsets)
```

```python
out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
```

iii. The notes justify this as matching the reference code path for camera motion energy while adapting only the final categorical output format.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is thresholded exactly like wheel speed: within-session tertiles across all retained trial-by-time samples, with a deterministic rank fallback if the quantiles are degenerate.

ii.
```python
whisk_cat, whisk_edges = tertiles(whisk[idx])
```

```python
def tertiles(x):
    edges = np.quantile(x, [1 / 3, 2 / 3])
    if edges[0] < edges[1]:
        return np.digitize(x, edges, right=False).astype(np.int64), edges
    ...
```

iii. The notes describe this as the same justified discretization policy used for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is interpolated at `stimOn_times + TIME`, so it uses the same trial onset and 20 ms step size as the neural bins. As with wheel, the coordinate stored in `TIME` is the bin right edge.

ii.
```python
query = onsets[:, None] + TIME[None, :]
...
whisk, whisk_good = interpolate_trials(mt, me, onsets)
```

```python
beg, end = onset + OFF_START, onset + OFF_END
relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```

iii. The notes say the dynamic outputs were deliberately aligned to the same common stimulus grid as neural activity.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed session-level data cause errors and session skips: missing required trial columns raise `RuntimeError`, missing whisker views raise `RuntimeError`, and sessions with fewer than two jointly valid trials are skipped. Trial-level missingness is handled by masking out trials whose interpolated wheel/whisker windows are incomplete or contain NaNs.

ii.
```python
missing = NEEDED - set(tr.columns)
if missing:
    raise RuntimeError(f'trials object missing columns {sorted(missing)}')
```

```python
good = ((query[:, 0] >= times[0]) & (query[:, -1] <= times[-1]))
out = np.full(query.shape, np.nan, dtype=np.float32)
...
return out, good & np.isfinite(out).all(axis=1)
```

```python
if idx.size < 2:
    raise RuntimeError(f'only {idx.size} jointly valid trials')
```

iii. The notes describe this as “drop invalid trials, skip unusable sessions,” and the trajectory shows the AI treated missing source fields as a blocking cache defect rather than something to impute.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant costs are per-probe spike loading/cluster merging and then per-trial spike binning for each session. The script's own notes also mention behavior interpolation and redundant loads as performance concerns.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
```

```python
for onset in onsets:
    ...
    counts = np.bincount(flat, minlength=n_neurons * N_TIME).reshape(n_neurons, N_TIME)
```

iii. The notes' “Code inefficiencies identified” and “Code speedups added” sections point to spike I/O, interpolation, and avoiding redundant loads as the main runtime considerations.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The script already vectorizes behavior interpolation across all trials, but it still leaves explicit Python loops for `trial_number_in_block`, per-trial spike binning, and per-trial input/output assembly. Those are the main remaining vectorization opportunities.

ii.
```python
for i in range(1, len(prob)):
    out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
```

```python
for onset in onsets:
    ...
    result.append(counts.astype(np.float32))
```

```python
for j, raw_i in enumerate(idx):
    inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
    ...
    inputs.append(inp); outputs.append(out)
```

iii. The notes say the AI intentionally vectorized interpolation and used `np.bincount`, but otherwise kept a simple one-session-at-a-time implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several lightweight constructions per kept trial: it creates full-length repeated arrays for static variables (`choice`, `prior`, `trial_number_in_block`) and performs a separate spike-window/binning pass for every trial. It also stores raw acronyms session-by-session and only maps them to Beryl at the end.

ii.
```python
for j, raw_i in enumerate(idx):
    inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
    ...
    out = np.vstack([np.full(N_TIME, choice), np.full(N_TIME, prior),
                     wheel_cat[j], whisk_cat[j]]).astype(np.int64)
```

```python
for onset in onsets:
    beg, end = onset + OFF_START, onset + OFF_END
    ...
```

```python
mapped = [br.acronym2acronym(x, mapping='Beryl') for x in region_names_all]
```

iii. No strong explicit justification was recorded beyond preferring straightforward code and per-session processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes optional plotting code for debugging, records `wheel_edges` and `whisker_edges` in `session_info` even though the downstream decoder only needs categorized outputs, keeps a `failed_sessions` log in metadata, and carries raw cluster acronyms until a later whole-dataset Beryl remapping step.

ii.
```python
if show:
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ...
    fig.savefig(f'/app/processing_{eid}.png', dpi=140); plt.close(fig)
```

```python
info = dict(..., whisker_view=view,
            wheel_edges=wheel_edges.tolist(), whisker_edges=whisk_edges.tolist(), seconds=elapsed)
```

```python
'session_info': infos, 'failed_sessions': failures,
```

iii. The notes justify these mainly as validation and debugging aids rather than essential decoder inputs.
