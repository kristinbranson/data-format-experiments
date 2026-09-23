# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads Allen Visual Behavior ophys data through `VisualBehaviorOphysProjectCache.from_local_cache`, then restricts the experiment table to experiment IDs whose NWB files are present locally. It further keeps only `active` experiments and groups them into sessions for later processing. Data for each session are loaded plane-by-plane with `get_behavior_ophys_experiment`.

ii.
```python
def get_cache():
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)

def available_experiment_table(cache):
    et = cache.get_ophys_experiment_table()
    avail = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                   for f in glob.glob(NWB_GLOB))
    sub = et.loc[et.index.isin(avail)].copy()
    return sub

cache = get_cache()
et = available_experiment_table(cache)
active = et[~et['passive']].copy()
...
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in eids]
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as necessary because the local cache only contains a 284-experiment subset and because `from_local_cache(..., use_static_cache=False)` matches the on-disk layout. The trajectory also shows it deliberately excluded passive sessions because they lack behavioral outputs.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values taken from the filtered experiment table. Session records later store `mouse_id` strings, and `subject_idx` is built from first occurrence order.

ii.
```python
active = et[~et['passive']].copy()
...
subjects, regions = [], []
...
if r['mouse_id'] not in subjects:
    subjects.append(r['mouse_id'])
data['subject_idx'].append(subjects.index(r['mouse_id']))
```

iii. The notes describe `mouse_id` as the SDK subject identifier and treat `subjects`/`subject_idx` as a direct mapping from session to animal.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. If a session has multiple imaging planes, the AI groups all experiments with that same `ophys_session_id` and merges them into one decoder session.

ii.
```python
groups = active.groupby('ophys_session_id')
session_ids = sorted(groups.groups.keys())
...
jobs = [(int(s), list(groups.get_group(s).index.values), region_map, i < n_debug,
         args.neural_signal)
        for i, s in enumerate(session_ids)]
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI explicitly says “Session = one `ophys_session_id`” and argues that multi-plane sessions should be merged because the planes are simultaneous recordings that share one behavior session.

## 1-d. How are the data split into trials?

i. Trials come from `ds0.trials`, but the AI does not use the full SDK `start_time` to `stop_time` window. Instead, it defines each trial as a fixed 24-bin window around `change_time`, from `-2.25 s` to `+3.75 s`.

ii.
```python
OFF_START = -2.25
OFF_END = 3.75
BIN_SIZE = 0.25
...
trials = ds0.trials
sel = ((trials['go'] | trials['catch']) & (~trials['aborted'])
       & (~trials['auto_rewarded']))
trials = trials[sel]
trials = trials[~trials['change_time'].isna()]
...
change_times = trials['change_time'].values.astype(np.float64)
edges_abs = change_times[:, None] + BIN_EDGES[None, :]
centers_abs = change_times[:, None] + BIN_CENTERS[None, :]
```

iii. The AI’s notes justify this as a fixed-length decoder-friendly trial definition: always inside the true trial, identical for all sessions, and aligned to the behaviorally meaningful change event.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes aborted and auto-rewarded trials, requires non-null `change_time`, and then further drops trials with any missing binned running or pupil values. Sessions with fewer than 2 remaining trials, or with no eye tracking, are excluded.

ii.
```python
sel = ((trials['go'] | trials['catch']) & (~trials['aborted'])
       & (~trials['auto_rewarded']))
trials = trials[sel]
trials = trials[~trials['change_time'].isna()]
...
if len(eye) == 0:
    out['error'] = 'no eye tracking data'
    return out
...
keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
...
if keep.sum() < 2:
    out['error'] = 'fewer than 2 trials with complete behaviour'
    return out
```

iii. The notes justify the extra dropping as a way to avoid fabricating decoded outputs. They specifically say blink gaps can be interpolated, but trials or sessions without complete remaining behavioral coverage are removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the final code path, the default neural signal is `dff_traces['dff']`. The script can also optionally use `events['events']` or `events['filtered_events']`, but `--neural-signal` defaults to `dff`.

ii.
```python
ap.add_argument('--neural-signal', type=str, default='dff',
                choices=['events', 'dff', 'filtered_events'],
                help='neural signal to bin (default: L0 detected calcium events)')
...
if signal == 'dff':
    ev = np.vstack(ds.dff_traces['dff'].values).astype(np.float64)
elif signal == 'filtered_events':
    ev = np.vstack(ds.events['filtered_events'].values).astype(np.float64)
else:
    ev = np.vstack(ds.events['events'].values).astype(np.float64)
```

iii. The trajectory and notes show that the AI initially preferred detected calcium events to match the paper, but later switched the default to dF/F after empirical decoder comparisons.

## 2-b. How is the `neural` data processed?

i. Neural traces from all planes in a session are binned into 250 ms bins aligned to `change_time`, then concatenated across planes on the neuron axis. If `dff` is used, values are averaged within each bin; event-based signals are summed.

ii.
```python
binned = bin_sum(ev, ts, edges_abs)
if signal == 'dff':
    binned = binned / np.maximum(n_per_bin, 1)
neural_planes.append(binned)
...
neural = np.concatenate(neural_planes, axis=0)
```

iii. The notes justify 250 ms bins as matching the image flash duration and guaranteeing at least two ophys frames per bin even in mesoscope sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit extra neuron-level QC beyond what the Allen SDK already returns. The code assumes valid ROIs are already filtered by the SDK and asserts only that traces and timestamps match.

ii.
```python
assert ev.shape[1] == ts.shape[0], 'trace/timestamps mismatch'
...
region_per_neuron += [region_map[int(eid)]] * ev.shape[0]
cell_ids += list(ds.events.index.values)
```

iii. In the notes, the AI repeatedly states that Allen ROI filtering is already applied (`exclude_invalid_rois=True` by default), so it does not apply further neuron curation itself.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to each trial’s `change_time` using absolute bin edges on the ophys timestamp clock. Each trial spans `[-2.25, +3.75]` seconds relative to the change.

ii.
```python
change_times = trials['change_time'].values.astype(np.float64)
edges_abs = change_times[:, None] + BIN_EDGES[None, :]
...
binned = bin_sum(ev, ts, edges_abs)
```

iii. The AI’s notes explicitly identify `trials.change_time` as the alignment event and cite Allen reference code that aligns analyses to stimulus change.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 250 ms bins, with 24 bins per trial. The code rebins all streams to this temporal grid.

ii.
```python
BIN_SIZE = 0.25
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
...
time_bin_size=BIN_SIZE * 1000.0,
```

iii. The notes justify 250 ms as the image-on duration and as a common bin size that works for both 31 Hz single-plane and ~11 Hz mesoscope data.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name` in the `change_detection` block, after removing omitted flashes.

ii.
```python
sp = ds0.stimulus_presentations
sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
shown = sp[sp['image_name'] != 'omitted']
flash_start = shown['start_time'].values.astype(np.float64)
flash_image = shown['image_name'].values.astype(str)
```

iii. The AI’s notes justify this as more faithful to the flash-by-flash stimulus stream than relying only on trial table columns.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 250 ms bin center, the AI finds the most recent non-omitted flash onset and assigns that flash’s `image_name`. It then builds a global image-to-index mapping across the dataset.

ii.
```python
j = np.searchsorted(flash_start, centers_abs, side='right') - 1
image_name = flash_image[j]
...
image_values = sorted({str(im) for r in good for im in np.unique(r['image_name'])})
image_to_idx = {im: i for i, im in enumerate(image_values)}
...
img = np.vectorize(lambda x: image_to_idx[str(x)])(r['image_name']).astype(np.int64)
```

iii. The notes say this “holds” image identity through the gray interval and omissions so the label tracks the currently operative image presentation interval.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated at the same 250 ms bin centers used for neural binning, relative to `change_time`.

ii.
```python
centers_abs = change_times[:, None] + BIN_CENTERS[None, :]
...
j = np.searchsorted(flash_start, centers_abs, side='right') - 1
image_name = flash_image[j]
...
out_trials.append(np.stack([
    img[t], r['image_change'][t], run_q[t], pup_q[t],
    np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64))
```

iii. The notes describe all outputs as sharing the same 24-bin ophys-aligned timebase.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`, together with flash onset times, rather than directly from the trial table’s `go` or `change_time` columns.

ii.
```python
flash_ischange = shown['is_change'].values.astype(bool)
...
image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)
```

iii. The notes justify this as explicitly marking whether a given flash interval is a real change flash in the stimulus stream.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code finds the most recent flash for each bin center and marks the bin as `1` only if that flash was a real change flash and the bin center is still within 750 ms of that flash onset.

ii.
```python
since = centers_abs - flash_start[j]
image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)
```

iii. The notes say this matches the paper’s image-presentation interval abstraction: one 250 ms image plus 500 ms gray.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` for `no_change`, `1` for `change`.

ii.
```python
output_values = [
    list(image_values),
    ['no_change', 'change'],
    [f'speed_q{i + 1}' for i in range(N_QUANTILES)],
    [f'pupil_q{i + 1}' for i in range(N_QUANTILES)],
    list(OUTCOMES),
]
```

iii. No further justification is needed in the notes beyond the decoder specification and the explicit binary label list.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is computed on the same 250 ms `change_time`-aligned trial bins as the neural data.

ii.
```python
centers_abs = change_times[:, None] + BIN_CENTERS[None, :]
...
image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)
```

iii. The notes say all time-varying outputs are evaluated on the common binned ophys clock.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed`, specifically its `timestamps` and `speed` columns.

ii.
```python
run = ds0.running_speed
run_t = run['timestamps'].values.astype(np.float64)
run_v = run['speed'].values.astype(np.float64)
```

iii. The notes point out that this is already AllenSDK-processed running speed.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes the mean running speed within each 250 ms trial bin, then discretizes values into five equal-percentile bins using global edges computed across all kept data.

ii.
```python
running = bin_mean_nan(run_v, run_t, edges_abs)
...
all_run = np.concatenate([r['running'].ravel() for r in good])
run_edges = quantile_bins(all_run)
...
run_q = digitize_with(r['run_edges'], r['running'])
```

iii. The notes justify global percentile binning as giving one consistent labeling scheme across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five quintile bins using global percentile edges and `np.digitize`.

ii.
```python
def quantile_bins(values, n_bins=N_QUANTILES):
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    return np.percentile(values, qs)

def digitize_with(edges, values):
    return np.digitize(values, edges).astype(np.int64)
```

iii. The notes explicitly call for “global equal-percentile bins” to satisfy the decoder task wording.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is summarized into the same 250 ms bins used for neural data, with bin edges defined relative to `change_time`.

ii.
```python
edges_abs = change_times[:, None] + BIN_EDGES[None, :]
...
running = bin_mean_nan(run_v, run_t, edges_abs)
...
out_trials.append(np.stack([
    img[t], r['image_change'][t], run_q[t], pup_q[t],
    np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64))
```

iii. The notes describe all decoder outputs as being on the same common 24-bin trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using `pupil_area` and converting it to diameter.

ii.
```python
eye = ds0.eye_tracking
...
pupil_area = eye['pupil_area'].values.astype(np.float64)
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The notes explicitly say the decoded quantity should be diameter, not area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts area to diameter, linearly interpolates short blink-related gaps up to 0.5 s, computes mean diameter per 250 ms trial bin, then discretizes to five global percentile bins.

ii.
```python
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_MAX_INTERP_GAP)
pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)
...
all_pup = np.concatenate([r['pupil'].ravel() for r in good])
pup_edges = quantile_bins(all_pup)
...
pup_q = digitize_with(r['pup_edges'], r['pupil'])
```

iii. The notes justify short-gap interpolation because blinks are brief, but still drop trials whose bins remain incomplete.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global equal-percentile bins using `np.percentile` and `np.digitize`.

ii.
```python
pup_edges = quantile_bins(all_pup)
...
pup_q = digitize_with(r['pup_edges'], r['pupil'])
...
[f'pupil_q{i + 1}' for i in range(N_QUANTILES)]
```

iii. The notes say global quintiles were chosen so the classes are defined consistently across the whole dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is averaged into the same fixed 250 ms trial bins as neural activity, aligned to `change_time`.

ii.
```python
edges_abs = change_times[:, None] + BIN_EDGES[None, :]
...
pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)
```

iii. The notes describe pupil as one of the time-varying outputs on the shared binned ophys clock.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the SDK trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = np.full(n_trials, -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    outcome[trials[name].values.astype(bool)] = k
```

iii. The notes say these are the canonical go/catch outcome labels and are mutually exclusive on kept trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome label is encoded as an integer 0 to 3 and then broadcast as a constant row across all 24 bins of the trial.

ii.
```python
out_trials.append(np.stack([
    img[t], r['image_change'][t], run_q[t], pup_q[t],
    np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64))
```

iii. The notes justify broadcasting because the target format stores all outputs together, while trial outcome is static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data by excluding locally unavailable experiments up front, rejecting sessions with no eye-tracking table, interpolating only short pupil gaps, then dropping any trial with NaN running or pupil bins. Failed sessions are returned with an error string and excluded later.

ii.
```python
sub = et.loc[et.index.isin(avail)].copy()
...
if len(eye) == 0:
    out['error'] = 'no eye tracking data'
    return out
...
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_MAX_INTERP_GAP)
...
keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
...
good = [r for r in results if not r['error']]
bad = [r for r in results if r['error']]
```

iii. The notes say missing decoded outputs should not be fabricated. The trajectory shows the AI inspected eye-tracking gaps and chose drop/interpolate rules after that exploration.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading full session data through the Allen SDK and then binning session arrays across trials. The code parallelizes at the session level with `multiprocessing.Pool`.

ii.
```python
print(f'processing {len(jobs)} sessions with {args.nproc} workers ...', flush=True)
...
with Pool(min(args.nproc, len(jobs))) as pool:
    results = pool.map(process_session, jobs)
...
print(f'session loading/binning took {t_proc:.1f} s '
      f'({t_proc / max(len(jobs), 1):.2f} s/session)')
```

iii. The notes and trajectory both say SDK loading is expensive, and the notes identify vectorized binning plus multiprocessing as explicit speedups.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the heaviest per-trial binning work using cumulative sums and `np.searchsorted`. The remaining Python loop over trials during final assembly could still be vectorized further, but the AI explicitly avoided a naïve per-trial slicing implementation.

ii.
```python
def bin_sum(values, timestamps, edges_abs):
    csum = np.concatenate([np.zeros((values.shape[0], 1), dtype=np.float64),
                           np.cumsum(values, axis=1)], axis=1)
    idx = np.searchsorted(timestamps, edges_abs.ravel())
    idx = idx.reshape(edges_abs.shape)
    take = csum[:, idx]
    return (take[:, :, 1:] - take[:, :, :-1]).astype(np.float32)
...
for t in range(n_trials):
    neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :]))
```

iii. In Step 6, the notes state that a naïve implementation would loop over trials and that the final binning was intentionally vectorized.

## 9-c. What processing does the code repeat multiple times?

i. The code recomputes bin occupancies (`n_per_bin`) separately for each experiment/plane within a session and reopens the Allen cache inside every worker call to `process_session`. It also repeats image-to-index conversion trial by trial during final assembly.

ii.
```python
def process_session(args):
    ...
    cache = get_cache()
    ...
    for ds, eid in zip(datasets, eids):
        ...
        n_per_bin = bin_sum(np.ones((1, ts.shape[0])), ts, edges_abs)
        binned = bin_sum(ev, ts, edges_abs)
...
for t in range(n_trials):
    ...
    out_trials.append(np.stack([
        img[t], r['image_change'][t], run_q[t], pup_q[t],
        np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64))
```

iii. The notes do not call these repetitions out as problems, but they do mention cache opening and trial slicing as places where inefficiency matters.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects extra fields that are not saved in the final dataset or used by the downstream decoder, including `cell_ids`, `go`, `catch`, per-session timing diagnostics, and large debug payloads when `--show-processing` is enabled. It also supports alternate neural signals and debug plotting logic that are not needed for the final decoder file.

ii.
```python
out.update(dict(
    ...
    go=trials['go'].values.astype(bool)[keep],
    catch=trials['catch'].values.astype(bool)[keep],
    regions=np.array(region_per_neuron),
    cell_ids=np.array(cell_ids),
    ...
    t_load=t_load,
    t_total=time.time() - t_start,
))
if want_debug:
    out['debug'] = dict(
        ophys_timestamps=np.asarray(datasets[0].ophys_timestamps, dtype=np.float64),
        events0=np.vstack(datasets[0].events['events'].values).astype(np.float32),
        ...
        lick_times=np.asarray(ds0.licks['timestamps'].values, dtype=np.float64),
        reward_times=np.asarray(ds0.rewards['timestamps'].values, dtype=np.float64),
    )
```

iii. The notes describe these as sanity-check and visualization aids rather than part of the required converted dataset.
