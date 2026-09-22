# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `bwm_release.csv`, optionally restricts it with `DATALIMIT_SUBSET.csv`, groups rows into session jobs, and uses a local ONE instance. `SessionLoader` loads trials, wheel, and camera motion energy; `SpikeSortingLoader` loads each probe. It also rebuilds stale ONE cache tables by merging three releases and pointing dataset rows to the newest matching on-disk revision.

ii.
```python
bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
if subset_file.exists():
    bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]
...
sess_loader = SessionLoader(one=one, eid=eid)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The trajectory says the shipped cache tables were stale relative to revision folders and otherwise 459 of 461 sessions would have empty trial tables. It therefore made the converter self-contained by repairing cache metadata and using the newest on-disk revisions.

## 1-b. How are the data split into subjects?

i. Subject identifiers come directly from the `subject` column of `bwm_release.csv`. Each session job carries its subject; assembly builds a unique subject list in first-session order and a per-session index into it.

ii.
```python
jobs.append((eid, list(grp.pid), list(grp.probe_name),
             grp.subject.iloc[0], grp.lab.iloc[0]))
...
if res['subject'] not in subjects:
    subjects.append(res['subject'])
subject_idx.append(subjects.index(res['subject']))
```

iii. The agent treated the release table's subject as the authoritative mouse identifier; no identifier was inferred from paths.

## 1-c. How are the data split into sessions?

i. The release is grouped by `eid`; each unique `eid` becomes one job. Multiple probe rows for the same `eid` are merged into one session population.

ii.
```python
for eid, grp in bwm_df.groupby('eid', sort=False):
    jobs.append((eid, list(grp.pid), list(grp.probe_name), ...))
```

iii. The trajectory and final response identify `eid` as the session unit and explain that probes share behavior and are not independent sessions.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies a table with one row per trial. After masks are applied, trial-indexed arrays are emitted as lists of per-trial matrices.

ii.
```python
sess_loader.load_trials()
trials = sess_loader.trials
...
neural.append([spikes[k] for k in range(spikes.shape[0])])
```

iii. No custom boundary inference was needed because the trial table already defines trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed for reaction time outside 0.08–2 s, feedback more than 10 s after go cue, missing values in six named fields, or `choice == 0`. After wheel and whisker resampling, trials without full finite coverage of both behavioral streams are also removed; sessions need at least two surviving trials.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in NAN_EXCLUDE:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
...
ok = wheel_mask & whisker_mask
```

iii. The agent says this is the mask from `load_trials_and_mask(max_trial_len=10.0)`, verified identical to that reference function. It keeps the initial unbiased block because prior probability is a required three-class output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from each probe's spike timestamps and cluster assignments. The cluster table contributes quality labels and anatomical acronyms used for filtering and region metadata.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
merged_times.append(spikes['times'])
merged_ids.append(spikes['clusters'].astype(np.int64) + offset)
```

iii. The agent followed the repository's spike-loading route and merged probes belonging to the same behavioral session.

## 2-b. How is the `neural` data processed?

i. Probe cluster IDs are offset and pooled, spikes are stably sorted by time, nonfinite spike times are dropped, and retained spikes are counted into unit-by-time arrays. Unlike the human solution, counts are not divided by 0.02 to make Hz.

ii.
```python
order = np.argsort(spike_times, kind='stable')
...
counts = np.bincount(c * N_BINS + b, minlength=n_clusters * N_BINS)
out[k] = counts.reshape(n_clusters, N_BINS)
```

iii. The trajectory says the spike binner was checked against the repository's `bin_spiking_data` on a full session and was bit-identical. The final metadata explicitly calls the values spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units require `clusters.label >= 1`, a Beryl region other than `root` or `void`, at least five good units from that region in the session, and representation of the region in at least two converted sessions.

ii.
```python
good = (clusters['label'].to_numpy() >= QC_LABEL) & ~np.isin(beryl, list(NON_GREY))
regions, counts = np.unique(beryl[good], return_counts=True)
enough = set(regions[counts >= MIN_NEURONS_PER_REGION])
...
if n >= MIN_SESSIONS_PER_REGION
```

iii. The agent deliberately followed the data paper's well-isolated/grey-matter and region inclusion criteria rather than Zhang code that retains all sorted units. It also cited the otherwise roughly 110 GB result and decoder memory concerns. This is broader filtering than the human conversion, which keeps `root` and has no 5-unit/2-session region cuts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each retained `stimOn_times`, spikes in `[onset-0.5, onset+1.5)` are selected and their bin is computed relative to that interval.

ii.
```python
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
...
b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```

iii. The agent took the alignment event and window directly from the repository parameter block and relied on IBL's synchronized session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms: 100 nonoverlapping bins spanning two seconds. Spike events are binned once; there is no smoothing or later rebinning.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The agent states these values are verbatim from `0_data_caching.py` and the methods paper.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the `stimOn_times` alignment event plus the fixed trial window and bin size, rather than measured as a separate raw signal.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy()
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
```

iii. The event, window, and resolution were selected to match the repository's stimulus-aligned decoding configuration.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code calculates the center of each of the 100 bins, from -0.49 through 1.49 s, and repeats that vector for every trial.

ii.
```python
inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = bin_centres
```

iii. No raw-signal processing is involved; it is a deterministic coordinate for the chosen grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Each time value labels the center of the corresponding 20 ms spike-count bin, with zero defined by the same `stimOn_times` used for spike selection.

ii.
```python
b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
inputs[:, 0, :] = bin_centres
```

iii. The agent regarded the common event and bin grid as sufficient alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It derives from consecutive values of `trials.probabilityLeft`; a change starts a new block.

ii.
```python
p = np.asarray(probability_left, dtype=float)
new_block[1:] = p[1:] != p[:-1]
```

iii. The trial table has no separate block ID, so the agent reconstructs blocks from the prior held constant within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. It computes a zero-based distance from the most recent block start on the complete trial table, then filters it with the trials and broadcasts the retained scalar across time.

ii.
```python
block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
return np.arange(len(p)) - block_start
...
inputs[:, 1, :] = block_idx[:, None]
```

iii. Computing before filtering preserves the position the mouse actually experienced even when intervening trials are later excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It derives from `trials.choice`, where +1 is left and -1 is right; no-response zeros have already been excluded.

ii.
```python
choice = ((1 - trials['choice'].to_numpy()) / 2).astype(np.int8)
```

iii. The agent says it empirically verified across sessions that +1 corresponds to a left report.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The arithmetic mapping converts +1 to left class 0 and -1 to right class 1, then broadcasts the per-trial label over all 100 bins.

ii.
```python
outputs[:, 0, :] = choice[:, None]
```

iii. This implements the class coding required by the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It derives from the trial-table column `probabilityLeft`.

ii.
```python
trials['probabilityLeft'].to_numpy()
```

iii. This column directly encodes the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Each value is assigned to its nearest member of `[0.2, 0.5, 0.8]`, producing 0, 1, or 2, and the label is broadcast over time.

ii.
```python
prior = np.abs(trials['probabilityLeft'].to_numpy()[:, None]
               - PRIOR_VALUES[None, :]).argmin(axis=1).astype(np.int8)
outputs[:, 1, :] = prior[:, None]
```

iii. The mapping is specified by the task. The broader trial mask ensures the source is finite, although nearest-value mapping is more permissive than an exact dictionary mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()` loads wheel timestamps and derives velocity from wheel position; the converter uses the absolute velocity.

ii.
```python
sess_loader.load_wheel()
sess_loader.wheel['times'].to_numpy(),
np.abs(sess_loader.wheel['velocity'].to_numpy())
```

iii. The agent followed the reference definition of speed as absolute loader-provided wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The loader's velocity is made nonnegative, finite samples are sorted, and `np.interp` resamples it on 100 trial-relative points. The resulting session-wide values are discretized into tertiles.

ii.
```python
binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
...
outputs[:, 2, :] = tertile_labels(wheel_speed)
```

iii. The agent says behavioral sampling matches `get_behavior_per_interval`; full-trace interpolation avoids extrapolation at the last point.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 1/3 and 2/3 quantiles of all retained wheel samples in a session define low, medium, and high classes; equality with a threshold goes to the upper class.

ii.
```python
edges = np.quantile(values, np.arange(1, N_OUTPUT_BINS) / N_OUTPUT_BINS)
return np.searchsorted(edges, values, side='right').astype(np.int8)
```

iii. The agent preferred per-session thresholds because wheel use differs substantially between mice and shared global thresholds would not have comparable behavioral meaning.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It uses the same stimulus onset and 100-index trial window, but samples behavior at right-bin edges (`-0.48, ..., 1.5`) whereas the time input labels neural bins by their centers (`-0.49, ..., 1.49`).

ii.
```python
grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
```

iii. The trajectory says the right-edge grid matches `get_behavior_per_interval` exactly and reports nearly identical categorical labels to the reference behavior implementation.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It derives from the `whiskerMotionEnergy` column and timestamps of side-camera motion-energy data. The left camera is tried first and the right camera is a fallback.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
    me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy()
```

iii. The camera preference follows the repository/reference logic and supports sessions missing one view.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Nonfinite released samples are removed, the trace is sorted by time, interpolated on the trial grid, and discretized into session tertiles. No filtering or normalization is added.

ii.
```python
finite = np.isfinite(values)
times, values = times[finite], values[finite]
...
whisker, whisker_mask = bin_behavior(...)
```

iii. The released trace is used directly; the agent cites camera-dependent arbitrary units as a reason for session-specific discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Low, medium, and high are defined by the session's 1/3 and 2/3 quantiles over retained interpolated values.

ii.
```python
outputs[:, 3, :] = tertile_labels(whisker)
```

iii. Per-session tertiles give balanced classes and avoid comparing arbitrary camera scales across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It uses the same stimulus event and trial indices, but—as for wheel speed—is evaluated at the right edges of neural bins rather than their centers.

ii.
```python
grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. The agent chose this grid to reproduce `get_behavior_per_interval` and relied on the shared IBL clock for cross-stream synchronization.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The cache index is repaired; nonfinite spikes and behavioral samples are removed; missing spike sorting probes are skipped; left camera failure falls back to right; trials lacking full behavior are dropped; loader failures and sessions with fewer than two usable trials or no qualifying neurons are reported and skipped.

ii.
```python
except Exception as e:
    return {'eid': eid, 'skip': f'{type(e).__name__}: {e}', ...}
...
if ok.sum() < 2:
    return {'eid': eid, 'skip': ...}
```

iii. The final trajectory reports 19 skipped sessions and explicitly flags the stale-cache issue; the strategy is to retain usable data while recording exclusions in metadata.

## 10-a. What are the most time-consuming steps of the code?

i. Cache-table filesystem indexing and, especially, loading and sorting large per-probe spike arrays dominate. Full conversion is parallelized by session across 16 spawned workers.

ii.
```python
for root, _dirs, files in os.walk(session_dir):
...
spikes, clusters, channels = ssl.load_spike_sorting()
...
with ctx.Pool(processes=args.n_workers) as pool:
```

iii. The trajectory spent substantial effort repairing/loading the cache and describes spike files and the unfiltered data volume as large; multiprocessing was chosen to make all-session conversion practical.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `bin_spikes`, membership list comprehensions for region filtering/indexing, job/result assembly loops, and cache-table per-session/per-dataset scans could be vectorized or batched. The spike loop is the clearest numerical candidate, though slices have variable lengths.

ii.
```python
for k in range(len(align_times)):
    t = spike_times[i0[k]:i1[k]]
...
good &= np.array([r in enough for r in beryl])
```

iii. The agent called its binner “vectorised” because each trial uses one `bincount`, and prioritized exact agreement with the reference. It did not discuss these remaining Python loops explicitly.

## 10-c. What processing does the code repeat multiple times?

i. Every worker independently constructs a ONE client and calls `build_patched_tables`; each camera attempt invokes motion-energy loading; region membership is traversed repeatedly during within-session QC, cross-session counting, filtering, and index conversion. Trial arrays are also successively masked and copied.

ii.
```python
one = get_one()
...
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
```

iii. The trajectory does not give a specific justification beyond worker isolation and fallback behavior; the repeated work is mostly structural overhead rather than the dominant spike I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges cluster channels and extensive trial columns primarily to support QC, then discards them; computes and carries `align_times` after final masking without using it again; retains reporting fields before converting them to metadata; scans and rewrites broad cache tables beyond the final subset. Neural units from regions failing the dataset-wide two-session rule are fully binned first and discarded only during assembly.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
align_times = align_times[ok]
...
spikes = res['neural'][:, keep, :]
```

iii. Most of this enables robust QC or metadata repair, but the trajectory does not claim downstream decoder use for it. In particular, late global region filtering necessarily wastes earlier binning work.
