# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the BWM release CSV, optionally restricts it with `DATALIMIT_SUBSET.csv`, groups probe rows into sessions, and uses an offline ONE client. `SessionLoader` loads trials, wheel, and camera motion energy; `SpikeSortingLoader` loads each probe's spikes and clusters. Sessions may be processed in parallel.

ii.
```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
if os.path.exists(DATALIMIT_SUBSET):
    subset = pd.read_csv(DATALIMIT_SUBSET)
    bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]
...
sess_loader = SessionLoader(one=one, eid=eid)
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. The trajectory says the release freeze supplies the intended session/probe IDs, while ONE resolves staged cached data without forcing network authentication. It also says probes in one session should be merged as in the methods code.

## 1-b. How are the data split into subjects?

i. Subject IDs come from the `subject` column of the release CSV. At assembly, unique successful-session subjects are sorted, and each session gets an integer `subject_idx`.

ii.
```python
sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0], ...))
subjects = sorted({r['subject'] for r in results})
subject_idx = np.array([subject_lookup[r['subject']] for r in results])
```

iii. The trajectory treats the release metadata as the authoritative subject identity; no filename parsing is needed.

## 1-c. How are the data split into sessions?

i. Rows of the BWM release CSV are grouped by `eid`; all probe IDs and names sharing an `eid` form one session. Results are sorted by `eid` before assembly.

ii.
```python
for eid, rows in bwm_df.groupby('eid', sort=False):
    sessions.append((eid, ..., list(rows.pid), list(rows.probe_name)))
...
results.sort(key=lambda r: r['eid'])
```

iii. The trajectory identifies `eid` as the unique session identifier and merges probes because they share behavior and are not independent sessions.

## 1-d. How are the data split into trials?

i. The trials table supplies one row per trial. Stimulus-centered interval starts/ends are computed per row, and per-trial neural, input, and output arrays are appended in a loop after masking.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
for i in range(n_trials):
    neural.append(binned[i].astype(np.float32))
```

iii. The trajectory regards the trials table as already defining trial boundaries and uses the requested stimulus-onset alignment window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained when reaction time is 0.08–2 s, trial length is at most 10 s, none of six named fields is null, and choice is nonzero. The agent additionally requires wheel and whisker traces to cover the full window without nonfinite interpolated values; sessions with fewer than two retained trials fail.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in NAN_EXCLUDE: query += f' | {event}.isnull()'
query += ' | (choice == 0)'
keep = trials_mask & wheel_ok & whisker_ok
```

iii. The agent says these are the methods pipeline's `load_trials_and_mask` criteria. It deliberately drops missing behavioral windows because the target decoder format should not contain NaNs, noting this differs from the reference pipeline's `allow_nans=True` behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from each probe's `spikes.times` and `spikes.clusters`; merged cluster/channel data supplies cluster indexing and anatomical acronyms.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
```

iii. The trajectory states spike times and Kilosort cluster assignments are the methods pipeline's neural source, with cluster metadata needed for region labels.

## 2-b. How is the `neural` data processed?

i. Probes are merged, cluster IDs reindexed, spikes sorted by time, and spikes counted in 20 ms bins for each two-second trial. The saved values are raw spike counts cast to `float32`; they are not divided by bin width or smoothed.

ii.
```python
counts = np.bincount(rows * N_BINS + bin_idx[keep],
                     minlength=n_units * N_BINS)
binned[trial] = counts.reshape(n_units, N_BINS)
...
neural.append(binned[i].astype(np.float32))
```

iii. The agent explicitly chose spike counts because it interpreted the methods pipeline as binning counts without smoothing and reported that its binning matched that implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No cluster-quality or in-brain filter is applied. All cluster IDs that fired at least once anywhere in the session are retained, including possible `void` regions.

ii.
```python
unit_ids = np.unique(spike_clusters)
...
acronyms = clusters['acronym'].to_numpy()[unit_ids]
beryl = BrainRegions().acronym2acronym(acronyms, mapping='Beryl')
```

iii. The trajectory says `prepare_data` calls `load_spiking_data(qc=None)` and the methods paper uses all Kilosort neurons. It acknowledges the data paper instead uses stringent well-isolated-unit QC, but prioritizes the decoding pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial covers `[stimOn_times - 0.5, stimOn_times + 1.5)`. Search-sorted spikes in that absolute interval are assigned bins relative to the interval start.

ii.
```python
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE)
```

iii. The agent chose visual stimulus onset and the −0.5-to-1.5 s window because these are the caching code's parameters and all streams share the synchronized session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms, with 100 nonoverlapping bins in a 2 s window. Spikes are binned once; no later temporal rebinning or interpolation of neural values occurs.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The trajectory cites the reference caching parameters and selects 20 ms because the time-varying behavioral outputs require that resolution.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the `stimOn_times` alignment event plus the fixed window and bin size; its values are relative bin centers and therefore identical across trials.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
```

iii. The agent says stimulus onset is the mandated alignment event and bin centers are the natural time coordinate for neural bins.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It constructs 100 evenly spaced centers, −0.49 through 1.49 s, and copies this row into every retained trial's input matrix.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
inp[0] = bin_centres
```

iii. The trajectory describes this as the centers of the configured 20 ms stimulus-relative bins.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Input element `t` is the center of neural bin `t`; both use the same `TIME_WINDOW`, `BINSIZE`, and `N_BINS`.

ii.
```python
bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE)
inp[0] = bin_centres
```

iii. The agent intended a one-to-one temporal coordinate for the neural bins.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived solely from consecutive values of the trials-table `probabilityLeft` column.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()
new_block[1:] = pleft[1:] != pleft[:-1]
```

iii. The trajectory notes there is no explicit block ID, while `probabilityLeft` is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Changes in `probabilityLeft` mark block starts; subtracting the corresponding block-start index produces a zero-based trial number. It is computed before filtering and broadcast across all 100 bins.

ii.
```python
block_id = np.cumsum(new_block) - 1
idx_in_block = np.arange(len(pleft)) - block_starts[block_id]
...
inp[1] = in_block[i]
```

iii. The agent computes it on the full table so excluded trials still advance the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the trials-table `choice` column.

ii.
```python
choice = np.where(trials['choice'].to_numpy()[keep] > 0, 0, 1)
```

iii. The trajectory identifies IBL choice `+1` as left and `-1` as right; no-choice trials were already filtered.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Retained `+1` values are mapped to category 0 (left), and `-1` values to category 1 (right), then broadcast across the trial's 100 time bins.

ii.
```python
choice = np.where(... > 0, 0, 1).astype(np.int64)
out[0] = choice[i]
```

iii. This implements the category mapping required by the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from the trials-table `probabilityLeft` column.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[keep]
```

iii. The agent identifies this as the current block's left-stimulus prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 map to classes 0, 1, and 2; unexpected values raise an error. The class is broadcast across time.

ii.
```python
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
out[1] = prior[i]
```

iii. The mapping is explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses the `times` and derived `velocity` columns produced by `SessionLoader.load_wheel()` from raw wheel timestamps and positions; speed is the velocity magnitude.

ii.
```python
sess_loader.load_wheel()
return (sess_loader.wheel['times'].to_numpy(),
        np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The trajectory says this matches `load_target_behavior`: absolute Gaussian-smoothed wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The loader-derived velocity is made absolute, linearly interpolated per trial at 100 query times, checked for coverage/nonfinite values, and discretized using session-wide tertiles.

ii.
```python
y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
wheel_class, wheel_edges = discretize(wheel[keep])
```

iii. The agent aimed to reproduce `get_behavior_per_interval` and chose session-wise tertiles for balanced labels despite between-session scale differences.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 1/3 and 2/3 quantiles over all retained wheel samples in a session form two thresholds; `np.digitize` creates low/medium/high classes. Equal thresholds are not actually made strictly increasing despite the comment.

ii.
```python
quantiles = np.quantile(values, np.arange(1, n_classes) / n_classes)
edges = np.maximum.accumulate(quantiles)
return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. The trajectory justifies within-session tertiles as scale-robust and approximately class-balancing; tied zero wheel speeds remain together.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is stimulus-aligned on the shared clock, but sampled at each bin's right edge (−0.48 through 1.50 s), whereas the neural input time coordinate describes bin centers (−0.49 through 1.49 s).

ii.
```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
y = interp1d(t, v, ...)(x)
```

iii. The agent says right-edge evaluation reproduces `get_behavior_per_interval` exactly and considered the resulting index-wise pairing aligned.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and `times` from left-camera motion energy, falling back to right-camera data if loading left fails.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
    me = sess_loader.motion_energy[key]
    return me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy()
```

iii. The trajectory says the reference behavior loader prefers left and falls back to right because some sessions lack left-camera data.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is not filtered or normalized. It is linearly interpolated per trial, subjected to coverage/nonfinite checks, and discretized using session-wide tertiles.

ii.
```python
whisker, whisker_ok = bin_behavior(whisker_times, whisker_vals, ...)
whisker_class, whisker_edges = discretize(whisker[keep])
```

iii. The agent says no extra preprocessing is warranted and session-specific thresholds address camera/illumination scale differences.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The 1/3 and 2/3 quantiles over all retained samples in the session define low, medium, and high categories via `np.digitize`.

ii.
```python
quantiles = np.quantile(values, np.arange(1, n_classes) / n_classes)
return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. The trajectory uses the same equal-frequency, within-session rationale as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus onset and session clock but sampled at neural-bin right edges rather than centers.

ii.
```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. The agent considered this a faithful reproduction of the methods behavior-binning helper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/null trial events are filtered; incomplete or nonfinite behavioral windows are filtered; missing left camera falls back to right. Any session exception is recorded and skipped, and sessions with fewer than two usable trials are rejected. NaN alignment times are temporarily replaced with zero only to keep intermediate searches defined, then excluded by the trial mask.

ii.
```python
align_times_filled = np.where(np.isnan(align_times), 0.0, align_times)
keep = trials_mask & wheel_ok & whisker_ok
except Exception as exc:
    return {'eid': args[0], 'error': ...}
```

iii. The trajectory emphasizes that the output format cannot carry missing behavioral targets, so dropping affected trials/sessions is preferable to retaining NaNs; it reports manually checking excluded sessions.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large per-probe spike sorting arrays, binning spikes/behavior for every trial, and serializing the very large converted pickle dominate. The code parallelizes independent sessions.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
with ctx.Pool(processes=args.n_workers, maxtasksperchild=4) as pool:
    for ... in pool.imap_unordered(...):
```

iii. The trajectory's repeated full-session runs and final 107 GB artifact support I/O and conversion as the expensive work; its final summary highlights multiprocessing and cached local reads.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `bin_spikes`, `bin_behavior`, and final list assembly could be further vectorized or batched. Probe and session loops are structurally appropriate; sessions are already parallelized.

ii.
```python
for trial in range(n_trials):
    ... binned[trial] = counts.reshape(n_units, N_BINS)
for trial in range(n_trials):
    ... y = interp1d(...)(x)
for i in range(n_trials):
    neural.append(...)
```

iii. The trajectory claims spike and behavior binning were made substantially vectorized and verified against reference helpers, but retains trial loops because each trial slices a different time interval.

## 10-c. What processing does the code repeat multiple times?

i. It creates a new ONE client per session, invokes interpolation separately for every trial and each of two behaviors, repeatedly casts/copies per-trial arrays, and recomputes identical behavioral query offsets per trial.

ii.
```python
one = get_one()
...
for trial in range(n_trials):
    y = interp1d(t, v, ...)(x)
...
for i in range(n_trials):
    neural.append(binned[i].astype(np.float32))
```

iii. The trajectory does not explicitly justify these repetitions; it mainly justifies per-session worker isolation and reference-equivalent interval processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Behavior is interpolated for every original trial before the trial-QC mask is applied, so traces for already-invalid trials are discarded. Cluster metadata is merged broadly although downstream use is principally acronyms. Tracebacks and extensive metadata are retained for bookkeeping rather than decoder training.

ii.
```python
wheel, wheel_ok = bin_behavior(..., interval_begs, interval_ends)
whisker, whisker_ok = bin_behavior(..., interval_begs, interval_ends)
keep = trials_mask & wheel_ok & whisker_ok
```

iii. The trajectory does not identify this waste; its focus is behavioral coverage checking and reproducibility. The ordering makes coverage masks easy to combine but performs avoidable work on trials already rejected by `trials_mask`.
