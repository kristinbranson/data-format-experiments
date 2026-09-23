# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the 459 released session IDs and subject/lab metadata from the methods repository's `bwm_release.csv`. For each session it creates a cached remote-mode `ONE` client, uses `SessionLoader` for trials, wheel, and camera motion energy, and `SpikeSortingLoader` for every probe returned by `one.eid2pid`. The underlying experimental arrays are therefore loaded through ONE/brainbox, not directly from `/app/data`.

ii.
```python
bwm = pd.read_csv(BWM_RELEASE, index_col=0)
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
sess_loader = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
pids, pnames = one.eid2pid(eid)
```

iii. The agent says this follows the reference caching script and uses cached Alyx responses to resolve revised staged datasets offline. It chose the 459-session release CSV as the authoritative released-session list.

## 1-b. How are the data split into subjects?

i. Subject labels come directly from `bwm_release.csv`. Converted sessions retain their subject string; final `subjects` is the sorted unique set and `subject_idx` maps each sorted session to it.

ii.
```python
jobs = [(r.eid, r.subject, r.lab, args.show_processing) for r in sess_df.itertuples()]
subjects = sorted({r['subject'] for r in good})
sub_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
```

iii. The release table already supplies unique subject identifiers, so no path parsing is needed.

## 1-c. How are the data split into sessions?

i. Each distinct release-table `eid` is one session and is independently processed. Results are sorted by EID before assembly.

ii.
```python
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
good.sort(key=lambda r: r['eid'])
```

iii. The agent treats EID as ONE's native session identifier and parallelizes independent sessions.

## 1-d. How are the data split into trials?

i. `load_trials_and_mask` returns a trials table with one row per trial. Each row's `stimOn_times` defines a window from -0.5 to +1.5 s; retained rows become separate list entries.

ii.
```python
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
t0s_all = align + TIME_WINDOW[0]
for i in range(n_keep):
    neural_trials.append(np.ascontiguousarray(counts[i]))
```

iii. The agent states this reproduces the reference trial geometry: stimulus-onset alignment, two seconds, 100 bins.

## 1-e. How are trials filtered based on quality controls?

i. It applies the reference `load_trials_and_mask`: RT 0.08–2 s, no choice 0, no NaNs in six task fields, and feedback-minus-go-cue no more than 10 s. It additionally requires finite alignment and complete wheel/camera coverage, and drops all-zero-spike trials. Sessions with fewer than two retained trials are dropped.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
keep = mask & finite_align & ws_valid & me_valid
nonempty = counts.sum(axis=(1, 2)) > 0
```

iii. The reference method code motivated the task mask and behavior-coverage conjunction. The agent argues all-zero trials represent ephys gaps/endings and cannot help decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices derive from every probe's `spikes.times` and `spikes.clusters`; merged cluster metadata supplies QC labels and anatomical acronyms.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
st = spikes['times'][smask]
sc = spikes['clusters'][smask]
```

iii. These are the raw variables used by the paper/reference spike-binning functions.

## 2-b. How is the `neural` data processed?

i. Probes are merged, cluster IDs are made contiguous, spikes are sorted, and spike counts are computed in non-overlapping 20 ms bins. The saved values are raw float32 counts, not firing rates or z-scores.

ii.
```python
bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
idx = spike_clusters[a:b] * nbins + bins
counts = np.bincount(idx, minlength=n_neurons * nbins)
out[k] = counts.reshape(n_neurons, nbins).astype(np.float32)
```

iii. The agent chose counts because the methods repository caches counts and standardizes later inside its decoder; it verified its counts were bit-identical to that spike-binning implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters must have `label >= 1`; Beryl `root` and `void` units are removed. Sessions with fewer than five remaining units are discarded.

ii.
```python
iok = clusters_labeled['label'] >= qc
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
keep = ~np.isin(beryl, NON_GREY)
if n_units < MIN_NEURONS:
    return {'eid': eid, 'skip': ...}
```

iii. The agent ties `label==1` to the data paper's well-isolated neurons and excludes `root`/`void` as non-grey matter. It imposed five neurons based on a paper session criterion and dataset manageability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial begins at `stimOn_times - 0.5`; spikes are binned until `stimOn_times + 1.5`, so stimulus onset is time zero.

ii.
```python
t0s_all = align + TIME_WINDOW[0]
t1s = t0s + nbins * binsize
i0 = np.searchsorted(spike_times, t0s, side='left')
```

iii. This exactly follows the mandated stimulus-onset alignment and reference caching geometry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms: 100 bins across two seconds. Raw spike times are binned once; no subsequent temporal resampling or smoothing is applied to neural data.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The agent cites the reference configuration and methods paper's 20 ms dynamic-behavior bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived conceptually from each trial's `stimOn_times` and the fixed window/bin grid, rather than from a separate recorded stream.

ii.
```python
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
```

iii. The required variable is defined by the alignment geometry.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code takes the centers of the 100 neural bins, from -0.49 through 1.49 s, and repeats this row for every trial.

ii.
```python
inputs.append(np.stack([bin_centers,
                        np.full(NBINS, tib[i], dtype=np.float32)]))
```

iii. Bin centers give each count bin a representative signed time.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `k` is the center of neural spike-count bin `k`; both use the same onset, window, bin size, and 100 columns.

ii.
```python
bin_centers = TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE
```

iii. The agent explicitly documents the time input as the neural bin center.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It derives from the full unfiltered trials table's `probabilityLeft`, whose consecutive constant runs define blocks.

ii.
```python
tib_all, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
tib = tib_all[keep].astype(np.float32)
```

iii. No raw block ID exists, so the prior's change points recover it.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A change starts a new block; trials within each block receive a zero-based sequential index. It is computed before filtering and broadcast across time.

ii.
```python
changed[1:] = ~(pl[1:] == pl[:-1])
block_id = np.cumsum(changed) - 1
idx[m] = np.arange(m.sum())
```

iii. Computing it before filtering preserves the animal's true position even when intervening trials are excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`.

ii.
```python
choice = trials['choice'].to_numpy()[keep]
```

iii. The agent empirically verified the IBL sign convention against stimulus side and correctness.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `+1` (left) maps to 0 and `-1` (right) maps to 1; retained no-choice trials are already excluded. The class is repeated for all 100 bins.

ii.
```python
choice_cls = (choice < 0).astype(np.int32)
np.full(NBINS, choice_cls[i], dtype=np.int32)
```

iii. This implements the task's requested left/right coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[keep]
```

iii. The trials table contains the task block prior directly.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 map to classes 0, 1, and 2 and are broadcast over time; an unexpected value skips the session.

ii.
```python
prior_cls = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
```

iii. This is the exact mapping requested by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from the wheel timestamps and absolute value of `SessionLoader`'s wheel velocity, itself computed from raw wheel position/timestamps.

ii.
```python
sess_loader.load_wheel()
wheel_times = sess_loader.wheel['times'].to_numpy()
wheel_speed = np.abs(sess_loader.wheel['velocity'].to_numpy())
```

iii. This matches the reference behavior loader's definition of wheel speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` supplies interpolated/filtered velocity; its magnitude is linearly interpolated to each trial's 100 bin right edges, then discretized using session-level tertiles over retained trial-bin values.

ii.
```python
grid = t0s[:, None] + np.arange(1, nbins + 1)[None, :] * binsize
interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
edges = np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])
```

iii. Right edges reproduce `get_behavior_per_interval`; per-session tertiles yield balanced classes.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. `np.digitize` applies the session's 33.33rd and 66.67th percentiles, producing 0/1/2 for low/medium/high.

ii.
```python
classes = np.digitize(values, edges, right=False).astype(np.int32)
```

iii. Per-session thresholds avoid imbalance and were applied only after trial retention.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It uses the same stimulus onset, two-second interval, 20 ms spacing, and 100 columns, but samples at each neural bin's right edge rather than its center.

ii.
```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
```

iii. The agent deliberately followed the methods repository's behavior interpolation grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and `times` from left or right camera motion-energy tables. Both are attempted, and the camera covering the most curated trials is selected (left on ties).

ii.
```python
sess_loader.load_motion_energy(views=[view])
cameras[cam] = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
if best is None or valid_c.sum() > best[2].sum():
    best = (cam, me_c, valid_c)
```

iii. The original left-then-right fallback lost usable trials in partial recordings, so the agent chose maximal coverage while preserving left preference on ties.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is otherwise unfiltered and unnormalized, interpolated to trial bin right edges, and discretized by per-session tertiles over retained values.

ii.
```python
me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
me_cls, me_edges = discretize_tertiles(me_k)
```

iii. Per-session thresholds accommodate arbitrary camera-dependent motion-energy scales.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The session's 33.33rd and 66.67th percentiles define low (0), medium (1), and high (2).

ii.
```python
edges = np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])
classes = np.digitize(values, edges, right=False).astype(np.int32)
```

iii. This produces approximately balanced classes despite between-camera scale variation.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It shares stimulus onset, interval, spacing, and column count with neural data, but is evaluated at 20 ms bin right edges rather than neural bin centers.

ii.
```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
```

iii. The agent copied the methods repository's continuous-behavior grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing probes are skipped; sessions without motion energy or enough units/trials are skipped; nonfinite/incompletely covered behavior trials and all-zero neural trials are removed. Exceptions return a skip record rather than aborting the run. Finite behavior samples are used for interpolation, but any NaN within a trial invalidates it.

ii.
```python
if not clusters or not spikes:
    return None, None
if not cameras:
    return {'eid': eid, 'skip': 'no whisker motion energy'}
except Exception as e:
    return {'eid': eid, 'skip': f'error: {e}', ...}
```

iii. The agent preferred dropping undefined observations/sessions and documented each skip, while salvaging sessions with another valid probe or camera.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large spike-sorting arrays is the dominant per-session operation; full-data pickle writing and decoder training are also costly overall. The code records trial, behavior, spike-load, behavior-bin, spike-bin, and total timings.

ii.
```python
t = time.time()
neural = load_session_neural(one, eid)
timings['spikes'] = time.time() - t
```

iii. The notes identify disk I/O for spike sorting as dominant and therefore parallelize at session level with 24 workers.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. `bin_spikes` still loops over trials around vectorized `searchsorted`/`bincount`; `bin_behavior` loops over trials only for validity checks; `trial_number_in_block` loops over blocks; final list assembly loops over trials. These could be further vectorized, though ragged trial slices make the present forms clear.

ii.
```python
for k in range(ntrials):
    a, b = i0[k], i1[k]
    ...
for b in np.unique(block_id):
    idx[m] = np.arange(m.sum())
```

iii. The agent describes spike binning as “vectorised” because expensive work within each trial uses NumPy, and reports it as bit-identical to the reference.

## 10-c. What processing does the code repeat multiple times?

i. Each worker independently constructs a ONE client; every session loads/bins wheel and each available camera over all trials before applying the combined mask. Both camera traces may be binned only to discard the lower-coverage one. Beryl mapping and several per-trial constant arrays are also repeatedly constructed.

ii.
```python
for cam in ('leftCamera', 'rightCamera'):
    me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
```

iii. Repetition supports independent multiprocessing and enables the deliberate best-camera comparison; the agent does not flag other repeated work as material.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and retains extensive diagnostics/timings and session metadata not used by the decoder. It bins behavior for trials later rejected and bins both cameras although one is discarded. `bin_spikes` also computes `t1s` without using it. Optional plotting performs additional diagnostic work only when requested.

ii.
```python
t1s = t0s + nbins * binsize
timings['bin_spikes'] = time.time() - t
```

iii. The diagnostic information was intentionally kept for sanity checks and auditability; the notes regard plotting as optional and the extra camera processing as necessary for robust selection.
