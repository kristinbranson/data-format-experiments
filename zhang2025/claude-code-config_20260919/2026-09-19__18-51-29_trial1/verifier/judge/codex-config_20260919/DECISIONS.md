# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the 459-session BWM freeze CSV, makes one job per unique `eid`, and uses local-cache-backed IBL loaders. Each worker loads every listed probe with `SpikeSortingLoader`, merges probes, and loads trials, wheel, and left-camera motion energy (right camera fallback) with `SessionLoader`. Failed sessions are recorded and omitted.

ii.
```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))
sp, cl, ch = ssl.load_spike_sorting()
trials, ref_mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
```

iii. The agent says this mirrors the reference loading path and imports `merge_probes` and `load_trials_and_mask` directly. It chose the published freeze to define the population and parallelized independent sessions.

## 1-b. How are the data split into subjects?

i. Subject labels come from `bwm_release.csv`. Converted sessions are sorted by `(subject, eid)`; unique subject names are sorted and each session receives an integer `subject_idx`.

ii.
```python
results.sort(key=lambda r: (r['subject'], r['eid']))
subjects = sorted({r['subject'] for r in results})
subject_idx = np.array([subj_lookup[r['subject']] for r in results], dtype=np.int64)
```

iii. The freeze provides authoritative subject IDs, avoiding filename/path inference.

## 1-c. How are the data split into sessions?

i. Each unique freeze `eid` is treated as one session; all its probes are merged into one neural population and one result dictionary.

ii.
```python
for i, eid in enumerate(eids):
    sub = bwm[bwm.eid == eid]
    jobs.append((eid, sub.pid.tolist(), sub.probe_name.tolist(), ...))
```

iii. EID is the IBL session identifier, and session-level parallelism was chosen because sessions are independent.

## 1-d. How are the data split into trials?

i. The loaded trials table supplies one row per trial. For each retained row, the agent makes a 2-second interval beginning at `stimOn_times - 0.5`, then emits one neural/input/output array.

ii.
```python
align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
interval_begs = align_times + WIN[0]
'neural': [np.ascontiguousarray(binned[k]) for k in range(n_kept)]
```

iii. The reference task already defines trials and requests stimulus-onset-aligned 2-second examples.

## 1-e. How are trials filtered based on quality controls?

i. It starts with imported `load_trials_and_mask(max_trial_len=10)`: RT 0.08–2 s, trial duration at most 10 s, required values finite, and nonzero choice. It additionally requires a recognized prior/choice, finite stimulus onset, complete intersection-of-probes ephys coverage, and acceptable wheel/camera coverage. Sessions need at least two retained trials; degenerate behavior tertiles reject a whole session.

ii.
```python
mask = ref_mask.to_numpy().astype(bool)
mask &= prior_code >= 0
mask &= np.isin(choice_raw, (-1.0, 1.0))
mask &= (interval_begs >= rec_span[0]) & (interval_begs + BINSIZE * NBINS <= rec_span[1])
mask &= wheel_ok & me_ok
```

iii. The imported mask was justified as literal reference curation. Extra coverage rules prevent missing recordings from becoming zeros/NaNs; tied tertiles are treated as broken sensors.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from each probe's `spikes.times` and `spikes.clusters`; cluster `label`, acronym, UUID, and channel/histology metadata determine unit inclusion and regions.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```

iii. This follows the IBL spike-sorting loaders and reference probe-merging path.

## 2-b. How is the `neural` data processed?

i. Probes are merged, retained clusters are remapped to consecutive unit IDs, and spikes are counted in 100 nonoverlapping 20-ms bins for each trial. The stored values are raw float32 spike counts, not rates or smoothed activity.

ii.
```python
b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
counts = np.bincount(uu * NBINS + b, minlength=n_units * NBINS)
out[k] = counts.reshape(n_units, NBINS)
```

iii. The agent states the method repository caches raw counts and normalization happens at decoding time; no smoothing is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units require `clusters.label >= 1` and a Beryl acronym other than `root` or `void`. Sessions require at least five retained neurons. All probes in a session are pooled.

ii.
```python
keep = (label >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
if len(keep_idx) < MIN_NEURONS:
    raise RuntimeError(...)
```

iii. The agent prioritizes the BWM paper's “well-isolated” units and grey matter over the methods code's `qc=None`, citing the reported 108 good units/probe and avoiding near-empty populations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every interval begins 0.5 s before `stimOn_times` and ends 1.5 s after it; spike times are assigned relative to that absolute interval start.

ii.
```python
WIN = (-0.5, 1.5)
interval_begs = align_times + WIN[0]
i0 = np.searchsorted(spike_times, safe, side='left')
```

iii. These are exactly the reference alignment event and window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms, with 100 bins over two seconds. Raw spikes are binned once; no later temporal rebinning or smoothing occurs.

ii.
```python
BINSIZE = 0.02
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))
```

iii. The agent cites the reference parameters and methods paper.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the fixed window/bin configuration anchored to trial `stimOn_times`, not from a separate sampled raw stream.

ii.
```python
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. The agent interprets the input as the time coordinate of each neural bin.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It constructs the 100 left bin edges `-0.50, -0.48, ..., 1.48` seconds and repeats them for every trial.

ii.
```python
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. The notes explicitly define time as the bin's left edge.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `i` labels the left edge of neural spike-count bin `i`, so both use the same onset, window, and index.

ii.
```python
b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. The agent regards bin-edge labeling as an exact index-wise alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived solely from transitions in the full trials table's `probabilityLeft` column.

ii.
```python
block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. No explicit block ID exists, while `probabilityLeft` is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the unfiltered trials in order, resets a zero-based counter when the prior changes, increments otherwise, and broadcasts the retained trial's number across 100 bins.

ii.
```python
if i > 0 and not (p[i] == p[i - 1] or (np.isnan(p[i]) and np.isnan(p[i - 1]))):
    count = 0
out[i] = count
count += 1
inputs[:, 1, :] = block_idx[mask][:, None]
```

iii. Computing before exclusions preserves the animal's true position rather than renumbering retained trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`, after no-response/invalid trials are removed.

ii.
```python
choice_raw = trials['choice'].to_numpy(dtype=float)
mask &= np.isin(choice_raw, (-1.0, 1.0))
```

iii. IBL stores left as +1, right as -1, and no response as 0.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `(1-choice)/2` maps +1 to category 0 (left) and -1 to category 1 (right), then the value is broadcast over time.

ii.
```python
choice_out = ((1 - choice_raw[mask]) / 2).astype(np.int64)
outputs[:, 0, :] = choice_out[:, None]
```

iii. This supplies the exact binary coding requested by the decoder task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from the trials table's `probabilityLeft`.

ii.
```python
prior_code = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. This is the experiment's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are mapped 0.2→0, 0.5→1, 0.8→2; other values cause trial removal. The category is repeated across all bins.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
outputs[:, 1, :] = prior_out[:, None]
```

iii. This is the mapping explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from wheel timestamps and position loaded by `SessionLoader`; the loader produces `wheel.velocity`, whose absolute value is used.

ii.
```python
sess_loader.load_wheel()
return (sess_loader.wheel['times'].to_numpy(dtype=float),
        np.abs(sess_loader.wheel['velocity'].to_numpy(dtype=float)))
```

iii. This matches the reference's `wheel-speed` target definition.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` interpolates/filter-differentiates wheel position into velocity; the code takes magnitude, linearly interpolates the session trace at trial bin right edges, then discretizes retained values by within-session tertiles.

ii.
```python
wheel_vals, wheel_ok = behavior_per_interval(wt, wv, interval_begs)
wheel_bin, wheel_edges = discretize_tertiles(wheel_kept)
```

iii. The agent says this follows reference extraction and uses session tertiles because units/distributions are session-specific and categorical outputs are required.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 33.33rd and 66.67th percentiles over all retained trial×time values in a session define low/medium/high; `searchsorted(..., side='right')` assigns 0/1/2. A session is rejected if edges tie.

ii.
```python
edges = np.percentile(flat, [100.0 / 3 * i for i in range(1, 3)])
binned = np.searchsorted(edges, flat, side='right').reshape(values.shape)
if not np.all(np.diff(edges) > 0): raise RuntimeError(...)
```

iii. Tertiles create roughly balanced classes; tied thresholds are interpreted as a stuck wheel/broken stream.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is queried on the same stimulus-aligned window, but at each neural bin's right edge (`-0.48 ... 1.50`) rather than its center.

ii.
```python
q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
interp = np.interp(q.ravel(), target_times, target_vals).reshape(q.shape)
```

iii. The agent chose right edges to reproduce `get_behavior_per_interval`'s query grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and timestamps from left-camera ROI motion energy, falling back to the right camera.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
    v = df['whiskerMotionEnergy'].to_numpy(dtype=float)
```

iii. This is the reference camera preference order and released whisker-pad measure.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Finite camera timestamps are retained, the raw motion-energy trace is linearly interpolated at bin right edges, and values are discretized with session tertiles; there is no filtering/normalization.

ii.
```python
return t[finite], v[finite], view
me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
me_bin, me_edges = discretize_tertiles(me_kept)
```

iii. The released trace is used directly; tertiles satisfy categorical-output requirements across camera-specific scales.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The same retained-session 33.33/66.67 percentile rule produces 0/1/2, and tied edges reject the session.

ii.
```python
me_bin, me_edges = discretize_tertiles(me_kept)
if not np.all(np.diff(me_edges) > 0): raise RuntimeError(...)
```

iii. This aims for equal occupancy and excludes a documented broken, mostly-zero ROI.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera timestamps share the session clock; interpolation evaluates the stimulus-aligned trace at each 20-ms bin's right edge, with coverage checks at both ends.

ii.
```python
idx_beg = np.searchsorted(target_times, safe_b, side='right')
q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
```

iii. This intentionally mirrors the reference behavior helper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing probes are skipped if others exist; missing whisker data tries the other camera. Invalid/NaN or incompletely covered trials are masked. Sessions with no spikes, fewer than five good units, fewer than two trials, missing behavior, or degenerate tertiles raise; worker exceptions are logged in `failed_sessions`. True sparse/all-zero neural trials inside valid recordings remain.

ii.
```python
except Exception as exc:
    return ('fail', eid, f'{type(exc).__name__}: {exc}')
mask &= wheel_ok & me_ok
if n_trials_kept < MIN_TRIALS: raise RuntimeError(...)
```

iii. The agent distinguishes missing recording coverage from genuine silence, excludes unusable data globally across streams, and preserves provenance for omitted sessions.

## 10-a. What are the most time-consuming steps of the code?

i. Spike-sorting I/O/merge is the main bottleneck (about 8 seconds/session in notes); spike binning and behavior interpolation are secondary. Full decoder training is costly but outside conversion.

ii.
```python
timing['load_spikes'] = time.time() - t
timing['load_behavior'] = time.time() - t
timing['bin_spikes'] = time.time() - t
```

iii. The notes identify NFS reads as the real bottleneck and overlap them using process-level session parallelism.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. `bin_spikes` still loops once per trial for slicing and `bincount`; `trial_number_in_block` loops over trials; probe loading and result-to-list assembly also loop but involve variable-sized I/O/data. The behavior query grid is already vectorized.

ii.
```python
for k in range(n_trials):
    ...
    out[k] = counts.reshape(n_units, NBINS)
for i in range(len(p)):
    ...
```

iii. The agent deliberately removed per-spike loops and per-trial interpolation-object construction; full vectorization of ragged spike windows is less straightforward.

## 10-c. What processing does the code repeat multiple times?

i. Each process lazily constructs its own ONE/atlas handle; every probe independently loads and merges sorting metadata; each behavior calls the same coverage/interpolation routine; outputs are copied into per-trial arrays after session matrices are built.

ii.
```python
for pid, pname in zip(pids, probe_names):
    sp, cl, ch = ssl.load_spike_sorting()
wheel_vals, wheel_ok = behavior_per_interval(...)
me_vals, me_ok = behavior_per_interval(...)
```

iii. Per-process connections are required for multiprocessing, per-probe work is needed before merging, and sharing one helper keeps wheel/camera treatment consistent.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges cluster and channel fields beyond those ultimately saved; computes recording spans, many statistics, UUID provenance, tertile edges, and optional diagnostic plots/timing that the decoder does not consume. It also creates continuous behavior arrays that are discarded after discretization and stores repeated per-trial categorical/time rows.

ii.
```python
result = {'info': {'cluster_uuids': ..., 'wheel_speed_tertile_edges': ...,
                   'mean_firing_rate_hz': ...}, 'timing': timing}
outputs[:, 2, :] = wheel_bin
```

iii. The discarded/intermediate work supports QC, reproducibility, failure diagnosis, and format compatibility; notes also say the reference's raw-ephys sampling-rate metadata load was intentionally omitted because it does not affect values.
