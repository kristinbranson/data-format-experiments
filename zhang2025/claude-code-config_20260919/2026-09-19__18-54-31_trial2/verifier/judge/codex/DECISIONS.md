# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the Brain-Wide Map freeze CSV, optionally restricts it with `DATALIMIT_SUBSET.csv`, and uses ONE/IBL loaders for trials, wheel, camera motion energy, and each listed probe's spike sorting. Sessions are processed independently in a process pool.

ii.
```python
bwm = pd.read_csv(BWM_FREEZE_FILE, index_col=0)
if os.path.exists(DATALIMIT_FILE):
    sub = pd.read_csv(DATALIMIT_FILE)
    bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
...
spikes, clusters, channels = loader.load_spike_sorting()
sess_loader.load_trials(); sess_loader.load_wheel()
sess_loader.load_motion_energy(views=[view])
```

iii. The notes say this is the same 459-session/699-insertion freeze used by the reference code, while ONE resolves the staged files and the subset prevents processing unavailable sessions.

## 1-b. How are the data split into subjects?

i. Subject labels come directly from the freeze CSV. The final sorted unique subject list is indexed per retained session.

ii.
```python
'subject': str(g.subject.iloc[0]),
...
subjects = sorted({r['subject'] for r in results})
'subject_idx': np.array([subject_index[r['subject']] for r in results])
```

iii. The agent treats the release's subject field as the authoritative mouse identifier.

## 1-c. How are the data split into sessions?

i. Rows of the release table are grouped by `eid`; probe rows within an `eid` are merged into one session population. Sessions are sorted by lab, subject, and date.

ii.
```python
for eid, g in bwm.groupby('eid', sort=False):
    g = g.sort_values('probe_name')
    sessions.append({'eid': str(eid), 'pids': [str(p) for p in g.pid], ...})
```

iii. The notes identify `eid` as the release's session identifier and state that probes in the same session share behavior and should be merged.

## 1-d. How are the data split into trials?

i. The trials table already has one row per trial. Each trial is represented by a two-second interval beginning 0.5 s before its `stimOn_times`; retained rows become separate list elements.

ii.
```python
align_times = trials_df[ALIGN_TIME].to_numpy(dtype=float)
interval_begs = align_times + TIME_WINDOW[0]
...
'neural': [neural[k] for k in range(n_keep)]
```

iii. This follows the reference interval geometry: stimulus onset, -0.5 to +1.5 s.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed for RT outside 0.08–2 s, trial length over 10 s, missing required events, no choice, nonfinite alignment time, or incomplete/nonfinite wheel or whisker coverage. Sessions need at least two retained trials.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in NAN_EXCLUDE: query += f' | {event}.isnull()'
query += ' | (choice == 0)'
...
keep &= beh_good[name]
```

iii. The agent says this ports `load_trials_and_mask` with the settings used by `prepare_data`, then applies the behavior-coverage mask used during reference alignment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from `spikes.times` and `spikes.clusters` for every probe; cluster labels and acronyms determine curation and regions.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
sel_times = spikes['times'][spike_idx]
sel_cl = sel_clusters.index.to_numpy()[ib].astype(np.int64)
```

iii. The notes describe this as a port of the IBL spike loader, omitting only a raw-AP sampling-rate lookup unused by conversion.

## 2-b. How is the `neural` data processed?

i. Selected spikes are counted per cluster in nonoverlapping 20 ms bins, vectorized with one `bincount` per probe; probe matrices are concatenated by neuron. The stored values remain spike counts (`float32`), not Hz.

ii.
```python
bin_of_spike = np.floor(t_rel / binsize).astype(np.int64)
lin = ((trial_of_spike * n_clusters) + clusters[flat_idx]) * n_bins + bin_of_spike
out += np.bincount(lin, ...).reshape(n_trials, n_clusters, n_bins)
neural = np.concatenate(neural_trials, axis=1)
```

iii. The agent claims the binning is bit-identical to reference `bincount2D` and documents the result as spike counts. It chose vectorization for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps `clusters.label >= 1`, excludes Beryl `root` and `void`, drops clusters with no spikes, and rejects sessions with fewer than five remaining neurons.

ii.
```python
iok = clusters_labeled['label'] >= qc
...
sel = (~np.isin(beryl, NON_GREY_MATTER)) & has_spikes
...
if neural.shape[1] < MIN_NEURONS_PER_SESSION: raise RuntimeError(...)
```

iii. The notes justify stringent units and grey matter from the data paper, no-spike removal from reference binning, and the five-neuron rule from a regional inclusion criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike times are sliced into `[stimOn-0.5, stimOn+1.5)` and binned relative to each interval start.

ii.
```python
interval_begs = align_times + TIME_WINDOW[0]
i0 = np.searchsorted(times, interval_begs, side='left')
t_rel = times[flat_idx] - interval_begs[trial_of_spike]
```

iii. The agent follows the reference parameters and uses the common synchronized session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, producing 100 bins over two seconds. Raw spikes are binned once; no smoothing or further rebinning is applied.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The notes prefer the executable reference and paper statement “20-ms bins, T=100” over one contradictory 50-ms sentence.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the requested `stimOn_times` alignment and the fixed time window/bin grid, rather than another measured stream.

ii.
```python
align_times = trials_df['stimOn_times'].to_numpy(dtype=float)
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
```

iii. The agent says this new decoder input is defined by the reference alignment geometry.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It computes the 100 bin centers from -0.49 through +1.49 s and repeats the same ramp for every trial.

ii.
```python
inputs[:, 0, :] = bin_centres[None, :]
```

iii. Bin centers were chosen as the natural timestamps representing neural bins.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Each value is the center of the corresponding neural count bin, on the same stimulus-relative grid.

ii.
```python
bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5))
```

iii. The notes explicitly validate the range and the shared 100-bin axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive runs of `trials.probabilityLeft` in the full trials table.

ii.
```python
tnb_all = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. No block ID is available, so constant-prior runs identify blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter increments while the prior is unchanged and resets on a change; it is computed before filtering and broadcast across all time bins.

ii.
```python
counter = counter + 1 if same else 0
out[i] = counter
...
inputs[:, 1, :] = tnb_all[keep_idx].astype(np.float32)[:, None]
```

iii. Pre-filter computation preserves the animal's true position even when adjacent trials are excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`.

ii.
```python
choice = trials_df['choice'].to_numpy()[keep_idx]
```

iii. The agent empirically checked IBL's sign convention against stimulus side on correct trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `+1` becomes left/class 0 and `-1` becomes right/class 1; the label is broadcast across time. No-choice trials were already removed.

ii.
```python
choice_cls = (choice < 0).astype(np.int64)
outputs[:, 0, :] = choice_cls[:, None]
```

iii. This matches the requested left/right encoding and verified IBL convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
pleft = trials_df['probabilityLeft'].to_numpy()[keep_idx]
```

iii. The task explicitly names the three block-prior values.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to classes 0, 1, and 2 using `isclose`, validated, and broadcast across time.

ii.
```python
for val, cls in ((0.2, 0), (0.5, 1), (0.8, 2)):
    pleft_cls[np.isclose(pleft, val)] = cls
outputs[:, 1, :] = pleft_cls[:, None]
```

iii. This is the mapping required by the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is the absolute value of `SessionLoader` wheel velocity, derived by that loader from wheel position and timestamps.

ii.
```python
sess_loader.load_wheel()
traces['wheel_speed'] = (sess_loader.wheel['times'].to_numpy(),
                         np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The notes say this matches reference `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Absolute velocity is linearly interpolated within each trial at 100 bin right edges, checked for finite/full coverage, then converted to per-session tercile classes.

ii.
```python
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
ws_edges = discretize_terciles(ws)
```

iii. The agent intentionally ports reference behavior interpolation and uses session thresholds to handle camera/session scale and balance classes.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Thresholds are the retained session's 1/3 and 2/3 quantiles across all trial-time values. Classes use `> e1` and `> e2`; degenerate thresholds receive a custom repair.

ii.
```python
e1, e2 = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
return ((values > e1).astype(np.int64) + (values > e2).astype(np.int64))
```

iii. Per-session terciles normalize differing scales and target balanced low/medium/high classes; edge repair addresses mass at zero.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It uses the same two-second trial interval but samples at each neural bin's right edge, whereas the time input represents bin centers.

ii.
```python
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. The agent says this exactly follows reference `get_behavior_per_interval` and therefore considered it aligned bin-for-bin.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and timestamps from left-camera ROI motion energy, falling back to the right camera.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
    me = sess_loader.motion_energy[key]
```

iii. The notes identify this as the reference camera preference/fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is linearly interpolated at trial-bin right edges, finite/full coverage is required, and values are discretized by session terciles.

ii.
```python
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
wme_edges = discretize_terciles(wme)
```

iii. The agent applies no extra filtering or normalization; session terciles account for camera-dependent arbitrary units.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same session-wide 1/3 and 2/3 quantiles, strict-greater class assignment, and degenerate-edge repair as wheel speed.

ii.
```python
wme_cls = apply_terciles(wme, *wme_edges)
```

iii. The notes justify balanced, session-relative classes because motion-energy scale differs by camera.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It shares stimulus onset and trial interval with neural data but is sampled at bin right edges, not neural-bin centers.

ii.
```python
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. The agent considered this the exact behavior-alignment operation in the supplied reference library.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trial events and incomplete/nonfinite behavior trials are dropped. Missing camera views trigger a fallback; sessions fail if neither view, enough trials, or enough curated neurons are available. Worker exceptions are recorded in metadata. NaN priors fail explicit validation.

ii.
```python
except Exception: continue
if cam_used is None: raise RuntimeError(...)
if not np.all(np.isfinite(y)): continue
return {'eid': session_info['eid'], 'error': ..., 'traceback': ...}
```

iii. The notes argue finite values are required by this decoder and enumerate failures rather than silently fabricating data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and binning spike sorting dominates (about 3.1 s for one probe and 6.5 s for two); pickle writing is another full-data cost.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
timings['spikes'] = time.time() - t0
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Timed sample runs in the notes show behavior work is under roughly 0.7 s/session and spike I/O/binning dominates.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Behavior interpolation still loops over trials; `trial_number_in_block`, probe loading, result-to-list conversion, region indexing, and summary concatenation also use loops/comprehensions. Spike binning is largely vectorized, though it constructs trial spike slices with a comprehension.

ii.
```python
for k in range(n_trials):
    ... interp1d(...)(x_interp)
flat_idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1) if b > a])
```

iii. The notes prioritize vectorizing expensive spike binning and keeping behavior interpolation in-process; its measured cost was only 0.0–0.1 s/session.

## 10-c. What processing does the code repeat multiple times?

i. It calls the same behavior-binning routine separately for wheel and whisker streams, computes/uses analogous terciles twice, maps regions with Python comprehensions, and creates a ONE client in every session worker task. Plotting also recomputes a population PSTH. A duplicated `ws_edges` line visible in one captured excerpt is not present in the final source.

ii.
```python
for name, (tt, tv) in traces.items():
    vals, good = bin_behaviour(tt, tv, interval_begs)
ws_edges = discretize_terciles(ws)
wme_edges = discretize_terciles(wme)
```

iii. The two behavior streams necessarily share processing; the agent reports avoiding larger repetition by sharing one `SessionLoader` and loading only required behaviors.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `channels` is loaded but only indirectly consumed during cluster merging; spike-sorter collection names and detailed timing are collected but not used by the decoder; optional plots compute diagnostics only. It also loads/keeps metadata fields such as lab/date and constructs verbose session metadata not used in training.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
sorters.add(collection)
timings['behaviour_bin'] = time.time() - t0
if show_processing: _plot_processing(...)
```

iii. The notes emphasize that it already removed more substantial waste from the original pipeline: six unused behaviors, repeated session lookup, raw-AP sampling-rate access, and per-signal process pools.
