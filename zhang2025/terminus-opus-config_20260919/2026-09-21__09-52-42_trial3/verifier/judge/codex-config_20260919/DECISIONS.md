# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the BWM release freeze (`bwm_release.csv`), optionally restricts it with `DATALIMIT_SUBSET.csv`, builds/uses an offline ONE index, and processes every unique session in parallel. Per session it loads every listed probe with `SpikeSortingLoader` and trials, wheel, and camera motion energy with `SessionLoader`.

ii.
```python
bwm = pd.read_csv(FREEZE_FILE, index_col=0)
eids = list(dict.fromkeys(bwm.eid))
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
sl = SessionLoader(one=one, eid=eid)
```

iii. The AI explains that the shipped release tables do not index revision folders, so it rebuilds a local index, and uses the release freeze because offline ONE cannot call `eid2pid`.

## 1-b. How are the data split into subjects?

i. Subject names come from the release table. Converted sessions retain `probes.iloc[0].subject`; unique names are sorted and each session receives an index into that list.

ii.
```python
subject=str(probes.iloc[0].subject)
subjects = sorted({r['subject'] for r in results})
subject_idx = np.array([sub_idx[r['subject']] for r in results], dtype=int)
```

iii. The release table already provides stable subject identifiers, so no filename parsing is needed.

## 1-c. How are the data split into sessions?

i. The release table's `eid` is treated as the session key; probe rows sharing an `eid` are grouped and converted together.

ii.
```python
eids = list(dict.fromkeys(bwm.eid))
jobs = [(e, bwm[bwm.eid == e], ...) for i, e in enumerate(eids)]
```

iii. This mirrors the reference release freeze and permits probes in one behavioral session to be merged.

## 1-d. How are the data split into trials?

i. One row of the loaded trials table defines a trial. Retained `stimOn_times` define windows from -0.5 to +1.5 seconds; neural and behavioral arrays are sliced/binned for each retained row.

ii.
```python
align = trials[ALIGN_TIME].to_numpy()[keep]
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
```

iii. The AI says this matches the reference stimulus-onset trialization and 2-second decoding window.

## 1-e. How are trials filtered based on quality controls?

i. It excludes RT outside 0.08–2 s, trials longer than 10 s from go cue to feedback, no-choice trials, and NaNs in six required columns. It subsequently requires complete wheel and whisker coverage and at least two surviving trials per session.

ii.
```python
bad = (rt < MIN_RT) | (rt > MAX_RT)
bad |= (tr['feedback_times'] - tr['goCue_times']) > MAX_TRIAL_LEN
bad |= tr['choice'] == 0
for col in NAN_EXCLUDE: bad |= tr[col].isna()
valid = ok_w & ok_m
```

iii. The AI identifies this as the reference `load_trials_and_mask(max_trial_len=10)` predicate, with behavior coverage required because both signals are mandatory outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from `spikes.times` and `spikes.clusters` for every probe. Cluster metrics and acronyms determine unit inclusion and region labels.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
times.append(spikes['times'])
clus.append(spikes['clusters'] + offset)
```

iii. These are the reference pipeline's spike-event and unit-assignment variables.

## 2-b. How is the `neural` data processed?

i. Probes are merged with noncolliding cluster offsets. Selected spikes are sorted, restricted to each trial window, and counted by unit in 100 20-ms bins. Unlike the human solution, counts are not divided by bin width, so the saved values are spike counts, not Hz.

ii.
```python
tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
flat = np.bincount(sc[s] * N_BINS + tb, minlength=n_units * N_BINS)
out[k] = flat.reshape(n_units, N_BINS)
```

iii. The AI says this is a vectorized, bit-identical equivalent of reference `bincount2D`; it deliberately emphasizes counts as the reference cached representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains clusters with `label >= 1`, removes Beryl `root` and `void`, and drops sessions with fewer than five retained units.

ii.
```python
keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, ('root', 'void'))
if len(unit_idx) < MIN_UNITS_PER_SESSION: raise RuntimeError(...)
```

iii. The AI notes that `label >= 1` exactly reproduces the paper's 75,708 well-isolated neurons. It interprets `root`/`void` as nongrey matter and the paper's five-neuron regional criterion as a session-level minimum.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window is anchored to `stimOn_times`; spike timestamps in `[onset-0.5, onset+1.5)` are placed into bins relative to that start.

ii.
```python
ALIGN_TIME = 'stimOn_times'
begs = align_times + TIME_WINDOW[0]
tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
```

iii. The AI follows the explicit stimulus-onset requirement and the reference caching parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms: 100 bins across two seconds. Raw spikes are binned once; there is no later rebinning or smoothing.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((1.5 - (-0.5)) / BINSIZE))
```

iii. The AI cites the reference code and paper's 2-s, 20-ms, 100-step configuration.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the fixed trial window, bin size, and `stimOn_times` alignment, rather than measured from another raw column.

ii.
```python
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
inputs[:, 0, :] = tgrid.astype(np.float32)
```

iii. The AI treats it as the decoding time axis implied by the mandated alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It creates 100 bin-end values from -0.48 through 1.50 s. It also creates an extra binary `stim_onset` input, although that was not one of the two requested inputs.

ii.
```python
tgrid = np.linspace(-0.5 + 0.02, 1.5, 100)
onset[int(round(0.5 / 0.02))] = 1.0
```

iii. The AI chose bin ends to match reference behavioral interpolation and added the indicator because the general format guidance says time events can be represented as binary series.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time value reports the end of each neural count bin: neural bin `i` spans `[start+i*20 ms, start+(i+1)*20 ms)`, while input `i` is the latter boundary.

ii.
```python
inputs[:, 0, :] = tgrid
# bin_spikes: floor((t - beg) / BINSIZE)
```

iii. The AI documents this convention explicitly and uses the same grid for behavior; it differs from the human solution's bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full unfiltered `trials.probabilityLeft` sequence; changes mark block starts.

ii.
```python
tib_all = trial_in_block(trials['probabilityLeft'].to_numpy())
new[1:] = pl[1:] != pl[:-1]
```

iii. No raw block ID exists, so the AI infers blocks from the prior, as does the human solution.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. It computes a zero-based distance from the latest prior change before trial filtering, then broadcasts the retained trial's scalar across all 100 time bins.

ii.
```python
idx = np.arange(len(pl))
return idx - np.maximum.accumulate(np.where(new, idx, 0))
inputs[:, 2, :] = tib[:, None]
```

iii. Counting before masking preserves the animal's actual position even when unusable trials are later removed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice` for retained trials.

ii.
```python
choice = trials['choice'].to_numpy()[keep]
```

iii. The AI verified the IBL sign convention empirically against stimulus side on correct trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `+1` (left) becomes 0 and `-1` (right) becomes 1; no-choice trials were already removed. The value is broadcast over time.

ii.
```python
y_choice = np.where(choice > 0, 0, 1).astype(np.int16)
outputs[:, 0, :] = y_choice[:, None]
```

iii. This is exactly the requested mapping and was checked against raw trials.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
pl = trials['probabilityLeft'].to_numpy()[keep]
```

iii. The AI verified that the column represents the literal left-stimulus probability.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 become 0, 1, and 2 and are broadcast across the trial. Unexpected values are removed defensively.

ii.
```python
y_prior = np.select([np.isclose(pl, .2), np.isclose(pl, .5), np.isclose(pl, .8)],
                    [0, 1, 2], default=-1)
outputs[:, 1, :] = y_prior[:, None]
```

iii. This is the mapping explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from absolute wheel velocity returned by `SessionLoader`, itself computed from wheel timestamps and positions.

ii.
```python
sl.load_wheel()
out['wheel_speed'] = (sl.wheel['times'].to_numpy(),
                      np.abs(sl.wheel['velocity'].to_numpy()))
```

iii. This matches reference `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` computes filtered velocity; the code takes its absolute value, linearly interpolates finite samples to trial bin ends, checks coverage, and discretizes the resulting retained session trace.

ii.
```python
out[k] = interp1d(t[finite], v[finite], kind='linear',
                  fill_value='extrapolate')(grid)
y_wheel, thr_w = tertile_bins(wheel)
```

iii. The AI says interpolation reproduces `get_behavior_per_interval`; discretization is task-mandated.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session 33⅓ and 66⅔ percentiles over all retained trial-bin samples define low, medium, and high.

ii.
```python
lo, hi = np.percentile(finite, [100 / 3, 200 / 3])
np.digitize(values, [lo, hi])
```

iii. The AI argues session scales differ and balanced within-session classes avoid encoding rig/session identity.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at onset-relative bin ends from -0.48 to 1.50 s, one value for each neural bin.

ii.
```python
grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
```

iii. This matches the reference behavior-resampling grid, though values correspond to neural-bin ends rather than centers.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and `times` from left-camera ROI motion energy, falling back to right camera.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sl.load_motion_energy(views=[view])
    vals = sl.motion_energy[cam]['whiskerMotionEnergy'].to_numpy()
```

iii. Left-then-right is the reference camera preference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released scalar trace is linearly interpolated over finite samples onto trial bin ends, coverage-checked, and discretized; no additional filtering or normalization is applied.

ii.
```python
whisk, ok_m = bin_behaviour(mt, mv, align)
y_whisk, thr_m = tertile_bins(whisk)
```

iii. The AI says the released motion-energy trace should be used directly and reference interpolation retained.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses per-session tertiles over all retained trial-bin samples, with a tiny upper-threshold adjustment for degenerate traces.

ii.
```python
lo, hi = np.percentile(finite, [100 / 3, 200 / 3])
if not (hi > lo): hi = lo + 1e-12
```

iii. The same balanced-class, session-specific-scale rationale as wheel speed is given.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera values are interpolated at the same 100 stimulus-relative bin ends used for wheel and paired one-for-one with neural bins.

ii.
```python
grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
```

iii. Camera and neural timestamps share the synchronized session clock; interpolation supplies the common grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing probes are skipped; sessions without spikes, either behavior, five good grey-matter units, or two fully covered trials are skipped. NaNs in required trial fields exclude trials; finite behavior samples are interpolated, but inadequate coverage excludes the trial. Worker exceptions become explicit skip records.

ii.
```python
if len(spikes) == 0 or 'times' not in spikes: continue
if beh['whisker_motion_energy'] is None: raise RuntimeError(...)
finite = np.isfinite(v)
return {'eid': eid, 'error': f'{type(exc).__name__}: {exc}'}
```

iii. The AI favors explicit exclusion rather than silently manufacturing mandatory outputs and reconciles every skipped session in its notes.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting dominates, followed by spike binning and behavior loading. The full-run cumulative timings were 2330.0 s, 664.6 s, and 428.6 s respectively (summed across parallel workers).

ii.
```python
timings['load_spikes'] = time.time() - t
timings['bin_spikes'] = time.time() - t
timings['load_behaviour'] = time.time() - t
```

iii. The agent instrumented each stage and reported the measured totals, rather than guessing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. `bin_spikes` and `bin_behaviour` still loop over trials; probe loading loops over probes, and result assembly loops over sessions/neurons. The per-trial loops could theoretically be further batched, although slicing variable-length windows makes the current approach clear and most work inside each iteration is NumPy/SciPy.

ii.
```python
for k in range(len(align_times)):
    flat = np.bincount(...)
for k in range(n):
    out[k] = interp1d(...)(grid)
```

iii. The AI describes its code as vectorized relative to the reference: streams are sorted once, boundaries use `searchsorted`, and parallelism is across sessions rather than spawning pools per behavior/session.

## 10-c. What processing does the code repeat multiple times?

i. Each worker constructs its own ONE client; each session loads and merges probe tables, separately interpolates wheel and whisker traces through the same `bin_behaviour` routine, and builds the same time/onset arrays. Diagnostic mode additionally stacks and summarizes converted arrays.

ii.
```python
one = get_one()
wheel, ok_w = bin_behaviour(wt, wv, align)
whisk, ok_m = bin_behaviour(mt, mv, align)
tgrid = np.linspace(...)
```

iii. The AI explicitly avoided one earlier repetition by building the local index once in the parent before starting workers; most remaining repetition is per-session necessity.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges full cluster and channel information before retaining a small QC subset, computes and stores detailed timing/diagnostic/session metadata not used by the decoder, creates an extra stimulus-onset input, and optionally builds expensive diagnostic plots. It deliberately omits the reference's unused raw-electrophysiology sampling-frequency lookup.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
df = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
inputs[:, 1, :] = onset
if show: plot_processing(...)
```

iii. The notes justify metadata/plots as validation aids and the onset indicator from generic task guidance; they identify and remove the reference sampling-frequency call because it is unused and network-dependent.
