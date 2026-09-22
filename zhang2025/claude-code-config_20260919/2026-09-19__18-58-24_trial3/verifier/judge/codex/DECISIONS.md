# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the BWM release CSV and cached session table, constructs every released session path, scans each session's `alf` tree to rebuild ONE's dataset table, then uses ONE loaders for trials, behavior, spikes, and clusters. It starts from all 459 release EIDs and parallelizes conversion by session.

ii.
```python
bwm = pd.read_csv(BWM_CSV, index_col=0)
for eid in bwm.eid.unique():
    r = sess.loc[eid]
    paths[eid] = (f"{ROOT}/{r['lab']}/Subjects/{r['subject']}/"
                  f"{str(r['date'])}/{int(r['number']):03d}")
...
one._cache['datasets'] = build_dataset_table(session_paths_map)
eids = list(bwm.eid.unique())
```

iii. The agent found the shipped dataset table stale relative to revised files on disk; rebuilding it made ONE resolve the staged files and avoided silently incomplete trial tables.

## 1-b. How are the data split into subjects?

i. Subject IDs come from `bwm_release.csv`. The final subject list is sorted and each retained session gets an index into it.

ii.
```python
info['subject'] = sub.subject.iloc[0]
subjects = sorted({r['subject'] for r in kept})
subj_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. The release metadata already supplies unique mouse identifiers, so no path parsing or inference is needed.

## 1-c. How are the data split into sessions?

i. Each unique BWM `eid` is treated as one session; all probe rows sharing that EID are merged within the session.

ii.
```python
eids = list(bwm.eid.unique())
sub = bwm[bwm.eid == eid]
for _, r in sub.iterrows():
    sp, cl, n_tot = load_spiking_data(one, r.pid, eid, r.probe_name)
```

iii. EID is the native IBL session identifier, and pooling probes follows the whole-session decoder design.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies one row per trial. Stimulus-onset windows are then indexed separately for every retained trial.

ii.
```python
if sess_loader.trials.empty:
    sess_loader.load_trials()
align_times = trials_sel[ALIGN_TIME].to_numpy()
for k in range(ntrials):
    neural_list.append(binned[k])
```

iii. The IBL trials table already defines trial boundaries and events, so the agent used its rows directly.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded for reaction time outside 0.08–2 s, choice 0, missing required events, or `feedback_times-goCue_times > 10 s`. The agent additionally rejects behavior windows lacking coverage/containing NaNs and windows with no spikes, and requires at least two surviving trials.

ii.
```python
nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
query += ' | (feedback_times - goCue_times > 10.0)'
query += ' | (choice == 0)'
...
keep = beh_good & neural_covered
```

iii. The first mask was presented as the BWM/reference mask. Behavior exclusions enforce complete non-NaN target arrays; the no-spike rule was justified as detecting ephys gaps rather than physiology.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from `spikes.times` and `spikes.clusters`; cluster labels select units and cluster acronyms provide region metadata.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
iok = clusters_labeled['label'] >= qc
selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)
```

iii. These are the released spike times and Kilosort assignments used by the reference loader.

## 2-b. How is the `neural` data processed?

i. Good clusters from all probes are renumbered and merged, spikes are sorted, and each trial is binned into 100 left-closed 20-ms bins. The stored values are raw float32 spike counts, not firing rates or z-scores.

ii.
```python
idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
flat = c[keep].astype(np.int64) * NBINS + idx[keep]
out[k] = np.bincount(flat, minlength=n_clusters * NBINS).reshape(n_clusters, NBINS)
```

iii. The agent said the reference caches counts and leaves normalization to model-side processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `clusters.label >= 1`. The agent retains all anatomical labels, including `root` and `void`, and drops a session if fewer than five good units remain. Trials with zero population spikes are also dropped.

ii.
```python
iok = clusters_labeled['label'] >= qc
...
if n_clusters < MIN_NEURONS:
    return None, info
neural_covered = binned.sum(axis=(1, 2)) > 0
```

iii. Label 1 exactly reproduced the paper's 75,708 well-isolated neurons. The agent argued whole-session decoding did not require region filtering, used the paper's five-neuron criterion at session level, and treated fully silent windows as recording dropouts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to each trial's `stimOn_times` over [-0.5, 1.5) s.

ii.
```python
ALIGN_TIME = 'stimOn_times'
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```

iii. This matches the specified stimulus alignment and the reference choice-decoding parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, yielding 100 bins in a two-second window. Spikes are counted directly into those bins; no smoothing or later rebinning is applied.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The agent cites both the caching parameters and methods paper's 100 × 20-ms representation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the fixed bin grid relative to `stimOn_times`, rather than measured from another raw signal.

ii.
```python
align_times = trials_sel[ALIGN_TIME].to_numpy()
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. Stimulus onset is the mandated alignment event; the time coordinate is therefore determined by the binning parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent computes the centers of the 100 bins, from -0.49 to 1.49 s, and repeats that vector for every trial.

ii.
```python
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
inp[0] = tvec
```

iii. Bin centers were chosen as the natural timestamp for binned neural samples.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Each value is the center timestamp of the corresponding neural spike-count bin.

ii.
```python
idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. Both use the same onset, window, bin width, and ordering.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive runs of `trials.probabilityLeft`.

ii.
```python
p = np.asarray(probability_left, dtype=float)
same = (p[1:] == p[:-1])
newblock[1:] = ~same
```

iii. There is no explicit block ID in the trials table, while the prior is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Changes in prior start a block; trials are numbered from zero within each block before filtering, and the retained value is broadcast over time.

ii.
```python
block_id = np.cumsum(newblock) - 1
for b in np.unique(block_id):
    m = block_id == b
    out[m] = np.arange(m.sum())
...
inp[1] = tnib_keep[k]
```

iii. Computing before exclusions preserves the animal's actual position rather than its position among retained trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice`.

ii.
```python
choice_raw = trials_keep['choice'].to_numpy()
```

iii. The agent verified IBL's sign convention against feedback and stimulus-side columns.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After dropping no-response trials, +1 becomes left/0 and -1 becomes right/1; the label is repeated across all 100 timepoints.

ii.
```python
choice = (choice_raw < 0).astype(np.int8)
out[0] = choice[k]
```

iii. This implements the task's requested left=0, right=1 coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
pleft = trials_keep['probabilityLeft'].to_numpy()
```

iii. This column is the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 map to 0, 1, and 2 using tolerant comparisons, then are broadcast over time. Unexpected values reject the session.

ii.
```python
for val, lab in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(pleft, val)] = lab
out[1] = prior[k]
```

iii. The mapping is explicitly required by the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from wheel timestamps and position loaded by `SessionLoader`; the loader supplies filtered velocity and the agent takes its absolute value.

ii.
```python
sess_loader.load_wheel()
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. This mirrors the reference's `wheel-speed = abs(velocity)` definition.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` uniformly interpolates position and computes filtered velocity; the absolute velocity is linearly interpolated per trial onto 100 right-bin-edge timestamps, with incomplete/NaN trials rejected, then discretized.

ii.
```python
x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. The agent transcribed the reference behavior routine and intentionally used its right-edge query grid.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The two thresholds are the within-session 1/3 and 2/3 quantiles over all retained trial-time values; `np.digitize` assigns low/medium/high, with a fallback for identical edges.

ii.
```python
edges = np.quantile(values.ravel(), [1. / 3., 2. / 3.])
labels = np.digitize(values, edges).astype(np.int8)
```

iii. Session tertiles accommodate scale differences and zero inflation while balancing the three decoder classes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Both use the same stimulus onset, two-second window, and 100 indices, but wheel values are sampled at bin right edges while neural samples represent counts over the bins (and the time input uses centers).

ii.
```python
begs = align_times + TIME_WINDOW[0]
x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
```

iii. The agent believed this exactly followed the reference behavior code and preserved indexwise alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It derives from `whiskerMotionEnergy` and camera timestamps, preferring the left camera and falling back to right.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
    whisker = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
```

iii. This matches the reference's camera preference and accommodates sessions lacking left-camera motion energy.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used without filtering or normalization, linearly interpolated to trial-relative right bin edges, checked for coverage/NaNs, and discretized.

ii.
```python
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
whisk_lab, whisk_edges = discretize_tertiles(whisk_vals)
```

iii. The agent found no extra reference preprocessing and used the same scheme as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses within-session tertile thresholds over all retained samples, with the same degenerate-edge fallback as wheel speed.

ii.
```python
edges = np.quantile(flat, [1. / 3., 2. / 3.])
labels = np.digitize(values, edges).astype(np.int8)
```

iii. Camera-dependent arbitrary units make fixed cross-session thresholds inappropriate; tertiles provide low/medium/high session-relative movement.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It shares stimulus onset, window, and 100 array positions with neural data, but is sampled at right edges rather than neural-bin centers.

ii.
```python
x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. The agent treated the reference routine's right-edge samples as index-aligned with corresponding neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Revised files are discovered by scanning disk. Missing behavior drops a session; incomplete or NaN behavior drops a trial; nonfinite spike times are removed; silent windows are dropped; unexpected priors reject a session; and sessions need two trials and five good units. Worker exceptions are recorded as skip reasons.

ii.
```python
finite = np.isfinite(spikes['times'])
keep = beh_good & neural_covered
if keep.sum() < 2: return None, info
...
except Exception as e:
    return None, {'eid': eid, 'skip_reason': f'{type(e).__name__}: {e}'}
```

iii. The agent prioritized complete finite arrays, documented drop reasons, and regarded silent windows as corrupted ephys coverage.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/merging spike sorting and session-scale file I/O dominate. The one-time filesystem scan and full conversion are also material, while binning itself was optimized.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
...
with ProcessPoolExecutor(max_workers=nw) as ex:
    for res, info in ex.map(_worker, rest, chunksize=1):
```

iii. Notes report the dataset-table scan at about 7 s, binning under 0.2 s/session, and move parallelism to 24 session workers.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning still loops over trials, behavior interpolation loops over trials, trial-number construction loops over blocks, dataset discovery walks every file, and final assembly loops over trials. The expensive reference per-trial multiprocessing was replaced with session-level parallelism and vectorized boundary searches, but the remaining variable-length slices were left as loops.

ii.
```python
for k in range(ntrials):
    a, b = i0[k], i1[k]
...
for k in range(ntrials):
    tt = target_times[idxs_beg[k]:idxs_end[k]]
```

iii. The agent judged the per-trial bincount/interpolation loops cheap and clearer than constructing large flattened index structures.

## 10-c. What processing does the code repeat multiple times?

i. `bin_behavior` repeats the same slicing, validation, interpolation, and mask construction separately for wheel and whisker. Per-trial assembly repeatedly allocates input/output arrays, and subject lookup uses repeated `list.index` calls.

ii.
```python
wheel_vals, wheel_good = bin_behavior(*traces['wheel-speed'], align_times)
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
...
subj_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. Applying one shared behavior helper ensures identical coverage and interpolation rules; the agent did not identify this as a meaningful bottleneck.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes extensive per-session diagnostics/timings and reads/keeps cluster metadata beyond the fields needed for final neural arrays. It also creates random dataset UUIDs and scans file sizes solely to rebuild ONE's cache. Plot-only data are generated only under `--show-processing` and then removed.

ii.
```python
info['n_rt_short'] = int(np.nansum(rt < 0.08))
info['n_correct'] = int(np.nansum(fb == 1))
...
rows.append((eid_uuid, uuid.uuid4(), os.path.getsize(...), ...))
```

iii. The diagnostics support paper-statistic checks and auditability; cache reconstruction works around stale metadata. They do not directly feed decoder training.
