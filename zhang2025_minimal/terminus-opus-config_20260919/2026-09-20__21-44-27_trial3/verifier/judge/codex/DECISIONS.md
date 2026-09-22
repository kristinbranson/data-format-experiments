# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the 459-session BWM freeze CSV, intersects it with sessions staged in a local ONE cache, and processes each retained EID through ONE loaders. It rebuilds ONE cache tables from staged files if needed, loads trials and behavior with `SessionLoader`, and loads each listed probe with `SpikeSortingLoader`. Sessions that raise an exception are skipped; 444 were ultimately retained.

ii.
```python
bwm_df = pd.read_csv(FREEZE_FILE, index_col=0)
available = set(one._cache['sessions'].index.astype(str))
eids = [e for e in bwm_df.eid.unique() if e in available]
jobs = [(e, bwm_df[bwm_df.eid == e]) for e in eids]
```

iii. The trajectory says the freeze contains the paper's released sessions, that ONE reproduces the reference loading pipeline, and that rebuilding cache metadata was necessary because staged revisions were newer than the shipped offline tables.

## 1-b. How are the data split into subjects?

i. Subject labels come from the freeze rows. During assembly, subjects are added in first-session order and every session gets an integer index into that list.

ii.
```python
sub = str(meta['subject'])
if sub not in subjects:
    subjects.append(sub)
subject_idx.append(subjects.index(sub))
```

iii. The agent treated the freeze's `subject` field as the authoritative unique mouse identifier; no parsing was needed.

## 1-c. How are the data split into sessions?

i. The EID is the session unit. Freeze rows are grouped by EID (preserving multiple probe rows), one job is created per EID, and each successful job becomes one session in the output.

ii.
```python
jobs = [(e, bwm_df[bwm_df.eid == e]) for e in eids]
for eid in eids:
    z = np.load(TMP_DIR / f'{eid}.npz', allow_pickle=True)
    neural.append([nb[i] for i in range(nb.shape[0])])
```

iii. The trajectory explains that probes from one session are not independent and therefore are merged rather than emitted as separate sessions.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies a table with one row per trial. Kept row indices define trial windows around each row's `stimOn_times`, and the first array dimension is converted into a list of trials.

ii.
```python
sess_loader.load_trials()
align = trials_df[ALIGN_TIME].to_numpy(dtype=float)
kidx = np.flatnonzero(keep)
neural.append([nb[i] for i in range(nb.shape[0])])
```

iii. The agent regarded the trials table as already defining trial boundaries; stimulus onset plus the fixed window defines each extracted segment.

## 1-e. How are trials filtered based on quality controls?

i. It uses `load_trials_and_mask(..., max_trial_len=10.0)`, which filters missing required trial fields, reaction times outside 0.08–2 s, no-choice trials, and trials longer than 10 s. It additionally requires nonmissing alignment and wheel and camera coverage across the decoding window, and rejects sessions with fewer than two surviving trials.

ii.
```python
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
keep = trials_mask & wheel_ok & me_ok & ~np.isnan(align)
if keep.sum() < 2:
    raise RuntimeError(f'only {keep.sum()} usable trials')
```

iii. The trajectory attributes this to the data paper's exclusions and to the reference `load_trials_and_mask`/behavior-alignment pipeline.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays come from merged probes' `spikes['times']` and `spikes['clusters']`. Cluster rows provide identities and anatomical acronyms.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
spikes, clusters = merge_probes(spikes_list, clusters_list)
st = spikes['times']
sc = spikes['clusters']
```

iii. The agent identified spike times and Kilosort cluster assignments as the reference pipeline's direct inputs, with probe merging required at session level.

## 2-b. How is the `neural` data processed?

i. All probes are merged, nonfinite spike records are removed, spikes are sorted, clusters that fired at least once are remapped contiguously, and spikes are counted into 100 nonoverlapping 20-ms bins. Crucially, the agent leaves these as spike counts rather than dividing by 0.02 s to produce Hz.

ii.
```python
binned = np.zeros((len(kidx), nneurons, NBINS), dtype=np.float32)
b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)
```

iii. The trajectory says the methods pipeline “bin[s] spike counts using all neurons” and therefore chose raw counts. It did not discuss the human conversion's subsequent division by bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No cluster-quality or in-brain filter is applied. Every merged Kilosort cluster represented in `spikes.clusters` is retained if it has at least one finite spike anywhere in the session; `root` and `void` anatomical labels can remain.

ii.
```python
finite = np.isfinite(st) & np.isfinite(sc)
st, sc = st[finite], sc[finite].astype(np.int64)
used = np.unique(sc)
acronyms = clusters['acronym'].to_numpy()[used]
```

iii. The agent explicitly reasoned that `prepare_data` uses `qc=None` and the methods paper says all neurons. It recognized the alternative stringent data-paper QC but favored exact reuse of the methods-code default.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial spans −0.5 to +1.5 s relative to `stimOn_times`. Absolute spike timestamps are searched within those bounds and converted to relative bin indices.

ii.
```python
align = trials_df[ALIGN_TIME].to_numpy(dtype=float)
beg = align + TIME_WINDOW[0]
end = align + TIME_WINDOW[1]
b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
```

iii. The agent states this exactly matches the reference caching parameters and relies on all streams sharing the synchronized session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms: the two-second window is divided into 100 nonoverlapping bins. Spikes are binned once; there is no subsequent temporal rebinning or smoothing.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The agent cites the methods paper and `0_data_caching.py` parameters as specifying 20-ms bins and 100 time steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the chosen `stimOn_times` alignment event and the fixed window/bin parameters, not from a separate sampled raw channel.

ii.
```python
ALIGN_TIME = 'stimOn_times'
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
```

iii. The trajectory says stimulus onset is the requested/reference alignment event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A common vector from −0.48 through +1.50 s is generated using the right edge of every 20-ms bin and copied to every trial.

ii.
```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. The agent believed the repository behavior helper uses `linspace(beg + binsize, end, nbins)` and chose that exact grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The right-edge time vector is assigned positionally to the 100 neural count bins. Thus input index `t` labels neural bin `t`, although the label is the bin's right edge rather than its center.

ii.
```python
inputs[:, 0, :] = BIN_TIMES[None, :]
outputs = np.empty((len(kidx), 4, NBINS), dtype=np.int64)
```

iii. The agent considered the shared 100-point reference grid sufficient for bin-for-bin alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the complete trials table's `probabilityLeft` sequence; a value change marks a new block.

ii.
```python
tib = trial_in_block(trials_df['probabilityLeft'].to_numpy())[kidx]
```

iii. The agent reasoned that `probabilityLeft` is constant within a block and the table has no separate block ID.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Trials are counted from zero, resetting when `probabilityLeft` changes. Counting happens before quality filtering, then retained trial indices are selected and values are broadcast across time.

ii.
```python
if i > 0 and pleft[i] != pleft[i - 1]:
    count = 0
out[i] = count
count += 1
inputs[:, 1, :] = tib[:, None]
```

iii. This preserves the animal's actual position in the original block even when intervening trials are later excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials_df['choice']`, where IBL encodes left as +1 and right as −1.

ii.
```python
choice = trials_df['choice'].to_numpy()[kidx]
```

iii. The trajectory says the convention was checked against `contrastLeft` on correct trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Negative choice becomes right=1 and positive choice becomes left=0; the per-trial class is broadcast across all 100 bins. No-choice trials were already filtered.

ii.
```python
choice_out = (choice < 0).astype(np.int64)
outputs[:, 0, :] = choice_out[:, None]
```

iii. This directly implements the requested left=0/right=1 mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from the trials table's `probabilityLeft` column.

ii.
```python
pleft = trials_df['probabilityLeft'].to_numpy()[kidx]
```

iii. The agent identifies this as the block prior supplied directly by the task data.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 are mapped to 0, 1, and 2, checked for unexpected values, and broadcast over time.

ii.
```python
prior_out = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
outputs[:, 1, :] = prior_out[:, None]
```

iii. The mapping is explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses the absolute value of `SessionLoader`'s wheel velocity and its timestamps; that velocity is internally derived from raw wheel position/timestamps.

ii.
```python
sess_loader.load_wheel()
np.abs(sess_loader.wheel['velocity'].to_numpy())
```

iii. The agent says this matches `load_target_behavior` and the reference use of absolute wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The loader-derived speed is linearly interpolated separately for each trial onto 100 right-edge sample times (−0.48 to +1.50 s), after coverage/NaN checks. The retained values are then discretized session-wise.

ii.
```python
x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
wheel_bin = discretize(wheel_speed[kidx])
```

iii. The agent chose right-edge interpolation because it interpreted `get_behavior_per_interval` as using `beg + binsize` through `end`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Across all retained time points in a session, the 1/3 and 2/3 quantiles are computed; duplicate edges are removed, and `np.digitize` assigns integer categories.

ii.
```python
edges = np.nanquantile(values, np.arange(1, nbins) / nbins)
edges = np.unique(edges)
return np.digitize(values, edges).astype(np.int64)
```

iii. The agent argued within-session equal-occupancy bins avoid incomparable physical scales and maintain usable classes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel values are sampled on the same number of indices and same stimulus-relative window as neural bins, but at each bin's right edge rather than the human reference's bin center.

ii.
```python
x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
outputs[:, 2, :] = wheel_bin
```

iii. The agent considered the repository's right-edge behavior grid to be the canonical alignment grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and frame `times` from the left camera when usable, otherwise the right camera.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
    df['whiskerMotionEnergy'].to_numpy()
```

iii. The agent says this follows the reference behavior loader and uses right camera only as a missing-data fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is not otherwise filtered or normalized. It is coverage-checked, linearly interpolated to the same 100 right-edge trial times, and discretized by session tertiles.

ii.
```python
v, ok = interp_behavior(df['times'].to_numpy(),
                        df['whiskerMotionEnergy'].to_numpy(), beg, end)
me_bin = discretize(me_vals[kidx])
```

iii. The agent found no additional processing in the reference path and justified session-wise thresholds because camera motion-energy units vary across sessions.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The 1/3 and 2/3 quantiles over all retained samples in that session define low, medium, and high categories, with duplicate edges removed.

ii.
```python
edges = np.nanquantile(values, np.arange(1, nbins) / nbins)
edges = np.unique(edges)
return np.digitize(values, edges).astype(np.int64)
```

iii. The trajectory emphasizes that motion-energy magnitudes depend on camera, illumination, and ROI, motivating per-session equal occupancy.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is sampled over the same stimulus-relative window and given 100 positions corresponding to neural bins, but those samples occur at bin right edges rather than centers.

ii.
```python
x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
outputs[:, 3, :] = me_bin
```

iii. The agent treated the right-edge grid used by the behavior helper as the reference alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Nonfinite spike entries are removed. Behavior trials with absent/NaN samples or inadequate edge coverage are rejected. Left camera failure falls back to right. Sessions with no spikes, no usable motion energy, fewer than two valid trials, or any other processing exception are skipped. Cache metadata is rebuilt if absent.

ii.
```python
finite = np.isfinite(st) & np.isfinite(sc)
if len(vv) == 0 or np.any(np.isnan(vv)):
    continue
except Exception as e:
    return eid, None, f'{type(e).__name__}: {e}'
```

iii. The trajectory reports 15 sessions dropped (14 without usable motion energy and one without usable trials) and frames dropping incomplete data as consistent with reference alignment behavior.

## 10-a. What are the most time-consuming steps of the code?

i. Walking the cache during table rebuilding, loading/merging large spike-sorting arrays, per-trial spike binning and behavior interpolation, serializing per-session NPZ files, assembling/writing the 106-GB pickle, and subsequent decoder loading/training are expensive. Session conversion is parallelized over 16 workers.

ii.
```python
for f in (d / 'alf').rglob('*'):
sp, cl, ch = ssl.load_spike_sorting()
with mp.Pool(args.n_workers, maxtasksperchild=4) as pool:
```

iii. The trajectory spent substantial time converting the full release and then loading/training on the 106-GB output; it identified the very large all-cluster representation as the main scale driver.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial behavior interpolation, per-trial spike binning, the Python `trial_in_block` loop, per-row probe loading, file-tree walking, and assembly loops are candidates. The trial loops are most relevant, though ragged source slices complicate full vectorization.

ii.
```python
for k in range(ntrials):
    values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
for j, k in enumerate(kidx):
    np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)
```

iii. The trajectory prioritized correctness and parallelization across sessions; it did not claim these inner loops were fully optimized.

## 10-c. What processing does the code repeat multiple times?

i. A ONE client and `BrainRegions` object are created per session; trial-wise interpolation constructs a new `interp1d` object for wheel and camera traces; search boundaries and arrays are processed separately for each behavior; session arrays are written to NPZ and later read back into the final pickle; subject/region membership uses repeated list scans.

ii.
```python
one = get_one()
brainreg = BrainRegions()
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
z = np.load(TMP_DIR / f'{eid}.npz', allow_pickle=True)
```

iii. The trajectory justifies per-worker client creation for multiprocessing and uses intermediate files to make large parallel conversion/assembly manageable; it gives no separate justification for repeated interpolator construction or linear list lookups.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Cache reconstruction computes IDs, hashes, QC categoricals, sizes, and metadata not used by conversion logic. `merge_clusters` attaches full cluster/channel metadata although only acronyms are used. The conversion also retains roughly 600,000 all-quality clusters, greatly enlarging the dataset, while downstream decoder dimensionality reduction discards most dimensions; temporary NPZ serialization is an extra pass. Camera fallback candidates may be processed and discarded if unusable.

ii.
```python
ds['id'] = [str(uuid.uuid5(ns, e + '/' + p)) for e, p in zip(ds.eid, ds.rel_path)]
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
np.savez(f, neural=binned, input=inputs, output=outputs, regions=beryl)
```

iii. Cache reconstruction was a pragmatic offline workaround, probe metadata supported correct merging/anatomy, and temporary files supported parallel assembly. The trajectory acknowledged that keeping all clusters produced a 106-GB dataset and very slow training, but retained them to follow its reading of the methods code.
