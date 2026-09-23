# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the 699-insertion Brain Wide Map freeze table (`bwm_release.csv`), takes its unique session IDs, and uses a local ONE client plus `SessionLoader` and `SpikeSortingLoader`. It processes sessions in separate worker processes. It does not discover sessions from the ONE release index or restrict them using `DATALIMIT_SUBSET.csv`.

ii.
```python
bwm = pd.read_csv(BWM_CSV, index_col=0)
eids = sorted(bwm.eid.unique())
sess_loader = SessionLoader(one=one, eid=eid)
ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
```

iii. The notes say the freeze file is the same session/probe inventory used by the reference pipeline, avoids needing `one.eid2pid`, and contains 459 sessions, 699 insertions, and 139 subjects matching the paper. Separate processes are used because a shared ONE instance was found not to be thread-safe.

## 1-b. How are the data split into subjects?

i. Subject labels come directly from the freeze table. After failed/skipped sessions are removed, unique subjects are sorted and each retained session receives an index into that list.

ii.
```python
eid2subject = bwm.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
subjects = sorted({eid2subject[r['eid']] for r in ok})
subject_idx = np.array([subject_index[eid2subject[r['eid']]] for r in ok])
```

iii. The agent treats the subject field in the official freeze table as the authoritative mouse identifier; its full result contains 136 retained mice.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the freeze table is one session. All insertion rows sharing that `eid` are passed together to one conversion task, and completed results are restored to sorted-EID order.

ii.
```python
tasks = [(eid, bwm[bwm.eid == eid][['pid', 'probe_name']].copy(), ...)
         for i, eid in enumerate(eids)]
results = [by_eid[e] for e in eids if e in by_eid]
```

iii. The notes state that an EID is already the IBL session identifier and probes from the same session share behavior, so they are merged rather than treated as independent sessions.

## 1-d. How are the data split into trials?

i. The agent loads the ALF trials table, whose rows are trials, applies a Boolean mask, and uses each retained row's `stimOn_times` to construct one neural/input/output item.

ii.
```python
trials, mask = load_trials_and_mask(...)
align_times = trials[ALIGN_EVENT].values[mask]
neural=[neural[k] for k in range(n_trials)]
```

iii. It explains that the trials table already defines trial boundaries and that all streams are cut using the retained trial's stimulus onset.

## 1-e. How are trials filtered based on quality controls?

i. The imported reference mask requires reaction time 0.08–2 s, nonmissing key events, feedback no more than 10 s after go cue, and nonzero choice. The agent then drops trials lacking complete finite wheel or motion-energy coverage. Sessions with fewer than two trials are skipped. It does not explicitly validate `probabilityLeft` against the three permitted values before mapping it.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid,
                                    max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
keep = valid_wheel & valid_me
if keep.sum() < MIN_TRIALS_PER_SESSION:
    return dict(eid=eid, skip='fewer than 2 trials with complete behaviour')
```

iii. The notes say using `load_trials_and_mask` verbatim best matches the decoding-paper pipeline, while complete behavior coverage is necessary for the requested time-varying outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from `spikes.times` and `spikes.clusters` for every probe in a session. Cluster `label` and `acronym` fields determine inclusion and anatomical metadata.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
times_list.append(np.asarray(spikes['times'])[sel])
clusters_list.append(remap[np.asarray(spikes['clusters'])[sel]])
```

iii. The agent follows the reference spike loader/merge pattern and uses merged cluster metadata for paper-based unit curation.

## 2-b. How is the `neural` data processed?

i. Kept spikes from all probes are reindexed, concatenated, and time-sorted; spike counts are accumulated in 100 nonoverlapping 20 ms bins. By default, the counts are then standardized independently at each time bin using one scalar mean and standard deviation pooled over all trials and neurons. They are not converted to Hz.

ii.
```python
np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
mu = spikes_binned.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
sd = spikes_binned.std(axis=(0, 1), keepdims=True, dtype=np.float64)
neural = ((spikes_binned.astype(np.float64) - mu) /
          np.where(sd > 0, sd, 1.0)).astype(np.float32)
```

iii. The agent argues that the supplied decoder does not normalize its inputs and that the Zhang decoding pipeline calls `standardize_spike_data`; it chose the exact pooled-per-time-bin transform after testing alternatives. This intentionally differs from storing firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit must have cluster QC `label >= 1` and its Beryl atlas acronym must be neither `root` nor `void`. Sessions without a surviving unit are removed. No minimum-neurons-per-region rule is applied.

ii.
```python
good = (clu['label'].values >= 1)
beryl = np.asarray(br.acronym2acronym(clu['acronym'].to_numpy(), mapping='Beryl'))
keep = good & ~np.isin(beryl, NON_GREY)
```

iii. The notes cite the data paper's 75,708 well-isolated-neuron statistic and grey-matter restriction. They reject region-level minimums because the downstream decoder pools the session population.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial uses `stimOn_times` as time zero and includes spikes in `[stimOn-0.5, stimOn+1.5)`. Absolute spike timestamps and trial events are assumed to share the synchronized session clock.

ii.
```python
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
i0 = np.searchsorted(spike_times, begs, side='left')
bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
```

iii. The notes identify stimulus onset and this two-second window as both instruction-required and the reference pipeline configuration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms: 100 bins across two seconds. Raw spike events are binned once; there is no subsequent temporal rebinning or smoothing.

ii.
```python
BIN_SIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
```

iii. The agent chose the runnable reference code and methods-paper main-text value of 20 ms despite a contradictory 50 ms sentence elsewhere.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the fixed analysis window and bin indices relative to each trial's `stimOn_times`, rather than from an additional raw signal. The agent also adds a separate binary stimulus-onset input not requested as a distinct input.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
align_times = trials[ALIGN_EVENT].values[mask]
```

iii. The notes say bin-center time is the natural continuous time-from-event coordinate, while the extra indicator was motivated by the general instruction that an onset time should be represented as a binary series.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The centers of the 100 bins are computed, from -0.49 through 1.49 seconds, cast to float32, and copied as the first row for every trial.

ii.
```python
time_axis = BIN_CENTRES.astype(np.float32)
arr[0] = time_axis
```

iii. No raw-data transformation is needed beyond constructing the common bin-center coordinate.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `k` is the center of neural bin `k`; both use the same -0.5 s start and 20 ms spacing around stimulus onset.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
```

iii. The agent regards the time vector as a direct description of the neural bin grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the unfiltered `trials.probabilityLeft` sequence; a value change marks the start of a block.

ii.
```python
tnb_all = trial_number_in_block(trials['probabilityLeft'].values)
new_block[1:] = pl[1:] != pl[:-1]
```

iii. The trials table has no direct block-number field, so the agent reconstructs blocks from the task prior.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code computes a zero-based distance from the most recent prior change before applying trial filters, retains the matching values after masking, and broadcasts each scalar across all 100 bins.

ii.
```python
block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(pl)), 0))
return np.arange(len(pl)) - block_start
arr[2] = tnb[k]
```

iii. Computing before masking preserves the animal's true position in the experimental block even when intervening trials are excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice` after the reference mask.

ii.
```python
choice_raw = trials['choice'].values[mask]
```

iii. The notes empirically checked all sessions to establish the ALF sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Positive (`+1`) is mapped to left/0 and all other retained values (expected `-1`) to right/1, then broadcast across time.

ii.
```python
choice = np.where(choice_raw > 0, 0, 1).astype(np.int8)
arr[0] = choice[k]
```

iii. The agent reports that correct left-stimulus trials consistently use `+1`, correct right-stimulus trials use `-1`, and no-choice trials have already been removed.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials.probabilityLeft` on retained trials.

ii.
```python
prob_left_raw = trials['probabilityLeft'].values[mask]
```

iii. This is the raw task variable that is held constant within each block.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to four decimals, mapped `0.2→0`, `0.5→1`, and `0.8→2`, cast to int8, and broadcast across time.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_map[round(float(p), 4)] for p in prob_left_raw], dtype=np.int8)
arr[1] = prior[k]
```

iii. This mapping is specified explicitly by the task; rounding protects dictionary lookup from floating-point representation noise.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses absolute `SessionLoader` wheel velocity, itself derived from raw wheel position and timestamps.

ii.
```python
sess_loader.load_wheel()
wheel_speed_raw, valid_wheel = interp_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
```

iii. The notes state that absolute velocity matches the reference definition of wheel speed and `SessionLoader` performs the recommended interpolation/filtering of raw wheel position.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The absolute velocity trace is sorted, nonfinite samples are removed, it is linearly interpolated at 100 query times per trial, and the resulting retained-session samples are discretized by session-wide quantiles.

ii.
```python
finite = np.isfinite(beh_times) & np.isfinite(beh_values)
f = interp1d(beh_times, beh_values, kind='linear', bounds_error=False,
             fill_value=(beh_values[0], beh_values[-1]))
vals[valid] = f(grid[valid])
```

iii. The agent says linear interpolation mirrors `get_behavior_per_interval`; per-session processing accommodates differing activity scales.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Thresholds are the 1/3 and 2/3 quantiles over every retained trial/time sample in that session. `searchsorted(..., side='right')` assigns classes 0/1/2; duplicate thresholds are nudged upward.

ii.
```python
edges = np.quantile(flat, [1/3, 2/3])
labels = np.searchsorted(edges, flat, side='right').astype(np.int8)
```

iii. The notes justify session terciles as balanced, scale-invariant low/medium/high categories suitable for balanced-accuracy evaluation.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is interpolated relative to the same `stimOn_times`, but at each neural bin's right edge (-0.48, -0.46, ..., 1.50 s), not at the neural bin centers.

ii.
```python
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 1.0)
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
```

iii. The agent claims right-edge evaluation matches the original `get_behavior_per_interval` use of `np.linspace(t_beg + binsize, t_end, n_bins)`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and `times` from the left-camera ROI motion-energy stream, falling back to the right camera if loading the left fails.

ii.
```python
for view in ('left', 'right'):
    sess_loader.load_motion_energy(views=[view])
    me = sess_loader.motion_energy[view + 'Camera']
```

iii. The notes say this left-first fallback exactly follows the reference behavior loader and retains sessions where only right-camera motion energy exists.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Nonfinite source samples are removed, timestamps are sorted, the released motion-energy trace is linearly interpolated at the trial grid, and session-wide terciles are applied. No extra filtering or normalization is used.

ii.
```python
me_raw, valid_me = interp_behavior(me['times'].to_numpy(),
                                   me['whiskerMotionEnergy'].to_numpy(), align_times)
me_lab, me_edges = discretize(me_raw)
```

iii. The released trace is considered already processed; interpolation and categorization are the only additional operations needed for the decoder format.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As for wheel speed, the code uses the session's 33.3rd and 66.7th percentiles over all retained trial/time samples, yielding int8 classes 0/1/2.

ii.
```python
me_lab, me_edges = discretize(me_raw)
edges = np.quantile(flat, qs)
```

iii. The agent emphasizes that motion-energy units depend on camera, lighting, and ROI, so within-session quantiles avoid inappropriate global thresholds and balance classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera samples share the session clock and are interpolated relative to the retained trial's stimulus onset, but at bin right edges rather than neural bin centers.

ii.
```python
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
vals[valid] = f(grid[valid])
```

iii. The agent uses the same behavior-grid implementation for wheel and whisker streams and attributes it to the original reference utility.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Nonfinite behavior samples are removed; trials without full finite wheel and camera coverage are dropped; left-camera failures fall back to right; empty probes are skipped; sessions with fewer than two usable trials or no kept neurons are skipped. Worker exceptions become recorded skip results rather than terminating conversion.

ii.
```python
finite = np.isfinite(beh_times) & np.isfinite(beh_values)
keep = valid_wheel & valid_me
except Exception as exc:
    return dict(eid=eid, skip='exception: %s' % exc,
                traceback=traceback.format_exc())
```

iii. The notes favor dropping unusable data rather than imputing it, using camera fallback where an equivalent released stream exists, and preserving skip reasons in metadata for auditability.

## 10-a. What are the most time-consuming steps of the code?

i. Per-session timings are collected for trial loading, wheel loading/interpolation, motion-energy loading/interpolation, spike loading, and spike binning. The documentation identifies spike-sorting I/O as the dominant conversion cost; full conversion is parallelized by session.

ii.
```python
timing['spikes'] = time.time() - t0
with ctx.Pool(processes=min(args.n_workers, len(tasks))) as pool:
    for i, res in enumerate(pool.imap_unordered(_worker, tasks)):
```

iii. Spike arrays are much larger than behavioral tables/traces. Process parallelism was chosen both for speed and because ONE was observed to be unsafe when shared across threads.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning still loops over trials, input/output assembly loops over trials, probe loading loops over insertions, and prior mapping uses a Python comprehension. Behavior interpolation is already vectorized across valid trials. The trial loops could be replaced by flattened trial-offset indices or stacked broadcasts, though I/O dominates.

ii.
```python
for k in range(n_trials):
    np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
for k in range(n_trials):
    arr = np.empty((3, N_BINS), dtype=np.float32)
```

iii. The agent describes spike binning as a vectorized equivalent of the reference at the within-trial level, prioritizing exactness and manageable memory over eliminating the outer trial loop.

## 10-c. What processing does the code repeat multiple times?

i. `interp_behavior` independently sorts, filters, builds grids, and interpolates wheel and motion energy. Per-trial construction repeatedly copies identical time and stimulus-indicator vectors and broadcasts scalar outputs. Each worker also constructs its own ONE and atlas objects by design.

ii.
```python
wheel_speed_raw, valid_wheel = interp_behavior(...)
me_raw, valid_me = interp_behavior(...)
arr[0] = time_axis
arr[1] = stim_indicator
```

iii. The repetition keeps the two behavioral streams under identical rules; separate worker clients are intentional because sharing ONE caused silent failures.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Diagnostic counts, timing, quantile edges, mean firing rate, and optional plots are computed but not used by the decoder (most are retained only in metadata). The extra `stim_onset` input duplicates information already represented by time since onset. With plotting enabled, raw example counts and figures are produced solely for validation. Cluster channel metadata are loaded/merged mainly to obtain QC/anatomy.

ii.
```python
raw_counts_example = spikes_binned[0].copy() if show_processing else None
stim_indicator[STIM_ONSET_BIN] = 1.0
return dict(... wheel_edges=wheel_edges, me_edges=me_edges,
            mean_rate=float(spikes_binned.mean() / BIN_SIZE), timing=timing)
```

iii. The agent intentionally retains diagnostics and the optional visual pipeline as sanity checks and provenance. It added the onset indicator based on its reading of the format guidance, although the requested continuous time input already encodes onset.
