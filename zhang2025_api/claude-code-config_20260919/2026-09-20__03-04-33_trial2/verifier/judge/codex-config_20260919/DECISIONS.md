# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the 459-session BWM freeze list from `bwm_release.csv`, creates a ONE client pointed at the staged cache, and uses `SpikeSortingLoader` once per probe plus `SessionLoader` for trials, wheel, and camera motion energy. Sessions are processed in parallel and failures are omitted.

ii.
```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(pd.unique(bwm_df.eid))
...
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
sp, cl, ch = ssl.load_spike_sorting()
...
sess_loader = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(one=one, eid=eid,
    max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
```

iii. The notes say the freeze CSV is the session list used by the method-paper code and matches 459 sessions, 699 insertions, and 139 mice. Remote-mode ONE was selected because cached Alyx metadata resolves revised dataset paths that local release tables did not.

## 1-b. How are the data split into subjects?

i. Subject labels come from the freeze CSV. Converted-session subjects are sorted, and each session receives an integer index into that list.

ii.
```python
subject_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
subjects = sorted({subject_of_eid[res['eid']] for res in results})
subject_index = {s: i for i, s in enumerate(subjects)}
```

iii. The agent states that the CSV provides the unique subject identity, so no filename/path inference is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in release-file order is one session and is converted independently; results are restored to that deterministic order after parallel processing.

ii.
```python
eids = list(pd.unique(bwm_df.eid))
...
results.sort(key=lambda r: order[r['eid']])
```

iii. The release freeze already defines sessions by `eid`; the notes explicitly choose its order for reproducibility.

## 1-d. How are the data split into trials?

i. `load_trials_and_mask` returns a trials table with one row per trial. Each retained row's stimulus onset defines a two-second neural/behavior window; arrays are finally converted to lists of per-trial matrices.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy()
binned = bin_spiking_data_fast(..., align_times[keep_trials])
...
data['neural'].append([np.ascontiguousarray(neural[k], dtype=np.float32)
                       for k in range(neural.shape[0])])
```

iii. The trials table already supplies the trial boundaries/events; the chosen alignment/window reproduces the reference parameters.

## 1-e. How are trials filtered based on quality controls?

i. The imported reference mask rejects missing required trial fields, reaction times outside 0.08–2 s, no-choice trials, and trials longer than 10 s. The agent additionally requires complete non-NaN wheel and whisker windows and rejects windows outside ephys coverage or intersecting a pooled-spike gap longer than 0.5 s.

ii.
```python
trials, mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
...
has_neural = neural_data_available(all_spike_times, align_times)
keep_trials = mask & wheel_ok & me_ok & has_neural
```

iii. The imported mask was chosen to prevent drift from the method code. NaN behavior cannot be represented as a categorical label, and the extra ephys mask was intended to eliminate all-zero windows after recording stops or during apparent acquisition dropouts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from merged `spikes.times` and `spikes.clusters` arrays for every probe. Cluster `label` and `acronym` determine neuron inclusion and region metadata.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
...
spike_times = np.ascontiguousarray(spikes['times'][sel])
spike_clusters = np.ascontiguousarray(sc[sel])
```

iii. The agent follows the reference spike loaders and merges probes because probes in one session observe the same behavior and constitute one population.

## 2-b. How is the `neural` data processed?

i. Probes are merged, retained clusters are renumbered, and spikes are counted into 100 non-overlapping 20 ms bins from −0.5 to +1.5 s. Counts are stored directly (converted from worker-side `uint8` to output `float32`), not divided by bin width or smoothed.

ii.
```python
b = np.floor((t - beg[k]) / BINSIZE).astype(np.int64)
counts = np.bincount(c[ok] * NBINS + b[ok], minlength=n_clusters * NBINS)
out[k] = counts.reshape(n_clusters, NBINS).astype(np.uint8)
...
np.ascontiguousarray(neural[k], dtype=np.float32)
```

iii. The notes say the method pipeline caches raw counts and standardizes only inside its model; therefore raw counts were judged the most faithful stored representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` and Beryl regions other than `root` or `void` are retained. A session must retain at least five such neurons.

ii.
```python
good = clusters['label'].to_numpy() >= QC_LABEL
grey = ~np.isin(beryl, NON_GREY_MATTER)
keep = np.nonzero(good & grey)[0]
...
if n_neurons < MIN_NEURONS:
    raise RuntimeError(...)
```

iii. The agent interprets `label == 1` as the data paper's well-isolated-neuron criterion and `root`/`void` exclusion as its grey-matter restriction. The five-neuron minimum is attributed to the data paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each retained trial, absolute spike times are windowed relative to `trials.stimOn_times` over [−0.5, +1.5) seconds.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
beg = align_times + TIME_WINDOW[0]
end = align_times + TIME_WINDOW[1]
```

iii. This exactly matches both the task's stimulus-onset requirement and the reference caching parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms: 100 bins across two seconds. Spikes are binned once; no smoothing, interpolation, or later temporal rebinning is applied to neural data.

ii.
```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The released caching code and model description use 20 ms and T=100; the agent explains that a paper mention of 50 ms refers to a different analysis.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the chosen `stimOn_times` alignment and fixed neural-bin grid, rather than another measured stream.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
align_times = trials[ALIGN_TIME].to_numpy()
```

iii. The input is defined as signed relative time on the reference trial window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The center of every 20 ms neural bin is computed, yielding −0.49 through +1.49 s, and the same vector is broadcast to every trial.

ii.
```python
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. Bin centers were selected to represent the time associated with each neural count.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Its entries are the centers of the exact bins used to count neural spikes, so input column `t` corresponds to neural column `t`.

ii.
```python
b = np.floor((t - beg[k]) / BINSIZE).astype(np.int64)
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. The shared fixed grid provides direct bin-for-bin alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the complete trials table's `probabilityLeft`; a change in probability starts a new block.

ii.
```python
in_block, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. No explicit block identifier exists, while the prior is constant within each experimental block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent detects changes, computes a zero-based running index since the latest change on the unfiltered table, then selects retained trials and broadcasts each number over time.

ii.
```python
new_block[1:] = ~(p[1:] == p[:-1])
block_id = np.cumsum(new_block) - 1
within = np.arange(len(p)) - starts[block_id]
...
inp[:, 1, :] = in_block[keep_trials][:, None]
```

iii. Computing before filtering preserves the animal's actual position even when intervening trials are later removed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice` after the trial mask has removed zero/no-response values.

ii.
```python
choice = tr['choice'].to_numpy()
```

iii. The agent empirically checked the IBL sign convention against correct lateralized trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `+1` (left) becomes 0 and `−1` (right) becomes 1, then the per-trial value is broadcast over 100 timepoints.

ii.
```python
choice_out = (choice < 0).astype(np.int64)
out[:, 0, :] = choice_out[:, None]
```

iii. This implements the requested left=0/right=1 mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from retained rows of `trials.probabilityLeft`.

ii.
```python
pleft = tr['probabilityLeft'].to_numpy()
```

iii. This field is the experimental block prior requested by the task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to classes 0, 1, and 2 and broadcast across time; unexpected values raise an error.

ii.
```python
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_out[np.isclose(pleft, value)] = code
out[:, 1, :] = prior_out[:, None]
```

iii. This is the exact mapping prescribed by the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is the absolute value of `SessionLoader`'s wheel velocity, paired with its timestamps. The loader derives velocity from raw wheel position/timestamps.

ii.
```python
sess_loader.load_wheel()
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. This reproduces the reference `wheel-speed = abs(velocity)` target and uses the loader's standard interpolation/filtering.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The absolute velocity trace is sliced per trial, rejected for gaps/NaNs, linearly interpolated/extrapolated onto 100 bin right edges, and discretized with session-level tertiles.

ii.
```python
x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
...
wheel_lab, wheel_thr = discretize_tertiles(wheel_kept)
```

iii. The agent says right-edge interpolation exactly follows `get_behavior_per_interval`; per-session thresholds make differently scaled sessions comparable.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Thresholds are the 1/3 and 2/3 quantiles of all finite retained trial-time samples within a session. `searchsorted(..., side='right')` assigns low/medium/high labels 0/1/2.

ii.
```python
thresholds = np.quantile(finite, [1. / 3., 2. / 3.])
labels = np.searchsorted(thresholds, values, side='right').astype(np.int64)
```

iii. Session-specific wheel vigor differs, so the agent chose balanced session-level classes rather than a global physical threshold.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Both use the same stimulus onset and 20 ms column count, but wheel samples represent bin right edges (−0.48…+1.50 s), whereas neural counts and the time input are represented by bin centers (−0.49…+1.49 s).

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 1)
```

iii. The agent deliberately follows the method code's behavioral resampling convention, treating each right-edge value as the behavior associated with that neural interval.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses the camera `whiskerMotionEnergy` values and frame times, preferring the left camera and falling back to the right.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
    me = sess_loader.motion_energy[key]
```

iii. This matches the reference fallback and uses the released whisker-pad ROI motion-energy signal.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is not otherwise filtered or normalized. It is coverage/NaN checked, interpolated/extrapolated at bin right edges, and categorized with session-level tertiles.

ii.
```python
me_vals, me_ok = bin_behaviour_per_trial(
    *beh_traces['whisker-motion-energy'], align_times)
me_lab, me_thr = discretize_tertiles(me_kept)
```

iii. The agent follows the method behavior interpolation; session tertiles address uncalibrated camera-, illumination-, and ROI-dependent units.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As for wheel speed, retained samples within each session are split at their 33.3rd and 66.7th percentiles into 0/1/2.

ii.
```python
thresholds = np.quantile(finite, [1. / 3., 2. / 3.])
labels = np.searchsorted(thresholds, values, side='right').astype(np.int64)
```

iii. Per-session tertiles avoid arbitrary cross-camera intensity scaling and yield balanced classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It shares the stimulus onset, two-second window, and 100 output columns, but is evaluated at each neural bin's right edge rather than its center.

ii.
```python
x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

iii. The agent chose the exact right-edge grid used by the method code's `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing probe sorting is skipped; a session failure is logged without aborting the full run. Trials with required-field NaNs, incomplete/NaN behavioral traces, or inferred missing ephys are removed. Sessions without motion energy, with fewer than five retained neurons, or fewer than two usable trials are omitted.

ii.
```python
if sp is None or 'times' not in sp or len(sp['times']) == 0:
    continue
...
if np.isnan(tv).any():
    continue
...
return {'eid': str(eid), 'error': f'{type(exc).__name__}: {exc}', ...}
```

iii. The rationale is to produce valid finite categorical targets and avoid meaningless all-zero neural trials while preserving the full conversion when isolated sessions are defective.

## 10-a. What are the most time-consuming steps of the code?

i. Loading hundreds of megabytes of spike arrays per probe is identified as the dominant cost; serialization of the roughly 12 GB final pickle is another material full-run cost.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes report that disk I/O dominates and therefore parallelize at session level.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning and behavior resampling still loop over trials. Search bounds are vectorized, but each differently located slice is handled separately; the final conversion also loops over sessions/trials to meet the nested-list format.

ii.
```python
for k in range(len(align_times)):
    ...
    counts = np.bincount(...)
...
for k in range(n):
    ...
    vals[k] = interp1d(...)(x)
```

iii. The agent describes these as vectorized replacements for the reference's heavier multiprocessing-per-trial approach; fully batching ragged slices was not considered worth the complexity.

## 10-c. What processing does the code repeat multiple times?

i. Per-session loading, filtering, binning, threshold calculation, and metadata collection repeat for every session; interpolation/coverage logic repeats for wheel and whisker streams, and cluster loading/merging repeats for each probe.

ii.
```python
for pid, pname in zip(pids, pnames):
    ...
wheel_vals, wheel_ok = bin_behaviour_per_trial(...)
me_vals, me_ok = bin_behaviour_per_trial(...)
```

iii. These repetitions correspond to independent probes/sessions and the two required signals. Shared helper functions keep the rules identical.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `load_spike_sorting()` loads more spike attributes than the final neural matrix requires, merged cluster tables carry metadata beyond label/acronym, and extensive timing/QC/session metadata is computed though the decoder mainly consumes neural/input/output arrays. Diagnostic plots and debug payloads occur only when requested. Conversely, the agent deliberately avoids the reference's unused AP sampling-rate query and unused behavior streams.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
...
timing['load_spikes'] = time.time() - t
```

iii. The notes explicitly identify raw-AP streaming and four unused behavior loads in the reference as waste and omit them; remaining extra metadata supports validation and provenance.
