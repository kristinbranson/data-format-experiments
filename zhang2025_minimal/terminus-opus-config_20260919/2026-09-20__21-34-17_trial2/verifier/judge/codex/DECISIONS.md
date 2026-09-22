# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent creates a ONE client for the local cache, reads the vendored `bwm_release.csv`, preserves first-seen session order, optionally restricts to `DATALIMIT_SUBSET.csv`, then processes every selected EID. Trials/behavior are loaded through the reference utilities and spike sorting through `SpikeSortingLoader`, once per probe.

ii.
```python
one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
          cache_dir='/app/data/one_cache')
bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
eids = list(dict.fromkeys(bwm_df.eid.tolist()))
...
for k, eid in enumerate(eids):
    sess = load_session(one, eid, args.n_workers)
```

iii. The trajectory says the release CSV contains 459 sessions/139 mice and that the ONE loaders reproduce the vendored methods pipeline. It chose the release freeze as the authoritative session list and used the optional subset CSV when present.

## 1-b. How are the data split into subjects?

i. Subject labels are taken from `bwm_release.csv`. A subject is appended on first encounter, and every retained session receives the corresponding index.

ii.
```python
eid2subject = dict(zip(bwm_df.eid, bwm_df.subject))
...
sub = eid2subject[eid]
if sub not in subjects:
    subjects.append(sub)
data['subject_idx'].append(subjects.index(sub))
```

iii. The agent treated the release table’s subject field as the unique mouse ID; no filename parsing was needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` is a session. `load_session` returns one session’s nested trial lists, which are appended once to the output.

ii.
```python
eids = list(dict.fromkeys(bwm_df.eid.tolist()))
...
data['neural'].append(sess['neural'])
data['input'].append(sess['input'])
data['output'].append(sess['output'])
```

iii. The trajectory identifies EIDs as the dataset’s native session boundaries and follows the BWM release order.

## 1-d. How are the data split into trials?

i. The trials table supplies one row per trial. Reference binning utilities produce one spike and behavior array per row; retained row indices (`keep`) select the final trials.

ii.
```python
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
...
keep = np.flatnonzero(good)
spk = binned_spikes[keep]
```

iii. The agent regarded the trials table as already defining trial boundaries and used the reference utility to align/bucket continuous streams around each row’s event.

## 1-e. How are trials filtered based on quality controls?

i. It uses `load_trials_and_mask(max_trial_len=10.0)`: reaction time in `[0.08, 2]` s, trial duration at most 10 s, no required-field NaNs, and no-choice trials removed. It additionally drops trials whose wheel or whisker arrays are missing, wrong-length, or non-finite, and drops sessions with fewer than two survivors.

ii.
```python
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
good = trials_mask.copy()
for beh in BEH_NAMES:
    ok = np.array([(tr is not None) and (np.asarray(tr).size == NBINS)
                   and bool(np.all(np.isfinite(np.asarray(tr, dtype=float))))
                   for tr in binned_beh[beh]])
    good &= ok
```

iii. The agent explicitly aimed to duplicate the methods-code mask. It justified the extra finite-behavior rule because NaN cannot be assigned a decoder category and the validator rejects non-finite values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each probe’s spike timestamps and spike-to-cluster IDs. Cluster acronyms are used for regions but not spike values.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
neural_dict = {'spike_times': spikes['times'],
               'spike_clusters': spikes['clusters'],
               'cluster_regions': clusters['acronym'].to_numpy()}
```

iii. The trajectory says these are the direct spike-sorting variables used by the reference caching code.

## 2-b. How is the `neural` data processed?

i. Probes are merged, spikes are counted in non-overlapping 20 ms bins, and trial arrays are transposed from time-by-neuron to neuron-by-time. Counts are stored as `float32`; they are not divided by bin width or smoothed.

ii.
```python
spikes, clusters = U.merge_probes(spikes_list, clusters_list)
binned_spikes, clusters_used = U.bin_spiking_data(..., **PARAMS)
...
neural.append(np.ascontiguousarray(spk[i].T, dtype=np.float32))
```

iii. The agent cited the methods implementation’s binned spike counts and “all neurons,” and intentionally omitted smoothing. It merged simultaneous probes because their behavior is shared and they are not independent sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It applies no cluster-quality threshold and retains all clusters returned by spike sorting. It also does not explicitly remove Beryl `void` clusters.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
...
regions, beryl_reg = U.list_brain_regions(neural_dict, **PARAMS)
reg_clu_ids = U.select_brain_regions(neural_dict, beryl_reg, regions[0], **PARAMS)
```

iii. The agent deliberately chose `qc=None`, arguing that the methods repository’s `prepare_data` bins all sorted clusters. It recognized that this differs from the data paper’s stringent-QC population but prioritized the methods code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to `stimOn_times` in a `[-0.5, 1.5]` s window through the reference binning utility.

ii.
```python
PARAMS = {'align_time': 'stimOn_times',
          'time_window': (-0.5, 1.5), ...}
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, **PARAMS)
```

iii. The trajectory found these exact parameters in `0_data_caching.py` and selected them because the task explicitly requires stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, giving 100 bins in two seconds. Raw spikes are binned once; no later temporal rebinning is performed.

ii.
```python
'binsize': 0.02,
NBINS = int(round((1.5 - (-0.5)) / 0.02))
```

iii. The agent used the 20 ms dynamic-behavior configuration in the methods code; it judged 50 ms inappropriate because wheel and whisker are time-varying outputs here.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the configured stimulus-relative window and bin size, with `stimOn_times` providing zero for every trial.

ii.
```python
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
         + PARAMS['time_window'][0]).astype(np.float32)
```

iii. The agent states this is the time grid used by the reference behavior interpolation.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It generates bin-end times from -0.48 through +1.50 seconds and repeats the same grid for every trial.

ii.
```python
inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
```

iii. The trajectory explicitly chose bin ends, believing that the vendored interpolation utility uses this grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Both arrays contain 100 positions under the same `[-0.5,1.5]`/20 ms configuration, but the time input labels positions by bin ends while a spike-count bin represents an interval.

ii.
```python
tgrid = np.arange(1, NBINS + 1) * 0.02 - 0.5
neural.append(np.ascontiguousarray(spk[i].T, dtype=np.float32))
```

iii. The agent considered the behavior interpolation endpoints and spike bins to be the same reference time axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived solely from consecutive values of `trials.probabilityLeft`.

ii.
```python
tinb = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())[keep]
```

iii. Because there is no explicit block ID, the agent inferred block boundaries when `probabilityLeft` changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. It counts from zero, increments while adjacent priors are equal, resets on a change, computes this before filtering, and broadcasts the result across time.

ii.
```python
if i > 0 and np.isclose(p[i], p[i - 1]): counter += 1
else: counter = 0
...
np.full(NBINS, tinb[i], dtype=np.float32)
```

iii. The agent wanted dropped trials still to advance the true behavioral block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`.

ii.
```python
choice = trials_df['choice'].to_numpy()[keep]
```

iii. The agent verified the IBL sign convention in repository code.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `+1` becomes left/0 and `-1` becomes right/1; no-choice trials were already filtered. The scalar is broadcast over 100 bins.

ii.
```python
choice_out = np.where(choice == -1, 1, 0).astype(np.int64)
np.full(NBINS, choice_out[i], dtype=np.int64)
```

iii. This directly implements the requested left=0/right=1 mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
pleft = trials_df['probabilityLeft'].to_numpy()[keep]
```

iii. The agent identifies this as the task’s block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to one decimal and mapped `0.2→0`, `0.5→1`, `0.8→2`, then broadcast across time.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
prior_out = np.array([PRIOR_MAP[round(float(p), 1)] for p in pleft])
```

iii. This is the categorical mapping explicitly required by the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The reference behavior loader obtains wheel timestamps and position and returns absolute filtered wheel velocity as `wheel-speed`.

ii.
```python
BEH_NAMES = ['wheel-speed', 'whisker-motion-energy']
binned_beh, _ = U.bin_behaviors(one, eid, BEH_NAMES, ...)
```

iii. The agent relied on the repository utility to preserve its wheel preprocessing exactly.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Inside the utility, wheel position is uniformly interpolated, filtered/differentiated to velocity, converted to absolute speed, and interpolated to trial timepoints. The resulting retained samples are then discretized by session tertiles.

ii.
```python
wheel = np.stack([np.asarray(binned_beh['wheel-speed'][i], dtype=float).ravel()
                  for i in keep])
wheel_out = discretize_tertiles(wheel)
```

iii. The trajectory says the IBL utility is the recommended/reference transformation and that session-specific tertiles suit the heavy-tailed, session-dependent units.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained wheel samples in a session are pooled. The 1/3 and 2/3 quantiles form low/medium/high thresholds; `searchsorted(..., side='right')` assigns 0/1/2.

ii.
```python
edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
return np.searchsorted(edges, values, side='right').astype(np.int64)
```

iii. The agent chose within-session tertiles to prevent fixed global cutoffs from collapsing sessions with different scales and to yield roughly balanced classes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. `U.bin_behaviors` aligns/interpolates it using the same `stimOn_times`, `[-0.5,1.5]` window, 20 ms setting, and 100 samples as neural data.

ii.
```python
U.bin_behaviors(..., trials_df=trials_df, **PARAMS)
U.bin_spiking_data(..., trials_df=trials_df, **PARAMS)
```

iii. The agent used one shared parameter dictionary specifically to keep modalities synchronized.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses left-camera ROI motion energy when available and otherwise right-camera ROI motion energy, together with that camera’s timestamps.

ii.
```python
('whisker-motion-energy', ['left-whisker-motion-energy',
                           'right-whisker-motion-energy'])
```

iii. This fallback follows the reference utility and allows sessions with either side camera.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Released ROI motion energy is interpolated onto the trial grid without extra filtering/normalization, stacked for retained trials, then discretized by session tertiles.

ii.
```python
whisk = np.stack([np.asarray(binned_beh['whisker-motion-energy'][i], dtype=float).ravel()
                  for i in keep])
whisk_out = discretize_tertiles(whisk)
```

iii. The agent found no reference normalization and retained the released signal; session tertiles handle camera-dependent scale.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the pooled within-session 1/3 and 2/3 quantiles and maps samples to 0/1/2.

ii.
```python
whisk_out = discretize_tertiles(whisk)
```

iii. The same balanced-class argument as wheel speed was used.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera signal is interpolated by `U.bin_behaviors` relative to each trial’s stimulus onset under the same 100-point parameterization as spikes.

ii.
```python
binned_beh, _ = U.bin_behaviors(..., trials_df=trials_df, **PARAMS)
```

iii. The agent states that shared session clocks plus common alignment parameters provide binwise alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing required behavior, absent spike sorting/no probes, no neurons, fewer than two good trials, and any load exception cause the session to be skipped and recorded. Individual behavior trials with missing/non-finite/wrong-length traces are dropped. Notably, absent sorting on one insertion aborts the whole session.

ii.
```python
if clusters is None:
    raise RuntimeError(...)
...
except Exception as exc:
    skipped.append({'eid': eid, 'reason': repr(exc)})
    continue
```

iii. The agent justified skipping because every required decoder stream must exist and NaNs cannot be categorical. It investigated each full-run skip and made failure reasons explicit.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and merging large spike-sorting arrays, binning spikes/behavior across all trials, and finally serializing the 106 GB result dominate. The trajectory reports roughly hours for the full 459-session conversion and a long pickle load during verification.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
binned_spikes, clusters_used = U.bin_spiking_data(..., n_workers=n_workers, **PARAMS)
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent’s runtime observations showed large sessions and spike I/O/bucketing dominate; it used worker parallelism inside reference utilities.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit trial-number loop, finite-trace loop, prior mapping comprehension, per-trial stacking loop, region-index comprehensions, and repeated list membership/index operations could be vectorized or replaced by lookup dictionaries. The reference utilities also contain per-trial interpolation/binning work, though they parallelize it.

ii.
```python
for i in range(len(p)):
...
for i in range(len(keep)):
    neural.append(...)
...
data['subject_idx'].append(subjects.index(sub))
```

iii. The trajectory focused on correctness and found runtime acceptable on the large-memory host; it did not document a dedicated vectorization pass.

## 10-c. What processing does the code repeat multiple times?

i. Behavior availability is loaded once in the precheck and then loaded again inside `U.bin_behaviors`. Static choice/prior/block values and the identical time grid are also allocated anew for every trial, and subject lookup scans the growing list repeatedly.

ii.
```python
if all('skip' in U.load_target_behavior(one, eid, t) for t in targets): ...
...
binned_beh, _ = U.bin_behaviors(one, eid, BEH_NAMES, ...)
```

iii. The precheck was deliberately added to turn an obscure downstream interpolation crash into an informative missing-stream error. Other repetition was accepted to construct the required per-trial format.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `channels` and builds merged cluster tables/region metadata beyond what neural count matrices need, although region output requires part of that work. It also constructs extensive `session_info`, skipped-session diagnostics, sorted per-session region summaries, and progress statistics that the decoder does not use. The precheck loads behavior only to test availability before reloading it.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
...
session_info.append({'lab': eid2lab[eid], ..., 'regions': sorted(set(...))})
```

iii. The agent retained metadata for provenance/debugging and the precheck for clearer error reporting, even though neither contributes to model features or targets.
