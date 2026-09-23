# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates sessions from the frozen release table `bwm_release.csv`, then loads each session's probes, trials, wheel, and motion-energy data through `ONE`, `SessionLoader`, and `SpikeSortingLoader`. It does not use `one.search(...)` to discover eligible sessions; instead it assumes the release CSV is the session list and lets failed sessions drop out later.

ii. ```python
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
...
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(pd.unique(bwm_df.eid))
```

```python
pids, pnames = one.eid2pid(eid)
...
sess_loader = SessionLoader(one=one, eid=eid)
...
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
```

iii. In `CONVERSION_NOTES.md`, the agent says it chose `bwm_release.csv` because the reference code uses that freeze file, while all actual data loading still goes through the ONE/brainbox APIs. It also documents a separate decision to use ONE's default remote mode against the staged REST cache because `mode='local'` returned incomplete trial tables.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the release CSV, not from `one.search(details=True)`. The agent builds a `subject_of_eid` mapping from `bwm_release.csv`, then creates `subjects` as sorted unique names and `subject_idx` from that mapping during assembly.

ii. ```python
subject_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
...
subjects = sorted({subject_of_eid[res['eid']] for res in results})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_index[subject_of_eid[r['eid']]] for r in results],
                        dtype=np.int64),
```

iii. The justification in the notes is that `bwm_release.csv` is the reference freeze file and already contains the session-to-subject mapping, so there is no need to derive subjects from paths or filenames.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`. Each `eid` is processed independently in `convert_session`, and results are sorted back into release-file order at the end.

ii. ```python
eids = list(pd.unique(bwm_df.eid))
...
def convert_session(eid, verbose=True, collect_debug=False):
    ...
```

```python
order = {e: i for i, e in enumerate(eids)}
results.sort(key=lambda r: order[r['eid']])
```

iii. The notes justify this as following the reference release freeze exactly and making session order deterministic.

## 1-d. How are the data split into trials?

i. Trials come from the session trials table loaded through `SessionLoader`; one row is treated as one trial. The imported `load_trials_and_mask(...)` returns the trials table plus a trial mask.

ii. ```python
sess_loader = SessionLoader(one=one, eid=eid)
...
trials, mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
```

iii. The agent's notes describe this as directly reusing the reference trial-loading helper instead of reimplementing the trial split.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials that pass the imported `load_trials_and_mask(...)`, have usable wheel and whisker traces over the full 2 s window, and also have full electrophysiology coverage with no long spike-train gaps. It further requires each session to retain at least two usable trials.

ii. ```python
trials, mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
...
wheel_vals, wheel_ok = bin_behaviour_per_trial(*beh_traces['wheel-speed'], align_times)
me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
...
has_neural = neural_data_available(all_spike_times, align_times)
keep_trials = mask & wheel_ok & me_ok & has_neural
...
if n_keep < MIN_TRIALS:
    raise RuntimeError(f'only {n_keep} usable trials')
```

iii. In `CONVERSION_NOTES.md`, the agent says it reused the reference `load_trials_and_mask(max_trial_len=10.0)`, but deliberately added behaviour-NaN rejection and an extra `neural_data_available` filter after finding "all neural data is zero" warnings in validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is ultimately derived from `spikes['times']` and `spikes['clusters']`, after loading and merging probes. Cluster metadata (`label`, `acronym`) are used for neuron selection and region labels, but not for the binned spike values themselves.

ii. ```python
sp, cl, ch = ssl.load_spike_sorting()
...
return merge_probes(spikes_list, clusters_list)
```

```python
all_spike_times = spikes['times']
...
spike_times = np.ascontiguousarray(spikes['times'][sel])
spike_clusters = np.ascontiguousarray(sc[sel])
```

iii. The notes state that the conversion mirrors the reference probe loading and merge logic, with spike times and cluster ids as the actual neural signal source.

## 2-b. How is the `neural` data processed?

i. The agent pools probes within a session, filters neurons, then bins spikes into 100 non-overlapping 20 ms bins from -0.5 s to +1.5 s around stimulus onset. It stores raw spike counts per bin, not firing rates in Hz.

ii. ```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

```python
b = np.floor((t - beg[k]) / BINSIZE).astype(np.int64)
ok = (b >= 0) & (b < NBINS)
counts = np.bincount(c[ok] * NBINS + b[ok], minlength=n_clusters * NBINS)
...
out[k] = counts.reshape(n_clusters, NBINS).astype(np.uint8)
```

iii. The justification in the notes is explicit: the agent chose raw counts because the reference cached dataset stores counts and model-time normalization happens later. This is a deliberate departure from converting counts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only neurons with `clusters['label'] >= 1` and Beryl acronyms not in `('root', 'void')`. It also rejects sessions with fewer than 5 retained neurons.

ii. ```python
QC_LABEL = 1.0
NON_GREY_MATTER = ('root', 'void')
MIN_NEURONS = 5
```

```python
beryl = np.asarray(brain_regions.acronym2acronym(
    clusters['acronym'].to_numpy(), mapping='Beryl'))
good = clusters['label'].to_numpy() >= QC_LABEL
grey = ~np.isin(beryl, NON_GREY_MATTER)
keep = np.nonzero(good & grey)[0]
...
if n_neurons < MIN_NEURONS:
    raise RuntimeError(f'only {n_neurons} well-isolated grey-matter neurons '
                       f'(< {MIN_NEURONS})')
```

iii. The notes justify this with the data paper's "well-isolated neuron" criterion and a grey-matter restriction, and explain that it deliberately departs from the reference caching code's `qc=None`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each retained trial, the bin window is defined as stimulus onset plus `(-0.5, 1.5)` s, and spikes are binned relative to that per-trial window.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align_times = trials[ALIGN_TIME].to_numpy()
...
beg = align_times + TIME_WINDOW[0]
end = align_times + TIME_WINDOW[1]
```

iii. The notes say this follows both the task instructions and the method-paper reference parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, for 100 bins total. There is no coarser temporal rebinning.

ii. ```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The notes say this reproduces the reference `params` and the method-paper description of 2 s trials with `T = 100`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the chosen alignment event `trials['stimOn_times']` plus a fixed bin grid. The per-trial input does not use a separately loaded raw time series; it is the signed time coordinate implied by the alignment window.

ii. ```python
ALIGN_TIME = 'stimOn_times'
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. The notes describe this as the signed bin-centre time relative to stimulus onset, matching the decoder-input specification.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent computes the 100 bin centres from -0.49 s to +1.49 s and broadcasts the same vector to every retained trial.

ii. ```python
inp = np.empty((n_keep, 2, NBINS), dtype=np.float32)
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. The justification is simply that this variable is defined by the chosen temporal grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same 100-bin trial grid as the neural array, with values at the bin centres for the same stimulus-aligned window.

ii. ```python
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
...
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. The notes say the time input is meant to be the neural bin grid itself, expressed as signed time.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`, treating each run of constant `probabilityLeft` as a block.

ii. ```python
def trial_number_in_block(probability_left):
    ...
    new_block[1:] = ~(p[1:] == p[:-1])
```

iii. The notes explain that the trials table does not provide an explicit block index, so the block structure is recovered from `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent computes a 0-based counter within each constant-`probabilityLeft` block on the full trials table before any filtering, then broadcasts the retained trials' values across time bins.

ii. ```python
block_id = np.cumsum(new_block) - 1
starts = np.nonzero(new_block)[0]
within = np.arange(len(p)) - starts[block_id]
return within.astype(np.float32), block_id
```

```python
in_block, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
inp[:, 1, :] = in_block[keep_trials][:, None]
```

iii. The notes justify computing this before trial exclusion so the count reflects the true block history the mouse experienced.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials['choice']`.

ii. ```python
choice = tr['choice'].to_numpy()
```

iii. The notes say the agent empirically verified the sign convention and then recoded it to the requested left/right labels.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps leftward choices to 0 and rightward choices to 1 using `(choice < 0)`, after the earlier trial mask has already removed no-response trials.

ii. ```python
# on correct trials with the stimulus on the left, choice == +1; on correct trials with
# the stimulus on the right, choice == -1. So +1 is a leftward choice.
choice_out = (choice < 0).astype(np.int64)          # left -> 0, right -> 1
```

iii. `CONVERSION_NOTES.md` says this mapping was checked empirically against left- and right-stimulus correct trials.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials['probabilityLeft']`.

ii. ```python
pleft = tr['probabilityLeft'].to_numpy()
```

iii. The notes describe `probabilityLeft` as the block prior in the IBL task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent recodes `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, and errors out on any unexpected value.

ii. ```python
prior_out = np.full(n_keep, -1, dtype=np.int64)
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_out[np.isclose(pleft, value)] = code
if np.any(prior_out < 0):
    bad = np.unique(pleft[prior_out < 0])
    raise RuntimeError(f'unexpected probabilityLeft values {bad}')
```

iii. The notes say this mapping follows the task specification directly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed comes from the wheel trace loaded by `SessionLoader.load_wheel()`, specifically the derived wheel `velocity` and its timestamps. In raw terms, this is based on wheel position/timestamps processed by `SessionLoader`.

ii. ```python
sess_loader.load_wheel()
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The notes say this reproduces the reference `load_target_behavior` logic: wheel speed is the absolute value of wheel velocity returned by `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent takes the absolute value of wheel velocity, resamples it within each trial onto the decoder bins using linear interpolation, then discretizes the retained session-wide values into three equal-occupancy categories.

ii. ```python
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
...
wheel_vals, wheel_ok = bin_behaviour_per_trial(*beh_traces['wheel-speed'], align_times)
...
wheel_lab, wheel_thr = discretize_tertiles(wheel_kept)
```

iii. The notes justify this by reference to `SessionLoader`'s wheel preprocessing and by arguing that per-session tertiles make the discrete target comparable across sessions.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded with per-session tertiles computed over all retained wheel-speed samples of that session, giving labels `0/1/2` for low/medium/high.

ii. ```python
def discretize_tertiles(values):
    finite = values[np.isfinite(values)]
    thresholds = np.quantile(finite, [1. / 3., 2. / 3.])
    labels = np.searchsorted(thresholds, values, side='right').astype(np.int64)
    return labels, thresholds
```

iii. The notes explicitly justify per-session tertiles because wheel-speed scale depends on the particular session and mouse.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is aligned per trial to the same stimulus-onset window, but the agent samples it at the right edge of each 20 ms bin rather than at the bin centre.

ii. ```python
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 1)
```

```python
x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

iii. The agent justifies this in the code comments and notes by saying it is following the reference helper `get_behavior_per_interval`, which samples behaviour at bin right edges.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the camera motion-energy table loaded by `SessionLoader.load_motion_energy(...)`: the `whiskerMotionEnergy` values and frame times, preferring the left camera and falling back to the right camera.

ii. ```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    ...
    me = sess_loader.motion_energy[key]
    times = me['times'].to_numpy()
    vals = me['whiskerMotionEnergy'].to_numpy()
```

iii. The notes say this follows the reference behaviour-loading logic and uses right-camera fallback only when left-camera data are unavailable.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released whisker motion-energy trace is used as-is, linearly interpolated within each trial onto the decoder bins, then discretized into per-session tertiles.

ii. ```python
me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
...
me_lab, me_thr = discretize_tertiles(me_kept)
```

iii. The notes justify avoiding extra filtering/normalization and again argue for session-specific tertiles because the motion-energy scale depends on camera and session specifics.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same per-session tertile procedure as wheel speed, producing low/medium/high labels `0/1/2`.

ii. ```python
me_lab, me_thr = discretize_tertiles(me_kept)
```

iii. The notes explicitly say both dynamic outputs use per-session tertiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, it is aligned to the same stimulus-onset trial window but sampled at bin right edges rather than centres.

ii. ```python
x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```

iii. The code comments say this is meant to match the reference behaviour helper's convention.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable data are mostly handled by dropping them: probes with no spikes/clusters are skipped, sessions without whisker motion energy fail, trials with invalid masks/behaviour windows/NaNs/no ephys coverage are dropped, sessions with too few neurons or trials are rejected, and worker exceptions are caught so one bad session does not abort the run.

ii. ```python
if sp is None or 'times' not in sp or len(sp['times']) == 0:
    continue
if cl is None or len(cl.get('channels', [])) == 0:
    continue
...
if camera_used is None:
    raise RuntimeError('no whisker motion energy available')
...
if np.isnan(tv).any():
    continue
...
if n_neurons < MIN_NEURONS:
    raise RuntimeError(...)
if n_keep < MIN_TRIALS:
    raise RuntimeError(...)
...
def _worker(eid):
    try:
        return convert_session(eid, verbose=True)
    except Exception as exc:
        return {'eid': str(eid), 'error': f'{type(exc).__name__}: {exc}', ...}
```

iii. The notes justify these guards as necessary to avoid invalid decoder inputs and to remove trials that validation exposed as lacking real electrophysiology despite having behavioural records.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies spike-sorting I/O as the dominant cost: loading `spikes.*.npy` and associated cluster data for each probe. It explicitly optimized around that assumption and only secondarily around binning.

ii. ```python
t = time.time()
spikes, clusters = load_session_spikes(one, eid, pids, pnames)
timing['load_spikes'] = time.time() - t
```

iii. `CONVERSION_NOTES.md` says spike sorting load is the dominant serial cost, whereas trial loading and binning are comparatively small.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining per-trial loops are still in `bin_spiking_data_fast(...)` and `bin_behaviour_per_trial(...)`. The agent partially vectorized them with `searchsorted`, but both still iterate over trials and could be pushed further if needed.

ii. ```python
for k in range(len(align_times)):
    if not finite[k]:
        continue
    t = spike_times[i0[k]:i1[k]]
    ...
```

```python
for k in range(n):
    if not finite[k]:
        continue
    tt = target_times[idx_beg[k]:idx_end[k]]
    ...
```

iii. The notes frame the spike-binning function as a vectorized replacement for the reference code, but these trial loops still remain as the clearest opportunities for further vectorization.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly performs per-trial interpolation of wheel and whisker traces and per-trial spike binning inside every session. It also recomputes small per-session bookkeeping such as thresholds, coverage masks, and metadata fields.

ii. ```python
wheel_vals, wheel_ok = bin_behaviour_per_trial(*beh_traces['wheel-speed'], align_times)
me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
```

```python
binned = bin_spiking_data_fast(spike_times, spike_clusters, n_neurons,
                               align_times[keep_trials])
...
wheel_lab, wheel_thr = discretize_tertiles(wheel_kept)
me_lab, me_thr = discretize_tertiles(me_kept)
```

iii. The notes present this as acceptable repeated per-session/per-trial work after removing the larger redundancies that existed in the reference pipeline.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent removed several heavier unnecessary steps from the reference pipeline, but its own code still computes substantial metadata and bookkeeping that the downstream decoder does not use: timing stats, per-session thresholds, failure logs, lab/date mappings, and optional plotting/debug payloads.

ii. ```python
timing = {}
...
result = dict(
    ...
    wheel_thresholds=wheel_thr,
    me_thresholds=me_thr,
    frac_correct=float((trials['feedbackType'].to_numpy() == 1).mean()),
    timing=timing,
)
```

```python
lab_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['lab'].to_dict()
date_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['date'].to_dict()
...
'failed_sessions': [{'eid': e, 'error': m} for e, m in failures],
```

iii. The notes explicitly say the agent skipped raw-ephys streaming and unused behaviour streams from the reference code, so the remaining extra work is mostly diagnostic/metadata overhead rather than core data processing.
