# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the per-session trial, wheel, motion-energy, and spike-sorting data through `ONE` plus `SessionLoader` and `SpikeSortingLoader`, using `mode='remote'`. However, it does not discover sessions through `one.search()`: it enumerates sessions from `/app/code/code_zhang2025/data/bwm_release.csv` and then converts each `eid`.

ii. 
```python
def get_one():
    from one.api import ONE
    return ONE(base_url=ONE_BASE_URL, mode='remote')
```

```python
bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eid2subject = bwm.groupby('eid').subject.first().to_dict()
eids = list(bwm.eid.unique())
```

```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
...
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. In `CONVERSION_NOTES.md`, the AI says `mode='remote'` is required because the staged cache needs cached REST metadata to resolve revisioned datasets and make `eid2pid` work offline. The trajectory shows it tested `mode='local'`, found revision/path mismatches, then switched to remote cached resolution.

## 1-b. How are the data split into subjects?

i. Subjects are not inferred from the ONE search metadata. The AI reads the subject names from `bwm_release.csv`, builds `eid -> subject`, then forms the final `subjects` list and `subject_idx` during assembly.

ii.
```python
bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eid2subject = bwm.groupby('eid').subject.first().to_dict()
```

```python
subjects = sorted({eid2subject[r['eid']] for r in ok})
subj_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subj_index[eid2subject[r['eid']]] for r in ok], dtype=np.int64),
```

iii. The justification in the notes is pragmatic: `bwm_release.csv` already contains the subject identity for each `eid`, so the AI reused that reference metadata instead of deriving it from the session paths or ONE search results.

## 1-c. How are the data split into sessions?

i. Sessions are treated as one `eid` each. The AI gets the set of sessions by taking the unique `eid` values from `bwm_release.csv`, then processes each session independently in `convert_session(eid)`.

ii.
```python
eids = list(bwm.eid.unique())
...
with mp.Pool(args.n_workers) as pool:
    for i, r in enumerate(pool.imap_unordered(_worker, [(e, False) for e in eids])):
```

```python
def convert_session(eid, show_processing=False, outdir='/app'):
```

iii. The AI’s notes treat the BWM release table as the authoritative session list and describe the conversion as session-parallel processing over those `eid`s.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the IBL trials table loaded by `SessionLoader`; each row is treated as one trial. All later masking and per-trial arrays index into this table.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
```

```python
align_all = trials[ALIGN_TIME].to_numpy()
...
idx_trials = np.where(mask)[0]
```

iii. The AI follows the reference assumption that the trials table is already one row per trial, so no extra parsing is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI first applies a direct reimplementation of the Zhang `load_trials_and_mask` query: it excludes trials with reaction time outside 0.08-2.0 s, trial length above 10 s, NaNs in key fields, and no-choice trials. It then adds three more filters: the full 2 s neural window must lie inside the spike recording, wheel and whisker traces must cover the whole trial window, and any trial with zero spikes across all retained neurons is removed.

ii.
```python
def compute_trials_mask(trials, min_rt=MIN_RT, max_rt=MAX_RT, max_trial_len=MAX_TRIAL_LEN,
                        nan_exclude=NAN_EXCLUDE, exclude_nochoice=True):
    query = f'(firstMovement_times - stimOn_times < {min_rt})'
    query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    if exclude_nochoice:
        query += ' | (choice == 0)'
    return ~trials.eval(query)
```

```python
covered = (align_all + TIME_WINDOW[0] >= rec_t0) & (align_all + TIME_WINDOW[1] <= rec_t1)
mask = mask & covered
...
good = wheel_valid & me_valid
...
nonzero = binned_spikes.sum(axis=(1, 2)) > 0
```

iii. `CONVERSION_NOTES.md` says the base mask is meant to match `load_trials_and_mask`, while the extra ephys-coverage and zero-spike filters were added after verification exposed all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is built from `spikes['times']` and `spikes['clusters']` across all probes in a session. Cluster metadata (`label` and `acronym`) are used only to decide which units to keep and how to label brain regions.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
spike_times_l.append(spikes['times'])
spike_clu_l.append(spikes['clusters'] + offset)
acronyms_l.append(clu['acronym'].to_numpy())
labels_l.append(clu['label'].to_numpy())
```

iii. The notes describe this as matching the reference probe-merging pipeline, with `label` and Beryl acronyms only supplying curation and region metadata.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, sorts all spikes by time, reindexes the retained clusters to a contiguous session-wide numbering, and bins spikes into 20 ms half-open bins over a 2 s trial window. The stored neural values are spike counts, not firing rates.

ii.
```python
spike_times = np.concatenate(spike_times_l)
spike_clusters = np.concatenate(spike_clu_l)
...
srt = np.argsort(spike_times, kind='stable')
spike_times = spike_times[srt]
spike_clusters = spike_clusters[srt]
```

```python
def bin_spikes(spike_times, spike_clusters, n_clusters, align_times,
               t_start=TIME_WINDOW[0], t_end=TIME_WINDOW[1], binsize=BINSIZE, n_bins=NBINS):
    out = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    ...
    b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
    ...
    cnt = np.bincount(idx, minlength=n_clusters * n_bins)
    out[k] = cnt.reshape(n_clusters, n_bins)
```

iii. In the notes, the AI explicitly states it stores “float32 spike counts (not z-scored)” and emphasizes matching the bin edges of `bincount2D`, while avoiding the reference code’s slower per-trial multiprocessing implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1.0`, maps them to Beryl regions, and excludes any neuron whose Beryl region is `void` or `root`. It also drops any session left with fewer than 5 retained neurons.

ii.
```python
QC_LABEL = 1.0
EXCLUDE_REGIONS = ('void', 'root')
MIN_NEURONS = 5
```

```python
beryl = br.acronym2acronym(acronyms, mapping='Beryl')
keep = (labels >= QC_LABEL) & (~np.isin(beryl, EXCLUDE_REGIONS))
keep_ids = np.where(keep)[0]
if len(keep_ids) < MIN_NEURONS:
    return dict(eid=eid, skipped='too few good neurons')
```

iii. The notes justify `label == 1` as the BWM paper’s stringent single-unit QC, and justify excluding `void/root` as a grey-matter restriction. The trajectory shows `MIN_NEURONS = 5` was added later to suppress degenerate sessions after verification.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For every retained trial, the neural window is from -0.5 s to +1.5 s relative to stimulus onset, and binning is done in that aligned frame.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
align_all = trials[ALIGN_TIME].to_numpy()
...
binned_spikes = bin_spikes(sp_t, sp_c, len(keep_ids), align)
```

iii. This is one of the clearest “match the reference” decisions in the notes: the AI repeatedly cites the Zhang params `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 20 ms bins and 100 bins per trial. There is no extra temporal rebinning after the initial binning step.

ii.
```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

```python
b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
```

iii. The AI’s notes explicitly anchor this to the reference papers’ “2 s trials, 20 ms bins, T=100” description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not loaded from a separate raw data column. The input is synthetically defined from the alignment event `stimOn_times` plus the fixed decoding window and bin size.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
```

```python
align_all = trials[ALIGN_TIME].to_numpy()
...
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. The notes say this variable is “required by the decoder-task spec” and should simply encode time within the aligned trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes a length-100 vector of bin centers from -0.49 s to 1.49 s and reuses that same vector for every trial.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)   # -0.49 ... 1.49
...
inputs.append(np.stack([bin_centres.astype(np.float32),
                        np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
```

iii. The justification is minimal: the notes say the decoder input should be time from stimulus onset, represented continuously on the trial grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same 100-bin trial structure as the neural data. The neural data are counted in the 20 ms bins, and the input records the centers of those same bins.

ii.
```python
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
...
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

```python
neural.append(binned_spikes[k])
inputs.append(np.stack([bin_centres.astype(np.float32),
                        np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
```

iii. The notes treat this as a direct consequence of using the same alignment event and binning scheme for all streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` trial column. A new block starts whenever `probabilityLeft` changes.

ii.
```python
def trial_number_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=float)
    newblock = np.ones(len(pl), dtype=bool)
    newblock[1:] = pl[1:] != pl[:-1]
```

iii. The notes explain that `probabilityLeft` is the only directly available block indicator, so block boundaries must be reconstructed from changes in that value.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a 0-based within-block counter over the full raw trial sequence, then indexes that array after trial filtering. The value is broadcast across time bins within each retained trial.

ii.
```python
idx = np.zeros(len(pl), dtype=np.int64)
c = 0
for i in range(len(pl)):
    c = 0 if newblock[i] else c + 1
    idx[i] = c
return idx
```

```python
tnb_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
tnb = tnb_all[idx_trials].astype(np.float32)
...
np.full(NBINS, tnb[k], dtype=np.float32)
```

iii. The notes explicitly justify computing the count before filtering so excluded trials do not corrupt the animal’s real position within the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the `trials['choice']` column.

ii.
```python
choice_raw = trials['choice'].to_numpy()[idx_trials]         # +1 left, -1 right
```

iii. The notes say the AI verified the IBL sign convention and concluded `+1` means left choice and `-1` means right choice.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After filtering out no-choice trials earlier, the AI recodes `choice` to a binary class: left becomes 0 and right becomes 1. The label is then repeated across all 100 bins of the trial.

ii.
```python
choice_lab = (choice_raw < 0).astype(np.int64)               # left -> 0, right -> 1
...
outputs.append(np.stack([np.full(NBINS, choice_lab[k], dtype=np.int64),
                         np.full(NBINS, prior_lab[k], dtype=np.int64),
                         wheel_lab[k], me_lab[k]], axis=0))
```

iii. The notes justify this as matching the task spec’s required coding of left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials['probabilityLeft']`.

ii.
```python
pleft_raw = trials['probabilityLeft'].to_numpy()[idx_trials]
```

iii. The notes tie this directly to the task’s requested “prior probability of left”.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps the three valid `probabilityLeft` values to class labels 0, 1, and 2 for 0.2, 0.5, and 0.8 respectively. Invalid values are dropped.

ii.
```python
prior_lab = np.select([pleft_raw == 0.2, pleft_raw == 0.5, pleft_raw == 0.8], [0, 1, 2],
                      default=-1).astype(np.int64)
if np.any(prior_lab < 0):
    keep_pl = prior_lab >= 0
    ...
```

iii. The notes say this recoding is required by the decoder output specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The AI derives wheel speed from the wheel trace loaded by `SessionLoader`. It uses the loader-provided `velocity` and takes its absolute value.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].to_numpy()
wheel_speed_raw = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. The notes explicitly connect this to the Zhang helper `load_target_behavior('wheel-speed')` and to the IBL wheel preprocessing done inside `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI uses `SessionLoader`’s processed wheel velocity, converts it to speed with `abs`, interpolates it onto a fixed 100-bin trial grid, and later discretizes the resulting values into per-session tertiles.

ii.
```python
wheel_speed_raw = np.abs(sl.wheel['velocity'].to_numpy())
...
wheel_binned, wheel_valid = bin_behavior(wheel_times, wheel_speed_raw, align)
...
wheel_lab, wheel_q = discretize_tertiles(wheel_binned, allrows)
```

iii. The notes say this was chosen to match the reference behavior extraction while satisfying the task’s requirement that outputs be categorical.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is thresholded into 3 classes by computing the 1/3 and 2/3 quantiles over all retained wheel-speed samples in that session. Values below the first threshold are 0, between thresholds are 1, and above the second threshold are 2.

ii.
```python
def discretize_tertiles(x, valid_rows):
    ref = x[valid_rows].ravel()
    q1, q2 = np.quantile(ref, [1. / 3., 2. / 3.])
    lab = np.zeros(x.shape, dtype=np.int64)
    lab[x > q1] = 1
    lab[x > q2] = 2
    return lab, (float(q1), float(q2))
```

```python
wheel_lab, wheel_q = discretize_tertiles(wheel_binned, allrows)
```

iii. In the notes, the AI justifies per-session tertiles by arguing that wheel scale is session-specific and balanced terciles make the decoder metric interpretable.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated separately for each retained trial using the same stimulus-aligned 2 s window as the neural data, but the AI places the behavior samples at `np.linspace(beg + binsize, end, n_bins)`, i.e. the right edge of each bin rather than the bin center.

ii.
```python
grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
...
out[k] = np.interp(grid[k], tt, vv)
```

```python
wheel_binned, wheel_valid = bin_behavior(wheel_times, wheel_speed_raw, align)
```

iii. The notes say this was intended to match the original Zhang `get_behavior_per_interval` implementation exactly.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the motion-energy trace loaded with `SessionLoader.load_motion_energy`. The AI prefers the left camera and falls back to the right camera if needed, then uses the `whiskerMotionEnergy` column and its timestamps.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        df = sl.motion_energy[key]
        me_times = df['times'].to_numpy()
        me_vals = df['whiskerMotionEnergy'].to_numpy()
        camera_used = view
        break
```

iii. The notes justify left-preferred/right-fallback behavior as the same rule used in the reference pipeline.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy trace as-is, interpolates it onto the per-trial 100-bin grid, and discretizes the interpolated values into per-session tertiles.

ii.
```python
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
...
me_lab, me_q = discretize_tertiles(me_binned, allrows)
```

iii. The notes say no extra normalization or filtering is applied beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same session-specific tertile rule as wheel speed: the flattened retained whisker trace for a session is split at its 1/3 and 2/3 quantiles into low, medium, and high classes.

ii.
```python
def discretize_tertiles(x, valid_rows):
    ref = x[valid_rows].ravel()
    q1, q2 = np.quantile(ref, [1. / 3., 2. / 3.])
    ...
```

```python
me_lab, me_q = discretize_tertiles(me_binned, allrows)
```

iii. The notes reuse the same justification as wheel speed: camera-dependent scale makes per-session thresholds more sensible.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, whisker motion energy is aligned per trial to the stimulus-onset-centered 2 s neural window, but sampled at bin right edges rather than centers.

ii.
```python
grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
...
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
```

iii. The notes justify this as a deliberate attempt to mirror `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles missing or bad data by dropping the affected trial or session. No-choice and NaN-contaminated trials are masked out. Sessions with no whisker motion energy, too few good neurons, too few retained trials, or loader exceptions are skipped. Trials with insufficient wheel/camera coverage, outside the ephys recording window, or with zero total spikes are also dropped.

ii.
```python
if me_times is None:
    return dict(eid=eid, skipped='no whisker motion energy')
```

```python
if len(keep_ids) < MIN_NEURONS:
    return dict(eid=eid, skipped='too few good neurons')
...
if good.sum() < MIN_TRIALS:
    return dict(eid=eid, skipped='too few trials with behaviour coverage')
...
if nonzero.sum() < MIN_TRIALS:
    return dict(eid=eid, skipped='too few trials with any spikes')
```

```python
def _worker(args):
    ...
    except Exception as e:
        traceback.print_exc()
        return dict(eid=eid, skipped=f'exception: {e}')
```

iii. The notes describe this as conservative curation driven by verification failures: when the decoder or spot checks exposed bad windows or degenerate trials, the AI added filters instead of imputing data.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike-sorting I/O as the main cost. Per the notes, loading spike sorting from disk dominates conversion time; the rest of the per-session processing is much cheaper.

ii.
```python
for pid, pname in zip(pids, pnames):
    ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. In `CONVERSION_NOTES.md`, the AI says session-level cost is “spike-sorting I/O, ~3-6 s per probe”, and that spike loading dominates runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has trial-by-trial Python loops in `bin_spikes`, `bin_behavior`, and the final trial assembly loop. The AI’s stated optimization work focused on replacing the original per-trial multiprocessing with `searchsorted`/`bincount`/`interp`, but it still kept one Python loop per trial in each stage.

ii.
```python
for k in range(n_trials):
    s, e = i0[k], i1[k]
    ...
    cnt = np.bincount(idx, minlength=n_clusters * n_bins)
    out[k] = cnt.reshape(n_clusters, n_bins)
```

```python
for k in range(n_trials):
    ib, ie = idx_beg[k], idx_end[k]
    ...
    out[k] = np.interp(grid[k], tt, vv)
```

```python
for k in range(n_trials):
    neural.append(binned_spikes[k])
    inputs.append(np.stack([bin_centres.astype(np.float32),
                            np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
```

iii. The notes argue that the expensive vectorization target was the reference code’s per-trial multiprocessing overhead, and that session-level parallelism plus NumPy inside the remaining loops was sufficient.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some small per-trial work that could be hoisted out: `bin_centres.astype(np.float32)` is recomputed inside the output-building loop, and constant choice/prior arrays are rebuilt trial by trial with `np.full`. It also computes both continuous wheel/whisker traces and their discretized labels even though only the labels survive into the final dataset.

ii.
```python
for k in range(n_trials):
    ...
    inputs.append(np.stack([bin_centres.astype(np.float32),
                            np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
    outputs.append(np.stack([np.full(NBINS, choice_lab[k], dtype=np.int64),
                             np.full(NBINS, prior_lab[k], dtype=np.int64),
                             wheel_lab[k], me_lab[k]], axis=0))
```

iii. There is no explicit note section on this, but the AI’s code structure shows it favored straightforward per-trial assembly over squeezing out these smaller repeated allocations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of processing are only for diagnostics or intermediate filtering and do not end up in the final pickle: `wheel_q`, `me_q`, `trial_idx`, `timing`, `total_time`, and the full continuous `wheel_binned`/`me_binned` arrays are computed but discarded after their labels or plots are produced. The script also builds `eid2lab` and never uses it.

ii.
```python
eid2lab = bwm.groupby('eid').lab.first().to_dict()
```

```python
result = dict(
    ...
    wheel_quantiles=wheel_q,
    me_quantiles=me_q,
    trial_idx=idx_trials,
    ...
    timing=timing,
    total_time=time.time() - t_start_session,
)
```

```python
wheel_binned, wheel_valid = bin_behavior(wheel_times, wheel_speed_raw, align)
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
...
wheel_lab, wheel_q = discretize_tertiles(wheel_binned, allrows)
me_lab, me_q = discretize_tertiles(me_binned, allrows)
```

iii. The notes show these extra computations were intentional for validation, debugging, and the optional `--show-processing` figures, not because the final decoder format needed them.
