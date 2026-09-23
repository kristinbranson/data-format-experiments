# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI first ensures the local ONE cache has usable parquet tables, then enumerates sessions and probe insertions from the frozen release CSV `code_zhang2025/data/bwm_release.csv`. It loads trial, wheel, and camera data with `SessionLoader`, and spike data with `SpikeSortingLoader`. It does not read `/app/data` files directly for the actual signals.

ii. 
```python
def ensure_cache_tables():
    tables = Path('/app/data/one_cache/sessions.pqt')
    if not tables.exists():
        import build_one_cache
        build_one_cache.main()
```

```python
def build_jobs(sample):
    bwm = pd.read_csv(BWM_FREEZE, index_col=0)
    for eid, g in bwm.groupby('eid', sort=True):
        probes = list(zip(g['pid'], g['probe_name']))
        jobs.append((eid, g['subject'].iloc[0], probes))
```

```python
sl = SessionLoader(one=one, eid=eid)
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
```

iii. In `CONVERSION_NOTES.md`, the AI says the staged release tables did not index the on-disk dataset revisions, so it rebuilt ONE cache tables and relied on `bwm_release.csv` for the session/probe mapping while keeping all actual data access through `ONE` and `brainbox`.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. In the assembled output, subjects are the sorted unique names, and `subject_idx` maps each kept session to its subject.

ii. 
```python
for eid, g in bwm.groupby('eid', sort=True):
    probes = list(zip(g['pid'], g['probe_name']))
    jobs.append((eid, g['subject'].iloc[0], probes))
```

```python
subjects = sorted({r['subject'] for r in ok})
sub2idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([sub2idx[r['subject']] for r in ok], dtype=np.int64),
```

iii. The notes justify this as reusing the same freeze file the reference code uses after the cache-index problem made direct session/probe resolution unreliable offline.

## 1-c. How are the data split into sessions?

i. Sessions are identified by `eid`. The AI groups the freeze table by `eid`, treating each group as one session and attaching that session’s probe list to the job.

ii. 
```python
for eid, g in bwm.groupby('eid', sort=True):
    probes = list(zip(g['pid'], g['probe_name']))
    jobs.append((eid, g['subject'].iloc[0], probes))
```

iii. The justification in the notes is that `bwm_release.csv` already gives the release session/probe mapping, and using it avoided depending on online Alyx resolution.

## 1-d. How are the data split into trials?

i. Trials come from the IBL trials table loaded by `SessionLoader` and `load_trials_and_mask`; each row of that table is one trial, and later arrays are indexed by the kept trial rows.

ii. 
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials, mask = load_trials_and_mask(
    one=one, eid=eid, min_rt=MIN_RT, max_rt=MAX_RT, nan_exclude='default',
    max_trial_len=MAX_TRIAL_LEN, exclude_unbiased=False, exclude_nochoice=True,
    sess_loader=sl)
```

iii. The AI’s notes describe this as following the reference helper directly rather than inventing a new trial parser.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies `load_trials_and_mask` with the reference thresholds, keeps the unbiased block, then further drops trials whose wheel or whisker traces do not cover the full window and trials whose neural window falls outside the recorded spike train. Sessions with fewer than 2 surviving trials are skipped.

ii. 
```python
trials, mask = load_trials_and_mask(
    one=one, eid=eid, min_rt=MIN_RT, max_rt=MAX_RT, nan_exclude='default',
    max_trial_len=MAX_TRIAL_LEN, exclude_unbiased=False, exclude_nochoice=True,
    sess_loader=sl)
```

```python
wheel_vals, wheel_good = bin_behavior(wt, wv, t_begs)
me_vals, me_good = bin_behavior(mt, mv, t_begs)
beh_good = wheel_good & me_good
```

```python
in_rec = ((t_begs >= spike_times_k[0] - BINSIZE)
          & (t_begs + N_BINS * BINSIZE <= spike_times_k[-1] + BINSIZE))
valid = beh_good & in_rec
if valid.sum() < MIN_TRIALS_PER_SESSION:
    res['skip'] = f'only {int(valid.sum())} trials survive behaviour/spike checks'
```

iii. The notes say this mirrors `load_trials_and_mask` and the reference behavior-coverage logic, with one extra spike-recording coverage check to avoid silently generating all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is derived from spike times and spike cluster assignments, after using the cluster and channel tables to attach QC labels and anatomical acronyms needed for filtering and region assignment.

ii. 
```python
spikes = dict(ssl._load_object(wanted))
clusters = ssl.load_spike_sorting_object('clusters')
channels = ssl.load_channels()
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

```python
return spikes['times'], spikes['clusters'], clusters, np.asarray(beryl)
```

iii. The notes say the AI deliberately reads only `spikes.times` and `spikes.clusters` from the spike object because the other spike arrays are unnecessary for conversion.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, filters the kept clusters, and bins spikes into 20 ms bins over a 2 s window. It stores the result as per-trial `(n_neurons, 100)` spike-count matrices in `float32`; it does not divide by bin width to convert to Hz.

ii. 
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

```python
b = np.floor(rel / BINSIZE).astype(np.int64)
flat = (trial_id * n_clusters + spike_clusters[idx]) * N_BINS + b
binned.reshape(-1)[:] = np.bincount(
    flat, minlength=n_trials * n_clusters * N_BINS).astype(np.float32)
```

```python
res.update({
    'neural': binned,                       # (n_trials, n_neurons, N_BINS)
```

iii. The notes justify this as a vectorized equivalent of the reference binner and explicitly state the choice to store raw spike counts rather than z-scored activity.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1` and whose Beryl-mapped acronym is not `root` or `void`. It then remaps the surviving cluster ids to a dense 0-based index.

ii. 
```python
label = clusters['label'].to_numpy()
good_unit = (label >= GOOD_UNIT_LABEL) & ~np.isin(beryl, NON_GREY)
```

```python
GOOD_UNIT_LABEL = 1.0
NON_GREY = ('root', 'void')
```

iii. The notes argue this matches the data paper’s “well-isolated grey-matter units” curation, and explicitly present it as a deliberate difference from the reference caching script, which keeps more clusters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural windows are aligned to `stimOn_times`. For each kept trial, the window start is `stimOn_times - 0.5`, and spikes are binned relative to that start over the next 2 seconds.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]
```

```python
rel = spike_times[idx] - t_begs[trial_id]
b = np.floor(rel / BINSIZE).astype(np.int64)
```

iii. The notes say this was chosen to match both the task instructions and the reference caching parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 20 ms bins across a 2 s window, yielding 100 time bins per trial. No further temporal rebinning is applied.

ii. 
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The notes repeatedly state that 20 ms and 100 bins were taken directly from the reference caching parameters and paper description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial alignment event `stimOn_times`, together with the fixed decoding window and bin size.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
```

```python
t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]
```

iii. The AI’s notes frame this input as the analytic bin-time coordinate implied by the same stimulus-aligned trial window used for the neural data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes the 100 bin centers analytically, from `-0.49` to `1.49` s in 20 ms steps, and broadcasts that vector to every trial.

ii. 
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE).astype(np.float32)
inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = bin_centers[None, :]
```

iii. The notes describe this as a direct construction from the chosen window and bin size, not a measured raw-data field.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It shares the same trial windows and time-bin grid as the neural data: neural counts are binned from the same `t_begs`, and the input stores the corresponding bin-center times for those bins.

ii. 
```python
t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]
```

```python
rel = spike_times[idx] - t_begs[trial_id]
b = np.floor(rel / BINSIZE).astype(np.int64)
```

```python
inputs[:, 0, :] = bin_centers[None, :]
```

iii. The notes say this input exists specifically to expose the neural bin positions relative to stimulus onset to the decoder.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`, using changes in that value to infer block boundaries.

ii. 
```python
def trial_number_in_block(probability_left):
    pl = np.asarray(probability_left, dtype=float)
    change = np.ones(len(pl), dtype=bool)
    change[1:] = pl[1:] != pl[:-1]
```

iii. The notes explicitly say block structure is recovered from `probabilityLeft`, the same field the reference code calls `block`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI detects each block start where `probabilityLeft` changes, computes a 0-based count within each block on the full trial table before filtering, then broadcasts the kept trials’ counts across time bins.

ii. 
```python
block_id = np.cumsum(change) - 1
starts = np.flatnonzero(change)
return (np.arange(len(pl)) - starts[block_id]).astype(np.float32)
```

```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
inputs[:, 1, :] = tib_all[keep][:, None]
```

iii. The notes justify computing it before trial exclusion so removed trials still advance the block counter, preserving the animal’s real within-block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials['choice']`.

ii. 
```python
choice = map_choice(trials['choice'].to_numpy()[keep])
```

iii. The notes say this follows the IBL trials table coding and the task’s requested left/right remapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI remaps IBL choice codes from `+1/-1` to `0/1` for left/right and broadcasts each trial’s value across all 100 time bins.

ii. 
```python
def map_choice(choice):
    out = np.full(len(choice), -1, dtype=np.int64)
    out[np.asarray(choice) == 1] = 0
    out[np.asarray(choice) == -1] = 1
    return out
```

```python
outputs[:, 0, :] = choice[:, None]
```

iii. The notes say the sign convention was checked against left- and right-stimulus trials before adopting the task’s left=`0`, right=`1` coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials['probabilityLeft']`.

ii. 
```python
prior = map_prior(trials['probabilityLeft'].to_numpy()[keep])
```

iii. The notes describe `probabilityLeft` as the block prior field used throughout the reference code and dataset.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI remaps `0.2`, `0.5`, and `0.8` to classes `0`, `1`, and `2`, then broadcasts the class across all time bins in the trial.

ii. 
```python
def map_prior(probability_left):
    pl = np.asarray(probability_left, dtype=float)
    out = np.full(len(pl), -1, dtype=np.int64)
    out[np.isclose(pl, 0.2)] = 0
    out[np.isclose(pl, 0.5)] = 1
    out[np.isclose(pl, 0.8)] = 2
    return out
```

```python
outputs[:, 1, :] = prior[:, None]
```

iii. The notes justify this as the exact mapping requested by the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from the wheel timestamps and position, using `SessionLoader.load_wheel()` to obtain the processed wheel trace and then taking absolute velocity.

ii. 
```python
def load_wheel_speed(one, eid):
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    return (sl.wheel['times'].to_numpy(),
            np.abs(sl.wheel['velocity'].to_numpy()))
```

iii. The notes say this matches the same source used by the reference behavior loader for `wheel-speed`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI uses the wheel velocity returned by `SessionLoader`, takes its absolute value, interpolates it onto the per-trial decoder grid, and then discretizes the interpolated values into 3 session-specific classes.

ii. 
```python
wt, wv = load_wheel_speed(one, eid)
wheel_vals, wheel_good = bin_behavior(wt, wv, t_begs)
```

```python
wheel_cls, wheel_edges = discretize_tertiles(wheel_vals)
outputs[:, 2, :] = wheel_cls
```

iii. The notes justify using the reference’s wheel-processing path and then discretizing because the decoder outputs must be categorical.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is thresholded per session into 3 bins using the 33.3rd and 66.7th percentiles of all retained binned values from that session. If the percentiles collapse because the trace is degenerate, the AI falls back to thresholds derived from the unique values.

ii. 
```python
flat = values.ravel()
edges = np.percentile(flat, [100.0 / 3.0, 200.0 / 3.0])
```

```python
if edges[0] == edges[1]:
    uniq = np.unique(flat)
    ...
return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. The notes justify per-session tertiles because wheel scale varies by mouse/session and balanced terciles give a clean 1/3 chance level.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI aligns wheel speed to the same stimulus-locked trial windows as the neural data, but samples the continuous trace at the right edge of each neural bin (`t_beg + 0.02, ..., t_end`) rather than at the bin center.

ii. 
```python
grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)
query = t_begs[good][:, None] + grid[None, :]
interp = np.interp(query.ravel(), target_times, target_vals).reshape(-1, N_BINS)
```

iii. The docstring and notes explicitly justify this as reproducing the reference helper `get_behavior_per_interval`, which evaluates behavior at the right edge of the spike-count bins.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the released camera motion-energy trace and frame times, preferring the left camera and falling back to the right camera when needed.

ii. 
```python
for view in ('left', 'right'):
    try:
        sl = SessionLoader(one=one, eid=eid)
        sl.load_motion_energy(views=[view])
        me = sl.motion_energy[f'{view}Camera']
        return (me['times'].to_numpy(),
                me['whiskerMotionEnergy'].to_numpy(), view)
```

iii. The notes say this matches the same left-then-right preference used by the reference behavior-loading code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released whisker motion-energy trace as-is, interpolates it onto the per-trial decoder grid, and discretizes the interpolated values into 3 session-specific classes.

ii. 
```python
mt, mv, view = load_whisker_me(one, eid)
me_vals, me_good = bin_behavior(mt, mv, t_begs)
```

```python
me_cls, me_edges = discretize_tertiles(me_vals)
outputs[:, 3, :] = me_cls
```

iii. The notes justify this as using the reference signal source unchanged, then adapting it to the categorical decoder-output requirement.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is thresholded per session into 3 bins using the 33.3rd and 66.7th percentiles of all retained binned values from that session, with the same degenerate-edge fallback used for wheel speed.

ii. 
```python
edges = np.percentile(flat, [100.0 / 3.0, 200.0 / 3.0])
...
return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. The notes justify per-session thresholds because whisker motion energy is in arbitrary camera/ROI-dependent units that are not comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, the AI aligns whisker motion energy to the same stimulus-locked windows as the neural data but samples the continuous trace at the right edge of each neural bin rather than at the center.

ii. 
```python
grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)
query = t_begs[good][:, None] + grid[None, :]
interp = np.interp(query.ravel(), target_times, target_vals).reshape(-1, N_BINS)
```

iii. The justification is the same as for wheel speed: the notes state this reproduces the reference behavior-alignment helper exactly.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly drops unusable data rather than imputing it. It rebuilds broken cache tables up front, skips sessions with no whisker video, no kept units, or too few valid trials, drops trials whose wheel or whisker traces do not span the full window or whose spike window lies outside the recording, and catches session-level exceptions so one failure does not abort the full run. It also handles degenerate discretization thresholds explicitly.

ii. 
```python
if not tables.exists():
    import build_one_cache
    build_one_cache.main()
```

```python
if mt is None:
    res['skip'] = 'no whisker motion energy (neither camera)'
    return res
```

```python
if res['n_units_kept'] == 0:
    res['skip'] = 'no well-isolated grey-matter units'
    return res
```

```python
except Exception as e:
    res['skip'] = f'{type(e).__name__}: {e}'
    res['traceback'] = traceback.format_exc()
```

iii. The notes emphasize robustness to real defects in the staged data and document several bugs the AI fixed while making the pipeline resilient.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike loading and merging as the dominant cost, with behavioral loading/binning and the final multi-GB pickle write also taking noticeable time.

ii. 
```python
timing['behavior'] = time.time() - t0
...
timing['spikes_load'] = time.time() - t0
...
timing['bin_spikes'] = time.time() - t0
```

```python
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In the notes, the AI states that reading spike sorting from disk dominates per-session time, and reports whole-run timing breakdowns showing spike I/O as the main cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI says the expensive per-trial spike-binning and behavior-interpolation loops from the reference code were exactly the places worth vectorizing, and it replaced them with one vectorized `np.bincount` and one vectorized `np.interp` per session. What remains is mainly session-level parallelism and short per-probe/per-session loops.

ii. 
```python
trial_id = np.repeat(np.arange(n_trials), counts)
...
binned.reshape(-1)[:] = np.bincount(
    flat, minlength=n_trials * n_clusters * N_BINS).astype(np.float32)
```

```python
query = t_begs[good][:, None] + grid[None, :]
interp = np.interp(query.ravel(), target_times, target_vals).reshape(-1, N_BINS)
```

iii. The notes explicitly call out the reference code’s per-trial `bincount2D` and `interp1d` pools as inefficient and describe the AI’s vectorized replacements as a major speedup.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats some session-level loader work: it instantiates `SessionLoader` separately for trials, wheel, and whisker motion energy; `load_whisker_me` may attempt the left camera and then the right; and the optional plotting path reloads raw wheel and motion-energy streams again for diagnostics.

ii. 
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
```

```python
def load_wheel_speed(one, eid):
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
```

```python
for view in ('left', 'right'):
    try:
        sl = SessionLoader(one=one, eid=eid)
        sl.load_motion_energy(views=[view])
```

iii. The notes focus more on removing repeated per-trial work from the reference code, but the conversion still preserves some repeated loader work because it keeps the implementation simple and supports the diagnostic plotting mode.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and keeps several diagnostic intermediates that are not part of the final neural/input/output arrays used downstream: per-session timing breakdowns, raw interpolated wheel and whisker traces, trial indices, and tertile edges. The `--show-processing` plots also reload raw data purely for inspection.

ii. 
```python
res.update({
    'trial_idx': keep,
    'wheel_edges': wheel_edges,
    'me_edges': me_edges,
    'wheel_raw': wheel_vals.astype(np.float32),
    'me_raw': me_vals.astype(np.float32),
    'timing': timing,
```

```python
if args.show_processing:
    one, _ = get_one()
    for r in ok[:2]:
        plot_processing(r, one)
```

iii. The notes justify these extras as sanity-check and debugging aids; they were used to validate alignment and edge cases, even though most are not consumed by `train_decoder.py`.
