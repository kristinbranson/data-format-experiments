# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not discover sessions through `one.search()`. Instead, it loads the session/probe manifest from the reference freeze file `bwm_release.csv`, extracts unique session IDs (`eid`s), and then processes each session with ONE-based loaders. Within each session it uses `SpikeSortingLoader` to read spikes/clusters per probe and `SessionLoader` plus the reference `load_trials_and_mask` to read the trials table and behavioral streams.

ii. 
```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))       # preserve freeze order
```

```python
for i, eid in enumerate(eids):
    sub = bwm[bwm.eid == eid]
    jobs.append((eid, sub.pid.tolist(), sub.probe_name.tolist(),
                 sub.subject.iloc[0], sub.lab.iloc[0], show, out_dir))
```

```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
sp, cl, ch = ssl.load_spike_sorting()
...
sess_loader = SessionLoader(one=one, eid=eid)
trials, ref_mask = load_trials_and_mask(one=one, eid=eid,
                                        max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
```

iii. In `CONVERSION_NOTES.md`, the AI says it is following the Zhang reference freeze (`code/code_zhang2025/data/bwm_release.csv`) and deliberately imports the reference code where practical, especially `load_trials_and_mask`, so curation matches the reference implementation literally.

## 1-b. How are the data split into subjects?

i. Subject identity comes from the `subject` column of `bwm_release.csv`. Each session job carries its subject name, and after conversion the AI builds a sorted unique `subjects` list and `subject_idx` per retained session.

ii. 
```python
jobs.append((eid, sub.pid.tolist(), sub.probe_name.tolist(),
             sub.subject.iloc[0], sub.lab.iloc[0], show, out_dir))
```

```python
subjects = sorted({r['subject'] for r in results})
subj_lookup = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subj_lookup[r['subject']] for r in results], dtype=np.int64),
```

iii. The notes justify this as using the release freeze metadata directly instead of inferring subjects from paths or filenames.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid`s in the freeze file. The code preserves freeze order, creates one job per `eid`, and treats each job result as one session in the output lists.

ii. 
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))       # preserve freeze order
```

```python
for i, eid in enumerate(eids):
    sub = bwm[bwm.eid == eid]
    jobs.append((eid, sub.pid.tolist(), sub.probe_name.tolist(),
                 sub.subject.iloc[0], sub.lab.iloc[0], show, out_dir))
```

```python
'neural': [r['neural'] for r in results],
'input': [r['input'] for r in results],
'output': [r['output'] for r in results],
```

iii. The AI’s notes treat the freeze file as the authoritative session list used by the reference code, so no further splitting logic is needed beyond unique `eid`.

## 1-d. How are the data split into trials?

i. The AI uses the trials table loaded by `load_trials_and_mask`; each row of the resulting `trials` dataframe is one trial. Trial-level masks are then applied, and retained rows define the per-trial neural/input/output entries.

ii. 
```python
sess_loader = SessionLoader(one=one, eid=eid)
trials, ref_mask = load_trials_and_mask(one=one, eid=eid,
                                        max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
n_trials_raw = len(trials)
mask = ref_mask.to_numpy().astype(bool)
```

```python
'neural': [np.ascontiguousarray(binned[k]) for k in range(n_kept)],
'input': [np.ascontiguousarray(inputs[k]) for k in range(n_kept)],
'output': [np.ascontiguousarray(outputs[k]) for k in range(n_kept)],
```

iii. The notes describe the trials table as the source of per-trial task variables and use the reference trial mask so that trial identity follows the reference implementation.

## 1-e. How are trials filtered based on quality controls?

i. The AI starts from the reference `load_trials_and_mask(..., max_trial_len=10.0)` mask, then further requires valid prior codes, valid binary choices, finite alignment times, full ephys coverage of the 2 s window, and full wheel and whisker trace coverage over the same window.

ii. 
```python
trials, ref_mask = load_trials_and_mask(one=one, eid=eid,
                                        max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
...
mask = ref_mask.to_numpy().astype(bool)

prior_code = map_prior(trials['probabilityLeft'].to_numpy())
mask &= prior_code >= 0
choice_raw = trials['choice'].to_numpy(dtype=float)
mask &= np.isin(choice_raw, (-1.0, 1.0))

align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
mask &= np.isfinite(align_times)
interval_begs = align_times + WIN[0]
...
mask &= (interval_begs >= rec_span[0]) & (interval_begs + BINSIZE * NBINS <= rec_span[1])
...
mask &= wheel_ok & me_ok
```

iii. The notes justify this as combining the reference trial curation with extra safeguards against mislabeled zero-neural trials and incomplete behavioral windows. Step 10 explicitly says the recording-span filter was added after finding trials whose behavior outlasted the ephys recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays are derived from spike times and cluster assignments loaded from spike sorting. Cluster metadata (`label`, `acronym`, `uuids`) are additionally used for QC, region mapping, and provenance, but the binned neural signal itself comes from `spikes['times']` and remapped `spikes['clusters']`.

ii. 
```python
sp, cl, ch = ssl.load_spike_sorting()
...
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```

```python
remap = np.full(int(np.max(spikes['clusters'])) + 2, -1, dtype=np.int64)
remap[keep_idx] = np.arange(len(keep_idx))
su = remap[spikes['clusters']]
sel = su >= 0
binned = bin_spikes(np.ascontiguousarray(spikes['times'][sel]),
                    np.ascontiguousarray(su[sel]),
                    len(keep_idx),
                    interval_begs[mask])
```

iii. The notes say the conversion follows the reference ephys path: load spike sorting per probe, merge probes within a session, and use cluster metadata only to decide which units survive and what region labels they get.

## 2-b. How is the `neural` data processed?

i. The AI merges all probes within a session, filters to surviving units, and bins spikes into 100 non-overlapping 20 ms bins per trial over a 2 s window aligned to stimulus onset. Unlike the human reference solution, it keeps raw spike counts rather than dividing by bin width to produce Hz.

ii. 
```python
if len(spikes_list) == 1:
    spikes, clusters = spikes_list[0], clusters_list[0]
    clusters = clusters.reset_index(drop=True)
else:
    spikes, clusters = merge_probes(spikes_list, clusters_list)
```

```python
def bin_spikes(spike_times, spike_units, n_units, interval_begs):
    ...
    i0 = np.searchsorted(spike_times, safe, side='left')
    i1 = np.searchsorted(spike_times, safe + BINSIZE * NBINS, side='left')
    for k in range(n_trials):
        ...
        b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        counts = np.bincount(uu * NBINS + b, minlength=n_units * NBINS)
        out[k] = counts.reshape(n_units, NBINS)
    return out
```

iii. In Step 1 and Step 5 of the notes, the AI says the reference cached dataset stores unnormalised spike counts and decoder-side normalisation should happen later, so it intentionally retains raw counts even though the target format would also permit rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1` and Beryl-mapped region not in `('root', 'void')`, then drops whole sessions with fewer than 5 surviving neurons.

ii. 
```python
QC_LABEL = 1.0
NON_GREY = ('root', 'void')
MIN_NEURONS = 5
```

```python
beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
label = clusters['label'].to_numpy(dtype=float)
keep = (label >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
return np.nonzero(keep)[0], beryl
```

```python
keep_idx, beryl = select_units(clusters)
...
if len(keep_idx) < MIN_NEURONS:
    raise RuntimeError(f'only {len(keep_idx)} well-isolated grey-matter units')
```

iii. The notes justify `label >= 1` as matching the data paper’s “well-isolated neurons” and justify dropping `root`/`void` as enforcing grey-matter-only inclusion. In Step 10, the AI says the `>= 5 neurons` session threshold was added after validation exposed near-empty neural populations that produced all-zero trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. For each retained trial the code defines the interval start as `stimOn_times - 0.5` and bins spikes over the next 2.0 s, so neural time zero is stimulus onset.

ii. 
```python
PARAMS = {
    'interval_len': 2,
    'binsize': 0.02,
    'single_region': False,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
```

```python
align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
interval_begs = align_times + WIN[0]
...
binned = bin_spikes(..., interval_begs[mask])
```

iii. The notes repeatedly justify this as matching both the decoder task and the reference Zhang parameters: stimulus-onset alignment, `[-0.5, 1.5]` s window, 20 ms bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 20 ms bins (`binsize = 0.02`) over 100 bins per trial. There is no later temporal rebinning or smoothing.

ii. 
```python
BINSIZE = PARAMS['binsize']
WIN = PARAMS['time_window']
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))          # 100
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)
BIN_RIGHT_EDGES = BIN_LEFT_EDGES + BINSIZE
```

```python
b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
np.clip(b, 0, NBINS - 1, out=b)
```

iii. The notes say this is copied from the reference `params` dict and from the papers’ “2-s trials, 20-ms bins, T = 100” description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI derives trial alignment from `trials.stimOn_times`, but the stored input values themselves are a fixed synthetic grid of bin left edges from `-0.5` to `1.48` s. So the raw data contribute the alignment event rather than per-bin time values.

ii. 
```python
align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
interval_begs = align_times + WIN[0]
```

```python
inputs = np.empty((n_kept, len(INPUT_NAMES), NBINS), dtype=np.float32)
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. The notes justify this as using a single shared time axis for all trials after stimulus-onset alignment; Step 5 explicitly describes `input[0]` as “left edge of bin i = -0.5 + 0.02*i”.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The input is not read from a raw time series. It is generated by constructing a 100-element evenly spaced grid of bin left edges using the global trial window and bin size, then broadcasting that same vector to every retained trial.

ii. 
```python
BINSIZE = PARAMS['binsize']
WIN = PARAMS['time_window']
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))          # 100
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)        # -0.50 ... 1.48
```

```python
inputs = np.empty((n_kept, len(INPUT_NAMES), NBINS), dtype=np.float32)
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. The notes present this as a design choice: once all modalities are aligned to `stimOn_times`, the time input can be represented by the common decoder grid itself.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI aligns the time input by using the same trial window and number of bins as the neural data. However, it stores bin left edges, while spike counts represent activity accumulated over each 20 ms interval.

ii. 
```python
interval_begs = align_times + WIN[0]
...
binned = bin_spikes(..., interval_begs[mask])
```

```python
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. The notes describe `input[0]` as the shared decoder time axis and emphasize that all streams use the same `stimOn_times` alignment and 100-bin window.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` column. A change in `probabilityLeft` marks the start of a new block.

ii. 
```python
def trial_number_in_block(probability_left):
    """0-based index of each trial within its constant-``probabilityLeft`` block."""
    p = np.asarray(probability_left, dtype=float)
```

```python
block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. The notes justify this by stating that the trials table has no explicit block ID, so block structure has to be reconstructed from `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a 0-based within-block counter over the full trials table before exclusions. The count resets whenever `probabilityLeft` changes, and the resulting scalar is broadcast across all 100 time bins of each retained trial.

ii. 
```python
out = np.zeros(len(p), dtype=np.int64)
count = 0
for i in range(len(p)):
    if i > 0 and not (p[i] == p[i - 1] or (np.isnan(p[i]) and np.isnan(p[i - 1]))):
        count = 0
    out[i] = count
    count += 1
return out
```

```python
inputs[:, 1, :] = block_idx[mask][:, None]
```

iii. Step 5 of the notes explicitly says this is computed before any trial exclusion so the value reflects the animal’s true position within the behavioral block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `Choice` comes from `trials['choice']`.

ii. 
```python
choice_raw = trials['choice'].to_numpy(dtype=float)
mask &= np.isin(choice_raw, (-1.0, 1.0))
```

```python
choice_out = ((1 - choice_raw[mask]) / 2).astype(np.int64)
```

iii. The notes say the IBL sign convention was checked empirically: `choice == +1` means left and `choice == -1` means right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI first excludes non-binary or missing choices, then recodes the kept values from IBL’s `+1/-1` convention to decoder labels `0/1` via `(1 - choice) / 2`. The per-trial label is broadcast across all 100 bins.

ii. 
```python
mask &= np.isin(choice_raw, (-1.0, 1.0))
...
choice_out = ((1 - choice_raw[mask]) / 2).astype(np.int64)      # +1 -> 0 (left)
...
outputs[:, 0, :] = choice_out[:, None]
```

iii. The notes justify the mapping as required by the task (`left = 0`, `right = 1`) and supported by explicit spot-checks against stimulus side on correct trials.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials['probabilityLeft']`.

ii. 
```python
prior_code = map_prior(trials['probabilityLeft'].to_numpy())
```

```python
def map_prior(probability_left):
    """{0.2, 0.5, 0.8} -> {0, 1, 2}; anything else -> -1 (trial dropped)."""
```

iii. The notes say the task’s prior is exactly the block prior encoded by `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, rejects any trial whose `probabilityLeft` is not one of those values, and broadcasts the categorical code across time bins.

ii. 
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
...
for value, code in PRIOR_MAP.items():
    out[np.isclose(p, value)] = code
```

```python
mask &= prior_code >= 0
prior_out = prior_code[mask]
...
outputs[:, 1, :] = prior_out[:, None]
```

iii. The notes justify this as a direct task-required recoding of the three block priors.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The AI uses the wheel stream loaded by `SessionLoader.load_wheel()`, specifically the wheel timestamps and the absolute value of the derived wheel velocity.

ii. 
```python
def load_wheel_speed(sess_loader):
    """|wheel velocity| -- ``load_target_behavior(one, eid, 'wheel-speed')``."""
    if sess_loader.wheel is None or len(sess_loader.wheel) == 0:
        sess_loader.load_wheel()
    return (sess_loader.wheel['times'].to_numpy(dtype=float),
            np.abs(sess_loader.wheel['velocity'].to_numpy(dtype=float)))
```

```python
wt, wv = load_wheel_speed(sess_loader)
```

iii. The notes justify this as matching the reference behavior loader: wheel speed is `abs(velocity)` after `SessionLoader` performs the standard wheel interpolation/filtering.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI takes `abs(velocity)`, resamples it onto the common 100-bin trial grid with the reference coverage checks, then discretizes all retained values within a session into three equal-occupancy bins using the 33.3 and 66.7 percentiles.

ii. 
```python
wheel_vals, wheel_ok = behavior_per_interval(wt, wv, interval_begs)
...
wheel_kept = wheel_vals[mask]
wheel_bin, wheel_edges = discretize_tertiles(wheel_kept)
```

```python
def discretize_tertiles(values):
    ...
    edges = np.percentile(flat, [100.0 / N_DISCRETE_BINS * i
                                 for i in range(1, N_DISCRETE_BINS)])
    binned = np.searchsorted(edges, flat, side='right').reshape(values.shape)
    return binned.astype(np.int64), edges
```

iii. The notes justify session-wise tertiles because wheel speed is session-specific and skewed, so a global threshold would collapse some sessions into one class. They also say this mirrors the spirit of the reference decoder’s per-session scaling.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded by taking tertiles over all retained wheel-speed samples in a session. The two percentile cutoffs define three categories `low`, `medium`, and `high`. If the tertile edges collapse, the whole session is rejected as degenerate.

ii. 
```python
wheel_bin, wheel_edges = discretize_tertiles(wheel_kept)
...
for name, edges in (('wheel speed', wheel_edges), ('whisker motion energy', me_edges)):
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f'{name} is degenerate (tertile edges {list(edges)})')
```

```python
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]
```

iii. Step 5 and Step 10 of the notes justify this as equal-occupancy discretization plus a later fix: sessions with tied tertile edges were dropped because they produced broken, non-varying outputs.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI aligns wheel speed to the same stimulus-onset trial windows as the neural data, but samples the continuous trace at the bin right edges rather than storing bin-center or bin-left-edge values.

ii. 
```python
# Query grid = bin right edges, exactly linspace(beg + binsize, end, NBINS).
q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
interp = np.interp(q.ravel(), target_times, target_vals).reshape(q.shape)
```

```python
wheel_vals, wheel_ok = behavior_per_interval(wt, wv, interval_begs)
...
outputs[:, 2, :] = wheel_bin
```

iii. The notes say this follows `get_behavior_per_interval` in the reference code literally, including its coverage checks and query grid on the bin right edges.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from camera frame times and whisker motion energy values from the side camera, preferring the left camera and falling back to the right camera if needed.

ii. 
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[key]
        t = df['times'].to_numpy(dtype=float)
        v = df['whiskerMotionEnergy'].to_numpy(dtype=float)
```

```python
mt, mv, me_view = load_whisker_me(sess_loader)
```

iii. The notes justify this as matching the reference `bin_behaviors` preference order: left camera first, right camera as fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released whisker motion energy trace as-is, applies the same interval coverage checks and resampling routine as for wheel speed, and then discretizes the retained values into within-session tertiles.

ii. 
```python
me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
...
me_kept = me_vals[mask]
me_bin, me_edges = discretize_tertiles(me_kept)
```

```python
outputs[:, 3, :] = me_bin
```

iii. The notes explicitly say there is no extra filtering or normalization beyond using the released whisker motion energy stream and the common resampling/discretization pipeline.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly like wheel speed: session-wise tertiles over all retained trial-by-time samples, producing `low`, `medium`, and `high`, with degenerate sessions rejected if the two edges are tied.

ii. 
```python
me_bin, me_edges = discretize_tertiles(me_kept)
...
for name, edges in (('wheel speed', wheel_edges), ('whisker motion energy', me_edges)):
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f'{name} is degenerate (tertile edges {list(edges)})')
```

iii. The notes justify this with the same argument as wheel speed and document a concrete broken-whisker-ROI session that motivated the degenerate-edge rejection.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to the same stimulus-onset trial windows as the neural data and resampled at the bin right edges of those windows.

ii. 
```python
q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
interp = np.interp(q.ravel(), target_times, target_vals).reshape(q.shape)
```

```python
mt, mv, me_view = load_whisker_me(sess_loader)
me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
```

iii. The notes justify this as a direct reimplementation of the reference behavior-alignment helper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles missing or bad data by exclusion rather than imputation. It skips empty spike-sorting loads, drops trials that fail trial QC or lack full wheel/whisker/ephys coverage, rejects sessions with too few good neurons or too few usable trials, and rejects sessions with degenerate tertile thresholds. It also records skipped-session reasons in metadata.

ii. 
```python
if len(sp) == 0 or 'times' not in sp or len(sp['times']) == 0:
    continue
...
if not spikes_list:
    raise RuntimeError('no spike sorting available')
```

```python
mask &= wheel_ok & me_ok
n_trials_kept = int(mask.sum())
if n_trials_kept < MIN_TRIALS:
    raise RuntimeError(f'only {n_trials_kept} usable trials')
```

```python
if len(keep_idx) < MIN_NEURONS:
    raise RuntimeError(f'only {len(keep_idx)} well-isolated grey-matter units')
...
if not np.all(np.diff(edges) > 0):
    raise RuntimeError(f'{name} is degenerate (tertile edges {list(edges)})')
```

```python
'failed_sessions': [{'eid': e, 'reason': m} for e, m in failures],
```

iii. The notes justify these exclusions as preventing silent corruption. Step 10 specifically documents fixes for trials outside the ephys recording and for degenerate whisker-motion sessions instead of filling missing values.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike-sorting I/O as the dominant cost, with spike loading much slower than trial loading, behavior loading, or binning. The notes also say the real bottleneck is overlapping NFS reads from the large spike arrays.

ii. 
```python
sp, cl, ch = ssl.load_spike_sorting()
```

```python
t = time.time()
spikes, clusters, rec_span = load_session_spikes(eid, pids, probe_names)
timing['load_spikes'] = time.time() - t
```

iii. In Step 6 and Step 7, the notes explicitly say spike sorting is the expensive stage and provide timing estimates showing `load_spikes` dominating per-session runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the major loops relative to the reference code. The remaining obvious Python loops are the per-trial loop inside `bin_spikes` and the per-trial loop in `trial_number_in_block`; the behavior interpolation has already been vectorized across all trials.

ii. 
```python
for k in range(n_trials):
    if not np.isfinite(begs[k]) or i1[k] <= i0[k]:
        continue
    tt = spike_times[i0[k]:i1[k]]
    uu = spike_units[i0[k]:i1[k]]
    b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
```

```python
for i in range(len(p)):
    if i > 0 and not (p[i] == p[i - 1] or (np.isnan(p[i]) and np.isnan(p[i - 1]))):
        count = 0
    out[i] = count
    count += 1
```

iii. The notes argue that the reference’s per-trial multiprocessing and per-trial interpolator construction were the main vectorization opportunities, and that the AI already removed those bottlenecks by moving to session-level vectorized search/interpolation.

## 10-c. What processing does the code repeat multiple times?

i. The AI does not repeat any major scientific preprocessing step. The remaining repetition is minor bookkeeping, such as refiltering `bwm` by `eid` inside the job-construction loop and converting `ref_mask` to NumPy in more than one place.

ii. 
```python
for i, eid in enumerate(eids):
    sub = bwm[bwm.eid == eid]
    ...
```

```python
mask = ref_mask.to_numpy().astype(bool)
...
'n_trials_ref_mask': int(ref_mask.to_numpy().sum()),
```

iii. The notes frame the conversion as intentionally avoiding repeated work by loading each session once, vectorizing the heavy transforms, and processing sessions independently.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI performs extra diagnostics and provenance work that the decoder itself does not need: optional `plot_processing` figures, detailed timing measurements, and metadata such as cluster UUIDs, retained raw-trial indices, tertile edges, and session-level summary stats.

ii. 
```python
timing = {}
...
timing['load_spikes'] = time.time() - t
...
timing['total'] = time.time() - t_start
```

```python
if show_processing:
    try:
        plot_processing(out_dir, eid, trials, mask, spikes, su, sel, keep_idx,
                        interval_begs, wt, wv, mt, mv, wheel_kept, me_kept,
                        wheel_bin, me_bin, wheel_edges, me_edges, binned,
                        inputs, outputs, me_view)
```

```python
'kept_trial_idx': np.nonzero(mask)[0].astype(np.int32),
'cluster_uuids': list(clusters['uuids'].to_numpy()[keep_idx]),
'wheel_speed_tertile_edges': [float(x) for x in wheel_edges],
'whisker_me_tertile_edges': [float(x) for x in me_edges],
'mean_firing_rate_hz': float(binned.mean() / BINSIZE),
```

iii. The notes justify these additions as validation and provenance rather than part of the actual converted decoder inputs/outputs.
