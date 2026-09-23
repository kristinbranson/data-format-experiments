# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the release inventory from `/app/code/code_zhang2025/data/bwm_release.csv`, optionally restricts it with `/app/data/DATALIMIT_SUBSET.csv`, and then loads every session through the ONE API. For each `eid` it uses `SessionLoader` for trials/wheel/motion energy and `SpikeSortingLoader` for each `(pid, probe_name)` listed for that session.

ii. ```python
bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))

def get_one():
    from one.api import ONE
    return ONE()

sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
sl.load_wheel()
sl.load_motion_energy(views=[view])

ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI says explicit `tables_dir` / local loading missed revisioned datasets, so bare `ONE()` was the only reliable way to resolve the staged cache. It also treats `bwm_release.csv` from the reference repo as the authoritative release list.

## 1-b. How are the data split into subjects?

i. Subjects come from the `subject` column of `bwm_release.csv`. After session filtering, the AI builds the sorted unique subject list and stores `subject_idx` for each kept session.

ii. ```python
subj_of = dict(zip(bwm.eid, bwm.subject))
subjects = sorted({subj_of[e] for e in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[subj_of[e]] for e in kept], dtype=np.int64)
```

iii. The notes say the release CSV already contains authoritative `(pid, eid, probe_name, subject, lab)` metadata, so no extra subject parsing was needed.

## 1-c. How are the data split into sessions?

i. Sessions are unique `eid` values from the release CSV. The script preserves release-file order and processes one task per unique `eid`.

ii. ```python
eids = list(dict.fromkeys(bwm.eid.tolist()))
tasks = [(e, by_eid[e], i < n_show) for i, e in enumerate(eids)]
```

iii. The AI treats the release file as already session-organized, so a session is one `eid`.

## 1-d. How are the data split into trials?

i. Trials come directly from the IBL trials table loaded by `SessionLoader`; each row is treated as one trial and masks decide which rows survive.

ii. ```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
```

iii. The AI follows the standard IBL assumption that the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI first applies a query-based trial mask modeled on `load_trials_and_mask`: reaction time 0.08-2.0 s, `feedback_times - goCue_times <= 10`, required fields non-null, and `choice != 0`. It then intersects this with wheel coverage, whisker coverage, and an added spike-coverage mask so all kept trials are fully covered by every stream.

ii. ```python
def build_trials_mask(trials):
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    return np.asarray(~trials.eval(query), dtype=bool).copy()
...
keep_trial = trial_mask & ws_ok & wm_ok
...
trial_mask &= spk_ok
```

iii. The notes say this was meant to literalize the reference trial mask and behavior coverage checks, then add spike coverage after the AI found real all-zero neural trials caused by recording gaps or spikes ending before behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from spike times and spike cluster assignments loaded per probe. Cluster acronyms and QC labels are used only to decide which units survive and how they are labeled.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
spk_times.append(np.asarray(spikes['times'], dtype=np.float64))
spk_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
acronyms.append(cdf['acronym'].to_numpy().astype(str))
labels.append(cdf['label'].to_numpy(dtype=np.float64))
```

iii. The notes repeatedly describe `spikes.times` and `spikes.clusters` as the core neural variables, with cluster metadata used for curation/regions only.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, reindexes cluster IDs so they are unique, sorts spikes by time, and bins them into 20 ms trial-relative bins. It stores raw spike counts per bin, not firing rates, as `(n_neurons, 100)` float32 arrays.

ii. ```python
def bin_spikes_trials(spike_times, spike_clusters, n_clusters, t_beg):
    out = np.zeros((n_trials, n_clusters, NBINS), dtype=np.float32)
    ...
    for k in range(n_trials):
        bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
        np.clip(bi, 0, NBINS - 1, out=bi)
        ci = spike_clusters[a:b]
        np.add.at(flat[k], ci * NBINS + bi, 1.0)
    return out
...
neural.append(binned[j])
```

iii. In Step 5 of the notes, the AI explicitly says “Spike counts, not rates, un-normalised” and claims this matches the cached reference representation and avoids unnecessary normalization before the decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1` and Beryl region not in `('root', 'void')`. It remaps spikes onto surviving units and later drops sessions with fewer than 5 surviving neurons.

ii. ```python
QC_LABEL = 1.0
NON_GREY = ('root', 'void')
...
beryl = np.asarray(br.acronym2acronym(acronyms, mapping='Beryl')).astype(str)
keep_unit = (labels >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
...
MIN_NEURONS_PER_SESSION = 5
if results[e]['n_neurons'] < MIN_NEURONS_PER_SESSION:
    results[e]['skip'] = ...
```

iii. The notes justify this as following the data paper’s well-isolated-neuron and grey-matter criteria rather than the methods code’s `qc=None`, and adapt the paper’s minimum-neuron rule from region level to session level.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to stimulus onset by starting each trial window at `stimOn_times - 0.5` and binning spikes relative to that aligned window. The neural window runs from -0.5 to +1.5 s around stimulus onset.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
t_beg = align + TIME_WINDOW[0]
bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
```

iii. The notes cite both the decoder task and the reference code for this alignment/window choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins (`BINSIZE = 0.02`) and 100 bins per 2 s trial. There is no additional temporal rebinning beyond binning onto that target grid.

ii. ```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

iii. The notes say this follows the reference pipeline parameters exactly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. `input[0]` is derived from the stimulus-onset alignment variable `trials.stimOn_times` plus the fixed aligned trial window. After alignment, the stored input is a canonical within-trial time axis rather than a raw session-clock trace.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
t_beg = align + TIME_WINDOW[0]
```

iii. The AI’s notes describe this input as the representative time of each aligned trial bin.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes `time_from_stim_on` as the right edge of each 20 ms bin: `-0.48, -0.46, ..., 1.50` seconds. The same 100-value vector is written into every trial.

ii. ```python
time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).astype(np.float32)
inp = np.empty((2, NBINS), dtype=np.float32)
inp[0] = time_axis
```

iii. The notes and trajectory say this was chosen because the reference behavior interpolation evaluates traces at bin right edges, so using the same representative time was considered cleaner.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. `time_from_stim_on` is aligned to the neural data by sharing the same 100-bin aligned trial grid used for neural binning. The representative time is the right edge of each neural bin.

ii. ```python
bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).astype(np.float32)
inp[0] = time_axis
```

iii. The AI justifies this as keeping every time-varying stream on the same aligned 20 ms grid, with only the representative point within the bin chosen differently from the human reference.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `input[1]` comes from `trials.probabilityLeft`. The AI infers block boundaries from changes in that value and counts trial position from each change point.

ii. ```python
def trial_number_in_block(prob_left):
    prev, cur = prob_left[:-1], prob_left[1:]
    is_new[1:] = ~((prev == cur) | (np.isnan(prev) & np.isnan(cur)))
    block_start = np.maximum.accumulate(np.where(is_new, np.arange(n), 0))
    return np.arange(n) - block_start
```

iii. The notes state that there is no explicit block-id field, so blocks must be recovered from runs of constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a 0-based within-block count before trial filtering, divides it by 100, and broadcasts the resulting scalar across all 100 time bins of that trial.

ii. ```python
TRIAL_IN_BLOCK_SCALE = 100.0
tib = trial_number_in_block(p_left).astype(np.float32) / TRIAL_IN_BLOCK_SCALE
inp[1] = tib[k]
```

iii. The notes/trajectory say the `/100` scaling was added to keep the feature O(1) for the downstream linear layer.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `output[0]` comes from the trials-table `choice` column.

ii. ```python
choice = trials['choice'].to_numpy(dtype=np.float64)
```

iii. The AI checked the sign convention against other task fields, then used `choice` directly.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI recodes `choice == +1` to class 0 (`left`) and `choice == -1` to class 1 (`right`), then broadcasts that class across the whole trial. `choice == 0` trials are removed upstream.

ii. ```python
choice_cls = (choice < 0).astype(np.int64)
out[0] = choice_cls[k]
```

iii. The notes say this mapping was chosen to match the decoder-task requirement `left = 0, right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. `output[1]` comes from the trials-table `probabilityLeft` column.

ii. ```python
p_left = trials['probabilityLeft'].to_numpy(dtype=np.float64)
```

iii. The AI treats this as the released block-prior variable and keeps the unbiased `0.5` trials as their own class.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, broadcasts the resulting class across the trial, and drops trials that do not map to one of those three values.

ii. ```python
prior_cls = np.full(len(p_left), -1, dtype=np.int64)
prior_cls[np.isclose(p_left, 0.2)] = 0
prior_cls[np.isclose(p_left, 0.5)] = 1
prior_cls[np.isclose(p_left, 0.8)] = 2
keep_trial = keep_trial & (prior_cls >= 0)
out[1] = prior_cls[k]
```

iii. The notes say this mapping was required by the decoder task and that keeping `0.5` trials was necessary because they form one of the required prior classes.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel data loaded by `SessionLoader.load_wheel()`, specifically the absolute value of the loader-computed wheel velocity trace.

ii. ```python
sl.load_wheel()
wheel_t = sl.wheel['times'].to_numpy(dtype=np.float64)
wheel_v = np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64))
```

iii. The AI relies on `SessionLoader` for the standard wheel preprocessing and then takes `abs(velocity)`, matching the reference variable choice.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. After loading wheel velocity, the AI drops non-finite samples and linearly interpolates absolute velocity onto each trial’s aligned 100-bin grid.

ii. ```python
gw = np.isfinite(wheel_t) & np.isfinite(wheel_v)
wheel_t, wheel_v = wheel_t[gw], wheel_v[gw]
ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)
```

iii. The notes say this mirrors `get_behavior_per_interval`, but replaces the reference per-trial pool with vectorized `np.interp`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is thresholded into three classes by computing session-specific 33.3rd and 66.7th percentile cuts over all kept trial bins in that session.

ii. ```python
def tertile_bins(values, mask):
    pool = values[mask].ravel()
    pool = pool[np.isfinite(pool)]
    lo, hi = np.percentile(pool, [100.0 / 3.0, 200.0 / 3.0])
    cls = (values > lo).astype(np.int64) + (values > hi).astype(np.int64)
    return cls, (float(lo), float(hi))
...
ws_cls, ws_thr = tertile_bins(ws_vals, keep_trial)
```

iii. The notes justify this by saying wheel-speed scale is heavily session-dependent and right-skewed, so per-session tertiles give more stable low/medium/high categories than fixed thresholds.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to neural data by interpolation onto the same aligned 100-bin trial grid. The AI evaluates wheel speed at bin right edges, so the streams share bins but not a bin-center representation.

ii. ```python
def interp_behavior(times, values, t_beg):
    grid = BINSIZE * np.arange(1, NBINS + 1)
    xi = t_beg[:, None] + grid[None, :]
    vals = np.interp(safe, times, values).astype(np.float32)
    return vals, ok
```

iii. The AI explicitly says it chose right-edge evaluation to match the reference behavior interpolation routine.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy comes from the side-camera motion-energy trace loaded by `SessionLoader.load_motion_energy`. The AI prefers the left camera and falls back to the right camera if needed, then uses the `whiskerMotionEnergy` column.

ii. ```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        me = sl.motion_energy[key]
        wt = me['times'].to_numpy(dtype=np.float64)
        wv = me['whiskerMotionEnergy'].to_numpy(dtype=np.float64)
        whisker_t, whisker_v, whisker_view = wt[good], wv[good], view
        break
```

iii. The notes say this matches the reference left-then-right fallback policy.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released whisker-motion-energy trace as-is apart from dropping non-finite samples, then linearly interpolates it onto each trial’s aligned 100-bin grid. No extra filtering or normalization is applied before discretization.

ii. ```python
wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
```

iii. The notes say the released motion-energy variable is used directly, with only alignment/interpolation plus later discretization added for the decoder task.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is thresholded with the same per-session tertile procedure used for wheel speed.

ii. ```python
wm_cls, wm_thr = tertile_bins(wm_vals, keep_trial)
```

iii. The AI uses the same session-specific low/medium/high rationale as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to neural data by interpolation onto the same aligned 100-bin trial grid, again using bin right edges as the representative time within each neural bin.

ii. ```python
wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
```

iii. The notes say the camera trace shares the same session clock and only needs interpolation onto the aligned trial grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles minor data problems by dropping the affected trials or sessions. It drops spikes with NaN times, drops non-finite wheel/whisker samples before interpolation, skips sessions with missing wheel/whisker/spike data, and uses coverage checks so incomplete wheel/whisker/spike windows do not silently become usable trials.

ii. ```python
finite = np.isfinite(spk_times)
spk_times, spk_clusters = spk_times[finite], spk_clusters[finite]
...
if whisker_t is None:
    return {'eid': eid, 'skip': 'no whisker motion energy'}
...
gw = np.isfinite(wheel_t) & np.isfinite(wheel_v)
wheel_t, wheel_v = wheel_t[gw], wheel_v[gw]
...
keep_trial = trial_mask & ws_ok & wm_ok
trial_mask &= spk_ok
```

iii. The notes emphasize dropping missing/incomplete data rather than filling it in, and specifically document the spike-coverage mask as a fix for real all-zero trials.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike-sorting load as the dominant time cost, with per-session timings around 2-6 s for spike loading versus 0.3-1 s for binning and 0.5-1 s for behavior load. It then parallelizes over sessions to reduce wall-clock time.

ii. ```python
timings['spikes'] = time.time() - t0
...
timings['behavior'] = time.time() - t0
...
timings['binning'] = time.time() - t0
```

iii. Step 7 of the notes gives a timing table and says spike sorting load is the heaviest step.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI says the main vectorization targets were the reference code’s per-trial spike-binning and per-trial behavior-interpolation loops, and it rewrote both at session scope. In its own code, the remaining obvious loops are the per-trial spike accumulation loop and the final per-trial assembly loop.

ii. ```python
for k in range(n_trials):
    ...
    np.add.at(flat[k], ci * NBINS + bi, 1.0)
...
for j, k in enumerate(kt):
    neural.append(binned[j])
    outputs.append(out)
```

iii. The notes explicitly present these changes as the vectorized replacements for the reference’s heavier nested per-trial multiprocessing.

## 10-c. What processing does the code repeat multiple times?

i. The AI tries to avoid repeated processing, but some work still repeats: each worker constructs its own `ONE()` and `BrainRegions()` objects per session, each trial is still looped over once inside `bin_spikes_trials`, and motion-energy loading tries left then right camera in sequence.

ii. ```python
def get_one():
    from one.api import ONE
    return ONE()
...
br = BrainRegions()
...
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
```

iii. The notes frame most repeated work as already reduced relative to the reference, but these repeated setup/lookup steps remain.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does some extra work that the downstream decoder does not need: timing/provenance bookkeeping, storing per-session thresholds and kept-trial indices in metadata, and, in `--show-processing` mode, building a large `_plot` payload only to render figures and then discard it.

ii. ```python
res = {
    ...
    'kept_trial_idx': kt.astype(np.int32),
    'wheel_thresholds': ws_thr,
    'whisker_thresholds': wm_thr,
    'timings': timings,
    'total_time': time.time() - t_start,
}
if show_processing:
    res['_plot'] = {...}
...
res.pop('_plot', None)
```

iii. The notes present these as diagnostics/provenance additions made during development rather than data strictly required for decoder training.
