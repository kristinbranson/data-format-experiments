# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API to access data from the IBL ONE cache. However, instead of using `one.search()` to discover sessions (as the reference does), it reads `bwm_release.csv` to get the authoritative list of sessions and probes. For each session, it uses `SessionLoader` for trials/wheel/motion energy and `SpikeSortingLoader` for spikes. The ONE client is instantiated with `ONE()` (no arguments), which reads from `~/.one/.caches`.

ii.
```python
bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
# ...
one = ONE()
# ...
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The AI documented that using `ONE()` with no arguments was critical because it reads `~/.one/.caches` which points to the correct cache_dir and resolves revisioned datasets properly. Using `bwm_release.csv` gives a known list of (eid, pid, probe_name, subject, lab) rather than discovering them via search.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. At assembly, subjects are the sorted unique names from the kept sessions, and `subject_idx` maps each session to its index.

ii.
```python
subj_of = dict(zip(bwm.eid, bwm.subject))
# ...
subjects = sorted({subj_of[e] for e in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
```

iii. The AI notes that `bwm_release.csv` provides the subject name directly alongside each eid and probe, so no parsing is needed.

## 1-c. How are the data split into sessions?

i. Sessions are identified by their unique `eid` from `bwm_release.csv`. Each eid represents one session. Probes within a session are grouped by eid.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))   # preserve file order, unique
by_eid = {e: list(g[['pid', 'probe_name']].itertuples(index=False, name=None))
          for e, g in bwm.groupby('eid')}
```

iii. The session is the natural unit of the release; no further splitting is needed.

## 1-d. How are the data split into trials?

i. The trials table (loaded via `SessionLoader.load_trials()`) has one row per trial. Trials are iterated over after filtering.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
```

iii. The trials table naturally provides one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI implements the full `load_trials_and_mask` from the reference code with these criteria: (1) reaction time between 0.08 and 2.0 s, (2) feedback_times - goCue_times <= 10.0 s, (3) no NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType, (4) choice != 0, (5) wheel and whisker trace coverage of the trial window, and (6) spike coverage of the trial window (an addition beyond the reference).

ii.
```python
def build_trials_mask(trials):
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    return np.asarray(~trials.eval(query), dtype=bool).copy()
```
Plus behavior and spike coverage:
```python
ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)
wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
keep_trial = trial_mask & ws_ok & wm_ok
# spike coverage also applied
trial_mask &= spk_ok
```

iii. The AI says this is a literal transcription of the reference `load_trials_and_mask(min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True)`. The spike coverage was added to fix all-zero neural trials found during validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe, loaded via `SpikeSortingLoader`. The cluster table provides quality labels and anatomical locations for filtering.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
cdf = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
spk_times.append(np.asarray(spikes['times'], dtype=np.float64))
spk_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
```

iii. Spike times and cluster assignments are directly available through the loader.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window (-0.5 to 1.5 s around stimulus onset), giving raw spike counts per unit per bin. The counts are stored as float32 but are **not** divided by the bin width -- the AI stores raw counts, not firing rates. When a session has two probes, their units are pooled into one population with offset cluster IDs.

ii.
```python
def bin_spikes_trials(spike_times, spike_clusters, n_clusters, t_beg):
    # ...
    bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
    np.clip(bi, 0, NBINS - 1, out=bi)
    ci = spike_clusters[a:b]
    np.add.at(flat[k], ci * NBINS + bi, 1.0)
    return out  # raw counts, not divided by BINSIZE
```
Assembly:
```python
neural.append(binned[j])  # stored as float32 counts
```

iii. The AI states: "Spike counts, not rates, un-normalised. The reference caches raw counts; the decoder does its own SVD/PCA. Counts keep the data integral-valued and sparse." The AI refers to the zhang2025 reference code's `bin_spiking_data`, which returns counts from `bincount2D`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (well-isolated neurons) are kept. Additionally, units whose Beryl acronym is either `root` or `void` are dropped. The AI drops both `root` and `void`, treating them as non-grey-matter.

ii.
```python
NON_GREY = ('root', 'void')
# ...
beryl = np.asarray(br.acronym2acronym(acronyms, mapping='Beryl')).astype(str)
keep_unit = (labels >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
```

iii. The AI justifies dropping both root and void based on the data paper's statement that "Final analyses were additionally restricted to regions that were designated grey matter." The AI notes this keeps 65,301 of 75,708 well-isolated units (dropping ~10,407 root/void units). They explicitly chose this over the reference code's `qc=None` approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial windows are defined as `stimOn_times + TIME_WINDOW[0]` to `stimOn_times + TIME_WINDOW[1]`. Spike times within each window are binned relative to the window start. All streams share the same session clock.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
t_beg = align + TIME_WINDOW[0]
# In bin_spikes_trials:
bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
```

iii. The AI notes that all streams are on the same session clock (synchronized upstream through a sync channel), so alignment is just a subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins total covering the 2 s window. No rebinning or interpolation is applied to the spike data.

ii.
```python
BINSIZE = 0.02                 # seconds
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

iii. The AI follows the reference code's `params['binsize'] = 0.02` and notes that both papers use 20 ms bins with T=100.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table. The input is a time axis computed as the **right edges** of the 100 bins: `TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)`, giving values from -0.48 to +1.50 s.

ii.
```python
time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The AI explains this matches the reference `get_behavior_per_interval` which evaluates at `np.linspace(interval_beg + binsize, interval_end, n_bins)` -- i.e., bin right edges.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond computing the right-edge time axis. The same 100-element vector is used for every trial.

ii.
```python
time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).astype(np.float32)
# ...
inp[0] = time_axis
```

iii. The AI justifies the right-edge convention by matching the reference code's behavior interpolation grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time axis represents the right edges of the same bins used for neural binning. Each bin covers `[stimOn + TIME_WINDOW[0] + i*BINSIZE, stimOn + TIME_WINDOW[0] + (i+1)*BINSIZE)`, and the time input for that bin is the right edge `stimOn + TIME_WINDOW[0] + (i+1)*BINSIZE`.

ii.
```python
# Neural bins cover:
bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
# Time input is right edge of the same bins:
time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The time input and neural bins share the same grid structure, differing only in which point within each bin is used as the representative time.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A new block starts whenever `probabilityLeft` changes value.

ii.
```python
def trial_number_in_block(prob_left):
    is_new[1:] = ~((prev == cur) | (np.isnan(prev) & np.isnan(cur)))
    block_start = np.maximum.accumulate(np.where(is_new, np.arange(n), 0))
    return np.arange(n) - block_start
```

iii. The AI notes the trials table carries no explicit block identifier, so blocks must be recovered from transitions in probabilityLeft.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based trial index within each block is computed, then **divided by 100** (`TRIAL_IN_BLOCK_SCALE = 100.0`). This scaling brings the values to O(1), comparable to the time input. The computation is done on all trials before filtering, so the block count reflects the animal's true position.

ii.
```python
TRIAL_IN_BLOCK_SCALE = 100.0
tib = trial_number_in_block(p_left).astype(np.float32) / TRIAL_IN_BLOCK_SCALE
# ...
inp[1] = tib[k]
```

iii. The AI justifies the /100 scaling: "the decoder concatenates inputs with 100 neural PCs and feeds a linear layer with no input standardisation." Without scaling, trial_number_in_block can reach ~90, which would dominate the O(1) time input and neural PCs.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column from the trials table, which takes values +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice = trials['choice'].to_numpy(dtype=np.float64)
choice_cls = (choice < 0).astype(np.int64)
```

iii. The AI verified the sign convention: choice == +1 corresponds to a leftward report, verified against contrastLeft/contrastRight and feedbackType.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoded: +1 (left) -> 0, -1 (right) -> 1. Trials with choice == 0 are excluded by the trial mask.

ii.
```python
choice_cls = (choice < 0).astype(np.int64)  # +1 -> 0 (left), -1 -> 1 (right)
```

iii. This matches the decoder task specification: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column from the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
p_left = trials['probabilityLeft'].to_numpy(dtype=np.float64)
prior_cls = np.full(len(p_left), -1, dtype=np.int64)
prior_cls[np.isclose(p_left, 0.2)] = 0
prior_cls[np.isclose(p_left, 0.5)] = 1
prior_cls[np.isclose(p_left, 0.8)] = 2
```

iii. The three values correspond to the block structure of the experiment. The AI uses `np.isclose` rather than exact equality for robustness.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Trials with unknown probabilityLeft (prior_cls == -1) are dropped.

ii.
```python
keep_trial = keep_trial & (prior_cls >= 0)
```

iii. Matches the decoder task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel position and timestamps (`_ibl_wheel.position`, `_ibl_wheel.timestamps`), loaded via `SessionLoader.load_wheel()`. The velocity is computed internally by the loader, and speed is the absolute value.

ii.
```python
sl.load_wheel()
wheel_t = sl.wheel['times'].to_numpy(dtype=np.float64)
wheel_v = np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64))
```

iii. Same as the reference: `wheel-speed` = `abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) `SessionLoader.load_wheel()` internally interpolates position onto a 1000 Hz grid and differentiates with a Butterworth low-pass filter to get velocity; (2) `abs(velocity)` gives speed, which is linearly interpolated onto the **right edges** of the 100 bins of each trial; (3) per-session tertile discretization at the 33.3/66.7 percentiles.

ii.
```python
ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)
# In interp_behavior:
grid = BINSIZE * np.arange(1, NBINS + 1)  # right edges
xi = t_beg[:, None] + grid[None, :]
vals = np.interp(safe, times, values).astype(np.float32)
```

iii. The AI follows the reference `get_behavior_per_interval` which evaluates at `np.linspace(beg+binsize, end, n_bins)` = right edges.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tertile bins at the 33.3rd and 66.7th percentiles, computed over all kept time bins of the session. Values are classified as 0 (low), 1 (medium), 2 (high) using `(values > lo) + (values > hi)`.

ii.
```python
def tertile_bins(values, mask):
    pool = values[mask].ravel()
    lo, hi = np.percentile(pool, [100.0 / 3.0, 200.0 / 3.0])
    cls = (values > lo).astype(np.int64) + (values > hi).astype(np.int64)
    return cls, (float(lo), float(hi))
```

iii. The AI justifies per-session tertiles: "Wheel speed and whisker ME are strongly right-skewed and their absolute scale differs by an order of magnitude between sessions... Tertiles give three equally populated, comparable classes."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same right-edge time grid as the time input, which corresponds to the right edges of the neural bins. Both streams are on the same session clock.

ii.
```python
grid = BINSIZE * np.arange(1, NBINS + 1)
xi = t_beg[:, None] + grid[None, :]
vals = np.interp(safe, times, values)
```

iii. Aligned by evaluating at the same time points as the time input.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy from the side camera: `<side>Camera.ROIMotionEnergy` with `_ibl_<side>Camera.times`. Left camera is preferred; right is the fallback.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        me = sl.motion_energy[key]
        wt = me['times'].to_numpy(dtype=np.float64)
        wv = me['whiskerMotionEnergy'].to_numpy(dtype=np.float64)
```

iii. Same left-then-right fallback as the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is (no filtering or normalization). It is linearly interpolated onto the right-edge bin grid, then discretized into per-session tertiles.

ii.
```python
wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
wm_cls, wm_thr = tertile_bins(wm_vals, keep_trial)
```

iii. Same approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: per-session tertile bins at 33.3/66.7 percentiles using `(values > lo) + (values > hi)`.

ii.
```python
wm_cls, wm_thr = tertile_bins(wm_vals, keep_trial)
```

iii. Same rationale as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel: interpolated onto the right-edge bin grid shared by all streams.

ii.
```python
# Same interp_behavior function used for both wheel and whisker
wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
```

iii. All behavioral streams are evaluated at the same time points.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of defense: (1) NaN spike times are dropped before sorting/binning; (2) Trials with NaN in critical fields are excluded by the trial mask; (3) Trials not covered by wheel, whisker, or spike data are dropped; (4) Sessions with < 5 well-isolated grey-matter neurons are dropped; (5) Sessions with < 2 usable trials are dropped; (6) Probes with no spike sorting are skipped; (7) A constant-value behavior trace gets a degenerate but valid discretization.

ii.
```python
finite = np.isfinite(spk_times)
spk_times, spk_clusters = spk_times[finite], spk_clusters[finite]
# ...
if keep_trial.sum() < 2:
    return {'eid': eid, 'skip': f'only {int(keep_trial.sum())} usable trials'}
# ...
MIN_NEURONS_PER_SESSION = 5
```

iii. The AI documented each edge case and fix in CONVERSION_NOTES Step 10. The spike coverage check was added after finding all-zero neural matrices in specific sessions.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting from disk, which involves reading large binary files (hundreds of MB per probe). The AI uses `ProcessPoolExecutor` with 24-32 workers for session-level parallelism.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The AI reports total conversion time of ~3.5 minutes with 32 workers (459 sessions).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes_trials` function loops over trials to bin spikes. Each trial uses `searchsorted` to isolate its window and `np.add.at` to accumulate counts. This could theoretically be vectorized by offsetting all spikes into a single flat array, though the AI notes the current approach is already fast.

ii.
```python
for k in range(n_trials):
    a, b = i0[k], i1[k]
    bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
    np.add.at(flat[k], ci * NBINS + bi, 1.0)
```

iii. The AI notes this is already much faster than the reference's per-trial multiprocessing pool and contributes ~0.3-1 s per session.

## 10-c. What processing does the code repeat multiple times?

i. The ONE client is re-instantiated in each worker process (`get_one()` called once per `load_session`). Each call reads `~/.one/.caches` and creates a new ONE object.

ii.
```python
def load_session(eid, probes, show_processing=False):
    one = get_one()
```

iii. This is necessary because the ONE client cannot be shared across forked processes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes behavior interpolation (`interp_behavior`) for ALL trials (including those that will be filtered out), then applies the trial mask afterward. The reference human solution only computes behavior for already-filtered trials.

ii.
```python
ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)  # all trials
wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)  # all trials
keep_trial = trial_mask & ws_ok & wm_ok  # then filter
```

iii. This is a design choice: the coverage check from `interp_behavior` feeds back into the trial mask, so the function must be called on all trials to determine which are valid. However, the actual interpolation of discarded trials is wasted work.
