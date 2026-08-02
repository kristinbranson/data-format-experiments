# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by scanning all subdirectories under `data/` for pairs of `data_structure_*.mat` and `motionEnergy_*.mat` files. It requires both files to exist for a session to be processed. `data_structure_*.mat` files are loaded via HDF5 (h5py) and `motionEnergy_*.mat` files via `scipy.io.loadmat`. All four data subdirectories (`Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`, `DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) are scanned indiscriminately.

ii.
```python
def discover_sessions(data_root):
    sessions = {}
    for subdir in Path(data_root).iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.glob('data_structure_*.mat'):
            stem = f.stem.replace('data_structure_', '')
            sessions.setdefault(stem, {})['data_structure'] = f
        for f in subdir.glob('motionEnergy_*.mat'):
            stem = f.stem.replace('motionEnergy_', '')
            sessions.setdefault(stem, {})['motion_energy'] = f
    return sessions

def load_data_structure(path):
    ...
    with hfile as h:
        obj = h['obj']
        # reads bp, clu, traj, meta fields
    ...

def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']
```

iii. The AI documented in CONVERSION_NOTES.md that `data_structure_*.mat` are HDF5 v7.3 MATLAB files requiring h5py, while `motionEnergy_*.mat` are older format readable by scipy. The AI iterates all subdirectories without filtering by experiment type.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is inferred from the filename stem by taking the first underscore-separated token (e.g., `JEB11_2022-05-10` yields subject `JEB11`). Unique subjects are collected as sessions are processed.

ii.
```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])
```

iii. The AI noted 18 unique subjects from filename parsing (CONVERSION_NOTES Step 2). The final converted data contains 12 subjects, as sessions without required data are skipped.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` / `motionEnergy_*.mat` pair defines one session. Sessions are processed independently in sorted order by stem name. Sessions are skipped if the data structure is unreadable, motion energy is missing, there are fewer than 2 valid trials, or fewer than 10 retained neural units.

ii.
```python
keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
...
for stem in keys:
    sess = process_session(stem, sessions[stem], edges, centers)
    if sess is None:
        continue
    out_sessions.append(sess)
```

iii. The AI documented that sessions lacking motion energy or neural data are skipped. The final dataset has 33 sessions.

## 1-d. How are the data split into trials?

i. Trial count comes from `bp.Ntrials`. Valid trials are those that pass the quality filter (excluding early and ignore trials). For each valid trial, neural data is histogrammed, behavioral labels are extracted, and trajectory/motion energy data is collected.

ii.
```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    for name in ['early', 'no']:
        if name in bp:
            arr = np.array(bp[name], dtype=float).reshape(-1)
            if arr.size == n:
                mask &= (arr == 0)
    return mask
```

iii. The AI documented that early lick and ignore trials are omitted, consistent with the reference paper's methods.

## 1-e. How are trials filtered based on quality controls?

i. Trials with `bp.early == 1` or `bp.no == 1` are excluded. No additional trial quality filtering is applied (e.g., no minimum correct trial count per direction, no stim trial exclusion).

ii.
```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    for name in ['early', 'no']:
        if name in bp:
            arr = np.array(bp[name], dtype=float).reshape(-1)
            if arr.size == n:
                mask &= (arr == 0)
    return mask
```

iii. The AI noted in CONVERSION_NOTES Step 3 that "Early lick and ignore trials are omitted from analyses" per the methods. However, the reference code conditions also exclude stimulation trials (`~stim.enable`) and the reference uses more complex condition-based trial selection (e.g., correct-only for some analyses, balanced trial numbers). The AI's approach includes error trials and stimulation trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu.trialtm` (per-unit spike times) and `obj.clu.trial` (per-unit trial indices).

ii.
```python
clu_ref = obj['clu'][()].reshape(-1)[0]
clu = deref(h, clu_ref)
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    if name in clu:
        ...
        clu_out[name] = [np.array(deref(h, r)).reshape(-1) for r in ds[()].reshape(-1)]
```

iii. The AI identified `clu.trialtm` and `clu.trial` as the key neural data sources in CONVERSION_NOTES Steps 1-2.

## 2-b. How is the `neural` data processed?

i. Spike times from `clu.trialtm` are histogrammed into 75ms bins on a grid from -2.5s to 2.5s. **No go-cue alignment is performed** — the raw `trialtm` values are binned directly without subtracting the per-trial go cue event time. **No smoothing** is applied (reference code applies a causal Gaussian kernel with width 15 samples). The result is raw spike counts per bin, not firing rates.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers

def bin_spikes_for_session(obj, trial_mask, edges):
    ...
    for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
        st = np.array(st, dtype=float).reshape(-1)
        tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
        ...
        for raw_t in np.where(trial_mask)[0]:
            counts, _ = np.histogram(st[tr == raw_t], bins=edges)
            unit_trials.append(counts.astype(np.float32))
```

iii. The AI documented using a 75ms bin size based on the DLC decoding scripts. However, the reference code natively bins at 5ms (`params.dt = 1/200`) and only rebins to 75ms for DLC-specific decoding. The AI does not subtract go cue times before binning, which is a significant deviation from the reference `alignSpikes.m` which explicitly computes `trialtm_aligned = trialtm - event`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units with an estimated firing rate <= 1 Hz are excluded. Sessions with fewer than 10 retained units are skipped. No cluster quality filtering (e.g., excluding 'garbage', 'noisy' labels) is applied.

ii.
```python
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
kept_units.append(ui)
...
if len(neural) < 2 or len(kept_units) < 10:
    return None
```

iii. The AI documented in CONVERSION_NOTES that "All units with firing rates > 1 Hz were included in most analyses" per the paper. However, the reference code's `getDefaultParams.m` sets `params.lowFR = 0.5` (0.5 Hz threshold), and `findClusters.m` with `params.quality = {'all'}` excludes 'garbage', 'noisy', and 'real?' quality clusters. The AI's FR calculation method also differs — it uses total spike count divided by total recording duration, while the reference uses mean PSTH firing rate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does NOT perform explicit go-cue alignment. The raw `trialtm` values are histogrammed into a fixed grid from -2.5s to 2.5s. The reference code subtracts `bp.ev.goCue` times from each spike's `trialtm` to produce `trialtm_aligned`, then bins the aligned times.

ii.
```python
# No goCue subtraction is performed
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The AI documented "Use go cue as alignment event" as a key decision, but the code does not implement the alignment step. The reference `alignSpikes.m` explicitly does: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 75ms time bins directly, yielding 66 time bins over the -2.5s to 2.5s window. No rebinning from a finer resolution is applied.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers
```

iii. The AI documented the 75ms bin size based on DLC decoding scripts' `rez.binSize = 75`. The reference code uses 5ms bins (`params.dt = 1/200`) for the native neural representation, then the DLC decoding scripts rebin to 75ms for behavioral feature decoding. The AI chose to bin directly at 75ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time bin centers of the common time grid, not from any raw data variable. It is the same vector of time values repeated for every trial.

ii.
```python
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The AI documented "Decoder input is only time from go cue" in the mapping plan. The time values are the bin centers from -2.4625s to 2.4625s.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The bin centers are computed as `edges[:-1] + bin_size / 2` where edges span -2.5 to 2.5 in 75ms steps. The same time vector is broadcast to every trial.

ii.
```python
centers = edges[:-1] + bin_size_s / 2
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. No additional processing; the time vector is a deterministic function of the grid parameters.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same time bin centers as the neural data grid, so they share the same temporal axis by construction. However, since the neural data is not actually go-cue-aligned (see 2-d), the input label "time from go cue" is misleading.

ii.
```python
edges, centers = build_time_grid()
# Same edges used for neural binning and input time vector
```

iii. The AI assumed the shared grid ensures alignment. The misalignment issue with neural data (lacking go cue subtraction) propagates here.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.L` (left lick indicator) and `bp.R` (right lick indicator) arrays.

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The AI documented mapping `bp.L` and `bp.R` to lick direction with left=0, right=1.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Per trial, if `R > L` then direction = 1 (right), else 0 (left). The value is replicated across all time bins as a per-trial constant.

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, lick_dir[i], dtype=np.int64),
```

iii. This is consistent with the reference code's trial conditions where R and L are binary trial-type indicators.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater` array.

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. The AI documented that context is encoded in `bp.autowater`, consistent with the reference code's `getBlockNum_AltContextTask.m`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. If `autowater > 0`, context = 0 (WC); otherwise context = 1 (DR). Replicated across all time bins.

ii.
```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
...
np.full(n_bins, context[i], dtype=np.int64),
```

iii. This matches the instructions (WC=0, DR=1) and is consistent with the reference code where `autowater` indicates water-cued blocks.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` and `bp.miss` arrays.

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The AI mapped hit > miss to correct (1), else incorrect (0).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Per trial, if `hit > miss` then outcome = 1 (correct), else 0 (incorrect). Replicated across all time bins.

ii.
```python
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, outcome[i], dtype=np.int64),
```

iii. The comparison `hit > miss` works because these are binary indicators. Note that trials where both hit and miss are 0 (ignore trials) would be labeled as incorrect, but these should already be filtered out by the trial mask.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` (video trajectory data), specifically the side-view camera's feature data. The code looks for features named 'tongue', 'left_tongue', or 'right_tongue' in `traj[0].featNames`.

ii.
```python
side = traj_views[0]
side_names = [str(x) for x in side.get('featNames', [])]
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
...
tongue.append(interp_feature_velocity(ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
```

iii. The AI identified side-view camera tongue features from `traj.featNames`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial: (1) Extract x,y positions from the 3D trajectory array `ts[:, :2, feat_idx]`. (2) Get frame times and compute relative times as `(frameTimes - 0.5) - goCue`. (3) Interpolate x,y positions to the time bin centers. (4) Compute velocity as `sqrt(gradient(x)^2 + gradient(y)^2)`. (5) NaN values in gradients are replaced with 0 for tongue.

ii.
```python
def interp_feature_velocity(ts, frame_times, align_time, centers, feat_idx, tongue=False):
    arr = np.array(ts, dtype=float)
    xy = arr[:, :2, feat_idx]
    ft = np.array(frame_times, dtype=float).reshape(-1)
    rel_t = (ft - 0.5) - float(align_time)
    x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
    y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
    xv = np.gradient(x)
    yv = np.gradient(y)
    if tongue:
        xv[np.isnan(xv)] = 0
        yv[np.isnan(yv)] = 0
    return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The AI computes velocity from interpolated positions, similar to the reference paper's description of "first-order derivative of the position vector." The use of `np.gradient` on interpolated positions is reasonable but differs from the reference pipeline which computes velocity at the native frame rate before rebinning.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The threshold is the median across ALL trials and ALL time bins in the session. Values > median → 1 (high), otherwise → 0 (low).

ii.
```python
tongue = np.stack(tongue, axis=0)
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The instructions specify "50th percentile per-session threshold." The AI uses `np.nanmedian(tongue)` across the entire session (all trials × all time bins), which implements a per-session median threshold. This is consistent with the instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue trajectory is aligned to go cue by computing `rel_t = (frameTimes - 0.5) - goCue` and interpolating to the same time bin centers used for neural data.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)  # align_time = goCue
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
```

iii. The trajectory data IS go-cue-aligned (unlike the neural data), creating a potential misalignment between neural and behavioral outputs.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from the bottom-view camera's trajectory data (`traj_views[1]`), looking for features named 'top_paw', 'bottom_paw', or 'paw'.

ii.
```python
bottom = traj_views[1]
bottom_names = [str(x) for x in bottom.get('featNames', [])]
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
```

iii. Consistent with the methods text: "paws were tracked using only the bottom view."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue: interpolate x,y positions to time grid, compute gradient-based velocity. For paw, NaN handling differs: NaN velocity values are replaced with median velocity, and the median is subtracted (centering).

ii.
```python
# tongue=False path in interp_feature_velocity:
xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
xv = xv - np.nanmedian(xv)
yv = yv - np.nanmedian(yv)
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The reference paper says "Missing values were filled in with the nearest available value for all features, except for the tongue." The AI's approach of replacing NaN with median and then median-subtracting differs from the reference's nearest-value filling.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session median threshold across all trials and time bins. > median → 1, else → 0.

ii.
```python
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. Consistent with the 50th percentile per-session threshold in the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue — aligned to go cue via `(frameTimes - 0.5) - goCue` and interpolated to the common time grid.

ii.
```python
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```

iii. Go-cue aligned, same potential misalignment with neural data as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from `me.data` in the `motionEnergy_*.mat` files, loaded via `scipy.io.loadmat`.

ii.
```python
def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']
```

iii. The AI identified motion energy files as per-trial variable-length traces.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each trial's motion energy trace is rebinned to the target number of time bins using linear interpolation between normalized [0, 1] axes. **No go-cue temporal alignment** is performed — the trace is simply stretched/compressed to fit the target bin count.

ii.
```python
def rebin_variable_trace(values, n_bins):
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    xp = np.linspace(0, 1, arr.size)
    xnew = np.linspace(0, 1, n_bins)
    return np.interp(xnew, xp, arr).astype(np.float32)
```

iii. The reference code (`loadMotionEnergy.m`) aligns motion energy to the go cue using frame times: `interp1(frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)`. The AI's approach simply maps the raw trace to the target bins without temporal alignment, which is a significant deviation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session median threshold across all trials and time bins. > median → 1, else → 0.

ii.
```python
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. The instructions say 50th percentile per-session. The reference code uses a manually set `moveThresh` value, but the instructions override this. The AI follows the instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is NOT aligned to go cue. It is simply rebinned from its native length to the target number of bins without temporal registration.

ii.
```python
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The reference code uses `interp1(frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` for proper temporal alignment. The AI's approach lacks this alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- Sessions with unreadable HDF5 files are skipped.
- Sessions without motion energy files are skipped.
- Sessions with missing `trialtm`/`trial` fields are skipped.
- Sessions with fewer than 10 units or 2 trials are skipped.
- For tongue velocity, NaN values from out-of-range interpolation are replaced with 0.
- For paw velocity, NaN values are replaced with median, then median-subtracted.
- For motion energy, empty traces yield zero arrays.
- Missing frame times are synthesized as `np.arange(1, n+1) / 400.0`.

ii.
```python
if obj is None:
    return None
if 'motion_energy' not in files:
    return None
if len(neural) < 2 or len(kept_units) < 10:
    return None
# For tongue NaN handling:
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
# For paw NaN handling:
xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
```

iii. The AI documented session skipping in CONVERSION_NOTES. The NaN handling approaches are ad-hoc and differ from the reference code's `fillmissing(..., 'nearest')` strategy.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the spike binning loop in `bin_spikes_for_session`, which iterates over every unit and every valid trial, calling `np.histogram` for each combination. With ~100+ units and ~200-400 trials per session, this creates tens of thousands of histogram calls.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The AI noted timing information in CONVERSION_NOTES but did not provide detailed profiling results.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner trial loop in `bin_spikes_for_session` could be vectorized using array operations instead of per-trial histogram calls. The trajectory interpolation loop in `build_traj_outputs` iterates per-trial and could also be partially vectorized.

ii.
```python
# Current per-trial loop:
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
    unit_trials.append(counts.astype(np.float32))
```

iii. No vectorization documentation was provided.

## 11-c. What processing does the code repeat multiple times?

i. The `go` cue times are re-read inside `build_traj_outputs` even though they are also available from `build_trial_labels`. The trial mask computation is done once, but the trial index lookup (`np.where(trial_mask)[0]`) is repeated for each unit during spike binning.

ii.
```python
# In build_trial_labels:
idx = np.where(trial_mask)[0]
# In build_traj_outputs:
go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)
```

iii. No documentation on repeated processing.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `clu.tm` and `clu.site` fields are loaded but never used. The `meta` field is loaded but only used incidentally. The paw velocity median subtraction step centers the velocity before computing magnitude, but this centering is unnecessary since the data is subsequently thresholded.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    if name in clu:
        clu_out[name] = ...
```

iii. Loading unused fields adds minor I/O overhead but does not significantly impact correctness.
