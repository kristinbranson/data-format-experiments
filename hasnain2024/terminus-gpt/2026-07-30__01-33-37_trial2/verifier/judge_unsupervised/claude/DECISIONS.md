# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans all subdirectories under `data/` for files matching `data_structure_*.mat` and `motionEnergy_*.mat`, pairing them by session stem (e.g., `EKH1_2021-08-07`). Only sessions with both a data_structure file and a motion_energy file are considered. `data_structure_*.mat` files are loaded with `h5py` (MATLAB v7.3/HDF5), and `motionEnergy_*.mat` files are loaded with `scipy.io.loadmat`. In practice, only sessions from `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` have motion energy files, so sessions from `DelayInhibition_BilatMC_Behavior` and `GoCueInhibition_BilatMC_Behavior` are excluded (those directories have no `motionEnergy_*.mat` files). Additionally, some `data_structure_*.mat` files (all JEB24 sessions, some JEB23 sessions) are unreadable (likely not HDF5 format) and are skipped.

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
```

```python
keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
```

iii. The agent documented in Step 2 of CONVERSION_NOTES.md that data directories contain four subdirectories. The filtering to sessions with both files was a pragmatic decision since motion energy is a required decoder output.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is inferred from the session filename stem by splitting on `_` and taking the first part (e.g., `EKH1` from `EKH1_2021-08-07`). A running list of unique subjects is maintained, with each session mapped to its subject index. The final dataset contains 12 subjects.

ii.
```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])
```

iii. The agent noted 18 unique subjects from filename parsing in Step 2, but after filtering (missing motion energy, unreadable files, missing neural data), only 12 subjects remain in the final dataset.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` + `motionEnergy_*.mat` pair constitutes one session. Sessions are processed individually in sorted order. Sessions are skipped if: (a) the data_structure file is unreadable, (b) no motion_energy file exists, (c) fewer than 2 valid trials remain, (d) fewer than 10 neural units pass FR filtering, or (e) trajectory data lacks paw features. The final dataset has 33 sessions.

ii.
```python
for stem in keys:
    sess = process_session(stem, sessions[stem], edges, centers)
    if sess is None:
        continue
    out_sessions.append(sess)
```

iii. The agent documented session exclusion criteria in Step 5 (Mapping Planning) and logged each skipped session with a reason in the conversion output.

## 1-d. How are the data split into trials?

i. Trial count per session comes from `bp.Ntrials`. Each trial's spike times, behavioral labels, trajectory data, and motion energy data are indexed by trial number. Only valid trials (passing the early/no filter) are included. Trial indices are 0-based in Python (converted from MATLAB 1-based).

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

iii. The agent identified from the methods text and reference code that early lick and ignore trials should be excluded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding those where `bp.early == 1` (early lick trials) or `bp.no == 1` (ignore trials). No other trial-level quality filtering is applied (e.g., the reference code's requirement for minimum correct DR/WC trials per direction is not enforced).

ii. Same as 1-d above.

iii. The agent documented in Step 3 that "early lick and ignore trials are omitted from analyses" based on the methods text. The reference code also balances trial counts for decoding, but the agent did not implement this (reasonably, since the decoder framework handles class imbalance).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu.trialtm` (per-unit spike times within each trial) and `obj.clu.trial` (trial index for each spike). These are arrays of spike times for each unit, with trial membership indicated by the corresponding trial index array.

ii.
```python
clu = obj['clu']
trialtm = clu['trialtm']
trialid = clu['trial']
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    st = np.array(st, dtype=float).reshape(-1)
    tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
```

iii. The agent identified `clu.trialtm` and `clu.trial` during Step 2 (Dataset Exploration) as per-unit spike times and trial indices.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into spike counts using `np.histogram` with a fixed time grid of 75 ms bins spanning [-2.5, 2.5] seconds. This produces 66 time bins per trial. **Critically, the spike times (`trialtm`) are NOT subtracted by the per-trial go cue time before binning.** The `trialtm` values are relative to trial start (bitStart), not relative to go cue. This means the neural data is effectively aligned to trial start, not to the go cue as intended.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers

def bin_spikes_for_session(obj, trial_mask, edges):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
        unit_trials.append(counts.astype(np.float32))
```

iii. The agent chose 75 ms bins based on `DLC_ContextDecoding.m` setting `rez.binSize = 75` and the [-2.5, 2.5] window from `getDefaultParams.m` showing `params.prep = [-2.5 -0.05]` and `params.move = [-2.5 1.5]`. However, the agent did not realize that `trialtm` is in trial-start coordinates and needed to be converted to go-cue coordinates by subtracting per-trial `goCue` time.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by firing rate: only units with FR > 1 Hz are kept. Sessions with fewer than 10 units after filtering are excluded. The `clu.quality` field is loaded but NOT used for filtering, despite the reference code having a `params.quality` parameter and the paper mentioning well-isolated single units.

ii.
```python
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
kept_units.append(ui)
```

```python
if len(neural) < 2 or len(kept_units) < 10:
    return None
```

iii. The agent documented in Step 3 that "all units with firing rates > 1 Hz were included in most analyses" and "sessions included only if they had at least 10 units," matching the paper. The default `params.quality = {'all'}` in the reference code means all quality levels are accepted, so omitting quality filtering is consistent with the default settings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **The neural data is NOT correctly aligned to the go cue.** The spike times in `trialtm` are relative to trial start (bitStart), and the agent bins them directly with edges [-2.5, 2.5] without subtracting the per-trial go cue time. For a typical trial where goCue occurs at ~2.5s after trial start, the binned window [-2.5, 2.5] in trialtm coordinates corresponds approximately to [-5.0, 0.0] in go-cue-relative time, capturing only pre-go-cue activity. For randomized delay trials where goCue varies (1.9 to 7.4s), the misalignment is inconsistent across trials.

ii.
```python
# No go cue subtraction - spikes binned directly in trialtm coordinates
counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The agent's CONVERSION_NOTES.md and trajectory show awareness that alignment should be to the go cue, but the implementation fails to subtract the per-trial goCue time from trialtm before binning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 75 ms, producing 66 bins across the [-2.5, 2.5] window. The native neural data is spike times (continuous), so the binning into 75 ms spike counts is the only temporal processing applied. The reference code's native `params.dt = 0.005` (5 ms) is not used; the DLC decoder bin size of 75 ms is applied directly.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers
```

iii. The agent found `rez.binSize = 75` ms in `DLC_ContextDecoding.m` and adopted it as the common bin size. This is documented in Step 1 and Step 4 of CONVERSION_NOTES.md.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is simply the bin center times from the fixed time grid, not derived from any raw data variable. It is the same for every trial: the center of each 75 ms bin from -2.5 to 2.5 seconds.

ii.
```python
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The agent designed this as a continuous time vector repeated for each trial, serving as a positional encoding for the decoder.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is applied. The input is simply the pre-computed bin centers array `[-2.4625, -2.3875, ..., 2.4625]`, broadcast identically for every trial.

ii. Same as 3-a.

iii. The agent noted this is "continuous time vector repeated for each trial" in Step 5.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same bin centers as the neural data's time grid, so they share the same number of time points (66). However, since the neural data is NOT actually aligned to the go cue (see 2-d), the input's claim of representing "time from go cue onset" is inconsistent with what the neural bins actually represent.

ii. Both neural binning and input use the same `centers` array from `build_time_grid()`.

iii. The agent assumed the binning grid directly represents go-cue-aligned time, but this is incorrect due to the missing go cue subtraction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.L` and `obj.bp.R`, which are per-trial binary arrays (0 or 1) indicating left and right lick trials.

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The agent mapped L/R to left=0/right=1 based on the decoder task specification.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Simple comparison: if R > L, direction is 1 (right); otherwise 0 (left). Since L and R are binary (0 or 1), this correctly identifies trial direction. The output is per-trial (constant across time bins), broadcast to all 66 time points.

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
# Later, broadcast to time:
np.full(n_bins, lick_dir[i], dtype=np.int64)
```

iii. The agent did not deeply discuss the semantics of L/R but the implementation is consistent with binary trial-type indicators.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, a per-trial array indicating whether autowater (water-cue) mode was active.

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. The agent derived this from `getBlockNum_AltContextTask.m`, which uses transitions in `bp.autowater` to identify context blocks.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Simple threshold: autowater > 0 maps to WC = 0, autowater == 0 maps to DR = 1. The reference code uses block-level transitions, but the agent uses a per-trial threshold. Since autowater is binary (0 or 1), both approaches yield the same result. The output is per-trial, broadcast to all time bins.

ii. Same as 5-a.

iii. The agent noted that the reference `DLC_ContextDecoding.m` labels AFC (i.e., DR) as Y=+1 and AW (i.e., WC) as Y=-1. The agent's WC=0, DR=1 mapping is consistent with the decoder task specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit` and `obj.bp.miss`, per-trial binary arrays.

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The agent mapped hit > miss to correct=1, otherwise incorrect=0, matching the decoder task specification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple comparison: hit > miss maps to correct=1, otherwise incorrect=0. Since early and ignore trials are already filtered, remaining trials should be either hit=1/miss=0 or hit=0/miss=1. The output is per-trial, broadcast to all time bins.

ii. Same as 6-a.

iii. The agent noted outcome encoding in Step 5 (Variable Mapping).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj[0].ts` (side camera view trajectory data) and `obj.traj[0].frameTimes` (frame timestamps), using a feature identified by name matching (`tongue`, `left_tongue`, or `right_tongue`) from `obj.traj[0].featNames`.

ii.
```python
side = traj_views[0]
side_names = [str(x) for x in side.get('featNames', [])]
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
tongue.append(interp_feature_velocity(ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
```

iii. The agent identified from `getDefaultParams.m` that camera view 0 is the side camera with tongue features and view 1 is the bottom camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The trajectory data for the selected tongue feature is extracted, frame times are converted to go-cue-relative time (subtracting 0.5s for camera sync offset and the per-trial go cue time), positions are interpolated to the bin centers, velocity is computed as the gradient of interpolated positions, NaN values are set to 0, and speed is computed as sqrt(vx^2 + vy^2).

**However, there is a critical bug in the dimension ordering.** The raw trajectory data has shape `(n_features, n_coords, n_frames)` (e.g., `(7, 3, 1792)`), but the code indexes it as `arr[:, :2, feat_idx]` which gives `(n_features, 2)` instead of `(n_frames, 2)`. This means the velocity is computed from 7 data points (one per feature) instead of 1792 frames, producing meaningless velocity traces.

ii.
```python
def interp_feature_velocity(ts, frame_times, align_time, centers, feat_idx, tongue=False):
    arr = np.array(ts, dtype=float)
    xy = arr[:, :2, feat_idx]  # BUG: should be arr[feat_idx, :2, :].T
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

iii. The agent based the velocity computation on `findPosition.m` and `findVelocity.m` from the reference code. The 0.5s camera sync offset comes from `WorkingWithDataObjs.m`. The NaN-to-0 handling for tongue matches `findVelocity.m` behavior.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is discretized using a per-session threshold at the median. Values strictly greater than the median are labeled 1 (high), otherwise 0 (low). The strict `>` (instead of `>=`) was chosen to avoid degenerate all-ones output when the median is 0 (common for tongue when no tongue is visible).

ii.
```python
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The agent noted in the trajectory that using `>=` caused degenerate outputs when the median was 0, and switched to strict `>`. The instructions specify `< 50th percentile` = 0 and `>= 50th percentile` = 1, so the agent's choice of `>` deviates from the specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity IS aligned to the go cue (subtracting per-trial goCue time from frame times), but the neural data is NOT aligned to the go cue. This creates a misalignment between the neural and tongue velocity time series. The tongue velocity uses the same bin centers as neural data, so both have 66 time points, but they represent different temporal windows relative to trial events.

ii. See 7-b for tongue alignment code and 2-d for neural alignment issue.

iii. The agent did not note or investigate this misalignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj[1].ts` (bottom camera view trajectory data) and `obj.traj[1].frameTimes`, using a feature identified by name matching (`top_paw`, `bottom_paw`, or `paw`) from `obj.traj[1].featNames`.

ii.
```python
bottom = traj_views[1]
bottom_names = [str(x) for x in bottom.get('featNames', [])]
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```

iii. The agent identified from `getDefaultParams.m` that view 1 (bottom camera) contains paw features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Similar to tongue velocity but with different NaN handling: NaN velocities are replaced with the median velocity, and the velocity is median-subtracted (baseline-corrected). **The same dimension-ordering bug as tongue velocity applies** -- the code extracts data from the wrong dimensions.

Additionally, the paw velocity code applies median subtraction (`xv = xv - np.nanmedian(xv)`, `yv = yv - np.nanmedian(yv)`), which is not clearly described in the reference code for velocity computation.

ii.
```python
# In interp_feature_velocity with tongue=False:
xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
xv = xv - np.nanmedian(xv)
yv = yv - np.nanmedian(yv)
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The agent distinguished tongue and paw processing based on the `tongue` flag, applying different NaN handling strategies.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session median threshold with strict `>`.

ii.
```python
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. Same rationale as tongue velocity (7-c).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same situation as tongue velocity: paw velocity IS go-cue-aligned but neural data is NOT, creating a misalignment. See 7-d.

ii. See 8-b and 2-d.

iii. Not discussed by the agent.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motionEnergy_*.mat` files, specifically the `me.data` field, which contains per-trial variable-length continuous traces.

ii.
```python
def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']
```

iii. The agent identified the motion energy file structure in Step 2.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy traces (variable length, e.g., 1792 samples) are resampled to 66 bins using linear interpolation. **No temporal alignment to go cue is performed** -- the trace is simply stretched/compressed from its native length to 66 bins using normalized coordinates [0, 1]. This means the same behavioral event (e.g., lick onset) may end up at different bin positions for trials with different durations.

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

iii. The agent noted motion energy traces are variable-length and need rebinning. The normalization approach was chosen for simplicity but does not align to any behavioral event.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session median threshold with strict `>`, same as tongue and paw velocity.

ii.
```python
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. Same rationale as 7-c.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is NOT aligned to the go cue or any specific behavioral event. It is simply resampled from its native length to 66 bins. The neural data is also not go-cue-aligned (see 2-d), but uses the trial-start reference frame. Since motion energy traces span the entire trial (from trial start), the resampled traces may partially overlap with the neural data's effective time window, but the alignment is approximate at best and inconsistent across trials of different lengths.

ii. See 9-b.

iii. The agent did not discuss the temporal alignment of motion energy relative to neural data or go cue.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data scenarios are handled:
- Sessions with unreadable HDF5 files are skipped with logging.
- Sessions missing motion energy files are skipped.
- Sessions with insufficient neural units (<10) or trials (<2) are skipped.
- Sessions without paw trajectory features are skipped.
- Tongue trajectory NaN values are set to 0.
- Paw trajectory NaN values are replaced with the median.
- Empty motion energy traces are replaced with zero vectors.
- Missing frame times are generated synthetically (at 400 Hz) or linearly interpolated.

ii.
```python
# Missing frame times
if ft.size == 0:
    ft = np.arange(1, xy.shape[0] + 1, dtype=float) / 400.0
else:
    ft = np.linspace(ft.min(), ft.max(), xy.shape[0])

# Tongue NaN handling
if tongue:
    xv[np.isnan(xv)] = 0
    yv[np.isnan(yv)] = 0

# Empty motion energy
if arr.size == 0:
    return np.zeros(n_bins, dtype=np.float32)
```

iii. The agent documented session skipping reasons in the conversion output and noted bug fixes for trajectory loading in CONVERSION_NOTES.md Step 10.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is `bin_spikes_for_session`, which loops over every unit and every trial to histogram spike times. Loading the HDF5 data structure files is also time-consuming due to extensive reference dereferencing. The agent did not include timing instrumentation despite instructions to do so.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The agent noted in Step 6 placeholders for "Code inefficiencies identified" and "Code speedups added" but left them as `[Note]` without filling in details.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over each unit and each trial separately, computing `np.histogram` one trial at a time. This could be vectorized by pre-sorting spikes by trial and using vectorized bincount or histogram operations across all trials simultaneously. The motion energy rebinning also uses a per-trial loop that could be batched.

ii. See 11-a code snippet for the nested loop.

iii. The agent did not document or attempt vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The `go` (goCue) array is loaded and reshaped in both `build_trial_labels` (implicitly through idx) and `build_traj_outputs`. The valid trial mask is computed once and reused, which is efficient. The `normalize_motion_energy_data` function has multiple nested fallback paths that check the same data in different ways.

ii.
```python
# goCue loaded in build_traj_outputs
go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)
# Also bp arrays loaded in build_trial_labels
L = np.array(bp['L'], dtype=float).reshape(-1)
```

iii. Not discussed by the agent.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `normalize_motion_energy_data` function has extensive fallback logic for different data formats (dict wrapping, nested keys, dtype names) that is largely unnecessary since all motion energy files follow the same format. The `load_data_structure` function loads metadata fields (`meta`) that are never used in the conversion. The `read_dataset_maybe_refs` function is defined but may not be called. The `safe_decode_ref_string` function handles deeply nested references that may not exist in the actual data.

ii.
```python
def normalize_motion_energy_data(me):
    # ~30 lines of fallback logic for different data formats
    ...
```

iii. The agent built these as defensive code for handling unknown data formats. The overhead is minimal per session.
