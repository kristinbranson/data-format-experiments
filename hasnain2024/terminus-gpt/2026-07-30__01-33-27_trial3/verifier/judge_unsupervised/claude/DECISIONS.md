# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing `data/Ephys_Behavior/data_structure_*.mat` files and pairing them with `motionEnergy_*.mat` sidecars by matching subject/date in filenames. Each `.mat` file is loaded via `h5py` (since all are MATLAB v7.3 / HDF5). The top-level object is `obj` containing subgroups `bp`, `clu`, `traj`, `meta`, etc. Only the `Ephys_Behavior` data family is used.

ii.
```python
def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
    motion_map = {}
    for mf in sorted(base.glob('motionEnergy_*.mat')):
        m = re.match(r'^motionEnergy_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', mf.name)
        if m:
            motion_map[(m.group(1), m.group(2))] = mf
    sessions = []
    for df in data_files:
        m = re.match(r'^data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', df.name)
        if not m:
            continue
        subj, day = m.group(1), m.group(2)
        sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
    return sessions

def load_mat_obj(path: Path):
    if is_hdf5_mat(path):
        return h5py.File(path, 'r')
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return x['obj'].flat[0]
```

iii. The AI chose `Ephys_Behavior` because this is the data family containing the two-context (DR vs WC) task data with `autowater` labels, matching the paper's context decoding analyses. It handled the HDF5 format after discovering that all files were MATLAB v7.3.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are extracted from the filenames (e.g., `JEB13` from `data_structure_JEB13_2022-09-13.mat`). Unique subject IDs are collected and sorted. Each session is assigned a `subject_idx` pointing into the sorted `subjects` list.

ii.
```python
subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
# ...
data['subject_idx'].append(subj_to_idx[sess.subject])
```

iii. Subject IDs are parsed directly from filenames, which is consistent with how the data files are organized.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file corresponds to one session. Sessions are filtered to only include "context-capable" sessions — those containing both autowater=0 (DR) and autowater=1 (WC) trials. This yields 22 of 25 sessions; 1 more (JEB6) is skipped due to an incompatible `clu` structure, leaving 21 sessions.

ii.
```python
def select_context_sessions(sessions: List[SessionRecord]) -> List[SessionRecord]:
    keep = []
    for s in sessions:
        try:
            obj = load_mat_obj(s.data_file)
            bp = get_bp_container(obj)
            autowater = get_trial_bool(bp, 'autowater')
            if autowater is None:
                continue
            vals = set(np.unique(autowater.astype(int)).tolist())
            if vals == {0, 1}:
                keep.append(s)
        except Exception as e:
            print(f'[warn] skipping {s.data_file.name}: {e}', flush=True)
    return keep
```

iii. The AI identified that two-context sessions are those with both DR and WC trial types, encoded via the `autowater` field. However, the paper reports 12 sessions from 6 mice for the two-context task — the AI's 21 sessions from 9+ mice is a significant overcount that was never resolved.

## 1-d. How are the data split into trials?

i. Trials are identified by the length of the `goCue` event array within each session's `bp.ev`. Each trial index corresponds to one entry in the event arrays.

ii.
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
```

iii. This follows the standard structure of the data files where each element of the event arrays corresponds to one trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial filters: (1) exclude early-lick trials (`early`), (2) require hit OR miss (exclude ignore trials), (3) exclude stimulation trials (`stim.enable`), (4) require exactly one of left/right lick direction, (5) require valid `autowater` value (0 or 1). Both hit and miss trials are kept.

ii.
```python
valid = np.ones(n_trials, dtype=bool)
if early is not None:
    valid &= ~early
if hit is not None and miss is not None:
    valid &= (hit | miss)
if stim_enable is not None:
    valid &= ~stim_enable
if right is not None and left is not None:
    valid &= (right ^ left)
if autowater is not None:
    valid &= np.isin(autowater.astype(int), [0, 1])
```

iii. The AI noted that the reference code (`getDefaultParams.m`) uses hit-only trials, but deliberately kept miss trials to make the `outcome` output non-degenerate. The paper's behavioral inclusion criterion (>=40 correct DR trials/direction, >=20 correct WC trials/direction) was investigated but not enforced because it reduced sessions too aggressively.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `clu` (cluster) group in the HDF5 session files. Specifically, `clu.trial` (which trial each spike belongs to) and `clu.trialtm` (the trial-aligned time of each spike) are used.

ii.
```python
clu_root = obj['obj']['clu']
clu = obj[clu_root[0,0]]
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    tr_ref = clu['trial'][ci,0]
    tm_ref = clu['trialtm'][ci,0]
    tr_arr = np.asarray(obj[tr_ref][()]).reshape(-1).astype(int)
    tm_arr = np.asarray(obj[tm_ref][()]).reshape(-1).astype(float)
    clu_trial.append(tr_arr)
    clu_trialtm.append(tm_arr)
```

iii. The AI identified that `trialtm` contains pre-aligned spike times (relative to go cue), making separate alignment unnecessary.

## 2-b. How is the `neural` data processed?

i. For each trial, spike times from `clu.trialtm` are histogrammed into 75 ms bins spanning -1.5 s to +1.5 s relative to go cue. This produces spike counts (not firing rates) in a (n_neurons, n_timepoints) matrix per trial.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
# per trial:
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The AI chose 75 ms bins based on the decoder scripts' `rez.binSize = 75`. The time window [-1.5, 1.5] is narrower than the reference code's [-2.5, 2.5] from `getDefaultParams.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No firing rate filter is applied. All clusters from the `clu` group are included regardless of firing rate.

ii. There is no code implementing `removeLowFRClusters` or any equivalent threshold. The neural extraction loop includes all clusters:
```python
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    # all clusters included
```

iii. The AI identified that the reference code uses `removeLowFRClusters.m` (with a 0.5 Hz threshold from `getDefaultParams.m`) and that the paper states units >1 Hz were included. However, the agent never implemented this filter, resulting in 5,512 neurons vs the paper's 522 for the two-context task.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The `clu.trialtm` field already contains spike times aligned to the go cue. The code selects spikes belonging to each trial using `clu.trial == trial_number` and bins those pre-aligned times.

ii.
```python
tr1 = tr + 1  # 1-indexed trial number
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The AI assumed `trialtm` is already aligned to the go cue event, which is consistent with how the reference code's `alignSpikes.m` produces trial-relative spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 75 ms (0.075 s). The raw spike times are directly binned at this resolution — no intermediate finer binning followed by rebinning is applied.

ii.
```python
BIN_SIZE_S = 0.075
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The 75 ms bin size matches the reference decoder scripts' `rez.binSize = 75` ms. The reference code first bins at `params.dt` (5 ms) then rebins, but the AI skips the intermediate step and bins directly at 75 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time bin centers themselves, not derived from any raw data variable. It is the array of time bin centers spanning -1.5 to +1.5 s.

ii.
```python
inp = time_bins[None, :].astype(np.float32)
```

iii. This is a synthetic variable representing the time axis relative to go cue onset, as specified in the decoder task instructions.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time bins are generated using `np.arange` from -1.5 to +1.5 with step 0.075 s. Each trial gets the same time array as its input.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
inp = time_bins[None, :].astype(np.float32)
```

iii. This is straightforward — the same time vector is used for all trials since the alignment event (go cue) defines t=0.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_bins` array defines both the neural binning edges and the input values, ensuring perfect alignment.

ii.
```python
# Neural binning uses:
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]])
# Input is:
inp = time_bins[None, :]
```

iii. Since time bins are the centers of the neural histogram bins, the input time values correspond exactly to the neural data columns.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the `R` (right) and `L` (left) boolean fields in `bp`.

ii.
```python
right = get_trial_bool(bp, 'R')
left = get_trial_bool(bp, 'L')
lick_dir = 1 if bool(right[tr]) else 0
```

iii. The AI uses the trial-level R/L labels directly, consistent with the reference code's condition definitions.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It is a per-trial binary value: right=1, left=0. The value is broadcast across all time bins.

ii.
```python
lick_dir = 1 if bool(right[tr]) else 0
np.full(time_bins.shape, lick_dir, dtype=np.int64)
```

iii. This matches the instructions (left=0, right=1, per-trial).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `autowater` field in `bp`.

ii.
```python
autowater = get_trial_bool(bp, 'autowater')
context = 0 if bool(autowater[tr]) else 1
```

iii. `autowater=True` maps to WC=0, `autowater=False` maps to DR=1, matching the decoder task specification.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It is a per-trial binary value: WC (autowater)=0, DR (non-autowater)=1, broadcast across time bins.

ii.
```python
context = 0 if bool(autowater[tr]) else 1
np.full(time_bins.shape, context, dtype=np.int64)
```

iii. Consistent with the instructions (WC=0, DR=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` boolean field in `bp`.

ii.
```python
hit = get_trial_bool(bp, 'hit')
outcome = 1 if bool(hit[tr]) else 0
```

iii. Trials where `hit=True` are correct (1), where `hit=False` are incorrect (0). Since early and ignore trials are already excluded, remaining non-hit trials are misses.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. It is a per-trial binary value: correct=1, incorrect=0, broadcast across time bins.

ii.
```python
outcome = 1 if bool(hit[tr]) else 0
np.full(time_bins.shape, outcome, dtype=np.int64)
```

iii. Matches the instructions (incorrect=0, correct=1).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the `traj` group in the HDF5 session files. The code searches for DLC-tracked feature names including `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`, `tongue`, `left_tongue`, `right_tongue`.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr,
    ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue',
     'tongue', 'left_tongue', 'right_tongue'], time_bins)
```

iii. The AI identified these as DLC-tracked tongue markers from the trajectory tracking data.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the x/y coordinates of the first matching tongue feature are extracted from `traj`. Velocity (speed) is computed as `sqrt(dx^2 + dy^2) / dt` where `dx`, `dy` are differences in position and `dt` is the frame time difference. The speed is then interpolated to the time bin centers using `np.interp`, with NaN-filling via nearest-value interpolation for gaps.

ii.
```python
xy = ts[feat_idx, :2, :]
dx = np.diff(xy[0], prepend=xy[0,0])
dy = np.diff(xy[1], prepend=xy[1,0])
dt = np.diff(ft, prepend=ft[0])
dt[dt <= 0] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0
speed = np.sqrt(dx*dx + dy*dy) / dt
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The AI computes raw pixel-space velocity from DLC trajectories. The nearest-fill interpolation mimics the reference code's `fillmissing` approach for motion energy.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All tongue velocity values across all trials in a session are pooled. The 50th percentile (median) is computed. Values >= threshold become 1 (high), values < threshold become 0 (low).

ii.
```python
tongue_all = np.concatenate([...])
tongue_thr = np.nanpercentile(tongue_all, 50)
# per trial:
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
```

iii. Matches the instructions (per-session 50th percentile threshold).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue trajectory frame times are aligned to the go cue by subtracting the go cue time (`rel_t = ft - go`), then speed values are interpolated to the same `time_bins` used for neural data.

ii.
```python
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The alignment uses the same go cue event and same time bins as the neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the `traj` group, searching for features named `paw`, `left_paw`, `right_paw`, `top_paw`, `bottom_paw`.

ii.
```python
paw_vals = extract_hdf5_traj_velocity(obj, tr,
    ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins)
```

iii. These are DLC-tracked paw markers from the trajectory tracking data.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue velocity: x/y coordinate differences -> speed -> interpolation to time bins -> nearest-fill for NaN gaps. Uses the same `extract_hdf5_traj_velocity` function.

ii. Same function as tongue (see 7-b), with different feature name candidates.

iii. The processing is identical to tongue velocity processing, just using paw feature names.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same approach as tongue: pool all paw velocity values across the session, compute 50th percentile, threshold into low (0) and high (1).

ii.
```python
paw_all = np.concatenate([...])
paw_thr = np.nanpercentile(paw_all, 50)
tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
```

iii. Matches the instructions (per-session 50th percentile threshold).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: frame times aligned to go cue, interpolated to same time bins.

ii. Same as 7-d.

iii. Uses identical alignment approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_*.mat` files, specifically the `me` variable containing per-trial motion energy time series.

ii.
```python
def load_motion_energy(path: Optional[Path], n_trials: int) -> Optional[List[np.ndarray]]:
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    me = x['me'].flat[0]
    data = np.asarray(unwrap_me_data(me)).reshape(-1)
```

iii. These sidecar files contain pre-computed motion energy from video recordings.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the raw motion energy array is interpolated to the time bins using `np.interp` with linearly spaced source times from T_START to T_END. NaN values are filled with the session median before thresholding.

ii.
```python
if raw.size > 1:
    x_old = np.linspace(T_START, T_END, raw.size)
    me_binned = np.interp(time_bins, x_old, raw)
# thresholding:
all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all))) for x in me_binned_all])
thr = np.nanpercentile(all_me, 50)
```

iii. The AI interpolates assuming the motion energy spans the full trial window. The reference code uses actual frame times and a specific offset (`frameTimes - 0.5`), which the AI does not replicate.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. All motion energy values across the session are pooled (with NaN replaced by session median), the 50th percentile is computed, and values are thresholded into low (0) and high (1).

ii.
```python
thr = np.nanpercentile(all_me, 50)
tmp[finite] = (meb[finite] >= thr).astype(np.int64)
```

iii. Matches the instructions (per-session 50th percentile).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The raw motion energy array is assumed to span [T_START, T_END] with uniform spacing, and is interpolated to the same `time_bins` used for neural data.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. This assumes the motion energy data spans the same time window as the neural data. The reference code uses actual frame times with a camera timing offset, which may produce different alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used: (1) Sessions with incompatible `clu` structures are skipped entirely (JEB6). (2) Sessions with <2 valid trials are skipped. (3) Missing motion energy files result in NaN values that are filled with session median during thresholding. (4) Missing tongue/paw trajectory data results in NaN values filled via nearest-value interpolation. (5) Zero or negative frame time differences are replaced with the median positive difference. (6) Nested MATLAB structs in motion energy data are recursively unwrapped.

ii.
```python
# Skip incompatible sessions:
if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
    raise ValueError('unsupported clu structure')
# Skip sessions with too few trials:
if len(neural) < 2:
    continue
# NaN filling for trajectories:
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The AI documented these handling strategies in CONVERSION_NOTES.md. The approach is generally reasonable but some edge cases (like the JEB6 skip) represent data loss.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is `build_session`, which takes 5-19 seconds per session. Within that, the tongue and paw velocity extraction is the dominant cost because `extract_hdf5_traj_velocity` is called 3 times per trial (once during initial output construction, and twice more during the thresholding loop).

ii. From conversion_full_out.txt, build times range from 5.64s to 19.14s per session, totaling ~200s for 21 sessions.

iii. The AI noted timing information but did not optimize the redundant trajectory extraction calls.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cluster spike loading loop (iterating over all clusters to load trial/trialtm arrays) could potentially be vectorized. The per-trial trajectory extraction loop could also be improved.

ii.
```python
for ci in range(n_clu):
    tr_ref = clu['trial'][ci,0]
    tm_ref = clu['trialtm'][ci,0]
    # ...
```

iii. The HDF5 object reference structure makes vectorization difficult, but batched loading could help.

## 11-c. What processing does the code repeat multiple times?

i. Tongue and paw velocity extraction is done **three times** per trial: once during the initial `output_trials` construction (line 288-289), once when pooling for threshold computation (line 321-322), and once when applying thresholds (line 334-335). This triples the most expensive operation.

ii.
```python
# First extraction (line 288):
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
# Second extraction for threshold (line 321):
tongue_all = np.concatenate([...extract_hdf5_traj_velocity(obj, tr, [...], time_bins)... for tr in trial_idx])
# Third extraction for applying threshold (line 334):
tong = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
```

iii. This is a clear inefficiency — the trajectory values should be computed once and cached.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The initial tongue and paw velocity extraction during the per-trial output construction (lines 288-289) is never used — the values are immediately overwritten by the thresholding pass (lines 334-345). The initial output rows for tongue/paw/motion_energy are set to zeros and then replaced.

ii.
```python
# These extractions are unused:
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins) if isinstance(obj, h5py.File) else np.full(...)
paw_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins) if isinstance(obj, h5py.File) else np.full(...)
# Initial zeros are overwritten:
np.zeros(time_bins.shape, dtype=np.int64),  # tongue - overwritten at line 340
np.zeros(time_bins.shape, dtype=np.int64),  # paw - overwritten at line 345
```

iii. This is dead code — `tongue_vals` and `paw_vals` from lines 288-289 are computed but never stored or used. The initial zero rows in the output matrix are subsequently replaced by the threshold-based values.
