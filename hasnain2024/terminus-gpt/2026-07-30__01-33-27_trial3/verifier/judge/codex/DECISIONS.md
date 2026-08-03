# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing `data/Ephys_Behavior/data_structure_*.mat`, pairs each with a same-stem `motionEnergy_*.mat` file if present, and never reads the hard-coded session/probe lists from the paper code. It ignores `RandomizedDelay_Ephys_Behavior` entirely.

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
```

iii. In `CONVERSION_NOTES.md`, the AI says the target is the "two-context ephys sessions" inside `Ephys_Behavior`, and in the trajectory it repeatedly justifies session discovery by the presence of both autowater states rather than by the authors' explicit loader lists.

## 1-b. How are the data split into subjects?

i. Subject IDs come from the filename regex capture before the date. The AI stores that string in `SessionRecord.subject`, then builds `subjects` as the sorted unique set of those selected session-level subject strings and `subject_idx` from that mapping.

ii.
```python
subj, day = m.group(1), m.group(2)
sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
```

```python
subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_idx[sess.subject])
```

iii. The notes say subject metadata would come from "filename / `obj.meta`", but the implemented path is the filename-derived subject ID.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_<subject>_<date>.mat` file from `Ephys_Behavior`. The AI then keeps only sessions whose `autowater` field contains both 0 and 1, and later silently drops sessions whose neural structure cannot be parsed or that end with fewer than two valid trials.

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

```python
try:
    with timed(f'build {sess.data_file.name}'):
        neural, inp, out, bri = build_session(obj, sess)
except Exception as e:
    print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True)
    continue
if len(neural) < 2:
    print(f'[warn] skipping {sess.data_file.name}: <2 valid trials', flush=True)
    continue
```

iii. The notes explicitly frame the intended subset as "context-capable sessions with both autowater states," and the trajectory shows the AI using that as its operational definition of the paper's two-context cohort.

## 1-d. How are the data split into trials?

i. The AI sets the session trial count from the length of `bp.ev.goCue`. It builds a boolean `valid` mask from per-trial behavioral fields, then uses `trial_idx = np.where(valid)[0]` as the retained trial list. Neural spikes are assigned to a trial by matching `clu.trial == tr + 1`.

ii.
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
...
valid = np.ones(n_trials, dtype=bool)
...
trial_idx = np.where(valid)[0]
```

```python
tr1 = tr + 1
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The trajectory shows the AI relying on the per-trial `bp` arrays and the `clu.trial` field rather than reconstructing trial boundaries from continuous time.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept only if it is not early, is either a hit or a miss, is not photostimulated if `stim.enable` exists, has exactly one of `R` and `L` true when both exist, and has an `autowater` value in `{0,1}`. Ignore trials are removed by the `hit | miss` requirement. There is no recording-length cut like the reference's last-spiking-trial trim.

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

iii. The notes and trajectory show the AI deliberately keeping miss trials so `outcome` would remain non-degenerate, while still excluding early and stim trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices come from the first referenced `obj['obj']['clu']` group only, specifically its `trial` and `trialtm` arrays for each cluster. The code reads `goCue`, but the neural binning itself only consumes `trial` and `trialtm`.

ii.
```python
clu_root = obj['obj']['clu']
clu = obj[clu_root[0,0]]
...
tr_ref = clu['trial'][ci,0]
tm_ref = clu['trialtm'][ci,0]
tr_arr = np.asarray(obj[tr_ref][()]).reshape(-1).astype(int)
tm_arr = np.asarray(obj[tm_ref][()]).reshape(-1).astype(float)
clu_trial.append(tr_arr)
clu_trialtm.append(tm_arr)
```

iii. In the trajectory, the AI notes that `clu` exposes raw spike times as `tm`, `trial`, and `trialtm`, and concludes that this is enough to "directly bin spikes per neuron per trial."

## 2-b. How is the `neural` data processed?

i. For each retained trial and each loaded cluster, the AI histograms the raw `trialtm` spike times into 75 ms bins from `-1.5` s to `1.5` s and stores the resulting counts as `float32`. It does not subtract go cue, convert counts to Hz, smooth, normalize, or concatenate multiple probes explicitly.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
mat = np.zeros((len(clu_trial), time_bins.size), dtype=np.float32)
...
mat[ci], _ = np.histogram(spikes, bins=binedges)
neural_trials.append(mat)
```

iii. The AI's notes repeatedly justify the 75 ms choice from the MATLAB decoder scripts (`rez.binSize = 75` ms), and the implementation follows that interpretation directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no per-unit neural QC. The code keeps every cluster found in the first parsed `clu` group for supported sessions. It does not use cluster quality labels or a firing-rate threshold. The only neural-related exclusion is skipping whole sessions when `clu` is in an unsupported layout.

ii.
```python
if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
    raise ValueError('unsupported clu structure')
...
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    ...
    clu_trial.append(tr_arr)
    clu_trialtm.append(tm_arr)
```

iii. `CONVERSION_NOTES.md` says low-FR filtering was still unresolved, and the trajectory explicitly acknowledges that the current script is extracting raw clusters "without FR filtering."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are not explicitly aligned to go cue in the final code. The AI bins `trialtm` directly, which means the alignment only works if `trialtm` were already go-cue-relative; the code itself never subtracts `goCue`.

ii.
```python
spikes = tm_arr[tr_arr == tr1]
if spikes.size:
    mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The trajectory shows the AI knew `goCue` was the requested alignment event, but that correction never made it into the neural binning path.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 75 ms bins over a `[-1.5, 1.5]` s window. Raw spike times, motion energy traces, and trajectory-derived velocities are all rebinned or interpolated onto that 75 ms grid.

ii.
```python
BIN_SIZE_S = 0.075
ALIGN_EVENT = 'goCue'
T_START = -1.5
T_END = 1.5
...
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The notes cite the paper code's decoder bin size (`75` ms) as the reason for using this coarser grid.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The actual input array is not read from a raw variable. It is synthesized from the fixed constants `T_START`, `T_END`, and `BIN_SIZE_S` as a generic time axis intended to represent time from go cue.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. In the notes, the AI frames this as satisfying the decoder requirement for a continuous time-from-go-cue input rather than as something loaded from the raw data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI builds a fixed `np.arange` time vector once per session and copies it into every retained trial as a `(1, T)` array.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
...
inp = time_bins[None, :].astype(np.float32)
input_trials.append(inp)
```

iii. The notes say the decoder task requires the time axis, and the implementation uses the simplest possible repeated session-wide grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_bins` vector is used both as the decoder input and as the center grid for the neural histogram edges, so the input is aligned to the AI's neural data by construction.

ii.
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
inp = time_bins[None, :].astype(np.float32)
```

iii. The trajectory repeatedly describes the common 75 ms grid as the intended shared alignment frame for all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The label is derived from the trial-type field `R`; `L` is only used as part of the trial-validity filter. The code does not derive lick direction from the combination of side and outcome or from lick timestamps.

ii.
```python
right = get_trial_bool(bp, 'R')
left = get_trial_bool(bp, 'L')
...
lick_dir = 1 if bool(right[tr]) else 0
```

iii. In the notes, the AI explicitly planned to use the trial direction fields `R`/`L` rather than post-hoc lick timing.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI maps `R=True` to `1` and everything else to `0`, then broadcasts that scalar over all time bins in the trial. It does not use `hit`/`miss` to infer the actual licked side, and it does not keep a no-lick category.

ii.
```python
output_trials.append(np.vstack([
    np.full(time_bins.shape, lick_dir, dtype=np.int64),
    ...
]))
```

iii. The justification in the notes is that the decoder target should come from the task's side labels, but the final code does not incorporate miss-vs-hit information.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived directly from the per-trial `autowater` field in `bp`.

ii.
```python
autowater = get_trial_bool(bp, 'autowater')
```

iii. The notes identify `autowater` as the raw context flag and describe the two-context task as DR/non-autowater versus WC/autowater.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater=True` as WC (`0`) and `autowater=False` as DR (`1`), then repeats that label across time bins.

ii.
```python
context = 0 if bool(autowater[tr]) else 1
...
np.full(time_bins.shape, context, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says the value convention should match the prompt exactly: `WC=0`, `DR=1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The final label is derived from `hit`, with `miss` only used upstream in the trial-validity mask. Because only hit or miss trials survive, `hit=False` on a retained trial implicitly means "incorrect."

ii.
```python
hit = get_trial_bool(bp, 'hit')
miss = get_trial_bool(bp, 'miss')
...
if hit is not None and miss is not None:
    valid &= (hit | miss)
...
outcome = 1 if bool(hit[tr]) else 0
```

iii. The trajectory shows the AI explicitly keeping miss trials so the required correct/incorrect output would vary instead of collapsing to all-correct.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI makes `outcome` binary: `1` for hit and `0` otherwise on the retained hit-or-miss trials, then broadcasts that value across all bins. It does not keep ignore trials or an ignore class.

ii.
```python
outcome = 1 if bool(hit[tr]) else 0
...
np.full(time_bins.shape, outcome, dtype=np.int64)
```

iii. The trajectory documents this as a deliberate tradeoff: keeping miss trials was necessary so the decoder-task `outcome` output would not be degenerate.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity comes from `obj['obj']['traj']`, using the per-trial `featNames`, `ts`, and `frameTimes` arrays and picking the first feature name that matches a tongue-related candidate list. The code also reads `goCue` to place the frames on a nominal go-cue-relative axis.

ii.
```python
names_arr = obj[traj['featNames'][trial_index0,0]][()]
...
ts = np.asarray(obj[traj['ts'][trial_index0,0]][()]).astype(float)
ft = np.asarray(obj[traj['frameTimes'][trial_index0,0]][()]).reshape(-1).astype(float)
...
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
```

```python
tongue_vals = extract_hdf5_traj_velocity(
    obj, tr,
    ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue',
     'tongue', 'left_tongue', 'right_tongue'],
    time_bins
)
```

iii. The notes and trajectory show the AI inspecting HDF5 `traj` layouts and feature names and then implementing a broad candidate-name search after several debugging rounds.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For the selected tongue feature, the AI computes frame-to-frame speed from raw `x`/`y` differences divided by frame-time differences, interpolates that speed onto the 75 ms decoder grid, and fills interior missing bins by interpolating across the finite samples. It does not use likelihood thresholds, per-run smoothing, video-offset correction, or two-view normalization/averaging.

ii.
```python
xy = ts[feat_idx, :2, :]
dx = np.diff(xy[0], prepend=xy[0,0])
dy = np.diff(xy[1], prepend=xy[1,0])
dt = np.diff(ft, prepend=ft[0])
dt[dt <= 0] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0
speed = np.sqrt(dx*dx + dy*dy) / dt
...
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The notes say the AI switched to nearest/interpolative filling after seeing degenerate sample outputs, and the trajectory describes that fix as the step that made tongue labels look balanced.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI concatenates all retained-trial tongue velocity values for a session, takes the 50th percentile, and assigns each finite bin to `0` or `1` according to whether it falls below or above that threshold. Bins that stay non-finite remain at `0` because the output buffer is zero-initialized.

ii.
```python
tongue_all = np.concatenate([
    np.nan_to_num(extract_hdf5_traj_velocity(...), nan=np.nan)
    for tr in trial_idx
]) if len(trial_idx) else np.array([])
tongue_thr = np.nanpercentile(tongue_all, 50) if tongue_all.size and np.isfinite(tongue_all).any() else np.nan
...
tmp = np.zeros(tong.shape, dtype=np.int64)
finite = np.isfinite(tong)
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
output_trials[i][3] = tmp
```

iii. The notes explicitly say the task required per-session median thresholding for the continuous movement outputs.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI subtracts the trial's `goCue` directly from `frameTimes` to make a relative time vector and then interpolates onto the same 75 ms `time_bins` grid used for neural data. It does not estimate or subtract any video/behavior clock offset.

ii.
```python
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The trajectory consistently treats go cue as the universal alignment event, but the notes never document a separate video-offset correction step.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the same `traj` structure as tongue velocity, again using `featNames`, `ts`, and `frameTimes`, but searching a paw-related candidate-name list instead of tongue names.

ii.
```python
paw_vals = extract_hdf5_traj_velocity(
    obj, tr,
    ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'],
    time_bins
)
```

iii. The notes show the AI had trouble identifying a stable paw feature early on and then broadened the candidate-name list to pick up any paw-like marker present in the session.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing is the same as tongue velocity in this code path: raw frame-to-frame `x`/`y` speed, interpolation to the 75 ms grid, and missing-value fill across finite bins. There is no likelihood gating, smoothing, or single-view selection rule from the reference implementation.

ii.
```python
speed = np.sqrt(dx*dx + dy*dy) / dt
...
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The notes describe these kinematic extractions as engineering fixes added to replace placeholder all-zero outputs rather than as a direct reproduction of the MATLAB paw pipeline.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI computes a per-session 50th percentile threshold from all concatenated paw velocities and assigns finite bins to binary low/high classes. Non-finite bins remain `0`.

ii.
```python
paw_all = np.concatenate([
    np.nan_to_num(extract_hdf5_traj_velocity(...), nan=np.nan)
    for tr in trial_idx
]) if len(trial_idx) else np.array([])
paw_thr = np.nanpercentile(paw_all, 50) if paw_all.size and np.isfinite(paw_all).any() else np.nan
...
tmp = np.zeros(paw.shape, dtype=np.int64)
finite = np.isfinite(paw)
tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
output_trials[i][4] = tmp
```

iii. This follows the same notes-based rationale as tongue velocity: the decoder task asked for a per-session median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same alignment strategy as tongue velocity in the AI code: subtract `goCue` from `frameTimes` and interpolate onto the shared 75 ms `time_bins` grid, with no explicit video-clock offset.

ii.
```python
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The AI's notes treat the common go-cue-relative bin grid as sufficient for all outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is read from the sidecar `motionEnergy_<subject>_<date>.mat` file, specifically from the `me` object after repeated unwrapping of nested `.data` wrappers.

ii.
```python
def load_motion_energy(path: Optional[Path], n_trials: int) -> Optional[List[np.ndarray]]:
    if path is None or not path.exists():
        return None
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    me = x['me'].flat[0]
    data = np.asarray(unwrap_me_data(me)).reshape(-1)
```

iii. The notes mention that motion-energy files came in multiple nested MATLAB layouts, and the AI added the wrapper-unwrapping logic to tolerate that variability.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each retained trial, the AI assumes the raw motion-energy vector spans the full `[-1.5, 1.5]` s decoder window, builds a synthetic `x_old = np.linspace(T_START, T_END, raw.size)`, and interpolates the trace onto the 75 ms grid. It does not use actual camera frame times, video offset, or go-cue-relative frame alignment.

ii.
```python
if me_trials is not None and tr < len(me_trials):
    raw = me_trials[tr]
    if raw.size > 1:
        x_old = np.linspace(T_START, T_END, raw.size)
        me_binned = np.interp(time_bins, x_old, raw)
    else:
        me_binned = np.full(time_bins.shape, np.nan)
```

iii. The trajectory shows this loader being patched mainly for robustness against nested MATLAB structs, not for exact temporal fidelity to the reference motion-energy processing.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates all binned motion-energy traces in a session, replaces NaNs with the session-wide median of the concatenated traces for the purpose of threshold calculation, takes the 50th percentile, and assigns each finite bin to `0` or `1`. Non-finite bins stay at `0`.

ii.
```python
all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all))) for x in me_binned_all])
thr = np.nanpercentile(all_me, 50)
for i, meb in enumerate(me_binned_all):
    tmp = np.zeros(meb.shape, dtype=np.int64)
    finite = np.isfinite(meb)
    tmp[finite] = (meb[finite] >= thr).astype(np.int64)
    output_trials[i][5] = tmp
```

iii. The notes say the per-session 50th-percentile threshold came from the decoder-task spec.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is implicit rather than event-based: the AI stretches each raw motion-energy trial to the fixed `[-1.5, 1.5]` s window and interpolates it onto `time_bins`. There is no use of per-frame timing or explicit go-cue subtraction.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. No stronger justification appears in the notes beyond getting a motion-energy stream onto the same decoder grid.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unreadable trajectory data produce all-NaN arrays; missing motion-energy files return `None`; too-short motion-energy trials are padded with all-NaN placeholders; unsupported neural layouts cause the whole session to be skipped. Later, non-finite movement bins default to class `0` because the temporary output arrays are initialized to zero and only finite entries are overwritten.

ii.
```python
except Exception:
    return np.full(time_bins.shape, np.nan, dtype=float)
```

```python
if path is None or not path.exists():
    return None
...
while len(out) < n_trials:
    out.append(np.full(1, np.nan))
```

```python
if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
    raise ValueError('unsupported clu structure')
```

```python
tmp = np.zeros(tong.shape, dtype=np.int64)
finite = np.isfinite(tong)
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
```

iii. The notes frame these mostly as pragmatic engineering fixes during debugging, especially for nested motion-energy layouts and incomplete trajectory extraction.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive part is `build_session`, not file loading. Within `build_session`, the repeated trajectory extraction/interpolation work and the per-cluster per-trial spike histograms dominate. The saved full-run log shows `load` calls taking about `0.01 s` while `build` calls take roughly `6-19 s` per session.

ii.
```python
for sess in sessions:
    with timed(f'load {sess.data_file.name}'):
        obj = load_mat_obj(sess.data_file)
    try:
        with timed(f'build {sess.data_file.name}'):
            neural, inp, out, bri = build_session(obj, sess)
```

iii. The runtime logs in `conversion_full_out.txt` make this clear: the build path is where the time is spent.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are straightforward optimization targets: the per-trial loop that histograms spikes cluster by cluster, the list-comprehension passes that recompute tongue and paw trajectories across all trials, and the final per-trial thresholding loop. The spike counting could have been vectorized across trials or cached more like the reference's `histogram2d` approach.

ii.
```python
for tr in trial_idx:
    ...
    for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
        spikes = tm_arr[tr_arr == tr1]
        if spikes.size:
            mat[ci], _ = np.histogram(spikes, bins=binedges)
```

```python
tongue_all = np.concatenate([... for tr in trial_idx]) if len(trial_idx) else np.array([])
paw_all = np.concatenate([... for tr in trial_idx]) if len(trial_idx) else np.array([])
...
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(...)
    paw = extract_hdf5_traj_velocity(...)
```

iii. The notes do not present these loops as intentional performance tradeoffs; they appear to be a consequence of getting a working pipeline assembled quickly.

## 11-c. What processing does the code repeat multiple times?

i. The AI repeatedly recomputes tongue and paw velocities for the same trials. Each feature is extracted once inside the main trial loop (where the result is unused), again across all trials to compute the session threshold, and again trial by trial to write thresholded outputs.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
paw_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
```

```python
tongue_all = np.concatenate([np.nan_to_num(extract_hdf5_traj_velocity(obj, tr, [...], time_bins), nan=np.nan) for tr in trial_idx]) if len(trial_idx) else np.array([])
paw_all = np.concatenate([np.nan_to_num(extract_hdf5_traj_velocity(obj, tr, [...], time_bins), nan=np.nan) for tr in trial_idx]) if len(trial_idx) else np.array([])
...
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
    paw = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
```

iii. The trajectory shows the kinematic path being patched several times during debugging, and this final repetition looks like residue from that iterative process.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is computing `tongue_vals` and `paw_vals` inside the first trial loop and then never using those arrays. The code also initializes movement outputs as all-zero placeholders before recomputing and overwriting them later, and it loads every candidate session once during `select_context_sessions` and again during actual conversion.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
paw_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
output_trials.append(np.vstack([
    np.full(time_bins.shape, lick_dir, dtype=np.int64),
    np.full(time_bins.shape, context, dtype=np.int64),
    np.full(time_bins.shape, outcome, dtype=np.int64),
    np.zeros(time_bins.shape, dtype=np.int64),
    np.zeros(time_bins.shape, dtype=np.int64),
    np.zeros(time_bins.shape, dtype=np.int64),
]))
```

iii. There is no explicit justification for these extra passes in the notes; they appear to be leftover scaffolding from intermediate versions of the converter.
