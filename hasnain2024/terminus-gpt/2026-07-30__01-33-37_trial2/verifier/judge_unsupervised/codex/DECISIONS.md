# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans every immediate subdirectory under `data/`, pairs `data_structure_*.mat` with `motionEnergy_*.mat` by shared filename stem, and only processes stems that have both files. `data_structure` files are read through a custom HDF5 loader, while motion-energy files are read with `scipy.io.loadmat`. Sessions are skipped if the `data_structure` file is unreadable as HDF5 or if motion energy is missing.

ii. ```python
def discover_sessions(data_root):
    sessions = {}
    for subdir in Path(data_root).iterdir():
        ...
        for f in subdir.glob('data_structure_*.mat'):
            stem = f.stem.replace('data_structure_', '')
            sessions.setdefault(stem, {})['data_structure'] = f
        for f in subdir.glob('motionEnergy_*.mat'):
            stem = f.stem.replace('motionEnergy_', '')
            sessions.setdefault(stem, {})['motion_energy'] = f

def load_data_structure(path):
    try:
        hfile = h5py.File(path, 'r')
    except OSError:
        return None

def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']

keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent explicitly justified this as “Restrict to sessions with all required outputs available.” Later trajectory steps added the unreadable-HDF5 skip path after discovering paired sessions that the loader could not open.

## 1-b. How are the data split into subjects?

i. Subject identity is inferred from the first underscore-delimited token of the session stem, e.g. `EKH1_2021-08-07 -> EKH1`.

ii. ```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])
...
subject, _ = infer_subject_session(stem)
```

iii. The notes repeatedly describe subject/date parsing from filenames, and the converted dataset’s `subjects` / `subject_idx` arrays are built from this filename convention.

## 1-c. How are the data split into sessions?

i. Each unique paired filename stem is treated as one session. Session order is the sorted order of these stems.

ii. ```python
for f in subdir.glob('data_structure_*.mat'):
    stem = f.stem.replace('data_structure_', '')
    sessions.setdefault(stem, {})['data_structure'] = f
...
keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
for stem in keys:
    sess = process_session(stem, sessions[stem], edges, centers)
```

iii. The agent’s notes describe the reference analyses as session-by-session, so it mirrored that organization by using one stem-pair as one session.

## 1-d. How are the data split into trials?

i. Trial count comes from `bp.Ntrials`. The script builds a Boolean valid-trial mask over that length, then keeps `np.where(trial_mask)[0]` as the trial list for the session.

ii. ```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    ...
    return mask

def build_trial_labels(obj, trial_mask):
    idx = np.where(trial_mask)[0]
    ...
    return idx, lick_dir, context, outcome
```

iii. The notes state that `bp.Ntrials` and trial-wise behavioral arrays define the native trial structure, and that converted trial count should equal the raw valid-trial count after filtering.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `early` and `no` trials. No additional trial-level QC is applied before label construction, beyond later skipping whole sessions that fail neural or feature availability checks.

ii. ```python
for name in ['early', 'no']:
    if name in bp:
        arr = np.array(bp[name], dtype=float).reshape(-1)
        if arr.size == n:
            mask &= (arr == 0)
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as matching the methods and reference analysis code: “Exclude early and ignore trials.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `obj.clu.trialtm` and `obj.clu.trial`, with `tm`, `site`, and `quality` loaded but not used for the final neural matrices.

ii. ```python
clu_ref = obj['clu'][()].reshape(-1)[0]
clu = deref(h, clu_ref)
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    ...

trialtm = clu['trialtm']
trialid = clu['trial']
```

iii. The Step 5 mapping table explicitly names ``obj.clu.trialtm + obj.clu.trial`` as the source for `neural`.

## 2-b. How is the `neural` data processed?

i. For each unit, the script treats `trialtm` as already trial-aligned spike times, histograms spikes into a fixed `[-2.5, 2.5)` window using 75 ms bins, and stacks kept units into a `(n_neurons, n_timepoints)` trial matrix of spike counts.

ii. ```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)

for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    st = np.array(st, dtype=float).reshape(-1)
    tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
        unit_trials.append(counts.astype(np.float32))
...
return [np.stack(x, axis=0) for x in neural_trials], np.array(kept_units, dtype=int)
```

iii. The notes justify this as “Use spike counts on a fixed time grid” and link it to the reference decoder’s 75 ms binning, but they do not claim to reproduce more detailed upstream PSTH/smoothing code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units with estimated firing rate `<= 1 Hz` are dropped. Entire sessions are skipped if they lack `trialtm` / `trial`, if fewer than 10 units survive, or if fewer than 2 valid trials remain. The `quality` field is loaded but unused.

ii. ```python
if 'trialtm' not in clu or 'trial' not in clu:
    return None, None
...
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
...
if len(neural) < 2 or len(kept_units) < 10:
    print(f'  skip_reason: insufficient_neural trials={len(neural)} units={len(kept_units)}', flush=True)
    return None
```

iii. The agent justified this with the methods summary “all units > 1 Hz” and “sessions need >= 10 units,” and also noted that the default reference params used `quality = {'all'}`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script does not realign spikes from absolute times. It assumes `clu.trialtm` is already aligned to the relevant event and bins those values on a grid centered at zero, while separately describing the alignment event as go cue in metadata.

ii. ```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2

counts, _ = np.histogram(st[tr == raw_t], bins=edges)
...
'temporal_alignment_event': 'Go cue onset',
```

iii. The notes say go cue is the reference alignment event in the original code, and the agent inferred that `trialtm` was already in that aligned coordinate frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 75 ms bins. For neural data, the only temporal binning is direct histogramming into that 75 ms grid; there is no second-stage rebin after that.

ii. ```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
...
'time_bin_size': 75.0,
```

iii. The agent explicitly justified 75 ms bins from its reading of `DLC_ContextDecoding.m`, which it summarized in the notes as `rez.binSize = 75` ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from an explicit raw session variable. The script constructs a synthetic fixed time grid and treats it as “time from go cue” for every trial.

ii. ```python
edges, centers = build_time_grid()
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The Step 5 notes say “Represent decoder input as continuous time from go cue” and treat that as a design choice tied to the decoder specification rather than to a specific raw field.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes bin centers once from the fixed 75 ms grid and then copies the same `(1, T)` array into every trial in every session.

ii. ```python
edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
centers = edges[:-1] + bin_size_s / 2
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The agent’s justification in the notes is that the decoder has a single continuous time input, shared across all aligned trials.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the same `centers` array that defines the neural spike-count bins, so the input time vector is aligned bin-for-bin with each neural matrix.

ii. ```python
edges, centers = build_time_grid()
neural, kept_units = bin_spikes_for_session(obj, trial_mask, edges)
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes explicitly describe the input as a “single input channel shared across trials/sessions” on the common go-cue-aligned grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from the behavioral arrays `bp.L` and `bp.R`.

ii. ```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The Step 5 mapping table names ``obj.bp.L`` and ``obj.bp.R`` as the source variables for lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. On each valid trial, the script sets the label to `1` if `R > L`, otherwise `0`, and then repeats that trial label across all 66 time bins.

ii. ```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, lick_dir[i], dtype=np.int64),
```

iii. The notes describe this as a per-trial categorical label with coding `left=0, right=1`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `bp.autowater`.

ii. ```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. The notes say context comes from “`bp.autowater` / trial groups” and link that choice to `getBlockNum_AltContextTask.m`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script maps `autowater > 0` to `0` and `autowater == 0` to `1`, interpreting those categories as `WC=0` and `DR=1`, then repeats the trial label across all time bins.

ii. ```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
...
np.full(n_bins, context[i], dtype=np.int64),
```

iii. The justification is tentative even in the notes: the agent wrote “Verify final sign/coding against trial-group logic,” but still implemented the direct `autowater` mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived from `bp.hit` and `bp.miss`.

ii. ```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The Step 5 mapping table explicitly names ``obj.bp.hit`` and ``obj.bp.miss`` for outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script labels each valid trial as correct if `hit > miss`, otherwise incorrect, and repeats that value across all time bins.

ii. ```python
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, outcome[i], dtype=np.int64),
```

iii. The notes describe this as the per-trial categorical mapping `incorrect=0, correct=1`.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It is derived from the side-camera trajectory view: `traj_views[0]['featNames']`, `traj_views[0]['ts']`, `traj_views[0]['frameTimes']`, plus per-trial `bp.ev.goCue`.

ii. ```python
side = traj_views[0]
side_names = [str(x) for x in side.get('featNames', [])]
go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)
...
ts_side = side['ts'][tr]
ft_side = side['frameTimes'][tr]
```

iii. The trajectory notes say the final fix was to load both camera views and use “side-view tongue plus bottom-view paw features.”

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script picks the first matching tongue feature among `tongue`, `left_tongue`, and `right_tongue`; interpolates x and y coordinates onto the neural time grid using `(frameTimes - 0.5) - goCue`; computes `np.gradient` of x and y; and converts that to Euclidean speed magnitude. For tongue, NaN gradients are simply zero-filled.

ii. ```python
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
...
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

iii. The notes justify the broad approach from DeepLabCut/video decoding references, but the detailed implementation came from the agent’s own reconstruction after discovering the two-camera layout in `WorkingWithDataObjs.m`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After continuous tongue-speed traces are computed for all valid trials in a session, the script thresholds them at the session-wide median of all values: values strictly greater than the median become `1`, otherwise `0`.

ii. ```python
tongue = np.stack(tongue, axis=0)
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The Step 5 notes say the decision was to “Discretize continuous behavioral outputs per session at 50th percentile.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is aligned per trial by subtracting that trial’s go-cue time from the side-camera `frameTimes`, applying the additional `-0.5` s synchronization offset, and interpolating onto the same `centers` grid used for neural data.

ii. ```python
go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)
...
rel_t = (ft - 0.5) - float(align_time)
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
```

iii. The justification came directly from the trajectory’s recovered reference-code comment in `WorkingWithDataObjs.m`: subtract `0.5` s from `frameTimes` to sync camera and spikeGLX recordings.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It is derived from the bottom-camera trajectory view: `traj_views[1]['featNames']`, `traj_views[1]['ts']`, `traj_views[1]['frameTimes']`, plus `bp.ev.goCue`.

ii. ```python
bottom = traj_views[1]
bottom_names = [str(x) for x in bottom.get('featNames', [])]
...
ts_bot = bottom['ts'][tr]
ft_bot = bottom['frameTimes'][tr]
```

iii. The notes say the decisive bug fix was realizing paw features live in the second camera view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script picks the first matching paw feature among `top_paw`, `bottom_paw`, and `paw`; interpolates x and y onto the aligned neural grid; takes gradients; replaces NaN gradients with the median gradient; recenters x/y gradients by subtracting their medians; and then takes Euclidean speed magnitude.

ii. ```python
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
...
xv = np.gradient(x)
yv = np.gradient(y)
if np.all(np.isnan(xv)) or np.all(np.isnan(yv)):
    return np.zeros(len(centers), dtype=np.float32)
xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
xv = xv - np.nanmedian(xv)
yv = yv - np.nanmedian(yv)
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The agent justified this as a practical reconstruction of a paw-velocity signal from DLC trajectories after learning that bottom-camera paw features exist, but it did not cite a matching reference function with the same exact formula.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The continuous paw-speed traces are thresholded at the session-wide median, with values strictly greater than the median labeled `1` and all others `0`.

ii. ```python
paw = np.stack(paw, axis=0)
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. The Step 5 notes justify this with the decoder requirement to binarize movement variables at the 50th percentile per session.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. It uses the same per-trial `(frameTimes - 0.5) - goCue` alignment and the same `centers` grid as neural and tongue data.

ii. ```python
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
...
rel_t = (ft - 0.5) - float(align_time)
```

iii. The justification is the same synchronization/alignment logic the agent recovered from `WorkingWithDataObjs.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is derived from the MATLAB `me` struct loaded from `motionEnergy_*.mat`, specifically the `data` field after normalization by `normalize_motion_energy_data`.

ii. ```python
def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']
...
me = load_motion_energy(files['motion_energy'])
me_data = normalize_motion_energy_data(me)
```

iii. The Step 5 mapping table names ``me.data`` as the source of motion energy and notes that `me.moveThresh` is present but unused.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The loader normalizes several possible MATLAB container layouts into a Python list of per-trial traces. Each trace is then linearly interpolated from a normalized `0..1` sample axis to the 66-bin neural grid.

ii. ```python
def normalize_motion_energy_data(me):
    ...
    if isinstance(data, np.ndarray):
        if data.dtype == object:
            return list(data.reshape(-1))
...
def rebin_variable_trace(values, n_bins):
    arr = np.array(values, dtype=float).reshape(-1)
    xp = np.linspace(0, 1, arr.size)
    xnew = np.linspace(0, 1, n_bins)
    return np.interp(xnew, xp, arr).astype(np.float32)
...
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The notes justify this as handling variable-length per-trial traces and rebinned behavioral streams, but the detailed normalized-time interpolation was the agent’s own implementation choice.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion-energy traces are stacked across valid trials and thresholded at the session-wide median, again using a strict `>` comparison.

ii. ```python
me_stack = np.stack(me_trials, axis=0)
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. The Step 5 notes justify this the same way as tongue and paw: a per-session median split to satisfy the decoder spec.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. There is no explicit event-time alignment. The code assumes trial correspondence is enough, and simply resamples each raw motion-energy trace to the neural bin count.

ii. ```python
if len(me_data) == 0 or max(idx) >= len(me_data):
    ...
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The notes acknowledge that the traces are variable-length and need “rebin/alignment to a common time grid,” but the final code implements only length normalization, not alignment from raw timestamps or go-cue times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly handles missing or malformed data by skipping sessions, zero-filling, or using simple fallbacks. Unreadable HDF5 files return `None`; missing `trialtm` / `trial`, missing motion-energy traces, or missing paw features cause session skips; empty continuous traces return zeros; `frameTimes` mismatches are replaced by either an assumed 400 Hz clock or linear interpolation; and NaNs in trajectory velocities are zero-filled for tongue or median-filled for paw.

ii. ```python
except OSError:
    return None
...
if 'trialtm' not in clu or 'trial' not in clu:
    return None, None
...
if arr.size == 0:
    return np.zeros(n_bins, dtype=np.float32)
...
if ft.size == 0:
    ft = np.arange(1, xy.shape[0] + 1, dtype=float) / 400.0
else:
    ft = np.linspace(ft.min(), ft.max(), xy.shape[0])
...
if tongue_disc is None or paw_disc is None:
    print('  skip_reason: no_paw_feature', flush=True)
    return None
```

iii. The trajectory shows these behaviors were added reactively while debugging edge cases, especially unreadable `data_structure` files and the missing second camera view.

## 11-a. What are the most time-consuming steps of the code?

i. The main expensive steps are the nested unit-by-trial spike histogramming, per-trial interpolation/gradient computation for tongue and paw, and per-trial motion-energy interpolation.

ii. ```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)

for tr in trial_idx:
    ...
    tongue.append(interp_feature_velocity(...))
    paw.append(interp_feature_velocity(...))

me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The notes’ Step 6 and later validation steps mention timing/efficiency concerns, and these loops are the dominant repeated computations visible in the final code.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit/per-trial histogram loop, the per-trial tongue/paw interpolation loop, the list comprehension over motion-energy traces, and the loop that builds output matrices one trial at a time are all vectorization candidates.

ii. ```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(...)

for tr in trial_idx:
    ...

me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]

for i in range(len(idx)):
    outputs.append(np.vstack([...]))
```

iii. The agent did not fully optimize these sections; Step 6 notes explicitly leave “Code inefficiencies identified” and “Code speedups added” mostly undocumented.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly converts arrays with `np.array(...).reshape(-1)`, repeats the same interpolation logic for different behavioral traces, and duplicates per-trial categorical labels across all time bins with `np.full`.

ii. ```python
st = np.array(st, dtype=float).reshape(-1)
tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
...
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
...
np.full(n_bins, lick_dir[i], dtype=np.int64),
np.full(n_bins, context[i], dtype=np.int64),
np.full(n_bins, outcome[i], dtype=np.int64),
```

iii. This repetition is implicit in the final implementation; the agent’s notes do not claim an optimized or shared abstraction for these operations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader reads `tm`, `site`, `quality`, `meta`, `fn`, and `NdroppedFrames` even though the final dataset does not use them. The code also computes continuous tongue, paw, and motion-energy traces only to discard them immediately after thresholding, and it parses the session date string in `infer_subject_session` even though only the subject prefix is used.

ii. ```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    ...
for name in ['ts', 'frameTimes', 'fn', 'NdroppedFrames']:
    ...
for name in meta.keys():
    ...
return parts[0], '_'.join(parts[1:])
...
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. These are side effects of a general-purpose loader plus a binarized output format. The notes never claim these extra loaded fields are used downstream.
