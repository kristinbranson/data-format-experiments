# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script only scans `data/Ephys_Behavior`, not the other dataset families. It pairs each `data_structure_<subject>_<date>.mat` file with an optional `motionEnergy_<subject>_<date>.mat` sidecar, then keeps only sessions whose `autowater` field contains both 0 and 1. Each session is loaded either with `h5py` for v7.3 MAT files or `scipy.io.loadmat` otherwise.

ii. 
```python
def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
    motion_map = {}
    for mf in sorted(base.glob('motionEnergy_*.mat')):
        ...
    for df in data_files:
        ...
        sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))

def load_mat_obj(path: Path):
    if is_hdf5_mat(path):
        return h5py.File(path, 'r')
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return x['obj'].flat[0]

sessions = discover_sessions()
sessions = select_context_sessions(sessions)
```

iii. In Step 5, the agent justified this as a hypothesis: use the “two-context ephys sessions” inferred from `Ephys_Behavior` sessions that contain both autowater and non-autowater trials, because it did not find a separate two-context folder.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the filename prefix captured by the regex in `discover_sessions()`. The final `subjects` list is the sorted set of all subject IDs present after `select_context_sessions()`, before later per-session build failures are removed.

ii. 
```python
m = re.match(r'^data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', df.name)
...
subj, day = m.group(1), m.group(2)
sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
...
subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
```

iii. The notes say subject metadata could come from filenames or `obj.meta`, but the final script chose the filename path and never switched to metadata fields.

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file is treated as one session. The file-level session order is preserved through the conversion, except that sessions can later be dropped if they fail during `build_session()` or end up with fewer than two valid trials.

ii. 
```python
for df in data_files:
    ...
    sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
...
for sess in sessions:
    ...
    neural, inp, out, bri = build_session(obj, sess)
    ...
    if len(neural) < 2:
        continue
```

iii. This follows the agent’s Step 2/Step 5 reading of the dataset as session-organized MAT files.

## 1-d. How are the data split into trials?

i. Trials are indexed from the length of the `goCue` event array. The script builds a boolean `valid` mask over those trial indices, then keeps `trial_idx = np.where(valid)[0]`. Each remaining trial becomes one neural matrix, one input array, and one output array.

ii. 
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
...
valid = np.ones(n_trials, dtype=bool)
...
trial_idx = np.where(valid)[0]
...
for tr in trial_idx:
    ...
    output_trials.append(...)
    input_trials.append(inp)
    neural_trials.append(mat)
```

iii. The agent’s notes identify per-trial processing as being organized around raw trial fields in `obj.bp`; the final code implements exactly that.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is implemented as a logical mask requiring: not `early`, `hit | miss` (which removes ignores), no stimulation if `stim.enable` exists, exactly one of `R` or `L`, and `autowater` in `{0,1}`. There is no later trial-count threshold like the paper’s “40 DR / 20 WC correct trials per direction” rule.

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

iii. The agent justified this from the methods excerpt and `getDefaultParams.m`: exclude early and ignore trials, use `stim.enable`, and derive context from `autowater`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `obj['obj']['clu']`, specifically the referenced `trial` and `trialtm` arrays for each cluster. No processed PSTHs or author-produced outputs are loaded; the script rebuilds trial-aligned spike counts directly from those raw cluster tables.

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

iii. The notes say neural data should come from `obj.clu` spike times / trialized spike data, matching the agent’s inspection of the HDF5 session files.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each cluster, the script selects spike times whose `trial` equals that trial index (1-based), then bins the corresponding `trialtm` values into 75 ms bins spanning -1.5 s to +1.5 s. The result is a per-trial `(n_neurons, 41)` spike-count matrix.

ii. 
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]])
...
mat = np.zeros((len(clu_trial), time_bins.size), dtype=np.float32)
tr1 = tr + 1
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
neural_trials.append(mat)
```

iii. In Step 5, the agent explicitly decided to use 75 ms bins and “convert to neuron x time trial matrices,” citing the decoder scripts’ `rez.binSize = 75`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The final script does not apply explicit neural quality filtering by firing rate, cluster quality, or the paper’s minimum-unit/session inclusion rule. It only skips sessions whose `clu` structure is unsupported or whose valid-trial count falls below two. This differs from the agent’s own notes, which planned low-firing-rate filtering.

ii. 
```python
if isinstance(obj, h5py.File):
    clu_root = obj['obj']['clu']
    clu = obj[clu_root[0,0]]
    if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
        raise ValueError('unsupported clu structure')
...
if len(neural) < 2:
    print(f'[warn] skipping {sess.data_file.name}: <2 valid trials', flush=True)
    continue
```

iii. The notes and trajectory say the agent intended to follow `removeLowFRClusters.m` and the paper’s firing-rate criteria, but that decision was not implemented in the final code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code assumes `clu/trialtm` is already trial-aligned and bins it around time 0 using the `goCue`-centered bin edges. It does not subtract `goCue` from spike times inside `build_session()`. The alignment decision is therefore “trust `trialtm` as pre-aligned to go cue.”

ii. 
```python
ALIGN_EVENT = 'goCue'
...
go = get_event(obj, ALIGN_EVENT)
...
spikes = tm_arr[tr_arr == tr1]
if spikes.size:
    mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. Step 5 says all modalities should be aligned to `goCue`; the final implementation operationalizes that for spikes by assuming the stored `trialtm` values already use that alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 75 ms bins. There is no additional rebinning step beyond the direct histogramming/interpolation onto that 75 ms grid.

ii. 
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The trajectory and notes repeatedly cite the decoder scripts’ `rez.binSize = 75` ms as the main justification.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw time-series variable. Instead, it is constructed from the fixed analysis window (`T_START`, `T_END`) and bin size around the chosen alignment event `goCue`.

ii. 
```python
BIN_SIZE_S = 0.075
ALIGN_EVENT = 'goCue'
T_START = -1.5
T_END = 1.5
...
inp = time_bins[None, :].astype(np.float32)
```

iii. The Step 5 mapping plan says this input should be the “time-from-go-cue vector repeated for each trial/time bin.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The processing is just constructing `np.arange(-1.5, 1.5, 0.075)` and storing it as a single-row array for every retained trial. There is no trial-specific adjustment once the session is on the common time axis.

ii. 
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
...
inp = time_bins[None, :].astype(np.float32)
input_trials.append(inp)
```

iii. The notes state this was required by the decoder specification rather than discovered from a raw dataset variable.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_bins` array is used for neural bin edges, time input, tongue velocity, paw velocity, and motion energy. So the input is aligned by construction to the same per-trial 41-bin grid as the neural data.

ii. 
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]])
...
inp = time_bins[None, :].astype(np.float32)
```

iii. Step 5 says all modalities should share the go-cue-aligned 75 ms time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial boolean fields `bp.R` and `bp.L`.

ii. 
```python
right = get_trial_bool(bp, 'R')
left = get_trial_bool(bp, 'L')
...
lick_dir = 1 if bool(right[tr]) else 0
```

iii. The notes explicitly say to use trial labels `R`/`L` rather than reconstructing direction from lick timestamps.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After filtering to trials with exactly one of `R` or `L`, the code maps right to 1 and left to 0, then broadcasts that scalar label across all 41 time bins for the trial.

ii. 
```python
if right is not None and left is not None:
    valid &= (right ^ left)
...
lick_dir = 1 if bool(right[tr]) else 0
...
np.full(time_bins.shape, lick_dir, dtype=np.int64)
```

iii. The justification in Step 5 is that the decoder task specifies left = 0, right = 1.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` field.

ii. 
```python
autowater = get_trial_bool(bp, 'autowater')
...
context = 0 if bool(autowater[tr]) else 1
```

iii. The notes tie this to `getDefaultParams.m` and the context decoder scripts, where autowater trials correspond to WC and non-autowater trials correspond to DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code recodes `autowater=True` as WC = 0 and `autowater=False` as DR = 1, then broadcasts the per-trial context label across time bins.

ii. 
```python
context = 0 if bool(autowater[tr]) else 1
...
np.full(time_bins.shape, context, dtype=np.int64)
```

iii. Step 5 states this recoding explicitly: convert raw `autowater` labels to the task convention `WC=0`, `DR=1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial boolean fields `hit` and `miss`.

ii. 
```python
hit = get_trial_bool(bp, 'hit')
miss = get_trial_bool(bp, 'miss')
...
outcome = 1 if bool(hit[tr]) else 0
```

iii. The notes map outcome to hit/miss after excluding early and ignore trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Only trials satisfying `hit | miss` are kept. Among those, `hit` becomes correct = 1 and everything else becomes incorrect = 0, again broadcast across all time bins for the trial.

ii. 
```python
if hit is not None and miss is not None:
    valid &= (hit | miss)
...
outcome = 1 if bool(hit[tr]) else 0
...
np.full(time_bins.shape, outcome, dtype=np.int64)
```

iii. The agent’s Step 5 mapping says outcome should be derived from `hit`/`miss`, with incorrect = 0 and correct = 1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the HDF5 `obj['obj']['traj']` structures: `featNames`, `ts`, and `frameTimes`. The code searches for one of several tongue-related feature names and uses the first match.

ii. 
```python
traj = obj[traj_ds[i,0]]
names_arr = obj[traj['featNames'][trial_index0,0]][()]
...
for cand in ['top_tongue', 'topleft_tongue', 'bottom_tongue',
             'bottomleft_tongue', 'tongue', 'left_tongue', 'right_tongue']:
    if cand in names:
        feat_idx = names.index(cand)
        break
ts = np.asarray(obj[traj['ts'][trial_index0,0]][()]).astype(float)
ft = np.asarray(obj[traj['frameTimes'][trial_index0,0]][()]).reshape(-1).astype(float)
```

iii. The notes said tongue outputs should come from `obj.traj` / kinematic features, informed by `loadKinData.m`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The selected tongue feature contributes its x/y coordinates. The code computes frame-to-frame Euclidean speed, divides by frame-to-frame `dt`, converts frame times to go-cue-relative time, linearly interpolates onto the 75 ms grid, and fills interior NaNs by interpolation over index positions.

ii. 
```python
xy = ts[feat_idx, :2, :]
dx = np.diff(xy[0], prepend=xy[0,0])
dy = np.diff(xy[1], prepend=xy[1,0])
dt = np.diff(ft, prepend=ft[0])
...
speed = np.sqrt(dx*dx + dy*dy) / dt
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
...
vals = np.interp(idx, idx[good], vals[good])
```

iii. The agent did not document a more detailed formula than “compute scalar tongue speed over aligned time bins”; the final code fills that gap with its own ad hoc speed calculation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All tongue-speed values from all retained trials in a session are concatenated, the session-wise 50th percentile is computed, and each finite bin is labeled low = 0 or high = 1 by that threshold.

ii. 
```python
tongue_all = np.concatenate([... for tr in trial_idx]) if len(trial_idx) else np.array([])
tongue_thr = np.nanpercentile(tongue_all, 50) if tongue_all.size and np.isfinite(tongue_all).any() else np.nan
...
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
output_trials[i][3] = tmp
```

iii. Step 5 explicitly says tongue velocity should be discretized using a per-session median threshold, because the task instructions require it.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Alignment is performed by subtracting the trial’s `goCue` time from the trajectory frame times, then interpolating onto the same `time_bins` grid used for spikes and inputs.

ii. 
```python
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The notes say all modalities should be go-cue aligned on a common 75 ms axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the same `traj` HDF5 structures as tongue velocity, but with paw-related candidate feature names.

ii. 
```python
for cand in ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw']:
    if cand in names:
        feat_idx = names.index(cand)
        break
```

iii. The agent’s mapping notes treat paw velocity exactly like tongue velocity: use kinematic trajectories from `obj.traj`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script uses the same speed-estimation procedure as for tongue velocity: compute x/y speed from successive frames, divide by frame `dt`, align frame times to `goCue`, interpolate onto the 75 ms bins, then threshold later.

ii. 
```python
paw_vals = extract_hdf5_traj_velocity(
    obj, tr, ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins
) if isinstance(obj, h5py.File) else np.full(time_bins.shape, np.nan)
...
paw_all = np.concatenate([np.nan_to_num(extract_hdf5_traj_velocity(obj, tr, ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins), nan=np.nan) for tr in trial_idx])
```

iii. As with tongue velocity, the notes only justify the broad idea (“compute scalar paw speed over aligned time bins”), not the exact formula chosen in code.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session-wide median of all finite paw-speed bins is used as the low/high threshold.

ii. 
```python
paw_thr = np.nanpercentile(paw_all, 50) if paw_all.size and np.isfinite(paw_all).any() else np.nan
...
tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
output_trials[i][4] = tmp
```

iii. The Step 5 plan explicitly says paw velocity should be thresholded at the per-session 50th percentile to match the task instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same frame-time minus `goCue` alignment and interpolation onto `time_bins` as tongue velocity, so it shares the neural time grid.

ii. 
```python
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. This follows the same “all modalities aligned to go cue” rationale recorded in Step 5.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the sidecar `motionEnergy_*.mat` file, specifically the MATLAB struct `me` and its `data` field. The code ignores `me.moveThresh`.

ii. 
```python
x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
me = x['me'].flat[0]
data = np.asarray(unwrap_me_data(me)).reshape(-1)
...
trial = np.asarray(elem).reshape(-1).astype(float)
```

iii. The notes identify `motionEnergy_*.mat` `me.data` as the source variable and cite `loadMotionEnergy.m` as the reference alignment function.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each retained trial, the code takes the raw motion-energy vector, assumes it spans exactly the fixed `[-1.5, 1.5]` analysis window, and linearly interpolates it onto `time_bins`. It does not use the reference code’s 400 Hz frame-time construction, `-0.5` offset, or nearest-value fill of leading NaNs.

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

iii. The notes had already documented the reference logic from `loadMotionEnergy.m`, but the final code replaced that with a simpler approximation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. After session-level interpolation, all motion-energy bins from the session are concatenated, a 50th-percentile threshold is computed, and each finite bin is labeled low or high. Missing bins stay at the default class 0.

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

iii. Step 5 says motion energy should be binarized with a per-session median threshold because the decoder task requires categorical outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned only by resampling each raw per-trial vector onto the fixed `time_bins` axis. Unlike the trajectory code, it does not use raw event times or frame times relative to `goCue`.

ii. 
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. The agent’s notes say motion energy should share the neural time axis, but the final implementation uses a simplified surrogate alignment rather than the reference alignment procedure it had documented earlier.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or awkward data by coercion rather than explicit QC. `get_trial_bool()` converts unreadable values to `False`. Missing motion-energy files return `None`; missing motion-energy trials are padded with one-point `NaN` arrays. Missing trajectory features return all-`NaN` vectors; later, because output arrays are initialized to zeros and only finite bins are rewritten, many missing kinematic bins silently become class 0. Unsupported `clu` structures cause the whole session to be skipped.

ii. 
```python
for i, v in enumerate(arr):
    try:
        out[i] = bool(v)
    except Exception:
        out[i] = False
...
while len(out) < n_trials:
    out.append(np.full(1, np.nan))
...
return np.full(time_bins.shape, np.nan, dtype=float)
...
except Exception as e:
    print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True)
    continue
```

iii. There is no strong documented justification for these fallback rules beyond making the conversion run end-to-end.

## 11-a. What are the most time-consuming steps of the code?

i. The main bottlenecks are per-cluster spike histogramming and repeated per-trial trajectory extraction/interpolation. The conversion log shows `build_session()` taking several seconds per session, and the code structure makes those two nested-loop regions the obvious hot spots.

ii. 
```python
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
...
tongue_all = np.concatenate([ ... extract_hdf5_traj_velocity(...) for tr in trial_idx])
paw_all = np.concatenate([ ... extract_hdf5_traj_velocity(...) for tr in trial_idx])
...
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(obj, tr, ...)
    paw = extract_hdf5_traj_velocity(obj, tr, ...)
```

iii. In Step 6, the notes reserved a place for “Code inefficiencies identified,” but the final implementation still reflects these expensive loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron histogram loop, the repeated list-comprehension calls to `extract_hdf5_traj_velocity()`, and the final per-trial threshold assignment loops are all vectorizable or cacheable. The script computes the same trajectory features multiple times per trial instead of reusing one aligned array.

ii. 
```python
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    ...
for i in range(n_clu):
    ...
tongue_all = np.concatenate([ ... for tr in trial_idx])
paw_all = np.concatenate([ ... for tr in trial_idx])
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(obj, tr, ...)
    paw = extract_hdf5_traj_velocity(obj, tr, ...)
```

iii. The notes acknowledged a need to identify inefficiencies, but no explicit optimization rationale appears in the final code.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly re-reads and re-processes the same trajectory streams. It calls `extract_hdf5_traj_velocity()` once inside the main trial loop, again across all trials to build session thresholds, and then again inside the final threshold-application loop. Motion-energy session statistics also repeatedly recompute concatenations and medians.

ii. 
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr, ...)
paw_vals = extract_hdf5_traj_velocity(obj, tr, ...)
...
tongue_all = np.concatenate([ ... extract_hdf5_traj_velocity(obj, tr, ...) for tr in trial_idx])
paw_all = np.concatenate([ ... extract_hdf5_traj_velocity(obj, tr, ...) for tr in trial_idx])
...
tong = extract_hdf5_traj_velocity(obj, tr, ...)
paw = extract_hdf5_traj_velocity(obj, tr, ...)
```

iii. No separate justification was documented for this repetition; it appears to be an implementation shortcut.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Inside the main trial loop, `tongue_vals` and `paw_vals` are computed and then thrown away immediately. More broadly, the code computes continuous tongue, paw, and motion-energy traces only to threshold them into binary outputs and discard the continuous values. It also keeps a `subjects` entry for some sessions that are later skipped.

ii. 
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
paw_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
output_trials.append(np.vstack([
    ...,
    np.zeros(time_bins.shape, dtype=np.int64),
    np.zeros(time_bins.shape, dtype=np.int64),
    np.zeros(time_bins.shape, dtype=np.int64),
]))
...
subjects = sorted({s.subject for s in sessions})
```

iii. The notes justified binary behavioral outputs because the task specification requires categorical decoder targets, but they did not justify recomputing values that are never used.
