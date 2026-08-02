# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all `data_structure_*.mat` files from `data/Ephys_Behavior/` and pairs them with corresponding `motionEnergy_*.mat` sidecar files. Only sessions from the `Ephys_Behavior` dataset family are used (other families like `RandomizedDelay_Ephys_Behavior`, `DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior` are ignored). Files are loaded using `h5py` for HDF5-format `.mat` files and `scipy.io.loadmat` for standard `.mat` files (motion energy). Subject and date are parsed from the filename pattern `data_structure_<SUBJECT>_<DATE>.mat`.

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

iii. The AI identified from the reference code and paper that the `Ephys_Behavior` directory contains the relevant two-context task data. The AI noted in CONVERSION_NOTES.md that the context decoder scripts define conditions using `autowater` trial labels within `Ephys_Behavior` data, and that the two-context task is identified by sessions containing both autowater=0 (DR) and autowater=1 (WC) trials.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are extracted from filenames using regex on `data_structure_<SUBJECT>_<DATE>.mat`. Unique subjects are sorted alphabetically and assigned integer indices. Each session maps to its subject via `subj_to_idx`.

ii.
```python
subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
# ...
data['subject_idx'].append(subj_to_idx[sess.subject])
```

iii. The AI used filename parsing rather than metadata fields (`obj.meta.anm`) within the HDF5 files. CONVERSION_NOTES.md documents that `obj.meta` contains `anm` (animal) metadata but the implementation relies on filenames, which is a reasonable and equivalent approach.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file represents one session. Sessions are processed sequentially. After context filtering, 22 of 25 sessions pass the autowater filter; 1 additional session (JEB6_2021-04-18) is skipped due to an unsupported `clu` structure, yielding 21 sessions.

ii.
```python
for sess in sessions:
    with timed(f'load {sess.data_file.name}'):
        obj = load_mat_obj(sess.data_file)
    try:
        with timed(f'build {sess.data_file.name}'):
            neural, inp, out, bri = build_session(obj, sess)
    except Exception as e:
        print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True)
        continue
```

iii. The AI documented in CONVERSION_NOTES.md that the paper reports 12 sessions from 6 mice for the two-context task, but the converter yields 21 sessions from 10 subjects. The AI acknowledged this discrepancy but did not apply the paper's behavioral inclusion criteria (>=40 correct DR trials/direction and >=20 correct WC trials/direction) to reduce the session count.

## 1-d. How are the data split into trials?

i. Trials are identified by the go cue event array `obj.bp.ev.goCue`, which defines `n_trials`. Valid trials are selected using a boolean mask that filters out early lick trials, ignore trials (neither hit nor miss), stimulation trials, ambiguous direction trials, and invalid autowater values.

ii.
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
# ... filtering ...
trial_idx = np.where(valid)[0]
```

iii. Trials are indexed by the 0-based position in the event arrays. The AI used `trial_idx` to iterate over valid trials and extract per-trial neural, input, and output data.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies five filters: (1) exclude early lick trials (`~early`), (2) require hit or miss (`hit | miss`, excluding ignore/no-response), (3) exclude optogenetic stimulation trials (`~stim_enable`), (4) require exactly one lick direction (`right XOR left`), (5) require valid autowater value (`autowater in {0,1}`). Unlike the reference code's `getDefaultParams.m` which uses hit-only conditions for the four primary PSTH conditions, the AI includes miss trials to keep the `outcome` decoder output non-degenerate.

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

iii. The AI documented this tradeoff in CONVERSION_NOTES.md: "enforcing the exact hit-only/no-stim conditions from `getDefaultParams.m` makes the `outcome` output degenerate (all correct), which conflicts with the decoder task requirement to predict incorrect vs correct outcome." The AI also did not implement the paper's session-level behavioral inclusion criteria (>=40 correct DR trials/direction, >=20 correct WC trials/direction) or the minimum 10-unit session inclusion criterion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu` in the HDF5 files. Each cluster (unit) has `trial` (1-indexed trial number for each spike) and `trialtm` (spike time relative to the trial alignment event, already aligned to go cue). All clusters from the first element of `obj.clu` are used.

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

iii. The AI identified through data exploration that `clu.trialtm` contains spike times already aligned to the go cue event. This was confirmed by inspecting actual values and verifying they fall within the expected time range.

## 2-b. How is the `neural` data processed?

i. Spike times from `clu.trialtm` are binned into 75 ms bins using `np.histogram`. Bin edges are defined centered on bin centers: `time_bins +/- BIN_SIZE_S/2`. The result is spike counts (not rates) per neuron per time bin, stored as float32.

ii.
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
# ...
mat = np.zeros((len(clu_trial), time_bins.size), dtype=np.float32)
tr1 = tr + 1
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
neural_trials.append(mat)
```

iii. The AI chose 75 ms bins based on the reference code's `rez.binSize = 75` in the decoder scripts. The reference code's `getDefaultParams.m` uses `params.dt = 1/200` (5 ms native resolution) with smoothing (`params.smooth = 15`), but the decoder scripts re-bin to 75 ms. The AI chose to bin directly at 75 ms rather than applying smoothing at the native resolution first.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT apply any firing rate filtering to neurons. All clusters present in `obj.clu` are included regardless of firing rate or quality. The reference code uses `removeLowFRClusters` with a threshold of 0.5 Hz (from `getDefaultParams.m`: `params.lowFR = 0.5`), and the paper states units with FR > 1 Hz are included in most analyses.

ii. There is no filtering code in the implementation. All `n_clu` clusters are included:
```python
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    # all clusters included, no FR check
```

iii. The AI noted in CONVERSION_NOTES.md Step 5 that "Apply low firing-rate filtering consistent with `removeLowFRClusters.m`" was planned, but it was never implemented in the final code. The verification output shows 5,512 total ALM neurons across 21 sessions (mean 262.5/session), which is much higher than the paper's 522 units across 12 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. The `clu.trialtm` field already contains spike times relative to the go cue, so no additional alignment is needed. Spikes for a given trial are selected by matching `clu.trial == trial_number + 1` (1-indexed).

ii.
```python
ALIGN_EVENT = 'goCue'
# ...
spikes = tm_arr[tr_arr == tr1]  # tr1 = tr + 1 (1-indexed)
mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The AI confirmed through data inspection that `trialtm` values fall within the expected range relative to go cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 75 ms (`BIN_SIZE_S = 0.075`). The time window spans -1.5 s to +1.5 s relative to go cue, yielding 41 time bins. No rebinning is applied — spikes are directly histogrammed into 75 ms bins. The reference code uses a native resolution of 5 ms (`params.dt = 1/200`) with Gaussian smoothing, then re-bins to 75 ms for decoding. The AI skips the intermediate smoothing step.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The AI documented in CONVERSION_NOTES.md that the decoder scripts use `rez.binSize = 75` ms and that this was chosen as the bin size for conversion.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time bin centers array, which is computed analytically from the time window parameters. It does not directly derive from any raw data variable — it is the time axis of the trial-aligned data.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
inp = time_bins[None, :].astype(np.float32)
```

iii. The AI constructed the time axis from the alignment parameters. This is consistent with the instruction to provide "time from go cue onset in seconds" as a continuous, time-varying decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time bins are generated using `np.arange` from -1.5 to +1.5 s in steps of 0.075 s. The same time vector is used for every trial. It is reshaped to (1, n_timepoints) for each trial.

ii.
```python
inp = time_bins[None, :].astype(np.float32)
input_trials.append(inp)
```

iii. No complex processing is needed — this is a deterministic time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time bins used for the input are the same bin centers used for the neural spike histogram edges. Both share the identical `time_bins` array, ensuring perfect alignment.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
# Used for both:
inp = time_bins[None, :]  # input
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]])  # neural
```

iii. The AI ensured alignment by using the same `time_bins` array for all data streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right lick) and `obj.bp.L` (left lick) boolean trial-level fields.

ii.
```python
right = get_trial_bool(bp, 'R')
left = get_trial_bool(bp, 'L')
# ...
lick_dir = 1 if bool(right[tr]) else 0
```

iii. The AI used the trial-level direction labels rather than post-hoc lick timestamps, consistent with the reference code's `getDefaultParams.m` which uses `R` and `L` conditions.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A per-trial scalar: `right=True` maps to 1, otherwise 0 (left). This is broadcast across all time bins as a constant per-trial value.

ii.
```python
lick_dir = 1 if bool(right[tr]) else 0
np.full(time_bins.shape, lick_dir, dtype=np.int64)
```

iii. The mapping matches the instruction: left=0, right=1. Broadcasting across time makes it time-varying in format but constant per trial.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, a per-trial boolean/integer field where 1=autowater (WC) and 0=non-autowater (DR).

ii.
```python
autowater = get_trial_bool(bp, 'autowater')
context = 0 if bool(autowater[tr]) else 1
```

iii. The AI identified from the reference code that `autowater` encodes the task context, with non-autowater trials being the DR (delayed response) condition and autowater trials being WC (water context).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The autowater value is inverted: autowater=True (WC) maps to 0, autowater=False (DR) maps to 1. This is broadcast across all time bins.

ii.
```python
context = 0 if bool(autowater[tr]) else 1
np.full(time_bins.shape, context, dtype=np.int64)
```

iii. The mapping matches the instruction: WC=0, DR=1. The AI documented this convention in CONVERSION_NOTES.md Step 5.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit` (correct trial) and `obj.bp.miss` (incorrect trial) boolean fields.

ii.
```python
hit = get_trial_bool(bp, 'hit')
miss = get_trial_bool(bp, 'miss')
outcome = 1 if bool(hit[tr]) else 0
```

iii. The trial filter ensures only hit or miss trials are included (`valid &= (hit | miss)`), so the outcome is always defined.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A per-trial scalar: `hit=True` maps to 1 (correct), otherwise 0 (incorrect). Broadcast across time bins.

ii.
```python
outcome = 1 if bool(hit[tr]) else 0
np.full(time_bins.shape, outcome, dtype=np.int64)
```

iii. The mapping matches the instruction: incorrect=0, correct=1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` — DLC trajectory data containing per-trial feature positions and frame times. The code searches for tongue-related features by trying names: `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`, `tongue`, `left_tongue`, `right_tongue`.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr,
    ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue',
     'tongue', 'left_tongue', 'right_tongue'], time_bins)
```

iii. The AI identified tongue features from the reference code's `params.traj_features` in `getDefaultParams.m` which lists tongue-related feature names for both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial: (1) find the first matching tongue feature in the trajectory data, (2) extract x,y pixel positions, (3) compute frame-to-frame speed as `sqrt(dx^2 + dy^2) / dt`, (4) convert frame times to go-cue-relative times (`frameTimes - goCue`), (5) interpolate speed onto the 75ms time bins using `np.interp`, (6) fill NaN gaps with nearest valid values.

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
```

iii. The AI computed velocity from pixel positions in the DLC trajectories. The reference code's `loadKinData.m` is a thin loader for precomputed kinematics files, so the AI had to derive velocity from raw trajectory data.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All tongue speed values across all valid trials in the session are concatenated. The 50th percentile is computed using `np.nanpercentile`. Values >= threshold map to 1 (high), below to 0 (low).

ii.
```python
tongue_all = np.concatenate([... for tr in trial_idx])
tongue_thr = np.nanpercentile(tongue_all, 50) if tongue_all.size and np.isfinite(tongue_all).any() else np.nan
# ...
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
output_trials[i][3] = tmp
```

iii. The per-session 50th percentile threshold matches the instruction specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times from the trajectory data are converted to go-cue-relative times by subtracting the go cue timestamp. Speed values are then interpolated onto the same `time_bins` array used for neural data.

ii.
```python
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The alignment uses the same go cue reference as neural data, ensuring temporal consistency.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj`, searching for features named: `paw`, `left_paw`, `right_paw`, `top_paw`, `bottom_paw`.

ii.
```python
paw_vals = extract_hdf5_traj_velocity(obj, tr,
    ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins)
```

iii. The reference code's `getDefaultParams.m` lists `top_paw` and `bottom_paw` as camera 1 features. The AI uses those plus additional candidate names.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical pipeline to tongue velocity: extract feature positions, compute speed from displacements, interpolate onto time bins, fill NaN gaps.

ii. Same `extract_hdf5_traj_velocity` function as for tongue, with different feature candidates.

iii. The AI reused the same velocity computation pipeline for both tongue and paw features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile threshold. Values >= threshold = 1 (high), below = 0 (low).

ii.
```python
paw_all = np.concatenate([... for tr in trial_idx])
paw_thr = np.nanpercentile(paw_all, 50)
tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
output_trials[i][4] = tmp
```

iii. Matches instruction specification for per-session 50th percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment approach as tongue velocity: frame times converted to go-cue-relative times, then interpolated onto the shared `time_bins`.

ii. Same code path through `extract_hdf5_traj_velocity`.

iii. Consistent with neural data alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from `motionEnergy_<SUBJECT>_<DATE>.mat` sidecar files. The top-level variable `me` contains per-trial motion energy time series in `me.data`.

ii.
```python
def load_motion_energy(path: Optional[Path], n_trials: int) -> Optional[List[np.ndarray]]:
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    me = x['me'].flat[0]
    data = np.asarray(unwrap_me_data(me)).reshape(-1)
    out = []
    for i in range(min(len(data), n_trials)):
        elem = data[i]
        trial = np.asarray(elem).reshape(-1).astype(float)
        out.append(trial)
```

iii. The AI matched the motion energy source files to their corresponding data structure files using filename pattern matching.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy arrays are interpolated onto the time bins using `np.interp` with linearly spaced source time points from T_START to T_END. The reference MATLAB code uses `interp1(frameTimes - 0.5 - alignTimes(trix), me.data{trix}, taxis)` with frame times at 400 Hz, followed by `fillmissing('nearest')`. The AI approximates this by using `np.linspace(T_START, T_END, raw.size)` as the source time axis, which assumes the motion energy data is already aligned to the trial time window.

ii.
```python
if raw.size > 1:
    x_old = np.linspace(T_START, T_END, raw.size)
    me_binned = np.interp(time_bins, x_old, raw)
else:
    me_binned = np.full(time_bins.shape, np.nan)
```

iii. The AI's motion energy alignment differs significantly from the reference code. The reference code uses actual frame times with a specific offset (`frameTimes - 0.5 - alignTimes`), while the AI assumes the data is uniformly distributed across the trial window. The AI also fills NaN values differently — the reference code fills with nearest valid values, while the AI fills with the median of all motion energy values before thresholding.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. All motion energy values (with NaN replaced by the session median) are concatenated, and the 50th percentile is computed. Values >= threshold = 1, below = 0.

ii.
```python
all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all))) for x in me_binned_all])
thr = np.nanpercentile(all_me, 50)
tmp[finite] = (meb[finite] >= thr).astype(np.int64)
output_trials[i][5] = tmp
```

iii. The reference code uses `me.moveThresh` (a per-session threshold stored in the data) for binarization (`me.move = me.data > me.moveThresh`), not the 50th percentile. The AI uses the 50th percentile as specified in the decoder task instructions, which differs from the reference code's approach.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes motion energy data spans the trial window (-1.5 to +1.5 s) uniformly and interpolates onto time bin centers. The reference code aligns using actual frame times minus 0.5 s minus the go cue time.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. This is an approximation. The reference code's `loadMotionEnergy.m` uses `frameTimes = (1:nFrames)/400` and then `interp1(frameTimes - 0.5 - alignTimes(trix), ...)`, which properly accounts for actual frame timing and the go cue alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Sessions with unsupported `clu` structures are skipped with a warning (JEB6). (2) Sessions with <2 valid trials are skipped. (3) Missing motion energy files result in NaN-filled arrays. (4) Missing trajectory features result in NaN-filled velocity arrays. (5) NaN values in tongue/paw velocity are filled with nearest valid values after interpolation. (6) NaN values in motion energy are replaced with the session median before thresholding. (7) Nested MATLAB structs in motion energy data are handled by the `unwrap_me_data` helper.

ii.
```python
# Skip session with unsupported structure
if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
    raise ValueError('unsupported clu structure')
# Skip sessions with too few trials
if len(neural) < 2:
    print(f'[warn] skipping {sess.data_file.name}: <2 valid trials', flush=True)
    continue
# Handle missing motion energy
if path is None or not path.exists():
    return None
# NaN filling in velocity
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The AI documented handling of edge cases in CONVERSION_NOTES.md, noting specific sessions that failed and the workarounds applied.

## 11-a. What are the most time-consuming steps of the code?

i. Based on the conversion log, the per-session `build_session` step takes 5-19 seconds per session, with the full conversion taking approximately 3-4 minutes for 21 sessions. The trajectory velocity extraction is the main bottleneck, as it must dereference HDF5 objects and parse feature names for each trial individually.

ii. From `conversion_full_out.txt`: build times range from 0.04s (skipped JEB6) to 19.14s (JEB14_2022-08-23).

iii. The AI logged timing for each session using the `timed` context manager.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial trajectory velocity extraction involves a Python loop over trials with per-trial HDF5 object dereferencing, string parsing, and interpolation. This cannot easily be vectorized due to HDF5 reference indirection, but could be parallelized. The per-cluster neural extraction loop could be partially vectorized using array operations.

ii.
```python
# Per-cluster loop (could batch)
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The AI did not document specific vectorization opportunities in CONVERSION_NOTES.md.

## 11-c. What processing does the code repeat multiple times?

i. Tongue and paw velocity extraction is computed **three times** for each trial: (1) inside the main per-trial loop (lines 288-289, computed but stored only temporarily), (2) concatenated across all trials for threshold computation (lines 321-322), and (3) again in the final thresholding loop (lines 333-335). This triples the runtime for trajectory extraction.

ii.
```python
# First computation (lines 288-289, inside per-trial loop — results not used for thresholding)
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
# Second computation (line 321, for threshold)
tongue_all = np.concatenate([... extract_hdf5_traj_velocity(obj, tr, [...], time_bins) for tr in trial_idx])
# Third computation (line 334, for final assignment)
tong = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
```

iii. The AI did not document this redundancy. The first computation in the per-trial loop is completely wasted since its results are overwritten later.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The first tongue/paw velocity computation in the per-trial loop (lines 288-289) produces values that are never used — the output array rows are initialized to zeros and later overwritten. (2) The `get_trial_bool` function uses element-wise Python iteration with try/except for type conversion, which is unnecessarily slow for numeric arrays. (3) The `nan_to_num` call in the threshold concatenation (line 321-322) with `nan=np.nan` is a no-op that doesn't actually remove NaN values.

ii.
```python
# Wasted computation — tongue_vals and paw_vals computed but not stored
tongue_vals = extract_hdf5_traj_velocity(...)  # line 288
paw_vals = extract_hdf5_traj_velocity(...)     # line 289
# Output rows initialized to zeros, overwritten later
np.zeros(time_bins.shape, dtype=np.int64),  # tongue placeholder
np.zeros(time_bins.shape, dtype=np.int64),  # paw placeholder
```

iii. This represents significant wasted computation, especially since `extract_hdf5_traj_velocity` involves HDF5 I/O and string parsing.
