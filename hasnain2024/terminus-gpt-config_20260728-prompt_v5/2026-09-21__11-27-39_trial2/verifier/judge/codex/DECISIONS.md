# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing every `data_structure_*.mat` file under `/app/data/Ephys_Behavior` and `/app/data/RandomizedDelay_Ephys_Behavior`, then pairs each with a same-folder `motionEnergy_*.mat` file if present. It does not use a fixed author-derived session list; instead it loops over all discovered files and skips sessions that raise exceptions during conversion.

ii. ```python
EPHYS_DIRS = ['Ephys_Behavior', 'RandomizedDelay_Ephys_Behavior']


def discover_sessions() -> List[SessionInfo]:
    sessions = []
    for task_dir in EPHYS_DIRS:
        d = DATA_ROOT / task_dir
        for data_file in sorted(d.glob('data_structure_*.mat')):
            ...
            motion_file = d / f'motionEnergy_{animal}_{date}.mat'
            if not motion_file.exists():
                motion_file = None
            sessions.append(SessionInfo(animal, date, task_dir, data_file, motion_file))
    return sessions
```
```python
for sess in sessions:
    ...
    try:
        neural_trials, input_trials, output_trials, subject, bri = build_session(sess)
    except Exception as e:
        print(f'SKIP {sess.data_file.name}: {e!r}')
        skipped.append((sess.data_file.name, repr(e)))
        continue
```

iii. The justification in the trajectory was pragmatic rather than reference-matching: the agent repeatedly described this as a generic ephys-session discovery strategy, then later documented that the resulting dataset was incomplete because skipped sessions and mixed MAT-file formats were not fully handled.

## 1-b. How are the data split into subjects?

i. Subjects are defined from filenames: the substring before the first underscore in `data_structure_<animal>_<date>.mat` becomes the animal/subject ID. During assembly, subjects are inserted in first-seen order and `subject_idx` maps each retained session to that subject.

ii. ```python
m = re.match(r'data_structure_([^_]+)_(.+)\.mat', data_file.name)
...
animal, date = m.groups()
...
sessions.append(SessionInfo(animal, date, task_dir, data_file, motion_file))
```
```python
subject_to_idx = {}
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data['subjects'])
    data['subjects'].append(subject)
data['subject_idx'].append(subject_to_idx[subject])
```

iii. The notes justify this by saying the dataset is organized by session filenames and that subject/session counts were extracted from filenames during dataset exploration.

## 1-c. How are the data split into sessions?

i. Each discovered `data_structure_*.mat` file is treated as one session. The code does not distinguish reference-approved versus excluded sessions beyond whether the file can be processed successfully.

ii. ```python
for data_file in sorted(d.glob('data_structure_*.mat')):
    ...
    sessions.append(SessionInfo(animal, date, task_dir, data_file, motion_file))
```
```python
data['neural'].append(neural_trials)
data['input'].append(input_trials)
data['output'].append(output_trials)
```

iii. In the trajectory the agent treated “one file = one session” as the simplest consistent rule, then later acknowledged this diverged from the reference because the reference analyses use an inclusion-listed subset.

## 1-d. How are the data split into trials?

i. Trials are defined by `bp['Ntrials']`. Neural trials are built for trial numbers `1..Ntrials`, and per-trial inputs/outputs are produced by looping over that same range.

ii. ```python
n_trials = int(np.array(bp['Ntrials']).squeeze())
...
for tr in range(1, n_trials + 1):
    mask = trials == tr
    if np.any(mask):
        mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```
```python
for tr in range(n_trials):
    input_trials.append(time[None, :].astype(np.float32))
    ...
    output_trials.append(out)
```

iii. The AI did not document a separate trial-boundary reconstruction; the implicit justification is that `Ntrials` and the per-spike `trial` field already define the trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The final code does not filter trials for early licks, photostimulation, or post-recording segments. All trials up to `bp['Ntrials']` are retained unless an entire session is skipped.

ii. ```python
n_trials = int(np.array(bp['Ntrials']).squeeze())
...
for tr in range(n_trials):
    input_trials.append(time[None, :].astype(np.float32))
    ...
    output_trials.append(out)
```

iii. The trajectory shows the agent knew from the notes that early-lick and photostim filtering existed in the reference, but the final script never implemented those filters. Its later notes instead focus on skipped sessions and unresolved context extraction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are built from `obj.clu` fields, especially each unit’s `trial`, `trialtm`, and `tm`. `trial` and `trialtm` define spike assignment to trials and time bins; `tm` is used only to estimate a mean firing rate for unit filtering.

ii. ```python
for key in ['tm', 'trial', 'trialtm', 'quality', 'site']:
    if key not in fields:
        continue
    ds = fields[key]
    ref = ds[i, 0]
    target = h[ref]
    unit[key] = np.array(target[()]).squeeze()
```
```python
trials = np.array(u['trial']).astype(int).ravel()
trialtm = np.array(u['trialtm']).astype(float).ravel()
```

iii. The notes identify `obj.clu` as the source of neural recordings and the trajectory discusses parsing `obj.clu` from HDF5 files as the core neural-loading path.

## 2-b. How is the `neural` data processed?

i. The code bins raw spike times into a common 25 ms grid over `[-2.5, 2.5]` and stores the resulting counts directly as `float32`. It does not convert counts to Hz, smooth them, subtract go-cue times, or otherwise normalize them.

ii. ```python
BIN_SIZE_S = 0.025
T_START = -2.5
T_END = 2.5
```
```python
def build_neural_trials(units: List[Dict[str, Any]], n_trials: int) -> List[np.ndarray]:
    time = common_time_axis()
    edges = np.concatenate([[time[0] - BIN_SIZE_S / 2], time + BIN_SIZE_S / 2])
    ...
    for ui, u in enumerate(units):
        ...
        for tr in range(1, n_trials + 1):
            mask = trials == tr
            if np.any(mask):
                mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
    return mats
```

iii. The code comment at the top says this was an “initial implementation” using a “conservative common window/binning,” with the implication that it might later be updated after inspection. The final script never replaced this provisional choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered only by a coarse firing-rate threshold: if the number of spikes in `tm` divided by `max(n_trials, 1)` exceeds 1.0, the unit is kept. Cluster quality labels are loaded but never used.

ii. ```python
def filter_units(units: List[Dict[str, Any]], n_trials: int) -> List[Dict[str, Any]]:
    kept = []
    session_dur = max(n_trials * 1.0, 1.0)
    for u in units:
        tm = np.array(u['tm']).astype(float).ravel()
        mean_fr = tm.size / session_dur
        if mean_fr > 1.0:
            kept.append(u)
    return kept
```

iii. The notes mention the reference’s `>1 Hz` rule, and the agent appears to have implemented only that piece. The trajectory later recognizes that quality-label filtering from the reference was still missing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Despite the metadata claiming go-cue alignment, the actual neural binning uses `trialtm` directly with no subtraction of the trial’s go-cue time. In effect, the neural data are aligned to whatever `trialtm` is measured from, not explicitly to go cue.

ii. ```python
trials = np.array(u['trial']).astype(int).ravel()
trialtm = np.array(u['trialtm']).astype(float).ravel()
...
mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```
```python
'metadata': {
    ...
    'temporal_alignment_event': 'go cue onset',
    ...
}
```

iii. The notes and trajectory repeatedly state that go-cue alignment was intended, but the final implementation does not carry that subtraction into the neural path.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 25 ms bins from `-2.5` to `2.5` s. There is no further rebinning after that initial histogramming/interpolation step.

ii. ```python
BIN_SIZE_S = 0.025
...
def common_time_axis() -> np.ndarray:
    edges = np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
    centers = edges[:-1] + BIN_SIZE_S / 2
    return centers.astype(np.float32)
```

iii. The only explicit justification is the top-level comment calling this a conservative common binning choice pending validation.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input time axis is not computed from a raw variable per trial. It is a synthetic common axis defined from the constants `T_START`, `T_END`, and `BIN_SIZE_S`.

ii. ```python
BIN_SIZE_S = 0.025
T_START = -2.5
T_END = 2.5
...
def common_time_axis() -> np.ndarray:
    edges = np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
    centers = edges[:-1] + BIN_SIZE_S / 2
    return centers.astype(np.float32)
```

iii. The agent’s Step 5 notes explicitly planned to “tile common time vector across trials as continuous time-from-go-cue,” so the justification is that this input is an analysis-defined axis rather than a stored raw measurement.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes the bin centers of the common axis and copies that same 1D time vector into every trial as a `(1, n_time)` array.

ii. ```python
def common_time_axis() -> np.ndarray:
    edges = np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
    centers = edges[:-1] + BIN_SIZE_S / 2
    return centers.astype(np.float32)
```
```python
for tr in range(n_trials):
    input_trials.append(time[None, :].astype(np.float32))
```

iii. The notes say this representation was chosen because the decoder task specifically requested time from go cue onset as a continuous, time-varying input.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector uses the same `common_time_axis()` grid as the neural data, so it is aligned to the neural bins by construction. However, because the neural path does not explicitly subtract go-cue times, this shared grid is only nominally go-cue aligned.

ii. ```python
time = common_time_axis()
```
```python
neural_trials = build_neural_trials(units, n_trials)
...
for tr in range(n_trials):
    input_trials.append(time[None, :].astype(np.float32))
```

iii. The Step 5 notes describe the input as the common aligned time axis for the decoder; the trajectory never identifies the missing go-cue subtraction in the neural branch.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The code derives lick direction from the per-trial `R` and `L` arrays in `bp`, combined with the derived outcome labels to mark ignore trials as “none.” It does not use `hit`/`miss` to flip the instructed side on incorrect trials.

ii. ```python
def infer_lick_direction(bp: Dict[str, Any], outcomes: np.ndarray, n_trials: int) -> np.ndarray:
    R = np.ravel(bp['R'])[:n_trials] > 0
    L = np.ravel(bp['L'])[:n_trials] > 0
    lick = np.full(n_trials, 2, dtype=np.int64)
    lick[L] = 0
    lick[R] = 1
    lick[outcomes == 2] = 2
    return lick
```

iii. The mapping plan in the notes mentions `obj.bp.R`, `obj.bp.L`, and lick-event logic; the final implementation simplifies that to direct use of `R`/`L` plus the ignore outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Trials flagged `L` become class 0, trials flagged `R` become class 1, and trials whose derived outcome is ignore become class 2. Miss trials are therefore left as the instructed side rather than being inverted to the actual lick side.

ii. ```python
lick = np.full(n_trials, 2, dtype=np.int64)
lick[L] = 0
lick[R] = 1
lick[outcomes == 2] = 2
```

iii. No stronger justification appears in the final notes; this was a simplified mapping chosen during implementation.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The code looks only at `bp['protocol']['nums']` when available. If there are exactly two unique finite values, it treats the larger one as one context and the smaller as the other; otherwise all trials default to class 0.

ii. ```python
def infer_context(sess: SessionInfo, bp: Dict[str, Any], n_trials: int) -> np.ndarray:
    ctx = np.zeros(n_trials, dtype=np.int64)
    protocol = bp.get('protocol', {})
    nums = protocol.get('nums', None) if isinstance(protocol, dict) else None
    if nums is not None:
        vals = np.ravel(nums)[:n_trials]
        uniq = [u for u in np.unique(vals) if np.isfinite(u)]
        if len(uniq) == 2:
            ctx = (vals == uniq.max()).astype(np.int64)
    return ctx
```

iii. The in-code comment says this is a “conservative initial mapping,” and the trajectory later explicitly records that behavioral context remained unresolved and constant DR in the produced dataset.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The default code path sets every trial to class 0. Only if `protocol.nums` has exactly two unique values does it assign class 1 to the larger value. The output metadata then names the classes `['DR', 'WC']`.

ii. ```python
ctx = np.zeros(n_trials, dtype=np.int64)
...
if len(uniq) == 2:
    ctx = (vals == uniq.max()).astype(np.int64)
```
```python
'output_values': [
    ['left', 'right', 'none'],
    ['DR', 'WC'],
    ...
],
```

iii. The trajectory repeatedly states this was an unresolved placeholder and that proper WC/DR extraction still needed human-quality follow-up.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial `hit`, `miss`, and `no` flags in `bp`.

ii. ```python
def infer_outcomes(bp: Dict[str, Any], n_trials: int) -> np.ndarray:
    hit = np.ravel(bp['hit'])[:n_trials] > 0
    miss = np.ravel(bp['miss'])[:n_trials] > 0
    no = np.ravel(bp['no'])[:n_trials] > 0
    ...
```

iii. The notes planned exactly this mapping (`hit`/`miss`/`no` to correct/incorrect/ignore), and the implementation follows that plan.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code initializes every trial as ignore (2), then relabels miss trials as incorrect (0) and hit trials as correct (1).

ii. ```python
out = np.full(n_trials, 2, dtype=np.int64)
out[miss] = 0
out[hit] = 1
out[no] = 2
```

iii. The notes justify this as the natural trial-level categorical outcome mapping required by the decoder task.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the first feature in the first trajectory object returned by `read_traj_trial_refs`, using its `ts` coordinates, `frameTimes`, and the trial `goCue` times. The code does not inspect `featNames` and does not combine tongue information from both camera views.

ii. ```python
def read_traj_trial_refs(h: h5py.File):
    traj = h['/obj/traj']
    arr = np.array(traj[()])
    if arr.dtype != object or arr.size == 0:
        return None
    return h[arr.flatten(order='F')[0]]
```
```python
ts = np.array(h[ts_refs[tr]][()])
ft = np.array(h[ft_refs[tr]][()]).astype(float).ravel()
...
x = ts[0, 0, :].astype(float)
y = ts[0, 1, :].astype(float)
p = ts[0, 2, :].astype(float)
```

iii. The Step 5 notes had planned to use trajectory features for tongue kinematics, but the trajectory later shows the implementation never reached a more specific feature-selection stage.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code keeps frames where `x`, `y`, and `frameTimes` are finite and confidence `p > 0.5`, computes frame-to-frame speed by finite differencing, places the speed at frame midpoints, interpolates that onto the common time axis, and leaves leading/trailing timepoints outside the observed interval as NaN. It applies no Gaussian smoothing and no multi-camera normalization.

ii. ```python
valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ft) & (p > 0.5)
...
speed = np.sqrt(np.diff(xv)**2 + np.diff(yv)**2) / dt
tmid = (tv[:-1] + tv[1:]) / 2
...
yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
idx = np.where(np.isfinite(yint))[0]
...
yint[:first] = np.nan
yint[last+1:] = np.nan
```

iii. The trajectory frames this as an initial practical alignment of tongue motion sufficient to produce a decoder-ready categorical signal, not as a faithful reproduction of the reference video-processing pipeline.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After alignment, tongue velocity is discretized sessionwise at the median of all finite values: values below the median become 0, values at or above the median become 1, and NaNs become 2.

ii. ```python
def discretize_session_median(arr2d: Optional[np.ndarray], missing_code: int) -> np.ndarray:
    ...
    thr = np.nanmedian(arr2d[valid])
    out = np.full(arr2d.shape, missing_code, dtype=np.int64)
    out[valid & (arr2d < thr)] = 0
    out[valid & (arr2d >= thr)] = 1
    return out
```
```python
tongue_disc = discretize_session_median(tongue_aligned, missing_code=2) if tongue_aligned is not None else None
```

iii. This directly follows the decoder task’s requested 50th-percentile split with a missing-data category.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. For each trial the code subtracts that trial’s `goCue` time from raw `frameTimes`, computes velocities in that frame-time coordinate, and interpolates onto the same common grid used by the neural data. It does not estimate or subtract a video-to-behavior clock offset.

ii. ```python
align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
...
tv = ft[valid] - align_times[tr]
...
yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
```

iii. The notes say the decoder should be go-cue aligned; the trajectory later identifies missing video-offset correction as part of the broader incompleteness of the final conversion.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. In the final output, paw velocity is not derived from any raw variable. Every paw-velocity timeseries is filled with the missing-data code 2.

ii. ```python
for tr in range(n_trials):
    ...
    paw = np.full(time.shape, 2, dtype=np.int64)
    ...
```

iii. The trajectory explicitly says paw velocity remained unavailable in the inspected ephys sessions, and the notes record that as an unresolved blocker.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. No paw-specific processing is performed. The code writes a constant vector of 2s for every trial and timepoint.

ii. ```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. The justification from the trajectory is that paw extraction was not successfully recovered, so the script preserved that missingness rather than inventing a signal.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It is not thresholded from a continuous estimate at all. Every sample is directly assigned category 2 (`not_visible`).

ii. ```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. The trajectory treats this as a placeholder caused by incomplete feature extraction.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. No alignment computation is carried out because no paw signal is constructed; the output is a constant missing-data vector on the common trial grid.

ii. ```python
paw = np.full(time.shape, 2, dtype=np.int64)
...
out = np.vstack([
    ...,
    paw[None, :],
    ...,
])
```

iii. The rationale is the same as above: paw processing was never completed in the final script.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the session’s standalone `motionEnergy_*.mat` file, specifically from its `me.data` content after extracting per-trial numeric arrays.

ii. ```python
def load_motion_energy(path: Optional[Path]):
    if path is None:
        return None
    m = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return m['me'].flat[0]
```
```python
data = getattr(me, 'data', None)
...
x = extract_numeric_motion_trial(data_arr.flat[tr])
```

iii. The notes identify the separate motion-energy files and the trajectory discusses supporting their mixed MAT layouts as part of the loading strategy.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code assumes motion-energy traces are sampled at 400 Hz, creates synthetic frame times `1..N / 400`, subtracts `0.5` s and the trial’s `goCue`, interpolates onto the common time axis, and then nearest-fills edge NaNs before discretization.

ii. ```python
frame_times = np.arange(1, x.size + 1, dtype=float) / 400.0
old_t = frame_times - 0.5 - align_times[tr]
...
y = np.interp(time, old_t[valid], x[valid], left=np.nan, right=np.nan)
...
y[:first] = y[first]
y[last+1:] = y[last]
```

iii. The Step 3 notes mention 400 Hz motion energy from the reference code; the trajectory frames the rest of this as a pragmatic approximation that was good enough to produce a structurally valid output file.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized with the same session-median helper used for tongue velocity: below median is 0, at/above median is 1, and missing is 2.

ii. ```python
motion_disc = discretize_session_median(motion_aligned, missing_code=2) if motion_aligned is not None else None
```
```python
def discretize_session_median(arr2d: Optional[np.ndarray], missing_code: int) -> np.ndarray:
    ...
    thr = np.nanmedian(arr2d[valid])
    ...
```

iii. This matches the prompt’s per-session 50th-percentile discretization requirement.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by interpolating the synthetic 400 Hz timebase, shifted by `0.5` s and the trial’s `goCue`, onto the same `common_time_axis()` used elsewhere. No video-offset correction or direct use of camera frame times is performed.

ii. ```python
align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
...
old_t = frame_times - 0.5 - align_times[tr]
...
y = np.interp(time, old_t[valid], x[valid], left=np.nan, right=np.nan)
```

iii. The trajectory later identifies skipped sessions and unresolved alignment fidelity as remaining blockers, so this should be understood as an approximate rather than reference-faithful alignment choice.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are handled in several ad hoc ways: sessions that raise exceptions are skipped entirely; missing motion-energy files return `None`; missing aligned tongue/motion bins become category 2 after discretization; leading/trailing NaNs in motion energy are edge-filled; and paw velocity is represented as entirely missing. The code does not attempt reference-style trial-level QC repair.

ii. ```python
if path is None:
    return None
```
```python
try:
    neural_trials, input_trials, output_trials, subject, bri = build_session(sess)
except Exception as e:
    print(f'SKIP {sess.data_file.name}: {e!r}')
    skipped.append((sess.data_file.name, repr(e)))
    continue
```
```python
y[:first] = y[first]
y[last+1:] = y[last]
```

iii. The trajectory explicitly documents session skipping, constant context, and unavailable paw velocity as unresolved blockers; the implemented policy was to keep the pipeline running and emit structurally valid outputs rather than stop on those mismatches.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive steps in this implementation are likely repeated MAT-file loading plus the nested per-unit/per-trial spike-binning loop in `build_neural_trials`, followed by per-trial interpolation loops for tongue and motion energy.

ii. ```python
with h5py.File(sess.data_file, 'r') as h:
    ...
```
```python
for ui, u in enumerate(units):
    ...
    for tr in range(1, n_trials + 1):
        mask = trials == tr
        if np.any(mask):
            mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```

iii. The AI did not document a formal runtime analysis in the notes. The only explicit justification is the general Step 6 plan to get something running first and improve efficiency later if needed.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization opportunity is the nested unit-by-trial loop in `build_neural_trials`. The per-trial loops in `align_tongue_speed`, `align_motion_energy`, and output assembly are also candidates for partial batching, though they are more irregular.

ii. ```python
for ui, u in enumerate(units):
    ...
    for tr in range(1, n_trials + 1):
        mask = trials == tr
        if np.any(mask):
            mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```
```python
for tr in range(n):
    ...
```

iii. There is no explicit written justification from the AI for leaving these loops unvectorized. The trajectory instead reflects an unfinished efficiency pass.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly rebuilds or recopies session-invariant structures: `common_time_axis()` is called in both `build_neural_trials` and `build_session`, and the same `time[None, :].astype(np.float32)` array is re-created for every trial. It also repeatedly allocates all-missing paw vectors per trial.

ii. ```python
def build_neural_trials(...):
    time = common_time_axis()
```
```python
time = common_time_axis()
...
for tr in range(n_trials):
    input_trials.append(time[None, :].astype(np.float32))
    paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. The AI did not document this as an intentional optimization tradeoff; it appears to be an unoptimized implementation detail.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of loaded or computed information are unused downstream: cluster `quality` and `site` are read but not used; probe locations are parsed but then reduced to a fixed all-zero `brain_region_idx`; the script defines unused SciPy fallback helpers; and the `--show-processing` flag is parsed but never used.

ii. ```python
for key in ['tm', 'trial', 'trialtm', 'quality', 'site']:
    ...
```
```python
probe_locs = read_probe_locations(h)
if not probe_locs:
    probe_locs = ['ALM']
brain_region_idx = np.zeros(len(units), dtype=np.int64)
```
```python
p.add_argument('--show-processing', action='store_true')
```

iii. The trajectory shows these were leftovers from an incomplete broader implementation; no explicit downstream justification is given for keeping them.
