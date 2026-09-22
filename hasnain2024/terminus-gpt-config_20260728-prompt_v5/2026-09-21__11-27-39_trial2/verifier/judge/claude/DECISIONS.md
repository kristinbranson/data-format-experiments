# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing `data_structure_*.mat` files in two directories (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). It parses the animal and date from filenames using a regex. All files are opened with `h5py` (HDF5/v7.3 format). Sessions that fail to open (non-HDF5 format) or lack a `clu` field are skipped. The AI also checks for companion `motionEnergy_*.mat` files in the same directory. The code does NOT use a hard-coded session list; instead it discovers all `.mat` files matching the pattern.

ii. Session discovery:
```python
def discover_sessions() -> List[SessionInfo]:
    sessions = []
    for task_dir in EPHYS_DIRS:
        d = DATA_ROOT / task_dir
        for data_file in sorted(d.glob('data_structure_*.mat')):
            m = re.match(r'data_structure_([^_]+)_(.+)\.mat', data_file.name)
            if not m:
                continue
            animal, date = m.groups()
            motion_file = d / f'motionEnergy_{animal}_{date}.mat'
            if not motion_file.exists():
                motion_file = None
            sessions.append(SessionInfo(animal, date, task_dir, data_file, motion_file))
    return sessions
```

Loading is done with h5py only:
```python
def build_session(sess: SessionInfo):
    with h5py.File(sess.data_file, 'r') as h:
        bp = read_bp(h)
        ...
```

iii. The AI noted in CONVERSION_NOTES.md that files are MATLAB v7.3/HDF5 and should be read with h5py. When non-HDF5 files failed to open, the AI logged them as skipped rather than implementing a scipy fallback. The CONVERSION_NOTES acknowledge this as a known incompleteness: "13 sessions were skipped due to mixed MAT-file formats."

## 1-b. How are the data split into subjects?

i. The animal name is parsed from the filename regex `data_structure_([^_]+)_(.+)\.mat`, taking the first capture group as the subject identifier. A `subject_to_idx` dictionary maps unique subject names to indices.

ii.
```python
m = re.match(r'data_structure_([^_]+)_(.+)\.mat', data_file.name)
animal, date = m.groups()
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data['subjects'])
    data['subjects'].append(subject)
data['subject_idx'].append(subject_to_idx[subject])
```

iii. The AI noted subjects are identified by filename parsing, which is consistent with the reference approach. However, subjects are not sorted; they appear in file-discovery order.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file is one session. The AI discovers all such files across the two ephys directories. Sessions that cannot be opened or processed are skipped. The final dataset has 34 sessions (out of 47 discovered files), because 13 were skipped due to format errors or missing `clu` data.

ii.
```python
for sess in sessions:
    try:
        neural_trials, input_trials, output_trials, subject, bri = build_session(sess)
    except Exception as e:
        print(f'SKIP {sess.data_file.name}: {e!r}')
        skipped.append((sess.data_file.name, repr(e)))
        continue
```

iii. The AI acknowledged in CONVERSION_NOTES that sessions were skipped due to format issues and that this was incomplete relative to reference expectations of 44 sessions.

## 1-d. How are the data split into trials?

i. The number of trials per session is read from `obj.bp.Ntrials`. All trials from 1 to `Ntrials` are included without any filtering. Neural trial matrices are built by looping over trial indices 1 through `n_trials`.

ii.
```python
n_trials = int(np.array(bp['Ntrials']).squeeze())
...
for tr in range(1, n_trials + 1):
    mask = trials == tr
    if np.any(mask):
        mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```

iii. The AI does not justify the lack of trial filtering in the code or notes.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. The AI does not exclude early-lick trials, photostimulation trials, or trials that extend past the end of the recording. All `Ntrials` trials are included in the output.

ii. There is no filtering code. The `build_session` function uses all `n_trials` directly:
```python
n_trials = int(np.array(bp['Ntrials']).squeeze())
units = read_clu_units(h)
units = filter_units(units, n_trials)
neural_trials = build_neural_trials(units, n_trials)
```

iii. The CONVERSION_NOTES mention trial curation rules from the paper (subsampling balanced trial sets) but these are described in the context of specific analyses, not as general filtering. The AI did not implement the paper's standard exclusion of early-lick and photostimulation trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu`, specifically the `trial` and `trialtm` fields of each cluster unit. Spike times (`trialtm`) are binned per trial to produce spike count matrices.

ii.
```python
def read_clu_units(h: h5py.File) -> List[Dict[str, Any]]:
    clu = h['/obj/clu']
    probe_groups = deref_cell(h, clu)
    ...
    for key in ['tm', 'trial', 'trialtm', 'quality', 'site']:
        ...
        unit[key] = np.array(target[()]).squeeze()
    units.append(unit)
```

```python
trials = np.array(u['trial']).astype(int).ravel()
trialtm = np.array(u['trialtm']).astype(float).ravel()
```

iii. The AI identified `obj.clu` as the source of neural data, which is consistent with the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are histogrammed into 25 ms bins (not 5 ms as in the reference). The spike counts are stored as raw counts (float32) without conversion to firing rates (Hz) and without any temporal smoothing.

ii.
```python
BIN_SIZE_S = 0.025
...
def build_neural_trials(units, n_trials):
    time = common_time_axis()
    edges = np.concatenate([[time[0] - BIN_SIZE_S / 2], time + BIN_SIZE_S / 2])
    ...
    mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```

iii. The AI initially noted it would use a "conservative common window/binning" but did not update this to match the reference parameters (5 ms bins, conversion to Hz, Gaussian smoothing). The CONVERSION_NOTES reference `params.dt = 1/200` from the code but the bin size was never corrected.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters units based on a simple firing rate criterion: `tm.size / session_dur > 1.0`. This uses the total number of spike times (`tm`) divided by the number of trials as a proxy for session duration. The AI does NOT filter based on cluster quality labels (e.g., `garbage`, `noisy`, `poor`).

ii.
```python
def filter_units(units, n_trials):
    kept = []
    session_dur = max(n_trials * 1.0, 1.0)
    for u in units:
        tm = np.array(u['tm']).astype(float).ravel()
        mean_fr = tm.size / session_dur
        if mean_fr > 1.0:
            kept.append(u)
    return kept
```

iii. The CONVERSION_NOTES mention the paper's >1 Hz firing rate criterion and quality-based filtering, but the implementation only uses a crude spike-count-based proxy without quality label filtering. The firing rate calculation is also incorrect: it divides the total number of spike times by the number of trials (treating each trial as 1 second), rather than computing the mean firing rate over the analysis window.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is NOT aligned to the go cue. Spike times (`trialtm`) are used directly without subtracting the go cue time. The time axis spans -2.5 to 2.5 s but this is relative to trial start, not to go cue onset.

ii.
```python
trialtm = np.array(u['trialtm']).astype(float).ravel()
...
mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```

The go cue time is read for tongue alignment but not used for neural alignment:
```python
align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
```

iii. The AI noted in its mapping plan that "Tile common time vector across trials as continuous time-from-go-cue" and "Align all streams to goCue," but the neural data implementation does not subtract the go cue time from `trialtm` before binning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 25 ms (0.025 s), producing 200 time bins over the -2.5 to 2.5 s window. The reference uses 5 ms bins (1000 time bins). No rebinning is applied.

ii.
```python
BIN_SIZE_S = 0.025
T_START = -2.5
T_END = 2.5
```

iii. The AI set this as an "initial implementation" with a "conservative common window/binning" and noted it would be updated after inspection. It was never updated to match the reference's 5 ms bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the common time axis, which is computed from the bin edges. It is not derived from any raw data variable.

ii.
```python
def common_time_axis() -> np.ndarray:
    edges = np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
    centers = edges[:-1] + BIN_SIZE_S / 2
    return centers.astype(np.float32)
```

iii. This is conceptually correct - the time axis is defined by the analysis window, not by raw data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers from -2.5 to 2.5 s in 25 ms steps. No further processing is applied.

ii.
```python
input_trials.append(time[None, :].astype(np.float32))
```

iii. The processing is straightforward but uses the wrong bin size (25 ms instead of 5 ms).

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Both use the same time axis from `common_time_axis()`. However, since the neural data is not actually aligned to the go cue (see 2-d), the input time axis claims to represent time from go cue but the neural data is actually aligned to trial start.

ii.
```python
time = common_time_axis()
...
neural_trials = build_neural_trials(units, n_trials)
...
input_trials.append(time[None, :].astype(np.float32))
```

iii. There is a misalignment: the input says "time from go cue" but the neural data is binned relative to trial start.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R`, `obj.bp.L`, and the outcome array. The instructed side (`R`/`L`) is used directly as the lick direction, with ignore trials getting the "none" label.

ii.
```python
def infer_lick_direction(bp, outcomes, n_trials):
    R = np.ravel(bp['R'])[:n_trials] > 0
    L = np.ravel(bp['L'])[:n_trials] > 0
    lick = np.full(n_trials, 2, dtype=np.int64)
    lick[L] = 0
    lick[R] = 1
    lick[outcomes == 2] = 2
    return lick
```

iii. The AI's notes indicate it maps `R`/`L` to right/left directly. However, this is the INSTRUCTED side, not the actual lick direction. On miss trials, the animal licks the opposite side from what was instructed, so using the instructed side directly gives the wrong lick direction for miss trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The instructed side (`R`=right, `L`=left) is mapped directly to lick direction codes (left=0, right=1). Ignore trials (outcome==2) are set to "none" (2). This does NOT account for miss trials where the animal licks the opposite of the instructed side.

ii. Same as 4-a above.

iii. The reference correctly infers lick direction by combining instructed side with outcome (hit = licked instructed side, miss = licked opposite side). The AI's approach is incorrect for miss trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI attempts to derive context from `obj.bp.protocol.nums`, not from `obj.bp.autowater` as in the reference.

ii.
```python
def infer_context(sess, bp, n_trials):
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

iii. The AI noted in CONVERSION_NOTES that it was using protocol numbers to identify contexts and that "protocol numbers in sampled ephys sessions do not show multi-context variation, so constant DR context in the sample may reflect the chosen sessions rather than a loading bug." In fact, the result is ALL sessions have context = DR (0), which is clearly wrong since the paper describes both WC and DR contexts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI checks if `protocol.nums` has exactly 2 unique values. If so, it maps the higher value to 1 (DR) and the lower to 0 (WC). If not, all trials default to 0 (DR). In practice, this results in all trials across all 34 sessions being labeled DR, which is incorrect.

ii. Same as 5-a.

iii. The verification output confirms: `behavioral_context: {DR (1.000)}` - all trials are DR. The reference uses `obj.bp.autowater` to correctly identify WC trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, `obj.bp.miss`, and `obj.bp.no`.

ii.
```python
def infer_outcomes(bp, n_trials):
    hit = np.ravel(bp['hit'])[:n_trials] > 0
    miss = np.ravel(bp['miss'])[:n_trials] > 0
    no = np.ravel(bp['no'])[:n_trials] > 0
    out = np.full(n_trials, 2, dtype=np.int64)
    out[miss] = 0
    out[hit] = 1
    out[no] = 2
    return out
```

iii. This is consistent with the reference, which also uses `hit` and `miss` flags. The AI also reads `no` but since the default is already 2 (ignore), this is redundant but not harmful.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss trials are coded as 0 (incorrect), hit trials as 1 (correct), and everything else as 2 (ignore). This matches the reference encoding.

ii. Same as 6-a.

iii. The mapping is correct: incorrect=0, correct=1, ignore=2.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The tongue velocity is derived from `obj.traj`, specifically the first camera's first tracked feature (index 0). The AI reads `ts` (tracking data) and `frameTimes` from the first element of the trajectory cell array.

ii.
```python
def align_tongue_speed(h, bp, n_trials, time):
    tg = read_traj_trial_refs(h)
    ...
    ts_refs = np.array(tg['ts'][()]).flatten(order='F')
    ft_refs = np.array(tg['frameTimes'][()]).flatten(order='F')
    ...
    x = ts[0, 0, :].astype(float)
    y = ts[0, 1, :].astype(float)
    p = ts[0, 2, :].astype(float)
```

iii. The AI uses only the first camera and the first tracked feature (index 0), without checking `featNames` to identify which feature is the tongue. The reference uses `tongue` from the side camera and `top_tongue` from the bottom camera, combining both views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes speed as `sqrt(dx^2 + dy^2) / dt` using finite differences, filters by likelihood > 0.5 (reference uses > 0.9), and interpolates onto the common time axis using `np.interp`. No Gaussian smoothing is applied to x/y coordinates before differentiation. No normalization is applied. No combining of two camera views.

ii.
```python
valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ft) & (p > 0.5)
...
speed = np.sqrt(np.diff(xv)**2 + np.diff(yv)**2) / dt
tmid = (tv[:-1] + tv[1:]) / 2
...
yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
```

iii. Key differences from reference: (1) likelihood threshold 0.5 vs 0.9, (2) no Gaussian smoothing of positions, (3) single camera instead of two, (4) np.interp instead of bin averaging, (5) no per-view normalization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses the session median of all valid (finite) values as the threshold. Values below the median are 0, at or above are 1, and NaN values (not tracked) are 2.

ii.
```python
def discretize_session_median(arr2d, missing_code):
    ...
    thr = np.nanmedian(arr2d[valid])
    out = np.full(arr2d.shape, missing_code, dtype=np.int64)
    out[valid & (arr2d < thr)] = 0
    out[valid & (arr2d >= thr)] = 1
    return out
```

iii. The use of median is close to the reference's 50th percentile but is computed using `np.nanmedian` instead of `np.nanpercentile(..., 50)`. These should give similar results. The overall approach is consistent with the instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI aligns tongue data to the go cue by subtracting `bp.ev.goCue` from frame times. However, no video offset correction is applied (the reference applies a bitcode-derived offset to convert from video clock to behavior clock before subtracting go cue). The result is then interpolated onto the common time axis.

ii.
```python
tv = ft[valid] - align_times[tr]
...
yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
```

iii. The AI does not apply the video offset correction from `findVideoOffset.m`, which is needed because the camera clock leads the behavior clock. This means tongue velocity is systematically misaligned in time.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is NOT implemented. All paw velocity values are set to the "not visible" code (2) for all trials and time bins.

ii.
```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. The AI noted in CONVERSION_NOTES: "In inspected ephys sessions, trajectory feature names consistently lack paw landmarks, so paw output remains `not_visible` for these sessions." The reference uses the `top_paw` feature from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. No processing - all values are hardcoded as "not visible" (2).

ii. Same as 8-a.

iii. The AI was unable to identify paw landmarks in the trajectory data.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Not applicable - all values are "not visible" (2).

ii. N/A

iii. N/A

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Not applicable - no paw data is computed.

ii. N/A

iii. N/A

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from companion `motionEnergy_*.mat` files using `scipy.io.loadmat`. The `me.data` field contains per-trial motion energy traces.

ii.
```python
def load_motion_energy(path):
    if path is None:
        return None
    m = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return m['me'].flat[0]
```

iii. This is consistent with the reference, which also loads from `motionEnergy_*.mat` files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI fabricates frame times as `np.arange(1, x.size + 1) / 400.0` (assuming 400 Hz sampling rate), subtracts 0.5 seconds and the go cue time, then interpolates onto the common time axis using `np.interp`. Edge NaN values are filled with nearest valid values.

ii.
```python
frame_times = np.arange(1, x.size + 1, dtype=float) / 400.0
old_t = frame_times - 0.5 - align_times[tr]
valid = np.isfinite(old_t) & np.isfinite(x)
y = np.interp(time, old_t[valid], x[valid], left=np.nan, right=np.nan)
# nearest fill for edge NaNs
idx = np.where(np.isfinite(y))[0]
y[:idx[0]] = y[idx[0]]
y[last+1:] = y[last]
```

iii. The reference uses actual camera frame times from `traj[side].frameTimes` corrected by the video offset. The AI instead fabricates frame times at 400 Hz with arbitrary offsets (-0.5 s), which is not grounded in the actual camera timing. The 400 Hz comes from a comment in the reference code's `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_session_median` function is used: values below session median are 0, at or above are 1, NaN values get code 2 ("no_video").

ii.
```python
motion_disc = discretize_session_median(motion_aligned, missing_code=2)
```

iii. This is consistent with the instructions' 50th percentile threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI fabricates frame times at 400 Hz starting from 1/400 s, subtracts 0.5 s and the go cue time, then interpolates onto the common time axis. No video offset correction is applied.

ii.
```python
frame_times = np.arange(1, x.size + 1, dtype=float) / 400.0
old_t = frame_times - 0.5 - align_times[tr]
```

iii. The 400 Hz and -0.5 s offset are hard-coded approximations rather than derived from the actual camera timing metadata. The reference uses actual `frameTimes` from the side camera with bitcode-derived video offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) sessions that fail to load are skipped entirely; (2) motion energy files that don't exist result in "no_video" for all time bins; (3) tongue tracking with insufficient valid points results in NaN (then "not visible"); (4) paw is entirely "not visible" for all sessions. Edge NaN values in motion energy are filled with nearest valid values rather than left as NaN.

ii.
```python
except Exception as e:
    print(f'SKIP {sess.data_file.name}: {e!r}')
    skipped.append((sess.data_file.name, repr(e)))
    continue
```

```python
# nearest fill for edge NaNs
y[:first] = y[first]
y[last+1:] = y[last]
```

iii. The edge-fill approach for motion energy is questionable - the reference leaves NaN bins as "no video" rather than filling them. Skipping entire sessions due to format issues loses significant data.

## 11-a. What are the most time-consuming steps of the code?

i. The AI does not report timing information. Based on code structure, the most time-consuming step is likely the per-unit, per-trial loop in `build_neural_trials`, which iterates over every unit and every trial individually to build histograms.

ii.
```python
for ui, u in enumerate(units):
    ...
    for tr in range(1, n_trials + 1):
        mask = trials == tr
        if np.any(mask):
            mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```

iii. The AI did not implement timing instrumentation or report bottlenecks.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The double loop over units and trials in `build_neural_trials` is the main candidate. The reference uses `np.histogram2d` to count all trials at once for each unit, which is much more efficient.

ii. Same as 11-a - the nested loop over units and trials.

iii. The reference vectorizes the trial dimension using `histogram2d`, reducing the inner loop from O(n_trials) to O(1) per unit.

## 11-c. What processing does the code repeat multiple times?

i. The code reads `bp['ev']['goCue']` multiple times - once in `align_tongue_speed` and once in `align_motion_energy`. The `common_time_axis()` function is called multiple times rather than computed once.

ii.
```python
# In align_tongue_speed:
align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
# In align_motion_energy:
align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
```

iii. These are minor inefficiencies.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads probe locations from `obj.meta.probe.loc` but then defaults to `['ALM']` if none are found. Since all sessions record from ALM, this probe location reading is unnecessary. The code also includes unused scipy fallback functions (`scipy_obj_to_bp`, `scipy_read_units`, `scipy_align_tongue`, `build_session_scipy`) at the bottom of the file that are never called.

ii.
```python
probe_locs = read_probe_locations(h)
if not probe_locs:
    probe_locs = ['ALM']
```

```python
# Dead code at bottom of file:
def scipy_obj_to_bp(obj_bp):
    ...
def scipy_read_units(obj_clu):
    ...
```

iii. The scipy fallback was noted as needed in CONVERSION_NOTES but was never integrated into the main processing pipeline, so these sessions remain skipped.
