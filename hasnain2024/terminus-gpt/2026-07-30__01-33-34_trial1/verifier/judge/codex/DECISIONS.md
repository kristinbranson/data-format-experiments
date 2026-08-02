# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a randomized-delay subset and loads one `data_structure_*.mat` file plus one `motionEnergy_*.mat` file per session from `data/RandomizedDelay_Ephys_Behavior`. It does not iterate over every raw directory in `/app/data`; instead it uses the 19-session `RAND_SESSIONS` list that mirrors the reference loader scripts for JEB11/JEB12/JEB23/JEB24. Each session is parsed by `process_session`, which first tries an HDF5/v7.3 reader and falls back to `scipy.io.loadmat` for older MAT files.

ii. ```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
RAND_SESSIONS = [
    'JEB11_2022-05-10','JEB11_2022-05-11',
    'JEB12_2022-05-12','JEB12_2022-05-13',
    'JEB23_2023-10-10', ...,
    'JEB24_2023-11-03'
]

base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
```

iii. In `CONVERSION_NOTES.md` Step 4-5, the agent says it chose the explicit randomized-delay loader list because the paper reports 19 randomized-delay sessions from 4 mice, while the raw directory contains 22 sessions. The trajectory repeats that rationale and treats the loader-script session list as the paper-consistent subset.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session key prefix before the first underscore, e.g. `JEB11_2022-05-10 -> JEB11`. The code keeps a running `subject_names` list and records one `subject_idx` per session.

ii. ```python
for key in keys:
    subj = key.split('_')[0]
    if subj not in subject_names:
        subject_names.append(subj)
    ...
    subject_idx.append(subject_names.index(subj))
```

iii. The notes justify this through the reference loader scripts, which enumerate sessions by mouse name and date, and the trajectory states that the randomized-delay subset consists of four mice: JEB11, JEB12, JEB23, and JEB24.

## 1-c. How are the data split into sessions?

i. Each `(subject, date)` file pair is treated as one session. The outer dataset lists are appended once per key in `RAND_SESSIONS`, so each session remains separate in `neural`, `input`, `output`, and `brain_region_idx`.

ii. ```python
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
    all_neural.append(neural)
    all_input.append(inp)
    all_output.append(out)
    brain_region_idx.append(bri)
```

iii. The notes say the loader-script session list is the authoritative session split for the paper’s randomized-delay analysis and should be used instead of blindly converting all raw files.

## 1-d. How are the data split into trials?

i. Within each session, the code uses `bp['Ntrials']` as the trial count, assumes behavior arrays are already trial-indexed, assumes spike events carry integer trial IDs, and builds one neural/input/output entry per trial.

ii. ```python
n_trials = bp['Ntrials']
neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
input_trials = make_time_input(n_trials, time_centers)
output_trials = []
for ti in range(n_trials):
    output_trials.append(np.vstack([...]))
```

iii. The notes document that `obj.bp.ev.goCue` is length `Ntrials`, `me.data` is per trial, and cluster spike records include `trial` labels, so the agent decided the natural split is one trial per row/index across these arrays.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they are not early-lick trials and not `no` trials, then further filtered if the neural matrix for that trial is entirely zero. Miss trials are kept. There is no trial filtering by `stim.enable`, delay length, or session-level condition counts.

ii. ```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
...
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
neural = [neural[i] for i in keep_idx]
input_trials = [input_trials[i] for i in keep_idx]
output_trials = [output_trials[i] for i in keep_idx]
```

iii. In Step 3-5 of the notes, the agent cites the methods statement that early lick and ignore trials are omitted, then later the trajectory says it added the all-zero-neural-trial drop after verification warnings. No stronger trial curation rule is documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived only from per-unit spike times and their trial IDs in `obj.clu`, specifically `trialtm` and `trial`. The agent ignores cluster quality labels except during its exploratory notes.

ii. ```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]

def extract_clu_old(obj):
    clu_arr = np.asarray(obj.clu, dtype=object).ravel()
    trialtm = [np.asarray(unwrap_obj(c).trialtm).ravel().astype(np.float32) for c in clu_arr]
    trial = [np.asarray(unwrap_obj(c).trial).ravel().astype(int) for c in clu_arr]
```

iii. The notes repeatedly identify `obj.clu` as the neural source, and the trajectory frames the converter as reconstructing per-trial neural matrices directly from `trialtm`/`trial`.

## 2-b. How is the `neural` data processed?

i. The agent bins each unit’s raw spike times directly into 75 ms bins between `-2.5` and `2.5` seconds and stores spike counts, not smoothed firing rates. It does not call the reference sequence of align-to-event, 10 ms PSTH generation, Gaussian smoothing, and low-firing-rate removal.

ii. ```python
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    n_units = len(trialtm_list)
    n_bins = len(time_edges) - 1
    per_trial = [np.zeros((n_units, n_bins), dtype=np.float32) for _ in range(n_trials)]
    for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
        for trial_idx in range(1, n_trials + 1):
            mask = tr == trial_idx
            if np.any(mask):
                counts, _ = np.histogram(ttm[mask], bins=time_edges)
                per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. The notes say the agent chose “75 ms bins as initial default” because the decoder scripts use `rez.binSize = 75`. The trajectory never claims to reproduce the upstream `alignSpikes -> getSeq -> removeLowFRClusters` pipeline; it instead focuses on direct binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neural quality filtering beyond removing trials whose entire neural matrix is zero. All units in `obj.clu` are kept, regardless of quality string or firing rate. Sessions are not dropped for having fewer than 10 units.

ii. ```python
trialtm, trial, n_units = extract_clu_h5(f)
...
brain_region_idx = np.zeros((n_units,), dtype=int)
...
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The notes acknowledge the paper’s curation rules (`quality`, `>1 Hz`, `>=10` units/session), but the trajectory never implements them; it only adds the all-zero-trial filter after a verifier warning.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are treated as if `trialtm` is already in go-cue-relative coordinates. The code never subtracts `bp['goCue']` from spikes and never reconstructs the paper’s `trialtm_aligned` field.

ii. ```python
trialtm, trial, n_units = extract_clu_h5(f)
...
counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

iii. The notes identify go cue as the alignment event, but the trajectory focuses on choosing the time window and does not mention recreating the reference `alignSpikes` step. That implies the agent assumed the stored `trialtm` could be binned directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 75 ms time bin (`BIN_SIZE = 0.075`) from `-2.5` to `2.5` seconds. The agent does one direct rebinning pass into those bins and does not preserve the 10 ms intermediate representation used in the reference loading code.

ii. ```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
...
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
```

iii. Step 1 and Step 5 of the notes justify 75 ms from the decoder scripts’ `rez.binSize = 75`, and the trajectory explicitly says it adopted 75 ms as the default analysis bin.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not computed from a per-trial raw timestamp array. The agent uses the synthetic bin centers implied by `T_START`, `T_END`, and `BIN_SIZE`, under the assumption that all trials are already aligned to go cue.

ii. ```python
def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
...
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
input_trials = make_time_input(n_trials, time_centers)
```

iii. The notes argue that go cue is the canonical alignment event and that a common time axis should be used for all trials. The trajectory does not show any attempt to derive a trial-specific time axis from raw go-cue timestamps beyond that alignment assumption.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes evenly spaced bin centers over the fixed window and copies the same 1 x T vector into every trial.

ii. ```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
input_trials = make_time_input(n_trials, time_centers)
```

iii. The notes justify this as the simplest way to express “time from go cue” once all modalities share a common aligned time base.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is aligned by construction: the time input uses the same `time_edges`/`time_centers` as the neural binning code, so each trial’s input vector has the same number of time points as the neural matrix.

ii. ```python
neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
...
input_trials = make_time_input(n_trials, time_centers)
```

iii. The notes and trajectory both treat the common go-cue-centered time grid as the core organizing principle across neural and behavioral streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `lick_direction` is derived from the behavioral flags `bp['L']` and `bp['R']`.

ii. ```python
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
...
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. The notes map left/right choice directly to `obj.bp.L` and `obj.bp.R`, consistent with how the reference condition strings are written.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code maps right trials to `1`, left trials to `0`, then repeats that scalar across all time bins in the trial.

ii. ```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
...
np.full((len(time_centers),), lick_dir[ti], dtype=np.int64)
```

iii. The notes justify this as a per-trial categorical output. The trajectory does not document any more complex transformation.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `behavioral_context` is derived from `bp['autowater']`.

ii. ```python
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
...
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. In Step 4-5 of the notes, the agent explicitly resolves context as `WC = autowater`, `DR = ~autowater`, citing the paper’s code and behavior scripts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps autowater-on trials to `0` (`WC`) and all others to `1` (`DR`), then repeats that category across time within each trial.

ii. ```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
...
np.full((len(time_centers),), context[ti], dtype=np.int64)
```

iii. The notes explicitly say “Map WC=autowater, DR=not autowater unless contradicted,” and the trajectory later treats this mapping as settled.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `outcome` is derived primarily from `bp['hit']`, with `bp['no']` removed earlier by trial filtering and non-hit valid trials treated as incorrect.

ii. ```python
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
...
valid = (bp['early'] == 0) & (bp['no'] == 0)
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. The notes describe outcome as “likely hit=1 and miss/no=0 after trial filtering,” which is the rule the code implements.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. After removing early and `no` trials, the code maps `hit` to `1` and every remaining non-hit trial to `0`, then repeats that scalar over time.

ii. ```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
...
np.full((len(time_centers),), outcome[ti], dtype=np.int64)
```

iii. The notes justify the binary reduction because the decoder task asks for `incorrect` versus `correct`, not separate `miss` and `no` categories.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` pose trajectories and frame times. The code prefers view-2 tongue landmarks (`top_tongue`, `bottom_tongue`, `topleft_tongue`, `bottomleft_tongue`) and falls back to view-1 features (`tongue`, `left_tongue`, `right_tongue`).

ii. ```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
...
ts, ft = getter(ti)
spd, tmid = speed_from_ts(ts, ft)
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The notes say tongue-related trajectories in `obj.traj` should drive the tongue output, and the trajectory says it “decoded inner `traj` feature names/time series” to do that.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The agent computes Euclidean frame-to-frame speed from the selected tongue feature’s x/y coordinates, then averages that speed within each neural time bin. It does not reconstruct the reference interpolation, smoothing, tongue-baseline replacement, or x/y-velocity feature set.

ii. ```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    dt = np.diff(frame_times).astype(np.float32)
    dt[dt <= 0] = np.nan
    dxy = np.diff(xy, axis=2)
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
    tmid = (frame_times[:-1] + frame_times[1:]) / 2
    return spd, tmid
```

iii. The notes only justify this at a high level as “compute velocity, then discretize per session.” The trajectory does not cite any specific reference function for the tongue-velocity formula, so this is an ad hoc simplification.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code concatenates all finite binned tongue-speed values in a session, takes the session median, and labels each valid sample as `1` if it is at or above that threshold, else `0`.

ii. ```python
def threshold_session_bins(arr_list):
    finite_chunks = [a[np.isfinite(a)] for a in arr_list if np.any(np.isfinite(a))]
    allv = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    thr = np.nanmedian(allv) if allv.size else np.nan
    ...
    b[valid] = (a[valid] >= thr).astype(np.int64)
...
tongue_bin, tongue_thr = threshold_session_bins(tongue_vals)
```

iii. The notes explicitly planned “session-level thresholding required by task,” and the trajectory reports that implementing non-degenerate tongue binning was one of the main fixes before validation passed.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code subtracts each trial’s `goCue` time from the tongue feature’s mid-frame times and bins the resulting samples into the same `time_edges` used for neural data.

ii. ```python
go = bp['goCue'][ti]
...
spd, tmid = speed_from_ts(ts, ft)
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The notes justify go-cue alignment globally. The trajectory never mentions the reference video offset correction, so the implemented alignment rule is simply “frame times minus go cue, then bin to neural edges.”

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the view-2 `obj.traj` features `top_paw` or `bottom_paw` plus their frame times.

ii. ```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
...
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
    spd, tmid = speed_from_ts(ts, ft)
    paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The notes map paw velocity to `obj.traj` paw-related trajectories and cite the DLC decoding scripts’ paw feature group as the reference motivation.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code computes Euclidean frame-to-frame speed for one selected paw landmark and averages it into neural bins. It does not reconstruct the reference interpolation, smoothing, baseline correction, or separate x/y velocity channels.

ii. ```python
ts, ft = get_stream1(ti)
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The notes only commit to “compute velocity, then discretize per session,” and the trajectory contains no deeper reference-based justification for the exact paw-speed formula.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the same per-session median-threshold rule as tongue velocity.

ii. ```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. The notes say session-level 50th-percentile thresholding is required for these outputs, and the trajectory treats this as a task-driven rule rather than something extracted from the MATLAB code.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw mid-frame times are shifted by the trial’s go-cue time and then binned on the same neural edges.

ii. ```python
go = bp['goCue'][ti]
...
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The notes justify the shared go-cue-centered time axis, and the trajectory does not mention any further correction such as `findVideoOffset`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the separate `motionEnergy_*.mat` file’s `me` struct, especially `me.data`. For timing, the code uses view-1 trajectory frame times from `obj.traj`.

ii. ```python
def load_motion_energy(path):
    me = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['me']
    move_thresh = getattr(me, 'moveThresh', None) if hasattr(me, '_fieldnames') else None
    ...
    out = [normalize_motion_elem(x) for x in arr.flat]
    return out, move_thresh
...
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. The notes explicitly map motion energy to `me.data`, note the existence of `moveThresh`, and describe the need to normalize several nested MAT-file layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code unwraps nested MAT structs/objects until it gets numeric trial traces, then averages each trace into neural time bins using view-1 frame times. If the frame-time length does not match the motion-energy vector, it replaces that trial with all-NaN motion energy.

ii. ```python
while hasattr(raw, '_fieldnames') and hasattr(raw, 'data'):
    raw = raw.data
arr = np.asarray(raw, dtype=object)
out = [normalize_motion_elem(x) for x in arr.flat]
...
if ti < len(motion_data):
    me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
    _, ft0 = get_stream0(ti)
    me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
else:
    me_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
```

iii. The trajectory spends many steps on this point: it documents mixed MAT formats, nested `me.data` wrappers, and later-session mismatches, and justifies the recursive unwrapping plus NaN fallback as robustness fixes needed to finish the conversion.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized with the same session-median thresholding used for tongue and paw velocity.

ii. ```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. The notes explicitly say the task requires per-session thresholding at the 50th percentile, so the agent applies that rule even though it also records the raw `moveThresh` in metadata.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The code aligns motion energy by subtracting `goCue` from view-1 frame times and binning the samples to the neural `time_edges`. It does not apply the reference video-start offset correction.

ii. ```python
go = bp['goCue'][ti]
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. The notes and trajectory justify a shared go-cue-centered grid, but they do not claim the more detailed `findVideoOffset` interpolation used by the paper code.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code is defensive about MAT-file format variation and missing traces. It supports HDF5 and old MAT session files, recursively unwraps nested motion-energy objects, returns empty arrays for empty object cells, replaces missing or mismatched motion-energy trials with NaNs, leaves missing kinematic bins as NaNs until thresholding, and drops all-zero neural trials. It does not imitate the reference code’s `fillmissing(..., 'nearest')` behavior for kinematic/motion traces.

ii. ```python
try:
    with h5py.File(data_path, 'r') as f:
        ...
except OSError:
    obj = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)['obj']
...
if isinstance(x, np.ndarray) and x.dtype == object:
    if x.size == 0:
        return np.array([], dtype=np.float32)
...
if ti < len(motion_data):
    ...
else:
    me_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
...
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The trajectory gives the clearest justification here: many later steps are about mixed MAT layouts, nested `me.data` containers, and verifier warnings, and each patch is explained as a robustness fix needed to keep the conversion running on all 19 sessions.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are the nested Python loops over units and trials for neural binning and the per-trial/per-stream kinematic extraction. Motion-energy normalization and MAT-file parsing are secondary costs.

ii. ```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        ...

for ti in range(n_trials):
    ...
    for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
        ...
```

iii. The notes’ “Code inefficiencies identified” section and the trajectory both highlight full-conversion runtime, repeated motion-energy parsing, and the cost of rebuilding per-trial tensors in Python loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit-by-trial spike histogram loop, the per-bin `nanbin_mean` loop, and the repeated per-trial kinematic loop could all have been vectorized or batched.

ii. ```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        ...

for bi in range(len(out)):
    m = idx == bi
    ...

for ti in range(n_trials):
    ...
```

iii. The notes identify speedups as still pending, and the trajectory repeatedly describes these loops as the work that dominates conversion runtime.

## 11-c. What processing does the code repeat multiple times?

i. It duplicates nearly the same extraction path for HDF5 versus old MAT session files, recalculates trajectory speeds trial-by-trial for each feature, and scans all session values again for each thresholded output.

ii. ```python
try:
    with h5py.File(data_path, 'r') as f:
        ...
        tongue_vals, paw_vals, me_vals = extract_binned_behavior_generic(...)
except OSError:
    obj = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)['obj']
    ...
    tongue_vals, paw_vals, me_vals = extract_binned_behavior_generic(...)

tongue_bin, tongue_thr = threshold_session_bins(tongue_vals)
paw_bin, paw_thr = threshold_session_bins(paw_vals)
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. The notes and trajectory both describe the mixed-format support as a practical necessity, but it still creates repeated logic and repeated passes through the same session data.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads several behavioral fields that are not used in the final outputs (`bitRand`, `reward`, `moveThresh` except for metadata), computes full time-varying copies of scalar trial labels, and parses motion-energy thresholds that are only stored in `session_info`. It also accepts CLI flags like `--full` and `--show-processing` that do nothing.

ii. ```python
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = ...
for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
    out[k] = ...
...
info = {
    'n_trials_raw': n_trials,
    'n_trials_kept': len(keep_idx),
    'moveThresh': move_thresh,
    'tongue_threshold': ...,
    'paw_threshold': ...,
    'motion_threshold': ...,
}
...
ap.add_argument('--full', action='store_true')
ap.add_argument('--sample', action='store_true')
ap.add_argument('--show-processing', action='store_true')
```

iii. The notes focus on getting a valid converted dataset rather than minimizing work, and the trajectory shows several pieces of exploratory or defensive bookkeeping that survive into the final script even though the decoder only consumes the final tensors.
