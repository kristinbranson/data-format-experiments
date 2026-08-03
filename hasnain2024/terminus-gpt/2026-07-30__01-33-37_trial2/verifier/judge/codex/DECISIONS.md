# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code discovers sessions by globbing every immediate subdirectory under `data/`, pairing `data_structure_*.mat` and `motionEnergy_*.mat` files by stem. It loads `data_structure` files with a custom HDF5 reader (`h5py`) and loads motion-energy files with `scipy.io.loadmat`. Sessions whose `data_structure` files are unreadable are skipped rather than handled with a second MATLAB reader.

ii. 
```python
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
    return sessions

def load_data_structure(path):
    try:
        hfile = h5py.File(path, 'r')
    except OSError:
        return None

def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']
```

iii. The notes say the AI found four data subdirectories and mixed MATLAB formats, and the trajectory shows it chose to “discover_sessions” from filenames rather than transcribing the reference session list. Later trajectory steps say unreadable sessions were intentionally skipped as a robustness patch.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the filename stem before the first underscore. The final `subjects` list preserves first-seen order during the session loop rather than sorting unique names.

ii.
```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])

subject, _ = infer_subject_session(stem)
...
if sess['subject'] not in subject_to_idx:
    subject_to_idx[sess['subject']] = len(subjects)
    subjects.append(sess['subject'])
```

iii. The notes repeatedly mention parsing subject/date from filenames, and the trajectory says filename parsing was the stable way the AI could get animal IDs from the shared files.

## 1-c. How are the data split into sessions?

i. Each unique `<subject>_<date>` stem with both a `data_structure` file and a `motionEnergy` file is treated as one session candidate. Candidate sessions are sorted lexicographically, processed one by one, and then dropped if they are unreadable, missing required fields, missing paw features, or fail minimum trial/unit checks.

ii.
```python
keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
for stem in keys:
    sess = process_session(stem, sessions[stem], edges, centers)
    if sess is None:
        continue
    out_sessions.append(sess)
```

iii. In the notes the AI planned to “restrict to sessions with all required outputs available,” and the trajectory shows later patches to skip unreadable sessions and sessions lacking usable neural fields.

## 1-d. How are the data split into trials?

i. Trials are defined by `bp.Ntrials`. A boolean mask is built over those trial indices, and kept trial indices are `np.where(trial_mask)[0]`. Neural spikes are grouped by matching each spike’s `trial` id to those retained raw trial indices; label and behavioral traces are indexed with the same retained indices.

ii.
```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    ...
    return mask

idx = np.where(trial_mask)[0]
...
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The notes show the AI identified `bp.Ntrials` and the per-trial `bp` arrays as the core trial structure in the raw files.

## 1-e. How are trials filtered based on quality controls?

i. The code drops trials marked `early` or `no` (ignore). It does not drop photostimulation trials. At the session level it also skips sessions without motion energy, without usable paw features, with missing neural trial fields, with fewer than 2 kept trials, or with fewer than 10 kept units.

ii.
```python
for name in ['early', 'no']:
    ...
    mask &= (arr == 0)

if len(neural) < 2 or len(kept_units) < 10:
    print(f'  skip_reason: insufficient_neural trials={len(neural)} units={len(kept_units)}', flush=True)
    return None
```

iii. The notes explicitly say “Exclude early and ignore trials” and “Filter units by firing rate > 1 Hz and sessions with >=10 units.” The trajectory shows the missing-field and unreadable-session skips were added later as robustness patches.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays are derived from `obj['clu']['trialtm']` and `obj['clu']['trial']`. The code loads `quality`, `tm`, and `site` too, but the final neural values are computed only from `trialtm` and `trial`.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    ...
    clu_out[name] = ...

trialtm = clu['trialtm']
trialid = clu['trial']
```

iii. The notes identified `clu.trialtm` and `clu.trial` as per-unit spike times and trial assignments that “must be binned by trial and time.”

## 2-b. How is the `neural` data processed?

i. For each unit, the code estimates a crude firing rate as total spikes divided by `window_length * number_of_kept_trials`, drops units at `<= 1 Hz`, and then histograms raw `trialtm` values into fixed bins for each kept trial. It does not subtract go cue, does not convert counts to Hz after binning, does not smooth the spike trains, and does not concatenate multiple probes explicitly.

ii.
```python
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
...
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
    unit_trials.append(counts.astype(np.float32))
```

iii. The notes say the AI inferred a `>1 Hz` unit threshold from the paper, and the trajectory shows it also borrowed `75 ms` bins from the DLC decoding scripts; there is no explicit justification in the notes for omitting go-cue subtraction or smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code filters neural data only by requiring `clu.trialtm` and `clu.trial` to exist, keeping units whose estimated mean firing rate exceeds `1 Hz`, and discarding whole sessions with fewer than 10 such units. It does not use the loaded `quality` labels at all.

ii.
```python
if 'trialtm' not in clu or 'trial' not in clu:
    return None, None
...
if fr <= 1.0:
    continue
...
if len(neural) < 2 or len(kept_units) < 10:
    return None
```

iii. The notes planned a `>1 Hz` and `>=10 units/session` filter. The notes also discussed cluster quality metadata, but the final code did not implement any quality-label filtering; no later written justification for dropping that step appears.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not explicitly align spikes to the go cue. It bins each unit’s stored `trialtm` values directly against the common `[-2.5, 2.5]` edges for every trial, so its neural alignment assumes `trialtm` is already in go-cue-centered coordinates.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    ...

counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The notes and trajectory repeatedly cite “go cue” as the intended alignment event, but the final implementation never subtracts `bp.ev.goCue` in the neural path. There is no explicit written justification for that omission.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted dataset uses `75 ms` bins from `-2.5` to `2.5 s`, giving `66` time bins per trial. Neural data are binned directly onto this grid; there is no second-stage temporal rebinning.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers
...
'time_bin_size': 75.0,
```

iii. The notes explicitly mention that `DLC_ContextDecoding.m` uses `75 ms` bins, and the trajectory shows the AI latched onto that binning choice early.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input time channel is not read from a raw array. It is synthesized from the common bin centers returned by `build_time_grid`, with go cue entering only as the conceptual alignment event.

ii.
```python
edges, centers = build_time_grid()
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes say the decoder input should be a “continuous time vector repeated for each trial/session.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs evenly spaced centers over `[-2.5, 2.5)` with `75 ms` step size and repeats the same `1 x 66` vector for every retained trial.

ii.
```python
edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
centers = edges[:-1] + bin_size_s / 2
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes and trajectory justify this as the decoder’s single continuous time input on the chosen common grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is aligned to the neural data only by sharing the same `centers` array that the code uses as the session-wide time grid for binning/interpolating all streams.

ii.
```python
edges, centers = build_time_grid()
...
counts, _ = np.histogram(st[tr == raw_t], bins=edges)
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The code has no separate alignment logic for the input channel; it simply reuses the same bin grid everywhere.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial `bp['L']` and `bp['R']` arrays only.

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
...
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The notes originally identified `L`/`R` as the obvious left/right trial variables. I did not find a later written justification for not using `hit`/`miss` to infer actual lick choice.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each kept trial, if `R > L` the code labels the trial `1` (right); otherwise it labels it `0` (left). There is no separate no-lick class, and miss trials are not inverted to the opposite lick direction.

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, lick_dir[i], dtype=np.int64)
```

iii. The notes and trajectory do not give an explicit justification for this simplification; it appears to come from equating instructed side with lick direction after dropping ignore trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp['autowater']`.

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
```

iii. The notes explicitly connect context decoding and block identity to `bp.autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code directly maps `autowater > 0` to `0` (`WC`) and all other trials to `1` (`DR`).

ii.
```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. The notes say `bp.autowater` encodes the WC versus DR distinction, and the prompt required `WC = 0`, `DR = 1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp['hit']` and `bp['miss']`.

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
```

iii. The notes identified `hit` and `miss` as the required per-trial outcome flags.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code labels outcome as `1` when `hit > miss`, otherwise `0`. Because ignore trials were already removed by `valid_trials`, the final dataset has only incorrect/correct and no third ignore class.

ii.
```python
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, outcome[i], dtype=np.int64)
```

iii. The notes justify dropping ignore trials before constructing outputs, which is why only two outcome classes remain in the final code.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from side-camera trajectory data in `obj['traj'][0]`: `ts`, `frameTimes`, and `featNames`, plus `bp.ev.goCue`. The code looks for a tongue feature named `tongue`, `left_tongue`, or `right_tongue`. It does not use the bottom-camera tongue feature.

ii.
```python
side = traj_views[0]
side_names = [str(x) for x in side.get('featNames', [])]
...
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
...
tongue.append(interp_feature_velocity(ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
```

iii. The trajectory shows the AI fixed an earlier one-view bug after discovering `obj.traj` has two views, but the final tongue path still uses only the side view in `build_traj_outputs`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code interpolates tongue `x` and `y` positions onto the common time centers after shifting frame times by `0.5 s` and subtracting the trial’s go cue. It then computes velocity as the Euclidean norm of `np.gradient(x)` and `np.gradient(y)`. For tongue specifically, NaN gradients are replaced by zero. There is no likelihood threshold, no smoothing, no segmentation into contiguous valid runs, and no cross-view normalization/averaging.

ii.
```python
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

iii. The trajectory explicitly cites `WorkingWithDataObjs.m` as justification for a `0.5 s` camera time shift and a `(frames, 3, bodypart)` trajectory layout. I did not find a written justification for dropping the reference likelihood filtering and two-view combination.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After all retained tongue velocity bins in a session are stacked, the code threshold them at the session-wide median. Values above the median become `1`; all others become `0`.

ii.
```python
tongue = np.stack(tongue, axis=0)
...
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The notes say continuous behavioral outputs should be discretized per session at the 50th percentile, following the task prompt.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code aligns tongue velocity by converting frame times to `rel_t = frameTimes - 0.5 - goCue[trial]` and then interpolating onto the same `centers` grid used for neural bins. It does not estimate a session-specific video-to-behavior offset from bitcode.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
```

iii. The trajectory cites the reference comment that frame times should be shifted by `0.5 s` to sync camera and spikeGLX timing; that is the explicit written justification I found for the alignment choice.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera trajectory data in `obj['traj'][1]`: `ts`, `frameTimes`, and `featNames`, plus `bp.ev.goCue`. The code looks for `top_paw`, then `bottom_paw`, then any name containing `paw`.

ii.
```python
bottom = traj_views[1]
bottom_names = [str(x) for x in bottom.get('featNames', [])]
...
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
...
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```

iii. The trajectory shows the two-view fix was motivated by the discovery that paw features live in the bottom camera view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The paw path uses the same interpolation and gradient-magnitude pipeline as the tongue path, but with different NaN handling: if all gradients are NaN it returns zeros; otherwise NaNs are filled with the median gradient and the median is subtracted before taking the norm. There is no likelihood threshold, no smoothing, and no contiguous-run logic.

ii.
```python
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

iii. I did not find an explicit written justification for this paw-specific NaN handling; it appears to be a robustness choice made during implementation.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded at the session-wide median across all retained paw-velocity bins. Bins above the median are `1`; all others are `0`.

ii.
```python
paw = np.stack(paw, axis=0)
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. The notes say the movement outputs were to be discretized at the 50th percentile per session.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The code aligns paw velocity with `frameTimes - 0.5 - goCue[trial]`, then interpolates onto the shared `centers` grid. As with tongue, it does not compute the reference session-specific video offset from bitcode.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
...
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```

iii. The same `0.5 s` shift from the trajectory notes is the only explicit written justification I found.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the `me` variable loaded from `motionEnergy_<session>.mat`, typically from `me['data']` after unwrapping nested structures.

ii.
```python
def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']

def normalize_motion_energy_data(me):
    data = me
    if isinstance(data, dict):
        data = data.get('data', data)
    ...
    return list(data.reshape(-1))
```

iii. The notes identified `motionEnergy_*.mat` as a separate source file with `me.data` and `moveThresh`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code unwraps the motion-energy container, then linearly interpolates each trial’s 1D trace by normalized sample index to the common number of bins. It does not use actual frame times, camera offset, or go-cue timestamps for the motion-energy trace itself.

ii.
```python
def rebin_variable_trace(values, n_bins):
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    xp = np.linspace(0, 1, arr.size)
    xnew = np.linspace(0, 1, n_bins)
    return np.interp(xnew, xp, arr).astype(np.float32)
...
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The notes describe motion energy as a variable-length per-trial continuous trace that “must be temporally aligned/rebinned.” No more specific written justification for this normalized-index interpolation appears.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded at the session-wide median across all binned motion-energy values in `me_stack`. Values above the median become `1`, otherwise `0`.

ii.
```python
me_stack = np.stack(me_trials, axis=0)
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. The notes say movement outputs should be split at the per-session 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is implicit and coarse: the code keeps trials in the same order as the neural data and rescales each trial’s motion-energy trace to the same number of bins. It does not align motion energy to go cue using frame times or any video clock offset.

ii.
```python
idx, lick_dir, context, outcome = build_trial_labels(obj, trial_mask)
...
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The notes say motion-energy traces needed “rebin/alignment to a common time grid,” but I found no explicit written justification for ignoring frame timing information.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mostly handles problems by skipping entire sessions or substituting zeros. Unreadable `data_structure` files, missing motion-energy files, missing neural trial-aligned fields, bad motion-energy length checks, no paw feature, and insufficient units/trials all cause the session to be skipped. Empty motion-energy traces yield all-zero rebinned traces. Missing or mismatched frame times are replaced with fabricated evenly spaced times, and missing/invalid trajectory arrays produce all-zero velocities.

ii.
```python
if obj is None:
    return None
if 'motion_energy' not in files:
    return None
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
```

iii. The trajectory explicitly describes several of these as robustness patches added after crashes or skipped-session issues.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive parts are likely reading/parsing MATLAB/HDF5 session files, the nested unit-by-trial spike histogramming in `bin_spikes_for_session`, and the per-trial interpolation/gradient work for trajectories and motion energy.

ii.
```python
obj = load_data_structure(files['data_structure'])
...
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
...
for tr in trial_idx:
    ...
    tongue.append(interp_feature_velocity(...))
    paw.append(interp_feature_velocity(...))
```

iii. I did not find an explicit efficiency analysis in the notes; this is inferred from the code structure and from the trajectory’s repeated discussion of long-running full conversions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code has several obvious Python loops that could be reduced: the per-unit/per-trial spike binning loop, the per-trial tongue/paw extraction loop, the per-trial motion-energy rebinning loop, and the final per-trial output assembly loop.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)

for tr in trial_idx:
    ...

me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]

for i in range(len(idx)):
    outputs.append(np.vstack([...]))
```

iii. There is no written justification in the notes for keeping these as Python loops.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes `np.where(trial_mask)[0]` inside the unit loop, repeatedly calls `np.histogram` once per kept trial per unit, repeatedly casts the same `centers[None, :]` array per trial, and repeatedly constructs full-length constant output rows trial by trial.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
...
for i in range(len(idx)):
    outputs.append(np.vstack([
        np.full(n_bins, lick_dir[i], dtype=np.int64),
        ...
    ]))
```

iii. No explicit justification for these repetitions appears in the notes or trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader reads several fields that the final conversion does not use (`tm`, `site`, `quality`, `meta`, `fn`, `NdroppedFrames`). It also defines helper functions that are not used in the final pipeline (`read_dataset_maybe_refs`, `read_char_ref_array`, `read_numeric_ref_array`). `infer_subject_session` returns a date component that is immediately discarded, and `kept_units` is stored only to size a zero-valued `brain_region_idx`.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    ...

for name in ['ts', 'frameTimes', 'fn', 'NdroppedFrames']:
    ...

def read_dataset_maybe_refs(h, ds):
    ...

subject, _ = infer_subject_session(stem)
...
'brain_region_idx': np.zeros(len(kept_units), dtype=np.int64),
```

iii. I did not find a written justification for these extra reads/helpers; they look like remnants of exploratory loading code that stayed in the final script.
