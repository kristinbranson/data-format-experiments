# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed the `data/` directory at runtime, pairing any `data_structure_*.mat` and `motionEnergy_*.mat` files with the same stem, instead of using the fixed session list from the reference. `data_structure` files are loaded only through `h5py`; unreadable files are skipped. Motion-energy files are loaded with `scipy.io.loadmat`.

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

def load_data_structure(path):
    out = {}
    try:
        hfile = h5py.File(path, 'r')
    except OSError:
        return None
```

iii. The notes justify this by saying the dataset had mixed MAT-file formats and that filename parsing found 18 unique subjects, so the agent treated session discovery from filenames as sufficient. Later notes also describe skipped sessions and report that 33 sessions were kept after conversion.

## 1-b. How are the data split into subjects (mice)?

i. The AI took the subject id to be the substring before the first underscore in the session stem, then built `subjects` in first-seen order and `subject_idx` from that order.

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

iii. The notes say filename parsing was used to count subjects because the filenames consistently encode animal ids.

## 1-c. How are the data split into sessions?

i. One session is any filename stem that has both a `data_structure` file and a `motionEnergy` file. Sessions are sorted lexicographically, processed one by one, and dropped entirely if `process_session` returns `None`.

ii.
```python
keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
for stem in keys:
    sess = process_session(stem, sessions[stem], edges, centers)
    if sess is None:
        continue
    out_sessions.append(sess)
```

iii. The notes frame this as selecting sessions that have the required modalities for decoder outputs, and the final notes explicitly say 33 sessions were kept.

## 1-d. How are the data split into trials?

i. Trials are indexed from `0` to `Ntrials - 1`. The AI constructs a boolean `trial_mask`, takes `idx = np.where(trial_mask)[0]`, and uses those raw trial indices to pull spikes, behavior labels, video features, and motion-energy traces.

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

iii. The notes justify this by saying `bp.Ntrials` and the per-trial behavioral arrays define the trials directly, and that `clu.trial` provides the trial assignment for spikes.

## 1-e. How are trials filtered based on quality controls?

i. The AI filtered out trials where `bp.early != 0` or `bp.no != 0`. It did not filter photostimulation trials. Sessions were also dropped if they ended up with fewer than 2 kept trials or fewer than 10 kept units.

ii.
```python
for name in ['early', 'no']:
    if name in bp:
        arr = np.array(bp[name], dtype=float).reshape(-1)
        if arr.size == n:
            mask &= (arr == 0)

if len(neural) < 2 or len(kept_units) < 10:
    return None
```

iii. The notes claim “Exclude early and ignore trials before constructing decoder dataset” and “sessions with >=10 units,” citing the methods text. No note justifies the omission of photostimulation filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from `obj['clu']['trialtm']` and `obj['clu']['trial']` taken from the first dereferenced `clu` entry only. It also used `bp.Ntrials` indirectly through `trial_mask`, but did not use cluster quality labels in the neural construction itself.

ii.
```python
clu_ref = obj['clu'][()].reshape(-1)[0]
clu = deref(h, clu_ref)
...
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    if name in clu:
        ...

trialtm = clu['trialtm']
trialid = clu['trial']
```

iii. The notes identify `clu.trialtm` and `clu.trial` as the key neural raw variables and mention `quality/site` as available metadata, but the code path actually only relies on `trialtm` and `trial` for building neural arrays.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the AI histogrammed raw spike times into a fixed `75 ms` grid for each kept trial and stored those bin counts directly as `float32`. It did not divide by bin width, smooth, z-score, or baseline-correct.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers

counts, _ = np.histogram(st[tr == raw_t], bins=edges)
unit_trials.append(counts.astype(np.float32))
```

iii. The notes repeatedly justify the `75 ms` bin choice by reference to DLC decoding scripts, but they do not give a separate justification for leaving neural data as unsmoothed spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI dropped units with an approximate firing rate of `<= 1 Hz`, computed as total spikes divided by `window_duration * n_valid_trials`. It did not filter on `quality` labels. Entire sessions were dropped if fewer than 10 units remained.

ii.
```python
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
...
if len(neural) < 2 or len(kept_units) < 10:
    return None
```

iii. The notes explicitly cite the paper’s “>1 Hz” rule and “sessions with >=10 units.” They do not explain why `quality` labels are loaded but never applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI did not subtract each trial’s go-cue time from spike times. It histogrammed `clu.trialtm` directly into the common `[-2.5, 2.5]` bin edges and implicitly treated those times as already aligned.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The notes say the dataset should be go-cue aligned, and they mention `bp.ev.goCue` as the alignment event, but there is no corresponding go-cue subtraction in the neural code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI used a `75 ms` time bin, spanning `-2.5` to `2.5 s`. Neural data were binned directly onto this grid, so the effective rebinning happens at conversion time from spikes to 75 ms counts.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
```

iii. The trajectory and notes explicitly justify `75 ms` by referring to MATLAB DLC decoding scripts with `rez.binSize = 75`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. In the code, this input is not derived from any per-trial raw variable. It is a synthetic vector of shared bin centers produced by `build_time_grid`.

ii.
```python
edges, centers = build_time_grid()
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes justify the concept of “time from go cue” from the task instructions and reference alignment choice, but the implementation itself uses only the fixed window, not raw `goCue` values.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI generated evenly spaced bin centers from `-2.5` to `2.5 s` in `75 ms` steps, then duplicated that same 1D array for every trial.

ii.
```python
edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
centers = edges[:-1] + bin_size_s / 2
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes tie this to the chosen common decoder grid; no other processing or raw-data-derived correction is described.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The AI aligned the input to the neural data only by using the same `edges/centers` arrays for both. There is no additional trial-specific alignment step.

ii.
```python
edges, centers = build_time_grid()
...
counts, _ = np.histogram(st[tr == raw_t], bins=edges)
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes justify using a shared time grid across modalities; they do not document any separate check that neural spikes were actually shifted onto the go-cue clock.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derived lick direction only from the per-trial `L` and `R` flags in `bp`. It did not use `hit`, `miss`, or `no` when defining the direction class.

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
...
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The notes identify `bp.L` and `bp.R` as the source variables and frame lick direction as a per-trial categorical label.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI converted each kept trial to a binary left/right label: `1` if `R > L`, otherwise `0`. It then repeated that scalar across all time bins of the trial.

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, lick_dir[i], dtype=np.int64)
```

iii. The notes do not give a separate technical justification beyond mapping left/right trial identity to a decoder target.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp['autowater']` flag.

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. The notes explicitly say context should be mapped from `bp.autowater` and cross-reference the MATLAB helper that derives blocks from autowater transitions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabeled `autowater > 0` as `WC` and the remaining trials as `DR`, encoded as `0` and `1` respectively, and repeated that per-trial label across time bins.

ii.
```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
...
np.full(n_bins, context[i], dtype=np.int64)
```

iii. The notes justify this directly from the task requirement and the paper/code’s use of autowater-defined context.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial `bp['hit']` and `bp['miss']` arrays.

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The notes justify this from the methods and from the same behavioral fields used elsewhere for trial labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI encoded outcome as a binary variable: `1` for trials where `hit > miss`, otherwise `0`. Because `valid_trials` already removes `bp.no != 0`, ignore trials are excluded instead of being assigned a third class.

ii.
```python
def valid_trials(bp):
    ...
    for name in ['early', 'no']:
        ...

outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The notes explicitly say to exclude ignore trials before building the decoder dataset, which is the justification that matches the implemented binary outcome.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derived tongue velocity from side-camera trajectory data: `traj_views[0]['ts']`, `traj_views[0]['frameTimes']`, `traj_views[0]['featNames']`, and `bp.ev.goCue`. It searched side-view feature names for `tongue`, `left_tongue`, or `right_tongue`.

ii.
```python
side = traj_views[0]
side_names = [str(x) for x in side.get('featNames', [])]
go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)
...
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
...
ts_side = side['ts'][tr]
ft_side = side['frameTimes'][tr]
```

iii. The trajectory shows the agent debugging feature-name extraction and concluding that side-view tongue features were available while many sessions lacked paw features. The notes also describe a “two-view trajectory loading bug” but the final code uses only the side-view tongue stream.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolated tongue x/y positions onto the decoder bin centers, took gradients of the interpolated traces, zero-filled NaN gradients for tongue, and used the Euclidean speed. It did not apply likelihood filtering, contiguous-run handling, Gaussian smoothing, per-view normalization, or multi-view averaging.

ii.
```python
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
xv = np.gradient(x)
yv = np.gradient(y)
if tongue:
    xv[np.isnan(xv)] = 0
    yv[np.isnan(yv)] = 0
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The notes justify making continuous outputs time-varying and using trajectory features from `traj.ts/frameTimes`, but they do not justify the specific interpolation-and-gradient procedure used here.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After building a session array of tongue speeds, the AI thresholded it at the overall median and produced only two classes: `0` and `1`.

ii.
```python
tongue = np.stack(tongue, axis=0)
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The notes justify a per-session `50th percentile` threshold, but they do not justify the lack of a `not visible` third class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI aligned tongue trajectories by computing `rel_t = (frame_times - 0.5) - goCue[trial]`, then interpolating onto the same decoder bin centers used elsewhere. It did not use the video/behavior offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart`.

ii.
```python
if ft.size == 0:
    ft = np.arange(1, xy.shape[0] + 1, dtype=float) / 400.0
else:
    ft = np.linspace(ft.min(), ft.max(), xy.shape[0])
rel_t = (ft - 0.5) - float(align_time)
```

iii. The notes say the intended alignment event is the go cue, but there is no explicit note defending the hard-coded `0.5` second offset or the absence of the paper’s bitcode-based correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI derived paw velocity from bottom-camera trajectory data: `traj_views[1]['ts']`, `traj_views[1]['frameTimes']`, `traj_views[1]['featNames']`, and `bp.ev.goCue`. It searched for `top_paw`, `bottom_paw`, or `paw`.

ii.
```python
bottom = traj_views[1]
bottom_names = [str(x) for x in bottom.get('featNames', [])]
...
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
...
ts_bot = bottom['ts'][tr]
ft_bot = bottom['frameTimes'][tr]
```

iii. The trajectory shows the agent focusing on finding sessions with any paw feature and skipping sessions without one. The notes justify paw extraction as necessary for the requested decoder outputs.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI used the same interpolate-then-gradient speed estimate as for tongue, but for non-tongue features it imputed NaN gradients with the median gradient and subtracted the median before computing speed.

ii.
```python
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
xv = np.gradient(x)
yv = np.gradient(y)
if np.all(np.isnan(xv)) or np.all(np.isnan(yv)):
    return np.zeros(len(centers), dtype=np.float32)
xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
xv = xv - np.nanmedian(xv)
yv = yv - np.nanmedian(yv)
```

iii. No explicit note justifies this specific preprocessing beyond the general plan to derive time-varying paw-velocity outputs from trajectory features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI split paw speeds at the median and encoded only two categories, `0` and `1`.

ii.
```python
paw = np.stack(paw, axis=0)
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. The notes justify the use of a per-session median threshold, but they do not justify omitting the prompt’s `not visible` class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw alignment uses the same `rel_t = (frame_times - 0.5) - goCue[trial]` formula and the same shared bin centers as the neural and input arrays.

ii.
```python
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
...
rel_t = (ft - 0.5) - float(align_time)
```

iii. The notes say all streams should share a common go-cue-centered grid, but they do not document the hard-coded frame-time offset.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the session’s separate `motionEnergy_*.mat` file, loaded as `me` and normalized into a per-trial Python list or array.

ii.
```python
def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']

me = load_motion_energy(files['motion_energy'])
me_data = normalize_motion_energy_data(me)
```

iii. The notes explicitly identify `me.data` as the relevant field and describe the need to unwrap different MAT layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI treated each trial’s raw motion-energy trace as a 1D series of arbitrary length and linearly interpolated it to the common number of decoder bins. It then thresholded the stacked result at the median.

ii.
```python
def rebin_variable_trace(values, n_bins):
    arr = np.array(values, dtype=float).reshape(-1)
    ...
    xp = np.linspace(0, 1, arr.size)
    xnew = np.linspace(0, 1, n_bins)
    return np.interp(xnew, xp, arr).astype(np.float32)

me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The notes justify that motion energy is already present as a per-trial continuous trace and must be rebinned to the decoder grid, but do not justify the choice to reparameterize only by trace index.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI used a single median split over the session’s rebinned motion-energy values and encoded only two classes, `0` and `1`.

ii.
```python
me_stack = np.stack(me_trials, axis=0)
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. The notes explicitly justify a session-wise 50th-percentile split, but they do not justify omitting the `no video` class required by the task.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned only implicitly by resampling each trial trace to the same number of decoder bins used for neural data. The code does not use camera frame times, the video/behavior offset, or go-cue-centered frame timestamps.

ii.
```python
edges, centers = build_time_grid()
...
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The notes say motion energy must be “rebinned/aligned to a common time grid around go cue,” but the implementation uses only uniform interpolation to `n_bins`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handled missing or awkward data by either skipping whole sessions or substituting fallback numeric values. Unreadable HDF5 files, missing motion-energy files, missing paw features, missing trial-aligned spikes, or insufficient units/trials all cause the entire session to be dropped. Empty traces become zeros; empty frame times are replaced by a synthetic `400 Hz` timebase; frame-time length mismatches are replaced by `linspace`; NaN tongue gradients are zero-filled and NaN paw gradients are median-filled.

ii.
```python
except OSError:
    return None
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
    return None
```

iii. The notes justify session skipping as restricting to sessions with all required outputs available, and they mention fixing trajectory-loading issues, but they do not justify converting missing visibility/video information into ordinary low-valued traces.

## 11-a. What are the most time-consuming steps of the code?

i. The code suggests the most time-consuming steps are session loading through `h5py`, the nested per-unit/per-trial spike histogram loop, and the per-trial trajectory interpolation loop. Unlike the reference, the AI did not vectorize spike counting across trials.

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
    tongue.append(...)
    paw.append(...)
```

iii. The notes do not explicitly discuss runtime here, so this is inferred from the control flow the AI wrote.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the nested unit-by-trial spike binning loop, the per-trial motion-energy interpolation loop, the per-trial tongue/paw loop, and the final per-trial output assembly loop.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)

me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]

for tr in trial_idx:
    ...

for i in range(len(idx)):
    outputs.append(np.vstack([...]))
```

iii. No explicit justification is given in the notes for leaving these loops unvectorized.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several computations across loops: `np.where(trial_mask)[0]` is recomputed inside spike binning, constant `centers[None, :]` arrays are rebuilt for every trial, per-trial `np.full(...)` arrays are rebuilt for each static label channel, and per-trial trajectory interpolation is run separately for tongue and paw.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    ...

inputs = [centers[None, :].astype(np.float32) for _ in idx]

for i in range(len(idx)):
    outputs.append(np.vstack([
        np.full(n_bins, lick_dir[i], dtype=np.int64),
        np.full(n_bins, context[i], dtype=np.int64),
        np.full(n_bins, outcome[i], dtype=np.int64),
```

iii. The notes do not mention these repetitions; this is inferred from the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `load_data_structure` reads several fields that are never used downstream, including `tm`, `site`, `quality`, most of `meta`, and auxiliary trajectory fields such as `fn` and `NdroppedFrames`. The motion-energy normalizer also walks through several structural cases that are discarded once a flat list of trials is obtained.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    if name in clu:
        ...

for name in ['ts', 'frameTimes', 'fn', 'NdroppedFrames']:
    if name in traj:
        ...

if 'meta' in obj:
    meta = obj['meta']
    meta_out = {}
    for name in meta.keys():
        ...
```

iii. The notes do not discuss this directly. It is visible from the loader: multiple raw fields are materialized, but only a subset contribute to the final saved dataset.
