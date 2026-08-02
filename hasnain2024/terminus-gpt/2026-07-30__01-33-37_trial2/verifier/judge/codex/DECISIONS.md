# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every immediate subdirectory under `data/`, pairs `data_structure_<stem>.mat` with `motionEnergy_<stem>.mat` by filename stem, then processes only sessions that have both files. `data_structure` files are loaded with `h5py` only; `motionEnergy` files are loaded with `scipy.io.loadmat`. During the main loop, unreadable `data_structure` files and sessions missing usable neural fields are skipped entirely.

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

sessions = discover_sessions('data')
keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
for stem in keys:
    sess = process_session(stem, sessions[stem], edges, centers)
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as restricting conversion to sessions with all required outputs available: “Need neural + behavior + trajectory/motion-energy data for decoder outputs.” The trajectory also records explicit skip reasons such as `unreadable_data_structure` and `missing_trial_aligned_spikes`.

## 1-b. How are the data split into subjects?

i. The agent treats the subject as the prefix before the first underscore in the session stem, and later builds `subjects` and `subject_idx` from those parsed subject IDs.

ii.
```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])

subject, _ = infer_subject_session(stem)
if sess['subject'] not in subject_to_idx:
    subject_to_idx[sess['subject']] = len(subjects)
    subjects.append(sess['subject'])
```

iii. The notes describe this as filename-based subject parsing from names such as `motionEnergy_<subject>_<date>.mat`.

## 1-c. How are the data split into sessions?

i. Each unique filename stem after removing `data_structure_` / `motionEnergy_` is treated as one session. The date portion of the stem is not used separately beyond remaining part of the session ID.

ii.
```python
for f in subdir.glob('data_structure_*.mat'):
    stem = f.stem.replace('data_structure_', '')
    sessions.setdefault(stem, {})['data_structure'] = f
for f in subdir.glob('motionEnergy_*.mat'):
    stem = f.stem.replace('motionEnergy_', '')
    sessions.setdefault(stem, {})['motion_energy'] = f
```

iii. The notes repeatedly refer to “per-session” files and give examples like `JEB11_2022-05-10`, so the agent used the file stem as the session boundary.

## 1-d. How are the data split into trials?

i. Trials are defined by the raw `bp.Ntrials` count, then restricted to indices where the `valid_trials` mask is true. Every kept raw trial index becomes one converted trial for neural, input, and output arrays.

ii.
```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    ...
    return mask

for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
    unit_trials.append(counts.astype(np.float32))

idx = np.where(trial_mask)[0]
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes say one sanity check was that converted trial count should equal raw valid-trial count `(~early & ~ignore)`.

## 1-e. How are trials filtered based on quality controls?

i. The only explicit trial QC is exclusion of trials flagged `early` or `no` (ignore). No other trial-level balancing or task-specific inclusion rule is applied during conversion.

ii.
```python
for name in ['early', 'no']:
    if name in bp:
        arr = np.array(bp[name], dtype=float).reshape(-1)
        if arr.size == n:
            mask &= (arr == 0)
```

iii. The notes justify this with: “Exclude early and ignore trials: Matches methods and analysis code.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural tensor is derived directly from `obj['clu']['trialtm']` and `obj['clu']['trial']`, which are loaded from the raw `clu` struct. The code also loads `tm`, `site`, and `quality`, but only `trialtm` and `trial` are used for neural matrices.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    if name in clu:
        ...
        clu_out[name] = [np.array(deref(h, r)).reshape(-1) for r in ds[()].reshape(-1)]

trialtm = clu['trialtm']
trialid = clu['trial']
```

iii. The notes describe the neural source mapping as “`obj.clu.trialtm` + `obj.clu.trial` (+ unit metadata from `clu`)”.

## 2-b. How is the `neural` data processed?

i. For each unit, the agent bins the stored per-trial spike times into a fixed `[-2.5, 2.5]` window with 75 ms bins using `np.histogram`, and keeps the resulting raw spike counts as `float32`. It does not reproduce the reference pipeline’s explicit `alignSpikes`, 5 ms binning, Gaussian smoothing, or later 75 ms rebinning.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers

for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
    unit_trials.append(counts.astype(np.float32))
...
return [np.stack(x, axis=0) for x in neural_trials], np.array(kept_units, dtype=int)
```

iii. The notes justify this as: “Use spike counts on a fixed time grid: Raw neural data are stored as per-unit spike times by trial, so conversion will bin to a common grid across sessions.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by an approximate firing-rate threshold greater than 1 Hz, computed as total spikes divided by `(5 s window × number of valid trials)`. Sessions are then dropped if fewer than 10 units remain or fewer than 2 valid trials remain. The code does not use `quality` labels for filtering.

ii.
```python
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
...
if len(neural) < 2 or len(kept_units) < 10:
    print(f'  skip_reason: insufficient_neural trials={len(neural)} units={len(kept_units)}', flush=True)
    return None
```

iii. The notes justify this with: “Filter units by firing rate > 1 Hz and sessions with >=10 units: Matches methods and `removeLowFRClusters` logic.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code assumes the stored `clu.trialtm` values are already on the desired trial-relative axis and bins them directly in a symmetric `[-2.5, 2.5]` window. It does not explicitly subtract `bp.ev.goCue` or call a separate alignment routine.

ii.
```python
trialtm = clu['trialtm']
...
counts, _ = np.histogram(st[tr == raw_t], bins=edges)
...
edges, centers = build_time_grid()
```

iii. The notes say the alignment event should be go cue, but the code-level justification is implicit: it trusted the available `trialtm` representation instead of recreating alignment from raw spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 75 ms bins over a 5 s window, yielding 66 time bins per trial. No secondary temporal rebinning is applied; the neural data are created directly at 75 ms resolution.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers

'metadata': {
    'time_bin_size': 75.0,
```

iii. The notes cite the reference decoder scripts’ 75 ms binning and say this suggested rebinned behavioral/neural streams for decoding analyses.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not computed from a raw per-trial stored vector. The code synthesizes the input from the common time-grid centers; conceptually it is anchored to the go-cue alignment choice rather than read from a raw variable.

ii.
```python
edges, centers = build_time_grid()
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes justify this as a “Single input channel shared across trials/sessions” and “Represent decoder input as continuous time from go cue.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code creates a fixed set of bin centers from `-2.4625` s to `2.4125` s and copies that same 1 × 66 vector into every trial of every session.

ii.
```python
edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
centers = edges[:-1] + bin_size_s / 2
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes explicitly planned to “Represent decoder input as continuous time from go cue: Single input channel shared across trials/sessions.”

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The neural data and input share the same `edges/centers` time grid. The input is literally the bin-center vector used for the session-wide time axis.

ii.
```python
edges, centers = build_time_grid()
...
neural, kept_units = bin_spikes_for_session(obj, trial_mask, edges)
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes frame the full dataset as “Go-cue aligned” and use the shared time grid as the synchronization mechanism.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial raw behavioral arrays `bp['L']` and `bp['R']`.

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The notes map “`obj.bp.L`, `obj.bp.R`” to lick-direction output.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each valid trial, the code assigns `1` when `R > L` and `0` otherwise, then repeats that discrete label across every time bin of the trial.

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, lick_dir[i], dtype=np.int64),
```

iii. The notes characterize lick direction as a per-trial categorical label built from the behavioral structure.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the raw `bp['autowater']` array.

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. The notes say “context from `bp.autowater` / trial groups” and cite `getBlockNum_AltContextTask.m`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code performs a direct binary mapping: autowater-on trials become WC (`0`), and all other valid trials become DR (`1`). That label is then repeated across all time bins for the trial.

ii.
```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
...
np.full(n_bins, context[i], dtype=np.int64),
```

iii. The notes justify this with the block/context interpretation of `autowater`, stating that context identity is encoded in behavioral metadata.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the raw `bp['hit']` and `bp['miss']` trial arrays.

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The notes map “`obj.bp.hit`, `obj.bp.miss`” to outcome and specify incorrect = 0, correct = 1.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. For each valid trial, the code sets outcome to `1` when `hit > miss`, otherwise `0`, and repeats the result across all time bins.

ii.
```python
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, outcome[i], dtype=np.int64),
```

iii. The notes justify this as using the trial-wise hit/miss metadata after excluding early and ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-view trajectory data: `traj[0]['ts']`, `traj[0]['frameTimes']`, `traj[0]['featNames']`, plus trial go-cue times from `bp.ev.goCue`.

ii.
```python
side = traj_views[0]
side_names = [str(x) for x in side.get('featNames', [])]
go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)
...
ts_side = side['ts'][tr]
ft_side = side['frameTimes'][tr]
...
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
```

iii. The notes say tongue output should come from tongue-related `traj` features and mention a later bug fix to ensure the side view was used for tongue.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code chooses the first matching tongue feature name, interpolates its x/y coordinates onto the common 75 ms centers using `(frameTimes - 0.5) - goCue`, computes `np.gradient` of x and y, zero-fills NaN tongue velocities, and takes Euclidean speed. It does not preserve the continuous speed after thresholding.

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
...
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The notes justify this in broad terms as using named `traj.featNames`, aligning to go cue, and discretizing per session; the trajectory says an earlier one-view bug was fixed so that side-view tongue and bottom-view paw were used.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After constructing the full session-by-time tongue-speed matrix, the agent applies a single session-wide median split: values greater than the session median become `1`, otherwise `0`.

ii.
```python
tongue = np.stack(tongue, axis=0)
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The notes justify this directly from the task requirements: “Discretize continuous behavioral outputs per session at 50th percentile.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned by interpolating the side-camera trajectory onto the same `centers` vector used by the neural bins, after subtracting trial go-cue time and a hard-coded `0.5` s video offset.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
...
tongue.append(interp_feature_velocity(ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
```

iii. The notes say the intended alignment event is go cue and that tongue/paw traces should be aligned with raw timestamps, but they do not document the hard-coded `0.5` offset.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-view trajectory data: `traj[1]['ts']`, `traj[1]['frameTimes']`, `traj[1]['featNames']`, plus `bp.ev.goCue`.

ii.
```python
bottom = traj_views[1]
bottom_names = [str(x) for x in bottom.get('featNames', [])]
...
ts_bot = bottom['ts'][tr]
ft_bot = bottom['frameTimes'][tr]
...
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
```

iii. The notes justify paw output as coming from paw-related `traj` features and note the two-view bug fix.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code selects the first paw-like feature name in the bottom view, interpolates x/y coordinates to the common centers, computes gradients, fills NaN gradients with the per-trace median, subtracts that median baseline, and takes Euclidean speed.

ii.
```python
xv = np.gradient(x)
yv = np.gradient(y)
...
xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
xv = xv - np.nanmedian(xv)
yv = yv - np.nanmedian(yv)
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The notes justify this only at a high level as deriving a per-time-bin paw scalar from `traj` features and discretizing it by the session median.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The full session paw-speed matrix is thresholded at its session-wide median; values above the median are `1`, otherwise `0`.

ii.
```python
paw = np.stack(paw, axis=0)
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. The notes cite the decoder task’s required 50th-percentile thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity is interpolated onto the same `centers` vector as the neural bins using bottom-camera frame times and trial go-cue time, again with a hard-coded `0.5` s offset.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
...
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```

iii. The notes say the output should be go-cue aligned and time-varying but do not further justify the offset choice.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the raw `me.data` field loaded from each `motionEnergy_*.mat` file.

ii.
```python
def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']
...
me = load_motion_energy(files['motion_energy'])
me_data = normalize_motion_energy_data(me)
```

iii. The notes map “`me.data`” to motion-energy output and note that raw files also contain `me.moveThresh`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code normalizes a few possible MATLAB struct layouts into a Python list of per-trial traces, then rescales each trial trace to 66 samples by interpolating over normalized trace index from 0 to 1. It does not use raw frame times, video offset, or the reference code’s session-specific alignment routine.

ii.
```python
def rebin_variable_trace(values, n_bins):
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    xp = np.linspace(0, 1, arr.size)
    xnew = np.linspace(0, 1, n_bins)
    return np.interp(xnew, xp, arr).astype(np.float32)

me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
me_stack = np.stack(me_trials, axis=0)
```

iii. The notes justify the broad plan as “Rebin to common time grid, discretize by session median,” but do not document this normalized-index interpolation shortcut.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The rebinned session-wide motion-energy matrix is median-split across all trials and time bins: values above the session median become `1`, otherwise `0`.

ii.
```python
me_stack = np.stack(me_trials, axis=0)
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. The notes justify this from the decoder task’s instruction to threshold motion energy at the 50th percentile per session.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is approximate only: the code forces each per-trial motion-energy trace to 66 bins by interpolation over normalized sample index, not by aligning motion-energy timestamps to go cue and the neural bin centers.

ii.
```python
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
...
outputs.append(np.vstack([
    ...
    me_disc[i],
]))
```

iii. The notes planned a sanity check that rebinned motion energy should match raw `me.data` after applying the same time grid, but the implemented code did not follow the reference alignment routine.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses heuristic fallbacks rather than a unified reference-style missing-data path. Unreadable `data_structure` files are skipped; sessions without `trialtm`/`trial` or without a paw feature are skipped; empty motion-energy traces become zeros; empty or mismatched frame times fall back to synthetic 400 Hz frame times or `linspace`; missing tongue velocity samples become zero; missing paw samples are median-filled after differentiation; and missing tongue feature names yield all-zero tongue output.

ii.
```python
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
...
if tongue:
    xv[np.isnan(xv)] = 0
    yv[np.isnan(yv)] = 0
...
if tongue_idx is None:
    tongue.append(np.zeros(len(centers), dtype=np.float32))
```

iii. The notes justify some of this pragmatically: explicit skip reasons are logged, and a prior trajectory bug was fixed by using two camera views. There is no detailed reference-based justification for the zero/median fill choices.

## 11-a. What are the most time-consuming steps of the code?

i. The dominant work is session-by-session repeated histogramming and interpolation: `bin_spikes_for_session` loops over units and then over all valid trials; `build_traj_outputs` loops over valid trials and interpolates two video streams; and motion energy is rebinned trial-by-trial.

ii.
```python
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

iii. In `CONVERSION_NOTES.md`, Step 6 has placeholders for “Code inefficiencies identified” and “Code speedups added,” but the implemented structure shows these nested Python loops are the main cost.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit-by-trial spike histogram loop, the trial loop inside `build_traj_outputs`, the per-trial motion-energy rebinner, and the final output-construction loop could all have been further vectorized or batched.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)

for tr in trial_idx:
    ...

me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]

for i in range(len(idx)):
    outputs.append(np.vstack([...]))
```

iii. This follows directly from the code structure; the notes do not document any meaningful vectorization beyond basic NumPy use.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes valid trial indices inside per-unit loops, repeatedly interpolates traces one trial at a time for tongue, paw, and motion energy, and repeatedly tiles scalar per-trial labels across all 66 time bins instead of storing them once.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    ...

for tr in trial_idx:
    ...

np.full(n_bins, lick_dir[i], dtype=np.int64),
np.full(n_bins, context[i], dtype=np.int64),
np.full(n_bins, outcome[i], dtype=np.int64),
```

iii. This is an implementation fact rather than a separately argued decision; the notes do not claim otherwise.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes continuous tongue, paw, and motion-energy traces only to immediately throw them away after median thresholding; it loads metadata and unused `clu` fields such as `tm`, `site`, and `quality` without using them downstream; and it defines helper readers (`read_dataset_maybe_refs`, `read_char_ref_array`, `read_numeric_ref_array`) that are not used by the final conversion path.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    ...

tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)

def read_dataset_maybe_refs(h, ds):
    ...
def read_char_ref_array(h, ds):
    ...
def read_numeric_ref_array(h, ds):
    ...
```

iii. The notes only justify the thresholding step because the decoder task requires categorical outputs; they do not provide a downstream use for the discarded continuous traces.
