# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script discovers session files by globbing `/app/data/Ephys_Behavior/` and `/app/data/RandomizedDelay_Ephys_Behavior/` for `data_structure_*.mat`, then keeps only sessions whose keys appear in the hard-coded `PROBE_MAP`. For each session it attaches a matching `motionEnergy_<anm>_<date>.mat` file if present, and opens the main `.mat` file through either an HDF5 (`H5Session`) or MATLAB v5 (`V5Session`) wrapper selected by `open_session()`.

ii. 
```python
def get_session_list():
    sessions = []
    data_dirs = [
        '/app/data/Ephys_Behavior/',
        '/app/data/RandomizedDelay_Ephys_Behavior/',
    ]

    for dpath in data_dirs:
        files = sorted(glob.glob(os.path.join(dpath, 'data_structure_*.mat')))
        for fp in files:
            key = os.path.basename(fp).replace('data_structure_', '').replace('.mat', '')
            if key not in PROBE_MAP:
                continue
```

```python
def open_session(filepath):
    if is_h5_format(filepath):
        return H5Session(filepath)
    else:
        return V5Session(filepath)
```

iii. The justification appears in `CONVERSION_NOTES.md` Step 6: the agent wanted dual-format `.mat` support and to include both ephys task folders while still respecting the probe assignments from the reference loading scripts.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the session key prefix before the underscore, e.g. `JEB19_2023-04-19 -> JEB19`. The output `subjects` list is built in first-seen order during the main loop, and `subject_idx` stores each session’s index into that encounter-ordered list.

ii.
```python
anm = session_key.split('_')[0]
...
if anm not in all_subjects:
    all_subjects.append(anm)
subject_idx.append(all_subjects.index(anm))
```

iii. The notes repeatedly describe sessions as `<animal>_<date>` and report a final total of 14 subjects, so the filename/session key is the agent’s source of truth for subject identity.

## 1-c. How are the data split into sessions?

i. Each matched `data_structure_<anm>_<date>.mat` file is treated as one session. Sessions from the fixed-delay and randomized-delay folders are pooled into one list and then processed uniformly, with one output entry per session in `neural`, `input`, and `output`.

ii.
```python
sessions.append({
    'key': key,
    'data_path': fp,
    'me_path': me_fp,
    'probes': PROBE_MAP[key],
})
...
for sess_info in sessions:
    result = process_session(...)
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. In `CONVERSION_NOTES.md` Step 5 the agent explicitly decided to include both ephys datasets because they share the same recording setup and decoder target format.

## 1-d. How are the data split into trials?

i. The script takes `sess.get_ntrials()` as the session trial count, reads per-trial behavioral arrays from `obj.bp`, and uses raw trial indices `0..n_trials-1` as trial identities. For neural data it iterates over all trial numbers and uses the cluster field `trial` to assign spikes to individual trials; for video and motion energy it indexes the per-trial arrays with the same trial indices.

ii.
```python
n_trials = sess.get_ntrials()
...
for j in range(n_trials):
    trial_num = j + 1  # 1-indexed trial number
    spk_mask = trial == trial_num
```

```python
valid_trials = np.where(valid_mask)[0]  # 0-indexed
```

iii. The rationale is implicit in the code comments: Bpod fields are treated as already trialized, and spike/video arrays are assumed to carry trial membership directly, so no trial-boundary reconstruction is attempted.

## 1-e. How are trials filtered based on quality controls?

i. The script first drops photostimulation trials and early-lick trials. It also skips any session with fewer than 2 remaining trials. After neural preprocessing it removes trials whose retained units have zero activity in every bin, interpreting those as trials after the ephys recording had ended.

ii.
```python
valid_mask = (stim_enable == 0) & (early == 0)
valid_trials = np.where(valid_mask)[0]
...
trial_has_spikes = np.any(trialdat > 0, axis=(0, 1))
valid_trials = np.array([t for t in valid_trials if trial_has_spikes[t]])
```

iii. `CONVERSION_NOTES.md` Step 5 says stim and early trials were excluded by condition logic from the reference code, and trajectory steps 171/174/189 say the all-zero-trial removal was added after discovering sessions where behavior outlasted the ephys recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the spike-sorted cluster fields `quality`, `trialtm`, and `trial` inside `obj.clu`, together with the per-trial event time `bp.ev.goCue`.

ii.
```python
trialtm = np.array(self.f[clu['trialtm'][i, 0]]).flatten()
trial = np.array(self.f[clu['trial'][i, 0]]).flatten().astype(int)
...
goCue = sess.get_ev_field('goCue')
```

iii. The notes identify `alignSpikes.m`, `findClusters.m`, and `removeLowFRClusters.m` as the relevant reference functions, so the agent centered its neural pipeline on those raw spike and event fields.

## 2-b. How is the `neural` data processed?

i. For each retained unit and each trial, spike times are aligned by subtracting the trial’s go cue, binned with `np.histogram` into a fixed `[-2.5, 2.5]` s window at 10 ms resolution, converted to firing rate by dividing by `DT`, and smoothed with a causal Gaussian kernel implemented by `smooth_signal()`. Units from all selected probes are concatenated.

ii.
```python
DT = 1.0/100  # 10 ms bins
...
counts, _ = np.histogram(spk_times, bins=EDGES)
rate = counts.astype(np.float32) / DT
trialdat[:, i, j] = smooth_signal(rate)
```

```python
def make_causal_gaussian_kernel(N):
    kern = gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0
    kern = kern / kern.sum()
    return kern
```

iii. `CONVERSION_NOTES.md` Steps 1 and 6 say the agent intended to match `WorkingWithDataObjs.m`, `getSeq.m`, and `mySmooth.m`, and therefore chose 10 ms bins and causal Gaussian smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters units in three ways: it drops clusters with manual quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}`, then removes units whose mean firing rate over valid trials is not greater than 1 Hz, and finally skips sessions with fewer than 10 units either before or after the FR filter.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10
...
if quality in EXCLUDE_QUALITIES:
    continue
...
mean_fr = np.mean(np.mean(trialdat[:, :, valid_trials], axis=2), axis=0)
keep_units = mean_fr > LOW_FR
```

iii. The notes cite the paper’s 1 Hz FR rule and session inclusion rule (`>= 10 units`) and say the quality labels come from `findClusters.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is done by subtracting the per-trial `goCue` time from each spike’s `trialtm` value before binning.

ii.
```python
spk_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(spk_times, bins=EDGES)
```

iii. The code comment explicitly says this is meant to match `alignSpikes.m: trialtm_aligned = trialtm - goCue(trial)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 ms bins (`DT = 1/100`) over `[-2.5, 2.5]` s, giving 500 time bins per trial. No later rebinning is applied; this fixed grid is the saved resolution.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0/100  # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME)
```

iii. The notes explicitly justify this by citing `dt=1/100` from the reference code, and the final notes report `500` time bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The saved input is not read from a raw per-trial variable. It is a synthetic time axis (`TIME`) defined around the alignment event `goCue`, with the conceptual raw dependency being that all trials are aligned to go cue onset.

ii.
```python
ALIGN_EVENT = 'goCue'
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME = EDGES[:-1] + DT / 2
...
input_trials.append(TIME.reshape(1, -1).astype(np.float32))
```

iii. `CONVERSION_NOTES.md` Step 5 describes `time_from_go_cue` as `obj.time = edges + dt/2` and treats it as a decoder input defined by the alignment grid rather than a direct raw field.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes a single vector of bin centers from the global `EDGES` array and reuses it for every trial in every session.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME = EDGES[:-1] + DT / 2
...
input_trials.append(TIME.reshape(1, -1).astype(np.float32))
```

iii. The notes describe this as a continuous, time-varying input directly tied to the chosen neural binning grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is aligned by construction: it is the exact same `TIME` grid used for neural spike binning and for interpolating the video-derived outputs.

ii.
```python
counts, _ = np.histogram(spk_times, bins=EDGES)
...
input_trials.append(TIME.reshape(1, -1).astype(np.float32))
```

iii. The code and notes both treat the common time base as the alignment mechanism, so no separate alignment step is applied to the input stream.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from behavioral trial flags `hit`, `miss`, `no`, `R`, and `L`.

ii.
```python
hit = sess.get_bp_field('hit')
miss = sess.get_bp_field('miss')
no = sess.get_bp_field('no')
R = sess.get_bp_field('R')
L = sess.get_bp_field('L')
```

iii. The in-code comments explain that `R` and `L` define the trial side, while `hit`, `miss`, and `no` determine whether the animal responded correctly, incorrectly, or not at all.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI maps each retained trial to `2` by default (`none`), then assigns `right` or `left` based on the combination of instructed side and outcome: hit means lick the instructed side, miss means lick the opposite side, and `no` means no lick. The per-trial class is then broadcast across all time bins of the saved output tensor.

ii.
```python
lick_dir = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    if no[ti] == 1:
        lick_dir[vi] = 2
    elif R[ti] == 1 and hit[ti] == 1:
        lick_dir[vi] = 1
    elif R[ti] == 1 and miss[ti] == 1:
        lick_dir[vi] = 0
    elif L[ti] == 1 and hit[ti] == 1:
        lick_dir[vi] = 0
    elif L[ti] == 1 and miss[ti] == 1:
        lick_dir[vi] = 1
```

iii. The surrounding comments state that `R/L` indicate the correct side and `hit/miss/no` reveal the actual lick behavior, so lick direction must be reconstructed from both.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` field in `obj.bp`.

ii.
```python
autowater = sess.get_bp_field('autowater')
```

iii. In the notes the agent maps `autowater=1` to the water-cued context and `autowater=0` to delayed response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly relabels `autowater` into two classes: `0` for WC and `1` for DR, then broadcasts that per-trial label across time.

ii.
```python
context = np.array([1 if autowater[ti] == 0 else 0 for ti in valid_trials], dtype=int)
...
out[1, :] = context[vi]
```

iii. The notes say this mapping follows the decoder specification: WC is class `0`, DR is class `1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit`, `miss`, and `no` trial flags.

ii.
```python
hit = sess.get_bp_field('hit')
miss = sess.get_bp_field('miss')
no = sess.get_bp_field('no')
```

iii. The agent’s notes treat the decoder as needing all trial types, so the explicit `no` field is kept to represent ignore trials rather than inferring them indirectly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script maps hit to `1` (`correct`), miss to `0` (`incorrect`), and `no` to `2` (`ignore`), then broadcasts the per-trial label across time.

ii.
```python
outcome = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    if hit[ti] == 1:
        outcome[vi] = 1
    elif miss[ti] == 1:
        outcome[vi] = 0
    elif no[ti] == 1:
        outcome[vi] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly states the class order `0=incorrect, 1=correct, 2=ignore`.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from side-camera trajectory data only: `sess.get_traj_data(0, ti)` supplies `ts` and `frameTimes`, and the script selects the feature named `tongue`. Alignment additionally depends on `goCue` and the session-wide video offset from `sglx.bitcode.bitstart`, `sglx.fs`, and `bp.ev.bitStart`.

ii.
```python
side_feats, _, _, _ = sess.get_traj_data(0, 0)
...
if fn == 'tongue':
    tongue_idx_side = fi
...
_, ts, frame_times, is_valid = sess.get_traj_data(0, ti)
aligned_times = frame_times - vidshift - goCue[ti]
```

iii. The notes say tongue velocity should come from DLC tongue position, and trajectory step 110 says the agent focused on tongue visibility problems in the tracked video stream.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each valid trial, the AI reads side-camera tongue x/y traces, marks frames visible when coordinates are not NaN, interpolates visibility to the neural time grid, fills missing positions with the mean visible position, interpolates the filled coordinates to the neural grid, computes velocity with `np.gradient`, takes Euclidean speed, and then forces speed to `0` wherever the tongue is not visible.

ii.
```python
tx = ts[:, 0, tongue_idx_side].astype(float)
ty = ts[:, 1, tongue_idx_side].astype(float)
vis_raw = ~(np.isnan(tx) | np.isnan(ty))
...
tx_filled[~vis_raw] = baseline_x
ty_filled[~vis_raw] = baseline_y
tx_i = np.interp(taxis, aligned_times, tx_filled)
ty_i = np.interp(taxis, aligned_times, ty_filled)
vx = np.gradient(tx_i)
vy = np.gradient(ty_i)
speed = np.sqrt(vx**2 + vy**2)
speed[~vis_interp] = 0
```

iii. The code comments say this is meant to match `setTongueBaselinePosition` and `findVelocity.m`, and trajectory steps 107, 110, and 116 show the agent specifically adjusting NaN handling and the “0 when not visible” rule.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI gathers all visible tongue-speed values within a session, computes their 50th percentile, assigns `0` below threshold and `1` at or above threshold, and uses `2` when the tongue is not visible.

ii.
```python
tongue_vel_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)
visible_tongue_vals = tongue_vel[tongue_visible & ~np.isnan(tongue_vel)]
if len(visible_tongue_vals) > 0:
    thresh_tongue = np.percentile(visible_tongue_vals, 50)
    ...
    tongue_vel_disc[valid, vi] = (tongue_vel[valid, vi] >= thresh_tongue).astype(int)
    tongue_vel_disc[~vis, vi] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 says the 50th-percentile split and `not visible` class were taken from the decoder specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The agent aligns side-camera frame times by subtracting a session-wide video offset and each trial’s `goCue`, then interpolates both visibility and position traces onto the neural `TIME` grid before differentiating and thresholding.

ii.
```python
vidshift = sess.get_video_offset()
...
aligned_times = frame_times - vidshift - goCue[ti]
vis_interp = np.interp(taxis, aligned_times, vis_raw.astype(float)) > 0.5
tx_i = np.interp(taxis, aligned_times, tx_filled)
ty_i = np.interp(taxis, aligned_times, ty_filled)
```

iii. The notes explicitly say trajectory signals were “interpolated to neural time axis,” and the code treats that interpolation as the alignment step.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera trajectory data. The script finds every bottom-view feature whose name contains `'paw'`, then processes each of those features.

ii.
```python
bottom_feats, _, _, _ = sess.get_traj_data(1, 0)
paw_indices_bottom = [fi for fi, fn in enumerate(bottom_feats) if 'paw' in fn]
...
_, ts, frame_times, is_valid = sess.get_traj_data(1, ti)
```

iii. `CONVERSION_NOTES.md` Step 5 says paw velocity comes from bottom-camera paw features, and the implementation broadens that to all paw-labeled bottom-camera tracks.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each selected paw feature, the AI treats non-NaN x values as visible, fills missing positions by nearest interpolation in frame index, interpolates x/y to the neural time grid, computes gradients, subtracts a baseline velocity estimated by the median frame-to-frame difference, converts to Euclidean speed, and averages the resulting speed traces across all bottom-camera paw features. Visibility is the union across paws.

ii.
```python
valid_idx = np.where(vis_raw)[0]
px_filled = np.interp(np.arange(len(px)), valid_idx, px[valid_idx])
py_filled = np.interp(np.arange(len(py)), valid_idx, py[valid_idx])
px_i = np.interp(taxis, aligned_times, px_filled)
py_i = np.interp(taxis, aligned_times, py_filled)
vx = np.gradient(px_i)
vy = np.gradient(py_i)
base_vx = np.median(np.diff(px_i))
base_vy = np.median(np.diff(py_i))
speed = np.sqrt(vx**2 + vy**2)
...
paw_vel[:, vi] = np.mean(paw_speeds, axis=0)
```

iii. The inline comments say this is intended to match `findPosition.m` and `findVelocity.m` for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Visible paw-speed samples within a session are median-split, with `0` below threshold, `1` at or above threshold, and `2` for bins marked not visible.

ii.
```python
paw_vel_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)
visible_paw_vals = paw_vel[paw_visible & ~np.isnan(paw_vel)]
if len(visible_paw_vals) > 0:
    thresh_paw = np.percentile(visible_paw_vals, 50)
    ...
    paw_vel_disc[valid, vi] = (paw_vel[valid, vi] >= thresh_paw).astype(int)
    paw_vel_disc[~vis, vi] = 2
```

iii. The notes say the 50th-percentile threshold and `not visible` class were imposed by the decoder output specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are shifted by the session video offset and the trial’s go cue, then paw positions are interpolated directly onto the neural time base before velocity is computed.

ii.
```python
aligned_times = frame_times - vidshift - goCue[ti]
px_i = np.interp(taxis, aligned_times, px_filled)
py_i = np.interp(taxis, aligned_times, py_filled)
```

iii. As with tongue velocity, the notes say the camera stream should be shifted and interpolated to the neural timeline.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from a separate `motionEnergy_<anm>_<date>.mat` file per session. The loader supports HDF5 v7.3 files, MATLAB v5 structs, and plain cell-array layouts.

ii.
```python
def load_motion_energy(me_filepath):
    try:
        if is_h5_format(me_filepath):
            ...
        else:
            d = sio.loadmat(me_filepath, squeeze_me=False)
            me_raw = d['me']
            ...
        return me_data, thresh
```

iii. `CONVERSION_NOTES.md` Step 6 and trajectory step 208 show the agent explicitly focused on supporting multiple motion-energy file layouts so all sessions could be loaded.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each valid trial, the script reads the motion-energy trace, uses side-camera `frameTimes` for timing, aligns them by subtracting video offset and go cue, interpolates the motion-energy values to the neural time grid, and nearest-fills any NaNs left in the interpolated trace.

ii.
```python
me_trial = np.array(me_data[ti]).flatten().astype(float)
_, _, frame_times, is_valid = sess.get_traj_data(0, ti)
aligned_times = frame_times - vidshift - goCue[ti]
me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])
nan_mask = np.isnan(me_interp)
if np.any(~nan_mask) and np.any(nan_mask):
    valid_idx = np.where(~nan_mask)[0]
    me_interp = np.interp(np.arange(len(me_interp)), valid_idx, me_interp[valid_idx])
```

iii. The comments say this is meant to match `loadMotionEnergy.m`, and the trajectory shows repeated debugging of motion-energy loading and alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The agent takes all non-NaN aligned motion-energy values within a session, splits them at the 50th percentile, and uses `2` only for bins with no valid motion-energy value (`no video`).

ii.
```python
me_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)
valid_me = ~np.isnan(me_data_aligned)
if np.any(valid_me) and me_available:
    me_vals = me_data_aligned[valid_me]
    thresh_me = np.percentile(me_vals, 50)
    ...
    me_disc[valid_t, vi] = (me_data_aligned[valid_t, vi] >= thresh_me).astype(int)
    me_disc[~valid_t, vi] = 2
```

iii. The notes explicitly map motion energy to the decoder’s per-session 50th-percentile threshold with a `no video` fallback class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy frame times are aligned using the same video offset and go-cue subtraction as the kinematic outputs, and the values are then interpolated onto the neural `TIME` axis.

ii.
```python
aligned_times = frame_times - vidshift - goCue[ti]
me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])
me_data_aligned[:, vi] = me_interp
```

iii. The notes say motion energy follows the same video-to-neural alignment strategy as the other camera-derived signals.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles irregularities by permissive loading and fallback behavior rather than strict exclusion. It supports multiple `.mat` formats, defaults the video offset to `0.5` s on failure, suppresses many per-trial exceptions with `except: pass`, fills missing tongue positions with a baseline position, fills missing paw positions and motion-energy samples by interpolation, uses category `2` for not-visible/no-video bins, and drops all-zero neural trials after FR filtering.

ii.
```python
def get_video_offset(self):
    try:
        ...
        return vidshift
    except Exception:
        return 0.5  # default
```

```python
tx_filled[~vis_raw] = baseline_x
...
px_filled = np.interp(np.arange(len(px)), valid_idx, px[valid_idx])
...
except Exception:
    pass
```

iii. `CONVERSION_NOTES.md` Step 10 and trajectory steps 171 and 208 show the agent prioritized keeping sessions/trials usable by adding fallbacks and post hoc cleanup when data omissions were found.

## 11-a. What are the most time-consuming steps of the code?

i. The code’s heaviest work is the nested trial-by-unit neural loop and the per-trial video/motion-energy interpolation loops. The neural path bins and smooths every unit separately for every trial, and the video path repeatedly loads trajectory data and interpolates traces for every valid trial.

ii.
```python
for i, unit in enumerate(all_units):
    ...
    for j in range(n_trials):
        ...
        counts, _ = np.histogram(spk_times, bins=EDGES)
        trialdat[:, i, j] = smooth_signal(rate)
```

```python
for vi, ti in enumerate(valid_trials):
    ...
    tx_i = np.interp(taxis, aligned_times, tx_filled)
    ...
    px_i = np.interp(taxis, aligned_times, px_filled)
    ...
    me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])
```

iii. The notes do not explicitly benchmark these sections, but the runtime structure of the script makes these loops the obvious hotspots.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are written in scalar/per-trial style and could have been vectorized: the unit-by-trial spike binning loop, the per-trial label construction loops for lick direction and outcome, the per-trial discretization loops for tongue/paw/motion energy, and some repeated interpolation/visibility calculations inside the video loops.

ii.
```python
for i, unit in enumerate(all_units):
    for j in range(n_trials):
        ...
```

```python
for vi, ti in enumerate(valid_trials):
    ...
for vi in range(len(valid_trials)):
    vis = tongue_visible[:, vi]
    ...
```

iii. There is no explicit justification in the notes for keeping these loops; this is an implicit implementation choice visible in the script itself.

## 11-c. What processing does the code repeat multiple times?

i. The script repeats several operations on the same trial-aligned time bases: it interpolates tongue visibility, tongue x, tongue y, each paw x/y trace, paw visibility, and motion energy separately; it reopens trajectory data per trial and per view; and it computes an initial `lick_dir` pass that is immediately overwritten by a second “redo” pass.

ii.
```python
vis_interp = np.interp(taxis, aligned_times, vis_raw.astype(float)) > 0.5
tx_i = np.interp(taxis, aligned_times, tx_filled)
ty_i = np.interp(taxis, aligned_times, ty_filled)
```

```python
lick_dir = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    ...

# Re-do lick direction properly:
lick_dir = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    ...
```

iii. The trajectory shows the agent iterating on these sections during debugging, but the final script still contains repeated work rather than consolidating it.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and retains several intermediates only to discard them before saving: continuous `tongue_vel`, `paw_vel`, and `me_data_aligned` are all reduced to discrete categories and never written out; the first lick-direction loop is dead work because it is overwritten; and the plotting machinery and `show-processing` path exist even though they are not part of the saved dataset.

ii.
```python
tongue_vel = np.full((N_TIMEBINS, len(valid_trials)), np.nan, dtype=np.float32)
paw_vel = np.full((N_TIMEBINS, len(valid_trials)), np.nan, dtype=np.float32)
me_data_aligned = np.full((N_TIMEBINS, len(valid_trials)), np.nan, dtype=np.float32)
...
output_trials.append(out)
```

```python
# Re-do lick direction properly:
lick_dir = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    ...
```

iii. The notes do not call this out explicitly; it is visible from the final script structure and from the presence of debugging/plotting helpers that do not affect the saved pickle.
