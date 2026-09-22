# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded the session list in `SESSION_META`, then iterated session-by-session. For each session it built paths to `data_structure_<animal>_<date>.mat` and `motionEnergy_<animal>_<date>.mat`, used `SessionData` to open the main MATLAB file in either v7.3 (`h5py`) or v5 (`scipy.io.loadmat`) format, and loaded motion energy from the sidecar file with a custom loader that handles multiple layouts.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]
...
for animal, date, probes, data_dir in SESSION_META:
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{animal}_{date}.mat')
    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{animal}_{date}.mat')
    sess = SessionData(data_path)
```

```python
def detect_mat_version(path):
    with open(path, 'rb') as f:
        header = f.read(16).decode('ascii', errors='replace')
    return '7.3' if '7.3' in header else '5.0'
```

iii. In trajectory steps 57, 75, 77, and 84, the agent said it was following the loading scripts for the session list and probes, and that it needed a unified reader because some sessions are v7.3 while others are v5, and motion-energy files occur in three layouts.

## 1-b. How are the data split into subjects?

i. The AI treated the first field of each `SESSION_META` tuple as the subject id and accumulated unique subjects in encounter order. `subject_idx` stores, for each session, the index of that animal in `subjects_set`.

ii.
```python
for animal, date, probes, data_dir in SESSION_META:
    ...
    if animal not in subjects_set:
        subjects_set.append(animal)
    all_subject_idx.append(subjects_set.index(animal))
...
'subjects': subjects_set,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. In trajectory steps 39 and 57, the agent reasoned about the dataset in terms of animal ids from the loading scripts and counted sessions by mouse, so the filename/session metadata was its source of truth for subject identity.

## 1-c. How are the data split into sessions?

i. The AI treated each entry of `SESSION_META` as one session and each session as one element in the outer lists of `neural`, `input`, and `output`. The source file is one `data_structure` file per `(animal, date, data_dir)` combination.

ii.
```python
for animal, date, probes, data_dir in SESSION_META:
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. In trajectory steps 39 and 57, the agent explicitly enumerated the fixed-delay and randomized-delay sessions from the authors’ loading scripts and said it would process those sessions uniformly.

## 1-d. How are the data split into trials?

i. The AI took `bp.Ntrials` as the session trial count, sliced all trial-wise behavioral arrays to that length, and represented each kept trial by its integer index in `valid_idx`. Spike data stay associated with trials via the per-spike `trial` field, and video/motion-energy data are processed per trial index.

ii.
```python
n_trials_total = sess.get_n_trials()
hit = sess.get_bp_field('hit')[:n_trials_total]
...
valid_idx = np.where(valid_mask)[0]
```

```python
for i, ti in enumerate(valid_idx):
    session_neural.append(trialdat[:, :, ti].T)
    session_input.append(time_axis.reshape(1, -1).copy())
    ...
    session_output.append(out)
```

iii. In trajectory step 59, the agent described the pipeline as operating per trial after trial filtering, with neural, video, and motion-energy signals aligned and resampled on a shared grid.

## 1-e. How are trials filtered based on quality controls?

i. The AI filtered out photostimulation and early-lick trials, required a valid outcome flag (`hit`, `miss`, or `no`), and later removed kept trials whose neural tensor was all zeros after unit filtering, interpreting those as post-recording trials. It also skipped whole sessions with fewer than two remaining trials.

ii.
```python
valid_mask = (stim_enable == 0) & (early == 0) & ((hit == 1) | (miss == 1) | (no == 1))
valid_idx = np.where(valid_mask)[0]
...
has_spikes = np.array([np.any(trialdat[:, :, ti] != 0) for ti in valid_idx])
valid_idx = valid_idx[has_spikes]
if len(valid_idx) < 2:
    ...
```

iii. In trajectory steps 57, 109, and 112, the agent said it would exclude stimulation and early-lick trials, then added a later filter for all-zero neural trials after verifier warnings, justifying those as trials after the ephys recording had stopped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from each unit’s `trialtm` and `trial` arrays, together with per-trial `goCue` times for alignment. It also read `quality` for unit filtering and `tm` even though `tm` is not used in the later computation.

ii.
```python
units.append({
    'tm': tm, 'trial': trial, 'trialtm': trialtm, 'quality': quality
})
...
aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
```

iii. In trajectory steps 57 and 59, the agent summarized the neural pipeline as selecting clusters by quality, aligning spikes to `goCue`, binning them at 5 ms, and filtering low-rate units.

## 2-b. How is the `neural` data processed?

i. The AI binned aligned spikes into 5 ms bins, converted counts to Hz by dividing by `DT`, and then applied its own implementation of the MATLAB `mySmooth.m` causal Gaussian smoother with a 15-sample window and `reflect` boundary handling.

ii.
```python
counts = np.histogram(aligned[mask], bins=edges)[0]
fr = counts.astype(float) / DT
trialdat[:, i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, 'reflect')
```

```python
def causal_gaussian_smooth(x, window, bctype='reflect'):
    ...
    kern[:int(window // 2)] = 0  # causal
    kern = kern / kern.sum()
    ...
```

iii. In trajectory steps 27, 50, 57, and 59, the agent explicitly decided to match `mySmooth.m`, describing the neural processing as causal Gaussian smoothing with a 15-sample window after 5 ms spike binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first filtered units by manual quality label, excluding only `garbage`, `gabrga`, `noisy`, and `real?`, and also dropping empty quality strings. It then removed units whose mean firing rate across all bins and trials was not greater than 1 Hz. Entire sessions were skipped if fewer than 10 units remained after either the quality filter or firing-rate filter.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10
...
units = [u for u in units
         if u['quality'].lower() not in {q.lower() for q in EXCLUDED_QUALITIES}
         and u['quality'] != '']
...
mean_frs = np.mean(trialdat, axis=(0, 2))
fr_mask = mean_frs > LOW_FR
```

iii. In trajectory steps 57 and 59, the agent justified this as following the paper’s `>1 Hz` criterion and the loading-code `quality='all'` behavior, which it interpreted as excluding the four canonical bad labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go-cue time of the corresponding trial from `trialtm`, so all subsequent binning is in seconds relative to go cue onset.

ii.
```python
aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
```

iii. In trajectory steps 57 and 59, the agent said it would use `goCue` as the common alignment event for neural and behavioral streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI used 5 ms bins over a window from -2.5 s to +2.5 s from go cue onset, giving 1000 bins per trial. Neural data are created directly on this grid; there is no second rebinning pass.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200
...
n_bins = int(round((TMAX - TMIN) / DT))
edges = np.linspace(TMIN, TMAX, n_bins + 1)
time_axis = edges[:-1] + DT / 2
```

iii. In trajectory steps 57 and 59, the agent explicitly chose the paper/default-parameter grid: `dt = 1/200`, `tmin = -2.5`, `tmax = 2.5`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The AI did not derive this from a stored raw-data vector. It created a synthetic time axis from `TMIN`, `TMAX`, and `DT`, intended to represent seconds from go cue onset.

ii.
```python
n_bins = int(round((TMAX - TMIN) / DT))
edges = np.linspace(TMIN, TMAX, n_bins + 1)
time_axis = edges[:-1] + DT / 2
```

iii. In trajectory step 59, the agent described the decoder input as time from go cue on the same 5 ms grid used for neural data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The only processing is constructing the bin-center vector and copying it into every trial as a `(1, n_time)` array.

ii.
```python
time_axis = edges[:-1] + DT / 2
...
session_input.append(time_axis.reshape(1, -1).copy())
```

iii. In trajectory step 59, the agent treated this as a direct grid definition rather than a quantity estimated from raw measurements.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the same `time_axis` used to define the neural histogram edges, so each input bin center corresponds to the same 5 ms interval that the spikes are counted into.

ii.
```python
edges = np.linspace(TMIN, TMAX, n_bins + 1)
time_axis = edges[:-1] + DT / 2
...
counts = np.histogram(aligned[mask], bins=edges)[0]
...
session_input.append(time_axis.reshape(1, -1).copy())
```

iii. In trajectory step 59, the agent described video and motion-energy streams as being interpolated onto this same neural time grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derived lick direction from the trial-level `hit`, `miss`, `R`, and `L` flags, and kept `no` indirectly because untouched trials remain in the default `none` class.

ii.
```python
hit = sess.get_bp_field('hit')[:n_trials_total]
miss = sess.get_bp_field('miss')[:n_trials_total]
R = sess.get_bp_field('R')[:n_trials_total]
L = sess.get_bp_field('L')[:n_trials_total]
```

iii. In trajectory step 59, the agent said it needed to work out whether `R/L` encoded the correct side and use `hit/miss/no` to derive lick direction, including ignore trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI initialized all kept trials as `none` (code 2), then set hits on `R` trials to right and hits on `L` trials to left. Misses are assigned the opposite side of the instructed port.

ii.
```python
lick_dir = np.full(n_valid, 2, dtype=np.int64)
for i, ti in enumerate(valid_idx):
    if hit[ti] == 1 and R[ti] == 1:
        lick_dir[i] = 1
    elif hit[ti] == 1 and L[ti] == 1:
        lick_dir[i] = 0
    elif miss[ti] == 1 and R[ti] == 1:
        lick_dir[i] = 0
    elif miss[ti] == 1 and L[ti] == 1:
        lick_dir[i] = 1
```

iii. In trajectory steps 57 and 59, the agent justified this as representing lick direction, including a separate “no lick/ignore” state.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI derived behavioral context from `autowater`.

ii.
```python
autowater = sess.get_bp_field('autowater')[:n_trials_total]
```

iii. In trajectory steps 39 and 57, the agent used the presence of autowater trials to reason about which sessions contained the water-cued context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly relabeled `autowater == 1` as WC (`0`) and all other trials as DR (`1`).

ii.
```python
context = np.where(autowater[valid_idx] == 1, 0, 1).astype(np.int64)
```

iii. In trajectory step 57, the agent summarized this as “autowater vs delayed reward.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derived outcome from `hit` and `miss`, while also reading `no` as part of the trial filter.

ii.
```python
hit = sess.get_bp_field('hit')[:n_trials_total]
miss = sess.get_bp_field('miss')[:n_trials_total]
no = sess.get_bp_field('no')[:n_trials_total]
```

iii. In trajectory step 57, the agent described per-trial outcome as “hit/miss/no.”

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI initialized all kept trials as ignore (`2`), changed hits to correct (`1`), and changed misses to incorrect (`0`).

ii.
```python
outcome = np.full(n_valid, 2, dtype=np.int64)
for i, ti in enumerate(valid_idx):
    if hit[ti] == 1:
        outcome[i] = 1
    elif miss[ti] == 1:
        outcome[i] = 0
```

iii. In trajectory steps 57 and 59, the agent justified keeping ignore trials as their own decoder class rather than discarding them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derived tongue velocity from the bottom camera DLC trajectories only, specifically feature index `0` (`top_tongue`) from `ts`, plus that trial’s `frameTimes`, session-level video offset, and per-trial `goCue`.

ii.
```python
n_traj_trials = sess.get_n_traj_trials(1)  # bottom cam
for trix in range(min(n_trials_total, n_traj_trials)):
    ts, ft = sess.get_traj_trial(1, trix)  # bottom cam
    ...
    tv, tvis = compute_velocity_from_dlc(ts, aligned_ft, 0, time_axis)
```

iii. In trajectory step 59, the agent explicitly said it was “leaning toward using the bottom cam’s `top_tongue` feature for velocity” after comparing visibility between views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computed frame-to-frame speed from raw x/y displacements divided by a fixed `VIDEO_FPS`, marked a velocity sample as visible only if both adjacent DLC confidence values were at least `0.6`, linearly interpolated both velocity and visibility to the neural time grid, zero-filled interpolated NaNs, and later median-split the visible samples across the session.

ii.
```python
x = ts[:, 0, feat_idx]
y = ts[:, 1, feat_idx]
conf = ts[:, 2, feat_idx]
...
dx = np.diff(x)
dy = np.diff(y)
vel = np.sqrt(dx**2 + dy**2) / dt_video
vis = (conf[:-1] >= CONF_THRESH) & (conf[1:] >= CONF_THRESH)
ft_vel = (frame_times_aligned[:-1] + frame_times_aligned[1:]) / 2
vel_interp = np.interp(time_axis, ft_vel, vel, left=np.nan, right=np.nan)
vis_interp = np.interp(time_axis, ft_vel, vis.astype(float), left=0, right=0) >= 0.5
vis_interp[np.isnan(vel_interp)] = False
vel_interp = np.nan_to_num(vel_interp, nan=0.0)
```

iii. In trajectory steps 57 and 59, the agent justified this by saying it would compute the derivative of position, treat low-confidence frames as “not visible,” and interpolate onto the neural time grid.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI discretized tongue velocity within each session by taking the 50th percentile over bins marked visible and assigning `0` below threshold, `1` at or above threshold, and `2` to bins marked invalid/not visible.

ii.
```python
def discretize_per_session(values, valid_mask):
    result = np.full_like(values, 2, dtype=np.int64)
    all_valid = values[valid_mask]
    if len(all_valid) > 0:
        thresh = np.percentile(all_valid, 50)
        result[valid_mask] = np.where(all_valid >= thresh, 1, 0)
    return result
...
tongue_disc = discretize_per_session(tv_valid, tvis_valid)
```

iii. In trajectory step 59, the agent said it would discretize visible timepoints against the session-specific 50th percentile and reserve a third state for “not visible.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI aligned tongue velocity by subtracting the session video offset and that trial’s `goCue` from each frame time, then interpolating the frame-wise velocity trace onto the shared neural `time_axis`.

ii.
```python
vidshift = sess.get_video_offset()
...
aligned_ft = ft - vidshift - go_cue[trix]
tv, tvis = compute_velocity_from_dlc(ts, aligned_ft, 0, time_axis)
```

iii. In trajectory step 59, the agent described this alignment exactly: shift the video timestamps by the bitcode-derived offset, re-center on go cue, and interpolate onto the neural grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI derived paw velocity from two bottom-camera DLC features, feature indices `4` (`top_paw`) and `5` (`bottom_paw`), again using frame times, video offset, and go-cue alignment.

ii.
```python
pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
pv_bot, pvis_bot = compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)
```

iii. In trajectory step 59, the agent said it would use bottom-camera paw keypoints and, after inspecting confidences, decided to combine the two paw traces instead of taking only one.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature the AI used the same derivative-and-interpolate routine as for tongue velocity. It then combined the two paw features by taking the maximum velocity when both were visible, otherwise whichever one was visible, and zero when neither was visible.

ii.
```python
pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
pv_bot, pvis_bot = compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)
combined_vis = pvis_top | pvis_bot
combined_vel = np.where(
    pvis_top & pvis_bot, np.maximum(pv_top, pv_bot),
    np.where(pvis_top, pv_top, np.where(pvis_bot, pv_bot, 0.0))
)
paw_vel[:, trix] = combined_vel
paw_vis[:, trix] = combined_vis
```

iii. In trajectory steps 57 and 59, the agent justified this as using bottom-camera paw tracking and turning low-confidence periods into “not visible” states on the shared time grid.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is discretized exactly like tongue velocity: per-session 50th percentile over visible bins, with `2` for invalid/not visible bins.

ii.
```python
paw_disc = discretize_per_session(pv_valid, pvis_valid)
```

iii. In trajectory step 59, the agent said it would apply the same session-median thresholding logic to paw velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same alignment as tongue velocity: subtract video offset and per-trial go cue from the frame times, then interpolate to the shared 5 ms neural grid.

ii.
```python
aligned_ft = ft - vidshift - go_cue[trix]
pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
pv_bot, pvis_bot = compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)
```

iii. In trajectory step 59, the agent said paw features would be aligned and resampled in the same way as tongue features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derived motion energy from the separate per-session `motionEnergy_<animal>_<date>.mat` file and used side-camera `frameTimes` from `obj.traj` for alignment.

ii.
```python
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{animal}_{date}.mat')
...
me_data = load_motion_energy(me_path)
...
_, ft = sess.get_traj_trial(0, trix)
```

iii. In trajectory steps 27, 49, and 84, the agent said motion energy should come from the sidecar motion-energy file and noted that the files occur in several layouts that must be unwrapped.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolated each motion-energy trace from camera frame times to the neural `time_axis`, trimmed length mismatches by `min_len`, and then filled internal/external NaNs by nearest-value interpolation before median-splitting across the session.

ii.
```python
me_aligned[:, trix] = np.interp(
    time_axis, aligned_ft[:min_len], me_trial[:min_len],
    left=np.nan, right=np.nan
)
...
if np.any(nans) and not np.all(nans):
    valid = ~nans
    me_aligned[:, trix] = np.interp(
        np.arange(n_time), np.where(valid)[0], col[valid]
    )
```

iii. In trajectory steps 27, 49, and 57, the agent justified interpolation and nearest-fill by referencing `loadMotionEnergy.m`, which interpolates onto the analysis time axis and fills missing values.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized per session at the 50th percentile over non-NaN bins, with the invalid/default class `2` reserved for bins with no video.

ii.
```python
me_disc = discretize_per_session(me_valid, ~np.isnan(me_valid))
```

iii. In trajectory step 57, the agent said it would compute the session-specific 50th percentile and keep a separate missing/no-video state.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligned motion energy by using side-camera frame times, subtracting session video offset and per-trial go cue, and interpolating the result to the same 5 ms grid used for neural data.

ii.
```python
_, ft = sess.get_traj_trial(0, trix)
aligned_ft = ft - vidshift - go_cue[trix]
me_aligned[:, trix] = np.interp(
    time_axis, aligned_ft[:min_len], me_trial[:min_len],
    left=np.nan, right=np.nan
)
```

iii. In trajectory steps 49 and 57, the agent explicitly described motion energy as being aligned with side-camera times after video-shift correction and then resampled to the neural grid.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI added multiple defensive fallbacks. It returns a default video offset of `0.5` s if offset extraction fails, returns `(None, None)` for unreadable trajectory trials, uses zeros plus invisible masks when trajectory data are missing or too short, handles several v5/v7.3 and motion-energy storage variants, trims motion-energy/frame-time length mismatches, and fills motion-energy NaNs with nearest values. Missing or unreadable trials are generally kept rather than dropped unless they fail the neural all-zero filter.

ii.
```python
if len(valid_bc) == 0 or len(valid_bp) == 0:
    return 0.5
...
except Exception:
    return 0.5
```

```python
if ts is None or len(frame_times_aligned) < 3:
    return np.zeros(n_time), np.zeros(n_time, dtype=bool)
...
min_len = min(len(aligned_ft), len(me_trial))
...
if np.any(nans) and not np.all(nans):
    ...
    me_aligned[:, trix] = np.interp(
        np.arange(n_time), np.where(valid)[0], col[valid]
    )
```

iii. In trajectory steps 65, 75, 77, 84, 91, and 109, the agent repeatedly discussed making the loader robust to mixed MATLAB formats and mixed motion-energy layouts, and said nearest filling was acceptable for motion energy while low-confidence video should become “not visible.”

## 11-a. What are the most time-consuming steps of the code?

i. From the code structure, the most expensive parts are the nested neural loop over units and trials in `bin_and_smooth_spikes`, followed by the per-trial video interpolation loops for tongue, paw, and motion energy. Unlike the reference solution, the AI code does not vectorize spike counting across trials.

ii.
```python
for i, unit in enumerate(units):
    aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
    for j in range(n_trials):
        mask = unit['trial'] == (j + 1)
        ...
        counts = np.histogram(aligned[mask], bins=edges)[0]
        ...
```

```python
for trix in range(min(n_trials_total, n_traj_trials)):
    ...
for trix in range(min(n_trials_total, len(me_data), n_side_trials)):
    ...
for trix in range(n_trials_total):
    ...
```

iii. The trajectory does not contain an explicit runtime analysis for conversion. The closest evidence is that the agent focused on correctness and later on decoder runtime, not conversion profiling, so this assessment comes mainly from the implementation it wrote.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest avoidable target is the nested loop over units and trials in `bin_and_smooth_spikes`; it could have been replaced by a per-unit `histogram2d` or similar vectorized counting across all trials at once. The repeated per-trial interpolation/fill loops for motion energy are also vectorizable in principle, though less cleanly because trial lengths differ.

ii.
```python
for i, unit in enumerate(units):
    aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
    for j in range(n_trials):
        mask = unit['trial'] == (j + 1)
        ...
```

```python
for trix in range(n_trials_total):
    col = me_aligned[:, trix]
    ...
```

iii. The trajectory never claims these loops were necessary. The agent instead concentrated on making the mixed-format loaders work, so the lack of vectorization appears to be an implementation choice rather than an explicitly justified one.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it rebuilds the lowercase exclusion set inside the unit-filter comprehension for every session, loops over every trial inside every unit during spike binning, and does separate interpolation passes for velocity, visibility, motion energy, and then motion-energy gap filling. It also scans all kept trials again to find all-zero neural tensors.

ii.
```python
units = [u for u in units
         if u['quality'].lower() not in {q.lower() for q in EXCLUDED_QUALITIES}
         and u['quality'] != '']
```

```python
vel_interp = np.interp(time_axis, ft_vel, vel, left=np.nan, right=np.nan)
vis_interp = np.interp(time_axis, ft_vel, vis.astype(float), left=0, right=0) >= 0.5
...
me_aligned[:, trix] = np.interp(...)
...
me_aligned[:, trix] = np.interp(...)
```

iii. The trajectory does not present these repetitions as deliberate optimizations. Most came from the agent incrementally patching the pipeline after verifier warnings and format-discovery steps.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and stores some raw fields it never uses downstream (`tm`, `L`, and often `no` except for filtering), computes continuous tongue/paw velocities only to immediately collapse them to ternary categories, computes visibility traces only to use them as masks, and reads both paw traces even though the final output is a single discretized paw variable. It also passes `time_axis` into `bin_and_smooth_spikes` without using it there.

ii.
```python
units.append({
    'tm': tm, 'trial': trial, 'trialtm': trialtm, 'quality': quality
})
...
L = sess.get_bp_field('L')[:n_trials_total]
...
no = sess.get_bp_field('no')[:n_trials_total]
```

```python
def bin_and_smooth_spikes(units, go_cue_times, n_trials, edges, time_axis):
    """..."""
    n_time = len(time_axis)
```

```python
pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
pv_bot, pvis_bot = compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)
```

iii. There is no explicit justification in the trajectory for these extra computations. They follow from the agent’s choice to keep a fairly generic loader and to derive the discrete outputs from richer intermediate continuous traces.
