# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads each session from a MATLAB file `data_structure_<anm>_<date>.mat` located in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior` directories. A hard-coded list `SESSION_META` of 44 session dictionaries specifies the animal, date, probe number, and directory for each session. The AI auto-detects the file format: it tries HDF5 (v7.3) first via `h5py`, and falls back to scipy.io (v5). Motion energy is loaded from a separate `motionEnergy_<anm>_<date>.mat` file using `scipy.io.loadmat`. Each format has its own dedicated loading function (`load_session_h5` and `load_session_v5`).

ii.
```python
SESSION_META = [
    {'anm': 'EKH1', 'date': '2021-08-07', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    ...
    {'anm': 'JEB24', 'date': '2023-11-03', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
]

def load_session(filepath, probe_num):
    """Load session data, auto-detecting file format."""
    try:
        import h5py
        f = h5py.File(filepath, 'r')
        f.close()
        return load_session_h5(filepath, probe_num)
    except:
        return load_session_v5(filepath, probe_num)
```

iii. The AI notes in CONVERSION_NOTES.md that the session list is "Based on the loading scripts in code/DataLoadingScripts/Recording and video/". The two loading functions extract behavioral data, cluster data, trajectory data, and SpikeGLX data into a flat dictionary rather than preserving the nested MATLAB structure.

## 1-b. How are the data split into subjects?

i. The subject (animal) is extracted from the `anm` field in the `SESSION_META` dictionary for each session. Subjects are accumulated into a list as sessions are processed, with new subjects appended when first encountered. `subject_idx` maps each session to its index in the subject list.

ii.
```python
subj = result['subject']  # from session_meta['anm']
if subj not in all_subjects:
    all_subjects.append(subj)
subject_idx.append(all_subjects.index(subj))
```

iii. The subject identity comes from the hard-coded session metadata, which is derived from the loading scripts. The AI maintains subject ordering by insertion order rather than sorting.

## 1-c. How are the data split into sessions?

i. One session corresponds to one entry in the `SESSION_META` list, keyed by animal name and date. Each session maps to one `.mat` file in either the `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior` directory. The AI processes 44 sessions total (25 fixed-delay + 19 randomized-delay). Unlike the reference, each session entry specifies only a single probe number rather than a list of probes.

ii.
```python
for sess_meta in sessions_to_process:
    result = process_session(sess_meta, PARAMS, show_processing=args.show_processing)
    if result is None:
        continue
```

iii. The AI states sessions are based on the loading scripts. However, the AI's session metadata only supports a single probe per session, so multi-probe sessions (e.g., JEB15_2022-07-26 which has probes [1, 2]) only use one probe.

## 1-d. How are the data split into trials?

i. Trials are determined by `bp.Ntrials`, which gives the total number of trials in a session. Each behavioral field (hit, miss, early, etc.) is read as an array of length `ntrials`. Spike data carries trial numbers that are 1-indexed, used to assign spikes to trials during binning.

ii.
```python
ntrials = int(bp['Ntrials'][0, 0])
data['ntrials'] = ntrials
data['hit'] = bp['hit'][0, :].astype(bool)
...
for trial_idx in range(ntrials):
    trial_num = trial_idx + 1  # 1-indexed
    spike_mask = trial_nums == trial_num
```

iii. The AI uses `Ntrials` to determine the number of trials, consistent with the reference approach. However, unlike the reference, it does not truncate per-trial fields to `Ntrials` length (which the reference does with `[:n_trials]` to handle fields stored longer than the trial count).

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a more restrictive filter than the reference. It excludes: (1) early-lick trials, (2) no-response trials (`bp.no`), (3) photostimulation trials (`bp.stim.enable`), and (4) requires trials to be either hit or miss. This effectively removes all ignore/no-response trials from the dataset. The reference only excludes early-lick and photostim trials, keeping ignore trials.

ii.
```python
valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
```

Additionally, after building neural data, it removes trials with all-zero neural activity:
```python
for ni, ii, oi in zip(neural_trials, input_trials, output_trials):
    if np.all(ni == 0):
        n_removed += 1
        continue
```

iii. The AI's CONVERSION_NOTES.md states "Trial filtering: Exclude early, no-response, stim trials; keep hit+miss" as a key decision. This is more restrictive than the reference, which keeps ignore trials and assigns them a third class for lick direction and outcome.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu{probe}` spike-sorted cluster data, specifically the `trialtm` (spike times relative to trial start), `trial` (trial number for each spike), and `quality` (manual curation label) fields. The go cue times `bp.ev.goCue` are used for temporal alignment.

ii.
```python
clu_info['trialtm'] = f[ref][()].flatten()
clu_info['trial'] = f[ref][()].flatten().astype(int)
clu_info['quality'] = read_h5_string(f, ref).strip()
...
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
```

iii. The AI correctly identifies the cluster data as the source of neural activity, using the same fields as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue by subtracting `goCue` from `trialtm`, then binned into 5 ms bins using `np.histogram`. The counts are converted to firing rates (Hz) by dividing by the bin width. Then a **causal** Gaussian kernel is applied for smoothing (window size N=15, with the first half of the kernel zeroed out). This differs from the reference, which uses a symmetric Gaussian (`gaussian_filter1d` with `mode='reflect'`).

ii.
```python
# Bin spikes
counts, _ = np.histogram(aligned_times, bins=edges)
# Convert to firing rate and smooth
fr = counts.astype(np.float32) / dt
fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
```

```python
def make_causal_gaussian_kernel(N):
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(N))))
    kern[:N // 2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
    return kern
```

iii. The AI notes in CONVERSION_NOTES.md: "Causal Gaussian smoothing matches mySmooth.m". The reference code's `mySmooth.m` does indeed use a causal kernel. However, the reference solution uses symmetric `gaussian_filter1d` with sigma=14ms (2.8 bins). The AI's kernel standard deviation is `np.std(np.arange(15))` which is ~4.32 samples, different from the reference's 2.8 bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied. First, clusters whose quality label (lowercased, stripped) matches the set `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded. Second, clusters whose mean firing rate (averaged over all trials and time) falls below 0.5 Hz are excluded. Sessions with fewer than 10 remaining units are dropped entirely.

ii.
```python
excluded = params['excluded_qualities']  # {'garbage', 'gabrga', 'noisy', 'real?'}
valid_clusters = []
for i, clu in enumerate(session_data['clusters']):
    q = clu['quality'].lower().strip().replace('\x00', '')
    if q in excluded:
        continue
    valid_clusters.append(clu)
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=0)
keep_mask = mean_fr > params['low_fr']  # 0.5 Hz
```

iii. The quality exclusion set matches the reference code's `findClusters.m` but does NOT include `'poor'`, which the reference solution adds. The firing rate threshold is 0.5 Hz (from `getDefaultParams.m`) versus the reference solution's 1.0 Hz (from the paper text). The minimum units per session (10) is an additional filter not present in the reference solution.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting the go cue time from each spike's trial-relative time: `trialtm - goCue[trial_idx]`. This is done per-neuron and per-trial in a nested loop.

ii.
```python
for trial_idx in range(ntrials):
    trial_num = trial_idx + 1
    spike_mask = trial_nums == trial_num
    if not np.any(spike_mask):
        continue
    aligned_times = trialtm[spike_mask] - goCue[trial_idx]
    counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The alignment logic is correct and matches the reference approach of `trialtm - goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (dt = 1/200 s) spanning -2.5 to +2.5 s from go cue, yielding 1000 time bins. No rebinning is applied. The bin edges are `np.arange(tmin, tmax + dt, dt)` and time centers are `edges[:-1] + dt/2`.

ii.
```python
PARAMS = {
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1.0 / 200.0,  # 5ms bins
    ...
}
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```

iii. This matches the reference's `getDefaultParams.m` parameters and the reference solution.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself, computed from the binning parameters (tmin, tmax, dt). It is not derived from any raw data variable; it is constructed as the centers of the 5 ms bins.

ii.
```python
time_input = time_axis.reshape(1, -1).astype(np.float32)
input_trials.append(time_input)
```

iii. This is identical to the reference approach.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond constructing the time axis from the bin parameters. The time axis is defined once and reused for every trial.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the bin centers of the same 5 ms grid used for neural binning, so they are inherently aligned.

ii.
```python
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```

iii. Same grid as neural data, consistent with reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction directly from `bp.R` (the instructed right-lick side). It does NOT use hit/miss to infer the actual lick direction. `R=1` maps to right (1), `R=0` maps to left (0).

ii.
```python
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

iii. The AI's CONVERSION_NOTES.md states the mapping is "L=0, R=1" under the variable mapping. This is conceptually incorrect: `bp.R` indicates the instructed direction, not the actual lick direction. On miss trials, the animal licked the opposite side from the instructed one.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct cast of the `bp.R` boolean to integer. No derivation from hit/miss is performed. The output has only 2 classes (left=0, right=1) because ignore trials are already excluded by the trial filter. The reference solution derives 3 classes (left=0, right=1, no lick=2) using the combination of instructed side and hit/miss outcome.

ii.
```python
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

Output values:
```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. Because the AI filters to only hit+miss trials, and then uses `bp.R` directly, this gives the instructed side rather than the actual lick direction. On hit trials these coincide, but on miss trials the actual lick is the opposite of `R`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. When `autowater == 0`, the trial is classified as DR (1); when non-zero, as WC (0).

ii.
```python
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. This correctly uses the autowater field to determine behavioral context. The mapping (WC=0, DR=1) matches the instructions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A boolean comparison: `autowater == 0` gives True for DR trials, cast to int (1). Autowater non-zero gives WC (0). This is the inverse logic of the reference (`np.where(autowater, WC=0, DR=1)`), but produces the same result.

ii.
```python
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. Equivalent to reference's `np.where(autowater, CONTEXT['WC'], CONTEXT['DR'])`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`. Since the AI filters to only hit+miss trials, a hit (1) is correct and a miss (0) is incorrect.

ii.
```python
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

iii. Because ignore trials are already excluded, the hit flag alone suffices for a 2-class outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct cast of `bp.hit` to integer. Only 2 classes: incorrect (0) and correct (1). The reference solution has 3 classes, including ignore (2), because it retains ignore trials.

ii.
```python
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

Output values:
```python
['incorrect', 'correct'],
```

iii. The instructions specify "incorrect = 0, correct = 1", which the AI follows. However, the instructions do not mandate removing ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from the DLC tracking in `obj.traj`, specifically the side camera (cam index 0) `tongue` feature. The AI uses only the side camera, not both cameras. Frame times (`frameTimes`) and the SpikeGLX bitcode data are used for temporal alignment.

ii.
```python
cam0 = session_data['traj'][0]  # side cam
feat_names = cam0['featNames']
tongue_idx = None
for fi, fn in enumerate(feat_names):
    if fn.lower() == 'tongue':
        tongue_idx = fi
        break
```

iii. The reference uses both side camera (`tongue`) and bottom camera (`top_tongue`) to get a combined estimate. The AI only uses one camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Several steps: (1) Extract x, y positions for the tongue feature from the tracking data. (2) Interpolate positions onto the neural time axis using `interp1d`. (3) Fill NaN positions with the session-mean position (baseline). (4) Compute velocity using `np.gradient` on the interpolated, filled positions. (5) Set velocity to 0 where the tongue was originally not visible. (6) Compute speed as `sqrt(xvel^2 + yvel^2)`. No likelihood filtering is applied (the reference filters at likelihood > 0.9). No smoothing of positions before differentiation (the reference applies a 5 ms Gaussian). No normalization by percentile.

ii.
```python
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
all_y[:, trial_idx] = fy(time_axis)
...
all_x_filled[np.isnan(all_x_filled)] = mean_x
all_y_filled[np.isnan(all_y_filled)] = mean_y
...
xvel = np.gradient(all_x_filled[:, trial_idx])
yvel = np.gradient(all_y_filled[:, trial_idx])
nan_mask = np.isnan(all_x[:, trial_idx])
xvel[nan_mask] = 0.0
yvel[nan_mask] = 0.0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The processing substantially differs from the reference: (a) interpolation to uniform grid vs. frame-resolution computation, (b) filling NaN with baseline vs. leaving as NaN, (c) no likelihood filtering, (d) no per-run smoothing, (e) single camera vs. dual camera with normalization. The AI's approach follows a different philosophy inspired by the reference code's `findPosition.m` (which interpolates) rather than the reference solution's frame-resolution approach.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized using the 50th percentile of non-NaN values as threshold. Values >= threshold become 1, < threshold become 0. NaN values are mapped to 0 (below threshold). Only 2 classes, unlike the reference's 3 classes (which includes "not visible" = 2).

ii.
```python
def discretize_velocity(vel_data):
    valid_vals = vel_data[~np.isnan(vel_data)]
    threshold = np.percentile(valid_vals, 50)
    if threshold <= np.min(valid_vals) + 1e-10:
        discretized = (vel_data > threshold).astype(int)
    else:
        discretized = (vel_data >= threshold).astype(int)
    discretized[np.isnan(vel_data)] = 0  # default for NaN
    return discretized, threshold
```

iii. The instructions specify "0: < 50th percentile, 1: >= 50th percentile", which the AI follows for the 2-class case. However, by filling NaN tongue positions with baseline and setting velocity to 0 where not visible, many bins that should be "not visible" instead have low velocity values that contaminate the threshold computation.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed from the SpikeGLX bitcode data (matching `findVideoOffset.m`). Frame times are corrected by subtracting the video offset and go cue time. Then the positions are interpolated onto the neural time axis using `interp1d`.

ii.
```python
vidshift = find_video_offset(session_data)
aligned_ft = ft - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
```

iii. The alignment approach is correct in principle, using the same video offset computation as the reference. However, the AI uses `nanmedian` for the behavior bitStart rather than mode, which could give slightly different results.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from the bottom camera (cam index 1) tracking data, using the `top_paw` feature. Falls back to any feature containing "paw" if `top_paw` is not found.

ii.
```python
cam1 = session_data['traj'][1]  # top cam
feat_names = cam1['featNames']
paw_idx = None
for fi, fn in enumerate(feat_names):
    if 'top_paw' in fn.lower():
        paw_idx = fi
        break
```

iii. The AI correctly identifies the bottom camera and `top_paw` feature, matching the reference.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated onto the neural time axis. NaN values are filled by linear interpolation. Velocity is computed using `np.gradient`. A baseline derivative is subtracted (median of `np.diff`). Speed is computed as the magnitude. This differs from the reference which computes velocity at frame resolution with likelihood filtering, per-run smoothing, and no baseline subtraction.

ii.
```python
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
x_interp = fx(time_axis)
y_interp = fy(time_axis)
# Fill missing
for arr in [x_interp, y_interp]:
    nans = np.isnan(arr)
    if np.any(nans) and np.any(~nans):
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
# Compute velocity
xvel = np.gradient(x_interp)
yvel = np.gradient(y_interp)
# Subtract baseline derivative
basederiv_x = np.nanmedian(np.diff(x_interp))
basederiv_y = np.nanmedian(np.diff(y_interp))
xvel -= basederiv_x
yvel -= basederiv_y
```

iii. The baseline subtraction is a departure from the reference. Interpolating NaN values and then computing velocity on the filled data produces different results from the reference's frame-resolution approach with likelihood gating.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same discretization as tongue velocity: 50th percentile threshold, 2 classes, NaN mapped to 0.

ii.
```python
paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)
```

iii. Same approach as tongue; reference uses 3 classes with "not visible".

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: video offset subtracted from frame times, go cue subtracted, then interpolated onto neural time axis.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. Same alignment as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_<anm>_<date>.mat` files. The AI handles three formats: (1) standard `me.data` cell array with `me.moveThresh`, (2) nested `me.data.data`, and (3) direct cell array.

ii.
```python
def load_motion_energy(me_filepath):
    d = sio.loadmat(me_filepath, squeeze_me=False)
    me_raw = d['me']
    ...
```

iii. The AI correctly handles multiple ME file formats, similar to the reference's loop-based unwrapping.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates motion energy values onto the neural time axis using `interp1d`. It then fills remaining NaN values using nearest-neighbor interpolation. This differs from the reference which bins frame-resolution values into 5 ms bins using mean.

ii.
```python
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)
...
# Fill NaN with nearest
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. The interpolation approach differs from the reference's binning approach. The NaN filling is a notable difference: the reference leaves bins with no frames as NaN and assigns them a "not visible" class.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same discretization as tongue/paw: 50th percentile threshold, 2 classes, NaN mapped to 0.

ii.
```python
me_disc, me_thresh = discretize_velocity(valid_me)
```

iii. Reference uses 3 classes with "not visible" for untracked bins.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times for alignment. The video offset is subtracted from frame times, then go cue time, and the values are interpolated onto the neural time axis.

ii.
```python
aligned_ft = ft - vidshift - goCue[trial_idx]
if len(trial_me) == len(ft):
    f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
    me_aligned[:, trial_idx] = f_interp(time_axis)
```

iii. The alignment approach is correct in principle, using the same camera frame times as the reference.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions with too few clusters after quality filtering (< 10) are skipped. (2) Trials with all-zero neural data are removed. (3) Missing motion energy data is filled with nearest-neighbor interpolation. (4) Missing tongue positions are filled with the session-mean position, and velocity is set to 0 where not visible. (5) Missing paw positions are filled by linear interpolation. (6) NaN in discretized outputs is mapped to 0 (below threshold).

ii.
```python
if len(valid_clusters) < params['min_units']:
    print(f"    WARNING: Only {len(valid_clusters)} units, skipping session")
    return None
...
if np.all(ni == 0):
    n_removed += 1
    continue
...
discretized[np.isnan(vel_data)] = 0
```

iii. The AI takes a "fill everything" approach, interpolating or filling missing values rather than preserving them as a distinct category. The reference preserves missing data as a "not visible" class.

## 11-a. What are the most time-consuming steps of the code?

i. The AI reports per-session processing times in the output, with data loading being the dominant cost. The spike binning loop (per-neuron, per-trial) is also computationally expensive due to being doubly nested. Total conversion takes about 247 seconds for 44 sessions.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    trialtm = clu['trialtm']
    trial_nums = clu['trial']
    for trial_idx in range(ntrials):
        trial_num = trial_idx + 1
        spike_mask = trial_nums == trial_num
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI notes in CONVERSION_NOTES.md that estimated full time is ~264s, actual was 247s.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning is done with a doubly nested loop (over neurons and trials), calling `np.histogram` once per neuron per trial. The reference vectorizes this with a single `np.histogram2d` call per neuron over all trials simultaneously. The tongue, paw, and motion energy processing also use per-trial loops.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        spike_mask = trial_nums == trial_num
        aligned_times = trialtm[spike_mask] - goCue[trial_idx]
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The per-neuron-per-trial spike binning loop is the most significant inefficiency compared to the reference's vectorized approach.

## 11-c. What processing does the code repeat multiple times?

i. The trajectory data is accessed multiple times for different features (tongue, paw, motion energy), each time re-reading frame times and extracting features. The `NdroppedFrames` check is done for tongue and paw separately. The video offset is computed once per session, which is efficient.

ii.
```python
# Tongue processing reads cam0 trials
for trial_idx in range(min(ntrials, len(cam0['trials']))):
    trial_info = cam0['trials'][trial_idx]
    ft = trial_info['frameTimes']
    ...
# Paw processing reads cam1 trials
for trial_idx in range(min(ntrials, len(cam1['trials']))):
    trial_info = cam1['trials'][trial_idx]
    ft = trial_info['frameTimes']
    ...
# ME processing also reads cam0 frame times
```

iii. Each stream (tongue, paw, ME) accesses frame times independently, but since they use different cameras/features, some repetition is unavoidable.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads several fields that are not used in the final output: `lickL`, `lickR` (lick times), `sample`, `delay` (event times), `NdroppedFrames`, and the `moveThresh` from motion energy files. The spike binning is done for ALL trials (including filtered ones) before selecting valid trials, which wastes computation.

ii.
```python
data['lickL'] = []
data['lickR'] = []
data['sample'] = ev['sample'][0, :]
data['delay'] = ev['delay'][0, :]
...
trialdat = np.zeros((n_timepts, n_neurons, ntrials), dtype=np.float32)
# Binning for ALL ntrials, then:
trialdat = trialdat[:, :, valid_trial_indices]  # ... but this line doesn't exist
# Instead valid_trial_indices is used later to select
```

iii. The most significant waste is computing firing rates for ALL trials (including early, no-response, and stim trials) before filtering. The reference bins only the trials that pass filtering.
