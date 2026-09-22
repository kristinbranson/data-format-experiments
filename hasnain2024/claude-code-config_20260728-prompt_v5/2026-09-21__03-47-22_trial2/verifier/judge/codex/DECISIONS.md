# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by scanning the two electrophysiology folders, `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`, for every file named `data_structure_*.mat`, skipping one hard-coded duplicate. Each session is loaded with a dual-path loader that first tries an HDF5 reader and then falls back to `scipy.io.loadmat`. Motion energy is loaded from a matching `motionEnergy_<animal>_<date>.mat` when present, otherwise from embedded `obj.me`. Trials are then taken from `bp['Ntrials']` and processed from the per-session arrays.

ii. 
```python
DATA_DIRS = {
    'Ephys': '/app/data/Ephys_Behavior',
    'RandDelay': '/app/data/RandomizedDelay_Ephys_Behavior',
}

SKIP_FILES = {
    'data_structure_JEB23_2023-10-20.mat',
}

def discover_sessions(sample=False):
    sessions = []
    for dir_key, data_dir in DATA_DIRS.items():
        ...
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            if fn in SKIP_FILES:
                continue
            ...
            sessions.append({
                'fpath': fpath,
                'me_path': me_path,
                'animal': animal,
                'date': date,
                'dir': dir_key,
                'session_id': f'{animal}_{date}',
            })
```

```python
def load_session(fpath):
    try:
        session = load_session_h5(fpath)
        return session
    except:
        pass
    try:
        session = load_session_scipy(fpath)
        return session
    except Exception as e:
        print(f"  ERROR loading {fpath}: {e}")
        return None
```

iii. The notes say the AI chose to include both electrophysiology folders, exclude behavior-only folders, and discover files from disk rather than transcribing the authors' session list. The trajectory also says it found and manually excluded one duplicate session (`JEB23_2023-10-20`).

## 1-b. How are the data split into subjects?

i. Subjects are derived from the filename token between `data_structure_` and the date, stored as `animal`, and deduplicated into a sorted `subjects` list. `subject_idx` is built from that mapping for each retained session.

ii. 
```python
parts = fn.replace('.mat', '').split('_')
animal = parts[2]
date = parts[3]
...
'session_id': f'{animal}_{date}',
```

```python
subjects_set = sorted(set(s['animal'] for s in sessions_info))
subject_to_idx = {s: i for i, s in enumerate(subjects_set)}
...
subject_idx_list.append(subject_to_idx[sess_info['animal']])
...
'subjects': subjects_set,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The notes explicitly describe the dataset as 14 animals across the neural sessions and rely on filenames for subject identity.

## 1-c. How are the data split into sessions?

i. One `data_structure_<animal>_<date>.mat` file is treated as one session. Sessions are discovered independently in the fixed-delay and randomized-delay directories and then processed uniformly. Sessions can later be skipped if they have no neural data, too few valid trials, or too few units after filtering.

ii. 
```python
for fn in sorted(os.listdir(data_dir)):
    if not fn.startswith('data_structure_'):
        continue
    ...
    sessions.append({
        'fpath': fpath,
        'me_path': me_path,
        'animal': animal,
        'date': date,
        'dir': dir_key,
        'session_id': f'{animal}_{date}',
    })
```

```python
result = process_session(...)
if result is None:
    continue
all_neural.append(result['neural'])
```

iii. The notes say the AI included both electrophysiology datasets and excluded sessions without neural data; later notes say it also excluded one duplicate to make the session count match the paper.

## 1-d. How are the data split into trials?

i. Trials are indexed by `bp['Ntrials']`, with trial-level arrays such as `hit`, `miss`, `early`, `autowater`, `goCue`, `frameTimes`, and spike `trial` numbers all assumed to share that indexing. The valid trial set is then selected by a boolean mask over these `Ntrials` entries.

ii. 
```python
session['Ntrials'] = int(np.array(bp['Ntrials']).flatten()[0])
...
n_trials = session['Ntrials']
```

```python
valid_trials = np.ones(n_trials, dtype=bool)
valid_trials[session['early']] = False
valid_trials[session['stim_enable']] = False
valid_trials[np.isnan(goCue)] = False

trial_indices = np.where(valid_trials)[0]
```

iii. The code treats the Bpod arrays as definitive trial tables. No additional reconstruction of trial boundaries is attempted.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops early-lick trials, photostimulation trials, and trials with `NaN` go-cue times. It does not explicitly remove trials that continue after electrophysiology recording has stopped; its notes acknowledge all-zero late trials in two sessions and keep them as artifacts rather than filtering them.

ii. 
```python
valid_trials = np.ones(n_trials, dtype=bool)
valid_trials[session['early']] = False
valid_trials[session['stim_enable']] = False
valid_trials[np.isnan(goCue)] = False
```

iii. The notes justify excluding early and stimulation trials as following the paper and reference code. They separately note that some late trials are all-zero neural data but describe them as genuine artifacts rather than conversion bugs, so they are retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from each neuron's `trialtm` spike times and `trial` trial identities within `obj.clu`, with `quality` used for curation and `goCue` used for alignment. The loader also reads `tm`, but downstream neural processing uses `trialtm` and `trial`.

ii. 
```python
neurons.append({
    'quality': quality,
    'trialtm': trialtm,
    'trial': trial,
})
```

```python
trialtm = neuron['trialtm']
trial = neuron['trial']
...
gc = goCue[t]
aligned = trialtm[spike_mask] - gc
```

iii. The notes identify `obj.clu` as the source of spikes and `obj.bp.ev.goCue` as the alignment event.

## 2-b. How is the `neural` data processed?

i. For each neuron and trial, spikes are selected by matching the spike's stored trial number, aligned by subtracting that trial's `goCue`, histogrammed into 10 ms bins from `-2.5` to `2.5` s, converted to firing rates in Hz, and smoothed with a custom causal Gaussian kernel of length 15 samples.

ii. 
```python
DT = 0.01
TMIN = -2.5
TMAX = 2.5
SMOOTH_WIN = 15
EDGES = np.arange(TMIN, TMAX + DT, DT)
```

```python
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / dt
fr = smooth_causal(fr, KERNEL, 'reflect')
trialdat[ni, :, t] = fr
```

iii. The notes repeatedly justify this as “matching reference code defaults,” specifically `dt = 1/100` and 15-sample causal smoothing from `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI excludes clusters whose quality labels are in `{'garbage', 'noisy', 'gabrga', 'real?', ''}`. After binning, it removes neurons with mean firing rate below 1 Hz over all retained trials and bins. Entire sessions are then dropped if fewer than 10 units remain.

ii. 
```python
EXCLUDE_QUALITY = {'garbage', 'noisy', 'gabrga', 'real?', ''}
LOW_FR_THRESH = 1.0
MIN_UNITS = 10
```

```python
if quality in EXCLUDE_QUALITY:
    continue
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=1)
keep_neurons = mean_fr >= LOW_FR_THRESH
trialdat = trialdat[keep_neurons, :, :]
...
if n_units < MIN_UNITS:
    print(f"  {session_id}: Too few units after filtering ({n_units}), skipping")
    return None
```

iii. The notes say this follows the paper and reference code’s “all” quality filter plus a 1 Hz minimum rate, and they explicitly add a minimum of 10 units per session as an inclusion criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the trial's go-cue time from `trialtm`, so the final spike histogram is in seconds relative to go-cue onset.

ii. 
```python
gc = goCue[t]
...
aligned = trialtm[spike_mask] - gc
counts, _ = np.histogram(aligned, bins=edges)
```

iii. The notes identify go cue as the standard alignment event and the code implements that directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural and behavioral time-varying data use 10 ms bins over a 5 s window, giving 500 time bins. There is no later rebinning step.

ii. 
```python
DT = 0.01            # 10 ms time bins (params.dt = 1/100)
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_CENTERS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_CENTERS)
```

iii. The notes explicitly choose 10 ms because the AI interpreted the reference code defaults as `params.dt = 1/100`, even though the paper text mentioned 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input time variable is not read directly from a raw field. It is generated from the converter's binning grid and implicitly tied to `goCue`, because all neural and video data are aligned relative to go-cue onset.

ii. 
```python
TIME_CENTERS = EDGES[:-1] + DT / 2
...
time_input = TIME_CENTERS.astype(np.float32)
result['input'].append(time_input[np.newaxis, :])
```

iii. The notes describe this as the continuous time vector from `-2.5` to `2.5` s around go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes a fixed vector of time-bin centers from the shared `EDGES` array and reuses that same vector for every trial.

ii. 
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_CENTERS = EDGES[:-1] + DT / 2
...
time_input = TIME_CENTERS.astype(np.float32)
```

iii. The notes justify this as using the same time window as the neural binning.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the neural binning grid: `TIME_CENTERS` are the centers of the same bins used to histogram aligned spikes.

ii. 
```python
counts, _ = np.histogram(aligned, bins=edges)
...
TIME_CENTERS = EDGES[:-1] + DT / 2
result['input'].append(time_input[np.newaxis, :])
```

iii. No separate alignment rule is used; the input is defined from the same shared bin grid as the neural data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from per-trial behavior flags: `hit`, `miss`, `L`, and `R`. The AI does not use the raw lick-time arrays for this output.

ii. 
```python
session['hit'] = np.array(bp['hit']).flatten().astype(bool)
session['miss'] = np.array(bp['miss']).flatten().astype(bool)
session['L'] = np.array(bp['L']).flatten().astype(bool)
session['R'] = np.array(bp['R']).flatten().astype(bool)
```

```python
lick_dir = np.full(n_valid, 2, dtype=np.int32)
...
if session['hit'][t]:
    if session['L'][t]:
        lick_dir[out_idx] = 0
    elif session['R'][t]:
        lick_dir[out_idx] = 1
elif session['miss'][t]:
    if session['L'][t]:
        lick_dir[out_idx] = 1
    elif session['R'][t]:
        lick_dir[out_idx] = 0
```

iii. The notes say `L/R` indicate stimulus direction rather than lick direction, so the lick output is inferred from stimulus side plus hit/miss.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI maps hit trials to the instructed side, miss trials to the opposite side, and leaves all other trials as “none.” It encodes left as `0`, right as `1`, and none as `2`, then broadcasts that per-trial label across all time bins.

ii. 
```python
lick_dir = np.full(n_valid, 2, dtype=np.int32)  # default: none
...
out[0, :] = lick_dir[i]
```

iii. The notes justify this as recovering actual lick direction from outcome and instructed side while keeping ignore trials as a third class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `autowater` trial flag.

ii. 
```python
session['autowater'] = np.array(bp['autowater']).flatten().astype(int)
```

```python
context = np.full(n_valid, 1, dtype=np.int32)  # default: DR
for out_idx, t in enumerate(trial_indices):
    if session['autowater'][t] == 1:
        context[out_idx] = 0
```

iii. The notes explicitly map `autowater` trials to WC and the rest to DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels each retained trial as WC (`0`) when `autowater == 1`, otherwise DR (`1`), and repeats that label across all time bins.

ii. 
```python
context = np.full(n_valid, 1, dtype=np.int32)
for out_idx, t in enumerate(trial_indices):
    if session['autowater'][t] == 1:
        context[out_idx] = 0
...
out[1, :] = context[i]
```

iii. The notes present this as a direct mapping from the trial table.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` and `miss` trial flags. The converter loads `no` as well, but outcome construction uses only hit and miss.

ii. 
```python
session['hit'] = np.array(bp['hit']).flatten().astype(bool)
session['miss'] = np.array(bp['miss']).flatten().astype(bool)
session['no'] = np.array(bp['no']).flatten().astype(bool)
```

```python
outcome = np.full(n_valid, 2, dtype=np.int32)  # default: ignore
for out_idx, t in enumerate(trial_indices):
    if session['hit'][t]:
        outcome[out_idx] = 1
    elif session['miss'][t]:
        outcome[out_idx] = 0
```

iii. The notes describe outcome as incorrect/correct/ignore and treat non-hit, non-miss trials as ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as incorrect `0` on miss trials, correct `1` on hit trials, and ignore `2` otherwise, then broadcast across time bins.

ii. 
```python
outcome = np.full(n_valid, 2, dtype=np.int32)
...
out[2, :] = outcome[i]
```

iii. The notes justify keeping ignore trials instead of dropping them so the converted dataset retains those trials for the other outputs.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` camera index 0, feature name `'tongue'`, using the per-frame `ts` coordinates/confidence, per-frame `frameTimes`, per-trial `goCue`, and the session-wide `vidshift` clock offset.

ii. 
```python
def compute_velocity_timeseries(session, cam_idx, feat_name, trial_indices, vidshift):
    ...
    cam = session['traj'][cam_idx]
    feat_idx = find_feature_index(cam['featNames'], feat_name)
    ...
    x = ts[feat_idx, 0, :]
    y = ts[feat_idx, 1, :]
    conf = ts[feat_idx, 2, :]
```

```python
tongue_vel, tongue_vis = compute_velocity_timeseries(
    session, cam_idx=0, feat_name='tongue',
    trial_indices=trial_indices, vidshift=vidshift)
```

iii. The notes claim the tongue came from the “bottom camera tongue feature,” but the code itself uses camera 0 and the feature named `tongue`. The rationale given in notes is only that this should yield a time-varying tongue signal suitable for decoder output.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes frame-to-frame Euclidean displacement times a fixed 400 Hz frame rate, marks frames below a confidence threshold of 0.5 as missing, linearly interpolates the resulting velocity onto the neural time-bin centers, and then discretizes the session-wide visible values at the 50th percentile. It does not smooth the tracked positions, differentiate with respect to actual frame times, or combine the two tongue views.

ii. 
```python
DLC_CONF_THRESH = 0.5
VIDEO_FPS = 400
```

```python
dx = np.diff(x)
dy = np.diff(y)
v = np.sqrt(dx**2 + dy**2) * VIDEO_FPS
v = np.concatenate([[0], v])

low_conf = conf < DLC_CONF_THRESH
v[low_conf] = np.nan
...
v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

```python
def discretize_velocity(velocity, visible):
    ...
    thresh = np.nanpercentile(vis_vals, 50)
    result[visible & (velocity < thresh)] = 0
    result[visible & (velocity >= thresh)] = 1
```

iii. The notes justify the median split as matching the decoder prompt and describe the velocity as Euclidean motion of the tracked point. No explicit justification is given for the lowered confidence threshold, interpolation, or omission of smoothing and dual-view fusion.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Session-wide visible tongue-velocity samples are split at the 50th percentile into `0` for below threshold and `1` for above-or-equal threshold; bins marked not visible are assigned `2`.

ii. 
```python
result = np.full((n_time, n_trials), 2, dtype=np.int32)
...
thresh = np.nanpercentile(vis_vals, 50)
result[visible & (velocity < thresh)] = 0
result[visible & (velocity >= thresh)] = 1
```

iii. The notes explicitly say the movement outputs should use per-session 50th-percentile thresholds and a third “not visible” class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI aligns each frame time by subtracting the session-wide `vidshift` and the trial's `goCue`, then interpolates tongue velocity onto the neural bin centers `TIME_CENTERS`.

ii. 
```python
aligned_ft = ft - vidshift - gc
...
v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
velocity[:, out_idx] = v_interp
```

iii. The notes say the video offset should match the reference `findVideoOffset.m`, and the code reuses the same aligned bin centers as the neural data via interpolation.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` camera index 1, feature name `'top_paw'`, again using `ts`, `frameTimes`, `goCue`, and `vidshift`.

ii. 
```python
paw_vel, paw_vis = compute_velocity_timeseries(
    session, cam_idx=1, feat_name='top_paw',
    trial_indices=trial_indices, vidshift=vidshift)
```

iii. The notes say paw velocity should come from a single tracked paw feature rather than combining views.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw velocity is processed with the same helper as tongue velocity: frame-to-frame Euclidean displacement at 400 Hz, confidence threshold 0.5, interpolation to neural time bins, and per-session median discretization with a not-visible class.

ii. 
```python
paw_vel, paw_vis = compute_velocity_timeseries(
    session, cam_idx=1, feat_name='top_paw',
    trial_indices=trial_indices, vidshift=vidshift)
paw_disc = discretize_velocity(paw_vel, paw_vis)
```

iii. The notes describe paw velocity as a time-varying DeepLabCut-derived output using the same thresholding convention as tongue velocity.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Visible paw-velocity samples are split at the session median into `0` and `1`, while bins that are not visible receive code `2`.

ii. 
```python
def discretize_velocity(velocity, visible):
    ...
    result = np.full((n_time, n_trials), 2, dtype=np.int32)
    thresh = np.nanpercentile(vis_vals, 50)
```

iii. The notes say all movement outputs use the same per-session 50th-percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are aligned by subtracting `vidshift` and per-trial `goCue`, then paw velocity is interpolated to the shared neural time centers.

ii. 
```python
aligned_ft = ft - vidshift - gc
...
v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. The notes justify reusing the same video-offset correction and time grid for all time-varying video outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived primarily from a matching `motionEnergy_<animal>_<date>.mat` file and secondarily from embedded `obj.me` if no external file is available. Alignment uses camera 0 frame times together with `goCue` and `vidshift`.

ii. 
```python
def load_motion_energy_file(me_path):
    d = scipy.io.loadmat(me_path, simplify_cells=True)
    me = d['me']
    ...
    return me_trials, float(thresh)
```

```python
me_trials = session.get('me_embedded', None)
has_me_file = me_path is not None and os.path.exists(me_path)

if has_me_file:
    me_trials_loaded, me_thresh = load_motion_energy_file(me_path)
    me_aligned = compute_motion_energy_timeseries(
        session, trial_indices, me_trials_loaded, vidshift)
elif me_trials is not None:
    me_aligned = compute_motion_energy_timeseries(
        session, trial_indices, me_trials, vidshift)
```

iii. The notes say some sessions had embedded motion energy and others separate files, so the AI supported both formats. The trajectory mentions fixing a second motion-energy file layout for randomized-delay sessions.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligns the per-frame motion-energy trace to trial time, interpolates it to neural time-bin centers, and discretizes valid samples by the session median into low/high, with NaNs becoming “no video.” It does not smooth or differentiate motion energy.

ii. 
```python
aligned_ft = ft - vidshift - gc
...
me_interp = np.interp(taxis, aligned_ft[valid], me_data[valid], left=np.nan, right=np.nan)
me_aligned[:, out_idx] = me_interp
```

```python
def discretize_motion_energy(me_aligned):
    ...
    thresh = np.nanpercentile(valid_vals, 50)
    result[valid & (me_aligned < thresh)] = 0
    result[valid & (me_aligned >= thresh)] = 1
```

iii. The notes justify the median split from the decoder prompt and treat motion energy as already precomputed upstream, so only temporal alignment and discretization are performed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Valid motion-energy samples are thresholded at the per-session 50th percentile into `0` and `1`, while missing samples remain `2` for no video.

ii. 
```python
result = np.full((n_time, n_trials), 2, dtype=np.int32)
...
thresh = np.nanpercentile(valid_vals, 50)
result[valid & (me_aligned < thresh)] = 0
result[valid & (me_aligned >= thresh)] = 1
```

iii. The notes explicitly specify a per-session 50th-percentile split and a third no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy frame times are taken from camera 0, shifted by `vidshift` and trial `goCue`, truncated to the shorter of the frame-time and motion-energy arrays, and linearly interpolated to the neural time centers. If frame times are missing, the AI synthesizes them as `np.arange(n_frames) / VIDEO_FPS + 0.5`.

ii. 
```python
cam = session['traj'][0]  # Use camera 0 (bottom) frame times
...
ft = cam['frameTimes'][t]
if ft is None or len(ft) == 0 or np.all(np.isnan(ft)):
    n_frames = len(me_trial)
    ft = np.arange(n_frames) / VIDEO_FPS + 0.5

aligned_ft = ft - vidshift - gc
min_len = min(len(aligned_ft), len(me_trial))
aligned_ft = aligned_ft[:min_len]
me_data = me_trial[:min_len]
...
me_interp = np.interp(taxis, aligned_ft[valid], me_data[valid], left=np.nan, right=np.nan)
```

iii. The notes justify using the same video-offset correction as other video streams. The fallback synthetic timestamps are an implementation choice rather than something justified in the notes.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses many permissive fallbacks. Missing lick arrays become empty arrays. Missing or unreadable session formats return `None`. Missing video offset defaults to `0.5` s. Missing frame times or feature arrays cause a trial's movement signal to remain all missing, which later becomes class `2`. Missing motion-energy frame times trigger synthetic 400 Hz timestamps with a 0.5 s offset. Trials with `NaN` go-cue are dropped entirely. It does not repair or remove trials that occur after recording has stopped.

ii. 
```python
except:
    session['vidshift'] = 0.5  # default padSec
```

```python
if ts is None or ft is None or np.isnan(gc):
    continue
if ft is None or len(ft) == 0:
    continue
if np.all(np.isnan(ft)):
    continue
```

```python
if ft is None or len(ft) == 0 or np.all(np.isnan(ft)):
    n_frames = len(me_trial)
    ft = np.arange(n_frames) / VIDEO_FPS + 0.5  # 0.5s pad offset
```

iii. The notes justify class `2` outputs as a way to preserve trials despite visibility problems and describe missing-video sessions as valid. They do not give a strong justification for fabricating motion-energy frame times or using a default `vidshift`.

## 11-a. What are the most time-consuming steps of the code?

i. The code structure suggests two main expensive stages: loading/parsing MATLAB files and the nested neural binning loop over neurons and trials. Video interpolation for tongue, paw, and motion energy also adds per-trial work, but the code explicitly measures only load time and spike-binning time for each session.

ii. 
```python
t0 = time.time()
session = load_session(fpath)
...
t_load = time.time() - t0
```

```python
t1 = time.time()
trialdat = align_and_bin_spikes(neurons, goCue, n_trials, EDGES, DT)
t_bin = time.time() - t1
...
print(f"  {session_id}: {len(neurons)} raw -> {n_units} units, {n_valid} trials "
      f"(load={t_load:.1f}s, bin={t_bin:.1f}s)")
```

iii. The notes and logs repeatedly report per-session load and bin times, implying that these were the stages the AI viewed as the dominant costs.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are left in scalar Python form: the neuron-by-trial spike histogram loop, the per-column convolution loop inside `smooth_causal`, the per-trial video alignment loops for each output, and the per-trial loops that assemble the final `neural`, `input`, and `output` lists.

ii. 
```python
for ni, neuron in enumerate(neurons):
    ...
    for t in range(n_trials):
        ...
        counts, _ = np.histogram(aligned, bins=edges)
```

```python
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kernel, mode='same')
```

```python
for out_idx, t in enumerate(trial_indices):
    ...
    v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. The notes do not defend these loops beyond prioritizing correctness and direct translation of the intended processing.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several conceptually similar passes: separate loops compute lick direction, context, and outcome even though they all traverse the same `trial_indices`; `compute_velocity_timeseries` is run separately for tongue and paw; and each trial is revisited again when building the output list. The same interpolation pattern is also reused separately for velocity and motion energy.

ii. 
```python
for out_idx, t in enumerate(trial_indices):
    if session['hit'][t]:
        ...

for out_idx, t in enumerate(trial_indices):
    if session['autowater'][t] == 1:
        ...

for out_idx, t in enumerate(trial_indices):
    if session['hit'][t]:
        ...
```

```python
tongue_vel, tongue_vis = compute_velocity_timeseries(...)
paw_vel, paw_vis = compute_velocity_timeseries(...)
```

```python
for i in range(n_valid):
    result['neural'].append(...)
    result['input'].append(...)
    ...
    result['output'].append(out)
```

iii. No explicit justification is given. The notes mainly focus on matching the desired outputs, not avoiding repeated passes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader reads several fields that are never used in the converted dataset, including raw spike `tm`, `sample`, `delay`, `lickL`, `lickR`, `no`, and motion-energy threshold metadata. The script also defines `compute_mean_fr` and an `all_subjects` list that are unused. More broadly, it loads all probes present in a file even though the authors' own analysis scripts specify session-specific probe choices.

ii. 
```python
tm_ref = dereffed['tm'][ni, 0]
...
tm = np.array(f[tm_ref]).flatten()
...
neurons.append({
    'quality': quality,
    'trialtm': trialtm,
    'trial': trial,
})
```

```python
session['sample'] = np.array(ev['sample']).flatten()
session['delay'] = np.array(ev['delay']).flatten()
...
session['lickL'] = []
session['lickR'] = []
```

```python
me_trials_loaded, me_thresh = load_motion_energy_file(me_path)
```

iii. The notes emphasize broad format support and exploratory sanity checks rather than minimizing unused intermediate work, so these extra reads and discarded values appear to be convenience choices.
