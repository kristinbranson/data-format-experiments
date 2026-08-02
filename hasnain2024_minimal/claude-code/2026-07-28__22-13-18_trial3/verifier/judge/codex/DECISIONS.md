# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes a session table in `EPHYS_SESSIONS`, then loops over it, opens each `data_structure_{animal}_{date}.mat` file, and reads it through a custom `SessionData` wrapper that supports both MATLAB v7.3/HDF5 and older v5 `.mat` files. It includes 25 `Ephys_Behavior` sessions and 20 `RandomizedDelay_Ephys_Behavior` sessions.

ii. 
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

class SessionData:
    def __init__(self, fpath):
        self.is_h5 = self._check_h5(fpath)
        if self.is_h5:
            self.f = h5py.File(fpath, 'r')
            self.obj = self.f['obj']
        else:
            mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
            self.obj_v5 = mat['obj']
```

```python
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
    if result is not None:
        all_sessions.append(result)
```

iii. The notes say this session list was meant to match the paper code's recording-and-video loaders. In trajectory step 46, the agent explicitly decided to "use all sessions from the Ephys_Behavior directory" and to include the RandomizedDelay sessions as well, because it thought DR-only sessions could still carry a constant DR context label. In step 101 it justified excluding `JEB24_2023-10-03` and `JEB24_2023-10-04` because those files lacked `clu` data.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `anm` field from each processed session. The script keeps the first-seen order of unique animal IDs in `subjects` and then builds `subject_idx` from each session's `anm`.

ii.
```python
subjects_set = []
...
if anm not in subjects_set:
    subjects_set.append(anm)

subjects = subjects_set
subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])
```

iii. This is not separately justified in the notes, but it follows from the session table design in `CONVERSION_NOTES.md`, which is organized by animal and session date.

## 1-c. How are the data split into sessions?

i. Each tuple in `EPHYS_SESSIONS` becomes one session. `process_session(...)` returns one session-level dictionary. Multi-probe sessions are merged inside that session by passing `probe_nums` into `get_clusters(...)`.

ii.
```python
def process_session(anm, date, probe_nums, data_dir, time_edges, time_axis):
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)
    ...
    clusters = sd.get_clusters(probe_nums)
```

```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': final_output_trials,
    'n_units': n_units_final,
    'anm': anm,
    'date': date,
    'n_trials': len(neural_trials),
}
```

iii. The notes say the session list was taken from the reference loading scripts, and the trajectory shows the agent deliberately combined probe lists such as `JEB15`'s `[1, 2]` into single sessions. It also chose to keep `JEB23_2023-10-20`, even though the reference loader comments that session out.

## 1-d. How are the data split into trials?

i. Trials are split by trial index within each session. The script first computes `valid_idx`, then for each valid trial it extracts `trialdat[:, :, trial_idx]`, creates one input array and one output array, and appends them as one trial entry.

ii.
```python
valid_idx = np.where(valid_trials)[0]
...
for trial_idx in valid_idx:
    neural = trialdat[:, :, trial_idx]
    input_data = time_axis.reshape(1, -1).copy()
    ...
    neural_trials.append(neural)
    input_trials.append(input_data)
    raw_outputs.append({...})
```

iii. The notes describe the dataset as a list of trials within each session and state that trial selection follows the reference trial filters.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they are `hit` or `miss`, are not stimulation trials, and are not early-lick trials. Ignore trials are excluded implicitly because they are neither `hit` nor `miss`. Sessions with fewer than 5 valid trials are skipped.

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
early = sd.get_trial_array('early').astype(bool)
stim_enable = sd.get_stim_enable().astype(bool)

valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]

if len(valid_idx) < 5:
    ...
    return None
```

iii. `CONVERSION_NOTES.md` says the intention was to match the reference conditions "include hit and miss; exclude stim, early, ignore/no-response". The trajectory summary in step 145 repeats that same interpretation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from cluster spike assignments: each kept cluster contributes its `quality`, `trial`, and `trialtm` arrays, and the per-trial spike times are aligned with `bp.ev.goCue`.

ii.
```python
clusters.append((q, trial, trialtm))
...
gocue = sd.get_event_times('goCue')
...
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        aligned_times = spike_times[trial_mask] - gocue[j]
```

iii. The notes say this was meant to follow the reference `alignSpikes.m` and `getSeq.m` pipeline: use trial-resolved spike times and align them to go cue.

## 2-b. How is the `neural` data processed?

i. For each unit and trial, spikes are histogrammed into fixed bins, converted to firing rate by dividing by `DT`, then smoothed with a causal Gaussian kernel using a reflect-like boundary condition.

ii.
```python
counts, _ = np.histogram(aligned_times, bins=time_edges)
fr = counts / DT
trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    ...
    kern = windows.gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0
    kern = kern / kern.sum()
    ...
```

iii. The notes say this was intended to match `getSeq.m` and `mySmooth.m`: 10 ms bins, 15-bin causal Gaussian, reflect boundary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script removes clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` and then removes units with mean firing rate `<= 1 Hz`, computed over the valid trials and time bins. Sessions with fewer than 10 surviving units are dropped.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]
...
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH

if keep_units.sum() < 10:
    ...
    return None
```

iii. The notes say this matches `findClusters.m` and `removeLowFRClusters.m`, and they add that sessions with fewer than 10 units were excluded to follow the Methods inclusion rule. The trajectory summary in step 145 repeats the same story.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every unit's spike times are aligned to go cue by subtracting the trial-specific `gocue[j]` before binning. The output bins therefore represent time relative to go-cue onset.

ii.
```python
gocue = sd.get_event_times('goCue')
...
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The notes explicitly say alignment is to go cue, and the trajectory repeatedly describes the converted data as "aligned to go cue".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 10 ms bins (`DT = 1/100`) over `[-2.5, 2.5]` seconds. There is no second-stage temporal rebinning after this histogramming step.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1/100  # 10 ms
...
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. The notes say this was chosen to match the reference scripts the agent used, especially `WorkingWithDataObjs.m`, rather than the 5 ms defaults still present in `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw timeseries variable. It is a synthetic time axis defined by `TMIN`, `TMAX`, and `DT`, interpreted relative to the raw `goCue` event.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
...
input_data = time_axis.reshape(1, -1).copy()
```

iii. The notes describe the decoder input as "time axis relative to go cue onset", which is exactly how the code constructs it.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes bin centers from the global edges and simply reshapes them to `(1, n_timebins)` for every trial.

ii.
```python
time_axis = time_edges[:-1] + DT / 2
...
input_data = time_axis.reshape(1, -1).copy()
```

iii. No extra justification is given beyond the notes' statement that the decoder input is continuous time relative to go cue.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the exact same `time_axis` and binning grid as the neural data, so each input timepoint corresponds one-to-one with a neural time bin.

ii.
```python
n_timebins = len(time_axis)
...
neural = trialdat[:, :, trial_idx]
input_data = time_axis.reshape(1, -1).copy()
```

iii. The notes say the input is "time axis relative to go cue onset", and the trajectory describes neural and behavior streams as interpolated or binned onto the same 10 ms grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The script derives lick direction from `bp.R` alone. If `R[trial_idx]` is true it labels the trial as right (`1`); otherwise left (`0`).

ii.
```python
R = sd.get_trial_array('R').astype(bool)
L = sd.get_trial_array('L').astype(bool)
...
lick_dir = 1 if R[trial_idx] else 0
```

iii. `CONVERSION_NOTES.md` says "lick_direction: left=0, right=1 (per-trial, from bp.R)". The notes do not mention combining `R/L` with `hit/miss`, and the trajectory summary in step 145 repeats the same simplification.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is no temporal processing. The per-trial binary label is broadcast across all time bins.

ii.
```python
out = np.zeros((6, n_timebins), dtype=np.int64)
out[0, :] = t['lick_dir']
```

iii. The notes frame lick direction as a per-trial output, so the agent duplicated that constant label over time.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii.
```python
autowater = sd.get_trial_array('autowater')
...
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. The tutorial in `WorkingWithDataObjs.m` says `autowater` can be used as a proxy for WC versus DR blocks, and `CONVERSION_NOTES.md` explicitly cites that proxy.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script maps `autowater == 1` to WC (`0`) and everything else to DR (`1`), then repeats that label over the entire trial.

ii.
```python
context = 0 if autowater[trial_idx] == 1 else 1
...
out[1, :] = t['context']
```

iii. The notes justify this as "WC = 0 (autowater=1), DR = 1 (autowater=0)". They also say DR-only sessions and RandomizedDelay sessions become all-DR.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit`, with miss trials treated as incorrect.

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
...
outcome = 1 if hit[trial_idx] else 0
```

iii. The notes explicitly say "outcome: incorrect=0 (miss), correct=1 (hit)".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script converts `hit` to a per-trial binary label and broadcasts it across all time bins.

ii.
```python
out[2, :] = t['outcome']
```

iii. The notes treat outcome as a per-trial categorical decoder output, so the implementation simply repeats it over time.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from bottom-camera trajectory data: the `frameTimes` and the x/y coordinates of one tongue feature, preferably `top_tongue`.

ii.
```python
bottom_feats = sd.get_traj_feature_names(1)
...
if name == 'top_tongue' and tongue_feat_idx is None:
    tongue_feat_idx = idx
...
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
```

iii. The notes say tongue velocity comes from bottom-camera DLC tracking of the `top_tongue` feature. The trajectory does not discuss alternatives beyond some feature-name fallback logic.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each valid trial, the script computes the Euclidean speed magnitude from frame-to-frame x/y differences at 400 Hz, places each speed sample at the midpoint of the frame interval, aligns it to go cue using the video offset, and linearly interpolates it onto the 10 ms neural grid.

ii.
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes justify this as "velocity = sqrt(dx^2 + dy^2) / dt at 400 Hz, interpolated to 10ms bins". In trajectory steps 124 and 126, the agent acknowledges that much of the signal remains NaN outside video coverage and that these NaNs later become class `0`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. It concatenates all tongue-velocity samples across the session, takes the per-session 50th percentile ignoring NaNs, then classifies each timepoint as `1` if it is at or above the threshold and `0` otherwise. NaNs are forced to class `0`.

ii.
```python
all_tongue = np.concatenate([t['tongue_vel'] for t in raw_outputs])
...
tongue_thresh = np.nanpercentile(all_tongue, 50) if not np.all(np.isnan(all_tongue)) else 0
...
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. The notes say the discretization follows the decoder task's per-session 50th-percentile instruction. In trajectory step 126, the agent explicitly says the threshold should be computed from valid, non-NaN values.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The raw video timestamps are corrected by the video offset and trial go-cue time, then interpolated onto the same `time_axis` used by the neural data.

ii.
```python
vidshift = sd.get_video_offset()
...
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes say this was meant to match `findVideoOffset.m` and the tutorial alignment convention `frameTimes - vidshift - goCue[trial]`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera trajectory data: `frameTimes` plus x/y coordinates of one paw feature, preferably `bottom_paw`.

ii.
```python
if name == 'bottom_paw' and paw_feat_idx is None:
    paw_feat_idx = idx
...
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
```

iii. The notes say paw velocity is computed "same as tongue but using `bottom_paw`". The trajectory does not give additional justification.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script applies the same processing as for tongue velocity: framewise Euclidean speed magnitude at 400 Hz, midpoint timestamps, go-cue/video-offset alignment, and interpolation onto the neural grid.

ii.
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. `CONVERSION_NOTES.md` says paw velocity is handled the same way as tongue velocity, but with `bottom_paw`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the per-session 50th percentile of all paw-velocity samples, ignoring NaNs, and assigns NaNs to class `0`.

ii.
```python
all_paw = np.concatenate([t['paw_vel'] for t in raw_outputs])
paw_thresh = np.nanpercentile(all_paw, 50) if not np.all(np.isnan(all_paw)) else 0
...
paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                   np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
```

iii. The notes say this follows the decoder task's requested per-session median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The frame timestamps are converted to go-cue-relative time by subtracting the video offset and trial go-cue time, then linearly interpolated onto the neural `time_axis`.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes say this uses the same alignment rule as the other video-derived signals.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_{animal}_{date}.mat` file, specifically its `me` variable's per-trial `data` arrays.

ii.
```python
def load_motion_energy(data_dir, anm, date):
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_path = os.path.join(DATA_ROOT, data_dir, me_fn)
    ...
    me_var = me_mat['me']
```

```python
if me_data is not None and trial_idx < len(me_data):
    ...
    me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The notes say motion energy is loaded from separate `motionEnergy_*.mat` files and that the loader was expanded to handle both struct and cell-array formats. Trajectory steps 110 and 137 mention those format differences explicitly.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script loads the motion-energy file, unwraps either a struct-backed or cell-backed `me` representation, then interpolates each trial's motion-energy trace from video frame times onto the neural time grid.

ii.
```python
if me_var.dtype.names is not None and 'data' in me_var.dtype.names:
    me_struct = me_var[0, 0] if me_var.ndim >= 2 else me_var[0]
    thresh = float(me_struct['moveThresh'].flatten()[0])
    data_field = me_struct['data']
    ...
    me_list = [data_arr[i].flatten().astype(float) for i in range(len(data_arr))]
```

```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The notes justify the loader changes as necessary because some motion-energy files were structs and others were cell arrays. The agent did not use the reference code's manual `moveThresh` for the final categories because the decoder task requested percentile-based discretization instead.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It concatenates the session's motion-energy samples, computes the 50th percentile ignoring NaNs, then binarizes each timepoint against that threshold. NaNs become class `0`.

ii.
```python
all_me = np.concatenate([t['motion_energy'] for t in raw_outputs])
me_thresh_50 = np.nanpercentile(all_me, 50) if not np.all(np.isnan(all_me)) else 0
...
me_disc = np.where(np.isnan(t['motion_energy']), 0,
                  np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)
```

iii. The notes say this follows the decoder task's explicit requirement, even though the paper code also ships a per-session manual motion-energy threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. For each trial, the script takes a camera frame-time vector, subtracts the video offset and trial go-cue time, and interpolates the motion-energy trace onto `time_axis`. In this implementation it uses `get_trial_frame_times(1, ...)`, which is the same internal camera index used for the bottom view elsewhere.

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The notes say the intended rule was `frameTimes - vidshift - goCue[trial]`, citing `findVideoOffset.m` and `WorkingWithDataObjs.m`. The trajectory does not mention the camera-index choice.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles many irregularities by skipping or defaulting. Missing session files are skipped. Missing `clu` data or too few valid trials/units cause the whole session to be skipped. Missing tongue/paw trajectories or motion energy leave NaNs in the continuous signals, and those NaNs later become class `0`. Motion-energy file-format mismatches are handled with loader branches. Failures are usually swallowed with bare `except`.

ii.
```python
if not os.path.exists(fpath):
    ...
    return None
...
except (KeyError, AttributeError) as e:
    ...
    return None
```

```python
tongue_vel = np.full(n_timebins, np.nan)
...
except:
    pass
...
me_interp = np.full(n_timebins, np.nan)
...
except:
    pass
```

```python
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0, ...)
paw_disc = np.where(np.isnan(t['paw_vel']), 0, ...)
me_disc = np.where(np.isnan(t['motion_energy']), 0, ...)
```

iii. The notes explicitly mention handling mixed motion-energy formats and a session with unavailable motion energy, and they say that unavailable motion energy yields all class `0`. Trajectory steps 101, 110, 124, 126, and 131 show the same pattern of "skip on missing neural data" and "default missing behavioral data to zeros after discretization".

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive work is the nested spike-binning loop over units and trials, followed by the per-trial video-derived interpolation blocks for tongue, paw, and motion energy, and the repeated loading of `.mat` files through `h5py`/`scipy.io`.

ii.
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=time_edges)
        ...
```

```python
for trial_idx in valid_idx:
    ...
    tongue_vel = np.interp(...)
    ...
    paw_vel = np.interp(...)
    ...
    me_interp = np.interp(...)
```

iii. This is not directly justified in the notes, but it follows from the code structure and the trajectory's concern with full-dataset runtime and repeated full-session conversions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over every trial for every unit could be vectorized or at least grouped by `trial_nums`; the nearly identical tongue/paw interpolation loops could be merged into one helper; and motion-energy interpolation could be batch-processed more aggressively.

ii.
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        ...
```

```python
for idx, name in enumerate(bottom_feats):
    ...
for idx, name in enumerate(bottom_feats):
    ...
```

iii. The trajectory does not discuss vectorization directly; this is inferred from the concrete repeated loop structure in `convert_data.py`.

## 11-c. What processing does the code repeat multiple times?

i. It repeats almost the same velocity pipeline twice, once for tongue and once for paw. It also repeats similar feature-name search logic and per-trial interpolation patterns across tongue, paw, and motion energy.

ii.
```python
# Tongue velocity
...
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
...
tongue_vel = np.interp(...)

# Paw velocity
...
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
...
paw_vel = np.interp(...)
```

iii. No explicit justification is given; this is simply how the final script was written.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `L` but never uses it. It returns `me_thresh` from `load_motion_energy(...)` but never uses it. It writes a `sample_data.pkl` side product that is not used by the main downstream decoder on `converted_data.pkl`. It also expands per-trial scalar labels (`lick_dir`, `context`, `outcome`) into full time-varying matrices even though they are constant within a trial.

ii.
```python
R = sd.get_trial_array('R').astype(bool)
L = sd.get_trial_array('L').astype(bool)
...
me_data, me_thresh = load_motion_energy(data_dir, anm, date)
```

```python
out[0, :] = t['lick_dir']
out[1, :] = t['context']
out[2, :] = t['outcome']
```

```python
sample_data = {
    'neural': neural[:n_sample],
    ...
}
```

iii. The notes justify the sample file as a convenience artifact for quick validation, but there is no claim that it is needed for the full downstream analysis. The rest is inferred directly from unused variables and duplicated constant outputs in the code.
