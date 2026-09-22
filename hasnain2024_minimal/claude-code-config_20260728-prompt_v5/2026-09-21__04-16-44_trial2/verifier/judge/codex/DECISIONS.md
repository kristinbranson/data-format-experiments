# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 44-session `SESSION_META` list spanning `/app/data/Ephys_Behavior` and `/app/data/RandomizedDelay_Ephys_Behavior`. For each session it loads `data_structure_<anm>_<date>.mat`, tries HDF5 first and then SciPy/MATLAB-v5 parsing, and separately attempts to load `motionEnergy_<anm>_<date>.mat`.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [2], EPHYS_DIR),
    ...
    ('JEB24', '2023-11-03', [1], RANDDELAY_DIR),
]

def load_session(anm, date, probe_list, data_dir):
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(data_dir, fn)
    try:
        sess = load_session_h5(fpath, probe_list)
    except Exception:
        sess = load_session_scipy(fpath, probe_list)
    ...
    me_fn = f'motionEnergy_{anm}_{date}.mat'
```

iii. In trajectory step 67 the agent said it would "load all sessions from both directories using the probes specified in the loading scripts." In step 60 it explicitly excluded the "Video only" sessions because they lacked neural data.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `anm` field in each `SESSION_META` tuple. The output `subjects` list preserves first-seen order rather than sorting, and `subject_idx` stores the index of each session's animal in that list.

ii.
```python
for anm, date, probes, data_dir in SESSION_META:
    ...
    if anm not in all_subjects:
        all_subjects.append(anm)
    subj_idx = all_subjects.index(anm)
    ...
    subject_idx_list.append(subj_idx)
```

iii. The trajectory does not contain a separate argument about subject indexing; this choice is implicit in the session list and final assembly code.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESSION_META` is treated as one session. After processing, each session becomes one element of `neural`, `input`, and `output`.

ii.
```python
for anm, date, probes, data_dir in SESSION_META:
    ...
    result = process_session(sess)
    if result is None:
        continue
    all_sessions.append(result)

neural = [s['neural'] for s in all_sessions]
inputs = [s['input'] for s in all_sessions]
outputs = [s['output'] for s in all_sessions]
```

iii. In steps 60 and 67 the agent says it is using the sessions from the loading scripts across both ephys directories, with behavior-only sessions omitted.

## 1-d. How are the data split into trials?

i. Trials are indexed as `0..ntrials-1` using `sess['ntrials']`. Neural spikes are assigned to trials by comparing `cluster['trial']` against `ti + 1`, and video/motion-energy streams are read from per-trial arrays at the same trial index.

ii.
```python
ntrials = sess['ntrials']
for ti in range(ntrials):
    spk_mask = spike_trials_raw == (ti + 1)
    ...
    ft = sess['trial_frame_times'][ti]
    tongue_xy = sess['trial_tongue_xy'][ti]
    ...
    me_trial = sess['me_data'][ti]
```

iii. The trajectory does not discuss trial segmentation separately; the code assumes the stored MATLAB trial-wise arrays already define the trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials where `stim.enable` or `early` are true. It also skips whole sessions with fewer than two remaining trials, but it does not remove late trials that extend past the end of the recording.

ii.
```python
trial_mask = ~sess['stim_enable'] & ~sess['early']
valid_trials = np.where(trial_mask)[0]

if len(valid_trials) < 2:
    print(f"  Skipping {sess['anm']}_{sess['date']}: too few valid trials")
    return None
```

iii. In trajectory step 60 the agent explicitly says it is "excluding stim and early-lick trials" and requiring at least 2 valid trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from cluster spike assignments and times: `obj.clu[*].trial` and `obj.clu[*].trialtm`. `quality` is used for QC filtering, and `goCue` is used to realign spikes.

ii.
```python
spike_trials = np.array(f[trial_ds[ci, 0]]).flatten()
spike_trialtm = np.array(f[trialtm_ds[ci, 0]]).flatten()
...
goCue = np.array(ev['goCue']).flatten()
...
aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
```

iii. In steps 39 and 41 the agent inspected `obj.clu` specifically to understand probe structure, cluster `quality`, `trial`, and `trialtm`.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into a `-2.5` to `2.5` s window with `DT = 10 ms`, converted to firing rates by dividing by bin width, and smoothed with a causal half-Gaussian kernel of length 15 bins using reflected padding.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15

counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts / DT
trialdat[:, ui, ti] = smooth_data(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. In trajectory step 60 the agent said it would use "10ms time bins, and 15ms causal smoothing," and the file docstring repeats that it is following a causal half-Gaussian pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters whose manual quality label lower-cases to `garbage`, `gabrga`, `noisy`, or `real?` are removed. Remaining units are filtered to mean firing rate `> 1 Hz` on valid trials, and sessions with fewer than 10 surviving units are skipped.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESH = 1.0
MIN_UNITS = 10

if q_str.lower() in {q.lower() for q in EXCLUDED_QUALITIES}:
    continue
...
mean_frs = np.mean(trialdat[:, :, valid_trials], axis=(0, 2))
good_units = mean_frs > LOW_FR_THRESH
if good_units.sum() < MIN_UNITS:
    return None
```

iii. In step 60 the agent says it is "keeping all clusters except those marked garbage or noisy, filtering units below 1 Hz firing rate, ... and requiring at least 10 units per session."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike is aligned by subtracting that trial's `goCue` time from `trialtm`. No interpolation or additional offset is applied to the neural stream.

ii.
```python
aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The docstring and step 60 both state that the neural data are aligned to go cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins over a 5 s window, so each trial has 500 time bins. The only temporal binning is the histogramming into those bins.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = edges[:-1] + DT / 2
```

iii. The agent repeatedly justified this with its reading of the MATLAB parameters, writing in step 60 and the script docstring that the reference used "dt = 10 ms (params.dt = 1/100)."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not taken from a stored trial field beyond the alignment event itself; it is the synthetic time axis relative to go cue defined by `TMIN`, `TMAX`, and `DT`.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100
time_centers = edges[:-1] + DT / 2
input_trials.append(taxis.reshape(1, -1).astype(np.float32))
```

iii. In step 60 the agent said it would align everything to go cue onset and use a fixed `-2.5` to `2.5` s window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script constructs evenly spaced bin centers from the fixed window and stores that same 1D time vector for every trial.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = edges[:-1] + DT / 2
...
input_trials.append(taxis.reshape(1, -1).astype(np.float32))
```

iii. No separate trajectory discussion exists beyond the agent's general claim in step 60 that all streams would share the go-cue-aligned window.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector is exactly the same `time_centers` array used as the neural bin centers, so it is aligned by construction.

ii.
```python
time_centers = edges[:-1] + DT / 2
taxis = time_centers
...
neural_trials.append(trialdat[:, :, ti].T.astype(np.float32))
input_trials.append(taxis.reshape(1, -1).astype(np.float32))
```

iii. This is implicit in the implementation; the trajectory only states the common go-cue alignment goal.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the trial outcome flags `hit` and `miss` plus the instructed side flag `R` (with left inferred from `not R`). The script also loads `L` and `no`, but the final rule uses `hit`, `miss`, and `R`.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
```

iii. In trajectory step 147 the agent explicitly reasoned that `R`/`L` encode the cued direction, not the animal's actual lick, so actual lick direction must be inferred from `hit`/`miss` plus cue side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hit trials are assigned the cued direction, miss trials the opposite direction, and all other trials are assigned a third "none" class.

ii.
```python
if sess['hit'][ti]:
    lick_dir = 1 if sess['R'][ti] else 0
elif sess['miss'][ti]:
    lick_dir = 0 if sess['R'][ti] else 1
else:
    lick_dir = 2
```

iii. Step 147 gives the AI's full justification: ignore trials should be "none," and `hit`/`miss` determine whether the animal licked with or against the cued side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from the per-trial `autowater` flag.

ii.
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
...
context = 0 if sess['autowater'][ti] else 1
```

iii. In step 60 the agent says it will "split by autowater into WC versus DR context."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script directly maps `autowater=True` to WC (`0`) and `False` to DR (`1`).

ii.
```python
# Context: 0=WC, 1=DR
context = 0 if sess['autowater'][ti] else 1
```

iii. The trajectory does not add further justification beyond the autowater-based split described in step 60.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit` and `miss`; the code also loads `no`, but does not use it in the final classification rule.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
...
outcome = 1 if sess['hit'][ti] else (0 if sess['miss'][ti] else 2)
```

iii. In step 60 the agent says outcome will be "categorized as hit, miss, or ignore for correct, incorrect, and no-response respectively."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The mapping is miss -> incorrect (`0`), hit -> correct (`1`), otherwise ignore (`2`).

ii.
```python
# Outcome: 0=incorrect, 1=correct, 2=ignore
outcome = 1 if sess['hit'][ti] else (0 if sess['miss'][ti] else 2)
```

iii. The trajectory's justification is again step 60's hit/miss/ignore mapping.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The script derives tongue velocity from the bottom camera's `top_tongue` coordinates and confidence in `obj.traj`, using per-trial `frameTimes`, plus session-wide `vidshift` and trial `goCue` for alignment. It does not load the side-camera `tongue` feature.

ii.
```python
# Load trajectory (bottom cam for tongue/paw)
bottom_ref = traj_data[1, 0]
...
tongue_idx = bottom_feat_names.index('top_tongue') if 'top_tongue' in bottom_feat_names else None
...
tongue_xy = sess['trial_tongue_xy'][ti]
tongue_conf = sess['trial_tongue_conf'][ti]
vel, vis = compute_velocity(tongue_xy, confidence=tongue_conf, fill_missing=False)
```

iii. In step 120 the agent focuses on DLC confidence and tongue visibility, arguing that tongue should not have missing samples filled. It does not justify dropping the side-camera tongue view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Tongue velocity is computed as frame-to-frame Euclidean speed from `x,y` differences at 400 Hz, with no smoothing. Frames with confidence `< 0.9` are treated as invisible, missing values are not filled for tongue, and the resulting velocity is linearly interpolated onto the neural time grid.

ii.
```python
def compute_velocity(xy_coords, confidence=None, fill_missing=True):
    ...
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    vel = np.sqrt(dx**2 + dy**2) * VIDEO_FPS
    vel = np.concatenate([[vel[0]], vel])
    vel[~visible] = np.nan

f_interp = interp1d(ft_aligned, vel, kind='linear',
                    bounds_error=False, fill_value=np.nan)
tongue_vel_aligned[:, ti] = f_interp(taxis)
```

iii. Step 120 is the main justification: the agent says tongue visibility should be confidence-gated, tongue gaps should not be filled, and invisible bins should become `not_visible`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes a per-session 50th percentile threshold from visible, non-NaN tongue velocities on valid trials. Visible bins below threshold are class `0`, visible bins at/above threshold are class `1`, and invisible bins stay class `2`.

ii.
```python
t_vals = tongue_vel_aligned[:, valid_trials][tongue_visible[:, valid_trials]]
t_vals = t_vals[~np.isnan(t_vals)]
tongue_thresh = np.percentile(t_vals, 50) if len(t_vals) > 0 else 0

tongue_disc = np.full(n_timebins, 2, dtype=np.int64)
vis = tongue_visible[:, ti] & ~np.isnan(tongue_vel_aligned[:, ti])
if vis.any():
    tongue_disc[vis & (tongue_vel_aligned[:, ti] < tongue_thresh)] = 0
    tongue_disc[vis & (tongue_vel_aligned[:, ti] >= tongue_thresh)] = 1
```

iii. Step 120 justifies the third class as a visibility class rather than as low velocity.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The script subtracts the session video offset and the trial's `goCue` from frame times, then interpolates the velocity trace and the visibility mask onto the same `taxis` vector used by the neural data.

ii.
```python
ft_aligned = ft - vidshift - goCue[ti]
f_interp = interp1d(ft_aligned, vel, kind='linear',
                    bounds_error=False, fill_value=np.nan)
v = f_interp(taxis)
f_vis = interp1d(ft_aligned, vis_float, kind='nearest',
                 bounds_error=False, fill_value=0)
```

iii. In step 64 the agent notes the video offset is about `0.49 s`; step 60 says it will apply the video offset and align everything to go cue onset.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera's `top_paw` coordinates and confidence, plus `frameTimes`, `vidshift`, and `goCue`.

ii.
```python
paw_idx = bottom_feat_names.index('top_paw') if 'top_paw' in bottom_feat_names else None
...
paw_xy = sess['trial_paw_xy'][ti]
paw_conf = sess['trial_paw_conf'][ti]
vel, vis = compute_velocity(paw_xy, confidence=paw_conf, fill_missing=True)
```

iii. The trajectory does not give a separate paw-specific rationale; the choice is implicit in the bottom-camera-only loading code.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw speed is computed as frame-to-frame Euclidean velocity at 400 Hz. Unlike tongue, missing coordinates are nearest-filled before differencing; after alignment to `taxis`, remaining NaNs are also filled with nearest values.

ii.
```python
vel, vis = compute_velocity(paw_xy, confidence=paw_conf, fill_missing=True)
...
f_interp = interp1d(ft_aligned, vel, kind='linear',
                    bounds_error=False, fill_value=np.nan)
paw_vel_aligned[:, ti] = v
...
fill_nearest(paw_vel_aligned)
```

iii. In step 120 the agent says it wants the same confidence-based visibility rule for paw, but unlike tongue it keeps the "fill missing" behavior for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A per-session median threshold is computed from visible, non-NaN paw velocities on valid trials. Visible bins are split into classes `0/1`, and bins outside the visibility mask remain class `2`.

ii.
```python
p_vals = paw_vel_aligned[:, valid_trials][paw_visible[:, valid_trials]]
p_vals = p_vals[~np.isnan(p_vals)]
paw_thresh = np.percentile(p_vals, 50) if len(p_vals) > 0 else 0

paw_disc = np.full(n_timebins, 2, dtype=np.int64)
vis = paw_visible[:, ti] & ~np.isnan(paw_vel_aligned[:, ti])
```

iii. No separate thresholding argument appears in the trajectory; the implementation follows the prompt's median split idea.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The script uses the same `ft_aligned = frameTimes - vidshift - goCue` rule as for tongue and then interpolates onto the neural time grid.

ii.
```python
ft_aligned = ft - vidshift - goCue[ti]
f_interp = interp1d(ft_aligned, vel, kind='linear',
                    bounds_error=False, fill_value=np.nan)
paw_vel_aligned[:, ti] = v
```

iii. The trajectory only gives the general alignment rationale from steps 60 and 64.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from `motionEnergy_<anm>_<date>.mat`, primarily from the `me['data']` field, with a fallback HDF5 reader. The session also stores `moveThresh` if present, but that threshold is not used downstream.

ii.
```python
me_mat = sio.loadmat(me_fpath, squeeze_me=True)
me = me_mat['me']
me_data_raw = me['data'].item()
sess['me_thresh'] = float(me['moveThresh'].item())
sess['me_data'] = [me_data_raw[i] for i in range(len(me_data_raw))]
```

iii. The trajectory does not discuss motion-energy loading in detail; the code shows the intended source.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates the per-frame motion-energy trace onto `taxis`, fills remaining NaNs with nearest values, and then discretizes by the per-session median.

ii.
```python
f_interp = interp1d(ft_aligned, me_trial, kind='linear',
                    bounds_error=False, fill_value=np.nan)
me_aligned[:, ti] = f_interp(taxis)
...
fill_nearest(me_aligned)
...
me_thresh = np.percentile(m_vals, 50)
```

iii. There is no explicit trajectory justification beyond the general decision to align all time-varying signals to the neural grid.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The script computes a session-level 50th percentile over non-NaN aligned motion-energy values. Bins below threshold are class `0`, bins at/above threshold are class `1`, and bins with no motion-energy value remain class `2`.

ii.
```python
me_disc = np.full(n_timebins, 2, dtype=np.int64)
if me_available:
    valid_me = ~np.isnan(me_aligned[:, ti])
    if valid_me.any():
        me_disc[valid_me & (me_aligned[:, ti] < me_thresh)] = 0
        me_disc[valid_me & (me_aligned[:, ti] >= me_thresh)] = 1
```

iii. No separate trajectory justification is given; this is an implementation of the prompt's per-session median split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the same `trial_frame_times` array used for the bottom-camera trajectory data, after subtracting `vidshift` and `goCue`, and then interpolated onto the neural time axis.

ii.
```python
ft = sess['trial_frame_times'][ti]
ft_aligned = ft - vidshift - goCue[ti]
...
if len(me_trial) == len(ft):
    f_interp = interp1d(ft_aligned, me_trial, kind='linear',
                        bounds_error=False, fill_value=np.nan)
```

iii. The trajectory only provides the general video-offset rationale from steps 60 and 64; it does not justify reusing bottom-camera frame times for motion energy.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data by mixing masking and nearest-fill. Tongue invisibility is tracked from DLC confidence and mapped to class `2`. Paw and motion-energy NaNs are filled by nearest values. Empty frame-time arrays are silently skipped, and many load/alignment failures are swallowed by broad `except Exception` blocks.

ii.
```python
if len(ft) == 0:
    continue
...
fill_nearest(paw_vel_aligned)
fill_nearest(me_aligned)
...
except Exception:
    pass
```

iii. Step 120 explicitly justifies confidence-gating tongue visibility and not filling tongue gaps. The rest of the missing-data handling is only implicit in the implementation.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are the nested unit-by-trial spike histogramming/smoothing loop and the per-trial interpolation of tongue, paw, and motion-energy traces. The script does not contain any evidence that file I/O was treated as the main bottleneck.

ii.
```python
for ui in range(n_units):
    ...
    for ti in range(ntrials):
        counts, _ = np.histogram(aligned_times, bins=edges)
        trialdat[:, ui, ti] = smooth_data(fr, SMOOTH_WINDOW, BC_TYPE)

for ti in range(ntrials):
    ...
    f_interp = interp1d(ft_aligned, vel, kind='linear', ...)
```

iii. The trajectory does not discuss runtime hotspots; this is inferred from the implementation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: the `for ui` / `for ti` neural binning loops, the per-column and per-NaN loops in `fill_nearest`, and the per-trial interpolation loops for video and motion energy.

ii.
```python
for ui in range(n_units):
    for ti in range(ntrials):
        ...

for col in range(arr.shape[1]):
    fill_nearest(arr[:, col])

for ti in range(ntrials):
    ...
    f_interp = interp1d(...)
```

iii. No vectorization rationale appears in the trajectory.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats spike binning independently for every unit and every trial, recomputes interpolators trial-by-trial for tongue, paw, and motion energy, and performs nearest-filling in Python loops. Unlike the reference, it does not share a single vectorized spike-counting pass across all trials of a unit.

ii.
```python
for ui in range(n_units):
    for ti in range(ntrials):
        spk_mask = spike_trials_raw == (ti + 1)
        ...

for ti in range(ntrials):
    f_interp = interp1d(ft_aligned, vel, kind='linear', ...)
```

iii. The trajectory does not acknowledge this repeated work; the script mostly just claims to follow the reference pipeline.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads or stores several things it never uses downstream: `L`, `no`, raw `qualities`, and `me_thresh` from the motion-energy file. It also fills paw and motion-energy gaps before only retaining discretized categories, so the exact filled continuous values are discarded once thresholds are applied.

ii.
```python
'qualities': all_qualities,
...
'me_thresh': None
...
L = np.array(bp['L']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
...
fill_nearest(paw_vel_aligned)
fill_nearest(me_aligned)
```

iii. The trajectory does not offer a justification for these extra computations or stored fields.
