# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `data/Ephys_Behavior` and `data/RandomizedDelay_Ephys_Behavior` for every `data_structure_*.mat` file, parses animal/date from filenames, looks up probe numbers from the MATLAB loading scripts when available, and auto-detects each `.mat` as HDF5 (`v7.3`) or MATLAB v5. It separately loads motion-energy files when present.

ii.
```python
def load_session_data(filepath):
    fmt = detect_mat_format(filepath)
    if fmt == 'hdf5':
        return load_session_data_hdf5(filepath)
    elif fmt == 'v5':
        return load_session_data_v5(filepath)

for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
    for fn in sorted(os.listdir(data_dir)):
        if not fn.startswith('data_structure_'):
            continue
        ...
        probes = probe_map.get((animal, date), [1])
        sessions.append({...})
```

iii. `CONVERSION_NOTES.md` says the agent intentionally implemented a dual-format loader and reconciled discrepancies by including available data even when the paper counts differed. The trajectory also shows it added loading-script parsing mainly to recover probe mappings.

## 1-b. How are the data split into subjects (mice)?

i. Each session inherits its subject from the `animal` token parsed from the filename. At the end, subjects are the sorted unique animal IDs, and `subject_idx` maps each included session back to that subject list.

ii.
```python
animal = m.group(1)
...
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The notes treat each animal name as one mouse and report subject counts by those IDs.

## 1-c. How are the data split into sessions?

i. The script treats each `data_structure_ANIMAL_DATE.mat` file as one session. Session order is just the sorted file order across the two neural-data directories, after excluding sessions that fail later checks.

ii.
```python
for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
    for fn in sorted(os.listdir(data_dir)):
        if not fn.startswith('data_structure_'):
            continue
        m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
        ...
        sessions.append({
            'animal': animal,
            'date': date,
            'filepath': filepath,
            'probes': probes,
        })
```

iii. In the trajectory, the agent noticed mismatches between the loading scripts and the file tree, then chose to iterate over the available files rather than strictly follow the reference session lists.

## 1-d. How are the data split into trials?

i. Trial count comes from `obj.bp.Ntrials`. Neural and video processing are computed with one trial index per Bpod trial, and later only `trial_indices = np.where(valid_trials)[0]` are exported.

ii.
```python
Ntrials = session['bp']['Ntrials']
...
for j in range(Ntrials):
    trial_num = j + 1
    spk_mask = trial_nums == trial_num
...
trial_indices = np.where(valid_trials)[0]
for trial_idx in trial_indices:
    neural = firing_rates[:, :, trial_idx].astype(np.float32)
```

iii. The notes describe `obj.bp` as the master trial structure and say the converted output is organized trial-by-trial within each session.

## 1-e. How are trials filtered based on quality controls?

i. Exported trials are restricted to trials that are not early licks, not stimulation trials, and not ignore (`no`) trials. Separately, a session is excluded unless it has more than 40 right-hit DR trials and more than 40 left-hit DR trials.

ii.
```python
r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
...
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
```

iii. `CONVERSION_NOTES.md` says this was meant to match `UseInclusionCritera.m` plus the methods text stating early-lick and ignore trials were omitted from analyses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from sorted spike fields in `obj.clu`: primarily `trial`, `trialtm`, and `quality`, together with `obj.bp.ev.goCue` for alignment.

ii.
```python
unit['trial'] = ...
unit['trialtm'] = ...
unit['quality'] = ...
...
goCue = session['bp']['ev']['goCue']
trial_nums = unit['trial']
trialtm = unit['trialtm']
```

iii. The notes explicitly map `obj.clu` spike times to `neural`, and cite `alignSpikes.m`, `getSeq.m`, and cluster-quality utilities as the reference path.

## 2-b. How is the `neural` data processed?

i. For each kept unit and each trial, the script subtracts that trial’s go-cue time from spike times, bins spikes into 5 ms bins over `[-2.5, 2.5]`, converts counts to firing rate by dividing by `DT`, and applies a causal Gaussian smoother of width 15 bins.

ii.
```python
edges = np.append(time_axis, time_axis[-1] + DT)
...
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float64) / DT
fr_smooth = smooth_signal(fr, SMOOTH_N)
```

iii. The notes say this is intended to match `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`, with `dt=1/200`, `tmin=-2.5`, `tmax=2.5`, and `smooth=15`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script drops units whose `quality` string contains `garbage`, then removes units whose mean firing rate is `<= 0.5 Hz`, and finally drops sessions with fewer than 10 remaining units.

ii.
```python
for u_idx, unit in enumerate(units):
    quality = unit['quality'].lower().strip()
    if 'garbage' not in quality:
        valid_units.append(u_idx)
...
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR
...
if n_neurons < MIN_UNITS:
    return None
```

iii. The notes justify this as matching `deleteGarbageClu.m`, `removeLowFRClusters.m`, and the methods statement that sessions needed at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each unit’s spike times are aligned by subtracting the per-trial go-cue timestamp, so time zero is go-cue onset.

ii.
```python
goCue = session['bp']['ev']['goCue']
aligned_times = trialtm[spk_mask] - goCue[j]
```

iii. The notes repeatedly state that temporal alignment was set to `goCue`, following both the instructions and `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 5 ms bins (`DT = 1/200`) over 1001 time points from `-2.5` s to `2.5` s. No later rebinning is applied.

ii.
```python
DT = 1.0 / 200.0
TMIN = -2.5
TMAX = 2.5
...
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
```

iii. The notes say this was copied from `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not loaded from a raw continuous channel. It is a synthetic time axis defined by fixed parameters and anchored conceptually to `obj.bp.ev.goCue`.

ii.
```python
DT = 1.0 / 200.0
TMIN = -2.5
TMAX = 2.5
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes map the decoder input to “time from goCue” and say the same axis is used for every trial.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The processing is just constructing a fixed `[-2.5, 2.5]` vector at 5 ms spacing and storing that same vector for each trial as a `(1, T)` array.

ii.
```python
def make_time_axis():
    time_axis = np.arange(TMIN, TMAX + DT/2, DT)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes justify this as the decoder’s required input rather than a field native to the source files.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_axis` used to bin neural spikes is also written into the input, so the input and neural data share the same sample grid and the same alignment event.

ii.
```python
time_axis = make_time_axis()
...
neural = firing_rates[:, :, trial_idx].astype(np.float32)
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes say the input axis was checked against the neural time axis as a sanity check.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the behavioral side labels in `obj.bp.R` and implicitly `obj.bp.L`.

ii.
```python
bp['L'] = ...
bp['R'] = ...
...
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. The notes map `obj.bp.R` directly to `lick_direction`, with `R=1` and left otherwise.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each kept trial, the script encodes right as `1`, left as `0`, then repeats that trial label across all time bins in the output matrix.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
...
out[0, :] = lick_dir
```

iii. The notes say this was a per-trial decoder target, but the code expands it to a time-varying constant because the output arrays are all stored as `(n_output, n_timepoints)`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, used as a proxy for whether a trial was water-cued (`WC`) or delayed-response (`DR`).

ii.
```python
bp['autowater'] = ...
...
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. The trajectory contains the tutorial comment that `obj.bp.autowater=1` “can be used as a proxy for obtaining water-cued blocks and delayed-response blocks of trials.”

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script binarizes context as `WC=0` when `autowater` is true and `DR=1` otherwise, then repeats that trial label across the whole time axis.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
...
out[1, :] = context
```

iii. `CONVERSION_NOTES.md` explicitly records this mapping in Step 5.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`; misses become the zero class because ignored and early trials are already excluded.

ii.
```python
bp['hit'] = ...
bp['miss'] = ...
bp['no'] = ...
...
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. The notes map `obj.bp.hit` directly to `outcome`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code encodes `correct=1` when `hit` is true, `incorrect=0` otherwise, and repeats that per-trial label across all time bins.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
...
out[2, :] = outcome
```

iii. The notes describe outcome as a per-trial categorical target.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity comes from DeepLabCut trajectories in `obj.traj`, camera 0, feature name `tongue`, using `ts`, `frameTimes`, and `NdroppedFrames`, plus `obj.bp.ev.goCue` and the SpikeGLX/bitcode timing fields used to compute video offset.

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
...
x = ts[feat_idx, 0, :]
y = ts[feat_idx, 1, :]
aligned_times = frame_times - vidshift - goCue[trix]
```

iii. The notes say tongue kinematics were intended to follow `getKinematicsFromVideo.m`, `findPosition.m`, and `findVelocity.m`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script interpolates tongue x/y coordinates onto the neural time axis, computes `np.gradient` on each coordinate, replaces tongue NaNs with zeros, and takes velocity magnitude `sqrt(xvel^2 + yvel^2)`.

ii.
```python
f_x = interp1d(aligned_times[valid], x[valid], ...)
f_y = interp1d(aligned_times[valid], y[valid], ...)
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes justify the approach as matching the reference kinematics code, and the trajectory shows the agent deliberately kept invisible-tongue periods at zero because it believed the MATLAB code did that too.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The threshold is the session-wide 50th percentile over all non-NaN tongue-velocity values from valid trials. Values `>= threshold` become class `1`; values below threshold are `0`; NaNs are forced to `0`.

ii.
```python
tongue_valid = tongue_vel[:, trial_indices]
tongue_thresh = np.nanpercentile(
    tongue_valid[~np.isnan(tongue_valid)], 50
) if np.any(~np.isnan(tongue_valid)) else 0
...
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. The trajectory shows the agent noticed that this made tongue class balance extremely skewed because the threshold was often `0`, considered changing the rule, and then explicitly kept `>=` because the task specification said “50th percentile” and `>=`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are timestamped by video frame times, shifted by the video-neural offset, centered on per-trial go cue, and linearly interpolated onto the same neural `time_axis`.

ii.
```python
vidshift = compute_video_offset(session)
...
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

iii. The notes say this was meant to match `findVideoOffset.m` and `findPosition.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DeepLabCut trajectories in camera 1, specifically the `top_paw` feature, with the same timing fields used for tongue velocity.

ii.
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. The notes only say “paw velocity”; the code makes the concrete choice to use `top_paw`, which is one of the camera-1 features present in the raw files.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script interpolates paw x/y to the neural axis, fills missing positions by nearest-neighbor carry, computes gradients, subtracts the median coordinate difference as a baseline derivative, fills missing velocity values, and takes velocity magnitude.

ii.
```python
if not is_tongue:
    xpos[:, trix] = _fill_nearest(xpos[:, trix])
    ypos[:, trix] = _fill_nearest(ypos[:, trix])
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
base_x = np.nanmedian(np.diff(xpos[:, trix]))
base_y = np.nanmedian(np.diff(ypos[:, trix]))
xv = _fill_nearest(xv - base_x)
yv = _fill_nearest(yv - base_y)
```

iii. The notes justify this as matching `findVelocity.m` for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Like tongue velocity, paw velocity is thresholded per session at the 50th percentile over all valid-trial values; `>= threshold` becomes `1`.

ii.
```python
paw_valid = paw_vel[:, trial_indices]
paw_thresh = np.nanpercentile(
    paw_valid[~np.isnan(paw_valid)], 50
) if np.any(~np.isnan(paw_valid)) else 0
...
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. The notes say the 50th-percentile split was chosen to satisfy the decoder task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned the same way as tongue trajectories: frame time minus video offset minus go cue, then interpolation to the neural `time_axis`.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

iii. The notes group this with the same video-alignment logic used for the tongue feature.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_ANIMAL_DATE.mat` file, specifically the `me.data` trial arrays, together with video frame times from `obj.traj`, go-cue times, and the computed video offset.

ii.
```python
data = sio.loadmat(me_filepath, squeeze_me=False)
me_raw = data['me']
...
me_trials.append(trial_me.flatten())
...
frame_times = trial_traj['frameTimes']
aligned_times = frame_times - vidshift - goCue[trix]
```

iii. The notes say this was intended to match `loadMotionEnergy.m`, and the trajectory shows the agent added support for a nested `me.data.data` case after seeing reference code comments.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The loader handles several `me` struct layouts, extracts one 1D motion-energy trace per trial, aligns it to go cue using video timing, linearly interpolates it to the neural axis, and nearest-fills remaining NaNs.

ii.
```python
if me_raw.dtype.names and 'data' in me_raw.dtype.names:
    me_data_field = me_raw['data'][0, 0]
    if ... and 'data' in me_data_field.dtype.names:
        me_data_arr = me_data_field['data'][0, 0]
...
f_me = interp1d(aligned_times[:n], trial_me[:n], ...)
me_aligned[:, trix] = f_me(time_axis)
...
me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. The trajectory shows the agent justified the nested-struct logic by pointing to the MATLAB line `if isstruct(me.data); me.data = me.data.data; end`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized at the session-wide median over valid trials, using `>= threshold` for class `1` and `0` otherwise, with NaNs mapped to `0`.

ii.
```python
me_valid = me_aligned[:, trial_indices]
me_thresh = np.nanpercentile(
    me_valid[~np.isnan(me_valid)], 50
) if np.any(~np.isnan(me_valid)) else 0
...
me_disc[me >= me_thresh] = 1
me_disc[np.isnan(me)] = 0
```

iii. The notes say the task specification, not the raw-file `moveThresh`, drove this discretization choice.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy traces are aligned by subtracting video offset and per-trial go cue from video frame times, then interpolating onto the same neural time grid.

ii.
```python
if trial_traj is not None and trial_traj['frameTimes'] is not None:
    frame_times = trial_traj['frameTimes']
    aligned_times = frame_times - vidshift - goCue[trix]
...
me_aligned[:, trix] = f_me(time_axis)
```

iii. The notes explicitly cite `loadMotionEnergy.m` and `findVideoOffset.m` for this alignment logic.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles many issues by silent fallbacks: missing reward/bitStart become NaN, missing `sglx` makes `vidshift=0`, missing trajectories or features propagate as NaN, missing motion-energy files create all-NaN arrays, and invalid HDF5 probe objects are skipped. At export time, NaN kinematic or motion-energy samples are usually forced into class `0`. Sessions that raise loader errors are skipped by the top-level `try/except`. All-zero neural trials are left in the final dataset.

ii.
```python
except:
    bp['ev']['reward'] = np.full(bp['Ntrials'], np.nan)
...
if session['sglx'] is None:
    return 0.0
...
if feat_idx is None:
    return np.full((len(time_axis), Ntrials), np.nan)
...
tv_disc[np.isnan(tv)] = 0
...
except Exception as e:
    print(f'  -> ERROR: {e}')
```

iii. The notes frame these as edge-case handling, mentioning invalid probes, v5/v7.3 differences, and sessions without `clu`; the verification log still reported zero-neural-data trials, which the agent chose not to fix.

## 11-a. What are the most time-consuming steps of the code?

i. The timing logs show spike processing is the main bottleneck, followed by file loading and then paw-velocity extraction. Tongue velocity and motion-energy alignment are relatively cheap.

ii.
```python
print(f'    Loaded in {t_load:.1f}s')
...
print(f'    Spike processing: {t_spk:.1f}s, {n_neurons} total units')
...
print(f'    Paw velocity: {t_paw:.1f}s')
print(f'    Motion energy: {t_me:.1f}s')
```

iii. `CONVERSION_NOTES.md` estimated about 8 s per session from these logs and called out spike processing as the expensive part.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the nested unit-by-trial spike-binning loop, the per-trial interpolation loops for tongue/paw/motion-energy alignment, and the quadratic nearest-fill helper that scans every NaN index against all valid indices.

ii.
```python
for i, u_idx in enumerate(valid_units):
    for j in range(Ntrials):
        counts, _ = np.histogram(...)

for trix in range(Ntrials):
    f_x = interp1d(...)
    f_y = interp1d(...)

for i in range(len(arr)):
    dists = np.abs(valid - i)
```

iii. The instructions asked the agent to vectorize bottlenecks; the notes mention timing analysis but do not claim these loops were actually optimized.

## 11-c. What processing does the code repeat multiple times?

i. It repeats the same interpolation/gradient pattern for tongue and paw, repeats nearest-fill operations on several signals, rebuilds the same input time vector for every trial, and processes all trials before later dropping invalid ones.

ii.
```python
tongue_vel = extract_feature_velocity(...)
paw_vel = extract_feature_velocity(...)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
...
for j in range(Ntrials):
    ...
trial_indices = np.where(valid_trials)[0]
```

iii. The trajectory shows the agent mostly focused on correctness and compatibility rather than de-duplicating these repeated steps.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script bins and smooths spikes for every Bpod trial before discarding early/stim/ignore trials, computes tongue/paw/motion-energy traces for those discarded trials too, and expands per-trial categorical labels into 1001-bin constant time series even though the label itself does not vary within a trial.

ii.
```python
for j in range(Ntrials):
    ...
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
...
out[0, :] = lick_dir
out[1, :] = context
out[2, :] = outcome
```

iii. The notes do not explicitly call these unnecessary, but they follow directly from the code structure and from the later filtering step.
