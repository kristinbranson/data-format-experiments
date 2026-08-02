# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent did not reproduce the reference code's `meta = load<animal>_ALMVideo(...)` workflow. Instead, it crawls the two data directories, takes every `data_structure_*.mat` file it finds, skips animals whose name starts with `MAH`, infers probe IDs by regex-parsing the MATLAB loading scripts, auto-detects MATLAB v7.3 HDF5 versus v5 `.mat`, and then loads each session file with either `h5py` or `scipy.io.loadmat`.

ii. Code snippets:
```python
for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
    ...
    for fn in sorted(os.listdir(data_dir)):
        if not fn.startswith('data_structure_'):
            continue
        ...
        sessions.append({
            'animal': animal,
            'date': date,
            'filepath': filepath,
            'probes': probes,
            'me_filepath': me_filepath if has_me else None,
            'data_dir': data_dir
        })
```

```python
def load_session_data(filepath):
    fmt = detect_mat_format(filepath)
    if fmt == 'hdf5':
        return load_session_data_hdf5(filepath)
    elif fmt == 'v5':
        return load_session_data_v5(filepath)
```

iii. In `CONVERSION_NOTES.md` Step 4 the agent explicitly says it "Included all available." The trajectory also shows the same rationale: it compared directory contents to the paper/code and chose to include all available files rather than only the sessions explicitly named by the reference loaders.

## 1-b. How are the data split into subjects?

i. Subjects are defined purely by the animal name in each included session result. After processing all sessions, the code takes the unique animal IDs, sorts them, and creates `subject_idx` by looking up each session's animal in that sorted list.

ii. Code snippets:
```python
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The notes repeatedly refer to "animals" as the subject unit and compare subject counts against the paper. There is no more elaborate justification than treating each animal code as one mouse.

## 1-c. How are the data split into sessions?

i. Each `data_structure_ANIMAL_DATE.mat` file is treated as one session. If the reference loading script listed two probes for that session, the agent concatenates the processed neural data from both probes into one session entry.

ii. Code snippets:
```python
m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
...
sessions.append({
    'animal': animal,
    'date': date,
    'filepath': filepath,
    'probes': probes,
```

```python
all_firing_rates = []
for p in probes:
    ...
    if fr is not None and len(kept) > 0:
        all_firing_rates.append(fr)
...
firing_rates = np.concatenate(all_firing_rates, axis=0)
```

iii. `CONVERSION_NOTES.md` Step 1 says "Two probes possible for some sessions (JEB15)," and the trajectory shows the agent intentionally mapping one file to one session while combining probes within that session.

## 1-d. How are the data split into trials?

i. Trials are defined by the raw behavioral trial count `bp['Ntrials']` and by the per-unit `trial` labels stored with spikes. The code processes spike data trial-by-trial for trial numbers `1..Ntrials`, then later builds trial-level neural/input/output entries for the subset of trials that pass trial filtering.

ii. Code snippets:
```python
Ntrials = session['bp']['Ntrials']
...
for j in range(Ntrials):
    trial_num = j + 1  # MATLAB 1-indexed
    spk_mask = trial_nums == trial_num
```

```python
trial_indices = np.where(valid_trials)[0]
...
for trial_idx in trial_indices:
    neural = firing_rates[:, :, trial_idx].astype(np.float32)
    ...
    output_trials.append(out)
```

iii. The justification is implicit: the raw files already organize behavior and spikes by trial. The trajectory shows the agent exploring `obj.bp.Ntrials`, `obj.clu[*].trial`, and then mirroring that organization.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies two trial/session-level filters. First, it excludes sessions unless they have more than 40 right-hit DR trials and more than 40 left-hit DR trials, using `R/L & hit & ~stim & ~autowater & ~early`. Second, within included sessions it keeps trials satisfying `~early & ~stim_enable & ~no`, so early-lick, stimulation, and ignore trials are removed but miss trials are retained.

ii. Code snippets:
```python
r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
return n_r > MIN_HIT_TRIALS and n_l > MIN_HIT_TRIALS, n_r, n_l
```

```python
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
trial_indices = np.where(valid_trials)[0]
```

iii. `CONVERSION_NOTES.md` Step 3 says "Trial curation: Exclude early lick and stim trials, session needs >40 R/L hit DR." The trajectory likewise cites `UseInclusionCritera.m` and the methods text about omitting early and ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from per-cluster spike assignments in `obj.clu`: specifically each unit's `trial`, `trialtm`, and `quality`, together with `bp.ev.goCue` for alignment.

ii. Code snippets:
```python
unit['trial'] = np.array(f[clu_group['trial'][u, 0]]).flatten().astype(int)
unit['trialtm'] = np.array(f[clu_group['trialtm'][u, 0]]).flatten()
unit['quality'] = read_h5_string(f, clu_group['quality'][u, 0])
```

```python
goCue = session['bp']['ev']['goCue']
trial_nums = unit['trial']
trialtm = unit['trialtm']
```

iii. The notes identify the same pipeline explicitly: "`obj.clu spike times -> neural`" and "alignSpikes: trialtm_aligned = trialtm - goCue(trial)."

## 2-b. How is the `neural` data processed?

i. For each retained unit and trial, the code subtracts that trial's go-cue time from `trialtm`, bins aligned spikes into 5 ms bins, converts counts to Hz by dividing by `DT`, and applies a causal Gaussian smoothing kernel of width 15 bins.

ii. Code snippets:
```python
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float64) / DT
fr_smooth = smooth_signal(fr, SMOOTH_N)
```

```python
DT = 1.0 / 200.0
SMOOTH_N = 15
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 10 say the agent intentionally matched `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered in two passes. First, any unit whose `quality` string contains `"garbage"` is dropped. Second, after binning/smoothing, the code computes each unit's mean firing rate over all time bins and all trials and keeps only units with mean FR greater than `0.5` Hz. Finally, sessions with fewer than 10 kept units are excluded.

ii. Code snippets:
```python
for u_idx, unit in enumerate(units):
    quality = unit['quality'].lower().strip()
    if 'garbage' not in quality:
        valid_units.append(u_idx)
```

```python
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR
...
if n_neurons < MIN_UNITS:
    return None
```

iii. The notes justify this by citing `deleteGarbageClu.m`, `removeLowFRClusters.m`, `lowFR=0.5`, and the paper's "at least 10 units" session rule. The trajectory shows the agent chose the default `0.5` Hz threshold from `getDefaultParams.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to go-cue onset on a per-trial basis by subtracting `goCue[j]` from each spike's `trialtm` before binning.

ii. Code snippets:
```python
goCue = session['bp']['ev']['goCue']
...
aligned_times = trialtm[spk_mask] - goCue[j]
```

iii. The notes repeatedly say the alignment event is `goCue`, directly citing `getDefaultParams.m` and `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses `DT = 1/200 = 0.005` s, i.e. 5 ms bins. The code does not do any further rebinning after that; all neural, input, and video-derived outputs are put directly onto this grid. The actual saved grid is `-2.5:0.005:2.5`, which yields 1001 saved samples.

ii. Code snippets:
```python
DT = 1.0 / 200.0  # 5ms time bins
TMIN = -2.5
TMAX = 2.5
```

```python
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = time_axis[time_axis <= TMAX + 1e-10]
```

iii. The notes justify 5 ms by citing `getDefaultParams.m`, but they also explicitly state the output has "1001 bins at 5ms," showing the agent believed the inclusive edge grid was the correct final axis.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a trial-varying raw field. The code uses a session-independent synthetic time axis created from `TMIN`, `TMAX`, and `DT`; go cue is treated as time zero because all streams are aligned to go cue beforehand.

ii. Code snippets:
```python
def make_time_axis():
    time_axis = np.arange(TMIN, TMAX + DT/2, DT)
```

```python
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The trajectory says "Input: time from goCue (same for all trials)," which is the entire justification the agent gives.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code simply creates the 1D `time_axis`, casts it to `float32`, reshapes it to `(1, n_timepoints)`, and reuses that exact array for every trial.

ii. Code snippets:
```python
time_axis = make_time_axis()
...
inp = time_axis.astype(np.float32).reshape(1, -1)  # (1, n_timepoints)
input_trials.append(inp)
```

iii. There is no deeper justification in the notes beyond "time from goCue as continuous variable."

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is aligned by construction: the same `time_axis` is used both for spike binning and for the saved input vector, so the input index is intended to correspond one-to-one with the neural time index.

ii. Code snippets:
```python
edges = np.append(time_axis, time_axis[-1] + DT)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The agent's notes say "Input data: time axis is [-2.5, 2.5] with 1001 bins at 5ms," indicating that it believed identical reuse of this axis was the correct alignment strategy.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the behavioral trial-type flags `bp['R']` and `bp['L']`.

ii. Code snippets:
```python
bp['L'] = np.array(f['obj/bp/L']).flatten().astype(bool)
bp['R'] = np.array(f['obj/bp/R']).flatten().astype(bool)
```

```python
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. The notes say "`obj.bp.R -> lick_direction` with `R=1, L=0`."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each kept trial, the code assigns `1` if the trial is marked `R`; otherwise it assigns `0`. That scalar is then repeated across all time bins in the saved `output` array.

ii. Code snippets:
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
...
out[0, :] = lick_dir
```

iii. The justification in the notes is just the label mapping "`R=1, L=0`."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `bp['autowater']` trial flag.

ii. Code snippets:
```python
bp['autowater'] = np.array(f['obj/bp/autowater']).flatten().astype(bool)
...
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. The notes explicitly map `autowater` to context and interpret `autowater=1` as WC.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater=True` to `0` (WC) and `autowater=False` to `1` (DR), then repeats that trial label across time.

ii. Code snippets:
```python
context = 0 if bp['autowater'][trial_idx] else 1
...
out[1, :] = context
```

iii. `CONVERSION_NOTES.md` Step 5 says "`obj.bp.autowater -> behavioral_context`" with "`WC=0, DR=1`."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `bp['hit']` flag, after ignore trials have already been excluded from the kept-trial set.

ii. Code snippets:
```python
bp['hit'] = np.array(f['obj/bp/hit']).flatten().astype(bool)
bp['no'] = np.array(f['obj/bp/no']).flatten().astype(bool)
...
valid_trials = valid_trials & ~bp['no']
```

```python
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. The notes map `hit` to outcome and say ignore trials are omitted from analyses.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Each kept trial gets `1` if `hit` is true and `0` otherwise; since `no` trials were filtered out, the remaining `0` cases are effectively incorrect/miss trials. The scalar is then repeated across time.

ii. Code snippets:
```python
outcome = 1 if bp['hit'][trial_idx] else 0
...
out[2, :] = outcome
```

iii. The agent's justification is the decoder-task mapping "`incorrect=0, correct=1`."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DeepLabCut trajectory data in `session['traj'][0]`, specifically the `ts` tensor for the feature named `"tongue"`, together with `frameTimes`, `NdroppedFrames`, `bp.ev.goCue`, and the video-neural offset computed from `bp.ev.bitStart` and `sglx.bitcode.bitstart`.

ii. Code snippets:
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
```

```python
feat_names = cam_data['feat_names']
...
x = ts[feat_idx, 0, :]
y = ts[feat_idx, 1, :]
aligned_times = frame_times - vidshift - goCue[trix]
```

iii. The notes say the agent chose "tongue feature from camera 0 (x,y velocity magnitude)." The trajectory records that camera-0 `tongue` was a deliberate choice for the decoder output.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code locates the `"tongue"` feature in side-view DLC data, interpolates x/y position onto the neural time axis aligned to go cue, computes x/y velocity with `np.gradient`, sets NaN tongue velocities to zero when the tongue is not visible, and then converts x/y velocities into a scalar magnitude with `sqrt(xvel**2 + yvel**2)`.

ii. Code snippets:
```python
f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
               bounds_error=False, fill_value=np.nan)
f_y = interp1d(aligned_times[valid], y[valid], kind='linear',
               bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

```python
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
...
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
...
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The justification given in the trajectory is that the task asked for a single tongue-velocity output, so the agent chose the camera-0 tongue feature and used velocity magnitude after matching the reference interpolation and derivative steps.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code pools all tongue-velocity samples from valid trials within a session, takes the 50th percentile as the per-session threshold, assigns `0` below threshold and `1` at or above threshold, and forces NaNs to category `0`.

ii. Code snippets:
```python
tongue_valid = tongue_vel[:, trial_indices]
tongue_thresh = np.nanpercentile(
    tongue_valid[~np.isnan(tongue_valid)], 50
) if np.any(~np.isnan(tongue_valid)) else 0
```

```python
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. The trajectory shows the agent wrestling with the fact that tongue thresholds often became `0` because the tongue was not visible much of the time, but it kept the percentile-based rule because the task specification explicitly required a session-wise 50th percentile threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is aligned by computing `aligned_times = frame_times - vidshift - goCue[trial]` and interpolating onto the same `time_axis` used for neural binning.

ii. Code snippets:
```python
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

iii. The notes cite `findPosition.m` and `findVelocity.m` as the intended reference behavior, and the trajectory states the goal was to "interpolate DLC positions to neural time axis aligned to goCue."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DeepLabCut trajectory data in `session['traj'][1]`, using the feature named `"top_paw"` plus `frameTimes`, `NdroppedFrames`, `bp.ev.goCue`, and the video-neural offset.

ii. Code snippets:
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

```python
x = ts[feat_idx, 0, :]
y = ts[feat_idx, 1, :]
aligned_times = frame_times - vidshift - goCue[trix]
```

iii. The trajectory explicitly says: "Paw velocity: use top_paw from camera 1 (x,y velocity magnitude)." That is the only justification given for choosing one paw feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code interpolates side-specific paw x/y positions onto the neural time axis, fills missing positions with nearest values, computes x/y velocity by `np.gradient`, subtracts a per-trial baseline derivative for non-tongue features, fills missing derivatives with nearest values, and converts x/y velocity to a scalar magnitude.

ii. Code snippets:
```python
if not is_tongue:
    mask_nan = np.isnan(xpos[:, trix])
    if np.any(mask_nan) and not np.all(mask_nan):
        xpos[:, trix] = _fill_nearest(xpos[:, trix])
        ypos[:, trix] = _fill_nearest(ypos[:, trix])
```

```python
base_x = np.nanmedian(np.diff(xpos[:, trix]))
base_y = np.nanmedian(np.diff(ypos[:, trix]))
xv = xv - base_x
yv = yv - base_y
...
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes justify the interpolation and derivative steps by citing `findPosition.m` and `findVelocity.m`; the magnitude reduction is the agent's own choice for producing one decoder output.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The code takes the session-wide 50th percentile of all paw-velocity samples from valid trials, then bins each timepoint as `0` below threshold or `1` at/above threshold, with NaNs forced to `0`.

ii. Code snippets:
```python
paw_valid = paw_vel[:, trial_indices]
paw_thresh = np.nanpercentile(
    paw_valid[~np.isnan(paw_valid)], 50
) if np.any(~np.isnan(paw_valid)) else 0
```

```python
pv_disc = np.zeros(len(time_axis), dtype=np.int64)
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. The justification is the decoder-task instruction requiring a per-session 50th percentile split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. It is aligned using `frame_times - vidshift - goCue[trial]` and interpolation to the shared `time_axis`.

ii. Code snippets:
```python
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

iii. The notes say the agent matched `findPosition.m` and therefore reused the same neural-aligned time base for paw kinematics.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from a separate `motionEnergy_ANIMAL_DATE.mat` file. The code reads `me.data` trial traces, optionally `moveThresh`, and uses side-camera `frameTimes` plus `bp.ev.goCue`, `bp.ev.bitStart`, and `sglx.bitcode.bitstart` to align the traces.

ii. Code snippets:
```python
me_fn = fn.replace('data_structure_', 'motionEnergy_')
me_filepath = os.path.join(data_dir, me_fn)
```

```python
me_data = load_motion_energy(sess_info['me_filepath'])
...
me_aligned = align_motion_energy(me_data, session, time_axis, vidshift)
```

iii. The notes explicitly cite `loadMotionEnergy.m` and `findVideoOffset.m` as the reference for this decision.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code loads several possible MATLAB layout variants for `me`, unwraps nested `me.data` structures when present, extracts each trial's 1D motion-energy trace, interpolates each trace onto the shared neural time axis using frame times aligned to go cue, and fills missing values with nearest samples.

ii. Code snippets:
```python
if me_raw.dtype.names and 'data' in me_raw.dtype.names:
    me_data_field = me_raw['data'][0, 0]
    if hasattr(me_data_field, 'dtype') and me_data_field.dtype.names and 'data' in me_data_field.dtype.names:
        me_data_arr = me_data_field['data'][0, 0]
```

```python
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
                bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_me(time_axis)
...
me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. The trajectory shows the agent debugging multiple `me` format variants and explicitly trying to match the reference code's special-case handling of nested `me.data`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The continuous aligned motion-energy samples from valid trials are thresholded at the within-session 50th percentile; values below are `0`, values at/above are `1`, and NaNs are forced to `0`.

ii. Code snippets:
```python
me_valid = me_aligned[:, trial_indices]
me_thresh = np.nanpercentile(
    me_valid[~np.isnan(me_valid)], 50
) if np.any(~np.isnan(me_valid)) else 0
```

```python
me_disc = np.zeros(len(time_axis), dtype=np.int64)
me_disc[me >= me_thresh] = 1
me_disc[np.isnan(me)] = 0
```

iii. The notes state this mapping directly in Step 5, and the trajectory says the manual `moveThresh` from the paper was not used because the decoder task overrode it with percentile binning.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Each trial's motion-energy trace is interpolated to the same neural `time_axis` after subtracting the video start offset and that trial's `goCue` time.

ii. Code snippets:
```python
aligned_times = frame_times - vidshift - goCue[trix]
...
me_aligned[:, trix] = f_me(time_axis)
```

iii. The justification in the notes is that this matches `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses broad fallback handling throughout. Missing `reward`/`bitStart` become NaN vectors, missing `sglx` makes `vidshift=0`, invalid probes become empty probe lists, missing or malformed motion-energy files produce all-NaN motion energy, missing frame times fall back to a synthetic 400 Hz grid, missing non-tongue positions/velocities are nearest-filled, tongue NaN velocities are set to zero, and NaN output values are ultimately forced into the `0` category. Trials with all-zero neural matrices are not removed.

ii. Code snippets:
```python
except:
    bp['ev']['reward'] = np.full(bp['Ntrials'], np.nan)
...
if session['sglx'] is None:
    return 0.0
```

```python
except Exception as e:
    print(f'  Warning: Could not load motion energy: {e}')
    return None
```

```python
tv_disc[np.isnan(tv)] = 0
pv_disc[np.isnan(pv)] = 0
me_disc[np.isnan(me)] = 0
```

iii. The notes frame these as edge-case handling for mixed MATLAB formats, bad probes, and missing video data. The trajectory explicitly mentions "Handle probes with invalid clu data," "Handled MATLAB v5 vs v7.3," and "all-zero neural data" warnings that were left in place.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are the nested neural loop in `process_spikes` (unit by trial histogramming and smoothing), the per-trial interpolation and gradient computation in `extract_feature_velocity`, and motion-energy interpolation in `align_motion_energy`. Those are the obvious hotspots in both asymptotic work and the agent's runtime logs.

ii. Code snippets:
```python
for i, u_idx in enumerate(valid_units):
    ...
    for j in range(Ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = smooth_signal(fr, SMOOTH_N)
```

```python
for trix in range(Ntrials):
    ...
    f_x = interp1d(...)
    f_y = interp1d(...)
    ...
    xv = np.gradient(xpos[:, trix])
```

iii. The agent's notes estimate "~8s per session" and the trajectory repeatedly highlights spike processing, tongue velocity, paw velocity, and motion energy as the major timed stages.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit/per-trial spike loop, the per-sample nearest-fill loop in `_fill_nearest`, the per-trial gradient/fill loops in `extract_feature_velocity`, and the final per-trial output-building loop could all have been made more vectorized.

ii. Code snippets:
```python
for i, u_idx in enumerate(valid_units):
    ...
    for j in range(Ntrials):
```

```python
for i in range(len(arr)):
    if mask[i]:
        dists = np.abs(valid - i)
        arr[i] = arr[valid[np.argmin(dists)]]
```

```python
for trial_idx in trial_indices:
    ...
    tv_disc = np.zeros(len(time_axis), dtype=np.int64)
    ...
    output_trials.append(out)
```

iii. There is no explicit justification in the notes because the agent was optimizing for correctness, not speed. This is a direct reading of the code structure.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly constructs identical per-trial input time vectors, repeatedly searches feature-name lists, repeatedly runs nearest-fill logic for each trial and signal, repeatedly discretizes three output streams with nearly identical code, and repeats format-detection/fallback logic across session files.

ii. Code snippets:
```python
inp = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(inp)
```

```python
for i, name in enumerate(feat_names):
    if name.lower() == feat_name.lower():
        feat_idx = i
        break
```

```python
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
...
pv_disc = np.zeros(len(time_axis), dtype=np.int64)
...
me_disc = np.zeros(len(time_axis), dtype=np.int64)
```

iii. This is not justified in the notes; it is simply how the final script was written.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads many raw fields it never uses for the decoder (`sample`, `delay`, `reward`, absolute `tm`, some cluster metadata), loads full continuous tongue/paw/motion traces only to immediately threshold them to binary categories, reads motion-energy `moveThresh` but ignores it for the saved output, and expands per-trial scalar labels (`lick_direction`, `behavioral_context`, `outcome`) into full-length constant time series.

ii. Code snippets:
```python
bp['ev']['sample'] = np.array(f['obj/bp/ev/sample']).flatten()
bp['ev']['delay'] = np.array(f['obj/bp/ev/delay']).flatten()
...
unit['tm'] = np.array(f[clu_group['tm'][u, 0]]).flatten()
```

```python
me_thresh = float(me_raw['moveThresh'][0, 0].flatten()[0])
...
me_disc[me >= me_thresh] = 1
```

```python
out[0, :] = lick_dir
out[1, :] = context
out[2, :] = outcome
```

iii. The notes do not discuss these as unnecessary; they are visible only from the implementation.
