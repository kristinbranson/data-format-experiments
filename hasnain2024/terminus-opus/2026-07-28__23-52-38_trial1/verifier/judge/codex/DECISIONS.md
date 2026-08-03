# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers candidate sessions by scanning `data/Ephys_Behavior` and `data/RandomizedDelay_Ephys_Behavior` for every `data_structure_*.mat` file, excluding only animals whose names start with `MAH`. It regex-parses the MATLAB loading scripts to recover probe IDs when possible, otherwise defaults to probe 1. Each session file is then loaded with an auto-detected MATLAB reader (`h5py` for v7.3/HDF5, `scipy.io.loadmat` for v5), and motion energy is loaded separately from the neighboring `motionEnergy_*.mat` file if present. This is a glob-based loader over all available files, not a fixed 44-session list.

ii.
```python
def load_session_data(filepath):
    fmt = detect_mat_format(filepath)
    if fmt == 'hdf5':
        return load_session_data_hdf5(filepath)
    elif fmt == 'v5':
        return load_session_data_v5(filepath)
```

```python
for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
    for fn in sorted(os.listdir(data_dir)):
        if not fn.startswith('data_structure_'):
            continue
        ...
        probes = probe_map.get((animal, date), [1])
        me_fn = fn.replace('data_structure_', 'motionEnergy_')
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as “dual-format support” plus using loading scripts for probe metadata while still including “all available” sessions. The trajectory shows it explicitly counted 47 files on disk and decided to process the available files rather than transcribing the authors’ session list.

## 1-b. How are the data split into subjects?

i. Subjects are defined from the filename prefix before the date, captured as `animal` during file discovery. After session processing, `subjects` is the sorted unique set of these animal IDs and `subject_idx` maps each included session to that list.

ii.
```python
m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
animal = m.group(1)
date = m.group(2)
```

```python
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The notes and trajectory both treat the filename as the reliable session identifier. There is no evidence the AI trusted an internal metadata field for subject identity.

## 1-c. How are the data split into sessions?

i. One discovered `data_structure_<animal>_<date>.mat` file is treated as one candidate session. The final session list is the subset of those files that survive later inclusion checks and loading errors. The two task folders are pooled into one dataset.

ii.
```python
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
for i, sess_info in enumerate(all_sessions):
    result = process_session(sess_info, time_axis, show_processing=args.show_processing)
    if result is not None:
        results.append(result)
```

iii. The trajectory shows the AI first inventoried files on disk, then used the loading scripts only to supplement probe information. In the notes it later described the result as 43 included sessions from 47 candidates.

## 1-d. How are the data split into trials?

i. Trials are indexed by `bp['Ntrials']`, and trial-dependent arrays are handled by looping over `range(Ntrials)` or by boolean masks over the same index space. Kept trials are the indices where the later `valid_trials` mask is true.

ii.
```python
bp['Ntrials'] = int(f['obj/bp/Ntrials'][0, 0])
...
for j in range(Ntrials):
    trial_num = j + 1
    spk_mask = trial_nums == trial_num
```

```python
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
trial_indices = np.where(valid_trials)[0]
```

iii. The AI’s notes summarize the raw data as one row per trial in the Bpod structure and use those trial indices consistently throughout the conversion. No extra trial-boundary reconstruction is attempted.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two kinds of filtering. First, entire sessions are excluded unless they contain more than 40 right-hit DR trials and more than 40 left-hit DR trials. Second, within included sessions it keeps only trials that are not early, not stimulation, and not `no`/ignore trials. It does not explicitly remove behavior-only late trials after recording stops.

ii.
```python
def check_session_inclusion(session_data, probes):
    r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    return n_r > MIN_HIT_TRIALS and n_l > MIN_HIT_TRIALS, n_r, n_l
```

```python
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
```

iii. In the notes, the AI explicitly cites `UseInclusionCritera.m` and says it excludes early-lick and stim trials plus sessions failing the `>40` DR-hit criterion. The trajectory also shows it deliberately dropped `no` trials so outcome could stay binary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probe(s) inside `session['clu']`, using each unit’s `trial`, `trialtm`, and `quality` fields together with `bp.ev.goCue`.

ii.
```python
unit['trial'] = np.array(f[clu_group['trial'][u, 0]]).flatten().astype(int)
unit['trialtm'] = np.array(f[clu_group['trialtm'][u, 0]]).flatten()
unit['quality'] = read_h5_string(f, clu_group['quality'][u, 0])
```

```python
units = session['clu'][probe_idx]
goCue = session['bp']['ev']['goCue']
```

iii. The notes and trajectory both describe the neural pipeline as “align spike times to goCue, bin, smooth, filter FR,” so the AI clearly viewed `trialtm` plus `goCue` as the core inputs.

## 2-b. How is the `neural` data processed?

i. For each kept probe, the AI loops over units and trials, subtracts the trial’s go cue from spike times, bins spikes with `np.histogram`, converts counts to Hz by dividing by `DT`, then smooths each trial’s rate vector with a hand-built causal Gaussian kernel of length 15. Probes are concatenated afterward.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float64) / DT
fr_smooth = smooth_signal(fr, SMOOTH_N)
firing_rates[i, :, j] = fr_smooth
```

```python
def causal_gaussian_kernel(N):
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(1, N+1))))
    kern[:N//2] = 0
    kern = kern / kern.sum()
    return kern
```

iii. The AI repeatedly states in the notes and trajectory that this is meant to match `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`: 5 ms bins, causal Gaussian smoothing, and go-cue alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered in three stages: only clusters whose `quality` string does not contain `'garbage'` are considered; then units with mean firing rate `<= 0.5` Hz are dropped; then sessions with fewer than 10 remaining units are excluded.

ii.
```python
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

iii. In `CONVERSION_NOTES.md`, the AI says it is using `quality='all'`, deleting only garbage clusters, applying the code’s `lowFR=0.5`, and enforcing a minimum of 10 units per session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting that trial’s go-cue time: `trialtm - goCue[trial]`.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
```

iii. The notes explicitly call this out as matching `alignSpikes.m`, and the trajectory repeatedly summarizes the neural alignment step that way.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses `DT = 1/200 = 5 ms`. The AI constructs a shared time axis from `-2.5` to `2.5` seconds inclusive, giving 1001 timepoints, and does not apply any later temporal rebinning.

ii.
```python
DT = 1.0 / 200.0
TMIN = -2.5
TMAX = 2.5
...
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
```

```python
edges = np.append(time_axis, time_axis[-1] + DT)
```

iii. The notes and trajectory justify the 5 ms choice as directly inherited from `getDefaultParams.m` (`dt = 1/200`) and the `-2.5` to `2.5` second window from the reference code.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw field directly. The AI defines it from the global alignment window (`TMIN`, `TMAX`, `DT`) as the canonical go-cue-centered time axis used everywhere else.

ii.
```python
def make_time_axis():
    time_axis = np.arange(TMIN, TMAX + DT/2, DT)
    return time_axis[time_axis <= TMAX + 1e-10]
```

iii. The notes describe this as “time from goCue” and treat it as a constructed decoder input rather than a field loaded from the files.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI creates the 5 ms shared time axis once, casts it to `float32`, reshapes it to `(1, n_timepoints)`, and reuses the same array for every trial.

ii.
```python
inp = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(inp)
```

iii. The notes justify this only as “Time axis -2.5 to 2.5, same for all trials.” No additional transformation is described.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the same `time_axis` that is also used to define spike histogram edges and to interpolate camera and motion-energy features, so all streams share one go-cue-centered grid.

ii.
```python
edges = np.append(time_axis, time_axis[-1] + DT)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes repeatedly state that the data were aligned to the go cue and put on a common neural/video time basis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The code derives lick direction only from `bp['R']`, with left implied by `not R`. It does not use `hit`, `miss`, or recorded lick times to infer the actual chosen direction.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. The mapping table in `CONVERSION_NOTES.md` explicitly says `obj.bp.R -> output[0]: lick_direction | R=1, L=0`, so this was a deliberate simplification rather than an accident.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It is a direct binary relabeling: right trials become 1, all others become 0, and that value is repeated across all time bins of the trial.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
...
out[0, :] = lick_dir
```

iii. The notes justify the choice only as the required left/right decoder output. The trajectory never shows the AI revisiting the fact that this is instructed side rather than actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp['autowater']`.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. The notes map `obj.bp.autowater` directly to behavioral context and describe WC vs DR in exactly those terms.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It is a direct binary relabeling: autowater trials become WC (`0`), everything else becomes DR (`1`), then that value is repeated over time.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
out[1, :] = context
```

iii. The notes explicitly say `WC=0, DR=1`, matching the prompt.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp['hit']` alone after the code has already excluded `bp['no']` trials. `bp['miss']` is not used directly.

ii.
```python
valid_trials = valid_trials & ~bp['no']
...
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. The notes’ variable-mapping table says `obj.bp.hit -> output[2]: outcome | incorrect=0, correct=1`, which matches the implementation.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI makes outcome binary: hit becomes `1`, anything else in the kept trial set becomes `0`, and that value is repeated across all time bins.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
...
out[2, :] = outcome
```

iii. The trajectory shows that the AI intentionally removed `no` trials so it could keep outcome binary rather than introduce an ignore class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from camera 0 (`session['traj'][0]`) using the feature named `'tongue'`, its `ts` x/y coordinates, its `frameTimes`, the session video offset from `sglx.bitcode_bitstart` and `bp.ev.bitStart`, and the trial go cue. The confidence/likelihood channel is present in `ts` but not used.

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
```

```python
x = ts[feat_idx, 0, :]
y = ts[feat_idx, 1, :]
aligned_times = frame_times - vidshift - goCue[trix]
```

iii. The notes say “Velocity computation matching findVelocity.m,” and the trajectory describes this as extracting DLC tongue kinematics from the aligned video stream. There is no justification in the notes for using only one camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI aligns video frame times to the go cue, linearly interpolates tongue x/y positions onto the neural time axis, takes discrete gradients of the interpolated positions, sets tongue NaN gradients to zero, and computes velocity magnitude as `sqrt(xvel^2 + yvel^2)`. It does not use the likelihood channel, does not smooth the tongue trace, does not segment contiguous valid runs, and does not combine the side and bottom tongue views.

ii.
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
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes and trajectory justify this as matching `findPosition.m` plus `findVelocity.m`. Later trajectory steps acknowledge that this led to tongue thresholds of zero in many sessions because missing tongue data became zeros.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes one session-wide 50th percentile over all non-NaN tongue-velocity samples from the kept trials. Each time bin is then coded as `1` if `>= threshold` and `0` otherwise; NaN bins are forced to `0`.

ii.
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

iii. The notes cite the prompt’s 50th-percentile split. The trajectory explicitly notes the resulting threshold was often `0.00` because untracked tongue periods had been turned into zeros.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video time is aligned by subtracting a session-wide video/behavior offset and then the trial’s go cue. After that, tongue position is interpolated directly onto the same `time_axis` used by neural activity.

ii.
```python
vidshift = vidFileOffset - bitStart_mode
...
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
```

iii. The notes say this matches `findVideoOffset.m` and alignment to the neural time basis. The trajectory also calls out the shared time axis as the intended alignment mechanism.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from camera 1 using the feature named `'top_paw'`, with x/y coordinates from `ts`, camera `frameTimes`, the computed video offset, and the trial go cue.

ii.
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. The notes describe this as using `top_paw` and matching the reference kinematics code for non-tongue features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The paw pipeline is the non-tongue branch of `extract_feature_velocity`: align frame times, linearly interpolate x/y positions to the neural time axis, nearest-fill missing positions, take gradients, subtract a baseline derivative from each axis, nearest-fill missing derivatives, and compute velocity magnitude.

ii.
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
```

iii. The notes and trajectory justify this as following `findPosition.m` and `findVelocity.m`, especially the baseline subtraction for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. One session-wide median is computed over all non-NaN paw-velocity samples from kept trials. Each bin becomes `1` if `>= threshold`, `0` otherwise, and NaNs are forced to `0`.

ii.
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

iii. The notes justify this by the prompt’s “50th percentile” discretization requirement.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are aligned as `frame_times - vidshift - goCue[trial]`, then paw position is interpolated to the shared neural time axis.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

iii. The AI’s notes present paw alignment as the same shared video-to-neural alignment used everywhere else.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the separate `motionEnergy_*.mat` file if it exists, specifically the per-trial arrays loaded into `me_data['data']`. Timing is taken from camera 0 `frameTimes`, plus the session video offset and trial go cue.

ii.
```python
if sess_info['me_filepath'] is not None:
    me_data = load_motion_energy(sess_info['me_filepath'])
```

```python
trial_me = me_data['data'][trix]
cam_data = session['traj'][0]
frame_times = trial_traj['frameTimes']
aligned_times = frame_times - vidshift - goCue[trix]
```

iii. The notes identify motion energy as a separate file and the trajectory shows the AI debugging multiple file-layout variants before settling on this loader.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI unwraps several possible MATLAB layouts for the motion-energy file, then for each trial linearly interpolates the motion-energy trace onto the neural time axis. If frame times are missing it synthesizes them at 400 Hz and subtracts a hard-coded `0.5` s offset. After interpolation, remaining NaNs are nearest-filled.

ii.
```python
if me_raw.dtype.names and 'data' in me_raw.dtype.names:
    ...
else:
    me_data_arr = me_raw
    me_thresh = 0
```

```python
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
                bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_me(time_axis)
...
me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. In the notes, the AI says `loadMotionEnergy.m` “aligns to goCue, interpolates to neural time axis.” The trajectory shows it specifically chased file-format variation and motion-energy loading bugs rather than changing the interpolation-based approach.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. One session-wide median is computed over all non-NaN aligned motion-energy values from kept trials. Each time bin becomes `1` if `>= threshold`, `0` otherwise, and NaNs are forced to `0`.

ii.
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

iii. The notes justify this by the prompt’s 50th-percentile split and later note that fixing the motion-energy loader produced nearly balanced `0/1` classes.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using side-camera frame times corrected by the session video offset and trial go cue, then interpolated to the shared neural time axis. When frame times are unavailable, synthetic 400 Hz frame times are used instead.

ii.
```python
frame_times = trial_traj['frameTimes']
aligned_times = frame_times - vidshift - goCue[trix]
```

```python
frame_times = np.arange(1, n_frames + 1) / VIDEO_FS
aligned_times = frame_times - 0.5 - goCue[trix]
```

iii. The notes and trajectory justify this as the same go-cue and video-offset alignment logic used for video-derived variables.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code relies heavily on fallback behavior. Missing MATLAB fields are replaced with defaults (`stim_enable` false, `reward`/`bitStart` NaN, `sglx` absent -> zero video offset). Invalid probes are skipped. Missing `frameTimes` trigger synthetic 400 Hz times. Missing feature values are interpolated where possible, then nearest-filled for paw and motion energy. Missing tongue values become zero velocity after gradient. Missing motion-energy files or failed loads produce all-NaN arrays that later become zeros during discretization.

ii.
```python
except:
    bp['stim_enable'] = np.zeros(bp['Ntrials'], dtype=bool)
...
except:
    session['sglx'] = None
```

```python
if trial['frameTimes'] is not None and not np.all(np.isnan(trial['frameTimes'])):
    frame_times = trial['frameTimes']
else:
    n_frames = ts.shape[2]
    frame_times = np.arange(1, n_frames + 1) / VIDEO_FS
```

```python
if not is_tongue:
    xpos[:, trix] = _fill_nearest(xpos[:, trix])
...
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
```

iii. The notes frame this as robustness to mixed MATLAB formats and inconsistent session contents. The trajectory shows the AI repeatedly patching loaders and fallbacks so conversion could finish despite format differences and missing neural/video content.

## 11-a. What are the most time-consuming steps of the code?

i. The code is instrumented to time loading, spike processing, tongue extraction, paw extraction, and motion-energy alignment separately. Given the implementation, the most expensive steps are the nested spike loops over units and trials plus the per-trial interpolation/gradient work for tongue and paw, with loading also explicitly timed.

ii.
```python
print(f'    Loaded in {t_load:.1f}s')
...
print(f'    Spike processing: {t_spk:.1f}s, {n_neurons} total units')
...
print(f'    Tongue velocity: {t_tongue:.1f}s')
print(f'    Paw velocity: {t_paw:.1f}s')
print(f'    Motion energy: {t_me:.1f}s')
```

```python
for i, u_idx in enumerate(valid_units):
    ...
    for j in range(Ntrials):
```

iii. The notes estimate roughly 8 seconds per session and about 5 minutes for a full run. The trajectory treated full-dataset runtime as acceptable and never attempted a different design to reduce these loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: the nested unit-by-trial spike loop, the per-trial interpolation loops in `extract_feature_velocity`, the scalar nearest-fill loop in `_fill_nearest`, and the per-trial motion-energy interpolation loop.

ii.
```python
for i, u_idx in enumerate(valid_units):
    ...
    for j in range(Ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
```

```python
for trix in range(Ntrials):
    ...
    xpos[:, trix] = f_x(time_axis)
```

```python
for i in range(len(arr)):
    if mask[i]:
        dists = np.abs(valid - i)
        arr[i] = arr[valid[np.argmin(dists)]]
```

iii. The AI did not document a vectorization rationale. The trajectory instead focused on correctness and completing the run with the mixed-format data.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several expensive operations: it loops through all trials separately for every unit when binning spikes; it interpolates x and y separately for every feature and trial; it nearest-fills arrays after interpolation; and it repeats per-trial constants (`lick_direction`, `context`, `outcome`) across all time bins even though they do not vary in time.

ii.
```python
for i, u_idx in enumerate(valid_units):
    for j in range(Ntrials):
        ...
```

```python
f_x = interp1d(...)
f_y = interp1d(...)
...
out[0, :] = lick_dir
out[1, :] = context
out[2, :] = outcome
```

iii. The notes do not describe these as redundant. Instead they emphasize that the resulting decoder performance was acceptable, so efficiency tradeoffs were not revisited.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader materializes several fields that the conversion never uses downstream, including spike `tm`, behavioral `sample`/`delay`/`reward`, motion-energy `moveThresh`, and `NdroppedFrames`. The script also parses every loading script on each run even though only the final probe map is kept. During spike processing it returns `kept_units`, but that list is only used to check its length and is otherwise discarded.

ii.
```python
unit['tm'] = unit_raw['tm'].flatten().astype(float)
...
bp['ev']['sample'] = get_field(ev_raw, 'sample').flatten().astype(float)
bp['ev']['delay'] = get_field(ev_raw, 'delay').flatten().astype(float)
```

```python
me_thresh = float(me_raw['moveThresh'][0, 0].flatten()[0])
...
trial_data['NdroppedFrames'] = float(nd.flatten()[0])
```

```python
fr, kept = process_spikes(session, p_idx, time_axis)
if fr is not None and len(kept) > 0:
    all_firing_rates.append(fr)
```

iii. The notes justify the broad loading strategy as robustness to mixed file formats and as exploratory parity with the reference code. They do not claim these extra fields are used by the final dataset.
