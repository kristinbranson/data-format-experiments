# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing every `data_structure_*.mat` file under `/app/data/Ephys_Behavior` and `/app/data/RandomizedDelay_Ephys_Behavior`, then pairs each with a same-named `motionEnergy_*.mat` file if present. Each session file is loaded with an HDF5 reader first and a `scipy.io.loadmat` fallback for older MATLAB files.

ii.
```python
DATA_DIRS = [
    '/app/data/Ephys_Behavior',
    '/app/data/RandomizedDelay_Ephys_Behavior',
]

def find_session_files(data_dirs):
    sessions = []
    for data_dir in data_dirs:
        data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
        for df in data_files:
            parts = os.path.basename(df).replace('data_structure_', '').replace('.mat', '')
            me_file = os.path.join(data_dir, f'motionEnergy_{parts}.mat')
            if not os.path.exists(me_file):
                me_file = None
            sessions.append({'data_file': df, 'me_file': me_file, 'animal_date': parts})
    return sessions
```
```python
def load_session_data(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'obj' not in f:
                raise ValueError("No obj field")
            obj = f['obj']
            if 'clu' not in obj:
                raise ValueError("No clu field - behavior-only session")
        return _load_session_data_h5(filepath)
    except (OSError, ValueError):
        pass
    return load_session_data_v5(filepath)
```

iii. In `CONVERSION_NOTES.md`, the AI says it used `h5py` for MATLAB v7.3 and `scipy.io` as fallback, loaded motion energy from separate files, and skipped two behavior-only sessions. The trajectory shows it explicitly decided to keep glob-based discovery and then skip sessions lacking `clu`.

## 1-b. How are the data split into subjects?

i. The AI treats the animal id as the filename prefix before the first underscore, stores that per session as `animal`, and later builds `subjects` as the sorted set of unique animals with `subject_idx` mapping each processed session into that list.

ii.
```python
basename = os.path.basename(session_file)
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]
```
```python
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)
```

iii. There is no deeper justification in the notes beyond reporting per-mouse counts. The choice is implicit in the code and consistent with the filename-based session parsing used throughout.

## 1-c. How are the data split into sessions?

i. The AI treats each discovered `data_structure_<animal>_<date>.mat` file as one session. It does not use the authors’ fixed session list or fixed probe assignments; instead it processes every file found in the two directories and skips sessions later if they fail loading or have no valid clusters.

ii.
```python
for data_dir in data_dirs:
    data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
    for df in data_files:
        sessions.append({
            'data_file': df,
            'me_file': me_file,
            'animal_date': parts,
        })
```
```python
for idx, sf in enumerate(session_files):
    try:
        result = process_session(sf['data_file'], sf['me_file'], idx)
    except Exception as e:
        ...
        continue
    if result is None:
        continue
```

iii. `CONVERSION_NOTES.md` says two sessions were skipped because they were behavior-only, and the trajectory shows the AI accepted “process all files, skip bad ones” as its session-selection rule rather than reconstructing the paper’s curated 44-session list.

## 1-d. How are the data split into trials?

i. The AI uses the Bpod trial arrays as the per-trial backbone and uses boolean indexing into those arrays to define valid trials. Neural spikes are assigned to trials through each unit’s `trial` field, which is interpreted as 1-based trial numbers. Trial boundaries are therefore taken directly from the stored trial-wise arrays rather than reconstructed.

ii.
```python
Ntrials = int(bp['Ntrials'][0, 0].flatten()[0])
session['R'] = bp['R'][0, 0].flatten().astype(float)
session['hit'] = bp['hit'][0, 0].flatten().astype(float)
session['goCue'] = ev['goCue'][0, 0].flatten()
```
```python
valid_trials = np.where(trial_mask)[0]
...
trial_num = trial_idx + 1  # MATLAB 1-indexed
spike_mask = spike_trials == trial_num
```

iii. The AI did not separately justify trial splitting in its notes. The trajectory indicates it believed the session structure was understood once it had decoded the `.mat` layout, and the code reflects direct use of stored per-trial arrays.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only non-stimulated, non-early, responding trials. Concretely, it drops optogenetic trials, drops early-lick trials, and also drops all ignore/no-response trials by requiring `hit` or `miss`. It does not implement the reference’s additional cutoff for trials that extend past the end of electrophysiology recording.

ii.
```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
valid_trials = np.where(trial_mask)[0]
```

iii. `CONVERSION_NOTES.md` explicitly says it filtered with `~stim.enable`, `~early`, and `hit | miss`, and presents that as matching the reference. There is no note about the late-trial/ephys-overhang filter; that omission appears to be accidental rather than deliberate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the cluster arrays inside `obj.clu`, specifically each unit’s `quality`, `trialtm`, and `trial` fields, together with `bp.ev.goCue` for alignment. It also uses probe metadata to decide which probes to include.

ii.
```python
unit['quality'] = h5_read_string(f, f[q_ref])
unit['trialtm'] = f[tm_ref][()].flatten()
unit['trial'] = f[trial_ref][()].flatten().astype(int)
```
```python
session['goCue'] = ev['goCue'][()].flatten()
```

iii. The notes describe spike alignment to go cue onset and unit filtering by quality and firing rate, which implies these are the core raw variables. The trajectory also mentions probe filtering as a key decision.

## 2-b. How is the `neural` data processed?

i. For each kept unit and kept trial, the AI subtracts the trial’s go-cue time from spike times, bins spikes into 5 ms bins over `[-2.5, 2.5]`, converts counts to firing rates in Hz, and applies a causal Gaussian smoothing kernel intended to mimic MATLAB `mySmooth.m`. The resulting array is later transposed into `(n_units, n_timebins)` per trial.

ii.
```python
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts / dt
fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
trialdat[:, ui, ti] = fr_smooth
```
```python
def causal_gaussian_smooth(x, kernel_width_samples):
    kern[:N // 2] = 0
    kern = kern / kern.sum()
    return np.convolve(x, kern, mode='same')
```

iii. `CONVERSION_NOTES.md` explicitly says the AI followed `getDefaultParams.m` and `mySmooth.m`, used 5 ms bins, aligned to go cue, and used a causal Gaussian kernel of width `N=15`. The trajectory repeatedly cites “matching MATLAB” as the reason for this choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only probes whose location string contains `ALM`, removes units labeled `garbage`, `gabrga`, `noisy`, or `real?`, keeps empty quality labels, and then drops units whose mean firing rate is not greater than `0.5` Hz over the aligned data. It does not exclude `poor` units.

ii.
```python
LOW_FR = 0.5
...
excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
...
if q not in excluded and q != '':
    indices.append(i)
elif q == '':
    indices.append(i)
```
```python
if 'ALM' not in loc_upper:
    continue
...
fr_mask = mean_frs > LOW_FR
```

iii. The notes explicitly justify this as “matching `findClusters.m` with quality='all'” and “matching `removeLowFRClusters.m`, default `lowFR=0.5`,” and the trajectory shows the AI intentionally added ALM-only filtering after inspecting probe metadata.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned by subtracting each trial’s `goCue` time from spike times belonging to that trial, then binning the aligned times into the fixed go-cue-centered bin edges.

ii.
```python
go_time = go_cue_times[trial_idx]
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The notes explicitly say alignment is to go cue onset and the trajectory reflects that the AI treated this as a core, settled requirement from the start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 5 ms bins (`DT = 1/200`) from `-2.5` to `+2.5` seconds around go cue, giving 1000 bins per trial. It does not perform any second-stage temporal rebinning after that grid is defined.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200  # 5 ms bins
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. `CONVERSION_NOTES.md` explicitly ties this to `getDefaultParams.m` and states that the neural and behavioral streams are put on that same 5 ms time axis.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The AI’s input time axis is not read directly from a raw array. It is constructed from the chosen alignment window and bin size, with the interpretation that those bins are time relative to each trial’s go cue.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The notes describe the input as “continuous time axis `[-2.4975, ..., 2.4975]` in 5ms steps,” so the AI’s justification is simply that the decoder input should be the shared go-cue-centered bin grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin centers once from the fixed `[-2.5, 2.5]` grid and then reuses the same `(1, 1000)` vector for every trial.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
input_trials.append(input_trial)
```

iii. There is no extra justification in the trajectory; the notes treat this as a direct consequence of the required alignment and time-bin specification.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The AI uses the same `TIME_AXIS` for neural bin definitions and for the stored input vector, so the input is aligned by construction to the neural bins.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
counts, _ = np.histogram(aligned_times, bins=edges)
...
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The notes explicitly say the kinematic outputs are interpolated to the neural time axis, which implies the AI treated `TIME_AXIS` as the common aligned grid for all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction directly from the trial-wise `R` flag after trial filtering, interpreting `R=1` as right and `R=0` as left. It does not use `hit`/`miss` to infer actual lick direction and does not retain a no-lick class.

ii.
```python
session['R'] = bp['R'][0, 0].flatten().astype(float)
...
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

iii. `CONVERSION_NOTES.md` explicitly states “Lick direction: `R=1` (right), `L=0` (left) from `obj.bp.R` field.” The trajectory does not show the AI revisiting this simplification after deciding to keep only hit/miss trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI performs a direct cast of the instructed-side flag `R` into the decoder label. There is no extra transformation for miss trials and no third class for no-response trials because those trials were filtered out earlier.

ii.
```python
lick_direction = session['R'][valid_trials].astype(np.int32)
...
output_trial[0, :] = lick_direction[ti]
```

iii. The notes justify this by presenting lick direction as a direct readout from `bp.R`, implicitly assuming that once ignore trials are removed the trial’s side flag is sufficient.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag.

ii.
```python
session['autowater'] = bp['autowater'][0, 0].flatten().astype(float)
...
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` explicitly says context is derived from `obj.bp.autowater`, with `autowater=1` interpreted as the water-cued context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater` into decoder classes with `WC=0` and `DR=1` by computing `1 - autowater`.

ii.
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
```

iii. The notes explicitly justify this as the prompt’s required coding: delayed-response trials are 1 and water-cued trials are 0.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` flag on already-filtered responding trials. The AI does not preserve ignore trials, so it does not need a separate `no`-trial outcome code in its own formulation.

ii.
```python
session['hit'] = bp['hit'][0, 0].flatten().astype(float)
session['miss'] = bp['miss'][0, 0].flatten().astype(float)
...
outcome = session['hit'][valid_trials].astype(np.int32)  # correct=1, incorrect=0
```

iii. `CONVERSION_NOTES.md` explicitly says outcome is from `obj.bp.hit`, and the earlier trial filter `hit | miss` is the implicit justification for not carrying an ignore state forward.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI uses a binary outcome code: `1` for hits and `0` for misses, because only hit/miss trials survive the trial filter. There is no third “ignore” category in the output array.

ii.
```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
...
outcome = session['hit'][valid_trials].astype(np.int32)
```

iii. The notes justify this as “correct=1, incorrect=0 from `obj.bp.hit`,” and frame the exclusion of ignore trials as part of its trial curation rather than as an outcome-coding decision.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity from the side camera only, using the `tongue` feature inside `obj.traj`, together with `frameTimes`, the session-wide video offset, and each trial’s `goCue` time.

ii.
```python
# Tongue velocity from side camera (cam 0), feature 'tongue'
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
```
```python
feat_names = cam['feat_names']
...
x_pos = ts[feat_idx, 0, :].copy()
y_pos = ts[feat_idx, 1, :].copy()
frame_times = trial_data.get('frameTimes')
```

iii. The notes explicitly say “Tongue velocity: Extracted from side camera (cam 0), `tongue` feature.” The trajectory also records the AI deciding that “tongue has many NaN values” and should be handled from that feature directly.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI takes x/y trajectories for the side-camera tongue marker, computes frame-to-frame velocity magnitude with `np.gradient` at 400 Hz, sets tongue velocity to zero where the tongue is not visible, linearly interpolates the frame-wise velocity onto the neural time axis, then discretizes the interpolated values per session at the median.

ii.
```python
dx = np.gradient(x_pos, dt_vid)
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)

if is_tongue:
    nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
    vel[nan_mask] = 0.0
    vel[np.isnan(vel)] = 0.0
```
```python
interp_fn = interp1d(aligned_frame_times, vel,
                    kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. `CONVERSION_NOTES.md` explicitly says the AI matched `findPosition.m`, computed `sqrt(dx^2 + dy^2)` at 400 Hz, set tongue velocity to 0 when not visible, and interpolated to the neural axis. The trajectory confirms that “tongue not visible -> velocity 0” was a conscious choice.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI discretizes tongue velocity into two categories per session using the 50th percentile threshold: values below the threshold become `0`, values at or above it become `1`. It does not represent untracked bins as a separate category.

ii.
```python
def discretize_per_session(values, percentile=50):
    valid = values[~np.isnan(values)]
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
    return discretized
```
```python
tongue_vel_disc = discretize_per_session(tongue_vel)
```

iii. The notes explicitly justify this as following the decoder-task specification, and also acknowledge that tongue velocity becomes degenerate because the session median is often 0 after invisible frames are turned into zeros.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI aligns tongue velocity by subtracting a session-wide `vidshift` and the per-trial `goCue` from frame times, then interpolating the frame-wise velocity to the same `TIME_AXIS` used for neural data.

ii.
```python
bitstart_sglx = np.nanmedian(sglx['bitcode']['bitstart'][()].flatten())
session['vidshift'] = bitstart_sglx / fs - bitStart_bp
```
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel,
                    kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The notes explicitly say the video offset matches `findVideoOffset.m` and that tongue velocity is “interpolated to neural time axis aligned to go cue.”

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI derives paw velocity from bottom-camera paw features in `obj.traj`, attempting to use both `top_paw` and `bottom_paw` and averaging them when both are available.

ii.
```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)

if paw_top is not None and paw_bot is not None:
    paw_vel = (paw_top + paw_bot) / 2.0
```

iii. `CONVERSION_NOTES.md` explicitly says paw velocity is extracted from the bottom camera and “averaged `top_paw` and `bottom_paw`.” The trajectory shows the AI intentionally “fix[ing] the paw velocity extraction” around that choice.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI fills missing non-tongue x/y positions with nearest interpolation, computes velocity magnitude by differentiating x and y at 400 Hz, interpolates the frame-wise velocity to the neural grid, optionally averages the two paw tracks, and then discretizes the result per session at the median.

ii.
```python
if not is_tongue:
    for pos in [x_pos, y_pos]:
        mask = np.isnan(pos)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])
```
```python
dx = np.gradient(x_pos, dt_vid)
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)
```

iii. The notes justify this with “matching MATLAB fillmissing nearest” for non-tongue features and describe interpolation to the neural axis as part of the pipeline.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded into two per-session categories using the 50th percentile of all non-NaN paw-velocity samples, with no explicit third category for bins lacking visible paw data.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel)
```
```python
threshold = np.percentile(valid, percentile)
discretized = (values >= threshold).astype(np.int32)
```

iii. The notes explicitly say paw velocity is discretized into two bins with a per-session 50th-percentile threshold to satisfy the decoder-task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligns paw velocity exactly as it aligns tongue velocity: subtract session `vidshift` and per-trial `goCue` from frame times, then interpolate onto the common neural `TIME_AXIS`.

ii.
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel,
                    kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The notes explicitly say paw velocity is “interpolated to neural time axis,” and the shared extractor function embodies that decision.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from session-matched `motionEnergy_*.mat` files, using the `me.data` trial traces and optionally the stored threshold field. The AI does not derive it from the `obj` struct when the standalone file exists.

ii.
```python
mat = scipy.io.loadmat(filepath, squeeze_me=False)
me = mat['me']
data_field = me['data'][0, 0]
thresh_field = me['moveThresh'][0, 0]
...
return {'data': me_data, 'moveThresh': threshold}
```

iii. `CONVERSION_NOTES.md` explicitly says motion energy came from separate `motionEnergy_*.mat` files and was then aligned using video frame times and video offset.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI takes the frame-wise motion-energy trace for each trial, aligns frame times to go cue, linearly interpolates the trace onto the neural time axis, fills remaining gaps by nearest interpolation or all zeros if the full trace is missing, and then discretizes the interpolated values per session at the median.

ii.
```python
interp_fn = interp1d(aligned_frame_times, me_trial,
                    kind='linear', bounds_error=False, fill_value=np.nan)
me_interp[:, ti] = interp_fn(TIME_AXIS)
```
```python
for ti in range(n_trials):
    col = me_interp[:, ti]
    mask = np.isnan(col)
    if mask.any() and not mask.all():
        valid_idx = np.where(~mask)[0]
        col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
    elif mask.all():
        me_interp[:, ti] = 0.0
```

iii. The notes justify this as “interpolated to neural time axis using video frame times and video offset,” and also document that some sessions had missing motion energy and were replaced with zeros.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes motion energy into two per-session categories at the 50th percentile, without preserving missing bins as a separate class.

ii.
```python
me_disc = discretize_per_session(me_interp)
```
```python
threshold = np.percentile(valid, percentile)
discretized = (values >= threshold).astype(np.int32)
```

iii. The notes explicitly say motion energy is discretized into two bins using a per-session 50th-percentile threshold because that is what the decoder-task specification asked for.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy using camera frame times when available, subtracting session `vidshift` and trial `goCue`, then interpolating to the neural time axis. If frame times are missing, it synthesizes them from a nominal 400 Hz grid and uses a fallback `-0.5` second offset.

ii.
```python
if frame_times is None:
    frame_times = np.arange(1, n_frames + 1) / 400.0
    aligned_frame_times = frame_times - 0.5 - go_time  # fallback offset
else:
    aligned_frame_times = frame_times - vidshift - go_time
```
```python
interp_fn = interp1d(aligned_frame_times, me_trial,
                    kind='linear', bounds_error=False, fill_value=np.nan)
me_interp[:, ti] = interp_fn(TIME_AXIS)
```

iii. The notes explicitly tie the main path to `findVideoOffset.m`; the synthetic-frame-time fallback is not justified there and appears to have been added pragmatically to keep conversion from failing.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally fills or defaults missing data instead of preserving missingness. For non-tongue kinematics it nearest-fills missing positions; for tongue it turns invisible frames into zero velocity; for interpolated kinematics and motion energy it nearest-fills NaNs after interpolation and sets all-NaN trials to zeros; if `frameTimes` are missing it fabricates them from a 400 Hz clock; if motion-energy files cannot be loaded it falls back to all-zero outputs for that stream; and if session loading fails it skips the session.

ii.
```python
if not is_tongue:
    pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])
...
if is_tongue:
    vel[nan_mask] = 0.0
    vel[np.isnan(vel)] = 0.0
```
```python
if frame_times is None or np.all(np.isnan(frame_times)):
    frame_times = np.arange(1, n_frames + 1) / 400.0
...
elif mask.all():
    velocities[:, ti] = 0.0
```
```python
if me_interp is not None:
    me_disc = discretize_per_session(me_interp)
else:
    me_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
```

iii. The notes explicitly justify nearest-fill for non-tongue features and zero tongue velocity when invisible, and list missing motion-energy sessions as a known issue handled by setting the output to all zeros. The trajectory shows these were pragmatic fixes added to keep full conversion running.

## 11-a. What are the most time-consuming steps of the code?

i. The code structure suggests the most time-consuming steps are the nested per-unit, per-trial neural binning loop and the per-trial interpolation loops for tongue, paw, and motion energy. File loading is also substantial, but unlike the reference solution the AI’s code adds heavy Python-level loops over units and trials.

ii.
```python
for ui, ci in enumerate(cluster_indices):
    ...
    for ti, trial_idx in enumerate(valid_trials):
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
```
```python
for ti, trial_idx in enumerate(valid_trials):
    ...
    interp_fn = interp1d(aligned_frame_times, vel,
                        kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The AI did not explicitly discuss runtime bottlenecks in its notes. This assessment is inferred from the code structure and from the trajectory, which shows repeated debugging of these loops rather than any attempt to vectorize them.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The neural spike-processing loop could have been vectorized across trials, since it currently bins each unit one trial at a time. The post-interpolation NaN-filling loops and the repeated per-trial interpolation in kinematic and motion-energy extraction are also potential vectorization targets, although the irregular frame counts make them less straightforward.

ii.
```python
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        spike_mask = spike_trials == trial_num
        counts, _ = np.histogram(aligned_times, bins=edges)
```
```python
for ti in range(n_trials):
    col = velocities[:, ti]
    ...
for ti, trial_idx in enumerate(valid_trials):
    interp_fn = interp1d(aligned_frame_times, vel, ...)
```

iii. There is no explicit justification in the notes for keeping these loops. The trajectory focuses on getting a working conversion rather than on performance optimization.

## 11-c. What processing does the code repeat multiple times?

i. The AI repeats similar extraction and interpolation work feature by feature: the same kinematic extraction function is run separately for tongue, `top_paw`, and `bottom_paw`, and the same interpolate-then-fill pattern is implemented separately for kinematics and motion energy. Neural spike alignment is also repeated independently for every unit and trial rather than being batched.

ii.
```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
```
```python
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        ...
```

iii. The AI does not justify this repeated structure in its notes. It appears to be a straightforward implementation choice made while iterating toward a working pipeline.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and stores several fields that are never used in the final conversion, including `L`, `sample`, `delay`, `no`, some metadata fields, and `moveThresh` from the motion-energy files. It also computes some bookkeeping values that do not affect the output, such as `all_dates`, `n_valid`, and region-label normalization branches for non-ALM areas that are skipped earlier.

ii.
```python
session['L'] = bp['L'][0, 0].flatten().astype(float)
session['no'] = bp['no'][0, 0].flatten().astype(float) if 'no' in bp.dtype.names else np.zeros(Ntrials)
session['sample'] = ev['sample'][0, 0].flatten() if 'sample' in ev.dtype.names else np.zeros(Ntrials)
session['delay'] = ev['delay'][()].flatten() if 'delay' in ev else np.zeros(Ntrials)
```
```python
threshold = float(thresh_field.flatten()[0])
return {'data': me_data, 'moveThresh': threshold}
```
```python
all_dates.append(result['date'])
...
elif 'TJM1' in label_clean.upper() or 'M1' in label_clean.upper():
    label_clean = 'tjM1'
```

iii. The notes do not discuss unnecessary processing directly. These are inferred from reading the code and comparing loaded/computed intermediates with what is actually written into `converted_data.pkl`.
