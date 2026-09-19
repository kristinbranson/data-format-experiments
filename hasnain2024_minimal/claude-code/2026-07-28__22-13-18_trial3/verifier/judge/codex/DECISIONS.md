# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded a session table, `EPHYS_SESSIONS`, and then iterated over it, loading one `data_structure_<anm>_<date>.mat` file per entry from `data/<data_dir>/`. It used a custom `SessionData` class to branch between MATLAB v7.3 HDF5 and older MAT files, and a separate `load_motion_energy` helper to load `motionEnergy_<anm>_<date>.mat`. Unlike the human reference, it included both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`, yielding 45 sessions after removing two behavior-only files.

ii. 
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
```

```python
class SessionData:
    def __init__(self, fpath):
        self.fpath = fpath
        self.is_h5 = self._check_h5(fpath)
        if self.is_h5:
            self.f = h5py.File(fpath, 'r')
            self.obj = self.f['obj']
        else:
            mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
            self.obj_v5 = mat['obj']
```

iii. In the trajectory, the AI explicitly debated whether to use only the main task sessions or also randomized-delay sessions, then decided to include both because the decoder still predicts lick direction, outcome, and context on DR-only sessions (step 46). It later removed `JEB24_2023-10-03` and `JEB24_2023-10-04` after noticing they lacked `clu` data and were not in the loading scripts (step 101).

## 1-b. How are the data split into subjects?

i. Subjects are the `anm` strings from the hard-coded session tuples. The final `subjects` list preserves first-seen order rather than sorting, and `subject_idx` is built by looking up each session's `anm` in that list.

ii. 
```python
all_sessions = []
subjects_set = []

for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
    if result is not None:
        all_sessions.append(result)
        if anm not in subjects_set:
            subjects_set.append(anm)

subjects = subjects_set
subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])
```

iii. The trajectory does not contain a separate justification for subject indexing. The closest justification is the repeated use of `(anm, date, probes, data_dir)` as the canonical session identifier and the final summary of “45 sessions from 14 mice” (steps 46 and 145).

## 1-c. How are the data split into sessions?

i. Each tuple in `EPHYS_SESSIONS` is treated as one session. `process_session` constructs `data_structure_<anm>_<date>.mat`, processes that file once, and returns one session-level bundle of trials. Sessions are therefore defined by the hard-coded `(animal, date, probes, folder)` entries rather than by discovery from disk.

ii. 
```python
def process_session(anm, date, probe_nums, data_dir, time_edges, time_axis):
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)
    ...
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

iii. The AI justified its session list by reading the loading scripts and reconciling them with the data folders, but then chose to include both the main and randomized-delay sets for decoder coverage rather than matching the paper’s analyzed subset exactly (step 46). It later removed two JEB24 dates only after discovering they were behavior-only (step 101).

## 1-d. How are the data split into trials?

i. Trials are indexed by position within Bpod arrays: `0..ntrials-1`. The script reads per-trial arrays such as `hit`, `miss`, `R`, `autowater`, `early`, `stim.enable`, and `goCue`, computes `valid_idx`, and then loops over those trial indices to extract one neural matrix, one input array, and one output array per kept trial.

ii. 
```python
ntrials = sd.get_ntrials()
...
valid_trials = (hit | miss) & ~stim_enable & ~early
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

iii. The trajectory shows the AI inspecting the Bpod arrays and trial-level fields directly and treating them as trial tables, but it does not give an explicit philosophical justification beyond following the dataset structure (steps 33 and 39).

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are `hit` or `miss`, and not `stim` or `early`. This drops ignore/no-response trials entirely. The script also skips whole sessions with fewer than 5 valid trials, and later skips sessions with fewer than 10 surviving units. It does not implement the reference's extra removal of trials that continue after recording stops.

ii. 
```python
valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]

if len(valid_idx) < 5:
    print(f'  WARNING: Too few valid trials ({len(valid_idx)}), skipping')
    sd.close()
    return None
```

iii. The AI repeatedly described this as “matching reference code” and summarized it as excluding stim, early-lick, and ignore/no-response trials (steps 98, 137, and 145). The trajectory does not mention the reference solution’s recording-length cutoff.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from cluster spike assignments and times: for each cluster, the code uses `quality`, `trial`, and `trialtm`. It aligns `trialtm` to `bp.ev.goCue`, and it uses `quality` only for filtering.

ii. 
```python
def get_clusters(self, probe_nums):
    """Get list of (quality, trial_array, trialtm_array) for each cluster."""
    ...
    clusters.append((q, trial, trialtm))
    return clusters
```

```python
gocue = sd.get_event_times('goCue')
...
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    ...
    aligned_times = spike_times[trial_mask] - gocue[j]
```

iii. The trajectory shows the AI reading `alignSpikes.m`, `getSeq.m`, and inspecting sample `clu` structures before implementing the extraction of per-cluster `trial` and `trialtm` arrays (steps 25, 26, and 36).

## 2-b. How is the `neural` data processed?

i. For each kept cluster and trial, spikes are histogrammed into 10 ms bins from `-2.5` to `2.5` s around the go cue, converted to Hz by dividing by `DT`, and smoothed with a custom causal Gaussian kernel of length 15 bins using a reflect-style prepadding scheme. The code stores the smoothed binned rates directly and concatenates probes within a session.

ii. 
```python
DT = 1/100  # 10 ms
SMOOTH_WINDOW = 15

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
```

iii. The AI justified this by claiming it matched the paper's/reference code processing, specifically “10ms bins” and “causal Gaussian smoothing” from the reference scripts it had read (steps 26, 63, 98, and 145).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script first removes clusters whose free-text quality labels are in `{'garbage', 'gabrga', 'noisy', 'real?'}`. It then computes mean firing rates over the retained trials and bins and keeps only units above 1 Hz. Finally, if fewer than 10 units remain, it drops the entire session.

ii. 
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESH = 1.0
...
filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]
...
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH
...
if keep_units.sum() < 10:
    print(f'  WARNING: Only {keep_units.sum()} units with FR>{LOW_FR_THRESH} Hz, skipping')
    sd.close()
    return None
```

iii. The AI said this matched `findClusters.m` and `removeLowFRClusters.m`, and in its notes explicitly justified keeping all cluster types except a short drop list and then removing units at `<=1 Hz` (steps 27, 137, and 145).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Within each trial, neural alignment is done by subtracting that trial’s `goCue` time from each spike time in `trialtm`. The aligned spikes are then histogrammed into the session-wide bin edges.

ii. 
```python
gocue = sd.get_event_times('goCue')
...
trial_mask = (trial_nums == (j + 1))
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The trajectory shows the AI reading `alignSpikes.m` and then implementing the same subtraction-based alignment around go cue onset (steps 25 and 145).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted dataset uses 10 ms bins. Spikes are not rebinned from a finer grid afterward; they are counted directly into that 10 ms grid and then smoothed.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 1/100  # 10 ms
...
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. The AI repeatedly stated that 10 ms binning was the intended reference behavior and used it consistently throughout the script and final summary (steps 63, 98, 137, and 145).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read as a raw signal. The script constructs it from the session-wide decoder time grid defined by `TMIN`, `TMAX`, and `DT`, with the interpretation that the grid is centered on the trial’s go cue alignment point.

ii. 
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
...
input_data = time_axis.reshape(1, -1).copy()
```

iii. The trajectory does not give a separate justification for the input beyond “Align to go cue” and using the neural time axis itself as the decoder input (steps 98 and 145).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No raw-data processing is performed. The script just computes bin centers from the fixed global time edges and copies that vector into every trial as a `(1, n_timebins)` array.

ii. 
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
...
input_data = time_axis.reshape(1, -1).copy()
```

iii. The trajectory does not discuss this separately; it is implied by the choice to use the aligned time axis as the sole decoder input.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same time grid used for spike binning. Neural counts are histogrammed into `time_edges`, and the input uses the corresponding `time_axis` bin centers, so each input bin corresponds to one neural bin.

ii. 
```python
counts, _ = np.histogram(aligned_times, bins=time_edges)
...
time_axis = time_edges[:-1] + DT / 2
input_data = time_axis.reshape(1, -1).copy()
```

iii. The AI did not spell this out in the trajectory, but it consistently described the entire conversion as go-cue aligned and wrote the input directly from the same grid used for neural binning (steps 98 and 145).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The code derives lick direction only from the trial’s instructed side, effectively `R` versus not `R`. It loads both `R` and `L`, but only `R` is actually used when populating the output.

ii. 
```python
R = sd.get_trial_array('R').astype(bool)
L = sd.get_trial_array('L').astype(bool)
...
lick_dir = 1 if R[trial_idx] else 0
```

iii. The trajectory does not provide a separate justification for reducing lick direction to the instructed side. The final notes simply described the output as “left (0) vs right (1)” and omitted any no-lick class (steps 138 and 145).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is no use of `hit`, `miss`, or `no` when computing lick direction once a trial has passed the `hit|miss` validity filter. The output is a binary per-trial label: `1` for right-instructed trials and `0` otherwise, repeated across all time bins.

ii. 
```python
lick_dir = 1 if R[trial_idx] else 0
...
out = np.zeros((6, n_timebins), dtype=np.int64)
out[0, :] = t['lick_dir']
```

iii. The AI’s written documentation frames lick direction as a two-class output, which matches the code but not the reference handling of miss and no-lick trials (steps 138 and 145).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `autowater`, with water-cued trials mapped to one class and all other trials mapped to the delayed-response class.

ii. 
```python
autowater = sd.get_trial_array('autowater')
...
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. In the trajectory, the AI explicitly used the presence of `autowater` trials to distinguish two-context sessions from DR-only sessions and then encoded context from that same signal (steps 45 and 46).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The processing is a direct relabeling: `autowater == 1` becomes class `0` and everything else becomes class `1`. The label is repeated across all bins in each kept trial.

ii. 
```python
context = 0 if autowater[trial_idx] == 1 else 1
...
out[1, :] = t['context']
```

iii. The AI justified including DR-only sessions by saying those sessions would simply have context fixed at DR, which is exactly how the code behaves (step 46).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` flag alone at write time, even though `miss` is loaded earlier for trial filtering. The code never creates a separate ignore class.

ii. 
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
...
outcome = 1 if hit[trial_idx] else 0
```

iii. The AI’s documentation described outcome as “incorrect (0) vs correct (1)” and, because ignore trials were filtered out upstream, did not preserve a third category (steps 137, 138, and 145).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script assigns `1` to hit trials and `0` to all other kept trials. Because `valid_trials` already excludes ignore trials, the “all other” branch effectively means miss trials only.

ii. 
```python
valid_trials = (hit | miss) & ~stim_enable & ~early
...
outcome = 1 if hit[trial_idx] else 0
...
out[2, :] = t['outcome']
```

iii. The trajectory explicitly framed ignore trials as filtered out before output creation, so the code only needed a two-class outcome variable (steps 98, 137, and 145).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The script derives tongue velocity from bottom-camera trajectory data only: it searches camera view `1` for `top_tongue` and then reads that trial’s `frameTimes`, `x`, and `y`. It also uses the per-session video offset and the trial’s `goCue` to align those frame times to the neural clock.

ii. 
```python
bottom_feats = sd.get_traj_feature_names(1)
...
if name == 'top_tongue' and tongue_feat_idx is None:
    tongue_feat_idx = idx
...
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
```

iii. The AI inspected the feature lists and saw `top_tongue` on the bottom camera, then used that view in the implementation (steps 38, 39, and 49). It did not justify omitting the side-camera tongue signal.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the code computes frame-to-frame Euclidean speed from raw `x` and `y`, assumes a constant `VIDEO_FPS = 400`, timestamps speeds at midpoints between adjacent frames, aligns those timestamps by subtracting video offset and go cue, and linearly interpolates the resulting continuous trace onto the neural time axis. It does not use DLC likelihoods, does not smooth positions, does not normalize views, and does not combine both tongue cameras.

ii. 
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The AI justified the alignment part by reading `findVideoOffset.m` (steps 51 and 52). Later it noticed that many bins were NaN/outside video coverage and reasoned that thresholds should be computed from valid values only, but it still kept the interpolation-based pipeline (step 126).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is thresholded at the session-wide 50th percentile of all concatenated tongue-velocity bins after interpolation. The output is binary: bins at or above threshold become `1`, below threshold become `0`, and NaNs are also forced to `0` instead of getting a separate “not visible” class.

ii. 
```python
all_tongue = np.concatenate([t['tongue_vel'] for t in raw_outputs])
...
tongue_thresh = np.nanpercentile(all_tongue, 50) if not np.all(np.isnan(all_tongue)) else 0
...
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. The trajectory explicitly discusses the 50th-percentile split and the problems caused by NaN-heavy video coverage, but the final code still maps NaNs to class `0` rather than to a dedicated missing-data class (step 126).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted to the neural/behavior clock using `vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`, then each trial’s `goCue` is subtracted, and the velocity trace is interpolated onto the same `time_axis` used for neural data.

ii. 
```python
vidshift = sd.get_video_offset()
...
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. This is one of the clearest trajectory justifications: the AI read `findVideoOffset.m`, restated the offset formula, and then implemented it directly before interpolation (steps 49, 51, 52, and 145).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from a bottom-camera paw feature. The search prefers `bottom_paw`, and if that is not found it falls back to the first feature name containing `paw`. It then uses that feature’s `frameTimes`, `x`, and `y`, plus `vidshift` and `goCue`.

ii. 
```python
paw_feat_idx = None
for idx, name in enumerate(bottom_feats):
    if name == 'bottom_paw' and paw_feat_idx is None:
        paw_feat_idx = idx
...
if paw_feat_idx is None:
    for idx, name in enumerate(bottom_feats):
        if 'paw' in name.lower():
            paw_feat_idx = idx
            break
```

iii. The trajectory shows that the AI inspected both `top_paw` and `bottom_paw` names in the bottom view feature list (steps 39 and 49), but the final code specifically prefers `bottom_paw`; no explicit justification is given for that choice.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw velocity uses the same processing as tongue velocity: frame-to-frame Euclidean speed from raw `x` and `y`, constant 400 Hz sampling, midpoint timestamps, subtraction of video offset and go cue, and interpolation onto the neural time bins. No likelihood masking, run-wise smoothing, or per-bin averaging is applied.

ii. 
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The AI did not separately justify paw processing. It treated it as parallel to tongue processing in both code and documentation (steps 137 and 145).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is split at the session-wide median of all interpolated paw-velocity bins. As with tongue velocity, the output is binary and NaNs are coerced into class `0` instead of a separate visibility class.

ii. 
```python
all_paw = np.concatenate([t['paw_vel'] for t in raw_outputs])
paw_thresh = np.nanpercentile(all_paw, 50) if not np.all(np.isnan(all_paw)) else 0
...
paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                   np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
```

iii. The trajectory does not discuss a paw-specific thresholding rationale beyond following the same session-median discretization pattern used for the other continuous outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same alignment method as tongue velocity: subtract session-level video offset, subtract the trial’s go cue, then interpolate onto the neural `time_axis`.

ii. 
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The alignment justification is inherited from the AI’s use of `findVideoOffset.m`; it did not offer a paw-specific argument (steps 51, 52, and 145).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the standalone `motionEnergy_<anm>_<date>.mat` file via `load_motion_energy`. The per-trial motion-energy trace is then paired with camera frame times from view `1` when the output is built.

ii. 
```python
def load_motion_energy(data_dir, anm, date):
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_path = os.path.join(DATA_ROOT, data_dir, me_fn)
    ...
    return me_list, thresh
```

```python
me_data, me_thresh = load_motion_energy(data_dir, anm, date)
...
ft = sd.get_trial_frame_times(1, trial_idx)
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The AI spent time inspecting motion-energy file layouts and added special handling for wrapped structs and object arrays, justifying the helper as necessary to tolerate multiple `.mat` layouts (steps 47, 68, 74, 80, 111, and 119).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script does not derive motion energy from video pixels itself. It takes the precomputed per-frame trace from file, aligns the frame times to go cue, linearly interpolates the trace onto the neural time axis, and later discretizes by session median.

ii. 
```python
me_interp = np.full(n_timebins, np.nan)
if me_data is not None and trial_idx < len(me_data):
    try:
        ft = sd.get_trial_frame_times(1, trial_idx)
        ft_aligned = ft - vidshift - gocue[trial_idx]
        me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
    except:
        pass
```

iii. The trajectory justification focused on loading robustness rather than on the interpolation step. In the final summary, the AI grouped motion energy with the other video-derived outputs as being “interpolated to neural time bins” (step 145).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded at the session-wide 50th percentile of all interpolated motion-energy values. The result is again binary, with NaNs mapped to class `0` instead of a dedicated “no video” class.

ii. 
```python
all_me = np.concatenate([t['motion_energy'] for t in raw_outputs])
me_thresh_50 = np.nanpercentile(all_me, 50) if not np.all(np.isnan(all_me)) else 0
...
me_disc = np.where(np.isnan(t['motion_energy']), 0,
                  np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)
```

iii. The trajectory does not contain a motion-energy-specific thresholding justification beyond reusing the 50th-percentile scheme required by the decoder task.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The script uses camera view `1` frame times, subtracts the session-level video offset and the trial’s go cue, and interpolates the motion-energy trace onto the neural time axis.

ii. 
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The AI reused the same `findVideoOffset.m`-based alignment logic it used for tongue and paw outputs, but it did not justify why motion energy should use bottom-camera frame times specifically (steps 52 and 145).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles many problems by skipping or defaulting rather than repairing them. Missing session files, missing `clu` data, too few trials, and too few units cause the session to be skipped. Missing or unreadable motion-energy files return `None`. Errors when extracting tongue, paw, or motion energy leave those trial arrays as all-NaN, which are later turned into class `0` during discretization.

ii. 
```python
if not os.path.exists(fpath):
    print(f'  WARNING: File not found: {fpath}')
    return None
...
except (KeyError, AttributeError) as e:
    print(f'  WARNING: No cluster data found ({e}), skipping')
    sd.close()
    return None
```

```python
tongue_vel = np.full(n_timebins, np.nan)
...
except:
    pass
...
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. The clearest trajectory justification is that the AI removed two behavior-only JEB24 sessions after discovering they lacked `clu` fields (step 101). It also reasoned about NaNs in video features as a thresholding problem rather than dropping trials altogether (step 126).

## 11-a. What are the most time-consuming steps of the code?

i. The trajectory does not explicitly profile runtime, but the implementation suggests the expensive part is the nested spike-binning loop over units and trials, followed by the per-trial video interpolation for tongue, paw, and motion energy. Unlike the reference, this script does not batch spike counting across trials.

ii. 
```python
trialdat = np.zeros((n_units, n_timebins, ntrials))
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        if not np.any(trial_mask):
            continue
        aligned_times = spike_times[trial_mask] - gocue[j]
        counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The AI did not justify runtime tradeoffs in the trajectory. The only relevant evidence is the structure of the implementation itself and the fact that later debugging focused on output quality, not on profiling or optimization.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the double loop over clusters and trials for neural spike binning. The per-trial loops that compute tongue, paw, and motion-energy traces could also be partly batched, but the biggest avoidable loop is the neural one because spike times are already annotated with trial numbers.

ii. 
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        ...
        counts, _ = np.histogram(aligned_times, bins=time_edges)
        fr = counts / DT
        trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The trajectory does not discuss vectorization. This is an inference from the code structure rather than an explicit statement by the AI.

## 11-c. What processing does the code repeat multiple times?

i. The script repeats nearly identical per-trial processing for tongue velocity, paw velocity, and motion energy: get feature/frame times, align by `vidshift` and `goCue`, interpolate to `time_axis`, and later discretize by a session median. It also repeatedly scans per-trial cluster membership inside the inner spike loop.

ii. 
```python
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
...
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
...
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The trajectory does not mention this repetition. It is visible from the code and contrasts with the reference’s more reusable camera-processing structure.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things are loaded or computed but then unused downstream: `L` is read but never used, `me_thresh` is returned from `load_motion_energy` but ignored, and the script builds an extra `sample_data.pkl` dataset even though the task only required the full converted dataset. More subtly, it computes neural time series for all trials before discarding non-valid trials via `valid_idx`.

ii. 
```python
L = sd.get_trial_array('L').astype(bool)
...
me_data, me_thresh = load_motion_energy(data_dir, anm, date)
```

```python
# Sample: first 3 sessions
n_sample = min(3, len(all_sessions))
sample_data = {
    'neural': neural[:n_sample],
    ...
}
```

iii. The trajectory does not acknowledge these as unnecessary. It did describe the extra sample output and README/notes generation as part of task completion, but those are outside the strict conversion requirement (steps 137 and 138).
