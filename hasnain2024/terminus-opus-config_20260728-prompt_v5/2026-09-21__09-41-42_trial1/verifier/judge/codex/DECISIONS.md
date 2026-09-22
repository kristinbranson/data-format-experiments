# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a `SESSION_META` table of `(animal, date) -> (probes, folder)` entries, then iterates over that table, opening one `data_structure_<animal>_<date>.mat` file per session with a `SessionData` wrapper that tries HDF5 first and falls back to MATLAB v5 loading. Motion energy is loaded separately from `motionEnergy_<animal>_<date>.mat`.

ii.
```python
SESSION_META = {
    ('EKH1', '2021-08-07'): ([2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03'): ([1], 'RandomizedDelay_Ephys_Behavior'),
}
...
for (anm, date), (probes, dtype) in sorted(SESSION_META.items()):
    data_file = os.path.join(DATA_DIR, dtype, f'data_structure_{anm}_{date}.mat')
    if os.path.exists(data_file):
        sessions.append((anm, date, probes, dtype))
```
```python
class SessionData:
    def _open(self):
        try:
            self.f = h5py.File(self.data_file, 'r')
            self.is_hdf5 = True
        except:
            data = scipy.io.loadmat(self.data_file, squeeze_me=False)
            self.obj = data['obj']
            self.is_hdf5 = False
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset contains both HDF5 and MATLAB v5 `.mat` files, and that the reference loading scripts identify which sessions belong in the analysis. The notes also say both task folders should be included.

## 1-b. How are the data split into subjects?

i. The AI treats the animal id in the filename tuple as the subject id. It builds the `subjects` list incrementally in first-seen order and stores `subject_idx` as the index of each session’s animal in that list.

ii.
```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'animal': anm,
    'date': date,
    ...
}
```
```python
if anm not in all_subjects:
    all_subjects.append(anm)
subject_idx_list.append(all_subjects.index(anm))
```

iii. The notes justify this implicitly by counting unique animal IDs from filenames and using those names as the mouse identities.

## 1-c. How are the data split into sessions?

i. The AI defines one session as one `(animal, date)` entry in `SESSION_META`, paired with a single `data_structure` file in one of the two task folders. A session is only kept if its file exists and it later passes session-level inclusion rules.

ii.
```python
def process_session(anm, date, probes, dataset_type, show_processing=False):
    data_file = os.path.join(DATA_DIR, dataset_type, f'data_structure_{anm}_{date}.mat')
    me_file = os.path.join(DATA_DIR, dataset_type, f'motionEnergy_{anm}_{date}.mat')
```
```python
if os.path.exists(data_file):
    sessions.append((anm, date, probes, dtype))
```

iii. In `CONVERSION_NOTES.md`, the AI says both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` sessions should be included, and it uses the animal-specific loading scripts as the source for session membership.

## 1-d. How are the data split into trials?

i. Trials are represented by their integer index in the behavioral arrays. The AI reads `Ntrials`, uses flattened Bpod arrays of that length, uses spike `trial` labels to assign spikes to trials, and iterates over `valid_trial_indices` when constructing the final per-trial arrays.

ii.
```python
n_trials_total = sd.get_n_trials()
...
valid_trial_indices = np.where(valid_trials)[0]
```
```python
for t_idx in valid_trial_indices:
    neural_trials.append(trialdat[:, :, t_idx].T.astype(np.float32))
    input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```
```python
trial, trialtm = sd.get_spike_data(probe_idx, clu_idx)
...
spk_mask = trial_int == (t + 1)
```

iii. The justification is implicit: the notes describe each session file as containing trial-wise behavior, spikes, and video data, so the AI uses the shared trial index across those structures.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops early-lick trials, no-response/ignore trials, and photostimulation trials. It also drops entire sessions unless they have more than 40 right-hit delayed-response trials and more than 40 left-hit delayed-response trials, and excludes sessions with fewer than 2 valid trials or fewer than 10 neurons. It does not remove end-of-recording trials that contain all-zero neural data.

ii.
```python
valid_trials = (early == 0) & (no_resp == 0) & (stim_enable == 0)
valid_trial_indices = np.where(valid_trials)[0]
```
```python
r_hit_dr = (R == 1) & (hit == 1) & (stim_enable == 0) & (autowater == 0) & (early == 0)
l_hit_dr = (L == 1) & (hit == 1) & (stim_enable == 0) & (autowater == 0) & (early == 0)
...
if n_r_hit_dr <= 40 or n_l_hit_dr <= 40:
    return None
```
```python
if n_neurons < 10:
    return None
```

iii. The notes explicitly justify this with `UseInclusionCritera.m`, saying sessions with `<=40` correct DR trials per direction were excluded. The trajectory also states that zero-neural-data trials at the end of recording were kept as “valid behavioral trials.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from per-cluster spike `trial` labels and `trialtm` spike times inside `obj.clu`, plus the per-trial `goCue` times from `obj.bp.ev.goCue`. Cluster `quality` labels are also used for filtering.

ii.
```python
def get_spike_data(self, probe_idx, cluster_idx):
    ...
    trial = np.array(self.f[trial_ref]).flatten()
    trialtm = np.array(self.f[trialtm_ref]).flatten()
    return trial, trialtm
```
```python
go_cue = sd.get_event('goCue')
qualities = sd.get_cluster_qualities(probe_idx)
```

iii. `CONVERSION_NOTES.md` says “obj.clu spike times” are the source for `neural`, aligned to `goCue`, and that neuron filtering follows the cluster-quality and low-firing-rate rules from the reference code/paper.

## 2-b. How is the `neural` data processed?

i. For each cluster, spikes are aligned to go cue, histogrammed into 5 ms bins, divided by bin width to get firing rate, then smoothed with a 15-sample causal Gaussian kernel with reflect padding. Probes from the same session are concatenated afterward.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```
```python
trialtm_aligned = trialtm - go_cue[trial_int - 1]
...
N_counts, _ = np.histogram(spk_times, bins=EDGES)
fr = N_counts / dt
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
trialdat[:, neuron_idx, t] = fr_smooth
```
```python
trialdat = np.concatenate(all_trialdat, axis=1)
```

iii. The notes repeatedly justify this as matching `getSeq.m` and `mySmooth.m`, describing the smoothing as a “15-sample causal gaussian” and the binning as 5 ms around go cue.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first excludes clusters whose quality label is empty or matches `garbage`, `gabrga`, `noisy`, or `real?`. It then removes clusters whose mean firing rate is not above 1 Hz.

ii.
```python
'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],
'lowFR': 1.0,
```
```python
def find_clusters(qualities, exclude_list):
    ...
    if q_clean == '' or q_clean in exclude_lower:
        continue
```
```python
mean_frs = np.mean(trialdat, axis=(0, 2))
fr_mask = mean_frs > PARAMS['lowFR']
trialdat = trialdat[:, fr_mask, :]
```

iii. `CONVERSION_NOTES.md` says this matches `findClusters.m` and the paper’s 1 Hz threshold. The notes explicitly cite “exclude garbage, noisy, real?” and “FR > 1 Hz.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the trial’s go-cue time from each spike’s `trialtm`.

ii.
```python
trialtm_aligned = trialtm - go_cue[trial_int - 1]
```

iii. The notes explicitly say this matches `alignSpikes.m`: “subtract goCue time from trialtm.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 5 ms bins from `-2.5` to `+2.5` seconds around go cue, giving 1000 timepoints. There is no later temporal rebinning for neural data beyond this initial binning and smoothing.

ii.
```python
'dt': 1/200,  # 5ms bins
'tmin': -2.5,
'tmax': 2.5,
```
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEPOINTS = len(TIME_AXIS)
```

iii. The notes justify 5 ms bins by citing the reference parameters and the paper’s temporal binning.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not taken from a raw continuous field. It is constructed from the global bin edges around the go cue and represented by the shared `TIME_AXIS`.

ii.
```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```
```python
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. The notes describe this as “Time from go cue as continuous variable” and say the input is a fixed time axis aligned to the task event.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes the center of each 5 ms bin in the `[-2.5, 2.5]` s window and repeats that same vector for every trial.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. The trajectory and notes justify this as the decoder input requested by the task, rather than as a raw measurement that needs additional processing.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time input is exactly the same time grid used for neural binning, so each input bin corresponds to the same interval used for firing-rate estimation.

ii.
```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
N_counts, _ = np.histogram(spk_times, bins=EDGES)
```

iii. The notes justify this by saying the input is “time from go cue” on the same 5 ms grid used throughout the conversion.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from trial-level behavioral fields `L`, `R`, `hit`, `miss`, and `no` (no response).

ii.
```python
L = sd.get_bp_field('L')
R = sd.get_bp_field('R')
hit = sd.get_bp_field('hit')
miss = sd.get_bp_field('miss')
no_resp = sd.get_bp_field('no')
```

iii. The notes explain the intended mapping as “L/R + hit/miss” to lick direction, with `none` for no response.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI maps hit-left to left, hit-right to right, miss-left to right, miss-right to left, and no-response to `none` (code 2). It then repeats that per-trial category across all time bins.

ii.
```python
lick_dir = np.full(n_trials_total, -1, dtype=np.int64)
lick_dir[(hit == 1) & (L == 1)] = 0  # left
lick_dir[(hit == 1) & (R == 1)] = 1  # right
lick_dir[(miss == 1) & (L == 1)] = 1  # licked right (wrong)
lick_dir[(miss == 1) & (R == 1)] = 0  # licked left (wrong)
lick_dir[no_resp == 1] = 2  # none
...
output_trial[0, :] = lick_dir[t_idx]
```

iii. `CONVERSION_NOTES.md` explicitly documents this mapping as `0=left, 1=right, 2=none`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag.

ii.
```python
autowater = sd.get_bp_field('autowater')
```

iii. The notes say `autowater` is the source variable for context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly casts `autowater` to integer and treats `0` as delayed-response and `1` as water-cued context. It repeats that per-trial value across all time bins.

ii.
```python
context = autowater.astype(np.int64)  # 0=DR, 1=WC
...
output_trial[1, :] = context[t_idx]
```

iii. `CONVERSION_NOTES.md` explicitly states `autowater -> output[1]: context` and documents the coding as `0=DR, 1=WC`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the behavioral flags `hit`, `miss`, and `no`.

ii.
```python
hit = sd.get_bp_field('hit')
miss = sd.get_bp_field('miss')
no_resp = sd.get_bp_field('no')
```

iii. The notes describe outcome as coming from “hit/miss/no.”

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps hit to correct (`1`), miss to incorrect (`0`), and no response to ignore (`2`), then repeats the resulting class across all bins in the trial.

ii.
```python
outcome = np.full(n_trials_total, -1, dtype=np.int64)
outcome[hit == 1] = 1  # correct
outcome[miss == 1] = 0  # incorrect
outcome[no_resp == 1] = 2  # ignore
...
output_trial[2, :] = outcome[t_idx]
```

iii. `CONVERSION_NOTES.md` explicitly documents this mapping as `0=incorrect, 1=correct, 2=ignore`.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from camera 0 (`cam_idx=0`) DeepLabCut trajectories for the single feature named `tongue`, plus frame times, go-cue times, and the session video offset.

ii.
```python
tongue_speed, tongue_visible = extract_feature_velocity(
    sd, 0, 'tongue', go_cue, n_trials_total, is_tongue=True
)
```
```python
ts, ft, ndf = sd.get_traj_trial(cam_idx, t)
...
aligned_ft = ft - vidshift - go_cue[t]
```

iii. The notes justify this as using “camera 0 'tongue' feature” and treating missing tongue tracking as `not_visible`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates tongue x/y coordinates from video frame times onto the neural time axis, marks visibility from whether the interpolated x/y are non-NaN, computes velocity with `np.gradient`, zeroes gradients in invisible bins, and takes speed as `sqrt(xvel^2 + yvel^2)`. It does not smooth the tracked positions and does not combine the second tongue view.

ii.
```python
x_interp = np.interp(taxis, aligned_ft, x)
y_interp = np.interp(taxis, aligned_ft, y)
...
if is_tongue:
    vis = ~np.isnan(x_interp) & ~np.isnan(y_interp)
    visible[:, t] = vis

    xvel = np.gradient(x_interp)
    yvel = np.gradient(y_interp)
    xvel[~vis] = 0
    yvel[~vis] = 0
...
spd = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes justify this only briefly, saying tongue velocity is “Speed = sqrt(xvel^2 + yvel^2)” and that invisible frames should map to category 2.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI applies a per-session 50th-percentile threshold over all visible tongue-speed samples. Visible bins below threshold become `0`, visible bins at or above threshold become `1`, and invisible bins remain `2`.

ii.
```python
def discretize_velocity(speed, visible, percentile_thresh=50):
    categories = np.full((n_time, n_trials), 2, dtype=np.int64)
    vis_mask = visible & ~np.isnan(speed)
    if np.sum(vis_mask) > 0:
        threshold = np.percentile(speed[vis_mask], percentile_thresh)
        categories[vis_mask & (speed < threshold)] = 0
        categories[vis_mask & (speed >= threshold)] = 1
```

iii. `CONVERSION_NOTES.md` explicitly says tongue velocity is “Discretized per-session 50th percentile.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by a session-wide video offset and by the trial’s go-cue time, then the tracked positions are interpolated directly onto the same `TIME_AXIS` used for neural data.

ii.
```python
vidshift = sd.get_video_offset()
...
aligned_ft = ft - vidshift - go_cue[t]
...
x_interp = np.interp(taxis, aligned_ft, x)
y_interp = np.interp(taxis, aligned_ft, y)
```

iii. The notes cite `findVideoOffset.m` and say video is “interpolated to neural time axis with video offset correction.”

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from camera 1 (`cam_idx=1`) DeepLabCut trajectories for the single feature named `bottom_paw`, plus frame times, go-cue times, and the video offset.

ii.
```python
paw_speed, paw_visible = extract_feature_velocity(
    sd, 1, 'bottom_paw', go_cue, n_trials_total, is_tongue=False
)
```

iii. The notes explicitly justify this as “From camera 1 'bottom_paw' feature.”

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates the paw x/y trajectories to the neural time axis, computes gradients, subtracts a median baseline derivative, fills missing velocity samples by nearest interpolation, and takes Euclidean speed. Visibility is defined by non-NaN interpolated coordinates.

ii.
```python
x_interp = np.interp(taxis, aligned_ft, x)
y_interp = np.interp(taxis, aligned_ft, y)
...
xvel = np.gradient(x_interp)
yvel = np.gradient(y_interp)

basederiv_x = np.nanmedian(np.diff(x_interp))
basederiv_y = np.nanmedian(np.diff(y_interp))
if not np.isnan(basederiv_x):
    xvel -= basederiv_x
if not np.isnan(basederiv_y):
    yvel -= basederiv_y

for arr in [xvel, yvel]:
    mask_nan = np.isnan(arr)
    if np.any(mask_nan) and not np.all(mask_nan):
        valid = ~mask_nan
        arr[mask_nan] = np.interp(np.where(mask_nan)[0], np.where(valid)[0], arr[valid])
```

iii. The notes justify this as “Speed = sqrt(xvel^2 + yvel^2). Paw not visible -> category 2.”

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same per-session median split as tongue velocity: visible bins below the 50th percentile are `0`, visible bins at or above it are `1`, and invisible bins are `2`.

ii.
```python
paw_cat = discretize_velocity(paw_speed, paw_visible, 50)
```

iii. `CONVERSION_NOTES.md` explicitly states a per-session 50th-percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligns paw video exactly like tongue video: it subtracts the session video offset and the trial’s go cue from frame times, then interpolates onto the common neural time axis.

ii.
```python
aligned_ft = ft - vidshift - go_cue[t]
x_interp = np.interp(taxis, aligned_ft, x)
y_interp = np.interp(taxis, aligned_ft, y)
```

iii. The notes justify this generally by saying video tracking is “interpolated to neural time axis with video offset correction.”

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motionEnergy_<animal>_<date>.mat` files loaded separately from the main session file. The AI attempts to read `me['data']` and `me['moveThresh']` from that file.

ii.
```python
def get_motion_energy_data(self, me_file):
    if not os.path.exists(me_file):
        return None, None
    try:
        me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
        me_struct = me_mat['me']
        me_data_raw = me_struct['data'][0, 0]
        me_thresh = me_struct['moveThresh'][0, 0].flatten()[0]
```

iii. The notes explicitly justify using the separate motion-energy files and say sessions with missing motion-energy data should be assigned category 2.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligns each trial’s frame-level motion-energy trace to go cue using video frame times, interpolates it onto the neural time axis, and nearest-fills NaNs when partial data exist. It does not smooth or differentiate motion energy.

ii.
```python
aligned_ft = ft - vidshift - go_cue[t]
...
me_aligned[:, t] = np.interp(taxis, aligned_ft, me_trial)
```
```python
for t in range(n_trials):
    col = me_aligned[:, t]
    mask = np.isnan(col)
    if np.any(mask) and not np.all(mask):
        valid = ~mask
        col[mask] = np.interp(np.where(mask)[0], np.where(valid)[0], col[valid])
        me_aligned[:, t] = col
```

iii. The notes justify this only at a high level by saying motion energy comes from the separate files and is discretized per session.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI applies a per-session 50th-percentile threshold over all non-NaN motion-energy values. Values below threshold become `0`, values at or above threshold become `1`, and missing/no-video bins become `2`.

ii.
```python
def discretize_motion_energy(me_data, has_video, percentile_thresh=50):
    if not has_video:
        return np.full((n_time, n_trials), 2, dtype=np.int64)
    ...
    threshold = np.percentile(me_data[valid_mask], percentile_thresh)
    categories[valid_mask & (me_data < threshold)] = 0
    categories[valid_mask & (me_data >= threshold)] = 1
```

iii. `CONVERSION_NOTES.md` explicitly says motion energy is “Discretized per-session 50th percentile.”

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses camera-0 frame times when available, subtracts the session video offset and trial go cue, and interpolates the frame-wise motion-energy trace onto the common neural time axis. If video frame times are unavailable, it falls back to synthetic 400 Hz frame times with an extra `0.5` s offset.

ii.
```python
ts, ft, ndf = sd.get_traj_trial(0, t)  # Camera 0 for frame times
if ft is None or len(ft) < 10:
    ft = np.arange(len(me_trial)) / 400.0
    aligned_ft = ft - 0.5 - go_cue[t]  # Fallback alignment
else:
    aligned_ft = ft - vidshift - go_cue[t]
```

iii. The notes justify the main path by saying motion energy is aligned with video frame times; the fallback alignment is only implicit in the code.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video is handled permissively. Trials with invalid dropped-frame metadata or all-NaN frame times return no trajectory data and therefore remain `not_visible`; partially missing paw velocities are filled by nearest interpolation; motion energy is nearest-filled where partial data exist and assigned `no_video` if the motion-energy file is absent or fails to load; if motion-energy frame times are unavailable, synthetic frame times are used. Trials after the neural recording ended are not removed, which leaves some all-zero neural trials in the output.

ii.
```python
if np.any(np.isnan(ndf)):
    return None, None, None
...
if np.all(np.isnan(ft)):
    return None, None, None
```
```python
for arr in [xvel, yvel]:
    mask_nan = np.isnan(arr)
    if np.any(mask_nan) and not np.all(mask_nan):
        valid = ~mask_nan
        arr[mask_nan] = np.interp(np.where(mask_nan)[0], np.where(valid)[0], arr[valid])
```
```python
if ft is None or len(ft) < 10:
    ft = np.arange(len(me_trial)) / 400.0
    aligned_ft = ft - 0.5 - go_cue[t]
```

iii. The notes justify these choices as sensible defaults for “missing motion energy,” “no video data,” and end-of-recording zero-neural trials that the AI considered acceptable to keep.

## 11-a. What are the most time-consuming steps of the code?

i. The code suggests that per-session neural computation and video extraction are the expensive stages after file loading: `compute_firing_rates` nests over neurons and trials, and `extract_feature_velocity`/`extract_motion_energy` loop over trials. The script times neural and video stages separately inside each session.

ii.
```python
t_neural = time.time()
trialdat = compute_firing_rates(sd, probe_idx, quality_indices, go_cue, n_trials_total)
print(f'    Neural data computed in {time.time()-t_neural:.1f}s')
```
```python
t_video = time.time()
tongue_speed, tongue_visible = extract_feature_velocity(...)
paw_speed, paw_visible = extract_feature_velocity(...)
me_aligned, has_me = extract_motion_energy(...)
print(f'    Video data extracted in {time.time()-t_video:.1f}s')
```

iii. `CONVERSION_NOTES.md` records runtime estimates and says full conversion took a few seconds per session, but it does not give a deeper justification beyond those timings.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several obvious loops remain unvectorized: the nested neuron-by-trial spike loop in `compute_firing_rates`, the per-trial loops in `extract_feature_velocity` and `extract_motion_energy`, and the final per-trial formatting loop that builds `neural`, `input`, and `output` trial lists.

ii.
```python
for neuron_idx, clu_idx in enumerate(cluster_indices):
    ...
    for t in range(n_trials):
        spk_mask = trial_int == (t + 1)
        ...
```
```python
for t in range(n_trials):
    try:
        ts, ft, ndf = sd.get_traj_trial(cam_idx, t)
        ...
```
```python
for t_idx in valid_trial_indices:
    neural_trials.append(trialdat[:, :, t_idx].T.astype(np.float32))
    ...
```

iii. The notes say the script should be efficient, but there is no explicit defense of these remaining loops in the documentation.

## 11-c. What processing does the code repeat multiple times?

i. The AI recomputes the session video offset separately for tongue extraction, paw extraction, and motion-energy extraction instead of caching it once per session. It also repeats similar interpolation logic independently for tongue, paw, and motion-energy streams.

ii.
```python
vidshift = sd.get_video_offset()
```
This call appears inside both `extract_feature_velocity(...)` and `extract_motion_energy(...)`.

iii. No explicit justification was documented. The repetition appears to come from structuring the code as separate helper functions rather than from a stated analysis choice.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes continuous interpolated position-derived speeds for tongue and paw and continuous interpolated motion-energy traces, then immediately discretizes them to three categories, discarding the continuous values. It also computes `none`/`ignore` labels even though the corresponding trials are already filtered out. For paw and motion energy it additionally fills missing values that are then reduced to low/high categories.

ii.
```python
spd = np.sqrt(xvel**2 + yvel**2)
speed[:, t] = spd
...
tongue_cat = discretize_velocity(tongue_speed, tongue_visible, 50)
paw_cat = discretize_velocity(paw_speed, paw_visible, 50)
me_cat = discretize_motion_energy(me_aligned, has_me, 50)
```
```python
lick_dir[no_resp == 1] = 2  # none
...
outcome[no_resp == 1] = 2  # ignore
```

iii. No explicit justification was documented beyond the general requirement to produce categorical decoder outputs.
