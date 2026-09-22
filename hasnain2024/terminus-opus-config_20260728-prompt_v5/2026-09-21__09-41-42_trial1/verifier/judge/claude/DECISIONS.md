# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat` in either `Ephys_Behavior/` or `RandomizedDelay_Ephys_Behavior/`. The AI hard-codes a `SESSION_META` dictionary of 45 sessions (including `JEB23_2023-10-20` which the reference does not include). It opens each file via a `SessionData` class that detects HDF5 vs v5 format. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` using `scipy.io.loadmat` with `squeeze_me=False`.

ii. Session list and loading:
```python
SESSION_META = {
    ('EKH1', '2021-08-07'): ([2], 'Ephys_Behavior'),
    ...
    ('JEB23', '2023-10-20'): ([1], 'RandomizedDelay_Ephys_Behavior'),  # extra vs reference
    ...
    ('JEB24', '2023-11-03'): ([1], 'RandomizedDelay_Ephys_Behavior'),
}

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

iii. The AI identified sessions from the authors' loading scripts. It includes `JEB23_2023-10-20` which does not appear in the reference's session list.

## 1-b. How are the data split into subjects?

i. The animal ID is the first element of the `(anm, date)` tuple key in `SESSION_META`. Subjects are collected as an ordered list during processing, and `subject_idx` maps each session to its subject.

ii.
```python
if anm not in all_subjects:
    all_subjects.append(anm)
subject_idx_list.append(all_subjects.index(anm))
```

iii. The subject is taken directly from the session metadata key rather than parsing the filename or reading from the data file.

## 1-c. How are the data split into sessions?

i. One session is one `(anm, date)` entry in `SESSION_META`. Files are found by constructing the path from the known dataset type (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). Sessions without existing data files are skipped. The AI found 45 session entries; 2 were excluded by its inclusion criteria, yielding 43 sessions.

ii.
```python
for (anm, date), (probes, dtype) in sorted(SESSION_META.items()):
    data_file = os.path.join(DATA_DIR, dtype, f'data_structure_{anm}_{date}.mat')
    if os.path.exists(data_file):
        sessions.append((anm, date, probes, dtype))
```

iii. The AI's session list includes `JEB23_2023-10-20` which the reference does not have, and excludes 2 JEB19 sessions via inclusion criteria (see 1-e).

## 1-d. How are the data split into trials?

i. Trials are indexed 0 through `n_trials_total - 1`, where `n_trials_total` comes from `bp.Ntrials`. Each per-trial behavioral field is read as a flat array of length `n_trials_total`.

ii.
```python
n_trials_total = sd.get_n_trials()
# ...
for t in range(n_trials):
    spk_mask = trial_int == (t + 1)
```

iii. Trials are defined by the Bpod table; no trial-boundary reconstruction is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies THREE filters: (1) early-lick trials (`early == 1`), (2) no-response/ignore trials (`no_resp == 1`), and (3) photostimulation trials (`stim_enable == 1`). Additionally, the AI applies session-level inclusion criteria requiring >40 right-hit DR and >40 left-hit DR trials, excluding 2 JEB19 sessions. Importantly, the AI does NOT filter out trials past the end of the recording.

ii.
```python
valid_trials = (early == 0) & (no_resp == 0) & (stim_enable == 0)
valid_trial_indices = np.where(valid_trials)[0]

# Session-level inclusion
if n_r_hit_dr <= 40 or n_l_hit_dr <= 40:
    print(f'    EXCLUDED: insufficient DR hit trials (need >40 each)')
    sd.close()
    return None
```

iii. The AI's CONVERSION_NOTES.md says "Exclude early lick, ignore, and stimulation trials" and references `UseInclusionCritera.m`. However, the reference solution does NOT filter ignore trials and does NOT apply session-level inclusion criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu` spike-sorted clusters. Each cluster has `trial` (1-indexed trial assignment), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). `bp.ev.goCue` provides alignment times.

ii.
```python
trial, trialtm = sd.get_spike_data(probe_idx, clu_idx)
# ...
trialtm_aligned = trialtm - go_cue[trial_int - 1]
```

iii. Same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 5 ms bins spanning -2.5 to +2.5 s from the go cue (1000 bins), converted to firing rate (divide by dt), then smoothed with a **causal** 15-sample Gaussian kernel. The kernel is `gausswin(15)` with the first half zeroed out, applied via `np.convolve` with reflected boundary conditions.

ii.
```python
kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))
kern[:N//2] = 0  # Make causal
kern = kern / kern.sum()

out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. The AI's notes say "causal gaussian kernel with window N=15, reflect boundary" matching its reading of `mySmooth.m`. However, the reference solution uses a **symmetric** Gaussian via `gaussian_filter1d` with sigma = 14ms/5ms = 2.8 bins, which is a two-sided (non-causal) filter.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Quality label exclusion of `garbage`, `gabrga`, `noisy`, `real?` (but NOT `poor`); clusters with empty quality labels are also excluded. (2) Mean firing rate must exceed 1 Hz. Sessions with fewer than 10 neurons after filtering are excluded entirely.

ii.
```python
PARAMS = {
    'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],
    ...
}

def find_clusters(qualities, exclude_list):
    exclude_lower = [e.lower() for e in exclude_list]
    indices = []
    for i, q in enumerate(qualities):
        q_clean = q.lower().strip().replace('\x00', '')
        if q_clean == '' or q_clean in exclude_lower:
            continue
        indices.append(i)
    return np.array(indices, dtype=int)

# FR filter:
mean_frs = np.mean(trialdat, axis=(0, 2))
fr_mask = mean_frs > PARAMS['lowFR']
```

iii. The AI follows `findClusters.m` for quality but omits `poor` from the drop list. The AI also rejects empty-label clusters, while the reference keeps them. Additionally, the AI has a minimum neuron count per session (>=10) which the reference does not.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go cue time of their trial: `trialtm - goCue[trial-1]`. This matches the reference.

ii.
```python
trial_int = trial.astype(int)
valid_mask = (trial_int >= 1) & (trial_int <= n_trials)
trialtm_aligned = trialtm[valid_mask] - go_cue[trial_int[valid_mask] - 1]
```

iii. Same approach as the reference's `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (dt = 1/200), spanning -2.5 to +2.5 s, yielding 1000 time points. No rebinning is applied; this is the native binning. The bin edges are computed as `np.arange(tmin, tmax + dt, dt)`.

ii.
```python
PARAMS = {
    'dt': 1/200,  # 5ms bins
    'tmin': -2.5,
    'tmax': 2.5,
}
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

iii. Matches the reference's `params.dt = 1/200` and `params.tmin/tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is a constructed time axis, not derived from raw data. It is the bin centers of the 1000 time bins.

ii.
```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
# ...
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. Same as the reference.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is defined directly from the bin parameters. No processing of raw data is involved.

ii. N/A

iii. N/A

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the bin centers of the same time grid used for neural data. Both share the same `EDGES` / `TIME_AXIS`, so they are inherently aligned.

ii.
```python
N_counts, _ = np.histogram(spk_times, bins=EDGES)
# ...
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. Same as reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, `bp.L`, `bp.R`, and `bp.no` (no-response).

ii.
```python
L = sd.get_bp_field('L')
R = sd.get_bp_field('R')
hit = sd.get_bp_field('hit')
miss = sd.get_bp_field('miss')
no_resp = sd.get_bp_field('no')
```

iii. Same raw variables as the reference, though the AI also reads `no_resp` (which is used for trial filtering rather than lick direction coding, since ignore trials are already excluded).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hit + L => left (0), hit + R => right (1), miss + L => right (1, wrong side), miss + R => left (0, wrong side), no_resp => none (2). However, since ignore trials are filtered out in the AI's code, the "none" (2) class never appears in the output data. The verification output confirms: `lick_direction: {left (0.487), right (0.513)}`.

ii.
```python
lick_dir = np.full(n_trials_total, -1, dtype=np.int64)
lick_dir[(hit == 1) & (L == 1)] = 0  # left
lick_dir[(hit == 1) & (R == 1)] = 1  # right
lick_dir[(miss == 1) & (L == 1)] = 1  # licked right (wrong)
lick_dir[(miss == 1) & (R == 1)] = 0  # licked left (wrong)
lick_dir[no_resp == 1] = 2  # none
```

iii. The logic for deriving lick direction from hit/miss and instructed side is correct, matching the reference. However, the ignore trial filtering means the "none" class is absent from the converted data.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` field.

ii.
```python
autowater = sd.get_bp_field('autowater')
```

iii. Same as reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater` is cast directly to int: autowater=0 => DR (coded 0), autowater=1 => WC (coded 1). The output_values list is `['DR', 'WC']`.

ii.
```python
context = autowater.astype(np.int64)  # 0=DR, 1=WC
```

iii. The reference uses the opposite coding: WC=0, DR=1. Both are valid as long as `output_values` matches. The AI's `output_values` is `['DR', 'WC']` which correctly maps index 0 to DR and index 1 to WC.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit`, `bp.miss`, and `bp.no` (no-response).

ii.
```python
hit = sd.get_bp_field('hit')
miss = sd.get_bp_field('miss')
no_resp = sd.get_bp_field('no')
```

iii. Same raw variables as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit => correct (1), miss => incorrect (0), no_resp => ignore (2). However, since ignore trials are filtered out, the "ignore" (2) class never appears. The verification output confirms: `outcome: {incorrect (0.136), correct (0.864)}`.

ii.
```python
outcome = np.full(n_trials_total, -1, dtype=np.int64)
outcome[hit == 1] = 1  # correct
outcome[miss == 1] = 0  # incorrect
outcome[no_resp == 1] = 2  # ignore
```

iii. The coding logic is correct but the trial filtering eliminates the ignore class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking from `obj.traj` camera 0 only, feature `tongue`. Uses `ts` (tracking coordinates), `frameTimes` (frame timestamps), and `featNames`. Also uses `bp.ev.goCue` and `sglx` bitcode fields for video offset.

ii.
```python
tongue_speed, tongue_visible = extract_feature_velocity(
    sd, 0, 'tongue', go_cue, n_trials_total, is_tongue=True)
```

iii. The AI uses only the side camera's `tongue` feature. The reference uses BOTH cameras (`tongue` from side, `top_tongue` from bottom) and combines them after normalizing each by its 90th percentile.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Frame times are aligned via video offset correction. Positions (x, y) are **interpolated** to the neural time axis using `np.interp`. For tongue (is_tongue=True), NaN-interpolated positions are marked as not visible and velocity is set to 0 at those points. Velocity is computed as `np.gradient` of the interpolated positions on the neural time axis. Speed = sqrt(xvel^2 + yvel^2).

ii.
```python
x_interp = np.interp(taxis, aligned_ft, x)
y_interp = np.interp(taxis, aligned_ft, y)

vis = ~np.isnan(x_interp) & ~np.isnan(y_interp)
visible[:, t] = vis

xvel = np.gradient(x_interp)
yvel = np.gradient(y_interp)
xvel[~vis] = 0
yvel[~vis] = 0

spd = np.sqrt(xvel**2 + yvel**2)
```

iii. The reference computes velocity at **frame resolution** (smoothing x/y with a 5ms Gaussian, computing gradient against real frame times within contiguous runs of tracked frames), then **averages** frames into bins. The AI instead interpolates positions to the neural time axis first, then computes gradient on the interpolated data. This is a fundamentally different approach.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of visible speed values. Values below threshold => 0, at/above threshold => 1, not visible => 2.

ii.
```python
def discretize_velocity(speed, visible, percentile_thresh=50):
    vis_mask = visible & ~np.isnan(speed)
    if np.sum(vis_mask) > 0:
        threshold = np.percentile(speed[vis_mask], percentile_thresh)
        categories[vis_mask & (speed < threshold)] = 0
        categories[vis_mask & (speed >= threshold)] = 1
    return categories
```

iii. The thresholding logic matches the reference's approach (50th percentile split), though the underlying speed values differ due to the different velocity computation and single-camera approach.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (from bitcode alignment) and the trial's go cue time. Positions are then interpolated directly to the neural time axis.

ii.
```python
aligned_ft = ft - vidshift - go_cue[t]
x_interp = np.interp(taxis, aligned_ft, x)
```

iii. The clock correction matches the reference's `findVideoOffset.m`. The alignment approach differs: AI interpolates to neural bins, reference bins frame-level values into the same grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from `obj.traj` camera 1 (bottom camera), feature `bottom_paw`.

ii.
```python
paw_speed, paw_visible = extract_feature_velocity(
    sd, 1, 'bottom_paw', go_cue, n_trials_total, is_tongue=False)
```

iii. The reference uses `top_paw` from the bottom camera. The AI uses `bottom_paw`, which is a different forepaw feature that the reference notes has unreliable tracking through the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same interpolation-then-gradient approach as tongue. For paw (is_tongue=False), the code subtracts the median derivative as a baseline correction, and fills NaN values with nearest-neighbor interpolation.

ii.
```python
# For non-tongue:
xvel = np.gradient(x_interp)
yvel = np.gradient(y_interp)

basederiv_x = np.nanmedian(np.diff(x_interp))
if not np.isnan(basederiv_x):
    xvel -= basederiv_x

# Fill missing with nearest
for arr in [xvel, yvel]:
    mask_nan = np.isnan(arr)
    if np.any(mask_nan) and not np.all(mask_nan):
        valid = ~mask_nan
        arr[mask_nan] = np.interp(np.where(mask_nan)[0], np.where(valid)[0], arr[valid])
```

iii. The baseline subtraction and NaN-filling are not present in the reference, which computes velocity at frame resolution within contiguous tracked runs and leaves untracked bins as NaN (later categorized as "not visible").

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile of visible speed values.

ii.
```python
paw_cat = discretize_velocity(paw_speed, paw_visible, 50)
```

iii. Same thresholding logic as tongue and as reference.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction, then interpolation to neural time axis.

ii. Same as 7-d.

iii. Same alignment approach difference as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files. Loaded with `scipy.io.loadmat(squeeze_me=False)`, accessing `me['data'][0,0]` for per-trial traces and `me['moveThresh'][0,0]` for the threshold.

ii.
```python
def get_motion_energy_data(self, me_file):
    me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
    me_struct = me_mat['me']
    me_data_raw = me_struct['data'][0, 0]
    me_thresh = me_struct['moveThresh'][0, 0].flatten()[0]
    me_data = []
    for i in range(me_data_raw.shape[0]):
        d = me_data_raw[i, 0]
        me_data.append(d.flatten())
    return me_data, float(me_thresh)
```

iii. The reference uses `simplify_cells=True` and handles three different wrapping formats via a `while isinstance(me, dict): me = me['data']` loop. The AI's approach with `squeeze_me=False` and fixed indexing may fail on the 3 files with double-wrapped data.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy traces are **interpolated** to the neural time axis using `np.interp`. Then NaN values are filled with nearest-neighbor interpolation. The result is discretized at the 50th percentile.

ii.
```python
me_aligned[:, t] = np.interp(taxis, aligned_ft, me_trial)

# Fill NaN with nearest
for t in range(n_trials):
    col = me_aligned[:, t]
    mask = np.isnan(col)
    if np.any(mask) and not np.all(mask):
        valid = ~mask
        col[mask] = np.interp(np.where(mask)[0], np.where(valid)[0], col[valid])
```

iii. The reference averages frames within each bin (using `_bin_frames`), leaving bins with no frames as NaN. The AI interpolates to the neural time axis and fills NaN, which means no "no video" bins would appear for sessions with video data. The NaN-filling changes the downstream discretization.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile of non-NaN values. Below threshold => 0, at/above => 1, no video => 2 (only for sessions entirely missing motion energy data).

ii.
```python
def discretize_motion_energy(me_data, has_video, percentile_thresh=50):
    if not has_video:
        return np.full((n_time, n_trials), 2, dtype=np.int64)
    categories = np.full((n_time, n_trials), 2, dtype=np.int64)
    valid_mask = ~np.isnan(me_data)
    threshold = np.percentile(me_data[valid_mask], percentile_thresh)
    categories[valid_mask & (me_data < threshold)] = 0
    categories[valid_mask & (me_data >= threshold)] = 1
    return categories
```

iii. Same general approach as reference but the NaN-filling in 9-b means fewer bins get the "no video" class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from camera 0 (side camera) are corrected by video offset and go cue, then motion energy is interpolated to the neural time axis. If frame times are unavailable, a fallback of `np.arange(len(me_trial)) / 400.0 - 0.5 - go_cue[t]` is used.

ii.
```python
ts, ft, ndf = sd.get_traj_trial(0, t)
if ft is None or len(ft) < 10:
    ft = np.arange(len(me_trial)) / 400.0
    aligned_ft = ft - 0.5 - go_cue[t]
else:
    aligned_ft = ft - vidshift - go_cue[t]
me_aligned[:, t] = np.interp(taxis, aligned_ft, me_trial)
```

iii. Same clock correction as reference. The fallback alignment is an approximation. Reference does not have a fallback and uses frame times directly with bin averaging.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Trials with `NdroppedFrames` containing NaN are skipped entirely. (2) Trials with fewer than 10 frames are skipped. (3) Trials with all-NaN frame times are skipped. (4) For paw velocity, NaN positions are filled with nearest-neighbor interpolation. (5) For motion energy, NaN bins are filled with nearest-neighbor interpolation. (6) Sessions missing motion energy files get all bins set to "no video" (2). (7) Bare `except` blocks silently catch and skip various errors.

ii.
```python
if len(ft) < 10:  # Skip trials with too few frames
    continue

# Fill missing with nearest (paw)
for arr in [xvel, yvel]:
    mask_nan = np.isnan(arr)
    if np.any(mask_nan) and not np.all(mask_nan):
        arr[mask_nan] = np.interp(np.where(mask_nan)[0], np.where(valid)[0], arr[valid])
```

iii. The reference preserves NaN/missing data as the "not visible" class without interpolation. The AI's approach of filling NaN values fabricates data and obscures genuinely missing information. The bare `except` blocks could silently swallow real errors.

## 11-a. What are the most time-consuming steps of the code?

i. Neural data computation (spike binning per cluster per trial) dominates, due to a per-trial loop inside the per-cluster loop. File loading is also significant. Total conversion time is ~164s for 43 sessions.

ii.
```python
for neuron_idx, clu_idx in enumerate(cluster_indices):
    # ...
    for t in range(n_trials):
        spk_mask = trial_int == (t + 1)
        spk_times = trialtm_aligned[spk_mask]
        N_counts, _ = np.histogram(spk_times, bins=EDGES)
```

iii. The nested loop (clusters x trials) is the main bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike counting loop iterates over both clusters AND trials individually (`for neuron_idx ... for t in range(n_trials)`). The reference vectorizes this with a single `np.histogram2d` call per cluster that bins all trials at once, eliminating the inner trial loop entirely.

ii.
```python
# AI's per-trial loop:
for t in range(n_trials):
    spk_mask = trial_int == (t + 1)
    spk_times = trialtm_aligned[spk_mask]
    N_counts, _ = np.histogram(spk_times, bins=EDGES)

# Reference's vectorized approach:
counts, _, _ = np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])
```

iii. The video offset is also recomputed for each call to `extract_feature_velocity` rather than cached once per session.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`sd.get_video_offset()`) is computed independently in `extract_feature_velocity` for tongue and paw, and again in `extract_motion_energy` -- three times per session instead of once. Each call recomputes the mode of the bitcode arrays.

ii.
```python
# Called separately in each of these:
tongue_speed, tongue_visible = extract_feature_velocity(sd, 0, 'tongue', ...)  # calls get_video_offset
paw_speed, paw_visible = extract_feature_velocity(sd, 1, 'bottom_paw', ...)   # calls get_video_offset
me_aligned, has_me = extract_motion_energy(sd, me_file, ...)                   # calls get_video_offset
```

iii. The reference caches the video offset once in `Camera.__init__`.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `me_thresh` (moveThresh) value is loaded from the motion energy file but never used. (2) The `NdroppedFrames` field is checked but its value is not used meaningfully. (3) The AI reads `bp.no` (no-response) and computes lick_dir and outcome values for ignore trials, but these values are never included in the output because ignore trials are filtered out.

ii.
```python
me_data, me_thresh = sd.get_motion_energy_data(me_file)  # me_thresh never used for discretization
# ...
no_resp = sd.get_bp_field('no')  # used only for trial filtering, not for output coding in practice
```

iii. The `me_thresh` could have been used as the motion energy threshold, but the AI uses the 50th percentile instead (as instructed).
