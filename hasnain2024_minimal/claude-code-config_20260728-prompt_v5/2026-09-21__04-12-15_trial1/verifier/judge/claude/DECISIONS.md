# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file, `data_structure_<anm>_<date>.mat`, located in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior` folders. The 44 session names and their probe assignments are hardcoded in `SESSION_META`, transcribed from the authors' loading scripts. Motion energy is loaded from separate `motionEnergy_<anm>_<date>.mat` files. The AI detects the MAT file version (v5 vs v7.3) by reading the file header and uses either `h5py` or `scipy.io.loadmat` accordingly through a `SessionData` class that provides a unified interface.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

def detect_mat_version(path):
    with open(path, 'rb') as f:
        header = f.read(16).decode('ascii', errors='replace')
    return '7.3' if '7.3' in header else '5.0'

class SessionData:
    def __init__(self, data_path):
        self.version = detect_mat_version(data_path)
        if self.version == '7.3':
            self._f = h5py.File(data_path, 'r')
            self._obj = self._f['obj']
        else:
            d = sio.loadmat(data_path, squeeze_me=False)
            self._obj_v5 = d['obj'][0, 0]
```

iii. The agent traced through the authors' `load<ANM>_ALMVideo.m` scripts to identify which sessions and probes to include, noting that some sessions are commented out in those scripts and should be excluded. The agent chose a hardcoded list because each session has a specific probe assignment.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of each `SESSION_META` tuple (e.g., `'EKH1'`). Subjects are collected into a unique list as encountered during iteration, and `subject_idx` maps each session to its position in that list.

ii.
```python
for animal, date, probes, data_dir in SESSION_META:
    ...
    if animal not in subjects_set:
        subjects_set.append(animal)
    all_subject_idx.append(subjects_set.index(animal))
```

iii. The agent directly used the animal name from the session metadata, which corresponds to the filename convention.

## 1-c. How are the data split into sessions?

i. One session corresponds to one entry in `SESSION_META` and one file on disk. The two task folders (fixed-delay and randomized-delay) are treated uniformly. The result is 44 sessions: 25 fixed-delay and 19 randomized-delay.

ii.
```python
for animal, date, probes, data_dir in SESSION_META:
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{animal}_{date}.mat')
    ...
    sess = SessionData(data_path)
```

iii. The agent identified the sessions from the authors' loading scripts, combining both task paradigms into a single dataset.

## 1-d. How are the data split into trials?

i. Trials are rows of the behavioral data table (`bp`), with `Ntrials` defining the count. Per-trial fields are truncated to `n_trials_total`. Spike data carries trial indices. Each valid trial becomes one element in the per-session lists.

ii.
```python
n_trials_total = sess.get_n_trials()
hit = sess.get_bp_field('hit')[:n_trials_total]
miss = sess.get_bp_field('miss')[:n_trials_total]
...
valid_idx = np.where(valid_mask)[0]
```

iii. The agent uses the Bpod behavioral data structure which defines trials directly.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) photostimulation trials (`stim_enable == 1`) are excluded, (2) early-lick trials (`early == 1`) are excluded, (3) trials must be hit, miss, or no-response (excluding malformed trials). Additionally, trials with all-zero neural data (where recording ended before behavior) are excluded. The agent also applies a minimum of 10 units per session filter (`MIN_UNITS = 10`).

ii.
```python
valid_mask = (stim_enable == 0) & (early == 0) & ((hit == 1) | (miss == 1) | (no == 1))
valid_idx = np.where(valid_mask)[0]
...
# Filter out valid trials with all-zero neural data (recording ended)
has_spikes = np.array([np.any(trialdat[:, :, ti] != 0) for ti in valid_idx])
valid_idx = valid_idx[has_spikes]
```

iii. The agent cited the paper's methods for excluding early-lick and photostim trials. The recording-cutoff filter was added after discovering late trials in some sessions had zero neural data. The `MIN_UNITS` filter comes from the paper stating "at least 10 units."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters from `obj.clu{probe}`, specifically each unit's `trial` (trial assignment, 1-based), `trialtm` (spike time relative to trial start), and `quality` (curation label). The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
units = sess.get_units_for_probe(probe_idx)
# Each unit has: tm, trial, trialtm, quality
...
aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
counts = np.histogram(aligned[mask], bins=edges)[0]
```

iii. The agent identified these fields from the data structure and reference code.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 5 ms bins spanning -2.5 to 2.5 s from the go cue, converted to firing rates (counts/DT), and then smoothed with a **causal** Gaussian kernel (window=15). The causal kernel zeros out the first half of a gausswin(15) kernel. Units from multiple probes are concatenated.

ii.
```python
def causal_gaussian_smooth(x, window, bctype='reflect'):
    n = np.arange(window)
    alpha = 2.5
    center = (window - 1) / 2
    kern = np.exp(-0.5 * ((n - center) / (center / alpha)) ** 2)
    kern[:int(window // 2)] = 0  # causal
    kern = kern / kern.sum()
    ...

trialdat = bin_and_smooth_spikes(all_units, go_cue, n_trials_total, edges, time_axis)
```

iii. The agent described this as "causal Gaussian smoothing matching mySmooth.m" with window size 15.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded (matched case-insensitively). Clusters with empty quality strings are also excluded. Then units whose mean firing rate over the window is <= 1 Hz are dropped. Sessions with fewer than 10 remaining units are skipped entirely.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
units = [u for u in units
         if u['quality'].lower() not in {q.lower() for q in EXCLUDED_QUALITIES}
         and u['quality'] != '']
...
mean_frs = np.mean(trialdat, axis=(0, 2))
fr_mask = mean_frs > LOW_FR
if np.sum(fr_mask) < MIN_UNITS:
    ...  # skip session
```

iii. The agent noted that the paper states "all units with firing rates exceeding 1 Hz were included." The quality labels follow `findClusters.m`. The MIN_UNITS=10 filter comes from the paper's stated inclusion criteria.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting the go cue time for each trial: `aligned = trialtm - goCue[trial-1]`. The aligned spikes are then binned into the time window.

ii.
```python
aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
...
counts = np.histogram(aligned[mask], bins=edges)[0]
```

iii. The agent followed the reference code's alignment method.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (DT = 1/200), spanning -2.5 to 2.5 s (1000 bins). No rebinning is applied; spikes are directly counted into these bins.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200         # 5 ms
n_bins = int(round((TMAX - TMIN) / DT))
edges = np.linspace(TMIN, TMAX, n_bins + 1)
time_axis = edges[:-1] + DT / 2
```

iii. The agent chose 5 ms matching `getDefaultParams.m`'s `params.dt = 1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is not derived from raw data variables. It is the center of each time bin in the binning grid, defined by the window parameters.

ii.
```python
time_axis = edges[:-1] + DT / 2
...
session_input.append(time_axis.reshape(1, -1).copy())
```

iii. The agent stated "inputs would be time from go cue onset."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers of the 1000 bins spanning -2.5 to 2.5 s. No further processing.

ii.
```python
edges = np.linspace(TMIN, TMAX, n_bins + 1)
time_axis = edges[:-1] + DT / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the binning grid itself. The same bin edges used for spike counting define the time axis used as the input, so alignment is intrinsic.

ii.
```python
counts = np.histogram(aligned[mask], bins=edges)[0]
...
session_input.append(time_axis.reshape(1, -1).copy())
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields: `bp.hit`, `bp.miss`, `bp.R` (right-instructed), and `bp.L` (left-instructed). The lick direction is inferred from the combination of instructed side and outcome.

ii.
```python
hit = sess.get_bp_field('hit')[:n_trials_total]
miss = sess.get_bp_field('miss')[:n_trials_total]
R = sess.get_bp_field('R')[:n_trials_total]
L = sess.get_bp_field('L')[:n_trials_total]
```

iii. The agent reasoned: "miss on a right trial means the animal licked left, and miss on a left trial means it licked right."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hit on a right trial = right lick (1), hit on a left trial = left lick (0), miss on a right trial = left lick (0, wrong side), miss on a left trial = right lick (1, wrong side), no response = none (2).

ii.
```python
lick_dir = np.full(n_valid, 2, dtype=np.int64)
for i, ti in enumerate(valid_idx):
    if hit[ti] == 1 and R[ti] == 1:
        lick_dir[i] = 1
    elif hit[ti] == 1 and L[ti] == 1:
        lick_dir[i] = 0
    elif miss[ti] == 1 and R[ti] == 1:
        lick_dir[i] = 0  # wrong side
    elif miss[ti] == 1 and L[ti] == 1:
        lick_dir[i] = 1  # wrong side
```

iii. The agent derived the lick direction from the combination of instructed side and outcome, assigning the opposite side for miss trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `bp.autowater`. Autowater trials are water-cued (WC) context; everything else is delayed-response (DR).

ii.
```python
autowater = sess.get_bp_field('autowater')[:n_trials_total]
...
context = np.where(autowater[valid_idx] == 1, 0, 1).astype(np.int64)
```

iii. The agent identified this field directly from the behavioral data structure.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabeling: autowater == 1 maps to WC (0), otherwise DR (1).

ii.
```python
context = np.where(autowater[valid_idx] == 1, 0, 1).astype(np.int64)
```

iii. Straightforward mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags: `bp.hit` and `bp.miss`. The third category (ignore) is inferred when neither hit nor miss.

ii.
```python
hit = sess.get_bp_field('hit')[:n_trials_total]
miss = sess.get_bp_field('miss')[:n_trials_total]
```

iii. Same fields used for lick direction.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit = correct (1), miss = incorrect (0), neither = ignore (2).

ii.
```python
outcome = np.full(n_valid, 2, dtype=np.int64)
for i, ti in enumerate(valid_idx):
    if hit[ti] == 1:
        outcome[i] = 1
    elif miss[ti] == 1:
        outcome[i] = 0
```

iii. The agent mapped hit/miss/ignore to the codes specified in the instructions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking data from `obj.traj`, specifically the **bottom camera only** (camera index 1), feature index 0 (`top_tongue`). Frame times are from the same camera's `frameTimes`. The video offset is computed from `sglx.bitcode.bitstart` and `bp.ev.bitStart`.

ii.
```python
n_traj_trials = sess.get_n_traj_trials(1)  # bottom cam
for trix in range(min(n_trials_total, n_traj_trials)):
    ts, ft = sess.get_traj_trial(1, trix)  # bottom cam
    ...
    tv, tvis = compute_velocity_from_dlc(ts, aligned_ft, 0, time_axis)  # top_tongue = feat 0
```

iii. The agent chose the bottom camera only, reasoning that "angle and length specifically came from the bottom cam."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Velocity is computed as the frame-to-frame displacement magnitude divided by a fixed frame interval (1/400 s): `sqrt(dx^2 + dy^2) / dt_video`. Visibility requires both the current and next frame to have confidence >= 0.6. The velocity is then **interpolated** onto the neural time axis using `np.interp`. Discretization splits at the session 50th percentile of visible values.

ii.
```python
dx = np.diff(x)
dy = np.diff(y)
vel = np.sqrt(dx**2 + dy**2) / dt_video
vis = (conf[:-1] >= CONF_THRESH) & (conf[1:] >= CONF_THRESH)
...
vel_interp = np.interp(time_axis, ft_vel, vel, left=np.nan, right=np.nan)
```

iii. The agent used a standard velocity computation with interpolation to the target time grid.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of visible (valid) values. Below threshold = 0, at/above threshold = 1, not visible = 2.

ii.
```python
def discretize_per_session(values, valid_mask):
    result = np.full_like(values, 2, dtype=np.int64)
    all_valid = values[valid_mask]
    if len(all_valid) > 0:
        thresh = np.percentile(all_valid, 50)
        result[valid_mask] = np.where(all_valid >= thresh, 1, 0)
    return result

tongue_disc = discretize_per_session(tv_valid, tvis_valid)
```

iii. Follows the instruction's specification for per-session 50th percentile thresholding.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset and go cue time: `aligned_ft = ft - vidshift - go_cue[trix]`. The velocity is then **interpolated** to the neural time axis using `np.interp`.

ii.
```python
aligned_ft = ft - vidshift - go_cue[trix]
tv, tvis = compute_velocity_from_dlc(ts, aligned_ft, 0, time_axis)
# Inside compute_velocity_from_dlc:
vel_interp = np.interp(time_axis, ft_vel, vel, left=np.nan, right=np.nan)
```

iii. The agent computed the video offset using the same bitcode-based method as the reference.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking from the **bottom camera** (camera index 1), using **both** `top_paw` (feature index 4) and `bottom_paw` (feature index 5). The maximum of the two paw velocities is taken.

ii.
```python
pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
pv_bot, pvis_bot = compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)
combined_vis = pvis_top | pvis_bot
combined_vel = np.where(
    pvis_top & pvis_bot, np.maximum(pv_top, pv_bot),
    np.where(pvis_top, pv_top, np.where(pvis_bot, pv_bot, 0.0))
)
```

iii. The agent reasoned: "I'm weighing whether to average both paws' velocities or take the max... I'll settle on taking the maximum of top_paw and bottom_paw velocities."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same frame-to-frame velocity computation as tongue. Both paw features are computed independently, then combined by taking the maximum velocity where both are visible, or the single available value where only one is visible. Interpolated to the neural time axis. Discretized at session 50th percentile.

ii.
```python
combined_vel = np.where(
    pvis_top & pvis_bot, np.maximum(pv_top, pv_bot),
    np.where(pvis_top, pv_top, np.where(pvis_bot, pv_bot, 0.0))
)
paw_disc = discretize_per_session(pv_valid, pvis_valid)
```

iii. The agent chose the max combination to represent overall paw movement.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session 50th percentile of visible values. 0 = below, 1 = at/above, 2 = not visible.

ii.
```python
paw_disc = discretize_per_session(pv_valid, pvis_valid)
```

iii. Follows the same discretization scheme as other movement variables.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: frame times corrected by video offset and go cue, then interpolated to neural time axis.

ii.
```python
aligned_ft = ft - vidshift - go_cue[trix]
pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
```

iii. Same method as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files. The AI handles multiple file formats (raw cell array, struct with data field, nested struct).

ii.
```python
def load_motion_energy(me_path):
    d = sio.loadmat(me_path, squeeze_me=False)
    me_raw = d['me']
    # Handles Format 1 (raw cell array), Format 2 (struct), nested structs
    ...
```

iii. The agent discovered three different file formats across the 44 sessions and handled each.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values (already one number per frame) are **interpolated** onto the neural time axis using `np.interp`. NaN values are then filled using nearest-neighbor interpolation (matching `loadMotionEnergy.m`'s `fillmissing`). Discretized at session 50th percentile.

ii.
```python
me_aligned[:, trix] = np.interp(
    time_axis, aligned_ft[:min_len], me_trial[:min_len],
    left=np.nan, right=np.nan
)
# Fill NaN with nearest value
for trix in range(n_trials_total):
    col = me_aligned[:, trix]
    nans = np.isnan(col)
    if np.any(nans) and not np.all(nans):
        valid = ~nans
        me_aligned[:, trix] = np.interp(
            np.arange(n_time), np.where(valid)[0], col[valid]
        )
```

iii. The agent matched `loadMotionEnergy.m`'s NaN filling behavior.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile. 0 = below, 1 = at/above, 2 = no video (NaN after all processing).

ii.
```python
me_disc = discretize_per_session(me_valid, ~np.isnan(me_valid))
```

iii. Same discretization approach as other movement variables.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side camera frame times are used (motion energy corresponds to the side camera). Frame times are corrected by video offset and go cue, then motion energy is interpolated onto the neural time axis.

ii.
```python
n_side_trials = sess.get_n_traj_trials(0)  # side cam
for trix in range(min(n_trials_total, len(me_data), n_side_trials)):
    _, ft = sess.get_traj_trial(0, trix)
    aligned_ft = ft - vidshift - go_cue[trix]
    me_aligned[:, trix] = np.interp(
        time_axis, aligned_ft[:min_len], me_trial[:min_len], ...)
```

iii. The agent correctly used the side camera for motion energy alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials with all-zero neural data (recording ended) are excluded. (2) Low-confidence DLC frames (< 0.6) are marked as not visible with velocity set to 0. (3) Motion energy NaNs are filled with nearest-neighbor interpolation. (4) Missing DLC data (ts is None or too few frames) returns zero velocity and not-visible. (5) Sessions with too few units or trials are skipped. (6) Broad try-except blocks catch unexpected errors in DLC and motion energy loading.

ii.
```python
if ts is None or len(frame_times_aligned) < 3:
    return np.zeros(n_time), np.zeros(n_time, dtype=bool)
...
has_spikes = np.array([np.any(trialdat[:, :, ti] != 0) for ti in valid_idx])
valid_idx = valid_idx[has_spikes]
```

iii. The agent used a defensive programming approach with try-except blocks and explicit missing data handling.

## 11-a. What are the most time-consuming steps of the code?

i. The spike binning loop is the most computationally expensive step, as it iterates over every unit and every trial individually to compute histograms. DLC processing also iterates per-trial. File loading is significant but unavoidable.

ii.
```python
for i, unit in enumerate(units):
    aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
    for j in range(n_trials):
        mask = unit['trial'] == (j + 1)
        counts = np.histogram(aligned[mask], bins=edges)[0]
        fr = counts.astype(float) / DT
        trialdat[:, i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, 'reflect')
```

iii. The agent noted concern about efficiency but prioritized correctness.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning has a nested loop over units and trials that could use `np.histogram2d` to bin all trials at once for each unit, as the reference does. The per-trial smoothing could also be done as a single matrix operation. The DLC velocity computation loops per trial but has variable frame counts preventing simple vectorization.

ii.
```python
# Current: nested loop
for i, unit in enumerate(units):
    for j in range(n_trials):
        mask = unit['trial'] == (j + 1)
        counts = np.histogram(aligned[mask], bins=edges)[0]
```

iii. The agent acknowledged efficiency concerns but chose correctness over optimization.

## 11-c. What processing does the code repeat multiple times?

i. The smoothing is applied per-unit per-trial inside the binning loop, which means the Gaussian kernel construction happens repeatedly. The video data is processed for all trials including filtered ones, then only valid trial indices are selected afterwards. Go cue times are accessed multiple times.

ii.
```python
# Smoothing applied inside the inner loop for each unit x trial
trialdat[:, i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, 'reflect')
```

iii. The agent chose a straightforward implementation approach.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes neural firing rates and DLC velocities for **all** trials (including invalid ones), then selects only valid trials afterwards. This means spike binning, smoothing, and velocity computation are done for early-lick trials and photostim trials that are later discarded. The code also computes velocity for features by hard-coded index rather than by name, risking misalignment if feature ordering varies.

ii.
```python
# Binning done for ALL n_trials_total trials
trialdat = bin_and_smooth_spikes(all_units, go_cue, n_trials_total, edges, time_axis)
# Then only valid trials selected
tv_valid = tongue_vel[:, valid_idx]
```

iii. The agent processed all trials upfront for simplicity, filtering afterwards.
