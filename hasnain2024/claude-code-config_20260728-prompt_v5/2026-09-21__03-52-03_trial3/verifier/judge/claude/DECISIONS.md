# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file, `data_structure_<anm>_<date>.mat`, in one of two folders (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). The 44 session names and their probes are hard-coded in `SESSION_META`, transcribed from the authors' loading scripts. The file format (v7.3 HDF5 vs v5.0) is auto-detected by checking the file header. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` files.

ii. Session metadata and loading:
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ...
    ('JEB24', '2023-11-03', [1], 'randdelay'),
]

def load_session(fpath):
    with open(fpath, 'rb') as ff:
        header = ff.read(15)
    if b'7.3' in header:
        return load_session_h5(fpath)
    else:
        return load_session_v5(fpath)
```

iii. The AI documented that sessions were transcribed from the authors' `load<ANM>_ALMVideo.m` loading scripts, excluding commented-out sessions and sessions without data files (JEB4, JEB5). The AI noted that both HDF5 and v5.0 MATLAB formats are present and need different loading code.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of the `SESSION_META` tuple (e.g., `'EKH1'`). At assembly, `subjects` is the sorted set of unique animal names, and `subject_idx` maps each session to its subject. After session-level filtering, 14 animals remain across 42 sessions.

ii.
```python
subjects = sorted(set(r['anm'] for r in all_results))
subject_idx = np.array([subjects.index(r['anm']) for r in all_results])
```

iii. The AI justified this as directly following the loading scripts, where each session is tagged with its animal.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_META` is one session, specified as `(animal, date, probes, data_dir_key)`. Both fixed-delay and randomized-delay sessions are processed uniformly. After filtering (inclusion criteria and min units), 42 of 44 defined sessions are included. Two JEB19 sessions are dropped for insufficient DR trials.

ii.
```python
for idx, (anm, date, probes, ddir) in enumerate(sessions_to_process):
    result = process_session(anm, date, probes, ddir, ...)
    if result is not None:
        all_results.append(result)
```

iii. The AI documented that 44 sessions were defined but 2 were skipped due to failing the >=40 R-hit and >=40 L-hit DR trial inclusion criterion from the paper's `UseInclusionCriteria`.

## 1-d. How are the data split into trials?

i. Trials are indexed by their position in the session's behavioral arrays (0-based). The number of trials is read from `session['ntrials']` (i.e., `obj.bp.Ntrials`). All per-trial fields (hit, miss, R, L, etc.) are read as arrays of that length. Trial filtering is applied via a boolean mask to select which trials are retained.

ii.
```python
session['ntrials'] = int(bp['Ntrials'][0, 0])
session['hit'] = bp['hit'][0, :].astype(bool)
...
trial_mask = ~session['stim_enable']
trial_indices = np.where(trial_mask)[0]
trialdat = trialdat[:, :, trial_indices]
```

iii. The AI uses the Bpod trial count as the source of truth for how many trials exist, consistent with the reference code.

## 1-e. How are trials filtered based on quality controls?

i. Only photostimulation trials are excluded from the data (`~stim_enable`). Early-lick trials are NOT excluded -- they are used only in the session inclusion criteria check (where the valid mask includes `~early & ~autowater & ~stim_enable` for counting R-hit and L-hit DR trials). Additionally, sessions must have >=40 right-hit and >=40 left-hit DR trials, and >=10 neural units after filtering, or the entire session is dropped.

ii. Inclusion criteria check (not actual data filtering):
```python
valid_mask = ~session['stim_enable'] & ~session['autowater'] & ~session['early']
r_hit = session['R'] & session['hit'] & valid_mask
l_hit = session['L'] & session['hit'] & valid_mask
if n_r_hit < 40 or n_l_hit < 40:
    ...skipping...
```

Actual trial filtering applied to data:
```python
trial_mask = ~session['stim_enable']
trial_indices = np.where(trial_mask)[0]
```

iii. The AI's CONVERSION_NOTES state: "Include ALL trials (hit, miss, no/ignore, early, autowater) to maximize training data" except stim-enabled trials. The AI argues that early-lick and autowater trials provide useful decoder training data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike-sorted cluster data from `obj.clu{probe}`, specifically `trial` (1-based trial number for each spike), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
trialtm = u['trialtm']
trial_ids = u['trial']
aligned_times = trialtm[spike_mask] - go_cue[t]
counts, _ = np.histogram(aligned_times, bins=EDGES)
```

iii. Consistent with the reference code's `alignSpikes.m` and `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, binned into 10 ms bins (DT=1/100), converted to firing rates (Hz) by dividing by the bin width, and smoothed with a causal Gaussian kernel. The kernel is a `gausswin(15)` with the first half zeroed out, matching `mySmooth.m`. The reflect boundary condition is applied by reflecting the first n points before convolution.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WIN = 15
...
def causal_gaussian_kernel(n):
    alpha = 2.5
    half = (n - 1) / 2
    t = np.arange(n) - half
    kern = np.exp(-0.5 * (alpha * t / half) ** 2)
    kern[:n // 2] = 0
    kern /= kern.sum()
    return kern

fr = counts.astype(np.float64) / DT
trialdat[:, ui, t] = smooth_signal(fr)
```

iii. The AI notes: "Most analysis scripts use params.dt=1/100 (10ms)" and "Causal Gaussian smoothing matching mySmooth.m (15-point window, reflect BC)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded (lower-cased for matching). Second, units with mean firing rate <= 1 Hz are removed. Additionally, sessions with fewer than 10 units remaining after filtering are dropped entirely.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10

def filter_clusters(units, excluded_qualities=EXCLUDED_QUALITIES):
    good_idx = []
    for i, u in enumerate(units):
        q = u['quality'].lower()
        if q not in excluded_qualities:
            good_idx.append(i)
    return good_idx

mean_frs = trialdat.mean(axis=(0, 2))
keep_mask = mean_frs > LOW_FR
...
if n_neurons < MIN_UNITS:
    ...skipping...
```

iii. The AI's CONVERSION_NOTES document: "Exclude clusters with quality = 'garbage', 'gabrga', 'noisy', 'real?' (quality='all' mode)" and "Remove units with mean FR <= 1 Hz." The AI also applies a minimum of 10 units per session based on the paper's mention of "at least 10 units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `goCue[trial]` from each spike time. The aligned spikes are then binned with histogram edges spanning -2.5 to 2.5 s.

ii.
```python
aligned_times = trialtm[spike_mask] - go_cue_times[t]
counts, _ = np.histogram(aligned_times, bins=EDGES)
```

iii. Same approach as the reference code's `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (DT = 1/100), producing 500 time bins over the -2.5 to 2.5 s window. No rebinning is applied -- spikes are directly binned at this resolution. This differs from the reference's default `params.dt = 1/200` (5 ms bins, 1000 time bins).

ii.
```python
DT = 1.0 / 100  # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. The AI justified this by noting "Most analysis scripts use params.dt=1/100" and "Use 1/100 (10ms) as most analysis scripts do."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is the time axis itself, derived from the bin edges. It is not loaded from the raw data but computed from the binning parameters (TMIN, TMAX, DT).

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. Defined by the binning grid, same approach as reference.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing -- the time axis is the bin centers of the histogram edges, computed once and reused for every trial.

ii. N/A

iii. N/A

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis IS the binning grid. The bin centers used as input correspond exactly to the same bins that neural spike counts are placed into.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial fields: `obj.bp.R` (right instructed), `obj.bp.hit`, and `obj.bp.miss`. The actual lick side is inferred from the combination of instructed side and outcome.

ii.
```python
if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 1  # right
elif session['L'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 0  # left
else:
    lick_dir[i] = 2  # none
```

iii. The AI describes this as deriving lick direction from the instructed side combined with whether the animal responded.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI assigns: if the instructed side is R and the animal responded (hit OR miss), label as "right" (1); if L and responded, label as "left" (0); otherwise "none" (2). This is **incorrect** -- on miss trials the animal licks the opposite direction, but the AI assigns the instructed direction regardless. So R+miss should be "left" (animal licked wrong port) but is labeled "right".

ii.
```python
if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 1  # right  <-- wrong for miss trials
elif session['L'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 0  # left   <-- wrong for miss trials
else:
    lick_dir[i] = 2  # none
```

iii. The AI does not note this distinction in its documentation. Its CONVERSION_NOTES describe lick direction as derived from R/L fields but do not address the hit/miss asymmetry.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `obj.bp.autowater`. Autowater trials are labeled WC (0), all others DR (1).

ii.
```python
if session['autowater'][ti]:
    context[i] = 0  # WC
else:
    context[i] = 1  # DR
```

iii. Consistent with the reference approach.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct binary mapping: autowater=True -> WC (0), autowater=False -> DR (1).

ii. Same as 5-a.

iii. Straightforward encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` and `obj.bp.miss`. A hit is correct (1), a miss is incorrect (0), and anything else is ignore (2).

ii.
```python
if session['hit'][ti]:
    outcome[i] = 1  # correct
elif session['miss'][ti]:
    outcome[i] = 0  # incorrect
else:
    outcome[i] = 2  # ignore
```

iii. Consistent with the reference approach.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct categorical mapping from the hit/miss flags. Three classes: incorrect (0), correct (1), ignore (2).

ii. Same as 6-a.

iii. Straightforward encoding matching the prompt's specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, specifically the side camera (view 0). The 'tongue' feature is located by name in `featNames`. The `ts` array provides x, y, and confidence for each frame.

ii.
```python
view_data = session['traj'][0]  # side cam
feat_names = view_data.get('featNames', [])
tongue_idx = None
for fi, fn in enumerate(feat_names):
    if fn.lower() == 'tongue':
        tongue_idx = fi
        break
```

iii. The AI only uses the side camera for tongue tracking, unlike the reference which uses both side and bottom cameras and combines them.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Velocity is computed as speed = sqrt(dx^2 + dy^2) using `np.gradient` on the raw x and y positions (no smoothing applied first). Frames with confidence < 0.1 are set to NaN. The speed is then interpolated (not binned) onto the neural time axis using `np.interp`. The result is discretized at the per-session 50th percentile.

ii.
```python
xvel = np.gradient(x)
yvel = np.gradient(y)
speed = np.sqrt(xvel**2 + yvel**2)
speed[conf < 0.1] = np.nan

aligned_ft = frame_times - vidshift - go_cue_times[t]
interp_speed = np.interp(taxis, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. Key differences from reference: (1) No smoothing of x/y before differentiation, (2) confidence threshold is 0.1 instead of 0.9, (3) interpolation instead of bin-averaging, (4) only one camera view used, (5) no normalization by percentile to handle different camera scales.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of visible (non-NaN) values as the threshold. Values below are 0, at or above are 1, NaN (not visible) is 2.

ii.
```python
def discretize_velocity(speed_matrix):
    result = np.full_like(speed_matrix, 2, dtype=np.int8)
    valid = speed_matrix[~np.isnan(speed_matrix)]
    threshold = np.percentile(valid, 50)
    visible = ~np.isnan(speed_matrix)
    result[visible & (speed_matrix < threshold)] = 0
    result[visible & (speed_matrix >= threshold)] = 1
    return result
```

iii. Matches the prompt's specification for discretization.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by the video offset (`vidshift`) and go cue time, then the tongue speed is interpolated onto the neural time axis using `np.interp`. The video offset is computed as the difference of median bitcode start times between the recording and behavior clocks.

ii.
```python
aligned_ft = frame_times - vidshift - go_cue_times[t]
interp_speed = np.interp(taxis, aligned_ft, speed, left=np.nan, right=np.nan)
```

Video offset:
```python
bit_start_bpod = float(np.nanmedian(ev['bitStart'][0, :]))
bit_start_sglx = float(np.nanmedian(sglx['bitcode']['bitstart'][0, :])) / float(sglx['fs'][0, 0])
session['vidshift'] = bit_start_sglx - bit_start_bpod
```

iii. The AI uses `nanmedian` for the video offset computation instead of `mode` as in the reference's `findVideoOffset.m`. The reference uses `pd.Series().mode()`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking in `obj.traj` from the bottom camera (view 1). The first feature containing 'paw' in its name is used.

ii.
```python
view_data = session['traj'][1]  # bottom cam
for fi, fn in enumerate(feat_names):
    if 'paw' in fn.lower():
        paw_idx = fi
        break
```

iii. The AI selects the first paw feature found, which could be either `top_paw` or `bottom_paw`. The reference specifically uses `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue velocity: raw `np.gradient` on x and y (no smoothing), confidence threshold of 0.1, then NaN values are filled with nearest-neighbor interpolation. The speed is then interpolated onto the neural time axis and discretized at the per-session 50th percentile.

ii.
```python
xvel = np.gradient(x)
yvel = np.gradient(y)
speed = np.sqrt(xvel**2 + yvel**2)
speed[conf < 0.1] = np.nan

# Forward-fill then back-fill
mask = np.isnan(speed)
if mask.any() and not mask.all():
    idx = np.where(~mask)[0]
    speed = np.interp(np.arange(len(speed)), idx, speed[idx])
```

iii. The AI fills NaN values for the paw (but not tongue), creating a difference in how missing data is handled between the two features. The reference does not fill NaN values for either.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same discretization as tongue velocity: per-session 50th percentile, with class 2 for not visible.

ii. Same `discretize_velocity` function as tongue.

iii. Matches prompt specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset subtracted, go cue subtracted, then `np.interp` to the neural time axis.

ii. Same alignment code as tongue velocity.

iii. Same approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file `motionEnergy_<anm>_<date>.mat`, loaded via `scipy.io.loadmat` or `h5py` depending on format. The nested struct is unwrapped recursively (some files have `me.data`, others `me.data.data`).

ii.
```python
me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
me_arr = me_mat['me']

def _unwrap_me_data(arr):
    if hasattr(arr, 'dtype') and arr.dtype.names and 'data' in arr.dtype.names:
        inner = arr['data']
        if inner.shape == (1, 1):
            return _unwrap_me_data(inner[0, 0])
        return inner
    return arr
```

iii. The AI correctly identifies and handles the multiple wrapping formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The motion energy trace is interpolated to the neural time axis using `np.interp`. Then, all remaining NaN values are filled with nearest-neighbor interpolation. The result is discretized at the per-session 50th percentile.

ii.
```python
me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
me_data[:, t] = me_interp

# Fill NaNs with nearest
for t in range(ntrials):
    col = me_data[:, t]
    mask = np.isnan(col)
    if mask.any() and not mask.all():
        idx = np.where(~mask)[0]
        me_data[:, t] = np.interp(np.arange(len(col)), idx, col[idx])
```

iii. The AI fills NaNs for motion energy, which means the "no video" class (2) never appears. The verification output confirms: `motion_energy: {below_50pct (0.500), above_50pct (0.500)}` -- no class 2 at all. The NaN filling matches the MATLAB `fillmissing('nearest')` but prevents the "no video" category from being used.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile, same as other movement variables. Class 0 below, class 1 at/above, class 2 for NaN.

ii.
```python
def discretize_motion_energy(me_matrix):
    result = np.full_like(me_matrix, 2, dtype=np.int8)
    threshold = np.percentile(valid, 50)
    result[visible & (me_matrix < threshold)] = 0
    result[visible & (me_matrix >= threshold)] = 1
    return result
```

iii. In practice class 2 never appears because all NaNs were filled beforehand.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected with the video offset and go cue time, then motion energy is interpolated to the neural time axis.

ii.
```python
aligned_ft = frame_times - vidshift - go_cue_times[t]
me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
```

iii. Same alignment approach as tongue/paw velocity.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) If trajectory data is None, the trial's tongue/paw speed stays as NaN (becomes "not visible" class 2). (2) For paw velocity, NaNs are filled with nearest-neighbor interpolation. (3) For motion energy, NaNs are filled with nearest-neighbor interpolation. (4) Frame time / tracking data mismatches are handled by truncating to the shorter length. (5) Exceptions during loading are caught with bare `except:` and silently skipped. (6) Trials past the recording end are not excluded; the verification output shows ~64 trials across 3 sessions with all-zero neural data.

ii.
```python
# Paw NaN filling
mask = np.isnan(speed)
if mask.any() and not mask.all():
    idx = np.where(~mask)[0]
    speed = np.interp(np.arange(len(speed)), idx, speed[idx])

# Frame time mismatch
if len(frame_times) != len(trial_me):
    min_len = min(len(frame_times), len(trial_me))
    ...
```

iii. The AI does not drop trials with missing neural data (past recording end), which the reference handles by finding the last trial with spikes and dropping later ones. The bare `except:` clauses risk silently hiding real errors.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files is the dominant cost. The per-trial spike binning loop is also slow because it iterates over each unit and each trial individually. Total conversion runs in ~116 seconds.

ii.
```python
for ui, u in enumerate(all_units):
    for t in range(ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=EDGES)
```

iii. The AI notes "Per-trial spike binning loop (could be vectorized)" as an identified inefficiency.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The double loop over units and trials for spike binning (lines 832-843) could be vectorized with `np.histogram2d` as the reference does. The per-trial loops for tongue velocity, paw velocity, and motion energy loading could also potentially be vectorized but are complicated by varying frame counts.

ii.
```python
for ui, u in enumerate(all_units):
    for t in range(ntrials):
        spike_mask = trial_ids == trial_num
        counts, _ = np.histogram(aligned_times, bins=EDGES)
```

iii. The AI acknowledged this: "could be vectorized with sparse matrices but complexity not worth it for ~44 sessions."

## 11-c. What processing does the code repeat multiple times?

i. The per-trial spike binning involves re-creating the boolean spike mask for every trial for every unit, even though the trial assignments could be precomputed. Frame times for trajectory data are loaded independently for tongue, paw, and motion energy even though they share the same camera.

ii. Each of `compute_tongue_velocity`, `compute_paw_velocity`, and `load_motion_energy` independently load trajectory data per trial.

iii. The AI did not specifically identify repeated processing as an issue.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads event times for 'sample' and 'delay' that are never used. It loads `obj.bp.no` (ignore flag) that is never used (outcome is derived from hit/miss). It loads `obj.bp.L` which is redundant with `obj.bp.R`. The full loading functions (`load_session_h5`, `load_session_v5`) extract many fields that are not needed for the conversion. The NaN filling for motion energy prevents the "no video" class from being used, making the discretization function's class 2 dead code. The `plot_processing` function is included even in non-plotting runs.

ii.
```python
session['sample'] = ev['sample'][0, :]
session['delay'] = ev['delay'][0, :]
session['no'] = bp['no'][0, :].astype(bool)
session['L'] = bp['L'][0, :].astype(bool)
```

iii. The AI did not comment on these.
