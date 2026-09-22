# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from only the `Ephys_Behavior` directory, using `h5py` to read HDF5-format `.mat` files. It discovers session files by globbing `DATA_DIR.glob('data_structure_*.mat')` and then filters against a hardcoded `PROBE_MAP` dictionary that maps `(animal, date)` tuples to 0-indexed probe lists. Only 25 fixed-delay sessions are loaded; the 19 randomized-delay sessions in `RandomizedDelay_Ephys_Behavior` are completely omitted. Motion energy is loaded separately via `scipy.io.loadmat`.

ii. Session discovery and filtering:
```python
DATA_DIR = Path('/app/data/Ephys_Behavior')
...
data_files = sorted(DATA_DIR.glob('data_structure_*.mat'))
for data_file in data_files:
    ...
    key = (anm, date)
    if key not in PROBE_MAP:
        print(f"  Skipping {basename}: no probe mapping")
        continue
```

Loading a single session (HDF5 only):
```python
with h5py.File(data_path, 'r') as f:
    obj = f['obj']
    bp = obj['bp']
    ...
```

iii. The agent's reasoning (Step 25-26) acknowledged the existence of RandomizedDelay sessions but decided to include only the 25 Ephys_Behavior sessions. The agent reasoned: "Ephys_Behavior directory has 25 sessions from 10 animals." The agent did not implement a fallback to `scipy.io` for v5 MATLAB files, only using `h5py`.

## 1-b. How are the data split into subjects?

i. The animal name is extracted from the filename by splitting on `_` to get the first part. After all sessions are loaded, unique subjects are sorted and a `subject_idx` array maps each session to its subject. The AI also attempts to read `obj.meta.anm` from the HDF5 file as a fallback.

ii.
```python
basename = data_file.stem.replace('data_structure_', '')
parts = basename.split('_', 1)
anm = parts[0]
...
unique_subjects = sorted(set(session_animals))
subject_idx = np.array([unique_subjects.index(a) for a in session_animals])
```

iii. The agent extracts the animal name from the filename, consistent with the reference approach.

## 1-c. How are the data split into sessions?

i. One session is one `.mat` file in `Ephys_Behavior`. The AI globs for all `data_structure_*.mat` files in that single directory and filters against `PROBE_MAP`. Only 25 sessions from the fixed-delay task are included; the 19 randomized-delay sessions are omitted entirely.

ii.
```python
DATA_DIR = Path('/app/data/Ephys_Behavior')
...
data_files = sorted(DATA_DIR.glob('data_structure_*.mat'))
```

iii. The agent acknowledged the RandomizedDelay sessions exist but chose to only include Ephys_Behavior sessions. The final summary states "25 sessions from 10 subjects."

## 1-d. How are the data split into trials?

i. Trials are defined by the per-trial fields of `obj.bp`. The number of trials is read from `bp['Ntrials']` and per-trial arrays (hit, miss, no, R, L, etc.) are read as row vectors from the HDF5 file. Each trial index corresponds to one entry in these arrays.

ii.
```python
ntrials = int(bp['Ntrials'][0, 0])
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
...
```

iii. The agent reads trial data directly from the Bpod structure, which is consistent with how the reference handles it.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are excluded. Additionally, a validity check requires trials to be hit, miss, or no (ignore). There is no filter for trials that extend past the end of the recording.

ii.
```python
valid_mask = (hit | miss | no) & ~stim_enable & ~early
valid_trials = np.where(valid_mask)[0]
```

iii. The agent follows the paper's exclusion of early-lick and photostim trials. However, there is no check for trials that run past the end of the recording (which the reference implements).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `obj.clu{probe}` cluster data: `trialtm` (spike times relative to trial start), `trial` (trial assignments, 1-based), and `quality` (manual curation labels). Go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
trialtm = f[trialtm_ref][()].flatten()
trial_nums = f[trial_ref][()].flatten().astype(int)
aligned_times = np.empty_like(trialtm)
for t_idx in range(len(trialtm)):
    tr = trial_nums[t_idx] - 1
    if 0 <= tr < ntrials:
        aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```

iii. Consistent with the reference: `trialtm` is the spike time relative to trial start and `goCue` is the alignment event.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, binned into 10 ms bins (DT = 1/100), converted to firing rates (Hz), and smoothed with a **causal** Gaussian kernel of window size 15 bins. The causal kernel zeros out the first half of a `gausswin(15)` and normalizes.

ii.
```python
DT = 1 / 100         # 10 ms bins
SMOOTH_WIN = 15       # causal Gaussian kernel window (bins)

def causal_gaussian_kernel(N):
    alpha = 2.5
    n = np.arange(N)
    center = (N - 1) / 2.0
    sigma = center / alpha
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    kern[:int(np.floor(len(kern) / 2))] = 0
    kern = kern / kern.sum()
    return kern
```

```python
def bin_and_smooth_spikes(spike_times, edges, dt, smooth_win, bctype):
    counts, _ = np.histogram(spike_times, bins=edges)
    fr = counts.astype(float) / dt
    fr = smooth_data(fr, smooth_win, bctype)
    return fr
```

iii. The agent chose a causal Gaussian kernel based on reading `mySmooth.m` from the reference code. However, the reference solution uses a symmetric Gaussian with sigma = 14 ms (matching `gausswin(15)` at 5 ms bins). The AI's 10 ms bin size differs from the reference's 5 ms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels matching `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded. Clusters with empty quality labels are also excluded (the AI skips clusters with `quality == ''`). Then units with mean firing rate <= 1 Hz are removed. Additionally, sessions with fewer than 10 units after filtering are skipped entirely.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
if quality.lower() in {q.lower() for q in EXCLUDE_QUALITIES}:
    continue
if quality == '':
    continue
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=1)
fr_mask = mean_fr > LOW_FR
```

iii. The agent excludes empty quality labels, which differs from the reference (which keeps them). The reference also excludes 'poor' quality, which the AI does not. The 10-unit minimum threshold for sessions is not in the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's `trialtm` is subtracted by its trial's `goCue` time, giving spike time relative to go cue onset. This is done per-spike in a loop.

ii.
```python
aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```

iii. Consistent with the reference approach of `trialtm - goCue[trial]`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (DT = 1/100), producing 500 time bins over the -2.5 to 2.5 s window. No rebinning is applied after the initial binning.

ii.
```python
DT = 1 / 100         # 10 ms bins
...
edges = np.arange(TMIN, TMAX + DT, DT)
n_timebins = len(edges) - 1
```

iii. The agent states in the docstring: "Bin size: 10 ms (dt = 1/100)" and references `params.dt = 1/100`. The reference uses 5 ms bins (params.dt = 1/200), yielding 1000 time bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the bin centers of the time grid, computed from `TMIN`, `TMAX`, and `DT`. It is not derived from any raw data variable.

ii.
```python
time_vec = edges[:-1] + DT / 2
...
input_trial = neural_time.reshape(1, -1).astype(np.float32)
```

iii. Same conceptual approach as the reference: the input is the time axis itself.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers: `edges[:-1] + DT/2`, yielding 500 values from -2.495 to 2.495 s.

ii.
```python
time_vec = edges[:-1] + DT / 2
```

iii. Straightforward computation of bin centers.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector uses the same bin edges as the neural data, so alignment is implicit — bin k in the input corresponds to the same time interval as bin k in the neural data.

ii.
```python
neural_time = time_vec  # (n_timebins,)
input_trial = neural_time.reshape(1, -1).astype(np.float32)
```

iii. Same approach as reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R` (right instruction), `bp.L` (left instruction), `bp.hit`, `bp.miss`, and `bp.no` (ignore). The lick direction is inferred from the combination of instruction side and outcome.

ii.
```python
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
R = bp['R'][0, :].astype(bool)
L = bp['L'][0, :].astype(bool)
...
lick_direction[(R & hit) | (L & miss)] = 1   # right
lick_direction[(L & hit) | (R & miss)] = 0   # left
lick_direction[no] = 2                         # none
```

iii. The agent correctly derives lick direction from instruction side and outcome flags.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port, a miss means the other. `R & hit` or `L & miss` → right (1); `L & hit` or `R & miss` → left (0); `no` → none (2). The encoding uses `left=0, right=1, none=2`.

ii. See 4-a code snippet.

iii. Consistent with the reference logic, though the reference initializes to `no lick` and then sets hits/misses, while the AI initializes to -1 and explicitly sets `no` trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` — autowater trials are WC (water-cued) context, others are DR (delayed-response).

ii.
```python
autowater = bp['autowater'][0, :].astype(bool)
context = np.where(autowater, 0, 1).astype(int)  # 0=WC, 1=DR
```

iii. Consistent with the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater → WC (0), not autowater → DR (1).

ii. See 5-a code snippet.

iii. Same as reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit`, `bp.miss`, and `bp.no` flags.

ii.
```python
outcome = np.full(ntrials, -1, dtype=int)
outcome[miss] = 0   # incorrect
outcome[hit] = 1    # correct
outcome[no] = 2     # ignore
```

iii. Consistent with reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: miss → incorrect (0), hit → correct (1), no → ignore (2).

ii. See 6-a code snippet.

iii. Same encoding as reference.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking from `obj.traj`, specifically the **side camera only** (view 0), tracking the `tongue` feature. The AI reads `ts` (tracked x, y, likelihood per feature per frame) and `frameTimes` from the side camera.

ii.
```python
side_data = f[traj_group[0, 0]]
side_feats = read_feat_names(f, side_data)
tongue_idx = side_feats.index('tongue') if 'tongue' in side_feats else None
...
ts = f[side_data['ts'][tr, 0]][()]  # (n_feats, 3, n_frames)
tongue_x = ts[tongue_idx, 0, :]
tongue_y = ts[tongue_idx, 1, :]
```

iii. The agent only uses the side camera for tongue tracking. The reference uses both side (`tongue`) and bottom (`top_tongue`) cameras and averages them after normalizing each by its 90th percentile.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. NaN positions (where the tongue is not visible) are identified. For visible frames, `nearest_fill` is applied to x and y coordinates (forward-then-backward nearest-neighbor interpolation), then speed is computed from the filled positions using `np.gradient` with a fixed video dt of 1/400 s. The speed at invisible frames is set to NaN. The speed is then interpolated to neural time bins using `np.interp`, and discretized at the session 50th percentile.

ii.
```python
tongue_nan = np.isnan(tongue_x) | np.isnan(tongue_y)
...
tx_filled = nearest_fill(tongue_x)
ty_filled = nearest_fill(tongue_y)
speed_raw = compute_speed(tx_filled, ty_filled, dt_video)
speed[visible] = speed_raw[visible]
```

```python
speed_interp = np.interp(neural_time, ft, speed_for_interp, left=np.nan, right=np.nan)
```

iii. The agent uses nearest-fill interpolation on positions before computing speed, then only keeps speed at visible frames. The reference instead computes speed within contiguous runs of valid frames after Gaussian smoothing (sigma=5ms), never filling across gaps.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile of all visible tongue speeds across the session is computed, then: visible speeds below threshold → 0, above/equal → 1, not visible → 2.

ii.
```python
tongue_thresh = np.median(all_tongue_speeds) if len(all_tongue_speeds) > 0 else 0
...
tv[vis & (speed_interp < tongue_thresh)] = 0
tv[vis & (speed_interp >= tongue_thresh)] = 1
```

iii. Uses `np.median` (equivalent to 50th percentile). Consistent with reference approach.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned by subtracting a hardcoded offset of 0.5 s (`PAD_SEC`) and the trial's go cue time: `aligned_ft = ft - PAD_SEC - goCue[tr]`. The aligned frame-level speed is then linearly interpolated (`np.interp`) to the neural time bins.

ii.
```python
PAD_SEC = 0.5
...
aligned_ft = ft - PAD_SEC - goCue[tr]
...
speed_interp = np.interp(neural_time, ft, speed_for_interp, left=np.nan, right=np.nan)
```

iii. The agent uses a hardcoded 0.5 s offset. The reference computes the video offset from bitcode timing (`sglx.bitcode.bitstart / sglx.fs` minus `bp.ev.bitStart`), which varies by session. The AI also uses linear interpolation to neural bins rather than averaging frames within bins.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DeepLabCut tracking from the **bottom camera** (view 1), tracking the `top_paw` feature.

ii.
```python
bot_data = f[traj_group[1, 0]]
bot_feats = read_feat_names(f, bot_data)
paw_idx = bot_feats.index('top_paw') if 'top_paw' in bot_feats else None
```

iii. Same feature and camera as reference (`top_paw` from the bottom camera).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: NaN positions are nearest-filled, speed computed with `np.gradient` at fixed 1/400 s dt, then only visible-frame speeds are kept. Speed is interpolated to neural time bins and discretized.

ii.
```python
px = nearest_fill(paw_x)
py = nearest_fill(paw_y)
speed_paw = compute_speed(px, py, dt_video)
paw_visible = ~paw_nan
paw_speed_trials[tr] = (aligned_ft, speed_paw, paw_visible)
```

iii. Uses nearest-fill interpolation instead of the reference's within-run Gaussian smoothing approach.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: 50th percentile threshold, with 0 = below, 1 = above/equal, 2 = not visible.

ii.
```python
paw_thresh = np.median(all_paw_speeds) if len(all_paw_speeds) > 0 else 0
```

iii. Consistent with reference.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: `frameTimes - PAD_SEC - goCue[tr]`, then `np.interp` to neural time bins.

ii.
```python
aligned_ft = ft - PAD_SEC - goCue[tr]
speed_interp = np.interp(neural_time, ft, speed, left=np.nan, right=np.nan)
```

iii. Uses hardcoded 0.5 s offset and linear interpolation, same issues as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `motionEnergy_<anm>_<date>.mat` files loaded via `scipy.io.loadmat`. The per-trial motion energy data is extracted from the `me` struct's `data` field.

ii.
```python
me_data = scipy.io.loadmat(me_path)
me_struct = me_data['me']
me_trial_data = me_struct['data'][0, 0]  # (ntrials, 1) cell array
```

iii. Same source as reference, though the loading method differs (the reference unwraps nested dict wrappers in a loop).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy values are interpolated to neural time bins using `np.interp` with the side camera's aligned frame times. No additional smoothing or processing.

ii.
```python
me_interp = np.interp(neural_time, ft, me_trial, left=np.nan, right=np.nan)
```

iii. Uses linear interpolation to neural bins rather than the reference's bin-averaging approach.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. 50th percentile of all valid motion energy values across the session, then: below → 0, above/equal → 1, no video → 2.

ii.
```python
me_thresh_50 = np.median(all_me_values) if len(all_me_values) > 0 else 0
...
me_disc[vis & (me_interp < me_thresh_50)] = 0
me_disc[vis & (me_interp >= me_thresh_50)] = 1
```

iii. Consistent with reference.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Uses the side camera's frame times (stored during tongue processing), aligned with `frameTimes - PAD_SEC - goCue[tr]`, then interpolated to neural time bins. Motion energy is only processed for trials where side camera frame times were successfully loaded.

ii.
```python
if len(me_trial) == len(ft):
    me_interp = np.interp(neural_time, ft, me_trial, left=np.nan, right=np.nan)
```

iii. Depends on successful tongue processing for frame times, which could miss trials where tongue failed but ME is available. Uses hardcoded offset and linear interpolation.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses broad `try/except` blocks that silently catch exceptions and mark data as `None` or unavailable. For tongue/paw positions, NaN values are nearest-filled before computing speed. If video data fails to load entirely, all video outputs default to "not visible" (2). Sessions with fewer than 10 units or fewer than 2 valid trials are skipped entirely.

ii.
```python
except Exception:
    tongue_speed_trials[tr] = None
...
except Exception as e:
    print(f"    Warning: could not load video data: {e}")
    has_video = False
```

iii. The agent uses nearest-fill for missing position data, while the reference never interpolates across gaps. The reference keeps trials with missing video and assigns them the "not visible" class. The AI's broad exception handling could mask real errors.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files is the most time-consuming step. The per-spike loop for alignment is also expensive (iterating over every spike to subtract go cue time), as is the per-unit, per-trial loop for binning and smoothing.

ii.
```python
for t_idx in range(len(trialtm)):
    tr = trial_nums[t_idx] - 1
    if 0 <= tr < ntrials:
        aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```

iii. The reference vectorizes spike alignment in a single subtraction. The AI's per-spike Python loop is significantly slower.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike alignment loop (per-spike) could be vectorized as a single array subtraction. The per-unit, per-trial spike binning loop (nested for u_idx, for tr) could be replaced with `np.histogram2d`. The per-trial video processing loops could potentially be partially vectorized.

ii.
```python
for u_idx in range(n_units):
    ...
    for tr in range(ntrials):
        tr_mask = strials == (tr + 1)
        ...
        trialdat[u_idx, tr, :] = bin_and_smooth_spikes(...)
```

iii. The reference uses `np.histogram2d` to bin all trials at once per cluster, avoiding the inner trial loop.

## 11-c. What processing does the code repeat multiple times?

i. The HDF5 file is opened twice: once for the main data loading and once at the end to read the animal name from `obj.meta.anm`. Feature names are potentially read multiple times across trials (though only one trial's names are used).

ii.
```python
# First open:
with h5py.File(data_path, 'r') as f:
    ...  # main processing

# Second open:
with h5py.File(data_path, 'r') as f:
    try:
        anm = h5_read_string(f, f['obj']['meta']['anm'])
```

iii. The second file open is unnecessary since the animal name could be extracted from the filename (which is done as a fallback anyway).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads `sample_times` and `delay_times` from `bp.ev` but never uses them. The `nearest_fill` function interpolates positions that are ultimately masked out (only visible-frame speeds are kept). The `me_thresh` from the motion energy file is read but not used (a 50th percentile is computed instead).

ii.
```python
sample_times = ev['sample'][0, :]
delay_times = ev['delay'][0, :]
...
me_thresh = float(me_struct['moveThresh'][0, 0].flatten()[0])
```

iii. These are minor inefficiencies that don't affect correctness.
