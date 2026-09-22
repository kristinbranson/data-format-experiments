# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code only loads sessions from `/app/data/Ephys_Behavior`, not from `RandomizedDelay_Ephys_Behavior`. It discovers sessions by globbing `data_structure_*.mat`, pairs each one with a same-folder `motionEnergy_*.mat` file if present, and opens the session file only through `h5py.File`, so it assumes HDF5/v7.3 layout rather than supporting both MATLAB file formats.

ii.
```python
DATA_DIR = Path('/app/data/Ephys_Behavior')
...
data_files = sorted(DATA_DIR.glob('data_structure_*.mat'))
...
with h5py.File(data_path, 'r') as f:
    obj = f['obj']
```

iii. In the trajectory, the AI explicitly decided to keep only the 25 `Ephys_Behavior` sessions because it viewed randomized-delay data as a separate experiment. It justified the loader as following `WorkingWithDataObjs.m` and the `load*_ALMVideo.m` scripts, and it treated the fixed-delay directory as the relevant analysis set.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the filename prefix before the first underscore, stored per session as `animal`, then deduplicated and sorted to build `subjects`; `subject_idx` maps each session to that sorted subject list.

ii.
```python
basename = data_file.stem.replace('data_structure_', '')
parts = basename.split('_', 1)
anm = parts[0]
...
return {
    ...
    'animal': anm,
}
...
unique_subjects = sorted(set(session_animals))
subject_idx = np.array([unique_subjects.index(a) for a in session_animals])
```

iii. The trajectory says the agent relied on filenames because it was enumerating sessions from disk and compiling probe assignments from the loading scripts by animal/date.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_<animal>_<date>.mat` file found in `/app/data/Ephys_Behavior`. The code uses a hard-coded `PROBE_MAP` to decide which ALM probe(s) to use for each session, and each successfully processed file becomes one session entry in `neural`, `input`, and `output`.

ii.
```python
PROBE_MAP = {
    ('EKH1', '2021-08-07'): [1],
    ...
    ('JGR3', '2021-11-18'): [0],
}
...
for data_file in data_files:
    ...
    key = (anm, date)
    if key not in PROBE_MAP:
        print(f"  Skipping {basename}: no probe mapping")
        continue
```

iii. The trajectory says the AI chose all 25 `Ephys_Behavior` sessions and treated randomized-delay sessions as excluded. It justified probe selection as being transcribed from the `load*_ALMVideo.m` scripts.

## 1-d. How are the data split into trials?

i. Trials are taken directly from per-trial arrays in `obj['bp']`, using `Ntrials` as the trial count. The code indexes trial-wise behavioral arrays by `tr`, bins spikes trial-by-trial using `trial_nums == (tr + 1)`, and stores one converted trial object per surviving trial index.

ii.
```python
ntrials = int(bp['Ntrials'][0, 0])
...
valid_trials = np.where(valid_mask)[0]
...
for tr in range(ntrials):
    tr_mask = strials == (tr + 1)
    ...
for tr in valid_trials:
    neural_trial = trialdat[:, tr, :]
    session_neural.append(neural_trial.astype(np.float32))
```

iii. The trajectory treated one row of the Bpod trial table as one trial and used the per-spike trial assignments already stored in the raw data, so it did not attempt to reconstruct trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with `valid_mask = (hit | miss | no) & ~stim_enable & ~early`, so early-lick and stim-enabled trials are removed. Sessions with fewer than two remaining trials are skipped entirely. The code does not implement the reference solution's extra cutoff for behavioral trials that continue after recording has effectively stopped.

ii.
```python
valid_mask = (hit | miss | no) & ~stim_enable & ~early
valid_trials = np.where(valid_mask)[0]

if len(valid_trials) < 2:
    print(f"    Skipping: only {len(valid_trials)} valid trials")
    return None
```

iii. The trajectory repeatedly states that the trial filter should match the paper by excluding stim and early-lick trials, and that hit, miss, and ignore trials should be retained for decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `obj['clu']` on the selected ALM probe(s), specifically each cluster's `quality`, `trialtm`, and `trial` arrays, together with `bp.ev.goCue` for alignment.

ii.
```python
goCue = ev['goCue'][0, :]
...
quality_ref = prb_data['quality'][clu_i, 0]
...
trialtm_ref = prb_data['trialtm'][clu_i, 0]
trial_ref = prb_data['trial'][clu_i, 0]

trialtm = f[trialtm_ref][()].flatten()
trial_nums = f[trial_ref][()].flatten().astype(int)
```

iii. The trajectory says the AI was following the ALM loading scripts and using trial-aligned spike times plus the go cue to build go-cue-centered firing rates.

## 2-b. How is the `neural` data processed?

i. For each unit and each trial, the code subtracts the trial's go cue from every spike time, bins spikes from `-2.5` to `2.5` s in `10 ms` bins, converts counts to firing rate by dividing by `DT`, and smooths with a custom causal Gaussian kernel of length 15 using a reflect-style padding scheme.

ii.
```python
DT = 1 / 100
SMOOTH_WIN = 15
...
aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
...
counts, _ = np.histogram(spike_times, bins=edges)
fr = counts.astype(float) / dt
fr = smooth_data(fr, smooth_win, bctype)
```

iii. The trajectory explicitly justifies `10 ms` bins by preferring the tutorial code path in `WorkingWithDataObjs.m` over the default analysis parameters, and it says the smoothing should be a causal Gaussian with `smooth=15` and `bctype='reflect'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code drops clusters whose lower-cased quality is in `{'garbage', 'gabrga', 'noisy', 'real?'}` and also drops clusters with empty quality strings. After building trial firing rates, it removes units with mean firing rate `<= 1 Hz`. It also skips whole sessions that end up with fewer than 10 units before or after firing-rate filtering.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
...
if quality.lower() in {q.lower() for q in EXCLUDE_QUALITIES}:
    continue
if quality == '':
    continue
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=1)
fr_mask = mean_fr > LOW_FR
trialdat = trialdat[fr_mask]
...
if n_units_final < 10:
    return None
```

iii. The trajectory says the AI chose the `findClusters.m`-style `'all'` quality setting, interpreted as excluding garbage/noisy-style labels, and applied the paper's `>1 Hz` firing-rate cutoff. It also decided to keep only sessions meeting a minimum usable unit count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned by subtracting the go-cue time of the spike's own trial from `trialtm`, producing spike times in seconds relative to go cue before histogramming.

ii.
```python
for t_idx in range(len(trialtm)):
    tr = trial_nums[t_idx] - 1
    if 0 <= tr < ntrials:
        aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```

iii. The trajectory states that go-cue alignment is the required event and that spikes are already on the behavior clock, so only the trial-specific subtraction is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use `10 ms` bins over `-2.5` to `2.5` s, giving 500 bins per trial. This is a rebinning of spike times into a coarser neural grid than the reference solution.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 100
...
edges = np.arange(TMIN, TMAX + DT, DT)
n_timebins = len(edges) - 1
time_vec = edges[:-1] + DT / 2
```

iii. The trajectory explicitly discusses the `1/100` versus `1/200` discrepancy and says the AI chose `10 ms` because it trusted the tutorial code more than the defaults.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a dedicated raw variable. It is constructed from the neural bin centers defined by `TMIN`, `TMAX`, and `DT`, with the meaning of "time from go cue" inherited from the spike alignment to `bp.ev.goCue`.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_vec = edges[:-1] + DT / 2
...
input_trial = neural_time.reshape(1, -1).astype(np.float32)
```

iii. The trajectory says the time input is a decoder-design choice: a go-cue-centered continuous time vector shared across trials.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code simply computes bin centers once from the chosen time window and bin size, then reshapes that vector to `(1, n_timebins)` and reuses it for every valid trial in the session.

ii.
```python
time_vec = edges[:-1] + DT / 2
...
neural_time = time_vec
...
input_trial = neural_time.reshape(1, -1).astype(np.float32)
session_input.append(input_trial)
```

iii. The trajectory gives no extra processing beyond defining a shared time axis centered on the go cue.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same bin-center vector used for the neural histogram grid, so each input timepoint corresponds to the neural bin at the same index.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_vec = edges[:-1] + DT / 2
...
trialdat[u_idx, tr, :] = bin_and_smooth_spikes(
    tr_spikes, edges, DT, SMOOTH_WIN, SMOOTH_BC
)
...
input_trial = neural_time.reshape(1, -1).astype(np.float32)
```

iii. The trajectory says the time input should be "shared across trials" and aligned directly to the same go-cue-centered neural bins.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial behavioral flags `R`, `L`, `hit`, `miss`, and `no`.

ii.
```python
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
no = bp['no'][0, :].astype(bool)
R = bp['R'][0, :].astype(bool)
L = bp['L'][0, :].astype(bool)
```

iii. The trajectory says `R`/`L` indicate the instructed or rewarded side, so actual lick direction must be inferred by combining those flags with hit/miss outcomes, with ignore/no-lick trials as a third class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code assigns right on `(R & hit) | (L & miss)`, left on `(L & hit) | (R & miss)`, and none on `no`. That per-trial class is then broadcast across all time bins for the trial.

ii.
```python
lick_direction = np.full(ntrials, -1, dtype=int)
lick_direction[(R & hit) | (L & miss)] = 1
lick_direction[(L & hit) | (R & miss)] = 0
lick_direction[no] = 2
...
output_trial[0, :] = lick_direction[tr]
```

iii. The trajectory explicitly reasons through this truth table and says ignore trials should map to a third "none/no lick" class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag in `obj['bp']`.

ii.
```python
autowater = bp['autowater'][0, :].astype(bool)
```

iii. The trajectory says `autowater` marks water-cued trials and can therefore be used to label WC versus DR context directly.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater=True` to `0` (WC) and `False` to `1` (DR), then broadcasts that trial label across time.

ii.
```python
context = np.where(autowater, 0, 1).astype(int)
...
output_trial[1, :] = context[tr]
```

iii. The trajectory justifies this as a direct relabeling of WC versus DR trials for decoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial `hit`, `miss`, and `no` behavioral flags.

ii.
```python
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
no = bp['no'][0, :].astype(bool)
```

iii. The trajectory says hit/miss/no are the natural three-way behavioral outcome partition and should be kept as correct, incorrect, and ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code assigns `0` to miss, `1` to hit, and `2` to no/ignore, then broadcasts the resulting per-trial category across time bins.

ii.
```python
outcome = np.full(ntrials, -1, dtype=int)
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
...
output_trial[2, :] = outcome[tr]
```

iii. The trajectory explicitly says the output should contain incorrect, correct, and ignore classes.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. In the final code, tongue velocity is derived only from the side camera trajectory entry `obj['traj'][0, 0]`, specifically its per-trial `frameTimes`, `featNames`, and `ts` arrays for the feature named `'tongue'`, together with `goCue` and a fixed `PAD_SEC` offset.

ii.
```python
PAD_SEC = 0.5
...
side_data = f[traj_group[0, 0]]
side_feats = read_feat_names(f, side_data)
tongue_idx = side_feats.index('tongue') if 'tongue' in side_feats else None
...
ft = f[side_data['frameTimes'][tr, 0]][()].flatten()
ts = f[side_data['ts'][tr, 0]][()]
tongue_x = ts[tongue_idx, 0, :]
tongue_y = ts[tongue_idx, 1, :]
```

iii. The trajectory says the AI intended to use tongue information from both side and bottom cameras because visibility differs by view, but the final implementation only uses the side-camera tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code marks visibility from `NaN` positions, nearest-fills `x` and `y`, computes frame-rate speed magnitude with `np.gradient` at a fixed `400 Hz`, linearly interpolates that speed to neural time bins, then thresholds the interpolated visible samples at the per-session median. There is no Gaussian smoothing, no explicit likelihood threshold, and no cross-camera normalization/averaging in the final implementation.

ii.
```python
dt_video = 1.0 / 400
...
tongue_nan = np.isnan(tongue_x) | np.isnan(tongue_y)
visible = ~tongue_nan
speed = np.full_like(tongue_x, np.nan)
if visible.sum() > 1:
    tx_filled = nearest_fill(tongue_x)
    ty_filled = nearest_fill(tongue_y)
    speed_raw = compute_speed(tx_filled, ty_filled, dt_video)
    speed[visible] = speed_raw[visible]
...
speed_interp = np.interp(neural_time, ft, speed_for_interp, left=np.nan, right=np.nan)
```

iii. In the trajectory, the AI justified tongue invisibility from missing DLC positions and aimed to compute speed magnitude, keep a separate invisibility class, and threshold at the session 50th percentile. It also stated that tongue NaNs should remain meaningful visibility markers, although the code still uses nearest-filled coordinates to compute speed on visible frames.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code collects all finite, visible tongue speeds in a session, sets `tongue_thresh` to their median, and labels visible bins below threshold as `0`, visible bins at or above threshold as `1`, and all other bins as `2` (`not visible`).

ii.
```python
all_tongue_speeds.extend(valid_speeds.tolist())
...
tongue_thresh = np.median(all_tongue_speeds) if len(all_tongue_speeds) > 0 else 0
...
tv = np.full(n_timebins, 2, dtype=np.int64)
vis = visible_interp & np.isfinite(speed_interp)
tv[vis & (speed_interp < tongue_thresh)] = 0
tv[vis & (speed_interp >= tongue_thresh)] = 1
```

iii. The trajectory explicitly says to use a per-session 50th-percentile threshold with a third class for bins where the tongue is not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. For each trial, the side-camera `frameTimes` are shifted by a fixed `0.5 s` offset and the trial's `goCue`, then the resulting frame-level speed is linearly interpolated onto the neural time vector.

ii.
```python
aligned_ft = ft - PAD_SEC - goCue[tr]
...
speed_interp = np.interp(neural_time, ft, speed_for_interp,
                         left=np.nan, right=np.nan)
visible_interp = np.interp(neural_time, ft, visible.astype(float),
                           left=0, right=0) > 0.5
```

iii. The trajectory says the AI deliberately chose the simpler tutorial-style alignment `frameTimes - 0.5 - goCue` instead of computing a session-specific bitcode offset with `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera trajectory entry `obj['traj'][1, 0]`, specifically `frameTimes`, `featNames`, and `ts` for the feature `'top_paw'`, together with `goCue` and the fixed `PAD_SEC` offset.

ii.
```python
bot_data = f[traj_group[1, 0]]
bot_feats = read_feat_names(f, bot_data)
paw_idx = bot_feats.index('top_paw') if 'top_paw' in bot_feats else None
...
ft = f[bot_data['frameTimes'][tr, 0]][()].flatten()
ts = f[bot_data['ts'][tr, 0]][()]
paw_x = ts[paw_idx, 0, :]
paw_y = ts[paw_idx, 1, :]
```

iii. The trajectory says the paw stream should come from the bottom camera's paw feature and be processed similarly to other kinematic variables.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code detects missing frames from `NaN` coordinates, nearest-fills those coordinates, computes speed magnitude with `np.gradient` at a fixed `400 Hz`, linearly interpolates speed to neural bins, and later median-thresholds visible bins. It does not smooth positions first and does not use a likelihood cutoff.

ii.
```python
paw_nan = np.isnan(paw_x) | np.isnan(paw_y)
...
px = nearest_fill(paw_x)
py = nearest_fill(paw_y)
speed_paw = compute_speed(px, py, dt_video)
paw_visible = ~paw_nan
...
speed_interp = np.interp(neural_time, ft, speed,
                         left=np.nan, right=np.nan)
```

iii. The trajectory says the AI intended nearest-value filling for non-tongue features and a speed-magnitude representation, with a separate "not visible" class where tracking is missing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The code pools visible paw speeds across the session, sets `paw_thresh` to their median, labels visible bins below threshold as `0`, visible bins at or above threshold as `1`, and invisible bins as `2`.

ii.
```python
all_paw_speeds.extend(valid_speeds.tolist())
...
paw_thresh = np.median(all_paw_speeds) if len(all_paw_speeds) > 0 else 0
...
pv = np.full(n_timebins, 2, dtype=np.int64)
vis = visible_interp & np.isfinite(speed_interp)
pv[vis & (speed_interp < paw_thresh)] = 0
pv[vis & (speed_interp >= paw_thresh)] = 1
```

iii. The trajectory explicitly says to use a per-session 50th-percentile split with a third category for invisibility.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are shifted by `PAD_SEC` and `goCue`, then paw speeds are linearly interpolated onto the neural bin centers.

ii.
```python
aligned_ft = ft - PAD_SEC - goCue[tr]
...
speed_interp = np.interp(neural_time, ft, speed,
                         left=np.nan, right=np.nan)
visible_interp = np.interp(neural_time, ft, visible.astype(float),
                           left=0, right=0) > 0.5
```

iii. The trajectory says the AI chose the same fixed-offset video alignment for all camera-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the separate `motionEnergy_<animal>_<date>.mat` file. The code reads `me_data['me']['data'][0, 0]` as the per-trial motion-energy traces and also reads `moveThresh`, though that threshold is not used later.

ii.
```python
me_data = scipy.io.loadmat(me_path)
me_struct = me_data['me']
me_trial_data = me_struct['data'][0, 0]
me_thresh = float(me_struct['moveThresh'][0, 0].flatten()[0])
```

iii. The trajectory says motion energy should come from the separate motion-energy files and be aligned to the neural timeline before discretization.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code does not smooth or differentiate motion energy. Instead, for trials whose motion-energy vector length matches the stored side-camera frame times, it linearly interpolates frame-level values onto neural time bins, pools all finite interpolated values across the session, and thresholds them at the median.

ii.
```python
if len(me_trial) == len(ft):
    me_interp = np.interp(neural_time, ft, me_trial,
                          left=np.nan, right=np.nan)
    me_all[tr] = me_interp
    all_me_values.extend(me_interp[np.isfinite(me_interp)].tolist())
...
me_thresh_50 = np.median(all_me_values) if len(all_me_values) > 0 else 0
```

iii. The trajectory says motion energy already exists as a per-frame scalar and therefore only needs alignment and per-session median discretization.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The code sets `me_thresh_50` to the session median of all finite interpolated motion-energy values, then labels finite bins below threshold as `0`, finite bins at or above threshold as `1`, and missing bins as `2` (`no video`).

ii.
```python
me_thresh_50 = np.median(all_me_values) if len(all_me_values) > 0 else 0
...
me_disc = np.full(n_timebins, 2, dtype=np.int64)
vis = np.isfinite(me_interp)
me_disc[vis & (me_interp < me_thresh_50)] = 0
me_disc[vis & (me_interp >= me_thresh_50)] = 1
```

iii. The trajectory explicitly says to discretize motion energy with a per-session 50th-percentile threshold and a third class for no-video bins.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned indirectly through the side-camera frame times stored earlier for the same trial. Those frame times were shifted by the fixed `0.5 s` offset and the trial's `goCue`, then motion-energy values are linearly interpolated onto neural time bins.

ii.
```python
side_frame_times[tr] = aligned_ft
...
ft = side_frame_times[tr]
...
me_interp = np.interp(neural_time, ft, me_trial,
                      left=np.nan, right=np.nan)
```

iii. The trajectory says motion energy should share the same video-to-neural alignment as the camera features, using the side-camera timing.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI code generally keeps trials and encodes failures as category `2`, but it also fills missing positions with `nearest_fill` before computing speeds. Missing or unreadable tongue/paw trials are set to `None` by broad `try/except` blocks and become all `not visible`. Motion-energy length mismatches or load failures become `None` and later all `no video`.

ii.
```python
def nearest_fill(arr):
    ...
    return out
...
try:
    ...
except Exception:
    tongue_speed_trials[tr] = None
...
except Exception:
    paw_speed_trials[tr] = None
...
if tr in me_all and me_all[tr] is not None:
    ...
else:
    output_trial[5, :] = 2
```

iii. The trajectory says the AI wanted NaNs to stand for invisibility, wanted nearest-value filling for non-tongue features, and preferred keeping trials with a third "not visible"/"no video" class rather than dropping them.

## 11-a. What are the most time-consuming steps of the code?

i. In this AI-written code, the most expensive operations are the nested per-unit, per-trial neural loops and the per-trial video interpolation loops, not just file loading. The script bins and smooths every unit separately for every trial and then loops again over all valid trials to interpolate tongue, paw, and motion-energy traces.

ii.
```python
for u_idx in range(n_units):
    ...
    for tr in range(ntrials):
        ...
        trialdat[u_idx, tr, :] = bin_and_smooth_spikes(...)
...
for tr in valid_trials:
    ...
    speed_interp = np.interp(...)
    ...
    me_interp = np.interp(...)
```

iii. The trajectory did not contain a runtime profile, but it does show the AI repeatedly debugging the video-extraction path and retaining these explicit Python loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized: the per-spike alignment loop over `t_idx`, the nested unit-by-trial neural loop, the `nearest_fill` forward/backward passes, and much of the per-trial interpolation/thresholding pipeline for tongue, paw, and motion energy.

ii.
```python
for t_idx in range(len(trialtm)):
    tr = trial_nums[t_idx] - 1
    ...
for u_idx in range(n_units):
    for tr in range(ntrials):
        ...
for i in range(1, len(out)):
    ...
for i in range(len(out)-2, -1, -1):
    ...
```

iii. The trajectory does not justify these loops as unavoidable; they are an implementation choice made while translating the MATLAB pipeline into Python.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work: it reopens the session file at the end only to recover the animal name, recomputes `{q.lower() for q in EXCLUDE_QUALITIES}` inside every cluster iteration, and makes separate passes over all valid trials for video extraction, threshold collection, and final output assembly.

ii.
```python
if quality.lower() in {q.lower() for q in EXCLUDE_QUALITIES}:
    continue
...
with h5py.File(data_path, 'r') as f:
    try:
        anm = h5_read_string(f, f['obj']['meta']['anm'])
...
for tr in range(ntrials):
    ...
for tr in valid_trials:
    ...
for tr in valid_trials:
    ...
```

iii. The trajectory does not explicitly justify these repeated passes; it mostly frames them as the practical way the AI chose to assemble the converted dataset.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads or computes several values that are never used downstream: `sample_times`, `delay_times`, `lickL_refs`, `lickR_refs`, `all_qualities`, `me_thresh`, and `all_animals`; it also defines helper functions `h5_deref` and `h5_read_string_from_ref` without using them. These are extra to the final converted dataset.

ii.
```python
sample_times = ev['sample'][0, :]
delay_times = ev['delay'][0, :]
lickL_refs = ev['lickL'][0, :]
lickR_refs = ev['lickR'][0, :]
...
all_qualities = []
...
me_thresh = float(me_struct['moveThresh'][0, 0].flatten()[0])
...
all_animals = []
...
def h5_deref(f, ref):
    ...
def h5_read_string_from_ref(f, ref):
    ...
```

iii. The trajectory does not provide a specific justification for these unused values; they appear to be leftovers from exploration or partially implemented ideas.
