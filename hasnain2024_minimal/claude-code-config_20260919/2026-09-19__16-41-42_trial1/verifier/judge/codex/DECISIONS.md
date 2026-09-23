# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a 25-session fixed-delay ALM session list in `SESSIONS`, loads only from `/app/data/Ephys_Behavior`, opens each `data_structure_<anm>_<date>.mat` with `h5py`, and loads motion energy from a same-folder `motionEnergy_<anm>_<date>.mat` with `scipy.io.loadmat`. It does not discover sessions across both ephys folders and does not support MATLAB v5 `data_structure` files.

ii. 
```python
DATA_DIR = '/app/data/Ephys_Behavior'

SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ...
    ('JEB19', '2023-04-21', [1]),
]

def process_session(anm, date, probes):
    path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
    with h5py.File(path, 'r') as f:
        obj = f['obj']
        ...

def load_motion_energy(anm, date, align_times, frame_times, has_video, ntrials):
    md = sio.loadmat(os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))
```

iii. In trajectory step 120, the agent justified this by saying it used the 25 fixed-delay ALM sessions from `load<ANM>_ALMVideo.m` and excluded `RandomizedDelay_Ephys_Behavior` because it viewed randomized-delay as a separate task variant whose variable delay would make a go-cue-aligned window inconsistent.

## 1-b. How are the data split into subjects?

i. Subjects are the animal IDs from the hard-coded `(anm, date, probes)` tuples. The agent accumulates unique `anm` values in first-seen order and stores each session’s `subject_idx` as the index of that animal in the running `subjects` list.

ii. 
```python
subjects = []
...
for anm, date, probes in SESSIONS:
    ...
    if anm not in subjects:
        subjects.append(anm)
    ...
    data['subject_idx'].append(subjects.index(anm))

data['subjects'] = subjects
```

iii. The trajectory does not contain a separate justification for subject splitting. The choice follows directly from the hard-coded session tuples and the final summary in step 120, which reports 25 sessions from 10 mice.

## 1-c. How are the data split into sessions?

i. One `(anm, date, probes)` tuple in `SESSIONS` is treated as one session. Each session is one `data_structure_<anm>_<date>.mat` file in `Ephys_Behavior`, processed independently by `process_session`, and then appended as one session entry in `neural`, `input`, `output`, and `brain_region_idx`.

ii. 
```python
SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ...
]

for anm, date, probes in SESSIONS:
    neural, inp, out, info = process_session(anm, date, probes)
    ...
    data['neural'].append(neural)
    data['input'].append(inp)
    data['output'].append(out)
```

iii. In trajectory step 120, the agent justified this by saying these 25 fixed-delay sessions were “the paper’s main dataset” and that randomized-delay sessions should be excluded as a separate task variant.

## 1-d. How are the data split into trials?

i. Trials are indexed by the Bpod trial count `bp.Ntrials` and the per-trial behavior vectors in `load_behavior`. The kept trial IDs are `np.flatnonzero(trial_mask)`. Neural trial assignment comes from `clu.trial`; video and motion-energy trial assignment comes from per-trial `traj` and `motionEnergy` entries indexed by the same trial number.

ii. 
```python
def load_behavior(obj):
    bp = obj['bp']
    n = int(_vec(bp, 'Ntrials')[0])
    beh = {
        'ntrials': n,
        ...
    }

trial_mask = (~beh['early']) & (~beh['stim'])
keep = np.flatnonzero(trial_mask)

trials = np.array(f[trial_refs[iclu]]).flatten().astype(int) - 1
...
for trial in range(ntrials):
    ...
```

iii. The trajectory does not state a separate trial-splitting rationale. The code comments and the final summary imply the agent treated the Bpod table and the per-trial MATLAB cell-array organization as the authoritative trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The agent filters trials only by dropping early-lick trials and photostimulation trials. It keeps ignore trials. It does not add the reference solution’s extra cut for trials that extend beyond the end of the neural recording.

ii. 
```python
# `~stim.enable & ~early`: photoinactivation trials perturb ALM
# activity, and "early lick" trials are omitted from all analyses.
# Ignore ("no") trials are *kept*, because "ignore" / "no lick" are
# required output categories for this decoding task.
trial_mask = (~beh['early']) & (~beh['stim'])
keep = np.flatnonzero(trial_mask)
```

iii. In trajectory step 120, the agent explicitly justified dropping `stim.enable` and `early` because the paper’s `params.condition` does so, while keeping ignore trials because the decoder task requires “none” / “ignore” output classes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `obj.clu` for the selected probe(s): `quality`, `trial`, and `trialtm`. Alignment uses the behavioral event `obj.bp.ev.goCue`.

ii. 
```python
clu = np.array(obj['clu'])
...
quality_refs = np.array(cl['quality']).flatten()
trial_refs = np.array(cl['trial']).flatten()
tm_refs = np.array(cl['trialtm']).flatten()
...
times = np.array(f[tm_refs[iclu]]).flatten()
times = times - align_times[trials]
```

iii. In trajectory step 120, the agent said alignment/binnig followed `alignSpikes.m` and `getSeq.m` with `obj.bp.ev.goCue`, and unit curation followed `findClusters.m` plus a firing-rate threshold.

## 2-b. How is the `neural` data processed?

i. For each kept cluster, the agent subtracts go-cue time from each spike, bins spikes into a fixed `[-2.5, 2.5)` window using 10 ms bins, converts counts to spikes/s by dividing by `DT`, and applies `my_smooth`, a causal Gaussian smoother with width `SMOOTH_N=15`. It does not use the reference solution’s 5 ms bins or `gaussian_filter1d` smoothing.

ii. 
```python
DT = 0.01
SMOOTH_N = 15

times = times - align_times[trials]

counts = np.zeros((keep_trials.size, NT))
...
bin_idx = np.floor((times[inwin] - TMIN) / DT).astype(int)
np.add.at(counts, (pos[inwin], bin_idx), 1.0)
rate = my_smooth((counts / DT).T).T
```

iii. In trajectory step 120, the agent explicitly justified 10 ms bins and causal Gaussian smoothing by saying this matched its reading of `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent drops clusters whose lower-cased quality label is in `{'garbage', 'gabrga', 'noisy', 'real?'}` and then drops units whose mean firing rate over kept trials and time bins is not greater than 1 Hz. It also skips whole sessions later if fewer than 10 units remain.

ii. 
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10

if qual.lower() in BAD_QUALITY:
    continue
...
mean_fr = rates.mean(axis=(1, 2)) if rates.size else np.zeros(0)
use = mean_fr > LOW_FR
rates = rates[use]
...
if info['nunits'] < MIN_UNITS:
    print(f"  skipping {anm} {date}: only {info['nunits']} units")
    continue
```

iii. In trajectory step 120, the agent justified this as the `findClusters` quality rule plus `params.lowFR`, and added a session-level `>=10` unit threshold based on the paper’s inclusion criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting that spike’s trial-specific go-cue time, `align_times[trials]`, where `align_times` is `bp.ev.goCue`.

ii. 
```python
ALIGN_EVENT = 'goCue'
...
'align': _vec(bp['ev'], ALIGN_EVENT),
...
times = np.array(f[tm_refs[iclu]]).flatten()
times = times - align_times[trials]
```

iii. In trajectory step 120, the agent explicitly said neural activity was aligned to `obj.bp.ev.goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 ms bins over `[-2.5, 2.5)` seconds. The code bins directly onto that grid; there is no later temporal rebinning step.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
NT = len(TAXIS)
```

iii. In trajectory step 120, the agent justified this as matching its reading of `getSeq.m` and reported a dataset with “500 bins/trial.”

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the analysis time grid centered on the go cue. The only raw behavioral variable involved is `bp.ev.goCue`, which defines the alignment event; the actual values saved are the fixed bin centers in `TAXIS`.

ii. 
```python
ALIGN_EVENT = 'goCue'
...
'align': _vec(bp['ev'], ALIGN_EVENT),
...
TAXIS = (EDGES + DT / 2)[:-1]
time_input = TAXIS.astype(np.float32).reshape(1, NT)
```

iii. In trajectory step 120, the agent explicitly said all streams were aligned to `obj.bp.ev.goCue`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The input is just the fixed 10 ms bin-center vector `TAXIS` reshaped to `(1, NT)` and copied into every kept trial. There is no trial-specific computation beyond repeating that common time axis.

ii. 
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
...
time_input = TAXIS.astype(np.float32).reshape(1, NT)
for pos_i, trial in enumerate(keep):
    ...
    input_trials.append(time_input.copy())
```

iii. The trajectory does not give a separate justification beyond the alignment/binning rationale in step 120.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same `TAXIS` grid that the neural data are binned onto. Neural spikes are histogrammed into bins defined by `TMIN`, `TMAX`, and `DT`, and the input is the corresponding bin-center vector.

ii. 
```python
bin_idx = np.floor((times[inwin] - TMIN) / DT).astype(int)
np.add.at(counts, (pos[inwin], bin_idx), 1.0)
...
TAXIS = (EDGES + DT / 2)[:-1]
time_input = TAXIS.astype(np.float32).reshape(1, NT)
```

iii. In trajectory step 120, the agent justified this by saying neural, tongue visibility, and motion-energy changes all appeared immediately after `t=0`, which it treated as an empirical alignment check.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The agent derives lick direction from the behavioral trial flags `R`, `L`, `hit`, and `miss`.

ii. 
```python
beh = {
    ...
    'R': _vec(bp, 'R') > 0.5,
    'L': _vec(bp, 'L') > 0.5,
    'hit': _vec(bp, 'hit') > 0.5,
    'miss': _vec(bp, 'miss') > 0.5,
    ...
}
...
right = (beh['R'] & beh['hit']) | (beh['L'] & beh['miss'])
left = (beh['L'] & beh['hit']) | (beh['R'] & beh['miss'])
```

iii. In trajectory step 120, the agent justified this as matching `funcs/getPrevChoice.m`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It codes rightward licks as `(R & hit) | (L & miss)`, leftward licks as `(L & hit) | (R & miss)`, and assigns class `2` (“none”) to all remaining trials.

ii. 
```python
right = (beh['R'] & beh['hit']) | (beh['L'] & beh['miss'])
left = (beh['L'] & beh['hit']) | (beh['R'] & beh['miss'])
lick_dir = np.full(n, 2, dtype=np.int8)      # 2 = none
lick_dir[left] = 0
lick_dir[right] = 1
```

iii. In trajectory step 120, the agent explicitly described this rule and said ignore trials were kept so that “none” would remain a required output class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp.autowater` flag.

ii. 
```python
'autowater': _vec(bp, 'autowater') > 0.5,
...
context = np.where(beh['autowater'], 0, 1).astype(np.int8)
```

iii. In trajectory step 120, the agent justified this directly: “context from `obj.bp.autowater`.”

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It is a direct relabeling: `autowater=True` becomes WC (`0`), otherwise DR (`1`).

ii. 
```python
context = np.where(beh['autowater'], 0, 1).astype(np.int8)
```

iii. The trajectory offers no extra justification beyond the direct statement in step 120 that context comes from `obj.bp.autowater`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The code loads `hit`, `miss`, and `no`, but the actual outcome labels are derived from `hit` and `miss`, with all other trials defaulting to ignore.

ii. 
```python
'hit': _vec(bp, 'hit') > 0.5,
'miss': _vec(bp, 'miss') > 0.5,
'no': _vec(bp, 'no') > 0.5,
...
outcome = np.full(n, 2, dtype=np.int8)
outcome[beh['miss']] = 0
outcome[beh['hit']] = 1
```

iii. In trajectory step 120, the agent summarized this as “outcome from hit/miss/no,” but the implementation effectively uses hit and miss and treats everything else as ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is a three-class relabeling: miss → incorrect (`0`), hit → correct (`1`), otherwise ignore (`2`).

ii. 
```python
outcome = np.full(n, 2, dtype=np.int8)       # 2 = ignore
outcome[beh['miss']] = 0
outcome[beh['hit']] = 1
```

iii. In trajectory step 120, the agent explicitly said “outcome from hit/miss/no” and that ignore trials were retained.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived only from the side-camera DeepLabCut feature `tongue` in `obj.traj`, plus each trial’s `frameTimes`. The code also uses `bp.ev.goCue` and the video/behavior bitcode timing fields in `obj.sglx` and `bp.ev.bitStart` to align those frames to go cue.

ii. 
```python
TONGUE_FEATURES = [(0, 'tongue')]
...
def video_shift(f, obj):
    bit_start = _mode(_vec(obj['bp']['ev'], 'bitStart'))
    fs = _vec(obj['sglx'], 'fs')[0]
    vid_file_offset = _mode(_vec(obj['sglx']['bitcode'], 'bitstart')) / fs
    return vid_file_offset - bit_start
...
pos, has_video, frame_times = load_video_traces(f, obj, align, n, vidshift)
...
x, y = pos[TONGUE_FEATURES[0]][0][trial], pos[TONGUE_FEATURES[0]][1][trial]
```

iii. In trajectory step 120, the agent justified this as following `findPosition.m` / `findVelocity.m` and specifically chose the side-camera `tongue` feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The agent first interpolates tongue x/y positions onto the common 10 ms analysis grid, then computes speed as the Euclidean norm of `np.gradient` on contiguous visible stretches. It does not apply an explicit likelihood threshold, does not smooth x/y before differentiating, does not compute frame-time gradients, and does not combine the bottom-camera tongue track.

ii. 
```python
pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
pos[(view, feat)][1][trial] = interp_nan(TAXIS, t_rel, ts[fi, 1])
...
visible = ~np.isnan(x)
...
idx = np.flatnonzero(visible)
if idx.size:
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for seg in splits:
        if seg.size == 1:
            vx[seg] = 0.0
            vy[seg] = 0.0
        else:
            vx[seg] = np.gradient(x[seg])
            vy[seg] = np.gradient(y[seg])
return np.sqrt(vx ** 2 + vy ** 2), visible
```

iii. In trajectory step 120, the agent justified this by saying it followed `findPosition.m` / `findVelocity.m` and that side-camera tongue non-detection should map to the “not visible” class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After computing all tongue-speed time points for a session, the agent takes the 50th percentile over visible time points from kept trials, then encodes each time point as `0` if below threshold, `1` if at/above threshold, and `2` if not visible.

ii. 
```python
def median_of(values, visible):
    vals = values[keep][visible[keep]]
    vals = vals[~np.isnan(vals)]
    return np.median(vals) if vals.size else np.inf

thr_tongue = median_of(tongue_speed, tongue_vis)
...
def discretize(values, visible, threshold):
    out = np.full(values.shape, 2, dtype=np.int8)
    ok = visible & ~np.isnan(values)
    out[ok] = (values[ok] >= threshold).astype(np.int8)
    return out
```

iii. In trajectory step 120, the agent explicitly said each movement variable was “split at its own session’s 50th percentile over visible timepoints.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The agent computes a session-wide video offset from `sglx.bitcode.bitstart / fs - bp.ev.bitStart`, subtracts that offset and the trial’s go cue from each frame time, and interpolates positions onto the same `TAXIS` grid used for the neural data.

ii. 
```python
def video_shift(f, obj):
    bit_start = _mode(_vec(obj['bp']['ev'], 'bitStart'))
    fs = _vec(obj['sglx'], 'fs')[0]
    vid_file_offset = _mode(_vec(obj['sglx']['bitcode'], 'bitstart')) / fs
    return vid_file_offset - bit_start
...
t_rel = ft - vidshift - align_times[trial]
...
pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
```

iii. In trajectory step 120, the agent said alignment was checked empirically because tongue visibility jumped immediately after `t=0`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera DeepLabCut features, `top_paw` and `bottom_paw`, plus their per-trial `frameTimes`. As with tongue velocity, frame alignment also uses `bp.ev.goCue`, `bp.ev.bitStart`, and `sglx.bitcode.bitstart`.

ii. 
```python
PAW_FEATURES = [(1, 'top_paw'), (1, 'bottom_paw')]
...
for i, key in enumerate(PAW_FEATURES):
    for trial in range(n):
        s, v = speed_from_position(pos[key][0][trial], pos[key][1][trial], fill=True)
```

iii. In trajectory step 120, the agent justified this as averaging `top_paw` and `bottom_paw` over whichever paw was detected.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The agent interpolates both paw tracks onto the common 10 ms grid, fills missing positions by nearest neighbor, computes gradients of x and y, subtracts the per-trial median frame-to-frame drift, converts each paw to speed, and averages the two paw speeds where at least one paw is visible.

ii. 
```python
if fill:
    xf, yf = fill_nearest(x), fill_nearest(y)
    if np.all(np.isnan(xf)):
        return np.full(NT, np.nan), visible
    vx, vy = np.gradient(xf), np.gradient(yf)
    vx = vx - np.nanmedian(np.diff(xf))
    vy = vy - np.nanmedian(np.diff(yf))
...
masked = np.where(paw_vis_all, paw_speed_all, np.nan)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    paw_speed = np.nanmean(masked, axis=0)
```

iii. In trajectory step 120, the agent justified this as following `findPosition.m` / `findVelocity.m` and said the two paw features should be averaged over whichever one was detected.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the session median over visible paw-speed time points from kept trials. Values below median are `0`, values at or above median are `1`, and invisible values are `2`.

ii. 
```python
thr_paw = median_of(paw_speed, paw_vis)
...
paw_cls = discretize(paw_speed, paw_vis, thr_paw)
```

iii. In trajectory step 120, the agent explicitly said each movement variable was split at its own session 50th percentile over visible time points.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The paw tracks use the same session-wide video offset and per-trial go-cue subtraction as the tongue tracks, and the paw positions are interpolated onto the same `TAXIS` grid used for neural binning.

ii. 
```python
t_rel = ft - vidshift - align_times[trial]
...
pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
pos[(view, feat)][1][trial] = interp_nan(TAXIS, t_rel, ts[fi, 1])
```

iii. The trajectory does not provide a paw-specific alignment justification beyond the general alignment/binning rationale in step 120.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate MATLAB file `motionEnergy_<anm>_<date>.mat`, specifically `me.data`, with timing taken from the side-camera `frameTimes` already aligned through `load_video_traces`.

ii. 
```python
def load_motion_energy(anm, date, align_times, frame_times, has_video, ntrials):
    md = sio.loadmat(os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))
    data = md['me'][0, 0]['data']
    if data.dtype != object:
        data = data[0, 0]['data']
    data = data.flatten()
```

iii. In trajectory step 120, the agent justified this as following `DataLoadingScripts/loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The agent interpolates each trial’s motion-energy trace onto the common `TAXIS` grid using the side-camera frame times, fills missing bins by nearest neighbor, and then later thresholds the result at the session median. It skips trials whose motion-energy length does not match the frame-time length.

ii. 
```python
for trial in range(ntrials):
    if not has_video[trial]:
        continue
    y = np.asarray(data[trial]).flatten().astype(float)
    t_rel = frame_times[trial]
    if y.size != t_rel.size:
        continue
    me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
```

iii. The trajectory does not contain a longer motion-energy-specific justification beyond step 120’s statement that motion energy followed `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The threshold is the session median over kept-trial motion-energy values that have video and are not NaN. Values below threshold are `0`, values at or above threshold are `1`, and bins with no usable value become `2`.

ii. 
```python
me_vis = has_video[0][:, None] & ~np.isnan(me)
thr_me = median_of(me, me_vis)
me_cls = discretize(me, me_vis, thr_me)
```

iii. In trajectory step 120, the agent explicitly said motion energy, like the other movement variables, was split at the session 50th percentile over visible time points.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using side-camera frame times corrected by the session video offset and the trial’s go cue, then interpolated to the same `TAXIS` grid used by the neural data.

ii. 
```python
t_rel = ft - vidshift - align_times[trial]
...
me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
```

iii. In trajectory step 120, the agent grouped motion energy with the other go-cue-aligned, common-grid outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable video trials are generally kept but marked invisible: `load_video_traces` skips empty or all-NaN `frameTimes`, and `discretize` maps non-visible bins to class `2`. For paw position and motion energy, the agent fills missing bins with nearest-neighbor values before computing speed or storing motion energy. For tongue position, it leaves gaps as NaN and computes gradients only on contiguous visible stretches. If motion-energy length mismatches frame-time length, that trial’s motion energy remains NaN.

ii. 
```python
if ft.size == 0 or np.all(np.isnan(ft)):
    continue
...
def fill_nearest(x):
    ...
    return x[take]
...
if fill:
    xf, yf = fill_nearest(x), fill_nearest(y)
...
if y.size != t_rel.size:
    continue
...
out = np.full(values.shape, 2, dtype=np.int8)
```

iii. In trajectory step 120, the agent explicitly justified tongue non-detection as “not visible.” The rest is only indirectly justified by code comments saying it was porting `findPosition.m`, `findVelocity.m`, and `loadMotionEnergy.m`.

## 11-a. What are the most time-consuming steps of the code?

i. The code’s obvious heavy steps are repeated session file I/O, per-cluster spike binning/smoothing in `load_spikes`, and the per-trial video interpolation / speed computation loops in `load_video_traces` and `speed_from_position`.

ii. 
```python
with h5py.File(path, 'r') as f:
    ...
for prb in probes:
    ...
    for iclu in range(quality_refs.size):
        ...
for trial in range(ntrials):
    for view, (wanted, ft_refs, ts_refs) in view_data.items():
        ...
for i, key in enumerate(PAW_FEATURES):
    for trial in range(n):
        ...
```

iii. The trajectory does not contain a direct performance analysis, but steps 94, 96, and 118 show the agent benchmarking a sample session and full conversion run, implying these are the expected hot paths.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over clusters in `load_spikes`, over trials and camera views in `load_video_traces`, over trials for tongue speed, over paw features and trials for paw speed, and over trials again when assembling outputs. Some of these are partly constrained by ragged per-trial frame counts, but the implementation is still loop-heavy.

ii. 
```python
for prb in probes:
    ...
    for iclu in range(quality_refs.size):
        ...

for trial in range(ntrials):
    for view, (wanted, ft_refs, ts_refs) in view_data.items():
        ...

for trial in range(n):
    ...

for i, key in enumerate(PAW_FEATURES):
    for trial in range(n):
        ...
```

iii. No explicit justification appears in the trajectory. The code comments imply the agent accepted these loops as a straightforward way to port the MATLAB logic.

## 11-c. What processing does the code repeat multiple times?

i. The code makes several separate passes over the same per-trial movement arrays: it first interpolates positions, then loops again to compute speeds, then loops again to compute thresholds, and finally loops again to discretize and assemble outputs. It also computes both paw feature velocities before averaging them.

ii. 
```python
pos, has_video, frame_times = load_video_traces(...)
...
for trial in range(n):
    s, v = speed_from_position(...)
...
for i, key in enumerate(PAW_FEATURES):
    for trial in range(n):
        s, v = speed_from_position(...)
...
thr_tongue = median_of(tongue_speed, tongue_vis)
thr_paw = median_of(paw_speed, paw_vis)
thr_me = median_of(me, me_vis)
...
tongue_cls = discretize(tongue_speed, tongue_vis, thr_tongue)
```

iii. The trajectory does not defend this explicitly. The repeated passes appear to be a byproduct of structuring the code as separate “load”, “compute”, “threshold”, and “assemble” stages.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Relative to the final decoder arrays, the code does extra work by loading both `L` and `no` even though derived labels could be formed without both; computing `qualities`, `nsingle_units`, and `quality_counts` only for metadata; computing both `top_paw` and `bottom_paw` velocities before collapsing them into one paw output; and interpolating continuous positions / motion-energy traces even though only final discrete classes are saved.

ii. 
```python
'L': _vec(bp, 'L') > 0.5,
'no': _vec(bp, 'no') > 0.5,
...
qualities.append(qual)
...
'nsingle_units': int(sum(q.lower() in ('excellent', 'great', 'good')
                         for q in qualities)),
'quality_counts': dict(Counter(q for q in qualities)),
...
PAW_FEATURES = [(1, 'top_paw'), (1, 'bottom_paw')]
...
pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
```

iii. The trajectory does not acknowledge these as unnecessary. They follow from the agent’s choice to preserve extra session metadata and to average two paw tracks before discretization.
