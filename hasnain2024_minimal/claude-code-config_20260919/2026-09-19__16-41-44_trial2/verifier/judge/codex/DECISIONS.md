# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the full dataset used by the human reference. It hard-codes 25 fixed-delay `Ephys_Behavior` ALM sessions in `SESSIONS`, opens each `data_structure_<anm>_<date>.mat` directly with `h5py`, and separately loads `motionEnergy_<anm>_<date>.mat` from the same folder. It does not search both ephys folders, does not include randomized-delay sessions, and does not implement the v5/v7.3 dual-loader used by the reference.

ii. <Code snippets>
```python
DATA_DIR = '/app/data/Ephys_Behavior'

SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ...
    ('JGR3',  '2021-11-18', [1]),
]
```
```python
path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
f = h5py.File(path, 'r')
obj = f['obj']
```
```python
fn = os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat')
if not os.path.exists(fn):
    return np.full((len(trials), NBINS), np.nan)
```

iii. The trajectory's final summary says the AI intentionally used "the 25 fixed-delay ALM ephys+video sessions in `Ephys_Behavior`" and explicitly excluded the randomized-delay dataset because its variable delays would mix task epochs in the pre-go-cue window and those mice never experienced the WC context. It also says the per-session probe ids were transcribed from `load<ANM>_ALMVideo.m`.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats the animal id (`anm`) from each hard-coded session tuple as the subject id. After converting sessions, it builds `subjects` as the sorted unique animal names and `subject_idx` by looking up each session's animal name in that list.

ii. <Code snippets>
```python
return {
    'neural': neural,
    'input': inputs,
    'output': outputs,
    'nunits': rates.shape[2],
    'session_info': {
        'subject': anm,
        'date': date,
```
```python
subjects = sorted({s['session_info']['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['session_info']['subject'])
                        for s in sessions], dtype=np.int64)
```

iii. The trajectory does not add a separate justification beyond the final summary's statement that the output contains 25 sessions from 10 mice. The subject split is implicit in the hard-coded `(anm, date, probes)` session tuples.

## 1-c. How are the data split into sessions?

i. One hard-coded `(anm, date, probes)` tuple in `SESSIONS` is one session. `main()` loops over that list, `convert_session()` loads one `data_structure` file for that tuple, and each returned session becomes one element in `data['neural']`, `data['input']`, and `data['output']`.

ii. <Code snippets>
```python
SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ...
]
```
```python
for anm, date, probes in SESSIONS:
    s = convert_session(anm, date, probes)
    if s is None:
        print(f'skipping {anm}_{date}')
        continue
    sessions.append(s)
```

iii. The trajectory's final summary says these are the "exact per-session ALM probe ids transcribed from `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`" and that the randomized-delay sessions were intentionally excluded.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the per-trial Bpod arrays loaded by `load_behavior()`. `convert_session()` builds a Boolean keep mask over those trial-wise arrays, converts kept trial indices to 1-based MATLAB trial ids, and then uses those trial ids everywhere else for neural, video, and per-trial outputs.

ii. <Code snippets>
```python
def load_behavior(f, obj):
    bp = obj['bp']
    ev = bp['ev']
    n = int(vec(bp, 'Ntrials')[0])
    b = {
        'R': np.nan_to_num(vec(bp, 'R')[:n]) > 0,
        'hit': np.nan_to_num(vec(bp, 'hit')[:n]) > 0,
        ...
        'goCue': vec(ev, ALIGN_EVENT)[:n],
    }
```
```python
keep = (~b['early']) & (~b['stim']) & np.isfinite(b['goCue'])
trials = np.where(keep)[0] + 1                  # 1-based MATLAB trial ids
```

iii. The trajectory's final summary says the AI kept hit, miss, and ignore trials, but dropped early-lick and photostimulation trials. It also says that after this filtering the trial structure was regular enough that no time warping was needed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by excluding early-lick trials, photostimulation trials, and trials with non-finite go-cue times. The AI does not implement the reference solution's extra removal of late behavioral trials that continue after the recording has stopped.

ii. <Code snippets>
```python
keep = (~b['early']) & (~b['stim']) & np.isfinite(b['goCue'])
trials = np.where(keep)[0] + 1
if len(trials) < 2:
    f.close()
    return None
```

iii. The trajectory's final summary explicitly justifies dropping `~early & ~stim.enable` because that is "the mask behind every condition in the paper's scripts," and says hit/miss/ignore trials were kept because outcome and lick direction are decoder targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from each kept cluster's spike times and trial assignments in `obj.clu` plus the per-trial go-cue times in `obj.bp.ev.goCue`. Cluster quality labels are also read for curation.

ii. <Code snippets>
```python
qualities = [mat_str(f, r) for r in np.array(g['quality']).flatten()]
...
tm = np.array(f[g['trialtm'][icell, 0]]).flatten()
tr = np.array(f[g['trial'][icell, 0]]).flatten().astype(np.int64)
...
aligned = tm - align[tr - 1]
```

iii. The script docstring and the final trajectory summary say neural alignment follows `alignSpikes.m` with `params.alignEvent = 'goCue'`, and neural processing follows `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the AI aligns spikes to the go cue, bins them into 10 ms bins from `-2.5` to `+2.5` s, converts counts to spikes/s, and smooths them with a causal 15-bin Gaussian implemented by `my_smooth()`. It then stores one `(n_units, n_bins)` matrix per trial.

ii. <Code snippets>
```python
DT = 1.0 / 100.0     # params.dt  (10 ms bins)
SMOOTH_N = 15        # params.smooth (bins of the causal Gaussian kernel)
```
```python
b = ((aligned[ok] - TMIN) / DT).astype(np.int64)
counts = np.bincount(row * NBINS + b,
                     minlength=len(trials) * NBINS).reshape(len(trials), NBINS)
```
```python
rates = counts.astype(np.float64) / DT
flat = rates.transpose(1, 0, 2).reshape(NBINS, -1)
flat = my_smooth(flat)
rates = flat.reshape(NBINS, len(trials), -1).transpose(1, 0, 2)
```

iii. The trajectory's final summary says the AI "ported `getSeq.m`/`mySmooth.m` exactly" as "10 ms spike counts -> spikes/s -> causal 15-bin Gaussian with the `'reflect'` boundary."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first drops clusters whose lower-cased quality label is in `{'garbage', 'gabrga', 'noisy', 'real?'}`. It then computes an all-trials PSTH for each remaining unit and keeps the unit only if the mean of that smoothed PSTH exceeds 1 Hz. It does not exclude `poor` clusters.

ii. <Code snippets>
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
```
```python
if q.lower() in BAD_QUALITY:
    continue
...
psth = my_smooth((counts.sum(axis=0) / len(trials) / DT)[:, None])[:, 0]
if psth.mean() > LOW_FR:
    keep_rates.append(counts)
```

iii. The trajectory's final summary says the AI used the "`findClusters.m` quality rule" and then "`removeLowFRClusters.m` (>1 Hz on the all-trials PSTH)." The docstring also states it was following those MATLAB functions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is converted from trial-relative time to time-from-go-cue by subtracting that trial's `goCue` timestamp. The aligned spike times are then binned on the common neural time axis.

ii. <Code snippets>
```python
aligned = tm - align[tr - 1]
ok = (row >= 0) & (aligned >= TMIN) & (aligned < TMAX)
```

iii. The docstring cites `alignSpikes.m`, and the trajectory's final summary says alignment is to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`DT = 1/100`) over a `-2.5` to `+2.5` s window, giving 500 bins per trial. There is no second-stage temporal rebinning after spike counting; the only later temporal processing is Gaussian smoothing on that same grid.

ii. <Code snippets>
```python
DT = 1.0 / 100.0
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
NBINS = len(TAXIS)
```

iii. The trajectory's final summary explicitly reports "500 bins/trial (10 ms) from -2.5 s to +2.5 s around the go cue" and says this was meant to mirror `getSeq.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read as a raw data field. It is constructed from the analysis time axis defined around go-cue alignment, using the same `TMIN`, `TMAX`, and `DT` that define the neural grid. The raw go-cue variable enters only indirectly by defining the alignment convention for that axis.

ii. <Code snippets>
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100.0
TAXIS = EDGES[:-1] + DT / 2
```
```python
time_input = TAXIS.astype(np.float32)[None, :]
```

iii. The trajectory does not give a separate justification beyond repeatedly saying all streams were aligned to the go cue and shared one common time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin centers on the common time axis and then copies that one-row time vector into every trial as the only decoder input.

ii. <Code snippets>
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
...
time_input = TAXIS.astype(np.float32)[None, :]
...
inputs.append(time_input.copy())
```

iii. The trajectory's final summary says "The decoder receives ALM population activity plus the time from the go cue," but gives no further processing rationale because this variable is defined by the target format rather than directly by the source data.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector is exactly the same `TAXIS` grid used to bin neural spikes and to interpolate video-derived signals, so its columns are aligned one-to-one with neural time bins.

ii. <Code snippets>
```python
TAXIS = EDGES[:-1] + DT / 2
...
neural.append(np.ascontiguousarray(rates[i].T))
inputs.append(time_input.copy())
```

iii. The trajectory's final summary says the video streams were also put "onto the same time axis," implying that the time input is the common grid shared by neural and behavioral data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial Bpod variables `R`, `L`, `hit`, and `miss`.

ii. <Code snippets>
```python
'R': np.nan_to_num(vec(bp, 'R')[:n]) > 0,
'L': np.nan_to_num(vec(bp, 'L')[:n]) > 0,
'hit': np.nan_to_num(vec(bp, 'hit')[:n]) > 0,
'miss': np.nan_to_num(vec(bp, 'miss')[:n]) > 0,
```
```python
licked_right = (b['R'][idx] & b['hit'][idx]) | (b['L'][idx] & b['miss'][idx])
licked_left = (b['L'][idx] & b['hit'][idx]) | (b['R'][idx] & b['miss'][idx])
```

iii. The trajectory's final summary justifies keeping hit, miss, and ignore trials specifically because lick direction and outcome are decoder targets.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI assigns class `2` ("none") by default, marks right licks when the animal either hit on an `R` trial or missed on an `L` trial, and marks left licks for the converse combinations. The label is then repeated across all time bins in the per-trial output array.

ii. <Code snippets>
```python
lick = np.full(len(idx), 2, dtype=np.int8)                     # 2 = none
licked_right = (b['R'][idx] & b['hit'][idx]) | (b['L'][idx] & b['miss'][idx])
licked_left = (b['L'][idx] & b['hit'][idx]) | (b['R'][idx] & b['miss'][idx])
lick[licked_left] = 0
lick[licked_right] = 1
```
```python
out[0] = lick[i]
```

iii. The trajectory does not provide additional justification beyond the final summary's statement that hit/miss/ignore trials were retained so lick direction could be decoded.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag in the Bpod data.

ii. <Code snippets>
```python
'autowater': np.nan_to_num(vec(bp, 'autowater')[:n]) > 0,
...
context = np.where(b['autowater'][idx], 0, 1).astype(np.int8)
```

iii. The trajectory's final summary distinguishes WC and DR sessions and states that the randomized-delay mice never experienced WC, which was part of the AI's reason for excluding them.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly relabels `autowater=True` as `0` (`WC`) and everything else as `1` (`DR`), then repeats that per-trial label across all bins.

ii. <Code snippets>
```python
context = np.where(b['autowater'][idx], 0, 1).astype(np.int8)  # 0 = WC, 1 = DR
...
out[1] = context[i]
```

iii. The trajectory's final summary says the fixed-delay sessions comprised "12 two-context sessions + 13 DR-only," which is the main contextual justification the AI recorded.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial `hit` and `miss` flags in the Bpod data.

ii. <Code snippets>
```python
'hit': np.nan_to_num(vec(bp, 'hit')[:n]) > 0,
'miss': np.nan_to_num(vec(bp, 'miss')[:n]) > 0,
...
outcome[b['miss'][idx]] = 0
outcome[b['hit'][idx]] = 1
```

iii. The trajectory's final summary says hit, miss, and ignore trials were all retained because outcome is one of the decoder targets.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI assigns `2` ("ignore") by default, then overwrites miss trials with `0` ("incorrect") and hit trials with `1` ("correct"). That label is repeated across all bins for each trial.

ii. <Code snippets>
```python
outcome = np.full(len(idx), 2, dtype=np.int8)                  # 2 = ignore
outcome[b['miss'][idx]] = 0                                    # 0 = incorrect
outcome[b['hit'][idx]] = 1                                     # 1 = correct
```
```python
out[2] = outcome[i]
```

iii. The trajectory's final summary explicitly says hit/miss/ignore trials were retained because outcome was being decoded.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-view DeepLabCut tongue marker only: `TONGUE_FEATURE = ('tongue', 1)`. The script also uses side-camera `frameTimes`, session video offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart`, and per-trial `goCue` times to align that marker onto the analysis time axis.

ii. <Code snippets>
```python
TONGUE_FEATURE = ('tongue', 1)
```
```python
wanted = [TONGUE_FEATURE] + PAW_FEATURES
...
t_src = ft - vidshift - align[j]
...
pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```
```python
tspeed, tvis = tongue_speed(pos[TONGUE_FEATURE[0]])
```

iii. The trajectory's final summary says the AI chose "Tongue speed from the side-view tongue marker with the paper's lick-onset baseline fill."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI first linearly interpolates side-view tongue x/y coordinates onto the common `TAXIS` grid. It defines visibility by finite coordinates, finds the first sample of each visible run, fills missing tongue coordinates with the session mean of those run-start positions, and then computes speed as the Euclidean norm of `np.gradient` on the filled x and y traces. It does not compute velocity from raw frame times run-by-run the way the reference solution does.

ii. <Code snippets>
```python
def tongue_speed(pos):
    visible = np.isfinite(pos[..., 0]) | np.isfinite(pos[..., 1])
    starts_x, starts_y = [], []
    for i in range(pos.shape[0]):
        for s in runs_of_true(visible[i]):
            starts_x.append(pos[i, s, 0])
            starts_y.append(pos[i, s, 1])
    mux = np.nanmean(starts_x) if starts_x else 0.0
    muy = np.nanmean(starts_y) if starts_y else 0.0

    filled = pos.copy()
    filled[..., 0] = np.where(np.isfinite(filled[..., 0]), filled[..., 0], mux)
    filled[..., 1] = np.where(np.isfinite(filled[..., 1]), filled[..., 1], muy)

    vx = np.gradient(filled[..., 0], axis=1)
    vy = np.gradient(filled[..., 1], axis=1)
    return np.sqrt(vx ** 2 + vy ** 2), visible
```

iii. The trajectory's final summary justifies this as using "the paper's lick-onset baseline fill" on the side-view tongue marker. No additional explicit rationale is given for using only the side view.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After computing the continuous tongue-speed array, the AI discretizes all valid session samples at the 50th percentile: `<50th percentile -> 0`, `>=50th percentile -> 1`, and invalid or missing samples -> `2`.

ii. <Code snippets>
```python
def discretize(values, valid, missing_code=2):
    out = np.full(values.shape, missing_code, dtype=np.int8)
    v = values[valid]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return out
    thresh = np.percentile(v, 50)
    finite = valid & np.isfinite(values)
    out[finite & (values < thresh)] = 0
    out[finite & (values >= thresh)] = 1
    return out
```
```python
tongue_code = discretize(tspeed, tvis)
```

iii. The trajectory's final summary says the decoder target used "the time-varying, per-session-median thresholded movement variables," which is the recorded justification for the 50th-percentile split.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI aligns the tongue coordinates by taking raw side-camera frame times, subtracting the session video offset and the trial's go cue, and then linearly interpolating the x/y traces onto the same `TAXIS` grid used for neural data. The visibility mask is additionally intersected with `has_video`.

ii. <Code snippets>
```python
vidshift = video_offset(f, obj)
pos, has_video = load_traj(f, obj, trials, align, vidshift)
```
```python
t_src = ft - vidshift - align[j]
...
pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```
```python
tvis &= has_video[:, None]
```

iii. The trajectory's final summary says video alignment used `findVideoOffset.m` and MATLAB-style `interp1` onto the same time axis as the neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-view DeepLabCut paw markers, `top_paw` and `bottom_paw`, both specified in `PAW_FEATURES`. As with tongue velocity, the script also uses camera `frameTimes`, bitcode-based video offset, and go-cue times for alignment.

ii. <Code snippets>
```python
PAW_FEATURES = [('top_paw', 2), ('bottom_paw', 2)]
```
```python
wanted = [TONGUE_FEATURE] + PAW_FEATURES
...
pspeed, pvis = paw_speed([pos[n] for n, _ in PAW_FEATURES])
```

iii. The trajectory's final summary explicitly says the AI used "the mean of the two bottom-view paw markers."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates both paw-marker x/y traces onto `TAXIS`, fills missing positions within each trial by nearest-neighbor propagation, removes each trial's median frame-to-frame displacement from the x and y gradients, computes speed for each marker, and then averages the two marker speeds wherever either is visible.

ii. <Code snippets>
```python
def paw_speed(pos_list):
    speeds, visibles = [], []
    for pos in pos_list:
        visible = np.isfinite(pos[..., 0]) | np.isfinite(pos[..., 1])
        sp = np.zeros(pos.shape[:2])
        for i in range(pos.shape[0]):
            if not visible[i].any():
                continue
            x = fill_nearest(pos[i, :, 0])
            y = fill_nearest(pos[i, :, 1])
            vx = np.gradient(x) - np.nanmedian(np.diff(x))
            vy = np.gradient(y) - np.nanmedian(np.diff(y))
            sp[i] = np.sqrt(vx ** 2 + vy ** 2)
        speeds.append(sp)
        visibles.append(visible)
    ...
    mean_speed = np.nansum(np.where(visibles, speeds, np.nan), axis=0) / \
                 np.maximum(visibles.sum(axis=0), 1)
    return mean_speed, any_visible
```

iii. The trajectory's final summary justifies this by saying paw speed was taken as "the mean of the two bottom-view paw markers with `'nearest'` fill and median-drift removal."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is discretized with the same session-wide 50th-percentile split used for tongue velocity: `0` below threshold, `1` at or above threshold, and `2` when the quantity is not measurable.

ii. <Code snippets>
```python
paw_code = discretize(pspeed, pvis)
```
```python
thresh = np.percentile(v, 50)
...
out[finite & (values < thresh)] = 0
out[finite & (values >= thresh)] = 1
```

iii. The trajectory's final summary groups paw velocity with the other "per-session-median thresholded movement variables."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw positions are aligned the same way as tongue positions: the script subtracts the session video offset and per-trial go cue from the raw frame times, interpolates paw coordinates onto `TAXIS`, and then computes paw speed on that neural-aligned grid.

ii. <Code snippets>
```python
t_src = ft - vidshift - align[j]
...
pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```
```python
pspeed, pvis = paw_speed([pos[n] for n, _ in PAW_FEATURES])
pvis &= has_video[:, None]
```

iii. The trajectory's final summary says all video-derived signals used the `findVideoOffset.m` clock correction and were placed on the same time axis as neural activity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the external `motionEnergy_<anm>_<date>.mat` file for each retained fixed-delay session, plus side-camera frame times or a synthetic fallback frame-time axis when the frame-time array is missing or mismatched.

ii. <Code snippets>
```python
fn = os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat')
if not os.path.exists(fn):
    return np.full((len(trials), NBINS), np.nan)
```
```python
m = scipy.io.loadmat(fn, struct_as_record=False, squeeze_me=False)
me = m['me'][0, 0]
data = me.data
while hasattr(data, '_fieldnames') or (data.dtype == object and data.size == 1
                                       and hasattr(data.flatten()[0], '_fieldnames')):
    data = data.data if hasattr(data, '_fieldnames') else data.flatten()[0].data
```

iii. The trajectory's final summary says motion energy came from `motionEnergy_*.mat` "as in `loadMotionEnergy.m`."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI unwraps nested MATLAB structs until it reaches the per-trial motion-energy arrays, aligns each trial either with real frame times or a fallback synthetic 400 Hz frame clock with a 0.5 s offset, linearly interpolates the motion-energy trace onto `TAXIS`, and fills missing bins by nearest-neighbor propagation.

ii. <Code snippets>
```python
if ft.size != y.size or not np.isfinite(ft).any():
    # loadMotionEnergy.m's catch branch: assume 400 Hz frames and a 0.5 s offset
    t_src = np.arange(1, y.size + 1) / 400.0 - 0.5 - align[j]
else:
    t_src = ft - vidshift - align[j]
out[i] = interp_to_taxis(t_src, y, TAXIS)
out[i] = fill_nearest(out[i])
```

iii. The trajectory's final summary cites `loadMotionEnergy.m` as the model. The docstring and comments also frame the fallback branch as a port of that MATLAB code.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized session-wise with the same median split as the other movement variables. Finite values below the median are class `0`, values at or above it are class `1`, and non-finite bins are class `2` ("no video").

ii. <Code snippets>
```python
me_code = discretize(me, np.isfinite(me))
```
```python
'output_values': [
    ...
    ['below_50th_pctile', 'at_or_above_50th_pctile', 'no_video'],
],
```

iii. The trajectory's final summary says the movement variables were thresholded at the session median and notes that motion-energy class `2` did not actually occur in the retained data, but was kept because the task spec required that coding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by converting each camera frame time to time-from-go-cue using the session video offset and per-trial go cue, then linearly interpolating that trace to the same `TAXIS` grid as the neural data. When frame times are unusable, a synthetic frame-time axis is substituted before interpolation.

ii. <Code snippets>
```python
if ft.size != y.size or not np.isfinite(ft).any():
    t_src = np.arange(1, y.size + 1) / 400.0 - 0.5 - align[j]
else:
    t_src = ft - vidshift - align[j]
out[i] = interp_to_taxis(t_src, y, TAXIS)
```

iii. The trajectory's final summary says motion energy was aligned using the same `findVideoOffset.m` correction and common time axis as the other video streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several error cases by skipping or imputing rather than dropping whole sessions. Trials with fewer than 10 side-camera frames or no finite frame times are marked as having no usable video; trials with `NdroppedFrames = NaN` are skipped for video loading; feature/frame-length mismatches are truncated to the shorter length before interpolation; missing or mismatched motion-energy frame times trigger a synthetic 400 Hz time base with a fixed 0.5 s offset; and missing interpolated bins in motion energy or paw coordinates are filled by nearest-neighbor propagation. Missing tongue positions are filled with the session mean run-start position before differentiation.

ii. <Code snippets>
```python
if ft.size < 10 or not np.isfinite(ft).any():
    continue
...
if nd.size and np.isnan(nd[0]):
    continue
```
```python
if xy.shape[0] != t_src.shape[0]:
    m = min(xy.shape[0], t_src.shape[0])
    pos[name][i] = interp_to_taxis(t_src[:m], xy[:m], TAXIS)
```
```python
if ft.size != y.size or not np.isfinite(ft).any():
    t_src = np.arange(1, y.size + 1) / 400.0 - 0.5 - align[j]
...
out[i] = fill_nearest(out[i])
```

iii. The trajectory's final summary says class `2` is used for "feature untracked at that timepoint / no video" and that the video pipeline followed `findVideoOffset.m` and related MATLAB functions. It does not separately justify the nearest-fill and synthetic-timebase choices beyond the code comments claiming fidelity to the paper code.

## 11-a. What are the most time-consuming steps of the code?

i. The code is likely dominated by repeated per-session HDF5 loading, per-unit spike binning across all kept cells, and per-trial interpolation of tracking and motion-energy signals. The main heavy loops are the session loop in `main()`, the probe/cell loop in `neural_matrix()`, and the per-trial loop in `load_traj()` and `load_motion_energy()`.

ii. <Code snippets>
```python
for anm, date, probes in SESSIONS:
    s = convert_session(anm, date, probes)
```
```python
for p in probes:
    ...
    for icell, q in enumerate(qualities):
        ...
        counts = np.bincount(...)
```
```python
for i, trial in enumerate(trials):
    ...
    pos[name][i] = interp_to_taxis(...)
```

iii. The trajectory does not explicitly discuss runtime hotspots. This assessment is inferred from the code structure rather than from an explicit AI justification.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalarized: iterating over probes and cells in `neural_matrix()`, looping over trials during trajectory and motion-energy loading, looping over trials inside `paw_speed()`, and building each trial's output arrays in Python. Some of these could potentially be vectorized, but the ragged frame counts and HDF5 object-reference layout make full vectorization awkward.

ii. <Code snippets>
```python
for p in probes:
    ...
    for icell, q in enumerate(qualities):
```
```python
for i, trial in enumerate(trials):
    ...
```
```python
for i in range(len(trials)):
    neural.append(np.ascontiguousarray(rates[i].T))
    inputs.append(time_input.copy())
    ...
```

iii. The trajectory gives no explicit optimization rationale here. The script prioritizes mirroring the MATLAB analysis steps over aggressive vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The code avoids some recomputation by reusing `TAXIS`, but it still repeats a few things: the same one-row time input is copied into every trial, separate trial loops are run for trajectory loading and for motion-energy loading, and separate post-processing passes are run for tongue, paw, and motion-energy outputs after shared alignment. It also performs repeated list-index lookup when constructing `subject_idx`.

ii. <Code snippets>
```python
time_input = TAXIS.astype(np.float32)[None, :]
...
inputs.append(time_input.copy())
```
```python
pos, has_video = load_traj(f, obj, trials, align, vidshift)
me = load_motion_energy(anm, date, f, obj, trials, align, vidshift)
...
tspeed, tvis = tongue_speed(pos[TONGUE_FEATURE[0]])
pspeed, pvis = paw_speed([pos[n] for n, _ in PAW_FEATURES])
```
```python
subject_idx = np.array([subjects.index(s['session_info']['subject'])
                        for s in sessions], dtype=np.int64)
```

iii. The trajectory does not explicitly discuss repeated work. The only stated priority was following the paper's pipeline closely.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several intermediate or bookkeeping values that the downstream decoder does not use: `sample`, `delay`, and `no` in `load_behavior()`; `keep_ids` in `neural_matrix()`; extensive session metadata counts; and full interpolated x/y position arrays for all requested features before collapsing them to discretized velocity codes. Those extra computations are mostly for faithful reconstruction or summary logging, not for the decoder's final inputs and outputs.

ii. <Code snippets>
```python
'no': np.nan_to_num(vec(bp, 'no')[:n]) > 0,
'sample': vec(ev, 'sample')[:n],
'delay': vec(ev, 'delay')[:n],
```
```python
keep_rates, keep_ids = [], []
...
keep_ids.append((p, icell))
...
return rates.astype(np.float32), nquality, keep_ids
```
```python
pos = {name: np.full((len(trials), NBINS, 2), np.nan) for name, _ in wanted}
...
'session_info': {
    'subject': anm,
    'date': date,
    'probes': probes,
    ...
}
```

iii. The trajectory does not explicitly justify these extras. The code comments mainly justify them as part of following the MATLAB pipeline or providing dataset summaries.
