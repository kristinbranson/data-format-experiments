# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load all available ephys data. It hard-codes 12 two-context `Ephys_Behavior` sessions in `SESSIONS`, builds fixed paths inside `/app/data/Ephys_Behavior`, opens the `data_structure` file with `h5py`, and loads motion energy from the paired `motionEnergy` file with `scipy.io.loadmat`.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'

SESSIONS = [
    ('JEB6',  '2021-04-18', 2),
    ...
    ('JEB19', '2023-04-21', 1),
]

def session_paths(anm, date):
    return (os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat'),
            os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))

def convert_session(anm, date, probe, verbose=True):
    data_path, me_path = session_paths(anm, date)
    f = h5py.File(data_path, 'r')
```

iii. In `CONVERSION_NOTES.md`, the AI says it intentionally restricted the dataset to the 12 two-context sessions because context is a required decoder output and these are “the only ephys sessions that contain both behavioural contexts.”

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the hard-coded `anm` value in each `(animal, date, probe)` tuple. Subjects are accumulated in first-seen order while sessions are assembled, and `subject_idx` stores the index of each session's animal in that list.

ii.
```python
for k, (anm, date, probe) in enumerate(sessions):
    ...
    if anm not in data['subjects']:
        data['subjects'].append(anm)
    data['subject_idx'].append(data['subjects'].index(anm))
```

iii. The notes justify this by saying the two-context subset contains 7 animals and that the paper’s “six mice” count appears to under-count by one.

## 1-c. How are the data split into sessions?

i. One session is one tuple in `SESSIONS` and one HDF5 `data_structure_<ANM>_<DATE>.mat` file plus its paired `motionEnergy_<ANM>_<DATE>.mat`. Each session becomes one entry in `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
sessions = SESSIONS[:2] if args.sample else SESSIONS

for k, (anm, date, probe) in enumerate(sessions):
    neural, inp, out, info, extras = convert_session(anm, date, probe)
    ...
    data['neural'].append(neural)
    data['input'].append(inp)
    data['output'].append(out)
```

iii. The stated justification is that these 12 sessions are exactly the ones loaded by the paper’s Figure 8 / Extended Data Figure 2a scripts and are the only sessions containing both WC and DR contexts.

## 1-d. How are the data split into trials?

i. Trials are defined directly from `obj.bp.Ntrials` and per-trial Bpod arrays. The code keeps a boolean `keep` mask over trial indices, then `keep_idx = np.nonzero(keep)[0]` determines which trial numbers are exported. Neural, input, and output arrays are assembled by iterating over those kept trial indices.

ii.
```python
bp = load_bpod(f, o)
n = bp['Ntrials']
...
keep_idx = np.nonzero(keep)[0]
...
for j, tr in enumerate(keep_idx):
    neural_trials.append(np.ascontiguousarray(neural[:, :, tr].T))
    input_trials.append(input_trial.copy())
    ...
    output_trials.append(out)
```

iii. The AI’s notes treat Bpod’s per-trial arrays as the authoritative trial structure and describe trial curation as masking those trial rows rather than reconstructing trial boundaries from spikes or video.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that are hit, miss, or no-response, excludes early-lick trials, excludes photostimulation trials, excludes trials with missing `goCue`, and then drops any kept trial that lacks usable video.

ii.
```python
keep = (bp['hit'] | bp['miss'] | bp['no']) & ~bp['early'] & ~bp['stim'] & ~np.isnan(bp['goCue'])
...
n_dropped_no_video = int((keep & ~kin['has_video']).sum())
keep = keep & kin['has_video']
keep_idx = np.nonzero(keep)[0]
```

iii. The notes say early/stim filtering matches every reference `params.condition`, ignore trials are retained because the decoder needs an `ignore` class, and the one no-video trial is dropped because three decoder outputs would otherwise be undefined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the selected probe’s cluster spike times and trial assignments, aligned with Bpod go-cue times: `clu['trialtm']`, `clu['trial']`, and `bp['goCue']`.

ii.
```python
tm = np.array(f[clu['trialtm'][ci, 0]]).flatten()
tr = np.array(f[clu['trial'][ci, 0]]).flatten().astype(int)
...
aligned = tm - gocue[tr - 1]
```

iii. The notes explicitly map `obj.clu{probe}(c).trialtm`, `obj.clu{probe}(c).trial`, and `obj.bp.ev.goCue` to exported neural data and cite `alignSpikes.m` / `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 10 ms bins over `[-2.5, 2.5]` seconds, divides by `DT` to convert to firing rate, then applies a causal Gaussian smoothing kernel implemented as a port of `mySmooth.m`.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
SMOOTH = 15
...
rate = bin_spikes(f, clu, qual_keep, bp['goCue'], n, edges)
rate_s = my_smooth(rate.reshape(t, -1)).reshape(rate.shape).astype(np.float32)
```

iii. The notes justify this as matching the figure scripts’ `dt = 1/100` and `mySmooth(...,15,'reflect')`, and say the output should be “single-trial smoothed firing rate” rather than raw counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only one ALM probe per session, removes clusters whose raw quality string exactly matches `garbage`, `gabrga`, `noisy`, or `real?`, then applies a low-firing-rate filter computed from smoothed trial data averaged across seven Figure-8 conditions and time, keeping units with mean firing rate `> 1 Hz`.

ii.
```python
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')
...
quality = [matstr(f, r).strip() for r in np.array(clu['quality']).flatten()]
qual_keep = [i for i, q in enumerate(quality) if q not in BAD_QUALITY]
...
fr_mask, mean_fr = low_fr_mask(rate_s, fig8_conditions(bp))
unit_idx = np.nonzero(fr_mask)[0]
```

iii. In the notes, the AI says this was chosen to match `findClusters.m` and `removeLowFRClusters.m` from the Figure 8 pipeline and to reproduce the paper’s reported 214 single units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the go cue of its own trial, `aligned = trialtm - goCue[trial-1]`, before binning.

ii.
```python
aligned = tm - gocue[tr - 1]
inwin = (aligned >= edges[0]) & (aligned < edges[-1])
h, _, _ = np.histogram2d(aligned[inwin], tr[inwin], bins=[edges, trial_bins])
```

iii. The AI’s notes say this is a direct port of `alignSpikes.m` and that all streams are aligned to go cue by task requirement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported neural data use 10 ms bins. Spikes are binned directly at that resolution; no additional temporal rebinning is applied afterward.

ii.
```python
DT = 0.01
edges = np.arange(TMIN, TMAX + DT / 2, DT)
taxis = (edges[:-1] + edges[1:]) / 2.0
```

iii. The AI justifies 10 ms by citing the paper’s figure scripts, which it says use `params.dt = 1/100`, and by preferring a window fully covered by video.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read from a dedicated raw variable. It is constructed from the chosen analysis window and bin size around the go cue, using the same bin edges used for neural alignment.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT)
taxis = (edges[:-1] + edges[1:]) / 2.0
input_trial = taxis.astype(np.float32)[None, :]
```

iii. The notes describe this as the single required decoder input and say it is simply the go-cue-centered bin-centre grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes bin edges from `TMIN`, `TMAX`, and `DT`, converts them to bin centres, casts to `float32`, and reuses the same `(1, T)` array for each trial.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT)
taxis = (edges[:-1] + edges[1:]) / 2.0
input_trial = taxis.astype(np.float32)[None, :]
...
input_trials.append(input_trial.copy())
```

iii. The notes justify this as matching the reference time axis and keeping the decoder input exactly equal to the neural bin grid.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same `taxis` bin-centre grid that neural spikes are binned onto, so timepoint `k` in `input` corresponds to time bin `k` in `neural`.

ii.
```python
taxis = (edges[:-1] + edges[1:]) / 2.0
...
rate = bin_spikes(f, clu, qual_keep, bp['goCue'], n, edges)
...
input_trial = taxis.astype(np.float32)[None, :]
```

iii. The notes explicitly say the input is “the 10 ms bin-centre grid” and use equality between `input` and the shared time axis as a sanity check.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from per-trial Bpod task variables `L`, `R`, `hit`, `miss`, and `no`.

ii.
```python
lick_dir = np.full(n, -1, dtype=np.int64)
lick_dir[(bp['L'] & bp['hit']) | (bp['R'] & bp['miss'])] = 0
lick_dir[(bp['R'] & bp['hit']) | (bp['L'] & bp['miss'])] = 1
lick_dir[bp['no']] = 2
```

iii. The notes say this follows the paper’s `getPrevChoice` logic, with an added `none` class required by the decoder task.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Trials are mapped to categorical codes: left lick `0`, right lick `1`, and no response `2`. The per-trial class is then broadcast across all time bins of that trial.

ii.
```python
lick_dir[(bp['L'] & bp['hit']) | (bp['R'] & bp['miss'])] = 0
lick_dir[(bp['R'] & bp['hit']) | (bp['L'] & bp['miss'])] = 1
lick_dir[bp['no']] = 2
...
out[0, :] = lick_dir[tr]
```

iii. The notes justify retaining no-response trials because the task explicitly asks for both an `ignore` outcome class and a `none` lick-direction class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the Bpod per-trial `autowater` flag.

ii.
```python
context = np.where(bp['autowater'], 0, 1).astype(np.int64)
```

iii. The AI notes say context blocks are defined by `bp.autowater`, matching the paper’s alternative-context-task code.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater == True` is relabelled to WC (`0`) and `False` to DR (`1`), then broadcast across time within each trial.

ii.
```python
context = np.where(bp['autowater'], 0, 1).astype(np.int64)
...
out[1, :] = context[tr]
```

iii. The notes say this is a direct recoding of the block variable and preserves the DR/WC block structure.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from Bpod per-trial `miss`, `hit`, and `no`.

ii.
```python
outcome = np.full(n, -1, dtype=np.int64)
outcome[bp['miss']] = 0
outcome[bp['hit']] = 1
outcome[bp['no']] = 2
```

iii. The notes describe this as the standard hit/miss/no mapping, with ignore trials retained because they are required by the decoder task.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps miss to incorrect (`0`), hit to correct (`1`), and no-response to ignore (`2`), then repeats the resulting class across all time bins of the trial.

ii.
```python
outcome[bp['miss']] = 0
outcome[bp['hit']] = 1
outcome[bp['no']] = 2
...
out[2, :] = outcome[tr]
```

iii. The AI’s notes say this matches the task’s required class order and is one of the deliberate deviations from the paper’s own exclusion of ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side camera only, using `obj.traj{side}.ts` for the feature named `tongue`, together with the trial’s go cue and the session video offset. The bottom camera tongue tracks are not used.

ii.
```python
v1 = f[o['traj'][0, 0]]      # side camera
...
i_tongue = n1.index('tongue')
...
x = resample(ts1[i_tongue, 0])
y = resample(ts1[i_tongue, 1])
```

iii. In the notes, the AI maps `obj.traj{1}(t).ts[:, :, tongue]` to `tongue_velocity` and says the feature is resampled onto the aligned 10 ms grid before computing speed.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code linearly interpolates tongue x/y from video frames onto the 10 ms analysis grid, treats finite interpolated samples as visible, nearest-fills gaps, computes speed as the Euclidean norm of `np.gradient` on the filled x/y traces, and leaves bins without visible samples as class 2 later.

ii.
```python
def resample(y, times=None):
    times = tt if times is None else times
    k = min(len(times), len(y))
    return np.interp(taxis, times[:k], y[:k], left=np.nan, right=np.nan)

x = resample(ts1[i_tongue, 0])
y = resample(ts1[i_tongue, 1])
vis = ~np.isnan(x)
if vis.any():
    vx = np.gradient(nearest_fill(x))
    vy = np.gradient(nearest_fill(y))
    tongue_speed[:, tr] = np.hypot(vx, vy)
```

iii. The notes justify this as following `findPosition.m` / `findVelocity.m` on the aligned decoder grid and then splitting by the per-session median over visible time points.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes a per-session 50th-percentile threshold over visible tongue-speed samples from kept trials only. Visible samples below threshold are class `0`, visible samples at or above threshold are class `1`, and everything else is class `2` (`not visible`).

ii.
```python
def discretize(values, visible, keep_trials):
    vals = values[:, keep_trials]
    vis = visible[:, keep_trials] & ~np.isnan(vals)
    ...
    thresh = np.percentile(vals[vis], 50)
    out = np.full(vals.shape, 2, dtype=np.int64)
    out[vis & (vals < thresh)] = 0
    out[vis & (vals >= thresh)] = 1
    return out, float(thresh)

tongue_cat, tongue_thresh = discretize(kin['tongue_speed'], kin['tongue_vis'], keep_idx)
```

iii. The notes explicitly justify computing the percentile only over visible samples because `not visible` is meant to remain its own class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are corrected by the session-wide offset from `findVideoOffset` logic, then shifted by each trial’s go cue, interpolated onto the same 10 ms `taxis` used for neural data, and exported on that shared grid.

ii.
```python
def video_offset(f, o, bp):
    fs = float(np.array(o['sglx']['fs'])[0, 0])
    bitstart = np.array(o['sglx']['bitcode']['bitstart']).flatten()
    return mode_value(bitstart) / fs - mode_value(bp['bitStart'])

def trial_frame_times(f, traj_view, trial, vidshift, gocue_t):
    ft = np.array(f[traj_view['frameTimes'][trial, 0]]).flatten()
    ...
    return ft - vidshift - gocue_t
```

iii. The notes cite `findVideoOffset.m` and repeatedly justify using one shared go-cue-aligned 10 ms grid for spikes, kinematics, and motion energy.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera `traj` data using the features `top_paw` and `bottom_paw`, plus the trial go cue and session video offset.

ii.
```python
v2 = f[o['traj'][1, 0]]      # bottom camera
n2 = feature_names(f, v2)
i_paws = [n2.index(p) for p in ('top_paw', 'bottom_paw') if p in n2]
```

iii. The notes state that paws are taken from the bottom camera and that the two paw markers are averaged after velocity extraction.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each available paw marker, the code interpolates x/y onto the 10 ms grid, nearest-fills missing positions, computes `np.gradient`, subtracts the median first difference of each coordinate as baseline-drift removal, converts to speed, masks invisible bins back to `NaN`, and averages speeds across the available paw markers.

ii.
```python
px = resample(ts2[i, 0], tt2)
py = resample(ts2[i, 1], tt2)
vi = ~np.isnan(px)
...
pxf, pyf = nearest_fill(px), nearest_fill(py)
vx = np.gradient(pxf) - np.median(np.diff(pxf))
vy = np.gradient(pyf) - np.median(np.diff(pyf))
s = np.hypot(vx, vy)
s[~vi] = np.nan
...
paw_speed[:, tr] = np.where(cnt > 0, np.nansum(sm, axis=0) / np.maximum(cnt, 1), np.nan)
```

iii. The notes justify this as using the paper’s bottom-view paw tracking with the reference baseline-drift correction and then applying the decoder’s required median split.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI uses the same `discretize` function as for tongue velocity: a per-session median over visible kept-trial samples, producing categories `0` below median, `1` at/above median, and `2` not visible.

ii.
```python
paw_cat, paw_thresh = discretize(kin['paw_speed'], kin['paw_vis'], keep_idx)
```

iii. The notes say this choice is driven by the task specification and preserves `not visible` as a separate category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw video times are corrected by the same session-wide video offset, shifted by the trial’s go cue, interpolated onto the shared 10 ms `taxis`, and exported on the same time axis as neural data.

ii.
```python
tt2 = trial_frame_times(f, v2, tr, vidshift, bp['goCue'][tr])
...
px = resample(ts2[i, 0], tt2)
py = resample(ts2[i, 1], tt2)
```

iii. The notes justify using the same go-cue-aligned grid for all streams so that paw bins line up directly with neural bins.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<ANM>_<DATE>.mat` file. The code reads `me.data` and, if necessary, unwraps an extra struct layer before turning it into a list of per-trial traces.

ii.
```python
def load_motion_energy(me_path):
    m = sio.loadmat(me_path, struct_as_record=False, squeeze_me=False)
    me = m['me'][0, 0]
    data = me.data
    if not isinstance(data, np.ndarray) or data.dtype != object:
        data = data[0, 0].data
    cells = [np.asarray(data[i, 0]).flatten() for i in range(data.shape[0])]
```

iii. The notes say this was chosen to match `loadMotionEnergy.m` and to handle the mixed file layouts found across sessions.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates each trial’s motion-energy trace from video-frame times onto the shared 10 ms grid, nearest-fills edge gaps within a trial, and then later discretizes the resulting values by the per-session median.

ii.
```python
mv = np.asarray(me_cells[tr], float).flatten()
k = min(len(mv), nf)
if k > 1:
    me[:, tr] = nearest_fill(
        np.interp(taxis, tt[:k], mv[:k], left=np.nan, right=np.nan))
```

iii. The notes justify this by explicitly referencing `loadMotionEnergy.m`, which interpolates onto the aligned analysis axis and applies `fillmissing(...,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The code computes a per-session 50th-percentile threshold over visible motion-energy samples from kept trials. Values below threshold become `0`, values at or above threshold become `1`, and class `2` is reserved for no-video cases.

ii.
```python
me_vis = ~np.isnan(kin['me']) & kin['has_video'][None, :]
me_cat, me_thresh = discretize(kin['me'], me_vis, keep_idx)
```

iii. The notes say the threshold should be computed only on available samples and that `no video` should remain a separate class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the same video clock correction and go-cue alignment as the kinematics, then is resampled onto the same 10 ms `taxis` used for neural data.

ii.
```python
tt = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])
...
me[:, tr] = nearest_fill(
    np.interp(taxis, tt[:k], mv[:k], left=np.nan, right=np.nan))
```

iii. The notes repeatedly describe motion energy as sharing the same video alignment path and common decoder time axis as the other time-varying outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or bad video is handled by rejecting trials with unusable frame times (`None` when `frameTimes` are empty or contain `NaN`) and by using nearest-filled interpolated positions within otherwise usable trials. Kinematic visibility masks come from whether the interpolated positions are finite. Motion energy is nearest-filled within a trial. No special truncation is applied for neural recording ending before behavioral trials do.

ii.
```python
def trial_frame_times(f, traj_view, trial, vidshift, gocue_t):
    ft = np.array(f[traj_view['frameTimes'][trial, 0]]).flatten()
    if ft.size == 0 or np.all(np.isnan(ft)):
        return None
    if np.isnan(ft).any():
        return None
    return ft - vidshift - gocue_t

def nearest_fill(x):
    ...

n_dropped_no_video = int((keep & ~kin['has_video']).sum())
keep = keep & kin['has_video']
```

iii. The notes justify dropping the one no-video trial because three decoder outputs would be undefined, and justify nearest-fill for motion energy as matching `loadMotionEnergy.m`. They also frame `not visible` versus `no video` as separate concepts.

## 11-a. What are the most time-consuming steps of the code?

i. The AI identifies kinematics and motion-energy extraction as the dominant runtime, with spike binning and smoothing much cheaper. Its recorded timings are roughly 1.3–2.2 s/session for kinematics, versus 0.11–0.35 s for spike binning and 0.07–0.32 s for smoothing.

ii.
```python
timing['bin_spikes'] = time.time() - t0
...
timing['smooth'] = time.time() - t0
...
timing['kinematics'] = time.time() - t0
...
timing=timing, elapsed=time.time() - t_start,
```

iii. `CONVERSION_NOTES.md` Step 6 and Step 7 explicitly summarize these timings and say “~2 s/session; 1.5–2 s of that is HDF5 reads of the DeepLabCut trajectories.”

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI calls out several optimizations it already made and leaves the remaining major loops over clusters (`bin_spikes`), trials (`extract_kinematics`), and final kept-trial assembly. The kinematics path still performs repeated per-trial interpolation and per-feature gradient calculations.

ii.
```python
for k, ci in enumerate(keep_idx):
    ...
    h, _, _ = np.histogram2d(aligned[inwin], tr[inwin], bins=[edges, trial_bins])

for tr in range(n):
    ...
    return np.interp(taxis, times[:k], y[:k], left=np.nan, right=np.nan)

for j, tr in enumerate(keep_idx):
    neural_trials.append(...)
    output_trials.append(out)
```

iii. The notes say the AI intentionally replaced slower per-trial/per-condition loops where it could, especially by using `np.histogram2d`, reshaped smoothing, and vectorized `nearest_fill`.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly interpolates separate video-derived streams trial by trial, computes gradients independently for tongue and each paw marker, and then discretizes tongue, paw, and motion energy in three separate passes. It also copies the same per-trial input time axis for every kept trial.

ii.
```python
def resample(y, times=None):
    ...
    return np.interp(taxis, times[:k], y[:k], left=np.nan, right=np.nan)

x = resample(ts1[i_tongue, 0])
y = resample(ts1[i_tongue, 1])
...
px = resample(ts2[i, 0], tt2)
py = resample(ts2[i, 1], tt2)
...
tongue_cat, tongue_thresh = discretize(...)
paw_cat, paw_thresh = discretize(...)
me_cat, me_thresh = discretize(...)
```

iii. The notes present these repeated passes as acceptable because each output needs its own session-level threshold and each feature has its own visibility pattern.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `lickL` and `lickR` events that are only used for diagnostics, computes `n_single_units` and `move_thresh` only for reporting, keeps extensive `extras` structures for optional plots, and loads/unwraps motion-energy metadata even though the downstream decoder only uses the discretized trace.

ii.
```python
out['lickL'] = [np.array(f[r]).flatten() for r in np.array(ev['lickL']).flatten()[:n]]
out['lickR'] = [np.array(f[r]).flatten() for r in np.array(ev['lickR']).flatten()[:n]]
...
n_single = sum(q.lower() in SINGLE_UNIT_QUALITY for q in qualities_kept)
...
me_cells, move_thresh = load_motion_energy(me_path)
...
extras = dict(bp=bp, kin=kin, taxis=taxis, keep_idx=keep_idx, mean_fr=mean_fr,
              neural=neural, tongue_cat=tongue_cat, paw_cat=paw_cat, me_cat=me_cat, ...)
```

iii. The notes justify most of this as provenance and sanity-check support rather than as part of the final exported dataset.
