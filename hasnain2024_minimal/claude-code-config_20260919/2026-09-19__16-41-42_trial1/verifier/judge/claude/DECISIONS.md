# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB v7.3 (HDF5) file, `data_structure_<anm>_<date>.mat`, loaded using `h5py`. Only the `Ephys_Behavior` folder is used (the 25 fixed-delay sessions); `RandomizedDelay_Ephys_Behavior` is excluded. The 25 session names and their probes are hard-coded in the `SESSIONS` list, transcribed from the authors' `load<ANM>_ALMVideo.m` scripts. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` using `scipy.io.loadmat`.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'

SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ('EKH3',  '2021-08-11', [2]),
    ...
    ('JEB19', '2023-04-21', [1]),
]

def process_session(anm, date, probes):
    path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
    with h5py.File(path, 'r') as f:
        obj = f['obj']
        ...
```

iii. The agent stated: "I excluded `RandomizedDelay_Ephys_Behavior`: it is a separate task variant (delay 0.3–3.6 s) analysed separately in the paper, so the sample/delay epochs would land at inconsistent times inside a go-cue-aligned window, and it contributes almost no WC trials."

## 1-b. How are the data split into subjects?

i. The animal name is the first element of the `SESSIONS` tuple (e.g., `'EKH1'`). Subjects are collected in order of first appearance and stored in a list. `subject_idx` maps each session to its subject index.

ii.
```python
for anm, date, probes in SESSIONS:
    ...
    if anm not in subjects:
        subjects.append(anm)
    ...
    data['subject_idx'].append(subjects.index(anm))
```

iii. The animal ID is taken directly from the session list, which mirrors the authors' loading scripts.

## 1-c. How are the data split into sessions?

i. One session is one entry in `SESSIONS`, keyed by `(anm, date, probes)`. Only the 25 fixed-delay sessions from `Ephys_Behavior` are included. Each session becomes one element of `neural`, `input`, and `output`. Sessions with fewer than 10 units (`MIN_UNITS = 10`) or fewer than 2 trials are skipped (though in practice none were skipped).

ii.
```python
if info['nunits'] < MIN_UNITS:
    print(f"  skipping {anm} {date}: only {info['nunits']} units")
    continue
if len(neural) < 2:
    print(f"  skipping {anm} {date}: only {len(neural)} trials")
    continue
```

iii. The agent justified session selection by noting these are "the paper's main dataset (25 sessions / 1,651 units)."

## 1-d. How are the data split into trials?

i. Trials are defined by `bp.Ntrials`. Each per-trial field (`hit`, `miss`, `R`, `early`, etc.) is read as a flattened vector. Trial indices surviving curation (`keep = np.flatnonzero(trial_mask)`) are used to index into the neural and behavioral arrays.

ii.
```python
def load_behavior(obj):
    bp = obj['bp']
    n = int(_vec(bp, 'Ntrials')[0])
    beh = {
        'ntrials': n,
        'R': _vec(bp, 'R') > 0.5,
        ...
    }
    return beh
```

iii. The Bpod table defines trials directly; no inference is needed.

## 1-e. How are trials filtered based on quality controls?

i. Two filters: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are dropped, following the paper. Ignore trials are kept because "none"/"ignore" are required output categories. There is no recording-length filter (unlike the reference).

ii.
```python
trial_mask = (~beh['early']) & (~beh['stim'])
keep = np.flatnonzero(trial_mask)
```

iii. The agent stated: "every condition in the paper's scripts is restricted to `~stim.enable & ~early`." Ignore trials are kept for the decoder task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`: each cluster carries `trial` (1-based trial index per spike), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). `bp.ev.goCue` provides alignment times.

ii.
```python
trials = np.array(f[trial_refs[iclu]]).flatten().astype(int) - 1
times = np.array(f[tm_refs[iclu]]).flatten()
times = times - align_times[trials]
```

iii. This follows the reference's `alignSpikes.m`.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, then histogrammed into 10 ms bins over [-2.5, 2.5] s (500 bins). Counts are converted to spikes/s by dividing by DT, then smoothed with a **causal** Gaussian kernel of width 15 bins (porting `mySmooth.m`). The causal kernel zeros the first half of a `gausswin(15)`.

ii.
```python
DT = 0.01            # s (10 ms bins, params.dt = 1/100)
SMOOTH_N = 15        # params.smooth, width of the causal gaussian kernel (bins)

def my_smooth(x, n=SMOOTH_N):
    kern = gausswin(n)
    kern[:n // 2] = 0.0          # causal
    kern = kern / kern.sum()
    ...

rate = my_smooth((counts / DT).T).T
```

iii. The agent ported `mySmooth.m` and `getSeq.m` from the paper's MATLAB code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters whose quality label (lowercased) is in `BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}` are dropped. Second, units with mean firing rate <= 1 Hz are dropped. Note: `poor` is NOT in the drop list (unlike the reference).

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0

if qual.lower() in BAD_QUALITY:
    continue
...
mean_fr = rates.mean(axis=(1, 2)) if rates.size else np.zeros(0)
use = mean_fr > LOW_FR
rates = rates[use]
```

iii. The agent cited `findClusters.m` for the quality rule and the paper for the 1 Hz threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are subtracted by `align_times[trial]` which is `bp.ev.goCue`, aligning to go cue onset.

ii.
```python
ALIGN_EVENT = 'goCue'
align = beh['align']  # = _vec(bp['ev'], ALIGN_EVENT)
times = times - align_times[trials]
```

iii. This follows `alignSpikes.m`: `trialtm_aligned = trialtm - event(trial)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (DT = 0.01 s), producing 500 time bins over [-2.5, 2.5] s. The agent interpreted `params.dt = 1/100` as 10 ms. No rebinning is applied.

ii.
```python
DT = 0.01            # s (10 ms bins, params.dt = 1/100)
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
NT = len(TAXIS)      # 500
```

iii. The agent cited `params.dt = 1/100` from the paper's scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The time axis is constructed from the bin edges, not from raw data. It represents time from go cue in seconds.

ii.
```python
TAXIS = (EDGES + DT / 2)[:-1]
time_input = TAXIS.astype(np.float32).reshape(1, NT)
```

iii. Defined by the binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is the bin centers of the [-2.5, 2.5] s window at 10 ms resolution. No processing beyond constructing the axis.

ii.
```python
TAXIS = (EDGES + DT / 2)[:-1]
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis IS the same binning grid used for neural data. Both use the same `EDGES` and `TAXIS`.

ii.
```python
time_input = TAXIS.astype(np.float32).reshape(1, NT)
input_trials.append(time_input.copy())
```

iii. Shared time axis ensures alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields: `bp.R`, `bp.L`, `bp.hit`, `bp.miss`. Lick direction is inferred from the combination of instructed side and outcome.

ii.
```python
right = (beh['R'] & beh['hit']) | (beh['L'] & beh['miss'])
left = (beh['L'] & beh['hit']) | (beh['R'] & beh['miss'])
```

iii. The agent cited `getPrevChoice.m` for this logic.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port; a miss means it licked the opposite. Left=0, right=1, none=2.

ii.
```python
lick_dir = np.full(n, 2, dtype=np.int8)      # 2 = none
lick_dir[left] = 0
lick_dir[right] = 1
```

iii. Same logic as reference; codes match the instructions.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`, which marks WC (water-cued) trials.

ii.
```python
'autowater': _vec(bp, 'autowater') > 0.5,
```

iii. Directly read from the trial table.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=True -> WC (0), otherwise DR (1).

ii.
```python
context = np.where(beh['autowater'], 0, 1).astype(np.int8)
```

iii. Matches the instructions (WC=0, DR=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss`. Trials that are neither hit nor miss are classified as ignore.

ii.
```python
'hit': _vec(bp, 'hit') > 0.5,
'miss': _vec(bp, 'miss') > 0.5,
```

iii. Same approach as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Three classes: incorrect (0) for miss, correct (1) for hit, ignore (2) for the rest.

ii.
```python
outcome = np.full(n, 2, dtype=np.int8)       # 2 = ignore
outcome[beh['miss']] = 0
outcome[beh['hit']] = 1
```

iii. Matches the instructions exactly.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking in `obj.traj`, specifically the `tongue` feature from the **side camera only** (view 0). `frameTimes` and `ts` (x, y, likelihood) are read. The video offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart` is also used.

ii.
```python
TONGUE_FEATURES = [(0, 'tongue')]
```

iii. The agent uses only the side camera for the tongue, unlike the reference which uses both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. DLC positions are interpolated onto the analysis time axis (`TAXIS`) using linear interpolation. The gradient of x and y is computed (without smoothing beyond interpolation), and speed is `sqrt(vx^2 + vy^2)`. For the tongue, gaps are NOT filled (`fill=False`): velocity is computed within each contiguous stretch of visible frames. The result is split at the session 50th percentile; bins where the tongue was not detected are class 2 ("not visible").

ii.
```python
def speed_from_position(x, y, fill):
    visible = ~np.isnan(x)
    if fill:
        xf, yf = fill_nearest(x), fill_nearest(y)
        ...
    else:
        # tongue: never filled
        vx = np.full(NT, np.nan)
        vy = np.full(NT, np.nan)
        idx = np.flatnonzero(visible)
        if idx.size:
            splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for seg in splits:
                if seg.size == 1:
                    vx[seg] = 0.0; vy[seg] = 0.0
                else:
                    vx[seg] = np.gradient(x[seg]); vy[seg] = np.gradient(y[seg])
    return np.sqrt(vx ** 2 + vy ** 2), visible
```

iii. The agent followed `findPosition.m` and `findVelocity.m`, noting that the tongue is never gap-filled.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Split at the session 50th percentile of all visible timepoints across kept trials. Below threshold=0, at/above threshold=1, not visible=2.

ii.
```python
def median_of(values, visible):
    vals = values[keep][visible[keep]]
    vals = vals[~np.isnan(vals)]
    return np.median(vals) if vals.size else np.inf

thr_tongue = median_of(tongue_speed, tongue_vis)
tongue_cls = discretize(tongue_speed, tongue_vis, thr_tongue)
```

iii. The 50th percentile split matches the instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are corrected by the video offset (from `findVideoOffset.m`), then the go cue time is subtracted. Positions are interpolated onto the analysis time axis (`TAXIS`, same grid as neural data).

ii.
```python
t_rel = ft - vidshift - align_times[trial]
pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
pos[(view, feat)][1][trial] = interp_nan(TAXIS, t_rel, ts[fi, 1])
```

iii. The video offset is computed following `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from the bottom camera (view 1), using **both** `top_paw` and `bottom_paw` features.

ii.
```python
PAW_FEATURES = [(1, 'top_paw'), (1, 'bottom_paw')]
```

iii. The agent uses both tracked paws, averaging their speeds. The reference uses only `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For paws, `fill=True`: missing positions are filled with the nearest valid value (`fill_nearest`, porting MATLAB's `fillmissing`), and the per-trial median frame-to-frame drift is subtracted. The gradient of the filled x and y is computed, and speed is the magnitude. The two paws are averaged (NaN where neither is visible). The result is split at the session 50th percentile.

ii.
```python
if fill:
    xf, yf = fill_nearest(x), fill_nearest(y)
    vx, vy = np.gradient(xf), np.gradient(yf)
    vx = vx - np.nanmedian(np.diff(xf))
    vy = vy - np.nanmedian(np.diff(yf))
...
paw_vis = paw_vis_all.any(axis=0)
masked = np.where(paw_vis_all, paw_speed_all, np.nan)
paw_speed = np.nanmean(masked, axis=0)
```

iii. The agent followed `findPosition.m` (nearest-fill for paws) and `findVelocity.m` (drift subtraction).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session 50th percentile of visible timepoints, three classes.

ii.
```python
thr_paw = median_of(paw_speed, paw_vis)
paw_cls = discretize(paw_speed, paw_vis, thr_paw)
```

iii. Matches the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction, go cue subtraction, interpolation onto `TAXIS`.

ii.
```python
pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
```

iii. Same alignment mechanism as all video-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files, loaded with `scipy.io.loadmat`. Only files from `Ephys_Behavior` are searched (since the AI excluded RandomizedDelay sessions).

ii.
```python
def load_motion_energy(anm, date, align_times, frame_times, has_video, ntrials):
    md = sio.loadmat(os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))
    data = md['me'][0, 0]['data']
    if data.dtype != object:
        data = data[0, 0]['data']
    data = data.flatten()
```

iii. The agent followed `loadMotionEnergy.m` for the nested struct unwrapping.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values (one per frame) are interpolated onto the analysis time axis, then gaps are filled with nearest-neighbor interpolation (`fill_nearest`). The result is split at the session 50th percentile.

ii.
```python
me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
```

iii. The `fill_nearest` step fills NaN gaps in the motion energy, which the reference does NOT do (it simply bins the raw values).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session 50th percentile of visible timepoints, three classes (below/above/no video).

ii.
```python
thr_me = median_of(me, me_vis)
me_cls = discretize(me, me_vis, thr_me)
```

iii. Matches the instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times, corrected by video offset and go cue, then interpolated onto `TAXIS`.

ii.
```python
me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
```

iii. Same alignment approach as all video streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Trials with no usable video (`frame_times` all NaN or not overlapping window) get NaN positions, which become "not visible" (class 2). (2) For paws, missing DLC detections are filled with nearest-neighbor (`fill_nearest`). (3) For the tongue, gaps are NOT filled; velocity is computed only within contiguous visible segments. (4) Motion energy gaps are filled with nearest-neighbor. (5) Sessions with <10 units or <2 trials would be skipped.

ii.
```python
if ft.size == 0 or np.all(np.isnan(ft)):
    continue
...
me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
```

iii. The agent used nearest-fill for paws and motion energy following the MATLAB code, and kept gaps for the tongue.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 files is the most time-consuming step. Each session requires opening a large `.mat` file with h5py, iterating over all clusters to read spike data, and loading video tracking data with multiple dereferences.

ii.
```python
with h5py.File(path, 'r') as f:
    obj = f['obj']
    ...
```

iii. File I/O dominates; computation is relatively fast.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cluster loop in `load_spikes` reads each cluster's spikes individually, then uses `np.add.at` for binning. The per-trial loops for video traces (`load_video_traces`) and speed computation (`speed_from_position`) iterate trial by trial. The per-trial motion energy loop could also be vectorized.

ii.
```python
for iclu in range(quality_refs.size):
    ...
    np.add.at(counts, (pos[inwin], bin_idx), 1.0)
```

iii. The per-cluster loop is necessary because each cluster's data is stored as separate HDF5 references. The per-trial loops handle variable-length frame data.

## 11-c. What processing does the code repeat multiple times?

i. The `_vec(bp, ...)` calls re-read behavioral fields from HDF5 multiple times (once in `load_behavior`, once in `video_shift`). Frame times are also read multiple times for different features from the same view in `load_video_traces`.

ii.
```python
def load_behavior(obj):
    ...
    'align': _vec(bp['ev'], ALIGN_EVENT),
...
def video_shift(f, obj):
    bit_start = _mode(_vec(obj['bp']['ev'], 'bitStart'))
```

iii. The redundancy is minor since HDF5 caches reads.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `nsingle_units`, `quality_counts`, `n_wc_trials`, `n_dr_trials`, and other diagnostic info that goes into metadata but is not used by the decoder. The `fill_nearest` call on motion energy fills gaps that would otherwise be marked as "no video" class — this processing changes the output values in a way that may not match the reference.

ii.
```python
info = {
    ...
    'nsingle_units': int(sum(q.lower() in ('excellent', 'great', 'good')
                             for q in qualities)),
    'quality_counts': dict(Counter(q for q in qualities)),
    ...
}
```

iii. These are diagnostics, not affecting the decoder directly.
