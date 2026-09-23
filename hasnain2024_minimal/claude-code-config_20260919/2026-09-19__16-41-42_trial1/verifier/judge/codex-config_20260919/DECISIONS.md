# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 25 fixed-delay sessions and their ALM probes, opens each `data_structure_*.mat` from `/app/data/Ephys_Behavior` with `h5py`, and loads the matching motion-energy file with `scipy.io.loadmat`. It does not load the 19 randomized-delay sessions.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'
SESSIONS = [('EKH1', '2021-08-07', [2]), ...]
for anm, date, probes in SESSIONS:
    neural, inp, out, info = process_session(anm, date, probes)
```

iii. The trajectory shows that the AI inspected the authors' `load<ANM>_ALMVideo.m` scripts and chose the 25 fixed-delay sessions used in several paper figures. It explicitly excluded randomized-delay sessions as a separate task variant whose variable sample/delay timing would be inconsistent in the go-cue window and which has almost no WC trials. That rationale conflicts with the instruction to convert all relevant supplied data and the reference's inclusion of those sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is the `anm` value in each hard-coded session tuple. Unique subjects are accumulated in first-seen order, and each retained session receives the corresponding integer index.

ii.
```python
if anm not in subjects:
    subjects.append(anm)
data['subject_idx'].append(subjects.index(anm))
```

iii. The trajectory shows inspection of filenames and loader metadata; the AI used the stable animal identifier embedded in its session list.

## 1-c. How are the data split into sessions?

i. Every hard-coded `(animal, date, probes)` tuple is one session and produces one element of `neural`, `input`, and `output`, unless it has fewer than 10 retained units or fewer than two trials.

ii.
```python
for anm, date, probes in SESSIONS:
    neural, inp, out, info = process_session(anm, date, probes)
    if info['nunits'] < MIN_UNITS: continue
    if len(neural) < 2: continue
```

iii. The AI followed the paper's session/probe loader scripts and its interpretation of the paper's minimum-unit criterion.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` determines the trial count. Per-trial behavior arrays are indexed directly; spike `trial` values are converted from one-based to zero-based; each retained trial becomes one matrix/list entry.

ii.
```python
n = int(_vec(bp, 'Ntrials')[0])
trials = np.array(f[trial_refs[iclu]]).flatten().astype(int) - 1
for pos_i, trial in enumerate(keep):
    neural_trials.append(rates[:, pos_i, :])
```

iii. The trajectory records direct inspection of `bp`, cluster trial indices, and the MATLAB data layout.

## 1-e. How are trials filtered based on quality controls?

i. Early-lick and photostimulation trials are excluded. Ignore trials are retained. Unlike the reference, trials after the electrophysiology recording ends are not explicitly removed.

ii.
```python
trial_mask = (~beh['early']) & (~beh['stim'])
keep = np.flatnonzero(trial_mask)
```

iii. The AI cites the paper scripts' repeated `~stim.enable & ~early` condition and explicitly retains ignore trials because ignore is a requested output category.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected probes in `obj.clu`: each cluster's `quality`, spike `trial`, and within-trial `trialtm`, plus `bp.ev.goCue` for alignment.

ii.
```python
quality_refs = np.array(cl['quality']).flatten()
trial_refs = np.array(cl['trial']).flatten()
tm_refs = np.array(cl['trialtm']).flatten()
times = times - align_times[trials]
```

iii. The AI inspected `alignSpikes.m`, `getSeq.m`, and live HDF5 cluster structures, then ported those fields.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins, divided by 0.01 s to produce Hz, then smoothed with a 15-bin causal half-Gaussian implementation. Probe populations are concatenated.

ii.
```python
DT = 0.01
np.add.at(counts, (pos[inwin], bin_idx), 1.0)
rate = my_smooth((counts / DT).T).T
```

iii. The AI states this mirrors `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`, interpreting `params.dt = 1/100` and `params.smooth = 15` as applicable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labeled garbage, `gabrga`, noisy, or `real?` are discarded; retained units must have mean rate strictly above 1 Hz. Entire sessions below 10 units are dropped. The reference additionally drops `poor` units.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
use = rates.mean(axis=(1, 2)) > LOW_FR
if info['nunits'] < MIN_UNITS: continue
```

iii. The AI cites `findClusters.m`, `removeLowFRClusters.m`, and the paper's 1 Hz/session inclusion statements.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's `goCue` time subtracted before binning over −2.5 to +2.5 seconds.

ii.
```python
times = times - align_times[trials]
inwin = (times >= TMIN) & (times < TMAX)
```

iii. The AI directly cites and ports `alignSpikes.m`'s subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (500 samples over five seconds). Raw spikes are temporally binned, and video traces are interpolated to the 10 ms bin-center axis. This differs from the reference's 5 ms/1000-bin grid.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
```

iii. It justified 10 ms from a paper analysis setting `params.dt = 1/100`, but did not reconcile that choice with the reference pipeline's 5 ms setting.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a generated bin-center grid relative to `bp.ev.goCue`, rather than a raw measured trace.

ii.
```python
TAXIS = (EDGES + DT / 2)[:-1]
time_input = TAXIS.astype(np.float32).reshape(1, NT)
```

iii. The AI used the requested alignment event and the same analysis grid as its neural matrices.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Uniform edges from −2.5 to +2.5 s are constructed at 10 ms spacing, and their centers are copied into every trial.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
input_trials.append(time_input.copy())
```

iii. The grid is intended to match `getSeq.m` and the decoder's time-varying input requirement.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact bins used for spike counts, so input column `k` corresponds to neural column `k`.

ii.
```python
bin_idx = np.floor((times[inwin] - TMIN) / DT).astype(int)
time_input = TAXIS.astype(np.float32).reshape(1, NT)
```

iii. Shared constants and dimensions provide alignment without further interpolation.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.R`, `bp.L`, `bp.hit`, and `bp.miss`; non-hit/non-miss trials are no-lick.

ii.
```python
right = (beh['R'] & beh['hit']) | (beh['L'] & beh['miss'])
left = (beh['L'] & beh['hit']) | (beh['R'] & beh['miss'])
```

iii. The AI cites `getPrevChoice.m`: correct trials lick the instructed side and misses lick the opposite side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Left is encoded 0, right 1, and none 2, then broadcast across all time bins of the trial.

ii.
```python
lick_dir = np.full(n, 2, dtype=np.int8)
lick_dir[left] = 0; lick_dir[right] = 1
out[0] = lick_dir[trial]
```

iii. The third class preserves requested ignore/no-lick trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from per-trial `bp.autowater`.

ii.
```python
'autowater': _vec(bp, 'autowater') > 0.5
```

iii. The AI identified autowater as the marker for uncued water-context trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater maps to WC=0 and all other trials to DR=1; the scalar class is broadcast over time.

ii.
```python
context = np.where(beh['autowater'], 0, 1).astype(np.int8)
out[1] = context[trial]
```

iii. This directly implements the two requested contexts.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit` and `bp.miss`; `bp.no` is loaded but the class is inferred as neither hit nor miss.

ii.
```python
'hit': _vec(bp, 'hit') > 0.5,
'miss': _vec(bp, 'miss') > 0.5,
'no': _vec(bp, 'no') > 0.5,
```

iii. The AI inspected `getOutcome.m` and treats the three outcome flags as mutually exclusive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss is incorrect=0, hit is correct=1, and the default is ignore=2; the result is broadcast across time.

ii.
```python
outcome = np.full(n, 2, dtype=np.int8)
outcome[beh['miss']] = 0
outcome[beh['hit']] = 1
```

iii. This matches the requested categorical labels while retaining ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera `tongue` feature's x/y positions from `obj.traj`, its frame times, `bp.ev.goCue`, and bitcode timing for video-clock correction. It omits the bottom-camera `top_tongue` used by the reference.

ii.
```python
TONGUE_FEATURES = [(0, 'tongue')]
pos[(view, feat)] = (...)
```

iii. The AI says the side camera follows `params.traj_features`; its trajectory inspected feature names but did not justify excluding the second tongue view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y are linearly interpolated onto the 10 ms grid. Within contiguous non-NaN stretches, `np.gradient` is applied to x and y by sample index (not seconds), and magnitudes are combined. No likelihood-based check, position smoothing, real-time derivative, two-camera scale normalization, or camera averaging is performed.

ii.
```python
pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
vx[seg] = np.gradient(x[seg])
vy[seg] = np.gradient(y[seg])
return np.sqrt(vx ** 2 + vy ** 2), visible
```

iii. The AI claims a port of `findPosition.m`/`findVelocity.m`, assuming coordinates already encode invisibility as NaN.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Across retained trials, all visible/non-NaN session timepoints are median-split: below=0, at/above=1, missing=2.

ii.
```python
thr_tongue = median_of(tongue_speed, tongue_vis)
tongue_cls = discretize(tongue_speed, tongue_vis, thr_tongue)
```

iii. This directly follows the prompt's per-session 50th-percentile threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video offset is computed from bitcode; each frame time becomes `frameTime - videoOffset - goCue`, then positions are interpolated at the neural grid centers.

ii.
```python
return vid_file_offset - bit_start
t_rel = ft - vidshift - align_times[trial]
interp_nan(TAXIS, t_rel, ...)
```

iii. The AI explicitly ports `findVideoOffset.m` and shares `TAXIS` with neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera x/y tracking for both `top_paw` and `bottom_paw`, plus frame/go-cue/bitcode timing. The reference uses only the reliable `top_paw`.

ii.
```python
PAW_FEATURES = [(1, 'top_paw'), (1, 'bottom_paw')]
```

iii. The AI states both paws follow the paper's tracked features, without addressing the poorer second-paw tracking noted by the reference.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Interpolated x/y gaps are nearest-filled, gradients are computed per sample, median frame-to-frame drift is subtracted, speeds are computed, and the two paws are averaged where visible.

ii.
```python
xf, yf = fill_nearest(x), fill_nearest(y)
vx, vy = np.gradient(xf), np.gradient(yf)
vx -= np.nanmedian(np.diff(xf))
paw_speed = np.nanmean(masked, axis=0)
```

iii. The AI cites `findVelocity.m` for fill and drift removal, and chose to combine two paws.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session median over retained, visible bins splits below=0 and at/above=1; neither-paw-visible bins are 2.

ii.
```python
thr_paw = median_of(paw_speed, paw_vis)
paw_cls = discretize(paw_speed, paw_vis, thr_paw)
```

iii. This follows the requested per-session 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are video-offset- and go-cue-corrected, then positions are interpolated to the common 10 ms centers.

ii.
```python
t_rel = ft - vidshift - align_times[trial]
pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
```

iii. The same shared-clock correction and grid are used for every stream.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from each session's separate `motionEnergy_<animal>_<date>.mat` `me.data`, aligned using side-camera frame times.

ii.
```python
md = sio.loadmat(os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))
data = md['me'][0, 0]['data']
```

iii. The AI inspected and cites the authors' `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The precomputed per-frame trace is interpolated to 10 ms centers and then nearest-filled, rather than averaged into bins as in the reference.

ii.
```python
me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
```

iii. The AI treats motion energy as already spatially reduced and only resamples it.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A retained-trial session median divides below=0 and at/above=1; absent/NaN video is class 2.

ii.
```python
thr_me = median_of(me, me_vis)
me_cls = discretize(me, me_vis, thr_me)
```

iii. This follows the prompt's 50th-percentile and no-video categories.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy samples use side-camera frame times after video-offset and go-cue subtraction, then interpolation to the shared grid. Size mismatches cause the whole trial trace to remain missing.

ii.
```python
t_rel = frame_times[trial]
if y.size != t_rel.size: continue
me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
```

iii. The AI assumes one motion-energy value per side-camera frame, consistent with the authors' loader.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Empty/NaN frame times leave video unavailable; outside-support interpolation yields NaN; paw gaps and motion energy are nearest-filled; tongue gaps remain missing; missing values become class 2. Motion-energy/frame-length mismatch discards that trial's trace. It does not support v5 data files or the alternate `autowater` structure needed by randomized sessions.

ii.
```python
if ft.size == 0 or np.all(np.isnan(ft)): continue
if y.size != t_rel.size: continue
out = np.full(values.shape, 2, dtype=np.int8)
```

iii. The AI sought to preserve trials and represent unusable video with the requested third category, while selectively emulating MATLAB nearest-fill behavior.

## 11-a. What are the most time-consuming steps of the code?

i. The AI did not benchmark or explicitly document this. Structurally, repeated HDF5 dereferencing, per-unit spike processing/smoothing, per-trial video interpolation, and construction/serialization of a large nested dataset are likely dominant.

ii.
```python
for iclu in range(quality_refs.size): ...
for trial in range(ntrials): ...
pickle.dump(data, fh, protocol=4)
```

iii. No trajectory justification or timing evidence was provided.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Unit smoothing loops over every trial inside `my_smooth`; video loading and tongue/paw speed loop over trials; assembly loops over retained trials. Some could be vectorized for rectangular arrays, although variable-length HDF5 frame data still requires some looping.

ii.
```python
for j in range(padded.shape[1]):
    out[:, j] = np.convolve(...)
for trial in range(n):
    s, v = speed_from_position(...)
```

iii. The AI did vectorize spike accumulation with `np.add.at`, but did not discuss remaining optimization choices.

## 11-c. What processing does the code repeat multiple times?

i. `speed_from_position` is called separately for every trial and feature; the same time input is copied per trial; per-trial output scalars are broadcast repeatedly. Video offset and raw positions are sensibly computed once per session.

ii.
```python
for i, key in enumerate(PAW_FEATURES):
    for trial in range(n):
        s, v = speed_from_position(...)
input_trials.append(time_input.copy())
```

iii. No explicit trajectory justification was given; the loops reflect trial/feature organization and output-format requirements.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `bp.no` is loaded but unused; `qualities` and detailed counts are retained only to build metadata; all trials' video/kinematics are processed before only retained trials are emitted; both paw traces are processed although the reference uses only `top_paw`; repeated time arrays and trial-constant outputs consume space.

ii.
```python
'no': _vec(bp, 'no') > 0.5,
for trial in range(n): ...
out[0] = lick_dir[trial]
```

iii. The AI did not explicitly discuss discarded processing. Some metadata work was evidently intended for auditability, while processing excluded trials was needed by its session-threshold calculation only insofar as it later masks them out.
