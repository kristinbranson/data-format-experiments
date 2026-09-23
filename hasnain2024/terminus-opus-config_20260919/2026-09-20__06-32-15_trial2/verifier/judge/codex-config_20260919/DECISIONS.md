# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 44 author-selected sessions (25 fixed-delay and 19 randomized-delay), their ALM probe numbers, task folder, and task type. Each session's `data_structure_<animal>_<date>.mat` is loaded with a version-aware v7/v7.3 reader; the separate motion-energy file is loaded during session processing. Sessions are processed in parallel.

ii.
```python
SESSIONS = [
    ('JEB6', '2021-04-18', [2], DATA_FIXED, 'fixed delay'),
    ...
    ('JEB24', '2023-11-03', [1], DATA_RAND, 'randomized delay'),
]
sess = load_session(fn, probes, feats)
me = load_motion_energy(mefn) if os.path.exists(mefn) else None
with Pool(min(args.nproc, len(jobs))) as pool:
    results = pool.map(process_session, jobs)
```

iii. `CONVERSION_NOTES.md` says this list was transcribed from the authors' uncommented `load*_ALMVideo.m` entries, excluding undistributed, duplicate, and no-unit files, and matches the paper's 25 + 19 session totals. Both MATLAB formats occur in the data.

## 1-b. How are the data split into subjects?

i. The animal identifier is taken directly from each hard-coded session tuple. Unique animals are appended in first-session order, and every retained session receives an index into that list.

ii.
```python
if r['anm'] not in data['subjects']:
    data['subjects'].append(r['anm'])
data['subject_idx'].append(data['subjects'].index(r['anm']))
```

iii. The notes treat the filename/session-list animal ID as authoritative and report 14 animals across the 44 sessions.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSIONS`, hence each dated `data_structure` file, is one session and becomes one outer-list element, unless excluded for too few trials or units.

ii.
```python
for r in results:
    if not r.get('ok'):
        continue
    data['neural'].append(r['neural'])
    data['input'].append(r['input'])
    data['output'].append(r['output'])
```

iii. The agent states that the authors' loader scripts define session inclusion and that fixed- and randomized-delay sessions should remain separate sessions in one dataset.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the number of trials. All per-trial behavioral arrays are truncated/indexed to that count; spike `trial` values are converted from one-based to zero-based; one trial matrix is emitted for every retained trial.

ii.
```python
N = int(bp['Ntrials'][0])
keep_trials = np.nonzero(keep)[0]
tr = c['trial'][i].astype(np.int64) - 1
neural = [np.ascontiguousarray(rates[i]).astype(np.float32) for i in range(kt.size)]
```

iii. The notes regard the Bpod table and the spike trial labels as explicit boundaries, so no boundary reconstruction is needed.

## 1-e. How are trials filtered based on quality controls?

i. Trials with photostimulation, early licking, or a nonfinite go cue are removed. After spike processing, any retained trial with zero spikes across all retained pre-filter units and the full window is also removed. Ignore trials are retained. Sessions need at least two retained trials.

ii.
```python
keep = (~stim[:N]) & (~early[:N]) & valid_align
has_spikes = rates.sum(axis=(1, 2)) > 0
rates = rates[has_spikes]
keep_trials = keep_trials[has_spikes]
```

iii. The first two exclusions follow the paper's condition strings. Ignore trials are required output classes. The zero-spike test is justified as detecting behavioral trials after the ephys recording ended.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu{probe}` clusters: per-spike `trial` and `trialtm`, cluster `quality`, and `bp.ev.goCue` for alignment.

ii.
```python
tr = c['trial'][i].astype(np.int64) - 1
tm = c['trialtm'][i]
aligned = tm[sel] - align_times[tr[sel]]
good = [i for i, q in enumerate(c['quality'])
        if q.strip().lower() not in BAD_QUALITY]
```

iii. The notes identify `alignSpikes.m`, `getSeq.m`, and `findClusters.m` as the reference counterparts.

## 2-b. How is the `neural` data processed?

i. Spikes are counted in 10 ms bins, divided by 0.01 s to produce Hz, smoothed with a 15-tap causal Gaussian using reflected leading data, then averaged five-at-a-time into 50 ms decoder bins. Selected probes are concatenated; no normalization or z-scoring is applied.

ii.
```python
rate = counts / DT_FINE
rate = my_smooth(rate)
rates.append(downsample_mean(rate).astype(np.float32))
```

iii. The agent says this mirrors `getSeq.m`/`mySmooth.m`, while 50 ms is documented as a compromise near the reference decoder's 75 ms bins that places the go cue on a bin edge.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Quality labels `garbage`, `gabrga`, `noisy`, and `real?` are rejected case-insensitively; units must have mean retained-window firing rate greater than 1 Hz. A session is rejected if fewer than 10 units remain.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
if rates.shape[1] < MIN_UNITS:
    return dict(ok=False, reason='fewer than %d units' % MIN_UNITS)
```

iii. The notes cite `findClusters.m`, `removeLowFRClusters.m`, the paper's >1 Hz rule, and its minimum-ten-unit session criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time, then binned on the common -2.5 to +2.5 s grid.

ii.
```python
aligned = tm[sel] - align_times[tr[sel]]
b = np.floor((aligned - TMIN) / DT_FINE).astype(np.int64)
```

iii. This is described as matching the authors' `alignSpikes.m` with `params.alignEvent='goCue'` (water drop for WC trials).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Initial resolution is 10 ms; five bins are averaged after smoothing, yielding 100 output bins of 50 ms over five seconds.

ii.
```python
DT_FINE = 0.01
DOWNSAMPLE = 5
DT_OUT = DT_FINE * DOWNSAMPLE
```

iii. The notes call 10 ms a common reference setting and 50 ms a decoder-oriented compromise with the reference's 75 ms decoding aggregation.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the configured go-cue-aligned window and bin grid rather than a separate raw column; raw `bp.ev.goCue` establishes zero for neural/video alignment.

ii.
```python
centres_out = centres[:nout * DOWNSAMPLE].reshape(nout, DOWNSAMPLE).mean(axis=1)
tin = TOUT.astype(np.float32)[None, :]
```

iii. The notes state that the -2.5 to +2.5 s window follows the paper and the input is the common decoder time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Fine bin centers are computed from the edges and groups of five centers are averaged, producing -2.475 through +2.475 s. The same row is copied for each trial.

ii.
```python
centres = edges[:-1] + DT_FINE / 2.0
centres_out = centres.reshape(nout, DOWNSAMPLE).mean(axis=1)
inputs = [tin.copy() for _ in range(kt.size)]
```

iii. The agent reports checks that every trial has the expected range and identical axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact 50 ms groups into which the go-cue-aligned neural rates are averaged.

ii.
```python
rates.append(downsample_mean(rate).astype(np.float32))
tin = TOUT.astype(np.float32)[None, :]
```

iii. The shared construction is intended to guarantee one-to-one time-bin correspondence.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.R`, `bp.L`, `bp.hit`, and `bp.miss`; non-hit/non-miss trials remain “none.”

ii.
```python
right_lick = (R[kt] & hit[kt]) | (L[kt] & miss[kt])
left_lick = (L[kt] & hit[kt]) | (R[kt] & miss[kt])
```

iii. The notes cite `getPrevChoice.m` and an independent check against first lickport contact.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct trials use the instructed side, incorrect trials use its opposite, and ignores are class 2. The per-trial class is repeated at every time point.

ii.
```python
lick_dir = np.full(kt.size, 2, dtype=np.int64)
lick_dir[right_lick] = 1
lick_dir[left_lick] = 0
o[0] = lick_dir[i]
```

iii. The agent says lick direction is not a direct field and this construction matches the reference choice definition.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from `bp.autowater`.

ii.
```python
aw = np.nan_to_num(bp['autowater']).astype(bool)
```

iii. The notes cite the authors' use of autowater as the proxy for WC versus DR blocks.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is mapped to WC (0); all other trials map to DR (1), repeated through time.

ii.
```python
context = np.where(aw[kt], 0, 1).astype(np.int64)
o[1] = context[i]
```

iii. This is described as a direct relabeling consistent with the requested class order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived from `bp.hit` and `bp.miss`; `bp.no` is loaded but the default class handles neither-hit-nor-miss trials.

ii.
```python
hit = np.nan_to_num(bp['hit']).astype(bool)
miss = np.nan_to_num(bp['miss']).astype(bool)
no = np.nan_to_num(bp['no']).astype(bool)
```

iii. The notes say these mutually exclusive flags implement the reference `getOutcome` mapping.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The default is ignore (2), overwritten by correct (1) for hit or incorrect (0) for miss, and repeated through time.

ii.
```python
outcome = np.full(kt.size, 2, dtype=np.int64)
outcome[hit[kt]] = 1
outcome[miss[kt]] = 0
o[2] = outcome[i]
```

iii. Ignore trials are retained specifically because the decoder specification requires that class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera `tongue` x/y coordinates from `obj.traj`, together with side-camera frame times, video/ephys bitcode timing, and each trial's go cue.

ii.
```python
TONGUE_FEAT = ('side', 'tongue')
tongue_sp, _ = feature_speed(sess, TONGUE_FEAT, kt, vidshift, align,
                             fill_missing=False)
```

iii. The notes say tongue missing values must remain missing, following `findPosition.m`; unlike the human solution, the agent did not combine the bottom-camera tongue.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y coordinates are linearly interpolated to the 10 ms grid without filling missing tongue positions. `np.gradient` is applied along samples and the Euclidean magnitude is taken; five samples are then NaN-averaged into each 50 ms output bin.

ii.
```python
pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], left=np.nan, right=np.nan)
vel = np.gradient(pos, axis=0)
sp = np.hypot(vel[:, 0], vel[:, 1])
tongue_ds = downsample_mean(tongue_sp, nanmean=True)
```

iii. The agent claims this mirrors `findPosition.m` plus `findVelocity.m`. It intentionally preserves tongue NaNs as invisibility.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over every finite trial/bin value: below is 0, at/above is 1, and NaN is not-visible class 2.

ii.
```python
thr = np.percentile(values[finite], 50)
cls[finite & (values >= thr)] = 1
cls[finite & (values < thr)] = 0
```

iii. The 50th-percentile threshold and explicit missing class come directly from the decoder specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The session clock offset is computed from bitcode starts. Corrected frame times have that offset and the trial go cue subtracted, then positions are interpolated to the neural fine centers and aggregated to the same 50 ms bins.

ii.
```python
return ft - vidshift - align_t
pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], left=np.nan, right=np.nan)
```

iii. The notes identify this as `findVideoOffset.m`, with an approximately 0.49 s session offset.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses both bottom-camera `top_paw` and `bottom_paw` x/y tracks, their frame times, bitcode sync, and go cue.

ii.
```python
PAW_FEATS = [('bottom', 'top_paw'), ('bottom', 'bottom_paw')]
paw_sps = [feature_speed(sess, k, kt, vidshift, align, fill_missing=True)[0]
           for k in PAW_FEATS]
```

iii. The notes justify both markers by the paper's statement that paws are tracked in the bottom view, though the human solution selects only `top_paw` as reliable.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each marker is interpolated to 10 ms. Missing positions are nearest-filled for differentiation, a baseline derivative is subtracted, speed magnitude is computed, and originally invisible samples are reset to NaN. The two paw speeds are NaN-averaged and then aggregated to 50 ms.

ii.
```python
pos = np.stack([np.interp(idx, idx[vis], pos[vis, d]) for d in range(2)], axis=1)
vel = np.gradient(pos, axis=0)
base = np.nanmedian(np.diff(pos, axis=0), axis=0)
vel = vel - base[0]
paw_sp = np.nanmean(np.stack(paw_sps, axis=0), axis=0)
```

iii. The notes cite the reference's nearest-fill rule for non-tongue features and `findVelocity.m`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Finite pooled session values are split at their median into 0/1; missing bins receive class 2.

ii.
```python
paw_cls, paw_thr = discretize(paw_ds)
```

iii. This follows the requested per-session 50th percentile and not-visible class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are corrected by the common bitcode offset and trial go cue, interpolated to 10 ms neural centers, and averaged into corresponding 50 ms bins.

ii.
```python
ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], left=np.nan, right=np.nan)
```

iii. The agent says all video streams use the same reference synchronization and output grid.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from each session's separate `motionEnergy_<animal>_<date>.mat` `me.data` traces, with side-camera frame times, bitcode sync, and go cues.

ii.
```python
mefn = os.path.join(ddir, 'motionEnergy_%s_%s.mat' % (anm, date))
me = load_motion_energy(mefn) if os.path.exists(mefn) else None
```

iii. The notes identify the standalone files as the reference `loadMotionEnergy.m` source and support several wrapper layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already reduced per-frame trace is linearly interpolated to 10 ms centers, edge/missing gaps are nearest-filled where possible, and groups of five samples are averaged into 50 ms values. No spatial recomputation or smoothing occurs.

ii.
```python
v = np.interp(TCENT, ft[valid], y[valid], left=np.nan, right=np.nan)
v = np.interp(idx, idx[ok], v[ok])
me_ds = downsample_mean(me_fine, nanmean=True)
```

iii. The notes say interpolation and nearest filling mirror `loadMotionEnergy.m`; the upstream file already contains the spatially reduced motion energy.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median of all finite motion-energy bins in a session is used: below 0, at/above 1, and unavailable video 2. The file's manual `moveThresh` is recorded but not used.

ii.
```python
me_cls, me_thr = discretize(me_ds)
me_thresh_session=(None if me is None else me['moveThresh'])
```

iii. The decoder explicitly requires a 50th-percentile session threshold, superseding the paper's manual threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are shifted by the video offset and go cue, interpolated to the same 10 ms centers, and aggregated into the same 50 ms bins as neural data.

ii.
```python
ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
v = np.interp(TCENT, ft[valid], y[valid], left=np.nan, right=np.nan)
```

iii. The notes state that motion energy has one value per side-camera frame, so it uses that camera's clock.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavioral NaNs become false; trials with missing go cues are dropped. Missing frame times are reconstructed at 400 Hz with a 0.5 s pad; length mismatches are truncated. Tongue gaps stay NaN, paw positions are nearest-filled for velocity but original gaps become class 2, motion energy is nearest-filled, and wholly unavailable video becomes class 2. Missing motion-energy files are tolerated.

ii.
```python
if ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, sess['nframes'][trial] + 1) / VIDEO_FS
    return ft - 0.5 - align_t
m = min(ft.size, xy.shape[0])
sp[~vis] = np.nan
```

iii. The notes cite the authors' fallback frame clock and nearest-fill behavior, while preserving explicit not-visible/no-video categories required by the task.

## 11-a. What are the most time-consuming steps of the code?

i. Per-session file loading dominates, followed by neural spike processing and video feature processing; timings for all three are captured. Multiprocessing reduces wall time.

ii.
```python
t_load = time.time() - t0
t_neural = time.time() - t1
t_video = time.time() - t2
```

iii. The notes report roughly 4–5 seconds per sample session and describe loading 100–300 MB MATLAB files as dominant.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over probes, clusters, spikes selected for row mapping, trials for video interpolation, features, and final trial assembly. Spike accumulation and smoothing are partly vectorized. Trial video loops are difficult to eliminate because traces are ragged.

ii.
```python
for i in good:
    ...
    for j in np.nonzero(ok)[0]:
        row[j] = keep_idx[tr[j]]
np.add.at(counts.reshape(-1), flat, 1.0)
for r, t in enumerate(keep_trials):
```

iii. The notes emphasize vectorized spike accumulation/FFT smoothing and parallel sessions; remaining video loops accommodate variable frame counts.

## 11-c. What processing does the code repeat multiple times?

i. `feature_speed` repeats interpolation, filling, and differentiation separately for tongue and two paw markers. The final discretizer repeats for three streams, and identical input arrays are copied per trial. However, video offset and raw session loading occur only once per session.

ii.
```python
tongue_sp, _ = feature_speed(...)
paw_sps = [feature_speed(...) for k in PAW_FEATS]
tongue_cls, tongue_thr = discretize(tongue_ds)
paw_cls, paw_thr = discretize(paw_ds)
me_cls, me_thr = discretize(me_ds)
```

iii. The shared functions make repetition explicit and consistent; thresholds must be computed separately for each session stream.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `bp.no` without using it in classification, computes visibility arrays that callers discard, computes continuous video speeds and manual motion thresholds only to emit categorical values/metadata, and may generate extensive diagnostic plots. It also computes both paw markers although only their average is retained.

ii.
```python
no = np.nan_to_num(bp['no']).astype(bool)
tongue_sp, _ = feature_speed(...)
me_thresh_session=(None if me is None else me['moveThresh'])
```

iii. Most intermediates are necessary to obtain or validate the categorical outputs; diagnostic plotting is optional (`--show-processing`) and the manual threshold is retained for provenance rather than analysis.
