# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 44 session dictionaries, including data folder, animal, date, and probe(s). Each `data_structure_*.mat` is auto-detected as HDF5/v7.3 or MATLAB v5 and loaded by a separate loader; a companion `motionEnergy_*.mat` is loaded later with SciPy. Missing session files are skipped.

ii.
```python
ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    # ... 43 more
]
if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
else:
    sd = _load_session_v5(data_path, sess)
```

iii. The trajectory says the session/probe list was taken from the authors' loading scripts, excluding unreferenced or commented-out files. It identified two MATLAB formats and aimed to include all available fixed- and randomized-delay sessions.

## 1-b. How are the data split into subjects?

i. The animal ID is stored explicitly in each session definition. During assembly, first occurrence creates an integer subject index; the result has 14 subjects.

ii.
```python
anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])
```

iii. The trajectory concluded that all available sessions should be included and reported 44 sessions from 14 mice.

## 1-c. How are the data split into sessions?

i. Every entry of `ALL_SESSIONS`, corresponding to one animal/date recording file, becomes one outer-list element in `neural`, `input`, and `output`, unless the file or minimum trial/unit checks cause it to be skipped.

ii.
```python
for sess in sessions:
    result, err = process_session(sess, data_dir)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The AI followed the authors' per-animal session-loading scripts and verified that all 44 selected sessions processed.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` determines the trial count. Behavioral arrays are truncated to that count; spikes use their 1-based `clu.trial` value, and camera/motion arrays are indexed by the same zero-based Python trial index.

ii.
```python
n_trials = int(bp['Ntrials'][0, 0])
hit = bp['hit'][0, 0].flatten()[:n_trials].astype(int)
mask = spk['trial'] == (ti + 1)
```

iii. The trajectory inspected both file layouts and built common per-trial structures so downstream processing could be shared.

## 1-e. How are trials filtered based on quality controls?

i. Only hit or miss trials are retained, and stimulation and early-lick trials are removed. Thus ignore/no-response trials are dropped. Sessions with fewer than two retained trials are skipped. Trials after electrophysiology recording ends are not removed.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]
if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. The AI explicitly claimed this matched `findTrials.m` condition `(hit|miss)&~stim.enable&~early`. Its notes acknowledged all-zero late neural trials in two sessions but kept them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected probes in `obj.clu`: each retained cluster's `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue` for alignment.

ii.
```python
spike_data.append({
    'trial': probe_arr['trial'][0, ci].flatten().astype(float),
    'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
})
aligned = spk['trialtm'][mask] - goCue[ti]
```

iii. The trajectory cites the paper's cluster selection and `alignSpikes.m` as the intended source pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins from -2.5 to +2.5 s, divided by 0.01 to obtain Hz, and smoothed with a custom causal Gaussian window of length 15 using reflected prefix padding. No normalization or baseline subtraction is applied.

ii.
```python
counts, _ = np.histogram(aligned, bins=EDGES)
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The AI believed `params.dt = 1/100` and that zeroing the first half of `gausswin(15)` reproduced `mySmooth.m`; it documented this as matching the paper pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters with empty labels or labels `garbage`, `gabrga`, `noisy`, or `real?` are excluded. Mean rate is then computed across every bin and every raw trial, and units at or below 1 Hz are removed. Sessions below 10 units before or after filtering are skipped; `poor` units remain.

ii.
```python
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']
mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
```

iii. The trajectory interpreted `findClusters.m` “all” mode as excluding those four labels and `removeLowFRClusters.m` as imposing the 1 Hz cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time has that trial's go-cue time subtracted before histogramming.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The AI states this follows `alignSpikes.m` and directly aligns to go-cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 500 bins of 10 ms each over a five-second window. Raw spikes are binned to this grid; video-derived streams are linearly interpolated onto its bin centers.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The AI believed 10 ms was the reference setting and repeatedly described the output as 500 time bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read per trial from a raw field; it is the fixed bin-center grid defined relative to the raw `bp.ev.goCue` alignment event.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The trajectory treats time-from-go-cue as a constructed decoder input shared by every trial.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes centers of adjacent 10 ms histogram edges spanning -2.5 to +2.5 seconds and copies the same `(1, 500)` array into every trial.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The AI chose the same window it believed the paper used.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact edges used to histogram go-cue-aligned spikes, so each input sample corresponds to one neural bin.

ii.
```python
counts, _ = np.histogram(aligned, bins=EDGES)
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The trajectory identifies this common axis as the alignment mechanism.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived solely from `bp.R` for retained hit/miss trials; `bp.L`, `hit`, and `miss` are loaded but not used to infer the actual response direction.

ii.
```python
L, R = sd['L'], sd['R']
out[0, :] = int(R[ti])
```

iii. The AI described this simply as left 0/right 1 and assumed filtering to hit/miss made the instructed side an adequate label.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The binary `R` flag is repeated across all time bins. Miss responses are not inverted, and no “none” class is emitted because ignore trials were removed.

ii.
```python
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The trajectory and README present only two classes, despite the requested left/right/none output.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from the per-trial `bp.autowater` flag.

ii.
```python
autowater = bp['autowater'][0, :n_trials].astype(int)
out[1, :] = 1 - int(autowater[ti])
```

iii. The AI reasoned that autowater identifies WC trials and that sessions without such trials are DR-only.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The flag is inverted so autowater/WC maps to 0 and non-autowater/DR maps to 1, then repeated through time.

ii.
```python
out[1, :] = 1 - int(autowater[ti])
```

iii. This was chosen to match the requested WC=0, DR=1 ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived from `bp.hit`; `miss` and `no` are loaded but retained trials are restricted to hit/miss.

ii.
```python
out[2, :] = int(hit[ti])
```

iii. The AI assumed miss=incorrect and hit=correct and excluded no-response trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit is directly cast to a binary class (miss/incorrect 0, hit/correct 1) and repeated across time. No ignore class is produced.

ii.
```python
out[2, :] = int(hit[ti])
```

iii. Its documentation lists only incorrect and correct, treating ignore trials as removed by the paper-style trial filter.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the bottom-camera `top_tongue` feature (or feature index 0 as a fallback): x, y, likelihood and frame times from `obj.traj`, plus video bitcode, sample rate, behavioral bit start, and go cue for alignment.

ii.
```python
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
t_spd = compute_speed(ts[:, 0, tongue_fi], ts[:, 1, tongue_fi], ft,
                      ts[:, 2, tongue_fi], DLC_CONF, fill_missing=True)
```

iii. The trajectory chose bottom-camera DLC kinematics and did not justify omitting the side-camera tongue beyond implementing the available feature path.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Low-confidence/NaN positions are nearest-filled, x/y first differences are divided by frame intervals, Euclidean speed is computed with an initial zero, then linearly interpolated to neural bin centers and any interpolation NaNs are nearest-filled. There is no positional smoothing or two-view normalization/combination.

ii.
```python
x, y = fill_positions_nearest(x, y, valid)
speed = np.sqrt((np.diff(x) / dt_vid)**2 + (np.diff(y) / dt_vid)**2)
interp_fn = interp1d(aln, t_spd, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The AI asserted nearest filling matched the paper and argued it makes tongue velocity near zero when invisible.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The threshold is the session median of strictly positive values from retained trials, not the median of all visible values. Values below threshold are 0 and values at/above are 1; class 2 is never used.

ii.
```python
tongue_nonzero = tongue_usable[tongue_usable > 0]
tongue_thresh = np.percentile(tongue_nonzero, 50)
out[3, :] = (tongue_vel[:, ti] >= tongue_thresh).astype(np.int64)
```

iii. The AI deliberately excluded zeros because it believed the tongue is visible only 4–8% of frames and wanted a useful median split.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video shift is computed from bitcode modes; each frame is expressed as `frameTime - vidshift - goCue`, then speed is interpolated to the 10 ms neural centers.

ii.
```python
vidshift = bc_bs_mode / fs - bitStart_mode
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The trajectory explicitly cites `findVideoOffset.m` and considered this clock correction a match.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `top_paw`, falling back to `bottom_paw`: x, y, likelihood, frame times, and the same clock/alignment fields as tongue velocity.

ii.
```python
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. The AI sought a usable paw feature and preferred `top_paw`; no detailed feature-reliability justification appears in its trajectory.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw follows the same nearest-fill, first-difference Euclidean speed, linear interpolation, and nearest-fill pipeline as tongue, with no Gaussian position smoothing.

ii.
```python
p_spd = compute_speed(..., DLC_CONF, fill_missing=True)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The AI intended common kinematic processing for both tracked features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A per-session median is computed over all filled values in retained trials. Below median maps to 0 and at/above maps to 1; no not-visible class is emitted.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. The AI followed the requested 50th percentile but treated filled/zero values as observed data.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times receive the session video shift and trial go-cue subtraction, after which speed is interpolated to the neural centers.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, p_spd, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. It reused the video alignment it believed matched `findVideoOffset.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads `me.data` from the standalone `motionEnergy_<animal>_<date>.mat` and uses side-camera `frameTimes`, bitcode-derived video shift, and go cue for timing.

ii.
```python
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
```

iii. The trajectory recognized the companion motion-energy files, but its notes admit several wrapper layouts were not handled.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each already-computed per-frame trace is linearly interpolated onto neural bin centers and extrapolation gaps are nearest-filled. If lengths differ or side times are missing, a synthetic 400 Hz axis starting at -0.5 s is used.

ii.
```python
if len(trial_me) != len(aln):
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The AI considered the trace already spatially processed and only resampled it; fallback timing was a robustness measure.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session median over all retained trial/bin values is used; values below are 0 and values at/above are 1. There is no class 2 for no video, so failed/all-zero loads become all class 1 because the threshold is zero.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. The AI applied the requested percentile but explicitly documented failed sessions becoming all “high.”

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera times are shifted to the behavior clock, aligned by go cue, and interpolated at the same 10 ms centers. Synthetic timing is substituted on missing/mismatched data.

ii.
```python
aln = side_ft[ti] - vidshift - goCue[ti]
interp_fn = interp1d(aln, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The AI intended to mirror camera alignment and added the fallback to prevent conversion failure.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Broad exceptions generally leave preallocated zero arrays. Invalid positions and out-of-range interpolation values are nearest-filled; all-invalid tracking returns zeros. Missing video shift falls back to 0.5 s, missing motion files leave zeros, motion parsing errors leave zeros, and mismatched motion lengths use a synthetic clock. Missing session files or under-sized sessions are skipped.

ii.
```python
except Exception:
    return np.zeros(len(x))
# ...
except Exception as e:
    vidshift = 0.5
# ...
except Exception:
    pass
```

iii. The AI prioritized completing all sessions and noted warnings in conversion notes. It knew some all-zero neural and failed motion-energy data remained, but accepted them because decoder validation still ran above chance.

## 11-a. What are the most time-consuming steps of the code?

i. The code does not instrument timings, but likely costs are loading large MATLAB/HDF5 structures, reading every HDF5 camera trial into memory, nested neuron-by-trial spike histogram/smoothing, and per-trial video interpolation.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The trajectory spent substantial time on full conversion and training but did not provide a formal performance analysis.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested spike loop can be replaced by a two-dimensional histogram per cluster. Some feature extraction and output assembly can also be batched, though ragged video trials still require per-trial handling.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
```

iii. The AI did not discuss vectorization in its final notes; it focused on correctness and cross-format support.

## 11-c. What processing does the code repeat multiple times?

i. It scans each cluster's entire spike-trial vector once for every trial, recomputes interpolation objects per stream/trial, calls `compute_speed` separately for tongue and paw, and for HDF5 paw redundantly retrieves the already-loaded trial tuple. It also copies the identical time axis into every trial.

ii.
```python
mask = spk['trial'] == (ti + 1)
# ...
if ti not in bot_cam['trials_data']:
    continue
ts_p, ft_p = bot_cam['trials_data'][ti]
```

iii. No explicit justification was given; these choices simplify the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `no_resp` and `L` but output construction does not use them, reads side-camera frame times even when motion energy fails, processes all raw trials before selecting retained ones, and creates firing rates for later-discarded trials and neurons. The `fill_missing` argument is accepted by `compute_speed` but ignored.

ii.
```python
no_resp = sd['no_resp']
L, R = sd['L'], sd['R']
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
```

iii. The trajectory did not identify these as waste; processing all trials supported its later global firing-rate calculation and straightforward indexing.
