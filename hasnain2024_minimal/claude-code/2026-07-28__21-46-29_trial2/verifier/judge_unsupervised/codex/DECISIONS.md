# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hardcodes a session manifest in `ALL_SESSIONS` covering `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`, then loads one `data_structure_<animal>_<date>.mat` file and one `motionEnergy_<animal>_<date>.mat` file per session. It auto-detects MATLAB v7.3/HDF5 versus v5 `.mat` files and routes them through separate loaders.

ii. 
```python
ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    ...
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-11-03', 'probes': [1]},
]

data_path = os.path.join(data_dir, sess['dir'],
                         f"data_structure_{sname}.mat")
me_path = os.path.join(data_dir, sess['dir'],
                       f"motionEnergy_{sname}.mat")

if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
else:
    sd = _load_session_v5(data_path, sess)
```

iii. In `CONVERSION_NOTES.md`, the agent says sessions were identified from the MATLAB loading scripts, with `JEB23 2023-10-20` and `JEB24 2023-10-03/04` excluded because they were not active in those scripts. The trajectory also shows the agent manually reconciled commented-out sessions and missing files.

## 1-b. How are the data split into subjects?

i. Sessions are assigned to subjects by the animal ID string in each session record (`sess['anm']`). A `subjects` dictionary assigns each unique animal name a consecutive integer index, and `subject_idx` stores that index once per retained session.

ii.
```python
anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])
...
subj_list = [''] * len(subjects)
for name, idx in subjects.items():
    subj_list[idx] = name
```

iii. The justification is implicit: the session manifest already carries the animal ID, and `CONVERSION_NOTES.md` organizes the included sessions by mouse.

## 1-c. How are the data split into sessions?

i. Each entry in `ALL_SESSIONS` becomes one candidate session. `convert_all()` calls `process_session()` once per entry, and each successful return contributes exactly one session-level element to `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
for sess in sessions:
    result, err = process_session(sess, data_dir)
    if result is None:
        print(f"  SKIPPED: {err}")
        continue

    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. The notes justify the session boundaries as coming from the paper repository’s per-animal loading scripts, including explicit probe assignments per date.

## 1-d. How are the data split into trials?

i. Inside each loaded session, all behavioral trials are first read from the raw `bp` structure. After applying a trial-use mask, each retained trial index `ti` contributes one neural matrix, one input matrix, and one output matrix that are appended to per-session Python lists.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]
...
for ti in trial_idx:
    neural_list.append(fr[:, :, ti].copy())
    input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
    ...
    output_list.append(out)
```

iii. In `CONVERSION_NOTES.md`, the agent explicitly says this matches the MATLAB `findTrials.m` condition `'(hit|miss)&~stim.enable&~early'`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three trial-level criteria: they must be `hit` or `miss`, they must not be stimulation trials (`stim_en == 0`), and they must not be early-lick trials (`early == 0`). Sessions are dropped entirely if fewer than two usable trials remain.

ii.
```python
MIN_TRIALS = 2
...
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]

if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. The notes justify this by citing `findTrials.m`; they also note that no-response trials are excluded automatically because they are neither `hit` nor `miss`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived from spike-cluster records in `obj.clu`, specifically each cluster’s per-spike trial assignment (`trial`) and within-trial spike times (`trialtm`), together with `bp.ev.goCue` for alignment.

ii.
```python
spike_data.append({
    'trial': probe_arr['trial'][0, ci].flatten().astype(float),
    'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
})
...
goCue = ev['goCue'][0, 0].flatten()[:n_trials].astype(float)
```

iii. `CONVERSION_NOTES.md` says the neural stream matches `alignSpikes.m`, `getSeq.m`, and `removeLowFRClusters.m`.

## 2-b. How is the `neural` data processed?

i. For each neuron and trial, the code subtracts the trial’s go-cue time from each spike time, bins aligned spikes into 10 ms bins from -2.5 s to +2.5 s, converts counts to firing rate by dividing by `DT`, and applies causal Gaussian smoothing with a 15-bin window and reflect-style padding.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The agent’s notes explicitly claim this mirrors `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first filters clusters by quality label, excluding `garbage`, `gabrga`, `noisy`, and `real?`. After binning and smoothing, it removes neurons whose mean firing rate across all bins and trials is `<= 1 Hz`. Finally, it rejects any session with fewer than 10 neurons before or after the FR filter.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']
...
mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
fr = fr[keep]
...
if n_neurons < MIN_UNITS:
    return None, f"Only {n_neurons} neurons after FR filter"
```

iii. `CONVERSION_NOTES.md` cites `findClusters.m` and `removeLowFRClusters.m`, and also cites the paper’s “at least 10 units per session” rule for the session-level minimum.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial’s spike times are aligned to go-cue onset by subtracting the raw `goCue[ti]` from each spike’s within-trial timestamp before binning.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The notes explicitly state “Align spike times: `trialtm - goCue`,” referencing `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins. There is no secondary rebinning; the one binning step directly produces the stored time axis and trial matrices.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The justification in the notes is that this matches `params.dt = 1/100` from the MATLAB workflow.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the choice to align everything to `goCue`; the stored input itself is not taken from a raw signal array, but from the fixed relative-time axis built around the go-cue alignment window.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The justification is implicit in the instructions and the script constants: the input is defined as time from go cue, so the agent materializes that as the common aligned bin-center axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs equally spaced bin centers from the shared bin edges spanning -2.5 s to +2.5 s around go cue, then copies that same `1 x T` array into every retained trial.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. There is no separate justification in the notes beyond the statement that decoder input is “time from go cue.”

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same aligned axis used for neural binning. Neural spikes are binned using `EDGES`, and the stored input uses the matching `TIME_AXIS` bin centers.

ii.
```python
counts, _ = np.histogram(aligned, bins=EDGES)
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The justification is implicit: the script uses one shared window and one shared time axis for all aligned modalities.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the raw trial-type flags `bp.L` and `bp.R`, but the stored value actually uses only the right-trial flag `R`, with left implicitly encoded as `0`.

ii.
```python
L, R = sd['L'], sd['R']
...
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The notes justify the label mapping as “left = 0, right = 1,” which matches the decoder task.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is no additional signal processing. The per-trial scalar category is copied across all time bins of the retained trial.

ii.
```python
out = np.zeros((6, T_BINS), dtype=np.int64)
out[0, :] = int(R[ti])
```

iii. The justification is implicit: the decoder task asked for a per-trial variable, and the agent chose to make all outputs time-varying where possible by repeating trial labels across time.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`, used as a proxy for WC versus DR blocks.

ii.
```python
autowater = sd['autowater']
...
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. The MATLAB tutorial in `WorkingWithDataObjs.m` says `obj.bp.autowater=1` can be used as a proxy for WC versus DR blocks, and the notes repeat that mapping.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The raw `autowater` bit is inverted so that WC maps to `0` and DR maps to `1`, then repeated across all time bins of the trial.

ii.
```python
out[1, :] = 1 - int(autowater[ti])
```

iii. The justification is explicit in the notes: the decoder spec required `WC = 0, DR = 1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial result flags `bp.hit` and `bp.miss`, but the stored value uses `hit` alone because the earlier trial filter already restricts retained trials to `hit` or `miss`.

ii.
```python
hit, miss = sd['hit'], sd['miss']
...
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. The notes justify this as “incorrect = 0, correct = 1,” with miss implicitly providing the `0` class after filtering.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. There is no further processing beyond converting `hit` to an integer class and repeating it across the time axis.

ii.
```python
out[2, :] = int(hit[ti])
```

iii. The justification is simply the decoder task’s requested binary outcome variable.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from bottom-camera DeepLabCut coordinates and confidence values for the `top_tongue` feature, the bottom-camera frame times, the video offset, and each trial’s go-cue time.

ii.
```python
feat_names = bot_cam['feat_names']
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
...
t_spd = compute_speed(
    ts[:, 0, tongue_fi], ts[:, 1, tongue_fi], ft,
    ts[:, 2, tongue_fi], DLC_CONF, fill_missing=True)
...
aln = ft - vidshift - goCue[ti]
```

iii. The notes justify the feature choice as “extract (x, y, confidence) for `top_tongue`,” and the trajectory shows the agent chose bottom-camera tongue tracking after inspecting the DLC feature list.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code treats low-confidence samples (`< 0.9`) as invalid, fills invalid tongue positions with the nearest valid position, computes instantaneous speed as the Euclidean norm of frame-to-frame velocity, interpolates that speed onto the neural time axis, and fills resulting NaNs by nearest interpolation.

ii.
```python
valid = ~np.isnan(x) & ~np.isnan(y)
if confidence is not None:
    valid = valid & (confidence >= conf_thresh)
...
if np.any(valid) and not np.all(valid):
    x, y = fill_positions_nearest(x, y, valid)
...
vx = np.diff(x) / dt_vid
vy = np.diff(y) / dt_vid
speed = np.sqrt(vx**2 + vy**2)
...
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The notes justify this by saying tongue is only visible during licking and that nearest filling yields near-zero speed when it is not visible. The trajectory shows the agent explicitly reasoned about this as a way to make tongue velocity usable for the decoder.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The script does not use the raw session median over all time bins. Instead, it takes the 50th percentile of only nonzero tongue-speed samples from retained trials, then labels bins below that value as `0` and bins at or above it as `1`. If no nonzero samples exist, it sets the threshold to `1.0`, making all bins low.

ii.
```python
tongue_usable = tongue_vel[:, trial_idx].flatten()
tongue_nonzero = tongue_usable[tongue_usable > 0]
if len(tongue_nonzero) > 0:
    tongue_thresh = np.percentile(tongue_nonzero, 50)
else:
    tongue_thresh = 1.0
...
out[3, :] = (tongue_vel[:, ti] >= tongue_thresh).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the nonzero-only threshold by arguing that the tongue is visible in only ~4-8% of frames and that zero-speed bins should automatically count as low.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue speed is aligned by converting bottom-camera frame times into ephys-relative time via `vidshift`, subtracting each trial’s `goCue`, then linearly interpolating onto `TIME_AXIS`, which is the same axis used for neural binning.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The notes say this matches the paper code’s video-offset computation and go-cue alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DeepLabCut `top_paw` or `bottom_paw` coordinates and confidence values, the same bottom-camera frame times, `vidshift`, and `goCue`.

ii.
```python
paw_fi = -1
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. The notes justify this from the methods statement that paws are tracked only from the bottom view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Low-confidence or NaN paw positions are filled by nearest valid position, instantaneous speed is computed from coordinate differences over frame-time differences, and the resulting speed trace is interpolated to the neural time axis.

ii.
```python
p_spd = compute_speed(
    ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p,
    ts_p[:, 2, paw_fi], DLC_CONF, fill_missing=True)
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The notes explicitly say paw velocity follows the “fill invalid positions, compute speed, align by video offset, interpolate” pipeline.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The threshold is the session-level 50th percentile of all retained-trial paw-velocity bins. Bins below threshold are `0`, bins at or above threshold are `1`.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
...
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. The notes justify this as the standard per-session median split requested by the decoder task.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same bottom-camera alignment procedure as tongue velocity: `frameTimes - vidshift - goCue`, followed by interpolation to `TIME_AXIS`.

ii.
```python
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The notes say all DLC-derived kinematic traces are aligned using video offset and the go cue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate per-session `motionEnergy_<animal>_<date>.mat` file, specifically `me.data`, together with side-camera frame times from `obj.traj`, the video offset, and the trial go-cue times.

ii.
```python
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
...
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
```

iii. The notes justify this as mirroring `loadMotionEnergy.m`, except that the final decoder output uses discretized motion-energy magnitude rather than the repository’s stored `me.move`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the code loads the continuous motion-energy trace, aligns it to go cue using side-camera frame times and `vidshift`, linearly interpolates it to the neural time axis, and fills missing interpolated values by nearest interpolation. If side-camera frame times are missing or length-mismatched, it falls back to a synthetic 400 Hz time base offset by `-0.5 s`.

ii.
```python
trial_me = me_trials[ti, 0].flatten().astype(np.float64)
...
if len(trial_me) != len(aln):
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
...
interp_fn = interp1d(aln, trial_me, kind='linear',
                     bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The notes justify the general alignment method by citing `loadMotionEnergy.m`; they separately admit that some sessions fell back to all-zero motion energy because the file layout was not handled.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The script computes the session-level 50th percentile of all retained-trial motion-energy bins and thresholds each time bin around that value.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
...
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. The justification is the decoder task itself, which required a per-session 50th-percentile discretization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using side-camera frame times shifted into the ephys clock with `vidshift`, then shifted again by each trial’s go cue, and interpolated to the same `TIME_AXIS` used for neural activity.

ii.
```python
aln = side_ft[ti] - vidshift - goCue[ti]
interp_fn = interp1d(aln, trial_me, kind='linear',
                     bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. `CONVERSION_NOTES.md` explicitly says motion energy is “aligned using side camera frame times and video offset” and “interpolated to neural time axis.”

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses broad fallback behavior rather than explicit repair or exclusion. Missing video offsets fall back to `0.5 s`; missing or low-confidence kinematic samples are nearest-filled or converted to all-zero speed; interpolation NaNs are nearest-filled; motion-energy load failures leave an all-zero `me_data`; missing motion-energy files also leave zeros; and late trials with all-zero neural data are kept in the dataset.

ii.
```python
except Exception as e:
    vidshift = 0.5
...
elif not np.any(valid):
    return np.zeros(len(x))
...
arr[nans] = f_interp(np.where(nans)[0])
...
except Exception as e:
    print(f"  WARNING: motion energy: {e}")
...
me_data = np.zeros((T_BINS, n_trials), dtype=np.float32)
```

iii. The notes openly justify these as pragmatic fallbacks: they document motion-energy failures, all-zero neural trials in two JEB24 sessions, and the decision to let those trials remain.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are the nested neuron-by-trial spike binning/smoothing loop, the trial-by-trial kinematic interpolation loops, and the full-session loading of large MATLAB/HDF5 objects. Those dominate runtime and memory.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        ...
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
...
for ti in range(n_trials):
    ...
    tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. There is no explicit defense beyond the desire to mirror MATLAB processing; the trajectory shows the agent prioritized getting a working end-to-end conversion over optimizing it.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron/per-trial spike histogram loop, the per-trial tongue and paw velocity interpolation loops, the per-trial motion-energy interpolation loop, and some quality-label parsing loops could all have been vectorized or batch-processed more aggressively.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        ...

for ti in range(n_trials):
    ...
    tongue_vel[:, ti] = ...
    paw_vel[:, ti] = ...

for ti in range(min(n_trials, me_trials.shape[0])):
    ...
    me_data[:, ti] = ...
```

iii. No explicit justification is given beyond code simplicity and fidelity to the MATLAB reference structure.

## 11-c. What processing does the code repeat multiple times?

i. It repeats almost identical loader logic for HDF5 and v5 formats, repeats the same interpolation pattern for tongue, paw, and motion energy, and repeatedly copies the same `TIME_AXIS` input into every retained trial. It also repeatedly scans all clusters and all trials even for sessions later skipped.

ii.
```python
if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
else:
    sd = _load_session_v5(data_path, sess)
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The trajectory suggests the duplication came from progressively adding support for both MATLAB formats rather than from a planned abstraction.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes neural matrices for all trials before discarding non-usable trials via `trial_idx`; it processes motion-energy and kinematic traces for all trials even though only retained trials are saved; it loads `no_resp`, `L`, and some other raw variables that are not all used directly in the final arrays; and it preserves known all-zero neural trials that add no useful information.

ii.
```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        ...
for ti in range(n_trials):
    ...
for ti in trial_idx:
    neural_list.append(fr[:, :, ti].copy())
```

iii. The agent’s notes acknowledge at least one such downstream-irrelevant case explicitly: all-zero neural trials in sessions 36 and 43 were left in place even though they “contribute no information.”
