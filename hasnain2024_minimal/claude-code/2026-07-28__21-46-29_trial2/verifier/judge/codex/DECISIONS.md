# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded a 44-session `ALL_SESSIONS` list spanning `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`, with per-session animal, date, folder, and probe assignments. For each session it opens `data_structure_<animal>_<date>.mat`, auto-detects whether it is HDF5/v7.3 or MATLAB v5, and separately opens `motionEnergy_<animal>_<date>.mat`.

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

iii. In the trajectory, the AI said it would include both task folders, use the loading scripts for the session list, skip absent data files, and support both MATLAB formats. It initially considered excluding the first three `JEB15` sessions, then reversed that and decided to include every session explicitly listed in the loading scripts.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the hard-coded session metadata field `anm`. During assembly, the script assigns each previously unseen animal a new integer index in order of first appearance and stores that in `subject_idx`.

ii.
```python
anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])

subj_list = [''] * len(subjects)
for name, idx in subjects.items():
    subj_list[idx] = name
```

iii. The trajectory justification was simply that each loading-script entry already names the animal, so that metadata should define the subject split.

## 1-c. How are the data split into sessions?

i. One entry of `ALL_SESSIONS` is treated as one session. Each such entry points to one `data_structure_...mat` file and one `motionEnergy_...mat` file in its specified folder, and each successful session becomes one element of `neural`, `input`, and `output`.

ii.
```python
for sess in sessions:
    result, err = process_session(sess, data_dir)
    ...
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. The AI justified this by saying it was following the per-animal MATLAB loading scripts and their probe assignments.

## 1-d. How are the data split into trials?

i. Trials are defined by `bp['Ntrials']`, and the script loops over `range(n_trials)`. Spike data are assigned to trials via `spk['trial'] == (ti + 1)`, while trial-level behavioral arrays are truncated to the first `n_trials` entries.

ii.
```python
n_trials = int(bp['Ntrials'][0, 0])
...
for ti in range(n_trials):
    mask = spk['trial'] == (ti + 1)
    ...
trial_idx = np.where(use)[0]
for ti in trial_idx:
    neural_list.append(fr[:, :, ti].copy())
```

iii. In the trajectory, the AI said it would treat the Bpod trial table as authoritative and use the stored per-trial arrays and trial labels directly rather than reconstructing trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI kept only trials with a response (`hit` or `miss`) and then removed early-lick and photostimulation trials. It did not keep ignore/no-response trials, and it did not remove trials that continue after the recording ended except indirectly if an entire session failed later checks.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]

if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. The trajectory says the AI wanted trials with valid lick direction and outcome labels, so it restricted the dataset to `hit|miss` trials and excluded `stim` and `early`. After verification warned about zero-neural late trials, it acknowledged those were likely end-of-recording trials, but it did not change the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `obj.clu` cluster entries, specifically the per-spike `trial` and `trialtm` arrays. Cluster `quality` is used for filtering, and `goCue` is used to align spike times.

ii.
```python
spike_data.append({
    'trial': probe_arr['trial'][0, ci].flatten().astype(float),
    'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
})
...
aligned = spk['trialtm'][mask] - goCue[ti]
```

iii. The AI explicitly said in the trajectory that it would align spikes to go cue, filter clusters by quality, and use the selected probe(s) from the loading scripts.

## 2-b. How is the `neural` data processed?

i. For each unit and trial, spikes are histogrammed into 10 ms bins from -2.5 s to +2.5 s around go cue, divided by bin width to form firing rates, and smoothed with a causal Gaussian-like kernel implemented by `causal_smooth`.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
...
counts, _ = np.histogram(aligned, bins=EDGES)
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The trajectory repeatedly says the AI believed the reference pipeline used 10 ms bins plus causal Gaussian smoothing from the MATLAB code, and it wrote the implementation to match that interpretation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are kept if their `quality` label is non-empty and not in `{'garbage', 'gabrga', 'noisy', 'real?'}`. After binning, units are removed if their mean firing rate across all bins and trials is not above 1 Hz. Sessions with fewer than 10 units before or after rate filtering are skipped.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']
...
mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
...
if n_raw < MIN_UNITS:
    return None, f"Only {n_raw} neurons"
if n_neurons < MIN_UNITS:
    return None, f"Only {n_neurons} neurons after FR filter"
```

iii. The AI justified this as using the reference `findClusters(...,'all')` behavior plus the paper's 1 Hz threshold. It did not mention the human reference's extra exclusion of `Poor` units, and it added session-level minimum-unit filters of its own.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Within each trial, the script subtracts that trial's `goCue` from each spike's `trialtm`, then bins the aligned times.

ii.
```python
mask = spk['trial'] == (ti + 1)
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The trajectory explicitly says the AI intended to align everything to go cue onset and considered this the correct spike alignment rule from the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins over a 5 s window, giving 500 time bins per trial. No later neural rebinning is applied.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
T_BINS = len(TIME_AXIS)
```

iii. The AI repeatedly stated in the trajectory that it thought the reference used 10 ms bins, and the sample verification output it inspected showed `T = 500`, which reinforced that decision.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read from a dedicated raw field. It is constructed from the fixed bin centers of the global aligned time axis around go cue.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The trajectory says the AI treated time-from-go-cue as a constructed decoder input defined by the alignment grid itself.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes the centers of the 10 ms bins spanning -2.5 s to +2.5 s and copies the same 1D vector into every trial as a `(1, T)` array.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. No separate justification was given beyond the AI's belief that this matched the reference's aligned bin grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time series uses the same `TIME_AXIS` that the neural histograms use via `EDGES`, so each input sample corresponds to the center of the same bin used for neural firing rates.

ii.
```python
counts, _ = np.histogram(aligned, bins=EDGES)
...
TIME_AXIS = EDGES[:-1] + DT / 2
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The trajectory justification was that all decoder inputs and outputs should share the go-cue-aligned neural time base.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The script loads `L`, `R`, `hit`, and `miss`, but the final lick-direction output actually uses only `R` after the trial filter has already restricted trials to `hit|miss`.

ii.
```python
L = bp['L'][0, 0].flatten()[:n_trials].astype(int)
R = bp['R'][0, 0].flatten()[:n_trials].astype(int)
hit = bp['hit'][0, 0].flatten()[:n_trials].astype(int)
miss = bp['miss'][0, 0].flatten()[:n_trials].astype(int)
...
out[0, :] = int(R[ti])
```

iii. In the trajectory, the AI said it would map lick direction from the `R` and `L` fields and only keep response trials, so it treated the trial-side label as the lick-direction label.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After filtering to hit/miss trials, the script codes `R==1` as right and `R==0` as left, and then repeats that scalar category across all time bins of the trial. It does not derive lick direction from hit versus miss, and it does not include a no-lick class.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
...
out = np.zeros((6, T_BINS), dtype=np.int64)
out[0, :] = int(R[ti])                  # lick direction: L=0, R=1
```

iii. The trajectory says the AI wanted only trials with valid left/right labels and viewed `R` as identifying the rewarded side the animal licked toward on usable trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `autowater` field in `bp`.

ii.
```python
autowater = bp['autowater'][0, 0].flatten()[:n_trials].astype(int)
...
out[1, :] = 1 - int(autowater[ti])
```

iii. The AI justified this in the trajectory by saying `autowater` distinguishes water-cued versus delayed-response context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script relabels `autowater==1` to WC (`0`) and `autowater==0` to DR (`1`), then repeats that label across time bins within the trial.

ii.
```python
out[1, :] = 1 - int(autowater[ti])       # context: WC=0, DR=1
```

iii. The trajectory explicitly says the AI intended `autowater=0` to mean delayed-response and `autowater=1` to mean water-contingent/water-cued.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` and `miss` trial flags, although the final code only writes `hit` because the trial filter has already removed ignore trials.

ii.
```python
hit = bp['hit'][0, 0].flatten()[:n_trials].astype(int)
miss = bp['miss'][0, 0].flatten()[:n_trials].astype(int)
...
out[2, :] = int(hit[ti])
```

iii. The trajectory says the AI wanted both correct and incorrect response trials and therefore filtered to `hit|miss` before encoding outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script converts outcome into a binary label repeated across all time bins: `1` for hit/correct, `0` for miss/incorrect. Ignore trials are removed rather than encoded as a third category.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
...
out[2, :] = int(hit[ti])                 # outcome: incorrect=0, correct=1
```

iii. The trajectory justifies this as focusing the decoder on trials with explicit behavioral responses and labels.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the bottom-camera DLC feature set in `traj`, specifically `top_tongue` if present, plus its `frameTimes`, the session video shift computed from `bp.ev.bitStart`, `sglx.bitcode.bitstart`, and `sglx.fs`, and the per-trial `goCue`.

ii.
```python
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
...
t_spd = compute_speed(
    ts[:, 0, tongue_fi], ts[:, 1, tongue_fi], ft,
    ts[:, 2, tongue_fi], DLC_CONF, fill_missing=True)
...
aln = ft - vidshift - goCue[ti]
```

iii. The trajectory shows the AI inspecting both camera feature sets and then deciding to use bottom-camera tongue tracking for velocity, even while noting that the paper tracked tongue in both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script computes framewise speed from x/y differences after nearest-filling invalid positions, linearly interpolates the resulting speed trace onto the neural time bins, then nearest-fills missing aligned bins. It does not smooth positions with a Gaussian and does not combine tongue estimates from both cameras.

ii.
```python
def compute_speed(x, y, ft, confidence=None, conf_thresh=DLC_CONF,
                  fill_missing=True):
    valid = ~np.isnan(x) & ~np.isnan(y)
    if confidence is not None:
        valid = valid & (confidence >= conf_thresh)
    if np.any(valid) and not np.all(valid):
        x, y = fill_positions_nearest(x, y, valid)
    elif not np.any(valid):
        return np.zeros(len(x))
    ...
    vx = np.diff(x) / dt_vid
    vy = np.diff(y) / dt_vid
    speed = np.sqrt(vx**2 + vy**2)
```

```python
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The trajectory says the AI thought filling missing positions with nearest values matched the paper's kinematic handling and would produce near-zero velocity when the tongue was not visible. After noticing that full-data median-thresholding made tongue almost always "high", it changed the thresholding rule rather than changing the velocity computation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is thresholded into only two categories, not three. The threshold is the 50th percentile of nonzero tongue speeds from the kept trials of that session; zero-valued bins remain in the low class.

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

iii. In the trajectory, the AI explicitly says it changed from full-data median splitting to median-of-nonzero values because tongue visibility is sparse and the ordinary median collapsed to zero.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The script aligns tongue frames by subtracting a session-wide video offset and the trial's `goCue`, then interpolates the resulting speed trace onto the same `TIME_AXIS` used for neural bins.

ii.
```python
vidshift = bc_bs_mode / fs - bitStart_mode
...
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The trajectory explicitly says the AI measured the offset at about 0.49 s, considered that consistent with a 0.5 s camera shift in the code comments, and therefore used `frameTimes - vidshift - goCue`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DLC tracking, preferring `top_paw` and falling back to `bottom_paw` if needed, together with that trial's camera `frameTimes`, the session `vidshift`, and the trial `goCue`.

ii.
```python
paw_fi = -1
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
...
p_spd = compute_speed(
    ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p,
    ts_p[:, 2, paw_fi], DLC_CONF, fill_missing=True)
```

iii. The trajectory states that paw velocity would come from the bottom camera and especially from `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw velocity uses the same `compute_speed` procedure as tongue velocity: nearest-fill invalid positions, differentiate x and y with respect to frame time, take speed magnitude, linearly interpolate onto the neural time axis, and nearest-fill remaining gaps.

ii.
```python
p_spd = compute_speed(
    ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p,
    ts_p[:, 2, paw_fi], DLC_CONF, fill_missing=True)
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The AI did not add a separate trajectory justification for paw processing; it presented this as parallel to tongue processing from the selected paw feature.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is split into only two classes using the 50th percentile of all aligned paw-velocity samples from the kept trials in that session. There is no explicit not-visible category.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
...
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. The trajectory did not provide a separate justification beyond following the requested per-session median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frames are aligned exactly like tongue frames: subtract session video offset and trial go cue, then interpolate to the shared neural time axis.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The trajectory justification was the same as for tongue alignment: use the measured video offset plus go-cue alignment so all streams share the decoder time base.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is read from the separate `motionEnergy_<session>.mat` file, specifically from `me['data']` after loading. Its alignment also uses side-camera frame times when available, together with `vidshift` and `goCue`.

ii.
```python
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
...
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
```

iii. The trajectory says the AI intended to load motion energy from the companion file rather than recompute it from video.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script treats motion energy as an already-computed per-frame scalar trace, linearly interpolates it onto the neural time axis, and nearest-fills missing aligned bins. If side-camera frame times are missing or length-mismatched, it falls back to a synthetic 400 Hz timeline offset by 0.5 s.

ii.
```python
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
    if len(trial_me) != len(aln):
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
else:
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]

interp_fn = interp1d(aln, trial_me, kind='linear',
                     bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The trajectory justification was that motion energy was already computed in the source data and only needed to be aligned to the decoder time axis.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is split into only two classes using the per-session 50th percentile over all aligned motion-energy samples from the kept trials. There is no explicit no-video category.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
...
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. The trajectory did not provide a special justification beyond applying the requested median split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. When side-camera frame times are available, the script subtracts `vidshift` and the trial's `goCue`, then interpolates motion energy onto `TIME_AXIS`. If frame times are unavailable or mismatched, it substitutes a synthetic 400 Hz time base before interpolation.

ii.
```python
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
    if len(trial_me) != len(aln):
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
else:
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
```

iii. The trajectory justification again relied on the session video-offset estimate and the idea that all streams should land on the same go-cue-centered neural grid.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or low-confidence kinematic samples are filled with nearest valid positions before speed is computed, and missing aligned bins are then filled with nearest valid values after interpolation. If all positions are invalid, velocity becomes all zeros. If video-offset extraction fails, the script uses `0.5` s as a fallback. If motion-energy frame times are missing or mismatched, the script substitutes a synthetic 400 Hz timeline. Many trial-level exceptions are silently ignored with `except Exception: pass`.

ii.
```python
if np.any(valid) and not np.all(valid):
    x, y = fill_positions_nearest(x, y, valid)
elif not np.any(valid):
    return np.zeros(len(x))
...
arr[nans] = f_interp(np.where(nans)[0])
...
except Exception as e:
    vidshift = 0.5
...
except Exception:
    pass
```

iii. The trajectory explicitly argues for nearest-filling missing tongue positions so non-visible periods become near-zero velocity, and it accepted approximate/fallback handling for camera timing when exact data were missing.

## 11-a. What are the most time-consuming steps of the code?

i. The code's heaviest computations are the nested neuron-by-trial spike histogram loop and the per-trial interpolation loops for tongue, paw, and motion-energy traces. Loading large MATLAB/HDF5 files is also substantial, but unlike the reference implementation this script adds a large amount of Python-loop work after loading.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        ...

for ti in range(n_trials):
    ...
    interp_fn = interp1d(aln, t_spd, kind='linear', ...)
    tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The trajectory does not explicitly benchmark these steps, but it does show the AI focusing most of its implementation on spike binning and per-trial camera processing rather than on reducing those loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron-by-trial spike loop could have been vectorized across trials, and the repeated per-trial interpolation for tongue, paw, and motion energy could potentially be replaced by binning/aggregation directly on frame indices. The thresholding and output assembly are already lightweight compared with those loops.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        ...

for ti in range(n_trials):
    ...
    tongue_vel[:, ti] = ...
    paw_vel[:, ti] = ...
...
for ti in range(min(n_trials, me_trials.shape[0])):
    ...
    me_data[:, ti] = ...
```

iii. No explicit trajectory justification was given for leaving these loops unvectorized.

## 11-c. What processing does the code repeat multiple times?

i. The script repeats trial-by-trial alignment and interpolation separately for tongue, paw, and motion energy. It also computes velocities separately for tongue and paw even when the same trial's bottom-camera data have already been loaded. More broadly, it processes all `n_trials` for spikes and videos and only afterwards discards unused trials via `trial_idx`.

ii.
```python
for ti in range(n_trials):
    ...
    tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
    ...
    paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))

for ti in range(min(n_trials, me_trials.shape[0])):
    ...
    me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The trajectory does not acknowledge this repeated work; it treats the per-stream computations as straightforward separate steps.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes neural firing rates, kinematic traces, and motion-energy traces for every trial before subsetting to `trial_idx`, so work on ignored/no-response/early/stim trials is discarded. It also loads and stores raw fields it never uses downstream, such as `no_resp` and `L`, and constructs diagnostic `q_counts` only for printing.

ii.
```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
...
tongue_vel = np.zeros((T_BINS, n_trials), dtype=np.float32)
paw_vel    = np.zeros((T_BINS, n_trials), dtype=np.float32)
...
for ti in trial_idx:
    neural_list.append(fr[:, :, ti].copy())
```

```python
no_resp = bp['no'][0, 0].flatten()[:n_trials].astype(int)
L = bp['L'][0, 0].flatten()[:n_trials].astype(int)
...
q_counts = {}
for q in qualities:
    q_counts[q] = q_counts.get(q, 0) + 1
```

iii. The trajectory does not justify this extra work; it appears to be a consequence of assembling complete per-session arrays first and filtering later.
