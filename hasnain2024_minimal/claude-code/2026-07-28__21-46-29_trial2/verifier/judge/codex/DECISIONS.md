# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes a session list in `ALL_SESSIONS`, with one entry per recording day and ALM probe selection. For each entry it builds paths to `data_structure_<animal>_<date>.mat` and `motionEnergy_<animal>_<date>.mat`, auto-detects whether the data structure file is MATLAB v7.3/HDF5 or MATLAB v5, and loads it with either `h5py` or `scipy.io.loadmat`. The full dataset is assembled by iterating over `ALL_SESSIONS` in `convert_all()`.

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

iii. `CONVERSION_NOTES.md` says the session list was taken from the MATLAB loading scripts in `DataLoadingScripts/Recording and video/`, with missing JEB4/JEB5 files excluded and JEB23 `2023-10-20` omitted because it is commented out in the reference loader. The notes also explicitly justify the HDF5 versus v5 auto-detection.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is taken from each session’s `anm` field. `convert_all()` builds a `subjects` dictionary from first occurrence order and stores one `subject_idx` entry per kept session.

ii.
```python
subjects = {}  # name → index

anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])
```

iii. There is no separate written rationale beyond matching the target format’s `subjects` and `subject_idx` fields. The choice is directly implied by the session metadata and the requested output structure.

## 1-c. How are the data split into sessions?

i. Each element of `ALL_SESSIONS` is treated as one session. `process_session()` returns one session-level dictionary, and `convert_all()` appends that session’s `neural`, `input`, `output`, and `brain_region_idx` entries to top-level lists.

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
    all_br_idx.append(result['brain_region_idx'])
```

iii. `CONVERSION_NOTES.md` frames the conversion as session-by-session replication of the reference loading scripts. The trajectory also shows the agent deciding to use the loading-script session boundaries rather than infer sessions from filenames dynamically.

## 1-d. How are the data split into trials?

i. After loading one session, the code computes a Boolean trial mask `use`, extracts `trial_idx = np.where(use)[0]`, and then creates one neural/input/output entry per surviving trial. Trial arrays are stored in per-session Python lists.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]

for ti in trial_idx:
    neural_list.append(fr[:, :, ti].copy())
    input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
    ...
    output_list.append(out)
```

iii. The written rationale is that the decoder format requires session lists of per-trial arrays, and `CONVERSION_NOTES.md` states that the trial inclusion rule was chosen to mirror the MATLAB `findTrials.m` condition used in the reference analyses.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are hit or miss trials, are not stimulation trials, and are not early-lick trials. No-response trials are excluded implicitly because they are neither hits nor misses. Sessions with fewer than two usable trials are skipped.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]

if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. `CONVERSION_NOTES.md` says this was intended to match `findTrials.m` with condition `'(hit|miss)&~stim.enable&~early'`, and explicitly notes that ignore/no-response trials drop out because they are neither hit nor miss.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from per-cluster `trial` and `trialtm` spike fields in `obj.clu`, combined across the ALM probe(s) listed for that session, with `goCue` from `obj.bp.ev.goCue` used for alignment.

ii.
```python
spike_data.append({
    'trial': probe_arr['trial'][0, ci].flatten().astype(float),
    'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
})
...
goCue = ev['goCue'][0, 0].flatten()[:n_trials].astype(float)
```

iii. The notes say the implementation matches the MATLAB loading pipeline built around `obj.clu`, `alignSpikes.m`, and `getSeq.m`. The agent’s trajectory shows it first inspecting the `.mat` object structure to confirm these fields.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, histogrammed into 10 ms bins from `-2.5` to `+2.5` s, converted to firing rates by dividing by `DT`, and smoothed with a causal Gaussian kernel of width `N=15` and reflect padding. The saved per-trial neural matrix has shape `(n_neurons, 500)`.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2

aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. `CONVERSION_NOTES.md` explicitly says this was meant to match `getSeq.m` and `mySmooth.m`: align to `goCue`, bin in 10 ms bins, convert counts to Hz, then apply causal Gaussian smoothing with reflect padding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first filters clusters by quality label, excluding `garbage`, `gabrga`, `noisy`, `real?`, and also excluding empty strings. It then removes neurons with mean firing rate `<= 1 Hz` after smoothing, and skips sessions with fewer than 10 neurons either before or after that firing-rate filter.

ii.
```python
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']

mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
fr = fr[keep]

if n_neurons < MIN_UNITS:
    return None, f"Only {n_neurons} neurons after FR filter"
```

iii. `CONVERSION_NOTES.md` says the quality filter was chosen to match `findClusters.m` with `'all'` and the FR filter to match `removeLowFRClusters.m`, plus the paper’s “at least 10 units per session” inclusion rule. The code’s extra exclusion of empty-string quality labels is not separately justified in the notes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting that trial’s `goCue` value before binning. The converted time axis is centered on go cue and runs from `-2.495` s to `+2.495` s in 10 ms bin centers.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The notes explicitly cite `alignSpikes.m` and say the alignment event was go cue, matching both the instructions and the reference MATLAB processing choice `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`DT = 0.01`) over a fixed 5 s window around go cue, for 500 total bins. There is no second rebinning step after this.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
T_BINS = len(TIME_AXIS)
```

iii. The notes justify this choice by pointing to `WorkingWithDataObjs.m` and `getSeq.m`, where the agent read `params.dt = 1/100` and treated that as the intended analysis bin size.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not computed from a trial-specific raw measurement beyond the choice of go-cue alignment. The script derives it from the fixed analysis window constants `TMIN`, `TMAX`, and `DT`, producing a shared time vector that represents seconds relative to go cue.

ii.
```python
DT = 0.01
TMIN = -2.5
TMAX = 2.5
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. There is no separate written justification in the notes. The choice follows directly from the decoder instruction that the only decoder input should be “Time from go cue onset in seconds.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code creates one 500-bin vector of bin centers relative to go cue and copies that same vector into every kept trial. No interpolation or per-trial transformation is applied.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The trajectory does not show a deeper rationale beyond satisfying the requested input specification with the same time base as the neural data.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is aligned by construction: it uses the exact same `TIME_AXIS` that the spike histograms are binned onto after go-cue subtraction.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. This is an implementation-level consequence of using the common post-alignment bin centers for both neural and input arrays; it is not separately argued in the notes.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is taken from the trial’s right-choice indicator `R`. Because the desired labels are left `= 0` and right `= 1`, the code uses `R[ti]` directly.

ii.
```python
R = sd['R']
...
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The agent inspected `obj.bp.L` and `obj.bp.R` during trajectory exploration and then used the simpler `R` encoding because it already matches the requested binary convention.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The chosen scalar label `int(R[ti])` is repeated across all 500 time bins so that `output` is fully time-varying, even though the underlying variable is really per-trial.

ii.
```python
out = np.zeros((6, T_BINS), dtype=np.int64)
out[0, :] = int(R[ti])
```

iii. There is no specific written rationale besides fitting the decoder’s accepted output shape. The target format allowed either per-trial or time-varying categorical outputs, and the script chose the time-varying form.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `autowater`, which the reference tutorial describes as a proxy for water-cued versus delayed-response blocks.

ii.
```python
autowater = sd['autowater']
...
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. `WorkingWithDataObjs.m` explicitly says `obj.bp.autowater = 1` can be used as a proxy for WC blocks. The agent followed that description and inverted it to satisfy the requested label order `WC = 0`, `DR = 1`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script converts `autowater` into the requested category convention by computing `1 - autowater`, then repeats that trial-level label across time bins.

ii.
```python
out[1, :] = 1 - int(autowater[ti])
```

iii. The transformation is justified implicitly by the decoder task’s label mapping rather than by a separate note in the trajectory.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the raw trial outcome fields `hit` and `miss`, but the actual stored label uses only `hit` because the trial filter has already restricted trials to hit or miss.

ii.
```python
hit, miss = sd['hit'], sd['miss']
...
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. The notes justify the hit/miss filtering, and once that filter is in place `hit` already has the requested encoding `correct = 1`, `incorrect = 0`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code converts each kept trial to a binary scalar `int(hit[ti])` and repeats it across all time bins.

ii.
```python
out[2, :] = int(hit[ti])
```

iii. As with lick direction and context, the script uses a time-varying representation for a per-trial variable because that is accepted by the target format.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the bottom-camera DeepLabCut trajectory for the single feature `top_tongue`, using its x position, y position, confidence score, frame times, the session video offset, and each trial’s go cue.

ii.
```python
feat_names = bot_cam['feat_names']
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
...
t_spd = compute_speed(
    ts[:, 0, tongue_fi], ts[:, 1, tongue_fi], ft,
    ts[:, 2, tongue_fi], DLC_CONF, fill_missing=True)
```

iii. `CONVERSION_NOTES.md` says the agent intentionally used the bottom-camera `top_tongue` feature. The trajectory shows that the tongue implementation changed during debugging, with the final version justified as giving near-zero speed when the tongue is not visible.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code applies a DLC confidence threshold of `0.9`, fills low-confidence or NaN tongue positions with the nearest valid position, computes frame-by-frame Euclidean speed `sqrt(vx^2 + vy^2)`, aligns it using `frameTimes - vidshift - goCue`, linearly interpolates to the neural time axis, and fills residual NaNs with nearest values.

ii.
```python
valid = ~np.isnan(x) & ~np.isnan(y)
if confidence is not None:
    valid = valid & (confidence >= conf_thresh)

if np.any(valid) and not np.all(valid):
    x, y = fill_positions_nearest(x, y, valid)

vx = np.diff(x) / dt_vid
vy = np.diff(y) / dt_vid
speed = np.sqrt(vx**2 + vy**2)
...
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. `CONVERSION_NOTES.md` claims this matches the paper’s processing for kinematic features, and explains the final choice as making the tongue near-zero during non-licking periods. The trajectory shows this was a deliberate late change: the agent edited the code to use `fill_missing=True` for tongue.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The threshold is the per-session median of non-zero tongue-velocity values from usable trials only. Bins at zero stay in the “low” class automatically; bins at or above the threshold become “high”.

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

iii. `CONVERSION_NOTES.md` explicitly justifies the non-zero median by saying the tongue is visible only during a small fraction of frames, so including zeros would collapse the threshold near zero. The trajectory shows this thresholding rule was added after the initial implementation.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue speed is aligned by converting camera frame times to the ephys clock with `vidshift`, subtracting each trial’s `goCue`, and interpolating the resulting trace to the common neural `TIME_AXIS`.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. `CONVERSION_NOTES.md` ties this directly to the reference `findVideoOffset.m` formula and to the requirement that all streams be aligned to go cue.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DLC trajectory for `top_paw` if present, otherwise `bottom_paw`, together with x/y coordinates, confidence, frame times, `vidshift`, and `goCue`.

ii.
```python
paw_fi = -1
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. `CONVERSION_NOTES.md` says the bottom camera was used for paw features and that `top_paw`/`bottom_paw` were the extracted candidates. There is no deeper written justification for choosing one paw feature rather than combining them.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw processing mirrors tongue processing except it uses the chosen paw feature. The code applies a confidence filter, fills invalid positions with the nearest valid position, computes Euclidean speed, aligns to go cue, interpolates to `TIME_AXIS`, and fills remaining NaNs with nearest values.

ii.
```python
p_spd = compute_speed(
    ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p,
    ts_p[:, 2, paw_fi], DLC_CONF, fill_missing=True)
...
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The notes say this was intended to follow the paper’s general rule that non-tongue missing values are filled with the nearest available position before velocity is computed.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded with the per-session 50th percentile over all paw-velocity bins from usable trials. Values below the threshold are “low”; values at or above are “high”.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
...
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. This follows the decoder instruction rather than a separate note from the reference code, which did not define a binary paw-velocity decoder target.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity is aligned in the same way as tongue velocity: `frameTimes - vidshift - goCue`, then linear interpolation to the common neural time axis.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The notes tie this to the same video-offset alignment logic used for the neural and tongue streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate session file `motionEnergy_<animal>_<date>.mat`, specifically from `me.data`, and aligned using side-camera frame times when available plus `vidshift` and `goCue`.

ii.
```python
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
...
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
```

iii. `CONVERSION_NOTES.md` says this was based on the reference `loadMotionEnergy.m` logic and on the fact that each ephys session ships with a separate motion-energy file.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the code loads the raw motion-energy trace, builds an aligned time vector from side-camera frame times if available or a 400 Hz synthetic grid otherwise, linearly interpolates the trace to `TIME_AXIS`, and fills residual NaNs with nearest values. If loading fails, the session’s motion energy stays as all zeros.

ii.
```python
me_data = np.zeros((T_BINS, n_trials), dtype=np.float32)

me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
...
interp_fn = interp1d(aln, trial_me, kind='linear',
                     bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
...
except Exception as e:
    print(f"  WARNING: motion energy: {e}")
```

iii. `CONVERSION_NOTES.md` says the intended design was to match `loadMotionEnergy.m`, but it also documents that several sessions fell into exception paths and were left with all-zero motion energy because the agent’s loader did not handle all file layouts correctly.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized with the per-session 50th percentile over all aligned bins from usable trials. Values below the threshold are “low”; values at or above are “high”.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
...
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. This choice comes from the decoder task, not from the original paper. `CONVERSION_NOTES.md` contrasts it with the paper’s manual bimodal threshold and treats the 50th percentile as the required decoder-specific change.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the same go-cue-centered time base as the neural data. When side-camera frame times are available, alignment is `side_frame_times - vidshift - goCue`; when they are missing or length-mismatched, the code falls back to a 400 Hz grid with a hard-coded `0.5 s` offset.

ii.
```python
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
    if len(trial_me) != len(aln):
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
else:
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
```

iii. The notes explicitly justify this with the reference `findVideoOffset.m` and `loadMotionEnergy.m` fallback behavior, which use either recorded frame times or a default 400 Hz/0.5 s camera offset.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses broad fallbacks rather than failing fast. Missing or malformed video offsets default to `0.5 s`; all-NaN trajectories return zero speed; missing low-confidence positions are nearest-filled; missing trial camera data are skipped; motion-energy load failures leave all-zero motion energy; length mismatches fall back to synthetic 400 Hz frame times; sessions with too few trials or neurons are skipped entirely.

ii.
```python
except Exception as e:
    vidshift = 0.5
...
elif not np.any(valid):
    return np.zeros(len(x))
...
except Exception:
    pass
...
except Exception as e:
    print(f"  WARNING: motion energy: {e}")
```

iii. `CONVERSION_NOTES.md` documents these fallbacks as sanity-preserving choices, especially for motion-energy failures and all-zero late-session neural data. The trajectory does not show the agent attempting a stricter recovery strategy once these warnings appeared.

## 11-a. What are the most time-consuming steps of the code?

i. The dominant cost is the nested neuron-by-trial spike binning and smoothing loop, which builds a full `(n_neurons, 500, n_trials)` array before trial filtering. The next largest cost is the per-trial video interpolation for tongue, paw, and motion energy.

ii.
```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        ...
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The agent did not document a performance analysis explicitly. This conclusion follows directly from the code structure and array sizes.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop over every neuron and every trial could be vectorized or at least grouped by trial indices. The DLC and motion-energy sections also repeat nearly identical per-trial interpolation work that could be batched once trajectories are in arrays.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        ...

for ti in range(n_trials):
    ...
    tongue_vel[:, ti] = ...
    ...
    paw_vel[:, ti] = ...
```

iii. No explicit vectorization rationale appears in the notes or trajectory; this is an inference from the implementation.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats nearly the same load logic twice for MATLAB v5 versus HDF5 files, computes tongue and paw speed with almost identical code paths, and separately re-runs interpolation and NaN filling for tongue, paw, and motion energy on every trial.

ii.
```python
if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
else:
    sd = _load_session_v5(data_path, sess)
...
t_spd = compute_speed(...)
...
p_spd = compute_speed(...)
...
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The agent did not give a written reason for keeping these repeated code paths separate. The repetition is visible from the script organization.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes neural, tongue, paw, and motion-energy traces for all trials before discarding excluded trials with `trial_idx`. It also expands trial-constant labels (`lick_direction`, `behavioral_context`, `outcome`) into full 500-bin time series even though the decoder format would also accept per-trial categorical outputs.

ii.
```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
...
tongue_vel = np.zeros((T_BINS, n_trials), dtype=np.float32)
paw_vel    = np.zeros((T_BINS, n_trials), dtype=np.float32)
...
for ti in trial_idx:
    out[0, :] = int(R[ti])
    out[1, :] = 1 - int(autowater[ti])
    out[2, :] = int(hit[ti])
```

iii. There is no explicit justification for these extra computations in the notes. They appear to be convenience choices rather than requirements of the downstream decoder.
