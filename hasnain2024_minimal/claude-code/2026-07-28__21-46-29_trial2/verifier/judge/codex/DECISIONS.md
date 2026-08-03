# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 44-session manifest in `ALL_SESSIONS`, with each entry naming the folder, animal, date, and probe list. For each session it opens `data_structure_<anm>_<date>.mat`, auto-detects whether the MATLAB file is HDF5/v7.3 or v5, and dispatches to separate loaders. Motion energy is loaded from a separate `motionEnergy_<anm>_<date>.mat` file in the same session folder. Trial-level behavioral arrays, cluster spike fields, bottom-camera tracking, and side-camera frame times are all pulled into one per-session dictionary before later processing.

ii.
```python
ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    ...
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-11-03', 'probes': [1]},
]

if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
else:
    sd = _load_session_v5(data_path, sess)
```

```python
data_path = os.path.join(data_dir, sess['dir'],
                         f"data_structure_{sname}.mat")
me_path = os.path.join(data_dir, sess['dir'],
                       f"motionEnergy_{sname}.mat")
```

iii. The justification is explicit in `CONVERSION_NOTES.md`: the session list was taken from the authors’ MATLAB loading scripts, JEB4/JEB5 were unavailable, and the code needed to support both MATLAB v7.3 and v5 files. The trajectory also shows the AI intentionally decided to include both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` sessions from those loading scripts.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken directly from each hard-coded session record’s `anm` field. During assembly, the AI creates one `subjects` list in first-seen order and one `subject_idx` entry per retained session.

ii.
```python
subjects = {}  # name → index
...
anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])
```

```python
subj_list = [''] * len(subjects)
for name, idx in subjects.items():
    subj_list[idx] = name
```

iii. The justification is implicit rather than argued in detail: because the session manifest already stores `anm`, the AI reuses that instead of re-reading subject metadata from inside the files.

## 1-c. How are the data split into sessions?

i. Each dictionary in `ALL_SESSIONS` is treated as one session. `process_session` turns each selected session into one element of `neural`, `input`, and `output`, and keeps the original top-level split between `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` only as metadata in the file path and session info.

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

iii. The justification comes from `CONVERSION_NOTES.md`: the AI states that the loading scripts define which sessions and probes to use, so it mirrors that session granularity.

## 1-d. How are the data split into trials?

i. Trials are indexed by array position from `0` to `n_trials - 1` using the behavioral arrays read from `bp`. The later `trial_idx` mask picks the kept trials. Spike times are assigned to trials using `spk['trial'] == (ti + 1)`, and video/motion signals are processed one trial index at a time.

ii.
```python
n_trials = sd['n_trials']
...
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]
```

```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
```

iii. The AI does not justify this separately, but the implementation assumes the Bpod trial arrays are authoritative and that the cluster `trial` field already gives per-spike trial membership.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only hit or miss trials and excludes photostimulation and early-lick trials. This removes all no-response/ignore trials from the dataset. It does not remove trials after the ephys recording ended; `CONVERSION_NOTES.md` instead acknowledges that some late trials have all-zero neural data and were left in place.

ii.
```python
# trial filter: (hit|miss) & ~stim & ~early
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]
```

```markdown
### Trial Filter
Matches `findTrials.m` condition `'(hit|miss)&~stim.enable&~early'`:
- Include hit and miss trials
- Exclude stimulation trials (`stim.enable == 1`)
- Exclude early-lick trials (`early == 1`)
- No-response trials are also excluded (they are neither hit nor miss)
```

iii. The justification is explicit in the notes: the AI believed `findTrials.m` should be matched exactly, so it excluded no-response trials. It separately noted but did not fix late all-zero-neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices come from cluster spike fields `trial` and `trialtm` taken from `obj['clu']` for the selected probes, together with per-trial `goCue` times from `bp.ev.goCue` for alignment. Cluster quality labels are used earlier to decide which clusters enter `spike_data`.

ii.
```python
for ci in valid_clu:
    spike_data.append({
        'trial': probe_arr['trial'][0, ci].flatten().astype(float),
        'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
    })
```

```python
goCue = ev['goCue'][0, 0].flatten()[:n_trials].astype(float)
```

iii. The justification is implicit in the code and notes: spike times are already stored per trial, so the AI only needed trial id, within-trial spike time, and go-cue time.

## 2-b. How is the `neural` data processed?

i. For each neuron and trial, the AI subtracts the trial go cue from each spike time, bins spikes into a fixed time grid, converts counts to firing rate by dividing by `DT`, and then smooths each per-trial firing-rate trace with a causal Gaussian kernel implemented by `gausswin` plus one-sided zeroing and convolution.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

```python
def causal_smooth(x, N=SMOOTH_N, bctype=BC_TYPE):
    kern = gausswin(N)
    kern[:N // 2] = 0
    kern /= kern.sum()
    ...
    out = np.convolve(x_padded, kern, mode='same')
    return out[trim:]
```

iii. The justification is explicit in `CONVERSION_NOTES.md` and the trajectory: the AI believed the reference used `WorkingWithDataObjs.m`, 10 ms bins, and a causal version of `mySmooth.m`, so it tried to reproduce that path.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first removes clusters whose quality label is `garbage`, `gabrga`, `noisy`, or `real?`, and also drops empty-string quality labels. It then removes units with mean firing rate `<= 1 Hz`. At the session level it also discards any session with fewer than 10 raw units or fewer than 10 units after the firing-rate filter.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']
```

```python
mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
fr = fr[keep]
...
if n_neurons < MIN_UNITS:
    return None, f"Only {n_neurons} neurons after FR filter"
```

iii. The notes justify this by claiming it matches `findClusters.m` in `"all"` mode and `removeLowFRClusters.m`, and by citing a paper-level minimum-units criterion for the extra `MIN_UNITS` session filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the trial’s go-cue time, so the stored neural traces are in seconds relative to go cue onset before binning.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The AI explicitly states in the notes that this matches `alignSpikes.m`: align spike times by `trialtm - goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins from `-2.5` to `+2.5` s, giving 500 bins per trial. It does not perform a second rebinning step after that.

ii.
```python
DT = 0.01           # 10ms bins (params.dt = 1/100)
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
T_BINS = len(TIME_AXIS)
```

iii. The justification is explicit but mistaken: both comments and notes say this was intended to match the reference `params.dt`, which the AI interpreted as `1/100`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read as a raw variable. It is created from the AI’s chosen analysis time grid, which is defined relative to go cue onset using `TMIN`, `TMAX`, and `DT`.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
```

```python
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The justification is implicit: once every stream is aligned to go cue onset, the bin centers themselves become the decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI simply constructs evenly spaced bin centers across the analysis window and copies that same 1D vector into every trial as a `(1, T)` array.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. No separate justification is given beyond using the common aligned time axis for the whole dataset.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time series is exactly the same time grid used for neural spike binning and for video interpolation, so alignment is enforced by construction.

ii.
```python
counts, _ = np.histogram(aligned, bins=EDGES)
...
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The AI’s implied justification is that all modalities should share one common go-cue-centered axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. In the final code, lick direction is derived from the per-trial instructed side `R` only: `R == 1` becomes right and `R == 0` becomes left. The earlier hit/miss arrays are used only for trial filtering, not for the lick-direction label itself.

ii.
```python
L, R = sd['L'], sd['R']
...
out[0, :] = int(R[ti])                  # lick direction: L=0, R=1
```

iii. The AI does not defend this explicitly in the notes. The trajectory suggests it focused on the prompt’s binary left/right output and assumed the retained hit/miss trials made the instructed side usable as the label.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No reconstruction of actual choice is done. The AI converts `R` to an integer and repeats that binary value across all time bins of the trial.

ii.
```python
out = np.zeros((6, T_BINS), dtype=np.int64)
out[0, :] = int(R[ti])                  # lick direction: L=0, R=1
```

iii. The implied justification is simplicity and compliance with the prompt’s two-class left/right requirement; there is no explicit discussion of miss trials or no-lick trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived directly from the per-trial `autowater` flag.

ii.
```python
autowater, stim_en = sd['autowater'], sd['stim_en']
...
out[1, :] = 1 - int(autowater[ti])       # context: WC=0, DR=1
```

iii. The notes justify this as the WC-versus-DR indicator: autowater trials are water-cued, the others are delayed-response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI remaps `autowater == 1` to WC (`0`) and `autowater == 0` to DR (`1`), then repeats that label across all bins in the trial.

ii.
```python
out[1, :] = 1 - int(autowater[ti])       # context: WC=0, DR=1
```

iii. The justification is explicit in the notes: this matches the prompt’s coding of WC as `0` and DR as `1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is encoded from the `hit` flag after the earlier trial filter has already limited the dataset to hit-or-miss trials. In effect, miss trials become `0` and hit trials become `1`; ignore trials never reach the output stage.

ii.
```python
hit, miss, early = sd['hit'], sd['miss'], sd['early']
...
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
...
out[2, :] = int(hit[ti])                 # outcome: incorrect=0, correct=1
```

iii. The notes justify the earlier trial filter with the claim that no-response trials should be excluded, which is the main reason a binary outcome became possible.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI uses a two-class encoding only: `hit == 1` is correct (`1`), and all retained non-hit trials are treated as incorrect (`0`). The value is repeated across all time bins.

ii.
```python
out[2, :] = int(hit[ti])                 # outcome: incorrect=0, correct=1
```

iii. The justification is implicit in both the notes and the code comments: this follows the prompt’s binary incorrect/correct specification after dropping ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity only from the bottom camera’s `top_tongue` DeepLabCut track, using x, y, confidence, frame times, the session video offset, and per-trial go-cue times. It does not use the side camera tongue track in the final code.

ii.
```python
feat_names = bot_cam['feat_names']
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
```

```python
t_spd = compute_speed(
    ts[:, 0, tongue_fi], ts[:, 1, tongue_fi], ft,
    ts[:, 2, tongue_fi], DLC_CONF, fill_missing=True)
```

iii. The trajectory shows the AI knew both cameras tracked the tongue, but it ultimately simplified to the bottom-camera `top_tongue` feature. `CONVERSION_NOTES.md` also describes “DLC velocities (Bottom Camera)” as the implemented path.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI fills low-confidence or missing positions with the nearest valid position, computes frame-to-frame speed from x/y differences, linearly interpolates the resulting speed onto the neural time axis, and then fills any interpolated NaNs with nearest values. It does not smooth x/y with a Gaussian, does not preserve missing bins as missing, and does not combine side and bottom tongue views.

ii.
```python
if np.any(valid) and not np.all(valid):
    x, y = fill_positions_nearest(x, y, valid)
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

iii. The justification is explicit in the notes: the AI believed nearest-fill before differentiation gave near-zero velocity when the tongue was not visible and meaningful velocity during licking.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes a per-session median threshold using only non-zero tongue-velocity values from kept trials, then thresholds each time bin into a binary low/high class. There is no separate “not visible” category.

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

iii. `CONVERSION_NOTES.md` explicitly justifies this as a workaround for the tongue being visible only rarely; the AI chose to compute the 50th percentile on non-zero values so the threshold would not collapse to zero.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes frame times relative to go cue by subtracting the session-wide video shift and the trial’s go cue, then linearly interpolates tongue speed onto the common neural bin-center axis and nearest-fills missing bins.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The notes justify the video-offset part by citing `findVideoOffset.m`, and justify the fill step as a way to avoid losing bins when the tongue is not visible.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DeepLabCut tracks, preferably `top_paw` but falling back to `bottom_paw` if necessary, together with the bottom-camera frame times, video offset, and go-cue times.

ii.
```python
paw_fi = -1
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. The notes describe bottom-camera paw tracking as the implemented source. The fallback to `bottom_paw` is an implementation choice visible in code rather than something justified in prose.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The paw path uses the same processing as the tongue path in the AI code: nearest-fill invalid x/y positions, compute frame-to-frame speed, interpolate onto the neural time axis, and nearest-fill the interpolated NaNs.

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

iii. The justification is only implicit: the AI reused the same velocity helper for both kinematic outputs.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI uses the 50th percentile of all paw-velocity samples from kept trials as a per-session threshold and converts each time bin to a binary low/high class, again without any third “not visible” category.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
...
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. The notes justify this as the standard per-session median split requested by the prompt.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw speed is aligned the same way as tongue speed: bottom-camera frame times are shifted by the video offset and go cue, then linearly interpolated to the neural time grid and nearest-filled.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The justification is the same offset-based clock alignment described in the notes for video data more generally.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the separate `motionEnergy_<session>.mat` file, using the `me['data']` field, and aligned using side-camera frame times when those are available.

ii.
```python
if os.path.exists(me_path):
    me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
    me_struct = me_mat['me']
    me_trials = me_struct['data'][0, 0]
```

iii. `CONVERSION_NOTES.md` explicitly says motion energy comes from the separate file and that some sessions needed special handling because the struct format varied.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI takes each trial’s motion-energy trace, aligns it in time, linearly interpolates it to the neural time axis, and nearest-fills missing values. If the side-camera frame count does not match the motion-energy length, it falls back to a synthetic `400 Hz` time base with an assumed `-0.5 s` offset. If loading fails, the whole session’s motion energy stays at zeros.

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

iii. The notes justify the fallback behavior pragmatically: several sessions had motion-energy loading issues, and the AI chose all-zero motion-energy arrays rather than dropping those sessions.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes the 50th percentile of all aligned motion-energy samples from kept trials and thresholds each bin into binary low/high classes. Missing bins are not preserved as missing because they are nearest-filled first.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
...
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. The explicit justification in the notes is that motion energy and paw velocity use the standard per-session median split from the prompt.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The preferred alignment uses side-camera frame times shifted by the session video offset and trial go cue, followed by linear interpolation onto the neural time axis. If those frame times are unavailable or length-mismatched, the AI substitutes an assumed `400 Hz` frame clock with a fixed `-0.5 s` offset.

ii.
```python
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
    if len(trial_me) != len(aln):
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
else:
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
```

iii. The offset-based part is justified in the notes by reference to `findVideoOffset.m`; the synthetic fallback is not justified beyond robustness to malformed files.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally imputes or papers over missing data rather than preserving missingness. It nearest-fills low-confidence or NaN tracking positions before computing velocity, nearest-fills NaNs after interpolation, uses a default `vidshift = 0.5` when video offset computation fails, substitutes synthetic frame times when motion-energy and frame counts disagree, and leaves failed motion-energy sessions as all zeros. It also keeps late behavioral trials with all-zero neural activity instead of removing them.

ii.
```python
if np.any(valid) and not np.all(valid):
    x, y = fill_positions_nearest(x, y, valid)
elif not np.any(valid):
    return np.zeros(len(x))
```

```python
except Exception as e:
    vidshift = 0.5
...
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

```markdown
### All-zero Neural Data (Sessions 36, 43)
Late trials ... have all-zero neural activity. This likely indicates the recording ended before the behavioral session. These trials pass through the pipeline but contribute no information.
```

iii. The notes explicitly justify these choices as robustness measures that keep sessions and bins usable for decoder training instead of dropping data.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive work in the AI code is the repeated per-neuron, per-trial spike histogramming and smoothing, plus the per-trial interpolation of tongue, paw, and motion-energy traces. File loading is also substantial, but the code adds large nested Python loops that the human reference avoids.

ii.
```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        ...
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

```python
for ti in range(n_trials):
    ...
    interp_fn = interp1d(aln, t_spd, kind='linear',
                         bounds_error=False, fill_value=np.nan)
    tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. There is no explicit runtime analysis in the notes. This conclusion is inferred from the implementation structure.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest vectorization opportunities are the nested neuron-by-trial spike loop, the per-trial interpolation loops for tongue, paw, and motion energy, and the repeated Python loops used to collect cluster qualities and camera data. The spike loop is the clearest missed optimization relative to the reference.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        ...
```

```python
for ti in range(n_trials):
    ...
    me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. No explicit justification is provided. The code comments focus on matching MATLAB behavior, not on vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The AI repeats several computations across trials that could have been organized more centrally: it recomputes spike masks for every neuron-trial pair, performs separate interpolation/fill passes for tongue, paw, and motion energy on the same session time grid, and processes all trials for neural and video streams even though only `trial_idx` trials are emitted.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
```

```python
for ti in range(n_trials):
    ...
    tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
    ...
    paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The code offers no explicit justification for these repeated passes; they appear to be straightforward implementation choices.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes neural firing rates, camera velocities, and motion-energy traces for all trials, then discards any trial not in `trial_idx` when assembling outputs. It also loads and stores fields such as `L`, `no_resp`, and all side-camera frame times even though they are not needed in the final emitted dataset, and it computes thresholds only after the full per-trial arrays already exist.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        ...
```

```python
for ti in range(n_trials):
    ...

for ti in trial_idx:
    neural_list.append(fr[:, :, ti].copy())
    ...
```

iii. There is no explicit justification for this extra work. It follows from the AI’s choice to process whole-session arrays first and apply the kept-trial mask only at the end.
