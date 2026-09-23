# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a list of 25 ALM sessions in `SESSIONS`, all from `/app/data/Ephys_Behavior`, and opens each `data_structure_<animal>_<date>.mat` with a custom `Session` reader imported from `sessionio.py`. Motion energy is loaded separately from `motionEnergy_<animal>_<date>.mat` in the same folder. It does not search both ephys folders or include the randomized-delay sessions from the human reference.

ii. 
```python
DATA_DIR = '/app/data/Ephys_Behavior'
...
SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ...
    ('JEB19', '2023-04-21', [1]),
]
...
path = os.path.join(DATA_DIR, 'data_structure_%s_%s.mat' % (anm, date))
s = Session(path)
```

iii. The trajectory says the agent chose “the ALM recording sessions listed in `load<ANM>_ALMVideo.m` (25 sessions, 10 mice), i.e. the two-context / DR ephys dataset used in the main figures,” and implemented a helper HDF5 reader (`sessionio.py`) to read those files directly.

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from the first element of each hard-coded session tuple (`anm`). During assembly, unique animals are appended in first-seen order, and `subject_idx` stores the index of each session’s animal in that list.

ii. 
```python
return dict(neural=neural, input=inp, output=outputs, regions=regions,
            subject=anm, info=info)
...
if res['subject'] not in data['subjects']:
    data['subjects'].append(res['subject'])
data['subject_idx'].append(data['subjects'].index(res['subject']))
```

iii. The trajectory justification is that the selected session list already encodes the animal identity, so the subject split follows the paper loaders’ session metadata rather than any unreliable field inside the `.mat` files.

## 1-c. How are the data split into sessions?

i. One `(animal, date, probes)` tuple in `SESSIONS` corresponds to one session. Each session becomes one element in `data['neural']`, `data['input']`, and `data['output']`.

ii. 
```python
for anm, date, probes in sessions:
    print('processing %s %s' % (anm, date), flush=True)
    res = process_session(anm, date, probes)
    if res is None:
        continue
    data['neural'].append(res['neural'])
    data['input'].append(res['input'])
    data['output'].append(res['output'])
```

iii. The trajectory states that the session list was transcribed from the paper’s `load<ANM>_ALMVideo.m` scripts, with the intended probe for each session taken from those loaders.

## 1-d. How are the data split into trials?

i. Trials are the Bpod trial indices from `0` to `ntrials_all - 1`. After trial filtering, `trials = np.where(keep)[0]` is the kept trial list. Neural spikes are assigned back to those trial indices through `trial_pos`, and all outputs are indexed by the same `trials` array.

ii. 
```python
ntrials_all = s.ntrials
...
keep = (~early) & (~stim) & np.isfinite(gocue)
trials = np.where(keep)[0]
...
trial_pos = -np.ones(ntrials_all + 1, dtype=int)
trial_pos[trials + 1] = np.arange(len(trials))   # bp trials are 1-indexed
...
ti = trial_pos[tr]
```

iii. The trajectory justification is implicit: the agent treated the Bpod table and the cluster `trial` field as the authoritative trial definitions, matching the MATLAB alignment code.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding early-lick trials, photostimulation trials, and trials with non-finite go-cue times. Ignore trials are kept. The code does not implement the reference solution’s extra cutoff for trials extending beyond the recording.

ii. 
```python
early = s.bp_flag('early')
stim = s.stim_enable()
...
keep = (~early) & (~stim) & np.isfinite(gocue)
trials = np.where(keep)[0]
```

iii. The trajectory explicitly says the agent followed the paper’s repeated `~early & ~stim.enable` conditions and kept ignore trials because `ignore` is one of the requested decoder targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from each cluster’s spike trial numbers and within-trial spike times, read through `s.clu_spikes(prb, ci)`, together with `bp.ev.goCue` for alignment.

ii. 
```python
gocue = s.ev('goCue')
...
tr, ttm = s.clu_spikes(prb, ci)
aligned = ttm - gocue[tr - 1]                 # alignSpikes.m
```

iii. The trajectory justification cites `alignSpikes.m` and `getSeq.m`: spike times are already stored relative to trial time, so only subtraction of the trial’s go-cue time is needed before binning.

## 2-b. How is the `neural` data processed?

i. For each kept probe and cluster, spikes are counted into `DT = 0.01` s bins over `[-2.5, 2.5]`, converted to firing rates by dividing by `DT`, then smoothed with a causal Gaussian kernel of width `SMOOTH_N = 15`. The two probes of a dual-probe session are concatenated.

ii. 
```python
DT = 0.01             # s, bin size (params.dt = 1/100)
SMOOTH_N = 15         # params.smooth
...
bi = np.floor((aligned - TMIN) / DT).astype(int)
...
np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
rate = smooth_causal(counts / DT)                 # getSeq.m
...
fr = np.concatenate(fr_list, axis=0)                  # (units, trials, time)
```

iii. The trajectory repeatedly says the agent chose to mimic `getSeq.m` and `mySmooth.m`, and it even tested the smoothing kernel to verify its causal direction matched MATLAB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, clusters whose quality label is in `('garbage', 'gabrga', 'noisy', 'real?')` are excluded. Then units with mean firing rate `<= 1 Hz` over the full trial window are removed. Finally, a session is dropped entirely if fewer than 10 units remain.

ii. 
```python
LOW_FR = 1.0
MIN_UNITS = 10
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')
...
cluix = [i for i, q in enumerate(qual) if q.lower() not in BAD_QUALITY]
...
meanfr = fr.mean(axis=(1, 2))
use = meanfr > LOW_FR
fr = fr[use]
...
if fr.shape[0] < MIN_UNITS or len(trials) < 2:
    ...
    return None
```

iii. The trajectory says the agent used the paper’s/manual scripts for bad quality labels, the Methods statement that units must exceed 1 Hz, and the Methods statement that sessions need at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the go-cue time of its own trial, producing seconds from go-cue onset before binning.

ii. 
```python
tr, ttm = s.clu_spikes(prb, ci)
aligned = ttm - gocue[tr - 1]                 # alignSpikes.m
```

iii. The trajectory explicitly references `alignSpikes.m` and says the dataset should be aligned to `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use `DT = 0.01` s bins, i.e. 10 ms, spanning `-2.5` to `+2.5` s. No additional temporal rebinning is applied after the initial spike binning or video interpolation.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2
```

iii. The trajectory states the agent chose `dt = 1/100 s` because that value appears in the tutorial and figure scripts it inspected.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw variable; it is the synthetic `TIME` axis defined by `TMIN`, `TMAX`, and `DT`, intended to represent seconds from go cue.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2
...
inp = [np.asarray(TIME, dtype=np.float32).reshape(1, NT)] * len(trials)
```

iii. The trajectory justification is that time-from-go-cue is the decoder input requested by the task, so the agent used the same bin centers as the neural representation.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing of raw measurements is performed. The code constructs a length-`NT` vector of bin centers and copies that same vector into every trial.

ii. 
```python
TIME = EDGES[:-1] + DT / 2
...
inp = [np.asarray(TIME, dtype=np.float32).reshape(1, NT)] * len(trials)
```

iii. The trajectory gives no deeper justification beyond matching the neural time axis and the requested alignment.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is aligned by construction: `TIME` is the exact same bin-center axis used for neural spike binning, so each input timepoint corresponds to one neural time bin.

ii. 
```python
TIME = EDGES[:-1] + DT / 2
...
bi = np.floor((aligned - TMIN) / DT).astype(int)
...
inp = [np.asarray(TIME, dtype=np.float32).reshape(1, NT)] * len(trials)
```

iii. The trajectory says the decoder input should share the same go-cue-centered axis as the neural data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the trial outcome flags `hit` and `miss` together with the instructed side flags `R` and `L`.

ii. 
```python
hit = s.bp_flag('hit')
miss = s.bp_flag('miss')
R = s.bp_flag('R')
L = s.bp_flag('L')
```

iii. The trajectory justification is that the paper/tutorial exposes hit/miss and side flags directly, while lick direction itself is not stored as a separate categorical variable.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code assigns `2` for no response by default, then rewrites hits as the instructed side and misses as the opposite side. The output is repeated across all time bins within a trial.

ii. 
```python
lick_dir = np.full(len(trials), 2, dtype=np.int64)
lick_dir[hit[trials] & R[trials]] = 1
lick_dir[hit[trials] & L[trials]] = 0
lick_dir[miss[trials] & R[trials]] = 0
lick_dir[miss[trials] & L[trials]] = 1
...
o[0] = lick_dir[k]
```

iii. The trajectory justification is explicit: ignore trials need a third class because `none` is one of the requested decoder outputs.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag.

ii. 
```python
autowater = s.bp_flag('autowater')   # 1 in water-cued (WC) blocks
```

iii. The trajectory says `bp.autowater` is the proxy for WC versus DR blocks, following the tutorial and Methods text.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Trials with `autowater == 1` are labeled `0` (WC) and all others are labeled `1` (DR). That per-trial value is then repeated across all time bins.

ii. 
```python
context = np.where(autowater[trials], 0, 1).astype(np.int64)
...
o[1] = context[k]
```

iii. The trajectory justification is that WC and DR are directly defined by the paper’s block structure and by `bp.autowater`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit`, `no`, and, implicitly, the default incorrect class for the remaining kept trials. `miss` is read earlier for lick direction but is not used directly in the `outcome` assignment.

ii. 
```python
hit = s.bp_flag('hit')
miss = s.bp_flag('miss')
no = s.bp_flag('no')
...
outcome = np.full(len(trials), 0, dtype=np.int64)
outcome[hit[trials]] = 1
outcome[no[trials]] = 2
```

iii. The trajectory justification is that ignore/no-response trials must be preserved as the requested `ignore` output class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome defaults to `0` (incorrect), then hit trials are rewritten to `1` (correct) and `no` trials are rewritten to `2` (ignore). The result is repeated across all time bins of the trial.

ii. 
```python
outcome = np.full(len(trials), 0, dtype=np.int64)
outcome[hit[trials]] = 1
outcome[no[trials]] = 2
...
o[2] = outcome[k]
```

iii. The trajectory explicitly says the agent retained ignore trials because the decoder is supposed to predict them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived only from the side-camera DeepLabCut tongue marker (`'tongue'`) and its frame times, together with `goCue` and the session-wide video offset. The bottom-camera tongue marker is not used.

ii. 
```python
vidshift = s.video_offset()
feats0 = s.traj_featnames(0)     # side cam
...
tongue_ix = feats0.index('tongue')
...
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
```

iii. The trajectory originally used only the side-camera tongue marker and later justified a `nan_gradient` change to preserve visible edge frames, but it never added the bottom-camera tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the side-camera tongue position is linearly interpolated onto the neural time axis, a NaN-aware discrete gradient is computed on the interpolated positions, and speed is the Euclidean norm of the x/y derivatives. The resulting per-bin speed is split later at the session median. No likelihood thresholding, per-run smoothing, or cross-camera normalization/averaging is performed.

ii. 
```python
pos = interp_to_axis(ft, xy, taxis)
vel = nan_gradient(pos)
tongue_speed[k] = np.hypot(vel[:, 0], vel[:, 1])
```

iii. The trajectory says the agent first used `np.gradient`, then changed to `nan_gradient` because `np.gradient` over-propagated NaNs and “artificially inflate[d] the ‘not visible’ class.”

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After all tongue-speed values are computed for the session, visible bins are thresholded at the 50th percentile. Bins with NaN tongue speed remain in class `2`.

ii. 
```python
def discretise(x):
    out = np.full(x.shape, 2, dtype=np.int64)
    vis = np.isfinite(x)
    if vis.any():
        thr = np.percentile(x[vis], 50)
        out[vis] = (x[vis] >= thr).astype(np.int64)
    return out

tongue_d = discretise(tongue_speed)
```

iii. The trajectory justification is the task instruction itself: discretize with a per-session 50th-percentile threshold and use a separate “not visible” class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are converted into the video clock used by the raw tracking, by forming `taxis = TIME + gocue[tr] + vidshift`, then tongue positions are interpolated onto that axis. This is equivalent to subtracting the video offset and go cue from frame times before aligning them to the neural bins.

ii. 
```python
vidshift = s.video_offset()
...
taxis = TIME + gocue[tr] + vidshift                 # into video clock
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
...
pos = interp_to_axis(ft, xy, taxis)
```

iii. The trajectory explicitly cites `findVideoOffset.m` and says DLC and motion-energy frame times should be shifted by `mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)` and aligned to go cue.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DeepLabCut marker `'top_paw'`, its frame times, `goCue`, and the session-wide video offset.

ii. 
```python
feats1 = s.traj_featnames(1)     # bottom cam
paw_ix = feats1.index('top_paw')
...
ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
```

iii. The trajectory says the agent followed the paper’s use of `top_paw` as the reliable paw feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Bottom-camera paw positions are interpolated onto the neural time axis, passed through the same `nan_gradient` helper, baseline-corrected by subtracting the median frame-to-frame displacement, and converted to speed by Euclidean norm. No explicit likelihood thresholding or smoothing is applied.

ii. 
```python
pos1 = interp_to_axis(ft1, xy1, taxis)
vel1 = nan_gradient(pos1)
base = np.nanmedian(np.diff(pos1, axis=0), axis=0)
vel1 = vel1 - base[None, :]
paw_speed[k] = np.hypot(vel1[:, 0], vel1[:, 1])
```

iii. The trajectory says the baseline subtraction was added specifically to mimic `findVelocity.m` for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw-speed values are thresholded at the session median over finite bins. NaN bins are assigned class `2`.

ii. 
```python
paw_d = discretise(paw_speed)
```

iii. The trajectory justification is again the task specification: per-session 50th-percentile discretization with a “not visible” class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw positions are aligned exactly like tongue positions: the code uses the per-trial go cue and the session video offset to build the neural-aligned video axis, then interpolates bottom-camera paw tracking onto that axis.

ii. 
```python
taxis = TIME + gocue[tr] + vidshift
ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
pos1 = interp_to_axis(ft1, xy1, taxis)
```

iii. The trajectory cites the same `findVideoOffset.m` logic for all video-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motionEnergy_<animal>_<date>.mat`, specifically the nested `me['data']` content, along with the side-camera frame times of the tongue camera and the session video offset.

ii. 
```python
mepath = os.path.join(DATA_DIR, 'motionEnergy_%s_%s.mat' % (anm, date))
if os.path.exists(mepath):
    medat = sio.loadmat(mepath)['me']['data'][0, 0]
    while medat.dtype.names is not None and 'data' in medat.dtype.names:
        medat = medat['data'][0, 0]
    medat = medat.ravel()
```

iii. The trajectory says the agent copied the MATLAB loader’s repeated unwrapping of nested `me.data` fields and used the standalone motion-energy files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial with video and matching frame counts, the motion-energy trace is linearly interpolated onto the neural time axis using side-camera frame times, then NaNs at the ends are filled with nearest-neighbor values. The resulting per-bin values are discretized later at the session median.

ii. 
```python
m = np.asarray(medat[tr], float).ravel()
ft, _ = s.traj_feat_trial(0, tr, tongue_ix)
...
vals = interp_to_axis(ft, m, taxis)
me_arr[k] = fill_nearest(vals)   # loadMotionEnergy.m fills nans
```

iii. The trajectory justification is that `loadMotionEnergy.m` interpolates to the aligned axis and fills NaNs with nearest neighbors.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion-energy values are thresholded at the session median across finite bins; NaN bins are assigned class `2`.

ii. 
```python
me_d = discretise(me_arr)
```

iii. The trajectory justification is the task requirement of per-session median discretization with a separate “no video” class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The code uses the same `taxis = TIME + gocue[tr] + vidshift` construction used for tongue and paw, and interpolates motion energy from side-camera frame times onto that axis.

ii. 
```python
taxis = TIME + gocue[tr] + vidshift
vals = interp_to_axis(ft, m, taxis)
```

iii. The trajectory explicitly groups DLC and motion energy together under the same `findVideoOffset.m` alignment rule.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or bad video data are handled by skipping that trial’s video-derived computations and leaving the affected arrays as NaN, which later become class `2` after discretization. Trials with non-finite `NdroppedFrames` are skipped before video processing. Missing motion-energy endpoints are filled by nearest-neighbor interpolation. The agent also replaced `np.gradient` with `nan_gradient` so isolated NaNs do not wipe out neighboring visible bins.

ii. 
```python
if ndrop is not None and not np.isfinite(ndrop[tr]):
    continue
...
if ft is None or xy is None or not np.isfinite(ft).any():
    continue
...
vals = interp_to_axis(ft, m, taxis)
me_arr[k] = fill_nearest(vals)
...
vis = np.isfinite(x)
...
out = np.full(x.shape, 2, dtype=np.int64)
```

iii. The trajectory explicitly justifies `nan_gradient` as avoiding an inflated “not visible” class and cites `fillmissing(...,'nearest')` from the MATLAB motion-energy loader as the reason for `fill_nearest`.

## 11-a. What are the most time-consuming steps of the code?

i. The code’s most expensive work appears to be the per-cluster spike binning loop and the per-trial interpolation of tongue, paw, and motion-energy traces. The agent trajectory does not separately benchmark runtime, but those are the dominant explicit loops in the implementation.

ii. 
```python
for ii, ci in enumerate(cluix):
    tr, ttm = s.clu_spikes(prb, ci)
    ...
    np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
...
for k, tr in enumerate(trials):
    ...
    pos = interp_to_axis(ft, xy, taxis)
    ...
    pos1 = interp_to_axis(ft1, xy1, taxis)
```

iii. The trajectory justification is only indirect: the agent spent most debugging effort on smoothing, velocity, and motion-energy interpolation rather than on I/O.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The cluster loop for spike accumulation and the per-trial loops for video interpolation, paw/tongue speed, and motion-energy interpolation could potentially be vectorized or batched more aggressively. The code processes each cluster and each kept trial separately.

ii. 
```python
for ii, ci in enumerate(cluix):
    ...
for k, tr in enumerate(trials):
    ...
for k, tr in enumerate(trials):
    if not has_video[k] or tr >= len(medat):
        continue
```

iii. The trajectory contains no explicit optimization discussion here, beyond the later addition of `nan_gradient` and baseline correction for correctness.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats construction of the aligned video axis `taxis = TIME + gocue[tr] + vidshift` for each trial and separately repeats interpolation-related work for tongue, paw, and motion energy. It also re-fetches side-camera frame times once for tongue and again for motion energy.

ii. 
```python
taxis = TIME + gocue[tr] + vidshift
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
...
taxis = TIME + gocue[tr] + vidshift
vals = interp_to_axis(ft, m, taxis)
```

iii. The trajectory does not mention this repetition; it appears from the code structure itself.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Relative to the human reference, very little extra processing is obviously wasted. The custom `Session` loader reads only the fields the script asks for instead of recursively materializing the entire MATLAB object tree. The only clearly extra work inside `convert_data.py` is building metadata/session summaries and brain-region labels that the decoder itself does not use.

ii. 
```python
locs = s.probe_locs()
...
info = dict(animal=anm, date=date, probes=list(probes),
            n_units=int(fr.shape[0]), n_trials=int(len(trials)),
            n_trials_total=int(ntrials_all),
            n_wc_trials=int(np.sum(context == 0)),
            n_dr_trials=int(np.sum(context == 1)),
            n_video_trials=int(has_video.sum()))
```

iii. The trajectory justification for the custom loader was pragmatism: it was written to read the needed MATLAB HDF5 fields directly and avoid bulky exploratory scripts after conversion was complete.
