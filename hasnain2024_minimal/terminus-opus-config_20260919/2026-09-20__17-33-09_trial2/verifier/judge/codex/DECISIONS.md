# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the authors' `load<ANM>_ALMVideo.m` scripts to enumerate sessions and probes, keeps only sessions whose `data_structure_<anm>_<date>.mat` file exists in either `/app/data/Ephys_Behavior` or `/app/data/RandomizedDelay_Ephys_Behavior`, and loads each session with a custom loader that supports both MATLAB v7.3 (`h5py`) and older MATLAB files (`scipy.io`). Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat`.

ii.
```python
def parse_sessions():
    out = []
    for fn in sorted(glob.glob(os.path.join(LOADSCRIPT_DIR, '*.m'))):
        ...
        if s.startswith('%'):
            continue
        ...
        cur = {'anm': anm, 'date': m.group(1), 'probe': [1]}
        ...

def find_files(anm, date):
    for d in DATA_DIRS:
        f = os.path.join(d, 'data_structure_%s_%s.mat' % (anm, date))
        if os.path.exists(f):
            me = os.path.join(d, 'motionEnergy_%s_%s.mat' % (anm, date))
            return f, (me if os.path.exists(me) else None), os.path.basename(d)

def load_session(fn, probes, traj_feats):
    if is_v73(fn):
        return _load_v73(fn, probes, traj_feats)
    return _load_v7(fn, probes, traj_feats)
```

iii. In the trajectory, the AI said it wanted to match the authors' own loading scripts rather than discover sessions by folder contents alone, and it explicitly chose a unified v7/v7.3 loader so all 44 paper sessions present on disk could be read.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `anm` field parsed from the loading scripts. At assembly time, the AI builds `subjects` as the sorted unique animal IDs and `subject_idx` by looking up each session's `anm`.

ii.
```python
sessions = parse_sessions()
...
subjects = sorted({r['anm'] for r in res})
...
'subjects': subjects,
'subject_idx': np.array([subjects.index(r['anm']) for r in res]),
```

iii. The trajectory repeatedly refers to sessions as `(anm, date, probe)` tuples and reports totals by animal, so the AI's justification was that the animal id from the loading scripts is the stable subject key.

## 1-c. How are the data split into sessions?

i. One session is one uncommented `(anm, date, probe)` entry from the authors' loading scripts, provided the matching `data_structure_*.mat` file exists. Each retained session becomes one element in the top-level `neural`, `input`, and `output` lists.

ii.
```python
sessions = parse_sessions()
sessions = [s for s in sessions if find_files(s['anm'], s['date'])[0] is not None]
...
'neural': [r['neural'] for r in res],
'input': [r['input'] for r in res],
'output': [r['output'] for r in res],
```

iii. In the trajectory, the AI said it found 25 `Ephys_Behavior` sessions and 19 `RandomizedDelay_Ephys_Behavior` sessions from the paper's scripts and decided to include all 44.

## 1-d. How are the data split into trials?

i. Trials are indexed by the Bpod trial count `Ntrials` and the per-trial arrays in `bp`. After trial filtering, `trials` is an array of trial indices, and all neural, behavioral, video, and motion-energy data are read using those trial indices.

ii.
```python
N = d['Ntrials']
...
keep = (d['early'] == 0) & (d['stim'] == 0) & np.isfinite(gocue)
...
trials = np.where(keep)[0]
...
for j, t in enumerate(trials):
    neural.append(np.ascontiguousarray(rates[:, :, j].T))
    ...
    out[0, :] = lick_dir[t]
```

iii. The trajectory shows the AI treating one trial as one entry in the `bp` arrays plus the corresponding trial entry in `clu`, `traj`, and motion-energy arrays; it also checked whether `bp.fidx` created a trial offset and concluded it did not.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with `early == 0`, `stim == 0`, and a finite go-cue time. It also drops any trailing trials beyond the available tongue-camera trial list, and after neural processing it drops any remaining trials whose smoothed neural tensor is entirely zero across all units and time bins.

ii.
```python
keep = (d['early'] == 0) & (d['stim'] == 0) & np.isfinite(gocue)
ntraj = len(d['traj'][TONGUE_VIEW]['trials'])
keep = keep[:N]
if ntraj < N:
    keep[ntraj:] = False
trials = np.where(keep)[0]
...
nonempty = np.any(rates != 0, axis=(0, 1))
if n_no_ephys:
    rates = rates[:, :, nonempty]
    trials = trials[nonempty]
```

iii. In the trajectory, the AI justified early-lick and photostimulation removal as matching the paper, and later added the zero-spike trial filter after noticing that in two sessions the recording ended before behavior did.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu` cluster entries, specifically each unit's `trial` and `trialtm` vectors, together with the per-trial `goCue` times used for alignment. Cluster quality labels are also read to filter units before rate construction.

ii.
```python
for p in probes:
    ...
    q = _h5str(f, g['quality'][i, 0]).strip()
    ...
    trial = np.array(f[g['trial'][i, 0]]).ravel().astype(int)
    trialtm = np.array(f[g['trialtm'][i, 0]]).ravel().astype(float)
    units.append({'quality': q, 'trial': trial, 'trialtm': trialtm, 'probe': p})
...
tm = u['trialtm'] - gocue[np.clip(tt - 1, 0, N - 1)]
```

iii. The trajectory says the AI was following `alignSpikes.m`: it identified `trialtm` as the spike time stream and `goCue` as the event needed to align it.

## 2-b. How is the `neural` data processed?

i. For each unit and kept trial, spikes are aligned to the go cue, counted into 10 ms bins over `[-2.5, 2.5]`, divided by `DT` to convert counts to firing rate, and smoothed with a custom port of `mySmooth.m`, implemented as a causal half-Gaussian with `SMOOTH = 15` and `reflect` padding.

ii.
```python
DT = 0.01
SMOOTH = 15
...
def my_smooth(x, N=SMOOTH, bctype=BCTYPE):
    kern = gausswin(N)
    kern[:N // 2] = 0
    kern = kern / kern.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xf)
    return out[trim:]
...
cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
rates[:, iu, :] = my_smooth(cnt / DT).astype(np.float32)
```

iii. The trajectory explicitly says the AI chose `dt = 1/100` and the causal half-Gaussian because it believed those were the relevant paper parameters and verified that this decoder still trained successfully.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removes units whose `quality` string is exactly one of `garbage`, `gabrga`, `noisy`, or `real?`, then removes units with mean firing rate `<= 1 Hz` over the processed tensor, and finally drops entire sessions if fewer than 10 units remain.

ii.
```python
QUAL_EXCLUDE = ('garbage', 'gabrga', 'noisy', 'real?')
LOW_FR = 1.0
MIN_UNITS = 10
...
if q in QUAL_EXCLUDE:
    continue
...
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
if use.sum() < MIN_UNITS:
    return None
```

iii. In the trajectory, the AI said it was following the paper's quality filtering and low-firing-rate rule, and it added the `>= 10 units` session rule as a paper inclusion criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each spike's trial-specific go-cue time from `trialtm`, so all spike times become seconds from go-cue onset before binning.

ii.
```python
tt = u['trial']
tm = u['trialtm'] - gocue[np.clip(tt - 1, 0, N - 1)]   # trialtm_aligned
```

iii. The trajectory states that the AI was reproducing `alignSpikes.m` and treating `obj.bp.ev.goCue` as the common alignment event for all sessions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10 ms bins from `-2.5 s` to `+2.5 s`, giving a fixed session-wide time axis `TAXIS`. There is no later rebinning; smoothing is applied on that same 10 ms grid.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
```

iii. The trajectory repeatedly says the AI chose `dt = 10 ms` because it interpreted the paper's scripts as using `params.dt = 1/100`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not taken from a dedicated raw variable beyond the alignment event itself; it is constructed as the bin-center time axis around the go cue, using the same `TMIN`, `TMAX`, and `DT` parameters that define the neural grid.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
...
time_in = TAXIS.astype(np.float32)[None, :]
```

iii. The trajectory treats this as a decoder-specific input derived from the chosen go-cue-centered analysis window rather than a separately recorded stream.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes the bin centers once as `TAXIS` and copies that same 1-by-time array into every trial.

ii.
```python
TAXIS = (EDGES + DT / 2)[:-1]
...
time_in = TAXIS.astype(np.float32)[None, :]
for j, t in enumerate(trials):
    inputs.append(time_in.copy())
```

iii. The trajectory frames this as the simplest way to provide the requested continuous decoder input on the same time basis as the neural data.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same `TAXIS` grid that spike counts are binned onto, so each input sample corresponds to the same time bin as the neural firing rates.

ii.
```python
cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
...
time_in = TAXIS.astype(np.float32)[None, :]
```

iii. The trajectory says the AI wanted all streams on one common go-cue-centered time axis, and used the neural binning grid as that axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the trial outcome flags `hit`, `miss`, and `no`, together with the instructed-side flags `R` and `L`.

ii.
```python
hit, miss, no = d['hit'] > 0, d['miss'] > 0, d['no'] > 0
R, L = d['R'] > 0, d['L'] > 0
```

iii. In the trajectory, the AI says it compared this rule against actual lick events and chose the same task-logic derivation used by the paper code.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI marks a trial as right-lick if `(R and hit) or (L and miss)`, left-lick otherwise among response trials, and sets ignore trials (`no`) to class 2.

ii.
```python
right_lick = (R & hit) | (L & miss)
lick_dir = np.where(no, 2, np.where(right_lick, 1, 0))       # 0 left,1 right,2 none
```

iii. The trajectory says the AI intentionally kept ignore trials as a separate class because the decoder specification requested an `ignore` outcome and those trials still carry other labels.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived solely from the per-trial `autowater` flag.

ii.
```python
aw = d['autowater'] > 0
```

iii. The trajectory describes `autowater` as the indicator of water-cued trials and treats all other trials as delayed-response trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater` trials as `0` (`WC`) and all others as `1` (`DR`).

ii.
```python
context = np.where(aw, 0, 1)                                 # 0 WC, 1 DR
```

iii. The trajectory says this was chosen to satisfy the prompt's requested `WC`/`DR` categorical output directly.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial flags `hit`, `miss`, and `no`.

ii.
```python
hit, miss, no = d['hit'] > 0, d['miss'] > 0, d['no'] > 0
```

iii. The trajectory notes that ignore trials needed to remain in the dataset, so the AI used the explicit `no` flag instead of inferring ignore as "not hit and not miss."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps miss to `0` (`incorrect`), hit to `1` (`correct`), and `no` to `2` (`ignore`).

ii.
```python
outcome = np.where(no, 2, np.where(hit, 1, 0))               # 0 incorrect,1 correct,2 ignore
```

iii. The trajectory justifies this as the requested decoder output coding while keeping ignore trials available for the other labels.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera DeepLabCut trajectory entry `obj.traj[0]`, using the `tongue` feature's x/y coordinates and frame times. It also uses `bitstart_sglx`, `fs`, `bitStart`, and `goCue` to put frames on the neural clock.

ii.
```python
TONGUE_VIEW, TONGUE_FEATS = 0, ['tongue']
...
tv = d['traj'][TONGUE_VIEW]
t_idx = [tv['feats'].index(f) for f in TONGUE_FEATS]
...
ft = tr['frameTimes']
tt = ft - vidshift - gocue[t]
```

iii. In the trajectory and docstring, the AI says it decided to use the side-camera tongue marker and label NaN time points as `not visible`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each kept trial, the AI linearly interpolates the side-camera tongue x/y traces onto the neural time grid, computes finite-difference velocities with `np.gradient`, converts them to scalar speed `sqrt(vx^2 + vy^2)`, and leaves NaNs wherever interpolation cannot produce a value.

ii.
```python
x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
y = interp_nan(TAXIS, tt, tr['xy'][:, 1, fi])
vx = np.gradient(x) / DT
vy = np.gradient(y) / DT
sp.append(np.sqrt(vx ** 2 + vy ** 2))
...
spd = np.where(np.all(np.isnan(sp), axis=0), np.nan, np.nanmean(sp, axis=0))
```

iii. The trajectory says the AI intentionally used speed so the output would be one scalar per time point, and it treated NaNs from missing tracking or out-of-range video as the `not visible` class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI pools all finite tongue-speed samples within a session, computes the 50th percentile, sets samples below that threshold to class `0`, samples at or above it to class `1`, and leaves non-finite bins as class `2`.

ii.
```python
def discretise(x):
    vis = np.isfinite(x)
    out = np.full(x.shape, 2, dtype=np.int64)
    if vis.sum() > 0:
        thr = np.percentile(x[vis], 50)
        out[vis] = (x[vis] >= thr).astype(np.int64)
    return out, (float(np.percentile(x[vis], 50)) if vis.sum() else None)
...
tongue_cat, tongue_thr = discretise(tongue_speed)
```

iii. The trajectory explicitly says the AI interpreted the prompt as requiring the threshold to be computed over visible samples only.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI estimates a session-wide video shift as `(median(bitstart_sglx) / fs) - median(bitStart)`, subtracts that shift and the trial's go cue from each frame time, interpolates the trace onto `TAXIS`, and thereby puts tongue velocity on the neural time grid.

ii.
```python
vidshift = (np.median(d['bitstart_sglx']) / d['fs']) - np.median(d['bitStart'])
...
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
y = interp_nan(TAXIS, tt, tr['xy'][:, 1, fi])
```

iii. In the trajectory, the AI said it was following `findVideoOffset.m` and wanted the video-derived variables on the same axis as the neural bins.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DeepLabCut trajectories `top_paw` and `bottom_paw` in `obj.traj[1]`, plus frame times and the same video-to-go-cue alignment variables used for tongue velocity.

ii.
```python
PAW_VIEW, PAW_FEATS = 1, ['top_paw', 'bottom_paw']
...
pv = d['traj'][PAW_VIEW]
p_idx = [pv['feats'].index(f) for f in PAW_FEATS]
```

iii. The trajectory and docstring say the AI chose to average the two paw markers into one scalar paw-speed variable.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates each paw marker's x/y trace onto `TAXIS`, computes velocity by finite differences, converts to scalar speed, and then averages the two paw-feature speeds with `nanmean` when at least one is finite.

ii.
```python
for fi in feats_idx:
    x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
    y = interp_nan(TAXIS, tt, tr['xy'][:, 1, fi])
    vx = np.gradient(x) / DT
    vy = np.gradient(y) / DT
    sp.append(np.sqrt(vx ** 2 + vy ** 2))
...
spd = np.where(np.all(np.isnan(sp), axis=0), np.nan, np.nanmean(sp, axis=0))
...
paw_speed[:, j] = spd
```

iii. The trajectory says the AI wanted one scalar speed per time bin and treated missing video coverage or missing tracking as NaN, later mapped to class `2`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded exactly like tongue velocity: per session, at the 50th percentile of finite paw-speed samples, with NaNs assigned to category `2`.

ii.
```python
paw_cat, paw_thr = discretise(paw_speed)
```

iii. The trajectory says this thresholding was taken directly from the decoder task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories use the same session-wide `vidshift`, trial-specific go-cue subtraction, and interpolation onto `TAXIS` as tongue trajectories, so paw velocity ends up on the neural grid.

ii.
```python
vidshift = (np.median(d['bitstart_sglx']) / d['fs']) - np.median(d['bitStart'])
...
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
```

iii. The trajectory says the AI wanted every video-derived output to share the neural alignment event and time base.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<anm>_<date>.mat` file, loaded per session into `me_raw`.

ii.
```python
def load_motion_energy(fn):
    m = sio.loadmat(fn, squeeze_me=True, struct_as_record=False)
    me = m['me']
    data = me.data
    if hasattr(data, '_fieldnames'):
        data = data.data
    out = [np.atleast_1d(np.asarray(d, dtype=float)).ravel() for d in np.atleast_1d(data)]
    return out
...
me_raw = load_motion_energy(me_fn) if me_fn else None
```

iii. The trajectory says the AI explicitly inspected the motion-energy file structure and chose the standalone file as the consistent source.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI treats motion energy as a per-frame scalar trace and linearly interpolates it onto `TAXIS`. If frame times and motion-energy length disagree, it falls back to synthetic frame times at 400 Hz with an additional `-0.5` second shift.

ii.
```python
if tr is not None and tr['frameTimes'].size == len(me_raw[t]):
    tt = tr['frameTimes'] - vidshift - gocue[t]
    me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])
elif tr is not None:
    ftimes = (np.arange(len(me_raw[t])) + 1) / 400.0
    me_t[:, j] = interp_nan(TAXIS, ftimes - 0.5 - gocue[t], me_raw[t])
```

iii. The trajectory says the AI wanted motion energy on the same decoder grid as neural and kinematic data, and accepted NaNs or synthetic timing fallbacks when exact frame timing was unavailable.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI applies the same `discretise` function used for tongue and paw velocity: per-session median split over finite samples, with non-finite bins assigned to class `2`.

ii.
```python
me_cat, me_thr = discretise(me_t)
```

iii. The trajectory says this follows the decoder task's requested per-session percentile thresholding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the same session-level `vidshift` and trial-level go cue from the relevant frame-time vector, then interpolating the motion-energy trace to the neural time grid.

ii.
```python
tt = tr['frameTimes'] - vidshift - gocue[t]
me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])
```

iii. The trajectory says the AI was trying to mirror the same video/alignment logic as the other video-derived outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles bad or missing video by leaving the derived signals as NaN, which later become category `2`. If a trial's `NdroppedFrames` is empty or non-finite, the video output for that trial is skipped. If frame times are missing entirely, it fabricates them as `(1..n)/400`. If motion-energy length mismatches video-frame length, it falls back to synthetic 400 Hz frame times with a `-0.5` second offset. It also drops trials with no neural activity at all.

ii.
```python
if tr is None or tr['Ndropped'].size == 0 or np.any(~np.isfinite(tr['Ndropped'])):
    continue
ft = tr['frameTimes']
if ft.size == 0 or not np.any(np.isfinite(ft)):
    ft = (np.arange(tr['xy'].shape[0]) + 1) / 400.0
...
elif tr is not None:
    ftimes = (np.arange(len(me_raw[t])) + 1) / 400.0
    me_t[:, j] = interp_nan(TAXIS, ftimes - 0.5 - gocue[t], me_raw[t])
...
nonempty = np.any(rates != 0, axis=(0, 1))
```

iii. In the trajectory, the AI says it wanted missing or out-of-range samples to become `not visible`/`no video` rather than be imputed as valid movement, but it also chose synthetic timing fallbacks for some missing frame-time cases so the trial could still be used.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive work in this code is the per-session neural spike binning and smoothing loop over units and trials, plus the per-trial interpolation and velocity computations for the video streams. The script also uses multiprocessing across sessions to reduce wall-clock time.

ii.
```python
for iu, u in enumerate(units):
    ...
    for j in range(len(trials)):
        if ends[j] > starts[j]:
            cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
    rates[:, iu, :] = my_smooth(cnt / DT).astype(np.float32)
...
for j, t in enumerate(trials):
    for view, feats_idx, dest in ((tv, t_idx, 'tongue'), (pv, p_idx, 'paw')):
        ...
        x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
...
with Pool(args.nproc, maxtasksperchild=1) as p:
    res = p.map(process_session, sessions, chunksize=1)
```

iii. In the trajectory, the AI described the converter as "per-session caching and multiprocessing" and emphasized full-dataset runtime improvements, which implies it saw session processing rather than assembly as the expensive part.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit-by-trial spike histogram loop could be vectorized further, and the per-trial/per-feature interpolation loops for tongue, paw, and motion energy are also written in scalar Python loops. The subject-index construction also does repeated list searches.

ii.
```python
for iu, u in enumerate(units):
    ...
    for j in range(len(trials)):
        if ends[j] > starts[j]:
            cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
...
for j, t in enumerate(trials):
    for view, feats_idx, dest in ((tv, t_idx, 'tongue'), (pv, p_idx, 'paw')):
        ...
'subject_idx': np.array([subjects.index(r['anm']) for r in res]),
```

iii. The trajectory shows that the AI addressed efficiency mainly by multiprocessing rather than by restructuring these loops.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats percentile computation inside `discretise` by calling `np.percentile(x[vis], 50)` twice, repeats `subjects.index(...)` for every session during assembly, and repeats interpolation/gradient work independently for each trial and each requested feature.

ii.
```python
if vis.sum() > 0:
    thr = np.percentile(x[vis], 50)
    out[vis] = (x[vis] >= thr).astype(np.int64)
return out, (float(np.percentile(x[vis], 50)) if vis.sum() else None)
...
'subject_idx': np.array([subjects.index(r['anm']) for r in res]),
```

iii. The trajectory does not mention any attempt to cache these repeated computations; it focuses instead on getting the whole dataset converted quickly with `Pool`.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader reads several fields that are not used downstream, including `sample`, `delay`, `has_fidx`, and probe-location metadata. The conversion also computes and stores session metadata such as `n_single_units` and the movement thresholds even though those values are not used by the decoder arrays.

ii.
```python
for k in ['goCue', 'sample', 'delay', 'bitStart']:
    out[k] = np.array(bp['ev'][k]).ravel().astype(float)
out['has_fidx'] = 'fidx' in bp
...
out['loc'] = _probe_locs_v73(f, o, probes)
...
'n_single_units': int(sum(1 for u in units
                          if u['quality'].strip() in ('excellent', 'great', 'good'))),
'tongue_speed_threshold': tongue_thr,
'paw_speed_threshold': paw_thr,
'motion_energy_threshold': me_thr,
```

iii. The trajectory shows the AI inspecting these fields during exploration and preserving some of them in metadata for documentation and sanity checking, even though they are not consumed by downstream decoder training.
