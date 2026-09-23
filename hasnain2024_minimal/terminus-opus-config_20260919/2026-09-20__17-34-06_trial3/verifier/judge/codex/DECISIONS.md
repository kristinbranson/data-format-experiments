# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata by parsing the authors' MATLAB meta scripts, but it only processes `data_structure_*.mat` files found by globbing `/app/data/Ephys_Behavior`. It does not include `RandomizedDelay_Ephys_Behavior`. Each selected `data_structure` file is opened directly with `h5py`, while motion-energy files are loaded separately with `scipy.io.loadmat`.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'

def parse_meta_scripts(meta_dir='/app/code/DataLoadingScripts/Recording and video'):
    sessions = {}
    for fn in sorted(glob.glob(os.path.join(meta_dir, '*.m'))):
        ...
        m = re.match(r"meta\(.*\)\.probe\s*=\s*\[?([0-9 ,]+)\]?", s)
        if m and anm is not None and cur_date is not None:
            probes = [int(p) for p in m.group(1).replace(',', ' ').split()]
            sessions[(anm, cur_date)] = probes
    return sessions

meta = parse_meta_scripts()
files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_structure_*.mat')))
```

```python
def load_session(dsfile, probes, verbose=True):
    f = h5py.File(dsfile, 'r')
    ...

def motion_energy(dsfile, f, trials, gocue):
    mefile = dsfile.replace('data_structure', 'motionEnergy')
    me = sio.loadmat(mefile)['me']
```

iii. In the trajectory, the agent decided to use only the fixed-delay `Ephys_Behavior` cohort and exclude `RandomizedDelay` because it considered the latter a separate cohort with little or no WC context (steps 32, 39, 45). It also concluded from inspection that the fixed-delay `data_structure` files were HDF5/v7.3 and so implemented direct `h5py` loading rather than a mixed loader.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from the filename stem as the animal prefix before the date. The `subjects` list is built in first-seen order during the session loop, and `subject_idx` stores the index of that subject in that list for each retained session.

ii.
```python
m = re.match(r'data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat', base)
anm, date = m.group(1), m.group(2)
...
if anm not in subjects:
    subjects.append(anm)
...
data['subject_idx'].append(subjects.index(anm))
```

iii. The trajectory shows the agent using the session filename and meta-script session naming as the authoritative session identity (steps 17, 18, 32). There is no separate in-file subject lookup.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_<animal>_<date>.mat` file in `/app/data/Ephys_Behavior`. Sessions are discovered by globbing that directory, then filtered by whether the parsed meta scripts contain a matching `(animal, date)` entry and by later unit-count quality control.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_structure_*.mat')))

for dsfile in files:
    base = os.path.basename(dsfile)
    m = re.match(r'data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat', base)
    anm, date = m.group(1), m.group(2)
    probes = meta.get((anm, date))
    if probes is None:
        print(f'{base}: no meta entry, skipping')
        continue
    res = load_session(dsfile, probes)
```

iii. In the trajectory, the agent explicitly chose the 25 fixed-delay sessions in `Ephys_Behavior` as the main dataset and treated `RandomizedDelay` as separate (steps 32, 39, 45).

## 1-d. How are the data split into trials?

i. Trials are indexed directly by the Bpod trial arrays in `obj/bp`. The code uses `Ntrials` to size the session, then keeps trial indices where at least one of `hit`, `miss`, or `no` is true and where the go-cue time is not NaN. Downstream arrays use those retained 0-based trial indices.

ii.
```python
ntrials = int(get('Ntrials')[0])
hit, miss, no = get('hit'), get('miss'), get('no')
...
keep = ((hit == 1) | (miss == 1) | (no == 1)) & (early != 1) & (stim != 1) & ~np.isnan(gocue)
trials = np.where(keep)[0]            # 0-based trial indices
```

iii. The trajectory shows the agent concluding that trial structure comes directly from the Bpod arrays and that no reconstruction of boundaries is needed (steps 24, 25, 45).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to "completed" trials only (`hit | miss | no`), with early-lick trials, photoinactivation trials, and trials lacking a finite `goCue` removed. The AI does not apply the reference solution's extra drop of behavior trials that continue after the recording stops.

ii.
```python
keep = ((hit == 1) | (miss == 1) | (no == 1)) & (early != 1) & (stim != 1) & ~np.isnan(gocue)
trials = np.where(keep)[0]
```

iii. The trajectory repeatedly states the intended curation as excluding early-lick and photostim trials while keeping ignore/no-response trials as a decoder target (steps 30, 45, 70). There is no sign that the agent identified the late-behavior / no-ephys tail-trial issue handled in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from per-probe cluster tables in `obj/clu`, specifically each cluster's `trialtm`, `trial`, and `quality`, together with per-trial `bp.ev.goCue` for alignment and probe-location fields for region labeling.

ii.
```python
gocue = np.array(bp['ev/goCue']).ravel()
...
clu = f[f['obj/clu'][prb - 1, 0]]
qual = [mstr(f, r).strip().lower() for r in np.array(clu['quality']).ravel()]
trialtm_refs = np.array(clu['trialtm']).ravel()
trial_refs = np.array(clu['trial']).ravel()
```

iii. The trajectory shows the agent inspecting `obj/clu` structure, cluster quality labels, and go-cue fields before implementing the neural pipeline (steps 24, 27, 43, 45).

## 2-b. How is the `neural` data processed?

i. For each kept probe and cluster, spike times are aligned by subtracting the trial's go cue, binned into 10 ms bins from -2.5 s to +2.5 s, converted to firing rates by dividing by bin width, then smoothed with a custom causal Gaussian kernel meant to mimic `mySmooth.m`.

ii.
```python
DT = 0.01        # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
...
ta = tt - gocue[tr]                     # align to go cue
b = np.floor((ta - TMIN) / DT).astype(np.int64)
...
counts = np.bincount(idx, minlength=len(trials) * NT).reshape(len(trials), NT)
sess_rates[ui] = counts / DT
...
sess_rates = smooth_causal(sess_rates)
```

iii. The trajectory shows the agent deciding that the paper used go-cue alignment, 10 ms bins, and causal Gaussian smoothing (steps 28, 45, 70). The agent's own earlier note that one script used `dt=1/200` was later overridden by this interpretation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only probes listed in the parsed meta scripts, drops clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}`, then drops units whose mean firing rate over the aligned window is `<= 1 Hz`. After concatenating retained probes, it drops entire sessions with fewer than 10 usable units.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10
...
cluid = [i for i, q in enumerate(qual) if q not in BAD_QUALITY]
...
mfr = sess_rates.mean(axis=(1, 2))
use = mfr > LOW_FR
...
if rates.shape[0] < MIN_UNITS:
    ...
    return None
```

iii. The trajectory shows the agent inferring this policy from the MATLAB pipeline and paper inclusion criteria, including an explicit decision to require at least 10 usable units per session and to include the JEB15 `tjM1` probe rather than restricting to ALM only (steps 45, 53, 70).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to go-cue onset by subtracting `bp.ev.goCue[trial]` from each spike's `trialtm` value before binning.

ii.
```python
ta = tt - gocue[tr]                     # align to go cue
```

iii. The trajectory consistently cites `goCue` as the alignment event from the authors' scripts and the task instructions (steps 8, 28, 45, 70).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural and time-varying output data use 10 ms bins over a 5 s window, producing 500 time bins per trial. There is no later rebinning step; the 10 ms grid is used throughout.

ii.
```python
DT = 0.01        # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
NT = len(TAXIS)
```

iii. The trajectory shows the agent settling on 10 ms bins as its interpretation of the reference scripts despite earlier notes about `dt=1/200` (steps 8, 28, 45, 69, 70).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read directly from a raw per-trial variable. It is a synthetic time axis defined from the chosen aligned bin grid around go cue; conceptually it is tied to the raw `goCue` event used for alignment.

ii.
```python
TAXIS = EDGES[:-1] + DT / 2
time_input = TAXIS.astype(np.float32).reshape(1, NT)
```

iii. The trajectory frames this as "time from go cue" rather than as a measured stream, i.e. the time basis implied by the alignment choice (steps 45, 69, 70).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes the centers of the 10 ms bins spanning -2.5 s to +2.5 s, casts them to `float32`, reshapes them to `(1, NT)`, and reuses that same array for every trial.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
...
time_input = TAXIS.astype(np.float32).reshape(1, NT)
input_sess = [time_input.copy() for _ in range(ntr)]
```

iii. The trajectory does not give a separate justification beyond using the neural time axis itself as the decoder input (steps 45, 69, 70).

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same time grid used to bin neural activity and interpolate the video streams, so each input timepoint corresponds to the same aligned bin used by the neural data.

ii.
```python
ta = tt - gocue[tr]
b = np.floor((ta - TMIN) / DT).astype(np.int64)
...
TAXIS = EDGES[:-1] + DT / 2
```

iii. The trajectory treats the time input as the shared aligned axis for the converted dataset rather than as an independent signal (steps 45, 69, 70).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the raw lick-event times `bp.ev.lickL` and `bp.ev.lickR` plus the trial's `goCue`, rather than from instructed side plus outcome.

ii.
```python
lickL = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
         for r in np.array(bp['ev/lickL']).ravel()]
lickR = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
         for r in np.array(bp['ev/lickR']).ravel()]
...
gc = gocue[t]
lt = lickL[t][lickL[t] > gc] if lickL[t].size else np.array([])
rt = lickR[t][lickR[t] > gc] if lickR[t].size else np.array([])
```

iii. The trajectory does not contain an extended explicit defense of this choice, but the docstring states the intended definition as the side of the first post-go-cue lick-port contact, consistent with the agent's inspection of lick-related helper functions (steps 33, 45).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each kept trial, the code finds the earliest left and right lick times that occur after the go cue. If neither exists, the class is `none` (`2`); otherwise the earlier side wins (`0` for left, `1` for right).

ii.
```python
lick_dir = np.full(len(trials), 2, dtype=np.int64)   # 2 = none
for i, t in enumerate(trials):
    gc = gocue[t]
    lt = lickL[t][lickL[t] > gc] if lickL[t].size else np.array([])
    rt = lickR[t][lickR[t] > gc] if lickR[t].size else np.array([])
    fl = lt.min() if lt.size else np.inf
    fr = rt.min() if rt.size else np.inf
    if np.isinf(fl) and np.isinf(fr):
        lick_dir[i] = 2
    elif fl <= fr:
        lick_dir[i] = 0
    else:
        lick_dir[i] = 1
```

iii. The trajectory implies the justification was to use a directly observed behavioral event rather than an inferred choice variable, but it does not spell out a longer argument.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from the Bpod per-trial `autowater` flag.

ii.
```python
early, autowater = get('early'), get('autowater')
```

iii. The trajectory shows the agent inspecting autowater fractions across sessions specifically to determine which sessions contained WC blocks and to define the context variable (steps 32, 39, 40, 41).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code converts `autowater == 1` to class `1` and everything else to class `0`. Its comments and `output_values` imply this means `0 = DR` and `1 = WC`.

ii.
```python
context = (autowater[trials] == 1).astype(np.int64)   # 0 = DR, 1 = WC
...
'output_values': [['left', 'right', 'none'],
                  ['DR', 'WC'],
                  ['incorrect', 'correct', 'ignore'],
```

iii. The trajectory shows the agent choosing `autowater` as a proxy for the WC context after inspecting session contents, but it does not separately justify the category order (steps 32, 39, 45).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial Bpod flags `hit`, `miss`, and `no`.

ii.
```python
hit, miss, no = get('hit'), get('miss'), get('no')
...
outcome = np.full(len(trials), 0, dtype=np.int64)
outcome[hit[trials] == 1] = 1
outcome[no[trials] == 1] = 2
```

iii. The trajectory states that ignore/no-response trials should be kept because the decoder must predict an `ignore` class, so the explicit `no` flag is used rather than dropping those trials (steps 30, 45, 70).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Trials default to class `0` (`incorrect`), then `hit` trials are relabeled as `1` (`correct`) and `no` trials as `2` (`ignore`).

ii.
```python
outcome = np.full(len(trials), 0, dtype=np.int64)     # 0 incorrect
outcome[hit[trials] == 1] = 1                         # 1 correct
outcome[no[trials] == 1] = 2                          # 2 ignore
```

iii. The trajectory consistently treats ignore trials as a retained decoder target rather than as excluded trials (steps 30, 45, 70).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` in the first camera view only, using the feature named `tongue`, together with that view's `frameTimes`, the session video offset, and per-trial `goCue`.

ii.
```python
traj = f['obj/traj']
...
def feat_speed(view, featnames, t):
    g, names = views[view]
    idxs = [names.index(n) for n in featnames if n in names]
    ...
    ft = np.array(f[np.array(g['frameTimes']).ravel()[t]]).ravel().astype(float)
    ts = np.array(f[np.array(g['ts']).ravel()[t]])
...
for i, t in enumerate(trials):
    tongue[i] = feat_speed(0, ['tongue'], t)
```

iii. The trajectory documents that the agent inspected tracking feature names and then chose side-camera tongue tracking; unlike the reference, it did not later decide to combine both tongue views (steps 37, 42, 45).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code linearly interpolates tracked x and y positions onto the neural time grid, computes simple discrete gradients of x and y on that grid, converts them to speed magnitude, and uses NaNs when the feature is absent. It does not apply the reference solution's likelihood threshold, contiguous-run handling, Gaussian smoothing, dual-view normalization, or bin-mean aggregation from frame space.

ii.
```python
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
vx = np.gradient(x)
vy = np.gradient(y)
sp.append(np.sqrt(vx ** 2 + vy ** 2))
```

iii. The trajectory only briefly comments that tongue should be class 2 when not visible and that the result looked reasonable in distribution checks; it does not defend the omitted likelihood/smoothing/dual-view steps (step 59).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After continuous tongue speed is computed for all trials of a session, the AI applies a per-session median split over finite values. Values below the 50th percentile become class `0`, values at or above it become class `1`, and NaNs become class `2`.

ii.
```python
def discretize(x):
    out = np.full(x.shape, 2, dtype=np.int64)
    finite = np.isfinite(x)
    if np.any(finite):
        thresh = np.percentile(x[finite], 50)
        out[finite & (x < thresh)] = 0
        out[finite & (x >= thresh)] = 1
    return out

tongue = discretize(res['tongue'])
```

iii. The trajectory explicitly notes that the prompt requested a per-session 50th-percentile split with class 2 for not-visible bins (steps 45, 59).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code first corrects frame times by a session-wide video/behavior offset, then subtracts each trial's go cue, and finally interpolates the tongue trajectory directly onto the neural time axis.

ii.
```python
vidshift = video_offset(f)
...
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
```

iii. The trajectory shows that the agent intentionally reused the authors' video-offset idea (`findVideoOffset.m`) and align-to-go-cue logic, but then chose interpolation to the neural axis rather than frame-bin averaging (steps 37, 45).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` in the second camera view, using both `top_paw` and `bottom_paw` features plus that view's `frameTimes`, the video offset, and per-trial `goCue`.

ii.
```python
for i, t in enumerate(trials):
    tongue[i] = feat_speed(0, ['tongue'], t)
    if nviews > 1:
        paw[i] = feat_speed(1, ['top_paw', 'bottom_paw'], t)
```

iii. In the trajectory, the agent explicitly states that it used both `top_paw` and `bottom_paw` because those are the default trajectory features listed for the bottom camera in the paper's scripts (step 59).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature present in the bottom view, the code interpolates x and y onto the neural axis, computes gradients on that axis, converts them to speed magnitude, and then averages across available paw features for each timepoint. It does not use the reference solution's likelihood thresholding, within-run smoothing, or single-feature choice.

ii.
```python
def feat_speed(view, featnames, t):
    ...
    for fi in idxs:
        x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
        y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
        vx = np.gradient(x)
        vy = np.gradient(y)
        sp.append(np.sqrt(vx ** 2 + vy ** 2))
    sp = np.array(sp)
    with np.errstate(invalid='ignore'):
        out = np.nanmean(sp, axis=0) if sp.shape[0] > 1 else sp[0]
```

iii. The trajectory's justification is limited to the note that both paw features appear in the authors' default feature list and that the resulting distributions looked reasonable (step 59).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. As for tongue velocity, the paw speed array is split at the per-session median of the finite values, with NaNs mapped to class `2`.

ii.
```python
paw = discretize(res['paw'])
```

```python
def discretize(x):
    ...
    thresh = np.percentile(x[finite], 50)
```

iii. The trajectory indicates this came from the decoder-task requirement for per-session 50th-percentile discretization (step 45).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned in the same way as tongue trajectories: subtract session-wide video offset, subtract the trial go cue, then interpolate to the neural time axis.

ii.
```python
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
```

iii. The trajectory treats paw and tongue alignment identically once the appropriate view is chosen (steps 37, 45, 59).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from a separate `motionEnergy_<session>.mat` file, specifically the nested `me['data']` array (with one extra `data` unwrap for some sessions), plus side-camera frame times from `obj.traj` and the session video offset / go cue for alignment.

ii.
```python
mefile = dsfile.replace('data_structure', 'motionEnergy')
me = sio.loadmat(mefile)['me']
data = me['data'][0, 0]
if data.dtype.names is not None and 'data' in data.dtype.names:
    data = data['data'][0, 0]
...
g = f[f['obj/traj'][0, 0]]
ftrefs = np.array(g['frameTimes']).ravel()
```

iii. The trajectory shows the agent debugging nested `me.data.data` structures and patching the loader after verification exposed missing motion-energy output for some JEB15 sessions (steps 56, 57, 58).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The continuous motion-energy trace is interpolated from frame times onto the neural time axis. If interpolation leaves NaNs at the edges but finite values exist elsewhere, those edge NaNs are filled with the nearest finite sample. The resulting continuous series is later median-split into categories.

ii.
```python
tt = ft - vidshift - gocue[t]
out[i] = interp_nan(TAXIS, tt, d)
...
if np.any(np.isnan(v)) and np.any(~np.isnan(v)):
    good = np.where(~np.isnan(v))[0]
    ...
    out[i] = np.where(np.isnan(v), v[choose], v)
```

iii. The trajectory justifies the nested-loader patch by reference to `loadMotionEnergy.m`, but does not provide a separate explicit defense of the interpolation and nearest-edge-fill behavior beyond trying to follow that MATLAB routine (steps 57, 58).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized with the same per-session median split used for tongue and paw velocity; finite values are split at the 50th percentile and NaNs become class `2` (`no_video`).

ii.
```python
me = discretize(res['me'])
...
'output_values': [
    ...,
    ['below_median', 'above_median', 'no_video']],
```

iii. The trajectory explicitly notes that the prompt required a three-class motion-energy output with a third "no video" category, even if rare (steps 45, 61, 62).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using side-camera frame times corrected by the session video offset and the trial go cue, then interpolated onto the same neural time grid used elsewhere.

ii.
```python
vidshift = video_offset(f)
g = f[f['obj/traj'][0, 0]]
...
tt = ft - vidshift - gocue[t]
out[i] = interp_nan(TAXIS, tt, d)
```

iii. The trajectory shows the agent intentionally reusing the same offset-and-go-cue alignment logic for motion energy as for video trajectories (steps 37, 45, 58).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases by returning NaNs that are later mapped to category `2`, or by skipping unusable trials within a stream. Trials with NaN `goCue` are dropped entirely. For video features, if frame times are missing or too short, the whole trial returns NaNs for that stream. Missing motion-energy files, mismatched motion-energy/frame-time lengths, or all-NaN frame times also return NaNs. The interpolation helper preserves NaNs when adjacent samples are missing, but motion energy additionally fills edge NaNs with the nearest finite value.

ii.
```python
keep = ... & ~np.isnan(gocue)
```

```python
if ft.size < 2 or np.all(np.isnan(ft)):
    return np.full(NT, np.nan)
```

```python
if not os.path.exists(mefile):
    return np.full((len(trials), NT), np.nan)
...
if d.size < 2 or ft.size != d.size or np.all(np.isnan(ft)):
    continue
```

```python
bad = np.isnan(y[idx]) | np.isnan(y[idx + 1])
out[inrange & bad] = np.nan
```

iii. The trajectory shows the agent paying attention to missing-video / no-video classes and debugging rare motion-energy corner cases, but it does not discuss the reference solution's more specific missing-frame handling or the late-recording trial issue (steps 56, 57, 58, 59, 61, 62).

## 11-a. What are the most time-consuming steps of the code?

i. The code is dominated by per-session loading and by Python loops over clusters and trials. The most expensive computed steps are likely the per-unit spike binning / smoothing in `load_session`, the per-trial interpolation and gradient calculations in `video_speeds`, and per-trial interpolation in `motion_energy`.

ii.
```python
for ui, cid in enumerate(cluid):
    ...
    counts = np.bincount(idx, minlength=len(trials) * NT).reshape(len(trials), NT)
...
sess_rates = smooth_causal(sess_rates)
```

```python
for i, t in enumerate(trials):
    tongue[i] = feat_speed(0, ['tongue'], t)
    if nviews > 1:
        paw[i] = feat_speed(1, ['top_paw', 'bottom_paw'], t)
```

iii. The trajectory does not contain a dedicated efficiency analysis. This assessment is inferred from the structure of the implemented code.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the convolution loop inside `smooth_causal`, the cluster loop in `load_session`, the per-trial loops in `video_speeds` and `motion_energy`, and the per-trial output-assembly loop in `main`. Unlike the reference solution, the AI does not use a single all-trials histogram call for spike counting.

ii.
```python
for i in range(flat.shape[0]):
    o[i] = np.convolve(flat[i], KERN, mode='same')
```

```python
for ui, cid in enumerate(cluid):
    ...
```

```python
for i, t in enumerate(trials):
    ...
```

iii. The trajectory does not justify these loops as deliberate tradeoffs; they appear to be straightforward implementation choices.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats some work across functions and trials. In particular, `video_offset(f)` is recomputed separately in `video_speeds` and `motion_energy`; interpolation is performed independently for each feature and trial; and the shared constant input / per-trial constant outputs are copied trial by trial when assembling the final dataset.

ii.
```python
def video_speeds(f, trials, gocue):
    vidshift = video_offset(f)
    ...

def motion_energy(dsfile, f, trials, gocue):
    ...
    vidshift = video_offset(f)
```

```python
input_sess = [time_input.copy() for _ in range(ntr)]
...
for i in range(ntr):
    out = np.empty((6, NT), dtype=np.int64)
    out[0] = res['lick_dir'][i]
    ...
```

iii. The trajectory does not mention this repeated work explicitly; it is visible in the code structure.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI avoids the reference solution's full recursive materialization of each `obj`, but it still does some unnecessary work: it reads `L` but never uses it, computes direct region labels even though almost all retained units are ALM, and repeatedly materializes trialwise copies of constant time/context labels because the target format requires per-trial arrays. It also interpolates motion energy and then fills edge NaNs even though the final output discards the continuous values after median discretization.

ii.
```python
R, L = get('R'), get('L')
```

```python
if np.any(np.isnan(v)) and np.any(~np.isnan(v)):
    ...
    out[i] = np.where(np.isnan(v), v[choose], v)
```

iii. The trajectory does not call out unnecessary work directly. This section is inferred from the final script.
