# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the authors' `loadANM_ALMVideo.m` meta scripts to extract session names and probe assignments. It then globs `data_structure_*.mat` files from `/app/data/Ephys_Behavior` only (not `RandomizedDelay_Ephys_Behavior`) and matches them against the parsed meta entries. Each `.mat` file is opened with `h5py` (they are MATLAB v7.3 HDF5 files). Motion energy files are loaded with `scipy.io.loadmat`.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'

def parse_meta_scripts(meta_dir='/app/code/DataLoadingScripts/Recording and video'):
    sessions = {}
    for fn in sorted(glob.glob(os.path.join(meta_dir, '*.m'))):
        # ... parse anm, date, probe from uncommented lines
        sessions[(anm, cur_date)] = probes
    return sessions

def main():
    meta = parse_meta_scripts()
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_structure_*.mat')))
    for dsfile in files:
        # match against meta entries
        probes = meta.get((anm, date))
        if probes is None:
            continue
        res = load_session(dsfile, probes)
```

iii. From the trajectory (step 32/39): The AI decided to use only the `Ephys_Behavior` folder (fixed-delay sessions, 25 sessions from 9 mice), reasoning that the randomized-delay dataset is "a separate cohort analysed separately in the paper (Fig. 8) with a different trial structure, and its sessions contain essentially no water-cued trials."

## 1-b. How are the data split into subjects?

i. The animal name is extracted from the data structure filename using a regex match (`data_structure_<anm>_<date>.mat`). A running list of unique subjects is built as sessions are processed.

ii.
```python
m = re.match(r'data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat', base)
anm, date = m.group(1), m.group(2)
# ...
if anm not in subjects:
    subjects.append(anm)
data['subject_idx'].append(subjects.index(anm))
```

iii. The animal ID is reliably encoded in the filename.

## 1-c. How are the data split into sessions?

i. Each `.mat` file from the `Ephys_Behavior` folder that has a matching entry in the parsed meta scripts becomes one session. Only 25 sessions from the fixed-delay task are included; the 19 randomized-delay sessions are excluded.

ii.
```python
for dsfile in files:
    base = os.path.basename(dsfile)
    m = re.match(r'data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat', base)
    anm, date = m.group(1), m.group(2)
    probes = meta.get((anm, date))
    if probes is None:
        continue
```

iii. From the trajectory (step 32): The AI reasoned that the randomized-delay dataset has a different trial structure and essentially no water-cued trials, so merging would compromise the context variable.

## 1-d. How are the data split into trials?

i. Each trial corresponds to one row/entry in the Bpod (`obj.bp`) arrays. The trial count `Ntrials` is read, and trial indices are 0-based. Spike times carry per-trial assignment via `clu.trial`.

ii.
```python
ntrials = int(get('Ntrials')[0])
hit, miss, no = get('hit'), get('miss'), get('no')
# ...
keep = ((hit == 1) | (miss == 1) | (no == 1)) & (early != 1) & (stim != 1) & ~np.isnan(gocue)
trials = np.where(keep)[0]
```

iii. The Bpod table defines trials directly; the AI uses the completion flags (hit, miss, no) to identify completed trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three criteria: (1) only completed trials (`hit | miss | no`), (2) early-lick trials (`early != 1`) are excluded, and (3) photostimulation trials (`stim.enable != 1`) are excluded. Additionally, trials with NaN go cue times are excluded. Sessions with fewer than 2 remaining trials are dropped.

ii.
```python
keep = ((hit == 1) | (miss == 1) | (no == 1)) & (early != 1) & (stim != 1) & ~np.isnan(gocue)
trials = np.where(keep)[0]
if len(trials) < 2:
    return None
```

iii. From trajectory (step 45): The AI follows the paper's exclusion of early-lick and photostimulation trials. The completion check (`hit|miss|no`) ensures only valid trials are included.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters in `obj.clu{probe}`, specifically `trialtm` (spike times relative to trial start), `trial` (trial assignment), and `quality` (manual curation label). The go cue times `bp.ev.goCue` provide the alignment event.

ii.
```python
trialtm_refs = np.array(clu['trialtm']).ravel()
trial_refs = np.array(clu['trial']).ravel()
# For each cluster:
tt = np.array(f[trialtm_refs[cid]]).ravel()
tr = np.array(f[trial_refs[cid]]).ravel().astype(np.int64) - 1  # 0-based
ta = tt - gocue[tr]  # align to go cue
```

iii. The agent follows the standard pipeline: spike times from the sorted clusters aligned to the go cue.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 10 ms bins (DT=0.01) from -2.5 to +2.5 s relative to the go cue (500 time bins). Spike counts are converted to firing rates (divided by DT) and smoothed with a causal Gaussian kernel (window of 15 bins), replicating the authors' `mySmooth.m` function. The kernel is a `gausswin(15)` with the first `floor(15/2)=7` coefficients zeroed out to make it causal.

ii.
```python
DT = 0.01  # 10 ms bins
SMOOTH = 15

def gausswin(N, alpha=2.5):
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)

def causal_kernel(N=SMOOTH):
    k = gausswin(N)
    k[:N // 2] = 0
    return k / k.sum()

# Binning:
b = np.floor((ta - TMIN) / DT).astype(np.int64)
counts = np.bincount(idx, minlength=len(trials) * NT).reshape(len(trials), NT)
sess_rates[ui] = counts / DT

# Smoothing:
sess_rates = smooth_causal(sess_rates)
```

iii. From the trajectory (step 8/45): The AI explicitly follows the reference MATLAB pipeline parameters: `params.dt = 1/200` (which is 5 ms but the AI interpreted as 10 ms), causal Gaussian smoothing with window 15 as in `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) clusters with quality labels in `{garbage, gabrga, noisy, real?}` are excluded, and (2) units with mean firing rate <= 1 Hz are dropped. Additionally, sessions with fewer than 10 remaining units are excluded entirely.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10

qual = [mstr(f, r).strip().lower() for r in np.array(clu['quality']).ravel()]
cluid = [i for i, q in enumerate(qual) if q not in BAD_QUALITY]
# ...
mfr = sess_rates.mean(axis=(1, 2))
use = mfr > LOW_FR
sess_rates = sess_rates[use]
# ...
if rates.shape[0] < MIN_UNITS:
    return None
```

iii. From the trajectory (step 45): The AI follows the paper's quality exclusion (matching `findClusters.m`) and firing rate threshold. The MIN_UNITS=10 threshold follows the paper: "Recording sessions were included for analysis only if they had at least 10 units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `goCue[trial]` from each spike time. This aligns all trials to the go cue onset at time 0.

ii.
```python
ta = tt - gocue[tr]  # align to go cue
b = np.floor((ta - TMIN) / DT).astype(np.int64)
```

iii. This matches the reference's `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (DT=0.01 s), yielding 500 time bins from -2.5 to +2.5 s. No additional rebinning is applied after the initial binning.

ii.
```python
DT = 0.01  # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
NT = len(TAXIS)  # 500
```

iii. From the trajectory (step 45): The AI states it uses `dt=10 ms` matching `params.dt = 1/200`. However, `1/200 = 0.005 s = 5 ms`, not 10 ms. The AI appears to have misinterpreted the reference parameter.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself — the bin centres of the time grid, computed from the bin edges. No raw data variable is needed.

ii.
```python
time_input = TAXIS.astype(np.float32).reshape(1, NT)
input_sess = [time_input.copy() for _ in range(ntr)]
```

iii. This is the standard approach: the input is the time relative to the alignment event.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centres: `EDGES[:-1] + DT/2`. No further processing.

ii.
```python
TAXIS = EDGES[:-1] + DT / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the neural binning grid itself. Spike times are binned into the same edges, so the input time and neural data share the same time axis by construction.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the raw lick event times: `bp.ev.lickL` and `bp.ev.lickR`, which record the times of left and right lick-port contacts. The first lick after the go cue determines the direction.

ii.
```python
lickL = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
         for r in np.array(bp['ev/lickL']).ravel()]
lickR = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
         for r in np.array(bp['ev/lickR']).ravel()]
```

iii. From the trajectory (step 45 docstring): "lick_direction: side of the first lick-port contact after the go cue (left/right/none)."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the AI finds the first lick event after the go cue from both `lickL` and `lickR` arrays. Whichever occurs first determines the direction (left=0, right=1). If neither lick occurs, the trial is coded as none=2.

ii.
```python
for i, t in enumerate(trials):
    gc = gocue[t]
    lt = lickL[t][lickL[t] > gc] if lickL[t].size else np.array([])
    rt = lickR[t][lickR[t] > gc] if lickR[t].size else np.array([])
    fl = lt.min() if lt.size else np.inf
    fr = rt.min() if rt.size else np.inf
    if np.isinf(fl) and np.isinf(fr):
        lick_dir[i] = 2
    elif fl <= fr:
        lick_dir[i] = 0  # left
    else:
        lick_dir[i] = 1  # right
```

iii. The AI uses the actual lick event timestamps rather than deriving direction from the instructed side and outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The `autowater` field from `obj.bp`. Autowater=1 indicates water-cued (WC) context.

ii.
```python
autowater = get('autowater')
context = (autowater[trials] == 1).astype(np.int64)  # 0 = DR, 1 = WC
```

iii. From the trajectory docstring: "context: DR vs WC block, from obj.bp.autowater."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=1 maps to WC (coded as 1), autowater=0 maps to DR (coded as 0). Note: the AI codes DR=0, WC=1, which is the opposite of the reference (WC=0, DR=1).

ii.
```python
context = (autowater[trials] == 1).astype(np.int64)  # 0 = DR, 1 = WC
```

iii. The AI's output_values list shows `['DR', 'WC']`, so index 0=DR, index 1=WC.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags: `hit`, `miss`, and `no` from `obj.bp`.

ii.
```python
hit, miss, no = get('hit'), get('miss'), get('no')
outcome = np.full(len(trials), 0, dtype=np.int64)  # 0 incorrect
outcome[hit[trials] == 1] = 1                       # 1 correct
outcome[no[trials] == 1] = 2                        # 2 ignore
```

iii. The AI reads all three flags directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling: miss -> incorrect (0), hit -> correct (1), no -> ignore (2).

ii.
```python
outcome = np.full(len(trials), 0, dtype=np.int64)  # 0 incorrect
outcome[hit[trials] == 1] = 1                       # 1 correct
outcome[no[trials] == 1] = 2                        # 2 ignore
```

iii. Straightforward mapping of the three outcome flags.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, specifically the side camera (view 0) with the `tongue` feature. `frameTimes` provide the temporal reference, and `ts` provides x, y, and likelihood per frame.

ii.
```python
tongue[i] = feat_speed(0, ['tongue'], t)
# feat_speed accesses:
ft = np.array(f[np.array(g['frameTimes']).ravel()[t]]).ravel().astype(float)
ts = np.array(f[np.array(g['ts']).ravel()[t]])  # (feat, 3, frames)
```

iii. From the trajectory docstring: "tongue_velocity: speed of the DeepLabCut-tracked tongue (side camera)."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates the tongue x and y positions onto the neural time axis (10 ms bins) using linear interpolation (`interp_nan`), then computes velocity as `np.gradient` of the interpolated positions, and speed as `sqrt(vx^2 + vy^2)`. No smoothing is applied to the positions before differentiation. No likelihood filtering is applied (NaN values from the DLC tracking where the feature is not visible naturally propagate).

ii.
```python
def feat_speed(view, featnames, t):
    x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
    y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
    vx = np.gradient(x)
    vy = np.gradient(y)
    sp.append(np.sqrt(vx ** 2 + vy ** 2))
```

iii. The AI follows the approach of interpolating to the neural time axis and computing velocity from the interpolated positions.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of finite values. Below threshold = 0, at/above = 1, NaN (not visible) = 2.

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
```

iii. Matches the instructions' specification for discretization.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed from the bitcode (matching `findVideoOffset.m`). Frame times are corrected: `frameTimes - videoOffset - goCue[trial]`. The tongue position is then interpolated onto the neural time axis (`TAXIS`) using linear interpolation, so it shares the same time bins as the neural data.

ii.
```python
def video_offset(f):
    fs = np.array(f['obj/sglx/fs']).ravel()[0]
    bitstart_sglx = np.array(f['obj/sglx/bitcode/bitstart']).ravel().astype(float)
    bitstart_bp = np.array(f['obj/bp/ev/bitStart']).ravel().astype(float)
    return mode(bitstart_sglx) / fs - mode(bitstart_bp)

# In feat_speed:
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
```

iii. The AI uses the same video offset computation as the reference's `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera (view 1) tracking of `top_paw` and `bottom_paw` features from `obj.traj`.

ii.
```python
paw[i] = feat_speed(1, ['top_paw', 'bottom_paw'], t)
```

iii. From the trajectory docstring: "paw_velocity: speed of the DLC-tracked paws (bottom camera, mean of top_paw and bottom_paw)."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: x and y of both `top_paw` and `bottom_paw` are interpolated onto the neural time axis, velocity computed via `np.gradient`, speed as the magnitude. The speeds of the two paw features are averaged (nanmean).

ii.
```python
def feat_speed(view, featnames, t):
    for fi in idxs:
        x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
        y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
        vx = np.gradient(x)
        vy = np.gradient(y)
        sp.append(np.sqrt(vx ** 2 + vy ** 2))
    out = np.nanmean(sp, axis=0) if sp.shape[0] > 1 else sp[0]
```

iii. The AI averages the two paw features together rather than using only one.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile split, with NaN mapped to class 2.

ii.
```python
paw = discretize(res['paw'])
```

iii. Matches the instructions' specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction + go cue alignment, then linear interpolation onto the neural time axis.

ii.
```python
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
```

iii. Same approach as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The `motionEnergy_<anm>_<date>.mat` files, loaded with `scipy.io.loadmat`. The `me.data` field contains one trace per trial, with one value per camera frame.

ii.
```python
def motion_energy(dsfile, f, trials, gocue):
    mefile = dsfile.replace('data_structure', 'motionEnergy')
    me = sio.loadmat(mefile)['me']
    data = me['data'][0, 0]
    if data.dtype.names is not None and 'data' in data.dtype.names:
        data = data['data'][0, 0]
```

iii. Handles the nested `me.data.data` structure as in `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The motion energy values are interpolated onto the neural time axis using linear interpolation (`interp_nan`). Edge NaN values (time points outside the video range) are filled with nearest valid values, following `loadMotionEnergy.m`.

ii.
```python
out[i] = interp_nan(TAXIS, tt, d)
# fill edge NaNs with nearest value
v = out[i]
if np.any(np.isnan(v)) and np.any(~np.isnan(v)):
    good = np.where(~np.isnan(v))[0]
    idx = np.clip(np.searchsorted(good, np.arange(NT)), 0, len(good) - 1)
    prev = np.maximum(idx - 1, 0)
    choose = np.where(np.abs(good[idx] - np.arange(NT)) <= np.abs(good[prev] - np.arange(NT)),
                      good[idx], good[prev])
    out[i] = np.where(np.isnan(v), v[choose], v)
```

iii. The nearest-value filling follows the MATLAB pipeline's approach.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as other movement variables: per-session 50th percentile split, NaN mapped to class 2.

ii.
```python
me = discretize(res['me'])
```

iii. Matches the instructions' specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same video offset correction as the tracking data. Frame times from the side camera are corrected, and the motion energy trace is interpolated onto the neural time axis.

ii.
```python
g = f[f['obj/traj'][0, 0]]
ftrefs = np.array(g['frameTimes']).ravel()
ft = np.array(f[ftrefs[t]]).ravel().astype(float)
tt = ft - vidshift - gocue[t]
out[i] = interp_nan(TAXIS, tt, d)
```

iii. Uses side camera frame times for alignment, matching the reference approach.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials with NaN go cue times are excluded. (2) Trials where `frameTimes` have fewer than 2 frames or are all NaN get NaN velocity/ME, which becomes class 2 (not visible/no video). (3) The nested motionEnergy structure is unwrapped. (4) Sessions where no probe location is found default to 'ALM'. (5) Sessions with fewer than 10 units are dropped entirely.

ii.
```python
# NaN goCue exclusion:
keep = ... & ~np.isnan(gocue)

# Missing video:
if ft.size < 2 or np.all(np.isnan(ft)):
    return np.full(NT, np.nan)

# Nested ME structure:
if data.dtype.names is not None and 'data' in data.dtype.names:
    data = data['data'][0, 0]
```

iii. The AI handles missing data by either excluding trials/sessions or mapping to "not visible" categories.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 files is the most time-consuming step, as the AI noted in the trajectory. The full conversion processes 25 sessions.

ii.
```python
f = h5py.File(dsfile, 'r')
```

iii. File I/O dominates runtime for any data conversion pipeline of this kind.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for computing lick direction iterates over trials and checks lick event arrays individually. The per-trial loop in `video_speeds` and `motion_energy` could potentially be vectorized. The per-unit loop for spike binning computes counts one cluster at a time.

ii.
```python
for i, t in enumerate(trials):  # lick direction loop
    # ...

for i, t in enumerate(trials):  # video_speeds loop
    tongue[i] = feat_speed(0, ['tongue'], t)
    paw[i] = feat_speed(1, ['top_paw', 'bottom_paw'], t)

for ui, cid in enumerate(cluid):  # per-unit spike binning
    # ...
```

iii. Each trial has a different number of frames and spikes, making full vectorization difficult.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed once per session and reused. However, the `interp_nan` function is called separately for x and y of each feature on each trial, and each `feat_speed` call re-reads `frameTimes` from the HDF5 file for every feature.

ii.
```python
# In feat_speed, frameTimes are read each call:
ft = np.array(f[np.array(g['frameTimes']).ravel()[t]]).ravel().astype(float)
```

iii. Reading frameTimes multiple times per trial (once per feature) is redundant since all features on the same camera share the same frame times.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The edge NaN filling for motion energy adds values at time bins outside the video range, which then get discretized into real categories (0 or 1) rather than "no video" (2). This fabricates motion energy values where none were measured. Additionally, all HDF5 data for unused fields (like spike waveforms) is accessed when navigating the file.

ii.
```python
# Edge NaN filling in motion_energy:
if np.any(np.isnan(v)) and np.any(~np.isnan(v)):
    good = np.where(~np.isnan(v))[0]
    # ... nearest value fill
    out[i] = np.where(np.isnan(v), v[choose], v)
```

iii. The nearest-value fill follows `loadMotionEnergy.m`, but the reference solution does not apply this fill, instead letting edge bins remain NaN and map to the "no video" class.
