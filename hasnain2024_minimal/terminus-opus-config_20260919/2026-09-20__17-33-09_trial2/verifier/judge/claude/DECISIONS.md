# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI dynamically parses the authors' MATLAB loading scripts (`load<ANM>_ALMVideo.m` files) in `/app/code/DataLoadingScripts/Recording and video/` to discover sessions. It uses regex to extract animal name, date, and probe numbers from uncommented lines. It then cross-references against files actually present on disk in the two data directories (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). Sessions are loaded using either `h5py` (for v7.3/HDF5 files) or `scipy.io.loadmat` (for v7 files), with format detection based on reading the file header. Only fields needed for conversion are extracted (not the entire object tree). Motion energy is loaded from a separate `motionEnergy_<anm>_<date>.mat` file using `scipy.io.loadmat`.

ii.
```python
def parse_sessions():
    """Read the paper's per-animal loading scripts -> [(anm, date, [probes])]."""
    out = []
    for fn in sorted(glob.glob(os.path.join(LOADSCRIPT_DIR, '*.m'))):
        anm, cur = None, None
        for line in open(fn):
            s = line.strip()
            if s.startswith('%'):
                continue
            m = re.search(r"\.anm\s*=\s*'([^']+)'", s)
            if m:
                anm = m.group(1)
            m = re.search(r"\.date\s*=\s*'([^']+)'", s)
            if m:
                if cur is not None:
                    out.append(cur)
                cur = {'anm': anm, 'date': m.group(1), 'probe': [1]}
            m = re.search(r"\.probe\s*=\s*\[?([0-9 ,]+)\]?\s*;", s)
            if m and cur is not None:
                cur['probe'] = [int(x) for x in re.findall(r'\d+', m.group(1))]
        if cur is not None:
            out.append(cur)
    return out
```

```python
def is_v73(fn):
    with open(fn, 'rb') as f:
        return b'MATLAB 7.3' in f.read(128)

def load_session(fn, probes, traj_feats):
    if is_v73(fn):
        return _load_v73(fn, probes, traj_feats)
    return _load_v7(fn, probes, traj_feats)
```

iii. The agent examined the authors' loading scripts and data directories to determine which sessions to include. It parsed the `.m` files programmatically rather than hard-coding the session list. It noted that commented-out sessions in the load scripts should be excluded.

## 1-b. How are the data split into subjects?

i. The animal ID is extracted from the session info (`sess['anm']`). Subjects are collected as the sorted set of unique animal names. `subject_idx` maps each session to its subject index.

ii.
```python
subjects = sorted({r['anm'] for r in res})
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subjects.index(r['anm']) for r in res]),
    ...
}
```

iii. The animal name is parsed from the load scripts (the `anm` field). This is functionally equivalent to extracting it from the filename.

## 1-c. How are the data split into sessions?

i. One session corresponds to one data file (`data_structure_<anm>_<date>.mat`). The AI discovers 44 sessions across both task folders (25 fixed-delay and 19 randomized-delay), treating them uniformly. Sessions with fewer than 10 units after quality filtering are excluded (`MIN_UNITS = 10`).

ii.
```python
sessions = parse_sessions()
sessions = [s for s in sessions if find_files(s['anm'], s['date'])[0] is not None]
...
if use.sum() < MIN_UNITS:
    return None
```

iii. The AI cross-referenced parsed session lists with available data files, matching the paper's counts.

## 1-d. How are the data split into trials?

i. Trials are defined by the Bpod behavioral data fields, which have `Ntrials` entries. Each trial has one go cue, one outcome, etc. The AI uses `Ntrials` to limit array lengths and indexes trials by their original position.

ii.
```python
N = d['Ntrials']
gocue = d[ALIGN_EVENT]
keep = (d['early'] == 0) & (d['stim'] == 0) & np.isfinite(gocue)
...
keep = keep[:N]
trials = np.where(keep)[0]
```

iii. The Bpod data structure directly defines trials, so no trial boundary inference is needed.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) early-lick trials (`bp.early == 1`) are excluded; (2) photostimulation trials (`bp.stim.enable == 1`) are excluded; (3) trials where the go cue is NaN are excluded. Additionally, after neural processing, trials with zero spikes across all units are dropped (addressing sessions where the ephys recording ended before the behavioral session).

ii.
```python
keep = (d['early'] == 0) & (d['stim'] == 0) & np.isfinite(gocue)
...
# Drop trials in which no unit fired a single spike.
nonempty = np.any(rates != 0, axis=(0, 1))
if n_no_ephys:
    rates = rates[:, :, nonempty]
    trials = trials[nonempty]
```

iii. The agent observed that early-lick and photostim trials are excluded from all analyses in the paper. The zero-spike trial filter was added after noticing that 2 sessions had trailing trials with no neural data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters from `obj.clu{probe}`, specifically the `trial` (1-based trial number for each spike), `trialtm` (spike time relative to trial start), and `quality` (manual curation label) fields. The go cue times from `bp.ev.goCue` are used for temporal alignment.

ii.
```python
units.append({'quality': q,
              'trial': np.atleast_1d(c.trial).astype(int),
              'trialtm': np.atleast_1d(c.trialtm).astype(float),
              'probe': p})
```

iii. The agent followed the reference code's approach of using `trialtm` (spike time in trial) and aligning to the go cue.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 10 ms time bins spanning -2.5 to 2.5 s from the go cue (500 bins total), converted to firing rates (Hz) by dividing by `DT`, then smoothed using a causal half-Gaussian kernel (port of `mySmooth.m`) with `N=15` bins and `reflect` boundary condition.

ii.
```python
DT = 0.01             # s, params.dt = 1/100
SMOOTH = 15           # params.smooth
...
cnt = np.zeros((NT, len(trials)))
for j in range(len(trials)):
    if ends[j] > starts[j]:
        cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
rates[:, iu, :] = my_smooth(cnt / DT).astype(np.float32)
```

```python
def my_smooth(x, N=SMOOTH, bctype=BCTYPE):
    """Port of utils/mySmooth.m: causal half-Gaussian smoothing along axis 0."""
    kern = gausswin(N)
    kern[:N // 2] = 0          # causal
    kern = kern / kern.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xf)
    return out[trim:]
```

iii. The agent studied `mySmooth.m` and `getDefaultParams.m` and ported the causal half-Gaussian smoothing kernel. It chose `params.dt = 1/100 = 10ms` based on the scripts it examined.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Clusters with quality labels in `('garbage', 'gabrga', 'noisy', 'real?')` are excluded (note: `poor` is NOT excluded). (2) Units with mean firing rate <= 1 Hz are removed. Additionally, sessions with fewer than 10 remaining units are excluded entirely.

ii.
```python
QUAL_EXCLUDE = ('garbage', 'gabrga', 'noisy', 'real?')
...
if q in QUAL_EXCLUDE:
    continue
...
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
if use.sum() < MIN_UNITS:
    return None
```

iii. The agent followed the reference code's `findClusters.m` for the quality exclusion list and the paper's 1 Hz firing rate threshold. The 10-unit minimum per session follows the paper's statement that "recording sessions were included for analysis only if they had at least 10 units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue: `trialtm` (spike time relative to trial start) is subtracted by the go cue time of that trial (`goCue[trial-1]`), putting spikes in seconds from go cue onset. Spikes are then histogrammed into the time bins.

ii.
```python
tm = u['trialtm'] - gocue[np.clip(tt - 1, 0, N - 1)]   # trialtm_aligned
...
cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
```

iii. This follows the reference's `alignSpikes.m` approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (`DT = 0.01`), producing 500 time bins over the -2.5 to 2.5 s window. No rebinning is applied — spikes are directly counted into these bins.

ii.
```python
DT = 0.01             # s, params.dt = 1/100
...
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]     # bin centres
NT = len(TAXIS)
```

iii. The AI chose `params.dt = 1/100` based on several scripts that use this value. However, the reference uses `params.dt = 1/200 = 5ms` (1000 bins).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is defined by the time axis itself — the bin centres of the time grid aligned to the go cue. It is not derived from any raw data variable beyond the go cue event that defines the alignment.

ii.
```python
TAXIS = (EDGES + DT / 2)[:-1]     # bin centres, obj.time
...
time_in = TAXIS.astype(np.float32)[None, :]
```

iii. The time from go cue is a constructed variable representing the decoder's temporal context.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond constructing the time axis. The bin centres are computed from the edges of the time grid.

ii.
```python
TAXIS = (EDGES + DT / 2)[:-1]
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis IS the neural binning grid itself. The same `EDGES` array defines both the spike histogram bins and the time input, so they are inherently aligned.

ii.
```python
cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
...
time_in = TAXIS.astype(np.float32)[None, :]
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.hit`, `bp.miss`, `bp.no`, `bp.R`, and `bp.L`. The hit/miss/no flags indicate trial outcome, and R/L indicate the instructed side.

ii.
```python
hit, miss, no = d['hit'] > 0, d['miss'] > 0, d['no'] > 0
R, L = d['R'] > 0, d['L'] > 0
```

iii. The agent used the combination of instructed side and outcome to derive lick direction, since lick direction itself is not directly recorded.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit on a right trial means a right lick; a miss on a left trial also means a right lick (licked the wrong side). The `no` flag indicates ignore trials (no lick). Codes: left=0, right=1, none=2.

ii.
```python
right_lick = (R & hit) | (L & miss)
lick_dir = np.where(no, 2, np.where(right_lick, 1, 0))   # 0 left, 1 right, 2 none
```

iii. The agent derived lick direction from hit/miss and instructed side, which is logically equivalent to the reference approach.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater=1 indicates water-cued (WC) context; autowater=0 indicates delayed-response (DR) context.

ii.
```python
aw = d['autowater'] > 0
context = np.where(aw, 0, 1)   # 0 WC, 1 DR
```

iii. The `autowater` field is a direct proxy for WC blocks as noted in the tutorial.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: autowater trials become WC (0), the rest DR (1).

ii.
```python
context = np.where(aw, 0, 1)   # 0 WC, 1 DR
```

iii. Matches the prompt's specification of WC and DR contexts.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`, `bp.miss`, and `bp.no`. These are the three mutually exclusive trial outcome flags.

ii.
```python
hit, miss, no = d['hit'] > 0, d['miss'] > 0, d['no'] > 0
outcome = np.where(no, 2, np.where(hit, 1, 0))   # 0 incorrect, 1 correct, 2 ignore
```

iii. The agent reads all three outcome fields directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct relabelling: miss->incorrect (0), hit->correct (1), no->ignore (2).

ii.
```python
outcome = np.where(no, 2, np.where(hit, 1, 0))
```

iii. Matches the prompt's specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from DeepLabCut tracking in `obj.traj`. The AI uses only the side camera (view 0) tongue feature (`'tongue'`). The `frameTimes`, `ts` (x, y, likelihood), and `NdroppedFrames` fields are used. The video offset is computed from `sglx.bitcode.bitstart`, `sglx.fs`, and `bp.ev.bitStart`.

ii.
```python
TONGUE_VIEW, TONGUE_FEATS = 0, ['tongue']
...
traj_feats = {TONGUE_VIEW: list(TONGUE_FEATS), PAW_VIEW: list(PAW_FEATS)}
```

iii. The agent identified the tongue as tracked in the side camera. It did not use the bottom camera's `top_tongue` feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The DLC x,y positions are interpolated onto the 10 ms time axis using linear interpolation (`interp_nan`). The velocity is computed as `np.gradient(x)/DT` and `np.gradient(y)/DT`, and the speed is the magnitude `sqrt(vx^2 + vy^2)`. NaN positions (where the tongue is not detected) produce NaN velocities. The speed is then discretized at the session's 50th percentile; NaN bins are assigned category 2 ("not visible").

ii.
```python
x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
y = interp_nan(TAXIS, tt, tr['xy'][:, 1, fi])
vx = np.gradient(x) / DT
vy = np.gradient(y) / DT
sp.append(np.sqrt(vx ** 2 + vy ** 2))
```

```python
def discretise(x):
    vis = np.isfinite(x)
    out = np.full(x.shape, 2, dtype=np.int64)
    if vis.sum() > 0:
        thr = np.percentile(x[vis], 50)
        out[vis] = (x[vis] >= thr).astype(np.int64)
    return out
```

iii. The agent followed the reference code's `findVelocity.m` approach of computing velocity as the gradient of position, expressed as a speed scalar for discretization. It used interpolation onto the time axis rather than binning from frame times.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session's 50th percentile (median) of all finite (visible) tongue speed values is used as the threshold. Values below the threshold get category 0, values >= threshold get category 1, and NaN (not visible) bins get category 2.

ii.
```python
def discretise(x):
    vis = np.isfinite(x)
    out = np.full(x.shape, 2, dtype=np.int64)
    if vis.sum() > 0:
        thr = np.percentile(x[vis], 50)
        out[vis] = (x[vis] >= thr).astype(np.int64)
    return out
```

iii. Matches the prompt's specification: 0 for < 50th percentile, 1 for >= 50th percentile, 2 for not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset and go cue time, then positions are linearly interpolated onto the same 10 ms time axis used for neural data. The video offset is computed as `median(sglx.bitcode.bitstart / fs) - median(bp.ev.bitStart)`.

ii.
```python
vidshift = (np.median(d['bitstart_sglx']) / d['fs']) - np.median(d['bitStart'])
...
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
```

iii. The agent followed the reference's `findVideoOffset.m` approach for clock correction. However, it used `median` instead of `mode` for the bitcode times.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from the bottom camera (view 1) tracking, using both `top_paw` and `bottom_paw` features. The same DLC tracking fields (ts, frameTimes) and video offset are used.

ii.
```python
PAW_VIEW, PAW_FEATS = 1, ['top_paw', 'bottom_paw']
```

iii. The agent chose to use both paw features from the bottom camera, averaging them together.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: x,y positions are interpolated onto the time axis, velocity computed via gradient, speed as magnitude. Both `top_paw` and `bottom_paw` speeds are computed and averaged (using `nanmean` to handle NaNs where one paw may not be tracked).

ii.
```python
for fi in feats_idx:
    x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
    y = interp_nan(TAXIS, tt, tr['xy'][:, 1, fi])
    vx = np.gradient(x) / DT
    vy = np.gradient(y) / DT
    sp.append(np.sqrt(vx ** 2 + vy ** 2))
sp = np.vstack(sp)
spd = np.where(np.all(np.isnan(sp), axis=0), np.nan, np.nanmean(sp, axis=0))
```

iii. The agent averaged both paw features to get a single paw speed.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same discretization as tongue: session 50th percentile of finite values as threshold, with category 2 for not visible.

ii.
```python
paw_cat, paw_thr = discretise(paw_speed)
```

iii. Matches the prompt's specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, positions interpolated onto the time axis.

ii.
```python
tt = ft - vidshift - gocue[t]
...
x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
```

iii. Same alignment approach as all video-derived variables.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from `motionEnergy_<anm>_<date>.mat` files. The motion energy data has one trace per trial with one value per camera frame. The data is accessed via `me.data` (with a guard for doubly-nested structs).

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
```

iii. The agent explored the motion energy file structure and handled the different wrapping levels.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated onto the time axis (same as tongue/paw). The frame times from the side camera are used. Then discretized at the session 50th percentile, with NaN bins assigned category 2 ("no video").

ii.
```python
if me_raw is not None and t < len(me_raw):
    tr = tv['trials'][t] if t < len(tv['trials']) else None
    if tr is not None and tr['frameTimes'].size == len(me_raw[t]):
        tt = tr['frameTimes'] - vidshift - gocue[t]
        me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])
```

iii. Motion energy is already a scalar per frame, so only interpolation and discretization are needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same discretization: session 50th percentile of finite values, with category 2 for "no video".

ii.
```python
me_cat, me_thr = discretise(me_t)
```

iii. Matches the prompt's specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by the video offset and go cue time, then motion energy is linearly interpolated onto the time axis.

ii.
```python
tt = tr['frameTimes'] - vidshift - gocue[t]
me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])
```

iii. Same alignment approach as tongue/paw. Uses the side camera frame times since motion energy is computed from that camera.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) If a trial has no trajectory data (`tr is None`) or invalid `NdroppedFrames`, the kinematic output is left as NaN (later mapped to "not visible" category 2). (2) If frame times are empty or all NaN, a synthetic time axis based on 400 Hz frame rate is used as a fallback. (3) If motion energy frame count doesn't match the side camera frame count, a synthetic timing is used. (4) Trials with zero spikes across all units are dropped.

ii.
```python
if tr is None or tr['Ndropped'].size == 0 or np.any(~np.isfinite(tr['Ndropped'])):
    continue          # video for this trial is unusable
ft = tr['frameTimes']
if ft.size == 0 or not np.any(np.isfinite(ft)):
    ft = (np.arange(tr['xy'].shape[0]) + 1) / 400.0
```

iii. The agent attempted to recover data where possible (e.g., synthetic frame times) rather than discarding trials.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files, particularly the large v7.3 HDF5 files. The agent used multiprocessing (`Pool`) to parallelize across sessions, which significantly reduced wall time (16s for all 44 sessions with 12 processes).

ii.
```python
if args.nproc > 1:
    with Pool(args.nproc, maxtasksperchild=1) as p:
        res = p.map(process_session, sessions, chunksize=1)
```

iii. The agent noted that loading dominated runtime and used multiprocessing to mitigate this.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit, per-trial spike binning loop: for each unit, spikes are sorted by trial and binned one trial at a time. This could potentially be vectorized using `np.histogram2d` (as the reference does). The per-trial kinematic processing loop could also potentially be vectorized, though the variable number of frames per trial makes this difficult.

ii.
```python
for iu, u in enumerate(units):
    ...
    for j in range(len(trials)):
        if ends[j] > starts[j]:
            cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
```

iii. The per-trial spike binning is the most obvious candidate for vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The trajectory data for each trial is accessed multiple times — once for the tongue view, once for the paw view, and once for motion energy alignment. The frame times and video offset calculation are done once per session, but the per-trial `interp_nan` calls are repeated for each feature within a trial.

ii.
```python
for j, t in enumerate(trials):
    for view, feats_idx, dest in ((tv, t_idx, 'tongue'), (pv, p_idx, 'paw')):
        ...
    # motion energy (sampled with the video frames)
    if me_raw is not None and t < len(me_raw):
        ...
```

iii. The interpolation of frame times onto the time axis could be shared between features on the same camera view, though the computational cost is minor.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads some fields that are not ultimately used (e.g., `NdroppedFrames`, probe locations `loc`). The `interp_nan` function performs linear interpolation of x,y positions onto the time axis, which introduces interpolated values between frames — this is an extra processing step that binning/averaging (as the reference does) would avoid. The code also loads the `L` field from `bp` even though it only uses `R` and the hit/miss flags to determine lick direction.

ii.
```python
for k in ['hit', 'miss', 'no', 'early', 'autowater', 'R', 'L']:
    out[k] = np.array(bp[k]).ravel().astype(float)
...
out['loc'] = _probe_locs_v73(f, o, probes)
```

iii. These are minor inefficiencies that don't materially affect the output.
