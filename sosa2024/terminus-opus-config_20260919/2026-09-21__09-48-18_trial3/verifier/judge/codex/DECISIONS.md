# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file under `/app/data/sub-*/*.nwb`, then reads each session directly with `h5py`. Within each file it loads the behavioral time series, reward timestamps, imaging metadata, ROI curation table, and per-plane fluorescence / neuropil arrays.

ii.
```python
files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
```

```python
def load_session(fn):
    with h5py.File(fn, 'r') as f:
        beh = f['processing/behavior/BehavioralTimeSeries']
        g = lambda k: beh[k]['data'][()].astype(np.float64)
        d = dict(
            file=fn,
            subject=f['general/subject/subject_id'][()].decode(),
            session_id=f['general/session_id'][()].decode(),
            scene=f['identifier'][()].decode().split('/')[-1],
            date=f['identifier'][()].decode().split('/')[-2],
            imaging_rate=float(f['general/optophysiology/ImagingPlane/imaging_rate'][()]),
            location=f['general/optophysiology/ImagingPlane/location'][()].decode(),
            pos=g('position'), speed=g('speed'), lick=g('lick'),
            rzone_flag=g('reward_zone'), env=g('environment'),
            trialnum=g('trial number'), autoreward=g('autoreward'),
            scanning=g('scanning'),
            t=beh['position']['timestamps'][()].astype(np.float64),
            reward_t=beh['Reward']['timestamps'][()].astype(np.float64),
            starts=np.where(g('trial_start') > 0)[0],
            stops=np.where(g('teleport') > 0)[0],
        )
```

iii. In `CONVERSION_NOTES.md`, the AI says the NWB files are one file per `(subject, session)` and that the behavioral series are already aligned to imaging frames, so direct HDF5 access is sufficient and avoids heavier `pynwb` loading.

## 1-b. How are the data split into subjects?

i. Subjects are split by NWB file metadata, with one subject id read from each file and later deduplicated across sessions. The file glob also assumes the dataset is organized by `sub-*` directories.

ii.
```python
d = dict(
    file=fn,
    subject=f['general/subject/subject_id'][()].decode(),
    session_id=f['general/session_id'][()].decode(),
    ...
)
```

```python
subjects = sorted({s['subject'] for s in sessions}, key=lambda x: int(x[1:]))
```

iii. The notes explicitly state that the DANDI release contains 11 subject directories and that `general/subject/subject_id` matches the released mouse ids (`m3`, `m4`, ..., `m19`).

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session ids are read from file metadata and the top-level dataset is organized as one list entry per converted session.

ii.
```python
files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
```

```python
d = dict(
    ...
    session_id=f['general/session_id'][()].decode(),
    ...
)
```

```python
data = dict(
    neural=[s['neural'] for s in sessions],
    input=[s['input'] for s in sessions],
    output=[s['output'] for s in sessions],
    ...
)
```

iii. The notes say the NWB release contains one file per `(subject, session)` and that this matches the paper's per-day session structure.

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start` and `teleport` events. The AI stores per-trial windows using the reference-style frame interval `[trial_start - 1, teleport - 1)`, not `[trial_start, teleport)`.

ii.
```python
starts=np.where(g('trial_start') > 0)[0],
stops=np.where(g('teleport') > 0)[0],
```

```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    sl = slice(s - 1, e - 1)
    T = (e - 1) - (s - 1)
    ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
```

iii. The notes justify this as matching `preprocessing.dff` and `glmUtils.get_timeseries_data`, which the AI interpreted as using `[trial_start-1, teleport-1)` everywhere.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops four kinds of trials or sessions: the first lap of every session because previous-trial outcome is undefined, trials with lick-sensor errors, trials with non-finite neural events, and sessions left with fewer than two usable trials.

ii.
```python
lick_error[i] = (np.sum(L > 2) / len(L)) > LICK_ERROR_THRESH
```

```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        continue
    if lick_error[i]:
        continue
    ...
    if not np.all(np.isfinite(ev)):
        print('  WARNING %s trial %d: non-finite events, trial dropped'
              % (os.path.basename(fn), i))
        continue
```

```python
sessions = [r[0] for r in results if r[0] is not None and len(r[0]['neural']) >= 2]
```

iii. The notes say lick-error laps were dropped because lick is a required decoder output, and first laps were dropped because previous outcome is undefined there. Non-finite event checks and the `>= 2 trials` rule are justified as decoder-format safeguards.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from raw suite2p fluorescence and neuropil traces in the NWB file, after restricting to ROIs marked `iscell`.

ii.
```python
ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = ps['iscell'][()][:, 0].astype(bool)
planes = sorted(int(p) for p in np.unique(ps['planeIdx'][()]))
Fl, Nl = [], []
for p in planes:
    rois = f['processing/ophys/Fluorescence/plane%d/rois' % p][()]
    keep = iscell[rois]
    Fl.append(f['processing/ophys/Fluorescence/plane%d/data' % p][:, keep].T)
    Nl.append(f['processing/ophys/Neuropil/plane%d/data' % p][:, keep].T)
```

iii. The notes explicitly reject the NWB `Deconvolved` series as the wrong signal for this task, because the paper's analyses recompute dF/F and then deconvolve it.

## 2-b. How is the `neural` data processed?

i. The AI computes neuropil-corrected per-trial dF/F from `F` and `Fneu`, builds a maximin baseline within each trial, smooths dF/F, and then runs OASIS deconvolution to obtain events. It also adds a non-reference numerical-stability rule that flags cells whose baseline collapses toward zero.

ii.
```python
f_ -= NEU_COEF * fneu_
...
for start, stop in zip(starts, stops):
    sl = slice(start - 1, stop - 1)
    f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
    base = gaussian_filter1d(f_[:, sl], BASELINE_SMOOTH_SIGMA, axis=-1)
    base = minimum_filter1d(base, MAXIMIN_WINDOW, axis=-1)
    base = maximum_filter1d(base, MAXIMIN_WINDOW, axis=-1)
    flow[:, sl] = base
...
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
...
for start, stop in zip(starts, stops):
    sl = slice(start - 1, stop - 1)
    dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH_SIGMA, axis=1)
    events[:, sl] = dcnv.oasis(
        np.ascontiguousarray(dff[:, sl], dtype=np.float32), OASIS_BATCH, TAU, fs)
```

```python
with np.errstate(invalid='ignore'):
    min_base = np.nanmin(flow, axis=1)
    scale = np.nanmedian(np.where(np.isnan(flow), np.nan, F), axis=1)
bad_baseline = ~(min_base > BASELINE_MIN_FRAC * scale)
```

iii. The notes say this is intended to port `reward_relative.preprocessing.dff` and the Methods section: neuropil subtraction with coefficient `0.7`, per-trial maximin baseline, sigma-2 smoothing, and OASIS deconvolution with `tau = 0.7`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered by three rules: keep only `iscell` ROIs, remove putative interneurons whose dF/F correlates with speed above `0.5`, and remove cells with unstable near-zero baselines.

ii.
```python
iscell = ps['iscell'][()][:, 0].astype(bool)
```

```python
valid = ~np.isnan(dff[0, :])
sp = S['speed'][valid]
X = dff[:, valid]
Xc = X - X.mean(axis=1, keepdims=True)
spc = sp - sp.mean()
denom = np.sqrt((Xc ** 2).sum(axis=1) * (spc ** 2).sum())
speed_corr = (Xc @ spc) / np.where(denom == 0, np.nan, denom)
is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH
keep_cells = (~is_int) & (~bad_baseline)
```

iii. The notes say the first two filters match the paper's curation, while the `bad_baseline` filter was added after the full run exposed exploding dF/F values in a tiny number of cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to trial start conceptually, but the stored per-trial neural arrays begin one frame before the trial-start flag because it uses `[trial_start - 1, teleport - 1)`.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
```

```python
metadata=dict(
    ...
    temporal_alignment_event='trial start (teleport into the linear track, position 0 cm)',
    off_start=-float(np.mean([s['dt'] for s in sessions])),
    off_end=None,
    ...
)
```

iii. The notes defend the negative `off_start` as a deliberate consequence of following the paper's windowing in `dff` / `glmUtils`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is one imaging frame, with no temporal rebinning. For two-plane recordings the per-plane sampling rate is `imaging_rate / n_planes`, but the stored trial arrays still keep the original frame clock.

ii.
```python
fs = S['imaging_rate'] / S['n_planes']
dt = float(np.median(np.diff(S['t'])))
```

```python
metadata=dict(
    ...
    time_bin_size=float(np.mean([s['dt'] for s in sessions]) * 1000.0),
    ...
)
```

iii. The notes repeatedly state that the behavioral series are already aligned to imaging frames at about `64.48 ms` per sample, so no extra resampling is necessary.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from the shared behavioral timestamps, specifically `position` timestamps loaded into `S['t']`.

ii.
```python
t=beh['position']['timestamps'][()].astype(np.float64),
```

```python
tt = S['t'][sl] - S['t'][s]
```

iii. The notes say all behavioral series share the imaging-frame timestamps, so any behavioral timestamp source would be equivalent.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI slices the trial window `[trial_start - 1, teleport - 1)` and subtracts the timestamp at `trial_start`, not the first timestamp in the stored slice. This produces a first sample of approximately `-dt`.

ii.
```python
sl = slice(s - 1, e - 1)
...
tt = S['t'][sl] - S['t'][s]
...
x[0] = tt
```

iii. The notes justify the negative first timepoint as part of the same reference-style trial window used for neural data.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the exact same frame slice as the neural events for each trial. The time input and neural matrix therefore share the same number of timepoints and same frame indices.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
...
tt = S['t'][sl] - S['t'][s]
...
x[0] = tt
```

iii. The notes say that all streams are already on the imaging frame clock, so alignment is achieved by slicing identical index ranges.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type comes from the behavioral `environment` time series.

ii.
```python
env=g('environment'),
```

```python
env_trial[i] = int(np.round(np.median(S['env'][sl])))
...
x[1] = env_trial[i]
```

iii. The notes identify `environment` as the NWB equivalent of the paper's `morph` / environment code.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the median environment value within the trial, rounds it to an integer, and then stores that same value across all timepoints of the converted trial.

ii.
```python
env_trial[i] = int(np.round(np.median(S['env'][sl])))
...
x[1] = env_trial[i]
```

iii. The notes say the environment value is constant within a lap, so reducing it to one per-trial value is just a robust way to represent that constant label.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the per-session trial loop index, not from the NWB `trial number` vector.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    x[2] = i
```

iii. The notes say this preserves the within-session lap index and keeps the reward-zone switch at lap `30` interpretable even after trial dropping.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond assigning the integer loop index to the whole trial.

ii.
```python
x[2] = i
```

iii. The notes describe this as a per-trial contextual input stored as a constant time series.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward timestamps and the reward-zone flag, via an intermediate per-trial `isreward` array.

ii.
```python
reward_frames = np.searchsorted(S['t'], S['reward_t'])
...
rzflag = S['rzone_flag'][sl] > 0
got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
isreward[i] = int(got_reward and np.any(rzflag))
```

```python
x[3] = isreward[i - 1]
```

iii. The notes say this mirrors the paper's `behavior.get_trial_types`, where a trial is rewarded only if reward delivery occurred while the reward zone was active.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes `isreward` for every trial, then shifts that array by one trial when constructing inputs. It drops the first trial entirely rather than assigning it a default previous outcome.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    isreward[i] = int(got_reward and np.any(rzflag))
```

```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        continue
    ...
    x[3] = isreward[i - 1]
```

iii. The notes explicitly justify dropping the first lap because "previous-trial outcome is undefined" there.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from per-frame position and a per-trial reward-zone interval inferred from the session's scene name, not from the noisy `reward_zone` time series itself.

ii.
```python
scene=f['identifier'][()].decode().split('/')[-1],
```

```python
def scene_reward_zones(scene, n_trials):
    ...
    coords = np.array([REWARD_ZONE_DICT[l] for l in labels], dtype=float)
    return coords, np.array(labels)
```

```python
pos = S['pos'][sl]
d_rz = reward_zone_distance(pos, rz_coords[i, 0], rz_coords[i, 1])
```

iii. The notes say this was chosen because the paper's reference code derives reward zones from the scene name and switch rule (`change_trial = 30`), and the AI validated the inferred zones against measured reward-zone-entry positions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed linear distance to the nearest point of the reward zone: negative before the zone, zero inside it, positive after it.

ii.
```python
def reward_zone_distance(pos, rz_start, rz_end):
    d = np.zeros_like(pos)
    before = pos < rz_start
    after = pos > rz_end
    d[before] = pos[before] - rz_start
    d[after] = pos[after] - rz_end
    return d
```

```python
d_rz = reward_zone_distance(pos, rz_coords[i, 0], rz_coords[i, 1])
```

iii. The notes say this choice matches the task wording "distance to any location in the reward zone", so the whole zone is encoded as distance `0`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is converted into 7 categories by explicit threshold comparisons matching the task bins.

ii.
```python
def digitize_rz_distance(d):
    out = np.full(d.shape, 3, dtype=np.int64)
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
    return out
```

```python
y[0] = digitize_rz_distance(d_rz)
```

iii. The notes say this manual digitization was used to preserve a dedicated `0 cm` class across the entire reward zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing position and reward-zone distance from the exact same per-trial frame slice used for neural events.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
...
pos = S['pos'][sl]
d_rz = reward_zone_distance(pos, rz_coords[i, 0], rz_coords[i, 1])
```

iii. The notes say inputs, outputs, and neural activity are all extracted from identical frame ranges, so no interpolation is needed.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the behavioral `position` series.

ii.
```python
pos=g('position'),
...
pos = S['pos'][sl]
```

iii. The notes describe the NWB `position` variable as the already aligned linear-track position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI takes the raw per-frame position values within each trial and bins them into five 90 cm track bins.

ii.
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
...
y[1] = np.digitize(pos, POSITION_EDGES)
```

iii. The notes say this is the task-specified discretization of the 450 cm track into equal-sized bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges `[90, 180, 270, 360]` produces five categories spanning the track.

ii.
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
...
y[1] = np.digitize(pos, POSITION_EDGES)
```

iii. The notes justify this as the direct implementation of the instruction's "5 equal-sized bins spanning the 450 cm track".

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same trial slice as the neural events, so it is frame-aligned without any resampling.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
...
pos = S['pos'][sl]
```

iii. The notes repeatedly state that the behavioral time series are already aligned to the imaging frame clock.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the behavioral `lick` time series.

ii.
```python
lick=g('lick'),
...
lick = S['lick'][sl]
```

iii. The notes describe this NWB variable as cumulative lick count per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the cumulative lick count at `>= 1` within each frame and separately excludes lick-error trials from the dataset.

ii.
```python
lick_error[i] = (np.sum(L > 2) / len(L)) > LICK_ERROR_THRESH
```

```python
lick = S['lick'][sl]
...
y[3] = (lick >= 1).astype(np.int64)
```

iii. The notes justify the trial exclusion with the paper's lick-sensor error rule and the binarization with the decoder task's binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the same trial frames as the neural events.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
...
lick = S['lick'][sl]
```

iii. The notes say the lick series already shares the imaging timestamps, so matching frame slices is enough.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the session identifier / scene name, which is parsed from the NWB `identifier` field.

ii.
```python
scene=f['identifier'][()].decode().split('/')[-1],
```

```python
rz_coords, rz_labels = scene_reward_zones(S['scene'], n_trials)
...
y[4] = RZ_LABELS.index(rz_labels[i])
```

iii. The notes say this mirrors the reference `behavior.get_reward_zones` logic and avoids using the noisier framewise `reward_zone` flag as the primary source of trial identity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI maps each scene to reward-zone labels `A/B/C`, applies the paper's switch-after-30-laps rule on switch sessions, and then converts the resulting label to integer `0/1/2`.

ii.
```python
CHANGE_TRIAL = 30
...
def scene_reward_zones(scene, n_trials):
    if '_to_' not in s:
        label = s[-1]
        labels = [label] * n_trials
    else:
        pre, post = s.split('_to_')
        first = pre[-1]
        second = post[-1]
        labels = [first] * min(CHANGE_TRIAL, n_trials)
        if n_trials > CHANGE_TRIAL:
            labels += [second] * (n_trials - CHANGE_TRIAL)
```

```python
y[4] = RZ_LABELS.index(rz_labels[i])
```

iii. The notes say this processing was validated against observed reward-zone-entry positions in the data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward timestamps plus the reward-zone flag, through the per-trial `isreward` variable.

ii.
```python
reward_frames = np.searchsorted(S['t'], S['reward_t'])
...
rzflag = S['rzone_flag'][sl] > 0
got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
isreward[i] = int(got_reward and np.any(rzflag))
```

```python
y[5] = isreward[i]
```

iii. The notes say this matches the paper's rewarded-trial definition rather than treating any reward timestamp alone as sufficient.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices with `searchsorted`, and each trial is labeled rewarded if a reward fell in the trial window while the reward zone was active. The label is then repeated across all timepoints of that trial.

ii.
```python
reward_frames = np.searchsorted(S['t'], S['reward_t'])
...
got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
isreward[i] = int(got_reward and np.any(rzflag))
...
y[5] = isreward[i]
```

iii. The notes explicitly tie this to `behavior.get_trial_types` from the reference code.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly: it truncates sessions where imaging has one extra frame, excludes unstable-baseline cells, drops trials with non-finite events, warns if the scene-derived reward zone does not match the observed reward-zone onset, and drops sessions with fewer than two remaining trials.

ii.
```python
if d['F'].shape[1] != n_beh:
    print('  NOTE %s: ophys has %d frames, behaviour %d; truncating to %d'
          % (os.path.basename(fn), d['F'].shape[1], n_beh, min(d['F'].shape[1], n_beh)))
    n = min(d['F'].shape[1], n_beh)
    d['F'] = d['F'][:, :n]
    d['Fneu'] = d['Fneu'][:, :n]
    ...
```

```python
bad_baseline = ~(min_base > BASELINE_MIN_FRAC * scale)
```

```python
if len(obs) and not (-1.0 <= zone_err <= 15.0):
    print('  WARNING %s: reward-zone mismatch, median(onset-zone_start)=%.1f cm'
          % (os.path.basename(fn), zone_err))
```

```python
if not np.all(np.isfinite(ev)):
    print('  WARNING %s trial %d: non-finite events, trial dropped'
          % (os.path.basename(fn), i))
    continue
```

iii. The notes describe the frame-count mismatch and unstable-baseline cells as problems found during the full conversion and fixed after diagnosis.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are reading large NWB / HDF5 arrays and computing dF/F plus OASIS events for each session. The AI also says HDF5 loading dominates wall-clock time, even after vectorization.

ii.
```python
t0 = time.time()
S = load_session(fn)
t_load = time.time() - t0
...
t1 = time.time()
dff, events, bad_baseline = compute_dff_and_events(S['F'], S['Fneu'], starts, stops, fs)
t_dff = time.time() - t1
```

```python
with ctx.Pool(min(args.nproc, len(files)), maxtasksperchild=1) as pool:
    for res in pool.imap(_worker, [(fn, fn in show_files, '/app') for fn in files]):
        results.append(res)
```

iii. In the notes, the AI gives timing tables showing that I/O and dF/F/OASIS dominate per-session cost, and that parallel session processing provides most of the wall-clock speedup.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI explicitly identifies the original per-cell interneuron-correlation loop as a vectorization target and rewrites it as a matrix operation. Remaining repeated trial loops are in `compute_dff_and_events` and in per-trial output assembly.

ii.
```python
X = dff[:, valid]
Xc = X - X.mean(axis=1, keepdims=True)
spc = sp - sp.mean()
denom = np.sqrt((Xc ** 2).sum(axis=1) * (spc ** 2).sum())
speed_corr = (Xc @ spc) / np.where(denom == 0, np.nan, denom)
```

```python
for start, stop in zip(starts, stops):
    ...

for i, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. The notes say the naive per-cell correlation loop was a clear inefficiency and that the remaining loops mostly reflect variable-length trial structure and the copied reference dF/F logic.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code loops over trial windows multiple times: once to copy trial samples into masked fluorescence arrays, once to build baselines, once to smooth / deconvolve, once to compute per-trial labels, and once to build output arrays. Optional plotting also recomputes several concatenated summaries.

ii.
```python
for start, stop in zip(starts, stops):
    f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
    fneu_[:, start - 1:stop - 1] = Fneu[:, start - 1:stop - 1]
...
for start, stop in zip(starts, stops):
    ...
    flow[:, sl] = base
...
for start, stop in zip(starts, stops):
    ...
    events[:, sl] = dcnv.oasis(...)
```

```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    isreward[i] = ...
    ...

for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    neural.append(...)
    inp.append(x)
    out.append(y)
```

iii. The notes emphasize that the AI removed a whole-dataset survey pass from the human solution, but its own session-level dF/F pipeline still revisits the same trial windows several times.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and retains full dF/F arrays even though only deconvolved events are saved, performs extensive session statistics and reward-zone sanity checks, and optionally creates large diagnostic plots. These are useful for validation but are not consumed by downstream decoder training.

ii.
```python
dff, events, bad_baseline = compute_dff_and_events(S['F'], S['Fneu'], starts, stops, fs)
```

```python
stats = dict(
    ...
    dff_min=float(np.nanmin(dff[keep_cells])), dff_max=float(np.nanmax(dff[keep_cells])),
    ev_max=float(np.nanmax(events[keep_cells])), dt=dt,
    ...
)
```

```python
if show:
    try:
        make_processing_plots(S, dff, events, keep_cells, rz_coords, rz_labels,
                             isreward, sess, plotdir)
```

iii. The notes frame these as deliberate sanity checks and debugging aids rather than part of the final dataset representation.
