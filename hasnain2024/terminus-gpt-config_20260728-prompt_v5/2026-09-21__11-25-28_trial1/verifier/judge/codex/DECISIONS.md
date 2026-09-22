# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script only scans one folder, `/app/data/RandomizedDelay_Ephys_Behavior`, with `glob('data_structure_*.mat')`. Each session file is loaded with `load_session`, which tries `h5py` first and falls back to `scipy.io.loadmat`. Motion energy is loaded separately from `motionEnergy_<session>.mat` in that same folder via `load_motion_energy_sidecar`. It does not use the paper authors' curated session list and does not read `/app/data/Ephys_Behavior`.

ii.
```python
BASE = Path('/app/data/RandomizedDelay_Ephys_Behavior')

def load_session(path):
    try:
        with h5py.File(path, 'r') as h:
            return load_session_h5(path, h)
    except OSError:
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return load_session_mat(path, mat['obj'])

def convert(sample=False):
    files = sorted(BASE.glob('data_structure_*.mat'))
```

iii. `CONVERSION_NOTES.md` repeatedly frames the dataset as the randomized-delay folder only, and the trajectory shows the agent counting the `data_structure_*.mat` and `motionEnergy_*.mat` files in that folder and proceeding from that assumption.

## 1-b. How are the data split into subjects?

i. Subject IDs are derived from the filename, using the third underscore-delimited token from `data_structure_<subject>_<date>.mat`. Unique subject names are sorted, and each retained session gets a `subject_idx` entry from that mapping.

ii.
```python
'subject': path.stem.split('_')[2],

subjects = sorted({s['subject'] for s in sessions})
subject_map = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subject_map[sess['subject']])
```

iii. The agent's notes say the files are named `data_structure_<subject>_<date>.mat`, so the subject extraction logic is directly tied to that filename convention.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_*.mat` file in the randomized-delay folder. After loading, sessions are filtered to those with `clu` data and at least 10 retained units.

ii.
```python
files = sorted(BASE.glob('data_structure_*.mat'))
...
sessions = [load_session(f) for f in files]
sessions = [s for s in sessions if s['has_clu']]
sessions = [s for s in sessions if len(select_units(s)) >= 10]
```

iii. `CONVERSION_NOTES.md` says the agent found 22 session files in the randomized-delay folder and decided to exclude sessions lacking `clu` and sessions with fewer than 10 retained units.

## 1-d. How are the data split into trials?

i. The code treats trial numbers as `1..ntrials`. Neural activity for unit `u` in trial `tr` is selected with `u['trial'] == tr`, and output variables are read as one entry per trial from raw boolean arrays such as `R`, `L`, `hit`, `miss`, `no`, and `autowater`.

ii.
```python
for tr in range(1, session['ntrials'] + 1):
    arr = np.zeros((len(units), len(centers)), dtype=np.float32)
    for i, u in enumerate(units):
        mask = (u['trial'] == tr)
```

```python
ntr = session['ntrials']
lick = np.full(ntr, 2, dtype=np.int64)
lick[session['L']] = 0
lick[session['R']] = 1
```

iii. The notes document `bp.Ntrials` and per-trial `bp.ev.goCue`, `R`, `L`, `hit`, `miss`, and related fields, so the agent justified a direct one-index-per-trial split rather than reconstructing trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. In the final script, trials are not filtered for early licks, photostimulation, or trials extending beyond the recording. All trials in each retained session are kept.

ii.
```python
for tr in range(1, session['ntrials'] + 1):
    ...
inputs = build_inputs(centers, sess['ntrials'])
outputs = build_outputs(sess, len(centers))
```

iii. The planning notes mention that early trials would "likely" be excluded, but the final implementation never applies any trial-level mask. The only curation that survived into code is session/unit filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices are derived from each unit's `trial` and `trialtm` arrays, with `quality` and `tm` used for unit filtering. `goCue` is loaded into the session object but is not used inside `build_neural_trials`.

ii.
```python
'goCue': np.array(bp['ev']['goCue'][()]).squeeze().astype(float),
...
units.append({'quality': q, 'tm': tm, 'trial': trial, 'trialtm': trialtm, 'site': site})
```

```python
mask = (u['trial'] == tr)
if np.any(mask):
    counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
```

iii. `CONVERSION_NOTES.md` says the intent was "goCue-aligned spike binning", but the final code only bins `trialtm` by trial number and uses `tm` only for the firing-rate filter.

## 2-b. How is the `neural` data processed?

i. For each retained unit and trial, spikes are histogrammed into 20 ms bins from `-2.4` to `2.0`, converted to Hz by dividing by `dt`, and smoothed with a hand-built Gaussian kernel with `sigma_bins=2.0`.

ii.
```python
def build_neural_trials(session, units, tmin=-2.4, tmax=2.0, dt=0.02, smooth_sigma=2.0):
    edges = np.arange(tmin, tmax + dt, dt)
    centers = edges[:-1] + dt / 2
    ...
                counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
                arr[i] = gaussian_smooth(counts.astype(np.float32) / dt, smooth_sigma)
```

iii. The notes and README explicitly describe 20 ms bins over `[-2.4, 2.0]`, so the agent believed this matched the relevant reference parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if their normalized quality label is in `{'multi','fair','good','great','excellent'}` and their overall firing rate estimated from `tm` exceeds 1 Hz. Entire sessions are then dropped if fewer than 10 units remain.

ii.
```python
KEEP_QUALITY = {'multi', 'fair', 'good', 'great', 'excellent'}

def unit_fr_gt1(unit):
    tm = unit['tm']
    ...
    return (tm.size / dur) > 1.0

def select_units(session):
    keep = [u for u in session['units'] if u['quality'] in KEEP_QUALITY and unit_fr_gt1(u)]
    return keep
```

```python
sessions = [s for s in sessions if len(select_units(s)) >= 10]
```

iii. The notes say the agent reconciled the raw unit count to the paper by keeping curated non-garbage labels, imposing `>1 Hz`, and requiring sessions with at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. The final code does not realign spikes to go cue. It bins `trialtm` directly, which keeps spikes on a trial-start-relative clock.

ii.
```python
mask = (u['trial'] == tr)
if np.any(mask):
    counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
```

iii. The justification in the notes conflicts with the code: Step 6 says "goCue-aligned spike binning", but no subtraction by `session['goCue']` is implemented.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural data are binned at 20 ms, with no later rebinning step.

ii.
```python
def build_neural_trials(session, units, tmin=-2.4, tmax=2.0, dt=0.02, smooth_sigma=2.0):
    edges = np.arange(tmin, tmax + dt, dt)
```

```python
'time_bin_size': 20.0,
'off_start': -2.4,
'off_end': 2.0,
```

iii. The notes and README both describe 20 ms bins and the `[-2.4, 2.0]` window as the chosen representation.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw measured variable. The input is synthesized from the bin centers returned by `build_neural_trials`.

ii.
```python
def build_inputs(centers, ntrials):
    arr = centers.astype(np.float32)[None, :]
    return [arr.copy() for _ in range(ntrials)]
```

iii. The notes and README both describe the decoder input simply as "time from go cue", consistent with defining it from the chosen time grid rather than from a separate raw stream.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes the input as the centers of the neural bin edges and repeats that same 1D time vector for every trial in the session.

ii.
```python
edges = np.arange(tmin, tmax + dt, dt)
centers = edges[:-1] + dt / 2
...
def build_inputs(centers, ntrials):
    arr = centers.astype(np.float32)[None, :]
    return [arr.copy() for _ in range(ntrials)]
```

iii. The agent's chosen neural window and bin size drive this computation, so the input inherits the same 20 ms `[-2.4, 2.0]` grid described in the notes.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same bin-center grid returned by `build_neural_trials`, so it is aligned to the neural bins by construction.

ii.
```python
neural_trials, centers = build_neural_trials(sess, units)
inputs = build_inputs(centers, sess['ntrials'])
```

iii. The agent consistently uses one shared grid for neural arrays and for the time input, even though that grid does not match the human reference grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The final script derives lick direction only from the per-trial boolean arrays `L` and `R`.

ii.
```python
lick = np.full(ntr, 2, dtype=np.int64)
lick[session['L']] = 0
lick[session['R']] = 1
```

iii. The notes mention `obj.bp.R`, `obj.bp.L`, and "or lick event side" as candidate raw variables, and the final code chooses the simplest direct relabeling of `L` and `R`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code initializes all trials to class `2` (`none`), then sets left trials to `0` and right trials to `1` based only on `L` and `R`.

ii.
```python
lick = np.full(ntr, 2, dtype=np.int64)
lick[session['L']] = 0
lick[session['R']] = 1
```

iii. The trajectory and notes show that the agent understood left/right trial-type fields existed; the final implementation uses them directly rather than inferring actual lick direction from hit/miss.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `autowater`.

ii.
```python
'autowater': np.array(bp['autowater'][()]).squeeze().astype(bool),
...
context = np.where(session['autowater'], 0, 1).astype(np.int64)  # WC, DR
```

iii. The notes explicitly say that `autowater` maps WC versus DR, matching the agent's final implementation.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It is a direct binary remapping: `autowater=True` becomes `0` (`WC`), otherwise `1` (`DR`).

ii.
```python
context = np.where(session['autowater'], 0, 1).astype(np.int64)  # WC, DR
```

iii. The notes say the code comments confirmed the polarity `autowater = WC`, so the final transform is just a relabeling.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial flags `miss`, `hit`, and `no`.

ii.
```python
'hit': np.array(bp['hit'][()]).squeeze().astype(bool),
'miss': np.array(bp['miss'][()]).squeeze().astype(bool),
'no': np.array(bp['no'][()]).squeeze().astype(bool),
```

```python
outcome = np.full(ntr, 2, dtype=np.int64)
outcome[session['miss']] = 0
outcome[session['hit']] = 1
outcome[session['no']] = 2
```

iii. The notes say the source data contain `hit`, `miss`, and `no`, and the agent decided to keep ignore trials as a third decoder class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Trials are mapped to `incorrect=0`, `correct=1`, and `ignore=2` via the `miss`, `hit`, and `no` flags respectively.

ii.
```python
outcome = np.full(ntr, 2, dtype=np.int64)
outcome[session['miss']] = 0
outcome[session['hit']] = 1
outcome[session['no']] = 2
```

iii. This mapping is explicitly documented in the notes and matches the output class ordering declared near the top of the script.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The final code derives tongue velocity from HDF5 `obj.traj` entries: `featNames`, `ts`, and `frameTimes`. It looks for tongue-related feature names `tongue`, `left_tongue`, and `right_tongue` in the first trajectory view it reads.

ii.
```python
def extract_h5_traj_view(h, traj_group, trial_idx):
    feat_cell = h[traj_group['featNames'][trial_idx, 0]]
    feat_names = [decode_h5_char(h, r) for r in feat_cell[()].ravel()]
    ts = np.array(h[traj_group['ts'][trial_idx, 0]][()])
    frame_times = np.array(h[traj_group['frameTimes'][trial_idx, 0]][()]).squeeze().astype(float)
    return feat_names, ts, frame_times
```

```python
spd0, valid0 = speed_from_ts(ts0, feat0, ['tongue','left_tongue','right_tongue'])
```

iii. The notes say the decoded trajectory feature names included tongue features, and Step 5 planned to build tongue velocity from trajectory data. The final implementation does that, but only for HDF5 sessions and without using the reference's two-camera scheme.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For the selected tongue feature, the code computes frame-to-frame Euclidean displacement in pixels, marks frames as valid if likelihood is at least `0.5`, resamples both speed and validity to the neural bin count by interpolation over normalized trace length, pools valid resampled values across the session, and splits them at the session median.

ii.
```python
def speed_from_ts(ts, feat_names, feature_candidates, like_thresh=0.5):
    ...
    dx = np.diff(xy[0], prepend=np.nan)
    dy = np.diff(xy[1], prepend=np.nan)
    spd = np.sqrt(dx**2 + dy**2)
    spd[~valid] = np.nan
    return spd, valid
```

```python
rr = resample_trace_to_bins(np.nan_to_num(spd0, nan=np.nanmedian(spd0[np.isfinite(spd0)]) if np.isfinite(spd0).any() else 0.0), ntime)
vv = resample_trace_to_bins(valid0.astype(float), ntime)
...
med = np.nanmedian(np.asarray(tongue_vals, dtype=float))
...
tongue_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
```

iii. The notes justify session-median discretization and mention tongue visibility issues, but they do not justify the final choices of a `0.5` likelihood threshold, no smoothing, no time-derivative by real frame times, and interpolation by trace index.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Bins with interpolated validity below `0.5` stay at class `2` (`not visible`). Valid bins are thresholded at the session median into `0` (<50th percentile) or `1` (>=50th percentile).

ii.
```python
tongue_out = np.full((ntr, ntime), 2, dtype=np.int64)
...
med = np.nanmedian(np.asarray(tongue_vals, dtype=float))
...
tongue_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
```

iii. The discretization rule matches the task prompt and the agent's notes, which repeatedly refer to a per-session median split with a separate not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is not aligned by go-cue time or by actual frame timestamps. Instead, the framewise speed trace is stretched or compressed to `ntime` bins with `resample_trace_to_bins`, so alignment is by relative index only.

ii.
```python
def resample_trace_to_bins(trace, ntime):
    ...
    x_old = np.linspace(0.0, 1.0, trace.size)
    x_new = np.linspace(0.0, 1.0, ntime)
    return np.interp(x_new, x_old, trace).astype(np.float32)
```

```python
feat0, ts0, ft0 = extract_h5_traj_view(h, view0, tr)
...
rr = resample_trace_to_bins(..., ntime)
```

iii. The code reads `frame_times` into `ft0` but never uses them. This shows the intended raw timing information was discarded during the final implementation.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` using `featNames`, `ts`, and `frameTimes`, but only from the second trajectory view the code opens. It accepts either `top_paw` or `bottom_paw`.

ii.
```python
feat1, ts1, ft1 = extract_h5_traj_view(h, view1, tr)
spd1, valid1 = speed_from_ts(ts1, feat1, ['top_paw','bottom_paw'])
```

iii. The notes say the raw trajectory data contain multiple tracked features and that paw velocity would be built from those trajectories. The final implementation chooses whichever of `top_paw` or `bottom_paw` it finds in that view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The processing mirrors the tongue pipeline used here: frame-to-frame Euclidean displacement, likelihood threshold `0.5`, interpolation of speed and validity to the neural bin count, and a per-session median split over valid resampled bins.

ii.
```python
spd1, valid1 = speed_from_ts(ts1, feat1, ['top_paw','bottom_paw'])
...
rr = resample_trace_to_bins(np.nan_to_num(spd1, nan=np.nanmedian(spd1[np.isfinite(spd1)]) if np.isfinite(spd1).any() else 0.0), ntime)
vv = resample_trace_to_bins(valid1.astype(float), ntime)
...
med = np.nanmedian(np.asarray(paw_vals, dtype=float))
...
paw_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
```

iii. The notes justify the median split and visibility class, but not the final interpolation-based alignment or the use of either paw landmark interchangeably.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The script uses class `2` for bins not marked valid after interpolating the validity mask, and uses the session median of valid paw speeds to assign classes `0` and `1`.

ii.
```python
paw_out = np.full((ntr, ntime), 2, dtype=np.int64)
...
med = np.nanmedian(np.asarray(paw_vals, dtype=float))
...
paw_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
```

iii. This directly follows the prompt's requested median split plus a separate not-visible class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Like tongue velocity, paw velocity is aligned only by interpolating the trace to the neural bin count. `frameTimes` are read but not used.

ii.
```python
feat1, ts1, ft1 = extract_h5_traj_view(h, view1, tr)
...
rr = resample_trace_to_bins(..., ntime)
vv = resample_trace_to_bins(valid1.astype(float), ntime)
```

iii. The final code drops actual frame timing, so the alignment decision is index-based rather than event-based.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived primarily from the sidecar `motionEnergy_<session>.mat` file loaded by `load_motion_energy_sidecar`. If that fails but `me_obj` exists in the session object, the code falls back to `obj.me`.

ii.
```python
def load_motion_energy_sidecar(session_name):
    f = BASE / f'motionEnergy_{session_name}.mat'
    ...
    me = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)['me']
```

```python
me_data, me_thresh = load_motion_energy_sidecar(session['session_name'])
...
elif session.get('me_obj', None) is not None:
    vals = np.asarray(session['me_obj']).squeeze()
```

iii. The notes explicitly say the agent found sidecar motion-energy files and planned to use them, with `obj.me` as a fallback when available.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. If per-trial traces are available, each trace is interpolated to the neural bin count, all values are pooled session-wide, and the session median is used as the threshold. If only a 1D `obj.me` vector exists, it is median-split per trial and then repeated across time bins.

ii.
```python
for trc in trial_traces:
    rr = resample_trace_to_bins(trc, ntime)
    ...
    session_vals.extend(rr.tolist())
...
med = np.nanmedian(np.asarray(session_vals, dtype=float))
...
motion[i] = (rr >= med).astype(np.int64)
```

```python
vals = np.asarray(session['me_obj']).squeeze()
...
cls = (vals >= med).astype(np.int64)
motion[:] = cls[:, None]
```

iii. The notes justify a conservative fallback strategy for irregular motion-energy formats, but the final code again uses interpolation by trace index rather than the paper's frame-time alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. When a motion-energy trace is present, bins are split at the session median into `0` and `1`. When no sidecar file is available and no usable `obj.me` exists, the default value `2` (`no_video`) is left in place for all bins.

ii.
```python
motion = np.full((ntr, ntime), 2, dtype=np.int64)
...
med = np.nanmedian(np.asarray(session_vals, dtype=float))
...
motion[i] = (rr >= med).astype(np.int64)
```

iii. This matches the prompt's requested categorical encoding and the agent's notes about using a no-video class when video-derived data are unavailable.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It is aligned by interpolating each trace to `ntime`; actual frame timestamps are not used.

ii.
```python
rr = resample_trace_to_bins(trc, ntime)
```

iii. The final script never combines motion energy with `frameTimes` or with any video offset estimate, so its alignment is only by normalized sample index.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly handles missing or malformed data by swallowing exceptions and falling back to defaults. Missing video-derived outputs remain class `2`; invalid speed samples are filled with the within-trace median (or `0`) before resampling, while validity is separately interpolated and thresholded. Motion-energy parsing failures silently leave the default `2` array in place.

ii.
```python
try:
    ...
except Exception:
    pass
```

```python
rr = resample_trace_to_bins(np.nan_to_num(spd0, nan=np.nanmedian(spd0[np.isfinite(spd0)]) if np.isfinite(spd0).any() else 0.0), ntime)
vv = resample_trace_to_bins(valid0.astype(float), ntime)
```

iii. `CONVERSION_NOTES.md` says the agent wanted conservative missing-data handling and separate not-visible / no-video classes. The final implementation achieves that mostly by default-filled arrays and silent fallbacks rather than by explicit auditing.

## 11-a. What are the most time-consuming steps of the code?

i. The main bottlenecks are the nested neural loops over every trial and every retained unit, and the per-trial video extraction/resampling loop that reopens each HDF5 session and decodes feature names repeatedly.

ii.
```python
for tr in range(1, session['ntrials'] + 1):
    ...
    for i, u in enumerate(units):
        ...
```

```python
with h5py.File(BASE / f"data_structure_{session['session_name']}.mat", 'r') as h:
    ...
    for tr in range(ntr):
        ...
```

iii. The notes themselves mention code-speed issues and that the script now uses "basic output construction"; they do not claim any strong optimization beyond making the conversion run.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-unit histogramming in `build_neural_trials` could be vectorized by aggregating spikes across trials per unit, and the per-trial output assembly with `np.vstack` could be preallocated once. The repeated resampling of each video trace and validity mask is also loop-heavy.

ii.
```python
for tr in range(1, session['ntrials'] + 1):
    ...
    for i, u in enumerate(units):
        ...
```

```python
outputs = []
for tr in range(ntr):
    out = np.vstack([
        np.full(ntime, lick[tr], dtype=np.int64),
        ...
    ])
```

iii. The notes contain a placeholder section for code inefficiencies, but the final implementation leaves these loops in straightforward scalar form.

## 11-c. What processing does the code repeat multiple times?

i. `select_units(sess)` is run once during session filtering and again during session conversion. The script also repeatedly rebuilds Gaussian kernels inside `gaussian_smooth`, reopens each session HDF5 file for video extraction after already loading the session, and repeatedly interpolates traces and validity masks one trial at a time.

ii.
```python
sessions = [s for s in sessions if len(select_units(s)) >= 10]
...
units = select_units(sess)
```

```python
with h5py.File(BASE / f"data_structure_{session['session_name']}.mat", 'r') as h:
```

iii. `CONVERSION_NOTES.md` does not document these repetitions as intentional reuse; they are side effects of the script's simple control flow.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several fields are loaded but not used in the final outputs: `sample`, `reward`, `site`, `me_thresh`, and the trajectory `frameTimes` variables `ft0`/`ft1`. `tm` is used only for unit filtering and then discarded. The script also imports modules such as `os`, `re`, and `Counter` without using them.

ii.
```python
'sample': np.array(bp['ev']['sample'][()]).squeeze().astype(float) if 'sample' in bp['ev'] else None,
'reward': np.array(bp['ev']['reward'][()]).squeeze().astype(float) if 'reward' in bp['ev'] else None,
...
site = int(np.array(h[clu['site'][i, 0]][()]).squeeze()) if 'site' in clu else -1
```

```python
feat0, ts0, ft0 = extract_h5_traj_view(h, view0, tr)
...
me_data, me_thresh = load_motion_energy_sidecar(session['session_name'])
```

iii. The notes emphasize broad exploration of available fields, and the final code still carries some of that exploratory loading even though those values never affect the saved dataset.
