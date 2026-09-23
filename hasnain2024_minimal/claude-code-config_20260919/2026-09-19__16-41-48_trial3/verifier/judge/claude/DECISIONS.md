# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat` in one of two folders (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). Motion energy is in a separate file `motionEnergy_<anm>_<date>.mat`. The 44 sessions are hard-coded in two lists (`FIXED_DELAY_SESSIONS` and `RANDOM_DELAY_SESSIONS`), transcribed from the authors' loading scripts. Files are read via a custom `matio.py` module that handles both MATLAB v7 and v7.3 (HDF5) formats. The AI also created a `matio.py` helper that skips large unused fields (`spkWavs`, `sglxfns`, etc.) during loading.

ii.
```python
FIXED_DELAY_SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ...
]
RANDOM_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1]),
    ...
]

# Loading:
def load_obj(path):
    return loadmat_var(path, 'obj')
```

iii. The agent examined the authors' `load<ANM>_ALMVideo.m` files to identify which sessions to include and which probes to use, and noted that commented-out entries should be excluded.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of the session tuple (e.g., `'JEB6'`). Subjects are accumulated in insertion order as sessions are processed. `subject_idx` maps each session to an index into the `subjects` list.

ii.
```python
if anm not in subjects:
    subjects.append(anm)
...
subject_idx.append(subjects.index(anm))
```

iii. The animal ID is directly available from the session definitions.

## 1-c. How are the data split into sessions?

i. One session is one entry in the session lists, identified by `(anm, date, probes)`. The code iterates over the combined list of 25 fixed-delay and 19 randomized-delay sessions, processing each one independently. Sessions with fewer than `MIN_UNITS=10` units or fewer than `MIN_TRIALS=2` trials are dropped.

ii.
```python
sessions = ([(a, d, p, FIXED_DELAY_DIR, 'fixed delay (DR/WC two-context)')
             for a, d, p in FIXED_DELAY_SESSIONS] +
            [(a, d, p, RANDOM_DELAY_DIR, 'randomized delay (DR only)')
             for a, d, p in RANDOM_DELAY_SESSIONS])

for anm, date, probes, folder, task in sessions:
    rez = process_session(anm, date, probes, folder, task)
    if rez is None:
        print('  skipped (too few units or trials)')
        continue
```

iii. Sessions were identified from the authors' loading scripts. The MIN_UNITS=10 filter comes from the paper's Methods ("Recording sessions were included for analysis only if they had at least 10 units").

## 1-d. How are the data split into trials?

i. Trials are identified by index (0-based) within each session. The total number of trials is read from `bp['Ntrials']`. Per-trial fields such as `hit`, `miss`, `R`, `early`, etc. are read as arrays of length `ntrials_all`. Trial indices that pass filtering are stored in `trials = np.flatnonzero(keep)`.

ii.
```python
ntrials_all = int(np.asarray(bp['Ntrials']).reshape(-1)[0])
hit = as_bool(bp['hit'], ntrials_all)
miss = as_bool(bp['miss'], ntrials_all)
...
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
trials = np.flatnonzero(keep)
```

iii. The Bpod table defines trials directly, with one entry per trial.

## 1-e. How are trials filtered based on quality controls?

i. Four filters: (1) early-lick trials are excluded (`~early`), (2) photostimulation trials are excluded (`~stim`), (3) trials without a valid (finite) go cue are excluded, and (4) only trials that are hit, miss, or ignore (`no`) are kept. Additionally, after neural processing, trials with zero total spikes across all units are dropped (handling sessions where recording stopped before behavior).

ii.
```python
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
trials = np.flatnonzero(keep)
...
# Late trials with no spikes:
has_spikes = rates.sum(axis=(0, 1)) > 0
if n_no_spikes:
    rates = rates[:, :, has_spikes]
    trials = trials[has_spikes]
```

iii. Early-lick and photostim exclusion follows the paper. The go cue validity check and the hit|miss|no requirement are additional safeguards. The zero-spike filter handles sessions where recording stops mid-behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu` (the spike-sorted clusters), specifically each cluster's `trial` (which trial each spike belongs to, 1-based) and `trialtm` (spike time relative to trial start). The go cue times from `bp.ev.goCue` provide the alignment event.

ii.
```python
utrial = np.asarray(unit['trial'], dtype=float).reshape(-1)
utm = np.asarray(unit['trialtm'], dtype=float).reshape(-1)
...
aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
dat[:, k] = bin_spikes(aligned, edges)
```

iii. The agent read the data structure and identified `clu.trial` and `clu.trialtm` as the spike data fields, consistent with the reference code.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into **10 ms** bins (DT=0.01, `params.dt = 1/100`), converted to spikes/s by dividing by DT, then smoothed with a **causal Gaussian kernel** (the authors' `mySmooth` function): a `gausswin(15)` with the first half zeroed, applied with 'reflect' boundary conditions. Units from multiple probes are concatenated.

ii.
```python
DT = 0.01            # s (params.dt = 1/100)
SMOOTH_N = 15        # bins (params.smooth), causal Gaussian

def my_smooth(x, N=SMOOTH_N):
    kern = gausswin(N)
    kern[:N // 2] = 0.0          # causal half
    kern = kern / kern.sum()
    xp = np.concatenate([x[:N], x], axis=0)
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xp)
    return out[N:]
...
dat = my_smooth(dat / DT)
```

iii. The agent explicitly chose 10 ms bins based on `params.dt = 1/100` from the reference code. The causal Gaussian kernel is a port of `utils/mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters whose quality label (lowered) is in `BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}` are excluded. Then units whose mean firing rate is ≤ 1 Hz are dropped (`LOW_FR = 1.0`). Additionally, sessions with fewer than 10 remaining units (`MIN_UNITS = 10`) are dropped entirely.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
...
quality = str(unit.get('quality', '')).replace('\x00', '').strip()
if quality.lower() in BAD_QUALITY:
    continue
...
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
if use.sum() < MIN_UNITS:
    return None
```

iii. The agent examined the `findClusters` function and enumerated all quality labels in the dataset (step 120). It identified `garbage`, `gabrga`, `noisy`, and `real?` as rejected labels. The 1 Hz firing rate cutoff and 10-unit minimum come from the paper's methods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting `goCue[trial]` from each spike's `trialtm`. The aligned spike times are then binned relative to the go cue.

ii.
```python
aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
dat[:, k] = bin_spikes(aligned, edges)
```

iii. This follows the reference's `alignSpikes.m` which subtracts the alignment event from `trialtm`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is **10 ms** (DT = 0.01 s), producing 500 bins spanning -2.5 to +2.5 s from the go cue. No rebinning is applied; spikes are counted directly into the 10 ms bins.

ii.
```python
TMIN = -2.5          # s, relative to the go cue
TMAX = 2.5           # s
DT = 0.01            # s (params.dt = 1/100)
...
edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
time = edges[:-1] + DT / 2
```

iii. The agent chose 10 ms based on `params.dt = 1/100` from the reference code's `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is a synthetic variable defined by the time bin centers of the analysis window, not derived from any raw data variable. It is the center of each bin in the [-2.5, 2.5] s window.

ii.
```python
time = edges[:-1] + DT / 2
...
time_row = time.astype(np.float32)[None, :]
```

iii. The time axis is defined by the binning grid and represents seconds from the go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing; the bin centers are computed directly from the time window and bin size parameters.

ii.
```python
edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
time = edges[:-1] + DT / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same binning grid used for the neural data. Each bin center corresponds to the same time interval as the corresponding neural bin.

ii.
```python
time_row = time.astype(np.float32)[None, :]
...
inputs.append(time_row.copy())
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields: `bp.hit`, `bp.miss`, `bp.R`, and `bp.L`. Hit/miss indicates whether the animal licked correctly/incorrectly, and R/L indicates the instructed side.

ii.
```python
hit = as_bool(bp['hit'], ntrials_all)
miss = as_bool(bp['miss'], ntrials_all)
R = as_bool(bp['R'], ntrials_all)
L = as_bool(bp['L'], ntrials_all)
```

iii. Lick direction is not recorded directly and must be inferred from the combination of instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A right lick is `(R & hit) | (L & miss)` — either a correct rightward lick or an incorrect lick when left was instructed. A left lick is the complement. Ignore trials (`no`) get class 2 ("none"). Codes: left=0, right=1, none=2.

ii.
```python
right = (R & hit) | (L & miss)
left = (L & hit) | (R & miss)
lick_dir = np.full(ntrials_all, 2, dtype=np.int8)   # 'none'
lick_dir[left] = 0
lick_dir[right] = 1
lick_dir[no] = 2
```

iii. This follows the paper's `getPrevChoice.m` definition.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `bp.autowater`. The AI also handles a coding difference: older sessions use 1=off/2=on while newer ones use 0/1.

ii.
```python
aw = np.asarray(bp['autowater'], dtype=float).reshape(-1)
autowater = (aw == 2) if np.nanmax(aw) > 1 else (np.nan_to_num(aw) > 0.5)
```

iii. Autowater directly indicates the WC context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater trials are coded as WC (0), all others as DR (1).

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)  # 0 = WC, 1 = DR
```

iii. Matches the prompt's WC/DR coding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = as_bool(bp['hit'], ntrials_all)
miss = as_bool(bp['miss'], ntrials_all)
no = as_bool(bp['no'], ntrials_all)
```

iii. The three outcome flags are mutually exclusive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss → incorrect (0), hit → correct (1), no → ignore (2).

ii.
```python
outcome = np.full(ntrials_all, 2, dtype=np.int8)     # 'ignore'
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
```

iii. Matches the prompt's specification of incorrect/correct/ignore.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking data in `obj.traj`, specifically the `tongue` feature from the **side camera only** (view index 0). The x, y coordinates and frame times are used. The video offset fields (`sglx.bitcode.bitstart`, `sglx.fs`, `bp.ev.bitStart`) are used for clock alignment.

ii.
```python
TONGUE_FEATURE = ('tongue', 0)          # (feature name, view index)
...
if tongue_ix is not None and trix < len(views[TONGUE_FEATURE[1]]):
    tt, ts = trial_video_times(views[TONGUE_FEATURE[1]][trix], vidshift, gocue[trix])
    if tt is not None:
        xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
        tongue_speed[:, k] = speed(xy[:, 0], xy[:, 1])
```

iii. The AI chose to use only the side camera view of the tongue, not both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Three steps: (1) Frame times are corrected using the video offset and aligned to the go cue. (2) The x, y positions are **linearly interpolated** (`interp_trace`) from frame times to the bin centers. NaN is used outside the source range. (3) Speed is computed as the magnitude of central/one-sided finite differences of x and y. No Gaussian smoothing is applied to the position before differentiation.

ii.
```python
def interp_trace(t_src, y_src, t_dst):
    """Linear interpolation with NaN outside the source range (MATLAB interp1)."""
    ...
    out[inside] = np.interp(t_dst[inside], t_src, y_src)
    return out

def speed(x, y):
    """Speed from a 2-D position trace, NaN-aware (central / one-sided differences)."""
    def deriv(v):
        d = np.full(v.shape, np.nan)
        fwd[:-1] = v[1:] - v[:-1]
        bwd[1:] = v[1:] - v[:-1]
        both = np.isfinite(fwd) & np.isfinite(bwd)
        d[both] = (fwd[both] + bwd[both]) / 2.0
        ...
    return np.sqrt(deriv(x) ** 2 + deriv(y) ** 2)
```

iii. The AI used interpolation to resample to bin centers rather than binning the frame-rate velocity.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session-wide median (50th percentile) of all finite tongue speed values is computed. Values below the median are class 0 ("below_median"), at or above are class 1 ("above_median"), and NaN values (tongue not visible) are class 2 ("not_visible").

ii.
```python
def discretize(x):
    out = np.full(x.shape, 2, dtype=np.int8)
    ok = np.isfinite(x)
    if ok.any():
        thresh = np.percentile(x[ok], 50)
        out[ok] = (x[ok] >= thresh).astype(np.int8)
    return out, thresh
```

iii. Follows the prompt's specification of 50th percentile threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (computed via `findVideoOffset` logic) and aligned to the go cue. The tracking data is then **interpolated** to the same bin centers as the neural data, so both share the same time axis.

ii.
```python
def video_shift(obj):
    return mode_(obj['sglx']['bitcode']['bitstart']) / float(obj['sglx']['fs']) \
        - mode_(obj['bp']['ev']['bitStart'])

def trial_video_times(trial, vidshift, align_time):
    ...
    return ft - vidshift - align_time, ts

# Interpolated to bin centers:
xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
```

iii. The video offset follows `findVideoOffset.m`. The mode calculation uses `np.unique` with `argmax(counts)` rather than `pd.Series.mode()`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking data in `obj.traj`, specifically **both** `top_paw` and `bottom_paw` features from the **bottom camera** (view index 1).

ii.
```python
PAW_FEATURES = [('top_paw', 1), ('bottom_paw', 1)]
...
for ix, _ in paw_ix:
    if ix is None:
        continue
    xy = interp_trace(tt, ts[:, 0:2, ix], time)
    xy = np.stack([fill_interior_nans(xy[:, 0]),
                   fill_interior_nans(xy[:, 1])], axis=1)
    sp.append(speed(xy[:, 0], xy[:, 1]))
if sp:
    paw_speed[:, k] = np.nanmean(np.stack(sp, axis=1), axis=1)
```

iii. The AI used both paws and averaged their speeds.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw: (1) x, y positions are interpolated to bin centers. (2) Interior NaNs (between the first and last valid sample) are filled with nearest-neighbor interpolation (`fill_interior_nans`). (3) Speed is computed from finite differences. (4) The speeds of both paws are averaged with `nanmean`.

ii.
```python
xy = interp_trace(tt, ts[:, 0:2, ix], time)
xy = np.stack([fill_interior_nans(xy[:, 0]),
               fill_interior_nans(xy[:, 1])], axis=1)
sp.append(speed(xy[:, 0], xy[:, 1]))
...
paw_speed[:, k] = np.nanmean(np.stack(sp, axis=1), axis=1)
```

iii. Interior NaN filling follows the authors' `fillmissing(...,'nearest')` approach.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: session median threshold, with class 0 below, class 1 at/above, class 2 for NaN (not visible).

ii.
```python
paw_cls, paw_thresh = discretize(paw_speed)
```

iii. Follows the prompt's 50th percentile specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue velocity: video offset correction, go cue alignment, then interpolation to bin centers.

ii.
```python
tt, ts = trial_video_times(views[1][trix], vidshift, gocue[trix])
xy = interp_trace(tt, ts[:, 0:2, ix], time)
```

iii. Uses the bottom camera's frame times.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The `motionEnergy_<anm>_<date>.mat` file, which contains one trace per trial with one value per camera frame. The file is loaded via `loadmat_var` and the nested `data` field is unwrapped.

ii.
```python
me_path = os.path.join(folder, 'motionEnergy_%s_%s.mat' % (anm, date))
me = loadmat_var(me_path, 'me')
me_data = me.get('data') if isinstance(me, dict) else me
if isinstance(me_data, dict):
    me_data = me_data.get('data')
```

iii. The motion energy files have varying nesting structure, handled by unwrapping.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-frame motion energy values are **interpolated** to bin centers using `interp_trace`, then interior NaNs are filled with nearest-neighbor (`fill_interior_nans`).

ii.
```python
y = np.asarray(me_data[trix], dtype=float).reshape(-1)
tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
n = min(tt.size, y.size)
me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
```

iii. Interpolation resamples to the neural time grid; interior NaN filling follows the authors' approach.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw: session median threshold, class 0 below, class 1 at/above, class 2 for NaN (no video).

ii.
```python
me_cls, me_thresh = discretize(me_trace)
```

iii. Follows the prompt's 50th percentile specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times (view 0), corrected by the video offset and aligned to the go cue, then interpolated to the same bin centers as neural data.

ii.
```python
tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
```

iii. Same clock correction and alignment as other video streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials with NaN go cue times are excluded via the `np.isfinite(gocue)` filter. (2) Trials with NaN `NdroppedFrames` are skipped for video (following `findPosition.m`). (3) Missing frame times trigger a fallback to a nominal 400 Hz frame clock. (4) Interior NaN values in paw and motion energy traces are filled with nearest-neighbor interpolation (`fill_interior_nans`). (5) Tongue speed NaNs (tongue not visible) are preserved and assigned class 2. (6) The `as_bool` helper handles NaN values and size mismatches in boolean trial fields.

ii.
```python
# Missing frame times fallback:
if ft is None or np.size(ft) == 0 or not np.any(np.isfinite(np.asarray(ft, float))):
    ft = (np.arange(ts.shape[0]) + 1) / 400.0
    vidshift = 0.5

# Interior NaN filling:
def fill_interior_nans(y):
    ...
    seg[bad] = seg[nearest]
    ...

# NaN go cue exclusion:
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
```

iii. The 400 Hz fallback comes from `findPosition.m`. Interior NaN filling follows the authors' `fillmissing` approach.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files dominates runtime. The `matio.py` reader must parse HDF5 or v7 structures for each session. The agent notes skipping large unused fields (`spkWavs`, `sglxfns`, etc.) to reduce loading time.

ii.
```python
SKIP_FIELDS = ('spkWavs', 'sglxfns', 'fnEpochs', 'svpth', 'sglxpth')
...
obj = load_obj(path)
```

iii. File I/O is the bottleneck; computation is relatively cheap.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for spike binning iterates over all kept trials per unit, calling `bin_spikes` for each trial individually. The per-trial video processing loops (tongue, paw, motion energy) each iterate over trials. The smoothing uses `np.apply_along_axis` with `np.convolve`, which loops internally.

ii.
```python
for k in range(trials.size):
    if stops[k] > starts[k]:
        aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
        dat[:, k] = bin_spikes(aligned, edges)
```

iii. The spike binning could potentially be vectorized with `np.histogram2d` as in the reference solution. The video loops are harder to vectorize due to variable frame counts.

## 11-c. What processing does the code repeat multiple times?

i. The `trial_video_times` function is called multiple times per trial (once for tongue, once for paw, once for motion energy), each time recomputing the frame-time correction. The `as_bool` function is called for each boolean trial field, each time doing NaN handling and size checks.

ii.
```python
# Called for tongue:
tt, ts = trial_video_times(views[TONGUE_FEATURE[1]][trix], vidshift, gocue[trix])
# Called again for paw:
tt, ts = trial_video_times(views[1][trix], vidshift, gocue[trix])
# Called again for motion energy:
tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
```

iii. The video offset itself is computed once per session, but frame time corrections are repeated.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and processes anatomical probe locations (`probe_locations`, `region_name`) even though nearly all sessions are ALM. It computes detailed session info statistics (number of single units, WC trial counts, excluded trial counts, velocity thresholds) that are stored in metadata but not used by the decoder. The `fill_interior_nans` function is applied to paw and motion energy but not to tongue, adding processing that may not improve decoder performance.

ii.
```python
# Probe location parsing:
locs = probe_locations(obj, len(clu))
region = region_name(locs[prb - 1])

# Detailed session statistics:
'n_single_units': int(sum(q.lower() in ('fair', 'good', 'great', 'excellent')
                          for q in qualities)),
'n_trials_excluded_early': int((early & ~stim).sum()),
```

iii. These provide useful metadata but are not needed for the decoder itself.
