# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes two session lists, `FIXED_DELAY_SESSIONS` and `RANDOM_DELAY_SESSIONS`, then loops over them in `main()`. For each session it loads `data_structure_<anm>_<date>.mat` with a custom MATLAB reader from `matio.py`, and separately loads `motionEnergy_<anm>_<date>.mat` with `loadmat_var` if the file exists. The trajectory says this was intended to mirror the authors' `load<ANM>_ALMVideo.m` scripts rather than globbing the folders.

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
...
for anm, date, probes, folder, task in sessions:
    rez = process_session(anm, date, probes, folder, task)
```

```python
path = os.path.join(folder, 'data_structure_%s_%s.mat' % (anm, date))
obj = load_obj(path)
...
me = loadmat_var(me_path, 'me')
```

iii. In the trajectory final summary, the AI says it used "exactly those enumerated in `DataLoadingScripts/Recording and video/load*_ALMVideo.m`" and excluded commented-out or unsupported entries. Its initial write also added `matio.py` specifically because the dataset mixes MATLAB v7 and v7.3 files.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats the `anm` field from each hard-coded session tuple as the subject id. During assembly it preserves subjects in first-seen order and builds `subject_idx` by indexing into that order.

ii.
```python
for anm, date, probes, folder, task in sessions:
    ...
    if anm not in subjects:
        subjects.append(anm)
    ...
    subject_idx.append(subjects.index(anm))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
```

iii. The trajectory does not contain a separate justification beyond the session-list design. The code structure shows that subject identity was taken directly from the hard-coded session metadata rather than from fields inside each `.mat` file.

## 1-c. How are the data split into sessions?

i. Each tuple `(animal, date, probes, folder, task)` is treated as one session. Fixed-delay and randomized-delay sessions are kept in separate lists but concatenated into one `sessions` iterable in `main()`. Each retained session contributes one element to `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
sessions = ([(a, d, p, FIXED_DELAY_DIR, 'fixed delay (DR/WC two-context)')
             for a, d, p in FIXED_DELAY_SESSIONS] +
            [(a, d, p, RANDOM_DELAY_DIR, 'randomized delay (DR only)')
             for a, d, p in RANDOM_DELAY_SESSIONS])
...
data['neural'].append(rez['neural'])
data['input'].append(rez['input'])
data['output'].append(rez['output'])
```

iii. The trajectory final summary explicitly says the AI kept the 25 fixed-delay plus 19 randomized-delay ephys sessions listed in the paper's loading scripts, because it believed those were the ALM recordings analyzed with the same go-cue-aligned pipeline.

## 1-d. How are the data split into trials?

i. The AI uses `bp['Ntrials']` as the master trial count, converts per-trial behavioral fields to length `ntrials_all`, and defines the kept trial indices with a Boolean mask. Neural spikes remain associated with trials through `unit['trial']`, and video data remain per-trial through `traj[view][trial]`.

ii.
```python
ntrials_all = int(np.asarray(bp['Ntrials']).reshape(-1)[0])
...
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
trials = np.flatnonzero(keep)
```

```python
utrial = np.asarray(unit['trial'], dtype=float).reshape(-1)
...
starts = np.searchsorted(utrial, trials + 1, side='left')
stops = np.searchsorted(utrial, trials + 1, side='right')
for k in range(trials.size):
    if stops[k] > starts[k]:
        aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
```

iii. The trajectory does not give a separate prose justification for trial splitting, but the final summary states that trials were kept after excluding early-lick and photostimulation trials, with hit, miss, and ignore trials preserved.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early-lick trials, photostimulation trials, trials with non-finite go cues, and trials that are not labeled hit, miss, or no-response. After neural preprocessing it also drops kept trials that have zero spikes across all retained units and all time bins. Sessions with fewer than two remaining trials are skipped.

ii.
```python
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
trials = np.flatnonzero(keep)
if trials.size < MIN_TRIALS:
    return None
```

```python
has_spikes = rates.sum(axis=(0, 1)) > 0
n_no_spikes = int((~has_spikes).sum())
if n_no_spikes:
    rates = rates[:, :, has_spikes]
    trials = trials[has_spikes]
    if trials.size < MIN_TRIALS:
        return None
```

iii. The trajectory final summary justifies the early/stim exclusion by saying every paper condition uses `~early` and `~stim.enable`. It also explicitly says 61 late trials in two JEB24 sessions were dropped because the sorted spike data end before behavior ends, leaving those trials with zero spikes on every unit.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `obj['clu']`, specifically each unit's `trial`, `trialtm`, and `quality` fields, together with `bp['ev']['goCue']` for alignment. Probe metadata from `obj['ex']['probe']['loc']` are used only for region labels, not to construct the neural activity itself.

ii.
```python
gocue = np.asarray(bp['ev']['goCue'], dtype=float).reshape(-1)
...
for unit in units:
    quality = str(unit.get('quality', '')).replace('\x00', '').strip()
    if quality.lower() in BAD_QUALITY:
        continue
    utrial = np.asarray(unit['trial'], dtype=float).reshape(-1)
    utm = np.asarray(unit['trialtm'], dtype=float).reshape(-1)
```

iii. The trajectory final summary says the neural data come from "all sorted clusters except the qualities rejected by `findClusters`" and are aligned to `obj.bp.ev.goCue`.

## 2-b. How is the `neural` data processed?

i. For each retained unit and retained trial, the AI bins go-cue-aligned spikes into 10 ms bins from -2.5 s to +2.5 s, converts counts to spikes/s by dividing by `DT`, and smooths along time with a causal half-Gaussian kernel implemented in `my_smooth`. The saved neural arrays are transposed to `(n_neurons, n_timepoints)` per trial.

ii.
```python
DT = 0.01
SMOOTH_N = 15
...
edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
time = edges[:-1] + DT / 2
...
dat[:, k] = bin_spikes(aligned, edges)
...
dat = my_smooth(dat / DT)
rates.append(dat.astype(np.float32))
```

```python
def my_smooth(x, N=SMOOTH_N):
    kern = gausswin(N)
    kern[:N // 2] = 0.0
    kern = kern / kern.sum()
    xp = np.concatenate([x[:N], x], axis=0)
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xp)
    return out[N:]
```

iii. The trajectory final summary says the AI intentionally used `params.dt = 1/100`, `tmin/tmax = ±2.5`, and the authors' causal Gaussian `mySmooth` with `N = 15`, because it interpreted those as the relevant settings from `WorkingWithDataObjs.m`, `EDFigure2a`, and `Figure3h`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI drops clusters whose lower-cased quality label is in `{'garbage', 'gabrga', 'noisy', 'real?'}`. It then drops units with mean firing rate `<= 1 Hz` after smoothing and drops whole sessions if fewer than 10 units remain.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10
...
if quality.lower() in BAD_QUALITY:
    continue
...
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
if use.sum() < MIN_UNITS:
    return None
```

iii. The trajectory final summary says this choice was based on `findClusters({'all'})`, `removeLowFRClusters`, and the Methods sentence that sessions needed at least 10 units. Earlier in the trajectory the AI also inspected observed quality labels before removing the empty-string label from `BAD_QUALITY`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the trial's `goCue` from `trialtm`, placing spikes on a time axis centered on go-cue onset.

ii.
```python
aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
dat[:, k] = bin_spikes(aligned, edges)
```

iii. The trajectory final summary says "spikes and video are aligned to the go cue (`obj.bp.ev.goCue`)" and treats that as the common alignment event for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins spanning -2.5 s to +2.5 s, for 500 time points per trial. Spike times are rebinned into these bins from their raw timestamps.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
...
edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
time = edges[:-1] + DT / 2
T = time.size
```

iii. The trajectory final summary explicitly reports "500 bins of 10 ms per trial" and justifies this by citing `params.dt = 1/100` in the paper code it chose to follow.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The AI defines this input from the session-wide bin centers of the aligned time grid, which is itself anchored to `bp['ev']['goCue']`.

ii.
```python
gocue = np.asarray(bp['ev']['goCue'], dtype=float).reshape(-1)
...
edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
time = edges[:-1] + DT / 2
time_row = time.astype(np.float32)[None, :]
```

iii. The trajectory final summary says the single decoder input is "time from go cue in seconds."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI does not estimate this from another recorded signal. It constructs the input directly as the vector of bin centers from -2.5 s to +2.5 s in 10 ms steps, then copies that vector into every trial.

ii.
```python
time = edges[:-1] + DT / 2
...
time_row = time.astype(np.float32)[None, :]
for k, trix in enumerate(trials):
    inputs.append(time_row.copy())
```

iii. The trajectory gives no deeper justification beyond using the common go-cue-centered bin grid as the decoder input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the center times of the same bins used for neural spike counts, so the two streams are aligned bin-for-bin.

ii.
```python
edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
time = edges[:-1] + DT / 2
...
dat[:, k] = bin_spikes(aligned, edges)
...
inputs.append(time_row.copy())
```

iii. The trajectory final summary states that the decoder input is the time from go cue and that all streams are aligned to the same go-cue-centered time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the behavioral flags `R`, `L`, `hit`, `miss`, and `no`.

ii.
```python
hit = as_bool(bp['hit'], ntrials_all)
miss = as_bool(bp['miss'], ntrials_all)
no = as_bool(bp['no'], ntrials_all)
R = as_bool(bp['R'], ntrials_all)
L = as_bool(bp['L'], ntrials_all)
```

iii. The trajectory final summary says lick direction follows the paper's choice definition from `funcs/getPrevChoice.m`, with ignore trials assigned to a separate `none` class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI sets right licks as `(R & hit) | (L & miss)`, left licks as `(L & hit) | (R & miss)`, and no-response trials as class 2 (`none`). It then repeats that scalar class across all time bins of the trial.

ii.
```python
right = (R & hit) | (L & miss)
left = (L & hit) | (R & miss)
lick_dir = np.full(ntrials_all, 2, dtype=np.int8)
lick_dir[left] = 0
lick_dir[right] = 1
lick_dir[no] = 2
...
out[0, :] = lick_dir[trix]
```

iii. The trajectory final summary explicitly cites `getPrevChoice.m` and says "a right lick is `(R&hit)|(L&miss)`; ignore trials have no lick."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI derives context from `bp['autowater']`, but with an extra recoding step: if the session contains values above 1, it interprets `2` as WC/on and `1` as DR/off; otherwise it treats positive values as WC.

ii.
```python
aw = np.asarray(bp['autowater'], dtype=float).reshape(-1)
if aw.size != ntrials_all:
    aw = np.resize(aw, ntrials_all)
# older sessions code autowater as 1 = off / 2 = on, newer ones as 0/1
autowater = (aw == 2) if np.nanmax(aw) > 1 else (np.nan_to_num(aw) > 0.5)
```

iii. The trajectory justification is the inline code comment above. In the final summary the AI also says context is `obj.bp.autowater`, which it calls the authors' WC/DR proxy.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. After the recoding above, the AI maps `autowater == True` to WC (`0`) and everything else to DR (`1`). It then repeats the per-trial context label across time bins.

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)
...
out[1, :] = context[trix]
```

iii. The trajectory final summary says context is decoded as `obj.bp.autowater`, with WC/DR alternation in fixed-delay sessions and mostly DR trials in randomized-delay sessions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derives outcome from `hit`, `miss`, and `no`.

ii.
```python
hit = as_bool(bp['hit'], ntrials_all)
miss = as_bool(bp['miss'], ntrials_all)
no = as_bool(bp['no'], ntrials_all)
```

iii. The trajectory final summary says it kept hit, miss, and ignore trials because correct/incorrect/ignore is one of the decoded outputs.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps miss to `0` (`incorrect`), hit to `1` (`correct`), and no-response to `2` (`ignore`), then repeats that per-trial label across all time bins.

ii.
```python
outcome = np.full(ntrials_all, 2, dtype=np.int8)
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
...
out[2, :] = outcome[trix]
```

iii. The trajectory final summary explicitly describes the outcome mapping as "miss/hit/no."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity from the side-camera tongue tracking in `obj['traj'][0]`, using the tracked `ts` positions and `frameTimes`, plus the session video/ephys clock offset and `goCue` for alignment.

ii.
```python
TONGUE_FEATURE = ('tongue', 0)
...
tongue_ix = feature_index(views[TONGUE_FEATURE[1]], TONGUE_FEATURE[0])
...
tt, ts = trial_video_times(views[TONGUE_FEATURE[1]][trix], vidshift, gocue[trix])
xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
tongue_speed[:, k] = speed(xy[:, 0], xy[:, 1])
```

iii. The trajectory final summary says tongue speed used the side-camera `tongue` feature and was aligned with the authors' `findVideoOffset` logic.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI aligns each trial's side-camera frame times to go cue, linearly interpolates tongue x/y positions onto the neural bin centers, computes speed from the interpolated positions, leaves NaNs where the feature is not visible, and then median-splits the resulting session-wide values. It does not smooth tracked positions per contiguous run and does not combine the second tongue view.

ii.
```python
tt, ts = trial_video_times(views[TONGUE_FEATURE[1]][trix], vidshift, gocue[trix])
if tt is not None:
    xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
    tongue_speed[:, k] = speed(xy[:, 0], xy[:, 1])
```

```python
def discretize(x):
    out = np.full(x.shape, 2, dtype=np.int8)
    ok = np.isfinite(x)
    if ok.any():
        thresh = np.percentile(x[ok], 50)
        out[ok] = (x[ok] >= thresh).astype(np.int8)
```

iii. The trajectory final summary says tongue speed was "interpolated onto the bin centers" and split at the session's 50th percentile, with class 2 marking time points where DeepLabCut did not see the tongue.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI thresholds tongue speed at the session median over finite values. Values below the median are class `0`, values at or above the median are class `1`, and NaNs become class `2` (`not_visible`).

ii.
```python
def discretize(x):
    out = np.full(x.shape, 2, dtype=np.int8)
    ok = np.isfinite(x)
    if ok.any():
        thresh = np.percentile(x[ok], 50)
        out[ok] = (x[ok] >= thresh).astype(np.int8)
    else:
        thresh = np.nan
    return out, thresh
...
tongue_cls, tongue_thresh = discretize(tongue_speed)
```

iii. The trajectory final summary says the movement variables were "split at the session's 50th percentile" and that class 2 marks time points where the feature was not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-wide video shift from `sglx.bitcode.bitstart / sglx.fs - bp.ev.bitStart`, subtracts that shift and the trial's `goCue` from video frame times, and then interpolates tongue position onto the same time vector used for neural data.

ii.
```python
def video_shift(obj):
    return mode_(obj['sglx']['bitcode']['bitstart']) / float(obj['sglx']['fs']) \
        - mode_(obj['bp']['ev']['bitStart'])
...
return ft - vidshift - align_time, ts
```

```python
tt, ts = trial_video_times(views[TONGUE_FEATURE[1]][trix], vidshift, gocue[trix])
xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
```

iii. The trajectory final summary explicitly says movement variables were aligned with the authors' video clock offset (`findVideoOffset`) and then interpolated onto the neural bin centers.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI derives paw velocity from bottom-camera tracking features `top_paw` and `bottom_paw` in `obj['traj'][1]`, using frame times and tracked x/y positions.

ii.
```python
PAW_FEATURES = [('top_paw', 1), ('bottom_paw', 1)]
...
paw_ix = [(feature_index(views[v], nm), v) for nm, v in PAW_FEATURES]
...
xy = interp_trace(tt, ts[:, 0:2, ix], time)
```

iii. The trajectory final summary says paw speed used "mean speed of the two tracked paws" from the bottom view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI aligns bottom-camera frames to go cue, interpolates each paw's x/y position onto neural bin centers, fills interior NaN gaps by nearest-neighbor carry-forward/backward within the covered interval, computes speed for each paw, averages the two paw speeds where available, and then median-splits the resulting session-wide values.

ii.
```python
xy = interp_trace(tt, ts[:, 0:2, ix], time)
xy = np.stack([fill_interior_nans(xy[:, 0]),
               fill_interior_nans(xy[:, 1])], axis=1)
sp.append(speed(xy[:, 0], xy[:, 1]))
...
paw_speed[:, k] = np.nanmean(np.stack(sp, axis=1), axis=1)
```

iii. The trajectory final summary says paw speed was computed as the mean of the two tracked bottom-view paws and that class 2 was used for time points outside camera coverage or without visibility.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI thresholds paw speed at the session median over finite values. Below-median is class `0`, at-or-above-median is class `1`, and NaNs are class `2` (`not_visible`).

ii.
```python
paw_cls, paw_thresh = discretize(paw_speed)
```

```python
if ok.any():
    thresh = np.percentile(x[ok], 50)
    out[ok] = (x[ok] >= thresh).astype(np.int8)
```

iii. The trajectory final summary says all three movement variables were split at the session median and used a third class for absent visibility.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same `trial_video_times()` alignment as tongue velocity: frame times are shifted by the session video offset and the trial's go cue, then interpolated onto the neural time grid.

ii.
```python
tt, ts = trial_video_times(views[1][trix], vidshift, gocue[trix])
...
xy = interp_trace(tt, ts[:, 0:2, ix], time)
```

iii. The trajectory final summary groups paw speed with the other video-derived outputs and says they were aligned using `findVideoOffset` and the common bin centers.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy from the separate `motionEnergy_<anm>_<date>.mat` file, loading the `me` variable and unwrapping one or two `data` layers if necessary. It uses side-camera frame times from `traj[0]` to place that trace on the aligned time axis.

ii.
```python
me_path = os.path.join(folder, 'motionEnergy_%s_%s.mat' % (anm, date))
me = loadmat_var(me_path, 'me')
me_data = me.get('data') if isinstance(me, dict) else me
if isinstance(me_data, dict):
    me_data = me_data.get('data')
```

iii. The trajectory final summary says motion energy was one of the three movement variables aligned with the authors' video clock offset.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates each trial's motion-energy trace onto the neural bin centers using the side-camera frame times, fills interior NaNs within the covered interval, and then median-splits the resulting values. It does not leave motion energy at the original per-frame sampling or average frames within bins.

ii.
```python
y = np.asarray(me_data[trix], dtype=float).reshape(-1)
...
tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
...
me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
...
me_cls, me_thresh = discretize(me_trace)
```

iii. The trajectory final summary says motion energy was interpolated onto the common bin centers and split at the session median.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI thresholds motion energy at the session median over finite values. Below-median is class `0`, at-or-above-median is class `1`, and NaNs are class `2` (`no_video` in `OUTPUT_VALUES`).

ii.
```python
me_cls, me_thresh = discretize(me_trace)
...
OUTPUT_VALUES = [
    ...
    ['below_median', 'above_median', 'no_video'],
]
```

iii. The trajectory final summary says the movement variables were median-split and that the third class captures absent video coverage.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses side-camera frame times, subtracts the session video shift and the trial's `goCue`, then interpolates the motion-energy trace onto the same time vector used for the neural data.

ii.
```python
tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
...
me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
```

iii. The trajectory final summary says motion energy used the authors' `findVideoOffset` alignment and the common go-cue-centered bins.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several repair/fallback rules rather than only preserving missingness. If `frameTimes` are missing or all-NaN, it fabricates a nominal 400 Hz frame clock and forces `vidshift = 0.5`. If frame-count and `ts` length mismatch, it truncates to the shorter length. If `NdroppedFrames` is NaN, it discards that video trial. For paw and motion-energy traces it fills interior NaNs by nearest-neighbor imputation. Remaining NaNs are converted to class 2 during discretization.

ii.
```python
if ft is None or np.size(ft) == 0 or not np.any(np.isfinite(np.asarray(ft, float))):
    ft = (np.arange(ts.shape[0]) + 1) / 400.0
    vidshift = 0.5
...
if ft.size != ts.shape[0]:
    n = min(ft.size, ts.shape[0])
    ft, ts = ft[:n], ts[:n]
...
if nd is not None and np.size(nd) == 1 and not np.isfinite(float(np.asarray(nd, float))):
    return None, None
```

```python
xy = np.stack([fill_interior_nans(xy[:, 0]),
               fill_interior_nans(xy[:, 1])], axis=1)
...
me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
```

iii. The trajectory contains the inline comment that the missing-`frameTimes` fallback comes from `findPosition.m` using a nominal 400 Hz clock. The code comment on `fill_interior_nans` says the fill was restricted to interior gaps so uncovered time points could still be labeled `not visible` or `no video`.

## 11-a. What are the most time-consuming steps of the code?

i. The code structure suggests the most expensive steps are loading each full MATLAB session object with `load_obj`, then the nested per-unit/per-trial spike binning loop and the per-trial interpolation over video traces. Unlike the reference solution, the AI does not record an explicit runtime diagnosis in the code.

ii.
```python
obj = load_obj(path)
...
for prb in probes:
    ...
    for unit in units:
        ...
        for k in range(trials.size):
            if stops[k] > starts[k]:
                aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
                dat[:, k] = bin_spikes(aligned, edges)
```

```python
for k, trix in enumerate(trials):
    ...
    xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
```

iii. The trajectory does not provide a dedicated justification for runtime hotspots. This summary is therefore reconstructed from the control flow of `process_session()`.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the inner loop over retained trials for each unit's spike binning, plus the repeated per-trial interpolation loops for tongue, paw, and motion energy. The AI leaves these as Python loops.

ii.
```python
for unit in units:
    ...
    for k in range(trials.size):
        if stops[k] > starts[k]:
            aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
            dat[:, k] = bin_spikes(aligned, edges)
```

```python
for k, trix in enumerate(trials):
    ...
    tongue_speed[:, k] = speed(xy[:, 0], xy[:, 1])
...
for k, trix in enumerate(trials):
    ...
    me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
```

iii. The trajectory does not discuss vectorization tradeoffs. The code itself shows that spike counting was implemented with explicit trial loops rather than a session-wide histogram.

## 11-c. What processing does the code repeat multiple times?

i. The AI rebuilds the bin edges and time vector inside every session, repeatedly scans trial feature names with `feature_index`, separately interpolates each movement stream trial-by-trial, and recomputes per-session medians independently for tongue, paw, and motion energy. It does not cache these across sessions.

ii.
```python
edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
time = edges[:-1] + DT / 2
```

```python
tongue_ix = feature_index(views[TONGUE_FEATURE[1]], TONGUE_FEATURE[0])
paw_ix = [(feature_index(views[v], nm), v) for nm, v in PAW_FEATURES]
...
tongue_cls, tongue_thresh = discretize(tongue_speed)
paw_cls, paw_thresh = discretize(paw_speed)
me_cls, me_thresh = discretize(me_trace)
```

iii. The trajectory does not explicitly justify repeated processing. This section is reconstructed from the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extra per-session metadata that are not needed for the decoder arrays, such as `n_single_units`, exclusion counts, trial/video coverage summaries, thresholds, and the free-text `task` label. It also keeps the intermediate continuous `tongue_speed`, `paw_speed`, and `me_trace` arrays only to collapse them immediately into 3-class outputs.

ii.
```python
info = {
    'animal': anm,
    'date': date,
    'task': task,
    ...
    'n_single_units': int(sum(q.lower() in ('fair', 'good', 'great', 'excellent')
                              for q in qualities)),
    ...
    'tongue_velocity_threshold': float(tongue_thresh),
    'paw_velocity_threshold': float(paw_thresh),
    'motion_energy_threshold': float(me_thresh),
}
```

```python
tongue_speed = np.full((T, trials.size), np.nan)
paw_speed = np.full((T, trials.size), np.nan)
...
me_trace = np.full((T, trials.size), np.nan)
...
tongue_cls, tongue_thresh = discretize(tongue_speed)
paw_cls, paw_thresh = discretize(paw_speed)
me_cls, me_thresh = discretize(me_trace)
```

iii. The trajectory final summary mentions several of these statistics when reporting the finished dataset, which suggests the AI intentionally kept them for validation and reporting rather than because the decoder format required them.
