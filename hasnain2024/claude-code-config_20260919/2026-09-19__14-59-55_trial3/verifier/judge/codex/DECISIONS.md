# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a `SESSIONS` list of 44 `(animal, date, probes, data_dir, task)` tuples, then loops over that list. For each session it opens `data_structure_<anm>_<date>.mat` with a format-dispatching reader and separately loads `motionEnergy_<anm>_<date>.mat`. It supports both MATLAB v7.3/HDF5 and v7/MAT5 session files and several motion-energy layouts.

ii.
```python
SESSIONS = [
    ('EKH1',  '2021-08-07', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ...
    ('JEB24', '2023-11-03', [1],    DATA_RAND,  'DR randomized delay'),
]

def open_session(path):
    return _H5Session(path) if _is_hdf5(path) else _V7Session(path)

def load_motion_energy(path):
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
    me = d['me']
    ...
    return [np.asarray(x, float).ravel() for x in cells.ravel()]
```

iii. In `CONVERSION_NOTES.md`, the AI says the session list was transcribed from the authors' `load*_ALMVideo.m` files so excluded/commented-out sessions stay excluded, and that both file readers were needed because the released dataset mixes MATLAB formats and motion-energy layouts.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the `anm` element of each session tuple. During assembly, the AI preserves first-seen subject order, stores unique subject names in `subjects`, and stores one `subject_idx` per session.

ii.
```python
for k, (anm, date, probes, datadir, task) in enumerate(sessions):
    ...
    if anm not in subjects:
        subjects.append(anm)
    ...
    data['subject_idx'].append(subjects.index(anm))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The notes justify subject assignment from the reference load scripts and filenames; no separate in-file metadata source is used.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESSIONS` is treated as one session. One `data_structure_*.mat` file becomes one element of `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
for k, (anm, date, probes, datadir, task) in enumerate(sessions):
    res = process_session(anm, date, probes, datadir, task, show=show, ...)
    data['neural'].append(res['neural'])
    data['input'].append(res['input'])
    data['output'].append(res['output'])
```

iii. The notes say this mirrors the authors' session/probe lists exactly, including both fixed-delay and randomized-delay ephys sessions.

## 1-d. How are the data split into trials?

i. The AI uses `obj.Ntrials` as the trial count and reads all trial-indexed behavior arrays directly from `obj.bp`. Neural and video outputs are computed for all trial indices `0..ntrials-1`, and then the retained trial indices `kt = np.flatnonzero(keep)` define the per-trial entries written into the converted dataset.

ii.
```python
ntrials = obj.Ntrials
gocue = obj.bp('ev.goCue')
hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
...
kt = np.flatnonzero(keep)
for j, t in enumerate(kt):
    neural.append(np.ascontiguousarray(trialdat[:, :, t].T))
    inputs.append(inp.copy())
    ...
    outputs.append(out)
```

iii. The code assumes the raw files already provide trial boundaries; the notes do not record a separate justification beyond following the reference data organization.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with `early >= 0.5`, `stim.enable >= 0.5`, or non-finite go-cue time. It also drops any trial whose retained neural population has zero spikes in every bin, interpreting those as trials after the ephys recording stopped.

ii.
```python
keep = (early < 0.5) & (stim < 0.5) & np.isfinite(gocue)
spk_per_trial = trialdat.sum(axis=(0, 1))
no_ephys = spk_per_trial <= 0
keep &= ~no_ephys
```

iii. In the notes, the AI cites the reference condition strings `~stim.enable & ~early` and additionally argues that all-zero-neural trials in two sessions are recording-overrun artifacts that should be removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural representation is derived from per-cluster spike times and trial indices read via `obj.spikes(prb, i)` plus per-trial go-cue times from `bp.ev.goCue`.

ii.
```python
tt, tr = obj.spikes(prb, i)
...
gocue = obj.bp('ev.goCue')
al = tt - gocue[tr - 1]
```

iii. The notes explicitly map neural data to `clu{probe}(i).trialtm`, `clu{probe}(i).trial`, and `bp.ev.goCue`.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, binned into 10 ms bins over `[-2.5, 2.5]` s, converted to spikes/s by dividing by `DT`, and causally smoothed with a 15-bin Gaussian-like FIR (`my_smooth`) implemented with `lfilter`. The saved neural arrays are `(n_units, 500)` firing-rate matrices.

ii.
```python
DT = 0.01
SMOOTH_N = 15

b = np.floor((al - TMIN) / DT).astype(np.int64)
...
out /= DT
sm = my_smooth(out.reshape(NT, -1)).astype(np.float32)
return sm.reshape(NT, nunits, ntrials)
```

iii. The notes say this was intended as a port of `alignSpikes.m + getSeq.m + mySmooth.m`, and justify the causal smoothing as matching the reference single-trial representation `obj.trialdat`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first filters clusters by lower-cased manual quality label, dropping only `garbage`, `gabrga`, `noisy`, and `real?`. It then computes mean firing rate over the aligned window across all trials and keeps only units with `FR > 1 Hz`.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
...
keep_clu[prb] = [i for i, qq in enumerate(q) if qq.lower() not in BAD_QUALITY]
...
fr = mean_firing_rates(obj, probes, keep_clu, gocue, ntrials)
use = fr > LOW_FR
```

iii. The notes justify this as following `findClusters.m` and `removeLowFRClusters.m`, with the deliberate tweak that quality matching is case-insensitive.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is done by subtracting each spike's trial's go-cue time from its trial-relative spike time, i.e. `trialtm - goCue[trial]`.

ii.
```python
tt, tr = obj.spikes(prb, i)
...
al = tt - gocue[tr - 1]
```

iii. The notes explicitly cite `alignSpikes.m` and state that go cue is the alignment event for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`DT = 0.01`) over `[-2.5, 2.5]` s, giving 500 bins per trial. No later temporal rebinning is applied inside `convert_data.py`.

ii.
```python
TMIN, TMAX = -2.5, 2.5
DT = 0.01

def time_axis():
    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    ...
    return edges, t[:-1]
```

iii. The notes justify 10 ms as the `params.dt = 1/100` setting from the reference MATLAB code.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a dedicated raw variable. It is constructed from the shared time grid defined around the go cue using `TMIN`, `TMAX`, and `DT`; conceptually it is time relative to `bp.ev.goCue`.

ii.
```python
def time_axis():
    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    ...
    return edges, t[:-1]

inp = TAXIS.astype(np.float32)[None, :]
```

iii. The notes say the decoder task required time from go cue as an input, so the AI used the aligned bin centers as that variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI creates bin centers by taking `edges + DT/2`, dropping the extra last center, casting to `float32`, and copying the same `(1, NT)` row into every retained trial.

ii.
```python
t = edges + DT / 2
return edges, t[:-1]
...
inp = TAXIS.astype(np.float32)[None, :]
for j, t in enumerate(kt):
    inputs.append(inp.copy())
```

iii. No additional justification is recorded beyond reusing the neural time axis.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time input uses the exact same `TAXIS` bins as the neural data, so each input sample index corresponds to the same aligned interval as each neural bin.

ii.
```python
EDGES, TAXIS = time_axis()
...
trialdat = bin_spikes(obj, probes, keep_clu, gocue, ntrials)
...
inp = TAXIS.astype(np.float32)[None, :]
```

iii. The notes explicitly say the time input is the shared neural binning grid itself.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from per-trial behavioral flags `R`, `L`, `hit`, `miss`, and `no`.

ii.
```python
hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
R, L = obj.bp('R'), obj.bp('L')
```

iii. The notes say this follows the reference choice logic, with `none` added for ignore trials required by the decoder task.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI encodes `none=2` by default, then sets right licks for `(R & hit) | (L & miss)`, left licks for `(L & hit) | (R & miss)`, and keeps `none` on `no` trials. The label is then broadcast across all time bins of that trial.

ii.
```python
lickdir = np.full(ntrials, 2, dtype=np.int8)
lickdir[((R > 0.5) & (hit > 0.5)) | ((L > 0.5) & (miss > 0.5))] = 1
lickdir[((L > 0.5) & (hit > 0.5)) | ((R > 0.5) & (miss > 0.5))] = 0
lickdir[no > 0.5] = 2
...
out[0] = lickdir[t]
```

iii. The notes justify keeping ignore trials because the task explicitly asks for a `none` lick-direction class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from the per-trial `autowater` flag.

ii.
```python
autowater = obj.bp('autowater')
```

iii. The notes say `autowater` is the reference proxy for DR vs WC context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI maps `autowater > 0.5` to WC (`0`) and everything else to DR (`1`), then broadcasts that per trial.

ii.
```python
context = np.where(autowater > 0.5, 0, 1).astype(np.int8)
...
out[1] = context[t]
```

iii. The notes say this is a direct relabelling of the reference context flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit`, `miss`, and `no`.

ii.
```python
hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
```

iii. The notes describe ignore trials as a required third class for this task.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI initializes to `-1`, then maps miss to incorrect (`0`), hit to correct (`1`), and `no` to ignore (`2`), raising an error if any retained trial remains unassigned.

ii.
```python
outcome = np.full(ntrials, -1, dtype=np.int8)
outcome[miss > 0.5] = 0
outcome[hit > 0.5] = 1
outcome[no > 0.5] = 2
...
if outcome[keep].min() < 0:
    raise RuntimeError(...)
```

iii. The notes say this follows the task's required incorrect/correct/ignore coding.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera DeepLabCut feature `tongue` only, using that feature's `x/y` coordinates from `ts`, its `frameTimes`, and the session's video/ephys clock offset plus per-trial go-cue times.

ii.
```python
TONGUE_VIEW, TONGUE_FEAT = 0, 'tongue'
...
tongue_speed, tongue_vis = kinematic_speed(obj, TONGUE_VIEW, TONGUE_FEAT,
                                           gocue, ntrials, vidshift)
```

iii. The notes justify side-camera `tongue` as the tongue feature choice and describe the output as a single velocity variable rather than separate x/y velocities.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes a session video offset, interpolates the raw tracked `x` and `y` traces from the side camera directly onto the neural time axis, marks bins visible where interpolated `x` and `y` are finite, computes `vx` and `vy` with a NaN-aware discrete gradient, and takes `sqrt(vx^2 + vy^2)` as speed. It does not smooth the coordinates, does not compute frame-resolution velocities first, and does not combine side and bottom tongue views.

ii.
```python
tv = ft - vidshift - gocue[t]
x = interp_matlab(taxis, tv, xy[0])
y = interp_matlab(taxis, tv, xy[1])
vis = np.isfinite(x) & np.isfinite(y)
...
vx = nan_gradient(x)
vy = nan_gradient(y)
speed[t] = np.sqrt(vx ** 2 + vy ** 2)
```

iii. The notes give two justifications: use a single scalar speed because the task asks for one tongue-velocity output, and use a NaN-aware derivative so visible tongue bins are not artificially assigned zero speed at visibility-bout edges.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After trial filtering, the AI computes the 50th percentile of all valid tongue-speed samples in the retained trials of that session. Valid samples are labeled `0` if below threshold and `1` if at or above threshold; invalid samples get class `2`.

ii.
```python
def discretize(values, valid, thresh=None):
    v = np.full(values.shape, 2, dtype=np.int8)
    if thresh is None:
        thresh = np.nanpercentile(values[valid], 50) if valid.any() else np.nan
    if valid.any():
        v[valid] = (values[valid] >= thresh).astype(np.int8)
    return v, thresh

tng_d, tng_thr = discretize(tongue_speed[keep], tongue_vis[keep])
```

iii. The notes justify per-session thresholds because camera scales differ across sessions and because the task explicitly requests a per-session median split.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-wide video offset from the bitcode pulse, subtracts that offset and the trial's go cue from frame times, then interpolates tongue position onto the shared neural time axis `TAXIS`.

ii.
```python
def video_offset(obj):
    bitstart = obj.sglx_bitstart()
    fs = obj.sglx_fs()
    bp_bitstart = obj.bp('ev.bitStart')
    ...
    return (stats.mode(bitstart, keepdims=False).mode / fs
            - stats.mode(bp_bitstart, keepdims=False).mode)

tv = ft - vidshift - gocue[t]
x = interp_matlab(taxis, tv, xy[0])
```

iii. The notes cite `findVideoOffset.m` and say all video-derived streams should be aligned to the same go-cue-centered neural axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DeepLabCut feature `top_paw`, using its `ts` coordinates, `frameTimes`, video offset, and `goCue`.

ii.
```python
PAW_VIEW, PAW_FEAT = 1, 'top_paw'
...
paw_speed, paw_vis = kinematic_speed(obj, PAW_VIEW, PAW_FEAT,
                                     gocue, ntrials, vidshift)
```

iii. The notes justify `top_paw` as the paw feature used in the reference figure code and the reliable bottom-camera paw track.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI applies the same pipeline as tongue velocity but on `top_paw`: interpolate x/y directly onto `TAXIS`, compute NaN-aware gradients, take speed magnitude, then discretize. It uses only one view and does not smooth coordinates before differentiating.

ii.
```python
x = interp_matlab(taxis, tv, xy[0])
y = interp_matlab(taxis, tv, xy[1])
...
vx = nan_gradient(x)
vy = nan_gradient(y)
speed[t] = np.sqrt(vx ** 2 + vy ** 2)
```

iii. The notes say paw velocity should be a single scalar speed variable and use the same visibility-aware derivative logic as tongue velocity.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI uses the same `discretize` helper: 50th percentile over valid retained paw-speed samples in that session, with `2` for invalid bins.

ii.
```python
paw_d, paw_thr = discretize(paw_speed[keep], paw_vis[keep])
```

iii. The notes say this follows the task's requested per-session median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw alignment uses the same video offset and go-cue subtraction as tongue alignment, then interpolates onto `TAXIS`.

ii.
```python
tv = ft - vidshift - gocue[t]
x = interp_matlab(taxis, tv, xy[0])
y = interp_matlab(taxis, tv, xy[1])
```

iii. The notes treat all video-derived streams as sharing the same ephys-aligned time base.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<anm>_<date>.mat` file, one per-trial trace per session, together with side-camera frame times, the session video offset, and per-trial go-cue times.

ii.
```python
mefn = os.path.join(datadir, f'motionEnergy_{anm}_{date}.mat')
me_cells = load_motion_energy(mefn)
me = motion_energy_trace(obj, me_cells, gocue, ntrials, vidshift)
```

iii. The notes say the standalone motion-energy file is the canonical source and required custom handling because released files use multiple layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI loads the per-trial traces, gets frame times from the side camera, optionally falls back to a nominal 400 Hz clock, and interpolates each trace directly onto `TAXIS`. It does not smooth or differentiate the signal before discretization.

ii.
```python
for t in range(min(ntrials, len(me_cells))):
    m = me_cells[t]
    ...
    ft = obj.frame_times(0, t)
    if ft is None or ft.size != m.size or not np.all(np.isfinite(ft)):
        ft = (np.arange(1, m.size + 1) / 400.0) - 0.5 + vidshift
    out[t] = interp_matlab(taxis, ft - vidshift - gocue[t], m)
```

iii. The notes say motion energy already exists as one scalar per frame, so the only required transformation is alignment to the neural time axis before thresholding.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes a per-session median over valid retained motion-energy samples, assigns `0` below threshold, `1` at or above threshold, and `2` where no aligned video sample exists.

ii.
```python
me_vis = np.isfinite(me)
me_d, me_thr = discretize(me[keep], me_vis[keep])
```

iii. The notes say this matches the task's requested per-session 50th-percentile discretization and uses a third class for no-video bins.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session-wide video offset and the trial go-cue time from side-camera frame times, then interpolating onto the shared neural time axis.

ii.
```python
ft = obj.frame_times(0, t)
...
out[t] = interp_matlab(taxis, ft - vidshift - gocue[t], m)
```

iii. The notes describe motion energy as following the same video/ephys alignment rule as the tracked kinematics.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI adds compatibility code for mixed MATLAB formats and multiple motion-energy layouts. For video, if `frameTimes` are missing, non-finite, or length-mismatched, it falls back to a nominal 400 Hz time base; if `xy` is missing or `NdroppedFrames` is non-finite, it leaves that trial/video stream as invalid. Invalid tongue/paw bins become class `2`, and invalid motion-energy bins become class `2` via `np.isfinite`. Trials with missing ephys coverage are dropped entirely.

ii.
```python
if ft is None or not np.all(np.isfinite(ft)) or ft.size != xy.shape[1]:
    ft = np.arange(1, xy.shape[1] + 1) / 400.0 - 0.5 + vidshift
...
if xy is None or not np.isfinite(obj.ndropped(view, t)):
    continue
...
me_vis = np.isfinite(me)
...
no_ephys = spk_per_trial <= 0
keep &= ~no_ephys
```

iii. The notes justify the file-format fallbacks as necessary to read the released data and justify the invalid-bin classes as required because the decoder format forbids NaNs.

## 11-a. What are the most time-consuming steps of the code?

i. The AI's own timing table says the video step is the main cost, followed by curation and then spike binning/smoothing. The code records per-session timings for `curation`, `neural`, `video`, and `total`.

ii.
```python
timing = {}
...
timing['curation'] = time.time() - t0
...
timing['neural'] = time.time() - t1
...
timing['video'] = time.time() - t1
...
timing['total'] = time.time() - t0
```

iii. In `CONVERSION_NOTES.md`, the AI reports about `~60 s` total for video, `~35 s` for curation, and `~10 s` for spike binning over the full 44-session conversion.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes spike binning with a flattened `(trial, bin)` index and vectorizes smoothing over all units/trials at once. The remaining obvious loops are trial loops in `kinematic_speed` and `motion_energy_trace`, plus probe/unit loops in `bin_spikes` and `mean_firing_rates`.

ii.
```python
idx = (tr[m] - 1) * NT + b[m]
cnt = np.bincount(idx, minlength=ntrials * NT)
...
sm = my_smooth(out.reshape(NT, -1)).astype(np.float32)
...
for t in range(ntrials):
    xy, ft = obj.traj_xy(view, t, featix)
```

iii. The notes explicitly say the AI removed per-spike Python loops and per-unit smoothing loops, leaving the irregular per-trial video loops in place.

## 11-c. What processing does the code repeat multiple times?

i. The AI code re-traverses the same retained spike trains twice, once in `mean_firing_rates` for low-FR filtering and again in `bin_spikes` for the final neural tensor. It also runs separate full per-trial passes for tongue, paw, and motion-energy signals.

ii.
```python
fr = mean_firing_rates(obj, probes, keep_clu, gocue, ntrials)
...
trialdat = bin_spikes(obj, probes, keep_clu, gocue, ntrials)
...
tongue_speed, tongue_vis = kinematic_speed(...)
paw_speed, paw_vis = kinematic_speed(...)
me = motion_energy_trace(...)
```

iii. There is no explicit justification for the repeated passes, though the notes imply they were acceptable because the total runtime stayed near two minutes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores several diagnostic/provenance quantities that are not needed for the decoder-facing arrays: `sample` and `delay` times for metadata/plots, `qualities`, `unit_probe`, many summary fractions and thresholds in `info`, and optional plotting support. These are useful for validation but not used by downstream decoding.

ii.
```python
sample = obj.bp('ev.sample')
delay = obj.bp('ev.delay')
...
qualities = {}
...
unit_quality, unit_region, unit_probe = [], [], []
...
info = dict(
    ...,
    median_delay_s=float(np.nanmedian(gocue - delay)),
    median_sample_s=float(np.nanmedian(delay - sample)),
    tongue_speed_threshold=float(tng_thr), paw_speed_threshold=float(paw_thr),
    motion_energy_threshold=float(me_thr),
    frac_context_WC=float(np.mean(context[keep] == 0)),
    ...
)
```

iii. The notes frame these extras as sanity-checking and provenance support rather than core conversion logic.
