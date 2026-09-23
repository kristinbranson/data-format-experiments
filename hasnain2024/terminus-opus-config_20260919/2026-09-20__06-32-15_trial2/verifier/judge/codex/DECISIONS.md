# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 44-session `SESSIONS` list transcribed from the authors' loader scripts. Each entry includes animal, date, probe(s), source directory, and task label. It loads each `data_structure_*.mat` session file with a custom mixed-format loader that handles MATLAB v7 and v7.3, reads only selected Bpod, cluster, trajectory, and sync fields, and loads `motionEnergy_*.mat` separately during per-session processing. Sessions are processed in parallel with `multiprocessing.Pool`.

ii.
```python
SESSIONS = [
    ('JEB6',  '2021-04-18', [2],    DATA_FIXED, 'fixed delay'),
    ...
    ('JEB24', '2023-11-03', [1],    DATA_RAND,  'randomized delay'),
]
```
```python
def load_session(fn, probes, feats):
    return _load_v73(fn, probes, feats) if is_v73(fn) else _load_v7(fn, probes, feats)
```
```python
fn = os.path.join(ddir, 'data_structure_%s_%s.mat' % (anm, date))
sess = load_session(fn, probes, feats)
...
mefn = os.path.join(ddir, 'motionEnergy_%s_%s.mat' % (anm, date))
me = load_motion_energy(mefn) if os.path.exists(mefn) else None
```

iii. In `CONVERSION_NOTES.md`, the AI says it used the uncommented author session loaders as the authoritative inclusion list, read only required fields for speed, and processed sessions in parallel.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `anm` field from each hard-coded session tuple. During assembly, the first occurrence of each animal is appended to `subjects`, and each session gets the index of its animal in that list.

ii.
```python
for r in results:
    ...
    if r['anm'] not in data['subjects']:
        data['subjects'].append(r['anm'])
    data['subject_idx'].append(data['subjects'].index(r['anm']))
```

iii. The notes say the session list is derived from the author loaders and that subject identity comes from the animal id in the filename/session tuple.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESSIONS` is treated as one session. `process_session` loads one `data_structure_<animal>_<date>.mat` file and returns one session-level entry for `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
def process_session(args):
    anm, date, probes, ddir, task, show = args
    fn = os.path.join(ddir, 'data_structure_%s_%s.mat' % (anm, date))
    ...
    result = dict(ok=True, neural=neural, input=inputs, output=outputs,
                  regions=regions, anm=anm, info=info)
```
```python
data['neural'].append(r['neural'])
data['input'].append(r['input'])
data['output'].append(r['output'])
```

iii. The AI justifies this as matching the paper's loader scripts, which enumerate 44 ephys sessions explicitly.

## 1-d. How are the data split into trials?

i. Trials are defined by the first `N = bp['Ntrials'][0]` Bpod trials. Per-trial behavioral arrays are sliced to those trials, and kept-trial indices are formed from boolean trial filters. Neural spikes retain their raw `cluster['trial']` labels and are mapped into those kept trial rows.

ii.
```python
bp = sess['bp']
N = int(bp['Ntrials'][0])
...
keep = (~stim[:N]) & (~early[:N]) & valid_align
keep_trials = np.nonzero(keep)[0]
```
```python
tr = c['trial'][i].astype(np.int64) - 1
...
keep_idx = {t: i for i, t in enumerate(keep_trials)}
```

iii. The notes describe `obj.bp` as the authoritative per-trial table and treat trial indices from Bpod and `clu.trial` as the trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI removes photostimulation trials (`stim.enable`), early-lick trials (`early`), and trials with non-finite go-cue times. After neural binning it also drops kept trials whose neural activity is all zero across all units and bins, interpreting those as behavioral trials that continued after ephys recording stopped.

ii.
```python
valid_align = np.isfinite(align[:N])
keep = (~stim[:N]) & (~early[:N]) & valid_align
keep_trials = np.nonzero(keep)[0]
```
```python
has_spikes = rates.sum(axis=(1, 2)) > 0
...
rates = rates[has_spikes]
keep_trials = keep_trials[has_spikes]
```

iii. The notes say `~stim.enable` and `~early` come directly from the reference condition strings, and the extra all-zero-neural trial drop was added after the AI found sessions where SpikeGLX ended before behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu` fields for the selected probe(s): `quality`, `trialtm`, and `trial`, plus `bp.ev.goCue` for alignment.

ii.
```python
clu[p] = dict(
    quality=[...],
    trialtm=[...],
    trial=[...])
```
```python
align = ev[ALIGN_EVENT].astype(float)
...
tr = c['trial'][i].astype(np.int64) - 1
tm = c['trialtm'][i]
aligned = tm[sel] - align_times[tr[sel]]
```

iii. The notes explicitly map neural output to `obj.clu{probe}(i).trialtm`, `.trial`, `.quality`, aligned to `bp.ev.goCue`.

## 2-b. How is the `neural` data processed?

i. For each kept cluster, spikes are aligned to go cue, counted into 10 ms bins over `[-2.5, 2.5]`, converted to spikes/s, smoothed with a causal Gaussian kernel (`my_smooth`, `SMOOTH_N=15`, reflect padding), then averaged in groups of 5 bins to produce 50 ms decoder bins.

ii.
```python
DT_FINE = 0.01
SMOOTH_N = 15
DOWNSAMPLE = 5
DT_OUT = DT_FINE * DOWNSAMPLE
```
```python
b = np.floor((aligned - TMIN) / DT_FINE).astype(np.int64)
...
rate = counts / DT_FINE
rate = my_smooth(rate)
rates.append(downsample_mean(rate).astype(np.float32))
```

iii. The notes justify this as following the reference `getSeq.m` and `mySmooth.m` at 10 ms resolution, then averaging into 50 ms bins as a decoder-oriented compromise.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps clusters whose lower-cased `quality` label is not in `{'garbage', 'gabrga', 'noisy', 'real?'}` and then drops clusters whose mean firing rate across kept trials and time bins is `<= 1 Hz`. Sessions with fewer than 10 surviving units are excluded.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10
```
```python
good = [i for i, q in enumerate(c['quality'])
        if q.strip().lower() not in BAD_QUALITY]
...
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
...
if rates.shape[1] < MIN_UNITS:
    return dict(ok=False, reason='fewer than %d units (%d)' % (MIN_UNITS, rates.shape[1]), ...)
```

iii. The notes cite `findClusters.m`, `removeLowFRClusters.m`, and the paper's session inclusion statement as the justification.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every spike time is aligned by subtracting that spike's trial go-cue time from `trialtm`, i.e. `trialtm - goCue[trial]`.

ii.
```python
tr = c['trial'][i].astype(np.int64) - 1
tm = c['trialtm'][i]
aligned = tm[sel] - align_times[tr[sel]]
```

iii. The AI states in the notes that this matches `alignSpikes.m` and the paper's use of go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The final converted data uses 50 ms bins. The AI first computes neural data on a 10 ms grid and then averages every 5 fine bins into one 50 ms output bin.

ii.
```python
DT_FINE = 0.01
DOWNSAMPLE = 5
DT_OUT = DT_FINE * DOWNSAMPLE
```
```python
def downsample_mean(x, k=DOWNSAMPLE, nanmean=False):
    n = (x.shape[-1] // k) * k
    y = x[..., :n].reshape(x.shape[:-1] + (n // k, k))
    ...
    return y.mean(axis=-1)
```

iii. The notes say this was chosen as a decoder-oriented compromise while keeping the reference's fine-bin processing before averaging.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The actual input vector is not read from a raw field. It is constructed from the configured aligned time axis, defined relative to the go cue, and reused for every trial.

ii.
```python
def time_axes():
    edges = np.round(np.arange(TMIN, TMAX + 1e-9, DT_FINE), 6)
    centres = edges[:-1] + DT_FINE / 2.0
    ...
    centres_out = centres[:nout * DOWNSAMPLE].reshape(nout, DOWNSAMPLE).mean(axis=1)
    return edges, centres, centres_out
```
```python
tin = TOUT.astype(np.float32)[None, :]
inputs = [tin.copy() for _ in range(kt.size)]
```

iii. The notes describe `time_from_go_cue` as a constructed decoder input based on the aligned bin centers.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes fine 10 ms bin centers over `[-2.5, 2.5]` and then averages each consecutive block of 5 centers to produce a 50 ms output-time vector.

ii.
```python
edges = np.round(np.arange(TMIN, TMAX + 1e-9, DT_FINE), 6)
centres = edges[:-1] + DT_FINE / 2.0
...
centres_out = centres[:nout * DOWNSAMPLE].reshape(nout, DOWNSAMPLE).mean(axis=1)
```

iii. The notes justify this as matching the chosen 50 ms decoder binning.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same `TOUT` bin-center vector used after neural downsampling, so each input timepoint corresponds to the same 50 ms interval as the neural data.

ii.
```python
EDGES, TCENT, TOUT = time_axes()
...
rates.append(downsample_mean(rate).astype(np.float32))
...
tin = TOUT.astype(np.float32)[None, :]
```

iii. The notes say `time_from_go_cue` is the decoder time grid itself.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from per-trial Bpod fields `R`, `L`, `hit`, `miss`, and implicitly `no` via the default `none` class.

ii.
```python
hit = np.nan_to_num(bp['hit']).astype(bool)
miss = np.nan_to_num(bp['miss']).astype(bool)
no = np.nan_to_num(bp['no']).astype(bool)
R = np.nan_to_num(bp['R']).astype(bool)
L = np.nan_to_num(bp['L']).astype(bool)
```

iii. The notes cite `getPrevChoice.m` and an independent check against first post-go-cue lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI initializes all kept trials to class 2 (`none`), sets class 1 (`right`) on `(R & hit) | (L & miss)`, and class 0 (`left`) on `(L & hit) | (R & miss)`. The resulting label is then repeated across all time bins of the trial.

ii.
```python
lick_dir = np.full(kt.size, 2, dtype=np.int64)
right_lick = (R[kt] & hit[kt]) | (L[kt] & miss[kt])
left_lick = (L[kt] & hit[kt]) | (R[kt] & miss[kt])
lick_dir[right_lick] = 1
lick_dir[left_lick] = 0
```
```python
o[0] = lick_dir[i]
```

iii. The notes say this matches the reference choice definition and keeps ignore/no-response trials as `none` because the decoder requires that class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp['autowater']`.

ii.
```python
aw = np.nan_to_num(bp['autowater']).astype(bool)
```

iii. The notes say `autowater==1` is used as the WC/DR context flag, following the paper/code.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater=True` becomes class 0 (`WC`) and `False` becomes class 1 (`DR`), repeated across all bins in the trial.

ii.
```python
context = np.where(aw[kt], 0, 1).astype(np.int64)
...
o[1] = context[i]
```

iii. The notes justify this as the direct DR/WC mapping used in the reference code.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from Bpod `hit`, `miss`, and `no`.

ii.
```python
hit = np.nan_to_num(bp['hit']).astype(bool)
miss = np.nan_to_num(bp['miss']).astype(bool)
no = np.nan_to_num(bp['no']).astype(bool)
```

iii. The notes say the decoder requires retaining ignore/no-response trials as an explicit class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI initializes every kept trial to class 2 (`ignore`), then sets hits to class 1 (`correct`) and misses to class 0 (`incorrect`), and repeats the label across time bins.

ii.
```python
outcome = np.full(kt.size, 2, dtype=np.int64)
outcome[hit[kt]] = 1
outcome[miss[kt]] = 0
...
o[2] = outcome[i]
```

iii. The notes justify the relabeling as matching the prompt's required categorical outputs.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera DeepLabCut feature `('side', 'tongue')`, its `xy` trajectories and `frameTimes`, plus session video/ephys sync (`sglx.bitcode.bitstart`, `sglx.fs`, `bp.ev.bitStart`) and per-trial `goCue`.

ii.
```python
TONGUE_FEAT = ('side', 'tongue')
```
```python
vidshift = video_offset(sess)
tongue_sp, _ = feature_speed(sess, TONGUE_FEAT, kt, vidshift, align, fill_missing=False)
```

iii. The notes say the AI chose the side-camera tongue feature and aligned it with the reference `findVideoOffset` procedure.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates tongue x/y positions onto the common 10 ms time grid using offset-corrected frame times, leaves missing values unfilled (`fill_missing=False`), computes velocity with `np.gradient` on the interpolated positions, converts to speed with `hypot`, averages into 50 ms bins with `nanmean`, and then discretizes per session.

ii.
```python
for d in range(2):
    pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], left=np.nan, right=np.nan)
...
vel = np.gradient(pos, axis=0)
sp = np.hypot(vel[:, 0], vel[:, 1])
sp[~vis] = np.nan
```
```python
tongue_ds = downsample_mean(tongue_sp, nanmean=True)
tongue_cls, tongue_thr = discretize(tongue_ds)
```

iii. The notes say this was intended to mirror the reference kinematic pipeline while preserving `not visible` bins for tongue.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI pools all finite binned tongue-speed values within a session, computes the 50th percentile, labels values below it as 0, values at or above it as 1, and assigns class 2 to NaN bins.

ii.
```python
def discretize(values, nan_class=2):
    finite = np.isfinite(values)
    cls = np.full(values.shape, nan_class, dtype=np.int64)
    if finite.any():
        thr = np.percentile(values[finite], 50)
        cls[finite & (values >= thr)] = 1
        cls[finite & (values < thr)] = 0
```
```python
tongue_cls, tongue_thr = discretize(tongue_ds)
```

iii. The notes explicitly cite the decoder specification's per-session median split and class 2 for not-visible bins.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a per-session video offset, subtracts it and the trial's go cue from frame times, interpolates tongue positions onto the common aligned 10 ms grid, and then averages to the same 50 ms grid used for neural data.

ii.
```python
def video_offset(sess):
    bs = sess['sglx']['bitstart']
    bstart = sess['bp']['ev']['bitStart']
    ...
    return vid_file_offset - float(smode(np.round(bstart, 6), keepdims=False).mode)
```
```python
def frame_times_for_trial(sess, trial, vidshift, align_t):
    ft = sess['frameTimes'][trial]
    ...
    return ft - vidshift - align_t
```

iii. The notes justify this as using the reference `findVideoOffset.m` synchronization.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera DeepLabCut features, `('bottom', 'top_paw')` and `('bottom', 'bottom_paw')`, plus the same frame-time, sync, and go-cue variables used for tongue alignment.

ii.
```python
PAW_FEATS = [('bottom', 'top_paw'), ('bottom', 'bottom_paw')]
```
```python
paw_sps = [feature_speed(sess, k, kt, vidshift, align, fill_missing=True)[0] for k in PAW_FEATS]
```

iii. The notes say paws are tracked in the bottom view and that the AI chose to average the two paw markers.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, the AI interpolates x/y onto the 10 ms aligned grid, nearest-fills missing positions (`fill_missing=True`), computes `np.gradient` velocity, subtracts a baseline derivative term, converts to speed, averages the two paw-feature speed matrices with `nanmean`, downsamples to 50 ms bins, and discretizes per session.

ii.
```python
if fill_missing:
    if vis.any():
        idx = np.arange(NFINE)
        pos = np.stack([np.interp(idx, idx[vis], pos[vis, d]) for d in range(2)], axis=1)
...
vel = np.gradient(pos, axis=0)
if fill_missing:
    base = np.nanmedian(np.diff(pos, axis=0), axis=0)
    vel = vel - base[0]
```
```python
paw_sp = np.nanmean(np.stack(paw_sps, axis=0), axis=0)
paw_ds = downsample_mean(paw_sp, nanmean=True)
paw_cls, paw_thr = discretize(paw_ds)
```

iii. The notes say this was meant to follow the reference's non-tongue fill policy while producing the required discrete decoder target.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. As with tongue velocity, the AI uses a per-session median split over finite 50 ms paw-speed bins, with NaNs assigned to class 2.

ii.
```python
paw_cls, paw_thr = discretize(paw_ds)
```
```python
cls[finite & (values >= thr)] = 1
cls[finite & (values < thr)] = 0
```

iii. The notes say this thresholding comes directly from the decoder specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are offset-corrected with `video_offset`, shifted by the trial go cue, interpolated onto the 10 ms aligned axis, and averaged to the same 50 ms output bins used for neural data.

ii.
```python
ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
...
pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], left=np.nan, right=np.nan)
...
paw_ds = downsample_mean(paw_sp, nanmean=True)
```

iii. The notes describe the paw alignment as sharing the same synchronization procedure and output time grid as the neural data.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the external `motionEnergy_<animal>_<date>.mat` file, specifically the per-trial `me['data']` traces, plus side-camera frame times and the same session/trial sync variables used for video alignment.

ii.
```python
def load_motion_energy(fn):
    m = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)
    me = m['me']
    ...
    data = [np.atleast_1d(np.array(d).ravel()).astype(float) for d in np.atleast_1d(raw)]
    return dict(data=data, moveThresh=thresh)
```
```python
mefn = os.path.join(ddir, 'motionEnergy_%s_%s.mat' % (anm, date))
me = load_motion_energy(mefn) if os.path.exists(mefn) else None
```

iii. The notes say the AI used the standalone motion-energy files because their structure is known and they cover the included sessions.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates each trial's motion-energy trace onto the aligned 10 ms grid, nearest-fills remaining NaNs, averages into 50 ms bins, and then discretizes with a per-session median threshold.

ii.
```python
v = np.interp(TCENT, ft[valid], y[valid], left=np.nan, right=np.nan)
ok = np.isfinite(v)
if not ok.all() and ok.any():
    idx = np.arange(NFINE)
    v = np.interp(idx, idx[ok], v[ok])
out[r] = v
```
```python
me_ds = downsample_mean(me_fine, nanmean=True)
me_cls, me_thr = discretize(me_ds)
```

iii. The notes justify this as following `loadMotionEnergy.m` for alignment, with the decoder specification replacing the paper's manual threshold by a median split.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI applies the same `discretize` function: per-session 50th percentile over finite values, with NaN bins assigned class 2 (`no_video` in metadata/output labels).

ii.
```python
me_cls, me_thr = discretize(me_ds)
```
```python
data['output_values'] = [
    ...
    ['below_median', 'above_median', 'no_video'],
]
```

iii. The notes explicitly say the prompt's 50th-percentile threshold replaces the original session-specific `moveThresh`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the same session video offset and go-cue subtraction as the tracked video features, is interpolated onto the common aligned 10 ms grid, and is then averaged to the 50 ms neural output grid.

ii.
```python
def motion_energy_trials(me, sess, keep_trials, vidshift, align_times):
    ...
    ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
    ...
    v = np.interp(TCENT, ft[valid], y[valid], left=np.nan, right=np.nan)
```

iii. The notes describe motion energy as sharing the same `findVideoOffset`-style alignment path as the other video signals.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles multiple data-format irregularities explicitly: mixed MATLAB v7/v7.3 files, multiple `motionEnergy` layouts, sessions missing `obj.clu`, missing motion-energy files, missing `obj.ex`, unlabeled quality strings, and trials after ephys ended. For video, if `frameTimes` are empty or all NaN it falls back to synthetic `(1:n)/400 - 0.5` timing; if `NdroppedFrames` is NaN it skips that trial for the feature; if frame-time and trajectory lengths differ it truncates to the shorter length.

ii.
```python
def frame_times_for_trial(sess, trial, vidshift, align_t):
    ft = sess['frameTimes'][trial]
    if ft.size == 0 or np.all(np.isnan(ft)):
        ft = (np.arange(1, sess['nframes'][trial] + 1) / VIDEO_FS)
        return ft - 0.5 - align_t
    return ft - vidshift - align_t
```
```python
if t < len(nd) and np.isnan(nd[t]):
    continue
...
if ft.size != xy.shape[0] or ft.size < 2:
    m = min(ft.size, xy.shape[0])
    if m < 2:
        continue
    ft, xy = ft[:m], xy[:m]
```

iii. The notes justify these as edge-case handling needed to make the full dataset convert cleanly and to mirror the reference fallback for frame times.

## 11-a. What are the most time-consuming steps of the code?

i. The AI says file loading dominates runtime, especially reading large HDF5 `data_structure` files. Neural binning is the next-largest cost for large sessions; video and motion-energy processing are comparatively small.

ii.
```python
t0 = time.time()
sess = load_session(fn, probes, feats)
t_load = time.time() - t0
...
t1 = time.time()
rates, ids = bin_spikes(...)
t_neural = time.time() - t1
...
t2 = time.time()
...
t_video = time.time() - t2
```

iii. `CONVERSION_NOTES.md` contains explicit timing tables stating that loading is the dominant cost and reporting per-session load/neural/video timings.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI says it already vectorized the expensive spike binning and smoothing steps, and left mostly ragged per-trial video loops that are hard to fully vectorize because frame counts differ by trial.

ii.
```python
flat = row[sel][inwin] * NFINE + b[inwin]
np.add.at(counts.reshape(-1), flat, 1.0)
```
```python
for r, t in enumerate(keep_trials):
    ...
for p in probes:
    ...
for i in good:
```

iii. The notes explicitly present `np.add.at`, FFT convolution, and multiprocessing as the key vectorization/parallelization choices.

## 11-c. What processing does the code repeat multiple times?

i. The AI's stated position is that it avoids major recomputation: the fine time axes and smoothing kernel are built once, video offset is computed once per session, and each feature stream is processed once per session before discretization.

ii.
```python
KERN = causal_kernel()
EDGES, TCENT, TOUT = time_axes()
```
```python
vidshift = video_offset(sess)
tongue_sp, _ = feature_speed(...)
paw_sps = [feature_speed(...) for k in PAW_FEATS]
me_fine = motion_energy_trials(...)
```

iii. The notes repeatedly justify intermediate caching/reuse as part of the AI's efficiency strategy.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI's documented position is that it minimizes unnecessary work by loading only required probes/features and by producing only the decoder-required outputs. The main discarded intermediates are the fine-resolution 10 ms neural/video signals, which are used only to create the 50 ms outputs.

ii.
```python
def _load_v73(fn, probes, feats):
    """Load only what we need from a v7.3 (HDF5) data_structure file."""
```
```python
rate = counts / DT_FINE
rate = my_smooth(rate)
rates.append(downsample_mean(rate).astype(np.float32))
```
```python
tongue_ds = downsample_mean(tongue_sp, nanmean=True)
paw_ds = downsample_mean(paw_sp, nanmean=True)
me_ds = downsample_mean(me_fine, nanmean=True)
```

iii. The notes emphasize selective loading and speed-oriented implementation rather than identifying much discarded processing beyond the temporary fine-bin representations.
