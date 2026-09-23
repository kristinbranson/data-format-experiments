# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script starts from a session list returned by `get_sessions()`, then processes each session by loading one `data_structure_*.mat` file with `load_obj`. Motion energy is loaded separately per session from `sess['mefile']` if present.

ii. ```python
sessions = get_sessions()
...
obj = load_obj(sess['datafile'])
...
if sess.get('mefile'):
    try:
        me_trials, _thr = load_motion_energy(sess['mefile'])
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this by saying the authoritative session/probe list should come from the reference `load<ANM>_ALMVideo.m` scripts, and that the `.mat` files must be handled with a uniform loader because the release mixes MATLAB v7.3 and v5 formats.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the parsed session metadata as `sess['anm']`. During assembly, new animal IDs are appended to `data['subjects']`, and each kept session gets a matching integer in `subject_idx`.

ii. ```python
anm = sess['anm']
if anm not in data['subjects']:
    data['subjects'].append(anm)
data['subject_idx'].append(data['subjects'].index(anm))
```

iii. The notes say the animal ID in the session metadata / filename is more reliable than fields inside the MATLAB object, so the AI carried that ID through the conversion.

## 1-c. How are the data split into sessions?

i. One parsed metadata entry becomes one session. `main()` iterates over the session list, calls `process_session` once per entry, and appends each result as one element of `data['neural']`, `data['input']`, and `data['output']`.

ii. ```python
jobs = [(s, args.show_processing, outdir) for s in sessions]
...
for sess, res in zip(sessions, results):
    ...
    data['neural'].append(res['neural'])
    data['input'].append(res['input'])
    data['output'].append(res['output'])
```

iii. The notes justify this by treating the meta-script session list as the same unit of analysis used by the authors: one recording day, one file, one output session.

## 1-d. How are the data split into trials?

i. Trials are indexed from the Bpod table. The code reads `Ntrials` and all per-trial flags from `obj['bp']`, builds a boolean mask over trials, then converts kept trials into 1-based MATLAB-style trial numbers.

ii. ```python
N = int(as_vector(bp['Ntrials'])[0])
...
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
trials = np.where(keep)[0] + 1
align_times = gocue[trials - 1]
```

iii. The notes and trajectory say the Bpod trial table is the authoritative trial structure, so no trial boundaries need to be reconstructed from spikes or video.

## 1-e. How are trials filtered based on quality controls?

i. The code drops early-lick trials, stimulation trials, trials with non-finite go-cue times, and trials that are not marked `hit`, `miss`, or `no`. After spike binning, it also drops any kept trial with zero total spikes across all quality-passing units by testing `spk_per_trial > 0`.

ii. ```python
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
...
rates, quals, spk_per_trial, units_per_probe = bin_spikes(...)
recorded = spk_per_trial > 0
if n_not_recorded:
    trials = trials[recorded]
    align_times = align_times[recorded]
    rates = rates[recorded]
```

iii. `CONVERSION_NOTES.md` says early and stimulation trials were excluded to match the paper, and says the zero-spike filter was added because some sessions continue behavior after ephys recording stops, producing warning-generating all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from spike-sorted clusters on the selected probe(s): `c['trial']`, `c['trialtm']`, and `c['quality']` inside each cluster returned by `get_clusters`. Alignment additionally uses the Bpod go-cue times from `bp['ev'][PARAMS['alignEvent']]`.

ii. ```python
clusters = get_clusters(obj, p)
...
tr = np.asarray(c['trial'], dtype=np.int64).ravel()
tm = np.asarray(c['trialtm'], dtype=float).ravel()
...
gocue = as_vector(bp['ev'][PARAMS['alignEvent']])[:N]
```

iii. The notes justify this as a direct port of the reference pipeline: clusters provide within-trial spike times, and `goCue` is the common alignment event for the whole analysis.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, placed into 10 ms bins from -2.5 s to 2.5 s, converted to spikes/s, and smoothed with a causal Gaussian kernel of width 15 bins using `my_smooth`. The per-trial output stored in the pickle is `rates[i].T`, so each trial becomes `(n_units, T)`.

ii. ```python
PARAMS = dict(
    tmin=-2.5,
    tmax=2.5,
    dt=0.01,
    smooth=15,
    bctype='reflect',
)
...
aligned = tm[sel] - align_times[pos]
b = np.floor((aligned - edges[0]) / PARAMS['dt']).astype(np.int64)
...
rates = counts / PARAMS['dt']
flat = rates.transpose(1, 0, 2).reshape(nb, -1)
sm = my_smooth(flat)
rates = sm.reshape(nb, ntr, -1).transpose(1, 0, 2).astype(np.float32)
```

iii. The AI explicitly justified the 10 ms / causal-kernel choice in its notes and trajectory by following `WorkingWithDataObjs.m` and `utils/mySmooth.m`, and by choosing the tutorial-style single-trial processing it believed the authors used for decoder inputs.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only from the selected probe(s). They are filtered first by string quality label, dropping labels in `{'garbage', 'gabrga', 'noisy', 'real?', ''}`, and then by mean firing rate, keeping only units with `meanfr > 1.0`.

ii. ```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}
...
keep = [i for i, c in enumerate(clusters)
        if str(c.get('quality', '')).strip().lower() not in BAD_QUALITY]
...
meanfr = rates.mean(axis=(0, 1))
keep_u = meanfr > PARAMS['lowFR']
rates = rates[:, :, keep_u]
```

iii. The notes justify the quality and firing-rate filters by citing `findClusters.m` and the paper’s “FR > 1 Hz” statement. They also state that only the ALM probes from the reference metadata should be used.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go-cue time of the same trial before binning: `aligned = trialtm - goCue[trial]`.

ii. ```python
gocue = as_vector(bp['ev'][PARAMS['alignEvent']])[:N]
...
aligned = tm[sel] - align_times[pos]
```

iii. The notes repeatedly cite `alignSpikes.m` and state that the whole dataset should be aligned to `obj.bp.ev.goCue`, including WC trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 10 ms bins over a 5 s window, giving 500 time points per trial. No additional temporal rebinning is applied after that.

ii. ```python
PARAMS = dict(
    tmin=-2.5,
    tmax=2.5,
    dt=0.01,
)
...
edges = np.arange(PARAMS['tmin'], PARAMS['tmax'] + 1e-12, PARAMS['dt'])
tm = edges[:-1] + PARAMS['dt'] / 2
```

iii. The AI justified 10 ms bins in `CONVERSION_NOTES.md` and the trajectory by preferring the tutorial parameters (`dt = 1/100`) over the 5 ms default in `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input time vector is synthesized from the configured alignment window and bin size, then interpreted relative to the per-trial go cue. The code does not recompute a unique time vector from raw timestamps for each trial.

ii. ```python
def time_axis():
    edges = np.arange(PARAMS['tmin'], PARAMS['tmax'] + 1e-12, PARAMS['dt'])
    tm = edges[:-1] + PARAMS['dt'] / 2
    return edges, tm
...
gocue = as_vector(bp['ev'][PARAMS['alignEvent']])[:N]
```

iii. The notes justify this as the decoder’s only required input: the common bin-center time axis around the go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script builds a uniform 10 ms bin-center axis from -2.5 s to +2.5 s, casts it to `float32`, and copies the same `(1, T)` array into every trial.

ii. ```python
_, tm = time_axis()
...
tin = taxis.astype(np.float32)[None, :]
for i in range(ntr):
    inp.append(tin.copy())
```

iii. The AI’s stated justification is simply that the decoder input should be the aligned neural time grid itself.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same `taxis` used to bin spikes and resample video-derived outputs, so every input time point corresponds to the same bin index used by neural and behavioral arrays.

ii. ```python
edges, taxis = time_axis()
...
b = np.floor((aligned - edges[0]) / PARAMS['dt']).astype(np.int64)
...
tin = taxis.astype(np.float32)[None, :]
```

iii. The AI explicitly framed this in the notes as “the neural binning grid itself” being reused as the decoder input.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial task side and outcome flags: `R`, `L`, `hit`, `miss`, and `no`.

ii. ```python
hit = as_vector(bp['hit'])[:N].astype(bool)
miss = as_vector(bp['miss'])[:N].astype(bool)
no = as_vector(bp['no'])[:N].astype(bool)
R = as_vector(bp['R'])[:N].astype(bool)
L = as_vector(bp['L'])[:N].astype(bool)
```

iii. The notes justify this because the raw files do not directly store “lick direction” as a categorical variable, so it has to be inferred from instructed side and trial outcome, with `no` treated as a third class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code marks a trial as right when `(R & hit) | (L & miss)` is true, marks `no` trials as class 2 (`none`), and treats the remaining kept trials as left.

ii. ```python
right = (R[tr0] & hit[tr0]) | (L[tr0] & miss[tr0])
lick = np.where(no[tr0], 2, np.where(right, 1, 0))
```

iii. `CONVERSION_NOTES.md` says this matches the paper’s choice definition while adding the explicit `none` class required by the decoder specification.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from the Bpod `autowater` flag.

ii. ```python
aw = as_vector(bp['autowater'])[:N].astype(bool)
```

iii. The AI’s notes say `autowater` is the WC-context indicator used in the reference code and matches `autowaterBlock` where that field exists.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code directly relabels `autowater` trials as WC and all others as DR, using integer codes 0 and 1 respectively.

ii. ```python
context = np.where(aw[tr0], 0, 1)
```

iii. The notes justify this as a direct mapping from the task structure and the decoder’s requested category order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial `hit`, `miss`, and `no` flags.

ii. ```python
hit = as_vector(bp['hit'])[:N].astype(bool)
miss = as_vector(bp['miss'])[:N].astype(bool)
no = as_vector(bp['no'])[:N].astype(bool)
```

iii. The notes justify keeping `no` as an explicit ignore class because the decoder task asked for `incorrect`, `correct`, and `ignore`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Misses are coded as 0 (`incorrect`), hits as 1 (`correct`), and `no` trials as 2 (`ignore`).

ii. ```python
outcome = np.where(no[tr0], 2, np.where(hit[tr0], 1, 0))
```

iii. The AI explicitly documents this as a required deviation from the paper’s behavior analyses, which often dropped ignore trials instead of keeping them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The implemented code uses only the side-camera tongue track, not both tongue views. It reads side-camera trajectory data via `get_traj_view(obj, 1)`, finds the `'tongue'` feature, and uses frame times together with `video_offset(obj)` and per-trial go-cue times.

ii. ```python
side = get_traj_view(obj, 1)
side_feats = get_feat_names(side[0])
...
tongue_ix = side_feats.index('tongue')
Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
```

iii. The notes say the AI intended to follow video alignment from `findVideoOffset.m` and initially discussed using the reference tongue features; however, the actual code only uses the side-camera tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The side-camera tongue x/y positions are first interpolated onto the neural time axis, then speed is computed as the magnitude of finite differences on those interpolated positions. Visibility is defined by whether interpolated x and y are finite, and isolated visible points with undefined gradients are set to zero speed.

ii. ```python
Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
tongue_speed = speed_from_positions(Xt, Yt)
tongue_visible = np.isfinite(Xt) & np.isfinite(Yt)
tongue_speed = np.where(tongue_visible & ~np.isfinite(tongue_speed), 0.0, tongue_speed)
```

iii. The notes justify interpolating video onto the neural axis and mention a later fix for NaN propagation at lick edges so visible tongue points would not be mislabeled as not visible.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code uses a per-session median of the visible tongue-speed values. Visible points below the median become class 0, visible points at or above the median become class 1, and non-visible points become class 2.

ii. ```python
def discretize(values, visible):
    out = np.full(values.shape, 2, dtype=np.int64)
    vis = visible & np.isfinite(values)
    if np.any(vis):
        thr = np.percentile(values[vis], 50)
        out[vis] = (values[vis] >= thr).astype(np.int64)
...
tongue_cls, tongue_thr = discretize(tongue_speed, tongue_visible)
```

iii. The AI explicitly justified the median split in the notes as required by the decoder task.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by a session-wide video/ephys offset and then by the trial’s go-cue time. After that, the tongue positions are interpolated onto the same `taxis` array used by the neural data.

ii. ```python
vidshift = get_vidshift(obj)
...
tt = ft - vidshift - align_times[i]
...
Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
```

iii. The notes tie this directly to `findVideoOffset.m` and to the requirement that all streams share the go-cue-centered neural time base.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from both bottom-camera paw tracks, `'top_paw'` and `'bottom_paw'`, using bottom-camera trajectory data and frame times.

ii. ```python
bottom = get_traj_view(obj, 2)
bot_feats = get_feat_names(bottom[0])
...
for f in ('top_paw', 'bottom_paw'):
    ix = bot_feats.index(f)
    Xp, Yp = interp_positions(bottom, ix, trials, align_times, taxis, vidshift)
```

iii. The notes say the AI intended to use bottom-camera paw tracking and eventually documented an average over the two tracked paws.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, the code interpolates x/y positions onto `taxis`, computes speed from interpolated positions, averages the two paw speeds with `nanmean`, and marks isolated visible-but-gradientless points as zero speed.

ii. ```python
paw_speeds.append(speed_from_positions(Xp, Yp))
paw_vis.append(np.isfinite(Xp) & np.isfinite(Yp))
...
paw_vis = np.any(np.stack(paw_vis), axis=0)
with np.errstate(invalid='ignore'):
    paw_speed = np.nanmean(np.stack(paw_speeds), axis=0)
paw_speed = np.where(paw_vis & ~np.isfinite(paw_speed), 0.0, paw_speed)
```

iii. The notes justify this as a time-varying movement output built on the same aligned video pipeline as tongue velocity, with a later median split to satisfy the decoder format.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. As with tongue velocity, a per-session median over visible paw-speed values is used. Visible bins are labeled 0/1 by the median split and non-visible bins are labeled 2.

ii. ```python
paw_cls, paw_thr = discretize(paw_speed, paw_vis)
```

iii. The notes explicitly say all three continuous movement outputs were discretized at the per-session 50th percentile because that is what the decoder task required.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw tracks use the same video-offset correction and per-trial go-cue subtraction as tongue tracks, then are interpolated onto the shared neural time axis.

ii. ```python
tt = ft - vidshift - align_times[i]
...
Xp, Yp = interp_positions(bottom, ix, trials, align_times, taxis, vidshift)
```

iii. The notes treat paw and tongue alignment identically: correct the camera clock once per session, then express everything on the neural go-cue grid.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from a separate `motionEnergy_*.mat` file when `sess['mefile']` exists, producing one per-trial 1-D trace.

ii. ```python
me_trials = None
if sess.get('mefile'):
    try:
        me_trials, _thr = load_motion_energy(sess['mefile'])
    except Exception:
        me_trials = None
```

iii. The notes justify using the standalone motion-energy files because they are the reference source and can be normalized across the different layouts present in the release.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion-energy traces are linearly interpolated onto the shared neural time axis using frame times from the side camera, then discretized later by a session median. There is no additional smoothing or differencing in this script.

ii. ```python
ME = np.full((ntr, taxis.size), np.nan)
...
d = np.asarray(me_trials[tr - 1], dtype=float).ravel()
...
ME[i] = np.interp(taxis, tt, d, left=np.nan, right=np.nan)
```

iii. The notes justify this by saying motion energy already exists as a per-frame scalar and only needs alignment / resampling before the decoder-specific median split.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Visible motion-energy samples are split at the per-session 50th percentile into classes 0 and 1, while missing / out-of-range samples remain class 2.

ii. ```python
me_cls, me_thr = discretize(ME, me_visible)
```

iii. The notes explicitly call this a task-driven deviation from the paper’s manual movement threshold: the decoder prompt required a per-session median split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy frame times are corrected by the same video offset and go-cue subtraction used for tracked features, then resampled onto `taxis`, the shared neural time axis.

ii. ```python
tt = ft - vidshift - align_times[i]
...
ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
```

iii. The notes justify this as matching the same offset-correction logic used in the reference `loadMotionEnergy.m` / `findVideoOffset.m` path.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several missing-data cases by fallback or by propagating “not visible” states. Missing or mismatched `frameTimes` trigger a synthetic 400 Hz time base with a default 0.5 s video shift; missing motion-energy files produce all-NaN traces; `np.interp` is followed by explicit NaN-mask propagation; and visible points with undefined gradients are forced to zero speed.

ii. ```python
if ft.size != ts.shape[0] or not np.any(np.isfinite(ft)):
    ft = (np.arange(1, ts.shape[0] + 1)) / VIDEO_FS
    tt = ft - DEFAULT_VIDSHIFT - align_times[i]
...
nanmask = np.interp(taxis, tt, np.isnan(v).astype(float), left=1.0, right=1.0) > 0
arr[i][nanmask] = np.nan
...
if sess.get('mefile'):
    ...
else:
    me_trials = None
```

iii. The notes justify these choices as pragmatic handling of minor data issues while preserving categorical visibility classes and avoiding format failures in downstream verification.

## 11-a. What are the most time-consuming steps of the code?

i. The code explicitly times session loading, spike processing, kinematics, and motion-energy alignment. From the AI’s notes and run logs, the dominant costs are loading each session object from disk and the per-session video/kinematics work.

ii. ```python
t0 = time.time()
timing = {}
obj = load_obj(sess['datafile'])
timing['load'] = time.time() - t0
...
timing['spikes'] = time.time() - t1
...
timing['kinematics'] = time.time() - t1
...
timing['motion_energy'] = time.time() - t1
```

iii. `CONVERSION_NOTES.md` says file reading dominates overall runtime, with kinematics and motion-energy processing the next largest per-session costs.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest remaining explicit loops are over probes, clusters, trials, and paw features. Examples are the per-cluster spike loop in `bin_spikes`, the per-trial interpolation loop in `interp_positions`, and the per-trial motion-energy loop in `motion_energy_on_axis`.

ii. ```python
for p in probes:
    ...
    for j, ci in enumerate(keep):
        ...
for i, tr in enumerate(trials):
    ...
for f in ('top_paw', 'bottom_paw'):
    ...
for i, tr in enumerate(trials):
    ...
```

iii. The notes do not dwell on optimization here, but the structure of the code suggests the AI accepted these loops because the raw video data are ragged across trials and features.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats very similar interpolation / resampling work for several streams: once for tongue x/y, once for each paw track, and again for motion energy. It also computes x and y interpolation separately inside `interp_positions`.

ii. ```python
Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
...
for f in ('top_paw', 'bottom_paw'):
    ...
    Xp, Yp = interp_positions(bottom, ix, trials, align_times, taxis, vidshift)
...
ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
```

iii. There is no explicit justification in the notes beyond staying close to the reference-style feature-by-feature processing and keeping each output stream separate.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads full session objects even though it uses only a subset of fields, computes richer metadata such as per-unit brain-region labels and per-session timing summaries, and optionally generates diagnostic plots. It also processes both paw tracks even though the decoder ultimately uses only one discretized paw-output channel.

ii. ```python
obj = load_obj(sess['datafile'])
...
regions = probe_regions(obj, [p for p, n in units_per_probe])
...
for f in ('top_paw', 'bottom_paw'):
    ...
if show_processing and ntr:
    plot_processing(...)
```

iii. The notes justify most of this as validation and sanity-check machinery: richer metadata, optional plots, and diagnostic processing were used to verify the conversion rather than to feed the final decoder directly.
