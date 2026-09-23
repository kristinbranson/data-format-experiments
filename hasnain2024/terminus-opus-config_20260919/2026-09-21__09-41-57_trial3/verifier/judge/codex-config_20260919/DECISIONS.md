# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter obtains the authoritative ephys-session/probe list by parsing the authors’ `load<ANM>_ALMVideo.m` metadata scripts. It processes the 44 listed sessions, loading mixed MATLAB v7.3/v5 objects through `matio.load_obj` and separate motion-energy files through `load_motion_energy`. Sessions are parallelized with a process pool.

ii.
```python
sessions = get_sessions()
with Pool(nw) as pool:
    results = pool.map(_worker, jobs)
```
```python
obj = load_obj(sess['datafile'])
me_trials, _thr = load_motion_energy(sess['mefile'])
```

iii. The notes say the authors’ metadata scripts identify usable sessions and ALM probes, while the custom loader is needed because 36 files are v7.3 and 11 are v5. Parsing rather than globbing excludes unrelated behavior-only and non-reference sessions.

## 1-b. How are the data split into subjects?

i. Each metadata entry supplies `anm`; unique animals are accumulated in encounter order, and each retained session gets an integer `subject_idx`. The full conversion has 14 subjects.

ii.
```python
anm = sess['anm']
if anm not in data['subjects']:
    data['subjects'].append(anm)
data['subject_idx'].append(data['subjects'].index(anm))
```

iii. The agent treated the animal ID in the author metadata/filename as more reliable than inconsistently present object metadata.

## 1-c. How are the data split into sessions?

i. Each `<animal>_<date>` data file selected by the metadata scripts is one session and becomes one element of `neural`, `input`, and `output`. Sessions with fewer than 10 curated units or two usable trials are skipped; all 44 selected sessions passed.

ii.
```python
for sess, res in zip(sessions, results):
    if info['nunits'] < PARAMS['min_units']:
        continue
    if info['ntrials_used'] < 2:
        continue
    data['neural'].append(res['neural'])
```

iii. The notes cite the paper’s requirement of at least 10 units per session and the reference metadata scripts as the session authority.

## 1-d. How are the data split into trials?

i. Trials are the first `bp.Ntrials` entries of the Bpod vectors. Kept zero-based rows are converted to the source’s 1-based trial numbers and used to index spike, behavior, tracking, and motion-energy data.

ii.
```python
N = int(as_vector(bp['Ntrials'])[0])
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
trials = np.where(keep)[0] + 1
```

iii. The notes state that Bpod directly defines trials and that the various streams already carry corresponding trial indices.

## 1-e. How are trials filtered based on quality controls?

i. Early-lick, stimulation, non-finite-go-cue, and invalid-outcome trials are excluded. The agent then drops every trial with zero spikes across all quality-retained units anywhere in the trial, intending to remove trailing behavior after ephys stopped.

ii.
```python
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
recorded = spk_per_trial > 0
trials = trials[recorded]
```

iii. Early/stim removal follows the paper. The zero-spike rule was added after finding trailing trials in two JEB24 sessions; the agent reasoned that they contained no neural information.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from selected probes in `obj.clu`: each cluster’s `trial`, `trialtm`, and `quality`, together with `bp.ev.goCue` for alignment and probe-location metadata for region labels.

ii.
```python
clusters = get_clusters(obj, p)
tr = np.asarray(c['trial'], dtype=np.int64).ravel()
tm = np.asarray(c['trialtm'], dtype=float).ravel()
aligned = tm[sel] - align_times[pos]
```

iii. The agent identified this as the reference `alignSpikes.m` pathway and selected probes from the authors’ loading scripts.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins from −2.5 to +2.5 s, divided by bin width to produce Hz, and smoothed with a 15-bin causal Gaussian kernel using reflect-style prefixing. No z-scoring or baseline subtraction is applied.

ii.
```python
np.add.at(counts[:, :, j], (pos[good], b[good]), 1.0)
rates = counts / PARAMS['dt']
sm = my_smooth(flat)
```
```python
k = _gausswin(N)
k[:N // 2] = 0.0
return k / k.sum()
```

iii. The agent chose parameters from `WorkingWithDataObjs.m` and described this as a direct port of `getSeq.m` and `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Labels are stripped/lowercased and clusters labeled `garbage`, `gabrga`, `noisy`, `real?`, or empty are removed. Units with mean retained-window firing rate at or below 1 Hz are then removed. Sessions need at least 10 remaining units.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}
keep_u = meanfr > PARAMS['lowFR']
```

iii. The notes cite `findClusters.m`, the methods’ “exceeding 1 Hz” rule, and the paper’s minimum-10-unit session criterion. Lowercasing was considered safer for inconsistent capitalization.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each within-trial spike time is shifted by that trial’s `bp.ev.goCue`, then assigned to the common relative-time bins.

ii.
```python
align_times = gocue[trials - 1]
aligned = tm[sel] - align_times[pos]
```

iii. This was justified as the direct equivalent of the authors’ `alignSpikes.m`; WC `goCue` denotes water-drop time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses 10 ms bins (500 bins over five seconds). It does not subsequently rebin; it smooths across 15 bins.

ii.
```python
PARAMS = dict(tmin=-2.5, tmax=2.5, dt=0.01, smooth=15)
```

iii. The agent selected the 10 ms setting in the tutorial’s `WorkingWithDataObjs.m`, noting that another default used 5 ms, and considered 10 ms feasible for memory and decoding.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the chosen alignment window and bin centers, with `bp.ev.goCue` defining time zero rather than contributing a different per-trial value.

ii.
```python
edges = np.arange(PARAMS['tmin'], PARAMS['tmax'] + 1e-12, PARAMS['dt'])
tm = edges[:-1] + PARAMS['dt'] / 2
```

iii. The notes identify the bin-center vector produced by `getSeq.m` as the required decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs equally spaced bin edges and adds half a bin to obtain centers from −2.495 to +2.495 s, cast to float32 and copied for every trial.

ii.
```python
tin = taxis.astype(np.float32)[None, :]
inp.append(tin.copy())
```

iii. The agent says this reproduces the reference time-axis construction.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the center of exactly the edges used to bin go-cue-shifted spikes, so input column and neural column indices coincide.

ii.
```python
b = np.floor((aligned - edges[0]) / PARAMS['dt']).astype(np.int64)
```

iii. The shared grid was explicitly used as an alignment sanity check.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from Bpod `R`, `L`, `hit`, `miss`, and `no` flags.

ii.
```python
right = (R[tr0] & hit[tr0]) | (L[tr0] & miss[tr0])
```

iii. Because actual chosen direction is not a single stored field, the notes infer it from instructed side and correctness and checked it against post-go-cue lick contacts.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct trials retain the instructed side, misses reverse it, and `no` trials become class 2; codes are left 0, right 1, none 2 and are repeated across time.

ii.
```python
lick = np.where(no[tr0], 2, np.where(right, 1, 0))
o[0] = lick[i]
```

iii. The agent states this matches the reference choice definition while preserving the prompt-required none category.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from `bp.autowater`.

ii.
```python
aw = as_vector(bp['autowater'])[:N].astype(bool)
```

iii. The notes verified this agrees with `autowaterBlock` where available and with reference trial-condition strings.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is mapped to WC 0 and its complement to DR 1, constant across the trial.

ii.
```python
context = np.where(aw[tr0], 0, 1)
o[1] = context[i]
```

iii. This is a direct categorical relabeling chosen to match the requested value order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses Bpod `hit`, `miss`, and `no` flags.

ii.
```python
hit = as_vector(bp['hit'])[:N].astype(bool)
miss = as_vector(bp['miss'])[:N].astype(bool)
no = as_vector(bp['no'])[:N].astype(bool)
```

iii. The agent notes these flags are mutually exclusive and directly encode the requested outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss is incorrect 0, hit is correct 1, and no response is ignore 2; the value is repeated over time.

ii.
```python
outcome = np.where(no[tr0], 2, np.where(hit[tr0], 1, 0))
o[2] = outcome[i]
```

iii. Ignore trials were retained because the decoder specification explicitly requires that category even though paper analyses often omit them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses x/y coordinates for only the side-camera `tongue` feature in `obj.traj{1}`, its `frameTimes`, and the video/behavior clock offset fields. It does not use the bottom-camera `top_tongue` feature.

ii.
```python
side = get_traj_view(obj, 1)
tongue_ix = side_feats.index('tongue')
Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
```

iii. The agent said this followed `findPosition.m`/`findVelocity.m` and treated DLC NaNs as invisibility. Its notes did not justify omitting the second tongue view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-camera x/y are linearly interpolated directly to the 10 ms neural centers, with NaN masks propagated. Central/one-sided coordinate differences are taken and combined as Euclidean speed. Undefined speed at an otherwise visible isolated point becomes zero. There is no coordinate smoothing and no division by elapsed seconds.

ii.
```python
tongue_speed = speed_from_positions(Xt, Yt)
tongue_speed = np.where(tongue_visible & ~np.isfinite(tongue_speed), 0.0, tongue_speed)
```
```python
return np.sqrt(vx ** 2 + vy ** 2)
```

iii. The notes describe this as matching the reference’s gradient operation and explain the one-sided fallback as preserving velocity wherever DLC marked the tongue visible.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A median is computed over all finite, visible tongue-speed samples in the session. Visible values below it are 0, values at/above it are 1, and invisible values are 2.

ii.
```python
thr = np.percentile(values[vis], 50)
out[vis] = (values[vis] >= thr).astype(np.int64)
```

iii. This follows the prompt’s per-session 50th-percentile requirement and keeps invisibility outside the threshold calculation.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame time is corrected by session `vidshift` and trial go cue, then positions are interpolated onto neural bin centers. If frame times are invalid, the code invents a 400 Hz frame grid and uses a fixed 0.5 s shift.

ii.
```python
tt = ft - vidshift - align_times[i]
arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
```

iii. The offset calculation is from `findVideoOffset.m`; the fixed-grid fallback was believed to be a reference fallback.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera x/y tracks for both `top_paw` and `bottom_paw`, plus their frame times and clock-offset fields.

ii.
```python
for f in ('top_paw', 'bottom_paw'):
    Xp, Yp = interp_positions(bottom, ix, trials, align_times, taxis, vidshift)
```

iii. The agent noted both paws are bottom-camera features and chose to average them; it did not document the reference concern that `bottom_paw` is unreliable during the delay.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw is interpolated to the neural grid and differentiated like the tongue. The two speeds are averaged where available; a point is visible if either paw is visible, and isolated undefined visible speeds are set to zero.

ii.
```python
paw_vis = np.any(np.stack(paw_vis), axis=0)
paw_speed = np.nanmean(np.stack(paw_speeds), axis=0)
```

iii. The notes characterize this as the reference kinematic pipeline and use both tracked paws to define the requested general “paw velocity.”

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session median of finite samples where either paw is visible splits classes 0 and 1; neither visible is class 2.

ii.
```python
paw_cls, paw_thr = discretize(paw_speed, paw_vis)
```

iii. This directly implements the requested per-session percentile and not-visible class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are shifted by video offset and go cue, and both paw coordinates are interpolated onto the neural centers, with the same fixed-grid fallback for missing times.

ii.
```python
tt = ft - vidshift - align_times[i]
arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
```

iii. The agent used a common corrected time axis for all streams and validated movement changes around the go cue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses each session’s separate `motionEnergy_<animal>_<date>.mat` trace and side-camera frame times, with `obj.me` only noted as an equivalent copy.

ii.
```python
me_trials, _thr = load_motion_energy(sess['mefile'])
ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
```

iii. Separate files exist for the selected sessions and robust loading was needed for three wrapper layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already spatially reduced per-frame trace is linearly interpolated to neural centers. It is not smoothed or differentiated.

ii.
```python
ME[i] = np.interp(taxis, tt, d, left=np.nan, right=np.nan)
```

iii. The agent reasoned that the motion-energy computation had already occurred upstream and should not be repeated.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median of finite motion-energy samples across the session splits low 0 and high 1; unavailable samples are class 2.

ii.
```python
me_cls, me_thr = discretize(ME, me_visible)
```

iii. The agent deliberately followed the decoder prompt’s 50th percentile rather than the source file’s manually chosen `moveThresh`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are offset-corrected and go-cue shifted, then motion energy is interpolated onto the same centers. Missing/mismatched frame times invoke a synthetic 400 Hz grid and 0.5 s shift.

ii.
```python
tt = ft - vidshift - align_times[i]
ME[i] = np.interp(taxis, tt, d, left=np.nan, right=np.nan)
```

iii. The notes cite `loadMotionEnergy.m` and the shared video-offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite go cues and zero-spike trials are dropped. Missing video coordinates remain NaN and become class 2. Invalid frame times or video offsets are replaced by a synthetic 400 Hz time base and/or 0.5 s offset. Missing motion-energy files/trials remain class 2. Loader exceptions return `None`, causing a whole session to be skipped.

ii.
```python
if ft.size != ts.shape[0] or not np.any(np.isfinite(ft)):
    ft = np.arange(1, ts.shape[0] + 1) / VIDEO_FS
    tt = ft - DEFAULT_VIDSHIFT - align_times[i]
```
```python
except Exception:
    return None
```

iii. The agent sought to preserve trials when only video was missing, using the explicit not-visible class. It viewed the 400 Hz/0.5 s fallback as reference behavior and dropped no-neural-information trials.

## 11-a. What are the most time-consuming steps of the code?

i. Loading large MATLAB objects, kinematic interpolation, and spike processing are timed per session. The agent identified file loading—especially waveforms—as dominant and used multiprocessing; full conversion took about 18 s with workers.

ii.
```python
timing['load'] = time.time() - t0
timing['spikes'] = time.time() - t1
timing['kinematics'] = time.time() - t1
```

iii. The notes’ profiling estimated 3–8 s per session serially and said loading dominates, motivating eight-worker processing.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-session work is parallelized, and spike accumulation/smoothing is vectorized across trials. Python loops remain over probes, units, trials during video interpolation/motion energy, output assembly, and smoothing columns.

ii.
```python
for j, ci in enumerate(keep):
    np.add.at(counts[:, :, j], (pos[good], b[good]), 1.0)
for i, tr in enumerate(trials):
    arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
```

iii. Ragged camera traces make trial loops awkward to vectorize. The agent explicitly replaced a slower trial-by-trial spike histogram and considered loading the true bottleneck.

## 11-c. What processing does the code repeat multiple times?

i. `interp_positions` separately repeats frame-time correction and x/y interpolation for tongue, top paw, and bottom paw. It also repeats NaN-mask interpolation for each coordinate. Output and input arrays are copied per trial; motion energy repeats similar time-axis construction.

ii.
```python
for k, arr in ((0, X), (1, Y)):
    arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
    nanmask = np.interp(taxis, tt, np.isnan(v).astype(float), ...)
```

iii. The agent cached the session offset and shared bin grid, but accepted repeated feature interpolation because each feature has distinct missingness.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads much of each object although only selected fields are used, computes/stores quality strings and detailed timing/diagnostic metadata, reads the motion-energy stored threshold but discards it, and computes plots only when requested. It deliberately skips large `spkWavs` fields.

ii.
```python
me_trials, _thr = load_motion_energy(sess['mefile'])
rates, quals, spk_per_trial, units_per_probe = bin_spikes(...)
```

iii. The notes identify waveform loading as unnecessary and optimize it away. Remaining diagnostics support validation even though the decoder does not consume them.
