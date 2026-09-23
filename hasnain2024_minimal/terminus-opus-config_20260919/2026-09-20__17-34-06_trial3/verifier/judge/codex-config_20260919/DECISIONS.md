# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads only HDF5 `data_structure_*.mat` files from `/app/data/Ephys_Behavior`. It discovers files by glob, but accepts a session only when its animal/date has an uncommented probe entry parsed from the authors' MATLAB metadata scripts. It deliberately excludes the entire randomized-delay folder.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'
files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_structure_*.mat')))
meta = parse_meta_scripts()
probes = meta.get((anm, date))
res = load_session(dsfile, probes)
```

iii. The trajectory says the fixed-delay recordings were treated as the paper's main dataset, while randomized-delay sessions were a separate Figure 8 cohort with a different trial structure and essentially no WC trials. The agent also treated the authors' loading scripts as the authority for included sessions/probes.

## 1-b. How are the data split into subjects?

i. The subject is parsed from each filename. Subjects are added in first-encounter order, and each session stores the corresponding integer index.

ii.
```python
m = re.match(r'data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat', base)
anm, date = m.group(1), m.group(2)
if anm not in subjects:
    subjects.append(anm)
data['subject_idx'].append(subjects.index(anm))
```

iii. The trajectory consistently identifies the recordings by animal/date from filenames and metadata scripts; it reports the final subject count after conversion.

## 1-c. How are the data split into sessions?

i. Each accepted `data_structure_<animal>_<date>.mat` file is one session and becomes one element of `neural`, `input`, and `output`. Sessions with fewer than two selected trials or fewer than ten retained units are omitted.

ii.
```python
if len(trials) < 2:
    return None
if rates.shape[0] < MIN_UNITS:
    return None
data['neural'].append(neural_sess)
```

iii. The agent cites a paper inclusion criterion of at least ten usable units and the target format's requirement of at least two trials per session.

## 1-d. How are the data split into trials?

i. Bpod arrays are treated as per-trial rows. Retained zero-based trial indices select spike, video, motion-energy, and behavioral entries. Each retained trial is emitted as a separate matrix.

ii.
```python
ntrials = int(get('Ntrials')[0])
trials = np.where(keep)[0]
neural_sess = [np.ascontiguousarray(rates[:, i, :]) for i in range(ntr)]
```

iii. The trajectory recognized Bpod as defining trials directly and used cluster `trial` labels and trial-indexed camera fields rather than reconstructing boundaries.

## 1-e. How are trials filtered based on quality controls?

i. It keeps completed hit, miss, or no-response trials, excludes early-lick and photostimulation trials, and requires a finite go cue. It does not apply the reference's last-ephys-trial cutoff.

ii.
```python
keep = ((hit == 1) | (miss == 1) | (no == 1)) & (early != 1) & (stim != 1) & ~np.isnan(gocue)
```

iii. The agent says early licks are omitted by the paper, photoinactivation should not be mixed with unperturbed activity, and ignore trials must remain because ignore is a requested decoder category.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each selected probe's cluster `trialtm` and `trial` arrays, plus cluster `quality` and Bpod `ev/goCue`. Probe-location metadata supplies brain-region labels.

ii.
```python
trialtm_refs = np.array(clu['trialtm']).ravel()
trial_refs = np.array(clu['trial']).ravel()
tt = np.array(f[trialtm_refs[cid]]).ravel()
tr = np.array(f[trial_refs[cid]]).ravel().astype(np.int64) - 1
ta = tt - gocue[tr]
```

iii. The agent states that `trialtm` is on the behavioral trial clock, so subtracting that trial's go cue performs the required alignment.

## 2-b. How is the `neural` data processed?

i. Spikes are counted in 10 ms bins, divided by 0.01 s to obtain Hz, smoothed with a custom 15-bin one-sided Gaussian convolution, and concatenated across selected probes. No normalization or baseline correction is applied.

ii.
```python
counts = np.bincount(idx, minlength=len(trials) * NT).reshape(len(trials), NT)
sess_rates[ui] = counts / DT
sess_rates = smooth_causal(sess_rates)
rates = np.concatenate(rates, axis=0)
```

iii. The trajectory says this follows `getSeq.m`/`mySmooth.m` and interprets scripts as using 10 ms bins and a causal Gaussian window of length 15.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labeled garbage, the typo gabrga, noisy, or real? are removed. Remaining units must have mean processed rate above 1 Hz. Sessions must retain at least ten units.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
cluid = [i for i, q in enumerate(qual) if q not in BAD_QUALITY]
use = sess_rates.mean(axis=(1, 2)) > LOW_FR
```

iii. The agent cites `findClusters.m`, the paper's greater-than-1-Hz criterion, and a paper session-inclusion rule of ten units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time is shifted by its trial's go-cue time, then assigned to the common window.

ii.
```python
ta = tt - gocue[tr]
b = np.floor((ta - TMIN) / DT).astype(np.int64)
```

iii. The agent cites `params.alignEvent = 'goCue'` and notes that the go cue is water presentation on WC trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 500 non-overlapping 10 ms bins from -2.5 to +2.5 seconds. Raw spikes are newly binned at this resolution; video streams are interpolated onto the same bin centers.

ii.
```python
DT = 0.01
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
```

iii. The trajectory explicitly concluded that the relevant scripts use `dt=1/100`, though the human reference found `params.dt=1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from the configured analysis window and bin width rather than read from a raw field; raw go-cue times are used to align spikes to this coordinate system.

ii.
```python
TAXIS = EDGES[:-1] + DT / 2
time_input = TAXIS.astype(np.float32).reshape(1, NT)
```

iii. The agent chose bin centers because the requested input is continuous time from the alignment event.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It computes centers of the 10 ms bins and copies the same 1-by-500 vector to every trial.

ii.
```python
TAXIS = EDGES[:-1] + DT / 2
input_sess = [time_input.copy() for _ in range(ntr)]
```

iii. The trajectory describes the time input as the bin centers of the neural time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the centers of the edges used to bin go-cue-shifted spikes, so columns correspond one-to-one.

ii.
```python
b = np.floor((ta - TMIN) / DT).astype(np.int64)
TAXIS = EDGES[:-1] + DT / 2
```

iii. The agent intended one shared neural/input/output time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `obj.bp.ev.lickL`, `lickR`, and `goCue`, not the instructed side and outcome flags used by the reference.

ii.
```python
lickL = [np.array(f[r]).ravel() for r in np.array(bp['ev/lickL']).ravel()]
lickR = [np.array(f[r]).ravel() for r in np.array(bp['ev/lickR']).ravel()]
```

iii. The agent's docstring says lick direction is the side of the first lick-port contact after the go cue, treating this as a direct behavioral measurement.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It finds the earliest left and right lick strictly after the go cue; the earlier side becomes left (0) or right (1), and no post-cue lick becomes none (2). The value is repeated over all time bins.

ii.
```python
lt = lickL[t][lickL[t] > gc]
rt = lickR[t][lickR[t] > gc]
lick_dir[i] = 0 if fl <= fr else 1
out[0] = res['lick_dir'][i]
```

iii. The agent selected the first post-go-cue port contact as the operational definition of direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived directly from per-trial `obj.bp.autowater`.

ii.
```python
early, autowater = get('early'), get('autowater')
context = (autowater[trials] == 1).astype(np.int64)
```

iii. The agent cites `WorkingWithDataObjs.m` as using autowater as the WC-block proxy.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater false maps to DR code 0 and true maps to WC code 1; the trial value is repeated over time.

ii.
```python
'output_values': [..., ['DR', 'WC'], ...]
out[1] = res['context'][i]
```

iii. This is presented as a direct binary relabeling of the block flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses the Bpod `hit`, `miss`, and `no` arrays.

ii.
```python
hit, miss, no = get('hit'), get('miss'), get('no')
```

iii. The agent kept no-response trials specifically because ignore is requested as an outcome class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The default/miss code is incorrect (0), hit is correct (1), and `no` is ignore (2), repeated across time.

ii.
```python
outcome = np.full(len(trials), 0, dtype=np.int64)
outcome[hit[trials] == 1] = 1
outcome[no[trials] == 1] = 2
```

iii. These codes follow the requested category order.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-camera `obj.traj` data: feature name `tongue`, per-trial `frameTimes`, and x/y rows of `ts`; it also uses bitcode timing and go cue for alignment. It does not use the bottom-camera `top_tongue` used by the reference.

ii.
```python
tongue[i] = feat_speed(0, ['tongue'], t)
ft = np.array(f[np.array(g['frameTimes']).ravel()[t]]).ravel()
ts = np.array(f[np.array(g['ts']).ravel()[t]])
```

iii. The agent's trajectory identifies the side-camera tongue as the chosen feature and regarded NaN coordinates as visibility information.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X and y are linearly interpolated to neural bin centers. `np.gradient` is applied with no time-spacing argument, and Euclidean magnitude is computed. There is no position smoothing, likelihood cutoff in this function, view normalization, or two-camera combination.

ii.
```python
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
vx = np.gradient(x); vy = np.gradient(y)
sp.append(np.sqrt(vx ** 2 + vy ** 2))
```

iii. The agent believed this matched the paper's position/velocity functions after clock correction and interpolation; it expected existing NaNs to encode untracked tongue frames.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One median is computed from every finite tongue value in the session. Values below it are 0, values at/above it are 1, and NaNs are 2 (not visible).

ii.
```python
thresh = np.percentile(x[finite], 50)
out[finite & (x < thresh)] = 0
out[finite & (x >= thresh)] = 1
```

iii. The median split and explicit missing category follow the decoder instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video-clock offset is computed from SpikeGLX and Bpod bit starts. Frame times are shifted by that offset and the trial go cue, then positions are interpolated directly at `TAXIS`.

ii.
```python
vidshift = video_offset(f)
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
```

iii. The agent cites `findVideoOffset.m` and intended exact correspondence with neural bin centers.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses the bottom-camera `top_paw` and `bottom_paw` x/y trajectories and frame times, plus bitcode/go-cue timing.

ii.
```python
paw[i] = feat_speed(1, ['top_paw', 'bottom_paw'], t)
```

iii. The trajectory says both paw features occur in the paper's default trajectory-feature list and therefore were averaged.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw's x/y is interpolated to the neural grid, coordinate gradients are combined into speed, then both paw speeds are NaN-averaged. As for tongue, the gradient is per bin rather than per second and there is no coordinate smoothing.

ii.
```python
sp.append(np.sqrt(vx ** 2 + vy ** 2))
out = np.nanmean(sp, axis=0) if sp.shape[0] > 1 else sp[0]
```

iii. The agent believed averaging both tracked paws followed the paper's configured features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It applies the session-wide finite-value median: below 0, at/above 1, NaN/not visible 2.

ii.
```python
paw = discretize(res['paw'])
```

iii. This directly implements the requested percentile split and missing category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera times are corrected with the session offset and trial go cue, and trajectories are interpolated at the neural bin centers.

ii.
```python
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
```

iii. The same shared-clock rationale as tongue is used.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads per-trial traces from the adjacent `motionEnergy_*.mat` file, including a nested `me.data.data` variant, and uses side-camera frame times, clock bit starts, and go cues.

ii.
```python
mefile = dsfile.replace('data_structure', 'motionEnergy')
me = sio.loadmat(mefile)['me']
data = me['data'][0, 0]
if data.dtype.names is not None and 'data' in data.dtype.names:
    data = data['data'][0, 0]
```

iii. The trajectory found multiple storage layouts and patched the nested layout to match `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already-computed motion-energy trace is linearly interpolated onto `TAXIS`. Any interpolation NaNs are then nearest-filled, including edge gaps. No further smoothing is performed.

ii.
```python
out[i] = interp_nan(TAXIS, tt, d)
out[i] = np.where(np.isnan(v), v[choose], v)
```

iii. The agent cites `loadMotionEnergy.m` for interpolation and nearest filling and treats the file values as already reduced motion energy.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session-wide median of finite aligned values splits class 0 from class 1; missing/no-video values are class 2.

ii.
```python
me = discretize(res['me'])
```

iii. This follows the decoder's 50th-percentile specification and preserves a no-video category.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are shifted by the video offset and trial go cue, then motion energy is interpolated to neural bin centers.

ii.
```python
tt = ft - vidshift - gocue[t]
out[i] = interp_nan(TAXIS, tt, d)
```

iii. The agent used the side camera because motion-energy samples correspond to those frames.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video/features remain NaN until discretization, when they become class 2. Missing motion-energy files, malformed arrays, length mismatches, and all-NaN frame times likewise yield class 2. Motion-energy interpolation gaps are nearest-filled when any valid values exist. Sessions without metadata are skipped; probe locations can default to ALM. The code only supports HDF5 session files.

ii.
```python
if not os.path.exists(mefile):
    return np.full((len(trials), NT), np.nan)
out = np.full(x.shape, 2, dtype=np.int64)
```

iii. The agent emphasized explicit not-visible/no-video categories because decoder outputs cannot remain NaN. It added guards after inspecting nested motion-energy files and probe-location variants.

## 11-a. What are the most time-consuming steps of the code?

i. The agent did not benchmark or explicitly document runtime hotspots. By inspection, HDF5 dereferencing, per-unit spike loading/binning, per-trial trajectory interpolation, and per-row convolution dominate.

ii.
```python
for ui, cid in enumerate(cluid): ...
for i in range(flat.shape[0]):
    o[i] = np.convolve(flat[i], KERN, mode='same')
for i, t in enumerate(trials): ...
```

iii. The trajectory focused on end-to-end completion and decoder validation, not profiling; no specific justification was given.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Smoothing loops over every unit-trial row and could use an axis-aware filter/convolution. Output construction and some per-trial work could be stacked/vectorized, although ragged HDF5 trajectory and spike arrays make the unit/trial loading loops harder to remove.

ii.
```python
for i in range(flat.shape[0]):
    o[i] = np.convolve(flat[i], KERN, mode='same')
for i in range(ntr):
    out = np.empty((6, NT), dtype=np.int64)
```

iii. The agent gave no vectorization analysis in the trajectory.

## 11-c. What processing does the code repeat multiple times?

i. `video_offset(f)` is computed once in `video_speeds` and again in `motion_energy`. The common trajectory interpolation/gradient logic is reused through `feat_speed`, while identical input vectors are copied for every trial and per-trial constants are repeatedly expanded across time.

ii.
```python
vidshift = video_offset(f)  # in video_speeds
vidshift = video_offset(f)  # again in motion_energy
input_sess = [time_input.copy() for _ in range(ntr)]
```

iii. The agent did not discuss repeated processing; its rationale centered on correctness and shared alignment.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads unused `R` and `L` fields. It parses/loads some metadata and arrays only for filtering or logging, and copies identical time inputs. Most computed values do enter the output, but continuous velocity/motion traces are discarded after categorization.

ii.
```python
R, L = get('R'), get('L')
tongue = discretize(res['tongue'])
paw = discretize(res['paw'])
me = discretize(res['me'])
```

iii. The trajectory does not identify unnecessary work; it validates the final categorical dataset and decoder performance.
