# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed `data_structure_*.mat` only in `/app/data/Ephys_Behavior`, retained files present in a hard-coded 25-session `PROBE_MAP`, loaded session data with `h5py`, and loaded companion motion-energy files with `scipy.io.loadmat`. It did not load `RandomizedDelay_Ephys_Behavior` or MATLAB v5 session files.

ii.
```python
DATA_DIR = Path('/app/data/Ephys_Behavior')
data_files = sorted(DATA_DIR.glob('data_structure_*.mat'))
if key not in PROBE_MAP:
    continue
result = load_session(str(data_file), me_path, PROBE_MAP[key])
```

iii. The trajectory says the agent chose “all 25 Ephys_Behavior sessions” and believed these were the paper's relevant ALM sessions. It noticed but did not resolve the discrepancy with the paper/session counts.

## 1-b. How are the data split into subjects?

i. A session's animal is read from `obj.meta.anm`, with the filename prefix as fallback. Unique animal strings are sorted and each session gets an integer index.

ii.
```python
try:
    anm = h5_read_string(f, f['obj']['meta']['anm'])
except:
    anm = os.path.basename(data_path).replace('data_structure_', '').split('_')[0]
unique_subjects = sorted(set(session_animals))
subject_idx = np.array([unique_subjects.index(a) for a in session_animals])
```

iii. The agent treated the animal identifier as the subject and reported 10 subjects in the selected sessions.

## 1-c. How are the data split into sessions?

i. Each selected `<animal>_<date>` file is one session and one outer-list element. Only 25 sessions in `Ephys_Behavior` are eligible.

ii.
```python
for data_file in data_files:
    basename = data_file.stem.replace('data_structure_', '')
    anm, date = basename.split('_', 1)
    all_sessions_neural.append(result['neural'])
```

iii. The hard-coded mapping was derived from the authors' loading scripts, but the agent explicitly limited the scope to one folder.

## 1-d. How are the data split into trials?

i. Trials are indices `0..Ntrials-1`. Per-trial behavioral vectors, cluster trial numbers, trajectory cells, and motion-energy cells are indexed with that trial number; selected trials become inner-list elements.

ii.
```python
ntrials = int(bp['Ntrials'][0, 0])
valid_trials = np.where(valid_mask)[0]
for tr in valid_trials:
    session_neural.append(trialdat[:, tr, :].astype(np.float32))
```

iii. The agent relied on the explicit Bpod trial table and the clusters' 1-based `trial` assignments, converting the latter to zero-based indices where needed.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be hit, miss, or no-response, must not have stimulation enabled, and must not be early-lick trials. Sessions with fewer than two selected trials are skipped. It does not remove behavioral trials occurring after ephys recording ended.

ii.
```python
valid_mask = (hit | miss | no) & ~stim_enable & ~early
valid_trials = np.where(valid_mask)[0]
if len(valid_trials) < 2:
    return None
```

iii. The trajectory says stimulation and early-lick exclusion was intended to match the reference conditions; the outcome-union retained hit, miss, and ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ALM probes in `obj.clu`: each cluster's `trialtm`, `trial`, and `quality`, plus `bp.ev.goCue` for alignment.

ii.
```python
trialtm = f[trialtm_ref][()].flatten()
trial_nums = f[trial_ref][()].flatten().astype(int)
aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```

iii. The agent identified the spike-sorted cluster fields and go cue as the required raw variables, following `alignSpikes.m`/`findClusters.m`.

## 2-b. How is the `neural` data processed?

i. Aligned spikes are histogrammed per unit and trial, divided by 10 ms to Hz, and convolved with a custom 15-bin one-sided (“causal”) Gaussian after prepending samples for reflection. The resulting arrays are stored as `float32`.

ii.
```python
counts, _ = np.histogram(spike_times, bins=edges)
fr = counts.astype(float) / dt
fr = smooth_data(fr, smooth_win, bctype)
kern[:int(np.floor(len(kern) / 2))] = 0
```

iii. The agent believed this matched MATLAB `mySmooth.m`, describing it as a causal Gaussian with window 15 and reflect boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It removes quality labels `garbage`, `gabrga`, `noisy`, and `real?`, also removes blank labels, then retains units whose mean rate across all trials and bins exceeds 1 Hz. It skips sessions with fewer than ten units before or after rate filtering.

ii.
```python
if quality.lower() in {q.lower() for q in EXCLUDE_QUALITIES}:
    continue
if quality == '':
    continue
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=1)
fr_mask = mean_fr > LOW_FR
```

iii. The trajectory cites `findClusters.m` and the paper's >1 Hz criterion, and adds a decoder-oriented minimum of ten units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time before binning.

ii.
```python
tr = trial_nums[t_idx] - 1
aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```

iii. The agent states this follows the reference's go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It uses 500 non-overlapping 10 ms bins from -2.5 to +2.5 s. Spikes are histogrammed directly to this grid; video streams are linearly interpolated to its bin centers.

ii.
```python
DT = 1 / 100
edges = np.arange(TMIN, TMAX + DT, DT)
time_vec = edges[:-1] + DT / 2
```

iii. The agent explicitly chose the tutorial's 10 ms convention, despite noting a 5 ms default elsewhere.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not computed from a varying raw field; it is the fixed bin grid defined by `TMIN`, `TMAX`, and `DT`, whose zero is defined by `bp.ev.goCue` alignment.

ii.
```python
time_vec = edges[:-1] + DT / 2
neural_time = time_vec
```

iii. The agent used the common aligned time axis as the requested continuous decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It takes each 10 ms bin's center, reshapes the vector to `(1, 500)`, converts it to `float32`, and repeats it for every trial.

ii.
```python
input_trial = neural_time.reshape(1, -1).astype(np.float32)
session_input.append(input_trial)
```

iii. The agent intended the values to express seconds relative to go-cue onset directly.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the centers of the same edges used to histogram neural spikes, so input column `j` corresponds to neural bin `j`.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_vec = edges[:-1] + DT / 2
trialdat[u_idx, tr, :] = bin_and_smooth_spikes(..., edges, ...)
```

iii. The agent deliberately reused one `neural_time` vector for alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses per-trial `bp.hit`, `bp.miss`, `bp.no`, `bp.R`, and `bp.L`.

ii.
```python
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
R = bp['R'][0, :].astype(bool)
L = bp['L'][0, :].astype(bool)
```

iii. The agent reasoned that instructed side plus correctness determines actual lick side, while `no` means no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Right-hit and left-miss map to right (1); left-hit and right-miss map to left (0); no-response maps to none (2). The scalar class is broadcast over time.

ii.
```python
lick_direction[(R & hit) | (L & miss)] = 1
lick_direction[(L & hit) | (R & miss)] = 0
lick_direction[no] = 2
output_trial[0, :] = lick_direction[tr]
```

iii. This implements the agent's stated interpretation of correct, incorrect, and ignored trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `bp.autowater`.

ii.
```python
autowater = bp['autowater'][0, :].astype(bool)
```

iii. The agent interpreted autowater trials as WC and other trials as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is relabeled WC (0), otherwise DR (1), then broadcast across the trial.

ii.
```python
context = np.where(autowater, 0, 1).astype(int)
output_trial[1, :] = context[tr]
```

iii. The trajectory cross-checked WC fractions and concluded this mapping was plausible.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
no = bp['no'][0, :].astype(bool)
```

iii. The three mutually exclusive behavioral flags directly represent the requested outcomes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss maps to incorrect (0), hit to correct (1), and no-response to ignore (2), broadcast across time.

ii.
```python
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
output_trial[2, :] = outcome[tr]
```

iii. The agent used the category order specified by the task.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera `tongue` feature's x/y coordinates and frame times from `obj.traj`; it does not use likelihood explicitly or the bottom-camera `top_tongue`. Go-cue times provide trial alignment.

ii.
```python
side_feats = read_feat_names(f, side_data)
tongue_idx = side_feats.index('tongue')
ft = f[side_data['frameTimes'][tr, 0]][()].flatten()
ts = f[side_data['ts'][tr, 0]][()]
tongue_x, tongue_y = ts[tongue_idx, 0, :], ts[tongue_idx, 1, :]
```

iii. The agent believed NaN coordinates already encoded tongue visibility and said it was keeping tongue NaNs in accordance with the methods.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Missing coordinates are nearest-filled temporarily, x/y gradients are computed assuming exactly 400 Hz, their Euclidean magnitude is retained only at originally visible frames, and speed plus a visibility mask are linearly interpolated to neural bin centers. No positional smoothing, likelihood cutoff, per-view scaling, second view, or within-bin averaging is used.

ii.
```python
tx_filled = nearest_fill(tongue_x)
ty_filled = nearest_fill(tongue_y)
speed_raw = compute_speed(tx_filled, ty_filled, dt_video)
speed[visible] = speed_raw[visible]
speed_interp = np.interp(neural_time, ft, np.where(np.isfinite(speed), speed, 0.0))
```

iii. The agent justified nearest filling as a way to calculate gradients and then mask missing observations back out.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over finite interpolated values from selected trials. Visible values below it are 0, values at or above it are 1, and missing/invisible bins are 2.

ii.
```python
tongue_thresh = np.median(all_tongue_speeds) if all_tongue_speeds else 0
tv[vis & (speed_interp < tongue_thresh)] = 0
tv[vis & (speed_interp >= tongue_thresh)] = 1
```

iii. This follows the prompt's per-session 50th-percentile split and third visibility class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code assumes a fixed 0.5 s video lead, calculates `frameTimes - 0.5 - goCue[trial]`, then linearly interpolates to neural bin centers.

ii.
```python
PAD_SEC = 0.5
aligned_ft = ft - PAD_SEC - goCue[tr]
speed_interp = np.interp(neural_time, ft, speed_for_interp, left=np.nan, right=np.nan)
```

iii. The agent stated that the 0.5 s offset matched the tutorial/reference pipeline.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `top_paw` x/y coordinates and frame times from `obj.traj`, plus go-cue times.

ii.
```python
paw_idx = bot_feats.index('top_paw')
paw_x = ts[paw_idx, 0, :]
paw_y = ts[paw_idx, 1, :]
```

iii. The agent selected `top_paw` as the relevant paw feature from the bottom view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Coordinates are nearest-filled, differentiated at an assumed 400 Hz, combined as speed, then linearly interpolated to neural centers and masked using interpolated original visibility. It does not smooth positions or average frames within bins.

ii.
```python
px = nearest_fill(paw_x); py = nearest_fill(paw_y)
speed_paw = compute_speed(px, py, dt_video)
speed_interp = np.interp(neural_time, ft, speed, left=np.nan, right=np.nan)
```

iii. The agent cited the paper's nearest-fill treatment for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session median of finite interpolated paw speeds is used; below is 0, at/above is 1, invisible is 2.

ii.
```python
paw_thresh = np.median(all_paw_speeds) if all_paw_speeds else 0
pv[vis & (speed_interp < paw_thresh)] = 0
pv[vis & (speed_interp >= paw_thresh)] = 1
```

iii. This was chosen to implement the requested per-session 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are shifted by the same fixed 0.5 s and trial go cue, then interpolated to neural bin centers.

ii.
```python
aligned_ft = ft - PAD_SEC - goCue[tr]
speed_interp = np.interp(neural_time, ft, speed, left=np.nan, right=np.nan)
```

iii. The agent applied the same assumed tutorial offset to all video features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads the companion `motionEnergy_<animal>_<date>.mat` file's `me.data` trial traces; `moveThresh` is read but unused. Side-camera frame times are used as timestamps.

ii.
```python
me_struct = scipy.io.loadmat(me_path)['me']
me_trial_data = me_struct['data'][0, 0]
me_thresh = float(me_struct['moveThresh'][0, 0].flatten()[0])
```

iii. The agent recognized motion energy as precomputed and associated it with side-camera frames.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. If trace and frame-time lengths match, raw motion energy is linearly interpolated to neural bin centers. No additional smoothing or spatial processing is performed.

ii.
```python
if len(me_trial) == len(ft):
    me_interp = np.interp(neural_time, ft, me_trial, left=np.nan, right=np.nan)
```

iii. The agent treated the stored trace as already processed and only resampled it.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median over all finite interpolated values in the session is used; below is 0, at/above is 1, and missing/no-video is 2. The file's `moveThresh` is discarded.

ii.
```python
me_thresh_50 = np.median(all_me_values) if all_me_values else 0
me_disc[vis & (me_interp < me_thresh_50)] = 0
me_disc[vis & (me_interp >= me_thresh_50)] = 1
```

iii. The prompt requires a per-session 50th-percentile threshold, so the agent correctly did not use the stored movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It reuses side-camera times shifted by fixed 0.5 s and the go cue, requires equal trace/time lengths, and interpolates onto the neural centers.

ii.
```python
side_frame_times[tr] = ft - PAD_SEC - goCue[tr]
me_interp = np.interp(neural_time, ft, me_trial, left=np.nan, right=np.nan)
```

iii. The agent assumed motion energy has one value per side-camera frame and shares the same video-clock correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Video loading and individual trial extraction are wrapped in broad exceptions. Missing trajectory/energy becomes class 2; coordinate gaps are nearest-filled for differentiation and then visibility-masked; out-of-range interpolation becomes missing. Missing animal metadata falls back to the filename. Sessions can be skipped for too few trials/units. The code does not handle v5 session files, nested motion-energy layouts, NaN frame times robustly, or trials after ephys ends.

ii.
```python
except Exception:
    tongue_speed_trials[tr] = None
...
output_trial[3, :] = 2
...
except:
    anm = ...split('_')[0]
```

iii. The agent intended class 2 to preserve trials when video was unavailable and used nearest filling according to its reading of the methods.

## 11-a. What are the most time-consuming steps of the code?

i. The likely dominant computation is nested unit-by-trial spike selection, histogramming, and convolution, followed by trialwise HDF5 dereferencing and video interpolation. The agent did not benchmark individual stages.

ii.
```python
for u_idx in range(n_units):
    for tr in range(ntrials):
        tr_mask = strials == (tr + 1)
        trialdat[u_idx, tr, :] = bin_and_smooth_spikes(...)
```

iii. The trajectory reports successful runtime/testing but gives no timing-based justification for bottlenecks.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-spike go-cue subtraction can be advanced-indexed; unit-by-trial masks/histograms can use a 2-D histogram; nearest filling can use indexed methods; and repeated interpolation/threshold collection could be array-oriented where trial lengths permit.

ii.
```python
for t_idx in range(len(trialtm)):
    aligned_times[t_idx] = trialtm[t_idx] - goCue[trial_nums[t_idx] - 1]
for u_idx in range(n_units):
    for tr in range(ntrials):
        tr_mask = strials == (tr + 1)
```

iii. The agent did not discuss these efficiency choices in its final rationale; it prioritized producing and validating the dataset.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly constructs a lower-cased quality set, scans every unit's spikes once per trial, reopens each session to read the animal, repeatedly appends all interpolated values into Python lists, and separately interpolates speed and visibility for tongue and paw.

ii.
```python
if quality.lower() in {q.lower() for q in EXCLUDE_QUALITIES}:
...
with h5py.File(data_path, 'r') as f:  # reopened for animal
```

iii. No explicit justification was recorded; these are implementation conveniences.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads unused sample/delay events and lick references, accumulates unused qualities, reads the unused `moveThresh`, creates an unused `all_animals`, computes all trials before retaining valid trials, and computes/stores visibility alongside values later reduced to categorical outputs.

ii.
```python
sample_times = ev['sample'][0, :]
delay_times = ev['delay'][0, :]
lickL_refs = ev['lickL'][0, :]
all_qualities.append(quality)
me_thresh = float(me_struct['moveThresh'][0, 0].flatten()[0])
```

iii. The trajectory does not justify these discarded intermediates; several came from exploratory or general-purpose loading code.
