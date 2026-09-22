# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 25 fixed-delay sessions and their ALM probes, and opens each `data_structure_*.mat` with `h5py`. Motion energy is loaded separately with `scipy.io.loadmat`. It deliberately excludes the 19 randomized-delay sessions.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'
SESSIONS = [('EKH1', '2021-08-07', [2]), ...]
for anm, date, probes in SESSIONS:
    s = convert_session(anm, date, probes)
```

iii. The trajectory says the list and probe IDs came from the authors' `load<ANM>_ALMVideo.m` scripts. The final response justifies excluding randomized-delay data because its varying delay changes the pre-go-cue task epochs and those animals lacked WC trials.

## 1-b. How are the data split into subjects?

i. The subject is the hard-coded `anm` portion of each session tuple. Unique IDs are sorted and each session receives its index. This yields 10 subjects because only the fixed-delay subset is loaded.

ii.
```python
subjects = sorted({s['session_info']['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['session_info']['subject'])
                        for s in sessions], dtype=np.int64)
```

iii. The trajectory treats the animal IDs in the authors' loading scripts and filenames as authoritative.

## 1-c. How are the data split into sessions?

i. Each `(anm, date, probes)` tuple and corresponding MATLAB file becomes one session-level list entry. Only the 25 files under `Ephys_Behavior` are used.

ii.
```python
path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
data = {'neural': [s['neural'] for s in sessions], ...}
```

iii. The agent describes these as the paper's fixed-delay ALM ephys/video sessions and intentionally treats randomized-delay data as out of scope.

## 1-d. How are the data split into trials?

i. Bpod `Ntrials` defines trial count. Retained zero-based mask positions are converted to the 1-based IDs used by spike and video fields; each retained trial becomes one array in each session list.

ii.
```python
n = int(vec(bp, 'Ntrials')[0])
trials = np.where(keep)[0] + 1
for i in range(len(trials)):
    neural.append(np.ascontiguousarray(rates[i].T))
```

iii. The agent inspected the MATLAB layout and explicitly debugged the 1-based trial mapping during the trajectory.

## 1-e. How are trials filtered based on quality controls?

i. Trials with early licks, photostimulation, or non-finite go cues are removed. Hit, miss, and ignore trials remain. It does not remove behavioral trials recorded after ephys ended.

ii.
```python
keep = (~b['early']) & (~b['stim']) & np.isfinite(b['goCue'])
trials = np.where(keep)[0] + 1
```

iii. The agent says `~early & ~stim.enable` is the mask used throughout the paper and all outcome classes are required for decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected `obj.clu` probe groups: cluster `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue` for alignment.

ii.
```python
tm = np.array(f[g['trialtm'][icell, 0]]).flatten()
tr = np.array(f[g['trial'][icell, 0]]).flatten().astype(np.int64)
aligned = tm - align[tr - 1]
```

iii. The agent cites `alignSpikes.m`, `findClusters.m`, and the per-session ALM probe lists.

## 2-b. How is the `neural` data processed?

i. Spikes are counted in 10 ms bins, converted to spikes/s, and smoothed with a causal 15-bin Gaussian FIR using reflected prefix padding. Selected probes are concatenated; there is no z-scoring or baseline subtraction.

ii.
```python
DT = 1.0 / 100.0
rates = counts.astype(np.float64) / DT
flat = my_smooth(flat)
```

iii. The agent believed this exactly ported `getSeq.m`/`mySmooth.m` and numerically checked its convolution implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labelled garbage, gabrga, noisy, or real? are rejected. Remaining units are retained only when the smoothed all-retained-trial PSTH has mean firing rate above 1 Hz. The `poor` label is not rejected.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
if q.lower() in BAD_QUALITY: continue
if psth.mean() > LOW_FR: keep_rates.append(counts)
```

iii. The agent attributes both rules to `findClusters.m` and `removeLowFRClusters.m` and inspected all quality labels in the files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's go-cue time subtracted before histogramming over −2.5 to +2.5 s.

ii.
```python
aligned = tm - align[tr - 1]
ok = (row >= 0) & (aligned >= TMIN) & (aligned < TMAX)
```

iii. The agent cites `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 10 ms (500 bins over five seconds). Raw spikes are histogrammed directly into that grid; video streams are interpolated to it.

ii.
```python
DT = 1.0 / 100.0
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
```

iii. The agent chose 10 ms based on its reading of analysis parameters, despite the default/reference conversion using 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a generated array of centers of the common go-cue-aligned bins, not a separately measured raw field. `bp.ev.goCue` determines the alignment semantics.

ii.
```python
TAXIS = EDGES[:-1] + DT / 2
time_input = TAXIS.astype(np.float32)[None, :]
```

iii. The agent uses the common grid to guarantee correspondence with neural samples.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It constructs evenly spaced 10 ms bin centers from −2.495 to +2.495 s and copies the same row into every trial.

ii.
```python
TAXIS = EDGES[:-1] + DT / 2
inputs.append(time_input.copy())
```

iii. No separate justification was given beyond matching the neural grid and chosen time window.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input consists of the centers of the exact edges used to histogram aligned spikes, so input column `k` corresponds to neural bin `k`.

ii.
```python
b = ((aligned[ok] - TMIN) / DT).astype(np.int64)
TAXIS = EDGES[:-1] + DT / 2
```

iii. The agent explicitly describes all streams as sharing the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses Bpod instructed-side flags `R` and `L` together with `hit` and `miss`; neither outcome becomes `none`.

ii.
```python
licked_right = (b['R'][idx] & b['hit'][idx]) | (b['L'][idx] & b['miss'][idx])
licked_left = (b['L'][idx] & b['hit'][idx]) | (b['R'][idx] & b['miss'][idx])
```

iii. The agent reasons that hits lick the instructed side, misses the opposite side, and ignores do not lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Left, right, and none are encoded 0, 1, and 2. The value is constant across all bins of a trial.

ii.
```python
lick = np.full(len(idx), 2, dtype=np.int8)
lick[licked_left] = 0; lick[licked_right] = 1
out[0] = lick[i]
```

iii. This directly implements the requested three categories.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from `obj.bp.autowater`.

ii.
```python
'autowater': np.nan_to_num(vec(bp, 'autowater')[:n]) > 0
```

iii. The agent identifies autowater trials as WC and all others as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is mapped to WC=0 and false to DR=1, repeated through the trial.

ii.
```python
context = np.where(b['autowater'][idx], 0, 1).astype(np.int8)
out[1] = context[i]
```

iii. This follows the requested class order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses Bpod `hit` and `miss`; `no` is loaded but the default class supplies ignore.

ii.
```python
'hit': ... vec(bp, 'hit')[:n] > 0,
'miss': ... vec(bp, 'miss')[:n] > 0,
```

iii. The agent treats hit/miss/neither as mutually exclusive correct/incorrect/ignore states.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. It initializes ignore=2, sets misses to incorrect=0 and hits to correct=1, then repeats the label over time.

ii.
```python
outcome = np.full(len(idx), 2, dtype=np.int8)
outcome[b['miss'][idx]] = 0
outcome[b['hit'][idx]] = 1
```

iii. This follows the prompt's category definitions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera `tongue` x/y positions from `obj.traj.ts`, with frame times, go cues, and clock bitcode for timing. It does not combine the bottom-camera tongue feature.

ii.
```python
TONGUE_FEATURE = ('tongue', 1)
pos, has_video = load_traj(...)
tspeed, tvis = tongue_speed(pos[TONGUE_FEATURE[0]])
```

iii. The agent says this follows the paper's lick-onset baseline-fill kinematics pipeline.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Tracking is linearly interpolated to 10 ms bin centers. Missing x/y values are filled with the session mean position at visible-run starts, then `np.gradient` is taken along bins and x/y displacements are combined. Visibility is retained separately for class 2.

ii.
```python
filled[..., 0] = np.where(np.isfinite(filled[..., 0]), filled[..., 0], mux)
vx = np.gradient(filled[..., 0], axis=1)
return np.sqrt(vx ** 2 + vy ** 2), visible
```

iii. The trajectory cites `setTongueBaselinePosition()` and `findVelocity.m` as the model for baseline filling and differentiation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over finite visible samples: below is 0, at/above is 1, invalid/not visible is 2.

ii.
```python
thresh = np.percentile(v, 50)
out[finite & (values < thresh)] = 0
out[finite & (values >= thresh)] = 1
```

iii. The agent follows the explicit per-session 50th-percentile requirement.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by a session video/ephys offset and the trial go cue, then linearly interpolated onto the neural bin centers.

ii.
```python
t_src = ft - vidshift - align[j]
pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```

iii. The agent cites `findVideoOffset.m` and uses bitcode timing to reconcile clocks.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses both `top_paw` and `bottom_paw` x/y tracks from the bottom camera, plus frame timing and clock-alignment fields.

ii.
```python
PAW_FEATURES = [('top_paw', 2), ('bottom_paw', 2)]
pspeed, pvis = paw_speed([pos[n] for n, _ in PAW_FEATURES])
```

iii. The agent says the paper's kinematics pipeline supports combining the two tracked paw markers.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated to 10 ms, nearest-filled per trial, differentiated, corrected by subtracting median frame-to-frame drift in each axis, converted to speed, and averaged over whichever paw markers are visible.

ii.
```python
x = fill_nearest(pos[i, :, 0])
vx = np.gradient(x) - np.nanmedian(np.diff(x))
mean_speed = np.nansum(np.where(visibles, speeds, np.nan), axis=0) / ...
```

iii. The agent attributes nearest filling and drift removal to `findPosition.m`/`findVelocity.m`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide median over valid samples defines below=0 and at/above=1; unavailable samples are 2.

ii.
```python
paw_code = discretize(pspeed, pvis)
```

iii. This follows the prompt's requested threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are clock-corrected, made relative to each trial's go cue, and interpolated to `TAXIS`.

ii.
```python
t_src = ft - vidshift - align[j]
pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```

iii. The same alignment path is intentionally shared by all video features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses the per-trial trace in `motionEnergy_<anm>_<date>.mat`, side-camera frame times, video clock offset, and go cue.

ii.
```python
fn = os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat')
me = m['me'][0, 0]
```

iii. The agent cites `loadMotionEnergy.m` and handles nested MATLAB wrappers seen in some files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The existing scalar trace is linearly interpolated to 10 ms centers and then all remaining gaps are nearest-filled. Mismatched frame counts use an assumed 400 Hz/0.5 s fallback clock.

ii.
```python
out[i] = interp_to_taxis(t_src, y, TAXIS)
out[i] = fill_nearest(out[i])
```

iii. The agent says the upstream file already contains reduced motion energy and the fallback mirrors the MATLAB catch branch.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The session median over finite samples creates classes 0 and 1; missing samples would be class 2, although nearest filling means class 2 does not occur in the produced data.

ii.
```python
me_code = discretize(me, np.isfinite(me))
```

iii. The agent follows the requested 50th percentile and flags in its final response that no-video class 2 is absent.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by video offset and trial go cue and interpolated to the neural grid; the fallback synthesizes frame times when counts mismatch.

ii.
```python
t_src = ft - vidshift - align[j]
out[i] = interp_to_taxis(t_src, y, TAXIS)
```

iii. The agent validates alignment qualitatively from movement changes around the go cue.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Numeric behavioral NaNs become false; missing bitcode falls back to 0.5 s; absent motion-energy files become NaN/class 2; motion-energy gaps are nearest-filled; paw gaps are nearest-filled; tongue gaps are baseline-filled for differentiation but remain class 2 via the visibility mask; malformed/short video trials remain unavailable. Sessions with fewer than two trials or no units are skipped.

ii.
```python
except Exception: return 0.5
if not os.path.exists(fn): return np.full(..., np.nan)
out[i] = fill_nearest(out[i])
```

iii. The agent says these fallbacks follow paper code and preserve usable neural/behavior data; it also reports validating shapes and decoder training.

## 11-a. What are the most time-consuming steps of the code?

i. The code does not profile stages. The likely expensive work is HDF5 loading, per-cluster spike binning/smoothing, per-trial video interpolation, and serialization/training validation.

ii.
```python
for icell, q in enumerate(qualities): ...
for i, trial in enumerate(trials): ...
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory reports full conversion runs and decoder runs but gives no timing-based attribution among conversion stages.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Session loops are structurally necessary, but cluster and ragged trial/video loops could potentially be batched after padding/index flattening. The final trial assembly loop could be replaced by bulk array construction before conversion to lists.

ii.
```python
for p in probes:
    for icell, q in enumerate(qualities): ...
for i in range(len(trials)):
    neural.append(...); outputs.append(out)
```

iii. The agent did vectorize spike counting within each unit with `np.bincount`, but did not explicitly discuss further vectorization.

## 11-c. What processing does the code repeat multiple times?

i. `load_traj` repeatedly dereferences trial/view HDF5 objects and interpolates each of three features; paw filling/gradients repeat for two markers; trial arrays are copied during final list assembly. Session-wide offset and thresholds are sensibly computed once.

ii.
```python
for name, view in wanted:
    ts = np.array(f[v['ts'][j, 0]])
    pos[name][i] = interp_to_taxis(...)
```

iii. The trajectory focuses on fidelity rather than identifying repeated work; the shared interpolation path is deliberate.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads behavior fields `no`, `sample`, and `delay` that do not affect converted arrays; tracks `keep_ids` without using it; computes metadata counts; and copies identical time inputs for every trial. It also computes filled tongue/paw speeds at invisible bins that are later overwritten by class 2.

ii.
```python
'no': ..., 'sample': ..., 'delay': ...
rates, nquality, keep_ids = neural_matrix(...)
inputs.append(time_input.copy())
```

iii. The agent did not justify these as necessary downstream; most support diagnostics, provenance, or a straightforward uniform implementation.
