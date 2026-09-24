# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 44 paper sessions and their MATLAB-indexed probes, resolves each to one of two ephys folders, and auto-detects MATLAB v7.3 versus v5. It loads only selected fields from each session and loads motion energy from a companion file.

ii. ```python
SESSION_META = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    # ... 43 more sessions ...
]

def load_session(fpath):
    with open(fpath, 'rb') as ff:
        header = ff.read(15)
    return load_session_h5(fpath) if b'7.3' in header else load_session_v5(fpath)
```

iii. The notes say the list and probes came from the authors’ per-animal loading scripts; commented-out/behavior-only sessions were excluded, and both MATLAB formats had to be supported.

## 1-b. How are the data split into subjects?

i. The animal identifier stored in each `SESSION_META` tuple defines the subject. After session filtering, unique IDs are sorted and every retained session receives its index.

ii. ```python
subjects = sorted(set(r['anm'] for r in all_results))
subject_idx = np.array([subjects.index(r['anm']) for r in all_results])
```

iii. The notes identify 14 animals across the fixed- and randomized-delay datasets.

## 1-c. How are the data split into sessions?

i. Each `(animal, date)` data file is initially one session. However, the agent then drops sessions with fewer than 10 retained units or fewer than 40 valid right-hit or left-hit DR trials, producing 42 rather than 44 output sessions.

ii. ```python
if n_neurons < MIN_UNITS:
    return None
if n_r_hit < 40 or n_l_hit < 40:
    return None
```

iii. The agent cites the paper’s at-least-10-units and `UseInclusionCritera` rules. Its notes report that JEB19 2023-04-19 and 2023-04-20 were skipped for insufficient DR trials.

## 1-d. How are the data split into trials?

i. Bpod arrays are indexed directly from zero to `Ntrials-1`; spike `trial` IDs are treated as one-based. The final trials are those indices where `stim_enable` is false.

ii. ```python
for t in range(ntrials):
    trial_num = t + 1
    spike_mask = trial_ids == trial_num
...
trial_indices = np.where(~session['stim_enable'])[0]
trialdat = trialdat[:, :, trial_indices]
```

iii. The agent treats Bpod rows, cluster trial IDs, trajectory entries, and motion-energy entries as corresponding trial records.

## 1-e. How are trials filtered based on quality controls?

i. Only photostimulation trials are removed from the output. Early-lick trials are retained, although early, autowater, and stim trials are excluded when deciding whether an entire session has enough correct DR trials. Trials after ephys recording ended are not removed.

ii. ```python
valid_mask = ~session['stim_enable'] & ~session['autowater'] & ~session['early']
...
trial_mask = ~session['stim_enable']
trial_indices = np.where(trial_mask)[0]
```

iii. The planning notes explicitly choose to include early trials to maximize decoder data, while excluding photoinactivation trials. Later review acknowledges 64 all-zero neural trials near recording ends and calls them acceptable.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: each unit’s `trialtm`, `trial`, and `quality`, plus per-trial `bp.ev.goCue` for alignment.

ii. ```python
trialtm = u['trialtm']
trial_ids = u['trial']
aligned_times = trialtm[spike_mask] - go_cue[t]
```

iii. The notes map `obj.clu` spike times through `alignSpikes`, `getSeq`, and quality curation.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned, histogrammed into 10 ms bins, divided by 0.01 s to obtain Hz, and smoothed with a custom 15-point causal half-Gaussian using reflected left padding. Selected probes are concatenated.

ii. ```python
DT = 1.0 / 100
counts, _ = np.histogram(aligned_times, bins=EDGES)
fr = counts.astype(np.float64) / DT
trialdat[:, ui, t] = smooth_signal(fr)
```

iii. The agent believed most analysis scripts used 10 ms and that its causal kernel matched `mySmooth.m`, so it preferred that over the 5 ms default.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes four lower-cased quality labels, retains units only when mean window firing rate is greater than 1 Hz, drops sessions below 10 retained units, and does not exclude `poor` units.

ii. ```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
keep_mask = trialdat.mean(axis=(0, 2)) > LOW_FR
if n_neurons < MIN_UNITS:
    return None
```

iii. The notes attribute the four labels to `findClusters`, the 1 Hz threshold to paper/analysis scripts, and 10 units to the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each spike, the go-cue time for its Bpod trial is subtracted from `trialtm` before binning in the common window.

ii. ```python
aligned_times = trialtm[spike_mask] - go_cue[t]
counts, _ = np.histogram(aligned_times, bins=EDGES)
```

iii. The notes identify `goCue` as the requested and reference alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted grid has 500 non-overlapping 10 ms bins from -2.5 to +2.5 s. Raw spikes are binned to this grid; camera streams are interpolated to its centers.

ii. ```python
TMIN, TMAX = -2.5, 2.5
DT = 1.0 / 100
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The agent chose 10 ms because it found that value in most analysis scripts, despite noting that some/default code used 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a generated bin-center axis defined relative to each trial’s raw `bp.ev.goCue`; its numerical values come from the global window and `DT`, not an additional raw signal.

ii. ```python
TIME_AXIS = EDGES[:-1] + DT / 2
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. The agent states that the time axis follows the go-cue-aligned `getSeq` edges.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs equally spaced edge values from -2.5 to +2.5 s and adds half a bin to obtain 500 centers, then repeats the same `(1, 500)` float32 array for every trial.

ii. ```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The notes describe the requested continuous time input as the neural time grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input values are the centers of the exact histogram bins used after spike times are shifted by each trial’s go cue.

ii. ```python
counts, _ = np.histogram(aligned_times, bins=EDGES)
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. The common grid was intentionally used for all neural and behavioral arrays.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `obj.bp.R`, `L`, `hit`, and `miss`; neither recorded left/right lick event times nor `no` are used.

ii. ```python
if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 1
elif session['L'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 0
```

iii. The notes say lick direction is mapped from R/L and no-response status.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code labels any responded R-instructed trial as right and any responded L-instructed trial as left, regardless of hit versus miss; all other trials are none. Thus misses are assigned the instructed rather than actual opposite lick.

ii. ```python
if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 1
elif session['L'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 0
else:
    lick_dir[i] = 2
```

iii. The agent’s notes claim this maps lick direction, but they do not recognize the miss-side inversion. Their sanity check only checks ignore/no-lick consistency.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived solely from the per-trial `obj.bp.autowater` flag.

ii. ```python
if session['autowater'][ti]:
    context[i] = 0
else:
    context[i] = 1
```

iii. The mapping plan identifies autowater as the raw context variable.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct categorical recode is used: autowater true becomes WC (0), otherwise DR (1), and the scalar is repeated across time.

ii. ```python
out[1, :] = context[i]
```

iii. The agent describes autowater trials as water-cued and all others as delayed-response.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `obj.bp.hit` and `obj.bp.miss`; loaded `obj.bp.no` is not needed in the final mapping.

ii. ```python
if session['hit'][ti]:
    outcome[i] = 1
elif session['miss'][ti]:
    outcome[i] = 0
else:
    outcome[i] = 2
```

iii. The notes map hit, miss, and no/ignore to outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit is correct (1), miss is incorrect (0), and neither is ignore (2); the per-trial code is repeated across all time bins.

ii. ```python
out[2, :] = outcome[i]
```

iii. This follows the requested categorical ordering and retains ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera trajectory view, looking for feature name `tongue`, then reads its x, y, confidence, and `frameTimes`. It also uses bitcode-derived video shift and go-cue times. It does not use the bottom-camera `top_tongue`.

ii. ```python
view_data = session['traj'][0]
...
if fn.lower() == 'tongue':
    tongue_idx = fi
```

iii. The notes planned tongue from the side camera. Unlike the human solution, the agent did not justify omitting the complementary bottom view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial it takes un-smoothed x/y differences with `np.gradient` per frame index, computes Euclidean speed in pixels/frame, marks confidence below 0.1 NaN, and linearly interpolates speed onto neural bin centers. It does not differentiate by real time, smooth coordinates, normalize view scale, bin-average frames, or combine views.

ii. ```python
xvel = np.gradient(x)
yvel = np.gradient(y)
speed = np.sqrt(xvel**2 + yvel**2)
speed[conf < 0.1] = np.nan
interp_speed = np.interp(taxis, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. The notes broadly cite `findVelocity` and interpolation, but the implementation uses a much weaker visibility cutoff and simplified velocity calculation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After retaining the session’s output trials, one median is calculated over all non-NaN tongue values. Visible values below it are 0, values at/above it are 1, and NaNs are 2.

ii. ```python
threshold = np.percentile(speed_matrix[~np.isnan(speed_matrix)], 50)
result[visible & (speed_matrix < threshold)] = 0
result[visible & (speed_matrix >= threshold)] = 1
```

iii. The 50th-percentile per-session threshold and not-visible category come directly from the task.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code estimates one session video offset as median ephys bit-start time divided by sample rate minus median behavioral bit-start time. It computes `frameTimes - vidshift - goCue[trial]` and interpolates at `TIME_AXIS`. Missing offset data silently uses 0.5 s.

ii. ```python
session['vidshift'] = bit_start_sglx - bit_start_bpod
...
aligned_ft = frame_times - vidshift - go_cue_times[t]
interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. The agent cites `findVideoOffset`; it chose medians and a documented 0.5 s fallback rather than the reference’s modal offset.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `obj.traj[1]`, selecting the first feature name containing `paw`, with x, y, confidence, frame times, video shift, and go cue.

ii. ```python
view_data = session['traj'][1]
for fi, fn in enumerate(feat_names):
    if 'paw' in fn.lower():
        paw_idx = fi
        break
```

iii. The notes planned bottom-camera paw tracking but did not pin the choice to the reliably tracked `top_paw` feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It calculates un-smoothed frame-index x/y gradients and speed, marks confidence below 0.1 missing, fills all internal missing speeds by nearest linear interpolation over indices, and interpolates to neural centers.

ii. ```python
speed = np.sqrt(np.gradient(x)**2 + np.gradient(y)**2)
speed[conf < 0.1] = np.nan
speed = np.interp(np.arange(len(speed)), idx, speed[idx])
```

iii. The agent says nearest filling follows reference code, although the human conversion deliberately preserves missing velocity as not visible.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same session-wide visible-value median split is applied: below 0, at/above 1, NaN 2.

ii. ```python
paw_disc = discretize_velocity(paw_speed)
```

iii. The task explicitly requires a per-session 50th-percentile threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are shifted by the session video offset and trial go cue, then velocity is interpolated at the 10 ms neural bin centers.

ii. ```python
aligned_ft = frame_times - vidshift - go_cue_times[t]
interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. The notes intend all video streams to share the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads the standalone `motionEnergy_<animal>_<date>.mat` trace and uses side-camera trajectory `frameTimes`, video bitcode shift, and go cue for timing.

ii. ```python
me_pattern = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
...
ft_ref = view_data['_traj_grp']['frameTimes'][t, 0]
```

iii. The notes document several nested motion-energy layouts and a later loader fix.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The precomputed per-frame trace is linearly interpolated to bin centers. Length mismatches are truncated, and remaining partial NaNs are filled over time. It is not averaged within temporal bins.

ii. ```python
min_len = min(len(frame_times), len(trial_me))
...
me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
...
me_data[:, t] = np.interp(np.arange(len(col)), idx, col[idx])
```

iii. The notes state motion energy is loaded, interpolated, and discretized, following their reading of `loadMotionEnergy`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A median over all non-NaN retained-session values is used; below is 0, at/above is 1, and NaN/no video is 2.

ii. ```python
threshold = np.percentile(valid, 50)
result[visible & (me_matrix < threshold)] = 0
result[visible & (me_matrix >= threshold)] = 1
```

iii. This directly implements the requested per-session 50th-percentile categorization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the median-derived video shift and trial go cue, then interpolated onto the 10 ms neural centers. If frame times are unavailable, synthetic 400 Hz times are used.

ii. ```python
frame_times = np.arange(len(trial_me)) / 400.0
aligned_ft = frame_times - vidshift - go_cue_times[t]
me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
```

iii. The agent cites video-neural offset correction and interpolation from the reference pipeline.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses broad exception handling and fallbacks: missing frame times become a synthetic 400 Hz clock, missing video offset becomes 0.5 s, frame/value length mismatches are truncated, missing paw and motion-energy values are filled, unavailable streams remain NaN and become category 2, and missing files/sessions may be skipped. It retains all-zero late neural trials.

ii. ```python
except:
    session['vidshift'] = 0.5
...
frame_times = np.arange(n_frames) / 400.0
...
min_len = min(len(frame_times), len(trial_me))
```

iii. The notes call end-of-recording zero trials acceptable and say missing behavioral video maps to the third category; they prioritize completing conversion despite heterogeneous files.

## 11-a. What are the most time-consuming steps of the code?

i. The agent reports about 2.8 s/session and identifies per-unit/per-trial spike histogramming as the main explicit inefficiency; file loading plus trajectory/motion processing are also substantial. It did not provide a measured stage profile.

ii. ```python
for ui, u in enumerate(all_units):
    for t in range(ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=EDGES)
```

iii. The notes estimate runtime from samples and identify per-trial spike binning as an inefficiency, while claiming vectorized histogram speedups that the final main path does not actually use.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested unit-by-trial spike loop could be replaced by a two-dimensional histogram. Scalar loops producing lick/context/outcome, packaging trials, and some per-time-column missing-value operations could also be vectorized; ragged camera-trial loading is less amenable.

ii. ```python
for ui, u in enumerate(all_units):
    for t in range(ntrials):
        ...
for i, ti in enumerate(trial_indices):
    ...
```

iii. The agent explicitly flags per-trial spike binning as vectorizable but says added vectorized histogram speedups; that claim conflicts with the code used by `process_session`.

## 11-c. What processing does the code repeat multiple times?

i. Spike alignment/binning exists twice (`align_and_bin_spikes` and an inline copy), though only the inline version runs. Tongue and paw repeat nearly identical trial loading, coordinate extraction, gradient, masking, and interpolation logic. Per-trial outputs are then expanded into full time rows.

ii. ```python
def align_and_bin_spikes(...):
    ...
# process_session repeats the same nested spike code
...
out[0, :] = lick_dir[i]
```

iii. The notes do not acknowledge most duplication and describe the implementation as minimal-field loading with speedups.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused fields such as `no`, `sample`, and `delay`; constructs trajectory metadata for all views; computes `go_cue_filtered` only for optional plots; and includes unused helper functions (`align_and_bin_spikes`, `remove_low_fr_clusters`, `smooth_matrix`). If plotting is enabled it also produces diagnostics not used by the dataset.

ii. ```python
session['no'] = bp['no'][0, :].astype(bool)
session['sample'] = ev['sample'][0, :]
session['delay'] = ev['delay'][0, :]
...
go_cue_filtered = go_cue[trial_indices]
```

iii. The agent says it loads only needed fields and frames plotting as optional validation, but several loaded values and whole helper paths never affect the saved result.


