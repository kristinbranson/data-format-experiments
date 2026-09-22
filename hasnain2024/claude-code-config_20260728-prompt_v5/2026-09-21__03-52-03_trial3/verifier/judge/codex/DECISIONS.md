# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a session list in `SESSION_META`, then loads each session from either `/app/data/Ephys_Behavior/` or `/app/data/RandomizedDelay_Ephys_Behavior/`. It uses `load_session()` to auto-detect MATLAB v7.3 versus v5 files, and separately loads `motionEnergy_<anm>_<date>.mat` for motion energy.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ...
    ('JEB24', '2023-11-03', [1], 'randdelay'),
]

data_dir = DATA_DIRS[data_dir_key]
fpath = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
session = load_session(fpath)
```

```python
def load_session(fpath):
    with open(fpath, 'rb') as ff:
        header = ff.read(15)
    if b'7.3' in header:
        return load_session_h5(fpath)
    else:
        return load_session_v5(fpath)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says it will include all ephys and randomized-delay sessions listed in the authors' loading scripts and handle both MATLAB formats plus separate motion-energy files.

## 1-b. How are the data split into subjects?

i. The AI uses the animal id from `SESSION_META` as the subject id, stores it as `anm` in each processed session, and later builds `subjects` and `subject_idx` from the unique animal names.

ii.
```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'anm': anm,
    'date': date,
    ...
}
```

```python
subjects = sorted(set(r['anm'] for r in all_results))
subject_idx = np.array([subjects.index(r['anm']) for r in all_results])
```

iii. `CONVERSION_NOTES.md` Step 5 says the loading scripts define the session metadata, and the output summary in Step 9 reports 14 subjects derived from those animal ids.

## 1-c. How are the data split into sessions?

i. One tuple in `SESSION_META` corresponds to one session. Each tuple provides animal, date, probe list, and which data directory to search. Each processed session becomes one element in `neural`, `input`, and `output`.

ii.
```python
for idx, (anm, date, probes, ddir) in enumerate(sessions_to_process):
    result = process_session(anm, date, probes, ddir, ...)
    if result is not None:
        all_results.append(result)
```

iii. The AI justifies this in Step 5 by saying session membership should follow the authors' loading scripts rather than data-directory globbing.

## 1-d. How are the data split into trials?

i. The AI treats `bp.Ntrials` as the number of trials, keeps per-trial arrays from `bp`, and loops over `t in range(ntrials)` when binning spikes and computing outputs. Trial selection is later done with `trial_indices`.

ii.
```python
session['ntrials'] = int(bp['Ntrials'][0, 0])
...
for t in range(ntrials):
    trial_num = t + 1
    spike_mask = trial_ids == trial_num
```

```python
trial_mask = ~session['stim_enable']
trial_indices = np.where(trial_mask)[0]
trialdat = trialdat[:, :, trial_indices]
```

iii. The notes treat Bpod trial fields as the canonical trial structure and do not describe any need to reconstruct trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies session inclusion using delayed-response hit counts with `~stim_enable & ~autowater & ~early`, but for the exported dataset it drops only stimulation trials. Early trials, autowater trials, ignore trials, and trials after recording end are kept.

ii.
```python
valid_mask = ~session['stim_enable'] & ~session['autowater'] & ~session['early']
r_hit = session['R'] & session['hit'] & valid_mask
l_hit = session['L'] & session['hit'] & valid_mask
...
if n_r_hit < 40 or n_l_hit < 40:
    ... return None
```

```python
trial_mask = ~session['stim_enable']
trial_indices = np.where(trial_mask)[0]
```

iii. In Step 5, the AI explicitly says it wants to keep "ALL trials" for decoder training except stimulation trials, while still enforcing the paper's session-inclusion rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from cluster spike times and trial ids in `obj.clu` plus go-cue times from `obj.bp.ev.goCue`. Cluster quality labels are used for filtering.

ii.
```python
units.append({'quality': q_str, 'trialtm': trialtm, 'trial': trial})
...
session['goCue'] = ev['goCue'][0, :]
```

iii. The notes map "obj.clu spike times" to `neural` and describe the reference pipeline as quality filter -> align spikes to go cue -> bin -> smooth.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 10 ms bins from -2.5 s to 2.5 s around go cue, converts counts to firing rates by dividing by `DT`, then applies a causal 15-point Gaussian smoothing kernel with reflect padding.

ii.
```python
DT = 1.0 / 100
EDGES = np.arange(TMIN, TMAX + DT, DT)
...
counts, _ = np.histogram(aligned_times, bins=EDGES)
fr = counts.astype(np.float64) / DT
trialdat[:, ui, t] = smooth_signal(fr)
```

```python
def causal_gaussian_kernel(n):
    ...
    kern[:n // 2] = 0
```

iii. In Steps 4 and 5, the AI says most analysis scripts use 10 ms bins and `mySmooth.m` uses a causal 15-point Gaussian with reflect boundary conditions, so it followed that path.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters whose lower-cased quality label is not in `{'garbage', 'gabrga', 'noisy', 'real?'}`, then removes neurons with mean firing rate `<= 1 Hz`, and finally drops whole sessions with fewer than 10 neurons after filtering.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
def filter_clusters(units, excluded_qualities=EXCLUDED_QUALITIES):
    ...
    if q not in excluded_qualities:
        good_idx.append(i)
```

```python
mean_frs = trialdat.mean(axis=(0, 2))
keep_mask = mean_frs > LOW_FR
...
if n_neurons < MIN_UNITS:
    ... return None
```

iii. Step 5 says the AI chose the paper's 1 Hz cutoff, the standard quality filter, and an additional minimum-10-units session filter based on the methods text.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each trial's go-cue time from `trialtm` before binning.

ii.
```python
aligned_times = trialtm[spike_mask] - go_cue[t]
counts, _ = np.histogram(aligned_times, bins=EDGES)
```

iii. The notes repeatedly identify `goCue` as the reference event and describe `alignSpikes` as subtracting that event time from spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a 10 ms time bin (`DT = 1/100`) and does not perform any additional rebinning after that.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. Step 4 says the AI resolved the `1/100` versus `1/200` ambiguity by choosing 10 ms because it believed that was the bin size used by most analysis scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The AI does not read a separate raw variable for this input. It constructs the input from the analysis time grid defined around the go cue.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. The notes map "Time axis" to `input[0]: time_from_go_cue` and describe it as a continuous range from -2.5 to 2.5 seconds.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes the input as the centers of uniformly spaced bins spanning -2.5 s to 2.5 s, then repeats that same 1-by-time array for every trial.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
for i in range(n_trials_out):
    input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this as using the same time base as the neural processing pipeline.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the neural binning grid. The same `TIME_AXIS` used to represent the decoder input is also the target time base for spike histograms and interpolated behavioral outputs.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
counts, _ = np.histogram(aligned_times, bins=EDGES)
...
interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. The notes say the decoder input should be the go-cue-centered time axis shared by all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. In code, the AI derives lick direction from the per-trial side flags `R` and `L` together with whether the trial was a `hit` or `miss`. Ignore/no-response trials become a third category.

ii.
```python
if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 1
elif session['L'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 0
else:
    lick_dir[i] = 2
```

iii. Step 5 maps `obj.bp.R`, `obj.bp.L`, and `obj.bp.no` to `lick_direction`, and the later notes treat the third class as the no-response case.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI labels any right-instructed hit or miss trial as `right`, any left-instructed hit or miss trial as `left`, and all other trials as `none`. It does not flip miss trials to the opposite lick direction.

ii.
```python
if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 1  # right
elif session['L'][ti] and (session['hit'][ti] or session['miss'][ti]):
    lick_dir[i] = 0  # left
else:
    lick_dir[i] = 2  # none
```

iii. The notes justify the output as a categorical left/right/none variable and emphasize keeping ignore trials as their own class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI derives behavioral context from the `autowater` flag.

ii.
```python
if session['autowater'][ti]:
    context[i] = 0  # WC
else:
    context[i] = 1  # DR
```

iii. Step 5 explicitly maps `obj.bp.autowater` to behavioral context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly relabels `autowater == True` as WC and everything else as DR.

ii.
```python
context = np.zeros(len(trial_indices), dtype=np.int8)
for i, ti in enumerate(trial_indices):
    if session['autowater'][ti]:
        context[i] = 0
    else:
        context[i] = 1
```

iii. In Step 5, the AI says this is a direct per-trial categorical mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derives outcome from `hit` and `miss`; if neither is true, the trial is treated as ignore.

ii.
```python
if session['hit'][ti]:
    outcome[i] = 1
elif session['miss'][ti]:
    outcome[i] = 0
else:
    outcome[i] = 2
```

iii. The notes map `hit`, `miss`, and `no` to outcome and describe the output as correct/incorrect/ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI relabels hits as `correct`, misses as `incorrect`, and all remaining trials as `ignore`.

ii.
```python
outcome = np.zeros(len(trial_indices), dtype=np.int8)
for i, ti in enumerate(trial_indices):
    if session['hit'][ti]:
        outcome[i] = 1  # correct
    elif session['miss'][ti]:
        outcome[i] = 0  # incorrect
    else:
        outcome[i] = 2  # ignore
```

iii. This follows the Step 5 mapping and the prompt's required categorical output.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses DLC trajectories from the first camera view only, specifically the feature named `tongue`, along with `frameTimes`, `goCue`, and a session video offset `vidshift`.

ii.
```python
view_data = session['traj'][0]  # side cam
...
if fn.lower() == 'tongue':
    tongue_idx = fi
```

```python
aligned_ft = frame_times - vidshift - go_cue_times[t]
```

iii. Step 5 says the tongue signal comes from DLC tracks and should be turned into total speed, and the AI's code operationalizes that using the side camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes tongue speed as `sqrt(gradient(x)^2 + gradient(y)^2)` on the raw per-frame coordinates from the side camera, marks frames with confidence `< 0.1` as missing, and linearly interpolates the speed trace onto the neural time axis. It does not smooth positions, use real frame-time derivatives, split by contiguous visible runs, or combine the second tongue view.

ii.
```python
xvel = np.gradient(x)
yvel = np.gradient(y)
speed = np.sqrt(xvel**2 + yvel**2)
speed[conf < 0.1] = np.nan
...
interp_speed = np.interp(taxis, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. In Step 5, the AI justifies tongue velocity as total speed from DLC and says missing visibility should remain category 2 after discretization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses the 50th percentile of all non-NaN tongue-speed values within a session as the threshold. Values below threshold are class 0, values at or above threshold are class 1, and NaNs become class 2.

ii.
```python
valid = speed_matrix[~np.isnan(speed_matrix)]
threshold = np.percentile(valid, 50)
...
result[visible & (speed_matrix < threshold)] = 0
result[visible & (speed_matrix >= threshold)] = 1
```

iii. Step 5 explicitly says tongue velocity should be discretized by the per-session 50th percentile and use category 2 for not-visible bins.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-level video offset from `bitStart` and `sglx.bitcode.bitstart`, subtracts both that offset and each trial's go cue from `frameTimes`, and then interpolates the result onto `TIME_AXIS`.

ii.
```python
bit_start_bpod = float(np.nanmedian(ev['bitStart'][0, :]))
bit_start_sglx = float(np.nanmedian(sglx['bitcode']['bitstart'][0, :])) / float(sglx['fs'][0, 0])
session['vidshift'] = bit_start_sglx - bit_start_bpod
```

```python
aligned_ft = frame_times - vidshift - go_cue_times[t]
interp_speed = np.interp(taxis, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. The notes say video should be offset-corrected with bitcode timing and then interpolated to the neural time base after alignment to go cue.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses the second trajectory view (`session['traj'][1]`) and selects the first feature whose name contains `'paw'`, rather than fixing the feature to `top_paw`.

ii.
```python
view_data = session['traj'][1]  # bottom cam
...
for fi, fn in enumerate(feat_names):
    if 'paw' in fn.lower():
        paw_idx = fi
        break
```

iii. Step 5 says paw velocity should come from DLC and the bottom camera, but the implementation leaves the exact paw feature ambiguous.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI computes paw speed as the framewise gradient magnitude of x and y coordinates, marks low-confidence frames as NaN, then fills missing values within a trial by 1D interpolation before interpolating the trace onto `TIME_AXIS`.

ii.
```python
xvel = np.gradient(x)
yvel = np.gradient(y)
speed = np.sqrt(xvel**2 + yvel**2)
speed[conf < 0.1] = np.nan
...
speed = np.interp(np.arange(len(speed)), idx, speed[idx])
...
interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. The notes justify paw velocity as total DLC speed with missing values handled sensibly and then discretized session-wise.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI thresholds paw velocity exactly the same way as tongue velocity: session-specific 50th percentile on non-NaN values, with NaNs mapped to class 2.

ii.
```python
def discretize_velocity(speed_matrix):
    ...
    threshold = np.percentile(valid, 50)
    ...
    result[visible & (speed_matrix < threshold)] = 0
    result[visible & (speed_matrix >= threshold)] = 1
```

iii. Step 5 says both tongue and paw velocity should use the prompt's per-session median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligns paw tracking by subtracting the session video offset and the trial's go-cue time from `frameTimes`, then interpolating onto the neural time axis.

ii.
```python
aligned_ft = frame_times - vidshift - go_cue_times[t]
interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. The notes treat paw tracking alignment the same way as tongue tracking alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI loads motion energy from the separate `motionEnergy_<anm>_<date>.mat` file, not from `obj.me`, and pairs it with trajectory frame times for alignment.

ii.
```python
me_pattern = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
me_files = glob.glob(me_pattern)
...
me_path = me_files[0]
```

iii. Step 5 explicitly says motion energy should come from separate `.mat` files and the notes describe bug fixes for several motion-energy file layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI unwraps different MATLAB storage layouts, truncates frame times and motion-energy arrays to a common length if necessary, aligns by video offset and go cue, interpolates to `TIME_AXIS`, and then fills NaNs within each trial by nearest/linear interpolation.

ii.
```python
if len(frame_times) != len(trial_me):
    min_len = min(len(frame_times), len(trial_me))
    frame_times = frame_times[:min_len]
    trial_me = trial_me[:min_len]

aligned_ft = frame_times - vidshift - go_cue_times[t]
me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
me_data[:, t] = me_interp
```

```python
for t in range(ntrials):
    col = me_data[:, t]
    mask = np.isnan(col)
    if mask.any() and not mask.all():
        idx = np.where(~mask)[0]
        me_data[:, t] = np.interp(np.arange(len(col)), idx, col[idx])
```

iii. The notes say motion energy should be loaded from separate files, interpolated to the neural time base, and missing values handled with nearest filling, which the AI believed matched the reference pipeline.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses the session-wide 50th percentile of non-NaN motion-energy samples. Values below threshold map to 0, values at or above threshold map to 1, and NaNs would map to 2.

ii.
```python
def discretize_motion_energy(me_matrix):
    result = np.full_like(me_matrix, 2, dtype=np.int8)
    valid = me_matrix[~np.isnan(me_matrix)]
    threshold = np.percentile(valid, 50)
    result[visible & (me_matrix < threshold)] = 0
    result[visible & (me_matrix >= threshold)] = 1
```

iii. Step 5 says motion energy should be discretized with the prompt's per-session 50th percentile rule and use category 2 for missing video.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses side-camera frame times from `traj`, subtracts the session video offset and trial go cue, then interpolates motion energy to the same `TIME_AXIS` used by the neural data.

ii.
```python
if len(session['traj']) > 0 and '_traj_grp' in session['traj'][0]:
    view_data = session['traj'][0]
    ft_ref = view_data['_traj_grp']['frameTimes'][t, 0]
    frame_times = np.array(view_data['_f'][ft_ref]).flatten()
...
aligned_ft = frame_times - vidshift - go_cue_times[t]
me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
```

iii. The notes describe motion energy as another video-derived stream that should be offset-corrected and placed on the neural time base.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several fallback rules: if `vidshift` cannot be computed it defaults to `0.5`; if `frameTimes` are missing it synthesizes them with `np.arange(...) / 400`; for paw and motion energy it fills missing samples by interpolation; for tongue it leaves low-confidence samples as NaN; and for motion-energy files it recursively unwraps alternative struct layouts. Trials with all-zero neural activity at session ends are left in the exported dataset.

ii.
```python
except:
    session['vidshift'] = 0.5
...
frame_times = np.arange(n_frames) / 400.0
...
speed = np.interp(np.arange(len(speed)), idx, speed[idx])
...
me_data[:, t] = np.interp(np.arange(len(col)), idx, col[idx])
```

iii. The notes frame these as pragmatic fixes for file-format irregularities and missing-video cases, and later accept the all-zero neural trials as "acceptable" rather than removing them.

## 11-a. What are the most time-consuming steps of the code?

i. The AI's implementation is dominated by per-unit, per-trial spike binning and by repeated trial-wise video processing and interpolation. It also spends meaningful time reading the MATLAB files and motion-energy files.

ii.
```python
for ui, u in enumerate(all_units):
    ...
    for t in range(ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=EDGES)
```

```python
for t in range(ntrials):
    ...
    interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. In Step 6, the AI explicitly calls out per-trial spike binning as the main inefficiency and estimates runtime from sample conversion timing.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several obvious Python loops in place: looping over units and trials during spike binning, looping over trials for tongue velocity, paw velocity, and motion energy, and looping over trials again while packaging outputs.

ii.
```python
for ui, u in enumerate(all_units):
    for t in range(ntrials):
        ...
```

```python
for t in range(ntrials):
    ...
for i in range(n_trials_out):
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(out)
```

iii. The notes acknowledge that spike binning could be vectorized but say the added complexity was not worth it for roughly 44 sessions.

## 11-c. What processing does the code repeat multiple times?

i. The AI duplicates spike-binning logic instead of reusing `align_and_bin_spikes()`, repeatedly reconstructs the same `TIME_AXIS` trial input inside the output-packaging loop, and separately performs similar interpolation/discretization passes for tongue, paw, and motion energy.

ii.
```python
def align_and_bin_spikes(...):
    ...

...
trialdat = np.zeros((N_TIMEBINS, len(all_units), ntrials), dtype=np.float32)
for ui, u in enumerate(all_units):
    ...
```

```python
for i in range(n_trials_out):
    input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. The notes claim the code is efficient and minimally repetitive, but the final implementation still duplicates a few major operations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and stores raw fields that are never used downstream, including `sample`, `delay`, and some trial flags. It also imports plotting machinery and computes `go_cue_filtered` only for optional plots, and it repeats constant per-trial output values across all time bins even though they are session-trial labels.

ii.
```python
session['sample'] = ev['sample'][0, :]
session['delay'] = ev['delay'][0, :]
session['no'] = bp['no'][0, :].astype(bool)
session['L'] = bp['L'][0, :].astype(bool)
```

```python
go_cue_filtered = go_cue[trial_indices]
...
out[0, :] = lick_dir[i]
out[1, :] = context[i]
out[2, :] = outcome[i]
```

iii. The notes emphasize visual debugging and broad data loading, so some extra processing remains even though it is not needed for the final decoder dataset.
