# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans both electrophysiology folders for every `data_structure_*.mat` file, parses the animal/date from each filename, and associates a motion-energy file when present. It parses the authors’ MATLAB loading scripts for probe choices, defaulting to probe 1 when no entry is found. Separate readers handle MATLAB v5 and v7.3/HDF5 files.

ii.
```python
for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
    for fn in sorted(os.listdir(data_dir)):
        if not fn.startswith('data_structure_'):
            continue
        probes = probe_map.get((animal, date), [1])
        sessions.append({'animal': animal, 'date': date, 'filepath': filepath,
                         'probes': probes, 'me_filepath': me_filepath if has_me else None})
```

iii. The notes say the dual-format loader was needed because most files are HDF5 but some randomized-delay files are MATLAB v5. The agent intended to include “all available” data and use loading scripts for probe metadata.

## 1-b. How are the data split into subjects?

i. Subject identity is parsed from the animal component of each filename. Unique included animals are sorted, and each included session gets an integer index into that list.

ii.
```python
animal = m.group(1)
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The notes identify 14 animals in the files and use filename animal IDs consistently, avoiding reliance on incomplete internal metadata.

## 1-c. How are the data split into sessions?

i. Each matching data-structure file is initially one session. Sessions are then excluded unless they have more than 40 left-hit and right-hit DR trials, at least ten retained units, and at least two retained trials. Exceptions also cause a session to be skipped.

ii.
```python
included, n_r, n_l = check_session_inclusion(session, probes)
if not included: return None
if n_neurons < MIN_UNITS: return None
if len(trial_indices) < 2: return None
```

iii. The agent says this matches `UseInclusionCritera.m`. Its final notes acknowledge 43 sessions rather than the reference 44, including an extra session not in the loading scripts while excluding others by its criteria.

## 1-d. How are the data split into trials?

i. Raw per-trial arrays use `bp.Ntrials`; spikes use their 1-based `unit['trial']` labels; trajectory entries and motion-energy cell entries are indexed by the same zero-based Python trial index. Retained trial indices select corresponding columns from all processed arrays.

ii.
```python
for j in range(Ntrials):
    trial_num = j + 1
    spk_mask = trial_nums == trial_num
...
for trial_idx in trial_indices:
    neural = firing_rates[:, :, trial_idx]
```

iii. The notes describe the session files as containing behavior, spikes, trajectories, and motion energy organized per trial; no inferred trial boundaries are used.

## 1-e. How are trials filtered based on quality controls?

i. Trials with early licks, photostimulation, or `bp.no` (ignore/no-response) are removed. The session-level hit-count criterion is computed after excluding early, stimulated, and autowater trials. Late trials after electrophysiology ended are not removed.

ii.
```python
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
```

iii. The agent cites the paper for removing early/stim trials, but explicitly chose to “Also exclude 'no' (ignore) trials.” Its review notes 30 all-zero late neural trials and says the decoder should handle them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected `obj.clu` probes: each unit’s `trial`, `trialtm`, and `quality`, together with `bp.ev.goCue` and `bp.Ntrials`.

ii.
```python
trial_nums = unit['trial']
trialtm = unit['trialtm']
aligned_times = trialtm[spk_mask] - goCue[j]
```

iii. The notes identify `alignSpikes.m`, `getSeq.m`, garbage deletion, and low-rate removal as the reference path being emulated.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned per trial, histogrammed, converted to Hz, and convolved with a custom 15-bin causal Gaussian. Retained probes are concatenated. There is no baseline subtraction or normalization.

ii.
```python
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float64) / DT
fr_smooth = smooth_signal(fr, SMOOTH_N)
firing_rates = np.concatenate(all_firing_rates, axis=0)
```

iii. The agent says causal smoothing matches `mySmooth.m`: a Gaussian window of 15 bins, with its first half zeroed, normalized, and convolved in `same` mode.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Any quality string containing `garbage` is excluded. Units whose mean processed rate over all times and trials is not greater than 0.5 Hz are excluded; sessions with fewer than ten remaining units are excluded.

ii.
```python
if 'garbage' not in quality: valid_units.append(u_idx)
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR
if n_neurons < MIN_UNITS: return None
```

iii. The notes prioritize `getDefaultParams.m` (`lowFR=0.5`, quality “all”) over the paper’s 1 Hz statement, and interpret “all” as keeping non-garbage Fair/Good/Poor/Multi units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike’s time relative to trial start is shifted by that trial’s go-cue time, then histogrammed on the common time grid.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The notes explicitly identify this as the operation in `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The nominal resolution is 5 ms. The code constructs 1001 sample locations from -2.5 through +2.5 seconds inclusive and treats them as left bin edges, appending a final +2.505-second edge; spikes are therefore rebinned into 1001 bins.

ii.
```python
DT = 1.0 / 200.0
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
edges = np.append(time_axis, time_axis[-1] + DT)
```

iii. The agent says this matches MATLAB `tmin:dt:tmax` and records “1001 bins at 5ms.”

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not copied from a raw variable; it is the fixed analysis grid defined relative to the raw `bp.ev.goCue` alignment event.

ii.
```python
time_axis = make_time_axis()
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes describe the input as the -2.5 to +2.5-second go-cue-relative time axis shared by all trials.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. `np.arange` generates values at 5-ms intervals including both endpoints; the same array is converted to float32 and reshaped for every retained trial.

ii.
```python
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = time_axis[time_axis <= TMAX + 1e-10]
```

iii. The agent’s justification is that this reproduces MATLAB’s inclusive colon expression.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input array is the same grid passed to spike processing, so its columns correspond directly to neural histogram columns.

ii.
```python
fr, kept = process_spikes(session, p_idx, time_axis)
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes state all streams use the go-cue-relative neural time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived solely from the instructed-side flag `bp.R` after no-response trials have been removed; `bp.L`, hit, and miss are not used when assigning this output.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. The mapping plan says `obj.bp.R` maps directly to lick direction (`R=1, L=0`).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Right-instructed trials receive 1 and all others receive 0, repeated across all time bins. There is no `none` class and incorrect trials are not reversed to reflect the port actually licked.

ii.
```python
out[0, :] = lick_dir
'output_values': [['left', 'right'], ...]
```

iii. The agent treated instructed direction as observed lick direction and removed `bp.no` trials, despite the requested left/right/none categories.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `bp.autowater`.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. The notes describe autowater as WC and its complement as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is relabeled WC=0 and non-autowater DR=1, then repeated over time.

ii.
```python
out[1, :] = context
'output_values': [['left', 'right'], ['WC', 'DR'], ...]
```

iii. This directly follows the mapping plan and requested labels.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The stored outcome uses only `bp.hit`; miss/non-hit becomes incorrect. Although `bp.miss` and `bp.no` are loaded, no-response trials are filtered before output construction.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. The mapping plan states hit maps to correct and everything retained otherwise maps to incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is binary (incorrect=0, correct=1), repeated across time. The requested ignore class is omitted because those trials were removed.

ii.
```python
out[2, :] = outcome
'output_values': [..., ['incorrect', 'correct'], ...]
```

iii. The agent explicitly excluded `bp.no` and documented only two outcome labels.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses `obj.traj` camera 0 feature `tongue`: x/y coordinates, frame times, and dropped-frame status. It also uses `bp.ev.goCue`, behavioral bit starts, and SpikeGLX bitcode/sample rate for clock alignment. The likelihood channel and bottom-camera `top_tongue` are not used.

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
x = ts[feat_idx, 0, :]
y = ts[feat_idx, 1, :]
```

iii. The notes claim velocity matches `findPosition`/`findVelocity`, but the mapping and implementation select a single tongue feature/view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Valid x/y samples are linearly interpolated onto the 5-ms grid. Gradients are taken per trial and NaN gradient values are set to zero; speed is Euclidean magnitude. No likelihood filtering, coordinate smoothing, true-time derivative, two-view normalization, or view combination is performed.

ii.
```python
xpos[:, trix] = f_x(time_axis)
xv = np.gradient(xpos[:, trix]); yv = np.gradient(ypos[:, trix])
xv[np.isnan(xv)] = 0; yv[np.isnan(yv)] = 0
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The agent believed interpolation plus `gradient()` matched the reference `findVelocity.m`; it also intentionally sets invisible tongue velocity to zero.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over all non-NaN samples in retained trials. Values below it are 0 and values at/above it are 1. NaNs are forced to 0; category 2 (“not visible”) is absent.

ii.
```python
tongue_thresh = np.nanpercentile(tongue_valid[~np.isnan(tongue_valid)], 50)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. The median split follows the prompt, but the agent’s notes/output values only define low/high and do not justify dropping the required visibility class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session offset is computed from the modes of behavior and electrophysiology bit starts. Frame times are shifted by that offset and the trial go cue, then linearly interpolated to the neural `time_axis`.

ii.
```python
vidshift = vidFileOffset - bitStart_mode
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], ...)
```

iii. The notes identify this as matching `findVideoOffset.m` and aligning all video streams to the neural grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses camera 1’s `top_paw` x/y coordinates, frame times, and dropped-frame status, plus the same clock fields used for tongue alignment. Likelihood is ignored.

ii.
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. The agent selected the bottom-view top paw as the relevant tracked feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Coordinates are linearly interpolated; missing interpolated positions are nearest-filled. Gradients are computed, median first differences are subtracted as baselines, remaining missing gradients are nearest-filled, and x/y magnitudes are combined.

ii.
```python
xpos[:, trix] = _fill_nearest(xpos[:, trix])
xv = np.gradient(xpos[:, trix]) - np.nanmedian(np.diff(xpos[:, trix]))
xv = _fill_nearest(xv)
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes say this follows `findVelocity.m`, including baseline derivative subtraction for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The retained session samples are split at their non-NaN median into 0/1; NaNs become 0 rather than the required class 2.

ii.
```python
paw_thresh = np.nanpercentile(paw_valid[~np.isnan(paw_valid)], 50)
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. The median is justified by the decoder specification, but only low/high labels are emitted.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Camera-1 frame times are offset-corrected, made relative to each trial’s go cue, and interpolated to the same `time_axis` as neural activity.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
xpos[:, trix] = f_x(time_axis)
```

iii. The notes use the common video-offset procedure for all kinematics.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from the separate session `motionEnergy_*.mat` trace, with camera-0 `frameTimes`; go cues and clock-offset metadata provide alignment.

ii.
```python
me_data = load_motion_energy(sess_info['me_filepath'])
cam_data = session['traj'][0]
```

iii. The loader handles bare, singly wrapped, and doubly wrapped motion-energy layouts, which the notes attribute to heterogeneous MATLAB files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already spatially reduced trace is linearly interpolated to the analysis grid and then nearest-filled wherever only part of a trial is missing. No additional smoothing is applied.

ii.
```python
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
                bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_me(time_axis)
me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. The agent says this matches `loadMotionEnergy.m` interpolation to the neural time axis.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session-wide median of retained non-NaN samples defines low=0 and high=1. Missing samples become 0, and sessions with no usable values use threshold zero. There is no “no video” class.

ii.
```python
me_thresh = np.nanpercentile(me_valid[~np.isnan(me_valid)], 50) if np.any(~np.isnan(me_valid)) else 0
me_disc[me >= me_thresh] = 1
me_disc[np.isnan(me)] = 0
```

iii. The median follows the task, but notes mention sessions with zero/missing motion-energy thresholds without correcting the missing third class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video shift and trial go cue; trace and time arrays are trimmed to their common length and interpolated onto the neural grid. A synthetic 400-Hz time base is used as fallback.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
n = min(len(trial_me), len(aligned_times))
me_aligned[:, trix] = f_me(time_axis)
```

iii. The notes describe common go-cue and video-offset alignment; fallback timing is defensive handling for absent frame times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Many loader fields use broad `try/except` fallbacks. Missing reward/bitStart become NaN; missing tracking becomes NaN arrays; absent motion energy becomes all NaN; non-tongue gaps and partial motion-energy gaps are nearest-filled; tongue NaNs and all missing output bins ultimately become class 0. Session exceptions are logged and skipped. Late all-zero neural trials are retained.

ii.
```python
except:
    bp['ev']['bitStart'] = np.full(bp['Ntrials'], np.nan)
...
arr[i] = arr[valid[np.argmin(dists)]]
...
tv_disc[np.isnan(tv)] = 0
```

iii. The notes frame these as robustness to format variants and missing fields, but acknowledge all-zero late trials and missing motion-energy sessions were left for the decoder to handle.

## 11-a. What are the most time-consuming steps of the code?

i. The code times loading, spike processing, tongue extraction, paw extraction, and motion energy separately. Its sample estimate was about eight seconds per session, and the largest computational work is nested spike binning/smoothing and per-trial video interpolation; full file loading is also substantial.

ii.
```python
t_load = time.time() - t0
...
t_spk = time.time() - t0
...
t_tongue = time.time() - t0
```

iii. The notes estimate roughly five minutes for the full conversion but do not provide an aggregate timing breakdown.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit-by-trial spike loop could be replaced by a 2-D histogram over trial and time. `_fill_nearest` performs a Python loop plus a distance search for every missing sample and could use indexed interpolation/distance transforms. Some trial-wise gradients and class assignment could also be batched once rectangular arrays exist.

ii.
```python
for i, u_idx in enumerate(valid_units):
    for j in range(Ntrials):
        spk_mask = trial_nums == trial_num
...
for i in range(len(arr)):
    if mask[i]: arr[i] = arr[valid[np.argmin(dists)]]
```

iii. The agent did not document these opportunities; it focused on correctness and estimated runtime as acceptable.

## 11-c. What processing does the code repeat multiple times?

i. It allocates/copies the identical time input for every trial; interpolates x and y separately for each feature/trial; performs analogous threshold/discretization three times; and repeatedly searches raw spike trial labels inside the nested trial loop. Feature extraction is separately repeated for tongue and paw, although the video offset is correctly computed once per session.

ii.
```python
inp = time_axis.astype(np.float32).reshape(1, -1)
...
f_x = interp1d(...); f_y = interp1d(...)
```

iii. No explicit justification is supplied; the notes only emphasize shared offset computation and matching named reference functions.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Loaders materialize unused fields (`tm`, sample/delay/reward events, likelihood, dropped-frame counts, motion-energy `moveThresh`), and process video/neural data for trials later discarded as early/stim/no-response. Optional plotting repeats discretization. `kept_units` is constructed only to test/count units and is not saved.

ii.
```python
unit['tm'] = unit_raw['tm'].flatten().astype(float)
bp['ev']['sample'] = ...
...
tongue_vel = extract_feature_velocity(...)
...
valid_trials = ~bp['early'] & ~bp['stim_enable'] & ~bp['no']
```

iii. The agent does not identify these as waste; it prioritizes a general loader and processes video before determining the final retained trial list.
