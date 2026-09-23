# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file under `/app/data/sub-*/*.nwb`, sorts them by subject and session number, and reads each session with `pynwb.NWBHDF5IO`. All downstream subject/session/trial structure is built from those files.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
files.sort(key=lambda p: (p.split('/')[-2], int(p.split('ses-')[1][:2])))
```
```python
io = NWBHDF5IO(path, 'r', load_namespaces=True)
nwb = io.read()
```

iii. In `CONVERSION_NOTES.md`, the AI says the archive is organized as one NWB file per mouse-day and explicitly notes that these should be read with `pynwb`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from each NWB file's `nwb.subject.subject_id`, and the final `subjects` list is built from the unique subject IDs encountered while iterating over sessions.

ii.
```python
subject = nwb.subject.subject_id
...
subjects, subject_idx, session_info = [], [], []
...
sub = info['subject']
if sub not in subjects:
    subjects.append(sub)
subject_idx.append(subjects.index(sub))
```

iii. The notes state that `nwb.subject.subject_id` holds the released mouse IDs (`m3`, `m4`, `m11`, etc.), so the AI used the NWB metadata rather than parsing only directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The AI derives the experiment day from `nwb.session_id` and preserves one converted session per input file.

ii.
```python
exp_day = int(nwb.session_id)
...
for k, (neural, inp, out, planes, info) in enumerate(ex.map(_worker, jobs)):
    ...
    data['neural'].append(neural)
    data['input'].append(inp)
    data['output'].append(out)
```

iii. `CONVERSION_NOTES.md` says the dataset is `/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, one file per mouse-day, so the session split follows the archive structure directly.

## 1-d. How are the data split into trials?

i. Trials are defined from the per-frame behavioral event series: trial starts are frames where `trial_start > 0`, and trial ends are frames where `teleport > 0`. Each emitted trial uses the half-open slice `[trial_start, teleport)`, excluding the teleport frame.

ii.
```python
tstart_inds = np.where(beh['trial_start'] > 0)[0]
teleport_inds = np.where(beh['teleport'] > 0)[0]
...
for i in range(n_trials_raw):
    s, e = int(tstart_inds[i]), int(teleport_inds[i])
    pos = beh['pos'][s:e]
    ...
    neural = neural_full[:, s:e]
```

iii. The notes justify this as matching the reference code's lap definition and explicitly say the teleport sample is excluded because its interpolated position is not meaningful.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials flagged as lick-sensor-error trials and also drops any trial containing non-finite neural or behavioral values. It does not apply a minimum-length filter because its exploration found the shortest kept trial was already long enough.

ii.
```python
lick_error = np.zeros(n_trials_raw, dtype=bool)
...
L = beh['lick'][s:e]
lick_error[i] = np.mean(L > 2) > LICK_ERROR_THRESH
```
```python
for i in range(n_trials_raw):
    if lick_error[i]:
        n_dropped_lick += 1
        continue
    ...
    if (not np.all(np.isfinite(neural)) or not np.all(np.isfinite(pos))
            or not np.all(np.isfinite(speed)) or not np.all(np.isfinite(lick))):
        n_dropped_nan += 1
        continue
```

iii. The notes say this follows the paper's lick-sensor QC (`>30%` of frames with cumulative lick count `>2`) and that such trials must be dropped here because `lick` is a decoder output and cannot be NaN.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from raw suite2p fluorescence and neuropil traces stored in the NWB `Fluorescence` and `Neuropil` groups, after restricting to `iscell` ROIs from `PlaneSegmentation`.

ii.
```python
seg = ophys['ImageSegmentation']['PlaneSegmentation']
iscell = np.asarray(seg['iscell'].data)[:, 0] > 0
plane_idx = np.asarray(seg['planeIdx'].data).astype(int)
...
rrs = ophys['Fluorescence'][key]
F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
Fneu_list.append(np.asarray(ophys['Neuropil'][key].data[:nframes, :])[:, mask].T
                 .astype(np.float32))
```

iii. The notes explicitly reject the NWB `Deconvolved` series as the wrong signal and say the paper's analyses recompute activity from raw `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The AI recomputes dF/F with the paper's maximin-baseline pipeline, optionally computes OASIS events, then writes dF/F by default to the pickle. The pipeline is: neuropil subtraction (`0.7`), add back mean neuropil within each baseline window, Gaussian smoothing (`sigma=15`), 300-sample min then max filters, `(F - baseline)/abs(baseline)`, then Gaussian smoothing (`sigma=2`).

ii.
```python
f_ -= NEU_COEF * fneu_
...
f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
x = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH_SIG])
x = ndimage.minimum_filter1d(x, MAXIMIN_WIN, axis=-1)
flow[:, s:e] = ndimage.maximum_filter1d(x, MAXIMIN_WIN, axis=-1)
...
dff[:, nanmask] = ((f_[:, nanmask] - flow[:, nanmask])
                   / np.abs(flow[:, nanmask]))
...
dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIG, axis=1)
if deconvolve:
    events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]),
                                OASIS_BATCH, TAU, fs)
```
```python
neural_full = (events if neural_signal == 'events' else dff)[keep_cells]
```

iii. In Step 5 of the notes, the AI says it followed the paper's dF/F code exactly but chose dF/F rather than events as the default emitted neural signal because it viewed dF/F as closer to raw data and observed better decoder accuracy on its sample run.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, the AI keeps only ROIs marked `iscell`. Then it removes putative interneurons using the paper's speed-correlation rule: cells with Pearson correlation `r(dF/F, speed) > 0.5`.

ii.
```python
iscell = np.asarray(seg['iscell'].data)[:, 0] > 0
...
r_speed = speed_correlation(dff, beh['speed'], nanmask)
is_int = np.nan_to_num(r_speed, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
```

iii. The notes describe this as matching the paper's two-stage neuron curation: suite2p manual curation plus putative interneuron exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The converted trial neural arrays are aligned to trial start simply by slicing neural frames on the same `[trial_start, teleport)` intervals used for the behavioral data. Time zero is therefore the first frame after `trial_start`.

ii.
```python
for i in range(n_trials_raw):
    s, e = int(tstart_inds[i]), int(teleport_inds[i])
    ...
    neural = neural_full[:, s:e]
```

iii. The notes say the decoder task requires trial-start alignment and that using the native imaging-frame time base avoids any extra temporal offset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging frame rate: `fs = rate / n_planes`, which corresponds to `1000 / 15.5078125 = 64.484 ms` per bin. No additional temporal binning or resampling is applied.

ii.
```python
fs = sess['rate'] / sess['n_planes']
...
'time_bin_size': 1000.0 / 15.5078125,
'sampling_rate_hz': 15.5078125,
```

iii. The notes state that the NWB behavior streams are already aligned to imaging frames, so further binning would only blur outputs like licking and speed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI derives this input from the per-frame behavior timestamps attached to the `position` time series.

ii.
```python
frame_times = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
...
behavior = dict(
    time=frame_times,
    pos=get('position'),
```

iii. The notes say the NWB behavior table is already one sample per imaging frame, so any frame-aligned behavior timestamp stream would work; the AI used the position timestamps as its session time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the trial's first timestamp from every timestamp in that trial so the input starts at zero and increases in seconds.

ii.
```python
t = beh['time'][s:e] - beh['time'][s]
...
inp = np.stack([
    t,
    np.full(len(pos), float(morph[i])),
    np.full(len(pos), float(trialnum[i])),
    np.full(len(pos), float(prev_outcome[i])),
], axis=0).astype(np.float32)
```

iii. The notes explicitly describe this as "time since trial start (s)" and show it as a sawtooth resetting at each trial boundary.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: both timestamps and neural data live on the same imaging-frame index, and both are sliced with the same `[trial_start, teleport)` frame bounds. The AI also truncates ophys to behavior length when one extra imaging frame exists.

ii.
```python
frame_times = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
nframes = len(frame_times)
...
F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
```
```python
t = beh['time'][s:e] - beh['time'][s]
neural = neural_full[:, s:e]
```

iii. The notes highlight that the NWB files already contain the output of the authors' `vr_align_to_2P`, so behavior and neural streams are sample-for-sample aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the per-frame `environment` behavior time series, stored by the AI as `morph`.

ii.
```python
behavior = dict(
    ...
    morph=get('environment'),
```

iii. The notes map this directly to the paper's `morph` / ENV1-vs-ENV2 variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the unique environment value within each trial, rounds it to an integer, and broadcasts that constant value across all frames in the emitted trial input array.

ii.
```python
m = np.unique(beh['morph'][s:e])
morph[i] = int(np.round(m[0]))
...
np.full(len(pos), float(morph[i])),
```

iii. The notes say environment is constant within a trial and should be emitted as a per-trial decoder input on the shared trial time base.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI derives trial number from the NWB `trial number` behavior series, taking its unique value within each trial and then broadcasting it across that trial.

ii.
```python
behavior = dict(
    ...
    trialnum=get('trial number'),
```
```python
tn = np.unique(beh['trialnum'][s:e])
trialnum[i] = int(tn[0])
```

iii. The notes list `behavior/trial number` as the source for the decoder input and say it is constant within every trial.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI extracts the unique within-trial `trial number` value, converts it to an integer, and repeats it over all timepoints in the trial input matrix.

ii.
```python
tn = np.unique(beh['trialnum'][s:e])
trialnum[i] = int(tn[0])
...
np.full(len(pos), float(trialnum[i])),
```

iii. The notes justify this as the native per-trial session index already present in the aligned behavior table.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The AI derives previous trial outcome from the sparse `Reward` timestamps converted to a per-frame reward vector, together with the per-frame `reward_zone` flag. It defines trial reward as `any(reward) and any(rzone)` on the previous trial.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
reward_frames = np.searchsorted(frame_times, reward_times)
reward = np.zeros(nframes)
np.add.at(reward, reward_frames, 1.0)
behavior['reward'] = reward
```
```python
isreward[i] = int(np.any(beh['reward'][s:e] > 0)
                  and np.any(beh['rzone'][s:e] > 0))
...
prev_outcome = np.concatenate([[0], isreward[:-1]]).astype(np.int64)
```

iii. The notes say this follows the reference paper's `get_trial_types` logic rather than using reward timestamps alone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a per-trial binary outcome for each trial, then shifts that vector by one trial so each trial receives the previous trial's reward outcome. The first trial gets `0`.

ii.
```python
isreward = np.zeros(n_trials_raw, dtype=np.int64)
...
isreward[i] = int(np.any(beh['reward'][s:e] > 0)
                  and np.any(beh['rzone'][s:e] > 0))
...
prev_outcome = np.concatenate([[0], isreward[:-1]]).astype(np.int64)
```
```python
np.full(len(pos), float(prev_outcome[i])),
```

iii. In Step 5, the AI explicitly documents "previous trial outcome = 0 for the first trial" and says dropping first trials would discard valid neural data unnecessarily.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the per-frame `position` series and the trial's active reward-zone coordinates inferred from the session `scene` name and trial index.

ii.
```python
scene = nwb.identifier.rstrip('/').split('/')[-1]
...
rz_coords, rz_labels = scene_reward_zones(sess['scene'], n_trials_raw)
...
pos = beh['pos'][s:e]
rz_start, rz_end = rz_coords[i]
```

iii. The notes say the reward-zone identity should come from the scene name, as in the reference code, because the `reward_zone` flag only appears on rewarded trials and cannot label omission trials by itself.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed distance to the nearest point of the active reward zone: `0` inside the zone, negative before the zone, positive after it, then discretizes that continuous distance.

ii.
```python
d = np.zeros_like(pos)
d[pos < rz_start] = pos[pos < rz_start] - rz_start
d[pos > rz_end] = pos[pos > rz_end] - rz_end
```
```python
def discretize_reward_distance(d):
    out = np.empty(d.shape, dtype=np.int64)
    out[:] = 3
    out[(d >= -10) & (d < 0)] = 2
    out[(d >= -50) & (d < -10)] = 1
    out[d < -50] = 0
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. The notes say this matches the decoder-task definition and the paper's reward-zone geometry.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses seven bins with thresholds equivalent to `[-inf, -50, -10, 0, 0+, 10, 50, inf]`, implemented with explicit comparisons so exact zero is its own category.

ii.
```python
out[:] = 3
out[(d >= -10) & (d < 0)] = 2
out[(d >= -50) & (d < -10)] = 1
out[d < -50] = 0
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The notes say the bins were taken directly from the decoder task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data are both sliced on the same frame indices for each trial, so distance-to-reward-zone is frame-aligned with neural activity.

ii.
```python
s, e = int(tstart_inds[i]), int(teleport_inds[i])
pos = beh['pos'][s:e]
...
neural = neural_full[:, s:e]
```

iii. The notes repeatedly emphasize that the behavior table is already aligned to imaging frames.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the per-frame `position` behavior series.

ii.
```python
behavior = dict(
    ...
    pos=get('position'),
```

iii. The notes describe `position` as current position on the 450 cm VR track in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI takes the raw within-trial position samples and discretizes them into five equal-width bins over the 450 cm track.

ii.
```python
def discretize_position(pos):
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. The notes say the 450 cm track length comes from the paper and the decoder task asked for five equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are effectively `<90`, `90-180`, `180-270`, `270-360`, and `>360` cm, implemented as floor-division by 90 cm and clipping to `[0, 4]`.

ii.
```python
TRACK_LENGTH = 450.0
...
return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. The notes justify clipping because rare samples can fall slightly outside the nominal `[0, 450]` range.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by using the same per-trial frame slice as the neural data.

ii.
```python
pos = beh['pos'][s:e]
...
neural = neural_full[:, s:e]
```

iii. The notes say neural and behavioral samples share the imaging-frame time base.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the per-frame `lick` behavior series.

ii.
```python
behavior = dict(
    ...
    lick=get('lick'),
```

iii. The notes describe this NWB series as cumulative lick count per imaging frame after the authors' frame alignment.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes lick counts so any positive lick count becomes `1` and zero stays `0`.

ii.
```python
(lick > 0).astype(np.int64)
```

iii. The notes say this matches the binary decoder-output requirement and mirrors the reference code's lick binarization.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the same trial frame window as the neural data.

ii.
```python
lick = beh['lick'][s:e]
...
neural = neural_full[:, s:e]
```

iii. The notes say no extra interpolation is needed because licks are already on imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the session `scene` name in the NWB identifier plus the within-session trial index, not from the `reward_zone` flag alone.

ii.
```python
scene = nwb.identifier.rstrip('/').split('/')[-1]
...
rz_coords, rz_labels = scene_reward_zones(sess['scene'], n_trials_raw)
```

iii. The notes say the AI copied the reference code's scene-based reward-zone mapping because it remains defined on omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses fixed and switching scene names, assigns zone labels `A/B/C`, and on switch sessions changes the active zone after trial `30`. The per-trial label is then broadcast across all frames of that trial.

ii.
```python
if '_to_' in scene:
    before, after = scene.split('_to_')
    first = before[-1]
    second = after[-1]
    labels = np.array([first] * min(CHANGE_TRIAL, n_trials) +
                      [second] * max(0, n_trials - CHANGE_TRIAL))
else:
    lab = scene[-1]
    labels = np.array([lab] * n_trials)
```
```python
np.full(len(pos), RZ_LABELS.index(rz_labels[i]), dtype=np.int64),
```

iii. The notes explicitly cite the paper's "switch after 30 trials" rule and document validation against the `reward_zone` flag positions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the sparse `Reward` timestamps converted to a per-frame reward vector, together with the `reward_zone` flag. A trial is rewarded only if both occur within that trial.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
reward_frames = np.searchsorted(frame_times, reward_times)
...
behavior['reward'] = reward
```
```python
isreward[i] = int(np.any(beh['reward'][s:e] > 0)
                  and np.any(beh['rzone'][s:e] > 0))
```

iii. The notes say this matches the paper's `get_trial_types` definition and avoids treating sparse reward timestamps alone as the full trial-outcome rule.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI first computes a per-trial `isreward` scalar, then broadcasts that scalar across every frame in the trial's output array.

ii.
```python
isreward = np.zeros(n_trials_raw, dtype=np.int64)
...
isreward[i] = int(np.any(beh['reward'][s:e] > 0)
                  and np.any(beh['rzone'][s:e] > 0))
```
```python
np.full(len(pos), isreward[i], dtype=np.int64),
```

iii. The notes say all outputs were emitted as time-varying when possible, so per-trial outputs were broadcast along the trial time axis.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI explicitly handles small archive inconsistencies: it truncates ophys to behavior length when one extra imaging frame exists, drops trials with lick-sensor errors, drops any trial with non-finite values, clips position-bin assignments for slight out-of-range samples, and guards a possible `keep_teleports` underflow case with `min(...)`.

ii.
```python
F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
```
```python
if lick_error[i]:
    n_dropped_lick += 1
    continue
...
if (not np.all(np.isfinite(neural)) or not np.all(np.isfinite(pos))
        or not np.all(np.isfinite(speed)) or not np.all(np.isfinite(lick))):
    n_dropped_nan += 1
    continue
```
```python
return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. The notes document each of these as dataset-specific edge cases discovered during exploration and validation, especially the one-frame ophys/behavior mismatch in some multi-plane sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The AI identifies NWB I/O, dF/F computation, and pickling the large final dataset as the main cost centers; these dominate wall-clock time, not the per-trial array assembly.

ii.
```python
t0 = time.time()
sess = load_session(path)
t_load = time.time() - t0
...
t1 = time.time()
dff, events, nanmask = compute_dff_events(...)
t_dff = time.time() - t1
```
```python
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 7 of the notes reports measured timings and specifically calls out NWB reads, dF/F, and pickle writing as the expensive steps.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the expensive speed-correlation loop. The remaining obviously serial loops are the per-window dF/F loops and the per-trial assembly loop, which still iterate in Python over variable-length trial windows.

ii.
```python
for s, e in windows:
    f_[:, s:e] = F[:, s:e]
    ...
for s, e in windows:
    ...
    x = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH_SIG])
```
```python
for i in range(n_trials_raw):
    ...
    neural_trials.append(np.ascontiguousarray(neural, dtype=np.float32))
    input_trials.append(inp)
    output_trials.append(out)
```

iii. The notes explicitly say the naive per-cell speed-correlation loop was a bottleneck and that the AI replaced it with one matrix product, leaving the variable-length trial loops as the main non-vectorized structure.

## 13-c. What processing does the code repeat multiple times?

i. The AI avoids the reference solution's separate survey pass, so repeated work is limited. The main repeated processing is multiple passes over the same trial boundaries: one pass to compute trial-level metadata (`isreward`, `morph`, `trialnum`, `lick_error`) and another pass to build the emitted per-trial arrays.

ii.
```python
for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
    isreward[i] = int(np.any(beh['reward'][s:e] > 0)
                      and np.any(beh['rzone'][s:e] > 0))
    ...
    lick_error[i] = np.mean(L > 2) > LICK_ERROR_THRESH
```
```python
for i in range(n_trials_raw):
    ...
    out = np.stack([...], axis=0)
    inp = np.stack([...], axis=0).astype(np.float32)
```

iii. The notes describe the AI script as a single-pass session converter with optional diagnostics, so the repeated work is mostly local per-session re-scanning rather than whole-dataset reloading.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded processing is optional diagnostic work: when `--show-processing` is enabled while writing dF/F, the code still computes OASIS events only for plots, and the plotting path recomputes/concatenates several derived traces that are not saved in the dataset. The script also preserves per-neuron plane indices in metadata even though `brain_region_idx` collapses all cells to CA1.

ii.
```python
dff, events, nanmask = compute_dff_events(sess['F'], sess['Fneu'], windows, fs,
                                          deconvolve=(neural_signal == 'events'
                                                      or show_processing))
```
```python
cat_out = np.concatenate(output_trials[:ntr_plot], axis=1)
cat_in = np.concatenate(input_trials[:ntr_plot], axis=1)
cat_neural = np.concatenate(neural_trials[:ntr_plot], axis=1)
```

iii. The notes say OASIS is skipped entirely when the emitted neural signal is dF/F unless diagnostics are requested, which implies the diagnostic path is the main intentionally extra computation.
