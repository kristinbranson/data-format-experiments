# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by globbing `/app/data/sub-*//*.nwb`, sorted by subject then experiment day. Each file is loaded with `pynwb.NWBHDF5IO`. A `ProcessPoolExecutor` (12 workers, spawn context) parallelizes per-session loading and conversion. The function `load_session` reads all behavior time series, raw fluorescence (`F`), neuropil (`Fneu`), and iscell flags from each NWB file.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
files.sort(key=lambda p: (p.split('/')[-2], int(p.split('ses-')[1][:2])))
# ...
with ProcessPoolExecutor(max_workers=nworkers, mp_context=ctx) as ex:
    for k, (neural, inp, out, planes, info) in enumerate(ex.map(_worker, jobs)):
```

```python
def load_session(path):
    io = NWBHDF5IO(path, 'r', load_namespaces=True)
    nwb = io.read()
    # ... reads behavior, F, Fneu, iscell, etc.
```

iii. The AI uses `glob` to find all NWB files and `pynwb` to load them, as instructed. The parallel processing with `spawn` context avoids OpenMP fork issues from suite2p.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from the NWB file's `nwb.subject.subject_id` field. The unique subjects list is built incrementally as sessions are processed.

ii.
```python
subject = nwb.subject.subject_id
# ...
sub = info['subject']
if sub not in subjects:
    subjects.append(sub)
subject_idx.append(subjects.index(sub))
```

iii. Using the NWB metadata field directly is reliable and matches the file naming convention (`sub-m11`, etc.).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The experiment day is read from `nwb.session_id`. Sessions are sorted by subject then experiment day.

ii.
```python
exp_day = int(nwb.session_id)
# ...
files.sort(key=lambda p: (p.split('/')[-2], int(p.split('ses-')[1][:2])))
```

iii. The one-file-per-session mapping is established by the DANDI data organization.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified from the `trial_start` and `teleport` behavior time series. Trial start indices are where `trial_start > 0`, trial end indices are where `teleport > 0`. Each trial spans `[trial_start_ind, teleport_ind)`, excluding the teleport sample.

ii.
```python
tstart_inds = np.where(beh['trial_start'] > 0)[0]
teleport_inds = np.where(beh['teleport'] > 0)[0]
assert len(tstart_inds) == len(teleport_inds)
assert np.all(teleport_inds > tstart_inds)
# ...
s, e = int(tstart_inds[i]), int(teleport_inds[i])
pos = beh['pos'][s:e]
```

iii. This matches the reference code's trial segmentation (`behavior.get_trial_types` uses `trial_start_inds[i]:teleport_inds[i]`). The teleport sample is excluded because its position value is a meaningless interpolation.

## 1-e. How are trials filtered based on quality controls?

i. Trials with lick-sensor error are dropped: if >30% of imaging frames in a trial have a cumulative lick count > 2, the trial is excluded. Additionally, trials with any NaN in neural, position, speed, or lick data are dropped. 81 trials are dropped for lick error across the full dataset, matching the paper exactly.

ii.
```python
LICK_ERROR_THRESH = 0.3
# ...
L = beh['lick'][s:e]
lick_error[i] = np.mean(L > 2) > LICK_ERROR_THRESH
# ...
if lick_error[i]:
    n_dropped_lick += 1
    continue
if (not np.all(np.isfinite(neural)) or not np.all(np.isfinite(pos))
        or not np.all(np.isfinite(speed)) or not np.all(np.isfinite(lick))):
    n_dropped_nan += 1
    continue
```

iii. From CONVERSION_NOTES.md: "Trials with lick-sensor error are dropped (81 trials), rather than NaN-ed as in the paper, because lick is a decoder output and NaN is not a permitted value." The paper states: "~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p traces: `Fluorescence` (raw F) and `Neuropil` (Fneu) from the NWB ophys processing module. The NWB `Deconvolved` series is NOT used, as it is suite2p's own deconvolution, not the paper's per-trial dF/F-based events.

ii.
```python
F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
Fneu_list.append(np.asarray(ophys['Neuropil'][key].data[:nframes, :])[:, mask].T.astype(np.float32))
```

iii. From CONVERSION_NOTES: "The NWB `Deconvolved` series is suite2p's own `spks` (non-zero before the first trial start, i.e. computed over the whole session from raw F), NOT the paper's per-trial dFF-based `events`."

## 2-b. How is the `neural` data processed?

i. dF/F is computed following the paper's `preprocessing.dff`: (1) neuropil subtraction (`F - 0.7 * Fneu`), (2) per-trial add-back of the neuropil mean, (3) per-trial maximin baseline (Gaussian smoothing with sigma=15, then 300-sample minimum filter, then 300-sample maximum filter), (4) dF/F = (F - baseline) / |baseline|, (5) Gaussian smoothing with sigma=2 samples. The default neural signal written to the pickle is **dF/F**, not the OASIS-deconvolved events (though events can be selected via `--neural-signal events`).

ii.
```python
def compute_dff_events(F, Fneu, windows, fs, deconvolve=True):
    f_ -= NEU_COEF * fneu_
    # ...
    f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
    x = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH_SIG])
    x = ndimage.minimum_filter1d(x, MAXIMIN_WIN, axis=-1)
    flow[:, s:e] = ndimage.maximum_filter1d(x, MAXIMIN_WIN, axis=-1)
    # ...
    dff[:, nanmask] = ((f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask]))
    # ...
    dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIG, axis=1)
```

iii. From CONVERSION_NOTES Step 5 Decision 2: "dF/F was chosen because (a) the paper treats it as the signal 'closest to the raw data' and uses it for spatial-peak, field and sequence analyses; (b) OASIS deconvolution is explicitly not interpreted as a spike rate by the authors... (c) it decodes better here — validation balanced accuracy on the 2-session sample was higher for every one of the six outputs."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) suite2p `iscell` manual curation from the NWB PlaneSegmentation (cells with `iscell[:,0] > 0`), and (2) putative interneuron exclusion: cells with Pearson r(dF/F, running speed) > 0.5 are dropped. The speed correlation is computed in a vectorized manner.

ii.
```python
iscell = np.asarray(seg['iscell'].data)[:, 0] > 0
# ...
r_speed = speed_correlation(dff, beh['speed'], nanmask)
is_int = np.nan_to_num(r_speed, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
neural_full = (events if neural_signal == 'events' else dff)[keep_cells]
```

iii. Matches the paper's Methods: "Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 +/- 0.85% of cells."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The data is aligned to trial start. Since the VR behavior data has already been interpolated onto imaging frame times by the authors' `vr_align_to_2P`, neural and behavioral data are sample-for-sample aligned with no further resampling needed. Each trial's neural data is simply sliced from `[trial_start_ind, teleport_ind)`.

ii.
```python
neural = neural_full[:, s:e]
```

iii. No temporal shifting is needed because alignment to trial start means offset = 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one imaging frame = 64.484 ms (1000/15.5078125 Hz). No rebinning is applied. For two-plane sessions (m17, m18), the scanner rate is 31 Hz but the per-plane rate is 15.5 Hz, so the effective time bin is the same.

ii.
```python
fs = sess['rate'] / sess['n_planes']  # per-plane imaging rate (Hz)
# ...
'time_bin_size': 1000.0 / 15.5078125,  # ms per imaging frame (64.484 ms)
```

iii. From CONVERSION_NOTES Step 5 Decision 1: "Temporal bins = imaging frames (64.484 ms). The VR behaviour in the NWB file has already been interpolated onto imaging frame times by the authors' vr_align_to_2P, so neural and behavioural streams are sample-for-sample aligned with no further resampling."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior time series timestamps (frame times), which are the same for all behavior series.

ii.
```python
frame_times = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
# ...
t = beh['time'][s:e] - beh['time'][s]
```

iii. All behavior time series share the same timestamps (one per imaging frame).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from each frame timestamp within the trial.

ii.
```python
t = beh['time'][s:e] - beh['time'][s]
```

iii. Simple subtraction to get elapsed time from trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Already aligned — both neural and behavior data use the same frame indexing. No additional alignment needed.

ii. Both are indexed by `[s:e]` where `s, e = tstart_inds[i], teleport_inds[i]`.

iii. The VR behavior was interpolated onto imaging frame times by the authors.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` (a.k.a. `morph`) behavior time series.

ii.
```python
behavior['morph'] = get('environment')
# ...
m = np.unique(beh['morph'][s:e])
morph[i] = int(np.round(m[0]))
```

iii. The `environment` variable is 0 for ENV1 and 1 for ENV2, constant within each trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique value of `morph` within the trial window is extracted and rounded to an integer. It is broadcast as a constant across all timepoints in the trial.

ii.
```python
morph[i] = int(np.round(m[0]))
# ...
np.full(len(pos), float(morph[i])),
```

iii. The rounding handles any floating-point imprecision in the stored values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the NWB `trial number` behavior time series.

ii.
```python
behavior['trialnum'] = get('trial number')
# ...
tn = np.unique(beh['trialnum'][s:e])
trialnum[i] = int(tn[0])
```

iii. The `trial number` field provides a 0-indexed trial counter within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The unique value of `trial number` within the trial window is extracted and cast to int. It is broadcast as a constant across all timepoints.

ii.
```python
trialnum[i] = int(tn[0])
# ...
np.full(len(pos), float(trialnum[i])),
```

iii. The value is taken directly from the NWB field with no additional processing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from two variables: the `Reward` time series (sparse, one timestamp per delivered reward) and the `reward_zone` behavior time series. A trial is considered rewarded if both `reward > 0` and `reward_zone > 0` occur within the trial window, following the reference code's `get_trial_types`.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
reward_frames = np.searchsorted(frame_times, reward_times)
reward = np.zeros(nframes)
np.add.at(reward, reward_frames, 1.0)
# ...
isreward[i] = int(np.any(beh['reward'][s:e] > 0) and np.any(beh['rzone'][s:e] > 0))
```

iii. The reference code `get_trial_types` defines `isreward = any(reward>0) AND any(rzone>0)`. The `rzone` check distinguishes genuine reward delivery from artifacts.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The previous trial outcome is the `isreward` value of the previous trial (shifted by one). For the first trial of a session, the value is 0 (no preceding trial). The value is constant across all timepoints in the trial.

ii.
```python
prev_outcome = np.concatenate([[0], isreward[:-1]]).astype(np.int64)
# ...
np.full(len(pos), float(prev_outcome[i])),
```

iii. The shift-by-one is computed before trial filtering, so dropped trials still contribute their outcome to the next trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone coordinates. Reward zone identity is determined from the **scene name** in the NWB `identifier` field, using `scene_reward_zones()` which ports the reference code's `behavior.get_reward_zones`. On switch days, the zone changes after trial 30 (`CHANGE_TRIAL=30`).

ii.
```python
scene = nwb.identifier.rstrip('/').split('/')[-1]
# ...
rz_coords, rz_labels = scene_reward_zones(sess['scene'], n_trials_raw)
# ...
rz_start, rz_end = rz_coords[i]
d = np.zeros_like(pos)
d[pos < rz_start] = pos[pos < rz_start] - rz_start
d[pos > rz_end] = pos[pos > rz_end] - rz_end
```

iii. From CONVERSION_NOTES Decision 6: "Reward-zone identity from the scene name, as the reference does, rather than from the reward_zone flag — the flag only fires on rewarded trials (~85%), so it cannot label omission trials."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the active reward zone is computed. Distance is 0 when inside the zone, negative when before it, and positive when past it.

ii.
```python
d = np.zeros_like(pos)
d[pos < rz_start] = pos[pos < rz_start] - rz_start
d[pos > rz_end] = pos[pos > rz_end] - rz_end
```

iii. Matches the paper's concept of signed distance to the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using a custom function `discretize_reward_distance` with explicit conditional assignments:
- 0: d < -50 cm
- 1: -50 <= d < -10 cm
- 2: -10 <= d < 0 cm
- 3: d == 0 (inside the zone)
- 4: 0 < d <= 10 cm
- 5: 10 < d <= 50 cm
- 6: d > 50 cm

ii.
```python
def discretize_reward_distance(d):
    out = np.empty(d.shape, dtype=np.int64)
    out[:] = 3                                  # d == 0: inside the reward zone
    out[(d >= -10) & (d < 0)] = 2
    out[(d >= -50) & (d < -10)] = 1
    out[d < -50] = 0
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. Matches the instructions' bin specification. Uses explicit conditions rather than `np.digitize` for clarity.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The position data and neural data share the same frame indices within each trial, so no additional alignment is needed.

ii. Both use `[s:e]` slicing where `s, e = tstart_inds[i], teleport_inds[i]`.

iii. Inherent alignment from the shared frame-based indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = beh['pos'][s:e]
```

iii. The `position` variable directly records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
discretize_position(pos)
```

iii. The raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins spanning the 450 cm track (90 cm per bin) using integer division and clipping:
- 0: < 90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: >= 360 cm

ii.
```python
def discretize_position(pos):
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. Division by 90 and integer truncation is equivalent to the bin edges [0, 90, 180, 270, 360, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data — no additional alignment needed.

ii. Both indexed by `[s:e]`.

iii. Inherent alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = beh['lick'][s:e]
```

iii. The `lick` variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
(lick > 0).astype(np.int64),
```

iii. The instructions specify binary output (no/yes). The raw lick values can be >1 (cumulative counts), so thresholding at >0 converts to binary, consistent with the reference code's `lickrate` function which sets `licks[licks>0]=1`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data — no additional alignment needed.

ii. Both indexed by `[s:e]`.

iii. Inherent alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the **scene name** in the NWB `identifier` field, parsed by `scene_reward_zones()`. Scene names like `Env1_LocationA_to_C` encode the reward zone identity and any within-session switches.

ii.
```python
scene = nwb.identifier.rstrip('/').split('/')[-1]
# ...
def scene_reward_zones(scene, n_trials):
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

iii. This ports the reference code's `behavior.get_reward_zones`, which maps scene names to zone labels. The reward zone changes after trial 30 on switch days (`CHANGE_TRIAL=30`), matching the paper.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone label (A/B/C) is mapped to an integer (0/1/2). The value is constant within each trial but broadcast over time.

ii.
```python
np.full(len(pos), RZ_LABELS.index(rz_labels[i]), dtype=np.int64),
```

iii. Direct mapping: A=0, B=1, C=2.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` time series (sparse timestamps) AND the `reward_zone` behavior time series. A trial is rewarded iff both a reward event AND a non-zero reward_zone flag occur within the trial.

ii.
```python
isreward[i] = int(np.any(beh['reward'][s:e] > 0) and np.any(beh['rzone'][s:e] > 0))
# ...
np.full(len(pos), isreward[i], dtype=np.int64),
```

iii. Matches the reference code's `get_trial_types`: `isreward = any(reward>0) AND any(rzone>0)`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary value (0 = omitted, 1 = rewarded) computed per trial and broadcast over all timepoints. The dual condition (reward AND reward_zone) ensures that only genuine reward deliveries in the reward zone are counted.

ii.
```python
isreward[i] = int(np.any(beh['reward'][s:e] > 0) and np.any(beh['rzone'][s:e] > 0))
```

iii. The `rzone` check prevents false positives from reward events outside the reward zone.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior frame mismatch**: Multi-plane sessions (m17/m18) can have one extra imaging frame; ophys data is truncated to the behavior length (`nframes`).
- **Lick-sensor errors**: Trials with >30% of frames having cumulative lick count > 2 are dropped entirely (81 trials).
- **NaN/Inf neural data**: Trials where any neural, position, speed, or lick value is not finite are dropped.
- **Reward timestamp alignment**: Reward timestamps are mapped to the nearest frame using `searchsorted` and clipped to valid indices.

ii.
```python
F_list.append(np.asarray(rrs.data[:nframes, :])[:, mask].T.astype(np.float32))
# truncates to nframes (behavior length)
# ...
if lick_error[i]:
    n_dropped_lick += 1
    continue
if (not np.all(np.isfinite(neural)) or ...):
    n_dropped_nan += 1
    continue
# ...
reward_frames = np.clip(reward_frames, 0, nframes - 1)
```

iii. From CONVERSION_NOTES: the one-frame correction is documented in the reference code's `vr_align_to_2P`. Lick error handling matches the paper's count of 81 trials.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **dF/F computation** — the per-trial maximin baseline and smoothing operations on large arrays (reported as `t_dff` in output, up to ~7s per session for m18).
2. **NWB file loading** — I/O bound, reading large fluorescence/neuropil arrays (reported as `t_load`, up to ~2s).
3. **Pickling** — serializing the full dataset.

ii. From conversion output: typical session takes 1-10s total, with dF/F dominating.

iii. Parallel processing (12 workers) brings total conversion time for 152 sessions down to ~31 seconds.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the speed correlation computation (matrix product instead of per-cell loop). The per-trial loop for building input/output arrays could theoretically be vectorized, but variable trial lengths make this impractical.

ii.
```python
def speed_correlation(dff, speed, nanmask):
    # Vectorised equivalent
    Dm = D - D.mean(axis=1, keepdims=True)
    Sm = S - S.mean()
    r = (Dm @ Sm) / denom
```

iii. Most operations are already vectorized at the per-trial level.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session in a single pass — no separate survey step. Each NWB file is read exactly once. This is more efficient than approaches that read data twice (once to survey, once to convert).

ii. N/A (single-pass design).

iii. The parallel `ProcessPoolExecutor` ensures each file is processed exactly once.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. When `--neural-signal dff` (the default), the OASIS deconvolution is still computed if `--show-processing` is enabled (for diagnostic plots), but the events are not used in the final output. Without `--show-processing`, deconvolution is skipped entirely when dF/F is selected, which is an optimization.

ii.
```python
dff, events, nanmask = compute_dff_events(sess['F'], sess['Fneu'], windows, fs,
                                          deconvolve=(neural_signal == 'events'
                                                      or show_processing))
```

iii. The conditional deconvolution avoids unnecessary computation in the default case.
