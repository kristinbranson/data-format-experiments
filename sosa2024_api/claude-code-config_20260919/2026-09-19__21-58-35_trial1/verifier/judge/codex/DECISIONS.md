# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every NWB file one directory below `/app/data`, sorted the paths, and processed every file (152 for the full run), optionally in eight worker processes. Each worker opened its file with `pynwb.NWBHDF5IO`, read fluorescence, neuropil, ROI metadata, and behavior, and the driver assembled all retained sessions into the pickle.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, "sub-*", "*.nwb")))
with NWBHDF5IO(path, "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes say there are 152 NWB files in 11 subject directories and emphasize that `pynwb`, not `h5py`, is used. Parallel session processing was chosen to reduce the roughly one-minute full-run time.

## 1-b. How are the data split into subjects?

i. Subject identity is read from `nwb.subject.subject_id`. During assembly, a subject is appended on first encounter and each session receives an index into that list.

ii.
```python
subject = nwb.subject.subject_id
if info["subject"] not in subjects:
    subjects.append(info["subject"])
subject_idx.append(subjects.index(info["subject"]))
```

iii. The notes report 11 unique mice and use NWB metadata as the authoritative identity rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. `process_session(path)` returns one session’s trial lists; sessions with fewer than two retained trials would be dropped at assembly.

ii.
```python
for r in ex.map(_worker, jobs):
    results.append(r)
...
if len(n) < 2:
    continue
neural.append(n)
```

iii. This follows the file organization and the target format’s minimum-two-trials requirement. In practice all 152 sessions were retained.

## 1-d. How are the data split into trials?

i. Starts are samples where `trial_start > 0`; ends are samples where `teleport > 0`. A trial is the half-open slice `[start, end)`, applied identically to neural and behavior.

ii.
```python
starts = np.where(beh["trial_start"] > 0)[0]
ends = np.where(beh["teleport"] > 0)[0]
...
neural_trials.append(dff[:, s:e])
```

iii. The notes define a trial as one lap and justify common `[start, stop)` boundaries to preserve exact neural/behavior alignment, noting a one-frame difference from one reference helper.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops trials meeting the paper’s lick-sensor-failure rule, overlapping `scanning < 0`, or shorter than two frames. A session with fewer than two survivors is dropped.

ii.
```python
trial_lick_error[i] = (np.mean(lick[s:e] > 2) > 0.3)
trial_unscanned[i] = np.any(scanning[s:e] < 0)
too_short = (ends - starts) < 2
keep_trial = ~(trial_lick_error | trial_unscanned | too_short)
```

iii. The notes say this reproduces exactly 81 lick-error trials; dropping them was preferred because decoder arrays cannot contain the NaNs used by the paper. No unscanned or short trials occurred.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `Fluorescence` (F), `Neuropil` (Fneu), `ImageSegmentation.iscell`, and `planeIdx` fields, with running `speed` used for interneuron filtering. The stored `Deconvolved` field is not used.

ii.
```python
fluo = ophys.data_interfaces["Fluorescence"].roi_response_series
neuro = ophys.data_interfaces["Neuropil"].roi_response_series
iscell = np.asarray(plane_seg["iscell"].data)[:, 0].astype(bool)
```

iii. The agent correctly observed that NWB `Deconvolved` contains Suite2p spikes from raw fluorescence, not the paper’s own dF/F-based signal.

## 2-b. How is the `neural` data processed?

i. ROIs are pooled across planes and restricted to `iscell`. Per trial, the agent subtracts `0.7*Fneu`, adds back trial-mean neuropil, applies Gaussian-15/maximin-300 baseline estimation, computes `(F-F0)/abs(F0)`, and Gaussian-smooths with sigma 2. It saves this dF/F directly; it does not perform the human solution’s OASIS deconvolution.

ii.
```python
f_ -= NEU_COEF * fneu_
tmp = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=1)
tmp = sp.ndimage.minimum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
flow[:, s:e] = sp.ndimage.maximum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
dff[:, in_trial] = (f_[:, in_trial] - flow[:, in_trial]) / np.abs(flow[:, in_trial])
```

iii. The agent explicitly chose dF/F because it considered it closer to raw data and found equal or better small-sample decoder accuracy. It acknowledged that the paper’s decoder uses deconvolved activity.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains manually curated `iscell` ROIs, then excludes cells whose whole-session in-trial dF/F has Pearson correlation greater than 0.5 with speed. It does not impose a running-speed sample threshold or place-cell selection.

ii.
```python
F = F[iscell]
...
speed_corr = (d0 @ s0) / denom
keep_cells = speed_corr <= SPEED_CORR_THRESH
dff = dff[keep_cells]
```

iii. Both cell filters follow the Methods. Low-speed samples are retained because speed below 2 cm/s is itself a required decoder output class; place-cell filtering was analysis-specific.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays are sliced beginning exactly at the `trial_start` sample, so column zero is the alignment event; they end before the teleport sample.

ii.
```python
for i in np.where(keep_trial)[0]:
    s, e = starts[i], ends[i]
    neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
```

iii. Behavior was already aligned to the imaging frame clock in the NWB, allowing shared indices with no interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native samples are retained without rebinning. The metadata uses the median behavioral timestamp difference, about 64.4836 ms (15.5078 Hz).

ii.
```python
info["dt"] = float(np.median(np.diff(ts)))
...
"time_bin_size": dt * 1000.0
```

iii. The notes report identical per-plane sampling across sessions and argue that native bins avoid resampling misalignment.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps attached to the raw `position` behavioral time series.

ii.
```python
timestamps = np.asarray(bts["position"].timestamps[:])
```

iii. The notes state all behavioral channels were already interpolated onto this imaging clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial-start index is subtracted from every timestamp in that trial.

ii.
```python
inp[0] = ts[s:e] - ts[s]
```

iii. This makes every trial start at exactly zero seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The identical `[s:e]` indices and frame-clock timestamps are used for both, producing the same number of columns. A possible single extra terminal fluorescence frame is removed before processing.

ii.
```python
if n_extra:
    F = F[:, :len(timestamps)]
...
inp[0] = ts[s:e] - ts[s]
```

iii. The agent traced the one-frame two-plane mismatch to reference alignment truncation and safely removes only the terminal extra frame.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
env_ts = beh["environment"]
ev = np.unique(env_ts[s:e])
```

iii. Inspection showed one nonnegative environment value per trial, with 0/1 corresponding to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded, the first remaining unique value is selected (or -1 if absent), and it is broadcast across the trial.

ii.
```python
ev = ev[ev >= 0]
trial_env[i] = int(ev[0]) if ev.size else -1
inp[1] = trial_env[i]
```

iii. The notes report that all actual trials had a single valid environment, so the fallback was defensive only.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based index of each detected `trial_start`, not the stored `trial number` values (although that series is loaded).

ii.
```python
for i in np.where(keep_trial)[0]:
    inp[2] = i
```

iii. The agent treats trial number as the original within-session lap index so dropped trials do not renumber later laps.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond broadcasting the original zero-based trial index over all time samples.

ii.
```python
inp[2] = i
```

iii. This is documented as a continuous per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives from `Reward` event timestamps plus the behavioral `reward_zone` time series. The reward timestamps are mapped to imaging frames with `searchsorted`.

ii.
```python
rew_frames = np.searchsorted(ts, S["reward_times"])
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(rzone_ts[s:e] > 0))
```

iii. This follows the reference `get_trial_types` definition rather than treating any reward timestamp alone as success.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trials after zero, the previous raw trial’s binary reward outcome is used and broadcast. Trial zero is assigned 1.

ii.
```python
prev_rewarded[0] = 1
prev_rewarded[1:] = trial_rewarded[:-1]
inp[3] = prev_rewarded[i]
```

iii. The agent inferred a rewarded predecessor from the approximately 30 warm-up trials immediately before imaging. This differs from the human solution’s explicit zero for trial zero.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavioral `position` and active reward-zone bounds. Zone identity is parsed from `nwb.identifier`’s scene string, switching after trial 30 where encoded; `reward_zone` is only a cross-check.

ii.
```python
scene = ident_parts[-1]
zone_labels = reward_zone_labels_from_scene(S["scene"], n_trials_raw)
zs, ze = REWARD_ZONES[zone_labels[i]]
```

iii. The agent preferred the reference scene-name method because `reward_zone` does not fire on omission trials and reports zero mismatches where an observation exists.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position before the zone is measured relative to its start, position after it relative to its end, and all positions inside receive distance zero; the result is then classified.

ii.
```python
d[before] = pos[before] - zone_start
d[after] = pos[after] - zone_end
out = np.full(pos.shape, 3, dtype=np.int64)
```

iii. This is signed distance to the nearest location in the active zone, exactly matching the task’s semantics.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement seven classes: `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
out[d < -50.0] = 0
out[(d >= -50.0) & (d < -10.0)] = 1
out[(d >= -10.0) & (d < 0.0)] = 2
out[(d > 0.0) & (d <= 10.0)] = 4
out[(d > 10.0) & (d <= 50.0)] = 5
out[d > 50.0] = 6
```

iii. Explicit masks avoid ambiguity around the required zero-only class.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and dF/F are sliced with the same `[s:e]` indices and written into arrays with the same T.

ii.
```python
p = pos[s:e]
out[0], _ = discretize_reward_distance(p, zs, ze)
```

iii. No resampling is needed because behavior already shares the imaging clock.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from behavioral `position`.

ii.
```python
pos = beh["position"]
p = pos[s:e]
```

iii. The raw variable is already in corridor centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial slice is digitized at four 90-cm edges and clipped to valid classes 0–4.

ii.
```python
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. Clipping safely absorbs small excursions outside the nominal 0–450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Edges `[90, 180, 270, 360]` produce the five requested equal-width categories.

ii.
```python
POS_BIN_EDGES = [90.0, 180.0, 270.0, 360.0]
```

iii. Five bins over a 450-cm track are 90 cm each.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same `[s:e]` sample window is used for position and neural matrices.

ii.
```python
neural_trials.append(dff[:, s:e])
p = pos[s:e]
```

iii. Shared imaging-clock indices guarantee column alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` count series.

ii.
```python
lick = beh["lick"]
```

iii. The notes identify this as cumulative lick count per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive count becomes 1 and all other samples become 0.

ii.
```python
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. This follows both the paper’s binarization and the requested binary output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced with the same trial boundaries as neural data.

ii.
```python
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. It is already sampled on the common imaging clock.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived primarily from the scene string in `nwb.identifier`; the behavioral `reward_zone` and position channels are used to validate it.

ii.
```python
zone_labels = reward_zone_labels_from_scene(S["scene"], n_trials_raw)
observed_zone[i] = int(np.argmin([abs(p - REWARD_ZONES[z][0]) for z in ZONE_NAMES]))
```

iii. This ports the paper’s `get_reward_zones` scene parsing and remains defined on omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. No-switch scenes repeat their terminal A/B/C label. Switch scenes repeat the pre-switch label for the first 30 trials and the final label thereafter; A/B/C are converted to 0/1/2 and broadcast.

ii.
```python
return [zone0] * min(change_trial, n_trials) + [zone1] * max(n_trials-change_trial, 0)
scene_zone = np.array([ZONE_NAMES.index(z) for z in zone_labels])
out[4] = scene_zone[i]
```

iii. The paper says switches occur after 30 trials, and the agent cross-checked every observable zone event.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It derives from `Reward` event timestamps and the trial’s behavioral `reward_zone` activity.

ii.
```python
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
```

iii. The conjunction matches the paper/reference trial-type definition.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are inserted onto the behavior frame axis via `searchsorted`; each trial is 1 only if it contains both a mapped reward and positive reward-zone samples. The scalar is broadcast over time.

ii.
```python
rew_frames = np.searchsorted(ts, S["reward_times"])
...
out[5] = trial_rewarded[i]
```

iii. This yielded the expected approximately 85% rewarded fraction.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent asserts structural consistency, removes at most one extra terminal fluorescence frame, uses `nansmooth` so NaNs do not propagate, supplies defensive missing-environment `-1`, filters bad trials, and drops sessions below two trials. Reward-zone observations may be absent but scene parsing still supplies labels.

ii.
```python
assert 0 <= n_extra <= 1
if n_extra:
    F = F[:, :len(timestamps)]
...
one[nan_inds] = 0.001
return a_nanless / one
```

iii. The notes trace the extra frame to two-plane alignment and report bookkeeping checks accounting for every raw trial.

## 13-a. What are the most time-consuming steps of the code?

i. Reading full F/Fneu arrays and the per-trial 2-D maximin filtering dominate conversion; pickling 9.63 GB is also material. Session work is parallelized.

ii.
```python
F[rows] = np.asarray(fluo[k].data[:]).T
tmp = sp.ndimage.minimum_filter1d(tmp, 300, axis=-1)
with ProcessPoolExecutor(max_workers=args.workers) as ex:
```

iii. The notes estimate loading at 0.4–1 s/session and processing at 1–3 s/session, identifying I/O and dF/F filtering as the main costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops in `compute_dff`, trial-summary construction, final trial assembly, plane loading, and diagnostic reward-distance plotting could potentially be reduced with masks or session-wide operations. Variable trial lengths make full vectorization awkward; cell-speed correlation is already vectorized.

ii.
```python
for s, e in zip(trial_starts, trial_ends):
    ...
for i, (s, e) in enumerate(zip(starts, ends)):
    ...
for i in np.where(keep_trial)[0]:
```

iii. The agent says trial loops are cheap because each iteration calls vectorized 2-D filters, and it explicitly optimized the expensive cell correlation into one matrix-vector product.

## 13-c. What processing does the code repeat multiple times?

i. Trial windows are traversed repeatedly to mask F/Fneu, add neuropil and estimate baselines, smooth dF/F, calculate trial summaries, assemble output arrays, and (optionally) draw diagnostics. Position digitization and reward-distance computation are repeated in diagnostic plots.

ii.
```python
for s, e in zip(trial_starts, trial_ends): ...  # occurs three times in compute_dff
for i, (s, e) in enumerate(zip(starts, ends)): ...
for i in np.where(keep_trial)[0]: ...
```

iii. The notes accept repeated trial passes as a memory-conscious and readable design; unlike the human reference’s survey/conversion design, NWBs are not loaded twice during a normal run.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused behavioral fields including `trial number`; computes `plane_idx` that is ultimately replaced by all-zero CA1 indices; computes `observed_zone`, mismatch statistics, timing, and detailed session diagnostics used only for checks/metadata. With `--show-processing`, it also recomputes discretizations solely for plots. The code never actually computes OASIS despite notes saying that deconvolution was implemented/tested.

ii.
```python
trial_num_ts = beh["trial number"]
plane_idx = S["plane_idx"][keep_cells]
...
region_idx.append(np.zeros(n[0].shape[0], dtype=np.int64))
```

iii. Most discarded work is deliberate validation or negligible bookkeeping. Loading every non-Reward behavior series via a dictionary comprehension is broader than required.
