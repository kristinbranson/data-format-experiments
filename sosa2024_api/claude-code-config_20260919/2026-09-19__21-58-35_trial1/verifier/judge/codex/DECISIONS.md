# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all sessions by globbing every `*.nwb` file under `/app/data/sub-*`, then processing each file as one session with `pynwb.NWBHDF5IO`. Within each file it reads the ophys and behavior processing groups, including fluorescence, neuropil, ROI metadata, behavioral time series, timestamps, and reward timestamps.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, "sub-*", "*.nwb")))

with NWBHDF5IO(path, "r", load_namespaces=True) as io:
    nwb = io.read()
    ophys = nwb.processing["ophys"]
    ...
    bts = (nwb.processing["behavior"]
           .data_interfaces["BehavioralTimeSeries"].time_series)
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset contains 152 NWB files in 11 `sub-*` directories and that the NWB already contains the VR-aligned behavior streams, so processing every NWB file with `pynwb` should capture the full dataset.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from each NWB file's `nwb.subject.subject_id`, and the final `subjects` list is built by deduplicating these IDs while assembling sessions.

ii.
```python
subject = nwb.subject.subject_id
...
subjects = []
...
if info["subject"] not in subjects:
    subjects.append(info["subject"])
subject_idx.append(subjects.index(info["subject"]))
```

iii. The AI's notes justify this by saying the NWB metadata and directory layout agree on the 11 mice, so using the NWB subject field is a direct way to map each session to a mouse.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity comes from `nwb.session_id`, and one call to `process_session` converts one NWB file into one output session entry.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, "sub-*", "*.nwb")))
...
session_id = nwb.session_id
...
for n, i_, o, pidx, info in results:
    neural.append(n)
    inputs.append(i_)
    outputs.append(o)
```

iii. The justification in the notes is that there are 152 NWB files corresponding to the 152 imaging sessions, matching the dataset structure described in the notes.

## 1-d. How are the data split into trials?

i. Trials are defined from the behavioral `trial_start` and `teleport` time series. The AI takes every frame where `trial_start > 0` as a trial start and every frame where `teleport > 0` as the corresponding trial end, then slices trial `i` as `[start_i, end_i)`.

ii.
```python
starts = np.where(beh["trial_start"] > 0)[0]
ends = np.where(beh["teleport"] > 0)[0]
assert len(starts) == len(ends), "trial_start / teleport count mismatch"
assert np.all(ends > starts), "teleport must follow its trial_start"
...
for i in np.where(keep_trial)[0]:
    s, e = starts[i], ends[i]
    T = e - s
```

iii. The AI's notes say the intended alignment event is trial start and that the end of a trial is the teleport event back into the intertrial interval. It also cites dataset-wide checks showing position runs from near 0 cm to near 450 cm within each extracted trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials unless they are flagged as lick-sensor failures, overlap `scanning < 0`, or are shorter than 2 imaging frames. It does not use the human reference's `<50`-frame minimum.

ii.
```python
trial_lick_error[i] = (np.mean(lick[s:e] > LICK_ERR_COUNT_THRESH)
                       > LICK_ERR_FRAC_THRESH)
trial_unscanned[i] = np.any(scanning[s:e] < 0)
...
trial_len = ends - starts
too_short = trial_len < 2
keep_trial = ~(trial_lick_error | trial_unscanned | too_short)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this by appealing to the paper's lick-sensor failure rule, the decoder format's prohibition on NaNs, and additional safeguards against missing imaging data or degenerate trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from the raw suite2p fluorescence traces `Fluorescence`, neuropil traces `Neuropil`, and the ROI metadata `iscell` and `planeIdx`. The AI pools planes using `planeIdx`, subsets to curated cells with `iscell`, and then computes neural features from `F` and `Fneu`.

ii.
```python
plane_seg = (ophys.data_interfaces["ImageSegmentation"]
             .plane_segmentations["PlaneSegmentation"])
iscell = np.asarray(plane_seg["iscell"].data)[:, 0].astype(bool)
plane_idx_all = np.asarray(plane_seg["planeIdx"].data).astype(int)
...
fluo = ophys.data_interfaces["Fluorescence"].roi_response_series
neuro = ophys.data_interfaces["Neuropil"].roi_response_series
...
F = F[iscell]
Fneu = Fneu[iscell]
```

iii. The notes explicitly justify not using the NWB `Deconvolved` field, stating that the paper analyzes signals recomputed from `F` and `Fneu`, not suite2p's stored deconvolution of raw fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI computes per-trial dF/F from `F` and `Fneu` by masking to in-trial frames, subtracting `0.7 * Fneu`, adding back the trial-mean neuropil, applying the maximin baseline (`sigma=15`, then 300-sample min/max filters), computing `(F - baseline) / abs(baseline)`, and smoothing with a 2-sample Gaussian. The final saved neural data are these dF/F traces; deconvolution is not applied in the saved output.

ii.
```python
def compute_dff(F, Fneu, trial_starts, trial_ends):
    f_ = np.full(F.shape, np.nan)
    fneu_ = np.full(F.shape, np.nan)
    for s, e in zip(trial_starts, trial_ends):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    ...
    f_ -= NEU_COEF * fneu_
    ...
    tmp = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=1)
    tmp = sp.ndimage.minimum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
    flow[:, s:e] = sp.ndimage.maximum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
    ...
    dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)

dff, in_trial = compute_dff(S["F"], S["Fneu"], starts, ends)
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
```

iii. The notes say this matches the paper's dF/F procedure and also state a deliberate decision to use dF/F rather than deconvolved events for the decoder, on the grounds that dF/F stays closer to the raw signal and performed as well or better in the AI's own spot checks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, only manually curated `iscell` ROIs are kept. Second, putative interneurons are removed if their dF/F is too correlated with running speed (`corr > 0.5`).

ii.
```python
iscell = np.asarray(plane_seg["iscell"].data)[:, 0].astype(bool)
...
F = F[iscell]
Fneu = Fneu[iscell]
...
d = dff[:, in_trial]
sp_ = speed[in_trial]
...
speed_corr = (d0 @ s0) / denom
speed_corr = np.nan_to_num(speed_corr, nan=0.0)
keep_cells = speed_corr <= SPEED_CORR_THRESH
dff = dff[keep_cells]
```

iii. The AI cites the paper's manual suite2p curation and the `corr(dF/F, speed) > 0.5` interneuron exclusion rule, and reports matching the paper's rough interneuron-exclusion rate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural traces are aligned to trial start by slicing each trial as `dff[:, start:end]`, where `start` is the `trial_start` frame. No further shifting or interpolation is applied.

ii.
```python
for i in np.where(keep_trial)[0]:
    s, e = starts[i], ends[i]
    neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
```

iii. The notes say the NWB behavior is already on the imaging frame clock, so trial alignment only requires consistent trial slicing at the `trial_start` event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native imaging frame resolution. The time bin is taken from the behavioral timestamps (`median(diff(ts))`), and no temporal rebinning or resampling is applied.

ii.
```python
dt = float(np.median([i["dt"] for i in infos]))
...
"metadata": {
    "time_bin_size": dt * 1000.0,
    "sampling_rate_hz": 1.0 / dt,
}
...
dt=float(np.median(np.diff(ts))),
```

iii. The AI's notes justify this by saying every session has the same ~15.5 Hz frame clock and the behavior streams have already been interpolated onto that clock upstream.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the shared behavioral/imaging timestamps, specifically from `BehavioralTimeSeries["position"].timestamps`.

ii.
```python
bts = (nwb.processing["behavior"]
       .data_interfaces["BehavioralTimeSeries"].time_series)
timestamps = np.asarray(bts["position"].timestamps[:])
...
ts = S["timestamps"]
```

iii. The notes say all behavioral channels share one imaging-aligned timestamp vector, so any of those timestamps could be used; the AI chose the `position` timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the first timestamp of that trial so the input begins at `0` seconds.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
inp[0] = ts[s:e] - ts[s]
```

iii. The justification is straightforward trial-start alignment: the decoder input should measure elapsed time from the start of the current trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is built from the same `[s:e]` slice used for the neural trial, so each time sample matches the corresponding neural frame.

ii.
```python
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
...
inp[0] = ts[s:e] - ts[s]
```

iii. The AI's notes say the behavior streams are already aligned to the 2P frame clock, so shared slicing is sufficient for alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
env_ts = beh["environment"]
```

iii. The notes identify `environment` as the NWB channel encoding `ENV1` vs `ENV2` on the imaging frame clock.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI takes the unique nonnegative environment value and uses that trial-level label for the whole trial, broadcasting it across timepoints in the input matrix.

ii.
```python
ev = np.unique(env_ts[s:e])
ev = ev[ev >= 0]
trial_env[i] = int(ev[0]) if ev.size else -1
...
inp[1] = trial_env[i]
```

iii. The notes justify this by saying each trial has exactly one valid environment value and that `-1` marks the intertrial interval, which is excluded from trials.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is derived from the trial segmentation itself, not from the raw `trial number` behavioral channel. It is the loop index `i` over the extracted trials defined by `trial_start` and `teleport`.

ii.
```python
starts = np.where(beh["trial_start"] > 0)[0]
ends = np.where(beh["teleport"] > 0)[0]
...
for i in np.where(keep_trial)[0]:
    ...
    inp[2] = i
```

iii. The notes say this avoids dependence on stray values in the stored `trial number` channel and keeps trial numbering tied to the extracted trial boundaries.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond assigning the per-session trial index and broadcasting it across the trial's timepoints.

ii.
```python
inp[2] = i
```

iii. The justification is simply that trial number is a per-trial contextual variable, so a constant value per trial is sufficient.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The AI first derives current-trial reward outcome from two raw sources: reward delivery timestamps from the `Reward` time series and reward-zone occupancy from the `reward_zone` channel. It then shifts that per-trial outcome back by one trial to create the previous-trial input.

ii.
```python
rew_frames = np.searchsorted(ts, S["reward_times"])
...
in_zone = rzone_ts[s:e] > 0
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
...
prev_rewarded[1:] = trial_rewarded[:-1]
```

iii. The AI's notes justify this by referring to the paper's `isreward` logic, which combines reward delivery with reward-zone occupancy rather than using the reward timestamps alone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI builds a per-trial `trial_rewarded` array, then sets `prev_rewarded[1:]` from the preceding trial. For the first trial, it assigns `1` rather than `0`, based on the assumption that warm-up trials immediately preceding imaging were rewarded.

ii.
```python
prev_rewarded = np.empty(n_trials_raw, dtype=np.int64)
prev_rewarded[0] = 1
prev_rewarded[1:] = trial_rewarded[:-1]
...
inp[3] = prev_rewarded[i]
```

iii. The notes justify the special case for trial 0 by citing the methods description of ~30 warm-up trials before imaging and arguing that "rewarded" is the best prior for the immediately preceding trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the animal's raw `position` trace and the trial's reward-zone identity. The reward-zone identity itself is parsed from the session `scene` string stored in `nwb.identifier`, not inferred directly from the `reward_zone` time series.

ii.
```python
ident_parts = nwb.identifier.rstrip("/").split("/")
scene = ident_parts[-1]
...
zone_labels = reward_zone_labels_from_scene(S["scene"], n_trials_raw)
...
zs, ze = REWARD_ZONES[zone_labels[i]]
p = pos[s:e]
out[0], _ = discretize_reward_distance(p, zs, ze)
```

iii. The AI's notes justify using the scene string because that is how the paper's `get_reward_zones` function works, and because the `reward_zone` channel does not fire on omission trials. The observed `reward_zone` entries are only used as a sanity check.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each position sample, the AI computes signed distance to the nearest edge of the active reward zone: negative before the zone, zero inside it, positive after it.

ii.
```python
def discretize_reward_distance(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    ...
    return out, d
```

iii. The notes cite the paper's reward-relative framing and describe this as a direct translation of position into distance from the current reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses seven hand-coded categories corresponding to the requested thresholds: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50` cm.

ii.
```python
out = np.full(pos.shape, 3, dtype=np.int64)
out[d < -50.0] = 0
out[(d >= -50.0) & (d < -10.0)] = 1
out[(d >= -10.0) & (d < 0.0)] = 2
out[(d > 0.0) & (d <= 10.0)] = 4
out[(d > 10.0) & (d <= 50.0)] = 5
out[d > 50.0] = 6
```

iii. The AI's notes say these thresholds follow the decoder specification exactly, with class `3` reserved for samples inside the reward zone where distance is exactly zero.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `[s:e]` behavioral slice that is paired with `dff[:, s:e]`, so it is frame-aligned to the neural data.

ii.
```python
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
...
p = pos[s:e]
out[0], _ = discretize_reward_distance(p, zs, ze)
```

iii. The AI's notes repeatedly state that all behavior channels share the imaging clock, so same-slice extraction is the intended alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii.
```python
pos = beh["position"]
...
p = pos[s:e]
```

iii. The notes describe `position` as the animal's VR corridor coordinate in centimeters, already aligned to the imaging frames.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the raw per-trial position trace and discretizes it into 5 bins. It also clips any out-of-range values into the edge bins.

ii.
```python
p = pos[s:e]
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. The notes justify clipping because a small number of samples fall slightly below 0 cm or above 450 cm, and the decoder specification still wants 5 bins spanning the track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The position trace is split into five 90 cm bins using edges at 90, 180, 270, and 360 cm.

ii.
```python
POS_BIN_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. The AI's notes say this is the direct implementation of the task instruction to divide the 450 cm track into five equal bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same `[s:e]` trial slice as the neural matrix, so every position sample corresponds to one neural frame.

ii.
```python
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
...
p = pos[s:e]
```

iii. The justification is the same shared-clock argument used for the other time-varying behavioral outputs.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = beh["lick"]
...
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. The notes describe `lick` as a count per imaging frame that must be binarized for the decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI thresholds the raw lick count at `> 0`, yielding a binary no-lick/lick vector.

ii.
```python
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. The notes justify this with the decoder requirement that lick be categorical and with the paper's own binarization of lick counts.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is extracted from the same `[s:e]` trial frames used for the neural data, so it is frame-aligned.

ii.
```python
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
...
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. The AI's notes say the lick series already lives on the imaging frame clock in the NWB file.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the session `scene` string inside `nwb.identifier`, which encodes the reward-zone identity and whether it switches after trial 30.

ii.
```python
ident_parts = nwb.identifier.rstrip("/").split("/")
scene = ident_parts[-1]
...
zone_labels = reward_zone_labels_from_scene(S["scene"], n_trials_raw)
scene_zone = np.array([ZONE_NAMES.index(z) for z in zone_labels])
...
out[4] = scene_zone[i]
```

iii. The notes justify this as a direct port of the paper's `get_reward_zones` logic and say the `reward_zone` channel is only used to cross-check the scene-derived labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string. Non-switch scenes get one zone label for all trials; switch scenes get an initial zone for the first 30 trials and a second zone for later trials.

ii.
```python
def reward_zone_labels_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    for z in ZONE_NAMES:
        if scene.endswith("_Location" + z):
            return [z] * n_trials
    ...
    return [zone0] * min(change_trial, n_trials) + \
           [zone1] * max(n_trials - change_trial, 0)
```

iii. The AI's notes justify the `change_trial=30` rule using both the paper text and checks against the observed reward-zone entries in the raw data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The saved per-trial reward outcome comes from reward delivery timestamps (`Reward`) combined with in-zone occupancy from the `reward_zone` behavioral channel.

ii.
```python
rew_frames = np.searchsorted(ts, S["reward_times"])
...
in_zone = rzone_ts[s:e] > 0
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
...
out[5] = trial_rewarded[i]
```

iii. The notes explicitly tie this to the paper's `get_trial_types` definition of rewarded trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped onto imaging-frame indices with `np.searchsorted`. Then each trial gets label `1` only if at least one reward timestamp falls inside the trial and the animal was in the reward zone on that trial. The label is broadcast across all timepoints of the trial output.

ii.
```python
rew_frames = np.searchsorted(ts, S["reward_times"])
...
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
...
out[5] = trial_rewarded[i]
```

iii. The AI's notes justify the extra `reward_zone` condition as matching the paper's reward/outcome logic, not just raw reward delivery events.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly rather than doing generic imputation: it trims a trailing neural frame if multi-plane fluorescence is one frame longer than behavior, drops lick-error trials, drops any trial overlapping `scanning < 0`, drops trials shorter than 2 frames, clips slight position overshoots into edge bins, and falls back on scene-derived reward-zone labels when the `reward_zone` channel is absent on omission trials.

ii.
```python
n_extra = F.shape[1] - len(timestamps)
assert 0 <= n_extra <= 1
if n_extra:
    F = F[:, :len(timestamps)]
    Fneu = Fneu[:, :len(timestamps)]
...
trial_lick_error[i] = (np.mean(lick[s:e] > LICK_ERR_COUNT_THRESH)
                       > LICK_ERR_FRAC_THRESH)
trial_unscanned[i] = np.any(scanning[s:e] < 0)
...
keep_trial = ~(trial_lick_error | trial_unscanned | too_short)
...
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. `CONVERSION_NOTES.md` contains a dedicated edge-case section explaining the fixed multi-plane length mismatch, omission-trial reward-zone issue, slight position overflow, and why invalid lick/scanning trials are removed rather than filled.

## 13-a. What are the most time-consuming steps of the code?

i. According to the AI's notes, the dominant costs are reading the NWB arrays (`F`, `Fneu`, behavior) and the per-session dF/F computation. The final pickle write is also called out as nontrivial.

ii.
```python
S = load_session(path)
...
dff, in_trial = compute_dff(S["F"], S["Fneu"], starts, ends)
...
with open(args.outfile, "wb") as f:
    pickle.dump(data, f, protocol=4)
```

iii. In the notes, the AI reports timing estimates showing NWB I/O at roughly `0.4-2.0 s/session`, dF/F plus curation up to `5.9 s/session`, and pickling taking about `14 s` on the full run.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code still uses Python loops over trials in `compute_dff`, over raw trials when deriving per-trial behavioral labels, and over kept trials when assembling per-trial matrices. The notes say these loops were kept because they are naturally trial-structured and relatively cheap, while speed-correlation across cells was vectorized.

ii.
```python
for s, e in zip(trial_starts, trial_ends):
    ...

for i, (s, e) in enumerate(zip(starts, ends)):
    ...

for i in np.where(keep_trial)[0]:
    ...
```

iii. The notes explicitly mention vectorizing the speed-correlation calculation as an optimization and describe the remaining trial loops as the main candidates that could be further vectorized if needed.

## 13-c. What processing does the code repeat multiple times?

i. The main conversion path avoids the reference solution's separate survey pass, but the AI does repeat some processing for diagnostics: when `--show-processing` is enabled it recomputes discretized position and reward-distance traces for plots, and it keeps `dff_all` solely so plotting can access the unfiltered dF/F.

ii.
```python
dff_all = dff
...
if show_processing:
    make_processing_plots(S, dff_all, keep_cells, speed_corr, starts, ends,
                          zone_labels, scene_zone, trial_rewarded, trial_env,
                          neural_trials, input_trials, output_trials,
                          kept_idx, plot_prefix)
```

iii. The notes emphasize that the code was structured to avoid a full exploratory survey pass; the repeated work that remains is mostly optional plotting and sanity-check computation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several diagnostic-only quantities that are not used by the downstream decoder data structure: `observed_zone` and `n_zone_mismatch` are only for sanity checks, `dff_all` is only for optional plotting, and `plane_idx` is returned from `process_session` but ignored when `brain_region_idx` is assembled as all zeros for CA1.

ii.
```python
observed_zone = np.full(n_trials_raw, -1, dtype=np.int64)
...
n_zone_mismatch = int(np.sum(observed_zone[seen] != scene_zone[seen]))
...
dff_all = dff
...
return neural_trials, input_trials, output_trials, plane_idx, info
...
for n, i_, o, pidx, info in results:
    ...
    region_idx.append(np.zeros(n[0].shape[0], dtype=np.int64))
```

iii. The notes frame these as deliberate validation aids: they were useful while checking correctness, but they do not affect the final decoder-facing arrays beyond metadata and optional plots.
