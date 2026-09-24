# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every NWB file matching `sub-*/sub-*_behavior+ophys.nwb` under `/app/data` and reads each one with `pynwb.NWBHDF5IO(..., load_namespaces=True)`. All 152 files (11 subjects) are found and converted. Every file is opened **twice**: a first pass that reads only `nwb.subject.subject_id` to build the ordered subject list, and a second pass that does the actual conversion. From each file it reads the `behavior/BehavioralTimeSeries` streams (`position`, `speed`, `lick`, `environment`, `reward_zone`, `trial number`, `trial_start`, `teleport`, `Reward`) and the `ophys` `Deconvolved` ROI response series plus the `ImageSegmentation` ROI table.

ii.
```python
DATA_ROOT = Path("/app/data")
...
def main():
    session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not session_paths:
        raise FileNotFoundError(f"No NWB files found under {DATA_ROOT}")

    subjects = []
    for path in session_paths:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            subject = io.read().subject.subject_id
        if subject not in subjects:
            subjects.append(subject)
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```
```python
def convert_session(path, subject_to_idx):
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        scene = nwb.identifier.split("/")[-1]
        behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
        ophys = nwb.processing["ophys"].data_interfaces
```

iii. From the trajectory (steps 19–35), the AI first enumerated the NWB contents of a sample file, confirmed that the directory layout is one level deep and uniform, and then ran a survey over all 152 files to confirm: matched `trial_start`/`teleport` counts in every session, a uniform ~64.48 ms sample period, and a consistent `behavior`/`ophys` layout. Step 34 notes "the NWB naming isn't perfectly uniform across sessions, so I'm making the loader robust to small schema differences", which is why the deconvolved interface is looked up by substring match rather than by a fixed key.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the NWB metadata field `nwb.subject.subject_id`, not from the directory name. They are accumulated in first-seen order over the sorted file list, giving `['m11','m12','m13','m14','m15','m17','m18','m19','m3','m4','m7']` (11 subjects). Each session stores the index of its subject in this list.

ii.
```python
    subjects = []
    for path in session_paths:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            subject = io.read().subject.subject_id
        if subject not in subjects:
            subjects.append(subject)
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
            "subject_idx": subject_to_idx[subject],
...
    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int16)
```

iii. No explicit written justification. The trajectory shows the AI reading `nwb.subject.subject_id` from the very first exploratory dump (step 19) and using it throughout its survey tables (steps 40, 57, 60), i.e. it treated the in-file subject metadata as the authoritative identifier. The verifier output confirms the resulting 11 subjects with 12–14 sessions each.

## 1-c. How are the data split into sessions?

i. One session per NWB file. No merging of sessions, and no cross-session neuron alignment. Sessions are ordered by the sorted file path (subject, then session number). A session is dropped only if it ends up with fewer than 2 kept trials (this never triggers — all 152 sessions are kept).

ii.
```python
    for path in session_paths:
        session = convert_session(path, subject_to_idx)
        if session["n_trials_kept"] < 2:
            continue

        data["neural"].append(session["neural"])
        data["input"].append(session["input"])
        data["output"].append(session["output"])
        data["subject_idx"].append(session["subject_idx"])
        data["brain_region_idx"].append(session["brain_region_idx"])
```

iii. The file naming (`sub-<id>_ses-<nn>_behavior+ophys.nwb`) and `nwb.session_id` make the one-file-per-session mapping obvious; the AI stored `session_id` and the scene string in `metadata['session_info']`. The `< 2 trials` guard was added to satisfy the stated format requirement that "there needs to be at least two trials within each session in order to evaluate the decoder performance."

## 1-d. How are the data split into trials?

i. A trial runs from a `trial_start` sample (inclusive) to the corresponding `teleport` sample (exclusive). The AI takes *all* indices where `trial_start > 0` and *all* indices where `teleport > 0` and zips them pairwise, raising an error if the two counts differ. This yields 12,216 raw trials.

ii.
```python
        trial_start = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
        teleport = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
        reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)

        if len(trial_start) != len(teleport):
            raise ValueError(f"{path.name}: trial starts and teleports do not match")

        ntrials = len(trial_start)
```
```python
        for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
            if drop_trial[trial_idx]:
                continue
            neural_trial = deconv[start:stop].T
            time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
            pos_trial_cm = position[start:stop]
            speed_trial_cm_s = speed[start:stop]
            lick_trial = lick[start:stop]
```

iii. Step 61 of the trajectory shows the AI printing `position` and `timestamps` around the first three `trial_start`/`teleport` indices in a session, after which it concluded (step 62): "Trial segmentation is clear now: the valid lap is `trial_start` inclusive to `teleport` exclusive." Step 42's survey confirmed that `trial_start` and `teleport` counts match in every one of the 152 sessions, which is why the simple `flatnonzero` pairing (rather than a rising-edge detector) was considered safe.

## 1-e. How are trials filtered based on quality controls?

i. One trial-level filter: a trial is dropped if more than 35% of its samples have a lick value greater than 2, which the paper's code treats as the lick detector being stuck. This drops 69 of 12,216 trials (0.56%), leaving 12,147. There is **no** minimum-trial-length filter and no maximum-duration filter (the AI considered excluding very long laps but did not implement it). Dropped trials still participate in the *previous*-trial-outcome computation, because the reward-outcome and zone arrays are computed over the full, undropped trial list first.

ii.
```python
LICK_ERROR_THRESHOLD = 0.35
...
        for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
            ...
            trial_lick = lick[start:stop]
            if np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD:
                drop_trial[trial_idx] = True
```
```python
            "trial_exclusion": {
                "lick_sensor_error_threshold": LICK_ERROR_THRESHOLD,
                "rule": "drop trials where more than 35% of frames have lick values greater than 2",
            },
```

iii. This rule is lifted from the paper's own `src/reward_relative/behavior.py::correct_lick_sensor_error`, which the AI read at step 46: "if >correction_thr (fraction) of samples have a cumulative lick count of >2 ... licks[t_start:t_end] = np.nan", with `correction_thr=0.35` as used by `lick_pos_std`. At step 56/59 the AI measured that the rule flags 69/12,216 trials and at step 65 concluded: "probably dropping the small number of lick-corrupted trials rather than fabricating lick labels". It also measured the trial-duration distribution (step 74/75: 287 trials > 30 s, 30 trials > 60 s, max 216 s) and the reasoning at step 73 says "if they're just a handful of disengagement trials, excluding them is likely better", but no duration filter was actually written into the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is taken straight from the NWB `Deconvolved` (Suite2p deconvolved fluorescence) ROI response series. The raw `Fluorescence` (F) and `Neuropil` (Fneu) series are never read. For multi-plane sessions each plane's series is read separately, masked by `iscell`, and the planes are concatenated in sorted plane order.

ii.
```python
def load_curated_deconvolved_activity(ophys_interfaces):
    deconv_name = next(name for name in ophys_interfaces.keys() if "Deconvolved" in name)
    deconv_iface = ophys_interfaces[deconv_name]
    plane_seg = ophys_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    keep_cells = get_iscell_mask(plane_seg)

    parts = []
    for plane_name in sorted(deconv_iface.roi_response_series.keys()):
        rrs = deconv_iface.roi_response_series[plane_name]
        roi_indices = np.asarray(rrs.rois.data[:], dtype=int)
        plane_data = np.asarray(rrs.data[:], dtype=np.float32)
        ...
        parts.append(plane_data[:, keep_cells[roi_indices]])

    return np.concatenate(parts, axis=1), keep_cells
```
```python
            "neural_signal": "Suite2p deconvolved events",
```

iii. At step 16 the AI noted that the paper's decoder notebook "uses `glmUtils.get_timeseries_data(...)` with deconvolved `events`", and at step 25 concluded: "The NWB files are already in the useful form: behavior streams are aligned frame-by-frame to imaging, and deconvolved fluorescence has the same number of samples. That means I can build the converter directly from NWB." In other words, it equated the NWB `Deconvolved` array with the paper's `sess.timeseries['events']`. It never verified this equivalence, and it never opened `preprocessing.py::dff`/`dff_dual` (its `sed -n '1,260p'` of `preprocessing.py` at step 18 stopped before those functions, and the methods.txt read at step 6 was truncated by the tool before the dF/F paragraph). Its step-54 grep did surface `pp.dff_dual(... neuropil_method ... baseline_method ... neu_coef ...)` in `utilities.py`, but it did not follow up.

## 2-b. How is the `neural` data processed?

i. Essentially no processing beyond curation and temporal aggregation. The stored deconvolved traces are read as float32, transposed to (n_neurons, n_frames), and then **summed** over non-overlapping 16-frame windows. There is no neuropil subtraction (`F - 0.7*Fneu`), no maximin baseline, no dF/F normalization, no 2-sample Gaussian smoothing, and no OASIS deconvolution with `tau = 0.7` — i.e. none of the paper's `preprocessing.dff` pipeline is reproduced.

ii.
```python
            neural_trial = deconv[start:stop].T
            ...
            binned = bin_trial(neural_trial=neural_trial, ...)
```
```python
    for bin_idx in range(nbins):
        start = bin_idx * BIN_FRAMES
        stop = min((bin_idx + 1) * BIN_FRAMES, nframes)
        sl = slice(start, stop)

        neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
```

iii. The justification is the same as 2-a: the AI believed the NWB `Deconvolved` field was already the paper's analysed signal, so no further processing was thought necessary. The summing (rather than averaging) over the 16-frame window is consistent with treating the deconvolved trace as a spike-rate-like quantity; the paper's own GLM code similarly multiplies deconvolved activity by 10 "to mimic spike number", which the AI saw in its step-54 grep. No explicit note on sum-vs-mean appears in the trajectory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: ROIs are restricted to Suite2p's manual curation flag `iscell[:, 0] == 1` (the first column of the `iscell` field of the `PlaneSegmentation` table), applied per plane via the ROI index mapping of each `roi_response_series`. This keeps ~46% of ROIs, 155–2341 cells per session, 138,678 cells total. **No putative-interneuron exclusion** (the paper's speed-correlation r > 0.5 rule) is applied, and there is no per-cell activity/SNR filter.

ii.
```python
def get_iscell_mask(plane_segmentation):
    iscell = np.asarray(plane_segmentation.to_dataframe()["iscell"].tolist())
    if iscell.ndim == 1:
        keep = iscell.astype(float) > 0
    else:
        keep = iscell[:, 0].astype(float) > 0
    return keep
```
```python
        parts.append(plane_data[:, keep_cells[roi_indices]])
```
```python
            "roi_filter": "iscell[:,0] == 1",
```

iii. Step 55: "The ROI table does include Suite2p `iscell`, and the deconvolved matrices still contain non-cell ROIs, so I will filter to curated cells (`iscell[:,0] == 1`) to match the repository's session objects." Step 57/60 checked the resulting per-session cell counts (mean keep fraction 0.46, min 155, max 2341) and found them plausible. The interneuron rule was visible to the AI — its step-54 grep returned `dayData.py` lines `exclude_int: True -- whether putative interneurons were excluded from analyses`, `int_thresh: 0.5`, `int_method: 'speed'` — but the trajectory contains no discussion of it and it was not implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start is achieved simply by slicing the session-wide activity matrix from the `trial_start` sample onward; no pre-event window is kept, so sample 0 of every trial is the trial-start frame. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (trials have variable length).

ii.
```python
            neural_trial = deconv[start:stop].T
            time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```
```python
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
```

iii. No separate justification is needed and none is given beyond the trial-segmentation reasoning at step 62. The AI did revisit the origin of the time axis at step 106 ("the first time bin currently starts around 0.5 s because I stored bin means. I'm switching `time_from_trial_start` to the left edge of each bin so the aligned trial truly starts at 0"), which is the only alignment-related correction it made.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — substantial rebinning. The native NWB sample period is 0.06448 s (~15.5 Hz). The AI aggregates a fixed **16 frames per bin**, giving a bin size of ~1031.7 ms (~0.97 Hz). Within each bin: neural activity is summed, position and speed are averaged, lick is OR-ed (`any(lick > 0)`), and time is the left edge. The final bin of a trial is a partial bin (`ceil`), so it can contain fewer than 16 frames. This shrinks trial lengths from a mean of ~217 samples to a mean of ~14 bins (min 6, max 210). All discretization (distance, position, speed) is applied *after* binning, i.e. to the bin-averaged behavioral values.

ii.
```python
# The NWB streams are aligned at ~15.5 Hz. Aggregating 16 imaging frames per bin
# keeps all modalities aligned while producing a dataset the shared decoder can
# train on end-to-end.
BIN_FRAMES = 16
```
```python
def bin_trial(neural_trial, time_trial_s, pos_trial_cm, speed_trial_cm_s, lick_trial, zone_label):
    nframes = neural_trial.shape[1]
    nbins = math.ceil(nframes / BIN_FRAMES)
    ...
    for bin_idx in range(nbins):
        start = bin_idx * BIN_FRAMES
        stop = min((bin_idx + 1) * BIN_FRAMES, nframes)
        sl = slice(start, stop)
        neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
        time_binned[bin_idx] = time_trial_s[start]
        pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
        speed_binned[bin_idx] = np.mean(speed_trial_cm_s[sl], dtype=np.float64)
        lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))
```
```python
            "time_bin_size": float(BIN_FRAMES * 1000.0 * 0.06448362720402656),
            "time_bin_size_frames": BIN_FRAMES,
```

iii. Step 69: "The remaining tradeoff is dataset size versus fidelity. At native imaging resolution the full dataset is too large for this shared decoder to train realistically, so I'm measuring trial durations and neuron counts to pick a temporal bin that still respects the frame-aligned source data." Steps 70–71 measured `frame_dt_s = 0.0644836`, mean 214.5 frames/trial, and the total bin counts for 0.25/0.5/1.0 s bins. Step 76 then committed to "fixed multi-frame temporal bins for a tractable decoder dataset". The premise (that native resolution is intractable) was never tested — the AI never attempted a native-resolution conversion or training run.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attribute of the `position` behavioral time series (all behavior streams share a common timestamp vector).

ii.
```python
        timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
```

iii. Step 29 dumped `data`, `timestamps`, and the median `dt` for every behavior stream, and the step-42 survey confirmed a common `dt_med` of 0.064484 s across all 152 sessions, so any stream's timestamps would have served. No further justification is recorded.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first timestamp is subtracted, then the per-bin value is the timestamp of the **first frame in that bin** (the bin's left edge), so the first value of every trial is exactly 0.0 s. Range across the dataset is 0.0–215.6 s.

ii.
```python
            time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```
```python
        time_binned[bin_idx] = time_trial_s[start]
```
```python
            input_trial = np.vstack([
                    binned["time"],
                    ...
```

iii. Step 106: "One small semantic fix is still worth making: the first time bin currently starts around 0.5 s because I stored bin means. I'm switching `time_from_trial_start` to the left edge of each bin so the aligned trial truly starts at 0." The verifier output before and after the patch confirms the change (`[0.5, 216.1]` → `[0.0, 215.6]`).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `start:stop` sample indices are used to slice the neural matrix and the timestamp vector, and the same 16-frame bin boundaries are used for both. The AI assumes, without asserting it, that NWB row *i* of the deconvolved matrix corresponds to behavior sample *i*. There is no check that the two streams have equal length and no cropping logic if they do not (in this dataset the deconvolved array is one frame longer than the behavior arrays in 10 of the 152 sessions, which is harmless because indices come from behavior).

ii.
```python
            neural_trial = deconv[start:stop].T
            time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```
```python
        if plane_data.shape[1] != roi_indices.shape[0]:
            raise ValueError(
                f"ROI response series {plane_name} has {plane_data.shape[1]} columns "
                f"but {roi_indices.shape[0]} ROI indices"
            )
```

iii. Step 24/25: having dumped the shapes of the behavior series and the ROI response series, the AI concluded "behavior streams are aligned frame-by-frame to imaging, and deconvolved fluorescence has the same number of samples". The only shape assertion it wrote is the ROI-count check above, which guards the neuron axis, not the time axis.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioral time series.

ii.
```python
        env = np.asarray(behavior["environment"].data[:], dtype=np.float32)
```

iii. Step 29 printed `np.unique(env)` and found only {-1 (sentinel), 0, 1}; the step-42 survey printed `env_vals` for every session and confirmed the same. This matches the paper's ENV1/ENV2 description.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative sentinel values are discarded, the rounded **median** of the remaining values over the trial is taken, and that single value is broadcast across all time bins of the trial. If a trial has no valid samples, the environment is taken from the scene string in `nwb.identifier` (1 if the scene's last environment is Env2, else 0).

ii.
```python
            env_trial = env[start:stop]
            env_trial = env_trial[env_trial >= 0]
            env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")
...
            input_trial = np.vstack([
                    binned["time"],
                    np.full(ntime, env_value, dtype=np.float32),
                    ...
```

iii. Step 58 explicitly checked that the per-trial rounded median of `environment` is constant within a trial and switches cleanly at trial 30 in a two-environment session (`[0]*30 + [1]*10`, "switch index 30"). Taking the median makes the value strictly per-trial as the instructions require and protects against the `-1` sentinel samples the AI observed at step 29.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The NWB `trial number` behavioral time series (rounded median over the trial), with the loop index as a fallback if the trial has no valid samples.

ii.
```python
        trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
...
            trialnum_trial = trial_number[start:stop]
            trialnum_trial = trialnum_trial[trialnum_trial >= 0]
            trialnum_value = float(np.round(np.nanmedian(trialnum_trial))) if len(trialnum_trial) else float(trial_idx)
```

iii. Step 29/30: "the NWB files already include per-frame `trial_start`, `teleport`, `trial number`, `environment`, ...". The AI used the stored stream as the natural source, filtering the `-1` sentinel it had observed. (I verified independently that for all 12,216 trials this rounded median is exactly equal to the 0-based index of the trial within the session, so the stored stream and the loop counter are interchangeable here.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Sentinel removal, rounded median, broadcast as a constant across the trial's time bins. No renumbering after the 69 lick-corrupted trials are dropped, so the trial numbers of a session with dropped trials have gaps (e.g. 0..79 with 79 missing). Range 0–99 across the dataset.

ii.
```python
            trialnum_value = float(np.round(np.nanmedian(trialnum_trial))) if len(trialnum_trial) else float(trial_idx)
...
                    np.full(ntime, trialnum_value, dtype=np.float32),
```

iii. The instructions call for "trial number (continuous, per trial)"; a constant per-trial broadcast satisfies this. No explicit trajectory note about the gaps left by dropped trials.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavioral time series' own `timestamps`, compared against the trial's start and end **timestamps** (not indices). Reward times are not snapped to the behavior sampling grid.

ii.
```python
        reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
        for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
            trial_reward_ts = reward_timestamps[
                (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
            ]
            reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
```

iii. Step 29 showed that `Reward` is a sparse event series with its own timestamps. Step 62–64: the AI counted, over the whole dataset, 10,342 rewarded trials, 52 "lapse" trials (reward zone entered but no reward) and 1,822 trials with no reward zone activation at all, and used this to reason (step 62) "I'm measuring how many unrewarded laps are true random omissions versus 'zone active but no reward' laps, because that affects both `reward_outcome` and the previous-trial input."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial reward outcome array is computed over **all** trials (before any are dropped), then shifted by one with 0 for the first trial. The value is broadcast as a constant across the trial's time bins. Because the shift is applied to the undropped array, dropping a lick-corrupted trial does not corrupt the previous-outcome of the trial that follows it.

ii.
```python
        prev_reward_outcomes = np.zeros(ntrials, dtype=np.int16)
        prev_reward_outcomes[1:] = reward_outcomes[:-1]
...
                    np.full(ntime, float(prev_reward_outcomes[trial_idx]), dtype=np.float32),
```

iii. Matches the instruction "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)". No further justification in the trajectory.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavioral time series, plus the trial's reward-zone label. The label comes from the `reward_zone` time series combined with `position`: the median position over the samples where `reward_zone > 0` is mapped to the nearest of three hard-coded zone centres (A 80–130, B 200–250, C 320–370 cm). If the zone never activates (omission trials, 1,822/12,216), the label falls back to a label parsed from the scene string in `nwb.identifier`, with the reward switch assumed to occur at trial 30.

ii.
```python
RZ_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
RZ_CENTERS = {k: 0.5 * (v[0] + v[1]) for k, v in RZ_BOUNDS.items()}
SWITCH_TRIAL = 30

def zone_from_position(position_cm):
    best_zone = min(RZ_CENTERS, key=lambda zone: abs(position_cm - RZ_CENTERS[zone]))
    return best_zone
```
```python
            rz_mask = reward_zone[start:stop] > 0
            if np.any(rz_mask):
                zone_labels.append(zone_from_position(float(np.nanmedian(position[start:stop][rz_mask]))))
            else:
                zone_labels.append(expected_zone_labels[trial_idx])
```
```python
def expected_trial_labels(scene, ntrials):
    envs, zones = parse_scene(scene)
    ...
    else:
        split = min(SWITCH_TRIAL, ntrials)
        env_by_trial = [envs[0]] * split + [envs[1]] * (ntrials - split)
        zone_by_trial = [zones[0]] * split + [zones[1]] * (ntrials - split)
    return env_by_trial, zone_by_trial
```

iii. Step 45 printed, for each non-zero `reward_zone` code, the min/max position and the environment, letting the AI recover the A/B/C position ranges. Step 66/67 enumerated all 26 distinct scene strings in the dataset (`Env1_LocationA`, `Env1_LocationA_to_B`, `Env1_A_to_Env2_B`, ...) so that `parse_scene` covers every case. Step 58 confirmed the within-session switch occurs at trial index 30. Step 76 summarizes: "scene-based reward-zone fallback plus observed zone detection when `reward_zone` is active".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest edge of the trial's reward zone, computed on the **bin-averaged** position: negative before the zone start, positive past the zone end, exactly 0 while inside. Then discretized into 7 classes.

ii.
```python
def reward_zone_distance(position_cm, zone_label):
    start_cm, end_cm = RZ_BOUNDS[zone_label]
    dist = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < start_cm
    after = position_cm > end_cm
    dist[before] = position_cm[before] - start_cm
    dist[after] = position_cm[after] - end_cm
    return dist
```
```python
    distance_cm = reward_zone_distance(pos_binned, zone_label)
    ...
        "distance_bin": discretize_distance(distance_cm),
```

iii. This is the standard "distance to any location in the reward zone" definition requested by the instructions (0 anywhere inside the zone). The zone bounds are the paper's, recovered at step 45 from the positions at which `reward_zone` is active.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes via explicit boolean masks, exactly the instruction's edges: `<-50`, `[-50,-10)`, `[-10,0)`, `==0`, `(0,10]`, `(10,50]`, `>50`. Note the AI puts exactly ±10 and ±50 into the *lower*-magnitude bin, i.e. the intervals are closed on the right for positive distances. Resulting class fractions: 0.244 / 0.098 / 0.075 / 0.225 / 0.020 / 0.071 / 0.268.

ii.
```python
def discretize_distance(distance_cm):
    out = np.full(distance_cm.shape, 6, dtype=np.int16)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```
```python
        "output_values": [
            ["< -50 cm", "-50 to -10 cm", "-10 to <0 cm", "0 cm", ">0 to 10 cm", ">10 to 50 cm", ">50 cm"],
            ...
```

iii. Directly transcribed from the "Decoder Outputs" spec in the instructions; class 3 is reserved for exactly-in-the-zone samples. No separate trajectory note.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same sample indices and same 16-frame bin boundaries as the neural data; both come out of `bin_trial` with identical `nbins`, so no further alignment is required.

ii.
```python
            binned = bin_trial(
                neural_trial=neural_trial,
                time_trial_s=time_trial_s,
                pos_trial_cm=pos_trial_cm,
                ...
            )
            ntime = binned["neural"].shape[1]
```
```python
    nbins = math.ceil(nframes / BIN_FRAMES)
```

iii. `nbins` is derived from the neural frame count, so the output time axis is defined by the neural data itself, guaranteeing equal length. No explicit justification.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioral time series (cm along the VR corridor).

ii.
```python
        position = np.asarray(behavior["position"].data[:], dtype=np.float32)
```

iii. Steps 29/45/61 inspected `position` values, its timestamps, and its values around trial boundaries; it is the obvious and only position stream.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Averaged over each 16-frame bin (`np.mean`), then discretized. No other transformation.

ii.
```python
        pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
    ...
        "position_bin": discretize_position(pos_binned),
```

iii. Consequence of the 16-frame binning decision (2-e). The comment at the top of the file, "Aggregating 16 imaging frames per bin keeps all modalities aligned", is the only recorded rationale.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins over the 450 cm track with open ends: `<90`, `[90,180)`, `[180,270)`, `[270,360)`, `>=360`. Class fractions 0.200 / 0.175 / 0.224 / 0.218 / 0.183.

ii.
```python
def discretize_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int16)
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out
```
```python
TRACK_LENGTH_CM = 450.0
```

iii. Directly from the instructions ("5 equal-sized bins spanning the 450 cm track"). The open first/last bins absorb the handful of samples that sit marginally outside [0, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical to 7-d: the same trial slice and the same 16-frame bin boundaries, with `nbins` set by the neural frame count.

ii.
```python
            pos_trial_cm = position[start:stop]
            ...
            binned = bin_trial(neural_trial=neural_trial, ..., pos_trial_cm=pos_trial_cm, ...)
```

iii. Same as 7-d — the neural array defines the time axis.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioral time series.

ii.
```python
        lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
```

iii. The AI observed at step 29 that `lick` is a per-frame count that can exceed 1, and at step 46 read the paper's `correct_lick_sensor_error`, which also treats `lick` as a cumulative count per sample.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized with an **OR over the whole 16-frame bin**: a bin is 1 if *any* of its up-to-16 frames has `lick > 0`. Trials whose lick sensor was stuck (>35% of frames with `lick > 2`) are dropped entirely rather than NaN-ed. The resulting marginal is 52.9% "yes".

ii.
```python
        lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))
```
```python
            if np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD:
                drop_trial[trial_idx] = True
```

iii. The `>0` binarization follows the instruction "Lick, time-varying. 0 = no, 1 = yes". The "any in bin" rule is the natural presence/absence aggregation for a binary event under the AI's 1.03 s binning; no trajectory note weighs it against, e.g., a majority or rate-based rule, and no note observes that it more than doubles the positive-class rate relative to the native-resolution signal.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slice, same bin boundaries, length fixed by the neural frame count.

ii.
```python
            lick_trial = lick[start:stop]
            ...
            binned = bin_trial(neural_trial=neural_trial, ..., lick_trial=lick_trial, ...)
```

iii. Same as 7-d/8-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. `reward_zone` and `position`, with a scene-string fallback — see 7-a. The label is mapped to 0/1/2 for A/B/C and broadcast across the trial's time bins.

ii.
```python
            reward_zone_value = {"A": 0, "B": 1, "C": 2}[zone_labels[trial_idx]]
...
            output_trial = np.vstack([
                    ...
                    np.full(ntime, reward_zone_value, dtype=np.int16),
                    ...
```
See 7-a for `zone_from_position` and `expected_trial_labels`.

iii. See 7-a. The resulting class balance is 0.332 / 0.336 / 0.332, essentially uniform, which is what the design of the experiment predicts.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Nearest-centre assignment from the median position during `reward_zone > 0`; scene-parsed fallback (with the switch at trial 30) for the 1,822 trials where the zone never activates; then a constant per-trial integer broadcast over time.

ii. See 7-a and the `reward_zone_value` snippet in 10-a.

iii. See 7-a. I independently checked this scheme: on all 10,394 trials where the reward zone does activate, the empirically detected label agrees with the scene-string prediction (0 disagreements), so the fallback used on the remaining 1,822 trials is well supported by the data even though the `SWITCH_TRIAL = 30` constant was hard-coded after inspecting a single session.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' timestamps, tested against the trial's start/end timestamps.

ii.
```python
        reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
            trial_reward_ts = reward_timestamps[
                (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
            ]
            reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
```

iii. See 6-a — step 63/64 quantified the rewarded/lapse/omission split (10,342 / 52 / 1,822) before the rule was fixed.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. 1 if at least one reward event falls in `[t_start, t_end)`, else 0; broadcast as a constant over the trial's time bins. Resulting marginal 0.842 rewarded. Note this is a timestamp-interval test rather than a snap-to-nearest-sample test, so no alignment tolerance assertion is needed or made.

ii.
```python
            reward_outcome_value = int(reward_outcomes[trial_idx])
...
                    np.full(ntime, reward_outcome_value, dtype=np.int16),
```

iii. Matches the instruction "Reward outcome, per-trial. 0 = no, 1 = yes". The AI's step-62 reasoning shows it deliberately treated both "true omission" (zone never armed) and "lapse" (zone armed, no reward) trials as outcome 0.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive behaviours, some hard failures and some silent:
- **Sentinel values** (`-1`) in `environment` and `trial number` are filtered out before taking the median, with a scene-derived / loop-index fallback if nothing valid remains.
- **Missing reward-zone activation** (1,822 trials) falls back to the scene-parsed zone label.
- **Trial-boundary mismatch**: if `len(trial_start) != len(teleport)`, the whole session raises a `ValueError` (never triggers on this dataset).
- **ROI/plane bookkeeping mismatch**: if a plane's deconvolved matrix has a different column count than its ROI index list, a `ValueError` is raised.
- **Unknown scene strings** raise a `ValueError` from `parse_scene`.
- **Lick-sensor failures** drop the trial (see 1-e).
- **Sessions with <2 kept trials** are dropped.
- **Not handled**: no check or cropping for a neural/behavior *length* mismatch. In this dataset the deconvolved array is exactly one frame longer than the behavior streams in 10 sessions (all m17/m18 two-plane sessions); because all indices are derived from the behavior arrays this is benign here, but the code would silently emit short trials if the neural array were ever the shorter one. There are also no NaN checks on `position`/`speed` (the code uses `nanmedian`/`mean`, so a NaN inside a bin would propagate into `pos_binned`).

ii.
```python
        if len(trial_start) != len(teleport):
            raise ValueError(f"{path.name}: trial starts and teleports do not match")
```
```python
            env_trial = env_trial[env_trial >= 0]
            env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")
```
```python
    raise ValueError(f"Unrecognized scene format: {scene}")
```
```python
        if session["n_trials_kept"] < 2:
            continue
```

iii. Most of these guards were written in response to problems the AI hit: step 84 ("I hit the first real schema difference: at least one multipane session has a deconvolved matrix whose column count doesn't match the first segmentation table I used") led to the ROI-index mapping and its assertion; step 66/67's enumeration of all 26 scene strings led to `parse_scene`'s strict regexes; step 42's survey showing matched `trial_start`/`teleport` counts everywhere justified making that a hard error rather than a repair.

## 13-a. What are the most time-consuming steps of the code?

i. In rough order:
1. **Reading the NWB files** — dominant. Every deconvolved matrix is materialized in full with `np.asarray(rrs.data[:])` (e.g. 32,673 × 2,100 float32 ≈ 275 MB for one plane), plus all nine behavior streams.
2. **Reading every file a second time** — `main()` opens all 152 files once just to collect `subject_id`, then `convert_session` reopens each one.
3. **`plane_segmentation.to_dataframe()`** — materializes the full ROI table (including pixel masks) just to read one column.
4. **The per-bin Python loop in `bin_trial`** — ~172,000 iterations total, each doing a NumPy slice-and-reduce over a (n_neurons × ≤16) block.
5. **Pickling the result** — the output is 621 MB.
The whole run took a few minutes (steps 90–95 show the conversion taking well over 90 s of polling).

ii.
```python
        plane_data = np.asarray(rrs.data[:], dtype=np.float32)
```
```python
    for path in session_paths:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            subject = io.read().subject.subject_id
```

iii. Step 82: "The conversion pass is still running; most of that time is spent streaming all 152 NWB files and building the trial lists." The AI correctly identified NWB I/O as the bottleneck but did not act on it.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Two clear candidates:
- **The bin loop in `bin_trial`**: because bins are a fixed 16 frames (with only the last one short), the whole trial could be reduced with `np.add.reduceat` (or pad-and-reshape) in one call per stream instead of `nbins` Python iterations. This is the only genuinely hot Python loop in the code.
- **The first per-trial loop in `convert_session`** (reward outcome, lick-error flag, zone label): the reward-outcome test could be done for all trials at once with `np.searchsorted` on `reward_timestamps` against the trial boundaries, and the lick-error fraction with `np.add.reduceat` over `(lick > 2)`.
- Minor: `zone_from_position` does a Python `min` over a dict per trial; it could be a single `np.argmin` over a (n_trials × 3) distance matrix.
The second per-trial loop is harder to vectorize because trials have different lengths.

ii.
```python
    for bin_idx in range(nbins):
        start = bin_idx * BIN_FRAMES
        stop = min((bin_idx + 1) * BIN_FRAMES, nframes)
        sl = slice(start, stop)
        neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
        ...
```
```python
        for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
            trial_reward_ts = reward_timestamps[
                (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
            ]
```

iii. No trajectory discussion of efficiency beyond noting the I/O cost; the loops are written for clarity.

## 13-c. What processing does the code repeat multiple times?

i.
- **Every NWB file is fully opened twice**: once in `main()` solely to read `nwb.subject.subject_id`, and once in `convert_session`. The subject ID is already in the file *path* (`sub-m11/...`), so this whole pass is avoidable.
- **The trial list is iterated twice** inside `convert_session` — first to compute reward outcomes, lick-error flags and zone labels for all trials, then again to build the arrays. (This one is necessary: the second loop needs `prev_reward_outcomes`, which needs the first loop to finish.)
- **`reward_timestamps` is re-scanned in full for every trial** (an O(n_trials × n_rewards) test instead of one `searchsorted`).
- **Zone labels and reward outcomes are computed for trials that are later dropped.**
- **`get_iscell_mask` calls `to_dataframe()`** on the full ROI table once per session, which is cheap relative to the traces but still materializes pixel masks that are never used.

ii.
```python
    subjects = []
    for path in session_paths:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            subject = io.read().subject.subject_id
```
```python
        for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
            ...            # pass 1: outcomes, lick flag, zone labels
        prev_reward_outcomes = np.zeros(ntrials, dtype=np.int16)
        prev_reward_outcomes[1:] = reward_outcomes[:-1]
        ...
        for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
            if drop_trial[trial_idx]:
                continue
            ...            # pass 2: build arrays
```

iii. No trajectory discussion. The double file-open appears to be an artifact of wanting a stable, globally-ordered `subjects` list before conversion begins.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts:
- `expected_trial_labels` computes and returns `env_by_trial`, which the caller discards (`_, expected_zone_labels = ...`) — the environment is always taken from the `environment` stream.
- `TRACK_LENGTH_CM` is defined and never used.
- `reward_outcomes`, `zone_labels` and the reward-timestamp scan are computed for the 69 trials that are subsequently dropped.
- `convert_session` returns bookkeeping fields (`raw_roi_count`, `kept_roi_count`, `n_trials_raw`, `n_trials_dropped_lick_sensor`, `source_file`, `scene`) that only feed `metadata['session_info']` and are unused by the decoder — harmless, and arguably useful provenance.
- `get_iscell_mask` builds a whole DataFrame (including ROI pixel masks) to read a single column.
- More substantively in the other direction: the 16-frame aggregation *discards* ~94% of the temporal samples that the source data provides and the decoder could have used.

ii.
```python
        _, expected_zone_labels = expected_trial_labels(scene, ntrials)
```
```python
TRACK_LENGTH_CM = 450.0
```
```python
def get_iscell_mask(plane_segmentation):
    iscell = np.asarray(plane_segmentation.to_dataframe()["iscell"].tolist())
```

iii. None of this is discussed in the trajectory. The per-trial constants that are broadcast to full time series (environment, trial number, previous outcome, reward zone, reward outcome) are *not* waste — the target format explicitly permits either form and recommends time-varying where possible.
